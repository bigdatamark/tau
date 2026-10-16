# [MODULE DOC]
# Runtime refresh and on-disk cache for the models.dev model catalog. Fetches the
# models.dev API plus NVIDIA's model list, converts them (via models_dev) into
# Tau's catalog document, validates it, and atomically persists it with an ETag
# and check time in Tau's models store file. Readers prefer this cache only when
# it is newer than the snapshot bundled with the package. Constants: store schema
# version 1, refresh interval of 4 hours, HTTP timeout of 15 seconds.
# Honours the TAU_OFFLINE environment variable (no network when set).
"""Persisted runtime refresh for the models.dev catalog snapshot."""

from __future__ import annotations

import json
import time
from contextlib import suppress
from dataclasses import dataclass
from os import environ
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import httpx

from tau_coding.models_dev import (
    MODELS_DEV_URL,
    NVIDIA_MODELS_URL,
    bundled_models_dev_catalog_document,
    models_dev_catalog_document,
    models_dev_catalog_overlay,
    nvidia_model_filter,
)
from tau_coding.paths import TauPaths

MODELS_STORE_SCHEMA_VERSION = 1
MODELS_REFRESH_INTERVAL_SECONDS = 4 * 60 * 60
MODELS_REFRESH_TIMEOUT_SECONDS = 15.0


class ModelsDevRefreshError(RuntimeError):
    # [CLASS DOC]
    # Exception raised when a catalog refresh fails due to HTTP errors or malformed
    # upstream data; wraps the original error as its cause.
    """Raised when a forced/runtime catalog refresh cannot complete."""


@dataclass(frozen=True, slots=True)
class ModelsDevRefreshResult:
    # [CLASS DOC]
    # Immutable outcome of a refresh attempt: whether new data was fetched and
    # written, whether the server answered 304 Not Modified, the number of models in
    # the resulting catalog, and the cache file path.
    refreshed: bool
    not_modified: bool
    model_count: int
    cache_path: Path


def models_store_path(paths: TauPaths | None = None) -> Path:
    # [FUNCTION DOC]
    # HELPER: Return the models store cache file path from the given path layout, or
    # from a default layout when none is given. No I/O.
    return (paths or TauPaths()).models_store_path


def cached_models_dev_catalog_document(
    paths: TauPaths | None = None,
) -> dict[str, Any] | None:
    # [FUNCTION DOC]
    # Return the cached catalog document only if the cache file is valid and newer
    # than the bundled snapshot. Inputs: optional path layout.
    # Behavior: reads/validates the cache (None if missing or invalid); loads the
    # bundled document's generated_at; if both generated_at values are integers and
    # the cached one is not newer, returns None. If either timestamp is missing the
    # cache is used.
    # Outputs: the catalog mapping or None. Side effects: reads files. Raises nothing
    # for cache problems (bundled document loading errors are handled by its loader).
    """Return a valid cache only when it is newer than the bundled snapshot."""
    cache = _read_cache(paths)
    if cache is None:
        return None
    bundled = bundled_models_dev_catalog_document() or {}
    bundled_at = bundled.get("generated_at")
    cached_at = cache["catalog"].get("generated_at")
    if isinstance(bundled_at, int) and isinstance(cached_at, int) and cached_at <= bundled_at:
        return None
    return cast(dict[str, Any], cache["catalog"])


def cached_models_dev_catalog_overlay(paths: TauPaths | None = None) -> dict[str, Any] | None:
    # [FUNCTION DOC]
    # Return the catalog overlay built from the cached document, or None when no
    # usable newer cache exists. Inputs: optional path layout. Side effects: file
    # reads. Raises ValueError only if conversion fails (unlikely, as the cache was
    # already validated).
    document = cached_models_dev_catalog_document(paths)
    return models_dev_catalog_overlay(document) if document is not None else None


async def refresh_models_dev_catalog(
    *,
    paths: TauPaths | None = None,
    force: bool = False,
    client: httpx.AsyncClient | None = None,
    now: float | None = None,
) -> ModelsDevRefreshResult:
    # [FUNCTION DOC]
    # Async refresh of the cached catalog. Inputs (keyword-only): optional path
    # layout, force flag (ignore the 4-hour freshness window), optional HTTP client
    # (otherwise one is created and closed here), optional current time override.
    # Behavior, in order:
    # 1) If TAU_OFFLINE is set: no network; returns not-refreshed with the model
    # count of the cache (or bundled snapshot, or 0).
    # 2) If not forced and the cache was checked within the interval: returns
    # not-refreshed with the cache's model count.
    # 3) GETs models.dev with JSON accept, a User-Agent, and If-None-Match from the
    # cached ETag. On 304 with a cache: updates checked_at, rewrites the cache and
    # returns not_modified=True.
    # 4) Otherwise requires success, parses JSON, GETs the NVIDIA model list,
    # builds the catalog document (with the NVIDIA provider filter and generated_at
    # in milliseconds), validates it by building an overlay, writes the cache
    # (schema version, checked_at, ETag, catalog) atomically and returns
    # refreshed=True with the model count.
    # Raises: ModelsDevRefreshError wrapping HTTP, type, value or JSON decode errors.
    # OSError from writing the cache is not wrapped and propagates as-is.
    # Side effects: network requests, cache file writes, closes an owned client.
    # KNOWN ISSUE: a 304 response when no cache exists falls through to
    # raise_for_status (which does not raise for 3xx) and JSON parsing of an empty
    # body, surfacing as a confusing refresh error.
    """Refresh models.dev plus Pi's NVIDIA filter and atomically cache the result."""
    resolved_paths = paths or TauPaths()
    path = models_store_path(resolved_paths)
    current_time = now if now is not None else time.time()
    cache = _read_cache(resolved_paths)
    if environ.get("TAU_OFFLINE") is not None:
        document = cache["catalog"] if cache is not None else bundled_models_dev_catalog_document()
        return ModelsDevRefreshResult(
            refreshed=False,
            not_modified=False,
            model_count=_model_count(document) if isinstance(document, dict) else 0,
            cache_path=path,
        )
    if (
        not force
        and cache is not None
        and current_time - cache["checked_at"] < MODELS_REFRESH_INTERVAL_SECONDS
    ):
        return ModelsDevRefreshResult(
            refreshed=False,
            not_modified=False,
            model_count=_model_count(cache["catalog"]),
            cache_path=path,
        )

    owned_client = client is None
    http = client or httpx.AsyncClient(timeout=MODELS_REFRESH_TIMEOUT_SECONDS)
    try:
        headers = {"Accept": "application/json", "User-Agent": "tau-model-catalog-refresh"}
        if cache is not None and cache.get("etag"):
            headers["If-None-Match"] = cache["etag"]
        response = await http.get(MODELS_DEV_URL, headers=headers)
        if response.status_code == 304 and cache is not None:
            cache["checked_at"] = current_time
            _write_cache(path, cache)
            return ModelsDevRefreshResult(
                refreshed=False,
                not_modified=True,
                model_count=_model_count(cache["catalog"]),
                cache_path=path,
            )
        response.raise_for_status()
        source = response.json()

        nvidia_response = await http.get(
            NVIDIA_MODELS_URL,
            headers={"Accept": "application/json", "User-Agent": "tau-model-catalog-refresh"},
        )
        nvidia_response.raise_for_status()
        nvidia_source = nvidia_response.json()

        # Imported here to avoid a catalog-loader import cycle during package startup.
        from tau_coding.catalog_loader import builtin_source_catalog

        document = models_dev_catalog_document(
            source,
            builtin_source_catalog(),
            provider_model_filters={"nvidia": nvidia_model_filter(source, nvidia_source)},
            generated_at=int(current_time * 1000),
        )
        # Validate before persistence so malformed upstream data cannot poison startup.
        models_dev_catalog_overlay(document)
        cache_document: dict[str, Any] = {
            "schema_version": MODELS_STORE_SCHEMA_VERSION,
            "checked_at": current_time,
            "etag": response.headers.get("etag"),
            "catalog": document,
        }
        _write_cache(path, cache_document)
        return ModelsDevRefreshResult(
            refreshed=True,
            not_modified=False,
            model_count=_model_count(document),
            cache_path=path,
        )
    except (httpx.HTTPError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ModelsDevRefreshError(f"Could not refresh model catalogs: {error}") from error
    finally:
        if owned_client:
            await http.aclose()


def _read_cache(paths: TauPaths | None) -> dict[str, Any] | None:
    # [FUNCTION DOC]
    # HELPER: Read and validate the cache file. Returns the parsed mapping only if it
    # is a mapping with the current schema version, a numeric checked_at, a mapping
    # catalog that passes overlay validation; otherwise (including missing file,
    # unreadable file, bad JSON or validation errors) returns None. Reads a file.
    path = models_store_path(paths)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != MODELS_STORE_SCHEMA_VERSION
        ):
            return None
        checked_at = value.get("checked_at")
        catalog = value.get("catalog")
        if not isinstance(checked_at, int | float) or not isinstance(catalog, dict):
            return None
        models_dev_catalog_overlay(catalog)
        return value
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _model_count(document: dict[str, Any]) -> int:
    # [FUNCTION DOC]
    # HELPER: Count models across all providers in a catalog document by summing the
    # lengths of each provider's "models" list; non-mapping providers or non-list
    # model entries are skipped; returns 0 if "providers" is not a mapping.
    providers = document.get("providers")
    if not isinstance(providers, dict):
        return 0
    return sum(
        len(provider.get("models", []))
        for provider in providers.values()
        if isinstance(provider, dict) and isinstance(provider.get("models"), list)
    )


def _write_cache(path: Path, value: dict[str, Any]) -> None:
    # [FUNCTION DOC]
    # HELPER: Atomically write a JSON mapping (indent 2, sorted keys, trailing
    # newline) to the path: ensures the parent directory exists, writes to a hidden
    # temporary file in the same directory, then renames it over the target.
    # On any exception, best-effort deletes the temp file and re-raises.
    # Side effects: filesystem writes. Raises OSError/TypeError on failure.
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            dir=path.parent,
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(value, temp_file, indent=2, sort_keys=True)
            temp_file.write("\n")
            temp_file.flush()
        temp_path.replace(path)
    except Exception:
        if temp_path is not None:
            with suppress(OSError):
                temp_path.unlink()
        raise
