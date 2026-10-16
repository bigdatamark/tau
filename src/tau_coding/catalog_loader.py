# [MODULE DOC]
# Loads Tau's provider catalog (providers, their models and per-model
# metadata) from TOML. Layers, in order: the packaged data/catalog.toml, the
# bundled generated models.dev overlay, the user's cached models.dev overlay,
# then the user overlay file catalog.toml in the Tau home. Raw TOML tables are
# merged, model tombstones (removed_models) are applied, then strictly
# validated with pydantic models into provider catalog entries. Also writes
# (upserts) provider definitions back into the user catalog atomically.
# Errors surface as CatalogError with source-prefixed dotted locations.
"""Load Tau's provider catalog from packaged and user TOML files."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Iterable, Mapping
from contextlib import suppress
from functools import cache
from importlib.resources import files
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    ValidationError,
)

from tau_agent.types import JSONValue
from tau_coding.paths import TauPaths
from tau_coding.provider_catalog import (
    AuthMethod,
    ModelCatalogMetadata,
    ModelCostTier,
    ModelInput,
    ProviderApi,
    ProviderCatalogEntry,
    ProviderKind,
)
from tau_coding.thinking import ThinkingLevel, ThinkingParameter

CATALOG_SCHEMA_VERSION = 1
USER_CATALOG_FILENAME = "catalog.toml"

# Thinking fields are merged as a group: an overlay that sets thinking_levels
# replaces all four, mirroring _merge_provider_config in provider_config.
_THINKING_FIELDS = ("thinking_levels", "thinking_models", "thinking_default", "thinking_parameter")

_NonEmptyString = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1),
]
_NonEmptyStringTuple = Annotated[tuple[_NonEmptyString, ...], Field(min_length=1)]
_PositiveInt = Annotated[StrictInt, Field(gt=0)]
_NonNegativeFloat = Annotated[float, Field(ge=0)]


class CatalogError(ValueError):
    # [CLASS DOC]
    # Exception (ValueError subclass) raised when a catalog file or merged
    # catalog is invalid TOML, has wrong structure or fails validation.
    """Raised when a Tau catalog file is invalid."""


class _CatalogCostTier(BaseModel):
    # [CLASS DOC]
    # HELPER: strict schema for one pricing tier: optional positive
    # max_input_tokens (the tier's upper bound), and non-negative input, output,
    # cacheRead, cacheWrite rates, plus optional cacheWrite1h. Frozen; unknown
    # keys forbidden.
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_input_tokens: _PositiveInt | None = None
    input: _NonNegativeFloat
    output: _NonNegativeFloat
    cacheRead: _NonNegativeFloat
    cacheWrite: _NonNegativeFloat
    # Optional 1-hour TTL cache-write rate (Anthropic bills those above the
    # 5-minute cacheWrite rate). Omitted from the cost dict when unset so
    # consumers can fall back to cacheWrite.
    cacheWrite1h: _NonNegativeFloat | None = None


class _CatalogModelMetadata(BaseModel):
    # [CLASS DOC]
    # HELPER: strict schema for per-model metadata: optional display name, API,
    # base URL, reasoning flag, input modalities, flat cost mapping, cost tiers,
    # positive context window and max tokens, headers, compat options, a map from
    # thinking level to provider value, and a list of unsupported thinking levels.
    # Frozen; unknown keys forbidden.
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: _NonEmptyString | None = None
    api: ProviderApi | None = None
    base_url: _NonEmptyString | None = None
    reasoning: StrictBool | None = None
    input: tuple[ModelInput, ...] = ()
    cost: dict[_NonEmptyString, _NonNegativeFloat] | None = None
    cost_tiers: tuple[_CatalogCostTier, ...] = ()
    context_window: _PositiveInt | None = None
    max_tokens: _PositiveInt | None = None
    headers: dict[_NonEmptyString, _NonEmptyString] = {}
    compat: dict[_NonEmptyString, Any] = {}
    thinking_level_map: dict[ThinkingLevel, _NonEmptyString] = {}
    unsupported_thinking_levels: tuple[ThinkingLevel, ...] = ()


class _CatalogProvider(BaseModel):
    # [CLASS DOC]
    # HELPER: strict schema for one provider table: required name, display name,
    # kind, base URL, API-key environment variable, non-empty models list,
    # default model and docs URL; optional credential name, API, context windows,
    # headers, compat, model metadata, thinking settings (levels, models, default,
    # parameter), removed models and auth methods (default API key only).
    # Strings are stripped and must be non-empty. Frozen; unknown keys forbidden.
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: _NonEmptyString
    display_name: _NonEmptyString
    kind: ProviderKind
    base_url: _NonEmptyString
    api_key_env: _NonEmptyString
    credential_name: _NonEmptyString | None = None
    models: _NonEmptyStringTuple
    default_model: _NonEmptyString
    docs_url: _NonEmptyString
    api: ProviderApi | None = None
    context_windows: dict[_NonEmptyString, _PositiveInt] | None = None
    headers: dict[_NonEmptyString, _NonEmptyString] = {}
    compat: dict[_NonEmptyString, Any] = {}
    model_metadata: dict[_NonEmptyString, _CatalogModelMetadata] = {}
    thinking_levels: tuple[ThinkingLevel, ...] | None = None
    thinking_models: tuple[_NonEmptyString, ...] = ()
    thinking_default: ThinkingLevel | None = None
    thinking_parameter: ThinkingParameter | None = None
    removed_models: tuple[_NonEmptyString, ...] = ()
    auth_methods: tuple[AuthMethod, ...] = ("api_key",)


class _CatalogFile(BaseModel):
    # [CLASS DOC]
    # HELPER: strict root schema: schema_version must be 1 plus a list of
    # providers.
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    providers: tuple[_CatalogProvider, ...] = ()


def builtin_catalog_resource_text() -> str:
    # [FUNCTION DOC]
    # Reads and returns the packaged data/catalog.toml text from the tau_coding
    # package resources. Raises on missing resource. Side effect: resource read.
    """Return the packaged builtin catalog TOML text."""
    return files("tau_coding").joinpath("data/catalog.toml").read_text(encoding="utf-8")


@cache
def builtin_catalog() -> tuple[ProviderCatalogEntry, ...]:
    # [FUNCTION DOC]
    # Cached: returns the built-in catalog entries with the bundled generated
    # models.dev data merged in and tombstones applied (defaults restored from
    # the source catalog). Raises CatalogError if the packaged catalog is invalid.
    """Return Tau's built-in catalog with generated models.dev model data."""
    raw = _builtin_raw_with_generated_models()
    filtered = _apply_model_tombstones(raw, base=_builtin_raw())
    return _entries_from_raw(filtered, source="built-in catalog.toml")


@cache
def builtin_source_catalog() -> tuple[ProviderCatalogEntry, ...]:
    # [FUNCTION DOC]
    # Cached: returns entries from the packaged catalog alone (no generated data),
    # with its own tombstones applied; used as generator input/fallback.
    """Return the application-owned catalog used as generator input/fallback."""
    raw = _builtin_raw()
    filtered = _apply_model_tombstones(raw, base=raw)
    return _entries_from_raw(filtered, source="built-in catalog.toml")


def user_catalog_path(paths: TauPaths | None = None) -> Path:
    # [FUNCTION DOC]
    # Returns the path of the user overlay catalog file inside the Tau home
    # (default paths when none given). No I/O.
    """Return the user-level catalog overlay path."""
    return (paths or TauPaths()).home / USER_CATALOG_FILENAME


def effective_catalog(paths: TauPaths | None = None) -> tuple[ProviderCatalogEntry, ...]:
    # [FUNCTION DOC]
    # Returns the catalog actually used at runtime: built-in plus generated plus
    # cached models.dev data, then the user overlay merged over it when the file
    # exists (root keys and schema version validated), with tombstones applied.
    # Raises CatalogError on invalid user TOML or invalid merged data.
    # Side effects: reads the user catalog and models.dev cache files.
    """Return bundled, refreshed, then user-overlaid provider model data."""
    builtin_raw = _builtin_raw_with_cached_models(paths)
    path = user_catalog_path(paths)
    if not path.exists():
        filtered = _apply_model_tombstones(builtin_raw, base=_builtin_raw())
        return _entries_from_raw(filtered, source="effective built-in catalog")
    overlay_raw = _parse_catalog_text(path.read_text(encoding="utf-8"), source=str(path))
    _validate_catalog_root(overlay_raw, source=str(path))
    merged = _merge_raw_catalogs(builtin_raw, overlay_raw)
    filtered = _apply_model_tombstones(merged, base=_builtin_raw())
    return _entries_from_raw(filtered, source=str(path))


def save_user_catalog_entries(
    entries: Iterable[ProviderCatalogEntry],
    paths: TauPaths | None = None,
) -> Path:
    # [FUNCTION DOC]
    # Upserts full provider definitions into the user catalog: loads and validates
    # the existing file (or starts a fresh one), replaces providers with the same
    # name in place, appends new ones, then rewrites the file as TOML atomically.
    # Output: the file path. Raises CatalogError on an invalid existing file and
    # OS errors on write failure. Side effects: creates directories, writes file.
    """Upsert full provider definitions into the user-level catalog file."""
    path = user_catalog_path(paths)
    if path.exists():
        raw = _parse_catalog_text(path.read_text(encoding="utf-8"), source=str(path))
        _validate_catalog_root(raw, source=str(path))
    else:
        raw = {"schema_version": CATALOG_SCHEMA_VERSION, "providers": []}

    providers = list(_raw_providers(raw))
    provider_indexes = {
        _raw_provider_name(provider): index for index, provider in enumerate(providers)
    }
    for entry in entries:
        raw_provider = _raw_provider_from_entry(entry)
        if entry.name in provider_indexes:
            providers[provider_indexes[entry.name]] = raw_provider
        else:
            provider_indexes[entry.name] = len(providers)
            providers.append(raw_provider)

    updated = {
        "schema_version": raw.get("schema_version", CATALOG_SCHEMA_VERSION),
        "providers": providers,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, _catalog_to_toml(updated))
    return path


@cache
def _builtin_raw() -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: cached parse of the packaged catalog TOML into raw tables.
    return _parse_catalog_text(builtin_catalog_resource_text(), source="built-in catalog.toml")


def _builtin_raw_with_cached_models(paths: TauPaths | None) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: merges the user's cached models.dev overlay (if any) over the
    # built-in-plus-generated raw catalog; keeps the merge only if it validates,
    # otherwise silently falls back to the unmerged data. Side effect: cache read.
    from tau_coding.models_dev_store import cached_models_dev_catalog_overlay

    raw = _builtin_raw_with_generated_models()
    overlay = cached_models_dev_catalog_overlay(paths)
    if overlay is None:
        return raw
    merged = _merge_generated_catalog(raw, overlay)
    try:
        _entries_from_raw(
            _apply_model_tombstones(merged, base=_builtin_raw()),
            source="cached models.dev catalog",
        )
    except CatalogError:
        return raw
    return merged


@cache
def _builtin_raw_with_generated_models() -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: cached; merges the bundled generated models.dev overlay over the
    # packaged raw catalog, falling back silently to the packaged data if the
    # overlay is missing or the merge fails validation.
    # Imported lazily because models_dev uses the catalog dataclasses imported by
    # this module. Missing or invalid generated data is deliberately non-fatal.
    from tau_coding.models_dev import bundled_models_dev_catalog_overlay

    raw = _builtin_raw()
    overlay = bundled_models_dev_catalog_overlay()
    if overlay is None:
        return raw
    merged = _merge_generated_catalog(raw, overlay)
    try:
        _entries_from_raw(
            _apply_model_tombstones(merged, base=raw),
            source="generated models.dev catalog",
        )
    except CatalogError:
        return raw
    return merged


def _parse_catalog_text(text: str, *, source: str) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: parses TOML text; raises CatalogError naming the source on decode
    # errors. Output: raw mapping.
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise CatalogError(f"{source}: invalid TOML: {error}") from error


def _validate_catalog_root(raw: dict[str, Any], *, source: str) -> None:
    # [FUNCTION DOC]
    # HELPER: requires only schema_version and providers keys, schema_version
    # present and equal to 1, and providers an array of tables; raises CatalogError
    # otherwise.
    allowed = {"schema_version", "providers"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise CatalogError(f"{source}: unknown catalog keys: {', '.join(unknown)}")
    if "schema_version" not in raw:
        raise CatalogError(f"{source}: schema_version is required")
    if raw["schema_version"] != CATALOG_SCHEMA_VERSION:
        raise CatalogError(f"{source}: unsupported schema_version: {raw['schema_version']!r}")
    _raw_providers(raw)


def _merge_raw_catalogs(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: merges overlay providers over base providers by name, preserving
    # base order and appending new providers; same-named providers are merged with
    # the provider merge rules. Schema version taken from overlay when present.
    """Merge overlay provider tables over base ones; overlay values win."""
    base_providers = _raw_providers(base)
    overlay_providers = _raw_providers(overlay)
    by_name: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for provider in base_providers:
        name = _raw_provider_name(provider)
        by_name[name] = provider
        order.append(name)
    for provider in overlay_providers:
        name = _raw_provider_name(provider)
        if name in by_name:
            by_name[name] = _merge_raw_provider(by_name[name], provider)
        else:
            by_name[name] = provider
            order.append(name)
    return {
        "schema_version": overlay.get("schema_version", base.get("schema_version")),
        "providers": [by_name[name] for name in order],
    }


def _merge_generated_catalog(base: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: applies generated model inventories to base providers only (new
    # providers in the generated data are ignored). For each matching provider
    # with a non-empty models list, merges tables, then replaces models with the
    # generated list exactly, drops context windows, metadata and thinking models
    # for models not in it, and resets the default model to the first generated
    # model if the old default is gone. Providers without generated models keep
    # base data.
    """Replace model inventories while retaining application-owned provider config."""
    generated_by_name = {
        _raw_provider_name(provider): provider for provider in _raw_providers(generated)
    }
    providers: list[dict[str, Any]] = []
    for provider in _raw_providers(base):
        name = _raw_provider_name(provider)
        overlay = generated_by_name.get(name)
        if overlay is None:
            providers.append(provider)
            continue
        merged = _merge_raw_provider(provider, overlay)
        models = overlay.get("models")
        if not isinstance(models, list) or not models:
            providers.append(provider)
            continue
        merged["models"] = list(models)
        allowed = set(models)
        for field in ("context_windows", "model_metadata"):
            values = merged.get(field)
            if isinstance(values, dict):
                merged[field] = {
                    model: value for model, value in values.items() if model in allowed
                }
        thinking_models = merged.get("thinking_models")
        if isinstance(thinking_models, list):
            merged["thinking_models"] = [model for model in thinking_models if model in allowed]
        if merged.get("default_model") not in allowed:
            merged["default_model"] = models[0]
        providers.append(merged)
    return {"schema_version": base.get("schema_version"), "providers": providers}


def _merge_raw_provider(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: shallow-merges one provider table, overlay winning, with special
    # rules: models become overlay models followed by base models (deduplicated);
    # removed_models are unioned; context windows, headers and compat mappings are
    # key-merged; model metadata is deep-merged per model; if the overlay sets
    # thinking_levels, the four thinking fields are taken as a group from the
    # overlay (absent ones removed). Output: new mapping (inputs not mutated).
    merged = {**base, **overlay}
    base_models = base.get("models", [])
    overlay_models = overlay.get("models", [])
    if isinstance(base_models, list) and isinstance(overlay_models, list):
        merged["models"] = list(dict.fromkeys([*overlay_models, *base_models]))
    base_removed = base.get("removed_models", [])
    overlay_removed = overlay.get("removed_models", [])
    if isinstance(base_removed, list) and isinstance(overlay_removed, list):
        merged["removed_models"] = list(dict.fromkeys([*base_removed, *overlay_removed]))
    for key in ("context_windows", "headers", "compat"):
        base_mapping = base.get(key)
        overlay_mapping = overlay.get(key)
        if isinstance(base_mapping, dict) and isinstance(overlay_mapping, dict):
            merged[key] = {**base_mapping, **overlay_mapping}
    base_metadata = base.get("model_metadata")
    overlay_metadata = overlay.get("model_metadata")
    if isinstance(base_metadata, dict) and isinstance(overlay_metadata, dict):
        merged["model_metadata"] = _merge_model_metadata(base_metadata, overlay_metadata)
    if "thinking_levels" in overlay:
        for field in _THINKING_FIELDS:
            if field in overlay:
                merged[field] = overlay[field]
            else:
                merged.pop(field, None)
    return merged


def _apply_model_tombstones(
    raw: dict[str, Any],
    *,
    base: dict[str, Any],
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: for providers with removed_models, filters those models out of
    # models, thinking models, context windows and metadata. If the default model
    # was removed, uses the base catalog's default when it is not removed, else
    # the first remaining model, else an empty string (which later fails
    # validation). Output: new raw catalog.
    # KNOWN ISSUE: the base default is reused even if it is no longer among the
    # remaining models, which then fails validation as "not in models".
    """Remove provider-scoped models withdrawn by bundled or user catalogs."""
    base_by_name = {_raw_provider_name(provider): provider for provider in _raw_providers(base)}
    providers: list[dict[str, Any]] = []
    for provider in _raw_providers(raw):
        removed = {model for model in provider.get("removed_models", []) if isinstance(model, str)}
        if not removed:
            providers.append(provider)
            continue
        filtered = {**provider}
        for field in ("models", "thinking_models"):
            values = filtered.get(field)
            if isinstance(values, list):
                filtered[field] = [model for model in values if model not in removed]
        for field in ("context_windows", "model_metadata"):
            values = filtered.get(field)
            if isinstance(values, dict):
                filtered[field] = {
                    model: value for model, value in values.items() if model not in removed
                }
        if filtered.get("default_model") in removed:
            name = _raw_provider_name(provider)
            base_default = base_by_name.get(name, {}).get("default_model")
            remaining = filtered.get("models", [])
            filtered["default_model"] = (
                base_default
                if isinstance(base_default, str) and base_default not in removed
                else remaining[0]
                if isinstance(remaining, list) and remaining
                else ""
            )
        providers.append(filtered)
    return {**raw, "providers": providers}


def _merge_model_metadata(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: per-model merge of metadata: when both sides are mappings, keys are
    # merged with overlay winning and cost, headers, compat and thinking level map
    # sub-mappings key-merged; otherwise the overlay value replaces the base.
    merged: dict[str, Any] = {**base}
    for model, overlay_metadata in overlay.items():
        base_metadata = merged.get(model)
        if isinstance(base_metadata, dict) and isinstance(overlay_metadata, dict):
            next_metadata = {**base_metadata, **overlay_metadata}
            for key in ("cost", "headers", "compat", "thinking_level_map"):
                base_mapping = base_metadata.get(key)
                overlay_mapping = overlay_metadata.get(key)
                if isinstance(base_mapping, dict) and isinstance(overlay_mapping, dict):
                    next_metadata[key] = {**base_mapping, **overlay_mapping}
            merged[model] = next_metadata
        else:
            merged[model] = overlay_metadata
    return merged


def _raw_providers(raw: dict[str, Any]) -> list[dict[str, Any]]:
    # [FUNCTION DOC]
    # HELPER: returns the providers list (default empty) or raises CatalogError if
    # it is not a list of tables.
    providers = raw.get("providers", [])
    if not isinstance(providers, list) or not all(isinstance(item, dict) for item in providers):
        raise CatalogError("catalog providers must be an array of tables ([[providers]])")
    return providers


def _raw_provider_name(provider: dict[str, Any]) -> str:
    # [FUNCTION DOC]
    # HELPER: returns a provider's stripped non-empty name or raises CatalogError.
    name = provider.get("name")
    if not isinstance(name, str) or not name.strip():
        raise CatalogError("catalog provider entries must have a non-empty string name")
    return name.strip()


def _entries_from_raw(raw: dict[str, Any], *, source: str) -> tuple[ProviderCatalogEntry, ...]:
    # [FUNCTION DOC]
    # HELPER: validates a raw catalog with the root schema (formatting pydantic
    # errors with provider names in locations), converts each provider to an entry,
    # and rejects duplicate provider names. Raises CatalogError naming the source.
    try:
        catalog = _CatalogFile.model_validate(raw)
    except ValidationError as error:
        raise CatalogError(f"{source}: {_format_validation_error(raw, error)}") from error
    entries = tuple(_entry_from_provider(provider, source=source) for provider in catalog.providers)
    names = [entry.name for entry in entries]
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise CatalogError(f"{source}: duplicate provider names: {', '.join(duplicates)}")
    return entries


def _entry_from_provider(provider: _CatalogProvider, *, source: str) -> ProviderCatalogEntry:
    # [FUNCTION DOC]
    # HELPER: cross-field validation and conversion of one provider: default
    # model, thinking models, context-window and metadata keys must all be in
    # models; thinking default must be among thinking levels; cost tiers must be
    # valid. Context windows from metadata fill in models lacking an explicit
    # entry. Compat values are checked to be JSON. Output: a catalog entry.
    # Raises CatalogError with a dotted location prefix.
    prefix = f"{source}: providers.{provider.name}"
    if provider.default_model not in provider.models:
        raise CatalogError(f"{prefix}.default_model: {provider.default_model!r} is not in models")
    for model in provider.thinking_models:
        if model not in provider.models:
            raise CatalogError(f"{prefix}.thinking_models: {model!r} is not in models")
    for model in provider.context_windows or {}:
        if model not in provider.models:
            raise CatalogError(f"{prefix}.context_windows: {model!r} is not in models")
    for model in provider.model_metadata:
        if model not in provider.models:
            raise CatalogError(f"{prefix}.model_metadata: {model!r} is not in models")
    if provider.thinking_default is not None and (
        provider.thinking_levels is None
        or provider.thinking_default not in provider.thinking_levels
    ):
        raise CatalogError(
            f"{prefix}.thinking_default: {provider.thinking_default!r} is not in thinking_levels"
        )

    for model, catalog_metadata in provider.model_metadata.items():
        _validate_cost_tiers(
            catalog_metadata.cost_tiers,
            field_name=f"{prefix}.model_metadata.{model}",
        )

    model_metadata = {
        model: _model_metadata_from_provider(metadata)
        for model, metadata in provider.model_metadata.items()
    }
    context_windows = dict(provider.context_windows or {})
    for model, metadata in model_metadata.items():
        if metadata.context_window is not None and model not in context_windows:
            context_windows[model] = metadata.context_window

    return ProviderCatalogEntry(
        name=provider.name,
        display_name=provider.display_name,
        kind=provider.kind,
        base_url=provider.base_url,
        api_key_env=provider.api_key_env,
        credential_name=provider.credential_name,
        models=provider.models,
        default_model=provider.default_model,
        docs_url=provider.docs_url,
        api=provider.api,
        context_windows=context_windows or None,
        headers=dict(provider.headers),
        compat=_json_object(provider.compat, f"{prefix}.compat"),
        model_metadata=model_metadata,
        thinking_levels=provider.thinking_levels,
        thinking_models=provider.thinking_models,
        thinking_default=provider.thinking_default,
        thinking_parameter=provider.thinking_parameter,
        removed_models=provider.removed_models,
        auth_methods=provider.auth_methods,
    )


def _validate_cost_tiers(
    tiers: tuple[_CatalogCostTier, ...],
    *,
    field_name: str,
) -> None:
    # [FUNCTION DOC]
    # HELPER: requires the final tier to have no max_input_tokens and all prior
    # tiers strictly increasing positive limits; raises CatalogError otherwise.
    if not tiers:
        return
    if tiers[-1].max_input_tokens is not None:
        raise CatalogError(f"{field_name}.cost_tiers: final tier must omit max_input_tokens")
    previous_limit = 0
    for index, tier in enumerate(tiers[:-1]):
        limit = tier.max_input_tokens
        if limit is None or limit <= previous_limit:
            raise CatalogError(
                f"{field_name}.cost_tiers.{index}.max_input_tokens: "
                "limits must be strictly increasing"
            )
        previous_limit = limit


def _model_metadata_from_provider(metadata: _CatalogModelMetadata) -> ModelCatalogMetadata:
    # [FUNCTION DOC]
    # HELPER: converts validated metadata to the catalog dataclass; unsupported
    # thinking levels are encoded as levels mapped to none in the level map, cost
    # tiers become rate mappings, and compat is checked for JSON values.
    thinking_level_map: dict[ThinkingLevel, str | None] = dict(metadata.thinking_level_map)
    for level in metadata.unsupported_thinking_levels:
        thinking_level_map[level] = None
    return ModelCatalogMetadata(
        name=metadata.name,
        api=metadata.api,
        base_url=metadata.base_url,
        reasoning=metadata.reasoning,
        input=metadata.input,
        cost=dict(metadata.cost) if metadata.cost else None,
        cost_tiers=tuple(
            ModelCostTier(
                max_input_tokens=tier.max_input_tokens,
                cost=_cost_tier_rates(tier),
            )
            for tier in metadata.cost_tiers
        ),
        context_window=metadata.context_window,
        max_tokens=metadata.max_tokens,
        headers=dict(metadata.headers),
        compat=_json_object(metadata.compat, "model_metadata.compat"),
        thinking_level_map=thinking_level_map,
    )


def _cost_tier_rates(tier: _CatalogCostTier) -> dict[str, float]:
    # [FUNCTION DOC]
    # HELPER: returns a mapping of input, output, cacheRead, cacheWrite rates,
    # including cacheWrite1h only when set.
    rates = {
        "input": tier.input,
        "output": tier.output,
        "cacheRead": tier.cacheRead,
        "cacheWrite": tier.cacheWrite,
    }
    if tier.cacheWrite1h is not None:
        rates["cacheWrite1h"] = tier.cacheWrite1h
    return rates


def _json_object(value: Mapping[str, Any], field_name: str) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: converts a mapping into a JSON object, validating every value.
    # Raises CatalogError on unsupported values.
    return {key: _json_value(item, f"{field_name}.{key}") for key, item in value.items()}


def _json_value(value: Any, field_name: str) -> JSONValue:
    # [FUNCTION DOC]
    # HELPER: recursively validates a JSON value (null, string, number, boolean,
    # list, string-keyed mapping); raises CatalogError naming the dotted field for
    # non-string keys or unsupported types.
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_value(item, field_name) for item in value]
    if isinstance(value, dict):
        output: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CatalogError(f"{field_name}: object keys must be strings")
            output[key] = _json_value(item, f"{field_name}.{key}")
        return output
    raise CatalogError(f"{field_name}: unsupported value {value!r}")


def _format_validation_error(raw: dict[str, Any], error: ValidationError) -> str:
    # [FUNCTION DOC]
    # HELPER: joins pydantic error issues as "dotted.location: message" separated
    # by semicolons.
    messages = []
    for issue in error.errors():
        location = ".".join(_dotted_location(raw, issue["loc"]))
        messages.append(f"{location}: {issue['msg']}")
    return "; ".join(messages)


def _dotted_location(raw: dict[str, Any], location: tuple[int | str, ...]) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: turns an error location into path parts, replacing the index after
    # "providers" with that provider's name when available.
    parts: list[str] = []
    for part in location:
        if parts and parts[-1] == "providers" and isinstance(part, int):
            providers = raw.get("providers")
            name = None
            if isinstance(providers, list) and part < len(providers):
                item = providers[part]
                if isinstance(item, dict):
                    name = item.get("name")
            parts.append(str(name) if isinstance(name, str) else str(part))
        else:
            parts.append(str(part))
    return parts


def _raw_provider_from_entry(entry: ProviderCatalogEntry) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: serializes a catalog entry back to a raw provider table, including
    # optional fields only when set and auth methods only when not the default.
    raw: dict[str, Any] = {
        "name": entry.name,
        "display_name": entry.display_name,
        "kind": entry.kind,
        "base_url": entry.base_url,
        "api_key_env": entry.api_key_env,
        "models": list(entry.models),
        "default_model": entry.default_model,
        "docs_url": entry.docs_url,
    }
    if entry.api is not None:
        raw["api"] = entry.api
    if entry.credential_name is not None:
        raw["credential_name"] = entry.credential_name
    if entry.context_windows:
        raw["context_windows"] = dict(entry.context_windows)
    if entry.headers:
        raw["headers"] = dict(entry.headers)
    if entry.compat:
        raw["compat"] = dict(entry.compat)
    if entry.model_metadata:
        raw["model_metadata"] = {
            model: _raw_model_metadata_from_entry(metadata)
            for model, metadata in entry.model_metadata.items()
        }
    if entry.thinking_levels is not None:
        raw["thinking_levels"] = list(entry.thinking_levels)
    if entry.thinking_models:
        raw["thinking_models"] = list(entry.thinking_models)
    if entry.thinking_default is not None:
        raw["thinking_default"] = entry.thinking_default
    if entry.thinking_parameter is not None:
        raw["thinking_parameter"] = entry.thinking_parameter
    if entry.removed_models:
        raw["removed_models"] = list(entry.removed_models)
    if entry.auth_methods != ("api_key",):
        raw["auth_methods"] = list(entry.auth_methods)
    return raw


def _raw_model_metadata_from_entry(metadata: ModelCatalogMetadata) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: serializes model metadata to a raw table with only set fields; level
    # map entries with none become the unsupported_thinking_levels list, cost
    # tiers become inline tables with optional max_input_tokens and rates.
    raw: dict[str, Any] = {}
    if metadata.name is not None:
        raw["name"] = metadata.name
    if metadata.api is not None:
        raw["api"] = metadata.api
    if metadata.base_url is not None:
        raw["base_url"] = metadata.base_url
    if metadata.reasoning is not None:
        raw["reasoning"] = metadata.reasoning
    if metadata.input:
        raw["input"] = list(metadata.input)
    if metadata.cost:
        raw["cost"] = dict(metadata.cost)
    if metadata.cost_tiers:
        raw["cost_tiers"] = [
            {
                **(
                    {"max_input_tokens": tier.max_input_tokens}
                    if tier.max_input_tokens is not None
                    else {}
                ),
                **tier.cost,
            }
            for tier in metadata.cost_tiers
        ]
    if metadata.context_window is not None:
        raw["context_window"] = metadata.context_window
    if metadata.max_tokens is not None:
        raw["max_tokens"] = metadata.max_tokens
    if metadata.headers:
        raw["headers"] = dict(metadata.headers)
    if metadata.compat:
        raw["compat"] = dict(metadata.compat)
    thinking_level_map = {
        level: value for level, value in metadata.thinking_level_map.items() if value is not None
    }
    unsupported = [level for level, value in metadata.thinking_level_map.items() if value is None]
    if thinking_level_map:
        raw["thinking_level_map"] = thinking_level_map
    if unsupported:
        raw["unsupported_thinking_levels"] = unsupported
    return raw


def _catalog_to_toml(raw: dict[str, Any]) -> str:
    # [FUNCTION DOC]
    # HELPER: renders a raw catalog as TOML text: schema_version line, then for
    # each provider an array-of-tables header, known scalar/list/inline keys in a
    # fixed order, a context windows sub-table and one sub-table per model
    # metadata entry. Ends with a single newline. Unknown provider keys are
    # dropped silently. Raises TypeError for unsupported values.
    lines = [f"schema_version = {raw.get('schema_version', CATALOG_SCHEMA_VERSION)}", ""]
    for provider in _raw_providers(raw):
        lines.append("[[providers]]")
        for key in (
            "name",
            "display_name",
            "kind",
            "base_url",
            "api_key_env",
            "credential_name",
            "models",
            "default_model",
            "docs_url",
            "api",
            "headers",
            "compat",
            "thinking_levels",
            "thinking_models",
            "thinking_default",
            "thinking_parameter",
            "removed_models",
            "auth_methods",
        ):
            if key in provider:
                lines.append(f"{key} = {_toml_value(provider[key])}")
        context_windows = provider.get("context_windows")
        if isinstance(context_windows, dict) and context_windows:
            lines.append("")
            lines.append("[providers.context_windows]")
            for model, context_window in context_windows.items():
                lines.append(f"{_toml_key(model)} = {_toml_value(context_window)}")
        model_metadata = provider.get("model_metadata")
        if isinstance(model_metadata, dict) and model_metadata:
            for model, metadata in model_metadata.items():
                if not isinstance(metadata, dict):
                    continue
                lines.append("")
                lines.append(f"[providers.model_metadata.{_toml_key(model)}]")
                for key, value in metadata.items():
                    lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _toml_key(value: str) -> str:
    # [FUNCTION DOC]
    # HELPER: returns the key bare when it is alphanumeric with dashes or
    # underscores and does not start with a digit, else a JSON-quoted key.
    # KNOWN ISSUE: an empty key raises IndexError instead of being quoted.
    if value.replace("_", "").replace("-", "").isalnum() and not value[0].isdigit():
        return value
    return json.dumps(value)


def _toml_value(value: object) -> str:
    # [FUNCTION DOC]
    # HELPER: renders a value as TOML: JSON-quoted strings, lowercase booleans,
    # numbers via their string form, arrays and inline tables recursively.
    # Raises TypeError for other types.
    # KNOWN ISSUE: infinity/NaN floats render as Python spellings that are not
    # valid TOML.
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{ "
            + ", ".join(
                f"{_toml_key(str(key))} = {_toml_value(item)}" for key, item in value.items()
            )
            + " }"
        )
    raise TypeError(f"Unsupported TOML value: {value!r}")


def _atomic_write_text(path: Path, text: str) -> None:
    # [FUNCTION DOC]
    # HELPER: writes text to a temporary file in the target directory then
    # renames it over the target; deletes the temporary file and re-raises on any
    # error. Side effect: file write/replace.
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
            temp_file.write(text)
            temp_file.flush()
        temp_path.replace(path)
    except Exception:
        if temp_path is not None:
            with suppress(OSError):
                temp_path.unlink()
        raise
