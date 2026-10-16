# [MODULE DOC]
# Tests for the models.dev catalog store: refreshing and persisting the remote catalog cache,
# choosing between the cache and the bundled catalog by age, ETag revalidation, offline mode,
# freshness-based skipping, and preserving the cache on failed refreshes. Network is simulated
# with httpx mock transports serving a JSON fixture.
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

import tau_coding.models_dev_store as store
from tau_coding.catalog_loader import effective_catalog
from tau_coding.models_dev import MODELS_DEV_URL, NVIDIA_MODELS_URL
from tau_coding.models_dev_store import (
    ModelsDevRefreshError,
    cached_models_dev_catalog_document,
    refresh_models_dev_catalog,
)
from tau_coding.paths import TauPaths

FIXTURE = Path(__file__).parent / "fixtures/models_dev_catalog.json"


def _source() -> dict[str, object]:
    # [FUNCTION DOC]
    # HELPER: load the models.dev fixture JSON from the tests fixtures directory and add an empty
    # "nvidia" provider entry; return the resulting mapping. Reads a file.
    source = json.loads(FIXTURE.read_text(encoding="utf-8"))
    source["nvidia"] = {"id": "nvidia", "models": {}}
    return source


def _client(source: dict[str, object]) -> httpx.AsyncClient:
    # [FUNCTION DOC]
    # HELPER: build an async httpx client backed by a mock transport that serves the given catalog
    # (status 200 with an ETag) for the models.dev URL, an empty data list for the NVIDIA models
    # URL, and fails the test for any other URL. Returns the client (caller closes it).
    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: mock transport handler implementing the routing above; raises an assertion error
        # on unexpected requests.
        if str(request.url) == MODELS_DEV_URL:
            return httpx.Response(200, json=source, headers={"etag": '"fixture"'})
        if str(request.url) == NVIDIA_MODELS_URL:
            return httpx.Response(200, json={"data": []})
        raise AssertionError(f"unexpected request: {request.url}")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.anyio
async def test_refresh_persists_catalog_and_effective_catalog_uses_newer_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC]
    # Async: force-refresh into a temp Tau home and verify the result reports refreshed, a
    # positive model count and an existing cache file. Then monkeypatch the bundled catalog's
    # generation time: when bundled is newer the cached document is ignored (None); when bundled
    # is older the cache is used, and the effective catalog's huggingface provider includes a
    # model only present in the fixture.
    paths = TauPaths(home=tmp_path / ".tau")
    async with _client(_source()) as client:
        result = await refresh_models_dev_catalog(
            paths=paths,
            force=True,
            client=client,
            now=1000.0,
        )

    assert result.refreshed
    assert result.model_count > 0
    assert result.cache_path.exists()
    monkeypatch.setattr(
        store,
        "bundled_models_dev_catalog_document",
        lambda: {"generated_at": 2_000_000},
    )
    assert cached_models_dev_catalog_document(paths) is None
    monkeypatch.setattr(store, "bundled_models_dev_catalog_document", lambda: {"generated_at": 0})
    assert cached_models_dev_catalog_document(paths) is not None
    huggingface = next(
        provider for provider in effective_catalog(paths) if provider.name == "huggingface"
    )
    assert "example/new-tool-model" in huggingface.models


@pytest.mark.anyio
async def test_refresh_revalidates_with_etag_and_preserves_cached_body(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Async: after an initial refresh, a second forced refresh must send the stored ETag as if-
    # none-match; on a 304 response the result is marked not-modified, the store's checked-at time
    # updates to the new time, and the cached catalog body is unchanged.
    paths = TauPaths(home=tmp_path / ".tau")
    async with _client(_source()) as client:
        await refresh_models_dev_catalog(paths=paths, force=True, client=client, now=1000.0)
    result_path = paths.home / "models-store.json"
    before_catalog = json.loads(result_path.read_text(encoding="utf-8"))["catalog"]

    def not_modified(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: mock handler asserting the conditional ETag header and returning 304.
        assert request.headers["if-none-match"] == '"fixture"'
        return httpx.Response(304)

    async with httpx.AsyncClient(transport=httpx.MockTransport(not_modified)) as client:
        result = await refresh_models_dev_catalog(
            paths=paths,
            force=True,
            client=client,
            now=2000.0,
        )

    after = json.loads(result_path.read_text(encoding="utf-8"))
    assert result.not_modified
    assert after["checked_at"] == 2000.0
    assert after["catalog"] == before_catalog


@pytest.mark.anyio
async def test_offline_mode_uses_bundled_catalog_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC]
    # Async: with TAU_OFFLINE set, a forced refresh must not touch the network, reports not
    # refreshed, and still yields a positive model count from the bundled catalog.
    monkeypatch.setenv("TAU_OFFLINE", "1")

    def no_network(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: mock handler that fails the test if any request is made.
        raise AssertionError("offline mode should avoid network")

    async with httpx.AsyncClient(transport=httpx.MockTransport(no_network)) as client:
        result = await refresh_models_dev_catalog(
            paths=TauPaths(home=tmp_path / ".tau"),
            force=True,
            client=client,
        )

    assert not result.refreshed
    assert result.model_count > 0


@pytest.mark.anyio
async def test_fresh_cache_skips_network_and_failed_force_preserves_it(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Async: after a refresh, a non-forced refresh one second later skips the network (not
    # refreshed); a forced refresh hitting a 503 raises the refresh error mentioning 503 and
    # leaves the cache file contents unchanged.
    paths = TauPaths(home=tmp_path / ".tau")
    async with _client(_source()) as client:
        first = await refresh_models_dev_catalog(paths=paths, force=True, client=client, now=1000.0)
    cached_text = first.cache_path.read_text(encoding="utf-8")

    def no_network(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: mock handler that fails the test if any request is made.
        raise AssertionError("fresh cache should avoid network")

    async with httpx.AsyncClient(transport=httpx.MockTransport(no_network)) as client:
        skipped = await refresh_models_dev_catalog(
            paths=paths,
            client=client,
            now=1001.0,
        )
    assert not skipped.refreshed

    def failure(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: mock handler that always returns HTTP 503.
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(failure)) as client:
        with pytest.raises(ModelsDevRefreshError, match="503"):
            await refresh_models_dev_catalog(
                paths=paths,
                force=True,
                client=client,
                now=2000.0,
            )
    assert first.cache_path.read_text(encoding="utf-8") == cached_text
