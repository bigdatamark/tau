# [MODULE DOC]
# Tests for CodexClientVersionResolver (tau_coding.codex_version), which resolves the OpenAI
# Codex client version to advertise by querying the npm "latest" release URL, caching the result
# (version, etag, checked_at, schema_version 1) in a JSON store under the Tau home, revalidating
# stale caches with If-None-Match, falling back to cache or to the built-in default version on
# failure or malformed versions, and honoring TAU_OFFLINE. Uses httpx MockTransport and injected clocks.
import json

import httpx
import pytest

from tau_ai.openai_codex import DEFAULT_OPENAI_CODEX_CLIENT_VERSION
from tau_coding.codex_version import (
    CODEX_NPM_LATEST_URL,
    CodexClientVersionResolver,
)
from tau_coding.paths import TauPaths


@pytest.mark.anyio
async def test_codex_version_resolver_fetches_and_caches_latest_release(tmp_path) -> None:
    # [FUNCTION DOC]
    # Async test: first call fetches the npm latest URL and returns its version; second call is served
    # from cache (exactly one request). Asserts the cache file contents including etag and checked_at.
    # Side effect: writes the version store in tmp_path.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock transport handler; records the request and returns 200 with version "0.153.4" and an etag.
        requests.append(request)
        return httpx.Response(200, json={"version": "0.153.4"}, headers={"etag": '"latest"'})

    paths = TauPaths(home=tmp_path / ".tau")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resolver = CodexClientVersionResolver(paths=paths, client=client, now=lambda: 1_000)
        assert await resolver() == "0.153.4"
        assert await resolver() == "0.153.4"

    assert len(requests) == 1
    assert str(requests[0].url) == CODEX_NPM_LATEST_URL
    assert json.loads(paths.codex_version_store_path.read_text(encoding="utf-8")) == {
        "checked_at": 1_000,
        "etag": '"latest"',
        "schema_version": 1,
        "version": "0.153.4",
    }


@pytest.mark.anyio
async def test_codex_version_resolver_revalidates_stale_cache(tmp_path) -> None:
    # [FUNCTION DOC]
    # Async test: with a stale cache (checked_at 1, clock 20000) the resolver sends If-None-Match with
    # the cached etag, accepts a 304, returns the cached version and updates checked_at to now.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock handler that records the request and returns 304 Not Modified.
        requests.append(request)
        return httpx.Response(304)

    paths = TauPaths(home=tmp_path / ".tau")
    paths.home.mkdir(parents=True)
    paths.codex_version_store_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checked_at": 1,
                "etag": '"cached"',
                "version": "0.154.0",
            }
        ),
        encoding="utf-8",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resolver = CodexClientVersionResolver(paths=paths, client=client, now=lambda: 20_000)
        assert await resolver() == "0.154.0"

    assert requests[0].headers["if-none-match"] == '"cached"'
    cache = json.loads(paths.codex_version_store_path.read_text(encoding="utf-8"))
    assert cache["checked_at"] == 20_000


@pytest.mark.anyio
async def test_codex_version_resolver_uses_stale_cache_on_failure(tmp_path) -> None:
    # [FUNCTION DOC]
    # Async test: when revalidation gets a 503, the stale cached version is still returned.
    paths = TauPaths(home=tmp_path / ".tau")
    paths.home.mkdir(parents=True)
    paths.codex_version_store_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checked_at": 1,
                "etag": None,
                "version": "0.154.0",
            }
        ),
        encoding="utf-8",
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock handler always returning 503.
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resolver = CodexClientVersionResolver(paths=paths, client=client, now=lambda: 20_000)
        assert await resolver() == "0.154.0"


@pytest.mark.anyio
async def test_codex_version_resolver_rejects_malformed_version(tmp_path) -> None:
    # [FUNCTION DOC]
    # Async test: a registry version that is not a plain release (pre-release suffix) is rejected and
    # the resolver returns the built-in default Codex client version.
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock handler returning a malformed/pre-release version string.
        return httpx.Response(200, json={"version": "999.999.999-next.1"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resolver = CodexClientVersionResolver(paths=TauPaths(home=tmp_path / ".tau"), client=client)
        assert await resolver() == DEFAULT_OPENAI_CODEX_CLIENT_VERSION


@pytest.mark.anyio
async def test_codex_version_resolver_offline_uses_cache_without_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    # [FUNCTION DOC]
    # Async test: with TAU_OFFLINE=1 and a stale cache, the resolver returns the cached version without
    # any network call (the handler would fail the test if invoked).
    monkeypatch.setenv("TAU_OFFLINE", "1")
    paths = TauPaths(home=tmp_path / ".tau")
    paths.home.mkdir(parents=True)
    paths.codex_version_store_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checked_at": 1,
                "etag": None,
                "version": "0.154.0",
            }
        ),
        encoding="utf-8",
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock handler that raises AssertionError to prove no network access happens offline.
        raise AssertionError("offline resolution must not use the network")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resolver = CodexClientVersionResolver(paths=paths, client=client)
        assert await resolver() == "0.154.0"
