# [MODULE DOC]
# Tests for an experimental Ollama adapter spike (fixture in tests/fixtures) that
# plugs into the dynamic provider registry and local backend registry using only
# Ollama's official HTTP APIs: OpenAI-compatible model listing plus native tags/ps
# endpoints for backend state. Verifies shapes fit without llama.cpp-specific concepts.
"""Bounded official-API Ollama adapter spike; not a shipped backend."""

from __future__ import annotations

import json

import httpx
import pytest

from fixtures.ollama_adapter_spike import OllamaAdapterSpike, context
from tau_coding.extensions import DynamicProviderRegistry, LocalBackendRegistry

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    # [FUNCTION DOC]
    # HELPER: Pytest fixture forcing anyio tests in this module to run on asyncio only.
    # Returns the string "asyncio".
    return "asyncio"


def _client() -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    # [FUNCTION DOC]
    # HELPER: Build an httpx async client backed by a mock transport that serves fake
    # Ollama responses: /v1/models (two models), /api/tags (same two names), /api/ps
    # (first model running). Any other path raises an assertion error.
    # Output: a pair of the client and a list that records every request (mutated as
    # requests arrive).
    requests: list[httpx.Request] = []

    def dispatch(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock transport handler. Appends the request to the shared request list,
        # then returns the canned JSON response for the known paths.
        # Raises AssertionError for unknown paths.
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "qwen3:8b", "object": "model", "owned_by": "library"},
                        {"id": "code:latest", "object": "model", "owned_by": "library"},
                    ],
                },
            )
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "qwen3:8b"}, {"name": "code:latest"}]},
            )
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})
        raise AssertionError(f"unexpected Ollama spike request: {request.url}")

    return httpx.AsyncClient(transport=httpx.MockTransport(dispatch)), requests


async def test_official_ollama_shapes_fit_without_llama_concepts() -> None:
    # [FUNCTION DOC]
    # Registers the spike's provider and backend under one id, refreshes both, and
    # asserts the provider is published with both model ids in order, the backend
    # reports the first model as running and selected, and exactly the three endpoints
    # were requested in order. Closes registries and client afterwards.
    client, requests = _client()
    adapter = OllamaAdapterSpike(client)
    providers = DynamicProviderRegistry(generation_id="generation")
    backends = LocalBackendRegistry(providers, generation_id="generation")
    providers.register("test:ollama-spike", adapter.provider())
    backends.register("test:ollama-spike", adapter.backend())

    provider_result = await providers.refresh("ollama-spike")
    backend_result = await backends.refresh("ollama-spike")

    assert provider_result.status == "published"
    assert provider_result.provider is not None
    assert [model.id for model in provider_result.provider.models] == [
        "qwen3:8b",
        "code:latest",
    ]
    assert backend_result.backend_status is not None
    assert backend_result.backend_status.models[0].state == "running"
    assert backend_result.backend_status.selected_model == "qwen3:8b"
    assert [request.url.path for request in requests] == [
        "/v1/models",
        "/api/tags",
        "/api/ps",
    ]
    await backends.aclose()
    await providers.aclose()
    await client.aclose()


async def test_ollama_openai_compatibility_uses_no_auth_and_offline_cache() -> None:
    # [FUNCTION DOC]
    # Refreshing the provider with network disallowed must still publish it, with an
    # empty model list, and must make no HTTP requests.
    client, requests = _client()
    adapter = OllamaAdapterSpike(client)
    providers = DynamicProviderRegistry(generation_id="generation")
    providers.register("test:ollama-spike", adapter.provider())
    provider_result = await providers.refresh("ollama-spike", allow_network=False)

    assert provider_result.status == "published"
    assert provider_result.provider is not None
    assert provider_result.provider.models == ()
    assert requests == []
    await providers.aclose()
    await client.aclose()


async def test_ollama_spike_rejects_malformed_native_state() -> None:
    # [FUNCTION DOC]
    # When the native endpoint returns "models" as an object instead of a list, the
    # adapter's status call must raise ValueError mentioning "models list".
    async def dispatch(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock handler returning HTTP 200 with a JSON body whose "models" is an empty
        # object, for any request.
        del request
        return httpx.Response(200, content=json.dumps({"models": {}}).encode())

    client = httpx.AsyncClient(transport=httpx.MockTransport(dispatch))
    adapter = OllamaAdapterSpike(client)
    with pytest.raises(ValueError, match="models list"):
        await adapter.status(context())
    await client.aclose()
