# [MODULE DOC]
# HELPER: test fixture module containing a non-shipping Ollama adapter spike
# used to validate that the generic extension seams (dynamic provider over an
# OpenAI-compatible transport, local backend status/configure/refresh) are
# sufficient for a second local backend. Endpoint shapes follow Ollama's
# documented API (/v1/models, /api/tags, /api/ps); tests feed deterministic
# HTTP responses through the supplied client.
"""Non-shipping Ollama adapter spike for Phase 6 contract validation.

This file deliberately lives under tests: it is an executable contract probe,
not a shipped Ollama integration. The endpoint shapes come from Ollama's
current official API documentation; the test supplies deterministic responses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import httpx

from tau_agent.harness import SimpleCancellationToken
from tau_coding.extensions import (
    DynamicProvider,
    LocalBackend,
    LocalBackendStatus,
    LocalConfigureResult,
    LocalConfigureSpec,
    LocalConfigValues,
    LocalModel,
    LocalOperationContext,
    LocalOperationResult,
    NoAuth,
    OpenAICompatibleTransport,
    ProviderModel,
    ProviderModelSnapshot,
)


@dataclass(slots=True)
class OllamaAdapterSpike:
    # [CLASS DOC]
    # HELPER: mutable slotted dataclass adapter holding an async HTTP client and
    # an endpoint (default http://ollama.test:11434, trailing slash stripped).
    # Builds a dynamic provider and a local backend that call back into itself.
    """Small adapter exercising only generic provider/backend seams."""

    client: httpx.AsyncClient
    endpoint: str = "http://ollama.test:11434"

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # HELPER: normalize the endpoint by stripping trailing slashes (mutates self).
        self.endpoint = self.endpoint.rstrip("/")

    def provider(self) -> DynamicProvider:
        # [FUNCTION DOC]
        # HELPER: return a DynamicProvider with id "ollama-spike", an
        # OpenAI-compatible transport at <endpoint>/v1 with no auth using the stored
        # client, and this adapter's model refresh callback.
        return DynamicProvider(
            id="ollama-spike",
            display_name="Ollama (spike)",
            transport=OpenAICompatibleTransport(
                base_url=f"{self.endpoint}/v1",
                auth=NoAuth(),
                client=self.client,
            ),
            refresh_models=self.refresh_models,
        )

    def backend(self) -> LocalBackend:
        # [FUNCTION DOC]
        # HELPER: return a LocalBackend with id/provider id "ollama-spike", an empty
        # configure spec, and this adapter's configure/status/refresh callbacks.
        return LocalBackend(
            id="ollama-spike",
            provider_id="ollama-spike",
            display_name="Ollama (spike)",
            configure_spec=LocalConfigureSpec(),
            configure=self.configure,
            status=self.status,
            refresh=self.refresh,
        )

    async def refresh_models(self, context) -> ProviderModelSnapshot:
        # [FUNCTION DOC]
        # HELPER: model refresh callback. If the refresh context disallows network,
        # returns a snapshot of the cached models. Otherwise GETs <endpoint>/v1/models,
        # raises for HTTP errors, requires a "data" list (ValueError otherwise), and
        # maps each entry with a string id to a provider model (openai-completions
        # API, compat copied from string object/owned_by fields). Output: snapshot
        # whose selected model is set only when exactly one model exists.
        if not context.allow_network:
            return ProviderModelSnapshot(context.cached_models)
        response = await self.client.get(f"{self.endpoint}/v1/models")
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(data, list):
            raise ValueError("Ollama /v1/models response has no data list")
        models = tuple(
            ProviderModel(
                id=item["id"],
                display_name=item.get("id"),
                api="openai-completions",
                compat={
                    key: item[key]
                    for key in ("object", "owned_by")
                    if isinstance(item.get(key), str)
                },
            )
            for item in data
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        )
        return ProviderModelSnapshot(models, models[0].id if len(models) == 1 else None)

    async def configure(
        self,
        values: LocalConfigValues,
        context: LocalOperationContext,
    ) -> LocalConfigureResult:
        # [FUNCTION DOC]
        # HELPER: no-op configure callback; ignores inputs and returns a committed
        # configure result.
        del values, context
        return LocalConfigureResult(committed=True)

    async def status(
        self, context: LocalOperationContext
    ) -> LocalBackendStatus | LocalOperationResult:
        # [FUNCTION DOC]
        # HELPER: status callback. Fetches installed tags and running models, builds
        # local models marked running or installed, and returns a ready status with
        # endpoint display <endpoint>/v1, the models, a selected model (one of the
        # running ids if it is installed), and actions configure/refresh plus use when
        # any models exist. HTTP and ValueError failures propagate.
        # KNOWN ISSUE: the selected model is taken from an unordered set of running
        # ids, so with several running models the selection is nondeterministic.
        del context
        tags_response, running_response = await self._get_model_state()
        installed = _models_from_tags(tags_response)
        running = _running_ids(running_response)
        models = tuple(
            LocalModel(model_id, state="running" if model_id in running else "installed")
            for model_id in installed
        )
        selected = next(iter(running), None)
        return LocalBackendStatus(
            state="ready",
            endpoint_display=f"{self.endpoint}/v1",
            models=models,
            selected_model=selected if selected in {item.id for item in models} else None,
            actions=("configure", "refresh", "use") if models else ("configure", "refresh"),
        )

    async def refresh(
        self, context: LocalOperationContext
    ) -> LocalBackendStatus | LocalOperationResult:
        # [FUNCTION DOC]
        # HELPER: refresh callback; simply returns the status result.
        return await self.status(context)

    async def _get_model_state(self) -> tuple[Mapping[str, object], Mapping[str, object]]:
        # [FUNCTION DOC]
        # HELPER: GET /api/tags then /api/ps, raise for HTTP errors, and return both
        # JSON payloads; raises ValueError if either is not an object.
        tags = await self.client.get(f"{self.endpoint}/api/tags")
        tags.raise_for_status()
        running = await self.client.get(f"{self.endpoint}/api/ps")
        running.raise_for_status()
        tags_payload = tags.json()
        running_payload = running.json()
        if not isinstance(tags_payload, Mapping) or not isinstance(running_payload, Mapping):
            raise ValueError("Ollama model state response is malformed")
        return tags_payload, running_payload


def _models_from_tags(payload: Mapping[str, object]) -> tuple[str, ...]:
    # [FUNCTION DOC]
    # HELPER: extract the tuple of non-empty string model names from an
    # /api/tags payload; raises ValueError if "models" is not a list.
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("Ollama /api/tags response has no models list")
    result = tuple(
        item.get("name")
        for item in models
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    )
    return tuple(model for model in result if model)


def _running_ids(payload: Mapping[str, object]) -> frozenset[str]:
    # [FUNCTION DOC]
    # HELPER: extract the set of string model names from an /api/ps payload;
    # raises ValueError if "models" is not a list.
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("Ollama /api/ps response has no models list")
    return frozenset(
        item.get("name")
        for item in models
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    )


def context() -> LocalOperationContext:
    # [FUNCTION DOC]
    # HELPER: build a deterministic local operation context for direct probes:
    # fresh cancellation token, action refresh, fixed generation/backend/source
    # ids, always-current check and a no-op progress callback.
    """Return a deterministic operation context for direct adapter probes."""
    return LocalOperationContext(
        signal=SimpleCancellationToken(),
        action="refresh",
        generation_id="ollama-spike",
        backend_id="ollama-spike",
        source_id="test:ollama-spike",
        _is_current=lambda: True,
        _progress=lambda _: None,
    )
