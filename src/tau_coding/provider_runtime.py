# [MODULE DOC]
# Runtime model-provider construction for Tau coding sessions (tau_coding layer bridging
# durable provider settings and extension-defined dynamic providers to tau_ai provider
# implementations). Builds Anthropic, OpenAI Codex, OpenAI-compatible, Google and Mistral
# providers; resolves API keys and OAuth credentials (refreshing and persisting rotated
# tokens under a per-event-loop, per-credential lock so refresh tokens are spent once);
# and wraps extension authentication errors behind a secret-safe boundary.
"""Runtime provider construction for Tau coding sessions."""

from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop, get_running_loop
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import replace
from inspect import isawaitable
from os import environ
from typing import Protocol
from weakref import WeakKeyDictionary

from tau_agent.provider import ModelProvider
from tau_ai.anthropic import AnthropicProvider
from tau_ai.env import AnthropicConfig, OpenAICompatibleConfig, RuntimeProviderAuth
from tau_ai.google import GoogleGenerativeAIProvider
from tau_ai.mistral import MistralConversationsProvider
from tau_ai.openai_codex import (
    OpenAICodexConfig,
    OpenAICodexCredentials,
    OpenAICodexProvider,
)
from tau_ai.openai_compatible import OpenAICompatibleProvider
from tau_coding.codex_version import CodexClientVersionResolver
from tau_coding.credentials import FileCredentialStore, OAuthCredential
from tau_coding.extensions.providers import (
    CredentialReader,
    DynamicProvider,
    OpenAICompatibleTransport,
    ProviderAuthError,
    ProviderModel,
    ProviderRuntimeContext,
    RequiredApiKey,
    ResolvedProviderAuth,
    _MissingRequiredApiKeyError,
    json_compatible_mapping,
    resolve_provider_auth,
)
from tau_coding.oauth import (
    account_id_from_access_token,
    oauth_credential_is_expired,
    refresh_openai_codex_token,
)
from tau_coding.oauth_registry import get_oauth_provider
from tau_coding.oauth_types import OAuthProvider
from tau_coding.paths import TauPaths
from tau_coding.provider_config import (
    AnthropicProviderConfig,
    OpenAICodexProviderConfig,
    OpenAICompatibleProviderConfig,
    ProviderConfig,
    ProviderConfigError,
    anthropic_cache_settings,
    anthropic_config_from_provider,
    openai_compatible_config_from_provider,
    provider_model_max_tokens,
    provider_model_supports_images,
    provider_thinking_levels,
    validate_huggingface_inference_provider,
    validate_provider_model,
)
from tau_coding.thinking import ThinkingLevel, normalize_thinking_level, reasoning_effort_for_level


class ClosableModelProvider(ModelProvider, Protocol):
    # [CLASS DOC]
    # Protocol: a model provider (stream_response) that Tau owns and can close via aclose.
    """Runtime provider object Tau owns and can close."""

    async def aclose(self) -> None:
        # [FUNCTION DOC]
        # Protocol method: release provider-owned resources such as HTTP clients.
        """Close any provider-owned resources."""
        ...


async def create_dynamic_model_provider(
    provider: DynamicProvider,
    *,
    model: str,
    credential_store: CredentialReader | None = None,
    environment: Mapping[str, str] | None = None,
) -> ClosableModelProvider:
    # [FUNCTION DOC]
    # Create a candidate runtime provider from a process-local (extension) DynamicProvider.
    # Inputs: the provider definition; keyword model id; optional credential reader
    # (defaults to the file credential store); optional environment mapping (defaults to
    # the process environment).
    # Behavior: looks up the model (ProviderConfigError if not listed), resolves auth via the
    # secret-safe resolver, and builds a runtime context (provider id + auth).
    # If the provider has a runtime factory: calls it with the context and model (awaiting
    # if it returns an awaitable). Validates the result has callable stream_response and
    # aclose (attribute access errors are treated as missing); otherwise best-effort closes
    # it (if closable, swallowing errors) and raises ProviderConfigError.
    # Otherwise the transport must be OpenAI-compatible (asserted). The effective API is the
    # model's API or the transport's; anything other than openai-completions/responses
    # raises ProviderConfigError. Headers are merged case-insensitively (transport, model,
    # auth; later wins). If auth resolved an API key while asking to omit the Authorization
    # header and no explicit Authorization header exists, raises ProviderConfigError.
    # Builds an OpenAI-compatible config (API key or empty, model/transport base URL,
    # timeouts/retries, max tokens, image support when the model lists the image modality,
    # JSON-compatible compat settings, provider name, no API inference from model id) and
    # returns an OpenAICompatibleProvider using the transport's optional shared client.
    # Never writes the dynamic definition to durable settings. May raise ProviderAuthError.
    """Create a candidate runtime from a process-local provider definition.

    Authentication is resolved only here, immediately before construction.
    This path never converts or writes the dynamic definition to durable
    ``ProviderConfig`` settings.
    """
    selected_model = _dynamic_model(provider, model)
    auth = await _resolve_dynamic_runtime_auth(
        provider,
        credentials=(credential_store if credential_store is not None else FileCredentialStore()),
        environment=environment if environment is not None else environ,
    )
    context = ProviderRuntimeContext(provider_id=provider.id, auth=auth)
    if provider.runtime_factory is not None:
        candidate = provider.runtime_factory(context, selected_model)
        runtime = await candidate if isawaitable(candidate) else candidate
        try:
            stream_response = getattr(runtime, "stream_response", None)
        except BaseException:  # noqa: BLE001 - extension object validation boundary
            stream_response = None
        try:
            close = getattr(runtime, "aclose", None)
        except BaseException:  # noqa: BLE001 - extension object validation boundary
            close = None
        if not callable(stream_response) or not callable(close):
            error = ProviderConfigError(
                f"Runtime factory for {provider.id} returned an unsupported provider"
            )
            if callable(close):
                try:
                    close_result = close()
                    if isawaitable(close_result):
                        await close_result
                except BaseException:  # noqa: BLE001 - preserve the validation error
                    pass
            raise error
        return runtime

    transport = provider.transport
    assert isinstance(transport, OpenAICompatibleTransport)
    selected_api = selected_model.api or transport.api
    if selected_api not in {"openai-completions", "openai-responses"}:
        raise ProviderConfigError(
            f"OpenAI-compatible dynamic provider {provider.id} cannot use api {selected_api}"
        )
    headers = _merge_dynamic_headers(
        transport.headers,
        selected_model.headers,
        auth.headers,
    )
    has_authorization = any(key.casefold() == "authorization" for key in headers)
    if auth.api_key is not None and auth.omit_authorization_header and not has_authorization:
        raise ProviderConfigError(
            f"OpenAI-compatible dynamic provider {provider.id} resolved an API key "
            "while requesting Authorization omission"
        )
    config = OpenAICompatibleConfig(
        api_key=auth.api_key or "",
        base_url=selected_model.base_url or transport.base_url,
        headers=headers,
        timeout_seconds=transport.timeout_seconds,
        max_retries=transport.max_retries,
        max_retry_delay_seconds=transport.max_retry_delay_seconds,
        api=selected_api,
        max_tokens=selected_model.max_tokens,
        supports_images=(
            selected_model.input_modalities is not None
            and "image" in selected_model.input_modalities
        ),
        compat=json_compatible_mapping(selected_model.compat),
        provider_name=provider.id,
        omit_authorization_header=auth.omit_authorization_header,
        # Dynamic providers explicitly own their API choice. A local model id
        # resembling gpt-* or *codex* must not reroute to /responses.
        infer_api_from_model=False,
    )
    return OpenAICompatibleProvider(config, client=transport.client)


async def _resolve_dynamic_runtime_auth(
    provider: DynamicProvider,
    *,
    credentials: CredentialReader,
    environment: Mapping[str, str],
) -> ResolvedProviderAuth:
    # [FUNCTION DOC]
    # HELPER: Resolve a dynamic provider's auth strategy with credentials and environment,
    # hiding extension-authored error text. Cancellation is re-raised as a fresh
    # CancelledError without its message or cause. A missing-required-key error is
    # re-raised as-is only when the strategy is exactly Tau's RequiredApiKey; otherwise, and
    # for any ProviderAuthError or other exception, raises a generic ProviderAuthError
    # ("Dynamic provider authentication resolution failed") with context suppressed.
    # Returns the resolved auth (API key, headers, omit-authorization flag).
    """Resolve extension auth behind a categorical secret-safe boundary."""
    try:
        return await resolve_provider_auth(
            provider.auth,
            credentials=credentials,
            environment=environment,
        )
    except asyncio.CancelledError:
        # Keep cancellation semantics without retaining an extension-authored
        # cancellation message that could contain credential material.
        raise asyncio.CancelledError from None
    except _MissingRequiredApiKeyError:
        # Preserve only Tau's exact strategy and host-authored missing-key error.
        if type(provider.auth) is RequiredApiKey:
            raise
        raise ProviderAuthError("Dynamic provider authentication resolution failed") from None
    except ProviderAuthError:
        # Custom strategies can raise ProviderAuthError too, so their arbitrary
        # text crosses the same categorical boundary as any extension exception.
        raise ProviderAuthError("Dynamic provider authentication resolution failed") from None
    except BaseException:  # noqa: BLE001 - extension authentication boundary
        raise ProviderAuthError("Dynamic provider authentication resolution failed") from None


def _dynamic_model(provider: DynamicProvider, model: str) -> ProviderModel:
    # [FUNCTION DOC]
    # HELPER: Return the provider model whose id exactly equals the requested id, else raise
    # ProviderConfigError naming the provider and model.
    for candidate in provider.models:
        if candidate.id == model:
            return candidate
    raise ProviderConfigError(f"Model is not configured for provider {provider.id}: {model}")


def _merge_dynamic_headers(*values: Mapping[str, str]) -> dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: Merge several header mappings case-insensitively; a later key replaces any
    # earlier key with the same case-folded name (the earlier spelling is removed and the
    # new spelling appended). Returns a new mapping.
    """Merge transport/model/auth headers case-insensitively, latest value winning."""
    merged: dict[str, str] = {}
    names: dict[str, str] = {}
    for value in values:
        for key, item in value.items():
            normalized = key.casefold()
            previous = names.get(normalized)
            if previous is not None:
                merged.pop(previous)
            names[normalized] = key
            merged[key] = item
    return merged


def create_model_provider(
    provider: ProviderConfig,
    *,
    credential_store: FileCredentialStore | None = None,
    model: str | None = None,
    thinking_level: ThinkingLevel | None = None,
    inference_provider: str | None = None,
    response_headers_observer: Callable[[Mapping[str, str]], None] | None = None,
) -> ClosableModelProvider:
    # [FUNCTION DOC]
    # Create a runtime provider from durable provider settings.
    # Inputs: provider config; keyword optional credential store (default file store),
    # model id, thinking level, Hugging Face inference-provider pin and a response-headers
    # observer callback.
    # Validates the model against the config if given. An inference provider is allowed
    # only for the "huggingface" provider with a model (else ProviderConfigError) and is
    # validated/normalised.
    # - Anthropic config: builds the Anthropic config; if an OAuth credential exists for a
    # registered OAuth provider, switches to bearer auth with the OAuth runtime API key and
    # merged headers, sets the Claude Code system prompt prefix required for OAuth, OAuth
    # cache retention, and an OAuth refreshing credential resolver. Returns AnthropicProvider.
    # - OpenAI Codex config: returns an OpenAICodexProvider with a Codex credential
    # resolver, base URL, headers, timeouts/retries, reasoning effort derived from the
    # thinking level, image support, and a client-version resolver rooted at the
    # credential store's directory.
    # - OpenAI-compatible config: builds the compatible config; applies a model alias
    # "<model>:<inference provider>" when pinned; attaches the headers observer; if an
    # OAuth credential exists, applies runtime auth (API key, base URL override, merged
    # headers) and an OAuth resolver. Then dispatches on the selected API:
    # "anthropic-messages" requires OAuth (else ProviderConfigError) and returns an
    # AnthropicProvider with bearer auth and gateway cache settings;
    # "google-generative-ai" -> Google provider; "mistral-conversations" -> Mistral
    # provider; otherwise OpenAICompatibleProvider.
    # Any other config type raises ProviderConfigError. Reads credentials from disk;
    # RuntimeError if an OAuth credential exists without a registered OAuth implementation.
    """Create a runtime model provider from durable provider settings."""
    if model is not None:
        validate_provider_model(provider, model)
    if inference_provider is not None:
        if provider.name != "huggingface" or model is None:
            raise ProviderConfigError(
                "Inference-provider pinning is only available for Hugging Face models"
            )
        inference_provider = validate_huggingface_inference_provider(inference_provider)
    credentials = credential_store or FileCredentialStore()
    if isinstance(provider, AnthropicProviderConfig):
        credential = _oauth_credential(provider, credentials)
        config = anthropic_config_from_provider(
            provider,
            credential_reader=credentials,
            model=model,
            thinking_level=thinking_level,
        )
        if credential is not None:
            runtime_auth = _required_oauth_provider(provider.name).runtime_auth(credential)
            oauth_retention, _ = anthropic_cache_settings(provider, model, oauth=True)
            config = replace(
                config,
                api_key=runtime_auth.api_key,
                bearer_auth=True,
                headers={**dict(config.headers or {}), **dict(runtime_auth.headers or {})},
                oauth_system_prompt="You are Claude Code, Anthropic's official CLI for Claude.",
                cache_retention=oauth_retention,
                credential_resolver=OAuthRuntimeCredentialResolver(
                    provider,
                    credential_store=credentials,
                ),
            )
        return AnthropicProvider(config)
    if isinstance(provider, OpenAICodexProviderConfig):
        return OpenAICodexProvider(
            OpenAICodexConfig(
                credential_resolver=OpenAICodexCredentialResolver(
                    provider,
                    credential_store=credentials,
                ),
                base_url=provider.base_url,
                provider_name=provider.name,
                headers=provider.headers,
                timeout_seconds=provider.timeout_seconds,
                max_retries=provider.max_retries,
                max_retry_delay_seconds=provider.max_retry_delay_seconds,
                reasoning_effort=_codex_reasoning_effort(
                    provider,
                    model=model,
                    thinking_level=thinking_level,
                ),
                supports_images=provider_model_supports_images(provider, model),
                client_version_resolver=CodexClientVersionResolver(
                    paths=TauPaths(home=credentials.path.parent)
                ),
            )
        )
    if isinstance(provider, OpenAICompatibleProviderConfig):
        credential = _oauth_credential(provider, credentials)
        compatible_config = openai_compatible_config_from_provider(
            provider,
            credential_reader=credentials,
            model=model,
            thinking_level=thinking_level,
        )
        if inference_provider is not None and model is not None:
            compatible_config = replace(
                compatible_config,
                model_aliases={model: f"{model}:{inference_provider}"},
            )
        if response_headers_observer is not None:
            compatible_config = replace(
                compatible_config,
                response_headers_observer=response_headers_observer,
            )
        if credential is not None:
            runtime_auth = _required_oauth_provider(provider.name).runtime_auth(credential)
            compatible_config = replace(
                compatible_config,
                api_key=runtime_auth.api_key,
                base_url=runtime_auth.base_url or compatible_config.base_url,
                headers={
                    **dict(compatible_config.headers or {}),
                    **dict(runtime_auth.headers or {}),
                },
                credential_resolver=OAuthRuntimeCredentialResolver(
                    provider,
                    credential_store=credentials,
                ),
            )
        selected_api = compatible_config.api
        if selected_api == "anthropic-messages":
            if credential is None:
                raise ProviderConfigError(
                    "Anthropic-protocol models on openai-compatible providers require OAuth"
                )
            gateway_retention, gateway_cache_control_on_tools = anthropic_cache_settings(
                provider, model, oauth=True
            )
            anthropic_config = AnthropicConfig(
                api_key=compatible_config.api_key,
                base_url=compatible_config.base_url,
                headers=compatible_config.headers,
                timeout_seconds=compatible_config.timeout_seconds,
                provider_name=compatible_config.provider_name,
                max_retries=compatible_config.max_retries,
                max_retry_delay_seconds=compatible_config.max_retry_delay_seconds,
                max_tokens=provider_model_max_tokens(provider, model),
                bearer_auth=True,
                credential_resolver=compatible_config.credential_resolver,
                supports_images=compatible_config.supports_images,
                # Resolved from compat like the first-party path, so a gateway
                # proxying real Claude can opt back in per provider or per model.
                cache_retention=gateway_retention,
                cache_control_on_tools=gateway_cache_control_on_tools,
            )
            return AnthropicProvider(anthropic_config)
        if selected_api == "google-generative-ai":
            return GoogleGenerativeAIProvider(compatible_config)
        if selected_api == "mistral-conversations":
            return MistralConversationsProvider(compatible_config)
        return OpenAICompatibleProvider(compatible_config)
    raise ProviderConfigError(f"Unsupported provider config: {provider.name}")


def _codex_reasoning_effort(
    provider: OpenAICodexProviderConfig,
    *,
    model: str | None,
    thinking_level: ThinkingLevel | None,
) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Map a thinking level to a Codex reasoning effort string.
    # Returns None if no level is given, the provider's thinking parameter is not
    # "reasoning.effort", or the model advertises no levels. Normalises the level; raises
    # ProviderConfigError listing available modes if it is not advertised. "off" -> None,
    # "minimal" -> "low", others -> the standard effort mapping.
    if thinking_level is None or provider.thinking_parameter != "reasoning.effort":
        return None
    levels = provider_thinking_levels(provider, model=model)
    if not levels:
        return None
    normalized = normalize_thinking_level(thinking_level)
    if normalized not in levels:
        selected_model = model or provider.default_model
        available = ", ".join(levels)
        raise ProviderConfigError(
            f"Thinking mode {normalized} is not available for "
            f"{provider.name}:{selected_model}. Available modes: {available}"
        )
    if normalized == "off":
        return None
    if normalized == "minimal":
        return "low"
    return reasoning_effort_for_level(normalized)


class OpenAICodexCredentialResolver:
    # [CLASS DOC]
    # Callable resolver producing OpenAI Codex credentials (access token + account id) for
    # each request, refreshing stored OAuth tokens when expired.
    """Resolve and refresh OpenAI Codex OAuth credentials for one request."""

    def __init__(
        self,
        provider: OpenAICodexProviderConfig,
        *,
        credential_store: FileCredentialStore,
    ) -> None:
        # [FUNCTION DOC]
        # Store the Codex provider config and the credential store. No I/O.
        self._provider = provider
        self._credential_store = credential_store

    async def __call__(self) -> OpenAICodexCredentials:
        # [FUNCTION DOC]
        # Return valid Codex credentials.
        # If the provider names a stored OAuth credential and it exists: refresh if expired,
        # then require an account id (RuntimeError if missing) and return it with the access
        # token. Otherwise, read the provider's API-key environment variable as an access JWT
        # and derive the account id from it (RuntimeError if not derivable). If neither source
        # is available, raises RuntimeError telling the user to run /login <provider>.
        # Side effects: may perform a network token refresh and write the credential store.
        """Return a valid Codex access token and account id."""
        credential_name = self._provider.credential_name
        if credential_name:
            credential = self._credential_store.get_oauth(credential_name)
            if credential is not None:
                credential = await self._refresh_if_needed(credential_name, credential)
                if credential.account_id is None:
                    raise RuntimeError("OpenAI Codex OAuth credential is missing account_id")
                return OpenAICodexCredentials(
                    access_token=credential.access,
                    account_id=credential.account_id,
                )

        access_token = environ.get(self._provider.api_key_env)
        if access_token:
            account_id = account_id_from_access_token(access_token)
            if account_id is None:
                raise RuntimeError(
                    f"{self._provider.api_key_env} must contain an OpenAI Codex access JWT"
                )
            return OpenAICodexCredentials(access_token=access_token, account_id=account_id)

        credential_hint = f"Run /login {self._provider.name}."
        raise RuntimeError(f"Missing OpenAI Codex OAuth credentials. {credential_hint}")

    async def _refresh_if_needed(
        self,
        credential_name: str,
        credential: OAuthCredential,
    ) -> OAuthCredential:
        # [FUNCTION DOC]
        # HELPER: Return the credential unchanged if not expired. Otherwise, under the shared
        # per-credential refresh lock, re-read the store (falling back to the given credential);
        # if the stored one is now valid return it; else refresh via the Codex token endpoint
        # using the stored refresh token, persist the result if it differs, and return it.
        # Network errors from refresh propagate.
        if not oauth_credential_is_expired(credential):
            return credential
        async with _refresh_lock(credential_name):
            stored = self._credential_store.get_oauth(credential_name) or credential
            if not oauth_credential_is_expired(stored):
                return stored
            refreshed = await refresh_openai_codex_token(stored.refresh)
            if refreshed != stored:
                self._credential_store.set_oauth(credential_name, refreshed)
        return refreshed


_REFRESH_LOCKS: MutableMapping[AbstractEventLoop, dict[str, asyncio.Lock]] = WeakKeyDictionary()


def _refresh_lock(credential_name: str) -> asyncio.Lock:
    # [FUNCTION DOC]
    # HELPER: Return the asyncio lock for one credential name, cached per running event loop
    # in a weak-keyed mapping (created on first use). Serialising refreshes ensures rotating
    # refresh tokens are spent at most once; per-loop caching avoids cross-loop lock errors.
    # Must be called with a running loop (RuntimeError otherwise).
    """Return this loop's refresh lock for one stored credential.

    Providers rotate the refresh token on use: the old one dies the moment a
    refresh succeeds. A session issues provider calls concurrently (the agent
    loop and session auto-naming, for two), so without serialization several
    tasks read the same expired credential and spend the same refresh token.
    One of them wins, the losers 400, and whichever write lands last can leave
    a superseded token on disk — which fails on the *next* run, long after the
    race that caused it. Holding this lock across the network call, and
    re-reading the store inside it, keeps a token spent at most once.

    Locks are cached per event loop because ``asyncio.Lock`` binds to the
    running loop on first contention: a lock cached across loops appears to
    work — the uncontended path never touches the loop — until two tasks
    contend it in a later loop and it raises.
    """
    locks = _REFRESH_LOCKS.setdefault(get_running_loop(), {})
    lock = locks.get(credential_name)
    if lock is None:
        lock = asyncio.Lock()
        locks[credential_name] = lock
    return lock


def _oauth_credential(
    provider: ProviderConfig,
    credential_store: FileCredentialStore,
) -> OAuthCredential | None:
    # [FUNCTION DOC]
    # HELPER: Return the stored OAuth credential for a provider config, or None when the
    # config has no credential name, no OAuth implementation is registered for the provider
    # name, or no OAuth credential is stored. Reads the credential store.
    if provider.credential_name is None or get_oauth_provider(provider.name) is None:
        return None
    return credential_store.get_oauth(provider.credential_name)


class OAuthRuntimeCredentialResolver:
    # [CLASS DOC]
    # Provider-neutral callable that refreshes OAuth credentials right before each request
    # and returns runtime auth (API key, base URL, headers).
    """Refresh provider-neutral OAuth credentials immediately before a request."""

    def __init__(
        self,
        provider: ProviderConfig,
        *,
        credential_store: FileCredentialStore,
    ) -> None:
        # [FUNCTION DOC]
        # Store the provider config and credential store. No I/O.
        self._provider = provider
        self._credential_store = credential_store

    async def __call__(self) -> RuntimeProviderAuth:
        # [FUNCTION DOC]
        # Resolve runtime auth. Raises RuntimeError if the config has no credential name or no
        # OAuth implementation is registered. Under the per-credential refresh lock, reads the
        # stored credential (RuntimeError suggesting /login if missing), calls the OAuth
        # provider's refresh (which may return the same credential when still valid), and
        # persists it if changed. Returns RuntimeProviderAuth built from the OAuth provider's
        # runtime auth for the (refreshed) credential. Side effects: network refresh, disk write.
        credential_name = self._provider.credential_name
        if credential_name is None:
            raise RuntimeError(f"Provider {self._provider.name} has no credential name")
        oauth_provider = _required_oauth_provider(self._provider.name)
        async with _refresh_lock(credential_name):
            # Read inside the lock: a task that waited here while another
            # refreshed sees the rotated credential and skips its own refresh.
            credential = self._credential_store.get_oauth(credential_name)
            if credential is None:
                raise RuntimeError(
                    f"Missing OAuth credentials for {self._provider.name}. "
                    f"Run /login {self._provider.name}."
                )
            refreshed = await oauth_provider.refresh(credential)
            if refreshed != credential:
                self._credential_store.set_oauth(credential_name, refreshed)
        auth = oauth_provider.runtime_auth(refreshed)
        return RuntimeProviderAuth(
            api_key=auth.api_key,
            base_url=auth.base_url,
            headers=auth.headers,
        )


def _required_oauth_provider(provider_name: str) -> OAuthProvider:
    # [FUNCTION DOC]
    # HELPER: Return the registered OAuth implementation for a provider name, raising
    # RuntimeError if none is registered.
    oauth_provider = get_oauth_provider(provider_name)
    if oauth_provider is None:
        raise RuntimeError(f"No OAuth implementation is registered for {provider_name}")
    return oauth_provider
