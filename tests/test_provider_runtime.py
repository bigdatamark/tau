# [MODULE DOC]
# Tests for tau_coding.provider_runtime: turning a provider configuration plus a
# file-backed credential store into a concrete tau_ai provider (Codex, Anthropic,
# OpenAI-compatible) with the right auth, image support, cache retention/affinity
# settings, model aliases, max tokens, base URL and reasoning-effort mapping.
# All tests use a temporary credentials file; they inspect the private provider
# config rather than making network calls.
from dataclasses import replace

import pytest

from tau_ai import AnthropicProvider, OpenAICodexProvider, OpenAICompatibleProvider
from tau_coding import provider_runtime
from tau_coding.credentials import FileCredentialStore, OAuthCredential
from tau_coding.provider_config import (
    AnthropicProviderConfig,
    OpenAICodexProviderConfig,
    OpenAICompatibleProviderConfig,
    ProviderConfigError,
    ProviderModelMetadata,
    provider_config_from_catalog_entry,
    resolve_startup_thinking_level,
)
from tau_coding.provider_runtime import OpenAICodexCredentialResolver, create_model_provider


def test_create_model_provider_returns_openai_codex_provider(tmp_path) -> None:
    # [FUNCTION DOC]
    # A default Codex config yields a Codex provider with a client-version resolver.
    store = FileCredentialStore(tmp_path / "credentials.json")

    provider = create_model_provider(
        OpenAICodexProviderConfig(),
        credential_store=store,
    )

    assert isinstance(provider, OpenAICodexProvider)
    assert provider._config.client_version_resolver is not None


def test_create_model_provider_uses_codex_model_image_capability(tmp_path) -> None:
    # [FUNCTION DOC]
    # For the catalog Codex config, a vision-capable model enables image support and a
    # text-only model disables it.
    store = FileCredentialStore(tmp_path / "credentials.json")
    config = provider_config_from_catalog_entry("openai-codex")

    vision_provider = create_model_provider(
        config,
        credential_store=store,
        model="gpt-5.6-sol",
    )
    text_provider = create_model_provider(
        config,
        credential_store=store,
        model="gpt-5.3-codex-spark",
    )

    assert isinstance(vision_provider, OpenAICodexProvider)
    assert isinstance(text_provider, OpenAICodexProvider)
    assert vision_provider._config.supports_images is True
    assert text_provider._config.supports_images is False


def test_direct_openai_runtime_enables_responses_cache_affinity(tmp_path) -> None:
    # [FUNCTION DOC]
    # The direct OpenAI catalog provider enables prompt cache keys and session-affinity
    # headers in OpenAI format.
    store = FileCredentialStore(tmp_path / "credentials.json")
    store.set_api_key("openai", "sk-test")

    provider = create_model_provider(
        provider_config_from_catalog_entry("openai"),
        credential_store=store,
        model="gpt-5.4",
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._config.compat["supportsPromptCacheKey"] is True
    assert provider._config.compat["sendSessionAffinityHeaders"] is True
    assert provider._config.compat["sessionAffinityFormat"] == "openai"


def test_huggingface_runtime_pins_backing_provider_with_model_alias(tmp_path) -> None:
    # [FUNCTION DOC]
    # Selecting an explicit Hugging Face inference provider aliases the model id to
    # "<model>:<provider>".
    store = FileCredentialStore(tmp_path / "credentials.json")
    store.set_api_key("huggingface", "hf-test")
    config = provider_config_from_catalog_entry("huggingface")

    provider = create_model_provider(
        config,
        credential_store=store,
        model="zai-org/GLM-5.2",
        inference_provider="deepinfra",
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._config.model_aliases == {"zai-org/GLM-5.2": "zai-org/GLM-5.2:deepinfra"}


def test_huggingface_runtime_rejects_policy_suffix(tmp_path) -> None:
    # [FUNCTION DOC]
    # A routing policy name such as "fastest" instead of an explicit inference provider
    # raises ProviderConfigError mentioning "explicit".
    store = FileCredentialStore(tmp_path / "credentials.json")
    store.set_api_key("huggingface", "hf-test")

    with pytest.raises(ProviderConfigError, match="explicit"):
        create_model_provider(
            provider_config_from_catalog_entry("huggingface"),
            credential_store=store,
            model="zai-org/GLM-5.2",
            inference_provider="fastest",
        )


def test_compatible_gateway_defaults_to_no_openai_cache_affinity(tmp_path) -> None:
    # [FUNCTION DOC]
    # Third-party OpenAI-compatible gateways (Together) disable prompt cache keys and
    # session-affinity headers by default.
    store = FileCredentialStore(tmp_path / "credentials.json")
    store.set_api_key("together", "gateway-key")

    provider = create_model_provider(
        provider_config_from_catalog_entry("together"),
        credential_store=store,
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._config.compat["supportsPromptCacheKey"] is False
    assert provider._config.compat["sendSessionAffinityHeaders"] is False


def test_create_model_provider_uses_anthropic_oauth_runtime_auth(tmp_path) -> None:
    # [FUNCTION DOC]
    # Anthropic OAuth credentials produce bearer auth with a credential resolver, an
    # OAuth system prompt, an Authorization header with the access token, and long
    # (1 hour) cache retention.
    store = FileCredentialStore(tmp_path / "credentials.json")
    store.set_oauth(
        "anthropic",
        OAuthCredential(
            access="anthropic-oauth-access",
            refresh="anthropic-refresh",
            expires=9999999999999,
        ),
    )

    provider = create_model_provider(AnthropicProviderConfig(), credential_store=store)

    assert isinstance(provider, AnthropicProvider)
    assert provider._config.bearer_auth is True
    assert provider._config.credential_resolver is not None
    assert provider._config.oauth_system_prompt is not None
    assert provider._config.headers is not None
    assert provider._config.headers["Authorization"] == "Bearer anthropic-oauth-access"
    # Subscription auth is not billed per token, so ask for the 1 hour cache TTL.
    assert provider._config.cache_retention == "long"


def test_anthropic_api_key_auth_keeps_the_default_cache_retention(tmp_path) -> None:
    # [FUNCTION DOC]
    # Anthropic API-key auth keeps short cache retention.
    """1h cache writes cost 2x base, so an API-key user must not get them silently."""
    store = FileCredentialStore(tmp_path / "credentials.json")
    store.set_api_key("anthropic", "sk-test")

    provider = create_model_provider(AnthropicProviderConfig(), credential_store=store)

    assert isinstance(provider, AnthropicProvider)
    assert provider._config.cache_retention == "short"


@pytest.mark.parametrize(
    "provider_name",
    ["minimax", "minimax-cn", "fireworks", "vercel-ai-gateway"],
)
def test_anthropic_protocol_gateways_disable_cache_breakpoints(
    provider_name: str,
    tmp_path,
) -> None:
    # [FUNCTION DOC]
    # Parametrized over gateways speaking the Anthropic protocol: cache retention is
    # "none" so no cache_control blocks are sent.
    """Gateways speaking the Anthropic protocol may reject cache_control blocks."""
    store = FileCredentialStore(tmp_path / "credentials.json")
    store.set_api_key(provider_name, "gateway-key")
    config = provider_config_from_catalog_entry(provider_name)

    provider = create_model_provider(config, credential_store=store)

    assert isinstance(provider, AnthropicProvider)
    assert provider._config.cache_retention == "none"


def test_provider_compat_can_re_enable_cache_breakpoints_on_a_gateway(tmp_path) -> None:
    # [FUNCTION DOC]
    # A compat override supportsCacheControl re-enables short retention and tool cache
    # breakpoints on a gateway.
    """A gateway proxying real Claude must be able to opt back in without a source edit."""
    store = FileCredentialStore(tmp_path / "credentials.json")
    store.set_api_key("minimax", "gateway-key")
    config = provider_config_from_catalog_entry("minimax")
    config = replace(config, compat={**config.compat, "supportsCacheControl": True})

    provider = create_model_provider(config, credential_store=store)

    assert isinstance(provider, AnthropicProvider)
    assert provider._config.cache_retention == "short"
    assert provider._config.cache_control_on_tools is True


def test_provider_compat_can_clamp_the_one_hour_ttl_on_oauth(tmp_path) -> None:
    # [FUNCTION DOC]
    # Compat supportsLongCacheRetention false downgrades OAuth retention to short.
    """The escape hatch if Anthropic ever stops honoring ttl=1h on subscriptions."""
    store = FileCredentialStore(tmp_path / "credentials.json")
    store.set_oauth(
        "anthropic",
        OAuthCredential(access="a", refresh="r", expires=9999999999999),
    )
    config = replace(AnthropicProviderConfig(), compat={"supportsLongCacheRetention": False})

    provider = create_model_provider(config, credential_store=store)

    assert isinstance(provider, AnthropicProvider)
    assert provider._config.cache_retention == "short"


def test_provider_compat_can_suppress_only_the_tools_breakpoint(tmp_path) -> None:
    # [FUNCTION DOC]
    # Compat supportsCacheControlOnTools false keeps short retention but disables the
    # tools cache breakpoint.
    """Some gateways accept cache_control everywhere except inside tool objects."""
    store = FileCredentialStore(tmp_path / "credentials.json")
    store.set_api_key("anthropic", "sk-test")
    config = replace(AnthropicProviderConfig(), compat={"supportsCacheControlOnTools": False})

    provider = create_model_provider(config, credential_store=store)

    assert isinstance(provider, AnthropicProvider)
    assert provider._config.cache_retention == "short"
    assert provider._config.cache_control_on_tools is False


def test_copilot_anthropic_protocol_models_disable_cache_breakpoints(tmp_path) -> None:
    # [FUNCTION DOC]
    # GitHub Copilot Claude models use the Anthropic provider with retention "none".
    """Copilot proxies the Anthropic protocol, so it gets no cache_control either."""
    store = FileCredentialStore(tmp_path / "credentials.json")
    store.set_oauth(
        "github-copilot",
        OAuthCredential(
            access="tid=1;proxy-ep=proxy.business.githubcopilot.com",
            refresh="github-token",
            expires=9999999999999,
        ),
    )

    provider = create_model_provider(
        provider_config_from_catalog_entry("github-copilot"),
        credential_store=store,
        model="claude-haiku-4.5",
    )

    assert isinstance(provider, AnthropicProvider)
    assert provider._config.cache_retention == "none"


def test_copilot_model_metadata_compat_can_enable_cache_breakpoints(tmp_path) -> None:
    # [FUNCTION DOC]
    # Per-model compat supportsCacheControl on a Copilot Claude model enables long
    # retention (Copilot uses OAuth/subscription auth).
    """Per-model compat reaches the openai-compatible Anthropic-protocol branch."""
    store = FileCredentialStore(tmp_path / "credentials.json")
    store.set_oauth(
        "github-copilot",
        OAuthCredential(
            access="tid=1;proxy-ep=proxy.business.githubcopilot.com",
            refresh="github-token",
            expires=9999999999999,
        ),
    )
    config = provider_config_from_catalog_entry("github-copilot")
    metadata = config.model_metadata["claude-haiku-4.5"]
    config = replace(
        config,
        model_metadata={
            **config.model_metadata,
            "claude-haiku-4.5": replace(
                metadata, compat={**metadata.compat, "supportsCacheControl": True}
            ),
        },
    )

    provider = create_model_provider(config, credential_store=store, model="claude-haiku-4.5")

    assert isinstance(provider, AnthropicProvider)
    assert provider._config.cache_retention == "long"


def test_create_model_provider_uses_model_max_tokens_for_anthropic_protocol_model(
    tmp_path,
) -> None:
    # [FUNCTION DOC]
    # Per-model max_tokens metadata is passed to the Anthropic-protocol provider.
    store = FileCredentialStore(tmp_path / "credentials.json")
    store.set_oauth(
        "github-copilot",
        OAuthCredential(
            access="tid=1",
            refresh="github-token",
            expires=9999999999999,
        ),
    )
    catalog_config = provider_config_from_catalog_entry("github-copilot")
    assert isinstance(catalog_config, OpenAICompatibleProviderConfig)
    metadata = dict(catalog_config.model_metadata)
    metadata["claude-haiku-4.5"] = replace(metadata["claude-haiku-4.5"], max_tokens=64_000)
    provider_config = replace(catalog_config, model_metadata=metadata)

    provider = create_model_provider(
        provider_config,
        credential_store=store,
        model="claude-haiku-4.5",
    )

    assert isinstance(provider, AnthropicProvider)
    assert provider._config.max_tokens == 64_000


def test_create_model_provider_uses_copilot_token_base_url(tmp_path) -> None:
    # [FUNCTION DOC]
    # The Copilot base URL is derived from the token's proxy endpoint ("proxy." becomes
    # "api.") and a credential resolver is installed.
    store = FileCredentialStore(tmp_path / "credentials.json")
    store.set_oauth(
        "github-copilot",
        OAuthCredential(
            access="tid=1;proxy-ep=proxy.business.githubcopilot.com",
            refresh="github-token",
            expires=9999999999999,
        ),
    )
    provider = create_model_provider(
        provider_config_from_catalog_entry("github-copilot"),
        credential_store=store,
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._config.base_url == "https://api.business.githubcopilot.com"
    assert provider._config.credential_resolver is not None


def test_create_model_provider_rejects_model_not_declared_for_provider(tmp_path) -> None:
    # [FUNCTION DOC]
    # Requesting a model not listed for a provider raises ProviderConfigError with the
    # provider and model names.
    store = FileCredentialStore(tmp_path / "credentials.json")
    provider_config = OpenAICompatibleProviderConfig(
        name="local",
        models=("qwen",),
        default_model="qwen",
    )

    with pytest.raises(
        ProviderConfigError,
        match="Model is not configured for provider local: llama",
    ):
        create_model_provider(provider_config, credential_store=store, model="llama")


def test_create_model_provider_maps_codex_reasoning_effort_like_pi(tmp_path) -> None:
    # [FUNCTION DOC]
    # Codex thinking levels map like Pi: off means no reasoning effort, minimal maps to
    # "low", xhigh stays "xhigh".
    store = FileCredentialStore(tmp_path / "credentials.json")
    provider_config = OpenAICodexProviderConfig(
        thinking_levels=("off", "minimal", "low", "medium", "high", "xhigh"),
        thinking_models=("gpt-5.5",),
        thinking_parameter="reasoning.effort",
    )

    off_provider = create_model_provider(
        provider_config,
        credential_store=store,
        model="gpt-5.5",
        thinking_level="off",
    )
    minimal_provider = create_model_provider(
        provider_config,
        credential_store=store,
        model="gpt-5.5",
        thinking_level="minimal",
    )
    xhigh_provider = create_model_provider(
        provider_config,
        credential_store=store,
        model="gpt-5.5",
        thinking_level="xhigh",
    )

    assert isinstance(off_provider, OpenAICodexProvider)
    assert isinstance(minimal_provider, OpenAICodexProvider)
    assert isinstance(xhigh_provider, OpenAICodexProvider)
    assert off_provider._config.reasoning_effort is None
    assert minimal_provider._config.reasoning_effort == "low"
    assert xhigh_provider._config.reasoning_effort == "xhigh"


def test_create_model_provider_coerces_unsupported_startup_thinking_level(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    # Regression: startup used to pass the global default ("medium") straight
    # to create_model_provider, which crashed for models like kimi-code:k3
    # that only support xhigh. Now k3 also supports low and high.
    # [FUNCTION DOC]
    # Regression test: an unsupported thinking level ("medium") for a model raises
    # ProviderConfigError, while the startup-resolved level (the provider's default)
    # works and maps to the model's provider-specific value ("max").
    # Side effects: sets a test API key environment variable via monkeypatch.
    monkeypatch.setenv("TAU_TEST_KIMI_CODE_API_KEY", "test-key")
    store = FileCredentialStore(tmp_path / "credentials.json")
    provider_config = OpenAICompatibleProviderConfig(
        name="kimi-code",
        api_key_env="TAU_TEST_KIMI_CODE_API_KEY",
        models=("k3",),
        default_model="k3",
        thinking_levels=("low", "medium", "high", "xhigh"),
        thinking_default="xhigh",
        thinking_parameter="reasoning_effort",
        model_metadata={
            "k3": ProviderModelMetadata(
                reasoning=True,
                thinking_level_map={
                    "off": None,
                    "minimal": None,
                    "low": "low",
                    "medium": None,
                    "high": "high",
                    "xhigh": "max",
                },
            ),
        },
    )

    with pytest.raises(
        ProviderConfigError,
        match="Thinking mode medium is not available for kimi-code:k3",
    ):
        create_model_provider(
            provider_config,
            credential_store=store,
            model="k3",
            thinking_level="medium",
        )

    provider = create_model_provider(
        provider_config,
        credential_store=store,
        model="k3",
        thinking_level=resolve_startup_thinking_level(provider_config, "k3"),
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._config.reasoning_effort == "max"


@pytest.mark.anyio
async def test_openai_codex_credential_resolver_refreshes_expired_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    # [FUNCTION DOC]
    # With expired stored Codex OAuth credentials, the resolver calls the (patched)
    # refresh function with the stored refresh token, returns the new access token and
    # account id, and persists the refreshed credential in the store.
    store = FileCredentialStore(tmp_path / "credentials.json")
    store.set_oauth(
        "openai-codex",
        OAuthCredential(
            access="old-access",
            refresh="old-refresh",
            expires=1,
            account_id="old-account",
        ),
    )

    async def fake_refresh(refresh_token: str) -> OAuthCredential:
        # [FUNCTION DOC]
        # HELPER: Stand-in token refresh; asserts the old refresh token was passed and
        # returns a fresh long-lived credential.
        assert refresh_token == "old-refresh"
        return OAuthCredential(
            access="new-access",
            refresh="new-refresh",
            expires=9999999999999,
            account_id="new-account",
        )

    monkeypatch.setattr(provider_runtime, "refresh_openai_codex_token", fake_refresh)

    resolver = OpenAICodexCredentialResolver(
        OpenAICodexProviderConfig(),
        credential_store=store,
    )

    credentials = await resolver()

    assert credentials.access_token == "new-access"
    assert credentials.account_id == "new-account"
    assert store.get_oauth("openai-codex") == OAuthCredential(
        access="new-access",
        refresh="new-refresh",
        expires=9999999999999,
        account_id="new-account",
    )
