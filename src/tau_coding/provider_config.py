# [MODULE DOC]
# Durable provider configuration for coding sessions. Defines immutable,
# self-validating provider config types (OpenAI-compatible, Anthropic, OpenAI
# Codex) with per-model metadata, the persisted provider settings (default
# provider, per-provider runtime preferences, scoped models for quick
# cycling) and their JSON schema (version 2, providers.json in the Tau home).
# Provider capabilities (models, URLs, metadata) come from the effective
# catalog (built-in plus user catalog.toml); providers.json only stores
# preferences, and legacy full-record files are migrated (with a .bak backup).
# Also resolves provider/model selection from CLI/env/settings, thinking-level
# availability and defaults, credential availability, and builds runtime
# tau_ai configs (API key lookup from credentials or environment, OAuth
# handling, headers, cache retention). Raises ProviderConfigError for invalid
# configuration.
"""Durable provider configuration for Tau coding sessions."""

from __future__ import annotations

import re
from contextlib import suppress
from dataclasses import dataclass, field, replace
from json import dumps, loads
from os import environ
from pathlib import Path
from shutil import copy2
from tempfile import NamedTemporaryFile
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

from tau_ai.env import (
    CACHE_RETENTION_LONG,
    CACHE_RETENTION_NONE,
    CACHE_RETENTION_SHORT,
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_OPENAI_COMPATIBLE_BASE_URL,
    DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES,
    DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS,
    DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
    AnthropicConfig,
    CacheRetention,
    OpenAICompatibleConfig,
)
from tau_ai.openai_codex import DEFAULT_OPENAI_CODEX_BASE_URL
from tau_coding.catalog_loader import effective_catalog, save_user_catalog_entries
from tau_coding.credentials import FileCredentialStore, credentials_path
from tau_coding.oauth_registry import get_oauth_provider
from tau_coding.paths import TauPaths
from tau_coding.provider_catalog import (
    BUILTIN_PROVIDER_CATALOG,
    ModelCatalogMetadata,
    ModelCostTier,
    ProviderApi,
    ProviderCatalogEntry,
    ProviderKind,
)
from tau_coding.thinking import (
    DEFAULT_THINKING_LEVEL,
    THINKING_LEVELS,
    ThinkingLevel,
    ThinkingParameter,
    anthropic_thinking_budget_for_level,
    normalize_thinking_level,
    normalize_thinking_levels,
    reasoning_effort_for_level,
)

DEFAULT_PROVIDER_NAME = "openai"
DEFAULT_MODEL = "gpt-5.4"
PROVIDER_SETTINGS_SCHEMA_VERSION = 2


class ProviderConfigError(ValueError):
    # [CLASS DOC]
    # Exception (ValueError subclass) raised for invalid provider configuration,
    # unknown providers/models, or unusable settings.
    """Raised when Tau provider configuration is invalid."""


class CredentialReader(Protocol):
    # [CLASS DOC]
    # Structural protocol for a credential lookup with a get-by-name method.
    """Credential lookup used while building runtime provider config."""

    def get(self, name: str) -> str | None: ...
                                            # [FUNCTION DOC]
                                            # Protocol method: return the stored secret for a name, or none.


@dataclass(frozen=True, slots=True)
class ProviderModelMetadata:
    # [CLASS DOC]
    # Immutable runtime metadata for one model: display name, API override, base
    # URL override, reasoning flag, input modalities, flat cost rates, cost tiers,
    # context window, max tokens, extra headers, compat options and a map from
    # thinking level to provider value (none meaning unsupported).
    """Runtime metadata for one configured model."""

    name: str | None = None
    api: ProviderApi | None = None
    base_url: str | None = None
    reasoning: bool | None = None
    input: tuple[str, ...] = ()
    cost: dict[str, float] = field(default_factory=dict)
    cost_tiers: tuple[ModelCostTier, ...] = ()
    context_window: int | None = None
    max_tokens: int | None = None
    headers: dict[str, str] = field(default_factory=dict)
    compat: dict[str, Any] = field(default_factory=dict)
    thinking_level_map: dict[ThinkingLevel, str | None] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Serializes all fields to a JSON-compatible mapping; cost tiers become
        # mappings of optional max_input_tokens plus rates. No side effects.
        """Serialize this model metadata to JSON-compatible data."""
        return {
            "name": self.name,
            "api": self.api,
            "base_url": self.base_url,
            "reasoning": self.reasoning,
            "input": list(self.input),
            "cost": dict(self.cost),
            "cost_tiers": [
                {
                    **(
                        {"max_input_tokens": tier.max_input_tokens}
                        if tier.max_input_tokens is not None
                        else {}
                    ),
                    **tier.cost,
                }
                for tier in self.cost_tiers
            ],
            "context_window": self.context_window,
            "max_tokens": self.max_tokens,
            "headers": dict(self.headers),
            "compat": dict(self.compat),
            "thinking_level_map": dict(self.thinking_level_map),
        }


@dataclass(frozen=True, slots=True)
class OpenAICompatibleProviderConfig:
    # [CLASS DOC]
    # Immutable durable config for an OpenAI-compatible provider (chat
    # completions, responses, Google or Mistral APIs): name, base URL, API, API
    # key env var, credential name, models, default model, context windows,
    # headers, compat, model metadata, timeout/retry settings, thinking
    # configuration (levels, thinking models, default, parameter), remembered
    # per-model thinking defaults and per-model inference provider routes.
    # Validated on construction.
    """Durable settings for one OpenAI-compatible provider."""

    name: str
    base_url: str = DEFAULT_OPENAI_COMPATIBLE_BASE_URL
    api: ProviderApi = "openai-completions"
    api_key_env: str = "OPENAI_API_KEY"
    credential_name: str | None = None
    models: tuple[str, ...] = (DEFAULT_MODEL,)
    default_model: str = DEFAULT_MODEL
    context_windows: dict[str, int] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    compat: dict[str, Any] = field(default_factory=dict)
    model_metadata: dict[str, ProviderModelMetadata] = field(default_factory=dict)
    timeout_seconds: float = DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES
    max_retry_delay_seconds: float = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS
    thinking_levels: tuple[ThinkingLevel, ...] | None = None
    thinking_models: tuple[str, ...] = ()
    thinking_default: ThinkingLevel | None = None
    thinking_parameter: ThinkingParameter | None = None
    thinking_defaults: dict[str, ThinkingLevel] = field(default_factory=dict)
    inference_providers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validates numbers (timeout, retries, retry delay), context windows, model
        # metadata keys, compat JSON, thinking configuration, thinking defaults and
        # inference provider routes. Raises ProviderConfigError on any violation.
        _validate_provider_numbers(
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            max_retry_delay_seconds=self.max_retry_delay_seconds,
        )
        _validate_context_windows(self.context_windows)
        _validate_model_metadata(self.models, self.model_metadata)
        _validate_json_object(self.compat, "Provider compat")
        _validate_thinking_config(
            thinking_levels=self.thinking_levels,
            thinking_models=self.thinking_models,
            thinking_default=self.thinking_default,
            thinking_parameter=self.thinking_parameter,
        )
        _validate_thinking_defaults(self.thinking_defaults)
        _validate_inference_providers(self.name, self.models, self.inference_providers)

    def to_json(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Serializes the full config with type "openai-compatible" to a JSON-ready
        # mapping (lists for tuples, copied mappings).
        """Serialize this provider config to JSON-compatible data."""
        return {
            "name": self.name,
            "type": "openai-compatible",
            "base_url": self.base_url,
            "api": self.api,
            "api_key_env": self.api_key_env,
            "credential_name": self.credential_name,
            "models": list(self.models),
            "default_model": self.default_model,
            "context_windows": dict(self.context_windows),
            "headers": dict(self.headers),
            "compat": dict(self.compat),
            "model_metadata": {
                model: metadata.to_json() for model, metadata in self.model_metadata.items()
            },
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_retry_delay_seconds": self.max_retry_delay_seconds,
            "thinking_levels": (
                list(self.thinking_levels) if self.thinking_levels is not None else None
            ),
            "thinking_models": list(self.thinking_models),
            "thinking_default": self.thinking_default,
            "thinking_parameter": self.thinking_parameter,
            "thinking_defaults": dict(self.thinking_defaults),
            "inference_providers": dict(self.inference_providers),
        }


@dataclass(frozen=True, slots=True)
class AnthropicProviderConfig:
    # [CLASS DOC]
    # Immutable durable config for Anthropic Messages API providers; same fields
    # as the OpenAI-compatible config minus inference provider routes, with
    # Anthropic defaults (name anthropic, ANTHROPIC_API_KEY, credential name
    # anthropic, default model claude-sonnet-4-6). Validated on construction.
    """Durable settings for Anthropic's Messages API."""

    name: str = "anthropic"
    base_url: str = DEFAULT_ANTHROPIC_BASE_URL
    api: ProviderApi = "anthropic-messages"
    api_key_env: str = "ANTHROPIC_API_KEY"
    credential_name: str | None = "anthropic"
    models: tuple[str, ...] = ("claude-sonnet-4-6",)
    default_model: str = "claude-sonnet-4-6"
    context_windows: dict[str, int] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    compat: dict[str, Any] = field(default_factory=dict)
    model_metadata: dict[str, ProviderModelMetadata] = field(default_factory=dict)
    timeout_seconds: float = DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES
    max_retry_delay_seconds: float = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS
    thinking_levels: tuple[ThinkingLevel, ...] | None = None
    thinking_models: tuple[str, ...] = ()
    thinking_default: ThinkingLevel | None = None
    thinking_parameter: ThinkingParameter | None = None
    thinking_defaults: dict[str, ThinkingLevel] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validates numbers, context windows, model metadata, compat, thinking
        # configuration and thinking defaults; raises ProviderConfigError.
        _validate_provider_numbers(
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            max_retry_delay_seconds=self.max_retry_delay_seconds,
        )
        _validate_context_windows(self.context_windows)
        _validate_model_metadata(self.models, self.model_metadata)
        _validate_json_object(self.compat, "Provider compat")
        _validate_thinking_config(
            thinking_levels=self.thinking_levels,
            thinking_models=self.thinking_models,
            thinking_default=self.thinking_default,
            thinking_parameter=self.thinking_parameter,
        )
        _validate_thinking_defaults(self.thinking_defaults)

    def to_json(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Serializes the full config with type "anthropic".
        """Serialize this provider config to JSON-compatible data."""
        return {
            "name": self.name,
            "type": "anthropic",
            "base_url": self.base_url,
            "api": self.api,
            "api_key_env": self.api_key_env,
            "credential_name": self.credential_name,
            "models": list(self.models),
            "default_model": self.default_model,
            "context_windows": dict(self.context_windows),
            "headers": dict(self.headers),
            "compat": dict(self.compat),
            "model_metadata": {
                model: metadata.to_json() for model, metadata in self.model_metadata.items()
            },
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_retry_delay_seconds": self.max_retry_delay_seconds,
            "thinking_levels": (
                list(self.thinking_levels) if self.thinking_levels is not None else None
            ),
            "thinking_models": list(self.thinking_models),
            "thinking_default": self.thinking_default,
            "thinking_parameter": self.thinking_parameter,
            "thinking_defaults": dict(self.thinking_defaults),
        }


@dataclass(frozen=True, slots=True)
class OpenAICodexProviderConfig:
    # [CLASS DOC]
    # Immutable durable config for the OpenAI Codex subscription (OAuth)
    # provider: no API or compat fields; defaults to the Codex base URL, access
    # token env var, credential name openai-codex and a fixed GPT-5.x model list
    # with default gpt-5.5. Validated on construction.
    """Durable settings for OpenAI Codex subscription OAuth."""

    name: str = "openai-codex"
    base_url: str = DEFAULT_OPENAI_CODEX_BASE_URL
    api_key_env: str = "OPENAI_CODEX_ACCESS_TOKEN"
    credential_name: str | None = "openai-codex"
    models: tuple[str, ...] = (
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.3-codex",
        "gpt-5.3-codex-spark",
        "gpt-5.2",
    )
    default_model: str = "gpt-5.5"
    context_windows: dict[str, int] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    model_metadata: dict[str, ProviderModelMetadata] = field(default_factory=dict)
    timeout_seconds: float = DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES
    max_retry_delay_seconds: float = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS
    thinking_levels: tuple[ThinkingLevel, ...] | None = None
    thinking_models: tuple[str, ...] = ()
    thinking_default: ThinkingLevel | None = None
    thinking_parameter: ThinkingParameter | None = None
    thinking_defaults: dict[str, ThinkingLevel] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validates numbers, context windows, model metadata, thinking configuration
        # and thinking defaults; raises ProviderConfigError.
        _validate_provider_numbers(
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            max_retry_delay_seconds=self.max_retry_delay_seconds,
        )
        _validate_context_windows(self.context_windows)
        _validate_model_metadata(self.models, self.model_metadata)
        _validate_thinking_config(
            thinking_levels=self.thinking_levels,
            thinking_models=self.thinking_models,
            thinking_default=self.thinking_default,
            thinking_parameter=self.thinking_parameter,
        )
        _validate_thinking_defaults(self.thinking_defaults)

    def to_json(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Serializes the full config with type "openai-codex".
        """Serialize this provider config to JSON-compatible data."""
        return {
            "name": self.name,
            "type": "openai-codex",
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "credential_name": self.credential_name,
            "models": list(self.models),
            "default_model": self.default_model,
            "context_windows": dict(self.context_windows),
            "headers": dict(self.headers),
            "model_metadata": {
                model: metadata.to_json() for model, metadata in self.model_metadata.items()
            },
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_retry_delay_seconds": self.max_retry_delay_seconds,
            "thinking_levels": (
                list(self.thinking_levels) if self.thinking_levels is not None else None
            ),
            "thinking_models": list(self.thinking_models),
            "thinking_default": self.thinking_default,
            "thinking_parameter": self.thinking_parameter,
            "thinking_defaults": dict(self.thinking_defaults),
        }


type ProviderConfig = (
    OpenAICompatibleProviderConfig | AnthropicProviderConfig | OpenAICodexProviderConfig
)


@dataclass(frozen=True, slots=True)
class ScopedModelConfig:
    # [CLASS DOC]
    # Immutable provider/model pair enabled for quick model cycling.
    """A provider/model pair enabled for quick model cycling."""

    provider: str
    model: str

    def to_json(self) -> dict[str, str]:
        # [FUNCTION DOC]
        # Returns a mapping with provider and model.
        """Serialize this scoped model reference."""
        return {"provider": self.provider, "model": self.model}


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    # [CLASS DOC]
    # Immutable provider settings: default provider name (default "openai"),
    # configured providers (default the built-in configs) and scoped models.
    """Tau provider settings loaded from Tau home."""

    default_provider: str = DEFAULT_PROVIDER_NAME
    providers: tuple[ProviderConfig, ...] = field(
        default_factory=lambda: builtin_provider_configs()
    )
    scoped_models: tuple[ScopedModelConfig, ...] = ()

    def get_provider(self, name: str | None = None) -> ProviderConfig:
        # [FUNCTION DOC]
        # Returns the provider with the given name (default the default provider).
        # Raises ProviderConfigError("Unknown provider") when absent.
        """Return a configured provider by name."""
        target = name or self.default_provider
        for provider in self.providers:
            if provider.name == target:
                return provider
        raise ProviderConfigError(f"Unknown provider: {target}")

    def to_json(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Serializes runtime preferences only: schema version 2, default provider,
        # per-provider preference mappings and scoped models. Full definitions are
        # not included (they live in the catalog).
        """Serialize runtime preferences to JSON-compatible data."""
        return {
            "schema_version": PROVIDER_SETTINGS_SCHEMA_VERSION,
            "default_provider": self.default_provider,
            "provider_preferences": {
                provider.name: _provider_preference_to_json(provider) for provider in self.providers
            },
            "scoped_models": [model.to_json() for model in self.scoped_models],
        }


@dataclass(frozen=True, slots=True)
class ProviderSelection:
    # [CLASS DOC]
    # Immutable resolved selection of a provider config and model for a run.
    """Resolved provider/model selection for a Tau run."""

    provider: ProviderConfig
    model: str


def builtin_provider_configs() -> tuple[ProviderConfig, ...]:
    # [FUNCTION DOC]
    # Returns provider configs for every entry of the built-in provider catalog,
    # in catalog order.
    """Return Tau's built-in provider configs."""
    return tuple(
        provider_config_from_catalog_entry(entry.name) for entry in BUILTIN_PROVIDER_CATALOG
    )


def provider_config_from_catalog_entry(name: str) -> ProviderConfig:
    # [FUNCTION DOC]
    # Returns the config for the named built-in catalog entry; raises
    # ProviderConfigError for unknown names.
    """Create a durable provider config from a built-in catalog entry."""
    for entry in BUILTIN_PROVIDER_CATALOG:
        if entry.name == name:
            return provider_config_from_entry(entry)
    raise ProviderConfigError(f"Unknown built-in provider: {name}")


def provider_config_from_entry(entry: ProviderCatalogEntry) -> ProviderConfig:
    # [FUNCTION DOC]
    # Converts a catalog entry into the matching config type by kind: anthropic
    # entries become Anthropic configs (API forced to anthropic-messages),
    # openai-codex entries become Codex configs, everything else becomes an
    # OpenAI-compatible config (entry API or kind default). Copies models,
    # headers, compat, metadata and thinking settings with empty thinking
    # defaults. Raises ProviderConfigError if the result fails validation.
    """Create a durable provider config from a catalog entry."""
    context_windows = dict(entry.context_windows or {})
    model_metadata = _provider_model_metadata_from_catalog(entry.model_metadata)
    if entry.kind == "anthropic":
        return AnthropicProviderConfig(
            name=entry.name,
            base_url=entry.base_url,
            api=_default_api_for_kind(entry.kind),
            api_key_env=entry.api_key_env,
            credential_name=entry.credential_name,
            models=entry.models,
            default_model=entry.default_model,
            context_windows=context_windows,
            headers=dict(entry.headers),
            compat=dict(entry.compat),
            model_metadata=model_metadata,
            thinking_levels=entry.thinking_levels,
            thinking_models=entry.thinking_models,
            thinking_default=entry.thinking_default,
            thinking_parameter=entry.thinking_parameter,
            thinking_defaults={},
        )
    if entry.kind == "openai-codex":
        return OpenAICodexProviderConfig(
            name=entry.name,
            base_url=entry.base_url,
            api_key_env=entry.api_key_env,
            credential_name=entry.credential_name,
            models=entry.models,
            default_model=entry.default_model,
            context_windows=context_windows,
            headers=dict(entry.headers),
            model_metadata=model_metadata,
            thinking_levels=entry.thinking_levels,
            thinking_models=entry.thinking_models,
            thinking_default=entry.thinking_default,
            thinking_parameter=entry.thinking_parameter,
            thinking_defaults={},
        )
    return OpenAICompatibleProviderConfig(
        name=entry.name,
        base_url=entry.base_url,
        api=entry.api or _default_api_for_kind(entry.kind),
        api_key_env=entry.api_key_env,
        credential_name=entry.credential_name,
        models=entry.models,
        default_model=entry.default_model,
        context_windows=context_windows,
        headers=dict(entry.headers),
        compat=dict(entry.compat),
        model_metadata=model_metadata,
        thinking_levels=entry.thinking_levels,
        thinking_models=entry.thinking_models,
        thinking_default=entry.thinking_default,
        thinking_parameter=entry.thinking_parameter,
        thinking_defaults={},
    )


def _default_api_for_kind(kind: str) -> ProviderApi:
    # [FUNCTION DOC]
    # HELPER: maps a provider kind to its default API name (anthropic-messages,
    # openai-codex-responses, google-generative-ai, mistral-conversations, else
    # openai-completions).
    if kind == "anthropic":
        return "anthropic-messages"
    if kind == "openai-codex":
        return "openai-codex-responses"
    if kind == "google-generative-ai":
        return "google-generative-ai"
    if kind == "mistral-conversations":
        return "mistral-conversations"
    return "openai-completions"


def _provider_model_metadata_from_catalog(
    model_metadata: dict[str, ModelCatalogMetadata],
) -> dict[str, ProviderModelMetadata]:
    # [FUNCTION DOC]
    # HELPER: converts catalog model metadata mappings into runtime model
    # metadata, copying collections.
    return {
        model: ProviderModelMetadata(
            name=metadata.name,
            api=metadata.api,
            base_url=metadata.base_url,
            reasoning=metadata.reasoning,
            input=tuple(metadata.input),
            cost=dict(metadata.cost or {}),
            cost_tiers=metadata.cost_tiers,
            context_window=metadata.context_window,
            max_tokens=metadata.max_tokens,
            headers=dict(metadata.headers),
            compat=dict(metadata.compat),
            thinking_level_map=dict(metadata.thinking_level_map),
        )
        for model, metadata in model_metadata.items()
    }


def default_openai_provider_config() -> OpenAICompatibleProviderConfig:
    # [FUNCTION DOC]
    # Returns the built-in "openai" provider config. Raises AssertionError if it
    # is not an OpenAI-compatible config.
    """Return Tau's default OpenAI-compatible provider entry."""
    provider = provider_config_from_catalog_entry(DEFAULT_PROVIDER_NAME)
    if not isinstance(provider, OpenAICompatibleProviderConfig):
        raise AssertionError("default OpenAI provider must be OpenAI-compatible")
    return provider


def provider_settings_path(paths: TauPaths | None = None) -> Path:
    # [FUNCTION DOC]
    # Returns the providers.json path in the Tau home. No I/O.
    """Return the durable provider settings path."""
    return (paths or TauPaths()).home / "providers.json"


def load_provider_settings(paths: TauPaths | None = None) -> ProviderSettings:
    # [FUNCTION DOC]
    # Loads settings. Without a file, returns defaults with providers from the
    # effective catalog (user catalog providers always, built-ins only when
    # credentialed via the append rule). With a file: parses JSON (must be an
    # object), builds settings; legacy files (no provider_preferences key) are
    # migrated onto catalog definitions and immediately saved with a strict
    # backup; otherwise preferences are merged with the current catalog.
    # Raises ProviderConfigError for invalid content; JSON decode and OS errors
    # propagate. Side effects: reads files; may write providers.json, its .bak
    # and the user catalog during migration.
    """Load durable provider settings, falling back to env-compatible defaults."""
    resolved_paths = paths or TauPaths()
    path = provider_settings_path(resolved_paths)
    if not path.exists():
        return ProviderSettings(providers=_effective_provider_configs(resolved_paths))
    raw = loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ProviderConfigError("Provider settings must be a JSON object")
    settings = provider_settings_from_json(raw, paths=resolved_paths)
    if "provider_preferences" not in raw:
        settings = _migrate_legacy_provider_settings(settings, paths=resolved_paths)
        _save_migrated_provider_settings(settings, paths=resolved_paths)
        return settings
    return _with_builtin_catalog_models(settings, paths=resolved_paths)


def save_provider_settings(settings: ProviderSettings, paths: TauPaths | None = None) -> Path:
    # [FUNCTION DOC]
    # Persists settings: upserts non-built-in provider definitions into the user
    # catalog, then atomically writes preferences JSON (after a best-effort
    # backup of the previous file). Output: the settings path. Side effects:
    # file writes.
    """Write durable provider preferences and return the path."""
    resolved_paths = paths or TauPaths()
    _save_provider_definitions_to_catalog(settings, paths=resolved_paths)
    path = provider_settings_path(resolved_paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_provider_settings(settings, path=path, backup=True)
    return path


def save_default_provider_model(
    *,
    provider_name: str,
    model: str,
    paths: TauPaths | None = None,
    fallback_settings: ProviderSettings | None = None,
) -> ProviderSettings:
    # [FUNCTION DOC]
    # Reloads the latest settings (or the fallback when no file exists), sets the
    # default provider/model, saves and returns the updated settings. Raises
    # ProviderConfigError for unknown provider or model.
    """Reload settings, persist one default provider/model change, and return them."""
    settings = _load_provider_settings_for_write(paths, fallback_settings=fallback_settings)
    updated = set_default_provider_model(settings, provider_name=provider_name, model=model)
    save_provider_settings(updated, paths)
    return updated


def save_provider_thinking_level(
    *,
    provider_name: str,
    model: str,
    thinking_level: ThinkingLevel,
    paths: TauPaths | None = None,
    fallback_settings: ProviderSettings | None = None,
) -> ProviderSettings:
    # [FUNCTION DOC]
    # Reloads settings, remembers the thinking level for one provider/model,
    # saves and returns them. Raises ProviderConfigError if the level is not
    # available for that model.
    """Reload settings, persist one provider/model thinking preference, and return them."""
    settings = _load_provider_settings_for_write(paths, fallback_settings=fallback_settings)
    updated = set_provider_thinking_level(
        settings,
        provider_name=provider_name,
        model=model,
        thinking_level=thinking_level,
    )
    save_provider_settings(updated, paths)
    return updated


def toggle_saved_stable_scoped_model(
    *,
    provider_name: str,
    model: str,
    paths: TauPaths | None = None,
    fallback_settings: ProviderSettings | None = None,
) -> ProviderSettings:
    # [FUNCTION DOC]
    # Reloads settings and toggles a provider/model pair in scoped models without
    # validating that the model is configured (for trusted dynamic providers),
    # then saves and returns them.
    """Toggle an already-authorized stable reference without saving a definition.

    Callers must restrict this path to trusted built-in dynamic providers.  The
    durable value is only the exact provider/model pair; availability continues
    to come from the process-local provider snapshot.
    """
    settings = _load_provider_settings_for_write(paths, fallback_settings=fallback_settings)
    target = ScopedModelConfig(provider=provider_name, model=model)
    existing = list(settings.scoped_models)
    if target in existing:
        existing = [item for item in existing if item != target]
    else:
        existing.append(target)
    updated = replace(settings, scoped_models=tuple(existing))
    save_provider_settings(updated, paths)
    return updated


def toggle_saved_scoped_model(
    *,
    provider_name: str,
    model: str,
    paths: TauPaths | None = None,
    fallback_settings: ProviderSettings | None = None,
) -> ProviderSettings:
    # [FUNCTION DOC]
    # Reloads settings and toggles a scoped provider/model pair (added at the end
    # or removed), saves and returns them. Raises ProviderConfigError if the
    # provider is unknown or the model is not configured for it.
    """Reload settings, toggle one scoped model, persist them, and return them."""
    settings = _load_provider_settings_for_write(paths, fallback_settings=fallback_settings)
    provider = settings.get_provider(provider_name)
    if model not in provider.models:
        raise ProviderConfigError(f"Model is not configured: {provider_name}:{model}")

    existing = list(settings.scoped_models)
    target = ScopedModelConfig(provider=provider_name, model=model)
    if target in existing:
        existing = [item for item in existing if item != target]
    else:
        existing.append(target)
    updated = replace(settings, scoped_models=tuple(existing))
    save_provider_settings(updated, paths)
    return updated


def upsert_saved_provider(
    provider: ProviderConfig,
    *,
    set_default: bool = False,
    paths: TauPaths | None = None,
    fallback_settings: ProviderSettings | None = None,
) -> ProviderSettings:
    # [FUNCTION DOC]
    # Reloads settings, adds or replaces one provider (optionally making it the
    # default), saves and returns the result.
    """Reload settings, upsert one provider entry, persist them, and return them."""
    settings = _load_provider_settings_for_write(paths, fallback_settings=fallback_settings)
    updated = upsert_provider(settings, provider, set_default=set_default)
    save_provider_settings(updated, paths)
    return updated


def _load_provider_settings_for_write(
    paths: TauPaths | None,
    *,
    fallback_settings: ProviderSettings | None = None,
) -> ProviderSettings:
    # [FUNCTION DOC]
    # HELPER: loads the on-disk settings when the file exists; otherwise returns
    # the fallback settings if given, else freshly loaded defaults.
    """Load the latest on-disk settings, falling back only when no file exists."""
    resolved_paths = paths or TauPaths()
    if provider_settings_path(resolved_paths).exists():
        return load_provider_settings(resolved_paths)
    if fallback_settings is not None:
        return fallback_settings
    return load_provider_settings(resolved_paths)


def set_default_provider_model(
    settings: ProviderSettings,
    *,
    provider_name: str,
    model: str,
) -> ProviderSettings:
    # [FUNCTION DOC]
    # Pure: returns settings with the given provider as default and its default
    # model changed. Raises ProviderConfigError for unknown provider/model.
    """Return settings with the default provider/model preference updated."""
    provider = settings.get_provider(provider_name)
    validate_provider_model(provider, model)
    updated_provider = replace(provider, default_model=model)
    providers = tuple(
        updated_provider if item.name == provider_name else item for item in settings.providers
    )
    return ProviderSettings(
        default_provider=provider_name,
        providers=providers,
        scoped_models=settings.scoped_models,
    )


def set_provider_thinking_level(
    settings: ProviderSettings,
    *,
    provider_name: str,
    model: str,
    thinking_level: ThinkingLevel,
) -> ProviderSettings:
    # [FUNCTION DOC]
    # Pure: normalizes the level and verifies it is available for the provider and
    # model, then returns settings with that model's remembered thinking default
    # updated. Raises ProviderConfigError listing the available modes otherwise.
    """Return settings with a remembered thinking level for one provider/model."""
    provider = settings.get_provider(provider_name)
    validate_provider_model(provider, model)
    normalized = normalize_thinking_level(thinking_level)
    available = provider_thinking_levels(provider, model=model)
    if normalized not in available:
        modes = ", ".join(available) or "none"
        raise ProviderConfigError(
            f"Thinking mode {normalized} is not available for "
            f"{provider_name}:{model}. Available modes: {modes}"
        )
    updated_provider = replace(
        provider,
        thinking_defaults={**provider.thinking_defaults, model: normalized},
    )
    providers = tuple(
        updated_provider if item.name == provider_name else item for item in settings.providers
    )
    return ProviderSettings(
        default_provider=settings.default_provider,
        providers=providers,
        scoped_models=settings.scoped_models,
    )


def upsert_openai_compatible_provider(
    settings: ProviderSettings,
    provider: OpenAICompatibleProviderConfig,
    *,
    set_default: bool = False,
) -> ProviderSettings:
    # [FUNCTION DOC]
    # Pure alias of upsert_provider for OpenAI-compatible providers.
    """Return settings with an OpenAI-compatible provider added or replaced."""
    return upsert_provider(settings, provider, set_default=set_default)


def upsert_provider(
    settings: ProviderSettings,
    provider: ProviderConfig,
    *,
    set_default: bool = False,
) -> ProviderSettings:
    # [FUNCTION DOC]
    # Pure: adds or replaces a provider by name; when replacing a built-in
    # provider the incoming config is merged with the existing one to keep local
    # customizations. Providers are re-sorted by name; default changes when
    # requested. Raises ProviderConfigError if the resulting default provider does
    # not exist.
    """Return settings with a provider added or replaced."""
    providers_by_name = {item.name: item for item in settings.providers}
    builtin_names = {entry.name for entry in BUILTIN_PROVIDER_CATALOG}
    if provider.name in providers_by_name and provider.name in builtin_names:
        provider = _merge_provider_config(providers_by_name[provider.name], provider)
    providers_by_name[provider.name] = provider
    default_provider = provider.name if set_default else settings.default_provider
    providers = tuple(providers_by_name[name] for name in sorted(providers_by_name))
    updated = ProviderSettings(
        default_provider=default_provider,
        providers=providers,
        scoped_models=settings.scoped_models,
    )
    updated.get_provider(default_provider)
    return updated


def _with_builtin_catalog_models(
    settings: ProviderSettings,
    *,
    paths: TauPaths | None = None,
) -> ProviderSettings:
    # [FUNCTION DOC]
    # HELPER: merges each saved provider with the current effective catalog
    # config of the same name (saved preferences win), appends catalog providers
    # per the append rule, and repairs the default provider to the first provider
    # (or "openai") if it disappeared. Side effects: reads catalog/credentials.
    """Return settings with the current provider catalog merged in."""
    catalog_configs = {config.name: config for config in _effective_provider_configs(paths)}
    providers = tuple(
        _merge_provider_config(provider, catalog_configs[provider.name])
        if provider.name in catalog_configs
        else provider
        for provider in settings.providers
    )
    providers = _append_catalog_providers(providers, catalog_configs, paths=paths)
    default_provider = settings.default_provider
    if default_provider not in {provider.name for provider in providers}:
        default_provider = providers[0].name if providers else DEFAULT_PROVIDER_NAME
    return ProviderSettings(
        default_provider=default_provider,
        providers=providers,
        scoped_models=settings.scoped_models,
    )


def _migrate_legacy_provider_settings(
    settings: ProviderSettings,
    *,
    paths: TauPaths,
) -> ProviderSettings:
    # [FUNCTION DOC]
    # HELPER: migrates legacy full provider records: catalog-known providers take
    # the catalog definition plus only the legacy preferences (default model kept
    # only if still listed, thinking defaults kept only for valid models/levels);
    # unknown providers are kept intact (to be saved as custom catalog entries).
    # Appends catalog providers, repairs the default and drops scoped models whose
    # provider or model no longer exists.
    """Move legacy full provider records onto catalog-owned definitions.

    Built-in and user-catalog provider capabilities come exclusively from the
    current effective catalog. Legacy records contribute only runtime
    preferences. Providers absent from the catalog remain intact so the
    migration can persist them as custom catalog entries.
    """
    catalog_configs = {config.name: config for config in _effective_provider_configs(paths)}
    providers: list[ProviderConfig] = []
    for legacy in settings.providers:
        catalog_provider = catalog_configs.get(legacy.name)
        if catalog_provider is None:
            providers.append(legacy)
            continue
        preferences = _provider_preference_to_json(legacy)
        if legacy.default_model not in catalog_provider.models:
            preferences.pop("default_model")
        preferences["thinking_defaults"] = {
            model: level
            for model, level in legacy.thinking_defaults.items()
            if model in catalog_provider.models
            and level in provider_thinking_levels(catalog_provider, model=model)
        }
        providers.append(_apply_provider_preference(catalog_provider, preferences))

    merged = _append_catalog_providers(tuple(providers), catalog_configs, paths=paths)
    names = {provider.name for provider in merged}
    default_provider = settings.default_provider
    if default_provider not in names:
        default_provider = merged[0].name if merged else DEFAULT_PROVIDER_NAME
    return ProviderSettings(
        default_provider=default_provider,
        providers=merged,
        scoped_models=tuple(
            scoped
            for scoped in settings.scoped_models
            if scoped.provider in names
            and scoped.model
            in next(provider.models for provider in merged if provider.name == scoped.provider)
        ),
    )


def _save_migrated_provider_settings(settings: ProviderSettings, *, paths: TauPaths) -> None:
    # [FUNCTION DOC]
    # HELPER: backs up the existing settings file (failure raises), saves
    # providers not present in the effective catalog as user catalog entries, then
    # writes the migrated preferences without another backup. Side effects: file
    # writes.
    """Persist one legacy migration after creating its required recovery backup."""
    path = provider_settings_path(paths)
    _backup_provider_settings(path, strict=True)
    catalog_names = {entry.name for entry in effective_catalog(paths)}
    custom_entries = [
        _catalog_entry_from_provider(provider)
        for provider in settings.providers
        if provider.name not in catalog_names
    ]
    if custom_entries:
        save_user_catalog_entries(custom_entries, paths=paths)
    _write_provider_settings(settings, path=path, backup=False)


def _backup_provider_settings(path: Path, *, strict: bool) -> None:
    # [FUNCTION DOC]
    # HELPER: copies the settings file to "<name>.bak" if it exists; strict mode
    # propagates copy errors, otherwise OS errors are ignored.
    """Copy existing settings to the recovery path, optionally requiring success."""
    if not path.exists():
        return
    if strict:
        copy2(path, path.with_suffix(path.suffix + ".bak"))
        return
    with suppress(OSError):
        copy2(path, path.with_suffix(path.suffix + ".bak"))


def _write_provider_settings(
    settings: ProviderSettings,
    *,
    path: Path,
    backup: bool,
) -> None:
    # [FUNCTION DOC]
    # HELPER: ensures the directory, optionally makes a best-effort backup, then
    # atomically writes the settings JSON (indented, sorted keys, trailing
    # newline).
    """Atomically write preferences, optionally retaining the previous file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup:
        _backup_provider_settings(path, strict=False)
    _atomic_write_text(path, dumps(settings.to_json(), indent=2, sort_keys=True) + "\n")


def _effective_provider_configs(paths: TauPaths | None = None) -> tuple[ProviderConfig, ...]:
    # [FUNCTION DOC]
    # HELPER: converts every effective catalog entry into a provider config.
    # Raises CatalogError/ProviderConfigError on invalid catalogs.
    """Return provider configs for the effective catalog (builtin + user overlay)."""
    return tuple(provider_config_from_entry(entry) for entry in effective_catalog(paths))


def _append_catalog_providers(
    providers: tuple[ProviderConfig, ...],
    catalog_configs: dict[str, ProviderConfig],
    *,
    paths: TauPaths | None,
) -> tuple[ProviderConfig, ...]:
    # [FUNCTION DOC]
    # HELPER: appends catalog providers missing from the list: user-catalog
    # (non-built-in) providers always, built-in providers only when they have
    # usable credentials in the credential file or environment. Side effects:
    # reads the credential store.
    """Append catalog providers: user-catalog ones always, builtins when credentialed."""
    credential_store = FileCredentialStore(credentials_path(paths) if paths else None)
    builtin_names = {entry.name for entry in BUILTIN_PROVIDER_CATALOG}
    provider_names = {provider.name for provider in providers}
    appended = list(providers)
    for provider in catalog_configs.values():
        if provider.name in provider_names:
            continue
        if provider.name not in builtin_names or provider_has_usable_credentials(
            provider, credential_reader=credential_store
        ):
            appended.append(provider)
            provider_names.add(provider.name)
    return tuple(appended)


def _merge_provider_config(existing: ProviderConfig, incoming: ProviderConfig) -> ProviderConfig:
    # [FUNCTION DOC]
    # HELPER: merges an incoming (catalog/replacement) config with an existing
    # one of the same type, keeping local customizations; different types return
    # the incoming config. For Codex: incoming models, existing default if still
    # listed, headers/context windows/metadata merged with existing winning,
    # existing timeouts/retries, existing thinking group when it defines levels,
    # existing thinking defaults. Other types delegate to type-specific merges.
    """Merge a replacement provider config without losing local customizations."""
    if type(existing) is not type(incoming):
        return incoming

    if isinstance(existing, OpenAICodexProviderConfig) and isinstance(
        incoming, OpenAICodexProviderConfig
    ):
        return replace(
            incoming,
            default_model=(
                existing.default_model
                if existing.default_model in incoming.models
                else incoming.default_model
            ),
            headers={**incoming.headers, **existing.headers},
            timeout_seconds=existing.timeout_seconds,
            max_retries=existing.max_retries,
            max_retry_delay_seconds=existing.max_retry_delay_seconds,
            context_windows={**incoming.context_windows, **existing.context_windows},
            model_metadata=_merge_provider_model_metadata(
                incoming.model_metadata,
                existing.model_metadata,
            ),
            thinking_levels=(
                existing.thinking_levels
                if existing.thinking_levels is not None
                else incoming.thinking_levels
            ),
            thinking_models=(
                existing.thinking_models
                if existing.thinking_levels is not None
                else incoming.thinking_models
            ),
            thinking_default=(
                existing.thinking_default
                if existing.thinking_levels is not None
                else incoming.thinking_default
            ),
            thinking_parameter=(
                existing.thinking_parameter
                if existing.thinking_levels is not None
                else incoming.thinking_parameter
            ),
            thinking_defaults=existing.thinking_defaults,
        )

    if isinstance(existing, OpenAICompatibleProviderConfig) and isinstance(
        incoming, OpenAICompatibleProviderConfig
    ):
        return _merge_openai_compatible_provider(existing, incoming)

    if isinstance(existing, AnthropicProviderConfig) and isinstance(
        incoming, AnthropicProviderConfig
    ):
        return _merge_anthropic_provider(existing, incoming)

    return incoming


def _merge_openai_compatible_provider(
    existing: OpenAICompatibleProviderConfig,
    incoming: OpenAICompatibleProviderConfig,
) -> OpenAICompatibleProviderConfig:
    # [FUNCTION DOC]
    # HELPER: merge for OpenAI-compatible configs: models are incoming then
    # existing (deduplicated), default kept if listed, headers/compat/context
    # windows/metadata merged with existing winning, existing timeouts/retries,
    # thinking group from existing when it defines levels, existing thinking
    # defaults and inference provider routes.
    models = _unique_strings((*incoming.models, *existing.models))
    return replace(
        incoming,
        models=models,
        default_model=(
            existing.default_model if existing.default_model in models else incoming.default_model
        ),
        headers={**incoming.headers, **existing.headers},
        compat={**incoming.compat, **existing.compat},
        model_metadata=_merge_provider_model_metadata(
            incoming.model_metadata,
            existing.model_metadata,
        ),
        timeout_seconds=existing.timeout_seconds,
        max_retries=existing.max_retries,
        max_retry_delay_seconds=existing.max_retry_delay_seconds,
        context_windows={**incoming.context_windows, **existing.context_windows},
        thinking_levels=(
            existing.thinking_levels
            if existing.thinking_levels is not None
            else incoming.thinking_levels
        ),
        thinking_models=(
            existing.thinking_models
            if existing.thinking_levels is not None
            else incoming.thinking_models
        ),
        thinking_default=(
            existing.thinking_default
            if existing.thinking_levels is not None
            else incoming.thinking_default
        ),
        thinking_parameter=(
            existing.thinking_parameter
            if existing.thinking_levels is not None
            else incoming.thinking_parameter
        ),
        thinking_defaults=existing.thinking_defaults,
        inference_providers=existing.inference_providers,
    )


def _merge_anthropic_provider(
    existing: AnthropicProviderConfig,
    incoming: AnthropicProviderConfig,
) -> AnthropicProviderConfig:
    # [FUNCTION DOC]
    # HELPER: same merge rules as the OpenAI-compatible merge, for Anthropic
    # configs (no inference routes).
    models = _unique_strings((*incoming.models, *existing.models))
    return replace(
        incoming,
        models=models,
        default_model=(
            existing.default_model if existing.default_model in models else incoming.default_model
        ),
        headers={**incoming.headers, **existing.headers},
        compat={**incoming.compat, **existing.compat},
        model_metadata=_merge_provider_model_metadata(
            incoming.model_metadata,
            existing.model_metadata,
        ),
        timeout_seconds=existing.timeout_seconds,
        max_retries=existing.max_retries,
        max_retry_delay_seconds=existing.max_retry_delay_seconds,
        context_windows={**incoming.context_windows, **existing.context_windows},
        thinking_levels=(
            existing.thinking_levels
            if existing.thinking_levels is not None
            else incoming.thinking_levels
        ),
        thinking_models=(
            existing.thinking_models
            if existing.thinking_levels is not None
            else incoming.thinking_models
        ),
        thinking_default=(
            existing.thinking_default
            if existing.thinking_levels is not None
            else incoming.thinking_default
        ),
        thinking_parameter=(
            existing.thinking_parameter
            if existing.thinking_levels is not None
            else incoming.thinking_parameter
        ),
        thinking_defaults=existing.thinking_defaults,
    )


def _merge_provider_model_metadata(
    incoming: dict[str, ProviderModelMetadata],
    existing: dict[str, ProviderModelMetadata],
) -> dict[str, ProviderModelMetadata]:
    # [FUNCTION DOC]
    # HELPER: merges per-model metadata: models only in existing are added; for
    # shared models existing truthy scalars win, mappings are key-merged with
    # existing winning, and non-empty existing inputs/tiers replace incoming.
    # KNOWN ISSUE: truthiness checks mean an existing value of zero or empty
    # cannot override the incoming value.
    merged = dict(incoming)
    for model, metadata in existing.items():
        if model not in merged:
            merged[model] = metadata
            continue
        base = merged[model]
        merged[model] = replace(
            base,
            name=metadata.name or base.name,
            api=metadata.api or base.api,
            base_url=metadata.base_url or base.base_url,
            reasoning=metadata.reasoning if metadata.reasoning is not None else base.reasoning,
            input=metadata.input or base.input,
            cost={**base.cost, **metadata.cost},
            cost_tiers=metadata.cost_tiers or base.cost_tiers,
            context_window=metadata.context_window or base.context_window,
            max_tokens=metadata.max_tokens or base.max_tokens,
            headers={**base.headers, **metadata.headers},
            compat={**base.compat, **metadata.compat},
            thinking_level_map={**base.thinking_level_map, **metadata.thinking_level_map},
        )
    return merged


def _unique_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    # [FUNCTION DOC]
    # HELPER: removes duplicates preserving first-seen order.
    """Return values with duplicates removed while preserving order."""
    return tuple(dict.fromkeys(values))


def _atomic_write_text(path: Path, text: str) -> None:
    # [FUNCTION DOC]
    # HELPER: writes text to a sibling temporary file and atomically replaces the
    # target; removes the temp file and re-raises on failure.
    """Write text through a sibling temp file and atomically replace the target."""
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


def _provider_preference_to_json(provider: ProviderConfig) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: serializes only runtime preferences of a provider: default model,
    # headers, timeout, retries, retry delay, thinking defaults, and inference
    # provider routes for OpenAI-compatible providers when non-empty.
    """Serialize only runtime preferences for one provider."""
    preference = {
        "default_model": provider.default_model,
        "headers": dict(provider.headers),
        "timeout_seconds": provider.timeout_seconds,
        "max_retries": provider.max_retries,
        "max_retry_delay_seconds": provider.max_retry_delay_seconds,
        "thinking_defaults": dict(provider.thinking_defaults),
    }
    if isinstance(provider, OpenAICompatibleProviderConfig) and provider.inference_providers:
        preference["inference_providers"] = dict(provider.inference_providers)
    return preference


def _save_provider_definitions_to_catalog(
    settings: ProviderSettings,
    *,
    paths: TauPaths | None,
) -> None:
    # [FUNCTION DOC]
    # HELPER: for each provider not in the effective catalog, or whose definition
    # differs from its catalog entry, upserts a catalog entry (keeping the
    # existing display name/docs URL) into the user catalog. Side effect: may
    # write catalog.toml.
    """Persist provider definitions that are not already represented by the catalog."""
    catalog_by_name = {entry.name: entry for entry in effective_catalog(paths)}
    entries_to_save = []
    for provider in settings.providers:
        entry = catalog_by_name.get(provider.name)
        if entry is None or _provider_definition_differs_from_catalog(provider, entry):
            entries_to_save.append(_catalog_entry_from_provider(provider, existing=entry))
    if entries_to_save:
        save_user_catalog_entries(entries_to_save, paths=paths)


def _provider_definition_differs_from_catalog(
    provider: ProviderConfig,
    entry: ProviderCatalogEntry,
) -> bool:
    # [FUNCTION DOC]
    # HELPER: returns true when kind, base URL, API key env, credential name,
    # models, API (only when the entry declares one), context windows, headers,
    # compat, model metadata or any thinking field differ from the catalog entry.
    # Note: headers are runtime preferences too, so a header-only change also
    # writes a catalog entry.
    """Return whether provider metadata changed enough to belong in catalog.toml."""
    if provider_kind(provider) != entry.kind:
        return True
    if provider.base_url != entry.base_url:
        return True
    if provider.api_key_env != entry.api_key_env:
        return True
    if provider.credential_name != entry.credential_name:
        return True
    if provider.models != entry.models:
        return True
    if getattr(provider, "api", None) != entry.api and entry.api is not None:
        return True
    if provider.context_windows != dict(entry.context_windows or {}):
        return True
    if provider.headers != dict(entry.headers):
        return True
    if getattr(provider, "compat", {}) != dict(entry.compat):
        return True
    if _catalog_model_metadata_from_provider(provider) != entry.model_metadata:
        return True
    if provider.thinking_levels != entry.thinking_levels:
        return True
    if provider.thinking_models != entry.thinking_models:
        return True
    if provider.thinking_default != entry.thinking_default:
        return True
    return provider.thinking_parameter != entry.thinking_parameter


def _catalog_entry_from_provider(
    provider: ProviderConfig,
    *,
    existing: ProviderCatalogEntry | None = None,
) -> ProviderCatalogEntry:
    # [FUNCTION DOC]
    # HELPER: builds a catalog entry from a provider config, reusing the existing
    # entry's display name, docs URL and default model (when still listed),
    # otherwise using the provider name, base URL and its default model.
    """Create catalog metadata from a runtime provider config."""
    return ProviderCatalogEntry(
        name=provider.name,
        display_name=existing.display_name if existing is not None else provider.name,
        kind=provider_kind(provider),
        base_url=provider.base_url,
        api_key_env=provider.api_key_env,
        api=getattr(provider, "api", None),
        credential_name=provider.credential_name,
        models=provider.models,
        default_model=(
            existing.default_model
            if existing is not None and existing.default_model in provider.models
            else provider.default_model
        ),
        docs_url=existing.docs_url if existing is not None else provider.base_url,
        context_windows=dict(provider.context_windows) or None,
        headers=dict(provider.headers),
        compat=dict(getattr(provider, "compat", {})),
        model_metadata=_catalog_model_metadata_from_provider(provider),
        thinking_levels=provider.thinking_levels,
        thinking_models=provider.thinking_models,
        thinking_default=provider.thinking_default,
        thinking_parameter=provider.thinking_parameter,
    )


def _catalog_model_metadata_from_provider(
    provider: ProviderConfig,
) -> dict[str, ModelCatalogMetadata]:
    # [FUNCTION DOC]
    # HELPER: converts runtime model metadata to catalog metadata, keeping only
    # text/image input modalities and turning empty cost into none.
    metadata_by_model = getattr(provider, "model_metadata", {})
    return {
        model: ModelCatalogMetadata(
            name=metadata.name,
            api=metadata.api,
            base_url=metadata.base_url,
            reasoning=metadata.reasoning,
            input=tuple(item for item in metadata.input if item in {"text", "image"}),
            cost=dict(metadata.cost) or None,
            cost_tiers=metadata.cost_tiers,
            context_window=metadata.context_window,
            max_tokens=metadata.max_tokens,
            headers=dict(metadata.headers),
            compat=dict(metadata.compat),
            thinking_level_map=dict(metadata.thinking_level_map),
        )
        for model, metadata in metadata_by_model.items()
    }


def provider_settings_from_json(
    data: dict[str, Any],
    *,
    paths: TauPaths | None = None,
) -> ProviderSettings:
    # [FUNCTION DOC]
    # Parses settings JSON. Accepts schema_version missing or 2 (otherwise raises).
    # Requires a string default provider. The current shape applies
    # provider_preferences onto effective catalog configs; the legacy shape
    # requires a non-empty providers list of full definitions with unique names.
    # Scoped models are parsed and deduplicated. Raises ProviderConfigError for
    # any invalid field. Side effect: reads the effective catalog.
    """Parse provider preferences from JSON-compatible data.

    The current providers.json shape stores runtime preferences under
    provider_preferences. The older providers[] shape is still accepted for
    migration and compatibility; saves rewrite it to provider_preferences and
    move custom provider definitions to catalog.toml.
    """
    schema_version = data.get("schema_version")
    if schema_version not in (None, PROVIDER_SETTINGS_SCHEMA_VERSION):
        raise ProviderConfigError(
            f"Unsupported provider settings schema_version: {schema_version!r}"
        )
    default_provider = _string(data.get("default_provider"), "default_provider")
    scoped_models = _scoped_models_from_json(data.get("scoped_models"))
    if "provider_preferences" in data:
        providers = _providers_with_preferences(
            data.get("provider_preferences"),
            paths=paths,
        )
        return ProviderSettings(
            default_provider=default_provider,
            providers=providers,
            scoped_models=scoped_models,
        )

    providers_data = data.get("providers")
    if not isinstance(providers_data, list) or not providers_data:
        raise ProviderConfigError(
            "Provider settings must include provider_preferences or legacy providers"
        )
    providers = tuple(_provider_from_json(item) for item in providers_data)
    names = [provider.name for provider in providers]
    if len(set(names)) != len(names):
        raise ProviderConfigError("Provider names must be unique")
    return ProviderSettings(
        default_provider=default_provider,
        providers=providers,
        scoped_models=scoped_models,
    )


def _providers_with_preferences(
    value: object,
    *,
    paths: TauPaths | None,
) -> tuple[ProviderConfig, ...]:
    # [FUNCTION DOC]
    # HELPER: requires an object; for each non-empty, unique provider name present
    # in the effective catalog applies its preferences to the catalog config.
    # Orphaned preferences (provider no longer in the catalog) are skipped.
    # Raises ProviderConfigError for bad or duplicate names (after stripping).
    if not isinstance(value, dict):
        raise ProviderConfigError("Provider settings field must be an object: provider_preferences")
    catalog_configs = {provider.name: provider for provider in _effective_provider_configs(paths)}
    providers = []
    seen: set[str] = set()
    for name, preference_data in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ProviderConfigError("Provider preference names must be non-empty strings")
        provider_name = name.strip()
        if provider_name in seen:
            raise ProviderConfigError("Provider preference names must be unique")
        if provider_name not in catalog_configs:
            # Preferences contain runtime overrides, not provider definitions. A
            # catalog entry may be removed independently, leaving an orphaned
            # preference behind. Ignore it so one stale entry cannot prevent Tau
            # from starting or running `tau setup` to register it again.
            continue
        providers.append(
            _apply_provider_preference(
                catalog_configs[provider_name],
                preference_data,
            )
        )
        seen.add(provider_name)
    return tuple(providers)


def _apply_provider_preference(
    provider: ProviderConfig,
    value: object,
) -> ProviderConfig:
    # [FUNCTION DOC]
    # HELPER: overlays a preference object onto a provider config, validating
    # each recognized key (default model reverts to the provider default if not
    # listed; headers; positive timeout; non-negative retries and delay; thinking
    # defaults with unknown models and unavailable levels dropped; inference
    # provider routes for OpenAI-compatible providers). Unknown keys are ignored
    # for forward compatibility. Raises ProviderConfigError for invalid values.
    if not isinstance(value, dict):
        raise ProviderConfigError("Provider preference entries must be objects")
    # Provider preferences are user-level state shared across Tau versions.
    # Ignore options introduced by newer versions while continuing to validate
    # every recognized option below.
    default_model = (
        _string(value.get("default_model"), f"provider_preferences.{provider.name}.default_model")
        if "default_model" in value
        else provider.default_model
    )
    if default_model not in provider.models:
        default_model = provider.default_model
    headers = (
        _string_dict(value.get("headers"), f"provider_preferences.{provider.name}.headers")
        if "headers" in value
        else provider.headers
    )
    timeout_seconds = (
        _positive_float(
            value.get("timeout_seconds"),
            f"provider_preferences.{provider.name}.timeout_seconds",
        )
        if "timeout_seconds" in value
        else provider.timeout_seconds
    )
    max_retries = (
        _non_negative_int(
            value.get("max_retries"),
            f"provider_preferences.{provider.name}.max_retries",
        )
        if "max_retries" in value
        else provider.max_retries
    )
    max_retry_delay_seconds = (
        _non_negative_float(
            value.get("max_retry_delay_seconds"),
            f"provider_preferences.{provider.name}.max_retry_delay_seconds",
        )
        if "max_retry_delay_seconds" in value
        else provider.max_retry_delay_seconds
    )
    thinking_defaults = (
        _thinking_defaults_dict(
            value.get("thinking_defaults"),
            provider,
            f"provider_preferences.{provider.name}.thinking_defaults",
            ignore_unknown_models=True,
            ignore_unavailable=True,
        )
        if "thinking_defaults" in value
        else provider.thinking_defaults
    )
    if isinstance(provider, OpenAICompatibleProviderConfig):
        inference_providers = (
            _inference_providers_dict(
                value.get("inference_providers"),
                provider,
                f"provider_preferences.{provider.name}.inference_providers",
            )
            if "inference_providers" in value
            else provider.inference_providers
        )
        return replace(
            provider,
            default_model=default_model,
            headers=headers,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_retry_delay_seconds=max_retry_delay_seconds,
            thinking_defaults=thinking_defaults,
            inference_providers=inference_providers,
        )
    return replace(
        provider,
        default_model=default_model,
        headers=headers,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_retry_delay_seconds=max_retry_delay_seconds,
        thinking_defaults=thinking_defaults,
    )


def _inference_providers_dict(
    value: object,
    provider: OpenAICompatibleProviderConfig,
    field_name: str,
) -> dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: parses a string mapping of model to route, keeps only configured
    # models with stripped routes and validates them.
    routes = _string_dict(value, field_name)
    routes = {model: route.strip() for model, route in routes.items() if model in provider.models}
    _validate_inference_providers(provider.name, provider.models, routes)
    return routes


def _thinking_defaults_dict(
    value: object,
    provider: ProviderConfig,
    field_name: str,
    *,
    ignore_unknown_models: bool = False,
    ignore_unavailable: bool = False,
) -> dict[str, ThinkingLevel]:
    # [FUNCTION DOC]
    # HELPER: parses a mapping of model to thinking level; optionally drops
    # unknown models and unavailable levels, otherwise raises ProviderConfigError
    # for an unknown model or a level not available for that model.
    raw = _raw_thinking_defaults_dict(value, field_name)
    if ignore_unknown_models:
        raw = {model: level for model, level in raw.items() if model in provider.models}
    valid: dict[str, ThinkingLevel] = {}
    for model, thinking_level in raw.items():
        validate_provider_model(provider, model)
        available = provider_thinking_levels(provider, model=model)
        if thinking_level not in available:
            if ignore_unavailable:
                continue
            modes = ", ".join(available) or "none"
            raise ProviderConfigError(
                f"Provider thinking default {thinking_level} is not available for "
                f"{provider.name}:{model}. Available modes: {modes}"
            )
        valid[model] = thinking_level
    return valid


def _raw_thinking_defaults_dict(value: object, field_name: str) -> dict[str, ThinkingLevel]:
    # [FUNCTION DOC]
    # HELPER: requires an object whose keys are strings and values valid thinking
    # level names; raises ProviderConfigError otherwise.
    if not isinstance(value, dict):
        raise ProviderConfigError(f"Provider field must be a thinking mode object: {field_name}")
    defaults: dict[str, ThinkingLevel] = {}
    for key, item in value.items():
        model = _string(key, field_name)
        thinking_level = _optional_thinking_level(item, f"{field_name}.{model}")
        if thinking_level is None:
            raise ProviderConfigError(f"Provider field must be a thinking mode: {field_name}")
        defaults[model] = thinking_level
    return defaults


def _scoped_models_from_json(value: object) -> tuple[ScopedModelConfig, ...]:
    # [FUNCTION DOC]
    # HELPER: parses an optional list of provider/model objects into scoped model
    # configs, removing duplicates. Raises ProviderConfigError for wrong types.
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ProviderConfigError("Provider settings field must be a list: scoped_models")
    scoped: list[ScopedModelConfig] = []
    seen: set[tuple[str, str]] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ProviderConfigError("Provider scoped_models entries must be objects")
        provider = _string(item.get("provider"), "scoped_models.provider")
        model = _string(item.get("model"), "scoped_models.model")
        key = (provider, model)
        if key not in seen:
            scoped.append(ScopedModelConfig(provider=provider, model=model))
            seen.add(key)
    return tuple(scoped)


def resolve_provider_selection(
    settings: ProviderSettings,
    *,
    provider_name: str | None = None,
    model: str | None = None,
) -> ProviderSelection:
    # [FUNCTION DOC]
    # Resolves the provider (given name or default) and model (given or the
    # provider's default) for a run and validates that the model is configured.
    # Raises ProviderConfigError for unknown providers, no default model or an
    # unconfigured model.
    """Resolve the provider and model for a run."""
    provider = settings.get_provider(provider_name)
    selected_model = model or provider.default_model
    if not selected_model:
        raise ProviderConfigError(f"Provider {provider.name} does not define a default model")
    validate_provider_model(provider, selected_model)
    return ProviderSelection(provider=provider, model=selected_model)


def validate_provider_model(provider: ProviderConfig, model: str) -> None:
    # [FUNCTION DOC]
    # Returns silently when the model is in the provider's models; otherwise
    # raises ProviderConfigError listing the available models sorted.
    """Raise when ``model`` is not declared by ``provider``."""
    if model in provider.models:
        return
    available = ", ".join(sorted(provider.models)) or "none"
    raise ProviderConfigError(
        f"Model is not configured for provider {provider.name}: {model}. "
        f"Available models: {available}"
    )


def provider_thinking_levels(
    provider: ProviderConfig,
    *,
    model: str | None = None,
) -> tuple[ThinkingLevel, ...]:
    # [FUNCTION DOC]
    # Returns the thinking levels (in canonical order) supported by a provider and
    # model (default model when none). Rules: metadata marking the model as
    # non-reasoning gives none; without provider-declared levels, only models
    # whose metadata says reasoning get levels derived from their level map
    # (xhigh and max only if explicitly mapped); with declared levels, models
    # outside a non-empty thinking models list get none, otherwise a level is
    # offered if declared or mapped in metadata, and supported by the metadata.
    """Return thinking levels supported by a provider/model pair."""
    selected_model = model or provider.default_model
    metadata = _metadata_for_model(provider, selected_model)
    if metadata is not None and metadata.reasoning is False:
        return ()
    if provider.thinking_levels is None:
        if metadata is None or metadata.reasoning is not True:
            return ()
        return _levels_from_thinking_map(metadata.thinking_level_map)
    if provider.thinking_models and selected_model not in provider.thinking_models:
        return ()
    return tuple(
        level
        for level in THINKING_LEVELS
        if (
            level in provider.thinking_levels
            or metadata is not None
            and metadata.thinking_level_map.get(level) is not None
        )
        and (metadata is None or _metadata_supports_thinking_level(metadata, level))
    )


def provider_thinking_unavailable_reason(
    provider: ProviderConfig,
    *,
    model: str | None = None,
) -> str | None:
    # [FUNCTION DOC]
    # Explains why no thinking modes are configurable: non-reasoning model,
    # Codex lacking a validated effort mapping, provider declaring no levels, or
    # model not in thinking models. Returns none when modes are available by
    # declaration or reasoning metadata.
    """Explain why a provider/model pair has no configurable thinking modes."""
    selected_model = model or provider.default_model
    metadata = _metadata_for_model(provider, selected_model)
    if metadata is not None and metadata.reasoning is False:
        return f"{provider.name}:{selected_model} is not a reasoning model"
    if provider.thinking_levels is None:
        if metadata is not None and metadata.reasoning is True:
            return None
        if isinstance(provider, OpenAICodexProviderConfig):
            return (
                "OpenAI Codex subscription can stream reasoning output, but Tau does "
                "not have a validated Codex transport mapping for changing reasoning "
                "effort yet"
            )
        return f"Provider {provider.name} does not declare thinking_levels"
    if provider.thinking_models and selected_model not in provider.thinking_models:
        return f"{provider.name}:{selected_model} is not declared in thinking_models"
    return None


def _levels_from_thinking_map(
    thinking_level_map: dict[ThinkingLevel, str | None],
) -> tuple[ThinkingLevel, ...]:
    # [FUNCTION DOC]
    # HELPER: canonical-ordered levels the level map supports.
    return tuple(
        level
        for level in THINKING_LEVELS
        if _thinking_level_map_supports(thinking_level_map, level)
    )


def _metadata_supports_thinking_level(
    metadata: ProviderModelMetadata,
    level: ThinkingLevel,
) -> bool:
    # [FUNCTION DOC]
    # HELPER: true when metadata has neither a reasoning flag nor a level map
    # (unknown, assume supported), else defers to the level map rule.
    if metadata.reasoning is None and not metadata.thinking_level_map:
        return True
    return _thinking_level_map_supports(metadata.thinking_level_map, level)


def _thinking_level_map_supports(
    thinking_level_map: dict[ThinkingLevel, str | None],
    level: ThinkingLevel,
) -> bool:
    # [FUNCTION DOC]
    # HELPER: a mapped level is supported when its value is not none; an
    # unmapped level is supported unless it is xhigh or max.
    if level in thinking_level_map:
        return thinking_level_map[level] is not None
    return level not in {"xhigh", "max"}


def _metadata_for_model(provider: ProviderConfig, model: str) -> ProviderModelMetadata | None:
    # [FUNCTION DOC]
    # HELPER: returns a model's metadata from the provider, or none.
    return getattr(provider, "model_metadata", {}).get(model)


def _provider_api(provider: ProviderConfig, model: str | None = None) -> ProviderApi | str:
    # [FUNCTION DOC]
    # HELPER: effective API for a model: metadata override, else the Codex
    # responses API for Codex configs, else the provider API.
    selected_model = model or provider.default_model
    metadata = _metadata_for_model(provider, selected_model)
    if metadata is not None and metadata.api is not None:
        return metadata.api
    if isinstance(provider, OpenAICodexProviderConfig):
        return "openai-codex-responses"
    return getattr(provider, "api", "openai-completions")


def _model_base_url(provider: ProviderConfig, model: str | None = None) -> str:
    # [FUNCTION DOC]
    # HELPER: model metadata base URL override, else the provider base URL.
    selected_model = model or provider.default_model
    metadata = _metadata_for_model(provider, selected_model)
    return metadata.base_url if metadata is not None and metadata.base_url else provider.base_url


def _model_headers(provider: ProviderConfig, model: str | None = None) -> dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: provider headers overlaid by model metadata headers.
    selected_model = model or provider.default_model
    metadata = _metadata_for_model(provider, selected_model)
    return {**provider.headers, **(metadata.headers if metadata is not None else {})}


def _model_compat(provider: ProviderConfig, model: str | None = None) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: compat options layered as detected defaults, then provider compat,
    # then model metadata compat (later wins).
    selected_model = model or provider.default_model
    metadata = _metadata_for_model(provider, selected_model)
    return {
        **_detected_compat(provider, selected_model),
        **getattr(provider, "compat", {}),
        **(metadata.compat if metadata is not None else {}),
    }


def _detected_compat(provider: ProviderConfig, model: str) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: infers OpenAI-compatible behavior flags from the provider name and
    # base URL host (Together, Z.ai, Moonshot, xAI, DeepSeek, Cerebras,
    # OpenRouter, OpenAI, Anthropic): store support, reasoning-effort support,
    # streaming usage, max tokens field name, thinking request format,
    # developer-role and strict-mode support, reasoning content field, and related
    # flags. Output: a compat mapping.
    base_url = _model_base_url(provider, model)
    is_together = provider.name == "together" or "api.together.ai" in base_url
    is_zai = provider.name == "zai" or "api.z.ai" in base_url
    is_moonshot = provider.name in {"moonshotai", "moonshotai-cn"} or "moonshot." in base_url
    is_grok = provider.name == "xai" or "api.x.ai" in base_url
    is_deepseek = provider.name == "deepseek" or "deepseek.com" in base_url
    is_cerebras = provider.name == "cerebras" or "cerebras.ai" in base_url
    is_openrouter = provider.name == "openrouter" or "openrouter.ai" in base_url
    is_openai_api = (
        urlsplit(base_url).hostname == urlsplit(DEFAULT_OPENAI_COMPATIBLE_BASE_URL).hostname
    )
    is_openai_responses = _provider_api(provider, model) == "openai-responses"
    is_nonstandard = is_cerebras or is_grok or is_together or is_deepseek or is_zai or is_moonshot
    use_max_tokens = is_moonshot or is_together
    is_anthropic_api = urlsplit(base_url).hostname == urlsplit(DEFAULT_ANTHROPIC_BASE_URL).hostname
    return {
        "supportsStore": not is_nonstandard,
        "supportsReasoningEffort": not (is_grok or is_zai or is_moonshot or is_together),
        "supportsUsageInStreaming": True,
        "maxTokensField": "max_tokens" if use_max_tokens else "max_completion_tokens",
        "thinkingFormat": (
            "deepseek"
            if is_deepseek
            else "zai"
            if is_zai
            else "together"
            if is_together
            else "openrouter"
            if is_openrouter
            else "openai"
        ),
        "supportsStrictMode": not (is_moonshot or is_together),
        "supportsLongCacheRetention": not is_together,
        # OpenAI's prompt-cache fields and affinity headers are not universally
        # accepted by compatible gateways. Default them on only for the official
        # endpoint; provider/model compat can opt another route in explicitly.
        "supportsPromptCacheKey": is_openai_api,
        "sendSessionAffinityHeaders": is_openai_api and is_openai_responses,
        "sessionAffinityFormat": "openrouter" if is_openrouter else "openai",
        # Only first-party Anthropic is known to accept cache_control. Several
        # catalog providers speak the Anthropic protocol through a gateway, and one
        # proxies to non-Anthropic models, so they default to no breakpoints. This
        # is a detected default, overridable per provider or per model.
        "supportsCacheControl": is_anthropic_api,
        "supportsCacheControlOnTools": True,
    }


def provider_model_max_tokens(provider: ProviderConfig, model: str | None = None) -> int | None:
    # [FUNCTION DOC]
    # Returns the model's catalog max output tokens, or none when unset or no
    # metadata.
    """Return the catalog output token limit for a model, or None when it is unset."""
    selected_model = model or provider.default_model
    metadata = _metadata_for_model(provider, selected_model)
    return metadata.max_tokens if metadata is not None else None


def provider_model_supports_images(provider: ProviderConfig, model: str | None = None) -> bool:
    # [FUNCTION DOC]
    # Returns true only when the model's metadata lists image input.
    selected_model = model or provider.default_model
    metadata = _metadata_for_model(provider, selected_model)
    return metadata is not None and "image" in metadata.input


def provider_default_thinking_level(
    provider: ProviderConfig,
    *,
    model: str | None = None,
) -> ThinkingLevel | None:
    # [FUNCTION DOC]
    # Returns the preferred level for a provider/model: none when no levels; the
    # provider's thinking default if available; else the global default level if
    # available; else the first available level.
    """Return the preferred thinking level for a provider/model pair."""
    levels = provider_thinking_levels(provider, model=model)
    if not levels:
        return None
    if provider.thinking_default in levels:
        return provider.thinking_default
    if DEFAULT_THINKING_LEVEL in levels:
        return DEFAULT_THINKING_LEVEL
    return levels[0]


def resolve_startup_thinking_level(
    provider: ProviderConfig,
    model: str,
    *,
    preferred: ThinkingLevel = DEFAULT_THINKING_LEVEL,
    cli_override: ThinkingLevel | None = None,
) -> ThinkingLevel | None:
    # [FUNCTION DOC]
    # Picks the startup thinking level. A CLI override is strict: raises
    # ProviderConfigError when the model has no levels or does not support it,
    # else is returned. Otherwise returns none when no levels exist, then the
    # remembered per-model level, the preferred (global) level, the provider
    # default, or the first available level, in that order.
    """Pick a valid startup thinking level for a provider/model pair.

    Startup (TUI and print mode) must never crash just because the remembered
    default model does not support the global default level. The level is
    resolved with the same precedence used when switching models mid-session:
    the remembered per-model preference wins, then the global ``preferred``
    level, then the provider/catalog default, then the first available level.

    An explicit ``cli_override`` (from ``--thinking``) takes precedence over
    everything, and unlike the fallback chain it is strict: requesting a level
    the model does not support raises :class:`ProviderConfigError` instead of
    silently falling back.

    Returns ``None`` when the model has no configurable thinking levels.
    """
    levels = provider_thinking_levels(provider, model=model)
    if cli_override is not None:
        if not levels:
            raise ProviderConfigError(f"Thinking modes are unavailable for {provider.name}:{model}")
        if cli_override not in levels:
            allowed = ", ".join(levels)
            raise ProviderConfigError(
                f'Thinking mode "{cli_override}" is not available for '
                f"{provider.name}:{model}. Available modes: {allowed}"
            )
        return cli_override
    if not levels:
        return None
    remembered = provider.thinking_defaults.get(model)
    if remembered in levels:
        return remembered
    if preferred in levels:
        return preferred
    return provider_default_thinking_level(provider, model=model) or levels[0]


def openai_compatible_config_from_provider(
    provider: OpenAICompatibleProviderConfig,
    *,
    credential_reader: CredentialReader | None = None,
    model: str | None = None,
    thinking_level: ThinkingLevel | None = None,
) -> OpenAICompatibleConfig:
    # [FUNCTION DOC]
    # Builds the tau_ai OpenAI-compatible runtime config for a model: API key from
    # credentials or environment, model base URL (the default openai provider
    # using OPENAI_API_KEY honors the OPENAI_BASE_URL environment override),
    # trailing slash removed, effective API, merged headers, timeouts/retries,
    # image support, mapped reasoning effort and parameter name, thinking format,
    # layered compat, the x-inference-provider response header for huggingface,
    # and whether to send an explicit "none" effort for off.
    # Raises RuntimeError when no API key is found and ProviderConfigError for an
    # unavailable thinking level. Side effect: reads environment and credentials.
    """Build OpenAI-compatible runtime config from durable settings."""
    api_key = _api_key_from_provider(provider, credential_reader=credential_reader)
    selected_model = model or provider.default_model
    base_url = _model_base_url(provider, selected_model)
    if provider.name == DEFAULT_PROVIDER_NAME and provider.api_key_env == "OPENAI_API_KEY":
        base_url = environ.get("OPENAI_BASE_URL", base_url)
    reasoning_effort = _reasoning_effort_from_provider(
        provider,
        model=selected_model,
        thinking_level=thinking_level,
    )
    compat = _model_compat(provider, selected_model)
    return OpenAICompatibleConfig(
        api_key=api_key,
        provider_name=provider.name,
        api=str(_provider_api(provider, selected_model)),
        base_url=base_url.rstrip("/"),
        headers=_model_headers(provider, selected_model),
        timeout_seconds=provider.timeout_seconds,
        max_retries=provider.max_retries,
        max_retry_delay_seconds=provider.max_retry_delay_seconds,
        supports_images=provider_model_supports_images(provider, selected_model),
        reasoning_effort=reasoning_effort,
        reasoning_effort_parameter=provider.thinking_parameter or "reasoning_effort",
        thinking_format=_thinking_format(provider, selected_model),
        compat=compat,
        response_provider_header=(
            "x-inference-provider" if provider.name == "huggingface" else None
        ),
        include_reasoning_effort_none=_include_reasoning_effort_none(
            provider,
            model=selected_model,
            thinking_level=thinking_level,
        ),
    )


def anthropic_config_from_provider(
    provider: AnthropicProviderConfig,
    *,
    credential_reader: CredentialReader | None = None,
    model: str | None = None,
    thinking_level: ThinkingLevel | None = None,
) -> AnthropicConfig:
    # [FUNCTION DOC]
    # Builds the tau_ai Anthropic runtime config: API key, base URL normalized to
    # end with /v1, cache retention and tool cache-control (API-key intent, i.e.
    # short retention), headers, timeouts/retries, max tokens, image support,
    # thinking budget (budget mode only), thinking effort and thinking mode.
    # Raises RuntimeError for missing keys and ProviderConfigError for an
    # unavailable thinking level.
    """Build Anthropic runtime config from durable settings."""
    api_key = _api_key_from_provider(provider, credential_reader=credential_reader)
    selected_model = model or provider.default_model
    thinking_budget_tokens = _anthropic_thinking_budget_from_provider(
        provider,
        model=selected_model,
        thinking_level=thinking_level,
    )
    base_url = _normalize_anthropic_base_url(_model_base_url(provider, selected_model))
    cache_retention, cache_control_on_tools = anthropic_cache_settings(provider, selected_model)
    return AnthropicConfig(
        api_key=api_key,
        provider_name=provider.name,
        base_url=base_url,
        cache_retention=cache_retention,
        cache_control_on_tools=cache_control_on_tools,
        headers=_model_headers(provider, selected_model),
        timeout_seconds=provider.timeout_seconds,
        max_retries=provider.max_retries,
        max_retry_delay_seconds=provider.max_retry_delay_seconds,
        max_tokens=provider_model_max_tokens(provider, selected_model),
        supports_images=provider_model_supports_images(provider, selected_model),
        thinking_budget_tokens=thinking_budget_tokens,
        thinking_effort=_reasoning_effort_from_anthropic_provider(
            provider,
            model=selected_model,
            thinking_level=thinking_level,
        ),
        thinking_mode=_anthropic_thinking_mode(
            provider, selected_model, thinking_level=thinking_level
        ),
    )


def provider_kind(provider: ProviderConfig) -> ProviderKind:
    # [FUNCTION DOC]
    # Returns the durable kind of a config: anthropic, openai-codex,
    # google-generative-ai or mistral-conversations (by API of OpenAI-compatible
    # configs), else openai-compatible.
    """Return the durable provider kind."""
    if isinstance(provider, AnthropicProviderConfig):
        return "anthropic"
    if isinstance(provider, OpenAICodexProviderConfig):
        return "openai-codex"
    if isinstance(provider, OpenAICompatibleProviderConfig):
        if provider.api == "google-generative-ai":
            return "google-generative-ai"
        if provider.api == "mistral-conversations":
            return "mistral-conversations"
    return "openai-compatible"


def provider_has_usable_credentials(
    provider: ProviderConfig,
    *,
    credential_reader: CredentialReader | None = None,
) -> bool:
    # [FUNCTION DOC]
    # Returns true when calls can be attempted without setup: a stored OAuth
    # credential exists for a provider with a registered OAuth implementation, a
    # stored API key credential exists, or the API key environment variable is
    # non-empty. Side effect: reads credentials and environment.
    """Return whether Tau can attempt calls for this provider without prompting setup."""
    if provider.credential_name and credential_reader is not None:
        get_oauth = getattr(credential_reader, "get_oauth", None)
        if (
            get_oauth_provider(provider.name) is not None
            and get_oauth is not None
            and get_oauth(provider.credential_name) is not None
        ):
            return True
        if credential_reader.get(provider.credential_name):
            return True
    return bool(environ.get(provider.api_key_env))


def _reasoning_effort_from_provider(
    provider: OpenAICompatibleProviderConfig,
    *,
    model: str | None,
    thinking_level: ThinkingLevel | None,
) -> str | None:
    # [FUNCTION DOC]
    # HELPER: maps a thinking level to the provider's reasoning effort value when
    # the thinking parameter is reasoning_effort or reasoning.effort and the
    # model has levels: metadata map value first, huggingface maps minimal to
    # low, else the generic effort mapping. Returns none when not applicable.
    # Raises ProviderConfigError for an unavailable level.
    if thinking_level is None or provider.thinking_parameter not in {
        "reasoning_effort",
        "reasoning.effort",
    }:
        return None

    levels = provider_thinking_levels(provider, model=model)
    if not levels:
        return None

    selected_model = model or provider.default_model
    normalized = normalize_thinking_level(thinking_level)
    if normalized not in levels:
        available = ", ".join(levels)
        raise ProviderConfigError(
            f"Thinking mode {normalized} is not available for "
            f"{provider.name}:{selected_model}. Available modes: {available}"
        )
    mapped = _metadata_thinking_value(provider, selected_model, normalized)
    if mapped is not None:
        return mapped
    if provider.name == "huggingface" and normalized == "minimal":
        # Hugging Face's router currently accepts low/medium/high/xhigh/max/none
        # for reasoning_effort, but rejects Pi/Tau's "minimal" label.
        return "low"
    return reasoning_effort_for_level(normalized)


def _anthropic_thinking_budget_from_provider(
    provider: AnthropicProviderConfig,
    *,
    model: str | None,
    thinking_level: ThinkingLevel | None,
) -> int | None:
    # [FUNCTION DOC]
    # HELPER: returns a token budget for the level when the provider uses the
    # anthropic.thinking parameter, the model is in budget (not adaptive) mode
    # and has levels; none otherwise. Raises ProviderConfigError for an
    # unavailable level.
    if thinking_level is None or provider.thinking_parameter != "anthropic.thinking":
        return None

    selected_model = model or provider.default_model
    if _anthropic_thinking_mode(provider, selected_model) == "adaptive":
        return None

    levels = provider_thinking_levels(provider, model=selected_model)
    if not levels:
        return None

    normalized = normalize_thinking_level(thinking_level)
    if normalized not in levels:
        available = ", ".join(levels)
        raise ProviderConfigError(
            f"Thinking mode {normalized} is not available for "
            f"{provider.name}:{selected_model}. Available modes: {available}"
        )
    return anthropic_thinking_budget_for_level(normalized)


def _metadata_thinking_value(
    provider: ProviderConfig,
    model: str,
    level: ThinkingLevel,
) -> str | None:
    # [FUNCTION DOC]
    # HELPER: the model metadata's provider string for a level, or none.
    metadata = _metadata_for_model(provider, model)
    if metadata is None:
        return None
    value = metadata.thinking_level_map.get(level)
    return value if isinstance(value, str) else None


def _thinking_format(provider: ProviderConfig, model: str) -> str:
    # [FUNCTION DOC]
    # HELPER: thinking request format from compat thinkingFormat if set, else
    # detected from provider name or base URL (deepseek, zai, together,
    # openrouter), else openai.
    compat = _model_compat(provider, model)
    value = compat.get("thinkingFormat")
    if isinstance(value, str) and value:
        return value
    base_url = _model_base_url(provider, model)
    if provider.name == "deepseek" or "deepseek.com" in base_url:
        return "deepseek"
    if provider.name == "zai" or "api.z.ai" in base_url:
        return "zai"
    if provider.name == "together" or "api.together.ai" in base_url:
        return "together"
    if provider.name == "openrouter" or "openrouter.ai" in base_url:
        return "openrouter"
    return "openai"


def _include_reasoning_effort_none(
    provider: ProviderConfig,
    *,
    model: str,
    thinking_level: ThinkingLevel | None,
) -> bool:
    # [FUNCTION DOC]
    # HELPER: true only when the level normalizes to off and the model metadata
    # maps off to "none"; invalid levels give false.
    if thinking_level is None:
        return False
    try:
        normalized = normalize_thinking_level(thinking_level)
    except ValueError:
        return False
    if normalized != "off":
        return False
    return _metadata_thinking_value(provider, model, "off") == "none"


def _reasoning_effort_from_anthropic_provider(
    provider: AnthropicProviderConfig,
    *,
    model: str,
    thinking_level: ThinkingLevel | None,
) -> str | None:
    # [FUNCTION DOC]
    # HELPER: effort string for Anthropic: none when no level or off, else the
    # metadata-mapped value or the normalized level name.
    # KNOWN ISSUE: unlike the budget path, the level is not checked against the
    # model's available levels.
    if thinking_level is None:
        return None
    selected_model = model
    normalized = normalize_thinking_level(thinking_level)
    if normalized == "off":
        return None
    mapped = _metadata_thinking_value(provider, selected_model, normalized)
    return mapped or normalized


def _anthropic_thinking_mode(
    provider: AnthropicProviderConfig,
    model: str,
    *,
    thinking_level: ThinkingLevel | None = None,
) -> str:
    # [FUNCTION DOC]
    # HELPER: "adaptive" when compat forces adaptive thinking (or "disabled" when
    # the level is off), otherwise "budget".
    compat = _model_compat(provider, model)
    if compat.get("forceAdaptiveThinking") is True:
        if thinking_level is not None and normalize_thinking_level(thinking_level) == "off":
            return "disabled"
        return "adaptive"
    return "budget"


def _normalize_anthropic_base_url(base_url: str) -> str:
    # [FUNCTION DOC]
    # HELPER: strips trailing slashes and appends /v1 unless already present.
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def anthropic_cache_settings(
    provider: ProviderConfig,
    model: str | None = None,
    *,
    oauth: bool = False,
) -> tuple[CacheRetention, bool]:
    # [FUNCTION DOC]
    # Resolves prompt-cache retention and tool cache-control for an
    # Anthropic-protocol request: no caching when compat says cache control is
    # unsupported; otherwise long retention for OAuth or short for API keys,
    # downgraded to short when long retention is unsupported; tool cache control
    # enabled unless compat disables it. Output: retention and a flag.
    """Resolve prompt-cache settings for one Anthropic-protocol request.

    Capability comes from compat, which layers a detected default under the
    provider's own compat and then per-model compat. Intent comes from the auth
    mode: subscription OAuth is not billed per token, so it asks for the 1 hour
    TTL, while an API key keeps the shorter default. Capability only ever narrows
    intent, so the two compose without any precedence rule.
    """
    compat = _model_compat(provider, model)
    if compat.get("supportsCacheControl") is False:
        return CACHE_RETENTION_NONE, False
    retention = CACHE_RETENTION_LONG if oauth else CACHE_RETENTION_SHORT
    if retention == CACHE_RETENTION_LONG and compat.get("supportsLongCacheRetention") is False:
        retention = CACHE_RETENTION_SHORT
    return retention, compat.get("supportsCacheControlOnTools") is not False


def _provider_from_json(data: object) -> ProviderConfig:
    # [FUNCTION DOC]
    # HELPER: parses one legacy full provider record: validates type (OpenAI
    # compatible, anthropic, openai-codex, google-generative-ai,
    # mistral-conversations), name, base URL (trailing slash removed), API, key
    # env, credential name, models, default model (appended to models if
    # missing), context windows, headers, compat, metadata, numbers with defaults,
    # thinking settings and defaults, and builds the matching config. Codex
    # records reject legacy compat. Raises ProviderConfigError on invalid data.
    if not isinstance(data, dict):
        raise ProviderConfigError("Provider entries must be JSON objects")
    provider_type = _string(data.get("type"), "providers[].type")
    if provider_type not in {
        "openai-compatible",
        "anthropic",
        "openai-codex",
        "google-generative-ai",
        "mistral-conversations",
    }:
        raise ProviderConfigError(f"Unsupported provider type: {provider_type}")
    name = _string(data.get("name"), "providers[].name")
    base_url = _string(data.get("base_url"), f"providers[{name}].base_url").rstrip("/")
    api = _optional_provider_api(data.get("api"), f"providers[{name}].api")
    api_key_env = _string(data.get("api_key_env"), f"providers[{name}].api_key_env")
    credential_name = _optional_string(
        data.get("credential_name"), f"providers[{name}].credential_name"
    )
    models = _string_tuple(data.get("models"), f"providers[{name}].models")
    default_model = _string(data.get("default_model"), f"providers[{name}].default_model")
    context_windows = _context_window_dict(
        data.get("context_windows", {}), f"providers[{name}].context_windows"
    )
    headers = _string_dict(data.get("headers", {}), f"providers[{name}].headers")
    compat = _json_dict(data.get("compat", {}), f"providers[{name}].compat")
    model_metadata = _model_metadata_dict(
        data.get("model_metadata", {}),
        models,
        f"providers[{name}].model_metadata",
    )
    timeout_seconds = _positive_float(
        data.get("timeout_seconds", DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS),
        f"providers[{name}].timeout_seconds",
    )
    max_retries = _non_negative_int(
        data.get("max_retries", DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES),
        f"providers[{name}].max_retries",
    )
    max_retry_delay_seconds = _non_negative_float(
        data.get(
            "max_retry_delay_seconds",
            DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS,
        ),
        f"providers[{name}].max_retry_delay_seconds",
    )
    thinking_levels = _optional_thinking_levels(
        data.get("thinking_levels"), f"providers[{name}].thinking_levels"
    )
    thinking_models = _optional_string_tuple(
        data.get("thinking_models"), f"providers[{name}].thinking_models"
    )
    thinking_default = _optional_thinking_level(
        data.get("thinking_default"), f"providers[{name}].thinking_default"
    )
    thinking_parameter = _optional_thinking_parameter(
        data.get("thinking_parameter"), f"providers[{name}].thinking_parameter"
    )
    thinking_defaults = _raw_thinking_defaults_dict(
        data.get("thinking_defaults", {}), f"providers[{name}].thinking_defaults"
    )
    if default_model not in models:
        models = (*models, default_model)
    if provider_type == "anthropic":
        return AnthropicProviderConfig(
            name=name,
            base_url=base_url,
            api=api or "anthropic-messages",
            api_key_env=api_key_env,
            credential_name=credential_name,
            models=models,
            default_model=default_model,
            context_windows=context_windows,
            headers=headers,
            compat=compat,
            model_metadata=model_metadata,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_retry_delay_seconds=max_retry_delay_seconds,
            thinking_levels=thinking_levels,
            thinking_models=thinking_models,
            thinking_default=thinking_default,
            thinking_parameter=thinking_parameter,
            thinking_defaults=thinking_defaults,
        )
    if provider_type == "openai-codex":
        _reject_codex_legacy_compat(compat)
        return OpenAICodexProviderConfig(
            name=name,
            base_url=base_url,
            api_key_env=api_key_env,
            credential_name=credential_name,
            models=models,
            default_model=default_model,
            context_windows=context_windows,
            headers=headers,
            model_metadata=model_metadata,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_retry_delay_seconds=max_retry_delay_seconds,
            thinking_levels=thinking_levels,
            thinking_models=thinking_models,
            thinking_default=thinking_default,
            thinking_parameter=thinking_parameter,
            thinking_defaults=thinking_defaults,
        )
    return OpenAICompatibleProviderConfig(
        name=name,
        base_url=base_url,
        api=api or _default_api_for_kind(provider_type),
        api_key_env=api_key_env,
        credential_name=credential_name,
        models=models,
        default_model=default_model,
        context_windows=context_windows,
        headers=headers,
        compat=compat,
        model_metadata=model_metadata,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_retry_delay_seconds=max_retry_delay_seconds,
        thinking_levels=thinking_levels,
        thinking_models=thinking_models,
        thinking_default=thinking_default,
        thinking_parameter=thinking_parameter,
        thinking_defaults=thinking_defaults,
    )


def _api_key_from_provider(
    provider: ProviderConfig,
    *,
    credential_reader: CredentialReader | None,
) -> str:
    # [FUNCTION DOC]
    # HELPER: returns the API key: stored credential by credential name, else an
    # OAuth access token when the provider has a registered OAuth implementation,
    # else the API key environment variable. Raises RuntimeError naming the env
    # var (and a /login hint) when none is found.
    # KNOWN ISSUE: a stored OAuth access token is returned without checking
    # expiry; callers must refresh it beforehand.
    if provider.credential_name and credential_reader is not None:
        credential = credential_reader.get(provider.credential_name)
        if credential:
            return credential
        get_oauth = getattr(credential_reader, "get_oauth", None)
        if get_oauth_provider(provider.name) is not None and get_oauth is not None:
            oauth_credential = get_oauth(provider.credential_name)
            if oauth_credential is not None:
                access = getattr(oauth_credential, "access", None)
                if isinstance(access, str) and access:
                    return access

    api_key = environ.get(provider.api_key_env)
    if api_key:
        return api_key
    credential_hint = f" or run /login {provider.name}" if provider.credential_name else ""
    raise RuntimeError(f"Missing provider API key. Set {provider.api_key_env}{credential_hint}.")


def _validate_provider_numbers(
    *,
    timeout_seconds: float,
    max_retries: int,
    max_retry_delay_seconds: float,
) -> None:
    # [FUNCTION DOC]
    # HELPER: requires timeout greater than 0 (not boolean), integer retries of 0
    # or more, and numeric retry delay of 0 or more; raises ProviderConfigError.
    # KNOWN ISSUE: timeout is not type-checked as numeric, so a non-number raises
    # TypeError rather than ProviderConfigError.
    if isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise ProviderConfigError("Provider timeout_seconds must be greater than 0")
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
        raise ProviderConfigError("Provider max_retries must be 0 or greater")
    if (
        not isinstance(max_retry_delay_seconds, int | float)
        or isinstance(max_retry_delay_seconds, bool)
        or max_retry_delay_seconds < 0
    ):
        raise ProviderConfigError("Provider max_retry_delay_seconds must be 0 or greater")


def _validate_context_windows(context_windows: dict[str, int]) -> None:
    # [FUNCTION DOC]
    # HELPER: requires non-empty string keys and positive integer (non-boolean)
    # values; raises ProviderConfigError.
    for model, context_window in context_windows.items():
        if not isinstance(model, str) or not model.strip():
            raise ProviderConfigError("Provider context_windows keys must be non-empty strings")
        if (
            not isinstance(context_window, int)
            or isinstance(context_window, bool)
            or context_window <= 0
        ):
            raise ProviderConfigError("Provider context_windows values must be positive integers")


def _validate_model_metadata(
    models: tuple[str, ...],
    model_metadata: dict[str, ProviderModelMetadata],
) -> None:
    # [FUNCTION DOC]
    # HELPER: requires every metadata key to be a configured model, positive
    # context window and max tokens, inputs only text/image, non-negative costs,
    # valid cost tiers, JSON compat, non-empty string headers, valid level names
    # and non-blank string or null level values. Raises ProviderConfigError.
    # KNOWN ISSUE: an invalid thinking level name raises the raw ValueError from
    # normalization instead of ProviderConfigError.
    model_names = set(models)
    for model, metadata in model_metadata.items():
        if model not in model_names:
            raise ProviderConfigError(f"Provider model_metadata key is not in models: {model}")
        if metadata.context_window is not None and metadata.context_window <= 0:
            raise ProviderConfigError("Provider model_metadata context_window must be positive")
        if metadata.max_tokens is not None and metadata.max_tokens <= 0:
            raise ProviderConfigError("Provider model_metadata max_tokens must be positive")
        if any(item not in {"text", "image"} for item in metadata.input):
            raise ProviderConfigError("Provider model_metadata input must contain text or image")
        if any(value < 0 for value in metadata.cost.values()):
            raise ProviderConfigError("Provider model_metadata cost values must be non-negative")
        _validate_runtime_cost_tiers(metadata.cost_tiers)
        _validate_json_object(metadata.compat, "Provider model_metadata compat")
        _validate_string_dict(metadata.headers, "Provider model_metadata headers")
        for level, value in metadata.thinking_level_map.items():
            normalize_thinking_level(level)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ProviderConfigError(
                    "Provider model_metadata thinking_level_map values must be strings or null"
                )


def _validate_runtime_cost_tiers(tiers: tuple[ModelCostTier, ...]) -> None:
    # [FUNCTION DOC]
    # HELPER: requires the final tier to omit max_input_tokens, non-negative tier
    # rates and strictly increasing limits; raises ProviderConfigError.
    if tiers and tiers[-1].max_input_tokens is not None:
        raise ProviderConfigError(
            "Provider model_metadata final cost tier must omit max_input_tokens"
        )
    previous_limit = 0
    for tier in tiers:
        if any(value < 0 for value in tier.cost.values()):
            raise ProviderConfigError(
                "Provider model_metadata cost tier values must be non-negative"
            )
        if tier.max_input_tokens is None:
            continue
        if tier.max_input_tokens <= previous_limit:
            raise ProviderConfigError(
                "Provider model_metadata cost tier limits must be strictly increasing"
            )
        previous_limit = tier.max_input_tokens


def _validate_string_dict(value: dict[str, str], field_name: str) -> None:
    # [FUNCTION DOC]
    # HELPER: requires non-empty string keys and values; raises
    # ProviderConfigError with the field name.
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ProviderConfigError(f"{field_name} keys must be non-empty strings")
        if not isinstance(item, str) or not item.strip():
            raise ProviderConfigError(f"{field_name} values must be non-empty strings")


def _validate_json_object(value: dict[str, Any], field_name: str) -> None:
    # [FUNCTION DOC]
    # HELPER: requires non-empty string keys and JSON-compatible values; raises
    # ProviderConfigError.
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ProviderConfigError(f"{field_name} keys must be non-empty strings")
        _validate_json_value(item, f"{field_name}.{key}")


def _validate_json_value(value: object, field_name: str) -> None:
    # [FUNCTION DOC]
    # HELPER: recursively accepts null, strings, numbers, booleans, lists and
    # string-keyed mappings; raises ProviderConfigError otherwise.
    if value is None or isinstance(value, str | int | float | bool):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, field_name)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProviderConfigError(f"{field_name} object keys must be strings")
            _validate_json_value(item, f"{field_name}.{key}")
        return
    raise ProviderConfigError(f"{field_name} must be JSON-compatible")


def _reject_codex_legacy_compat(compat: dict[str, Any]) -> None:
    # [FUNCTION DOC]
    # HELPER: raises ProviderConfigError when a legacy Codex record has compat.
    if compat:
        raise ProviderConfigError("OpenAI Codex legacy provider compat is not supported")


_HF_INFERENCE_PROVIDER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_huggingface_inference_provider(value: str) -> str:
    # [FUNCTION DOC]
    # Strips and validates an explicit Hugging Face inference provider suffix:
    # must start alphanumeric and contain only letters, digits, dot, underscore
    # or dash, and must not be a routing policy (fastest, cheapest, preferred).
    # Output: the normalized value. Raises ProviderConfigError.
    """Return a normalized explicit Hugging Face provider suffix."""
    normalized = value.strip()
    if not _HF_INFERENCE_PROVIDER_PATTERN.fullmatch(normalized):
        raise ProviderConfigError(
            "Hugging Face inference provider must contain only letters, numbers, '.', '_', or '-'"
        )
    if normalized in {"fastest", "cheapest", "preferred"}:
        raise ProviderConfigError(
            f"Hugging Face inference provider must be explicit, not routing policy: {normalized}"
        )
    return normalized


def _validate_inference_providers(
    provider_name: str,
    models: tuple[str, ...],
    inference_providers: dict[str, str],
) -> None:
    # [FUNCTION DOC]
    # HELPER: inference routes are allowed only for the huggingface provider;
    # each must reference a configured model and be a valid explicit route.
    # Raises ProviderConfigError.
    if inference_providers and provider_name != "huggingface":
        raise ProviderConfigError(
            "inference_providers preferences are only supported for the huggingface provider"
        )
    for model, route in inference_providers.items():
        if model not in models:
            raise ProviderConfigError(
                f"Inference-provider preference references unknown model: {model}"
            )
        validate_huggingface_inference_provider(route)


def _validate_thinking_defaults(thinking_defaults: dict[str, ThinkingLevel]) -> None:
    # [FUNCTION DOC]
    # HELPER: requires non-empty model keys and valid thinking level names;
    # raises ProviderConfigError.
    for model, thinking_level in thinking_defaults.items():
        if not isinstance(model, str) or not model.strip():
            raise ProviderConfigError("Provider thinking_defaults keys must be non-empty strings")
        try:
            normalize_thinking_level(thinking_level)
        except ValueError as exc:
            raise ProviderConfigError(str(exc)) from exc


def _validate_thinking_config(
    *,
    thinking_levels: tuple[ThinkingLevel, ...] | None,
    thinking_models: tuple[str, ...],
    thinking_default: ThinkingLevel | None,
    thinking_parameter: ThinkingParameter | None,
) -> None:
    # [FUNCTION DOC]
    # HELPER: when levels are unset, other thinking fields must be unset too;
    # otherwise levels must already be normalized, thinking models non-empty
    # strings, the default among the levels and the parameter one of
    # reasoning_effort, reasoning.effort or anthropic.thinking. Raises
    # ProviderConfigError.
    if thinking_levels is None:
        if thinking_models or thinking_default is not None or thinking_parameter is not None:
            raise ProviderConfigError(
                "Provider thinking_levels must be set before thinking metadata"
            )
        return
    try:
        normalized = normalize_thinking_levels(thinking_levels)
    except ValueError as exc:
        raise ProviderConfigError(str(exc)) from exc
    if normalized != thinking_levels:
        raise ProviderConfigError("Provider thinking_levels must be normalized")
    if any(not isinstance(model, str) or not model.strip() for model in thinking_models):
        raise ProviderConfigError("Provider thinking_models must contain non-empty strings")
    if thinking_default is not None and thinking_default not in thinking_levels:
        raise ProviderConfigError("Provider thinking_default must be in thinking_levels")
    if thinking_parameter not in {
        None,
        "reasoning_effort",
        "reasoning.effort",
        "anthropic.thinking",
    }:
        raise ProviderConfigError(
            "Provider thinking_parameter must be reasoning_effort, reasoning.effort, "
            "or anthropic.thinking"
        )


def _reject_unimplemented_thinking_config(
    *,
    provider_type: str,
    thinking_levels: tuple[ThinkingLevel, ...] | None,
) -> None:
    # [FUNCTION DOC]
    # HELPER: raises ProviderConfigError when thinking levels are set for a
    # provider type without thinking support. Currently unused in this module.
    if thinking_levels is not None:
        raise ProviderConfigError(f"{provider_type} thinking controls are not implemented yet")


def _optional_provider_api(value: object, field_name: str) -> ProviderApi | None:
    # [FUNCTION DOC]
    # HELPER: returns none or a supported API name; raises ProviderConfigError
    # for anything else.
    if value is None:
        return None
    if value in {
        "openai-completions",
        "openai-responses",
        "anthropic-messages",
        "openai-codex-responses",
        "google-generative-ai",
        "mistral-conversations",
    }:
        return cast(ProviderApi, value)
    raise ProviderConfigError(f"Provider field has unsupported API: {field_name}")


def _optional_string(value: object, field_name: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: returns none or the stripped non-empty string; raises
    # ProviderConfigError otherwise.
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProviderConfigError(f"Provider field must be a non-empty string: {field_name}")
    return value.strip()


def _string(value: object, field_name: str) -> str:
    # [FUNCTION DOC]
    # HELPER: returns the stripped non-empty string or raises
    # ProviderConfigError naming the field.
    if not isinstance(value, str) or not value.strip():
        raise ProviderConfigError(f"Provider field must be a non-empty string: {field_name}")
    return value.strip()


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    # [FUNCTION DOC]
    # HELPER: requires a non-empty list of non-blank strings; returns them
    # stripped. Raises ProviderConfigError.
    if not isinstance(value, list) or not value:
        raise ProviderConfigError(f"Provider field must be a non-empty string list: {field_name}")
    items = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if len(items) != len(value):
        raise ProviderConfigError(f"Provider field must be a string list: {field_name}")
    return items


def _optional_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    # [FUNCTION DOC]
    # HELPER: none gives an empty tuple; otherwise requires a list of non-blank
    # strings and returns them stripped. Raises ProviderConfigError.
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ProviderConfigError(f"Provider field must be a string list: {field_name}")
    items = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if len(items) != len(value):
        raise ProviderConfigError(f"Provider field must be a string list: {field_name}")
    return items


def _optional_thinking_levels(
    value: object,
    field_name: str,
) -> tuple[ThinkingLevel, ...] | None:
    # [FUNCTION DOC]
    # HELPER: none or a normalized tuple of levels from a list; raises
    # ProviderConfigError for non-lists or invalid levels.
    if value is None:
        return None
    if not isinstance(value, list):
        raise ProviderConfigError(f"Provider field must be a thinking mode list: {field_name}")
    try:
        return normalize_thinking_levels(value)
    except ValueError as exc:
        raise ProviderConfigError(str(exc)) from exc


def _optional_thinking_level(value: object, field_name: str) -> ThinkingLevel | None:
    # [FUNCTION DOC]
    # HELPER: none or a normalized level from a string; raises
    # ProviderConfigError otherwise.
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProviderConfigError(f"Provider field must be a thinking mode: {field_name}")
    try:
        return normalize_thinking_level(value)
    except ValueError as exc:
        raise ProviderConfigError(str(exc)) from exc


def _optional_thinking_parameter(
    value: object,
    field_name: str,
) -> ThinkingParameter | None:
    # [FUNCTION DOC]
    # HELPER: none or one of the three supported thinking parameter names;
    # raises ProviderConfigError otherwise.
    if value is None:
        return None
    if value == "reasoning_effort":
        return "reasoning_effort"
    if value == "reasoning.effort":
        return "reasoning.effort"
    if value == "anthropic.thinking":
        return "anthropic.thinking"
    raise ProviderConfigError(
        f"Provider field must be reasoning_effort, reasoning.effort, "
        f"or anthropic.thinking: {field_name}"
    )


def _string_dict(value: object, field_name: str) -> dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: parses a mapping of non-blank strings to non-blank strings,
    # stripping keys and values; raises ProviderConfigError.
    if not isinstance(value, dict):
        raise ProviderConfigError(f"Provider field must be a string object: {field_name}")
    items: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ProviderConfigError(f"Provider field must be a string object: {field_name}")
        if not isinstance(item, str) or not item.strip():
            raise ProviderConfigError(f"Provider field must be a string object: {field_name}")
        items[key.strip()] = item.strip()
    return items


def _json_dict(value: object, field_name: str) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: parses a mapping with non-blank string keys (stripped) and
    # JSON-compatible values; raises ProviderConfigError.
    if not isinstance(value, dict):
        raise ProviderConfigError(f"Provider field must be an object: {field_name}")
    items: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ProviderConfigError(f"Provider field must have string keys: {field_name}")
        _validate_json_value(item, f"{field_name}.{key}")
        items[key.strip()] = item
    return items


def _model_metadata_dict(
    value: object,
    models: tuple[str, ...],
    field_name: str,
) -> dict[str, ProviderModelMetadata]:
    # [FUNCTION DOC]
    # HELPER: parses legacy model metadata objects keyed by configured model,
    # validating each field (name, API, base URL, reasoning, input, cost, cost
    # tiers, context window, max tokens, headers, compat, level map). Output: a
    # mapping of model to metadata. Raises ProviderConfigError.
    if not isinstance(value, dict):
        raise ProviderConfigError(f"Provider field must be an object: {field_name}")
    model_names = set(models)
    items: dict[str, ProviderModelMetadata] = {}
    for key, item in value.items():
        model = _string(key, field_name)
        if model not in model_names:
            raise ProviderConfigError(f"Provider model_metadata key is not in models: {model}")
        if not isinstance(item, dict):
            raise ProviderConfigError(
                f"Provider model_metadata entries must be objects: {field_name}"
            )
        items[model] = ProviderModelMetadata(
            name=_optional_string(item.get("name"), f"{field_name}.{model}.name"),
            api=_optional_provider_api(item.get("api"), f"{field_name}.{model}.api"),
            base_url=_optional_string(item.get("base_url"), f"{field_name}.{model}.base_url"),
            reasoning=_optional_bool(item.get("reasoning"), f"{field_name}.{model}.reasoning"),
            input=_optional_string_tuple(item.get("input"), f"{field_name}.{model}.input"),
            cost=_float_dict(item.get("cost", {}), f"{field_name}.{model}.cost"),
            cost_tiers=_cost_tiers(item.get("cost_tiers", []), f"{field_name}.{model}.cost_tiers"),
            context_window=_optional_positive_int(
                item.get("context_window"), f"{field_name}.{model}.context_window"
            ),
            max_tokens=_optional_positive_int(
                item.get("max_tokens"), f"{field_name}.{model}.max_tokens"
            ),
            headers=_string_dict(item.get("headers", {}), f"{field_name}.{model}.headers"),
            compat=_json_dict(item.get("compat", {}), f"{field_name}.{model}.compat"),
            thinking_level_map=_thinking_level_map_dict(
                item.get("thinking_level_map", {}),
                f"{field_name}.{model}.thinking_level_map",
            ),
        )
    return items


def _cost_tiers(value: object, field_name: str) -> tuple[ModelCostTier, ...]:
    # [FUNCTION DOC]
    # HELPER: parses a list of cost tier objects with only known fields;
    # input/output/cacheRead/cacheWrite are required non-negative numbers,
    # cacheWrite1h optional, max_input_tokens optional positive. Validates tier
    # ordering. Raises ProviderConfigError.
    if not isinstance(value, list):
        raise ProviderConfigError(f"Provider field must be an array: {field_name}")
    tiers: list[ModelCostTier] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ProviderConfigError(f"Provider cost tiers must be objects: {field_name}")
        tier_field = f"{field_name}.{index}"
        allowed = {
            "max_input_tokens",
            "input",
            "output",
            "cacheRead",
            "cacheWrite",
            "cacheWrite1h",
        }
        if set(item) - allowed:
            raise ProviderConfigError(f"Provider cost tier has unknown fields: {tier_field}")
        cost = {
            key: _non_negative_float(item.get(key), f"{tier_field}.{key}")
            for key in ("input", "output", "cacheRead", "cacheWrite")
        }
        if item.get("cacheWrite1h") is not None:
            cost["cacheWrite1h"] = _non_negative_float(
                item.get("cacheWrite1h"), f"{tier_field}.cacheWrite1h"
            )
        tiers.append(
            ModelCostTier(
                max_input_tokens=_optional_positive_int(
                    item.get("max_input_tokens"), f"{tier_field}.max_input_tokens"
                ),
                cost=cost,
            )
        )
    result = tuple(tiers)
    _validate_runtime_cost_tiers(result)
    return result


def _thinking_level_map_dict(
    value: object,
    field_name: str,
) -> dict[ThinkingLevel, str | None]:
    # [FUNCTION DOC]
    # HELPER: parses a mapping of level name to non-blank string (stripped) or
    # null; raises ProviderConfigError.
    if not isinstance(value, dict):
        raise ProviderConfigError(f"Provider field must be an object: {field_name}")
    items: dict[ThinkingLevel, str | None] = {}
    for key, item in value.items():
        level = _optional_thinking_level(key, field_name)
        if level is None:
            raise ProviderConfigError(f"Provider field must be a thinking mode: {field_name}")
        if item is not None and (not isinstance(item, str) or not item.strip()):
            raise ProviderConfigError(
                f"Provider field values must be strings or null: {field_name}"
            )
        items[level] = item.strip() if isinstance(item, str) else None
    return items


def _float_dict(value: object, field_name: str) -> dict[str, float]:
    # [FUNCTION DOC]
    # HELPER: parses a mapping of non-blank keys (stripped) to non-negative
    # numbers converted to floats; raises ProviderConfigError.
    if not isinstance(value, dict):
        raise ProviderConfigError(f"Provider field must be a number object: {field_name}")
    items: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ProviderConfigError(f"Provider field must be a number object: {field_name}")
        if not isinstance(item, int | float) or isinstance(item, bool) or item < 0:
            raise ProviderConfigError(f"Provider field values must be non-negative: {field_name}")
        items[key.strip()] = float(item)
    return items


def _optional_bool(value: object, field_name: str) -> bool | None:
    # [FUNCTION DOC]
    # HELPER: none or a boolean; raises ProviderConfigError otherwise.
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ProviderConfigError(f"Provider field must be a boolean: {field_name}")
    return value


def _optional_positive_int(value: object, field_name: str) -> int | None:
    # [FUNCTION DOC]
    # HELPER: none or a positive non-boolean integer; raises ProviderConfigError.
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ProviderConfigError(f"Provider field must be a positive integer: {field_name}")
    return value


def _context_window_dict(value: object, field_name: str) -> dict[str, int]:
    # [FUNCTION DOC]
    # HELPER: parses a mapping of non-blank keys (stripped) to positive integers;
    # raises ProviderConfigError.
    if not isinstance(value, dict):
        raise ProviderConfigError(f"Provider field must be an integer object: {field_name}")
    items: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ProviderConfigError(f"Provider field must be an integer object: {field_name}")
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise ProviderConfigError(
                f"Provider field values must be positive integers: {field_name}"
            )
        items[key.strip()] = item
    return items


def _positive_float(value: object, field_name: str) -> float:
    # [FUNCTION DOC]
    # HELPER: numeric non-boolean value greater than 0 converted to float; raises
    # ProviderConfigError.
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ProviderConfigError(f"Provider field must be a positive number: {field_name}")
    converted = float(value)
    if converted <= 0:
        raise ProviderConfigError(f"Provider field must be greater than 0: {field_name}")
    return converted


def _non_negative_int(value: object, field_name: str) -> int:
    # [FUNCTION DOC]
    # HELPER: non-boolean integer of 0 or more; raises ProviderConfigError.
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProviderConfigError(f"Provider field must be a non-negative integer: {field_name}")
    if value < 0:
        raise ProviderConfigError(f"Provider field must be 0 or greater: {field_name}")
    return value


def _non_negative_float(value: object, field_name: str) -> float:
    # [FUNCTION DOC]
    # HELPER: numeric non-boolean value of 0 or more converted to float; raises
    # ProviderConfigError.
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ProviderConfigError(f"Provider field must be a non-negative number: {field_name}")
    converted = float(value)
    if converted < 0:
        raise ProviderConfigError(f"Provider field must be 0 or greater: {field_name}")
    return converted
