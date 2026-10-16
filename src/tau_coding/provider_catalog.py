# [MODULE DOC]
# Built-in provider catalog types for Tau's login/setup flows. Defines type
# aliases for provider kinds, wire APIs, model inputs, thinking-level maps and
# auth methods; dataclasses describing cost tiers, per-model metadata and
# provider entries; and loads the bundled catalog at import time (via the
# catalog loader) into BUILTIN_PROVIDER_CATALOG.
# Side effect: importing this module loads the catalog data.
"""Built-in provider catalog for Tau login/setup flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from tau_agent.types import JSONValue
from tau_coding.thinking import ThinkingLevel, ThinkingParameter

ProviderKind = Literal[
    "openai-compatible",
    "anthropic",
    "openai-codex",
    "google-generative-ai",
    "mistral-conversations",
]
ProviderApi = Literal[
    "openai-completions",
    "openai-responses",
    "anthropic-messages",
    "openai-codex-responses",
    "google-generative-ai",
    "mistral-conversations",
]
ModelInput = Literal["text", "image"]
ThinkingLevelMap = dict[ThinkingLevel, str | None]
AuthMethod = Literal["api_key", "oauth"]


@dataclass(frozen=True, slots=True)
class ModelCostTier:
    # [CLASS DOC]
    # Frozen dataclass: a cost mapping (rate name -> price) that applies to
    # requests whose input tokens are up to an optional max (None = unbounded).
    """Model rates that apply up to an optional input-token limit."""

    cost: dict[str, float]
    max_input_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ModelCatalogMetadata:
    # [CLASS DOC]
    # Frozen dataclass of per-model catalog metadata: display name, wire API,
    # base URL override, reasoning support, supported input kinds, flat cost,
    # ordered cost tiers, context window, max output tokens, extra headers,
    # compatibility flags, and a map of thinking level to provider-specific value.
    # All fields optional / defaulted.
    """Provider-catalog metadata for a single model."""

    name: str | None = None
    api: ProviderApi | None = None
    base_url: str | None = None
    reasoning: bool | None = None
    input: tuple[ModelInput, ...] = ()
    cost: dict[str, float] | None = None
    cost_tiers: tuple[ModelCostTier, ...] = ()
    context_window: int | None = None
    max_tokens: int | None = None
    headers: dict[str, str] = field(default_factory=dict)
    compat: dict[str, JSONValue] = field(default_factory=dict)
    thinking_level_map: ThinkingLevelMap = field(default_factory=dict)


def model_cost_for_input_tokens(
    metadata: ModelCatalogMetadata,
    input_tokens: int,
) -> dict[str, float] | None:
    # [FUNCTION DOC]
    # Pick the applicable rate mapping for a request size. Inputs: model metadata
    # and an input token count. Iterates cost tiers in order and returns a copy of
    # the first tier whose max is unset or not less than the count; otherwise
    # returns a copy of the flat cost, or None when none is defined.
    # Raises ValueError if the count is not a non-negative int (bools rejected).
    # No side effects.
    """Return model rates for an input size, falling back to the flat base cost."""
    if not isinstance(input_tokens, int) or isinstance(input_tokens, bool) or input_tokens < 0:
        raise ValueError("input_tokens must be a non-negative integer")
    for tier in metadata.cost_tiers:
        if tier.max_input_tokens is None or input_tokens <= tier.max_input_tokens:
            return dict(tier.cost)
    return dict(metadata.cost) if metadata.cost is not None else None


@dataclass(frozen=True, slots=True)
class ProviderCatalogEntry:
    # [CLASS DOC]
    # Frozen dataclass describing one built-in provider shown during login: id
    # name, display name, provider kind, base URL, API key env var, credential
    # name, model ids, default model, docs URL, optional wire API, context windows,
    # headers, compat flags, per-model metadata, thinking levels / thinking-capable
    # models / default level / parameter style, removed models, and supported auth
    # methods (default api key only).
    """A built-in provider Tau can present during login."""

    name: str
    display_name: str
    kind: ProviderKind
    base_url: str
    api_key_env: str
    credential_name: str | None
    models: tuple[str, ...]
    default_model: str
    docs_url: str
    api: ProviderApi | None = None
    context_windows: dict[str, int] | None = None
    headers: dict[str, str] = field(default_factory=dict)
    compat: dict[str, JSONValue] = field(default_factory=dict)
    model_metadata: dict[str, ModelCatalogMetadata] = field(default_factory=dict)
    thinking_levels: tuple[ThinkingLevel, ...] | None = None
    thinking_models: tuple[str, ...] = ()
    thinking_default: ThinkingLevel | None = None
    thinking_parameter: ThinkingParameter | None = None
    removed_models: tuple[str, ...] = ()
    auth_methods: tuple[AuthMethod, ...] = ("api_key",)


def _load_builtin_catalog() -> tuple[ProviderCatalogEntry, ...]:
    # [FUNCTION DOC]
    # HELPER: lazily import the catalog loader (avoiding a circular import, since
    # the loader imports this module's types) and return the built-in catalog
    # tuple of provider entries. Propagates loader errors.
    # Imported lazily: catalog_loader imports ProviderCatalogEntry from this module.
    from tau_coding.catalog_loader import builtin_catalog

    return builtin_catalog()


BUILTIN_PROVIDER_CATALOG: tuple[ProviderCatalogEntry, ...] = _load_builtin_catalog()


def builtin_provider_entry(name: str) -> ProviderCatalogEntry | None:
    # [FUNCTION DOC]
    # Return the built-in catalog entry whose name equals the given provider name
    # by linear search, or None if absent. No side effects.
    """Return a built-in catalog entry by provider name."""
    for entry in BUILTIN_PROVIDER_CATALOG:
        if entry.name == name:
            return entry
    return None
