# [MODULE DOC]
# Process-global registry of OAuth (subscription login) providers. It is
# pre-populated with the built-in Anthropic, GitHub Copilot and OpenAI Codex
# providers and lets extensions register, replace or remove providers by their
# stable ID. State: a module-level mapping from provider ID to provider object,
# plus an immutable tuple of the built-in instances used for restore/reset.
# Side effects: importing instantiates the three built-in providers.
"""Built-in and extension-ready OAuth provider registry."""

from __future__ import annotations

from collections.abc import Iterable

from tau_coding.oauth import OpenAICodexOAuthProvider
from tau_coding.oauth_anthropic import AnthropicOAuthProvider
from tau_coding.oauth_github_copilot import GitHubCopilotOAuthProvider
from tau_coding.oauth_types import OAuthProvider

_BUILTIN_PROVIDERS: tuple[OAuthProvider, ...] = tuple(
    [AnthropicOAuthProvider(), GitHubCopilotOAuthProvider(), OpenAICodexOAuthProvider()]
)
_registry: dict[str, OAuthProvider] = {provider.id: provider for provider in _BUILTIN_PROVIDERS}


def get_oauth_provider(provider_id: str) -> OAuthProvider | None:
    # [FUNCTION DOC]
    # Look up a registered OAuth provider.
    # Inputs: the provider ID string. Outputs: the provider, or None if no provider
    # is registered under that ID. Raises nothing; no side effects.
    """Return a registered OAuth provider by stable provider ID."""
    return _registry.get(provider_id)


def get_oauth_providers() -> tuple[OAuthProvider, ...]:
    # [FUNCTION DOC]
    # Return a tuple snapshot of all registered providers in registration
    # (insertion) order. No inputs; no side effects; raises nothing.
    """Return all registered OAuth providers in registration order."""
    return tuple(_registry.values())


def oauth_provider_ids() -> frozenset[str]:
    # [FUNCTION DOC]
    # Return a frozen set of the currently registered provider IDs, used to validate
    # subscription login targets. No inputs; no side effects; raises nothing.
    """Return IDs accepted by Tau's subscription login flow."""
    return frozenset(_registry)


def register_oauth_provider(provider: OAuthProvider) -> None:
    # [FUNCTION DOC]
    # Register a provider, replacing any existing provider with the same ID (built-in
    # or custom). Inputs: a provider object with an id attribute.
    # Raises: ValueError if the ID is empty or whitespace-only.
    # Outputs: none. Side effects: mutates the global registry. A newly added ID is
    # appended in order; a replaced ID keeps its original position.
    """Register or replace an OAuth provider implementation."""
    if not provider.id.strip():
        raise ValueError("OAuth provider id must not be empty")
    _registry[provider.id] = provider


def unregister_oauth_provider(provider_id: str) -> None:
    # [FUNCTION DOC]
    # Remove a provider by ID. If the ID belongs to a built-in provider, the original
    # built-in instance is restored instead of being removed; otherwise the entry is
    # deleted (silently ignoring unknown IDs).
    # Inputs: the provider ID. Outputs: none. Raises nothing.
    # Side effects: mutates the global registry.
    """Remove a custom provider or restore a replaced built-in provider."""
    builtin = next(
        (provider for provider in _BUILTIN_PROVIDERS if provider.id == provider_id),
        None,
    )
    if builtin is None:
        _registry.pop(provider_id, None)
    else:
        _registry[provider_id] = builtin


def reset_oauth_providers(providers: Iterable[OAuthProvider] = _BUILTIN_PROVIDERS) -> None:
    # [FUNCTION DOC]
    # Clear the registry and repopulate it from the given providers (defaults to the
    # built-in tuple), keyed by each provider's ID in iteration order; later
    # duplicates win. Mainly used by tests for determinism.
    # Inputs: an iterable of providers. Outputs: none. Raises nothing (no empty-ID
    # validation is performed here). Side effects: replaces global registry contents.
    """Reset the registry, primarily for deterministic extension tests."""
    _registry.clear()
    _registry.update((provider.id, provider) for provider in providers)
