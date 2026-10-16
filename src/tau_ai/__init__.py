# [PACKAGE DOC]
# Public facade of the tau_ai package: the provider/model streaming layer. It re-exports provider
# implementations (Anthropic, OpenAI-compatible, OpenAI Codex, Google Generative AI, Mistral,
# Fake), provider configuration/env helpers and defaults, assistant streaming event types, model
# catalog/limits providers, and the provider protocol plus cancellation token.
# Inputs/outputs: none at runtime beyond imports; the public export list is computed as every
# module-level global name not starting with an underscore (including imported submodule-level
# names).
# Side effects: importing this package imports all provider submodules. No exceptions raised
# beyond import errors.
"""Provider and Pi-compatible model streaming layer for Tau."""

# ruff: noqa: F401 - this module intentionally defines the public facade

from tau_ai.anthropic import AnthropicProvider
from tau_ai.env import (
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES,
    DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS,
    DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
    AnthropicConfig,
    OpenAICompatibleConfig,
    RuntimeProviderAuth,
    openai_compatible_config_from_env,
)
from tau_ai.events import (
    AssistantDoneEvent,
    AssistantErrorEvent,
    AssistantMessageEvent,
    AssistantStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from tau_ai.fake import FakeProvider
from tau_ai.google import GoogleGenerativeAIProvider
from tau_ai.mistral import MistralConversationsProvider
from tau_ai.model_catalog import ModelCatalogProvider, RuntimeModel, RuntimeModelCatalog
from tau_ai.model_limits import ModelLimitsProvider, RuntimeModelLimits
from tau_ai.openai_codex import (
    DEFAULT_OPENAI_CODEX_BASE_URL,
    OpenAICodexConfig,
    OpenAICodexCredentials,
    OpenAICodexProvider,
)
from tau_ai.openai_compatible import OpenAICompatibleProvider
from tau_ai.provider import CancellationToken, ModelProvider

__all__ = [name for name in globals() if not name.startswith("_")]
