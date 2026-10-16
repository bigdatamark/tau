# [MODULE DOC]
# Legacy, provider-neutral streaming event models (tau_ai, private module).
# Each event is a strict Pydantic model (extra fields forbidden) discriminated by a
# literal "type" field. Adapters emit these while streaming a single model response;
# the union alias at the bottom names all of them. No functions; no side effects.
"""Provider-neutral streaming events emitted by model adapters."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from tau_agent.messages import AssistantMessage
from tau_agent.tools import ToolCall
from tau_agent.types import JSONValue


class ProviderResponseStartEvent(BaseModel):
    # [CLASS DOC]
    # Event "response_start": a model response began. Fields: model id and optional
    # name of the upstream provider that actually served the response (for routers).
    """The provider has started a model response."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["response_start"] = "response_start"
    model: str
    response_provider: str | None = None


class ProviderRetryEvent(BaseModel):
    # [CLASS DOC]
    # Event "retry": the adapter will retry a transient failure. Fields: attempt number,
    # maximum attempts, delay in seconds before the retry, human message, and optional
    # JSON-compatible extra data mapping.
    """The provider adapter is retrying a transient request failure."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["retry"] = "retry"
    attempt: int
    max_attempts: int
    delay_seconds: float
    message: str
    data: dict[str, JSONValue] | None = None


class ProviderTextDeltaEvent(BaseModel):
    # [CLASS DOC]
    # Event "text_delta": one fragment of visible assistant text (delta string).
    """A streamed text fragment from the provider."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text_delta"] = "text_delta"
    delta: str


class ProviderThinkingDeltaEvent(BaseModel):
    # [CLASS DOC]
    # Event "thinking_delta": one fragment of reasoning/thinking text (delta string).
    """A streamed thinking/reasoning fragment from the provider."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["thinking_delta"] = "thinking_delta"
    delta: str


class ProviderToolCallEvent(BaseModel):
    # [CLASS DOC]
    # Event "tool_call": a fully assembled tool call (id, name, arguments) requested by
    # the model.
    """A complete tool call requested by the model."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_call"] = "tool_call"
    tool_call: ToolCall


class ProviderResponseEndEvent(BaseModel):
    # [CLASS DOC]
    # Event "response_end": the response completed. Fields: the final assistant message
    # and the optional raw provider finish reason.
    """The provider has completed a model response."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["response_end"] = "response_end"
    message: AssistantMessage
    finish_reason: str | None = None


class ProviderErrorEvent(BaseModel):
    # [CLASS DOC]
    # Event "error": a provider-level failure to surface to the agent. Fields: message,
    # optional JSON-compatible data mapping, optional upstream provider name.
    """A provider-level error that can be surfaced by the agent layer."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["error"] = "error"
    message: str
    data: dict[str, JSONValue] | None = None
    response_provider: str | None = None


type ProviderEvent = (
    ProviderResponseStartEvent
    | ProviderRetryEvent
    | ProviderTextDeltaEvent
    | ProviderThinkingDeltaEvent
    | ProviderToolCallEvent
    | ProviderResponseEndEvent
    | ProviderErrorEvent
)
