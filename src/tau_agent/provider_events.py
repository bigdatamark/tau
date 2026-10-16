# [MODULE DOC]
# Canonical assistant streaming event contract owned by the portable agent
# layer (re-exported by tau_ai.events). Providers translate their wire streams
# into these events; the agent loop consumes them to build AssistantMessages.
# Behavior: defines pydantic wire models each tagged by a literal "type" field,
# mirroring Pi's assistant message event protocol: start, text_start/delta/end,
# thinking_start/delta/end, toolcall_start/delta/end, done and error. Every
# incremental event carries the content block index and the current partial
# AssistantMessage snapshot. Also defines DoneReason ("stop", "length",
# "toolUse"), ErrorReason ("aborted", "error") and the AssistantMessageEvent
# discriminated union over all event models.
# Side effects: none beyond class definition.
"""Pi-compatible assistant stream events owned by the portable agent layer."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from tau_agent.messages import AssistantMessage, ToolCall, WireModel


class AssistantStartEvent(WireModel):
    # [CLASS DOC]
    # "start" event: first event of a stream, carrying the initial partial
    # assistant message.
    type: Literal["start"] = "start"
    partial: AssistantMessage


class TextStartEvent(WireModel):
    # [CLASS DOC]
    # "text_start" event: a text content block opened at the given content index;
    # carries the partial message.
    type: Literal["text_start"] = "text_start"
    content_index: int
    partial: AssistantMessage


class TextDeltaEvent(WireModel):
    # [CLASS DOC]
    # "text_delta" event: appended text fragment for the block at content index,
    # plus the updated partial message.
    type: Literal["text_delta"] = "text_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class TextEndEvent(WireModel):
    # [CLASS DOC]
    # "text_end" event: text block at content index finished; carries the full
    # block text and the partial message.
    type: Literal["text_end"] = "text_end"
    content_index: int
    content: str
    partial: AssistantMessage


class ThinkingStartEvent(WireModel):
    # [CLASS DOC]
    # "thinking_start" event: a reasoning/thinking block opened at content index.
    type: Literal["thinking_start"] = "thinking_start"
    content_index: int
    partial: AssistantMessage


class ThinkingDeltaEvent(WireModel):
    # [CLASS DOC]
    # "thinking_delta" event: appended thinking fragment for the block at content
    # index, plus partial message.
    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class ThinkingEndEvent(WireModel):
    # [CLASS DOC]
    # "thinking_end" event: thinking block finished with its complete content text.
    type: Literal["thinking_end"] = "thinking_end"
    content_index: int
    content: str
    partial: AssistantMessage


class ToolCallStartEvent(WireModel):
    # [CLASS DOC]
    # "toolcall_start" event: a tool call block opened at content index.
    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int
    partial: AssistantMessage


class ToolCallDeltaEvent(WireModel):
    # [CLASS DOC]
    # "toolcall_delta" event: raw fragment of streamed tool-call arguments (JSON
    # text) for the block at content index.
    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class ToolCallEndEvent(WireModel):
    # [CLASS DOC]
    # "toolcall_end" event: tool call block complete; carries the parsed ToolCall
    # and the partial message.
    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage


DoneReason = Literal["stop", "length", "toolUse"]
ErrorReason = Literal["aborted", "error"]


class AssistantDoneEvent(WireModel):
    # [CLASS DOC]
    # "done" event: terminal success event with the stop reason (stop, length or
    # toolUse) and the final AssistantMessage.
    type: Literal["done"] = "done"
    reason: DoneReason
    message: AssistantMessage


class AssistantErrorEvent(WireModel):
    # [CLASS DOC]
    # "error" event: terminal failure event with reason (aborted or error) and an
    # AssistantMessage describing the error (typically carrying the error message
    # and stop reason).
    type: Literal["error"] = "error"
    reason: ErrorReason
    error: AssistantMessage


type AssistantMessageEvent = Annotated[
    AssistantStartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | AssistantDoneEvent
    | AssistantErrorEvent,
    Field(discriminator="type"),
]
