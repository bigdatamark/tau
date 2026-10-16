# [MODULE DOC]
# Pi-compatible event models emitted by Tau's portable agent loop. UI layers,
# print mode and RPC consume these instead of talking to the loop directly.
# Each event is a wire model (pydantic) with a literal "type" discriminator; the
# AgentEvent alias is the discriminated union of all of them, so serialized
# events can be parsed back into the correct class by their "type" field.
# Lifecycle order: agent_start, then per turn turn_start, message_start /
# message_update* / message_end for each message, tool_execution_start /
# update* / end for each tool call, turn_end, and finally agent_end.
# No behavior beyond data validation and serialization.
"""Pi-compatible events emitted by Tau's portable agent layer."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from tau_agent.messages import AgentMessage, ToolResultMessage, WireModel
from tau_agent.provider_events import AssistantMessageEvent
from tau_agent.tools import AgentToolResult
from tau_agent.types import JSONValue


class AgentStartEvent(WireModel):
    # [CLASS DOC]
    # Event marking the start of an agent run. No fields besides type.
    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(WireModel):
    # [CLASS DOC]
    # Event marking the end of an agent run; carries the list of messages produced
    # during the run (defaults to empty).
    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage] = Field(default_factory=list)


class TurnStartEvent(WireModel):
    # [CLASS DOC]
    # Event marking the start of one turn (one assistant response plus any tool
    # executions it triggers). No fields besides type.
    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(WireModel):
    # [CLASS DOC]
    # Event marking the end of a turn; carries the turn's assistant message and the
    # tool result messages produced in that turn (default empty).
    type: Literal["turn_end"] = "turn_end"
    message: AgentMessage
    tool_results: list[ToolResultMessage] = Field(default_factory=list)


class MessageStartEvent(WireModel):
    # [CLASS DOC]
    # Event emitted when a message (user, assistant, or tool result) begins; carries
    # the message as known so far.
    type: Literal["message_start"] = "message_start"
    message: AgentMessage


class MessageUpdateEvent(WireModel):
    # [CLASS DOC]
    # Streaming update for an in-progress assistant message: carries the partial
    # message and the underlying provider streaming event, which serializes under
    # the camelCase key "assistantMessageEvent" for Pi compatibility.
    type: Literal["message_update"] = "message_update"
    message: AgentMessage
    assistant_message_event: AssistantMessageEvent = Field(
        serialization_alias="assistantMessageEvent"
    )


class MessageEndEvent(WireModel):
    # [CLASS DOC]
    # Event emitted when a message is complete; carries the final message.
    type: Literal["message_end"] = "message_end"
    message: AgentMessage


class ToolExecutionStartEvent(WireModel):
    # [CLASS DOC]
    # Event emitted before a tool runs: tool-call ID, tool name and the call's
    # argument mapping (default empty).
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: dict[str, JSONValue] = Field(default_factory=dict)


class ToolExecutionUpdateEvent(WireModel):
    # [CLASS DOC]
    # Progress event during a tool run: tool-call ID, tool name, arguments, and a
    # partial tool result produced so far.
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    args: dict[str, JSONValue] = Field(default_factory=dict)
    partial_result: AgentToolResult


class ToolExecutionEndEvent(WireModel):
    # [CLASS DOC]
    # Event emitted after a tool finishes: tool-call ID, tool name, the final tool
    # result, and whether it ended in error.
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: AgentToolResult
    is_error: bool


type AgentEvent = Annotated[
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent,
    Field(discriminator="type"),
]
