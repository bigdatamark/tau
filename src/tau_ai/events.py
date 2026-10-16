# [MODULE DOC]
# HELPER: Thin re-export module exposing the canonical assistant stream event
# types under the tau_ai namespace.
# Behavior: imports the assistant stream event models and the DoneReason /
# ErrorReason literal aliases from tau_agent.provider_events and lists them in
# the public export list. Defines no logic of its own.
# Inputs/outputs: none at runtime; importing it yields the same class objects as
# tau_agent.provider_events (identity is preserved, not copies).
# Side effects: importing tau_agent.provider_events. Raises nothing itself.
"""Public re-exports of the canonical assistant stream event contract."""

from tau_agent.provider_events import (
    AssistantDoneEvent,
    AssistantErrorEvent,
    AssistantMessageEvent,
    AssistantStartEvent,
    DoneReason,
    ErrorReason,
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

__all__ = [
    "AssistantDoneEvent",
    "AssistantErrorEvent",
    "AssistantMessageEvent",
    "AssistantStartEvent",
    "DoneReason",
    "ErrorReason",
    "TextDeltaEvent",
    "TextEndEvent",
    "TextStartEvent",
    "ThinkingDeltaEvent",
    "ThinkingEndEvent",
    "ThinkingStartEvent",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "ToolCallStartEvent",
]
