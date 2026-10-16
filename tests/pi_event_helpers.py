# [MODULE DOC]
# HELPER: test-only constructors for canonical Pi-style provider stream events
# (assistant start, text delta, thinking delta, tool-call end, done, error) so
# tests can script fake providers without repeating boilerplate.
"""Canonical Pi event constructors used by the test suite."""

from tau_agent.messages import AssistantMessage, ThinkingContent, ToolCall
from tau_agent.provider_events import (
    AssistantDoneEvent,
    AssistantErrorEvent,
    AssistantStartEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEndEvent,
)


def assistant_start(model: str = "fake") -> AssistantStartEvent:
    # [FUNCTION DOC]
    # HELPER: build an assistant start event whose partial message carries the given
    # model name (default "fake"). Returns the event; no side effects.
    return AssistantStartEvent(partial=AssistantMessage(model=model))


def text_delta(delta: str) -> TextDeltaEvent:
    # [FUNCTION DOC]
    # HELPER: build a text delta event at content index 0 with the given delta text;
    # the partial message's content is just that delta (not accumulated text).
    return TextDeltaEvent(content_index=0, delta=delta, partial=AssistantMessage(content=delta))


def thinking_delta(delta: str) -> ThinkingDeltaEvent:
    # [FUNCTION DOC]
    # HELPER: build a thinking delta event at content index 0 whose partial message
    # contains a single thinking block holding the delta text.
    return ThinkingDeltaEvent(
        content_index=0,
        delta=delta,
        partial=AssistantMessage(content=[ThinkingContent(thinking=delta)]),
    )


def tool_call_end(tool_call: ToolCall) -> ToolCallEndEvent:
    # [FUNCTION DOC]
    # HELPER: build a tool-call end event at content index 0 for the given tool call,
    # with a partial message containing only that tool call.
    return ToolCallEndEvent(
        content_index=0,
        tool_call=tool_call,
        partial=AssistantMessage(content=[tool_call]),
    )


def assistant_done(
    message: AssistantMessage | dict[str, object],
    finish_reason: str | None = None,
) -> AssistantDoneEvent:
    # [FUNCTION DOC]
    # HELPER: build a terminal done event. Accepts an assistant message or a mapping
    # validated into one. Stop reason is "toolUse" if the message has tool calls or
    # the finish reason is a tool-call alias; "length" for length/max-token aliases;
    # otherwise "stop". Mutates the message's stop_reason (also the caller's object
    # when a message instance is passed) and returns the done event with that reason.
    # Raises a validation error if the mapping is not a valid assistant message.
    final = (
        message
        if isinstance(message, AssistantMessage)
        else AssistantMessage.model_validate(message)
    )
    if final.tool_calls or finish_reason in {"tool_calls", "tool_use", "toolUse"}:
        reason = "toolUse"
    elif finish_reason in {"length", "max_tokens", "MAX_TOKENS", "incomplete"}:
        reason = "length"
    else:
        reason = "stop"
    final.stop_reason = reason
    return AssistantDoneEvent(reason=reason, message=final)


def assistant_error(message: str, data: object = None) -> AssistantErrorEvent:
    # [FUNCTION DOC]
    # HELPER: build an error event whose error message has stop reason "error" and
    # the given error text. The extra data argument is accepted and discarded.
    del data
    error = AssistantMessage(stop_reason="error", error_message=message)
    return AssistantErrorEvent(reason="error", error=error)
