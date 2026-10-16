# [MODULE DOC]
# Contract tests for the canonical Pi-shaped event stream: the agent harness wraps
# provider assistant-message events in message_update events, emits message and
# tool-execution lifecycle events, and tau_agent stays independent of tau_ai.
"""Focused contract tests for the canonical Pi-shaped event stream."""

from pathlib import Path

import pytest

from tau_agent import (
    AgentEndEvent,
    AgentHarness,
    AgentHarnessConfig,
    AssistantMessage,
    MessageEndEvent,
    MessageUpdateEvent,
    TextContent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResultMessage,
)
from tau_agent.provider_events import (
    AssistantDoneEvent,
    AssistantStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from tau_ai import FakeProvider


def test_tau_agent_does_not_import_tau_ai() -> None:
    # [FUNCTION DOC]
    # Scans every Python file under src/tau_agent (relative to the current working
    # directory, so must run from repo root) and asserts none imports tau_ai.
    # Side effects: reads source files.
    for path in Path("src/tau_agent").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "from tau_ai" not in source, path
        assert "import tau_ai" not in source, path


def test_tau_ai_reexports_canonical_event_classes() -> None:
    # [FUNCTION DOC]
    # Asserts tau_ai.events re-exports the same text-delta event class object defined in
    # tau_agent.provider_events (identity, not a copy).
    from tau_ai.events import TextDeltaEvent as public

    assert public is TextDeltaEvent


@pytest.mark.anyio
async def test_text_stream_has_nested_updates_and_terminal_messages() -> None:
    # [FUNCTION DOC]
    # Drives the harness with a fake provider streaming a single "hello" text block and
    # asserts the exact outer event type sequence (agent/turn/message start and end, with
    # three message_update events), that the updates wrap text_start/text_delta/text_end
    # in order, and that the agent-end event's final message equals the hello message.
    empty = AssistantMessage(model="fake")
    hello = AssistantMessage(model="fake", content=[TextContent(text="hello")])
    provider = FakeProvider(
        [
            [
                AssistantStartEvent(partial=empty),
                TextStartEvent(content_index=0, partial=empty),
                TextDeltaEvent(content_index=0, delta="hello", partial=hello),
                TextEndEvent(content_index=0, content="hello", partial=hello),
                AssistantDoneEvent(reason="stop", message=hello),
            ]
        ]
    )
    harness = AgentHarness(AgentHarnessConfig(provider=provider, model="fake", system="test"))

    events = [event async for event in harness.prompt("hi")]

    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_update",
        "message_update",
        "message_update",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    updates = [event for event in events if isinstance(event, MessageUpdateEvent)]
    assert [event.assistant_message_event.type for event in updates] == [
        "text_start",
        "text_delta",
        "text_end",
    ]
    assert isinstance(events[-1], AgentEndEvent)
    assert events[-1].messages[-1] == hello


@pytest.mark.anyio
async def test_tool_result_gets_execution_and_message_lifecycle_events() -> None:
    # [FUNCTION DOC]
    # Fake provider first requests a call to an unregistered tool, then finishes with
    # text. Asserts tool-execution start/end and the tool-result message-end share the
    # same tool call id and that the execution ended as an error (unknown tool).
    call = ToolCall(id="call-1", name="missing", arguments={})
    partial = AssistantMessage(model="fake", content=[call])
    provider = FakeProvider(
        [
            [
                AssistantStartEvent(partial=AssistantMessage(model="fake")),
                ToolCallStartEvent(content_index=0, partial=partial),
                ToolCallEndEvent(content_index=0, tool_call=call, partial=partial),
                AssistantDoneEvent(reason="toolUse", message=partial),
            ],
            [
                AssistantStartEvent(partial=AssistantMessage(model="fake")),
                AssistantDoneEvent(
                    reason="stop",
                    message=AssistantMessage(model="fake", content=[TextContent(text="done")]),
                ),
            ],
        ]
    )
    harness = AgentHarness(AgentHarnessConfig(provider=provider, model="fake", system="test"))

    events = [event async for event in harness.prompt("use it")]

    start = next(event for event in events if isinstance(event, ToolExecutionStartEvent))
    end = next(event for event in events if isinstance(event, ToolExecutionEndEvent))
    result_end = next(
        event
        for event in events
        if isinstance(event, MessageEndEvent) and isinstance(event.message, ToolResultMessage)
    )
    assert start.tool_call_id == end.tool_call_id == result_end.message.tool_call_id
    assert end.is_error is True
