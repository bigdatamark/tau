# [MODULE DOC]
# Tests for AgentHarness, the reusable agent brain: Pi-style event lifecycle for
# prompt, subscriber push events and unsubscribe, rejecting overlapping runs,
# follow-up queue draining (one-at-a-time and "all" modes), passing tools to the
# loop, queue mutator snapshots, and repair of interrupted tool calls on
# cancellation and at run entry, including listener notification. Uses the fake
# provider and fake tools for determinism.
import asyncio
from collections.abc import Mapping

import pytest

from pi_event_helpers import assistant_done, assistant_start, text_delta, tool_call_end
from tau_agent import (
    AgentHarness,
    AgentHarnessConfig,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    MessageEndEvent,
    MessageStartEvent,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from tau_agent.types import JSONValue
from tau_ai import FakeProvider


def _texts(harness: AgentHarness) -> list[tuple[str, str]]:
    # [FUNCTION DOC]
    # HELPER: return (role, text) pairs for every message in the harness transcript,
    # using an empty string when a message has no text attribute.
    return [(message.role, getattr(message, "text", "")) for message in harness.messages]


@pytest.mark.anyio
async def test_prompt_appends_user_and_assistant_with_pi_lifecycle() -> None:
    # [FUNCTION DOC]
    # Asserts a single prompt yields agent_start, turn_start, user message start/end,
    # assistant message start/end, turn_end, agent_end, and the transcript holds the
    # user and assistant messages.
    assistant = AssistantMessage(content="Hello")
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=FakeProvider([[assistant_start(), assistant_done(assistant)]]),
            model="fake",
            system="You are Tau.",
        )
    )

    events = [event async for event in harness.prompt("Hi")]

    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    starts = [event for event in events if isinstance(event, MessageStartEvent)]
    assert [event.message.role for event in starts] == ["user", "assistant"]
    assert _texts(harness) == [("user", "Hi"), ("assistant", "Hello")]


@pytest.mark.anyio
async def test_subscribers_receive_nested_message_updates_and_unsubscribe() -> None:
    # [FUNCTION DOC]
    # Asserts subscribers see message_update events and end with agent_end, that an
    # unsubscribed listener sees nothing from a later continue run, and that the
    # session id is forwarded to every provider call.
    assistant = AssistantMessage(content="Hello")
    provider = FakeProvider(
        [
            [assistant_start(), text_delta("Hello"), assistant_done(assistant)],
            [assistant_start(), assistant_done(assistant)],
        ]
    )
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=provider,
            model="fake",
            system="You are Tau.",
            session_id="session-123",
        )
    )
    seen: list[str] = []
    unsubscribe = harness.subscribe(lambda event: seen.append(event.type))

    _ = [event async for event in harness.prompt("Hi")]
    unsubscribe()
    _ = [event async for event in harness.continue_()]

    assert "message_update" in seen
    assert seen[-1] == "agent_end"
    assert len(provider.calls) == 2
    assert provider.session_ids == ["session-123", "session-123"]


@pytest.mark.anyio
async def test_harness_rejects_overlap_and_drains_followups() -> None:
    # [FUNCTION DOC]
    # Asserts starting a second prompt while running raises RuntimeError
    # ("already running"), and a queued follow-up is processed as a new turn after
    # the first answer.
    first = AssistantMessage(content="First")
    second = AssistantMessage(content="Second")
    provider = FakeProvider(
        [
            [assistant_start(), assistant_done(first)],
            [assistant_start(), assistant_done(second)],
        ]
    )
    harness = AgentHarness(
        AgentHarnessConfig(provider=provider, model="fake", system="You are Tau.")
    )

    queued = False
    async for event in harness.prompt("Hi"):
        if (
            isinstance(event, MessageStartEvent)
            and event.message.role == "assistant"
            and not queued
        ):
            with pytest.raises(RuntimeError, match="already running"):
                harness.prompt("overlap")
            harness.follow_up("Later")
            queued = True

    assert _texts(harness) == [
        ("user", "Hi"),
        ("assistant", "First"),
        ("user", "Later"),
        ("assistant", "Second"),
    ]


@pytest.mark.anyio
async def test_harness_queue_mode_all_drains_messages_together() -> None:
    # [FUNCTION DOC]
    # Asserts in queue mode "all" two follow-ups queued mid-run are both delivered
    # as user messages in order in the next turn.
    first = AssistantMessage(content="First")
    second = AssistantMessage(content="Second")
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=FakeProvider(
                [
                    [assistant_start(), assistant_done(first)],
                    [assistant_start(), assistant_done(second)],
                ]
            ),
            model="fake",
            system="You are Tau.",
            queue_mode="all",
        )
    )

    async for event in harness.prompt("Hi"):
        if (
            isinstance(event, MessageEndEvent)
            and isinstance(event.message, AssistantMessage)
            and event.message.text == "First"
        ):
            harness.follow_up("Second prompt")
            harness.follow_up("Third prompt")

    assert [text for role, text in _texts(harness) if role == "user"] == [
        "Hi",
        "Second prompt",
        "Third prompt",
    ]


@pytest.mark.anyio
async def test_harness_passes_canonical_tools_to_loop() -> None:
    # [FUNCTION DOC]
    # Asserts a tool call from the provider executes the registered echo tool,
    # producing a tool result message with the tool name and output, and that the
    # provider receives the exact tool list.
    async def execute(
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal=None,  # noqa: ANN001
        on_update=None,  # noqa: ANN001
    ) -> AgentToolResult:
        # [FUNCTION DOC]
        # HELPER: fake echo tool executor returning the "text" argument as content.
        del tool_call_id, signal, on_update
        return AgentToolResult(content=str(arguments["text"]))

    tool = AgentTool(
        name="echo",
        label="Echo",
        description="Echo text.",
        parameters={"type": "object"},
        execute_fn=execute,
    )
    call = ToolCall(id="call-1", name="echo", arguments={"text": "hi"})
    first = AssistantMessage(content=[call])
    final = AssistantMessage(content="Done")
    provider = FakeProvider(
        [
            [assistant_start(), tool_call_end(call), assistant_done(first, "toolUse")],
            [assistant_start(), assistant_done(final)],
        ]
    )
    harness = AgentHarness(
        AgentHarnessConfig(provider=provider, model="fake", system="You are Tau.", tools=[tool])
    )

    _ = [event async for event in harness.prompt("echo")]

    result = next(message for message in harness.messages if isinstance(message, ToolResultMessage))
    assert result.tool_name == "echo"
    assert result.text == "hi"
    assert provider.calls[0][3] == [tool]


def test_queue_mutators_return_canonical_snapshots() -> None:
    # [FUNCTION DOC]
    # Asserts steer and follow_up return queue snapshots, pop_latest_* remove and
    # return the newest queued message, clear_queues returns what was cleared, and
    # the pending count becomes zero.
    harness = AgentHarness(
        AgentHarnessConfig(provider=FakeProvider([]), model="fake", system="You are Tau.")
    )

    harness.steer("First")
    second = harness.steer("Second").steering[-1]
    later = harness.follow_up("Later").follow_up[-1]
    assert harness.pop_latest_steering() == second
    assert harness.pop_latest_follow_up() == later
    assert [message.text for message in harness.queued_messages.steering] == ["First"]

    cleared = harness.clear_queues()
    assert [message.text for message in cleared.steering] == ["First"]
    assert harness.pending_message_count == 0


def _blocking_tool(tool_started: "asyncio.Event", release: "asyncio.Event") -> AgentTool:
    # [FUNCTION DOC]
    # HELPER: build a "hang" tool whose executor sets the started event and then
    # waits on the release event before returning "done".
    async def hang(
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal=None,  # noqa: ANN001
        on_update=None,  # noqa: ANN001
    ) -> AgentToolResult:
        # [FUNCTION DOC]
        # HELPER: fake tool executor that signals start and blocks until released.
        del tool_call_id, arguments, signal, on_update
        tool_started.set()
        await release.wait()
        return AgentToolResult(content="done")

    return AgentTool(
        name="hang",
        label="Hang",
        description="Block until released.",
        parameters={"type": "object"},
        execute_fn=hang,
    )


def _blocking_run_harness(tool_started: "asyncio.Event", release: "asyncio.Event") -> AgentHarness:
    # [FUNCTION DOC]
    # HELPER: build a harness whose fake provider requests one "hang" tool call, so
    # a run blocks inside tool execution.
    call = ToolCall(id="call-1", name="hang", arguments={})
    assistant = AssistantMessage(content=[call])
    provider = FakeProvider(
        [[assistant_start(), tool_call_end(call), assistant_done(assistant, "toolUse")]]
    )
    return AgentHarness(
        AgentHarnessConfig(
            provider=provider,
            model="fake",
            system="You are Tau.",
            tools=[_blocking_tool(tool_started, release)],
        )
    )


@pytest.mark.anyio
async def test_cancelled_run_notifies_listeners_of_interrupted_tool_repair() -> None:
    # [FUNCTION DOC]
    # Regression: cancels a run blocked in a tool and asserts the transcript ends
    # with an "interrupted by user" tool result and subscribers received exactly one
    # tool-result message_end for the call.
    # Regression: the cancelled-cleanup repair emitted no events, so
    # push-based subscribers never saw it.
    tool_started = asyncio.Event()
    release = asyncio.Event()
    harness = _blocking_run_harness(tool_started, release)
    seen: list[object] = []
    harness.subscribe(seen.append)

    async def consume() -> None:
        # [FUNCTION DOC]
        # HELPER: drain the prompt event stream so it can run as a task.
        async for _event in harness.prompt("go"):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(tool_started.wait(), timeout=5)
    harness.cancel()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    repair = harness.messages[-1]
    assert isinstance(repair, ToolResultMessage)
    assert repair.text == "Tool call interrupted by user"
    repair_ends = [
        event
        for event in seen
        if isinstance(event, MessageEndEvent) and isinstance(event.message, ToolResultMessage)
    ]
    assert [event.message.tool_call_id for event in repair_ends] == ["call-1"]


@pytest.mark.anyio
async def test_listener_error_during_teardown_does_not_mask_cancellation() -> None:
    # [FUNCTION DOC]
    # Asserts a listener raising during the cancellation repair does not replace the
    # CancelledError, and the repair message is still appended.
    tool_started = asyncio.Event()
    release = asyncio.Event()
    harness = _blocking_run_harness(tool_started, release)

    def explode(event: object) -> None:
        # [FUNCTION DOC]
        # HELPER: listener that raises RuntimeError on tool-result message_end events.
        if isinstance(event, MessageEndEvent) and isinstance(event.message, ToolResultMessage):
            raise RuntimeError("listener exploded")

    harness.subscribe(explode)

    async def consume() -> None:
        # [FUNCTION DOC]
        # HELPER: drain the prompt event stream so it can run as a task.
        async for _event in harness.prompt("go"):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(tool_started.wait(), timeout=5)
    harness.cancel()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert isinstance(harness.messages[-1], ToolResultMessage)


def test_harness_repairs_interrupted_tool_calls() -> None:
    # [FUNCTION DOC]
    # Asserts append_interrupted_tool_results adds one error tool result with the
    # "interrupted by user" text for a dangling tool call and returns the count 1.
    call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
    harness = AgentHarness(
        AgentHarnessConfig(provider=FakeProvider([]), model="fake", system="You are Tau."),
        messages=[AssistantMessage(content=[TextContent(text="Reading"), call])],
    )

    assert harness.append_interrupted_tool_results() == 1
    repair = harness.messages[-1]
    assert isinstance(repair, ToolResultMessage)
    assert repair.is_error is True
    assert repair.text == "Tool call interrupted by user"


@pytest.mark.anyio
async def test_entry_path_repair_is_pushed_to_listeners() -> None:
    # [FUNCTION DOC]
    # Asserts a transcript with a dangling tool call is repaired at prompt entry,
    # emitting message start/end for the repair right after agent/turn start, and
    # both the subscriber and the event consumer see its message_end.
    # A transcript can reach prompt()/continue_() with a dangling tool call
    # (for example after a persist failure killed the previous run). The
    # synthetic repair must flow through events so push persistence sees it.
    call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
    harness = AgentHarness(
        AgentHarnessConfig(
            provider=FakeProvider(
                [[assistant_start(), assistant_done(AssistantMessage(content="Recovered."))]]
            ),
            model="fake",
            system="You are Tau.",
        ),
        messages=[AssistantMessage(content=[call])],
    )
    seen: list[object] = []
    harness.subscribe(seen.append)

    events = [event async for event in harness.prompt("continue")]

    assert [event.type for event in events[:4]] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
    ]
    repair = next(message for message in harness.messages if isinstance(message, ToolResultMessage))
    assert repair.tool_call_id == "call-1"
    listener_ends = [
        event.message.tool_call_id
        for event in seen
        if isinstance(event, MessageEndEvent) and isinstance(event.message, ToolResultMessage)
    ]
    assert listener_ends == ["call-1"]
    consumer_ends = [
        event.message.tool_call_id
        for event in events
        if isinstance(event, MessageEndEvent) and isinstance(event.message, ToolResultMessage)
    ]
    assert consumer_ends == ["call-1"]
