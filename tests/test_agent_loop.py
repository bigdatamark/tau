# [MODULE DOC]
# Deterministic tests for the portable agent loop (tau_agent.loop.run_agent_loop)
# using the scripted FakeProvider and fake tools: canonical event ordering,
# provider wait timing, nested thinking events, tool execution and tool-result
# message lifecycle, tool call id/signal/progress plumbing, unknown tools, provider
# errors, replay hygiene for failed assistants and malformed tool history,
# steering/follow-up injection and the max_turns guard. Pure in-memory; no I/O.
import asyncio
from collections.abc import AsyncIterator, Mapping
from itertools import chain, repeat

import pytest

import tau_agent.loop as loop_module
from pi_event_helpers import (
    assistant_done,
    assistant_error,
    assistant_start,
    text_delta,
    thinking_delta,
    tool_call_end,
)
from tau_agent import (
    AgentEvent,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    MessageEndEvent,
    MessageUpdateEvent,
    SimpleCancellationToken,
    TextContent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionUpdateEvent,
    ToolResultMessage,
    UserMessage,
)
from tau_agent.loop import run_agent_loop
from tau_agent.provider_events import ThinkingDeltaEvent
from tau_agent.types import JSONValue
from tau_ai import CancellationToken, FakeProvider


async def _collect(stream: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    # [FUNCTION DOC]
    # HELPER: Drain an async event stream into a list and return it.
    return [event async for event in stream]


def _tool(
    name: str,
    execute_fn,
) -> AgentTool:  # noqa: ANN001
    # [FUNCTION DOC]
    # HELPER: Build a minimal agent tool with the given name, a title-cased label,
    # a "Run <name>." description, an empty object schema and the given executor.
    return AgentTool(
        name=name,
        label=name.title(),
        description=f"Run {name}.",
        parameters={"type": "object"},
        execute_fn=execute_fn,
    )


@pytest.mark.anyio
async def test_agent_loop_streams_canonical_nested_events() -> None:
    # [FUNCTION DOC]
    # Runs one text-only turn and asserts the exact event sequence (agent_start,
    # turn_start, message_start, two message_update, message_end, turn_end,
    # agent_end), that update events carry the provider text deltas, and that the
    # final assistant message is appended to the caller's message list.
    messages: list[AgentMessage] = [UserMessage(content="Say hello")]
    assistant = AssistantMessage(content="Hello", model="fake")
    provider = FakeProvider(
        [[assistant_start(), text_delta("Hel"), text_delta("lo"), assistant_done(assistant)]]
    )

    events = await _collect(
        run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=messages,
            tools=[],
        )
    )

    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_update",
        "message_update",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    updates = [event for event in events if isinstance(event, MessageUpdateEvent)]
    assert [event.assistant_message_event.delta for event in updates] == ["Hel", "lo"]  # type: ignore[union-attr]
    assert messages == [messages[0], assistant]


@pytest.mark.anyio
async def test_agent_loop_records_only_monotonic_provider_wait_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC]
    # Monkeypatches the loop's monotonic clock with scripted ticks and asserts the
    # assistant timing records time-to-first-output of 1250 ms and a total duration
    # of 5000 ms, i.e. only provider wait windows are measured.
    ticks = chain(
        [
            0,
            100_000_000,
            1_000_000_000,
            2_150_000_000,
            10_000_000_000,
            13_750_000_000,
        ],
        repeat(13_750_000_000),
    )
    monkeypatch.setattr(loop_module, "monotonic_ns", lambda: next(ticks))
    messages: list[AgentMessage] = [UserMessage(content="Say hello")]
    assistant = AssistantMessage(content="Hello", model="fake")
    assistant.usage.output = 100
    provider = FakeProvider([[assistant_start(), text_delta("Hello"), assistant_done(assistant)]])

    await _collect(
        run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=messages,
            tools=[],
        )
    )

    assert assistant.timing is not None
    assert assistant.timing.time_to_first_output_ms == 1250
    assert assistant.timing.total_duration_ms == 5000


@pytest.mark.anyio
async def test_agent_loop_does_not_invent_ttft_without_an_output_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC]
    # With a provider that emits no deltas, asserts time-to-first-output stays None
    # while total duration (1100 ms from scripted ticks) is still recorded.
    ticks = chain(
        [0, 100_000_000, 1_000_000_000, 2_000_000_000],
        repeat(2_000_000_000),
    )
    monkeypatch.setattr(loop_module, "monotonic_ns", lambda: next(ticks))
    assistant = AssistantMessage(content="Non-streamed", model="fake")
    provider = FakeProvider([[assistant_start(), assistant_done(assistant)]])

    await _collect(
        run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=[UserMessage(content="Say hello")],
            tools=[],
        )
    )

    assert assistant.timing is not None
    assert assistant.timing.time_to_first_output_ms is None
    assert assistant.timing.total_duration_ms == 1100


@pytest.mark.anyio
async def test_agent_loop_nests_thinking_events_without_losing_final_message() -> None:
    # [FUNCTION DOC]
    # Asserts thinking deltas are surfaced as nested assistant-message events inside
    # message_update events and the final provider message is what gets appended.
    messages: list[AgentMessage] = [UserMessage(content="Think briefly")]
    assistant = AssistantMessage(content="Done", model="fake")
    provider = FakeProvider(
        [
            [
                assistant_start(),
                thinking_delta("hidden "),
                thinking_delta("reasoning"),
                text_delta("Done"),
                assistant_done(assistant),
            ]
        ]
    )

    events = await _collect(
        run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=messages,
            tools=[],
        )
    )

    nested = [
        event.assistant_message_event
        for event in events
        if isinstance(event, MessageUpdateEvent)
        and isinstance(event.assistant_message_event, ThinkingDeltaEvent)
    ]
    assert [event.delta for event in nested] == ["hidden ", "reasoning"]
    assert messages[-1] == assistant
    # The final provider message is the canonical persistence boundary.
    assert isinstance(messages[-1], AssistantMessage)


@pytest.mark.anyio
async def test_agent_loop_executes_tool_and_emits_tool_result_message_lifecycle() -> None:
    # [FUNCTION DOC]
    # Scripts a tool-use turn followed by a final turn and asserts: a tool result
    # message is appended with role, tool name, text and details from the tool; the
    # result message emits only message_end (no start/update of its own in the
    # filtered check); three message_start events overall; and the second provider
    # call received user, assistant and tool result messages.
    async def execute(
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: CancellationToken | None = None,
        on_update=None,  # noqa: ANN001
    ) -> AgentToolResult:
        # [FUNCTION DOC]
        # HELPER: Fake read tool returning text "contents of <path>" and details with
        # the path.
        del tool_call_id, signal, on_update
        return AgentToolResult(
            content=[TextContent(text=f"contents of {arguments['path']}")],
            details={"path": arguments["path"]},
        )

    tool_call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
    first = AssistantMessage(content=[TextContent(text="Reading."), tool_call], model="fake")
    final = AssistantMessage(content="Done.", model="fake")
    provider = FakeProvider(
        [
            [assistant_start(), tool_call_end(tool_call), assistant_done(first, "toolUse")],
            [assistant_start(), text_delta("Done."), assistant_done(final)],
        ]
    )
    messages: list[AgentMessage] = [UserMessage(content="Read README.md")]

    events = await _collect(
        run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=messages,
            tools=[_tool("read", execute)],
        )
    )

    result = next(message for message in messages if isinstance(message, ToolResultMessage))
    assert result.role == "toolResult"
    assert result.tool_name == "read"
    assert result.text == "contents of README.md"
    assert result.details == {"path": "README.md"}
    result_lifecycle = [
        event.type
        for event in events
        if isinstance(event, (MessageEndEvent,)) and event.message is result
    ]
    assert result_lifecycle == ["message_end"]
    assert [event.type for event in events].count("message_start") == 3
    assert provider.calls[1][2] == messages[:3]


@pytest.mark.anyio
async def test_agent_loop_passes_call_id_signal_and_progress_to_tool() -> None:
    # [FUNCTION DOC]
    # Asserts the tool executor receives the call id and the exact cancellation
    # token, and that a progress update it reports surfaces as a tool execution
    # update event with the partial text.
    observed: list[tuple[str, CancellationToken | None]] = []

    async def execute(
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: CancellationToken | None = None,
        on_update=None,  # noqa: ANN001
    ) -> AgentToolResult:
        # [FUNCTION DOC]
        # HELPER: Fake tool recording (call id, signal), asserting a progress callback
        # exists, sending a "working" partial result, yielding once and returning "done".
        del arguments
        observed.append((tool_call_id, signal))
        assert on_update is not None
        on_update(AgentToolResult(content="working"))
        await asyncio.sleep(0)
        return AgentToolResult(content="done")

    call = ToolCall(id="call-1", name="work", arguments={})
    first = AssistantMessage(content=[call], model="fake")
    final = AssistantMessage(content="finished", model="fake")
    provider = FakeProvider(
        [
            [assistant_start(), tool_call_end(call), assistant_done(first, "toolUse")],
            [assistant_start(), assistant_done(final)],
        ]
    )
    signal = SimpleCancellationToken()

    events = await _collect(
        run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=[UserMessage(content="work")],
            tools=[_tool("work", execute)],
            signal=signal,
        )
    )

    assert observed == [("call-1", signal)]
    updates = [event for event in events if isinstance(event, ToolExecutionUpdateEvent)]
    assert [event.partial_result.text for event in updates] == ["working"]


@pytest.mark.anyio
async def test_agent_loop_records_unknown_tool_as_canonical_error_result() -> None:
    # [FUNCTION DOC]
    # Asserts calling an unregistered tool produces an error tool execution end
    # event and an error tool result message with text "Tool missing not found".
    call = ToolCall(id="call-1", name="missing", arguments={})
    assistant = AssistantMessage(content=[call], model="fake")
    messages: list[AgentMessage] = [UserMessage(content="Use it")]
    provider = FakeProvider(
        [[assistant_start(), tool_call_end(call), assistant_done(assistant, "toolUse")]]
    )

    events = await _collect(
        run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=messages,
            tools=[],
            max_turns=1,
        )
    )

    end = next(event for event in events if isinstance(event, ToolExecutionEndEvent))
    assert end.is_error is True
    assert end.result.text == "Tool missing not found"
    result = next(message for message in messages if isinstance(message, ToolResultMessage))
    assert result.is_error is True
    assert result.text == "Tool missing not found"


@pytest.mark.anyio
async def test_agent_loop_converts_provider_error_to_assistant_error_message() -> None:
    # [FUNCTION DOC]
    # Asserts a provider error event yields the short event sequence without updates
    # and appends an assistant message with stop reason "error" and the error text.
    messages: list[AgentMessage] = [UserMessage(content="hello")]
    provider = FakeProvider([[assistant_error("provider failed")]])

    events = await _collect(
        run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=messages,
            tools=[],
        )
    )

    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    error = messages[-1]
    assert isinstance(error, AssistantMessage)
    assert error.stop_reason == "error"
    assert error.error_message == "provider failed"


@pytest.mark.anyio
async def test_agent_loop_excludes_empty_failed_assistant_from_next_provider_call() -> None:
    # [FUNCTION DOC]
    # Runs a failing call then a second run; asserts the failed assistant stays in
    # the stored history but is excluded from the messages replayed to the provider,
    # which sees only the two user prompts, and that the recovered message ends the
    # history.
    messages: list[AgentMessage] = []
    recovered = AssistantMessage(content="recovered", model="fake")
    provider = FakeProvider(
        [
            [assistant_error("provider failed")],
            [assistant_start(), assistant_done(recovered)],
        ]
    )

    await _collect(
        run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=messages,
            tools=[],
            prompts=[UserMessage(content="hello")],
        )
    )
    failed = messages[-1]
    assert isinstance(failed, AssistantMessage)
    assert failed.stop_reason == "error"

    await _collect(
        run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=messages,
            tools=[],
            prompts=[UserMessage(content="continue")],
        )
    )

    assert failed in messages
    replayed = provider.calls[1][2]
    assert [message.text for message in replayed] == ["hello", "continue"]
    assert failed not in replayed
    assert messages[-1] is recovered


@pytest.mark.anyio
async def test_agent_loop_repairs_malformed_tool_history_before_provider_call() -> None:
    # [FUNCTION DOC]
    # Given history with an unanswered tool call, a later user message and an
    # orphan tool result, asserts the provider receives the assistant, a synthetic
    # error tool result for the unanswered call, then the user message, with the
    # orphan dropped from the replay but kept in stored history.
    call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
    assistant = AssistantMessage(content=[call])
    late_user = UserMessage(content="continue")
    orphan = ToolResultMessage(tool_call_id="call-missing", tool_name="bash", content="orphan")
    recovered = AssistantMessage(content="recovered", model="fake")
    provider = FakeProvider([[assistant_start(), assistant_done(recovered)]])
    messages: list[AgentMessage] = [assistant, late_user, orphan]

    await _collect(
        run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=messages,
            tools=[],
        )
    )

    replayed = provider.calls[0][2]
    assert replayed[0] is assistant
    repair = replayed[1]
    assert isinstance(repair, ToolResultMessage)
    assert repair.tool_call_id == "call-1"
    assert repair.is_error is True
    assert replayed[2] is late_user
    assert orphan not in replayed
    assert orphan in messages


@pytest.mark.anyio
async def test_agent_loop_injects_steering_and_follow_up_messages() -> None:
    # [FUNCTION DOC]
    # Asserts steering messages are injected after tool execution and follow-up
    # messages after the agent would stop, producing user messages start/steer/
    # follow up and three provider calls.
    call = ToolCall(id="call-1", name="work", arguments={})

    async def execute(
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal=None,  # noqa: ANN001
        on_update=None,  # noqa: ANN001
    ) -> AgentToolResult:
        # [FUNCTION DOC]
        # HELPER: Fake tool that ignores its inputs and returns "ok".
        del tool_call_id, arguments, signal, on_update
        return AgentToolResult(content="ok")

    first = AssistantMessage(content=[call], model="fake")
    second = AssistantMessage(content="second", model="fake")
    third = AssistantMessage(content="third", model="fake")
    provider = FakeProvider(
        [
            [assistant_start(), tool_call_end(call), assistant_done(first, "toolUse")],
            [assistant_start(), assistant_done(second)],
            [assistant_start(), assistant_done(third)],
        ]
    )
    steering = [UserMessage(content="steer")]
    follow_up = [UserMessage(content="follow up")]

    def pop(queue: list[UserMessage]) -> tuple[UserMessage, ...]:
        # [FUNCTION DOC]
        # HELPER: Pop and return the first queued message as a one-element tuple, or an
        # empty tuple when the queue is empty (mutates the queue).
        return (queue.pop(0),) if queue else ()

    messages: list[AgentMessage] = [UserMessage(content="start")]
    await _collect(
        run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=messages,
            tools=[_tool("work", execute)],
            get_steering_messages=lambda: pop(steering),
            get_follow_up_messages=lambda: pop(follow_up),
        )
    )

    assert [message.text for message in messages if isinstance(message, UserMessage)] == [
        "start",
        "steer",
        "follow up",
    ]
    assert len(provider.calls) == 3


@pytest.mark.anyio
async def test_agent_loop_stops_with_assistant_error_after_max_turns() -> None:
    # [FUNCTION DOC]
    # Asserts that with max_turns=1 and a tool-use response, the loop stops after one
    # provider call and appends an assistant error "Agent stopped after max_turns=1".
    call = ToolCall(id="call-1", name="missing", arguments={})
    assistant = AssistantMessage(content=[call], model="fake")
    provider = FakeProvider(
        [[assistant_start(), tool_call_end(call), assistant_done(assistant, "toolUse")]]
    )
    messages: list[AgentMessage] = [UserMessage(content="loop")]

    await _collect(
        run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=messages,
            tools=[],
            max_turns=1,
        )
    )

    error = messages[-1]
    assert isinstance(error, AssistantMessage)
    assert error.stop_reason == "error"
    assert error.error_message == "Agent stopped after max_turns=1"
    assert len(provider.calls) == 1
