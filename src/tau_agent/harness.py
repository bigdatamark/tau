# [MODULE DOC]
# AgentHarness: the stateful, reusable "agent brain" of Tau (Pi's Agent class
# equivalent). Wraps the stateless run_agent_loop with a persistent message
# transcript, event listener fan-out, steering/follow-up queues, cancellation
# and interrupted-tool-call repair. Independent of coding policy, CLI and UI.
# Also defines EventListener (sync or async callable taking an AgentEvent),
# QueueMode ("one_at_a_time" or "all"), QueuedMessages, AgentHarnessConfig and
# SimpleCancellationToken.
# Side effects: none at import.
"""Stateful reusable agent harness built on the Pi-compatible loop."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Literal

from tau_agent.events import AgentEvent, MessageEndEvent, MessageStartEvent
from tau_agent.loop import AfterToolCall, BeforeToolCall, run_agent_loop
from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolResultMessage,
    UserMessage,
)
from tau_agent.provider import ModelProvider
from tau_agent.tools import AgentTool

EventListener = Callable[[AgentEvent], Awaitable[None] | None]
QueueMode = Literal["one_at_a_time", "all"]


@dataclass(frozen=True, slots=True)
class QueuedMessages:
    # [CLASS DOC]
    # Frozen, slotted dataclass snapshot of the pending steering and follow-up
    # message tuples (both default empty).
    steering: tuple[AgentMessage, ...] = ()
    follow_up: tuple[AgentMessage, ...] = ()

    @property
    def count(self) -> int:
        # [FUNCTION DOC]
        # Property: total number of queued messages (steering plus follow-up). Pure.
        return len(self.steering) + len(self.follow_up)


@dataclass(slots=True)
class AgentHarnessConfig:
    # [CLASS DOC]
    # Mutable, slotted dataclass of harness configuration: model provider, model
    # id, system prompt, tool list (default empty), optional max turns, queue mode
    # (default one message at a time), optional session id passed to the provider,
    # and optional before/after tool call hooks forwarded to the loop.
    provider: ModelProvider
    model: str
    system: str
    tools: list[AgentTool] = field(default_factory=list)
    max_turns: int | None = None
    queue_mode: QueueMode = "one_at_a_time"
    session_id: str | None = None
    before_tool_call: BeforeToolCall | None = None
    after_tool_call: AfterToolCall | None = None


class SimpleCancellationToken:
    # [CLASS DOC]
    # Minimal cancellation flag object satisfying the loop/tool cancellation token
    # protocol: starts uncancelled, can be cancelled once (idempotent), and reports
    # its state.
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Initialises the cancelled flag to false.
        self._cancelled = False

    def cancel(self) -> None:
        # [FUNCTION DOC]
        # Sets the cancelled flag to true. Idempotent; never raises.
        self._cancelled = True

    def is_cancelled(self) -> bool:
        # [FUNCTION DOC]
        # Returns the current cancelled flag. Pure.
        return self._cancelled


class AgentHarness:
    # [CLASS DOC]
    # Stateful agent harness. Owns the config, a mutable message list (the
    # transcript passed to and extended by the loop), registered listeners, the
    # current run's cancellation token, a running flag and two FIFO queues
    # (steering: injected mid-run between turns; follow-up: processed after the
    # agent would otherwise stop). Only one run may be active at a time.
    """Reusable stateful agent brain independent of coding/UI policy."""

    def __init__(
        self,
        config: AgentHarnessConfig,
        *,
        messages: Sequence[AgentMessage] = (),
    ) -> None:
        # [FUNCTION DOC]
        # Stores the config, copies the optional initial messages into a new list and
        # initialises empty listeners, no current signal, not running, empty queues.
        # Inputs: config; keyword-only initial message sequence. No side effects.
        self._config = config
        self._messages = list(messages)
        self._listeners: list[EventListener] = []
        self._current_signal: SimpleCancellationToken | None = None
        self._running = False
        self._steering_queue: deque[AgentMessage] = deque()
        self._follow_up_queue: deque[AgentMessage] = deque()

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        # [FUNCTION DOC]
        # Property returning an immutable tuple snapshot of the transcript.
        return tuple(self._messages)

    @property
    def config(self) -> AgentHarnessConfig:
        # [FUNCTION DOC]
        # Property returning the (mutable) config object itself.
        return self._config

    @property
    def is_running(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether a prompt/continue run is currently active (set when the
        # run is requested, cleared when the run generator finishes).
        return self._running

    @property
    def queued_messages(self) -> QueuedMessages:
        # [FUNCTION DOC]
        # Property: a new QueuedMessages snapshot of both queues.
        return QueuedMessages(tuple(self._steering_queue), tuple(self._follow_up_queue))

    @property
    def pending_message_count(self) -> int:
        # [FUNCTION DOC]
        # Property: total number of queued steering and follow-up messages.
        return self.queued_messages.count

    def has_queued_messages(self) -> bool:
        # [FUNCTION DOC]
        # Returns true when either queue is non-empty. Pure.
        return bool(self._steering_queue or self._follow_up_queue)

    def append_message(self, message: AgentMessage) -> None:
        # [FUNCTION DOC]
        # Appends one message to the transcript (no events emitted).
        self._messages.append(message)

    def replace_messages(self, messages: Sequence[AgentMessage]) -> None:
        # [FUNCTION DOC]
        # Replaces the transcript with a fresh list copy of the given messages (no
        # events emitted). Used e.g. after compaction or branch navigation.
        self._messages = list(messages)

    def subscribe(self, listener: EventListener) -> Callable[[], None]:
        # [FUNCTION DOC]
        # Registers an event listener called for every event of every run (awaited if
        # it returns an awaitable).
        # Inputs: listener. Outputs: an unsubscribe function that removes the listener
        # and silently ignores it already being removed.
        # Side effects: mutates the listener list.
        self._listeners.append(listener)

        def unsubscribe() -> None:
            # [FUNCTION DOC]
            # HELPER: Removes the captured listener from the harness listener list,
            # suppressing the error raised when it is no longer present.
            with suppress(ValueError):
                self._listeners.remove(listener)

        return unsubscribe

    def cancel(self) -> None:
        # [FUNCTION DOC]
        # Requests cancellation of the active run by cancelling its token; no-op when
        # no run is active. The loop observes the token cooperatively.
        if self._current_signal is not None:
            self._current_signal.cancel()

    def steer(self, content: str) -> QueuedMessages:
        # [FUNCTION DOC]
        # Queues a plain-text steering message (wrapped as a UserMessage).
        # Outputs: the updated QueuedMessages snapshot. Mutates the steering queue.
        return self.steer_message(UserMessage(content=content))

    def steer_message(self, message: AgentMessage) -> QueuedMessages:
        # [FUNCTION DOC]
        # Appends an arbitrary agent message to the steering queue; returns the
        # updated snapshot.
        self._steering_queue.append(message)
        return self.queued_messages

    def follow_up(self, content: str) -> QueuedMessages:
        # [FUNCTION DOC]
        # Queues a plain-text follow-up message (wrapped as a UserMessage); returns
        # the updated snapshot.
        return self.follow_up_message(UserMessage(content=content))

    def follow_up_message(self, message: AgentMessage) -> QueuedMessages:
        # [FUNCTION DOC]
        # Appends an arbitrary agent message to the follow-up queue; returns the
        # updated snapshot.
        self._follow_up_queue.append(message)
        return self.queued_messages

    def clear_queues(self) -> QueuedMessages:
        # [FUNCTION DOC]
        # Empties both queues and returns a snapshot of what they held before.
        snapshot = self.queued_messages
        self._steering_queue.clear()
        self._follow_up_queue.clear()
        return snapshot

    def pop_latest_follow_up(self) -> AgentMessage | None:
        # [FUNCTION DOC]
        # Removes and returns the most recently queued follow-up message (LIFO end),
        # or none when the queue is empty.
        return self._follow_up_queue.pop() if self._follow_up_queue else None

    def pop_latest_steering(self) -> AgentMessage | None:
        # [FUNCTION DOC]
        # Removes and returns the most recently queued steering message, or none.
        return self._steering_queue.pop() if self._steering_queue else None

    def prompt_message(self, message: AgentMessage) -> AsyncIterator[AgentEvent]:
        # [FUNCTION DOC]
        # Starts a run with the given message as the new prompt.
        # Inputs: an agent message. Outputs: an async iterator of AgentEvents (the run
        # executes only as it is iterated).
        # Raises RuntimeError immediately if a run is already active.
        # Side effects: sets the running flag eagerly.
        # KNOWN ISSUE: the running flag is set before iteration starts and only
        # cleared in the generator's finally block, so if the returned iterator is
        # never iterated (or never closed) the harness stays marked as running and
        # later prompts raise.
        self._ensure_not_running()
        self._running = True
        return self._run(prompts=(message,))

    def prompt(self, content: str) -> AsyncIterator[AgentEvent]:
        # [FUNCTION DOC]
        # Convenience wrapper: wraps text in a UserMessage and delegates to
        # prompt_message. Same outputs/exceptions/side effects.
        return self.prompt_message(UserMessage(content=content))

    def continue_(self) -> AsyncIterator[AgentEvent]:
        # [FUNCTION DOC]
        # Starts a run with no new prompt, continuing from the current transcript (e.g.
        # after an error or to process queued messages). Raises RuntimeError if already
        # running; sets the running flag eagerly (same caveat as prompt_message).
        # Outputs: async iterator of AgentEvents.
        self._ensure_not_running()
        self._running = True
        return self._run()

    async def _run(
        self,
        *,
        prompts: Sequence[AgentMessage] = (),
    ) -> AsyncIterator[AgentEvent]:
        # [FUNCTION DOC]
        # Async generator that performs one run.
        # Behavior: creates a fresh cancellation token as the current signal; first
        # appends synthetic interrupted tool results for any dangling tool calls and
        # passes them to the loop as prelude messages (so they are emitted as events
        # and persisted); then iterates run_agent_loop with the config, the live
        # transcript list (which the loop extends), prompts, tools, max turns, signal,
        # session id, queue drain callbacks and tool hooks. Each event is first sent to
        # all listeners, then yielded to the consumer.
        # Finally (always): if cancelled, appends interrupted results for dangling
        # calls again and pushes start/end message events for them to listeners,
        # suppressing listener exceptions; clears the current signal if still its own;
        # clears the running flag.
        # Inputs: keyword-only prompts. Outputs: yields AgentEvents.
        # Exceptions: loop/provider and listener errors during the run propagate.
        signal = SimpleCancellationToken()
        self._current_signal = signal
        try:
            # Repair dangling tool calls here, not in prompt()/continue_(),
            # so the synthetic results flow through events and reach push
            # subscribers (persistence) as well as the consumer.
            repaired_from = len(self._messages)
            self._append_interrupted_tool_results()
            repairs = self._messages[repaired_from:]
            async for event in run_agent_loop(
                provider=self._config.provider,
                model=self._config.model,
                system=self._config.system,
                messages=self._messages,
                prompts=prompts,
                prelude_messages=repairs,
                tools=self._config.tools,
                max_turns=self._config.max_turns,
                signal=signal,
                session_id=self._config.session_id,
                get_steering_messages=self._drain_steering_messages,
                get_follow_up_messages=self._drain_follow_up_messages,
                before_tool_call=self._config.before_tool_call,
                after_tool_call=self._config.after_tool_call,
            ):
                await self._notify(event)
                yield event
        finally:
            if signal.is_cancelled():
                repaired_from = len(self._messages)
                self._append_interrupted_tool_results()
                # The consumer is usually gone here; push the repairs to
                # subscribers. Listener errors are suppressed; cancellation
                # itself is not.
                for message in self._messages[repaired_from:]:
                    with suppress(Exception):
                        await self._notify(MessageStartEvent(message=message))
                        await self._notify(MessageEndEvent(message=message))
            if self._current_signal is signal:
                self._current_signal = None
            self._running = False

    async def _notify(self, event: AgentEvent) -> None:
        # [FUNCTION DOC]
        # HELPER: Calls every listener (iterating over a copy so listeners may
        # unsubscribe) with the event, awaiting awaitable results sequentially.
        # Listener exceptions propagate.
        for listener in list(self._listeners):
            result = listener(event)
            if isawaitable(result):
                await result

    def _ensure_not_running(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Raises RuntimeError advising steer()/follow_up() when a run is
        # already active; otherwise does nothing.
        if self._running:
            raise RuntimeError(
                "AgentHarness is already running; use steer() or follow_up() to queue messages."
            )

    def _drain_steering_messages(self) -> tuple[AgentMessage, ...]:
        # [FUNCTION DOC]
        # HELPER: Loop callback draining the steering queue per queue mode.
        return self._drain_queue(self._steering_queue)

    def _drain_follow_up_messages(self) -> tuple[AgentMessage, ...]:
        # [FUNCTION DOC]
        # HELPER: Loop callback draining the follow-up queue per queue mode.
        return self._drain_queue(self._follow_up_queue)

    def _drain_queue(self, queue: deque[AgentMessage]) -> tuple[AgentMessage, ...]:
        # [FUNCTION DOC]
        # HELPER: Drains a deque according to the configured queue mode: empty tuple
        # when empty; in "all" mode returns and clears everything; otherwise pops and
        # returns only the oldest message. Mutates the deque.
        if not queue:
            return ()
        if self._config.queue_mode == "all":
            messages = tuple(queue)
            queue.clear()
            return messages
        return (queue.popleft(),)

    def append_interrupted_tool_results(self) -> int:
        # [FUNCTION DOC]
        # Public wrapper that appends synthetic interrupted results for dangling tool
        # calls (no events emitted) and returns how many messages were added.
        before = len(self._messages)
        self._append_interrupted_tool_results()
        return len(self._messages) - before

    def _append_interrupted_tool_results(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Collects the ids of all existing tool results, then for each tool
        # call of each assistant message without a result appends, at the end of the
        # transcript, an error ToolResultMessage with text "Tool call interrupted by
        # user". Each id is repaired once. Mutates the transcript; no events.
        # KNOWN ISSUE: results are appended at the transcript end rather than adjacent
        # to their call, so an older dangling call followed by later messages yields
        # a non-adjacent result; providers depend on tool_history repair to fix order.
        returned_ids = {
            message.tool_call_id
            for message in self._messages
            if isinstance(message, ToolResultMessage)
        }
        for message in tuple(self._messages):
            if not isinstance(message, AssistantMessage):
                continue
            for call in message.tool_calls:
                if call.id in returned_ids:
                    continue
                returned_ids.add(call.id)
                self._messages.append(
                    ToolResultMessage(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        content=[TextContent(text="Tool call interrupted by user")],
                        is_error=True,
                    )
                )
