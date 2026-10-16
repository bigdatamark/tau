# [MODULE DOC]
# Deterministic fake model provider used by tests of the agent loop (tau_ai layer).
# It satisfies the same streaming-provider shape as real providers: a stream_response
# method returning an async iterator of assistant message events, but instead of
# calling a network API it replays pre-scripted event lists in order. Records every
# call so tests can assert on the model, system prompt, messages, tools and session id.
"""Deterministic Pi-compatible model provider for tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable

from tau_agent.messages import AgentMessage
from tau_agent.tools import AgentTool
from tau_ai.events import AssistantMessageEvent
from tau_ai.provider import CancellationToken


class FakeProvider:
    # [CLASS DOC]
    # HELPER: Test provider that replays predefined assistant event streams, one per call.
    # State: a queue of scripted streams (each a list of assistant events), a list of
    # recorded calls (model, system, copied message list, copied tool list), and a list
    # of recorded session ids. Each stream_response call consumes the next scripted stream;
    # when the queue is exhausted it yields an empty stream.
    """A provider that replays predefined assistant event streams."""

    def __init__(self, streams: Iterable[Iterable[AssistantMessageEvent]]) -> None:
        # [FUNCTION DOC]
        # HELPER: Store the scripted streams and initialise call recording.
        # Inputs: an iterable of iterables of assistant message events; each inner iterable
        # is materialised into a list so it can be replayed once. Output: none.
        # Side effects: initialises the stream queue, an empty calls list and an empty
        # session_ids list. Raises nothing itself (errors only if the iterables raise).
        self._streams = [list(stream) for stream in streams]
        self.calls: list[tuple[str, str, list[AgentMessage], list[AgentTool]]] = []
        self.session_ids: list[str | None] = []

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        # [FUNCTION DOC]
        # HELPER: Record the request and return an async iterator over the next scripted stream.
        # Inputs (keyword-only): model name, system prompt, conversation messages, available
        # tools, optional cancellation token, optional session id.
        # Behavior: appends a tuple of (model, system, shallow copy of messages, shallow copy
        # of tools) to calls and the session id to session_ids, then pops the first scripted
        # stream from the queue (or uses an empty list when none remain). This happens eagerly
        # at call time, not when iteration begins.
        # Output: an async iterator yielding the scripted events in order.
        # Side effects: mutates the calls, session_ids and stream queue. Raises nothing.
        self.calls.append((model, system, list(messages), list(tools)))
        self.session_ids.append(session_id)
        stream = self._streams.pop(0) if self._streams else []

        async def iterator() -> AsyncIterator[AssistantMessageEvent]:
            # [FUNCTION DOC]
            # HELPER: Async generator yielding each scripted event of the popped stream.
            # Before each event it checks the cancellation token (if provided); once the token
            # reports cancelled, the generator stops silently without yielding further events
            # and without raising. Inputs come from the enclosing closure; yields assistant events.
            for event in stream:
                if signal is not None and signal.is_cancelled():
                    return
                yield event

        return iterator()
