# [MODULE DOC]
# Private bridge that converts Tau's transitional internal provider-parser
# events (response start/end, text and thinking deltas, tool calls, retries,
# errors) into Pi-style assistant message stream events (start, block
# start/delta/end for text and thinking, tool call start/end, done, error).
# Maintains a growing partial assistant message and attaches a deep-copied
# snapshot of it to every emitted event so consumers never see later mutation.
"""Translate Tau's transitional provider parser output into Pi stream events."""

from __future__ import annotations

from collections.abc import AsyncIterator

from tau_agent.messages import (
    AssistantMessage,
    AssistantMessageDiagnostic,
    TextContent,
    ThinkingContent,
    Usage,
)
from tau_ai._provider_events import (
    ProviderErrorEvent,
    ProviderEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderRetryEvent,
    ProviderTextDeltaEvent,
    ProviderThinkingDeltaEvent,
    ProviderToolCallEvent,
)
from tau_ai.events import (
    AssistantDoneEvent,
    AssistantErrorEvent,
    AssistantMessageEvent,
    AssistantStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)


def _snapshot(message: AssistantMessage) -> AssistantMessage:
    # [FUNCTION DOC]
    # HELPER: returns a deep copy of the partial assistant message for embedding in
    # an event. No side effects.
    return message.model_copy(deep=True)


async def _end_active_block(
    partial: AssistantMessage,
    index: int | None,
) -> AsyncIterator[AssistantMessageEvent]:
    # [FUNCTION DOC]
    # HELPER: async generator that closes the block at the given content index.
    # Inputs: the partial message and an optional content index. Yields a text-end
    # event (with the full block text) for a text block, a thinking-end event (with
    # the full thinking text) for a thinking block, and nothing when the index is
    # none or the block is another kind (for example a tool call).
    # Raises IndexError if the index is out of range. No mutation.
    """End the active text/thinking block before the provider changes channels."""
    if index is None:
        return
    block = partial.content[index]
    if isinstance(block, TextContent):
        yield TextEndEvent(
            content_index=index,
            content=block.text,
            partial=_snapshot(partial),
        )
    elif isinstance(block, ThinkingContent):
        yield ThinkingEndEvent(
            content_index=index,
            content=block.thinking,
            partial=_snapshot(partial),
        )


def _copy_replay_metadata(target: AssistantMessage, source: AssistantMessage) -> None:
    # [FUNCTION DOC]
    # HELPER: copies provider replay metadata from the parser's final message onto
    # the streamed final message, pairing blocks positionally per kind: the n-th
    # thinking block gets the source's thinking signature and redacted flag, and
    # the n-th text block gets the source's text signature. Extra blocks on either
    # side are ignored. Returns nothing; mutates the target's blocks in place.
    """Copy provider metadata onto streamed blocks without changing their order."""
    source_thinking = [block for block in source.content if isinstance(block, ThinkingContent)]
    target_thinking = [block for block in target.content if isinstance(block, ThinkingContent)]
    for target_block, source_block in zip(target_thinking, source_thinking, strict=False):
        target_block.thinking_signature = source_block.thinking_signature
        target_block.redacted = source_block.redacted

    source_text = [block for block in source.content if isinstance(block, TextContent)]
    target_text = [block for block in target.content if isinstance(block, TextContent)]
    for target_text_block, source_text_block in zip(target_text, source_text, strict=False):
        target_text_block.text_signature = source_text_block.text_signature


def _finish_reason(value: str | None, *, has_tools: bool) -> str:
    # [FUNCTION DOC]
    # HELPER: maps a provider finish reason to a Pi stop reason. Returns
    # "toolUse" when the message has tool calls or the value is a tool-call reason
    # (tool_calls, tool_use, toolUse); "length" for length, max_tokens, MAX_TOKENS
    # or incomplete; otherwise "stop". No side effects.
    if has_tools or value in {"tool_calls", "tool_use", "toolUse"}:
        return "toolUse"
    if value in {"length", "max_tokens", "MAX_TOKENS", "incomplete"}:
        return "length"
    return "stop"


async def canonicalize_provider_stream(
    source: AsyncIterator[ProviderEvent],
    *,
    api: str,
    provider: str,
    model: str,
    independent_channels: bool = False,
) -> AsyncIterator[AssistantMessageEvent]:
    # [FUNCTION DOC]
    # Async generator that canonicalizes one internal provider stream into Pi
    # assistant events.
    # Inputs: the source async iterator of provider events, keyword api, provider
    # and model names stamped onto messages, and independent_channels (true for
    # Chat Completions, where text and thinking may interleave without ending each
    # other's block).
    # Behavior:
    # - Retry events are dropped. A start event is emitted once, either on the
    #   provider response-start event (recording its response provider) or lazily
    #   before the first other event.
    # - Text/thinking deltas: in sequential mode, a change of kind ends the active
    #   block and starts a new one; in independent mode each kind reuses its own
    #   open block per channel. Deltas are appended to the block text and a delta
    #   event is emitted.
    # - Tool call: ends all open text/thinking blocks, appends a deep copy of the
    #   tool call to the content and emits tool-call start then end immediately.
    # - Response end: ends open blocks, then builds the final message from the
    #   parser's message (for usage/metadata) but with the streamed content order
    #   (falling back to the parser's content if nothing streamed), stamps
    #   api/provider/model and response provider, copies replay signatures, sets
    #   the stop reason and emits a done event.
    # - Error event: emits an error event carrying a copy of the partial message
    #   with stop reason error, the error text and a provider_error diagnostic
    #   holding the event data.
    # - If the source ends without any event a start is still emitted; if no done
    #   or error was emitted, a synthetic error ("Provider stream ended without a
    #   terminal event") with zeroed usage is emitted.
    # Yields: assistant message events. Raises whatever the source iterator raises.
    # Side effects: none beyond consuming the source.
    # KNOWN ISSUE: after a terminal event the loop keeps consuming and may emit
    # further events (including a second done/error) if the source continues; an
    # error event also does not emit end events for still-open blocks.
    """Canonicalize one old internal parser stream.

    Provider parsers remain isolated behind this private bridge while they are
    migrated incrementally. The public provider protocol exposes only Pi events.
    Chat Completions uses independent channels: a channel switch does not end
    either block. Other transports preserve their sequential block ordering.
    """
    partial = AssistantMessage(api=api, provider=provider, model=model)
    active_index: int | None = None
    active_kind: str | None = None
    channel_indices: dict[str, int] = {}
    started = False
    terminal = False

    async for event in source:
        if isinstance(event, ProviderRetryEvent):
            # Retries are provider-internal at the Pi AI boundary.
            continue
        if isinstance(event, ProviderResponseStartEvent):
            if event.response_provider is not None:
                partial.response_provider = event.response_provider
            if not started:
                started = True
                yield AssistantStartEvent(partial=_snapshot(partial))
            continue
        if not started:
            started = True
            yield AssistantStartEvent(partial=_snapshot(partial))

        if isinstance(event, ProviderTextDeltaEvent):
            if independent_channels:
                active_index = channel_indices.get("text")
                active_kind = "text" if active_index is not None else None
            if active_kind != "text":
                if not independent_channels:
                    async for end_event in _end_active_block(partial, active_index):
                        yield end_event
                active_index = len(partial.content)
                active_kind = "text"
                partial.content.append(TextContent(text=""))
                channel_indices["text"] = active_index
                yield TextStartEvent(content_index=active_index, partial=_snapshot(partial))
            assert active_index is not None
            block = partial.content[active_index]
            assert isinstance(block, TextContent)
            block.text += event.delta
            yield TextDeltaEvent(
                content_index=active_index,
                delta=event.delta,
                partial=_snapshot(partial),
            )
        elif isinstance(event, ProviderThinkingDeltaEvent):
            if independent_channels:
                active_index = channel_indices.get("thinking")
                active_kind = "thinking" if active_index is not None else None
            if active_kind != "thinking":
                if not independent_channels:
                    async for end_event in _end_active_block(partial, active_index):
                        yield end_event
                active_index = len(partial.content)
                active_kind = "thinking"
                partial.content.append(ThinkingContent(thinking=""))
                channel_indices["thinking"] = active_index
                yield ThinkingStartEvent(
                    content_index=active_index,
                    partial=_snapshot(partial),
                )
            assert active_index is not None
            block = partial.content[active_index]
            assert isinstance(block, ThinkingContent)
            block.thinking += event.delta
            yield ThinkingDeltaEvent(
                content_index=active_index,
                delta=event.delta,
                partial=_snapshot(partial),
            )
        elif isinstance(event, ProviderToolCallEvent):
            indices = list(channel_indices.values()) if independent_channels else [active_index]
            for index in indices:
                async for end_event in _end_active_block(partial, index):
                    yield end_event
            channel_indices.clear()
            active_index = None
            active_kind = None
            index = len(partial.content)
            partial.content.append(event.tool_call.model_copy(deep=True))
            yield ToolCallStartEvent(content_index=index, partial=_snapshot(partial))
            yield ToolCallEndEvent(
                content_index=index,
                tool_call=event.tool_call,
                partial=_snapshot(partial),
            )
        elif isinstance(event, ProviderResponseEndEvent):
            indices = list(channel_indices.values()) if independent_channels else [active_index]
            for index in indices:
                async for end_event in _end_active_block(partial, index):
                    yield end_event
            channel_indices.clear()
            active_index = None
            active_kind = None

            # Preserve the exact streamed content order. The parser's final
            # message remains authoritative only for response metadata/usage.
            final = event.message.model_copy(deep=True)
            final.api = api
            final.provider = provider
            final.model = model
            final.response_provider = event.message.response_provider or partial.response_provider
            final.content = [block.model_copy(deep=True) for block in partial.content]
            if not final.content and event.message.content:
                final.content = [block.model_copy(deep=True) for block in event.message.content]
            _copy_replay_metadata(final, event.message)
            final.stop_reason = _finish_reason(
                event.finish_reason,
                has_tools=bool(final.tool_calls),
            )  # type: ignore[assignment]
            yield AssistantDoneEvent(reason=final.stop_reason, message=final)  # type: ignore[arg-type]
            terminal = True
        elif isinstance(event, ProviderErrorEvent):
            error = partial.model_copy(deep=True)
            error.response_provider = event.response_provider or partial.response_provider
            error.stop_reason = "error"
            error.error_message = event.message
            error.diagnostics = [
                AssistantMessageDiagnostic(type="provider_error", details=event.data)
            ]
            yield AssistantErrorEvent(reason="error", error=error)
            terminal = True

    if not started:
        yield AssistantStartEvent(partial=_snapshot(partial))
    if not terminal:
        error = partial.model_copy(deep=True)
        error.stop_reason = "error"
        error.error_message = "Provider stream ended without a terminal event"
        error.usage = Usage()
        yield AssistantErrorEvent(reason="error", error=error)
