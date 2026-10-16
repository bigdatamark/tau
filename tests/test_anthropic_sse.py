# [MODULE DOC]
# Tests for the Anthropic provider's server-sent-events parsing, using an httpx
# mock transport instead of the network.
from collections.abc import AsyncIterator

import httpx
import pytest

from tau_agent import UserMessage
from tau_ai import AnthropicConfig, AnthropicProvider, AssistantDoneEvent


async def _collect(stream: AsyncIterator[object]) -> list[object]:
    # [FUNCTION DOC]
    # HELPER: Drain an async iterator into a list.
    # Input: any async iterator. Output: list of all yielded items in order.
    # Raises whatever the iterator raises; consumes the iterator.
    return [event async for event in stream]


@pytest.mark.anyio
async def test_anthropic_provider_ignores_empty_data_frames() -> None:
    # [FUNCTION DOC]
    # Test that an SSE stream containing an empty "data:" frame between real events
    # is skipped rather than failing JSON parsing, and the provider still finishes
    # with an assistant-done event whose message text is "ok".
    # Uses a mock HTTP client returning a canned event stream (message_start, empty
    # frame, one text delta, message_delta with end_turn, message_stop).
    # Raises an assertion error on failure; no I/O beyond the in-memory mock.
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock transport handler; ignores the request and returns HTTP 200 with
        # an event-stream content type and the canned SSE body described above.
        return httpx.Response(
            200,
            text=(
                'data: {"type":"message_start","message":{"usage":{}}}\n\n'
                "data:\n\n"
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"ok"}}\n\n'
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
                '"usage":{"output_tokens":1}}\n\n'
                'data: {"type":"message_stop"}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            AnthropicConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
            ),
            client=client,
        )
        events = await _collect(
            provider.stream_response(
                model="claude-test",
                system="You are Tau.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert isinstance(events[-1], AssistantDoneEvent)
    assert events[-1].message.text == "ok"
