# [MODULE DOC]
# Mistral provider adapter (tau_ai layer) for Mistral's streaming chat
# completions API (the "mistral-conversations" API id). Converts Tau agent
# messages and tools into Mistral's JSON request format, streams the SSE
# response with retries, parses chunks into provider-neutral events and hands
# them to canonicalize_provider_stream, which produces Pi-compatible assistant
# message events.
# Side effects: network I/O only when streaming.
"""Mistral Conversations provider."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from json import JSONDecodeError, dumps, loads
from typing import Any, Protocol

import httpx

from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    ImageContent,
    ThinkingContent,
    ToolResultMessage,
    UserMessage,
    assistant_content,
    message_to_user,
)
from tau_agent.tools import AgentTool, ToolCall
from tau_agent.types import JSONValue
from tau_ai._provider_events import (
    ProviderErrorEvent,
    ProviderEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderTextDeltaEvent,
    ProviderThinkingDeltaEvent,
    ProviderToolCallEvent,
)
from tau_ai.content import (
    NON_VISION_TOOL_IMAGE_PLACEHOLDER,
    NON_VISION_USER_IMAGE_PLACEHOLDER,
    text_and_images,
)
from tau_ai.env import OpenAICompatibleConfig
from tau_ai.events import AssistantMessageEvent
from tau_ai.http import create_async_client
from tau_ai.http_errors import provider_http_error_message
from tau_ai.provider import CancellationToken
from tau_ai.retry import (
    RETRYABLE_TRANSPORT_ERRORS,
    provider_retry_event,
    retry_delay_seconds,
    wait_for_retry,
)
from tau_ai.stream import canonicalize_provider_stream
from tau_ai.tool_call_ids import portable_tool_call_id


class MistralConversationsProvider:
    # [CLASS DOC]
    # ModelProvider implementation for Mistral. Holds an OpenAI-compatible style
    # config (base URL, API key, extra headers, reasoning effort, max tokens, image
    # support, timeouts, retry limits, provider display name) and an optional
    # injected HTTP client; it owns (and will close) a lazily created client when
    # none was injected.
    """Provider adapter for Mistral's streaming chat API."""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Stores config and optional client; records ownership when no client given.
        self._config = config
        self._client = client
        self._owns_client = client is None

    async def aclose(self) -> None:
        # [FUNCTION DOC]
        # Closes and forgets the HTTP client only if this provider created it.
        # Side effects: network client shutdown. Injected clients are left open.
        """Close the underlying HTTP client if this provider created it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

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
        # Public provider entry point: streams one model response.
        # Inputs: model id, system prompt, message list, tools, optional cancellation
        # token, session id (ignored).
        # Outputs: async iterator of canonical AssistantMessageEvents (start, deltas,
        # tool calls, done/error) tagged with api "mistral-conversations", provider
        # "mistral" and the model. Lazy: no request until iterated.
        """Stream one response as Pi-compatible assistant message events."""
        del session_id
        raw = self._stream_provider_events(
            model=model, system=system, messages=messages, tools=tools, signal=signal
        )
        return canonicalize_provider_stream(
            raw, api="mistral-conversations", provider="mistral", model=model
        )

    def _stream_provider_events(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        # [FUNCTION DOC]
        # HELPER: Builds the request payload from config (reasoning effort, max
        # tokens, image support) and returns the raw provider event stream for POST
        # <base>/v1/chat/completions.
        """Stream one Mistral response as provider-neutral events."""
        payload = _build_mistral_payload(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            reasoning_effort=self._config.reasoning_effort,
            max_tokens=self._config.max_tokens,
            supports_images=self._config.supports_images,
        )
        return self._stream(
            model=model,
            url=f"{_mistral_base_url(self._config.base_url)}/chat/completions",
            payload=payload,
            signal=signal,
        )

    def _stream(
        self,
        *,
        model: str,
        url: str,
        payload: Mapping[str, JSONValue],
        signal: CancellationToken | None,
    ) -> AsyncIterator[ProviderEvent]:
        # [FUNCTION DOC]
        # HELPER: Returns an async iterator performing the HTTP streaming request with
        # retry logic (see the nested iterator).
        async def iterator() -> AsyncIterator[ProviderEvent]:
            # [FUNCTION DOC]
            # HELPER: Async generator doing the request loop. Headers = configured extra
            # headers plus a Bearer Authorization with the API key. For each attempt it
            # creates a fresh stream parser and POSTs the JSON payload.
            # On HTTP status >= 400: reads the body; if retryable (408, 409, 425, 429 or
            # >= 500, under max retries) yields a provider retry event with computed
            # backoff delay, waits (returns silently if cancelled during the wait) and
            # retries; otherwise yields a ProviderErrorEvent with a friendly HTTP message
            # and status/body data, then stops.
            # On success: yields a response start event, then for each SSE "data:" line
            # (stopping silently if cancelled) feeds the parser, yielding its events until
            # "[DONE]"; finally yields the parser's finalize events (tool calls and the
            # response end event).
            # On retryable transport errors: retries with a retry event only if no content
            # was emitted yet and attempts remain; otherwise yields an error event with the
            # message and attempt count.
            # Side effects: network I/O, sleeping between retries.
            # KNOWN ISSUE: when cancelled mid-stream it returns without any end or error
            # event; downstream canonicalization must handle the truncated stream.
            client = self._get_client()
            headers = {
                **dict(self._config.headers or {}),
                "Authorization": f"Bearer {self._config.api_key}",
            }
            attempt = 0
            while True:
                parser = _MistralStreamParser()
                try:
                    async with client.stream(
                        "POST", url, json=payload, headers=headers
                    ) as response:
                        if response.status_code >= 400:
                            body = await response.aread()
                            body_text = body.decode(errors="replace")
                            if self._should_retry(attempt, status_code=response.status_code):
                                delay = retry_delay_seconds(
                                    attempt,
                                    max_delay_seconds=self._config.max_retry_delay_seconds,
                                )
                                yield provider_retry_event(
                                    attempt=attempt,
                                    max_retries=self._config.max_retries,
                                    delay_seconds=delay,
                                    reason=f"HTTP {response.status_code}",
                                    data={"status_code": response.status_code, "body": body_text},
                                )
                                attempt += 1
                                if not await wait_for_retry(delay, signal=signal):
                                    return
                                continue
                            yield ProviderErrorEvent(
                                message=provider_http_error_message(
                                    provider_name=self._config.provider_name,
                                    status_code=response.status_code,
                                    body=body_text,
                                    model=model,
                                ),
                                data={"status_code": response.status_code, "body": body_text},
                            )
                            return

                        yield ProviderResponseStartEvent(model=model)
                        async for line in response.aiter_lines():
                            if signal is not None and signal.is_cancelled():
                                return
                            event = _parse_sse_line(line)
                            if event is None:
                                continue
                            events, stop = parser.feed(event)
                            for parser_event in events:
                                yield parser_event
                            if stop:
                                break
                        for parser_event in parser.finalize():
                            yield parser_event
                        return
                except RETRYABLE_TRANSPORT_ERRORS as exc:
                    if not parser.emitted_content and self._should_retry(attempt):
                        delay = retry_delay_seconds(
                            attempt,
                            max_delay_seconds=self._config.max_retry_delay_seconds,
                        )
                        yield provider_retry_event(
                            attempt=attempt,
                            max_retries=self._config.max_retries,
                            delay_seconds=delay,
                            reason="network error",
                            data={"error": str(exc), "error_type": type(exc).__name__},
                        )
                        attempt += 1
                        if not await wait_for_retry(delay, signal=signal):
                            return
                        continue
                    yield ProviderErrorEvent(message=str(exc), data={"attempts": attempt + 1})
                    return

        return iterator()

    def _get_client(self) -> httpx.AsyncClient:
        # [FUNCTION DOC]
        # HELPER: Returns the existing client or creates (and caches) one with the
        # configured timeout.
        if self._client is None:
            self._client = create_async_client(timeout=self._config.timeout_seconds)
        return self._client

    def _should_retry(self, attempt: int, *, status_code: int | None = None) -> bool:
        # [FUNCTION DOC]
        # HELPER: False once attempt reaches max retries; otherwise true for transport
        # errors (no status) and statuses 408, 409, 425, 429 or >= 500.
        if attempt >= self._config.max_retries:
            return False
        return status_code is None or status_code in {408, 409, 425, 429} or status_code >= 500


class _StreamParser(Protocol):
    # [CLASS DOC]
    # HELPER: Structural protocol for a stream parser: an emitted_content flag, a
    # feed method taking one SSE data payload and returning events plus a stop
    # flag, and a finalize method returning closing events.
    emitted_content: bool

    # [FUNCTION DOC]
    # Protocol signature: consume one SSE data string; return events and whether
    # the stream is done.
    def feed(self, event: str) -> tuple[list[ProviderEvent], bool]: ...

    # [FUNCTION DOC]
    # Protocol signature: return the closing events after the stream ends.
    def finalize(self) -> list[ProviderEvent]: ...


class _MistralStreamParser:
    # [CLASS DOC]
    # HELPER: Incremental parser for Mistral streaming chunks. Accumulates text
    # parts, thinking parts, per-index tool call builders, the latest finish
    # reason and whether any content has been emitted (used to decide retries).
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Initialises empty accumulators and flags.
        self.emitted_content = False
        self._content_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._tool_call_builders: dict[int, _ToolCallBuilder] = {}
        self._finish_reason: str | None = None

    def feed(self, event: str) -> tuple[list[ProviderEvent], bool]:
        # [FUNCTION DOC]
        # Consumes one SSE data payload. "[DONE]" returns no events and stop=true.
        # Invalid JSON, non-object chunks, missing choices or non-mapping delta yield
        # no events. Otherwise records finish reason (snake or camel case, keeping the
        # previous one when absent), emits a text delta event per text fragment and a
        # thinking delta event per thinking fragment, and merges tool call deltas into
        # builders keyed by their index (default 0). Marks emitted_content when any
        # text, thinking or tool call delta appears. Returns events and stop=false.
        # Raises ValueError if a tool call index is not int-convertible.
        if event == "[DONE]":
            return [], True
        chunk = _loads_object(event)
        if chunk is None:
            return [], False
        choice = _first_choice(chunk)
        if choice is None:
            return [], False
        self._finish_reason = (
            choice.get("finish_reason") or choice.get("finishReason") or self._finish_reason
        )
        delta = choice.get("delta")
        if not isinstance(delta, Mapping):
            return [], False
        events: list[ProviderEvent] = []
        for content in _content_deltas(delta):
            self.emitted_content = True
            self._content_parts.append(content)
            events.append(ProviderTextDeltaEvent(delta=content))
        for thinking in _thinking_deltas(delta):
            self.emitted_content = True
            self._thinking_parts.append(thinking)
            events.append(ProviderThinkingDeltaEvent(delta=thinking))
        for tool_call_delta in _tool_call_deltas(delta):
            self.emitted_content = True
            index = int(tool_call_delta.get("index", 0))
            builder = self._tool_call_builders.setdefault(index, _ToolCallBuilder())
            builder.add_delta(tool_call_delta)
        return events, False

    def finalize(self) -> list[ProviderEvent]:
        # [FUNCTION DOC]
        # Builds tool calls from builders in index order, emits one tool call event per
        # call, then a response end event with an AssistantMessage whose content is
        # the joined text plus tool calls, with a joined thinking block prepended when
        # any thinking was seen. Finish reason is the recorded one, else "tool_calls"
        # when tool calls exist, else "stop".
        tool_calls = [
            builder.build(index) for index, builder in sorted(self._tool_call_builders.items())
        ]
        events: list[ProviderEvent] = [
            ProviderToolCallEvent(tool_call=tool_call) for tool_call in tool_calls
        ]
        content = assistant_content("".join(self._content_parts), tool_calls)
        if self._thinking_parts:
            content.insert(0, ThinkingContent(thinking="".join(self._thinking_parts)))
        events.append(
            ProviderResponseEndEvent(
                message=AssistantMessage(content=content),
                finish_reason=self._finish_reason or ("tool_calls" if tool_calls else "stop"),
            )
        )
        return events


class _ToolCallBuilder:
    # [CLASS DOC]
    # HELPER: Accumulates one streamed tool call: id, function name and argument
    # text fragments.
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Initialises empty id, name and argument parts.
        self.id = ""
        self.name = ""
        self.arguments_parts: list[str] = []

    def add_delta(self, delta: Mapping[str, Any]) -> None:
        # [FUNCTION DOC]
        # Merges one tool call delta: adopts a string id unless it is the literal
        # "null"; if a function mapping is present adopts its name and appends string
        # arguments, or JSON-encodes object arguments and appends them.
        call_id = delta.get("id")
        if isinstance(call_id, str) and call_id != "null":
            self.id = call_id
        function = delta.get("function")
        if not isinstance(function, Mapping):
            return
        name = function.get("name")
        if isinstance(name, str):
            self.name = name
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            self.arguments_parts.append(arguments)
        elif isinstance(arguments, Mapping):
            self.arguments_parts.append(dumps(arguments))

    def build(self, index: int) -> ToolCall:
        # [FUNCTION DOC]
        # Creates a ToolCall: id falls back to "tool-call-<index>"; arguments are the
        # parsed joined JSON object, an empty mapping when no text, or a mapping with
        # "_raw_arguments" holding the raw text when it is not a JSON object.
        arguments_text = "".join(self.arguments_parts)
        arguments = _loads_object(arguments_text) if arguments_text else {}
        if arguments is None:
            arguments = {"_raw_arguments": arguments_text}
        return ToolCall(
            id=self.id or f"tool-call-{index}",
            name=self.name,
            arguments=arguments,
        )


def _build_mistral_payload(
    *,
    model: str,
    system: str,
    messages: list[AgentMessage],
    tools: list[AgentTool],
    reasoning_effort: str | None,
    max_tokens: int | None,
    supports_images: bool = False,
) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Builds the request JSON: model, stream true, system message (if any)
    # followed by converted messages; max_tokens when set; tools when any. If a
    # reasoning effort other than "none" is configured, sets reasoning_effort to
    # "high" for models known to accept it, otherwise prompt_mode "reasoning".
    # KNOWN ISSUE: the configured effort level is ignored; any non-"none" effort
    # is sent as "high".
    payload: dict[str, JSONValue] = {
        "model": model,
        "stream": True,
        "messages": [
            *_system_messages(system),
            *_messages_to_mistral(messages, supports_images=supports_images),
        ],
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = [_tool_to_mistral(tool) for tool in tools]
    if reasoning_effort is not None and reasoning_effort != "none":
        if _uses_reasoning_effort(model):
            payload["reasoning_effort"] = "high"
        else:
            payload["prompt_mode"] = "reasoning"
    return payload


def _system_messages(system: str) -> list[dict[str, JSONValue]]:
    # [FUNCTION DOC]
    # HELPER: Returns a single system-role message list for a non-empty prompt,
    # else an empty list.
    return [{"role": "system", "content": system}] if system else []


def _messages_to_mistral(
    messages: list[AgentMessage], *, supports_images: bool
) -> list[dict[str, JSONValue]]:
    # [FUNCTION DOC]
    # HELPER: Converts agent messages to Mistral messages. User messages become
    # plain text, or a content list of text plus data-URL image parts when images
    # are supported/present (unsupported images become a placeholder in text).
    # Tool results become tool-role messages with portable call id, tool name and
    # text (or "(see attached image)" / "(no tool output)"); their images are
    # collected and emitted as one follow-up user message once a non-tool-result
    # message (or the end) is reached, since tool messages cannot carry images.
    # Other messages go through _message_to_mistral.
    converted: list[dict[str, JSONValue]] = []
    pending_tool_images: list[ImageContent] = []
    for message in messages:
        if pending_tool_images and not isinstance(message, ToolResultMessage):
            converted.append(_mistral_tool_image_message(pending_tool_images))
            pending_tool_images = []
        if isinstance(message, UserMessage):
            text, images = text_and_images(
                message.content,
                supports_images=supports_images,
                image_placeholder=NON_VISION_USER_IMAGE_PLACEHOLDER,
            )
            if images:
                content: list[JSONValue] = []
                if text:
                    content.append({"type": "text", "text": text})
                content.extend(
                    {
                        "type": "image_url",
                        "image_url": f"data:{image.mime_type};base64,{image.data}",
                    }
                    for image in images
                )
                converted.append({"role": "user", "content": content})
            else:
                converted.append({"role": "user", "content": text})
            continue
        if isinstance(message, ToolResultMessage):
            text, images = text_and_images(
                message.content,
                supports_images=supports_images,
                image_placeholder=NON_VISION_TOOL_IMAGE_PLACEHOLDER,
            )
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": portable_tool_call_id(message.tool_call_id),
                    "name": message.tool_name,
                    "content": text or ("(see attached image)" if images else "(no tool output)"),
                }
            )
            pending_tool_images.extend(images)
            continue
        converted.append(_message_to_mistral(message))
    if pending_tool_images:
        converted.append(_mistral_tool_image_message(pending_tool_images))
    return converted


def _mistral_tool_image_message(images: list[ImageContent]) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Builds a user message containing the text "Attached image(s) from
    # tool result:" followed by data-URL image parts for each image.
    content: list[JSONValue] = [{"type": "text", "text": "Attached image(s) from tool result:"}]
    content.extend(
        {
            "type": "image_url",
            "image_url": f"data:{image.mime_type};base64,{image.data}",
        }
        for image in images
    )
    return {"role": "user", "content": content}


def _message_to_mistral(message: AgentMessage) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Converts one message: user -> user text; assistant -> assistant text
    # plus reasoning_content when thinking exists and tool_calls when present; tool
    # result -> tool message; any other message kind is first converted to a user
    # message and then converted recursively.
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.text}
    if isinstance(message, AssistantMessage):
        item: dict[str, JSONValue] = {"role": "assistant", "content": message.text}
        if message.thinking_text:
            item["reasoning_content"] = message.thinking_text
        if message.tool_calls:
            item["tool_calls"] = [
                _tool_call_to_mistral(tool_call) for tool_call in message.tool_calls
            ]
        return item
    if isinstance(message, ToolResultMessage):
        return {
            "role": "tool",
            "tool_call_id": portable_tool_call_id(message.tool_call_id),
            "name": message.tool_name,
            "content": message.text,
        }
    return _message_to_mistral(message_to_user(message))


def _tool_to_mistral(tool: AgentTool) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Converts an AgentTool to a non-strict function tool definition with
    # name, description and a copy of its input JSON schema.
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.input_schema),
            "strict": False,
        },
    }


def _tool_call_to_mistral(tool_call: ToolCall) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Converts a ToolCall to Mistral's function call format with portable
    # id and JSON-encoded arguments.
    return {
        "id": portable_tool_call_id(tool_call.id),
        "type": "function",
        "function": {"name": tool_call.name, "arguments": dumps(tool_call.arguments)},
    }


def _mistral_base_url(base_url: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Strips trailing slashes and appends "/v1" unless already present.
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _uses_reasoning_effort(model: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: True for the model ids that accept a reasoning_effort parameter
    # (mistral-small-2603, mistral-small-latest, mistral-medium-3.5).
    return model in {"mistral-small-2603", "mistral-small-latest", "mistral-medium-3.5"}


def _parse_sse_line(line: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Returns the stripped payload after "data:" for SSE data lines, else
    # none (blank lines, comments, other fields).
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    return line.removeprefix("data:").strip()


def _loads_object(value: str) -> dict[str, JSONValue] | None:
    # [FUNCTION DOC]
    # HELPER: Parses JSON text and returns it only if it is an object; none on
    # decode errors or other JSON types.
    try:
        loaded = loads(value)
    except JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _first_choice(chunk: Mapping[str, Any]) -> Mapping[str, Any] | None:
    # [FUNCTION DOC]
    # HELPER: Returns the first element of a chunk's non-empty choices list when it
    # is a mapping, else none.
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    return choice if isinstance(choice, Mapping) else None


def _content_deltas(delta: Mapping[str, Any]) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Extracts text fragments from a delta's content: a non-empty string,
    # or from a list the non-empty strings and "text" typed items' text.
    content = delta.get("content")
    if isinstance(content, str) and content:
        return [content]
    if not isinstance(content, list):
        return []
    output: list[str] = []
    for item in content:
        if isinstance(item, str) and item:
            output.append(item)
        elif isinstance(item, Mapping) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str) and text:
                output.append(text)
    return output


def _thinking_deltas(delta: Mapping[str, Any]) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Extracts thinking fragments from "thinking" typed items in a delta's
    # content list; their thinking value may be a string or a list of parts with
    # text fields. Empty strings are skipped.
    content = delta.get("content")
    if not isinstance(content, list):
        return []
    output: list[str] = []
    for item in content:
        if not isinstance(item, Mapping) or item.get("type") != "thinking":
            continue
        thinking = item.get("thinking")
        if isinstance(thinking, str) and thinking:
            output.append(thinking)
        elif isinstance(thinking, list):
            for part in thinking:
                if isinstance(part, Mapping):
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        output.append(text)
    return output


def _tool_call_deltas(delta: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    # [FUNCTION DOC]
    # HELPER: Returns mapping entries of the delta's tool_calls (or toolCalls) list,
    # or an empty list.
    tool_calls = delta.get("tool_calls") or delta.get("toolCalls")
    if not isinstance(tool_calls, list):
        return []
    return [tool_call for tool_call in tool_calls if isinstance(tool_call, Mapping)]
