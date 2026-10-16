# [MODULE DOC]
# Google Generative Language (Gemini/Gemma) streaming provider for tau_ai.
# Converts Tau agent messages and tools into a streamGenerateContent request
# (contents, systemInstruction, generationConfig with token limit and thinking
# config, functionDeclarations), streams SSE chunks, parses them into
# provider-neutral events (text, thinking, tool calls, final message), retries on
# retryable HTTP statuses, transport errors and empty truncated streams, and
# canonicalizes the result into Pi-compatible assistant message events.
"""Google Generative AI provider."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from json import JSONDecodeError, loads

import httpx

from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    ImageContent,
    TextContent,
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


class GoogleGenerativeAIProvider:
    # [CLASS DOC]
    # Provider adapter. Holds the OpenAI-compatible-shaped config (base URL, API
    # key, headers, retry limits, timeouts, reasoning effort, max tokens, image
    # support) and an HTTP client, created lazily and owned when not injected.
    """Provider adapter for Google's Generative Language streaming API."""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Store the config and optional injected client; remember whether the provider
        # owns its client (true when none was injected). No I/O.
        self._config = config
        self._client = client
        self._owns_client = client is None

    async def aclose(self) -> None:
        # [FUNCTION DOC]
        # Close and drop the HTTP client only when the provider created it. Async I/O.
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
        # Public provider entry point. Ignores the session id. Builds the raw provider
        # event stream and wraps it with the canonicalizer labeled api
        # "google-generative-ai", provider "google" and the model. Returns an async
        # iterator of assistant message events; no work happens until iterated.
        """Stream one response as Pi-compatible assistant message events."""
        del session_id
        raw = self._stream_provider_events(
            model=model, system=system, messages=messages, tools=tools, signal=signal
        )
        return canonicalize_provider_stream(
            raw, api="google-generative-ai", provider="google", model=model
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
        # HELPER: return an async iterator of provider-neutral events for one request.
        """Stream one Gemini response as provider-neutral events."""

        async def iterator() -> AsyncIterator[ProviderEvent]:
            # [FUNCTION DOC]
            # HELPER: the request loop. Builds the payload and the URL
            # "<base>/models/<model>:streamGenerateContent?alt=sse&key=<api key>" with
            # configured headers plus JSON content type. Each attempt uses a fresh parser
            # and POSTs with streaming. HTTP 400+: if retryable (408, 409, 425, 429, 5xx and
            # attempts remain) yields a retry event, waits (stopping silently if cancelled)
            # and retries; otherwise yields an error event with a formatted message and the
            # status/body. On success yields a response-start event, then parses each SSE
            # data line, yielding parser events and stopping on a fatal parse error or when
            # the signal is cancelled. If the stream ended with no finish reason and no
            # content it retries; otherwise yields the parser's final events. Retryable
            # transport errors retry if no content was emitted, else yield an error event
            # with the attempt count. Side effects: network I/O, sleeping between retries.
            # KNOWN ISSUE: the API key is placed in the URL query string, where it can leak
            # into logs or exception messages; cancellation mid-stream returns silently
            # without a terminal event.
            client = self._get_client()
            payload = _build_google_payload(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                reasoning_effort=self._config.reasoning_effort,
                max_tokens=self._config.max_tokens,
                supports_images=self._config.supports_images,
            )
            url = (
                f"{self._config.base_url.rstrip('/')}/models/"
                f"{model}:streamGenerateContent?alt=sse&key={self._config.api_key}"
            )
            headers = {
                **dict(self._config.headers or {}),
                "content-type": "application/json",
            }

            attempt = 0
            parser = _GoogleStreamParser()
            while True:
                parser = _GoogleStreamParser()
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
                            for parser_event in parser.feed(event):
                                yield parser_event
                            if parser.fatal:
                                return
                        if (
                            not parser.has_finish_reason
                            and not parser.emitted_content
                            and self._should_retry(attempt)
                        ):
                            delay = retry_delay_seconds(
                                attempt,
                                max_delay_seconds=self._config.max_retry_delay_seconds,
                            )
                            yield provider_retry_event(
                                attempt=attempt,
                                max_retries=self._config.max_retries,
                                delay_seconds=delay,
                                reason="stream ended without finishReason",
                            )
                            attempt += 1
                            if not await wait_for_retry(delay, signal=signal):
                                return
                            continue
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
        # HELPER: return the HTTP client, lazily creating one with the configured
        # timeout (mutates provider state).
        if self._client is None:
            self._client = create_async_client(timeout=self._config.timeout_seconds)
        return self._client

    def _should_retry(self, attempt: int, *, status_code: int | None = None) -> bool:
        # [FUNCTION DOC]
        # HELPER: false once attempts reach max retries; otherwise true for transport
        # or stream retries (no status) and for statuses 408, 409, 425, 429 or 5xx.
        if attempt >= self._config.max_retries:
            return False
        return status_code is None or status_code in {408, 409, 425, 429} or status_code >= 500


class _GoogleStreamParser:
    # [CLASS DOC]
    # HELPER: stateful parser for one Gemini stream attempt. Accumulates text and
    # thinking parts and tool calls, tracks the last finish reason, and exposes
    # flags for whether any content was emitted and whether a fatal error occurred.
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # HELPER: initialize empty accumulators and false flags.
        self.emitted_content = False
        self.fatal = False
        self._content_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._tool_calls: list[ToolCall] = []
        self._finish_reason: str | None = None

    @property
    def has_finish_reason(self) -> bool:
        # [FUNCTION DOC]
        # HELPER property: true once any chunk carried a string finishReason.
        return self._finish_reason is not None

    def feed(self, event: str) -> list[ProviderEvent]:
        # [FUNCTION DOC]
        # HELPER: parse one SSE data payload. Invalid JSON or non-object sets fatal and
        # returns a single error event. Only the first candidate is used; records its
        # finishReason; for each part, non-empty text becomes a thinking delta when
        # marked as thought or a text delta otherwise, and a functionCall becomes a tool
        # call (id defaulting to "tool-call-N", name, args object, optional thought
        # signature) emitted as a tool call event. Returns the list of events; mutates
        # parser state.
        chunk = _loads_object(event)
        if chunk is None:
            self.fatal = True
            return [ProviderErrorEvent(message="Google returned an invalid JSON stream chunk")]
        events: list[ProviderEvent] = []
        candidates = chunk.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return []
        candidate = candidates[0]
        if not isinstance(candidate, Mapping):
            return []
        finish_reason = candidate.get("finishReason")
        if isinstance(finish_reason, str):
            self._finish_reason = finish_reason
        content = candidate.get("content")
        if not isinstance(content, Mapping):
            return events
        parts = content.get("parts")
        if not isinstance(parts, list):
            return events
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                self.emitted_content = True
                if part.get("thought") is True:
                    self._thinking_parts.append(text)
                    events.append(ProviderThinkingDeltaEvent(delta=text))
                else:
                    self._content_parts.append(text)
                    events.append(ProviderTextDeltaEvent(delta=text))
            function_call = part.get("functionCall")
            if isinstance(function_call, Mapping):
                self.emitted_content = True
                default_id = f"tool-call-{len(self._tool_calls)}"
                thought_signature = part.get("thoughtSignature")
                tool_call = ToolCall(
                    id=_string_or_default(function_call.get("id"), default_id),
                    name=_string_or_default(function_call.get("name"), ""),
                    arguments=_object_or_empty(function_call.get("args")),
                    thought_signature=thought_signature
                    if isinstance(thought_signature, str)
                    else None,
                )
                self._tool_calls.append(tool_call)
                events.append(ProviderToolCallEvent(tool_call=tool_call))
        return events

    def finalize(self) -> list[ProviderEvent]:
        # [FUNCTION DOC]
        # HELPER: if no finish reason was seen, return an error event. Otherwise build
        # the assistant content from joined text and tool calls, prepend a single
        # joined thinking block if any, and return one response-end event with the
        # normalized finish reason.
        if self._finish_reason is None:
            return [ProviderErrorEvent(message="Google stream ended without finishReason")]
        content = assistant_content("".join(self._content_parts), self._tool_calls)
        if self._thinking_parts:
            content.insert(0, ThinkingContent(thinking="".join(self._thinking_parts)))
        return [
            ProviderResponseEndEvent(
                message=AssistantMessage(content=content),
                finish_reason=_normalize_finish_reason(
                    self._finish_reason, has_tool_calls=bool(self._tool_calls)
                ),
            )
        ]


def _build_google_payload(
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
    # HELPER: build the request body: converted contents for all messages, a
    # systemInstruction when the system prompt is non-empty, generationConfig with
    # maxOutputTokens and thinkingConfig when applicable, and one tools entry with
    # function declarations when tools exist. Returns the JSON-ready mapping.
    config: dict[str, JSONValue] = {}
    payload: dict[str, JSONValue] = {
        "contents": [
            item
            for message in messages
            for item in _messages_to_google(message, model=model, supports_images=supports_images)
        ],
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    if max_tokens is not None:
        config["maxOutputTokens"] = max_tokens
    thinking_config = _google_thinking_config(model, reasoning_effort)
    if thinking_config is not None:
        config["thinkingConfig"] = thinking_config
    if config:
        payload["generationConfig"] = config
    if tools:
        payload["tools"] = [{"functionDeclarations": [_tool_to_google(tool) for tool in tools]}]
    return payload


def _google_thinking_config(
    model: str, reasoning_effort: str | None
) -> dict[str, JSONValue] | None:
    # [FUNCTION DOC]
    # HELPER: map a reasoning effort to Gemini thinking config. No effort gives
    # nothing. "none" disables thinking: LOW level for Gemini 3 Pro (cannot fully
    # disable), MINIMAL for Gemini 3 Flash and Gemma 4, else zero budget. Upper-case
    # level names pass through as thinkingLevel with thoughts included. Otherwise a
    # model-specific token budget is used when known, else a thinkingLevel.
    if reasoning_effort is None:
        return None
    if reasoning_effort == "none":
        if _is_gemini3_pro_model(model):
            return {"thinkingLevel": "LOW"}
        if _is_gemini3_flash_model(model) or _is_gemma4_model(model):
            return {"thinkingLevel": "MINIMAL"}
        return {"thinkingBudget": 0}
    if reasoning_effort in {"MINIMAL", "LOW", "MEDIUM", "HIGH"}:
        return {"includeThoughts": True, "thinkingLevel": reasoning_effort}
    budget = _google_budget(model, reasoning_effort)
    if budget is None:
        return {"includeThoughts": True, "thinkingLevel": _google_level(model, reasoning_effort)}
    return {"includeThoughts": True, "thinkingBudget": budget}


def _google_budget(model: str, effort: str) -> int | None:
    # [FUNCTION DOC]
    # HELPER: return a thinking token budget for Gemini 2.5 models by effort
    # (xhigh treated as high), nothing for unknown efforts or level-based models
    # (Gemini 3, Gemma 4), and -1 (dynamic budget) for other models.
    normalized = effort.lower()
    if normalized == "xhigh":
        normalized = "high"
    if normalized not in {"minimal", "low", "medium", "high"}:
        return None
    if _is_gemini3_pro_model(model) or _is_gemini3_flash_model(model) or _is_gemma4_model(model):
        return None
    if "2.5-pro" in model:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 32768}[normalized]
    if "2.5-flash-lite" in model:
        return {"minimal": 512, "low": 2048, "medium": 8192, "high": 24576}[normalized]
    if "2.5-flash" in model:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 24576}[normalized]
    return -1


def _google_level(model: str, effort: str) -> str:
    # [FUNCTION DOC]
    # HELPER: map effort (xhigh as high) to a thinkingLevel: Gemini 3 Pro supports
    # only LOW or HIGH, Gemma 4 only MINIMAL or HIGH; others map directly with HIGH
    # as the fallback.
    normalized = effort.lower()
    if normalized == "xhigh":
        normalized = "high"
    if _is_gemini3_pro_model(model):
        return "LOW" if normalized in {"minimal", "low"} else "HIGH"
    if _is_gemma4_model(model):
        return "MINIMAL" if normalized in {"minimal", "low"} else "HIGH"
    return {
        "minimal": "MINIMAL",
        "low": "LOW",
        "medium": "MEDIUM",
        "high": "HIGH",
    }.get(normalized, "HIGH")


def _is_gemini3_pro_model(model: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: case-insensitive check that the model name contains "gemini-3" and "pro".
    return "gemini-3" in model.lower() and "pro" in model.lower()


def _is_gemini3_flash_model(model: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: case-insensitive check for "gemini-3" and "flash" in the model name.
    return "gemini-3" in model.lower() and "flash" in model.lower()


def _is_gemma4_model(model: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: case-insensitive check for "gemma-4" or "gemma4" in the model name.
    return "gemma-4" in model.lower() or "gemma4" in model.lower()


def _messages_to_google(
    message: AgentMessage, *, model: str, supports_images: bool
) -> list[dict[str, JSONValue]]:
    # [FUNCTION DOC]
    # HELPER: convert one agent message into Google content entries. User messages
    # become a user entry with text and inline images (or placeholders when images
    # unsupported). Assistant messages become a model entry of text parts, thought
    # parts (with signature) and functionCall parts (portable id, name, args, thought
    # signature), using one empty text part when nothing remains. Tool results become
    # a user functionResponse with "output" or "error" text and portable id; images
    # are attached inside the response for models that support multimodal function
    # responses, otherwise sent as a separate user entry prefixed by a label. Other
    # message kinds are converted to a user message first. Returns a list of entries.
    if isinstance(message, UserMessage):
        text, images = text_and_images(
            message.content,
            supports_images=supports_images,
            image_placeholder=NON_VISION_USER_IMAGE_PLACEHOLDER,
        )
        user_parts: list[JSONValue] = []
        if text:
            user_parts.append({"text": text})
        user_parts.extend(_google_image(image) for image in images)
        return [{"role": "user", "parts": user_parts}]
    if isinstance(message, AssistantMessage):
        parts: list[JSONValue] = []
        for block in message.content:
            if isinstance(block, TextContent):
                parts.append({"text": block.text})
            elif isinstance(block, ThinkingContent):
                part: dict[str, JSONValue] = {"text": block.thinking, "thought": True}
                if block.thinking_signature is not None:
                    part["thoughtSignature"] = block.thinking_signature
                parts.append(part)
            elif isinstance(block, ToolCall):
                part = {
                    "functionCall": {
                        "id": portable_tool_call_id(block.id),
                        "name": block.name,
                        "args": dict(block.arguments),
                    }
                }
                if block.thought_signature is not None:
                    part["thoughtSignature"] = block.thought_signature
                parts.append(part)
        return [{"role": "model", "parts": parts or [{"text": ""}]}]
    if isinstance(message, ToolResultMessage):
        text, images = text_and_images(
            message.content,
            supports_images=supports_images,
            image_placeholder=NON_VISION_TOOL_IMAGE_PLACEHOLDER,
        )
        response: dict[str, JSONValue] = {
            "name": message.tool_name,
            "response": {"output" if not message.is_error else "error": text},
        }
        if message.tool_call_id:
            response["id"] = portable_tool_call_id(message.tool_call_id)
        image_parts: list[JSONValue] = [_google_image(image) for image in images]
        if image_parts and _supports_multimodal_function_response(model):
            response["parts"] = image_parts
        converted: list[dict[str, JSONValue]] = [
            {"role": "user", "parts": [{"functionResponse": response}]}
        ]
        if image_parts and not _supports_multimodal_function_response(model):
            separate_image_parts: list[JSONValue] = [
                {"text": "Tool result image:"},
                *image_parts,
            ]
            converted.append({"role": "user", "parts": separate_image_parts})
        return converted
    return _messages_to_google(
        message_to_user(message), model=model, supports_images=supports_images
    )


def _google_image(image: ImageContent) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: convert image content to an inlineData part with MIME type and base64 data.
    return {"inlineData": {"mimeType": image.mime_type, "data": image.data}}


def _supports_multimodal_function_response(model: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: true for non-Gemini models and Gemini 3+, false for older "gemini-" models.
    normalized = model.lower()
    return not normalized.startswith("gemini-") or normalized.startswith("gemini-3")


def _tool_to_google(tool: AgentTool) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: convert an agent tool to a function declaration with name, description
    # and a sanitized copy of its input schema.
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": _sanitize_google_schema(dict(tool.input_schema)),
    }


_UNSUPPORTED_GOOGLE_SCHEMA_KEYS = frozenset({"additionalProperties", "$schema"})


def _sanitize_google_schema(value: JSONValue) -> JSONValue:
    # [FUNCTION DOC]
    # HELPER: recursively copy a JSON value, removing "additionalProperties" and
    # "$schema" keys from every object, since Gemini rejects them.
    """Strip JSON Schema keywords Gemini's OpenAPI-subset parser rejects."""
    if isinstance(value, dict):
        return {
            key: _sanitize_google_schema(subvalue)
            for key, subvalue in value.items()
            if key not in _UNSUPPORTED_GOOGLE_SCHEMA_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_google_schema(item) for item in value]
    return value


def _parse_sse_line(line: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: return the stripped payload of an SSE "data:" line, or nothing for
    # blank or non-data lines.
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    return line.removeprefix("data:").strip()


def _loads_object(value: str) -> dict[str, JSONValue] | None:
    # [FUNCTION DOC]
    # HELPER: parse JSON text and return it only if it is an object; nothing on
    # decode errors or other JSON types.
    try:
        loaded = loads(value)
    except JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _string_or_default(value: object, default: str) -> str:
    # [FUNCTION DOC]
    # HELPER: return the value if it is a non-empty string, else the default.
    return value if isinstance(value, str) and value else default


def _object_or_empty(value: object) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: return the value if it is a mapping object, else an empty mapping.
    return value if isinstance(value, dict) else {}


def _normalize_finish_reason(reason: str | None, *, has_tool_calls: bool) -> str:
    # [FUNCTION DOC]
    # HELPER: "tool_calls" whenever tool calls exist; "length" for MAX_TOKENS,
    # MODEL_ARMOR or RECITATION; otherwise "stop".
    # KNOWN ISSUE: MODEL_ARMOR and RECITATION are content-policy stops, not length
    # limits, and SAFETY/blocked reasons are reported as a normal "stop".
    if has_tool_calls:
        return "tool_calls"
    if reason in {"MAX_TOKENS", "MODEL_ARMOR", "RECITATION"}:
        return "length"
    return "stop"
