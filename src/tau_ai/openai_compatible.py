# [MODULE DOC]
# OpenAI-compatible provider for the tau_ai layer. Serves most models over
# "/chat/completions" and routes models that need it ("codex" models and the
# gpt-5.4 / gpt-5.5 families, when the config infers the API from the model, or
# always when the config's API is "openai-responses") to "/responses".
# Both paths share one streaming POST envelope (auth, optional credential
# resolver, session-affinity headers, retries on transient HTTP statuses and on
# transport errors before content is emitted, cancellation) and differ in
# payload builders and SSE stream parsers. Handles many vendor dialects via a
# compat mapping and a thinking format (openai, zai, qwen, qwen-chat-template,
# deepseek, openrouter, together). Output is canonicalized to Pi-compatible
# assistant message events.
"""OpenAI-compatible chat completions provider.

Most OpenAI-compatible models are served over `/chat/completions`. Newer
reasoning models (e.g. ``gpt-5.5``/``gpt-5.4`` and the ``*-codex`` family)
reject the combination of function tools and ``reasoning_effort`` on that
endpoint and require ``/v1/responses`` instead. This adapter routes those
models to the Responses API at request time while leaving every other model on
the original chat-completions path unchanged.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import suppress
from json import JSONDecodeError, dumps, loads
from typing import Any, Protocol

import httpx

from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    AssistantMessageDiagnostic,
    ImageContent,
    ThinkingContent,
    ToolResultMessage,
    Usage,
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
    messages_have_images,
    text_and_images,
)
from tau_ai.env import OpenAICompatibleConfig
from tau_ai.events import AssistantMessageEvent
from tau_ai.http import create_async_client
from tau_ai.http_errors import provider_http_error_message
from tau_ai.openai_cache import is_direct_openai_url, openai_prompt_cache_key
from tau_ai.provider import CancellationToken
from tau_ai.retry import (
    RETRYABLE_TRANSPORT_ERRORS,
    provider_retry_event,
    retry_delay_seconds,
    wait_for_retry,
)
from tau_ai.stream import canonicalize_provider_stream
from tau_ai.tool_call_ids import portable_tool_call_id

# Models that reject function tools + reasoning_effort on /chat/completions and
# must use the /v1/responses endpoint instead.
_RESPONSES_ONLY_PREFIXES: tuple[str, ...] = ("gpt-5.5", "gpt-5.4")


def _use_responses_api(model: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: Return True when the trimmed, lowercased model ID contains "codex" or
    # starts with "gpt-5.5" or "gpt-5.4". Pure.
    """Return whether ``model`` must be served over the Responses API."""
    normalized = model.strip().lower()
    if "codex" in normalized:
        return True
    return any(normalized.startswith(prefix) for prefix in _RESPONSES_ONLY_PREFIXES)


class OpenAICompatibleProvider:
    # [CLASS DOC]
    # Provider adapter for OpenAI-compatible APIs holding the config and an
    # optional HTTP client (created lazily and owned when not supplied).
    """Provider adapter for OpenAI-compatible `/chat/completions` APIs.

    Models that require it are transparently served over `/v1/responses`.
    """

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Inputs: an OpenAICompatibleConfig and an optional shared HTTP client. Records
        # ownership of the client. No I/O.
        self._config = config
        self._client = client
        self._owns_client = client is None

    async def aclose(self) -> None:
        # [FUNCTION DOC]
        # Close the HTTP client only if this provider created it, then forget it.
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
        # Public streaming entry point. Inputs: model, system prompt, messages, tools,
        # optional cancellation token and session ID (used for prompt-cache key and
        # session affinity). Output: an async iterator of canonical events labelled with
        # the config's API and provider name (default "openai-compatible"); text and
        # thinking are treated as independent channels except on the Responses path.
        # Lazy; no request until iterated.
        """Stream one response as Pi-compatible assistant message events."""
        raw = self._stream_provider_events(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            signal=signal,
            session_id=session_id,
        )
        return canonicalize_provider_stream(
            raw,
            api=self._config.api,
            provider=getattr(self._config, "provider_name", "openai-compatible"),
            model=model,
            independent_channels=not (
                self._config.api == "openai-responses"
                or (self._config.infer_api_from_model and _use_responses_api(model))
            ),
        )

    def _stream_provider_events(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        # [FUNCTION DOC]
        # HELPER: Choose the Responses or chat-completions path (see module doc) and
        # return its provider-event iterator.
        """Stream one model response as provider-neutral events."""
        if self._config.api == "openai-responses" or (
            self._config.infer_api_from_model and _use_responses_api(model)
        ):
            return self._stream_responses(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                signal=signal,
                session_id=session_id,
            )
        return self._stream_chat_completions(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            signal=signal,
            session_id=session_id,
        )

    def _stream_chat_completions(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        # [FUNCTION DOC]
        # HELPER: Derive the session affinity ID and optional prompt cache key, build
        # the chat payload (applying model aliases and config options) and return the
        # shared stream for "<base>/chat/completions" with the chat parser.
        """Stream one chat completion response as provider-neutral events."""
        affinity_id = openai_prompt_cache_key(session_id)
        cache_key = self._prompt_cache_key(affinity_id)
        payload = _build_chat_payload(
            model=self._config.model_aliases.get(model, model),
            system=system,
            messages=messages,
            tools=tools,
            reasoning_effort=self._config.reasoning_effort,
            reasoning_effort_parameter=self._config.reasoning_effort_parameter,
            thinking_format=self._config.thinking_format,
            compat=self._config.compat,
            max_tokens=self._config.max_tokens,
            include_reasoning_effort_none=self._config.include_reasoning_effort_none,
            supports_images=self._config.supports_images,
            prompt_cache_key=cache_key,
        )
        return self._stream(
            model=model,
            url=f"{self._config.base_url.rstrip('/')}/chat/completions",
            payload=payload,
            parser_factory=_ChatStreamParser,
            session_id=affinity_id,
            session_affinity_format=self._session_affinity_format(responses=False),
            has_images=(self._config.supports_images and messages_have_images(messages)),
            signal=signal,
        )

    def _stream_responses(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        # [FUNCTION DOC]
        # HELPER: Same as the chat variant but builds a Responses payload and streams
        # "<base>/responses" with the Responses parser.
        """Stream one `/v1/responses` response as provider-neutral events."""
        affinity_id = openai_prompt_cache_key(session_id)
        cache_key = self._prompt_cache_key(affinity_id)
        payload = _build_responses_payload(
            model=self._config.model_aliases.get(model, model),
            system=system,
            messages=messages,
            tools=tools,
            reasoning_effort=self._config.reasoning_effort,
            max_tokens=self._config.max_tokens,
            supports_images=self._config.supports_images,
            prompt_cache_key=cache_key,
        )
        return self._stream(
            model=model,
            url=f"{self._config.base_url.rstrip('/')}/responses",
            payload=payload,
            parser_factory=_ResponsesStreamParser,
            session_id=affinity_id,
            session_affinity_format=self._session_affinity_format(responses=True),
            has_images=(self._config.supports_images and messages_have_images(messages)),
            signal=signal,
        )

    def _stream(
        self,
        *,
        model: str,
        url: str,
        payload: Mapping[str, JSONValue],
        parser_factory: Callable[[], _StreamParser],
        session_id: str | None = None,
        session_affinity_format: str | None = None,
        has_images: bool = False,
        signal: CancellationToken | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        # [FUNCTION DOC]
        # HELPER: Return the shared streaming envelope iterator for one endpoint; see
        # the nested iterator.
        """Run the shared streaming POST + retry envelope for a given endpoint.

        The per-endpoint differences (SSE chunk handling and final-message
        assembly) live in the ``_StreamParser`` produced by ``parser_factory``;
        everything else — HTTP, status/network retries, cancellation, the
        opening ``response_start`` event — is identical across endpoints.
        """

        async def iterator() -> AsyncIterator[ProviderEvent]:
            # [FUNCTION DOC]
            # HELPER: Execute the request with retries and yield provider events.
            # Setup: copies configured headers; adds the Copilot vision header for
            # github-copilot requests with images; applies a credential resolver (API key,
            # extra headers, and an optional base URL that replaces the request URL while
            # keeping the endpoint); adds a Bearer Authorization header unless omitted by
            # config or already present (case-insensitive); adds session affinity headers.
            # Loop: a fresh parser per attempt; reads an optional response-provider header.
            # HTTP 400 or above: retry with a retry event and backoff for transient
            # statuses (408, 409, 425, 429, 5xx) while attempts remain (stop silently if
            # cancelled during the wait), else yield an error event (formatted message,
            # status, body, attempts, response provider) and stop. Success: yield a start
            # event, feed each SSE data payload to the parser yielding its events, stop
            # early when the parser says so, stop silently on cancellation; if the parser
            # emitted a fatal error, end; otherwise get the final events, call the optional
            # response-headers observer (its exceptions become a diagnostic on the final
            # assistant message, never failures) and yield them. Transport errors are
            # retried when no content was emitted and attempts remain, else yield an error.
            # Side effects: network I/O, sleeps.
            # KNOWN ISSUE: a transport-error retry after the start event was yielded yields
            # another start event on the next attempt; cancellation mid-stream ends
            # without any end or error event.
            client = self._get_client()
            api_key = self._config.api_key
            request_url = url
            headers = dict(self._config.headers or {})
            if self._config.provider_name == "github-copilot" and has_images:
                headers["Copilot-Vision-Request"] = "true"
            if self._config.credential_resolver is not None:
                auth = await self._config.credential_resolver()
                api_key = auth.api_key
                headers.update(auth.headers or {})
                if auth.base_url is not None:
                    endpoint = (
                        "/responses"
                        if url.rstrip("/").endswith("/responses")
                        else "/chat/completions"
                    )
                    request_url = f"{auth.base_url.rstrip('/')}{endpoint}"
            if not self._config.omit_authorization_header:
                has_authorization = any(key.casefold() == "authorization" for key in headers)
                if not has_authorization:
                    headers["Authorization"] = f"Bearer {api_key}"
            _apply_session_affinity_headers(headers, session_id, session_affinity_format)

            attempt = 0
            while True:
                parser = parser_factory()
                try:
                    async with client.stream(
                        "POST", request_url, json=payload, headers=headers
                    ) as response:
                        response_provider = _response_header_value(
                            response,
                            self._config.response_provider_header,
                        )
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
                                    data={
                                        "status_code": response.status_code,
                                        "body": body_text,
                                    },
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
                                data={
                                    "status_code": response.status_code,
                                    "body": body_text,
                                    "attempts": attempt + 1,
                                },
                                response_provider=response_provider,
                            )
                            return

                        yield ProviderResponseStartEvent(
                            model=model,
                            response_provider=response_provider,
                        )

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

                        if parser.fatal:
                            return
                        final_events = parser.finalize()
                        observer = self._config.response_headers_observer
                        if observer is not None:
                            try:
                                observer(dict(response.headers))
                            except Exception as exc:
                                # Observer reporting is also best-effort; response
                                # completion must never depend on metadata hooks.
                                with suppress(Exception):
                                    _append_response_observer_diagnostic(final_events, exc)
                        for parser_event in final_events:
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
                            data={
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                            },
                        )
                        attempt += 1
                        if not await wait_for_retry(delay, signal=signal):
                            return
                        continue
                    yield ProviderErrorEvent(
                        message=str(exc),
                        data={"attempts": attempt + 1},
                    )
                    return

        return iterator()

    def _prompt_cache_key(self, affinity_id: str | None) -> str | None:
        # [FUNCTION DOC]
        # HELPER: Return the affinity ID as prompt cache key when the compat flag
        # supportsPromptCacheKey is true, or is unset and the base URL is OpenAI's own
        # API; otherwise None.
        supports = self._config.compat.get("supportsPromptCacheKey")
        if supports is not True and not (
            supports is None and is_direct_openai_url(self._config.base_url)
        ):
            return None
        return affinity_id

    def _session_affinity_format(self, *, responses: bool) -> str | None:
        # [FUNCTION DOC]
        # HELPER: Return the session affinity header format (compat
        # sessionAffinityFormat, default "openai") when sendSessionAffinityHeaders is
        # true, or unset for a Responses request to OpenAI's own API; otherwise None.
        sends_headers = self._config.compat.get("sendSessionAffinityHeaders")
        if sends_headers is not True and not (
            sends_headers is None and responses and is_direct_openai_url(self._config.base_url)
        ):
            return None
        value = self._config.compat.get("sessionAffinityFormat")
        return value if isinstance(value, str) else "openai"

    def _get_client(self) -> httpx.AsyncClient:
        # [FUNCTION DOC]
        # HELPER: Return the HTTP client, creating one with the configured timeout on
        # first use.
        if self._client is None:
            self._client = create_async_client(timeout=self._config.timeout_seconds)
        return self._client

    def _should_retry(self, attempt: int, *, status_code: int | None = None) -> bool:
        # [FUNCTION DOC]
        # HELPER: True if attempts remain and either no status was given or the status
        # is transient.
        if attempt >= self._config.max_retries:
            return False
        return status_code is None or _is_transient_status(status_code)


def _response_header_value(response: httpx.Response, header_name: str | None) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Return the trimmed value of the configured response header, or None if
    # no header is configured, it is absent, or it is blank. Pure.
    """Return one normalized response metadata header when configured."""
    if header_name is None:
        return None
    value = response.headers.get(header_name)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _apply_session_affinity_headers(
    headers: dict[str, str],
    session_id: str | None,
    affinity_format: str | None,
) -> None:
    # [FUNCTION DOC]
    # HELPER: When both a session ID and format are given, set "x-session-id"
    # (format "openrouter") or "session_id" (format "openai"); other formats add
    # nothing. Mutates the header mapping.
    if session_id is None or affinity_format is None:
        return
    if affinity_format == "openrouter":
        headers["x-session-id"] = session_id
        return
    if affinity_format == "openai":
        headers["session_id"] = session_id


def _append_response_observer_diagnostic(
    events: list[ProviderEvent],
    exc: Exception,
) -> None:
    # [FUNCTION DOC]
    # HELPER: Add a "response_headers_observer_error" diagnostic (error text and
    # type) to the message of the first response-end event in the list. Mutates
    # that message.
    for event in events:
        if isinstance(event, ProviderResponseEndEvent):
            diagnostic = AssistantMessageDiagnostic(
                type="response_headers_observer_error",
                details={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            event.message.diagnostics = [
                *(event.message.diagnostics or []),
                diagnostic,
            ]
            return


class _StreamParser(Protocol):
    # [CLASS DOC]
    # HELPER: Structural interface for per-endpoint SSE parsers: an emitted_content
    # flag (any model output seen, making mid-stream drops non-retryable), a fatal
    # flag (a terminal error was emitted; finalize must not be called), feed and
    # finalize.
    """Per-endpoint SSE handler driven by the shared streaming envelope."""

    # True once any model output (text/thinking/tool args) has been emitted;
    # the envelope uses it to decide whether a mid-stream drop is retryable.
    emitted_content: bool
    # True when the parser already emitted a terminal error event and the
    # envelope must not call finalize().
    fatal: bool

    def feed(self, event: str) -> tuple[list[ProviderEvent], bool]:
        # [FUNCTION DOC]
        # HELPER: Consume one SSE data payload and return (events to yield, whether to
        # stop reading).
        """Consume one SSE ``data:`` payload, returning (events, should_stop)."""
        ...

    def finalize(self) -> list[ProviderEvent]:
        # [FUNCTION DOC]
        # HELPER: Return the trailing tool-call events and the response-end event.
        """Return the trailing tool-call and response-end events."""
        ...


class _ChatStreamParser:
    # [CLASS DOC]
    # HELPER: Stateful parser for chat-completions chunks accumulating text,
    # thinking (with the field name used as signature), tool-call builders by
    # index, finish reason and usage.
    """Parser for OpenAI `/chat/completions` SSE chunks."""

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Initialize empty accumulators and false flags.
        self.emitted_content = False
        self.fatal = False
        self._content_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._thinking_signature: str | None = None
        self._tool_call_builders: dict[int, _ToolCallBuilder] = {}
        self._finish_reason: str | None = None
        self._usage: Usage | None = None

    def feed(self, event: str) -> tuple[list[ProviderEvent], bool]:
        # [FUNCTION DOC]
        # HELPER: "[DONE]" stops. Invalid JSON sets fatal and returns an error event and
        # stop. Top-level usage replaces usage; otherwise usage on the first choice is
        # used. Records the latest non-empty finish reason. From the delta: non-empty
        # content yields a text delta; the first non-empty of reasoning_content,
        # reasoning or thinking yields a thinking delta (the first field name seen is
        # kept as signature); tool-call deltas feed builders. Any of these set
        # emitted_content. Returns events and whether to stop.
        # KNOWN ISSUE: an empty "data:" payload (which _parse_sse_line returns as an
        # empty string, not None) is treated as invalid JSON and aborts the stream,
        # unlike the Anthropic parser which skips it.
        if event == "[DONE]":
            return [], True

        chunk = _loads_object(event)
        if chunk is None:
            self.fatal = True
            return [ProviderErrorEvent(message="Provider returned invalid JSON chunk")], True

        # The final usage chunk (from stream_options) carries usage at the top
        # level and often has empty choices.
        chunk_usage = chunk.get("usage")
        if isinstance(chunk_usage, Mapping):
            self._usage = _parse_chunk_usage(chunk_usage)

        choice = _first_choice(chunk)
        if choice is None:
            return [], False

        # Fallback: some providers (e.g. Moonshot) attach usage to the choice
        # instead of the chunk. Matches Pi's per-chunk `!chunk.usage` guard: the
        # fallback applies whenever this chunk lacks top-level usage.
        choice_usage = choice.get("usage")
        if not isinstance(chunk_usage, Mapping) and isinstance(choice_usage, Mapping):
            self._usage = _parse_chunk_usage(choice_usage)

        self._finish_reason = choice.get("finish_reason") or self._finish_reason
        delta = choice.get("delta")
        if not isinstance(delta, Mapping):
            return [], False

        events: list[ProviderEvent] = []
        content = delta.get("content")
        if isinstance(content, str) and content:
            self.emitted_content = True
            self._content_parts.append(content)
            events.append(ProviderTextDeltaEvent(delta=content))

        thinking = _thinking_delta(delta)
        if thinking is not None:
            field_name, text = thinking
            self.emitted_content = True
            self._thinking_parts.append(text)
            self._thinking_signature = self._thinking_signature or field_name
            events.append(ProviderThinkingDeltaEvent(delta=text))

        for tool_call_delta in _tool_call_deltas(delta):
            self.emitted_content = True
            index = int(tool_call_delta.get("index", 0))
            builder = self._tool_call_builders.setdefault(index, _ToolCallBuilder())
            builder.add_delta(tool_call_delta)

        return events, False

    def finalize(self) -> list[ProviderEvent]:
        # [FUNCTION DOC]
        # HELPER: Build tool calls in index order, yield one event per call, then a
        # response-end event with the assistant message (thinking first, then text and
        # tool calls), usage (empty if none) and finish reason.
        tool_calls = [
            builder.build(index) for index, builder in sorted(self._tool_call_builders.items())
        ]
        events: list[ProviderEvent] = [
            ProviderToolCallEvent(tool_call=tool_call) for tool_call in tool_calls
        ]
        content = assistant_content("".join(self._content_parts), tool_calls)
        if self._thinking_parts:
            content.insert(
                0,
                ThinkingContent(
                    thinking="".join(self._thinking_parts),
                    thinking_signature=self._thinking_signature,
                ),
            )
        events.append(
            ProviderResponseEndEvent(
                message=AssistantMessage(
                    content=content,
                    usage=self._usage or Usage(),
                ),
                finish_reason=self._finish_reason,
            )
        )
        return events


class _ResponsesStreamParser:
    # [CLASS DOC]
    # HELPER: Stateful parser for Responses API events accumulating text, thinking,
    # reasoning items by ID, function-call builders by item ID, terminal status and
    # usage.
    """Parser for OpenAI `/v1/responses` SSE events."""

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Initialize empty accumulators and false flags.
        self.emitted_content = False
        self.fatal = False
        self._content_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._reasoning_items: dict[str, dict[str, JSONValue]] = {}
        self._tool_call_builders: dict[str, _ResponsesToolCallBuilder] = {}
        self._status: str | None = None
        self._usage: Usage | None = None

    def feed(self, event: str) -> tuple[list[ProviderEvent], bool]:
        # [FUNCTION DOC]
        # HELPER: Ignores "[DONE]", invalid JSON and untyped events. Output text and
        # refusal deltas yield text deltas; reasoning summary/text deltas yield
        # thinking deltas; a finished summary part adds a blank-line separator if
        # thinking exists; output item added/done register reasoning items and
        # function-call builders (ID, call ID, name, arguments, output index);
        # function-call argument deltas accumulate (marking content emitted) and the
        # done event sets final arguments; completed/incomplete records status and
        # usage and stops; failed and error events set fatal, yield an error and stop.
        # The Responses API has no [DONE] sentinel; it ends with a terminal
        # event (completed/incomplete/failed) handled below.
        if event == "[DONE]":
            return [], False

        chunk = _loads_object(event)
        if chunk is None:
            return [], False

        chunk_type = chunk.get("type")
        if not isinstance(chunk_type, str):
            return [], False

        if chunk_type in ("response.output_text.delta", "response.refusal.delta"):
            delta = chunk.get("delta")
            if isinstance(delta, str) and delta:
                self.emitted_content = True
                self._content_parts.append(delta)
                return [ProviderTextDeltaEvent(delta=delta)], False

        elif chunk_type in (
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        ):
            delta = chunk.get("delta")
            if isinstance(delta, str) and delta:
                self.emitted_content = True
                self._thinking_parts.append(delta)
                return [ProviderThinkingDeltaEvent(delta=delta)], False

        elif chunk_type == "response.reasoning_summary_part.done":
            if self._thinking_parts:
                separator = "\n\n"
                self._thinking_parts.append(separator)
                return [ProviderThinkingDeltaEvent(delta=separator)], False

        elif chunk_type == "response.output_item.added":
            item = chunk.get("item")
            _register_reasoning_item(self._reasoning_items, item)
            _register_responses_item(
                self._tool_call_builders,
                item,
                output_index=chunk.get("output_index"),
            )

        elif chunk_type == "response.function_call_arguments.delta":
            item_id = chunk.get("item_id")
            if isinstance(item_id, str):
                builder = self._tool_call_builders.setdefault(item_id, _ResponsesToolCallBuilder())
                builder.add_arguments_delta(chunk.get("delta"))
                self.emitted_content = True

        elif chunk_type == "response.function_call_arguments.done":
            item_id = chunk.get("item_id")
            if isinstance(item_id, str):
                builder = self._tool_call_builders.setdefault(item_id, _ResponsesToolCallBuilder())
                builder.set_final(arguments=chunk.get("arguments"))

        elif chunk_type == "response.output_item.done":
            item = chunk.get("item")
            _register_reasoning_item(self._reasoning_items, item)
            _finalize_responses_item(
                self._tool_call_builders,
                item,
                output_index=chunk.get("output_index"),
            )

        elif chunk_type in ("response.completed", "response.incomplete"):
            self._status = _responses_finish_reason(chunk)
            self._usage = _usage_from_responses_event(chunk) or self._usage
            return [], True

        elif chunk_type == "response.failed":
            self.fatal = True
            return [_responses_failure_event(chunk)], True

        elif chunk_type == "error":
            self.fatal = True
            return [
                ProviderErrorEvent(message=_responses_error_message(chunk), data={"event": chunk})
            ], True

        return [], False

    def finalize(self) -> list[ProviderEvent]:
        # [FUNCTION DOC]
        # HELPER: Build tool calls ordered by output index, yield their events, then a
        # response-end event whose finish reason is "tool_calls" when tools were
        # called, "length" for incomplete, else "stop". Thinking content uses the first
        # registered reasoning item, serialized as JSON, as its signature (for replay).
        # KNOWN ISSUE: only the first reasoning item is preserved, so responses with
        # multiple reasoning items lose the others on replay.
        tool_calls = [
            builder.build(index)
            for index, builder in enumerate(_ordered_builders(self._tool_call_builders))
        ]
        events: list[ProviderEvent] = [
            ProviderToolCallEvent(tool_call=tool_call) for tool_call in tool_calls
        ]
        finish_reason = _normalize_finish_reason(self._status, has_tool_calls=bool(tool_calls))
        content = assistant_content("".join(self._content_parts), tool_calls)
        if self._thinking_parts:
            content.insert(
                0,
                ThinkingContent(
                    thinking="".join(self._thinking_parts),
                    thinking_signature=(
                        dumps(next(iter(self._reasoning_items.values())))
                        if self._reasoning_items
                        else None
                    ),
                ),
            )
        events.append(
            ProviderResponseEndEvent(
                message=AssistantMessage(
                    content=content,
                    usage=self._usage or Usage(),
                ),
                finish_reason=finish_reason,
            )
        )
        return events


class _ToolCallBuilder:
    # [CLASS DOC]
    # HELPER: Accumulator for a streamed chat-completions tool call (ID, name,
    # argument fragments).
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Start with empty ID, name and fragments.
        self.id = ""
        self.name = ""
        self.arguments_parts: list[str] = []

    def add_delta(self, delta: Mapping[str, Any]) -> None:
        # [FUNCTION DOC]
        # HELPER: Update ID and function name when present as strings and append any
        # string argument fragment. Mutates the builder.
        call_id = delta.get("id")
        if isinstance(call_id, str):
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

    def build(self, index: int) -> ToolCall:
        # [FUNCTION DOC]
        # HELPER: Parse joined arguments as a JSON object (empty gives an empty
        # mapping, invalid is kept under "_raw_arguments") and return a tool call with
        # the ID or "tool-call-<index>".
        arguments_text = "".join(self.arguments_parts)
        arguments = _loads_object(arguments_text) if arguments_text else {}
        if arguments is None:
            arguments = {"_raw_arguments": arguments_text}

        return ToolCall(
            id=self.id or f"tool-call-{index}",
            name=self.name,
            arguments=arguments,
        )


class _ResponsesToolCallBuilder:
    # [CLASS DOC]
    # HELPER: Accumulator for a streamed Responses function_call item: call ID,
    # name, output index, argument fragments and final arguments text.
    """Accumulates a streamed Responses-API ``function_call`` output item."""

    def __init__(
        self,
        *,
        call_id: str = "",
        name: str = "",
        output_index: int = 0,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Store optional initial call ID, name and output index; empty
        # fragments and no final arguments.
        self.call_id = call_id
        self.name = name
        self.output_index = output_index
        self.arguments_parts: list[str] = []
        self.arguments_final: str | None = None

    def add_arguments_delta(self, delta: object) -> None:
        # [FUNCTION DOC]
        # HELPER: Append the delta if it is a string.
        if isinstance(delta, str):
            self.arguments_parts.append(delta)

    def set_final(
        self,
        *,
        call_id: str | None = None,
        name: str | None = None,
        arguments: object = None,
        output_index: int | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Overwrite call ID and name when non-empty, final arguments when a
        # string, and output index when given. Mutates the builder.
        if call_id:
            self.call_id = call_id
        if name:
            self.name = name
        if isinstance(arguments, str):
            self.arguments_final = arguments
        if output_index is not None:
            self.output_index = output_index

    def build(self, index: int) -> ToolCall:
        # [FUNCTION DOC]
        # HELPER: Prefer the final arguments text over joined fragments, parse as for
        # chat tool calls, and return a tool call with the call ID or
        # "tool-call-<index>".
        arguments_text = (
            self.arguments_final
            if self.arguments_final is not None
            else "".join(self.arguments_parts)
        )
        arguments = _loads_object(arguments_text) if arguments_text else {}
        if arguments is None:
            arguments = {"_raw_arguments": arguments_text}

        return ToolCall(
            id=self.call_id or f"tool-call-{index}",
            name=self.name,
            arguments=arguments,
        )


def _build_chat_payload(
    *,
    model: str,
    system: str,
    messages: list[AgentMessage],
    tools: list[AgentTool],
    reasoning_effort: str | None = None,
    reasoning_effort_parameter: str = "reasoning_effort",
    thinking_format: str = "openai",
    compat: Mapping[str, JSONValue] | None = None,
    max_tokens: int | None = None,
    include_reasoning_effort_none: bool = False,
    supports_images: bool = False,
    prompt_cache_key: str | None = None,
) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Build the chat-completions request body: model, stream, system
    # message followed by converted messages; optional prompt_cache_key;
    # stream_options include_usage unless compat disables usage streaming; store
    # false unless compat disables it; max tokens under "max_tokens" or (default)
    # "max_completion_tokens" per compat; an OpenRouter provider routing object if
    # given; reasoning fields per thinking format (effort suppressed when compat
    # disables reasoning effort, except for zai); tools and, if compat zaiToolStream
    # is true, tool_stream. Pure.
    resolved_compat = dict(compat or {})
    supports_store = bool(resolved_compat.get("supportsStore", True))
    supports_usage = bool(resolved_compat.get("supportsUsageInStreaming", True))
    supports_reasoning_effort = bool(resolved_compat.get("supportsReasoningEffort", True))
    max_tokens_field = _string_compat(
        resolved_compat.get("maxTokensField"), default="max_completion_tokens"
    )
    payload: dict[str, JSONValue] = {
        "model": model,
        "stream": True,
        "messages": [
            _system_message(system),
            *_messages_to_openai_chat(messages, supports_images=supports_images),
        ],
    }
    if prompt_cache_key is not None:
        payload["prompt_cache_key"] = prompt_cache_key
    if supports_usage:
        payload["stream_options"] = {"include_usage": True}
    if supports_store:
        payload["store"] = False
    if max_tokens is not None:
        payload["max_tokens" if max_tokens_field == "max_tokens" else "max_completion_tokens"] = (
            max_tokens
        )
    openrouter_provider = resolved_compat.get("openrouterProvider")
    if isinstance(openrouter_provider, dict):
        payload["provider"] = openrouter_provider
    _apply_chat_reasoning(
        payload,
        reasoning_effort=(
            reasoning_effort if supports_reasoning_effort or thinking_format == "zai" else None
        ),
        reasoning_effort_parameter=reasoning_effort_parameter,
        thinking_format=thinking_format,
        include_reasoning_effort_none=include_reasoning_effort_none,
        supports_reasoning_effort=supports_reasoning_effort,
    )
    if tools:
        payload["tools"] = [_tool_to_openai(tool) for tool in tools]
        if resolved_compat.get("zaiToolStream") is True:
            payload["tool_stream"] = True
    return payload


def _apply_chat_reasoning(
    payload: dict[str, JSONValue],
    *,
    reasoning_effort: str | None,
    reasoning_effort_parameter: str,
    thinking_format: str,
    include_reasoning_effort_none: bool,
    supports_reasoning_effort: bool = True,
) -> None:
    # [FUNCTION DOC]
    # HELPER: Write reasoning controls into the payload (in place); reasoning is
    # enabled when an effort is given and is not "none". zai: thinking enabled or
    # disabled plus reasoning_effort when supported and enabled. qwen:
    # enable_thinking flag. qwen-chat-template: chat_template_kwargs with
    # enable_thinking and preserve_thinking. deepseek: thinking type plus effort
    # when enabled. openrouter or the "reasoning.effort" parameter style: a
    # reasoning object with the effort, or effort "none" when requested. together:
    # reasoning enabled flag plus effort when enabled. Default: reasoning_effort
    # when enabled, or "none" when include-none is set.
    reasoning_enabled = reasoning_effort is not None and reasoning_effort != "none"
    if thinking_format == "zai":
        # Z.AI's OpenAI-compatible API uses the provider-specific ``thinking``
        # object for every GLM model.  Only GLM-5.2+ accepts the separate
        # reasoning_effort field, so keep that decision model-specific via
        # supportsReasoningEffort instead of dropping the logical toggle.
        payload["thinking"] = {"type": "enabled" if reasoning_enabled else "disabled"}
        if supports_reasoning_effort and reasoning_enabled:
            payload["reasoning_effort"] = reasoning_effort
        return
    if thinking_format == "qwen":
        payload["enable_thinking"] = reasoning_enabled
        return
    if thinking_format == "qwen-chat-template":
        payload["chat_template_kwargs"] = {
            "enable_thinking": reasoning_enabled,
            "preserve_thinking": True,
        }
        return
    if thinking_format == "deepseek":
        payload["thinking"] = {"type": "enabled" if reasoning_enabled else "disabled"}
        if reasoning_enabled:
            payload["reasoning_effort"] = reasoning_effort
        return
    if thinking_format == "openrouter" or reasoning_effort_parameter == "reasoning.effort":
        if reasoning_enabled:
            payload["reasoning"] = {"effort": reasoning_effort}
        elif include_reasoning_effort_none:
            payload["reasoning"] = {"effort": "none"}
        return
    if thinking_format == "together":
        payload["reasoning"] = {"enabled": reasoning_enabled}
        if reasoning_enabled:
            payload["reasoning_effort"] = reasoning_effort
        return
    if reasoning_enabled or include_reasoning_effort_none:
        payload["reasoning_effort"] = reasoning_effort or "none"


def _string_compat(value: object, *, default: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Return the value if it is a non-empty string, else the default.
    return value if isinstance(value, str) and value else default


def _build_responses_payload(
    *,
    model: str,
    system: str,
    messages: list[AgentMessage],
    tools: list[AgentTool],
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
    supports_images: bool = False,
    prompt_cache_key: str | None = None,
) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Build the Responses request body: model, stream, store false,
    # instructions (system prompt), converted input items, optional
    # prompt_cache_key, max_output_tokens, a reasoning object with normalized
    # effort and automatic summaries when an effort is set, and function tools.
    # Pure.
    payload: dict[str, JSONValue] = {
        "model": model,
        "stream": True,
        # Stay stateless: the full transcript is resent every turn, so there is
        # no need for server-side retention. ``store: false`` also keeps the
        # path usable for zero-data-retention orgs, which reject ``store: true``.
        "store": False,
        "instructions": system,
        "input": _messages_to_responses_input(messages, supports_images=supports_images),
    }
    if prompt_cache_key is not None:
        payload["prompt_cache_key"] = prompt_cache_key
    if max_tokens is not None:
        payload["max_output_tokens"] = max_tokens
    effort = _normalize_responses_effort(reasoning_effort)
    if effort is not None:
        # ``summary: auto`` streams ``response.reasoning_summary_text.delta``
        # events so the agent's thinking is visible, mirroring the reasoning
        # deltas surfaced on the chat-completions path.
        payload["reasoning"] = {"effort": effort, "summary": "auto"}
    if tools:
        payload["tools"] = [_tool_to_responses(tool) for tool in tools]
    return payload


def _normalize_responses_effort(reasoning_effort: str | None) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Trim and lowercase the effort; None, empty or "none" become None.
    """Map an internal reasoning level to a Responses-API effort, or drop it."""
    if reasoning_effort is None:
        return None
    normalized = reasoning_effort.strip().lower()
    if normalized in ("", "none"):
        return None
    return normalized


def _messages_to_responses_input(
    messages: list[AgentMessage], *, supports_images: bool = False
) -> list[JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Convert messages to Responses input items: user messages as text or
    # input_text/input_image parts; assistant messages as replayed reasoning items
    # (thinking signatures that parse to JSON objects), then the assistant text,
    # then function_call items with portable call IDs and JSON-encoded arguments;
    # tool results as function_call_output items (text, image parts, or "(no tool
    # output)"). Other message types are skipped. Pure.
    # KNOWN ISSUE: unlike the chat path, messages that are not user, assistant or
    # tool-result messages (e.g. custom messages) are silently dropped instead of
    # being converted to user messages.
    items: list[JSONValue] = []
    for message in messages:
        if isinstance(message, UserMessage):
            text, images = text_and_images(
                message.content,
                supports_images=supports_images,
                image_placeholder=NON_VISION_USER_IMAGE_PLACEHOLDER,
            )
            if images:
                content: list[JSONValue] = []
                if text:
                    content.append({"type": "input_text", "text": text})
                content.extend(_openai_input_image(image) for image in images)
                items.append({"role": "user", "content": content})
            else:
                items.append({"role": "user", "content": text})
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ThinkingContent) and block.thinking_signature:
                    try:
                        reasoning_item = loads(block.thinking_signature)
                    except (TypeError, ValueError):
                        reasoning_item = None
                    if isinstance(reasoning_item, dict):
                        items.append(reasoning_item)
            if message.text:
                items.append({"role": "assistant", "content": message.text})
            for tool_call in message.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": portable_tool_call_id(tool_call.id),
                        "name": tool_call.name,
                        "arguments": dumps(tool_call.arguments),
                    }
                )
        elif isinstance(message, ToolResultMessage):
            text, images = text_and_images(
                message.content,
                supports_images=supports_images,
                image_placeholder=NON_VISION_TOOL_IMAGE_PLACEHOLDER,
            )
            output: JSONValue
            if images:
                output_parts: list[JSONValue] = []
                if text:
                    output_parts.append({"type": "input_text", "text": text})
                output_parts.extend(_openai_input_image(image) for image in images)
                output = output_parts
            else:
                output = text or "(no tool output)"
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": portable_tool_call_id(message.tool_call_id),
                    "output": output,
                }
            )
    return items


def _openai_input_image(image: ImageContent) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Convert an image into an input_image part with automatic detail and a
    # base64 data URL. Pure.
    return {
        "type": "input_image",
        "detail": "auto",
        "image_url": f"data:{image.mime_type};base64,{image.data}",
    }


def _tool_to_responses(tool: AgentTool) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Convert a tool to a Responses function definition (name, description,
    # copied parameters schema). Pure.
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": dict(tool.input_schema),
    }


def _register_reasoning_item(
    items: dict[str, dict[str, JSONValue]],
    item: object,
) -> None:
    # [FUNCTION DOC]
    # HELPER: If the item is a reasoning item with a string ID, store a copy under
    # that ID (mutates the mapping).
    if not isinstance(item, Mapping) or item.get("type") != "reasoning":
        return
    item_id = item.get("id")
    if isinstance(item_id, str):
        items[item_id] = dict(item)


def _register_responses_item(
    builders: dict[str, _ResponsesToolCallBuilder],
    item: object,
    *,
    output_index: object,
) -> None:
    # [FUNCTION DOC]
    # HELPER: For a function_call item with a string ID, get or create its builder
    # and set call ID, name, non-empty arguments and output index (mutates).
    if not isinstance(item, Mapping) or item.get("type") != "function_call":
        return
    item_id = item.get("id")
    if not isinstance(item_id, str):
        return
    raw_arguments = item.get("arguments")
    builder = builders.setdefault(item_id, _ResponsesToolCallBuilder())
    builder.set_final(
        call_id=_str_or_none(item.get("call_id")),
        name=_str_or_none(item.get("name")),
        arguments=raw_arguments if isinstance(raw_arguments, str) and raw_arguments else None,
        output_index=_int_or_none(output_index),
    )


def _finalize_responses_item(
    builders: dict[str, _ResponsesToolCallBuilder],
    item: object,
    *,
    output_index: object,
) -> None:
    # [FUNCTION DOC]
    # HELPER: Like _register_responses_item for a completed item, setting any
    # string arguments (including empty) as final (mutates).
    if not isinstance(item, Mapping) or item.get("type") != "function_call":
        return
    item_id = item.get("id")
    if not isinstance(item_id, str):
        return
    builder = builders.setdefault(item_id, _ResponsesToolCallBuilder())
    builder.set_final(
        call_id=_str_or_none(item.get("call_id")),
        name=_str_or_none(item.get("name")),
        arguments=item.get("arguments"),
        output_index=_int_or_none(output_index),
    )


def _ordered_builders(
    builders: dict[str, _ResponsesToolCallBuilder],
) -> list[_ResponsesToolCallBuilder]:
    # [FUNCTION DOC]
    # HELPER: Return builders sorted by output index. Pure.
    return [
        builder for _, builder in sorted(builders.items(), key=lambda pair: pair[1].output_index)
    ]


def _responses_finish_reason(chunk: Mapping[str, Any]) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Return the string status of the event's response object, or None.
    response = chunk.get("response")
    if isinstance(response, Mapping):
        status = response.get("status")
        if isinstance(status, str):
            return status
    return None


def _normalize_finish_reason(status: str | None, *, has_tool_calls: bool) -> str:
    # [FUNCTION DOC]
    # HELPER: Map to "tool_calls" when tool calls exist, "length" for status
    # incomplete, otherwise "stop". Pure.
    """Map a Responses-API status to chat-completions-style finish reasons."""
    if has_tool_calls:
        return "tool_calls"
    if status == "incomplete":
        return "length"
    return "stop"


def _responses_failure_event(chunk: Mapping[str, Any]) -> ProviderErrorEvent:
    # [FUNCTION DOC]
    # HELPER: Build an error event from a response.failed event using the nested
    # error message (default "Provider response failed") with the event as data.
    message = "Provider response failed"
    response = chunk.get("response")
    if isinstance(response, Mapping):
        error = response.get("error")
        if isinstance(error, Mapping):
            error_message = error.get("message")
            if isinstance(error_message, str) and error_message:
                message = error_message
    return ProviderErrorEvent(message=message, data={"event": dict(chunk)})


def _responses_error_message(chunk: Mapping[str, Any]) -> str:
    # [FUNCTION DOC]
    # HELPER: Return the event's message, else a nested error message, else
    # "Provider stream error". Pure.
    message = chunk.get("message")
    if isinstance(message, str) and message:
        return message
    error = chunk.get("error")
    if isinstance(error, Mapping):
        nested = error.get("message")
        if isinstance(nested, str) and nested:
            return nested
    return "Provider stream error"


def _str_or_none(value: object) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Return the value if it is a non-empty string, else None.
    return value if isinstance(value, str) and value else None


def _system_message(system: str) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Return a system-role chat message with the given content.
    return {"role": "system", "content": system}


def _messages_to_openai_chat(
    messages: list[AgentMessage], *, supports_images: bool
) -> list[dict[str, JSONValue]]:
    # [FUNCTION DOC]
    # HELPER: Convert messages to chat messages. User messages become text or
    # text/image_url parts. Tool results become "tool" messages (text, or "(see
    # attached image)" / "(no tool output)"); their images are buffered and emitted
    # as one following user message ("Attached image(s) from tool result:") once a
    # non-tool-result message (or the end) is reached, since tool messages cannot
    # carry images. Other messages use _message_to_openai. Pure.
    converted: list[dict[str, JSONValue]] = []
    pending_tool_images: list[ImageContent] = []
    for message in messages:
        if pending_tool_images and not isinstance(message, ToolResultMessage):
            converted.append(_openai_tool_image_message(pending_tool_images))
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
                        "image_url": {"url": f"data:{image.mime_type};base64,{image.data}"},
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
        converted.append(_message_to_openai(message))
    if pending_tool_images:
        converted.append(_openai_tool_image_message(pending_tool_images))
    return converted


def _openai_tool_image_message(images: list[ImageContent]) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Build a user message with an introductory text part and one
    # image_url data-URL part per image. Pure.
    content: list[JSONValue] = [{"type": "text", "text": "Attached image(s) from tool result:"}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": f"data:{image.mime_type};base64,{image.data}"},
        }
        for image in images
    )
    return {"role": "user", "content": content}


def _message_to_openai(message: AgentMessage) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Convert one message to chat format: user text; assistant text plus
    # joined thinking under its signature field name (reasoning_content by default,
    # only if the signature is reasoning_content, reasoning or thinking) and tool
    # calls; tool results as tool messages; other types via conversion to a user
    # message. Pure.
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.text}

    if isinstance(message, AssistantMessage):
        item: dict[str, JSONValue] = {"role": "assistant", "content": message.text}
        thinking = [block for block in message.content if isinstance(block, ThinkingContent)]
        if thinking:
            signature = thinking[0].thinking_signature or "reasoning_content"
            if signature in {"reasoning_content", "reasoning", "thinking"}:
                item[signature] = "".join(block.thinking for block in thinking)
        if message.tool_calls:
            item["tool_calls"] = [
                _tool_call_to_openai(tool_call) for tool_call in message.tool_calls
            ]
        return item

    if isinstance(message, ToolResultMessage):
        return {
            "role": "tool",
            "tool_call_id": portable_tool_call_id(message.tool_call_id),
            "name": message.tool_name,
            "content": message.text,
        }
    return _message_to_openai(message_to_user(message))


def _tool_to_openai(tool: AgentTool) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Convert a tool to a chat "function" tool definition. Pure.
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.input_schema),
        },
    }


def _tool_call_to_openai(tool_call: ToolCall) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Convert a tool call to chat format with a portable ID and
    # JSON-encoded arguments. Pure.
    return {
        "id": portable_tool_call_id(tool_call.id),
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": dumps(tool_call.arguments),
        },
    }


def _parse_sse_line(line: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Return the stripped payload of an SSE "data:" line (possibly an empty
    # string), or None for blank or non-data lines. Pure.
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    return line.removeprefix("data:").strip()


def _loads_object(value: str) -> dict[str, JSONValue] | None:
    # [FUNCTION DOC]
    # HELPER: Parse JSON and return it only if it is an object; None otherwise.
    try:
        loaded = loads(value)
    except JSONDecodeError:
        return None
    if isinstance(loaded, dict):
        return loaded
    return None


def _first_choice(chunk: Mapping[str, Any]) -> Mapping[str, Any] | None:
    # [FUNCTION DOC]
    # HELPER: Return the first element of the chunk's choices list if it is a
    # mapping, else None.
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return None
    return choice


def _int_or_zero(value: object) -> int:
    # [FUNCTION DOC]
    # HELPER: Return the value if it is a non-boolean integer, else 0.
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _int_or_none(value: object) -> int | None:
    # [FUNCTION DOC]
    # HELPER: Return the value if it is a non-boolean integer, else None.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _parse_chunk_usage(raw: Mapping[str, Any]) -> Usage:
    # [FUNCTION DOC]
    # HELPER: Convert chat usage into Usage: cache reads from cached_tokens (falling
    # back to prompt_cache_hit_tokens only when absent), cache writes from
    # cache_write_tokens, fresh input as prompt tokens minus reads and writes
    # (minimum 0), output from completion tokens, reasoning tokens when details
    # exist, and the total as the sum. Pure.
    """Parse an OpenAI-compatible ``usage`` payload into a Usage.

    Ports Pi's openai-completions.ts parseChunkUsage: ``cached_tokens`` are
    cache reads, writes are subtracted from the prompt to leave the fresh input,
    and ``completion_tokens`` already includes reasoning tokens. Cost is left
    unset (None) because Tau has no per-model pricing table.
    """
    prompt_tokens = _int_or_zero(raw.get("prompt_tokens"))
    prompt_details = raw.get("prompt_tokens_details")
    cached_tokens: int | None = None
    cache_write = 0
    if isinstance(prompt_details, Mapping):
        cached_tokens = _int_or_none(prompt_details.get("cached_tokens"))
        cache_write = _int_or_zero(prompt_details.get("cache_write_tokens"))
    # Nullish fallback, matching Pi's `cached_tokens ?? prompt_cache_hit_tokens
    # ?? 0` (DeepSeek reports cache hits in prompt_cache_hit_tokens): a reported
    # 0 does not fall through.
    if cached_tokens is None:
        cached_tokens = _int_or_none(raw.get("prompt_cache_hit_tokens"))
    cache_read = cached_tokens or 0
    fresh_input = max(0, prompt_tokens - cache_read - cache_write)
    output = _int_or_zero(raw.get("completion_tokens"))
    reasoning = None
    completion_details = raw.get("completion_tokens_details")
    if isinstance(completion_details, Mapping):
        reasoning = _int_or_zero(completion_details.get("reasoning_tokens"))
    return Usage(
        input=fresh_input,
        output=output,
        cache_read=cache_read,
        cache_write=cache_write,
        reasoning=reasoning,
        total_tokens=fresh_input + output + cache_read + cache_write,
    )


def _usage_from_responses_event(chunk: Mapping[str, Any]) -> Usage | None:
    # [FUNCTION DOC]
    # HELPER: Convert the usage of a Responses terminal event into Usage (None if
    # absent): cache read/write from input details, fresh input as input tokens
    # minus those (minimum 0), output tokens, reasoning only when output details
    # exist, and the provider-reported total. Pure.
    """Parse billed usage from a `/v1/responses` terminal event.

    Mirrors the Codex adapter's ``_usage_from_response``: cache reads and
    writes are subtracted from ``input_tokens`` to leave fresh input. Cost is
    left unset because Tau has no per-model pricing table.
    """
    response = chunk.get("response")
    if not isinstance(response, Mapping):
        return None
    raw = response.get("usage")
    if not isinstance(raw, Mapping):
        return None
    input_details = raw.get("input_tokens_details")
    cache_read = (
        _int_or_zero(input_details.get("cached_tokens"))
        if isinstance(input_details, Mapping)
        else 0
    )
    cache_write = (
        _int_or_zero(input_details.get("cache_write_tokens"))
        if isinstance(input_details, Mapping)
        else 0
    )
    output_details = raw.get("output_tokens_details")
    # Leave reasoning None (not 0) when the provider reports no breakdown,
    # honoring the "None = not reported" contract on Usage.
    reasoning = (
        _int_or_zero(output_details.get("reasoning_tokens"))
        if isinstance(output_details, Mapping)
        else None
    )
    return Usage(
        input=max(0, _int_or_zero(raw.get("input_tokens")) - cache_read - cache_write),
        output=_int_or_zero(raw.get("output_tokens")),
        cache_read=cache_read,
        cache_write=cache_write,
        reasoning=reasoning,
        total_tokens=_int_or_zero(raw.get("total_tokens")),
    )


def _tool_call_deltas(delta: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    # [FUNCTION DOC]
    # HELPER: Return the mapping entries of the delta's tool_calls list (empty if
    # not a list).
    tool_calls = delta.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [tool_call for tool_call in tool_calls if isinstance(tool_call, Mapping)]


def _thinking_delta(delta: Mapping[str, Any]) -> tuple[str, str] | None:
    # [FUNCTION DOC]
    # HELPER: Return (field name, text) for the first non-empty string among
    # reasoning_content, reasoning and thinking in the delta, else None.
    for field_name in ("reasoning_content", "reasoning", "thinking"):
        value = delta.get(field_name)
        if isinstance(value, str) and value:
            return field_name, value
    return None


def _is_transient_status(status_code: int) -> bool:
    # [FUNCTION DOC]
    # HELPER: True for HTTP 408, 409, 425, 429 and any 5xx status.
    return status_code in {408, 409, 425, 429} or status_code >= 500
