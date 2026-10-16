# [MODULE DOC]
# Provider adapter for ChatGPT-subscription OpenAI Codex, which speaks the OpenAI
# Responses API over Server-Sent Events at chatgpt.com/backend-api/codex. It
# converts Tau agent messages and tools into a Responses request payload, streams
# the SSE reply into provider-neutral events (text, thinking, tool calls, end,
# errors, retries) that are then canonicalized into Pi-style assistant message
# events, retries transient HTTP/transport/in-stream failures with backoff, and
# discovers the account's model catalog and limits from the /codex/models
# endpoint. Credentials (bearer token + ChatGPT account id) come from an async
# resolver so OAuth refresh stays outside this module.
"""OpenAI Codex subscription Responses provider."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from json import JSONDecodeError, dumps, loads
from platform import machine, release, system
from typing import Any, Literal, cast

import httpx

from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolResultMessage,
    Usage,
    UserMessage,
    assistant_content,
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
from tau_ai.env import (
    DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES,
    DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS,
    DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
)
from tau_ai.events import AssistantMessageEvent
from tau_ai.http import create_async_client
from tau_ai.http_errors import provider_http_error_message
from tau_ai.model_catalog import RuntimeModel, RuntimeModelCatalog, RuntimeThinkingLevel
from tau_ai.model_limits import RuntimeModelLimits
from tau_ai.openai_cache import openai_prompt_cache_key
from tau_ai.provider import CancellationToken
from tau_ai.retry import (
    RETRYABLE_TRANSPORT_ERRORS,
    provider_retry_event,
    retry_delay_seconds,
    wait_for_retry,
)
from tau_ai.stream import canonicalize_provider_stream

DEFAULT_OPENAI_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
DEFAULT_OPENAI_CODEX_CLIENT_VERSION = "0.153.4"


@dataclass(frozen=True, slots=True)
class OpenAICodexCredentials:
    # [CLASS DOC]
    # Immutable pair of OAuth access token and ChatGPT account id sent with each
    # request.
    """Bearer token and account id required by ChatGPT Codex Responses."""

    access_token: str
    account_id: str


type OpenAICodexCredentialResolver = Callable[[], Awaitable[OpenAICodexCredentials]]
type OpenAICodexClientVersionResolver = Callable[[], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class OpenAICodexConfig:
    # [CLASS DOC]
    # Immutable adapter configuration: async credential resolver; base URL; extra
    # headers; HTTP timeout; max retries and max backoff delay; originator header
    # value; optional reasoning effort and reasoning summary mode ("auto"); whether
    # the model accepts images; provider display name used in error messages; the
    # fallback Codex client version string, an optional async resolver for the
    # latest version, and the model-catalog request timeout (5s).
    """Configuration for the OpenAI Codex subscription Responses endpoint."""

    credential_resolver: OpenAICodexCredentialResolver
    base_url: str = DEFAULT_OPENAI_CODEX_BASE_URL
    headers: Mapping[str, str] | None = None
    timeout_seconds: float = DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES
    max_retry_delay_seconds: float = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS
    originator: str = "tau"
    reasoning_effort: str | None = None
    reasoning_summary: str = "auto"
    supports_images: bool = False
    provider_name: str = "OpenAI Codex"
    # The endpoint requires an official Codex compatibility version and filters
    # newer models from older clients. Tau resolves the latest release at runtime;
    # this bundled value remains the offline/error fallback.
    client_version: str = DEFAULT_OPENAI_CODEX_CLIENT_VERSION
    client_version_resolver: OpenAICodexClientVersionResolver | None = None
    model_catalog_timeout_seconds: float = 5.0


class OpenAICodexProvider:
    # [CLASS DOC]
    # Model provider implementation for Codex. Lazily creates (and then owns) an
    # HTTP client if none was injected, caches the discovered model catalog and
    # limits after the first fetch, and remembers the account id used for the most
    # recent catalog request.
    """Provider adapter for ChatGPT subscription Codex Responses over SSE."""

    def __init__(
        self,
        config: OpenAICodexConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Store the config and optional injected client (an injected client is never
        # closed by this provider); initialize empty catalog/limit caches and account id.
        self._config = config
        self._client = client
        self._owns_client = client is None
        self._discovered_model_catalog: RuntimeModelCatalog | None = None
        self._discovered_model_limits: dict[str, RuntimeModelLimits] | None = None
        self._account_id: str | None = None

    @property
    def account_id(self) -> str | None:
        # [FUNCTION DOC]
        # Read-only property: account id from the latest catalog fetch, or None if the
        # catalog has not been fetched.
        """Return the account associated with the latest catalog request."""
        return self._account_id

    async def aclose(self) -> None:
        # [FUNCTION DOC]
        # Close and forget the HTTP client only if this provider created it. Idempotent.
        """Close the underlying HTTP client if this provider created it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def discover_models(self) -> RuntimeModelCatalog:
        # [FUNCTION DOC]
        # Return the account's selectable Codex models, fetching and caching the catalog
        # on first use. Raises HTTP/transport errors or credential resolver errors from
        # the fetch (not cached on failure).
        """Discover the authenticated account's selectable Codex models."""
        await self._ensure_model_catalog()
        assert self._discovered_model_catalog is not None
        return self._discovered_model_catalog

    async def discover_model_limits(self, model: str) -> RuntimeModelLimits | None:
        # [FUNCTION DOC]
        # Return runtime limits for a model id from the (lazily fetched, cached)
        # catalog, or None when the model is unknown or has no usable limits. Raises as
        # discover_models.
        """Discover model limits from the authenticated Codex model catalog."""
        await self._ensure_model_catalog()
        assert self._discovered_model_limits is not None
        return self._discovered_model_limits.get(model)

    async def _ensure_model_catalog(self) -> None:
        # [FUNCTION DOC]
        # HELPER: If not cached yet, fetch the catalog payload and parse both the model
        # catalog and the per-model limits into the caches. Raises fetch errors.
        if self._discovered_model_catalog is not None:
            return
        payload = await self._fetch_model_catalog()
        self._discovered_model_catalog = _parse_codex_model_catalog(payload)
        self._discovered_model_limits = _parse_codex_model_limits(payload)

    async def _fetch_model_catalog(self) -> object:
        # [FUNCTION DOC]
        # HELPER: GET the models endpoint derived from the base URL with query parameter
        # client_version (from the resolver if configured, else the bundled version),
        # auth headers (accept JSON, no content-type) and the catalog timeout. Records
        # the account id. Returns the decoded JSON body. Raises httpx status/transport
        # errors, JSON decode errors, or credential resolver errors.
        client = self._get_client()
        credentials = await self._config.credential_resolver()
        self._account_id = credentials.account_id
        headers = _build_codex_headers(
            self._config.headers,
            access_token=credentials.access_token,
            account_id=credentials.account_id,
            originator=self._config.originator,
        )
        headers["accept"] = "application/json"
        headers.pop("content-type", None)
        client_version = (
            await self._config.client_version_resolver()
            if self._config.client_version_resolver is not None
            else self._config.client_version
        )
        response = await client.get(
            _resolve_codex_models_url(self._config.base_url),
            params={"client_version": client_version},
            headers=headers,
            timeout=self._config.model_catalog_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

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
        # Public streaming entry point. Inputs (keyword-only): model id, system prompt,
        # conversation messages, tools, optional cancellation token and optional session
        # id (used for prompt caching). Returns an async iterator of Pi-compatible
        # assistant message events produced by canonicalizing the raw provider event
        # stream with api "openai-codex-responses" and provider "openai-codex". Lazy:
        # no request is made until iteration starts.
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
            raw, api="openai-codex-responses", provider="openai-codex", model=model
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
        # HELPER: Return an async generator of provider-neutral events for one request.
        # See the nested iterator for behavior.
        """Stream one Codex Responses request as provider-neutral events."""

        async def iterator() -> AsyncIterator[ProviderEvent]:
            # [FUNCTION DOC]
            # HELPER: Performs the request with retries. Builds the payload once (with a
            # prompt cache key derived from the session id), then loops per attempt:
            # resolves credentials, builds headers (including session-id = cache key), and
            # POSTs as a stream.
            # On HTTP status >= 400: reads the body; if retryable (attempt below max retries
            # and status 408/409/425/429/5xx, but not a terminal quota/billing 429) yields a
            # retry event, waits (returning silently if cancelled) and retries; otherwise
            # yields an error event with a friendly message, status, body and attempt count.
            # On success: yields a response-start event then relays parsed SSE events. An
            # in-stream error event that arrives before any text/tool/thinking output and
            # looks transient (overloaded, rate limit, timeout, ...) is not relayed; instead
            # a retry event is yielded and the request is retried. Otherwise events are
            # relayed until the stream ends.
            # Transport errors (httpx errors, SSL errors) before content is emitted are
            # retried the same way; afterwards (or when retries are exhausted) an error event
            # is yielded. Any other exception becomes an error event with the attempt count.
            # Never raises for request failures; ends after the first terminal outcome.
            # KNOWN ISSUE: the transport-retry check ignores already-emitted thinking deltas
            # (unlike the in-stream retry check), so a retried attempt can duplicate
            # reasoning text; also each retried attempt yields a fresh response-start event.
            client = self._get_client()
            cache_key = openai_prompt_cache_key(session_id)
            payload = _build_codex_payload(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                reasoning_effort=self._config.reasoning_effort,
                reasoning_summary=self._config.reasoning_summary,
                supports_images=self._config.supports_images,
                prompt_cache_key=cache_key,
            )
            url = _resolve_codex_url(self._config.base_url)

            attempt = 0
            while True:
                emitted_content = False
                emitted_thinking = False
                try:
                    credentials = await self._config.credential_resolver()
                    headers = _build_codex_headers(
                        self._config.headers,
                        access_token=credentials.access_token,
                        account_id=credentials.account_id,
                        originator=self._config.originator,
                        session_id=cache_key,
                    )
                    async with client.stream(
                        "POST",
                        url,
                        json=payload,
                        headers=headers,
                    ) as response:
                        if response.status_code >= 400:
                            body = await response.aread()
                            body_text = body.decode(errors="replace")
                            if self._should_retry(
                                attempt,
                                status_code=response.status_code,
                                body=body_text,
                            ):
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
                            )
                            return

                        yield ProviderResponseStartEvent(model=model)
                        stream_error: dict[str, JSONValue] | None = None
                        async for event in _codex_provider_events(response, signal=signal):
                            if isinstance(
                                event,
                                ProviderTextDeltaEvent | ProviderToolCallEvent,
                            ):
                                emitted_content = True
                            elif isinstance(event, ProviderThinkingDeltaEvent):
                                emitted_thinking = True
                            if (
                                isinstance(event, ProviderErrorEvent)
                                and not emitted_content
                                and not emitted_thinking
                                and self._should_retry(attempt)
                                and _retryable_stream_error_event(event)
                            ):
                                stream_error = _stream_error_event_data(event)
                                break
                            yield event
                        if stream_error is None:
                            return
                        code, _message = _stream_error_details(stream_error)
                        delay = retry_delay_seconds(
                            attempt,
                            max_delay_seconds=self._config.max_retry_delay_seconds,
                        )
                        yield provider_retry_event(
                            attempt=attempt,
                            max_retries=self._config.max_retries,
                            delay_seconds=delay,
                            reason=f"stream error ({code or 'unknown'})",
                            data={"event": stream_error},
                        )
                        attempt += 1
                        if not await wait_for_retry(delay, signal=signal):
                            return
                        continue
                except RETRYABLE_TRANSPORT_ERRORS as exc:
                    if not emitted_content and self._should_retry(attempt):
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
                except Exception as exc:  # noqa: BLE001 - provider errors are surfaced as events
                    yield ProviderErrorEvent(message=str(exc), data={"attempts": attempt + 1})
                    return

        return iterator()

    def _get_client(self) -> httpx.AsyncClient:
        # [FUNCTION DOC]
        # HELPER: Return the HTTP client, creating one with the configured timeout on
        # first use (marked as owned at construction when none was injected).
        if self._client is None:
            self._client = create_async_client(timeout=self._config.timeout_seconds)
        return self._client

    def _should_retry(
        self,
        attempt: int,
        *,
        status_code: int | None = None,
        body: str = "",
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: Return False when the attempt index has reached max retries; otherwise
        # True when no status code is given (transport/stream errors) or when the status
        # and body are considered retryable.
        if attempt >= self._config.max_retries:
            return False
        return status_code is None or _is_retryable_status(status_code, body)


class _ToolCallBuilder:
    # [CLASS DOC]
    # HELPER: Mutable accumulator for one streamed function call: call id, optional
    # Responses item id, function name and argument text fragments.
    def __init__(self, *, call_id: str, item_id: str | None, name: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Store call id, item id and name (keyword-only) and start with no
        # argument fragments.
        self.call_id = call_id
        self.item_id = item_id
        self.name = name
        self.arguments_parts: list[str] = []

    def add_delta(self, delta: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Append a streamed argument JSON fragment.
        """Append a streamed tool-argument fragment."""
        self.arguments_parts.append(delta)

    def set_arguments(self, arguments: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Replace all fragments with the final complete argument string.
        """Replace streamed tool arguments with final provider arguments."""
        self.arguments_parts = [arguments]

    def update_from_item(self, item: Mapping[str, Any]) -> None:
        # [FUNCTION DOC]
        # HELPER: Overwrite call id and item id from a completed item when they are
        # non-empty strings, and the name when it is a string.
        """Fill in metadata from a completed function-call item."""
        call_id = item.get("call_id")
        if isinstance(call_id, str) and call_id:
            self.call_id = call_id
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id:
            self.item_id = item_id
        name = item.get("name")
        if isinstance(name, str):
            self.name = name

    def build(self) -> ToolCall:
        # [FUNCTION DOC]
        # HELPER: Join fragments and parse them as a JSON object (empty text gives an
        # empty mapping; unparsable or non-object text gives a mapping with the raw text
        # under "_raw_arguments"). Returns a tool call whose id is "<call id>|<item id>"
        # (item id defaults to "fc_<call id>") so the item id can be replayed later.
        """Build a complete Tau tool call."""
        arguments_text = "".join(self.arguments_parts)
        arguments = _loads_object(arguments_text) if arguments_text else {}
        if arguments is None:
            arguments = {"_raw_arguments": arguments_text}
        item_id = self.item_id or f"fc_{self.call_id}"
        return ToolCall(
            id=f"{self.call_id}|{item_id}",
            name=self.name,
            arguments=arguments,
        )


def _build_codex_payload(
    *,
    model: str,
    system: str,
    messages: list[AgentMessage],
    tools: list[AgentTool],
    reasoning_effort: str | None = None,
    reasoning_summary: str = "auto",
    supports_images: bool = False,
    prompt_cache_key: str | None = None,
) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Build the Responses request body: model, store false, stream true,
    # instructions (system prompt or a generic default), input items converted from
    # messages, low text verbosity, include encrypted reasoning content, automatic
    # tool choice and parallel tool calls; adds prompt_cache_key when given, a
    # reasoning object (effort plus summary mode) when an effort is configured, and
    # function tool definitions when tools exist. Returns the mapping.
    payload: dict[str, JSONValue] = {
        "model": model,
        "store": False,
        "stream": True,
        "instructions": system or "You are a helpful assistant.",
        "input": _messages_to_responses_input(messages, supports_images=supports_images),
        "text": {"verbosity": "low"},
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
    if prompt_cache_key is not None:
        payload["prompt_cache_key"] = prompt_cache_key
    if reasoning_effort is not None:
        payload["reasoning"] = {
            "effort": reasoning_effort,
            "summary": reasoning_summary,
        }
    if tools:
        payload["tools"] = [_tool_to_codex(tool) for tool in tools]
    return payload


def _messages_to_responses_input(
    messages: list[AgentMessage], *, supports_images: bool = False
) -> list[JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Convert agent messages to Responses input items. User messages become
    # a user item with input_text and input_image parts (images replaced by a
    # placeholder text when images are unsupported). Assistant messages: thinking
    # blocks whose signature holds a JSON reasoning item are replayed verbatim;
    # text blocks become completed assistant message items with output_text and an
    # id from the text signature or "msg_<n>" (counter across all assistant
    # messages); every tool call becomes a function_call item with the call id,
    # name, JSON-encoded arguments and the item id split from the combined id. Tool
    # results become function_call_output items whose output is text (or
    # "(no tool output)") or a list of text/image parts when images are present.
    # Other message types are skipped. Returns the list.
    items: list[JSONValue] = []
    assistant_index = 0
    for message in messages:
        if isinstance(message, UserMessage):
            text, images = text_and_images(
                message.content,
                supports_images=supports_images,
                image_placeholder=NON_VISION_USER_IMAGE_PLACEHOLDER,
            )
            content: list[JSONValue] = []
            if text:
                content.append({"type": "input_text", "text": text})
            content.extend(_codex_input_image(image) for image in images)
            items.append({"role": "user", "content": content})
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ThinkingContent) and block.thinking_signature:
                    try:
                        reasoning_item = loads(block.thinking_signature)
                    except (TypeError, ValueError):
                        reasoning_item = None
                    if isinstance(reasoning_item, dict):
                        items.append(reasoning_item)
                elif isinstance(block, TextContent):
                    items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": block.text,
                                    "annotations": [],
                                }
                            ],
                            "status": "completed",
                            "id": block.text_signature or f"msg_{assistant_index}",
                        }
                    )
                    assistant_index += 1
            for tool_call in message.tool_calls:
                call_id, item_id = _split_tool_call_id(tool_call.id)
                item: dict[str, JSONValue] = {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": tool_call.name,
                    "arguments": dumps(tool_call.arguments),
                }
                if item_id:
                    item["id"] = item_id
                items.append(item)
        elif isinstance(message, ToolResultMessage):
            call_id, _item_id = _split_tool_call_id(message.tool_call_id)
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
                output_parts.extend(_codex_input_image(image) for image in images)
                output = output_parts
            else:
                output = text or "(no tool output)"
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }
            )
    return items


def _codex_input_image(image: ImageContent) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Convert an image block into an input_image part with auto detail and a
    # base64 data URL built from its MIME type and data.
    return {
        "type": "input_image",
        "detail": "auto",
        "image_url": f"data:{image.mime_type};base64,{image.data}",
    }


def _tool_to_codex(tool: AgentTool) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # HELPER: Convert a tool to a Responses function tool definition (name,
    # description, a copy of the JSON Schema parameters, strict set to null).
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": dict(tool.input_schema),
        "strict": None,
    }


async def _codex_provider_events(
    response: httpx.Response,
    *,
    signal: CancellationToken | None,
) -> AsyncIterator[ProviderEvent]:
    # [FUNCTION DOC]
    # HELPER: Parse a Codex SSE response into provider events. Stops silently when
    # the token is cancelled. Handles: "error" and "response.failed" (yield an error
    # event carrying the raw event, then stop); output_item.added (remember
    # reasoning items; start tracking function-call builders); function call
    # argument delta/done (update the matching builder); output_text.delta (yield
    # text delta); reasoning / reasoning summary / reasoning text deltas (yield
    # thinking delta); reasoning summary part done (insert a blank-line separator
    # into thinking); output_item.done/completed (store final reasoning items; for
    # function calls finalize the builder, yield a tool call event and stop tracking
    # it; for a message item when no text was streamed, yield its full text);
    # response.done/completed/incomplete (capture finish reason = response status
    # and usage, then stop reading). Finally yields a response-end event with an
    # assistant message whose content is text plus tool calls, prefixed by one
    # thinking block (with a JSON signature of a reasoning item) when any thinking
    # was streamed, and the usage (or empty usage).
    # KNOWN ISSUE: only the first reasoning item is kept as the thinking signature,
    # so responses with several reasoning items lose the others on replay.
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    reasoning_items: dict[str, dict[str, JSONValue]] = {}
    tool_calls: list[ToolCall] = []
    active_tools: list[_ToolCallBuilder] = []
    tools_by_item_id: dict[str, _ToolCallBuilder] = {}
    tools_by_call_id: dict[str, _ToolCallBuilder] = {}
    tools_by_output_index: dict[int, _ToolCallBuilder] = {}
    finish_reason: str | None = None
    usage: Usage | None = None

    async for event in _iter_sse_objects(response):
        if signal is not None and signal.is_cancelled():
            return
        event_type = event.get("type")
        if not isinstance(event_type, str):
            continue

        if event_type == "error":
            yield ProviderErrorEvent(
                message=_error_message(event, fallback="OpenAI Codex returned an error"),
                data={"event": event},
            )
            return

        if event_type == "response.failed":
            yield ProviderErrorEvent(
                message=_response_error_message(event),
                data={"event": event},
            )
            return

        if event_type == "response.output_item.added":
            item = event.get("item")
            if isinstance(item, Mapping) and item.get("type") == "reasoning":
                item_id = item.get("id")
                if isinstance(item_id, str):
                    reasoning_items[item_id] = dict(item)
            elif isinstance(item, Mapping) and item.get("type") == "function_call":
                _track_tool_builder(
                    _tool_builder_from_item(item),
                    event,
                    active_tools=active_tools,
                    by_item_id=tools_by_item_id,
                    by_call_id=tools_by_call_id,
                    by_output_index=tools_by_output_index,
                )

        elif event_type == "response.function_call_arguments.delta":
            delta = event.get("delta")
            tool_builder = _tool_builder_for_event(
                event,
                active_tools=active_tools,
                by_item_id=tools_by_item_id,
                by_call_id=tools_by_call_id,
                by_output_index=tools_by_output_index,
            )
            if tool_builder is not None and isinstance(delta, str):
                tool_builder.add_delta(delta)

        elif event_type == "response.function_call_arguments.done":
            arguments = event.get("arguments")
            tool_builder = _tool_builder_for_event(
                event,
                active_tools=active_tools,
                by_item_id=tools_by_item_id,
                by_call_id=tools_by_call_id,
                by_output_index=tools_by_output_index,
            )
            if tool_builder is not None and isinstance(arguments, str):
                tool_builder.set_arguments(arguments)

        elif event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                content_parts.append(delta)
                yield ProviderTextDeltaEvent(delta=delta)

        elif event_type in {
            "response.reasoning.delta",
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        }:
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                thinking_parts.append(delta)
                yield ProviderThinkingDeltaEvent(delta=delta)

        elif event_type == "response.reasoning_summary_part.done":
            if thinking_parts:
                separator = "\n\n"
                thinking_parts.append(separator)
                yield ProviderThinkingDeltaEvent(delta=separator)

        elif event_type in {
            "response.output_item.done",
            "response.output_item.completed",
        }:
            item = event.get("item")
            if isinstance(item, Mapping) and item.get("type") == "reasoning":
                item_id = item.get("id")
                if isinstance(item_id, str):
                    reasoning_items[item_id] = dict(item)
            elif isinstance(item, Mapping) and item.get("type") == "function_call":
                tool_builder = _tool_builder_for_event(
                    event,
                    active_tools=active_tools,
                    by_item_id=tools_by_item_id,
                    by_call_id=tools_by_call_id,
                    by_output_index=tools_by_output_index,
                )
                if tool_builder is None:
                    tool_builder = _tool_builder_from_item(item)
                    _track_tool_builder(
                        tool_builder,
                        event,
                        active_tools=active_tools,
                        by_item_id=tools_by_item_id,
                        by_call_id=tools_by_call_id,
                        by_output_index=tools_by_output_index,
                    )
                else:
                    tool_builder.update_from_item(item)
                arguments = item.get("arguments")
                if isinstance(arguments, str):
                    tool_builder.set_arguments(arguments)
                tool_call = tool_builder.build()
                tool_calls.append(tool_call)
                _untrack_tool_builder(
                    tool_builder,
                    active_tools=active_tools,
                    by_item_id=tools_by_item_id,
                    by_call_id=tools_by_call_id,
                    by_output_index=tools_by_output_index,
                )
                yield ProviderToolCallEvent(tool_call=tool_call)
            elif isinstance(item, Mapping) and item.get("type") == "message" and not content_parts:
                text = _text_from_done_message(item)
                if text:
                    content_parts.append(text)
                    yield ProviderTextDeltaEvent(delta=text)

        elif event_type in {
            "response.done",
            "response.completed",
            "response.incomplete",
        }:
            finish_reason = _finish_reason_from_response(event)
            usage = _usage_from_response(event) or usage
            break

    content = assistant_content("".join(content_parts), tool_calls)
    if thinking_parts:
        content.insert(
            0,
            ThinkingContent(
                thinking="".join(thinking_parts),
                thinking_signature=(
                    dumps(next(iter(reasoning_items.values()))) if reasoning_items else None
                ),
            ),
        )
    yield ProviderResponseEndEvent(
        message=AssistantMessage(
            content=content,
            usage=usage or Usage(),
        ),
        finish_reason=finish_reason,
    )


async def _iter_sse_objects(response: httpx.Response) -> AsyncIterator[dict[str, JSONValue]]:
    # [FUNCTION DOC]
    # HELPER: Async generator over SSE "data:" payloads of a response. Accumulates
    # data lines (whitespace-trimmed) until a blank line, then joins them with
    # newlines and yields the payload if it parses as a JSON object (others are
    # dropped). Non-data lines (event:, comments) are ignored; "[DONE]" ends the
    # stream; a trailing unterminated payload is flushed at the end.
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        stripped = line.strip()
        if not stripped:
            if data_lines:
                data = "\n".join(data_lines).strip()
                data_lines = []
                parsed = _loads_object(data)
                if parsed is not None:
                    yield parsed
            continue
        if not stripped.startswith("data:"):
            continue
        value = stripped.removeprefix("data:").strip()
        if value == "[DONE]":
            break
        data_lines.append(value)

    if data_lines:
        parsed = _loads_object("\n".join(data_lines).strip())
        if parsed is not None:
            yield parsed


def _tool_builder_from_item(item: Mapping[str, Any]) -> _ToolCallBuilder:
    # [FUNCTION DOC]
    # HELPER: Create a tool call builder from a function_call item, defaulting the
    # call id to "call_0", item id to None and name to an empty string when absent.
    call_id = item.get("call_id")
    name = item.get("name")
    item_id = item.get("id")
    return _ToolCallBuilder(
        call_id=call_id if isinstance(call_id, str) and call_id else "call_0",
        item_id=item_id if isinstance(item_id, str) and item_id else None,
        name=name if isinstance(name, str) else "",
    )


def _track_tool_builder(
    builder: _ToolCallBuilder,
    event: Mapping[str, Any],
    *,
    active_tools: list[_ToolCallBuilder],
    by_item_id: dict[str, _ToolCallBuilder],
    by_call_id: dict[str, _ToolCallBuilder],
    by_output_index: dict[int, _ToolCallBuilder],
) -> None:
    # [FUNCTION DOC]
    # HELPER: Register a builder as active (once) and index it by item id, call id
    # and the event's output index when present. Mutates the given collections.
    if builder not in active_tools:
        active_tools.append(builder)
    if builder.item_id:
        by_item_id[builder.item_id] = builder
    if builder.call_id:
        by_call_id[builder.call_id] = builder
    output_index = _event_output_index(event)
    if output_index is not None:
        by_output_index[output_index] = builder


def _untrack_tool_builder(
    builder: _ToolCallBuilder,
    *,
    active_tools: list[_ToolCallBuilder],
    by_item_id: dict[str, _ToolCallBuilder],
    by_call_id: dict[str, _ToolCallBuilder],
    by_output_index: dict[int, _ToolCallBuilder],
) -> None:
    # [FUNCTION DOC]
    # HELPER: Remove a builder from the active list and from every index where it is
    # the registered builder. Mutates the given collections.
    if builder in active_tools:
        active_tools.remove(builder)
    if builder.item_id and by_item_id.get(builder.item_id) is builder:
        del by_item_id[builder.item_id]
    if builder.call_id and by_call_id.get(builder.call_id) is builder:
        del by_call_id[builder.call_id]
    for output_index, tracked_builder in tuple(by_output_index.items()):
        if tracked_builder is builder:
            del by_output_index[output_index]


def _tool_builder_for_event(
    event: Mapping[str, Any],
    *,
    active_tools: list[_ToolCallBuilder],
    by_item_id: dict[str, _ToolCallBuilder],
    by_call_id: dict[str, _ToolCallBuilder],
    by_output_index: dict[int, _ToolCallBuilder],
) -> _ToolCallBuilder | None:
    # [FUNCTION DOC]
    # HELPER: Find the builder an event refers to, trying item id, then call id,
    # then output index; if none match but exactly one builder is active, return
    # it; otherwise None.
    item_id = _event_item_id(event)
    if item_id is not None and item_id in by_item_id:
        return by_item_id[item_id]
    call_id = _event_call_id(event)
    if call_id is not None and call_id in by_call_id:
        return by_call_id[call_id]
    output_index = _event_output_index(event)
    if output_index is not None and output_index in by_output_index:
        return by_output_index[output_index]
    if len(active_tools) == 1:
        return active_tools[0]
    return None


def _event_item_id(event: Mapping[str, Any]) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Return the event's non-empty item_id, else its item's non-empty id,
    # else None.
    item_id = event.get("item_id")
    if isinstance(item_id, str) and item_id:
        return item_id
    item = event.get("item")
    if isinstance(item, Mapping):
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id:
            return item_id
    return None


def _event_call_id(event: Mapping[str, Any]) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Return the event's non-empty call_id, else its item's non-empty
    # call_id, else None.
    call_id = event.get("call_id")
    if isinstance(call_id, str) and call_id:
        return call_id
    item = event.get("item")
    if isinstance(item, Mapping):
        call_id = item.get("call_id")
        if isinstance(call_id, str) and call_id:
            return call_id
    return None


def _event_output_index(event: Mapping[str, Any]) -> int | None:
    # [FUNCTION DOC]
    # HELPER: Return the event's output_index if it is a non-boolean integer, else
    # None.
    output_index = event.get("output_index")
    if isinstance(output_index, int) and not isinstance(output_index, bool):
        return output_index
    return None


def _text_from_done_message(item: Mapping[str, Any]) -> str:
    # [FUNCTION DOC]
    # HELPER: Concatenate output_text texts and refusal texts from a completed
    # message item's content list; returns empty string if content is not a list.
    content = item.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, Mapping):
            continue
        if part.get("type") == "output_text":
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
        elif part.get("type") == "refusal":
            refusal = part.get("refusal")
            if isinstance(refusal, str):
                parts.append(refusal)
    return "".join(parts)


def _finish_reason_from_response(event: Mapping[str, Any]) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Return the string status of the event's response object (e.g.
    # completed, incomplete), or None.
    response = event.get("response")
    if not isinstance(response, Mapping):
        return None
    status = response.get("status")
    if isinstance(status, str):
        return status
    return None


def _int_or_zero(value: object) -> int:
    # [FUNCTION DOC]
    # HELPER: Return the value if it is a non-boolean integer, else 0.
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _usage_from_response(event: Mapping[str, Any]) -> Usage | None:
    # [FUNCTION DOC]
    # HELPER: Build usage from the response's usage object: cache read = cached
    # tokens and cache write = cache write tokens from input details; fresh input =
    # input tokens minus both (floored at 0); output and total tokens; reasoning
    # tokens from output details or None when no details are given. Returns None if
    # response or usage is missing. Cost is not computed.
    """Parse billed usage from a Responses ``response.completed``-style event.

    Cache reads and writes are subtracted from ``input_tokens`` to leave fresh
    input. Cost is left unset (None) because Tau has no per-model pricing table.
    """
    response = event.get("response")
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


def _response_error_message(event: Mapping[str, Any]) -> str:
    # [FUNCTION DOC]
    # HELPER: Message for response.failed: the extracted error message, else
    # "OpenAI Codex response failed: <code>", else a generic failure message.
    code, message = _stream_error_details(event)
    if message:
        return message
    if code:
        return f"OpenAI Codex response failed: {code}"
    return "OpenAI Codex response failed"


def _error_message(event: Mapping[str, Any], *, fallback: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Message for a top-level error event: extracted message, else the code,
    # else the given fallback.
    code, message = _stream_error_details(event)
    if message:
        return message
    if code:
        return code
    return fallback


def _stream_error_details(event: Mapping[str, Any]) -> tuple[str | None, str | None]:
    # [FUNCTION DOC]
    # HELPER: Extract (code, message) from a Codex error event by checking, in
    # order, the event itself, its nested "error" object, and response.error; the
    # first non-empty string message and code win. If no code is found, falls back
    # to the nested error's "type". Either part may be None.
    """Extract the machine code and human message from a Codex stream error.

    Codex reports failures either as a top-level ``error`` SSE event with a
    nested ``error`` object (``{"type":"error","error":{"code":...}}``) or
    as ``response.failed`` with the failure under ``response.error``. Looking
    only at top-level fields hides details such as ``server_is_overloaded``
    behind a generic fallback message.
    """
    sources: list[Mapping[str, Any]] = [event]
    nested = event.get("error")
    if isinstance(nested, Mapping):
        sources.append(nested)
    response = event.get("response")
    if isinstance(response, Mapping):
        response_error = response.get("error")
        if isinstance(response_error, Mapping):
            sources.append(response_error)

    message: str | None = None
    code: str | None = None
    for source in sources:
        if message is None:
            raw_message = source.get("message")
            if isinstance(raw_message, str) and raw_message:
                message = raw_message
        if code is None:
            raw_code = source.get("code")
            if isinstance(raw_code, str) and raw_code:
                code = raw_code
    if code is None and isinstance(nested, Mapping):
        nested_type = nested.get("type")
        if isinstance(nested_type, str) and nested_type:
            code = nested_type
    return code, message


_TRANSIENT_STREAM_ERROR_MARKERS = (
    "overloaded",
    "service_unavailable",
    "temporarily_unavailable",
    "rate_limit",
    "internal_error",
    "server_error",
    "timeout",
)


def _stream_error_event_data(event: ProviderErrorEvent) -> dict[str, JSONValue] | None:
    # [FUNCTION DOC]
    # HELPER: Return the raw SSE event mapping stored under "event" in an error
    # event's data, or None.
    """Return the raw SSE event attached to a provider stream error."""
    data = event.data
    if not isinstance(data, dict):
        return None
    raw = data.get("event")
    return raw if isinstance(raw, dict) else None


def _retryable_stream_error_event(event: ProviderErrorEvent) -> bool:
    # [FUNCTION DOC]
    # HELPER: Return True when an in-stream error's code/message (lowercased) is
    # non-empty, is not a terminal quota/billing limit, and contains one of the
    # transient markers (overloaded, service_unavailable, temporarily_unavailable,
    # rate_limit, internal_error, server_error, timeout).
    """Return True when an in-stream Codex error looks transient and retryable."""
    raw = _stream_error_event_data(event)
    if raw is None:
        return False
    code, message = _stream_error_details(raw)
    haystack = " ".join(part for part in (code, message) if part).lower()
    if not haystack or _is_terminal_rate_limit(haystack):
        return False
    return any(marker in haystack for marker in _TRANSIENT_STREAM_ERROR_MARKERS)


def _build_codex_headers(
    configured_headers: Mapping[str, str] | None,
    *,
    access_token: str,
    account_id: str,
    originator: str,
    session_id: str | None = None,
) -> dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: Merge configured headers with required Codex headers (bearer auth,
    # chatgpt-account-id, originator, a "tau (<os> <release>; <arch>)" User-Agent,
    # OpenAI-Beta responses=experimental, SSE accept and JSON content type; the
    # required ones override configured values) and add session-id when given.
    # Returns a new mapping.
    headers = {
        **dict(configured_headers or {}),
        "Authorization": f"Bearer {access_token}",
        "chatgpt-account-id": account_id,
        "originator": originator,
        "User-Agent": f"tau ({system()} {release()}; {machine()})",
        "OpenAI-Beta": "responses=experimental",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }
    if session_id is not None:
        headers["session-id"] = session_id
    return headers


def _resolve_codex_url(base_url: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Normalize a base URL (trailing slashes removed) to the Codex responses
    # endpoint, appending "/responses" or "/codex/responses" as needed.
    normalized = base_url.rstrip("/")
    if normalized.endswith("/codex/responses"):
        return normalized
    if normalized.endswith("/codex"):
        return f"{normalized}/responses"
    return f"{normalized}/codex/responses"


def _resolve_codex_models_url(base_url: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Normalize a base URL to the Codex models endpoint ("/codex/models"),
    # handling bases that already end in /codex or /codex/responses.
    normalized = base_url.rstrip("/")
    if normalized.endswith("/codex/responses"):
        return f"{normalized.removesuffix('/responses')}/models"
    if normalized.endswith("/codex"):
        return f"{normalized}/models"
    return f"{normalized}/codex/models"


def _parse_codex_model_catalog(payload: object) -> RuntimeModelCatalog:
    # [FUNCTION DOC]
    # HELPER: Parse the catalog payload into a runtime model catalog containing only
    # items with a non-empty unique slug and visibility "list". Each model gets its
    # display name, limits, input modalities, thinking levels and a default level
    # (dropped if not among the supported levels). Sorted by descending integer
    # priority (default 0), then original order. Malformed payloads yield an empty
    # catalog.
    """Parse models the Codex backend marks visible to ChatGPT accounts."""
    if not isinstance(payload, Mapping):
        return RuntimeModelCatalog(())
    items = payload.get("models")
    if not isinstance(items, list):
        return RuntimeModelCatalog(())

    candidates: list[tuple[int, int, RuntimeModel]] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            continue
        model_id = item.get("slug")
        if (
            not isinstance(model_id, str)
            or not model_id
            or model_id in seen
            or item.get("visibility") != "list"
        ):
            continue
        seen.add(model_id)
        limits = _runtime_model_limits(item)
        modalities = _codex_input_modalities(item.get("input_modalities"))
        thinking_levels = _codex_thinking_levels(item.get("supported_reasoning_levels"))
        default_thinking = _codex_thinking_level(item.get("default_reasoning_level"))
        if default_thinking not in thinking_levels:
            default_thinking = None
        priority = item.get("priority")
        candidates.append(
            (
                priority if isinstance(priority, int) and not isinstance(priority, bool) else 0,
                index,
                RuntimeModel(
                    id=model_id,
                    name=(
                        item.get("display_name")
                        if isinstance(item.get("display_name"), str) and item.get("display_name")
                        else None
                    ),
                    limits=limits,
                    input_modalities=modalities,
                    thinking_levels=thinking_levels,
                    default_thinking_level=default_thinking,
                ),
            )
        )
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    return RuntimeModelCatalog(tuple(candidate[2] for candidate in candidates))


def _codex_input_modalities(value: object) -> tuple[Literal["text", "image"], ...]:
    # [FUNCTION DOC]
    # HELPER: Map input_modalities to a tuple of known modalities: missing means
    # text and image (legacy), a non-list means text only, otherwise the known
    # modalities present in the list (text first), defaulting to text only.
    # Official Codex clients treat an omitted field as legacy text+image support.
    if value is None:
        return ("text", "image")
    if not isinstance(value, list):
        return ("text",)
    known: tuple[Literal["text", "image"], ...] = ("text", "image")
    modalities = tuple(item for item in known if item in value)
    return modalities or ("text",)


def _codex_thinking_levels(value: object) -> tuple[RuntimeThinkingLevel, ...]:
    # [FUNCTION DOC]
    # HELPER: Map supported_reasoning_levels (list of objects with "effort") to a
    # deduplicated tuple of Tau thinking levels in order; non-lists yield empty.
    if not isinstance(value, list):
        return ()
    levels: list[RuntimeThinkingLevel] = []
    for item in value:
        raw = item.get("effort") if isinstance(item, Mapping) else None
        level = _codex_thinking_level(raw)
        if level is not None and level not in levels:
            levels.append(level)
    return tuple(levels)


def _codex_thinking_level(value: object) -> RuntimeThinkingLevel | None:
    # [FUNCTION DOC]
    # HELPER: Map a Codex effort string to a Tau level: "none" to "off", known
    # levels minimal..max to themselves, anything else to None.
    if value == "none":
        return "off"
    if value in {"minimal", "low", "medium", "high", "xhigh", "max"}:
        return cast(RuntimeThinkingLevel, value)
    return None


def _parse_codex_model_limits(payload: object) -> dict[str, RuntimeModelLimits]:
    # [FUNCTION DOC]
    # HELPER: Build a mapping from model slug to runtime limits for every catalog
    # item (regardless of visibility) that has a slug and usable limits; malformed
    # payloads yield an empty mapping.
    if not isinstance(payload, Mapping):
        return {}
    models = payload.get("models")
    if not isinstance(models, list):
        return {}

    parsed: dict[str, RuntimeModelLimits] = {}
    for item in models:
        if not isinstance(item, Mapping):
            continue
        model = item.get("slug")
        limits = _runtime_model_limits(item)
        if not isinstance(model, str) or not model or limits is None:
            continue
        parsed[model] = limits
    return parsed


def _runtime_model_limits(item: Mapping[object, object]) -> RuntimeModelLimits | None:
    # [FUNCTION DOC]
    # HELPER: Build runtime limits from a catalog item: context window from
    # context_window or max_context_window (positive ints); returns None if absent
    # or if the effective context percent (default 100) exceeds 100. Also carries
    # optional max output tokens and auto-compact token limit.
    context_window = _positive_int(item.get("context_window")) or _positive_int(
        item.get("max_context_window")
    )
    if context_window is None:
        return None
    effective_percent = _positive_int(item.get("effective_context_window_percent")) or 100
    if effective_percent > 100:
        return None
    return RuntimeModelLimits(
        context_window=context_window,
        max_output_tokens=_positive_int(item.get("max_output_tokens")),
        effective_context_window_percent=effective_percent,
        auto_compact_token_limit=_positive_int(item.get("auto_compact_token_limit")),
    )


def _positive_int(value: object) -> int | None:
    # [FUNCTION DOC]
    # HELPER: Return the value if it is a positive non-boolean integer, else None.
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return None
    return value


def _split_tool_call_id(value: str) -> tuple[str, str | None]:
    # [FUNCTION DOC]
    # HELPER: Split a combined "<call id>|<item id>" tool call id at the first bar;
    # returns (call id, item id or None). Ids without a bar return (id, None).
    if "|" not in value:
        return value, None
    call_id, item_id = value.split("|", 1)
    return call_id, item_id or None


def _loads_object(value: str) -> dict[str, JSONValue] | None:
    # [FUNCTION DOC]
    # HELPER: Parse a string as JSON and return it only if it is an object; returns
    # None on decode errors or non-object values.
    try:
        loaded = loads(value)
    except JSONDecodeError:
        return None
    if isinstance(loaded, dict):
        return loaded
    return None


def _is_retryable_status(status_code: int, body: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: True for 408, 409, 425, 429 and any 5xx status, except a 429 whose
    # body indicates a terminal quota/billing limit.
    if status_code == 429 and _is_terminal_rate_limit(body):
        return False
    return status_code in {408, 409, 425, 429} or status_code >= 500


def _is_terminal_rate_limit(body: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if the lowercased text contains a marker of a non-recoverable
    # usage/quota/billing limit (usage limit error names, monthly limit, available
    # balance, insufficient_quota, out of budget, quota exceeded, billing).
    normalized = body.lower()
    markers = (
        "gousagelimiterror",
        "freeusagelimiterror",
        "monthly usage limit reached",
        "available balance",
        "insufficient_quota",
        "out of budget",
        "quota exceeded",
        "billing",
    )
    return any(marker in normalized for marker in markers)
