# [MODULE DOC]
# Provider-layer tests for tau_ai: the fake provider, environment config
# parsing, and the HTTP providers (OpenAI-compatible chat completions and
# Responses API, OpenAI Codex, Google Generative AI, Anthropic, GitHub
# Copilot variants). Each test injects an httpx client with a MockTransport
# handler that records requests and returns canned SSE bodies or error
# statuses, then asserts on the request payload/headers and the canonical
# assistant event stream (start/deltas/end/done/error), retries, cancellation,
# usage accounting and diagnostics. No real network access.
import ssl
from collections.abc import AsyncIterator, Mapping
from json import loads

import httpx
import pytest

from tau_agent import (
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    ImageContent,
    SimpleCancellationToken,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from tau_agent.loop import run_agent_loop
from tau_agent.messages import assistant_content
from tau_agent.types import JSONValue
from tau_ai import (
    AnthropicConfig,
    AnthropicProvider,
    AssistantDoneEvent,
    AssistantErrorEvent,
    AssistantStartEvent,
    FakeProvider,
    GoogleGenerativeAIProvider,
    OpenAICodexConfig,
    OpenAICodexCredentials,
    OpenAICodexProvider,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    RuntimeModel,
    RuntimeModelCatalog,
    RuntimeModelLimits,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallEndEvent,
    openai_compatible_config_from_env,
)


async def _collect(stream: AsyncIterator[object]) -> list[object]:
    # [FUNCTION DOC]
    # HELPER: Drains an async iterator into a list and returns it.
    return [event async for event in stream]


def _provider_tool(
    name: str,
    description: str,
    parameters: Mapping[str, JSONValue],
) -> AgentTool:
    # [FUNCTION DOC]
    # HELPER: Builds an AgentTool with the given name (also used as label),
    # description and parameter schema whose execute function returns the
    # arguments' string form. Used to test tool schema serialisation.
    async def execute(
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: object | None = None,
        on_update: object | None = None,
    ) -> AgentToolResult:
        # [FUNCTION DOC]
        # HELPER: Fake tool execute function; ignores call id, signal and update
        # callback and returns a result whose text is the string form of the
        # arguments.
        del tool_call_id, signal, on_update
        return AgentToolResult(content=str(arguments))

    return AgentTool(
        name=name,
        label=name,
        description=description,
        parameters=parameters,
        execute_fn=execute,  # type: ignore[arg-type]
    )


@pytest.mark.anyio
async def test_fake_provider_replays_scripted_events() -> None:
    # [FUNCTION DOC]
    # Verifies FakeProvider yields the scripted events unchanged and records the
    # model and system prompt of each call.
    start = AssistantMessage(model="fake-model")
    final = AssistantMessage(content="hello", model="fake-model")
    scripted = [
        AssistantStartEvent(partial=start),
        TextDeltaEvent(content_index=0, delta="hello", partial=final),
        AssistantDoneEvent(reason="stop", message=final),
    ]
    provider = FakeProvider([scripted])

    events = await _collect(
        provider.stream_response(
            model="fake-model",
            system="system prompt",
            messages=[UserMessage(content="hi")],
            tools=[],
        )
    )

    assert events == scripted
    assert provider.calls[0][0] == "fake-model"
    assert provider.calls[0][1] == "system prompt"


def test_openai_compatible_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Verifies OPENAI_* environment variables populate the config (API key, base
    # URL with trailing slash stripped, timeout, max retries, max retry delay).
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "2")
    monkeypatch.setenv("OPENAI_MAX_RETRY_DELAY_SECONDS", "0.25")

    config = openai_compatible_config_from_env()

    assert config.api_key == "test-key"
    assert config.base_url == "https://example.test/v1"
    assert config.timeout_seconds == 12.5
    assert config.max_retries == 2
    assert config.max_retry_delay_seconds == 0.25


def test_openai_compatible_config_from_env_rejects_invalid_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC]
    # Verifies a zero timeout raises RuntimeError mentioning "greater than 0".
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "0")

    with pytest.raises(RuntimeError, match="greater than 0"):
        openai_compatible_config_from_env()


def test_openai_compatible_config_from_env_rejects_invalid_retry_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC]
    # Verifies negative max retries raises RuntimeError mentioning "0 or greater".
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "-1")

    with pytest.raises(RuntimeError, match="0 or greater"):
        openai_compatible_config_from_env()


@pytest.mark.anyio
async def test_openai_compatible_provider_uses_configured_timeout() -> None:
    # [FUNCTION DOC]
    # Verifies the lazily created HTTP client uses the configured connect/read
    # timeout and is closed afterwards.
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            api_key="test-key",
            base_url="https://example.test/v1",
            timeout_seconds=7.5,
        )
    )
    try:
        client = provider._get_client()

        assert client.timeout.connect == 7.5
        assert client.timeout.read == 7.5
    finally:
        await provider.aclose()


@pytest.mark.anyio
async def test_openai_compatible_provider_formats_request_and_streams_text() -> None:
    # [FUNCTION DOC]
    # Verifies the chat completions URL, bearer auth and custom headers, the
    # payload (model, stream, system + user messages, no cache key/affinity
    # headers/reasoning for non-OpenAI hosts) and the canonical text event
    # sequence ending in a "stop" done event with "Hello".
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns two text delta chunks then [DONE].
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
                headers={"X-HF-Bill-To": "my-org"},
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Tau.",
                messages=[UserMessage(content="Say hello")],
                tools=[],
                session_id="unsupported-session",
            )
        )

    assert [event.type for event in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "done",
    ]
    assert isinstance(events[-1], AssistantDoneEvent)
    assert events[-1].message.text == "Hello"
    assert events[-1].reason == "stop"

    request = requests[0]
    assert request.url == "https://example.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["x-hf-bill-to"] == "my-org"

    payload = loads(request.content)
    assert payload["model"] == "test-model"
    assert payload["stream"] is True
    assert "prompt_cache_key" not in payload
    assert "session_id" not in request.headers
    assert "x-client-request-id" not in request.headers
    assert "reasoning_effort" not in payload
    assert payload["messages"] == [
        {"role": "system", "content": "You are Tau."},
        {"role": "user", "content": "Say hello"},
    ]


@pytest.mark.anyio
async def test_openai_compatible_provider_uses_wire_model_alias_but_reports_logical_model() -> None:
    # [FUNCTION DOC]
    # Verifies model aliases rewrite the model sent on the wire while the final
    # message reports the logical model id.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns one "ok" chunk with stop.
        requests.append(request)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://router.huggingface.co/v1",
                model_aliases={"zai-org/GLM-5.2": "zai-org/GLM-5.2:deepinfra"},
            ),
            client=client,
        )
        events = await _collect(
            provider.stream_response(
                model="zai-org/GLM-5.2",
                system="You are Tau.",
                messages=[UserMessage(content="hello")],
                tools=[],
            )
        )

    assert loads(requests[0].content)["model"] == "zai-org/GLM-5.2:deepinfra"
    assert isinstance(events[-1], AssistantDoneEvent)
    assert events[-1].message.model == "zai-org/GLM-5.2"


@pytest.mark.anyio
async def test_openai_compatible_provider_observes_headers_after_success() -> None:
    # [FUNCTION DOC]
    # Verifies the response headers observer is invoked only for the successful
    # retry (not the failed 500 attempt) and both attempts use the aliased model.
    observed: list[dict[str, str]] = []
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: first call returns 500 with a provider header, later calls a
        # successful stream with x-inference-provider "deepinfra".
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                500,
                headers={"x-inference-provider": "failed-provider"},
            )
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n',
            headers={
                "content-type": "text/event-stream",
                "x-inference-provider": "deepinfra",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://router.huggingface.co/v1",
                max_retries=1,
                max_retry_delay_seconds=0,
                model_aliases={"zai-org/GLM-5.2": "zai-org/GLM-5.2:deepinfra"},
                response_headers_observer=lambda headers: observed.append(dict(headers)),
            ),
            client=client,
        )
        await _collect(
            provider.stream_response(
                model="zai-org/GLM-5.2",
                system="You are Tau.",
                messages=[UserMessage(content="hello")],
                tools=[],
            )
        )

    assert [loads(request.content)["model"] for request in requests] == [
        "zai-org/GLM-5.2:deepinfra",
        "zai-org/GLM-5.2:deepinfra",
    ]
    assert len(observed) == 1
    assert observed[0]["x-inference-provider"] == "deepinfra"


@pytest.mark.anyio
async def test_openai_compatible_provider_observer_failure_keeps_completed_response() -> None:
    # [FUNCTION DOC]
    # Verifies an exception raised by the headers observer does not fail the
    # response; it is recorded as a "response_headers_observer_error" diagnostic.
    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns a successful "ok" stream with a provider header.
        del request
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n',
            headers={
                "content-type": "text/event-stream",
                "x-inference-provider": "deepinfra",
            },
        )

    def fail_observer(headers: Mapping[str, str]) -> None:
        # [FUNCTION DOC]
        # HELPER: Headers observer that always raises PermissionError.
        del headers
        raise PermissionError("session index is read-only")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://router.huggingface.co/v1",
                response_headers_observer=fail_observer,
            ),
            client=client,
        )
        events = await _collect(
            provider.stream_response(
                model="zai-org/GLM-5.2",
                system="You are Tau.",
                messages=[UserMessage(content="hello")],
                tools=[],
            )
        )

    assert isinstance(events[-1], AssistantDoneEvent)
    assert events[-1].message.text == "ok"
    assert events[-1].message.diagnostics[-1].type == "response_headers_observer_error"
    assert events[-1].message.diagnostics[-1].details == {
        "error": "session index is read-only",
        "error_type": "PermissionError",
    }


@pytest.mark.anyio
async def test_openai_compatible_provider_does_not_retry_after_partial_output() -> None:
    # [FUNCTION DOC]
    # Verifies a read error after partial content is not retried and ends with
    # an error event preserving the partial text.
    requests: list[httpx.Request] = []

    class FailingStream(httpx.AsyncByteStream):
        # [CLASS DOC]
        # HELPER: Async byte stream yielding one content chunk then raising
        # httpx.ReadError.
        async def __aiter__(self) -> AsyncIterator[bytes]:
            # [FUNCTION DOC]
            # HELPER: Yields one SSE chunk with "partial", then raises ReadError.
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise httpx.ReadError("stream dropped")

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns a 200 with the failing stream.
        requests.append(request)
        return httpx.Response(
            200,
            stream=FailingStream(),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://router.huggingface.co/v1",
                max_retries=2,
                max_retry_delay_seconds=0,
                model_aliases={"zai-org/GLM-5.2": "zai-org/GLM-5.2:deepinfra"},
            ),
            client=client,
        )
        events = await _collect(
            provider.stream_response(
                model="zai-org/GLM-5.2",
                system="You are Tau.",
                messages=[UserMessage(content="hello")],
                tools=[],
            )
        )

    assert len(requests) == 1
    assert isinstance(events[-1], AssistantErrorEvent)
    assert events[-1].error.text == "partial"


@pytest.mark.anyio
async def test_openai_compatible_provider_reports_raw_ssl_error_after_partial_output() -> None:
    # [FUNCTION DOC]
    # Verifies a raw ssl.SSLError after partial content is surfaced as an error
    # event (no retry) keeping the partial text.
    requests: list[httpx.Request] = []

    class FailingStream(httpx.AsyncByteStream):
        # [CLASS DOC]
        # HELPER: Async byte stream yielding one chunk then raising ssl.SSLError.
        async def __aiter__(self) -> AsyncIterator[bytes]:
            # [FUNCTION DOC]
            # HELPER: Yields one SSE chunk with "partial", then raises a bad-record-MAC
            # SSLError.
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise ssl.SSLError(1, "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] bad record mac")

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns a 200 with the failing stream.
        requests.append(request)
        return httpx.Response(
            200,
            stream=FailingStream(),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://router.huggingface.co/v1",
                max_retries=2,
                max_retry_delay_seconds=0,
            ),
            client=client,
        )
        events = await _collect(
            provider.stream_response(
                model="zai-org/GLM-5.2",
                system="You are Tau.",
                messages=[UserMessage(content="hello")],
                tools=[],
            )
        )

    assert len(requests) == 1
    assert isinstance(events[-1], AssistantErrorEvent)
    assert events[-1].error.text == "partial"


@pytest.mark.anyio
async def test_openai_chat_completions_sends_prompt_cache_key_without_affinity_headers() -> None:
    # [FUNCTION DOC]
    # Verifies api.openai.com chat requests send the session id as
    # prompt_cache_key but no session_id / x-client-request-id headers.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns an "ok" stream.
        requests.append(request)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
            ),
            client=client,
        )
        await _collect(
            provider.stream_response(
                model="gpt-4.1",
                system="You are Tau.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
                session_id="chat-session",
            )
        )

    assert loads(requests[0].content)["prompt_cache_key"] == "chat-session"
    assert "session_id" not in requests[0].headers
    assert "x-client-request-id" not in requests[0].headers


@pytest.mark.anyio
async def test_openai_compatible_provider_includes_configured_reasoning_effort() -> None:
    # [FUNCTION DOC]
    # Verifies a configured reasoning effort is sent as reasoning_effort.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns an "ok" stream.
        requests.append(request)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
                reasoning_effort="high",
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Tau.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert isinstance(events[-1], AssistantDoneEvent)
    assert loads(requests[0].content)["reasoning_effort"] == "high"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("reasoning_effort", "supports_reasoning_effort", "expected_thinking", "expected_effort"),
    [
        ("high", False, {"type": "enabled"}, None),
        ("none", False, {"type": "disabled"}, None),
        ("high", True, {"type": "enabled"}, "high"),
    ],
)
async def test_zai_provider_serializes_thinking_protocol(
    reasoning_effort: str,
    supports_reasoning_effort: bool,
    expected_thinking: dict[str, str],
    expected_effort: str | None,
) -> None:
    # [FUNCTION DOC]
    # Parametrized: with the "zai" thinking format the payload has a thinking
    # object enabled/disabled by effort, reasoning_effort only when the compat
    # flag allows it, and never enable_thinking.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns an "ok" stream.
        requests.append(request)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://api.z.ai/api/paas/v4",
                reasoning_effort=reasoning_effort,
                thinking_format="zai",
                compat={"supportsReasoningEffort": supports_reasoning_effort},
            ),
            client=client,
        )
        await _collect(
            provider.stream_response(
                model="glm-5.1",
                system="You are Tau.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    payload = loads(requests[0].content)
    assert payload["thinking"] == expected_thinking
    assert payload.get("reasoning_effort") == expected_effort
    assert "enable_thinking" not in payload


@pytest.mark.anyio
async def test_unsupported_reasoning_effort_guard_remains_for_openai_format() -> None:
    # [FUNCTION DOC]
    # Verifies supportsReasoningEffort false suppresses reasoning_effort.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns an "ok" stream.
        requests.append(request)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
                reasoning_effort="high",
                compat={"supportsReasoningEffort": False},
            ),
            client=client,
        )
        await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Tau.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert "reasoning_effort" not in loads(requests[0].content)


@pytest.mark.anyio
async def test_openai_compatible_provider_includes_openrouter_provider_routing() -> None:
    # [FUNCTION DOC]
    # Verifies the openrouterProvider compat value is sent as "provider".
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns an "ok" stream then [DONE].
        requests.append(request)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                compat={"openrouterProvider": {"ignore": ["Nebius"]}},
            ),
            client=client,
        )

        await _collect(
            provider.stream_response(
                model="nvidia/nemotron-3-super-120b-a12b",
                system="You are Tau.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert loads(requests[0].content)["provider"] == {"ignore": ["Nebius"]}


@pytest.mark.anyio
async def test_openai_compatible_provider_supports_nested_reasoning_effort_parameter() -> None:
    # [FUNCTION DOC]
    # Verifies a "reasoning.effort" parameter name produces a nested reasoning
    # object on the chat completions endpoint instead of reasoning_effort.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns an "ok" stream then [DONE].
        requests.append(request)
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
                reasoning_effort="high",
                reasoning_effort_parameter="reasoning.effort",
            ),
            client=client,
        )

        # A model served over /chat/completions (not gpt-5.5/5.4/codex, which
        # route to /v1/responses) exercises the nested reasoning.effort payload.
        await _collect(
            provider.stream_response(
                model="custom-reasoner",
                system="You are Tau.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert requests[0].url == "https://example.test/v1/chat/completions"
    assert loads(requests[0].content)["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in loads(requests[0].content)


@pytest.mark.anyio
async def test_google_provider_sends_system_instruction_at_top_level() -> None:
    # [FUNCTION DOC]
    # Verifies Google requests put systemInstruction at top level and map low
    # effort to a thinkingConfig with includeThoughts and a 2048 token budget.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns a Google candidate "ok" with STOP.
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},'
                '"finishReason":"STOP"}]}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = GoogleGenerativeAIProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                reasoning_effort="low",
            ),
            client=client,
        )

        await _collect(
            provider.stream_response(
                model="gemini-2.5-flash",
                system="You are Tau.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    payload = loads(requests[0].content)
    assert payload["systemInstruction"] == {"parts": [{"text": "You are Tau."}]}
    assert "systemInstruction" not in payload["generationConfig"]
    assert payload["generationConfig"]["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingBudget": 2048,
    }


@pytest.mark.anyio
async def test_google_provider_round_trips_thought_signature() -> None:
    # [FUNCTION DOC]
    # Verifies a functionCall thoughtSignature is kept on the ToolCall and sent
    # back on the replayed function call part in the next request.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records each request and returns a functionCall part with a thought
        # signature and STOP.
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"candidates":[{"content":{"parts":[{"functionCall":'
                '{"id":"call-1","name":"bash","args":{"command":"ls"}},'
                '"thoughtSignature":"sig-123"}]},"finishReason":"STOP"}]}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = GoogleGenerativeAIProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://generativelanguage.googleapis.com/v1beta",
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gemini-2.5-flash",
                system="",
                messages=[UserMessage(content="List files")],
                tools=[],
            )
        )

        assistant_message = events[-1].message
        assert assistant_message.tool_calls[0].thought_signature == "sig-123"

        await _collect(
            provider.stream_response(
                model="gemini-2.5-flash",
                system="",
                messages=[
                    UserMessage(content="List files"),
                    assistant_message,
                    ToolResultMessage(
                        tool_call_id="call-1", tool_name="bash", content=[TextContent(text="a.py")]
                    ),
                ],
                tools=[],
            )
        )

    second_payload = loads(requests[1].content)
    tool_call_part = second_payload["contents"][1]["parts"][0]
    assert tool_call_part["thoughtSignature"] == "sig-123"


@pytest.mark.anyio
async def test_google_provider_strips_unsupported_schema_keywords_from_tools() -> None:
    # [FUNCTION DOC]
    # Verifies additionalProperties is removed recursively from tool parameter
    # schemas sent to Google while other keywords are kept.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns an "ok" STOP candidate.
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},'
                '"finishReason":"STOP"}]}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    tool = _provider_tool(
        "bash",
        "Run a shell command.",
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "command": {"type": "string"},
                "env": {
                    "type": "array",
                    "items": {"type": "string", "additionalProperties": False},
                },
            },
        },
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = GoogleGenerativeAIProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://generativelanguage.googleapis.com/v1beta",
            ),
            client=client,
        )

        await _collect(
            provider.stream_response(
                model="gemini-2.5-flash",
                system="",
                messages=[UserMessage(content="List files")],
                tools=[tool],
            )
        )

    payload = loads(requests[0].content)
    parameters = payload["tools"][0]["functionDeclarations"][0]["parameters"]
    assert "additionalProperties" not in parameters
    assert "additionalProperties" not in parameters["properties"]["env"]["items"]
    assert parameters["properties"]["command"] == {"type": "string"}


@pytest.mark.anyio
async def test_google_provider_errors_when_stream_ends_during_thinking() -> None:
    # [FUNCTION DOC]
    # Verifies a stream ending without finishReason after thought content is not
    # retried and yields an error keeping the thinking text.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns only a thought part, no finishReason.
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"candidates":[{"content":{"parts":['
                '{"text":"partial thought","thought":true}]}}]}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = GoogleGenerativeAIProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                max_retries=2,
                max_retry_delay_seconds=0,
            ),
            client=client,
        )
        events = await _collect(
            provider.stream_response(
                model="gemini-2.5-flash",
                system="",
                messages=[UserMessage(content="Think")],
                tools=[],
            )
        )

    assert len(requests) == 1
    assert isinstance(events[-1], AssistantErrorEvent)
    assert events[-1].error.thinking_text == "partial thought"
    assert events[-1].error.error_message == "Google stream ended without finishReason"


@pytest.mark.anyio
async def test_google_provider_does_not_execute_tool_from_incomplete_stream() -> None:
    # [FUNCTION DOC]
    # Runs the full agent loop: a function call in a stream without finishReason
    # must produce an error assistant message and the tool must not execute.
    executed = False

    async def execute(
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: object | None = None,
        on_update: object | None = None,
    ) -> AgentToolResult:
        # [FUNCTION DOC]
        # HELPER: Fake tool execute that flags (via a closure variable) that it ran.
        nonlocal executed
        del tool_call_id, arguments, signal, on_update
        executed = True
        return AgentToolResult(content="unexpected")

    tool = AgentTool(
        name="read",
        label="read",
        description="Read a file.",
        parameters={"type": "object"},
        execute_fn=execute,  # type: ignore[arg-type]
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns text plus a functionCall part without finishReason.
        return httpx.Response(
            200,
            text=(
                'data: {"candidates":[{"content":{"parts":['
                '{"text":"I will inspect it."},'
                '{"functionCall":{"id":"call-1","name":"read",'
                '"args":{"path":"README.md"}}}]}}]}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    messages = [UserMessage(content="Read README.md")]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = GoogleGenerativeAIProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://generativelanguage.googleapis.com/v1beta",
            ),
            client=client,
        )
        await _collect(
            run_agent_loop(
                provider=provider,
                model="gemini-2.5-flash",
                system="",
                messages=messages,
                tools=[tool],
            )
        )

    assert executed is False
    assert isinstance(messages[-1], AssistantMessage)
    assert messages[-1].stop_reason == "error"
    assert messages[-1].text == "I will inspect it."
    assert [call.name for call in messages[-1].tool_calls] == ["read"]


@pytest.mark.anyio
async def test_google_provider_retries_empty_clean_close_then_errors() -> None:
    # [FUNCTION DOC]
    # Verifies empty streams are retried up to max retries (3 requests) and then
    # fail with "Google stream ended without finishReason".
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns an empty 200 event stream.
        requests.append(request)
        return httpx.Response(
            200,
            text="",
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = GoogleGenerativeAIProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                max_retries=2,
                max_retry_delay_seconds=0,
            ),
            client=client,
        )
        events = await _collect(
            provider.stream_response(
                model="gemini-2.5-flash",
                system="",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert len(requests) == 3
    assert isinstance(events[-1], AssistantErrorEvent)
    assert events[-1].error.error_message == "Google stream ended without finishReason"


@pytest.mark.anyio
async def test_google_provider_accepts_explicit_stop_finish_reason() -> None:
    # [FUNCTION DOC]
    # Verifies STOP maps to a "stop" done event with the text.
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns an "ok" candidate with STOP.
        return httpx.Response(
            200,
            text=(
                'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},'
                '"finishReason":"STOP"}]}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = GoogleGenerativeAIProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://generativelanguage.googleapis.com/v1beta",
            ),
            client=client,
        )
        events = await _collect(
            provider.stream_response(
                model="gemini-2.5-flash",
                system="",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert isinstance(events[-1], AssistantDoneEvent)
    assert events[-1].reason == "stop"
    assert events[-1].message.text == "ok"


@pytest.mark.anyio
async def test_google_provider_maps_max_tokens_finish_reason_to_length() -> None:
    # [FUNCTION DOC]
    # Verifies MAX_TOKENS maps to reason and stop reason "length".
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns a "partial" candidate with MAX_TOKENS.
        return httpx.Response(
            200,
            text=(
                'data: {"candidates":[{"content":{"parts":[{"text":"partial"}]},'
                '"finishReason":"MAX_TOKENS"}]}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = GoogleGenerativeAIProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://generativelanguage.googleapis.com/v1beta",
            ),
            client=client,
        )
        events = await _collect(
            provider.stream_response(
                model="gemini-2.5-flash",
                system="",
                messages=[UserMessage(content="Write a lot")],
                tools=[],
            )
        )

    assert isinstance(events[-1], AssistantDoneEvent)
    assert events[-1].reason == "length"
    assert events[-1].message.stop_reason == "length"


@pytest.mark.anyio
async def test_google_provider_errors_on_truncated_json_chunk() -> None:
    # [FUNCTION DOC]
    # Verifies an invalid JSON chunk produces an error without retry.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns a truncated JSON data line.
        requests.append(request)
        return httpx.Response(
            200,
            text='data: {"candidates":[\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = GoogleGenerativeAIProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://generativelanguage.googleapis.com/v1beta",
                max_retries=2,
                max_retry_delay_seconds=0,
            ),
            client=client,
        )
        events = await _collect(
            provider.stream_response(
                model="gemini-2.5-flash",
                system="",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert len(requests) == 1
    assert isinstance(events[-1], AssistantErrorEvent)
    assert events[-1].error.error_message == "Google returned an invalid JSON stream chunk"


@pytest.mark.anyio
async def test_openai_compatible_provider_streams_reasoning_content() -> None:
    # [FUNCTION DOC]
    # Verifies reasoning_content deltas become thinking events/blocks (signature
    # "reasoning_content") ordered before text in the final message.
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns two reasoning chunks, a content chunk and [DONE].
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"reasoning_content":"plan "}}]}\n\n'
                'data: {"choices":[{"delta":{"reasoning_content":"steps"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="test-key", base_url="https://example.test/v1"),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Tau.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert [event.type for event in events] == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_delta",
        "text_start",
        "text_delta",
        "thinking_end",
        "text_end",
        "done",
    ]
    thinking_events = [event for event in events if isinstance(event, ThinkingDeltaEvent)]
    assert [event.delta for event in thinking_events] == ["plan ", "steps"]
    assert isinstance(events[-1], AssistantDoneEvent)
    assert events[-1].message.text == "done"
    assert [block.type for block in events[-1].message.content] == ["thinking", "text"]
    assert events[-1].message.thinking_text == "plan steps"
    thinking = events[-1].message.content[0]
    assert isinstance(thinking, ThinkingContent)
    assert thinking.thinking_signature == "reasoning_content"


@pytest.mark.anyio
async def test_openai_compatible_provider_replays_persisted_reasoning() -> None:
    # [FUNCTION DOC]
    # Verifies prior assistant thinking with the reasoning_content signature is
    # replayed as reasoning_content on the assistant message.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns a "next" stream.
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"next"},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    prior = AssistantMessage(
        content=[
            ThinkingContent(thinking="prior plan", thinking_signature="reasoning_content"),
            TextContent(text="prior answer"),
        ]
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="test-key", base_url="https://example.test/v1"),
            client=client,
        )
        await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Tau.",
                messages=[UserMessage(content="first"), prior],
                tools=[],
            )
        )

    payload = loads(requests[0].content)
    replay = payload["messages"][-1]
    assert replay["content"] == "prior answer"
    assert replay["reasoning_content"] == "prior plan"


@pytest.mark.anyio
async def test_openai_compatible_provider_streams_tool_calls() -> None:
    # [FUNCTION DOC]
    # Verifies tool definitions are serialised as functions and fragmented tool
    # call arguments are assembled into one ToolCall with a "toolUse" done reason.
    tool = _provider_tool(
        "read",
        "Read a file.",
        {"type": "object", "properties": {"path": {"type": "string"}}},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: asserts the tools payload, then returns two tool call delta chunks.
        payload = loads(request.content)
        assert payload["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file.",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ]
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1",'
                '"function":{"name":"read","arguments":"{\\"path\\":"}}]}}]}\n\n'
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"\\"README.md\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="test-key", base_url="https://example.test/v1"),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Tau.",
                messages=[UserMessage(content="Read README.md")],
                tools=[tool],
            )
        )

    tool_call_events = [event for event in events if isinstance(event, ToolCallEndEvent)]

    assert [event.tool_call for event in tool_call_events] == [
        ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
    ]
    assert isinstance(events[-1], AssistantDoneEvent)
    assert events[-1].message.tool_calls == (
        ToolCall(id="call-1", name="read", arguments={"path": "README.md"}),
    )
    assert events[-1].reason == "toolUse"


@pytest.mark.anyio
async def test_openai_compatible_provider_reports_resolved_response_provider() -> None:
    # [FUNCTION DOC]
    # Verifies the response provider header from the successful retry (after a
    # 503) is reported on the start partial and final message, with provider name.
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: first call returns 503, later calls an "ok" stream with a different
        # x-inference-provider header (counts attempts via a closure variable).
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                503,
                text="temporarily unavailable",
                headers={"x-inference-provider": "provider-before-failover"},
            )
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={
                "content-type": "text/event-stream",
                "x-inference-provider": "provider-after-failover",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://router.huggingface.co/v1",
                provider_name="huggingface",
                response_provider_header="x-inference-provider",
                max_retries=1,
                max_retry_delay_seconds=0,
            ),
            client=client,
        )
        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Tau.",
                messages=[UserMessage(content="hello")],
                tools=[],
            )
        )

    assert attempts == 2
    assert isinstance(events[0], AssistantStartEvent)
    assert events[0].partial.response_provider == "provider-after-failover"
    assert isinstance(events[-1], AssistantDoneEvent)
    assert events[-1].message.provider == "huggingface"
    assert events[-1].message.response_provider == "provider-after-failover"


@pytest.mark.anyio
async def test_openai_compatible_provider_retries_transient_status() -> None:
    # [FUNCTION DOC]
    # Verifies a 500 is retried once and the second attempt's text stream is
    # emitted without extra events.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: first call returns 500, later calls an "ok" stream.
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(500, text="try again")
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
                max_retries=1,
                max_retry_delay_seconds=0,
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Tau.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert len(requests) == 2
    assert [event.type for event in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_end",
        "done",
    ]


@pytest.mark.anyio
async def test_openai_compatible_provider_cancellation_stops_retry_backoff() -> None:
    # [FUNCTION DOC]
    # Verifies a pre-cancelled signal stops after the first 503 instead of
    # waiting for backoff, yielding start then error.
    requests: list[httpx.Request] = []
    signal = SimpleCancellationToken()

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and always returns 503.
        requests.append(request)
        return httpx.Response(503, text="try later")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
                max_retries=2,
                max_retry_delay_seconds=1,
            ),
            client=client,
        )

        signal.cancel()
        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Tau.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
                signal=signal,
            )
        )

    assert len(requests) == 1
    assert [event.type for event in events] == ["start", "error"]
    assert isinstance(events[-1], AssistantErrorEvent)
    assert events[-1].reason == "error"


@pytest.mark.anyio
async def test_openai_compatible_provider_does_not_retry_non_transient_status() -> None:
    # [FUNCTION DOC]
    # Verifies a 400 is not retried; the error message includes provider, status,
    # model and the JSON error message; diagnostics include body and attempts.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns 400 with a JSON error body.
        requests.append(request)
        return httpx.Response(
            400,
            json={"error": {"message": "The selected model is unavailable."}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                provider_name="test-openai",
                base_url="https://example.test/v1",
                max_retries=3,
                max_retry_delay_seconds=0,
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Tau.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert len(requests) == 1
    assert isinstance(events[-1], AssistantErrorEvent)
    assert events[-1].error.error_message == (
        "test-openai request failed with status 400 for model test-model: "
        "The selected model is unavailable."
    )
    assert events[-1].error.diagnostics[0].details == {
        "status_code": 400,
        "body": '{"error":{"message":"The selected model is unavailable."}}',
        "attempts": 1,
    }


@pytest.mark.anyio
async def test_openai_compatible_provider_includes_plain_http_error_body_in_message() -> None:
    # [FUNCTION DOC]
    # Verifies a plain-text 400 body is included in the error message and
    # diagnostics.
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns 400 with plain text.
        return httpx.Response(400, text="bad request details")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                provider_name="test-openai",
                base_url="https://example.test/v1",
                max_retries=0,
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Tau.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert isinstance(events[-1], AssistantErrorEvent)
    assert events[-1].error.error_message == (
        "test-openai request failed with status 400 for model test-model: bad request details"
    )
    assert events[-1].error.diagnostics[0].details == {
        "status_code": 400,
        "body": "bad request details",
        "attempts": 1,
    }


@pytest.mark.anyio
async def test_openai_codex_provider_discovers_and_caches_live_model_catalog() -> None:
    # [FUNCTION DOC]
    # Verifies Codex model discovery: GET codex/models with client_version query,
    # bearer token, account id and JSON accept headers; listed models (including
    # ones not supported in the API) become RuntimeModels with limits, input
    # modalities (default text+image), thinking levels and default level; hidden
    # and invalid rows are dropped; the catalog is cached (one request) and
    # discover_model_limits reads from it.
    requests: list[httpx.Request] = []

    async def credentials() -> OpenAICodexCredentials:
        # [FUNCTION DOC]
        # HELPER: Async Codex credential resolver returning a fixed access token and account id.
        return OpenAICodexCredentials(access_token="access-token", account_id="account-1")

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns a JSON model list with listed,
        # subscription-only, hidden and invalid rows.
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "slug": "gpt-5.6-sol",
                        "display_name": "GPT-5.6 Sol",
                        "visibility": "list",
                        "supported_in_api": True,
                        "priority": 10,
                        "context_window": 372_000,
                        "max_context_window": 372_000,
                        "effective_context_window_percent": 95,
                        "auto_compact_token_limit": 330_000,
                        "max_output_tokens": 128_000,
                        "input_modalities": ["text", "image"],
                        "default_reasoning_level": "high",
                        "supported_reasoning_levels": [
                            {"effort": "low", "description": "Fast"},
                            {"effort": "high", "description": "Deep"},
                        ],
                    },
                    {
                        "slug": "subscription-only",
                        "display_name": "Subscription only",
                        "visibility": "list",
                        "supported_in_api": False,
                        "priority": 5,
                        "supported_reasoning_levels": [],
                    },
                    {
                        "slug": "hidden",
                        "visibility": "hide",
                        "supported_in_api": True,
                        "context_window": 100_000,
                    },
                    {"slug": "invalid", "context_window": -1},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(
            OpenAICodexConfig(
                credential_resolver=credentials,
                base_url="https://chatgpt.test/backend-api",
                client_version="0.2.0",
            ),
            client=client,
        )

        catalog = await provider.discover_models()
        cached_catalog = await provider.discover_models()
        limits = await provider.discover_model_limits("gpt-5.6-sol")

    expected_limits = RuntimeModelLimits(
        context_window=372_000,
        max_output_tokens=128_000,
        effective_context_window_percent=95,
        auto_compact_token_limit=330_000,
    )
    assert catalog == RuntimeModelCatalog(
        (
            RuntimeModel(
                id="gpt-5.6-sol",
                name="GPT-5.6 Sol",
                limits=expected_limits,
                input_modalities=("text", "image"),
                thinking_levels=("low", "high"),
                default_thinking_level="high",
            ),
            RuntimeModel(
                id="subscription-only",
                name="Subscription only",
                input_modalities=("text", "image"),
            ),
        )
    )
    assert cached_catalog == catalog
    assert limits == expected_limits
    assert limits == RuntimeModelLimits(
        context_window=372_000,
        max_output_tokens=128_000,
        effective_context_window_percent=95,
        auto_compact_token_limit=330_000,
    )
    assert len(requests) == 1
    assert str(requests[0].url) == (
        "https://chatgpt.test/backend-api/codex/models?client_version=0.2.0"
    )
    assert requests[0].headers["authorization"] == "Bearer access-token"
    assert requests[0].headers["chatgpt-account-id"] == "account-1"
    assert requests[0].headers["accept"] == "application/json"


@pytest.mark.anyio
async def test_openai_codex_provider_uses_resolved_latest_client_version() -> None:
    # [FUNCTION DOC]
    # Verifies a client version resolver overrides the static version, is called
    # once across discovery and limit lookups, and its value is sent.
    requests: list[httpx.Request] = []
    version_calls = 0

    async def credentials() -> OpenAICodexCredentials:
        # [FUNCTION DOC]
        # HELPER: Async Codex credential resolver returning a fixed access token and account id.
        return OpenAICodexCredentials(access_token="access-token", account_id="account-1")

    async def client_version() -> str:
        # [FUNCTION DOC]
        # HELPER: Async version resolver returning "0.153.4" and counting calls.
        nonlocal version_calls
        version_calls += 1
        return "0.153.4"

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns a one-model JSON list.
        requests.append(request)
        return httpx.Response(
            200,
            json={"models": [{"slug": "new-model", "visibility": "list"}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(
            OpenAICodexConfig(
                credential_resolver=credentials,
                base_url="https://chatgpt.test/backend-api",
                client_version="0.144.3",
                client_version_resolver=client_version,
            ),
            client=client,
        )
        catalog = await provider.discover_models()
        await provider.discover_model_limits("new-model")

    assert [model.id for model in catalog.models] == ["new-model"]
    assert version_calls == 1
    assert str(requests[0].url) == (
        "https://chatgpt.test/backend-api/codex/models?client_version=0.153.4"
    )


@pytest.mark.anyio
async def test_openai_codex_provider_includes_http_error_detail_in_message() -> None:
    # [FUNCTION DOC]
    # Verifies a Codex 400 JSON error becomes a friendly message with provider,
    # status, model and detail, plus diagnostics with body and attempts.
    async def credentials() -> OpenAICodexCredentials:
        # [FUNCTION DOC]
        # HELPER: Async Codex credential resolver returning a fixed access token and account id.
        return OpenAICodexCredentials(access_token="access-token", account_id="account-1")

    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns 400 with a JSON error body.
        return httpx.Response(
            400,
            json={"error": {"message": "The requested model does not exist."}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(
            OpenAICodexConfig(
                credential_resolver=credentials,
                base_url="https://chatgpt.test/backend-api",
                provider_name="openai-codex",
                max_retries=0,
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="Say hello")],
                tools=[],
            )
        )

    assert isinstance(events[-1], AssistantErrorEvent)
    assert events[-1].error.error_message == (
        "openai-codex request failed with status 400 for model gpt-5.5: "
        "The requested model does not exist."
    )
    assert events[-1].error.diagnostics[0].details == {
        "status_code": 400,
        "body": '{"error":{"message":"The requested model does not exist."}}',
        "attempts": 1,
    }


@pytest.mark.anyio
async def test_openai_codex_provider_includes_plain_http_error_body_in_message() -> None:
    # [FUNCTION DOC]
    # Verifies a plain-text Codex 400 body appears in the message and diagnostics.
    async def credentials() -> OpenAICodexCredentials:
        # [FUNCTION DOC]
        # HELPER: Async Codex credential resolver returning a fixed access token and account id.
        return OpenAICodexCredentials(access_token="access-token", account_id="account-1")

    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns 400 with plain text.
        return httpx.Response(400, text="bad request details")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(
            OpenAICodexConfig(
                credential_resolver=credentials,
                base_url="https://chatgpt.test/backend-api",
                provider_name="openai-codex",
                max_retries=0,
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="Say hello")],
                tools=[],
            )
        )

    assert isinstance(events[-1], AssistantErrorEvent)
    assert events[-1].error.error_message == (
        "openai-codex request failed with status 400 for model gpt-5.5: bad request details"
    )
    assert events[-1].error.diagnostics[0].details == {
        "status_code": 400,
        "body": "bad request details",
        "attempts": 1,
    }


_CODEX_OVERLOAD_ERROR_SSE = (
    'data: {"type":"error","error":{"type":"service_unavailable_error",'
    '"code":"server_is_overloaded","message":"Our servers are currently '
    'overloaded. Please try again later.","param":null},"sequence_number":2}\n\n'
)

_CODEX_TEXT_SSE = (
    'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
    'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
)


@pytest.mark.anyio
async def test_openai_codex_provider_surfaces_nested_stream_error_message() -> None:
    # [FUNCTION DOC]
    # Verifies an in-stream "error" event surfaces its nested error message and
    # the raw event in diagnostics (no retries configured).
    async def credentials() -> OpenAICodexCredentials:
        # [FUNCTION DOC]
        # HELPER: Async Codex credential resolver returning a fixed access token and account id.
        return OpenAICodexCredentials(access_token="access-token", account_id="account-1")

    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns the shared overload error SSE body.
        return httpx.Response(
            200,
            text=_CODEX_OVERLOAD_ERROR_SSE,
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(
            OpenAICodexConfig(
                credential_resolver=credentials,
                base_url="https://chatgpt.test/backend-api",
                provider_name="openai-codex",
                max_retries=0,
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="Say hello")],
                tools=[],
            )
        )

    assert isinstance(events[-1], AssistantErrorEvent)
    assert events[-1].error.error_message == (
        "Our servers are currently overloaded. Please try again later."
    )
    assert events[-1].error.diagnostics[0].details == {
        "event": {
            "type": "error",
            "error": {
                "type": "service_unavailable_error",
                "code": "server_is_overloaded",
                "message": "Our servers are currently overloaded. Please try again later.",
                "param": None,
            },
            "sequence_number": 2,
        }
    }


@pytest.mark.anyio
async def test_openai_codex_provider_retries_transient_stream_error() -> None:
    # [FUNCTION DOC]
    # Verifies an overload stream error is retried and the successful second
    # stream is emitted cleanly.
    requests: list[httpx.Request] = []

    async def credentials() -> OpenAICodexCredentials:
        # [FUNCTION DOC]
        # HELPER: Async Codex credential resolver returning a fixed access token and account id.
        return OpenAICodexCredentials(access_token="access-token", account_id="account-1")

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: first call returns the overload error SSE, later calls the "ok" text SSE.
        requests.append(request)
        body = _CODEX_OVERLOAD_ERROR_SSE if len(requests) == 1 else _CODEX_TEXT_SSE
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(
            OpenAICodexConfig(
                credential_resolver=credentials,
                base_url="https://chatgpt.test/backend-api",
                provider_name="openai-codex",
                max_retries=1,
                max_retry_delay_seconds=0,
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert len(requests) == 2
    assert [event.type for event in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_end",
        "done",
    ]


@pytest.mark.anyio
async def test_openai_codex_provider_retries_transient_response_failed() -> None:
    # [FUNCTION DOC]
    # Verifies a response.failed event with an overload code is retried.
    requests: list[httpx.Request] = []

    async def credentials() -> OpenAICodexCredentials:
        # [FUNCTION DOC]
        # HELPER: Async Codex credential resolver returning a fixed access token and account id.
        return OpenAICodexCredentials(access_token="access-token", account_id="account-1")

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: first call returns response.failed (overloaded), later calls "ok" text.
        requests.append(request)
        body = (
            'data: {"type":"response.failed","response":{"status":"failed",'
            '"error":{"code":"server_is_overloaded",'
            '"message":"The server is overloaded."}}}\n\n'
            if len(requests) == 1
            else _CODEX_TEXT_SSE
        )
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(
            OpenAICodexConfig(
                credential_resolver=credentials,
                base_url="https://chatgpt.test/backend-api",
                provider_name="openai-codex",
                max_retries=1,
                max_retry_delay_seconds=0,
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert len(requests) == 2
    assert [event.type for event in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_end",
        "done",
    ]


@pytest.mark.anyio
async def test_openai_codex_provider_surfaces_stream_error_after_retry_exhaustion() -> None:
    # [FUNCTION DOC]
    # Verifies after exhausting retries the overload message is surfaced.
    requests: list[httpx.Request] = []

    async def credentials() -> OpenAICodexCredentials:
        # [FUNCTION DOC]
        # HELPER: Async Codex credential resolver returning a fixed access token and account id.
        return OpenAICodexCredentials(access_token="access-token", account_id="account-1")

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records requests and always returns the overload error SSE.
        requests.append(request)
        return httpx.Response(
            200,
            text=_CODEX_OVERLOAD_ERROR_SSE,
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(
            OpenAICodexConfig(
                credential_resolver=credentials,
                base_url="https://chatgpt.test/backend-api",
                provider_name="openai-codex",
                max_retries=1,
                max_retry_delay_seconds=0,
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="Say hello")],
                tools=[],
            )
        )

    assert len(requests) == 2
    assert isinstance(events[-1], AssistantErrorEvent)
    assert events[-1].error.error_message == (
        "Our servers are currently overloaded. Please try again later."
    )


@pytest.mark.anyio
async def test_openai_codex_provider_does_not_retry_non_transient_stream_error() -> None:
    # [FUNCTION DOC]
    # Verifies an invalid API key stream error is not retried.
    requests: list[httpx.Request] = []

    async def credentials() -> OpenAICodexCredentials:
        # [FUNCTION DOC]
        # HELPER: Async Codex credential resolver returning a fixed access token and account id.
        return OpenAICodexCredentials(access_token="access-token", account_id="account-1")

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records requests and returns an invalid_api_key error event.
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"type":"error","error":{"type":"invalid_request_error",'
                '"code":"invalid_api_key","message":"Invalid API key.","param":null},'
                '"sequence_number":1}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(
            OpenAICodexConfig(
                credential_resolver=credentials,
                base_url="https://chatgpt.test/backend-api",
                provider_name="openai-codex",
                max_retries=2,
                max_retry_delay_seconds=0,
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="Say hello")],
                tools=[],
            )
        )

    assert len(requests) == 1
    assert isinstance(events[-1], AssistantErrorEvent)
    assert events[-1].error.error_message == "Invalid API key."


@pytest.mark.anyio
async def test_openai_codex_provider_formats_request_and_streams_text() -> None:
    # [FUNCTION DOC]
    # Verifies the Codex responses URL, auth/account/originator/beta and custom
    # headers, the payload (model, store false, stream, instructions, input_text
    # user content) and the text event sequence.
    requests: list[httpx.Request] = []

    async def credentials() -> OpenAICodexCredentials:
        # [FUNCTION DOC]
        # HELPER: Async Codex credential resolver returning a fixed access token and account id.
        return OpenAICodexCredentials(access_token="access-token", account_id="account-1")

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: asserts the payload and returns two output_text deltas then completed.
        requests.append(request)
        payload = loads(request.content)
        assert payload["model"] == "gpt-5.5"
        assert payload["store"] is False
        assert payload["stream"] is True
        assert payload["instructions"] == "You are Tau."
        assert payload["input"] == [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Say hello"}],
            }
        ]
        return httpx.Response(
            200,
            text=(
                'data: {"type":"response.output_text.delta","delta":"Hel"}\n\n'
                'data: {"type":"response.output_text.delta","delta":"lo"}\n\n'
                'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(
            OpenAICodexConfig(
                credential_resolver=credentials,
                base_url="https://chatgpt.test/backend-api",
                headers={"X-Test": "enabled"},
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="Say hello")],
                tools=[],
            )
        )

    assert [event.type for event in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "done",
    ]
    assert isinstance(events[-1], AssistantDoneEvent)
    assert events[-1].message.text == "Hello"
    assert events[-1].reason == "stop"

    request = requests[0]
    assert request.url == "https://chatgpt.test/backend-api/codex/responses"
    assert request.headers["authorization"] == "Bearer access-token"
    assert request.headers["chatgpt-account-id"] == "account-1"
    assert request.headers["originator"] == "tau"
    assert request.headers["openai-beta"] == "responses=experimental"
    assert request.headers["x-test"] == "enabled"


@pytest.mark.anyio
async def test_openai_codex_provider_includes_configured_reasoning_effort() -> None:
    # [FUNCTION DOC]
    # Verifies reasoning effort is sent as reasoning effort plus "auto" summary.
    requests: list[httpx.Request] = []

    async def credentials() -> OpenAICodexCredentials:
        # [FUNCTION DOC]
        # HELPER: Async Codex credential resolver returning a fixed access token and account id.
        return OpenAICodexCredentials(access_token="access-token", account_id="account-1")

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns a completed event.
        requests.append(request)
        return httpx.Response(
            200,
            text='data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(
            OpenAICodexConfig(
                credential_resolver=credentials,
                base_url="https://chatgpt.test/backend-api",
                reasoning_effort="high",
            ),
            client=client,
        )

        await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="Say hello")],
                tools=[],
            )
        )

    assert loads(requests[0].content)["reasoning"] == {
        "effort": "high",
        "summary": "auto",
    }


@pytest.mark.anyio
async def test_openai_codex_provider_omits_reasoning_when_unset() -> None:
    # [FUNCTION DOC]
    # Verifies no reasoning field is sent without a configured effort.
    requests: list[httpx.Request] = []

    async def credentials() -> OpenAICodexCredentials:
        # [FUNCTION DOC]
        # HELPER: Async Codex credential resolver returning a fixed access token and account id.
        return OpenAICodexCredentials(access_token="access-token", account_id="account-1")

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns a completed event.
        requests.append(request)
        return httpx.Response(
            200,
            text='data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(
            OpenAICodexConfig(
                credential_resolver=credentials,
                base_url="https://chatgpt.test/backend-api",
            ),
            client=client,
        )

        await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="Say hello")],
                tools=[],
            )
        )

    assert "reasoning" not in loads(requests[0].content)


@pytest.mark.anyio
async def test_openai_codex_provider_streams_reasoning_deltas() -> None:
    # [FUNCTION DOC]
    # Verifies reasoning and reasoning_text deltas become thinking events that
    # close before the text block.
    async def credentials() -> OpenAICodexCredentials:
        # [FUNCTION DOC]
        # HELPER: Async Codex credential resolver returning a fixed access token and account id.
        return OpenAICodexCredentials(access_token="access-token", account_id="account-1")

    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns two reasoning deltas, an output text delta and completed.
        return httpx.Response(
            200,
            text=(
                'data: {"type":"response.reasoning.delta","delta":"trace "}\n\n'
                'data: {"type":"response.reasoning_text.delta","delta":"details"}\n\n'
                'data: {"type":"response.output_text.delta","delta":"Done"}\n\n'
                'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(
            OpenAICodexConfig(
                credential_resolver=credentials,
                base_url="https://chatgpt.test/backend-api",
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="Say done")],
                tools=[],
            )
        )

    assert [event.type for event in events] == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_delta",
        "thinking_end",
        "text_start",
        "text_delta",
        "text_end",
        "done",
    ]
    thinking_events = [event for event in events if isinstance(event, ThinkingDeltaEvent)]
    assert [event.delta for event in thinking_events] == ["trace ", "details"]
    assert isinstance(events[-1], AssistantDoneEvent)
    assert events[-1].message.text == "Done"


@pytest.mark.anyio
async def test_openai_codex_provider_preserves_reasoning_summary_part_boundaries() -> None:
    # [FUNCTION DOC]
    # Verifies each reasoning summary part end inserts a blank-line separator
    # delta into the thinking stream and block.
    async def credentials() -> OpenAICodexCredentials:
        # [FUNCTION DOC]
        # HELPER: Async Codex credential resolver returning a fixed access token and account id.
        return OpenAICodexCredentials(access_token="access-token", account_id="account-1")

    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns two summary deltas each followed by part.done, text, completed.
        return httpx.Response(
            200,
            text=(
                'data: {"type":"response.reasoning_summary_text.delta",'
                '"delta":"**First step**"}\n\n'
                'data: {"type":"response.reasoning_summary_part.done"}\n\n'
                'data: {"type":"response.reasoning_summary_text.delta",'
                '"delta":"**Second step**"}\n\n'
                'data: {"type":"response.reasoning_summary_part.done"}\n\n'
                'data: {"type":"response.output_text.delta","delta":"Done"}\n\n'
                'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(
            OpenAICodexConfig(
                credential_resolver=credentials,
                base_url="https://chatgpt.test/backend-api",
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="Say done")],
                tools=[],
            )
        )

    thinking_events = [event for event in events if isinstance(event, ThinkingDeltaEvent)]
    assert [event.delta for event in thinking_events] == [
        "**First step**",
        "\n\n",
        "**Second step**",
        "\n\n",
    ]
    end = events[-1]
    assert isinstance(end, AssistantDoneEvent)
    thinking = end.message.content[0]
    assert isinstance(thinking, ThinkingContent)
    assert thinking.thinking == "**First step**\n\n**Second step**\n\n"


@pytest.mark.anyio
async def test_openai_codex_provider_streams_tool_calls() -> None:
    # [FUNCTION DOC]
    # Verifies Codex tool definitions (flat function format, strict none) and
    # that function call items become ToolCalls with "call_id|item_id" ids.
    async def credentials() -> OpenAICodexCredentials:
        # [FUNCTION DOC]
        # HELPER: Async Codex credential resolver returning a fixed access token and account id.
        return OpenAICodexCredentials(access_token="access-token", account_id="account-1")

    tool = _provider_tool(
        "read",
        "Read a file.",
        {"type": "object", "properties": {"path": {"type": "string"}}},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: asserts tools payload and returns function call added/delta/done events.
        payload = loads(request.content)
        assert payload["tools"] == [
            {
                "type": "function",
                "name": "read",
                "description": "Read a file.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                "strict": None,
            }
        ]
        return httpx.Response(
            200,
            text=(
                'data: {"type":"response.output_item.added",'
                '"item":{"type":"function_call","id":"fc-1","call_id":"call-1","name":"read"}}\n\n'
                'data: {"type":"response.function_call_arguments.delta","delta":"{\\"path\\":"}\n\n'
                'data: {"type":"response.function_call_arguments.done",'
                '"arguments":"{\\"path\\":\\"README.md\\"}"}\n\n'
                'data: {"type":"response.output_item.done",'
                '"item":{"type":"function_call","id":"fc-1","call_id":"call-1",'
                '"name":"read","arguments":"{\\"path\\":\\"README.md\\"}"}}\n\n'
                'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(
            OpenAICodexConfig(
                credential_resolver=credentials,
                base_url="https://chatgpt.test/backend-api",
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="Read README.md")],
                tools=[tool],
            )
        )

    tool_call_events = [event for event in events if isinstance(event, ToolCallEndEvent)]

    assert [event.tool_call for event in tool_call_events] == [
        ToolCall(id="call-1|fc-1", name="read", arguments={"path": "README.md"})
    ]
    assert isinstance(events[-1], AssistantDoneEvent)
    assert events[-1].message.tool_calls == (
        ToolCall(id="call-1|fc-1", name="read", arguments={"path": "README.md"}),
    )


@pytest.mark.anyio
async def test_openai_codex_provider_routes_parallel_tool_argument_streams() -> None:
    # [FUNCTION DOC]
    # Verifies interleaved argument deltas for two parallel function calls are
    # routed by item id and produce both tool calls in order.
    async def credentials() -> OpenAICodexCredentials:
        # [FUNCTION DOC]
        # HELPER: Async Codex credential resolver returning a fixed access token and account id.
        return OpenAICodexCredentials(access_token="access-token", account_id="account-1")

    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns two interleaved function call item streams then completed.
        return httpx.Response(
            200,
            text=(
                'data: {"type":"response.output_item.added","output_index":0,'
                '"item":{"type":"function_call","id":"fc-1","call_id":"call-1","name":"read"}}\n\n'
                'data: {"type":"response.output_item.added","output_index":1,'
                '"item":{"type":"function_call","id":"fc-2","call_id":"call-2","name":"run"}}\n\n'
                'data: {"type":"response.function_call_arguments.delta",'
                '"item_id":"fc-1","delta":"{\\"path\\":"}\n\n'
                'data: {"type":"response.function_call_arguments.delta",'
                '"item_id":"fc-2","delta":"{\\"cmd\\":"}\n\n'
                'data: {"type":"response.function_call_arguments.done",'
                '"item_id":"fc-1","arguments":"{\\"path\\":\\"README.md\\"}"}\n\n'
                'data: {"type":"response.output_item.done","output_index":0,'
                '"item":{"type":"function_call","id":"fc-1","call_id":"call-1","name":"read"}}\n\n'
                'data: {"type":"response.function_call_arguments.done",'
                '"item_id":"fc-2","arguments":"{\\"cmd\\":\\"pwd\\"}"}\n\n'
                'data: {"type":"response.output_item.done","output_index":1,'
                '"item":{"type":"function_call","id":"fc-2","call_id":"call-2","name":"run"}}\n\n'
                'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(
            OpenAICodexConfig(
                credential_resolver=credentials,
                base_url="https://chatgpt.test/backend-api",
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="Use two tools")],
                tools=[],
            )
        )

    tool_call_events = [event for event in events if isinstance(event, ToolCallEndEvent)]

    assert [event.tool_call for event in tool_call_events] == [
        ToolCall(id="call-1|fc-1", name="read", arguments={"path": "README.md"}),
        ToolCall(id="call-2|fc-2", name="run", arguments={"cmd": "pwd"}),
    ]
    assert isinstance(events[-1], AssistantDoneEvent)
    assert events[-1].message.tool_calls == (
        ToolCall(id="call-1|fc-1", name="read", arguments={"path": "README.md"}),
        ToolCall(id="call-2|fc-2", name="run", arguments={"cmd": "pwd"}),
    )


@pytest.mark.anyio
async def test_anthropic_provider_formats_request_and_streams_text() -> None:
    # [FUNCTION DOC]
    # Verifies the Anthropic messages URL, x-api-key, anthropic-version and beta
    # headers, payload (model, stream, system as a cached text block, user text
    # block with ephemeral cache_control) and the text event sequence ending in a
    # "stop" done event.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns message_start, two text deltas,
        # end_turn message_delta and message_stop.
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"type":"message_start","message":{"content":[]}}\n\n'
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"Hel"}}\n\n'
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"lo"}}\n\n'
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n'
                'data: {"type":"message_stop"}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            AnthropicConfig(
                api_key="test-key",
                base_url="https://api.anthropic.test/v1",
                headers={"anthropic-beta": "fine-grained-tool-streaming-2025-05-14"},
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="claude-test",
                system="You are Tau.",
                messages=[UserMessage(content="Say hello")],
                tools=[],
            )
        )

    assert [event.type for event in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "done",
    ]
    assert isinstance(events[-1], AssistantDoneEvent)
    assert events[-1].message.text == "Hello"
    assert events[-1].reason == "stop"

    request = requests[0]
    assert request.url == "https://api.anthropic.test/v1/messages"
    assert request.headers["x-api-key"] == "test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.headers["anthropic-beta"] == "fine-grained-tool-streaming-2025-05-14"

    payload = loads(request.content)
    assert payload["model"] == "claude-test"
    assert payload["stream"] is True
    assert payload["system"] == [
        {
            "type": "text",
            "text": "You are Tau.",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert payload["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Say hello",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("configured_max_tokens", "expected_max_tokens"),
    [(64_000, 64_000), (None, 4096)],
)
async def test_anthropic_provider_sends_configured_max_tokens(
    configured_max_tokens: int | None,
    expected_max_tokens: int,
) -> None:
    # [FUNCTION DOC]
    # Parametrized: configured max tokens are sent, defaulting to 4096 when unset.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns message_stop only.
        requests.append(request)
        return httpx.Response(
            200,
            text='data: {"type":"message_stop"}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            AnthropicConfig(
                api_key="test-key",
                base_url="https://api.anthropic.test/v1",
                max_tokens=configured_max_tokens,
            ),
            client=client,
        )

        await _collect(
            provider.stream_response(
                model="claude-test",
                system="You are Tau.",
                messages=[UserMessage(content="Say hello")],
                tools=[],
            )
        )

    payload = loads(requests[0].content)
    assert payload["max_tokens"] == expected_max_tokens


@pytest.mark.anyio
async def test_anthropic_provider_includes_configured_thinking_budget() -> None:
    # [FUNCTION DOC]
    # Verifies a thinking budget enables thinking with budget_tokens and raises
    # max_tokens to 9216 (the budget plus 1024).
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns message_stop only.
        requests.append(request)
        return httpx.Response(
            200,
            text='data: {"type":"message_stop"}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            AnthropicConfig(
                api_key="test-key",
                base_url="https://api.anthropic.test/v1",
                thinking_budget_tokens=8192,
            ),
            client=client,
        )

        await _collect(
            provider.stream_response(
                model="claude-test",
                system="You are Tau.",
                messages=[UserMessage(content="Say hello")],
                tools=[],
            )
        )

    payload = loads(requests[0].content)
    assert payload["max_tokens"] == 9216
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 8192}


@pytest.mark.anyio
async def test_anthropic_provider_explicitly_disables_default_adaptive_thinking() -> None:
    # [FUNCTION DOC]
    # Verifies thinking_mode "disabled" sends thinking type disabled and no
    # output_config for a model that would default to adaptive thinking.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns message_stop only.
        requests.append(request)
        return httpx.Response(
            200,
            text='data: {"type":"message_stop"}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            AnthropicConfig(
                api_key="test-key",
                base_url="https://api.anthropic.test/v1",
                thinking_mode="disabled",
            ),
            client=client,
        )

        await _collect(
            provider.stream_response(
                model="claude-opus-5",
                system="You are Tau.",
                messages=[UserMessage(content="Say hello")],
                tools=[],
            )
        )

    payload = loads(requests[0].content)
    assert payload["thinking"] == {"type": "disabled"}
    assert "output_config" not in payload


@pytest.mark.anyio
async def test_anthropic_provider_streams_thinking_deltas() -> None:
    # [FUNCTION DOC]
    # Verifies thinking_delta blocks stream as thinking events closed before the
    # text block.
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns two thinking deltas, a text delta, end_turn and message_stop.
        return httpx.Response(
            200,
            text=(
                'data: {"type":"message_start","message":{"content":[]}}\n\n'
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"thinking_delta","thinking":"trace "}}\n\n'
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"thinking_delta","thinking":"details"}}\n\n'
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"Done"}}\n\n'
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n'
                'data: {"type":"message_stop"}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            AnthropicConfig(api_key="test-key", base_url="https://api.anthropic.test/v1"),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="claude-test",
                system="You are Tau.",
                messages=[UserMessage(content="Say done")],
                tools=[],
            )
        )

    assert [event.type for event in events] == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_delta",
        "thinking_end",
        "text_start",
        "text_delta",
        "text_end",
        "done",
    ]
    thinking_events = [event for event in events if isinstance(event, ThinkingDeltaEvent)]
    assert [event.delta for event in thinking_events] == ["trace ", "details"]
    assert isinstance(events[-1], AssistantDoneEvent)
    assert events[-1].message.text == "Done"


@pytest.mark.anyio
async def test_anthropic_provider_retries_transient_status_with_event() -> None:
    # [FUNCTION DOC]
    # Verifies a 503 is retried and the second stream's events are emitted.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: first call returns 503, later calls an "ok" text stream.
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(
            200,
            text=(
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"ok"}}\n\n'
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n'
                'data: {"type":"message_stop"}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            AnthropicConfig(
                api_key="test-key",
                base_url="https://api.anthropic.test/v1",
                max_retries=1,
                max_retry_delay_seconds=0,
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

    assert len(requests) == 2
    assert [event.type for event in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_end",
        "done",
    ]


@pytest.mark.anyio
async def test_anthropic_provider_retries_raw_ssl_error() -> None:
    # [FUNCTION DOC]
    # Verifies a raw ssl.SSLError raised before any response is treated as a
    # retryable transport error.
    # Post-handshake TLS record errors escape httpx as raw ssl.SSLError rather
    # than an httpx.TransportError; they should still be retried.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: first call raises ssl.SSLError, later calls return an "ok" stream.
        requests.append(request)
        if len(requests) == 1:
            raise ssl.SSLError(1, "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] bad record mac")
        return httpx.Response(
            200,
            text=(
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"ok"}}\n\n'
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n'
                'data: {"type":"message_stop"}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            AnthropicConfig(
                api_key="test-key",
                base_url="https://api.anthropic.test/v1",
                max_retries=1,
                max_retry_delay_seconds=0,
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

    assert len(requests) == 2
    assert isinstance(events[-1], AssistantDoneEvent)
    assert events[-1].message.text == "ok"


@pytest.mark.anyio
async def test_anthropic_provider_retries_overloaded_529() -> None:
    # [FUNCTION DOC]
    # Verifies Anthropic's 529 overloaded status is retried.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: first call returns 529, later calls an "ok" stream.
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(529, text="overloaded")
        return httpx.Response(
            200,
            text=(
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"ok"}}\n\n'
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n'
                'data: {"type":"message_stop"}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            AnthropicConfig(
                api_key="test-key",
                base_url="https://api.anthropic.test/v1",
                max_retries=1,
                max_retry_delay_seconds=0,
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

    assert len(requests) == 2
    assert [event.type for event in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_end",
        "done",
    ]


@pytest.mark.anyio
async def test_anthropic_provider_retries_transient_stream_error() -> None:
    # [FUNCTION DOC]
    # Verifies an in-stream overloaded_error before content is retried.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: first call returns an overloaded error event, later an "ok" stream.
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                text=(
                    'data: {"type":"error","error":{"type":"overloaded_error",'
                    '"message":"Overloaded"}}\n\n'
                ),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            text=(
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"ok"}}\n\n'
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n'
                'data: {"type":"message_stop"}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            AnthropicConfig(
                api_key="test-key",
                base_url="https://api.anthropic.test/v1",
                max_retries=1,
                max_retry_delay_seconds=0,
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

    assert len(requests) == 2
    assert [event.type for event in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_end",
        "done",
    ]


@pytest.mark.anyio
async def test_anthropic_provider_surfaces_stream_error_after_retry_exhaustion() -> None:
    # [FUNCTION DOC]
    # Verifies after max retries (3 requests) the overloaded message is surfaced
    # with the raw event and attempt count in diagnostics.
    requests: list[httpx.Request] = []
    error_event = {
        "type": "error",
        "error": {"type": "overloaded_error", "message": "Overloaded"},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records requests and always returns an overloaded error event.
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"type":"error","error":{"type":"overloaded_error",'
                '"message":"Overloaded"}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            AnthropicConfig(
                api_key="test-key",
                base_url="https://api.anthropic.test/v1",
                max_retries=2,
                max_retry_delay_seconds=0,
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

    assert len(requests) == 3
    assert isinstance(events[-1], AssistantErrorEvent)
    assert events[-1].error.error_message == "Overloaded"
    assert events[-1].error.diagnostics[0].details == {
        "event": error_event,
        "attempts": 3,
    }


@pytest.mark.anyio
async def test_anthropic_provider_does_not_retry_non_transient_stream_error() -> None:
    # [FUNCTION DOC]
    # Verifies an authentication_error stream event is not retried.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records requests and returns an authentication error event.
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"type":"error","error":{"type":"authentication_error",'
                '"message":"Invalid API key"}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            AnthropicConfig(
                api_key="test-key",
                base_url="https://api.anthropic.test/v1",
                max_retries=2,
                max_retry_delay_seconds=0,
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

    assert len(requests) == 1
    assert isinstance(events[-1], AssistantErrorEvent)
    assert events[-1].error.error_message == "Invalid API key"


@pytest.mark.anyio
async def test_anthropic_provider_does_not_retry_stream_error_after_content() -> None:
    # [FUNCTION DOC]
    # Verifies a stream error after partial text is not retried and the error
    # message keeps the partial text.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records requests and returns a text delta followed by an error event.
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"partial"}}\n\n'
                'data: {"type":"error","error":{"type":"overloaded_error",'
                '"message":"Overloaded"}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            AnthropicConfig(
                api_key="test-key",
                base_url="https://api.anthropic.test/v1",
                max_retries=2,
                max_retry_delay_seconds=0,
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

    assert len(requests) == 1
    assert isinstance(events[-1], AssistantErrorEvent)
    assert events[-1].error.text == "partial"
    assert events[-1].error.error_message == "Overloaded"


@pytest.mark.anyio
async def test_anthropic_provider_cancellation_stops_stream_error_retry_backoff() -> None:
    # [FUNCTION DOC]
    # Verifies cancellation observed during retry backoff stops further requests
    # and the canonical stream ends with a "no terminal event" error.
    requests: list[httpx.Request] = []

    class CancelDuringBackoff(SimpleCancellationToken):
        # [CLASS DOC]
        # HELPER: Cancellation token that reports cancelled from its second check on.
        def __init__(self) -> None:
            # [FUNCTION DOC]
            # HELPER: Initialises the base token and a zero check counter.
            super().__init__()
            self.checks = 0

        def is_cancelled(self) -> bool:
            # [FUNCTION DOC]
            # HELPER: Increments the check counter and returns true once it reaches 2.
            self.checks += 1
            return self.checks >= 2

    signal = CancelDuringBackoff()

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records requests and always returns an overloaded error event.
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"type":"error","error":{"type":"overloaded_error",'
                '"message":"Overloaded"}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            AnthropicConfig(
                api_key="test-key",
                base_url="https://api.anthropic.test/v1",
                max_retries=2,
                max_retry_delay_seconds=1,
            ),
            client=client,
        )
        events = await _collect(
            provider.stream_response(
                model="claude-test",
                system="You are Tau.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
                signal=signal,
            )
        )

    assert len(requests) == 1
    assert isinstance(events[-1], AssistantErrorEvent)
    assert events[-1].error.error_message == "Provider stream ended without a terminal event"


@pytest.mark.anyio
async def test_anthropic_provider_includes_http_error_detail_in_message() -> None:
    # [FUNCTION DOC]
    # Verifies a 400 JSON error body becomes a friendly message and diagnostics.
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns 400 with a JSON error body.
        return httpx.Response(
            400,
            json={"error": {"message": "model: invalid-model is not supported"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            AnthropicConfig(
                api_key="test-key",
                provider_name="anthropic",
                base_url="https://api.anthropic.test/v1",
                max_retries=0,
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

    assert isinstance(events[-1], AssistantErrorEvent)
    assert events[-1].error.error_message == (
        "anthropic request failed with status 400 for model claude-test: "
        "model: invalid-model is not supported"
    )
    assert events[-1].error.diagnostics[0].details == {
        "status_code": 400,
        "body": '{"error":{"message":"model: invalid-model is not supported"}}',
        "attempts": 1,
    }


def _weather_tool() -> AgentTool:
    # [FUNCTION DOC]
    # HELPER: Builds a get_weather test tool with a city string parameter.
    return _provider_tool(
        "get_weather",
        "Get current weather for a city.",
        {"type": "object", "properties": {"city": {"type": "string"}}},
    )


def test_use_responses_api_routes_only_restricted_models() -> None:
    # [FUNCTION DOC]
    # Verifies the private routing predicate sends gpt-5.5 (incl. pro, any case),
    # gpt-5.4 and codex models to the Responses API and gpt-5.1 to chat
    # completions.
    from tau_ai.openai_compatible import _use_responses_api

    assert _use_responses_api("gpt-5.5") is True
    assert _use_responses_api("gpt-5.5-pro") is True
    assert _use_responses_api("gpt-5.4") is True
    assert _use_responses_api("gpt-5.3-codex") is True
    assert _use_responses_api("GPT-5.5") is True
    assert _use_responses_api("gpt-5.1") is False
    assert _use_responses_api("gpt-5") is False
    assert _use_responses_api("gpt-4o") is False
    assert _use_responses_api("test-model") is False


@pytest.mark.anyio
async def test_responses_api_formats_request_for_restricted_model() -> None:
    # [FUNCTION DOC]
    # Verifies gpt-5.5 on an OpenAI-compatible provider uses /responses with
    # store false, instructions, reasoning effort plus auto summary, flat function
    # tools, and input items: user text, function_call (empty assistant text
    # omitted), function_call_output and the next user message.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns two output text deltas then completed.
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"type":"response.output_text.delta","delta":"Sun"}\n\n'
                'data: {"type":"response.output_text.delta","delta":"ny"}\n\n'
                'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    messages = [
        UserMessage(content="weather in Paris?"),
        AssistantMessage(
            content=assistant_content(
                "", [ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})]
            )
        ),
        ToolResultMessage(
            tool_call_id="call_1",
            tool_name="get_weather",
            content=[TextContent(text='{"temp_c": 19}')],
        ),
        UserMessage(content="summarize"),
    ]

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
                reasoning_effort="medium",
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=messages,
                tools=[_weather_tool()],
            )
        )

    assert [event.type for event in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "done",
    ]
    assert isinstance(events[-1], AssistantDoneEvent)
    assert events[-1].message.text == "Sunny"
    assert events[-1].reason == "stop"

    request = requests[0]
    assert request.url == "https://example.test/v1/responses"

    payload = loads(request.content)
    assert payload["model"] == "gpt-5.5"
    assert payload["stream"] is True
    assert payload["store"] is False
    assert payload["instructions"] == "You are Tau."
    assert payload["reasoning"] == {"effort": "medium", "summary": "auto"}
    # Responses-API tools are flat (no nested "function" object).
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]
    # The assistant turn has empty content, so only its function_call appears.
    assert payload["input"][0] == {"role": "user", "content": "weather in Paris?"}
    function_call = payload["input"][1]
    assert function_call["type"] == "function_call"
    assert function_call["call_id"] == "call_1"
    assert function_call["name"] == "get_weather"
    assert loads(function_call["arguments"]) == {"city": "Paris"}
    assert payload["input"][2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"temp_c": 19}',
    }
    assert payload["input"][3] == {"role": "user", "content": "summarize"}


@pytest.mark.anyio
async def test_responses_api_parses_streamed_tool_call() -> None:
    # [FUNCTION DOC]
    # Verifies Responses API function call events produce one ToolCall with the
    # call_id and parsed arguments and a "toolUse" done reason.
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns function call added, two argument deltas, done events, completed.
        return httpx.Response(
            200,
            text=(
                'data: {"type":"response.output_item.added","output_index":0,'
                '"item":{"id":"fc_1","type":"function_call","call_id":"call_abc",'
                '"name":"get_weather","arguments":""}}\n\n'
                'data: {"type":"response.function_call_arguments.delta",'
                '"item_id":"fc_1","delta":"{\\"city\\":"}\n\n'
                'data: {"type":"response.function_call_arguments.delta",'
                '"item_id":"fc_1","delta":"\\"Paris\\"}"}\n\n'
                'data: {"type":"response.function_call_arguments.done",'
                '"item_id":"fc_1","arguments":"{\\"city\\":\\"Paris\\"}"}\n\n'
                'data: {"type":"response.output_item.done","output_index":0,'
                '"item":{"id":"fc_1","type":"function_call","call_id":"call_abc",'
                '"name":"get_weather","arguments":"{\\"city\\":\\"Paris\\"}"}}\n\n'
                'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="test-key", base_url="https://example.test/v1"),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="weather?")],
                tools=[_weather_tool()],
            )
        )

    tool_call_events = [e for e in events if isinstance(e, ToolCallEndEvent)]
    assert len(tool_call_events) == 1
    assert tool_call_events[0].tool_call.id == "call_abc"
    assert tool_call_events[0].tool_call.name == "get_weather"
    assert tool_call_events[0].tool_call.arguments == {"city": "Paris"}

    end = events[-1]
    assert isinstance(end, AssistantDoneEvent)
    assert len(end.message.tool_calls) == 1
    assert end.message.tool_calls[0].id == "call_abc"
    assert end.message.tool_calls[0].arguments == {"city": "Paris"}
    assert end.reason == "toolUse"


@pytest.mark.anyio
async def test_responses_api_streams_refusal_as_text() -> None:
    # [FUNCTION DOC]
    # Verifies refusal deltas stream as text events and the refusal done text is
    # the final message text.
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns two refusal deltas, refusal done and completed.
        return httpx.Response(
            200,
            text=(
                'data: {"type":"response.refusal.delta","delta":"I can"}\n\n'
                'data: {"type":"response.refusal.delta","delta":"not help with that."}\n\n'
                'data: {"type":"response.refusal.done",'
                '"refusal":"I cannot help with that."}\n\n'
                'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="test-key", base_url="https://example.test/v1"),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="unsafe request")],
                tools=[],
            )
        )

    assert [event.type for event in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "done",
    ]
    text_deltas = [e.delta for e in events if isinstance(e, TextDeltaEvent)]
    assert text_deltas == ["I can", "not help with that."]
    end = events[-1]
    assert isinstance(end, AssistantDoneEvent)
    assert end.message.text == "I cannot help with that."
    assert end.reason == "stop"


@pytest.mark.anyio
async def test_responses_api_streams_reasoning_summary_as_thinking() -> None:
    # [FUNCTION DOC]
    # Verifies reasoning summary deltas become a thinking block before the text.
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns one summary delta, one output text delta and completed.
        return httpx.Response(
            200,
            text=(
                'data: {"type":"response.reasoning_summary_text.delta",'
                '"delta":"Considering"}\n\n'
                'data: {"type":"response.output_text.delta","delta":"Answer"}\n\n'
                'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
                reasoning_effort="high",
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="think")],
                tools=[],
            )
        )

    assert [event.type for event in events] == [
        "start",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "text_start",
        "text_delta",
        "text_end",
        "done",
    ]
    thinking = next(e for e in events if isinstance(e, ThinkingDeltaEvent))
    assert thinking.delta == "Considering"


@pytest.mark.anyio
async def test_responses_api_preserves_reasoning_summary_part_boundaries() -> None:
    # [FUNCTION DOC]
    # Verifies summary part ends insert blank-line separators into thinking.
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns two summary deltas each followed by part.done, text, completed.
        return httpx.Response(
            200,
            text=(
                'data: {"type":"response.reasoning_summary_text.delta",'
                '"delta":"**First step**"}\n\n'
                'data: {"type":"response.reasoning_summary_part.done"}\n\n'
                'data: {"type":"response.reasoning_summary_text.delta",'
                '"delta":"**Second step**"}\n\n'
                'data: {"type":"response.reasoning_summary_part.done"}\n\n'
                'data: {"type":"response.output_text.delta","delta":"Answer"}\n\n'
                'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
                reasoning_effort="high",
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="think")],
                tools=[],
            )
        )

    thinking_events = [event for event in events if isinstance(event, ThinkingDeltaEvent)]
    assert [event.delta for event in thinking_events] == [
        "**First step**",
        "\n\n",
        "**Second step**",
        "\n\n",
    ]
    end = events[-1]
    assert isinstance(end, AssistantDoneEvent)
    thinking = end.message.content[0]
    assert isinstance(thinking, ThinkingContent)
    assert thinking.thinking == "**First step**\n\n**Second step**\n\n"


@pytest.mark.anyio
async def test_responses_api_omits_reasoning_when_effort_is_none() -> None:
    # [FUNCTION DOC]
    # Verifies effort "none" drops the reasoning field while tools are still sent.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns completed.
        requests.append(request)
        return httpx.Response(
            200,
            text='data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
                reasoning_effort="none",
            ),
            client=client,
        )

        await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="hi")],
                tools=[_weather_tool()],
            )
        )

    payload = loads(requests[0].content)
    # gpt-5.5 rejects tools + reasoning on /chat/completions; with thinking off
    # the reasoning field is dropped entirely so tools still work over /responses.
    assert "reasoning" not in payload
    assert "tools" in payload


@pytest.mark.anyio
async def test_responses_api_surfaces_stream_failure() -> None:
    # [FUNCTION DOC]
    # Verifies response.failed yields one error event with the nested message and
    # the raw event in diagnostics.
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns a response.failed event.
        return httpx.Response(
            200,
            text=(
                'data: {"type":"response.failed","response":{"status":"failed",'
                '"error":{"message":"model exploded"}}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="test-key", base_url="https://example.test/v1"),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="hi")],
                tools=[],
            )
        )

    error_events = [e for e in events if isinstance(e, AssistantErrorEvent)]
    assert len(error_events) == 1
    assert error_events[0].error.error_message == "model exploded"
    # The raw event is preserved for debugging (code/param/type, etc.).
    assert error_events[0].error.diagnostics[0].details is not None
    assert error_events[0].error.diagnostics[0].details["event"]["type"] == "response.failed"


@pytest.mark.anyio
async def test_responses_api_orders_parallel_tool_calls_by_output_index() -> None:
    # [FUNCTION DOC]
    # Verifies tool calls are ordered by output index even when done events
    # arrive out of order.
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns two function call items whose done events arrive reversed.
        return httpx.Response(
            200,
            text=(
                'data: {"type":"response.output_item.added","output_index":0,'
                '"item":{"id":"fc_a","type":"function_call","call_id":"call_a",'
                '"name":"get_weather","arguments":"{\\"city\\":\\"A\\"}"}}\n\n'
                'data: {"type":"response.output_item.added","output_index":1,'
                '"item":{"id":"fc_b","type":"function_call","call_id":"call_b",'
                '"name":"get_weather","arguments":"{\\"city\\":\\"B\\"}"}}\n\n'
                # Done events arrive out of order to prove sorting by output_index.
                'data: {"type":"response.output_item.done","output_index":1,'
                '"item":{"id":"fc_b","type":"function_call","call_id":"call_b",'
                '"name":"get_weather","arguments":"{\\"city\\":\\"B\\"}"}}\n\n'
                'data: {"type":"response.output_item.done","output_index":0,'
                '"item":{"id":"fc_a","type":"function_call","call_id":"call_a",'
                '"name":"get_weather","arguments":"{\\"city\\":\\"A\\"}"}}\n\n'
                'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="test-key", base_url="https://example.test/v1"),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="weather?")],
                tools=[_weather_tool()],
            )
        )

    end = events[-1]
    assert isinstance(end, AssistantDoneEvent)
    assert [tc.id for tc in end.message.tool_calls] == ["call_a", "call_b"]
    assert [tc.arguments["city"] for tc in end.message.tool_calls] == ["A", "B"]


@pytest.mark.anyio
async def test_responses_api_surfaces_top_level_error_event() -> None:
    # [FUNCTION DOC]
    # Verifies a top-level "error" event yields one error with its message.
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns a single top-level error event.
        return httpx.Response(
            200,
            text='data: {"type":"error","message":"rate limited"}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="test-key", base_url="https://example.test/v1"),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="hi")],
                tools=[],
            )
        )

    error_events = [e for e in events if isinstance(e, AssistantErrorEvent)]
    assert len(error_events) == 1
    assert error_events[0].error.error_message == "rate limited"


@pytest.mark.anyio
async def test_responses_api_maps_incomplete_status_to_length() -> None:
    # [FUNCTION DOC]
    # Verifies response.incomplete ends with a "length" done event keeping text.
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns a text delta then response.incomplete.
        return httpx.Response(
            200,
            text=(
                'data: {"type":"response.output_text.delta","delta":"partial"}\n\n'
                'data: {"type":"response.incomplete","response":{"status":"incomplete"}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="test-key", base_url="https://example.test/v1"),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="hi")],
                tools=[],
            )
        )

    end = events[-1]
    assert isinstance(end, AssistantDoneEvent)
    assert end.message.text == "partial"
    assert end.reason == "length"


@pytest.mark.anyio
async def test_openai_compatible_provider_reports_usage() -> None:
    # [FUNCTION DOC]
    # Verifies chat completions request include_usage and map usage: input =
    # prompt minus cached, cache read, reasoning tokens, total, zero cost.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns text then a usage-only chunk and [DONE].
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":30,"completion_tokens":5,'
                '"prompt_tokens_details":{"cached_tokens":10},'
                '"completion_tokens_details":{"reasoning_tokens":2}}}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Tau.",
                messages=[UserMessage(content="Say hi")],
                tools=[],
            )
        )

    # The request opts in to streamed usage reporting.
    assert loads(requests[0].content)["stream_options"] == {"include_usage": True}

    assert isinstance(events[-1], AssistantDoneEvent)
    usage = events[-1].message.usage
    assert usage is not None
    assert usage.input == 20  # 30 prompt - 10 cached
    assert usage.output == 5
    assert usage.cache_read == 10
    assert usage.cache_write == 0
    assert usage.reasoning == 2
    assert usage.total_tokens == 35
    assert usage.cost.total == 0


@pytest.mark.anyio
async def test_openai_codex_provider_reports_usage_and_sends_cache_affinity() -> None:
    # [FUNCTION DOC]
    # Verifies Codex truncates the session id to 64 chars for prompt_cache_key and
    # the session-id header (no client request id), and maps usage: input minus
    # cached and cache-written tokens, reasoning, total.
    requests: list[httpx.Request] = []

    async def credentials() -> OpenAICodexCredentials:
        # [FUNCTION DOC]
        # HELPER: Async Codex credential resolver returning a fixed access token and account id.
        return OpenAICodexCredentials(access_token="access-token", account_id="account-1")

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns text then completed with usage.
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n'
                'data: {"type":"response.completed","response":{"status":"completed",'
                '"usage":{"input_tokens":50,"output_tokens":8,"total_tokens":58,'
                '"input_tokens_details":{"cached_tokens":12,"cache_write_tokens":8},'
                '"output_tokens_details":{"reasoning_tokens":3}}}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(
            OpenAICodexConfig(
                credential_resolver=credentials,
                base_url="https://chatgpt.test/backend-api",
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="Say hi")],
                tools=[],
                session_id="c" * 70,
            )
        )

    cache_key = "c" * 64
    assert loads(requests[0].content)["prompt_cache_key"] == cache_key
    assert requests[0].headers["session-id"] == cache_key
    assert "x-client-request-id" not in requests[0].headers
    assert isinstance(events[-1], AssistantDoneEvent)
    usage = events[-1].message.usage
    assert usage is not None
    assert usage.input == 30  # 50 input - 12 cached - 8 written
    assert usage.output == 8
    assert usage.cache_read == 12
    assert usage.cache_write == 8
    assert usage.reasoning == 3
    assert usage.total_tokens == 58
    assert usage.cost.total == 0


@pytest.mark.anyio
async def test_openai_compatible_responses_reports_usage_and_sends_cache_affinity() -> None:
    # [FUNCTION DOC]
    # Verifies OpenAI Responses requests send prompt_cache_key and a session_id
    # header and map usage like Codex.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns text then completed with usage.
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n'
                'data: {"type":"response.completed","response":{"status":"completed",'
                '"usage":{"input_tokens":50,"output_tokens":8,"total_tokens":58,'
                '"input_tokens_details":{"cached_tokens":12,"cache_write_tokens":8},'
                '"output_tokens_details":{"reasoning_tokens":3}}}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="Say hi")],
                tools=[],
                session_id="responses-session",
            )
        )

    assert loads(requests[0].content)["prompt_cache_key"] == "responses-session"
    assert requests[0].headers["session_id"] == "responses-session"
    assert "x-client-request-id" not in requests[0].headers
    assert isinstance(events[-1], AssistantDoneEvent)
    usage = events[-1].message.usage
    assert usage is not None
    assert usage.input == 30  # 50 input - 12 cached - 8 written
    assert usage.output == 8
    assert usage.cache_read == 12
    assert usage.cache_write == 8
    assert usage.reasoning == 3
    assert usage.total_tokens == 58
    assert usage.cost.total == 0


@pytest.mark.anyio
async def test_anthropic_provider_reports_usage() -> None:
    # [FUNCTION DOC]
    # Verifies Anthropic usage: input excludes cache, output updated by
    # message_delta, cache read/write and 1h cache write, total of all four.
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns message_start with usage, a text delta, message_delta with output
        # usage and message_stop.
        return httpx.Response(
            200,
            text=(
                'data: {"type":"message_start","message":{"content":[],"usage":'
                '{"input_tokens":100,"output_tokens":1,"cache_read_input_tokens":40,'
                '"cache_creation_input_tokens":25,'
                '"cache_creation":{"ephemeral_1h_input_tokens":10}}}}\n\n'
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"Hi"}}\n\n'
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
                '"usage":{"output_tokens":7}}\n\n'
                'data: {"type":"message_stop"}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            AnthropicConfig(
                api_key="test-key",
                base_url="https://api.anthropic.test/v1",
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="claude-test",
                system="You are Tau.",
                messages=[UserMessage(content="Say hi")],
                tools=[],
            )
        )

    assert isinstance(events[-1], AssistantDoneEvent)
    usage = events[-1].message.usage
    assert usage is not None
    assert usage.input == 100  # Anthropic input excludes cache reads and writes
    assert usage.output == 7  # updated by message_delta
    assert usage.cache_read == 40
    assert usage.cache_write == 25
    assert usage.cache_write_1h == 10
    assert usage.total_tokens == 172  # 100 + 7 + 40 + 25
    assert usage.cost.total == 0


@pytest.mark.anyio
async def test_openai_compatible_provider_can_disable_usage_in_streaming() -> None:
    # [FUNCTION DOC]
    # Verifies supportsUsageInStreaming false omits stream_options and leaves zero
    # usage.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns an "ok" stream.
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
                compat={"supportsUsageInStreaming": False},
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Tau.",
                messages=[UserMessage(content="Say ok")],
                tools=[],
            )
        )

    assert "stream_options" not in loads(requests[0].content)
    assert isinstance(events[-1], AssistantDoneEvent)
    assert events[-1].message.usage.total_tokens == 0


@pytest.mark.anyio
async def test_openai_compatible_provider_reads_usage_from_choice_fallback() -> None:
    # [FUNCTION DOC]
    # Verifies usage nested on the choice (Moonshot style) is read.
    # Moonshot-style: usage lives on the choice, not at the chunk top level.
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns a chunk whose choice carries usage, then [DONE].
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop",'
                '"usage":{"prompt_tokens":15,"completion_tokens":4}}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Tau.",
                messages=[UserMessage(content="Say hi")],
                tools=[],
            )
        )

    assert isinstance(events[-1], AssistantDoneEvent)
    usage = events[-1].message.usage
    assert usage is not None
    assert usage.input == 15
    assert usage.output == 4
    assert usage.cache_read == 0
    assert usage.total_tokens == 19


@pytest.mark.anyio
async def test_openai_compatible_provider_falls_back_to_prompt_cache_hit_tokens() -> None:
    # [FUNCTION DOC]
    # Verifies prompt_cache_hit_tokens (DeepSeek style) is used as cache read when
    # cached_tokens is absent.
    # DeepSeek-style: cache reads come via prompt_cache_hit_tokens, and there
    # is no prompt_tokens_details.cached_tokens to prefer.
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns text then usage with prompt_cache_hit_tokens.
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":6,'
                '"prompt_cache_hit_tokens":16}}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Tau.",
                messages=[UserMessage(content="Say hi")],
                tools=[],
            )
        )

    assert isinstance(events[-1], AssistantDoneEvent)
    usage = events[-1].message.usage
    assert usage is not None
    assert usage.input == 24  # 40 prompt - 16 cache hits
    assert usage.cache_read == 16
    assert usage.output == 6
    assert usage.total_tokens == 46


@pytest.mark.anyio
async def test_openai_compatible_provider_reported_zero_cached_tokens_wins() -> None:
    # [FUNCTION DOC]
    # Verifies an explicit cached_tokens of zero is not overridden by
    # prompt_cache_hit_tokens (nullish fallback semantics).
    # Nullish semantics (Pi: cached_tokens ?? prompt_cache_hit_tokens ?? 0):
    # an explicitly reported cached_tokens of 0 must not fall through to
    # prompt_cache_hit_tokens.
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns text then usage with cached_tokens 0 and cache hit tokens 16.
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":6,'
                '"prompt_tokens_details":{"cached_tokens":0},'
                '"prompt_cache_hit_tokens":16}}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://example.test/v1",
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="test-model",
                system="You are Tau.",
                messages=[UserMessage(content="Say hi")],
                tools=[],
            )
        )

    assert isinstance(events[-1], AssistantDoneEvent)
    usage = events[-1].message.usage
    assert usage is not None
    assert usage.cache_read == 0
    assert usage.input == 40


@pytest.mark.anyio
async def test_anthropic_provider_reports_usage_from_message_delta_only() -> None:
    # [FUNCTION DOC]
    # Verifies usage reported only on message_delta still produces a Usage with
    # no reasoning count.
    # No usage on message_start: the message_delta usage alone must still
    # produce a Usage (the `usage or Usage()` branch).
    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns message_start without usage, text, message_delta with usage.
        return httpx.Response(
            200,
            text=(
                'data: {"type":"message_start","message":{"content":[]}}\n\n'
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"Hi"}}\n\n'
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
                '"usage":{"input_tokens":12,"output_tokens":3}}\n\n'
                'data: {"type":"message_stop"}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            AnthropicConfig(
                api_key="test-key",
                base_url="https://api.anthropic.test/v1",
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="claude-test",
                system="You are Tau.",
                messages=[UserMessage(content="Say hi")],
                tools=[],
            )
        )

    assert isinstance(events[-1], AssistantDoneEvent)
    usage = events[-1].message.usage
    assert usage is not None
    assert usage.input == 12
    assert usage.output == 3
    assert usage.cache_read == 0
    assert usage.cache_write == 0
    assert usage.reasoning is None
    assert usage.total_tokens == 15


@pytest.mark.anyio
async def test_openai_codex_provider_leaves_reasoning_none_when_unreported() -> None:
    # [FUNCTION DOC]
    # Verifies Codex leaves reasoning none and cache read zero when not reported.
    async def credentials() -> OpenAICodexCredentials:
        # [FUNCTION DOC]
        # HELPER: Async Codex credential resolver returning a fixed access token and account id.
        return OpenAICodexCredentials(access_token="access-token", account_id="account-1")

    def handler(_request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: returns text then completed with basic usage.
        return httpx.Response(
            200,
            text=(
                'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n'
                'data: {"type":"response.completed","response":{"status":"completed",'
                '"usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12}}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICodexProvider(
            OpenAICodexConfig(
                credential_resolver=credentials,
                base_url="https://chatgpt.test/backend-api",
            ),
            client=client,
        )

        events = await _collect(
            provider.stream_response(
                model="gpt-5.5",
                system="You are Tau.",
                messages=[UserMessage(content="Say hi")],
                tools=[],
            )
        )

    assert isinstance(events[-1], AssistantDoneEvent)
    usage = events[-1].message.usage
    assert usage is not None
    assert usage.reasoning is None
    assert usage.cache_read == 0
    assert usage.total_tokens == 12


@pytest.mark.anyio
async def test_github_copilot_sends_vision_header_for_tool_result_images() -> None:
    # [FUNCTION DOC]
    # Verifies the Copilot Responses provider sends Copilot-Vision-Request when a
    # tool result contains an image.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns completed.
        requests.append(request)
        return httpx.Response(
            200,
            text='data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    message = ToolResultMessage(
        tool_call_id="call-1",
        tool_name="read",
        content=[ImageContent(data="aW1hZ2U=", mime_type="image/png")],
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test-key",
                base_url="https://copilot.test",
                api="openai-responses",
                provider_name="github-copilot",
                supports_images=True,
            ),
            client=client,
        )
        await _collect(
            provider.stream_response(
                model="gpt-5.4",
                system="You are Tau.",
                messages=[message],
                tools=[],
            )
        )

    assert requests[0].headers["Copilot-Vision-Request"] == "true"


@pytest.mark.anyio
async def test_github_copilot_anthropic_sends_vision_header() -> None:
    # [FUNCTION DOC]
    # Verifies the Copilot Anthropic provider sends Copilot-Vision-Request for
    # tool result images.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # [FUNCTION DOC]
        # HELPER: Mock HTTP transport handler: records the request and returns message_stop.
        requests.append(request)
        return httpx.Response(
            200,
            text='data: {"type":"message_stop"}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    message = ToolResultMessage(
        tool_call_id="call-1",
        tool_name="read",
        content=[ImageContent(data="aW1hZ2U=", mime_type="image/png")],
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AnthropicProvider(
            AnthropicConfig(
                api_key="test-key",
                base_url="https://copilot.test",
                provider_name="github-copilot",
                supports_images=True,
            ),
            client=client,
        )
        await _collect(
            provider.stream_response(
                model="claude-sonnet-4.6",
                system="You are Tau.",
                messages=[message],
                tools=[],
            )
        )

    assert requests[0].headers["Copilot-Vision-Request"] == "true"
