# [MODULE DOC]
# Tests for print-mode renderers in tau_coding.rendering: the streaming transcript
# renderer (stdout text, stderr tool/retry/error lines), the final-text renderer (prints
# only the final assistant message, errors on finish) and the JSONL event renderer
# (canonical Pi-style camelCase event objects). Uses pytest capsys to capture output.
import json

import pytest

from tau_agent import (
    AgentToolResult,
    AssistantMessage,
    CustomMessage,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    TextContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from tau_agent.provider_events import TextDeltaEvent, ThinkingDeltaEvent
from tau_coding.events import AutoRetryEndEvent, AutoRetryStartEvent, QueueUpdateEvent
from tau_coding.rendering import FinalTextRenderer, JsonEventRenderer, TranscriptRenderer


def _assistant_update(event) -> MessageUpdateEvent:  # noqa: ANN001
    # [FUNCTION DOC]
    # HELPER: Wrap a provider assistant-message event in an agent message-update event whose
    # message is the event's partial assistant message.
    return MessageUpdateEvent(message=event.partial, assistant_message_event=event)


def _error_end(message: str) -> MessageEndEvent:
    # [FUNCTION DOC]
    # HELPER: Build a message-end event for an assistant message with stop reason "error"
    # and the given error message text.
    return MessageEndEvent(message=AssistantMessage(stop_reason="error", error_message=message))


def test_transcript_renderer_streams_text_and_tool_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # [FUNCTION DOC]
    # Feeds start, a thinking delta, two text deltas, an auto-retry start, and tool start/
    # update/end events. Asserts stdout is exactly "Hello" plus newline, thinking text never
    # appears, and stderr contains the retry message, the tool call line, progress text,
    # the success check mark with tool name, and the result text; finish returns True.
    renderer = TranscriptRenderer()
    partial = AssistantMessage(content="")

    renderer.render(MessageStartEvent(message=partial))
    renderer.render(
        _assistant_update(
            ThinkingDeltaEvent(content_index=0, delta="hidden reasoning", partial=partial)
        )
    )
    renderer.render(
        _assistant_update(TextDeltaEvent(content_index=0, delta="Hel", partial=partial))
    )
    renderer.render(_assistant_update(TextDeltaEvent(content_index=0, delta="lo", partial=partial)))
    renderer.render(
        AutoRetryStartEvent(
            attempt=2,
            max_attempts=3,
            delay_ms=0,
            error_message="Retrying provider request 2/3 after HTTP 503.",
        )
    )
    renderer.render(
        ToolExecutionStartEvent(tool_call_id="call-1", tool_name="read", args={"path": "a.py"})
    )
    renderer.render(
        ToolExecutionUpdateEvent(
            tool_call_id="call-1",
            tool_name="read",
            args={"path": "a.py"},
            partial_result=AgentToolResult(content="reading"),
        )
    )
    renderer.render(
        ToolExecutionEndEvent(
            tool_call_id="call-1",
            tool_name="read",
            result=AgentToolResult(content="done"),
            is_error=False,
        )
    )

    captured = capsys.readouterr()
    assert renderer.finish() is True
    assert captured.out == "Hello\n"
    assert "hidden reasoning" not in captured.out
    assert "hidden reasoning" not in captured.err
    assert "… Retrying provider request 2/3 after HTTP 503." in captured.err
    assert "→ read a.py" in captured.err
    assert "… reading" in captured.err
    assert "✓ read" in captured.err
    assert "done" in captured.err


def test_transcript_renderer_keeps_full_bash_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # [FUNCTION DOC]
    # A bash tool start renders the full command with a "(timeout 120s)" suffix on stderr
    # and omits the human description argument.
    renderer = TranscriptRenderer()
    command = "git diff --check && git commit -m 'Finish work'"

    renderer.render(
        ToolExecutionStartEvent(
            tool_call_id="call-1",
            tool_name="bash",
            args={
                "command": command,
                "description": "Validating and committing changes",
                "timeout": 120,
            },
        )
    )

    captured = capsys.readouterr()
    assert f"$ {command} (timeout 120s)" in captured.err
    assert "Validating and committing changes" not in captured.err


def test_transcript_renderer_uses_custom_message_renderer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # [FUNCTION DOC]
    # A display custom message at message end is rendered through the supplied custom
    # renderer (called with type, content, details, expanded False); its Rich markup output
    # appears on stderr and raw message content does not.
    def render(custom_type: str, content: str, details: object, expanded: bool) -> str | None:
        # [FUNCTION DOC]
        # HELPER: Fake custom message renderer asserting the expected type and non-expanded
        # mode, returning a Rich markup string.
        assert custom_type == "subagent-notification"
        assert expanded is False
        return "[bold]✓ research done[/bold]"

    renderer = TranscriptRenderer(custom_message_renderer=render)
    renderer.render(
        MessageEndEvent(
            message=CustomMessage(
                custom_type="subagent-notification",
                content="<task-notification>raw xml</task-notification>",
                details={"id": "run-1"},
            )
        )
    )

    captured = capsys.readouterr()
    assert "✓ research done" in captured.err
    assert "raw xml" not in captured.err


def test_transcript_renderer_fails_on_assistant_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # [FUNCTION DOC]
    # An assistant error message end prints "Error: provider failed" to stderr and makes
    # finish return False.
    renderer = TranscriptRenderer()

    renderer.render(_error_end("provider failed"))

    captured = capsys.readouterr()
    assert renderer.finish() is False
    assert "Error: provider failed" in captured.err


@pytest.mark.parametrize(
    "renderer_type",
    [TranscriptRenderer, FinalTextRenderer, JsonEventRenderer],
)
def test_renderers_finish_successfully_after_recovered_auto_retry(
    renderer_type: type[TranscriptRenderer] | type[FinalTextRenderer] | type[JsonEventRenderer],
) -> None:
    # [FUNCTION DOC]
    # Parametrised over all three renderers: an error followed by a successful auto-retry
    # end event clears the failure so finish returns True.
    renderer = renderer_type()

    renderer.render(_error_end("provider failed"))
    renderer.render(AutoRetryEndEvent(success=True, attempt=1))

    assert renderer.finish() is True


def test_final_text_renderer_prints_only_final_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # [FUNCTION DOC]
    # Streaming deltas print nothing; the final assistant message end prints its text plus
    # newline to stdout. finish returns True throughout.
    renderer = FinalTextRenderer()
    partial = AssistantMessage(content="ignored")

    renderer.render(
        _assistant_update(TextDeltaEvent(content_index=0, delta="ignored", partial=partial))
    )
    assert renderer.finish() is True
    assert capsys.readouterr().out == ""

    renderer.render(MessageEndEvent(message=AssistantMessage(content="Final answer")))
    assert renderer.finish() is True
    assert capsys.readouterr().out == "Final answer\n"


def test_final_text_renderer_prints_errors_on_finish(capsys: pytest.CaptureFixture[str]) -> None:
    # [FUNCTION DOC]
    # Errors are deferred: nothing on stderr at message end, then finish returns False and
    # prints the error message to stderr.
    renderer = FinalTextRenderer()

    renderer.render(_error_end("provider failed"))
    assert capsys.readouterr().err == ""
    assert renderer.finish() is False
    assert "Error: provider failed" in capsys.readouterr().err


def test_json_renderer_emits_canonical_jsonl(capsys: pytest.CaptureFixture[str]) -> None:
    # [FUNCTION DOC]
    # Each event becomes one JSON line: message_start, queue_update with steering and
    # followUp lists, message_update with nested assistantMessageEvent thinking_delta, and
    # a message end with stopReason "error"; finish returns False.
    renderer = JsonEventRenderer()
    partial = AssistantMessage(content=[TextContent(text="hidden reasoning")])

    renderer.render(MessageStartEvent(message=AssistantMessage()))
    renderer.render(QueueUpdateEvent(steering=("adjust",), follow_up=("after",)))
    renderer.render(
        _assistant_update(
            ThinkingDeltaEvent(
                content_index=0,
                delta="hidden reasoning",
                partial=partial,
            )
        )
    )
    renderer.render(_error_end("provider failed"))

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["type"] == "message_start"
    assert lines[1] == {
        "type": "queue_update",
        "steering": ["adjust"],
        "followUp": ["after"],
    }
    assert lines[2]["type"] == "message_update"
    assert lines[2]["assistantMessageEvent"]["type"] == "thinking_delta"
    assert lines[3]["message"]["stopReason"] == "error"
    assert renderer.finish() is False
