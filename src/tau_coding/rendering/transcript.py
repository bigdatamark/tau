# [MODULE DOC]
# Human-readable streaming transcript renderer for print-mode CLI output (Pi-compatible).
# Assistant text deltas stream raw to stdout (via typer.echo without newlines), while
# tool calls, tool progress, retries, tool results, custom extension messages and
# errors are printed with Rich styling to stderr. Tracks whether the run failed so the
# CLI can choose an exit status.
"""Human-readable Pi-compatible streaming transcript renderer."""

import typer
from rich.console import Console
from rich.text import Text

from tau_agent.events import (
    AgentEndEvent,
    MessageEndEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from tau_agent.messages import AssistantMessage, CustomMessage, ToolCall
from tau_ai.events import TextDeltaEvent
from tau_coding.events import AutoRetryEndEvent, AutoRetryStartEvent, CodingSessionEvent
from tau_coding.extensions.api import CustomMessageMarkup
from tau_coding.tui.state import format_tool_call_block


class TranscriptRenderer:
    # [CLASS DOC]
    # Stateful renderer that consumes coding-session events one at a time.
    # State: whether assistant text has started, whether a terminating newline has been
    # written, whether the run failed, a stderr Rich console (highlighting off) and an
    # optional custom-message renderer supplied by extensions.
    # KNOWN ISSUE: the started/ended flags are never reset between assistant messages, so
    # after the first assistant text block is terminated, later streamed text blocks do not
    # get a separating/terminating newline from this renderer.
    def __init__(
        self,
        *,
        custom_message_renderer: CustomMessageMarkup | None = None,
        **_: object,
    ) -> None:
        # [FUNCTION DOC]
        # Initialise renderer state. Inputs (keyword-only): optional custom message renderer
        # (callable taking custom type, text, details mapping or None, and an expanded flag,
        # returning Rich renderable or falsy); any other keyword arguments are accepted and
        # ignored for interface compatibility with other renderers. Creates a stderr Console.
        self._assistant_started = False
        self._assistant_ended = False
        self._failed = False
        self._console = Console(stderr=True, highlight=False)
        self._custom_message_renderer = custom_message_renderer

    def render(self, event: CodingSessionEvent) -> None:
        # [FUNCTION DOC]
        # Render one coding-session event; returns nothing.
        # - Message update with a text delta: marks assistant started, echoes delta to stdout
        # without newline. Other nested updates (thinking, tool-call deltas) are ignored.
        # - Tool execution start: ends any open text line, prints the formatted tool-call
        # block (non-compact) in cyan to stderr.
        # - Tool execution update: ends open text line; prints partial text prefixed by an
        # ellipsis in dim style if non-empty.
        # - Auto-retry start: ends open text line; prints the retry error message dimmed.
        # - Auto-retry end with success: clears the failed flag.
        # - Tool execution end: prints a check mark (green) or cross (red) with tool name, then
        # each line of result text indented by two spaces.
        # - Message end for a custom message: if a renderer exists and the message is marked
        # for display, calls it (details passed only if it is a mapping) and prints any result.
        # - Message end for an assistant message: on stop reason "error" sets failed and
        # prints "Error: <message>" in red; then performs a final newline.
        # - Agent end: performs a final newline.
        # Side effects: writes to stdout/stderr and mutates internal flags.
        if isinstance(event, MessageUpdateEvent):
            nested = event.assistant_message_event
            if isinstance(nested, TextDeltaEvent):
                self._assistant_started = True
                typer.echo(nested.delta, nl=False)
            return
        if isinstance(event, ToolExecutionStartEvent):
            self._newline()
            call = ToolCall(id=event.tool_call_id, name=event.tool_name, arguments=event.args)
            self._console.print(Text(format_tool_call_block(call, compact=False), style="cyan"))
            return
        if isinstance(event, ToolExecutionUpdateEvent):
            self._newline()
            if event.partial_result.text:
                self._console.print(Text(f"… {event.partial_result.text}", style="bright_black"))
            return
        if isinstance(event, AutoRetryStartEvent):
            self._newline()
            self._console.print(Text(f"… {event.error_message}", style="bright_black"))
            return
        if isinstance(event, AutoRetryEndEvent) and event.success:
            self._failed = False
            return
        if isinstance(event, ToolExecutionEndEvent):
            status = "✗" if event.is_error else "✓"
            style = "red" if event.is_error else "green"
            self._console.print(Text(f"{status} {event.tool_name}", style=style))
            if event.result.text:
                for line in event.result.text.splitlines():
                    self._console.print(Text(f"  {line}"))
            return
        if isinstance(event, MessageEndEvent) and isinstance(event.message, CustomMessage):
            if self._custom_message_renderer is None or not event.message.display:
                return
            rendered = self._custom_message_renderer(
                event.message.custom_type,
                event.message.text,
                event.message.details if isinstance(event.message.details, dict) else None,
                False,
            )
            if rendered:
                self._newline()
                self._console.print(rendered)
            return
        if isinstance(event, MessageEndEvent) and isinstance(event.message, AssistantMessage):
            if event.message.stop_reason == "error":
                self._failed = True
                self._newline()
                self._console.print(
                    Text(f"Error: {event.message.error_message or 'Error'}", style="red")
                )
            self._newline(final=True)
            return
        if isinstance(event, AgentEndEvent):
            self._newline(final=True)

    def finish(self) -> bool:
        # [FUNCTION DOC]
        # Return True when the run did not end in an (unrecovered) assistant error, else False.
        return not self._failed

    def _newline(self, *, final: bool = False) -> None:
        # [FUNCTION DOC]
        # HELPER: Terminate the streamed assistant text line exactly once.
        # If assistant text started and has not ended, writes a newline to stdout and marks
        # ended. Else, when called with final=True and no text ever started, just marks ended
        # (no output). Keyword input: final flag. Returns nothing.
        if self._assistant_started and not self._assistant_ended:
            typer.echo()
            self._assistant_ended = True
        elif final and not self._assistant_started:
            self._assistant_ended = True
