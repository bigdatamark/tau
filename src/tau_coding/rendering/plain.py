# [MODULE DOC]
# Plain final-text renderer used by print mode (non-interactive CLI).
# Mirrors Pi's print mode: it consumes coding-session events silently and, at the
# end of the run, prints only the last assistant text to stdout, or the collected
# error messages to stderr if the final assistant turn failed.
"""Pi-style final text renderer for print mode."""

import typer

from tau_agent.events import MessageEndEvent
from tau_agent.messages import AssistantMessage
from tau_coding.events import AutoRetryEndEvent, CodingSessionEvent


class FinalTextRenderer:
    # [CLASS DOC]
    # Stateful event consumer for print mode. Tracks the latest assistant text, whether
    # the run ended in an error, and the error messages seen so far. Call render for
    # each session event, then finish once the run completes.
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Initialize empty state: no last assistant text, not failed, no error messages.
        # Inputs: none. Outputs: none. No side effects beyond setting attributes.
        self._last_assistant_text = ""
        self._failed = False
        self._error_messages: list[str] = []

    def render(self, event: CodingSessionEvent) -> None:
        # [FUNCTION DOC]
        # Update renderer state from one coding-session event; prints nothing.
        # Behavior:
        # - A successful auto-retry-end event clears the failed flag and error list
        #   (a retry recovered from earlier errors) and returns.
        # - Any event that is not a message-end carrying an assistant message is ignored.
        # - For an assistant message-end: remember its text as the last assistant text;
        #   if its stop reason is "error" or "aborted", set failed to true only for
        #   "error" (aborted clears it) and append its error message if present.
        # Inputs: event, any coding-session event. Outputs: none. Raises nothing.
        # Side effects: mutates internal state.
        # KNOWN ISSUE: an assistant message ending normally after an earlier error does
        # not reset the failed flag or error list, so an older error can still win in finish
        # unless an auto-retry-end success event arrived.
        if isinstance(event, AutoRetryEndEvent) and event.success:
            self._failed = False
            self._error_messages.clear()
            return
        if not isinstance(event, MessageEndEvent) or not isinstance(
            event.message, AssistantMessage
        ):
            return
        self._last_assistant_text = event.message.text
        if event.message.stop_reason in {"error", "aborted"}:
            self._failed = event.message.stop_reason == "error"
            if event.message.error_message:
                self._error_messages.append(event.message.error_message)

    def finish(self) -> bool:
        # [FUNCTION DOC]
        # Emit the final output of the run.
        # If failed: writes each collected error as "Error: <message>" to stderr and
        # returns False. Otherwise writes the last assistant text to stdout if non-empty
        # and returns True.
        # Outputs: boolean success flag (used as the process exit status).
        # Side effects: terminal output via typer.
        if self._failed:
            for message in self._error_messages:
                typer.echo(f"Error: {message}", err=True)
            return False
        if self._last_assistant_text:
            typer.echo(self._last_assistant_text)
        return True
