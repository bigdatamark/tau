# [MODULE DOC]
# Renderer for Pi-compatible JSON print mode: writes every coding-session event as one JSON line
# to stdout and tracks whether the run ended in failure. Used by the CLI when JSON output is
# requested.
"""Pi-compatible JSON event stream renderer."""

import typer

from tau_agent.events import MessageEndEvent
from tau_agent.messages import AssistantMessage
from tau_coding.events import AutoRetryEndEvent, CodingSessionEvent


class JsonEventRenderer:
    # [CLASS DOC]
    # Stateful event sink for JSON mode. Holds a single failure flag; render() is called once per
    # coding-session event and finish() reports overall success.
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Initialize the renderer with the failure flag cleared. No inputs, no output, no side
        # effects.
        self._failed = False

    def render(self, event: CodingSessionEvent) -> None:
        # [FUNCTION DOC]
        # Serialize one coding-session event to JSON (field aliases used, null fields omitted) and
        # print it as a line via typer echo to stdout.
        # Before printing, update failure state: a successful auto-retry-end event clears the
        # failure flag; a message-end event whose message is an assistant message with stop reason
        # "error" sets it.
        # Input: the event. Output: none. Side effects: stdout write, mutates the failure flag.
        # Raises nothing itself (serialization errors would propagate).
        if isinstance(event, AutoRetryEndEvent) and event.success:
            self._failed = False
        if (
            isinstance(event, MessageEndEvent)
            and isinstance(event.message, AssistantMessage)
            and event.message.stop_reason == "error"
        ):
            self._failed = True
        typer.echo(event.model_dump_json(by_alias=True, exclude_none=True))

    def finish(self) -> bool:
        # [FUNCTION DOC]
        # Return true when no unrecovered assistant error was seen (failure flag clear), false
        # otherwise. No side effects.
        return not self._failed
