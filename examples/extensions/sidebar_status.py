# [MODULE DOC]
# Example Tau extension that adds a small "extension status" section to the TUI
# sidebar showing how many turns have completed in the current session.
# Demonstrates subscribing to lifecycle events and using the optional sidebar UI
# surface, degrading gracefully when the frontend has no sidebar.
"""Add a small, host-framed status section to Tau's TUI sidebar."""

from tau_coding.extensions import ExtensionAPI, ExtensionContext


def setup(tau: ExtensionAPI) -> None:
    # [FUNCTION DOC]
    # Extension entry point invoked by Tau's extension loader.
    # Inputs: the extension API object used to register event handlers.
    # Behavior: keeps a per-extension turn counter in a closure, and registers
    # handlers for session start (reset counter and show), turn end (increment and
    # show) and session shutdown (remove the sidebar section).
    # Outputs: none. Side effects: registers three event handlers on the API.
    # Raises nothing directly.
    """Show and update a turn counter when the active frontend has a sidebar."""
    turn_count = 0

    def show(context: ExtensionContext) -> None:
        # [FUNCTION DOC]
        # HELPER: Render the current turn count into the sidebar section "turns".
        # Inputs: the extension context, whose UI may or may not expose a sidebar.
        # Behavior: looks up a sidebar attribute on the context UI; if present and it
        # reports itself as supported, sets the "turns" section with a title and one
        # line of Rich-markup content showing the completed turn count.
        # Outputs: none. Side effects: mutates sidebar UI state. No-op without a sidebar.
        sidebar = getattr(context.ui, "sidebar", None)
        if sidebar is not None and sidebar.supported:
            sidebar.set_section(
                "turns",
                title="extension status",
                content=[f"[green]{turn_count}[/green] completed turns"],
            )

    def on_session_start(event: object, context: ExtensionContext) -> None:
        # [FUNCTION DOC]
        # Handler for the session start event: ignores the event payload, resets the
        # shared turn counter to zero and re-renders the sidebar section.
        # Inputs: the event object (unused) and the extension context. Outputs: none.
        # Side effects: mutates the closure counter and the sidebar.
        nonlocal turn_count
        del event
        turn_count = 0
        show(context)

    def on_turn_end(event: object, context: ExtensionContext) -> None:
        # [FUNCTION DOC]
        # Handler for the turn end event: ignores the event, increments the shared turn
        # counter by one and re-renders the sidebar section.
        # Inputs: the event object (unused) and the extension context. Outputs: none.
        # Side effects: mutates the closure counter and the sidebar.
        nonlocal turn_count
        del event
        turn_count += 1
        show(context)

    def on_session_shutdown(event: object, context: ExtensionContext) -> None:
        # [FUNCTION DOC]
        # Handler for the session shutdown event: ignores the event and, if the UI has a
        # sidebar, removes the "turns" section (without checking the supported flag).
        # Inputs: the event object (unused) and the extension context. Outputs: none.
        # Side effects: mutates the sidebar.
        del event
        sidebar = getattr(context.ui, "sidebar", None)
        if sidebar is not None:
            sidebar.remove_section("turns")

    tau.on("session_start", on_session_start)
    tau.on("turn_end", on_turn_end)
    tau.on("session_shutdown", on_session_shutdown)
