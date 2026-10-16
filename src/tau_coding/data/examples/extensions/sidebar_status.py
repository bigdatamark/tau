# [MODULE DOC]
# Example user extension shipped with Tau's data files: adds a "turns" status
# section to the TUI sidebar that counts completed agent turns.
# Behavior: exposes a setup entry point that the extension loader calls with an
# ExtensionAPI; it registers session_start, turn_end and session_shutdown hooks.
# Works only when the active frontend exposes a supported sidebar; otherwise the
# hooks are harmless no-ops for display.
# Side effects: mutates sidebar sections of the host UI through the context.
"""Add a small, host-framed status section to Tau's TUI sidebar."""

from tau_coding.extensions import ExtensionAPI, ExtensionContext


def setup(tau: ExtensionAPI) -> None:
    # [FUNCTION DOC]
    # Extension entry point. Registers three event handlers on the given extension
    # API and keeps a per-setup turn counter in a closure variable.
    # Inputs: the ExtensionAPI used to register handlers via its "on" method.
    # Outputs: none.
    # Side effects: registers handlers for "session_start" (reset counter, render),
    # "turn_end" (increment counter, render) and "session_shutdown" (remove the
    # "turns" sidebar section). Raises nothing itself.
    """Show and update a turn counter when the active frontend has a sidebar."""
    turn_count = 0

    def show(context: ExtensionContext) -> None:
        # [FUNCTION DOC]
        # HELPER: Renders the current turn counter into the sidebar.
        # Inputs: the extension context; looks up an optional "sidebar" attribute on the
        # context's UI object.
        # Outputs: none.
        # Side effects: if a sidebar exists and reports itself supported, sets the
        # "turns" section titled "extension status" with one Rich-markup line showing
        # the count. Silently does nothing otherwise.
        sidebar = getattr(context.ui, "sidebar", None)
        if sidebar is not None and sidebar.supported:
            sidebar.set_section(
                "turns",
                title="extension status",
                content=[f"[green]{turn_count}[/green] completed turns"],
            )

    def on_session_start(event: object, context: ExtensionContext) -> None:
        # [FUNCTION DOC]
        # Handler for session_start. Ignores the event payload, resets the closure turn
        # counter to zero, then re-renders the sidebar section.
        # Inputs: the event object (unused) and the extension context. Outputs: none.
        # Side effects: mutates the shared counter and the sidebar.
        nonlocal turn_count
        del event
        turn_count = 0
        show(context)

    def on_turn_end(event: object, context: ExtensionContext) -> None:
        # [FUNCTION DOC]
        # Handler for turn_end. Ignores the event, increments the closure turn counter
        # by one and re-renders the sidebar section.
        # Inputs: event (unused) and extension context. Outputs: none.
        # Side effects: mutates the shared counter and the sidebar.
        nonlocal turn_count
        del event
        turn_count += 1
        show(context)

    def on_session_shutdown(event: object, context: ExtensionContext) -> None:
        # [FUNCTION DOC]
        # Handler for session_shutdown. Removes the "turns" section from the sidebar if
        # the UI has a sidebar attribute (does not check "supported" here).
        # Inputs: event (unused) and extension context. Outputs: none.
        # Side effects: mutates the sidebar.
        del event
        sidebar = getattr(context.ui, "sidebar", None)
        if sidebar is not None:
            sidebar.remove_section("turns")

    tau.on("session_start", on_session_start)
    tau.on("turn_end", on_turn_end)
    tau.on("session_shutdown", on_session_shutdown)
