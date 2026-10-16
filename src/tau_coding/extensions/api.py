# [MODULE DOC]
# Extension-facing API types and hook payloads for tau_coding extensions (port of Pi's extension
# API). Defines: the agent event type names extensions may subscribe to (plus an "agent_event"
# wildcard) and lifecycle hook names (session_start, session_shutdown, input, tool_call,
# tool_result, project_trust); type aliases for message/tool renderers (returning markup strings
# so extensions need not import Textual) and the component seam (slot widgets, sidebar sections,
# main views, key interceptors, whose widget types are Textual's, referenced only for type
# checking); protocols for main-view handles, the component bridge and the host UI bridge; null
# and stderr UI bridges for print mode; a per-load generation token guarding against stale API
# use after /reload; frozen dataclass hook events/results; and the facades extensions receive
# (sidebar, UI, context, API). Textual is never imported at module load.
"""Extension-facing API types and hook payloads."""

from __future__ import annotations

import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast
from uuid import uuid4

from tau_agent.messages import AgentMessage, ToolResultMessage
from tau_agent.tools import AgentTool, AgentToolResult
from tau_agent.types import JSONValue

if TYPE_CHECKING:
    from textual import events
    from textual.widget import Widget

    from tau_coding.extensions.providers import DynamicProvider
    from tau_coding.extensions.runtime import ExtensionRuntime
    from tau_coding.local_backends import LocalBackend
    from tau_coding.paths import TauPaths
    from tau_coding.project_trust import ProjectTrustResolution
    from tau_coding.session import ModelChoice
    from tau_coding.tui.config import TuiTheme

AGENT_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "agent_start",
        "agent_end",
        "agent_settled",
        "turn_start",
        "turn_end",
        "queue_update",
        "message_start",
        "message_update",
        "message_end",
        "tool_execution_start",
        "tool_execution_update",
        "tool_execution_end",
        "compaction_start",
        "compaction_end",
        "entry_appended",
        "session_info_changed",
        "thinking_level_changed",
        "auto_retry_start",
        "auto_retry_end",
    }
)
AGENT_EVENT_WILDCARD = "agent_event"

LIFECYCLE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "session_start",
        "session_shutdown",
        "input",
        "tool_call",
        "tool_result",
        "project_trust",
    }
)

SessionLifecycleReason = Literal["startup", "reload", "new", "resume", "branch", "quit"]
DeliverAs = Literal["steer", "follow_up"]
NotifyLevel = Literal["info", "warning", "error"]


@dataclass(frozen=True, slots=True)
class CustomMessageView:
    # [CLASS DOC]
    # Immutable read-only view of a custom message given to a message renderer: the custom type
    # (selects the renderer), the LLM-context text content, and optional structured details mapping.
    """Read-only view of a custom message handed to a message renderer.

    Ports Pi's ``CustomMessage``: ``custom_type`` selects the renderer,
    ``content`` is the LLM-context text, and ``details`` carries arbitrary
    structured data the renderer formats.
    """

    custom_type: str
    content: str
    details: Mapping[str, JSONValue] | None = None


@dataclass(frozen=True, slots=True)
class MessageRenderOptions:
    # [CLASS DOC]
    # Immutable options passed to a message renderer; currently only whether the row is expanded.
    """Options passed to a message renderer (ports Pi's ``MessageRenderOptions``)."""

    expanded: bool = False


# A message renderer returns a Rich-markup (or plain) string, NOT a Textual
# widget, so extensions never import the TUI toolkit (deviation from Pi's
# ``Component`` return; see the phase-21 custom-renderer Ruling).
MessageRenderer = Callable[[CustomMessageView, MessageRenderOptions], str]

# Host-side resolver installed into render paths: given a custom message's
# fields and whether it is expanded, return rendered markup or ``None`` to fall
# back to the raw content. Errors are swallowed by the resolver, never raised
# into the frontend.
CustomMessageMarkup = Callable[[str, str, "Mapping[str, JSONValue] | None", bool], "str | None"]

# Host-side resolver installed into render paths: given a tool call's name and
# arguments, return the friendly invocation line from the tool's `render_call`
# or ``None`` to fall back to generic formatting. Errors are swallowed by the
# resolver, never raised into the frontend.
ToolCallMarkup = Callable[[str, "Mapping[str, JSONValue]"], "str | None"]

# Host-side resolver installed into render paths: given the tool name, its
# result, and whether the row is expanded, return the display markup from the tool's
# `render_result` or ``None`` to fall back to the generic result block. Errors
# are swallowed by the resolver, never raised into the frontend.
ToolResultMarkup = Callable[[str, AgentToolResult, bool], "str | None"]

# --- component seam ---------------------------------------------------------
# Widget-hosting capability that lets an extension mount its own Textual widgets
# into host-owned slots and a main-area view, and intercept keys pre-dispatch
# (before the host's priority bindings and the focused widget). The "component"
# type is Textual's own ``Widget`` (referenced only under TYPE_CHECKING so
# print-mode stays import-clean): Textual is deliberately part of the public
# extension contract. Extensions build against the Textual version tau pins; a
# Textual major bump is a coordinated break for core and extensions together.
# History and measured trade-offs: dev-notes/design/component-seam-experiment.md.
# It replaced the older transcript-source seam, which Step 3 removed from core.

Placement = Literal["above_prompt", "below_prompt"]

# Factories run on the UI thread and receive the live theme (theme handoff,
# mirrors Pi's ``(tui, theme) => Component``).
SlotWidgetFactory = Callable[["TuiTheme"], "Widget"]
# A slot widget may be given as a factory or, for the simple case, as a plain
# list of display lines the HOST turns into a widget — this lets an extension
# mount text without importing Textual at all (ports Pi's ``string[]`` form of
# ``setWidget``). Strings are rendered as Rich markup, with a literal-text
# fallback if the markup is malformed.
SlotWidgetContent = Sequence[str] | SlotWidgetFactory
# Sidebar bodies use the same data-first shape: simple sections provide Rich
# display lines without importing Textual; advanced sections provide a widget
# factory receiving the live theme.
SidebarWidgetFactory = Callable[["TuiTheme"], "Widget"]
SidebarContent = Sequence[str] | SidebarWidgetFactory
# The main-view factory also receives the handle so the widget can close itself.
MainViewFactory = Callable[["MainViewHandle", "TuiTheme"], "Widget"]

# Pre-dispatch key hook (ports Pi's ``onTerminalInput``). Returns True to
# consume the key. Fires for every main-screen key regardless of focus; the host
# passes the Textual ``Key`` event and the current prompt text so the handler
# can self-gate (Pi gates on ``getEditorText() === ""``).
KeyInterceptor = Callable[["events.Key", str], bool]


_DEFAULT_THEME: TuiTheme | None = None


def _default_theme() -> TuiTheme:
    # [FUNCTION DOC]
    # HELPER: Return the shared default TUI theme (Tau dark), importing the TUI config lazily on first
    # use and caching it in a module global. Side effect: first call imports TUI config and sets the
    # global. Never raises under normal installs.
    """Return a shared default theme without importing the TUI at module load.

    The import is deferred (and cached) so merely importing the extensions API
    stays free of the Textual/TUI dependency graph; only an extension that
    actually reads ``theme`` in print mode pays for it, and it never raises.
    """
    global _DEFAULT_THEME
    if _DEFAULT_THEME is None:
        from tau_coding.tui.config import TAU_DARK_THEME

        _DEFAULT_THEME = TAU_DARK_THEME
    return _DEFAULT_THEME


class MainViewHandle(Protocol):
    # [CLASS DOC]
    # Protocol for a handle to an open main-area extension view (Pi's overlay handle, trimmed).
    # Implementations must provide close(result) (first close wins; later ones no-op; unmounts the view
    # and restores the transcript), an async wait() resolving to the close result (or None when
    # force-cleared, quarantined, superseded, or already closed), and an is_open property.
    """Handle to an open main-area view (ports Pi's ``OverlayHandle``, trimmed).

    Carries Pi's ``done(result)`` semantics: the factory (or a key interceptor)
    calls ``close(result)`` to tear the view down *and* hand a value back to
    whoever opened it, and the opener awaits :meth:`wait` for that value. This
    is the result-resolution half of Pi's ``ctx.ui.custom<T>``, kept on the
    synchronous open/handle model rather than an ``async`` open.

    ``close()`` unmounts the view and restores the main transcript. It is safe
    to call more than once; the first close wins and later closes are no-ops.
    """

    def close(self, result: object | None = None) -> None:
        # [FUNCTION DOC]
        # Protocol method: close the view and resolve wait() with the given result; idempotent. Body is a stub.
        """Close the view, resolving :meth:`wait` with ``result`` (Pi's ``done``).

        The first close wins: its ``result`` is what :meth:`wait` returns, and
        any later ``close(...)`` is a no-op. Safe to call more than once.
        """
        ...

    async def wait(self) -> object | None:
        # [FUNCTION DOC]
        # Protocol method (async): await teardown and return the close result or None. Body is a stub.
        """Await the view's teardown and return the result passed to ``close``.

        Resolves with the value handed to :meth:`close` (``None`` when closed
        with no result). Also resolves with ``None`` — never hangs — when the
        view is force-cleared on a session rebind, quarantined after a widget
        crash, or superseded by a later ``open_main_view``. Returns immediately
        if the view was already closed before ``wait`` is awaited.
        """
        ...

    # [FUNCTION DOC]
    # Protocol property: whether the view is still open. Body is a stub.
    @property
    def is_open(self) -> bool: ...


class ComponentBridge(Protocol):
    # [CLASS DOC]
    # Protocol for the host widget-hosting capability (context.ui.components). Declares: whether
    # components are supported, the live theme, current prompt text, request_render, set_slot_widget
    # (factory or display lines under a key, None removes, ordered per placement), open_main_view
    # (replaces the transcript in place, returns a handle) and register_key_interceptor (pre-dispatch
    # key hook returning an unsubscribe callable; ctrl+c/ctrl+d are never offered). Null bridges
    # implement all of these as no-ops.
    """Host widget-hosting capability, exposed via ``context.ui.components``.

    Part of what a :class:`UiBridge` provides when a TUI is attached;
    :class:`NullUiBridge`/:class:`StderrUiBridge` implement it as no-ops so an
    extension stays fully functional (just widget-less) in print mode. Check
    :attr:`supports_components` before building widgets.
    """

    @property
    def supports_components(self) -> bool:
        # [FUNCTION DOC]
        # Protocol property: whether the frontend can host extension widgets. Stub.
        """Return whether the frontend can host extension widgets."""
        ...

    @property
    def theme(self) -> TuiTheme:
        # [FUNCTION DOC]
        # Protocol property: the live TUI theme passed to widget factories. Stub.
        """Return the live TUI theme handed to widget factories."""
        ...

    def get_prompt_text(self) -> str:
        # [FUNCTION DOC]
        # Protocol method: return the current prompt-editor text. Stub.
        """Return the current prompt-editor text (Pi's getEditorText).

        Key interceptors receive the prompt text as their second argument;
        this is for reads outside the key path.
        """
        ...

    def request_render(self) -> None:
        # [FUNCTION DOC]
        # Protocol method: ask the host to re-render mounted extension widgets. Stub.
        """Ask the host to re-render mounted extension widgets (Pi's requestRender)."""
        ...

    def set_slot_widget(
        self,
        key: str,
        content: SlotWidgetContent | None,
        *,
        placement: Placement = "above_prompt",
    ) -> None:
        # [FUNCTION DOC]
        # Protocol method: mount, replace, or (content None) remove a slot widget under a key at the given
        # placement (default above the prompt). Content is a theme-taking widget factory or Rich-markup lines. Stub.
        """Mount an extension widget into a prompt-adjacent slot under ``key``.

        ``content`` is either a ``factory(theme) -> Widget`` callable or a plain
        list of display lines (``Sequence[str]``) the host renders as Rich
        markup — the string form lets simple extensions avoid importing Textual.
        Passing ``content=None`` unmounts and forgets that key. Re-setting a key
        replaces its content. Multiple keys per placement mount in call order.
        Placement defaults to ``"above_prompt"`` (Pi's ``aboveEditor``).
        """
        ...

    def open_main_view(self, factory: MainViewFactory) -> MainViewHandle:
        # [FUNCTION DOC]
        # Protocol method: mount factory(handle, theme) as the main-area view and return its handle. Stub.
        """Mount ``factory(handle, theme)`` as a full main-area view.

        The widget replaces the main transcript in place (a display-toggled
        sibling, not a modal screen), so prompt-adjacent widgets such as slot
        widgets stay visible. ``handle.close(result)`` restores the transcript
        and resolves ``await handle.wait()`` with ``result`` (Pi's ``done``),
        so the opener can show a view and get an answer back.
        """
        ...

    def register_key_interceptor(self, handler: KeyInterceptor) -> Callable[[], None]:
        # [FUNCTION DOC]
        # Protocol method: register a key interceptor (key event, prompt text) -> consumed flag; returns an
        # unsubscribe callable. Stub.
        """Register a pre-dispatch key hook; return an unsubscribe callable.

        Ports Pi's ``onTerminalInput``. The handler sees a key before the
        host's app-level priority bindings and before the focused widget, so it
        can own navigation keys the host otherwise reserves. It fires for EVERY
        main-screen key regardless of focus (never while a modal screen is on
        top), so the handler MUST self-gate and return ``True`` only for keys it
        consumes.

        The host's hard interrupt/exit keys (``ctrl+c`` and ``ctrl+d``) are
        reserved: the interceptor is never consulted for them and cannot
        consume them, so a bug in the handler can never swallow the session's
        escape hatches. All other keys — escape, enter, arrows, tab — remain
        interceptable.
        """
        ...


class ExtensionSidebar:
    # [CLASS DOC]
    # Extension-owned facade over host-framed TUI sidebar sections, scoped to one extension name and
    # load generation. Keys and titles are whitespace-stripped and must be non-empty. Operations are
    # silently ignored when the UI bridge lacks sidebar support. All members assert the generation is active.
    """Extension-owned facade for host-framed TUI sidebar sections.

    Sections are isolated by extension name and stable key. Re-setting a key
    updates it without changing its position; removing and re-adding it appends
    it after the remaining extension sections. The host owns headings,
    separators, width, scrolling, sidebar placement, and lifecycle cleanup.
    """

    def __init__(
        self,
        runtime: ExtensionRuntime,
        extension_name: str,
        generation: ExtensionGeneration,
    ) -> None:
        # [FUNCTION DOC]
        # Store the runtime, owning extension name and generation token. No side effects.
        self._runtime = runtime
        self._extension_name = extension_name
        self._generation = generation

    @property
    def supported(self) -> bool:
        # [FUNCTION DOC]
        # Property: assert the generation is active (raises ExtensionError if stale), then return the
        # runtime UI bridge's supports_sidebar attribute as a bool (False if absent).
        """Return whether the active frontend can display sidebar sections."""
        self._generation.assert_active()
        return bool(getattr(self._runtime.ui, "supports_sidebar", False))

    def set_section(self, key: str, *, title: str, content: SidebarContent) -> None:
        # [FUNCTION DOC]
        # Add or replace a sidebar section. Inputs: key, keyword title and content (lines or theme factory).
        # Asserts the generation is active; raises ExtensionError for empty (after strip) key or title.
        # Returns without action if unsupported; otherwise calls the bridge's set_sidebar_section (if it
        # exists) with extension name, stripped key, stripped title and content. Returns nothing.
        """Add or replace a host-framed sidebar section under stable ``key``."""
        self._generation.assert_active()
        normalized_key = key.strip()
        normalized_title = title.strip()
        if not normalized_key:
            raise ExtensionError("sidebar section key must not be empty")
        if not normalized_title:
            raise ExtensionError("sidebar section title must not be empty")
        if not self.supported:
            return
        setter = getattr(self._runtime.ui, "set_sidebar_section", None)
        if setter is not None:
            setter(
                self._extension_name,
                normalized_key,
                title=normalized_title,
                content=content,
            )

    def remove_section(self, key: str) -> None:
        # [FUNCTION DOC]
        # Remove this extension's sidebar section under the key. Asserts active; raises ExtensionError for
        # an empty key; no-op if unsupported; otherwise calls the bridge's remove_sidebar_section if present.
        """Remove this extension's sidebar section under ``key``, if present."""
        self._generation.assert_active()
        normalized_key = key.strip()
        if not normalized_key:
            raise ExtensionError("sidebar section key must not be empty")
        if not self.supported:
            return
        remover = getattr(self._runtime.ui, "remove_sidebar_section", None)
        if remover is not None:
            remover(self._extension_name, normalized_key)


class ExtensionError(RuntimeError):
    # [CLASS DOC]
    # Exception (a RuntimeError) raised when an extension misuses the API, e.g. stale handles after
    # reload, empty sidebar keys, or actions before binding.
    """Raised when an extension misuses the API (e.g. actions before binding)."""


_STALE_MESSAGE = (
    "extension instance is stale after reload: state captured before /reload"
    " (a saved `tau` API object, context, or ui handle) must not be reused;"
    " the reloaded extension received a fresh API in its new setup()"
)


class ExtensionGeneration:
    # [CLASS DOC]
    # Liveness token for one extension load generation (Pi's assertActive/invalidate). Holds a random
    # hex id and an optional stale message; only /reload invalidates it, session rebinds do not.
    """Liveness token for one extension load generation.

    Ports Pi's ``assertActive``/``invalidate`` staleness guard: every
    :class:`ExtensionAPI` method and every :class:`ExtensionContext`/
    :class:`ExtensionUi` read checks this token before touching the runtime,
    so state captured before a `/reload` fails loudly instead of silently
    acting against the new registration set. Only reload invalidates; session
    rebinding (resume/new/branch) keeps the generation alive by design (see
    the phase-21 lifecycle Ruling).
    """

    __slots__ = ("_id", "_stale_message")

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Create a new active generation with a fresh random hex id and no stale message.
        self._id = uuid4().hex
        self._stale_message: str | None = None

    @property
    def id(self) -> str:
        # [FUNCTION DOC]
        # Property: the generation's process-local hex id.
        """Return this generation's stable process-local identity."""
        return self._id

    @property
    def active(self) -> bool:
        # [FUNCTION DOC]
        # Property: True while no stale message has been set.
        """Return whether this generation is still the live one."""
        return self._stale_message is None

    def invalidate(self, message: str | None = None) -> None:
        # [FUNCTION DOC]
        # Mark the generation stale with the given message or the default stale-after-reload message;
        # only the first invalidation's message is kept. Side effect: mutates the token.
        """Mark this generation stale; the first message wins (Pi parity)."""
        if self._stale_message is None:
            self._stale_message = message or _STALE_MESSAGE

    def assert_active(self) -> None:
        # [FUNCTION DOC]
        # Raise ExtensionError carrying the stale message if invalidated; otherwise return nothing.
        """Raise :class:`ExtensionError` when this generation is stale."""
        if self._stale_message is not None:
            raise ExtensionError(self._stale_message)


@dataclass(frozen=True, slots=True)
class TurnStartEvent:
    # [CLASS DOC]
    # Immutable Pi-shaped extension event for the start of a turn: zero-based turn index, millisecond
    # timestamp, and a fixed type "turn_start" (not an init argument).
    """Pi-shaped extension event fired at the start of a turn.

    The portable agent event intentionally has no session metadata. The coding
    session adapter adds the zero-based turn index and millisecond timestamp
    before dispatching the event to extensions.
    """

    turn_index: int
    timestamp: int
    type: Literal["turn_start"] = field(default="turn_start", init=False)


@dataclass(frozen=True, slots=True)
class TurnEndEvent:
    # [CLASS DOC]
    # Immutable Pi-shaped extension event after a turn: turn index, the assistant/agent message, the
    # list of tool result messages, and fixed type "turn_end".
    """Pi-shaped extension event fired after one assistant/tool turn."""

    turn_index: int
    message: AgentMessage
    tool_results: list[ToolResultMessage]
    type: Literal["turn_end"] = field(default="turn_end", init=False)


@dataclass(frozen=True, slots=True)
class SessionStartEvent:
    # [CLASS DOC]
    # Immutable payload for session_start: the lifecycle reason (startup, reload, new, resume, branch, quit).
    """Payload for the `session_start` lifecycle event."""

    reason: SessionLifecycleReason


@dataclass(frozen=True, slots=True)
class SessionShutdownEvent:
    # [CLASS DOC]
    # Immutable payload for session_shutdown: the lifecycle reason.
    """Payload for the `session_shutdown` lifecycle event."""

    reason: SessionLifecycleReason


@dataclass(frozen=True, slots=True)
class InputEvent:
    # [CLASS DOC]
    # Immutable payload for the input hook: raw prompt text before expansion, its source ("interactive"
    # default or "extension"), and the streaming behavior ("steer"/"follow_up") when the agent is
    # running, or None on the idle path.
    """Payload for the `input` hook: raw user prompt text, before expansion.

    Mirrors Pi's `InputEvent`. `source` says where the input came from:
    ``"interactive"`` for TUI/print-mode user input, ``"extension"`` for a turn
    an extension started via ``send_user_message``/``send_custom_message``.
    `streaming_behavior` says how the input will be queued when the agent is
    mid-run (``"steer"``/``"follow_up"``), and is ``None`` on the idle prompt
    path.

    Pi's `images` field is omitted (Tau has no image input yet) and Pi's
    ``"rpc"`` source is omitted (Tau has no RPC mode). Both defaults keep
    existing handlers that read only ``.text`` working unchanged.
    """

    text: str
    source: Literal["interactive", "extension"] = "interactive"
    streaming_behavior: Literal["steer", "follow_up"] | None = None


@dataclass(frozen=True, slots=True)
class InputHookResult:
    # [CLASS DOC]
    # Immutable result from an input hook: action "continue" (default, unchanged), "transform" (replace
    # with text; chains across handlers) or "handled" (consume input, optionally show message).
    """Result of an `input` hook handler.

    `action="continue"` leaves the text unchanged, `"transform"` replaces it
    with `text` (transforms chain across handlers), and `"handled"` consumes
    the input entirely, optionally showing `message` to the user.
    """

    action: Literal["continue", "transform", "handled"] = "continue"
    text: str | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCallHookEvent:
    # [CLASS DOC]
    # Immutable payload for the tool_call hook before execution: tool name and argument mapping (no call id).
    """Payload for the `tool_call` hook, before a tool executes.

    Carries no tool-call id: the hook runs inside the tool executor seam,
    which the agent loop invokes without the id. Use the observation events
    (`tool_execution_start`/`tool_execution_end`) for id correlation.
    """

    tool_name: str
    arguments: Mapping[str, JSONValue]


@dataclass(frozen=True, slots=True)
class ToolCallHookResult:
    # [CLASS DOC]
    # Immutable result from a tool_call hook: block flag with optional reason (wins and short-circuits),
    # or replacement arguments to rewrite the call.
    """Result of a `tool_call` hook handler.

    Set `block=True` (with an optional `reason`) to prevent execution, or
    return replacement `arguments` to rewrite the call. Blocking wins over
    argument rewrites and short-circuits remaining handlers.
    """

    block: bool = False
    reason: str | None = None
    arguments: Mapping[str, JSONValue] | None = None


@dataclass(frozen=True, slots=True)
class ToolResultHookEvent:
    # [CLASS DOC]
    # Immutable payload for the tool_result hook: tool name, arguments and the tool's result.
    """Payload for the `tool_result` hook, after a tool executes."""

    tool_name: str
    arguments: Mapping[str, JSONValue]
    result: AgentToolResult


@dataclass(frozen=True, slots=True)
class ToolResultHookResult:
    # [CLASS DOC]
    # Immutable result from a tool_result hook: optional replacement text content and/or details mapping.
    """Result of a `tool_result` hook handler; set fields to override."""

    content: str | None = None
    details: dict[str, JSONValue] | None = None


ExtensionHandler = Callable[[object, "ExtensionContext"], object | Awaitable[object]]
# Command handlers are sync-only: the slash-command path (CommandRegistry ->
# CodingSession.handle_command -> TUI submit) is synchronous end to end.
ExtensionCommandHandler = Callable[["str", "ExtensionCommandContext"], "str | None"]


@dataclass(frozen=True, slots=True)
class ExtensionRuntimeDiagnostic:
    # [CLASS DOC]
    # Immutable record of a runtime failure in an extension handler: extension name, event name, message.
    """A runtime failure raised by an extension handler."""

    extension: str
    event: str
    message: str


class UiBridge(Protocol):
    # [CLASS DOC]
    # Protocol for host-provided UI capabilities used by extensions: has_ui, notify, async select/
    # confirm/input dialogs (with optional timeout seconds auto-dismissing to the no-op default), plus
    # the component seam members (see ComponentBridge) and sidebar section management and
    # clear_components (host-driven teardown on reload/session rebind).
    """Host-provided UI capabilities available to extensions.

    Dialog methods (`select`/`confirm`/`input`) are async and mirror Pi's
    `ctx.ui`. Without an interactive frontend they return the Pi no-op
    defaults (`None`/`False`/`None`). `timeout` (seconds) auto-dismisses a
    dialog with the no-op default; `None` waits indefinitely.
    """

    @property
    def has_ui(self) -> bool:
        # [FUNCTION DOC]
        # Protocol property: whether an interactive UI is attached. Stub.
        """Return whether an interactive UI is attached."""
        ...

    def notify(self, message: str, level: NotifyLevel = "info") -> None:
        # [FUNCTION DOC]
        # Protocol method: show a notification at level info/warning/error. Stub.
        """Show a notification to the user (no-op without a UI)."""
        ...

    async def select(
        self,
        title: str,
        options: Sequence[str],
        *,
        timeout: float | None = None,
    ) -> str | None:
        # [FUNCTION DOC]
        # Protocol method (async): show a picker over options; return the chosen option or None. Stub.
        """Show a picker; return the chosen option, or None on cancel."""
        ...

    async def confirm(
        self,
        title: str,
        message: str,
        *,
        timeout: float | None = None,
    ) -> bool:
        # [FUNCTION DOC]
        # Protocol method (async): show a confirmation; return True only if confirmed. Stub.
        """Show a confirmation; return True only if confirmed."""
        ...

    async def input(
        self,
        title: str,
        placeholder: str = "",
        *,
        timeout: float | None = None,
    ) -> str | None:
        # [FUNCTION DOC]
        # Protocol method (async): show a text prompt with placeholder; return text or None. Stub.
        """Show a text prompt; return the entered text, or None on cancel."""
        ...

    # -- component seam -- see ComponentBridge for docs -----------------------

    @property
    def supports_components(self) -> bool:
        # [FUNCTION DOC]
        # Protocol property: whether widgets can be hosted. Stub.
        """Return whether the frontend can host extension widgets."""
        ...

    @property
    def theme(self) -> TuiTheme:
        # [FUNCTION DOC]
        # Protocol property: live TUI theme. Stub.
        """Return the live TUI theme handed to widget factories."""
        ...

    def get_prompt_text(self) -> str:
        # [FUNCTION DOC]
        # Protocol method: current prompt text. Stub.
        """Return the current prompt-editor text."""
        ...

    def request_render(self) -> None:
        # [FUNCTION DOC]
        # Protocol method: request re-render of extension widgets. Stub.
        """Ask the host to re-render mounted extension widgets."""
        ...

    def set_slot_widget(
        self,
        key: str,
        content: SlotWidgetContent | None,
        *,
        placement: Placement = "above_prompt",
    ) -> None:
        # [FUNCTION DOC]
        # Protocol method: mount/replace/remove a slot widget by key and placement. Stub.
        """Mount or remove an extension slot widget (factory or string lines)."""
        ...

    def open_main_view(self, factory: MainViewFactory) -> MainViewHandle:
        # [FUNCTION DOC]
        # Protocol method: open a main-area view and return its handle. Stub.
        """Open a full main-area extension view."""
        ...

    def register_key_interceptor(self, handler: KeyInterceptor) -> Callable[[], None]:
        # [FUNCTION DOC]
        # Protocol method: register a key interceptor; returns an unsubscribe callable. Stub.
        """Register a pre-dispatch key hook; return an unsubscribe callable.

        See the base bridge protocol: the handler is consulted before the
        host's priority bindings and the focused widget, fires for every
        main-screen key regardless of focus, and must self-gate. The hard
        interrupt/exit keys (``ctrl+c`` and ``ctrl+d``) are reserved and never
        reach the interceptor.
        """
        ...

    @property
    def supports_sidebar(self) -> bool:
        # [FUNCTION DOC]
        # Protocol property: whether sidebar sections can be hosted. Stub.
        """Return whether this frontend can host extension sidebar sections."""
        ...

    def set_sidebar_section(
        self,
        extension_name: str,
        key: str,
        *,
        title: str,
        content: SidebarContent,
    ) -> None:
        # [FUNCTION DOC]
        # Protocol method: add/replace a sidebar section owned by (extension name, key). Stub.
        """Add or replace one extension-owned, host-framed sidebar section."""
        ...

    def remove_sidebar_section(self, extension_name: str, key: str) -> None:
        # [FUNCTION DOC]
        # Protocol method: remove a sidebar section owned by (extension name, key) if present. Stub.
        """Remove one extension-owned sidebar section, if present."""
        ...

    def clear_components(self) -> None:
        # [FUNCTION DOC]
        # Protocol method: tear down all extension UI (slots, main view resolving waits with None, key
        # interceptors) on reload or session rebind. Stub.
        """Tear down all extension-owned UI (host-driven, not for extensions).

        The runtime drives this on `/reload` (the stale generation's widgets
        and interceptors must not outlive its registrations) and on session
        rebinds (resume/new), before ``session_start`` fires so handlers can
        re-mount. Slot widgets and any main view are unmounted (a pending
        ``wait()`` resolves with ``None``) and key interceptors are dropped.
        """
        ...


class _DeadMainViewHandle:
    # [CLASS DOC]
    # HELPER: No-op main-view handle returned when no UI can host a view; never open.
    """A no-op main-view handle returned when no UI can host a view."""

    def close(self, result: object | None = None) -> None:
        # [FUNCTION DOC]
        # HELPER: Ignore the close request and result.
        """Do nothing: there is no view to close (``result`` is ignored)."""

    async def wait(self) -> object | None:
        # [FUNCTION DOC]
        # HELPER: Async; return None immediately.
        """Return None immediately: a dead handle never opens a view."""
        return None

    @property
    def is_open(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: Property; always False.
        """Return False: a dead handle is never open."""
        return False


class NullUiBridge:
    # [CLASS DOC]
    # UI bridge used when no interactive frontend is attached: dialogs return Pi no-op defaults
    # (None/False/None), notifications are ignored, components/sidebar unsupported and no-ops, theme
    # is the lazily loaded default theme, prompt text is empty, main views get a dead handle.
    """UI bridge used when no interactive frontend is attached."""

    @property
    def has_ui(self) -> bool:
        # [FUNCTION DOC]
        # Property: always False.
        """Return False: print mode has no interactive UI."""
        return False

    def notify(self, message: str, level: NotifyLevel = "info") -> None:
        # [FUNCTION DOC]
        # Ignore the notification.
        """Ignore notifications without a UI."""

    async def select(
        self,
        title: str,
        options: Sequence[str],
        *,
        timeout: float | None = None,
    ) -> str | None:
        # [FUNCTION DOC]
        # Async: return None without showing anything.
        """Return None: no UI to pick from (Pi no-op default)."""
        return None

    async def confirm(
        self,
        title: str,
        message: str,
        *,
        timeout: float | None = None,
    ) -> bool:
        # [FUNCTION DOC]
        # Async: return False without showing anything.
        """Return False: no UI to confirm with (Pi no-op default)."""
        return False

    async def input(
        self,
        title: str,
        placeholder: str = "",
        *,
        timeout: float | None = None,
    ) -> str | None:
        # [FUNCTION DOC]
        # Async: return None without showing anything.
        """Return None: no UI to enter text into (Pi no-op default)."""
        return None

    # -- component seam -------------------------------------------------------

    @property
    def supports_components(self) -> bool:
        # [FUNCTION DOC]
        # Property: always False.
        """Return False: print mode cannot host widgets."""
        return False

    @property
    def theme(self) -> TuiTheme:
        # [FUNCTION DOC]
        # Property: return the default theme (lazy import, cached).
        """Return a usable default theme (never raise; print-mode may read it)."""
        return _default_theme()

    def get_prompt_text(self) -> str:
        # [FUNCTION DOC]
        # Return an empty string.
        """Return an empty prompt: there is no editor in print mode."""
        return ""

    def request_render(self) -> None:
        # [FUNCTION DOC]
        # Do nothing.
        """Do nothing: there is no frontend to re-render."""

    def set_slot_widget(
        self,
        key: str,
        content: SlotWidgetContent | None,
        *,
        placement: Placement = "above_prompt",
    ) -> None:
        # [FUNCTION DOC]
        # Do nothing (no slot to mount into).
        """Do nothing: there is no slot to mount into."""

    def open_main_view(self, factory: MainViewFactory) -> MainViewHandle:
        # [FUNCTION DOC]
        # Return a new dead main-view handle; the factory is never called.
        """Return a dead handle: there is no main area to host a view."""
        return _DeadMainViewHandle()

    def register_key_interceptor(self, handler: KeyInterceptor) -> Callable[[], None]:
        # [FUNCTION DOC]
        # Return a no-op unsubscribe callable; the handler is never registered.
        """Return a no-op unsubscribe: no key stream to intercept."""
        return lambda: None

    @property
    def supports_sidebar(self) -> bool:
        # [FUNCTION DOC]
        # Property: always False.
        """Return False: print mode has no interactive sidebar."""
        return False

    def set_sidebar_section(
        self,
        extension_name: str,
        key: str,
        *,
        title: str,
        content: SidebarContent,
    ) -> None:
        # [FUNCTION DOC]
        # Do nothing (no sidebar).
        """Do nothing: there is no sidebar in print mode."""

    def remove_sidebar_section(self, extension_name: str, key: str) -> None:
        # [FUNCTION DOC]
        # Do nothing (no sidebar).
        """Do nothing: there is no sidebar in print mode."""

    def clear_components(self) -> None:
        # [FUNCTION DOC]
        # Do nothing (nothing mounted).
        """Do nothing: no components were ever mounted."""


class StderrUiBridge(NullUiBridge):
    # [CLASS DOC]
    # Print-mode UI bridge: identical to the null bridge except notifications are written to stderr.
    """UI bridge that writes extension notifications to stderr (print mode).

    Inherits the Pi no-op dialog defaults from `NullUiBridge`; only
    `notify` is observable in print mode.
    """

    def notify(self, message: str, level: NotifyLevel = "info") -> None:
        # [FUNCTION DOC]
        # Print "[extension:LEVEL] message" to standard error. Side effect: stderr output.
        """Print the notification to stderr."""
        print(f"[extension:{level}] {message}", file=sys.stderr)


@dataclass(frozen=True, slots=True)
class ExtensionCommandContext:
    # [CLASS DOC]
    # Immutable context for extension slash-command handlers: command name, raw argument string and the
    # extension's API object.
    """Context passed to extension slash-command handlers."""

    name: str
    args: str
    api: ExtensionAPI


class ExtensionUi:
    # [CLASS DOC]
    # UI facade exposed to extensions as context.ui (Pi's ctx.ui). Delegates every call to the runtime's
    # current UI bridge, asserting the owning generation is active first (raising ExtensionError when
    # stale). Also exposes the bridge as a components property and a sidebar facade property.
    """Interactive UI facade exposed to extensions as `context.ui`.

    Mirrors Pi's `ctx.ui`: async `select`/`confirm`/`input` dialogs plus a
    synchronous `notify`. Every call delegates to the host UI bridge, which
    returns the Pi no-op defaults when no interactive frontend is attached.
    Every member (trivial reads included, matching Pi) asserts the owning
    load generation is still active and raises :class:`ExtensionError` when
    the facade was captured before a `/reload`.
    """

    def __init__(
        self,
        runtime: ExtensionRuntime,
        generation: ExtensionGeneration | None = None,
        *,
        extension_name: str = "",
    ) -> None:
        # [FUNCTION DOC]
        # Store the runtime and generation (a fresh generation when None) and build the sidebar facade for
        # the given extension name (default empty).
        self._runtime = runtime
        self._generation = generation if generation is not None else ExtensionGeneration()
        self._sidebar = ExtensionSidebar(runtime, extension_name, self._generation)

    @property
    def has_ui(self) -> bool:
        # [FUNCTION DOC]
        # Property: assert active, then return whether the runtime UI bridge has an interactive UI.
        """Return whether an interactive UI is attached."""
        self._generation.assert_active()
        return self._runtime.ui.has_ui

    async def select(
        self,
        title: str,
        options: Sequence[str],
        *,
        timeout: float | None = None,
    ) -> str | None:
        # [FUNCTION DOC]
        # Async: assert active, then delegate a picker to the bridge; returns the choice or None.
        """Prompt the user to pick an option; None on cancel/no UI."""
        self._generation.assert_active()
        return await self._runtime.ui.select(title, options, timeout=timeout)

    async def confirm(
        self,
        title: str,
        message: str,
        *,
        timeout: float | None = None,
    ) -> bool:
        # [FUNCTION DOC]
        # Async: assert active, then delegate a confirmation dialog; returns True only if confirmed.
        """Ask the user to confirm; True only if confirmed."""
        self._generation.assert_active()
        return await self._runtime.ui.confirm(title, message, timeout=timeout)

    async def input(
        self,
        title: str,
        placeholder: str = "",
        *,
        timeout: float | None = None,
    ) -> str | None:
        # [FUNCTION DOC]
        # Async: assert active, then delegate a text input dialog; returns text or None.
        """Prompt the user for text; None on cancel/no UI."""
        self._generation.assert_active()
        return await self._runtime.ui.input(title, placeholder, timeout=timeout)

    @property
    def sidebar(self) -> ExtensionSidebar:
        # [FUNCTION DOC]
        # Property: assert active, return this extension's sidebar facade.
        """Return this extension's host-framed sidebar capability."""
        self._generation.assert_active()
        return self._sidebar

    @property
    def components(self) -> ComponentBridge:
        # [FUNCTION DOC]
        # Property: assert active, return the runtime's UI bridge viewed as a component bridge (no copy).
        """Return the host widget-hosting capability.

        Straight pass-through to the installed UI bridge, which implements the
        :class:`ComponentBridge` members (the TUI hosts real widgets; the
        print-mode bridges are no-ops with ``supports_components == False``).
        Gate widget work on ``context.ui.components.supports_components``.
        A stale facade raises here, before the bridge is ever reachable.
        """
        self._generation.assert_active()
        return cast("ComponentBridge", self._runtime.ui)

    def notify(self, message: str, level: NotifyLevel = "info") -> None:
        # [FUNCTION DOC]
        # Assert active, then forward the notification and level to the UI bridge. Side effect: UI/stderr output.
        """Show a notification in the UI, if one is attached."""
        self._generation.assert_active()
        self._runtime.ui.notify(message, level)


class ExtensionContext:
    # [CLASS DOC]
    # Read-only session context exposed to extensions (Pi's ctx). Every property asserts the generation
    # is active (ExtensionError when stale) then reads from the runtime's session view, paths or UI.
    """Read-only session context exposed to extensions.

    Every property (trivial reads included, matching Pi's context getters)
    asserts the owning load generation is still active, so a context captured
    before a `/reload` raises :class:`ExtensionError` instead of reading the
    reloaded world.
    """

    def __init__(
        self,
        runtime: ExtensionRuntime,
        generation: ExtensionGeneration | None = None,
        *,
        extension_name: str = "",
    ) -> None:
        # [FUNCTION DOC]
        # Store runtime and generation (fresh when None) and build the UI facade for the extension name.
        self._runtime = runtime
        self._generation = generation if generation is not None else ExtensionGeneration()
        self._ui = ExtensionUi(runtime, self._generation, extension_name=extension_name)

    @property
    def cwd(self) -> Path:
        # [FUNCTION DOC]
        # Property: session working directory.
        """Return the session working directory."""
        self._generation.assert_active()
        return self._runtime.session_view.cwd

    @property
    def paths(self) -> TauPaths:
        # [FUNCTION DOC]
        # Property: resolved Tau filesystem paths from the runtime.
        """Return the resolved Tau filesystem paths for this session."""
        self._generation.assert_active()
        return self._runtime.paths

    @property
    def model(self) -> str:
        # [FUNCTION DOC]
        # Property: active model name.
        """Return the active model name."""
        self._generation.assert_active()
        return self._runtime.session_view.model

    @property
    def provider_name(self) -> str:
        # [FUNCTION DOC]
        # Property: active provider name.
        """Return the active provider name."""
        self._generation.assert_active()
        return self._runtime.session_view.provider_name

    @property
    def available_providers(self) -> tuple[str, ...]:
        # [FUNCTION DOC]
        # Property: tuple of provider ids the session can call.
        """Return provider IDs that the active session can call."""
        self._generation.assert_active()
        return tuple(self._runtime.session_view.available_providers)

    @property
    def available_model_choices(self) -> tuple[ModelChoice, ...]:
        # [FUNCTION DOC]
        # Property: the session view's available provider/model choices (cast, not copied).
        """Return provider/model IDs from the active session's runtime view."""
        self._generation.assert_active()
        return cast("tuple[ModelChoice, ...]", self._runtime.session_view.available_model_choices)

    @property
    def inference_provider(self) -> str | None:
        # [FUNCTION DOC]
        # Property: Hugging Face inference-provider pin or None.
        """Return the active Hugging Face inference-provider pin, if any."""
        self._generation.assert_active()
        return self._runtime.session_view.inference_provider

    @property
    def inference_provider_mode(self) -> str:
        # [FUNCTION DOC]
        # Property: Hugging Face routing mode string (automatic or fixed).
        """Return whether Hugging Face routing is automatic or explicitly fixed."""
        self._generation.assert_active()
        return self._runtime.session_view.inference_provider_mode

    @property
    def session_id(self) -> str | None:
        # [FUNCTION DOC]
        # Property: current session id or None.
        """Return the current session id, if the session is indexed."""
        self._generation.assert_active()
        return self._runtime.session_view.session_id

    @property
    def session_name(self) -> str | None:
        # [FUNCTION DOC]
        # Property: session display name or None.
        """Return the session's human-friendly name, if it has one."""
        self._generation.assert_active()
        return self._runtime.session_view.session_name

    @property
    def thinking_level(self) -> str:
        # [FUNCTION DOC]
        # Property: active thinking level for future turns.
        """Return the active thinking mode for future turns."""
        self._generation.assert_active()
        return self._runtime.session_view.thinking_level

    @property
    def system_prompt(self) -> str:
        # [FUNCTION DOC]
        # Property: active system prompt text.
        """Return the active system prompt."""
        self._generation.assert_active()
        return self._runtime.session_view.system_prompt

    @property
    def project_trust_resolution(self) -> ProjectTrustResolution | None:
        # [FUNCTION DOC]
        # Property: the frozen project-trust resolution snapshot, or None if trust was never resolved.
        """Return the completed project-input trust decision for the session cwd.

        The resolution is a frozen snapshot with the ``trusted`` decision plus
        ``source``/``saved_path`` diagnostics. It is ``None`` only for sessions
        that never resolved trust (for example a bare ``CodingSession`` built
        outside ``CodingSession.load``). During a ``session_start("reload")``
        hook it still reflects the outgoing generation's decision, like every
        other session property read at that point.
        """
        self._generation.assert_active()
        return self._runtime.session_view.project_trust_resolution

    @property
    def project_trusted(self) -> bool:
        # [FUNCTION DOC]
        # Property: True only when a trust resolution exists and is trusted.
        """Return whether the session cwd's project inputs were approved.

        A host that constructs child ``CodingSession``s (e.g. subagents) should
        forward this as ``CodingSessionConfig(trust_override="approve")`` when
        ``True`` and ``"decline"`` when ``False``, so children inherit the
        user's decision instead of re-resolving headlessly as untrusted.
        """
        self._generation.assert_active()
        resolution = self._runtime.session_view.project_trust_resolution
        return resolution is not None and resolution.trusted

    @property
    def is_running(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether an agent run is active.
        """Return whether an agent run is currently active."""
        self._generation.assert_active()
        return self._runtime.session_view.is_running

    @property
    def transcript(self) -> tuple[AgentMessage, ...]:
        # [FUNCTION DOC]
        # Property: tuple of deep copies of the active-branch messages (summaries already folded in), so
        # mutation cannot affect the live session.
        """Return the active-path parent conversation as read-only copies.

        Mirrors the read access Pi extensions get via
        ``ctx.sessionManager.getBranch()``: the user/assistant/tool messages on
        the current branch, with compaction and branch summaries already folded
        in as ``UserMessage`` entries (Tau has no separate summary message
        type). Each message is deep-copied so an extension mutating a returned
        object cannot corrupt the live session transcript.
        """
        self._generation.assert_active()
        messages = self._runtime.session_view.messages
        return tuple(message.model_copy(deep=True) for message in messages)

    @property
    def has_ui(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether an interactive UI is attached.
        """Return whether an interactive UI is attached."""
        self._generation.assert_active()
        return self._runtime.ui.has_ui

    @property
    def ui(self) -> ExtensionUi:
        # [FUNCTION DOC]
        # Property: the UI facade (dialogs, notify, components, sidebar).
        """Return the interactive UI facade (Pi's `ctx.ui`).

        Use `await context.ui.select/confirm/input(...)` to drive dialogs.
        Because command handlers are sync (see the docs), a `/command` that
        needs a dialog should spawn a loop task that awaits `context.ui`.
        """
        self._generation.assert_active()
        return self._ui


class ExtensionAPI:
    # [CLASS DOC]
    # The object passed to each extension's setup(tau) entry point. Holds the runtime, extension name,
    # source id (default "extension-name:NAME"), generation and context. Every member asserts the
    # generation is active, then delegates registration/actions to the runtime under the source id.
    """The object handed to each extension's `setup(tau)` entry point.

    Every method and property asserts the load generation first (Pi's
    ``assertActive`` parity): after `/reload` replaces the registration set,
    a `tau` object captured by the previous instance raises
    :class:`ExtensionError` on any use instead of silently acting against
    the new world.
    """

    def __init__(
        self,
        runtime: ExtensionRuntime,
        extension_name: str,
        generation: ExtensionGeneration | None = None,
        *,
        source_id: str | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Store runtime, name, source id (or derived default), generation (fresh when None) and build the context.
        self._runtime = runtime
        self._extension_name = extension_name
        self._source_id = source_id or f"extension-name:{extension_name}"
        self._generation = generation if generation is not None else ExtensionGeneration()
        self._context = ExtensionContext(
            runtime,
            self._generation,
            extension_name=extension_name,
        )

    @property
    def name(self) -> str:
        # [FUNCTION DOC]
        # Property: extension name.
        """Return this extension's name."""
        self._generation.assert_active()
        return self._extension_name

    @property
    def context(self) -> ExtensionContext:
        # [FUNCTION DOC]
        # Property: the read-only session context.
        """Return read-only session context."""
        self._generation.assert_active()
        return self._context

    def set_inference_provider(self, route: str | None) -> str:
        # [FUNCTION DOC]
        # Set (route) or reset (None) the Hugging Face inference route via the session view; returns its
        # description string. Side effect: changes session routing.
        """Select or reset the active Hugging Face session route."""
        self._generation.assert_active()
        return self._runtime.session_view.set_inference_provider(route)

    def register_tool(self, tool: AgentTool) -> None:
        # [FUNCTION DOC]
        # Register an agent tool with the runtime (first registration per name wins). Side effect: runtime state.
        """Register an agent tool (first registration per name wins)."""
        self._generation.assert_active()
        self._runtime.register_tool(self._source_id, self._extension_name, tool)

    def register_provider(self, provider: DynamicProvider) -> None:
        # [FUNCTION DOC]
        # Register or atomically replace this source's dynamic provider layer in the runtime.
        """Register or atomically replace this source's dynamic provider layer."""
        self._generation.assert_active()
        self._runtime.register_provider(self._source_id, provider)

    def update_provider(self, provider: DynamicProvider) -> bool:
        # [FUNCTION DOC]
        # Update this source's provider snapshot keeping its layer token; returns the runtime's success flag.
        """Update this source's provider snapshot while preserving its layer token."""
        self._generation.assert_active()
        return self._runtime.update_provider(self._source_id, provider)

    def register_local_backend(self, backend: LocalBackend) -> None:
        # [FUNCTION DOC]
        # Register a local backend paired with this source's provider layer.
        """Register a backend paired with this source's provider layer."""
        self._generation.assert_active()
        self._runtime.register_local_backend(self._source_id, backend)

    def register_command(
        self,
        name: str,
        handler: ExtensionCommandHandler,
        *,
        description: str = "",
        usage: str | None = None,
        aliases: tuple[str, ...] = (),
    ) -> None:
        # [FUNCTION DOC]
        # Register a sync slash command with name, handler, keyword description, optional usage and aliases.
        # Errors (e.g. duplicates) come from the runtime.
        """Register a slash command backed by this extension."""
        self._generation.assert_active()
        self._runtime.register_command(
            self._source_id,
            self._extension_name,
            name,
            handler,
            description=description,
            usage=usage,
            aliases=aliases,
        )

    def add_prompt_guideline(self, guideline: str) -> None:
        # [FUNCTION DOC]
        # Register a standalone system-prompt guideline line (deduplicated at prompt build time).
        """Add a standalone guideline line to the system prompt.

        Tool-attached guidance belongs on the tool (`prompt_snippet`,
        `prompt_guidelines`); this is for behavioral guidance not tied to
        any tool. Duplicate lines are de-duplicated at prompt build time.
        """
        self._generation.assert_active()
        self._runtime.register_prompt_guideline(self._source_id, self._extension_name, guideline)

    def add_prompt_section(self, title: str | None, body: str) -> None:
        # [FUNCTION DOC]
        # Register a free-form system-prompt section with optional title and body.
        """Append a free-form, optionally titled section to the system prompt.

        Use this for structured, always-on extension context such as procedures,
        paragraphs, and code blocks. Use :meth:`add_prompt_guideline` for one
        behavioral bullet instead.
        """
        self._generation.assert_active()
        self._runtime.register_prompt_section(
            self._source_id,
            self._extension_name,
            title,
            body,
        )

    def on(
        self,
        event: str,
        handler: ExtensionHandler | None = None,
    ) -> Callable[[ExtensionHandler], ExtensionHandler] | ExtensionHandler:
        # [FUNCTION DOC]
        # Subscribe a handler to an event name. With a handler: subscribe now and return it. Without:
        # return a decorator that asserts active, subscribes the decorated function and returns it.
        # Event name validation happens in the runtime.
        """Subscribe to an event, directly or as a decorator."""
        self._generation.assert_active()
        if handler is not None:
            self._runtime.subscribe(self._source_id, event, handler)
            return handler

        def decorator(decorated: ExtensionHandler) -> ExtensionHandler:
            # [FUNCTION DOC]
            # HELPER: Decorator form; assert active, subscribe the function for the captured event, return it.
            self._generation.assert_active()
            self._runtime.subscribe(self._source_id, event, decorated)
            return decorated

        return decorator

    def send_user_message(
        self,
        content: str,
        *,
        deliver_as: DeliverAs = "follow_up",
    ) -> None:
        # [FUNCTION DOC]
        # Queue a user message for the current or next run, delivered as "follow_up" (default) or "steer".
        """Queue a user message for the active or next agent run."""
        self._generation.assert_active()
        self._runtime.send_user_message(content, deliver_as=deliver_as)

    def register_message_renderer(
        self,
        custom_type: str,
        renderer: MessageRenderer,
    ) -> None:
        # [FUNCTION DOC]
        # Register a Rich-markup string renderer for a custom message type (first registration wins).
        """Register a renderer for custom messages with this ``custom_type``.

        Ports Pi's ``registerMessageRenderer``: the first registration per
        ``custom_type`` wins. The renderer receives a :class:`CustomMessageView`
        and :class:`MessageRenderOptions` and returns a Rich-markup string; it
        must not return a Textual widget (that keeps extensions TUI-free).
        """
        self._generation.assert_active()
        self._runtime.register_message_renderer(
            self._source_id, self._extension_name, custom_type, renderer
        )

    def send_custom_message(
        self,
        content: str,
        *,
        custom_type: str,
        details: dict[str, JSONValue] | None = None,
        deliver_as: DeliverAs = "follow_up",
        trigger_turn: bool = True,
    ) -> None:
        # [FUNCTION DOC]
        # Send a custom message (content enters LLM context; type/details drive rendering), delivered as
        # follow-up or steer; when trigger_turn (default) starts a turn if idle.
        """Send a custom message that renders via a registered renderer.

        Ports Pi's ``sendMessage``: ``content`` still enters LLM context, while
        ``custom_type``/``details`` let a registered renderer format the
        transcript block. With ``trigger_turn`` (the default) the message starts
        a turn when the session is idle, mirroring ``send_user_message``; set it
        to ``False`` to only queue for the next run.
        """
        self._generation.assert_active()
        self._runtime.send_custom_message(
            content,
            custom_type=custom_type,
            details=details,
            deliver_as=deliver_as,
            trigger_turn=trigger_turn,
        )

    async def append_entry(self, namespace: str, data: dict[str, JSONValue]) -> None:
        # [FUNCTION DOC]
        # Async: persist extension data as a custom session entry under the namespace. Side effect: session write.
        """Persist extension-owned data to the session as a custom entry."""
        self._generation.assert_active()
        await self._runtime.append_custom_entry(namespace, data)

    async def set_label(self, entry_id: str, label: str | None) -> None:
        # [FUNCTION DOC]
        # Async: set or clear (None) a bookmark label on an existing session entry. Side effect: session write.
        """Set or clear a bookmark on an existing session entry."""
        self._generation.assert_active()
        await self._runtime.set_label(entry_id, label)

    def notify(self, message: str, level: NotifyLevel = "info") -> None:
        # [FUNCTION DOC]
        # Assert active, then forward a notification to the runtime's UI bridge.
        """Show a notification in the UI, if one is attached."""
        self._generation.assert_active()
        self._runtime.ui.notify(message, level)


@dataclass(slots=True)
class RegisteredExtension:
    # [CLASS DOC]
    # Mutable bookkeeping for one loaded extension in the runtime: name, source id, file path (None for
    # built-ins), its API object, source kind (built-in/user/explicit/project; default explicit), a
    # hidden flag, and a mapping from event name to subscribed handlers.
    """Book-keeping for one loaded extension inside the runtime."""

    name: str
    source_id: str
    path: Path | None
    api: ExtensionAPI
    source: Literal["built-in", "user", "explicit", "project"] = "explicit"
    hidden: bool = False
    handlers: dict[str, list[ExtensionHandler]] = field(default_factory=dict)
