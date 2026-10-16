# [MODULE DOC]
# The interactive Textual TUI for Tau coding sessions (the tau_coding frontend layer; the
# reusable harness in tau_agent never imports this). It defines the main app class
# (TauTuiApp), the multiline prompt editor (PromptInput), many modal screens (pickers for
# sessions, models, providers, themes, prompt templates, tools reference, login/OAuth flows,
# extension dialogs), the bridge that lets extensions mount UI (dialogs, slot widgets,
# sidebar sections, main views, key interceptors), sidebar file editing with atomic writes,
# and many small formatting helpers. The app consumes CodingSession / agent events through
# TuiEventAdapter and TuiState and renders them into the transcript widgets.
# Inputs: a CodingSession (or the means to build one) plus TUI settings loaded from disk.
# Outputs: none directly; running the app drives terminal I/O.
# Side effects at import: none beyond module constants and type aliases; importing does not
# touch the environment (the Herdr mouse workaround is applied only when called).
"""Minimal Textual app for Tau coding sessions."""

from __future__ import annotations

import asyncio
import errno
import os
import stat
import tempfile
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum, auto
from inspect import isawaitable
from io import StringIO
from pathlib import Path
from typing import Any, BinaryIO, ClassVar, Literal, Protocol, TypeVar, cast

from rich.console import Console, Group
from rich.style import Style
from rich.text import Text
from textual import constants as textual_constants
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingsMap
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import Key, Resize
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import (
    Button,
    Input,
    Label,
    ListItem,
    ListView,
    OptionList,
    Static,
    TextArea,
)
from textual.worker import Worker

from tau_agent.events import (
    AgentEndEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    CustomMessage,
    TextContent,
    ThinkingContent,
    UserMessage,
)
from tau_agent.provider import CancellationToken
from tau_agent.provider_events import (
    AssistantErrorEvent,
    AssistantMessageEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
)
from tau_agent.tools import AgentTool
from tau_agent.types import JSONValue
from tau_coding.catalog_loader import save_user_catalog_entries
from tau_coding.commands import (
    LOGIN_PROVIDER_ALIASES,
    CommandRegistry,
    create_default_command_registry,
    format_reload_summary,
)
from tau_coding.credentials import FileCredentialStore, OAuthCredential
from tau_coding.events import (
    AgentSettledEvent,
    AutoRetryStartEvent,
    CodingSessionEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    QueueUpdateEvent,
)
from tau_coding.extensions.api import (
    KeyInterceptor,
    MainViewFactory,
    MainViewHandle,
    Placement,
    SidebarContent,
    SlotWidgetContent,
    SlotWidgetFactory,
)
from tau_coding.oauth import login_openai_codex
from tau_coding.oauth_registry import get_oauth_provider, oauth_provider_ids
from tau_coding.oauth_types import (
    OAuthAuthInfo,
    OAuthDeviceCodeInfo,
    OAuthLoginCallbacks,
    OAuthPrompt,
    OAuthSelectPrompt,
)
from tau_coding.project_trust import ProjectTrustRequest, TrustChoice, TrustOverride
from tau_coding.prompt_templates import PromptTemplate
from tau_coding.provider_catalog import (
    BUILTIN_PROVIDER_CATALOG,
    ProviderCatalogEntry,
    builtin_provider_entry,
)
from tau_coding.provider_config import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER_NAME,
    OpenAICompatibleProviderConfig,
    ProviderConfig,
    ProviderConfigError,
    ProviderSelection,
    load_provider_settings,
    provider_config_from_catalog_entry,
    provider_has_usable_credentials,
    resolve_provider_selection,
    resolve_startup_thinking_level,
    save_provider_settings,
    upsert_openai_compatible_provider,
    upsert_saved_provider,
)
from tau_coding.provider_runtime import ClosableModelProvider, create_model_provider
from tau_coding.resources import ResourceDiagnostic, TauResourcePaths
from tau_coding.session import (
    TREE_RUNNING_MESSAGE,
    CodingSession,
    CodingSessionConfig,
    ModelChoice,
    SessionTreeBranchResult,
    SessionTreeChoice,
    is_context_overflow_error,
    jsonl_session_storage,
    parse_terminal_command,
)
from tau_coding.session_manager import CodingSessionRecord, SessionManager
from tau_coding.session_preparation import prepare_coding_session
from tau_coding.shell_config import load_shell_settings
from tau_coding.skills import Skill
from tau_coding.system_prompt import SystemPromptInspection
from tau_coding.thinking import ThinkingLevel
from tau_coding.tui.adapter import TuiEventAdapter
from tau_coding.tui.autocomplete import (
    CompletionItem,
    CompletionKind,
    CompletionOption,
    CompletionState,
    build_completion_state,
)
from tau_coding.tui.config import (
    TAU_DARK_THEME,
    TuiKeybindings,
    TuiSettings,
    TuiTheme,
    TuiThemeName,
    load_tui_settings,
    save_tui_settings,
)
from tau_coding.tui.file_drop import normalize_dropped_paths
from tau_coding.tui.local_backends import (
    LocalBackendPickerScreen,
    LocalBackendScreen,
    LocalChoiceConfirmScreen,
    LocalConfirmScreen,
    LocalSearchResultsScreen,
)
from tau_coding.tui.project_trust import ProjectTrustScreen, prompt_project_trust
from tau_coding.tui.state import TuiState, format_terminal_command_result_block
from tau_coding.tui.terminal_notification import TerminalNotificationController
from tau_coding.tui.terminal_title import TerminalTitleController
from tau_coding.tui.themes import (
    available_tui_theme_names,
    load_custom_tui_themes,
    set_custom_tui_themes,
    textual_theme_for_tui_theme,
    theme_css_variables,
)
from tau_coding.tui.widgets import (
    CompactSessionInfo,
    SessionSidebar,
    SidebarFileItem,
    TranscriptView,
    _custom_markup_to_text,
    _sidebar_separator,
    render_completion_suggestions,
)

_textual_theme_for_tau_theme = textual_theme_for_tui_theme
_theme_css_variables = theme_css_variables

type BindingEntry = Binding | tuple[str, str] | tuple[str, str, str]
SIDEBAR_MIN_WIDTH = 96
SIDEBAR_MIN_HEIGHT = 38
ACTIVITY_TICK_SECONDS = 0.15
ACTIVITY_COLOR_FADE_STEPS = 24
ACTIVITY_INDICATOR_HEIGHT = 3
COMPLETION_MAX_VISIBLE_LINES = 16
COMPLETION_INITIAL_TERMINAL_FRACTION = 3
COMPLETION_MIN_TRANSCRIPT_LINES = 4
COMPLETION_WIDGET_CHROME_LINES = 3
NO_STORED_CREDENTIALS_MESSAGE = (
    "No stored credentials to remove. /logout only removes credentials saved by /login; "
    "environment variables and providers.json config are unchanged."
)


def _configure_herdr_textual_mouse() -> None:
    # [FUNCTION DOC]
    # HELPER: Works around mouse-coordinate problems in the Herdr terminal multiplexer.
    # If the environment variable HERDR_ENV equals "1" and TEXTUAL_SMOOTH_SCROLL is not already
    # set, sets TEXTUAL_SMOOTH_SCROLL to "0" in the process environment and also flips Textual's
    # already-imported SMOOTH_SCROLL constant to False so cell-based mouse coordinates are used.
    # Inputs: none (reads os.environ). Returns nothing. Raises nothing.
    # Side effects: mutates os.environ and a Textual module-level constant.
    """Keep Textual on cell mouse coordinates inside affected Herdr versions."""
    if os.environ.get("HERDR_ENV") != "1" or "TEXTUAL_SMOOTH_SCROLL" in os.environ:
        return
    os.environ["TEXTUAL_SMOOTH_SCROLL"] = "0"
    # Textual reads this environment variable while importing constants, before
    # Tau reaches the TUI runner. Update the loaded value for this process too.
    textual_constants.SMOOTH_SCROLL = False  # type: ignore[misc]


class LoginRequiredProvider:
    # [CLASS DOC]
    # Placeholder model provider used when no usable credentials exist, so the TUI can start
    # before the user logs in. Every streaming request immediately yields a single assistant
    # error event carrying the configured message (for example "please /login").
    # State: the message string. Implements the provider protocol (aclose, stream_response).
    """Placeholder provider used so the TUI can open before login."""

    def __init__(self, message: str) -> None:
        # [FUNCTION DOC]
        # Stores the login-needed message that every response will report.
        # Inputs: message text. Returns nothing. Side effects: sets self.message.
        self.message = message

    async def aclose(self) -> None:
        # [FUNCTION DOC]
        # No-op close; there are no resources. Returns nothing, raises nothing.
        """Close provider resources."""

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
        # Returns an async iterator that yields exactly one AssistantErrorEvent with reason "error"
        # whose error AssistantMessage has the requested model, stop_reason "error" and
        # error_message equal to self.message. All other inputs (system prompt, messages, tools,
        # cancellation signal, session id) are accepted and ignored.
        # Outputs: async iterator of assistant message events. Raises nothing. No side effects.
        """Surface a login-needed provider error."""
        del system, messages, tools, signal, session_id

        async def iterator() -> AsyncIterator[AssistantMessageEvent]:
            # [FUNCTION DOC]
            # HELPER: Inner async generator that builds the error AssistantMessage and yields one
            # AssistantErrorEvent, then finishes.
            error = AssistantMessage(
                model=model,
                stop_reason="error",
                error_message=self.message,
            )
            yield AssistantErrorEvent(reason="error", error=error)

        return iterator()


_DialogResult = TypeVar("_DialogResult")


@dataclass(frozen=True, slots=True)
class _SidebarContribution:
    # [CLASS DOC]
    # HELPER: Frozen record of one extension-owned sidebar section (title and content) kept so
    # the section can be rebuilt when the theme or layout is rebuilt.
    """One extension-owned sidebar section retained for theme rebuilds."""

    title: str
    content: SidebarContent


class _TuiExtensionUiBridge:
    # [CLASS DOC]
    # Adapter that implements the extension UI API (tau_coding.extensions.api) on top of the
    # running TauTuiApp. Extensions call notify/select/confirm/input for dialogs and the
    # component seam (slot widgets, main views, key interceptors, sidebar sections). Nearly all
    # methods delegate to private methods on the app. Maps extension severity names
    # ("info"/"warning"/"error") to Textual notification severities via _SEVERITIES.
    """Route extension UI requests to the running Textual app."""

    _SEVERITIES: ClassVar[dict[str, Literal["information", "warning", "error"]]] = {
        "info": "information",
        "warning": "warning",
        "error": "error",
    }

    def __init__(self, app: TauTuiApp) -> None:
        # [FUNCTION DOC]
        # Stores the owning app. Inputs: the TauTuiApp. Returns nothing.
        self._app = app

    @property
    def has_ui(self) -> bool:
        # [FUNCTION DOC]
        # Property; always True because an interactive UI is attached.
        """Return True: an interactive TUI is attached."""
        return True

    def notify(self, message: str, level: str = "info") -> None:
        # [FUNCTION DOC]
        # Shows a toast via the app's deduplicating _notify. Inputs: message text and level name;
        # unknown levels map to "information". Returns nothing. Side effect: UI notification.
        """Show an extension notification through the app's dedupe path."""
        self._app._notify(message, severity=self._SEVERITIES.get(level, "information"))

    async def select(
        self,
        title: str,
        options: Sequence[str],
        *,
        timeout: float | None = None,
    ) -> str | None:
        # [FUNCTION DOC]
        # Pushes an ExtensionSelectScreen listing the options (themed with the app's resolved theme)
        # and awaits the user's choice. Inputs: title, option strings, optional timeout in seconds.
        # Returns the chosen option string, or None on cancel or timeout.
        # Side effects: pushes a modal screen.
        """Show a modal picker; return the choice, or None on cancel/timeout."""
        theme = self._app.tui_settings.resolved_theme
        screen: ModalScreen[str | None] = ExtensionSelectScreen(title, options, theme=theme)
        return await self._run_dialog(screen, default=None, timeout=timeout)

    async def confirm(
        self,
        title: str,
        message: str,
        *,
        timeout: float | None = None,
    ) -> bool:
        # [FUNCTION DOC]
        # Pushes an ExtensionConfirmScreen and awaits the answer. Inputs: title, message, optional
        # timeout. Returns True only if the user confirmed; False on decline, cancel or timeout.
        """Show a modal confirmation; True only if confirmed."""
        theme = self._app.tui_settings.resolved_theme
        screen: ModalScreen[bool] = ExtensionConfirmScreen(title, message, theme=theme)
        return await self._run_dialog(screen, default=False, timeout=timeout)

    async def input(
        self,
        title: str,
        placeholder: str = "",
        *,
        timeout: float | None = None,
    ) -> str | None:
        # [FUNCTION DOC]
        # Pushes an ExtensionInputScreen and awaits submitted text. Inputs: title, placeholder,
        # optional timeout. Returns the entered text (possibly empty) or None on cancel/timeout.
        """Show a modal text prompt; return the text, or None on cancel/timeout."""
        theme = self._app.tui_settings.resolved_theme
        screen: ModalScreen[str | None] = ExtensionInputScreen(title, placeholder, theme=theme)
        return await self._run_dialog(screen, default=None, timeout=timeout)

    # -- component seam -- pass-through to the app ----------------------------

    @property
    def supports_components(self) -> bool:
        # [FUNCTION DOC]
        # Property; always True: the Textual TUI can host extension widgets.
        """Return True: a Textual TUI can host extension widgets."""
        return True

    @property
    def theme(self) -> TuiTheme:
        # [FUNCTION DOC]
        # Property returning the app's currently resolved TuiTheme (used by widget factories).
        """Return the live TUI theme handed to widget factories."""
        return self._app.tui_settings.resolved_theme

    def get_prompt_text(self) -> str:
        # [FUNCTION DOC]
        # Returns the current text in the prompt editor by delegating to the app. No side effects.
        """Return the current prompt-editor text (Pi's getEditorText).

        Interceptors do not need this — the host passes the prompt text as
        their second argument; it exists for reads outside the key path.
        """
        return self._app._current_prompt_text()

    def request_render(self) -> None:
        # [FUNCTION DOC]
        # Asks the app to re-render all mounted extension components. Returns nothing.
        """Re-render mounted extension widgets (analog of Pi's requestRender)."""
        self._app._refresh_extension_components()

    def set_slot_widget(
        self,
        key: str,
        content: SlotWidgetContent | None,
        *,
        placement: Placement = "above_prompt",
    ) -> None:
        # [FUNCTION DOC]
        # Mounts, replaces or (when content is None) removes an extension slot widget identified by
        # key at the given placement (default above the prompt). Content may be a widget factory or
        # lines of text. Delegates to the app; side effect: mounts/unmounts widgets.
        """Mount or remove an extension slot widget by key (factory or lines)."""
        self._app._set_extension_slot_widget(key, content, placement)

    def open_main_view(self, factory: MainViewFactory) -> MainViewHandle:
        # [FUNCTION DOC]
        # Opens a full main-area extension view built by the factory, hiding the transcript.
        # Returns a MainViewHandle (a dead no-op handle if opening failed). Delegates to the app.
        """Open a full main-area extension view (display-toggled, not modal)."""
        return self._app._open_extension_main_view(factory)

    def register_key_interceptor(self, handler: KeyInterceptor) -> Callable[[], None]:
        # [FUNCTION DOC]
        # Registers a key interceptor consulted before Textual bindings for every main-screen key
        # (not while a modal is on top). The handler must self-gate and return True only for keys
        # it consumes. Returns a callable that unregisters it. Side effect: mutates app state.
        """Register a pre-dispatch key hook; return an unsubscribe callable.

        Ports Pi's ``onTerminalInput``. The handler is consulted in
        ``TauTuiApp.on_event`` before Textual's app-level priority bindings and
        before the focused widget receives the key, so it can own navigation
        keys (``up``/``down``/``tab``/…) that tau otherwise binds with
        ``priority=True``. Because it fires for EVERY main-screen key regardless
        of which widget holds focus, the handler MUST self-gate (e.g. on the
        prompt text and its own state) and return ``True`` only for keys it
        actually consumes. It is never consulted while a modal screen (dialog,
        picker, command palette) is on top.
        """
        return self._app._register_extension_key_interceptor(handler)

    @property
    def supports_sidebar(self) -> bool:
        # [FUNCTION DOC]
        # Property; True unless the TUI settings put the sidebar position at "off".
        """Return whether the configured TUI sidebar can host sections."""
        return self._app.tui_settings.sidebar_position != "off"

    def set_sidebar_section(
        self,
        extension_name: str,
        key: str,
        *,
        title: str,
        content: SidebarContent,
    ) -> None:
        # [FUNCTION DOC]
        # Adds or replaces one sidebar section owned by the named extension under the given key,
        # with a title and content. Delegates to the app; side effect: sidebar rebuild.
        """Add or replace one host-framed extension sidebar section."""
        self._app._set_extension_sidebar_section(
            extension_name,
            key,
            title=title,
            content=content,
        )

    def remove_sidebar_section(self, extension_name: str, key: str) -> None:
        # [FUNCTION DOC]
        # Removes one sidebar section owned by an extension. Delegates to the app.
        """Remove one extension-owned sidebar section."""
        self._app._remove_extension_sidebar_section(extension_name, key)

    def clear_components(self) -> None:
        # [FUNCTION DOC]
        # Tears down all extension-owned UI (slot widgets, main view, sidebar sections, key
        # interceptors) via the app; used by /reload and session rebinds.
        """Tear down all extension-owned UI (runtime-driven: /reload, rebind)."""
        self._app._clear_extension_components()

    async def _run_dialog(
        self,
        screen: ModalScreen[_DialogResult],
        *,
        default: _DialogResult,
        timeout: float | None,
    ) -> _DialogResult:
        # [FUNCTION DOC]
        # HELPER: Pushes a modal screen and awaits its dismissal using a callback that resolves an
        # asyncio future (works outside Textual worker contexts).
        # Inputs: the modal screen, a default result, an optional timeout in seconds.
        # Returns the dismissed value, or the default when dismissed with None or on timeout.
        # On timeout it dismisses the screen with the default only if it is still the top screen,
        # suppressing any exception. Raises nothing itself (TimeoutError is caught).
        # Side effects: pushes (and possibly dismisses) a screen.
        # KNOWN ISSUE: if another screen covers the dialog before timeout, the stale dialog stays on
        # the screen stack until the user closes it manually (acknowledged in code comments).
        """Push a modal and await its dismissal via a callback-resolved future.

        Uses ``push_screen(screen, callback)`` + an ``asyncio.Future`` rather
        than ``push_screen_wait`` (which requires a Textual worker context);
        this pattern works from any coroutine on the app's event loop,
        including a task spawned by a sync ``/command`` handler. On ``timeout``
        (seconds) the dialog auto-dismisses and the no-op ``default`` returns.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[_DialogResult] = loop.create_future()

        def _resolve(result: _DialogResult | None) -> None:
            # [FUNCTION DOC]
            # HELPER: Dismissal callback; if the future is still pending, sets its result to the
            # default when result is None, otherwise to the result.
            # Textual passes None when a screen is dismissed with no value;
            # map that (and any explicit cancel) to the no-op default.
            if not future.done():
                future.set_result(default if result is None else result)

        self._app.push_screen(screen, _resolve)
        if timeout is None:
            return await future
        try:
            return await asyncio.wait_for(future, timeout)
        except TimeoutError:
            # `Screen.dismiss` only works while the dialog is the top screen.
            # Known limitation: if another screen was pushed on top before the
            # timeout fired, the stale dialog stays on the stack (its future
            # result is discarded by `_resolve` racing `future.done()`) until
            # the covering screen closes and the user dismisses it manually.
            if screen.is_current:
                with suppress(Exception):
                    screen.dismiss(default)
            return default


class _MainViewHandle:
    # [CLASS DOC]
    # Host-side handle for an open extension main view. Tracks open state, the mounted widget
    # (set later by the app) and a future resolved exactly once on teardown. close(result)
    # resolves wait() with result and tells the app to unmount; other host teardown paths
    # (rebind, quarantine, supersede) call _resolve(None) so awaiting tasks never hang.
    """Host-side handle to an open extension main view.

    ``close(result)`` is idempotent and routes back to the app, which unmounts
    the widget and restores the main transcript; it also resolves ``wait()``
    with ``result`` (Pi's ``done(result)``). Every other teardown path the host
    owns — session rebind, quarantine, being superseded by a later
    ``open_main_view`` — resolves ``wait()`` with ``None`` via
    :meth:`_resolve`, so an awaiting extension task never hangs.
    """

    def __init__(self, app: TauTuiApp, result: asyncio.Future[object | None]) -> None:
        # [FUNCTION DOC]
        # Stores the app, marks the view open, sets widget to None and keeps the result future.
        # Inputs: app, an asyncio future created on the app loop. Returns nothing.
        self._app = app
        self._open = True
        self.widget: Widget | None = None
        # Created on the app's event loop at open time; resolved exactly once by
        # the first teardown (close/clear/quarantine/supersede) to wake wait().
        self._result = result

    def close(self, result: object | None = None) -> None:
        # [FUNCTION DOC]
        # Idempotently closes the view: if already closed does nothing; otherwise marks closed,
        # resolves the future with result, and asks the app to close the main view (unmount the
        # widget, restore the transcript). Returns nothing.
        """Close the view, resolving ``wait()`` with ``result`` (safe to repeat)."""
        if not self._open:
            return
        self._open = False
        self._resolve(result)
        self._app._close_extension_main_view(self)

    def _resolve(self, result: object | None) -> None:
        # [FUNCTION DOC]
        # HELPER: Sets the future's result if not already done; later calls are no-ops.
        """Resolve the pending ``wait()`` future once; later calls are no-ops."""
        if not self._result.done():
            self._result.set_result(result)

    async def wait(self) -> object | None:
        # [FUNCTION DOC]
        # Awaits the result future and returns the value passed to close (None when torn down by
        # the host). Suspends until teardown.
        """Await teardown and return the ``close`` result (``None`` if cleared)."""
        return await self._result

    @property
    def is_open(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether the view is still open (False after close).
        """Return whether the view is still open."""
        return self._open


class _DeadMainViewHandle:
    # [CLASS DOC]
    # No-op MainViewHandle returned when a main view could not be opened: never open, close
    # does nothing, wait returns None immediately.
    """A no-op main-view handle returned when a view could not be opened."""

    def close(self, result: object | None = None) -> None:
        # [FUNCTION DOC]
        # Does nothing; result is ignored.
        """Do nothing: there is no view to close (``result`` is ignored)."""

    async def wait(self) -> object | None:
        # [FUNCTION DOC]
        # Returns None immediately.
        """Return None immediately: a dead handle never opens a view."""
        return None

    @property
    def is_open(self) -> bool:
        # [FUNCTION DOC]
        # Property; always False.
        """Return False: a dead handle is never open."""
        return False


class CompletionActionTarget(Protocol):
    # [CLASS DOC]
    # Typing protocol describing the app-level actions that PromptInput forwards to (completion
    # accept/next/previous, cancel, palettes, cycling thinking/model, toggles, queued-message
    # editing, prompt and follow-up submission). TauTuiApp satisfies it. No behavior.
    """App actions used by the prompt input completion bindings."""

    def action_accept_completion(self) -> None: ...
        # [FUNCTION DOC]
        # Protocol stub: accept the highlighted completion.

    def action_cancel(self) -> None: ...
        # [FUNCTION DOC]
        # Protocol stub: cancel the current operation or completion.

    def action_completion_next(self) -> None: ...
        # [FUNCTION DOC]
        # Protocol stub: select the next completion option.

    def action_completion_previous(self) -> None: ...
        # [FUNCTION DOC]
        # Protocol stub: select the previous completion option.

    def action_open_command_palette(self) -> None: ...
        # [FUNCTION DOC]
        # Protocol stub: open the command palette.

    def action_open_session_picker(self) -> None: ...
        # [FUNCTION DOC]
        # Protocol stub: open the session picker.

    def action_cycle_thinking(self) -> None: ...
        # [FUNCTION DOC]
        # Protocol stub: cycle the thinking level.

    def action_cycle_model(self) -> None: ...
        # [FUNCTION DOC]
        # Protocol stub: cycle to the next scoped model.

    def action_cycle_model_reverse(self) -> None: ...
        # [FUNCTION DOC]
        # Protocol stub: cycle to the previous scoped model.

    def action_toggle_tool_results(self) -> None: ...
        # [FUNCTION DOC]
        # Protocol stub: toggle tool result visibility.

    def action_toggle_thinking(self) -> None: ...
        # [FUNCTION DOC]
        # Protocol stub: toggle thinking-token visibility.

    def action_edit_queued_message(self) -> bool: ...
        # [FUNCTION DOC]
        # Protocol stub: pull a queued message back into the editor; returns True if one was
        # handled (so the caller should not move the cursor).

    async def action_submit_prompt(self) -> None: ...
        # [FUNCTION DOC]
        # Protocol stub (async): submit the prompt.

    async def action_submit_follow_up(self) -> None: ...
        # [FUNCTION DOC]
        # Protocol stub (async): submit the prompt as a follow-up message.


class SessionCompletionRecord(Protocol):
    # [CLASS DOC]
    # Typing protocol for session records shown in resume completions: id, optional title,
    # model name, working directory path, and updated_at timestamp (seconds). No behavior.
    """Session metadata needed to render resume picker completions."""

    id: str
    title: str | None
    model: str
    cwd: Path
    updated_at: float


PASTE_DISPLAY_THRESHOLD = 2_000


class PromptInput(TextArea):
    # [CLASS DOC]
    # Multiline prompt editor (a Textual TextArea) with Tau keybindings. Adds word-motion
    # bindings, footer-mode-dependent bindings (normal, completion, file_completion, running)
    # built by _prompt_bindings, forwarding of completion/palette/toggle actions to the app
    # (CompletionActionTarget), Input-like value/cursor_position compatibility properties,
    # file-drop path normalization, and collapsing of very large pastes (over
    # PASTE_DISPLAY_THRESHOLD characters) into placeholders that are expanded on submission.
    # State: keybindings, base bindings, footer mode, pending pastes (placeholder, content
    # pairs), paste counter, and shell_mode_style (style used to color a terminal-command
    # prefix).
    """Multiline prompt input with completion key bindings."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        # Herdr may re-encode terminal Esc+b/f word motion as Kitty Alt+b/f.
        Binding("alt+left,alt+b", "cursor_word_left", show=False),
        Binding("alt+right,alt+f", "cursor_word_right", show=False),
        Binding("alt+shift+left", "cursor_word_left(True)", show=False),
        Binding("alt+shift+right", "cursor_word_right(True)", show=False),
    ]
    shell_mode_style: str = ""

    def __init__(
        self,
        *,
        tui_keybindings: TuiKeybindings | None = None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC]
        # Initializes the TextArea (defaulting highlight_cursor_line to False), stores keybindings
        # (default TuiKeybindings), copies the base bindings, sets footer mode "normal", empty
        # pending pastes and counter 0, then applies the prompt bindings.
        # Inputs: optional keybindings plus TextArea keyword arguments. Returns nothing.
        kwargs.setdefault("highlight_cursor_line", False)
        super().__init__(**kwargs)
        self.tui_keybindings = tui_keybindings or TuiKeybindings()
        self._base_bindings = self._bindings.copy()
        self._footer_mode: Literal["normal", "completion", "file_completion", "running"] = "normal"
        self._pending_pastes: list[tuple[str, str]] = []
        self._paste_placeholder_counter = 0
        self._apply_prompt_bindings()

    def set_footer_mode(
        self,
        mode: Literal["normal", "completion", "file_completion", "running"],
    ) -> None:
        # [FUNCTION DOC]
        # Switches the footer mode; if unchanged does nothing, otherwise stores it, rebuilds the
        # bindings map and refreshes Textual's displayed bindings. Returns nothing.
        """Switch the prompt bindings shown by Textual's built-in footer."""
        if mode == self._footer_mode:
            return
        self._footer_mode = mode
        self._apply_prompt_bindings()
        self.refresh_bindings()

    def _apply_prompt_bindings(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Replaces the widget's bindings with the base bindings merged with the bindings
        # produced for the current keybindings and footer mode. Mutates self._bindings.
        self._bindings = BindingsMap.merge(
            [
                self._base_bindings,
                BindingsMap(_prompt_bindings(self.tui_keybindings, mode=self._footer_mode)),
            ]
        )

    @property
    def value(self) -> str:
        # [FUNCTION DOC]
        # Property alias for the editor text (compatibility with Input.value).
        """Compatibility alias for tests and code that previously used Input.value."""
        return self.text

    @value.setter
    def value(self, text: str) -> None:
        # [FUNCTION DOC]
        # Setter: replaces the editor text.
        self.text = text

    @property
    def cursor_position(self) -> int:
        # [FUNCTION DOC]
        # Property: the cursor as a flat character offset, computed from the (row, column)
        # location by summing the lengths of the previous lines plus one newline each.
        """Return a flat cursor offset for Input compatibility."""
        row, column = self.cursor_location
        lines = self.text.split("\n")
        return sum(len(line) + 1 for line in lines[:row]) + column

    @cursor_position.setter
    def cursor_position(self, offset: int) -> None:
        # [FUNCTION DOC]
        # Setter: clamps the offset to the text length and moves the cursor to the corresponding
        # (row, column) location. Side effect: cursor moves.
        text = self.text
        bounded = max(0, min(offset, len(text)))
        before = text[:bounded]
        self.move_cursor((before.count("\n"), len(before.rsplit("\n", 1)[-1])))

    def action_accept_completion(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the app's accept-completion action.
        """Accept the selected app-level completion."""
        self._completion_target().action_accept_completion()

    def action_completion_next(self) -> None:
        # [FUNCTION DOC]
        # If completion options are showing, selects the next one in the app; otherwise moves the
        # cursor down a line.
        """Select the next app-level completion or move down in the prompt."""
        if self._has_completion_options():
            self._completion_target().action_completion_next()
        else:
            self.action_cursor_down()

    def action_completion_previous(self) -> None:
        # [FUNCTION DOC]
        # If completion options are showing, selects the previous one; else if the app pulled a
        # queued message into the editor, stops; else moves the cursor up a line.
        """Select the previous app-level completion or move up in the prompt."""
        if self._has_completion_options():
            self._completion_target().action_completion_previous()
        elif self._completion_target().action_edit_queued_message():
            return
        else:
            self.action_cursor_up()

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the app's cancel action.
        """Run the app-level cancel action."""
        self._completion_target().action_cancel()

    def action_open_command_palette(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the app's command-palette action.
        """Open the app-level command palette."""
        self._completion_target().action_open_command_palette()

    def action_open_session_picker(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the app's session-picker action.
        """Open the app-level session picker."""
        self._completion_target().action_open_session_picker()

    def action_cycle_thinking(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the app's thinking-level cycle action.
        """Cycle the app-level thinking mode."""
        self._completion_target().action_cycle_thinking()

    def action_cycle_model(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the app's forward model cycle action.
        """Cycle the app-level scoped model forward."""
        self._completion_target().action_cycle_model()

    def action_cycle_model_reverse(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the app's reverse model cycle action.
        """Cycle the app-level scoped model backward."""
        self._completion_target().action_cycle_model_reverse()

    def action_toggle_tool_results(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the app's tool-results toggle.
        """Toggle app-level tool result display."""
        self._completion_target().action_toggle_tool_results()

    def action_toggle_thinking(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the app's thinking-visibility toggle.
        """Toggle app-level thinking-token display."""
        self._completion_target().action_toggle_thinking()

    def action_clear_prompt(self) -> None:
        # [FUNCTION DOC]
        # Clears the prompt: does nothing if text is selected (so the key can copy); otherwise, if
        # non-empty, empties the text, moves the cursor to the start and drops pending pastes.
        """Clear the current prompt."""
        if self.selected_text:
            return
        if self.text:
            self.text = ""
            self.move_cursor((0, 0))
            self._clear_pending_paste()

    def render_line(self, y: int) -> Strip:
        # [FUNCTION DOC]
        # Renders one line; returns a blank zero-width strip when the content width is zero or
        # negative (avoids a Textual crash during narrow layout switches), else defers to TextArea.
        """Render safely while a narrow terminal leaves no content width.

        Textual's placeholder wrapping currently raises when the content width
        is zero. This can happen briefly while a narrow terminal pane is
        switching from the sidebar layout to compact mode.
        """
        if self.content_size.width <= 0:
            return Strip.blank(0, self.visual_style.rich_style)
        return super().render_line(y)

    def get_line(self, line_index: int) -> Text:
        # [FUNCTION DOC]
        # Returns one line's rich Text; when shell_mode_style is set and the text starts with a
        # terminal-command prefix (per _terminal_command_prefix_span), styles the line with that
        # style from the prefix start on line 0 and from column 0 on later lines.
        """Retrieve one prompt line, coloring terminal commands like a running tool."""
        line = super().get_line(line_index)
        if not self.shell_mode_style:
            return line
        span = _terminal_command_prefix_span(self.text)
        if span is None:
            return line
        start, _ = span
        line.stylize(self.shell_mode_style, start if line_index == 0 else 0)
        return line

    async def action_submit_follow_up(self) -> None:
        # [FUNCTION DOC]
        # Awaits the app's follow-up submission action.
        """Submit the prompt as an app-level follow-up."""
        await self._completion_target().action_submit_follow_up()

    async def action_submit_prompt(self) -> None:
        # [FUNCTION DOC]
        # Awaits the app's prompt submission action.
        """Submit the prompt through the app-level action."""
        await self._completion_target().action_submit_prompt()

    def action_insert_newline(self) -> None:
        # [FUNCTION DOC]
        # Inserts a newline at the cursor.
        """Insert a newline in the prompt."""
        self.insert("\n")

    async def action_quit(self) -> None:
        # [FUNCTION DOC]
        # Awaits the app's quit action.
        """Quit the app through the app-level action."""
        await self.app.action_quit()

    def action_scroll_down(self) -> None:
        # [FUNCTION DOC]
        # Remaps scroll-down to completion-next (selection or cursor down).
        """Use down arrow for completion selection while focused."""
        self.action_completion_next()

    def action_scroll_up(self) -> None:
        # [FUNCTION DOC]
        # Remaps scroll-up to completion-previous.
        """Use up arrow for completion selection while focused."""
        self.action_completion_previous()

    def on_paste(self, event: events.Paste) -> None:
        # [FUNCTION DOC]
        # Paste event handler: if handle_pasted_text consumed the text (file drop or large paste),
        # stops the event and prevents Textual's default insertion.
        """Handle file drops and collapse very large pastes to a placeholder.

        Terminals deliver OS drag-and-drop as typed text, which Textual reports
        as a paste; when the pasted text is only existing file paths, insert the
        normalized paths instead of the raw (possibly escaped) drop text.
        """
        if self.handle_pasted_text(event.text):
            event.stop()
            event.prevent_default()

    def handle_pasted_text(self, text: str) -> bool:
        # [FUNCTION DOC]
        # Applies Tau paste rules. If the text normalizes to dropped file paths, inserts those
        # paths and returns True; if the text has at most PASTE_DISPLAY_THRESHOLD characters
        # returns False (caller inserts verbatim); otherwise inserts a large-paste placeholder and
        # returns True. Side effects: edits the text and pending-paste state.
        """Apply Tau's paste rules to *text*.

        Returns ``True`` when the text was inserted here (file drop or large
        paste placeholder) and ``False`` when it should be inserted verbatim by
        the caller (or by Textual's default paste handling).
        """
        dropped_paths = normalize_dropped_paths(text)
        if dropped_paths is not None:
            self._insert_dropped_paths(dropped_paths)
            return True
        if len(text) <= PASTE_DISPLAY_THRESHOLD:
            return False
        self._show_large_paste_placeholder(text)
        return True

    def insert_pasted_text(self, text: str) -> None:
        # [FUNCTION DOC]
        # Inserts pasted text that arrived outside Textual's normal paste path (e.g. a drop while
        # the terminal is unfocused): applies handle_pasted_text and, if not consumed, inserts the
        # text verbatim at the cursor. Returns nothing.
        """Insert pasted text that Textual could not deliver to this widget.

        Used for drops that arrive while the terminal is unfocused, where no
        default paste handler runs, so verbatim insertion is done here.
        """
        if not self.handle_pasted_text(text):
            self.insert(text)

    def _insert_dropped_paths(self, insertion: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Inserts the normalized dropped-path string at the cursor, prefixing a space if
        # the preceding character is not whitespace and suffixing a space if there is no following
        # text or the next character is not whitespace. Side effect: edits text.
        """Insert dropped paths at the cursor, separated from surrounding text."""
        position = self.cursor_position
        before = self.text[:position]
        after = self.text[position:]
        if before and not before[-1].isspace():
            insertion = f" {insertion}"
        if not after or not after[0].isspace():
            insertion = f"{insertion} "
        self.insert(insertion)

    def _show_large_paste_placeholder(self, content: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Increments the paste counter, builds a placeholder for the content, records the
        # (placeholder, content) pair in pending pastes, and inserts the placeholder at the cursor.
        """Store large pasted text and render a compact placeholder."""
        self._paste_placeholder_counter += 1
        placeholder = self._large_paste_placeholder(content, self._paste_placeholder_counter)
        self._pending_pastes.append((placeholder, content))
        self.insert(placeholder)

    def _large_paste_placeholder(self, content: str, paste_number: int) -> str:
        # [FUNCTION DOC]
        # HELPER: Builds the placeholder label "[Pasted content #N: ...]" listing the character
        # count (thousands-separated), the line count if more than one line, and the size in KB with
        # one decimal if at least 1 KB. Pure; returns the string.
        """Build the display text for a large paste."""
        char_count = len(content)
        line_count = content.count("\n") + 1
        kb = char_count / 1024
        parts: list[str] = [f"{char_count:,} characters"]
        if line_count > 1:
            parts.append(f"{line_count} lines")
        if kb >= 1:
            parts.append(f"{kb:.1f} KB")
        return f"[Pasted content #{paste_number}: {', '.join(parts)}]"

    def _clear_pending_paste(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Drops all stored large-paste contents.
        """Forget any stored large paste content."""
        self._pending_pastes.clear()

    def sync_pending_paste(self) -> None:
        # [FUNCTION DOC]
        # Keeps only pending pastes whose placeholder still appears in the current text, so edited
        # or deleted placeholders are no longer expanded. Mutates pending pastes.
        """Invalidate stored paste content when its placeholder is edited away."""
        self._pending_pastes = [
            (placeholder, content)
            for placeholder, content in self._pending_pastes
            if placeholder in self.text
        ]

    def text_for_submission(self) -> str:
        # [FUNCTION DOC]
        # Returns the text to submit: syncs pending pastes, then replaces the first occurrence of
        # each intact placeholder with its stored content. Does not modify the editor.
        """Return the prompt text, expanding intact large-paste placeholders."""
        self.sync_pending_paste()
        text = self.text
        for placeholder, content in self._pending_pastes:
            text = text.replace(placeholder, content, 1)
        return text

    async def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Key handler that routes Tau keys before default TextArea handling, according to the
        # configured keybindings: queue-follow-up key submits a follow-up; plain enter submits the
        # prompt; the insert-newline key inserts a newline; accept-completion, cancel, command
        # palette, session picker, thinking cycle (via _is_thinking_cycle_key), model cycle
        # forward/reverse, tool-results toggle and thinking toggle forward to the app; the
        # copy-message key does nothing if text is selected (allowing copy) and otherwise clears the
        # prompt (without clearing pending pastes); completion-next selects the next completion or
        # moves the cursor down; completion-previous behaves like action_completion_previous; quit
        # calls the quit action. Matched keys are stopped (and default-prevented where text would
        # change). Unmatched keys fall through. Returns nothing.
        # KNOWN ISSUE: the copy-message branch clears the prompt but, unlike action_clear_prompt,
        # does not clear stored large-paste contents (they are only dropped later by syncing).
        """Route completion and submission keys before default input handling.

        Extension key interceptors are consulted upstream in
        :meth:`TauTuiApp.on_event` (pre-dispatch, before app-level priority
        bindings), so there is no interceptor splice here.
        """
        keybindings = self.tui_keybindings
        if event.key == keybindings.queue_follow_up:
            event.stop()
            event.prevent_default()
            await self._completion_target().action_submit_follow_up()
        elif event.key == "enter":
            event.stop()
            event.prevent_default()
            await self._completion_target().action_submit_prompt()
        elif event.key == keybindings.insert_newline:
            event.stop()
            event.prevent_default()
            self.insert("\n")
        elif event.key == keybindings.accept_completion:
            event.stop()
            self._completion_target().action_accept_completion()
        elif event.key == keybindings.cancel:
            event.stop()
            self._completion_target().action_cancel()
        elif event.key == keybindings.command_palette:
            event.stop()
            self._completion_target().action_open_command_palette()
        elif event.key == keybindings.session_picker:
            event.stop()
            self._completion_target().action_open_session_picker()
        elif _is_thinking_cycle_key(event.key, keybindings.thinking_cycle):
            event.stop()
            self._completion_target().action_cycle_thinking()
        elif event.key == keybindings.model_cycle:
            event.stop()
            self._completion_target().action_cycle_model()
        elif event.key == keybindings.model_cycle_reverse:
            event.stop()
            self._completion_target().action_cycle_model_reverse()
        elif event.key == keybindings.toggle_tool_results:
            event.stop()
            self._completion_target().action_toggle_tool_results()
        elif event.key == keybindings.toggle_thinking:
            event.stop()
            self._completion_target().action_toggle_thinking()
        elif event.key == keybindings.copy_message:
            if self.selected_text:
                return
            event.stop()
            event.prevent_default()
            if self.text:
                self.text = ""
                self.move_cursor((0, 0))
        elif event.key == keybindings.completion_next:
            event.stop()
            if self._has_completion_options():
                self._completion_target().action_completion_next()
            else:
                self.action_cursor_down()
        elif event.key == keybindings.completion_previous:
            event.stop()
            self.action_completion_previous()
        elif event.key == keybindings.quit:
            event.stop()
            await self.action_quit()

    def _has_completion_options(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: True if the app has a completion state with at least one item (looked up
        # defensively with getattr).
        completion_state = getattr(self.app, "_completion_state", None)
        return bool(getattr(completion_state, "items", ()))

    def _completion_target(self) -> CompletionActionTarget:
        # [FUNCTION DOC]
        # HELPER: Returns the app cast to the CompletionActionTarget protocol.
        return cast(CompletionActionTarget, self.app)


class ExtensionSelectScreen(ModalScreen[str | None]):
    # [CLASS DOC]
    # Modal picker for extension select dialogs. Shows a title, a ListView of option labels and
    # help text. Escape cancels (dismiss None); up/down/enter move and select; selecting
    # dismisses with the chosen option string. Must be allow-listed in the app's completion
    # actions so global priority arrow bindings reach it.
    """Modal option picker backing `context.ui.select`.

    Binding/key wiring mirrors `SessionPickerScreen`. Note: the app binds
    Up/Down globally with priority (completion navigation), so this screen
    must also be listed in the `action_completion_next/previous` and
    `action_accept_completion` screen allowlists for arrow keys to reach
    the option list.
    """

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "select_cursor", "Select", show=False),
    ]

    def __init__(
        self,
        title: str,
        options: Sequence[str],
        *,
        theme: TuiTheme,
    ) -> None:
        # [FUNCTION DOC]
        # Stores the title, options (as a tuple) and theme. Returns nothing.
        super().__init__()
        self.title_text = title
        self.options = tuple(options)
        self.theme = theme

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields a Vertical containing the title Static, the option ListView (markup disabled) and
        # a help line.
        """Compose the option picker."""
        with Vertical(id="extension-select"):
            yield Static(self.title_text, id="extension-select-title", markup=False)
            yield ListView(
                *[ListItem(Label(option, markup=False)) for option in self.options],
                id="extension-select-list",
            )
            yield Static("Enter selects - Escape cancels", id="extension-select-help")

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Sets the list index to the first option and focuses the list.
        """Focus the option list for keyboard navigation."""
        option_list = self.query_one("#extension-select-list", ListView)
        option_list.index = 0
        option_list.focus()

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Routes up/down/enter keys to the cursor and select actions, stopping the events.
        """Route arrow and enter keys to the option list."""
        if event.key == "up":
            event.stop()
            self.action_cursor_up()
        elif event.key == "down":
            event.stop()
            self.action_cursor_down()
        elif event.key == "enter":
            event.stop()
            self.action_select_cursor()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # [FUNCTION DOC]
        # Dismisses the screen with the option at the selected index.
        """Dismiss with the chosen option."""
        self.dismiss(self.options[event.index])

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Moves the list cursor up.
        """Move to the previous option."""
        self.query_one("#extension-select-list", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Moves the list cursor down.
        """Move to the next option."""
        self.query_one("#extension-select-list", ListView).action_cursor_down()

    def action_select_cursor(self) -> None:
        # [FUNCTION DOC]
        # Selects the highlighted option (triggers the ListView selected handler).
        """Select the highlighted option."""
        self.query_one("#extension-select-list", ListView).action_select_cursor()

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with None.
        """Close without choosing an option."""
        self.dismiss(None)


class ExtensionConfirmScreen(ModalScreen[bool]):
    # [CLASS DOC]
    # Modal yes/no confirmation for extension confirm dialogs: title, message, and a two-item
    # list (Yes, No). Selecting dismisses with True for Yes (index 0), False for No; escape
    # dismisses with False.
    """Modal yes/no confirmation backing `context.ui.confirm`.

    Binding/key wiring mirrors `SessionPickerScreen`; see
    `ExtensionSelectScreen` for the app-level Up/Down allowlist requirement.
    """

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "select_cursor", "Select", show=False),
    ]

    def __init__(self, title: str, message: str, *, theme: TuiTheme) -> None:
        # [FUNCTION DOC]
        # Stores title, message and theme. Returns nothing.
        super().__init__()
        self.title_text = title
        self.message = message
        self.theme = theme

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields a Vertical with title, message, a Yes/No ListView and help text.
        """Compose the confirmation dialog."""
        with Vertical(id="extension-confirm"):
            yield Static(self.title_text, id="extension-confirm-title", markup=False)
            yield Static(self.message, id="extension-confirm-message", markup=False)
            yield ListView(
                ListItem(Label("Yes", markup=False)),
                ListItem(Label("No", markup=False)),
                id="extension-confirm-list",
            )
            yield Static("Enter selects - Escape cancels", id="extension-confirm-help")

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Selects Yes (index 0) and focuses the choice list.
        """Focus the choice list."""
        choice_list = self.query_one("#extension-confirm-list", ListView)
        choice_list.index = 0
        choice_list.focus()

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Routes up/down/enter to cursor and select actions, stopping the events.
        """Route arrow and enter keys to the choice list."""
        if event.key == "up":
            event.stop()
            self.action_cursor_up()
        elif event.key == "down":
            event.stop()
            self.action_cursor_down()
        elif event.key == "enter":
            event.stop()
            self.action_select_cursor()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # [FUNCTION DOC]
        # Dismisses with True if index 0 (Yes) was selected, else False.
        """Dismiss with the confirmation result (Yes is index 0)."""
        self.dismiss(event.index == 0)

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Moves the choice cursor up.
        """Move to the previous choice."""
        self.query_one("#extension-confirm-list", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Moves the choice cursor down.
        """Move to the next choice."""
        self.query_one("#extension-confirm-list", ListView).action_cursor_down()

    def action_select_cursor(self) -> None:
        # [FUNCTION DOC]
        # Selects the highlighted choice.
        """Select the highlighted choice."""
        self.query_one("#extension-confirm-list", ListView).action_select_cursor()

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with False.
        """Close, declining the confirmation."""
        self.dismiss(False)


class ExtensionInputScreen(ModalScreen[str | None]):
    # [CLASS DOC]
    # Modal single-line text prompt for extension input dialogs (and reused elsewhere). Enter
    # dismisses with the entered text; escape dismisses with None.
    """Modal single-line text prompt backing `context.ui.input`."""

    BINDINGS: ClassVar[list[BindingEntry]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(
        self,
        title: str,
        placeholder: str = "",
        *,
        theme: TuiTheme,
        value: str = "",
    ) -> None:
        # [FUNCTION DOC]
        # Stores title, placeholder, theme and an initial value. Returns nothing.
        super().__init__()
        self.title_text = title
        self.placeholder = placeholder
        self.theme = theme
        self.value = value

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields a Vertical with title, an Input prefilled with the value and placeholder, and help.
        """Compose the text prompt."""
        with Vertical(id="extension-input"):
            yield Static(self.title_text, id="extension-input-title", markup=False)
            yield Input(
                value=self.value,
                placeholder=self.placeholder,
                id="extension-input-field",
            )
            yield Static("Enter submits - Escape cancels", id="extension-input-help")

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Focuses the text field.
        """Focus the text field."""
        self.query_one("#extension-input-field", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # [FUNCTION DOC]
        # For submissions from this screen's field only: stops the event and dismisses with the
        # submitted value.
        """Dismiss with the entered text."""
        if event.input.id != "extension-input-field":
            return
        event.stop()
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with None.
        """Close without submitting text."""
        self.dismiss(None)


class ToolsReferenceSearchInput(Input):
    # [CLASS DOC]
    # Search Input for the tools reference screen whose priority bindings keep escape, up,
    # down and enter local, routing them to the owning ToolsReferenceScreen.
    """Search input that keeps tool-reference navigation local."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        Binding("up", "cursor_up", "Up", show=False, priority=True),
        Binding("down", "cursor_down", "Down", show=False, priority=True),
        Binding("enter", "open_selected", "Open", show=False, priority=True),
    ]

    def _reference(self) -> ToolsReferenceScreen:
        # [FUNCTION DOC]
        # HELPER: Returns the containing screen cast to ToolsReferenceScreen.
        return cast(ToolsReferenceScreen, self.screen)

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Routes up/down/escape/enter to the reference screen's cursor, cancel and open actions,
        # stopping and default-preventing the events so the search text is unchanged.
        """Route navigation without changing the search text."""
        if event.key == "up":
            event.stop()
            event.prevent_default()
            self._reference().action_cursor_up()
        elif event.key == "down":
            event.stop()
            event.prevent_default()
            self._reference().action_cursor_down()
        elif event.key == "escape":
            event.stop()
            event.prevent_default()
            self._reference().action_cancel()
        elif event.key == "enter":
            event.stop()
            event.prevent_default()
            self._reference().action_open_selected()

    def action_open_selected(self) -> None:
        # [FUNCTION DOC]
        # Opens the selected tool's description via the reference screen.
        self._reference().action_open_selected()


class ToolsReferenceScreen(ModalScreen[None]):
    # [CLASS DOC]
    # Modal searchable table of available agent tools (name, origin, description length).
    # Built-in tools are listed first (sorted case-insensitively), then extension tools grouped
    # by extension in registration order. Enter opens a CommandOutputScreen with the full
    # description; escape closes.
    """Searchable tool table with navigable description details."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "cancel", "Close"),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "open_selected", "Open", show=False),
    ]

    def __init__(
        self,
        tools: Sequence[AgentTool],
        *,
        extension_sources: Mapping[str, str],
        theme: TuiTheme,
    ) -> None:
        # [FUNCTION DOC]
        # Stores extension sources (tool name to extension name), the ordered tools, the visible
        # tools (initially all) and theme.
        super().__init__()
        self.extension_sources = dict(extension_sources)
        self.tools = self._order_tools(tools)
        self.visible_tools = self.tools
        self.theme = theme

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields the title, a ToolsReferenceSearchInput, a header row, the tool ListView and help.
        """Compose the tool reference."""
        with Vertical(id="tools-reference"):
            yield Static("Available tools", id="tools-reference-title")
            yield ToolsReferenceSearchInput(placeholder="Search tools", id="tools-reference-search")
            yield Static(
                self._table_row("Tool", "Origin", "Description"),
                id="tools-reference-header",
            )
            yield ListView(id="tools-reference-list")
            yield Static("Enter opens description - Escape closes", id="tools-reference-help")

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Populates the list with all tools and focuses the search input.
        """Populate the list and focus search on open."""
        self._refresh_tools("")
        self.query_one("#tools-reference-search", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        # [FUNCTION DOC]
        # On changes to the search input, stops the event and refilters the list.
        if event.input.id == "tools-reference-search":
            event.stop()
            self._refresh_tools(event.value)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # [FUNCTION DOC]
        # Stops the event and opens the description of the selected tool.
        """Open the selected tool's full description."""
        event.stop()
        self._open_tool(event.index)

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Moves the list cursor up.
        self.query_one("#tools-reference-list", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Moves the list cursor down.
        self.query_one("#tools-reference-list", ListView).action_cursor_down()

    def action_open_selected(self) -> None:
        # [FUNCTION DOC]
        # Opens the tool at the list's current index, if any.
        tool_list = self.query_one("#tools-reference-list", ListView)
        if tool_list.index is not None:
            self._open_tool(tool_list.index)

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismisses the screen with None.
        self.dismiss(None)

    def _open_tool(self, index: int) -> None:
        # [FUNCTION DOC]
        # HELPER: If the index is within the visible tools, pushes a CommandOutputScreen titled
        # "<name> — <origin>" showing the tool description (or "No description").
        if index >= len(self.visible_tools):
            return
        tool = self.visible_tools[index]
        self.app.push_screen(
            CommandOutputScreen(
                f"{tool.name} — {self._source_label(tool)}",
                tool.description or "No description",
                theme=self.theme,
            )
        )

    def _refresh_tools(self, query: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Filters tools by a case-insensitive, stripped query matching name, label,
        # description or origin; rebuilds the list with one row per tool (showing description length
        # in characters) and selects the first, or a single disabled message row ("No tools
        # available." / "No tools match your search.") when empty. Mutates visible_tools and list.
        needle = query.casefold().strip()
        self.visible_tools = tuple(
            tool
            for tool in self.tools
            if not needle
            or needle in tool.name.casefold()
            or needle in tool.label.casefold()
            or needle in tool.description.casefold()
            or needle in self._source_label(tool).casefold()
        )
        tool_list = self.query_one("#tools-reference-list", ListView)
        tool_list.clear()
        if not self.visible_tools:
            message = "No tools available." if not self.tools else "No tools match your search."
            tool_list.append(ListItem(Label(message, markup=False), disabled=True))
            return
        tool_list.extend(
            [
                ListItem(
                    Label(
                        self._table_row(
                            tool.name,
                            self._source_label(tool),
                            f"{len(tool.description)} chars",
                        ),
                        markup=False,
                    )
                )
                for tool in self.visible_tools
            ]
        )
        tool_list.index = 0

    def _order_tools(self, tools: Sequence[AgentTool]) -> tuple[AgentTool, ...]:
        # [FUNCTION DOC]
        # HELPER: Orders tools: built-ins (names not in extension sources) sorted case-insensitively,
        # then for each distinct extension (in mapping order) its tools that exist. Returns a tuple.
        # Note tools listed in extension sources but absent from the tool list are skipped.
        tools_by_name = {tool.name: tool for tool in tools}
        builtins = sorted(
            (tool for tool in tools if tool.name not in self.extension_sources),
            key=lambda tool: tool.name.casefold(),
        )
        extension_tools: list[AgentTool] = []
        seen_extensions: set[str] = set()
        for extension in self.extension_sources.values():
            if extension in seen_extensions:
                continue
            seen_extensions.add(extension)
            extension_tools.extend(
                tools_by_name[tool_name]
                for tool_name, source in self.extension_sources.items()
                if source == extension and tool_name in tools_by_name
            )
        return tuple([*builtins, *extension_tools])

    def _table_row(self, name: str, source: str, description: str) -> str:
        # [FUNCTION DOC]
        # HELPER: Formats a fixed-width row: name padded to the longest tool name (at least "Tool"),
        # origin padded to the longest origin label (at least "Origin"), then the description, with
        # two-space gaps.
        name_width = max((len(tool.name) for tool in self.tools), default=len("Tool"))
        source_width = max(
            (len(self._source_label(tool)) for tool in self.tools),
            default=len("Origin"),
        )
        return (
            f"{name:<{max(name_width, len('Tool'))}}  "
            f"{source:<{max(source_width, len('Origin'))}}  {description}"
        )

    def _source_label(self, tool: AgentTool) -> str:
        # [FUNCTION DOC]
        # HELPER: Returns the extension name owning the tool, or "Built in".
        extension = self.extension_sources.get(tool.name)
        return extension if extension is not None else "Built in"


class SessionPickerSearchInput(Input):
    # [CLASS DOC]
    # Search Input used by the session picker (and reused by the prompt-template picker) that
    # keeps navigation keys local: up/down move the picker cursor, left/right switch focus
    # between projects and sessions columns, ctrl+enter archives, f2 shows archived, escape
    # cancels.
    # KNOWN ISSUE: the f2 priority binding names action "show_archived", which this Input does
    # not define; f2 actually works only through the _on_key override.
    """Search input that keeps session-picker navigation local to the picker."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        Binding("up", "cursor_up", "Up", show=False, priority=True),
        Binding("down", "cursor_down", "Down", show=False, priority=True),
        Binding("f2", "show_archived", "Archived", show=False, priority=True),
    ]

    def _picker(self) -> SessionPickerScreen:
        # [FUNCTION DOC]
        # HELPER: Returns the containing screen cast to SessionPickerScreen (duck-typed; the
        # prompt-template picker also provides the cursor/cancel actions).
        return cast(SessionPickerScreen, self.screen)

    async def _on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Low-level key hook: when inside a SessionPickerScreen and the key is ctrl+enter or f2,
        # stops the event and archives the highlighted entry (ctrl+enter) or switches to the
        # archived view (f2); otherwise defers to Input's default key processing.
        """Route the archive shortcut before Textual's default input handling."""
        if event.key in {"ctrl+enter", "f2"} and isinstance(self.screen, SessionPickerScreen):
            event.stop()
            event.prevent_default()
            if event.key == "f2":
                self.screen.action_show_archived()
            else:
                self.screen.action_archive_cursor()
            return
        await super()._on_key(event)

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Routes up/down to picker cursor moves, left/right to project/session focus (session
        # picker only), ctrl+enter to archive (session picker only) and escape to cancel; matched
        # keys are stopped and default-prevented.
        """Route picker control keys before the input edits its text."""
        if event.key == "up":
            event.stop()
            event.prevent_default()
            self.action_cursor_up()
        elif event.key == "down":
            event.stop()
            event.prevent_default()
            self.action_cursor_down()
        elif event.key in {"left", "right"} and isinstance(self.screen, SessionPickerScreen):
            event.stop()
            event.prevent_default()
            if event.key == "left":
                self.screen.action_focus_projects()
            else:
                self.screen.action_focus_sessions()
        elif event.key == "ctrl+enter" and isinstance(self.screen, SessionPickerScreen):
            event.stop()
            event.prevent_default()
            self.screen.action_archive_cursor()
        elif event.key == "escape":
            event.stop()
            event.prevent_default()
            self.action_cancel()

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Moves the owning picker's cursor up.
        """Move the session picker selection up."""
        self._picker().action_cursor_up()

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Moves the owning picker's cursor down.
        """Move the session picker selection down."""
        self._picker().action_cursor_down()

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Cancels (closes) the owning picker.
        """Close the session picker."""
        self._picker().action_cancel()


@dataclass(frozen=True, slots=True)
class PromptTemplatePickerResult:
    # [CLASS DOC]
    # Frozen result of the prompt-template picker: the chosen action ("insert" or "edit") and
    # the template.
    """Action selected from the prompt-template picker."""

    action: Literal["insert", "edit"]
    template: PromptTemplate


class PromptTemplatePickerScreen(ModalScreen[PromptTemplatePickerResult | None]):
    # [CLASS DOC]
    # Modal searchable picker over loaded prompt templates, sorted by lowercase name. Enter
    # dismisses with an insert result, ctrl+e with an edit result, escape with None. Search
    # matches name or description case-insensitively.
    """Searchable picker for loaded prompt templates."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "select_cursor", "Select", show=False),
        Binding("ctrl+e", "edit_cursor", "Edit", show=False, priority=True),
    ]

    def __init__(self, templates: Sequence[PromptTemplate]) -> None:
        # [FUNCTION DOC]
        # Stores templates sorted by lowercase name; all are initially visible.
        super().__init__()
        self.templates = tuple(sorted(templates, key=lambda item: item.name.lower()))
        self.visible_templates = self.templates

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields the title, a SessionPickerSearchInput, the template ListView and a help Static.
        with Vertical(id="prompt-template-picker"):
            yield Static("Prompt templates", id="prompt-template-picker-title")
            yield SessionPickerSearchInput(
                placeholder="Search prompt templates", id="prompt-template-picker-search"
            )
            yield ListView(id="prompt-template-picker-list")
            yield Static("", id="prompt-template-picker-help")

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Focuses the search field and fills the list.
        self.query_one("#prompt-template-picker-search", Input).focus()
        self._refresh_list()

    def on_input_changed(self, event: Input.Changed) -> None:
        # [FUNCTION DOC]
        # For this picker's search input: stops the event, filters templates whose name or
        # description contains the casefolded query, and refreshes the list.
        if event.input.id != "prompt-template-picker-search":
            return
        event.stop()
        query = event.value.casefold()
        self.visible_templates = tuple(
            template
            for template in self.templates
            if query in template.name.casefold() or query in (template.description or "").casefold()
        )
        self._refresh_list()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # [FUNCTION DOC]
        # On submit from the search field, stops the event and selects the highlighted template.
        if event.input.id == "prompt-template-picker-search":
            event.stop()
            self.action_select_cursor()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # [FUNCTION DOC]
        # Stops the event and selects the highlighted template.
        event.stop()
        self.action_select_cursor()

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Moves the list cursor up.
        self.query_one("#prompt-template-picker-list", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Moves the list cursor down.
        self.query_one("#prompt-template-picker-list", ListView).action_cursor_down()

    def action_select_cursor(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with an "insert" result for the highlighted template, if any.
        template = self._selected_template()
        if template is not None:
            self.dismiss(PromptTemplatePickerResult(action="insert", template=template))

    def action_edit_cursor(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with an "edit" result for the highlighted template, if any.
        """Open the selected template in Tau's prompt editor."""
        template = self._selected_template()
        if template is not None:
            self.dismiss(PromptTemplatePickerResult(action="edit", template=template))

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with None.
        self.dismiss(None)

    def _selected_template(self) -> PromptTemplate | None:
        # [FUNCTION DOC]
        # HELPER: Returns the visible template at the list index, or None if no valid index.
        picker_list = self.query_one("#prompt-template-picker-list", ListView)
        index = picker_list.index
        if index is None or index >= len(self.visible_templates):
            return None
        return self.visible_templates[index]

    def _refresh_list(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Rebuilds the list with "/name — description" rows, sets the index to 0 (or None
        # when empty), and updates the help text depending on whether there are matches, templates
        # but no matches, or no templates at all.
        picker_list = self.query_one("#prompt-template-picker-list", ListView)
        picker_list.clear()
        picker_list.extend(
            ListItem(
                Label(
                    f"/{template.name} — {template.description or 'No description'}",
                    markup=False,
                )
            )
            for template in self.visible_templates
        )
        picker_list.index = 0 if self.visible_templates else None
        if self.visible_templates:
            help_text = "Enter inserts - Ctrl+E edits - Escape closes"
        elif self.templates:
            help_text = "No matching prompt templates - Escape closes"
        else:
            help_text = "No prompt templates loaded - Escape closes"
        self.query_one("#prompt-template-picker-help", Static).update(help_text)


class PromptTemplateEditorScreen(ModalScreen[str | None]):
    # [CLASS DOC]
    # Modal editor for a prompt template's Markdown source. Ctrl+S dismisses with the edited
    # text (the caller saves it); escape dismisses with None.
    """Edit one prompt-template Markdown file inside the TUI."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save", show=False, priority=True),
    ]

    def __init__(self, template: PromptTemplate, source: str) -> None:
        # [FUNCTION DOC]
        # Stores the template and its current source text.
        super().__init__()
        self.template = template
        self.source = source

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields title "Edit /name", the template path, a TextArea with the source and help text.
        with Vertical(id="prompt-template-editor"):
            yield Static(f"Edit /{self.template.name}", id="prompt-template-editor-title")
            yield Static(str(self.template.path), id="prompt-template-editor-path")
            yield TextArea(self.source, id="prompt-template-editor-input")
            yield Static(
                "Ctrl+S saves - Escape returns without saving",
                id="prompt-template-editor-help",
            )

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Focuses the TextArea.
        self.query_one("#prompt-template-editor-input", TextArea).focus()

    def action_save(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with the current TextArea text.
        source = self.query_one("#prompt-template-editor-input", TextArea).text
        self.dismiss(source)

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with None (discarding edits).
        self.dismiss(None)


def _write_staged_utf8(handle: BinaryIO, source: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Encodes the text as UTF-8 and writes it fully to an open binary file handle,
    # looping over partial writes.
    # Raises OSError("staged write did not make progress") if a write returns None or zero.
    # Side effect: writes to the handle.
    """Write complete UTF-8 editor contents to an open staging file."""
    remaining = memoryview(source.encode("utf-8"))
    while remaining:
        written = handle.write(remaining)
        if written is None or written <= 0:
            raise OSError("staged write did not make progress")
        remaining = remaining[written:]


@dataclass(frozen=True, slots=True)
class _SidebarFileSnapshot:
    # [CLASS DOC]
    # HELPER: Frozen snapshot of a sidebar file at load time: the resolved target path (symlinks
    # followed) and the exact bytes read. Used to detect concurrent changes before saving.
    """Resolved target and exact bytes observed when a sidebar file was loaded."""

    target: Path
    content: bytes


def _read_sidebar_file(path: Path) -> tuple[str, _SidebarFileSnapshot]:
    # [FUNCTION DOC]
    # HELPER: Resolves the path strictly (must exist), reads its bytes, and returns the UTF-8
    # decoded text with a snapshot of target and bytes.
    # Raises FileNotFoundError/OSError on missing or unreadable files and UnicodeDecodeError on
    # non-UTF-8 content.
    """Read a sidebar file without normalizing its encoded contents."""
    target = path.resolve(strict=True)
    content = target.read_bytes()
    return content.decode("utf-8"), _SidebarFileSnapshot(target=target, content=content)


def _atomic_write_sidebar_file(
    path: Path,
    source: str,
    expected: _SidebarFileSnapshot,
) -> _SidebarFileSnapshot:
    # [FUNCTION DOC]
    # HELPER: Atomically replaces a sidebar file with new text, refusing if it changed on disk.
    # Steps: resolve the path strictly; if the resolved target differs from the snapshot target
    # raise OSError ("File target changed on disk"); open the target write-only (without
    # truncation) to let the OS enforce permissions, and raise PermissionError(EACCES) if the
    # file has no write bits at all; remember its permission mode. Then create a temp file
    # ".<name>.*.tmp" in the same directory, chmod it to that mode, write the UTF-8 text,
    # flush and fsync; re-resolve and re-read the target and raise OSError ("File changed on
    # disk") if the target or bytes differ from the snapshot; finally os.replace the temp file
    # over the resolved target (so a symlink keeps pointing at the replaced file).
    # Returns a new snapshot (same target, new bytes). On any failure the temp file is
    # removed and an open descriptor closed (cleanup errors suppressed).
    # Raises OSError/PermissionError as described or from the file system.
    # Side effects: file system writes.
    # KNOWN ISSUE: there is a small race window between the final content check and os.replace.
    """Atomically replace an unchanged, writable file and preserve its mode/symlink."""
    target = path.resolve(strict=True)
    replacement = source.encode("utf-8")
    if target != expected.target:
        raise OSError(f"File target changed on disk; reopen before saving: {path}")
    # Ask the OS to enforce ownership/ACL rules without truncating the target.
    authorization = os.open(
        target,
        os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        target_stat = os.fstat(authorization)
        # Privileged processes may open 0444 files, but the editor treats an
        # explicitly read-only resource as not authorized for replacement.
        if target_stat.st_mode & 0o222 == 0:
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), target)
        target_mode = stat.S_IMODE(target_stat.st_mode)
    finally:
        os.close(authorization)

    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(raw_temporary)
        os.chmod(temporary, target_mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            _write_staged_utf8(handle, source)
            handle.flush()
            os.fsync(handle.fileno())
        current_target = path.resolve(strict=True)
        if current_target != expected.target or current_target.read_bytes() != expected.content:
            raise OSError(f"File changed on disk; reopen before saving: {path}")
        os.replace(temporary, target)
        temporary = None
        return _SidebarFileSnapshot(target=target, content=replacement)
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()


class SidebarFileEditor(Vertical):
    # [CLASS DOC]
    # Main-area editor view (a Vertical opened as an extension-style main view) for a file
    # chosen in the session sidebar. Shows title, path, a TextArea, help and status lines.
    # Ctrl+S saves in a background worker without closing; escape closes via the main-view
    # handle. Tracks the last saved text, the on-disk snapshot and a saving flag.
    """Main-area editor for a file selected from the session sidebar."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "close", "Close", show=False, priority=True),
        Binding("ctrl+s", "save", "Save", show=False, priority=True),
    ]

    def __init__(
        self,
        *,
        handle: MainViewHandle,
        path: Path,
        label: str,
        kind: str,
        source: str,
        snapshot: _SidebarFileSnapshot,
    ) -> None:
        # [FUNCTION DOC]
        # Stores the main-view handle, path, display label, kind (for the title), initial text,
        # snapshot; sets saved text to the initial text and saving to False. Widget id is
        # "sidebar-file-editor".
        super().__init__(id="sidebar-file-editor")
        self.handle = handle
        self.path = path
        self.label = label
        self.kind = kind
        self.source = source
        self._saved_source = source
        self._snapshot = snapshot
        self._saving = False

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields title "Edit <kind>: <label>", the path, a TextArea with the text, help and an
        # empty status line.
        yield Static(f"Edit {self.kind}: {self.label}", id="sidebar-file-editor-title")
        yield Static(str(self.path), id="sidebar-file-editor-path")
        yield TextArea(self.source, id="sidebar-file-editor-input")
        yield Static(
            "Ctrl+S saves - Escape closes",
            id="sidebar-file-editor-help",
        )
        yield Static("", id="sidebar-file-editor-status")

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Focuses the TextArea.
        self.query_one("#sidebar-file-editor-input", TextArea).focus()

    @property
    def is_dirty(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if the mounted TextArea text differs from the last saved text; False if
        # the TextArea is not mounted.
        """Return whether the mounted editor differs from its last saved source."""
        try:
            source = self.query_one("#sidebar-file-editor-input", TextArea).text
        except NoMatches:
            return False
        return source != self._saved_source

    def action_save(self) -> None:
        # [FUNCTION DOC]
        # Starts a save unless one is in progress: sets the saving flag, shows "Saving…" and runs
        # _save as a non-exclusive app worker.
        """Write the current editor contents without closing the editor."""
        if self._saving:
            return
        self._saving = True
        self.query_one("#sidebar-file-editor-status", Static).update("Saving…")
        self.app.run_worker(self._save(), exclusive=False)

    async def _save(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Reads the editor text and runs _atomic_write_sidebar_file in a thread. On any
        # exception shows "Could not save <path>: <error>" in the status line and an error
        # notification; on success updates the snapshot and saved text and shows/notifies
        # "Saved <path>". Always clears the saving flag. Never raises.
        source = self.query_one("#sidebar-file-editor-input", TextArea).text
        try:
            snapshot = await asyncio.to_thread(
                _atomic_write_sidebar_file,
                self.path,
                source,
                self._snapshot,
            )
        except Exception as exc:  # noqa: BLE001 - filesystem worker boundary
            message = f"Could not save {self.path}: {exc}"
            self.query_one("#sidebar-file-editor-status", Static).update(message)
            cast(TauTuiApp, self.app)._notify(message, severity="error")
        else:
            self._snapshot = snapshot
            self._saved_source = source
            message = f"Saved {self.path}"
            self.query_one("#sidebar-file-editor-status", Static).update(message)
            cast(TauTuiApp, self.app)._notify(message)
        finally:
            self._saving = False

    def action_close(self) -> None:
        # [FUNCTION DOC]
        # Closes the view via the handle (restoring the transcript). Unsaved edits are discarded.
        """Close the editor and restore the transcript."""
        self.handle.close()


class SessionPickerScreen(ModalScreen[str | None]):
    # [CLASS DOC]
    # Modal two-column session navigator: a projects column (working directories, local project
    # marked with a dot and always present) and a sessions column for the selected project,
    # under a search field, with Active/Archived tabs. Enter resumes (dismisses with session
    # id), ctrl+enter or delete archives (or unarchives in archived view) the highlighted
    # session or whole project via app workers, f2 toggles archived view, left/right switch
    # columns, escape dismisses with None. Supports incremental background loading
    # (update_records/finish_loading) and optimistic removal/restoration of records after
    # archive operations.
    """Project-and-session navigator for indexed sessions."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("left", "focus_projects", "Projects", show=False),
        Binding("right", "focus_sessions", "Sessions", show=False),
        Binding("enter", "select_cursor", "Select", show=False),
        Binding("ctrl+enter", "archive_cursor", "Archive", show=False, priority=True),
        Binding("f2", "show_archived", "Archived", show=False, priority=True),
    ]

    CSS = """
    #session-picker {
        width: 110;
        max-width: 94%;
        height: auto;
        max-height: 85%;
    }

    #session-picker-tabs {
        height: 3;
    }

    #session-picker-tabs Button {
        width: 1fr;
        border: none;
    }

    #session-picker-tabs Button.-active {
        background: $tau-highlight-background;
        color: $tau-highlight-text;
        text-style: bold;
    }

    #session-picker-columns {
        height: auto;
        border: tall $tau-border;
        background: $tau-transcript-background;
    }

    .session-picker-column {
        height: auto;
        background: $tau-transcript-background;
    }

    #session-picker-project-column {
        width: 34;
        border-right: tall $tau-border;
    }

    #session-picker-session-column {
        width: 1fr;
    }

    .session-picker-column-title {
        height: 1;
        padding: 0 1;
        color: $tau-muted-text;
        text-style: bold;
    }

    .-active-column > .session-picker-column-title {
        color: $tau-accent;
    }

    #session-picker-project-list,
    #session-picker-list {
        height: auto;
        max-height: 16;
        border: none;
        background: $tau-transcript-background;
    }

    #session-picker-list {
        padding: 0 1;
    }
    """

    def __init__(
        self,
        records: Sequence[SessionCompletionRecord],
        *,
        archived_records: Sequence[SessionCompletionRecord] = (),
        local_cwd: Path,
        theme: TuiTheme,
        loading_other_projects: bool = False,
        current_project_loaded: bool = True,
    ) -> None:
        # [FUNCTION DOC]
        # Stores active and archived records as tuples, resolves the local cwd, sets theme, empty
        # search, active column "sessions", empty hidden-cwd set, groups records by project, selects
        # the first project, and stores loading flags.
        super().__init__()
        self.records = tuple(records)
        self.archived_records = tuple(archived_records)
        self.showing_archived = False
        self.local_cwd = local_cwd.resolve()
        self.theme = theme
        self.search_value = ""
        self.active_column: Literal["projects", "sessions"] = "sessions"
        self.hidden_cwds: set[Path] = set()
        self.records_by_project = self._group_records_by_project()
        self.project_cwds = tuple(self.records_by_project)
        self.selected_project_index = 0
        self.visible_records: tuple[SessionCompletionRecord, ...] = ()
        self.loading_other_projects = loading_other_projects
        self.current_project_loaded = current_project_loaded

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields the title, Active/Archived tab buttons, the search input, the two OptionList
        # columns (with titles) and the help line.
        """Compose project and session columns under one search field."""
        with Vertical(id="session-picker"):
            yield Static("Sessions", id="session-picker-title")
            with Horizontal(id="session-picker-tabs"):
                yield Button("Active", id="session-picker-active-tab")
                yield Button("Archived", id="session-picker-archived-tab")
            yield SessionPickerSearchInput(
                placeholder="Search sessions in selected project",
                id="session-picker-search",
            )
            with Horizontal(id="session-picker-columns"):
                with Vertical(
                    id="session-picker-project-column",
                    classes="session-picker-column",
                ):
                    yield Static("Projects", classes="session-picker-column-title")
                    yield OptionList(id="session-picker-project-list", markup=False, compact=True)
                with Vertical(
                    id="session-picker-session-column",
                    classes="session-picker-column -active-column",
                ):
                    yield Static(
                        "",
                        id="session-picker-session-title",
                        classes="session-picker-column-title",
                    )
                    yield OptionList(id="session-picker-list", markup=False, compact=True)
            yield Static("", id="session-picker-help")

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Focuses search and refreshes tabs, project list, session list and help.
        """Start in the current project's recent-session column."""
        self.query_one("#session-picker-search", Input).focus()
        self._refresh_tabs()
        self._refresh_project_list()
        self._refresh_session_list()
        self._update_help()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        # [FUNCTION DOC]
        # Switches to the active or archived view depending on which tab button was pressed.
        if event.button.id == "session-picker-active-tab":
            self._set_archive_view(False)
        elif event.button.id == "session-picker-archived-tab":
            self._set_archive_view(True)

    def on_input_changed(self, event: Input.Changed) -> None:
        # [FUNCTION DOC]
        # On search-field changes: stops the event, stores the query and refreshes the session list.
        """Filter sessions in the selected project."""
        if event.input.id != "session-picker-search":
            return
        event.stop()
        self.search_value = event.value
        self._refresh_session_list()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # [FUNCTION DOC]
        # On search submit: stops the event and performs the select action.
        if event.input.id != "session-picker-search":
            return
        event.stop()
        self.action_select_cursor()

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Maps up/down/left/right/enter/delete keys to cursor, column focus, select and archive
        # actions, stopping the event when matched.
        """Route navigation while keeping typing focus in the search field."""
        actions = {
            "up": self.action_cursor_up,
            "down": self.action_cursor_down,
            "left": self.action_focus_projects,
            "right": self.action_focus_sessions,
            "enter": self.action_select_cursor,
            "delete": self.action_archive_cursor,
        }
        action = actions.get(event.key)
        if action is not None:
            event.stop()
            action()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        # [FUNCTION DOC]
        # When the highlighted project changes, updates the selected project index and refreshes
        # the session list immediately.
        """Show sessions for the highlighted project immediately."""
        if event.option_list.id != "session-picker-project-list":
            return
        index = event.option_index
        if index == self.selected_project_index:
            return
        self.selected_project_index = index
        self._refresh_session_list()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        # [FUNCTION DOC]
        # Option selected: in the projects list, select that project, refresh sessions and then
        # unarchive it (archived view) or move focus to sessions; in the sessions list, unarchive
        # (archived view) or resume the highlighted session.
        event.stop()
        if event.option_list.id == "session-picker-project-list":
            self.selected_project_index = event.option_index
            self._refresh_session_list()
            if self.showing_archived:
                self.action_unarchive_cursor()
            else:
                self.action_focus_sessions()
            return
        if self.showing_archived:
            self.action_unarchive_cursor()
        else:
            self._select_visible_record()

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Moves the cursor up in the active column's list.
        self._active_list().action_cursor_up()

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Moves the cursor down in the active column's list.
        self._active_list().action_cursor_down()

    def action_focus_projects(self) -> None:
        # [FUNCTION DOC]
        # Makes the projects column active.
        self._set_active_column("projects")

    def action_focus_sessions(self) -> None:
        # [FUNCTION DOC]
        # Makes the sessions column active.
        self._set_active_column("sessions")

    def action_select_cursor(self) -> None:
        # [FUNCTION DOC]
        # Enter behavior: unarchive in archived view; otherwise focus sessions when in projects,
        # or dismiss with the highlighted session id.
        if self.showing_archived:
            self.action_unarchive_cursor()
        elif self.active_column == "projects":
            self.action_focus_sessions()
        else:
            self._select_visible_record()

    def action_show_active(self) -> None:
        # [FUNCTION DOC]
        # Switches to the active-sessions view.
        self._set_archive_view(False)

    def action_show_archived(self) -> None:
        # [FUNCTION DOC]
        # Toggles between active and archived views.
        self._set_archive_view(not self.showing_archived)

    def _set_archive_view(self, show_archived: bool) -> None:
        # [FUNCTION DOC]
        # HELPER: If the view changes, remembers the selected project's cwd, switches view,
        # regroups records, keeps that project selected if present (else index 0), and refreshes
        # tabs and both lists.
        if self.showing_archived == show_archived:
            return
        selected_cwd = self.project_cwds[self.selected_project_index] if self.project_cwds else None
        self.showing_archived = show_archived
        self.records_by_project = self._group_records_by_project()
        self.project_cwds = tuple(self.records_by_project)
        if selected_cwd in self.project_cwds:
            self.selected_project_index = self.project_cwds.index(selected_cwd)
        else:
            self.selected_project_index = 0
        self._refresh_tabs()
        self._refresh_project_list()
        self._refresh_session_list()

    def action_archive_cursor(self) -> None:
        # [FUNCTION DOC]
        # Archives (active view) or unarchives (archived view) the highlighted target: in the
        # projects column the whole project directory, in the sessions column the highlighted
        # session record. Work is done by app coroutines (_archive_session_project,
        # _unarchive_session_project, _archive_session_record, _unarchive_session_record) run as
        # non-exclusive workers that later call back into this screen. No-op when no projects.
        """Archive or unarchive the highlighted session or project."""
        app = cast(TauTuiApp, self.app)
        if not self.project_cwds:
            return
        if self.active_column == "projects":
            cwd = self.project_cwds[self.selected_project_index]
            worker = (
                app._unarchive_session_project(cwd, self)
                if self.showing_archived
                else app._archive_session_project(cwd, self)
            )
            app.run_worker(worker, exclusive=False)
            return
        index = self.query_one("#session-picker-list", OptionList).highlighted
        if index is not None and index < len(self.visible_records):
            record = self.visible_records[index]
            worker = (
                app._unarchive_session_record(record, self)
                if self.showing_archived
                else app._archive_session_record(record, self)
            )
            app.run_worker(worker, exclusive=False)

    def action_unarchive_cursor(self) -> None:
        # [FUNCTION DOC]
        # Alias of action_archive_cursor (which unarchives in archived view).
        self.action_archive_cursor()

    async def _remove_records(self, cwd: Path, session_id: str | None = None) -> None:
        # [FUNCTION DOC]
        # Called after a successful archive: if this screen is still on top, moves either all
        # records for the resolved cwd (session_id None) or one record from active to archived,
        # hides the cwd from the active view if it has no remaining active sessions, and rebuilds
        # the lists. Raises StopIteration if session_id is not found among the records.
        """Move newly archived metadata out of the active picker."""
        if self.app.screen is not self:
            return
        resolved_cwd = Path(cwd).resolve()
        if session_id is None:
            moved = tuple(
                record for record in self.records if Path(record.cwd).resolve() == resolved_cwd
            )
            self.records = tuple(
                record for record in self.records if Path(record.cwd).resolve() != resolved_cwd
            )
            self.archived_records = (*self.archived_records, *moved)
            self.hidden_cwds.add(resolved_cwd)
        else:
            record = self._record_by_id(session_id)
            self.records = tuple(item for item in self.records if item.id != session_id)
            self.archived_records = (*self.archived_records, record)
            if not any(Path(item.cwd).resolve() == resolved_cwd for item in self.records):
                self.hidden_cwds.add(resolved_cwd)
        self._rebuild_visible_records()

    async def _restore_records(self, cwd: Path, session_id: str | None = None) -> None:
        # [FUNCTION DOC]
        # Called after a successful unarchive: if this screen is on top, moves all archived records
        # of the cwd (or the one record) back into active records, unhides the cwd and rebuilds.
        # Raises StopIteration if session_id is unknown.
        """Move restored metadata from the archived picker into the active list."""
        if self.app.screen is not self:
            return
        resolved_cwd = Path(cwd).resolve()
        if session_id is None:
            restored = tuple(
                record
                for record in self.archived_records
                if Path(record.cwd).resolve() == resolved_cwd
            )
            self.records = (*self.records, *restored)
            self.archived_records = tuple(
                record
                for record in self.archived_records
                if Path(record.cwd).resolve() != resolved_cwd
            )
        else:
            self.records = (*self.records, self._record_by_id(session_id))
            self.archived_records = tuple(
                record for record in self.archived_records if record.id != session_id
            )
        self.hidden_cwds.discard(resolved_cwd)
        self._rebuild_visible_records()

    def _record_by_id(self, session_id: str) -> SessionCompletionRecord:
        # [FUNCTION DOC]
        # HELPER: Returns the active or archived record with the given id; raises StopIteration if
        # none matches.
        return next(
            record for record in (*self.records, *self.archived_records) if record.id == session_id
        )

    def _rebuild_visible_records(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Regroups records, clamps the selected project index into range (0 when empty)
        # and refreshes both lists.
        self.records_by_project = self._group_records_by_project()
        self.project_cwds = tuple(self.records_by_project)
        if self.project_cwds:
            self.selected_project_index = min(
                self.selected_project_index, len(self.project_cwds) - 1
            )
        else:
            self.selected_project_index = 0
        self._refresh_project_list()
        self._refresh_session_list()

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with None.
        self.dismiss(None)

    def update_records(
        self,
        records: Sequence[SessionCompletionRecord],
        *,
        archived_records: Sequence[SessionCompletionRecord] | None = None,
        loading_other_projects: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # Replaces records after background loading, preserving the selected project (by cwd) and
        # the highlighted session (by id) when they still exist; optionally replaces archived
        # records; updates the loading flag; marks the current project as loaded; refreshes lists.
        """Replace records after background loading while preserving navigation."""
        selected_cwd = self.project_cwds[self.selected_project_index] if self.project_cwds else None
        session_list = self.query_one("#session-picker-list", OptionList)
        selected_session_id = None
        if session_list.highlighted is not None and session_list.highlighted < len(
            self.visible_records
        ):
            selected_session_id = self.visible_records[session_list.highlighted].id

        self.records = tuple(records)
        if archived_records is not None:
            self.archived_records = tuple(archived_records)
        self.records_by_project = self._group_records_by_project()
        self.project_cwds = tuple(self.records_by_project)
        try:
            self.selected_project_index = self.project_cwds.index(selected_cwd)
        except ValueError:
            self.selected_project_index = 0
        self.loading_other_projects = loading_other_projects
        self.current_project_loaded = True
        self._refresh_project_list()
        self._refresh_session_list()

        if selected_session_id is not None:
            for index, record in enumerate(self.visible_records):
                if record.id == selected_session_id:
                    session_list.highlighted = index
                    break

    def finish_loading(self) -> None:
        # [FUNCTION DOC]
        # Clears the loading-other-projects flag (used when background loading fails) and updates
        # the help text.
        """Remove the loading state when a background refresh fails."""
        self.loading_other_projects = False
        self._update_help()

    def _active_list(self) -> OptionList:
        # [FUNCTION DOC]
        # HELPER: Returns the OptionList for the active column.
        selector = (
            "#session-picker-project-list"
            if self.active_column == "projects"
            else "#session-picker-list"
        )
        return self.query_one(selector, OptionList)

    def _set_active_column(self, column: Literal["projects", "sessions"]) -> None:
        # [FUNCTION DOC]
        # HELPER: Sets the active column, toggles the "-active-column" CSS class on both columns
        # and updates the help text.
        self.active_column = column
        projects = self.query_one("#session-picker-project-column", Vertical)
        sessions = self.query_one("#session-picker-session-column", Vertical)
        projects.set_class(column == "projects", "-active-column")
        sessions.set_class(column == "sessions", "-active-column")
        self._update_help()

    def _select_visible_record(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Dismisses with the highlighted visible session's id if the index is valid.
        index = self.query_one("#session-picker-list", OptionList).highlighted
        if index is not None and index < len(self.visible_records):
            self.dismiss(self.visible_records[index].id)

    def _group_records_by_project(
        self,
    ) -> dict[Path, tuple[SessionCompletionRecord, ...]]:
        # [FUNCTION DOC]
        # HELPER: Groups the current tab's records (archived or active) by resolved cwd, always
        # starting with the local cwd (possibly empty); in the active view, skips hidden cwds.
        # Returns a mapping from path to a tuple of records, in insertion order.
        """Group the selected tab's records once for linear picker refreshes."""
        source = self.archived_records if self.showing_archived else self.records
        grouped: dict[Path, list[SessionCompletionRecord]] = {self.local_cwd: []}
        for record in source:
            cwd = Path(record.cwd).resolve()
            if not self.showing_archived and cwd in self.hidden_cwds:
                continue
            grouped.setdefault(cwd, []).append(record)
        return {cwd: tuple(records) for cwd, records in grouped.items()}

    def _refresh_tabs(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Toggles the "-active" class on the tab buttons and sets the search placeholder
        # for the current view.
        active = self.query_one("#session-picker-active-tab", Button)
        archived = self.query_one("#session-picker-archived-tab", Button)
        active.set_class(not self.showing_archived, "-active")
        archived.set_class(self.showing_archived, "-active")
        self.query_one("#session-picker-search", Input).placeholder = (
            "Search archived sessions in selected project"
            if self.showing_archived
            else "Search sessions in selected project"
        )

    def _refresh_project_list(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Fills the projects OptionList with folder names (full path if no name), marking the
        # local project with a dot, and highlights the selected index.
        project_list = self.query_one("#session-picker-project-list", OptionList)
        items: list[str] = []
        for cwd in self.project_cwds:
            marker = "● " if cwd == self.local_cwd else "  "
            folder_name = cwd.name or str(cwd)
            items.append(f"{marker}{folder_name}")
        project_list.set_options(items)
        project_list.highlighted = self.selected_project_index if self.project_cwds else None

    def _refresh_session_list(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Updates the session-column title ("Recent sessions"/"Archived sessions" plus the
        # cwd), filters the selected project's records by the search text with
        # _filter_session_records, fills the sessions list with _session_picker_label labels,
        # highlights the first and updates help. Empty when there are no projects.
        label = "Archived sessions" if self.showing_archived else "Recent sessions"
        if not self.project_cwds:
            self.visible_records = ()
            self.query_one("#session-picker-session-title", Static).update(label)
        else:
            selected_cwd = self.project_cwds[self.selected_project_index]
            self.query_one("#session-picker-session-title", Static).update(
                f"{label} — {selected_cwd}"
            )
            project_records = self.records_by_project[selected_cwd]
            self.visible_records = _filter_session_records(project_records, self.search_value)
        session_list = self.query_one("#session-picker-list", OptionList)
        session_list.set_options(_session_picker_label(record) for record in self.visible_records)
        session_list.highlighted = 0 if self.visible_records else None
        self._update_help()

    def _update_help(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Chooses the help text based on loading state, empty state, archived view and
        # active column, and writes it into the help Static.
        if self.loading_other_projects and not self.current_project_loaded:
            text = "Loading sessions… - Escape closes"
        elif self.loading_other_projects:
            text = "Loading other projects… - Current sessions are ready - Escape closes"
        elif not self.project_cwds:
            text = "No active sessions or projects - Escape closes"
        elif not self.visible_records and self.active_column == "sessions":
            text = "No matching sessions - Left selects a project - Escape closes"
        elif self.showing_archived and self.active_column == "projects":
            text = "Up/Down selects project - Enter unarchives - F2 shows active - Escape closes"
        elif self.showing_archived:
            text = "Left selects project - Enter unarchives - F2 shows active - Escape closes"
        elif self.active_column == "projects":
            text = (
                "Up/Down selects project - Right opens sessions - "
                "Ctrl+Enter archives - F2 shows archived - Escape closes"
            )
        else:
            text = (
                "Left selects project - Up/Down navigates - Enter resumes - "
                "Ctrl+Enter archives - F2 shows archived - Escape closes"
            )
        self.query_one("#session-picker-help", Static).update(text)


class SkillPickerSearchInput(Input):
    # [CLASS DOC]
    # Search Input for the skill picker whose priority bindings and key handler keep up/down,
    # escape, f1 (show description) and ctrl+enter (show in transcript) local to the picker.
    """Search input that keeps skill-picker navigation local."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        Binding("up", "cursor_up", "Up", show=False, priority=True),
        Binding("down", "cursor_down", "Down", show=False, priority=True),
    ]

    def _picker(self) -> SkillPickerScreen:
        # [FUNCTION DOC]
        # HELPER: Returns the containing screen cast to SkillPickerScreen.
        return cast(SkillPickerScreen, self.screen)

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Routes up/down/escape/f1/ctrl+enter to the picker actions, stopping and
        # default-preventing the events.
        """Route picker control keys before the input edits its text."""
        if event.key == "up":
            event.stop()
            event.prevent_default()
            self.action_cursor_up()
        elif event.key == "down":
            event.stop()
            event.prevent_default()
            self.action_cursor_down()
        elif event.key == "escape":
            event.stop()
            event.prevent_default()
            self.action_cancel()
        elif event.key == "f1":
            event.stop()
            event.prevent_default()
            self.action_show_description()
        elif event.key == "ctrl+enter":
            event.stop()
            event.prevent_default()
            self.action_show_in_transcript()

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the picker's cursor-up action.
        self._picker().action_cursor_up()

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the picker's cursor-down action.
        self._picker().action_cursor_down()

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the picker's cancel action.
        self._picker().action_cancel()

    def action_show_description(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the picker's show-description action.
        self._picker().action_show_description()

    def action_show_in_transcript(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the picker's show-in-transcript action.
        self._picker().action_show_in_transcript()


@dataclass(frozen=True, slots=True)
class SkillPickerResult:
    # [CLASS DOC]
    # Frozen result of the skill picker: the chosen skill and action ("insert" into the prompt,
    # or "transcript" to show the full skill in the transcript).
    """A skill selection and the requested inspection action."""

    skill: Skill
    action: Literal["insert", "transcript"]


class SkillPickerScreen(ModalScreen[SkillPickerResult | None]):
    # [CLASS DOC]
    # Modal searchable list of all loaded skills, sorted case-insensitively by name. Enter
    # dismisses with an insert result, f1 opens a CommandOutputScreen with the description,
    # ctrl+enter dismisses with a transcript result, escape with None.
    """Searchable modal containing every loaded skill."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("up", "cursor_up", "Up", show=False, priority=True),
        Binding("down", "cursor_down", "Down", show=False, priority=True),
        Binding("enter", "select_cursor", "Insert", show=False, priority=True),
        Binding("f1", "show_description", "Description", show=False, priority=True),
        Binding("ctrl+enter", "show_in_transcript", "Transcript", show=False, priority=True),
    ]

    def __init__(self, skills: Sequence[Skill], *, theme: TuiTheme) -> None:
        # [FUNCTION DOC]
        # Stores skills sorted by casefolded name (all initially visible) and theme.
        super().__init__()
        self.skills = tuple(sorted(skills, key=lambda skill: skill.name.casefold()))
        self.visible_skills = self.skills
        self.theme = theme

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields title, SkillPickerSearchInput, the skill ListView and help line.
        with Vertical(id="skill-picker"):
            yield Static("Skills", id="skill-picker-title")
            yield SkillPickerSearchInput(placeholder="Search skills", id="skill-picker-search")
            yield ListView(id="skill-picker-list")
            yield Static("", id="skill-picker-help")

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Focuses search and fills the list with all skills.
        self.query_one("#skill-picker-search", Input).focus()
        self._refresh_skill_list("")

    def on_input_changed(self, event: Input.Changed) -> None:
        # [FUNCTION DOC]
        # On search changes, stops the event and refilters.
        if event.input.id == "skill-picker-search":
            event.stop()
            self._refresh_skill_list(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # [FUNCTION DOC]
        # On search submit, stops the event and selects the highlighted skill.
        if event.input.id == "skill-picker-search":
            event.stop()
            self._select_visible_skill()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # [FUNCTION DOC]
        # Stops the event and selects the highlighted skill.
        event.stop()
        self._select_visible_skill()

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Decrements the list index (not below 0) if an index is set.
        skill_list = self.query_one("#skill-picker-list", ListView)
        if skill_list.index is not None:
            skill_list.index = max(0, skill_list.index - 1)

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Increments the list index up to the last visible skill if an index is set.
        skill_list = self.query_one("#skill-picker-list", ListView)
        if skill_list.index is not None:
            skill_list.index = min(len(self.visible_skills) - 1, skill_list.index + 1)

    def action_select_cursor(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with an insert result for the selected skill, if any.
        skill = self._selected_skill()
        if skill is not None:
            self.dismiss(SkillPickerResult(skill, "insert"))

    def action_show_description(self) -> None:
        # [FUNCTION DOC]
        # Pushes a CommandOutputScreen showing the selected skill's description (or "No
        # description").
        skill = self._selected_skill()
        if skill is not None:
            self.app.push_screen(
                CommandOutputScreen(
                    f"Skill description: {skill.name}",
                    skill.description or "No description",
                    theme=self.theme,
                )
            )

    def action_show_in_transcript(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with a transcript result for the selected skill, if any.
        skill = self._selected_skill()
        if skill is not None:
            self.dismiss(SkillPickerResult(skill, "transcript"))

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with None.
        self.dismiss(None)

    def _selected_skill(self) -> Skill | None:
        # [FUNCTION DOC]
        # HELPER: Returns the visible skill at the list index, or None if no index or no visible
        # skills.
        index = self.query_one("#skill-picker-list", ListView).index
        if index is None or not self.visible_skills:
            return None
        return self.visible_skills[index]

    def _select_visible_skill(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Same as action_select_cursor.
        self.action_select_cursor()

    def _refresh_skill_list(self, search: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Filters skills whose name or description contains the casefolded, stripped query
        # (all when empty), rebuilds list rows (name and description labels), selects the first,
        # and sets help text for no skills / no matches / normal.
        query = search.casefold().strip()
        self.visible_skills = tuple(
            skill
            for skill in self.skills
            if not query
            or query in skill.name.casefold()
            or query in (skill.description or "").casefold()
        )
        skill_list = self.query_one("#skill-picker-list", ListView)
        skill_list.clear()
        skill_list.extend(
            ListItem(
                Horizontal(
                    Label(skill.name, classes="skill-picker-name", markup=False),
                    Label(
                        skill.description or "No description",
                        classes="skill-picker-description",
                        markup=False,
                    ),
                    classes="skill-picker-row",
                )
            )
            for skill in self.visible_skills
        )
        skill_list.index = 0 if self.visible_skills else None
        if not self.skills:
            help_text = "No skills loaded - Escape closes"
        elif not self.visible_skills:
            help_text = "No matching skills - Escape closes"
        else:
            help_text = "Enter inserts - F1 describes - Ctrl+Enter shows full skill"
        self.query_one("#skill-picker-help", Static).update(help_text)


@dataclass(frozen=True, slots=True)
class TreePickerResult:
    # [CLASS DOC]
    # Frozen result of the tree picker: the session entry id to branch from, whether to
    # summarize the abandoned branch, and optional custom summary instructions.
    """Tree-picker branch selection."""

    entry_id: str
    summarize: bool = False
    custom_instructions: str | None = None


class _TreePickerListItem(ListItem):
    # [CLASS DOC]
    # HELPER: ListItem for one session-tree choice whose label (from _tree_picker_label) is
    # re-rendered with highlight-aware colors when the highlight changes.
    """Tree entry that keeps inline label colors readable when highlighted."""

    def __init__(
        self,
        choice: SessionTreeChoice,
        *,
        theme: TuiTheme,
        show_label_timestamp: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # Stores choice, theme and timestamp flag, then builds the item with a non-markup Label
        # rendered by _tree_picker_label.
        self.choice = choice
        self.theme = theme
        self.show_label_timestamp = show_label_timestamp
        super().__init__(
            Label(
                _tree_picker_label(
                    choice,
                    theme=theme,
                    show_label_timestamp=show_label_timestamp,
                ),
                markup=False,
            )
        )

    def watch_highlighted(self, value: bool) -> None:
        # [FUNCTION DOC]
        # Runs the base highlight watcher and then re-renders the label with the highlighted flag.
        """Recolor inline label spans when the list highlight changes."""
        super().watch_highlighted(value)
        self.query_one(Label).update(
            _tree_picker_label(
                self.choice,
                theme=self.theme,
                highlighted=value,
                show_label_timestamp=self.show_label_timestamp,
            )
        )


class TreePickerScreen(ModalScreen[TreePickerResult | None]):
    # [CLASS DOC]
    # Modal picker over the session tree for branching (/tree). Keys: enter branches, s branches
    # with summary, c branches with custom summary instructions (via
    # BranchSummaryInstructionsScreen), l edits the entry's label (ExtensionInputScreen),
    # ctrl+t toggles tool-call entries, ctrl+f toggles labeled-only filtering, ctrl+l toggles
    # label timestamps, escape cancels. Label changes are persisted through an optional async
    # callback returning the change timestamp.
    """Modal picker for branching from a previous session entry."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "select_cursor", "Branch", show=False),
        Binding("s", "select_with_summary", "Summarize", show=False),
        Binding("c", "select_with_custom_summary", "Custom summary", show=False),
        Binding("ctrl+t", "toggle_tool_calls", "Tool calls", show=False),
        Binding("l", "edit_label", "Label", show=False),
        Binding("ctrl+f", "toggle_labeled_only", "Labeled", show=False),
        Binding("ctrl+l", "toggle_label_timestamps", "Label time", show=False),
    ]

    def __init__(
        self,
        choices: Sequence[SessionTreeChoice],
        *,
        theme: TuiTheme,
        on_label_change: Callable[[str, str | None], Awaitable[float]] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Stores choices, theme, the optional label-change callback, and initial filter flags (tool
        # calls shown, all entries, timestamps hidden).
        super().__init__()
        self.choices = tuple(choices)
        self.theme = theme
        self.on_label_change = on_label_change
        self.show_tool_calls = True
        self.labeled_only = False
        self.show_label_timestamps = False

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields title, a ListView of tree items and the help line.
        """Compose the tree picker."""
        with Vertical(id="tree-picker"):
            yield Static("Session Tree", id="tree-picker-title")
            yield ListView(
                *self._list_items(),
                id="tree-picker-list",
            )
            yield Static(
                self._help_text(),
                id="tree-picker-help",
            )

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Selects the active choice (via _active_tree_choice_index) and focuses the list.
        """Focus the tree list for keyboard navigation."""
        tree_list = self.query_one("#tree-picker-list", ListView)
        tree_list.index = _active_tree_choice_index(self.choices)
        tree_list.focus()

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Routes up/down/enter/s/c/ctrl+t/l/ctrl+f/ctrl+l to the corresponding actions, stopping
        # the events.
        """Route tree picker keys to the list."""
        if event.key == "up":
            event.stop()
            self.action_cursor_up()
        elif event.key == "down":
            event.stop()
            self.action_cursor_down()
        elif event.key == "enter":
            event.stop()
            self.action_select_cursor()
        elif event.key == "s":
            event.stop()
            self.action_select_with_summary()
        elif event.key == "c":
            event.stop()
            self.action_select_with_custom_summary()
        elif event.key == "ctrl+t":
            event.stop()
            self.action_toggle_tool_calls()
        elif event.key == "l":
            event.stop()
            self.action_edit_label()
        elif event.key == "ctrl+f":
            event.stop()
            self.action_toggle_labeled_only()
        elif event.key == "ctrl+l":
            event.stop()
            self.action_toggle_label_timestamps()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # [FUNCTION DOC]
        # Dismisses with a plain branch result for the selected visible choice.
        """Dismiss with the selected entry id."""
        self.dismiss(TreePickerResult(entry_id=self._visible_choices()[event.index].entry_id))

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Moves the list cursor up.
        """Move to the previous tree entry."""
        self.query_one("#tree-picker-list", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Moves the list cursor down.
        """Move to the next tree entry."""
        self.query_one("#tree-picker-list", ListView).action_cursor_down()

    def action_select_cursor(self) -> None:
        # [FUNCTION DOC]
        # Selects the highlighted entry (triggers on_list_view_selected).
        """Branch from the highlighted entry without a summary."""
        self.query_one("#tree-picker-list", ListView).action_select_cursor()

    def action_select_with_summary(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with summarize=True for the highlighted entry, if any.
        """Branch from the highlighted entry with a branch summary."""
        tree_list = self.query_one("#tree-picker-list", ListView)
        index = tree_list.index
        if index is None:
            return
        self.dismiss(
            TreePickerResult(entry_id=self._visible_choices()[index].entry_id, summarize=True)
        )

    def action_select_with_custom_summary(self) -> None:
        # [FUNCTION DOC]
        # Pushes BranchSummaryInstructionsScreen; when it returns, dismisses with a summarize
        # result carrying the instructions for the entry at the captured index.
        """Branch from the highlighted entry with custom summary instructions."""
        tree_list = self.query_one("#tree-picker-list", ListView)
        index = tree_list.index
        if index is None:
            return
        self.app.push_screen(
            BranchSummaryInstructionsScreen(theme=self.theme),
            callback=lambda instructions: self._dismiss_with_custom_summary(index, instructions),
        )

    def _dismiss_with_custom_summary(self, index: int, instructions: str | None) -> None:
        # [FUNCTION DOC]
        # HELPER: Ignores cancellation (None) and stale indexes; otherwise dismisses with
        # summarize=True and the custom instructions.
        if instructions is None:
            return
        visible_choices = self._visible_choices()
        if index >= len(visible_choices):
            return
        self.dismiss(
            TreePickerResult(
                entry_id=visible_choices[index].entry_id,
                summarize=True,
                custom_instructions=instructions,
            )
        )

    def action_toggle_tool_calls(self) -> None:
        # [FUNCTION DOC]
        # Toggles tool-call visibility and refreshes the list in a worker, keeping the selection.
        """Toggle tool-call entries in the tree picker."""
        selected_entry_id = self._selected_entry_id()
        self.show_tool_calls = not self.show_tool_calls
        self.run_worker(self._refresh_choices(selected_entry_id=selected_entry_id))

    def action_toggle_labeled_only(self) -> None:
        # [FUNCTION DOC]
        # Toggles the labeled-only filter and refreshes in a worker, keeping the selection.
        """Toggle the Pi-style labeled-entry filter."""
        selected_entry_id = self._selected_entry_id()
        self.labeled_only = not self.labeled_only
        self.run_worker(self._refresh_choices(selected_entry_id=selected_entry_id))

    def action_toggle_label_timestamps(self) -> None:
        # [FUNCTION DOC]
        # Toggles label timestamp display and refreshes in a worker, keeping the selection.
        """Toggle display of the latest label-change timestamp."""
        selected_entry_id = self._selected_entry_id()
        self.show_label_timestamps = not self.show_label_timestamps
        self.run_worker(self._refresh_choices(selected_entry_id=selected_entry_id))

    def action_edit_label(self) -> None:
        # [FUNCTION DOC]
        # Opens a text prompt prefilled with the selected entry's label; the result is passed to
        # _handle_label_input. No-op without a selection.
        """Open a prefilled editor; submitting an empty value clears the label."""
        selected = self._selected_choice()
        if selected is None:
            return
        self.app.push_screen(
            ExtensionInputScreen(
                "Label this session entry",
                "Empty clears the label",
                theme=self.theme,
                value=selected.bookmark_label or "",
            ),
            callback=lambda value: self._handle_label_input(selected.entry_id, value),
        )

    def _handle_label_input(self, entry_id: str, value: str | None) -> None:
        # [FUNCTION DOC]
        # HELPER: Ignores cancel; otherwise applies the label in a worker.
        if value is None:
            return
        self.run_worker(self._apply_label(entry_id, value))

    async def _apply_label(self, entry_id: str, value: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Strips the value (empty means clear, i.e. None), calls the label-change callback
        # to persist it and get a timestamp; on any error (including a missing callback, which
        # raises RuntimeError internally) shows an error notification and returns. On success
        # updates the matching choice's label and timestamp (cleared when label removed) and
        # refreshes the list selecting that entry.
        normalized = value.strip() or None
        try:
            if self.on_label_change is None:
                raise RuntimeError("Session labels are not available.")
            timestamp = await self.on_label_change(entry_id, normalized)
        except Exception as exc:  # noqa: BLE001 - keep the tree open and surface persistence errors
            self.app.notify(f"Error: {exc}", severity="error")
            return
        self.choices = tuple(
            replace(
                choice,
                bookmark_label=normalized,
                label_timestamp=timestamp if normalized is not None else None,
            )
            if choice.entry_id == entry_id
            else choice
            for choice in self.choices
        )
        await self._refresh_choices(selected_entry_id=entry_id)

    async def _refresh_choices(self, *, selected_entry_id: str | None = None) -> None:
        # [FUNCTION DOC]
        # HELPER: Rebuilds the list items for the current filters, reselects the given (or current)
        # entry via _tree_choice_index, and updates the help text. Awaits list clear/extend.
        selected_entry_id = selected_entry_id or self._selected_entry_id()
        tree_list = self.query_one("#tree-picker-list", ListView)
        await tree_list.clear()
        await tree_list.extend(self._list_items())
        visible_choices = self._visible_choices()
        tree_list.index = (
            _tree_choice_index(visible_choices, selected_entry_id) if visible_choices else None
        )
        self.query_one("#tree-picker-help", Static).update(self._help_text())

    def _selected_entry_id(self) -> str | None:
        # [FUNCTION DOC]
        # HELPER: Returns the entry id of the highlighted visible choice, or None.
        tree_list = self.query_one("#tree-picker-list", ListView)
        index = tree_list.index
        visible_choices = self._visible_choices()
        if index is None or index >= len(visible_choices):
            return None
        return visible_choices[index].entry_id

    def _selected_choice(self) -> SessionTreeChoice | None:
        # [FUNCTION DOC]
        # HELPER: Returns the full choice for the highlighted entry id, or None.
        entry_id = self._selected_entry_id()
        return next((choice for choice in self.choices if choice.entry_id == entry_id), None)

    def _visible_choices(self) -> tuple[SessionTreeChoice, ...]:
        # [FUNCTION DOC]
        # HELPER: Returns choices passing the filters (tool calls hidden when disabled; only
        # labeled entries when labeled-only).
        return tuple(
            choice
            for choice in self.choices
            if (self.show_tool_calls or not choice.is_tool_call)
            and (not self.labeled_only or choice.bookmark_label is not None)
        )

    def _list_items(self) -> list[ListItem]:
        # [FUNCTION DOC]
        # HELPER: Builds a _TreePickerListItem per visible choice.
        return [
            _TreePickerListItem(
                choice,
                theme=self.theme,
                show_label_timestamp=self.show_label_timestamps,
            )
            for choice in self._visible_choices()
        ]

    def _help_text(self) -> str:
        # [FUNCTION DOC]
        # HELPER: Builds the one-line help text including the current filter states.
        tool_call_state = "shown" if self.show_tool_calls else "hidden"
        labeled_state = "only" if self.labeled_only else "all"
        time_state = "shown" if self.show_label_timestamps else "hidden"
        return (
            "Enter branch · L label/clear · S summary · C custom · "
            f"Ctrl+T tool calls {tool_call_state} · Ctrl+F labels {labeled_state} · "
            f"Ctrl+L times {time_state} · Esc close"
        )

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with None.
        """Close the picker without selecting an entry."""
        self.dismiss(None)


class BranchSummaryInstructionsScreen(ModalScreen[str | None]):
    # [CLASS DOC]
    # Modal TextArea prompt for custom branch-summary instructions. Ctrl+Enter dismisses with
    # the stripped text (None if empty); escape dismisses with None.
    """Prompt for custom branch-summary instructions."""

    BINDINGS: ClassVar[list[BindingEntry]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, *, theme: TuiTheme) -> None:
        # [FUNCTION DOC]
        # Stores the theme.
        super().__init__()
        self.theme = theme

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields title, a TextArea and help line.
        """Compose the custom-instructions prompt."""
        with Vertical(id="branch-summary-instructions"):
            yield Static(
                "Custom summarization instructions",
                id="branch-summary-instructions-title",
            )
            yield TextArea(id="branch-summary-instructions-input")
            yield Static(
                "Ctrl+Enter submits - Escape returns to tree",
                id="branch-summary-instructions-help",
            )

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Focuses the TextArea.
        """Focus the instruction editor."""
        self.query_one("#branch-summary-instructions-input", TextArea).focus()

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Ctrl+Enter submits; escape cancels; both stop the event.
        """Submit on Ctrl+Enter and cancel on Escape."""
        if event.key == "ctrl+enter":
            event.stop()
            self.action_submit()
        elif event.key == "escape":
            event.stop()
            self.action_cancel()

    def action_submit(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with the stripped text, or None if empty.
        """Submit custom instructions."""
        value = self.query_one("#branch-summary-instructions-input", TextArea).text.strip()
        self.dismiss(value or None)

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with None.
        """Cancel custom instructions."""
        self.dismiss(None)


class CommandOutputScroll(VerticalScroll):
    # [CLASS DOC]
    # VerticalScroll with priority up/down bindings that scroll exactly one line.
    """Scrollable command output area with deterministic arrow-key scrolling."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("up", "scroll_up", "Scroll up", show=False, priority=True),
        Binding("down", "scroll_down", "Scroll down", show=False, priority=True),
    ]

    def action_scroll_up(self) -> None:
        # [FUNCTION DOC]
        # Decrements scroll_y by one, not below zero.
        """Scroll command output up."""
        self.scroll_y = max(0, self.scroll_y - 1)

    def action_scroll_down(self) -> None:
        # [FUNCTION DOC]
        # Increments scroll_y by one, not beyond the maximum.
        """Scroll command output down."""
        self.scroll_y = min(self.max_scroll_y, self.scroll_y + 1)


class CommandOutputScreen(ModalScreen[None]):
    # [CLASS DOC]
    # Dismissible modal showing slash-command output text in a scrollable area. Escape or enter
    # closes; up/down scroll. auto_copy_selection controls whether selecting text copies it
    # (and the help text mentions it).
    """Dismissible modal for slash-command output."""

    auto_copy_selection: bool = False

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "close", "Close"),
        Binding("enter", "close", "Close"),
        Binding("up", "scroll_up", "Scroll up", show=False, priority=True),
        Binding("down", "scroll_down", "Scroll down", show=False, priority=True),
    ]

    def __init__(
        self,
        title: str,
        message: str,
        *,
        theme: TuiTheme,
        auto_copy_selection: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # Stores title, message, theme and the auto-copy flag.
        super().__init__()
        self.title_text = title
        self.message = message
        self.theme = theme
        self.auto_copy_selection = auto_copy_selection

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields the title, a CommandOutputScroll containing the non-markup body, and help text.
        """Compose command output."""
        with Vertical(id="command-output"):
            yield Static(self.title_text, id="command-output-title")
            with CommandOutputScroll(id="command-output-scroll"):
                yield Static(self.message, id="command-output-body", markup=False)
            yield Static(self._help_text(), id="command-output-help")

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Focuses the scroll area.
        """Focus the scroll area so arrow keys navigate long output."""
        self.query_one("#command-output-scroll", VerticalScroll).focus()

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Routes up/down keys to scroll actions, stopping them.
        """Route arrow keys to the command output scroll area."""
        if event.key == "up":
            event.stop()
            self.action_scroll_up()
        elif event.key == "down":
            event.stop()
            self.action_scroll_down()

    def action_close(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with None.
        """Close the command output modal."""
        self.dismiss(None)

    def _help_text(self) -> str:
        # [FUNCTION DOC]
        # HELPER: Returns the help line, mentioning select-to-copy when auto-copy is enabled.
        if self.auto_copy_selection:
            return "Select text to copy - Enter or Escape closes"
        return "Enter or Escape closes"

    def action_scroll_up(self) -> None:
        # [FUNCTION DOC]
        # Scrolls the output area up one line.
        """Scroll command output up."""
        self.query_one("#command-output-scroll", CommandOutputScroll).action_scroll_up()

    def action_scroll_down(self) -> None:
        # [FUNCTION DOC]
        # Scrolls the output area down one line.
        """Scroll command output down."""
        self.query_one("#command-output-scroll", CommandOutputScroll).action_scroll_down()


class LoginProviderSearchInput(Input):
    # [CLASS DOC]
    # Search Input for the login provider picker that keeps up/down/escape local, forwarding
    # them to the picker.
    """Search input that keeps provider-picker navigation local."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        Binding("up", "cursor_up", "Up", show=False, priority=True),
        Binding("down", "cursor_down", "Down", show=False, priority=True),
    ]

    def _picker(self) -> LoginProviderPickerScreen:
        # [FUNCTION DOC]
        # HELPER: Returns the containing screen cast to LoginProviderPickerScreen.
        return cast(LoginProviderPickerScreen, self.screen)

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Routes up/down/escape to the picker, stopping and default-preventing the events.
        """Route picker control keys before the input edits its text."""
        if event.key == "up":
            event.stop()
            event.prevent_default()
            self.action_cursor_up()
        elif event.key == "down":
            event.stop()
            event.prevent_default()
            self.action_cursor_down()
        elif event.key == "escape":
            event.stop()
            event.prevent_default()
            self.action_cancel()

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the picker's cursor-up action.
        """Move the provider picker selection up."""
        self._picker().action_cursor_up()

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the picker's cursor-down action.
        """Move the provider picker selection down."""
        self._picker().action_cursor_down()

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the picker's cancel action.
        """Close the provider picker."""
        self._picker().action_cancel()


class _LoginFlowAction(Enum):
    # [CLASS DOC]
    # HELPER: Enum of navigation signals returned by nested login screens; BACK means "return
    # to the previous step of the login flow".
    """Navigation actions returned by nested login screens."""

    BACK = auto()


class LoginProviderPickerScreen(ModalScreen[str | _LoginFlowAction | None]):
    # [CLASS DOC]
    # Modal searchable provider picker used in the login flow (and similar flows). Dismisses
    # with the chosen provider name, with _LoginFlowAction.BACK on escape when back_on_cancel is
    # set (otherwise None), or with None on ctrl+d (close the whole flow).
    """Searchable provider picker for the TUI login flow."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+d", "close", "Close", priority=True),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "select_cursor", "Select", show=False),
    ]

    def __init__(
        self,
        providers: Sequence[ProviderCatalogEntry],
        *,
        theme: TuiTheme,
        title: str = "Login",
        back_on_cancel: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # Stores providers, back-on-cancel flag, visible providers (all), theme and title.
        super().__init__()
        self.providers = tuple(providers)
        self.back_on_cancel = back_on_cancel
        self.visible_providers = self.providers
        self.theme = theme
        self.title_text = title

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields the title, a LoginProviderSearchInput, a ListView of provider labels
        # (_login_provider_label) and help text.
        """Compose the provider picker."""
        with Vertical(id="login-provider-picker"):
            yield Static(self.title_text, id="login-provider-title")
            yield LoginProviderSearchInput(
                placeholder="Search providers",
                id="login-provider-search",
            )
            yield ListView(
                *[
                    ListItem(Label(_login_provider_label(provider), markup=False))
                    for provider in self.providers
                ],
                id="login-provider-list",
            )
            yield Static("Enter selects - Escape closes", id="login-provider-help")

    async def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Focuses search and awaits a list refresh (so the first item is selectable).
        """Focus the provider search field."""
        self.query_one("#login-provider-search", Input).focus()
        await self._refresh_provider_list()

    async def on_input_changed(self, event: Input.Changed) -> None:
        # [FUNCTION DOC]
        # On search changes: stops the event, filters providers with _filter_login_providers and
        # awaits a list refresh.
        """Filter providers as the search value changes."""
        if event.input.id != "login-provider-search":
            return
        event.stop()
        self.visible_providers = _filter_login_providers(self.providers, event.value)
        await self._refresh_provider_list()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # [FUNCTION DOC]
        # On search submit: stops the event and selects the highlighted (or first) provider.
        """Select the highlighted provider from the search field."""
        if event.input.id != "login-provider-search":
            return
        event.stop()
        self._select_visible_provider()

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Routes up/down/enter to cursor and select actions, stopping the events.
        """Route provider picker keys to the list."""
        if event.key == "up":
            event.stop()
            self.action_cursor_up()
        elif event.key == "down":
            event.stop()
            self.action_cursor_down()
        elif event.key == "enter":
            event.stop()
            self.action_select_cursor()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # [FUNCTION DOC]
        # Stops the event and selects the highlighted provider.
        """Dismiss with the selected provider name."""
        event.stop()
        self._select_visible_provider()

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Moves the list cursor up.
        """Move to the previous provider."""
        self.query_one("#login-provider-list", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Moves the list cursor down.
        """Move to the next provider."""
        self.query_one("#login-provider-list", ListView).action_cursor_down()

    def action_select_cursor(self) -> None:
        # [FUNCTION DOC]
        # Selects the highlighted provider.
        """Select the highlighted provider."""
        self._select_visible_provider()

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with BACK if back_on_cancel, else None.
        """Go back in a login flow, or close a standalone provider picker."""
        self.dismiss(_LoginFlowAction.BACK if self.back_on_cancel else None)

    def action_close(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with None (ends the whole login flow).
        """Close the entire login flow."""
        self.dismiss(None)

    def _select_visible_provider(self) -> None:
        # [FUNCTION DOC]
        # HELPER: If any providers are visible, dismisses with the name of the highlighted provider,
        # falling back to the first when no index is set.
        if not self.visible_providers:
            return
        provider_list = self.query_one("#login-provider-list", ListView)
        # Fall back to the first match: submitting from the search field can
        # land here before the refreshed list has applied its highlight.
        index = provider_list.index
        self.dismiss(self.visible_providers[0 if index is None else index].name)

    async def _refresh_provider_list(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Awaits clearing and refilling the list with visible providers, then sets index 0
        # (None when empty) and updates help text ("No matching providers" when empty).
        provider_list = self.query_one("#login-provider-list", ListView)
        # Await the mounts: assigning the index while the list is still empty
        # validates it back to None, leaving the first provider unreachable
        # with the down key (issue #494).
        await provider_list.clear()
        await provider_list.extend(
            [
                ListItem(Label(_login_provider_label(provider), markup=False))
                for provider in self.visible_providers
            ]
        )
        provider_list.index = 0 if self.visible_providers else None
        help_text = (
            "Enter selects - Escape closes"
            if self.visible_providers
            else "No matching providers - Escape closes"
        )
        self.query_one("#login-provider-help", Static).update(help_text)


@dataclass(frozen=True, slots=True)
class CustomProviderLoginResult:
    # [CLASS DOC]
    # Frozen record of details collected by the custom OpenAI-compatible provider login flow:
    # provider id, display name, base URL, API-key environment variable name, model ids, default
    # model and the API key itself.
    """Provider details collected by the custom-provider login flow."""

    provider_name: str
    display_name: str
    base_url: str
    api_key_env: str
    models: tuple[str, ...]
    default_model: str
    api_key: str


class LoginMethodPickerScreen(ModalScreen[str | None]):
    # [CLASS DOC]
    # Modal picker for the login method: "subscription" (OAuth), "api-key" (built-in provider
    # key) or "custom" (OpenAI-compatible provider). Dismisses with that string, or None on
    # escape/ctrl+d. Arrow navigation wraps around.
    """Login method picker for the TUI login flow."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("ctrl+d", "cancel", "Close", priority=True),
        Binding("up", "cursor_up", "Up", show=False, priority=True),
        Binding("down", "cursor_down", "Down", show=False, priority=True),
        Binding("enter", "select_cursor", "Select", show=False, priority=True),
    ]

    def __init__(self, *, theme: TuiTheme) -> None:
        # [FUNCTION DOC]
        # Stores the theme.
        super().__init__()
        self.theme = theme

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields title, intro, a LoginMethodListView with the three methods (with ids) and help.
        """Compose the login method picker."""
        with Vertical(id="login-method-picker"):
            yield Static("Login", id="login-method-title")
            yield Static("Choose how to authenticate.", id="login-method-intro")
            yield LoginMethodListView(
                ListItem(
                    Label("Subscription — OAuth account", markup=False),
                    id="login-method-subscription",
                ),
                ListItem(
                    Label("API key — built-in provider", markup=False),
                    id="login-method-api-key",
                ),
                ListItem(
                    Label("Custom provider — OpenAI-compatible", markup=False),
                    id="login-method-custom",
                ),
                id="login-method-list",
            )
            yield Static("Enter selects - Escape/Ctrl+D closes", id="login-method-help")

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Selects the first (subscription) item and focuses the list.
        """Focus the default subscription method."""
        method_list = self.query_one("#login-method-list", ListView)
        method_list.index = 0
        method_list.focus()

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Routes up/down/enter to the wrapping cursor and select actions, stopping the events.
        """Route arrow keys between login method buttons."""
        if event.key == "up":
            event.stop()
            self.action_cursor_up()
        elif event.key == "down":
            event.stop()
            self.action_cursor_down()
        elif event.key == "enter":
            event.stop()
            self.action_select_cursor()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        # [FUNCTION DOC]
        # Maps a pressed button id to its method string and dismisses.
        # KNOWN ISSUE: the screen composes no buttons, so this handler is effectively dead code.
        """Dismiss with the selected login method."""
        if event.button.id == "login-method-subscription":
            self.dismiss("subscription")
        elif event.button.id == "login-method-api-key":
            self.dismiss("api-key")
        elif event.button.id == "login-method-custom":
            self.dismiss("custom")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # [FUNCTION DOC]
        # Maps the selected list item id to "subscription", "api-key" or "custom" and dismisses.
        """Dismiss with the selected login method."""
        if event.item.id == "login-method-subscription":
            self.dismiss("subscription")
        elif event.item.id == "login-method-api-key":
            self.dismiss("api-key")
        elif event.item.id == "login-method-custom":
            self.dismiss("custom")

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with None.
        """Close without selecting a login method."""
        self.dismiss(None)

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Moves selection up with wrap-around.
        """Focus the previous login method."""
        self._move_method_cursor(offset=-1)

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Moves selection down with wrap-around.
        """Focus the next login method."""
        self._move_method_cursor(offset=1)

    def action_select_cursor(self) -> None:
        # [FUNCTION DOC]
        # Selects the highlighted method.
        """Select the currently focused login method."""
        self.query_one("#login-method-list", ListView).action_select_cursor()

    def _move_method_cursor(self, *, offset: int) -> None:
        # [FUNCTION DOC]
        # HELPER: Moves the list index by offset modulo the item count (from 0 if unset); sets None
        # when empty.
        method_list = self.query_one("#login-method-list", ListView)
        item_count = len(method_list.children)
        if item_count == 0:
            method_list.index = None
            return
        current_index = method_list.index if method_list.index is not None else 0
        method_list.index = (current_index + offset) % item_count


class LoginMethodListView(ListView):
    # [CLASS DOC]
    # ListView whose cursor up/down actions wrap around at the ends.
    """List view with wrapping arrow navigation for the login method picker."""

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Moves to the previous item with wrap-around.
        """Move to the previous login method."""
        self._move_cursor(offset=-1)

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Moves to the next item with wrap-around.
        """Move to the next login method."""
        self._move_cursor(offset=1)

    def _move_cursor(self, *, offset: int) -> None:
        # [FUNCTION DOC]
        # HELPER: Moves index by offset modulo the child count (from 0 if unset); None when empty.
        item_count = len(self.children)
        if item_count == 0:
            self.index = None
            return
        current_index = self.index if self.index is not None else 0
        self.index = (current_index + offset) % item_count


class ThemePickerScreen(ModalScreen[TuiThemeName | None]):
    # [CLASS DOC]
    # Modal list of available TUI theme names with the current one marked; enter dismisses with
    # the chosen name, escape with None.
    """Theme picker for the available TUI themes."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("up", "cursor_up", "Up", show=False, priority=True),
        Binding("down", "cursor_down", "Down", show=False, priority=True),
        Binding("enter", "select_cursor", "Select", show=False, priority=True),
    ]

    def __init__(
        self,
        *,
        current_theme: TuiThemeName,
        theme: TuiTheme,
        theme_names: tuple[TuiThemeName, ...],
    ) -> None:
        # [FUNCTION DOC]
        # Stores the current theme name, the active theme and the available theme names.
        super().__init__()
        self.current_theme = current_theme
        self.theme = theme
        self.theme_names = theme_names

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields title, a ListView of labels from _theme_picker_label and help text.
        """Compose the theme picker."""
        with Vertical(id="theme-picker"):
            yield Static("Theme", id="theme-picker-title")
            yield ListView(
                *[
                    ListItem(
                        Label(
                            _theme_picker_label(theme_name, current_theme=self.current_theme),
                            markup=False,
                        )
                    )
                    for theme_name in self.theme_names
                ],
                id="theme-picker-list",
            )
            yield Static("Enter selects - Escape closes", id="theme-picker-help")

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Highlights the current theme (index 0 if absent) and focuses the list.
        """Select the current theme."""
        theme_list = self.query_one("#theme-picker-list", ListView)
        try:
            theme_list.index = self.theme_names.index(self.current_theme)
        except ValueError:
            theme_list.index = 0
        theme_list.focus()

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Routes up/down/enter to the list actions, stopping the events.
        """Route theme picker keys to the list."""
        if event.key == "up":
            event.stop()
            self.action_cursor_up()
        elif event.key == "down":
            event.stop()
            self.action_cursor_down()
        elif event.key == "enter":
            event.stop()
            self.action_select_cursor()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # [FUNCTION DOC]
        # Dismisses with the theme name at the selected index.
        """Dismiss with the selected theme name."""
        self.dismiss(self.theme_names[event.index])

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Moves the list cursor up.
        """Move to the previous theme."""
        self.query_one("#theme-picker-list", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Moves the list cursor down.
        """Move to the next theme."""
        self.query_one("#theme-picker-list", ListView).action_cursor_down()

    def action_select_cursor(self) -> None:
        # [FUNCTION DOC]
        # Selects the highlighted theme.
        """Select the highlighted theme."""
        self.query_one("#theme-picker-list", ListView).action_select_cursor()

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with None.
        """Close without selecting a theme."""
        self.dismiss(None)


class ModelPickerSearchInput(Input):
    # [CLASS DOC]
    # Search Input for the model picker that keeps up/down, tab/ctrl+i (toggle all/scoped mode)
    # and escape local to the picker.
    """Search input that keeps model-picker control keys local to the picker."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        Binding("tab", "toggle_mode", "Mode", show=False, priority=True),
        Binding("ctrl+i", "toggle_mode", "Mode", show=False, priority=True),
        Binding("up", "cursor_up", "Up", show=False, priority=True),
        Binding("down", "cursor_down", "Down", show=False, priority=True),
    ]

    def _picker(self) -> ModelPickerScreen:
        # [FUNCTION DOC]
        # HELPER: Returns the containing screen cast to ModelPickerScreen.
        return cast(ModelPickerScreen, self.screen)

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Routes up/down/tab/ctrl+i/escape to the picker actions, stopping and default-preventing
        # the events.
        """Route picker control keys before the input edits its text."""
        if event.key == "up":
            event.stop()
            event.prevent_default()
            self.action_cursor_up()
        elif event.key == "down":
            event.stop()
            event.prevent_default()
            self.action_cursor_down()
        elif event.key in {"tab", "ctrl+i"}:
            event.stop()
            event.prevent_default()
            self.action_toggle_mode()
        elif event.key == "escape":
            event.stop()
            event.prevent_default()
            self.action_cancel()

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the picker's cursor-up action.
        """Move the model picker selection up."""
        self._picker().action_cursor_up()

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the picker's cursor-down action.
        """Move the model picker selection down."""
        self._picker().action_cursor_down()

    def action_toggle_mode(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the picker's mode toggle.
        """Toggle between all and scoped picker modes."""
        self._picker().action_toggle_mode()

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Forwards to the picker's cancel action.
        """Close the model picker."""
        self._picker().action_cancel()


class ModelPickerScreen(ModalScreen[ModelChoice | None]):
    # [CLASS DOC]
    # Modal model picker (/model and scoped-model editing). Shows all available models plus any
    # scoped models (scoped models not currently available are marked unavailable), with an
    # "all" vs "scoped" mode toggled by tab, a search filter, and per-row labels showing
    # current/scoped state. Enter dismisses with the chosen ModelChoice; escape with None. In
    # "scoped" picker kind, selecting toggles scoping via a callback instead. Supports a
    # background refresh worker and an on-first-refresh hook set by the app.
    """Model picker for the active TUI provider."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("tab", "toggle_mode", "Mode", show=False, priority=True),
        Binding("ctrl+i", "toggle_mode", "Mode", show=False, priority=True),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "accept_model", "Select", show=False),
    ]

    def __init__(
        self,
        choices: Sequence[ModelChoice],
        *,
        scoped_choices: Sequence[ModelChoice],
        current_model: str,
        provider_name: str,
        theme: TuiTheme,
        on_toggle_scoped: Callable[[ModelChoice], Sequence[ModelChoice]] | None = None,
        picker_kind: Literal["model", "scoped"] = "model",
        initial_thinking_level: str | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Deduplicates available and scoped choices preserving order, computes unavailable scoped
        # choices (scoped minus available), sets choices to available followed by extra scoped
        # ones, all visible; stores current model, provider name, theme, toggle callback, picker
        # kind, initial thinking level; mode "all"; empty search; no refresh hook or worker.
        super().__init__()
        self.on_first_refresh: Callable[[], None] | None = None
        self.refresh_worker: Worker[None] | None = None
        self.initial_thinking_level = initial_thinking_level
        available = tuple(dict.fromkeys(choices))
        self.scoped_choices = tuple(dict.fromkeys(scoped_choices))
        self.unavailable_choices = frozenset(self.scoped_choices) - frozenset(available)
        self.choices = tuple(dict.fromkeys((*available, *self.scoped_choices)))
        self.visible_choices = self.choices
        self.current_model = current_model
        self.provider_name = provider_name
        self.theme = theme
        self.on_toggle_scoped = on_toggle_scoped
        self.picker_kind = picker_kind
        self.mode: Literal["all", "scoped"] = "all"
        self.search_value = ""

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields title ("Model: <provider>" or "Scoped models"), an empty tabs Static, the search
        # input, an OptionList with a label per choice, and an empty help Static.
        """Compose the model picker."""
        with Vertical(id="model-picker"):
            title = (
                f"Model: {self.provider_name}" if self.picker_kind == "model" else "Scoped models"
            )
            yield Static(title, id="model-picker-title")
            yield Static("", id="model-picker-tabs")
            yield ModelPickerSearchInput(placeholder="Search models", id="model-picker-search")
            yield OptionList(
                *(self._label_for_choice(choice) for choice in self.choices),
                id="model-picker-list",
                markup=False,
                compact=True,
            )
            yield Static("", id="model-picker-help")

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Focuses search, refreshes selection/help without rebuilding rows, and schedules the
        # on-first-refresh hook after the next refresh if set.
        """Focus the search field."""
        search = self.query_one("#model-picker-search", Input)
        search.focus()
        # compose() already mounted the cached rows; only set selection/help.
        self._refresh_model_list(rebuild_rows=False)
        if self.on_first_refresh is not None:
            self.call_after_refresh(self.on_first_refresh)

    def on_input_changed(self, event: Input.Changed) -> None:
        # [FUNCTION DOC]
        # On search changes (if the value actually changed): stops the event, stores the query and
        # refreshes the list.
        """Filter model choices as the search value changes."""
        if event.input.id != "model-picker-search":
            return
        event.stop()
        if self.search_value == event.value:
            return
        self.search_value = event.value
        self._refresh_model_list()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # [FUNCTION DOC]
        # On search submit: stops the event and selects the highlighted choice.
        """Select the highlighted model from the search field."""
        if event.input.id != "model-picker-search":
            return
        event.stop()
        self._select_visible_choice()

    def _reset_model_list_index(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Highlights the current provider/model choice in the visible list, or the first
        # row; None when nothing is visible.
        """Move selection to the current model or first visible row."""
        model_list = self.query_one("#model-picker-list", OptionList)
        if not self.visible_choices:
            model_list.highlighted = None
            return
        try:
            model_list.highlighted = self.visible_choices.index(
                ModelChoice(provider_name=self.provider_name, model=self.current_model)
            )
        except ValueError:
            model_list.highlighted = 0

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Routes up/down/enter/tab/ctrl+i to cursor, accept and mode-toggle actions, stopping the
        # events.
        """Route model picker keys to the list."""
        if event.key == "up":
            event.stop()
            self.action_cursor_up()
        elif event.key == "down":
            event.stop()
            self.action_cursor_down()
        elif event.key == "enter":
            event.stop()
            self.action_accept_model()
        elif event.key in {"tab", "ctrl+i"}:
            event.stop()
            self.action_toggle_mode()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        # [FUNCTION DOC]
        # Stops the event and selects the highlighted visible choice.
        """Handle the selected row."""
        event.stop()
        self._select_visible_choice()

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Moves the list cursor up.
        """Move to the previous model."""
        self.query_one("#model-picker-list", OptionList).action_cursor_up()

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Moves the list cursor down.
        """Move to the next model."""
        self.query_one("#model-picker-list", OptionList).action_cursor_down()

    def action_accept_model(self) -> None:
        # [FUNCTION DOC]
        # Selects the highlighted visible choice.
        """Select the highlighted model."""
        self._select_visible_choice()

    def action_toggle_mode(self) -> None:
        # [FUNCTION DOC]
        # Switches mode between "all" and "scoped" and refreshes the list.
        """Toggle between all models and scoped models."""
        self.mode = "scoped" if self.mode == "all" else "all"
        self._refresh_model_list()

    def action_toggle_scoped(self) -> None:
        # [FUNCTION DOC]
        # If a toggle callback exists and a row is highlighted, calls the callback with that choice
        # and replaces scoped choices with its (deduplicated) result, then refreshes the list.
        """Add or remove the highlighted model from scoped models."""
        if self.on_toggle_scoped is None or not self.visible_choices:
            return
        model_list = self.query_one("#model-picker-list", OptionList)
        index = model_list.highlighted
        if index is None:
            return
        choice = self.visible_choices[index]
        self.scoped_choices = tuple(dict.fromkeys(self.on_toggle_scoped(choice)))
        self._refresh_model_list()

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Cancels any background refresh and dismisses with None.
        """Close without selecting a model."""
        self.cancel_refresh()
        self.dismiss(None)

    def cancel_refresh(self) -> None:
        # [FUNCTION DOC]
        # Cancels the refresh worker if one exists and has not finished. Returns nothing.
        """Stop work owned by this picker when it is no longer visible."""
        if self.refresh_worker is not None and not self.refresh_worker.is_finished:
            self.refresh_worker.cancel()

    def on_unmount(self) -> None:
        # [FUNCTION DOC]
        # Cancels the refresh worker when the screen is removed without being dismissed.
        """Also cancel when a caller removes the picker without dismissing it."""
        self.cancel_refresh()

    def update_choices(
        self,
        choices: Sequence[ModelChoice],
        scoped_choices: Sequence[ModelChoice],
    ) -> None:
        # [FUNCTION DOC]
        # Applies a refreshed model catalog while the picker is open: deduplicates the inputs,
        # recomputes the combined choices and unavailable scoped set, and if anything changed
        # updates state and refreshes the list (no-op otherwise).
        """Publish a refreshed catalog without replacing the open picker."""
        available = tuple(dict.fromkeys(choices))
        scoped = tuple(dict.fromkeys(scoped_choices))
        updated = tuple(dict.fromkeys((*available, *scoped)))
        unavailable = frozenset(scoped) - frozenset(available)
        if (
            scoped == self.scoped_choices
            and updated == self.choices
            and unavailable == self.unavailable_choices
        ):
            return
        self.scoped_choices = scoped
        self.unavailable_choices = unavailable
        self.choices = updated
        self._refresh_model_list()

    def _select_visible_choice(self) -> None:
        # [FUNCTION DOC]
        # HELPER: For the highlighted visible choice: in "scoped" picker kind toggles its scoping;
        # otherwise ignores unavailable choices, or cancels refresh and dismisses with the choice.
        # No-op with no visible choices or no highlight.
        if not self.visible_choices:
            return
        model_list = self.query_one("#model-picker-list", OptionList)
        index = model_list.highlighted
        if index is None:
            return
        choice = self.visible_choices[index]
        if self.picker_kind == "scoped":
            self.action_toggle_scoped()
            return
        if choice in self.unavailable_choices:
            return
        self.cancel_refresh()
        self.dismiss(choice)

    def _label_for_choice(self, choice: ModelChoice) -> str:
        # [FUNCTION DOC]
        # HELPER: Builds the row label with _model_picker_label, passing current model/provider and
        # whether the choice is scoped or unavailable.
        return _model_picker_label(
            choice,
            current_model=self.current_model,
            current_provider=self.provider_name,
            scoped=choice in self.scoped_choices,
            unavailable=choice in self.unavailable_choices,
        )

    def _refresh_model_list(self, *, rebuild_rows: bool = True) -> None:
        # [FUNCTION DOC]
        # HELPER: Recomputes visible choices from scoped or all choices (by mode) filtered by the
        # search text with _filter_model_choices; optionally rebuilds the OptionList rows; resets
        # the highlight; updates the tab indicator and the help text (wording depends on picker
        # kind, mode, emptiness and the scoped count).
        base_choices = self.scoped_choices if self.mode == "scoped" else self.choices
        self.visible_choices = _filter_model_choices(base_choices, self.search_value)
        if rebuild_rows:
            model_list = self.query_one("#model-picker-list", OptionList)
            model_list.clear_options()
            model_list.add_options(
                self._label_for_choice(choice) for choice in self.visible_choices
            )
        self._reset_model_list_index()
        scope_count = len(self.scoped_choices)
        tabs = self.query_one("#model-picker-tabs", Static)
        if self.picker_kind == "scoped":
            if self.mode == "all":
                tabs.update("Tabs: ● All models  ○ Scoped models")
                help_text = (
                    "all models: no matching models - Tab switches to scoped models"
                    if not self.visible_choices
                    else (
                        "All models - Enter toggles scoped model - Tab switches tabs - "
                        f"{scope_count} scoped - active model is unchanged"
                    )
                )
            else:
                tabs.update("Tabs: ○ All models  ● Scoped models")
                help_text = (
                    "scoped models: no scoped models - Tab switches to all models"
                    if not self.visible_choices
                    else "Scoped models - Enter removes scoped model - Tab switches tabs"
                )
        elif self.mode == "all":
            tabs.update("Tabs: ● All models  ○ Scoped models")
            help_text = (
                "all models: no matching models - Tab switches to scoped models"
                if not self.visible_choices
                else (
                    "All models - Enter selects active model - Tab switches tabs - "
                    f"{scope_count} scoped"
                )
            )
        else:
            tabs.update("Tabs: ○ All models  ● Scoped models")
            help_text = (
                "scoped models: no matching models - Tab switches to all models"
                if not self.visible_choices
                else "Scoped models - Enter selects active model - Tab switches tabs"
            )
        self.query_one("#model-picker-help", Static).update(help_text)


class CustomProviderLoginScreen(ModalScreen[CustomProviderLoginResult | _LoginFlowAction | None]):
    # [CLASS DOC]
    # Modal form for adding a custom OpenAI-compatible provider. Enter in a field moves to the
    # next field (in _INPUT_ORDER); enter in the last field validates and dismisses with a
    # CustomProviderLoginResult. Escape dismisses with BACK; ctrl+d with None. Validation
    # errors are shown in the help line and focus the offending field.
    """Prompt for adding an OpenAI-compatible custom provider."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "back", "Back"),
        Binding("ctrl+d", "close", "Close", priority=True),
    ]

    _INPUT_ORDER: ClassVar[tuple[str, ...]] = (
        "custom-provider-name",
        "custom-provider-display-name",
        "custom-provider-base-url",
        "custom-provider-api-key-env",
        "custom-provider-models",
        "custom-provider-default-model",
        "custom-provider-api-key",
    )

    def __init__(self, *, theme: TuiTheme) -> None:
        # [FUNCTION DOC]
        # Stores the theme.
        super().__init__()
        self.theme = theme

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields the title, help line, seven Inputs (provider name, display name, base URL, API key
        # environment variable, comma-separated models, default model, password API key) and footer.
        """Compose the custom provider prompt."""
        with Vertical(id="login-screen"):
            yield Static("Add custom provider", id="login-title")
            yield Static(
                "Short provider name is used in commands/config.",
                id="custom-provider-help",
            )
            yield Input(placeholder="Provider name/id, e.g. nebius", id="custom-provider-name")
            yield Input(
                placeholder="Display name shown in UI, e.g. Nebius AI Studio",
                id="custom-provider-display-name",
            )
            yield Input(
                placeholder="OpenAI-compatible base URL, e.g. https://api.studio.nebius.ai/v1",
                id="custom-provider-base-url",
            )
            yield Input(
                placeholder="API key environment variable fallback, e.g. NEBIUS_API_KEY",
                id="custom-provider-api-key-env",
            )
            yield Input(
                placeholder="Model ids, comma-separated, e.g. model-a, model-b",
                id="custom-provider-models",
            )
            yield Input(
                placeholder="Default model id, must be listed above",
                id="custom-provider-default-model",
            )
            yield Input(
                placeholder="Paste API key to save for this provider",
                password=True,
                id="custom-provider-api-key",
            )
            yield Static(
                "Enter advances/saves - Escape goes back - Ctrl+D closes",
                id="login-footer",
            )

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Focuses the provider-name field.
        """Focus the first provider-detail field."""
        self.query_one("#custom-provider-name", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # [FUNCTION DOC]
        # For known fields: stops the event; advances focus unless it is the last field; on the last
        # field collects and validates the result and dismisses with it if valid.
        """Advance through fields, then dismiss with provider details."""
        input_id = event.input.id
        if input_id not in self._INPUT_ORDER:
            return
        event.stop()
        if input_id != self._INPUT_ORDER[-1]:
            self._focus_next(input_id)
            return
        result = self._collect_result()
        if result is not None:
            self.dismiss(result)

    def _focus_next(self, input_id: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Focuses the Input that follows the given field id in _INPUT_ORDER.
        index = self._INPUT_ORDER.index(input_id)
        self.query_one(f"#{self._INPUT_ORDER[index + 1]}", Input).focus()

    def _collect_result(self) -> CustomProviderLoginResult | None:
        # [FUNCTION DOC]
        # HELPER: Validates and gathers the form. Required (non-blank after stripping): provider
        # name, base URL, API key env var, models, default model, API key. Models are split on
        # commas, stripped, blanks removed and deduplicated; at least one required; the default
        # model must be in the list. Display name defaults to the provider name. Returns the result,
        # or None after showing an error and focusing the field.
        provider_name = self._field("custom-provider-name", "Provider name")
        if provider_name is None:
            return None
        base_url = self._field("custom-provider-base-url", "Base URL")
        if base_url is None:
            return None
        api_key_env = self._field("custom-provider-api-key-env", "API key environment variable")
        if api_key_env is None:
            return None
        models_text = self._field("custom-provider-models", "Model ids")
        if models_text is None:
            return None
        models = tuple(
            dict.fromkeys(item.strip() for item in models_text.split(",") if item.strip())
        )
        if not models:
            self.query_one("#custom-provider-help", Static).update(
                "At least one model id is required."
            )
            self.query_one("#custom-provider-models", Input).focus()
            return None
        default_model = self._field("custom-provider-default-model", "Default model")
        if default_model is None:
            return None
        if default_model not in models:
            self.query_one("#custom-provider-help", Static).update(
                "Default model must be included in the model list."
            )
            self.query_one("#custom-provider-default-model", Input).focus()
            return None
        api_key = self._field("custom-provider-api-key", "API key")
        if api_key is None:
            return None
        display_name = self.query_one("#custom-provider-display-name", Input).value.strip()
        return CustomProviderLoginResult(
            provider_name=provider_name,
            display_name=display_name or provider_name,
            base_url=base_url,
            api_key_env=api_key_env,
            models=models,
            default_model=default_model,
            api_key=api_key,
        )

    def _field(self, input_id: str, label: str) -> str | None:
        # [FUNCTION DOC]
        # HELPER: Returns the stripped value of a field, or shows "<label> is required.", focuses the
        # field and returns None when blank.
        value = self.query_one(f"#{input_id}", Input).value.strip()
        if value:
            return value
        self.query_one("#custom-provider-help", Static).update(f"{label} is required.")
        self.query_one(f"#{input_id}", Input).focus()
        return None

    def action_back(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with BACK (return to the login method picker).
        """Return to the login method picker."""
        self.dismiss(_LoginFlowAction.BACK)

    def action_close(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with None (closes the whole login flow).
        """Close the entire login flow."""
        self.dismiss(None)


class LoginScreen(ModalScreen[str | _LoginFlowAction | None]):
    # [CLASS DOC]
    # Modal password prompt for a built-in provider's API key. Enter dismisses with the
    # stripped key (None if blank); escape dismisses with BACK; ctrl+d with None.
    """Password prompt for saving a provider API key."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "back", "Back"),
        Binding("ctrl+d", "close", "Close", priority=True),
    ]

    def __init__(self, provider: ProviderCatalogEntry, *, theme: TuiTheme) -> None:
        # [FUNCTION DOC]
        # Stores the provider catalog entry and theme.
        super().__init__()
        self.provider = provider
        self.theme = theme

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields title "Login: <display name>", help, a password Input and footer.
        """Compose the provider login prompt."""
        with Vertical(id="login-screen"):
            yield Static(f"Login: {self.provider.display_name}", id="login-title")
            yield Static("Paste this provider's API key.", id="login-help")
            yield Input(placeholder="Paste API key", password=True, id="login-api-key")
            yield Static("Enter saves - Escape goes back - Ctrl+D closes", id="login-footer")

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Focuses the API key field.
        """Focus the API key field."""
        self.query_one("#login-api-key", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # [FUNCTION DOC]
        # For the key field: stops the event and dismisses with the stripped value or None.
        """Dismiss with the submitted API key."""
        if event.input.id != "login-api-key":
            return
        event.stop()
        self.dismiss(event.value.strip() or None)

    def action_back(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with BACK.
        """Return to the login method picker without saving."""
        self.dismiss(_LoginFlowAction.BACK)

    def action_close(self) -> None:
        # [FUNCTION DOC]
        # Dismisses with None.
        """Close the entire login flow."""
        self.dismiss(None)


class OAuthLoginScreen(ModalScreen[OAuthCredential | _LoginFlowAction | None]):
    # [CLASS DOC]
    # Modal that runs an OAuth (subscription) login for a provider. On mount it starts the
    # provider's login coroutine (from the OAuth registry, or an injected login callable) in an
    # exclusive worker, supplying callbacks that display the authorization URL (as an OSC 8
    # hyperlink, also copied to the clipboard), device codes, progress messages and prompts,
    # and that read manual codes / redirect URLs from an Input. On success dismisses with the
    # OAuthCredential; failures are shown in the help line and the screen stays open. Escape
    # dismisses with BACK, ctrl+d with None (both cancel any pending manual-code wait).
    """OAuth login flow for providers backed by subscription auth."""

    BINDINGS: ClassVar[list[BindingEntry]] = [
        Binding("escape", "back", "Back"),
        Binding("ctrl+d", "close", "Close", priority=True),
    ]

    def __init__(
        self,
        provider: ProviderCatalogEntry,
        *,
        theme: TuiTheme,
        login: Callable[[OAuthLoginCallbacks], Awaitable[OAuthCredential]] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Stores provider, theme, optional injected login callable, and initializes manual-code
        # future/value to None and the allow-empty flag to False.
        super().__init__()
        self.provider = provider
        self.theme = theme
        self._login = login
        self._manual_code_future: asyncio.Future[str] | None = None
        self._manual_code_value: str | None = None
        self._prompt_allows_empty = False

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields title, help, an empty URL Static, the manual code Input and footer.
        """Compose the OAuth login prompt."""
        with Vertical(id="login-screen"):
            yield Static(f"Login: {self.provider.display_name}", id="login-title")
            yield Static("Follow the provider instructions to complete login.", id="login-help")
            yield Static("", id="login-oauth-url")
            yield Input(
                placeholder="Paste redirect URL or authorization code",
                id="login-oauth-code",
            )
            yield Static("Enter submits - Escape goes back - Ctrl+D closes", id="login-footer")

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Focuses the code field and starts _run_login as an exclusive worker.
        """Focus the manual-code field and start OAuth."""
        self.query_one("#login-oauth-code", Input).focus()
        self.run_worker(self._run_login(), exclusive=True)

    async def _run_login(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Resolves the login callable (injected, else registry provider's login); raises
        # RuntimeError internally if none. Awaits it with OAuthLoginCallbacks wired to this screen.
        # On any exception shows "OAuth failed: <error>" and returns; on success dismisses with the
        # credential.
        try:
            oauth_provider = get_oauth_provider(self.provider.name)
            login = self._login or (oauth_provider.login if oauth_provider is not None else None)
            if login is None:
                raise RuntimeError(f"No OAuth implementation for {self.provider.name}")
            credential = await login(
                OAuthLoginCallbacks(
                    on_auth=self._show_auth,
                    on_device_code=self._show_device_code,
                    on_prompt=self._prompt_for_code,
                    on_select=self._select_option,
                    on_progress=self._show_progress,
                    on_manual_code_input=self._manual_code_input,
                )
            )
        except Exception as exc:  # noqa: BLE001 - surface OAuth failures in the TUI
            self.query_one("#login-help", Static).update(f"OAuth failed: {exc}")
            return
        self.dismiss(credential)

    def _show_auth(self, info: OAuthAuthInfo) -> None:
        # [FUNCTION DOC]
        # HELPER: on_auth callback: shows the URL as a link, copies it to the clipboard (errors
        # suppressed), notifies that it was copied, and shows any instructions in the help line.
        self._show_url(info.url)
        # Copy only the browser-flow URL. It is hundreds of characters long and
        # wraps across the dialog, so hand-selecting it is what corrupts it;
        # taking over the clipboard is worth it there and nowhere else.
        with suppress(Exception):
            self.app.copy_to_clipboard(info.url)
        self.notify("Authorization URL copied to clipboard.")
        if info.instructions:
            self.query_one("#login-help", Static).update(info.instructions)

    def _show_url(self, url: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Shows a URL in the URL Static as a single rich Text with a hyperlink style so
        # clicking any wrapped line opens the intact URL.
        """Display a URL as one clickable unit.

        Authorization URLs are far wider than the dialog, so they render across
        several wrapped lines. Selecting those lines by hand tends to corrupt
        the URL — a query parameter split across a wrap picks up the line break
        or trailing padding and the provider rejects the request. An OSC 8
        hyperlink keeps a click on any wrapped line opening the intact URL.
        """
        self.query_one("#login-oauth-url", Static).update(Text(url, style=Style(link=url)))

    def _show_device_code(self, info: OAuthDeviceCodeInfo) -> None:
        # [FUNCTION DOC]
        # HELPER: on_device_code callback: shows the verification URI and "Open the URL and enter
        # code: <code>". No clipboard copy.
        # No clipboard copy here: the verification URI is short and clickable,
        # and the thing the user carries to the browser is the code below it.
        self._show_url(info.verification_uri)
        self.query_one("#login-help", Static).update(
            f"Open the URL and enter code: {info.user_code}"
        )

    def _show_progress(self, message: str) -> None:
        # [FUNCTION DOC]
        # HELPER: on_progress callback: shows the message in the help line.
        self.query_one("#login-help", Static).update(message)

    async def _prompt_for_code(self, prompt: OAuthPrompt) -> str:
        # [FUNCTION DOC]
        # HELPER: on_prompt callback: shows the prompt message, temporarily allows empty submission
        # if the prompt allows it, and awaits manual input. Returns the entered text.
        self.query_one("#login-help", Static).update(prompt.message)
        self._prompt_allows_empty = prompt.allow_empty
        try:
            return await self._manual_code_input()
        finally:
            self._prompt_allows_empty = False

    async def _select_option(self, prompt: OAuthSelectPrompt) -> str | None:
        # [FUNCTION DOC]
        # HELPER: on_select callback: shows the prompt message and automatically returns the id of
        # the first option (None if no options) without asking the user.
        # KNOWN ISSUE: the user is never actually offered a choice; the first option always wins.
        self.query_one("#login-help", Static).update(prompt.message)
        return prompt.options[0].id if prompt.options else None

    async def _manual_code_input(self) -> str:
        # [FUNCTION DOC]
        # HELPER: Returns a previously submitted manual value if one exists; otherwise creates a
        # future and awaits the user's submission (clearing the future afterwards). Raises
        # CancelledError if the screen cancels the wait.
        # KNOWN ISSUE: the first submitted value is cached forever, so a later prompt in the same
        # flow immediately receives the earlier value instead of waiting for new input.
        if self._manual_code_value is not None:
            return self._manual_code_value
        loop = asyncio.get_running_loop()
        self._manual_code_future = loop.create_future()
        try:
            return await self._manual_code_future
        finally:
            self._manual_code_future = None

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # [FUNCTION DOC]
        # For the code field: stops the event; ignores blank input unless the current prompt allows
        # empty; stores the value and resolves a pending manual-code future with it.
        """Resolve the manual OAuth code fallback."""
        if event.input.id != "login-oauth-code":
            return
        event.stop()
        value = event.value.strip()
        if not value and not self._prompt_allows_empty:
            return
        self._manual_code_value = value
        if self._manual_code_future is not None and not self._manual_code_future.done():
            self._manual_code_future.set_result(value)

    def action_back(self) -> None:
        # [FUNCTION DOC]
        # Cancels any pending manual-code wait and dismisses with BACK.
        """Return to the login method picker without saving credentials."""
        self._cancel_manual_code_input()
        self.dismiss(_LoginFlowAction.BACK)

    def action_close(self) -> None:
        # [FUNCTION DOC]
        # Cancels any pending manual-code wait and dismisses with None.
        """Close the entire login flow without saving credentials."""
        self._cancel_manual_code_input()
        self.dismiss(None)

    def _cancel_manual_code_input(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Cancels the manual-code future if pending.
        if self._manual_code_future is not None and not self._manual_code_future.done():
            self._manual_code_future.cancel()


#: Keys an extension key interceptor is never consulted for. These flow
#: straight to normal dispatch so a buggy interceptor (one that returns True
#: too broadly) cannot swallow the session's hard interrupt/exit reflexes and
#: brick the TUI. Deliberately minimal — only the always-available escape
#: hatches: ``ctrl+d`` (the ``quit`` action, exits the app) and ``ctrl+c``
#: (Tau binds it to ``clear_prompt``, but it is the terminal-standard
#: SIGINT/interrupt reflex users hit to bail). NOT reserved: escape/enter/
#: arrows/tab/left/right — those are load-bearing for the tau-subagents
#: extension and must stay interceptable. This is Tau's counterpart to Pi's
#: ``RESERVED_KEYBINDINGS_FOR_EXTENSION_CONFLICTS`` (runner.ts:69), applied
#: here to the pre-dispatch interceptor rather than a registerShortcut API.
RESERVED_EXTENSION_INTERCEPTOR_KEYS: frozenset[str] = frozenset({"ctrl+c", "ctrl+d"})


class TauTuiApp(App[None]):
    # [CLASS DOC]
    # The interactive Textual application for a CodingSession (Tau's counterpart to Pi's
    # interactive mode). Responsibilities:
    # - Layout: a workspace with a SessionSidebar (left, right or hidden, responsive to size)
    #   and a main pane containing the TranscriptView, extension slots (main, above prompt,
    #   below prompt), a queued-messages line, the prompt row (PromptInput), compact session
    #   info and an autocomplete area. CSS uses Tau theme variables.
    # - Input: prompt submission (steer vs follow-up while running), slash commands via
    #   session.handle_command, terminal commands ("!cmd"), autocomplete for commands, files,
    #   skills, sessions, etc., keybindings built from TuiKeybindings.
    # - Running: prompts, compaction and terminal commands run in Textual workers; agent and
    #   coding-session events are fed through TuiEventAdapter into TuiState and re-rendered;
    #   activity indicator animations, terminal title and turn notifications.
    # - Flows: login (API key, OAuth, custom provider), model/scoped model pickers, session
    #   picker/resume/archive, tree branching, themes, skills, prompt templates, tools reference.
    # - Extension UI bridge: dialogs, slot widgets, sidebar sections, main views and key
    #   interceptors with serialized mount/unmount and failure quarantine.
    # State is held on the instance (session, state, settings, workers, extension tracking).
    """Interactive Textual frontend for a ``CodingSession``."""

    TITLE = "Tau"
    CSS = """
    Screen {
        layout: vertical;
        background: $tau-screen-background;
        color: $tau-screen-text;
    }

    Toast {
        background: $tau-chrome-background;
        color: $tau-chrome-text;
    }

    Toast .toast--title {
        color: $tau-accent;
    }

    #workspace {
        height: 1fr;
    }

    #sidebar {
        width: 40;
        min-width: 36;
        height: 1fr;
        padding: 1 1 1 2;
        background: $tau-prompt-background;
        border: none;
    }

    #sidebar-scroll {
        height: 1fr;
        scrollbar-size-vertical: 1;
    }

    #sidebar-content {
        height: auto;
    }

    #sidebar .sidebar-separator {
        height: auto;
    }

    #sidebar .sidebar-resource-section {
        width: 1fr;
        height: auto;
        padding: 0;
        background: transparent;
        border: none;
    }

    #sidebar .sidebar-resource-section:focus-within {
        background-tint: transparent;
    }

    #sidebar .sidebar-resource-section CollapsibleTitle {
        width: 1fr;
        padding: 0 0 0 1;
        color: $tau-prompt-text;
        text-style: none;
        background: transparent;
    }

    #sidebar .sidebar-resource-section CollapsibleTitle:hover,
    #sidebar .sidebar-resource-section CollapsibleTitle:focus {
        color: $tau-prompt-text;
        text-style: none;
        background: transparent;
    }

    #sidebar .sidebar-resource-section Contents {
        padding: 1 0 0 1;
    }

    #sidebar .sidebar-section-title {
        height: 1;
        padding: 0 0 0 1;
        color: $tau-prompt-text;
        text-style: bold;
    }

    #sidebar .sidebar-file-list {
        width: 1fr;
        height: auto;
    }

    #sidebar .sidebar-resource-origin,
    #sidebar .sidebar-file-empty,
    #sidebar .sidebar-file-overflow {
        width: 1fr;
        height: auto;
        color: $tau-muted-text;
    }

    #sidebar .sidebar-file-item {
        width: 1fr;
        height: auto;
        color: $tau-muted-text;
        background: transparent;
    }

    #sidebar .sidebar-file-item:hover,
    #sidebar .sidebar-file-item:focus {
        color: $tau-highlight-text;
        background: $tau-highlight-background;
        text-style: underline;
    }

    #sidebar-extension-sections,
    #sidebar .extension-sidebar-section,
    #sidebar .extension-sidebar-body {
        width: 1fr;
        height: auto;
    }

    #sidebar .extension-sidebar-title {
        height: auto;
        padding: 0 0 0 1;
    }

    #sidebar .extension-sidebar-body {
        padding: 1 0 0 1;
    }

    #sidebar-brand {
        height: auto;
        color: $tau-prompt-text;
    }

    TauTuiApp.-hide-sidebar #sidebar {
        display: none;
    }

    TauTuiApp.-hide-sidebar #main-pane {
        padding-left: 1;
    }

    TauTuiApp.-sidebar-right #sidebar {
        dock: right;
    }

    #main-pane {
        width: 1fr;
        padding: 1 1 0 1;
    }

    #transcript {
        height: 1fr;
        border: none;
        background: $tau-transcript-background;
        padding: 0 0 0 2;
        overflow-x: auto;
        scrollbar-size-vertical: 0;
        scrollbar-size-horizontal: 1;
    }

    /* Component seam: generic extension mount points. */
    #main-slot {
        display: none;
        height: 1fr;
        border: none;
        background: $tau-transcript-background;
        padding: 0 0 0 2;
        overflow-x: auto;
        scrollbar-size-vertical: 0;
        scrollbar-size-horizontal: 1;
    }

    #sidebar-file-editor {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
    }

    #sidebar-file-editor-title {
        height: 1;
        color: $tau-accent;
        text-style: bold;
    }

    #sidebar-file-editor-path,
    #sidebar-file-editor-help,
    #sidebar-file-editor-status {
        height: 1;
        color: $tau-muted-text;
    }

    #sidebar-file-editor-path {
        margin-bottom: 1;
    }

    #sidebar-file-editor-input {
        height: 1fr;
        background: $tau-prompt-background;
        color: $tau-prompt-text;
        border: tall $tau-prompt-border;
    }

    #above-prompt-slot {
        height: auto;
        max-height: 8;
        margin: 0 1 0 1;
        padding: 0;
        background: $tau-screen-background;
    }

    #below-prompt-slot {
        height: auto;
        max-height: 8;
        margin: 0 1 0 1;
        padding: 0;
        background: $tau-screen-background;
    }

    #queued-messages {
        height: auto;
        max-height: 8;
        margin: 0 1 1 1;
        padding: 0 1;
        background: $tau-screen-background;
        color: $tau-muted-text;
    }

    #prompt-row {
        height: auto;
        margin: 0 1 1 1;
    }

    #prompt-prefix {
        width: 2;
        height: 3;
        padding: 0 0 0 0;
        margin: 0;
        content-align: center middle;
        color: $tau-accent;
        text-style: bold;
    }

    #prompt {
        width: 1fr;
        height: auto;
        background: $tau-prompt-background;
        color: $tau-prompt-text;
        border: none;
        border-left: tall transparent;
        margin: 0;
        padding: 1 1;
        max-height: 8;
    }

    #prompt:focus {
        border-left: tall $tau-prompt-border;
    }

    #prompt.-shell-mode {
        border-left: tall $tau-tool-running;
    }

    #compact-session-info {
        height: auto;
        max-height: 3;
        margin: 0 1 1 1;
        padding: 0 1;
        color: $tau-muted-text;
    }

    #autocomplete {
        height: auto;
        max-height: 18;
        margin: 0 1 1 1;
        padding: 0 1;
        background: $tau-autocomplete-background;
        color: $tau-screen-text;
        border: tall $tau-border;
        overflow-y: auto;
    }

    SessionPickerScreen,
    PromptTemplatePickerScreen,
    PromptTemplateEditorScreen,
    SkillPickerScreen,
    TreePickerScreen,
    ToolsReferenceScreen,
    CommandOutputScreen {
        align: center middle;
    }

    #session-picker,
    #prompt-template-picker,
    #prompt-template-editor,
    #skill-picker,
    #tree-picker,
    #tools-reference {
        width: 76;
        max-width: 90%;
        height: auto;
        max-height: 70%;
        padding: 1 2;
        background: $tau-chrome-background;
        border: tall $tau-border;
    }

    #session-picker-title,
    #prompt-template-picker-title,
    #prompt-template-editor-title,
    #skill-picker-title,
    #tree-picker-title,
    #tools-reference-title {
        height: 1;
        color: $tau-chrome-text;
        text-style: bold;
        margin-bottom: 1;
    }

    #session-picker-search,
    #prompt-template-picker-search,
    #skill-picker-search,
    #tools-reference-search {
        height: 3;
        margin-bottom: 1;
        background: $tau-prompt-background;
        color: $tau-prompt-text;
        border: tall $tau-prompt-border;
    }

    #tools-reference-header {
        height: 1;
        color: $tau-muted-text;
        text-style: bold;
    }

    #prompt-template-editor {
        height: 80%;
    }

    #prompt-template-editor-path {
        height: 1;
        margin-bottom: 1;
        color: $tau-muted-text;
    }

    #prompt-template-editor-input {
        height: 1fr;
        background: $tau-prompt-background;
        color: $tau-prompt-text;
        border: tall $tau-prompt-border;
    }

    #session-picker-list,
    #prompt-template-picker-list,
    #skill-picker-list,
    #tree-picker-list,
    #tools-reference-list {
        height: auto;
        max-height: 16;
        background: $tau-transcript-background;
        border: tall $tau-border;
    }

    ListView,
    OptionList {
        scrollbar-background: $tau-transcript-background;
        scrollbar-color: $tau-border;
        scrollbar-color-hover: $tau-highlight-background;
        scrollbar-background-hover: $tau-transcript-background;
        scrollbar-color-active: $tau-accent;
        scrollbar-background-active: $tau-transcript-background;
        scrollbar-size-vertical: 2;
    }

    ListView > ListItem.-highlight {
        background: $tau-highlight-background;
        color: $tau-highlight-text;
    }

    ListView > ListItem.-highlight Label {
        background: $tau-highlight-background;
        color: $tau-highlight-text;
    }

    OptionList > .option-list--option-highlighted {
        background: $tau-highlight-background;
        color: $tau-highlight-text;
    }

    #skill-picker-list .skill-picker-row {
        height: 1;
    }

    #skill-picker-list .skill-picker-name {
        width: 35%;
        text-style: bold;
    }

    #skill-picker-list .skill-picker-description {
        width: 65%;
        color: $tau-muted-text;
    }

    #skill-picker-list ListItem.-highlight .skill-picker-description {
        color: $tau-highlight-text;
    }

    #session-picker-help,
    #prompt-template-picker-help,
    #prompt-template-editor-help,
    #skill-picker-help,
    #tree-picker-help,
    #tools-reference-help {
        height: 1;
        margin-top: 1;
        color: $tau-muted-text;
    }

    #tree-picker-help {
        height: auto;
    }

    ExtensionSelectScreen,
    ExtensionConfirmScreen,
    ExtensionInputScreen,
    LocalBackendPickerScreen,
    LocalBackendScreen,
    LocalChoiceConfirmScreen,
    LocalConfigureScreen,
    LocalConfirmScreen,
    LocalModelActionScreen,
    LocalSearchResultsScreen,
    ProjectTrustScreen {
        align: center middle;
    }

    ProjectTrustScreen {
        background: $tau-screen-background 60%;
    }

    #extension-select,
    #extension-confirm,
    #extension-input {
        width: 76;
        max-width: 90%;
        height: auto;
        max-height: 70%;
        padding: 1 2;
        background: $tau-chrome-background;
        border: tall $tau-border;
    }

    #extension-select-title,
    #extension-confirm-title,
    #extension-input-title {
        height: auto;
        color: $tau-chrome-text;
        text-style: bold;
        margin-bottom: 1;
    }

    #extension-confirm-message {
        height: auto;
        color: $tau-chrome-text;
        margin-bottom: 1;
    }

    #extension-select-list,
    #extension-confirm-list {
        height: auto;
        max-height: 16;
        background: $tau-transcript-background;
        border: tall $tau-border;
    }

    #extension-input-field {
        background: $tau-transcript-background;
        border: tall $tau-border;
    }

    #extension-select-help,
    #extension-confirm-help,
    #extension-input-help {
        height: 1;
        margin-top: 1;
        color: $tau-muted-text;
    }

    #local-backend-picker,
    #local-backend-screen,
    #local-configure-screen,
    #local-confirm-screen,
    #local-model-action-screen,
    #local-search-results-screen {
        width: 82;
        max-width: 92%;
        height: auto;
        max-height: 82%;
        padding: 1 2;
        background: $tau-chrome-background;
        border: tall $tau-border;
    }

    #local-backend-picker-title,
    #local-backend-title,
    #local-configure-title,
    #local-confirm-title,
    #local-model-action-title,
    #local-search-results-title {
        height: auto;
        color: $tau-chrome-text;
        text-style: bold;
        margin-bottom: 1;
    }

    #local-backend-picker-help,
    #local-backend-help,
    #local-configure-screen Label,
    #local-confirm-message,
    #local-search-results-help {
        color: $tau-muted-text;
    }

    #local-backend-list,
    #local-backend-status,
    #local-backend-progress,
    #local-model-list,
    #local-action-menu,
    #local-confirm-list,
    #local-choice-list,
    #local-search-results-list,
    #local-configure-screen Input,
    #local-configure-screen Select,
    #local-model-action-input {
        background: $tau-transcript-background;
        border: tall $tau-border;
        margin-top: 1;
    }

    #local-backend-list,
    #local-model-list,
    #local-action-menu,
    #local-confirm-list,
    #local-choice-list,
    #local-search-results-list {
        height: auto;
        max-height: 16;
    }

    #local-model-list,
    #local-action-menu {
        max-height: 10;
    }

    #local-model-list:focus,
    #local-action-menu:focus {
        border: tall $tau-accent;
    }

    #local-model-list.local-section-inactive > ListItem.-highlight,
    #local-action-menu.local-section-inactive > ListItem.-highlight,
    #local-model-list.local-section-inactive > ListItem.-highlight Label,
    #local-action-menu.local-section-inactive > ListItem.-highlight Label {
        background: $tau-transcript-background;
        color: $tau-chrome-text;
    }

    #local-model-section-title,
    #local-action-section-title {
        height: 1;
        margin-top: 1;
        color: $tau-chrome-text;
        text-style: bold;
    }

    #local-backend-progress-bar {
        width: 100%;
        margin-top: 1;
    }

    #local-backend-progress-bar Bar {
        width: 1fr;
    }

    #local-backend-progress-bar Bar > .bar--bar,
    #local-backend-progress-bar Bar > .bar--complete,
    #local-backend-progress-bar Bar > .bar--indeterminate {
        color: $tau-accent;
        background: $tau-border;
    }

    #local-backend-picker-footer,
    #local-backend-footer,
    #local-configure-footer,
    #local-confirm-footer,
    #local-model-action-footer,
    #local-search-results-footer {
        height: 1;
        margin-top: 1;
        color: $tau-muted-text;
    }

    #local-backend-progress {
        min-height: 1;
        color: $tau-muted-text;
    }

    #project-trust-dialog {
        width: 76;
        max-width: 92%;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        background: $tau-chrome-background;
        color: $tau-chrome-text;
        border: tall $tau-border;
    }

    #project-trust-title {
        color: $tau-chrome-text;
    }

    #project-trust-path-label,
    #project-trust-summary-label,
    #project-trust-boundary,
    #project-trust-help {
        color: $tau-muted-text;
    }

    #project-trust-list {
        background: $tau-transcript-background;
        color: $tau-screen-text;
        border: tall $tau-border;
    }

    #project-trust-list ListItem.-highlight,
    #project-trust-list ListItem.-highlight Label {
        background: $tau-highlight-background;
        color: $tau-highlight-text;
    }

    #command-output {
        width: 76;
        max-width: 90%;
        height: auto;
        max-height: 70%;
        padding: 1 2;
        background: $tau-chrome-background;
        color: $tau-chrome-text;
        border: tall $tau-border;
    }

    #command-output-title {
        height: 1;
        color: $tau-chrome-text;
        text-style: bold;
        margin-bottom: 1;
    }

    #command-output-scroll {
        height: auto;
        max-height: 18;
        background: $tau-transcript-background;
        border: tall $tau-border;
    }

    #command-output-body {
        color: $tau-screen-text;
        padding: 1;
    }

    #command-output-help {
        height: 1;
        margin-top: 1;
        color: $tau-muted-text;
    }

    LoginMethodPickerScreen,
    LoginProviderPickerScreen,
    ThemePickerScreen,
    ModelPickerScreen {
        align: center middle;
    }

    #login-method-picker,
    #login-provider-picker,
    #theme-picker,
    #model-picker {
        width: 76;
        max-width: 90%;
        height: auto;
        max-height: 70%;
        padding: 1 2;
        background: $tau-chrome-background;
        color: $tau-chrome-text;
        border: tall $tau-border;
    }

    #login-method-title,
    #login-provider-title,
    #theme-picker-title,
    #model-picker-title {
        height: 1;
        color: $tau-chrome-text;
        text-style: bold;
        margin-bottom: 1;
    }

    #model-picker-tabs {
        height: 1;
        color: $tau-muted-text;
        margin-bottom: 1;
    }

    #login-method-list,
    #login-provider-list,
    #theme-picker-list,
    #model-picker-list {
        height: auto;
        max-height: 12;
        background: $tau-transcript-background;
        color: $tau-screen-text;
        border: tall $tau-border;
    }

    #login-method-list ListItem Label,
    #login-provider-list ListItem Label,
    #theme-picker-list ListItem Label,
    #model-picker-list {
        color: $tau-screen-text;
    }

    #login-method-list ListItem.-highlight Label,
    #login-provider-list ListItem.-highlight Label,
    #theme-picker-list ListItem.-highlight Label {
        background: $tau-highlight-background;
        color: $tau-highlight-text;
    }

    #model-picker-list > .option-list--option-highlighted {
        background: $tau-highlight-background;
        color: $tau-highlight-text;
    }

    #login-method-intro {
        height: 1;
        color: $tau-muted-text;
        margin-bottom: 1;
    }

    #login-method-list {
        height: auto;
        max-height: 10;
    }

    #login-provider-search,
    #model-picker-search {
        height: 3;
        margin-bottom: 1;
        background: $tau-prompt-background;
        color: $tau-prompt-text;
        border: tall $tau-prompt-border;
    }

    #login-method-help,
    #login-provider-help,
    #theme-picker-help,
    #model-picker-help {
        height: 1;
        margin-top: 1;
        color: $tau-muted-text;
    }

    CustomProviderLoginScreen,
    LoginScreen,
    OAuthLoginScreen {
        align: center middle;
    }

    #login-screen {
        width: 72;
        max-width: 92%;
        height: auto;
        /* A wrapped authorization URL makes this dialog taller than a short
           terminal. Cap it at the screen and scroll instead of overflowing:
           overflowing centers the excess, which pushes the paste field and
           the footer off the bottom and the title off the top. */
        max-height: 100%;
        overflow-y: auto;
        padding: 1 2;
        background: $tau-chrome-background;
        border: tall $tau-border;
    }

    #login-title {
        height: 1;
        color: $tau-chrome-text;
        text-style: bold;
        margin-bottom: 1;
    }

    #login-help,
    #custom-provider-help {
        height: 1;
        color: $tau-muted-text;
        margin-bottom: 1;
    }

    #login-api-key,
    #login-oauth-code,
    #custom-provider-name,
    #custom-provider-display-name,
    #custom-provider-base-url,
    #custom-provider-api-key-env,
    #custom-provider-models,
    #custom-provider-default-model,
    #custom-provider-api-key {
        background: $tau-prompt-background;
        color: $tau-prompt-text;
        border: tall $tau-prompt-border;
        margin-bottom: 1;
    }

    #login-oauth-url {
        /* Authorization URLs run ~470 chars (Anthropic); clipping them means a
           user copying the URL out of the TUI loses the trailing query
           parameters and the provider rejects the request. Keep every line. */
        min-height: 1;
        height: auto;
        color: $tau-chrome-text;
        margin-bottom: 1;
    }

    #login-footer {
        height: 1;
        color: $tau-muted-text;
    }
    """
    BINDINGS: ClassVar[list[BindingEntry]] = []

    def __init__(
        self,
        session: CodingSession,
        *,
        tui_settings: TuiSettings | None = None,
        startup_message: str | None = None,
        startup_notice: str | None = None,
        startup_update_notice: str | None = None,
        startup_alerts: Sequence[str] = (),
        startup_notices: Sequence[str] = (),
        initial_prompt: str | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Builds the app around a session. Stores settings (default TuiSettings), startup message,
        # startup notices (new list plus the legacy single notice), and initial prompt; registers
        # Tau themes with Textual and applies the resolved theme name (guarded so the theme watcher
        # does not persist it); installs app bindings from the keybindings; creates TuiState with
        # the session's skills and adds status items for the update notice ("update" highlight),
        # alerts ("alert" highlight), notices, and a missing-theme fallback warning; loads existing
        # session messages into the state; creates the TuiEventAdapter; initializes all extension
        # component tracking structures and locks; connects the extension runtime to this app;
        # initializes worker handles, run ids, optimistic user messages, completion state, activity
        # animation state, terminal title and notification controllers, focus tracking, notification
        # dedupe keys and clipboard capability cache; finally syncs the terminal title.
        # Inputs: the CodingSession and optional startup options. Returns nothing.
        # Side effects: mutates Textual theme registry; may read session message history.
        self.tui_settings = tui_settings or TuiSettings()
        self.startup_message = startup_message
        legacy_notices = (startup_notice,) if startup_notice else ()
        self.startup_notices = tuple((*startup_notices, *legacy_notices))
        self.initial_prompt = initial_prompt
        # This override is deliberately separate from durable settings. It is
        # reset with every app instance and never participates in tui.json.
        self._sidebar_visibility_override: bool | None = None
        super().__init__()
        self._register_tau_textual_themes()
        # Assign the resolved theme's name: it is always registered, while the
        # raw settings value may name a custom theme that failed to load. The
        # guard keeps the watcher from persisting the fallback over the user's
        # configured theme.
        self._applying_settings_theme = True
        self.theme = self.tui_settings.resolved_theme.name
        self._applying_settings_theme = False
        self._bindings = BindingsMap(_app_bindings(self.tui_settings.keybindings))
        self.session = session
        self.state = TuiState(skills=session.skills)
        if startup_update_notice is not None:
            self.state.add_item("status", startup_update_notice, highlight="update")
        for alert in startup_alerts:
            self.state.add_item("status", alert, highlight="alert")
        for notice in self.startup_notices:
            self.state.add_item("status", notice)
        if self.tui_settings.theme != self.tui_settings.resolved_theme.name:
            self.state.add_item(
                "status",
                f"Theme '{self.tui_settings.theme}' was not found; "
                f"using {self.tui_settings.resolved_theme.name}.",
            )
        self._prompt_history: tuple[str, ...] = ()
        self._load_session_messages_from_session()
        self.adapter = TuiEventAdapter(self.state)
        # Component seam: host-owned tracking of extension
        # widgets so a reload/rebind can force-clear them and a crash can
        # quarantine them. Must exist before _connect_extension_runtime, which
        # clears them on every bind.
        # `_extension_slot_widgets` holds the *intended* widget per key (the swap
        # target, set synchronously); `_extension_slot_mounted` tracks what is
        # actually mounted. A deferred remove() must fully drain before the next
        # mount of the same-id widget, so slot/main-view swaps run on a serialized
        # async continuation (see `_reconcile_slot`/`_reconcile_main_view`).
        self._extension_slot_widgets: dict[str, Widget] = {}
        self._extension_slot_mounted: dict[str, Widget] = {}
        self._extension_slot_slot_ids: dict[str, str] = {}
        self._extension_slot_locks: dict[str, asyncio.Lock] = {}
        self._extension_key_interceptors: list[KeyInterceptor] = []
        self._extension_sidebar_contributions: dict[tuple[str, str], _SidebarContribution] = {}
        self._extension_sidebar_widgets: dict[tuple[str, str], Widget] = {}
        self._extension_sidebar_mounted: dict[tuple[str, str], Widget] = {}
        self._extension_sidebar_lock = asyncio.Lock()
        self._extension_sidebar_theme: TuiTheme | None = None
        self._extension_main_view: _MainViewHandle | None = None
        self._extension_main_view_mounted: Widget | None = None
        self._extension_main_view_lock = asyncio.Lock()
        self._extension_swap_tasks: set[asyncio.Task[None]] = set()
        self._extension_component_failures_reported: set[str] = set()
        self._connect_extension_runtime(session)
        self._prompt_worker: Worker[None] | None = None
        self._compaction_worker: Worker[None] | None = None
        self._compacting = False
        self._compaction_run_id = 0
        self._prompt_run_id = 0
        self._optimistic_user_messages: list[tuple[int, str]] = []
        self._completion_state = CompletionState()
        self._completion_visible_line_budget: int | None = None
        self._activity_frame = 0
        self._activity_timer: Timer | None = None
        self._last_tool_timer_refresh_at = 0.0
        self._last_activity_indicator_key: tuple[object, ...] | None = None
        self._last_queue_render_key: tuple[object, ...] | None = None
        self._terminal_title = TerminalTitleController()
        self._terminal_notification = TerminalNotificationController(
            self.tui_settings.turn_notification
        )
        self._app_has_focus = True
        self._active_notification_keys: set[tuple[str, str]] = set()
        self._supports_pyperclip: bool | None = None
        self._sync_session_title()

    async def prompt_project_trust(self, request: ProjectTrustRequest) -> TrustChoice | None:
        # [FUNCTION DOC]
        # Pushes a ProjectTrustScreen for the request and awaits the user's choice (must be called
        # from a worker context since it uses push_screen_wait). Returns the TrustChoice or None.
        """Resolve a trust request through the active Textual modal stack."""
        return await self.push_screen_wait(ProjectTrustScreen(request))

    def _sync_session_title(self) -> None:
        # [FUNCTION DOC]
        # Updates the terminal tab title from the session (delegates to _sync_terminal_title).
        """Reflect the active session name in the terminal tab title."""
        self._sync_terminal_title()

    def _is_working(self) -> bool:
        # [FUNCTION DOC]
        # Returns True while an agent turn is running or a compaction is in progress.
        """Return whether the app should show working affordances (agent turn or compaction)."""
        return self.state.running or self._compacting

    def _sync_terminal_title(self) -> None:
        # [FUNCTION DOC]
        # Pushes the session title, working state and current animation frame to the terminal
        # title controller. Side effect: writes a terminal title escape sequence.
        """Reflect the active session name and running state in the terminal tab title."""
        self._terminal_title.update(
            getattr(self.session, "session_title", None),
            running=self._is_working(),
            frame=self._activity_frame,
        )

    def _sync_text_selection_state(self) -> None:
        # [FUNCTION DOC]
        # Sets the class-level ALLOW_SELECT flag to disallow native text selection while running and
        # clears any current selection on the active screen (errors suppressed).
        # Note this mutates the class attribute, so it affects all instances of the app class.
        """Disable native text selection while the transcript is mutating."""
        type(self).ALLOW_SELECT = not self.state.running
        if self.state.running and self.screen_stack:
            with suppress(Exception):
                self.screen.clear_selection()

    def copy_to_clipboard(self, text: str) -> None:
        # [FUNCTION DOC]
        # Copies text to the clipboard: lazily detects whether pyperclip is importable (cached), uses
        # it if so (errors suppressed), and always also calls Textual's OSC 52 clipboard copy.
        """Copy text using pyperclip when available, then Textual's fallback."""
        if self._supports_pyperclip is None:
            try:
                import pyperclip  # type: ignore[import-untyped]
            except ImportError:
                self._supports_pyperclip = False
            else:
                self._supports_pyperclip = True
        if self._supports_pyperclip:
            import pyperclip

            with suppress(Exception):
                pyperclip.copy(text)
        super().copy_to_clipboard(text)

    def _register_tau_textual_themes(self) -> None:
        # [FUNCTION DOC]
        # Clears Textual's registered themes and registers a Textual theme for every available Tau
        # theme name, so Textual's own theme controls stay in sync with /theme.
        """Register Tau themes with Textual's theme system.

        Textual exposes its own theme menu and command palette entries. Registering
        Tau's built-in themes there makes those controls update the same theme as
        `/theme` instead of changing only Textual's chrome.
        """
        self._registered_themes.clear()
        for theme_name in available_tui_theme_names():
            self.register_theme(_textual_theme_for_tau_theme(theme_name))

    def _reload_session_themes(self) -> None:
        # [FUNCTION DOC]
        # Reloads custom themes for the active session: uses session.theme_dirs if present, else
        # computes theme directories from the cwd with project resources enabled only when project
        # trust is unresolved or trusted. Loading errors (OSError/RuntimeError) fail closed (no
        # custom themes) with an error notification. Installs the custom themes, re-registers
        # themes, re-applies the resolved theme (forcing a CSS refresh if the name is unchanged)
        # without persisting, and shows each loading diagnostic with matching severity.
        """Rebind custom themes to the active session's accepted trust snapshot."""
        theme_dirs = getattr(self.session, "theme_dirs", None)
        if theme_dirs is None:
            trust_resolution = getattr(self.session, "project_trust_resolution", None)
            trusted = trust_resolution is None or trust_resolution.trusted
            theme_dirs = TauResourcePaths(
                cwd=self.session.cwd,
                project_resources_enabled=trusted,
            ).themes_dirs
        try:
            custom_themes, diagnostics = load_custom_tui_themes(theme_dirs)
        except (OSError, RuntimeError) as exc:
            # Theme discovery must fail closed after a trust/cwd transition:
            # never retain themes from the previous project snapshot.
            custom_themes = {}
            diagnostics = []
            self._notify(f"Could not reload custom themes: {exc}", severity="error")

        set_custom_tui_themes(custom_themes)
        self._register_tau_textual_themes()
        resolved_theme = self.tui_settings.resolved_theme.name
        self._applying_settings_theme = True
        try:
            if self.theme == resolved_theme:
                # Re-apply CSS when a same-named custom theme changed in place.
                self._watch_theme(resolved_theme)
            else:
                self.theme = resolved_theme
        finally:
            self._applying_settings_theme = False
        for diagnostic in diagnostics:
            severity: Literal["information", "warning", "error"] = (
                "error"
                if diagnostic.severity == "error"
                else "warning"
                if diagnostic.severity == "warning"
                else "information"
            )
            self._notify(diagnostic.format(), severity=severity)

    def _watch_theme(self, theme_name: str) -> None:
        # [FUNCTION DOC]
        # Theme watcher override: after Textual's handling, if the new theme is a Tau theme, the
        # change was not made by settings application, and it differs from the saved theme, updates
        # TUI settings and saves them to disk (tui.json). Side effect: file write.
        """Keep Textual theme changes synchronized with Tau's durable TUI theme."""
        super()._watch_theme(theme_name)
        if theme_name not in available_tui_theme_names():
            return
        if getattr(self, "_applying_settings_theme", False):
            return
        tau_theme: TuiThemeName = theme_name
        if self.tui_settings.theme == tau_theme:
            return
        self._replace_tui_settings(theme=tau_theme)
        save_tui_settings(self.tui_settings)

    def get_theme_variable_defaults(self) -> dict[str, str]:
        # [FUNCTION DOC]
        # Returns Textual's default theme variables merged with Tau CSS variables for the resolved
        # Tau theme.
        """Return Tau-specific CSS variables for the selected TUI theme."""
        variables = super().get_theme_variable_defaults()
        return {**variables, **_theme_css_variables(self.tui_settings.resolved_theme)}

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yields the full widget tree: workspace with sidebar and main pane (transcript, main slot,
        # above-prompt slot, queued-messages Static, prompt row with the tau prefix and PromptInput
        # whose placeholder mentions the newline key, compact session info, autocomplete Static,
        # below-prompt slot).
        """Compose the TUI widgets."""
        with Horizontal(id="workspace"):
            yield SessionSidebar(id="sidebar")
            with Vertical(id="main-pane"):
                yield TranscriptView(
                    id="transcript",
                    min_width=1,
                    wrap=True,
                    highlight=True,
                    markup=False,
                )
                # Component seam: host-managed mount points for
                # extension widgets. Empty until an extension mounts into them.
                yield Container(id="main-slot")
                yield Container(id="above-prompt-slot")
                yield Static("", id="queued-messages")
                with Horizontal(id="prompt-row"):
                    yield Static("τ", id="prompt-prefix")
                    yield PromptInput(
                        placeholder=(
                            "Ask Tau…  Enter submits, "
                            f"{_key_hint(self.tui_settings.keybindings.insert_newline)} "
                            "inserts a newline"
                        ),
                        id="prompt",
                        tui_keybindings=self.tui_settings.keybindings,
                    )
                yield CompactSessionInfo(id="compact-session-info")
                yield Static("", id="autocomplete")
                yield Container(id="below-prompt-slot")

    async def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Startup: sets the prompt's shell-mode style from the theme's tool border style, syncs
        # shell mode, focuses the prompt, applies responsive layout and sidebar position, refreshes
        # the UI, text selection state and completions, shows the startup message as a warning,
        # awaits the session's deferred session_start emission (so extensions can use the UI), and
        # submits the initial prompt (stripped) if provided.
        """Focus the prompt when the app starts."""
        prompt = self.query_one(PromptInput)
        prompt.shell_mode_style = self.tui_settings.resolved_theme.role_styles["tool"].border
        self._sync_prompt_shell_mode(prompt.text)
        prompt.focus()
        self._update_responsive_layout(self.size.width, self.size.height)
        self._apply_sidebar_position()
        self._refresh()
        self._sync_text_selection_state()
        self._refresh_completions()
        if self.startup_message:
            self._notify(self.startup_message, severity="warning")
        # UI is live and the bridge is installed (__init__) — release the
        # deferred session_start so handlers can notify / open dialogs.
        await self.session.emit_pending_session_start()
        if self.initial_prompt and self.initial_prompt.strip():
            await self._submit_prompt(self.initial_prompt.strip())

    async def on_event(self, event: events.Event) -> None:
        # [FUNCTION DOC]
        # Pre-dispatch event hook: for non-forwarded key events not in
        # RESERVED_EXTENSION_INTERCEPTOR_KEYS, when interceptors exist and only the main screen is on
        # the stack, runs extension key interceptors with the current prompt text; if one consumes
        # the key, stops and prevents the event. Otherwise defers to Textual's normal dispatch.
        """Consult extension key interceptors before Textual's dispatch.

        Ports Pi's ``onTerminalInput``: a registered interceptor sees a key at
        the earliest point in key processing — before tau's app-level priority
        bindings (``down``/``up``/``tab``/``alt+enter`` in ``_app_bindings``)
        and before the focused widget. Textual's ``App.on_event`` runs
        ``_check_bindings(key, priority=True)`` ahead of forwarding a key to the
        focused widget, so a focused extension widget would otherwise never
        receive those keys; this pre-dispatch hook is the only place an
        extension can own them.

        Interceptors are consulted only on the main screen (never while a modal
        dialog/picker sits on the screen stack) and only when at least one is
        registered, so the default path is untouched. Interceptors therefore
        see EVERY main-screen key regardless of focus and must self-gate.

        The hard interrupt/exit keys in
        :data:`RESERVED_EXTENSION_INTERCEPTOR_KEYS` are skipped entirely, so
        they always reach normal dispatch even behind a misbehaving interceptor.
        """
        if (
            isinstance(event, events.Key)
            and not event.is_forwarded
            and event.key not in RESERVED_EXTENSION_INTERCEPTOR_KEYS
            and self._extension_key_interceptors
            and len(self.screen_stack) <= 1
            and self._run_extension_key_interceptors(event, self._current_prompt_text())
        ):
            event.stop()
            event.prevent_default()
            return
        await super().on_event(event)

    def on_unmount(self) -> None:
        # [FUNCTION DOC]
        # Teardown: stops the activity timer, restores the terminal title and clears all extension
        # components.
        """Stop activity animations and drop extension widgets on teardown."""
        if self._activity_timer is not None:
            self._activity_timer.stop()
            self._activity_timer = None
        self._terminal_title.restore()
        self._clear_extension_components()

    def on_app_blur(self) -> None:
        # [FUNCTION DOC]
        # Records that the app lost focus (so turn-complete notifications should fire).
        """Remember that terminal attention should be requested when the run settles."""
        self._app_has_focus = False

    def on_app_focus(self) -> None:
        # [FUNCTION DOC]
        # Records that the app has focus (suppressing turn notifications).
        """Suppress turn notifications while the Tau terminal surface is active."""
        self._app_has_focus = True

    def on_paste(self, event: events.Paste) -> None:
        # [FUNCTION DOC]
        # Handles pastes that arrive while no widget is focused (e.g. OS file drops): if something
        # is focused or the current screen has no prompt, does nothing; otherwise stops the event
        # and inserts the text into the prompt through its paste rules.
        """Route pastes that arrive while no widget holds keyboard focus.

        Textual clears widget focus whenever the terminal reports lost focus
        (``CSI ? 1004 h``), and pastes are dropped when nothing is focused. OS
        drag-and-drop from sources that never hand focus back to the terminal --
        notably the macOS Dock -- delivers the dropped paths in exactly that
        state, so the paste bubbles up here instead of reaching the prompt.
        Clipboard pastes always require terminal focus, so this only reroutes
        drops that would otherwise be silently discarded.
        """
        if self.focused is not None:
            return
        try:
            prompt = self.screen.query_one("#prompt", PromptInput)
        except NoMatches:
            # A modal screen owns the input; leave its own handling alone.
            return
        event.stop()
        prompt.insert_pasted_text(event.text)

    def on_resize(self, event: Resize) -> None:
        # [FUNCTION DOC]
        # On terminal resize, resets the cached completion line budget and recomputes responsive
        # layout.
        """Update responsive chrome when the terminal changes size."""
        self._completion_visible_line_budget = None
        self._update_responsive_layout(event.size.width, event.size.height)

    @on(SidebarFileItem.OpenRequested)
    def on_sidebar_file_open_requested(self, event: SidebarFileItem.OpenRequested) -> None:
        # [FUNCTION DOC]
        # Opens a sidebar file in the main-area SidebarFileEditor. If an open editor has unsaved
        # changes, warns, updates its status and refocuses it instead. Read errors (OSError,
        # UnicodeDecodeError) are reported as error notifications. Otherwise opens a main view whose
        # factory builds the editor with the read text and snapshot. Stops the event.
        """Open a sidebar resource file in the main-area editor."""
        event.stop()
        item = event.item
        current = self._extension_main_view
        if (
            current is not None
            and isinstance(current.widget, SidebarFileEditor)
            and current.widget.is_dirty
        ):
            message = "Save or close the current file before opening another."
            self._notify(message, severity="warning")
            current.widget.query_one("#sidebar-file-editor-status", Static).update(message)
            current.widget.query_one("#sidebar-file-editor-input", TextArea).focus()
            return
        try:
            source, snapshot = _read_sidebar_file(item.path)
        except (OSError, UnicodeDecodeError) as exc:
            self._notify(f"Could not read {item.path}: {exc}", severity="error")
            return
        self._open_extension_main_view(
            lambda handle, theme: SidebarFileEditor(
                handle=handle,
                path=item.path,
                label=item.file_label,
                kind=item.kind,
                source=source,
                snapshot=snapshot,
            )
        )

    def on_click(self, event: events.Click) -> None:
        # [FUNCTION DOC]
        # On a primary-button click, refocuses the prompt unless an extension main view is open
        # (which owns the keyboard). Missing prompt is ignored.
        """Return keyboard focus to the prompt after clicks in the main TUI."""
        if event.button != 1:
            return
        if self._extension_main_view is not None:
            # An extension main view (e.g. a subagent conversation viewer) owns
            # the main area and its keyboard; yanking focus back to the prompt
            # would silently reroute every key — esc, toggles, typed text — to
            # the main chat. Clicking the prompt itself still focuses it via
            # Textual's native mouse-down focus.
            return
        with suppress(NoMatches):
            self.screen.query_one("#prompt", PromptInput).focus()

    @on(events.TextSelected)
    async def on_text_selected(self) -> None:
        # [FUNCTION DOC]
        # When text is selected and auto-copy is enabled (globally in settings or on the active
        # screen), copies the selection to the clipboard and notifies.
        """Optionally copy selected text automatically."""
        active_screen = self.screen
        if not (
            self.tui_settings.auto_copy_selection
            or getattr(active_screen, "auto_copy_selection", False)
        ):
            return
        selection = active_screen.get_selected_text()
        if selection:
            self.copy_to_clipboard(selection)
            self._notify("Copied selection to clipboard.")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        # [FUNCTION DOC]
        # On prompt text changes: syncs pending pastes, updates shell-mode styling, rebuilds the
        # completion state from the text and cursor, and refreshes the completion display.
        """Update prompt autocomplete when the prompt text changes."""
        if event.text_area.id != "prompt":
            return
        prompt = self.query_one("#prompt", PromptInput)
        prompt.sync_pending_paste()
        # Read text and cursor from the widget so both come from one snapshot.
        text = prompt.text
        self._sync_prompt_shell_mode(text)
        self._completion_state = self._build_completion_state(text, cursor=prompt.cursor_position)
        self._refresh_completions()

    def on_text_area_selection_changed(self, event: TextArea.SelectionChanged) -> None:
        # [FUNCTION DOC]
        # On prompt selection changes, schedules a check (after the Changed handler) that closes
        # completions when the caret leaves the completed token.
        """Close prompt autocomplete when the caret leaves the completed token."""
        if event.text_area.id != "prompt":
            return
        # Edits post SelectionChanged before Changed; check after Changed has rebuilt.
        self.call_later(self._close_completions_if_caret_left_token)

    def _close_completions_if_caret_left_token(self) -> None:
        # [FUNCTION DOC]
        # HELPER: If completions are showing and the cursor is outside the first item's token range
        # (start exclusive, end inclusive), clears the completion state and refreshes.
        if not self._completion_state.items:
            return
        item = self._completion_state.items[0]
        cursor = self.query_one("#prompt", PromptInput).cursor_position
        if item.start < cursor <= item.end:
            return
        self._completion_state = CompletionState()
        self._refresh_completions()

    async def action_submit_prompt(self) -> None:
        # [FUNCTION DOC]
        # Enter handling: if a non-file completion is selected and accepting it changes the prompt
        # text, accept it and stop; otherwise submit the prompt with "steer" streaming behavior.
        """Accept a changing non-file completion, or submit the current prompt text."""
        selected = self._completion_state.selected
        if selected is not None and selected.kind is not CompletionKind.FILE_REFERENCE:
            prompt = self.query_one("#prompt", PromptInput)
            text_before_completion = prompt.text
            self.action_accept_completion()
            if prompt.text != text_before_completion:
                return
        await self._submit_prompt_from_editor(streaming_behavior="steer")

    async def action_submit_follow_up(self) -> None:
        # [FUNCTION DOC]
        # Submits the prompt with "follow_up" streaming behavior (queued after the current run).
        """Submit the current prompt as a queued follow-up while running."""
        await self._submit_prompt_from_editor(streaming_behavior="follow_up")

    async def _submit_prompt_from_editor(
        self,
        *,
        streaming_behavior: Literal["steer", "follow_up"],
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Core submission path for the prompt editor.
        # 1. Takes the editor text with large-paste placeholders expanded; if blank, clears the
        #    editor, pending pastes and completions and returns.
        # 2. While a compaction is active: warns (a second /compact is refused; other text is put
        #    back in the editor with the cursor at the end) and returns.
        # 3. Clears the editor, pending pastes and completions.
        # 4. Terminal commands (parsed by parse_terminal_command, e.g. "!cmd" / "!!cmd") run in an
        #    exclusive "terminal-command" worker, optionally adding output to context.
        # 5. Otherwise asks session.handle_command; for handled slash commands applies every
        #    requested effect in order: model / scoped-model pickers (return early), clear state,
        #    reload (session.reload, theme reload, summary message; ValueError becomes a message),
        #    new session, compaction (refused while compaction or agent/queue activity is ongoing,
        #    restoring the text; otherwise started in a worker), export (errors notified), resume by
        #    id, resume picker, prompt-template picker, tree picker (refused while running), login
        #    picker, custom provider login, local backend picker, sidebar toggle, login for a given
        #    provider/method, logout picker/logout, model switch in a worker, tools reference, skills
        #    picker, theme picker, session rename (ValueError notified and returns), thinking level,
        #    theme; then refreshes skills, shows the command message as a notification, transcript
        #    entry or modal (based on helper predicates), refreshes the UI and exits the app if
        #    requested.
        # 6. Non-command text: remembered in prompt history, then queued (with the given streaming
        #    behavior) if the agent is running, else submitted as a new prompt.
        # Inputs: streaming_behavior "steer" or "follow_up". Returns nothing. Errors are surfaced as
        # notifications rather than raised. Side effects: many UI and session mutations.
        prompt = self.query_one("#prompt", PromptInput)
        raw_text = prompt.text_for_submission()

        text = raw_text.strip()
        if not text:
            prompt.text = ""
            prompt._clear_pending_paste()
            self._completion_state = CompletionState()
            self._refresh_completions()
            return

        if self._is_compaction_active():
            if text.startswith("/compact"):
                self._notify("A compaction is already running.", severity="warning")
            else:
                prompt.text = raw_text
                prompt.move_cursor(_text_end_location(raw_text))
                self._notify(
                    "Compaction is still running. You can keep editing, but wait to submit.",
                    severity="warning",
                )
            return

        prompt.text = ""
        prompt._clear_pending_paste()
        self._completion_state = CompletionState()
        self._refresh_completions()

        terminal_command = parse_terminal_command(text)
        if terminal_command is not None:
            self.run_worker(
                self._run_terminal_command(
                    terminal_command.command,
                    add_to_context=terminal_command.add_to_context,
                ),
                group="terminal-command",
                exclusive=True,
            )
            return

        command = self.session.handle_command(text)
        if command.handled:
            if command.model_picker_requested:
                self._open_model_picker()
                return
            if command.scoped_models_picker_requested:
                self._open_scoped_models_picker()
                return
            if command.clear_requested:
                self.state.clear()
            if command.reload_requested:
                try:
                    summary = await self.session.reload()
                except ValueError as exc:
                    command = replace(command, message=f"Could not reload: {exc}")
                else:
                    self._reload_session_themes()
                    command = replace(command, message=format_reload_summary(summary))
            if command.new_session_requested:
                await self._new_session()
            if command.compact_summary is not None:
                if self._is_compaction_active():
                    self._notify("A compaction is already running.", severity="warning")
                elif self._is_agent_or_queue_active():
                    prompt.text = raw_text
                    prompt.move_cursor(_text_end_location(raw_text))
                    self._notify(
                        "Wait for the current agent turn and queued messages to finish "
                        "before compacting.",
                        severity="warning",
                    )
                    return
                else:
                    self._compaction_worker = self.run_worker(
                        self._run_compaction(command.compact_summary),
                        exclusive=False,
                    )
            if command.export_requested:
                try:
                    exported_path = await self.session.export(
                        command.export_destination,
                        format=command.export_format,
                    )
                    self._append_command_message(
                        text,
                        f"Exported session to {exported_path}",
                    )
                except Exception as exc:  # noqa: BLE001 - surface command failures in the TUI
                    self._notify(f"Could not export session: {exc}", severity="error")
            if command.resume_session_id is not None:
                await self._resume_session(command.resume_session_id)
            if command.resume_picker_requested:
                self.action_open_session_picker()
            if command.prompts_picker_requested:
                self._open_prompt_template_picker()
            if command.tree_picker_requested:
                if self._is_agent_or_queue_active():
                    prompt.text = raw_text
                    prompt.move_cursor(_text_end_location(raw_text))
                    self._notify(TREE_RUNNING_MESSAGE, severity="warning")
                    return
                await self._open_tree_picker()
            if command.login_picker_requested:
                self._open_login_picker()
            if command.custom_provider_login_requested:
                self._open_custom_provider_login()
            if command.local_requested:
                self._open_local_backend_picker()
            if command.sidebar_toggle_requested:
                self._toggle_sidebar_visibility()
            if command.login_provider is not None:
                self._open_login(command.login_provider, method=command.login_method)
            if command.logout_picker_requested:
                self._open_logout_picker()
            if command.logout_provider is not None:
                self._logout(command.logout_provider)
            if command.model_selection_model is not None:
                self.run_worker(
                    self._switch_model(
                        ModelChoice(
                            provider_name=command.model_selection_provider
                            or self.session.provider_name,
                            model=command.model_selection_model,
                        )
                    ),
                    exclusive=False,
                )
            if command.tools_picker_requested:
                self._open_tools_reference()
            if command.skills_picker_requested:
                self._open_skills_picker()
            if command.theme_picker_requested:
                self._open_theme_picker()
            if command.session_name is not None:
                try:
                    await self.session.set_session_name(command.session_name)
                except ValueError as exc:
                    self._notify(f"Could not rename session: {exc}", severity="error")
                    self._refresh()
                    return
                self._sync_session_title()
            if command.thinking_level is not None:
                await self._set_thinking_level(command.thinking_level)
            if command.theme is not None:
                self._set_tui_theme(command.theme)
            self.state.set_skills(self.session.skills)
            if command.message:
                if _command_message_uses_notification(text, command.message):
                    self._notify(command.message)
                elif _command_message_uses_transcript(text):
                    self._append_command_message(
                        text,
                        command.message,
                        system_prompt_inspection=command.system_prompt_inspection,
                    )
                else:
                    self._show_command_message(text, command.message)
            self._refresh()
            if command.exit_requested:
                self.exit()
            return

        if self.state.running:
            self._remember_prompt(text)
            await self._queue_prompt(text, streaming_behavior=streaming_behavior)
            return

        self._remember_prompt(text)
        await self._submit_prompt(text)

    def _remember_prompt(self, text: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Appends non-blank submitted text to the in-memory prompt history tuple.
        """Remember a submitted user prompt for lightweight input recall."""
        if not text.strip():
            return
        self._prompt_history = (*self._prompt_history, text)

    def _load_session_messages_from_session(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Loads the session's messages into the TUI state and rebuilds prompt history from
        # non-blank user message texts.
        """Load visible session messages and reseed prompt history from them."""
        self.state.load_messages(self.session.messages)
        self._prompt_history = tuple(
            message.text
            for message in self.session.messages
            if isinstance(message, UserMessage) and message.text.strip()
        )

    def _is_compaction_active(self) -> bool:
        # [FUNCTION DOC]
        # Returns True if the compaction worker exists and is neither finished nor cancelled, or the
        # compacting flag is set.
        """Return whether a manual compaction worker is still running."""
        worker = self._compaction_worker
        if worker is not None and not worker.is_finished and not worker.is_cancelled:
            return True
        return self._compacting

    def _is_agent_or_queue_active(self) -> bool:
        # [FUNCTION DOC]
        # Syncs queue state and returns True if the TUI state is running, the session reports
        # is_running, the prompt worker is active, or there are queued messages.
        """Return whether compaction would race an active or queued agent turn."""
        self._sync_queue_state()
        worker = self._prompt_worker
        is_worker_active = worker is not None and not worker.is_finished and not worker.is_cancelled
        is_session_running = bool(getattr(self.session, "is_running", False))
        return (
            self.state.running
            or is_session_running
            or is_worker_active
            or self.state.queued_message_count > 0
        )

    async def _run_compaction(self, summary: str) -> None:
        # [FUNCTION DOC]
        # HELPER (worker body): Runs a manual compaction with the given summary instructions.
        # Increments the compaction run id, sets compacting, clears the transcript and shows
        # "Compacting session…", then awaits session.compact. Cancellation returns silently; other
        # errors show "Error: ..." notifications. In all cases, if this run is still the latest,
        # clears compacting/worker and refreshes chrome. On success reloads skills and messages,
        # notifies the returned message, refreshes, and fires a terminal notification if the app is
        # unfocused.
        """Run manual compaction without disabling prompt editing."""
        self._compaction_run_id += 1
        run_id = self._compaction_run_id
        self._compacting = True
        try:
            self.state.clear()
            self.state.add_item("status", "Compacting session…")
            self._refresh()
            compact_message = await self.session.compact(summary)
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001 - surface command failures in the TUI
            self._notify(f"Error: {exc}", severity="error")
            return
        finally:
            # A cancelled run can tear down after a newer compaction started, so only
            # clear working state this run still owns.
            if self._compaction_run_id == run_id:
                self._compacting = False
                self._compaction_worker = None
                self._refresh_chrome_if_mounted()
        self.state.clear()
        self.state.set_skills(self.session.skills)
        self._load_session_messages_from_session()
        self._notify(compact_message)
        self._refresh()
        if not self._app_has_focus:
            self._terminal_notification.notify_turn_finished()

    async def _submit_prompt(
        self,
        text: str,
        *,
        source: Literal["interactive", "extension"] = "interactive",
        custom_type: str | None = None,
        details: dict[str, JSONValue] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Starts an agent run for text. Increments the prompt run id; for plain (non-custom)
        # prompts that should be optimistically rendered, records (run id, text) and renders the
        # user message immediately; then starts _run_prompt in an exclusive worker and stores it.
        # Inputs: text, source ("interactive" or "extension"), optional custom type and details.
        """Add a prompt to the transcript and start the agent worker."""
        self._prompt_run_id += 1
        run_id = self._prompt_run_id
        # Custom messages are never rendered optimistically: the optimistic
        # dedupe matches on exact content equality with the post-expansion
        # event (see _consume_optimistic_user_event), and a mismatch would
        # double-render. They render once, from the confirmed user event,
        # which carries their custom_type/details.
        if custom_type is None and _should_optimistically_render_prompt(text):
            self._optimistic_user_messages.append((run_id, text))
            await self._append_optimistic_user_message(text)
        self._prompt_worker = self.run_worker(
            self._run_prompt(text, run_id, source=source, custom_type=custom_type, details=details),
            exclusive=True,
        )

    async def _append_optimistic_user_message(
        self,
        text: str,
        *,
        custom_type: str | None = None,
        details: dict[str, JSONValue] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Adds a user (or custom) message to the state and appends only the new transcript
        # items incrementally (scrolling to the end), then refreshes chrome. Falls back to a full
        # refresh if there is no screen yet or no transcript widget.
        """Render a submitted user message immediately without rebuilding the transcript."""
        start_index = len(self.state.items)
        self.state.add_user_message(text, custom_type=custom_type, details=details)
        self._follow_transcript_output()
        if not self.screen_stack:
            self._refresh()
            return
        theme = self.tui_settings.resolved_theme
        try:
            transcript = self.query_one("#transcript", TranscriptView)
        except NoMatches:
            self._refresh()
            return
        for item in self.state.items[start_index:]:
            await transcript.append_item(
                item,
                theme=theme,
                show_tool_results=self.state.show_tool_results,
                scroll_end=True,
                custom_markup=self.state.resolve_custom_markup(
                    item, expanded=self.state.show_tool_results
                ),
            )
        self._refresh_chrome(theme=theme)

    def _consume_optimistic_user_event(self, event: CodingSessionEvent, *, run_id: int) -> bool:
        # [FUNCTION DOC]
        # HELPER: If the event is a MessageEndEvent carrying a UserMessage whose content exactly
        # equals a pending optimistic text for this run, removes that pending entry and returns True
        # (the message is already rendered). Otherwise False.
        # Note: content is compared directly, so block-list content never matches and falls through
        # to the transformed-message reconciliation.
        """Return whether a user event confirms an already-rendered optimistic message."""
        if not isinstance(event, MessageEndEvent) or not isinstance(event.message, UserMessage):
            return False
        for index, (pending_run_id, pending_text) in enumerate(self._optimistic_user_messages):
            if pending_run_id == run_id and pending_text == event.message.content:
                del self._optimistic_user_messages[index]
                return True
        return False

    def _replace_transformed_optimistic_user_message(
        self, event: CodingSessionEvent, *, run_id: int
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: For a confirmed user MessageEndEvent that did not exactly match: takes this run's
        # first pending optimistic text, removes it, rewrites the most recent user transcript item
        # with that text to the confirmed message text (e.g. after an extension input hook
        # transformed it), refreshes and resyncs the title, and returns True. False if not
        # applicable.
        """Reconcile a transformed prompt with its optimistic render.

        An extension `input` hook may transform the submitted text inside
        session.prompt, so the confirmed UserMessage no longer matches the
        optimistically rendered original (the exact-equality path above).
        Rewrite the optimistic item in place and redraw, instead of letting
        the confirmed event append a second user item alongside the stale
        original. Runs after _consume_optimistic_user_event, so it only fires
        when this run's pending optimistic text mismatches — the run's own
        prompt confirmation is the first user event of the run, so a queued
        steering/follow-up user message can never be mistaken for it.
        """
        if not isinstance(event, MessageEndEvent) or not isinstance(event.message, UserMessage):
            return False
        for index, (pending_run_id, pending_text) in enumerate(self._optimistic_user_messages):
            if pending_run_id != run_id:
                continue
            del self._optimistic_user_messages[index]
            for item in reversed(self.state.items):
                if item.role == "user" and item.text == pending_text:
                    item.text = event.message.text
                    break
            self._refresh()
            self._sync_session_title()
            return True
        return False

    def _clear_optimistic_user_messages(self, *, run_id: int) -> None:
        # [FUNCTION DOC]
        # HELPER: Removes all pending optimistic messages belonging to the run id.
        """Drop unconfirmed optimistic messages once their run is no longer active."""
        self._optimistic_user_messages = [
            pending for pending in self._optimistic_user_messages if pending[0] != run_id
        ]

    async def _append_confirmed_user_message(self, message: AgentMessage) -> None:
        # [FUNCTION DOC]
        # HELPER: Renders a confirmed (non-optimistic) message incrementally: user messages by text;
        # custom messages only if marked for display (with custom type and dict details); anything
        # else triggers a full refresh.
        """Render a non-optimistic user/custom event incrementally when possible."""
        if isinstance(message, UserMessage):
            await self._append_optimistic_user_message(message.text)
            return
        if isinstance(message, CustomMessage):
            if message.display:
                await self._append_optimistic_user_message(
                    message.text,
                    custom_type=message.custom_type,
                    details=message.details if isinstance(message.details, dict) else None,
                )
            return
        self._refresh()

    def _connect_extension_runtime(self, session: CodingSession) -> None:
        # [FUNCTION DOC]
        # HELPER: If the session has an extension runtime: clears existing extension components,
        # installs a _TuiExtensionUiBridge, registers the turn-requested callback, and wires the
        # runtime's custom message, tool call and tool result renderers into the TUI state.
        """Give the extension runtime a UI bridge and an idle-run entry point."""
        runtime = getattr(session, "extension_runtime", None)
        if runtime is None:
            return
        # Force-clear any extension widgets before installing the new bridge.
        # This runs once, at construction; later teardowns (/reload, resume,
        # new) come through the installed bridge's clear_components(), driven
        # by the runtime, so extension widgets and key interceptors never
        # survive a world they were mounted in.
        self._clear_extension_components()
        runtime.set_ui_bridge(_TuiExtensionUiBridge(self))
        runtime.set_turn_requested_callback(self._on_extension_turn_requested)
        # Let the transcript render custom messages via registered renderers.
        self.state.custom_renderer = runtime.render_custom_message
        # Let tool calls render through their tool's render_call, if any.
        self.state.tool_call_renderer = runtime.render_tool_call
        # And tool results through their tool's render_result, if any.
        self.state.tool_result_renderer = runtime.render_tool_result

    def _on_extension_turn_requested(
        self,
        content: str,
        custom_type: str | None = None,
        details: dict[str, JSONValue] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Callback from the extension runtime asking to send a message as a turn; schedules
        # _deliver_extension_message on the app loop.
        """Deliver an extension message through the serialized prompt path."""
        self.call_later(self._deliver_extension_message, content, custom_type, details)

    async def _deliver_extension_message(
        self,
        content: str,
        custom_type: str | None = None,
        details: dict[str, JSONValue] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: If a run is in progress (session running or prompt worker set), queues the content
        # as a follow-up message on the session (if supported); otherwise submits it as an
        # "extension"-sourced prompt with optional custom type and details.
        if self.session.is_running or self._prompt_worker is not None:
            # A run started while the delivery was in flight; drain with it.
            queue_follow_up = getattr(self.session, "queue_follow_up_message", None)
            if callable(queue_follow_up):
                queue_follow_up(content, custom_type=custom_type, details=details)
            return
        await self._submit_prompt(
            content, source="extension", custom_type=custom_type, details=details
        )

    # -- component seam ------------------------------------------------------

    def _current_prompt_text(self) -> str:
        # [FUNCTION DOC]
        # Returns the prompt editor text, or an empty string when the prompt widget does not exist.
        """Return the prompt-editor text, or "" before the prompt exists."""
        try:
            return self.query_one("#prompt", PromptInput).text
        except NoMatches:
            return ""

    def _register_extension_key_interceptor(self, handler: KeyInterceptor) -> Callable[[], None]:
        # [FUNCTION DOC]
        # HELPER: Appends the interceptor to the list and returns an unsubscribe function.
        """Register a pre-dispatch key interceptor; return an unsubscribe fn."""
        self._extension_key_interceptors.append(handler)

        def unsubscribe() -> None:
            # [FUNCTION DOC]
            # HELPER: Removes the interceptor from the list; missing entries are ignored.
            with suppress(ValueError):
                self._extension_key_interceptors.remove(handler)

        return unsubscribe

    def _run_extension_key_interceptors(self, event: Key, text: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: Calls each interceptor (on a snapshot of the list) with the key event and prompt
        # text; returns True as soon as one returns truthy. Exceptions are recorded once per
        # interceptor (with notification) and treated as not consumed. Returns False otherwise.
        """Consult interceptors; return True if one consumed the key.

        Each call is guarded: a raising interceptor is diagnosed once and
        treated as "not consumed", so a broken interceptor degrades to normal
        typing rather than a dead prompt.
        """
        for interceptor in tuple(self._extension_key_interceptors):
            try:
                if interceptor(event, text):
                    return True
            except Exception as exc:  # noqa: BLE001 - isolation boundary
                # Notify like the other failure classes so a broken interceptor
                # is not silently invisible, and dedup per-interceptor so a
                # second faulty handler still gets diagnosed.
                self._record_extension_component_failure(
                    f"key_interceptor:{id(interceptor)}", exc, notify=True
                )
        return False

    def _schedule_extension_swap(self, coro: Coroutine[object, object, None]) -> None:
        # [FUNCTION DOC]
        # HELPER: Wraps a reconcile coroutine in an asyncio task kept in a strong-reference set until
        # done. If there is no running loop, closes the coroutine instead.
        """Run a slot/main-view reconcile coroutine on the app loop.

        The task is retained until it finishes so it cannot be garbage-collected
        mid-flight (asyncio only holds a weak reference). If there is no running
        loop (only possible outside a live TUI), the coroutine is closed rather
        than left un-awaited.
        """
        try:
            task = asyncio.ensure_future(coro)
        except RuntimeError:  # no running loop — not a live TUI
            coro.close()
            return
        self._extension_swap_tasks.add(task)
        task.add_done_callback(self._extension_swap_tasks.discard)

    @staticmethod
    def _string_slot_widget(lines: Sequence[str]) -> Static:
        # [FUNCTION DOC]
        # HELPER (static): Joins lines with newlines and returns a Static rendering them as Rich
        # markup, falling back to literal text when markup is invalid.
        """Build a slot ``Static`` from display lines (Rich markup, safe fallback).

        Joins ``lines`` with newlines and parses them as Rich markup; if the
        markup is malformed the literal text is shown instead, mirroring the
        custom-message renderer's guard so a bad string never crashes the TUI.
        """
        content = "\n".join(lines)
        return Static(_custom_markup_to_text(content))

    def _set_extension_slot_widget(
        self,
        key: str,
        content: SlotWidgetContent | None,
        placement: Placement,
    ) -> None:
        # [FUNCTION DOC]
        # Sets or clears an extension slot widget. Content may be None (unmount), a factory taking
        # the theme, a string (one line) or a sequence of lines (converted to a factory building a
        # markup Static). Builds the new widget immediately (factory errors are recorded with a
        # notification and abort), records the intended widget and slot container ("above-prompt-slot"
        # for above_prompt, else "below-prompt-slot") or removes them, and schedules a serialized
        # reconcile for the key. Returns nothing.
        """Mount an extension widget into a prompt-adjacent slot, or unmount it.

        ``content`` is a ``factory(theme)`` callable, a list of display lines
        the host renders into a ``Static``, or ``None`` to unmount. The string
        form is normalized into a factory here so the reconcile/quarantine/
        replace machinery below is untouched.

        The intended widget is recorded synchronously so mid-swap reads (clear,
        quarantine, refresh) see what *should* occupy the slot; the actual
        mount/unmount runs on a serialized continuation so a deferred remove()
        of a same-id widget fully drains before the replacement mounts (else the
        DOM briefly holds two widgets with one id -> ``DuplicateIds``).
        """
        factory: SlotWidgetFactory | None
        if content is None:
            factory = None
        elif callable(content):
            # Check callable() first: a Sequence[str] test must never swallow a
            # factory (and a factory is not a Sequence).
            factory = content
        else:
            # A plain list of display lines: build the widget host-side so the
            # extension needs no Textual import. A bare str is treated as one
            # line (never split into characters).
            lines = [content] if isinstance(content, str) else list(content)
            factory = lambda _theme: self._string_slot_widget(lines)  # noqa: E731
        new_widget: Widget | None = None
        if factory is not None:
            try:
                new_widget = factory(self.tui_settings.resolved_theme)
            except Exception as exc:  # noqa: BLE001 - isolation boundary
                self._record_extension_component_failure(f"slot:{key}", exc, notify=True)
                return
        slot_id = "above-prompt-slot" if placement == "above_prompt" else "below-prompt-slot"
        if new_widget is None:
            self._extension_slot_widgets.pop(key, None)
            self._extension_slot_slot_ids.pop(key, None)
        else:
            self._extension_slot_widgets[key] = new_widget
            self._extension_slot_slot_ids[key] = slot_id
        self._schedule_extension_swap(self._reconcile_slot(key))

    async def _reconcile_slot(self, key: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Under a per-key lock, removes a mounted widget that is no longer the intended one
        # (errors suppressed), then mounts the current intended widget into its slot container if
        # not already mounted. Mount failures drop the intent (if unchanged) and record a failure
        # with notification. Last writer wins for bursts of updates.
        """Make the mounted slot widget match the intended target (serialized).

        Reads the *live* target each time (not a snapshot), so a burst of set
        calls collapses to "last writer wins": the first continuation removes the
        stale mount, later ones find the target already satisfied and no-op.
        """
        lock = self._extension_slot_locks.setdefault(key, asyncio.Lock())
        async with lock:
            target = self._extension_slot_widgets.get(key)
            mounted = self._extension_slot_mounted.get(key)
            if mounted is not None and mounted is not target:
                with suppress(Exception):
                    await mounted.remove()
                if self._extension_slot_mounted.get(key) is mounted:
                    self._extension_slot_mounted.pop(key, None)
            # Re-read after the await; the target may have changed meanwhile.
            target = self._extension_slot_widgets.get(key)
            if target is None or self._extension_slot_mounted.get(key) is target:
                return
            slot_id = self._extension_slot_slot_ids.get(key, "below-prompt-slot")
            try:
                self.query_one(f"#{slot_id}", Container).mount(target)
            except Exception as exc:  # noqa: BLE001 - isolation boundary
                if self._extension_slot_widgets.get(key) is target:
                    self._extension_slot_widgets.pop(key, None)
                    self._extension_slot_slot_ids.pop(key, None)
                self._record_extension_component_failure(f"slot:{key}", exc, notify=True)
                return
            self._extension_slot_mounted[key] = target

    def _build_extension_sidebar_widget(
        self,
        owner: tuple[str, str],
        contribution: _SidebarContribution,
        *,
        theme: TuiTheme,
    ) -> Widget | None:
        # [FUNCTION DOC]
        # HELPER: Builds the host frame for one extension sidebar section: calls a factory content
        # with the theme, or renders string/line content as a markup Static; requires the body to be
        # a Textual Widget (else TypeError internally). Returns a Vertical with a separator, a bold
        # title in the theme's prompt text color and a container holding the body. On any exception
        # records a per-section failure (notifying, attributed to the extension) and returns None.
        """Build one host-framed sidebar section, isolating its body factory."""
        content = contribution.content
        try:
            if callable(content):
                body = content(theme)
            else:
                lines = [content] if isinstance(content, str) else list(content)
                body = self._string_slot_widget(lines)
            if not isinstance(body, Widget):
                raise TypeError("sidebar factory must return a Textual Widget")
            header = Text(contribution.title, style=f"bold {theme.prompt_text}")
            return Vertical(
                Static(_sidebar_separator(theme=theme), classes="sidebar-separator"),
                Static(header, classes="extension-sidebar-title"),
                Container(body, classes="extension-sidebar-body"),
                classes="extension-sidebar-section",
            )
        except Exception as exc:  # noqa: BLE001 - isolation boundary
            extension_name, key = owner
            self._record_extension_component_failure(
                f"sidebar:{extension_name}:{key}",
                exc,
                notify=True,
                extension_name=extension_name,
            )
            return None

    def _set_extension_sidebar_section(
        self,
        extension_name: str,
        key: str,
        *,
        title: str,
        content: SidebarContent,
    ) -> None:
        # [FUNCTION DOC]
        # Adds or updates an extension sidebar section (no-op when the sidebar is "off").
        # Normalizes content (factory kept; a string becomes a one-line tuple; sequences become
        # tuples). Skips if the contribution and theme are unchanged. If both old and new contents
        # are static lines and the mounted widget is current under the same theme, updates the title
        # and body in place (failures recorded and abort). Otherwise builds a new framed widget,
        # records contribution, widget and theme, and schedules a serialized sidebar reconcile.
        """Add or update a sidebar contribution while preserving key order."""
        if self.tui_settings.sidebar_position == "off":
            return
        owner = (extension_name, key)
        normalized_content: SidebarContent
        if callable(content):
            normalized_content = content
        elif isinstance(content, str):
            normalized_content = (content,)
        else:
            normalized_content = tuple(content)
        contribution = _SidebarContribution(title=title, content=normalized_content)
        theme = self.tui_settings.resolved_theme
        previous = self._extension_sidebar_contributions.get(owner)
        if previous == contribution and self._extension_sidebar_theme == theme:
            return
        mounted = self._extension_sidebar_mounted.get(owner)
        target = self._extension_sidebar_widgets.get(owner)
        if (
            previous is not None
            and not callable(previous.content)
            and not callable(normalized_content)
            and mounted is not None
            and mounted is target
            and self._extension_sidebar_theme == theme
        ):
            try:
                header = Text(title, style=f"bold {theme.prompt_text}")
                mounted.query_one(".extension-sidebar-title", Static).update(header)
                body = mounted.query_one(".extension-sidebar-body", Container).query_one(Static)
                body.update(_custom_markup_to_text("\n".join(normalized_content)))
            except Exception as exc:  # noqa: BLE001 - isolation boundary
                self._record_extension_component_failure(
                    f"sidebar:{extension_name}:{key}",
                    exc,
                    notify=True,
                    extension_name=extension_name,
                )
                return
            self._extension_sidebar_contributions[owner] = contribution
            return
        widget = self._build_extension_sidebar_widget(owner, contribution, theme=theme)
        if widget is None:
            return
        self._extension_sidebar_contributions[owner] = contribution
        self._extension_sidebar_widgets[owner] = widget
        self._extension_sidebar_theme = theme
        self._schedule_extension_swap(self._reconcile_sidebar())

    def _remove_extension_sidebar_section(self, extension_name: str, key: str) -> None:
        # [FUNCTION DOC]
        # Forgets one extension sidebar contribution and its intended widget (if known) and schedules
        # a sidebar reconcile to unmount it.
        """Forget and unmount one extension-owned sidebar contribution."""
        owner = (extension_name, key)
        if owner not in self._extension_sidebar_contributions:
            return
        self._extension_sidebar_contributions.pop(owner, None)
        self._extension_sidebar_widgets.pop(owner, None)
        self._schedule_extension_swap(self._reconcile_sidebar())

    def _rebuild_extension_sidebar_sections(self, *, theme: TuiTheme) -> None:
        # [FUNCTION DOC]
        # Rebuilds every extension sidebar section's widget for a new theme (no-op if the sidebar is
        # off), records the theme and schedules a reconcile if any widget was rebuilt.
        """Recreate sidebar factories for a changed live theme."""
        if self.tui_settings.sidebar_position == "off":
            return
        changed = False
        for owner, contribution in tuple(self._extension_sidebar_contributions.items()):
            widget = self._build_extension_sidebar_widget(owner, contribution, theme=theme)
            if widget is not None:
                self._extension_sidebar_widgets[owner] = widget
                changed = True
        self._extension_sidebar_theme = theme
        if changed:
            self._schedule_extension_swap(self._reconcile_sidebar())

    async def _reconcile_sidebar(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Under the sidebar lock, makes the mounted extension sidebar sections match the
        # intended ones in registration order: returns early if identical or the sections container
        # is missing; removes stale mounted roots (errors suppressed); then mounts each missing
        # target before the next already-mounted later section. Mount failures drop the contribution
        # and record a failure. Finally rebuilds the mounted map to only current targets.
        """Mount sidebar sections in registration order after removals drain."""
        async with self._extension_sidebar_lock:
            target_items = tuple(self._extension_sidebar_widgets.items())
            mounted_items = tuple(self._extension_sidebar_mounted.items())
            if mounted_items == target_items:
                return
            try:
                slot = self.query_one("#sidebar-extension-sections", Container)
            except NoMatches:
                return
            # Remove only stale roots. An unchanged Textual widget cannot be
            # removed and mounted again: removal prunes its composed children.
            # Keeping unchanged roots also avoids rerunning unrelated factories.
            for owner, mounted in mounted_items:
                if self._extension_sidebar_widgets.get(owner) is mounted:
                    continue
                with suppress(Exception):
                    await mounted.remove()
                if self._extension_sidebar_mounted.get(owner) is mounted:
                    self._extension_sidebar_mounted.pop(owner, None)
            # Re-read after awaits: rapid updates collapse to the latest target.
            target_items = tuple(self._extension_sidebar_widgets.items())
            for index, (owner, target) in enumerate(target_items):
                if self._extension_sidebar_mounted.get(owner) is target:
                    continue
                later_mounted = next(
                    (
                        self._extension_sidebar_mounted.get(later_owner)
                        for later_owner, _ in target_items[index + 1 :]
                        if self._extension_sidebar_mounted.get(later_owner) is not None
                    ),
                    None,
                )
                try:
                    await slot.mount(target, before=later_mounted)
                except Exception as exc:  # noqa: BLE001 - isolation boundary
                    extension_name, key = owner
                    if self._extension_sidebar_widgets.get(owner) is target:
                        self._extension_sidebar_widgets.pop(owner, None)
                        self._extension_sidebar_contributions.pop(owner, None)
                    self._record_extension_component_failure(
                        f"sidebar:{extension_name}:{key}",
                        exc,
                        notify=True,
                        extension_name=extension_name,
                    )
                    continue
                self._extension_sidebar_mounted[owner] = target
            self._extension_sidebar_mounted = {
                owner: target
                for owner, target in target_items
                if self._extension_sidebar_mounted.get(owner) is target
            }

    def _open_extension_main_view(self, factory: MainViewFactory) -> MainViewHandle:
        # [FUNCTION DOC]
        # Opens an extension main view. Creates a _MainViewHandle with a new future, calls
        # factory(handle, theme) to build the widget; on factory error records a failure, resolves
        # the handle with None and returns a _DeadMainViewHandle. Otherwise stores the widget on the
        # handle, releases any previous handle (its wait resolves None and its close becomes a no-op),
        # makes the new handle current and schedules a serialized main-view reconcile (mounting
        # happens later). Returns the handle immediately. Prompt focus is left unchanged.
        """Open a display-toggled main-area view mounting ``factory(handle, theme)``.

        Prompt focus is intentionally left where it is (the extension widget can
        focus its own composer), so a registered key interceptor keeps firing
        while the prompt is focused and can close the view on Esc.

        The handle is returned synchronously (the factory needs it and callers
        store it at once), but the mount is sequenced after any previous view's
        remove() drains, so switching views never collides on the shared main
        slot. ``is_open`` reports the *intended* state: the new handle is open
        the instant it is returned even though its widget mounts a tick later.
        """
        handle = _MainViewHandle(self, asyncio.get_event_loop().create_future())
        try:
            widget = factory(handle, self.tui_settings.resolved_theme)
        except Exception as exc:  # noqa: BLE001 - isolation boundary
            self._record_extension_component_failure("main_view", exc, notify=True)
            # Nothing awaits this handle (the extension gets the dead one), but
            # resolve it anyway so no future leaks unresolved.
            handle._resolve(None)
            return _DeadMainViewHandle()
        handle.widget = widget
        previous = self._extension_main_view
        if previous is not None and previous is not handle:
            # Superseded (last writer wins): its close() becomes a no-op so a
            # stale Esc/unmount can't tear down the view that replaced it, and
            # its pending wait() resolves with None.
            self._release_main_view_handle(previous)
        self._extension_main_view = handle
        self._schedule_extension_swap(self._reconcile_main_view())
        return handle

    async def _reconcile_main_view(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Under the main-view lock, removes a mounted widget that is no longer the intended
        # one, then mounts the intended widget into "#main-slot", hides the transcript and shows the
        # slot. On mount failure clears and releases the handle, records a failure and restores the
        # transcript. If nothing is intended or mounted, restores the transcript.
        """Make the mounted main view match the intended handle (serialized)."""
        async with self._extension_main_view_lock:
            target = self._extension_main_view
            target_widget = target.widget if target is not None else None
            mounted = self._extension_main_view_mounted
            if mounted is not None and mounted is not target_widget:
                with suppress(Exception):
                    await mounted.remove()
                if self._extension_main_view_mounted is mounted:
                    self._extension_main_view_mounted = None
            # Re-read after the await.
            target = self._extension_main_view
            target_widget = target.widget if target is not None else None
            if target_widget is not None and self._extension_main_view_mounted is not target_widget:
                try:
                    slot = self.query_one("#main-slot", Container)
                    slot.mount(target_widget)
                except Exception as exc:  # noqa: BLE001 - isolation boundary
                    if self._extension_main_view is target:
                        self._extension_main_view = None
                    self._release_main_view_handle(target)
                    self._record_extension_component_failure("main_view", exc, notify=True)
                    self._restore_main_transcript()
                    return
                self._extension_main_view_mounted = target_widget
                with suppress(NoMatches):
                    self.query_one("#transcript", TranscriptView).display = False
                slot.display = True
            elif self._extension_main_view is None and self._extension_main_view_mounted is None:
                # A close (target cleared) with nothing left to show.
                self._restore_main_transcript()

    def _close_extension_main_view(self, handle: _MainViewHandle) -> None:
        # [FUNCTION DOC]
        # Closes the main view if the handle is the current one: clears it and schedules a reconcile
        # (which unmounts and restores the transcript). Stale handles are ignored.
        """Unmount a main view and restore the main transcript (sequenced)."""
        if self._extension_main_view is not handle:
            return
        self._extension_main_view = None
        self._schedule_extension_swap(self._reconcile_main_view())

    def _release_main_view_handle(self, handle: _MainViewHandle | None) -> None:
        # [FUNCTION DOC]
        # HELPER: Marks a host-torn-down handle closed and resolves its wait with None. Ignores None.
        """Mark a host-torn-down handle closed and resolve its ``wait()`` with None.

        Used by the teardown paths the host drives itself (supersede, session
        rebind, mount failure, quarantine) — as opposed to an explicit
        ``handle.close(result)`` from the extension — so a pending ``wait()``
        never leaks unresolved and ``is_open`` reports False.
        """
        if handle is None:
            return
        handle._open = False
        handle._resolve(None)

    def _restore_main_transcript(self) -> None:
        # [FUNCTION DOC]
        # HELPER: If a screen exists, hides the main slot, shows the transcript and puts it back in
        # follow mode, and focuses the prompt (missing widgets ignored).
        """Hide the main slot and bring the main transcript back into focus."""
        if not self.screen_stack:
            return
        with suppress(NoMatches):
            self.query_one("#main-slot", Container).display = False
        with suppress(NoMatches):
            pane = self.query_one("#transcript", TranscriptView)
            pane.display = True
            # Re-anchor the restored main transcript so returning to a live
            # conversation lands at the bottom (mirrors _follow_transcript_output).
            pane.follow_output()
        with suppress(NoMatches):
            self.query_one("#prompt", PromptInput).focus()

    def _refresh_extension_components(self) -> None:
        # [FUNCTION DOC]
        # Calls refresh on every intended slot widget, sidebar section widget and the main view
        # widget, suppressing errors.
        """Re-render all mounted extension widgets (analog of requestRender)."""
        for widget in (
            *self._extension_slot_widgets.values(),
            *self._extension_sidebar_widgets.values(),
        ):
            with suppress(Exception):
                widget.refresh()
        handle = self._extension_main_view
        if handle is not None and handle.widget is not None:
            with suppress(Exception):
                handle.widget.refresh()

    def _clear_extension_components(self) -> None:
        # [FUNCTION DOC]
        # Force-clears all extension UI: clears intended slot widgets and schedules reconciles to
        # unmount every known slot key; clears sidebar contributions/widgets/theme and schedules a
        # sidebar reconcile; clears and releases the main view handle and schedules a main-view
        # reconcile; clears key interceptors; resets the once-per-context failure dedupe set.
        """Force-clear every tracked extension widget, view, and interceptor.

        The runtime drives this through the UI bridge on `/reload` and session
        rebinds (resume/new); it also runs on app teardown, so a leaked
        extension widget never survives a session switch. Intent is cleared
        synchronously — mid-swap reads and in-flight continuations then see
        empty state — while the actual unmounts run on the same serialized
        per-key reconciles as ordinary swaps, so a clear followed immediately
        by a re-mount of a same-id widget (a session_start handler re-mounting
        after a rebind) can never hold two widgets with one id
        (``DuplicateIds``).
        """
        slot_keys = {*self._extension_slot_widgets, *self._extension_slot_mounted}
        self._extension_slot_widgets.clear()
        self._extension_slot_slot_ids.clear()
        for key in slot_keys:
            self._schedule_extension_swap(self._reconcile_slot(key))
        self._extension_sidebar_contributions.clear()
        self._extension_sidebar_widgets.clear()
        self._extension_sidebar_theme = None
        self._schedule_extension_swap(self._reconcile_sidebar())
        handle = self._extension_main_view
        self._extension_main_view = None
        self._release_main_view_handle(handle)
        self._schedule_extension_swap(self._reconcile_main_view())
        self._extension_key_interceptors.clear()
        # A recurring failure context must notify again in the new world.
        self._extension_component_failures_reported.clear()

    def _tracked_extension_widgets(self) -> tuple[Widget, ...]:
        # [FUNCTION DOC]
        # HELPER: Returns a de-duplicated tuple (by identity) of every extension widget the host
        # tracks: intended/mounted slot and sidebar widgets, and intended/mounted main view widget.
        """Return every extension widget the host currently tracks (intended or mounted)."""
        widgets: list[Widget] = []
        seen: set[int] = set()
        for widget in (
            *self._extension_slot_widgets.values(),
            *self._extension_slot_mounted.values(),
            *self._extension_sidebar_widgets.values(),
            *self._extension_sidebar_mounted.values(),
        ):
            if id(widget) not in seen:
                seen.add(id(widget))
                widgets.append(widget)
        handle = self._extension_main_view
        main_widgets = (
            handle.widget if handle is not None else None,
            self._extension_main_view_mounted,
        )
        for main_widget in main_widgets:
            if main_widget is not None and id(main_widget) not in seen:
                seen.add(id(main_widget))
                widgets.append(main_widget)
        return tuple(widgets)

    def _extension_root_for(self, widget: Widget, tracked: tuple[Widget, ...]) -> Widget | None:
        # [FUNCTION DOC]
        # HELPER: Walks up the widget's parent chain and returns the first tracked root it reaches,
        # or None.
        """Return the tracked extension root that owns ``widget``, if any."""
        node: Widget | None = widget
        while node is not None:
            for root in tracked:
                if node is root:
                    return root
            node = node.parent if isinstance(node.parent, Widget) else None
        return None

    def _quarantine_extension_widget(self, error: BaseException) -> bool:
        # [FUNCTION DOC]
        # HELPER: Tries to find which tracked extension widget caused an exception by scanning the
        # traceback frames for a local "self" that is a widget inside a tracked root. If none, returns
        # False. Otherwise hides and disables the culprit; if it is the main view, clears and releases
        # the handle, removes it and restores the transcript; if it is a sidebar section, forgets that
        # section; otherwise forgets it from the slot trackers; removes the widget (errors suppressed);
        # records the failure with a notification (attributed to the extension for sidebar sections).
        # Returns True.
        # Note: when quarantining a slot widget the slot-container id mapping for that key is left
        # behind (harmless, but stale).
        """Remove the tracked extension widget implicated in ``error``.

        Returns True when a culprit was found and torn down (so the app can
        swallow the exception and stay alive), False otherwise (so core bugs
        still surface). Textual runs ``render`` on the compositor's own reflow
        loop, so a child's render/compose/on_mount crash cannot be caught at the
        mount site; walking the traceback for a frame owned by a tracked widget
        is the only handle we get.
        """
        tracked = self._tracked_extension_widgets()
        if not tracked:
            return False
        tb = error.__traceback__
        culprit: Widget | None = None
        while tb is not None:
            candidate = tb.tb_frame.f_locals.get("self")
            if isinstance(candidate, Widget):
                root = self._extension_root_for(candidate, tracked)
                if root is not None:
                    culprit = root
                    break
            tb = tb.tb_next
        if culprit is None:
            return False
        # Suppress the ghost first: a widget that crashed in on_mount never
        # finished mounting, so remove() cannot fully prune it, but hiding and
        # disabling it makes it inert and invisible. A render-crash widget
        # removes cleanly. Either way the app keeps running.
        with suppress(Exception):
            culprit.display = False
        with suppress(Exception):
            culprit.disabled = True
        sidebar_owner: tuple[str, str] | None = None
        if (
            self._extension_main_view is not None and self._extension_main_view.widget is culprit
        ) or self._extension_main_view_mounted is culprit:
            if self._extension_main_view_mounted is culprit:
                self._extension_main_view_mounted = None
            handle = self._extension_main_view
            self._extension_main_view = None
            self._release_main_view_handle(handle)
            with suppress(Exception):
                culprit.remove()
            self._restore_main_transcript()
        else:
            sidebar_owner = next(
                (
                    owner
                    for owner, widget in (
                        *self._extension_sidebar_widgets.items(),
                        *self._extension_sidebar_mounted.items(),
                    )
                    if widget is culprit
                ),
                None,
            )
            if sidebar_owner is not None:
                self._extension_sidebar_widgets.pop(sidebar_owner, None)
                self._extension_sidebar_mounted.pop(sidebar_owner, None)
                self._extension_sidebar_contributions.pop(sidebar_owner, None)
            else:
                for tracker in (self._extension_slot_widgets, self._extension_slot_mounted):
                    key = next((k for k, w in tracker.items() if w is culprit), None)
                    if key is not None:
                        tracker.pop(key, None)
            with suppress(Exception):
                culprit.remove()
        extension_name = sidebar_owner[0] if sidebar_owner else None
        context = (
            f"sidebar:{sidebar_owner[0]}:{sidebar_owner[1]}"
            if sidebar_owner
            else f"render:{id(culprit)}"
        )
        self._record_extension_component_failure(
            context,
            error,
            notify=True,
            extension_name=extension_name,
        )
        return True

    def _handle_exception(self, error: Exception) -> None:
        # [FUNCTION DOC]
        # Overrides Textual's private exception handler: if the error can be attributed to an
        # extension widget it is quarantined and the app keeps running; otherwise defers to Textual
        # (which normally exits the app with the traceback).
        """Quarantine a crashing extension widget instead of tearing down.

        Overrides Textual's private ``App._handle_exception`` (there is no
        public error hook — ``hasattr(App, "on_exception")`` is False on the
        pinned Textual). If the traceback touches a tracked extension widget we
        remove it and keep running; otherwise we defer to Textual's default so
        core's own bugs still surface. This private-API coupling is a contract
        cost the component-seam experiment deliberately accepts.
        """
        if self._quarantine_extension_widget(error):
            return
        super()._handle_exception(error)

    def _record_extension_component_failure(
        self,
        context: str,
        error: BaseException,
        *,
        notify: bool = False,
        extension_name: str | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Logs the full traceback to the app log every time; then, once per context string,
        # reports the failure to the extension runtime (when an extension name is given and a runtime
        # exists) and, if notify, shows an error notification with a summary of up to 120 characters.
        """Diagnose an extension-component failure once per context.

        The notification carries a short exception summary so the failure is
        identifiable at a glance; the full traceback goes to the app log for a
        post-mortem (the two together are what let us pin the deferred-remove
        ``DuplicateIds`` race).
        """
        # Always log the traceback, even on a duplicate context, so a repeating
        # failure leaves a full trail.
        with suppress(Exception):
            self.log.error(
                f"Extension component failed ({context}):\n"
                + "".join(traceback.format_exception(type(error), error, error.__traceback__))
            )
        if context in self._extension_component_failures_reported:
            return
        self._extension_component_failures_reported.add(context)
        if extension_name is not None:
            runtime = getattr(self.session, "extension_runtime", None)
            if runtime is not None:
                runtime.record_ui_failure(extension_name, context, error)
        if notify:
            summary = f"{type(error).__name__}: {error}"
            if len(summary) > 120:
                summary = summary[:117] + "..."
            self._notify(
                f"An extension component failed ({context}) and was removed ({summary}).",
                severity="error",
            )

    def _follow_transcript_output(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Puts the transcript back into follow-output mode (no-op without a screen or
        # transcript).
        """Put the transcript back in follow mode for explicit user actions."""
        if not self.screen_stack:
            return
        with suppress(NoMatches):
            self.query_one("#transcript", TranscriptView).follow_output()

    async def _run_terminal_command(self, command: str, *, add_to_context: bool) -> None:
        # [FUNCTION DOC]
        # HELPER (worker body): Runs a user terminal command ("!cmd") through
        # session.run_terminal_command. If unsupported, shows an error and returns. Adds a "tool"
        # transcript item "$ <command>" (always showing its result), appends it to the transcript
        # and refreshes chrome. Awaits the command; on exception writes a failed result block with
        # the error text into the item, notifies "Could not run command", updates the item and
        # returns. On success (if the item still exists) updates its text to the executed command
        # and its result block (ok flag, whether output was added to the agent context, output),
        # re-follows output and updates the transcript.
        # Inputs: command text and whether to add the output to the conversation context.
        run_terminal_command = getattr(self.session, "run_terminal_command", None)
        if not callable(run_terminal_command):
            self._notify("Terminal commands are not available.", severity="error")
            return

        item_index = len(self.state.items)
        self.state.add_item(
            "tool",
            f"$ {command.strip()}",
            always_show_tool_result=True,
        )
        item = self.state.items[item_index]
        self._follow_transcript_output()
        transcript = self.query_one("#transcript", TranscriptView)
        await transcript.append_item(
            item,
            theme=self.tui_settings.resolved_theme,
            show_tool_results=True,
            scroll_end=True,
        )
        self._refresh_chrome()

        try:
            result = await run_terminal_command(command, add_to_context=add_to_context)
        except Exception as exc:  # noqa: BLE001 - surface command execution failures in the TUI
            if item_index < len(self.state.items):
                item = self.state.items[item_index]
                item.tool_result_text = format_terminal_command_result_block(
                    ok=False,
                    added_to_context=add_to_context,
                    output=str(exc),
                )
            self._notify(f"Could not run command: {exc}", severity="error")
            await transcript.update_item(
                item,
                theme=self.tui_settings.resolved_theme,
                show_tool_results=True,
            )
            self._refresh_chrome()
            return

        if item_index >= len(self.state.items):
            return
        item = self.state.items[item_index]
        item.text = f"$ {result.command}"
        item.tool_result_text = format_terminal_command_result_block(
            ok=result.ok,
            added_to_context=result.added_to_context,
            output=result.output,
        )
        self._follow_transcript_output()
        await transcript.update_item(
            item,
            theme=self.tui_settings.resolved_theme,
            show_tool_results=True,
        )
        self._refresh_chrome()

    def _replace_tui_settings(self, *, theme: TuiThemeName) -> None:
        # [FUNCTION DOC]
        # HELPER: Replaces the immutable TUI settings with a copy whose theme is changed (other
        # fields copied). Does not save.
        """Replace the current immutable TUI settings with a new theme."""
        self.tui_settings = TuiSettings(
            keybindings=self.tui_settings.keybindings,
            theme=theme,
            auto_copy_selection=self.tui_settings.auto_copy_selection,
            sidebar_position=self.tui_settings.sidebar_position,
            turn_notification=self.tui_settings.turn_notification,
        )

    def _set_tui_theme(self, theme: TuiThemeName) -> None:
        # [FUNCTION DOC]
        # HELPER: Sets the TUI theme: unknown names produce an error notification; otherwise updates
        # settings, saves them to disk, applies the Textual theme and refreshes the UI.
        if theme not in available_tui_theme_names():
            self._notify(f"Unknown theme: {theme}", severity="error")
            return
        self._replace_tui_settings(theme=theme)
        save_tui_settings(self.tui_settings)
        self.theme = theme
        self._refresh()

    async def _queue_prompt(
        self,
        text: str,
        *,
        streaming_behavior: Literal["steer", "follow_up"],
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Sends text to the running session as a steering or follow-up message
        # (session.prompt with streaming behavior) and applies any yielded events to the state.
        # Errors are shown as "Could not queue message". Refreshes chrome afterwards.
        """Queue a prompt for the active agent worker."""
        try:
            async for event in self.session.prompt(text, streaming_behavior=streaming_behavior):
                self.adapter.apply(event)
        except Exception as exc:  # noqa: BLE001 - surface queueing failures in the TUI
            self._notify(f"Could not queue message: {exc}", severity="error")
            return
        self._refresh_chrome()

    async def _run_prompt(
        self,
        text: str,
        run_id: int | None = None,
        *,
        source: Literal["interactive", "extension"] = "interactive",
        custom_type: str | None = None,
        details: dict[str, JSONValue] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER (worker body): Runs one prompt through session.prompt and streams events into the
        # UI. For each event: stops if a newer run started (run id mismatch); swallows the user
        # event that confirms an optimistic render (or reconciles a transformed one); otherwise
        # applies the event to TuiState via the adapter (skipping user message-end events when a
        # screen exists, since those are rendered incrementally); syncs selection state; on an
        # assistant error message attaches the diagnostic log path and, unless the session will
        # auto-retry, a retry hint; on a failed overflow compaction attaches the log path; then
        # updates the mounted transcript incrementally; and on AgentSettledEvent fires a terminal
        # notification if unfocused. On unexpected exceptions (for the current run): restores the
        # prompt text into an empty editor if the session has a pending selection, records a
        # formatted error item, stops running state and refreshes. Always clears this run's
        # optimistic messages and, if still current, the prompt worker handle.
        # Inputs: text, optional run id (defaults to the current one), source, custom type, details.
        """Run one prompt and stream session events into the TUI state."""
        active_run_id = self._prompt_run_id if run_id is None else run_id
        try:
            async for event in self.session.prompt(
                text, source=source, custom_type=custom_type, details=details
            ):
                if active_run_id != self._prompt_run_id:
                    return
                if self._consume_optimistic_user_event(event, run_id=active_run_id):
                    self._sync_text_selection_state()
                    self._refresh_chrome()
                    continue
                if self._replace_transformed_optimistic_user_message(event, run_id=active_run_id):
                    self._sync_text_selection_state()
                    continue
                if not (_is_user_message_end_event(event) and self.screen_stack):
                    self.adapter.apply(event)
                self._sync_text_selection_state()
                if (
                    isinstance(event, MessageEndEvent)
                    and isinstance(event.message, AssistantMessage)
                    and event.message.stop_reason == "error"
                ):
                    _attach_diagnostic_log_path_to_error(self.state, self.session)
                    will_auto_retry = getattr(self.session, "will_auto_retry", None)
                    if not (callable(will_auto_retry) and will_auto_retry(event.message)):
                        _attach_retry_hint_to_error(self.state, event.message)
                elif (
                    isinstance(event, CompactionEndEvent)
                    and event.reason == "overflow"
                    and (event.aborted or event.error_message)
                ):
                    _attach_diagnostic_log_path_to_error(self.state, self.session)
                await self._apply_streaming_transcript_event(event)
                if isinstance(event, AgentSettledEvent) and not self._app_has_focus:
                    self._terminal_notification.notify_turn_finished()
        except Exception as exc:  # noqa: BLE001 - surface unexpected worker errors in the TUI
            if active_run_id != self._prompt_run_id:
                return
            if getattr(self.session, "has_pending_selection", False):
                prompt = self.query_one("#prompt", PromptInput)
                if not prompt.text:
                    prompt.text = text
                    prompt.move_cursor(_text_end_location(text))
                    prompt.focus()
            message = _format_prompt_error(exc, self.session)
            self.state.error = message
            self.state.add_item("error", message)
            self.state.running = False
            self._sync_text_selection_state()
            self._refresh()
        finally:
            self._clear_optimistic_user_messages(run_id=active_run_id)
            if active_run_id == self._prompt_run_id:
                self._prompt_worker = None

    async def _apply_streaming_transcript_event(self, event: CodingSessionEvent) -> None:
        # [FUNCTION DOC]
        # HELPER: Applies one session event to mounted transcript widgets without full redraws (falls
        # back to full refresh when no screen/transcript): agent start refreshes chrome; agent end
        # finishes the streaming assistant message; text/thinking deltas append to the streaming
        # widgets; thinking end finalizes thinking; user/custom message ends are appended; assistant
        # message ends with error/aborted stop reasons trigger a full rebuild, multi-block or
        # thinking messages replace provisional widgets with canonical items, and simple messages
        # finalize with the final text; tool start/update/end events update (or append) the tool's
        # transcript item using expanded/collapsed rendering; auto-retry and overflow compaction
        # starts append the newest status item; failed overflow compaction ends rebuild fully;
        # queue updates and everything else refresh chrome only.
        """Apply an agent event to mounted transcript widgets without full redraws."""
        if not self.screen_stack:
            self._refresh()
            return
        theme = self.tui_settings.resolved_theme
        try:
            transcript = self.query_one("#transcript", TranscriptView)
        except NoMatches:
            self._refresh()
            return
        if isinstance(event, AgentStartEvent):
            self._refresh_chrome()
            return
        if isinstance(event, AgentEndEvent):
            await transcript.finish_assistant_message()
            self._refresh_chrome()
            return
        if isinstance(event, MessageStartEvent):
            return
        if isinstance(event, MessageUpdateEvent):
            nested = event.assistant_message_event
            if isinstance(nested, TextDeltaEvent):
                await transcript.append_assistant_delta(nested.delta, theme=theme)
            elif isinstance(nested, ThinkingEndEvent):
                await transcript.finish_thinking_message()
            elif isinstance(nested, ThinkingDeltaEvent):
                await transcript.append_thinking_delta(
                    nested.delta,
                    theme=theme,
                    show_thinking=self.state.show_thinking,
                )
            return
        if isinstance(event, MessageEndEvent):
            if isinstance(event.message, (UserMessage, CustomMessage)):
                await self._append_confirmed_user_message(event.message)
                self._sync_session_title()
                return
            if isinstance(event.message, AssistantMessage):
                if event.message.stop_reason in {"error", "aborted"}:
                    # The adapter projected any partial response plus the error
                    # into canonical display state. Rebuild once at this terminal
                    # boundary so the mounted transcript cannot drop the error.
                    self._refresh()
                    return
                visible_blocks = [
                    block
                    for block in event.message.content
                    if (
                        isinstance(block, TextContent)
                        and bool(block.text)
                        or isinstance(block, ThinkingContent)
                        and bool(block.thinking)
                    )
                ]
                canonical_items = self.state.items[-len(visible_blocks) :] if visible_blocks else []
                if (
                    any(isinstance(block, ThinkingContent) for block in visible_blocks)
                    or len(visible_blocks) > 1
                ):
                    # Replace only this message's provisional streaming widgets;
                    # unrelated history remains mounted and selectable.
                    await transcript.finish_structured_assistant_message(
                        canonical_items,
                        theme=theme,
                        show_thinking=self.state.show_thinking,
                    )
                else:
                    canonical_item = canonical_items[-1] if canonical_items else None
                    await transcript.finish_assistant_message(
                        event.message.text,
                        item=canonical_item,
                    )
                self._refresh_chrome()
                return
            return
        if isinstance(event, ToolExecutionStartEvent):
            await transcript.finish_assistant_message()
            item = self.state.find_tool_item(event.tool_call_id)
            if item is not None:
                expanded = self.state.show_tool_results or item.always_show_tool_result
                updated = await transcript.update_item(
                    item,
                    theme=theme,
                    show_tool_results=expanded,
                    invocation=self.state.resolve_tool_invocation(item, expanded=expanded),
                    result_markup=self.state.resolve_tool_result(item, expanded=expanded),
                )
                if not updated:
                    await transcript.append_item(
                        item,
                        theme=theme,
                        show_tool_results=expanded,
                        invocation=self.state.resolve_tool_invocation(item, expanded=expanded),
                    )
            self._refresh_chrome()
            return
        if isinstance(event, ToolExecutionUpdateEvent):
            await transcript.finish_assistant_message()
            updated_item = self.state.find_tool_item(event.tool_call_id)
            if updated_item is not None:
                expanded = self.state.show_tool_results or updated_item.always_show_tool_result
                await transcript.update_item(
                    updated_item,
                    theme=theme,
                    show_tool_results=expanded,
                    invocation=self.state.resolve_tool_invocation(updated_item, expanded=expanded),
                    result_markup=self.state.resolve_tool_result(updated_item, expanded=expanded),
                )
            self._refresh_chrome()
            return
        if isinstance(event, (AutoRetryStartEvent, CompactionStartEvent)):
            await transcript.finish_assistant_message()
            if self.state.items and (
                isinstance(event, AutoRetryStartEvent) or event.reason == "overflow"
            ):
                await transcript.append_item(
                    self.state.items[-1],
                    theme=theme,
                    show_tool_results=self.state.show_tool_results,
                )
            self._refresh_chrome()
            return
        if isinstance(event, CompactionEndEvent):
            if event.reason == "overflow" and (event.aborted or event.error_message):
                self._refresh()
            else:
                self._refresh_chrome()
            return
        if isinstance(event, ToolExecutionEndEvent):
            updated_item = self.state.find_tool_item(event.tool_call_id)
            if updated_item is not None:
                expanded = self.state.show_tool_results or updated_item.always_show_tool_result
                await transcript.update_item(
                    updated_item,
                    theme=theme,
                    show_tool_results=expanded,
                    invocation=self.state.resolve_tool_invocation(updated_item, expanded=expanded),
                    result_markup=self.state.resolve_tool_result(updated_item, expanded=expanded),
                )
            self._refresh_chrome()
            return
        if isinstance(event, QueueUpdateEvent):
            self._refresh_chrome()
            return
        self._refresh_chrome()

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Escape/cancel action: cancels an active compaction if any, otherwise interrupts the
        # active prompt (both with notifications).
        """Cancel the active compaction or agent turn."""
        if self._cancel_active_compaction(notify=True):
            return
        self._cancel_active_prompt(notify=True)

    def _cancel_active_compaction(self, *, notify: bool) -> bool:
        # [FUNCTION DOC]
        # HELPER: If a compaction worker is active, cancels it, invalidates its run id, clears
        # compaction state, reloads the transcript from the session, refreshes and optionally
        # notifies "Cancelled compaction.". Returns True if something was cancelled, else False.
        """Cancel the active manual compaction worker and restore visible session state."""
        worker = self._compaction_worker
        if worker is None or worker.is_finished or worker.is_cancelled:
            return False

        worker.cancel()
        self._compaction_run_id += 1
        self._compaction_worker = None
        self._compacting = False
        self.state.clear()
        self.state.set_skills(self.session.skills)
        self._load_session_messages_from_session()
        self._refresh()
        if notify:
            self._notify("Cancelled compaction.")
        return True

    def _cancel_active_prompt(self, *, notify: bool, interrupt: bool = False) -> None:
        # [FUNCTION DOC]
        # HELPER: If anything is running (TUI state, session, or an uncancelled worker), invalidates
        # the current run id (so late events are ignored), calls session.cancel if available, cancels
        # the worker, clears running state and the assistant buffer, syncs selection, refreshes and
        # optionally notifies "Interrupted current operation.".
        # KNOWN ISSUE: the interrupt parameter is accepted but ignored.
        """Cancel the active prompt worker and ignore any late events from it."""
        del interrupt
        worker = self._prompt_worker
        is_worker_active = worker is not None and not worker.is_cancelled
        is_session_running = bool(getattr(self.session, "is_running", False))
        if not (self.state.running or is_session_running or is_worker_active):
            return

        self._prompt_run_id += 1
        cancel = getattr(self.session, "cancel", None)
        if callable(cancel):
            cancel()
        if worker is not None and not worker.is_cancelled:
            worker.cancel()
        self._prompt_worker = None
        self.state.running = False
        self.state.assistant_buffer = ""
        self._sync_text_selection_state()
        self._refresh()
        if notify:
            self._notify("Interrupted current operation.")

    def action_accept_completion(self) -> None:
        # [FUNCTION DOC]
        # Tab/accept action. On a model picker toggles mode; on the tools reference opens the
        # selected tool; on other known picker/modal screens selects the cursor item. On the main
        # screen applies the selected completion to the prompt text, moves the cursor to the item's
        # post-apply position and rebuilds completions. No-op if nothing applies.
        """Accept the currently selected prompt completion."""
        if isinstance(self.screen, ModelPickerScreen):
            self.screen.action_toggle_mode()
            return
        if isinstance(self.screen, ToolsReferenceScreen):
            self.screen.action_open_selected()
            return
        if isinstance(
            self.screen,
            SessionPickerScreen
            | PromptTemplatePickerScreen
            | SkillPickerScreen
            | TreePickerScreen
            | LoginMethodPickerScreen
            | LoginProviderPickerScreen
            | ThemePickerScreen
            | ExtensionSelectScreen
            | ExtensionConfirmScreen
            | LocalBackendPickerScreen
            | LocalBackendScreen
            | LocalChoiceConfirmScreen
            | LocalConfirmScreen
            | LocalSearchResultsScreen
            | ProjectTrustScreen,
        ):
            self.screen.action_select_cursor()
            return
        prompt = self.query_one("#prompt", PromptInput)
        item = self._completion_state.selected
        applied = self._apply_selected_completion(prompt.text)
        if applied is None or item is None:
            return
        prompt.text = applied
        cursor = item.cursor_after_apply()
        prompt.cursor_position = cursor
        self._completion_state = self._build_completion_state(prompt.text, cursor=cursor)
        self._refresh_completions()

    def action_completion_next(self) -> None:
        # [FUNCTION DOC]
        # Down action. Moves the cursor down in the sidebar file editor or prompt-template editor,
        # scrolls command output, moves the cursor in known picker screens; on the main screen
        # selects the next completion, or moves the prompt cursor down if no completions.
        """Select the next prompt completion or move down in the active editor."""
        if isinstance(self.focused, TextArea) and self.focused.id == "sidebar-file-editor-input":
            self.focused.action_cursor_down()
            return
        if isinstance(self.screen, PromptTemplateEditorScreen):
            self.screen.query_one("#prompt-template-editor-input", TextArea).action_cursor_down()
            return
        if isinstance(self.screen, CommandOutputScreen):
            self.screen.action_scroll_down()
            return
        if isinstance(
            self.screen,
            SessionPickerScreen
            | PromptTemplatePickerScreen
            | SkillPickerScreen
            | TreePickerScreen
            | LoginMethodPickerScreen
            | LoginProviderPickerScreen
            | ThemePickerScreen
            | ModelPickerScreen
            | ToolsReferenceScreen
            | ExtensionSelectScreen
            | ExtensionConfirmScreen
            | LocalBackendPickerScreen
            | LocalBackendScreen
            | LocalChoiceConfirmScreen
            | LocalConfirmScreen
            | LocalSearchResultsScreen
            | ProjectTrustScreen,
        ):
            self.screen.action_cursor_down()
            return
        if not self._completion_state.items:
            self.query_one("#prompt", PromptInput).action_cursor_down()
            return
        self._completion_state = self._completion_state.select_next()
        self._refresh_completions()

    def action_completion_previous(self) -> None:
        # [FUNCTION DOC]
        # Up action. Mirrors action_completion_next for editors, output and pickers; on the main
        # screen selects the previous completion, or (with no completions) tries in order: pulling
        # the latest queued message back for editing, recalling the previous prompt into an empty
        # editor, and finally moving the prompt cursor up.
        """Select the previous prompt completion or move up in the active editor."""
        if isinstance(self.focused, TextArea) and self.focused.id == "sidebar-file-editor-input":
            self.focused.action_cursor_up()
            return
        if isinstance(self.screen, PromptTemplateEditorScreen):
            self.screen.query_one("#prompt-template-editor-input", TextArea).action_cursor_up()
            return
        if isinstance(self.screen, CommandOutputScreen):
            self.screen.action_scroll_up()
            return
        if isinstance(
            self.screen,
            SessionPickerScreen
            | PromptTemplatePickerScreen
            | SkillPickerScreen
            | TreePickerScreen
            | LoginMethodPickerScreen
            | LoginProviderPickerScreen
            | ThemePickerScreen
            | ModelPickerScreen
            | ToolsReferenceScreen
            | ExtensionSelectScreen
            | ExtensionConfirmScreen
            | LocalBackendPickerScreen
            | LocalBackendScreen
            | LocalChoiceConfirmScreen
            | LocalConfirmScreen
            | LocalSearchResultsScreen
            | ProjectTrustScreen,
        ):
            self.screen.action_cursor_up()
            return
        if not self._completion_state.items:
            if self.action_edit_queued_message():
                return
            if self.action_recall_previous_prompt():
                return
            self.query_one("#prompt", PromptInput).action_cursor_up()
            return
        self._completion_state = self._completion_state.select_previous()
        self._refresh_completions()

    def action_recall_previous_prompt(self) -> bool:
        # [FUNCTION DOC]
        # If the prompt is empty (whitespace-only counts as empty) and history exists, fills the
        # prompt with the most recent submitted prompt, moves the cursor to the end and rebuilds
        # completions. Returns True if recalled, else False.
        """Recall the most recent submitted prompt into an empty prompt input."""
        prompt = self.query_one("#prompt", PromptInput)
        # Only recall into an empty input so an accidental Up press does not
        # erase a prompt the user is still writing.
        if prompt.text.strip() or not self._prompt_history:
            return False
        previous_prompt = self._prompt_history[-1]
        prompt.text = previous_prompt
        prompt.move_cursor(_text_end_location(previous_prompt))
        self._completion_state = self._build_completion_state(prompt.text)
        self._refresh_completions()
        return True

    def action_edit_queued_message(self) -> bool:
        # [FUNCTION DOC]
        # While running and with an empty prompt, pops the latest queued message from the session
        # into the prompt for editing, syncs queue state, rebuilds completions and refreshes.
        # Returns True on success, False otherwise.
        """Move the latest queued message back into the prompt for editing."""
        if not self.state.running:
            return False
        prompt = self.query_one("#prompt", PromptInput)
        if prompt.text.strip():
            return False

        message = self._pop_latest_queued_message()
        if not message:
            return False
        prompt.text = message
        prompt.move_cursor(_text_end_location(message))
        self._sync_queue_state()
        self._completion_state = self._build_completion_state(prompt.text)
        self._refresh()
        return True

    def action_edit_queued_follow_up(self) -> bool:
        # [FUNCTION DOC]
        # Alias of action_edit_queued_message.
        """Move the latest queued message back into the prompt for editing."""
        return self.action_edit_queued_message()

    def _pop_latest_queued_message(self) -> str | None:
        # [FUNCTION DOC]
        # HELPER: Pops the latest queued follow-up message from the session if supported and
        # non-empty, otherwise the latest steering message; returns it or None.
        """Pop the latest queued follow-up or steering message from the session."""
        pop_follow_up = getattr(self.session, "pop_latest_follow_up_message", None)
        if callable(pop_follow_up):
            message = pop_follow_up()
            if isinstance(message, str) and message:
                return message

        pop_steering = getattr(self.session, "pop_latest_steering_message", None)
        if callable(pop_steering):
            message = pop_steering()
            if isinstance(message, str) and message:
                return message

        return None

    def action_open_command_palette(self) -> None:
        # [FUNCTION DOC]
        # Focuses the prompt, replaces its text with "/" (discarding any existing text), puts the
        # cursor after it and shows command completions.
        """Open the slash-command palette in the prompt."""
        prompt = self.query_one("#prompt", PromptInput)
        prompt.focus()
        prompt.text = "/"
        prompt.move_cursor((0, 1))
        self._completion_state = self._build_completion_state(prompt.text)
        self._refresh_completions()

    def action_open_session_picker(self) -> None:
        # [FUNCTION DOC]
        # Opens the session picker: refuses while running ("already working") or when the session
        # has no session manager ("No sessions found."). Pushes an empty SessionPickerScreen in
        # loading state (resuming the chosen id on dismissal) and starts a worker that loads records.
        """Open local sessions immediately, then load other projects."""
        if self.state.running:
            self._notify("Tau is already working. Press Escape to cancel.")
            return
        if getattr(self.session, "session_manager", None) is None:
            self._notify("No sessions found.")
            return
        picker = SessionPickerScreen(
            (),
            local_cwd=Path(self.session.cwd),
            theme=self.tui_settings.resolved_theme,
            loading_other_projects=True,
            current_project_loaded=False,
        )
        self.push_screen(picker, callback=self._handle_session_picker_result)
        self.run_worker(self._refresh_open_session_picker(picker), exclusive=False)

    async def _refresh_open_session_picker(self, picker: SessionPickerScreen) -> None:
        # [FUNCTION DOC]
        # HELPER (worker body): Loads current-project session records in a thread and, once the
        # picker is mounted and still on top, shows them while other projects load (failures warn
        # if the picker is still open). Then loads all active and archived records in threads and
        # updates the picker; on failure ends the loading state and warns.
        """Load session indexes without blocking Textual's event loop."""
        try:
            local_records = await asyncio.to_thread(_local_session_records, self.session)
        except Exception as exc:  # noqa: BLE001 - still attempt the global index
            if self.screen is picker:
                self._notify(f"Could not load current project sessions: {exc}", severity="warning")
        else:
            if not await self._wait_for_open_session_picker(picker):
                return
            picker.update_records(local_records, loading_other_projects=True)

        try:
            records = await asyncio.to_thread(_session_records, self.session)
            archived_records = await asyncio.to_thread(_archived_session_records, self.session)
        except Exception as exc:  # noqa: BLE001 - keep local sessions usable
            if self.screen is picker:
                picker.finish_loading()
                self._notify(f"Could not load other projects: {exc}", severity="warning")
            return
        if await self._wait_for_open_session_picker(picker):
            picker.update_records(records, archived_records=archived_records)

    async def _wait_for_open_session_picker(self, picker: SessionPickerScreen) -> bool:
        # [FUNCTION DOC]
        # HELPER: Returns False if the picker is not the current screen; otherwise yields to the
        # event loop until it is mounted (returning False if it stops being current) and returns
        # True.
        """Wait until this picker is mounted, or report that it was closed."""
        if self.screen is not picker:
            return False
        while not picker.is_mounted:
            await asyncio.sleep(0)
            if self.screen is not picker:
                return False
        return True

    def _open_prompt_template_picker(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Pushes a PromptTemplatePickerScreen over the session's prompt templates with
        # _handle_prompt_template_picker_result as callback.
        self.push_screen(
            PromptTemplatePickerScreen(self.session.prompt_templates),
            callback=self._handle_prompt_template_picker_result,
        )

    def _handle_prompt_template_picker_result(
        self, result: PromptTemplatePickerResult | None
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Focuses the prompt. On cancel does nothing. For "edit": reads the template file
        # (errors notify and reopen the picker) and pushes a PromptTemplateEditorScreen whose result
        # goes to _handle_prompt_template_edit. For "insert": sets the prompt to "/<name>", moves the
        # cursor to the end and refreshes completions.
        prompt = self.query_one("#prompt", PromptInput)
        prompt.focus()
        if result is None:
            return
        if result.action == "edit":
            try:
                source = result.template.path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                self._notify(f"Could not read /{result.template.name}: {exc}", severity="error")
                self._open_prompt_template_picker()
                return
            self.push_screen(
                PromptTemplateEditorScreen(result.template, source),
                callback=lambda edited: self._handle_prompt_template_edit(result.template, edited),
            )
            return
        invocation = f"/{result.template.name}"
        prompt.text = invocation
        prompt.move_cursor(_text_end_location(invocation))
        self._completion_state = self._build_completion_state(invocation)
        self._refresh_completions()

    def _handle_prompt_template_edit(self, template: PromptTemplate, source: str | None) -> None:
        # [FUNCTION DOC]
        # HELPER: If the edit was cancelled, reopens the template picker; otherwise saves it in a
        # worker.
        if source is None:
            self._open_prompt_template_picker()
            return
        self.run_worker(self._save_prompt_template_edit(template, source), exclusive=False)

    async def _save_prompt_template_edit(self, template: PromptTemplate, source: str) -> None:
        # [FUNCTION DOC]
        # HELPER (worker body): Writes the edited template source to its file (UTF-8; on OSError
        # notifies and reopens the picker), then reloads session resources (awaiting if needed). On
        # reload failure notifies that the file was saved but resources were not reloaded; on success
        # refreshes skills, resets completions, refreshes and notifies. Always reopens the picker
        # afterwards (except on write failure, which already reopened it).
        # Note: the write is a plain synchronous write (not atomic) performed on the event loop.
        try:
            template.path.write_text(source, encoding="utf-8")
        except OSError as exc:
            self._notify(f"Could not save /{template.name}: {exc}", severity="error")
            self._open_prompt_template_picker()
            return

        try:
            reload_result = self.session.reload()
            if isawaitable(reload_result):
                await reload_result
        except Exception as exc:  # noqa: BLE001 - saved file remains valid; surface reload errors
            self._notify(
                f"Saved /{template.name}, but could not reload resources: {exc}",
                severity="error",
            )
        else:
            self.state.set_skills(self.session.skills)
            self._completion_state = self._build_completion_state("")
            self._refresh()
            self._notify(f"Saved /{template.name} and reloaded resources.")
        self._open_prompt_template_picker()

    def _open_skills_picker(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Pushes a SkillPickerScreen over the session's skills.
        """Open loaded-skill discovery."""
        self.push_screen(
            SkillPickerScreen(self.session.skills, theme=self.tui_settings.resolved_theme),
            callback=self._handle_skill_picker_result,
        )

    def _handle_skill_picker_result(self, result: SkillPickerResult | None) -> None:
        # [FUNCTION DOC]
        # HELPER: Handles the skill picker result: cancel clears the prompt; "insert" sets the prompt
        # to "/skill:<name>"; "transcript" clears the prompt and adds a status item with the skill
        # name and full content (noting it is not added to context) and refreshes. Moves the cursor
        # to the end and focuses the prompt.
        prompt = self.query_one("#prompt", PromptInput)
        if result is None:
            prompt.text = ""
        elif result.action == "insert":
            prompt.text = f"/skill:{result.skill.name}"
        else:
            prompt.text = ""
            self.state.add_item(
                "status",
                f"Skill: {result.skill.name} (not added to context)\n{result.skill.content}",
            )
            self._refresh()
        prompt.move_cursor(_text_end_location(prompt.text))
        prompt.focus()

    def action_cycle_thinking(self) -> None:
        # [FUNCTION DOC]
        # Cycles the thinking level: uses session.preview_cycle_thinking_level synchronously if
        # available (errors notified) and refreshes chrome; otherwise runs the async
        # _cycle_thinking_level in a worker.
        """Cycle the active thinking mode."""
        preview = getattr(self.session, "preview_cycle_thinking_level", None)
        if preview is not None:
            try:
                preview()
            except Exception as exc:  # noqa: BLE001 - report invalid selections
                self._notify(f"Could not change thinking mode: {exc}", severity="error")
                return
            self._refresh_chrome()
            return
        self.run_worker(self._cycle_thinking_level(), exclusive=False)

    def action_cycle_model(self) -> None:
        # [FUNCTION DOC]
        # Cycles forward through scoped models.
        """Cycle forward through scoped models."""
        self._cycle_model(reverse=False)

    def action_cycle_model_reverse(self) -> None:
        # [FUNCTION DOC]
        # Cycles backward through scoped models.
        """Cycle backward through scoped models."""
        self._cycle_model(reverse=True)

    def _cycle_model(self, *, reverse: bool) -> None:
        # [FUNCTION DOC]
        # HELPER: Refuses while running. Uses session.preview_cycle_scoped_model(reverse) if
        # available (errors notified) and refreshes chrome; otherwise runs _cycle_scoped_model in a
        # worker.
        if self.state.running:
            self._notify("Tau is already working. Press Escape to cancel.")
            return
        preview = getattr(self.session, "preview_cycle_scoped_model", None)
        if preview is not None:
            try:
                preview(reverse=reverse)
            except Exception as exc:  # noqa: BLE001 - report invalid selections
                self._notify(f"Could not switch scoped model: {exc}", severity="error")
                return
            self._refresh_chrome()
            return
        self.run_worker(self._cycle_scoped_model(reverse=reverse), exclusive=False)

    def action_toggle_tool_results(self) -> None:
        # [FUNCTION DOC]
        # Toggles tool result expansion in state and updates transcript tool items in a worker.
        """Toggle inline tool result details without rebuilding unrelated history."""
        self.state.toggle_tool_results()
        self.run_worker(self._update_tool_results_visibility(), exclusive=False)

    async def _update_tool_results_visibility(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Asks the transcript to re-render tool items for the current visibility setting.
        transcript = self.query_one("#transcript", TranscriptView)
        await transcript.update_tool_results_visibility(
            self.state,
            theme=self.tui_settings.resolved_theme,
        )

    def action_toggle_thinking(self) -> None:
        # [FUNCTION DOC]
        # Toggles thinking visibility in state and updates the transcript's thinking widgets.
        """Toggle thinking-token display in the transcript."""
        self.state.toggle_thinking()
        transcript = self.query_one("#transcript", TranscriptView)
        transcript.update_thinking_visibility(
            self.state,
            theme=self.tui_settings.resolved_theme,
        )

    def _handle_session_picker_result(self, session_id: str | None) -> None:
        # [FUNCTION DOC]
        # HELPER: Resumes the selected session id in a worker (no-op when cancelled).
        if session_id is None:
            return
        self.run_worker(self._resume_session(session_id), exclusive=False)

    async def _archive_session_record(
        self, record: SessionCompletionRecord, picker: SessionPickerScreen
    ) -> None:
        # [FUNCTION DOC]
        # HELPER (worker body): Archives one session via the session manager's archive_session in a
        # thread (warns if unsupported; errors notified). If archived, updates the open picker and
        # notifies.
        """Persist a session archive and update the open picker."""
        manager = self.session.session_manager
        archive = getattr(manager, "archive_session", None)
        if not callable(archive):
            self._notify("This session manager does not support archiving.", severity="warning")
            return
        try:
            archived = await asyncio.to_thread(archive, record.id)
        except Exception as exc:  # noqa: BLE001 - surface filesystem failures in the TUI
            self._notify(f"Could not archive session: {exc}", severity="error")
            return
        if archived:
            await picker._remove_records(record.cwd, record.id)
            self._notify(f"Archived session: {record.id}")

    async def _archive_session_project(self, cwd: Path, picker: SessionPickerScreen) -> None:
        # [FUNCTION DOC]
        # HELPER (worker body): Archives every session of a project directory via archive_project in
        # a thread; updates the picker and notifies on success; warnings/errors as above.
        """Persist a project archive and update the open picker."""
        manager = self.session.session_manager
        archive = getattr(manager, "archive_project", None)
        if not callable(archive):
            self._notify("This session manager does not support archiving.", severity="warning")
            return
        try:
            archived = await asyncio.to_thread(archive, cwd)
        except Exception as exc:  # noqa: BLE001 - surface filesystem failures in the TUI
            self._notify(f"Could not archive project: {exc}", severity="error")
            return
        if archived:
            await picker._remove_records(cwd)
            self._notify(f"Archived project: {_short_path(Path(cwd))}")

    async def _unarchive_session_record(
        self, record: SessionCompletionRecord, picker: SessionPickerScreen
    ) -> None:
        # [FUNCTION DOC]
        # HELPER (worker body): Restores one archived session via unarchive_session in a thread;
        # updates the picker and notifies on success.
        """Persist a session restore and update the open picker."""
        manager = self.session.session_manager
        unarchive = getattr(manager, "unarchive_session", None)
        if not callable(unarchive):
            self._notify("This session manager does not support restoring.", severity="warning")
            return
        try:
            restored = await asyncio.to_thread(unarchive, record.id)
        except Exception as exc:  # noqa: BLE001 - surface filesystem failures in the TUI
            self._notify(f"Could not restore session: {exc}", severity="error")
            return
        if restored:
            await picker._restore_records(record.cwd, record.id)
            self._notify(f"Restored session: {record.id}")

    async def _unarchive_session_project(self, cwd: Path, picker: SessionPickerScreen) -> None:
        # [FUNCTION DOC]
        # HELPER (worker body): Restores a whole archived project via unarchive_project in a thread;
        # updates the picker and notifies on success.
        """Persist a project restore and update the open picker."""
        manager = self.session.session_manager
        unarchive = getattr(manager, "unarchive_project", None)
        if not callable(unarchive):
            self._notify("This session manager does not support restoring.", severity="warning")
            return
        try:
            restored = await asyncio.to_thread(unarchive, cwd)
        except Exception as exc:  # noqa: BLE001 - surface filesystem failures in the TUI
            self._notify(f"Could not restore project: {exc}", severity="error")
            return
        if restored:
            await picker._restore_records(cwd)
            self._notify(f"Restored project: {_short_path(Path(cwd))}")

    async def _resume_session(self, session_id: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Resumes a session by id through session.resume, reloads themes, reloads transcript
        # and skills, appends the new short cwd to the message if the working directory changed,
        # and notifies. Errors are shown as "Error: ...". Always refreshes.
        try:
            previous_cwd = Path(self.session.cwd).resolve()
            resume_message = await self.session.resume(session_id)
            self._reload_session_themes()
            self.state.clear()
            self.state.set_skills(self.session.skills)
            self._load_session_messages_from_session()
            current_cwd = Path(self.session.cwd).resolve()
            if current_cwd != previous_cwd:
                resume_message = f"{resume_message} ({_short_path(current_cwd)})"
            self._notify(resume_message)
        except Exception as exc:  # noqa: BLE001 - surface command failures in the TUI
            self._notify(f"Error: {exc}", severity="error")
        self._refresh()

    async def _open_tree_picker(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Opens the tree picker unless the agent or queue is active (warns), the session has
        # no tree support, loading choices fails (error notification), or there are no choices.
        # Pushes TreePickerScreen with label editing wired to _set_tree_label.
        if self._is_agent_or_queue_active():
            self._notify(TREE_RUNNING_MESSAGE, severity="warning")
            return
        tree_choices = getattr(self.session, "tree_choices", None)
        if tree_choices is None:
            self._notify("Session tree is not available.", severity="warning")
            return
        try:
            choices = tuple(await tree_choices())
        except Exception as exc:  # noqa: BLE001 - surface command failures in the TUI
            self._notify(f"Error: {exc}", severity="error")
            return
        if not choices:
            self._notify("No session entries are available for branching.", severity="warning")
            return
        self.push_screen(
            TreePickerScreen(
                choices,
                theme=self.tui_settings.resolved_theme,
                on_label_change=self._set_tree_label,
            ),
            callback=self._handle_tree_picker_result,
        )

    async def _set_tree_label(self, entry_id: str, label: str | None) -> float:
        # [FUNCTION DOC]
        # HELPER: Persists a tree entry label via session.set_label (awaiting if needed), notifies
        # "Label cleared." or "Label set to [...]" and returns the entry's timestamp as a float.
        # Raises RuntimeError if labels are unsupported (and any error from set_label).
        set_label = getattr(self.session, "set_label", None)
        if set_label is None:
            raise RuntimeError("Session labels are not available.")
        entry = set_label(entry_id, label)
        if isawaitable(entry):
            entry = await entry
        self._notify("Label cleared." if label is None else f"Label set to [{label}].")
        return float(entry.timestamp)

    def _handle_tree_picker_result(self, result: TreePickerResult | None) -> None:
        # [FUNCTION DOC]
        # HELPER: On a tree picker result, branches to the entry in a worker with the summary options.
        if result is None:
            return
        self.run_worker(
            self._branch_to_tree_entry(
                result.entry_id,
                summarize=result.summarize,
                custom_instructions=result.custom_instructions,
            ),
            exclusive=False,
        )

    async def _branch_to_tree_entry(
        self,
        entry_id: str,
        *,
        summarize: bool,
        custom_instructions: str | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER (worker body): Branches the session to an entry. Refuses if the agent or queue is
        # active or branching is unsupported. When summarizing, shows "Summarizing branch…" first.
        # Calls session.branch_to_entry (awaiting if needed), reloads the transcript and skills; for a
        # SessionTreeBranchResult prefills the prompt with input_prefill if any and notifies its
        # message; for a string result notifies it. Errors become "Error: ..." notifications. Always
        # refreshes.
        if self._is_agent_or_queue_active():
            self._notify(TREE_RUNNING_MESSAGE, severity="warning")
            return
        branch_to_entry = getattr(self.session, "branch_to_entry", None)
        if branch_to_entry is None:
            self._notify("Session tree is not available.", severity="warning")
            return
        try:
            if summarize:
                self.state.clear()
                self.state.add_item("status", "Summarizing branch…")
                self._refresh()

            result = branch_to_entry(
                entry_id,
                summarize=summarize,
                custom_instructions=custom_instructions,
            )
            if isawaitable(result):
                result = await result
            self.state.clear()
            self.state.set_skills(self.session.skills)
            self._load_session_messages_from_session()
            if isinstance(result, SessionTreeBranchResult):
                if result.input_prefill is not None:
                    prompt = self.query_one("#prompt", PromptInput)
                    prompt.value = result.input_prefill
                    prompt.move_cursor(_text_end_location(result.input_prefill))
                    prompt.focus()
                self._notify(result.message)
            elif isinstance(result, str):
                self._notify(result)
        except Exception as exc:  # noqa: BLE001 - surface command failures in the TUI
            self._notify(f"Error: {exc}", severity="error")
        self._refresh()

    async def _new_session(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Starts a new session: cancels any active prompt silently, calls session.new_session
        # (notifying if unsupported), reloads themes, transcript and skills. Errors notified. Always
        # refreshes.
        self._cancel_active_prompt(notify=False, interrupt=True)
        new_session = getattr(self.session, "new_session", None)
        if new_session is None:
            self._notify("Session manager is not available.")
            return
        try:
            await new_session()
            self._reload_session_themes()
            self.state.clear()
            self.state.set_skills(self.session.skills)
            self._load_session_messages_from_session()
        except Exception as exc:  # noqa: BLE001 - surface command failures in the TUI
            self._notify(f"Error: {exc}", severity="error")
        self._refresh()

    def _apply_selected_completion(self, value: str) -> str | None:
        # [FUNCTION DOC]
        # HELPER: Returns the prompt text with the selected completion applied, or None if nothing is
        # selected.
        item = self._completion_state.selected
        if item is None:
            return None
        return item.apply(value)

    def _append_command_message(
        self,
        command_text: str,
        message: str,
        *,
        system_prompt_inspection: SystemPromptInspection | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Adds a non-persistent status item showing command output: title from
        # _command_output_title (as a Markdown heading with a blank-line separator for /system,
        # otherwise one newline), followed by the message; marks /system output and attaches system
        # prompt sources when an inspection is given.
        """Append non-persistent command output to the visible transcript."""
        is_system_prompt = command_text.split(maxsplit=1)[0].casefold() == "/system"
        separator = "\n\n" if is_system_prompt else "\n"
        title = _command_output_title(command_text)
        if is_system_prompt:
            title = f"### {title}"
        self.state.add_item(
            "status",
            f"{title}{separator}{message}",
            system_prompt=is_system_prompt,
            system_prompt_sources=(
                system_prompt_inspection.sources if system_prompt_inspection is not None else None
            ),
        )

    def _show_command_message(self, command_text: str, message: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Shows command output in a CommandOutputScreen titled from the command, with
        # auto-copy-selection enabled only for /session.
        self.push_screen(
            CommandOutputScreen(
                _command_output_title(command_text),
                message,
                theme=self.tui_settings.resolved_theme,
                auto_copy_selection=command_text.strip().split(maxsplit=1)[0] == "/session",
            )
        )

    def _open_login_picker(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Pushes the LoginMethodPickerScreen with _handle_login_method_result as callback.
        self.push_screen(
            LoginMethodPickerScreen(theme=self.tui_settings.resolved_theme),
            callback=self._handle_login_method_result,
        )

    def _handle_login_method_result(self, method: str | None) -> None:
        # [FUNCTION DOC]
        # HELPER: Routes the chosen login method: "subscription" or "api-key" open a
        # LoginProviderPickerScreen (with BACK support) over the matching built-in providers;
        # "custom" opens the custom-provider form; unknown methods error; no providers warns.
        if method is None:
            return
        if method == "subscription":
            providers = _subscription_login_providers(BUILTIN_PROVIDER_CATALOG)
        elif method == "api-key":
            providers = _api_key_login_providers(BUILTIN_PROVIDER_CATALOG)
        elif method == "custom":
            self._open_custom_provider_login()
            return
        else:
            self._notify(f"Unknown login method: {method}", severity="error")
            return
        if not providers:
            self._notify("No login providers are available for that method.", severity="warning")
            return
        self.push_screen(
            LoginProviderPickerScreen(
                providers,
                theme=self.tui_settings.resolved_theme,
                back_on_cancel=True,
            ),
            callback=lambda provider_name: self._handle_login_provider_result(
                provider_name,
                method=method,
            ),
        )

    def _handle_login_provider_result(
        self,
        provider_name: str | _LoginFlowAction | None,
        *,
        method: str | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: BACK reopens the login method picker; a provider name opens the login for it with
        # the method; None does nothing.
        if provider_name is _LoginFlowAction.BACK:
            self._open_login_picker()
        elif provider_name is not None:
            self._open_login(provider_name, method=method)

    def _open_custom_provider_login(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Pushes the CustomProviderLoginScreen.
        self.push_screen(
            CustomProviderLoginScreen(theme=self.tui_settings.resolved_theme),
            callback=self._handle_custom_provider_login_result,
        )

    def _handle_custom_provider_login_result(
        self,
        result: CustomProviderLoginResult | _LoginFlowAction | None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Handles the custom provider form result: BACK reopens the login method picker;
        # None does nothing. Otherwise builds an OpenAI-compatible provider config (base URL with
        # trailing slashes removed, credential name equal to provider name) and a matching catalog
        # entry, then: saves the entry to the user catalog, stores the API key in the file
        # credential store, upserts the provider into providers settings (not as default) and saves
        # them, reloads session provider settings, and switches the session to the provider (without
        # persisting the default; falls back to the one-argument call on TypeError). Errors notify
        # "Could not save custom provider" and stop. Success notifies and refreshes.
        # Side effects: writes catalog, credential and settings files.
        # KNOWN ISSUE: catching TypeError to detect an older set_provider signature can also hide a
        # genuine TypeError raised inside set_provider and silently retry it.
        if result is _LoginFlowAction.BACK:
            self._open_login_picker()
            return
        if result is None:
            return
        provider = OpenAICompatibleProviderConfig(
            name=result.provider_name,
            base_url=result.base_url.rstrip("/"),
            api_key_env=result.api_key_env,
            credential_name=result.provider_name,
            models=result.models,
            default_model=result.default_model,
        )
        catalog_entry = ProviderCatalogEntry(
            name=provider.name,
            display_name=result.display_name,
            kind="openai-compatible",
            base_url=provider.base_url,
            api_key_env=provider.api_key_env,
            credential_name=provider.credential_name,
            models=provider.models,
            default_model=provider.default_model,
            docs_url=provider.base_url,
        )
        try:
            save_user_catalog_entries((catalog_entry,))
            FileCredentialStore().set(provider.credential_name or provider.name, result.api_key)
            settings = load_provider_settings()
            updated = upsert_openai_compatible_provider(settings, provider, set_default=False)
            save_provider_settings(updated)
            self.session.reload_provider_settings()
            try:
                self.session.set_provider(provider.name, persist_default=False)
            except TypeError:
                self.session.set_provider(provider.name)
        except Exception as exc:  # noqa: BLE001 - surface login failures in the TUI
            self._notify(f"Could not save custom provider: {exc}", severity="error")
            return
        self._notify(f"Saved custom provider {result.display_name}.")
        self._refresh()

    def _open_login(self, provider_name: str, *, method: str | None = None) -> None:
        # [FUNCTION DOC]
        # HELPER: Opens the login screen for a built-in provider (unknown names error). Uses OAuth
        # when the method is "subscription", or when no method was given and the provider does not
        # support API keys, provided an OAuth implementation is registered; for "openai-codex" a
        # wrapper around login_openai_codex is injected. Otherwise opens the API-key LoginScreen.
        # Results route to the navigation handlers (which support BACK).
        entry = builtin_provider_entry(provider_name)
        if entry is None:
            self._notify(f"Unknown provider: {provider_name}", severity="error")
            return
        use_oauth = method == "subscription" or (
            method is None and "api_key" not in entry.auth_methods
        )
        if use_oauth and get_oauth_provider(entry.name) is not None:
            login = None
            if entry.name == "openai-codex":

                async def login(callbacks: OAuthLoginCallbacks) -> OAuthCredential:
                    # [FUNCTION DOC]
                    # HELPER: Adapts OAuthLoginCallbacks to login_openai_codex's keyword callbacks (auth, prompt,
                    # manual code input, progress) and returns the resulting credential.
                    return await login_openai_codex(
                        on_auth=callbacks.on_auth,
                        on_prompt=callbacks.on_prompt,
                        on_manual_code_input=callbacks.on_manual_code_input,
                        on_progress=callbacks.on_progress,
                    )

            self.push_screen(
                OAuthLoginScreen(
                    entry,
                    theme=self.tui_settings.resolved_theme,
                    login=login,
                ),
                callback=lambda credential: self._handle_oauth_login_navigation_result(
                    entry, credential
                ),
            )
            return
        self.push_screen(
            LoginScreen(entry, theme=self.tui_settings.resolved_theme),
            callback=lambda api_key: self._handle_api_key_login_navigation_result(entry, api_key),
        )

    def _handle_api_key_login_navigation_result(
        self,
        entry: ProviderCatalogEntry,
        result: str | _LoginFlowAction | None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: BACK reopens the login method picker; otherwise handles the API key result.
        if result is _LoginFlowAction.BACK:
            self._open_login_picker()
        else:
            self._handle_login_result(entry, result)

    def _handle_login_result(self, entry: ProviderCatalogEntry, api_key: str | None) -> None:
        # [FUNCTION DOC]
        # HELPER: Saves an API key login: ignores None; errors if the provider has no credential
        # name; otherwise stores the key in the credential store, upserts the provider config from
        # the catalog (not default), reloads provider settings and switches the session provider
        # (TypeError fallback as elsewhere). Errors notify "Could not save login". Success notifies
        # and refreshes. Side effects: credential and settings files.
        if api_key is None:
            return
        if entry.credential_name is None:
            self._notify(
                f"Provider {entry.name} does not support saved credentials.",
                severity="error",
            )
            return
        try:
            FileCredentialStore().set(entry.credential_name, api_key)
            provider = provider_config_from_catalog_entry(entry.name)
            upsert_saved_provider(provider, set_default=False)
            self.session.reload_provider_settings()
            try:
                self.session.set_provider(entry.name, persist_default=False)
            except TypeError:
                self.session.set_provider(entry.name)
        except Exception as exc:  # noqa: BLE001 - surface login failures in the TUI
            self._notify(f"Could not save login: {exc}", severity="error")
            return
        self._notify(f"Saved login for {entry.display_name}.")
        self._refresh()

    def _handle_oauth_login_navigation_result(
        self,
        entry: ProviderCatalogEntry,
        result: OAuthCredential | _LoginFlowAction | None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: BACK reopens the login method picker; otherwise handles the OAuth credential.
        if result is _LoginFlowAction.BACK:
            self._open_login_picker()
        else:
            self._handle_oauth_login_result(entry, result)

    def _handle_oauth_login_result(
        self,
        entry: ProviderCatalogEntry,
        credential: OAuthCredential | None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Same as _handle_login_result but stores an OAuth credential (set_oauth).
        if credential is None:
            return
        if entry.credential_name is None:
            self._notify(
                f"Provider {entry.name} does not support saved credentials.",
                severity="error",
            )
            return
        try:
            FileCredentialStore().set_oauth(entry.credential_name, credential)
            provider = provider_config_from_catalog_entry(entry.name)
            upsert_saved_provider(provider, set_default=False)
            self.session.reload_provider_settings()
            try:
                self.session.set_provider(entry.name, persist_default=False)
            except TypeError:
                self.session.set_provider(entry.name)
        except Exception as exc:  # noqa: BLE001 - surface login failures in the TUI
            self._notify(f"Could not save login: {exc}", severity="error")
            return
        self._notify(f"Saved login for {entry.display_name}.")
        self._refresh()

    def _open_logout_picker(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Opens a LoginProviderPickerScreen titled "Logout" over providers that currently
        # have stored credentials; warns with NO_STORED_CREDENTIALS_MESSAGE if none.
        providers = _stored_credential_providers(BUILTIN_PROVIDER_CATALOG)
        if not providers:
            self._notify(NO_STORED_CREDENTIALS_MESSAGE, severity="warning")
            return
        self.push_screen(
            LoginProviderPickerScreen(
                providers,
                theme=self.tui_settings.resolved_theme,
                title="Logout",
            ),
            callback=self._handle_logout_provider_result,
        )

    def _handle_logout_provider_result(self, provider_name: str | _LoginFlowAction | None) -> None:
        # [FUNCTION DOC]
        # HELPER: Logs out of the chosen provider when the result is a provider name.
        if isinstance(provider_name, str):
            self._logout(provider_name)

    def _logout(self, provider_name: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Removes stored credentials for a built-in provider: unknown names error; providers
        # without a credential name or without a stored entry warn with NO_STORED_CREDENTIALS_MESSAGE.
        # Deletes the credential, reloads provider settings (errors notify "Could not log out"), then
        # notifies (OAuth-style wording for openai-codex kind, otherwise noting environment variables
        # and providers.json are unchanged) and refreshes. Side effect: credential file write.
        entry = builtin_provider_entry(provider_name)
        if entry is None:
            self._notify(f"Unknown provider: {provider_name}", severity="error")
            return

        if entry.credential_name is None:
            self._notify(NO_STORED_CREDENTIALS_MESSAGE, severity="warning")
            return
        credential_store = FileCredentialStore()
        if not _credential_store_has_entry(credential_store, entry.credential_name):
            self._notify(NO_STORED_CREDENTIALS_MESSAGE, severity="warning")
            return

        try:
            credential_store.delete(entry.credential_name)
            self.session.reload_provider_settings()
        except Exception as exc:  # noqa: BLE001 - surface logout failures in the TUI
            self._notify(f"Could not log out: {exc}", severity="error")
            return

        if entry.kind == "openai-codex":
            self._notify(f"Logged out of {entry.display_name}.")
        else:
            self._notify(
                f"Removed stored API key for {entry.display_name}. "
                "Environment variables and providers.json config are unchanged."
            )
        self._refresh()

    def _available_model_choices(self) -> tuple[ModelChoice, ...]:
        # [FUNCTION DOC]
        # HELPER: Returns the session's available_model_choices if present, otherwise one choice per
        # available model of the current provider, as a tuple.
        fallback_choices = (
            ModelChoice(provider_name=self.session.provider_name, model=model)
            for model in self.session.available_models
        )
        return tuple(
            getattr(
                self.session,
                "available_model_choices",
                fallback_choices,
            )
        )

    def _open_local_backend_picker(self) -> None:
        # [FUNCTION DOC]
        # Opens the local-backend picker from the extension runtime's local backend registry; warns
        # if unavailable. Result handled by _handle_local_backend_picker_result.
        """Open the generic local-backend chooser and require confirmation."""
        runtime = getattr(self.session, "extension_runtime", None)
        registry = getattr(runtime, "local_backend_registry", None)
        if registry is None:
            self._notify("Local backend controls are unavailable.", severity="warning")
            return
        self.push_screen(
            LocalBackendPickerScreen(registry, theme=self.tui_settings.resolved_theme),
            callback=self._handle_local_backend_picker_result,
        )

    def _handle_local_backend_picker_result(self, backend_id: str | None) -> None:
        # [FUNCTION DOC]
        # HELPER: Defers finishing the picker result to after the picker is popped (Textual invokes
        # the callback before popping the screen).
        # Screen.dismiss() invokes its result callback before popping the screen.
        # Defer the transition so the picker cannot pop the backend screen that
        # this callback opens.
        self.call_later(self._finish_local_backend_picker, backend_id)

    def _finish_local_backend_picker(self, backend_id: str | None) -> None:
        # [FUNCTION DOC]
        # HELPER: Refocuses the prompt; if a backend was chosen and is still available in the
        # registry, pushes a LocalBackendScreen wired to model switching, notifications and an
        # idleness check; otherwise warns that it is no longer available.
        self._restore_prompt_focus()
        if backend_id is None:
            return
        runtime = getattr(self.session, "extension_runtime", None)
        registry = getattr(runtime, "local_backend_registry", None)
        if registry is None or registry.effective(backend_id) is None:
            self._notify("The selected local backend is no longer available.", severity="warning")
            return
        self.push_screen(
            LocalBackendScreen(
                registry,
                backend_id,
                theme=self.tui_settings.resolved_theme,
                on_use=self._use_local_model,
                notify_callback=self._notify_local_backend,
                is_idle=lambda: not self._is_agent_or_queue_active(),
            )
        )

    def _restore_prompt_focus(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Focuses the prompt if it exists.
        with suppress(NoMatches):
            self.query_one("#prompt", PromptInput).focus()

    def _notify_local_backend(self, message: str, level: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Maps a local-backend level ("info"/"warning"/"error", default info) to a Textual
        # severity and shows a deduplicated notification.
        severity: Literal["information", "warning", "error"] = {
            "info": "information",
            "warning": "warning",
            "error": "error",
        }.get(level, "information")  # type: ignore[assignment]
        self._notify(message, severity=severity)

    async def _use_local_model(self, provider_id: str, model_id: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Switches to the given local provider/model unless the agent or queue is active
        # (warns).
        if self._is_agent_or_queue_active():
            self._notify(
                "Tau is still working. Press Escape to interrupt before switching models.",
                severity="warning",
            )
            return
        await self._switch_model(ModelChoice(provider_name=provider_id, model=model_id))

    def _open_tools_reference(self) -> None:
        # [FUNCTION DOC]
        # Pushes a ToolsReferenceScreen for the session's tools and extension tool sources.
        """Open a read-only view of tools from the active session."""
        self.push_screen(
            ToolsReferenceScreen(
                self.session.tools,
                extension_sources=self.session.extension_tool_sources,
                theme=self.tui_settings.resolved_theme,
            )
        )

    def _open_model_picker(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Opens the model picker (picker kind "model") with available and scoped choices,
        # unless there are no choices, no scoped models and no stale active model (warns to /login).
        # Starts a catalog refresh after the first render and switches the model on selection.
        choices = self._available_model_choices()
        scoped = tuple(getattr(self.session, "scoped_model_choices", ()))
        if (
            not choices
            and not scoped
            and not getattr(self.session, "has_stale_active_model", False)
        ):
            self._notify(
                "No configured providers are usable. Run /login to set up a provider.",
                severity="warning",
            )
            return
        picker = ModelPickerScreen(
            choices,
            scoped_choices=scoped,
            current_model=self.session.model,
            provider_name=self.session.provider_name,
            theme=self.tui_settings.resolved_theme,
            on_toggle_scoped=None,
            picker_kind="model",
        )
        picker.on_first_refresh = lambda: self._start_model_picker_refresh(picker)
        self.push_screen(
            picker, callback=lambda choice: self._handle_model_picker_result(picker, choice)
        )

    def _start_model_picker_refresh(self, picker: ModelPickerScreen) -> None:
        # [FUNCTION DOC]
        # HELPER: If the picker is still on top, starts the catalog refresh worker and stores it on
        # the picker (so it can be cancelled).
        if self.screen is picker:
            picker.refresh_worker = self.run_worker(
                self._refresh_open_model_picker(picker), exclusive=False
            )

    async def _refresh_open_model_picker(self, picker: ModelPickerScreen) -> None:
        # [FUNCTION DOC]
        # HELPER (worker body): Calls session.refresh_model_catalogs if available; on error warns
        # (if the picker is still open); on success, if the picker is still on top and mounted,
        # updates its choices.
        refresh = getattr(self.session, "refresh_model_catalogs", None)
        if not callable(refresh):
            return
        try:
            await refresh()
        except Exception as error:
            if self.screen is picker:
                self._notify(f"Could not refresh model catalogs: {error}", severity="warning")
            return
        if self.screen is picker and picker.is_mounted:
            picker.update_choices(
                self._available_model_choices(),
                tuple(getattr(self.session, "scoped_model_choices", ())),
            )

    def _open_scoped_models_picker(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Opens the model picker in "scoped" kind (enter toggles membership via
        # _toggle_scoped_model), recording the initial thinking level; same empty-state guard and
        # refresh behavior as the model picker.
        choices = self._available_model_choices()
        scoped = tuple(getattr(self.session, "scoped_model_choices", ()))
        if (
            not choices
            and not scoped
            and not getattr(self.session, "has_stale_active_model", False)
        ):
            self._notify(
                "No configured providers are usable. Run /login to set up a provider.",
                severity="warning",
            )
            return
        picker = ModelPickerScreen(
            choices,
            scoped_choices=scoped,
            current_model=self.session.model,
            provider_name=self.session.provider_name,
            theme=self.tui_settings.resolved_theme,
            on_toggle_scoped=self._toggle_scoped_model,
            picker_kind="scoped",
            initial_thinking_level=self.session.thinking_level,
        )
        picker.on_first_refresh = lambda: self._start_model_picker_refresh(picker)
        self.push_screen(
            picker, callback=lambda choice: self._handle_scoped_models_picker_result(picker, choice)
        )

    def _toggle_scoped_model(self, choice: ModelChoice) -> Sequence[ModelChoice]:
        # [FUNCTION DOC]
        # HELPER: Calls session.toggle_scoped_model for the choice and returns the new scoped list;
        # if unsupported warns, and on error notifies; in both cases returns the current scoped list.
        toggle_scoped_model = getattr(self.session, "toggle_scoped_model", None)
        if toggle_scoped_model is None:
            self._notify("Scoped model controls are not available.", severity="warning")
            return tuple(getattr(self.session, "scoped_model_choices", ()))
        try:
            return tuple(toggle_scoped_model(choice))
        except Exception as exc:  # noqa: BLE001 - surface session state failures in the TUI
            self._notify(f"Could not update scoped models: {exc}", severity="error")
            return tuple(getattr(self.session, "scoped_model_choices", ()))

    def _handle_scoped_models_picker_result(
        self, picker: ModelPickerScreen, choice: ModelChoice | None
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: On close of the scoped picker cancels its refresh and, if the thinking level
        # changed meanwhile, refreshes chrome after the next refresh. The choice is ignored.
        del choice
        picker.cancel_refresh()
        if (
            picker.initial_thinking_level is not None
            and picker.initial_thinking_level != self.session.thinking_level
        ):
            self.call_after_refresh(self._refresh_chrome)

    def _handle_model_picker_result(
        self, picker: ModelPickerScreen, choice: ModelChoice | None
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Cancels the picker refresh and, if a model was chosen, schedules the switch after
        # the next refresh.
        picker.cancel_refresh()
        if choice is not None:
            self.call_after_refresh(self._start_selected_model_switch, choice)

    def _start_selected_model_switch(self, choice: ModelChoice) -> None:
        # [FUNCTION DOC]
        # HELPER: Runs _switch_model for the choice in a worker.
        self.run_worker(self._switch_model(choice), exclusive=False)

    async def _switch_model(self, choice: ModelChoice) -> None:
        # [FUNCTION DOC]
        # HELPER: Switches the session model using the first available API: preview_model_choice,
        # select_provider_model (awaited if needed), set_model_choice, or set_provider (if the
        # provider changes) plus set_model. Errors notify "Could not switch model". Refreshes chrome
        # on success.
        try:
            select = getattr(self.session, "preview_model_choice", None)
            if select is None:
                select = getattr(self.session, "select_provider_model", None)
            if select is not None:
                result = select(choice)
                if isawaitable(result):
                    await result
            else:
                set_model_choice = getattr(self.session, "set_model_choice", None)
                if set_model_choice is None:
                    if choice.provider_name != self.session.provider_name:
                        self.session.set_provider(choice.provider_name)
                    self.session.set_model(choice.model)
                else:
                    set_model_choice(choice)
        except Exception as exc:  # noqa: BLE001 - surface model switch failures in the TUI
            self._notify(f"Could not switch model: {exc}", severity="error")
            return
        self._refresh_chrome()

    def _open_theme_picker(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Pushes the ThemePickerScreen with the current theme and all available names.
        self.push_screen(
            ThemePickerScreen(
                current_theme=self.tui_settings.theme,
                theme=self.tui_settings.resolved_theme,
                theme_names=available_tui_theme_names(),
            ),
            callback=self._handle_theme_picker_result,
        )

    def _handle_theme_picker_result(self, theme: TuiThemeName | None) -> None:
        # [FUNCTION DOC]
        # HELPER: Applies the chosen theme (ignores cancel).
        if theme is None:
            return
        self._set_tui_theme(theme)

    async def _set_thinking_level(self, level: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Sets the thinking level via preview_thinking_level or set_thinking_level (awaited
        # if needed); warns if unsupported; errors notify. Refreshes chrome on success.
        setter = getattr(self.session, "preview_thinking_level", None)
        if setter is None:
            setter = getattr(self.session, "set_thinking_level", None)
        if setter is None:
            self._notify("Thinking controls are not available.", severity="warning")
            return
        try:
            result = setter(level)
            if isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 - surface session state failures in the TUI
            self._notify(f"Could not change thinking mode: {exc}", severity="error")
            return
        self._refresh_chrome()

    async def _cycle_thinking_level(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Cycles the thinking level via session.cycle_thinking_level (awaited if needed);
        # warns if unsupported; errors notify. Refreshes chrome on success.
        cycler = getattr(self.session, "cycle_thinking_level", None)
        if cycler is None:
            self._notify("Thinking controls are not available.", severity="warning")
            return
        try:
            result = cycler()
            if isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 - surface session state failures in the TUI
            self._notify(f"Could not change thinking mode: {exc}", severity="error")
            return
        self._refresh_chrome()

    async def _cycle_scoped_model(self, *, reverse: bool = False) -> None:
        # [FUNCTION DOC]
        # HELPER: Cycles the scoped model via session.cycle_scoped_model(reverse) (awaited if
        # needed); warns if unsupported; errors notify. Refreshes chrome on success.
        cycler = getattr(self.session, "cycle_scoped_model", None)
        if cycler is None:
            self._notify("Scoped model controls are not available.", severity="warning")
            return
        try:
            result = cycler(reverse=reverse)
            if isawaitable(result):
                result = await result
        except Exception as exc:  # noqa: BLE001 - surface session state failures in the TUI
            self._notify(f"Could not switch scoped model: {exc}", severity="error")
            return
        self._refresh_chrome()

    def _notify(
        self,
        message: str,
        *,
        severity: Literal["information", "warning", "error"] = "information",
    ) -> None:
        # [FUNCTION DOC]
        # Shows a Textual toast (markup disabled) unless an identical (message, severity) toast is
        # already active; the dedupe key expires after the notification timeout via a timer.
        key = (message, severity)
        if key in self._active_notification_keys:
            return
        self._active_notification_keys.add(key)
        self.set_timer(
            self.NOTIFICATION_TIMEOUT,
            lambda: self._active_notification_keys.discard(key),
            name=f"notification-dedupe-{hash(key)}",
        )
        self.notify(message, severity=severity, markup=False)

    def _refresh(self) -> None:
        # [FUNCTION DOC]
        # Full refresh: refreshes chrome and rebuilds the transcript from state with the theme.
        theme = self.tui_settings.resolved_theme
        self._refresh_chrome(theme=theme)
        transcript = self.query_one("#transcript", TranscriptView)
        transcript.update_from_state(self.state, theme=theme)

    def _refresh_chrome(self, *, theme: TuiTheme | None = None) -> None:
        # [FUNCTION DOC]
        # Refreshes everything except the transcript: syncs title, selection state and queue state;
        # updates the sidebar and compact session info from the session; rebuilds extension sidebar
        # sections if the theme changed; re-renders the queued-messages line only when its content or
        # theme changed (hidden when empty); syncs the activity indicator and footer bindings.
        """Refresh non-transcript chrome without remounting transcript blocks."""
        theme = theme or self.tui_settings.resolved_theme
        self._sync_session_title()
        self._sync_text_selection_state()
        self._sync_queue_state()
        sidebar = self.query_one("#sidebar", SessionSidebar)
        sidebar.update_from_session(self.session, theme=theme)
        if self._extension_sidebar_theme != theme:
            self._rebuild_extension_sidebar_sections(theme=theme)
        compact_info = self.query_one("#compact-session-info", CompactSessionInfo)
        compact_info.update_from_session(self.session, theme=theme)
        queued_messages = self.query_one("#queued-messages", Static)
        queue_render_key = (
            self.state.queued_steering,
            self.state.queued_follow_up,
            theme.name,
            theme.muted_text,
        )
        if queue_render_key != self._last_queue_render_key:
            self._last_queue_render_key = queue_render_key
            queued_messages.display = self.state.queued_message_count > 0
            queued_messages.update(_render_queued_messages(self.state, theme=theme))
        self._sync_activity_indicator()
        self._refresh_footer_bindings()

    def _sync_queue_state(self) -> None:
        # [FUNCTION DOC]
        # HELPER: If the session exposes queue_update_event, applies that event to the TUI state so
        # queued counts are current.
        queue_event = getattr(self.session, "queue_update_event", None)
        if not callable(queue_event):
            return
        self.adapter.apply(queue_event())

    def _refresh_chrome_if_mounted(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Refreshes chrome only when screens exist, ignoring missing widgets (teardown races).
        """Refresh chrome when the app is mounted, ignoring teardown races."""
        if not self.screen_stack:
            return
        with suppress(NoMatches):
            self._refresh_chrome()

    def _sync_activity_indicator(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Syncs the terminal title; while working, starts (or resumes) the repeating activity
        # timer (every ACTIVITY_TICK_SECONDS) and applies the indicator; when idle, resets the frame
        # to 0, pauses the timer and applies the idle indicator.
        self._sync_terminal_title()
        if self._is_working():
            if self._activity_timer is None:
                self._activity_timer = self.set_interval(
                    ACTIVITY_TICK_SECONDS,
                    self._tick_activity,
                    name="activity-indicator",
                )
            else:
                self._activity_timer.resume()
            self._apply_activity_indicator()
            return
        self._activity_frame = 0
        if self._activity_timer is not None:
            self._activity_timer.pause()
        self._apply_activity_indicator()

    def _tick_activity(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Timer callback: while working, advances the animation frame, re-renders the
        # indicator and terminal title, and at most once per second schedules a refresh of the
        # running tool row's elapsed time.
        if not self._is_working():
            return
        self._activity_frame += 1
        self._apply_activity_indicator()
        self._sync_terminal_title()
        now = asyncio.get_running_loop().time()
        if now - self._last_tool_timer_refresh_at >= 1.0:
            self._last_tool_timer_refresh_at = now
            self.call_later(self._refresh_pending_tool_timer)

    async def _refresh_pending_tool_timer(self) -> None:
        # [FUNCTION DOC]
        # HELPER: While running, finds the most recent tool transcript item without a result and
        # re-renders it (so its elapsed timer updates). No-op if none or no transcript.
        """Refresh elapsed time on the tool row that is currently executing."""
        if not self.state.running:
            return
        item = next(
            (
                candidate
                for candidate in reversed(self.state.items)
                if candidate.role == "tool" and candidate.tool_result_text is None
            ),
            None,
        )
        if item is None:
            return
        try:
            transcript = self.query_one("#transcript", TranscriptView)
        except NoMatches:
            return
        expanded = self.state.show_tool_results or item.always_show_tool_result
        await transcript.update_item(
            item,
            theme=self.tui_settings.resolved_theme,
            show_tool_results=expanded,
            invocation=self.state.resolve_tool_invocation(item, expanded=expanded),
            result_markup=self.state.resolve_tool_result(item, expanded=expanded),
        )

    def _apply_activity_indicator(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Updates the prompt's left border color and the prompt prefix (tau symbol, "$" in
        # shell mode, or the animated square while working), skipping the update if the render key
        # (theme colors, frame, working state, shell mode) is unchanged. No-op if widgets are missing.
        theme = self.tui_settings.resolved_theme
        try:
            prompt = self.query_one("#prompt", PromptInput)
            prompt_prefix = self.query_one("#prompt-prefix", Static)
        except NoMatches:
            return
        shell_mode = _is_terminal_command_prompt(prompt.text)
        render_key = (
            theme.name,
            theme.accent,
            theme.screen_background,
            theme.prompt_border,
            self._activity_frame,
            self._is_working(),
            shell_mode,
        )
        if render_key == self._last_activity_indicator_key:
            return
        self._last_activity_indicator_key = render_key
        prompt.styles.border_left = (
            "tall",
            _activity_prompt_border_color(
                theme,
                frame=self._activity_frame,
                running=self._is_working(),
                shell_mode=shell_mode,
            ),
        )
        prompt_prefix.update(
            _render_activity_indicator(
                theme,
                frame=self._activity_frame,
                running=self._is_working(),
                shell_mode=shell_mode,
            ),
            layout=False,
        )

    def _refresh_completions(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Shows or hides the autocomplete Static. With no items, resets the line budget and
        # renders an empty state. Otherwise renders a window of the completion items (keeping the
        # selection visible) that fits the line budget and widget width. Always refreshes footer
        # bindings.
        suggestions = self.query_one("#autocomplete", Static)
        suggestions.display = bool(self._completion_state.items)
        if not self._completion_state.items:
            self._completion_visible_line_budget = None
            suggestions.update(
                render_completion_suggestions(
                    CompletionState(),
                    theme=self.tui_settings.resolved_theme,
                )
            )
            self._refresh_footer_bindings()
            return
        max_lines = self._completion_window_line_budget(suggestions)
        suggestions.update(
            render_completion_suggestions(
                _visible_completion_state(
                    self._completion_state,
                    max_lines=max_lines,
                    width=max(suggestions.content_size.width or suggestions.size.width, 1),
                ),
                theme=self.tui_settings.resolved_theme,
            )
        )
        self._refresh_footer_bindings()

    def _completion_window_line_budget(self, suggestions: Static) -> int:
        # [FUNCTION DOC]
        # HELPER: Returns how many lines the completion window may use. Before layout (zero height)
        # uses a cached or initial estimate; afterwards keeps the maximum of the cached budget and the
        # measured limit so navigation never shrinks the box progressively.
        """Return a stable completion window size for the current suggestion box.

        The autocomplete widget has ``height: auto``. If we used its current
        rendered height as the next render limit unconditionally, selecting an
        item could render fewer rows, which would shrink the widget, which would
        then make the next render limit smaller again. Keep the largest measured
        height for the current completion session so navigation does not feed
        back into progressively smaller boxes.
        """
        measured_limit = _completion_visible_line_limit(suggestions)
        if suggestions.size.height <= 0:
            if self._completion_visible_line_budget is None:
                self._completion_visible_line_budget = self._initial_completion_line_budget()
            return self._completion_visible_line_budget
        self._completion_visible_line_budget = max(
            self._completion_visible_line_budget or measured_limit,
            measured_limit,
        )
        return self._completion_visible_line_budget

    def _initial_completion_line_budget(self) -> int:
        # [FUNCTION DOC]
        # HELPER: Estimates the first completion window size: COMPLETION_MAX_VISIBLE_LINES when the
        # terminal height is unknown; otherwise the minimum of the max lines, rows left after
        # reserving the minimum transcript, widget chrome and visible prompt/info/queue widgets, and
        # a third of the terminal height; at least 1.
        """Estimate the first completion window size before Textual lays it out."""
        terminal_height = self.size.height
        if terminal_height <= 0:
            return COMPLETION_MAX_VISIBLE_LINES

        reserved_rows = COMPLETION_MIN_TRANSCRIPT_LINES + COMPLETION_WIDGET_CHROME_LINES
        for selector in ("#prompt-row", "#compact-session-info", "#queued-messages"):
            with suppress(NoMatches):
                widget = self.query_one(selector)
                if widget.display:
                    reserved_rows += widget.outer_size.height

        available_rows = terminal_height - reserved_rows
        terminal_fraction_rows = max(1, terminal_height // COMPLETION_INITIAL_TERMINAL_FRACTION)
        return max(
            1,
            min(COMPLETION_MAX_VISIBLE_LINES, available_rows, terminal_fraction_rows),
        )

    def _update_responsive_layout(self, width: int, height: int) -> None:
        # [FUNCTION DOC]
        # HELPER: Decides sidebar visibility: a session override wins; "off" hides; otherwise shows
        # only when the terminal is at least SIDEBAR_MIN_WIDTH wide and SIDEBAR_MIN_HEIGHT tall.
        # Toggles the "-hide-sidebar" CSS class.
        if self._sidebar_visibility_override is not None:
            show_sidebar = self._sidebar_visibility_override
        elif self.tui_settings.sidebar_position == "off":
            show_sidebar = False
        else:
            show_sidebar = width >= SIDEBAR_MIN_WIDTH and height >= SIDEBAR_MIN_HEIGHT
        self.set_class(not show_sidebar, "-hide-sidebar")

    def _apply_sidebar_position(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Docks the sidebar right when configured "right", or when configured "off" but
        # explicitly shown for this session; toggles the "-sidebar-right" class.
        """Apply the configured (or off-setting fallback) sidebar position."""
        pos = self.tui_settings.sidebar_position
        # A configured ``off`` has no prior visible position, so an explicit
        # session-only show uses the normal right-hand placement.
        show_right = pos == "right" or (pos == "off" and self._sidebar_visibility_override is True)
        self.set_class(show_right, "-sidebar-right")

    def _toggle_sidebar_visibility(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Flips the session-only sidebar visibility override (not saved to settings),
        # reapplies position and layout, and notifies "Sidebar shown/hidden for this session.".
        """Toggle sidebar visibility without changing durable TUI settings."""
        currently_visible = not self.has_class("-hide-sidebar")
        self._sidebar_visibility_override = not currently_visible
        self._apply_sidebar_position()
        self._update_responsive_layout(self.size.width, self.size.height)
        state = "shown" if self._sidebar_visibility_override else "hidden"
        self._notify(f"Sidebar {state} for this session.")

    def _build_completion_state(self, text: str, *, cursor: int | None = None) -> CompletionState:
        # [FUNCTION DOC]
        # HELPER: Builds the completion state for prompt text and cursor using the session's command
        # registry, skills, prompt templates, models, providers (plus login aliases), thinking
        # levels, theme names, the cwd for file references, and session options only when the text
        # starts with "/resume ".
        registry = _session_command_registry(self.session)
        return build_completion_state(
            text,
            cursor=cursor,
            command_registry=registry,
            skills=self.session.skills,
            prompt_templates=self.session.prompt_templates,
            model_names=self.session.available_models,
            provider_names=(
                *self.session.available_providers,
                *LOGIN_PROVIDER_ALIASES,
            ),
            thinking_levels=getattr(self.session, "available_thinking_levels", ()),
            theme_names=available_tui_theme_names(),
            session_options=(_session_options(self.session) if text.startswith("/resume ") else ()),
            cwd=self.session.cwd,
        )

    def _refresh_footer_bindings(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Sets the prompt footer mode from the completion state and working flag.
        prompt = self.query_one("#prompt", PromptInput)
        prompt.set_footer_mode(
            _prompt_footer_mode(self._completion_state, working=self._is_working())
        )

    def _sync_prompt_shell_mode(self, text: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Updates the prompt's shell-mode style and "-shell-mode" class based on whether the
        # text starts with a terminal-command prefix, refreshes it and re-applies the indicator.
        prompt = self.query_one("#prompt", PromptInput)
        prompt.shell_mode_style = self.tui_settings.resolved_theme.role_styles["tool"].border
        prompt.set_class(_is_terminal_command_prompt(text), "-shell-mode")
        prompt.refresh()
        self._apply_activity_indicator()


def _activity_prompt_border_color(
    theme: TuiTheme,
    *,
    frame: int,
    running: bool,
    shell_mode: bool,
) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the prompt border color: the tool role border color in shell mode,
    # otherwise the theme's prompt border. frame and running are ignored (no animation).
    """Return the prompt border color for the current activity animation frame."""
    del frame, running
    if shell_mode:
        return theme.role_styles["tool"].border
    return theme.prompt_border


def _render_activity_indicator(
    theme: TuiTheme,
    *,
    frame: int,
    running: bool,
    shell_mode: bool = False,
) -> Text:
    # [FUNCTION DOC]
    # HELPER: Renders the prompt prefix as rich Text: a bold "$" (tool border color) in shell
    # mode when idle; a bold tau symbol (accent) when idle; while running, a column of
    # ACTIVITY_INDICATOR_HEIGHT rows with a square bouncing up and down by frame, followed by a
    # two-step fading trail blended toward the screen background. Raises ValueError (via color
    # blending) if theme colors are not #rrggbb.
    """Render the prompt prefix: a moving square while running, ``$`` in shell mode."""
    if shell_mode and not running:
        return Text("$", style=f"bold {theme.role_styles['tool'].border}")
    if not running:
        return Text("τ", style=f"bold {theme.accent}")

    cycle_length = (ACTIVITY_INDICATOR_HEIGHT - 1) * 2
    cycle_position = frame % cycle_length
    active_row = (
        cycle_position
        if cycle_position < ACTIVITY_INDICATOR_HEIGHT
        else cycle_length - cycle_position
    )
    direction = 1 if cycle_position < ACTIVITY_INDICATOR_HEIGHT else -1
    trail_rows = {
        active_row: theme.accent,
        active_row - direction: _blend_hex_colors(
            theme.accent,
            theme.screen_background,
            fraction=0.35,
        ),
        active_row - (direction * 2): _blend_hex_colors(
            theme.accent,
            theme.screen_background,
            fraction=0.65,
        ),
    }

    rendered = Text()
    for row in range(ACTIVITY_INDICATOR_HEIGHT):
        color = trail_rows.get(row)
        if color is None:
            rendered.append(" ")
        else:
            rendered.append("■", style=color)
        if row < ACTIVITY_INDICATOR_HEIGHT - 1:
            rendered.append("\n")
    return rendered


def _is_terminal_command_prompt(text: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if the text begins (after whitespace) with "!" or "!!".
    """Return whether the prompt is currently in terminal-command mode."""
    return _terminal_command_prefix_span(text) is not None


def _should_optimistically_render_prompt(text: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if the stripped text is non-empty and not a slash command (slash commands may
    # be expanded by the session, so they are not rendered optimistically).
    """Return whether submitted text can be safely shown before session expansion."""
    stripped = text.strip()
    return bool(stripped) and not stripped.startswith("/")


def _is_user_message_end_event(event: CodingSessionEvent) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if the event is a MessageEndEvent carrying a user or custom message.
    """Return whether an agent event closes a user-context message."""
    return isinstance(event, MessageEndEvent) and isinstance(
        event.message, (UserMessage, CustomMessage)
    )


def _terminal_command_prefix_span(text: str) -> tuple[int, int] | None:
    # [FUNCTION DOC]
    # HELPER: Returns the (start, end) character span of a leading "!!" or "!" prefix after any
    # leading whitespace, or None.
    """Return the input span for a leading ! or !! terminal-command prefix."""
    leading_whitespace = len(text) - len(text.lstrip())
    stripped = text[leading_whitespace:]
    if stripped.startswith("!!"):
        return (leading_whitespace, leading_whitespace + 2)
    if stripped.startswith("!"):
        return (leading_whitespace, leading_whitespace + 1)
    return None


def _blend_hex_colors(start: str, end: str, *, fraction: float) -> str:
    # [FUNCTION DOC]
    # HELPER: Linearly blends two #rrggbb colors by fraction (0 = start, 1 = end), rounding each
    # channel, and returns a lowercase #rrggbb string. Raises ValueError for malformed colors.
    """Blend two ``#rrggbb`` colors by ``fraction``."""
    start_rgb = _hex_to_rgb(start)
    end_rgb = _hex_to_rgb(end)
    blended = tuple(
        round(start_channel + (end_channel - start_channel) * fraction)
        for start_channel, end_channel in zip(start_rgb, end_rgb, strict=True)
    )
    return f"#{blended[0]:02x}{blended[1]:02x}{blended[2]:02x}"


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    # [FUNCTION DOC]
    # HELPER: Parses #rrggbb (leading # optional) into an RGB tuple. Raises ValueError if not six
    # hex digits (or int parsing fails).
    value = color.removeprefix("#")
    if len(value) != 6:
        raise ValueError(f"Expected #rrggbb color, got {color!r}")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def _completion_visible_line_limit(suggestions: Static) -> int:
    # [FUNCTION DOC]
    # HELPER: Returns the widget's height clamped to 1..COMPLETION_MAX_VISIBLE_LINES, or the max
    # when not yet laid out.
    """Return the number of completion render lines that fit in the widget body."""
    if suggestions.size.height > 0:
        return max(min(COMPLETION_MAX_VISIBLE_LINES, suggestions.size.height), 1)
    return COMPLETION_MAX_VISIBLE_LINES


def _visible_completion_state(
    state: CompletionState,
    *,
    max_lines: int,
    width: int | None = None,
) -> CompletionState:
    # [FUNCTION DOC]
    # HELPER: Returns a sub-window of completion items that fits max_lines rendered lines (taking
    # category headers and wrapped descriptions into account) and keeps the selected item
    # visible: first advances the start until the selected item renders above the last line,
    # then trims items from the end, then advances the start further if still too tall. Returns
    # an empty state if there are no items or max_lines is not positive.
    """Return a completion-state window with the selected item visible."""
    if not state.items or max_lines <= 0:
        return CompletionState()

    selected_line_limit = max(max_lines - 1, 1)
    start = 0
    while start < state.selected_index:
        candidate = CompletionState(
            items=state.items[start:],
            selected_index=state.selected_index - start,
        )
        if _completion_selected_render_line(candidate, width=width) < selected_line_limit:
            break
        start += 1

    end = len(state.items)
    while end > state.selected_index + 1:
        candidate = CompletionState(
            items=state.items[start:end],
            selected_index=state.selected_index - start,
        )
        if _completion_render_line_count(candidate, width=width) <= max_lines:
            break
        end -= 1

    while start < state.selected_index:
        candidate = CompletionState(
            items=state.items[start:end],
            selected_index=state.selected_index - start,
        )
        if _completion_render_line_count(candidate, width=width) <= max_lines:
            break
        start += 1

    return CompletionState(
        items=state.items[start:end],
        selected_index=state.selected_index - start,
    )


def _completion_selected_render_line(state: CompletionState, *, width: int | None = None) -> int:
    # [FUNCTION DOC]
    # HELPER: Computes the zero-based rendered line of the selected completion item, counting
    # category header lines, blank separators between categories and extra wrapped lines.
    """Return the rendered line number for the selected completion item."""
    line = 0
    has_rendered_text = False
    previous_category: str | None = None
    for index, item in enumerate(state.items):
        if item.category != previous_category:
            if has_rendered_text:
                line += 1
            if item.category:
                line += 1
                has_rendered_text = True
            previous_category = item.category
        elif has_rendered_text:
            line += 1
        if index == state.selected_index:
            return line
        line += _completion_item_extra_wrapped_lines(item, width=width)
        has_rendered_text = True
    return line


def _completion_render_line_count(state: CompletionState, *, width: int | None = None) -> int:
    # [FUNCTION DOC]
    # HELPER: Computes the total rendered line count of a completion state (items, category
    # headers, separator lines and wrapped descriptions); 0 when empty.
    """Return how many lines the completion state renders into."""
    if not state.items:
        return 0
    line_count = 0
    previous_category: str | None = None
    for index, item in enumerate(state.items):
        if item.category != previous_category:
            if index:
                line_count += 1
            if item.category:
                line_count += 1
            previous_category = item.category
        line_count += 1 + _completion_item_extra_wrapped_lines(item, width=width)
    return line_count


def _completion_item_extra_wrapped_lines(
    item: CompletionItem,
    *,
    width: int | None,
) -> int:
    # [FUNCTION DOC]
    # HELPER: Renders a single completion item to plain text with a Rich console of the given
    # width (using the default dark theme) and returns the number of lines beyond the first; 0
    # when width is unknown/non-positive or the item has no description.
    """Return extra rendered lines used when a completion description wraps."""
    if width is None or width <= 0 or not item.description:
        return 0
    output = StringIO()
    console = Console(
        file=output,
        width=width,
        force_terminal=False,
        color_system=None,
        legacy_windows=False,
    )
    console.print(
        render_completion_suggestions(
            CompletionState(items=(item,), selected_index=0),
            theme=TAU_DARK_THEME,
        ),
        end="",
    )
    line_count = len(output.getvalue().splitlines())
    return max(line_count - 1, 0)


def _session_command_registry(session: CodingSession) -> CommandRegistry:
    # [FUNCTION DOC]
    # HELPER: Returns the session's command registry if it is a CommandRegistry, else a default
    # registry.
    registry = getattr(session, "command_registry", None)
    if isinstance(registry, CommandRegistry):
        return registry
    return create_default_command_registry()


def _session_options(session: CodingSession) -> tuple[CompletionOption, ...]:
    # [FUNCTION DOC]
    # HELPER: Builds resume completion options for all indexed session records.
    return tuple(_session_option(record) for record in _session_records(session))


def _local_session_records(session: CodingSession) -> tuple[SessionCompletionRecord, ...]:
    # [FUNCTION DOC]
    # HELPER: Returns the current project's session records from the session manager (calling
    # list_sessions with the cwd, or without arguments on TypeError), filtered to records whose
    # resolved cwd equals the session's. Empty without a manager.
    """Return only the current project's indexed sessions."""
    manager = getattr(session, "session_manager", None)
    if manager is None:
        return ()
    try:
        records = manager.list_sessions(session.cwd)
    except TypeError:
        records = manager.list_sessions()
    local_cwd = Path(session.cwd).resolve()
    return tuple(record for record in records if Path(record.cwd).resolve() == local_cwd)


def _archived_session_records(session: CodingSession) -> tuple[SessionCompletionRecord, ...]:
    # [FUNCTION DOC]
    # HELPER: Returns archived session records via the manager's list_archived_sessions, or an
    # empty tuple if unsupported.
    """Return archived session metadata for the picker's archived tab."""
    manager = getattr(session, "session_manager", None)
    list_archived = getattr(manager, "list_archived_sessions", None)
    if not callable(list_archived):
        return ()
    return tuple(list_archived())


def _session_records(session: CodingSession) -> tuple[SessionCompletionRecord, ...]:
    # [FUNCTION DOC]
    # HELPER: Returns all indexed session records for resume: calls the manager's list_sessions
    # with no arguments (falling back to passing the cwd on TypeError), then orders records from
    # the current working directory first, followed by the rest, each group in manager order.
    # Empty without a manager.
    # KNOWN ISSUE: the docstring promises newest-first ordering, but no sorting is done here (it
    # relies on the manager's order); also the argument fallback order is the opposite of
    # _local_session_records.
    """Return indexed sessions for resume, current directory first.

    Sessions from the session's working directory are listed newest-first,
    followed by sessions from every other directory (also newest-first), so a
    session from another project is visible in the picker without leaving it.
    """
    manager = getattr(session, "session_manager", None)
    if manager is None:
        return ()
    try:
        records = list(manager.list_sessions())
    except TypeError:
        # Older managers only accept an explicit cwd argument.
        records = list(manager.list_sessions(session.cwd))
    local_cwd = Path(session.cwd).resolve()
    local = [record for record in records if Path(record.cwd).resolve() == local_cwd]
    other = [record for record in records if Path(record.cwd).resolve() != local_cwd]
    return tuple(local) + tuple(other)


def _session_option(record: SessionCompletionRecord) -> CompletionOption:
    # [FUNCTION DOC]
    # HELPER: Builds a completion option for a session: value is the id; description joins the
    # title (or "Untitled session"), the model if any, and the short cwd with " - ".
    description_parts = [record.title if record.title else "Untitled session"]
    if record.model:
        description_parts.append(record.model)
    description_parts.append(_short_path(record.cwd))
    return CompletionOption(value=record.id, description=" - ".join(description_parts))


def _short_path(path: Path) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the path with the home directory replaced by "~" ("~" for home itself), or
    # the full path string if it is outside home.
    home = Path.home()
    try:
        relative = path.relative_to(home)
        return "~" if relative == Path(".") else f"~/{relative}"
    except ValueError:
        return str(path)


def _session_picker_label(record: SessionCompletionRecord) -> str:
    # [FUNCTION DOC]
    # HELPER: Builds a session picker row: relative age, then the title if it is a real name, then
    # the model, joined by two spaces.
    # The project column provides directory context. Keep the model last so it
    # truncates before the relative age and title.
    parts = [_session_updated_at_label(record.updated_at)]
    title = _named_session_title(record.title)
    if title is not None:
        parts.append(title)
    if record.model:
        parts.append(record.model)
    return "  ".join(parts)


def _filter_session_records(
    records: Sequence[SessionCompletionRecord],
    query: str,
) -> tuple[SessionCompletionRecord, ...]:
    # [FUNCTION DOC]
    # HELPER: Filters records whose title or model contains the stripped, casefolded query; all
    # records when the query is blank.
    normalized = query.strip().casefold()
    if not normalized:
        return tuple(records)
    return tuple(
        record
        for record in records
        if normalized in (record.title or "").casefold() or normalized in record.model.casefold()
    )


def _tree_picker_label(
    choice: SessionTreeChoice,
    *,
    theme: TuiTheme,
    highlighted: bool = False,
    show_label_timestamp: bool = False,
) -> Text:
    # [FUNCTION DOC]
    # HELPER: Renders a session-tree choice as rich Text: "* " for the active entry, the label's
    # leading indentation, an optional "[bookmark] " in success color (highlight color when
    # highlighted), an optional label timestamp "YYYY-MM-DD HH:MM" in muted color, then the body
    # where the part before the first ":" (the author/role) is colored with the accent (or
    # highlight) color.
    marker = "* " if choice.active else "  "
    label = choice.label
    indent_width = len(label) - len(label.lstrip(" "))
    indent = label[:indent_width]
    body = label[indent_width:]
    author, separator, rest = body.partition(":")
    text = Text(f"{marker}{indent}")
    if choice.bookmark_label is not None:
        bookmark_color = theme.highlight_text if highlighted else theme.success
        text.append(f"[{choice.bookmark_label}] ", style=bookmark_color)
        if show_label_timestamp and choice.label_timestamp is not None:
            timestamp = datetime.fromtimestamp(choice.label_timestamp).strftime("%Y-%m-%d %H:%M")
            timestamp_color = theme.highlight_text if highlighted else theme.muted_text
            text.append(f"{timestamp} ", style=timestamp_color)
    if separator:
        author_color = theme.highlight_text if highlighted else theme.accent
        text.append(author, style=author_color)
        text.append(f"{separator}{rest}")
    else:
        text.append(body)
    return text


def _active_tree_choice_index(choices: Sequence[SessionTreeChoice]) -> int:
    # [FUNCTION DOC]
    # HELPER: Returns the index of the active tree choice, or 0.
    return _tree_choice_index(choices, None)


def _tree_choice_index(choices: Sequence[SessionTreeChoice], entry_id: str | None) -> int:
    # [FUNCTION DOC]
    # HELPER: Returns the index of the choice with the given entry id if found; otherwise the
    # index of the active choice; otherwise 0.
    if entry_id is not None:
        for index, choice in enumerate(choices):
            if choice.entry_id == entry_id:
                return index
    for index, choice in enumerate(choices):
        if choice.active:
            return index
    return 0


def _session_updated_at_label(timestamp: float) -> str:
    # [FUNCTION DOC]
    # HELPER: Formats a timestamp (seconds) relative to now: "now" (under a minute), "Nm ago",
    # "Nh ago", "yesterday", "Nd ago" (under a week), else a "Mon DD" date. Uses local time.
    """Return a compact relative age label for a session (e.g. ``2h ago``)."""
    delta = (datetime.now() - datetime.fromtimestamp(timestamp)).total_seconds()
    minutes = int(delta // 60)
    if minutes < 1:
        return "now"
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days}d ago"
    return datetime.fromtimestamp(timestamp).strftime("%b %d")


def _named_session_title(title: str | None) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Returns the stripped title, or None for missing, blank or "untitled session"
    # (case-insensitive) titles.
    if title is None:
        return None
    stripped = title.strip()
    if not stripped or stripped.lower() == "untitled session":
        return None
    return stripped


def _login_provider_label(provider: ProviderCatalogEntry) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns "<display name> — <provider id>".
    return f"{provider.display_name} — {provider.name}"


def _subscription_login_providers(
    providers: Sequence[ProviderCatalogEntry],
) -> tuple[ProviderCatalogEntry, ...]:
    # [FUNCTION DOC]
    # HELPER: Returns providers that have a registered OAuth implementation.
    provider_ids = oauth_provider_ids()
    return tuple(provider for provider in providers if provider.name in provider_ids)


def _api_key_login_providers(
    providers: Sequence[ProviderCatalogEntry],
) -> tuple[ProviderCatalogEntry, ...]:
    # [FUNCTION DOC]
    # HELPER: Returns providers whose auth methods include "api_key".
    return tuple(provider for provider in providers if "api_key" in provider.auth_methods)


def _stored_credential_providers(
    providers: Sequence[ProviderCatalogEntry],
) -> tuple[ProviderCatalogEntry, ...]:
    # [FUNCTION DOC]
    # HELPER: Returns providers with a credential name that currently have a stored API key or
    # OAuth credential in the file credential store. Side effect: reads the credential file.
    credential_store = FileCredentialStore()
    return tuple(
        provider
        for provider in providers
        if provider.credential_name is not None
        and _credential_store_has_entry(credential_store, provider.credential_name)
    )


def _credential_store_has_entry(
    credential_store: FileCredentialStore,
    credential_name: str,
) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if the store has an API key or OAuth credential under the name.
    return (
        credential_store.get(credential_name) is not None
        or credential_store.get_oauth(credential_name) is not None
    )


def _theme_picker_label(theme_name: TuiThemeName, *, current_theme: TuiThemeName) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns "✓ <name>" for the current theme, otherwise "  <name>".
    marker = "✓" if theme_name == current_theme else " "
    return f"{marker} {theme_name}"


def _model_picker_label(
    choice: ModelChoice,
    *,
    current_model: str,
    current_provider: str,
    scoped: bool = False,
    unavailable: bool = False,
) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns "<marker><provider>:<model><suffix>" where marker is "* " for the current
    # provider/model and suffix contains " [scoped]" and/or " [unavailable]".
    marker = (
        "* "
        if (choice.provider_name == current_provider and choice.model == current_model)
        else "  "
    )
    suffix = (" [scoped]" if scoped else "") + (" [unavailable]" if unavailable else "")
    return f"{marker}{choice.provider_name}:{choice.model}{suffix}"


def _filter_login_providers(
    providers: Sequence[ProviderCatalogEntry],
    query: str,
) -> tuple[ProviderCatalogEntry, ...]:
    # [FUNCTION DOC]
    # HELPER: Filters providers whose id or display name contains the stripped, casefolded query.
    normalized = query.strip().casefold()
    if not normalized:
        return tuple(providers)
    return tuple(
        provider
        for provider in providers
        if normalized in provider.name.casefold() or normalized in provider.display_name.casefold()
    )


def _filter_model_choices(choices: Sequence[ModelChoice], query: str) -> tuple[ModelChoice, ...]:
    # [FUNCTION DOC]
    # HELPER: Filters model choices whose provider name or model contains the stripped,
    # lowercased query.
    normalized = query.strip().lower()
    if not normalized:
        return tuple(choices)
    return tuple(
        choice
        for choice in choices
        if normalized in choice.provider_name.lower() or normalized in choice.model.lower()
    )


def _command_message_uses_transcript(command_text: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if the command (first word, casefolded) is /reload or /system, whose output is
    # shown inline in the transcript.
    """Return whether slash-command output should appear inline in the transcript."""
    command_name = command_text.split(maxsplit=1)[0].casefold()
    return command_name in {"/reload", "/system"}


def _command_message_uses_notification(command_text: str, message: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: True for /name commands whose message starts with "Session renamed: " (shown as a
    # toast).
    """Return whether slash-command output should appear as a notification."""
    command_name = command_text.split(maxsplit=1)[0].casefold()
    return command_name == "/name" and message.startswith("Session renamed: ")


def _command_output_title(command_text: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns "/<command name>" from the first word of the text ("/help" if empty after
    # removing the slash).
    command_name = command_text.split(maxsplit=1)[0].removeprefix("/")
    return f"/{command_name or 'help'}"


def _is_thinking_cycle_key(key: str, configured_key: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if the key equals the configured thinking-cycle key, or the configured key is
    # "shift+tab" and the terminal reported "backtab".
    if key == configured_key:
        return True
    return configured_key == "shift+tab" and key == "backtab"


def _render_queued_messages(state: TuiState, *, theme: TuiTheme) -> Group:
    # [FUNCTION DOC]
    # HELPER: Renders queued messages as a Rich Group: one row per steering message ("↪ steering
    # · queued: ") then per follow-up ("↳ follow-up · queued: "), each followed by a one-line
    # preview.
    """Render queued prompts stacked above the prompt input."""
    rows: list[Text] = []
    for message in state.queued_steering:
        row = Text("↪ steering · queued: ", style=theme.muted_text)
        row.append(_queued_message_preview(message), style=theme.prompt_text)
        rows.append(row)
    for message in state.queued_follow_up:
        row = Text("↳ follow-up · queued: ", style=theme.muted_text)
        row.append(_queued_message_preview(message), style=theme.prompt_text)
        rows.append(row)
    return Group(*rows)


def _queued_message_preview(message: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the first line of the message, or "" if empty.
    """Return the single-line preview shown above the prompt."""
    lines = message.splitlines()
    return lines[0] if lines else ""


def _prompt_footer_mode(
    completion_state: CompletionState,
    *,
    working: bool,
) -> Literal["normal", "completion", "file_completion", "running"]:
    # [FUNCTION DOC]
    # HELPER: Chooses the prompt footer mode: "file_completion" if a file-reference completion is
    # selected, "completion" for other selections, else "running" while working, else "normal".
    selected = completion_state.selected
    if selected is not None:
        if selected.kind is CompletionKind.FILE_REFERENCE:
            return "file_completion"
        return "completion"
    if working:
        return "running"
    return "normal"


def _key_hint(key: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Formats a key spec for display by capitalizing each "+"-separated part (e.g.
    # "ctrl+j" becomes "Ctrl+J").
    return "+".join(part.capitalize() for part in key.split("+"))


def _app_bindings(keybindings: TuiKeybindings) -> list[Binding]:
    # [FUNCTION DOC]
    # HELPER: Builds the app-level Textual bindings from configured keybindings: cancel, command
    # palette, session picker, thinking cycle, model cycle (forward and hidden reverse), and
    # priority bindings for accept completion, follow-up submission and completion next/previous,
    # plus toggles, clear prompt (copy_message key) and quit.
    return [
        Binding(keybindings.cancel, "cancel", "Cancel"),
        Binding(keybindings.command_palette, "open_command_palette", "Commands"),
        Binding(keybindings.session_picker, "open_session_picker", "Sessions"),
        Binding(keybindings.thinking_cycle, "cycle_thinking", "Thinking"),
        Binding(keybindings.model_cycle, "cycle_model", "Model"),
        Binding(
            keybindings.model_cycle_reverse,
            "cycle_model_reverse",
            "Previous model",
            show=False,
        ),
        Binding(
            keybindings.accept_completion,
            "accept_completion",
            "Complete",
            priority=True,
        ),
        Binding(
            keybindings.queue_follow_up,
            "submit_follow_up",
            "Follow-up",
            priority=True,
        ),
        Binding(
            keybindings.completion_next,
            "completion_next",
            "Next completion",
            priority=True,
        ),
        Binding(
            keybindings.completion_previous,
            "completion_previous",
            "Previous completion",
            priority=True,
        ),
        Binding(keybindings.toggle_tool_results, "toggle_tool_results", "Tool results"),
        Binding(keybindings.toggle_thinking, "toggle_thinking", "Thinking tokens"),
        Binding(keybindings.copy_message, "clear_prompt", "Clear input"),
        Binding(keybindings.quit, "quit", "Quit"),
    ]


def _prompt_bindings(
    keybindings: TuiKeybindings,
    *,
    mode: Literal["normal", "completion", "file_completion", "running"],
) -> list[Binding]:
    # [FUNCTION DOC]
    # HELPER: Builds the PromptInput bindings for a footer mode. Completion modes show Complete
    # (with "/Enter" in the display for non-file completions), Choose and Close, plus "Submit raw"
    # on enter in file completion mode; running mode shows Steer (enter), Follow-up, Cancel and
    # the thinking/tools toggles; normal mode shows Submit, Newline, Commands, Sessions,
    # Thinking, Model, hidden reverse model, Clear and Quit. All are priority bindings, and every
    # other configured action is appended as a hidden priority binding.
    if mode in {"completion", "file_completion"}:
        bindings = [
            Binding(
                keybindings.accept_completion,
                "accept_completion",
                "Complete",
                key_display=(
                    _key_hint(keybindings.accept_completion)
                    if mode == "file_completion"
                    else f"{_key_hint(keybindings.accept_completion)}/Enter"
                ),
                priority=True,
            ),
            Binding(
                keybindings.completion_next,
                "completion_next",
                "Choose",
                key_display=(
                    f"{_key_hint(keybindings.completion_previous)}/"
                    f"{_key_hint(keybindings.completion_next)}"
                ),
                priority=True,
            ),
            Binding(keybindings.cancel, "cancel", "Close", priority=True),
        ]
        if mode == "file_completion":
            bindings.insert(1, Binding("enter", "submit_prompt", "Submit raw", priority=True))
        return bindings + _hidden_prompt_bindings(keybindings, visible_bindings=bindings)
    if mode == "running":
        bindings = [
            Binding("enter", "submit_prompt", "Steer", priority=True),
            Binding(keybindings.queue_follow_up, "submit_follow_up", "Follow-up", priority=True),
            Binding(keybindings.cancel, "cancel", "Cancel", priority=True),
            Binding(
                keybindings.toggle_thinking,
                "toggle_thinking",
                "Thinking",
                priority=True,
            ),
            Binding(
                keybindings.toggle_tool_results,
                "toggle_tool_results",
                "Tools",
                priority=True,
            ),
        ]
        return bindings + _hidden_prompt_bindings(keybindings, visible_bindings=bindings)
    bindings = [
        Binding("enter", "submit_prompt", "Submit", priority=True),
        Binding(
            keybindings.insert_newline,
            "insert_newline",
            "Newline",
            priority=True,
        ),
        Binding(keybindings.command_palette, "open_command_palette", "Commands", priority=True),
        Binding(keybindings.session_picker, "open_session_picker", "Sessions", priority=True),
        Binding(keybindings.thinking_cycle, "cycle_thinking", "Thinking", priority=True),
        Binding(keybindings.model_cycle, "cycle_model", "Model", priority=True),
        Binding(
            keybindings.model_cycle_reverse,
            "cycle_model_reverse",
            "Previous model",
            show=False,
            priority=True,
        ),
        Binding(
            keybindings.copy_message,
            "clear_prompt",
            "Clear",
            priority=True,
        ),
        Binding(keybindings.quit, "quit", "Quit", priority=True),
    ]
    return bindings + _hidden_prompt_bindings(keybindings, visible_bindings=bindings)


def _hidden_prompt_bindings(
    keybindings: TuiKeybindings,
    *,
    visible_bindings: Sequence[Binding],
) -> list[Binding]:
    # [FUNCTION DOC]
    # HELPER: Returns hidden priority bindings for each configured key/action pair whose key is
    # not already used by the visible bindings (comma-separated keys are split).
    visible_keys = {key for binding in visible_bindings for key in binding.key.split(",")}
    candidates = (
        (keybindings.command_palette, "open_command_palette"),
        (keybindings.session_picker, "open_session_picker"),
        (keybindings.queue_follow_up, "submit_follow_up"),
        (keybindings.insert_newline, "insert_newline"),
        (keybindings.thinking_cycle, "cycle_thinking"),
        (keybindings.model_cycle, "cycle_model"),
        (keybindings.model_cycle_reverse, "cycle_model_reverse"),
        (keybindings.toggle_tool_results, "toggle_tool_results"),
        (keybindings.toggle_thinking, "toggle_thinking"),
        (keybindings.copy_message, "clear_prompt"),
        (keybindings.accept_completion, "accept_completion"),
        (keybindings.completion_next, "completion_next"),
        (keybindings.completion_previous, "completion_previous"),
        (keybindings.quit, "quit"),
    )
    return [
        Binding(key, action, show=False, priority=True)
        for key, action in candidates
        if key not in visible_keys
    ]


def _text_end_location(text: str) -> tuple[int, int]:
    # [FUNCTION DOC]
    # HELPER: Returns the (row, column) TextArea location of the end of the text.
    # KNOWN ISSUE: for text that contains exactly one newline as its very first character (e.g.
    # a newline followed by "x"), the part before the last newline is empty, so the row is
    # computed as 0 instead of 1.
    """Return the TextArea cursor location at the end of text."""
    line, _, column_text = text.rpartition("\n")
    return (line.count("\n") + 1 if line else 0, len(column_text))


def _format_prompt_error(exc: BaseException, session: CodingSession) -> str:
    # [FUNCTION DOC]
    # HELPER: Formats an exception as "Error: <message or exception type name>", appending
    # "\nLog: <path>" if the session exposes a diagnostic log Path.
    detail = str(exc) or type(exc).__name__
    message = f"Error: {detail}"
    log_path = getattr(session, "last_diagnostic_log_path", None)
    if isinstance(log_path, Path):
        return f"{message}\nLog: {log_path}"
    return message


_TERMINAL_ERROR_RETRY_HINT = "Run ended before completion. Send a message to retry."


def _attach_retry_hint_to_error(state: TuiState, message: AssistantMessage) -> None:
    # [FUNCTION DOC]
    # HELPER: Unless the error is a context overflow (which is auto-compacted and retried),
    # appends the retry hint to the state's error text and to the most recent error transcript
    # item, if not already present. Mutates state.
    """Clarify that a terminal provider error ended the run and can be retried.

    Context-overflow errors are auto-compacted and retried by the session, so
    they are skipped to avoid asking the user to retry while Tau already is.
    """
    if is_context_overflow_error(message):
        return
    if state.error is not None and _TERMINAL_ERROR_RETRY_HINT not in state.error:
        state.error = f"{state.error}\n{_TERMINAL_ERROR_RETRY_HINT}"
    for item in reversed(state.items):
        if item.role == "error":
            if _TERMINAL_ERROR_RETRY_HINT not in item.text:
                item.text = f"{item.text}\n{_TERMINAL_ERROR_RETRY_HINT}"
            return


def _attach_diagnostic_log_path_to_error(state: TuiState, session: CodingSession) -> None:
    # [FUNCTION DOC]
    # HELPER: If the session has a diagnostic log Path and the state has an error, rewrites the
    # error as "Error: <error>\nLog: <path>", updates the latest error item's text (or adds a new
    # error item). Mutates state.
    # KNOWN ISSUE: not idempotent; each call prefixes another "Error: " and appends another log
    # line, and errors that already start with "Error: " get the prefix twice.
    log_path = getattr(session, "last_diagnostic_log_path", None)
    if not isinstance(log_path, Path) or state.error is None:
        return
    message = f"Error: {state.error}\nLog: {log_path}"
    state.error = message
    for item in reversed(state.items):
        if item.role == "error":
            item.text = message
            return
    state.add_item("error", message)


def _explicit_resume_record(
    manager: SessionManager,
    *,
    session_id: str | None,
) -> CodingSessionRecord | None:
    # [FUNCTION DOC]
    # HELPER: Returns the session record for an explicit --session id via the manager, None when
    # no id is given. Raises RuntimeError("Unknown session: ...") if not found.
    if session_id is None:
        return None
    record = manager.get_session(session_id)
    if record is None:
        raise RuntimeError(f"Unknown session: {session_id}")
    return record


def _create_startup_session_record(
    manager: SessionManager,
    *,
    cwd: Path,
    selection: ProviderSelection,
    inference_provider: str | None = None,
) -> CodingSessionRecord:
    # [FUNCTION DOC]
    # HELPER: Prepares a new session record for startup with cwd, model and provider; when an
    # inference provider is given, passes it with mode "fixed", falling back on TypeError to
    # older manager signatures (without mode, then without inference provider).
    # Returns the CodingSessionRecord. Side effects depend on the manager (may create files).
    if inference_provider is None:
        return manager.prepare_session(
            cwd=cwd,
            model=selection.model,
            provider_name=selection.provider.name,
        )
    try:
        return manager.prepare_session(
            cwd=cwd,
            model=selection.model,
            provider_name=selection.provider.name,
            inference_provider=inference_provider,
            inference_provider_mode="fixed",
        )
    except TypeError:
        try:
            return manager.prepare_session(
                cwd=cwd,
                model=selection.model,
                provider_name=selection.provider.name,
                inference_provider=inference_provider,
            )
        except TypeError:
            return manager.prepare_session(
                cwd=cwd,
                model=selection.model,
                provider_name=selection.provider.name,
            )


def _resolve_tui_startup_selection(
    settings: Any,
    *,
    record: Any | None,
    provider_name: str | None,
    model: str | None,
    explicit_resume: bool,
) -> ProviderSelection:
    # [FUNCTION DOC]
    # HELPER: Chooses the startup provider/model: explicit provider or model wins
    # (resolve_provider_selection); on explicit resume uses the record's provider/model if
    # resolvable; otherwise the default selection if it has usable credentials, else the first
    # provider with usable credentials, else the default. May raise ProviderConfigError from
    # resolution. Reads the credential store.
    if provider_name is not None or model is not None:
        return resolve_provider_selection(settings, provider_name=provider_name, model=model)

    if explicit_resume:
        record_selection = _selection_from_session_record(settings, record)
        if record_selection is not None:
            return record_selection

    default_selection = resolve_provider_selection(settings)
    if provider_has_usable_credentials(
        default_selection.provider,
        credential_reader=FileCredentialStore(),
    ):
        return default_selection

    fallback_selection = _first_usable_startup_selection(settings)
    return fallback_selection or default_selection


def _first_usable_startup_selection(settings: Any) -> ProviderSelection | None:
    # [FUNCTION DOC]
    # HELPER: Returns a selection for the first configured provider with usable credentials
    # (with its default model), or None.
    credential_store = FileCredentialStore()
    for provider in settings.providers:
        if provider_has_usable_credentials(provider, credential_reader=credential_store):
            return ProviderSelection(provider=provider, model=provider.default_model)
    return None


def _selection_from_session_record(settings: Any, record: Any | None) -> ProviderSelection | None:
    # [FUNCTION DOC]
    # HELPER: Derives a selection from a session record: None without a record or model; if the
    # record names a provider, resolves it (any error yields None); otherwise the first usable
    # scoped model choice with the same model, otherwise the first credentialed provider listing
    # that model; else None.
    if record is None:
        return None
    record_model = getattr(record, "model", None)
    if not isinstance(record_model, str) or not record_model:
        return None

    record_provider = getattr(record, "provider_name", None)
    if isinstance(record_provider, str) and record_provider:
        try:
            return resolve_provider_selection(
                settings,
                provider_name=record_provider,
                model=record_model,
            )
        except Exception:
            return None

    for choice in _usable_scoped_startup_choices(settings):
        if choice.model == record_model:
            return resolve_provider_selection(
                settings,
                provider_name=choice.provider_name,
                model=choice.model,
            )

    credential_store = FileCredentialStore()
    for provider in settings.providers:
        if record_model not in provider.models:
            continue
        if not provider_has_usable_credentials(provider, credential_reader=credential_store):
            continue
        return ProviderSelection(provider=provider, model=record_model)
    return None


def _usable_scoped_startup_choices(settings: Any) -> tuple[ModelChoice, ...]:
    # [FUNCTION DOC]
    # HELPER: Returns scoped model settings entries whose provider exists, lists the model, and
    # has usable credentials, as ModelChoice values.
    credential_store = FileCredentialStore()
    choices: list[ModelChoice] = []
    for item in settings.scoped_models:
        try:
            provider = settings.get_provider(item.provider)
        except Exception:
            continue
        if item.model not in provider.models:
            continue
        if not provider_has_usable_credentials(provider, credential_reader=credential_store):
            continue
        choices.append(ModelChoice(provider_name=item.provider, model=item.model))
    return tuple(choices)


def _resource_conflict_alert(
    diagnostics: Sequence[ResourceDiagnostic],
) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Builds one startup alert listing skill/prompt diagnostics that report overriding a
    # lower-precedence resource ("- skill 'x': <path> overrides <shadowed path>"), with a header
    # and a closing instruction. Returns None when there are no such conflicts.
    """Format skill and prompt precedence conflicts as one startup alert."""
    prefix = "overrides lower-precedence resource at "
    conflicts = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.kind in {"skill", "prompt"}
        and diagnostic.name is not None
        and diagnostic.path is not None
        and diagnostic.message.startswith(prefix)
    ]
    if not conflicts:
        return None

    lines = ["Conflicting skills/prompts detected:"]
    for diagnostic in conflicts:
        resource_kind = "skill" if diagnostic.kind == "skill" else "prompt template"
        shadowed_path = diagnostic.message.removeprefix(prefix)
        lines.append(
            f"- {resource_kind} '{diagnostic.name}': {diagnostic.path} overrides {shadowed_path}"
        )
    lines.append("Rename or remove duplicate resources to clear this alert.")
    return "\n".join(lines)


def _startup_inference_provider(
    selection: ProviderSelection,
    record: CodingSessionRecord | None,
) -> str | None:
    # [FUNCTION DOC]
    # HELPER: For the "huggingface" OpenAI-compatible provider only, returns the inference
    # provider for the selected model: the record's value when the record uses the same model,
    # else the provider config's per-model mapping (or None). None for other providers.
    provider = selection.provider
    if not isinstance(provider, OpenAICompatibleProviderConfig) or provider.name != "huggingface":
        return None
    if record is not None and record.model == selection.model:
        return record.inference_provider
    return provider.inference_providers.get(selection.model)


def _startup_inference_provider_mode(
    selection: ProviderSelection,
    record: CodingSessionRecord | None,
) -> Literal["automatic", "fixed"]:
    # [FUNCTION DOC]
    # HELPER: Returns the record's inference provider mode when its model matches the selection;
    # otherwise "fixed" if a configured inference provider exists for the model, else "automatic".
    if record is not None and record.model == selection.model:
        return record.inference_provider_mode
    return "fixed" if _startup_inference_provider(selection, None) is not None else "automatic"


async def run_tui_app(
    *,
    model: str | None,
    cwd: Path,
    session_id: str | None = None,
    new_session: bool = False,
    provider_name: str | None = None,
    auto_compact_token_threshold: int | None = None,
    initial_prompt: str | None = None,
    session_manager: SessionManager | None = None,
    startup_notice: str | None = None,
    startup_update_notice: str | None = None,
    startup_notices: Sequence[str] = (),
    extension_paths: tuple[Path, ...] = (),
    extensions_enabled: bool = True,
    project_extensions_enabled: bool = False,
    custom_system_prompt: str | None = None,
    append_system_prompt: str | None = None,
    trust_override: TrustOverride | None = None,
    thinking_level_override: ThinkingLevel | None = None,
) -> str | None:
    # [FUNCTION DOC]
    # Entry point used by the CLI to run the interactive TUI.
    # Steps: applies the Herdr mouse workaround; rejects using a session id together with
    # new_session (RuntimeError); loads provider and shell settings; uses the given or a default
    # SessionManager; loads the explicit resume record (RuntimeError if unknown); resolves the
    # startup selection, tolerating ProviderConfigError only for dynamic providers (resuming a
    # record with a provider, or explicit provider and model), re-raising otherwise. Determines
    # provider/model names, the Hugging Face inference provider and mode, and creates the
    # initial model provider; if creation fails with RuntimeError (or there is no selection and
    # nothing explicit) uses a LoginRequiredProvider and a login-required startup message. Creates
    # a startup session record when not resuming (ProviderConfigError if neither selection nor an
    # explicit provider+model exists), prepares the CodingSession (trust prompting, extensions,
    # system prompt overrides, deferred writes) and adopts it; if the user cancelled project
    # trust returns None. Loads custom themes, assembles startup notices and resource-conflict
    # alerts, builds TauTuiApp, wires the session's trust prompt to the app, and runs the app.
    # Finally closes the session and, if the session did not take ownership, the initial
    # provider (errors suppressed).
    # Returns the active session id if the session was persisted (known to the manager), else
    # None. Raises RuntimeError / ProviderConfigError / ValueError from setup as described.
    # Side effects: terminal I/O, settings and session file access.
    """Run the Textual app and return the active id when its session is persisted."""
    _configure_herdr_textual_mouse()
    if new_session and session_id is not None:
        raise RuntimeError("--session and --new-session cannot be used together")

    provider_settings = load_provider_settings()
    shell_settings = load_shell_settings()
    manager = session_manager or SessionManager()
    record = _explicit_resume_record(
        manager,
        session_id=session_id,
    )
    selection: ProviderSelection | None = None
    try:
        selection = _resolve_tui_startup_selection(
            provider_settings,
            record=record,
            provider_name=provider_name,
            model=model,
            explicit_resume=session_id is not None,
        )
    except ProviderConfigError:
        # A resumed record may point at a process-local provider that is not in
        # durable settings. Let the staged loader resolve it after trusted
        # built-in/project extensions are loaded.
        dynamic_resume = (
            session_id is not None
            and record is not None
            and record.provider_name is not None
            and provider_name is None
            and model is None
        )
        explicit_dynamic = provider_name is not None and model is not None
        if not dynamic_resume and not explicit_dynamic:
            raise
    startup_message: str | None = None
    startup_error_notice: str | None = None
    explicit_selection = provider_name is not None or model is not None
    selected_provider_name: str = (
        provider_name
        if provider_name is not None
        else (record.provider_name if record is not None else None)
        or (selection.provider.name if selection is not None else DEFAULT_PROVIDER_NAME)
    )
    selected_model = (
        model
        if explicit_selection and model is not None
        else (record.model if record is not None else None)
        or (selection.model if selection is not None else DEFAULT_MODEL)
    )
    # Keep static-provider construction compatible with embedded TUI callers,
    # while dynamic providers are deliberately left for CodingSession.load()
    # after trusted extension setup. The provider passed below is owned by the
    # prepared session when the real loader is used.
    initial_provider: ClosableModelProvider | None = None
    runtime_provider_config: ProviderConfig | None = selection.provider if selection else None
    inference_provider = _startup_inference_provider(selection, record) if selection else None
    inference_provider_mode: Literal["automatic", "fixed"] = (
        _startup_inference_provider_mode(selection, record) if selection else "automatic"
    )
    if selection is not None:
        try:
            initial_provider = create_model_provider(
                selection.provider,
                model=selection.model,
                inference_provider=inference_provider,
                thinking_level=resolve_startup_thinking_level(
                    selection.provider,
                    selection.model,
                    cli_override=thinking_level_override,
                ),
            )
        except RuntimeError as exc:
            login_required_message = (
                "Login required. Run /login to choose a provider, "
                f"or /login {selected_provider_name} to continue with the current provider."
            )
            startup_message = f"{login_required_message}\n\nStartup error: {exc}"
            startup_error_notice = (
                f"Startup provider creation failed for "
                f"{selection.provider.name}:{selection.model}: {exc}"
            )
            initial_provider = LoginRequiredProvider(startup_message)
            runtime_provider_config = None
    elif not explicit_selection:
        startup_message = (
            "Login required. Run /login to choose a provider, "
            f"or /login {selected_provider_name} to continue with the current provider."
        )
        initial_provider = LoginRequiredProvider(startup_message)
    session: CodingSession | None = None
    try:
        index_on_first_persist = False
        if record is None:
            if selection is not None:
                record = _create_startup_session_record(
                    manager,
                    cwd=cwd,
                    selection=selection,
                    inference_provider=inference_provider,
                )
            else:
                if provider_name is None or model is None:
                    raise ProviderConfigError(
                        "An explicit provider and model are required for this startup."
                    )
                record = manager.prepare_session(
                    cwd=cwd,
                    model=model,
                    provider_name=provider_name,
                )
            index_on_first_persist = manager.get_session(record.id) is None

        prepared = await prepare_coding_session(
            CodingSessionConfig(
                provider=initial_provider,
                model=record.model or selected_model,
                cwd=record.cwd,
                storage=jsonl_session_storage(record.path),
                session_id=record.id,
                session_manager=manager,
                provider_name=selected_provider_name,
                inference_provider=inference_provider,
                inference_provider_mode=inference_provider_mode,
                requested_provider=provider_name if explicit_selection else None,
                requested_model=model if explicit_selection else None,
                session_provider_name=record.provider_name,
                provider_settings=provider_settings,
                runtime_provider_config=runtime_provider_config,
                auto_compact_token_threshold=auto_compact_token_threshold,
                index_on_first_persist=index_on_first_persist,
                shell_command_prefix=shell_settings.shell_command_prefix,
                extension_paths=extension_paths,
                extensions_enabled=extensions_enabled,
                project_extensions_enabled=project_extensions_enabled,
                custom_system_prompt=custom_system_prompt,
                append_system_prompt=append_system_prompt,
                thinking_level_override=thinking_level_override,
                trust_override=trust_override,
                trust_default=shell_settings.default_project_trust,
                trust_interactive=True,
                trust_prompt=prompt_project_trust,
                defer_authoritative_writes=True,
                owns_initial_provider=initial_provider is not None,
            ),
            session_loader=CodingSession,
        )
        try:
            session = await prepared.adopt()
        except ValueError:
            candidate = prepared.session
            trust_resolution = getattr(candidate, "project_trust_resolution", None)
            if trust_resolution is None or not trust_resolution.cancelled:
                raise
            # The preparation object already closed the unpublished candidate.
            # Do not close that candidate again from the outer finally block.
            del candidate
            return None
        trust_resolution = getattr(session, "project_trust_resolution", None)
        if trust_resolution is not None and trust_resolution.cancelled:
            return None

        theme_dirs = getattr(session, "theme_dirs", None)
        if theme_dirs is None:
            trusted = trust_resolution is None or trust_resolution.trusted
            theme_dirs = TauResourcePaths(
                cwd=record.cwd,
                project_resources_enabled=trusted,
            ).themes_dirs
        custom_themes, theme_diagnostics = load_custom_tui_themes(theme_dirs)
        set_custom_tui_themes(custom_themes)
        legacy_notices = (startup_notice,) if startup_notice else ()
        error_notices = (startup_error_notice,) if startup_error_notice else ()
        theme_notices = tuple(diagnostic.format() for diagnostic in theme_diagnostics)
        all_startup_notices = tuple(
            (*error_notices, *startup_notices, *legacy_notices, *theme_notices)
        )
        resource_conflict_alert = _resource_conflict_alert(
            getattr(session, "resource_diagnostics", ())
        )
        startup_alerts = (resource_conflict_alert,) if resource_conflict_alert is not None else ()
        app = TauTuiApp(
            session,
            tui_settings=load_tui_settings(),
            startup_message=startup_message,
            startup_update_notice=startup_update_notice,
            startup_alerts=startup_alerts,
            startup_notices=all_startup_notices,
            initial_prompt=initial_prompt,
        )
        set_trust_prompt = getattr(session, "set_project_trust_prompt", None)
        if set_trust_prompt is not None:
            prompt_trust = getattr(app, "prompt_project_trust", None)
            if prompt_trust is not None:
                set_trust_prompt(prompt_trust)
        await app.run_async()
    finally:
        if session is not None:
            close_session = getattr(session, "aclose", None)
            if close_session is not None:
                await close_session()
        # Compatibility for lightweight test/embedded session loaders that do
        # not expose ownership. A real CodingSession owns the exact candidate,
        # so this branch does not double-close it.
        if (
            initial_provider is not None
            and getattr(session, "provider", None) is not initial_provider
        ):
            with suppress(Exception):
                await initial_provider.aclose()

    active_session_id: str | None = getattr(session, "session_id", None)
    if active_session_id is None or manager.get_session(active_session_id) is None:
        return None
    return active_session_id
