# [MODULE DOC]
# Generic Textual modal screens that host local inference backends (such as a
# llama.cpp integration) without knowing their protocols: a backend picker, the
# main backend screen (status, model list, action menu, progress bar, async
# operations via the local backend registry), a schema-driven configuration
# form, yes/no and multi-choice confirmations, a search-result variant chooser,
# and a single-input model reference prompt. Also defines a block-style progress
# bar renderable and label/size formatting helpers. Type aliases describe the
# host callbacks: use a model (provider id, model id; may be async), notify
# (message, level) and an idle check.
"""Generic Textual host screens for local backends.

This module intentionally knows only the local-backend contracts. Protocol
names and provider-specific management concepts stay in backend extensions.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from typing import ClassVar, Literal, cast

from rich.console import Console, ConsoleOptions, RenderResult
from rich.style import StyleType
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.events import DescendantFocus, Key
from textual.renderables.bar import Bar as BarRenderable
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, ProgressBar, Select, Static

from tau_coding.local_backends import (
    LocalAction,
    LocalBackendRegistry,
    LocalBackendStatus,
    LocalConfigureSpec,
    LocalConfigValues,
    LocalConfirmationChoice,
    LocalConfirmationRequest,
    LocalModel,
    LocalOperationResult,
    LocalProgress,
    LocalSearchResult,
    ProgressCallback,
)
from tau_coding.tui.config import TuiTheme

LocalUseCallback = Callable[[str, str], Awaitable[None] | None]
LocalNotifyCallback = Callable[[str, str], None]
LocalIdleCallback = Callable[[], bool]


class _BlockProgressRenderable(BarRenderable):
    # [CLASS DOC]
    # HELPER: Rich renderable replacing Textual's bar: draws solid blocks for the
    # highlighted range and a thin line elsewhere.
    """Render determinate progress as solid blocks over a thin track."""

    def __init__(
        self,
        highlight_range: tuple[float, float] = (0, 0),
        highlight_style: StyleType = "default",
        background_style: StyleType = "default",
        **_: object,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: store the highlighted range and the highlight/background styles;
        # extra keyword arguments from Textual are ignored.
        self.highlight_range = highlight_range
        self.highlight_style = highlight_style
        self.background_style = background_style

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # [FUNCTION DOC]
        # HELPER: yield one text line of the available width where each cell whose
        # center lies within the highlight range is a full block in the highlight style
        # and other cells are a line in the background style.
        del console
        width = options.max_width
        start, end = self.highlight_range
        bar = Text(end="")
        for cell in range(width):
            highlighted = cell + 0.5 >= start and cell + 0.5 < end
            bar.append(
                "█" if highlighted else "─",
                style=self.highlight_style if highlighted else self.background_style,
            )
        yield bar


class _LocalDownloadProgressBar(ProgressBar):
    # [CLASS DOC]
    # HELPER: progress bar widget using the block renderable for download progress.
    BAR_RENDERABLE = _BlockProgressRenderable


class LocalBackendPickerScreen(ModalScreen[str | None]):
    # [CLASS DOC]
    # Modal that lists effective local backends (marking recommended and
    # unavailable ones) and dismisses with the chosen backend id, or nothing on
    # cancel. Preselects the recommended backend, else the first.
    """Explicitly confirm a backend choice, including when there is one."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "confirm", "Select", show=False),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
    ]

    def __init__(self, registry: LocalBackendRegistry, *, theme: TuiTheme) -> None:
        # [FUNCTION DOC]
        # Store registry and theme, snapshot the effective backend views, and choose the
        # initial selection (recommended, first, or nothing when empty).
        super().__init__()
        self.registry = registry
        self.theme = theme
        self.views = registry.effective_backends()
        recommended = next((view for view in self.views if view.recommended), None)
        self.selected = (
            recommended.backend.id
            if recommended is not None
            else self.views[0].backend.id
            if self.views
            else None
        )

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yield the title and either an empty-state message or help text, the backend
        # list and a footer.
        with Vertical(id="local-backend-picker"):
            yield Static("Local backends", id="local-backend-picker-title")
            if not self.views:
                yield Static("No local backends are available.", id="local-backend-empty")
            else:
                yield Static(
                    "Choose a backend. The recommended choice is marked.",
                    id="local-backend-picker-help",
                )
                yield ListView(
                    *[ListItem(Label(self._label(view), markup=False)) for view in self.views],
                    id="local-backend-list",
                )
                yield Static(
                    "↑/↓ navigate - Enter selects - Escape closes",
                    id="local-backend-picker-footer",
                )

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # When backends exist, place the list cursor at the selected backend and focus it.
        if self.views:
            backend_list = self.query_one("#local-backend-list", ListView)
            backend_list.index = self._selected_index()
            backend_list.focus()

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Intercept up, down and enter keys (stopping propagation) and route them to the
        # cursor-up, cursor-down and select-cursor actions so the focused list handles them.
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
        # Record the selected backend id and dismiss the modal with it.
        event.stop()
        self.selected = self.views[event.index].backend.id
        self.dismiss(self.selected)

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Move the list cursor up and sync the selection. Raises a Textual query error
        # if no list exists (empty backend state).
        self.query_one("#local-backend-list", ListView).action_cursor_up()
        self._sync_selected()

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Move the list cursor down and sync the selection (same empty-state caveat).
        self.query_one("#local-backend-list", ListView).action_cursor_down()
        self._sync_selected()

    def action_select_cursor(self) -> None:
        # [FUNCTION DOC]
        # Ask the list to select the highlighted item, triggering the selected handler.
        self.query_one("#local-backend-list", ListView).action_select_cursor()

    def action_confirm(self) -> None:
        # [FUNCTION DOC]
        # Enter binding: same as selecting the cursor item.
        self.action_select_cursor()

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismiss with nothing.
        self.dismiss(None)

    def _sync_selected(self) -> None:
        # [FUNCTION DOC]
        # HELPER: update the selected backend id from the list index (nothing when no backends).
        if not self.views:
            self.selected = None
            return
        index = self.query_one("#local-backend-list", ListView).index
        if index is not None and 0 <= index < len(self.views):
            self.selected = self.views[index].backend.id

    def _selected_index(self) -> int:
        # [FUNCTION DOC]
        # HELPER: index of the selected backend in the views, defaulting to 0.
        if self.selected is None:
            return 0
        return next(
            (index for index, view in enumerate(self.views) if view.backend.id == self.selected),
            0,
        )

    @staticmethod
    def _label(view) -> str:  # type: ignore[no-untyped-def]
        # [FUNCTION DOC]
        # HELPER: display name plus "Recommended" and "unavailable" markers.
        marker = " — Recommended" if view.recommended else ""
        effective = "" if view.use_available else " — unavailable"
        return f"{view.backend.display_name}{marker}{effective}"


class LocalBackendScreen(ModalScreen[None]):
    # [CLASS DOC]
    # Main modal for one backend. Shows status text, a model list and an action
    # menu (Tab switches section), a progress line and bar, and runs one registry
    # operation at a time (refresh, configure, doctor, reset, load/unload/download
    # model, search) as an asyncio task, rendering results, confirmations,
    # diagnostics and notifications. Reattaches to server-owned downloads and
    # watches them to completion. Operations require the host to be idle.
    """Generic local-backend action screen."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "cancel", "Close"),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "select_cursor", "Select", show=False),
        Binding("tab", "toggle_section", "Switch section", show=False),
        Binding("shift+tab", "toggle_section", "Switch section", show=False),
    ]

    def __init__(
        self,
        registry: LocalBackendRegistry,
        backend_id: str,
        *,
        theme: TuiTheme,
        on_use: LocalUseCallback | None = None,
        notify_callback: LocalNotifyCallback | None = None,
        is_idle: LocalIdleCallback | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Store registry, backend id, theme, optional use/notify/idle callbacks (with
        # no-op/always-idle defaults) and initialize status, selection, list items,
        # task handles, progress subscription and fraction, and the closing flag.
        super().__init__()
        self.registry = registry
        self.backend_id = backend_id
        self.theme = theme
        self.on_use = on_use
        self._notify_callback = notify_callback or (lambda message, level: None)
        self._is_idle = is_idle or (lambda: True)
        self.status: LocalBackendStatus | None = None
        self._selected_model_id: str | None = None
        self._model_items: tuple[str, ...] = ()
        self._action_items: tuple[tuple[str, str], ...] = ()
        self._worker: asyncio.Task[None] | None = None
        self._active_action: LocalAction | None = None
        self._use_task: asyncio.Task[None] | None = None
        self._download_watch_task: asyncio.Task[None] | None = None
        self._progress_unsubscribe: Callable[[], None] | None = None
        self._progress_fraction: float | None = None
        self._closing = False

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yield title, help, status placeholder, model and action section titles and
        # lists, a hidden progress message, a full-width block progress bar and a footer.
        with Vertical(id="local-backend-screen"):
            yield Static("Local backend", id="local-backend-title")
            yield Static(
                "Choose a model or backend action.",
                id="local-backend-help",
            )
            yield Static("Looking for a local server…", id="local-backend-status")
            yield Static("Models", id="local-model-section-title")
            yield ListView(id="local-model-list")
            yield Static("Actions", id="local-action-section-title")
            yield ListView(id="local-action-menu")
            yield Static("", id="local-backend-progress")
            progress_bar = _LocalDownloadProgressBar(
                total=1,
                show_percentage=False,
                show_eta=False,
                id="local-backend-progress-bar",
            )
            progress_bar.styles.width = "100%"
            yield progress_bar
            yield Static(
                "↑/↓ navigate - Tab switches section - Enter selects - Escape closes",
                id="local-backend-footer",
            )

    async def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Render default sections (configure, refresh), focus actions, hide progress
        # widgets, attach to an in-flight download (starting a completion watcher if
        # attached), then start an immediate refresh probe. Side effects: tasks.
        await self._render_sections(None)
        self.query_one("#local-action-menu", ListView).focus()
        self.query_one("#local-backend-progress", Static).styles.display = "none"
        progress_bar = self.query_one("#local-backend-progress-bar", ProgressBar)
        progress_bar.styles.display = "none"
        progress_bar.query_one("#bar").styles.width = "1fr"
        # Reattach to a server-owned download before probing so refresh output
        # cannot hide its replayed byte progress.
        if self._attach_download_progress():
            self._download_watch_task = asyncio.create_task(self._watch_download_completion())
        # Opening a backend is an explicit user action, so probe its effective
        # saved/environment/default endpoint immediately. Backends still own
        # which endpoint that means; the generic host never scans. A detached
        # download observer keeps its progress visible during this refresh.
        self._start_operation("refresh")

    def on_unmount(self) -> None:
        # [FUNCTION DOC]
        # Mark closing; cancel any running non-download operation in the registry and
        # its worker task (downloads are server-owned and only detached); unsubscribe
        # from progress; cancel the download watcher and any pending model-use task.
        """Stop host-owned tasks before Textual detaches this modal."""
        self._closing = True
        # A server-side download belongs to llama.cpp, not this modal. Closing
        # the UI only detaches from it; explicit cancellation remains available
        # from the Actions section after reopening /local.
        if self._active_action is not None and self._active_action != "download_model":
            self.registry.cancel(self.backend_id, self._active_action)
            if self._worker is not None and not self._worker.done():
                self._worker.cancel()
        if self._progress_unsubscribe is not None:
            self._progress_unsubscribe()
            self._progress_unsubscribe = None
        if self._download_watch_task is not None and not self._download_watch_task.done():
            self._download_watch_task.cancel()
        if self._use_task is not None and not self._use_task.done():
            self._use_task.cancel()

    def _attach_download_progress(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: subscribe to registry progress for the download action, scheduling
        # rendering after refresh. Returns whether a subscription was established
        # (i.e. a download is running).
        def progress(item: LocalProgress) -> None:
            # [FUNCTION DOC]
            # HELPER: progress callback deferring rendering to after the next UI refresh.
            self.call_after_refresh(self._render_observed_download_progress, item)

        self._progress_unsubscribe = self.registry.observe_progress(
            self.backend_id,
            "download_model",
            cast(ProgressCallback, progress),
        )
        return self._progress_unsubscribe is not None

    def _render_observed_download_progress(self, item: LocalProgress) -> None:
        # [FUNCTION DOC]
        # HELPER: show download progress with the bar, only while still subscribed and
        # the download is still running (ignores stale replays).
        # A replay queued during mount can run after the operation finishes.
        # Never let that stale callback restore downloading text over refreshed
        # available-model state.
        if self._progress_unsubscribe is not None and self.registry.operation_running(
            self.backend_id, "download_model"
        ):
            self._set_progress(item.message, fraction=item.fraction, show_bar=True)

    async def _watch_download_completion(self) -> None:
        # [FUNCTION DOC]
        # HELPER: poll every 0.25 seconds until the download stops (returning quietly
        # when cancelled), unsubscribe from progress, wait for any running worker, then
        # clear the fraction and start a refresh if the UI is still usable.
        try:
            while self.registry.operation_running(self.backend_id, "download_model"):
                await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            return
        if self._progress_unsubscribe is not None:
            self._progress_unsubscribe()
            self._progress_unsubscribe = None
        if self._can_update_ui:
            worker = self._worker
            if worker is not None and not worker.done():
                with suppress(asyncio.CancelledError):
                    await asyncio.shield(worker)
            if self._can_update_ui:
                self._progress_fraction = None
                self._start_operation("refresh")

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        # [FUNCTION DOC]
        # Update section focus styling when the model list or action menu gains focus.
        if event.widget.id in {"local-model-list", "local-action-menu"}:
            self._update_section_focus()

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Intercept up, down and enter keys (stopping propagation) and route them to the
        # cursor-up, cursor-down and select-cursor actions so the focused list handles them.
        if event.key == "up":
            event.stop()
            self.action_cursor_up()
        elif event.key == "down":
            event.stop()
            self.action_cursor_down()
        elif event.key == "enter":
            event.stop()
            self.action_select_cursor()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        # [FUNCTION DOC]
        # Sync the selected model when the model list highlight moves.
        if event.list_view.id == "local-model-list":
            self._sync_selected_model()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # [FUNCTION DOC]
        # Stop the event; a model selection records the model and activates it, an
        # action selection activates that action.
        event.stop()
        if event.list_view.id == "local-model-list":
            if 0 <= event.index < len(self._model_items):
                self._selected_model_id = self._model_items[event.index]
                self._activate_selected_model()
        elif event.list_view.id == "local-action-menu":
            self._activate_action(event.index)

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Move up within the focused list; from the top of the actions (or no index)
        # jump to the last model and focus the model list. Syncs the selected model.
        models = self.query_one("#local-model-list", ListView)
        actions = self.query_one("#local-action-menu", ListView)
        if actions.has_focus and self._model_items and actions.index in {None, 0}:
            models.index = len(self._model_items) - 1
            models.focus()
            self._sync_selected_model()
            return
        self._focused_list().action_cursor_up()
        self._sync_selected_model()

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Move down within the focused list; from the last model jump to the first
        # action and focus the action menu. Syncs the selected model.
        models = self.query_one("#local-model-list", ListView)
        actions = self.query_one("#local-action-menu", ListView)
        if models.has_focus and self._action_items and models.index == len(self._model_items) - 1:
            actions.index = 0
            actions.focus()
            return
        self._focused_list().action_cursor_down()
        self._sync_selected_model()

    def action_select_cursor(self) -> None:
        # [FUNCTION DOC]
        # Select the highlighted item in the focused list.
        self._focused_list().action_select_cursor()

    def action_toggle_section(self) -> None:
        # [FUNCTION DOC]
        # Switch focus between models and actions (when the target has items) and
        # update section styling.
        models = self.query_one("#local-model-list", ListView)
        actions = self.query_one("#local-action-menu", ListView)
        if models.has_focus and self._action_items:
            actions.focus()
        elif self._model_items:
            models.focus()
        self._update_section_focus()

    def _update_section_focus(self) -> None:
        # [FUNCTION DOC]
        # HELPER: dim the unfocused list via a CSS class and mark the focused section
        # title with "focused".
        models = self.query_one("#local-model-list", ListView)
        actions = self.query_one("#local-action-menu", ListView)
        models_focused = models.has_focus
        models.set_class(not models_focused, "local-section-inactive")
        actions.set_class(models_focused, "local-section-inactive")
        self.query_one("#local-model-section-title", Static).update(
            "Models — focused" if models_focused else "Models"
        )
        self.query_one("#local-action-section-title", Static).update(
            "Actions" if models_focused else "Actions — focused"
        )

    def _focused_list(self) -> ListView:
        # [FUNCTION DOC]
        # HELPER: return the model list if focused, else the action menu.
        models = self.query_one("#local-model-list", ListView)
        if models.has_focus:
            return models
        return self.query_one("#local-action-menu", ListView)

    def _activate_action(self, index: int) -> None:
        # [FUNCTION DOC]
        # HELPER: dispatch the action menu item at the index: configure form, refresh,
        # doctor, reset confirmation, download/search input prompt, or cancel-download
        # confirmation. Out-of-range indexes are ignored.
        if not 0 <= index < len(self._action_items):
            return
        token = self._action_items[index][0]
        if token == "configure":
            self._open_configure()
        elif token == "refresh":
            self._start_operation("refresh")
        elif token == "doctor":
            self._start_operation("doctor")
        elif token == "reset":
            self._confirm_reset()
        elif token in {"download_model", "search_models"}:
            self._open_model_action(cast(LocalAction, token))
        elif token == "cancel_download":
            self._confirm_cancel_download()

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismiss the modal; cleanup happens in on_unmount.
        # on_unmount owns cancellation; let Textual begin the screen pop first.
        self.dismiss(None)

    def _open_configure(self) -> None:
        # [FUNCTION DOC]
        # HELPER: cancel a running worker (and the registry refresh) so the user need
        # not wait, read the backend's configure spec (errors become messages) and push
        # the configuration form with a callback.
        # KNOWN ISSUE: it cancels the worker whatever action it runs, while only asking
        # the registry to cancel "refresh".
        # Do not make users wait for an unavailable default probe before they
        # can enter a custom endpoint.
        if self._worker is not None and not self._worker.done():
            self.registry.cancel(self.backend_id, "refresh")
            self._worker.cancel()
        view = self.registry.effective(self.backend_id)
        if view is None:
            self._show_message("This backend is no longer available.", "error")
            return
        try:
            spec = view.backend.read_configure_spec()
        except Exception:  # noqa: BLE001 - a backend must not crash the host
            self._show_message("Could not read the backend configuration.", "error")
            return
        self.app.push_screen(
            LocalConfigureScreen(spec, theme=self.theme),
            callback=self._handle_configuration,
        )

    def _handle_configuration(self, values: LocalConfigValues | None) -> None:
        # [FUNCTION DOC]
        # HELPER: form callback; start a configure operation with the values unless cancelled.
        if values is None:
            return
        self._start_operation("configure", values=values)

    def _confirm_reset(self) -> None:
        # [FUNCTION DOC]
        # HELPER: require idle, then push a yes/no confirmation for resetting settings.
        if not self._is_idle():
            self._show_message(
                "Tau must be idle before resetting local backend settings.",
                "warning",
            )
            return
        self.app.push_screen(
            LocalConfirmScreen(
                "Reset local backend?",
                "Remove this backend's saved integration settings? Stored credentials "
                "require a separate confirmation.",
                theme=self.theme,
            ),
            callback=self._handle_reset_confirmation,
        )

    def _handle_reset_confirmation(self, confirmed: bool | None) -> None:
        # [FUNCTION DOC]
        # HELPER: start the reset operation when confirmed.
        if confirmed:
            self._start_operation("reset")

    def _open_model_action(self, action: LocalAction) -> None:
        # [FUNCTION DOC]
        # HELPER: require idle, then push an input prompt titled for download or search
        # with a matching placeholder; its result starts the operation. Other actions
        # show a warning (unreachable from current callers).
        if not self._is_idle():
            self._show_message(
                "Tau must be idle before changing local backend models.",
                "warning",
            )
            return
        labels = {
            "download_model": "Download model",
            "search_models": "Search Hugging Face models",
        }
        placeholders = {
            "download_model": "owner/repository[:quantization]",
            "search_models": "Hugging Face model ID or search query",
        }
        if action not in labels:
            self._show_message("Select a model from the model list first.", "warning")
            return
        self.app.push_screen(
            LocalModelActionScreen(
                labels[action],
                placeholder=placeholders[action],
                theme=self.theme,
            ),
            callback=lambda model_id: self._handle_model_action(action, model_id),
        )

    def _handle_model_action(self, action: LocalAction, model_id: str | None) -> None:
        # [FUNCTION DOC]
        # HELPER: start the action with the entered model id unless cancelled.
        if model_id is not None:
            self._start_operation(action, model_id=model_id)

    def _activate_selected_model(self) -> None:
        # [FUNCTION DOC]
        # HELPER: for the selected model: loaded/sleeping/available/unknown-state models
        # open the use/unload chooser; loading, downloading or "unknown" states warn;
        # otherwise load it if the backend supports loading, else warn.
        model = self._selected_model()
        if model is None or self.status is None:
            return
        if model.state in {"loaded", "sleeping", "available", None}:
            self._choose_loaded_model_action(model.id)
            return
        if model.state in {"loading", "downloading", "unknown"}:
            self._show_message(f"{model.id} is currently {model.state}.", "warning")
            return
        if "load_model" in self.status.actions:
            self._start_operation("load_model", model_id=model.id)
        else:
            self._show_message(f"{model.id} is not currently available to use.", "warning")

    def _selected_model(self) -> LocalModel | None:
        # [FUNCTION DOC]
        # HELPER: return the status model matching the selected id, or nothing.
        if self.status is None or self._selected_model_id is None:
            return None
        return next(
            (model for model in self.status.models if model.id == self._selected_model_id),
            None,
        )

    def _choose_loaded_model_action(self, model_id: str) -> None:
        # [FUNCTION DOC]
        # HELPER: push a choice confirmation (use recommended, unload, cancel) for a model.
        request = LocalConfirmationRequest(
            f"Choose an action for {model_id!r}.",
            (
                LocalConfirmationChoice("use", "Use model", True),
                LocalConfirmationChoice("unload", "Unload model"),
                LocalConfirmationChoice("cancel", "Cancel"),
            ),
        )
        self.app.push_screen(
            LocalChoiceConfirmScreen(request, theme=self.theme),
            callback=lambda choice: self._handle_loaded_model_action(model_id, choice),
        )

    def _handle_loaded_model_action(self, model_id: str, choice: str | None) -> None:
        # [FUNCTION DOC]
        # HELPER: "use" switches to the model; "unload" starts an unload operation.
        if choice == "use":
            self._use_model(model_id)
        elif choice == "unload":
            self._start_operation("unload_model", model_id=model_id)

    def _sync_selected_model(self) -> None:
        # [FUNCTION DOC]
        # HELPER: set the selected model id from the model list index when valid.
        models = self.query_one("#local-model-list", ListView)
        index = models.index
        if index is not None and 0 <= index < len(self._model_items):
            self._selected_model_id = self._model_items[index]

    def _start_operation(
        self,
        action: LocalAction,
        *,
        values: Mapping[str, str] | LocalConfigValues | None = None,
        model_id: str | None = None,
        confirmation: str | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: refuse (with a warning) when not idle or a worker is running;
        # otherwise mark the active action and start the operation task.
        if not self._is_idle():
            self._show_message(
                "Tau must be idle before changing local backend settings.",
                "warning",
            )
            return
        if self._worker is not None and not self._worker.done():
            self._show_message("An operation is already in progress.", "warning")
            return
        self._active_action = action
        self._worker = asyncio.create_task(
            self._run_operation(action, values, model_id, confirmation)
        )

    def _confirm_cancel_download(self) -> None:
        # [FUNCTION DOC]
        # HELPER: push a keep/cancel confirmation for the server-side download.
        request = LocalConfirmationRequest(
            "Cancel the active server-side download? Already transferred data may be discarded.",
            (
                LocalConfirmationChoice("keep", "Keep downloading"),
                LocalConfirmationChoice("cancel_download", "Cancel download"),
            ),
        )
        self.app.push_screen(
            LocalChoiceConfirmScreen(request, theme=self.theme),
            callback=self._cancel_download,
        )

    def _cancel_download(self, choice: str | None) -> None:
        # [FUNCTION DOC]
        # HELPER: on "cancel_download", ask the registry to cancel the download and
        # report whether one was found.
        if choice != "cancel_download":
            return
        if self.registry.cancel(self.backend_id, "download_model"):
            self._show_message("Download cancellation requested.", "warning")
        else:
            self._show_message("No active download was found.", "warning")

    async def _run_operation(
        self,
        action: LocalAction,
        values: Mapping[str, str] | LocalConfigValues | None,
        model_id: str | None,
        confirmation: str | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: run one registry operation. Shows "Working" progress unless a refresh
        # must preserve observed download progress; re-renders sections before a
        # confirmed download. Calls the matching registry method with a progress
        # callback (assertions require values/model id where needed). Cancellation
        # returns silently; other exceptions show only the exception type name. Then,
        # if the UI is usable: stale or cancelled results warn; a new status is stored
        # and rendered; a confirmation request opens a choice screen that re-runs the
        # action with the choice; field errors, messages, search results (opening the
        # results screen), diagnostics and orphaned credentials are reported.
        self._active_action = action
        preserve_download_progress = action == "refresh" and self._progress_unsubscribe is not None
        if not preserve_download_progress:
            self._progress_fraction = None
            self._set_progress("Working…", show_bar=action == "download_model")
        if action == "download_model" and confirmation == "download":
            await self._render_sections(self.status)

        def progress(item: LocalProgress) -> None:
            # [FUNCTION DOC]
            # HELPER: update the progress line and fraction unless preserving download progress.
            if not preserve_download_progress:
                self._set_progress(
                    item.message,
                    fraction=item.fraction,
                    show_bar=action == "download_model",
                )

        try:
            if action == "configure":
                assert values is not None
                result = await self.registry.configure(
                    self.backend_id,
                    values,
                    progress=cast(ProgressCallback, progress),
                )
            elif action == "refresh":
                result = await self.registry.refresh(
                    self.backend_id,
                    progress=cast(ProgressCallback, progress),
                )
            elif action == "doctor":
                result = await self.registry.doctor(
                    self.backend_id,
                    progress=cast(ProgressCallback, progress),
                )
            elif action == "reset":
                result = await self.registry.reset(
                    self.backend_id,
                    progress=cast(ProgressCallback, progress),
                )
            elif action in {"load_model", "unload_model", "download_model"}:
                assert model_id is not None
                manage_action = cast(
                    Literal["load_model", "unload_model", "download_model"], action
                )
                result = await self.registry.manage_model(
                    self.backend_id,
                    manage_action,
                    model_id,
                    progress=cast(ProgressCallback, progress),
                    confirmation=confirmation,
                )
            elif action == "search_models":
                assert model_id is not None
                result = await self.registry.search_models(
                    self.backend_id,
                    model_id,
                    progress=cast(ProgressCallback, progress),
                )
            else:
                result = LocalOperationResult(message="Unsupported action.")
        except asyncio.CancelledError:
            self._active_action = None
            return
        except Exception as exc:  # noqa: BLE001 - host keeps modal alive
            self._active_action = None
            if self._can_update_ui:
                self._show_message(f"Could not complete action: {type(exc).__name__}", "error")
            return
        self._active_action = None
        if not self._can_update_ui:
            return
        if result.stale:
            self._show_message("The backend changed while this action was running.", "warning")
            return
        if result.cancelled:
            self._show_message("Action cancelled.", "warning")
            return
        if result.backend_status is not None:
            self.status = result.backend_status
            await self._render_status(result.backend_status)
        if result.confirmation is not None:
            self._open_operation_confirmation(result.confirmation, action, values, model_id)
            self._set_progress("")
            return
        if result.field_errors:
            self._show_message(
                "Configuration was not saved. Review the fields and try again.",
                "error",
            )
        elif result.message:
            self._show_message(result.message, "info")
        elif action == "refresh" and not preserve_download_progress:
            self._set_progress("")
        if result.search_results:
            self._set_progress("Select a model and quantization to download.")
            self.app.push_screen(
                LocalSearchResultsScreen(result.search_results, theme=self.theme),
                callback=self._download_search_result,
            )
        for diagnostic in result.diagnostics:
            message = (
                f"{diagnostic.stage}: {diagnostic.message}"
                if diagnostic.stage
                else diagnostic.message
            )
            self._show_message(message, diagnostic.severity)
        if result.credential_orphaned:
            self._set_progress("Credential cleanup needs attention.")

    def _download_search_result(self, model_id: str | None) -> None:
        # [FUNCTION DOC]
        # HELPER: start a download for the chosen search variant id.
        if model_id is not None:
            self._start_operation("download_model", model_id=model_id)

    def _open_operation_confirmation(
        self,
        request: LocalConfirmationRequest,
        action: LocalAction,
        values: Mapping[str, str] | LocalConfigValues | None,
        model_id: str | None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: push a choice confirmation for a backend request; the choice resumes
        # the operation.
        self.app.push_screen(
            LocalChoiceConfirmScreen(request, theme=self.theme),
            callback=lambda choice: self._resume_confirmed_operation(
                choice, action, values, model_id
            ),
        )

    def _resume_confirmed_operation(
        self,
        choice: str | None,
        action: LocalAction,
        values: Mapping[str, str] | LocalConfigValues | None,
        model_id: str | None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: restart the same operation with the chosen confirmation value unless cancelled.
        if choice is not None:
            self._start_operation(action, values=values, model_id=model_id, confirmation=choice)

    def _use_selected(self) -> None:
        # [FUNCTION DOC]
        # HELPER: use the selected model (or the status's selected model), warning when
        # none or when not loaded.
        # KNOWN ISSUE: never called anywhere in this module (dead code).
        model = self._selected_model()
        model_id = (
            model.id if model is not None else self.status.selected_model if self.status else None
        )
        if model_id is None:
            self._show_message(
                "Refresh this backend and select an available model first.",
                "warning",
            )
            return
        if model is not None and model.state not in {"loaded", "sleeping", "available", None}:
            self._show_message(f"{model.id} must be loaded before it can be used.", "warning")
            return
        self._use_model(model_id)

    def _use_model(self, model_id: str) -> None:
        # [FUNCTION DOC]
        # HELPER: guard idle state, running worker, running use task, missing host
        # callback, and backend availability; then call the host use callback with the
        # backend's provider id and model id, awaiting an async result in a task.
        if not self._is_idle():
            self._show_message("Tau must be idle before switching models.", "warning")
            return
        if self._worker is not None and not self._worker.done():
            self._show_message("An operation is already in progress.", "warning")
            return
        if self._use_task is not None and not self._use_task.done():
            self._show_message("A model switch is already in progress.", "warning")
            return
        if self.on_use is None:
            self._show_message("Model selection is unavailable in this host.", "warning")
            return
        view = self.registry.effective(self.backend_id)
        if view is None or not view.use_available:
            self._show_message(
                "Using this backend is unavailable while it is shadowed.",
                "warning",
            )
            return
        result = self.on_use(view.backend.provider_id, model_id)
        if result is not None:
            self._use_task = asyncio.create_task(self._await_use(result))

    async def _await_use(self, result: Awaitable[None]) -> None:
        # [FUNCTION DOC]
        # HELPER: await the host's model switch; cancellation is ignored and other
        # errors produce a generic error message.
        try:
            await result
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001 - keep host modal alive
            if self._can_update_ui:
                self._show_message("Could not switch to the selected model.", "error")

    async def _render_status(self, status: LocalBackendStatus) -> None:
        # [FUNCTION DOC]
        # HELPER: write status text (state, endpoint, authentication, no-models note,
        # selected model, cached/stale notes, diagnostics) and re-render the sections.
        lines = [f"State: {status.state}"]
        if status.endpoint_display:
            lines.append(f"Endpoint: {status.endpoint_display}")
        lines.append(f"Authentication: {status.authentication_source}")
        if not status.models:
            lines.append("Models: none discovered")
        if status.selected_model:
            lines.append(f"Selected: {status.selected_model}")
        if status.cached:
            lines.append("Using cached results.")
        if status.stale:
            lines.append("Results may be stale.")
        for diagnostic in status.diagnostics:
            lines.append(
                f"{diagnostic.stage}: {diagnostic.message}"
                if diagnostic.stage
                else diagnostic.message
            )
        self.query_one("#local-backend-status", Static).update("\n".join(lines))
        await self._render_sections(status)

    async def _render_sections(self, status: LocalBackendStatus | None) -> None:
        # [FUNCTION DOC]
        # HELPER: rebuild the model list (labels, visibility, preferred selection) and
        # the action menu (known actions supported by the status, or configure and
        # refresh without status, with "Cancel active download" first while a download
        # runs), restoring the prior action cursor and choosing focus sensibly.
        model_list = self.query_one("#local-model-list", ListView)
        action_menu = self.query_one("#local-action-menu", ListView)
        action_focused = action_menu.has_focus
        had_models = bool(self._model_items)
        prior_action = None
        if action_menu.index is not None and 0 <= action_menu.index < len(self._action_items):
            prior_action = self._action_items[action_menu.index][0]

        actions = set(status.actions) if status is not None else {"configure", "refresh"}
        models = status.models if status is not None else ()
        selected_model = status.selected_model if status is not None else None
        self._model_items = tuple(model.id for model in models)
        await model_list.clear()
        await model_list.extend(
            ListItem(Label(_model_label(model, selected_model), markup=False)) for model in models
        )
        model_list.styles.display = "block" if models else "none"
        if models:
            preferred_model = self._selected_model_id or selected_model or models[0].id
            model_list.index = next(
                (index for index, model in enumerate(models) if model.id == preferred_model),
                0,
            )
            self._sync_selected_model()
        else:
            model_list.index = None
            self._selected_model_id = None

        action_labels = (
            ("search_models", "Search Hugging Face models…"),
            ("download_model", "Download an exact Hugging Face model…"),
            ("configure", "Configure connection…"),
            ("refresh", "Refresh server state"),
            ("doctor", "Run Doctor"),
            ("reset", "Reset integration settings…"),
        )
        self._action_items = tuple(
            (action, label) for action, label in action_labels if action in actions
        )
        if self._active_action == "download_model" or self.registry.operation_running(
            self.backend_id, "download_model"
        ):
            self._action_items = (
                ("cancel_download", "Cancel active download…"),
                *self._action_items,
            )
        await action_menu.clear()
        await action_menu.extend(
            ListItem(Label(label, markup=False)) for _, label in self._action_items
        )
        action_menu.styles.display = "block" if self._action_items else "none"
        if self._action_items:
            action_menu.index = next(
                (
                    index
                    for index, (token, _) in enumerate(self._action_items)
                    if token == prior_action
                ),
                0,
            )
        else:
            action_menu.index = None

        if models and (not had_models or not action_focused):
            model_list.focus()
        elif self._action_items:
            action_menu.focus()
        elif models:
            model_list.focus()

    @property
    def _can_update_ui(self) -> bool:
        # [FUNCTION DOC]
        # HELPER property: true when not closing and the screen is mounted, attached
        # and current.
        return not self._closing and self.is_mounted and self.is_attached and self.is_current

    def _set_progress(
        self,
        message: str,
        *,
        fraction: float | None = None,
        show_bar: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: update the progress line and bar if the UI is usable. Ignores
        # indeterminate bar updates when byte progress is known; stores or clears the
        # fraction; shows/hides the message and bar; sets an indeterminate or fractional
        # bar. Missing widgets are tolerated.
        if not self._can_update_ui:
            return
        if show_bar and fraction is None and self._progress_fraction is not None:
            # Catalog polling only knows that a download is active. Do not let
            # that coarser update erase newer byte progress from the SSE stream.
            return
        if fraction is not None:
            self._progress_fraction = fraction
        elif not show_bar:
            self._progress_fraction = None
        with suppress(NoMatches):
            progress_message = self.query_one("#local-backend-progress", Static)
            progress_message.update(message)
            progress_message.styles.display = "block" if message else "none"
            progress_bar = self.query_one("#local-backend-progress-bar", ProgressBar)
            progress_bar.styles.display = "block" if show_bar or fraction is not None else "none"
            if fraction is None:
                progress_bar.update(total=None, progress=0)
            else:
                progress_bar.update(total=1, progress=fraction)

    def _show_message(self, message: str, level: str) -> None:
        # [FUNCTION DOC]
        # HELPER: when the UI is usable, forward the message to the host notify callback
        # and show it on the progress line.
        if not self._can_update_ui:
            return
        self._notify_callback(message, level)
        self._set_progress(message)


class LocalConfigureScreen(ModalScreen[LocalConfigValues | None]):
    # [CLASS DOC]
    # Modal form generated from a backend configure spec: choice fields as selects,
    # text and secret fields as inputs (secrets masked). Dismisses with the values
    # and the set of secret keys, or nothing on cancel.
    """Render arbitrary text, secret, and choice fields without backend UI code."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save", show=False),
    ]

    def __init__(self, spec: LocalConfigureSpec, *, theme: TuiTheme) -> None:
        # [FUNCTION DOC]
        # Store the spec and theme and assign a widget id to each field key.
        super().__init__()
        self.spec = spec
        self.theme = theme
        self._field_ids = {
            field.key: f"local-config-input-{index}" for index, field in enumerate(spec.fields)
        }

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yield the title, a label and widget per field, and the footer.
        with Vertical(id="local-configure-screen"):
            yield Static("Configure local backend", id="local-configure-title")
            for field in self.spec.fields:
                field_id = self._field_ids[field.key]
                yield Label(field.label, id=f"local-config-label-{field_id}")
                if field.kind == "choice":
                    yield Select(
                        [(choice, choice) for choice in field.choices],
                        allow_blank=not field.required,
                        id=field_id,
                    )
                else:
                    yield Input(
                        placeholder=field.placeholder or "",
                        password=field.kind == "secret",
                        id=field_id,
                    )
            yield Static(
                "Enter advances/saves - Ctrl+S saves - Escape cancels",
                id="local-configure-footer",
            )

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Focus the first field when any exist.
        if self.spec.fields:
            self.query_one(f"#{self._field_ids[self.spec.fields[0].key]}").focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # [FUNCTION DOC]
        # For form inputs, Enter advances focus to the next field or saves on the last.
        field_ids = tuple(self._field_ids[field.key] for field in self.spec.fields)
        if event.input.id not in field_ids:
            return
        event.stop()
        index = field_ids.index(event.input.id)
        if index == len(field_ids) - 1:
            self.action_save()
        else:
            self.query_one(f"#{field_ids[index + 1]}").focus()

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismiss with nothing.
        self.dismiss(None)

    def action_save(self) -> None:
        # [FUNCTION DOC]
        # Collect each field's value (non-string select values become empty strings),
        # note secret keys, and dismiss with the configuration values.
        values: dict[str, str] = {}
        secret_keys: set[str] = set()
        for field in self.spec.fields:
            widget = self.query_one(f"#{self._field_ids[field.key]}")
            if field.kind == "choice":
                selected = cast(Select[str], widget).value
                value = selected if isinstance(selected, str) else ""
            else:
                value = cast(Input, widget).value
            values[field.key] = value
            if field.kind == "secret":
                secret_keys.add(field.key)
        self.dismiss(LocalConfigValues(values, secret_keys=frozenset(secret_keys)))


class LocalConfirmScreen(ModalScreen[bool | None]):
    # [CLASS DOC]
    # Yes/No modal for destructive actions, defaulting to No; dismisses with true,
    # false, or nothing on cancel.
    """Small generic confirmation used for destructive backend actions."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "select_cursor", "Select", show=False),
    ]

    def __init__(self, title: str, message: str, *, theme: TuiTheme) -> None:
        # [FUNCTION DOC]
        # Store the title, message and theme.
        super().__init__()
        self.title_text = title
        self.message = message
        self.theme = theme

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yield title, message, the Yes/No list and footer (markup disabled).
        with Vertical(id="local-confirm-screen"):
            yield Static(self.title_text, id="local-confirm-title", markup=False)
            yield Static(self.message, id="local-confirm-message", markup=False)
            yield ListView(
                ListItem(Label("Yes", markup=False)),
                ListItem(Label("No", markup=False)),
                id="local-confirm-list",
            )
            yield Static(
                "↑/↓ navigate - Enter selects - Escape cancels",
                id="local-confirm-footer",
            )

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Put the cursor on "No" and focus the list.
        choices = self.query_one("#local-confirm-list", ListView)
        choices.index = 1
        choices.focus()

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Intercept up, down and enter keys (stopping propagation) and route them to the
        # cursor-up, cursor-down and select-cursor actions so the focused list handles them.
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
        # Dismiss with whether "Yes" (index 0) was selected.
        event.stop()
        self.dismiss(event.index == 0)

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Move the list cursor up.
        self.query_one("#local-confirm-list", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Move the list cursor down.
        self.query_one("#local-confirm-list", ListView).action_cursor_down()

    def action_select_cursor(self) -> None:
        # [FUNCTION DOC]
        # Select the highlighted item.
        self.query_one("#local-confirm-list", ListView).action_select_cursor()

    def action_confirm(self) -> None:
        # [FUNCTION DOC]
        # Dismiss with true. KNOWN ISSUE: no binding references this action (unused).
        self.dismiss(True)

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismiss with nothing.
        self.dismiss(None)


class LocalChoiceConfirmScreen(ModalScreen[str | None]):
    # [CLASS DOC]
    # Modal rendering a backend confirmation request's message and choices
    # (recommended marked); dismisses with the chosen value or nothing.
    """Render an arbitrary backend confirmation without protocol knowledge."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "select_cursor", "Select", show=False),
    ]

    def __init__(self, request: LocalConfirmationRequest, *, theme: TuiTheme) -> None:
        # [FUNCTION DOC]
        # Store the request and theme.
        super().__init__()
        self.request = request
        self.theme = theme

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yield title, message, choice list and footer.
        with Vertical(id="local-confirm-screen"):
            yield Static("Confirm backend action", id="local-confirm-title", markup=False)
            yield Static(self.request.message, id="local-confirm-message", markup=False)
            yield ListView(
                *[
                    ListItem(
                        Label(
                            f"{choice.label} — recommended" if choice.recommended else choice.label,
                            markup=False,
                        )
                    )
                    for choice in self.request.choices
                ],
                id="local-choice-list",
            )
            yield Static(
                "↑/↓ navigate - Enter selects - Escape cancels",
                id="local-confirm-footer",
            )

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Place the cursor on the recommended choice, else a "cancel" choice, else the
        # first, and focus the list.
        choices = self.query_one("#local-choice-list", ListView)
        choices.index = next(
            (index for index, choice in enumerate(self.request.choices) if choice.recommended),
            next(
                (
                    index
                    for index, choice in enumerate(self.request.choices)
                    if choice.value == "cancel"
                ),
                0,
            ),
        )
        choices.focus()

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Intercept up, down and enter keys (stopping propagation) and route them to the
        # cursor-up, cursor-down and select-cursor actions so the focused list handles them.
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
        # Dismiss with the selected choice's value.
        event.stop()
        self.dismiss(self.request.choices[event.index].value)

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Move the list cursor up.
        self.query_one("#local-choice-list", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Move the list cursor down.
        self.query_one("#local-choice-list", ListView).action_cursor_down()

    def action_select_cursor(self) -> None:
        # [FUNCTION DOC]
        # Select the highlighted choice.
        self.query_one("#local-choice-list", ListView).action_select_cursor()

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismiss with nothing.
        self.dismiss(None)


class LocalSearchResultsScreen(ModalScreen[str | None]):
    # [CLASS DOC]
    # Modal listing downloadable variants flattened from search results; dismisses
    # with the chosen variant id or nothing.
    """Choose one backend-provided artifact variant for download."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "confirm", "Download", show=False),
    ]

    def __init__(
        self,
        results: tuple[LocalSearchResult, ...],
        *,
        theme: TuiTheme,
    ) -> None:
        # [FUNCTION DOC]
        # Store results and theme and flatten them into (id, label, recommended) options.
        super().__init__()
        self.results = results
        self.theme = theme
        self.options = tuple(
            option for result in results for option in _search_result_options(result)
        )

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yield title, help, the options list and footer.
        with Vertical(id="local-search-results-screen"):
            yield Static("Download model", id="local-search-results-title", markup=False)
            yield Static(
                "Choose a Hugging Face model variant. llama.cpp performs the download.",
                id="local-search-results-help",
                markup=False,
            )
            yield ListView(
                *[ListItem(Label(label, markup=False)) for _, label, _ in self.options],
                id="local-search-results-list",
            )
            yield Static(
                "↑/↓ navigate - Enter continues - Escape cancels",
                id="local-search-results-footer",
            )

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # If options exist, put the cursor on the first recommended one (else first) and focus.
        if not self.options:
            return
        index = next(
            (index for index, (_, _, recommended) in enumerate(self.options) if recommended),
            0,
        )
        model_list = self.query_one("#local-search-results-list", ListView)
        model_list.index = index
        model_list.focus()

    def on_key(self, event: Key) -> None:
        # [FUNCTION DOC]
        # Intercept up, down and enter keys (stopping propagation) and route them to the
        # cursor-up, cursor-down and select-cursor actions so the focused list handles them.
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
        # Confirm the highlighted option.
        event.stop()
        self.action_confirm()

    def action_cursor_up(self) -> None:
        # [FUNCTION DOC]
        # Move the list cursor up.
        self.query_one("#local-search-results-list", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        # [FUNCTION DOC]
        # Move the list cursor down.
        self.query_one("#local-search-results-list", ListView).action_cursor_down()

    def action_select_cursor(self) -> None:
        # [FUNCTION DOC]
        # Select the highlighted option.
        self.query_one("#local-search-results-list", ListView).action_select_cursor()

    def action_confirm(self) -> None:
        # [FUNCTION DOC]
        # Dismiss with the highlighted option's id when valid; no-op without options.
        if not self.options:
            return
        index = self.query_one("#local-search-results-list", ListView).index
        if index is not None and 0 <= index < len(self.options):
            self.dismiss(self.options[index][0])

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismiss with nothing.
        self.dismiss(None)


class LocalModelActionScreen(ModalScreen[str | None]):
    # [CLASS DOC]
    # Modal with one text input collecting an opaque model reference; dismisses with
    # the stripped non-empty value or nothing on cancel.
    """Collect one opaque model reference for a backend-provided action."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        title: str,
        *,
        placeholder: str = "Model identifier",
        theme: TuiTheme,
    ) -> None:
        # [FUNCTION DOC]
        # Store title, placeholder and theme.
        super().__init__()
        self.title_text = title
        self.placeholder = placeholder
        self.theme = theme

    def compose(self) -> ComposeResult:
        # [FUNCTION DOC]
        # Yield the title, input and footer.
        with Vertical(id="local-model-action-screen"):
            yield Static(self.title_text, id="local-model-action-title", markup=False)
            yield Input(placeholder=self.placeholder, id="local-model-action-input")
            yield Static(
                "Enter continues - Escape cancels",
                id="local-model-action-footer",
            )

    def on_mount(self) -> None:
        # [FUNCTION DOC]
        # Focus the input.
        self.query_one("#local-model-action-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # [FUNCTION DOC]
        # Stop the event and submit.
        event.stop()
        self._submit()

    def _submit(self) -> None:
        # [FUNCTION DOC]
        # HELPER: dismiss with the stripped value if non-empty; otherwise stay open.
        value = self.query_one("#local-model-action-input", Input).value.strip()
        if value:
            self.dismiss(value)

    def action_cancel(self) -> None:
        # [FUNCTION DOC]
        # Dismiss with nothing.
        self.dismiss(None)


def _model_label(model: LocalModel, selected_model: str | None) -> str:
    # [FUNCTION DOC]
    # HELPER: model display name (with id in parentheses when different) followed by
    # details: its state ("unloaded" shown as "available to load") and "active" if
    # it is the selected model.
    label = model.display_name or model.id
    if label != model.id:
        label = f"{label} ({model.id})"
    details = []
    if model.state:
        details.append("available to load" if model.state == "unloaded" else model.state)
    if model.id == selected_model:
        details.append("active")
    return label + (" — " + " · ".join(details) if details else "")


def _search_result_options(
    result: LocalSearchResult,
) -> tuple[tuple[str, str, bool], ...]:
    # [FUNCTION DOC]
    # HELPER: flatten one search result into (id, label, recommended) options: a
    # single non-recommended entry when it has no variants, else one per variant
    # labeled with variant label, size, recommended and restricted markers.
    if not result.options:
        restricted = " — restricted" if result.restricted else ""
        return ((result.id, result.label + restricted, False),)
    options: list[tuple[str, str, bool]] = []
    for option in result.options:
        details = [option.label]
        if option.size_bytes is not None:
            details.append(_format_bytes(option.size_bytes))
        if option.recommended:
            details.append("recommended")
        if result.restricted:
            details.append("restricted")
        options.append((option.id, f"{result.label} — " + " · ".join(details), option.recommended))
    return tuple(options)


def _format_bytes(value: int) -> str:
    # [FUNCTION DOC]
    # HELPER: human-readable binary size: whole bytes below 1 KiB, else one decimal
    # in KiB, MiB, GiB or TiB.
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{value} B"


__all__ = [
    "LocalBackendPickerScreen",
    "LocalBackendScreen",
    "LocalConfigureScreen",
    "LocalChoiceConfirmScreen",
    "LocalConfirmScreen",
    "LocalModelActionScreen",
    "LocalSearchResultsScreen",
]
