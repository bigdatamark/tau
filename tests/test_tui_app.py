# [MODULE DOC] Test suite for the Textual TUI frontend of tau_coding (tau_coding.tui): the TauTuiApp
# application, its modal screens (pickers, login, editors, extension dialogs), sidebar and
# compact session-info renderers, transcript widgets and windowing, chat-item rendering,
# TuiState projection, the TuiEventAdapter, theming, keybindings, autocomplete, terminal
# title/notification controllers and herdr-terminal mouse configuration.
# Tests drive the app headlessly via Textual's run_test pilot against a FakeSession stand-in
# for the real coding session, so no provider or network access is needed. Tests mutate
# environment variables and home directories via monkeypatch/isolate_home; no other I/O
# beyond temporary directories. Failures surface as assertion errors.
import asyncio
import re
import threading
from collections.abc import AsyncIterator
from datetime import datetime
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO

import pytest
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from textual import events
from textual.color import Color
from textual.containers import Container, VerticalScroll
from textual.content import Content
from textual.content import Style as TextualStyle
from textual.geometry import Offset
from textual.selection import SELECT_ALL, Selection
from textual.widgets import (
    Collapsible,
    Input,
    Label,
    ListItem,
    ListView,
    OptionList,
    Static,
    TextArea,
)
from textual.widgets import Markdown as TextualMarkdown
from textual.widgets.markdown import MarkdownStream

from conftest import isolate_home
from tau_agent import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AgentToolResult,
    AssistantMessage,
    CustomMessage,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolResultMessage,
    UserMessage,
)
from tau_agent.messages import assistant_content
from tau_agent.provider_events import TextDeltaEvent, ThinkingDeltaEvent
from tau_coding.catalog_loader import user_catalog_path
from tau_coding.commands import CommandResult
from tau_coding.credentials import FileCredentialStore, OAuthCredential
from tau_coding.events import (
    AgentSettledEvent,
    CodingSessionEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    QueueUpdateEvent,
    SessionAgentEndEvent,
)
from tau_coding.extensions import (
    DynamicProvider,
    DynamicProviderRegistry,
    LocalBackend,
    LocalBackendRegistry,
    LocalBackendStatus,
    LocalConfigureResult,
    LocalConfigureSpec,
    LocalModel,
    NoAuth,
    OpenAICompatibleTransport,
    ProviderModel,
)
from tau_coding.paths import TauPaths
from tau_coding.prompt_templates import PromptTemplate
from tau_coding.provider_config import (
    OpenAICodexProviderConfig,
    OpenAICompatibleProviderConfig,
    ProviderSelection,
    ProviderSettings,
    ScopedModelConfig,
    save_provider_settings,
)
from tau_coding.resources import ResourceDiagnostic
from tau_coding.session import (
    ModelChoice,
    SessionTreeBranchResult,
    SessionTreeChoice,
    TerminalCommandResult,
)
from tau_coding.session_manager import CodingSessionRecord
from tau_coding.session_stats import SessionStats
from tau_coding.skills import Skill, format_skill_invocation
from tau_coding.system_prompt import (
    ProjectContextFile,
    SystemPromptInspection,
    SystemPromptSource,
    format_system_prompt_inspection,
)
from tau_coding.tools import create_coding_tools
from tau_coding.tui import app as tui_app
from tau_coding.tui.adapter import TuiEventAdapter
from tau_coding.tui.app import (
    COMPLETION_MAX_VISIBLE_LINES,
    PASTE_DISPLAY_THRESHOLD,
    RESERVED_EXTENSION_INTERCEPTOR_KEYS,
    CommandOutputScreen,
    CustomProviderLoginResult,
    CustomProviderLoginScreen,
    ExtensionConfirmScreen,
    ExtensionInputScreen,
    ExtensionSelectScreen,
    LoginMethodPickerScreen,
    LoginProviderPickerScreen,
    LoginScreen,
    ModelPickerScreen,
    OAuthLoginScreen,
    PromptInput,
    PromptTemplateEditorScreen,
    PromptTemplatePickerScreen,
    SessionPickerScreen,
    SidebarFileEditor,
    SkillPickerScreen,
    TauTuiApp,
    ThemePickerScreen,
    ToolsReferenceScreen,
    TreePickerScreen,
    _activity_prompt_border_color,
    _completion_selected_render_line,
    _render_activity_indicator,
    _resource_conflict_alert,
    _session_picker_label,
    _session_records,
    _short_path,
    _terminal_command_prefix_span,
    _textual_theme_for_tau_theme,
    _theme_css_variables,
    _TuiExtensionUiBridge,
    _visible_completion_state,
)
from tau_coding.tui.autocomplete import CompletionItem, CompletionKind, CompletionState
from tau_coding.tui.config import (
    HIGH_CONTRAST_THEME,
    TAU_DARK_THEME,
    TAU_LIGHT_THEME,
    TuiKeybindings,
    TuiSettings,
    TuiTheme,
    tui_settings_path,
)
from tau_coding.tui.local_backends import LocalBackendScreen, LocalConfirmScreen
from tau_coding.tui.state import ChatItem, TuiState
from tau_coding.tui.terminal_notification import TerminalNotificationController
from tau_coding.tui.terminal_title import TerminalTitleController
from tau_coding.tui.widgets import (
    TRANSCRIPT_WINDOW_ITEMS,
    TRANSCRIPT_WINDOW_OVERSCAN_ITEMS,
    CompactSessionInfo,
    LeftAlignedMarkdownHeading,
    SessionSidebar,
    SidebarFileItem,
    StreamingTranscriptMessageWidget,
    SystemPromptSectionWidget,
    SystemPromptSourcesWidget,
    TauMarkdownBlock,
    ThemedMarkdownWidget,
    TranscriptMessageWidget,
    TranscriptView,
    TranscriptWindowBoundary,
    _comma_list,
    _compact_token_count,
    _format_milliseconds,
    _sidebar_brand,
    _split_rich_style_colors,
    _styled_cwd,
    _syntax_language,
    _system_prompt_markdown,
    _transcript_plain_body_text,
    render_chat_item,
    render_compact_session_info,
    render_session_sidebar,
    transcript_item_selection_text,
)

ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def test_herdr_uses_textual_cell_mouse_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Verifies that when running under herdr (HERDR_ENV set) with no explicit smooth-scroll
    # setting, the herdr mouse configuration sets TEXTUAL_SMOOTH_SCROLL to "0" in the
    # environment and disables Textual's SMOOTH_SCROLL constant.
    # Inputs: monkeypatch fixture. Side effects: mutates environment and textual constants (restored by monkeypatch).
    monkeypatch.setenv("HERDR_ENV", "1")
    monkeypatch.delenv("TEXTUAL_SMOOTH_SCROLL", raising=False)
    monkeypatch.setattr(tui_app.textual_constants, "SMOOTH_SCROLL", True)

    tui_app._configure_herdr_textual_mouse()

    assert tui_app.os.environ["TEXTUAL_SMOOTH_SCROLL"] == "0"
    assert tui_app.textual_constants.SMOOTH_SCROLL is False


def test_herdr_preserves_explicit_textual_smooth_scroll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC] Verifies that under herdr an explicit TEXTUAL_SMOOTH_SCROLL="1" is preserved and the
    # Textual SMOOTH_SCROLL constant stays enabled. Inputs: monkeypatch. Side effects: env mutation.
    monkeypatch.setenv("HERDR_ENV", "1")
    monkeypatch.setenv("TEXTUAL_SMOOTH_SCROLL", "1")
    monkeypatch.setattr(tui_app.textual_constants, "SMOOTH_SCROLL", True)

    tui_app._configure_herdr_textual_mouse()

    assert tui_app.os.environ["TEXTUAL_SMOOTH_SCROLL"] == "1"
    assert tui_app.textual_constants.SMOOTH_SCROLL is True


def test_non_herdr_terminal_keeps_textual_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Verifies that outside herdr (no HERDR_ENV) the configuration leaves TEXTUAL_SMOOTH_SCROLL
    # unset and SMOOTH_SCROLL enabled. Inputs: monkeypatch. Side effects: env mutation.
    monkeypatch.delenv("HERDR_ENV", raising=False)
    monkeypatch.delenv("TEXTUAL_SMOOTH_SCROLL", raising=False)
    monkeypatch.setattr(tui_app.textual_constants, "SMOOTH_SCROLL", True)

    tui_app._configure_herdr_textual_mouse()

    assert "TEXTUAL_SMOOTH_SCROLL" not in tui_app.os.environ
    assert tui_app.textual_constants.SMOOTH_SCROLL is True


def _strip_ansi(text: str) -> str:
    # [FUNCTION DOC] HELPER: removes ANSI escape sequences (matched by the module ANSI_PATTERN) from text.
    # Inputs: a string. Output: the string with escapes stripped. No side effects.
    return ANSI_PATTERN.sub("", text)


def _style_color_escape(color: str) -> str:
    # [FUNCTION DOC] HELPER: converts a "#rrggbb" hex color into the SGR foreground truecolor parameter
    # fragment "38;2;R;G;B" used to look for styled output in rendered ANSI text.
    # Inputs: hex color string (leading "#" optional). Output: the fragment string.
    # Raises ValueError if the hex digits are invalid.
    stripped = color.lstrip("#")
    red, green, blue = (int(stripped[index : index + 2], 16) for index in (0, 2, 4))
    return f"38;2;{red};{green};{blue}"


def _style_rgb(color: str) -> str:
    # [FUNCTION DOC] HELPER: converts a "#rrggbb" hex color into the "rgb(R,G,B)" form Rich/Textual use when
    # stringifying styles. Inputs: hex color string. Output: the rgb string. Raises ValueError on bad hex.
    stripped = color.lstrip("#")
    red, green, blue = (int(stripped[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgb({red},{green},{blue})"


class FakeSessionState:
    # [CLASS DOC] HELPER: minimal stand-in for the agent state object exposed by a session; only carries a
    # class-level thinking_level attribute (default "medium") that FakeSession updates.
    thinking_level = "medium"


class FakeSession:
    # [CLASS DOC] HELPER: test double for the coding session (AgentSession) consumed by TauTuiApp.
    # Exposes the attributes the TUI reads (cwd, provider/model, model choices, tools, skills,
    # prompt templates, context files, token estimates, thinking levels, session stats,
    # system prompt and sources, diagnostics, extension names) with deterministic defaults,
    # and implements the async/sync session API (commands, model/provider switching, scoped
    # model cycling, compaction, export, resume, tree navigation and labels, new session,
    # cancel, queue inspection, terminal commands, prompting). Every call records its
    # arguments in list/counter attributes so tests can assert on what the TUI invoked.
    def __init__(self, messages=(), events=()) -> None:
        # [FUNCTION DOC] Initializes the fake with the given prebuilt messages (transcript) and events (replayed
        # by prompt) plus fixed defaults: cwd /workspace/project, provider "openai", model
        # "fake-model", three model choices including a "local" provider one, real coding tools
        # for cwd, one "review" skill, one AGENTS.md context file, sample SessionStats, thinking
        # level "medium", and empty recording lists/counters.
        # Inputs: iterables of messages and agent events (converted to tuples). Output: none.
        self.messages = tuple(messages)
        self.events = tuple(events)
        self.cwd = Path("/workspace/project")
        self.provider_name = "openai"
        self.model = "fake-model"
        self.available_models = ("fake-model", "other-model")
        self.available_model_choices = (
            ModelChoice(provider_name="openai", model="fake-model"),
            ModelChoice(provider_name="openai", model="other-model"),
            ModelChoice(provider_name="local", model="local-model"),
        )
        self.scoped_model_choices: tuple[ModelChoice, ...] = ()
        self.available_providers = ("openai",)
        self.tools = tuple(create_coding_tools(cwd=self.cwd))
        self.extension_tool_sources: dict[str, str] = {}
        self.skills = (
            Skill(
                name="review",
                path=self.cwd / ".tau" / "skills" / "review" / "SKILL.md",
                content="Review code",
            ),
        )
        self.prompt_templates = ()
        self.context_files = (
            ProjectContextFile(path=str(self.cwd / "AGENTS.md"), content="Follow rules."),
        )
        self.system_prompt_files: tuple[Path, ...] = ()
        self.context_token_estimate = 12034
        self.has_provider_context_usage = True
        self.auto_compact_token_threshold = 200000
        self.context_window_tokens = 216384
        self.thinking_level = "medium"
        self.available_thinking_levels = ("off", "minimal", "low", "medium", "high", "xhigh")
        self.state = FakeSessionState()
        self.resource_diagnostics = ()
        self.extension_names = ("permission-gate", "subagents")
        self.session_stats = SessionStats(
            turn_count=14,
            tool_call_count=23,
            input_tokens=1_200_000,
            output_tokens=48_000,
            cached_input_tokens=1_140_000,
            latest_prompt_tokens=1_200_000,
            latest_cached_input_tokens=1_188_000,
            timed_output_tokens=48_000,
            response_duration_ms=1_200_000,
            time_to_first_output_ms=18_000,
            timed_first_output_count=15,
            estimated_cost=1.24,
        )
        self.system_prompt = "You are Tau."
        self.system_prompt_sources: tuple[SystemPromptSource, ...] | None = None
        self.session_manager = None
        self._session_title: str | None = None
        self.compact_summaries: list[str] = []
        self.resumed_session_ids: list[str] = []
        self.tree_branch_requests: list[tuple[str, bool, str | None]] = []
        self.tree_label_requests: list[tuple[str, str | None]] = []
        self.tree_labels: dict[str, tuple[str, float]] = {}
        self.new_session_count = 0
        self.prompt_texts: list[str] = []
        self.prompt_sources: list[str] = []
        self.reload_count = 0
        self.provider_reload_count = 0
        self.model_catalog_refresh_count = 0
        self.queued_steering_messages: tuple[str, ...] = ()
        self.queued_follow_up_messages: tuple[str, ...] = ()
        self.streaming_behaviors: list[str | None] = []
        self.terminal_commands: list[tuple[str, bool]] = []
        self.cancel_count = 0
        self.export_calls: list[tuple[Path | None, str | None]] = []
        self.session_start_emissions = 0

    async def emit_pending_session_start(self) -> None:
        # [FUNCTION DOC] HELPER: records that the pending session-start event emission was requested by
        # incrementing session_start_emissions. Async, returns nothing.
        self.session_start_emissions += 1

    @property
    def session_title(self) -> str | None:
        # [FUNCTION DOC] HELPER: read-only property returning the current session title (None until set_session_name).
        return self._session_title

    def handle_command(self, text: str) -> CommandResult:
        # [FUNCTION DOC] HELPER: fake slash-command dispatcher. Maps fixed command strings to CommandResult values
        # mimicking the real command layer: /session (message), /reload (increments reload_count,
        # swaps skills to a single "reloaded" skill), /system (builds a SystemPromptInspection from
        # system_prompt and system_prompt_sources or a default runtime source), picker requests
        # (/skills, /resume, /prompts, /tree, /login, /logout, /model, /tools, /scoped-models,
        # /theme), /sidebar toggle, /new, /compact [summary], /export [args -> out.jsonl jsonl],
        # /resume ID, /login variants (custom/new/add, anthropic-api, anthropic-subscription,
        # provider), /logout provider, /thinking LEVEL, /theme NAME and /name NAME.
        # Inputs: the raw command text. Output: CommandResult; handled=False for anything else.
        # Side effects: only /reload mutates state.
        if text == "/session":
            return CommandResult(
                handled=True,
                message="Session info",
            )
        if text == "/reload":
            self.reload_count += 1
            self.skills = (
                Skill(
                    name="reloaded",
                    path=self.cwd / "reloaded.md",
                    content="Reloaded skill",
                ),
            )
            return CommandResult(
                handled=True,
                message="Reloaded local coding resources and project context.",
            )
        if text == "/system":
            inspection = SystemPromptInspection(
                text=self.system_prompt,
                sources=self.system_prompt_sources
                or (
                    SystemPromptSource(
                        kind="runtime",
                        label="Effective system prompt",
                        source="active Tau session",
                        content=self.system_prompt,
                    ),
                ),
            )
            return CommandResult(
                handled=True,
                message=format_system_prompt_inspection(inspection),
                system_prompt_inspection=inspection,
            )
        if text == "/skills":
            return CommandResult(handled=True, skills_picker_requested=True)
        if text == "/new":
            return CommandResult(handled=True, new_session_requested=True)
        if text == "/compact":
            return CommandResult(handled=True, compact_summary="")
        if text.startswith("/compact "):
            return CommandResult(handled=True, compact_summary=text.removeprefix("/compact "))
        if text == "/export":
            return CommandResult(handled=True, export_requested=True)
        if text.startswith("/export "):
            return CommandResult(
                handled=True,
                export_requested=True,
                export_destination=Path("out.jsonl"),
                export_format="jsonl",
            )
        if text.startswith("/resume "):
            return CommandResult(handled=True, resume_session_id=text.removeprefix("/resume "))
        if text == "/resume":
            return CommandResult(handled=True, resume_picker_requested=True)
        if text == "/prompts":
            return CommandResult(handled=True, prompts_picker_requested=True)
        if text == "/tree":
            return CommandResult(handled=True, tree_picker_requested=True)
        if text == "/login":
            return CommandResult(handled=True, login_picker_requested=True)
        if text in {"/login custom", "/login new", "/login add"}:
            return CommandResult(handled=True, custom_provider_login_requested=True)
        if text == "/login anthropic-api":
            return CommandResult(
                handled=True,
                login_provider="anthropic",
                login_method="api-key",
            )
        if text == "/login anthropic-subscription":
            return CommandResult(
                handled=True,
                login_provider="anthropic",
                login_method="subscription",
            )
        if text.startswith("/login "):
            return CommandResult(handled=True, login_provider=text.removeprefix("/login "))
        if text == "/logout":
            return CommandResult(handled=True, logout_picker_requested=True)
        if text.startswith("/logout "):
            return CommandResult(handled=True, logout_provider=text.removeprefix("/logout "))
        if text == "/model":
            return CommandResult(handled=True, model_picker_requested=True)
        if text == "/tools":
            return CommandResult(handled=True, tools_picker_requested=True)
        if text in {"/scoped-models", "/scoped models"}:
            return CommandResult(handled=True, scoped_models_picker_requested=True)
        if text.startswith("/thinking "):
            return CommandResult(handled=True, thinking_level=text.removeprefix("/thinking "))
        if text == "/theme":
            return CommandResult(handled=True, theme_picker_requested=True)
        if text == "/sidebar":
            return CommandResult(handled=True, sidebar_toggle_requested=True)
        if text.startswith("/theme "):
            return CommandResult(handled=True, theme=text.removeprefix("/theme "))
        if text.startswith("/name "):
            name = text.removeprefix("/name ")
            return CommandResult(
                handled=True,
                session_name=name,
                message=f"Session renamed: {name}",
            )
        return CommandResult(handled=False)

    def set_model(self, model: str) -> None:
        # [FUNCTION DOC] HELPER: sets the active model name. Side effect: mutates model.
        self.model = model

    def set_model_choice(self, choice: ModelChoice) -> None:
        # [FUNCTION DOC] HELPER: applies a ModelChoice by setting provider then model. Side effect: mutates provider_name/model (and available_models for "local").
        self.set_provider(choice.provider_name)
        self.set_model(choice.model)

    def toggle_scoped_model(self, choice: ModelChoice) -> tuple[ModelChoice, ...]:
        # [FUNCTION DOC] HELPER: toggles a ModelChoice in or out of scoped_model_choices (append if absent,
        # remove if present). Output: the new tuple of scoped choices. Side effect: mutates state.
        scoped = list(self.scoped_model_choices)
        if choice in scoped:
            scoped.remove(choice)
        else:
            scoped.append(choice)
        self.scoped_model_choices = tuple(scoped)
        return self.scoped_model_choices

    def cycle_scoped_model(self, *, reverse: bool = False) -> ModelChoice:
        # [FUNCTION DOC] HELPER: cycles to the next (or previous when reverse) scoped model relative to the
        # current provider/model, wrapping around; if current is not scoped it starts at the first
        # (forward) or last (reverse) entry. Output: the chosen ModelChoice, which is applied.
        # Raises ValueError when no scoped models are configured.
        if not self.scoped_model_choices:
            raise ValueError("No scoped models configured.")
        current = ModelChoice(provider_name=self.provider_name, model=self.model)
        try:
            index = self.scoped_model_choices.index(current)
        except ValueError:
            index = -1 if not reverse else 0
        delta = -1 if reverse else 1
        choice = self.scoped_model_choices[(index + delta) % len(self.scoped_model_choices)]
        self.set_model_choice(choice)
        return choice

    def set_provider(self, provider_name: str) -> None:
        # [FUNCTION DOC] HELPER: sets provider_name; when switching to "local" restricts available_models to ("local-model",).
        self.provider_name = provider_name
        if provider_name == "local":
            self.available_models = ("local-model",)

    def reload(self) -> None:
        # [FUNCTION DOC] HELPER: records a resource reload by incrementing reload_count.
        self.reload_count += 1

    def reload_provider_settings(self) -> None:
        # [FUNCTION DOC] HELPER: records a provider-settings reload by incrementing provider_reload_count.
        self.provider_reload_count += 1

    async def refresh_model_catalogs(self) -> None:
        # [FUNCTION DOC] HELPER: async; records a model-catalog refresh by incrementing model_catalog_refresh_count.
        self.model_catalog_refresh_count += 1

    async def set_session_name(self, name: str) -> str:
        # [FUNCTION DOC] HELPER: async; stores the session title and returns the name unchanged.
        self._session_title = name
        return name

    async def set_thinking_level(self, level: str) -> str:
        # [FUNCTION DOC] HELPER: async; sets thinking level on both the session and its state and returns the
        # status text "Thinking mode: LEVEL". No validation is performed.
        self.thinking_level = level
        self.state.thinking_level = level
        return f"Thinking mode: {level}"

    async def cycle_thinking_level(self) -> str:
        # [FUNCTION DOC] HELPER: async; advances thinking_level to the next entry of available_thinking_levels
        # (wrapping), syncs state, returns "Thinking mode: LEVEL". Raises ValueError if the current
        # level is not in the list.
        levels = self.available_thinking_levels
        current_index = levels.index(self.thinking_level)
        self.thinking_level = levels[(current_index + 1) % len(levels)]
        self.state.thinking_level = self.thinking_level
        return f"Thinking mode: {self.thinking_level}"

    async def compact(self, summary: str) -> str:
        # [FUNCTION DOC] HELPER: async fake compaction. Records the summary, replaces messages with a single
        # summary user message, sets context_token_estimate to 42, returns a fixed status string.
        self.compact_summaries.append(summary)
        self.messages = (UserMessage(content="Previous conversation summary:\nGenerated summary"),)
        self.context_token_estimate = 42
        return "Compacted 2 context entries."

    async def export(self, destination: Path | None = None, *, format: str | None = None) -> Path:
        # [FUNCTION DOC] HELPER: async fake export. Records (destination, format) and returns cwd/session.html
        # regardless of the arguments.
        self.export_calls.append((destination, format))
        return self.cwd / "session.html"

    async def resume(self, session_id: str) -> str:
        # [FUNCTION DOC] HELPER: async fake resume. Records the session id, replaces messages with one
        # "Restored prompt" user message, sets token estimate 456, returns "Resumed session: ID".
        self.resumed_session_ids.append(session_id)
        self.messages = (UserMessage(content="Restored prompt"),)
        self.context_token_estimate = 456
        return f"Resumed session: {session_id}"

    async def tree_choices(self) -> tuple[SessionTreeChoice, ...]:
        # [FUNCTION DOC] HELPER: async; returns a fixed four-entry session tree (root, a tool call, "left" with
        # any bookmark label/timestamp recorded via set_label, and active "right").
        return (
            SessionTreeChoice(entry_id="root", label="user: Root"),
            SessionTreeChoice(entry_id="tool", label="tool call: read", is_tool_call=True),
            SessionTreeChoice(
                entry_id="left",
                label="assistant: Left",
                bookmark_label=self.tree_labels.get("left", (None, None))[0],
                label_timestamp=self.tree_labels.get("left", (None, None))[1],
            ),
            SessionTreeChoice(entry_id="right", label="assistant: Right", active=True),
        )

    async def set_label(self, entry_id: str, label: str | None) -> SimpleNamespace:
        # [FUNCTION DOC] HELPER: async fake label setter. Records (entry_id, label); stores label with a
        # synthetic increasing timestamp, or removes it when label is None. Output: an object with
        # a timestamp attribute.
        self.tree_label_requests.append((entry_id, label))
        timestamp = 1_700_000_000.0 + len(self.tree_label_requests)
        if label is None:
            self.tree_labels.pop(entry_id, None)
        else:
            self.tree_labels[entry_id] = (label, timestamp)
        return SimpleNamespace(timestamp=timestamp)

    async def branch_to_entry(
        self,
        entry_id: str,
        *,
        summarize: bool = False,
        custom_instructions: str | None = None,
    ) -> str:
        # [FUNCTION DOC] HELPER: async fake tree branch. Records (entry_id, summarize, custom_instructions),
        # replaces messages with a "Branched to ID" user message, returns a status string.
        self.tree_branch_requests.append((entry_id, summarize, custom_instructions))
        self.messages = (UserMessage(content=f"Branched to {entry_id}"),)
        return f"Branched session at {entry_id}."

    async def new_session(self) -> str:
        # [FUNCTION DOC] HELPER: async; starts a fake new session: increments new_session_count, clears messages
        # and token estimate, returns a fixed status string.
        self.new_session_count += 1
        self.messages = ()
        self.context_token_estimate = 0
        return "Started new session: new-session"

    def cancel(self) -> None:
        # [FUNCTION DOC] HELPER: records a cancellation by incrementing cancel_count.
        self.cancel_count += 1

    def queue_update_event(self) -> QueueUpdateEvent:
        # [FUNCTION DOC] HELPER: builds a QueueUpdateEvent from the current queued steering and follow-up messages.
        return QueueUpdateEvent(
            steering=self.queued_steering_messages,
            follow_up=self.queued_follow_up_messages,
        )

    async def run_terminal_command(
        self,
        command: str,
        *,
        add_to_context: bool,
    ) -> TerminalCommandResult:
        # [FUNCTION DOC] HELPER: async fake shell runner. Records (command, add_to_context) and returns a
        # successful TerminalCommandResult with output "command output" and exit code 0.
        self.terminal_commands.append((command, add_to_context))
        return TerminalCommandResult(
            command=command,
            output="command output",
            exit_code=0,
            ok=True,
            added_to_context=add_to_context,
        )

    def pop_latest_follow_up_message(self) -> str | None:
        # [FUNCTION DOC] HELPER: removes and returns the most recently queued follow-up message, or None if empty.
        if not self.queued_follow_up_messages:
            return None
        message = self.queued_follow_up_messages[-1]
        self.queued_follow_up_messages = self.queued_follow_up_messages[:-1]
        return message

    def pop_latest_steering_message(self) -> str | None:
        # [FUNCTION DOC] HELPER: removes and returns the most recently queued steering message, or None if empty.
        if not self.queued_steering_messages:
            return None
        message = self.queued_steering_messages[-1]
        self.queued_steering_messages = self.queued_steering_messages[:-1]
        return message

    async def prompt(
        self,
        text: str,
        *,
        streaming_behavior: str | None = None,
        source: str = "interactive",
        custom_type: str | None = None,
        details: dict[str, object] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        # [FUNCTION DOC] HELPER: async generator fake for prompting. Records text, source and streaming behavior.
        # For "steer" or "follow_up" it appends text to the matching queue and yields one
        # QueueUpdateEvent; otherwise it yields each preconfigured event in order.
        # custom_type and details are accepted but ignored.
        self.prompt_texts.append(text)
        self.prompt_sources.append(source)
        self.streaming_behaviors.append(streaming_behavior)
        if streaming_behavior == "steer":
            self.queued_steering_messages = (*self.queued_steering_messages, text)
            yield self.queue_update_event()
            return
        if streaming_behavior == "follow_up":
            self.queued_follow_up_messages = (*self.queued_follow_up_messages, text)
            yield self.queue_update_event()
            return
        for event in self.events:
            yield event


def _visible_footer_bindings(app: TauTuiApp) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: returns a mapping from binding description to displayed key for every currently
    # active, visible binding on the app's screen (what a Footer would show).
    # Inputs: a running TauTuiApp. Output: description-to-key mapping.
    """Return active bindings that a Textual Footer would render if mounted."""
    return {
        binding.description: binding.key_display or binding.key
        for _, binding, _enabled, _tooltip in app.screen.active_bindings.values()
        if binding.show
    }


@pytest.mark.parametrize(
    ("milliseconds", "expected"),
    [(999, "999ms"), (1000, "1.0s"), (999.6, "1.0s")],
)
def test_format_milliseconds_handles_unit_boundary(
    milliseconds: float,
    expected: str,
) -> None:
    # [FUNCTION DOC] Parametrized: verifies the duration formatter emits "NNNms" below one second and
    # one-decimal seconds at/after the boundary (999.6ms rounds to "1.0s"). Fails via assertion.
    assert _format_milliseconds(milliseconds) == expected


def test_session_sidebar_renders_session_metadata() -> None:
    # [FUNCTION DOC] Verifies the full session sidebar for a default FakeSession: shows session/context/usage
    # sections, untitled title, turn and tool-call counts, token usage with cost, cache rates,
    # TPS/TTFT, auto-compact threshold, tool list, skills grouped under ./.tau/skills and
    # extension names, while omitting provider/model/thinking/location/branch and the brand art.
    console = Console(record=True, width=80)

    console.print(render_session_sidebar(FakeSession()))

    output = console.export_text()
    assert "████████" not in output
    assert "τ = 2π" not in output
    assert "session" in output
    assert "context" in output
    assert "AGENTS.md" in output
    assert "12k" not in output
    assert "Untitled session" in output
    assert "provider" not in output
    assert "openai" not in output
    assert "fake-model" not in output
    assert "thinking" not in output
    assert "location" not in output
    assert "branch" not in output
    assert "14 turns, 23 tool calls" in output
    assert "usage" in output
    assert "cumulative usage" not in output
    assert "1.2m in, 48k out · ~$1.24" in output
    assert "cache: 99% latest · 95% session" in output
    assert "avg TPS: 40.0 · avg TTFT: 1.2s" in output
    assert "auto at 200k" in output
    assert "read, write, edit, bash" in output
    assert re.search(r"\./\.tau/skills\s+• review", output)
    assert "permission-gate, subagents" in output


def test_session_sidebar_shows_all_skills() -> None:
    # [FUNCTION DOC] Verifies the sidebar lists all seven skills without a "(N more)" truncation.
    session = FakeSession()
    session.skills = tuple(
        Skill(
            name=f"skill-{index}",
            path=session.cwd / ".tau" / "skills" / f"skill-{index}" / "SKILL.md",
            content="Skill",
        )
        for index in range(1, 8)
    )
    console = Console(record=True, width=80)

    console.print(render_session_sidebar(session))

    output = console.export_text()
    for index in range(1, 8):
        assert f"• skill-{index}" in output
    assert "more)" not in output


def test_session_sidebar_marks_user_only_skills_with_hollow_bullets() -> None:
    # [FUNCTION DOC] Verifies skills with model invocation disabled are shown with a hollow bullet while
    # model-visible skills use a filled bullet.
    session = FakeSession()
    session.skills = (
        Skill(
            name="model-visible",
            path=session.cwd / ".tau/skills/model-visible/SKILL.md",
            content="Model-visible skill",
        ),
        Skill(
            name="user-only",
            path=session.cwd / ".tau/skills/user-only/SKILL.md",
            content="User-only skill",
            disable_model_invocation=True,
        ),
    )
    console = Console(record=True, width=80)

    console.print(render_session_sidebar(session))

    output = console.export_text()
    assert "• model-visible" in output
    assert "◦ user-only" in output
    assert "• user-only" not in output


def test_session_sidebar_groups_skills_by_origin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Verifies skills are grouped under their origin directory headers in order: user ~/.tau,
    # user ~/.agents, project ./.tau, project ./.agents. Inputs: monkeypatch, tmp_path (isolated home).
    isolate_home(monkeypatch, tmp_path)
    session = FakeSession()
    session.cwd = tmp_path / "project"
    session.skills = (
        Skill("project-agents", session.cwd / ".agents/skills/project-agents/SKILL.md", ""),
        Skill("user-tau", tmp_path / ".tau/skills/user-tau/SKILL.md", ""),
        Skill("project-tau", session.cwd / ".tau/skills/project-tau/SKILL.md", ""),
        Skill("user-agents", tmp_path / ".agents/skills/user-agents/SKILL.md", ""),
    )
    console = Console(record=True, width=80)

    console.print(render_session_sidebar(session))

    output = console.export_text()
    expected_groups = (
        ("~/.tau/skills", "user-tau"),
        ("~/.agents/skills", "user-agents"),
        ("./.tau/skills", "project-tau"),
        ("./.agents/skills", "project-agents"),
    )
    assert all(
        re.search(rf"{re.escape(origin)}\s+• {name}", output) for origin, name in expected_groups
    )
    assert [output.index(origin) for origin, _name in expected_groups] == sorted(
        output.index(origin) for origin, _name in expected_groups
    )


def test_session_sidebar_orders_configured_tau_home_before_shared_and_project_resources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Verifies a configured non-default Tau home (~/.tau-personal) is listed before shared
    # ~/.agents and project resources in the skills section. Inputs: monkeypatch, tmp_path.
    isolate_home(monkeypatch, tmp_path)
    session = FakeSession()
    session.cwd = tmp_path / "project"
    session.skills = (
        Skill("project-agents", session.cwd / ".agents/skills/project-agents/SKILL.md", ""),
        Skill("user-tau", tmp_path / ".tau-personal/skills/user-tau/SKILL.md", ""),
        Skill("project-tau", session.cwd / ".tau/skills/project-tau/SKILL.md", ""),
        Skill("user-agents", tmp_path / ".agents/skills/user-agents/SKILL.md", ""),
    )
    console = Console(record=True, width=80)

    console.print(render_session_sidebar(session))

    output = console.export_text()
    expected_origins = (
        "~/.tau-personal/skills",
        "~/.agents/skills",
        "./.tau/skills",
        "./.agents/skills",
    )
    assert [output.index(origin) for origin in expected_origins] == sorted(
        output.index(origin) for origin in expected_origins
    )


def test_session_sidebar_groups_and_shows_all_prompts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Verifies prompt templates are grouped by origin (same ordering as skills) and all eight
    # are shown with no truncation marker. Inputs: monkeypatch, tmp_path.
    isolate_home(monkeypatch, tmp_path)
    session = FakeSession()
    session.cwd = tmp_path / "project"
    session.prompt_templates = (
        PromptTemplate("project-agents", session.cwd / ".agents/prompts/project-agents.md", ""),
        PromptTemplate("user-tau", tmp_path / ".tau/prompts/user-tau.md", ""),
        PromptTemplate("project-tau", session.cwd / ".tau/prompts/project-tau.md", ""),
        PromptTemplate("user-agents", tmp_path / ".agents/prompts/user-agents.md", ""),
        *tuple(
            PromptTemplate(
                f"extra-{index}",
                session.cwd / f".tau/prompts/extra-{index}.md",
                "",
            )
            for index in range(1, 5)
        ),
    )
    console = Console(record=True, width=80)

    console.print(render_session_sidebar(session))

    output = console.export_text()
    expected_groups = (
        ("~/.tau/prompts", "user-tau"),
        ("~/.agents/prompts", "user-agents"),
        ("./.tau/prompts", "extra-1"),
        ("./.agents/prompts", "project-agents"),
    )
    assert all(
        re.search(rf"{re.escape(origin)}\s+• {name}", output) for origin, name in expected_groups
    )
    assert "• project-tau" in output
    for index in range(1, 5):
        assert f"• extra-{index}" in output
    assert "more)" not in output
    assert [output.index(origin) for origin, _name in expected_groups] == sorted(
        output.index(origin) for origin, _name in expected_groups
    )


def test_session_sidebar_limits_context_files_to_five() -> None:
    # [FUNCTION DOC] Verifies the context-files section shows at most five files followed by "...(2 more)".
    session = FakeSession()
    session.context_files = tuple(
        ProjectContextFile(path=str(session.cwd / f"context-{index}.md"), content="Rules")
        for index in range(1, 8)
    )
    console = Console(record=True, width=80)

    console.print(render_session_sidebar(session))

    output = console.export_text()
    for index in range(1, 6):
        assert f"• context-{index}.md" in output
    assert "context-6.md" not in output
    assert "context-7.md" not in output
    assert "...(2 more)" in output


def test_session_sidebar_lists_active_system_prompt_files() -> None:
    # [FUNCTION DOC] Verifies active system prompt files are listed under a "system prompt" section with
    # cwd-relative and home-abbreviated paths.
    session = FakeSession()
    session.system_prompt_files = (
        session.cwd / ".tau" / "SYSTEM.md",
        Path.home() / ".tau" / "APPEND_SYSTEM.md",
    )
    console = Console(record=True, width=80)

    console.print(render_session_sidebar(session))

    output = console.export_text()
    assert "system prompt" in output
    assert "• .tau/SYSTEM.md" in output
    assert "• ~/.tau/APPEND_SYSTEM.md" in output


def test_session_sidebar_omits_system_prompt_section_without_active_files() -> None:
    # [FUNCTION DOC] Verifies the "system prompt" section is absent when no system prompt files are active.
    console = Console(record=True, width=80)

    console.print(render_session_sidebar(FakeSession()))

    assert "system prompt" not in console.export_text()


def test_comma_list_limits_by_rendered_lines_instead_of_item_count() -> None:
    # [FUNCTION DOC] Verifies the comma-separated list helper truncates by rendered line count (three lines
    # plus a "...(N more)" line) rather than item count, at narrow and wide console widths.
    items = [f"i{index}" for index in range(1, 16)]

    narrow_console = Console(record=True, width=12)
    narrow_console.print(_comma_list(items, empty="Empty", theme=TAU_DARK_THEME))
    narrow_output = narrow_console.export_text()
    assert "i9" in narrow_output
    assert "i10" not in narrow_output
    assert "...(6 more)" in narrow_output
    assert len(narrow_output.splitlines()) == 4

    wide_console = Console(record=True, width=30)
    wide_console.print(_comma_list(items, empty="Empty", theme=TAU_DARK_THEME))
    wide_output = wide_console.export_text()
    assert "i15" in wide_output
    assert "more)" not in wide_output
    assert len(wide_output.splitlines()) == 3


@pytest.mark.parametrize(("item_count", "hidden_label"), [(1, None), (2, "...(1 more)")])
def test_comma_list_represents_an_oversized_first_item(
    item_count: int,
    hidden_label: str | None,
) -> None:
    # [FUNCTION DOC] Parametrized: verifies a first item wider than the console is cropped with an ellipsis
    # and, when other items exist, followed by a hidden-count label.
    oversized_name = "x" * 40
    console = Console(record=True, width=12)

    console.print(
        _comma_list(
            [oversized_name, "second-item"][:item_count],
            empty="Empty",
            theme=TAU_DARK_THEME,
        )
    )

    output = console.export_text()
    assert output.startswith("x" * 12)
    assert "…" in output
    if hidden_label is None:
        assert len(output.splitlines()) == 3
        assert "more)" not in output
    else:
        assert len(output.splitlines()) == 4
        assert hidden_label in output


@pytest.mark.parametrize(
    ("attribute", "prefix"),
    [("tools", "tool"), ("extension_names", "extension")],
)
def test_session_sidebar_limits_comma_separated_sections_to_three_lines(
    attribute: str,
    prefix: str,
) -> None:
    # [FUNCTION DOC] Parametrized over tools and extension names: verifies these comma-separated sidebar
    # sections are capped at three lines with a "...(" overflow marker.
    session = FakeSession()
    names = tuple(f"{prefix}-item-{index}" for index in range(1, 20))
    values: tuple[object, ...] = names
    if attribute != "extension_names":
        values = tuple(SimpleNamespace(name=name) for name in names)
    setattr(session, attribute, values)
    console = Console(record=True, width=40)

    console.print(render_session_sidebar(session))

    output = console.export_text()
    assert names[0] in output
    assert names[-1] not in output
    assert "...(" in output


def test_session_sidebar_uses_na_when_cost_is_unavailable() -> None:
    # [FUNCTION DOC] Verifies the usage line shows "$N/A" when no cost estimate exists.
    session = FakeSession()
    session.session_stats = SessionStats(input_tokens=1200, output_tokens=300)
    console = Console(record=True, width=80)

    console.print(render_session_sidebar(session))

    output = console.export_text()
    assert "$N/A" in output
    assert "cost unavailable" not in output


def test_session_sidebar_omits_cache_rate_for_providers_without_caching() -> None:
    # [FUNCTION DOC] Verifies no cache-rate text appears when stats contain no cached tokens.
    session = FakeSession()
    session.session_stats = SessionStats(input_tokens=1200, output_tokens=300)
    console = Console(record=True, width=80)

    console.print(render_session_sidebar(session))

    output = console.export_text()
    assert "cached" not in output
    assert "1.2k in, 300 out" in output


def test_session_sidebar_shows_latest_cache_miss_with_session_rate() -> None:
    # [FUNCTION DOC] Verifies the cache line reports 0% for the latest prompt (a cache miss) alongside the
    # cumulative session hit rate (25%).
    session = FakeSession()
    session.session_stats = SessionStats(
        input_tokens=2_000,
        output_tokens=300,
        cached_input_tokens=500,
        cache_write_tokens=500,
        latest_prompt_tokens=1_000,
    )
    console = Console(record=True, width=80)

    console.print(render_session_sidebar(session))

    output = console.export_text()
    assert "cache: 0% latest · 25% session" in output


def test_session_sidebar_brand_includes_current_version() -> None:
    # [FUNCTION DOC] Verifies the sidebar brand text includes the "τ = 2π" mark and the current package
    # version (0.4.8). KNOWN ISSUE: the version string is hard-coded and breaks on version bumps.
    console = Console(record=True, width=80)

    console.print(_sidebar_brand(theme=TAU_DARK_THEME))

    assert "τ = 2π  0.4.8" in console.export_text()


def test_session_sidebar_uses_prominent_title_and_accented_section_headers() -> None:
    # [FUNCTION DOC] Verifies the sidebar uses no boxed panels: the session title is bold accent and indented,
    # section headers are bold prompt-text colored, section content uses the description color,
    # and sections are separated by horizontal rules rather than box borders. Relies on the
    # internal renderable structure order (title at index 0, activity section at index 2).
    console = Console(record=True, width=80)
    session = FakeSession()
    session._session_title = "Customer bugfix"
    sidebar = render_session_sidebar(session)
    panels = [renderable for renderable in sidebar.renderables if isinstance(renderable, Panel)]
    session_name = sidebar.renderables[0]
    activity_section = sidebar.renderables[2]
    activity_header = activity_section.renderables[0]
    activity_content = activity_section.renderables[1]

    console.print(sidebar)

    output = console.export_text()
    assert panels == []
    assert session_name.left == 1
    assert str(session_name.renderable.style) == f"bold {TAU_DARK_THEME.accent}"
    assert activity_header.left == 1
    assert str(activity_header.renderable.style) == f"bold {TAU_DARK_THEME.prompt_text}"
    assert str(activity_content.renderable.style) == TAU_DARK_THEME.completion_description
    assert " context" in output
    assert " tools" in output
    assert "─" in output
    assert "┌" not in output
    assert "│" not in output


def test_session_sidebar_lists_multiple_context_files() -> None:
    # [FUNCTION DOC] Verifies multiple context files render with cwd-relative, home-abbreviated (~) and
    # external absolute paths as appropriate.
    session = FakeSession()
    session.context_files = (
        ProjectContextFile(path=str(session.cwd / "AGENTS.md"), content="Root rules."),
        ProjectContextFile(
            path=str(session.cwd / ".agents" / "AGENTS.md"),
            content="Agent rules.",
        ),
        ProjectContextFile(path="docs/AGENTS.md", content="Docs rules."),
        ProjectContextFile(
            path=str(Path.home() / ".agents" / "AGENTS.md"),
            content="User rules.",
        ),
        ProjectContextFile(path="/Users/alex/.agents/AGENTS.md", content="External rules."),
    )
    console = Console(record=True, width=100)

    console.print(render_session_sidebar(session))

    output = console.export_text()
    assert "AGENTS.md" in output
    assert ".agents/AGENTS.md" in output
    assert "docs/AGENTS.md" in output
    assert "~/.agents/AGENTS.md" in output
    assert str(Path.home() / ".agents" / "AGENTS.md") not in output
    assert "/Users/alex/.agents/AGENTS.md" in output


def test_compact_session_info_renders_sidebar_facts() -> None:
    # [FUNCTION DOC] Verifies the compact (narrow layout) session info shows cwd with branch placeholder,
    # provider:model with thinking level, and the context usage on the next line.
    console = Console(record=True, width=120)

    console.print(render_compact_session_info(FakeSession()))

    output = console.export_text()
    lines = output.splitlines()
    provider_line = next(index for index, line in enumerate(lines) if "openai:fake-model" in line)
    context_line = next(index for index, line in enumerate(lines) if "12k/200k" in line)
    assert "/workspace/project (--)" in output
    assert "context 12k/200k" not in output
    assert "openai:fake-model" in lines[provider_line]
    assert "(medium)" in lines[provider_line]
    assert context_line == provider_line + 1


def test_compact_session_info_omits_unavailable_thinking_controls() -> None:
    # [FUNCTION DOC] Verifies the compact info omits the thinking-level suffix when no levels are available.
    console = Console(record=True, width=120)
    session = FakeSession()
    session.available_thinking_levels = ()

    console.print(render_compact_session_info(session))

    provider_line = next(
        line for line in console.export_text().splitlines() if "openai:fake-model" in line
    )
    assert "unavailable" not in provider_line
    assert "fake-model (" not in provider_line


def test_compact_session_info_shows_unknown_without_provider_usage() -> None:
    # [FUNCTION DOC] Verifies the compact info shows "?/200k" when provider-reported context usage is unknown.
    console = Console(record=True, width=120)
    session = FakeSession()
    session.has_provider_context_usage = False

    console.print(render_compact_session_info(session))

    assert "?/200k" in console.export_text()


def test_compact_session_info_redraws_when_provider_usage_becomes_available() -> None:
    # [FUNCTION DOC] Verifies CompactSessionInfo re-renders (calls update again) when provider context usage
    # becomes available, switching from "?/200k" to "12k/200k". Replaces the widget's update
    # method with a list append to capture renderables.
    session = FakeSession()
    session.has_provider_context_usage = False
    widget = CompactSessionInfo()
    updates: list[object] = []
    widget.update = updates.append  # type: ignore[method-assign]

    widget.update_from_session(session)
    first_console = Console(record=True, width=120)
    first_console.print(updates[-1])
    assert "?/200k" in first_console.export_text()

    session.has_provider_context_usage = True
    widget.update_from_session(session)
    assert len(updates) == 2
    second_console = Console(record=True, width=120)
    second_console.print(updates[-1])
    second_output = second_console.export_text()
    assert "12k/200k" in second_output
    assert "?/200k" not in second_output


def test_compact_session_info_styles_provider_as_metadata() -> None:
    # [FUNCTION DOC] Verifies the compact info styles the provider name with the description (metadata) color
    # and the ":model" part with the prompt-text color in truecolor output.
    console = Console(record=True, width=120, color_system="truecolor")

    console.print(render_compact_session_info(FakeSession()))

    output = console.export_text(styles=True)
    assert f"\x1b[{_style_color_escape(TAU_DARK_THEME.completion_description)}mopenai" in output
    assert f"\x1b[{_style_color_escape(TAU_DARK_THEME.prompt_text)}m:fake-model" in output


def test_compact_session_info_styles_parent_path_as_metadata() -> None:
    # [FUNCTION DOC] Verifies the styled cwd text splits into parent path (metadata color), final directory
    # (prompt color) and branch suffix (metadata color) spans.
    cwd = _styled_cwd(Path("/workspace/project"), theme=TAU_DARK_THEME)

    assert cwd.plain == "/workspace/project (--)"
    assert str(cwd.spans[0].style) == TAU_DARK_THEME.completion_description
    assert str(cwd.spans[1].style) == TAU_DARK_THEME.prompt_text
    assert str(cwd.spans[2].style) == TAU_DARK_THEME.completion_description


def test_compact_token_count_uses_thousands_suffix() -> None:
    # [FUNCTION DOC] Verifies compact token counts: 0 -> "0k", under 500 -> "<1k", otherwise rounded thousands.
    assert _compact_token_count(0) == "0k"
    assert _compact_token_count(499) == "<1k"
    assert _compact_token_count(12034) == "12k"
    assert _compact_token_count(12500) == "13k"


def test_compact_session_info_wraps_to_available_width() -> None:
    # [FUNCTION DOC] Verifies compact session info wraps into multiple lines within a 36-column console.
    console = Console(record=True, width=36)

    console.print(render_compact_session_info(FakeSession()))

    lines = console.export_text().splitlines()
    assert len(lines) > 1
    assert max(len(line) for line in lines) <= 36


def test_render_chat_item_custom_uses_markup() -> None:
    # [FUNCTION DOC] Verifies a custom chat item renders the supplied Rich markup instead of its raw text.
    console = Console(record=True, width=60)
    item = ChatItem(
        role="custom",
        text="<task-notification>raw</task-notification>",
        custom_type="subagent-notification",
    )

    console.print(render_chat_item(item, custom_markup="[bold]research complete[/bold]"))
    output = console.export_text()

    assert "research complete" in output
    assert "raw" not in output


def test_render_chat_item_custom_falls_back_to_raw_without_markup() -> None:
    # [FUNCTION DOC] Verifies a custom chat item falls back to its raw text when no markup is supplied.
    console = Console(record=True, width=60)
    item = ChatItem(
        role="custom",
        text="raw notification body",
        custom_type="subagent-notification",
    )

    console.print(render_chat_item(item, custom_markup=None))
    output = console.export_text()

    assert "raw notification body" in output


def test_render_chat_item_custom_survives_malformed_markup() -> None:
    # [FUNCTION DOC] Verifies malformed Rich markup for a custom item does not raise and renders literally.
    console = Console(record=True, width=60)
    item = ChatItem(role="custom", text="raw", custom_type="c")

    # Malformed Rich markup must not raise; it renders literally instead.
    console.print(render_chat_item(item, custom_markup="[bold unterminated"))
    output = console.export_text()

    assert "unterminated" in output


def test_state_add_user_message_with_custom_type_creates_custom_item() -> None:
    # [FUNCTION DOC] Verifies TuiState.add_user_message with a custom_type creates a "custom" role item
    # carrying custom_type, details and the raw text.
    state = TuiState()

    state.add_user_message("raw body", custom_type="subagent-notification", details={"id": "x"})

    assert len(state.items) == 1
    item = state.items[0]
    assert item.role == "custom"
    assert item.custom_type == "subagent-notification"
    assert item.details == {"id": "x"}
    assert item.text == "raw body"


def test_state_resolve_custom_markup_uses_installed_renderer() -> None:
    # [FUNCTION DOC] Verifies resolve_custom_markup delegates to the installed custom renderer callback.
    state = TuiState()
    state.custom_renderer = lambda ct, content, details, expanded: f"[{ct}]{content}"
    state.add_user_message("hi", custom_type="c")

    markup = state.resolve_custom_markup(state.items[0], expanded=False)

    assert markup == "[c]hi"


def test_state_resolve_custom_markup_none_without_renderer() -> None:
    # [FUNCTION DOC] Verifies resolve_custom_markup returns None when no custom renderer is installed.
    state = TuiState()
    state.add_user_message("hi", custom_type="c")

    assert state.resolve_custom_markup(state.items[0], expanded=False) is None


def test_state_load_messages_projects_custom_type_on_resume() -> None:
    # [FUNCTION DOC] Verifies load_messages projects displayable CustomMessages into custom items (keeping
    # type and details) and skips custom messages marked display=False.
    state = TuiState()
    state.load_messages(
        [
            UserMessage(content="hello"),
            CustomMessage(
                content="<task-notification/>",
                custom_type="subagent-notification",
                details={"id": "run-1"},
            ),
            CustomMessage(
                content="hidden model context",
                custom_type="extension:hidden",
                display=False,
            ),
        ]
    )

    assert [item.role for item in state.items] == ["user", "custom"]
    assert state.items[1].custom_type == "subagent-notification"
    assert state.items[1].details == {"id": "run-1"}


def test_tui_event_adapter_hides_non_display_custom_messages() -> None:
    # [FUNCTION DOC] Verifies the event adapter ignores MessageEndEvent for non-display custom messages.
    state = TuiState()
    adapter = TuiEventAdapter(state)

    adapter.apply(
        MessageEndEvent(
            message=CustomMessage(
                content="hidden model context",
                custom_type="extension:hidden",
                display=False,
            )
        )
    )

    assert state.items == []


def test_chat_items_render_as_unlabeled_blocks() -> None:
    # [FUNCTION DOC] Verifies user chat items render without role labels, prefixed by the left accent bar.
    console = Console(record=True, width=40)

    console.print(render_chat_item(ChatItem(role="user", text="Read the file")))
    output = console.export_text()

    assert "Read the file" in output
    assert "you:" not in output
    assert "assistant:" not in output
    assert "tool:" not in output
    assert "▌ Read the file" in output


def test_chat_items_use_left_accent_instead_of_box_border() -> None:
    # [FUNCTION DOC] Verifies assistant items use a left accent bar rather than a box border.
    console = Console(record=True, width=40)

    console.print(render_chat_item(ChatItem(role="assistant", text="Done.")))
    output = console.export_text()

    assert "▌ Done." in output
    assert "┌" not in output
    assert "└" not in output


def test_chat_items_have_bottom_padding() -> None:
    # [FUNCTION DOC] Verifies rendered chat items end with a blank padding line.
    console = Console(record=True, width=40)

    console.print(render_chat_item(ChatItem(role="user", text="Read the file")))
    output = console.export_text().splitlines()

    assert output[-1].strip() == ""


def test_chat_items_fold_long_unbroken_text_to_console_width() -> None:
    # [FUNCTION DOC] Verifies long unbroken words are folded so no line exceeds the console width.
    console = Console(record=True, width=36)
    long_text = "supercalifragilisticexpialidocious" * 2

    console.print(render_chat_item(ChatItem(role="assistant", text=long_text)))
    output = console.export_text()

    assert max(len(line) for line in output.splitlines()) <= 36


def test_chat_items_use_configured_theme_accent() -> None:
    # [FUNCTION DOC] Verifies a chat item uses the provided theme's accent color (high-contrast green).
    console = Console(record=True, width=40)

    console.print(
        render_chat_item(
            ChatItem(role="assistant", text="Done."),
            theme=HIGH_CONTRAST_THEME,
        )
    )
    output = console.export_text(styles=True)

    assert "Done." in output
    assert "38;2;0;255;102" in output


def test_chat_items_render_fenced_code_without_markers() -> None:
    # [FUNCTION DOC] Verifies fenced code blocks render code without the backtick fences or language tag.
    console = Console(record=True, width=60)
    item = ChatItem(
        role="assistant",
        text='Here is code:\n\n```python\nprint("hi")\n```',
    )

    console.print(render_chat_item(item))
    output = console.export_text()

    assert 'print("hi")' in output
    assert "```" not in output
    assert "python" not in output


def test_assistant_chat_items_apply_syntax_highlighting_to_code_fences() -> None:
    # [FUNCTION DOC] Verifies assistant code fences are syntax highlighted (keyword color on code background).
    # Relies on exact ANSI sequences from the current Pygments theme.
    console = Console(record=True, width=80, color_system="truecolor")
    item = ChatItem(role="assistant", text="```python\ndef hi():\n    return 1\n```")

    console.print(render_chat_item(item))
    output = console.export_text(styles=True)

    assert "def" in output
    assert "return" in output
    assert "\x1b[94;48;2;22;27;33mdef" in output
    assert "\x1b[94;48;2;22;27;33mreturn" in output


def test_chat_items_fallback_unknown_fenced_language_to_plain_code() -> None:
    # [FUNCTION DOC] Verifies unknown fence languages map to the plain "text" lexer.
    assert _syntax_language("definitely-not-a-lexer") == "text"

    console = Console(record=True, width=60)
    item = ChatItem(role="assistant", text="```definitely-not-a-lexer\nvalue\n```")

    console.print(render_chat_item(item))
    output = console.export_text()

    assert "value" in output
    assert "```" not in output
    assert "definitely-not-a-lexer" not in output


def test_tool_chat_items_hide_and_show_result_text() -> None:
    # [FUNCTION DOC] Verifies tool items show only the invocation line when collapsed and include the result
    # text when rendered with show_tool_results.
    item = ChatItem(
        role="tool",
        text="→ read README.md",
        tool_result_text="✓ read\nfull file contents",
    )

    collapsed_console = Console(record=True, width=80)
    collapsed_console.print(render_chat_item(item))
    collapsed = collapsed_console.export_text()

    expanded_console = Console(record=True, width=80)
    expanded_console.print(render_chat_item(item, show_tool_results=True))
    expanded = expanded_console.export_text()

    assert "→ read" in collapsed
    assert "full file contents" not in collapsed
    assert "→ read" in expanded
    assert "full file contents" in expanded


EDIT_TOOL_RESULT_WITH_PATCH = (
    "✓ edit\n"
    "Successfully replaced 1 block.\n"
    "\n"
    "Patch:\n"
    "--- a/README.md\n"
    "+++ b/README.md\n"
    "@@\n"
    "-old\n"
    "+new"
)


def test_expanded_edit_tool_result_renders_patch_as_colored_diff() -> None:
    # [FUNCTION DOC] Verifies an expanded edit tool result renders its "Patch:" unified diff with red removed
    # lines and green added lines.
    item = ChatItem(
        role="tool",
        text="→ edit README.md",
        tool_result_text=EDIT_TOOL_RESULT_WITH_PATCH,
    )

    console = Console(record=True, width=100, color_system="truecolor")
    console.print(render_chat_item(item, show_tool_results=True))

    plain = console.export_text(clear=False)
    styled = console.export_text(styles=True)
    assert "Patch:" in plain
    assert "-old" in plain
    assert "+new" in plain
    assert re.search(r"\x1b\[91;[^m]*m-old", styled)
    assert re.search(r"\x1b\[92;[^m]*m\+new", styled)


def test_transcript_plain_tool_body_renders_patch_as_colored_diff() -> None:
    # [FUNCTION DOC] Verifies the transcript plain-body renderer also colors the edit patch as a diff.
    item = ChatItem(
        role="tool",
        text="→ edit README.md",
        tool_result_text=EDIT_TOOL_RESULT_WITH_PATCH,
    )
    body = _transcript_plain_body_text(
        item,
        text=transcript_item_selection_text(item, show_tool_results=True),
        body_style="#cbd5e1 on #000000",
        theme=TAU_DARK_THEME,
        show_tool_results=True,
    )

    console = Console(record=True, width=100, color_system="truecolor")
    console.print(body)

    plain = console.export_text(clear=False)
    styled = console.export_text(styles=True)
    assert "Patch:" in plain
    assert "-old" in plain
    assert "+new" in plain
    assert re.search(r"\x1b\[91;[^m]*m-old", styled)
    assert re.search(r"\x1b\[92;[^m]*m\+new", styled)


def test_thinking_chat_items_use_distinct_style_and_markdown() -> None:
    # [FUNCTION DOC] Verifies thinking items render Markdown (no literal asterisks) in the muted gray thinking color.
    console = Console(record=True, width=80)

    console.print(render_chat_item(ChatItem(role="thinking", text="**Plan**\n\nHidden reasoning")))

    output = console.export_text(styles=True)
    plain = console.export_text()
    assert "Plan" in output
    assert "**Plan**" not in plain
    assert "Hidden reasoning" in output
    assert "38;2;156;163;175" in output


def test_skill_chat_items_use_distinct_compact_style() -> None:
    # [FUNCTION DOC] Verifies skill items render in the distinct lavender skill color.
    console = Console(record=True, width=80)

    console.print(render_chat_item(ChatItem(role="skill", text="Using skill: review")))

    output = console.export_text(styles=True)
    assert "Using skill: review" in output
    assert "38;2;229;212;239" in output


def test_skill_chat_items_expand_with_tool_results_toggle() -> None:
    # [FUNCTION DOC] Verifies skill items hide the loaded skill body when collapsed and show it when expanded.
    item = ChatItem(
        role="skill",
        text="Loading skill: review",
        tool_result_text="✓ read\n# Review\nFull noisy instructions.",
    )
    collapsed_console = Console(record=True, width=80)
    collapsed_console.print(render_chat_item(item, show_tool_results=False))
    collapsed = collapsed_console.export_text()
    expanded_console = Console(record=True, width=80)
    expanded_console.print(render_chat_item(item, show_tool_results=True))
    expanded = expanded_console.export_text()

    assert "Loading skill: review" in collapsed
    assert "Full noisy instructions" not in collapsed
    assert "Loading skill: review" in expanded
    assert "Full noisy instructions" in expanded


def test_branch_summary_chat_items_expand_with_tool_results_toggle() -> None:
    # [FUNCTION DOC] Verifies branch-summary items show a placeholder when collapsed and a "Branch Summary"
    # heading plus the summary text when expanded.
    item = ChatItem(
        role="branch_summary",
        text="Branch summary (Ctrl+O to expand)",
        tool_result_text="Detailed summary text",
    )
    collapsed_console = Console(record=True, width=80)
    collapsed_console.print(render_chat_item(item, show_tool_results=False))
    collapsed = collapsed_console.export_text()
    expanded_console = Console(record=True, width=80)
    expanded_console.print(render_chat_item(item, show_tool_results=True))
    expanded = expanded_console.export_text()

    assert "Branch summary (Ctrl+O to expand)" in collapsed
    assert "Detailed summary text" not in collapsed
    assert "Branch Summary" in expanded
    assert "Detailed summary text" in expanded


def test_compaction_summary_chat_items_expand_with_tool_results_toggle() -> None:
    # [FUNCTION DOC] Verifies compaction-summary items show a placeholder when collapsed and a "Compaction
    # Summary" heading plus text when expanded.
    item = ChatItem(
        role="compaction_summary",
        text="Compaction summary (Ctrl+O to expand)",
        tool_result_text="Detailed compaction text",
    )
    collapsed_console = Console(record=True, width=80)
    collapsed_console.print(render_chat_item(item, show_tool_results=False))
    collapsed = collapsed_console.export_text()
    expanded_console = Console(record=True, width=80)
    expanded_console.print(render_chat_item(item, show_tool_results=True))
    expanded = expanded_console.export_text()

    assert "Compaction summary (Ctrl+O to expand)" in collapsed
    assert "Detailed compaction text" not in collapsed
    assert "Compaction Summary" in expanded
    assert "Detailed compaction text" in expanded


def test_tui_state_indexes_tool_items_and_clears_index() -> None:
    # [FUNCTION DOC] Verifies TuiState indexes tool items by call id and clear() drops that index.
    state = TuiState()
    tool_call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})

    state.add_tool_call(tool_call)
    item = state.items[-1]

    assert state.find_tool_item("call-1") is item
    state.clear()
    assert state.find_tool_item("call-1") is None


def test_tui_state_compacts_branch_summary_messages() -> None:
    # [FUNCTION DOC] Verifies a branch-summary user message loads as a collapsed branch_summary item with the
    # inner summary text as its expandable result.
    state = tui_app.TuiState()

    state.load_messages(
        [
            UserMessage(
                content=(
                    "The following is a summary of a branch that this conversation "
                    "came back from:\n<summary>\nImportant context.\n</summary>"
                )
            )
        ]
    )

    assert [(item.role, item.text, item.tool_result_text) for item in state.items] == [
        ("branch_summary", "Branch summary (Ctrl+O to expand)", "Important context.")
    ]


def test_tui_state_compacts_compaction_summary_messages() -> None:
    # [FUNCTION DOC] Verifies a "Previous conversation summary:" user message loads as a compaction_summary item.
    state = tui_app.TuiState()

    state.load_messages(
        [UserMessage(content="Previous conversation summary:\nCompacted prior work.")]
    )

    assert [(item.role, item.text, item.tool_result_text) for item in state.items] == [
        (
            "compaction_summary",
            "Compaction summary (Ctrl+O to expand)",
            "Compacted prior work.",
        )
    ]


def test_tui_state_compacts_expanded_skill_messages() -> None:
    # [FUNCTION DOC] Verifies an expanded skill invocation message loads as a compact "Using skill" item
    # followed by the user's own text.
    skill = Skill(
        name="review",
        path=Path("/workspace/.tau/skills/review.md"),
        content="# Review\nFull noisy instructions.",
        description="Review code",
    )
    state = tui_app.TuiState()

    state.load_messages(
        [
            UserMessage(
                content=format_skill_invocation(
                    skill,
                    "check the auth flow",
                )
            )
        ]
    )

    assert [(item.role, item.text) for item in state.items] == [
        ("skill", "Using skill: review"),
        ("user", "check the auth flow"),
    ]


def test_tui_state_renders_restored_skill_file_reads_with_skill_style() -> None:
    # [FUNCTION DOC] Verifies a restored read tool call targeting a known skill file is shown as a skill item
    # ("Loading skill: NAME") with the read result as expandable text.
    skill = Skill(
        name="review",
        path=Path("/workspace/.tau/skills/review.md"),
        content="# Review\nFull noisy instructions.",
        description="Review code",
    )
    state = tui_app.TuiState(skills=(skill,))

    state.load_messages(
        [
            AssistantMessage(
                content=assistant_content(
                    "Reading skill.",
                    [
                        ToolCall(
                            id="call-1",
                            name="read",
                            arguments={"path": "/workspace/.tau/skills/review.md"},
                        )
                    ],
                )
            ),
            ToolResultMessage(
                tool_call_id="call-1",
                tool_name="read",
                content=[TextContent(text="# Review\nFull noisy instructions.")],
            ),
        ]
    )

    assert [(item.role, item.text, item.tool_result_text) for item in state.items] == [
        ("assistant", "Reading skill.", None),
        ("skill", "Loading skill: review", "✓ read\n# Review\nFull noisy instructions."),
    ]


def test_light_theme_tool_success_uses_dark_text_without_background() -> None:
    # [FUNCTION DOC] Verifies light-theme tool success text uses dark green foreground without a background color.
    console = Console(record=True, width=80)
    console.print(
        render_chat_item(
            ChatItem(role="tool", text="→ read README.md", tool_result_text="✓ read\ncontents"),
            theme=TAU_LIGHT_THEME,
            show_tool_results=True,
        )
    )

    output = console.export_text(styles=True)

    assert "38;2;22;101;52" in output
    assert "38;2;22;101;52;48;2" not in output


def test_light_theme_tool_error_uses_red_text_without_background() -> None:
    # [FUNCTION DOC] Verifies light-theme tool error text uses red foreground without a background color.
    console = Console(record=True, width=80)
    console.print(
        render_chat_item(
            ChatItem(role="tool", text="$ false", tool_result_text="✗ bash\nfailed"),
            theme=TAU_LIGHT_THEME,
            show_tool_results=True,
        )
    )

    output = console.export_text(styles=True)

    assert "38;2;185;28;28" in output
    assert "38;2;185;28;28;48;2" not in output


def test_dark_theme_markdown_code_uses_aqua_highlight() -> None:
    # [FUNCTION DOC] Verifies dark-theme inline Markdown code uses the aqua color, not the old orange.
    console = Console(record=True, width=80)
    console.print(render_chat_item(ChatItem(role="assistant", text="Use `tau` here.")))

    output = console.export_text(styles=True)

    assert "38;2;117;158;149" in output
    assert "38;2;219;148;90" not in output


def test_assistant_markdown_titles_use_highlight_color_and_left_alignment() -> None:
    # [FUNCTION DOC] Verifies assistant Markdown headings use the theme heading color and are left aligned.
    console = Console(record=True, width=60, color_system="truecolor")
    console.print(render_chat_item(ChatItem(role="assistant", text="# Title\n\n## Header")))

    output = console.export_text(styles=True)
    plain_output = _strip_ansi(output)

    assert _style_color_escape(TAU_DARK_THEME.markdown_heading) in output
    assert "Title" in plain_output
    assert not plain_output.splitlines()[1].startswith(" " * 20)
    assert LeftAlignedMarkdownHeading.LEVEL_ALIGN["h1"] == "left"


def test_dark_theme_markdown_links_use_theme_link_color() -> None:
    # [FUNCTION DOC] Verifies Markdown links use the dark theme link color.
    console = Console(record=True, width=80, color_system="truecolor")
    console.print(
        render_chat_item(ChatItem(role="assistant", text="Read [docs](https://example.com)."))
    )

    output = console.export_text(styles=True)

    assert "38;2;147;197;253" in output


def test_dark_theme_markdown_bullets_use_theme_bullet_color() -> None:
    # [FUNCTION DOC] Verifies Markdown bullets use the theme bullet color.
    console = Console(record=True, width=80, color_system="truecolor")
    console.print(render_chat_item(ChatItem(role="assistant", text="- first\n- second")))

    output = console.export_text(styles=True)

    assert _style_color_escape(TAU_DARK_THEME.markdown_bullet) in output


def test_markdown_tables_use_highlight_color_for_headers() -> None:
    # [FUNCTION DOC] Verifies Markdown table headers use the theme accent color rather than default cyan.
    console = Console(record=True, width=80, color_system="truecolor")
    console.print(
        render_chat_item(
            ChatItem(role="assistant", text="| Name | Value |\n| --- | --- |\n| A | B |")
        )
    )

    output = console.export_text(styles=True)

    assert _style_color_escape(TAU_DARK_THEME.accent) in output
    assert "\x1b[36" not in output


@pytest.mark.anyio
async def test_textual_markdown_widget_uses_theme_link_style() -> None:
    # [FUNCTION DOC] Async (anyio): verifies the Textual Markdown widget in the running app adopts the theme
    # link style, has no default link style, underlines on hover, and marks only the link text
    # span as clickable.
    app = TauTuiApp(
        FakeSession([AssistantMessage(content="Read [docs](https://example.com).")]),
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        markdown = app.query_one(ThemedMarkdownWidget)
        block = app.query_one(TauMarkdownBlock)

    link_spans = [
        span
        for span in block.content.spans
        if isinstance(span.style, TextualStyle) and "@click" in span.style.meta
    ]
    assert markdown.tau_link_style == TAU_DARK_THEME.markdown_link
    assert not block.styles.link_style
    assert block.styles.link_style_hover.underline is True
    assert [(span.start, span.end) for span in link_spans] == [(5, 9)]
    assert block.content.plain[5:9] == "docs"


@pytest.mark.anyio
async def test_transcript_markdown_paragraphs_keep_bottom_margin() -> None:
    # [FUNCTION DOC] Async: verifies each transcript Markdown paragraph block keeps a bottom margin of 1.
    app = TauTuiApp(
        FakeSession([AssistantMessage(content="First paragraph.\n\nSecond paragraph.")]),
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        blocks = list(app.query(TauMarkdownBlock))

    assert len(blocks) == 2
    assert [block.styles.margin.bottom for block in blocks] == [1, 1]


def test_system_prompt_markdown_highlights_markup_tags_as_inline_code() -> None:
    # [FUNCTION DOC] Verifies the system-prompt Markdown converter wraps XML-like tag lines in inline code.
    prompt = (
        '<project_instructions path="/workspace/AGENTS.md">\nUse `rg`.\n</project_instructions>'
    )

    rendered = _system_prompt_markdown(prompt)

    assert rendered == (
        '`<project_instructions path="/workspace/AGENTS.md">`\nUse `rg`.\n`</project_instructions>`'
    )


def test_system_prompt_markdown_skips_tags_inside_fenced_code() -> None:
    # [FUNCTION DOC] Verifies tags inside fenced code blocks are left untouched.
    prompt = "```xml\n<project>\n```\n\n<visible>"

    assert _system_prompt_markdown(prompt) == "```xml\n<project>\n```\n\n`<visible>`"


def test_system_prompt_markdown_skips_tags_inside_existing_inline_code() -> None:
    # [FUNCTION DOC] Verifies tags already inside inline code spans are left untouched.
    prompt = "Use `<project>` as an example, then use <visible>."

    assert _system_prompt_markdown(prompt) == "Use `<project>` as an example, then use `<visible>`."


def test_system_prompt_markdown_uses_longer_delimiter_for_backticks_in_tags() -> None:
    # [FUNCTION DOC] Verifies a tag containing a backtick is wrapped with a longer double-backtick delimiter.
    prompt = '<project value="a`b">'

    assert _system_prompt_markdown(prompt) == '``<project value="a`b">``'


def test_system_prompt_markdown_preserves_markdown_autolinks() -> None:
    # [FUNCTION DOC] Verifies URL and email autolinks are preserved while real tags are wrapped.
    prompt = "<https://example.com> <user@example.com> <project>"

    assert _system_prompt_markdown(prompt) == (
        "<https://example.com> <user@example.com> `<project>`"
    )


def test_system_prompt_markdown_skips_tags_in_indented_code_blocks() -> None:
    # [FUNCTION DOC] Verifies tags in indented (four-space) code blocks are left untouched.
    prompt = "    <project>\n\n<visible>"

    assert _system_prompt_markdown(prompt) == "    <project>\n\n`<visible>`"


def test_system_prompt_markdown_preserves_uri_autolinks_without_double_slashes() -> None:
    # [FUNCTION DOC] Verifies scheme autolinks without "//" (http:, tel:, urn:) are preserved.
    prompt = "<http:foo> <tel:123456> <urn:isbn:9780141036144> <project>"

    assert _system_prompt_markdown(prompt) == (
        "<http:foo> <tel:123456> <urn:isbn:9780141036144> `<project>`"
    )


def test_textual_markdown_uses_theme_highlight_and_aqua_inline_code() -> None:
    # [FUNCTION DOC] Verifies the generated Textual CSS variables expose the theme's Markdown colors.
    variables = _theme_css_variables(TAU_LIGHT_THEME)

    assert variables["tau-markdown-highlight"] == TAU_LIGHT_THEME.markdown_heading
    assert variables["tau-markdown-table-header"] == TAU_LIGHT_THEME.markdown_table_header
    assert variables["tau-markdown-table-border"] == TAU_LIGHT_THEME.markdown_table_border
    assert variables["tau-markdown-inline-code"] == TAU_LIGHT_THEME.markdown_inline_code
    assert (
        variables["tau-markdown-code-block-background"]
        == TAU_LIGHT_THEME.markdown_code_block_background
    )
    assert variables["tau-markdown-link"] == TAU_LIGHT_THEME.markdown_link
    assert variables["tau-markdown-bullet"] == TAU_LIGHT_THEME.markdown_bullet


def test_light_theme_markdown_code_uses_aqua_without_background() -> None:
    # [FUNCTION DOC] Verifies light-theme inline code uses teal foreground without a background.
    console = Console(record=True, width=80)
    console.print(
        render_chat_item(
            ChatItem(role="assistant", text="Use `tau` here."),
            theme=TAU_LIGHT_THEME,
        )
    )

    output = console.export_text(styles=True)

    assert "38;2;15;118;110" in output
    assert "38;2;15;118;110;48;2" not in output


def test_expanded_tool_invocation_blank_line_stays_separate_from_result() -> None:
    # [FUNCTION DOC] Verifies an expanded tool invocation containing blank lines and diff-like text stays
    # separate from the result and is not diff-colored.
    invocation = "$ python <<'PY'\n\nPatch:\n-old\nPY"
    item = ChatItem(
        role="tool",
        text="$ compact",
        tool_result_text="✓ bash\nfinished",
    )
    body = _transcript_plain_body_text(
        item,
        text=f"{invocation}\n\n{item.tool_result_text}",
        body_style=TAU_DARK_THEME.role_styles["tool"].body,
        theme=TAU_DARK_THEME,
        show_tool_results=True,
        invocation=invocation,
    )

    console = Console(record=True, width=100, color_system="truecolor")
    console.print(body)

    assert console.export_text(clear=False) == f"{invocation}\n\n✓ bash\nfinished\n"
    assert "\x1b[91" not in console.export_text(styles=True)


def test_pending_tool_invocation_colors_tool_name_but_not_arguments() -> None:
    # [FUNCTION DOC] Verifies a pending tool invocation colors only the tool name with the running accent,
    # leaving arguments in the body color.
    console = Console(record=True, width=80)
    item = ChatItem(role="tool", text="→ read README.md")
    console.print(
        _transcript_plain_body_text(
            item,
            text=item.text,
            body_style=TAU_DARK_THEME.role_styles["tool"].body,
            theme=TAU_DARK_THEME,
        )
    )

    output = console.export_text(styles=True)

    accent = "38;2;138;122;82;48;2;0;0;0m"
    body = "38;2;203;213;225;48;2;0;0;0m"
    assert f"{accent}read" in output
    assert f"{body} README.md" in output
    assert f"{accent} README.md" not in output


def test_tool_chat_items_color_description_not_details_or_results() -> None:
    # [FUNCTION DOC] Verifies successful/failed tool items color only the tool name by status (green/red);
    # arguments, status lines and result text stay neutral.
    success_console = Console(record=True, width=80)
    success_console.print(
        render_chat_item(
            ChatItem(role="tool", text="→ read README.md", tool_result_text="✓ read\ncontents"),
            show_tool_results=True,
        )
    )
    success_output = success_console.export_text(styles=True)

    error_console = Console(record=True, width=80)
    error_console.print(
        render_chat_item(
            ChatItem(role="tool", text="$ false", tool_result_text="✗ bash\nfailed"),
            show_tool_results=True,
        )
    )
    error_output = error_console.export_text(styles=True)

    green = "38;2;156;255;177"
    red = "38;2;255;79;79"
    white = "38;2;203;213;225"

    assert green in success_output
    assert f"{green};48;2;0;0;0mread" in success_output
    assert f"{white};48;2;0;0;0m README.md" in success_output
    assert f"{green};48;2;0;0;0m README.md" not in success_output
    assert f"{green};48;2;0;0;0m✓ read" not in success_output
    assert f"{green};48;2;0;0;0mcontents" not in success_output

    assert red in error_output
    assert f"{white};48;2;0;0;0m✗ bash" in error_output
    assert f"{red};48;2;0;0;0m✗ bash" not in error_output
    assert f"{red};48;2;0;0;0mfailed" not in error_output


def test_grouped_read_details_stay_neutral() -> None:
    # [FUNCTION DOC] Verifies grouped read summaries color the header line but leave file-list lines neutral.
    green = "38;2;156;255;177;48;2;0;0;0m"
    body = "38;2;203;213;225;48;2;0;0;0m"
    text = "→ Read 5 files\n  - a.py\n  - b.py\n  - c.py\n  - d.py\n  - e.py"
    console = Console(record=True, width=100, color_system="truecolor")
    item = ChatItem(role="tool", text=text, tool_result_text="✓ tool")
    console.print(
        _transcript_plain_body_text(
            item,
            text=text,
            body_style=TAU_DARK_THEME.role_styles["tool"].body,
            theme=TAU_DARK_THEME,
        )
    )
    output = console.export_text(styles=True)

    assert f"{green}Read 5 files" in output
    assert f"{body}  - a.py" in output
    assert f"{body}  - e.py" in output
    assert f"{green}  - a.py" not in output


def test_grouped_write_paths_stay_neutral() -> None:
    # [FUNCTION DOC] Verifies grouped write summaries color the header but leave path lines neutral.
    green = "38;2;156;255;177;48;2;0;0;0m"
    body = "38;2;203;213;225;48;2;0;0;0m"
    text = "→ Written 2 files\n  - a.py\n  - b.py"
    console = Console(record=True, width=100, color_system="truecolor")
    item = ChatItem(role="tool", text=text, tool_name="write", tool_result_text="✓ write group")
    console.print(
        _transcript_plain_body_text(
            item,
            text=text,
            body_style=TAU_DARK_THEME.role_styles["tool"].body,
            theme=TAU_DARK_THEME,
        )
    )
    output = console.export_text(styles=True)

    assert f"{green}Written 2 files" in output
    assert f"{body}  - a.py" in output
    assert f"{green}  - a.py" not in output


def test_bash_description_without_command_keeps_full_status_color() -> None:
    # [FUNCTION DOC] Verifies a bash item showing its description colors the whole description by status and
    # does not display the long command.
    command = "echo " + "x" * 120
    item = ChatItem(
        role="tool",
        text="→ Running long command",
        tool_name="bash",
        tool_arguments={"command": command, "description": "Running long command"},
        tool_result_text="✓ bash",
    )
    console = Console(record=True, width=100, color_system="truecolor")
    console.print(
        _transcript_plain_body_text(
            item,
            text=item.text,
            body_style=TAU_DARK_THEME.role_styles["tool"].body,
            theme=TAU_DARK_THEME,
        )
    )

    output = console.export_text(styles=True)
    assert "38;2;156;255;177;48;2;0;0;0mRunning long command" in output
    assert command not in console.export_text()


def test_tool_batch_colors_each_description_by_its_own_status() -> None:
    # [FUNCTION DOC] Verifies each entry in a tool batch is colored by its own status (success green, failure
    # red, running accent).
    item = ChatItem(
        role="tool",
        text="batch",
        tool_result_text="✗ tool batch",
        tool_batch_items=[
            ChatItem(
                role="tool",
                text="→ Finished action",
                tool_name="bash",
                tool_result_text="✓ bash",
            ),
            ChatItem(
                role="tool",
                text="→ Failed action",
                tool_name="bash",
                tool_result_text="✗ bash",
            ),
            ChatItem(
                role="tool",
                text="→ Running action",
                tool_name="bash",
                started_at=1.0,
            ),
        ],
    )
    console = Console(record=True, width=100, color_system="truecolor")
    console.print(
        _transcript_plain_body_text(
            item,
            text=item.text,
            body_style=TAU_DARK_THEME.role_styles["tool"].body,
            theme=TAU_DARK_THEME,
        )
    )
    output = console.export_text(styles=True)

    assert "38;2;156;255;177;48;2;0;0;0mFinished action" in output
    assert "38;2;255;79;79;48;2;0;0;0mFailed action" in output
    assert "38;2;138;122;82;48;2;0;0;0mRunning action" in output
    assert "$ false" not in console.export_text()


def test_partially_completed_read_group_keeps_running_color() -> None:
    # [FUNCTION DOC] Verifies a partially complete grouped read still uses the running (border) color.
    item = ChatItem(
        role="tool",
        text="→ Reading 2 files · 1/2 complete\n  - a.py\n  - b.py",
        tool_name="read",
        tool_result_text="… read group",
        started_at=1.0,
    )
    console = Console(record=True, width=100, color_system="truecolor")
    console.print(
        _transcript_plain_body_text(
            item,
            text=item.text,
            body_style=TAU_DARK_THEME.role_styles["tool"].body,
            theme=TAU_DARK_THEME,
        )
    )

    output = console.export_text(styles=True)
    running_color = _style_color_escape(TAU_DARK_THEME.role_styles["tool"].border)
    assert f"{running_color};48;2;0;0;0mReading 2 files" in output


def test_tool_batch_body_stays_one_selectable_text_renderable() -> None:
    # [FUNCTION DOC] Verifies a tool batch body is a single Rich Text whose plain text joins each entry line,
    # so the batch remains one selectable renderable.
    item = ChatItem(
        role="tool",
        text="batch",
        tool_batch_items=[
            ChatItem(role="tool", text="→ First action", tool_name="bash"),
            ChatItem(role="tool", text="→ Second action", tool_name="bash"),
        ],
    )

    body = _transcript_plain_body_text(
        item,
        text=item.text,
        body_style=TAU_DARK_THEME.role_styles["tool"].body,
        theme=TAU_DARK_THEME,
    )

    assert isinstance(body, Text)
    assert body.plain == "→ First action\n→ Second action"


def test_assistant_chat_items_render_markdown_lists() -> None:
    # [FUNCTION DOC] Verifies assistant Markdown lists render as bullets.
    console = Console(record=True, width=60)
    item = ChatItem(role="assistant", text="Plan:\n\n- inspect\n- patch")

    console.print(render_chat_item(item))
    output = console.export_text()

    assert "Plan:" in output
    assert "• inspect" in output
    assert "• patch" in output
    assert "- inspect" not in output


def test_assistant_chat_items_render_markdown_tables() -> None:
    # [FUNCTION DOC] Verifies assistant Markdown tables render cells without the separator row.
    console = Console(record=True, width=60)
    item = ChatItem(
        role="assistant",
        text="| File | Status |\n| --- | --- |\n| README.md | updated |",
    )

    console.print(render_chat_item(item))
    output = console.export_text()

    assert "File" in output
    assert "Status" in output
    assert "README.md" in output
    assert "updated" in output
    assert "---" not in output


def test_user_chat_items_keep_markdown_literal() -> None:
    # [FUNCTION DOC] Verifies user messages keep Markdown syntax literal (no bullet conversion).
    console = Console(record=True, width=60)
    item = ChatItem(role="user", text="- keep this literal")

    console.print(render_chat_item(item))
    output = console.export_text()

    assert "- keep this literal" in output
    assert "• keep this literal" not in output


def test_chat_items_preserve_malformed_fenced_code() -> None:
    # [FUNCTION DOC] Verifies an unterminated code fence is preserved literally rather than dropped.
    console = Console(record=True, width=60)
    item = ChatItem(role="assistant", text='```python\nprint("hi")')

    console.print(render_chat_item(item))
    output = console.export_text()

    assert "```python" in output
    assert 'print("hi")' in output


@pytest.mark.anyio
async def test_transcript_message_widget_extracts_plain_text_selection() -> None:
    # [FUNCTION DOC] Async: verifies a mounted transcript message widget extracts the plain text of a
    # character-range selection.
    app = TauTuiApp(
        FakeSession(
            messages=[
                UserMessage(content="alpha beta\ngamma"),
            ]
        )
    )

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        widget = app.query_one(TranscriptMessageWidget)

        assert widget.get_selection(Selection(Offset(6, 0), Offset(10, 0))) == (
            "beta",
            "\n",
        )


@pytest.mark.anyio
async def test_transcript_message_widget_renders_full_height_role_block() -> None:
    # [FUNCTION DOC] Async: verifies a multi-line user message widget is a full-height block: role background
    # on container and body, vertical padding, a left border in the role color, and whole-widget
    # selection returning the original text. Uses the high-contrast theme.
    plain_text = "alpha beta gamma\nsecond line\nthird line"
    app = TauTuiApp(
        FakeSession(messages=[UserMessage(content=plain_text)]),
        tui_settings=TuiSettings(theme="high-contrast"),
    )

    role_style = HIGH_CONTRAST_THEME.role_styles["user"]
    _, expected_background = _split_rich_style_colors(role_style.body)
    assert expected_background == HIGH_CONTRAST_THEME.prompt_background
    background = Color.parse(expected_background)
    border = Color.parse(role_style.border)

    async with app.run_test(size=(60, 30)) as pilot:
        await pilot.pause()
        widget = app.query_one(TranscriptMessageWidget)
        body = widget.query_one(".transcript-message-body")

        # A multi-line message must occupy more than a single row so the accent
        # and background have to span the full message height.
        assert widget.size.height > 1

        # The container owns the role background and a real left border, so the
        # block is rectangular and the accent spans every wrapped line.
        assert widget.styles.background == background
        assert body.styles.background == background
        assert widget.styles.padding.top == 1
        assert widget.styles.padding.bottom == 1
        edge_type, edge_color = widget.styles.border_left
        assert edge_type != "none"
        assert edge_color == border

        # Selecting the whole message still yields the original plain text.
        assert widget.get_selection(SELECT_ALL) == (plain_text, "\n")


@pytest.mark.anyio
async def test_streaming_transcript_applies_role_foreground() -> None:
    # [FUNCTION DOC] Async: verifies streamed thinking and assistant widgets apply their role foreground colors
    # immediately while streaming.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        transcript = app.query_one("#transcript", TranscriptView)

        thinking_fg, _ = _split_rich_style_colors(TAU_DARK_THEME.role_styles["thinking"].body)
        assistant_fg, _ = _split_rich_style_colors(TAU_DARK_THEME.role_styles["assistant"].body)

        # Streamed thinking is dimmed immediately, matching the finalized block
        # instead of shifting color on the next redraw.
        await transcript.append_thinking_delta(
            "reasoning", theme=TAU_DARK_THEME, show_thinking=True
        )
        await pilot.pause()
        thinking = next(
            w for w in app.query(StreamingTranscriptMessageWidget) if w.item.role == "thinking"
        )
        assert thinking.styles.color == Color.parse(thinking_fg)

        await transcript.append_assistant_delta("answer", theme=TAU_DARK_THEME)
        await pilot.pause()
        assistant = next(
            w for w in app.query(StreamingTranscriptMessageWidget) if w.item.role == "assistant"
        )
        assert assistant.styles.color == Color.parse(assistant_fg)


@pytest.mark.anyio
async def test_tool_execution_updates_render_in_place() -> None:
    # [FUNCTION DOC] Async: verifies agent tool execution updates replace the progress line in one tool widget
    # in place, and the end event clears the transient progress text.
    app = TauTuiApp(FakeSession())

    async def stream(event: AgentEvent) -> None:
        # [FUNCTION DOC] HELPER: applies an agent event to the app's adapter (state) and then to the streaming
        # transcript, mirroring the app's live event path. Async; mutates app state and widgets.
        app.adapter.apply(event)
        await app._apply_streaming_transcript_event(event)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await stream(
            ToolExecutionStartEvent(
                tool_call_id="call-1", tool_name="agent", args={"prompt": "explore"}
            )
        )
        await stream(
            ToolExecutionUpdateEvent(
                tool_call_id="call-1",
                tool_name="agent",
                args={},
                partial_result=AgentToolResult(content="agent-1: bash · turn 1"),
            )
        )
        await pilot.pause()

        tool_widgets = [w for w in app.query(TranscriptMessageWidget) if w.item.role == "tool"]
        assert len(tool_widgets) == 1
        assert "agent-1: bash · turn 1" in tool_widgets[0].selection_text

        # A later update replaces the progress line instead of appending a block.
        await stream(
            ToolExecutionUpdateEvent(
                tool_call_id="call-1",
                tool_name="agent",
                args={},
                partial_result=AgentToolResult(content="agent-1: turn 2 done"),
            )
        )
        await pilot.pause()
        tool_widgets = [w for w in app.query(TranscriptMessageWidget) if w.item.role == "tool"]
        assert len(tool_widgets) == 1
        assert "agent-1: turn 2 done" in tool_widgets[0].selection_text
        assert "turn 1" not in tool_widgets[0].selection_text

        # The final result clears the transient progress line.
        await stream(
            ToolExecutionEndEvent(
                tool_call_id="call-1",
                tool_name="agent",
                result=AgentToolResult(content="report"),
                is_error=False,
            )
        )
        await pilot.pause()
        tool_widgets = [w for w in app.query(TranscriptMessageWidget) if w.item.role == "tool"]
        assert len(tool_widgets) == 1
        assert "turn 2 done" not in tool_widgets[0].selection_text


@pytest.mark.anyio
async def test_batched_reads_share_one_live_transcript_row() -> None:
    # [FUNCTION DOC] Async: verifies parallel read calls share a single live transcript row that shows
    # "Reading N files", progress, then "Read N files", and expands (ctrl+o) into per-call lines
    # without read output.
    app = TauTuiApp(FakeSession())

    async def stream(event: AgentEvent) -> None:
        # [FUNCTION DOC] HELPER: applies an event to the adapter then to the streaming transcript (see above).
        app.adapter.apply(event)
        await app._apply_streaming_transcript_event(event)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await stream(
            MessageEndEvent(
                message=AssistantMessage(
                    content=[
                        ToolCall(id="call-1", name="read", arguments={"path": "a.py"}),
                        ToolCall(id="call-2", name="read", arguments={"path": "b.py"}),
                    ]
                )
            )
        )
        await stream(
            ToolExecutionStartEvent(tool_call_id="call-1", tool_name="read", args={"path": "a.py"})
        )
        await stream(
            ToolExecutionStartEvent(tool_call_id="call-2", tool_name="read", args={"path": "b.py"})
        )
        await pilot.pause()

        tool_widgets = [w for w in app.query(TranscriptMessageWidget) if w.item.role == "tool"]
        assert len(tool_widgets) == 1
        assert tool_widgets[0].selection_text == "→ Reading 2 files\n  - a.py\n  - b.py"

        await stream(
            ToolExecutionEndEvent(
                tool_call_id="call-1",
                tool_name="read",
                result=AgentToolResult(content="one"),
                is_error=False,
            )
        )
        await pilot.pause()
        assert tool_widgets[0].selection_text == (
            "→ Reading 2 files · 1/2 complete\n  - a.py\n  - b.py"
        )

        await stream(
            ToolExecutionEndEvent(
                tool_call_id="call-2",
                tool_name="read",
                result=AgentToolResult(content="two"),
                is_error=False,
            )
        )
        await pilot.pause()
        assert tool_widgets[0].selection_text == "→ Read 2 files\n  - a.py\n  - b.py"

        await pilot.press("ctrl+o")
        await pilot.pause()
        assert tool_widgets[0].selection_text == "→ read a.py\n→ read b.py"
        assert "one" not in tool_widgets[0].selection_text
        assert "two" not in tool_widgets[0].selection_text


@pytest.mark.anyio
async def test_mixed_tool_batch_uses_one_widget_and_expands_each_row() -> None:
    # [FUNCTION DOC] Async: verifies a restored mixed batch (bash, reads, bash) renders as one tool widget and
    # ctrl+o expands each row, showing bash commands/results while grouped reads omit contents.
    app = TauTuiApp(
        FakeSession(
            messages=[
                AssistantMessage(
                    content=[
                        ToolCall(
                            id="bash-1",
                            name="bash",
                            arguments={"command": "echo one", "description": "Doing thing one"},
                        ),
                        ToolCall(id="read-1", name="read", arguments={"path": "a.py"}),
                        ToolCall(id="read-2", name="read", arguments={"path": "b.py"}),
                        ToolCall(
                            id="bash-2",
                            name="bash",
                            arguments={"command": "echo two", "description": "Doing thing two"},
                        ),
                    ]
                ),
                ToolResultMessage(tool_call_id="bash-1", tool_name="bash", content="one"),
                ToolResultMessage(tool_call_id="read-1", tool_name="read", content="alpha"),
                ToolResultMessage(tool_call_id="read-2", tool_name="read", content="beta"),
                ToolResultMessage(tool_call_id="bash-2", tool_name="bash", content="two"),
            ]
        )
    )

    async with app.run_test(size=(120, 30)) as pilot:
        widget = next(w for w in app.query(TranscriptMessageWidget) if w.item.role == "tool")
        assert len([w for w in app.query(TranscriptMessageWidget) if w.item.role == "tool"]) == 1
        assert widget.selection_text == (
            "→ Doing thing one\n→ Read 2 files\n  - a.py\n  - b.py\n→ Doing thing two"
        )

        await pilot.press("ctrl+o")
        await pilot.pause()

        assert widget.selection_text == (
            "→ Doing thing one\n$ echo one\n\n✓ bash\none\n\n"
            "→ read a.py\n→ read b.py\n\n"
            "→ Doing thing two\n$ echo two\n\n✓ bash\ntwo"
        )
        assert "alpha" not in widget.selection_text
        assert "beta" not in widget.selection_text


@pytest.mark.anyio
async def test_tool_completion_updates_row_without_redrawing_history() -> None:
    # [FUNCTION DOC] Async: verifies a tool completion updates the existing pending tool widget in place
    # (new border color, result text) without remounting earlier history widgets.
    app = TauTuiApp(FakeSession(messages=[UserMessage(content="earlier")]))

    async def stream(event: AgentEvent) -> None:
        # [FUNCTION DOC] HELPER: applies an event to the adapter then to the streaming transcript.
        app.adapter.apply(event)
        await app._apply_streaming_transcript_event(event)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        app.state.show_tool_results = True
        history_widget = next(
            widget for widget in app.query(TranscriptMessageWidget) if widget.item.text == "earlier"
        )
        await stream(
            ToolExecutionStartEvent(
                tool_call_id="call-1",
                tool_name="read",
                args={"path": "README.md"},
            )
        )
        pending_tool_widget = next(
            widget for widget in app.query(TranscriptMessageWidget) if widget.item.role == "tool"
        )
        pending_border_color = pending_tool_widget.styles.border_left[1]
        await stream(
            ToolExecutionEndEvent(
                tool_call_id="call-1",
                tool_name="read",
                result=AgentToolResult(content="contents"),
                is_error=False,
            )
        )
        await pilot.pause()

        assert history_widget.parent is app.query_one("#transcript", TranscriptView)
        tool_widget = next(
            widget for widget in app.query(TranscriptMessageWidget) if widget.item.role == "tool"
        )
        assert tool_widget is pending_tool_widget
        assert tool_widget.styles.border_left[1] != pending_border_color
        assert "✓ read" in tool_widget.selection_text
        assert "contents" in tool_widget.selection_text


@pytest.mark.anyio
async def test_assistant_message_renders_without_role_block() -> None:
    # [FUNCTION DOC] Async: verifies assistant messages render without left border or role background.
    app = TauTuiApp(
        FakeSession([AssistantMessage(content="line one\nline two")]),
        tui_settings=TuiSettings(theme="high-contrast"),
    )

    async with app.run_test(size=(60, 30)) as pilot:
        await pilot.pause()
        widget = next(w for w in app.query(TranscriptMessageWidget) if w.item.role == "assistant")
        body = widget.query_one(".transcript-message-body")

        # Assistant output flows as plain prose: no left accent and no role
        # background, so it reads the same while streaming and once finalized.
        assert not widget.styles.has_rule("border_left")
        assert widget.styles.background.a == 0
        assert body.styles.background.a == 0


@pytest.mark.anyio
async def test_long_transcript_mounts_bounded_latest_window_and_pages_earlier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC] Async: verifies a long transcript mounts only the latest window (window size patched to
    # 6, page size 2) with an "earlier" boundary, and scrolling to the top pages earlier items
    # in while keeping the mounted widget count bounded with both boundaries.
    # Inputs: monkeypatch (patches widget module constants).
    from tau_coding.tui import widgets as tui_widgets

    monkeypatch.setattr(tui_widgets, "TRANSCRIPT_WINDOW_ITEMS", 6)
    monkeypatch.setattr(tui_widgets, "TRANSCRIPT_WINDOW_PAGE_ITEMS", 2)
    app = TauTuiApp(
        FakeSession(messages=[UserMessage(content=f"message {index}") for index in range(12)])
    )

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        transcript = app.query_one("#transcript", TranscriptView)
        assert transcript._window_start == 6
        assert transcript._window_end == 12
        assert [widget.item.text for widget in app.query(TranscriptMessageWidget)] == [
            f"message {index}" for index in range(6, 12)
        ]
        [top_boundary] = list(app.query(TranscriptWindowBoundary))
        assert top_boundary.direction == "earlier"

        transcript.scroll_to(y=0, animate=False, immediate=True)
        await pilot.pause()

        assert transcript._window_start == 4
        assert transcript._window_end == 10
        assert [widget.item.text for widget in app.query(TranscriptMessageWidget)] == [
            f"message {index}" for index in range(4, 10)
        ]
        assert {boundary.direction for boundary in app.query(TranscriptWindowBoundary)} == {
            "earlier",
            "later",
        }
        assert len(transcript._item_widgets) == 6


@pytest.mark.anyio
async def test_long_transcript_incremental_appends_keep_mounted_window_bounded() -> None:
    # [FUNCTION DOC] Async: verifies incremental appends past the window keep the mounted widget count within
    # window plus overscan while state still holds every item.
    initial_count = TRANSCRIPT_WINDOW_ITEMS + TRANSCRIPT_WINDOW_OVERSCAN_ITEMS
    app = TauTuiApp(
        FakeSession(
            messages=[UserMessage(content=f"message {index}") for index in range(initial_count)]
        )
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        transcript = app.query_one("#transcript", TranscriptView)
        for index in range(TRANSCRIPT_WINDOW_OVERSCAN_ITEMS + 1):
            app.state.add_item("user", f"new message {index}")
            await transcript.append_item(app.state.items[-1], scroll_end=True)
        await pilot.pause()

        mounted = list(app.query(TranscriptMessageWidget))
        assert len(mounted) <= TRANSCRIPT_WINDOW_ITEMS + TRANSCRIPT_WINDOW_OVERSCAN_ITEMS
        assert mounted[-1].item.text == f"new message {TRANSCRIPT_WINDOW_OVERSCAN_ITEMS}"
        assert len(app.state.items) == initial_count + TRANSCRIPT_WINDOW_OVERSCAN_ITEMS + 1


@pytest.mark.anyio
async def test_transcript_resize_preserves_mounted_message_widgets() -> None:
    # [FUNCTION DOC] Async: verifies resizing the terminal keeps the same mounted transcript widget instances.
    app = TauTuiApp(
        FakeSession(messages=[AssistantMessage(content=f"answer {index}") for index in range(20)])
    )

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        before = tuple(app.query(TranscriptMessageWidget))

        await pilot.resize_terminal(55, 24)
        await pilot.pause()

        assert tuple(app.query(TranscriptMessageWidget)) == before


@pytest.mark.anyio
async def test_streaming_transcript_deltas_do_not_force_scroll_end_during_scrollback() -> None:
    # [FUNCTION DOC] Async: verifies streaming deltas do not call scroll_end while the user is scrolled back.
    # Wraps transcript.scroll_end with a counting function.
    app = TauTuiApp(
        FakeSession(
            messages=[
                UserMessage(content=f"message {index}\n" + "line\n" * 4) for index in range(12)
            ]
        )
    )

    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        transcript = app.query_one("#transcript", TranscriptView)
        transcript.follow_output()
        await pilot.pause()
        transcript.scroll_to(
            y=max(0, transcript.max_scroll_y - 5),
            animate=False,
            immediate=True,
        )
        await pilot.pause()
        forced_scrolls = 0
        original_scroll_end = transcript.scroll_end

        def tracking_scroll_end(*args: object, **kwargs: object) -> None:
            # [FUNCTION DOC] HELPER: counting wrapper around the original scroll_end; increments the enclosing
            # forced_scrolls counter and delegates with all arguments.
            nonlocal forced_scrolls
            forced_scrolls += 1
            original_scroll_end(*args, **kwargs)

        transcript.scroll_end = tracking_scroll_end  # type: ignore[method-assign]

        await transcript.append_assistant_delta("alpha")
        await transcript.append_assistant_delta(" beta")
        await pilot.pause()

    assert forced_scrolls == 0


@pytest.mark.anyio
async def test_streaming_transcript_deltas_follow_when_at_bottom() -> None:
    # [FUNCTION DOC] Async: verifies streamed deltas keep the transcript pinned to the bottom when following.
    app = TauTuiApp(
        FakeSession(
            messages=[
                UserMessage(content=f"message {index}\n" + "line\n" * 4) for index in range(12)
            ]
        )
    )

    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        transcript = app.query_one("#transcript", TranscriptView)
        transcript.follow_output()
        await pilot.pause()
        assert transcript.is_vertical_scroll_end

        await transcript.append_assistant_delta("alpha\n" * 20)
        for _ in range(5):
            await pilot.pause()
            if transcript.is_vertical_scroll_end:
                break

        assert transcript.is_vertical_scroll_end


@pytest.mark.anyio
async def test_streaming_transcript_deltas_preserve_user_scrollback() -> None:
    # [FUNCTION DOC] Async: verifies streamed deltas leave the scroll position unchanged when scrolled back.
    app = TauTuiApp(
        FakeSession(
            messages=[
                UserMessage(content=f"message {index}\n" + "line\n" * 4) for index in range(12)
            ]
        )
    )

    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        transcript = app.query_one("#transcript", TranscriptView)
        transcript.follow_output()
        await pilot.pause()
        assert transcript.max_scroll_y > 0

        transcript.scroll_to(
            y=max(0, transcript.max_scroll_y - 5),
            animate=False,
            immediate=True,
        )
        await pilot.pause()
        scrollback_y = transcript.scroll_y
        assert not transcript.is_vertical_scroll_end

        await transcript.append_assistant_delta("alpha\n" * 20)
        await pilot.pause()

        assert transcript.scroll_y == scrollback_y
        assert not transcript.is_vertical_scroll_end


@pytest.mark.anyio
async def test_streaming_transcript_deltas_do_not_apply_stale_follow_scroll() -> None:
    # [FUNCTION DOC] Async: verifies a follow-scroll scheduled before the user scrolled back is not applied.
    app = TauTuiApp(
        FakeSession(
            messages=[
                UserMessage(content=f"message {index}\n" + "line\n" * 4) for index in range(12)
            ]
        )
    )

    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        transcript = app.query_one("#transcript", TranscriptView)
        transcript.follow_output()
        await pilot.pause()
        assert transcript.is_vertical_scroll_end

        await transcript.append_assistant_delta("alpha\n" * 20)
        transcript.scroll_to(
            y=max(0, transcript.max_scroll_y - 5),
            animate=False,
            immediate=True,
        )
        scrollback_y = transcript.scroll_y
        assert not transcript.is_vertical_scroll_end

        await pilot.pause()

        assert transcript.scroll_y == scrollback_y
        assert not transcript.is_vertical_scroll_end


@pytest.mark.anyio
async def test_streaming_transcript_fractional_scrollback_after_refollow_stops_following() -> None:
    # [FUNCTION DOC] Async: verifies a fractional scroll-back after re-following disables follow mode so
    # subsequent deltas do not move the viewport.
    app = TauTuiApp(
        FakeSession(
            messages=[
                UserMessage(content=f"message {index}\n" + "line\n" * 4) for index in range(12)
            ]
        )
    )

    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        transcript = app.query_one("#transcript", TranscriptView)
        transcript.follow_output()
        await pilot.pause()
        assert transcript.is_vertical_scroll_end

        transcript.scroll_to(
            y=max(0, transcript.max_scroll_y - 5),
            animate=False,
            immediate=True,
        )
        await pilot.pause()
        assert not transcript.is_vertical_scroll_end

        transcript.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        assert transcript.is_vertical_scroll_end

        transcript.scroll_to(
            y=max(0, transcript.max_scroll_y - 0.6),
            animate=False,
            immediate=True,
        )
        scrollback_y = transcript.scroll_y
        assert not transcript.is_vertical_scroll_end

        await transcript.append_assistant_delta("alpha\n" * 20)
        await pilot.pause()

        assert transcript.scroll_y == scrollback_y
        assert not transcript.is_vertical_scroll_end


@pytest.mark.anyio
async def test_tui_transcript_selects_only_one_message() -> None:
    # [FUNCTION DOC] Async: verifies selecting all of one message yields only that message's text.
    app = TauTuiApp(
        FakeSession(
            messages=[
                UserMessage(content="first message"),
                AssistantMessage(content="second message"),
            ]
        )
    )

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        messages = list(app.query(TranscriptMessageWidget))

        app.screen.selections = {messages[0]: SELECT_ALL}

        assert app.screen.get_selected_text() == "first message"


@pytest.mark.anyio
async def test_tui_transcript_extracts_adjacent_message_selection() -> None:
    # [FUNCTION DOC] Async: verifies a selection spanning three messages joins partial and full texts with newlines.
    app = TauTuiApp(
        FakeSession(
            messages=[
                UserMessage(content="first one"),
                AssistantMessage(content="middle message"),
                UserMessage(content="third item"),
            ]
        )
    )

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        messages = list(app.query(TranscriptMessageWidget))

        app.screen.selections = {
            messages[0]: Selection(Offset(6, 0), None),
            messages[1]: SELECT_ALL,
            messages[2]: Selection(None, Offset(5, 0)),
        }

        assert app.screen.get_selected_text() == "one\nmiddle message\nthird"


@pytest.mark.anyio
async def test_tui_auto_copies_selected_text_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Async: verifies selected text is copied to the clipboard when auto-copy is enabled.
    # Inputs: monkeypatch (replaces copy_to_clipboard with a list append).
    app = TauTuiApp(
        FakeSession(messages=[UserMessage(content="copy this")]),
        tui_settings=TuiSettings(auto_copy_selection=True),
    )
    copied: list[str] = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        message = app.query_one(TranscriptMessageWidget)
        app.screen.selections = {message: SELECT_ALL}

        await app.on_text_selected()

    assert copied == ["copy this"]


@pytest.mark.anyio
async def test_tui_auto_copy_selection_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Async: verifies nothing is copied when auto-copy selection is disabled.
    app = TauTuiApp(
        FakeSession(messages=[UserMessage(content="do not copy")]),
        tui_settings=TuiSettings(auto_copy_selection=False),
    )
    copied: list[str] = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        message = app.query_one(TranscriptMessageWidget)
        app.screen.selections = {message: SELECT_ALL}

        await app.on_text_selected()

    assert copied == []


def test_transcript_selection_text_tracks_tool_result_visibility() -> None:
    # [FUNCTION DOC] Verifies tool selection text includes result text only when tool results are shown.
    item = ChatItem(
        role="tool",
        text="→ read README.md",
        tool_result_text="✓ read\nREADME contents",
    )

    assert transcript_item_selection_text(item, show_tool_results=False) == "→ read README.md"
    assert transcript_item_selection_text(item, show_tool_results=True) == (
        "→ read README.md\n\n✓ read\nREADME contents"
    )


@pytest.mark.anyio
async def test_tool_transcript_uses_native_markdown_without_custom_selection_painting() -> None:
    # [FUNCTION DOC] Async: verifies an appended tool item widget returns invocation plus result as selection
    # text and contains no Markdown fence widgets.
    app = TauTuiApp(FakeSession(messages=[]))
    item = ChatItem(
        role="tool",
        text="→ read README.md",
        tool_result_text="✓ read\nREADME contents",
    )

    async with app.run_test(size=(120, 30)) as pilot:
        transcript = app.query_one("#transcript", TranscriptView)
        widget = await transcript.append_item(item, show_tool_results=True)
        await pilot.pause()

        assert isinstance(widget, TranscriptMessageWidget)
        assert widget.get_selection(SELECT_ALL) == (
            "→ read README.md\n\n✓ read\nREADME contents",
            "\n",
        )
        assert list(widget.query("MarkdownFence")) == []


@pytest.mark.anyio
async def test_tui_message_start_does_not_mount_empty_assistant_message() -> None:
    # [FUNCTION DOC] Async: verifies an empty assistant MessageStartEvent does not mount a streaming widget.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 30)) as pilot:
        await app._apply_streaming_transcript_event(MessageStartEvent(message=AssistantMessage()))
        await pilot.pause()

        assert list(app.query(StreamingTranscriptMessageWidget)) == []


@pytest.mark.anyio
async def test_tui_streaming_deltas_update_active_message_without_full_refresh() -> None:
    # [FUNCTION DOC] Async: runs a prompt whose fake events stream two text deltas and verifies the app does
    # only one full refresh, feeds the deltas to the Markdown stream as fragments, never
    # replaces/re-updates the whole streaming widget, and ends with "alpha beta" in the
    # transcript. Temporarily patches StreamingTranscriptMessageWidget.replace_text/update and
    # MarkdownStream.write at class level (restored in a finally block) and the app's _refresh.
    partial = AssistantMessage()
    session = FakeSession(
        events=[
            AgentStartEvent(),
            MessageStartEvent(message=partial),
            MessageUpdateEvent(
                message=partial,
                assistant_message_event=TextDeltaEvent(
                    content_index=0, delta="alpha ", partial=partial
                ),
            ),
            MessageUpdateEvent(
                message=partial,
                assistant_message_event=TextDeltaEvent(
                    content_index=0, delta="beta", partial=partial
                ),
            ),
            MessageEndEvent(message=AssistantMessage(content="alpha beta")),
            AgentEndEvent(),
        ]
    )
    app = TauTuiApp(session)
    stream_replacements: list[str] = []
    stream_writes: list[str] = []
    full_stream_updates: list[str] = []
    original_replace_text = StreamingTranscriptMessageWidget.replace_text
    original_stream_update = StreamingTranscriptMessageWidget.update
    original_stream_write = MarkdownStream.write

    def tracking_stream_update(
        self: StreamingTranscriptMessageWidget,
        markdown: str,
    ) -> object:
        # [FUNCTION DOC] HELPER: tracking replacement for the streaming widget's update; records non-empty
        # Markdown passed and delegates to the original. Output: whatever the original returns.
        if markdown:
            full_stream_updates.append(markdown)
        return original_stream_update(self, markdown)

    async def tracking_stream_write(self: MarkdownStream, fragment: str) -> None:
        # [FUNCTION DOC] HELPER: async tracking replacement for MarkdownStream.write; records the fragment and
        # delegates to the original.
        stream_writes.append(fragment)
        await original_stream_write(self, fragment)

    async def tracking_replace_text(
        self: StreamingTranscriptMessageWidget,
        text: str,
    ) -> None:
        # [FUNCTION DOC] HELPER: async tracking replacement for replace_text; records the text and delegates.
        stream_replacements.append(text)
        await original_replace_text(self, text)

    StreamingTranscriptMessageWidget.replace_text = tracking_replace_text  # type: ignore[method-assign]
    StreamingTranscriptMessageWidget.update = tracking_stream_update  # type: ignore[method-assign]
    MarkdownStream.write = tracking_stream_write  # type: ignore[method-assign]
    full_refreshes = 0

    original_refresh = app._refresh

    def tracking_refresh() -> None:
        # [FUNCTION DOC] HELPER: counts full app refreshes in the enclosing scope then calls the original _refresh.
        nonlocal full_refreshes
        full_refreshes += 1
        original_refresh()

    app._refresh = tracking_refresh  # type: ignore[method-assign]

    try:
        async with app.run_test(size=(120, 30)) as pilot:
            await app._run_prompt("stream")
            await pilot.pause()

            transcript = app.query_one("#transcript", TranscriptView)
            streamed = app.query_one(StreamingTranscriptMessageWidget)
            transcript_text = "\n".join(line.text for line in transcript.lines)
    finally:
        StreamingTranscriptMessageWidget.replace_text = original_replace_text  # type: ignore[method-assign]
        StreamingTranscriptMessageWidget.update = original_stream_update  # type: ignore[method-assign]
        MarkdownStream.write = original_stream_write  # type: ignore[method-assign]

    assert full_refreshes == 1
    assert stream_writes == ["alpha ", "beta"]
    assert stream_replacements == []
    assert full_stream_updates == []
    assert streamed.selection_text == "alpha beta"
    assert "alpha beta" in transcript_text


@pytest.mark.anyio
async def test_tui_submit_prompt_optimistically_appends_user_message_without_full_refresh() -> None:
    # [FUNCTION DOC] Async: verifies submitting a prompt appends the user message optimistically without a
    # full refresh, sends it to the session once, and renders it exactly once.
    session = FakeSession(
        messages=[UserMessage(content=f"Earlier {index}") for index in range(3)],
        events=[
            AgentStartEvent(),
            MessageEndEvent(message=UserMessage(content="New prompt")),
            AgentEndEvent(),
        ],
    )
    app = TauTuiApp(session)
    full_refreshes = 0
    original_refresh = app._refresh

    def tracking_refresh() -> None:
        # [FUNCTION DOC] HELPER: counts full refreshes (nonlocal counter) and delegates to the original _refresh.
        nonlocal full_refreshes
        full_refreshes += 1
        original_refresh()

    app._refresh = tracking_refresh  # type: ignore[method-assign]

    async with app.run_test(size=(120, 30)) as pilot:
        full_refreshes = 0
        await app._submit_prompt("New prompt")
        await pilot.pause()
        await pilot.pause()

        transcript = app.query_one("#transcript", TranscriptView)
        user_messages = [item.text for item in app.state.items if item.role == "user"]
        transcript_lines = [line.text for line in transcript.lines]

    assert full_refreshes == 0
    assert session.prompt_texts == ["New prompt"]
    assert user_messages == ["Earlier 0", "Earlier 1", "Earlier 2", "New prompt"]
    assert transcript_lines.count("New prompt") == 1


@pytest.mark.anyio
async def test_tui_transformed_prompt_replaces_optimistic_user_message() -> None:
    # [FUNCTION DOC] Async: verifies when the session confirms a transformed user message (e.g. an extension
    # input hook rewrote it), the optimistic item is replaced in place, not duplicated.
    # An extension `input` hook may transform the submitted text inside
    # session.prompt, so the confirmed UserMessage content differs from the
    # optimistically rendered original. The optimistic item must be rewritten
    # in place — not left rendered alongside a second, transformed user item.
    session = FakeSession(
        events=[
            AgentStartEvent(),
            MessageEndEvent(message=UserMessage(content="rewritten words")),
            AgentEndEvent(),
        ],
    )
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await app._submit_prompt("original words")
        await pilot.pause()
        await pilot.pause()

        transcript = app.query_one("#transcript", TranscriptView)
        user_messages = [item.text for item in app.state.items if item.role == "user"]
        transcript_lines = [line.text for line in transcript.lines]

    assert user_messages == ["rewritten words"]
    assert "original words" not in transcript_lines
    assert transcript_lines.count("rewritten words") == 1


@pytest.mark.anyio
async def test_tui_submit_prompt_does_not_optimistically_append_slash_commands() -> None:
    # [FUNCTION DOC] Async: verifies slash-command prompts are not optimistically appended; only the expanded
    # user message from the session appears.
    session = FakeSession(
        events=[
            AgentStartEvent(),
            MessageEndEvent(message=UserMessage(content="Expanded prompt")),
            AgentEndEvent(),
        ],
    )
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await app._submit_prompt("/review src/app.py")
        await pilot.pause()
        await pilot.pause()

        transcript = app.query_one("#transcript", TranscriptView)
        user_messages = [item.text for item in app.state.items if item.role == "user"]
        transcript_lines = [line.text for line in transcript.lines]

    assert session.prompt_texts == ["/review src/app.py"]
    assert user_messages == ["Expanded prompt"]
    assert "/review src/app.py" not in transcript_lines
    assert transcript_lines.count("Expanded prompt") == 1


def test_prompt_input_shows_placeholder_for_large_paste() -> None:
    # [FUNCTION DOC] Verifies a large paste is shown as a "[Pasted content #1: N characters, M lines]"
    # placeholder while text_for_submission returns the full content.
    prompt = PromptInput()
    pasted = ("line\n" * 500) + "end"

    prompt.on_paste(events.Paste(pasted))

    assert prompt.text.startswith("[Pasted content #1:")
    assert f"{len(pasted):,} characters" in prompt.text
    assert "501 lines" in prompt.text
    assert prompt.text_for_submission() == pasted


def test_prompt_input_keeps_small_paste_default_behavior() -> None:
    # [FUNCTION DOC] Verifies a paste at or below the threshold is not intercepted (default behavior; the
    # unmounted widget text stays empty).
    prompt = PromptInput()
    event = events.Paste("x" * PASTE_DISPLAY_THRESHOLD)

    prompt.on_paste(event)

    assert prompt.text == ""
    assert prompt.text_for_submission() == ""


def test_prompt_input_preserves_edits_around_large_paste() -> None:
    # [FUNCTION DOC] Verifies text typed before and after a large-paste placeholder is preserved on submission.
    prompt = PromptInput()
    pasted = "x" * (PASTE_DISPLAY_THRESHOLD + 1)
    prompt.text = "before "
    prompt.move_cursor((0, len(prompt.text)))

    prompt.on_paste(events.Paste(pasted))
    prompt.text = f"{prompt.text}\nafter"

    assert prompt.text_for_submission() == f"before {pasted}\nafter"


def test_prompt_input_does_not_submit_deleted_large_paste() -> None:
    # [FUNCTION DOC] Verifies a deleted placeholder's paste content is not submitted.
    prompt = PromptInput()
    pasted = "x" * (PASTE_DISPLAY_THRESHOLD + 1)
    prompt.on_paste(events.Paste(pasted))

    prompt.text = "replacement prompt"

    assert prompt.text_for_submission() == "replacement prompt"


def test_prompt_input_expands_multiple_large_pastes() -> None:
    # [FUNCTION DOC] Verifies multiple large pastes get numbered placeholders and all expand on submission.
    prompt = PromptInput()
    first = "a" * (PASTE_DISPLAY_THRESHOLD + 1)
    second = "b" * (PASTE_DISPLAY_THRESHOLD + 2)

    prompt.on_paste(events.Paste(first))
    prompt.insert("\nbetween\n")
    prompt.on_paste(events.Paste(second))

    assert "[Pasted content #1:" in prompt.text
    assert "[Pasted content #2:" in prompt.text
    assert prompt.text_for_submission() == f"{first}\nbetween\n{second}"


def test_prompt_input_drops_only_deleted_large_paste_placeholder() -> None:
    # [FUNCTION DOC] Verifies removing only the first placeholder drops just that paste while keeping the second.
    prompt = PromptInput()
    first = "a" * (PASTE_DISPLAY_THRESHOLD + 1)
    second = "b" * (PASTE_DISPLAY_THRESHOLD + 2)

    prompt.on_paste(events.Paste(first))
    first_placeholder = prompt.text
    prompt.insert("\n")
    prompt.on_paste(events.Paste(second))
    prompt.text = prompt.text.replace(first_placeholder, "removed", 1)

    assert prompt.text_for_submission() == f"removed\n{second}"


@pytest.mark.anyio
async def test_tui_submit_large_paste_sends_full_content() -> None:
    # [FUNCTION DOC] Async: verifies pressing enter after a large paste sends the full pasted content.
    session = FakeSession()
    app = TauTuiApp(session)
    pasted = "x" * (PASTE_DISPLAY_THRESHOLD + 1)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.on_paste(events.Paste(pasted))
        await pilot.press("enter")
        await pilot.pause()

    assert session.prompt_texts == [pasted]


@pytest.mark.anyio
async def test_tui_submit_replaced_large_paste_does_not_send_stale_content() -> None:
    # [FUNCTION DOC] Async: verifies replacing the prompt text after a large paste sends only the new text.
    session = FakeSession()
    app = TauTuiApp(session)
    pasted = "x" * (PASTE_DISPLAY_THRESHOLD + 1)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.on_paste(events.Paste(pasted))
        prompt.text = "replacement prompt"
        await pilot.press("enter")
        await pilot.pause()

    assert session.prompt_texts == ["replacement prompt"]


@pytest.mark.anyio
async def test_tui_submit_multiple_large_pastes_sends_all_full_content() -> None:
    # [FUNCTION DOC] Async: verifies multiple large pastes are all expanded when the prompt is submitted.
    session = FakeSession()
    app = TauTuiApp(session)
    first = "a" * (PASTE_DISPLAY_THRESHOLD + 1)
    second = "b" * (PASTE_DISPLAY_THRESHOLD + 2)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.on_paste(events.Paste(first))
        prompt.insert("\nthen\n")
        prompt.on_paste(events.Paste(second))
        await pilot.press("enter")
        await pilot.pause()

    assert session.prompt_texts == [f"{first}\nthen\n{second}"]


@pytest.mark.anyio
async def test_tui_caret_leaving_mention_closes_completions_but_does_not_open(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Async: verifies moving the caret within a mention keeps file completions, while moving
    # it outside closes them and moving back does not reopen them. Inputs: tmp_path project.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    session = FakeSession()
    session.cwd = tmp_path
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.text = "look at @a and fix it"
        prompt.cursor_position = len("look at @a")
        prompt.insert("p")
        await pilot.pause()
        assert [item.display for item in app._completion_state.items] == ["@src/app.py"]

        prompt.cursor_position = len("look at @a")
        await pilot.pause()
        assert [item.display for item in app._completion_state.items] == ["@src/app.py"]

        prompt.cursor_position = len("look at @ap and")
        await pilot.pause()
        assert app._completion_state.items == ()

        prompt.cursor_position = len("look at @ap")
        await pilot.pause()
        assert app._completion_state.items == ()


@pytest.mark.anyio
async def test_tui_enter_submits_with_caret_parked_in_complete_mention(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Async: verifies enter submits the raw prompt when the caret sits inside an already
    # complete @mention (no completion popup intercepts it). Inputs: tmp_path.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    session = FakeSession()
    session.cwd = tmp_path
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.text = "look at @src/app.py and fix"
        await pilot.pause()
        prompt.cursor_position = len("look at @src")
        await pilot.pause()
        assert app._completion_state.items == ()
        await pilot.press("enter")
        await pilot.pause()

    assert session.prompt_texts == ["look at @src/app.py and fix"]


@pytest.mark.anyio
async def test_tui_accepting_mid_prompt_completion_keeps_cursor_after_mention(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Async: verifies accepting a mid-prompt @file completion replaces the partial mention and
    # places the cursor right after the inserted mention, closing completions. Inputs: tmp_path.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    session = FakeSession()
    session.cwd = tmp_path
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.text = "look at @ap and fix it"
        prompt.cursor_position = len("look at @ap")
        prompt.insert("p")
        await pilot.pause()
        app.action_accept_completion()
        await pilot.pause()

        assert prompt.text == "look at @src/app.py and fix it"
        assert prompt.cursor_position == len("look at @src/app.py")
        assert app._completion_state.items == ()


@pytest.mark.anyio
async def test_tui_app_mounts_sidebar_and_transcript() -> None:
    # [FUNCTION DOC] Async: verifies the app mounts the sidebar, a transcript with min width 1 and a
    # soft-wrapping TextArea prompt.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 30)):
        assert app.query_one("#sidebar") is not None
        transcript = app.query_one("#transcript")
        assert transcript is not None
        assert transcript.min_width == 1
        prompt = app.query_one("#prompt")
        assert isinstance(prompt, TextArea)
        assert prompt.soft_wrap is True


def test_tau_markdown_block_is_not_selectable_until_mounted() -> None:
    # [FUNCTION DOC] Verifies an unmounted TauMarkdownBlock is not selectable. Uses a minimal fake token object.
    markdown = TextualMarkdown("example")
    block = TauMarkdownBlock(
        markdown,
        type("Token", (), {"map": (0, 1), "level": 0, "type": "paragraph_open"})(),
    )

    assert block.allow_select is False


@pytest.mark.anyio
async def test_tau_markdown_block_remains_selectable_after_mount() -> None:
    # [FUNCTION DOC] Async: verifies a mounted Markdown block is selectable.
    app = TauTuiApp(
        FakeSession([AssistantMessage(content="Read [docs](https://example.com).")]),
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        block = app.query_one(TauMarkdownBlock)
        assert block.parent is not None
        assert block.allow_select is True


@pytest.mark.anyio
async def test_detached_transcript_message_ignores_stale_selection_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC] Async: verifies a streaming message widget is selectable only while mounted and that a
    # stale compositor hit on the removed widget does not start a text selection.
    # Inputs: monkeypatch (fakes the screen's widget hit-test).
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        widget = StreamingTranscriptMessageWidget(
            ChatItem(role="assistant", text="Select this message."), theme=TAU_DARK_THEME
        )
        assert widget.allow_select is False
        await app.query_one("#transcript").mount(widget)
        await widget.finalize()
        await pilot.pause()
        assert widget.allow_select is True
        await widget.remove()
        assert widget.parent is None
        assert widget.allow_select is False

        # Model a compositor hit left over from before the widget was removed.
        from textual.geometry import Offset

        monkeypatch.setattr(
            app.screen, "get_widget_and_offset_at", lambda x, y: (widget, Offset(0, 0))
        )
        app.screen._forward_event(events.MouseDown(None, 1, 1, 0, 0, 1, False, False, False))
        assert app.screen._select_state is None
        await pilot.pause()


@pytest.mark.anyio
async def test_tui_app_disables_text_selection_while_agent_is_running() -> None:
    # [FUNCTION DOC] Async: verifies app-level text selection is disabled while the agent runs and restored after.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 30)):
        assert app.ALLOW_SELECT is True

        app.adapter.apply(AgentStartEvent())
        app._refresh_chrome()

        assert app.ALLOW_SELECT is False

        app.adapter.apply(AgentEndEvent())
        app._refresh_chrome()

        assert app.ALLOW_SELECT is True


@pytest.mark.anyio
async def test_prompt_input_does_not_highlight_active_line() -> None:
    # [FUNCTION DOC] Async: verifies the prompt input does not highlight the cursor line.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 30)) as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "Keep the prompt background uniform."
        await pilot.pause()

        assert prompt.highlight_cursor_line is False


def test_terminal_command_prefix_span_detects_shell_mode_prefix() -> None:
    # [FUNCTION DOC] Verifies detection of the leading "!" / "!!" shell-mode prefix span (allowing leading
    # spaces) and None when "!" is not at the start.
    assert _terminal_command_prefix_span("! pwd") == (0, 1)
    assert _terminal_command_prefix_span("!! pwd") == (0, 2)
    assert _terminal_command_prefix_span("  !! pwd") == (2, 4)
    assert _terminal_command_prefix_span("hello ! pwd") is None


def test_activity_indicator_shows_dollar_sign_in_shell_mode() -> None:
    # [FUNCTION DOC] Verifies the idle activity indicator shows a bold "$" in the tool color in shell mode.
    theme = TAU_LIGHT_THEME

    rendered = _render_activity_indicator(theme, frame=0, running=False, shell_mode=True)

    assert rendered.plain == "$"
    assert rendered.style == f"bold {theme.role_styles['tool'].border}"


def test_activity_indicator_keeps_running_animation_in_shell_mode() -> None:
    # [FUNCTION DOC] Verifies the running animation frame is still shown in shell mode (not "$").
    theme = TAU_LIGHT_THEME

    rendered = _render_activity_indicator(theme, frame=0, running=True, shell_mode=True)

    assert rendered.plain != "$"


def test_activity_prompt_border_uses_tool_running_color_in_shell_mode() -> None:
    # [FUNCTION DOC] Verifies the prompt border uses the tool running color in idle shell mode.
    theme = TAU_LIGHT_THEME

    assert (
        _activity_prompt_border_color(theme, frame=0, running=False, shell_mode=True)
        == theme.role_styles["tool"].border
    )


@pytest.mark.anyio
async def test_tui_app_highlights_prompt_shell_mode() -> None:
    # [FUNCTION DOC] Async: verifies typing a shell-mode prompt adds the shell class, shows "$", colors every
    # line of the command in the tool color, and reverting to normal text clears it all and
    # restores the tau indicator.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 30)) as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        indicator = app.query_one("#prompt-prefix", Static)
        prompt.value = "!! pwd"
        await pilot.pause()

        assert prompt.has_class("-shell-mode")
        assert indicator.render().plain == "$"
        tool_running_color = app.tui_settings.resolved_theme.role_styles["tool"].border
        assert (
            _activity_prompt_border_color(
                app.tui_settings.resolved_theme,
                frame=0,
                running=False,
                shell_mode=prompt.has_class("-shell-mode"),
            )
            == tool_running_color
        )
        assert prompt.get_line(0).spans[-1].start == 0
        assert prompt.get_line(0).spans[-1].end == len("!! pwd")
        assert str(prompt.get_line(0).spans[-1].style) == tool_running_color

        prompt.value = "! pwd\nls -la"
        await pilot.pause()

        assert prompt.get_line(1).spans[-1].start == 0
        assert prompt.get_line(1).spans[-1].end == len("ls -la")
        assert str(prompt.get_line(1).spans[-1].style) == tool_running_color

        prompt.value = "ask tau"
        await pilot.pause()

        assert not prompt.has_class("-shell-mode")
        assert prompt.get_line(0).spans == []
        assert indicator.render().plain == "τ"


@pytest.mark.anyio
async def test_tui_app_omits_footer_but_keeps_shortcuts_active() -> None:
    # [FUNCTION DOC] Async: verifies no Footer or hint widget is mounted while the default shortcut bindings
    # remain active with their expected keys.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 40)):
        assert not app.query("Footer")
        assert len(app.query("#shortcut-hints")) == 0
        assert _visible_footer_bindings(app) == {
            "Quit": "ctrl+d",
            "Clear": "ctrl+c",
            "Commands": "ctrl+k",
            "Submit": "enter",
            "Newline": "shift+enter",
            "Sessions": "ctrl+r",
            "Thinking": "shift+tab",
            "Model": "ctrl+p",
            "Cancel": "escape",
        }


@pytest.mark.anyio
async def test_tui_app_footer_hints_update_for_completions() -> None:
    # [FUNCTION DOC] Async: verifies active bindings switch to completion navigation hints for slash completions.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 30)):
        prompt = app.query_one("#prompt")
        prompt.value = "/se"
        app._completion_state = app._build_completion_state(prompt.value)
        app._refresh_completions()

        assert _visible_footer_bindings(app) == {
            "Choose": "Up/Down",
            "Complete": "Tab/Enter",
            "Close": "escape",
        }


@pytest.mark.anyio
async def test_tui_app_footer_hints_explain_file_reference_enter_behavior(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Async: verifies for @file completions, Tab completes and Enter submits raw text. Inputs: tmp_path.
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")
    session = FakeSession()
    session.cwd = tmp_path
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 30)):
        prompt = app.query_one("#prompt")
        prompt.value = "inspect @READ"
        app._completion_state = app._build_completion_state(prompt.value)
        app._refresh_completions()

        assert _visible_footer_bindings(app) == {
            "Choose": "Up/Down",
            "Complete": "Tab",
            "Submit raw": "enter",
            "Close": "escape",
        }


@pytest.mark.anyio
async def test_tui_app_footer_hints_update_while_running() -> None:
    # [FUNCTION DOC] Async: verifies bindings while the agent runs: steer, follow-up, cancel, thinking toggle, tools.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 30)):
        app.adapter.apply(AgentStartEvent())
        app._refresh()

        assert _visible_footer_bindings(app) == {
            "Steer": "enter",
            "Follow-up": "alt+enter",
            "Cancel": "escape",
            "Thinking": "ctrl+t",
            "Tools": "ctrl+o",
        }


@pytest.mark.anyio
async def test_tui_prompt_grows_to_six_lines_then_scrolls() -> None:
    # [FUNCTION DOC] Async: verifies the prompt grows up to six lines of content and then scrolls.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 30)) as pilot:
        prompt = app.query_one("#prompt", TextArea)
        assert prompt.size.height == 1
        assert prompt.outer_size.height == 3

        prompt.text = "x" * 700
        await pilot.pause()
        assert prompt.size.height == 6
        assert prompt.outer_size.height == 8

        prompt.text = "x" * 1000
        await pilot.pause()
        assert prompt.size.height == 6
        assert prompt.max_scroll_y > 0


@pytest.mark.anyio
async def test_tui_sidebar_is_visible_on_medium_windows() -> None:
    # [FUNCTION DOC] Async: verifies the sidebar is visible at 120 columns with width 40, left padding, no
    # borders, brand pinned to the bottom, prompt background color, and compact info shown.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 40)):
        sidebar = app.query_one("#sidebar")
        sidebar_brand = app.query_one("#sidebar-brand", Static)
        compact_info = app.query_one("#compact-session-info")
        assert sidebar.display is True
        assert sidebar.region.width == 40
        assert sidebar.styles.padding.left == 2
        assert sidebar.styles.border_left[0] == ""
        assert sidebar.styles.border_right[0] == ""
        assert sidebar.styles.border_top[0] == ""
        assert sidebar.styles.border_bottom[0] == ""
        assert sidebar_brand.region.bottom == sidebar.content_region.bottom
        assert sidebar.styles.background == Color.parse(TAU_DARK_THEME.prompt_background)
        assert compact_info.display is True
        assert not app.has_class("-hide-sidebar")


@pytest.mark.anyio
async def test_tui_sidebar_resource_sections_expand_independently() -> None:
    # [FUNCTION DOC] Async: verifies skills and prompts sidebar sections are collapsible headings with counts
    # (skills include a token estimate), styled spans, transparent hover/focus backgrounds, and
    # expand/collapse independently on click.
    session = FakeSession()
    session.prompt_templates = (
        PromptTemplate("explain", session.cwd / ".tau/prompts/explain.md", "Explain"),
        PromptTemplate("fix", session.cwd / ".tau/prompts/fix.md", "Fix"),
    )
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 40)) as pilot:
        skills = app.query_one("#sidebar-skills", Collapsible)
        prompts = app.query_one("#sidebar-prompts", Collapsible)

        skills_heading = Content.from_markup(skills.title)
        prompts_heading = Content.from_markup(prompts.title)
        assert re.fullmatch(r"skills \(1 · ~\d+(?:\.\d+)?k? tokens\)", skills_heading.plain)
        assert prompts_heading.plain == "prompts (2)"
        assert str(skills_heading.spans[0].style) == f"bold {TAU_DARK_THEME.prompt_text}"
        assert str(skills_heading.spans[1].style) == TAU_DARK_THEME.completion_description
        assert str(prompts_heading.spans[0].style) == f"bold {TAU_DARK_THEME.prompt_text}"
        assert str(prompts_heading.spans[1].style) == TAU_DARK_THEME.completion_description
        assert skills.collapsed is True
        assert prompts.collapsed is True

        skill_title = app.query_one("#sidebar-skills CollapsibleTitle")
        await pilot.hover("#sidebar-skills CollapsibleTitle")
        assert skill_title.styles.background == Color.parse("transparent")

        skill_title.focus()
        await pilot.pause()
        assert skill_title.styles.background == Color.parse("transparent")
        assert skills.styles.background_tint == Color.parse("transparent")

        await pilot.click("#sidebar-skills CollapsibleTitle")
        assert skills.collapsed is False
        assert prompts.collapsed is True

        await pilot.click("#sidebar-prompts CollapsibleTitle")
        assert skills.collapsed is False
        assert prompts.collapsed is False

        await pilot.click("#sidebar-skills CollapsibleTitle")
        assert skills.collapsed is True
        assert prompts.collapsed is False


@pytest.mark.anyio
async def test_tui_sidebar_context_file_opens_in_main_editor_and_saves(tmp_path: Path) -> None:
    # [FUNCTION DOC] Async: verifies clicking a context file opens it in the main-area editor (hiding the
    # transcript), ctrl+s saves to disk with a status message, and escape returns focus to the
    # prompt. Side effects: writes a file in tmp_path.
    context_path = tmp_path / "AGENTS.md"
    context_path.write_text("Original context.\n", encoding="utf-8")
    session = FakeSession()
    session.cwd = tmp_path
    session.context_files = (
        ProjectContextFile(path=str(context_path), content="Original context."),
    )
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.click("#sidebar-context-content .sidebar-file-item")
        await pilot.pause()

        editor = app.query_one("#sidebar-file-editor", SidebarFileEditor)
        assert editor.path == context_path
        assert editor.kind == "Context"
        assert not app.query_one("#transcript", TranscriptView).display
        editor_input = editor.query_one("#sidebar-file-editor-input", TextArea)
        assert editor_input.text == "Original context.\n"

        editor_input.text = "Updated context.\n"
        await pilot.press("ctrl+s")
        await pilot.pause()

        assert context_path.read_text(encoding="utf-8") == "Updated context.\n"
        assert editor.query_one("#sidebar-file-editor-status", Static).render().plain == (
            f"Saved {context_path}"
        )
        await pilot.press("escape")
        await pilot.pause()
        assert app.query_one("#transcript", TranscriptView).display
        assert app.query_one("#prompt", PromptInput).has_focus


@pytest.mark.anyio
async def test_tui_sidebar_prompt_file_opens_in_main_editor(tmp_path: Path) -> None:
    # [FUNCTION DOC] Async: verifies clicking a prompt template opens it in the editor and arrow keys move the
    # editor cursor normally. Inputs: tmp_path.
    prompt_path = tmp_path / ".tau" / "prompts" / "review.md"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text("Review this.\n", encoding="utf-8")
    session = FakeSession()
    session.cwd = tmp_path
    session.prompt_templates = (
        PromptTemplate("review", prompt_path, "Review this.", "Review changes"),
    )
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 40)) as pilot:
        app.query_one("#sidebar-prompts", Collapsible).collapsed = False
        await pilot.pause()
        await pilot.click("#sidebar-prompts-content .sidebar-file-item")
        await pilot.pause()

        editor = app.query_one("#sidebar-file-editor", SidebarFileEditor)
        assert editor.path == prompt_path
        assert editor.kind == "Prompt"
        editor_input = editor.query_one("#sidebar-file-editor-input", TextArea)
        assert editor_input.text == "Review this.\n"

        editor_input.text = "first\nsecond"
        editor_input.move_cursor((0, 0))
        await pilot.press("right")
        assert editor_input.cursor_location == (0, 1)
        await pilot.press("down")
        assert editor_input.cursor_location == (1, 1)
        await pilot.press("left")
        assert editor_input.cursor_location == (1, 0)
        await pilot.press("up")
        assert editor_input.cursor_location == (0, 0)


@pytest.mark.anyio
async def test_tui_sidebar_skill_opens_only_main_skill_file(tmp_path: Path) -> None:
    # [FUNCTION DOC] Async: verifies a skill directory shows a single sidebar item and opening it edits only
    # the main SKILL.md file, not supporting files. Inputs: tmp_path.
    skill_path = tmp_path / ".agents" / "skills" / "review" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("# Review\n", encoding="utf-8")
    (skill_path.parent / "reference.md").write_text("Supporting notes.\n", encoding="utf-8")
    session = FakeSession()
    session.cwd = tmp_path
    session.skills = (Skill("review", skill_path, "# Review", "Review changes"),)
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 40)) as pilot:
        app.query_one("#sidebar-skills", Collapsible).collapsed = False
        await pilot.pause()
        skill_items = app.query("#sidebar-skills-content .sidebar-file-item")
        assert len(skill_items) == 1
        await pilot.click("#sidebar-skills-content .sidebar-file-item")
        await pilot.pause()

        editor = app.query_one("#sidebar-file-editor", SidebarFileEditor)
        assert editor.path == skill_path
        assert editor.kind == "Skill"
        editor_path = editor.query_one("#sidebar-file-editor-path", Static).render().plain
        assert "reference.md" not in editor_path


@pytest.mark.anyio
async def test_tui_sidebar_file_items_have_clear_hover_style(tmp_path: Path) -> None:
    # [FUNCTION DOC] Async: verifies hovering a sidebar file item applies highlight background, text color
    # and underline. Inputs: tmp_path.
    context_path = tmp_path / "AGENTS.md"
    context_path.write_text("Rules.\n", encoding="utf-8")
    session = FakeSession()
    session.cwd = tmp_path
    session.context_files = (ProjectContextFile(path=str(context_path), content="Rules."),)
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 40)) as pilot:
        item = app.query_one("#sidebar-context-content .sidebar-file-item", SidebarFileItem)
        await pilot.hover("#sidebar-context-content .sidebar-file-item")

        assert item.styles.background == Color.parse(TAU_DARK_THEME.highlight_background)
        assert item.styles.color == Color.parse(TAU_DARK_THEME.highlight_text)
        assert item.styles.text_style.underline


@pytest.mark.anyio
async def test_tui_sidebar_editor_keeps_contents_open_after_save_failure(tmp_path: Path) -> None:
    # [FUNCTION DOC] Async: verifies a save failure (file and directory deleted) keeps the editor open with a
    # "Could not save" status and does not recreate the file. Inputs: tmp_path.
    context_path = tmp_path / "nested" / "AGENTS.md"
    context_path.parent.mkdir()
    context_path.write_text("Rules.\n", encoding="utf-8")
    session = FakeSession()
    session.cwd = tmp_path
    session.context_files = (ProjectContextFile(path=str(context_path), content="Rules."),)
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.click("#sidebar-context-content .sidebar-file-item")
        await pilot.pause()
        context_path.unlink()
        context_path.parent.rmdir()

        editor = app.query_one("#sidebar-file-editor", SidebarFileEditor)
        editor.query_one("#sidebar-file-editor-input", TextArea).text = "Unsaved changes.\n"
        await pilot.press("ctrl+s")
        await pilot.pause()

        assert editor.is_mounted
        assert (
            "Could not save"
            in editor.query_one("#sidebar-file-editor-status", Static).render().plain
        )
        assert not context_path.exists()


@pytest.mark.anyio
async def test_tui_sidebar_editor_reports_unexpected_save_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC] Async: verifies an unexpected exception from the atomic writer is reported in the status,
    # resets the saving flag and leaves the file unchanged. Inputs: tmp_path, monkeypatch.
    context_path = tmp_path / "AGENTS.md"
    context_path.write_text("Rules.\n", encoding="utf-8")
    session = FakeSession()
    session.cwd = tmp_path
    session.context_files = (ProjectContextFile(path=str(context_path), content="Rules."),)
    app = TauTuiApp(session)

    def fail_unexpectedly(*args: object) -> None:
        # [FUNCTION DOC] HELPER: replacement writer that always raises RuntimeError("simulated unexpected failure").
        del args
        raise RuntimeError("simulated unexpected failure")

    monkeypatch.setattr(tui_app, "_atomic_write_sidebar_file", fail_unexpectedly)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.click("#sidebar-context-content .sidebar-file-item")
        await pilot.pause()
        editor = app.query_one("#sidebar-file-editor", SidebarFileEditor)
        editor.query_one("#sidebar-file-editor-input", TextArea).text = "Updated.\n"

        await pilot.press("ctrl+s")
        await pilot.pause()

        status = editor.query_one("#sidebar-file-editor-status", Static).render().plain
        assert "Could not save" in status
        assert "simulated unexpected failure" in status
        assert editor._saving is False
        assert context_path.read_text(encoding="utf-8") == "Rules.\n"


@pytest.mark.anyio
async def test_tui_sidebar_editor_refuses_external_file_changes(tmp_path: Path) -> None:
    # [FUNCTION DOC] Async: verifies saving is refused with "File changed on disk" when the file was modified
    # externally after opening, leaving the external content and no temp files.
    context_path = tmp_path / "AGENTS.md"
    context_path.write_text("Original.\n", encoding="utf-8")
    session = FakeSession()
    session.cwd = tmp_path
    session.context_files = (ProjectContextFile(path=str(context_path), content="Original."),)
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.click("#sidebar-context-content .sidebar-file-item")
        await pilot.pause()
        editor = app.query_one("#sidebar-file-editor", SidebarFileEditor)
        editor.query_one("#sidebar-file-editor-input", TextArea).text = "Editor changes.\n"
        context_path.write_text("External changes.\n", encoding="utf-8")

        await pilot.press("ctrl+s")
        await pilot.pause()

        status = editor.query_one("#sidebar-file-editor-status", Static).render().plain
        assert "File changed on disk" in status
        assert editor.is_mounted
        assert context_path.read_text(encoding="utf-8") == "External changes.\n"
        assert not tuple(tmp_path.glob(f".{context_path.name}.*.tmp"))


@pytest.mark.anyio
async def test_tui_sidebar_editor_blocks_switch_with_unsaved_changes(tmp_path: Path) -> None:
    # [FUNCTION DOC] Async: verifies requesting another file while the editor has unsaved changes keeps the
    # current file open with its edits and shows a "Save or close" status.
    first_path = tmp_path / "AGENTS.md"
    second_path = tmp_path / "CLAUDE.md"
    first_path.write_text("First.\n", encoding="utf-8")
    second_path.write_text("Second.\n", encoding="utf-8")
    session = FakeSession()
    session.cwd = tmp_path
    session.context_files = (
        ProjectContextFile(path=str(first_path), content="First."),
        ProjectContextFile(path=str(second_path), content="Second."),
    )
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 40)) as pilot:
        items = list(app.query("#sidebar-context-content .sidebar-file-item"))
        items[0].post_message(SidebarFileItem.OpenRequested(items[0]))
        await pilot.pause()
        editor = app.query_one("#sidebar-file-editor", SidebarFileEditor)
        editor.query_one("#sidebar-file-editor-input", TextArea).text = "Unsaved.\n"

        items[1].post_message(SidebarFileItem.OpenRequested(items[1]))
        await pilot.pause()

        assert app.query_one("#sidebar-file-editor", SidebarFileEditor) is editor
        assert editor.path == first_path
        assert editor.query_one("#sidebar-file-editor-input", TextArea).text == "Unsaved.\n"
        status = editor.query_one("#sidebar-file-editor-status", Static).render().plain
        assert status == "Save or close the current file before opening another."


@pytest.mark.anyio
async def test_tui_sidebar_editor_partial_staging_failure_is_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC] Async: verifies a failure while staging the temp file (partial write then OSError) leaves
    # the original bytes intact, reports the error and cleans up temp files.
    # Inputs: tmp_path, monkeypatch.
    context_path = tmp_path / "AGENTS.md"
    original = b"Original rules.\n"
    context_path.write_bytes(original)
    session = FakeSession()
    session.cwd = tmp_path
    session.context_files = (ProjectContextFile(path=str(context_path), content="Original rules."),)
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.click("#sidebar-context-content .sidebar-file-item")
        await pilot.pause()

        def fail_after_partial_stage(handle: BinaryIO, source: str) -> None:
            # [FUNCTION DOC] HELPER: replacement staging writer that writes and flushes partial bytes to the handle
            # then raises OSError("simulated disk full"). Source text is ignored.
            del source
            handle.write(b"partial")
            handle.flush()
            raise OSError("simulated disk full")

        monkeypatch.setattr(tui_app, "_write_staged_utf8", fail_after_partial_stage)
        editor = app.query_one("#sidebar-file-editor", SidebarFileEditor)
        editor.query_one("#sidebar-file-editor-input", TextArea).text = "Replacement rules.\n"
        await pilot.press("ctrl+s")
        await pilot.pause()

        status = editor.query_one("#sidebar-file-editor-status", Static).render().plain
        assert editor.is_mounted
        assert "Could not save" in status
        assert "simulated disk full" in status
        assert context_path.read_bytes() == original
        assert not tuple(tmp_path.glob(f".{context_path.name}.*.tmp"))


@pytest.mark.anyio
async def test_tui_sidebar_editor_rejects_read_only_target(tmp_path: Path) -> None:
    # [FUNCTION DOC] Async: verifies saving to a read-only file fails with "Permission denied" status, sends a
    # single error notification, and preserves content, mode and no temp files.
    # Side effects: chmods a file in tmp_path. KNOWN ISSUE: may not fail as expected when run
    # as root, since root can write read-only files.
    context_path = tmp_path / "AGENTS.md"
    original = b"Original rules.\n"
    context_path.write_bytes(original)
    context_path.chmod(0o444)
    original_mode = context_path.stat().st_mode
    session = FakeSession()
    session.cwd = tmp_path
    session.context_files = (ProjectContextFile(path=str(context_path), content="Original rules."),)
    app = TauTuiApp(session)
    notifications: list[tuple[str, str | None]] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records (message, severity-if-string) notifications in place of the app's _notify.
        severity = kwargs.get("severity")
        notifications.append((message, severity if isinstance(severity, str) else None))

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.click("#sidebar-context-content .sidebar-file-item")
        await pilot.pause()

        editor = app.query_one("#sidebar-file-editor", SidebarFileEditor)
        editor.query_one("#sidebar-file-editor-input", TextArea).text = "Replacement rules.\n"
        await pilot.press("ctrl+s")
        await pilot.pause()

        status = editor.query_one("#sidebar-file-editor-status", Static).render().plain
        assert editor.is_mounted
        assert "Could not save" in status
        assert "Permission denied" in status
        assert notifications == [(status, "error")]
        assert context_path.read_bytes() == original
        assert context_path.stat().st_mode == original_mode
        assert not tuple(tmp_path.glob(f".{context_path.name}.*.tmp"))


def test_sidebar_file_atomic_save_preserves_symlink_and_permissions(tmp_path: Path) -> None:
    # [FUNCTION DOC] Verifies the atomic sidebar save writes through a symlink (keeping the link), preserves
    # the target's permission bits and leaves no temp files. Inputs: tmp_path.
    target = tmp_path / "actual.md"
    target.write_text("Original.\n", encoding="utf-8")
    target.chmod(0o640)
    link = tmp_path / "AGENTS.md"
    link.symlink_to(target.name)

    _, snapshot = tui_app._read_sidebar_file(link)
    tui_app._atomic_write_sidebar_file(link, "Updated.\n", snapshot)

    assert link.is_symlink()
    assert link.read_text(encoding="utf-8") == "Updated.\n"
    assert target.stat().st_mode & 0o777 == 0o640
    assert not tuple(tmp_path.glob(f".{target.name}.*.tmp"))


def test_sidebar_file_atomic_save_replace_failure_keeps_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC] Verifies a failing os.replace raises OSError and leaves the original file and no temp
    # files. Inputs: tmp_path, monkeypatch.
    target = tmp_path / "AGENTS.md"
    original = b"Original.\n"
    target.write_bytes(original)

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        # [FUNCTION DOC] HELPER: replacement for os.replace that always raises OSError("simulated replace failure").
        del source, destination
        raise OSError("simulated replace failure")

    monkeypatch.setattr(tui_app.os, "replace", fail_replace)
    _, snapshot = tui_app._read_sidebar_file(target)
    with pytest.raises(OSError, match="replace failure"):
        tui_app._atomic_write_sidebar_file(target, "Updated.\n", snapshot)

    assert target.read_bytes() == original
    assert not tuple(tmp_path.glob(f".{target.name}.*.tmp"))


def test_sidebar_file_atomic_save_works_without_fchmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC] Verifies the atomic save still works on platforms lacking os.fchmod. Inputs: tmp_path, monkeypatch.
    target = tmp_path / "AGENTS.md"
    target.write_text("Original.\n", encoding="utf-8")
    _, snapshot = tui_app._read_sidebar_file(target)
    monkeypatch.delattr(tui_app.os, "fchmod", raising=False)

    tui_app._atomic_write_sidebar_file(target, "Updated.\n", snapshot)

    assert target.read_text(encoding="utf-8") == "Updated.\n"


@pytest.mark.anyio
async def test_tui_sidebar_refreshes_skill_tokens_when_model_invocation_changes() -> None:
    # [FUNCTION DOC] Async: verifies the skills heading token estimate drops to ~0 when the only skill becomes
    # user-only (model invocation disabled) after a sidebar update.
    session = FakeSession()
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 40)):
        sidebar = app.query_one("#sidebar", SessionSidebar)
        skills = app.query_one("#sidebar-skills", Collapsible)
        initial_title = Content.from_markup(skills.title).plain
        assert initial_title != "skills (1 · ~0 tokens)"

        skill = session.skills[0]
        session.skills = (
            Skill(
                name=skill.name,
                path=skill.path,
                content=skill.content,
                description=skill.description,
                disable_model_invocation=True,
            ),
        )
        sidebar.update_from_session(session)

        assert Content.from_markup(skills.title).plain == "skills (1 · ~0 tokens)"


@pytest.mark.anyio
async def test_tui_sidebar_scrolls_when_all_skills_overflow() -> None:
    # [FUNCTION DOC] Async: verifies an expanded list of 30 skills makes the sidebar scroll area scrollable
    # while the brand stays pinned at the bottom.
    session = FakeSession()
    session.skills = tuple(
        Skill(
            name=f"skill-{index}",
            path=session.cwd / ".tau" / "skills" / f"skill-{index}" / "SKILL.md",
            content="Skill",
        )
        for index in range(1, 31)
    )
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 40)) as pilot:
        scroll = app.query_one("#sidebar-scroll", VerticalScroll)
        brand = app.query_one("#sidebar-brand", Static)
        app.query_one("#sidebar-skills", Collapsible).collapsed = False
        await pilot.wait_for_scheduled_animations()

        assert scroll.max_scroll_y > 0
        assert brand.region.bottom == app.query_one("#sidebar").content_region.bottom
        scroll.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        assert 0 < scroll.scroll_y <= scroll.max_scroll_y


@pytest.mark.anyio
async def test_tui_sidebar_relayouts_when_reload_changes_resource_count() -> None:
    # [FUNCTION DOC] Async: verifies the sidebar relayouts when a reload grows then shrinks the skill count,
    # updating heading counts, virtual height and scrollability while keeping expansion state.
    session = FakeSession()
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 40)) as pilot:
        sidebar = app.query_one("#sidebar", SessionSidebar)
        scroll = app.query_one("#sidebar-scroll", VerticalScroll)
        await pilot.pause()
        initial_virtual_height = scroll.virtual_size.height
        assert scroll.max_scroll_y == 0

        session.skills = tuple(
            Skill(
                name=f"skill-{index}",
                path=session.cwd / ".tau" / "skills" / f"skill-{index}" / "SKILL.md",
                content="Skill",
            )
            for index in range(1, 31)
        )
        sidebar.update_from_session(session)
        app.query_one("#sidebar-skills", Collapsible).collapsed = False
        await pilot.pause()

        expanded_virtual_height = scroll.virtual_size.height
        assert re.fullmatch(
            r"skills \(30 · ~\d+(?:\.\d+)?k? tokens\)",
            Content.from_markup(app.query_one("#sidebar-skills", Collapsible).title).plain,
        )
        assert expanded_virtual_height > initial_virtual_height
        assert scroll.max_scroll_y > 0

        session.skills = session.skills[:1]
        sidebar.update_from_session(session)
        await pilot.pause()

        skills = app.query_one("#sidebar-skills", Collapsible)
        assert re.fullmatch(
            r"skills \(1 · ~\d+(?:\.\d+)?k? tokens\)",
            Content.from_markup(skills.title).plain,
        )
        assert skills.collapsed is False
        assert scroll.virtual_size.height < expanded_virtual_height
        assert scroll.max_scroll_y == 0


@pytest.mark.anyio
async def test_tui_sidebar_fills_workspace_height() -> None:
    # [FUNCTION DOC] Async: verifies the sidebar fills the full workspace height.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 40)):
        workspace = app.query_one("#workspace")
        sidebar = app.query_one("#sidebar")

        assert sidebar.region.height == workspace.region.height
        assert sidebar.outer_size.height == workspace.size.height


@pytest.mark.anyio
async def test_tui_sidebar_hides_on_narrow_windows() -> None:
    # [FUNCTION DOC] Async: verifies the sidebar hides (compact info shown, -hide-sidebar class) at 80 columns.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(80, 30)):
        sidebar = app.query_one("#sidebar")
        compact_info = app.query_one("#compact-session-info")
        assert sidebar.display is False
        assert compact_info.display is True
        assert app.has_class("-hide-sidebar")


@pytest.mark.anyio
async def test_prompt_renders_when_narrow_layout_has_no_content_width() -> None:
    # [FUNCTION DOC] Async: verifies the prompt can render a line even when its content width is zero.
    """A narrow pane can briefly give the prompt zero content cells."""
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(3, 10)):
        prompt = app.query_one("#prompt", PromptInput)
        assert prompt.content_size.width == 0
        prompt.render_line(0)


@pytest.mark.anyio
async def test_tui_sidebar_hides_on_short_windows() -> None:
    # [FUNCTION DOC] Async: verifies the sidebar hides on short (30 row) windows.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 30)):
        sidebar = app.query_one("#sidebar")
        compact_info = app.query_one("#compact-session-info")
        assert sidebar.display is False
        assert compact_info.display is True
        assert app.has_class("-hide-sidebar")


@pytest.mark.anyio
async def test_tui_sidebar_visibility_updates_on_resize() -> None:
    # [FUNCTION DOC] Async: verifies sidebar visibility follows terminal resizes across width/height thresholds.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 40)) as pilot:
        sidebar = app.query_one("#sidebar")
        compact_info = app.query_one("#compact-session-info")
        assert sidebar.display is True
        assert compact_info.display is True

        await pilot.resize_terminal(width=80, height=40)
        await pilot.pause()
        assert sidebar.display is False
        assert compact_info.display is True

        await pilot.resize_terminal(width=120, height=30)
        await pilot.pause()
        assert sidebar.display is False
        assert compact_info.display is True

        await pilot.resize_terminal(width=120, height=40)
        await pilot.pause()
        assert sidebar.display is True
        assert compact_info.display is True


@pytest.mark.anyio
async def test_tui_sidebar_shows_on_right_by_default() -> None:
    # [FUNCTION DOC] Async: verifies the default sidebar position is right.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 40)):
        sidebar = app.query_one("#sidebar")
        assert sidebar.display is True
        assert app.has_class("-sidebar-right")
        assert not app.has_class("-sidebar-off")


@pytest.mark.anyio
async def test_tui_sidebar_shows_on_left_when_configured() -> None:
    # [FUNCTION DOC] Async: verifies sidebar_position="left" shows the sidebar without the right class.
    app = TauTuiApp(FakeSession(), tui_settings=TuiSettings(sidebar_position="left"))

    async with app.run_test(size=(120, 40)):
        sidebar = app.query_one("#sidebar")
        assert sidebar.display is True
        assert not app.has_class("-sidebar-right")
        assert not app.has_class("-sidebar-off")


@pytest.mark.anyio
@pytest.mark.parametrize("position", ["left", "right"])
async def test_tui_sidebar_command_toggles_visibility_without_changing_position(
    position: str,
) -> None:
    # [FUNCTION DOC] Parametrized async: verifies /sidebar toggles visibility without changing position or settings.
    app = TauTuiApp(FakeSession(), tui_settings=TuiSettings(sidebar_position=position))

    async with app.run_test(size=(120, 40)) as pilot:
        sidebar = app.query_one("#sidebar")
        assert sidebar.display is True
        assert app.has_class("-sidebar-right") is (position == "right")

        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/sidebar"
        await pilot.press("enter")
        await pilot.pause()
        assert sidebar.display is False
        assert app.has_class("-sidebar-right") is (position == "right")
        assert app.tui_settings.sidebar_position == position

        prompt.value = "/sidebar"
        await pilot.press("enter")
        await pilot.pause()
        assert sidebar.display is True
        assert app.has_class("-sidebar-right") is (position == "right")


@pytest.mark.anyio
async def test_tui_sidebar_command_hides_user_sidebar_across_resize() -> None:
    # [FUNCTION DOC] Async: verifies a user-hidden sidebar stays hidden across responsive resizes.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(120, 40)) as pilot:
        sidebar = app.query_one("#sidebar")
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/sidebar"
        await pilot.press("enter")
        await pilot.pause()
        assert sidebar.display is False

        await pilot.resize_terminal(width=80, height=30)
        await pilot.resize_terminal(width=120, height=40)
        await pilot.pause()
        assert sidebar.display is False


@pytest.mark.anyio
async def test_tui_sidebar_off_can_be_shown_temporarily_on_right() -> None:
    # [FUNCTION DOC] Async: verifies with position "off", /sidebar temporarily shows it on the right without
    # persisting settings, and toggling again hides it.
    app = TauTuiApp(FakeSession(), tui_settings=TuiSettings(sidebar_position="off"))

    async with app.run_test(size=(120, 40)) as pilot:
        sidebar = app.query_one("#sidebar")
        assert sidebar.display is False
        assert not app.has_class("-sidebar-right")

        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/sidebar"
        await pilot.press("enter")
        await pilot.pause()
        assert sidebar.display is True
        assert app.has_class("-sidebar-right")
        assert app.tui_settings.sidebar_position == "off"
        assert not tui_settings_path().exists()

        prompt.value = "/sidebar"
        await pilot.press("enter")
        await pilot.pause()
        assert sidebar.display is False
        assert not app.has_class("-sidebar-right")


@pytest.mark.anyio
async def test_tui_sidebar_is_hidden_when_off() -> None:
    # [FUNCTION DOC] Async: verifies the sidebar is hidden when position is "off".
    app = TauTuiApp(FakeSession(), tui_settings=TuiSettings(sidebar_position="off"))

    async with app.run_test(size=(120, 30)):
        sidebar = app.query_one("#sidebar")
        assert sidebar.display is False
        assert app.has_class("-hide-sidebar")
        assert not app.has_class("-sidebar-right")


@pytest.mark.anyio
async def test_tui_sidebar_right_still_hides_on_small_windows() -> None:
    # [FUNCTION DOC] Async: verifies a right sidebar still hides on small windows.
    app = TauTuiApp(FakeSession(), tui_settings=TuiSettings(sidebar_position="right"))

    async with app.run_test(size=(80, 30)):
        sidebar = app.query_one("#sidebar")
        assert sidebar.display is False
        assert app.has_class("-hide-sidebar")
        assert app.has_class("-sidebar-right")


@pytest.mark.anyio
async def test_tui_sidebar_off_ignores_responsive_toggle() -> None:
    # [FUNCTION DOC] Async: verifies "off" keeps the sidebar hidden even on large-enough windows.
    app = TauTuiApp(FakeSession(), tui_settings=TuiSettings(sidebar_position="off"))

    async with app.run_test(size=(120, 30)):
        sidebar = app.query_one("#sidebar")
        assert sidebar.display is False
        assert app.has_class("-hide-sidebar")
        assert not app.has_class("-sidebar-right")


@pytest.mark.anyio
async def test_tui_transcript_reflows_when_terminal_resizes() -> None:
    # [FUNCTION DOC] Async: verifies transcript content reflows within the viewport width after a resize,
    # with no horizontal scroll offset.
    app = TauTuiApp(
        FakeSession(
            messages=[
                UserMessage(
                    content=(
                        "Please summarize this very long sentence that should wrap cleanly "
                        "inside the transcript when the terminal becomes narrower."
                    )
                )
            ]
        )
    )

    async with app.run_test(size=(120, 30)) as pilot:
        transcript = app.query_one("#transcript")
        assert transcript.virtual_size.width <= transcript.scrollable_content_region.width

        await pilot.resize_terminal(width=64, height=30)
        await pilot.pause()

        assert transcript.virtual_size.width <= transcript.scrollable_content_region.width
        assert transcript.scroll_offset.x == 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("code", "has_horizontal_overflow"),
    [
        ("x = 1", False),
        ("value = '" + ("x" * 140) + "'", True),
    ],
)
async def test_tui_transcript_code_block_scrollbar_matches_overflow(
    code: str,
    has_horizontal_overflow: bool,
) -> None:
    # [FUNCTION DOC] Parametrized async: verifies code fences show a horizontal scrollbar only when the code
    # overflows horizontally.
    app = TauTuiApp(FakeSession(messages=[AssistantMessage(content=f"```python\n{code}\n```")]))

    async with app.run_test(size=(64, 30)) as pilot:
        await pilot.pause()
        transcript = app.query_one("#transcript", TranscriptView)
        fence = app.query_one("MarkdownFence")
        assert transcript.styles.overflow_x == "auto"
        assert transcript.styles.scrollbar_size_horizontal == 1
        assert fence.styles.scrollbar_size_horizontal == 1
        assert fence.styles.overflow_x == "auto"
        assert fence.allow_horizontal_scroll is True
        assert (fence.max_scroll_x > 0) is has_horizontal_overflow
        assert fence.show_horizontal_scrollbar is has_horizontal_overflow


@pytest.mark.anyio
async def test_tui_transcript_code_fence_ignores_invalid_highlighter_spans() -> None:
    # [FUNCTION DOC] Async: verifies invalid highlighter spans for an INI fence are dropped so every span is
    # within the rendered plain text bounds.
    app = TauTuiApp(
        FakeSession(
            messages=[AssistantMessage(content="```ini\nkeybind = alt+arrow_left=text:\\\n```")]
        )
    )

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        label = app.query_one("#code-content", Label)
        content = label.render()

    assert isinstance(content, Content)
    assert content.plain == "keybind = alt+arrow_left=text:\\"
    assert all(0 <= span.start < span.end <= len(content.plain) for span in content.spans)


@pytest.mark.anyio
async def test_streaming_code_block_hides_horizontal_scrollbar_until_finalized() -> None:
    # [FUNCTION DOC] Async: verifies a streaming code fence hides its horizontal scrollbar until the message is
    # finalized, after which it is shown.
    app = TauTuiApp(FakeSession())
    long_code_line = "value = '" + ("x" * 140) + "'"

    async with app.run_test(size=(64, 30)) as pilot:
        await pilot.pause()
        transcript = app.query_one("#transcript", TranscriptView)

        await transcript.append_assistant_delta("```python\n" + long_code_line)
        await pilot.pause()

        streaming_fence = app.query_one("MarkdownFence")
        assert streaming_fence.max_scroll_x > 0
        assert streaming_fence.styles.scrollbar_size_horizontal == 0
        assert streaming_fence.show_horizontal_scrollbar is False

        await transcript.finish_assistant_message("```python\n" + long_code_line + "\n```")
        await pilot.pause()

        finalized_fence = app.query_one("MarkdownFence")
        assert finalized_fence.max_scroll_x > 0
        assert finalized_fence.styles.scrollbar_size_horizontal == 1
        assert finalized_fence.show_horizontal_scrollbar is True


@pytest.mark.anyio
@pytest.mark.parametrize(
    "theme",
    [TAU_DARK_THEME, TAU_LIGHT_THEME, HIGH_CONTRAST_THEME],
    ids=lambda theme: theme.name,
)
async def test_transcript_code_fence_uses_theme_background(theme: TuiTheme) -> None:
    # [FUNCTION DOC] Parametrized async over built-in themes: verifies code fences use the theme's code-block
    # background (regression for Textual light-mode rule precedence).
    """Code fences keep Tau's themed background for every built-in theme.

    Regression test: Textual's built-in `MarkdownFence:light` rule outranks
    Tau's plain `ThemedMarkdownWidget MarkdownFence` selector, which used to
    leave light-theme code fences with a transparent white background.
    """
    app = TauTuiApp(
        FakeSession([AssistantMessage(content="```python\nx = 1\n```")]),
        tui_settings=TuiSettings(theme=theme.name),
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        fence = app.query_one("MarkdownFence")

    assert fence.styles.background == Color.parse(theme.markdown_code_block_background)


def test_tui_app_uses_configured_theme_css_variables() -> None:
    # [FUNCTION DOC] Verifies the app's CSS variables and Textual theme follow the configured high-contrast theme.
    app = TauTuiApp(FakeSession(), tui_settings=TuiSettings(theme="high-contrast"))

    variables = app.get_theme_variable_defaults()

    assert variables["tau-screen-background"] == "#000000"
    assert variables["tau-prompt-background"] == "#1a1a1a"
    assert variables["tau-prompt-border"] == "#00ff66"
    assert app.theme == "high-contrast"
    assert app.current_theme.name == "high-contrast"


def test_tui_app_registers_and_applies_custom_theme() -> None:
    # [FUNCTION DOC] Verifies a registered custom theme is applied by name and supplies CSS variables.
    # Side effects: sets the global custom-theme registry, reset in a finally block.
    from tau_coding.tui.themes import (
        THEME_COLOR_FIELDS,
        TRANSCRIPT_ROLES,
        parse_tui_theme_json,
        set_custom_tui_themes,
    )

    theme_data = {
        "name": "midnight",
        "colors": dict.fromkeys(THEME_COLOR_FIELDS, "#123456"),
        "roles": {role: {"border": "#123456", "body": "#e0e0e0"} for role in TRANSCRIPT_ROLES},
    }
    set_custom_tui_themes({"midnight": parse_tui_theme_json(theme_data)})
    try:
        app = TauTuiApp(FakeSession(), tui_settings=TuiSettings(theme="midnight"))

        assert app.theme == "midnight"
        assert app.get_theme_variable_defaults()["tau-screen-background"] == "#123456"
    finally:
        set_custom_tui_themes({})


@pytest.mark.anyio
async def test_tui_app_removes_source_project_themes_after_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC] Async: verifies resuming into another session reloads custom themes (patched to none),
    # falling back to tau-dark and unregistering the project theme while keeping the user's
    # configured theme name. Side effects: global theme registry, reset in finally.
    from tau_coding.tui.themes import (
        THEME_COLOR_FIELDS,
        TRANSCRIPT_ROLES,
        available_tui_theme_names,
        parse_tui_theme_json,
        set_custom_tui_themes,
    )

    theme_data = {
        "name": "project-theme",
        "colors": dict.fromkeys(THEME_COLOR_FIELDS, "#123456"),
        "roles": {role: {"border": "#123456", "body": "#e0e0e0"} for role in TRANSCRIPT_ROLES},
    }
    project_theme = parse_tui_theme_json(theme_data)
    set_custom_tui_themes({"project-theme": project_theme})
    monkeypatch.setattr(tui_app, "load_custom_tui_themes", lambda _dirs: ({}, []))
    session = FakeSession()
    session.theme_dirs = ()
    app = TauTuiApp(session, tui_settings=TuiSettings(theme="project-theme"))
    try:
        async with app.run_test() as pilot:
            await app._resume_session("destination-session")
            await pilot.pause()

            assert app.theme == "tau-dark"
            assert "project-theme" not in app.available_themes
            assert "project-theme" not in available_tui_theme_names()
            assert app.tui_settings.theme == "project-theme"
    finally:
        set_custom_tui_themes({})


def test_tui_app_falls_back_to_tau_dark_when_theme_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC] Verifies a missing configured theme falls back to tau-dark with a status message and does
    # not persist the fallback to settings. Inputs: monkeypatch, tmp_path (isolated home).
    isolate_home(monkeypatch, tmp_path)

    app = TauTuiApp(FakeSession(), tui_settings=TuiSettings(theme="missing-theme"))

    assert app.theme == "tau-dark"
    assert any("missing-theme" in item.text for item in app.state.items if item.role == "status")
    # The fallback must not be persisted over the user's configured theme:
    # if the theme file reappears, their choice should be honored again.
    assert not tui_settings_path().exists()


def test_tui_app_uses_light_theme_css_variables() -> None:
    # [FUNCTION DOC] Verifies the light theme's CSS variable values including footer colors.
    app = TauTuiApp(FakeSession(), tui_settings=TuiSettings(theme="tau-light"))

    variables = app.get_theme_variable_defaults()

    assert variables["tau-screen-background"] == "#ffffff"
    assert variables["tau-chrome-background"] == "#f3f4f6"
    assert variables["tau-muted-text"] == "#475569"
    assert variables["tau-prompt-background"] == "#f8fafc"
    assert variables["tau-prompt-border"] == "#2563eb"
    assert variables["footer-background"] == "#f3f4f6"
    assert variables["footer-foreground"] == "#111827"
    assert variables["footer-description-foreground"] == "#111827"
    assert variables["footer-key-foreground"] == "#0f766e"
    assert app.current_theme.dark is False


@pytest.mark.anyio
@pytest.mark.parametrize(
    "theme",
    [TAU_DARK_THEME, TAU_LIGHT_THEME, HIGH_CONTRAST_THEME],
    ids=lambda theme: theme.name,
)
async def test_list_view_scrollbars_use_theme_colors(theme: TuiTheme) -> None:
    # [FUNCTION DOC] Parametrized async over built-in themes: verifies picker option-list scrollbars use the
    # theme's transcript background, border, highlight and accent colors with width 2.
    app = TauTuiApp(FakeSession(), tui_settings=TuiSettings(theme=theme.name))

    async with app.run_test() as pilot:
        await app.push_screen(
            SessionPickerScreen([], local_cwd=Path("/workspace/project"), theme=theme)
        )
        await pilot.pause()

        list_view = app.screen.query_one("#session-picker-list", OptionList)

        assert list_view.styles.scrollbar_background == Color.parse(theme.transcript_background)
        assert list_view.styles.scrollbar_color == Color.parse(theme.border)
        assert list_view.styles.scrollbar_background_hover == Color.parse(
            theme.transcript_background
        )
        assert list_view.styles.scrollbar_color_hover == Color.parse(theme.highlight_background)
        assert list_view.styles.scrollbar_background_active == Color.parse(
            theme.transcript_background
        )
        assert list_view.styles.scrollbar_color_active == Color.parse(theme.accent)
        assert list_view.styles.scrollbar_size_vertical == 2


def test_tui_app_registers_only_tau_themes_with_textual() -> None:
    # [FUNCTION DOC] Verifies only the three built-in Tau themes are registered with Textual.
    app = TauTuiApp(FakeSession())

    assert tuple(app.available_themes) == ("tau-dark", "tau-light", "high-contrast")


def test_textual_theme_mapping_uses_tau_theme_values() -> None:
    # [FUNCTION DOC] Verifies the Tau-to-Textual theme mapping copies accent, dark flag and CSS variables.
    textual_theme = _textual_theme_for_tau_theme("tau-light")

    assert textual_theme.name == "tau-light"
    assert textual_theme.primary == TAU_LIGHT_THEME.accent
    assert textual_theme.dark is False
    assert textual_theme.variables["tau-screen-background"] == TAU_LIGHT_THEME.screen_background


def test_tau_dark_theme_uses_aqua_as_its_shared_accent() -> None:
    # [FUNCTION DOC] Verifies the default dark theme uses aqua as its shared accent for highlights, headings
    # and bullets, black backgrounds, and the prompt background for user message bodies.
    theme = TuiSettings().resolved_theme

    assert theme.accent == "#a7f3f0"
    assert theme.highlight_background == theme.accent
    assert theme.markdown_heading == theme.accent
    assert theme.markdown_bullet == theme.accent
    assert theme.screen_background == "#000000"
    assert theme.transcript_background == "#000000"
    assert theme.prompt_background == "#101419"
    assert theme.role_styles["user"].body.endswith(f"on {theme.prompt_background}")
    assert theme.role_styles["assistant"].body.endswith("on #000000")


@pytest.mark.parametrize(
    "theme",
    [TAU_DARK_THEME, TAU_LIGHT_THEME, HIGH_CONTRAST_THEME],
)
def test_autocomplete_and_picker_share_theme_highlight(theme: TuiTheme) -> None:
    # [FUNCTION DOC] Parametrized: verifies completion selection styles and CSS highlight variables share the
    # theme's highlight text/background colors.
    completion_text, completion_background = _split_rich_style_colors(theme.completion_selected)
    description_text, description_background = _split_rich_style_colors(
        theme.completion_selected_description
    )
    variables = _theme_css_variables(theme)

    assert completion_text == theme.highlight_text
    assert completion_background == theme.highlight_background
    assert description_text == theme.highlight_text
    assert description_background == theme.highlight_background
    assert variables["tau-highlight-background"] == theme.highlight_background
    assert variables["tau-highlight-text"] == theme.highlight_text


def test_tau_light_theme_uses_light_chat_backgrounds() -> None:
    # [FUNCTION DOC] Verifies the light theme's background, text, syntax theme and role body colors.
    theme = TuiSettings(theme="tau-light").resolved_theme

    assert theme.screen_background == "#ffffff"
    assert theme.transcript_background == "#ffffff"
    assert theme.prompt_text == "#111827"
    assert theme.markdown_heading == theme.accent
    assert theme.markdown_bullet == theme.accent
    assert theme.syntax_theme == "ansi_light"
    assert theme.role_styles["user"].body == f"#111827 on {theme.prompt_background}"
    assert theme.role_styles["assistant"].body == "#111827"
    assert theme.role_styles["tool"].body == "#1f2937"
    assert theme.role_styles["error"].border == "#b91c1c"


def test_tui_app_loads_restored_messages_into_display_state() -> None:
    # [FUNCTION DOC] Verifies restored session messages are projected into display items, including an edit
    # tool result combined with its patch details.
    app = TauTuiApp(
        FakeSession(
            messages=[
                UserMessage(content="Read the file"),
                AssistantMessage(
                    content=assistant_content(
                        "I'll inspect it.",
                        [ToolCall(id="call-1", name="edit", arguments={"path": "README.md"})],
                    )
                ),
                ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="edit",
                    content=[TextContent(text="Successfully replaced 1 block.")],
                    details={"patch": "--- README.md\n+++ README.md\n@@\n-old\n+new"},
                ),
            ]
        )
    )

    assert [(item.role, item.text, item.tool_result_text) for item in app.state.items] == [
        ("user", "Read the file", None),
        ("assistant", "I'll inspect it.", None),
        (
            "tool",
            "→ edit README.md",
            "✓ edit\n"
            "Successfully replaced 1 block.\n"
            "\n"
            "Patch:\n"
            "--- README.md\n"
            "+++ README.md\n"
            "@@\n"
            "-old\n"
            "+new",
        ),
    ]


@pytest.mark.anyio
async def test_tui_app_shows_activity_indicator_while_running() -> None:
    # [FUNCTION DOC] Async: verifies the prompt prefix shows "τ" when idle and an animated block indicator
    # while running (advancing on tick), the prompt left border stays the neutral color, no
    # status widgets exist, and activity timing constants have expected values.
    app = TauTuiApp(FakeSession())

    async with app.run_test():
        prompt = app.query_one("#prompt")
        indicator = app.query_one("#prompt-prefix")

        assert not app.query("#status")
        assert not app.query("#activity-status")
        assert prompt.styles.border_left[1].hex.lower() == "#2d3748"
        assert prompt.styles.border_top[0] == ""
        assert prompt.styles.border_right[0] == ""
        assert prompt.styles.border_bottom[0] == ""
        assert indicator.render().plain == "τ"

        app.adapter.apply(AgentStartEvent())
        app._refresh()

        assert pytest.approx(tui_app.ACTIVITY_TICK_SECONDS) == 0.15
        assert tui_app.ACTIVITY_COLOR_FADE_STEPS == 24
        assert prompt.styles.border_left[1].hex.lower() == "#2d3748"
        assert indicator.render().plain.startswith("■")

        app._tick_activity()

        assert prompt.styles.border_left[1].hex.lower() == "#2d3748"
        assert indicator.render().plain.splitlines()[1] == "■"

        app.adapter.apply(AgentEndEvent())
        app._refresh()

        assert not app.query("#status")
        assert prompt.styles.border_left[1].hex.lower() == "#2d3748"
        assert indicator.render().plain == "τ"


@pytest.mark.anyio
async def test_tui_app_updates_terminal_title_for_running_and_named_session() -> None:
    # [FUNCTION DOC] Async: verifies the terminal title (OSC 0) shows the session title, a spinner frame while
    # running that advances on tick, updates on rename, and resets to plain "τ" on exit.
    # Uses a TerminalTitleController writing to a list.
    session = FakeSession()
    session._session_title = "build notes"
    app = TauTuiApp(session)
    writes: list[str] = []
    app._terminal_title = TerminalTitleController(enabled=True, writer=writes.append)

    async with app.run_test():
        assert writes[-1] == "\x1b]0;τ | build notes\x07"

        app.adapter.apply(AgentStartEvent())
        app._refresh()
        assert writes[-1] == "\x1b]0;⠋ τ | build notes\x07"

        app._tick_activity()
        assert writes[-1] == "\x1b]0;⠙ τ | build notes\x07"

        session._session_title = "ship notes"
        app._refresh_chrome()
        assert writes[-1] == "\x1b]0;⠙ τ | ship notes\x07"

        app.adapter.apply(AgentEndEvent())
        app._refresh()
        assert writes[-1] == "\x1b]0;τ | ship notes\x07"

    assert writes[-1] == "\x1b]0;τ\x07"


@pytest.mark.anyio
async def test_tui_app_notifies_when_agent_settles_while_unfocused() -> None:
    # [FUNCTION DOC] Async: verifies a desktop turn notification (OSC 9) is emitted only when the agent
    # settles while the app is unfocused.
    class SettledSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession variant whose prompt yields agent start, agent end and an
        # AgentSettledEvent.
        async def prompt(
            self,
            text: str,
            *,
            streaming_behavior: str | None = None,
            source: str = "interactive",
            custom_type: str | None = None,
            details: dict[str, object] | None = None,
        ) -> AsyncIterator[CodingSessionEvent]:
            # [FUNCTION DOC] HELPER: async generator; records the prompt text and yields start, end and settled
            # events, ignoring the other arguments.
            del streaming_behavior, source, custom_type, details
            self.prompt_texts.append(text)
            yield AgentStartEvent()
            yield AgentEndEvent()
            yield AgentSettledEvent()

    app = TauTuiApp(SettledSession(), tui_settings=TuiSettings(turn_notification="desktop"))
    writes: list[str] = []
    app._terminal_notification = TerminalNotificationController(
        "desktop",
        enabled=True,
        writer=writes.append,
        environ={"TERM_PROGRAM": "ghostty"},
    )

    async with app.run_test():
        await app._run_prompt("focused")
        assert writes == []

        app.on_app_blur()
        await app._run_prompt("background")
        assert writes == ["\x1b]9;Tau turn finished\x07"]

        app.on_app_focus()
        await app._run_prompt("focused again")
        assert writes == ["\x1b]9;Tau turn finished\x07"]


@pytest.mark.anyio
async def test_tui_app_updates_terminal_title_after_auto_session_naming() -> None:
    # [FUNCTION DOC] Async: verifies a session auto-named during a prompt updates the terminal title and the
    # sidebar content.
    class AutoNamingSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession variant that sets its title to "Debug login" mid-prompt.
        async def prompt(
            self,
            text: str,
            *,
            streaming_behavior: str | None = None,
            source: str = "interactive",
            custom_type: str | None = None,
            details: dict[str, object] | None = None,
        ) -> AsyncIterator[AgentEvent]:
            # [FUNCTION DOC] HELPER: async generator; records text, yields agent start, sets the session title, then
            # yields the user MessageEndEvent and agent end.
            del streaming_behavior, source, custom_type, details
            self.prompt_texts.append(text)
            yield AgentStartEvent()
            self._session_title = "Debug login"
            yield MessageEndEvent(message=UserMessage(content=text))
            yield AgentEndEvent()

    app = TauTuiApp(AutoNamingSession())
    writes: list[str] = []
    app._terminal_title = TerminalTitleController(enabled=True, writer=writes.append)

    async with app.run_test():
        assert writes[-1] == "\x1b]0;τ\x07"

        await app._run_prompt("debug the login flow")

        assert "\x1b]0;τ | Debug login\x07" in writes
        sidebar_content = app.query_one("#sidebar-content", Static)
        console = Console(record=True, width=80, file=StringIO())
        console.print(sidebar_content.content)
        assert "Debug login" in console.export_text()
        assert writes[-1] == "\x1b]0;τ | Debug login\x07"


@pytest.mark.anyio
async def test_tui_app_clears_activity_status_on_error() -> None:
    # [FUNCTION DOC] Async: verifies an assistant error message clears the running indicator back to idle.
    app = TauTuiApp(FakeSession())

    async with app.run_test():
        prompt = app.query_one("#prompt")
        indicator = app.query_one("#prompt-prefix")
        app.adapter.apply(AgentStartEvent())
        app._refresh()
        app.adapter.apply(
            MessageEndEvent(
                message=AssistantMessage(stop_reason="error", error_message="provider failed")
            )
        )
        app._refresh()

        assert not app.query("#status")
        assert not app.query("#activity-status")
        assert prompt.styles.border_left[1].hex.lower() == "#2d3748"
        assert prompt.styles.border_top[0] == ""
        assert prompt.styles.border_right[0] == ""
        assert prompt.styles.border_bottom[0] == ""
        assert indicator.render().plain == "τ"


@pytest.mark.anyio
async def test_textual_theme_change_persists_tau_theme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC] Async: verifies changing the Textual app theme persists the Tau theme to settings.
    # Inputs: tmp_path, monkeypatch (isolated home). Side effects: writes settings file.
    isolate_home(monkeypatch, tmp_path)
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        app.theme = "tau-light"
        await pilot.pause()

    assert app.tui_settings.theme == "tau-light"
    assert tui_settings_path().read_text(encoding="utf-8").find('"theme": "tau-light"') != -1


@pytest.mark.anyio
async def test_tui_app_skills_picker_filters_and_inserts_without_submitting() -> None:
    # [FUNCTION DOC] Async: verifies /skills opens a picker sorted by name with search focus, arrow navigation,
    # an empty-search message, description-based filtering, and that enter inserts
    # "/skill:NAME" into the prompt without submitting.
    session = FakeSession()
    session.skills = (
        Skill("zebra", Path("/skills/zebra/SKILL.md"), "", "Work with stripes"),
        Skill("alpha", Path("/skills/alpha/SKILL.md"), "", "Review Python code"),
    )
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.text = "/skills"
        await pilot.press("enter")
        await pilot.pause()

        picker = app.screen
        assert isinstance(picker, SkillPickerScreen)
        assert [
            [label.render().plain for label in item.query(Label)] for item in picker.query(ListItem)
        ] == [
            ["alpha", "Review Python code"],
            ["zebra", "Work with stripes"],
        ]
        assert picker.query_one("#skill-picker-search", Input).has_focus

        skill_list = picker.query_one("#skill-picker-list", ListView)
        assert skill_list.index == 0
        await pilot.press("down")
        await pilot.pause()
        assert skill_list.index == 1
        await pilot.press("up")
        await pilot.pause()
        assert skill_list.index == 0

        search = picker.query_one("#skill-picker-search", Input)
        search.value = "missing"
        await pilot.pause()
        assert not picker.query(ListItem)
        assert (
            picker.query_one("#skill-picker-help", Static)
            .render()
            .plain.startswith("No matching skills")
        )

        search.value = ""
        await pilot.press("p", "y", "t", "h", "o", "n", "space", "c", "o", "d", "e")
        await pilot.pause()
        assert search.value == "python code"
        assert [label.render().plain for label in picker.query(Label)] == [
            "alpha",
            "Review Python code",
        ]
        await pilot.press("enter")
        await pilot.pause()

        assert prompt.text == "/skill:alpha"
        assert prompt.has_focus
        assert session.prompt_texts == []


@pytest.mark.anyio
async def test_tui_app_skills_picker_previews_description_and_shows_content_in_transcript() -> None:
    # [FUNCTION DOC] Async: verifies F1 in the skills picker previews the description in an output screen, and
    # ctrl+enter shows the skill content in the transcript as a status item without adding it
    # to context or prompting.
    session = FakeSession()
    session.skills = (
        Skill(
            "review",
            Path("/skills/review/SKILL.md"),
            "# Review\n\nInspect every changed file.",
            "Review changes carefully across the whole repository.",
        ),
    )
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.text = "/skills"
        await pilot.press("enter")
        await pilot.pause()

        picker = app.screen
        assert isinstance(picker, SkillPickerScreen)
        await pilot.press("f1")
        await pilot.pause()

        description = app.screen
        assert isinstance(description, CommandOutputScreen)
        assert description.query_one("#command-output-body", Static).render().plain == (
            "Review changes carefully across the whole repository."
        )
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is picker

        await pilot.press("ctrl+enter")
        await pilot.pause()

        assert app.screen is not picker
        assert app.state.items[-1].role == "status"
        assert app.state.items[-1].text == (
            "Skill: review (not added to context)\n# Review\n\nInspect every changed file."
        )
        assert prompt.text == ""
        assert prompt.has_focus
        assert session.prompt_texts == []
        assert session.messages == ()


@pytest.mark.anyio
async def test_tui_app_skills_picker_cancel_clears_prompt_and_shows_empty_states() -> None:
    # [FUNCTION DOC] Async: verifies the empty skills picker shows "No skills loaded" and escape clears the prompt.
    session = FakeSession()
    session.skills = ()
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.text = "/skills"
        await pilot.press("enter")
        await pilot.pause()

        picker = app.screen
        assert isinstance(picker, SkillPickerScreen)
        assert (
            picker.query_one("#skill-picker-help", Static)
            .render()
            .plain.startswith("No skills loaded")
        )
        await pilot.press("escape")
        await pilot.pause()

        assert prompt.text == ""
        assert prompt.has_focus
        assert session.prompt_texts == []


@pytest.mark.anyio
async def test_tui_app_theme_command_opens_picker_and_persists_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC] Async: verifies /theme opens a picker marking the current theme, supports arrow
    # navigation, and selecting applies and persists the theme. Inputs: tmp_path, monkeypatch.
    isolate_home(monkeypatch, tmp_path)
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/theme"
        await pilot.press("enter")
        await pilot.pause()

        picker = app.screen
        assert isinstance(picker, ThemePickerScreen)
        assert [str(item.query_one(Label).render()) for item in picker.query(ListItem)] == [
            "✓ tau-dark",
            "  tau-light",
            "  high-contrast",
        ]

        theme_list = picker.query_one("#theme-picker-list", ListView)
        assert theme_list.index == 0
        await pilot.press("down")
        await pilot.pause()
        assert theme_list.index == 1
        await pilot.press("up")
        await pilot.pause()
        assert theme_list.index == 0
        await pilot.press("down", "enter")
        await pilot.pause()

        assert app.tui_settings.theme == "tau-light"
        assert app.theme == "tau-light"
        assert tui_settings_path().read_text(encoding="utf-8").find('"theme": "tau-light"') != -1
        assert app.get_theme_variable_defaults()["tau-screen-background"] == "#ffffff"


@pytest.mark.parametrize(
    "theme",
    [TAU_DARK_THEME, TAU_LIGHT_THEME, HIGH_CONTRAST_THEME],
)
@pytest.mark.anyio
async def test_theme_picker_highlight_uses_theme_selection_palette(theme: TuiTheme) -> None:
    # [FUNCTION DOC] Parametrized async: verifies the theme picker highlighted row uses the theme's highlight colors.
    theme_names = (TAU_DARK_THEME.name, TAU_LIGHT_THEME.name, HIGH_CONTRAST_THEME.name)
    app = TauTuiApp(FakeSession(), tui_settings=TuiSettings(theme=theme.name))

    async with app.run_test() as pilot:
        picker = ThemePickerScreen(
            current_theme=theme.name,
            theme=theme,
            theme_names=theme_names,
        )
        app.push_screen(picker)
        await pilot.pause()

        highlighted_item = picker.query_one("ListItem.-highlight", ListItem)
        highlighted_label = highlighted_item.query_one(Label)
        assert highlighted_label.styles.background == Color.parse(theme.highlight_background)
        assert highlighted_label.styles.color == Color.parse(theme.highlight_text)


@pytest.mark.anyio
async def test_extension_select_dialog_returns_choice() -> None:
    # [FUNCTION DOC] Async: verifies the extension UI bridge select dialog returns the option chosen via arrow
    # keys and enter.
    app = TauTuiApp(FakeSession())  # type: ignore[arg-type]

    async with app.run_test() as pilot:
        await pilot.pause()
        bridge = _TuiExtensionUiBridge(app)
        task = asyncio.ensure_future(bridge.select("Pick", ["alpha", "beta"]))
        await pilot.pause()

        assert isinstance(app.screen, ExtensionSelectScreen)
        # Arrow down to the second option, then confirm with Enter — real
        # key presses, exercising the app-level Up/Down routing.
        await pilot.press("down")
        await pilot.press("enter")
        result = await task

    assert result == "beta"


@pytest.mark.anyio
async def test_extension_select_dialog_escape_returns_none() -> None:
    # [FUNCTION DOC] Async: verifies escape in the select dialog returns None.
    app = TauTuiApp(FakeSession())  # type: ignore[arg-type]

    async with app.run_test() as pilot:
        await pilot.pause()
        bridge = _TuiExtensionUiBridge(app)
        task = asyncio.ensure_future(bridge.select("Pick", ["alpha", "beta"]))
        await pilot.pause()
        await pilot.press("escape")
        result = await task

    assert result is None


@pytest.mark.anyio
async def test_extension_confirm_dialog_yes_and_cancel() -> None:
    # [FUNCTION DOC] Async: verifies the confirm dialog returns True for Yes, False for No, and False on escape.
    app = TauTuiApp(FakeSession())  # type: ignore[arg-type]

    async with app.run_test() as pilot:
        await pilot.pause()
        bridge = _TuiExtensionUiBridge(app)

        yes_task = asyncio.ensure_future(bridge.confirm("Ship?", "to prod"))
        await pilot.pause()
        assert isinstance(app.screen, ExtensionConfirmScreen)
        await pilot.press("enter")  # "Yes" is highlighted first
        assert await yes_task is True

        no_task = asyncio.ensure_future(bridge.confirm("Ship?", "to prod"))
        await pilot.pause()
        assert isinstance(app.screen, ExtensionConfirmScreen)
        await pilot.press("down")  # arrow highlight to "No"
        await pilot.press("enter")
        assert await no_task is False

        cancel_task = asyncio.ensure_future(bridge.confirm("Ship?", "to prod"))
        await pilot.pause()
        await pilot.press("escape")
        assert await cancel_task is False


@pytest.mark.anyio
async def test_local_modals_receive_app_level_arrow_navigation() -> None:
    # [FUNCTION DOC] Async: verifies local backend modals get app-level arrow routing: model list and action
    # menu navigation, escape returning focus to the prompt, and the confirm screen defaulting
    # to "No". Builds real dynamic provider and local backend registries; closes the registry.
    providers = DynamicProviderRegistry(generation_id="local-navigation")
    providers.register(
        "source",
        DynamicProvider(
            id="local-provider",
            display_name="Local provider",
            models=(ProviderModel("first"), ProviderModel("second")),
            default_model="first",
            transport=OpenAICompatibleTransport(
                base_url="http://example.test/v1",
                auth=NoAuth(),
            ),
        ),
    )
    registry = LocalBackendRegistry(providers, generation_id="local-navigation")

    async def status(context):
        # [FUNCTION DOC] HELPER: async fake backend status callback returning a ready status with two unloaded
        # models and configure/refresh actions.
        del context
        return LocalBackendStatus(
            state="ready",
            models=(
                LocalModel("first", state="unloaded"),
                LocalModel("second", state="unloaded"),
            ),
            actions=("configure", "refresh"),
        )

    registry.register(
        "source",
        LocalBackend(
            id="local",
            provider_id="local-provider",
            display_name="Local",
            configure_spec=LocalConfigureSpec(),
            configure=lambda values, context: LocalConfigureResult(committed=True),
            status=status,
            refresh=status,
        ),
    )
    app = TauTuiApp(FakeSession())  # type: ignore[arg-type]

    async with app.run_test() as pilot:
        await pilot.pause()
        app.session.extension_runtime = SimpleNamespace(local_backend_registry=registry)
        prompt = app.query_one("#prompt", PromptInput)
        prompt.focus()
        app._open_local_backend_picker()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, LocalBackendScreen)
        assert len(app.screen_stack) == 2
        model_list = app.screen.query_one("#local-model-list", ListView)
        action_menu = app.screen.query_one("#local-action-menu", ListView)
        assert model_list.has_focus
        assert model_list.index == 0
        assert action_menu.index == 0
        await pilot.press("down")
        assert model_list.index == 1
        await pilot.press("up")
        assert model_list.index == 0
        await pilot.press("down", "down")
        assert action_menu.has_focus
        assert action_menu.index == 0
        await pilot.press("up")
        assert model_list.has_focus
        assert model_list.index == 1

        await pilot.press("escape")
        await pilot.pause()
        assert len(app.screen_stack) == 1
        assert app.focused is prompt
        await pilot.press("x")
        assert prompt.text == "x"

        selected: list[bool | None] = []
        app.push_screen(
            LocalConfirmScreen("Load model?", "This is expensive.", theme=TAU_DARK_THEME),
            callback=selected.append,
        )
        await pilot.pause()
        choices = app.screen.query_one("#local-confirm-list", ListView)
        assert choices.index == 1  # No is the safe default.
        await pilot.press("up")
        await pilot.press("enter")
        assert selected == [True]

    await registry.aclose()


@pytest.mark.anyio
async def test_extension_input_dialog_returns_text() -> None:
    # [FUNCTION DOC] Async: verifies the extension input dialog returns typed text on enter.
    app = TauTuiApp(FakeSession())  # type: ignore[arg-type]

    async with app.run_test() as pilot:
        await pilot.pause()
        bridge = _TuiExtensionUiBridge(app)
        task = asyncio.ensure_future(bridge.input("Name", "hint"))
        await pilot.pause()

        assert isinstance(app.screen, ExtensionInputScreen)
        await pilot.press("h", "i")
        await pilot.press("enter")
        result = await task

    assert result == "hi"


@pytest.mark.anyio
async def test_extension_input_dialog_escape_returns_none() -> None:
    # [FUNCTION DOC] Async: verifies escape in the input dialog returns None.
    app = TauTuiApp(FakeSession())  # type: ignore[arg-type]

    async with app.run_test() as pilot:
        await pilot.pause()
        bridge = _TuiExtensionUiBridge(app)
        task = asyncio.ensure_future(bridge.input("Name", "hint"))
        await pilot.pause()
        await pilot.press("escape")
        result = await task

    assert result is None


class _RenderCallRuntime:
    # [CLASS DOC] HELPER: stub extension runtime for connecting to the app; accepts UI bridge and turn
    # callbacks without storing them, renders no custom messages, renders "agent" tool calls as
    # a friendly line from the description, and renders agent results with details as a
    # completion line (with " · expanded" suffix when expanded).
    """Minimal extension-runtime stub for `_connect_extension_runtime`."""

    def set_ui_bridge(self, bridge) -> None:  # noqa: ANN001
        # [FUNCTION DOC] HELPER: ignores the provided UI bridge.
        del bridge

    def set_turn_requested_callback(self, callback) -> None:  # noqa: ANN001
        # [FUNCTION DOC] HELPER: ignores the provided turn-requested callback.
        del callback

    def render_custom_message(self, custom_type, content, details, expanded):  # noqa: ANN001
        # [FUNCTION DOC] HELPER: custom-message renderer that always returns None (no markup).
        return None

    def render_tool_call(self, name, arguments):  # noqa: ANN001
        # [FUNCTION DOC] HELPER: returns "▸ agent · DESCRIPTION" for the agent tool, else None.
        if name == "agent":
            return f"▸ agent · {arguments.get('description')}"
        return None

    def render_tool_result(self, tool_name, result, expanded):  # noqa: ANN001
        # [FUNCTION DOC] HELPER: for agent results with details returns Rich markup "✓ DESCRIPTION completed"
        # (plus " · expanded" when expanded); otherwise None.
        if tool_name == "agent" and result.details:
            suffix = " · expanded" if expanded else ""
            return f"[green]✓[/green] {result.details.get('description')} completed{suffix}"
        return None


class _CustomMessageRuntime(_RenderCallRuntime):
    # [CLASS DOC] HELPER: runtime stub that additionally renders subagent-notification custom messages
    # with details into a completion card.
    """Runtime stub whose custom renderer produces a card for notifications."""

    def render_custom_message(self, custom_type, content, details, expanded):  # noqa: ANN001
        # [FUNCTION DOC] HELPER: returns "✓ DESCRIPTION completed" markup for subagent notifications with
        # details, else None.
        del content, expanded
        if custom_type != "subagent-notification" or not details:
            return None
        return f"[green]✓[/green] {details.get('description')} completed"


@pytest.mark.anyio
async def test_restored_tool_calls_render_via_render_call() -> None:
    # [FUNCTION DOC] Async: verifies restored agent tool calls render lazily via the extension's render_call
    # hook (friendly line, not raw prompt), even though messages load before the runtime connects.
    # Real startup order: session messages load into TuiState BEFORE the
    # extension runtime connects, so the friendly line only appears if tool
    # invocations resolve lazily at render time (not baked in at load).
    session = FakeSession(
        messages=[
            AssistantMessage(
                content=assistant_content(
                    "",
                    [
                        ToolCall(
                            id="call-1",
                            name="agent",
                            arguments={
                                "prompt": "long prompt",
                                "description": "Summarize codebase",
                            },
                        )
                    ],
                )
            )
        ]
    )
    session.extension_runtime = _RenderCallRuntime()
    app = TauTuiApp(session)  # type: ignore[arg-type]

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        widget = next(w for w in app.query(TranscriptMessageWidget) if w.item.role == "tool")
        assert "▸ agent · Summarize codebase" in widget.selection_text
        assert "long prompt" not in widget.selection_text


@pytest.mark.anyio
async def test_render_call_line_composes_with_live_update_text() -> None:
    # [FUNCTION DOC] Async: verifies the render_call line is combined with live progress text from tool updates.
    session = FakeSession()
    session.extension_runtime = _RenderCallRuntime()
    app = TauTuiApp(session)  # type: ignore[arg-type]

    async def stream(event: AgentEvent) -> None:
        # [FUNCTION DOC] HELPER: applies an event to the adapter then the streaming transcript.
        app.adapter.apply(event)
        await app._apply_streaming_transcript_event(event)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await stream(
            ToolExecutionStartEvent(
                tool_call_id="call-1",
                tool_name="agent",
                args={"prompt": "x", "description": "Summarize codebase"},
            )
        )
        await stream(
            ToolExecutionUpdateEvent(
                tool_call_id="call-1",
                tool_name="agent",
                args={},
                partial_result=AgentToolResult(content="agent-1: bash · turn 1"),
            )
        )
        await pilot.pause()

        widget = next(w for w in app.query(TranscriptMessageWidget) if w.item.role == "tool")
        assert "▸ agent · Summarize codebase" in widget.selection_text
        assert "agent-1: bash · turn 1" in widget.selection_text


@pytest.mark.anyio
async def test_tool_result_renders_via_render_result() -> None:
    # [FUNCTION DOC] Async: verifies an extension render_result hook replaces the generic tool result block on
    # the collapsed row (keeping the render_call line) and its expanded variant is used after
    # toggling tool results; raw result text never shows.
    session = FakeSession()
    session.extension_runtime = _RenderCallRuntime()
    app = TauTuiApp(session)  # type: ignore[arg-type]

    async def stream(event: AgentEvent) -> None:
        # [FUNCTION DOC] HELPER: applies an event to the adapter then the streaming transcript.
        app.adapter.apply(event)
        await app._apply_streaming_transcript_event(event)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await stream(
            ToolExecutionStartEvent(
                tool_call_id="call-1",
                tool_name="agent",
                args={"prompt": "x", "description": "Summarize codebase"},
            )
        )
        await stream(
            ToolExecutionEndEvent(
                tool_call_id="call-1",
                tool_name="agent",
                is_error=False,
                result=AgentToolResult(
                    content=[TextContent(text="the raw result body")],
                    details={"description": "Summarize codebase"},
                ),
            )
        )
        await pilot.pause()

        widget = next(w for w in app.query(TranscriptMessageWidget) if w.item.role == "tool")
        # The tool's render_result markup replaces the generic result block on
        # the collapsed row; the render_call invocation line stays above it.
        assert "▸ agent · Summarize codebase" in widget.selection_text
        assert "✓ Summarize codebase completed" in widget.selection_text
        assert "the raw result body" not in widget.selection_text

        # Expanding tool results re-renders through the expanded variant.
        app.action_toggle_tool_results()
        await pilot.pause()
        widget = next(w for w in app.query(TranscriptMessageWidget) if w.item.role == "tool")
        assert "✓ Summarize codebase completed · expanded" in widget.selection_text
        assert "the raw result body" not in widget.selection_text


@pytest.mark.anyio
async def test_restored_tool_results_render_via_render_result() -> None:
    # [FUNCTION DOC] Async: verifies restored tool results also render lazily through render_result.
    # Real startup order: session messages load into TuiState BEFORE the
    # extension runtime connects, so the card only appears if tool results
    # resolve lazily at render time (not baked in at load).
    session = FakeSession(
        messages=[
            AssistantMessage(
                content=assistant_content(
                    "",
                    [
                        ToolCall(
                            id="call-1",
                            name="agent",
                            arguments={"prompt": "x", "description": "Summarize codebase"},
                        )
                    ],
                )
            ),
            ToolResultMessage(
                tool_call_id="call-1",
                tool_name="agent",
                content=[TextContent(text="the raw result body")],
                details={"description": "Summarize codebase"},
            ),
        ]
    )
    session.extension_runtime = _RenderCallRuntime()
    app = TauTuiApp(session)  # type: ignore[arg-type]

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        widget = next(w for w in app.query(TranscriptMessageWidget) if w.item.role == "tool")
        assert "✓ Summarize codebase completed" in widget.selection_text
        assert "the raw result body" not in widget.selection_text


@pytest.mark.anyio
async def test_pending_tool_row_keeps_static_marker_while_running() -> None:
    # [FUNCTION DOC] Async: verifies a pending tool row keeps a static marker while activity ticks (same
    # widget, no spinner), shows elapsed time "(1m 23s)" after the pending timer refresh, and
    # keeps its render_call line after completion.
    session = FakeSession()
    session.extension_runtime = _RenderCallRuntime()
    app = TauTuiApp(session)  # type: ignore[arg-type]

    async def stream(event: AgentEvent) -> None:
        # [FUNCTION DOC] HELPER: applies an event to the adapter then the streaming transcript.
        app.adapter.apply(event)
        await app._apply_streaming_transcript_event(event)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        app.state.running = True
        await stream(
            ToolExecutionStartEvent(
                tool_call_id="call-1",
                tool_name="agent",
                args={"prompt": "x", "description": "Summarize codebase"},
            )
        )

        widget = next(w for w in app.query(TranscriptMessageWidget) if w.item.role == "tool")
        assert widget.selection_text == "▸ agent · Summarize codebase"

        # The run-wide activity indicator continues ticking without replacing
        # the transcript marker with a second spinner.
        app._tick_activity()
        app._tick_activity()
        await pilot.pause()
        same_widget = next(w for w in app.query(TranscriptMessageWidget) if w.item.role == "tool")
        assert same_widget is widget
        assert same_widget.selection_text == "▸ agent · Summarize codebase"

        # Long-running tools retain useful elapsed-time feedback alongside the
        # same static marker.
        tool_item = next(item for item in app.state.items if item.role == "tool")
        assert tool_item.started_at is not None
        tool_item.started_at -= 83
        await app._refresh_pending_tool_timer()
        await pilot.pause()
        assert widget.selection_text == "▸ agent · Summarize codebase (1m 23s)"

        await stream(
            ToolExecutionEndEvent(
                tool_call_id="call-1",
                tool_name="agent",
                is_error=False,
                result=AgentToolResult(content="done"),
            )
        )
        await pilot.pause()
        widget = next(w for w in app.query(TranscriptMessageWidget) if w.item.role == "tool")
        assert widget.selection_text.startswith("▸ agent · Summarize codebase")


@pytest.mark.anyio
async def test_activity_animation_throttles_tool_timer_layout_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC] Async: verifies activity ticks schedule the pending-tool timer refresh at most once per
    # interval (no scheduling within the interval, one after it elapses).
    # Inputs: monkeypatch (replaces call_later with a recorder).
    app = TauTuiApp(FakeSession())
    scheduled: list[object] = []

    async with app.run_test() as pilot:
        await pilot.pause()
        app.state.running = True
        app._last_tool_timer_refresh_at = asyncio.get_running_loop().time()
        monkeypatch.setattr(app, "call_later", lambda callback, *args: scheduled.append(callback))

        app._tick_activity()
        app._tick_activity()
        app._tick_activity()
        assert scheduled == []

        app._last_tool_timer_refresh_at -= 1.0
        app._tick_activity()
        assert scheduled == [app._refresh_pending_tool_timer]


@pytest.mark.anyio
async def test_extension_select_dialog_timeout_returns_default() -> None:
    # [FUNCTION DOC] Async: verifies a select dialog with a short timeout auto-dismisses and returns None.
    app = TauTuiApp(FakeSession())  # type: ignore[arg-type]

    async with app.run_test() as pilot:
        await pilot.pause()
        bridge = _TuiExtensionUiBridge(app)
        # A tiny timeout elapses with no interaction; the dialog auto-dismisses.
        result = await bridge.select("Pick", ["alpha", "beta"], timeout=0.05)
        await pilot.pause()

    assert result is None


@pytest.mark.anyio
async def test_tui_app_theme_command_argument_updates_theme_and_persists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # [FUNCTION DOC] Async: verifies "/theme NAME" applies and persists the theme. Inputs: tmp_path, monkeypatch.
    isolate_home(monkeypatch, tmp_path)
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/theme tau-light"
        await pilot.press("enter")

        assert app.tui_settings.theme == "tau-light"
        assert app.theme == "tau-light"
        assert tui_settings_path().read_text(encoding="utf-8").find('"theme": "tau-light"') != -1
        assert app.get_theme_variable_defaults()["tau-screen-background"] == "#ffffff"


@pytest.mark.anyio
async def test_tui_app_new_command_starts_new_visible_state() -> None:
    # [FUNCTION DOC] Async: verifies /new starts a new session, clears visible items without a notification,
    # and leaves prompt history recall empty.
    app = TauTuiApp(FakeSession(messages=[UserMessage(content="Earlier")]))
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages in place of the app's _notify, ignoring options.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/new"
        await pilot.press("enter")

        assert app.session.new_session_count == 1
        assert app.state.items == []
        assert notifications == []

        await pilot.press("up")
        await pilot.pause()

        assert prompt.value == ""


@pytest.mark.anyio
async def test_tui_app_compact_command_runs_session_compaction() -> None:
    # [FUNCTION DOC] Async: verifies "/compact TEXT" calls session compaction with the text and the transcript
    # becomes a single compaction summary item.
    session = FakeSession(messages=[UserMessage(content="Earlier")])
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/compact Summary of earlier work."
        await pilot.press("enter")

        assert session.compact_summaries == ["Summary of earlier work."]
        assert [(item.role, item.text) for item in app.state.items] == [
            ("compaction_summary", "Compaction summary (Ctrl+O to expand)")
        ]
        assert app.state.items[0].tool_result_text == "Generated summary"


@pytest.mark.anyio
async def test_tui_app_compact_command_accepts_no_instructions() -> None:
    # [FUNCTION DOC] Async: verifies bare /compact passes an empty summary instruction.
    session = FakeSession(messages=[UserMessage(content="Earlier")])
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/compact"
        await pilot.press("enter")

        assert session.compact_summaries == [""]


@pytest.mark.anyio
@pytest.mark.parametrize("blocked_command", ["/new", "/resume abc123"])
async def test_tui_app_blocks_session_commands_while_compacting(blocked_command: str) -> None:
    # [FUNCTION DOC] Parametrized async: verifies /new and /resume are blocked with a notification (prompt text
    # kept) while a manual compaction is running, and compaction completes afterwards.
    started = asyncio.Event()
    finish = asyncio.Event()

    class SlowCompactSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession whose compaction blocks on an external event after signalling start.
        async def compact(self, summary: str) -> str:
            # [FUNCTION DOC] HELPER: async; records the summary, signals started, waits for finish, then replaces
            # messages with a summary message, sets token estimate 42 and returns a status string.
            self.compact_summaries.append(summary)
            started.set()
            await finish.wait()
            self.messages = (
                UserMessage(content="Previous conversation summary:\nGenerated summary"),
            )
            self.context_token_estimate = 42
            return "Compacted 2 context entries."

    session = SlowCompactSession(messages=[UserMessage(content="Earlier")])
    app = TauTuiApp(session)
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/compact Summary of earlier work."
        await pilot.press("enter")
        await asyncio.wait_for(started.wait(), timeout=1)

        prompt.value = blocked_command
        await pilot.press("enter")
        await pilot.pause()

        assert session.new_session_count == 0
        assert session.resumed_session_ids == []
        assert prompt.value == blocked_command
        assert notifications == [
            "Compaction is still running. You can keep editing, but wait to submit."
        ]

        finish.set()
        await pilot.pause()

        assert session.compact_summaries == ["Summary of earlier work."]


@pytest.mark.anyio
async def test_tui_app_blocks_compact_command_while_agent_is_running() -> None:
    # [FUNCTION DOC] Async: verifies /compact is refused with a notification while the agent is running.
    session = FakeSession()
    app = TauTuiApp(session)
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        app.state.running = True
        prompt = app.query_one("#prompt")
        prompt.value = "/compact Summary of earlier work."
        await pilot.press("enter")
        await pilot.pause()

        assert session.compact_summaries == []
        assert prompt.value == "/compact Summary of earlier work."
        assert notifications == [
            "Wait for the current agent turn and queued messages to finish before compacting."
        ]


@pytest.mark.anyio
async def test_tui_app_blocks_compact_command_while_follow_up_is_queued() -> None:
    # [FUNCTION DOC] Async: verifies /compact is refused while a follow-up message is queued.
    session = FakeSession()
    session.queued_follow_up_messages = ("after this",)
    app = TauTuiApp(session)
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        app._refresh()
        prompt = app.query_one("#prompt")
        prompt.value = "/compact Summary of earlier work."
        await pilot.press("enter")
        await pilot.pause()

        assert session.compact_summaries == []
        assert prompt.value == "/compact Summary of earlier work."
        assert notifications == [
            "Wait for the current agent turn and queued messages to finish before compacting."
        ]


@pytest.mark.anyio
async def test_tui_app_escape_cancels_active_compaction() -> None:
    # [FUNCTION DOC] Async: verifies escape cancels an active compaction (worker cleared, transcript unchanged,
    # "Cancelled compaction." notification) and later commands work normally.
    started = asyncio.Event()
    finish = asyncio.Event()

    class SlowCompactSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession whose compaction blocks until released.
        async def compact(self, summary: str) -> str:
            # [FUNCTION DOC] HELPER: async; records summary, signals start, waits for finish, then replaces messages
            # with a summary, sets token estimate and returns a status string.
            self.compact_summaries.append(summary)
            started.set()
            await finish.wait()
            self.messages = (
                UserMessage(content="Previous conversation summary:\nGenerated summary"),
            )
            self.context_token_estimate = 42
            return "Compacted 2 context entries."

    session = SlowCompactSession(messages=[UserMessage(content="Earlier")])
    app = TauTuiApp(session)
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/compact Summary of earlier work."
        await pilot.press("enter")
        await asyncio.wait_for(started.wait(), timeout=1)

        await pilot.press("escape")
        await pilot.pause()

        assert app._compaction_worker is None
        assert [(item.role, item.text) for item in app.state.items] == [("user", "Earlier")]
        assert notifications == ["Cancelled compaction."]

        prompt.value = "/new"
        await pilot.press("enter")
        await pilot.pause()

        assert session.new_session_count == 1
        assert session.messages == ()
        assert not any(item.role == "compaction_summary" for item in app.state.items)


@pytest.mark.anyio
async def test_tui_app_shows_working_state_during_manual_compaction() -> None:
    # [FUNCTION DOC] Async: verifies manual compaction shows a working state (running footer mode, animated
    # indicator, spinner terminal title) without marking the agent as running, and returns to
    # idle after completion.
    started = asyncio.Event()
    finish = asyncio.Event()

    class SlowCompactSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession whose compaction blocks until released.
        async def compact(self, summary: str) -> str:
            # [FUNCTION DOC] HELPER: async; records summary, signals start, waits for finish, returns status.
            self.compact_summaries.append(summary)
            started.set()
            await finish.wait()
            return "Compacted 2 context entries."

    session = SlowCompactSession(messages=[UserMessage(content="Earlier")])
    session._session_title = "build notes"
    app = TauTuiApp(session)
    titles: list[str] = []
    app._terminal_title = TerminalTitleController(enabled=True, writer=titles.append)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        indicator = app.query_one("#prompt-prefix", Static)
        prompt.value = "/compact Summary of earlier work."
        await pilot.press("enter")
        await asyncio.wait_for(started.wait(), timeout=1)
        await pilot.pause()

        assert app._is_working() is True
        assert app.state.running is False
        assert app._is_agent_or_queue_active() is False
        assert app._is_compaction_active() is True
        assert prompt._footer_mode == "running"
        assert indicator.render().plain != "τ"
        assert titles[-1] == "\x1b]0;⠋ τ | build notes\x07"

        app._tick_activity()
        assert titles[-1] == "\x1b]0;⠙ τ | build notes\x07"

        finish.set()
        await pilot.pause()
        await pilot.pause()

        assert app._is_working() is False
        assert app._is_compaction_active() is False
        assert prompt._footer_mode == "normal"
        assert indicator.render().plain == "τ"
        assert titles[-1] == "\x1b]0;τ | build notes\x07"


@pytest.mark.anyio
async def test_tui_app_clears_working_state_when_manual_compaction_fails() -> None:
    # [FUNCTION DOC] Async: verifies a failing compaction reports "Error: boom" and clears the working state.
    class FailingCompactSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession whose compaction always raises RuntimeError("boom").
        async def compact(self, summary: str) -> str:
            # [FUNCTION DOC] HELPER: async; always raises RuntimeError("boom").
            del summary
            raise RuntimeError("boom")

    app = TauTuiApp(FailingCompactSession(messages=[UserMessage(content="Earlier")]))
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/compact Summary"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert notifications == ["Error: boom"]
        assert app._is_working() is False
        assert app._is_compaction_active() is False


@pytest.mark.anyio
async def test_tui_app_clears_working_state_when_compaction_is_cancelled() -> None:
    # [FUNCTION DOC] Async: verifies cancelling a compaction with escape clears the working state.
    started = asyncio.Event()
    finish = asyncio.Event()

    class SlowCompactSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession whose compaction blocks until released.
        async def compact(self, summary: str) -> str:
            # [FUNCTION DOC] HELPER: async; signals start, waits for finish, returns status.
            del summary
            started.set()
            await finish.wait()
            return "Compacted 2 context entries."

    app = TauTuiApp(SlowCompactSession(messages=[UserMessage(content="Earlier")]))
    app._notify = lambda message, **kwargs: None  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/compact Summary"
        await pilot.press("enter")
        await asyncio.wait_for(started.wait(), timeout=1)
        await pilot.pause()
        assert app._is_working() is True

        await pilot.press("escape")
        await pilot.pause()

        assert app._is_working() is False
        assert app._is_compaction_active() is False


@pytest.mark.anyio
async def test_tui_app_keeps_working_state_when_recompacting_during_cancel_teardown() -> None:
    # [FUNCTION DOC] Async: verifies that when a cancelled compaction's teardown completes after a second
    # compaction started, the app stays in the working state tied to the second worker until
    # it finishes.
    class GatedCompactSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession with per-call start events, a teardown gate delaying cancellation
        # cleanup, and a shared finish event.
        def __init__(self, messages=()) -> None:
            # [FUNCTION DOC] HELPER: initializes the base fake and the start/teardown/finish events and a call counter.
            super().__init__(messages=messages)
            self.started = (asyncio.Event(), asyncio.Event())
            self.teardown_gate = asyncio.Event()
            self.finish = asyncio.Event()
            self.calls = 0

        async def compact(self, summary: str) -> str:
            # [FUNCTION DOC] HELPER: async; records the summary and signals the start event for this call, waits for
            # finish; if cancelled, waits for the teardown gate before re-raising CancelledError.
            # Returns a status string on normal completion. Raises IndexError beyond two calls.
            index = self.calls
            self.calls += 1
            self.compact_summaries.append(summary)
            self.started[index].set()
            try:
                await self.finish.wait()
            except asyncio.CancelledError:
                # Delay teardown so it lands after the next compaction started.
                await self.teardown_gate.wait()
                raise
            return "Compacted 2 context entries."

    session = GatedCompactSession(messages=[UserMessage(content="Earlier")])
    app = TauTuiApp(session)
    app._notify = lambda message, **kwargs: None  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/compact First summary"
        await pilot.press("enter")
        await asyncio.wait_for(session.started[0].wait(), timeout=1)
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()

        prompt.value = "/compact Second summary"
        await pilot.press("enter")
        await asyncio.wait_for(session.started[1].wait(), timeout=1)
        await pilot.pause()
        second_worker = app._compaction_worker

        session.teardown_gate.set()
        await pilot.pause()
        await pilot.pause()

        assert app._is_working() is True
        assert app._is_compaction_active() is True
        assert app._compaction_worker is second_worker

        session.finish.set()
        await pilot.pause()
        await pilot.pause()

        assert app._is_working() is False
        assert app._is_compaction_active() is False


@pytest.mark.anyio
async def test_tui_app_clears_working_state_when_compaction_setup_fails() -> None:
    # [FUNCTION DOC] Async: verifies an exception raised during compaction setup (refresh fails) is reported
    # and leaves no working/compaction state.
    app = TauTuiApp(FakeSession(messages=[UserMessage(content="Earlier")]))
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    def failing_refresh() -> None:
        # [FUNCTION DOC] HELPER: replacement refresh that raises RuntimeError("no matches").
        raise RuntimeError("no matches")

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test():
        app._refresh = failing_refresh  # type: ignore[method-assign]
        await app._run_compaction("Summary")

        assert notifications == ["Error: no matches"]
        assert app._is_working() is False
        assert app._is_compaction_active() is False


@pytest.mark.anyio
async def test_tui_app_notifies_once_when_manual_compaction_finishes_unfocused() -> None:
    # [FUNCTION DOC] Async: verifies a desktop notification is sent once when manual compaction finishes while
    # unfocused, and not when focused.
    app = TauTuiApp(
        FakeSession(messages=[UserMessage(content="Earlier")]),
        tui_settings=TuiSettings(turn_notification="desktop"),
    )
    writes: list[str] = []
    app._terminal_notification = TerminalNotificationController(
        "desktop",
        enabled=True,
        writer=writes.append,
        environ={"TERM_PROGRAM": "ghostty"},
    )
    app._notify = lambda message, **kwargs: None  # type: ignore[method-assign]

    async with app.run_test():
        await app._run_compaction("focused summary")
        assert writes == []

        app.on_app_blur()
        await app._run_compaction("background summary")
        assert writes == ["\x1b]9;Tau turn finished\x07"]


@pytest.mark.anyio
async def test_tui_app_notifies_once_for_turn_with_automatic_compaction() -> None:
    # [FUNCTION DOC] Async: verifies a turn containing automatic compaction emits exactly one notification.
    class AutoCompactSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession whose prompt yields start, compaction start/end, end and settled events.
        async def prompt(
            self,
            text: str,
            *,
            streaming_behavior: str | None = None,
            source: str = "interactive",
            custom_type: str | None = None,
            details: dict[str, object] | None = None,
        ) -> AsyncIterator[CodingSessionEvent]:
            # [FUNCTION DOC] HELPER: async generator yielding the events above after recording the prompt text.
            del streaming_behavior, source, custom_type, details
            self.prompt_texts.append(text)
            yield AgentStartEvent()
            yield CompactionStartEvent(reason="overflow")
            yield CompactionEndEvent(reason="overflow")
            yield AgentEndEvent()
            yield AgentSettledEvent()

    app = TauTuiApp(AutoCompactSession(), tui_settings=TuiSettings(turn_notification="desktop"))
    writes: list[str] = []
    app._terminal_notification = TerminalNotificationController(
        "desktop",
        enabled=True,
        writer=writes.append,
        environ={"TERM_PROGRAM": "ghostty"},
    )

    async with app.run_test():
        app.on_app_blur()
        await app._run_prompt("work")

        assert writes == ["\x1b]9;Tau turn finished\x07"]


@pytest.mark.anyio
async def test_tui_app_export_command_runs_session_export() -> None:
    # [FUNCTION DOC] Async: verifies /export with arguments calls session export with path and format, adds a
    # status item with the exported path, sends no notification and no prompt.
    session = FakeSession(messages=[UserMessage(content="Earlier")])
    app = TauTuiApp(session)
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/export --format jsonl out.jsonl"
        await pilot.press("enter")

        assert session.export_calls == [(Path("out.jsonl"), "jsonl")]
        assert notifications == []
        assert app.state.items[-1] == ChatItem(
            role="status",
            text="/export\nExported session to /workspace/project/session.html",
        )
        assert session.prompt_texts == []


@pytest.mark.anyio
async def test_tui_app_resume_command_reloads_visible_state() -> None:
    # [FUNCTION DOC] Async: verifies "/resume ID" resumes, reloads visible items and prompt history.
    session = FakeSession(messages=[UserMessage(content="Earlier")])
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/resume session-1"
        await pilot.press("enter")

        assert session.resumed_session_ids == ["session-1"]
        assert [(item.role, item.text) for item in app.state.items] == [
            ("user", "Restored prompt"),
        ]

        await pilot.press("up")
        await pilot.pause()

        assert prompt.value == "Restored prompt"


@pytest.mark.anyio
async def test_tui_app_resume_picker_loads_other_projects_in_background() -> None:
    # [FUNCTION DOC] Async: verifies ctrl+r opens the session picker immediately in a loading state, loads the
    # current project sessions in a background thread first, then other projects, populating
    # the project list. Uses threading events to control the fake manager.
    local = CodingSessionRecord(
        id="local",
        path=Path("/workspace/project/local.jsonl"),
        cwd=Path("/workspace/project"),
        model="local-model",
        title="Local session",
        created_at=1.0,
        updated_at=2.0,
    )
    other = CodingSessionRecord(
        id="other",
        path=Path("/elsewhere/other.jsonl"),
        cwd=Path("/elsewhere"),
        model="other-model",
        title="Other session",
        created_at=1.0,
        updated_at=3.0,
    )
    local_started = threading.Event()
    release_local = threading.Event()
    global_started = threading.Event()
    release_global = threading.Event()

    class DelayedSessionManager:
        # [CLASS DOC] HELPER: fake session manager whose list_sessions blocks on threading events: with a cwd it
        # returns only the local record, otherwise the global list. Fails an assertion on a 2s timeout.
        def list_sessions(self, cwd: Path | None = None) -> list[CodingSessionRecord]:
            # [FUNCTION DOC] HELPER: signals the matching started event, waits up to 2 seconds for release (asserting
            # it arrived), and returns the local list (cwd given) or other+local (global).
            if cwd is not None:
                local_started.set()
                assert release_local.wait(timeout=2)
                return [local]
            global_started.set()
            assert release_global.wait(timeout=2)
            return [other, local]

    session = FakeSession()
    session.session_manager = DelayedSessionManager()
    app = TauTuiApp(session)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("ctrl+r")
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, SessionPickerScreen)
        assert screen.loading_other_projects is True
        assert screen.current_project_loaded is False
        assert screen.visible_records == ()
        assert "Loading sessions" in str(screen.query_one("#session-picker-help", Static).render())
        assert await asyncio.to_thread(local_started.wait, 1)

        release_local.set()
        for _ in range(20):
            await pilot.pause()
            if screen.current_project_loaded:
                break
        assert [record.id for record in screen.visible_records] == ["local"]
        assert "Loading other projects" in str(
            screen.query_one("#session-picker-help", Static).render()
        )
        assert await asyncio.to_thread(global_started.wait, 1)

        release_global.set()
        for _ in range(20):
            await pilot.pause()
            if not screen.loading_other_projects:
                break

        assert screen.loading_other_projects is False
        project_list = screen.query_one("#session-picker-project-list", OptionList)
        assert project_list.option_count == 2
        assert [record.id for record in screen.visible_records] == ["local"]


@pytest.mark.anyio
async def test_tui_app_resume_command_opens_session_picker() -> None:
    # [FUNCTION DOC] Async: verifies bare /resume opens the session picker with the first row highlighted and
    # the transcript unchanged.
    record = CodingSessionRecord(
        id="session-1",
        path=Path("/workspace/project/session-1.jsonl"),
        cwd=Path("/workspace/project"),
        model="fake-model",
        title="Test session",
        created_at=1.0,
        updated_at=2.0,
    )
    session = FakeSession(messages=[UserMessage(content="Earlier")])
    session.session_manager = _FakeSessionManager([record])
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/resume"
        await pilot.press("enter")

        assert isinstance(app.screen, SessionPickerScreen)
        picker_list = app.screen.query_one("#session-picker-list", OptionList)
        assert picker_list.highlighted == 0
        assert [(item.role, item.text) for item in app.state.items] == [("user", "Earlier")]


def test_session_records_lists_local_directory_first_then_others() -> None:
    # [FUNCTION DOC] Verifies the session record list for the picker places current-directory sessions
    # (newest first) ahead of sessions from other directories.
    # Records are supplied newest-first (as the real manager returns them); the
    # picker must keep current-directory sessions ahead of every other directory.
    records = [
        CodingSessionRecord(
            id="other-new",
            path=Path("/elsewhere/new.jsonl"),
            cwd=Path("/elsewhere"),
            model="m",
            title=None,
            created_at=1.0,
            updated_at=40.0,
        ),
        CodingSessionRecord(
            id="local-new",
            path=Path("/workspace/project/new.jsonl"),
            cwd=Path("/workspace/project"),
            model="m",
            title=None,
            created_at=1.0,
            updated_at=30.0,
        ),
        CodingSessionRecord(
            id="local-old",
            path=Path("/workspace/project/old.jsonl"),
            cwd=Path("/workspace/project"),
            model="m",
            title=None,
            created_at=1.0,
            updated_at=20.0,
        ),
        CodingSessionRecord(
            id="other-old",
            path=Path("/elsewhere/old.jsonl"),
            cwd=Path("/elsewhere"),
            model="m",
            title=None,
            created_at=1.0,
            updated_at=10.0,
        ),
    ]
    session = FakeSession()  # cwd == /workspace/project
    session.session_manager = _FakeSessionManager(records)

    assert [record.id for record in _session_records(session)] == [
        "local-new",
        "local-old",
        "other-new",
        "other-old",
    ]


def test_session_picker_label_leaves_project_context_to_project_column() -> None:
    # [FUNCTION DOC] Verifies a session picker label contains the title, omits the project path (shown in the
    # project column), and places the model after the title.
    cwd = Path("/home/user/project")
    record = CodingSessionRecord(
        id="s",
        path=cwd / "s.jsonl",
        cwd=cwd,
        model="gpt-5.4",
        title="My session",
        created_at=1.0,
        updated_at=2.0,
    )
    label = _session_picker_label(record)

    assert "My session" in label
    assert _short_path(cwd) not in label
    assert label.index(record.model) > label.index("My session")


@pytest.mark.anyio
async def test_session_picker_navigates_projects_in_left_column() -> None:
    # [FUNCTION DOC] Async: verifies the session picker's project column lists the current project (marked)
    # and others, has the expected borders/padding, shows sessions for the highlighted
    # project with a matching title, supports left/right column switching, and enter resumes.
    records = [
        CodingSessionRecord(
            id="local-1",
            path=Path("/workspace/project/local.jsonl"),
            cwd=Path("/workspace/project"),
            model="m",
            title="Local session",
            created_at=1.0,
            updated_at=30.0,
        ),
        CodingSessionRecord(
            id="other-1",
            path=Path("/elsewhere/other.jsonl"),
            cwd=Path("/elsewhere"),
            model="m",
            title="Other session",
            created_at=1.0,
            updated_at=20.0,
        ),
    ]
    session = FakeSession()
    session.session_manager = _FakeSessionManager(records)
    app = TauTuiApp(session)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("ctrl+r")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SessionPickerScreen)
        project_list = screen.query_one("#session-picker-project-list", OptionList)
        session_list = screen.query_one("#session-picker-list", OptionList)

        project_labels = [str(option.prompt) for option in project_list.options]
        assert project_labels == ["● project", "  elsewhere"]
        columns = screen.query_one("#session-picker-columns")
        project_column = screen.query_one("#session-picker-project-column")
        session_column = screen.query_one("#session-picker-session-column")
        assert columns.styles.border.top[0] == "tall"
        assert project_column.styles.border.top[0] == ""
        assert project_column.styles.border.right[0] == "tall"
        assert session_column.styles.border.top[0] == ""
        assert [record.id for record in screen.visible_records] == ["local-1"]
        assert str(screen.query_one("#session-picker-session-title", Static).render()) == (
            "Recent sessions — /workspace/project"
        )
        assert session_list.styles.padding.left == 1
        assert session_list.styles.padding.right == 1
        assert screen.active_column == "sessions"

        await pilot.press("left", "down")
        await pilot.pause()
        assert screen.active_column == "projects"
        assert project_list.highlighted == 1
        assert [record.id for record in screen.visible_records] == ["other-1"]
        assert str(screen.query_one("#session-picker-session-title", Static).render()) == (
            "Recent sessions — /elsewhere"
        )
        assert "Other session" in str(session_list.get_option_at_index(0).prompt)

        await pilot.press("right", "enter")
        await pilot.pause()
        assert session.resumed_session_ids == ["other-1"]


@pytest.mark.anyio
async def test_session_picker_rapid_project_refresh_is_stable() -> None:
    # [FUNCTION DOC] Async: verifies repeated session list refreshes keep the first row highlighted and the
    # option count stable, with rows rendered virtually (no child widgets).
    records = [
        CodingSessionRecord(
            id=f"local-{i}",
            path=Path(f"/workspace/project/l{i}.jsonl"),
            cwd=Path("/workspace/project"),
            model="m",
            title=None,
            created_at=1.0,
            updated_at=float(30 - i),
        )
        for i in range(3)
    ]
    session = FakeSession()
    session.session_manager = _FakeSessionManager(records)
    app = TauTuiApp(session)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("ctrl+r")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SessionPickerScreen)
        picker_list = screen.query_one("#session-picker-list", OptionList)
        for _ in range(4):
            screen._refresh_session_list()
            await pilot.pause()
        assert picker_list.highlighted == 0
        assert picker_list.option_count == len(screen.visible_records) == 3
        # OptionList renders virtual lines instead of mounting one widget per record.
        assert list(picker_list.children) == []


@pytest.mark.anyio
async def test_session_picker_first_row_highlighted_after_search_refill() -> None:
    # [FUNCTION DOC] Async: verifies after filtering to no matches and back, the first row is highlighted again.
    # After filtering to no matches and back, the first row must be current again
    # and actually highlighted (not a stale highlight left on a removed item).
    records = [
        CodingSessionRecord(
            id=f"s-{i}",
            path=Path(f"/workspace/project/s{i}.jsonl"),
            cwd=Path("/workspace/project"),
            model="m",
            title=None,
            created_at=1.0,
            updated_at=float(30 - i),
        )
        for i in range(5)
    ]
    session = FakeSession()  # cwd == /workspace/project
    session.session_manager = _FakeSessionManager(records)
    app = TauTuiApp(session)

    async with app.run_test(size=(60, 24)) as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/resume"
        await pilot.press("enter")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SessionPickerScreen)
        picker_list = screen.query_one("#session-picker-list", OptionList)
        # Move off the first row, then filter to no matches and back.
        await pilot.press("down")
        assert picker_list.highlighted == 1
        screen.search_value = "zzz-no-match"
        screen._refresh_session_list()
        await pilot.pause()
        assert picker_list.highlighted is None
        screen.search_value = ""
        screen._refresh_session_list()
        await pilot.pause()
        # After the refill the first row is current again and highlighted.
        assert picker_list.highlighted == 0


@pytest.mark.anyio
async def test_session_picker_column_navigation_at_boundaries_does_not_raise() -> None:
    # [FUNCTION DOC] Async: verifies cursor movement past list boundaries in the project column does not
    # raise, and focusing sessions with no highlight works.
    records = [
        CodingSessionRecord(
            id=f"session-{i}",
            path=Path(f"/project-{i}/session.jsonl"),
            cwd=Path(f"/project-{i}"),
            model="m",
            title=None,
            created_at=1.0,
            updated_at=float(30 - i),
        )
        for i in range(3)
    ]
    session = FakeSession()
    session.session_manager = _FakeSessionManager(records)
    app = TauTuiApp(session)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("ctrl+r", "left")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SessionPickerScreen)
        project_list = screen.query_one("#session-picker-project-list", OptionList)
        for _ in range(16):
            screen.action_cursor_down()
        for _ in range(16):
            screen.action_cursor_up()
        assert project_list.highlighted == 0
        screen.action_focus_sessions()
        assert screen.query_one("#session-picker-list", OptionList).highlighted is None


@pytest.mark.anyio
async def test_session_picker_project_click_opens_its_sessions() -> None:
    # [FUNCTION DOC] Async: verifies clicking a project switches to the sessions column for that project
    # without resuming anything.
    other = CodingSessionRecord(
        id="other-1",
        path=Path("/elsewhere/other.jsonl"),
        cwd=Path("/elsewhere"),
        model="m",
        title="Other session",
        created_at=1.0,
        updated_at=20.0,
    )
    session = FakeSession()
    session.session_manager = _FakeSessionManager([other])
    app = TauTuiApp(session)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("ctrl+r")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SessionPickerScreen)
        project_list = screen.query_one("#session-picker-project-list", OptionList)
        await pilot.click(project_list, offset=Offset(2, 1))
        await pilot.pause()

        assert isinstance(app.screen, SessionPickerScreen)
        assert screen.active_column == "sessions"
        assert [record.id for record in screen.visible_records] == ["other-1"]
        assert session.resumed_session_ids == []


def test_short_path_home_directory_is_tilde() -> None:
    # [FUNCTION DOC] Verifies the short path of the home directory is "~".
    assert _short_path(Path.home()) == "~"


@pytest.mark.anyio
async def test_prompt_arrow_keys_move_between_lines_without_completions() -> None:
    # [FUNCTION DOC] Async: verifies up/down move between prompt lines when no completions are shown.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "first\nsecond"
        prompt.move_cursor((1, 3))

        await pilot.press("up")
        assert prompt.cursor_location == (0, 3)

        await pilot.press("down")
        assert prompt.cursor_location == (1, 3)


@pytest.mark.anyio
async def test_tui_app_submits_multiline_prompt_with_enter() -> None:
    # [FUNCTION DOC] Async: verifies shift+enter inserts a newline and enter submits the full multi-line prompt.
    session = FakeSession(
        events=[
            AgentStartEvent(),
            MessageEndEvent(message=UserMessage(content="first\nsecond")),
            AgentEndEvent(),
        ]
    )
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "first"
        prompt.cursor_position = len(prompt.value)
        await pilot.press("shift+enter")
        prompt.value += "second"
        await pilot.press("enter")
        await pilot.pause()

    assert session.prompt_texts == ["first\nsecond"]
    assert prompt.value == ""


@pytest.mark.anyio
@pytest.mark.parametrize(("key", "hint"), [("f7", "F7"), ("ctrl+j", "Ctrl+J")])
async def test_tui_app_uses_configured_newline_keybinding(key: str, hint: str) -> None:
    # [FUNCTION DOC] Parametrized async: verifies a configured newline key inserts a newline (with matching
    # placeholder hint and binding) instead of submitting, and enter still submits.
    session = FakeSession(events=[AgentStartEvent(), AgentEndEvent()])
    app = TauTuiApp(
        session,
        tui_settings=TuiSettings(keybindings=TuiKeybindings(insert_newline=key)),
    )

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "first"
        prompt.cursor_position = len(prompt.value)

        await pilot.press(key)

        assert prompt.value == "first\n"
        assert session.prompt_texts == []
        assert f"{hint} inserts a newline" in prompt.placeholder
        assert _visible_footer_bindings(app)["Newline"] == key

        prompt.value += "second"
        await pilot.press("enter")
        await pilot.pause()

        assert session.prompt_texts == ["first\nsecond"]
        assert prompt.value == ""


@pytest.mark.anyio
async def test_tui_extension_turn_delivers_source_extension() -> None:
    # [FUNCTION DOC] Async: verifies extension-delivered idle turns are prompted with source "extension".
    # An extension-initiated idle turn threads source="extension" through the
    # serialized prompt path; ordinary user submits stay source="interactive".
    class IdleSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession variant reporting that no agent run is active.
        @property
        def is_running(self) -> bool:
            # [FUNCTION DOC] HELPER: property that always returns False (not running).
            return False

    session = IdleSession(events=[AgentStartEvent(), AgentEndEvent()])
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        await app._deliver_extension_message("from extension")
        await pilot.pause()
        await pilot.pause()

    assert session.prompt_texts == ["from extension"]
    assert session.prompt_sources == ["extension"]


@pytest.mark.anyio
async def test_tui_idle_extension_custom_message_renders_card_not_raw_content() -> None:
    # [FUNCTION DOC] Async: verifies an idle-delivered custom extension message renders via the custom
    # renderer card, not raw content, and is not duplicated as a user item.
    # A custom message delivered while idle (e.g. a background-subagent
    # completion) must render through the registered custom renderer, never as
    # its raw content. The optimistic prompt path skips custom messages (it
    # cannot carry their metadata); they render once, from the confirmed user
    # event, which does.
    raw = "<task-notification>agent-1 completed</task-notification>"

    class IdleSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession variant reporting that no agent run is active.
        @property
        def is_running(self) -> bool:
            # [FUNCTION DOC] HELPER: property that always returns False.
            return False

    details = {"description": "Summarize codebase"}
    session = IdleSession(
        events=[
            AgentStartEvent(),
            MessageEndEvent(
                message=CustomMessage(
                    content=raw,
                    custom_type="subagent-notification",
                    details=details,
                )
            ),
            AgentEndEvent(),
        ]
    )
    session.extension_runtime = _CustomMessageRuntime()
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await app._deliver_extension_message(raw, "subagent-notification", details)
        await pilot.pause()
        await pilot.pause()

        custom_items = [item for item in app.state.items if item.role == "custom"]
        user_items = [item for item in app.state.items if item.role == "user"]
        widget = next(w for w in app.query(TranscriptMessageWidget) if w.item.role == "custom")
        selection = widget.selection_text

    assert [item.custom_type for item in custom_items] == ["subagent-notification"]
    assert user_items == []
    assert "✓ Summarize codebase completed" in selection
    assert "<task-notification>" not in selection


@pytest.mark.anyio
async def test_tui_mid_run_custom_follow_up_renders_card_not_raw_content() -> None:
    # [FUNCTION DOC] Async: verifies a custom message drained mid-run as a MessageEndEvent keeps its
    # custom_type metadata and renders as a card next to the normal user message.
    # A custom message queued into an active run (queue_follow_up_message)
    # drains back as a user MessageEndEvent carrying custom_type; the
    # incremental streaming path must preserve the metadata so the message
    # renders as a card, not raw content.
    raw = "<task-notification>agent-1 completed</task-notification>"
    session = FakeSession(
        events=[
            AgentStartEvent(),
            MessageEndEvent(message=UserMessage(content="New prompt")),
            MessageEndEvent(
                message=CustomMessage(
                    content=raw,
                    custom_type="subagent-notification",
                    details={"description": "Summarize codebase"},
                )
            ),
            AgentEndEvent(),
        ]
    )
    session.extension_runtime = _CustomMessageRuntime()
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await app._submit_prompt("New prompt")
        await pilot.pause()
        await pilot.pause()

        custom_items = [item for item in app.state.items if item.role == "custom"]
        user_texts = [item.text for item in app.state.items if item.role == "user"]
        widget = next(w for w in app.query(TranscriptMessageWidget) if w.item.role == "custom")
        selection = widget.selection_text

    assert [item.custom_type for item in custom_items] == ["subagent-notification"]
    assert user_texts == ["New prompt"]
    assert "✓ Summarize codebase completed" in selection
    assert "<task-notification>" not in selection


@pytest.mark.anyio
async def test_structured_assistant_redraw_preserves_extension_custom_card() -> None:
    # [FUNCTION DOC] Async: verifies the structured redraw for an assistant message with thinking and text
    # does not lose the earlier extension custom card rendering.
    raw = "<task-notification>agent-1 completed</task-notification>"
    partial = AssistantMessage()
    final = AssistantMessage(content=[ThinkingContent(thinking="plan"), TextContent(text="done")])
    session = FakeSession(
        events=[
            AgentStartEvent(),
            MessageEndEvent(
                message=CustomMessage(
                    content=raw,
                    custom_type="subagent-notification",
                    details={"description": "Summarize codebase"},
                )
            ),
            MessageStartEvent(message=partial),
            MessageUpdateEvent(
                message=partial,
                assistant_message_event=ThinkingDeltaEvent(
                    content_index=0,
                    delta="plan",
                    partial=partial,
                ),
            ),
            MessageUpdateEvent(
                message=partial,
                assistant_message_event=TextDeltaEvent(
                    content_index=1,
                    delta="done",
                    partial=partial,
                ),
            ),
            MessageEndEvent(message=final),
            AgentEndEvent(),
        ]
    )
    session.extension_runtime = _CustomMessageRuntime()
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await app._run_prompt("run")
        await pilot.pause()

        custom_widget = next(
            widget for widget in app.query(TranscriptMessageWidget) if widget.item.role == "custom"
        )
        assert "✓ Summarize codebase completed" in custom_widget.selection_text
        assert "<task-notification>" not in custom_widget.selection_text


@pytest.mark.anyio
async def test_structured_assistant_finalization_preserves_existing_widget_identity() -> None:
    # [FUNCTION DOC] Async: verifies finalizing a structured assistant message keeps existing earlier widgets
    # mounted (same identity) with their custom card text.
    raw = "<task-notification>agent-1 completed</task-notification>"
    partial = AssistantMessage()
    session = FakeSession(
        messages=[
            CustomMessage(
                content=raw,
                custom_type="subagent-notification",
                details={"description": "Summarize codebase"},
            )
        ],
        events=[
            AgentStartEvent(),
            MessageStartEvent(message=partial),
            MessageUpdateEvent(
                message=partial,
                assistant_message_event=ThinkingDeltaEvent(
                    content_index=0,
                    delta="plan",
                    partial=partial,
                ),
            ),
            MessageUpdateEvent(
                message=partial,
                assistant_message_event=TextDeltaEvent(
                    content_index=1,
                    delta="done",
                    partial=partial,
                ),
            ),
            MessageEndEvent(
                message=AssistantMessage(
                    content=[ThinkingContent(thinking="plan"), TextContent(text="done")]
                )
            ),
            AgentEndEvent(),
        ],
    )
    session.extension_runtime = _CustomMessageRuntime()
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        custom_widget = next(
            widget for widget in app.query(TranscriptMessageWidget) if widget.item.role == "custom"
        )
        await app._run_prompt("run")
        await pilot.pause()

        assert custom_widget.parent is app.query_one("#transcript", TranscriptView)
        assert "✓ Summarize codebase completed" in custom_widget.selection_text


@pytest.mark.anyio
async def test_structured_assistant_ignores_empty_final_content_blocks() -> None:
    # [FUNCTION DOC] Async: verifies empty final content blocks (empty thinking) are not rendered as lines.
    partial = AssistantMessage()
    session = FakeSession(
        messages=[UserMessage(content="earlier")],
        events=[
            AgentStartEvent(),
            MessageStartEvent(message=partial),
            MessageUpdateEvent(
                message=partial,
                assistant_message_event=TextDeltaEvent(
                    content_index=1,
                    delta="done",
                    partial=partial,
                ),
            ),
            MessageEndEvent(
                message=AssistantMessage(
                    content=[ThinkingContent(thinking=""), TextContent(text="done")]
                )
            ),
            AgentEndEvent(),
        ],
    )
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await app._run_prompt("run")
        await pilot.pause()

        transcript = app.query_one("#transcript", TranscriptView)
        assert [line.text for line in transcript.lines] == ["earlier", "done"]


@pytest.mark.anyio
async def test_tui_app_prompts_picker_filters_and_inserts_without_submitting() -> None:
    # [FUNCTION DOC] Async: verifies /prompts opens a template picker with focused search, arrow navigation,
    # empty-match message, filtering, and enter inserting "/NAME" without submitting.
    session = FakeSession()
    session.prompt_templates = (
        PromptTemplate(
            name="review",
            path=Path("review.md"),
            content="Review this.",
            description="Inspect changes",
        ),
        PromptTemplate(
            name="test",
            path=Path("test.md"),
            content="Test this.",
            description="Run checks",
        ),
    )
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/prompts"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, PromptTemplatePickerScreen)
        search = app.screen.query_one("#prompt-template-picker-search", Input)
        picker_list = app.screen.query_one("#prompt-template-picker-list", ListView)
        assert search.has_focus
        assert picker_list.index == 0
        await pilot.press("down")
        assert picker_list.index == 1
        await pilot.press("up")
        assert picker_list.index == 0
        await pilot.press("z")
        assert app.screen.visible_templates == ()
        assert "No matching prompt templates" in str(
            app.screen.query_one("#prompt-template-picker-help", Static).content
        )
        search.value = "tes"
        await pilot.pause()
        assert [template.name for template in app.screen.visible_templates] == ["test"]

        await pilot.press("enter")
        await pilot.pause()

        assert prompt.value == "/test"
        assert prompt.has_focus
        assert app.state.items == []


@pytest.mark.anyio
async def test_tui_app_prompts_picker_edits_template_and_reloads(tmp_path: Path) -> None:
    # [FUNCTION DOC] Async: verifies ctrl+e in the prompt picker opens an editor with working cursor keys,
    # ctrl+s saves the template to disk, reloads session resources and returns to the picker.
    # Inputs: tmp_path.
    template_path = tmp_path / "review.md"
    template_path.write_text("Original prompt.\n", encoding="utf-8")
    session = FakeSession()
    session.prompt_templates = (
        PromptTemplate(
            name="review",
            path=template_path,
            content="Original prompt.",
            description="Inspect changes",
        ),
    )
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/prompts"
        await pilot.press("enter")
        await pilot.pause()

        picker = app.screen
        assert isinstance(picker, PromptTemplatePickerScreen)
        assert "Ctrl+E edits" in str(
            picker.query_one("#prompt-template-picker-help", Static).content
        )
        await pilot.press("ctrl+e")
        await pilot.pause()

        editor = app.screen
        assert isinstance(editor, PromptTemplateEditorScreen)
        editor_input = editor.query_one("#prompt-template-editor-input", TextArea)
        assert editor_input.text == "Original prompt.\n"
        editor_input.text = "first\nsecond"
        editor_input.move_cursor((0, 0))
        await pilot.press("right")
        assert editor_input.cursor_location == (0, 1)
        await pilot.press("down")
        assert editor_input.cursor_location == (1, 1)
        await pilot.press("left")
        assert editor_input.cursor_location == (1, 0)
        await pilot.press("up")
        assert editor_input.cursor_location == (0, 0)

        editor_input.text = "Updated prompt.\n"
        await pilot.press("ctrl+s")
        await pilot.pause()

        assert template_path.read_text(encoding="utf-8") == "Updated prompt.\n"
        assert session.reload_count == 1
        assert isinstance(app.screen, PromptTemplatePickerScreen)
        assert prompt.value == ""


@pytest.mark.anyio
async def test_tui_app_prompts_picker_cancel_and_empty_state() -> None:
    # [FUNCTION DOC] Async: verifies the empty prompt picker message and that escape clears the prompt.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/prompts"
        await pilot.press("enter")
        await pilot.pause()

        picker = app.screen
        assert isinstance(picker, PromptTemplatePickerScreen)
        assert "No prompt templates loaded" in str(
            picker.query_one("#prompt-template-picker-help", Static).content
        )
        await pilot.press("escape")
        await pilot.pause()

        assert prompt.value == ""
        assert prompt.has_focus
        assert app.state.items == []


@pytest.mark.anyio
async def test_tui_app_completes_custom_prompt_slash_command() -> None:
    # [FUNCTION DOC] Async: verifies Tab completes a custom prompt template slash command.
    session = FakeSession()
    session.prompt_templates = (
        PromptTemplate(
            name="example",
            path=Path("example.md"),
            content="Example prompt.",
            description="Run the example prompt.",
        ),
    )
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/exa"
        app._completion_state = app._build_completion_state(prompt.value)
        app._refresh_completions()

        await pilot.press("tab")

        assert prompt.value == "/example"


@pytest.mark.anyio
async def test_tui_app_completes_registered_slash_command() -> None:
    # [FUNCTION DOC] Async: verifies Tab completes a built-in slash command.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/se"
        app._completion_state = app._build_completion_state(prompt.value)
        app._refresh_completions()

        await pilot.press("tab")

        assert prompt.value == "/session"


@pytest.mark.anyio
async def test_tui_app_enter_completes_slash_then_second_enter_submits() -> None:
    # [FUNCTION DOC] Async: verifies first enter accepts the slash completion and a second enter submits it,
    # showing the command output screen.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/se"
        app._completion_state = app._build_completion_state(prompt.value)
        app._refresh_completions()

        await pilot.press("enter")
        await pilot.pause()

        assert prompt.value == "/session"
        assert app.session.prompt_texts == []
        assert not isinstance(app.screen, CommandOutputScreen)

        await pilot.press("enter")
        await pilot.pause()

        assert prompt.value == ""
        assert isinstance(app.screen, CommandOutputScreen)
        assert app.screen.message == "Session info"


@pytest.mark.anyio
async def test_tui_app_enter_submits_directly_typed_exact_slash_command() -> None:
    # [FUNCTION DOC] Async: verifies enter submits an exactly typed slash command whose completion equals the
    # text, showing the command output screen.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/session"
        app._completion_state = app._build_completion_state(prompt.value)
        app._refresh_completions()

        selected = app._completion_state.selected
        assert selected is not None
        assert selected.replacement == prompt.value

        await pilot.press("enter")
        await pilot.pause()

        assert prompt.value == ""
        assert isinstance(app.screen, CommandOutputScreen)
        assert app.screen.message == "Session info"


@pytest.mark.anyio
async def test_tui_app_tab_completion_then_enter_submits() -> None:
    # [FUNCTION DOC] Async: verifies Tab completes a slash command and the following enter submits it.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/se"
        app._completion_state = app._build_completion_state(prompt.value)
        app._refresh_completions()

        await pilot.press("tab")
        assert prompt.value == "/session"
        assert app.session.prompt_texts == []

        await pilot.press("enter")
        await pilot.pause()

        assert prompt.value == ""
        assert isinstance(app.screen, CommandOutputScreen)
        assert app.screen.message == "Session info"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("raw_text", "expected_kind"),
    (
        ("/skill:review", CompletionKind.SKILL),
        ("/example", CompletionKind.PROMPT_TEMPLATE),
        ("/model fake-model", CompletionKind.ARGUMENT),
    ),
)
async def test_tui_app_enter_submits_exact_non_file_completion_kinds(
    raw_text: str,
    expected_kind: CompletionKind,
) -> None:
    # [FUNCTION DOC] Parametrized async: verifies enter submits text whose selected completion (skill, prompt
    # template, argument) already equals the text, sending it to the session unchanged.
    session = FakeSession()
    session.prompt_templates = (
        PromptTemplate(
            name="example",
            path=Path("example.md"),
            content="Example prompt.",
        ),
    )
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = raw_text
        app._completion_state = app._build_completion_state(prompt.value)
        app._refresh_completions()

        selected = app._completion_state.selected
        assert selected is not None
        assert selected.kind is expected_kind
        assert app._apply_selected_completion(raw_text) == raw_text

        await pilot.press("enter")
        await pilot.pause()

        assert prompt.value == ""
        assert session.prompt_texts == [raw_text]


@pytest.mark.anyio
async def test_tui_app_enter_accepts_arrow_selected_non_file_completion() -> None:
    # [FUNCTION DOC] Async: verifies enter accepts an arrow-selected slash completion into the prompt instead
    # of submitting.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/s"
        app._completion_state = app._build_completion_state(prompt.value)
        app._refresh_completions()
        await pilot.press("down")
        selected = app._completion_state.selected
        assert selected is not None

        await pilot.press("enter")
        await pilot.pause()

        assert prompt.value == selected.replacement
        assert app.session.prompt_texts == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    "raw_text",
    (
        "inspect @main",
        "/skill:review inspect @main",
        "/example inspect @main",
    ),
)
async def test_tui_app_enter_submits_raw_file_reference_text(
    tmp_path: Path,
    raw_text: str,
) -> None:
    # [FUNCTION DOC] Parametrized async: verifies enter submits raw text even when an @file completion is
    # selected (Tab is required to accept file references). Inputs: tmp_path.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")

    session = FakeSession()
    session.cwd = tmp_path
    session.prompt_templates = (
        PromptTemplate(
            name="example",
            path=tmp_path / "example.md",
            content="Example prompt.",
        ),
    )
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = raw_text
        app._completion_state = app._build_completion_state(prompt.value)
        app._refresh_completions()

        selected = app._completion_state.selected
        assert selected is not None
        assert selected.kind is CompletionKind.FILE_REFERENCE
        await pilot.press("enter")
        await pilot.pause()

        assert prompt.value == ""
        assert session.prompt_texts == [raw_text]


@pytest.mark.anyio
async def test_tui_app_accepts_file_reference_completion(tmp_path: Path) -> None:
    # [FUNCTION DOC] Async: verifies Tab accepts an @file completion found by fuzzy path match. Inputs: tmp_path.
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")

    session = FakeSession()
    session.cwd = tmp_path
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "inspect @main"
        app._completion_state = app._build_completion_state(prompt.value)
        app._refresh_completions()

        assert [item.display for item in app._completion_state.items] == ["@src/main.py"]
        await pilot.press("tab")

        assert prompt.value == "inspect @src/main.py"


@pytest.mark.anyio
async def test_tui_app_accepts_shell_path_completion(tmp_path: Path) -> None:
    # [FUNCTION DOC] Async: verifies shell-mode prompts complete plain file paths. Inputs: tmp_path.
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")

    session = FakeSession()
    session.cwd = tmp_path
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "!cat READ"
        app._completion_state = app._build_completion_state(prompt.value)
        app._refresh_completions()

        assert [item.display for item in app._completion_state.items] == ["README.md"]
        await pilot.press("tab")

        assert prompt.value == "!cat README.md"


@pytest.mark.anyio
async def test_tui_app_completes_skill_name() -> None:
    # [FUNCTION DOC] Async: verifies Tab completes a "/skill:" name.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/skill:r"
        app._completion_state = app._build_completion_state(prompt.value)
        app._refresh_completions()

        await pilot.press("tab")

        assert prompt.value == "/skill:review"


@pytest.mark.anyio
async def test_tui_app_completes_model_argument() -> None:
    # [FUNCTION DOC] Async: verifies Tab completes a /model argument.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/model fak"
        app._completion_state = app._build_completion_state(prompt.value)
        app._refresh_completions()

        await pilot.press("tab")

        assert prompt.value == "/model fake-model"


@pytest.mark.anyio
async def test_tui_app_completes_resume_session_argument() -> None:
    # [FUNCTION DOC] Async: verifies /resume argument completion shows title, model and cwd as description and
    # completes the session id.
    session = FakeSession()
    session.session_manager = _FakeSessionManager(
        [
            CodingSessionRecord(
                id="session-1",
                path=Path("/tmp/session-1.jsonl"),
                cwd=Path("/workspace/project"),
                model="fake-model",
                title="Session",
                created_at=1.0,
                updated_at=2.0,
            )
        ]
    )
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/resume sess"
        app._completion_state = app._build_completion_state(prompt.value)
        app._refresh_completions()

        assert app._completion_state.selected is not None
        assert app._completion_state.selected.description == (
            "Session - fake-model - /workspace/project"
        )

        await pilot.press("tab")

        assert prompt.value == "/resume session-1"


@pytest.mark.anyio
async def test_tui_app_session_picker_resumes_selected_session() -> None:
    # [FUNCTION DOC] Async: verifies ctrl+r then enter resumes the highlighted session and reloads the transcript.
    session = FakeSession(messages=[UserMessage(content="Earlier")])
    session.session_manager = _FakeSessionManager(
        [
            CodingSessionRecord(
                id="session-1",
                path=Path("/tmp/session-1.jsonl"),
                cwd=Path("/workspace/project"),
                model="fake-model",
                title="Session",
                created_at=1.0,
                updated_at=2.0,
            )
        ]
    )
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        await pilot.press("ctrl+r")
        assert isinstance(app.screen, SessionPickerScreen)

        await pilot.press("enter")
        await pilot.pause()

        assert session.resumed_session_ids == ["session-1"]
        assert [(item.role, item.text) for item in app.state.items] == [
            ("user", "Restored prompt"),
        ]


@pytest.mark.anyio
async def test_tui_app_session_picker_archives_selected_session() -> None:
    # [FUNCTION DOC] Async: verifies ctrl+enter archives the selected session, keeps the search text and
    # empties the filtered list.
    session = FakeSession()
    record = CodingSessionRecord(
        id="session-1",
        path=Path("/tmp/session-1.jsonl"),
        cwd=Path("/workspace/project"),
        model="fake-model",
        title="Session",
        created_at=1.0,
        updated_at=2.0,
    )
    manager = _FakeSessionManager([record])
    session.session_manager = manager
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        await pilot.press("ctrl+r")
        assert isinstance(app.screen, SessionPickerScreen)
        search = app.screen.query_one("#session-picker-search", Input)
        assert search.has_focus
        search.value = "session"
        await pilot.pause()
        await pilot.press("ctrl+enter")
        await pilot.pause()

        assert manager.archived_session_ids == ["session-1"]
        assert search.value == "session"
        assert app.screen.query_one("#session-picker-list", OptionList).option_count == 0


@pytest.mark.anyio
async def test_tui_app_session_picker_archives_selected_project() -> None:
    # [FUNCTION DOC] Async: verifies ctrl+enter in the project column archives the highlighted project's sessions.
    session = FakeSession()
    first = CodingSessionRecord(
        id="session-1",
        path=Path("/tmp/session-1.jsonl"),
        cwd=Path("/workspace/first"),
        model="fake-model",
        title="First",
        created_at=1.0,
        updated_at=3.0,
    )
    second = CodingSessionRecord(
        id="session-2",
        path=Path("/tmp/session-2.jsonl"),
        cwd=Path("/workspace/second"),
        model="other-model",
        title="Second",
        created_at=1.0,
        updated_at=2.0,
    )
    manager = _FakeSessionManager([first, second])
    session.session_manager = manager
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        await pilot.press("ctrl+r")
        assert isinstance(app.screen, SessionPickerScreen)
        await pilot.press("left", "down", "ctrl+enter")
        await pilot.pause()

        assert manager.archived_project_cwds == [first.cwd]
        project_options = app.screen.query_one("#session-picker-project-list", OptionList)
        assert project_options.option_count == 2


@pytest.mark.anyio
async def test_tui_app_session_picker_archived_tab_restores_session_and_project() -> None:
    # [FUNCTION DOC] Async: verifies F2 shows archived sessions, enter restores an archived session, and enter
    # in the project column restores an archived project.
    session = FakeSession()
    active = CodingSessionRecord(
        id="active",
        path=Path("/tmp/active.jsonl"),
        cwd=Path("/workspace/project"),
        model="fake-model",
        title="Active",
        created_at=1.0,
        updated_at=3.0,
    )
    archived_session = CodingSessionRecord(
        id="archived-session",
        path=Path("/tmp/archived-session.jsonl"),
        cwd=Path("/workspace/project"),
        model="fake-model",
        title="Archived session",
        created_at=1.0,
        updated_at=2.0,
    )
    archived_project = CodingSessionRecord(
        id="archived-project",
        path=Path("/tmp/archived-project.jsonl"),
        cwd=Path("/workspace/archived"),
        model="other-model",
        title="Archived project",
        created_at=1.0,
        updated_at=1.0,
    )
    manager = _FakeSessionManager([active], [archived_session, archived_project])
    session.session_manager = manager
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        await pilot.press("ctrl+r")
        await pilot.pause()
        assert isinstance(app.screen, SessionPickerScreen)
        await pilot.press("f2")
        assert app.screen.showing_archived is True
        assert app.screen.query_one("#session-picker-list", OptionList).option_count == 1

        await pilot.press("enter")
        await pilot.pause()
        assert manager.restored_session_ids == ["archived-session"]

        await pilot.press("left", "down", "enter")
        await pilot.pause()
        assert manager.restored_project_cwds == [archived_project.cwd]
        assert app.screen.query_one("#session-picker-list", OptionList).option_count == 0


@pytest.mark.anyio
async def test_tui_app_session_picker_shows_human_readable_session_metadata() -> None:
    # [FUNCTION DOC] Async: verifies session labels show a short date, the title (omitting "Untitled session")
    # and the model, without the session id.
    updated_at = datetime(2026, 6, 19, 14, 30).timestamp()
    session = FakeSession()
    session.session_manager = _FakeSessionManager(
        [
            CodingSessionRecord(
                id="session-1",
                path=Path("/tmp/session-1.jsonl"),
                cwd=Path("/workspace/project"),
                model="fake-model",
                title="Untitled session",
                created_at=1.0,
                updated_at=updated_at,
            ),
            CodingSessionRecord(
                id="session-2",
                path=Path("/tmp/session-2.jsonl"),
                cwd=Path("/workspace/project"),
                model="other-model",
                title="Named work",
                created_at=1.0,
                updated_at=updated_at,
            ),
        ]
    )
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        await pilot.press("ctrl+r")
        assert isinstance(app.screen, SessionPickerScreen)
        session_list = app.screen.query_one("#session-picker-list", OptionList)
        labels = [option.prompt for option in session_list.options]

    assert labels == [
        "Jun 19  fake-model",
        "Jun 19  Named work  other-model",
    ]
    assert "session-1" not in "\n".join(str(label) for label in labels)
    assert "Untitled session" not in "\n".join(str(label) for label in labels)


@pytest.mark.anyio
async def test_tui_app_session_picker_arrow_keys_select_session() -> None:
    # [FUNCTION DOC] Async: verifies down then enter resumes the second session.
    session = FakeSession(messages=[UserMessage(content="Earlier")])
    session.session_manager = _FakeSessionManager(
        [
            CodingSessionRecord(
                id="session-1",
                path=Path("/tmp/session-1.jsonl"),
                cwd=Path("/workspace/project"),
                model="fake-model",
                title=None,
                created_at=1.0,
                updated_at=3.0,
            ),
            CodingSessionRecord(
                id="session-2",
                path=Path("/tmp/session-2.jsonl"),
                cwd=Path("/workspace/project"),
                model="other-model",
                title=None,
                created_at=1.0,
                updated_at=2.0,
            ),
        ]
    )
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        await pilot.press("ctrl+r")
        assert isinstance(app.screen, SessionPickerScreen)
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()

        assert session.resumed_session_ids == ["session-2"]


@pytest.mark.anyio
async def test_tui_app_session_picker_search_filters_sessions() -> None:
    # [FUNCTION DOC] Async: verifies the focused search box filters sessions by title and enter resumes the match.
    updated_at = datetime(2026, 6, 19, 14, 30).timestamp()
    session = FakeSession(messages=[UserMessage(content="Earlier")])
    session.session_manager = _FakeSessionManager(
        [
            CodingSessionRecord(
                id="session-1",
                path=Path("/tmp/session-1.jsonl"),
                cwd=Path("/workspace/project"),
                model="fake-model",
                title="Refactor auth",
                created_at=1.0,
                updated_at=updated_at,
            ),
            CodingSessionRecord(
                id="session-2",
                path=Path("/tmp/session-2.jsonl"),
                cwd=Path("/workspace/project"),
                model="other-model",
                title="Add search bar",
                created_at=1.0,
                updated_at=updated_at,
            ),
        ]
    )
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        await pilot.press("ctrl+r")
        assert isinstance(app.screen, SessionPickerScreen)

        search = app.screen.query_one("#session-picker-search", Input)
        assert search.has_focus

        search.value = "search bar"
        await pilot.pause()

        session_list = app.screen.query_one("#session-picker-list", OptionList)
        labels = [str(option.prompt) for option in session_list.options]
        assert labels == ["Jun 19  Add search bar  other-model"]

        await pilot.press("enter")
        await pilot.pause()

        assert session.resumed_session_ids == ["session-2"]


@pytest.mark.anyio
async def test_tui_app_session_picker_search_does_not_match_workspace_path() -> None:
    # [FUNCTION DOC] Async: verifies session search matches title and model but not the workspace path.
    session = FakeSession()
    session.session_manager = _FakeSessionManager(
        [
            CodingSessionRecord(
                id="session-1",
                path=Path("/tmp/session-1.jsonl"),
                cwd=Path("/workspace/path-query"),
                model="model-query",
                title="Named session",
                created_at=1.0,
                updated_at=2.0,
            ),
        ]
    )
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        await pilot.press("ctrl+r")
        assert isinstance(app.screen, SessionPickerScreen)

        search = app.screen.query_one("#session-picker-search", Input)
        session_list = app.screen.query_one("#session-picker-list", OptionList)

        await pilot.press("left", "down", "right")
        search.value = "path-query"
        await pilot.pause()
        assert session_list.option_count == 0

        for query in ("model-query", "named"):
            search.value = query
            await pilot.pause()
            assert session_list.option_count == 1


@pytest.mark.anyio
async def test_tui_app_session_picker_search_with_no_matches_shows_help_text() -> None:
    # [FUNCTION DOC] Async: verifies a no-match search shows the "No matching sessions" help text.
    session = FakeSession()
    session.session_manager = _FakeSessionManager(
        [
            CodingSessionRecord(
                id="session-1",
                path=Path("/tmp/session-1.jsonl"),
                cwd=Path("/workspace/project"),
                model="fake-model",
                title="Refactor auth",
                created_at=1.0,
                updated_at=2.0,
            ),
        ]
    )
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        await pilot.press("ctrl+r")
        assert isinstance(app.screen, SessionPickerScreen)

        search = app.screen.query_one("#session-picker-search", Input)
        search.value = "nonexistent"
        await pilot.pause()

        session_list = app.screen.query_one("#session-picker-list", OptionList)
        assert session_list.option_count == 0
        help_text = app.screen.query_one("#session-picker-help", Static)
        assert str(help_text.render()) == (
            "No matching sessions - Left selects a project - Escape closes"
        )


@pytest.mark.anyio
async def test_tui_app_blocks_tree_picker_while_agent_is_running() -> None:
    # [FUNCTION DOC] Async: verifies /tree is refused with a notification while the agent is running.
    session = FakeSession()
    app = TauTuiApp(session)
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        app.state.running = True
        prompt = app.query_one("#prompt")
        prompt.value = "/tree"
        await pilot.press("enter")
        await pilot.pause()

        assert session.tree_branch_requests == []
        assert prompt.value == "/tree"
        assert not isinstance(app.screen, TreePickerScreen)
        assert notifications == [
            "Tau is still working. Press Escape to interrupt before using /tree."
        ]


@pytest.mark.anyio
async def test_tui_app_blocks_tree_branch_selection_while_agent_is_running() -> None:
    # [FUNCTION DOC] Async: verifies branch selection inside an open tree picker is refused once the agent runs.
    session = FakeSession()
    app = TauTuiApp(session)
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/tree"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, TreePickerScreen)
        app.state.running = True
        await pilot.press("enter")
        await pilot.pause()

        assert session.tree_branch_requests == []
        assert notifications == [
            "Tau is still working. Press Escape to interrupt before using /tree."
        ]


@pytest.mark.anyio
async def test_tui_app_tree_picker_branches_with_summary() -> None:
    # [FUNCTION DOC] Async: verifies the tree picker starts on the active entry with highlight/accent label
    # styles, arrow navigation updates styles, "s" branches with summary, the transcript is
    # reloaded and prompt history recalls the branched message.
    session = FakeSession()
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/tree"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, TreePickerScreen)
        tree_list = app.screen.query_one("#tree-picker-list", ListView)
        assert tree_list.index == 3
        rendered_labels = [item.query_one(Label).render() for item in tree_list.children]
        labels = [str(label) for label in rendered_labels]
        assert labels == [
            "  user: Root",
            "  tool call: read",
            "  assistant: Left",
            "* assistant: Right",
        ]
        assert str(rendered_labels[0].spans[0].style) == _style_rgb(TAU_DARK_THEME.accent)
        assert str(rendered_labels[3].spans[0].style) == _style_rgb(TAU_DARK_THEME.highlight_text)

        await pilot.press("up")
        await pilot.pause()
        assert tree_list.index == 2
        left_label = tree_list.children[2].query_one(Label).render()
        right_label = tree_list.children[3].query_one(Label).render()
        assert str(left_label.spans[0].style) == _style_rgb(TAU_DARK_THEME.highlight_text)
        assert str(right_label.spans[0].style) == _style_rgb(TAU_DARK_THEME.accent)
        await pilot.press("s")
        await pilot.pause()

        assert session.tree_branch_requests == [("left", True, None)]
        assert [(item.role, item.text) for item in app.state.items] == [
            ("user", "Branched to left"),
        ]

        prompt = app.query_one("#prompt")
        await pilot.press("up")
        await pilot.pause()

        assert prompt.value == "Branched to left"


@pytest.mark.anyio
async def test_tui_app_tree_picker_prefills_selected_user_message() -> None:
    # [FUNCTION DOC] Async: verifies branching to a user entry prefills the prompt with that message (cursor at
    # end) instead of submitting it.
    class PrefillSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession whose branch_to_entry returns a result with an input prefill.
        async def branch_to_entry(
            self,
            entry_id: str,
            *,
            summarize: bool = False,
            custom_instructions: str | None = None,
        ) -> SessionTreeBranchResult:
            # [FUNCTION DOC] HELPER: async; records the request, clears messages and returns a SessionTreeBranchResult
            # with message text and input_prefill "Root".
            self.tree_branch_requests.append((entry_id, summarize, custom_instructions))
            self.messages = ()
            return SessionTreeBranchResult(
                message=f"Branched session before {entry_id}.",
                input_prefill="Root",
            )

    session = PrefillSession()
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/tree"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, TreePickerScreen)
        await pilot.press("up", "up", "up")
        await pilot.press("enter")
        await pilot.pause()

        assert session.tree_branch_requests == [("root", False, None)]
        assert session.prompt_texts == []
        assert [(item.role, item.text) for item in app.state.items] == []
        assert prompt.value == "Root"
        assert prompt.cursor_location == (0, 4)


@pytest.mark.anyio
async def test_tui_app_tree_summary_clears_transcript_while_summarizing() -> None:
    # [FUNCTION DOC] Async: verifies a summarizing branch shows only a "Summarizing branch…" status until the
    # summary completes, then the branched transcript.
    started = asyncio.Event()
    finish = asyncio.Event()

    class SlowSummarySession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession whose branch_to_entry blocks until released.
        async def branch_to_entry(
            self,
            entry_id: str,
            *,
            summarize: bool = False,
            custom_instructions: str | None = None,
        ) -> str:
            # [FUNCTION DOC] HELPER: async; records the request, signals start, waits for finish, then sets messages
            # and returns a status string.
            self.tree_branch_requests.append((entry_id, summarize, custom_instructions))
            started.set()
            await finish.wait()
            self.messages = (UserMessage(content=f"Branched to {entry_id}"),)
            return f"Branched session at {entry_id}."

    session = SlowSummarySession(messages=[UserMessage(content="Old thread")])
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/tree"
        await pilot.press("enter")
        await pilot.pause()

        await pilot.press("up")
        await pilot.press("s")
        await pilot.pause()
        await started.wait()

        assert [(item.role, item.text) for item in app.state.items] == [
            ("status", "Summarizing branch…"),
        ]

        finish.set()
        await pilot.pause()

        assert session.tree_branch_requests == [("left", True, None)]
        assert [(item.role, item.text) for item in app.state.items] == [
            ("user", "Branched to left"),
        ]


@pytest.mark.anyio
async def test_tui_app_tree_picker_toggles_tool_calls() -> None:
    # [FUNCTION DOC] Async: verifies ctrl+t in the tree picker hides tool-call entries (keeping the active
    # entry selected and noting it in help) and shows them again.
    session = FakeSession()
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/tree"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, TreePickerScreen)
        tree_list = app.screen.query_one("#tree-picker-list", ListView)
        assert tree_list.index == 3

        await pilot.press("ctrl+t")
        await pilot.pause()

        labels = [str(item.query_one(Label).render()) for item in tree_list.children]
        assert labels == [
            "  user: Root",
            "  assistant: Left",
            "* assistant: Right",
        ]
        assert tree_list.index == 2
        assert "tool calls hidden" in str(
            app.screen.query_one("#tree-picker-help", Static).render()
        )

        await pilot.press("ctrl+t")
        await pilot.pause()

        labels = [str(item.query_one(Label).render()) for item in tree_list.children]
        assert labels == [
            "  user: Root",
            "  tool call: read",
            "  assistant: Left",
            "* assistant: Right",
        ]
        assert tree_list.index == 3


@pytest.mark.anyio
async def test_tui_tree_labels_filter_timestamps_and_clear() -> None:
    # [FUNCTION DOC] Async: verifies "l" sets a bookmark label via an input dialog, ctrl+l shows label
    # timestamps, ctrl+f filters to labelled entries, and clearing the label removes it.
    session = FakeSession()
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/tree"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("up", "l")
        await pilot.pause()

        field = app.screen.query_one("#extension-input-field", Input)
        field.value = "checkpoint"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, TreePickerScreen)
        tree_list = app.screen.query_one("#tree-picker-list", ListView)
        labels = [str(item.query_one(Label).render()) for item in tree_list.children]
        assert "[checkpoint] assistant: Left" in labels[2]
        assert session.tree_label_requests == [("left", "checkpoint")]

        await pilot.press("ctrl+l")
        await pilot.pause()
        assert "2023-11-14" in str(tree_list.children[2].query_one(Label).render())

        await pilot.press("ctrl+f")
        await pilot.pause()
        assert len(tree_list.children) == 1
        assert "[checkpoint]" in str(tree_list.children[0].query_one(Label).render())

        await pilot.press("l")
        await pilot.pause()
        field = app.screen.query_one("#extension-input-field", Input)
        assert field.value == "checkpoint"
        field.value = ""
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, TreePickerScreen)
        assert session.tree_label_requests == [("left", "checkpoint"), ("left", None)]
        assert len(app.screen.query_one("#tree-picker-list", ListView).children) == 0


@pytest.mark.anyio
def test_completion_selected_render_line_accounts_for_group_headers() -> None:
    # [FUNCTION DOC] Verifies the selected completion's render line accounts for category header lines.
    # KNOWN ISSUE: synchronous test carries the anyio marker, which has no effect here.
    state = CompletionState(
        items=(
            CompletionItem(
                display="/session",
                replacement="/session",
                start=0,
                end=2,
                kind=CompletionKind.COMMAND,
                category="Commands",
            ),
            CompletionItem(
                display="/example",
                replacement="/example",
                start=0,
                end=2,
                kind=CompletionKind.PROMPT_TEMPLATE,
                category="Custom prompts",
            ),
        ),
        selected_index=1,
    )

    assert _completion_selected_render_line(state) == 3


@pytest.mark.anyio
def test_visible_completion_state_keeps_selected_item_in_render_window() -> None:
    # [FUNCTION DOC] Verifies the windowed completion state keeps a deep selection visible within the line
    # budget while dropping some items. KNOWN ISSUE: sync test with a meaningless anyio marker.
    items = tuple(
        CompletionItem(
            display=f"/prompt-{index:02d}",
            replacement=f"/prompt-{index:02d}",
            start=0,
            end=1,
            kind=CompletionKind.PROMPT_TEMPLATE,
            category="Custom prompts",
        )
        for index in range(30)
    )
    state = CompletionState(items=items, selected_index=24)

    visible = _visible_completion_state(state, max_lines=8)

    assert visible.selected is not None
    assert visible.selected.display == "/prompt-24"
    assert visible.selected_index < len(visible.items)
    assert len(visible.items) < len(items)
    assert _completion_selected_render_line(visible) < 8


def test_visible_completion_state_accounts_for_wrapped_descriptions() -> None:
    # [FUNCTION DOC] Verifies completion windowing accounts for wrapped descriptions at a given width.
    items = tuple(
        CompletionItem(
            display=f"/prompt-{index:02d}",
            replacement=f"/prompt-{index:02d}",
            start=0,
            end=1,
            kind=CompletionKind.PROMPT_TEMPLATE,
            description=(
                "This prompt has a long description that wraps across multiple lines "
                "inside the completion table."
            ),
            category="Custom prompts",
        )
        for index in range(12)
    )
    state = CompletionState(items=items, selected_index=8)

    visible = _visible_completion_state(state, max_lines=8, width=48)

    assert visible.selected is not None
    assert visible.selected.display == "/prompt-08"
    assert tui_app._completion_render_line_count(visible, width=48) <= 8
    assert _completion_selected_render_line(visible, width=48) < 7


def test_visible_completion_state_keeps_selected_item_above_bottom_edge() -> None:
    # [FUNCTION DOC] Verifies the selected item is kept above the bottom edge of the completion window.
    items = tuple(
        CompletionItem(
            display=f"/prompt-{index:02d}",
            replacement=f"/prompt-{index:02d}",
            start=0,
            end=1,
            kind=CompletionKind.PROMPT_TEMPLATE,
            category="Custom prompts",
        )
        for index in range(30)
    )
    state = CompletionState(items=items, selected_index=15)

    visible = _visible_completion_state(state, max_lines=8)

    assert visible.selected is not None
    assert visible.selected.display == "/prompt-15"
    assert _completion_selected_render_line(visible) < 7


@pytest.mark.anyio
async def test_tui_app_scrolls_completion_selection_into_view() -> None:
    # [FUNCTION DOC] Async: verifies the completion window budget fits the terminal and repeated next-selection
    # keeps the selection scrolled into view.
    session = FakeSession()
    session.prompt_templates = tuple(
        PromptTemplate(
            name=f"prompt-{index:02d}",
            path=Path(f"prompt-{index:02d}.md"),
            content="Run.",
        )
        for index in range(30)
    )
    app = TauTuiApp(session)

    async with app.run_test(size=(80, 24)) as pilot:
        prompt = app.query_one("#prompt")
        prompt.focus()
        prompt.value = "/"
        app._completion_state = app._build_completion_state(prompt.value)
        app._refresh_completions()
        await pilot.pause()
        autocomplete = app.query_one("#autocomplete", Static)
        visible_line_limit = app._completion_window_line_budget(autocomplete)
        assert visible_line_limit < tui_app.COMPLETION_MAX_VISIBLE_LINES
        visible = tui_app._visible_completion_state(
            app._completion_state,
            max_lines=visible_line_limit,
        )
        assert visible.items[0].display != "/prompt-00"

        for _ in range(35):
            app.action_completion_next()
            await pilot.pause()

        visible = tui_app._visible_completion_state(
            app._completion_state,
            max_lines=visible_line_limit,
        )
        selected = app._completion_state.selected
        assert selected is not None
        assert visible.selected is not None
        assert visible.selected.display == selected.display
        assert visible.items[0].display != "/prompt-00"
        assert _completion_selected_render_line(visible) < visible_line_limit - 1


@pytest.mark.anyio
async def test_tui_app_uses_terminal_space_for_initial_completion_window() -> None:
    # [FUNCTION DOC] Async: verifies the initial completion window budget uses available terminal space and stays
    # stable when the selection moves.
    session = FakeSession()
    session.prompt_templates = tuple(
        PromptTemplate(
            name=f"prompt-{index:02d}",
            path=Path(f"prompt-{index:02d}.md"),
            content="Run.",
        )
        for index in range(30)
    )
    app = TauTuiApp(session)

    async with app.run_test(size=(80, 18)) as pilot:
        prompt = app.query_one("#prompt")
        prompt.focus()
        prompt.value = "/"
        await pilot.pause()
        autocomplete = app.query_one("#autocomplete", Static)

        initial_line_budget = app._completion_window_line_budget(autocomplete)
        assert initial_line_budget < COMPLETION_MAX_VISIBLE_LINES

        app.action_completion_next()
        await pilot.pause()
        assert app._completion_window_line_budget(autocomplete) == initial_line_budget


@pytest.mark.anyio
@pytest.mark.parametrize("edit_key", ["down", "r"])
async def test_tui_app_keeps_completion_window_height_stable_after_edit(
    edit_key: str,
) -> None:
    # [FUNCTION DOC] Parametrized async: verifies the completion window budget stays stable after navigation
    # or an edit shrinks the list.
    session = FakeSession()
    session.prompt_templates = tuple(
        PromptTemplate(
            name=f"prompt-{index:02d}",
            path=Path(f"prompt-{index:02d}.md"),
            content="Run.",
        )
        for index in range(30)
    )
    app = TauTuiApp(session)

    async with app.run_test(size=(80, 24)) as pilot:
        prompt = app.query_one("#prompt")
        prompt.focus()
        prompt.value = "/"
        await pilot.pause()
        autocomplete = app.query_one("#autocomplete", Static)
        initial_line_budget = app._completion_window_line_budget(autocomplete)

        await pilot.press(edit_key)
        await pilot.pause()

        assert autocomplete.size.height < initial_line_budget
        assert app._completion_window_line_budget(autocomplete) == initial_line_budget


@pytest.mark.anyio
async def test_tui_app_cycles_completion_selection() -> None:
    # [FUNCTION DOC] Async: verifies scrolling down in the prompt cycles the completion selection.
    app = TauTuiApp(FakeSession())

    async with app.run_test():
        prompt = app.query_one("#prompt")
        prompt.focus()
        prompt.value = "/s"
        app._completion_state = app._build_completion_state(prompt.value)
        app._refresh_completions()

        first = app._completion_state.selected.display if app._completion_state.selected else None
        prompt.action_scroll_down()
        second = app._completion_state.selected.display if app._completion_state.selected else None

        assert first != second


@pytest.mark.anyio
async def test_tui_app_opens_command_palette_from_keybinding() -> None:
    # [FUNCTION DOC] Async: verifies ctrl+k opens the command palette by inserting "/" and showing completions.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        await pilot.press("ctrl+k")

        assert prompt.value == "/"
        assert app._completion_state.items
        assert any(item.display == "/session" for item in app._completion_state.items)
        assert app.query_one("#autocomplete").display is True


def test_tui_model_picker_guides_setup_when_no_provider_is_usable() -> None:
    # [FUNCTION DOC] Verifies opening the model picker with no usable providers emits a warning notification
    # directing the user to /login.
    class UnusableProviderSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession with no available models or model choices.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: initializes the base fake then empties model lists.
            super().__init__()
            self.available_models = ()
            self.available_model_choices = ()

    session = UnusableProviderSession()
    app = TauTuiApp(session)
    notifications: list[tuple[str, str | None]] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records (message, severity-if-string) notifications.
        severity = kwargs.get("severity")
        notifications.append((message, severity if isinstance(severity, str) else None))

    app._notify = fake_notify  # type: ignore[method-assign]

    app._open_model_picker()

    assert notifications == [
        ("No configured providers are usable. Run /login to set up a provider.", "warning")
    ]


@pytest.mark.anyio
async def test_tui_model_picker_opens_with_empty_catalog_and_stale_active_model() -> None:
    # [FUNCTION DOC] Async: verifies the model picker still opens (empty) when the catalog is empty but a stale
    # active model exists.
    session = FakeSession()
    session.available_models = ()
    session.available_model_choices = ()
    session.has_stale_active_model = True  # type: ignore[attr-defined]
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        app._open_model_picker()
        await pilot.pause()
        assert isinstance(app.screen, ModelPickerScreen)
        assert app.screen.visible_choices == ()


@pytest.mark.anyio
async def test_tui_app_deduplicates_active_notifications() -> None:
    # [FUNCTION DOC] Async: verifies identical active notifications are deduplicated per severity.
    app = TauTuiApp(FakeSession())

    async with app.run_test(notifications=True) as pilot:
        app._notify("Thinking controls are not available.", severity="warning")
        app._notify("Thinking controls are not available.", severity="warning")
        app._notify("Thinking controls are not available.", severity="error")
        await pilot.pause()

        active_notifications = tuple(app._notifications)

    assert [
        (notification.message, notification.severity) for notification in active_notifications
    ] == [
        ("Thinking controls are not available.", "warning"),
        ("Thinking controls are not available.", "error"),
    ]


@pytest.mark.anyio
async def test_tui_app_notifications_render_literal_markup_text() -> None:
    # [FUNCTION DOC] Async: verifies notifications are shown as literal text (markup disabled).
    app = TauTuiApp(FakeSession())

    async with app.run_test(notifications=True) as pilot:
        app._notify("Error: value [type=extra_forbidden]", severity="error")
        await pilot.pause()

        [notification] = tuple(app._notifications)

    assert notification.message == "Error: value [type=extra_forbidden]"
    assert notification.markup is False


@pytest.mark.anyio
async def test_tui_app_clicking_transcript_refocuses_prompt() -> None:
    # [FUNCTION DOC] Async: verifies clicking the transcript returns focus to the prompt.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        transcript = app.query_one("#transcript", TranscriptView)
        transcript.focus()
        await pilot.pause()
        assert app.screen.focused is transcript

        await pilot.click("#transcript")
        await pilot.pause()

        assert app.screen.focused is prompt


@pytest.mark.anyio
async def test_tui_app_help_uses_modal_instead_of_transcript() -> None:
    # [FUNCTION DOC] Async: verifies /session output appears in a focused modal output screen, not the transcript.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/session"
        await pilot.press("enter")

        assert isinstance(app.screen, CommandOutputScreen)
        assert app.state.items == []
        assert "Session info" in app.screen.message
        scroll = app.screen.query_one("#command-output-scroll", VerticalScroll)
        assert scroll is not None
        assert app.screen.focused is scroll


@pytest.mark.anyio
async def test_tui_app_tools_reference_opens_filters_and_cancels() -> None:
    # [FUNCTION DOC] Async: verifies /tools opens a searchable tools reference with origin and description
    # length, enter/click opens the description modal, extension sources show by name, and
    # escape returns to the prompt.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/tools"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, ToolsReferenceScreen)
        assert app.screen.focused is app.screen.query_one("#tools-reference-search")
        assert len(app.screen.visible_tools) == len(app.session.tools)
        assert app.state.items == []

        await pilot.press("b", "a", "s", "h")
        await pilot.pause()
        assert [tool.name for tool in app.screen.visible_tools] == ["bash"]
        [label] = app.screen.query("#tools-reference-list Label")
        rendered = str(label.render())
        bash_tool = app.screen.visible_tools[0]
        assert "bash" in rendered
        assert "Built in" in rendered
        assert f"{len(bash_tool.description)} chars" in rendered
        assert bash_tool.description not in rendered

        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, CommandOutputScreen)
        assert app.screen.title_text == "bash — Built in"
        assert app.screen.message == bash_tool.description

        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, ToolsReferenceScreen)
        app.screen.extension_sources["bash"] = "shell-tools"
        app.screen._refresh_tools("bash")
        await pilot.pause()
        [label] = app.screen.query("#tools-reference-list Label")
        assert "shell-tools" in str(label.render())
        assert "Extension:" not in str(label.render())

        await pilot.click("#tools-reference-list > ListItem")
        await pilot.pause()
        assert isinstance(app.screen, CommandOutputScreen)
        assert app.screen.title_text == "bash — shell-tools"
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, ToolsReferenceScreen)

        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, ToolsReferenceScreen)
        assert prompt.value == ""


@pytest.mark.anyio
async def test_tui_app_tools_reference_groups_origins_and_searches_extension_names() -> None:
    # [FUNCTION DOC] Async: verifies tools are grouped built-in first then by extension, and search matches
    # extension names.
    session = FakeSession()
    tools = {tool.name: tool for tool in session.tools}
    session.tools = (tools["bash"], tools["write"], tools["read"], tools["edit"])
    session.extension_tool_sources = {
        "edit": "first-extension",
        "bash": "first-extension",
        "write": "second-extension",
    }
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/tools"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, ToolsReferenceScreen)
        assert [tool.name for tool in app.screen.tools] == ["read", "edit", "bash", "write"]

        await pilot.press(*"first-extension")
        await pilot.pause()
        assert [tool.name for tool in app.screen.visible_tools] == ["edit", "bash"]


@pytest.mark.anyio
async def test_tui_app_tools_reference_shows_empty_and_no_match_states() -> None:
    # [FUNCTION DOC] Async: verifies empty and no-match messages in the tools reference.
    session = FakeSession()
    session.tools = ()
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/tools"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, ToolsReferenceScreen)
        labels = app.screen.query("#tools-reference-list Label")
        assert [str(label.render()) for label in labels] == ["No tools available."]

    session = FakeSession()
    app = TauTuiApp(session)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", PromptInput)
        prompt.value = "/tools"
        await pilot.press("enter")
        await pilot.press("z", "z", "z")
        await pilot.pause()
        labels = app.screen.query("#tools-reference-list Label")
        assert [str(label.render()) for label in labels] == ["No tools match your search."]


@pytest.mark.anyio
async def test_tui_app_reload_appends_command_output_to_transcript() -> None:
    # [FUNCTION DOC] Async: verifies /reload appends a status item, reloads the session and refreshes skills.
    session = FakeSession()
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/reload"
        await pilot.press("enter")
        await pilot.pause()

        assert not isinstance(app.screen, CommandOutputScreen)
        assert session.reload_count == 1
        assert app.state.items == [
            ChatItem(
                role="status",
                text="/reload\nReloaded local coding resources and project context.",
            )
        ]
        assert [skill.name for skill in app.state.skills] == ["reloaded"]


@pytest.mark.anyio
async def test_tui_app_system_appends_markdown_command_output_to_transcript() -> None:
    # [FUNCTION DOC] Async: verifies /system appends a Markdown status item rendered as per-source sections
    # with source colors and tinted backgrounds.
    session = FakeSession()
    base_prompt = "You are Tau.\n"
    context_prompt = "\n".join(f"Guideline {index}" for index in range(80))
    session.system_prompt = base_prompt + context_prompt
    session.system_prompt_sources = (
        SystemPromptSource(
            kind="default",
            label="Tau default prompt",
            source="tau_coding.system_prompt",
            content=base_prompt,
        ),
        SystemPromptSource(
            kind="context",
            label="Project instructions",
            source="/repo/AGENTS.md",
            content=context_prompt,
        ),
    )
    app = TauTuiApp(session)

    async with app.run_test(size=(100, 20)) as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/system"
        await pilot.press("enter")
        await pilot.pause()

        assert not isinstance(app.screen, CommandOutputScreen)
        assert len(app.state.items) == 1
        item = app.state.items[0]
        assert item.role == "status"
        assert item.system_prompt is True
        assert item.text.startswith(
            "### /system\n\n#### 01 · Tau default prompt\n\n"
            "**Source:** `tau_coding.system_prompt`\n\nYou are Tau."
        )
        transcript = app.query_one("#transcript", TranscriptView)
        message = transcript.query_one(TranscriptMessageWidget)
        source_view = message.query_one(SystemPromptSourcesWidget)
        sections = list(source_view.query(SystemPromptSectionWidget))
        assert [section.source_color for section in sections] == [
            TAU_DARK_THEME.accent,
            TAU_DARK_THEME.role_styles["branch_summary"].border,
        ]
        assert sections[0].styles.background == Color.parse(
            TAU_DARK_THEME.transcript_background
        ).blend(Color.parse(TAU_DARK_THEME.accent), 0.08)
        assert sections[1].styles.background == Color.parse(
            TAU_DARK_THEME.transcript_background
        ).blend(
            Color.parse(TAU_DARK_THEME.role_styles["branch_summary"].border),
            0.08,
        )
        assert len(list(message.query(ThemedMarkdownWidget))) == 2


@pytest.mark.anyio
async def test_tui_app_omits_textual_header() -> None:
    # [FUNCTION DOC] Async: verifies no Textual Header is mounted.
    app = TauTuiApp(FakeSession())

    async with app.run_test():
        assert not app.query("Header")


@pytest.mark.anyio
async def test_tui_app_name_updates_sidebar() -> None:
    # [FUNCTION DOC] Async: verifies /name updates the sidebar title.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/name Customer bugfix"
        await pilot.press("enter")
        await pilot.pause()

        sidebar_content = app.query_one("#sidebar-content", Static)
        console = Console(record=True, width=80, file=StringIO())
        console.print(sidebar_content.content)
        assert "Customer bugfix" in console.export_text()


@pytest.mark.anyio
async def test_tui_app_name_success_uses_notification_instead_of_modal() -> None:
    # [FUNCTION DOC] Async: verifies /name success uses a notification, not a modal or transcript item.
    app = TauTuiApp(FakeSession())
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/name Customer bugfix"
        await pilot.press("enter")
        await pilot.pause()

        assert notifications == ["Session renamed: Customer bugfix"]
        assert not isinstance(app.screen, CommandOutputScreen)
        assert app.state.items == []


@pytest.mark.anyio
async def test_tui_app_command_modal_arrow_keys_scroll_output() -> None:
    # [FUNCTION DOC] Async: verifies arrow keys scroll long command modal output.
    app = TauTuiApp(FakeSession())
    long_message = "\n".join(f"line {index}" for index in range(80))

    async with app.run_test(size=(100, 20)) as pilot:
        app._show_command_message("/long", long_message)
        await pilot.pause()

        assert isinstance(app.screen, CommandOutputScreen)
        scroll = app.screen.query_one("#command-output-scroll", VerticalScroll)
        await pilot.pause()
        assert scroll.max_scroll_y > 0
        assert app.screen.focused is scroll
        assert scroll.scroll_y == 0

        await pilot.press("down")
        await pilot.pause()

        assert scroll.scroll_y > 0


@pytest.mark.anyio
async def test_tui_app_command_modal_renders_literal_markup_text() -> None:
    # [FUNCTION DOC] Async: verifies the command modal renders bracketed text literally.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        app._show_command_message("/session", "Session [info]\n/session")
        await pilot.pause()

        assert isinstance(app.screen, CommandOutputScreen)
        body = app.screen.query_one("#command-output-body")
        assert str(body.render()) == "Session [info]\n/session"


@pytest.mark.anyio
async def test_tui_app_command_modal_uses_centered_picker_style() -> None:
    # [FUNCTION DOC] Async: verifies the command modal's centered layout and size constraints.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        app._show_command_message("/session", "Session info")
        await pilot.pause()

        assert isinstance(app.screen, CommandOutputScreen)
        command_output = app.screen.query_one("#command-output")
        command_scroll = app.screen.query_one("#command-output-scroll")
        assert app.screen.styles.align == ("center", "middle")
        assert command_output.styles.width.value == 76
        assert command_output.styles.max_width.value == 90
        assert command_output.styles.height.is_auto
        assert command_output.styles.max_height.value == 70
        assert command_scroll.styles.height.is_auto
        assert command_scroll.styles.max_height.value == 18


@pytest.mark.anyio
async def test_tui_app_session_modal_auto_copies_selected_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC] Async: verifies the /session modal copies selected text even with global auto-copy off.
    # Inputs: monkeypatch.
    app = TauTuiApp(FakeSession(), tui_settings=TuiSettings(auto_copy_selection=False))
    copied: list[str] = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)

    async with app.run_test() as pilot:
        app._show_command_message("/session", "Session info")
        await pilot.pause()

        assert isinstance(app.screen, CommandOutputScreen)
        body = app.screen.query_one("#command-output-body")
        app.screen.selections = {body: SELECT_ALL}

        await app.on_text_selected()

    assert copied == ["Session info"]


@pytest.mark.anyio
async def test_tui_app_non_session_modal_uses_global_auto_copy_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC] Async: verifies other modals respect the disabled global auto-copy setting.
    app = TauTuiApp(FakeSession(), tui_settings=TuiSettings(auto_copy_selection=False))
    copied: list[str] = []
    monkeypatch.setattr(app, "copy_to_clipboard", copied.append)

    async with app.run_test() as pilot:
        app._show_command_message("/hotkeys", "Shortcut info")
        await pilot.pause()

        assert isinstance(app.screen, CommandOutputScreen)
        body = app.screen.query_one("#command-output-body")
        app.screen.selections = {body: SELECT_ALL}

        await app.on_text_selected()

    assert copied == []


@pytest.mark.anyio
async def test_tui_app_escape_cancels_running_session_from_prompt() -> None:
    # [FUNCTION DOC] Async: verifies escape while running cancels the session, clears running state and notifies.
    class RunningSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession variant that always reports running.
        @property
        def is_running(self) -> bool:
            # [FUNCTION DOC] HELPER: property that always returns True.
            return True

    session = RunningSession()
    app = TauTuiApp(session)
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        app.adapter.apply(AgentStartEvent())
        app._refresh()

        await pilot.press("escape")

        assert session.cancel_count == 1
        assert app.state.running is False
        assert notifications == ["Interrupted current operation."]


@pytest.mark.anyio
async def test_tui_app_new_command_cancels_active_run_and_ignores_late_events() -> None:
    # [FUNCTION DOC] Async: verifies /new during a run cancels it, bumps the run id, clears state, and late
    # events from the old run id are ignored.
    session = FakeSession()
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        app.adapter.apply(AgentStartEvent())
        app._refresh()
        old_run_id = app._prompt_run_id
        prompt = app.query_one("#prompt")
        prompt.value = "/new"

        await pilot.press("enter")

        assert session.cancel_count == 1
        assert session.new_session_count == 1
        assert app._prompt_run_id == old_run_id + 1
        assert app.state.items == []
        assert app.state.running is False

        session.events = (MessageEndEvent(message=AssistantMessage(content="late old output")),)
        await app._run_prompt("old prompt", old_run_id)

        assert app.state.items == []


@pytest.mark.anyio
async def test_tui_app_escape_without_running_does_not_append_transcript_status() -> None:
    # [FUNCTION DOC] Async: verifies escape while idle adds no transcript status or notification.
    app = TauTuiApp(FakeSession(messages=[UserMessage(content="Earlier")]))
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        await pilot.press("escape")

        assert [(item.role, item.text) for item in app.state.items] == [("user", "Earlier")]
        assert notifications == []


@pytest.mark.anyio
async def test_tui_app_uses_configured_command_palette_keybinding() -> None:
    # [FUNCTION DOC] Async: verifies a configured command-palette key replaces ctrl+k.
    app = TauTuiApp(
        FakeSession(),
        tui_settings=TuiSettings(keybindings=TuiKeybindings(command_palette="ctrl+j")),
    )

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        await pilot.press("ctrl+k")

        assert prompt.value == ""
        assert app._completion_state.items == ()

        await pilot.press("ctrl+j")

        assert prompt.value == "/"
        assert app._completion_state.items
        assert any(item.display == "/session" for item in app._completion_state.items)


@pytest.mark.anyio
async def test_tui_app_quits_from_focused_prompt_with_default_keybinding() -> None:
    # [FUNCTION DOC] Async: verifies ctrl+d is a visible Quit binding on the prompt and exits the app.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        visible_bindings = [
            binding for binding in prompt._bindings.get_bindings_for_key("ctrl+d") if binding.show
        ]

        assert any(
            binding.action == "quit" and binding.description == "Quit"
            for binding in visible_bindings
        )

        await pilot.press("ctrl+d")
        await pilot.pause()

        assert app._exit is True


@pytest.mark.anyio
async def test_tui_app_uses_configured_completion_keybinding() -> None:
    # [FUNCTION DOC] Async: verifies a configured accept-completion key replaces Tab.
    app = TauTuiApp(
        FakeSession(),
        tui_settings=TuiSettings(keybindings=TuiKeybindings(accept_completion="f2")),
    )

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/se"
        app._completion_state = app._build_completion_state(prompt.value)
        app._refresh_completions()

        await pilot.press("tab")
        assert prompt.value == "/se"

        await pilot.press("f2")
        assert prompt.value == "/session"


@pytest.mark.anyio
async def test_tui_login_saves_provider_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Async: verifies "/login openai" opens the API-key screen and submitting saves the key to
    # credentials.json, reloads provider settings (not resources), switches provider, and never
    # echoes the key. Inputs: monkeypatch, tmp_path (isolated home).
    isolate_home(monkeypatch, tmp_path)
    session = FakeSession()
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/login openai"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, LoginScreen)

        api_key_input = app.screen.query_one("#login-api-key", Input)
        api_key_input.value = "stored-openai-key"
        await pilot.press("enter")
        await pilot.pause()

    assert session.reload_count == 0
    assert session.provider_reload_count == 1
    assert session.provider_name == "openai"
    assert session.prompt_texts == []
    assert all(item.text != "stored-openai-key" for item in app.state.items)
    assert (tmp_path / ".tau" / "credentials.json").read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_tui_anthropic_subscription_alias_opens_oauth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC] Async: verifies "/login anthropic-subscription" opens the OAuth login screen for anthropic
    # and starts the provider login. Inputs: monkeypatch (fake OAuth provider lookup).
    login_started = asyncio.Event()

    class FakeOAuthProvider:
        # [CLASS DOC] HELPER: fake OAuth provider whose login signals start then waits forever.
        async def login(self, _callbacks: object) -> OAuthCredential:
            # [FUNCTION DOC] HELPER: async; sets login_started, blocks indefinitely; the trailing AssertionError is unreachable.
            login_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    fake_provider = FakeOAuthProvider()
    monkeypatch.setattr(tui_app, "get_oauth_provider", lambda _name: fake_provider)
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/login anthropic-subscription"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, OAuthLoginScreen)
        assert app.screen.provider.name == "anthropic"
        assert login_started.is_set()


@pytest.mark.anyio
async def test_tui_anthropic_api_alias_opens_api_key_login() -> None:
    # [FUNCTION DOC] Async: verifies "/login anthropic-api" opens the API-key login screen for anthropic.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/login anthropic-api"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, LoginScreen)
        assert app.screen.provider.name == "anthropic"


@pytest.mark.anyio
async def test_tui_login_openai_codex_saves_oauth_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Async: verifies OpenAI Codex OAuth login saves OAuth credentials, switches provider without
    # changing the default provider, and never shows tokens. Inputs: monkeypatch, tmp_path.
    isolate_home(monkeypatch, tmp_path)
    credential_future = asyncio.get_running_loop().create_future()

    async def fake_login_openai_codex(**_kwargs: object) -> OAuthCredential:
        # [FUNCTION DOC] HELPER: async fake login that awaits a test-controlled future for the credential.
        return await credential_future

    monkeypatch.setattr(tui_app, "login_openai_codex", fake_login_openai_codex)
    session = FakeSession()
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/login openai-codex"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, OAuthLoginScreen)
        credential_future.set_result(
            OAuthCredential(
                access="access-token",
                refresh="refresh-token",
                expires=123456,
                account_id="account-1",
            )
        )
        await pilot.pause()

    assert session.reload_count == 0
    assert session.provider_reload_count == 1
    assert session.provider_name == "openai-codex"
    assert tui_app.load_provider_settings().default_provider == "openai"
    assert all("access-token" not in item.text for item in app.state.items)
    credentials = (tmp_path / ".tau" / "credentials.json").read_text(encoding="utf-8")
    assert '"type": "oauth"' in credentials
    assert "refresh-token" in credentials


@pytest.mark.anyio
async def test_tui_login_custom_provider_writes_catalog_and_preferences(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Async: verifies a custom provider login result writes a catalog entry (trailing slash
    # stripped from base URL), provider preferences with default model, the stored API key, and
    # switches to the provider. Inputs: monkeypatch, tmp_path.
    isolate_home(monkeypatch, tmp_path)
    session = FakeSession()
    app = TauTuiApp(session)

    async with app.run_test():
        app._handle_custom_provider_login_result(
            CustomProviderLoginResult(
                provider_name="nebius",
                display_name="Nebius AI Studio",
                base_url="https://api.studio.nebius.ai/v1/",
                api_key_env="NEBIUS_API_KEY",
                models=("deepseek-ai/DeepSeek-V4-Pro", "Qwen/Qwen3-Coder"),
                default_model="deepseek-ai/DeepSeek-V4-Pro",
                api_key="stored-nebius-key",
            )
        )

    paths = TauPaths(home=tmp_path / ".tau")
    catalog = user_catalog_path(paths).read_text(encoding="utf-8")
    settings = tui_app.load_provider_settings(paths)

    assert 'name = "nebius"' in catalog
    assert 'display_name = "Nebius AI Studio"' in catalog
    assert 'base_url = "https://api.studio.nebius.ai/v1"' in catalog
    assert 'models = ["deepseek-ai/DeepSeek-V4-Pro", "Qwen/Qwen3-Coder"]' in catalog
    assert settings.get_provider("nebius").default_model == "deepseek-ai/DeepSeek-V4-Pro"
    assert FileCredentialStore(tmp_path / ".tau" / "credentials.json").get("nebius") == (
        "stored-nebius-key"
    )
    assert session.provider_reload_count == 1
    assert session.provider_name == "nebius"


@pytest.mark.anyio
async def test_tui_login_custom_provider_opens_from_slash_command() -> None:
    # [FUNCTION DOC] Async: verifies "/login custom" opens the custom provider screen with name field focused.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/login custom"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, CustomProviderLoginScreen)
        assert app.screen.query_one("#custom-provider-name", Input).has_focus


@pytest.mark.anyio
async def test_tui_login_preserves_existing_scoped_models_and_providers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Async: verifies logging in to a built-in provider preserves existing default provider,
    # provider configs and scoped models while storing the new credential.
    isolate_home(monkeypatch, tmp_path)
    tau_home = tmp_path / ".tau"
    settings = ProviderSettings(
        default_provider="local",
        providers=(
            OpenAICompatibleProviderConfig(
                name="local",
                base_url="http://localhost:11434/v1",
                api_key_env="LOCAL_API_KEY",
                models=("qwen",),
                default_model="qwen",
            ),
        ),
        scoped_models=(ScopedModelConfig(provider="local", model="qwen"),),
    )
    save_provider_settings(settings)
    session = FakeSession()
    app = TauTuiApp(session)
    entry = tui_app.builtin_provider_entry("openrouter")
    assert entry is not None

    async with app.run_test():
        app._handle_login_result(entry, "stored-openrouter-key")

    saved = tui_app.load_provider_settings()
    assert saved.default_provider == "local"
    assert saved.get_provider("local").default_model == "qwen"
    assert saved.get_provider("openrouter").credential_name == "openrouter"
    assert saved.scoped_models == (ScopedModelConfig(provider="local", model="qwen"),)
    assert FileCredentialStore(tau_home / "credentials.json").get("openrouter") == (
        "stored-openrouter-key"
    )


@pytest.mark.anyio
async def test_tui_login_provider_does_not_change_default_startup_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Async: verifies login switches the session provider but keeps the default startup provider.
    isolate_home(monkeypatch, tmp_path)
    session = FakeSession()
    app = TauTuiApp(session)
    entry = tui_app.builtin_provider_entry("openrouter")
    assert entry is not None

    async with app.run_test():
        app._handle_login_result(entry, "stored-openrouter-key")

    assert session.provider_reload_count == 1
    assert session.provider_name == "openrouter"
    assert tui_app.load_provider_settings().default_provider == "openai"


@pytest.mark.anyio
async def test_tui_logout_without_stored_credentials_shows_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Async: verifies /logout with no stored credentials shows a warning and does not reload.
    isolate_home(monkeypatch, tmp_path)
    session = FakeSession()
    app = TauTuiApp(session)
    notifications: list[tuple[str, str | None]] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records (message, severity-if-string) notifications.
        severity = kwargs.get("severity")
        notifications.append((message, severity if isinstance(severity, str) else None))

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/logout"
        await pilot.press("enter")
        await pilot.pause()

    assert notifications == [(tui_app.NO_STORED_CREDENTIALS_MESSAGE, "warning")]
    assert session.provider_reload_count == 0


@pytest.mark.anyio
async def test_tui_logout_removes_stored_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Async: verifies "/logout openai" removes the stored key, reloads providers and notifies.
    isolate_home(monkeypatch, tmp_path)
    credential_path = tmp_path / ".tau" / "credentials.json"
    FileCredentialStore(credential_path).set("openai", "stored-openai-key")
    session = FakeSession()
    app = TauTuiApp(session)
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/logout openai"
        await pilot.press("enter")
        await pilot.pause()

    assert FileCredentialStore(credential_path).get("openai") is None
    assert session.provider_reload_count == 1
    assert notifications == [
        "Removed stored API key for OpenAI. "
        "Environment variables and providers.json config are unchanged."
    ]


@pytest.mark.anyio
async def test_tui_logout_removes_oauth_credential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Async: verifies logging out of an OAuth provider removes its OAuth credential and notifies.
    isolate_home(monkeypatch, tmp_path)
    credential_path = tmp_path / ".tau" / "credentials.json"
    FileCredentialStore(credential_path).set_oauth(
        "openai-codex",
        OAuthCredential(
            access="access-token",
            refresh="refresh-token",
            expires=123456,
            account_id="account-1",
        ),
    )
    session = FakeSession()
    app = TauTuiApp(session)
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/logout openai-codex"
        await pilot.press("enter")
        await pilot.pause()

    assert FileCredentialStore(credential_path).get_oauth("openai-codex") is None
    assert session.provider_reload_count == 1
    assert notifications == ["Logged out of OpenAI Codex subscription."]


@pytest.mark.anyio
async def test_tui_logout_opens_stored_credential_provider_picker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC] Async: verifies bare /logout with stored credentials opens a "Logout" picker listing them.
    isolate_home(monkeypatch, tmp_path)
    FileCredentialStore(tmp_path / ".tau" / "credentials.json").set("anthropic", "stored-key")
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/logout"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, LoginProviderPickerScreen)
        title = app.screen.query_one("#login-provider-title", Static)
        assert str(title.render()) == "Logout"
        provider_list = app.screen.query_one("#login-provider-list", ListView)
        labels = [str(item.query_one(Label).render()) for item in provider_list.children]
        assert labels == ["Anthropic — anthropic"]


@pytest.mark.anyio
async def test_tui_login_opens_method_picker() -> None:
    # [FUNCTION DOC] Async: verifies /login opens the method picker (subscription, API key, custom) focused on
    # the first entry.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/login"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, LoginMethodPickerScreen)
        method_list = app.screen.query_one("#login-method-list", ListView)
        labels = [str(item.query_one(Label).render()) for item in method_list.children]
        assert labels == [
            "Subscription — OAuth account",
            "API key — built-in provider",
            "Custom provider — OpenAI-compatible",
        ]
        assert app.screen.focused is method_list
        assert method_list.index == 0


@pytest.mark.anyio
async def test_tui_login_method_picker_supports_arrow_keys() -> None:
    # [FUNCTION DOC] Async: verifies arrow navigation in the method picker and that API key leads to the
    # provider picker starting with OpenAI.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/login"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, LoginMethodPickerScreen)
        method_list = app.screen.query_one("#login-method-list", ListView)
        assert app.screen.focused is method_list
        assert method_list.index == 0

        await pilot.press("down")
        await pilot.pause()
        assert method_list.index == 1

        await pilot.press("up")
        await pilot.pause()
        assert method_list.index == 0

        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, LoginProviderPickerScreen)
        provider_list = app.screen.query_one("#login-provider-list", ListView)
        labels = [str(item.query_one(Label).render()) for item in provider_list.children]
        assert labels[0] == "OpenAI — openai"


@pytest.mark.anyio
async def test_tui_login_escape_returns_from_provider_picker_to_method_picker() -> None:
    # [FUNCTION DOC] Async: verifies escape in the provider picker returns to the method picker.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/login"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, LoginProviderPickerScreen)

        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, LoginMethodPickerScreen)


@pytest.mark.anyio
async def test_tui_login_ctrl_d_closes_modal_without_closing_app() -> None:
    # [FUNCTION DOC] Async: verifies ctrl+d closes login modals without exiting the app.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/login"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, LoginProviderPickerScreen)

        await pilot.press("ctrl+d")
        await pilot.pause()
        assert app.is_running
        assert not isinstance(app.screen, LoginProviderPickerScreen)
        assert not isinstance(app.screen, LoginMethodPickerScreen)


@pytest.mark.anyio
async def test_tui_login_subscription_opens_oauth_provider_picker() -> None:
    # [FUNCTION DOC] Async: verifies the subscription option lists OAuth providers without model names.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/login"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, LoginMethodPickerScreen)
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, LoginProviderPickerScreen)
        provider_list = app.screen.query_one("#login-provider-list", ListView)
        labels = [str(item.query_one(Label).render()) for item in provider_list.children]
        assert labels == [
            "OpenAI Codex subscription — openai-codex",
            "Anthropic — anthropic",
            "GitHub Copilot — github-copilot",
        ]
        assert "gpt-5.5" not in "\n".join(labels)


@pytest.mark.anyio
async def test_tui_login_api_provider_picker_highlights_first_provider() -> None:
    # [FUNCTION DOC] Async: regression test (issue 494) verifying the API provider picker highlights the first
    # entry initially and after filtering, so down/enter pick the expected provider.
    # Regression test for issue #494: the list was populated and its index set
    # in the same pass, so the index was assigned while the list was still
    # empty and validated back to None. Nothing was highlighted when the picker
    # opened and the first down key skipped past the first provider.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/login"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, LoginMethodPickerScreen)
        app.screen.action_cursor_down()
        app.screen.action_select_cursor()
        await pilot.pause()

        assert isinstance(app.screen, LoginProviderPickerScreen)
        screen = app.screen
        provider_list = screen.query_one("#login-provider-list", ListView)

        # The initial refresh must highlight the first provider before the
        # picker can receive a navigation key.
        assert provider_list.index == 0
        assert provider_list.highlighted_child is not None

        # Drive filtering through the search input event path. The refreshed
        # list must still highlight its first match before Down/Enter arrive.
        search = screen.query_one("#login-provider-search", Input)
        search.value = "moonshot"
        await pilot.wait_for_scheduled_animations()
        assert provider_list.index == 0
        assert screen.visible_providers[0].name.startswith("moonshot")

        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, LoginScreen)
        assert app.screen.provider.name == "moonshotai-cn"


@pytest.mark.anyio
async def test_tui_login_api_provider_picker_filters_by_name_and_display_name() -> None:
    # [FUNCTION DOC] Async: verifies the provider picker filters by both provider id and display name.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/login"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, LoginMethodPickerScreen)
        app.screen.action_cursor_down()
        app.screen.action_select_cursor()
        await pilot.pause()

        assert isinstance(app.screen, LoginProviderPickerScreen)
        search = app.screen.query_one("#login-provider-search", Input)
        search.value = "kimi"

        # Flush the asynchronous filter events completely
        await pilot.wait_for_scheduled_animations()

        provider_list = app.screen.query_one("#login-provider-list", ListView)
        labels = [str(item.query_one(Label).render()) for item in provider_list.children]
        assert "Moonshot AI (Kimi) — moonshotai" in labels
        assert "Kimi Code subscription — kimi-code" in labels

        search.value = "moonshotai"

        # Flush the second async filter event
        await pilot.wait_for_scheduled_animations()

        labels = [str(item.query_one(Label).render()) for item in provider_list.children]
        assert labels == [
            "Moonshot AI (Kimi) — moonshotai",
            "Moonshot AI (China) — moonshotai-cn",
        ]

        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, LoginScreen)
        assert app.screen.provider.name == "moonshotai"


@pytest.mark.anyio
async def test_tui_login_api_provider_picker_handles_no_matches() -> None:
    # [FUNCTION DOC] Async: verifies a no-match provider search shows help text, clears the index, and enter
    # does nothing.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/login"
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, LoginMethodPickerScreen)
        app.screen.action_cursor_down()
        app.screen.action_select_cursor()
        await pilot.pause()

        assert isinstance(app.screen, LoginProviderPickerScreen)
        search = app.screen.query_one("#login-provider-search", Input)
        search.value = "no-such-provider"
        # Refreshing the list awaits its mounts, so flush them before asserting.
        await pilot.wait_for_scheduled_animations()

        provider_list = app.screen.query_one("#login-provider-list", ListView)
        assert len(provider_list.children) == 0
        assert provider_list.index is None
        help_text = app.screen.query_one("#login-provider-help", Static)
        assert str(help_text.render()) == "No matching providers - Escape closes"

        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, LoginProviderPickerScreen)


@pytest.mark.anyio
async def test_tui_login_api_key_opens_api_provider_picker() -> None:
    # [FUNCTION DOC] Async: verifies the API key picker excludes subscription providers and selection opens
    # the login screen for the chosen provider.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/login"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, LoginMethodPickerScreen)
        app.screen.action_cursor_down()
        app.screen.action_select_cursor()
        await pilot.pause()

        assert isinstance(app.screen, LoginProviderPickerScreen)
        provider_list = app.screen.query_one("#login-provider-list", ListView)
        labels = [str(item.query_one(Label).render()) for item in provider_list.children]
        assert labels[0] == "OpenAI — openai"
        assert "OpenAI Codex subscription — openai-codex" not in labels

        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, LoginScreen)
        assert app.screen.provider.name == "anthropic"


@pytest.mark.anyio
async def test_tui_model_opens_interactive_picker() -> None:
    # [FUNCTION DOC] Async: verifies /model opens a picker with tabs, the current model marked, search
    # filtering, tab switching, and enter switching provider/model after a catalog refresh,
    # without notifications.
    session = FakeSession()
    app = TauTuiApp(session)
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/model"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, ModelPickerScreen)
        tabs = app.screen.query_one("#model-picker-tabs", Static)
        assert str(tabs.render()) == "Tabs: ● All models  ○ Scoped models"
        model_list = app.screen.query_one("#model-picker-list", OptionList)
        labels = [
            str(model_list.get_option_at_index(i).prompt) for i in range(model_list.option_count)
        ]
        assert labels == [
            "* openai:fake-model",
            "  openai:other-model",
            "  local:local-model",
        ]

        search = app.screen.query_one("#model-picker-search", Input)
        assert search.has_focus
        search.value = "local"
        await pilot.pause()

        labels = [
            str(model_list.get_option_at_index(i).prompt) for i in range(model_list.option_count)
        ]
        assert labels == ["  local:local-model"]

        await pilot.press("tab")
        await pilot.pause()
        assert str(tabs.render()) == "Tabs: ○ All models  ● Scoped models"

        await pilot.press("tab")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

    assert session.provider_name == "local"
    assert session.model == "local-model"
    assert session.prompt_texts == []
    assert session.model_catalog_refresh_count == 1
    assert notifications == []


@pytest.mark.anyio
@pytest.mark.parametrize("command", ["/model", "/scoped-models"])
async def test_tui_closing_model_picker_cancels_its_refresh_without_chrome_rebuild(
    command: str,
) -> None:
    # [FUNCTION DOC] Parametrized async: verifies closing the model picker cancels its pending catalog refresh
    # and does not rebuild the app chrome.
    class PendingCatalogSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession whose catalog refresh blocks forever and records start/cancellation.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: initializes the base fake plus started and cancelled events.
            super().__init__()
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def refresh_model_catalogs(self) -> None:
            # [FUNCTION DOC] HELPER: async; signals start, waits forever, and on cancellation sets cancelled and re-raises.
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    session = PendingCatalogSession()
    app = TauTuiApp(session)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = command
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, ModelPickerScreen)
        assert session.started.is_set()
        chrome_refreshes = 0
        original_refresh_chrome = app._refresh_chrome

        def track_chrome(*args: object, **kwargs: object) -> None:
            # [FUNCTION DOC] HELPER: counts chrome refreshes (nonlocal) and delegates to the original.
            nonlocal chrome_refreshes
            chrome_refreshes += 1
            original_refresh_chrome(*args, **kwargs)

        app._refresh_chrome = track_chrome  # type: ignore[method-assign]
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, ModelPickerScreen)
        assert session.cancelled.is_set()
        assert chrome_refreshes == 0


@pytest.mark.anyio
async def test_tui_scoped_picker_refreshes_chrome_when_thinking_changes() -> None:
    # [FUNCTION DOC] Async: verifies closing the scoped picker refreshes chrome once when the thinking level
    # changed while it was open.
    session = FakeSession()
    original_toggle = session.toggle_scoped_model

    def toggle_and_change_thinking(choice: ModelChoice) -> tuple[ModelChoice, ...]:
        # [FUNCTION DOC] HELPER: wraps toggle_scoped_model to also set thinking level "high" before delegating.
        session.thinking_level = "high"
        return original_toggle(choice)

    session.toggle_scoped_model = toggle_and_change_thinking  # type: ignore[method-assign]
    app = TauTuiApp(session)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/scoped-models"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, ModelPickerScreen)
        assert app.screen.initial_thinking_level == "medium"
        assert session.thinking_level == "high"
        refreshes = 0
        original_refresh_chrome = app._refresh_chrome

        def track_chrome(*args: object, **kwargs: object) -> None:
            # [FUNCTION DOC] HELPER: counts chrome refreshes and delegates to the original.
            nonlocal refreshes
            refreshes += 1
            original_refresh_chrome(*args, **kwargs)

        app._refresh_chrome = track_chrome  # type: ignore[method-assign]
        await pilot.press("escape")
        await pilot.pause()
        assert refreshes == 1


@pytest.mark.anyio
async def test_tui_model_selection_runs_after_picker_closes() -> None:
    # [FUNCTION DOC] Async: verifies the model switch runs only after the picker screen has closed.
    session = FakeSession()
    app = TauTuiApp(session)
    was_closed: list[bool] = []
    async with app.run_test() as pilot:

        async def track_switch(choice: ModelChoice) -> None:
            # [FUNCTION DOC] HELPER: async replacement switcher recording whether the picker was already dismissed.
            del choice
            was_closed.append(not isinstance(app.screen, ModelPickerScreen))

        app._switch_model = track_switch  # type: ignore[method-assign]
        prompt = app.query_one("#prompt")
        prompt.value = "/model"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert was_closed == [True]


@pytest.mark.anyio
async def test_tui_model_picker_virtualizes_large_cached_catalog() -> None:
    # [FUNCTION DOC] Async: verifies a 1200-model catalog is rendered virtually in an OptionList without ListItems.
    session = FakeSession()
    session.available_model_choices = tuple(
        ModelChoice("openai", f"model-{index}") for index in range(1200)
    )
    app = TauTuiApp(session)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/model"
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, ModelPickerScreen)
        model_list = app.screen.query_one("#model-picker-list", OptionList)
        assert model_list.option_count == 1200
        assert not list(model_list.query(ListItem))


@pytest.mark.anyio
@pytest.mark.parametrize("command", ["/model", "/scoped-models"])
async def test_tui_model_picker_shows_cached_choices_before_remote_refresh(command: str) -> None:
    # [FUNCTION DOC] Parametrized async: verifies cached choices show immediately, then update in place after the
    # remote refresh, and re-applying identical choices keeps option identity.
    class SlowCatalogSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession whose catalog refresh waits for release then replaces choices.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: initializes the base fake plus refresh_started and finish_refresh events.
            super().__init__()
            self.refresh_started = asyncio.Event()
            self.finish_refresh = asyncio.Event()

        async def refresh_model_catalogs(self) -> None:
            # [FUNCTION DOC] HELPER: async; signals start, waits for release, then sets two model choices.
            self.refresh_started.set()
            await self.finish_refresh.wait()
            self.available_model_choices = (
                ModelChoice("openai", "fake-model"),
                ModelChoice("openai", "new-model"),
            )

    session = SlowCatalogSession()
    app = TauTuiApp(session)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = command
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, ModelPickerScreen)
        picker = app.screen
        model_list = picker.query_one("#model-picker-list", OptionList)
        assert model_list.option_count == 3
        assert picker.visible_choices[1] == ModelChoice("openai", "other-model")
        assert session.refresh_started.is_set()

        session.finish_refresh.set()
        await pilot.pause()
        assert model_list.option_count == 2
        assert picker.visible_choices[1] == ModelChoice("openai", "new-model")
        first = model_list.get_option_at_index(0)
        picker.update_choices(session.available_model_choices, ())
        assert model_list.get_option_at_index(0) is first


@pytest.mark.anyio
async def test_tui_scoped_models_picker_toggles_scoped_models_without_switching_model() -> None:
    # [FUNCTION DOC] Async: verifies /scoped-models enter toggles scoped membership (shown as "[scoped]")
    # without switching the active model.
    session = FakeSession()
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/scoped-models"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, ModelPickerScreen)
        assert session.model_catalog_refresh_count == 1
        tabs = app.screen.query_one("#model-picker-tabs", Static)
        assert str(tabs.render()) == "Tabs: ● All models  ○ Scoped models"
        await pilot.press("enter")
        await pilot.pause()

        assert session.scoped_model_choices == (
            ModelChoice(provider_name="openai", model="fake-model"),
        )
        assert session.provider_name == "openai"
        assert session.model == "fake-model"
        model_list = app.screen.query_one("#model-picker-list", OptionList)
        labels = [
            str(model_list.get_option_at_index(i).prompt) for i in range(model_list.option_count)
        ]
        assert labels[0] == "* openai:fake-model [scoped]"

        await pilot.press("enter")
        await pilot.pause()

        assert session.scoped_model_choices == ()
        assert session.provider_name == "openai"
        assert session.model == "fake-model"


@pytest.mark.anyio
async def test_tui_scoped_models_picker_tab_shows_only_scoped_models_for_unselect() -> None:
    # [FUNCTION DOC] Async: verifies the scoped tab lists only scoped models and enter unscopes the selection.
    session = FakeSession()
    session.scoped_model_choices = (
        ModelChoice(provider_name="openai", model="fake-model"),
        ModelChoice(provider_name="openai", model="other-model"),
    )
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/scoped-models"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, ModelPickerScreen)
        assert session.model_catalog_refresh_count == 1
        await pilot.press("tab")
        await pilot.pause()

        tabs = app.screen.query_one("#model-picker-tabs", Static)
        assert str(tabs.render()) == "Tabs: ○ All models  ● Scoped models"
        model_list = app.screen.query_one("#model-picker-list", OptionList)
        labels = [
            str(model_list.get_option_at_index(i).prompt) for i in range(model_list.option_count)
        ]
        assert labels == [
            "* openai:fake-model [scoped]",
            "  openai:other-model [scoped]",
        ]

        await pilot.press("enter")
        await pilot.pause()

        assert session.scoped_model_choices == (
            ModelChoice(provider_name="openai", model="other-model"),
        )
        assert session.provider_name == "openai"
        assert session.model == "fake-model"
        labels = [
            str(model_list.get_option_at_index(i).prompt) for i in range(model_list.option_count)
        ]
        assert labels == ["  openai:other-model [scoped]"]

        await pilot.press("tab")
        await pilot.pause()

        tabs = app.screen.query_one("#model-picker-tabs", Static)
        assert str(tabs.render()) == "Tabs: ● All models  ○ Scoped models"


@pytest.mark.anyio
async def test_tui_app_runs_terminal_command_and_adds_context() -> None:
    # [FUNCTION DOC] Async: verifies "! cmd" runs a terminal command added to context and renders a tool item
    # with the bash status and output, without prompting the agent.
    session = FakeSession()
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "! pwd"
        await pilot.press("enter")
        await pilot.pause()

    assert session.terminal_commands == [("pwd", True)]
    assert session.prompt_texts == []
    assert [(item.role, item.text, item.tool_result_text) for item in app.state.items] == [
        ("tool", "$ pwd", "✓ bash · added to context\ncommand output")
    ]


@pytest.mark.anyio
async def test_tui_app_runs_terminal_command_without_context() -> None:
    # [FUNCTION DOC] Async: verifies "!! cmd" runs without adding to context and always shows its result.
    session = FakeSession()
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "!! pwd"
        await pilot.press("enter")
        await pilot.pause()

    assert session.terminal_commands == [("pwd", False)]
    assert session.prompt_texts == []
    assert app.state.items[-1].tool_result_text == "✓ bash · not added to context\ncommand output"
    assert app.state.items[-1].always_show_tool_result is True


@pytest.mark.anyio
async def test_tui_app_terminal_command_does_not_cancel_active_agent() -> None:
    # [FUNCTION DOC] Async: verifies a terminal command can run while the agent is active without cancelling it.
    started = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()

    class RunningSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession whose prompt starts, blocks until released, then ends and settles.
        async def prompt(
            self,
            text: str,
            **kwargs: object,
        ) -> AsyncIterator[AgentEvent]:
            # [FUNCTION DOC] HELPER: async generator; records text, yields start, signals started, waits for release,
            # yields end and settled, then sets completed.
            del kwargs
            self.prompt_texts.append(text)
            yield AgentStartEvent()
            started.set()
            await release.wait()
            yield AgentEndEvent()
            yield AgentSettledEvent()
            completed.set()

    session = RunningSession()
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "keep working"
        await pilot.press("enter")
        await started.wait()

        prompt.value = "!! code ."
        await pilot.press("enter")
        await pilot.pause()

        assert session.terminal_commands == [("code .", False)]
        assert not completed.is_set()

        release.set()
        await pilot.pause()

        assert completed.is_set()
        assert app.state.running is False


@pytest.mark.anyio
@pytest.mark.parametrize("add_to_context", [True, False])
async def test_tui_app_renders_terminal_command_while_running(add_to_context: bool) -> None:
    # [FUNCTION DOC] Parametrized async: verifies a running terminal command renders immediately as a pending
    # "$ cmd" tool item and receives its result with the correct context label when done.
    session = FakeSession()
    app = TauTuiApp(session)
    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_run_terminal_command(
        command: str,
        *,
        add_to_context: bool,
    ) -> TerminalCommandResult:
        # [FUNCTION DOC] HELPER: async fake runner that signals start, waits for release, returns a successful result.
        started.set()
        await release.wait()
        return TerminalCommandResult(
            command=command,
            output="finished",
            exit_code=0,
            ok=True,
            added_to_context=add_to_context,
        )

    session.run_terminal_command = fake_run_terminal_command  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        task = asyncio.create_task(
            app._run_terminal_command("sleep 1", add_to_context=add_to_context)
        )
        await started.wait()
        await pilot.pause()

        assert [(item.role, item.text, item.tool_result_text) for item in app.state.items] == [
            ("tool", "$ sleep 1", None)
        ]
        assert app.state.items[-1].always_show_tool_result is True

        release.set()
        await task

    context_label = "added to context" if add_to_context else "not added to context"
    assert app.state.items[-1].tool_result_text == f"✓ bash · {context_label}\nfinished"


@pytest.mark.anyio
async def test_tui_app_marks_failed_terminal_command_as_error() -> None:
    # [FUNCTION DOC] Async: verifies a non-zero exit renders a failed (✗) bash result.
    session = FakeSession()
    app = TauTuiApp(session)

    async def fake_run_terminal_command(
        command: str,
        *,
        add_to_context: bool,
    ) -> TerminalCommandResult:
        # [FUNCTION DOC] HELPER: async fake runner returning a failed result with exit code 2.
        return TerminalCommandResult(
            command=command,
            output="failed",
            exit_code=2,
            ok=False,
            added_to_context=add_to_context,
        )

    session.run_terminal_command = fake_run_terminal_command  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "!! false"
        await pilot.press("enter")
        await pilot.pause()

    assert session.prompt_texts == []
    assert app.state.items[-1].text == "$ false"
    assert app.state.items[-1].tool_result_text == "✗ bash · not added to context\nfailed"


@pytest.mark.anyio
async def test_tui_app_marks_terminal_command_exception_as_failed() -> None:
    # [FUNCTION DOC] Async: verifies an exception from the runner renders as a failed result with its message.
    session = FakeSession()
    app = TauTuiApp(session)

    async def fake_run_terminal_command(
        command: str,
        *,
        add_to_context: bool,
    ) -> TerminalCommandResult:
        # [FUNCTION DOC] HELPER: async fake runner that always raises RuntimeError("boom").
        del command, add_to_context
        raise RuntimeError("boom")

    session.run_terminal_command = fake_run_terminal_command  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "!! false"
        await pilot.press("enter")
        await pilot.pause()

    assert app.state.items[-1].text == "$ false"
    assert app.state.items[-1].tool_result_text == "✗ bash · not added to context\nboom"


@pytest.mark.anyio
async def test_tui_app_renders_terminal_command_output_when_tool_results_are_collapsed() -> None:
    # [FUNCTION DOC] Async (no awaits): verifies an always-show terminal item renders its output even when tool
    # results are collapsed.
    item = ChatItem(
        role="tool",
        text="$ pwd",
        tool_result_text="✓ bash · not added to context\ncommand output",
        always_show_tool_result=True,
    )

    console = Console(record=True, width=80)
    console.print(render_chat_item(item, show_tool_results=item.always_show_tool_result))

    assert "command output" in console.export_text()


@pytest.mark.anyio
async def test_tui_app_limits_terminal_command_output_preview() -> None:
    # [FUNCTION DOC] Async: verifies terminal output previews are truncated to 120 lines with a "N more lines" note.
    session = FakeSession()
    app = TauTuiApp(session)
    output = "\n".join(f"line {index}" for index in range(130))

    async def fake_run_terminal_command(
        command: str,
        *,
        add_to_context: bool,
    ) -> TerminalCommandResult:
        # [FUNCTION DOC] HELPER: async fake runner returning 130 lines of successful output.
        return TerminalCommandResult(
            command=command,
            output=output,
            exit_code=0,
            ok=True,
            added_to_context=add_to_context,
        )

    session.run_terminal_command = fake_run_terminal_command  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "!! seq 130"
        await pilot.press("enter")
        await pilot.pause()

    result_text = app.state.items[-1].tool_result_text
    assert result_text is not None
    assert "line 119" in result_text
    assert "line 120" not in result_text
    assert "10 more lines" in result_text


@pytest.mark.anyio
async def test_tui_app_toggles_tool_results_from_keybinding() -> None:
    # [FUNCTION DOC] Async: verifies ctrl+o toggles show_tool_results on and off.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        assert app.state.show_tool_results is False
        await pilot.press("ctrl+o")
        await pilot.pause()
        assert app.state.show_tool_results is True
        await pilot.press("ctrl+o")
        await pilot.pause()

    assert app.state.show_tool_results is False


@pytest.mark.anyio
async def test_tool_result_toggle_expands_full_bash_command() -> None:
    # [FUNCTION DOC] Async: verifies a bash item shows its description collapsed and the full multi-line command
    # plus result when expanded.
    command = "python - <<'PY'\nprint('one')\nprint('two')\nPY"
    app = TauTuiApp(
        FakeSession(
            messages=[
                AssistantMessage(
                    content=[
                        ToolCall(
                            id="call-1",
                            name="bash",
                            arguments={
                                "command": command,
                                "description": "Running inline script",
                            },
                        )
                    ]
                ),
                ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="bash",
                    content="finished",
                ),
            ]
        )
    )

    async with app.run_test() as pilot:
        widget = next(w for w in app.query(TranscriptMessageWidget) if w.item.role == "tool")
        assert widget.selection_text == "→ Running inline script"

        await pilot.press("ctrl+o")
        await pilot.pause()

        widget = next(w for w in app.query(TranscriptMessageWidget) if w.item.role == "tool")
        assert widget.selection_text == (
            f"→ Running inline script\n$ {command}\n\n✓ bash\nfinished"
        )


@pytest.mark.anyio
async def test_tool_result_toggle_preserves_unrelated_message_widgets() -> None:
    # [FUNCTION DOC] Async: verifies toggling tool results keeps unrelated message widgets mounted.
    app = TauTuiApp(
        FakeSession(
            messages=[
                UserMessage(content="earlier"),
                AssistantMessage(
                    content=[ToolCall(id="call-1", name="read", arguments={"path": "README.md"})]
                ),
                ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="read",
                    content=[TextContent(text="contents")],
                ),
            ]
        )
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        history_widget = next(
            widget for widget in app.query(TranscriptMessageWidget) if widget.item.text == "earlier"
        )

        await pilot.press("ctrl+o")
        await pilot.pause()

        assert history_widget.parent is app.query_one("#transcript", TranscriptView)
        tool_widget = next(
            widget for widget in app.query(TranscriptMessageWidget) if widget.item.role == "tool"
        )
        assert "contents" in tool_widget.selection_text


@pytest.mark.anyio
async def test_tui_app_queues_steering_prompt_while_running() -> None:
    # [FUNCTION DOC] Async: verifies enter while running queues a steering message, clears the prompt, and the
    # queue display shows only its first line.
    session = FakeSession()
    app = TauTuiApp(session)
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        app.state.running = True
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "adjust course\nwith extra detail"

        await pilot.press("enter")
        await pilot.pause()

        queued_messages = app.query_one("#queued-messages")
        assert prompt.text == ""
        assert session.prompt_texts == ["adjust course\nwith extra detail"]
        assert session.streaming_behaviors == ["steer"]
        assert app.state.queued_steering == ("adjust course\nwith extra detail",)
        assert app.state.queued_follow_up == ()
        assert queued_messages.display is True
        rendered_queue = tui_app._render_queued_messages(
            app.state,
            theme=app.tui_settings.resolved_theme,
        )
        rendered_rows = [str(row) for row in rendered_queue.renderables]
        assert "↪ steering · queued: adjust course" in rendered_rows
        assert all("with extra detail" not in row for row in rendered_rows)

    assert notifications == []


@pytest.mark.anyio
async def test_tui_app_queues_follow_up_prompt_from_keybinding() -> None:
    # [FUNCTION DOC] Async: verifies alt+enter while running queues a follow-up shown by its first line.
    session = FakeSession()
    app = TauTuiApp(session)
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        app.state.running = True
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "after this\nwith extra detail"

        await pilot.press("alt+enter")
        await pilot.pause()

        queued_messages = app.query_one("#queued-messages")
        assert prompt.text == ""
        assert session.prompt_texts == ["after this\nwith extra detail"]
        assert session.streaming_behaviors == ["follow_up"]
        assert app.state.queued_steering == ()
        assert app.state.queued_follow_up == ("after this\nwith extra detail",)
        assert queued_messages.display is True
        rendered_queue = tui_app._render_queued_messages(
            app.state,
            theme=app.tui_settings.resolved_theme,
        )
        rendered_rows = [str(row) for row in rendered_queue.renderables]
        assert "↳ follow-up · queued: after this" in rendered_rows
        assert all("with extra detail" not in row for row in rendered_rows)

    assert notifications == []


@pytest.mark.anyio
async def test_tui_app_up_arrow_edits_latest_queued_follow_up() -> None:
    # [FUNCTION DOC] Async: verifies up on an empty prompt pops the latest queued follow-up into the prompt.
    session = FakeSession(messages=[UserMessage(content="remembered prompt")])
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        app.state.running = True
        session.queued_follow_up_messages = ("first follow-up", "latest follow-up")
        app._refresh()

        prompt = app.query_one("#prompt", TextArea)
        prompt.focus()
        prompt.text = ""
        await pilot.press("up")
        await pilot.pause()

        assert prompt.text == "latest follow-up"
        assert session.queued_follow_up_messages == ("first follow-up",)
        assert app.state.queued_follow_up == ("first follow-up",)
        queued_messages = app.query_one("#queued-messages")
        assert queued_messages.display is True


@pytest.mark.anyio
async def test_tui_app_up_arrow_edits_latest_queued_steering_message() -> None:
    # [FUNCTION DOC] Async: verifies up pops the latest queued steering message into the prompt.
    session = FakeSession(messages=[UserMessage(content="remembered prompt")])
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        app.state.running = True
        session.queued_steering_messages = ("first steering", "latest steering")
        app._refresh()

        prompt = app.query_one("#prompt", TextArea)
        prompt.focus()
        prompt.text = ""
        await pilot.press("up")
        await pilot.pause()

        assert prompt.text == "latest steering"
        assert session.queued_steering_messages == ("first steering",)
        assert app.state.queued_steering == ("first steering",)
        queued_messages = app.query_one("#queued-messages")
        assert queued_messages.display is True


@pytest.mark.anyio
async def test_tui_app_up_arrow_prefers_queued_follow_up_before_steering() -> None:
    # [FUNCTION DOC] Async: verifies queued follow-ups are recalled before steering messages.
    session = FakeSession(messages=[UserMessage(content="remembered prompt")])
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        app.state.running = True
        session.queued_steering_messages = ("queued steering",)
        session.queued_follow_up_messages = ("queued follow-up",)
        app._refresh()

        prompt = app.query_one("#prompt", TextArea)
        prompt.focus()
        prompt.text = ""
        await pilot.press("up")
        await pilot.pause()

        assert prompt.text == "queued follow-up"
        assert session.queued_steering_messages == ("queued steering",)
        assert session.queued_follow_up_messages == ()
        assert app.state.queued_steering == ("queued steering",)
        assert app.state.queued_follow_up == ()


@pytest.mark.anyio
async def test_tui_app_up_arrow_recalls_latest_sent_prompt_when_input_is_empty() -> None:
    # [FUNCTION DOC] Async: verifies up on an empty prompt recalls the latest sent prompt with cursor at end.
    session = FakeSession()
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.text = "first prompt"
        await pilot.press("enter")
        await pilot.pause()
        prompt.text = "latest prompt"
        await pilot.press("enter")
        await pilot.pause()

        prompt.text = ""
        await pilot.press("up")
        await pilot.pause()

        assert prompt.text == "latest prompt"
        assert prompt.cursor_location == (0, len("latest prompt"))
        assert session.prompt_texts == ["first prompt", "latest prompt"]


@pytest.mark.anyio
async def test_tui_app_up_arrow_recalls_latest_restored_user_message() -> None:
    # [FUNCTION DOC] Async: verifies up recalls the latest restored user message.
    app = TauTuiApp(
        FakeSession(
            messages=[
                UserMessage(content="earlier prompt"),
                AssistantMessage(content="response"),
                UserMessage(content="restored prompt"),
            ]
        )
    )

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.focus()
        prompt.text = ""

        await pilot.press("up")
        await pilot.pause()

        assert prompt.text == "restored prompt"
        assert prompt.cursor_location == (0, len("restored prompt"))


@pytest.mark.anyio
async def test_tui_app_up_arrow_preserves_non_empty_prompt_movement() -> None:
    # [FUNCTION DOC] Async: verifies up in a non-empty multi-line prompt moves the cursor instead of recalling.
    app = TauTuiApp(FakeSession(messages=[UserMessage(content="remembered prompt")]))

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.focus()
        prompt.text = "first line\nsecond line"
        prompt.move_cursor((1, len("second line")))

        await pilot.press("up")
        await pilot.pause()

        assert prompt.text == "first line\nsecond line"
        assert prompt.cursor_location == (0, len("first line"))


@pytest.mark.anyio
async def test_tui_app_toggles_thinking_tokens_from_keybinding_while_running() -> None:
    # [FUNCTION DOC] Async: verifies ctrl+t while running toggles thinking text vs. a placeholder hint without
    # stopping the run or notifying.
    app = TauTuiApp(FakeSession())
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    def transcript_text() -> str:
        # [FUNCTION DOC] HELPER: returns the transcript's rendered lines joined with newlines.
        transcript = app.query_one("#transcript", TranscriptView)
        return "\n".join(line.text for line in transcript.lines)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        app.state.running = True
        app.state.add_thinking_delta("internal plan")
        app.state.add_item("assistant", "final answer")
        app._refresh()
        await pilot.pause()

        assert app.state.show_thinking is False
        assert "final answer" in transcript_text()
        assert "Thinking… Press Ctrl+T to show thinking tokens." in transcript_text()
        assert "internal plan" not in transcript_text()

        await pilot.press("ctrl+t")
        await pilot.pause()
        assert app.state.show_thinking is True
        assert app.state.running is True
        assert "internal plan" in transcript_text()
        assert "Thinking… Press Ctrl+T to show thinking tokens." not in transcript_text()

        await pilot.press("ctrl+t")
        await pilot.pause()
        assert app.state.show_thinking is False
        assert "Thinking… Press Ctrl+T to show thinking tokens." in transcript_text()
        assert "internal plan" not in transcript_text()

    assert notifications == []


@pytest.mark.anyio
async def test_tui_app_hidden_thinking_placeholder_stays_before_streamed_answer() -> None:
    # [FUNCTION DOC] Async: verifies the hidden-thinking placeholder stays before the streamed answer.
    partial = AssistantMessage()
    session = FakeSession(
        events=[
            AgentStartEvent(),
            MessageUpdateEvent(
                message=partial,
                assistant_message_event=ThinkingDeltaEvent(
                    content_index=0, delta="private plan", partial=partial
                ),
            ),
            MessageStartEvent(message=partial),
            MessageUpdateEvent(
                message=partial,
                assistant_message_event=TextDeltaEvent(
                    content_index=0, delta="public answer", partial=partial
                ),
            ),
            MessageEndEvent(message=AssistantMessage(content="public answer")),
            AgentEndEvent(),
        ]
    )
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        await app._run_prompt("stream")
        await pilot.pause()

        transcript = app.query_one("#transcript", TranscriptView)
        assert [line.text for line in transcript.lines] == [
            "Thinking… Press Ctrl+T to show thinking tokens.",
            "public answer",
        ]


@pytest.mark.anyio
async def test_tui_app_restored_thinking_toggles_in_persisted_order() -> None:
    # [FUNCTION DOC] Async: verifies restored interleaved thinking/text blocks toggle in persisted order.
    app = TauTuiApp(
        FakeSession(
            messages=(
                UserMessage(content="prompt"),
                AssistantMessage(
                    content=[
                        ThinkingContent(thinking="first plan"),
                        TextContent(text="first answer"),
                        ThinkingContent(thinking="second plan"),
                        TextContent(text="second answer"),
                    ]
                ),
            )
        )
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        transcript = app.query_one("#transcript", TranscriptView)
        assert [line.text for line in transcript.lines] == [
            "prompt",
            "Thinking… Press Ctrl+T to show thinking tokens.",
            "first answer",
            "Thinking… Press Ctrl+T to show thinking tokens.",
            "second answer",
        ]

        await pilot.press("ctrl+t")
        await pilot.pause()
        assert [line.text for line in transcript.lines] == [
            "prompt",
            "first plan",
            "first answer",
            "second plan",
            "second answer",
        ]

        await pilot.press("ctrl+t")
        await pilot.pause()
        assert [line.text for line in transcript.lines] == [
            "prompt",
            "Thinking… Press Ctrl+T to show thinking tokens.",
            "first answer",
            "Thinking… Press Ctrl+T to show thinking tokens.",
            "second answer",
        ]


@pytest.mark.anyio
async def test_tui_app_thinking_toggle_preserves_unrelated_items() -> None:
    # [FUNCTION DOC] Async: verifies toggling thinking only swaps thinking items and keeps later appended items.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        app.state.add_item("user", "first prompt")
        app.state.add_thinking_delta("plan one")
        app.state.add_item("assistant", "first answer")
        app.state.add_item("user", "second prompt")
        app.state.add_thinking_delta("plan two")
        app.state.add_item("assistant", "second answer")
        app._refresh()
        await pilot.pause()

        transcript = app.query_one("#transcript", TranscriptView)
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert [line.text for line in transcript.lines] == [
            "first prompt",
            "plan one",
            "first answer",
            "second prompt",
            "plan two",
            "second answer",
        ]

        app.state.add_item("status", "late status")
        await transcript.append_item(app.state.items[-1])
        await pilot.pause()
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert [line.text for line in transcript.lines] == [
            "first prompt",
            "Thinking… Press Ctrl+T to show thinking tokens.",
            "first answer",
            "second prompt",
            "Thinking… Press Ctrl+T to show thinking tokens.",
            "second answer",
            "late status",
        ]


@pytest.mark.anyio
async def test_tui_prompt_ctrl_c_clears_text() -> None:
    # [FUNCTION DOC] Async: verifies ctrl+c clears the prompt text.
    app = TauTuiApp(FakeSession(messages=(UserMessage(content="User prompt"),)))

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", TextArea)
        prompt.focus()
        prompt.text = "discard this prompt"
        await pilot.pause()
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert prompt.text == ""


@pytest.mark.anyio
async def test_tui_shortcuts_use_immediate_preview_when_available() -> None:
    # [FUNCTION DOC] Async: verifies model/thinking shortcuts use the session's immediate preview methods when present.
    session = FakeSession()
    session.scoped_model_choices = (
        ModelChoice("openai", "fake-model"),
        ModelChoice("openai", "other-model"),
    )

    def preview_model(*, reverse: bool = False) -> ModelChoice:
        # [FUNCTION DOC] HELPER: fake preview that sets the session model to the second scoped choice and returns it.
        del reverse
        choice = session.scoped_model_choices[1]
        session.model = choice.model
        return choice

    def preview_thinking() -> str:
        # [FUNCTION DOC] HELPER: fake preview that sets thinking level "high" and returns a status string.
        session.thinking_level = "high"
        return "Thinking mode: high"

    session.preview_cycle_scoped_model = preview_model  # type: ignore[attr-defined]
    session.preview_cycle_thinking_level = preview_thinking  # type: ignore[attr-defined]
    app = TauTuiApp(session)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+p")
        assert session.model == "other-model"
        await pilot.press("shift+tab")
        assert session.thinking_level == "high"


@pytest.mark.anyio
async def test_tui_app_cycles_thinking_from_keybinding() -> None:
    # [FUNCTION DOC] Async: verifies shift+tab cycles thinking (medium to high) without notifying.
    session = FakeSession()
    app = TauTuiApp(session)
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        await pilot.press("shift+tab")
        await pilot.pause()

    assert session.thinking_level == "high"
    assert notifications == []


@pytest.mark.anyio
async def test_tui_app_cycles_thinking_from_keybinding_while_running() -> None:
    # [FUNCTION DOC] Async: verifies thinking can be cycled while the agent is running.
    session = FakeSession()
    app = TauTuiApp(session)
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        app.state.running = True
        await pilot.press("shift+tab")
        await pilot.pause()

    assert session.thinking_level == "high"
    assert notifications == []


@pytest.mark.anyio
async def test_tui_app_cycles_scoped_model_from_keybinding() -> None:
    # [FUNCTION DOC] Async: verifies ctrl+p cycles to the next scoped model silently.
    session = FakeSession()
    session.scoped_model_choices = (
        ModelChoice(provider_name="openai", model="fake-model"),
        ModelChoice(provider_name="openai", model="other-model"),
    )
    app = TauTuiApp(session)
    notifications: list[str] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        await pilot.press("ctrl+p")
        await pilot.pause()

    assert session.provider_name == "openai"
    assert session.model == "other-model"
    assert notifications == []


@pytest.mark.anyio
async def test_tui_app_cycles_scoped_model_backward_from_keybinding() -> None:
    # [FUNCTION DOC] Async: verifies ctrl+shift+p cycles scoped models backwards (wrapping to the last).
    session = FakeSession()
    session.scoped_model_choices = (
        ModelChoice(provider_name="openai", model="fake-model"),
        ModelChoice(provider_name="openai", model="other-model"),
        ModelChoice(provider_name="anthropic", model="third-model"),
    )
    app = TauTuiApp(session)

    async with app.run_test() as pilot:
        await pilot.press("ctrl+shift+p")
        await pilot.pause()

    assert session.provider_name == "anthropic"
    assert session.model == "third-model"


@pytest.mark.anyio
async def test_tui_app_cycles_scoped_model_without_redrawing_transcript() -> None:
    # [FUNCTION DOC] Async: verifies cycling the model does not redraw a long transcript.
    session = FakeSession(
        messages=[UserMessage(content=f"Earlier prompt {index}") for index in range(120)]
    )
    session.scoped_model_choices = (
        ModelChoice(provider_name="openai", model="fake-model"),
        ModelChoice(provider_name="openai", model="other-model"),
    )
    app = TauTuiApp(session)
    transcript_refreshes = 0

    async with app.run_test() as pilot:
        transcript = app.query_one("#transcript", TranscriptView)

        def fake_update_from_state(*args: object, **kwargs: object) -> None:
            # [FUNCTION DOC] HELPER: counts transcript update_from_state calls instead of redrawing.
            del args, kwargs
            nonlocal transcript_refreshes
            transcript_refreshes += 1

        transcript.update_from_state = fake_update_from_state  # type: ignore[method-assign]
        await pilot.press("ctrl+p")
        await pilot.pause()

    assert session.provider_name == "openai"
    assert session.model == "other-model"
    assert transcript_refreshes == 0


@pytest.mark.anyio
async def test_tui_app_uses_configured_thinking_keybinding() -> None:
    # [FUNCTION DOC] Async: verifies a configured thinking key replaces shift+tab.
    session = FakeSession()
    app = TauTuiApp(
        session,
        tui_settings=TuiSettings(keybindings=TuiKeybindings(thinking_cycle="f3")),
    )

    async with app.run_test() as pilot:
        await pilot.press("shift+tab")
        await pilot.pause()
        assert session.thinking_level == "medium"

        await pilot.press("f3")
        await pilot.pause()

    assert session.thinking_level == "high"


@pytest.mark.anyio
async def test_tui_prompt_worker_refreshes_directly() -> None:
    # [FUNCTION DOC] Async: verifies the prompt worker (run outside run_test) refreshes once at start and end
    # and clears running.
    app = TauTuiApp(FakeSession(events=[AgentStartEvent(), AgentEndEvent()]))
    refreshes = 0

    def fake_refresh() -> None:
        # [FUNCTION DOC] HELPER: counts refresh calls.
        nonlocal refreshes
        refreshes += 1

    app._refresh = fake_refresh  # type: ignore[method-assign]

    await app._run_prompt("hello")

    assert refreshes == 2
    assert app.state.running is False


@pytest.mark.anyio
async def test_tui_prompt_worker_shows_diagnostic_log_path_for_error_event(tmp_path: Path) -> None:
    # [FUNCTION DOC] Async: verifies an assistant error event shows an error item including the diagnostic
    # log path and retry hint. Inputs: tmp_path.
    class ErrorSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession emitting an errored assistant message, with a diagnostic log path.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: initializes the fake with the error event sequence and sets last_diagnostic_log_path.
            super().__init__(
                events=[
                    AgentStartEvent(),
                    MessageEndEvent(
                        message=AssistantMessage(
                            stop_reason="error", error_message="provider failed"
                        )
                    ),
                    AgentEndEvent(),
                ]
            )
            self.last_diagnostic_log_path = tmp_path / "tau-home" / "logs" / "agent-calls.jsonl"

    session = ErrorSession()
    app = TauTuiApp(session)
    app._refresh = lambda: None  # type: ignore[method-assign]

    await app._run_prompt("break")

    assert app.state.error == (
        f"Error: provider failed\nLog: {session.last_diagnostic_log_path}\n"
        "Run ended before completion. Send a message to retry."
    )
    assert app.state.items[-1].role == "error"
    assert app.state.items[-1].text == app.state.error
    assert app.state.running is False


@pytest.mark.anyio
async def test_tui_prompt_worker_shows_recovery_status_instead_of_overflow_error() -> None:
    # [FUNCTION DOC] Async: verifies a context overflow that triggers compaction and retry shows a recovery
    # status instead of an error and ends with the recovered answer.
    class OverflowSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession emitting overflow error, compaction with retry, then a recovered answer.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: initializes the fake with the overflow-recovery event sequence.
            super().__init__(
                events=[
                    AgentStartEvent(),
                    MessageEndEvent(
                        message=AssistantMessage(
                            stop_reason="error",
                            error_message="prompt is too long: context window exceeded",
                        )
                    ),
                    SessionAgentEndEvent(will_retry=False),
                    CompactionStartEvent(reason="overflow"),
                    CompactionEndEvent(reason="overflow", will_retry=True),
                    AgentStartEvent(),
                    MessageEndEvent(message=AssistantMessage(content="Recovered answer")),
                    SessionAgentEndEvent(will_retry=False),
                    AgentSettledEvent(),
                ]
            )

    session = OverflowSession()
    app = TauTuiApp(session)
    app._refresh = lambda: None  # type: ignore[method-assign]

    await app._run_prompt("break")

    assert app.state.error is None
    assert not any(item.role == "error" for item in app.state.items)
    assert any(
        item.role == "status" and "compacting and retrying" in item.text for item in app.state.items
    )
    assert app.state.items[-1].text == "Recovered answer"
    assert app.state.running is False


@pytest.mark.anyio
async def test_tui_prompt_worker_surfaces_overflow_when_compaction_fails() -> None:
    # [FUNCTION DOC] Async: verifies the overflow error is surfaced when overflow compaction fails.
    message = AssistantMessage(
        stop_reason="error",
        error_message="prompt is too long: context window exceeded",
    )
    session = FakeSession(
        events=[
            AgentStartEvent(),
            MessageEndEvent(message=message),
            SessionAgentEndEvent(will_retry=False),
            CompactionStartEvent(reason="overflow"),
            CompactionEndEvent(
                reason="overflow",
                error_message="Overflow compaction failed",
            ),
            AgentSettledEvent(),
        ]
    )
    app = TauTuiApp(session)
    app._refresh = lambda: None  # type: ignore[method-assign]

    await app._run_prompt("break")

    assert app.state.error is not None
    assert "context window exceeded" in app.state.error
    assert app.state.items[-1].role == "error"
    assert app.state.running is False


@pytest.mark.anyio
async def test_tui_prompt_worker_mounts_provider_error_in_live_transcript() -> None:
    # [FUNCTION DOC] Async: verifies a provider error during a live prompt is mounted as an error widget with
    # the retry hint and the run ends.
    error = AssistantMessage(stop_reason="error", error_message="provider failed")
    app = TauTuiApp(
        FakeSession(
            events=[
                AgentStartEvent(),
                MessageStartEvent(message=error),
                MessageEndEvent(message=error),
                AgentEndEvent(),
            ]
        )
    )

    async with app.run_test(size=(120, 30)) as pilot:
        await app._run_prompt("continue")
        await pilot.pause()

        errors = [
            widget for widget in app.query(TranscriptMessageWidget) if widget.item.role == "error"
        ]
        assert [widget.item.text for widget in errors] == [
            "Error: provider failed\nRun ended before completion. Send a message to retry."
        ]
        assert app.state.running is False


@pytest.mark.anyio
async def test_tui_prompt_worker_shows_diagnostic_log_path_on_failure(tmp_path: Path) -> None:
    # [FUNCTION DOC] Async: verifies an exception with an empty message raised by session.prompt is reported
    # using the exception class name plus the diagnostic log path. Inputs: tmp_path.
    class EmptyMessageError(Exception):
        # [CLASS DOC] HELPER: exception type whose string form is empty.
        def __str__(self) -> str:
            # [FUNCTION DOC] HELPER: returns an empty string.
            return ""

    class FailingSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession whose prompt raises EmptyMessageError and exposes a diagnostic log path.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: initializes the base fake and sets last_diagnostic_log_path under tmp_path.
            super().__init__()
            self.last_diagnostic_log_path = tmp_path / "tau-home" / "logs" / "agent-calls.jsonl"

        async def prompt(self, text: str, **kwargs: object) -> AsyncIterator[AgentEvent]:
            # [FUNCTION DOC] HELPER: async generator that records the text then raises EmptyMessageError (the yield
            # only makes it a generator and is never reached).
            self.prompt_texts.append(text)
            raise EmptyMessageError()
            yield  # pragma: no cover

    session = FailingSession()
    app = TauTuiApp(session)
    app._refresh = lambda: None  # type: ignore[method-assign]

    await app._run_prompt("break")

    assert app.state.error == (f"Error: EmptyMessageError\nLog: {session.last_diagnostic_log_path}")
    assert app.state.items[-1].role == "error"
    assert app.state.items[-1].text == app.state.error
    assert app.state.running is False


@pytest.mark.anyio
async def test_tui_prompt_worker_refreshes_context_after_message_changes() -> None:
    # [FUNCTION DOC] Async: verifies the worker refreshes after each message/tool event so the displayed
    # context estimate tracks session changes, and items are projected correctly.
    class ContextChangingSession(FakeSession):
        # [CLASS DOC] HELPER: FakeSession whose prompt changes context_token_estimate between yielded events.
        async def prompt(self, text: str, **kwargs: object) -> AsyncIterator[AgentEvent]:
            # [FUNCTION DOC] HELPER: async generator yielding start, user message, assistant message, tool start/end and
            # end events while bumping the token estimate from 10 to 50.
            self.prompt_texts.append(text)
            self.context_token_estimate = 10
            yield AgentStartEvent()
            self.context_token_estimate = 20
            yield MessageEndEvent(message=UserMessage(content=text))
            self.context_token_estimate = 30
            yield MessageEndEvent(message=AssistantMessage(content="Using a tool."))
            self.context_token_estimate = 40
            yield ToolExecutionStartEvent(
                tool_call_id="call-1",
                tool_name="read",
                args={"path": "README.md"},
            )
            yield ToolExecutionEndEvent(
                tool_call_id="call-1",
                tool_name="read",
                result=AgentToolResult(content="contents"),
                is_error=False,
            )
            self.context_token_estimate = 50
            yield AgentEndEvent()

    session = ContextChangingSession()
    app = TauTuiApp(session)
    observed_context: list[int] = []

    def fake_refresh() -> None:
        # [FUNCTION DOC] HELPER: records the session's token estimate at each refresh.
        observed_context.append(session.context_token_estimate)

    app._refresh = fake_refresh  # type: ignore[method-assign]

    await app._run_prompt("read README")

    assert observed_context == [10, 20, 30, 40, 40, 50]
    assert [(item.role, item.text, item.tool_result_text) for item in app.state.items] == [
        ("user", "read README", None),
        ("assistant", "Using a tool.", None),
        ("tool", "→ read README.md", "✓ read\ncontents"),
    ]


@pytest.mark.anyio
async def test_tui_resume_refreshes_context_after_session_swap() -> None:
    # [FUNCTION DOC] Async: verifies resume refreshes once after the session swap (showing the new estimate),
    # notifies, and reloads items.
    session = FakeSession(messages=[UserMessage(content="Earlier")])
    app = TauTuiApp(session)
    observed_context: list[int] = []
    notifications: list[str] = []

    def fake_refresh() -> None:
        # [FUNCTION DOC] HELPER: records the session's token estimate at each refresh.
        observed_context.append(session.context_token_estimate)

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records notification messages.
        del kwargs
        notifications.append(message)

    app._refresh = fake_refresh  # type: ignore[method-assign]
    app._notify = fake_notify  # type: ignore[method-assign]

    await app._resume_session("session-1")

    assert observed_context == [456]
    assert notifications == ["Resumed session: session-1"]
    assert [(item.role, item.text) for item in app.state.items] == [
        ("user", "Restored prompt"),
    ]


@pytest.mark.anyio
async def test_tui_app_shows_startup_update_notice_first_in_bright_yellow() -> None:
    # [FUNCTION DOC] Async: verifies startup update notice, resource alert and notices render first in the
    # transcript (update in bright yellow, alert in error color) without notifications or
    # being added to session messages.
    session = FakeSession(messages=[UserMessage(content="Earlier prompt")])
    app = TauTuiApp(
        session,
        startup_update_notice="Tau 0.2.0 is available",
        startup_alerts=("Conflicting skills/prompts detected",),
        startup_notices=("Tau updated to 0.2.0",),
    )
    notifications: list[tuple[str, str | None]] = []

    def fake_notify(message: str, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: records (message, severity-if-string) notifications.
        severity = kwargs.get("severity")
        notifications.append((message, severity if isinstance(severity, str) else None))

    app._notify = fake_notify  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        await pilot.pause()
        transcript = app.query_one("#transcript", TranscriptView)
        assert [line.text for line in transcript.lines] == [
            "Tau 0.2.0 is available",
            "Conflicting skills/prompts detected",
            "Tau updated to 0.2.0",
            "Earlier prompt",
        ]
        widgets = list(transcript.query(TranscriptMessageWidget))
        update_widget = widgets[0]
        assert update_widget.item.highlight == "update"
        assert update_widget._role_style.border == "#ffff00"
        assert update_widget._role_style.body == "bold #ffff00"
        alert_widget = widgets[1]
        assert alert_widget.item.highlight == "alert"
        assert alert_widget._role_style.border == TAU_DARK_THEME.error
        assert alert_widget._role_style.body == f"bold {TAU_DARK_THEME.error}"

    assert notifications == []
    assert [message.text for message in session.messages] == ["Earlier prompt"]


def test_resource_conflict_alert_includes_skill_and_prompt_locations(tmp_path: Path) -> None:
    # [FUNCTION DOC] Verifies the resource conflict alert lists skill and prompt conflicts with their paths,
    # ignores other diagnostics, and ends with remediation text. Inputs: tmp_path.
    diagnostics = (
        ResourceDiagnostic(
            kind="skill",
            name="review",
            path=tmp_path / "project" / ".agents" / "skills" / "review" / "SKILL.md",
            message=(
                "overrides lower-precedence resource at "
                f"{tmp_path / 'home' / '.tau' / 'skills' / 'review' / 'SKILL.md'}"
            ),
        ),
        ResourceDiagnostic(
            kind="prompt",
            name="ship",
            path=tmp_path / "project" / ".tau" / "prompts" / "ship.md",
            message=(
                "overrides lower-precedence resource at "
                f"{tmp_path / 'home' / '.agents' / 'prompts' / 'ship.md'}"
            ),
        ),
        ResourceDiagnostic(kind="context", message="unrelated warning"),
    )

    alert = _resource_conflict_alert(diagnostics)

    assert alert is not None
    assert "Conflicting skills/prompts detected:" in alert
    assert "skill 'review'" in alert
    assert "prompt template 'ship'" in alert
    assert str(diagnostics[0].path) in alert
    assert str(diagnostics[1].path) in alert
    assert "unrelated warning" not in alert
    assert alert.endswith("Rename or remove duplicate resources to clear this alert.")


@pytest.mark.anyio
async def test_tui_app_runs_initial_prompt() -> None:
    # [FUNCTION DOC] Async: verifies an initial_prompt is submitted automatically on startup.
    session = FakeSession(
        events=[
            AgentStartEvent(),
            MessageEndEvent(message=UserMessage(content="explain this repo")),
            AgentEndEvent(),
        ]
    )
    app = TauTuiApp(session, initial_prompt="explain this repo")

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()

    assert session.prompt_texts == ["explain this repo"]
    assert any(item.role == "user" and item.text == "explain this repo" for item in app.state.items)


def test_huggingface_startup_route_prefers_resumed_session_pin(tmp_path: Path) -> None:
    # [FUNCTION DOC] Verifies the Hugging Face startup inference provider prefers a resumed session's pinned
    # provider, else the configured per-model provider. Inputs: tmp_path.
    provider = OpenAICompatibleProviderConfig(
        name="huggingface",
        models=("zai-org/GLM-5.2",),
        default_model="zai-org/GLM-5.2",
        inference_providers={"zai-org/GLM-5.2": "deepinfra"},
    )
    selection = ProviderSelection(provider=provider, model="zai-org/GLM-5.2")
    record = CodingSessionRecord(
        id="session-id",
        path=tmp_path / "session.jsonl",
        cwd=tmp_path,
        model="zai-org/GLM-5.2",
        title=None,
        created_at=1.0,
        updated_at=1.0,
        provider_name="huggingface",
        inference_provider="fireworks-ai",
    )

    assert tui_app._startup_inference_provider(selection, record) == "fireworks-ai"
    assert tui_app._startup_inference_provider(selection, None) == "deepinfra"


@pytest.mark.anyio
async def test_run_tui_app_falls_back_to_first_credentialed_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC] Async: verifies run_tui_app falls back from an uncredentialed default provider to the first
    # provider with stored credentials, and closes session then provider after the app runs.
    # Inputs: monkeypatch (fakes stores, settings, provider factory, session, app), tmp_path.
    isolate_home(monkeypatch, tmp_path)
    calls: list[str] = []

    class FakeCredentialStore:
        # [CLASS DOC] HELPER: fake credential store with an API key only for "openai" and no OAuth.
        def get(self, name: str) -> str | None:
            # [FUNCTION DOC] HELPER: returns "stored-key" for "openai", else None.
            return "stored-key" if name == "openai" else None

        def get_oauth(self, name: str) -> object | None:
            # [FUNCTION DOC] HELPER: always returns None.
            return None

    record = CodingSessionRecord(
        id="new-session",
        path=tmp_path / "new-session.jsonl",
        cwd=tmp_path,
        model="gpt-5.5",
        title=None,
        created_at=1.0,
        updated_at=1.0,
        provider_name="openai",
    )

    class FakeProvider:
        # [CLASS DOC] HELPER: fake provider recording closure.
        async def aclose(self) -> None:
            # [FUNCTION DOC] HELPER: async; records "provider_closed".
            calls.append("provider_closed")

    class FakeManager:
        # [CLASS DOC] HELPER: fake session manager returning the prepared record and no existing sessions.
        def prepare_session(
            self,
            *,
            cwd: Path,
            model: str,
            provider_name: str | None = None,
        ) -> CodingSessionRecord:
            # [FUNCTION DOC] HELPER: records the prepare call parameters and returns the fixed record.
            calls.append(f"prepare:{cwd}:{model}:{provider_name}")
            return record

        def get_session(self, session_id: str) -> CodingSessionRecord | None:
            # [FUNCTION DOC] HELPER: always returns None.
            return None

    class LoadedSession:
        # [CLASS DOC] HELPER: fake loaded session recording closure.
        async def aclose(self) -> None:
            # [FUNCTION DOC] HELPER: async; records "session_closed".
            calls.append("session_closed")

    class FakeCodingSession:
        # [CLASS DOC] HELPER: fake CodingSession class with an async load factory.
        @classmethod
        async def load(cls, config: object) -> LoadedSession:
            # [FUNCTION DOC] HELPER: async classmethod; asserts the config uses provider "openai", records "load", and
            # returns a LoadedSession.
            assert config.provider_name == "openai"  # type: ignore[attr-defined]
            calls.append("load")
            return LoadedSession()

    class FakeApp:
        # [CLASS DOC] HELPER: fake app class asserting construction arguments and recording run.
        def __init__(self, session: LoadedSession, **kwargs: object) -> None:
            # [FUNCTION DOC] HELPER: asserts the session type and that no startup message was passed.
            assert isinstance(session, LoadedSession)
            assert kwargs["startup_message"] is None

        async def run_async(self) -> None:
            # [FUNCTION DOC] HELPER: async; records "run".
            calls.append("run")

    settings = ProviderSettings(
        default_provider="local",
        providers=(
            OpenAICompatibleProviderConfig(
                name="local",
                base_url="http://localhost:11434/v1",
                api_key_env="LOCAL_API_KEY",
                credential_name=None,
                models=("qwen",),
                default_model="qwen",
            ),
            OpenAICompatibleProviderConfig(
                name="openai",
                credential_name="openai",
                models=("gpt-5.5",),
                default_model="gpt-5.5",
            ),
        ),
    )
    monkeypatch.setattr(tui_app, "FileCredentialStore", lambda: FakeCredentialStore())
    monkeypatch.setattr(tui_app, "load_provider_settings", lambda: settings)
    monkeypatch.setattr(tui_app, "load_tui_settings", lambda: TuiSettings())
    monkeypatch.setattr(
        tui_app,
        "create_model_provider",
        lambda provider, **kwargs: (
            calls.append(f"provider:{provider.name}:{kwargs['model']}") or FakeProvider()
        ),
    )
    monkeypatch.setattr(tui_app, "CodingSession", FakeCodingSession)
    monkeypatch.setattr(tui_app, "TauTuiApp", FakeApp)

    await tui_app.run_tui_app(cwd=tmp_path, model=None, session_manager=FakeManager())

    assert calls == [
        "provider:openai:gpt-5.5",
        f"prepare:{tmp_path}:gpt-5.5:openai",
        "load",
        "run",
        "session_closed",
        "provider_closed",
    ]


@pytest.mark.anyio
async def test_run_tui_app_exits_when_startup_trust_is_cancelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC] Async: verifies that when startup project trust is cancelled, run_tui_app returns None
    # without opening the app and closes session and provider. Inputs: monkeypatch, tmp_path.
    isolate_home(monkeypatch, tmp_path)
    calls: list[str] = []
    record = CodingSessionRecord(
        id="new-session",
        path=tmp_path / "new-session.jsonl",
        cwd=tmp_path,
        model="gpt-5",
        title=None,
        created_at=1.0,
        updated_at=1.0,
        provider_name="openai",
    )

    class FakeProvider:
        # [CLASS DOC] HELPER: fake provider recording closure.
        async def aclose(self) -> None:
            # [FUNCTION DOC] HELPER: async; records "provider_closed".
            calls.append("provider_closed")

    class FakeManager:
        # [CLASS DOC] HELPER: fake session manager recording prepare.
        def prepare_session(
            self,
            *,
            cwd: Path,
            model: str,
            provider_name: str | None = None,
        ) -> CodingSessionRecord:
            # [FUNCTION DOC] HELPER: records "prepare" and returns the fixed record.
            calls.append("prepare")
            return record

        def get_session(self, session_id: str) -> CodingSessionRecord | None:
            # [FUNCTION DOC] HELPER: always returns None.
            return None

    class CancelledSession:
        # [CLASS DOC] HELPER: fake session whose trust resolution is cancelled; records closure.
        project_trust_resolution = SimpleNamespace(cancelled=True)

        async def aclose(self) -> None:
            # [FUNCTION DOC] HELPER: async; records "session_closed".
            calls.append("session_closed")

    class FakeCodingSession:
        # [CLASS DOC] HELPER: fake CodingSession factory.
        @classmethod
        async def load(cls, config: object) -> CancelledSession:
            # [FUNCTION DOC] HELPER: async classmethod; records "load" and returns a CancelledSession.
            calls.append("load")
            return CancelledSession()

    class UnexpectedApp:
        # [CLASS DOC] HELPER: app class that must not be constructed.
        def __init__(self, session: object, **kwargs: object) -> None:
            # [FUNCTION DOC] HELPER: always raises AssertionError (the main TUI must not open).
            raise AssertionError("main TUI must not open after startup trust cancellation")

    settings = ProviderSettings(
        default_provider="openai",
        providers=(
            OpenAICompatibleProviderConfig(
                name="openai",
                models=("gpt-5",),
                default_model="gpt-5",
            ),
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "stored-key")
    monkeypatch.setattr(tui_app, "load_provider_settings", lambda: settings)
    monkeypatch.setattr(tui_app, "create_model_provider", lambda *args, **kwargs: FakeProvider())
    monkeypatch.setattr(tui_app, "CodingSession", FakeCodingSession)
    monkeypatch.setattr(tui_app, "TauTuiApp", UnexpectedApp)

    result = await tui_app.run_tui_app(cwd=tmp_path, model=None, session_manager=FakeManager())

    assert result is None
    assert calls == ["prepare", "load", "session_closed", "provider_closed"]


@pytest.mark.anyio
async def test_run_tui_app_surfaces_startup_provider_error_in_login_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC] Async: regression test verifying a non-auth startup provider error is included in the
    # "Login required" startup message and notices, and resource conflict alerts are passed.
    # Inputs: monkeypatch, tmp_path.
    # Regression: a non-auth RuntimeError during startup provider construction
    # was silently replaced by a generic "Login required" placeholder, hiding the
    # real cause. The TUI must surface the underlying error to the user.
    captured: dict[str, object] = {}

    record = CodingSessionRecord(
        id="new-session",
        path=tmp_path / "new-session.jsonl",
        cwd=tmp_path,
        model="qwen",
        title=None,
        created_at=1.0,
        updated_at=1.0,
        provider_name="local",
    )

    class FakeProvider:
        # [CLASS DOC] HELPER: fake provider with a no-op close.
        async def aclose(self) -> None:
            # [FUNCTION DOC] HELPER: async no-op.
            pass

    class FakeManager:
        # [CLASS DOC] HELPER: fake manager returning the fixed record.
        def prepare_session(
            self,
            *,
            cwd: Path,
            model: str,
            provider_name: str | None = None,
        ) -> CodingSessionRecord:
            # [FUNCTION DOC] HELPER: returns the fixed record.
            return record

        def get_session(self, session_id: str) -> CodingSessionRecord | None:
            # [FUNCTION DOC] HELPER: always returns None.
            return None

    class LoadedSession:
        # [CLASS DOC] HELPER: fake loaded session exposing one skill conflict diagnostic.
        resource_diagnostics = (
            ResourceDiagnostic(
                kind="skill",
                name="review",
                path=tmp_path / ".agents" / "skills" / "review" / "SKILL.md",
                message=(
                    "overrides lower-precedence resource at "
                    f"{tmp_path / '.tau' / 'skills' / 'review' / 'SKILL.md'}"
                ),
            ),
        )

    class FakeCodingSession:
        # [CLASS DOC] HELPER: fake CodingSession factory.
        @classmethod
        async def load(cls, config: object) -> LoadedSession:
            # [FUNCTION DOC] HELPER: async classmethod returning a LoadedSession.
            return LoadedSession()

    class FakeApp:
        # [CLASS DOC] HELPER: fake app capturing startup message, alerts and notices.
        def __init__(self, session: LoadedSession, **kwargs: object) -> None:
            # [FUNCTION DOC] HELPER: stores the startup keyword arguments in the captured mapping.
            captured["startup_message"] = kwargs["startup_message"]
            captured["startup_alerts"] = kwargs["startup_alerts"]
            captured["startup_notices"] = kwargs["startup_notices"]

        async def run_async(self) -> None:
            # [FUNCTION DOC] HELPER: async no-op.
            pass

    settings = ProviderSettings(
        default_provider="local",
        providers=(
            OpenAICompatibleProviderConfig(
                name="local",
                base_url="http://localhost:11434/v1",
                api_key_env="LOCAL_API_KEY",
                credential_name=None,
                models=("qwen",),
                default_model="qwen",
            ),
        ),
    )

    def _boom(provider: object, **kwargs: object) -> object:
        # [FUNCTION DOC] HELPER: provider factory replacement that always raises RuntimeError (backend refused).
        raise RuntimeError("connection to provider backend refused")

    monkeypatch.setattr(tui_app, "load_provider_settings", lambda: settings)
    monkeypatch.setattr(tui_app, "load_tui_settings", lambda: TuiSettings())
    monkeypatch.setattr(tui_app, "create_model_provider", _boom)
    monkeypatch.setattr(tui_app, "LoginRequiredProvider", lambda message: FakeProvider())
    monkeypatch.setattr(tui_app, "CodingSession", FakeCodingSession)
    monkeypatch.setattr(tui_app, "TauTuiApp", FakeApp)

    await tui_app.run_tui_app(cwd=tmp_path, model=None, session_manager=FakeManager())

    startup_message = captured["startup_message"]
    assert "Login required" in startup_message
    assert "connection to provider backend refused" in startup_message
    alerts = captured["startup_alerts"]
    assert len(alerts) == 1
    assert "skill 'review'" in alerts[0]
    notices = captured["startup_notices"]
    assert any("connection to provider backend refused" in n for n in notices)


@pytest.mark.anyio
async def test_run_tui_app_ignores_latest_directory_provider_model_for_new_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC] Async: verifies a new session uses the default provider/model rather than the latest
    # session's provider for the directory. Inputs: monkeypatch, tmp_path.
    # KNOWN ISSUE: load_tui_settings is monkeypatched twice (redundant).
    isolate_home(monkeypatch, tmp_path)
    calls: list[str] = []
    latest_record = CodingSessionRecord(
        id="latest-session",
        path=tmp_path / "latest-session.jsonl",
        cwd=tmp_path / "other",
        model="gpt-5.5",
        title=None,
        created_at=1.0,
        updated_at=1.0,
        provider_name="openai-codex",
    )
    created_record = CodingSessionRecord(
        id="new-session",
        path=tmp_path / "new-session.jsonl",
        cwd=tmp_path,
        model="gpt-5",
        title=None,
        created_at=2.0,
        updated_at=2.0,
        provider_name="openai",
    )

    class FakeProvider:
        # [CLASS DOC] HELPER: fake provider recording closure.
        async def aclose(self) -> None:
            # [FUNCTION DOC] HELPER: async; records "provider_closed".
            calls.append("provider_closed")

    class FakeManager:
        # [CLASS DOC] HELPER: fake manager with a latest session for the directory.
        def latest_session_for_cwd(self, cwd: Path) -> CodingSessionRecord | None:
            # [FUNCTION DOC] HELPER: records the lookup and returns the latest record.
            calls.append(f"latest:{cwd}")
            return latest_record

        def prepare_session(
            self,
            *,
            cwd: Path,
            model: str,
            provider_name: str | None = None,
        ) -> CodingSessionRecord:
            # [FUNCTION DOC] HELPER: records prepare parameters and returns the created record.
            calls.append(f"prepare:{cwd}:{model}:{provider_name}")
            return created_record

        def get_session(self, session_id: str) -> CodingSessionRecord | None:
            # [FUNCTION DOC] HELPER: always returns None.
            return None

    class FakeCodingSession:
        # [CLASS DOC] HELPER: fake CodingSession factory.
        @classmethod
        async def load(cls, config: object) -> str:
            # [FUNCTION DOC] HELPER: async classmethod asserting provider "openai" and model "gpt-5"; records load and
            # returns a placeholder string session.
            assert config.provider_name == "openai"  # type: ignore[attr-defined]
            assert config.model == "gpt-5"  # type: ignore[attr-defined]
            calls.append("load")
            return "session"

    class FakeApp:
        # [CLASS DOC] HELPER: fake app asserting the session and recording run.
        def __init__(self, session: str, **kwargs: object) -> None:
            # [FUNCTION DOC] HELPER: asserts the placeholder session.
            assert session == "session"

        async def run_async(self) -> None:
            # [FUNCTION DOC] HELPER: async; records "run".
            calls.append("run")

    settings = ProviderSettings(
        default_provider="openai",
        providers=(
            OpenAICompatibleProviderConfig(
                name="openai",
                models=("gpt-5",),
                default_model="gpt-5",
            ),
            OpenAICodexProviderConfig(
                name="openai-codex",
                models=("gpt-5.5",),
                default_model="gpt-5.5",
            ),
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "stored-key")
    monkeypatch.setattr(tui_app, "load_provider_settings", lambda: settings)
    monkeypatch.setattr(tui_app, "load_tui_settings", lambda: TuiSettings())
    monkeypatch.setattr(
        tui_app,
        "create_model_provider",
        lambda provider, **kwargs: (
            calls.append(f"provider:{provider.name}:{kwargs['model']}") or FakeProvider()
        ),
    )
    monkeypatch.setattr(tui_app, "CodingSession", FakeCodingSession)
    monkeypatch.setattr(tui_app, "TauTuiApp", FakeApp)
    monkeypatch.setattr(tui_app, "load_tui_settings", lambda: TuiSettings())

    await tui_app.run_tui_app(cwd=tmp_path, model=None, session_manager=FakeManager())

    assert calls == [
        "provider:openai:gpt-5",
        f"prepare:{tmp_path}:gpt-5:openai",
        "load",
        "run",
        "provider_closed",
    ]


@pytest.mark.anyio
async def test_run_tui_app_does_not_start_new_session_from_scoped_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC] Async: verifies a credentialed scoped model does not override the default provider when
    # starting a new session. Inputs: monkeypatch, tmp_path.
    isolate_home(monkeypatch, tmp_path)
    calls: list[str] = []
    latest = CodingSessionRecord(
        id="latest-session",
        path=tmp_path / "latest-session.jsonl",
        cwd=tmp_path,
        model="gpt-5.5",
        title=None,
        created_at=1.0,
        updated_at=1.0,
        provider_name="openai",
    )
    record = CodingSessionRecord(
        id="new-session",
        path=tmp_path / "new-session.jsonl",
        cwd=tmp_path,
        model="gpt-5.5",
        title=None,
        created_at=2.0,
        updated_at=2.0,
        provider_name="openai-codex",
    )

    class FakeProvider:
        # [CLASS DOC] HELPER: fake provider recording closure.
        async def aclose(self) -> None:
            # [FUNCTION DOC] HELPER: async; records "provider_closed".
            calls.append("provider_closed")

    class FakeCredentialStore:
        # [CLASS DOC] HELPER: fake credential store with OAuth only for "openai-codex".
        def get(self, name: str) -> str | None:
            # [FUNCTION DOC] HELPER: always returns None.
            return None

        def get_oauth(self, name: str) -> object | None:
            # [FUNCTION DOC] HELPER: returns a placeholder object for "openai-codex", else None.
            return object() if name == "openai-codex" else None

    class FakeManager:
        # [CLASS DOC] HELPER: fake manager with a latest session.
        def latest_session_for_cwd(self, cwd: Path) -> CodingSessionRecord | None:
            # [FUNCTION DOC] HELPER: records the lookup and returns the latest record.
            calls.append(f"latest:{cwd}")
            return latest

        def prepare_session(
            self,
            *,
            cwd: Path,
            model: str,
            provider_name: str | None = None,
        ) -> CodingSessionRecord:
            # [FUNCTION DOC] HELPER: records prepare parameters and returns the new record.
            calls.append(f"prepare:{cwd}:{model}:{provider_name}")
            return record

        def get_session(self, session_id: str) -> CodingSessionRecord | None:
            # [FUNCTION DOC] HELPER: always returns None.
            return None

    class FakeCodingSession:
        # [CLASS DOC] HELPER: fake CodingSession factory.
        @classmethod
        async def load(cls, config: object) -> str:
            # [FUNCTION DOC] HELPER: async classmethod asserting provider "openai"; records load.
            assert config.provider_name == "openai"  # type: ignore[attr-defined]
            calls.append("load")
            return "session"

    class FakeApp:
        # [CLASS DOC] HELPER: fake app asserting the session and recording run.
        def __init__(self, session: str, **kwargs: object) -> None:
            # [FUNCTION DOC] HELPER: asserts the placeholder session.
            assert session == "session"

        async def run_async(self) -> None:
            # [FUNCTION DOC] HELPER: async; records "run".
            calls.append("run")

    settings = ProviderSettings(
        default_provider="openai",
        providers=(
            OpenAICompatibleProviderConfig(
                name="openai",
                models=("gpt-5.5",),
                default_model="gpt-5.5",
            ),
            OpenAICodexProviderConfig(
                name="openai-codex",
                models=("gpt-5.5",),
                default_model="gpt-5.5",
            ),
        ),
        scoped_models=(ScopedModelConfig(provider="openai-codex", model="gpt-5.5"),),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "stored-key")
    monkeypatch.setattr(tui_app, "FileCredentialStore", lambda: FakeCredentialStore())
    monkeypatch.setattr(tui_app, "load_provider_settings", lambda: settings)
    monkeypatch.setattr(tui_app, "load_tui_settings", lambda: TuiSettings())
    monkeypatch.setattr(
        tui_app,
        "create_model_provider",
        lambda provider, **kwargs: (
            calls.append(f"provider:{provider.name}:{kwargs['model']}") or FakeProvider()
        ),
    )
    monkeypatch.setattr(tui_app, "CodingSession", FakeCodingSession)
    monkeypatch.setattr(tui_app, "TauTuiApp", FakeApp)

    await tui_app.run_tui_app(cwd=tmp_path, model=None, session_manager=FakeManager())

    assert calls == [
        "provider:openai:gpt-5.5",
        f"prepare:{tmp_path}:gpt-5.5:openai",
        "load",
        "run",
        "provider_closed",
    ]


@pytest.mark.anyio
async def test_run_tui_app_creates_new_session_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC] Async: verifies run_tui_app prepares a new session by default (never the implicit default
    # session), passes threshold, persistence and system prompt options to the session config,
    # and passes settings and initial prompt to the app. Inputs: monkeypatch, tmp_path.
    isolate_home(monkeypatch, tmp_path)
    calls: list[str] = []
    record = CodingSessionRecord(
        id="new-session",
        path=tmp_path / "new-session.jsonl",
        cwd=tmp_path,
        model="fake-model",
        title=None,
        created_at=1.0,
        updated_at=1.0,
    )

    class FakeProvider:
        # [CLASS DOC] HELPER: fake provider recording closure.
        async def aclose(self) -> None:
            # [FUNCTION DOC] HELPER: async; records "provider_closed".
            calls.append("provider_closed")

    class FakeManager:
        # [CLASS DOC] HELPER: fake manager recording prepare and get calls.
        def prepare_session(
            self,
            *,
            cwd: Path,
            model: str,
            provider_name: str | None = None,
        ) -> CodingSessionRecord:
            # [FUNCTION DOC] HELPER: records prepare parameters and returns the record.
            calls.append(f"prepare:{cwd}:{model}:{provider_name}")
            return record

        def get_session(self, session_id: str) -> CodingSessionRecord | None:
            # [FUNCTION DOC] HELPER: records the lookup and returns None.
            calls.append(f"get:{session_id}")
            return None

        def get_or_create_default_session(self, *, cwd: Path, model: str) -> CodingSessionRecord:
            # [FUNCTION DOC] HELPER: always raises AssertionError (default session must not be opened).
            raise AssertionError("default session should not be opened implicitly")

    class FakeCodingSession:
        # [CLASS DOC] HELPER: fake CodingSession factory.
        @classmethod
        async def load(cls, config: object) -> str:
            # [FUNCTION DOC] HELPER: async classmethod asserting the config fields; records load.
            assert config.provider_name == "local"  # type: ignore[attr-defined]
            assert config.auto_compact_token_threshold == 1000  # type: ignore[attr-defined]
            assert config.index_on_first_persist is True  # type: ignore[attr-defined]
            assert config.system is None  # type: ignore[attr-defined]
            assert config.custom_system_prompt == "Custom base"  # type: ignore[attr-defined]
            assert config.append_system_prompt == "First\n\nSecond"  # type: ignore[attr-defined]
            calls.append("load")
            return "session"

    class FakeApp:
        # [CLASS DOC] HELPER: fake app asserting the session, settings type and initial prompt; records run.
        def __init__(self, session: str, **kwargs: object) -> None:
            # [FUNCTION DOC] HELPER: asserts the placeholder session, a TuiSettings instance and the initial prompt.
            assert session == "session"
            assert isinstance(kwargs["tui_settings"], TuiSettings)
            assert kwargs["initial_prompt"] == "explain this repo"

        async def run_async(self) -> None:
            # [FUNCTION DOC] HELPER: async; records "run".
            calls.append("run")

    settings = ProviderSettings(
        default_provider="local",
        providers=(
            OpenAICompatibleProviderConfig(
                name="local",
                base_url="http://localhost:11434/v1",
                api_key_env="LOCAL_API_KEY",
                models=("local-model",),
                default_model="local-model",
            ),
        ),
    )
    monkeypatch.setattr(tui_app, "load_provider_settings", lambda: settings)
    monkeypatch.setattr(
        tui_app,
        "create_model_provider",
        lambda provider, **kwargs: FakeProvider(),
    )
    monkeypatch.setattr(tui_app, "CodingSession", FakeCodingSession)
    monkeypatch.setattr(tui_app, "TauTuiApp", FakeApp)
    monkeypatch.setattr(tui_app, "load_tui_settings", lambda: TuiSettings())

    await tui_app.run_tui_app(
        model=None,
        cwd=tmp_path,
        provider_name="local",
        auto_compact_token_threshold=1000,
        initial_prompt="explain this repo",
        session_manager=FakeManager(),
        custom_system_prompt="Custom base",
        append_system_prompt="First\n\nSecond",
    )

    assert calls == [
        f"prepare:{tmp_path}:local-model:local",
        "get:new-session",
        "load",
        "run",
        "provider_closed",
    ]


@pytest.mark.anyio
async def test_run_tui_app_returns_session_id_when_persisted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC] Async: verifies run_tui_app returns None when the new session was never persisted and the
    # session id once the manager reports it persisted. Inputs: monkeypatch, tmp_path.
    isolate_home(monkeypatch, tmp_path)
    record = CodingSessionRecord(
        id="new-session",
        path=tmp_path / "new-session.jsonl",
        cwd=tmp_path,
        model="fake-model",
        title=None,
        created_at=1.0,
        updated_at=1.0,
    )

    class FakeProvider:
        # [CLASS DOC] HELPER: fake provider with a no-op close.
        async def aclose(self) -> None:
            # [FUNCTION DOC] HELPER: async no-op.
            return None

    class FakeManager:
        # [CLASS DOC] HELPER: fake manager whose get_session depends on a mutable persisted flag.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: initializes persisted to False.
            self.persisted = False

        def prepare_session(
            self,
            *,
            cwd: Path,
            model: str,
            provider_name: str | None = None,
        ) -> CodingSessionRecord:
            # [FUNCTION DOC] HELPER: returns the fixed record.
            return record

        def get_session(self, session_id: str) -> CodingSessionRecord | None:
            # [FUNCTION DOC] HELPER: returns the record only when persisted is True, else None.
            return record if self.persisted else None

    class FakeSession:
        # [CLASS DOC] HELPER: minimal loaded session with a fixed session_id and no-op close.
        session_id = "new-session"

        async def aclose(self) -> None:
            # [FUNCTION DOC] HELPER: async no-op.
            return None

    class FakeCodingSession:
        # [CLASS DOC] HELPER: fake CodingSession factory.
        @classmethod
        async def load(cls, config: object) -> FakeSession:
            # [FUNCTION DOC] HELPER: async classmethod returning a new minimal session.
            return FakeSession()

    class FakeApp:
        # [CLASS DOC] HELPER: fake app that does nothing.
        def __init__(self, session: FakeSession, **kwargs: object) -> None:
            # [FUNCTION DOC] HELPER: accepts and ignores constructor arguments.
            pass

        async def run_async(self) -> None:
            # [FUNCTION DOC] HELPER: async no-op.
            return None

    settings = ProviderSettings(
        default_provider="local",
        providers=(
            OpenAICompatibleProviderConfig(
                name="local",
                base_url="http://localhost:11434/v1",
                api_key_env="LOCAL_API_KEY",
                models=("local-model",),
                default_model="local-model",
            ),
        ),
    )
    monkeypatch.setattr(tui_app, "load_provider_settings", lambda: settings)
    monkeypatch.setattr(
        tui_app,
        "create_model_provider",
        lambda provider, **kwargs: FakeProvider(),
    )
    monkeypatch.setattr(tui_app, "CodingSession", FakeCodingSession)
    monkeypatch.setattr(tui_app, "TauTuiApp", FakeApp)
    monkeypatch.setattr(tui_app, "load_tui_settings", lambda: TuiSettings())

    manager = FakeManager()
    manager.persisted = False
    result = await tui_app.run_tui_app(
        model=None,
        cwd=tmp_path,
        provider_name="local",
        session_manager=manager,
    )
    assert result is None

    manager.persisted = True
    result = await tui_app.run_tui_app(
        model=None,
        cwd=tmp_path,
        provider_name="local",
        session_manager=manager,
    )
    assert result == "new-session"


@pytest.mark.anyio
async def test_run_tui_app_opens_when_provider_login_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC] Async: verifies the TUI still opens with a login-required placeholder provider when no
    # provider is usable; the startup message suggests /login without leaking env var names and
    # notices list the provider error first, then the update notice. Inputs: monkeypatch, tmp_path.
    isolate_home(monkeypatch, tmp_path)
    calls: list[str] = []
    record = CodingSessionRecord(
        id="new-session",
        path=tmp_path / "new-session.jsonl",
        cwd=tmp_path,
        model="fake-model",
        title=None,
        created_at=1.0,
        updated_at=1.0,
    )

    class FakeManager:
        # [CLASS DOC] HELPER: fake manager recording prepare.
        def prepare_session(
            self,
            *,
            cwd: Path,
            model: str,
            provider_name: str | None = None,
        ) -> CodingSessionRecord:
            # [FUNCTION DOC] HELPER: records prepare parameters and returns the record.
            calls.append(f"prepare:{cwd}:{model}:{provider_name}")
            return record

        def get_session(self, session_id: str) -> CodingSessionRecord | None:
            # [FUNCTION DOC] HELPER: always returns None.
            return None

    class FakeCodingSession:
        # [CLASS DOC] HELPER: fake CodingSession factory.
        @classmethod
        async def load(cls, config: object) -> str:
            # [FUNCTION DOC] HELPER: async classmethod recording the provider class name; returns a placeholder session.
            calls.append(f"load:{type(config.provider).__name__}")  # type: ignore[attr-defined]
            return "session"

    class FakeApp:
        # [CLASS DOC] HELPER: fake app asserting the startup message and notices contents.
        def __init__(self, session: str, **kwargs: object) -> None:
            # [FUNCTION DOC] HELPER: asserts message and notices as described for the enclosing test.
            assert session == "session"
            message = str(kwargs["startup_message"])
            assert "Tau 0.2.0 is available" not in message
            notices = kwargs["startup_notices"]
            # The startup provider error is surfaced first, then the update notice.
            assert any("Startup provider creation failed" in n for n in notices)
            assert "Missing provider API key." in notices[0]
            assert "Tau 0.2.0 is available" in notices
            assert "Login required. Run /login" in message
            assert "/login openai" in message
            assert "OPENAI_API_KEY" not in message
            assert "environment variable" not in message

        async def run_async(self) -> None:
            # [FUNCTION DOC] HELPER: async; records "run".
            calls.append("run")

    monkeypatch.setattr(tui_app, "load_provider_settings", lambda: ProviderSettings())
    monkeypatch.setattr(tui_app, "provider_has_usable_credentials", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        tui_app,
        "create_model_provider",
        lambda provider, **kwargs: (_ for _ in ()).throw(RuntimeError("Missing provider API key.")),
    )
    monkeypatch.setattr(tui_app, "CodingSession", FakeCodingSession)
    monkeypatch.setattr(tui_app, "TauTuiApp", FakeApp)
    monkeypatch.setattr(tui_app, "load_tui_settings", lambda: TuiSettings())

    await tui_app.run_tui_app(
        cwd=tmp_path,
        model=None,
        session_manager=FakeManager(),
        startup_notice="Tau 0.2.0 is available",
    )

    assert calls == [f"prepare:{tmp_path}:gpt-5.4:openai", "load:LoginRequiredProvider", "run"]


@pytest.mark.anyio
async def test_run_tui_app_resumes_explicit_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC] Async: verifies an explicit session id resumes that session using its recorded provider
    # and model (not creating a new one) and passes system prompt options. Inputs: monkeypatch, tmp_path.
    isolate_home(monkeypatch, tmp_path)
    calls: list[str] = []
    record = CodingSessionRecord(
        id="session-1",
        path=tmp_path / "session-1.jsonl",
        cwd=tmp_path,
        model="fake-model",
        title=None,
        created_at=1.0,
        updated_at=1.0,
        provider_name="local",
    )

    class FakeProvider:
        # [CLASS DOC] HELPER: fake provider recording closure.
        async def aclose(self) -> None:
            # [FUNCTION DOC] HELPER: async; records "provider_closed".
            calls.append("provider_closed")

    class FakeManager:
        # [CLASS DOC] HELPER: fake manager that refuses creation and returns the stored record.
        def create_session(
            self,
            *,
            cwd: Path,
            model: str,
            provider_name: str | None = None,
        ) -> CodingSessionRecord:
            # [FUNCTION DOC] HELPER: always raises AssertionError (resume must not create a session).
            raise AssertionError("explicit resume should not create a new session")

        def get_session(self, session_id: str) -> CodingSessionRecord | None:
            # [FUNCTION DOC] HELPER: records the lookup and returns the record.
            calls.append(f"get:{session_id}")
            return record

    class FakeCodingSession:
        # [CLASS DOC] HELPER: fake CodingSession factory.
        @classmethod
        async def load(cls, config: object) -> str:
            # [FUNCTION DOC] HELPER: async classmethod asserting provider, model and system prompt fields; records load.
            assert config.provider_name == "local"  # type: ignore[attr-defined]
            assert config.model == "fake-model"  # type: ignore[attr-defined]
            assert config.system is None  # type: ignore[attr-defined]
            assert config.custom_system_prompt == "Resume base"  # type: ignore[attr-defined]
            assert config.append_system_prompt == "Resume append"  # type: ignore[attr-defined]
            calls.append("load")
            return "session"

    class FakeApp:
        # [CLASS DOC] HELPER: fake app asserting session and settings; records run.
        def __init__(self, session: str, **kwargs: object) -> None:
            # [FUNCTION DOC] HELPER: asserts the placeholder session and a TuiSettings instance.
            assert session == "session"
            assert isinstance(kwargs["tui_settings"], TuiSettings)

        async def run_async(self) -> None:
            # [FUNCTION DOC] HELPER: async; records "run".
            calls.append("run")

    settings = ProviderSettings(
        default_provider="openai",
        providers=(
            OpenAICompatibleProviderConfig(
                name="openai",
                models=("gpt-5.5",),
                default_model="gpt-5.5",
            ),
            OpenAICompatibleProviderConfig(
                name="local",
                base_url="http://localhost:11434/v1",
                api_key_env="LOCAL_API_KEY",
                models=("fake-model",),
                default_model="fake-model",
            ),
        ),
    )
    monkeypatch.setattr(tui_app, "load_provider_settings", lambda: settings)
    monkeypatch.setattr(
        tui_app,
        "create_model_provider",
        lambda provider, **kwargs: (
            calls.append(f"provider:{provider.name}:{kwargs['model']}") or FakeProvider()
        ),
    )
    monkeypatch.setattr(tui_app, "CodingSession", FakeCodingSession)
    monkeypatch.setattr(tui_app, "TauTuiApp", FakeApp)
    monkeypatch.setattr(tui_app, "load_tui_settings", lambda: TuiSettings())

    await tui_app.run_tui_app(
        model=None,
        cwd=tmp_path,
        session_id="session-1",
        session_manager=FakeManager(),
        custom_system_prompt="Resume base",
        append_system_prompt="Resume append",
    )

    assert calls == [
        "get:session-1",
        "provider:local:fake-model",
        "load",
        "run",
        "provider_closed",
    ]


@pytest.mark.anyio
async def test_run_tui_app_ignores_uncredentialed_provider_when_matching_resume_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC] Async: verifies resuming a session without a recorded provider skips an uncredentialed
    # provider offering the same model and picks the credentialed one. Inputs: monkeypatch, tmp_path.
    isolate_home(monkeypatch, tmp_path)
    calls: list[str] = []
    record = CodingSessionRecord(
        id="session-1",
        path=tmp_path / "session-1.jsonl",
        cwd=tmp_path,
        model="shared-model",
        title=None,
        created_at=1.0,
        updated_at=1.0,
    )

    class FakeCredentialStore:
        # [CLASS DOC] HELPER: fake credential store with an API key only for "openai".
        def get(self, name: str) -> str | None:
            # [FUNCTION DOC] HELPER: returns "stored-key" for "openai", else None.
            return "stored-key" if name == "openai" else None

        def get_oauth(self, name: str) -> object | None:
            # [FUNCTION DOC] HELPER: always returns None.
            return None

    class FakeProvider:
        # [CLASS DOC] HELPER: fake provider recording closure.
        async def aclose(self) -> None:
            # [FUNCTION DOC] HELPER: async; records "provider_closed".
            calls.append("provider_closed")

    class FakeManager:
        # [CLASS DOC] HELPER: fake manager returning the stored record.
        def get_session(self, session_id: str) -> CodingSessionRecord | None:
            # [FUNCTION DOC] HELPER: records the lookup and returns the record.
            calls.append(f"get:{session_id}")
            return record

    class FakeCodingSession:
        # [CLASS DOC] HELPER: fake CodingSession factory.
        @classmethod
        async def load(cls, config: object) -> str:
            # [FUNCTION DOC] HELPER: async classmethod asserting provider "openai"; records load.
            assert config.provider_name == "openai"  # type: ignore[attr-defined]
            calls.append("load")
            return "session"

    class FakeApp:
        # [CLASS DOC] HELPER: fake app asserting the session; records run.
        def __init__(self, session: str, **kwargs: object) -> None:
            # [FUNCTION DOC] HELPER: asserts the placeholder session.
            assert session == "session"

        async def run_async(self) -> None:
            # [FUNCTION DOC] HELPER: async; records "run".
            calls.append("run")

    settings = ProviderSettings(
        default_provider="openai",
        providers=(
            OpenAICompatibleProviderConfig(
                name="local",
                base_url="http://localhost:11434/v1",
                api_key_env="LOCAL_API_KEY",
                credential_name=None,
                models=("shared-model",),
                default_model="shared-model",
            ),
            OpenAICompatibleProviderConfig(
                name="openai",
                credential_name="openai",
                models=("shared-model",),
                default_model="shared-model",
            ),
        ),
    )
    monkeypatch.delenv("LOCAL_API_KEY", raising=False)
    monkeypatch.setattr(tui_app, "FileCredentialStore", lambda: FakeCredentialStore())
    monkeypatch.setattr(tui_app, "load_provider_settings", lambda: settings)
    monkeypatch.setattr(tui_app, "load_tui_settings", lambda: TuiSettings())
    monkeypatch.setattr(
        tui_app,
        "create_model_provider",
        lambda provider, **kwargs: (
            calls.append(f"provider:{provider.name}:{kwargs['model']}") or FakeProvider()
        ),
    )
    monkeypatch.setattr(tui_app, "CodingSession", FakeCodingSession)
    monkeypatch.setattr(tui_app, "TauTuiApp", FakeApp)

    await tui_app.run_tui_app(
        model=None,
        cwd=tmp_path,
        session_id="session-1",
        session_manager=FakeManager(),
    )

    assert calls == [
        "get:session-1",
        "provider:openai:shared-model",
        "load",
        "run",
        "provider_closed",
    ]


class _FakeSessionManager:
    # [CLASS DOC] HELPER: in-memory session manager for picker tests. Holds active and archived record
    # lists and records every archive/restore request; archiving or restoring moves matching
    # records between the two lists (projects matched by resolved cwd).
    def __init__(
        self,
        records: list[CodingSessionRecord],
        archived_records: list[CodingSessionRecord] | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: stores the active records (same list object, mutated later) and archived records
        # (new empty list if none), and initializes request recording lists.
        self._records = records
        self._archived_records = archived_records or []
        self.archived_session_ids: list[str] = []
        self.archived_project_cwds: list[Path] = []
        self.restored_session_ids: list[str] = []
        self.restored_project_cwds: list[Path] = []

    def list_sessions(self, cwd: Path | None = None) -> list[CodingSessionRecord]:
        # [FUNCTION DOC] HELPER: returns all active records regardless of cwd.
        del cwd
        return self._records

    def list_archived_sessions(self, cwd: Path | None = None) -> list[CodingSessionRecord]:
        # [FUNCTION DOC] HELPER: returns all archived records regardless of cwd.
        del cwd
        return self._archived_records

    def archive_session(self, session_id: str) -> bool:
        # [FUNCTION DOC] HELPER: records the id, moves matching records to the archive; returns whether any moved.
        self.archived_session_ids.append(session_id)
        moved = [record for record in self._records if record.id == session_id]
        self._records = [record for record in self._records if record.id != session_id]
        self._archived_records.extend(moved)
        return bool(moved)

    def archive_project(self, cwd: Path) -> bool:
        # [FUNCTION DOC] HELPER: records the cwd, moves all records whose resolved cwd matches to the archive;
        # returns whether the active list shrank. Touches the filesystem only via path resolution.
        resolved = Path(cwd).resolve()
        self.archived_project_cwds.append(Path(cwd))
        before = len(self._records)
        moved = [record for record in self._records if Path(record.cwd).resolve() == resolved]
        self._records = [
            record for record in self._records if Path(record.cwd).resolve() != resolved
        ]
        self._archived_records.extend(moved)
        return len(self._records) != before

    def unarchive_session(self, session_id: str) -> bool:
        # [FUNCTION DOC] HELPER: records the id, moves matching archived records back; returns whether any moved.
        self.restored_session_ids.append(session_id)
        moved = [record for record in self._archived_records if record.id == session_id]
        self._archived_records = [
            record for record in self._archived_records if record.id != session_id
        ]
        self._records.extend(moved)
        return bool(moved)

    def unarchive_project(self, cwd: Path) -> bool:
        # [FUNCTION DOC] HELPER: records the cwd, moves archived records for that project back; returns whether any moved.
        resolved = Path(cwd).resolve()
        self.restored_project_cwds.append(Path(cwd))
        moved = [
            record for record in self._archived_records if Path(record.cwd).resolve() == resolved
        ]
        self._archived_records = [
            record for record in self._archived_records if Path(record.cwd).resolve() != resolved
        ]
        self._records.extend(moved)
        return bool(moved)


# --- component seam pilot tests ---------------------------------------------
# These drive the generic widget-hosting seam on the real TauTuiApp via the
# host bridge, using an in-test caller with no subagents vocabulary. The legacy
# agents-strip pilot tests they once coexisted with were deleted in Step 3
# (the transcript-source seam and host agent UI left core).


def _component_bridge(app: TauTuiApp) -> _TuiExtensionUiBridge:
    # [FUNCTION DOC] HELPER: returns an extension UI bridge bound to the given running app.
    """Return a host component bridge bound to a running app."""
    return _TuiExtensionUiBridge(app)


class _CrashOnRender(Static):
    # [CLASS DOC] HELPER: Static widget whose render always raises, to test render-crash quarantine.
    def render(self):  # noqa: ANN201
        # [FUNCTION DOC] HELPER: always raises RuntimeError("boom-in-render").
        raise RuntimeError("boom-in-render")


class _CrashOnMount(Static):
    # [CLASS DOC] HELPER: Static widget whose mount handler always raises, to test mount-crash quarantine.
    def on_mount(self) -> None:
        # [FUNCTION DOC] HELPER: always raises RuntimeError("boom-in-on-mount").
        raise RuntimeError("boom-in-on-mount")


@pytest.mark.anyio
async def test_component_slot_widget_mounts_and_unmounts() -> None:
    # [FUNCTION DOC] Async: verifies an extension slot widget mounts into the below-prompt slot and setting it
    # to None unmounts and unregisters it.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        bridge.set_slot_widget(
            "demo",
            lambda theme: Static("slot-content", id="ext-slot-demo"),
            placement="below_prompt",
        )
        await pilot.pause()

        slot = app.query_one("#below-prompt-slot", Container)
        assert slot.query("#ext-slot-demo")
        assert "demo" in app._extension_slot_widgets

        bridge.set_slot_widget("demo", None, placement="below_prompt")
        await pilot.pause()
        assert not slot.query("#ext-slot-demo")
        assert "demo" not in app._extension_slot_widgets


@pytest.mark.anyio
async def test_component_slot_widget_replace_reregisters() -> None:
    # [FUNCTION DOC] Async: verifies re-setting a slot key replaces the old widget, keeping one registration.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        bridge.set_slot_widget(
            "k", lambda theme: Static("one", id="ext-one"), placement="below_prompt"
        )
        await pilot.pause()
        bridge.set_slot_widget(
            "k", lambda theme: Static("two", id="ext-two"), placement="below_prompt"
        )
        await pilot.pause()

        slot = app.query_one("#below-prompt-slot", Container)
        assert not slot.query("#ext-one")
        assert slot.query("#ext-two")
        assert len(app._extension_slot_widgets) == 1


@pytest.mark.anyio
async def test_component_slot_widget_strings_mount_and_render() -> None:
    # [FUNCTION DOC] Async: verifies a list of strings mounts as a host-built Static showing every line.
    """A plain list of lines mounts as a Static the host builds (no widget defined)."""
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        bridge.set_slot_widget("lines", ["hello", "world"], placement="below_prompt")
        await pilot.pause()

        slot = app.query_one("#below-prompt-slot", Container)
        statics = slot.query(Static)
        assert statics
        text = statics.first().render().plain
        assert "hello" in text
        assert "world" in text
        assert "lines" in app._extension_slot_widgets


@pytest.mark.anyio
async def test_component_slot_widget_strings_replace_updates_content() -> None:
    # [FUNCTION DOC] Async: verifies replacing string lines updates the single Static's content.
    """Re-setting a key with different lines replaces the rendered content."""
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        bridge.set_slot_widget("lines", ["first"], placement="below_prompt")
        await pilot.pause()
        bridge.set_slot_widget("lines", ["second"], placement="below_prompt")
        await pilot.pause()

        slot = app.query_one("#below-prompt-slot", Container)
        statics = slot.query(Static)
        assert len(statics) == 1
        text = statics.first().render().plain
        assert "second" in text
        assert "first" not in text


@pytest.mark.anyio
async def test_component_slot_widget_strings_none_unmounts() -> None:
    # [FUNCTION DOC] Async: verifies setting a string slot to None unmounts it and forgets the key.
    """Setting a string slot to None unmounts and forgets the key."""
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        bridge.set_slot_widget("lines", ["visible"], placement="below_prompt")
        await pilot.pause()
        slot = app.query_one("#below-prompt-slot", Container)
        assert slot.query(Static)

        bridge.set_slot_widget("lines", None, placement="below_prompt")
        await pilot.pause()
        assert not slot.query(Static)
        assert "lines" not in app._extension_slot_widgets


@pytest.mark.anyio
async def test_component_slot_widget_strings_malformed_markup_falls_back() -> None:
    # [FUNCTION DOC] Async: verifies malformed markup in string lines renders literally without failure reports.
    """Malformed Rich markup in a string widget renders literally, never crashes."""
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        # An unclosed tag is invalid Rich markup; from_markup would raise.
        bridge.set_slot_widget("bad", ["[not-a-tag"], placement="below_prompt")
        await pilot.pause()

        assert app.is_running
        slot = app.query_one("#below-prompt-slot", Container)
        statics = slot.query(Static)
        assert statics
        text = statics.first().render().plain
        assert "[not-a-tag" in text
        assert app._extension_component_failures_reported == set()


@pytest.mark.anyio
async def test_component_slot_widget_default_placement_is_above_prompt() -> None:
    # [FUNCTION DOC] Async: verifies the default slot placement is above the prompt.
    """With no explicit placement, a slot widget mounts into #above-prompt-slot."""
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        bridge.set_slot_widget("k", lambda theme: Static("x", id="ext-default"))
        await pilot.pause()

        assert app.query_one("#above-prompt-slot", Container).query("#ext-default")
        assert not app.query_one("#below-prompt-slot", Container).query("#ext-default")


@pytest.mark.anyio
async def test_component_main_view_open_and_close_restores_transcript() -> None:
    # [FUNCTION DOC] Async: verifies opening a main view hides the transcript and closing restores it and
    # refocuses the prompt.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        handle = bridge.open_main_view(
            lambda handle, theme: Static("main-view", id="ext-main-view")
        )
        await pilot.pause()

        assert handle.is_open
        assert app.query_one("#main-slot", Container).display
        assert not app.query_one("#transcript", TranscriptView).display
        assert app.query("#ext-main-view")

        handle.close()
        await pilot.pause()

        assert not handle.is_open
        assert not app.query_one("#main-slot", Container).display
        assert app.query_one("#transcript", TranscriptView).display
        assert not app.query("#ext-main-view")
        assert app.query_one("#prompt", PromptInput).has_focus


@pytest.mark.anyio
async def test_click_does_not_steal_focus_while_main_view_open() -> None:
    # [FUNCTION DOC] Async: verifies clicks do not move focus back to the prompt while a main view is open.
    # The app-level click handler refocuses the prompt after main-TUI clicks,
    # but an open extension main view owns the keyboard: yanking focus back
    # would silently reroute esc/toggles/typed text to the main chat.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        view = Static("main-view", id="ext-main-view")
        view.can_focus = True
        bridge.open_main_view(lambda handle, theme: view)
        await pilot.pause()
        view.focus()
        await pilot.pause()
        assert app.screen.focused is view

        await pilot.click("#ext-main-view")
        await pilot.pause()

        assert app.screen.focused is view
        assert not app.query_one("#prompt", PromptInput).has_focus


@pytest.mark.anyio
async def test_component_main_view_close_result_resolves_wait() -> None:
    # [FUNCTION DOC] Async: verifies close(result) resolves wait() with that result, and later closes are ignored.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        handle = bridge.open_main_view(
            lambda handle, theme: Static("main-view", id="ext-main-view")
        )
        await pilot.pause()

        sentinel = object()
        handle.close(sentinel)
        # wait() may be awaited after close already happened: returns at once.
        assert await handle.wait() is sentinel
        # Idempotent: a later close does not overwrite the first result.
        handle.close("ignored")
        assert await handle.wait() is sentinel


@pytest.mark.anyio
async def test_component_main_view_close_without_result_resolves_none() -> None:
    # [FUNCTION DOC] Async: verifies close() without a result resolves wait() with None.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        handle = bridge.open_main_view(
            lambda handle, theme: Static("main-view", id="ext-main-view")
        )
        await pilot.pause()

        handle.close()
        assert await handle.wait() is None


@pytest.mark.anyio
async def test_component_main_view_superseded_resolves_first_wait_with_none() -> None:
    # [FUNCTION DOC] Async: verifies opening a second main view closes the first, resolving its wait with None.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        first = bridge.open_main_view(lambda h, theme: Static("one", id="ext-one"))
        await pilot.pause()
        assert first.is_open

        # Opening a second view supersedes (last writer wins) the first.
        second = bridge.open_main_view(lambda h, theme: Static("two", id="ext-two"))
        await pilot.pause()

        assert not first.is_open
        assert await first.wait() is None
        assert second.is_open


@pytest.mark.anyio
async def test_component_main_view_rebind_resolves_pending_wait_with_none() -> None:
    # [FUNCTION DOC] Async: verifies clearing extension components resolves a pending wait() with None.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        handle = bridge.open_main_view(lambda h, theme: Static("v", id="ext-view"))
        await pilot.pause()

        # Start awaiting before the teardown so a leaked future would hang here.
        waiter = asyncio.ensure_future(handle.wait())
        await pilot.pause()
        assert not waiter.done()

        app._clear_extension_components()
        await pilot.pause()

        assert not handle.is_open
        assert await asyncio.wait_for(waiter, timeout=1.0) is None


@pytest.mark.anyio
async def test_component_key_interceptor_consumes_and_gates_on_prompt_text() -> None:
    # [FUNCTION DOC] Async: verifies a key interceptor sees keys with the prompt text and can consume keys
    # only when it returns True (here "j" on an empty prompt).
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)
        seen: list[tuple[str, str]] = []

        def interceptor(event, text: str) -> bool:  # noqa: ANN001
            # [FUNCTION DOC] HELPER: records (key, prompt text) and consumes only "j" when the prompt is empty.
            seen.append((event.key, text))
            return event.key == "j" and text == ""

        bridge.register_key_interceptor(interceptor)

        prompt = app.query_one("#prompt", PromptInput)
        prompt.focus()
        await pilot.pause()

        # Empty prompt: interceptor consumes "j" (nothing typed).
        await pilot.press("j")
        assert prompt.text == ""
        # Non-empty prompt: interceptor declines, so "j" types normally.
        await pilot.press("a")
        await pilot.press("j")
        assert prompt.text == "aj"
        assert ("j", "") in seen


@pytest.mark.anyio
async def test_component_consumed_escape_preempts_cancel() -> None:
    # [FUNCTION DOC] Async: verifies a consuming interceptor prevents escape from cancelling, and after
    # unsubscribing escape cancels again.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)
        cancels: list[int] = []
        app.action_cancel = lambda: cancels.append(1)  # type: ignore[method-assign]

        unsubscribe = bridge.register_key_interceptor(lambda event, text: event.key == "escape")
        app.query_one("#prompt", PromptInput).focus()
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()
        assert cancels == []  # interceptor consumed escape before action_cancel

        # With no interceptor consuming it, escape falls through to cancel.
        unsubscribe()
        await pilot.press("escape")
        await pilot.pause()
        assert cancels == [1]


@pytest.mark.anyio
async def test_component_key_interceptor_failure_degrades_to_typing() -> None:
    # [FUNCTION DOC] Async: verifies a raising interceptor never blocks typing and is reported once per interceptor.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        def boom(event, text: str) -> bool:  # noqa: ANN001
            # [FUNCTION DOC] HELPER: interceptor that always raises RuntimeError.
            raise RuntimeError("interceptor exploded")

        bridge.register_key_interceptor(boom)
        prompt = app.query_one("#prompt", PromptInput)
        prompt.focus()
        await pilot.pause()

        await pilot.press("z")
        assert prompt.text == "z"  # broken interceptor never blocks typing
        assert app.is_running
        # Failures are keyed per-interceptor now (so a second faulty handler
        # still gets diagnosed) and notify like the other failure classes.
        assert any(
            key.startswith("key_interceptor:") for key in app._extension_component_failures_reported
        )


@pytest.mark.anyio
async def test_component_key_interceptor_preempts_priority_binding() -> None:
    # [FUNCTION DOC] Async: verifies interceptors run before app-level priority bindings such as down.
    """A key bound with priority=True on the app (down) is still interceptable.

    down/up/tab/alt+enter are app-level priority bindings, which Textual checks
    before forwarding a key to the focused widget. The interceptor now runs in
    TauTuiApp.on_event, ahead of that priority check, so it can own those keys.
    """
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)
        completion_next: list[int] = []
        app.action_completion_next = lambda: completion_next.append(1)  # type: ignore[method-assign]

        seen: list[str] = []

        def interceptor(event, text: str) -> bool:  # noqa: ANN001
            # [FUNCTION DOC] HELPER: records keys and consumes only "down".
            seen.append(event.key)
            return event.key == "down"

        bridge.register_key_interceptor(interceptor)
        app.query_one("#prompt", PromptInput).focus()
        await pilot.pause()

        await pilot.press("down")
        await pilot.pause()

        assert "down" in seen  # interceptor saw the priority-bound key
        assert completion_next == []  # completion_next preempted, not run


@pytest.mark.anyio
async def test_component_key_interceptor_skipped_while_modal_open() -> None:
    # [FUNCTION DOC] Async: verifies interceptors are not consulted while a modal screen is on top.
    """Interceptors see main-screen keys only; a modal on top is never intercepted."""
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)
        seen: list[str] = []

        # Consume everything — so if this fired while the modal is up, the modal
        # could never be dismissed.
        bridge.register_key_interceptor(lambda event, text: (seen.append(event.key), True)[1])

        confirm_task = asyncio.ensure_future(bridge.confirm("Ship?", "to prod"))
        await pilot.pause()
        assert isinstance(app.screen, ExtensionConfirmScreen)

        # A key while the modal is on top must NOT reach the interceptor.
        await pilot.press("j")
        await pilot.pause()
        assert "j" not in seen

        # And the modal still handles its own keys (escape dismisses -> False).
        await pilot.press("escape")
        assert await confirm_task is False
        assert "escape" not in seen


@pytest.mark.anyio
async def test_component_interceptor_never_consumes_reserved_interrupt_keys() -> None:
    # [FUNCTION DOC] Async: verifies reserved keys (ctrl+c, ctrl+d) bypass even a greedy interceptor so clear
    # and quit still work, while escape remains interceptable.
    """The hard interrupt/exit keys bypass the interceptor entirely.

    A buggy interceptor that returns True for everything must not be able to
    swallow ctrl+c/ctrl+d and brick the session: those keys are skipped before
    the consult (never reach the interceptor) and flow to normal dispatch, so
    the app's escape hatches always fire. Everything else (e.g. escape) stays
    interceptable.
    """
    assert "ctrl+c" in RESERVED_EXTENSION_INTERCEPTOR_KEYS
    assert "ctrl+d" in RESERVED_EXTENSION_INTERCEPTOR_KEYS
    assert "escape" not in RESERVED_EXTENSION_INTERCEPTOR_KEYS

    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)
        seen: list[str] = []
        quits: list[int] = []

        async def fake_quit() -> None:  # don't actually tear down the pilot
            # [FUNCTION DOC] HELPER: async replacement quit that records the call instead of exiting.
            quits.append(1)

        app.action_quit = fake_quit  # type: ignore[method-assign]

        # Greedy interceptor: consumes literally every key it is consulted for.
        bridge.register_key_interceptor(lambda event, text: (seen.append(event.key), True)[1])
        prompt = app.query_one("#prompt", PromptInput)
        prompt.focus()
        await pilot.pause()

        # ctrl+c (SIGINT/interrupt reflex, bound to clear_prompt): never consulted,
        # and its bound action still fires (the prompt is cleared).
        prompt.text = "hi"
        prompt.move_cursor((0, 2))
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert "ctrl+c" not in seen
        assert prompt.text == ""

        # ctrl+d (quit / hard exit): never consulted, and quit still fires.
        await pilot.press("ctrl+d")
        await pilot.pause()
        assert "ctrl+d" not in seen
        assert quits == [1]

        # A non-reserved key is still routed through the interceptor.
        await pilot.press("escape")
        await pilot.pause()
        assert "escape" in seen


@pytest.mark.anyio
async def test_component_factory_crash_is_isolated() -> None:
    # [FUNCTION DOC] Async: verifies a raising slot widget factory is isolated: app stays up, nothing mounted,
    # and the failure is recorded.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        def exploding_factory(theme):  # noqa: ANN001, ANN202
            # [FUNCTION DOC] HELPER: factory that always raises RuntimeError("factory exploded").
            raise RuntimeError("factory exploded")

        bridge.set_slot_widget("bad", exploding_factory, placement="below_prompt")
        await pilot.pause()

        assert app.is_running
        assert "bad" not in app._extension_slot_widgets
        assert not app.query_one("#below-prompt-slot", Container).children
        assert "slot:bad" in app._extension_component_failures_reported


@pytest.mark.anyio
async def test_component_render_crash_is_quarantined() -> None:
    # [FUNCTION DOC] Async: verifies a widget that raises on render is quarantined and removed.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        bridge.set_slot_widget("crash", lambda theme: _CrashOnRender("x", id="ext-crash"))
        await pilot.pause()
        await pilot.pause()

        # The app survives the render exception and the widget is quarantined.
        assert app.is_running
        assert "crash" not in app._extension_slot_widgets
        assert not app.query("#ext-crash")


@pytest.mark.anyio
async def test_component_mount_crash_is_quarantined() -> None:
    # [FUNCTION DOC] Async: verifies a widget that raises on mount is untracked and hidden; uses a short real
    # sleep because the widget's pending messages never drain.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        bridge.set_slot_widget("crash", lambda theme: _CrashOnMount("x", id="ext-crash-mount"))
        # A widget that crashes in on_mount never finishes mounting, so its
        # pending message never drains (pilot.pause would time out); sleep to
        # let _handle_exception run instead. The app must stay alive and the
        # widget must be untracked and made inert.
        await asyncio.sleep(0.3)

        assert app.is_running
        assert "crash" not in app._extension_slot_widgets
        ghosts = app.query("#ext-crash-mount")
        assert all(not widget.display for widget in ghosts)


@pytest.mark.anyio
async def test_component_bridge_clear_components_tears_down_extension_ui() -> None:
    # [FUNCTION DOC] Async: verifies bridge.clear_components removes slot widgets, interceptors and the main
    # view (resolving its wait), restores the transcript, and stale unsubscribes are no-ops.
    # clear_components is the teardown seam the runtime drives on /reload and
    # session rebinds (resume/new): every slot widget, the main view (with its
    # pending wait() resolved), and all key interceptors must go.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        bridge.set_slot_widget("k", lambda theme: Static("x", id="ext-bridge-clear"))
        unsubscribe = bridge.register_key_interceptor(lambda event, text: False)
        handle = bridge.open_main_view(lambda h, theme: Static("v", id="ext-bridge-view"))
        await pilot.pause()
        assert app._extension_slot_widgets
        assert app._extension_key_interceptors
        assert handle.is_open
        waiter = asyncio.ensure_future(handle.wait())

        bridge.clear_components()
        await pilot.pause()
        await pilot.pause()

        assert app._extension_slot_widgets == {}
        assert app._extension_key_interceptors == []
        assert app._extension_main_view is None
        assert not handle.is_open
        assert await waiter is None
        assert app.query_one("#transcript", TranscriptView).display
        assert not app.query("#ext-bridge-clear")
        assert not app.query("#ext-bridge-view")
        # A stale unsubscribe after the clear is a safe no-op.
        unsubscribe()


@pytest.mark.anyio
async def test_component_rebind_clears_slots_and_interceptors() -> None:
    # [FUNCTION DOC] Async: verifies reconnecting an extension runtime clears all slots, interceptors and the
    # main view.
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bridge = _component_bridge(app)

        bridge.set_slot_widget("k", lambda theme: Static("x", id="ext-rebind"))
        bridge.register_key_interceptor(lambda event, text: False)
        handle = bridge.open_main_view(lambda h, theme: Static("v", id="ext-view"))
        await pilot.pause()
        assert app._extension_slot_widgets
        assert app._extension_key_interceptors
        assert handle.is_open

        # Rebinding a session force-clears everything before the new bridge.
        session = FakeSession()
        session.extension_runtime = _RenderCallRuntime()  # type: ignore[attr-defined]
        app._connect_extension_runtime(session)  # type: ignore[arg-type]
        await pilot.pause()

        assert app._extension_slot_widgets == {}
        assert app._extension_key_interceptors == []
        assert app._extension_main_view is None
        assert not handle.is_open
        assert app.query_one("#transcript", TranscriptView).display
        assert not app.query("#ext-rebind")
        assert not app.query("#ext-view")


@pytest.mark.anyio
async def test_tui_login_provider_search_autofocus() -> None:
    # [FUNCTION DOC] Async: verifies the login provider picker's search input receives focus on mount.
    app = TauTuiApp(FakeSession())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/login"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, LoginMethodPickerScreen)
        app.screen.action_cursor_down()
        app.screen.action_select_cursor()

        await pilot.wait_for_scheduled_animations()

        search = app.screen.query_one("#login-provider-search", Input)
        assert search.has_focus, "Search input failed to automatically gain focus on screen mount."
