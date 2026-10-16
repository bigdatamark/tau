# [MODULE DOC]
# Slash-command layer for Tau coding sessions. Defines the session protocol that
# handlers read from, the immutable CommandResult a handler returns (flags and
# values the frontend acts on: exit, new session, compact, export, pickers,
# login/logout, model/theme/thinking selection, messages), the command context
# and registration records, the CommandRegistry that parses "/name args" and
# dispatches, the default built-in command set, and the handlers themselves.
# Handlers are synchronous and side-effect free apart from /model's direct
# set_model fallback; frontends perform the requested actions.
# LOGIN_PROVIDER_ALIASES maps convenience login names to (provider, method).
"""Slash command registry for Tau coding sessions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tau_agent.tools import AgentTool
from tau_coding.prompt_templates import PromptTemplate
from tau_coding.provider_catalog import BUILTIN_PROVIDER_CATALOG, builtin_provider_entry
from tau_coding.reload import CodingReloadSummary, ReloadCategorySummary
from tau_coding.resources import ResourceDiagnostic
from tau_coding.session_manager import (
    CodingSessionRecord,
    SessionManager,
    normalize_session_name,
)
from tau_coding.skills import Skill
from tau_coding.system_prompt import (
    ProjectContextFile,
    SystemPromptInspection,
    SystemPromptSource,
    format_system_prompt_inspection,
)
from tau_coding.thinking import normalize_thinking_level

LOGIN_PROVIDER_ALIASES = {
    "anthropic-api": ("anthropic", "api-key"),
    "anthropic-subscription": ("anthropic", "subscription"),
}


class CommandSession(Protocol):
    # [CLASS DOC]
    # Protocol of the session attributes handlers may read: cwd, model, provider,
    # available models/providers, tools, skills, prompt templates, context files,
    # token estimates and thresholds, thinking levels, diagnostics, system prompt
    # and its inspection, session id/title/manager, plus two mutators. Handlers also
    # probe optional extra attributes dynamically.
    """Session attributes available to slash-command handlers."""

    @property
    def cwd(self) -> Path: ...
        # [FUNCTION DOC]
        # Protocol property: working directory of the session.

    @property
    def model(self) -> str: ...
        # [FUNCTION DOC]
        # Protocol property: active model id.

    @property
    def provider_name(self) -> str: ...
        # [FUNCTION DOC]
        # Protocol property: active provider name.

    @property
    def available_models(self) -> Sequence[str]: ...
        # [FUNCTION DOC]
        # Protocol property: model ids available for the active provider.

    @property
    def available_providers(self) -> Sequence[str]: ...
        # [FUNCTION DOC]
        # Protocol property: configured provider names.

    @property
    def tools(self) -> Sequence[AgentTool]: ...
        # [FUNCTION DOC]
        # Protocol property: agent tools active in the session.

    @property
    def skills(self) -> Sequence[Skill]: ...
        # [FUNCTION DOC]
        # Protocol property: loaded skills.

    @property
    def prompt_templates(self) -> Sequence[PromptTemplate]: ...
        # [FUNCTION DOC]
        # Protocol property: loaded prompt templates.

    @property
    def context_files(self) -> Sequence[ProjectContextFile]: ...
        # [FUNCTION DOC]
        # Protocol property: loaded project context files.

    @property
    def context_token_estimate(self) -> int: ...
        # [FUNCTION DOC]
        # Protocol property: estimated tokens of current context.

    @property
    def auto_compact_token_threshold(self) -> int | None: ...
        # [FUNCTION DOC]
        # Protocol property: token count that triggers auto compaction, or nothing.

    @property
    def context_window_tokens(self) -> int: ...
        # [FUNCTION DOC]
        # Protocol property: model context window size in tokens.

    @property
    def thinking_level(self) -> str: ...
        # [FUNCTION DOC]
        # Protocol property: current thinking level name.

    @property
    def available_thinking_levels(self) -> Sequence[str]: ...
        # [FUNCTION DOC]
        # Protocol property: thinking levels supported by the model (empty if none).

    @property
    def resource_diagnostics(self) -> Sequence[ResourceDiagnostic]: ...
        # [FUNCTION DOC]
        # Protocol property: resource loading diagnostics.

    @property
    def system_prompt(self) -> str: ...
        # [FUNCTION DOC]
        # Protocol property: effective system prompt text.

    @property
    def system_prompt_inspection(self) -> SystemPromptInspection: ...
        # [FUNCTION DOC]
        # Protocol property: source-attributed system prompt breakdown.

    @property
    def session_id(self) -> str | None: ...
        # [FUNCTION DOC]
        # Protocol property: persistent session id, or nothing.

    @property
    def session_title(self) -> str | None: ...
        # [FUNCTION DOC]
        # Protocol property: session title, or nothing.

    @property
    def session_manager(self) -> SessionManager | None: ...
        # [FUNCTION DOC]
        # Protocol property: session index manager, or nothing.

    def ensure_session_indexed(self) -> None: ...
        # [FUNCTION DOC]
        # Protocol method: make sure the session is recorded in the session index.

    def set_model(self, model: str) -> None: ...
        # [FUNCTION DOC]
        # Protocol method: switch the active model id (mutates the session).


@dataclass(frozen=True, slots=True)
class CommandResult:
    # [CLASS DOC]
    # Frozen outcome of a command. handled is false when the input should be sent as
    # an ordinary prompt. Remaining fields default to "nothing requested" and tell
    # the frontend what to do: exit, clear, reload, new session, compact (with
    # optional instructions), export (destination and format), resume a session id
    # or open pickers (resume, prompts, tree, login, logout, model, tools, scoped
    # models, skills, theme), custom provider login, local backends, sidebar toggle,
    # login/logout provider and method, model selection, thinking level, theme,
    # a message to display, a system prompt inspection and a new session name.
    """Result of handling a coding-session slash command."""

    handled: bool
    exit_requested: bool = False
    clear_requested: bool = False
    reload_requested: bool = False
    new_session_requested: bool = False
    compact_summary: str | None = None
    export_requested: bool = False
    export_destination: Path | None = None
    export_format: str | None = None
    resume_session_id: str | None = None
    resume_picker_requested: bool = False
    prompts_picker_requested: bool = False
    tree_picker_requested: bool = False
    login_picker_requested: bool = False
    custom_provider_login_requested: bool = False
    local_requested: bool = False
    sidebar_toggle_requested: bool = False
    login_provider: str | None = None
    login_method: str | None = None
    logout_picker_requested: bool = False
    logout_provider: str | None = None
    model_picker_requested: bool = False
    model_selection_provider: str | None = None
    model_selection_model: str | None = None
    tools_picker_requested: bool = False
    scoped_models_picker_requested: bool = False
    skills_picker_requested: bool = False
    theme_picker_requested: bool = False
    thinking_level: str | None = None
    theme: str | None = None
    message: str | None = None
    system_prompt_inspection: SystemPromptInspection | None = None
    session_name: str | None = None


@dataclass(frozen=True, slots=True)
class CommandContext:
    # [CLASS DOC]
    # Frozen context passed to a handler: the session, the registry, the full
    # stripped input text, the resolved command name and the argument text.
    """Runtime context passed to slash-command handlers."""

    session: CommandSession
    registry: CommandRegistry
    text: str
    name: str
    args: str


CommandHandler = Callable[[CommandContext], CommandResult]


@dataclass(frozen=True, slots=True)
class SlashCommand:
    # [CLASS DOC]
    # Frozen registration record: name, description, usage string, handler function,
    # aliases and extra search terms for completion.
    """A registered slash command and its user-facing metadata."""

    name: str
    description: str
    usage: str
    handler: CommandHandler
    aliases: tuple[str, ...] = ()
    search_terms: tuple[str, ...] = ()


class CommandRegistry:
    # [CLASS DOC]
    # Registry mapping normalized command names to commands and aliases to names.
    """Parse, register, list, and execute slash commands."""

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Create empty command and alias mappings.
        self._commands: dict[str, SlashCommand] = {}
        self._aliases: dict[str, str] = {}

    def register(self, command: SlashCommand) -> None:
        # [FUNCTION DOC]
        # Register a command under its normalized name (lowercased, leading slash
        # removed) and each normalized alias. Raises ValueError for a duplicate name or
        # an alias colliding with a command or alias. Note: a name added later is not
        # checked against existing aliases. Mutates the registry.
        """Register a slash command and its aliases."""
        name = _normalize_name(command.name)
        if name in self._commands:
            raise ValueError(f"Duplicate slash command: /{name}")
        self._commands[name] = command
        for alias in command.aliases:
            normalized_alias = _normalize_name(alias)
            if normalized_alias in self._commands or normalized_alias in self._aliases:
                raise ValueError(f"Duplicate slash command alias: /{normalized_alias}")
            self._aliases[normalized_alias] = name

    def get(self, name: str) -> SlashCommand | None:
        # [FUNCTION DOC]
        # Return the command for a name or alias (normalized), or nothing.
        """Return a command by name or alias."""
        normalized = _normalize_name(name)
        command_name = self._aliases.get(normalized, normalized)
        return self._commands.get(command_name)

    def list_commands(self) -> tuple[SlashCommand, ...]:
        # [FUNCTION DOC]
        # Return all commands (not aliases) sorted by name.
        """Return registered commands sorted by name."""
        return tuple(self._commands[name] for name in sorted(self._commands))

    def execute(self, session: CommandSession, text: str) -> CommandResult:
        # [FUNCTION DOC]
        # Parse and run a command. Returns an unhandled result for non-slash text,
        # /skill: invocations, an empty name, or unknown commands (so prompt templates
        # can expand them). "/scoped models" is accepted as /scoped-models. Otherwise
        # calls the handler with a new context and returns its result; handler
        # exceptions propagate.
        """Execute a slash command, or return unhandled for ordinary prompts."""
        stripped = text.strip()
        if not stripped.startswith("/"):
            return CommandResult(handled=False)

        if stripped.startswith("/skill:"):
            return CommandResult(handled=False)

        name, args = _parse_command(stripped)
        if not name:
            return CommandResult(handled=False)

        command = self.get(name)
        if command is None and name == "scoped" and args.lower() == "models":
            command = self.get("scoped-models")
            name = "scoped-models"
            args = ""
        if command is None:
            return CommandResult(handled=False)

        return command.handler(
            CommandContext(session=session, registry=self, text=stripped, name=name, args=args)
        )


def create_default_command_registry() -> CommandRegistry:
    # [FUNCTION DOC]
    # Build a registry with the built-in commands: quit (alias exit), new, compact,
    # export, session, system, skill, skills, hotkeys, prompts, reload, resume, tree,
    # name, model, tools, scoped-models, theme, local, sidebar, login, logout, each
    # with usage, description and search terms. Returns the registry.
    """Create Tau's built-in slash command registry."""
    registry = CommandRegistry()
    registry.register(
        SlashCommand(
            name="quit",
            usage="/quit",
            description="Exit the current session.",
            handler=_exit_command,
            aliases=("exit",),
        )
    )
    registry.register(
        SlashCommand(
            name="new",
            usage="/new",
            description="Start a new session.",
            handler=_new_command,
            search_terms=("clear", "reset"),
        )
    )
    registry.register(
        SlashCommand(
            name="compact",
            usage="/compact [instructions]",
            description="Summarize and compact active context.",
            handler=_compact_command,
        )
    )
    registry.register(
        SlashCommand(
            name="export",
            usage="/export [--format html|jsonl] [destination]",
            description="Export the current session.",
            handler=_export_command,
        )
    )
    registry.register(
        SlashCommand(
            name="session",
            usage="/session",
            description="Show session info and stats.",
            handler=_status_command,
            search_terms=("info",),
        )
    )
    registry.register(
        SlashCommand(
            name="system",
            usage="/system",
            description="Show the active system prompt without saving it.",
            handler=_system_command,
            search_terms=("prompt", "instructions"),
        )
    )
    registry.register(
        SlashCommand(
            name="skill",
            usage="/skill:<name> [request]",
            description="Expand a loaded skill into your prompt.",
            handler=_skill_command,
            search_terms=("skills",),
        )
    )
    registry.register(
        SlashCommand(
            name="skills",
            usage="/skills",
            description="Browse and insert a loaded skill.",
            handler=_skills_command,
            search_terms=("skill", "picker", "search"),
        )
    )
    registry.register(
        SlashCommand(
            name="hotkeys",
            usage="/hotkeys",
            description="Show common keyboard shortcuts.",
            handler=_hotkeys_command,
            search_terms=("keys", "shortcuts", "bindings"),
        )
    )
    registry.register(
        SlashCommand(
            name="prompts",
            usage="/prompts",
            description="Choose a loaded prompt template.",
            handler=_prompts_command,
            search_terms=("templates", "picker"),
        )
    )
    registry.register(
        SlashCommand(
            name="reload",
            usage="/reload",
            description="Reload local resources and project context.",
            handler=_reload_command,
        )
    )
    registry.register(
        SlashCommand(
            name="resume",
            usage="/resume [session-id]",
            description="Resume a previous session.",
            handler=_resume_command,
            search_terms=("history", "previous"),
        )
    )
    registry.register(
        SlashCommand(
            name="tree",
            usage="/tree",
            description="Branch from a previous session entry.",
            handler=_tree_command,
            search_terms=("branch", "history", "fork"),
        )
    )
    registry.register(
        SlashCommand(
            name="name",
            usage="/name <new name>",
            description="Rename the current session.",
            handler=_name_command,
            search_terms=("rename", "title"),
        )
    )
    registry.register(
        SlashCommand(
            name="model",
            usage="/model",
            description="Choose the active model.",
            handler=_model_command,
        )
    )
    registry.register(
        SlashCommand(
            name="tools",
            usage="/tools",
            description="Browse tools available to the active session.",
            handler=_tools_command,
            search_terms=("capabilities", "reference"),
        )
    )
    registry.register(
        SlashCommand(
            name="scoped-models",
            usage="/scoped-models",
            description="Choose models available to quick-cycle with Ctrl+P.",
            handler=_scoped_models_command,
            search_terms=("scope", "quick", "cycle", "ctrl+p"),
        )
    )
    registry.register(
        SlashCommand(
            name="theme",
            usage="/theme [name]",
            description="Show or set the TUI theme.",
            handler=_theme_command,
            search_terms=("light", "dark", "contrast"),
        )
    )
    registry.register(
        SlashCommand(
            name="local",
            usage="/local",
            description="Configure and manage local backends.",
            handler=_local_command,
            search_terms=("backends", "inference"),
        )
    )
    registry.register(
        SlashCommand(
            name="sidebar",
            usage="/sidebar",
            description="Show or hide the TUI sidebar for this session.",
            handler=_sidebar_command,
            search_terms=("toggle", "visibility", "panel"),
        )
    )
    registry.register(
        SlashCommand(
            name="login",
            usage="/login [provider]",
            description="Connect a provider with OAuth or an API key.",
            handler=_login_command,
        )
    )
    registry.register(
        SlashCommand(
            name="logout",
            usage="/logout [provider]",
            description="Remove saved credentials for a built-in provider.",
            handler=_logout_command,
        )
    )
    return registry


def _help_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: return a message listing every command's usage and description.
    # KNOWN ISSUE: not registered in the default registry, so there is no /help.
    lines = ["Available commands:"]
    for command in context.registry.list_commands():
        lines.append(f"{command.usage}\t{command.description}")
    return CommandResult(handled=True, message="\n".join(lines))


def _exit_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /quit handler; requests exit with an "Exiting session." message.
    return CommandResult(handled=True, exit_requested=True, message="Exiting session.")


def _new_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /new handler; requests a new session.
    return CommandResult(handled=True, new_session_requested=True)


def _compact_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /compact handler; requests compaction with the stripped arguments as
    # custom summary instructions (empty string when none).
    return CommandResult(
        handled=True,
        compact_summary=context.args.strip(),
    )


def _export_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /export handler; parses optional format and destination and requests
    # an export, or returns the parse error message.
    try:
        export_format, destination = _parse_export_args(context.args)
    except ValueError as exc:
        return CommandResult(handled=True, message=str(exc))
    return CommandResult(
        handled=True,
        export_requested=True,
        export_destination=destination,
        export_format=export_format,
    )


def _status_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /session handler; builds a multi-line status message: model, provider,
    # cwd, counts of tools/skills/templates/context files, token estimate and window,
    # Hugging Face routing mode when applicable, optional context window source and
    # model limit discovery error, a provider-usage or estimated token breakdown
    # when context usage is available, thinking lines, diagnostics count, auto
    # compact threshold, session id and name. Reads optional attributes dynamically.
    session = context.session
    context_usage = getattr(session, "context_usage", None)
    lines = [
        f"Model: {session.model}",
        f"Provider: {session.provider_name}",
        f"CWD: {session.cwd}",
        f"Tools: {len(session.tools)}",
        f"Skills: {len(session.skills)}",
        f"Prompt templates: {len(session.prompt_templates)}",
        f"Context files: {len(session.context_files)}",
        f"Estimated context tokens: {session.context_token_estimate}",
        f"Context window: {session.context_window_tokens}",
    ]
    if session.provider_name == "huggingface":
        route = getattr(session, "inference_provider", None)
        mode = getattr(session, "inference_provider_mode", "fixed" if route else "automatic")
        if mode == "automatic":
            route_status = f"automatic (currently {route})" if route else "automatic"
        else:
            route_status = f"{route} (fixed)" if route else "fixed"
        lines.append(f"Hugging Face inference provider: {route_status}")
    context_window_source = getattr(session, "context_window_source", None)
    if context_window_source:
        lines.append(f"Context window source: {context_window_source}")
    discovery_error = getattr(session, "model_limits_discovery_error", None)
    if discovery_error:
        lines.append(f"Model limit discovery: unavailable ({discovery_error})")
    if context_usage is not None:
        if context_usage.uses_provider_usage:
            lines.append(
                "Context token basis: "
                f"provider={context_usage.provider_tokens}, "
                f"estimated trailing={context_usage.trailing_tokens}",
            )
        else:
            lines.append(
                "Context token breakdown: "
                f"system={context_usage.system_tokens}, "
                f"messages={context_usage.message_tokens}, "
                f"tools={context_usage.tool_tokens}",
            )
    lines.extend(_thinking_status_lines(session))
    lines.append(f"Resource diagnostics: {len(session.resource_diagnostics)}")
    if session.auto_compact_token_threshold is not None:
        lines.append(f"Auto compact threshold: {session.auto_compact_token_threshold}")
    if session.session_id is not None:
        lines.append(f"Session: {session.session_id}")
    if session.session_title:
        lines.append(f"Session name: {session.session_title}")
    return CommandResult(handled=True, message="\n".join(lines))


def _system_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /system handler; rejects arguments with a usage message. Uses the
    # session's prompt inspection, or synthesizes a single runtime source from the
    # system prompt, and returns its formatted text plus the inspection.
    if context.args:
        return CommandResult(handled=True, message="Usage: /system")
    inspection = getattr(context.session, "system_prompt_inspection", None)
    if inspection is None:
        inspection = SystemPromptInspection(
            text=context.session.system_prompt,
            sources=(
                SystemPromptSource(
                    kind="runtime",
                    label="Effective system prompt",
                    source="active Tau session",
                    content=context.session.system_prompt,
                ),
            ),
        )
    return CommandResult(
        handled=True,
        message=format_system_prompt_inspection(inspection),
        system_prompt_inspection=inspection,
    )


def _hotkeys_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /hotkeys handler; returns a fixed list of keyboard shortcuts.
    lines = [
        "Common keyboard shortcuts:",
        "- Enter: submit prompt",
        "- Shift+Enter: insert newline",
        "- Alt+Enter: queue follow-up while running",
        "- Esc: cancel active run",
        "- Ctrl+K: open slash-command completions",
        "- Ctrl+R: open session picker",
        "- Ctrl+P / Shift+Ctrl+P: cycle scoped models forward / backward",
        "- Shift+Tab: cycle thinking mode",
        "- Ctrl+T: toggle thinking tokens",
        "- Ctrl+O: collapse or expand tool output",
        "- Ctrl+C: clear prompt input",
        "- Ctrl+D: quit",
    ]
    return CommandResult(handled=True, message="\n".join(lines))


def _skills_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /skills handler; requests the skills picker (usage message if args given).
    if context.args:
        return CommandResult(handled=True, message="Usage: /skills")
    return CommandResult(handled=True, skills_picker_requested=True)


def _resources_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: summarize skill, template and context file counts plus all resource
    # diagnostics. KNOWN ISSUE: not registered as any command (dead code).
    session = context.session
    lines = [
        f"Skills: {len(session.skills)}",
        f"Prompt templates: {len(session.prompt_templates)}",
        f"Context files: {len(session.context_files)}",
    ]
    if session.resource_diagnostics:
        lines.append("")
        lines.extend(_format_diagnostics(session.resource_diagnostics))
    else:
        lines.append("Resource diagnostics: none")
    return CommandResult(handled=True, message="\n".join(lines))


def _reload_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /reload handler; only requests a reload since reloading is async and
    # performed by the frontend.
    # Reload owns async extension lifecycle hooks, so frontends execute it from
    # their async command path rather than inside this synchronous registry.
    return CommandResult(handled=True, reload_requested=True)


def _context_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: list active project context files (or say none) plus context-kind
    # diagnostics. KNOWN ISSUE: not registered as any command (dead code).
    session = context.session
    if not session.context_files:
        lines = ["No project context files loaded."]
        if session.resource_diagnostics:
            lines.append("")
            lines.extend(_format_diagnostics(session.resource_diagnostics, kind="context"))
        return CommandResult(handled=True, message="\n".join(lines))

    lines = ["Active project context files:"]
    lines.extend(f"- {context_file.path}" for context_file in session.context_files)
    if session.resource_diagnostics:
        lines.append("")
        lines.extend(_format_diagnostics(session.resource_diagnostics, kind="context"))
    return CommandResult(handled=True, message="\n".join(lines))


def _skill_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: bare /skill handler; returns a hint explaining /skill:<name> usage.
    return CommandResult(
        handled=True,
        message="Use /skill:<name> [request] to expand a loaded skill into your prompt.",
    )


def _prompts_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /prompts handler; requests the prompt template picker (usage if args).
    if context.args:
        return CommandResult(handled=True, message="Usage: /prompts")
    return CommandResult(handled=True, prompts_picker_requested=True)


def _resume_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /resume handler; without args opens the resume picker; with an id
    # checks the session manager exists and knows the id, then requests resuming
    # it, otherwise returns an explanatory message. Reads the session index.
    if not context.args:
        return CommandResult(handled=True, resume_picker_requested=True)
    manager = context.session.session_manager
    if manager is None:
        return CommandResult(handled=True, message="Session manager is not available.")
    session_id = context.args.strip()
    if manager.get_session(session_id) is None:
        return CommandResult(handled=True, message=f"Unknown session: {session_id}")
    return CommandResult(
        handled=True,
        resume_session_id=session_id,
    )


def _tree_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /tree handler; requests the branch tree picker (usage if args).
    if context.args:
        return CommandResult(handled=True, message="Usage: /tree")
    return CommandResult(handled=True, tree_picker_requested=True)


def _name_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /name handler; requires a session manager and id. Without args shows
    # the current title (index record, then session title, then "Untitled session")
    # and usage. With args validates and normalizes the name and returns it as the
    # requested session name with a confirmation message; invalid names return the
    # error message.
    manager = context.session.session_manager
    session_id = context.session.session_id
    if manager is None or session_id is None:
        return CommandResult(handled=True, message="Session manager is not available.")

    if not context.args:
        record = manager.get_session(session_id)
        title = (
            record.title if record is not None else context.session.session_title
        ) or "Untitled session"
        return CommandResult(
            handled=True,
            message=f"Current session name: {title}\nUsage: /name <new name>",
        )

    try:
        name = _validated_session_name(context.args)
    except ValueError as exc:
        return CommandResult(handled=True, message=str(exc))

    return CommandResult(
        handled=True,
        session_name=name,
        message=f"Session renamed: {name}",
    )


def _format_sessions(context: CommandContext) -> str:
    # [FUNCTION DOC]
    # HELPER: format the indexed sessions for the cwd as a list, or a message when no
    # manager or no sessions. KNOWN ISSUE: unused by any registered command.
    manager = context.session.session_manager
    if manager is None:
        return "Session manager is not available."

    records = manager.list_sessions(context.session.cwd)
    if not records:
        return "No sessions found."

    lines = ["Indexed sessions:"]
    for record in records:
        lines.append(_format_session_record(record))
    return "\n".join(lines)


def _tools_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /tools handler; requests the tools picker (arguments ignored).
    return CommandResult(handled=True, tools_picker_requested=True)


def _model_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /model handler; without args requests the model picker. With a model
    # id, rejects it if the provider has a non-empty model list not containing it.
    # If the session supports provider-model selection, returns a selection
    # request; otherwise calls set_model directly (mutation) and reports it.
    # The picker uses the current snapshot and refreshes catalogs after opening.
    # Rebuilding the active provider here can prevent the picker from opening.
    if context.args:
        model = context.args.strip()
        available_models = set(context.session.available_models)
        if available_models and model not in available_models:
            models = ", ".join(sorted(available_models))
            return CommandResult(
                handled=True,
                message=f"Unknown model for provider {context.session.provider_name}: {model}\n"
                f"Available models: {models}",
            )
        if callable(getattr(context.session, "select_provider_model", None)):
            return CommandResult(
                handled=True,
                model_selection_provider=context.session.provider_name,
                model_selection_model=model,
            )
        context.session.set_model(model)
        return CommandResult(handled=True, message=f"Current model: {model}")

    return CommandResult(handled=True, model_picker_requested=True)


def _scoped_models_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /scoped-models handler; requests the scoped models picker (usage if args).
    if context.args:
        return CommandResult(handled=True, message="Usage: /scoped-models")
    return CommandResult(handled=True, scoped_models_picker_requested=True)


def _thinking_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: thinking-mode handler. Without args shows status lines plus available
    # modes (or the current model when unavailable). With args: errors if the model
    # has no thinking levels (with reason when known), normalizes the level
    # (reporting ValueError text), rejects levels not available, else returns the
    # requested thinking level. Note: not registered in the default registry.
    session = context.session
    available = tuple(session.available_thinking_levels)
    if not context.args:
        lines = _thinking_status_lines(session)
        if available:
            lines.append(f"Available modes: {', '.join(available)}")
        else:
            lines.insert(1, f"Current model: {session.provider_name}:{session.model}")
        return CommandResult(handled=True, message="\n".join(lines))

    if not available:
        message = f"Thinking controls are unavailable for {session.provider_name}:{session.model}"
        reason = _thinking_unavailable_reason(session)
        if reason:
            message = f"{message}: {reason}"
        return CommandResult(
            handled=True,
            message=message,
        )
    try:
        level = normalize_thinking_level(context.args)
    except ValueError as exc:
        return CommandResult(handled=True, message=str(exc))
    if level not in available:
        modes = ", ".join(available)
        return CommandResult(
            handled=True,
            message=(
                f"Thinking mode {level} is not available for "
                f"{session.provider_name}:{session.model}\n"
                f"Available modes: {modes}"
            ),
        )
    return CommandResult(handled=True, thinking_level=level)


def _thinking_status_lines(session: CommandSession) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: return "Thinking mode: <level>" when levels exist, else an unavailable
    # line plus the reason line if one is known.
    if tuple(session.available_thinking_levels):
        return [f"Thinking mode: {session.thinking_level}"]
    lines = ["Thinking mode: unavailable"]
    reason = _thinking_unavailable_reason(session)
    if reason:
        lines.append(f"Thinking unavailable: {reason}")
    return lines


def _thinking_unavailable_reason(session: CommandSession) -> str | None:
    # [FUNCTION DOC]
    # HELPER: return the session's optional non-empty thinking unavailable reason
    # string, or nothing.
    reason = getattr(session, "thinking_unavailable_reason", None)
    return reason if isinstance(reason, str) and reason else None


def _theme_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /theme handler; without args opens the theme picker; with a name,
    # lazily imports the TUI theme list (to avoid importing Textual early) and
    # returns the theme if known, else an error listing available themes.
    if not context.args:
        return CommandResult(handled=True, theme_picker_requested=True)

    # Imported lazily so importing this module never pulls in `tau_coding.tui`
    # (whose package __init__ imports Textual) until /theme actually executes.
    from tau_coding.tui.themes import available_tui_theme_names

    theme_name = context.args.strip()
    available = available_tui_theme_names()
    if theme_name not in available:
        themes = ", ".join(available)
        return CommandResult(
            handled=True,
            message=f"Unknown theme: {theme_name}\nAvailable themes: {themes}",
        )
    return CommandResult(handled=True, theme=theme_name)


def _local_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /local handler; requests the local backends UI (usage if args).
    if context.args:
        return CommandResult(handled=True, message="Usage: /local")
    return CommandResult(handled=True, local_requested=True)


def _sidebar_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /sidebar handler; requests a sidebar toggle (usage if args).
    if context.args:
        return CommandResult(handled=True, message="Usage: /sidebar")
    return CommandResult(handled=True, sidebar_toggle_requested=True)


def _login_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /login handler. "custom", "new" or "add" requests custom provider
    # login; a name is resolved through aliases (setting a login method) and the
    # built-in catalog, returning the provider and method or an unknown-provider
    # message listing options; no args opens the login picker.
    provider_name = context.args.strip()
    if provider_name in {"custom", "new", "add"}:
        return CommandResult(handled=True, custom_provider_login_requested=True)
    if provider_name:
        aliased_provider = LOGIN_PROVIDER_ALIASES.get(provider_name)
        if aliased_provider is not None:
            provider_name, login_method = aliased_provider
        else:
            login_method = None
        entry = builtin_provider_entry(provider_name)
        if entry is None:
            providers = ", ".join(
                [
                    *(entry.name for entry in BUILTIN_PROVIDER_CATALOG),
                    *LOGIN_PROVIDER_ALIASES,
                ]
            )
            return CommandResult(
                handled=True,
                message=(
                    f"Unknown login provider: {provider_name}\nAvailable providers: {providers}"
                ),
            )
        return CommandResult(
            handled=True,
            login_provider=entry.name,
            login_method=login_method,
        )

    return CommandResult(handled=True, login_picker_requested=True)


def _logout_command(context: CommandContext) -> CommandResult:
    # [FUNCTION DOC]
    # HELPER: /logout handler; with a built-in provider name requests logout for it,
    # unknown names return an error listing providers, no args opens the picker.
    provider_name = context.args.strip()
    if provider_name:
        entry = builtin_provider_entry(provider_name)
        if entry is None:
            providers = ", ".join(entry.name for entry in BUILTIN_PROVIDER_CATALOG)
            return CommandResult(
                handled=True,
                message=(
                    f"Unknown logout provider: {provider_name}\nAvailable providers: {providers}"
                ),
            )
        return CommandResult(handled=True, logout_provider=entry.name)

    return CommandResult(handled=True, logout_picker_requested=True)


def _format_session_record(record: CodingSessionRecord) -> str:
    # [FUNCTION DOC]
    # HELPER: format a session record as "- id: title (model) cwd" with "Untitled"
    # for a missing title.
    title = record.title or "Untitled"
    return f"- {record.id}: {title} ({record.model}) {record.cwd}"


def _format_diagnostics(
    diagnostics: Sequence[ResourceDiagnostic], *, kind: str | None = None
) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: format diagnostics (optionally filtered by kind) as a header plus
    # bullet lines, or a single "none" line when nothing matches.
    filtered = [diagnostic for diagnostic in diagnostics if kind is None or diagnostic.kind == kind]
    if not filtered:
        return ["Resource diagnostics: none"]
    lines = ["Resource diagnostics:"]
    lines.extend(f"- {diagnostic.format()}" for diagnostic in filtered)
    return lines


def format_reload_summary(summary: CodingReloadSummary) -> str:
    # [FUNCTION DOC]
    # Build the multi-line /reload report: per-category totals and change deltas
    # for skills, prompt templates, extensions, context files and diagnostics,
    # whether the next-turn system prompt was rebuilt, and a note that provider
    # config is not refreshed. Pure; returns the text.
    lines = [
        "Reloaded local coding resources and project context.",
        "Resources:",
        f"- Skills: {_format_reload_category(summary.skills)}",
        f"- Prompt templates: {_format_reload_category(summary.prompt_templates)}",
        f"- Extensions: {_format_reload_category(summary.extensions)}",
        "Context:",
        f"- Project context files: {_format_reload_category(summary.context_files)}",
        "- Next-turn system prompt: "
        + ("rebuilt" if summary.system_prompt_rebuilt else "unchanged"),
        "Diagnostics:",
        f"- Resource diagnostics: {_format_reload_category(summary.diagnostics)}",
        "Provider config:",
        "- Not refreshed by /reload; use /login or /model for provider/model settings.",
    ]
    return "\n".join(lines)


def _format_reload_category(summary: ReloadCategorySummary) -> str:
    # [FUNCTION DOC]
    # HELPER: format "<after> total (changed|unchanged[, +/-delta])".
    status = "changed" if summary.changed else "unchanged"
    delta = _format_count_delta(summary.delta)
    suffix = f", {delta}" if delta is not None else ""
    return f"{summary.after} total ({status}{suffix})"


def _format_count_delta(delta: int) -> str | None:
    # [FUNCTION DOC]
    # HELPER: signed delta string, or nothing for zero.
    if delta == 0:
        return None
    return f"{delta:+d}"


def _parse_command(text: str) -> tuple[str, str]:
    # [FUNCTION DOC]
    # HELPER: split "/name rest" at the first space into the normalized name and
    # stripped arguments. KNOWN ISSUE: only a space separates; a tab or newline
    # after the name becomes part of the command name.
    command, separator, args = text[1:].partition(" ")
    return _normalize_name(command), args.strip() if separator else ""


def _parse_export_args(args: str) -> tuple[str | None, Path | None]:
    # [FUNCTION DOC]
    # HELPER: parse export arguments: "--format X" or "--format=X" and at most one
    # destination path (user-expanded). Raises ValueError for a missing format value,
    # unknown dash options or extra positional arguments. The format value is not
    # validated here. Returns the format and destination (each possibly nothing).
    parts = args.split()
    export_format: str | None = None
    destination: Path | None = None
    index = 0
    while index < len(parts):
        part = parts[index]
        if part == "--format":
            index += 1
            if index >= len(parts):
                raise ValueError("Usage: /export [--format html|jsonl] [destination]")
            export_format = parts[index]
        elif part.startswith("--format="):
            export_format = part.partition("=")[2]
        elif part.startswith("-"):
            raise ValueError(f"Unknown export option: {part}")
        elif destination is None:
            destination = Path(part).expanduser()
        else:
            raise ValueError("Usage: /export [--format html|jsonl] [destination]")
        index += 1
    return export_format, destination


def _validated_session_name(value: str) -> str:
    # [FUNCTION DOC]
    # HELPER: raise ValueError with usage for blank input, else return the
    # normalized session name (which may itself raise ValueError).
    if not value.strip():
        raise ValueError("Usage: /name <new name>")
    return normalize_session_name(value)


def _normalize_name(name: str) -> str:
    # [FUNCTION DOC]
    # HELPER: strip whitespace, remove one leading slash and lowercase.
    return name.strip().removeprefix("/").lower()
