# [MODULE DOC]
# The coding-agent environment (Pi's AgentSession) for tau_coding. CodingSession
# wraps the reusable AgentHarness (the in-memory agent brain) with everything that
# makes it a coding session:
# - durable, append-only session entries (JSONL storage) with a session tree,
#   branching, labels, and replay into harness state on load;
# - provider/model selection (built-in catalog providers and extension-registered
#   dynamic providers), thinking levels, model-limit and catalog discovery;
# - Tau resources: skills, prompt templates, project context files, custom/append
#   system prompts, all gated by project trust;
# - extensions (runtime, tools, commands, providers) including reload;
# - context-window estimation, manual and automatic compaction, branch summaries;
# - auto-retry of transient provider errors, steering/follow-up queues, terminal
#   ("!" / "!!") commands, session naming, export, stats, and diagnostics logging.
# It emits coding-session events (agent events plus Tau-specific events) for
# frontends to render; it does not depend on any UI toolkit.
"""Persistent coding-session wrapper built on AgentHarness."""

from __future__ import annotations

import asyncio
import string
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from os import environ
from pathlib import Path
from typing import Literal

from tau_agent.events import AgentEndEvent, AgentEvent, MessageEndEvent, ToolExecutionEndEvent
from tau_agent.harness import AgentHarness, AgentHarnessConfig, QueuedMessages
from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    CustomMessage,
    Usage,
    UserMessage,
    message_text,
    sum_usage,
)
from tau_agent.provider import ModelProvider
from tau_agent.provider_events import AssistantDoneEvent, AssistantErrorEvent, TextDeltaEvent
from tau_agent.session import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    CustomMessageEntry,
    JsonlSessionStorage,
    LabelEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionInfoEntry,
    SessionState,
    SessionStorage,
    ThinkingLevelChangeEntry,
)
from tau_agent.session.entries import SessionEntry
from tau_agent.session.jsonl import entry_to_json_line
from tau_agent.session.tree import SessionTreeError, path_to_entry
from tau_agent.tool_history import ToolHistoryRepair, repair_tool_history
from tau_agent.tools import AgentTool
from tau_agent.types import JSONValue
from tau_ai.model_catalog import ModelCatalogProvider, RuntimeModel, RuntimeModelCatalog
from tau_ai.model_limits import ModelLimitsProvider, RuntimeModelLimits
from tau_coding.branch_summary import summarize_branch_messages_with_model
from tau_coding.codex_model_store import (
    cached_codex_model_catalog,
    save_codex_model_catalog,
)
from tau_coding.commands import CommandRegistry, CommandResult, create_default_command_registry
from tau_coding.context import discover_project_context_with_diagnostics
from tau_coding.context_window import (
    DEFAULT_COMPACTION_KEEP_RECENT_TOKENS,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    SUMMARIZATION_SYSTEM_PROMPT,
    ContextUsageEstimate,
    auto_compaction_threshold_for_context_window,
    build_compaction_summary_prompt,
    estimate_context_usage,
    estimate_message_tokens,
    summarize_messages_for_compaction,
)
from tau_coding.credentials import FileCredentialStore, credentials_path
from tau_coding.diagnostics import (
    AgentCallDiagnosticContext,
    AgentCallDiagnosticLogger,
    new_agent_call_run_id,
)
from tau_coding.events import (
    AgentSettledEvent,
    AutoRetryEndEvent,
    AutoRetryStartEvent,
    CodingSessionEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    QueueUpdateEvent,
    SessionAgentEndEvent,
    SessionInfoChangedEvent,
    ThinkingLevelChangedEvent,
)
from tau_coding.extensions.provider_registry import DynamicProviderRegistry
from tau_coding.extensions.providers import DynamicProvider, ProviderModel
from tau_coding.extensions.runtime import ExtensionRuntime
from tau_coding.models_dev_store import ModelsDevRefreshResult, refresh_models_dev_catalog
from tau_coding.oauth import account_id_from_access_token
from tau_coding.paths import TauPaths
from tau_coding.project_trust import (
    CanonicalProjectPath,
    ProjectTrustCoordinator,
    ProjectTrustResolution,
    ProjectTrustStore,
    TrustDefault,
    TrustOverride,
    TrustPrompt,
    format_trust_diagnostic,
)
from tau_coding.prompt_templates import (
    PromptTemplate,
    expand_prompt_template_command,
    load_prompt_templates_with_diagnostics,
)
from tau_coding.provider_config import (
    OpenAICodexProviderConfig,
    OpenAICompatibleProviderConfig,
    ProviderConfig,
    ProviderConfigError,
    ProviderModelMetadata,
    ProviderSettings,
    load_provider_settings,
    provider_default_thinking_level,
    provider_has_usable_credentials,
    provider_model_supports_images,
    provider_thinking_levels,
    provider_thinking_unavailable_reason,
    resolve_provider_selection,
    resolve_startup_thinking_level,
    save_default_provider_model,
    save_provider_thinking_level,
    toggle_saved_scoped_model,
    toggle_saved_stable_scoped_model,
    validate_huggingface_inference_provider,
    validate_provider_model,
)
from tau_coding.provider_runtime import (
    ClosableModelProvider,
    create_dynamic_model_provider,
    create_model_provider,
)
from tau_coding.reload import CodingReloadSummary, ReloadCategorySummary
from tau_coding.resources import (
    ResourceDiagnostic,
    ResourceError,
    TauResourcePaths,
    discover_system_prompt_resources,
    resource_paths_with_cwd,
    resource_paths_with_project_trust,
)
from tau_coding.session_export import (
    default_session_export_artifact_path,
    export_session_artifact,
    normalize_export_format,
)
from tau_coding.session_manager import (
    InferenceProviderMode,
    SessionManager,
    normalize_session_name,
)
from tau_coding.session_stats import SessionStats, calculate_session_stats
from tau_coding.skills import Skill, expand_skill_command, load_skills_with_diagnostics
from tau_coding.system_prompt import (
    BuildSystemPromptOptions,
    ProjectContextFile,
    PromptSection,
    SystemPromptInspection,
    SystemPromptSource,
    build_system_prompt,
    build_system_prompt_inspection,
)
from tau_coding.thinking import (
    DEFAULT_THINKING_LEVEL,
    THINKING_LEVELS,
    ThinkingLevel,
    next_thinking_level,
    normalize_thinking_level,
)
from tau_coding.tools import ImageSupportState, create_bash_tool, create_coding_tools

StreamingBehavior = Literal["steer", "follow_up"]
SESSION_NAME_SYSTEM_PROMPT = (
    "You write concise coding-agent session names. Reply with only a short title, "
    "maximum four words, no quotes, no punctuation-only output."
)
TREE_RUNNING_MESSAGE = "Tau is still working. Press Escape to interrupt before using /tree."


async def _await_cleanup_completion[CleanupResult](
    task: asyncio.Task[CleanupResult],
) -> bool:
    # [FUNCTION DOC]
    # HELPER: Await a cleanup task through a shield so caller cancellation is not
    # forwarded into it. Repeated cancellations are absorbed until the task is done.
    # Any exception from the task is swallowed (its owner inspects the result).
    # Input: the cleanup task. Output: True if the caller was cancelled while waiting.
    # Cancellation is reported, not re-raised; callers decide whether to propagate it.
    """Wait through caller cancellation without forwarding it into cleanup.

    Returns whether cancellation was observed. Repeated requests remain
    contained until the independently owned cleanup task reaches a terminal
    state.
    """
    cancelled = False
    while True:
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
            if not task.done():
                continue
        except BaseException:  # cleanup outcome is inspected by its owner
            pass
        return cancelled


async def _finish_adopted_runtime_close(runtime: ExtensionRuntime) -> None:
    # [FUNCTION DOC]
    # HELPER: Close an outgoing extension runtime after a new one was published: run
    # its async close in a named task, wait for it without propagating cancellation,
    # and retrieve and discard its result or error so adoption never fails because of
    # cleanup.
    """Finish outgoing cleanup after publication without failing adoption."""
    task = asyncio.create_task(
        runtime.aclose(),
        name="tau-adopted-extension-runtime-close",
    )
    await _await_cleanup_completion(task)
    # Publication is already committed. Cleanup cancellation/failure must not
    # masquerade as rollback while the task's outcome still gets retrieved.
    with suppress(BaseException):
        task.result()


async def _finish_aborted_session_close(session: CodingSession) -> None:
    # [FUNCTION DOC]
    # HELPER: Close a coding session that was prepared but never published (e.g. a
    # failed reload or switch): run its close in a task, wait without propagating
    # cancellation, and discard any error so the original failure stays primary.
    """Discharge an unpublished session's resources without masking its abort."""
    task = asyncio.create_task(
        session.aclose(),
        name="tau-aborted-coding-session-close",
    )
    await _await_cleanup_completion(task)
    # The pre-publication failure remains primary, but every owned provider was
    # attempted by CodingSession's durable, idempotent close task.
    with suppress(BaseException):
        task.result()


@dataclass(frozen=True, slots=True)
class ModelChoice:
    # [CLASS DOC]
    # Immutable pair of provider name and model id that the user can select.
    """A selectable model and the provider that serves it."""

    provider_name: str
    model: str


@dataclass(frozen=True, slots=True)
class ModelSelectionResult:
    # [CLASS DOC]
    # Immutable result of a model selection: the chosen provider/model and whether the
    # active selection actually changed.
    """Result of a candidate-first provider/model selection."""

    choice: ModelChoice
    changed: bool


@dataclass(frozen=True, slots=True)
class TerminalCommandResult:
    # [CLASS DOC]
    # Immutable result of an input-bar shell command: command text, captured output,
    # exit code (None if cancelled/unknown), success flag, and whether the output was
    # added to the model context.
    """Result of an input-bar terminal command."""

    command: str
    output: str
    exit_code: int | None
    ok: bool
    added_to_context: bool


@dataclass(frozen=True, slots=True)
class SessionTreeChoice:
    # [CLASS DOC]
    # Immutable row for the session tree picker: entry id, display label, whether it is
    # the active leaf, whether it represents a tool call, optional bookmark label and
    # the label's timestamp.
    """One branchable entry in the active session tree."""

    entry_id: str
    label: str
    active: bool = False
    is_tool_call: bool = False
    bookmark_label: str | None = None
    label_timestamp: float | None = None


@dataclass(frozen=True, slots=True)
class SessionTreeBranchResult:
    # [CLASS DOC]
    # Immutable result of moving the active leaf: a status message and optional text to
    # prefill into the input bar (e.g. the text of a user message branched from).
    """Result of moving the active session tree leaf."""

    message: str
    input_prefill: str | None = None


@dataclass(frozen=True, slots=True)
class TerminalCommandRequest:
    # [CLASS DOC]
    # Immutable parsed shell command from the input bar: the command and whether its
    # output should be added to context ("!" adds, "!!" does not).
    """Parsed input-bar terminal command request."""

    command: str
    add_to_context: bool


@dataclass(frozen=True, slots=True)
class SessionResources:
    # [CLASS DOC]
    # Immutable bundle of Tau-owned resources loaded for a session: skills, prompt
    # templates, context files, custom system prompt text and path, combined append
    # system prompt text plus the individual append prompts and their paths, and
    # resource diagnostics.
    """Tau-owned resources loaded around a coding session."""

    skills: tuple[Skill, ...]
    prompt_templates: tuple[PromptTemplate, ...]
    context_files: tuple[ProjectContextFile, ...]
    custom_system_prompt: str | None
    custom_system_prompt_path: Path | None
    append_system_prompt: str | None
    append_system_prompts: tuple[str, ...]
    append_system_prompt_paths: tuple[Path, ...]
    diagnostics: tuple[ResourceDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class CompactionPlan:
    # [CLASS DOC]
    # Immutable plan for a compaction run: id of the first entry kept verbatim, number
    # of entries being replaced, and the messages to summarize.
    """Prepared active-context entries for a compaction run."""

    first_kept_entry_id: str
    replaced_entry_count: int
    messages_to_summarize: tuple[AgentMessage, ...]


@dataclass(frozen=True, slots=True)
class _GeneratedSummary:
    # [CLASS DOC]
    # HELPER: Immutable summary text plus the usage spent producing it and the
    # provider/model (and upstream response provider) that produced it, for accounting.
    """Summary text paired with the provider usage spent producing it."""

    text: str
    usage: Usage | None
    provider: str | None = None
    model: str | None = None
    response_provider: str | None = None


@dataclass(frozen=True, slots=True)
class ManualCompactionResult:
    # [CLASS DOC]
    # Immutable outcome of /compact: summary text, first kept entry id, estimated tokens
    # before, estimated tokens after, and the number of entries replaced.
    """Structured result from one manual compaction."""

    summary: str
    first_kept_entry_id: str
    tokens_before: int
    estimated_tokens_after: int
    replaced_entry_count: int


@dataclass(frozen=True, slots=True)
class _PendingMessageWrite:
    # [CLASS DOC]
    # HELPER: Immutable pairing of a message and the stable session entry created for
    # it, kept so a failed persistence attempt can be retried with the same entry id.
    """Stable entry retained while a message persistence attempt is retried."""

    message: AgentMessage
    entry: MessageEntry | CustomMessageEntry


@dataclass(frozen=True, slots=True)
class CodingSessionConfig:
    # [CLASS DOC]
    # Immutable configuration for creating or loading a coding session: initial
    # provider instance (may be None to build from settings), model id, session
    # storage, cwd, base system prompt and custom/append prompt overrides, explicit
    # context files, built-in tool override, resource paths, session id/manager,
    # command registry, provider name plus Hugging Face inference provider and mode,
    # requested provider/model from the CLI, provider name recorded in the session,
    # provider settings, runtime model catalogs and runtime provider config, an
    # optional dynamic provider, whether the session owns (must close) the initial
    # provider, auto-compaction threshold and enable flag, thinking level and a one-shot
    # override, whether to index the session on first persist, a shell command prefix,
    # skills enable flag, tool allow-list, extension paths and enable flags, an
    # optional prebuilt extension runtime, project trust coordinator and policy inputs,
    # and whether authoritative writes are deferred until a prepared session is adopted.
    """Configuration for a persistent coding session."""

    provider: ModelProvider | None
    model: str
    storage: SessionStorage
    cwd: Path
    system: str | None = None
    custom_system_prompt: str | None = None
    append_system_prompt: str | None = None
    context_files: tuple[ProjectContextFile, ...] = ()
    tools: list[AgentTool] | None = None
    resource_paths: TauResourcePaths | None = None
    session_id: str | None = None
    session_manager: SessionManager | None = None
    command_registry: CommandRegistry | None = None
    provider_name: str = "openai"
    inference_provider: str | None = None
    inference_provider_mode: InferenceProviderMode | None = None
    requested_provider: str | None = None
    requested_model: str | None = None
    session_provider_name: str | None = None
    provider_settings: ProviderSettings | None = None
    runtime_model_catalogs: Mapping[str, RuntimeModelCatalog] = field(default_factory=dict)
    runtime_provider_config: ProviderConfig | None = None
    dynamic_provider: DynamicProvider | None = None
    owns_initial_provider: bool = False
    auto_compact_token_threshold: int | None = None
    auto_compact_enabled: bool = True
    thinking_level: ThinkingLevel = DEFAULT_THINKING_LEVEL
    thinking_level_override: ThinkingLevel | None = None
    """One-shot startup override (e.g. ``--thinking``) for the session's level.

    Takes precedence over remembered per-model defaults and replayed session
    state, is validated strictly against the active model's available levels
    when the provider configuration is known, and is never persisted as a new
    remembered default.
    """
    index_on_first_persist: bool = False
    shell_command_prefix: str | None = None
    skills_enabled: bool = True
    """Whether skill discovery is enabled for this session.

    When ``True`` (the default), skills are discovered from the resource paths
    and their index is injected into the system prompt as ``<available_skills>``,
    and ``/skill:`` commands expand against them. When ``False``, skill discovery
    is suppressed for the whole session: no skills load, the ``<available_skills>``
    index is omitted, and ``/skill:`` commands find nothing to expand. This mirrors
    Pi's loader-level ``noSkills`` flag, which suppresses only skill discovery and
    leaves prompt templates and project context files (AGENTS.md) unaffected. It is
    the seam hosts use to construct skill-less sessions (e.g. a subagent type that
    gets no skills).
    """
    allowed_tool_names: frozenset[str] | None = None
    """Optional allow-list applied to the session's final tool set.

    ``tools`` only replaces the built-in tool set; extension-registered tools are
    composed on top of it afterwards. This filter is applied *after* that
    composition (on initial load and again on every reload), so it caps the
    complete tool list by name regardless of where a tool came from. An
    extension tool that overrides a built-in of the same name is subject to the
    list like any other. ``None`` (the default) disables filtering; names that
    match no tool are ignored. It is the seam hosts use to construct sessions
    with a restricted tool set (e.g. a subagent type limited to ``read``/``bash``).
    """
    extension_paths: tuple[Path, ...] = ()
    extensions_enabled: bool = True
    project_extensions_enabled: bool = False
    extension_runtime: ExtensionRuntime | None = None
    project_trust_coordinator: ProjectTrustCoordinator | None = None
    trust_override: TrustOverride | None = None
    trust_default: TrustDefault = "ask"
    trust_interactive: bool = False
    trust_prompt: TrustPrompt | None = None
    # Shared preparation keeps transcript/trust/index writes staged until
    # PreparedCodingSession.adopt() reaches its durable commit point.
    defer_authoritative_writes: bool = False


class CodingSession:
    # [CLASS DOC]
    # Tau's coding-agent environment wrapper around AgentHarness. Owns durable session
    # entries, tools, resources, providers, extensions, compaction, and commands, and
    # streams coding-session events to frontends. Instances are created via the async
    # load class method, used for prompts/commands, and must be closed with aclose.
    """Tau's coding-agent environment wrapper.

    `AgentHarness` owns the in-memory agent brain. `CodingSession` owns the
    coding-session environment around it: durable session entries, default coding
    tools, and a small command seam for later phases.
    """

    def __init__(
        self,
        config: CodingSessionConfig,
        *,
        state: SessionState,
        harness: AgentHarness,
        last_parent_id: str | None,
        skills: tuple[Skill, ...] = (),
        prompt_templates: tuple[PromptTemplate, ...] = (),
        context_files: tuple[ProjectContextFile, ...] = (),
        custom_system_prompt: str | None = None,
        custom_system_prompt_path: Path | None = None,
        append_system_prompt: str | None = None,
        append_system_prompts: tuple[str, ...] = (),
        append_system_prompt_paths: tuple[Path, ...] = (),
        resource_diagnostics: tuple[ResourceDiagnostic, ...] = (),
        command_registry: CommandRegistry | None = None,
        pending_initial_entries: tuple[SessionEntry, ...] = (),
        extension_runtime: ExtensionRuntime | None = None,
        image_support: ImageSupportState | None = None,
        project_trust_resolution: ProjectTrustResolution | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Construct a session from already-prepared parts (normally called only by load).
        # Inputs: the config; keyword replayed session state, the harness, last parent
        # entry id for appends, loaded resources (skills, templates, context files, system
        # prompt pieces and paths), resource diagnostics, command registry (defaults to the
        # built-in registry), initial entries not yet written to storage, extension runtime
        # (default empty runtime), image support state, and the project trust resolution.
        # Initializes provider/inference-provider state (mode defaults to "fixed" when an
        # inference provider is given, else "automatic"), provider settings (current and
        # durable copies), runtime catalogs, resource paths bound to the cwd, compaction
        # settings, preview state, thinking level from replayed state or the active model's
        # default, caches, owned-provider list, diagnostic logger, credential store, model
        # limit tracking, trust-commit flag, and message persistence bookkeeping, then
        # subscribes the persistence listener to harness events.
        # Side effects: subscribes to the harness; creates logger/store objects (no I/O).
        self._config = config
        self._state = state
        self._harness = harness
        self._extension_runtime = extension_runtime or ExtensionRuntime()
        self._provider_registry = self._extension_runtime.provider_registry
        self._image_support = image_support or ImageSupportState()
        self._session_start_pending = False
        self._last_parent_id = last_parent_id
        self._pending_initial_entries = pending_initial_entries
        self._prepared_entries: list[SessionEntry] = list(pending_initial_entries)
        self._skills = skills
        self._prompt_templates = prompt_templates
        self._context_files = context_files
        self._custom_system_prompt = custom_system_prompt
        self._custom_system_prompt_path = custom_system_prompt_path
        self._append_system_prompt = append_system_prompt
        self._append_system_prompts = append_system_prompts
        self._append_system_prompt_paths = append_system_prompt_paths
        self._resource_diagnostics = resource_diagnostics
        self._command_registry = command_registry or create_default_command_registry()
        self._provider_name = config.provider_name
        self._inference_provider = config.inference_provider
        self._inference_provider_mode: InferenceProviderMode = config.inference_provider_mode or (
            "fixed" if config.inference_provider is not None else "automatic"
        )
        self._provider_settings = config.provider_settings
        self._durable_provider_settings = config.provider_settings
        self._runtime_model_catalogs = dict(config.runtime_model_catalogs)
        self._model_catalog_discovery_errors: dict[str, str] = {}
        self._runtime_provider_config = config.runtime_provider_config
        self._resource_paths = resource_paths_with_cwd(config.resource_paths, config.cwd)
        self._auto_compact_token_threshold = config.auto_compact_token_threshold
        self._auto_compact_enabled = config.auto_compact_enabled
        self._preview_model: ModelChoice | None = None
        self._preview_thinking: ThinkingLevel | None = None
        self._thinking_level = _state_thinking_level(
            state,
            default=_default_thinking_level_for_active_model(self),
        )
        self._context_usage_cache: ContextUsageEstimate | None = None
        self._session_stats_cache: tuple[SessionState, CodingSessionConfig, SessionStats] | None = (
            None
        )
        self._owned_providers: list[ClosableModelProvider] = []
        self._close_task: asyncio.Task[None] | None = None
        self._diagnostic_logger = AgentCallDiagnosticLogger.from_paths(self._resource_paths.paths)
        self._credential_store = FileCredentialStore(
            credentials_path(self._resource_paths.paths) if self._resource_paths.paths else None
        )
        self._last_diagnostic_log_path: Path | None = None
        self._runtime_model_limits: RuntimeModelLimits | None = None
        self._runtime_model_limits_key: tuple[str, str] | None = None
        self._model_limits_discovery_error: str | None = None
        self._project_trust_resolution = project_trust_resolution
        self._project_trust_commit_pending = False
        self._persistence_unsubscribe: Callable[[], None] | None = None
        self._persisted_message_ids: set[int] = set()
        self._ended_message_ids: set[int] = set()
        self._pending_message_writes: dict[int, _PendingMessageWrite] = {}
        self._attach_persistence_listener()

    @classmethod
    async def load(cls, config: CodingSessionConfig) -> CodingSession:
        # [FUNCTION DOC]
        # Create a ready-to-use session from storage (async class method).
        # Steps:
        # 1. Read all entries. If empty, synthesize staged initial entries (session info
        #    with cwd, model change with initial model/provider, thinking level change);
        #    otherwise detach entries whose parents are missing. Replay into state.
        # 2. Build a fresh extension runtime bound to Tau paths (reusing injected
        #    credentials, environment, built-in credentials/HTTP client from a previous
        #    runtime if given) and load user/extra extensions (never project ones yet).
        # 3. Resolve project trust (extensions may decide; not cached; persisted unless
        #    writes are deferred), canonicalize cwd, gate resource paths by trust, load
        #    resources, and add a trust diagnostic when protected inputs exist. If trusted
        #    and project extensions are enabled, load project extensions.
        # 4. Seed the Codex runtime catalog from the on-disk cache when available, then pick
        #    the provider/model (CLI request, then replayed state, then session provider)
        #    and build the provider if none was given or the Codex model must be
        #    rediscovered (closing a replaced owned initial provider), updating staged
        #    initial entries to the final selection.
        # 5. Determine image support, compose tools (built-ins or override, plus extension
        #    tools, then the allow-list), build the system prompt (unless an explicit system
        #    was given), create the harness with replayed messages, carry over the UI bridge,
        #    and construct the session; record provider ownership; persist the Codex catalog
        #    if discovered.
        # 6. Persist tool-history repairs, apply catalogs, thinking override and sync, refresh
        #    runtime provider and model limits, bind extensions and attach their harness
        #    listener, and mark session_start and trust commit as pending.
        # Output: the session. Raises provider/config/trust/storage errors; on failure after
        # construction the session is closed before re-raising.
        # Side effects: storage reads/writes, network discovery, extension loading/imports,
        # possible trust prompts.
        """Load a coding session from append-only storage."""
        entries = await config.storage.read_all()
        pending_initial_entries: tuple[SessionEntry, ...] = ()
        if not entries:
            info = SessionInfoEntry(cwd=str(config.cwd))
            initial_model = _initial_model_for_config(config)
            model = ModelChangeEntry(
                parent_id=info.id,
                model=initial_model,
                provider=config.requested_provider or config.provider_name,
            )
            thinking = ThinkingLevelChangeEntry(
                parent_id=model.id,
                thinking_level=_initial_thinking_level_for_config(config, model=initial_model),
            )
            entries = [info, model, thinking]
            pending_initial_entries = (info, model, thinking)
        else:
            entries = _detach_missing_parents(entries)

        state = SessionState.from_entries(entries)
        unfiltered_resource_paths = resource_paths_with_cwd(config.resource_paths, config.cwd)

        # A runtime is cwd-bound because it may contain project registrations.
        # Always stage a fresh eligible-only runtime for a destination snapshot;
        # never reuse source-project code across replacement trust boundaries.
        previous_runtime = config.extension_runtime
        runtime_paths = unfiltered_resource_paths.paths or TauPaths(
            home=unfiltered_resource_paths.root,
            agents_home=unfiltered_resource_paths.agents_root or Path.home() / ".agents",
        )
        credential_store = FileCredentialStore(credentials_path(runtime_paths))
        injected_credentials = (
            previous_runtime.provider_credentials if previous_runtime is not None else None
        )
        extension_runtime = ExtensionRuntime(
            paths=runtime_paths,
            credentials=injected_credentials or credential_store,
            environment=(
                previous_runtime.provider_environment if previous_runtime is not None else None
            ),
            built_in_credentials=(
                previous_runtime.built_in_credentials if previous_runtime is not None else None
            ),
            built_in_http_client=(
                previous_runtime.built_in_http_client if previous_runtime is not None else None
            ),
            durable_providers=(
                config.provider_settings.providers if config.provider_settings else ()
            ),
        )
        extension_runtime.load(
            unfiltered_resource_paths,
            extra_paths=config.extension_paths,
            include_resource_dirs=config.extensions_enabled,
            include_project_dir=False,
        )

        coordinator = config.project_trust_coordinator or ProjectTrustCoordinator(
            ProjectTrustStore(
                unfiltered_resource_paths.paths or TauPaths(home=unfiltered_resource_paths.root)
            )
        )
        summary, trust_resolution = await coordinator.resolve(
            config.cwd,
            override=config.trust_override,
            default=config.trust_default,
            interactive=config.trust_interactive,
            prompt=config.trust_prompt,
            extension_deciders=(extension_runtime.decide_project_trust,),
            cache_result=False,
            persist=not config.defer_authoritative_writes,
        )
        canonical_cwd = summary.cwd.value
        resource_paths = resource_paths_with_project_trust(
            resource_paths_with_cwd(config.resource_paths, canonical_cwd),
            trusted=trust_resolution.trusted,
        )
        config = replace(
            config,
            cwd=canonical_cwd,
            resource_paths=resource_paths,
            project_trust_coordinator=coordinator,
            extension_runtime=extension_runtime,
        )
        resources = _load_session_resources(
            resource_paths,
            config.context_files,
            skills_enabled=config.skills_enabled,
            system_prompt_enabled=config.system is None,
            custom_system_prompt_explicit=config.custom_system_prompt is not None,
        )
        if summary.categories:
            resources = replace(
                resources,
                diagnostics=(
                    *resources.diagnostics,
                    ResourceDiagnostic(
                        kind="project-trust",
                        message=format_trust_diagnostic(summary, trust_resolution),
                        severity="info" if trust_resolution.trusted else "warning",
                    ),
                ),
            )

        if trust_resolution.trusted and config.project_extensions_enabled:
            extension_runtime.load(
                resource_paths,
                include_resource_dirs=True,
                include_project_dir=True,
                include_user_dir=False,
            )

        runtime_model_catalogs = dict(config.runtime_model_catalogs)
        if config.provider_settings is not None and "openai-codex" not in runtime_model_catalogs:
            codex_config = _provider_config_or_none(config.provider_settings, "openai-codex")
            if isinstance(codex_config, OpenAICodexProviderConfig):
                account_id = _codex_account_id(codex_config, credential_store)
                cached_catalog = cached_codex_model_catalog(
                    runtime_paths,
                    account_id=account_id,
                )
                if cached_catalog is not None:
                    runtime_model_catalogs["openai-codex"] = cached_catalog
        config = replace(config, runtime_model_catalogs=runtime_model_catalogs)
        selection_settings = _provider_settings_with_runtime_catalogs(
            config.provider_settings,
            runtime_model_catalogs,
        )
        selected_provider_name = (
            config.requested_provider or state.provider or config.session_provider_name
        )
        selected_model = config.requested_model or state.model
        rediscover_codex = False
        if selected_provider_name == "openai-codex" and config.provider_settings is not None:
            selected_config = _provider_config_or_none(
                config.provider_settings,
                selected_provider_name,
            )
            rediscover_codex = (
                isinstance(selected_config, OpenAICodexProviderConfig)
                and selected_model not in selected_config.models
            )
        preparation_settings = config.provider_settings
        startup_catalog: RuntimeModelCatalog | None = None
        effective_selected_config = (
            _provider_config_or_none(selection_settings, selected_provider_name)
            if selection_settings is not None
            else None
        )
        if (
            isinstance(effective_selected_config, OpenAICodexProviderConfig)
            and selected_model in effective_selected_config.models
        ):
            preparation_settings = selection_settings
        if config.provider is None or rediscover_codex:
            prepared = await _prepare_provider_selection(
                replace(config, provider_settings=preparation_settings),
                state=state,
                provider_registry=extension_runtime.provider_registry,
                credential_store=credential_store,
            )
            if config.provider is not None and config.owns_initial_provider:
                try:
                    await config.provider.aclose()  # type: ignore[attr-defined]
                except BaseException:
                    await prepared.provider.aclose()
                    raise
            startup_catalog = prepared.runtime_model_catalog
            if startup_catalog is not None:
                runtime_model_catalogs[prepared.provider_name] = startup_catalog
            config = replace(
                config,
                provider=prepared.provider,
                runtime_model_catalogs=runtime_model_catalogs,
                model=prepared.model,
                provider_name=prepared.provider_name,
                inference_provider=prepared.inference_provider,
                inference_provider_mode=prepared.inference_provider_mode,
                runtime_provider_config=prepared.runtime_provider_config,
                dynamic_provider=prepared.dynamic_provider,
                owns_initial_provider=True,
            )
            if pending_initial_entries:
                pending_initial_entries = tuple(
                    entry.model_copy(
                        update={"model": config.model, "provider": config.provider_name}
                    )
                    if isinstance(entry, ModelChangeEntry)
                    else entry
                    for entry in pending_initial_entries
                )
                # Initial fallback metadata was created before live discovery.
                # Replay the corrected entries so runtime and durable selection agree.
                state = SessionState.from_entries(list(pending_initial_entries))
        assert config.provider is not None
        active_model = _runtime_model_for_state(config, state)
        image_support = ImageSupportState(
            supported=_configured_model_supports_images(config, active_model)
        )
        base_tools = (
            config.tools
            if config.tools is not None
            else create_coding_tools(
                cwd=config.cwd,
                shell_command_prefix=config.shell_command_prefix,
                image_support=image_support,
            )
        )
        tools = _filter_tools(
            extension_runtime.compose_tools(base_tools), config.allowed_tool_names
        )
        system = (
            config.system
            if config.system is not None
            else build_system_prompt(
                BuildSystemPromptOptions(
                    cwd=config.cwd,
                    tools=tools,
                    skills=resources.skills,
                    custom_prompt=(
                        config.custom_system_prompt
                        if config.custom_system_prompt is not None
                        else resources.custom_system_prompt
                    ),
                    append_sections=_append_prompt_sections(
                        resources.append_system_prompts,
                        resources.append_system_prompt_paths,
                        config.append_system_prompt,
                    ),
                    context_files=resources.context_files,
                    extra_guidelines=extension_runtime.prompt_guidelines,
                    extra_sections=extension_runtime.sourced_prompt_sections,
                    custom_prompt_source=_custom_prompt_source(
                        explicit=config.custom_system_prompt is not None,
                        path=resources.custom_system_prompt_path,
                    ),
                )
            )
        )
        harness = AgentHarness(
            AgentHarnessConfig(
                provider=config.provider,
                model=active_model,
                system=system,
                tools=tools,
                session_id=config.session_id,
            ),
            messages=state.messages,
        )
        if previous_runtime is not None:
            extension_runtime.set_ui_bridge(previous_runtime.ui)
        session = cls(
            config,
            state=state,
            harness=harness,
            last_parent_id=_last_parent_id_from_state(state),
            skills=resources.skills,
            prompt_templates=resources.prompt_templates,
            context_files=resources.context_files,
            custom_system_prompt=resources.custom_system_prompt,
            custom_system_prompt_path=resources.custom_system_prompt_path,
            append_system_prompt=resources.append_system_prompt,
            append_system_prompts=resources.append_system_prompts,
            append_system_prompt_paths=resources.append_system_prompt_paths,
            resource_diagnostics=resources.diagnostics,
            command_registry=config.command_registry or extension_runtime.build_command_registry(),
            pending_initial_entries=pending_initial_entries,
            extension_runtime=extension_runtime,
            image_support=image_support,
            project_trust_resolution=trust_resolution,
        )
        if config.owns_initial_provider:
            # Ownership starts before any repair/discovery work so every
            # failure path has exactly one closer for the candidate.
            session._owned_providers.append(config.provider)  # type: ignore[arg-type]
        if startup_catalog is not None and config.provider_name == "openai-codex":
            session._persist_codex_model_catalog(
                config.provider,
                startup_catalog,
                (
                    config.runtime_provider_config
                    if isinstance(config.runtime_provider_config, OpenAICodexProviderConfig)
                    else None
                ),
            )
        await session._persist_active_tool_history_repairs()
        try:
            session._apply_runtime_model_catalogs()
            session._apply_thinking_level_override()
            session._sync_thinking_level_to_active_model()
            if not config.owns_initial_provider and config.provider is not None:
                session._refresh_runtime_provider()
            await session._refresh_runtime_model_limits()
            extension_runtime.bind(session)
            # Attach to session._harness, not the local `harness`:
            # Tool-history repair above updates the active harness before extension
            # listeners attach.
            extension_runtime.attach_harness_listener(session._harness.subscribe)
            # session_start is deferred: hosts emit it via emit_pending_session_start()
            # after installing their UI bridge.
            session._session_start_pending = True
            session._project_trust_commit_pending = not trust_resolution.cancelled
        except BaseException:
            # Once constructed, this session explicitly owns every candidate
            # provider until load returns it to the caller.
            await _finish_aborted_session_close(session)
            raise
        return session

    @property
    def cwd(self) -> Path:
        # [FUNCTION DOC]
        # Property: the canonical session working directory.
        """Return the session working directory."""
        return self._config.cwd

    @property
    def model(self) -> str:
        # [FUNCTION DOC]
        # Property: the active model id, or the previewed model while a preview is active.
        """Return the active model for this session."""
        if self._preview_model is not None:
            return self._preview_model.model
        return self._harness.config.model

    @property
    def provider_name(self) -> str:
        # [FUNCTION DOC]
        # Property: the active provider name, or the previewed provider during a preview.
        """Return the active provider name."""
        if self._preview_model is not None:
            return self._preview_model.provider_name
        return self._provider_name

    @property
    def provider(self) -> ModelProvider:
        # [FUNCTION DOC]
        # Property: the runtime model provider currently configured on the harness.
        """Return the currently active runtime provider."""
        return self._harness.config.provider

    @property
    def inference_provider(self) -> str | None:
        # [FUNCTION DOC]
        # Property: the pinned Hugging Face backing provider name, or None.
        """Return the pinned Hugging Face backing provider, if any."""
        return self._inference_provider

    @property
    def inference_provider_mode(self) -> InferenceProviderMode:
        # [FUNCTION DOC]
        # Property: "automatic" or "fixed" Hugging Face routing mode.
        """Return whether Hugging Face routing is automatic or explicitly fixed."""
        return self._inference_provider_mode

    @property
    def _active_dynamic_provider(self) -> DynamicProvider | None:
        # [FUNCTION DOC]
        # HELPER: Property: the effective dynamic provider definition registered for the
        # active provider name, or None if the active provider is not dynamic.
        effective = self._provider_registry.effective(self.provider_name)
        if effective is None or not isinstance(effective.definition, DynamicProvider):
            return None
        return effective.definition

    @property
    def _active_dynamic_model(self) -> ProviderModel | None:
        # [FUNCTION DOC]
        # HELPER: Property: the dynamic provider's model entry matching the active model id,
        # or None.
        dynamic = self._active_dynamic_provider
        if dynamic is None:
            return None
        return next((model for model in dynamic.models if model.id == self.model), None)

    @property
    def available_providers(self) -> tuple[str, ...]:
        # [FUNCTION DOC]
        # Property: provider names that can be used: configured providers with usable
        # credentials, then dynamic providers that have models (or are the current
        # provider), without duplicates. Falls back to just the current provider name.
        """Return provider names Tau can call with available credentials."""
        names: list[str] = []
        if self._provider_settings is not None:
            names.extend(provider.name for provider in self._usable_provider_configs())
        for effective in self._provider_registry.effective_providers():
            if (
                isinstance(effective.definition, DynamicProvider)
                and (effective.definition.models or effective.definition.id == self._provider_name)
                and effective.definition.id not in names
            ):
                names.append(effective.definition.id)
        return tuple(names) or (self._provider_name,)

    def provider_config(self, provider_name: str) -> ProviderConfig | None:
        # [FUNCTION DOC]
        # Return the configuration for a provider name from the provider settings (None if
        # unknown); without settings, return the runtime provider config only for the active
        # provider.
        """Return configured metadata for a provider available to this session."""
        if self._provider_settings is None:
            return self._runtime_provider_config if provider_name == self.provider_name else None
        try:
            return self._provider_settings.get_provider(provider_name)
        except ProviderConfigError:
            return None

    @property
    def available_models(self) -> tuple[str, ...]:
        # [FUNCTION DOC]
        # Property: model ids for the active provider: the dynamic provider's models; just
        # the current model without settings or for an unknown provider; an empty tuple if
        # the configured provider has no usable credentials; else its configured models.
        """Return model names for the active provider when it is usable."""
        dynamic = self._active_dynamic_provider
        if dynamic is not None:
            return tuple(model.id for model in dynamic.models)
        if self._provider_settings is None:
            return (self.model,)
        try:
            provider = self._provider_settings.get_provider(self.provider_name)
        except ProviderConfigError:
            return (self.model,)
        if not self._provider_is_usable(provider):
            return ()
        return provider.models

    @property
    def available_model_choices(self) -> tuple[ModelChoice, ...]:
        # [FUNCTION DOC]
        # Property: all selectable provider/model pairs: every dynamic provider's models
        # first, then usable configured providers' models (skipping names shadowed by
        # dynamic providers). Without settings and no choices, returns the current pair.
        """Return provider/model choices from the effective runtime view."""
        choices: list[ModelChoice] = []
        dynamic_ids: set[str] = set()
        for effective in self._provider_registry.effective_providers():
            if isinstance(effective.definition, DynamicProvider):
                dynamic = effective.definition
                dynamic_ids.add(dynamic.id)
                choices.extend(ModelChoice(dynamic.id, model.id) for model in dynamic.models)
        if self._provider_settings is not None:
            choices.extend(
                ModelChoice(provider.name, model)
                for provider in self._usable_provider_configs()
                if provider.name not in dynamic_ids
                for model in provider.models
            )
        if not choices and self._provider_settings is None:
            return (ModelChoice(provider_name=self._provider_name, model=self.model),)
        return tuple(choices)

    @property
    def scoped_model_choices(self) -> tuple[ModelChoice, ...]:
        # [FUNCTION DOC]
        # Property: the user's saved scoped models (from settings) that are currently
        # available, plus references to trusted built-in providers with stable scoped
        # references even if not currently available. Empty without settings.
        """Return scoped references, including inert trusted built-in references."""
        if self._provider_settings is None:
            return ()
        available = set(self.available_model_choices)
        choices: list[ModelChoice] = []
        for item in self._provider_settings.scoped_models:
            choice = ModelChoice(provider_name=item.provider, model=item.model)
            if choice in available or self._stable_dynamic_scoped_provider(item.provider):
                choices.append(choice)
        return tuple(choices)

    @property
    def has_stale_active_model(self) -> bool:
        # [FUNCTION DOC]
        # Property: True when the committed provider/model pair is no longer among the
        # available choices.
        """Whether the committed runtime model disappeared from the picker snapshot."""
        return ModelChoice(self._provider_name, self._harness.config.model) not in (
            self.available_model_choices
        )

    @property
    def unavailable_scoped_model_choices(self) -> tuple[ModelChoice, ...]:
        # [FUNCTION DOC]
        # Property: scoped choices that are not currently available (inert stable refs).
        """Return persisted references that have no current provider snapshot row."""
        available = set(self.available_model_choices)
        return tuple(choice for choice in self.scoped_model_choices if choice not in available)

    def _stable_dynamic_scoped_provider(self, provider_name: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: True if any registry layer for the provider comes from a built-in source
        # and opts into stable scoped references.
        return any(
            layer.token.source_id.startswith("built-in:")
            and layer.provider.stable_scoped_references
            for layer in self._provider_registry.layers(provider_name)
        )

    def _effective_stable_dynamic_scoped_provider(self, provider_name: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: True if the provider's effective (winning) registration is a built-in
        # dynamic provider that opts into stable scoped references.
        effective = self._provider_registry.effective(provider_name)
        return bool(
            effective is not None
            and effective.source_id.startswith("built-in:")
            and isinstance(effective.definition, DynamicProvider)
            and effective.definition.stable_scoped_references
        )

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        # [FUNCTION DOC]
        # Property: tuple of the agent tools currently configured on the harness.
        """Return the tools available to the agent."""
        return tuple(self._harness.config.tools)

    @property
    def extension_tool_sources(self) -> dict[str, str]:
        # [FUNCTION DOC]
        # Property: mapping from active extension-provided tool name to owning extension.
        """Map active extension-provided tools to their owning extension."""
        return self._extension_runtime.extension_tool_sources

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        # [FUNCTION DOC]
        # Property: the current in-memory transcript held by the harness.
        """Return the restored/current transcript."""
        return self._harness.messages

    @property
    def state(self) -> SessionState:
        # [FUNCTION DOC]
        # Property: the last replayed durable session state.
        """Return the last replayed durable session state."""
        return self._state

    async def session_entries(self) -> tuple[SessionEntry, ...]:
        # [FUNCTION DOC]
        # Return all stored session entries (async read) as a tuple for inspection.
        """Return append-only entries for frontend session inspection."""
        return tuple(await self._read_session_entries())

    async def tree_choices(self) -> tuple[SessionTreeChoice, ...]:
        # [FUNCTION DOC]
        # Build rows for the session tree picker from stored entries: lay out entries in
        # tree order with branch indentation, resolve bookmark labels and their timestamps,
        # mark the branchable entry representing the active leaf, flag tool-call rows, and
        # keep only branchable entries. Returns a tuple of choices. Side effects: storage read.
        """Return branchable session entries for a tree picker."""
        entries = await self._read_session_entries()
        ordered_entries, branch_indents = _tree_layout(entries)
        labels_by_id, label_timestamps_by_id = _resolved_labels(entries)
        active_choice_id = _active_branchable_entry_id(entries, self._state.active_leaf_id)
        return tuple(
            SessionTreeChoice(
                entry_id=entry.id,
                label=_tree_choice_label(entry, branch_indent=branch_indents.get(entry.id, 0)),
                active=entry.id == active_choice_id,
                is_tool_call=_is_tool_call_tree_entry(entry),
                bookmark_label=labels_by_id.get(entry.id),
                label_timestamp=label_timestamps_by_id.get(entry.id),
            )
            for entry in ordered_entries
            if _is_branchable_tree_entry(entry)
        )

    async def set_label(self, target_id: str, label: str | None) -> LabelEntry:
        # [FUNCTION DOC]
        # Set or clear a bookmark label on an existing entry by appending a label entry on
        # the active path (blank or None label clears it). Requires the session to be idle
        # (RuntimeError otherwise) and flushes pending message writes first.
        # Inputs: target entry id, label text or None. Output: the appended label entry.
        # Raises ValueError for an unknown target id. Side effects: storage append, parent
        # chain advance, state replay.
        """Set or clear the bookmark on an existing session entry."""
        if self._harness.is_running:
            raise RuntimeError(TREE_RUNNING_MESSAGE)
        await self._flush_pending_message_writes(context=self._diagnostic_context())
        entries = await self._read_session_entries()
        if target_id not in {entry.id for entry in entries}:
            raise ValueError(f"Unknown session entry: {target_id}")
        normalized = label.strip() if label is not None else ""
        entry = LabelEntry(
            parent_id=self._last_parent_id,
            target_id=target_id,
            label=normalized or None,
        )
        await self._append_session_entry(entry)
        self._last_parent_id = entry.id
        await self._refresh_persisted_state(leaf_id=entry.id)
        return entry

    async def branch_to_entry(
        self,
        entry_id: str,
        *,
        summarize: bool = False,
        custom_instructions: str | None = None,
        replace_instructions: bool = False,
    ) -> SessionTreeBranchResult:
        # [FUNCTION DOC]
        # Move the active leaf to an earlier entry without deleting history.
        # Inputs: entry id; keyword summarize flag, custom summary instructions, and whether
        # they replace the default instructions.
        # Behavior: requires idle (RuntimeError), flushes pending writes, validates the
        # entry exists and is branchable (ValueError otherwise). With summarize, messages on
        # the active path after the entry are summarized by the model and stored as a
        # branch summary entry child of the target, which becomes the new leaf. Without
        # summarize, choosing a user message moves to its parent and returns its text as an
        # input prefill (so the user can edit and resend). Clears previews, replays state
        # for the new leaf, repairs/persists tool history or replaces harness messages,
        # resets context cache, recomputes thinking level for the active model, and
        # refreshes the runtime provider.
        # Output: a result with a status message and optional prefill.
        # Side effects: possible model call and storage append; harness and state mutation.
        """Move the active leaf to a previous entry, preserving existing history."""
        if self._harness.is_running:
            raise RuntimeError(TREE_RUNNING_MESSAGE)
        await self._flush_pending_message_writes(context=self._diagnostic_context())
        entries = await self._read_session_entries()
        by_id = {entry.id: entry for entry in entries}
        if entry_id not in by_id:
            raise ValueError(f"Unknown session entry: {entry_id}")
        selected_entry = by_id[entry_id]
        if not _is_branchable_tree_entry(selected_entry):
            raise ValueError(f"Session entry cannot be branched from: {entry_id}")

        target_id: str | None = entry_id
        input_prefill: str | None = None
        summary_entry: BranchSummaryEntry | None = None
        if summarize:
            abandoned_messages = _messages_after_entry_on_active_path(
                entries,
                entry_id,
                self._last_parent_id,
            )
            if abandoned_messages:
                generated = await self._summarize_branch_messages(
                    abandoned_messages,
                    custom_instructions=custom_instructions,
                    replace_instructions=replace_instructions,
                )
                summary_entry = BranchSummaryEntry(
                    parent_id=entry_id,
                    branch_root_id=entry_id,
                    summary=generated.text,
                    usage=generated.usage,
                    provider=generated.provider,
                    model=generated.model,
                    response_provider=generated.response_provider,
                )
                await self._append_session_entry(summary_entry)
                target_id = summary_entry.id
        elif selected_entry.type == "message" and isinstance(selected_entry.message, UserMessage):
            target_id = selected_entry.parent_id
            input_prefill = selected_entry.message.text

        self._preview_model = None
        self._preview_thinking = None
        self._last_parent_id = target_id

        # Plain navigation is in-memory only. A summary above is the sole write,
        # and becomes the file-order tip when present.
        await self._refresh_persisted_state(leaf_id=target_id)
        history_repair = await self._persist_active_tool_history_repairs()
        if history_repair is None:
            self._harness.replace_messages(self._state.messages)
        self._invalidate_context_usage_cache()
        self._thinking_level = _state_thinking_level(
            self._state,
            default=_default_thinking_level_for_active_model(self),
        )
        self._sync_thinking_level_to_active_model()
        self._refresh_runtime_provider()
        suffix = " with branch summary" if summary_entry is not None else ""
        if history_repair is not None:
            suffix += " and repaired malformed tool history"
        if input_prefill is not None:
            return SessionTreeBranchResult(
                message=f"Branched session before {entry_id}{suffix}.",
                input_prefill=input_prefill,
            )
        return SessionTreeBranchResult(message=f"Branched session at {target_id}{suffix}.")

    @property
    def thinking_level(self) -> ThinkingLevel:
        # [FUNCTION DOC]
        # Property: the active thinking level, or the previewed level during a preview.
        """Return the active thinking mode for future turns."""
        if self._preview_thinking is not None:
            return self._preview_thinking
        return self._thinking_level

    @property
    def available_thinking_levels(self) -> tuple[ThinkingLevel, ...]:
        # [FUNCTION DOC]
        # Property: thinking levels supported by the active provider/model: the dynamic
        # model's declared levels (empty if unknown or undeclared); all levels when there
        # are no provider settings; empty if the provider config is missing; otherwise the
        # levels the provider config allows for the model.
        """Return thinking modes supported by the active provider/model."""
        dynamic = self._active_dynamic_provider
        if dynamic is not None:
            model = self._active_dynamic_model
            if model is None:
                return ()
            return model.thinking_levels or ()
        if self._provider_settings is None:
            return THINKING_LEVELS
        provider = self._selected_provider_config()
        if provider is None:
            return ()
        return provider_thinking_levels(provider, model=self.model)

    @property
    def thinking_unavailable_reason(self) -> str | None:
        # [FUNCTION DOC]
        # Property: None when thinking levels are available; otherwise a human explanation
        # (dynamic model metadata missing, levels undeclared or empty, provider settings
        # unavailable, or the provider config's own reason).
        """Return why thinking controls are unavailable for the active model."""
        if self.available_thinking_levels:
            return None
        dynamic = self._active_dynamic_provider
        if dynamic is not None:
            model = self._active_dynamic_model
            if model is None:
                return f"{self.provider_name}:{self.model} metadata is not available"
            if model.thinking_levels is None:
                return (
                    f"{self.provider_name}:{self.model} does not declare configurable "
                    "thinking levels"
                )
            return f"{self.provider_name}:{self.model} declares no configurable thinking levels"
        provider = self._selected_provider_config()
        if provider is None:
            return "Active provider settings are not available"
        return provider_thinking_unavailable_reason(provider, model=self.model)

    @property
    def storage(self) -> SessionStorage:
        # [FUNCTION DOC]
        # Property: the backing session storage.
        """Return the backing session storage."""
        return self._config.storage

    async def export(
        self,
        destination: Path | None = None,
        *,
        format: str | None = None,
    ) -> Path:
        # [FUNCTION DOC]
        # Export the session to a file artifact and return its path.
        # Inputs: optional destination path; keyword format (otherwise inferred from the
        # destination suffix, default "html"; normalized and validated).
        # Reads all entries, resolves the output path (default location derived from the
        # session file and cwd), and writes the artifact with a title, source (session file
        # path or id) and the current system prompt. Raises on invalid format or I/O errors.
        # Side effects: storage read and file write.
        """Export the current session to a user-facing artifact."""
        entries = await self._read_session_entries()
        session_path = _storage_path(self._config.storage)
        export_format = normalize_export_format(
            format or (destination.suffix.removeprefix(".") if destination else "html")
        )
        output_path = _resolve_export_destination(
            destination,
            cwd=self.cwd,
            session_path=session_path,
            format=export_format,
        )
        return export_session_artifact(
            entries,
            output_path,
            title=_session_export_title(self),
            source=str(session_path) if session_path is not None else self.session_id,
            format=export_format,
            system_prompt=self.system_prompt,
        )

    @property
    def skills(self) -> tuple[Skill, ...]:
        # [FUNCTION DOC]
        # Property: loaded skills (empty when skills are disabled or untrusted).
        """Return loaded skills."""
        return self._skills

    @property
    def prompt_templates(self) -> tuple[PromptTemplate, ...]:
        # [FUNCTION DOC]
        # Property: loaded prompt templates.
        """Return loaded prompt templates."""
        return self._prompt_templates

    @property
    def context_files(self) -> tuple[ProjectContextFile, ...]:
        # [FUNCTION DOC]
        # Property: active project context files.
        """Return active project context files."""
        return self._context_files

    @property
    def system_prompt_files(self) -> tuple[Path, ...]:
        # [FUNCTION DOC]
        # Property: discovered custom system prompt path (if any) followed by append system
        # prompt paths.
        """Return active discovered system-prompt resource files."""
        custom_paths = (
            (self._custom_system_prompt_path,)
            if self._custom_system_prompt_path is not None
            else ()
        )
        return (*custom_paths, *self._append_system_prompt_paths)

    @property
    def context_token_estimate(self) -> int:
        # [FUNCTION DOC]
        # Property: estimated total tokens of the active provider context.
        """Return the best available token count for the active provider context."""
        return self.context_usage.total_tokens

    @property
    def has_provider_context_usage(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether the estimate is anchored by real provider-reported usage.
        """Return whether valid provider usage anchors the active context count."""
        return self.context_usage.uses_provider_usage

    @property
    def context_usage(self) -> ContextUsageEstimate:
        # [FUNCTION DOC]
        # Property: structured context usage estimate for the harness system prompt,
        # messages and tools; computed lazily and cached until invalidated.
        """Return structured context accounting for the active provider context."""
        if self._context_usage_cache is None:
            self._context_usage_cache = estimate_context_usage(
                system=self._harness.config.system,
                messages=self._harness.messages,
                tools=tuple(self._harness.config.tools),
            )
        return self._context_usage_cache

    @property
    def system_prompt(self) -> str:
        # [FUNCTION DOC]
        # Property: the effective system prompt currently configured on the harness.
        """Return the effective system prompt sent to the model."""
        return self._harness.config.system

    @property
    def system_prompt_inspection(self) -> SystemPromptInspection:
        # [FUNCTION DOC]
        # Property: the effective prompt with source attribution. With an explicit system
        # override, returns a single "system" source. Otherwise rebuilds the inspection from
        # current resources and extension sections; if its text differs from the live
        # prompt (e.g. modified at runtime), returns a single "runtime" source instead.
        """Return the effective prompt with source attribution for local inspection."""
        if self._config.system is not None:
            return SystemPromptInspection(
                text=self.system_prompt,
                sources=(
                    SystemPromptSource(
                        kind="system",
                        label="System prompt override",
                        source="CodingSessionConfig.system",
                        content=self.system_prompt,
                    ),
                ),
            )
        inspection = build_system_prompt_inspection(
            BuildSystemPromptOptions(
                cwd=self.cwd,
                tools=self.tools,
                skills=self.skills,
                custom_prompt=(
                    self._config.custom_system_prompt
                    if self._config.custom_system_prompt is not None
                    else self._custom_system_prompt
                ),
                append_sections=_append_prompt_sections(
                    self._append_system_prompts,
                    self._append_system_prompt_paths,
                    self._config.append_system_prompt,
                ),
                context_files=self.context_files,
                extra_guidelines=self._extension_runtime.prompt_guidelines,
                extra_sections=self._extension_runtime.sourced_prompt_sections,
                custom_prompt_source=_custom_prompt_source(
                    explicit=self._config.custom_system_prompt is not None,
                    path=self._custom_system_prompt_path,
                ),
            )
        )
        if inspection.text == self.system_prompt:
            return inspection
        return SystemPromptInspection(
            text=self.system_prompt,
            sources=(
                SystemPromptSource(
                    kind="runtime",
                    label="Effective system prompt",
                    source="active Tau session (runtime-composed)",
                    content=self.system_prompt,
                ),
            ),
        )

    @property
    def auto_compact_token_threshold(self) -> int | None:
        # [FUNCTION DOC]
        # Property: the auto-compaction token threshold, or None when disabled. Uses the
        # configured threshold, else the live model limits' effective limit (if they belong
        # to the active provider/model), else a value derived from the context window.
        """Return the effective automatic compaction threshold, if any."""
        if not self._auto_compact_enabled:
            return None
        if self._auto_compact_token_threshold is not None:
            return self._auto_compact_token_threshold
        if self._runtime_model_limits_key == (self.provider_name, self.model):
            limits = self._runtime_model_limits
            if limits is not None:
                return limits.effective_auto_compact_token_limit
        return auto_compaction_threshold_for_context_window(self.context_window_tokens)

    def set_auto_compaction_enabled(self, enabled: bool) -> None:
        # [FUNCTION DOC]
        # Enable or disable automatic compaction for future turns; updates both the flag
        # and the stored config.
        """Enable or disable automatic compaction for future turns."""
        self._auto_compact_enabled = enabled
        self._config = replace(self._config, auto_compact_enabled=enabled)

    @property
    def auto_compaction_enabled(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether automatic compaction is enabled.
        """Return whether automatic compaction is enabled."""
        return self._auto_compact_enabled

    @property
    def context_window_tokens(self) -> int:
        # [FUNCTION DOC]
        # Property: the active model's context window: live discovered limits when they
        # match the active provider/model, else the provider config's per-model value, else
        # the default context window.
        """Return the active model's discovered or configured context window."""
        if self._runtime_model_limits_key == (self.provider_name, self.model):
            limits = self._runtime_model_limits
            if limits is not None:
                return limits.context_window
        provider = self._active_provider_config()
        if provider is None:
            return DEFAULT_CONTEXT_WINDOW_TOKENS
        return provider.context_windows.get(self.model, DEFAULT_CONTEXT_WINDOW_TOKENS)

    @property
    def context_window_source(self) -> str:
        # [FUNCTION DOC]
        # Property: "provider live catalog" when live limits apply, else "configured
        # catalog".
        """Return where the active context-window limit came from."""
        if (
            self._runtime_model_limits_key == (self.provider_name, self.model)
            and self._runtime_model_limits is not None
        ):
            return "provider live catalog"
        return "configured catalog"

    @property
    def model_limits_discovery_error(self) -> str | None:
        # [FUNCTION DOC]
        # Property: the last non-fatal live model-limit discovery error message, or None.
        """Return the last non-fatal live model-limit discovery error."""
        return self._model_limits_discovery_error

    @property
    def command_registry(self) -> CommandRegistry:
        # [FUNCTION DOC]
        # Property: the slash-command registry.
        """Return the slash-command registry used by this session."""
        return self._command_registry

    @property
    def project_trust_resolution(self) -> ProjectTrustResolution | None:
        # [FUNCTION DOC]
        # Property: the project trust resolution for this cwd, or None.
        """Return this cwd's completed project-input trust resolution."""
        return self._project_trust_resolution

    @property
    def theme_dirs(self) -> tuple[Path, ...]:
        # [FUNCTION DOC]
        # Property: theme directories allowed by the trust-filtered resource paths.
        """Return theme directories permitted by this session's trust snapshot."""
        return self._resource_paths.themes_dirs

    @property
    def resource_diagnostics(self) -> tuple[ResourceDiagnostic, ...]:
        # [FUNCTION DOC]
        # Property: resource diagnostics, plus each trust resolution diagnostic as an
        # error-severity "project-trust" diagnostic, plus extension runtime diagnostics.
        """Return non-fatal resource and extension diagnostics."""
        trust_diagnostics: tuple[ResourceDiagnostic, ...] = ()
        if self._project_trust_resolution is not None:
            trust_diagnostics = tuple(
                ResourceDiagnostic(kind="project-trust", message=message, severity="error")
                for message in self._project_trust_resolution.diagnostics
            )
        return self._resource_diagnostics + trust_diagnostics + self._extension_runtime.diagnostics

    @property
    def extension_runtime(self) -> ExtensionRuntime:
        # [FUNCTION DOC]
        # Property: the bound extension runtime.
        """Return the extension runtime bound to this session."""
        return self._extension_runtime

    @property
    def extension_names(self) -> tuple[str, ...]:
        # [FUNCTION DOC]
        # Property: loaded extension names in load order.
        """Return loaded extension names in load order."""
        return self._extension_runtime.extension_names

    @property
    def session_stats(self) -> SessionStats:
        # [FUNCTION DOC]
        # Property: cumulative activity and billed usage statistics for the active branch,
        # cached while the state and config objects are unchanged (identity check).
        """Return cumulative activity and billed usage for the active branch."""
        cached = self._session_stats_cache
        if cached is not None and cached[0] is self._state and cached[1] is self._config:
            return cached[2]
        stats = calculate_session_stats(self._state.entries, pricing=self._pricing_for_response)
        self._session_stats_cache = (self._state, self._config, stats)
        return stats

    def _pricing_for_response(
        self,
        provider_name: str,
        model: str,
        input_tokens: int,
    ) -> dict[str, float] | None:
        # [FUNCTION DOC]
        # HELPER: Pricing callback for stats: find the provider config for the name and the
        # model's metadata; return the cost of the first cost tier whose input-token cap
        # covers the request (or is uncapped), else the base cost, else None. Returns a new
        # mapping from cost component to price.
        provider = _provider_config_for_name(self._config, provider_name)
        if (
            provider is None
            or provider.name != provider_name
            or not hasattr(provider, "model_metadata")
        ):
            return None
        metadata = provider.model_metadata.get(model)
        if metadata is None:
            return None
        for tier in metadata.cost_tiers:
            if tier.max_input_tokens is None or input_tokens <= tier.max_input_tokens:
                return dict(tier.cost)
        return dict(metadata.cost) if metadata.cost else None

    async def emit_pending_session_start(self) -> None:
        # [FUNCTION DOC]
        # Emit the deferred extension "session_start" event (reason "startup") once, then
        # commit the staged project trust resolution. No-op if nothing is pending.
        # Side effects: runs extension handlers, may write the trust store.
        """Emit the `session_start` deferred by `load`, once per session.

        Hosts call this after installing their UI bridge so `session_start`
        handlers can use notifications and dialogs (Pi's ordering: the UI
        starts before extensions initialize). Idempotent; a no-op for
        sessions that adopted an already-started extension runtime.
        """
        if not self._session_start_pending:
            return
        await self._extension_runtime.emit_session_start("startup")
        self._commit_project_trust_resolution()
        self._session_start_pending = False

    def _commit_project_trust_resolution(self) -> None:
        # [FUNCTION DOC]
        # HELPER: If a trust commit is pending, publish the resolution through the
        # coordinator for the cwd (writing deferred decisions), then clear the flag.
        """Publish staged trust only after the session becomes live."""
        if not self._project_trust_commit_pending:
            return
        coordinator = self._config.project_trust_coordinator
        resolution = self._project_trust_resolution
        if coordinator is not None and resolution is not None:
            coordinator.commit(CanonicalProjectPath(self.cwd), resolution)
        self._project_trust_commit_pending = False

    def queue_steering_message(
        self,
        content: str,
        *,
        custom_type: str | None = None,
        details: dict[str, JSONValue] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Queue a message to steer the current run: a custom message when a custom type is
        # given (with optional details), else a user message. Side effects: harness queue.
        """Queue a steering user message (extension runtime seam)."""
        message: AgentMessage = (
            CustomMessage(custom_type=custom_type, content=content, details=details)
            if custom_type is not None
            else UserMessage(content=content)
        )
        self._harness.steer_message(message)

    def queue_follow_up_message(
        self,
        content: str,
        *,
        custom_type: str | None = None,
        details: dict[str, JSONValue] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Queue a follow-up message delivered after the current run finishes: custom or
        # user message as above. Side effects: harness queue.
        """Queue a follow-up user message (extension runtime seam)."""
        message: AgentMessage = (
            CustomMessage(custom_type=custom_type, content=content, details=details)
            if custom_type is not None
            else UserMessage(content=content)
        )
        self._harness.follow_up_message(message)

    async def append_custom_entry(self, namespace: str, data: dict[str, JSONValue]) -> None:
        # [FUNCTION DOC]
        # Persist an extension-owned custom data entry on the active branch (advancing the
        # parent chain so it is replayed on resume) and refresh state.
        # Inputs: namespace and JSON data. Side effects: storage append, state replay.
        """Persist an extension-owned custom entry on the active branch path.

        The entry advances the append-only tree parent chain so it stays on the
        replayed root-to-leaf path (off-path custom entries would be invisible
        to `SessionState` after resume).
        """
        entry = CustomEntry(parent_id=self._last_parent_id, namespace=namespace, data=data)
        await self._append_session_entry(entry)
        self._last_parent_id = entry.id
        await self._refresh_persisted_state(leaf_id=entry.id)

    @property
    def session_id(self) -> str | None:
        # [FUNCTION DOC]
        # Property: the session manager id, if the session is indexed.
        """Return this session's manager id, if indexed."""
        return self._config.session_id

    @property
    def session_title(self) -> str | None:
        # [FUNCTION DOC]
        # Property: the indexed title from the session manager, or None if not indexed,
        # no manager, or no record. Side effects: manager lookup.
        """Return this session's indexed human-friendly title, if named."""
        if self._config.session_id is None or self._config.session_manager is None:
            return None
        record = self._config.session_manager.get_session(self._config.session_id)
        if record is None:
            return None
        return record.title

    @property
    def session_name(self) -> str | None:
        # [FUNCTION DOC]
        # Property: alias of session_title.
        """Return this session's indexed human-friendly name, if named."""
        return self.session_title

    @property
    def session_manager(self) -> SessionManager | None:
        # [FUNCTION DOC]
        # Property: the session manager, if any.
        """Return the session manager, if available."""
        return self._config.session_manager

    @property
    def is_running(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether an agent run is active.
        """Return whether this session currently has an active agent run."""
        return self._harness.is_running

    def _require_idle(self, operation: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Raise RuntimeError naming the operation if an agent run is active.
        """Reject replacement-like operations until an active turn is drained."""
        if self._harness.is_running:
            raise RuntimeError(
                f"Cannot {operation} while Tau is working. "
                "Press Escape to interrupt and wait for it to finish."
            )

    @property
    def queued_message_count(self) -> int:
        # [FUNCTION DOC]
        # Property: number of queued steering plus follow-up messages.
        """Return the number of queued steering and follow-up messages."""
        return self._harness.pending_message_count

    @property
    def queued_messages(self) -> QueuedMessages:
        # [FUNCTION DOC]
        # Property: the harness's queued steering and follow-up messages.
        """Return queued steering and follow-up messages."""
        return self._harness.queued_messages

    @property
    def queued_steering_messages(self) -> tuple[str, ...]:
        # [FUNCTION DOC]
        # Property: visible text of each queued steering message, for UI display.
        """Return queued steering message text for UI display."""
        return tuple(message_text(message) for message in self._harness.queued_messages.steering)

    @property
    def queued_follow_up_messages(self) -> tuple[str, ...]:
        # [FUNCTION DOC]
        # Property: visible text of each queued follow-up message, for UI display.
        """Return queued follow-up message text for UI display."""
        return tuple(message_text(message) for message in self._harness.queued_messages.follow_up)

    @property
    def last_diagnostic_log_path(self) -> Path | None:
        # [FUNCTION DOC]
        # Property: path of the last agent-call diagnostic log written, or None.
        """Return the last diagnostic log path written by this session."""
        return self._last_diagnostic_log_path

    def set_project_trust_prompt(self, prompt: TrustPrompt) -> None:
        # [FUNCTION DOC]
        # Install the frontend's trust prompt and mark trust resolution as interactive for
        # later reloads/replacements. Mutates the stored config.
        """Install the active frontend's trust prompt for reload/replacement."""
        self._config = replace(
            self._config,
            trust_interactive=True,
            trust_prompt=prompt,
        )

    def cancel(self) -> None:
        # [FUNCTION DOC]
        # Request cancellation of the running agent turn, if any (delegates to harness).
        """Cancel the currently running agent turn, if any."""
        self._harness.cancel()

    def queue_update_event(self) -> QueueUpdateEvent:
        # [FUNCTION DOC]
        # Return a queue-update event carrying the current steering and follow-up texts.
        """Return the current queue state as a coding-session event."""
        return QueueUpdateEvent(
            steering=self.queued_steering_messages,
            follow_up=self.queued_follow_up_messages,
        )

    def clear_queued_messages(self) -> QueuedMessages:
        # [FUNCTION DOC]
        # Clear both message queues and return what was removed.
        """Clear queued steering and follow-up messages."""
        return self._harness.clear_queues()

    def pop_latest_follow_up_message(self) -> str | None:
        # [FUNCTION DOC]
        # Remove the most recently queued follow-up and return its text, or None.
        """Remove and return the most recently queued follow-up message."""
        message = self._harness.pop_latest_follow_up()
        return None if message is None else message_text(message)

    def pop_latest_steering_message(self) -> str | None:
        # [FUNCTION DOC]
        # Remove the most recently queued steering message and return its text, or None.
        """Remove and return the most recently queued steering message."""
        message = self._harness.pop_latest_steering()
        return None if message is None else message_text(message)

    def set_model(self, model: str) -> None:
        # [FUNCTION DOC]
        # Synchronously switch the active model within the current provider: validate it
        # against the provider config (ProviderConfigError if not configured), set it on the
        # harness, recompute the Hugging Face inference provider and mode, coerce the
        # thinking level, rebuild the runtime provider, update image support, save it as the
        # default model in settings, and update the session index record.
        # Side effects: settings file write, session index write, provider construction.
        # KNOWN ISSUE: unlike the async selection path, no model-change entry is appended to
        # the session transcript, so resuming the session may restore the previous model.
        """Switch the active model for future turns and make it the default."""
        provider = self._active_provider_config()
        if provider is not None:
            validate_provider_model(provider, model)
        self._harness.config.model = model
        self._inference_provider = _configured_inference_provider(provider, model)
        self._inference_provider_mode = _configured_inference_provider_mode(provider, model)
        self._sync_thinking_level_to_active_model()
        self._refresh_runtime_provider()
        self._sync_image_support()
        self._persist_default_model_choice()
        if self._config.session_id is not None and self._config.session_manager is not None:
            self._config.session_manager.touch_session(
                self._config.session_id,
                model=model,
                provider_name=self.provider_name,
                inference_provider=self._inference_provider,
                inference_provider_mode=self._inference_provider_mode,
                preserve_inference_provider=False,
            )

    async def apply_startup_model_override(self, model: str) -> None:
        # [FUNCTION DOC]
        # Activate an explicit startup model (e.g. from the CLI) before the next turn:
        # validate it, return early if already active, otherwise set it on the harness,
        # recompute inference routing, sync thinking level, rebuild the runtime provider,
        # update image support, and append a model-change entry (refreshing state). Does not
        # change saved defaults. Raises ProviderConfigError for invalid models.
        """Activate and persist an explicit startup model before the next turn."""
        provider = self._active_provider_config()
        if provider is not None:
            validate_provider_model(provider, model)
        if self.model == model:
            return

        self._harness.config.model = model
        self._inference_provider = _configured_inference_provider(provider, model)
        self._inference_provider_mode = _configured_inference_provider_mode(provider, model)
        self._sync_thinking_level_to_active_model()
        self._refresh_runtime_provider()
        self._sync_image_support()
        entry = ModelChangeEntry(
            parent_id=self._last_parent_id,
            model=model,
            provider=self.provider_name,
        )
        await self._append_session_entry(entry)
        self._last_parent_id = entry.id
        await self._refresh_persisted_state(leaf_id=entry.id)

    async def select_provider_model(
        self, choice: ModelChoice, *, persist_default: bool = True
    ) -> ModelSelectionResult:
        # [FUNCTION DOC]
        # Switch provider and model with candidate-first publication.
        # Inputs: the choice; keyword persist_default (save as default for configured
        # providers). Raises RuntimeError while running. Returns unchanged when already
        # active.
        # Behavior: build a candidate provider first (dynamic provider via its factory with
        # thinking level clamped to the model's first declared level and image support from
        # its modalities; or a configured provider validated against settings, with
        # inference routing, coerced thinking level and a response-header observer for
        # Hugging Face), then append a provider-aware model-change entry. If anything fails
        # before the entry is committed, the candidate is closed and the error propagates.
        # After commit, synchronously swap the harness provider/model, session fields,
        # config, thinking level, image support and parent id; invalidate caches; refresh
        # state (index failures become a warning diagnostic); close the replaced provider
        # (errors ignored); persist the default if requested.
        # Output: selection result with changed=True. Side effects: storage, settings,
        # provider lifecycle.
        """Switch provider/model with candidate-first durable publication.

        No active state changes until the candidate runtime exists and the
        provider-aware model entry is committed.
        """
        if self._harness.is_running:
            raise RuntimeError(
                "Tau is still working. Press Escape to interrupt before switching models."
            )
        if choice.provider_name == self.provider_name and choice.model == self.model:
            return ModelSelectionResult(choice, changed=False)

        candidate: ClosableModelProvider | None = None
        selected_dynamic: DynamicProvider | None = None
        selected_config: ProviderConfig | None = None
        selected_inference: str | None = None
        selected_inference_mode: InferenceProviderMode = "automatic"
        selected_thinking = self._thinking_level
        selected_image_support: bool | None = None
        try:
            effective = self._provider_registry.effective(choice.provider_name)
            if effective is not None and isinstance(effective.definition, DynamicProvider):
                selected_dynamic = effective.definition
                selected_model = next(
                    (item for item in selected_dynamic.models if item.id == choice.model), None
                )
                if selected_model is None:
                    raise ProviderConfigError(
                        "Model is not available for provider "
                        f"{choice.provider_name}: {choice.model}"
                    )
                candidate = await create_dynamic_model_provider(
                    selected_dynamic,
                    model=choice.model,
                    credential_store=self._credential_store,
                )
                if (
                    selected_model.thinking_levels is not None
                    and selected_thinking not in selected_model.thinking_levels
                ):
                    selected_thinking = (
                        selected_model.thinking_levels[0]
                        if selected_model.thinking_levels
                        else DEFAULT_THINKING_LEVEL
                    )
                selected_image_support = (
                    "image" in selected_model.input_modalities
                    if selected_model.input_modalities is not None
                    else None
                )
            else:
                if self._provider_settings is None:
                    raise ProviderConfigError(f"Provider is not available: {choice.provider_name}")
                selected_config = self._provider_settings.get_provider(choice.provider_name)
                validate_provider_model(selected_config, choice.model)
                selected_inference = _configured_inference_provider(selected_config, choice.model)
                selected_inference_mode = _configured_inference_provider_mode(
                    selected_config, choice.model
                )
                selected_thinking = _coerced_thinking_level(
                    selected_config,
                    model=choice.model,
                    current=self._thinking_level,
                )
                candidate = _create_runtime_provider(
                    selected_config,
                    credential_store=self._credential_store,
                    model=choice.model,
                    thinking_level=selected_thinking,
                    inference_provider=selected_inference,
                    response_headers_observer=(
                        self._observe_response_headers
                        if selected_config.name == "huggingface"
                        else None
                    ),
                )
                selected_image_support = provider_model_supports_images(
                    selected_config, choice.model
                )

            entry = ModelChangeEntry(
                parent_id=self._last_parent_id,
                model=choice.model,
                provider=choice.provider_name,
            )
            await self._append_session_entry(entry)
        except BaseException:
            if candidate is not None:
                await candidate.aclose()
            raise

        # From this point the transcript is authoritative. Only synchronous
        # assignments happen before control returns to the frontend.
        old_provider = self._harness.config.provider
        assert candidate is not None
        self._owned_providers.append(candidate)
        self._harness.config.provider = candidate
        self._harness.config.model = choice.model
        self._provider_name = choice.provider_name
        self._inference_provider = selected_inference
        self._inference_provider_mode = selected_inference_mode
        self._runtime_provider_config = selected_config
        self._config = replace(
            self._config,
            provider=candidate,
            model=choice.model,
            provider_name=choice.provider_name,
            inference_provider=selected_inference,
            inference_provider_mode=selected_inference_mode,
            runtime_provider_config=selected_config,
            dynamic_provider=selected_dynamic,
        )
        self._thinking_level = selected_thinking
        self._image_support.supported = selected_image_support
        self._last_parent_id = entry.id
        self._invalidate_runtime_model_limits()
        self._invalidate_context_usage_cache()
        try:
            await self._refresh_persisted_state(leaf_id=entry.id)
        except Exception as exc:  # committed history wins over repairable index failure
            self._resource_diagnostics = (
                *self._resource_diagnostics,
                ResourceDiagnostic(
                    kind="session-index",
                    message=f"Committed model switch requires index repair: {type(exc).__name__}",
                    severity="warning",
                ),
            )
        if old_provider is not candidate:
            with suppress(Exception):
                await self._close_replaced_provider(old_provider)
        if selected_config is not None and persist_default:
            self._persist_default_model_choice()
        return ModelSelectionResult(choice, changed=True)

    def set_inference_provider(self, route: str | None) -> str:
        # [FUNCTION DOC]
        # Pin (or reset to automatic with None) the Hugging Face backing provider for this
        # session. Requires the active provider to be "huggingface" (ProviderConfigError
        # otherwise); validates the route, builds a new runtime provider, updates the session
        # index, stores routing state, and activates the provider.
        # Output: the pinned route name or a message describing automatic routing.
        """Select or reset the active Hugging Face session route."""
        if self.provider_name != "huggingface":
            raise ProviderConfigError(
                "Inference-provider routing requires the huggingface provider"
            )
        normalized = validate_huggingface_inference_provider(route) if route is not None else None
        mode: InferenceProviderMode = "fixed" if normalized is not None else "automatic"
        provider, provider_config = self._build_runtime_provider(
            inference_provider=normalized,
        )
        self._owned_providers.append(provider)
        if self._config.session_manager is not None and self._config.session_id is not None:
            self._config.session_manager.touch_session(
                self._config.session_id,
                model=self.model,
                provider_name=self.provider_name,
                inference_provider=normalized,
                inference_provider_mode=mode,
                preserve_inference_provider=False,
            )
        self._inference_provider = normalized
        self._inference_provider_mode = mode
        self._config = replace(
            self._config,
            inference_provider=normalized,
            inference_provider_mode=mode,
        )
        self._activate_runtime_provider(provider, provider_config)
        return normalized or "automatic (will pin after the next successful response)"

    def set_model_choice(self, choice: ModelChoice) -> None:
        # [FUNCTION DOC]
        # Synchronously switch to a provider/model pair: same provider uses set_model,
        # another provider uses the internal provider switch.
        """Switch provider/model as one operation."""
        if choice.provider_name == self.provider_name:
            self.set_model(choice.model)
            return
        self._set_provider_model(choice.provider_name, choice.model)

    def is_scoped_model(self, choice: ModelChoice) -> bool:
        # [FUNCTION DOC]
        # Return whether the pair is in the current scoped model choices.
        """Return whether a provider/model pair is in the scoped model list."""
        return choice in self.scoped_model_choices

    def toggle_scoped_model(self, choice: ModelChoice) -> tuple[ModelChoice, ...]:
        # [FUNCTION DOC]
        # Add or remove a provider/model pair in the persisted scoped model list.
        # Requires provider settings (ProviderConfigError). For dynamic providers only
        # trusted built-in providers with stable references are allowed, and the model must
        # be available unless it is being removed; for configured providers it must be
        # available. Uses the regular toggle for configured catalog models and the stable
        # reference toggle otherwise. Updates durable settings and re-applies runtime
        # catalogs (or, for lightweight session doubles without durable settings, the
        # settings directly), resyncs thinking level, and returns the new scoped choices.
        # Side effects: settings file write.
        """Add or remove a model from the persisted scoped model list."""
        if self._provider_settings is None:
            raise ProviderConfigError("Provider settings are not available for this session")
        available = set(self.available_model_choices)
        existing = choice in self.scoped_model_choices
        durable_settings = getattr(self, "_durable_provider_settings", self._provider_settings)
        effective = self._provider_registry.effective(choice.provider_name)
        if effective is not None and isinstance(effective.definition, DynamicProvider):
            if not (
                self._effective_stable_dynamic_scoped_provider(choice.provider_name)
                or (existing and self._stable_dynamic_scoped_provider(choice.provider_name))
            ):
                raise ProviderConfigError(
                    "Only effective trusted built-in dynamic providers support scoped references"
                )
            if choice not in available and not existing:
                raise ProviderConfigError(
                    f"Model is unavailable: {choice.provider_name}:{choice.model}"
                )
            toggle = toggle_saved_stable_scoped_model
        else:
            if choice not in available:
                raise ProviderConfigError(
                    f"Model is not available: {choice.provider_name}:{choice.model}"
                )
            durable_provider = (
                durable_settings.get_provider(choice.provider_name)
                if durable_settings is not None
                else None
            )
            toggle = (
                toggle_saved_scoped_model
                if durable_provider is not None and choice.model in durable_provider.models
                else toggle_saved_stable_scoped_model
            )

        updated_settings = toggle(
            provider_name=choice.provider_name,
            model=choice.model,
            paths=self._resource_paths.paths,
            fallback_settings=durable_settings,
        )
        if hasattr(self, "_durable_provider_settings"):
            self._durable_provider_settings = updated_settings
            self._apply_runtime_model_catalogs()
        else:  # narrowly supports lightweight host/test session doubles
            self._provider_settings = updated_settings
        self._sync_thinking_level_to_active_model()
        return self.scoped_model_choices

    @property
    def has_pending_selection(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether a previewed model or thinking level is waiting to be committed.
        """Whether a TUI selection has not yet been committed to history."""
        return self._preview_model is not None or self._preview_thinking is not None

    def preview_model_choice(self, choice: ModelChoice) -> None:
        # [FUNCTION DOC]
        # Select a model for the next turn without constructing or persisting anything.
        # Raises RuntimeError while running and ProviderConfigError if unavailable. Also
        # previews thinking "off" when the new model has no levels, or its first level when
        # the current level is unsupported. Invalidates live model limits.
        """Select a model for the next turn without constructing or persisting a provider."""
        if self._harness.is_running:
            raise RuntimeError("Cannot switch models while Tau is working")
        if choice not in self.available_model_choices:
            raise ProviderConfigError(
                f"Model is not available: {choice.provider_name}:{choice.model}"
            )
        self._preview_model = choice
        levels = self.available_thinking_levels
        if not levels:
            self._preview_thinking = "off"
        elif self.thinking_level not in levels:
            self._preview_thinking = levels[0]
        self._invalidate_runtime_model_limits()

    def preview_thinking_level(self, level: str) -> str:
        # [FUNCTION DOC]
        # Preview a thinking level for the next turn after normalizing and validating it
        # against the (previewed) model's levels. Raises ValueError if unavailable.
        # Returns a "Thinking mode: <level>" message.
        """Select a thinking level for the next turn without touching durable settings."""
        normalized = normalize_thinking_level(level)
        levels = self.available_thinking_levels
        if not levels:
            raise ValueError(_unavailable_thinking_message(self))
        if normalized not in levels:
            raise ValueError(
                f"Thinking mode {normalized} is not available. Available: {', '.join(levels)}"
            )
        self._preview_thinking = normalized
        return f"Thinking mode: {normalized}"

    def preview_cycle_thinking_level(self) -> str:
        # [FUNCTION DOC]
        # Preview the next supported thinking level after the current one. Returns the
        # status message; raises ValueError like preview_thinking_level.
        return self.preview_thinking_level(
            next_thinking_level(self.thinking_level, available=self.available_thinking_levels)
        )

    def preview_cycle_scoped_model(self, *, reverse: bool = False) -> ModelChoice:
        # [FUNCTION DOC]
        # Preview the next (or previous with reverse) available scoped model, wrapping
        # around; if the current model is not scoped, starts from the first (or last).
        # Raises ProviderConfigError when no scoped models are available. Returns the choice.
        """Cycle through available scoped choices without changing durable state."""
        available = set(self.available_model_choices)
        scoped = tuple(choice for choice in self.scoped_model_choices if choice in available)
        if not scoped:
            raise ProviderConfigError("No scoped models configured.")
        current = ModelChoice(self.provider_name, self.model)
        try:
            index = scoped.index(current)
        except ValueError:
            index = 0 if reverse else -1
        choice = scoped[(index + (-1 if reverse else 1)) % len(scoped)]
        self.preview_model_choice(choice)
        return choice

    async def _commit_preview(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Before the next accepted user turn, commit pending previews: clear them,
        # select the previewed model (without saving defaults), then apply the previewed
        # thinking level: force "off" for models without levels, otherwise set it (without
        # saving preference), or, if a model switch already produced that level, record a
        # thinking-level entry and emit the change event. On any error the previews are
        # restored and the error re-raised.
        """Commit the final pending selection before the next accepted user turn."""
        choice, thinking = self._preview_model, self._preview_thinking
        if choice is None and thinking is None:
            return
        self._preview_model = None
        self._preview_thinking = None
        try:
            if choice is not None:
                await self.select_provider_model(choice, persist_default=False)
            if thinking is not None and thinking != self._state.thinking_level:
                if thinking == "off" and not self.available_thinking_levels:
                    # A non-reasoning model has no effort control to rebuild.
                    self._thinking_level = "off"
                if thinking != self._thinking_level and self.available_thinking_levels:
                    await self.set_thinking_level(thinking, persist_preference=False)
                elif thinking == self._thinking_level:
                    # A model switch can clamp thinking before set_thinking_level
                    # runs. Record that effective change on the same branch.
                    entry = ThinkingLevelChangeEntry(
                        parent_id=self._last_parent_id, thinking_level=thinking
                    )
                    await self._append_session_entry(entry)
                    self._last_parent_id = entry.id
                    await self._refresh_persisted_state(leaf_id=entry.id)
                    await self._extension_runtime.emit_event(
                        ThinkingLevelChangedEvent(level=thinking)
                    )
        except Exception:
            self._preview_model = choice
            self._preview_thinking = thinking
            raise

    def cycle_scoped_model(self, *, reverse: bool = False) -> ModelChoice:
        # [FUNCTION DOC]
        # Synchronously switch to the next (or previous) available scoped model, wrapping
        # around. Raises ProviderConfigError if none. Returns the chosen pair.
        """Switch to the next currently available configured scoped model."""
        available = set(self.available_model_choices)
        scoped = tuple(choice for choice in self.scoped_model_choices if choice in available)
        if not scoped:
            raise ProviderConfigError("No scoped models configured.")
        current = ModelChoice(provider_name=self.provider_name, model=self.model)
        try:
            current_index = scoped.index(current)
        except ValueError:
            current_index = -1 if not reverse else 0
        delta = -1 if reverse else 1
        choice = scoped[(current_index + delta) % len(scoped)]
        self.set_model_choice(choice)
        return choice

    def set_provider(self, provider_name: str, *, persist_default: bool = True) -> None:
        # [FUNCTION DOC]
        # Switch to a configured provider using its default model (requires provider
        # settings; ProviderConfigError otherwise), optionally saving it as default.
        """Switch the active provider and reset to that provider's default model."""
        if self._provider_settings is None:
            raise ProviderConfigError("Provider settings are not available for this session")
        provider_config = self._provider_settings.get_provider(provider_name)
        self._set_provider_model(
            provider_name,
            provider_config.default_model,
            persist_default=persist_default,
        )

    def _set_provider_model(
        self,
        provider_name: str,
        model: str,
        *,
        persist_default: bool = True,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Synchronously switch provider and model: requires settings and a model
        # configured for the provider (ProviderConfigError), coerces the thinking level,
        # builds the runtime provider (RuntimeError converted to ProviderConfigError),
        # records ownership, swaps it onto the harness, updates provider/routing state,
        # invalidates live limits, updates image support, optionally persists defaults, and
        # updates the session index. Old providers are closed only when the session closes.
        # KNOWN ISSUE: no model-change entry is written to the transcript.
        """Switch active provider/model without constructing an intermediate provider."""
        if self._provider_settings is None:
            raise ProviderConfigError("Provider settings are not available for this session")

        provider_config = self._provider_settings.get_provider(provider_name)
        if model not in provider_config.models:
            raise ProviderConfigError(f"Model is not configured: {provider_name}:{model}")
        thinking_level = _coerced_thinking_level(
            provider_config,
            model=model,
            current=self._thinking_level,
        )
        try:
            provider = _create_runtime_provider(
                provider_config,
                credential_store=self._credential_store,
                model=model,
                thinking_level=thinking_level,
                inference_provider=_configured_inference_provider(provider_config, model),
                response_headers_observer=(
                    self._observe_response_headers
                    if provider_config.name == "huggingface"
                    else None
                ),
            )
        except RuntimeError as exc:
            raise ProviderConfigError(str(exc)) from exc
        self._owned_providers.append(provider)
        self._harness.config.provider = provider
        self._provider_name = provider_config.name
        self._inference_provider = _configured_inference_provider(provider_config, model)
        self._inference_provider_mode = _configured_inference_provider_mode(provider_config, model)
        self._runtime_provider_config = provider_config
        self._invalidate_runtime_model_limits()
        self._harness.config.model = model
        self._thinking_level = thinking_level
        self._sync_image_support()
        if persist_default:
            self._persist_default_model_choice()
        if self._config.session_id is not None and self._config.session_manager is not None:
            self._config.session_manager.touch_session(
                self._config.session_id,
                model=model,
                provider_name=self.provider_name,
                inference_provider=self._inference_provider,
                inference_provider_mode=self._inference_provider_mode,
                preserve_inference_provider=False,
            )

    async def set_thinking_level(self, level: str, *, persist_preference: bool = True) -> str:
        # [FUNCTION DOC]
        # Set and persist the thinking level for future turns.
        # Normalizes the level and validates it against the active model (ValueError if
        # unavailable or unsupported). If unchanged, returns immediately. Otherwise rebuilds
        # the runtime provider (restoring the previous level on ProviderConfigError), appends
        # a thinking-level entry, optionally saves the preference in settings, refreshes
        # state, and emits a thinking-level-changed extension event.
        # Output: "Thinking mode: <level>".
        """Persist and activate a thinking mode for future turns."""
        normalized = normalize_thinking_level(level)
        available = self.available_thinking_levels
        if not available:
            raise ValueError(_unavailable_thinking_message(self))
        if normalized not in available:
            modes = ", ".join(available)
            raise ValueError(
                f"Thinking mode {normalized} is not available for "
                f"{self._provider_name}:{self.model}. Available modes: {modes}"
            )
        if normalized == self._thinking_level:
            return f"Thinking mode: {normalized}"

        previous = self._thinking_level
        self._thinking_level = normalized
        try:
            self._refresh_runtime_provider()
        except ProviderConfigError:
            self._thinking_level = previous
            raise

        entry = ThinkingLevelChangeEntry(
            parent_id=self._last_parent_id,
            thinking_level=normalized,
        )
        await self._append_session_entry(entry)
        self._last_parent_id = entry.id

        if persist_preference:
            self._persist_thinking_level_choice()
        await self._refresh_persisted_state(leaf_id=entry.id)
        await self._extension_runtime.emit_event(ThinkingLevelChangedEvent(level=normalized))
        return f"Thinking mode: {normalized}"

    async def cycle_thinking_level(self) -> str:
        # [FUNCTION DOC]
        # Set the next supported thinking level after the current one (persisted).
        """Cycle to the next supported thinking mode and persist it."""
        return await self.set_thinking_level(
            next_thinking_level(
                self._thinking_level,
                available=self.available_thinking_levels,
            )
        )

    def _selected_provider_config(self) -> ProviderConfig | None:
        # [FUNCTION DOC]
        # HELPER: Provider config used for displayed capabilities: the active config when no
        # model preview exists; otherwise the previewed provider's config from settings
        # (None if settings are missing or the provider is unknown).
        """Resolve displayed capabilities without changing the committed runtime."""
        if self._preview_model is None:
            return self._active_provider_config()
        if self._provider_settings is None:
            return None
        try:
            return self._provider_settings.get_provider(self.provider_name)
        except ProviderConfigError:
            return None

    def _active_provider_config(self) -> ProviderConfig | None:
        # [FUNCTION DOC]
        # HELPER: The committed provider's config from settings (None without settings or
        # for unknown providers). For Codex, when the committed model is missing from the
        # settings view but present in the runtime config, returns the runtime config so a
        # selected account-specific model stays valid.
        if self._provider_settings is None:
            return None
        try:
            provider = self._provider_settings.get_provider(self._provider_name)
        except ProviderConfigError:
            return None
        runtime = self._runtime_provider_config
        if (
            isinstance(provider, OpenAICodexProviderConfig)
            and self._harness.config.model not in provider.models
            and runtime is not None
            and runtime.name == provider.name
            and self._harness.config.model in runtime.models
        ):
            # Picker visibility must not invalidate an already selected runtime.
            return runtime
        return provider

    def _apply_thinking_level_override(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Apply the one-shot startup thinking override during load. For dynamic
        # providers it must be one of the active model's declared levels (ProviderConfigError
        # otherwise). Without a provider config it is applied as is. Otherwise it is resolved
        # via the startup resolver, which may validate strictly and raise or return None (no
        # change).
        """Apply the one-shot startup thinking override to the loaded session.

        Runs once during :meth:`load`, after session state is replayed and
        before the level is synced to the active model, so the override wins
        over both remembered defaults and the resumed transcript state. The
        override is validated strictly against either the active dynamic model
        or the durable provider configuration when its capabilities are known.
        """
        override = self._config.thinking_level_override
        if override is None:
            return
        dynamic = self._active_dynamic_provider
        if dynamic is not None:
            model = self._active_dynamic_model
            levels = model.thinking_levels if model is not None else None
            if not levels:
                raise ProviderConfigError(
                    f"Thinking modes are unavailable for {dynamic.id}:{self.model}"
                )
            if override not in levels:
                allowed = ", ".join(levels)
                raise ProviderConfigError(
                    f'Thinking mode "{override}" is not available for '
                    f"{dynamic.id}:{self.model}. Available modes: {allowed}"
                )
            self._thinking_level = override
            return
        provider = self._active_provider_config()
        if provider is None:
            self._thinking_level = override
            return
        resolved = resolve_startup_thinking_level(provider, self.model, cli_override=override)
        if resolved is not None:
            self._thinking_level = resolved

    def _sync_thinking_level_to_active_model(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Coerce the current thinking level to one valid for the committed model of
        # the active provider config, preferring the provider's remembered per-model
        # default. No-op when there is no provider config.
        provider = self._active_provider_config()
        if provider is None:
            return
        committed_model = self._harness.config.model
        self._thinking_level = _coerced_thinking_level(
            provider,
            model=committed_model,
            current=self._thinking_level,
            preferred=provider.thinking_defaults.get(committed_model),
        )

    def _sync_image_support(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Update the shared image-support state from the active (or runtime)
        # provider config for the committed model; None when unknown.
        provider = self._active_provider_config() or self._runtime_provider_config
        self._image_support.supported = (
            provider_model_supports_images(provider, self._harness.config.model)
            if provider is not None
            else None
        )

    def _persist_default_model_choice(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Save the active provider/model as the default in durable settings, but
        # only for models listed in the durable catalog (account-specific models are not
        # written); then re-apply runtime catalogs and resync thinking. Side effect:
        # settings write. May raise ProviderConfigError if the provider is unknown to the
        # durable settings.
        if self._durable_provider_settings is None:
            return
        durable_provider = self._durable_provider_settings.get_provider(self.provider_name)
        if self.model not in durable_provider.models:
            # Account-specific inventory is deliberately not written into catalog.toml.
            return
        self._durable_provider_settings = save_default_provider_model(
            provider_name=self.provider_name,
            model=self.model,
            paths=self._resource_paths.paths,
            fallback_settings=self._durable_provider_settings,
        )
        self._apply_runtime_model_catalogs()
        self._sync_thinking_level_to_active_model()

    def _persist_thinking_level_choice(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Save the current thinking level as the per-model preference in durable
        # settings when it is valid for the active provider/model; ProviderConfigError is
        # ignored. Re-applies runtime catalogs on success.
        if self._provider_settings is None:
            return
        provider = self._active_provider_config()
        if provider is None or self._thinking_level not in provider_thinking_levels(
            provider,
            model=self.model,
        ):
            return
        try:
            self._durable_provider_settings = save_provider_thinking_level(
                provider_name=self.provider_name,
                model=self.model,
                thinking_level=self._thinking_level,
                paths=self._resource_paths.paths,
                fallback_settings=self._durable_provider_settings,
            )
            self._apply_runtime_model_catalogs()
        except ProviderConfigError:
            return

    def _observe_response_headers(self, headers: Mapping[str, str]) -> None:
        # [FUNCTION DOC]
        # HELPER: Response-header observer for Hugging Face in automatic routing with no
        # pinned route: reads the x-inference-provider header (case-insensitive), validates
        # it, builds a provider pinned to that route (tracking ownership), updates the
        # session index, stores the route in session state/config, and activates the new
        # provider. Invalid or missing headers are ignored.
        if (
            self.provider_name != "huggingface"
            or self._inference_provider_mode != "automatic"
            or self._inference_provider is not None
        ):
            return
        route = next(
            (value for key, value in headers.items() if key.casefold() == "x-inference-provider"),
            None,
        )
        if route is None:
            return
        try:
            route = validate_huggingface_inference_provider(route)
        except ProviderConfigError:
            return
        provider, provider_config = self._build_runtime_provider(
            inference_provider=route,
        )
        # Track staged providers immediately so a later index-write failure does
        # not leak a provider-owned client. The active runtime remains unchanged.
        self._owned_providers.append(provider)
        if self._config.session_manager is not None and self._config.session_id is not None:
            self._config.session_manager.touch_session(
                self._config.session_id,
                model=self.model,
                provider_name=self.provider_name,
                inference_provider=route,
                inference_provider_mode="automatic",
                preserve_inference_provider=False,
            )
        self._inference_provider = route
        self._config = replace(
            self._config,
            inference_provider=route,
            inference_provider_mode="automatic",
        )
        self._activate_runtime_provider(provider, provider_config)

    def will_auto_retry(self, message: AssistantMessage) -> bool:
        # [FUNCTION DOC]
        # Return whether the session will automatically retry after this assistant error:
        # context-overflow errors (handled by compaction) or retryable errors on an
        # automatically pinned Hugging Face route.
        """Return whether session orchestration will retry this assistant error."""
        return is_context_overflow_error(message) or self._should_auto_failover_huggingface_route(
            message
        )

    def _should_auto_failover_huggingface_route(
        self,
        message: AssistantMessage,
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: True when using Hugging Face in automatic mode with a pinned route and the
        # error is a retryable route error.
        return (
            self.provider_name == "huggingface"
            and self._inference_provider_mode == "automatic"
            and self._inference_provider is not None
            and is_retryable_huggingface_route_error(message)
        )

    def _reset_automatic_inference_provider_for_failover(self) -> str:
        # [FUNCTION DOC]
        # HELPER: Drop the automatically pinned Hugging Face route: build an unpinned
        # provider, track it, update the session index, clear the route in state/config, and
        # activate it. Returns the failed route name. Raises ProviderConfigError if no route
        # was pinned.
        failed_route = self._inference_provider
        if failed_route is None:
            raise ProviderConfigError("Hugging Face failover requires a pinned route")
        provider, provider_config = self._build_runtime_provider(inference_provider=None)
        self._owned_providers.append(provider)
        if self._config.session_manager is not None and self._config.session_id is not None:
            self._config.session_manager.touch_session(
                self._config.session_id,
                model=self.model,
                provider_name=self.provider_name,
                inference_provider=None,
                inference_provider_mode="automatic",
                preserve_inference_provider=False,
            )
        self._inference_provider = None
        self._config = replace(
            self._config,
            inference_provider=None,
            inference_provider_mode="automatic",
        )
        self._activate_runtime_provider(provider, provider_config)
        return failed_route

    async def _run_huggingface_route_failover(
        self,
        *,
        context: AgentCallDiagnosticContext,
    ) -> AsyncIterator[CodingSessionEvent]:
        # [FUNCTION DOC]
        # HELPER: Async generator that retries the interrupted run once without a pinned
        # route. Emits (to extensions and caller) an auto-retry-start event, continues the
        # harness and forwards its events (wrapping agent-end as a session agent-end with
        # will_retry false, invalidating context cache on tool ends), records the final
        # error from any error/aborted assistant message and logs assistant errors, closes
        # the event stream, logs the failover outcome, and finally emits an auto-retry-end
        # event with success and final error.
        """Retry the interrupted run once through unsuffixed Hugging Face routing."""
        failed_route = self._reset_automatic_inference_provider_for_failover()
        retry_start = AutoRetryStartEvent(
            attempt=1,
            max_attempts=1,
            delay_ms=0,
            error_message=f"Hugging Face route {failed_route} failed; rerouting automatically",
        )
        await self._extension_runtime.emit_event(retry_start)
        yield retry_start

        retry_events = self._harness.continue_()
        self._invalidate_context_usage_cache()
        final_error: str | None = None
        try:
            async for retry_event in retry_events:
                if isinstance(retry_event, ToolExecutionEndEvent):
                    self._invalidate_context_usage_cache()
                if (
                    isinstance(retry_event, MessageEndEvent)
                    and isinstance(retry_event.message, AssistantMessage)
                    and retry_event.message.stop_reason in {"error", "aborted"}
                ):
                    final_error = retry_event.message.error_message or "Provider request aborted"
                    if retry_event.message.stop_reason == "error":
                        self._last_diagnostic_log_path = (
                            self._diagnostic_logger.log_assistant_error(
                                context=context,
                                phase="agent_loop_route_failover",
                                message=retry_event.message,
                            )
                        )
                if isinstance(retry_event, AgentEndEvent):
                    yield SessionAgentEndEvent(
                        messages=retry_event.messages,
                        will_retry=False,
                    )
                else:
                    yield retry_event
        finally:
            aclose = getattr(retry_events, "aclose", None)
            if aclose is not None:
                with suppress(Exception):
                    await aclose()

        failover_succeeded = final_error is None
        self._last_diagnostic_log_path = self._diagnostic_logger.log_huggingface_route_failover(
            context=context,
            failed_route=failed_route,
            replacement_route=self._inference_provider,
            success=failover_succeeded,
            error_message=final_error,
        )
        retry_end = AutoRetryEndEvent(
            success=failover_succeeded,
            attempt=1,
            final_error=final_error,
        )
        await self._extension_runtime.emit_event(retry_end)
        yield retry_end

    def _build_runtime_provider(
        self,
        *,
        inference_provider: str | None,
    ) -> tuple[ClosableModelProvider, ProviderConfig]:
        # [FUNCTION DOC]
        # HELPER: Build a new runtime provider for the committed model using the active (or
        # runtime) provider config, current thinking level and the given inference route
        # (with the header observer for Hugging Face). Returns the provider and the config
        # used. Raises ProviderConfigError if no runtime config exists, the model is invalid,
        # or construction fails (RuntimeError converted).
        if self._runtime_provider_config is None:
            raise ProviderConfigError("Runtime provider configuration is unavailable")
        provider_config = self._active_provider_config() or self._runtime_provider_config
        committed_model = self._harness.config.model
        validate_provider_model(provider_config, committed_model)
        try:
            provider = _create_runtime_provider(
                provider_config,
                credential_store=self._credential_store,
                model=committed_model,
                thinking_level=self._thinking_level,
                inference_provider=inference_provider,
                response_headers_observer=(
                    self._observe_response_headers
                    if provider_config.name == "huggingface"
                    else None
                ),
            )
        except RuntimeError as exc:
            raise ProviderConfigError(str(exc)) from exc
        return provider, provider_config

    def _activate_runtime_provider(
        self,
        provider: ClosableModelProvider,
        provider_config: ProviderConfig,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Install the provider on the harness, record its config, and invalidate
        # live model limits. Does not close the previous provider.
        self._harness.config.provider = provider
        self._runtime_provider_config = provider_config
        self._invalidate_runtime_model_limits()

    def _refresh_runtime_provider(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Rebuild and activate the runtime provider with the current inference route
        # (e.g. after thinking or model changes), tracking ownership. No-op without a
        # runtime provider config.
        if self._runtime_provider_config is None:
            return
        provider, provider_config = self._build_runtime_provider(
            inference_provider=self._inference_provider,
        )
        self._owned_providers.append(provider)
        self._activate_runtime_provider(provider, provider_config)

    def _invalidate_runtime_model_limits(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Forget live model limits, their key, and the last discovery error.
        self._runtime_model_limits = None
        self._runtime_model_limits_key = None
        self._model_limits_discovery_error = None

    async def _refresh_runtime_model_limits(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Discover live limits for the active provider/model once per key. Skipped
        # for Codex catalog providers when TAU_OFFLINE is set. For Codex with a cached
        # catalog, uses that catalog's model limits. Otherwise, catalog-capable providers
        # discover their model catalog (published to runtime catalogs and, for Codex,
        # persisted to disk), and limit-capable providers discover limits for the model.
        # Any exception is recorded as a non-fatal discovery error (and per-provider
        # catalog error); static configuration remains the fallback.
        # Side effects: network calls, file writes, state mutation.
        key = (self.provider_name, self.model)
        if self._runtime_model_limits_key == key:
            return
        self._runtime_model_limits = None
        self._runtime_model_limits_key = key
        self._model_limits_discovery_error = None
        provider = self._harness.config.provider
        if (
            self.provider_name == "openai-codex"
            and isinstance(provider, ModelCatalogProvider)
            and environ.get("TAU_OFFLINE") is not None
        ):
            return
        try:
            cached_catalog = self._runtime_model_catalogs.get(self.provider_name)
            if self.provider_name == "openai-codex" and cached_catalog is not None:
                model = cached_catalog.model(self.model)
                self._runtime_model_limits = model.limits if model is not None else None
                return
            if isinstance(provider, ModelCatalogProvider):
                catalog = await provider.discover_models()
                self._publish_runtime_model_catalog(self.provider_name, catalog)
                if self.provider_name == "openai-codex":
                    self._persist_codex_model_catalog(provider, catalog)
            if isinstance(provider, ModelLimitsProvider):
                self._runtime_model_limits = await provider.discover_model_limits(self.model)
        except Exception as exc:  # noqa: BLE001 - static catalog remains the safe fallback
            error = f"{type(exc).__name__}: {exc}"
            self._model_limits_discovery_error = error
            if isinstance(provider, ModelCatalogProvider):
                self._model_catalog_discovery_errors[self.provider_name] = error

    async def reload(self) -> CodingReloadSummary:
        # [FUNCTION DOC]
        # Reload resources and extensions by staging a complete replacement and publishing
        # it atomically (Pi's /reload).
        # Requires idle (RuntimeError). Captures signatures of current resources, tools,
        # guidelines and prompt sections; stages a fresh extension runtime (user/extra
        # extensions only), re-resolves project trust with refresh and no caching (a
        # cancelled decision raises ValueError and keeps everything), loads resources under
        # the staged trust, adds a trust diagnostic, loads project extensions if trusted and
        # enabled, composes and filters tools, builds the command registry, and rebuilds the
        # system prompt only if its inputs changed (and no explicit system override).
        # Then emits session_shutdown to the old runtime, clears its UI components, binds
        # and starts the staged runtime (session_start "reload"); afterwards synchronously
        # commits trust, retires the old runtime and swaps all session fields, harness tools
        # and system prompt, attaches extension listeners, and closes the old runtime.
        # Output: a summary of changes per category (see the continuation below).
        # Side effects: filesystem scans, extension imports and lifecycle events, trust
        # store writes, possible prompts.
        # KNOWN ISSUE: if a failure occurs after staging the runtime but before publication
        # (e.g. a cancelled trust decision or an error in the staged session_start), the
        # staged runtime is not closed; and if staged session_start fails, the old runtime
        # has already received session_shutdown and had its UI cleared while staying active.
        """Stage and atomically publish a complete replacement snapshot."""
        self._require_idle("reload")
        before_skills = _skill_signatures(self._skills)
        before_prompt_templates = _prompt_template_signatures(self._prompt_templates)
        before_context_files = _context_file_signatures(self._context_files)
        before_diagnostics = _diagnostic_signatures(self.resource_diagnostics)
        before_system_prompt_inputs = _system_prompt_resource_signatures(
            skills=self._skills,
            context_files=self._context_files,
            custom_system_prompt=self._custom_system_prompt,
            custom_system_prompt_path=self._custom_system_prompt_path,
            append_system_prompt=self._append_system_prompt,
            append_system_prompt_paths=self._append_system_prompt_paths,
        )
        before_extensions = _extension_signatures(self._extension_runtime)
        before_tool_names = tuple(tool.name for tool in self._harness.config.tools)
        before_guidelines = self._extension_runtime.prompt_guidelines
        before_sections = self._extension_runtime.sourced_prompt_sections

        # Nothing below mutates the live session. Eligible extensions are loaded
        # first so project code cannot import before the destination decision.
        unfiltered_paths = resource_paths_with_cwd(self._config.resource_paths, self.cwd)
        previous_ui = self._extension_runtime.ui
        staged_runtime = ExtensionRuntime(
            durable_providers=(
                self._provider_settings.providers if self._provider_settings else ()
            ),
            credentials=self._extension_runtime.provider_credentials or self._credential_store,
            environment=self._extension_runtime.provider_environment,
            built_in_credentials=self._extension_runtime.built_in_credentials,
            built_in_http_client=self._extension_runtime.built_in_http_client,
            paths=(
                self._resource_paths.paths
                or TauPaths(
                    home=self._resource_paths.root,
                    agents_home=self._resource_paths.agents_root or Path.home() / ".agents",
                )
            ),
        )
        staged_runtime.load(
            unfiltered_paths,
            extra_paths=self._config.extension_paths,
            include_resource_dirs=self._config.extensions_enabled,
            include_project_dir=False,
        )

        staged_resolution = self._project_trust_resolution
        staged_paths = self._resource_paths
        coordinator = self._config.project_trust_coordinator
        trust_summary = None
        if coordinator is not None:
            trust_summary, staged_resolution = await coordinator.resolve(
                self.cwd,
                override=self._config.trust_override,
                default=self._config.trust_default,
                interactive=self._config.trust_interactive,
                prompt=self._config.trust_prompt,
                extension_deciders=(staged_runtime.decide_project_trust,),
                refresh=True,
                cache_result=False,
            )
            if staged_resolution.cancelled:
                raise ValueError("Project trust decision cancelled; keeping current resources")
            staged_paths = resource_paths_with_project_trust(
                resource_paths_with_cwd(self._config.resource_paths, trust_summary.cwd.value),
                trusted=staged_resolution.trusted,
            )

        resources = _load_session_resources(
            staged_paths,
            self._config.context_files,
            skills_enabled=self._config.skills_enabled,
            system_prompt_enabled=self._config.system is None,
            custom_system_prompt_explicit=self._config.custom_system_prompt is not None,
        )
        if trust_summary is not None and trust_summary.categories:
            assert staged_resolution is not None
            resources = replace(
                resources,
                diagnostics=(
                    *resources.diagnostics,
                    ResourceDiagnostic(
                        kind="project-trust",
                        message=format_trust_diagnostic(trust_summary, staged_resolution),
                        severity="info" if staged_resolution.trusted else "warning",
                    ),
                ),
            )
        if (
            staged_resolution is not None
            and staged_resolution.trusted
            and self._config.project_extensions_enabled
        ):
            staged_runtime.load(
                staged_paths,
                include_resource_dirs=True,
                include_project_dir=True,
                include_user_dir=False,
            )

        base_tools = (
            self._config.tools
            if self._config.tools is not None
            else create_coding_tools(
                cwd=self._config.cwd,
                shell_command_prefix=self._config.shell_command_prefix,
                image_support=self._image_support,
            )
        )
        staged_tools = _filter_tools(
            staged_runtime.compose_tools(base_tools), self._config.allowed_tool_names
        )
        staged_commands = self._config.command_registry or staged_runtime.build_command_registry()
        after_system_prompt_inputs = _system_prompt_resource_signatures(
            skills=resources.skills,
            context_files=resources.context_files,
            custom_system_prompt=resources.custom_system_prompt,
            custom_system_prompt_path=resources.custom_system_prompt_path,
            append_system_prompt=resources.append_system_prompt,
            append_system_prompt_paths=resources.append_system_prompt_paths,
        )
        after_guidelines = staged_runtime.prompt_guidelines
        after_sections = staged_runtime.sourced_prompt_sections
        system_prompt_rebuilt = self._config.system is None and (
            before_system_prompt_inputs != after_system_prompt_inputs
            or before_tool_names != tuple(tool.name for tool in staged_tools)
            or before_guidelines != after_guidelines
            or before_sections != after_sections
        )
        staged_system = self._harness.config.system
        if system_prompt_rebuilt:
            staged_system = build_system_prompt(
                BuildSystemPromptOptions(
                    cwd=self._config.cwd,
                    tools=staged_tools,
                    skills=resources.skills,
                    custom_prompt=(
                        self._config.custom_system_prompt
                        if self._config.custom_system_prompt is not None
                        else resources.custom_system_prompt
                    ),
                    append_sections=_append_prompt_sections(
                        resources.append_system_prompts,
                        resources.append_system_prompt_paths,
                        self._config.append_system_prompt,
                    ),
                    context_files=resources.context_files,
                    extra_guidelines=after_guidelines,
                    extra_sections=after_sections,
                    custom_prompt_source=_custom_prompt_source(
                        explicit=self._config.custom_system_prompt is not None,
                        path=resources.custom_system_prompt_path,
                    ),
                )
            )

        # Cancellable lifecycle work stays before the publication boundary. The
        # staged runtime may inspect the still-live session during session_start;
        # cancellation leaves that prior snapshot/runtime/cache untouched.
        old_runtime = self._extension_runtime
        await old_runtime.emit_session_shutdown("reload")
        old_runtime.clear_ui_components()
        staged_runtime.set_ui_bridge(previous_ui)
        staged_runtime.bind(self)
        await staged_runtime.emit_session_start("reload")

        # Publication is synchronous: cancellation can no longer report failure
        # after only part of the live snapshot or trust cache was adopted.
        if coordinator is not None and trust_summary is not None:
            assert staged_resolution is not None
            coordinator.commit(trust_summary.cwd, staged_resolution)
        old_runtime.retire()
        self._resource_paths = staged_paths
        self._config = replace(
            self._config,
            cwd=staged_paths.cwd or self._config.cwd,
            resource_paths=staged_paths,
            extension_runtime=staged_runtime,
        )
        self._project_trust_resolution = staged_resolution
        self._skills = resources.skills
        self._prompt_templates = resources.prompt_templates
        self._context_files = resources.context_files
        self._custom_system_prompt = resources.custom_system_prompt
        self._custom_system_prompt_path = resources.custom_system_prompt_path
        self._append_system_prompt = resources.append_system_prompt
        self._append_system_prompts = resources.append_system_prompts
        self._append_system_prompt_paths = resources.append_system_prompt_paths
        self._resource_diagnostics = resources.diagnostics
        self._command_registry = staged_commands
        self._extension_runtime = staged_runtime
        self._provider_registry = staged_runtime.provider_registry
        self._harness.config.tools = staged_tools
        self._harness.config.system = staged_system
        if system_prompt_rebuilt:
            self._invalidate_context_usage_cache()
        staged_runtime.attach_harness_listener(self._harness.subscribe)
        # Retirement invalidates publication synchronously; async close then
        # waits for cooperative provider callback cleanup or reports bounded
        # containment while the outgoing runtime still owns every task handle.
        # Caller cancellation at this committed seam is contained so reload
        # cannot report failure after the fresh snapshot became active.
        await _finish_adopted_runtime_close(old_runtime)

        return CodingReloadSummary(
            skills=_category_summary(before_skills, _skill_signatures(resources.skills)),
            prompt_templates=_category_summary(
                before_prompt_templates, _prompt_template_signatures(resources.prompt_templates)
            ),
            context_files=_category_summary(
                before_context_files, _context_file_signatures(resources.context_files)
            ),
            extensions=_category_summary(before_extensions, _extension_signatures(staged_runtime)),
            diagnostics=_category_summary(
                before_diagnostics, _diagnostic_signatures(self.resource_diagnostics)
            ),
            system_prompt_rebuilt=system_prompt_rebuilt,
        )

    async def refresh_model_catalogs(self, *, force: bool = False) -> ModelsDevRefreshResult:
        # [FUNCTION DOC]
        # Refresh the public models.dev catalog (optionally forced) and reload provider
        # settings, then always refresh the authenticated Codex catalog. If the public
        # refresh failed, its error is raised after the Codex refresh. Returns the public
        # refresh result. Side effects: network, cache files, settings reload.
        """Refresh public and authenticated catalogs and publish them to this session."""
        public_error: Exception | None = None
        result: ModelsDevRefreshResult | None = None
        try:
            result = await refresh_models_dev_catalog(
                paths=self._resource_paths.paths,
                force=force,
            )
            self.reload_provider_settings()
        except Exception as error:  # noqa: BLE001 - authenticated refresh still runs
            public_error = error
        await self._refresh_codex_model_catalog()
        if public_error is not None:
            raise public_error
        assert result is not None
        return result

    async def _refresh_codex_model_catalog(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Refresh the account-specific Codex model inventory using a temporary fresh
        # provider (so newly released models are discovered even when Codex is active):
        # skipped when TAU_OFFLINE is set, without durable settings, without a Codex config,
        # or without usable credentials. Publishes the catalog to runtime catalogs (not to
        # catalog settings) and persists it to the per-account cache; errors are recorded
        # as discovery errors. The temporary provider is always closed.
        """Refresh the account-specific Codex inventory without making it durable."""
        if environ.get("TAU_OFFLINE") is not None or self._durable_provider_settings is None:
            return
        try:
            provider_config = self._durable_provider_settings.get_provider("openai-codex")
        except ProviderConfigError:
            return
        if not isinstance(provider_config, OpenAICodexProviderConfig):
            return
        if not self._provider_is_usable(provider_config):
            return

        # Use a fresh provider even when Codex is active. The active provider caches
        # its startup snapshot so reusing it would make `/model` unable to discover
        # versions or models released during a long-running Tau process.
        temporary_provider = create_model_provider(
            provider_config,
            credential_store=self._credential_store,
            model=None,
            thinking_level=None,
        )
        if not isinstance(temporary_provider, ModelCatalogProvider):
            await temporary_provider.aclose()
            return
        try:
            catalog = await temporary_provider.discover_models()
            self._publish_runtime_model_catalog(provider_config.name, catalog)
            self._persist_codex_model_catalog(temporary_provider, catalog, provider_config)
        except Exception as exc:  # noqa: BLE001 - static catalog remains the safe fallback
            self._model_catalog_discovery_errors[provider_config.name] = (
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            await temporary_provider.aclose()

    def _persist_codex_model_catalog(
        self,
        provider: object,
        catalog: RuntimeModelCatalog,
        provider_config: OpenAICodexProviderConfig | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Save a Codex catalog to the per-account on-disk cache. The Codex config
        # comes from the argument or the runtime config; the account id from the provider
        # object or stored credentials. Skips silently without config or account id;
        # OS/type/value errors while saving are ignored.
        config = provider_config or (
            self._runtime_provider_config
            if isinstance(self._runtime_provider_config, OpenAICodexProviderConfig)
            else None
        )
        if config is None:
            return
        account_id = getattr(provider, "account_id", None) or _codex_account_id(
            config,
            self._credential_store,
        )
        if not isinstance(account_id, str) or not account_id:
            return
        with suppress(OSError, TypeError, ValueError):
            save_codex_model_catalog(
                catalog,
                account_id=account_id,
                paths=self._resource_paths.paths,
            )

    def _publish_runtime_model_catalog(
        self,
        provider_name: str,
        catalog: RuntimeModelCatalog,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Store a discovered catalog for a provider, clear its discovery error,
        # invalidate live limits, and re-apply catalogs to the effective settings.
        # Raises ValueError if the catalog has no models.
        if not catalog.models:
            raise ValueError("provider returned an empty model catalog")
        self._runtime_model_catalogs[provider_name] = catalog
        self._model_catalog_discovery_errors.pop(provider_name, None)
        self._invalidate_runtime_model_limits()
        self._apply_runtime_model_catalogs()

    def _apply_runtime_model_catalogs(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Recompute effective provider settings by overlaying each runtime catalog
        # onto the durable settings' providers (None when there are no durable settings).
        settings = self._durable_provider_settings
        if settings is None:
            self._provider_settings = None
            return
        providers = tuple(
            _provider_with_runtime_model_catalog(
                provider,
                self._runtime_model_catalogs.get(provider.name),
            )
            for provider in settings.providers
        )
        self._provider_settings = replace(settings, providers=providers)

    def reload_provider_settings(self) -> None:
        # [FUNCTION DOC]
        # Reload provider settings from disk (e.g. after login) and re-apply runtime
        # catalogs. If the committed model is still valid for the active provider, resync
        # thinking level, rebuild the runtime provider and image support; on
        # ProviderConfigError restore the previous settings and thinking level and re-raise.
        # No-op when the session has no provider settings. Side effects: settings read.
        """Reload provider settings for login and model-selection flows."""
        if self._provider_settings is None:
            return
        previous_settings = self._provider_settings
        previous_durable_settings = self._durable_provider_settings
        previous_thinking_level = self._thinking_level
        self._durable_provider_settings = load_provider_settings(self._resource_paths.paths)
        self._apply_runtime_model_catalogs()
        active_config = self._active_provider_config()
        if active_config is None or self._harness.config.model not in active_config.models:
            # The refreshed picker snapshot is authoritative for new selections,
            # but cannot invalidate an already constructed, working provider.
            return
        try:
            self._sync_thinking_level_to_active_model()
            self._refresh_runtime_provider()
            self._sync_image_support()
        except ProviderConfigError:
            self._provider_settings = previous_settings
            self._durable_provider_settings = previous_durable_settings
            self._thinking_level = previous_thinking_level
            raise

    async def resume(self, session_id: str) -> str:
        # [FUNCTION DOC]
        # Replace this session's active state with another indexed session (/resume).
        # Requires idle, flushes pending writes, requires a session manager and a known
        # record (ValueError otherwise). Determines the destination provider/model from the
        # record: dynamic providers, unknown providers, missing settings or live-only Codex
        # models are re-resolved in a freshly built runtime ("dynamic resume", deferring
        # authoritative writes); configured providers are validated and reuse the current
        # provider object. Loads the destination session (new cwd, storage, trust,
        # extensions) with this session's settings and options; for legacy records without
        # a provider, carries over the current model. Then adopts the replacement (closing
        # it on any failure). Output: "Resumed session: <id>".
        """Replace this session's active state with another indexed session."""
        self._require_idle("resume")
        await self._flush_pending_message_writes(context=self._diagnostic_context())
        manager = self._config.session_manager
        if manager is None:
            raise ValueError("Session manager is not available")
        record = manager.get_session(session_id)
        if record is None:
            raise ValueError(f"Unknown session: {session_id}")

        provider_name = self._provider_name
        runtime_provider_config = self._runtime_provider_config
        model = self.model
        restore_record_model = False
        dynamic_resume = False
        if record.provider_name:
            effective = self._provider_registry.effective(record.provider_name)
            if effective is not None and isinstance(effective.definition, DynamicProvider):
                # Dynamic definitions are generation-local. Re-resolve the
                # reference in the fresh destination runtime instead of
                # carrying this runtime's provider object across cwd/trust
                # boundaries.
                provider_name = record.provider_name
                model = record.model
                runtime_provider_config = None
                dynamic_resume = True
                restore_record_model = True
            else:
                provider_name = record.provider_name
                model = record.model
                restore_record_model = True
                if self._provider_settings is None:
                    # The destination runtime may provide a process-local
                    # definition even when the source session did not.
                    dynamic_resume = True
                    runtime_provider_config = None
                else:
                    try:
                        runtime_provider_config = self._provider_settings.get_provider(
                            record.provider_name
                        )
                    except ProviderConfigError:
                        # Do not reject a dynamic overlay merely because the
                        # current cwd has not loaded its destination extension.
                        dynamic_resume = True
                        runtime_provider_config = None
                    else:
                        if (
                            isinstance(runtime_provider_config, OpenAICodexProviderConfig)
                            and model not in runtime_provider_config.models
                        ):
                            # Re-discover a live-only destination model before validation.
                            dynamic_resume = True
                            runtime_provider_config = None
                        else:
                            validate_provider_model(runtime_provider_config, model)

        replacement = await type(self).load(
            CodingSessionConfig(
                provider=None if dynamic_resume else self._harness.config.provider,
                model=model,
                cwd=record.cwd,
                storage=jsonl_session_storage(record.path),
                system=self._config.system,
                custom_system_prompt=self._config.custom_system_prompt,
                append_system_prompt=self._config.append_system_prompt,
                context_files=self._config.context_files,
                resource_paths=self._config.resource_paths,
                session_id=record.id,
                session_manager=manager,
                command_registry=self._config.command_registry,
                provider_name=provider_name,
                inference_provider=None if dynamic_resume else record.inference_provider,
                inference_provider_mode=record.inference_provider_mode,
                requested_provider=provider_name if dynamic_resume else None,
                requested_model=model if dynamic_resume else None,
                session_provider_name=record.provider_name,
                provider_settings=self._durable_provider_settings,
                runtime_provider_config=runtime_provider_config,
                auto_compact_token_threshold=self._auto_compact_token_threshold,
                auto_compact_enabled=self._auto_compact_enabled,
                thinking_level=self._thinking_level,
                shell_command_prefix=self._config.shell_command_prefix,
                skills_enabled=self._config.skills_enabled,
                allowed_tool_names=self._config.allowed_tool_names,
                extension_paths=self._config.extension_paths,
                extensions_enabled=self._config.extensions_enabled,
                project_extensions_enabled=self._config.project_extensions_enabled,
                extension_runtime=self._extension_runtime,
                project_trust_coordinator=self._config.project_trust_coordinator,
                trust_override=self._config.trust_override,
                trust_default=self._config.trust_default,
                trust_interactive=self._config.trust_interactive,
                trust_prompt=self._config.trust_prompt,
                defer_authoritative_writes=dynamic_resume,
                owns_initial_provider=dynamic_resume,
            )
        )
        try:
            if not restore_record_model:
                # Only provider-less legacy records inherit the source model.
                # The staged loader has already resolved provider-aware records
                # against the destination's (possibly freshly discovered) catalog.
                replacement._harness.config.model = self._harness.config.model
                replacement._sync_thinking_level_to_active_model()
                replacement._refresh_runtime_provider()
                replacement._sync_image_support()
        except BaseException:
            # resume owns the loaded candidate until adoption takes ownership.
            await _finish_aborted_session_close(replacement)
            raise
        await self._adopt_replacement(replacement, reason="resume")
        return f"Resumed session: {record.id}"

    async def set_session_name(self, name: str) -> str:
        # [FUNCTION DOC]
        # Normalize and persist a session name (indexing the session if necessary); if it
        # changed, emit a session-info-changed extension event. Returns the persisted (or
        # normalized, if unchanged) name. Raises ValueError without a session manager/id.
        """Persist a session name and notify extensions after it changes."""
        normalized = normalize_session_name(name)
        persisted = self._persist_session_name(
            normalized,
            only_if_unnamed=False,
            index_if_missing=True,
        )
        if persisted is None:
            return normalized
        await self._extension_runtime.emit_event(SessionInfoChangedEvent(name=persisted))
        return persisted

    def _persist_session_name(
        self,
        name: str,
        *,
        only_if_unnamed: bool,
        index_if_missing: bool,
    ) -> str | None:
        # [FUNCTION DOC]
        # HELPER: Write a normalized title to the session index. Requires manager and id
        # (ValueError). Optionally indexes a missing record first. Returns None when no
        # record exists, when only_if_unnamed and a title already exists, or when the title
        # is unchanged; otherwise touches the record with the current model/provider and
        # title and returns the stored title. Raises ValueError if the update fails while
        # indexing was requested.
        """Persist and return a changed name; return None for a no-op."""
        normalized = normalize_session_name(name)
        manager = self._config.session_manager
        session_id = self._config.session_id
        if manager is None or session_id is None:
            raise ValueError("Session manager is not available")
        record = manager.get_session(session_id)
        if record is None and index_if_missing:
            self.ensure_session_indexed()
            record = manager.get_session(session_id)
        if record is None:
            return None
        if only_if_unnamed and record.title:
            return None
        if record.title == normalized:
            return None
        updated = manager.touch_session(
            session_id,
            model=self.model,
            provider_name=self.provider_name,
            title=normalized,
        )
        if updated is None:
            if index_if_missing:
                raise ValueError(f"Unknown session: {session_id}")
            return None
        return updated.title or normalized

    async def new_session(self) -> str:
        # [FUNCTION DOC]
        # Replace this session with a fresh, not-yet-indexed session in the same cwd (/new).
        # Requires idle, flushes pending writes, requires a session manager. Chooses the
        # default provider/model from settings (thinking coerced), or keeps the current
        # ones; for dynamic providers picks the same model if offered, else the provider
        # default or first model. Prepares a session record (with inference routing when
        # pinned), loads a replacement with a fresh provider when settings exist, index on
        # first persist, and deferred writes for dynamic providers, then adopts it.
        # Output: "Started new session: <id>".
        """Replace this session's active state with a pending unindexed session."""
        self._require_idle("start a new session")
        await self._flush_pending_message_writes(context=self._diagnostic_context())
        manager = self._config.session_manager
        if manager is None:
            raise ValueError("Session manager is not available")

        provider_name = self._provider_name
        model = self._harness.config.model
        runtime_provider_config = self._runtime_provider_config
        thinking_level = self._thinking_level
        if self._provider_settings is not None:
            selection = resolve_provider_selection(self._provider_settings)
            provider_name = selection.provider.name
            model = selection.model
            runtime_provider_config = selection.provider
            thinking_level = _coerced_thinking_level(
                selection.provider,
                model=model,
                current=self._thinking_level,
            )

        effective = self._provider_registry.effective(provider_name)
        dynamic_provider = (
            effective.definition
            if effective is not None and isinstance(effective.definition, DynamicProvider)
            else None
        )
        if dynamic_provider is not None:
            model = next(
                (item.id for item in dynamic_provider.models if item.id == model),
                dynamic_provider.default_model
                or (dynamic_provider.models[0].id if dynamic_provider.models else model),
            )
            runtime_provider_config = None
        inference_provider = (
            None
            if dynamic_provider is not None
            else _configured_inference_provider(runtime_provider_config, model)
        )
        inference_provider_mode: InferenceProviderMode = (
            "automatic"
            if dynamic_provider is not None
            else _configured_inference_provider_mode(runtime_provider_config, model)
        )
        record = (
            manager.prepare_session(
                cwd=self.cwd,
                model=model,
                provider_name=provider_name,
                inference_provider=inference_provider,
                inference_provider_mode=inference_provider_mode,
            )
            if inference_provider is not None
            else manager.prepare_session(
                cwd=self.cwd,
                model=model,
                provider_name=provider_name,
            )
        )
        replacement = await type(self).load(
            replace(
                self._config,
                # The old runtime may belong to a different provider. Stage a
                # fresh runtime for the new session's default provider/model.
                provider=(
                    self._harness.config.provider if self._provider_settings is None else None
                ),
                model=record.model or model,
                cwd=record.cwd,
                storage=jsonl_session_storage(record.path),
                session_id=record.id,
                provider_name=provider_name,
                inference_provider=inference_provider,
                inference_provider_mode=inference_provider_mode,
                requested_provider=provider_name if dynamic_provider is not None else None,
                requested_model=model if dynamic_provider is not None else None,
                session_provider_name=provider_name,
                provider_settings=self._durable_provider_settings,
                runtime_provider_config=runtime_provider_config,
                dynamic_provider=dynamic_provider,
                owns_initial_provider=False,
                defer_authoritative_writes=dynamic_provider is not None,
                thinking_level=thinking_level,
                index_on_first_persist=True,
                extension_runtime=self._extension_runtime,
            )
        )
        await self._adopt_replacement(replacement, reason="new")
        return f"Started new session: {record.id}"

    async def _adopt_replacement(
        self,
        replacement: CodingSession,
        *,
        reason: Literal["new", "resume", "branch"],
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Adopt a freshly loaded replacement session into this object.
        # Pre-publication (cancellable; on any failure the replacement is closed and the
        # error re-raised): reject a cancelled trust decision (ValueError), commit the
        # replacement's staged entries to storage, emit session_shutdown to the old runtime
        # and clear its UI, emit session_start on the new runtime, commit trust, and clear
        # the pending start flag. Publication (synchronous): retire the old runtime, copy all
        # state fields (config, state, harness, resources, provider state, catalogs, caches,
        # pending writes, runtime, credentials, logger, image support, trust), move
        # persistence listening to this object, take over provider ownership, rebind the
        # extension runtime to this session and attach its harness listener; finally close
        # the old runtime with cancellation contained.
        """Adopt a replacement session's state and re-bind the extension runtime.

        The extension runtime is long-lived and shared with the replacement; it
        must be re-bound to this outer session object because later state
        (transcript persistence, parent ids) mutates here, not on the discarded
        replacement instance.
        """
        old_runtime = self._extension_runtime
        try:
            if (
                replacement.project_trust_resolution is not None
                and replacement.project_trust_resolution.cancelled
            ):
                raise ValueError("Project trust decision cancelled; current session unchanged")

            # Destination transcript initialization/repair is the durable
            # commit point. It must succeed before the outgoing runtime enters
            # its shutdown path.
            await replacement._commit_prepared_entries()

            # The replacement remains the explicit owner of its providers
            # through every cancellable/erroring pre-publication seam.
            await old_runtime.emit_session_shutdown(reason)
            old_runtime.clear_ui_components()
            await replacement._extension_runtime.emit_session_start(reason)
            replacement._commit_project_trust_resolution()
            replacement._session_start_pending = False
        except BaseException:
            await _finish_aborted_session_close(replacement)
            raise

        # Every cancellable boundary has completed. Adopt synchronously so a
        # reported cancellation cannot expose only part of the destination.
        old_runtime.retire()
        self._config = replacement._config
        self._state = replacement._state
        self._harness = replacement._harness
        self._preview_model = None
        self._preview_thinking = None
        # Detach the replacement's persistence listener so writes advance
        # this session's parent pointers, not the discarded replacement's.
        if replacement._persistence_unsubscribe is not None:
            replacement._persistence_unsubscribe()
            replacement._persistence_unsubscribe = None
        self._attach_persistence_listener()
        self._invalidate_context_usage_cache()
        self._last_parent_id = replacement._last_parent_id
        self._skills = replacement._skills
        self._prompt_templates = replacement._prompt_templates
        self._context_files = replacement._context_files
        self._custom_system_prompt = replacement._custom_system_prompt
        self._custom_system_prompt_path = replacement._custom_system_prompt_path
        self._append_system_prompt = replacement._append_system_prompt
        self._append_system_prompts = replacement._append_system_prompts
        self._append_system_prompt_paths = replacement._append_system_prompt_paths
        self._resource_diagnostics = replacement._resource_diagnostics
        self._command_registry = replacement._command_registry
        self._provider_name = replacement._provider_name
        self._inference_provider = replacement._inference_provider
        self._inference_provider_mode = replacement._inference_provider_mode
        self._provider_settings = replacement._provider_settings
        self._durable_provider_settings = replacement._durable_provider_settings
        self._runtime_model_catalogs = replacement._runtime_model_catalogs
        self._model_catalog_discovery_errors = replacement._model_catalog_discovery_errors
        self._runtime_provider_config = replacement._runtime_provider_config
        self._resource_paths = replacement._resource_paths
        self._auto_compact_token_threshold = replacement._auto_compact_token_threshold
        self._auto_compact_enabled = replacement._auto_compact_enabled
        self._thinking_level = replacement._thinking_level
        self._pending_initial_entries = replacement._pending_initial_entries
        self._pending_message_writes = replacement._pending_message_writes
        self._extension_runtime = replacement._extension_runtime
        self._provider_registry = replacement._provider_registry
        self._credential_store = replacement._credential_store
        self._diagnostic_logger = replacement._diagnostic_logger
        self._runtime_model_limits = replacement._runtime_model_limits
        self._runtime_model_limits_key = replacement._runtime_model_limits_key
        self._model_limits_discovery_error = replacement._model_limits_discovery_error
        self._prepared_entries = replacement._prepared_entries
        self._owned_providers.extend(replacement._owned_providers)
        replacement._owned_providers.clear()
        self._image_support = replacement._image_support
        self._project_trust_resolution = replacement._project_trust_resolution
        self._project_trust_commit_pending = False
        self._session_start_pending = False
        self._extension_runtime.bind(self)
        self._extension_runtime.attach_harness_listener(self._harness.subscribe)
        # Adoption is already committed. Finish outgoing cleanup under a
        # shielded owner and contain cancellation rather than reporting that
        # the requested destination failed to replace the source session.
        await _finish_adopted_runtime_close(old_runtime)

    async def compact_detailed(self, instructions: str | None = None) -> ManualCompactionResult:
        # [FUNCTION DOC]
        # Manually compact older context (/compact) while keeping a recent-entry boundary.
        # Flushes pending writes, builds a plan (ValueError if there is not enough context),
        # estimates tokens before, generates a summary with optional custom instructions,
        # appends a compaction entry (rebuilding active context), and returns a structured
        # result with before/after token estimates. Side effects: model call, storage
        # append.
        """Compact older context while preserving a real recent-entry boundary."""
        await self._flush_pending_message_writes(context=self._diagnostic_context())
        plan = self._recent_preserving_compaction_plan()
        if plan is None:
            raise ValueError("Not enough context to compact while preserving recent entries")
        tokens_before = self.context_token_estimate
        generated = await self._generate_compaction_summary(
            plan.messages_to_summarize,
            custom_instructions=instructions,
        )
        await self._append_compaction(
            generated.text,
            first_kept_entry_id=plan.first_kept_entry_id,
            tokens_before=tokens_before,
            usage=generated.usage,
            provider=generated.provider,
            model=generated.model,
            response_provider=generated.response_provider,
        )
        return ManualCompactionResult(
            summary=generated.text,
            first_kept_entry_id=plan.first_kept_entry_id,
            tokens_before=tokens_before,
            estimated_tokens_after=self.context_token_estimate,
            replaced_entry_count=plan.replaced_entry_count,
        )

    async def compact(self, instructions: str | None = None) -> str:
        # [FUNCTION DOC]
        # Run manual compaction and return "Compacted <n> context entries.".
        """Generate a manual compaction summary and rebuild active context."""
        result = await self.compact_detailed(instructions)
        return f"Compacted {result.replaced_entry_count} context entries."

    async def aclose(self) -> None:
        # [FUNCTION DOC]
        # Close all owned extension runtime and provider resources exactly once. Starts (or
        # reuses) a durable close task, waits for it without letting cancellation interrupt
        # it, then re-raises a caller cancellation (as CancelledError) or the close task's
        # error. Later calls observe the same task idempotently.
        """Close every owned extension/provider resource exactly once.

        Caller cancellation is remembered but cannot cancel the durable close
        task. The first call propagates it only after all ownership ledgers are
        discharged; later calls observe the same completed task idempotently.
        """
        close_task = self._close_task
        if close_task is None:
            close_task = asyncio.create_task(
                self._close_owned_resources(),
                name="tau-coding-session-close",
            )
            self._close_task = close_task

        cancelled = await _await_cleanup_completion(close_task)
        error: BaseException | None = None
        try:
            close_task.result()
        except BaseException as exc:  # all resources were attempted before this outcome
            error = exc
        if cancelled:
            raise asyncio.CancelledError
        if error is not None:
            raise error

    async def _close_owned_resources(self) -> None:
        # [FUNCTION DOC]
        # HELPER: The single close pass: emit session_shutdown ("quit") if the extension
        # runtime is active, clear its UI components, close the runtime, then close every
        # owned provider (each removed from the ledger before its one close attempt). Each
        # step runs even if earlier ones fail; the first error is re-raised at the end.
        """Run the sole close pass, continuing after individual failures."""
        error: BaseException | None = None
        try:
            if self._extension_runtime.active:
                await self._extension_runtime.emit_session_shutdown("quit")
        except BaseException as exc:
            error = exc

        # Final close has no successor sharing the UI bridge. Remove any
        # source-owned widgets/interceptors before invalidating the API.
        try:
            self._extension_runtime.clear_ui_components()
        except BaseException as exc:
            if error is None:
                error = exc

        # A runtime may already be synchronously retired by replacement. Close
        # still owns the async drain/containment step and must never skip it.
        try:
            await self._extension_runtime.aclose()
        except BaseException as exc:
            if error is None:
                error = exc

        # Remove each provider from the ownership ledger before its only close
        # attempt. One hostile provider cannot prevent later providers closing.
        providers = tuple(self._owned_providers)
        self._owned_providers.clear()
        for provider in providers:
            try:
                await provider.aclose()
            except BaseException as exc:
                if error is None:
                    error = exc

        if error is not None:
            raise error

    def handle_command(self, text: str) -> CommandResult:
        # [FUNCTION DOC]
        # Handle a slash command synchronously. Prompt-template commands are reported as not
        # handled (they are expanded later by prompt); everything else goes to the command
        # registry. Returns the command result.
        """Handle coding-session slash commands.

        Prompt-template slash commands are expansion directives, so they remain
        unhandled here and flow through `prompt()` for on-the-fly replacement.
        """
        if expand_prompt_template_command(text, self._prompt_templates) is not None:
            return CommandResult(handled=False)
        return self._command_registry.execute(self, text)

    def ensure_session_indexed(self) -> None:
        # [FUNCTION DOC]
        # Add the session to the resume index if it has an id and manager and no record
        # exists yet (recording cwd, model, provider and routing), clear the
        # index-on-first-persist flag, and write any pending initial session-file entries.
        # Side effects: index and session file writes.
        """Persist pending session metadata and add this session to the resume index."""
        if self._config.session_id is None or self._config.session_manager is None:
            return
        if self._config.session_manager.get_session(self._config.session_id) is None:
            self._config.session_manager.create_session(
                cwd=self.cwd,
                model=self.model,
                provider_name=self.provider_name,
                inference_provider=self._inference_provider,
                inference_provider_mode=self._inference_provider_mode,
                session_id=self._config.session_id,
            )
        self._config = replace(self._config, index_on_first_persist=False)
        self._ensure_session_file_initialized()

    def expand_prompt_text(self, text: str) -> str:
        # [FUNCTION DOC]
        # Expand input text using prompt templates first (slash template commands), then
        # skill commands (/skill:name); return the text unchanged if neither matches.
        # May raise ResourceError from expansion.
        """Expand prompt text using loaded markdown resources."""
        expanded_prompt = expand_prompt_template_command(text, self._prompt_templates)
        if expanded_prompt is not None:
            return expanded_prompt
        expanded_skill = expand_skill_command(text, self._skills)
        return expanded_skill if expanded_skill is not None else text

    async def run_terminal_command(
        self,
        command: str,
        *,
        add_to_context: bool,
    ) -> TerminalCommandResult:
        # [FUNCTION DOC]
        # Run a shell command from the input bar in the session cwd using the bash tool
        # (with the configured command prefix). Raises ValueError for empty commands.
        # Extracts an integer exit code from the tool details. When add_to_context is set,
        # flushes pending writes and appends a user message describing the command and its
        # output to the harness and storage.
        # Output: a terminal command result (ok means exit code 0).
        # Side effects: process execution, optional transcript/storage writes.
        """Run a shell command in the session cwd, optionally adding output to context."""
        normalized_command = command.strip()
        if not normalized_command:
            raise ValueError("Terminal command cannot be empty")

        bash_tool = create_bash_tool(
            cwd=self.cwd,
            shell_command_prefix=self._config.shell_command_prefix,
        )
        result = await bash_tool.execute("terminal-command", {"command": normalized_command})
        exit_code = None
        if isinstance(result.details, dict):
            raw_exit_code = result.details.get("exit_code")
            exit_code = raw_exit_code if isinstance(raw_exit_code, int) else None

        if add_to_context:
            await self._flush_pending_message_writes(context=self._diagnostic_context())
            context_message = UserMessage(
                content=_terminal_command_context_message(
                    normalized_command,
                    result.text,
                )
            )
            self._harness.append_message(context_message)
            self._invalidate_context_usage_cache()
            await self._persist_message(context_message)

        return TerminalCommandResult(
            command=normalized_command,
            output=result.text,
            exit_code=exit_code,
            ok=exit_code == 0,
            added_to_context=add_to_context,
        )

    async def prompt(
        self,
        content: str,
        *,
        streaming_behavior: StreamingBehavior | None = None,
        source: Literal["interactive", "extension"] = "interactive",
        custom_type: str | None = None,
        details: dict[str, JSONValue] | None = None,
    ) -> AsyncIterator[CodingSessionEvent]:
        # [FUNCTION DOC]
        # Main async-generator entry for a user turn; yields coding-session events.
        # Inputs: content; keyword streaming behavior ("steer"/"follow_up" for queuing while
        # running), source ("interactive" or "extension") for input hooks, optional custom
        # message type and details.
        # Behavior: run extension input hooks (a handled input may show a notification and
        # ends without events); expand templates/skills (unexpected errors are logged).
        # If already running: queue as steering or follow-up and yield a queue-update
        # event, or raise RuntimeError without a streaming behavior. Otherwise flush pending
        # writes, commit previews, refresh model limits, possibly auto-compact first, then
        # prompt the harness with a user or custom message and forward its events: tool
        # ends invalidate context accounting, assistant errors are logged and classified
        # as context overflow or Hugging Face route failure, agent-end is wrapped as a
        # session agent-end with will_retry, and after the first user message-end the
        # session may be auto-named. On overflow: emit compaction start/end events, compact,
        # and if successful emit auto-retry start, continue the harness once, forward its
        # events, and emit auto-retry end. On route failure: run the failover retry. Else:
        # try auto-compaction after the prompt. Unexpected exceptions are logged and
        # re-raised. Finally the event stream is closed, missing message persists are
        # retried, and an agent-settled event is emitted (to extensions and yielded).
        # Side effects: model calls, storage writes, extension events, logs.
        """Append a user prompt, run the agent, and persist new messages.

        ``custom_type``/``details`` attach custom-message render metadata to the
        appended ``UserMessage`` (used when an extension delivers a custom
        message that starts an idle session's turn). ``source`` marks who
        initiated the turn for the `input` hook (``"extension"`` when an
        extension started it, ``"interactive"`` otherwise).
        """
        context = self._diagnostic_context()
        input_outcome = await self._extension_runtime.run_input_hooks(
            content, source=source, streaming_behavior=streaming_behavior
        )
        if input_outcome.handled:
            if input_outcome.message:
                self._extension_runtime.ui.notify(input_outcome.message)
            return
        content = input_outcome.text
        try:
            expanded_content = self.expand_prompt_text(content)
        except ResourceError:
            raise
        except Exception as exc:
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase="expand_prompt",
                exc=exc,
            )
            raise

        if self._harness.is_running:
            if streaming_behavior == "steer":
                self._harness.steer(expanded_content)
                session_event_0 = self.queue_update_event()
                await self._extension_runtime.emit_event(session_event_0)
                yield session_event_0
                return
            if streaming_behavior == "follow_up":
                self._harness.follow_up(expanded_content)
                session_event_0 = self.queue_update_event()
                await self._extension_runtime.emit_event(session_event_0)
                yield session_event_0
                return
            raise RuntimeError(
                "CodingSession is already running; pass streaming_behavior to queue a message."
            )

        await self._flush_pending_message_writes(context=context)
        await self._commit_preview()
        await self._refresh_runtime_model_limits()
        await self._try_auto_compact(context=context, phase="auto_compact_before_prompt")
        # id() values can be reused once earlier message objects are freed.
        self._ended_message_ids.clear()
        self._persisted_message_ids.clear()
        events: AsyncIterator[AgentEvent] | None = None
        settled_event: AgentSettledEvent | None = None
        auto_name_attempted = False
        overflow_message: AssistantMessage | None = None
        route_failure_message: AssistantMessage | None = None
        try:
            prompt_message: AgentMessage
            if custom_type is not None:
                prompt_message = CustomMessage(
                    custom_type=custom_type,
                    content=expanded_content,
                    display=True,
                    details=details,
                )
            else:
                prompt_message = UserMessage(content=expanded_content)
            events = self._harness.prompt_message(prompt_message)
            self._invalidate_context_usage_cache()
            async for event in events:
                auto_name_message: str | None = None
                if (
                    isinstance(event, MessageEndEvent)
                    and not auto_name_attempted
                    and isinstance(event.message, UserMessage)
                ):
                    auto_name_attempted = True
                    auto_name_message = event.message.text
                if isinstance(event, ToolExecutionEndEvent):
                    self._invalidate_context_usage_cache()
                if (
                    isinstance(event, MessageEndEvent)
                    and isinstance(event.message, AssistantMessage)
                    and event.message.stop_reason == "error"
                ):
                    self._last_diagnostic_log_path = self._diagnostic_logger.log_assistant_error(
                        context=context,
                        phase="agent_loop",
                        message=event.message,
                    )
                    if is_context_overflow_error(event.message):
                        overflow_message = event.message
                    elif self._should_auto_failover_huggingface_route(event.message):
                        route_failure_message = event.message
                if isinstance(event, AgentEndEvent):
                    yield SessionAgentEndEvent(
                        messages=event.messages,
                        will_retry=(
                            overflow_message is not None or route_failure_message is not None
                        ),
                    )
                else:
                    yield event
                # Let frontends render the confirmed, expanded prompt before
                # session naming performs its separate provider request.
                if auto_name_message is not None:
                    await self._try_auto_name_session(auto_name_message, context=context)
            if overflow_message is not None:
                session_event_1 = CompactionStartEvent(reason="overflow")
                await self._extension_runtime.emit_event(session_event_1)
                yield session_event_1
                compacted = await self._try_overflow_compact(context=context)
                compaction_end = CompactionEndEvent(
                    reason="overflow",
                    result=None,
                    aborted=not compacted,
                    will_retry=compacted,
                    error_message=None if compacted else "Overflow compaction failed",
                )
                await self._extension_runtime.emit_event(compaction_end)
                yield compaction_end
                if compacted:
                    retry_start = AutoRetryStartEvent(
                        attempt=1,
                        max_attempts=1,
                        delay_ms=0,
                        error_message=overflow_message.error_message or "Context overflow",
                    )
                    await self._extension_runtime.emit_event(retry_start)
                    yield retry_start
                    events = self._harness.continue_()
                    self._invalidate_context_usage_cache()
                    overflow_retry_error: str | None = None
                    async for retry_event in events:
                        if isinstance(retry_event, ToolExecutionEndEvent):
                            self._invalidate_context_usage_cache()
                        if (
                            isinstance(retry_event, MessageEndEvent)
                            and isinstance(retry_event.message, AssistantMessage)
                            and retry_event.message.stop_reason in {"error", "aborted"}
                        ):
                            overflow_retry_error = (
                                retry_event.message.error_message or "Provider request aborted"
                            )
                            if retry_event.message.stop_reason == "error":
                                self._last_diagnostic_log_path = (
                                    self._diagnostic_logger.log_assistant_error(
                                        context=context,
                                        phase="agent_loop_retry",
                                        message=retry_event.message,
                                    )
                                )
                        if isinstance(retry_event, AgentEndEvent):
                            yield SessionAgentEndEvent(
                                messages=retry_event.messages,
                                will_retry=False,
                            )
                        else:
                            yield retry_event
                    session_event_4 = AutoRetryEndEvent(
                        success=overflow_retry_error is None,
                        attempt=1,
                        final_error=overflow_retry_error,
                    )
                    await self._extension_runtime.emit_event(session_event_4)
                    yield session_event_4
            elif route_failure_message is not None:
                async for failover_event in self._run_huggingface_route_failover(context=context):
                    yield failover_event
            else:
                await self._try_auto_compact(context=context, phase="auto_compact_after_prompt")
        except Exception as exc:
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase="agent_loop",
                exc=exc,
            )
            raise
        finally:
            try:
                await self._reconcile_run_persistence(events, context=context)
            finally:
                if events is not None:
                    settled_event = await self._dispatch_agent_settled()
        if settled_event is not None:
            yield settled_event

    async def continue_(self) -> AsyncIterator[CodingSessionEvent]:
        # [FUNCTION DOC]
        # Async generator that continues the agent from the restored transcript (no new
        # user message). Flushes pending writes, refreshes model limits, forwards harness
        # continue events (logging assistant errors, wrapping agent-end, invalidating
        # context accounting on tool ends), runs Hugging Face route failover if needed, then
        # tries auto-compaction. Errors are logged and re-raised. Finally reconciles
        # persistence and yields an agent-settled event.
        """Continue the agent from restored state and persist new messages."""
        context = self._diagnostic_context()
        await self._flush_pending_message_writes(context=context)
        await self._refresh_runtime_model_limits()
        # id() values can be reused once earlier message objects are freed.
        self._ended_message_ids.clear()
        self._persisted_message_ids.clear()
        events: AsyncIterator[AgentEvent] | None = None
        settled_event: AgentSettledEvent | None = None
        route_failure_message: AssistantMessage | None = None
        try:
            events = self._harness.continue_()
            self._invalidate_context_usage_cache()
            async for event in events:
                if isinstance(event, ToolExecutionEndEvent):
                    self._invalidate_context_usage_cache()
                if (
                    isinstance(event, MessageEndEvent)
                    and isinstance(event.message, AssistantMessage)
                    and event.message.stop_reason == "error"
                ):
                    self._last_diagnostic_log_path = self._diagnostic_logger.log_assistant_error(
                        context=context,
                        phase="agent_loop",
                        message=event.message,
                    )
                    if self._should_auto_failover_huggingface_route(event.message):
                        route_failure_message = event.message
                if isinstance(event, AgentEndEvent):
                    yield SessionAgentEndEvent(
                        messages=event.messages,
                        will_retry=route_failure_message is not None,
                    )
                else:
                    yield event
            if route_failure_message is not None:
                async for failover_event in self._run_huggingface_route_failover(context=context):
                    yield failover_event
            await self._try_auto_compact(context=context, phase="auto_compact_after_continue")
        except Exception as exc:
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase="agent_loop",
                exc=exc,
            )
            raise
        finally:
            try:
                await self._reconcile_run_persistence(events, context=context)
            finally:
                if events is not None:
                    settled_event = await self._dispatch_agent_settled()
        if settled_event is not None:
            yield settled_event

    async def _dispatch_agent_settled(self) -> AgentSettledEvent:
        # [FUNCTION DOC]
        # HELPER: Create an agent-settled event, emit it to extensions, and return it.
        """Dispatch and return the final session event for one started run."""
        event = AgentSettledEvent()
        await self._extension_runtime.emit_event(event)
        return event

    def _diagnostic_context(self) -> AgentCallDiagnosticContext:
        # [FUNCTION DOC]
        # HELPER: Build a diagnostics context (provider, model, cwd, session id, and a fresh
        # run id) for logging one agent call.
        return AgentCallDiagnosticContext(
            provider_name=self._provider_name,
            model=self.model,
            cwd=self.cwd,
            session_id=self.session_id,
            run_id=new_agent_call_run_id(),
        )

    async def _persist_active_tool_history_repairs(self) -> ToolHistoryRepair | None:
        # [FUNCTION DOC]
        # HELPER: Detect malformed tool history on the active path (e.g. tool calls without
        # results) and write one complete repair branch: a "tau.session-history-repair"
        # custom entry with diagnostic data, the repaired message suffix, a model-change
        # entry (if a model is known), a thinking-level entry, and copies of other custom
        # entries that were after the repair point. With deferred writes the entries are
        # staged in memory and state is replayed from them; otherwise they are appended and
        # state refreshed. Replaces harness messages and invalidates context accounting.
        # Returns the repair description, or None when nothing needed repair.
        """Stage or append one complete repair branch for malformed history."""
        plan = _tool_history_repair_plan(
            self._state.messages,
            context_entry_ids=self._state.context_entry_ids,
            entries=self._state.entries,
        )
        if plan is None:
            return None

        parent_id, suffix, repair = plan
        active_model = self._state.model
        active_thinking_level = self._state.thinking_level
        active_entries = self._state.entries
        parent_index = next(
            (index for index, entry in enumerate(active_entries) if entry.id == parent_id),
            -1,
        )
        custom_entries = [
            entry
            for entry in active_entries[parent_index + 1 :]
            if isinstance(entry, CustomEntry) and entry.namespace != "tau.session-history-repair"
        ]
        staged: list[SessionEntry] = [
            CustomEntry(
                parent_id=parent_id,
                namespace="tau.session-history-repair",
                data={"version": 1, **repair.diagnostic_data()},
            )
        ]
        parent_id = staged[-1].id
        for message in suffix:
            entry = _session_entry_for_message(parent_id=parent_id, message=message)
            staged.append(entry)
            parent_id = entry.id
        if active_model is not None:
            model_entry = ModelChangeEntry(
                parent_id=parent_id,
                model=active_model,
                provider=self.provider_name,
            )
            staged.append(model_entry)
            parent_id = model_entry.id
        thinking_entry = ThinkingLevelChangeEntry(
            parent_id=parent_id,
            thinking_level=active_thinking_level,
        )
        staged.append(thinking_entry)
        parent_id = thinking_entry.id
        for custom_entry in custom_entries:
            copied_entry = CustomEntry(
                parent_id=parent_id,
                namespace=custom_entry.namespace,
                data=custom_entry.data,
            )
            staged.append(copied_entry)
            parent_id = copied_entry.id
        if self._config.defer_authoritative_writes:
            self._prepared_entries.extend(staged)
            self._last_parent_id = parent_id
            replay_entries = [*self._state.entries, *staged]
            self._state = SessionState.from_entries(replay_entries, leaf_id=parent_id)
        else:
            await self._append_session_batch(staged)
            self._last_parent_id = parent_id
            await self._refresh_persisted_state(leaf_id=parent_id)
        self._harness.replace_messages(self._state.messages)
        self._invalidate_context_usage_cache()
        return repair

    def _attach_persistence_listener(self) -> None:
        # [FUNCTION DOC]
        # HELPER: (Re-)subscribe the message-end persistence callback to the current harness,
        # unsubscribing any previous subscription. Harness stubs without subscribe are
        # tolerated.
        """(Re-)attach push persistence to the current harness.

        Persistence subscribes to harness events rather than running in the
        event consumer, which the TUI tears down on interrupt.
        """
        if self._persistence_unsubscribe is not None:
            self._persistence_unsubscribe()
            self._persistence_unsubscribe = None
        # Command-only tests construct sessions with stub harnesses.
        subscribe = getattr(self._harness, "subscribe", None)
        if subscribe is not None:
            self._persistence_unsubscribe = subscribe(self._persist_on_message_end)

    async def _persist_on_message_end(self, event: AgentEvent) -> None:
        # [FUNCTION DOC]
        # HELPER: Harness listener: on message-end events, record the message identity as
        # ended and persist it.
        if isinstance(event, MessageEndEvent):
            self._ended_message_ids.add(id(event.message))
            await self._persist_message(event.message)

    async def _persist_message(self, message: AgentMessage) -> None:
        # [FUNCTION DOC]
        # HELPER: Persist one completed message at the active branch tip idempotently. The
        # entry is created once per message object (keyed by object identity) and reused on
        # retries; on retry, storage is read to skip re-appending an entry that already
        # reached disk. Advances the parent id, refreshes state, records the message as
        # persisted, drops the pending record, and invalidates context accounting.
        # Raises storage errors (leaving the pending record for a later retry).
        """Persist one completed message at the active branch tip, idempotently.

        Message lifecycle events are the durable-message boundary. A stable entry
        id lets a retry recognize an append that reached disk before raising.
        Only retries pay for a full-file read.
        """
        message_id = id(message)
        pending = self._pending_message_writes.get(message_id)
        is_retry = pending is not None
        if pending is None:
            entry = _session_entry_for_message(parent_id=self._last_parent_id, message=message)
            pending = _PendingMessageWrite(message=message, entry=entry)
            self._pending_message_writes[message_id] = pending

        durable_ids = (
            {entry.id for entry in await self._read_session_entries()} if is_retry else frozenset()
        )
        if pending.entry.id not in durable_ids:
            await self._append_session_entry(pending.entry)
        self._last_parent_id = pending.entry.id

        await self._refresh_persisted_state(leaf_id=self._last_parent_id)
        self._persisted_message_ids.add(message_id)
        self._pending_message_writes.pop(message_id, None)
        self._invalidate_context_usage_cache()

    async def _reconcile_run_persistence(
        self,
        events: AsyncIterator[AgentEvent] | None,
        *,
        context: AgentCallDiagnosticContext,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: At the end of a run: close the event stream (errors ignored), then retry
        # persisting every transcript message that ended but was not persisted (failures are
        # logged, not raised, to avoid masking cancellation), and clear the tracking sets.
        """Close a run and retry failed persists still present in the transcript.

        Keyed on message identity, not counts: the loop emits an assistant's
        ``message_end`` before appending it to the transcript. A message whose
        persist and append both failed cannot be retried here; the repair at
        the next run start re-synthesizes and persists its tool result.
        """
        if events is not None:
            aclose = getattr(events, "aclose", None)
            if aclose is not None:
                with suppress(Exception):
                    await aclose()
        for message in self._harness.messages:
            message_id = id(message)
            if (
                message_id in self._ended_message_ids
                and message_id not in self._persisted_message_ids
            ):
                # Runs in a finally: a repeat failure must not mask cancellation.
                try:
                    await self._persist_message(message)
                except Exception as exc:  # noqa: BLE001 - preserve cancellation
                    self._log_persistence_failure(context=context, exc=exc)
        self._ended_message_ids.clear()
        self._persisted_message_ids.clear()

    async def _flush_pending_message_writes(self, *, context: AgentCallDiagnosticContext) -> None:
        # [FUNCTION DOC]
        # HELPER: Before storage-dependent actions, retry every pending message write,
        # re-appending the message to the harness transcript first if it is missing there.
        # Failures are logged and re-raised.
        """Finish earlier partial writes before another storage-dependent action."""
        harness_message_ids = {id(message) for message in self._harness.messages}
        for pending in tuple(self._pending_message_writes.values()):
            if id(pending.message) not in harness_message_ids:
                self._harness.append_message(pending.message)
                harness_message_ids.add(id(pending.message))
            try:
                await self._persist_message(pending.message)
            except Exception as exc:
                self._log_persistence_failure(context=context, exc=exc)
                raise

    def _log_persistence_failure(
        self, *, context: AgentCallDiagnosticContext, exc: BaseException
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Log a persistence exception to the diagnostic logger (logging errors are
        # ignored) and remember the log path.
        with suppress(Exception):
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase="session_persistence_reconcile",
                exc=exc,
            )

    def _invalidate_context_usage_cache(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Mark the cached context usage estimate dirty.
        """Mark context accounting dirty after transcript/system/tool changes."""
        self._context_usage_cache = None

    async def _refresh_persisted_state(self, *, leaf_id: str | None) -> None:
        # [FUNCTION DOC]
        # HELPER: Re-read storage, replay state for the given leaf, and touch the session
        # index record with the current model, provider and routing (when indexed).
        entries = await self._read_session_entries()
        self._state = SessionState.from_entries(entries, leaf_id=leaf_id)
        if self._config.session_id is not None and self._config.session_manager is not None:
            self._config.session_manager.touch_session(
                self._config.session_id,
                model=self.model,
                provider_name=self.provider_name,
                inference_provider=self._inference_provider,
                inference_provider_mode=self._inference_provider_mode,
                preserve_inference_provider=False,
            )

    async def _read_session_entries(self) -> list[SessionEntry]:
        # [FUNCTION DOC]
        # HELPER: Read all stored entries and detach parents that point to missing entries
        # (e.g. imported external history), returning a list.
        """Read stored entries, detaching roots imported from external history."""
        return _detach_missing_parents(await self._config.storage.read_all())

    async def _commit_prepared_entries(self) -> None:
        # [FUNCTION DOC]
        # HELPER: When writes were deferred, append the staged entries not already on disk
        # (as one batch when the storage supports it, else one by one), clear staged and
        # pending initial entries, stop deferring writes, and index the session if
        # requested (index errors become a warning diagnostic). No-op when not deferred.
        """Durably commit the staged startup/repair batch exactly once."""
        if not self._config.defer_authoritative_writes:
            return
        durable_ids = {entry.id for entry in await self._config.storage.read_all()}
        missing = tuple(entry for entry in self._prepared_entries if entry.id not in durable_ids)
        if missing:
            append_batch = getattr(self._config.storage, "append_batch", None)
            if append_batch is None:
                # Compatibility storage implementations predate the atomic
                # batch contract.  Production storage always takes this path.
                for entry in missing:
                    await self._config.storage.append(entry)
            else:
                await append_batch(missing)
        self._pending_initial_entries = ()
        self._prepared_entries.clear()
        self._config = replace(self._config, defer_authoritative_writes=False)
        try:
            if self._config.index_on_first_persist:
                self._index_current_session()
        except Exception as exc:  # index is a rebuildable cache, not authority
            self._record_session_index_diagnostic(exc)

    async def _append_session_entry(self, entry: SessionEntry) -> None:
        # [FUNCTION DOC]
        # HELPER: Ensure initial/staged entries are written, then append one entry.
        """Append one durable entry after flushing deferred session metadata."""
        await self._ensure_session_initialized()
        await self._config.storage.append(entry)

    async def _append_session_batch(self, entries: Sequence[SessionEntry]) -> None:
        # [FUNCTION DOC]
        # HELPER: Ensure initialization, then append several entries atomically via the
        # storage batch method, falling back to sequential appends for legacy storage.
        """Append entries as one storage transaction, with legacy fallback."""
        await self._ensure_session_initialized()
        append_batch = getattr(self._config.storage, "append_batch", None)
        if append_batch is None:
            for entry in entries:
                await self._config.storage.append(entry)
            return
        await append_batch(tuple(entries))

    async def _close_replaced_provider(self, provider: ModelProvider) -> None:
        # [FUNCTION DOC]
        # HELPER: Close a provider replaced by a committed switch, only if this session owns
        # it (removing it from the ledger first). Errors propagate to the caller.
        """Close a Tau-owned provider once after a committed publication."""
        if provider not in self._owned_providers:
            return
        self._owned_providers.remove(provider)
        close = getattr(provider, "aclose", None)
        if close is not None:
            await close()

    async def _ensure_session_initialized(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Before the first durable write: commit staged entries when writes are
        # deferred; otherwise write pending initial entries and index the session if
        # requested (index errors recorded as diagnostics).
        if self._config.defer_authoritative_writes and self._prepared_entries:
            await self._commit_prepared_entries()
            return
        if not self._pending_initial_entries:
            return
        await self._write_pending_initial_entries()
        if self._config.index_on_first_persist:
            try:
                self._index_current_session()
            except Exception as exc:
                self._record_session_index_diagnostic(exc)

    async def _write_pending_initial_entries(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Append the pending initial entries not already present in storage (batch
        # when supported) and clear them.
        durable_ids = {entry.id for entry in await self._config.storage.read_all()}
        missing = tuple(
            entry for entry in self._pending_initial_entries if entry.id not in durable_ids
        )
        if missing:
            append_batch = getattr(self._config.storage, "append_batch", None)
            if append_batch is not None:
                await append_batch(missing)
            else:
                for entry in missing:
                    await self._config.storage.append(entry)
        self._pending_initial_entries = ()

    def _ensure_session_file_initialized(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Synchronously append any pending initial entries to the storage file and
        # clear them (used by the synchronous index path).
        # KNOWN ISSUE: unlike the async path, it does not check which entries are already
        # on disk, and it ignores staged entries when writes are deferred.
        if not self._pending_initial_entries:
            return
        for entry in self._pending_initial_entries:
            _append_session_entry_sync(self._config.storage, entry)
        self._pending_initial_entries = ()

    def _record_session_index_diagnostic(self, exc: BaseException) -> None:
        # [FUNCTION DOC]
        # HELPER: Add a warning "session-index" diagnostic naming the exception type.
        self._resource_diagnostics = (
            *self._resource_diagnostics,
            ResourceDiagnostic(
                kind="session-index",
                message=f"Session index needs repair: {type(exc).__name__}",
                severity="warning",
            ),
        )

    def _index_current_session(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Create a session index record for this session if it has an id and
        # manager and no record exists yet.
        # KNOWN ISSUE: the inference provider mode is not passed, unlike the equivalent call
        # in ensure_session_indexed, so the index may record the default mode.
        if self._config.session_id is None or self._config.session_manager is None:
            return
        existing = self._config.session_manager.get_session(self._config.session_id)
        if existing is not None:
            return
        self._config.session_manager.create_session(
            cwd=self.cwd,
            model=self.model,
            provider_name=self.provider_name,
            inference_provider=self._inference_provider,
            session_id=self._config.session_id,
        )

    async def _try_auto_compact(
        self,
        *,
        context: AgentCallDiagnosticContext,
        phase: str,
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: Run automatic compaction if needed; any exception is logged with the
        # given phase and reported as False so the turn is not lost. Returns whether
        # compaction happened.
        try:
            return await self._maybe_auto_compact()
        except Exception as exc:  # noqa: BLE001 - automatic compaction must not lose a turn
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase=phase,
                exc=exc,
            )
            return False

    async def _try_overflow_compact(
        self,
        *,
        context: AgentCallDiagnosticContext,
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: Compact after a context-overflow error using the recent-preserving plan.
        # Returns False when no plan exists or on any exception (logged); True on success.
        try:
            plan = self._recent_preserving_compaction_plan()
            if plan is None:
                return False
            tokens_before = self.context_token_estimate
            generated = await self._generate_compaction_summary(plan.messages_to_summarize)
            await self._append_compaction(
                generated.text,
                first_kept_entry_id=plan.first_kept_entry_id,
                tokens_before=tokens_before,
                usage=generated.usage,
                provider=generated.provider,
                model=generated.model,
                response_provider=generated.response_provider,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - the original overflow remains visible
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase="overflow_compact",
                exc=exc,
            )
            return False

    async def _try_auto_name_session(
        self,
        first_message: str,
        *,
        context: AgentCallDiagnosticContext,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: After the first user message, name an unnamed indexed session: ask the
        # model for a short title (errors logged), fall back to a heuristic name from the
        # message, persist it only if still unnamed, and emit a session-info-changed event.
        if not self._should_auto_name_session():
            return
        try:
            title = await self._generate_session_name(first_message)
        except Exception as exc:  # noqa: BLE001 - naming must not interrupt the agent turn
            self._last_diagnostic_log_path = self._diagnostic_logger.log_exception(
                context=context,
                phase="auto_name_session",
                exc=exc,
            )
            title = _fallback_session_name(first_message)
        if title is None:
            title = _fallback_session_name(first_message)
        if title is None:
            return
        persisted = self._persist_session_name(
            title,
            only_if_unnamed=True,
            index_if_missing=False,
        )
        if persisted is not None:
            await self._extension_runtime.emit_event(SessionInfoChangedEvent(name=persisted))

    def _should_auto_name_session(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: True when the session is indexed-capable, its record (if any) has no
        # title, and the transcript contains exactly one user message.
        if self._config.session_id is None or self._config.session_manager is None:
            return False
        record = self._config.session_manager.get_session(self._config.session_id)
        if record is not None and record.title:
            return False
        return sum(isinstance(message, UserMessage) for message in self._harness.messages) == 1

    async def _generate_session_name(self, first_message: str) -> str | None:
        # [FUNCTION DOC]
        # HELPER: Ask the active provider (no tools, naming system prompt) for a session
        # name for the first message; collect text deltas or the final message text and
        # sanitize it. Returns the name or None. Raises RuntimeError on a provider error
        # event. Side effect: a model request.
        prompt = (
            "Create a concise session name for this first user message. "
            "Use at most four words.\n\n"
            f"User message:\n{first_message}"
        )
        text_parts: list[str] = []
        final_text: str | None = None
        async for event in self._harness.config.provider.stream_response(
            model=self.model,
            system=SESSION_NAME_SYSTEM_PROMPT,
            messages=[UserMessage(content=prompt)],
            tools=[],
        ):
            if isinstance(event, TextDeltaEvent):
                text_parts.append(event.delta)
            elif isinstance(event, AssistantDoneEvent):
                final_text = event.message.text
            elif isinstance(event, AssistantErrorEvent):
                raise RuntimeError(
                    f"Session naming failed: {event.error.error_message or event.reason}"
                )
        return _sanitize_session_name(final_text if final_text is not None else "".join(text_parts))

    def _provider_is_usable(self, provider: ProviderConfig) -> bool:
        # [FUNCTION DOC]
        # HELPER: Whether the provider config has usable credentials in the session's
        # credential store (or environment).
        return provider_has_usable_credentials(
            provider,
            credential_reader=self._credential_store,
        )

    def _usable_provider_configs(self) -> tuple[ProviderConfig, ...]:
        # [FUNCTION DOC]
        # HELPER: Tuple of configured providers with usable credentials (empty without
        # settings).
        if self._provider_settings is None:
            return ()
        return tuple(
            provider
            for provider in self._provider_settings.providers
            if self._provider_is_usable(provider)
        )

    async def _maybe_auto_compact(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: Automatically compact when enabled, the active context has at least two
        # entries, and the context estimate exceeds the threshold, using the
        # recent-preserving plan. Returns whether a compaction entry was appended.
        # Raises on summarization or storage failure.
        threshold = self.auto_compact_token_threshold
        if threshold is None or threshold <= 0:
            return False
        if len(self._state.context_entry_ids) < 2:
            return False
        if self.context_token_estimate <= threshold:
            return False
        plan = self._recent_preserving_compaction_plan()
        if plan is None:
            return False
        tokens_before = self.context_token_estimate
        generated = await self._generate_compaction_summary(plan.messages_to_summarize)
        await self._append_compaction(
            generated.text,
            first_kept_entry_id=plan.first_kept_entry_id,
            tokens_before=tokens_before,
            usage=generated.usage,
            provider=generated.provider,
            model=generated.model,
            response_provider=generated.response_provider,
        )
        return True

    async def _generate_compaction_summary(
        self,
        messages: tuple[AgentMessage, ...],
        *,
        custom_instructions: str | None = None,
    ) -> _GeneratedSummary:
        # [FUNCTION DOC]
        # HELPER: Ask the active model (summarization system prompt, no tools) to summarize
        # messages with optional custom instructions. Uses the final message text or
        # concatenated deltas, stripped. Returns the summary with summed usage, provider,
        # model and upstream provider. Raises RuntimeError on provider error events or an
        # empty summary.
        prompt = build_compaction_summary_prompt(
            messages,
            custom_instructions=custom_instructions,
        )
        text_parts: list[str] = []
        final_text: str | None = None
        response_usages: list[Usage] = []
        response_provider: str | None = None
        summary_messages: list[AgentMessage] = [UserMessage(content=prompt)]
        async for event in self._harness.config.provider.stream_response(
            model=self.model,
            system=SUMMARIZATION_SYSTEM_PROMPT,
            messages=summary_messages,
            tools=[],
        ):
            if isinstance(event, TextDeltaEvent):
                text_parts.append(event.delta)
            elif isinstance(event, AssistantDoneEvent):
                final_text = event.message.text
                response_usages.append(event.message.usage)
                response_provider = event.message.response_provider
            elif isinstance(event, AssistantErrorEvent):
                raise RuntimeError(
                    f"Compaction summarization failed: {event.error.error_message or event.reason}"
                )

        summary = (final_text if final_text is not None else "".join(text_parts)).strip()
        if not summary:
            raise RuntimeError("Compaction summarization returned an empty summary")
        return _GeneratedSummary(
            text=summary,
            usage=sum_usage(response_usages) if response_usages else None,
            provider=self.provider_name,
            model=self.model,
            response_provider=response_provider,
        )

    async def _summarize_branch_messages(
        self,
        messages: tuple[AgentMessage, ...],
        *,
        custom_instructions: str | None = None,
        replace_instructions: bool = False,
    ) -> _GeneratedSummary:
        # [FUNCTION DOC]
        # HELPER: Summarize abandoned branch messages with the model (custom instructions
        # optionally replacing defaults); any failure or empty result falls back to a local
        # heuristic summary without usage or provider attribution.
        try:
            result = await summarize_branch_messages_with_model(
                provider=self._harness.config.provider,
                model=self.model,
                messages=messages,
                custom_instructions=custom_instructions,
                replace_instructions=replace_instructions,
            )
        except Exception:
            result = None
        if result is not None:
            summary, usage, response_provider = result
            return _GeneratedSummary(
                text=summary,
                usage=usage,
                provider=self.provider_name,
                model=self.model,
                response_provider=response_provider,
            )
        return _GeneratedSummary(
            text=summarize_messages_for_compaction(messages),
            usage=None,
        )

    def _recent_preserving_compaction_plan(self) -> CompactionPlan | None:
        # [FUNCTION DOC]
        # HELPER: Plan a compaction that keeps roughly the most recent configured token
        # budget verbatim, cutting at a user-message boundary when possible. Returns None if
        # fewer than two context rows exist or the cut would keep everything or nothing.
        rows = self._active_context_rows()
        if len(rows) < 2:
            return None

        first_kept_index = _first_recent_context_index(
            rows,
            keep_recent_tokens=DEFAULT_COMPACTION_KEEP_RECENT_TOKENS,
        )
        if first_kept_index <= 0:
            return None

        if first_kept_index >= len(rows):
            return None

        replaced = rows[:first_kept_index]
        return CompactionPlan(
            first_kept_entry_id=rows[first_kept_index][0],
            replaced_entry_count=len(replaced),
            messages_to_summarize=tuple(message for _entry_id, message in replaced),
        )

    def _active_context_rows(self) -> tuple[tuple[str, AgentMessage], ...]:
        # [FUNCTION DOC]
        # HELPER: Pair each active context entry id with its message (strict zip; raises
        # ValueError on length mismatch).
        return tuple(zip(self._state.context_entry_ids, self._state.messages, strict=True))

    async def _append_compaction(
        self,
        summary: str,
        *,
        first_kept_entry_id: str,
        tokens_before: int | None = None,
        usage: Usage | None = None,
        provider: str | None = None,
        model: str | None = None,
        response_provider: str | None = None,
    ) -> CompactionEntry:
        # [FUNCTION DOC]
        # HELPER: Append a compaction entry (summary, first kept entry id, tokens before,
        # usage and attribution) at the active tip, refresh state, replace harness messages
        # with the compacted context, and invalidate accounting. Returns the entry.
        # Raises ValueError if the first kept entry is not in the active context.
        if first_kept_entry_id not in self._state.context_entry_ids:
            raise ValueError("First kept entry is not in the active context")

        compaction = CompactionEntry(
            parent_id=self._last_parent_id,
            summary=summary,
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
            usage=usage,
            provider=provider,
            model=model,
            response_provider=response_provider,
        )
        await self._append_session_entry(compaction)
        self._last_parent_id = compaction.id

        await self._refresh_persisted_state(leaf_id=compaction.id)
        self._harness.replace_messages(self._state.messages)
        self._invalidate_context_usage_cache()
        return compaction


def _first_recent_context_index(
    rows: tuple[tuple[str, AgentMessage], ...],
    *,
    keep_recent_tokens: int,
) -> int:
    # [FUNCTION DOC]
    # HELPER: Choose the index of the first context row to keep. Walking backwards,
    # accumulate estimated tokens until the keep budget is reached; if never reached,
    # return 0 (keep everything). If the boundary row is a user message, keep from it
    # (or, at index 0, from the next user message, else 0). Otherwise keep from the next
    # user message after it, else from the first non-tool-result row at or after it,
    # else the end. A non-positive budget returns the row count.
    if keep_recent_tokens <= 0:
        return len(rows)

    accumulated_tokens = 0
    candidate_index: int | None = None
    for index in range(len(rows) - 1, -1, -1):
        _entry_id, message = rows[index]
        accumulated_tokens += estimate_message_tokens(message)
        if accumulated_tokens >= keep_recent_tokens:
            candidate_index = index
            break

    if candidate_index is None:
        return 0

    candidate_message = rows[candidate_index][1]
    if candidate_message.role == "user":
        if candidate_index > 0:
            return candidate_index
        next_user_index = _next_user_message_index(rows, start=1)
        return next_user_index if next_user_index is not None else 0

    next_user_index = _next_user_message_index(rows, start=candidate_index + 1)
    if next_user_index is not None:
        return next_user_index

    for index in range(candidate_index, len(rows)):
        if rows[index][1].role != "toolResult":
            return index
    return len(rows)


def _next_user_message_index(
    rows: tuple[tuple[str, AgentMessage], ...],
    *,
    start: int,
) -> int | None:
    # [FUNCTION DOC]
    # HELPER: Index of the first user-role row at or after start, or None.
    for index in range(start, len(rows)):
        if rows[index][1].role == "user":
            return index
    return None


def is_context_overflow_error(message: AssistantMessage) -> bool:
    # [FUNCTION DOC]
    # Return True when an assistant error message contains any common context-overflow
    # phrase (case-insensitive), e.g. "context length", "prompt is too long",
    # "token limit". Pure heuristic.
    """Return True when an assistant error looks like a context overflow."""
    text = message.error_message or ""
    normalized = text.lower()
    markers = (
        "context length",
        "context window",
        "context limit",
        "maximum context",
        "max context",
        "input is too long",
        "input length",
        "prompt is too long",
        "too many tokens",
        "token limit",
        "exceeds the limit",
        "exceeded the limit",
    )
    return any(marker in normalized for marker in markers)


def is_retryable_huggingface_route_error(message: AssistantMessage) -> bool:
    # [FUNCTION DOC]
    # Return True when an assistant error produced no content and its first
    # provider_error diagnostic with an integer status code has status 408, 409, 425,
    # 429 or any 5xx, meaning rerouting is safe. Pure.
    """Return whether a pre-output Hugging Face HTTP failure is safe to reroute."""
    if message.content:
        return False
    for diagnostic in message.diagnostics or []:
        if diagnostic.type != "provider_error" or diagnostic.details is None:
            continue
        status_code = diagnostic.details.get("status_code")
        if isinstance(status_code, int) and not isinstance(status_code, bool):
            return status_code in {408, 409, 425, 429} or status_code >= 500
    return False


def _session_entry_for_message(
    *, parent_id: str | None, message: AgentMessage
) -> MessageEntry | CustomMessageEntry:
    # [FUNCTION DOC]
    # HELPER: Build the persisted entry for a completed message: custom messages become
    # custom-message entries (timestamp converted from milliseconds to seconds, with
    # type, content, display flag and details); all others become message entries.
    """Build the canonical persisted entry for one completed runtime message."""
    if isinstance(message, CustomMessage):
        return CustomMessageEntry(
            parent_id=parent_id,
            timestamp=message.timestamp / 1000,
            custom_type=message.custom_type,
            content=message.content,
            display=message.display,
            details=message.details,
        )
    return MessageEntry(parent_id=parent_id, message=message)


def _detach_missing_parents(entries: list[SessionEntry]) -> list[SessionEntry]:
    # [FUNCTION DOC]
    # HELPER: Return a new list where entries whose parent id refers to a missing entry
    # get a None parent (copied); other entries unchanged.
    """Return entries with dangling parent pointers detached from external history."""
    entry_ids = {entry.id for entry in entries}
    return [
        entry.model_copy(update={"parent_id": None})
        if entry.parent_id is not None and entry.parent_id not in entry_ids
        else entry
        for entry in entries
    ]


def _last_parent_id_from_state(state: SessionState) -> str | None:
    # [FUNCTION DOC]
    # HELPER: The parent for the next append: the active leaf id, else the last entry's
    # id, else None.
    if state.active_leaf_id is not None:
        return state.active_leaf_id
    if state.entries:
        return state.entries[-1].id
    return None


def _active_branchable_entry_id(
    entries: list[SessionEntry], active_leaf_id: str | None
) -> str | None:
    # [FUNCTION DOC]
    # HELPER: The id of the nearest branchable entry on the path from root to the active
    # leaf (searching from the leaf back). Returns None without a leaf or match; if the
    # path cannot be built, returns the leaf id itself.
    if active_leaf_id is None:
        return None
    try:
        path = path_to_entry(entries, active_leaf_id)
    except SessionTreeError:
        return active_leaf_id
    return next((entry.id for entry in reversed(path) if _is_branchable_tree_entry(entry)), None)


def _resolved_labels(entries: list[SessionEntry]) -> tuple[dict[str, str], dict[str, float]]:
    # [FUNCTION DOC]
    # HELPER: Walk label entries in storage order and return two mappings keyed by target
    # entry id: the latest non-blank label and its timestamp (blank labels clear).
    """Resolve the latest label change for every target in storage order."""
    labels: dict[str, str] = {}
    timestamps: dict[str, float] = {}
    for entry in entries:
        if not isinstance(entry, LabelEntry):
            continue
        label = entry.label.strip() if entry.label is not None else ""
        if label:
            labels[entry.target_id] = label
            timestamps[entry.target_id] = entry.timestamp
        else:
            labels.pop(entry.target_id, None)
            timestamps.pop(entry.target_id, None)
    return labels, timestamps


def _is_branchable_tree_entry(entry: SessionEntry) -> bool:
    # [FUNCTION DOC]
    # HELPER: Compaction and branch-summary entries, and message entries holding user or
    # assistant messages, are branchable; everything else is not.
    if entry.type in {"compaction", "branch_summary"}:
        return True
    if entry.type != "message":
        return False
    return isinstance(entry.message, UserMessage | AssistantMessage)


def _tree_choice_label(entry: SessionEntry, *, branch_indent: int = 0) -> str:
    # [FUNCTION DOC]
    # HELPER: Prefix the entry's display title with two spaces per indent level.
    prefix = "  " * branch_indent
    return f"{prefix}{_tree_entry_title(entry)}"


def _tree_branch_indents(entries: list[SessionEntry]) -> dict[str, int]:
    # [FUNCTION DOC]
    # HELPER: Return just the per-entry indentation mapping from the tree layout.
    return _tree_layout(entries)[1]


def _ordered_tree_entries(entries: list[SessionEntry]) -> tuple[SessionEntry, ...]:
    # [FUNCTION DOC]
    # HELPER: Return just the display-ordered entries from the tree layout.
    return _tree_layout(entries)[0]


def _tree_layout(
    entries: list[SessionEntry],
) -> tuple[tuple[SessionEntry, ...], dict[str, int]]:
    # [FUNCTION DOC]
    # HELPER: Compute the session tree display order and indentation without recursion.
    # Ignores "leaf" entries. Computes each entry's longest path length to a leaf
    # bottom-up (cycles keep a finite fallback). Children are ordered longest branch
    # first (stable for ties); the longest child continues at the parent's indent (main
    # branch) while shorter siblings are emitted first, indented one level deeper. Uses
    # a depth-first stack with a seen set; entries unreachable from roots (e.g. cycles)
    # are appended afterwards as their own subtrees.
    # Output: the ordered entries and a mapping from entry id to indent level.
    tree_entries = [entry for entry in entries if entry.type != "leaf"]
    children_by_parent: dict[str | None, list[SessionEntry]] = {}
    entries_by_id: dict[str, SessionEntry] = {}
    for entry in tree_entries:
        children_by_parent.setdefault(entry.parent_id, []).append(entry)
        entries_by_id[entry.id] = entry

    # Resolve each child's longest path to a leaf without recursion. Processing
    # leaves upward also keeps deep sessions safe. Malformed cycles retain a
    # finite fallback length and are handled by the traversal's `seen` set.
    branch_lengths = dict.fromkeys(entries_by_id, 1)
    remaining_children = {
        entry_id: len(children_by_parent.get(entry_id, ())) for entry_id in entries_by_id
    }
    pending = [entry_id for entry_id, count in remaining_children.items() if count == 0]
    while pending:
        entry_id = pending.pop()
        parent_id = entries_by_id[entry_id].parent_id
        if parent_id not in remaining_children:
            continue
        branch_lengths[parent_id] = max(branch_lengths[parent_id], branch_lengths[entry_id] + 1)
        remaining_children[parent_id] -= 1
        if remaining_children[parent_id] == 0:
            pending.append(parent_id)

    def ordered_children(parent_id: str | None) -> list[SessionEntry]:
        # [FUNCTION DOC]
        # HELPER: Children of a parent id sorted by descending branch length.
        return sorted(
            children_by_parent.get(parent_id, ()),
            key=lambda child: branch_lengths.get(child.id, 1),
            reverse=True,
        )

    ordered: list[SessionEntry] = []
    indents: dict[str, int] = {}
    seen: set[str] = set()

    def child_stack_items(
        children: list[SessionEntry], parent_indent: int
    ) -> list[tuple[SessionEntry, int]]:
        # [FUNCTION DOC]
        # HELPER: Turn a sorted child list into stack items (entry, indent) so that, when
        # popped, the shorter alternate siblings come out first at indent+1 and the main
        # (longest) child last at the parent's indent.
        if not children:
            return []
        main_child, *alternate_children = children
        display_children = [*alternate_children, main_child]
        return [
            (child, parent_indent if child is main_child else parent_indent + 1)
            for child in reversed(display_children)
        ]

    def append_subtrees(children: list[SessionEntry], parent_indent: int) -> None:
        # The longest child is the unindented main branch. Emit shorter siblings
        # immediately after their parent, indented one level, before continuing
        # down the main branch. Stable length sorting preserves storage order
        # when histories have equal lengths.
        # [FUNCTION DOC]
        # HELPER: Depth-first traversal from the given children, appending each unseen entry
        # to the ordered list with its indent and pushing its own children. Mutates the
        # enclosing ordered list, indent mapping and seen set.
        stack = child_stack_items(children, parent_indent)
        while stack:
            entry, indent = stack.pop()
            if entry.id in seen:
                continue
            seen.add(entry.id)
            ordered.append(entry)
            indents[entry.id] = indent
            stack.extend(child_stack_items(ordered_children(entry.id), indent))

    append_subtrees(ordered_children(None), 0)
    for entry in tree_entries:
        if entry.id not in seen:
            append_subtrees([entry], 0)
    return tuple(ordered), indents


def _is_tool_call_tree_entry(entry: SessionEntry) -> bool:
    # [FUNCTION DOC]
    # HELPER: True for message entries holding an assistant message with tool calls.
    return (
        entry.type == "message"
        and isinstance(entry.message, AssistantMessage)
        and bool(entry.message.tool_calls)
    )


def _tree_entry_title(entry: SessionEntry) -> str:
    # [FUNCTION DOC]
    # HELPER: Display title for a tree entry: "tool call: <names>" for text-less
    # assistant tool-call messages, "<role>: <preview>" for other messages, "custom:",
    # "compaction summary:" and "branch summary:" previews, or the entry type otherwise.
    match entry.type:
        case "message":
            message = entry.message
            if (
                isinstance(message, AssistantMessage)
                and message.tool_calls
                and not message.text.strip()
            ):
                tool_names = ", ".join(call.name for call in message.tool_calls)
                return f"tool call: {tool_names}"
            return f"{message.role}: {_message_text_preview(message)}"
        case "custom_message":
            return f"custom: {_short_preview(message_text(_custom_message_from_entry(entry)))}"
        case "compaction":
            return f"compaction summary: {_short_preview(entry.summary)}"
        case "branch_summary":
            return f"branch summary: {_short_preview(entry.summary)}"
        case _:
            return entry.type


def _message_text_preview(message: AgentMessage) -> str:
    # [FUNCTION DOC]
    # HELPER: Short whitespace-collapsed preview of a message's visible text.
    return _short_preview(message_text(message))


def _short_preview(text: str, *, limit: int = 72) -> str:
    # [FUNCTION DOC]
    # HELPER: Collapse whitespace; return it if within the limit (default 72), "(empty)"
    # if blank, else truncate to limit-1 characters plus "...".
    # KNOWN ISSUE: truncated previews are limit+2 characters long, slightly exceeding
    # the limit.
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized or "(empty)"
    return f"{normalized[: limit - 1]}..."


def _messages_after_entry_on_active_path(
    entries: list[SessionEntry],
    entry_id: str,
    active_leaf_id: str | None,
) -> tuple[AgentMessage, ...]:
    # [FUNCTION DOC]
    # HELPER: Messages (including custom messages rebuilt from entries) that follow the
    # given entry on the path from root to the active leaf; empty if there is no leaf,
    # the path is invalid, or the entry is not on it.
    if active_leaf_id is None:
        return ()
    try:
        active_path = path_to_entry(entries, active_leaf_id)
    except SessionTreeError:
        return ()
    try:
        target_index = next(
            index for index, entry in enumerate(active_path) if entry.id == entry_id
        )
    except StopIteration:
        return ()
    messages: list[AgentMessage] = []
    for entry in active_path[target_index + 1 :]:
        if entry.type == "message":
            messages.append(entry.message)
        elif entry.type == "custom_message":
            messages.append(_custom_message_from_entry(entry))
    return tuple(messages)


def _custom_message_from_entry(entry: CustomMessageEntry) -> CustomMessage:
    # [FUNCTION DOC]
    # HELPER: Rebuild a custom message from a custom-message entry, converting the
    # timestamp from seconds to rounded milliseconds.
    return CustomMessage(
        custom_type=entry.custom_type,
        content=entry.content,
        display=entry.display,
        details=entry.details,
        timestamp=round(entry.timestamp * 1000),
    )


def _storage_path(storage: SessionStorage) -> Path | None:
    # [FUNCTION DOC]
    # HELPER: The storage's file path attribute if it is a Path, else None.
    path = getattr(storage, "path", None)
    return path if isinstance(path, Path) else None


def _resolve_export_destination(
    destination: Path | None,
    *,
    cwd: Path,
    session_path: Path | None,
    format: str,
) -> Path:
    # [FUNCTION DOC]
    # HELPER: Choose the export file path. No destination: default artifact path next to
    # the cwd derived from the session file, or "<cwd>/tau-session.<format>". Relative
    # destinations are resolved against the cwd; a path with a suffix is used as is;
    # otherwise it is treated as a directory and a default file name (session file stem
    # or "tau-session") is placed inside it.
    if destination is None:
        if session_path is not None:
            return default_session_export_artifact_path(
                session_path,
                destination_dir=cwd,
                format=format,
            )
        return cwd / f"tau-session.{format}"

    resolved = destination if destination.is_absolute() else cwd / destination
    if resolved.suffix:
        return resolved
    name = session_path.stem if session_path is not None else "tau-session"
    return default_session_export_artifact_path(
        Path(name),
        destination_dir=resolved,
        format=format,
    )


def _session_export_title(session: CodingSession) -> str:
    # [FUNCTION DOC]
    # HELPER: Export title: the indexed session title if any, else "Tau session <id>",
    # else "Tau Session Export".
    manager = session.session_manager
    session_id = session.session_id
    if manager is not None and session_id is not None:
        record = manager.get_session(session_id)
        if record is not None and record.title:
            return record.title
    return f"Tau session {session_id}" if session_id is not None else "Tau Session Export"


@dataclass(frozen=True, slots=True)
class _PreparedProvider:
    # [CLASS DOC]
    # HELPER: Immutable provider candidate built during load after trust/extension
    # staging: provider instance, provider name, model, Hugging Face routing and mode,
    # runtime provider config (None for dynamic), dynamic provider definition, and an
    # optionally discovered runtime model catalog.
    """A runtime candidate built after the destination environment is trusted."""

    provider: ClosableModelProvider
    provider_name: str
    model: str
    inference_provider: str | None
    inference_provider_mode: InferenceProviderMode
    runtime_provider_config: ProviderConfig | None
    dynamic_provider: DynamicProvider | None
    runtime_model_catalog: RuntimeModelCatalog | None = None


async def _prepare_provider_selection(
    config: CodingSessionConfig,
    *,
    state: SessionState,
    provider_registry: DynamicProviderRegistry,
    credential_store: FileCredentialStore | None = None,
) -> _PreparedProvider:
    # [FUNCTION DOC]
    # HELPER: Resolve and construct the startup provider.
    # Provider name comes from the CLI request, replayed state, or the session's
    # provider. If that name is an effective dynamic provider: the model is the request,
    # the state's model (same provider), the provider default, or the only model; if
    # still unknown or not offered and the provider can refresh (and the choice was
    # explicit or resumed), refresh it with network (5 s timeout) and retry defaults.
    # Raises ProviderConfigError if no model or model unavailable; builds it via the
    # dynamic factory (errors converted to ProviderConfigError).
    # Otherwise requires provider settings (ProviderConfigError), and for a Codex model
    # not in the static list (unless TAU_OFFLINE) attempts live catalog discovery with a
    # temporary provider (errors ignored, provider always closed) overlaying the
    # catalog. Then resolves the provider/model selection, routing and mode, and builds
    # the runtime provider with the startup thinking level (RuntimeError converted).
    # Output: a prepared provider. Side effects: network discovery, provider creation.
    """Resolve and construct the provider after extension/project staging."""
    requested_provider = config.requested_provider
    requested_model = config.requested_model
    provider_name = requested_provider or state.provider or config.session_provider_name
    explicit = requested_provider is not None or requested_model is not None

    if provider_name is not None:
        effective = provider_registry.effective(provider_name)
        if effective is not None and isinstance(effective.definition, DynamicProvider):
            dynamic = effective.definition
            model = requested_model
            if model is None and state.provider == provider_name:
                model = state.model
            if model is None:
                model = dynamic.default_model
            if model is None and len(dynamic.models) == 1:
                model = dynamic.models[0].id
            if (
                (model is None or not any(item.id == model for item in dynamic.models))
                and dynamic.refresh_models is not None
                and (explicit or state.provider == provider_name)
            ):
                refreshed = await provider_registry.refresh(
                    provider_name,
                    allow_network=True,
                    timeout_seconds=5.0,
                )
                if refreshed.provider is not None:
                    dynamic = refreshed.provider
                if model is None:
                    model = dynamic.default_model
                    if model is None and len(dynamic.models) == 1:
                        model = dynamic.models[0].id
            if model is None:
                raise ProviderConfigError(
                    f"Provider {provider_name} has no selectable models; refresh it explicitly."
                )
            selected = next((item for item in dynamic.models if item.id == model), None)
            if selected is None:
                raise ProviderConfigError(
                    f"Model is not available for provider {provider_name}: {model}"
                )
            try:
                runtime = await create_dynamic_model_provider(
                    dynamic,
                    model=model,
                    credential_store=provider_registry.credentials,
                    environment=provider_registry.environment,
                )
            except (ProviderConfigError, RuntimeError) as exc:
                raise ProviderConfigError(str(exc)) from exc
            return _PreparedProvider(
                provider=runtime,
                provider_name=provider_name,
                model=model,
                inference_provider=None,
                inference_provider_mode="automatic",
                runtime_provider_config=None,
                dynamic_provider=dynamic,
            )

    settings = config.provider_settings
    discovered_catalog: RuntimeModelCatalog | None = None
    if settings is None:
        raise ProviderConfigError(
            f"Provider is not available after trusted extension loading: "
            f"{provider_name or config.model}"
        )
    selected_model = requested_model or (state.model if state.provider == provider_name else None)
    selected_provider = settings.get_provider(provider_name or settings.default_provider)
    if (
        isinstance(selected_provider, OpenAICodexProviderConfig)
        and selected_model is not None
        and selected_model not in selected_provider.models
        and environ.get("TAU_OFFLINE") is None
    ):
        # Live-only explicit/resumed models need discovery before static validation.
        discovery_provider = None
        try:
            discovery_provider = create_model_provider(
                selected_provider,
                credential_store=credential_store,
                model=None,
                thinking_level=None,
            )
            if isinstance(discovery_provider, ModelCatalogProvider):
                catalog = await discovery_provider.discover_models()
                if catalog.models:
                    discovered_catalog = catalog
                    live_provider = _provider_with_runtime_model_catalog(selected_provider, catalog)
                    settings = replace(
                        settings,
                        providers=tuple(
                            live_provider if item.name == selected_provider.name else item
                            for item in settings.providers
                        ),
                    )
        except Exception:  # noqa: BLE001 - retain ordinary static validation on failure
            pass
        finally:
            if discovery_provider is not None:
                await discovery_provider.aclose()
    selection = resolve_provider_selection(
        settings,
        provider_name=provider_name,
        model=selected_model,
    )
    inference_provider = _session_inference_provider(
        config,
        state,
        selection.provider.name,
        selection.model,
    )
    inference_provider_mode = _session_inference_provider_mode(
        config,
        state,
        selection.provider,
        selection.model,
        inference_provider,
    )
    try:
        runtime = _create_runtime_provider(
            selection.provider,
            credential_store=credential_store,
            model=selection.model,
            inference_provider=inference_provider,
            thinking_level=resolve_startup_thinking_level(
                selection.provider,
                selection.model,
            ),
        )
    except RuntimeError as exc:
        raise ProviderConfigError(str(exc)) from exc
    return _PreparedProvider(
        provider=runtime,
        provider_name=selection.provider.name,
        model=selection.model,
        inference_provider=inference_provider,
        inference_provider_mode=inference_provider_mode,
        runtime_provider_config=selection.provider,
        dynamic_provider=None,
        runtime_model_catalog=discovered_catalog,
    )


def _session_inference_provider(
    config: CodingSessionConfig,
    state: SessionState,
    provider_name: str,
    model: str,
) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Hugging Face routing to use: None for other providers; the configured
    # route when resuming the same provider/model; otherwise the provider config's
    # per-model inference provider (or None).
    """Preserve HF routing only for the same logical provider/model."""
    if provider_name != "huggingface":
        return None
    if state.provider == provider_name and state.model == model:
        return config.inference_provider
    provider = (
        config.provider_settings.get_provider(provider_name) if config.provider_settings else None
    )
    if isinstance(provider, OpenAICompatibleProviderConfig):
        return provider.inference_providers.get(model)
    return None


def _session_inference_provider_mode(
    config: CodingSessionConfig,
    state: SessionState,
    provider: ProviderConfig,
    model: str,
    inference_provider: str | None,
) -> InferenceProviderMode:
    # [FUNCTION DOC]
    # HELPER: Routing mode: "automatic" for non-Hugging Face; for the same resumed
    # provider/model the configured mode (or "fixed" if a route exists); else the mode
    # configured for the model.
    """Preserve automatic/fixed HF routing for the same resumed model."""
    if provider.name != "huggingface":
        return "automatic"
    if state.provider == provider.name and state.model == model:
        return config.inference_provider_mode or (
            "fixed" if inference_provider is not None else "automatic"
        )
    return _configured_inference_provider_mode(provider, model)


def _filter_tools(
    tools: Sequence[AgentTool], allowed_tool_names: frozenset[str] | None
) -> list[AgentTool]:
    # [FUNCTION DOC]
    # HELPER: Apply the allow-list to a composed tool list (by name); None keeps all.
    # Returns a new list.
    """Apply ``CodingSessionConfig.allowed_tool_names`` to a composed tool list."""
    if allowed_tool_names is None:
        return list(tools)
    return [tool for tool in tools if tool.name in allowed_tool_names]


def _configured_model_supports_images(config: CodingSessionConfig, model: str) -> bool | None:
    # [FUNCTION DOC]
    # HELPER: Whether the configured model accepts images: from the dynamic model's
    # modalities (None if unknown), else from the runtime or settings provider config,
    # else None.
    if config.dynamic_provider is not None:
        selected = next((item for item in config.dynamic_provider.models if item.id == model), None)
        if selected is None or selected.input_modalities is None:
            return None
        return "image" in selected.input_modalities
    provider = config.runtime_provider_config
    if provider is None and config.provider_settings is not None:
        try:
            provider = config.provider_settings.get_provider(config.provider_name)
        except ProviderConfigError:
            return None
    return provider_model_supports_images(provider, model) if provider is not None else None


def _initial_model_for_config(config: CodingSessionConfig) -> str:
    # [FUNCTION DOC]
    # HELPER: Model for a brand-new session: the configured model, or the provider's
    # default if the configured model is not valid for it (only when both settings and
    # a runtime config exist).
    if config.provider_settings is None or config.runtime_provider_config is None:
        return config.model
    provider = _provider_config_for_name(config, config.provider_name)
    if provider is None:
        return config.model
    try:
        validate_provider_model(provider, config.model)
    except ProviderConfigError:
        return provider.default_model
    return config.model


def _runtime_model_for_state(config: CodingSessionConfig, state: SessionState) -> str:
    # [FUNCTION DOC]
    # HELPER: Model to run after replay: the config model for dynamic providers; else the
    # replayed state model (or config model), replaced by the config model or provider
    # default when not valid for the provider.
    state_model = state.model or config.model
    if config.dynamic_provider is not None:
        return config.model
    if config.provider_settings is None or config.runtime_provider_config is None:
        return state_model
    provider = _provider_config_for_name(config, config.provider_name)
    if provider is None:
        return state_model
    try:
        validate_provider_model(provider, state_model)
    except ProviderConfigError:
        return config.model if config.model in provider.models else provider.default_model
    return state_model


def _initial_thinking_level_for_config(
    config: CodingSessionConfig,
    *,
    model: str,
) -> ThinkingLevel:
    # [FUNCTION DOC]
    # HELPER: Thinking level for a brand-new session: a startup override (resolved
    # against the provider when known), else the provider's preferred level for the
    # model, else the configured level.
    provider = _provider_config_for_name(config, config.provider_name)
    if config.thinking_level_override is not None:
        if provider is None:
            return config.thinking_level_override
        resolved = resolve_startup_thinking_level(
            provider,
            model,
            cli_override=config.thinking_level_override,
        )
        if resolved is not None:
            return resolved
    if provider is None:
        return config.thinking_level
    return _preferred_thinking_level_for_model(
        provider,
        model=model,
        fallback=config.thinking_level,
    )


def _provider_config_for_name(
    config: CodingSessionConfig,
    provider_name: str,
) -> ProviderConfig | None:
    # [FUNCTION DOC]
    # HELPER: Provider config for a name: a matching Codex runtime config first (keeps
    # live models), then settings, then the runtime config regardless of name, else None.
    # KNOWN ISSUE: the final fallback returns the runtime config even when its name
    # does not match; callers like pricing must re-check the name.
    if (
        isinstance(config.runtime_provider_config, OpenAICodexProviderConfig)
        and config.runtime_provider_config.name == provider_name
    ):
        return config.runtime_provider_config
    if config.provider_settings is not None:
        try:
            return config.provider_settings.get_provider(provider_name)
        except ProviderConfigError:
            pass
    if config.runtime_provider_config is not None:
        return config.runtime_provider_config
    return None


def _state_thinking_level(
    state: SessionState,
    default: ThinkingLevel,
) -> ThinkingLevel:
    # [FUNCTION DOC]
    # HELPER: The replayed state's thinking level, normalized, or the given default when
    # the state has none (tolerates state objects without the attribute).
    thinking_level = getattr(state, "thinking_level", None)
    if thinking_level is None:
        return default
    return normalize_thinking_level(thinking_level)


def _create_runtime_provider(
    provider: ProviderConfig,
    *,
    credential_store: FileCredentialStore | None,
    model: str,
    thinking_level: ThinkingLevel | None,
    inference_provider: str | None,
    response_headers_observer: Callable[[Mapping[str, str]], None] | None = None,
) -> ClosableModelProvider:
    # [FUNCTION DOC]
    # HELPER: Build a runtime provider from a provider config with the credential store,
    # model and thinking level, passing the Hugging Face inference route and the
    # response-header observer only when they are set. Propagates provider factory
    # errors (ProviderConfigError or RuntimeError).
    if inference_provider is None and response_headers_observer is None:
        return create_model_provider(
            provider,
            credential_store=credential_store,
            model=model,
            thinking_level=thinking_level,
        )
    if inference_provider is None:
        return create_model_provider(
            provider,
            credential_store=credential_store,
            model=model,
            thinking_level=thinking_level,
            response_headers_observer=response_headers_observer,
        )
    return create_model_provider(
        provider,
        credential_store=credential_store,
        model=model,
        thinking_level=thinking_level,
        inference_provider=inference_provider,
        response_headers_observer=response_headers_observer,
    )


def _configured_inference_provider(
    provider: ProviderConfig | None,
    model: str,
) -> str | None:
    # [FUNCTION DOC]
    # HELPER: The configured Hugging Face route for a model (only for the
    # OpenAI-compatible "huggingface" provider), else None.
    if not isinstance(provider, OpenAICompatibleProviderConfig) or provider.name != "huggingface":
        return None
    return provider.inference_providers.get(model)


def _configured_inference_provider_mode(
    provider: ProviderConfig | None,
    model: str,
) -> InferenceProviderMode:
    # [FUNCTION DOC]
    # HELPER: "fixed" when a route is configured for the model, else "automatic".
    return "fixed" if _configured_inference_provider(provider, model) is not None else "automatic"


def _default_thinking_level_for_active_model(session: CodingSession) -> ThinkingLevel:
    # [FUNCTION DOC]
    # HELPER: Default thinking level for the session's active model: the provider's
    # preferred level, falling back to the configured level (also used when no provider
    # config is available).
    provider = session._active_provider_config()
    if provider is None:
        return session._config.thinking_level
    return _preferred_thinking_level_for_model(
        provider,
        model=session.model,
        fallback=session._config.thinking_level,
    )


def _preferred_thinking_level_for_model(
    provider: ProviderConfig,
    *,
    model: str,
    fallback: ThinkingLevel,
) -> ThinkingLevel:
    # [FUNCTION DOC]
    # HELPER: Choose a thinking level for a model: the remembered per-model default if
    # supported; else the fallback if supported or if the model lists no levels; else the
    # provider's default level for the model, else the first supported level.
    levels = provider_thinking_levels(provider, model=model)
    preferred = provider.thinking_defaults.get(model)
    if preferred in levels:
        return preferred
    if fallback in levels or not levels:
        return fallback
    default = provider_default_thinking_level(provider, model=model)
    return default or levels[0]


def _coerced_thinking_level(
    provider: ProviderConfig,
    *,
    model: str,
    current: ThinkingLevel,
    preferred: ThinkingLevel | None = None,
) -> ThinkingLevel:
    # [FUNCTION DOC]
    # HELPER: Keep the current level if supported (or if no levels exist); else use the
    # preferred level if supported; else the provider default for the model, else the
    # first supported level.
    levels = provider_thinking_levels(provider, model=model)
    if not levels or current in levels:
        return current
    if preferred in levels:
        return preferred
    default = provider_default_thinking_level(provider, model=model)
    return default or levels[0]


def _unavailable_thinking_message(session: CodingSession) -> str:
    # [FUNCTION DOC]
    # HELPER: Error text "Thinking controls are unavailable for <provider>:<model>",
    # with the specific reason appended when known.
    message = f"Thinking controls are unavailable for {session.provider_name}:{session.model}"
    reason = session.thinking_unavailable_reason
    if reason:
        return f"{message}: {reason}"
    return message


def _sanitize_session_name(text: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Normalize model or user text into a session name: collapse whitespace,
    # strip surrounding quotes and punctuation, strip punctuation/quotes from each word,
    # drop empty words, and keep at most four words. Returns None if nothing remains.
    cleaned = " ".join(text.split()).strip()
    cleaned = cleaned.strip("\"'`“”‘’")
    cleaned = cleaned.strip(string.punctuation + " ")
    words = [word.strip(string.punctuation + "\"'`“”‘’") for word in cleaned.split()]
    words = [word for word in words if word]
    if not words:
        return None
    return " ".join(words[:4])


def _fallback_session_name(first_message: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Heuristic session name from the first user message (sanitized, first four
    # words) or None.
    return _sanitize_session_name(first_message)


def _terminal_command_context_message(command: str, output: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Markdown text telling the model the user ran a terminal command, with the
    # command in a bash code block and the output in a text code block.
    return (
        "Terminal command executed by the user.\n\n"
        f"Command:\n```bash\n{command}\n```\n\n"
        f"Output:\n```text\n{output}\n```"
    )


def parse_terminal_command(text: str) -> TerminalCommandRequest | None:
    # [FUNCTION DOC]
    # Parse input-bar shell syntax: "!!cmd" runs without adding output to context,
    # "!cmd" runs and adds it; leading/trailing whitespace is ignored. Returns a request,
    # or None for other text or an empty command.
    """Parse input-bar terminal command syntax."""
    stripped = text.strip()
    if stripped.startswith("!!"):
        command = stripped[2:].strip()
        if not command:
            return None
        return TerminalCommandRequest(command=command, add_to_context=False)
    if stripped.startswith("!"):
        command = stripped[1:].strip()
        if not command:
            return None
        return TerminalCommandRequest(command=command, add_to_context=True)
    return None


def _category_summary(
    before: tuple[tuple[object, ...], ...],
    after: tuple[tuple[object, ...], ...],
) -> ReloadCategorySummary:
    # [FUNCTION DOC]
    # HELPER: Build a reload category summary with before/after counts and whether the
    # signature tuples differ.
    return ReloadCategorySummary(
        before=len(before),
        after=len(after),
        changed=before != after,
    )


def _skill_signatures(skills: tuple[Skill, ...]) -> tuple[tuple[object, ...], ...]:
    # [FUNCTION DOC]
    # HELPER: Comparable tuples (name, path, description, content, model-invocation
    # flag) for each skill, used to detect reload changes.
    return tuple(
        (
            skill.name,
            str(skill.path),
            skill.description,
            skill.content,
            skill.disable_model_invocation,
        )
        for skill in skills
    )


def _prompt_template_signatures(
    prompt_templates: tuple[PromptTemplate, ...],
) -> tuple[tuple[object, ...], ...]:
    # [FUNCTION DOC]
    # HELPER: Comparable tuples (name, path, description, content) for each template.
    return tuple(
        (template.name, str(template.path), template.description, template.content)
        for template in prompt_templates
    )


def _context_file_signatures(
    context_files: tuple[ProjectContextFile, ...],
) -> tuple[tuple[object, ...], ...]:
    # [FUNCTION DOC]
    # HELPER: Comparable (path, content) pairs for context files.
    return tuple((context_file.path, context_file.content) for context_file in context_files)


def _diagnostic_signatures(
    diagnostics: tuple[ResourceDiagnostic, ...],
) -> tuple[tuple[object, ...], ...]:
    # [FUNCTION DOC]
    # HELPER: Comparable tuples (kind, message, path, name, severity) for diagnostics.
    return tuple(
        (
            diagnostic.kind,
            diagnostic.message,
            str(diagnostic.path) if diagnostic.path is not None else None,
            diagnostic.name,
            diagnostic.severity,
        )
        for diagnostic in diagnostics
    )


def _extension_signatures(runtime: ExtensionRuntime) -> tuple[tuple[object, ...], ...]:
    # [FUNCTION DOC]
    # HELPER: One-element tuples of loaded extension names (order-sensitive).
    return tuple((name,) for name in runtime.extension_names)


def _system_prompt_resource_signatures(
    *,
    skills: tuple[Skill, ...],
    context_files: tuple[ProjectContextFile, ...],
    custom_system_prompt: str | None,
    custom_system_prompt_path: Path | None,
    append_system_prompt: str | None,
    append_system_prompt_paths: tuple[Path, ...],
) -> tuple[object, ...]:
    # [FUNCTION DOC]
    # HELPER: Comparable snapshot of every resource input that affects the system prompt:
    # skills sorted by name (name, path, description, invocation flag; content
    # excluded), context file signatures, custom prompt text and path, combined append
    # prompt text, and append prompt paths.
    prompt_skills = tuple(
        (skill.name, str(skill.path), skill.description, skill.disable_model_invocation)
        for skill in sorted(skills, key=lambda item: item.name)
    )
    return (
        prompt_skills,
        _context_file_signatures(context_files),
        custom_system_prompt,
        str(custom_system_prompt_path) if custom_system_prompt_path is not None else None,
        append_system_prompt,
        tuple(str(path) for path in append_system_prompt_paths),
    )


def _provider_config_or_none(
    settings: ProviderSettings | None,
    provider_name: str | None,
) -> ProviderConfig | None:
    # [FUNCTION DOC]
    # HELPER: Look up a provider config, returning None when settings or name are
    # missing or the provider is unknown.
    if settings is None or provider_name is None:
        return None
    try:
        return settings.get_provider(provider_name)
    except ProviderConfigError:
        return None


def _codex_account_id(
    provider: OpenAICodexProviderConfig,
    credential_store: FileCredentialStore,
) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Codex account id from the stored OAuth credential (if a credential name is
    # configured), else decoded from an access token in the provider's API-key
    # environment variable, else None.
    if provider.credential_name:
        credential = credential_store.get_oauth(provider.credential_name)
        if credential is not None and credential.account_id:
            return credential.account_id
    return account_id_from_access_token(environ.get(provider.api_key_env, ""))


def _provider_settings_with_runtime_catalogs(
    settings: ProviderSettings | None,
    catalogs: Mapping[str, RuntimeModelCatalog],
) -> ProviderSettings | None:
    # [FUNCTION DOC]
    # HELPER: Return settings with each provider overlaid by its runtime catalog (or the
    # input unchanged when there are no settings or catalogs).
    if settings is None or not catalogs:
        return settings
    providers = tuple(
        _provider_with_runtime_model_catalog(provider, catalogs.get(provider.name))
        for provider in settings.providers
    )
    return replace(settings, providers=providers)


def _provider_with_runtime_model_catalog(
    provider: ProviderConfig,
    catalog: RuntimeModelCatalog | None,
) -> ProviderConfig:
    # [FUNCTION DOC]
    # HELPER: Overlay a live account-specific catalog onto a Codex provider config (other
    # providers unchanged): models from the catalog, default model kept if present else
    # the first catalog model, context windows from live limits or existing config,
    # per-model metadata merged, thinking models, and thinking defaults. Does not touch
    # durable settings.
    # KNOWN ISSUE: an empty catalog (e.g. a stale on-disk cache) would make choosing the
    # first model fail with an index error.
    """Overlay one account-specific catalog without changing durable settings."""
    if catalog is None or not isinstance(provider, OpenAICodexProviderConfig):
        return provider

    models = tuple(model.id for model in catalog.models)
    metadata = {
        model.id: _runtime_provider_model_metadata(provider, model) for model in catalog.models
    }
    context_windows = {
        model.id: (
            model.limits.context_window
            if model.limits is not None
            else provider.context_windows[model.id]
        )
        for model in catalog.models
        if model.limits is not None or model.id in provider.context_windows
    }
    return replace(
        provider,
        models=models,
        default_model=(provider.default_model if provider.default_model in models else models[0]),
        context_windows=context_windows,
        model_metadata=metadata,
        thinking_models=tuple(model.id for model in catalog.models if model.thinking_levels),
        thinking_defaults=_runtime_model_thinking_defaults(provider, catalog),
    )


def _runtime_model_thinking_defaults(
    provider: OpenAICodexProviderConfig,
    catalog: RuntimeModelCatalog,
) -> dict[str, ThinkingLevel]:
    # [FUNCTION DOC]
    # HELPER: Mapping from model id to thinking default: the user's remembered default,
    # else the catalog's default level, omitted when neither exists.
    defaults: dict[str, ThinkingLevel] = {}
    for model in catalog.models:
        default = provider.thinking_defaults.get(model.id) or model.default_thinking_level
        if default is not None:
            defaults[model.id] = default
    return defaults


def _runtime_provider_model_metadata(
    provider: OpenAICodexProviderConfig,
    model: RuntimeModel,
) -> ProviderModelMetadata:
    # [FUNCTION DOC]
    # HELPER: Merge a live catalog model into existing per-model metadata: display name,
    # reasoning flag (true if any thinking levels), input modalities, context window and
    # max output tokens from live limits when available, and a thinking-level map where
    # supported levels map to themselves ("off" maps to "none") and others to None.
    existing = provider.model_metadata.get(model.id, ProviderModelMetadata())
    supported = set(model.thinking_levels)
    thinking_level_map = {
        level: ("none" if level == "off" else level) if level in supported else None
        for level in THINKING_LEVELS
    }
    return replace(
        existing,
        name=model.name or existing.name,
        reasoning=True if model.thinking_levels else existing.reasoning,
        input=model.input_modalities,
        context_window=(
            model.limits.context_window if model.limits is not None else existing.context_window
        ),
        max_tokens=(
            model.limits.max_output_tokens
            if model.limits is not None and model.limits.max_output_tokens is not None
            else existing.max_tokens
        ),
        thinking_level_map=thinking_level_map,
    )


def _load_session_resources(
    resource_paths: TauResourcePaths,
    explicit_context_files: tuple[ProjectContextFile, ...],
    *,
    skills_enabled: bool = True,
    system_prompt_enabled: bool = True,
    custom_system_prompt_explicit: bool = False,
) -> SessionResources:
    # [FUNCTION DOC]
    # HELPER: Load all resources for the given (trust-filtered) paths: skills (unless
    # disabled), prompt templates, project context files (merged after explicit ones),
    # and custom/append system prompt files (discovery disabled when an explicit system
    # prompt is used; an explicit custom prompt suppresses the file). Collects all
    # diagnostics. Returns a SessionResources bundle. Side effects: filesystem reads.
    loaded_skills: list[Skill]
    skill_diagnostics: list[ResourceDiagnostic]
    if skills_enabled:
        loaded_skills, skill_diagnostics = load_skills_with_diagnostics(resource_paths)
    else:
        loaded_skills, skill_diagnostics = [], []
    loaded_prompt_templates, prompt_diagnostics = load_prompt_templates_with_diagnostics(
        resource_paths
    )
    discovered_context, context_diagnostics = discover_project_context_with_diagnostics(
        resource_paths
    )
    system_prompts = discover_system_prompt_resources(
        resource_paths,
        custom_prompt_explicit=custom_system_prompt_explicit,
        enabled=system_prompt_enabled,
    )
    return SessionResources(
        skills=tuple(loaded_skills),
        prompt_templates=tuple(loaded_prompt_templates),
        context_files=_merge_context_files(explicit_context_files, discovered_context),
        custom_system_prompt=system_prompts.custom_prompt,
        custom_system_prompt_path=system_prompts.custom_prompt_path,
        append_system_prompt=system_prompts.append_prompt,
        append_system_prompts=system_prompts.append_prompts,
        append_system_prompt_paths=system_prompts.append_prompt_paths,
        diagnostics=tuple(
            [
                *skill_diagnostics,
                *prompt_diagnostics,
                *context_diagnostics,
                *system_prompts.diagnostics,
            ]
        ),
    )


def _append_prompt_sections(
    prompts: tuple[str, ...],
    paths: tuple[Path, ...],
    explicit_prompt: str | None,
) -> tuple[PromptSection, ...]:
    # [FUNCTION DOC]
    # HELPER: Build prompt sections pairing each append prompt with its source file path
    # (strict pairing), followed by the CLI-provided append prompt if any.
    """Pair append content with its file or CLI origin in composition order."""
    sections = [
        PromptSection(title=None, body=prompt, source=str(path))
        for prompt, path in zip(prompts, paths, strict=True)
    ]
    if explicit_prompt is not None:
        sections.append(
            PromptSection(
                title=None,
                body=explicit_prompt,
                source="CLI --append-system-prompt",
            )
        )
    return tuple(sections)


def _custom_prompt_source(*, explicit: bool, path: Path | None) -> str:
    # [FUNCTION DOC]
    # HELPER: Source label for the custom prompt: "CLI --system-prompt" when explicit,
    # else the file path, else "runtime configuration".
    if explicit:
        return "CLI --system-prompt"
    return str(path) if path is not None else "runtime configuration"


def _merge_context_files(
    explicit: tuple[ProjectContextFile, ...],
    discovered: tuple[ProjectContextFile, ...],
) -> tuple[ProjectContextFile, ...]:
    # [FUNCTION DOC]
    # HELPER: Concatenate explicit then discovered context files, dropping later
    # duplicates by path.
    merged: list[ProjectContextFile] = []
    seen: set[str] = set()
    for context_file in (*explicit, *discovered):
        if context_file.path in seen:
            continue
        seen.add(context_file.path)
        merged.append(context_file)
    return tuple(merged)


def _tool_history_repair_plan(
    messages: tuple[AgentMessage, ...],
    *,
    context_entry_ids: tuple[str, ...],
    entries: tuple[SessionEntry, ...],
) -> tuple[str | None, tuple[AgentMessage, ...], ToolHistoryRepair] | None:
    # [FUNCTION DOC]
    # HELPER: Run tool-history repair on the active messages. If nothing changed, return
    # None. Otherwise find the longest unchanged message prefix and return the parent id
    # to branch from (the last unchanged entry, or the first context entry's parent when
    # no prefix is unchanged), the repaired message suffix, and the repair record.
    repair = repair_tool_history(messages)
    if not repair.changed:
        return None

    common_prefix_length = 0
    for old_message, repaired_message in zip(messages, repair.messages, strict=False):
        if old_message != repaired_message:
            break
        common_prefix_length += 1

    if common_prefix_length > 0:
        parent_id: str | None = context_entry_ids[common_prefix_length - 1]
    elif context_entry_ids:
        entries_by_id = {entry.id: entry for entry in entries}
        first_entry = entries_by_id.get(context_entry_ids[0])
        parent_id = first_entry.parent_id if first_entry is not None else None
    else:
        parent_id = None

    return parent_id, repair.messages[common_prefix_length:], repair


def default_session_path(cwd: Path) -> Path:
    # [FUNCTION DOC]
    # Return Tau's default per-project session file path under the user's Tau home for
    # the given cwd.
    """Return Tau's default user-home session path for a project cwd."""
    return TauPaths().default_session_path(cwd)


def jsonl_session_storage(path: str | Path) -> JsonlSessionStorage:
    # [FUNCTION DOC]
    # Create JSONL append-only session storage for a file path.
    """Convenience factory for local JSONL coding-session storage."""
    return JsonlSessionStorage(path)


def _append_session_entry_sync(storage: SessionStorage, entry: SessionEntry) -> None:
    # [FUNCTION DOC]
    # HELPER: Synchronously append one entry as a JSON line to JSONL storage (creating
    # parent directories). Raises RuntimeError for other storage types.
    # Side effect: file write.
    """Append an entry synchronously for slash commands that cannot await storage."""
    if isinstance(storage, JsonlSessionStorage):
        storage.path.parent.mkdir(parents=True, exist_ok=True)
        with storage.path.open("a", encoding="utf-8") as file:
            file.write(entry_to_json_line(entry))
        return
    raise RuntimeError("Session storage does not support synchronous initialization")
