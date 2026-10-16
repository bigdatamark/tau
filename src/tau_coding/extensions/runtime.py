# [MODULE DOC]
# Extension runtime for the coding app (Pi's extension runner equivalent).
# Owns one generation of loaded extensions (trusted built-ins first, then
# filesystem/explicit extensions), records what they register (tools, slash
# commands, message renderers, prompt guidelines/sections, event handlers,
# dynamic providers and local backends), binds to a coding session, fans out
# harness and lifecycle events to handlers, wraps every tool with tool_call /
# tool_result hook seams, runs input hooks and project-trust handlers, and
# collects diagnostics. Extensions are an isolation boundary: exceptions and
# bad return values from extension code are recorded as diagnostics rather
# than propagated. Reload/retire invalidate the generation so stale API
# objects raise ExtensionError.
"""Extension runtime: hook dispatch, tool wrapping, and session binding."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from inspect import isawaitable
from pathlib import Path
from time import time_ns
from typing import Literal, Protocol, cast

import httpx

import tau_coding.built_in_extensions as built_in_extension_registry
from tau_agent.events import AgentEvent, AgentStartEvent
from tau_agent.events import TurnEndEvent as AgentTurnEndEvent
from tau_agent.events import TurnStartEvent as AgentTurnStartEvent
from tau_agent.messages import AgentMessage, TextContent
from tau_agent.tools import (
    AgentTool,
    AgentToolResult,
    ToolCancellationToken,
    ToolUpdateCallback,
)
from tau_agent.types import JSONValue
from tau_coding.built_in_extensions import BuiltInExtension, BuiltInExtensionContext
from tau_coding.commands import (
    CommandContext,
    CommandRegistry,
    CommandResult,
    SlashCommand,
    create_default_command_registry,
)
from tau_coding.credentials import CredentialStore, FileCredentialStore, credentials_path
from tau_coding.extensions.api import (
    AGENT_EVENT_TYPES,
    AGENT_EVENT_WILDCARD,
    LIFECYCLE_EVENT_TYPES,
    CustomMessageView,
    ExtensionAPI,
    ExtensionCommandContext,
    ExtensionCommandHandler,
    ExtensionContext,
    ExtensionError,
    ExtensionGeneration,
    ExtensionHandler,
    InputEvent,
    InputHookResult,
    MessageRenderer,
    MessageRenderOptions,
    NullUiBridge,
    RegisteredExtension,
    SessionLifecycleReason,
    SessionShutdownEvent,
    SessionStartEvent,
    ToolCallHookEvent,
    ToolCallHookResult,
    ToolResultHookEvent,
    ToolResultHookResult,
    TurnEndEvent,
    TurnStartEvent,
    UiBridge,
)
from tau_coding.extensions.loader import (
    ExtensionSourceMetadata,
    LoadedExtension,
    load_extensions,
    unload_extension_modules,
)
from tau_coding.extensions.provider_registry import (
    DynamicProviderRegistry,
    ProviderRegistryCloseResult,
)
from tau_coding.extensions.providers import CredentialReader, DynamicProvider
from tau_coding.local_backends import LocalBackend, LocalBackendRegistry
from tau_coding.paths import TauPaths
from tau_coding.project_trust import (
    ExtensionTrustResult,
    ProjectTrustEvent,
    ProjectTrustResolution,
)
from tau_coding.provider_config import ProviderConfig
from tau_coding.resources import ResourceDiagnostic, TauResourcePaths
from tau_coding.system_prompt import PromptSection

# Host callback that delivers a message through the frontend's serialized run
# path when the session is idle. Carries the same presentation metadata as a
# queued message so custom messages render correctly whether they trigger a new
# turn or are injected into a running one.
TurnRequestedCallback = Callable[[str, "str | None", "dict[str, JSONValue] | None"], None]


class BoundModelChoice(Protocol):
    # [CLASS DOC]
    # Structural protocol for a provider/model pair exposed by a bound session.
    """Provider/model identifiers exposed by a bound session."""

    @property
    def provider_name(self) -> str: ...
                                    # [FUNCTION DOC]
                                    # Protocol property: the provider name.

    @property
    def model(self) -> str: ...
                            # [FUNCTION DOC]
                            # Protocol property: the model id.


class BoundSession(Protocol):
    # [CLASS DOC]
    # Structural protocol describing the slice of the coding session the runtime
    # and extension API use: read-only state (cwd, model, provider, available
    # providers and model choices, inference provider route and mode, session id
    # and name, thinking level, system prompt, project trust resolution, running
    # flag, messages) and actions (queue steering/follow-up messages, append a
    # custom entry, set a label, set the inference provider route).
    """The slice of `CodingSession` the extension runtime binds to."""

    @property
    def cwd(self) -> Path: ...
                           # [FUNCTION DOC]
                           # Protocol property: session working directory.

    @property
    def model(self) -> str: ...
                            # [FUNCTION DOC]
                            # Protocol property: active model id.

    @property
    def provider_name(self) -> str: ...
                                    # [FUNCTION DOC]
                                    # Protocol property: active provider name.

    @property
    def available_providers(self) -> tuple[str, ...]: ...
                                                      # [FUNCTION DOC]
                                                      # Protocol property: names of configured providers.

    @property
    def available_model_choices(self) -> tuple[BoundModelChoice, ...]: ...
                                                                       # [FUNCTION DOC]
                                                                       # Protocol property: selectable provider/model choices.

    @property
    def inference_provider(self) -> str | None: ...
                                                # [FUNCTION DOC]
                                                # Protocol property: current inference provider route, or none.

    @property
    def inference_provider_mode(self) -> str: ...
                                              # [FUNCTION DOC]
                                              # Protocol property: textual mode of inference provider routing.

    @property
    def session_id(self) -> str | None: ...
                                        # [FUNCTION DOC]
                                        # Protocol property: persisted session id, or none.

    @property
    def session_name(self) -> str | None: ...
                                          # [FUNCTION DOC]
                                          # Protocol property: session display name, or none.

    @property
    def thinking_level(self) -> str: ...
                                     # [FUNCTION DOC]
                                     # Protocol property: current thinking level name.

    @property
    def system_prompt(self) -> str: ...
                                    # [FUNCTION DOC]
                                    # Protocol property: current assembled system prompt.

    @property
    def project_trust_resolution(self) -> ProjectTrustResolution | None: ...
                                                                         # [FUNCTION DOC]
                                                                         # Protocol property: project trust resolution, or none.

    @property
    def is_running(self) -> bool: ...
                                  # [FUNCTION DOC]
                                  # Protocol property: whether an agent run is in progress.

    @property
    def messages(self) -> tuple[AgentMessage, ...]: ...
                                                    # [FUNCTION DOC]
                                                    # Protocol property: transcript messages in order.

    def queue_steering_message(
        self,
        content: str,
        *,
        custom_type: str | None = None,
        details: dict[str, JSONValue] | None = None,
    ) -> None: ...
               # [FUNCTION DOC]
               # Protocol method: queue a message to steer the running turn, with optional
               # custom type and details metadata.

    def queue_follow_up_message(
        self,
        content: str,
        *,
        custom_type: str | None = None,
        details: dict[str, JSONValue] | None = None,
    ) -> None: ...
               # [FUNCTION DOC]
               # Protocol method: queue a message delivered after the current run, with
               # optional custom type and details metadata.

    async def append_custom_entry(self, namespace: str, data: dict[str, JSONValue]) -> None: ...
                                                                                             # [FUNCTION DOC]
                                                                                             # Protocol method: asynchronously persist a custom entry under a namespace.

    async def set_label(self, target_id: str, label: str | None) -> object: ...
                                                                            # [FUNCTION DOC]
                                                                            # Protocol method: asynchronously set or clear a bookmark label on an entry.

    def set_inference_provider(self, route: str | None) -> str: ...
                                                                # [FUNCTION DOC]
                                                                # Protocol method: set (or clear with none) the inference provider route and
                                                                # return the effective route.


@dataclass(frozen=True, slots=True)
class ExtensionCommand:
    # [CLASS DOC]
    # Immutable record of an extension slash command: owning extension name,
    # source id, normalized name (lowercase, no slash), description, usage text,
    # aliases and handler.
    """A slash command registered by an extension."""

    extension: str
    source_id: str
    name: str
    description: str
    usage: str
    aliases: tuple[str, ...]
    handler: ExtensionCommandHandler


@dataclass(frozen=True, slots=True)
class RegisteredExtensionTool:
    # [CLASS DOC]
    # Immutable record pairing an extension tool with its extension name and
    # source id.
    """A tool registered by an extension."""

    extension: str
    source_id: str
    tool: AgentTool


@dataclass(frozen=True, slots=True)
class InputHookOutcome:
    # [CLASS DOC]
    # Immutable combined result of input hooks: whether the input was handled
    # (consumed), the possibly transformed text and an optional message to show.
    """Combined outcome of running all `input` hooks over prompt text."""

    handled: bool
    text: str
    message: str | None = None


class ExtensionRuntime:
    # [CLASS DOC]
    # Owns loaded extensions for one prepared session snapshot and dispatches
    # between them and the session. State: the generation token, built-in
    # declarations and context, provider and local-backend registries (plus
    # retired ones awaiting drain), registered extensions, tools by name, commands
    # by name, prompt guidelines and sections tagged by source, renderers by
    # custom type, diagnostics (load-time and runtime), resolved paths, bound
    # session, UI bridge, turn-request callback, harness unsubscribe function and
    # an extension-facing turn index.
    """Owns loaded extensions and dispatches events between them and a session.

    Each runtime belongs to one prepared session snapshot. Reload and
    destination replacement stage a fresh runtime, then retire the prior
    generation only after preparation succeeds. This prevents project extension
    registrations from crossing cwd trust boundaries.
    """

    def __init__(
        self,
        *,
        ui: UiBridge | None = None,
        durable_providers: Sequence[ProviderConfig] = (),
        credentials: CredentialReader | None = None,
        environment: Mapping[str, str] | None = None,
        built_in_extensions: Sequence[BuiltInExtension] | None = None,
        paths: TauPaths | None = None,
        built_in_credentials: CredentialStore | None = None,
        built_in_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Creates a fresh generation and its registries.
        # Inputs (keyword): optional UI bridge (default null bridge), durable provider
        # configs, read-only credential reader, environment mapping, built-in
        # extension declarations (default the global registry), Tau paths, mutable
        # credential store for built-ins and an externally owned HTTP client.
        # The built-in credential store is the explicit one, else the reader if it
        # supports set/delete/names, else a file store at the default credentials path.
        # Built-in context environment is a copy of the given mapping or the process
        # environment. No I/O beyond object construction.
        self._generation = ExtensionGeneration()
        self._built_in_extensions = tuple(
            built_in_extension_registry.BUILT_IN_EXTENSIONS
            if built_in_extensions is None
            else built_in_extensions
        )
        self._built_ins_loaded = False
        self._durable_providers = tuple(durable_providers)
        self._provider_credentials = credentials
        self._provider_environment = environment
        built_in_paths = paths or TauPaths()
        resolved_built_in_credentials: CredentialStore = (
            built_in_credentials
            if built_in_credentials is not None
            else (
                cast(CredentialStore, credentials)
                if credentials is not None
                and all(
                    callable(getattr(credentials, name, None))
                    for name in ("set", "delete", "names")
                )
                else FileCredentialStore(credentials_path(built_in_paths))
            )
        )
        self._built_in_context = BuiltInExtensionContext(
            paths=built_in_paths,
            credential_store=resolved_built_in_credentials,
            environment=dict(environment) if environment is not None else dict(os.environ),
            http_client=built_in_http_client,
        )
        self._provider_registry = DynamicProviderRegistry(
            self._durable_providers,
            generation_id=self._generation.id,
            credentials=credentials,
            environment=environment,
        )
        self._local_backend_registry = LocalBackendRegistry(
            self._provider_registry,
            generation_id=self._generation.id,
        )
        self._retired_provider_registries: list[DynamicProviderRegistry] = []
        self._retired_local_backend_registries: list[LocalBackendRegistry] = []
        self._extensions: list[RegisteredExtension] = []
        self._tools: dict[str, RegisteredExtensionTool] = {}
        self._commands: dict[str, ExtensionCommand] = {}
        self._prompt_guidelines: list[tuple[str, str, str]] = []
        self._prompt_sections: list[tuple[str, str, PromptSection]] = []
        self._message_renderers: dict[str, tuple[str, str, MessageRenderer]] = {}
        self._renderer_failures_reported: set[str] = set()
        self._load_diagnostics: list[ResourceDiagnostic] = []
        self._runtime_diagnostics: list[ResourceDiagnostic] = []
        # Keep constructor-provided paths visible until ``load`` installs the
        # authoritative resource-path snapshot.
        self._paths: TauPaths = paths or TauPaths()
        self._session: BoundSession | None = None
        self._ui: UiBridge = ui or NullUiBridge()
        self._turn_requested: TurnRequestedCallback | None = None
        self._harness_unsubscribe: Callable[[], None] | None = None
        self._extension_turn_index = 0

    # -- loading -----------------------------------------------------------

    def load(
        self,
        paths: TauResourcePaths,
        *,
        extra_paths: Sequence[Path] = (),
        include_resource_dirs: bool = True,
        include_project_dir: bool = False,
        include_user_dir: bool = True,
    ) -> None:
        # [FUNCTION DOC]
        # Resolves Tau paths from the resource paths (falling back to root and agents
        # root, default agents home in the user home), loads built-ins once, then
        # discovers extensions (extra paths, resource dirs, project dir and user dir
        # per flags) and runs setup for each, collecting loader diagnostics.
        # Returns nothing. Side effects: imports extension modules and runs setup.
        """Load built-ins, then discover extensions and run isolated setup."""
        self._paths = paths.paths or TauPaths(
            home=paths.root,
            agents_home=paths.agents_root or Path.home() / ".agents",
        )
        self._load_built_ins()
        result = load_extensions(
            paths,
            extra_paths=extra_paths,
            include_resource_dirs=include_resource_dirs,
            include_project_dir=include_project_dir,
            include_user_dir=include_user_dir,
        )
        self._load_diagnostics.extend(result.diagnostics)
        for extension in result.extensions:
            self._setup_extension(extension)

    def _load_built_ins(self) -> None:
        # [FUNCTION DOC]
        # HELPER: runs once per generation: sets up every built-in declaration as a
        # loaded extension with source "built-in"; declarations needing the runtime
        # context get a wrapper passing the built-in context.
        """Load each trusted declaration once, before filesystem sources."""
        if self._built_ins_loaded:
            return
        self._built_ins_loaded = True
        for declaration in self._built_in_extensions:
            setup = declaration.setup
            if declaration.setup_with_context is not None:

                def setup_with_runtime_context(
                    api: ExtensionAPI,
                    declaration: BuiltInExtension = declaration,
                ) -> None:
                    # [FUNCTION DOC]
                    # HELPER: setup wrapper calling the declaration's context-aware setup with the
                    # API and the runtime's built-in context (declaration bound by default arg).
                    assert declaration.setup_with_context is not None
                    declaration.setup_with_context(api, self._built_in_context)

                setup = setup_with_runtime_context
            self._setup_extension(
                LoadedExtension(
                    name=declaration.name,
                    path=None,
                    source_id=declaration.source_id,
                    setup=setup,
                    source="built-in",
                    hidden=declaration.hidden,
                )
            )

    @property
    def active(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether this generation is still active (not invalidated).
        """Return whether this runtime generation still owns live registrations."""
        return self._generation.active

    def retire(self) -> None:
        # [FUNCTION DOC]
        # Idempotently retires the generation: retires backend and provider
        # registries (cancelling their work), invalidates the generation, unsubscribes
        # from the harness and clears all registrations, the turn callback and the
        # session. Does not clear host UI (successor may own it).
        """Invalidate this generation and release all source-owned work."""
        if not self._generation.active:
            return
        # Replacement callers clear host UI before a successor uses the shared
        # bridge; clearing here could erase that successor's freshly mounted UI.
        # Provider retirement synchronously detaches every layer and requests
        # cancellation of generation-owned provider and backend work.
        self._local_backend_registry.retire()
        self._provider_registry.retire()
        self._generation.invalidate()
        if self._harness_unsubscribe is not None:
            self._harness_unsubscribe()
            self._harness_unsubscribe = None
        self._extensions.clear()
        self._tools.clear()
        self._commands.clear()
        self._prompt_guidelines.clear()
        self._prompt_sections.clear()
        self._message_renderers.clear()
        self._renderer_failures_reported.clear()
        self._turn_requested = None
        self._session = None

    async def aclose(self) -> ProviderRegistryCloseResult:
        # [FUNCTION DOC]
        # Async: retires, then closes all backend registries and all provider
        # registries (current and previously retired) concurrently. Keeps retired
        # provider registries that did not drain for a later attempt.
        # Output: close result with drained true when no discovery tasks were
        # contained, and the contained count. Always re-invalidates and unsubscribes.
        # KNOWN ISSUE: an undrained current provider registry is not kept for retry,
        # unlike previously retired ones.
        """Retire this generation and report drain or bounded containment."""
        self.retire()
        provider_registries = (*self._retired_provider_registries, self._provider_registry)
        backend_registries = (
            *self._retired_local_backend_registries,
            self._local_backend_registry,
        )
        try:
            _, provider_results = await asyncio.gather(
                asyncio.gather(*(registry.aclose() for registry in backend_registries)),
                asyncio.gather(*(registry.aclose() for registry in provider_registries)),
            )
            self._retired_provider_registries = [
                registry
                for registry, result in zip(provider_registries, provider_results, strict=True)
                if not result.drained and registry is not self._provider_registry
            ]
            self._retired_local_backend_registries = []
            contained = sum(result.contained_discovery_tasks for result in provider_results)
            return ProviderRegistryCloseResult(
                drained=contained == 0,
                contained_discovery_tasks=contained,
            )
        finally:
            self._generation.invalidate()
            if self._harness_unsubscribe is not None:
                self._harness_unsubscribe()
                self._harness_unsubscribe = None

    def reset_for_reload(self) -> None:
        # [FUNCTION DOC]
        # Prepares for a reload: clears extension UI, retires current registries into
        # the retired lists, invalidates and replaces the generation (stale API
        # objects then raise ExtensionError), creates fresh registries, unsubscribes
        # from the harness, clears all registrations and diagnostics, marks built-ins
        # unloaded and unloads imported extension modules. The bound session is kept.
        """Drop all registrations and imported modules ahead of a re-load.

        Also invalidates the current extension generation (Pi's ``invalidate``
        parity): any `tau` API object, context, or ui facade captured before
        the reload — including one held by a still-running background task —
        raises :class:`ExtensionError` on its next use instead of acting
        against the fresh registration set. Session rebinding does not come
        through here and never invalidates.
        """
        # Host-side extension UI (slot widgets, main views, key interceptors)
        # belongs to the outgoing generation. Tear it down while that generation
        # is still active so host cleanup triggered by component disposal can
        # safely use its API; only then make every captured API/context stale.
        self.clear_ui_components()
        self._local_backend_registry.retire()
        self._retired_local_backend_registries.append(self._local_backend_registry)
        self._provider_registry.retire()
        self._retired_provider_registries.append(self._provider_registry)
        self._generation.invalidate()
        self._generation = ExtensionGeneration()
        self._built_ins_loaded = False
        self._provider_registry = DynamicProviderRegistry(
            self._durable_providers,
            generation_id=self._generation.id,
            credentials=self._provider_credentials,
            environment=self._provider_environment,
        )
        self._local_backend_registry = LocalBackendRegistry(
            self._provider_registry,
            generation_id=self._generation.id,
        )
        if self._harness_unsubscribe is not None:
            self._harness_unsubscribe()
            self._harness_unsubscribe = None
        self._extensions.clear()
        self._tools.clear()
        self._commands.clear()
        self._prompt_guidelines.clear()
        self._prompt_sections.clear()
        self._message_renderers.clear()
        self._renderer_failures_reported.clear()
        self._load_diagnostics.clear()
        self._runtime_diagnostics.clear()
        unload_extension_modules()

    def _setup_extension(self, extension: LoadedExtension) -> None:
        # [FUNCTION DOC]
        # HELPER: registers one loaded extension and runs its setup with a new API
        # object. A duplicate source id is ignored with a diagnostic (first wins). If
        # setup raises, the extension and all its registrations are removed and an
        # error diagnostic ("setup failed" or "built-in setup failed") is recorded.
        source_id = extension.source_id
        if self._extension_by_source(source_id) is not None:
            self._load_diagnostics.append(
                ResourceDiagnostic(
                    kind="extension",
                    name=extension.name,
                    path=extension.path,
                    message="duplicate extension source ignored (first-loaded wins)",
                )
            )
            return
        api = ExtensionAPI(
            self,
            extension.name,
            self._generation,
            source_id=source_id,
        )
        registered = RegisteredExtension(
            name=extension.name,
            source_id=source_id,
            path=extension.path,
            api=api,
            source=extension.source,
            hidden=extension.hidden,
        )
        self._extensions.append(registered)
        try:
            extension.setup(api)
        except Exception as exc:  # noqa: BLE001 - extensions are an isolation boundary
            self._extensions.remove(registered)
            self._remove_registrations(source_id)
            self._load_diagnostics.append(
                ResourceDiagnostic(
                    kind="extension",
                    name=extension.name,
                    path=extension.path,
                    message=(
                        f"built-in setup failed: {exc!r}"
                        if extension.source == "built-in"
                        else f"setup failed: {exc!r}"
                    ),
                    severity="error",
                )
            )

    def _remove_registrations(self, source_id: str) -> None:
        # [FUNCTION DOC]
        # HELPER: removes every tool, command, guideline, section, renderer, provider
        # and local backend owned by the given source id.
        self._tools = {
            name: registration
            for name, registration in self._tools.items()
            if registration.source_id != source_id
        }
        self._commands = {
            name: command
            for name, command in self._commands.items()
            if command.source_id != source_id
        }
        self._prompt_guidelines = [
            (owner, extension, guideline)
            for owner, extension, guideline in self._prompt_guidelines
            if owner != source_id
        ]
        self._prompt_sections = [
            (owner, extension, section)
            for owner, extension, section in self._prompt_sections
            if owner != source_id
        ]
        self._message_renderers = {
            custom_type: registration
            for custom_type, registration in self._message_renderers.items()
            if registration[0] != source_id
        }
        self._provider_registry.unregister_source(source_id)
        self._local_backend_registry.unregister_source(source_id)

    # -- registration (called through ExtensionAPI) -------------------------

    def register_provider(self, source_id: str, provider: DynamicProvider) -> None:
        # [FUNCTION DOC]
        # Registers or atomically replaces the source's dynamic provider layer.
        # Exceptions from the registry propagate.
        """Register or atomically replace one exact extension source layer."""
        self._provider_registry.register(source_id, provider)

    def update_provider(self, source_id: str, provider: DynamicProvider) -> bool:
        # [FUNCTION DOC]
        # Publishes a new provider snapshot for the source without invalidating paired
        # backends; returns the registry's success flag.
        """Publish a provider snapshot without invalidating paired backends."""
        return self._provider_registry.update(source_id, provider)

    def register_local_backend(self, source_id: str, backend: LocalBackend) -> None:
        # [FUNCTION DOC]
        # Registers a local backend against the source's provider layer.
        """Register a backend against its exact source-owned provider layer."""
        self._local_backend_registry.register(source_id, backend)

    def register_tool(self, source_id: str, extension_name: str, tool: AgentTool) -> None:
        # [FUNCTION DOC]
        # Registers an extension tool by name; if the name is taken by another
        # extension tool, records a diagnostic and ignores it (first wins).
        """Register an extension tool; first registration per name wins."""
        existing = self._tools.get(tool.name)
        if existing is not None:
            self._load_diagnostics.append(
                ResourceDiagnostic(
                    kind="extension",
                    name=extension_name,
                    message=(
                        f"tool `{tool.name}` already registered by extension"
                        f" `{existing.extension}`; ignoring duplicate"
                    ),
                )
            )
            return
        self._tools[tool.name] = RegisteredExtensionTool(
            extension=extension_name,
            source_id=source_id,
            tool=tool,
        )

    def register_command(
        self,
        source_id: str,
        extension_name: str,
        name: str,
        handler: ExtensionCommandHandler,
        *,
        description: str = "",
        usage: str | None = None,
        aliases: tuple[str, ...] = (),
    ) -> None:
        # [FUNCTION DOC]
        # Registers a slash command under its normalized name (stripped, leading slash
        # removed, lowercased) with usage defaulting to "/name"; duplicates produce a
        # diagnostic and are ignored.
        """Register an extension slash command; first registration wins."""
        normalized = name.strip().removeprefix("/").lower()
        existing = self._commands.get(normalized)
        if existing is not None:
            self._load_diagnostics.append(
                ResourceDiagnostic(
                    kind="extension",
                    name=extension_name,
                    message=(
                        f"command `/{normalized}` already registered by extension"
                        f" `{existing.extension}`; ignoring duplicate"
                    ),
                )
            )
            return
        self._commands[normalized] = ExtensionCommand(
            extension=extension_name,
            source_id=source_id,
            name=normalized,
            description=description,
            usage=usage or f"/{normalized}",
            aliases=aliases,
            handler=handler,
        )

    def register_message_renderer(
        self,
        source_id: str,
        extension_name: str,
        custom_type: str,
        renderer: MessageRenderer,
    ) -> None:
        # [FUNCTION DOC]
        # Registers a renderer for a stripped custom message type; empty types and
        # duplicates produce diagnostics and are ignored.
        """Register a custom-message renderer; first registration per type wins."""
        normalized = custom_type.strip()
        if not normalized:
            self._load_diagnostics.append(
                ResourceDiagnostic(
                    kind="extension",
                    name=extension_name,
                    message="empty custom_type for message renderer ignored",
                )
            )
            return
        existing = self._message_renderers.get(normalized)
        if existing is not None:
            self._load_diagnostics.append(
                ResourceDiagnostic(
                    kind="extension",
                    name=extension_name,
                    message=(
                        f"message renderer for `{normalized}` already registered by"
                        f" extension `{existing[1]}`; ignoring duplicate"
                    ),
                )
            )
            return
        self._message_renderers[normalized] = (source_id, extension_name, renderer)

    def render_custom_message(
        self,
        custom_type: str,
        content: str,
        details: Mapping[str, JSONValue] | None,
        expanded: bool,
    ) -> str | None:
        # [FUNCTION DOC]
        # Renders a custom message via its registered renderer. Inputs: type, content,
        # details and expanded flag. Output: markup, or none when no renderer exists,
        # it raises, or it returns a non-string (failure diagnosed once per type).
        # KNOWN ISSUE: the once-per-key failure set is shared with render_tool_call,
        # which uses bare tool names as keys, so a custom type equal to a tool name
        # suppresses the other's diagnostic.
        """Render a custom message to markup, or ``None`` to fall back to raw text.

        Installed into every render path (TUI state, print transcript). A missing
        renderer or a renderer that raises or returns a non-string yields
        ``None`` so the frontend renders the raw ``content`` instead of crashing.
        Failures are diagnosed once per ``custom_type`` (render paths re-run on
        every redraw, which would otherwise grow diagnostics without bound).
        """
        registration = self._message_renderers.get(custom_type)
        if registration is None:
            return None
        _, extension_name, renderer = registration
        view = CustomMessageView(custom_type=custom_type, content=content, details=details)
        options = MessageRenderOptions(expanded=expanded)
        try:
            markup = renderer(view, options)
        except Exception as exc:  # noqa: BLE001 - a renderer must never crash the frontend
            if custom_type not in self._renderer_failures_reported:
                self._renderer_failures_reported.add(custom_type)
                self._record_runtime_failure(extension_name, f"message_renderer:{custom_type}", exc)
            return None
        if not isinstance(markup, str):
            if custom_type not in self._renderer_failures_reported:
                self._renderer_failures_reported.add(custom_type)
                self._record_bad_result(extension_name, f"message_renderer:{custom_type}", markup)
            return None
        return markup

    def render_tool_call(
        self,
        name: str,
        arguments: Mapping[str, JSONValue],
    ) -> str | None:
        # [FUNCTION DOC]
        # Renders a tool call line via the extension tool's render_call. Output: the
        # line, or none when unavailable, failing or returning a non-string (diagnosed
        # once per tool name).
        """Render a tool call via its tool's `render_call`, or ``None``.

        Installed into frontends as the tool-call display resolver. A tool
        without a `render_call`, or a renderer that raises or returns a
        non-string, yields ``None`` so the frontend falls back to its
        generic invocation formatting. Failures are diagnosed once per tool
        name (render paths re-run on every redraw).
        """
        registered = self._tools.get(name)
        if registered is None or registered.tool.render_call is None:
            return None
        try:
            line = registered.tool.render_call(arguments)
        except Exception as exc:  # noqa: BLE001 - a renderer must never crash the frontend
            if name not in self._renderer_failures_reported:
                self._renderer_failures_reported.add(name)
                self._record_runtime_failure(registered.extension, f"render_call:{name}", exc)
            return None
        if line is not None and not isinstance(line, str):
            if name not in self._renderer_failures_reported:
                self._renderer_failures_reported.add(name)
                self._record_bad_result(registered.extension, f"render_call:{name}", line)
            return None
        return line

    def render_tool_result(
        self,
        tool_name: str,
        result: AgentToolResult,
        expanded: bool,
    ) -> str | None:
        # [FUNCTION DOC]
        # Renders a tool result via the extension tool's render_result with the
        # expanded flag. Output: markup or none with the same fallback/diagnostic
        # rules (keyed "render_result:<tool>").
        """Render a named tool's result via `render_result`, or ``None``.

        Installed into frontends as the tool-result display resolver, the
        counterpart of `render_tool_call` for the other end of the row's
        lifecycle. A tool without a `render_result`, or a renderer that raises
        or returns a non-string, yields ``None`` so the frontend falls back to
        its generic result formatting. Failures are diagnosed once per tool
        name (render paths re-run on every redraw).
        """
        registered = self._tools.get(tool_name)
        if registered is None or registered.tool.render_result is None:
            return None
        failure_key = f"render_result:{tool_name}"
        try:
            markup = registered.tool.render_result(result, expanded=expanded)
        except Exception as exc:  # noqa: BLE001 - a renderer must never crash the frontend
            if failure_key not in self._renderer_failures_reported:
                self._renderer_failures_reported.add(failure_key)
                self._record_runtime_failure(
                    registered.extension, f"render_result:{tool_name}", exc
                )
            return None
        if markup is not None and not isinstance(markup, str):
            if failure_key not in self._renderer_failures_reported:
                self._renderer_failures_reported.add(failure_key)
                self._record_bad_result(registered.extension, f"render_result:{tool_name}", markup)
            return None
        return markup

    def register_prompt_guideline(
        self, source_id: str, extension_name: str, guideline: str
    ) -> None:
        # [FUNCTION DOC]
        # Adds a stripped guideline line; empty ones produce a diagnostic.
        """Register a standalone system-prompt guideline line."""
        normalized = guideline.strip()
        if not normalized:
            self._load_diagnostics.append(
                ResourceDiagnostic(
                    kind="extension",
                    name=extension_name,
                    message="empty prompt guideline ignored",
                )
            )
            return
        self._prompt_guidelines.append((source_id, extension_name, normalized))

    def register_prompt_section(
        self,
        source_id: str,
        extension_name: str,
        title: str | None,
        body: str,
    ) -> None:
        # [FUNCTION DOC]
        # Adds a prompt section with stripped body and optional stripped title (empty
        # becomes none). Empty bodies and titles containing line breaks produce
        # diagnostics and are ignored.
        """Register a free-form system-prompt section."""
        normalized_body = body.strip()
        if not normalized_body:
            self._load_diagnostics.append(
                ResourceDiagnostic(
                    kind="extension",
                    name=extension_name,
                    message="empty prompt section ignored",
                )
            )
            return
        normalized_title = title.strip() if title is not None else None
        if normalized_title == "":
            normalized_title = None
        if normalized_title is not None and any(
            separator in normalized_title for separator in ("\r", "\n")
        ):
            self._load_diagnostics.append(
                ResourceDiagnostic(
                    kind="extension",
                    name=extension_name,
                    message="prompt section ignored because its title spans multiple lines",
                )
            )
            return
        self._prompt_sections.append(
            (
                source_id,
                extension_name,
                PromptSection(title=normalized_title, body=normalized_body),
            )
        )

    def subscribe(self, source_id: str, event: str, handler: ExtensionHandler) -> None:
        # [FUNCTION DOC]
        # Subscribes a handler to an agent event type, lifecycle event or the
        # wildcard. Unknown event names produce a diagnostic and are ignored.
        # Raises ExtensionError if the source id is not a loaded extension.
        """Subscribe an extension handler to a named event."""
        known = (
            event in AGENT_EVENT_TYPES
            or event in LIFECYCLE_EVENT_TYPES
            or event == AGENT_EVENT_WILDCARD
        )
        if not known:
            self._load_diagnostics.append(
                ResourceDiagnostic(
                    kind="extension",
                    name=self._extension_display_name(source_id),
                    message=f"unknown event `{event}`; handler ignored",
                )
            )
            return
        extension = self._extension_by_source(source_id)
        if extension is None:
            raise ExtensionError(f"unknown extension source: {source_id}")
        extension.handlers.setdefault(event, []).append(handler)

    # -- binding -------------------------------------------------------------

    def bind(self, session: BoundSession) -> None:
        # [FUNCTION DOC]
        # Binds or rebinds the runtime to a session (no generation change).
        """Bind (or re-bind) the runtime to a coding session."""
        self._session = session

    def attach_harness_listener(
        self,
        subscribe: Callable[[Callable[[AgentEvent], Awaitable[None] | None]], Callable[[], None]],
    ) -> None:
        # [FUNCTION DOC]
        # Subscribes the runtime's agent-event adapter through the given subscribe
        # function, first unsubscribing any previous listener; stores the new
        # unsubscribe function.
        """Subscribe the event fan-out to a harness, replacing any prior one."""
        if self._harness_unsubscribe is not None:
            self._harness_unsubscribe()
        self._harness_unsubscribe = subscribe(self._on_agent_event)

    @property
    def provider_credentials(self) -> CredentialReader | None:
        # [FUNCTION DOC]
        # Property: the read-only credential reader given at construction, or none.
        """Return the read-only credential reader used by dynamic runtimes."""
        return self._provider_credentials

    @property
    def provider_environment(self) -> Mapping[str, str] | None:
        # [FUNCTION DOC]
        # Property: the environment mapping given at construction, or none.
        """Return the environment snapshot used by dynamic provider auth."""
        return self._provider_environment

    @property
    def built_in_credentials(self) -> CredentialStore:
        # [FUNCTION DOC]
        # Property: the mutable credential store used by built-ins.
        """Return the injected mutable store used by trusted built-ins."""
        return self._built_in_context.credential_store

    @property
    def built_in_http_client(self) -> httpx.AsyncClient | None:
        # [FUNCTION DOC]
        # Property: the externally owned HTTP client for built-ins, or none.
        """Return the externally owned HTTP client used by trusted built-ins."""
        return self._built_in_context.http_client

    def set_ui_bridge(self, ui: UiBridge) -> None:
        # [FUNCTION DOC]
        # Installs the frontend UI bridge.
        """Install the frontend UI bridge (TUI, print-mode fallback, or test)."""
        self._ui = ui

    def clear_ui_components(self) -> None:
        # [FUNCTION DOC]
        # Asks the UI bridge to tear down all extension-owned UI components.
        """Ask the host frontend to tear down all extension-owned UI.

        Invoked on `/reload` (via ``reset_for_reload``) and by session
        replacement flows (resume/new) before ``session_start`` fires, so
        widgets and key interceptors never outlive the world that mounted
        them while handlers keep the chance to re-mount.
        """
        self._ui.clear_components()

    def set_turn_requested_callback(self, callback: TurnRequestedCallback | None) -> None:
        # [FUNCTION DOC]
        # Installs (or clears) the host callback used to start a turn with a message
        # when the session is idle.
        """Install the host callback used to deliver messages while idle.

        The callback receives the message content plus optional custom-message
        metadata and is expected to submit it through the host's serialized run
        path (the TUI uses the same exclusive worker as user submissions, so
        extension turns cannot race user runs).
        """
        self._turn_requested = callback

    @property
    def ui(self) -> UiBridge:
        # [FUNCTION DOC]
        # Property: the active UI bridge.
        """Return the active UI bridge."""
        return self._ui

    @property
    def session_view(self) -> BoundSession:
        # [FUNCTION DOC]
        # Property: the bound session; raises ExtensionError when unbound.
        """Return the bound session, raising if the runtime is unbound."""
        if self._session is None:
            raise ExtensionError(
                "extension API used before the session was bound; "
                "register handlers in setup() and act on events instead"
            )
        return self._session

    @property
    def provider_registry(self) -> DynamicProviderRegistry:
        # [FUNCTION DOC]
        # Property: this generation's dynamic provider registry.
        """Return this staged runtime generation's process-local provider registry."""
        return self._provider_registry

    @property
    def local_backend_registry(self) -> LocalBackendRegistry:
        # [FUNCTION DOC]
        # Property: this generation's local backend registry.
        """Return this staged runtime generation's local-backend registry."""
        return self._local_backend_registry

    @property
    def paths(self) -> TauPaths:
        # [FUNCTION DOC]
        # Property: resolved Tau filesystem paths.
        """Return the resolved Tau filesystem paths for this runtime."""
        return self._paths

    @property
    def extension_names(self) -> tuple[str, ...]:
        # [FUNCTION DOC]
        # Property: names of non-hidden extensions in load order.
        """Return visible extension names in load order."""
        return tuple(extension.name for extension in self._extensions if not extension.hidden)

    @property
    def extension_metadata(self) -> tuple[ExtensionSourceMetadata, ...]:
        # [FUNCTION DOC]
        # Property: source metadata (name, source id, source, hidden, path) for all
        # loaded extensions in load order.
        """Return active visible and hidden source metadata in load order."""
        return tuple(
            ExtensionSourceMetadata(
                name=extension.name,
                source_id=extension.source_id,
                source=extension.source,
                hidden=extension.hidden,
                path=extension.path,
            )
            for extension in self._extensions
        )

    @property
    def diagnostics(self) -> tuple[ResourceDiagnostic, ...]:
        # [FUNCTION DOC]
        # Property: load diagnostics, then runtime diagnostics, then provider-registry
        # diagnostics converted to provider-kind diagnostics.
        """Return load-time, handler, and provider-refresh diagnostics."""
        provider_diagnostics = tuple(
            ResourceDiagnostic(
                kind="provider",
                name=diagnostic.token.source_id,
                message=(
                    f"{diagnostic.message} for provider "
                    f"`{diagnostic.token.provider_id}` ({diagnostic.reason})"
                ),
            )
            for diagnostic in self._provider_registry.diagnostics
        )
        return (
            tuple(self._load_diagnostics) + tuple(self._runtime_diagnostics) + provider_diagnostics
        )

    @property
    def extension_tools(self) -> tuple[AgentTool, ...]:
        # [FUNCTION DOC]
        # Property: extension tools in registration order.
        """Return extension-registered tools in registration order."""
        return tuple(registration.tool for registration in self._tools.values())

    @property
    def extension_tool_sources(self) -> dict[str, str]:
        # [FUNCTION DOC]
        # Property: mapping from extension tool name to owning extension name.
        """Map extension-registered tool names to their owning extension."""
        return {name: registration.extension for name, registration in self._tools.items()}

    @property
    def prompt_guidelines(self) -> tuple[str, ...]:
        # [FUNCTION DOC]
        # Property: guideline lines in registration order.
        """Return standalone guideline lines in registration order."""
        return tuple(guideline for _, _, guideline in self._prompt_guidelines)

    @property
    def prompt_sections(self) -> tuple[PromptSection, ...]:
        # [FUNCTION DOC]
        # Property: prompt sections in registration order.
        """Return free-form prompt sections in registration order."""
        return tuple(section for _, _, section in self._prompt_sections)

    @property
    def sourced_prompt_sections(self) -> tuple[PromptSection, ...]:
        # [FUNCTION DOC]
        # Property: prompt sections copied with source "extension: <name>".
        """Return prompt sections annotated with their owning extension."""
        return tuple(
            PromptSection(
                title=section.title,
                body=section.body,
                source=f"extension: {extension}",
            )
            for _owner, extension, section in self._prompt_sections
        )

    # -- actions (called through ExtensionAPI) --------------------------------

    def send_user_message(self, content: str, *, deliver_as: str = "follow_up") -> None:
        # [FUNCTION DOC]
        # Delivers a user message: queued (steer or follow-up) while running, else
        # starts a turn through the host callback, else queued as follow-up.
        # Raises ExtensionError if unbound.
        """Deliver a user message into the active run, or start one when idle."""
        self._deliver_message(content, deliver_as=deliver_as, trigger_turn=True)

    def send_custom_message(
        self,
        content: str,
        *,
        custom_type: str,
        details: dict[str, JSONValue] | None = None,
        deliver_as: str = "follow_up",
        trigger_turn: bool = True,
    ) -> None:
        # [FUNCTION DOC]
        # Delivers a custom message with type and details metadata using the same
        # rules; trigger_turn false never starts a new turn while idle.
        """Deliver a custom message carrying render metadata through the pipeline."""
        self._deliver_message(
            content,
            deliver_as=deliver_as,
            trigger_turn=trigger_turn,
            custom_type=custom_type,
            details=details,
        )

    def _deliver_message(
        self,
        content: str,
        *,
        deliver_as: str,
        trigger_turn: bool,
        custom_type: str | None = None,
        details: dict[str, JSONValue] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: shared delivery: running session gets a steering message when
        # deliver_as is "steer" else a follow-up; idle with trigger and callback calls
        # the callback; otherwise queues a follow-up for the next run.
        session = self.session_view
        if session.is_running:
            if deliver_as == "steer":
                session.queue_steering_message(content, custom_type=custom_type, details=details)
            else:
                session.queue_follow_up_message(content, custom_type=custom_type, details=details)
            return
        if trigger_turn and self._turn_requested is not None:
            self._turn_requested(content, custom_type, details)
            return
        # No host run-path registered (print mode, tests) or trigger_turn=False:
        # queue for whichever run happens next.
        session.queue_follow_up_message(content, custom_type=custom_type, details=details)

    async def append_custom_entry(self, namespace: str, data: dict[str, JSONValue]) -> None:
        # [FUNCTION DOC]
        # Async: persists a custom entry via the bound session. Raises ExtensionError
        # if unbound.
        """Persist a `CustomEntry` through the bound session."""
        await self.session_view.append_custom_entry(namespace, data)

    async def set_label(self, target_id: str, label: str | None) -> None:
        # [FUNCTION DOC]
        # Async: sets or clears an entry bookmark via the bound session.
        """Set or clear a per-entry session bookmark through the bound session."""
        await self.session_view.set_label(target_id, label)

    # -- tools ----------------------------------------------------------------

    def compose_tools(self, builtin_tools: Sequence[AgentTool]) -> list[AgentTool]:
        # [FUNCTION DOC]
        # Merges built-in tools with extension tools: same-named extension tools
        # replace built-ins in place, remaining extension tools are appended in
        # registration order; every resulting tool is wrapped with hook seams.
        # Output: new list of tools.
        """Merge built-in and extension tools, then wrap all with hook seams.

        Extension tools override built-ins with the same name in place;
        extension-only tools append in registration order.
        """
        merged: list[AgentTool] = []
        extension_tools = dict(self._tools)
        for tool in builtin_tools:
            override = extension_tools.pop(tool.name, None)
            merged.append(override.tool if override is not None else tool)
        merged.extend(registration.tool for registration in extension_tools.values())
        return [self._wrap_tool(tool) for tool in merged]

    def _wrap_tool(self, tool: AgentTool) -> AgentTool:
        # [FUNCTION DOC]
        # HELPER: returns a copy of the tool whose execute function runs tool_call
        # hooks (may block or rewrite arguments), executes the original tool, then
        # runs tool_result hooks. All other tool fields are preserved.
        async def executor(
            tool_call_id: str,
            arguments: Mapping[str, JSONValue],
            signal: ToolCancellationToken | None = None,
            on_update: ToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            # [FUNCTION DOC]
            # HELPER: wrapped execute function. A blocked call returns a "Tool call
            # blocked: <reason>" text result without running the tool. Exceptions from the
            # underlying tool propagate (result hooks then do not run).
            call_outcome = await self._run_tool_call_hooks(tool.name, arguments)
            if call_outcome.block:
                reason = call_outcome.reason or "blocked by an extension"
                return AgentToolResult(content=[TextContent(text=f"Tool call blocked: {reason}")])
            effective_arguments = (
                call_outcome.arguments if call_outcome.arguments is not None else arguments
            )
            result = await tool.execute(
                tool_call_id,
                effective_arguments,
                signal=signal,
                on_update=on_update,
            )
            return await self._run_tool_result_hooks(tool.name, effective_arguments, result)

        return AgentTool(
            name=tool.name,
            label=tool.label,
            description=tool.description,
            parameters=tool.parameters,
            execute_fn=executor,
            prompt_snippet=tool.prompt_snippet,
            prompt_guidelines=tool.prompt_guidelines,
            prepare_arguments=tool.prepare_arguments,
            execution_mode=tool.execution_mode,
            render_call=tool.render_call,
            render_result=tool.render_result,
        )

    async def _run_tool_call_hooks(
        self,
        tool_name: str,
        arguments: Mapping[str, JSONValue],
    ) -> ToolCallHookResult:
        # [FUNCTION DOC]
        # HELPER: runs tool_call handlers in load order with fresh contexts, chaining
        # argument rewrites. Returns a block result on the first block or on any
        # handler exception (fail-safe, recorded); ignores none and records
        # unsupported results; returns rewritten arguments if any changed, else an
        # empty result.
        effective: Mapping[str, JSONValue] = arguments
        for owner, handler in self._handlers_for("tool_call"):
            event = ToolCallHookEvent(tool_name=tool_name, arguments=effective)
            try:
                result = await _resolve(handler(event, self._fresh_context(owner.source_id)))
            except Exception as exc:  # noqa: BLE001 - fail-safe: an error blocks the tool
                self._record_runtime_failure(owner.name, "tool_call", exc)
                return ToolCallHookResult(
                    block=True,
                    reason=f"extension `{owner.name}` tool_call hook failed: {exc}",
                )
            if result is None:
                continue
            if not isinstance(result, ToolCallHookResult):
                self._record_bad_result(owner.name, "tool_call", result)
                continue
            if result.block:
                return ToolCallHookResult(block=True, reason=result.reason)
            if result.arguments is not None:
                effective = result.arguments
        if effective is arguments:
            return ToolCallHookResult()
        return ToolCallHookResult(arguments=effective)

    async def _run_tool_result_hooks(
        self,
        tool_name: str,
        arguments: Mapping[str, JSONValue],
        result: AgentToolResult,
    ) -> AgentToolResult:
        # [FUNCTION DOC]
        # HELPER: runs tool_result handlers in order; each may replace the content
        # with one text block and/or replace details, producing a copied result passed
        # to the next. Exceptions and unsupported results are recorded and skipped.
        # Output: final result.
        current = result
        for owner, handler in self._handlers_for("tool_result"):
            event = ToolResultHookEvent(tool_name=tool_name, arguments=arguments, result=current)
            try:
                outcome = await _resolve(handler(event, self._fresh_context(owner.source_id)))
            except Exception as exc:  # noqa: BLE001 - result hooks are observational-ish
                self._record_runtime_failure(owner.name, "tool_result", exc)
                continue
            if outcome is None:
                continue
            if not isinstance(outcome, ToolResultHookResult):
                self._record_bad_result(owner.name, "tool_result", outcome)
                continue
            updates: dict[str, object] = {}
            if outcome.content is not None:
                updates["content"] = [TextContent(text=outcome.content)]
            if outcome.details is not None:
                updates["details"] = outcome.details
            if updates:
                current = current.model_copy(update=updates)
        return current

    # -- commands ---------------------------------------------------------------

    def build_command_registry(self) -> CommandRegistry:
        # [FUNCTION DOC]
        # Builds the default command registry plus one slash command per extension
        # command (default description "Extension command (<ext>)", search terms
        # include the extension name). Registration conflicts become diagnostics.
        # Output: the registry.
        """Build a session command registry: defaults plus extension commands."""
        registry = create_default_command_registry()
        for command in self._commands.values():
            slash_command = SlashCommand(
                name=command.name,
                description=command.description or f"Extension command ({command.extension}).",
                usage=command.usage,
                handler=self._command_handler(command),
                aliases=command.aliases,
                search_terms=(command.extension, "extension"),
            )
            try:
                registry.register(slash_command)
            except ValueError as exc:
                self._load_diagnostics.append(
                    ResourceDiagnostic(
                        kind="extension",
                        name=command.extension,
                        message=f"could not register command `/{command.name}`: {exc}",
                    )
                )
        return registry

    def _command_handler(
        self, command: ExtensionCommand
    ) -> Callable[[CommandContext], CommandResult]:
        # [FUNCTION DOC]
        # HELPER: returns a synchronous command handler adapting an extension command.
        def handler(context: CommandContext) -> CommandResult:
            # [FUNCTION DOC]
            # HELPER: calls the extension handler with the args and an extension command
            # context; returns a handled result with its message, or on exception a
            # handled "failed" message and a recorded diagnostic.
            extension_context = ExtensionCommandContext(
                name=command.name,
                args=context.args,
                api=self._api_for(command.source_id),
            )
            try:
                message = command.handler(context.args, extension_context)
            except Exception as exc:  # noqa: BLE001 - extensions are an isolation boundary
                self._record_runtime_failure(command.extension, f"command:/{command.name}", exc)
                return CommandResult(
                    handled=True,
                    message=f"Extension command /{command.name} failed: {exc}",
                )
            return CommandResult(handled=True, message=message)

        return handler

    # -- event dispatch -----------------------------------------------------------

    async def decide_project_trust(self, event: ProjectTrustEvent) -> ExtensionTrustResult | None:
        # [FUNCTION DOC]
        # Async: asks project_trust handlers in order; returns the first trust result
        # whose decision is not "defer", else none. Exceptions and bad results are
        # recorded and skipped (treated as defer).
        """Return the first decisive eligible extension trust result.

        This runtime must contain only built-in, user, and explicit extensions;
        callers load project extensions only after this method resolves.
        """
        for owner, handler in self._handlers_for("project_trust"):
            try:
                result = await _resolve(handler(event, self._fresh_context(owner.source_id)))
            except Exception as exc:  # noqa: BLE001 - trust handlers fail closed/defer
                self._record_runtime_failure(owner.name, "project_trust", exc)
                continue
            if result is None:
                continue
            if not isinstance(result, ExtensionTrustResult):
                self._record_bad_result(owner.name, "project_trust", result)
                continue
            if result.decision != "defer":
                return result
        return None

    async def emit_session_start(self, reason: SessionLifecycleReason) -> None:
        # [FUNCTION DOC]
        # Async: dispatches a session_start event with the reason to handlers.
        """Dispatch `session_start` to subscribed extensions."""
        await self._emit_lifecycle("session_start", SessionStartEvent(reason=reason))

    async def emit_session_shutdown(self, reason: SessionLifecycleReason) -> None:
        # [FUNCTION DOC]
        # Async: dispatches a session_shutdown event with the reason to handlers.
        """Dispatch `session_shutdown` to subscribed extensions."""
        await self._emit_lifecycle("session_shutdown", SessionShutdownEvent(reason=reason))

    async def run_input_hooks(
        self,
        text: str,
        *,
        source: Literal["interactive", "extension"] = "interactive",
        streaming_behavior: Literal["steer", "follow_up"] | None = None,
    ) -> InputHookOutcome:
        # [FUNCTION DOC]
        # Async: runs input handlers in order over the text (with source and streaming
        # behavior on the event). A "handled" result stops and returns handled with
        # the current text and message; "transform" with text replaces the text for
        # later handlers. Exceptions and bad results are recorded and skipped.
        # Output: the outcome (not handled with final text by default).
        """Run `input` hooks over prompt text; transforms chain, handled wins.

        `source`/`streaming_behavior` are surfaced to handlers on the
        `InputEvent` payload; they do not change chaining semantics.
        """
        current = text
        for owner, handler in self._handlers_for("input"):
            try:
                result = await _resolve(
                    handler(
                        InputEvent(
                            text=current,
                            source=source,
                            streaming_behavior=streaming_behavior,
                        ),
                        self._fresh_context(owner.source_id),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - extensions are an isolation boundary
                self._record_runtime_failure(owner.name, "input", exc)
                continue
            if result is None:
                continue
            if not isinstance(result, InputHookResult):
                self._record_bad_result(owner.name, "input", result)
                continue
            if result.action == "handled":
                return InputHookOutcome(handled=True, text=current, message=result.message)
            if result.action == "transform" and result.text is not None:
                current = result.text
        return InputHookOutcome(handled=False, text=current)

    async def emit_event(self, event: object) -> None:
        # [FUNCTION DOC]
        # Async: dispatches an event to handlers for its type and then wildcard
        # handlers, recording exceptions. Raises TypeError if the event has no string
        # type attribute.
        """Dispatch one canonical agent or coding-session event to extensions."""
        event_type = getattr(event, "type", None)
        if not isinstance(event_type, str):
            raise TypeError("Extension events must expose a string type")
        handlers = list(self._handlers_for(event_type))
        handlers.extend(self._handlers_for(AGENT_EVENT_WILDCARD))
        for owner, handler in handlers:
            try:
                await _resolve(handler(event, self._fresh_context(owner.source_id)))
            except Exception as exc:  # noqa: BLE001 - extensions are an isolation boundary
                self._record_runtime_failure(owner.name, event_type, exc)

    async def _on_agent_event(self, event: AgentEvent) -> None:
        # [FUNCTION DOC]
        # HELPER: harness listener; resets the turn index on agent start, converts
        # core turn start/end events into extension turn events (with turn index,
        # millisecond timestamp, message and tool results), emits, and increments the
        # index after each turn end.
        """Adapt core turn events to Pi's extension-facing session metadata."""
        extension_event: object = event
        if isinstance(event, AgentStartEvent):
            self._extension_turn_index = 0
        elif isinstance(event, AgentTurnStartEvent):
            extension_event = TurnStartEvent(
                turn_index=self._extension_turn_index,
                timestamp=time_ns() // 1_000_000,
            )
        elif isinstance(event, AgentTurnEndEvent):
            extension_event = TurnEndEvent(
                turn_index=self._extension_turn_index,
                message=event.message,
                tool_results=list(event.tool_results),
            )
        await self.emit_event(extension_event)
        if isinstance(event, AgentTurnEndEvent):
            self._extension_turn_index += 1

    async def _emit_lifecycle(self, event_name: str, payload: object) -> None:
        # [FUNCTION DOC]
        # HELPER: dispatches a lifecycle payload to handlers of the named event,
        # recording exceptions.
        for owner, handler in self._handlers_for(event_name):
            try:
                await _resolve(handler(payload, self._fresh_context(owner.source_id)))
            except Exception as exc:  # noqa: BLE001 - extensions are an isolation boundary
                self._record_runtime_failure(owner.name, event_name, exc)

    # -- internals -------------------------------------------------------------

    def _handlers_for(self, event: str) -> Iterator[tuple[RegisteredExtension, ExtensionHandler]]:
        # [FUNCTION DOC]
        # HELPER: yields (extension, handler) pairs for an event in load and
        # subscription order.
        for extension in self._extensions:
            for handler in extension.handlers.get(event, ()):
                yield extension, handler

    def _extension_by_source(self, source_id: str) -> RegisteredExtension | None:
        # [FUNCTION DOC]
        # HELPER: returns the registered extension with the source id, or none.
        for extension in self._extensions:
            if extension.source_id == source_id:
                return extension
        return None

    def _extension_display_name(self, source_id: str) -> str:
        # [FUNCTION DOC]
        # HELPER: extension name for a source id, falling back to the id.
        extension = self._extension_by_source(source_id)
        return extension.name if extension is not None else source_id

    def _fresh_context(self, source_id: str) -> ExtensionContext:
        # [FUNCTION DOC]
        # HELPER: builds a new extension context for one handler call bound to the
        # extension API's generation. Raises ExtensionError for unknown sources.
        """Return a fresh context for one handler invocation."""
        api = self._api_for(source_id)
        return ExtensionContext(
            self,
            api._generation,
            extension_name=self._extension_display_name(source_id),
        )

    def _api_for(self, source_id: str) -> ExtensionAPI:
        # [FUNCTION DOC]
        # HELPER: returns the extension API object for a source id or raises
        # ExtensionError.
        extension = self._extension_by_source(source_id)
        if extension is None:
            raise ExtensionError(f"unknown extension source: {source_id}")
        return extension.api

    def record_ui_failure(self, extension: str, context: str, exc: BaseException) -> None:
        # [FUNCTION DOC]
        # Appends an error diagnostic for a failed extension UI component.
        """Record a host-isolated extension UI failure in session diagnostics."""
        self._runtime_diagnostics.append(
            ResourceDiagnostic(
                kind="extension",
                name=extension,
                message=f"UI component `{context}` failed: {exc!r}",
                severity="error",
            )
        )

    def _record_runtime_failure(self, extension: str, event: str, exc: Exception) -> None:
        # [FUNCTION DOC]
        # HELPER: appends an error diagnostic "handler for <event> raised: <exc>".
        self._runtime_diagnostics.append(
            ResourceDiagnostic(
                kind="extension",
                name=extension,
                message=f"handler for `{event}` raised: {exc!r}",
                severity="error",
            )
        )

    def _record_bad_result(self, extension: str, event: str, result: object) -> None:
        # [FUNCTION DOC]
        # HELPER: appends a diagnostic noting an unsupported handler result type.
        self._runtime_diagnostics.append(
            ResourceDiagnostic(
                kind="extension",
                name=extension,
                message=(
                    f"handler for `{event}` returned unsupported"
                    f" result type {type(result).__name__}; ignored"
                ),
            )
        )


async def _resolve(value: object) -> object:
    # [FUNCTION DOC]
    # HELPER: awaits the value if awaitable, otherwise returns it, so handlers can
    # be sync or async.
    if isawaitable(value):
        return await value
    return value
