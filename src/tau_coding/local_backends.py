# [MODULE DOC]
# Provider-neutral local-backend contracts and the generation-local backend registry
# (tau_coding extensions layer). Extensions describe local model servers (e.g. llama.cpp)
# as LocalBackend declarations: a configuration schema, transactional configure, status,
# refresh and optional doctor/reset/load/unload/download/search callbacks. Backends
# return structured, validated data (status snapshots, diagnostics, progress,
# confirmation requests, search results) and never touch UI. The registry binds each
# backend to the provider layer registered by the same source, tracks source layers
# (later registrations shadow earlier ones), runs one operation per (layer, action) as
# an asyncio task with a cancellation token, redacts submitted secrets from every
# backend-authored string, marks results stale when the layer is replaced or the
# generation retired, and drains operations within a bounded cancellation window on
# close. The host (TUI) owns rendering and model selection.
"""Provider-neutral local-backend contracts and generation-local registry.

Backends describe protocol work and structured data.  They never construct UI
widgets or receive a frontend object.  The registry owns source/generation
identity, cancellation, and stale-result containment; the host owns rendering
and session/model selection.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol

from tau_agent.harness import SimpleCancellationToken
from tau_coding.extensions.provider_registry import (
    DynamicProviderRegistry,
    ProviderLayerToken,
)

LOCAL_OPERATION_CANCELLATION_TIMEOUT_SECONDS = 0.25
_SUPERVISED_LOCAL_OPERATION_TASKS: set[asyncio.Task[object]] = set()

LocalFieldKind = Literal["text", "secret", "choice"]
LocalConnectionState = Literal[
    "unconfigured",
    "connecting",
    "loading",
    "ready",
    "unavailable",
    "stale",
    "error",
    "cancelled",
]
LocalAction = Literal[
    "configure",
    "refresh",
    "use",
    "doctor",
    "reset",
    "load_model",
    "unload_model",
    "download_model",
    "search_models",
]
LocalDiagnosticSeverity = Literal["info", "warning", "error"]


class LocalBackendError(ValueError):
    # [CLASS DOC]
    # ValueError raised when a backend contract value or registration is invalid.
    """Raised when a backend contract or registration is invalid."""


class LocalOperationError(RuntimeError):
    # [CLASS DOC]
    # RuntimeError raised for host-owned operation failures (retired registry, unknown backend).
    """Raised for host-owned local operation failures."""


@dataclass(frozen=True, slots=True)
class LocalConfigField:
    # [CLASS DOC]
    # Immutable definition of one host-rendered configuration field: key, label, kind
    # ("text", "secret" or "choice"), required flag, optional placeholder, and choices.
    """One host-rendered configuration field.

    ``secret`` values are accepted only through :class:`LocalConfigValues` and
    are deliberately excluded from reprs and diagnostics by the host.
    """

    key: str
    label: str
    kind: LocalFieldKind
    required: bool = False
    placeholder: str | None = None
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate the field; raises LocalBackendError when key or label is blank, kind is
        # unsupported, required is not a bool, placeholder is not a string or None, choices is
        # not a tuple of non-empty strings, a choice field has no choices, the key has
        # surrounding whitespace, a non-choice field defines choices, or choices repeat.
        _non_empty(self.key, "Configuration field key")
        _non_empty(self.label, "Configuration field label")
        if self.kind not in {"text", "secret", "choice"}:
            raise LocalBackendError("Configuration field kind is unsupported")
        if not isinstance(self.required, bool):
            raise LocalBackendError("Configuration field required flag must be boolean")
        if self.placeholder is not None and not isinstance(self.placeholder, str):
            raise LocalBackendError("Configuration field placeholder must be a string or None")
        if not isinstance(self.choices, tuple) or any(
            not isinstance(choice, str) or not choice for choice in self.choices
        ):
            raise LocalBackendError("Configuration choices must be non-empty strings")
        if self.kind == "choice" and not self.choices:
            raise LocalBackendError("Choice configuration fields must define choices")
        if self.key != self.key.strip():
            raise LocalBackendError("Configuration field key must not have surrounding whitespace")
        if self.kind != "choice" and self.choices:
            raise LocalBackendError("Only choice configuration fields may define choices")
        if len(set(self.choices)) != len(self.choices):
            raise LocalBackendError("Configuration choices must be unique")


@dataclass(frozen=True, slots=True)
class LocalConfigureSpec:
    # [CLASS DOC]
    # Immutable configuration schema: a tuple of LocalConfigField values.
    """Complete configuration schema rendered by the host."""

    fields: tuple[LocalConfigField, ...] = ()

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Raise LocalBackendError if fields is not a tuple of LocalConfigField or keys repeat.
        if not isinstance(self.fields, tuple) or any(
            not isinstance(field, LocalConfigField) for field in self.fields
        ):
            raise LocalBackendError("Configuration fields must be LocalConfigField values")
        keys = [field.key for field in self.fields]
        if len(keys) != len(set(keys)):
            raise LocalBackendError("Configuration field keys must be unique")


ReadConfigureSpec = LocalConfigureSpec | Callable[[], LocalConfigureSpec]


@dataclass(frozen=True, slots=True)
class LocalConfigValues(Mapping[str, str]):
    # [CLASS DOC]
    # Immutable, read-only mapping of submitted configuration values (string to string)
    # plus the set of keys that are secrets. Its repr shows only the keys, never values.
    # Backends should not retain it after configure returns.
    """Ephemeral submitted configuration values.

    The mapping remains usable by backend code, but its values never appear in
    reprs.  Backends should not retain it after their transactional configure
    callback returns.
    """

    _values: Mapping[str, str] = field(repr=False)
    secret_keys: frozenset[str] = field(default_factory=frozenset, repr=False)

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Copy and validate the submitted values (LocalBackendError if any key or value is not a
        # string), require every secret key to be among submitted keys (LocalBackendError), and
        # freeze the copy as a read-only mapping proxy and secret keys as a frozenset.
        copied: dict[str, str] = {}
        for key, value in self._values.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise LocalBackendError("Configuration values must map strings to strings")
            copied[key] = value
        if not self.secret_keys.issubset(copied):
            raise LocalBackendError("Secret configuration keys must be submitted values")
        object.__setattr__(self, "_values", MappingProxyType(copied))
        object.__setattr__(self, "secret_keys", frozenset(self.secret_keys))

    def __getitem__(self, key: str) -> str:
        # [FUNCTION DOC]
        # Return the submitted value for a key; raises KeyError if absent.
        return self._values[key]

    def __iter__(self):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC]
        # Iterate over submitted keys.
        return iter(self._values)

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Return the number of submitted values.
        return len(self._values)

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Return a repr listing only the submitted keys, keeping values (secrets) out of logs.
        return f"LocalConfigValues(keys={tuple(self._values)!r})"


@dataclass(frozen=True, slots=True)
class LocalModel:
    # [CLASS DOC]
    # Immutable, safe model info from a backend: exact id (non-empty, no surrounding
    # whitespace), optional non-blank display name, optional non-blank state string.
    """Safe model information returned by a local backend."""

    id: str
    display_name: str | None = None
    state: str | None = None

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate the id, display name and state; raises LocalBackendError on violations.
        _identifier(self.id, "Local model id")
        if self.display_name is not None:
            _non_empty(self.display_name, "Local model display name")
        if self.state is not None:
            _non_empty(self.state, "Local model state")


@dataclass(frozen=True, slots=True)
class LocalDiagnostic:
    # [CLASS DOC]
    # Immutable secret-free diagnostic: non-blank message, severity "info"/"warning"/"error"
    # (default info) and optional non-blank stage label.
    """Secret-free diagnostic suitable for host rendering."""

    message: str
    severity: LocalDiagnosticSeverity = "info"
    stage: str | None = None

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate message, severity and stage; raises LocalBackendError on violations.
        _non_empty(self.message, "Diagnostic message")
        if self.severity not in {"info", "warning", "error"}:
            raise LocalBackendError("Diagnostic severity is unsupported")
        if self.stage is not None:
            _non_empty(self.stage, "Diagnostic stage")


@dataclass(frozen=True, slots=True)
class LocalProgress:
    # [CLASS DOC]
    # Immutable progress update: non-blank message, optional fraction in [0, 1] (None means
    # indeterminate), and a done flag.
    """Bounded progress update emitted by a backend operation."""

    message: str
    fraction: float | None = None
    done: bool = False

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate message, fraction range and boolean done flag; raises LocalBackendError.
        _non_empty(self.message, "Progress message")
        if self.fraction is not None and not 0 <= self.fraction <= 1:
            raise LocalBackendError("Progress fraction must be between 0 and 1")
        if not isinstance(self.done, bool):
            raise LocalBackendError("Progress done flag must be boolean")


@dataclass(frozen=True, slots=True)
class LocalArtifactOption:
    # [CLASS DOC]
    # Immutable downloadable variant of a search result: exact id, label, optional
    # non-negative integer size in bytes (booleans rejected) and recommended flag.
    """One backend-neutral variant offered for a discovered artifact."""

    id: str
    label: str
    size_bytes: int | None = None
    recommended: bool = False

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate id, label, size and recommended flag; raises LocalBackendError.
        _identifier(self.id, "Artifact option id")
        _non_empty(self.label, "Artifact option label")
        if self.size_bytes is not None and (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise LocalBackendError("Artifact option size must be a non-negative integer")
        if not isinstance(self.recommended, bool):
            raise LocalBackendError("Artifact option recommended flag must be boolean")


@dataclass(frozen=True, slots=True)
class LocalSearchResult:
    # [CLASS DOC]
    # Immutable search hit: exact id, label, restricted flag (e.g. gated), tuple of unique
    # artifact options and optional non-blank diagnostic.
    """One host-renderable backend search result and its selectable variants."""

    id: str
    label: str
    restricted: bool = False
    options: tuple[LocalArtifactOption, ...] = ()
    diagnostic: str | None = None

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate id, label, restricted flag, options type/uniqueness and diagnostic; raises
        # LocalBackendError.
        _identifier(self.id, "Search result id")
        _non_empty(self.label, "Search result label")
        if not isinstance(self.restricted, bool):
            raise LocalBackendError("Search result restricted flag must be boolean")
        if not isinstance(self.options, tuple) or any(
            not isinstance(option, LocalArtifactOption) for option in self.options
        ):
            raise LocalBackendError("Search result options are malformed")
        if len({option.id for option in self.options}) != len(self.options):
            raise LocalBackendError("Search result option ids must be unique")
        if self.diagnostic is not None:
            _non_empty(self.diagnostic, "Search result diagnostic")


@dataclass(frozen=True, slots=True)
class LocalConfirmationChoice:
    # [CLASS DOC]
    # Immutable confirmation option: exact value identifier, label and recommended flag.
    """One host-rendered option answering a backend confirmation request."""

    value: str
    label: str
    recommended: bool = False

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate value, label and recommended flag; raises LocalBackendError.
        _identifier(self.value, "Confirmation choice value")
        _non_empty(self.label, "Confirmation choice label")
        if not isinstance(self.recommended, bool):
            raise LocalBackendError("Confirmation choice recommended flag must be boolean")


@dataclass(frozen=True, slots=True)
class LocalConfirmationRequest:
    # [CLASS DOC]
    # Immutable question a backend needs answered before proceeding: message plus a
    # non-empty tuple of unique-valued choices with at most one recommended. The host
    # renders it and resubmits the chosen value via the operation context's confirmation;
    # an unanswered request leaves the operation uncommitted.
    """An explicit decision a backend needs before an operation may proceed.

    Backends declare structured questions; the host renders, confirms, and
    resubmits the chosen value through the operation context.  A request is
    never an instruction for the host to guess: an unanswered request leaves
    the operation uncommitted.
    """

    message: str
    choices: tuple[LocalConfirmationChoice, ...]

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate message, choices tuple (non-empty, correct types, unique values) and at most
        # one recommended choice; raises LocalBackendError.
        _non_empty(self.message, "Confirmation request message")
        if not isinstance(self.choices, tuple) or not self.choices:
            raise LocalBackendError("Confirmation request choices must be a non-empty tuple")
        if any(not isinstance(choice, LocalConfirmationChoice) for choice in self.choices):
            raise LocalBackendError("Confirmation request choices are malformed")
        values = [choice.value for choice in self.choices]
        if len(set(values)) != len(values):
            raise LocalBackendError("Confirmation request choice values must be unique")
        if sum(1 for choice in self.choices if choice.recommended) > 1:
            raise LocalBackendError("At most one confirmation choice may be recommended")


@dataclass(frozen=True, slots=True)
class LocalBackendStatus:
    # [CLASS DOC]
    # Immutable status snapshot: connection state (unconfigured, connecting, loading, ready,
    # unavailable, stale, error, cancelled), optional endpoint display text, authentication
    # source ("none", "environment", "stored credential"), models, optional selected model
    # id, available actions, diagnostics, and cached/stale flags.
    """Structured status snapshot; no backend-specific protocol vocabulary."""

    state: LocalConnectionState
    endpoint_display: str | None = None
    authentication_source: Literal["none", "environment", "stored credential"] = "none"
    models: tuple[LocalModel, ...] = ()
    selected_model: str | None = None
    actions: tuple[LocalAction, ...] = ()
    diagnostics: tuple[LocalDiagnostic, ...] = ()
    cached: bool = False
    stale: bool = False

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate the snapshot; raises LocalBackendError for unsupported state or auth source,
        # blank endpoint text, non-LocalModel models or duplicate model ids, a selected model not
        # among the models, unsupported or duplicate actions, or malformed diagnostics.
        if self.state not in {
            "unconfigured",
            "connecting",
            "loading",
            "ready",
            "unavailable",
            "stale",
            "error",
            "cancelled",
        }:
            raise LocalBackendError("Local connection state is unsupported")
        if self.endpoint_display is not None:
            _non_empty(self.endpoint_display, "Endpoint display value")
        if self.authentication_source not in {"none", "environment", "stored credential"}:
            raise LocalBackendError("Authentication source is unsupported")
        if not isinstance(self.models, tuple) or any(
            not isinstance(model, LocalModel) for model in self.models
        ):
            raise LocalBackendError("Local models must be LocalModel values")
        model_ids = [model.id for model in self.models]
        if len(model_ids) != len(set(model_ids)):
            raise LocalBackendError("Local model ids must be unique")
        if self.selected_model is not None and self.selected_model not in model_ids:
            raise LocalBackendError("Selected model must belong to the status snapshot")
        if not isinstance(self.actions, tuple) or any(
            action
            not in {
                "configure",
                "refresh",
                "use",
                "doctor",
                "reset",
                "load_model",
                "unload_model",
                "download_model",
                "search_models",
            }
            for action in self.actions
        ):
            raise LocalBackendError("Local actions are unsupported")
        if len(set(self.actions)) != len(self.actions):
            raise LocalBackendError("Local actions must be unique")
        if not isinstance(self.diagnostics, tuple) or any(
            not isinstance(item, LocalDiagnostic) for item in self.diagnostics
        ):
            raise LocalBackendError("Local diagnostics must be structured values")


@dataclass(frozen=True, slots=True)
class LocalOperationContext:
    # [CLASS DOC]
    # Immutable per-operation inputs passed to backend callbacks: cancellation token, the
    # action, generation id, backend id, source id, an optional confirmation value chosen by
    # the user, and private callables to check currency and publish progress.
    """Inputs owned by one backend operation."""

    signal: SimpleCancellationToken
    action: LocalAction
    generation_id: str
    backend_id: str
    source_id: str
    _is_current: Callable[[], bool] = field(repr=False)
    _progress: Callable[[LocalProgress], None] = field(repr=False)
    confirmation: str | None = None

    @property
    def cancelled(self) -> bool:
        # [FUNCTION DOC]
        # Property: True once the operation's cancellation token is cancelled.
        return self.signal.is_cancelled()

    @property
    def stale(self) -> bool:
        # [FUNCTION DOC]
        # Property: True when the operation's registry layer is no longer current (replaced,
        # unregistered or generation retired).
        return not self._is_current()

    def report_progress(self, progress: LocalProgress) -> None:
        # [FUNCTION DOC]
        # Publish one progress update to the host unless the operation is cancelled or stale;
        # the registry redacts secrets, records it and notifies listeners.
        """Publish one structured progress event to the host."""
        if not self.cancelled and not self.stale:
            self._progress(progress)


@dataclass(frozen=True, slots=True)
class LocalOperationResult:
    # [CLASS DOC]
    # Immutable host-renderable outcome: optional status, message, diagnostics, progress
    # history, cancelled/stale/committed flags, per-field error messages,
    # credential-orphaned flag, optional confirmation request, and search results.
    """Common host-renderable operation result."""

    backend_status: LocalBackendStatus | None = None
    message: str | None = None
    diagnostics: tuple[LocalDiagnostic, ...] = ()
    progress: tuple[LocalProgress, ...] = ()
    cancelled: bool = False
    stale: bool = False
    committed: bool = False
    field_errors: Mapping[str, str] = field(default_factory=dict)
    credential_orphaned: bool = False
    confirmation: LocalConfirmationRequest | None = None
    search_results: tuple[LocalSearchResult, ...] = ()

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate types (message string or None; tuples of diagnostics, progress and search
        # results; string-to-string field errors; confirmation type) and that a committed result
        # does not also request confirmation. Raises LocalBackendError.
        if self.message is not None and not isinstance(self.message, str):
            raise LocalBackendError("Operation message must be a string or None")
        if not isinstance(self.diagnostics, tuple) or any(
            not isinstance(item, LocalDiagnostic) for item in self.diagnostics
        ):
            raise LocalBackendError("Operation diagnostics must be structured values")
        if not isinstance(self.progress, tuple) or any(
            not isinstance(item, LocalProgress) for item in self.progress
        ):
            raise LocalBackendError("Operation progress must be structured values")
        if not isinstance(self.field_errors, Mapping):
            raise LocalBackendError("Configuration field errors must be a mapping")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.field_errors.items()
        ):
            raise LocalBackendError("Configuration field errors must contain strings")
        if self.confirmation is not None and not isinstance(
            self.confirmation, LocalConfirmationRequest
        ):
            raise LocalBackendError("Operation confirmation must be a structured request")
        if self.confirmation is not None and self.committed:
            raise LocalBackendError("A committed operation cannot request confirmation")
        if not isinstance(self.search_results, tuple) or any(
            not isinstance(item, LocalSearchResult) for item in self.search_results
        ):
            raise LocalBackendError("Operation search results are malformed")


@dataclass(frozen=True, slots=True)
class LocalConfigureResult:
    # [CLASS DOC]
    # Immutable backend result for a transactional configure attempt: committed flag,
    # optional status, message, field errors, diagnostics and credential-orphaned flag. Not
    # validated itself; the registry converts it into a LocalOperationResult.
    """Backend result for a transactional configuration attempt."""

    committed: bool = False
    backend_status: LocalBackendStatus | None = None
    message: str | None = None
    field_errors: Mapping[str, str] = field(default_factory=dict)
    diagnostics: tuple[LocalDiagnostic, ...] = ()
    credential_orphaned: bool = False


@dataclass(frozen=True, slots=True)
class LocalBackendLayerToken:
    # [CLASS DOC]
    # Immutable identity of one registered backend layer: backend id, source id,
    # generation id, unique layer id ("<generation>:<sequence>") and the paired provider
    # layer token. Used to detect staleness.
    backend_id: str
    source_id: str
    generation_id: str
    layer_id: str
    provider_layer: ProviderLayerToken


@dataclass(frozen=True, slots=True)
class LocalBackendView:
    # [CLASS DOC]
    # Immutable host-facing view of a backend layer: the backend, its token, whether it is
    # recommended, and whether its paired provider layer is currently effective.
    """A backend plus effective-source information for host rendering."""

    backend: LocalBackend
    token: LocalBackendLayerToken
    recommended: bool = False
    provider_layer_effective: bool = True

    @property
    def use_available(self) -> bool:
        # [FUNCTION DOC]
        # Property: True when the backend's provider layer is the effective provider, i.e. the
        # backend is not shadowed and its models can be used.
        return self.provider_layer_effective


class ProgressCallback(Protocol):
    # [CLASS DOC]
    # Protocol for host progress listeners: called with one LocalProgress value.
    def __call__(self, progress: LocalProgress) -> None: ...
        # [FUNCTION DOC]
        # Protocol call signature; receives a progress update and returns nothing.


ConfigureLocalBackend = Callable[
    [LocalConfigValues, LocalOperationContext],
    LocalConfigureResult
    | LocalOperationResult
    | Awaitable[LocalConfigureResult | LocalOperationResult],
]
ReadLocalBackendStatus = Callable[
    [LocalOperationContext],
    LocalBackendStatus
    | LocalOperationResult
    | Awaitable[LocalBackendStatus | LocalOperationResult],
]
RefreshLocalBackend = ReadLocalBackendStatus
DoctorLocalBackend = ReadLocalBackendStatus
ResetLocalBackend = ReadLocalBackendStatus
ManageModel = Callable[
    [str, LocalOperationContext],
    LocalOperationResult | Awaitable[LocalOperationResult],
]
DownloadModel = ManageModel
SearchModels = ManageModel


@dataclass(frozen=True, slots=True)
class LocalBackend:
    # [CLASS DOC]
    # Immutable, provider-neutral declaration of a local backend capability: exact id,
    # paired provider id, display name, configure spec (or a zero-argument callable
    # returning one), required configure/status/refresh callbacks and optional doctor,
    # reset, load/unload/download model and search callbacks, plus a recommended flag.
    # Callbacks may be sync or async and return status/result values described above.
    """Provider-neutral local backend capability declaration."""

    id: str
    provider_id: str
    display_name: str
    configure_spec: ReadConfigureSpec
    configure: ConfigureLocalBackend
    status: ReadLocalBackendStatus
    refresh: RefreshLocalBackend
    doctor: DoctorLocalBackend | None = None
    reset: ResetLocalBackend | None = None
    load_model: ManageModel | None = None
    unload_model: ManageModel | None = None
    download_model: DownloadModel | None = None
    search_models: SearchModels | None = None
    recommended: bool = False

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate ids and display name, that configure/status/refresh are callable, that the
        # configure spec is a spec or callable, the recommended flag is boolean and optional
        # callbacks are callable when present. Raises LocalBackendError.
        _identifier(self.id, "Local backend id")
        _identifier(self.provider_id, "Local backend provider id")
        _non_empty(self.display_name, "Local backend display name")
        if not callable(self.configure) or not callable(self.status) or not callable(self.refresh):
            raise LocalBackendError("Local backend configure, status, and refresh must be callable")
        if not callable(self.configure_spec) and not isinstance(
            self.configure_spec, LocalConfigureSpec
        ):
            raise LocalBackendError("Local backend configure_spec is unsupported")
        if not isinstance(self.recommended, bool):
            raise LocalBackendError("Local backend recommended flag must be boolean")
        for callback in (
            self.doctor,
            self.reset,
            self.load_model,
            self.unload_model,
            self.download_model,
            self.search_models,
        ):
            if callback is not None and not callable(callback):
                raise LocalBackendError("Local backend capability must be callable")

    def read_configure_spec(self) -> LocalConfigureSpec:
        # [FUNCTION DOC]
        # Return the configure spec, calling it if it is a callable; raises LocalBackendError
        # if the result is not a LocalConfigureSpec. Exceptions from the callable propagate.
        value = self.configure_spec() if callable(self.configure_spec) else self.configure_spec
        if not isinstance(value, LocalConfigureSpec):
            raise LocalBackendError("Local backend configure_spec returned an unsupported value")
        return value


@dataclass(eq=False, slots=True)
class _Operation:
    # [CLASS DOC]
    # HELPER: Mutable bookkeeping for one running operation: cancellation token, task,
    # progress history, listener callbacks, latest progress, latest progress with a
    # fraction (so state-only updates don't hide byte progress on replay), and whether
    # cancellation was already requested.
    signal: SimpleCancellationToken
    task: asyncio.Task[LocalOperationResult]
    progress: list[LocalProgress]
    listeners: list[ProgressCallback]
    latest_progress: LocalProgress | None = None
    latest_fractional_progress: LocalProgress | None = None
    cancellation_requested: bool = False


class LocalBackendRegistry:
    # [CLASS DOC]
    # Generation-local, source-bound registry of local backends and their running operations.
    # State: paired provider registry, generation id, optional recommended backend id,
    # per-backend ordered list of (token, backend, sequence) layers (last is effective),
    # a sequence counter, running operations keyed by (token, action), and a retired flag.
    """Generation-local, source-bound local backend registry."""

    def __init__(
        self,
        provider_registry: DynamicProviderRegistry,
        *,
        generation_id: str | None = None,
        recommended_backend_id: str | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Create the registry for a provider registry. Generation id defaults to the provider
        # registry's; a mismatching explicit id raises LocalBackendError. Optional recommended
        # backend id marks that backend recommended in views.
        self._providers = provider_registry
        self._generation_id = generation_id or provider_registry.generation_id
        if self._generation_id != provider_registry.generation_id:
            raise LocalBackendError(
                "Local backend registry and provider registry must share a generation"
            )
        self._recommended_backend_id = recommended_backend_id
        self._layers: dict[str, list[tuple[LocalBackendLayerToken, LocalBackend, int]]] = {}
        self._sequence = 0
        self._operations: dict[tuple[LocalBackendLayerToken, LocalAction], _Operation] = {}
        self._retired = False

    @property
    def generation_id(self) -> str:
        # [FUNCTION DOC]
        # Property returning the registry's generation id.
        return self._generation_id

    def register(self, source_id: str, backend: LocalBackend) -> LocalBackendLayerToken:
        # [FUNCTION DOC]
        # Register a backend for a source. Raises LocalOperationError if retired, and
        # LocalBackendError for an invalid source id or when the same source has not registered
        # the backend's provider. Creates a new layer token, replaces any previous layer from
        # the same source (moving the new one to the top), cancels the replaced layer's
        # operations, and returns the token.
        """Register a backend only against its exact source provider layer."""
        self._assert_active()
        source = _source_id(source_id)
        provider_layer = self._providers.layer_token(backend.provider_id, source)
        if provider_layer is None:
            raise LocalBackendError(
                f"Local backend {backend.id} must be paired with provider {backend.provider_id} "
                "registered by the same source"
            )
        self._sequence += 1
        token = LocalBackendLayerToken(
            backend_id=backend.id,
            source_id=source,
            generation_id=self._generation_id,
            layer_id=f"{self._generation_id}:{self._sequence}",
            provider_layer=provider_layer,
        )
        current = self._layers.get(backend.id, [])
        replaced = [item for item in current if item[0].source_id == source]
        self._layers[backend.id] = [item for item in current if item[0].source_id != source] + [
            (token, backend, self._sequence)
        ]
        for old_token, _, _ in replaced:
            self._cancel_token(old_token)
        return token

    def unregister(self, backend_id: str, source_id: str) -> bool:
        # [FUNCTION DOC]
        # Remove the layer for one backend id and source, keeping other sources' layers; cancels
        # its operations. Returns True if something was removed, else False. Raises
        # LocalBackendError for an invalid source id.
        """Remove one exact source layer and preserve preceding layers."""
        source = _source_id(source_id)
        current = self._layers.get(backend_id, [])
        removed = [item for item in current if item[0].source_id == source]
        if not removed:
            return False
        remaining = [item for item in current if item[0].source_id != source]
        if remaining:
            self._layers[backend_id] = remaining
        else:
            self._layers.pop(backend_id, None)
        for token, _, _ in removed:
            self._cancel_token(token)
        return True

    def unregister_source(self, source_id: str) -> None:
        # [FUNCTION DOC]
        # Remove every backend layer registered by the given source.
        source = _source_id(source_id)
        for backend_id in tuple(self._layers):
            self.unregister(backend_id, source)

    def layers(self, backend_id: str) -> tuple[LocalBackendLayerToken, ...]:
        # [FUNCTION DOC]
        # Return the layer tokens for a backend id in registration order (empty if unknown).
        return tuple(token for token, _, _ in self._layers.get(backend_id, ()))

    def effective(self, backend_id: str) -> LocalBackendView | None:
        # [FUNCTION DOC]
        # Return the view of the top (most recent) layer for a backend, or None if none.
        # The view's provider_layer_effective is True only if the same source's provider layer
        # still matches the token's paired layer and is the provider registry's effective layer.
        current = self._layers.get(backend_id)
        if not current:
            return None
        token, backend, _ = current[-1]
        provider = self._providers.layer_token(backend.provider_id, token.source_id)
        effective_provider = self._providers.effective(backend.provider_id)
        provider_effective = (
            provider == token.provider_layer
            and effective_provider is not None
            and effective_provider.layer_token == provider
        )
        return LocalBackendView(
            backend=backend,
            token=token,
            recommended=backend.recommended or backend.id == self._recommended_backend_id,
            provider_layer_effective=provider_effective,
        )

    def all_views(self, backend_id: str) -> tuple[LocalBackendView, ...]:
        # [FUNCTION DOC]
        # Return views for every layer of a backend, including shadowed ones, computed as in
        # effective.
        """Return every source layer, including shadowed backend layers."""
        views: list[LocalBackendView] = []
        for token, backend, _ in self._layers.get(backend_id, ()):
            provider = self._providers.layer_token(backend.provider_id, token.source_id)
            effective_provider = self._providers.effective(backend.provider_id)
            views.append(
                LocalBackendView(
                    backend=backend,
                    token=token,
                    recommended=backend.recommended or backend.id == self._recommended_backend_id,
                    provider_layer_effective=(
                        provider == token.provider_layer
                        and effective_provider is not None
                        and effective_provider.layer_token == provider
                    ),
                )
            )
        return tuple(views)

    def effective_backends(self) -> tuple[LocalBackendView, ...]:
        # [FUNCTION DOC]
        # Return the effective view of every registered backend in registration order.
        """Return one view per backend in deterministic registration order."""
        views = [self.effective(backend_id) for backend_id in self._layers]
        return tuple(view for view in views if view is not None)

    def operation_running(self, backend_id: str, action: LocalAction) -> bool:
        # [FUNCTION DOC]
        # Return True if any not-yet-finished operation for the backend id and action exists
        # (in any layer).
        """Return whether one effective backend action is still in flight."""
        return any(
            token.backend_id == backend_id and operation == action and not state.task.done()
            for (token, operation), state in self._operations.items()
        )

    def observe_progress(
        self,
        backend_id: str,
        action: LocalAction,
        callback: ProgressCallback,
    ) -> Callable[[], None] | None:
        # [FUNCTION DOC]
        # Subscribe a listener to a running operation for the backend id and action. Returns
        # None if none is running. Otherwise appends the listener, immediately replays the most
        # informative progress (latest fractional progress, else latest progress; listener
        # exceptions suppressed), and returns an unsubscribe function.
        """Observe one running action and replay its best current progress."""
        state = next(
            (
                state
                for (token, operation), state in self._operations.items()
                if token.backend_id == backend_id and operation == action and not state.task.done()
            ),
            None,
        )
        if state is None:
            return None
        state.listeners.append(callback)
        current = state.latest_fractional_progress or state.latest_progress
        if current is not None:
            with suppress(Exception):
                callback(current)

        def unsubscribe() -> None:
            # [FUNCTION DOC]
            # HELPER: Remove the listener from the operation if still present; idempotent.
            with suppress(ValueError):
                state.listeners.remove(callback)

        return unsubscribe

    def cancel(self, backend_id: str, action: LocalAction | None = None) -> bool:
        # [FUNCTION DOC]
        # Request cancellation of all operations of a backend id (in any layer), optionally only
        # for one action. Returns True if any cancellation was newly requested.
        cancelled = False
        for (token, operation), state in tuple(self._operations.items()):
            if token.backend_id == backend_id and (action is None or operation == action):
                cancelled = _request_operation_cancellation(state) or cancelled
        return cancelled

    async def configure(
        self,
        backend_id: str,
        values: Mapping[str, str],
        *,
        progress: ProgressCallback | None = None,
    ) -> LocalOperationResult:
        # [FUNCTION DOC]
        # Run the backend's transactional configure with submitted values.
        # Resolves the effective view (LocalOperationError if retired/unknown), reads the spec
        # and validates values; on field errors returns a result with those errors without
        # calling the backend. Otherwise runs the "configure" action with non-empty secret
        # values registered for redaction. Returns the operation result.
        view = self._require_view(backend_id)
        spec = view.backend.read_configure_spec()
        config_values, errors = _validate_values(spec, values)
        if errors:
            return LocalOperationResult(field_errors=errors)
        secret_values = tuple(
            config_values[key] for key in config_values.secret_keys if config_values[key]
        )
        return await self._run(
            view,
            "configure",
            lambda context: _invoke(view.backend.configure, config_values, context),
            progress=progress,
            secrets=secret_values,
        )

    async def status(
        self, backend_id: str, *, progress: ProgressCallback | None = None
    ) -> LocalOperationResult:
        # [FUNCTION DOC]
        # Run the backend's status callback. It is tracked under the "refresh" action, so it
        # supersedes (cancels) a concurrent refresh of the same layer and vice versa.
        view = self._require_view(backend_id)
        return await self._run(
            view,
            "refresh",
            lambda context: _invoke(view.backend.status, context),
            progress=progress,
        )

    async def refresh(
        self, backend_id: str, *, progress: ProgressCallback | None = None
    ) -> LocalOperationResult:
        # [FUNCTION DOC]
        # Run the backend's refresh callback under the "refresh" action and return its result.
        view = self._require_view(backend_id)
        return await self._run(
            view,
            "refresh",
            lambda context: _invoke(view.backend.refresh, context),
            progress=progress,
        )

    async def doctor(
        self, backend_id: str, *, progress: ProgressCallback | None = None
    ) -> LocalOperationResult:
        # [FUNCTION DOC]
        # Run the optional doctor callback; if absent returns a warning diagnostic "This action
        # is unavailable." without running anything.
        view = self._require_view(backend_id)
        doctor = view.backend.doctor
        if doctor is None:
            return LocalOperationResult(
                diagnostics=(LocalDiagnostic("This action is unavailable.", "warning"),)
            )
        return await self._run(
            view,
            "doctor",
            lambda context: _invoke(doctor, context),
            progress=progress,
        )

    async def reset(
        self, backend_id: str, *, progress: ProgressCallback | None = None
    ) -> LocalOperationResult:
        # [FUNCTION DOC]
        # Run the optional reset callback. Returns an "unavailable" warning when absent, or a
        # warning when the backend is shadowed by another source; otherwise runs "reset".
        view = self._require_view(backend_id)
        reset = view.backend.reset
        if reset is None:
            return LocalOperationResult(
                diagnostics=(LocalDiagnostic("This action is unavailable.", "warning"),)
            )
        if not view.use_available:
            return LocalOperationResult(
                diagnostics=(
                    LocalDiagnostic(
                        "Reset is unavailable while this backend is shadowed by another source.",
                        "warning",
                    ),
                )
            )
        return await self._run(
            view,
            "reset",
            lambda context: _invoke(reset, context),
            progress=progress,
        )

    async def manage_model(
        self,
        backend_id: str,
        action: Literal["load_model", "unload_model", "download_model"],
        model_id: str,
        *,
        progress: ProgressCallback | None = None,
        confirmation: str | None = None,
    ) -> LocalOperationResult:
        # [FUNCTION DOC]
        # Run load_model, unload_model or download_model for a model id, passing an optional
        # confirmation value through the context. Returns an "unavailable" warning when the
        # callback is missing or when the backend is shadowed; otherwise the operation result.
        view = self._require_view(backend_id)
        callback = getattr(view.backend, action)
        if callback is None:
            return LocalOperationResult(
                diagnostics=(LocalDiagnostic("This action is unavailable.", "warning"),)
            )
        if not view.use_available:
            return LocalOperationResult(
                diagnostics=(
                    LocalDiagnostic(
                        "This action is unavailable while this backend is shadowed by "
                        "another source.",
                        "warning",
                    ),
                )
            )
        return await self._run(
            view,
            action,
            lambda context: _invoke(callback, model_id, context),
            progress=progress,
            confirmation=confirmation,
        )

    async def search_models(
        self,
        backend_id: str,
        query: str,
        *,
        progress: ProgressCallback | None = None,
    ) -> LocalOperationResult:
        # [FUNCTION DOC]
        # Run the optional search callback with the trimmed query. Returns warnings when absent,
        # shadowed, or when the query is blank; otherwise the operation result.
        view = self._require_view(backend_id)
        callback = view.backend.search_models
        if callback is None:
            return LocalOperationResult(
                diagnostics=(LocalDiagnostic("This action is unavailable.", "warning"),)
            )
        if not view.use_available:
            return LocalOperationResult(
                diagnostics=(
                    LocalDiagnostic(
                        "This action is unavailable while this backend is shadowed by "
                        "another source.",
                        "warning",
                    ),
                )
            )
        if not query.strip():
            return LocalOperationResult(
                diagnostics=(LocalDiagnostic("Enter a search query.", "warning"),)
            )
        return await self._run(
            view,
            "search_models",
            lambda context: _invoke(callback, query.strip(), context),
            progress=progress,
        )

    def retire(self) -> None:
        # [FUNCTION DOC]
        # Retire the generation once: mark retired, drop all layers and request cancellation of
        # all running operations. Idempotent. Later operations raise LocalOperationError.
        if self._retired:
            return
        self._retired = True
        self._layers.clear()
        for token, _ in tuple(self._operations):
            self._cancel_token(token)

    async def aclose(self) -> None:
        # [FUNCTION DOC]
        # Retire the registry, then concurrently cancel and drain every running operation, each
        # waiting at most the bounded cancellation timeout (0.25 s) without re-cancelling;
        # operations still running are kept referenced until they finish and their late results
        # are reported as stale. Exceptions are collected, not raised.
        """Retire the generation and boundedly drain backend operations.

        A backend is extension code and can ignore task cancellation while it
        unwinds a network client or subprocess.  Do not let final session
        cleanup hang forever: request cancellation once, wait for the fixed
        containment window, and keep a still-running task strongly reachable
        until it actually finishes.  Its context is stale, so late results are
        discarded by :meth:`_run`.
        """
        self.retire()
        operations = tuple(self._operations.values())
        if operations:
            await asyncio.gather(
                *(_cancel_and_drain_operation(operation) for operation in operations),
                return_exceptions=True,
            )

    async def _run(
        self,
        view: LocalBackendView,
        action: LocalAction,
        callback: Callable[[LocalOperationContext], object],
        *,
        progress: ProgressCallback | None,
        secrets: Sequence[str] = (),
        confirmation: str | None = None,
    ) -> LocalOperationResult:
        # [FUNCTION DOC]
        # HELPER: Run one backend operation for a view and action.
        # Cancels any existing operation with the same (token, action) key, creates a fresh
        # cancellation token and context (currency check bound to the token; progress reports
        # redacted with the given secrets, recorded and fanned out to listeners), starts the
        # invocation as a named asyncio task, stores it as the operation for the key and awaits
        # it, removing the entry afterwards if still the same operation. If the layer is no
        # longer current when it finishes, returns a stale result with only progress and the
        # cancelled flag; otherwise returns the operation's result. Cancellation of the awaiting
        # caller propagates into the task.
        token = view.token
        key = (token, action)
        prior = self._operations.get(key)
        if prior is not None:
            self._cancel_token(token, action)
        signal = SimpleCancellationToken()
        progress_events: list[LocalProgress] = []
        listeners = [progress] if progress is not None else []
        operation: _Operation
        context = LocalOperationContext(
            signal=signal,
            action=action,
            generation_id=self._generation_id,
            backend_id=token.backend_id,
            source_id=token.source_id,
            confirmation=confirmation,
            _is_current=lambda: self._token_is_current(token),
            _progress=lambda item: _record_progress(_redact_progress(item, secrets), operation),
        )
        task = asyncio.create_task(
            self._invoke_operation(context, callback, progress_events, secrets),
            name=f"tau-local-backend:{token.source_id}:{token.backend_id}:{action}",
        )
        operation = _Operation(
            signal=signal,
            task=task,
            progress=progress_events,
            listeners=listeners,
        )
        self._operations[key] = operation
        try:
            result = await task
        finally:
            if self._operations.get(key) is operation:
                self._operations.pop(key, None)
        if not self._token_is_current(token):
            return LocalOperationResult(
                progress=tuple(progress_events),
                cancelled=result.cancelled,
                stale=True,
            )
        return result

    async def _invoke_operation(
        self,
        context: LocalOperationContext,
        callback: Callable[[LocalOperationContext], object],
        progress_events: list[LocalProgress],
        secrets: Sequence[str],
    ) -> LocalOperationResult:
        # [FUNCTION DOC]
        # HELPER: Call the operation callback (awaiting if needed) and normalise its value.
        # Task cancellation sets the token cancelled and returns a cancelled result; any other
        # exception returns a generic "The backend operation failed. Try again." error
        # diagnostic (backend text is never surfaced). If the token was cancelled meanwhile,
        # returns a cancelled result. A LocalConfigureResult, LocalBackendStatus or
        # LocalOperationResult is converted into a LocalOperationResult (merging recorded
        # progress before backend-supplied progress) and passed through secret redaction; any
        # other value yields an "invalid result" error diagnostic.
        # KNOWN ISSUE: redaction rebuilds validated values after the exception guard, so if
        # redaction makes values invalid (e.g. two model ids both become the redaction marker),
        # the resulting LocalBackendError escapes instead of becoming a diagnostic.
        try:
            value = callback(context)
            value = await value if inspect.isawaitable(value) else value
        except asyncio.CancelledError:
            context.signal.cancel()
            return LocalOperationResult(progress=tuple(progress_events), cancelled=True)
        except Exception:
            return LocalOperationResult(
                diagnostics=(LocalDiagnostic("The backend operation failed. Try again.", "error"),),
                progress=tuple(progress_events),
            )
        if context.cancelled:
            return LocalOperationResult(progress=tuple(progress_events), cancelled=True)
        if isinstance(value, LocalConfigureResult):
            return _safe_operation_result(
                LocalOperationResult(
                    backend_status=value.backend_status,
                    message=value.message,
                    diagnostics=value.diagnostics,
                    progress=tuple(progress_events),
                    committed=value.committed,
                    field_errors=value.field_errors,
                    credential_orphaned=value.credential_orphaned,
                ),
                secrets,
            )
        if isinstance(value, LocalBackendStatus):
            return _safe_operation_result(
                LocalOperationResult(
                    backend_status=value,
                    progress=tuple(progress_events),
                ),
                secrets,
            )
        if isinstance(value, LocalOperationResult):
            return _safe_operation_result(
                LocalOperationResult(
                    backend_status=value.backend_status,
                    message=value.message,
                    diagnostics=value.diagnostics,
                    progress=tuple((*progress_events, *value.progress)),
                    cancelled=value.cancelled,
                    stale=value.stale,
                    committed=value.committed,
                    field_errors=value.field_errors,
                    credential_orphaned=value.credential_orphaned,
                    confirmation=value.confirmation,
                    search_results=value.search_results,
                ),
                secrets,
            )
        return LocalOperationResult(
            diagnostics=(LocalDiagnostic("The backend returned an invalid result.", "error"),),
            progress=tuple(progress_events),
        )

    def _require_view(self, backend_id: str) -> LocalBackendView:
        # [FUNCTION DOC]
        # HELPER: Return the effective view for a backend id; raises LocalOperationError if the
        # registry is retired or the backend is unknown.
        if self._retired:
            raise LocalOperationError("Local backend registry generation is retired")
        view = self.effective(backend_id)
        if view is None:
            raise LocalOperationError(f"Unknown local backend: {backend_id}")
        return view

    def _token_is_current(self, token: LocalBackendLayerToken) -> bool:
        # [FUNCTION DOC]
        # HELPER: True if not retired, the token's generation matches, and the token is still
        # one of the backend's registered layers (not necessarily the top one).
        if self._retired or token.generation_id != self._generation_id:
            return False
        return any(item[0] == token for item in self._layers.get(token.backend_id, ()))

    def _cancel_token(
        self,
        token: LocalBackendLayerToken,
        action: LocalAction | None = None,
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: Request cancellation of operations for one token (optionally one action).
        # Returns True if any cancellation was newly requested.
        cancelled = False
        for (operation_token, operation_action), operation in tuple(self._operations.items()):
            if operation_token == token and (action is None or operation_action == action):
                cancelled = _request_operation_cancellation(operation) or cancelled
        return cancelled

    def _assert_active(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Raise LocalOperationError if the registry generation has been retired.
        if self._retired:
            raise LocalOperationError("Local backend registry generation is retired")


def _request_operation_cancellation(operation: _Operation) -> bool:
    # [FUNCTION DOC]
    # HELPER: Cancel the operation's token every time, but cancel its task only on the first
    # request. Returns True if this call made the first request, else False.
    """Signal and cancel one operation at most once."""
    operation.signal.cancel()
    if operation.cancellation_requested:
        return False
    operation.cancellation_requested = True
    if not operation.task.done():
        operation.task.cancel()
    return True


async def _cancel_and_drain_operation(operation: _Operation) -> bool:
    # [FUNCTION DOC]
    # HELPER: Bounded drain of one operation. Returns True immediately if done. Otherwise
    # requests cancellation (once), registers the task in a module-level set so it stays
    # referenced until done (with a done callback releasing it), and waits up to the
    # cancellation timeout for it via a shield (so the wait itself does not re-cancel).
    # Returns whether the task is done after timeout, True on completion or task exceptions.
    # If the drainer itself is cancelled, re-raises unless the task already finished.
    """Wait through one bounded cancellation window without re-cancelling."""
    if operation.task.done():
        return True
    _request_operation_cancellation(operation)
    _SUPERVISED_LOCAL_OPERATION_TASKS.add(operation.task)
    operation.task.add_done_callback(_release_supervised_operation)
    try:
        async with asyncio.timeout(LOCAL_OPERATION_CANCELLATION_TIMEOUT_SECONDS):
            await asyncio.shield(operation.task)
    except TimeoutError:
        return operation.task.done()
    except asyncio.CancelledError:
        # The cleanup owner itself was cancelled.  The task remains supervised
        # and receives no second cancellation while its backend unwinds.
        if operation.task.done():
            return True
        raise
    except Exception:
        return True
    return True


def _release_supervised_operation(task: asyncio.Task[object]) -> None:
    # [FUNCTION DOC]
    # HELPER: Done callback: remove the task from the supervised set and retrieve its
    # exception (suppressing cancellation) so asyncio does not log it as unretrieved.
    _SUPERVISED_LOCAL_OPERATION_TASKS.discard(task)
    with suppress(asyncio.CancelledError):
        task.exception()


def _validate_values(
    spec: LocalConfigureSpec,
    values: Mapping[str, str],
) -> tuple[LocalConfigValues, Mapping[str, str]]:
    # [FUNCTION DOC]
    # HELPER: Validate submitted values against a spec. Errors: unknown keys ("Unknown
    # configuration field."), required fields blank after trimming ("This field is
    # required."), choice values not among choices ("Choose one of the listed values.").
    # Returns (LocalConfigValues of all submitted values with submitted secret-field keys
    # marked secret, read-only mapping of errors). Non-string values cause LocalBackendError.
    submitted = dict(values)
    fields = {field.key: field for field in spec.fields}
    errors: dict[str, str] = {}
    for key in submitted:
        if key not in fields:
            errors[key] = "Unknown configuration field."
    for key, config_field in fields.items():
        value = submitted.get(key, "")
        if config_field.required and not value.strip():
            errors[key] = "This field is required."
        if config_field.kind == "choice" and value and value not in config_field.choices:
            errors[key] = "Choose one of the listed values."
    secret_keys = frozenset(
        field.key for field in spec.fields if field.kind == "secret" and field.key in submitted
    )
    return LocalConfigValues(submitted, secret_keys=secret_keys), MappingProxyType(errors)


def _record_progress(item: LocalProgress, operation: _Operation) -> None:
    # [FUNCTION DOC]
    # HELPER: Append progress to the operation's history, update latest and (if it has a
    # fraction) latest-fractional progress, and notify each listener, suppressing listener
    # exceptions so UI teardown cannot fail backend work.
    operation.progress.append(item)
    operation.latest_progress = item
    if item.fraction is not None:
        operation.latest_fractional_progress = item
    for callback in tuple(operation.listeners):
        # A closing or replaced host must not turn backend work into a failed
        # transaction merely because progress rendering disappeared.
        with suppress(Exception):
            callback(item)


async def _invoke(callback: Callable[..., object], *args: object) -> object:
    # [FUNCTION DOC]
    # HELPER: Call a backend callback with flexible arity and await its result if needed.
    # Inspects the signature: if it has no required positional parameters, calls it with no
    # arguments; if it accepts fewer positional parameters than supplied, passes only the
    # leading ones; otherwise passes all. If the signature cannot be inspected, passes all.
    # KNOWN ISSUE: a callback whose positional parameters all have defaults is called with no
    # arguments, so it never receives the model id/context even if it could accept them.
    """Invoke documented callbacks, accepting a no-context test double too."""
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        value = callback(*args)
    else:
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        ]
        required = [
            parameter for parameter in positional if parameter.default is inspect.Parameter.empty
        ]
        if len(required) == 0:
            value = callback()
        elif len(positional) < len(args):
            value = callback(*args[: len(positional)])
        else:
            value = callback(*args)
    return await value if inspect.isawaitable(value) else value


def _redact(value: str | None, secrets: Sequence[str]) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Replace every non-empty secret occurrence in a string with "[redacted]";
    # None passes through.
    if value is None:
        return None
    result = value
    for secret in secrets:
        if secret:
            result = result.replace(secret, "[redacted]")
    return result


def _redact_progress(item: LocalProgress, secrets: Sequence[str]) -> LocalProgress:
    # [FUNCTION DOC]
    # HELPER: Return a copy of a progress update with its message redacted (falling back to
    # "[redacted]" if it becomes empty), keeping fraction and done.
    return LocalProgress(
        _redact(item.message, secrets) or "[redacted]",
        item.fraction,
        item.done,
    )


def _safe_operation_result(
    result: LocalOperationResult,
    secrets: Sequence[str],
) -> LocalOperationResult:
    # [FUNCTION DOC]
    # HELPER: Rebuild an operation result with every backend-authored string redacted:
    # status endpoint, model ids/names/states, selected model, status and result
    # diagnostics, progress messages, message, field errors, confirmation message and
    # choice labels (choice values kept), and search result ids/labels/options/diagnostics.
    # committed is forced False when there are field errors. May raise LocalBackendError if
    # redacted values fail validation.
    """Keep backend-authored values out of host messages and diagnostics."""
    status = result.backend_status
    if status is not None:
        status = LocalBackendStatus(
            state=status.state,
            endpoint_display=_redact(status.endpoint_display, secrets),
            authentication_source=status.authentication_source,
            models=tuple(
                LocalModel(
                    id=_redact(model.id, secrets) or "[redacted]",
                    display_name=_redact(model.display_name, secrets),
                    state=_redact(model.state, secrets),
                )
                for model in status.models
            ),
            selected_model=_redact(status.selected_model, secrets),
            actions=status.actions,
            diagnostics=tuple(
                LocalDiagnostic(
                    _redact(item.message, secrets) or "[redacted]",
                    item.severity,
                    item.stage,
                )
                for item in status.diagnostics
            ),
            cached=status.cached,
            stale=status.stale,
        )
    confirmation = result.confirmation
    if confirmation is not None:
        confirmation = LocalConfirmationRequest(
            message=_redact(confirmation.message, secrets) or "[redacted]",
            choices=tuple(
                LocalConfirmationChoice(
                    choice.value,
                    _redact(choice.label, secrets) or "[redacted]",
                    choice.recommended,
                )
                for choice in confirmation.choices
            ),
        )
    return LocalOperationResult(
        backend_status=status,
        message=_redact(result.message, secrets),
        diagnostics=tuple(
            LocalDiagnostic(
                _redact(item.message, secrets) or "[redacted]",
                item.severity,
                item.stage,
            )
            for item in result.diagnostics
        ),
        progress=tuple(
            LocalProgress(
                _redact(item.message, secrets) or "[redacted]",
                item.fraction,
                item.done,
            )
            for item in result.progress
        ),
        cancelled=result.cancelled,
        stale=result.stale,
        committed=result.committed and not result.field_errors,
        field_errors={
            key: _redact(message, secrets) or "[redacted]"
            for key, message in result.field_errors.items()
        },
        credential_orphaned=result.credential_orphaned,
        confirmation=confirmation,
        search_results=tuple(
            LocalSearchResult(
                id=_redact(item.id, secrets) or "[redacted]",
                label=_redact(item.label, secrets) or "[redacted]",
                restricted=item.restricted,
                options=tuple(
                    LocalArtifactOption(
                        id=_redact(option.id, secrets) or "[redacted]",
                        label=_redact(option.label, secrets) or "[redacted]",
                        size_bytes=option.size_bytes,
                        recommended=option.recommended,
                    )
                    for option in item.options
                ),
                diagnostic=_redact(item.diagnostic, secrets),
            )
            for item in result.search_results
        ),
    )


def _source_id(value: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Validate a source id is non-blank without surrounding whitespace (else
    # LocalBackendError) and return it.
    _non_empty(value, "Local backend source id")
    if value != value.strip():
        raise LocalBackendError("Local backend source id must not have surrounding whitespace")
    return value


def _non_empty(value: str, label: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Raise LocalBackendError "<label> must be a non-empty string" unless the value
    # is a string with non-whitespace content.
    if not isinstance(value, str) or not value.strip():
        raise LocalBackendError(f"{label} must be a non-empty string")


def _identifier(value: str, label: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Require a non-empty string with no surrounding whitespace; raises
    # LocalBackendError naming the label otherwise.
    _non_empty(value, label)
    if value != value.strip():
        raise LocalBackendError(f"{label} must not have surrounding whitespace")


__all__ = [
    "LOCAL_OPERATION_CANCELLATION_TIMEOUT_SECONDS",
    "ConfigureLocalBackend",
    "DoctorLocalBackend",
    "DownloadModel",
    "LocalAction",
    "LocalArtifactOption",
    "LocalBackend",
    "LocalBackendError",
    "LocalBackendLayerToken",
    "LocalBackendRegistry",
    "LocalBackendStatus",
    "LocalBackendView",
    "LocalConfigField",
    "LocalConfigValues",
    "LocalConfigureResult",
    "LocalConfigureSpec",
    "LocalConfirmationChoice",
    "LocalConfirmationRequest",
    "LocalConnectionState",
    "LocalDiagnostic",
    "LocalFieldKind",
    "LocalModel",
    "LocalOperationContext",
    "LocalOperationError",
    "LocalOperationResult",
    "LocalProgress",
    "LocalSearchResult",
    "ManageModel",
    "ReadConfigureSpec",
    "ReadLocalBackendStatus",
    "RefreshLocalBackend",
    "ResetLocalBackend",
    "SearchModels",
]
