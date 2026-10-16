# [MODULE DOC]
# Frontend-independent contracts for "dynamic" providers that extensions register at
# runtime within one Tau process (tau_coding extensions layer). Defines:
# - auth strategies (required key, optional key, no auth) that resolve credentials
#   from Tau's credential store and then the environment, only at the moment of use,
#   into a runtime-only ResolvedProviderAuth whose secrets are hidden from repr;
# - validated, immutable model descriptions and complete model snapshots;
# - a descriptor for reusing Tau's OpenAI-compatible transport, or alternatively a
#   custom runtime factory that builds a closable model provider;
# - the DynamicProvider definition tying identity, models, transport/factory, auth
#   and an optional model-refresh callback together.
# All dataclasses validate eagerly in their post-init hooks and raise
# DynamicProviderError (a ValueError) or ProviderAuthError; mappings are defensively
# copied into read-only views, and JSON metadata is deep-frozen (lists become tuples,
# objects become read-only mappings). Authorization headers must never be supplied
# as static headers; they come only from the auth strategy.
"""Frontend-free contracts for process-local extension providers."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from inspect import isawaitable
from types import MappingProxyType
from typing import Protocol

import httpx

from tau_agent.provider import CancellationToken, ModelProvider
from tau_agent.types import JSONPrimitive, JSONValue
from tau_ai.env import (
    DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES,
    DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS,
    DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
)
from tau_coding.provider_catalog import ModelInput, ProviderApi
from tau_coding.thinking import ThinkingLevel

_PROVIDER_APIS = frozenset(
    {
        "openai-completions",
        "openai-responses",
        "anthropic-messages",
        "openai-codex-responses",
        "google-generative-ai",
        "mistral-conversations",
    }
)
_THINKING_LEVELS = frozenset({"off", "minimal", "low", "medium", "high", "xhigh", "max"})

type ModelCost = Mapping[str, float]
type ImmutableJSONValue = (
    JSONPrimitive | tuple[ImmutableJSONValue, ...] | Mapping[str, ImmutableJSONValue]
)
type ModelCompat = Mapping[str, JSONValue | ImmutableJSONValue]


class DynamicProviderError(ValueError):
    # [CLASS DOC]
    # Validation error for malformed dynamic provider contracts (a ValueError).
    """Raised when a dynamic provider contract is invalid."""


class ProviderAuthError(RuntimeError):
    # [CLASS DOC]
    # Error raised when provider authentication cannot be resolved or is invalid.
    """Raised when required provider authentication cannot be resolved."""


class _MissingRequiredApiKeyError(ProviderAuthError):
    # [CLASS DOC]
    # HELPER: Subclass marking the host-written "missing required key" message as safe to
    # show to users verbatim at runtime boundaries (it contains no secrets).
    """Host-authored missing-key guidance safe to preserve at runtime boundaries."""


class CredentialReader(Protocol):
    # [CLASS DOC]
    # Protocol for read-only credential lookup by name (e.g. Tau's credential store).
    """Read-only credential lookup used by dynamic auth strategies."""

                                            # [FUNCTION DOC]
                                            # Return the stored secret for a credential name, or None if absent. Protocol stub.
    def get(self, name: str) -> str | None: ...


@dataclass(frozen=True, slots=True)
class ProviderAuthContext:
    # [CLASS DOC]
    # Immutable inputs available while resolving auth: a credential reader and an
    # environment mapping (both hidden from repr).
    """Secret-bearing inputs available only while resolving provider auth."""

    credentials: CredentialReader = field(repr=False)
    environment: Mapping[str, str] = field(repr=False)


@dataclass(frozen=True, slots=True)
class ResolvedProviderAuth:
    # [CLASS DOC]
    # Immutable, runtime-only result of auth resolution: optional API key, extra auth
    # headers, a human-readable source description (e.g. "stored credential"), and
    # whether the Authorization header should be omitted entirely. Secrets are hidden
    # from repr; there is no serialization.
    """Authentication resolved immediately before refresh or runtime creation.

    Secret-bearing fields are deliberately omitted from ``repr``. This value is
    runtime-only and has no serialization helper.
    """

    api_key: str | None = field(default=None, repr=False)
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    source: str = field(default="none", repr=False)
    omit_authorization_header: bool = True

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate the resolved auth: the API key must be None or a non-blank string; the
        # omit flag must be a boolean; the source must be a non-blank string; headers must
        # map non-blank string names (case-insensitively unique) to strings. If no API key
        # is given, Authorization is not omitted, and no Authorization header is present,
        # the result is rejected. Replaces headers with a read-only copy.
        # Raises ProviderAuthError for auth problems, DynamicProviderError for bad headers.
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or not self.api_key.strip()
        ):
            raise ProviderAuthError("Resolved provider API key must be a non-empty string or None")
        if not isinstance(self.omit_authorization_header, bool):
            raise ProviderAuthError("Authorization omission flag must be a boolean")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ProviderAuthError("Authentication source must be a non-empty string")
        headers = _string_mapping(self.headers, "Auth headers")
        has_authorization = _has_header(headers, "authorization")
        if self.api_key is None and not self.omit_authorization_header and not has_authorization:
            raise ProviderAuthError(
                "Resolved provider auth must supply an API key or Authorization header"
            )
        object.__setattr__(self, "headers", headers)


class ProviderAuth(Protocol):
    # [CLASS DOC]
    # Protocol for an auth strategy: an async resolve method taking an auth context and
    # returning a ResolvedProviderAuth. Resolved values must not be persisted.
    """Resolve dynamic provider authentication without persisting it."""

                                                                                   # [FUNCTION DOC]
                                                                                   # Protocol stub: resolve authentication from the given context.
    async def resolve(self, context: ProviderAuthContext) -> ResolvedProviderAuth: ...


@dataclass(frozen=True, slots=True)
class RequiredApiKey:
    # [CLASS DOC]
    # Auth strategy requiring an API key: first the named stored credential, then the
    # named environment variable.
    """Resolve a required API key from Tau credentials, then the environment."""

    credential_name: str
    env_var: str

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate that the credential name and environment variable name are non-blank
        # strings; raises DynamicProviderError otherwise.
        _require_non_empty(self.credential_name, "Credential name")
        _require_non_empty(self.env_var, "Environment variable")

    async def resolve(self, context: ProviderAuthContext) -> ResolvedProviderAuth:
        # [FUNCTION DOC]
        # Return auth using the stored credential if non-empty (source "stored
        # credential"), else the environment variable if non-empty (source names the
        # variable); both send an Authorization header. If neither is set, raise the
        # missing-required-key error telling the user which credential to store or which
        # variable to set. No side effects beyond reads.
        stored = context.credentials.get(self.credential_name)
        if stored:
            return ResolvedProviderAuth(
                api_key=stored,
                source="stored credential",
                omit_authorization_header=False,
            )
        environment_value = context.environment.get(self.env_var)
        if environment_value:
            return ResolvedProviderAuth(
                api_key=environment_value,
                source=f"environment variable {self.env_var}",
                omit_authorization_header=False,
            )
        raise _MissingRequiredApiKeyError(
            f"Missing required authentication. Store credential `{self.credential_name}` "
            f"or set {self.env_var}."
        )


@dataclass(frozen=True, slots=True)
class OptionalApiKey:
    # [CLASS DOC]
    # Auth strategy where an API key is optional (e.g. local servers): same lookup order
    # as the required strategy, but falls back to no auth.
    """Resolve an optional API key, omitting Authorization when absent."""

    credential_name: str
    env_var: str

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate that the credential and environment variable names are non-blank;
        # raises DynamicProviderError otherwise.
        _require_non_empty(self.credential_name, "Credential name")
        _require_non_empty(self.env_var, "Environment variable")

    async def resolve(self, context: ProviderAuthContext) -> ResolvedProviderAuth:
        # [FUNCTION DOC]
        # Return auth from the stored credential, else from the environment variable, else
        # an empty auth that omits the Authorization header. Never raises for absence.
        stored = context.credentials.get(self.credential_name)
        if stored:
            return ResolvedProviderAuth(
                api_key=stored,
                source="stored credential",
                omit_authorization_header=False,
            )
        environment_value = context.environment.get(self.env_var)
        if environment_value:
            return ResolvedProviderAuth(
                api_key=environment_value,
                source=f"environment variable {self.env_var}",
                omit_authorization_header=False,
            )
        return ResolvedProviderAuth()


@dataclass(frozen=True, slots=True)
class NoAuth:
    # [CLASS DOC]
    # Auth strategy that never reads credentials or the environment.
    """Authentication strategy that never reads credentials or environment."""

    async def resolve(self, context: ProviderAuthContext) -> ResolvedProviderAuth:
        # [FUNCTION DOC]
        # Ignore the context and return an empty auth that omits Authorization.
        del context
        return ResolvedProviderAuth()


@dataclass(frozen=True, slots=True)
class ProviderModel:
    # [CLASS DOC]
    # Immutable description of one model in a provider snapshot: id, optional display
    # name, API protocol, base URL, reasoning flag, input modalities (text/image),
    # context window, max output tokens, cost mapping, extra headers (hidden from repr),
    # compat flags mapping, and supported thinking levels. Unknown metadata stays None.
    """One model in a complete dynamic provider snapshot.

    Unknown metadata remains ``None``. Runtime headers are never represented in
    ``repr`` and this type intentionally has no generic persistence method.
    """

    id: str
    display_name: str | None = None
    api: ProviderApi | None = None
    base_url: str | None = None
    reasoning: bool | None = None
    input_modalities: tuple[ModelInput, ...] | None = None
    context_window: int | None = None
    max_tokens: int | None = None
    cost: ModelCost | None = None
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    compat: ModelCompat = field(default_factory=dict)
    thinking_levels: tuple[ThinkingLevel, ...] | None = None

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate and normalize the model: id must be non-blank without surrounding
        # whitespace; display name and base URL non-blank if set; api must be one of the
        # supported protocol names; reasoning a boolean; modalities a tuple of unique
        # "text"/"image" values; context window and max tokens positive integers; cost
        # copied into a read-only mapping of finite non-negative floats; headers copied
        # read-only and must not include Authorization; compat deep-frozen as JSON; thinking
        # levels a tuple of unique known levels (off through max).
        # Raises DynamicProviderError on any violation. Mutates its own frozen fields via
        # object attribute setting.
        _require_identifier(self.id, "Model id")
        if self.display_name is not None:
            _require_non_empty(self.display_name, "Model display name")
        if self.api is not None and self.api not in _PROVIDER_APIS:
            raise DynamicProviderError("Model api is unsupported")
        if self.base_url is not None:
            _require_non_empty(self.base_url, "Model base URL")
        if self.reasoning is not None and not isinstance(self.reasoning, bool):
            raise DynamicProviderError("Model reasoning must be a boolean or None")
        if self.input_modalities is not None:
            if not isinstance(self.input_modalities, tuple):
                raise DynamicProviderError("Model input modalities must be a tuple or None")
            if len(set(self.input_modalities)) != len(self.input_modalities):
                raise DynamicProviderError("Model input modalities must not contain duplicates")
            if any(value not in {"text", "image"} for value in self.input_modalities):
                raise DynamicProviderError("Model input modalities must be text or image")
        _validate_optional_positive_int(self.context_window, "Model context window")
        _validate_optional_positive_int(self.max_tokens, "Model max tokens")
        if self.cost is not None:
            object.__setattr__(self, "cost", _cost_mapping(self.cost))
        headers = _string_mapping(self.headers, "Model headers")
        _reject_authorization_header(headers, "Model headers")
        object.__setattr__(self, "headers", headers)
        object.__setattr__(self, "compat", _json_mapping(self.compat, "Model compat"))
        if self.thinking_levels is not None:
            if not isinstance(self.thinking_levels, tuple):
                raise DynamicProviderError("Model thinking levels must be a tuple or None")
            if any(level not in _THINKING_LEVELS for level in self.thinking_levels):
                raise DynamicProviderError("Model thinking level is unsupported")
            if len(set(self.thinking_levels)) != len(self.thinking_levels):
                raise DynamicProviderError("Model thinking levels must not contain duplicates")


@dataclass(frozen=True, slots=True)
class ProviderModelSnapshot:
    # [CLASS DOC]
    # Immutable, complete set of models returned by discovery plus an optional default
    # model id; replaced atomically on refresh.
    """Complete atomic candidate model snapshot returned by discovery."""

    models: tuple[ProviderModel, ...] = ()
    default_model: str | None = None

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate the snapshot (tuple of models with unique ids; default, if any, must be
        # one of them). Raises DynamicProviderError.
        _validate_model_snapshot(self.models, self.default_model)


@dataclass(frozen=True, slots=True)
class OpenAICompatibleTransport:
    # [CLASS DOC]
    # Descriptor telling Tau to drive this provider with its built-in OpenAI-compatible
    # streaming client: base URL, protocol (chat completions or responses), auth
    # strategy, extra static headers, timeout, retry count and max retry delay, and an
    # optional externally owned HTTP client (never closed by Tau; excluded from repr and
    # equality).
    """Descriptor for Tau's existing OpenAI-compatible streaming transport.

    ``client`` is an optional externally owned client seam for trusted adapters
    and deterministic tests. Providers never close it; the caller owns its
    lifetime. Normal transports leave it unset and use Tau's standard client.
    """

    base_url: str
    api: ProviderApi = "openai-completions"
    auth: ProviderAuth = field(default_factory=NoAuth, repr=False)
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    timeout_seconds: float = DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES
    max_retry_delay_seconds: float = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS
    client: httpx.AsyncClient | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate: non-blank base URL; auth has a callable resolve; api is
        # "openai-completions" or "openai-responses"; headers are a valid read-only mapping
        # without Authorization; timeout is a finite number greater than zero (booleans
        # rejected); max retries a non-negative integer; max retry delay a finite
        # non-negative number. Raises DynamicProviderError.
        _require_non_empty(self.base_url, "Transport base URL")
        if not callable(getattr(self.auth, "resolve", None)):
            raise DynamicProviderError("Transport auth must define resolve(context)")
        if self.api not in {"openai-completions", "openai-responses"}:
            raise DynamicProviderError(
                "OpenAI-compatible transport api must be openai-completions or openai-responses"
            )
        headers = _string_mapping(self.headers, "Transport headers")
        _reject_authorization_header(headers, "Transport headers")
        object.__setattr__(self, "headers", headers)
        if (
            not isinstance(self.timeout_seconds, int | float)
            or isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise DynamicProviderError("Transport timeout must be greater than 0")
        if (
            not isinstance(self.max_retries, int)
            or isinstance(self.max_retries, bool)
            or self.max_retries < 0
        ):
            raise DynamicProviderError("Transport max retries must be 0 or greater")
        if (
            not isinstance(self.max_retry_delay_seconds, int | float)
            or isinstance(self.max_retry_delay_seconds, bool)
            or not math.isfinite(self.max_retry_delay_seconds)
            or self.max_retry_delay_seconds < 0
        ):
            raise DynamicProviderError("Transport max retry delay must be 0 or greater")


@dataclass(frozen=True, slots=True)
class ProviderRuntimeContext:
    # [CLASS DOC]
    # Immutable inputs given to a custom runtime factory: the provider id and the
    # resolved auth (hidden from repr).
    """Frontend-independent inputs passed to a custom runtime factory."""

    provider_id: str
    auth: ResolvedProviderAuth = field(repr=False)


class ClosableModelProvider(ModelProvider, Protocol):
    # [CLASS DOC]
    # Protocol for model providers created by a runtime factory: a normal model provider
    # that also exposes an async close method for releasing resources.
    """Runtime provider returned by a dynamic provider factory."""

                                    # [FUNCTION DOC]
                                    # Protocol stub: release resources held by the provider.
    async def aclose(self) -> None: ...


RuntimeFactory = Callable[
    [ProviderRuntimeContext, ProviderModel],
    Awaitable[ClosableModelProvider] | ClosableModelProvider,
]


@dataclass(frozen=True, slots=True)
class ProviderRefreshContext:
    # [CLASS DOC]
    # Immutable inputs for one model-discovery refresh: a cancellation token, whether
    # network access is allowed, the currently cached models, and resolved auth.
    """Inputs for one bounded, generation-owned discovery operation."""

    signal: CancellationToken
    allow_network: bool
    cached_models: tuple[ProviderModel, ...]
    auth: ResolvedProviderAuth = field(repr=False)


RefreshModels = Callable[[ProviderRefreshContext], Awaitable[ProviderModelSnapshot]]


@dataclass(frozen=True, slots=True)
class DynamicProvider:
    # [CLASS DOC]
    # Immutable, complete provider definition owned by one source (extension/layer):
    # id, display name, model snapshot and default model, exactly one of an
    # OpenAI-compatible transport or a custom runtime factory, the auth strategy used
    # with a runtime factory, an optional async refresh callback producing new
    # snapshots, and an opt-in flag allowing the host to persist "provider id + model
    # id" references for trusted built-ins.
    """A complete process-local provider definition owned by one source layer.

    ``stable_scoped_references`` is an opt-in reserved for trusted built-in
    providers with a stable identity: it lets the host persist scoped-model
    references as ``provider id + exact model id`` pairs only.  Definitions,
    endpoints, credentials, and discovered metadata are never persisted, and
    references resolve only while this trusted source is loaded and its live
    or safely cached snapshot contains the model.
    """

    id: str
    display_name: str
    models: tuple[ProviderModel, ...] = ()
    default_model: str | None = None
    transport: OpenAICompatibleTransport | None = None
    runtime_factory: RuntimeFactory | None = field(default=None, repr=False)
    runtime_auth: ProviderAuth = field(default_factory=NoAuth, repr=False)
    refresh_models: RefreshModels | None = field(default=None, repr=False)
    stable_scoped_references: bool = False

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate the definition: id is a clean identifier; display name non-blank; model
        # snapshot valid; exactly one of transport or runtime factory; transport must be the
        # OpenAI-compatible descriptor and then every model's api must be unset or an
        # OpenAI-compatible protocol; the runtime factory and refresh callback must be
        # callable; with a factory, runtime auth must have a callable resolve; the stable
        # references flag must be boolean. Raises DynamicProviderError.
        _require_identifier(self.id, "Provider id")
        _require_non_empty(self.display_name, "Provider display name")
        _validate_model_snapshot(self.models, self.default_model)
        if (self.transport is None) == (self.runtime_factory is None):
            raise DynamicProviderError(
                "Dynamic provider must define exactly one transport or runtime factory"
            )
        if self.transport is not None and not isinstance(self.transport, OpenAICompatibleTransport):
            raise DynamicProviderError("Dynamic provider transport is unsupported")
        if self.transport is not None and any(
            model.api not in {None, "openai-completions", "openai-responses"}
            for model in self.models
        ):
            raise DynamicProviderError(
                "OpenAI-compatible provider models must use an OpenAI-compatible api"
            )
        if self.runtime_factory is not None and not callable(self.runtime_factory):
            raise DynamicProviderError("Dynamic provider runtime factory must be callable")
        if self.runtime_factory is not None and not callable(
            getattr(self.runtime_auth, "resolve", None)
        ):
            raise DynamicProviderError("Runtime factory auth must define resolve(context)")
        if self.refresh_models is not None and not callable(self.refresh_models):
            raise DynamicProviderError("Dynamic provider refresh callback must be callable")
        if not isinstance(self.stable_scoped_references, bool):
            raise DynamicProviderError(
                "Dynamic provider stable scoped references flag must be boolean"
            )

    @property
    def auth(self) -> ProviderAuth:
        # [FUNCTION DOC]
        # Property: the effective auth strategy, the transport's auth when a transport is
        # used, otherwise the runtime auth.
        """Return the auth strategy for refresh and runtime creation."""
        if self.transport is not None:
            return self.transport.auth
        return self.runtime_auth

    def with_snapshot(self, snapshot: ProviderModelSnapshot) -> DynamicProvider:
        # [FUNCTION DOC]
        # Return a new definition identical to this one but with the snapshot's models and
        # default model (re-validated). Does not mutate this instance.
        """Return this definition with one validated complete snapshot."""
        return DynamicProvider(
            id=self.id,
            display_name=self.display_name,
            models=snapshot.models,
            default_model=snapshot.default_model,
            transport=self.transport,
            runtime_factory=self.runtime_factory,
            runtime_auth=self.runtime_auth,
            refresh_models=self.refresh_models,
            stable_scoped_references=self.stable_scoped_references,
        )


async def resolve_provider_auth(
    auth: ProviderAuth,
    *,
    credentials: CredentialReader,
    environment: Mapping[str, str],
) -> ResolvedProviderAuth:
    # [FUNCTION DOC]
    # Resolve an auth strategy against the given credential reader and environment.
    # Accepts strategies whose resolve returns either an awaitable or a plain value.
    # Returns the resolved auth; raises ProviderAuthError if the result is not a
    # ResolvedProviderAuth, and propagates any error raised by the strategy.
    """Resolve one auth strategy and validate its runtime-only result."""
    resolved = auth.resolve(ProviderAuthContext(credentials=credentials, environment=environment))
    value = await resolved if isawaitable(resolved) else resolved
    if not isinstance(value, ResolvedProviderAuth):
        raise ProviderAuthError("Provider auth returned an unsupported result")
    return value


def _validate_model_snapshot(
    models: tuple[ProviderModel, ...],
    default_model: str | None,
) -> None:
    # [FUNCTION DOC]
    # HELPER: Ensure models is a tuple of ProviderModel values with unique ids and that
    # the default model, if given, is a clean identifier present among them. Raises
    # DynamicProviderError.
    if not isinstance(models, tuple) or any(
        not isinstance(model, ProviderModel) for model in models
    ):
        raise DynamicProviderError("Provider models must be a tuple of ProviderModel values")
    ids = [model.id for model in models]
    if len(ids) != len(set(ids)):
        raise DynamicProviderError("Provider model ids must be unique")
    if default_model is not None:
        _require_identifier(default_model, "Default model")
        if default_model not in ids:
            raise DynamicProviderError("Default model must belong to the provider snapshot")


def _require_identifier(value: str, label: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Ensure a value is a non-blank string without leading/trailing whitespace;
    # raises DynamicProviderError naming the label.
    _require_non_empty(value, label)
    if value != value.strip():
        raise DynamicProviderError(f"{label} must not have surrounding whitespace")


def _require_non_empty(value: str, label: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Ensure a value is a string that is not blank; raises DynamicProviderError
    # naming the label.
    if not isinstance(value, str) or not value.strip():
        raise DynamicProviderError(f"{label} must be a non-empty string")


def _validate_optional_positive_int(value: int | None, label: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Ensure a value is None or a positive integer (booleans rejected); raises
    # DynamicProviderError naming the label.
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise DynamicProviderError(f"{label} must be a positive integer")


def _string_mapping(value: Mapping[str, str], label: str) -> Mapping[str, str]:
    # [FUNCTION DOC]
    # HELPER: Copy a header-style mapping into a read-only mapping, requiring non-blank
    # string keys and string values and rejecting keys that collide case-insensitively.
    # Raises DynamicProviderError.
    copied: dict[str, str] = {}
    normalized_keys: set[str] = set()
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(item, str):
            raise DynamicProviderError(f"{label} must map non-empty strings to strings")
        normalized_key = key.casefold()
        if normalized_key in normalized_keys:
            raise DynamicProviderError(f"{label} must not contain duplicate header names")
        normalized_keys.add(normalized_key)
        copied[key] = item
    return MappingProxyType(copied)


def _has_header(headers: Mapping[str, str], name: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: Return whether a header name is present in the mapping, case-insensitively.
    normalized = name.casefold()
    return any(key.casefold() == normalized for key in headers)


def _reject_authorization_header(headers: Mapping[str, str], label: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Raise DynamicProviderError if the mapping contains an Authorization header
    # (it must come from the auth strategy instead).
    if _has_header(headers, "authorization"):
        raise DynamicProviderError(
            f"{label} must resolve Authorization through the provider auth strategy"
        )


def _cost_mapping(value: Mapping[str, float]) -> Mapping[str, float]:
    # [FUNCTION DOC]
    # HELPER: Copy a cost mapping into a read-only mapping of floats, requiring non-blank
    # string keys and finite, non-negative, non-boolean numeric values. Raises
    # DynamicProviderError.
    # KNOWN ISSUE: the error message mentions only "finite numbers" although negative
    # values are also rejected.
    copied: dict[str, float] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(item, int | float)
            or isinstance(item, bool)
            or not math.isfinite(item)
            or item < 0
        ):
            raise DynamicProviderError("Model cost must map non-empty strings to finite numbers")
        copied[key] = float(item)
    return MappingProxyType(copied)


def json_compatible_mapping(value: ModelCompat) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # Return a mutable deep copy of deep-frozen compat metadata: read-only mappings
    # become ordinary dictionaries and tuples become lists. Pure.
    """Return a mutable JSON-compatible copy of deeply frozen metadata."""
    return {key: _copy_mutable_json(item) for key, item in value.items()}


def _json_mapping(value: ModelCompat, label: str) -> ModelCompat:
    # [FUNCTION DOC]
    # HELPER: Deep-freeze a compat mapping into a read-only mapping after requiring
    # non-blank string keys; values are frozen via the JSON copier. Raises
    # DynamicProviderError.
    copied: dict[str, ImmutableJSONValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise DynamicProviderError(f"{label} keys must be non-empty strings")
        copied[key] = _copy_json(item, label)
    return MappingProxyType(copied)


def _copy_json(value: JSONValue | ImmutableJSONValue, label: str) -> ImmutableJSONValue:
    # [FUNCTION DOC]
    # HELPER: Recursively deep-freeze a JSON value: primitives returned as is (floats
    # must be finite), lists/tuples become tuples, mappings with string keys become
    # read-only mappings. Any other type raises DynamicProviderError.
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DynamicProviderError(f"{label} must contain finite JSON numbers")
        return value
    if isinstance(value, list | tuple):
        return tuple(_copy_json(item, label) for item in value)
    if isinstance(value, Mapping):
        copied: dict[str, ImmutableJSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DynamicProviderError(f"{label} object keys must be strings")
            copied[key] = _copy_json(item, label)
        return MappingProxyType(copied)
    raise DynamicProviderError(f"{label} must contain only JSON values")


def _copy_mutable_json(value: JSONValue | ImmutableJSONValue) -> JSONValue:
    # [FUNCTION DOC]
    # HELPER: Recursively convert a frozen JSON value back to mutable form: primitives
    # as is, sequences to lists, mappings to dictionaries.
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, list | tuple):
        return [_copy_mutable_json(item) for item in value]
    return {key: _copy_mutable_json(item) for key, item in value.items()}
