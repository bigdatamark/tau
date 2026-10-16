# [MODULE DOC]
# Provider-neutral OAuth contracts for the coding application: immutable value
# types describing browser auth URLs, device codes, text and selection prompts,
# runtime request auth; callback type aliases a frontend supplies to a login
# flow; the OAuthProvider protocol that concrete providers implement; and a
# helper to read string metadata from stored credentials. OAuthFlowKind is the
# literal set "browser" or "device_code".
"""Provider-neutral OAuth contracts used by Tau's coding application."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from tau_agent.types import JSONValue
from tau_coding.credentials import OAuthCredential

OAuthFlowKind = Literal["browser", "device_code"]


@dataclass(frozen=True, slots=True)
class OAuthAuthInfo:
    # [CLASS DOC]
    # Frozen value: authorization URL the user must open plus optional instructions,
    # passed to the frontend's on_auth callback in a browser flow.
    """Authorization URL and optional instructions for a browser flow."""

    url: str
    instructions: str | None = None


@dataclass(frozen=True, slots=True)
class OAuthDeviceCodeInfo:
    # [CLASS DOC]
    # Frozen value: user code and verification URI from a device authorization
    # request, with optional polling interval and expiry in seconds.
    """User-facing values returned by an OAuth device authorization request."""

    user_code: str
    verification_uri: str
    interval_seconds: float | None = None
    expires_in_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class OAuthPrompt:
    # [CLASS DOC]
    # Frozen value: request for free-text input with message, optional placeholder,
    # and whether an empty answer is acceptable.
    """Text input requested by an OAuth provider before or during login."""

    message: str
    placeholder: str | None = None
    allow_empty: bool = False


@dataclass(frozen=True, slots=True)
class OAuthSelectOption:
    # [CLASS DOC]
    # Frozen value: one selectable choice with a stable id and display label.
    """One choice in a provider-defined OAuth selection prompt."""

    id: str
    label: str


@dataclass(frozen=True, slots=True)
class OAuthSelectPrompt:
    # [CLASS DOC]
    # Frozen value: a selection request with a message and a tuple of options.
    """Selection input requested by an OAuth provider."""

    message: str
    options: tuple[OAuthSelectOption, ...]


@dataclass(frozen=True, slots=True)
class OAuthRuntimeAuth:
    # [CLASS DOC]
    # Frozen value: credentials for actual API requests derived from a stored OAuth
    # credential: API key/token, optional base URL override, optional extra headers.
    """Request authentication derived from a stored OAuth credential."""

    api_key: str
    base_url: str | None = None
    headers: Mapping[str, str] | None = None


AuthCallback = Callable[[OAuthAuthInfo], None]
DeviceCodeCallback = Callable[[OAuthDeviceCodeInfo], None]
PromptCallback = Callable[[OAuthPrompt], Awaitable[str]]
SelectCallback = Callable[[OAuthSelectPrompt], Awaitable[str | None]]
ManualCodeCallback = Callable[[], Awaitable[str]]
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class OAuthLoginCallbacks:
    # [CLASS DOC]
    # Frozen bundle of frontend callbacks for a login flow: on_auth (show URL),
    # on_device_code (show code), on_prompt (async text input), on_select (async
    # selection, nothing means cancelled), optional on_progress messages, optional
    # manual code input, and an optional preferred flow kind.
    """Frontend-independent callbacks available to an OAuth login flow."""

    on_auth: AuthCallback
    on_device_code: DeviceCodeCallback
    on_prompt: PromptCallback
    on_select: SelectCallback
    on_progress: ProgressCallback | None = None
    on_manual_code_input: ManualCodeCallback | None = None
    method: OAuthFlowKind | None = None


class OAuthProvider(Protocol):
    # [CLASS DOC]
    # Protocol for provider-specific OAuth implementations registered with Tau:
    # exposes identity, supported flow kinds, login, refresh and runtime auth.
    """Provider-specific OAuth behavior registered with Tau."""

    @property
    def id(self) -> str:
        # [FUNCTION DOC]
        # Property: stable identifier used for the provider and its stored credential.
        """Stable provider/credential identifier."""
        ...

    @property
    def name(self) -> str:
        # [FUNCTION DOC]
        # Property: human-readable provider name for UI.
        """User-facing provider name."""
        ...

    @property
    def flow_kinds(self) -> Sequence[OAuthFlowKind]:
        # [FUNCTION DOC]
        # Property: the sequence of interactive flow kinds this provider supports.
        """Interactive flow families supported by this provider."""
        ...

    async def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredential:
        # [FUNCTION DOC]
        # Run the interactive login using the supplied callbacks and return the new
        # credential to persist. Implementations do network I/O and invoke callbacks;
        # may raise on failure or cancellation.
        """Complete login and return credentials to persist."""
        ...

    async def refresh(self, credential: OAuthCredential) -> OAuthCredential:
        # [FUNCTION DOC]
        # Exchange an expired credential for a fresh one (network I/O); may raise.
        """Refresh an expired credential."""
        ...

    def runtime_auth(self, credential: OAuthCredential) -> OAuthRuntimeAuth:
        # [FUNCTION DOC]
        # Convert a stored credential into request auth (key, base URL, headers).
        # Pure transformation; may raise if the credential lacks required data.
        """Convert stored credentials to request auth."""
        ...


def oauth_metadata_string(
    metadata: Mapping[str, JSONValue],
    name: str,
) -> str | None:
    # [FUNCTION DOC]
    # Look up a key in credential metadata and return its stripped value if it is a
    # non-empty string; otherwise return nothing. Never raises.
    """Return one non-empty string from provider-specific OAuth metadata."""
    value = metadata.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
