# [MODULE DOC]
# Shared, trust-aware staging of coding sessions for startup and session replacement.
# Frontends (CLI print mode, TUI) call prepare_coding_session to load a candidate
# CodingSession with authoritative (durable) writes deferred, then either adopt it
# (committing staged session entries and taking ownership) or abort it (closing it).
# This gives every frontend the same "candidate first, publish on adopt" durability
# boundary, analogous to Pi's session-switch flow.
"""Shared trust-aware staging for coding-session startup and replacement."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from tau_agent.provider import ModelProvider
from tau_coding.session import CodingSession, CodingSessionConfig


@dataclass(frozen=True, slots=True)
class SessionPreparationRequest:
    # [CLASS DOC]
    # Immutable, frontend-neutral description of a session to stage.
    # Fields: an opaque storage object, the destination working directory, the model name,
    # optional provider name, optional session-level provider override and optional
    # session id to resume. Pure data; not consumed by code in this module.
    """Frontend-neutral request for a staged coding session."""

    storage: object
    destination_cwd: Path
    model: str
    provider: str | None = None
    session_provider: str | None = None
    session_id: str | None = None


@dataclass(slots=True)
class PreparedCodingSession:
    # [CLASS DOC]
    # Wrapper around a candidate CodingSession whose resources are not authoritative until
    # adopted. Tracks a lifecycle state string: "prepared" -> "adopted" or "aborted".
    """A candidate session whose resources are not authoritative until adopted."""

    session: CodingSession
    _state: str = "prepared"

    @property
    def provider(self) -> ModelProvider:
        # [FUNCTION DOC]
        # Property returning the candidate session's model provider. No side effects.
        return self.session.provider

    async def adopt(self) -> CodingSession:
        # [FUNCTION DOC]
        # Commit the staged session and hand ownership of it to the caller.
        # Behavior: if state is not "prepared", raises RuntimeError naming the current state.
        # If the session exposes a project trust resolution that was cancelled, aborts the
        # candidate and raises ValueError. Otherwise calls the session's optional private
        # commit-prepared-entries coroutine (if present) to append the staged entry batch;
        # any exception (including cancellation, as BaseException) triggers abort and is
        # re-raised. On success sets state to "adopted" and returns the session.
        # Side effects: durable session writes via commit; may close the session on failure.
        """Commit staged entries, then transfer the candidate's ownership."""
        if self._state != "prepared":
            raise RuntimeError(f"Prepared session is already {self._state}")
        trust_resolution = getattr(self.session, "project_trust_resolution", None)
        if trust_resolution is not None and trust_resolution.cancelled:
            await self.abort()
            raise ValueError("Project trust decision cancelled; session was not adopted")
        try:
            commit = getattr(self.session, "_commit_prepared_entries", None)
            if commit is not None:
                await commit()
        except BaseException:
            await self.abort()
            raise
        self._state = "adopted"
        return self.session

    async def abort(self) -> None:
        # [FUNCTION DOC]
        # Close an unpublished candidate at most once.
        # No-op if state is not "prepared" (already adopted or aborted). Otherwise sets state
        # to "aborted" and awaits the session's optional aclose coroutine if it exists.
        # Returns nothing; exceptions from aclose propagate (state is already "aborted").
        """Close an unpublished candidate exactly once."""
        if self._state != "prepared":
            return
        self._state = "aborted"
        close = getattr(self.session, "aclose", None)
        if close is not None:
            await close()


async def prepare_coding_session(
    config: CodingSessionConfig,
    *,
    session_loader: type[CodingSession] | None = None,
) -> PreparedCodingSession:
    # [FUNCTION DOC]
    # Load a candidate CodingSession through the shared trust/provider lifecycle.
    # Inputs: a CodingSessionConfig and optional keyword session_loader class (defaults to
    # CodingSession) providing an async load class method.
    # Behavior: copies the config with defer_authoritative_writes forced to True, awaits
    # loader.load on it, and wraps the result in a PreparedCodingSession in "prepared" state.
    # Output: the PreparedCodingSession. Exceptions raised by load propagate.
    # Side effects: whatever loading performs (reading session files, building provider,
    # resource discovery); durable writes are deferred until adopt.
    """Prepare a session through the shared trust/provider lifecycle.

    Application frontends use this entry point with authoritative writes
    deferred. Adoption appends the complete staged batch before exposing the
    candidate. Supplying a provider remains the compatibility seam for static
    embedded callers, which retain the historical lazy initial write.
    """
    # Every frontend gets the same candidate-first durability boundary, even
    # when it supplies a compatibility provider object. Provider ownership is
    # controlled independently by CodingSessionConfig.owns_initial_provider.
    staged_config = replace(config, defer_authoritative_writes=True)
    loader = session_loader or CodingSession
    session = await loader.load(staged_config)
    return PreparedCodingSession(session)


__all__ = [
    "PreparedCodingSession",
    "SessionPreparationRequest",
    "prepare_coding_session",
]
