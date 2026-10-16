# [MODULE DOC]
# Coding-session event models (Pi-compatible) emitted by AgentSession in the
# tau_coding layer on top of the core AgentEvent stream; consumed by
# frontends (print mode, TUI) and SDK users.
# Behavior: defines pydantic wire models discriminated by a literal "type"
# field, a CompactionReason literal ("manual", "threshold", "overflow"), the
# SessionOwnEvent discriminated union of session-only events, and the
# CodingSessionEvent / AgentSessionEvent aliases (agent event or session event).
# Side effects: none beyond class definition.
"""Pi-compatible coding-session events consumed by frontends and SDK users."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from tau_agent.events import AgentEvent
from tau_agent.messages import AgentMessage, WireModel
from tau_agent.session.entries import SessionEntry


class SessionAgentEndEvent(WireModel):
    # [CLASS DOC]
    # Session-level "agent_end" event: the final list of agent messages produced
    # by a run plus a flag telling whether the session will automatically retry
    # (e.g. after a retryable provider error). Fields: type, messages (defaults to
    # empty list), will_retry (defaults to false).
    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage] = Field(default_factory=list)
    will_retry: bool = Field(False)


class AgentSettledEvent(WireModel):
    # [CLASS DOC]
    # Marker event "agent_settled" emitted when the session has fully finished
    # processing (no pending retries/compaction/queued work). No payload fields.
    type: Literal["agent_settled"] = "agent_settled"


class QueueUpdateEvent(WireModel):
    # [CLASS DOC]
    # "queue_update" event carrying snapshots of the queued steering messages and
    # follow-up messages as tuples of strings (both default to empty).
    type: Literal["queue_update"] = "queue_update"
    steering: tuple[str, ...] = ()
    follow_up: tuple[str, ...] = Field(())


CompactionReason = Literal["manual", "threshold", "overflow"]


class CompactionStartEvent(WireModel):
    # [CLASS DOC]
    # "compaction_start" event signalling context compaction began, with the
    # reason (manual, threshold or overflow).
    type: Literal["compaction_start"] = "compaction_start"
    reason: CompactionReason


class CompactionEndEvent(WireModel):
    # [CLASS DOC]
    # "compaction_end" event: reason, optional opaque result object, whether it was
    # aborted, whether the session will retry the prompt afterwards, and an
    # optional error message on failure.
    type: Literal["compaction_end"] = "compaction_end"
    reason: CompactionReason
    result: object | None = None
    aborted: bool = False
    will_retry: bool = Field(False)
    error_message: str | None = Field(None)


class EntryAppendedEvent(WireModel):
    # [CLASS DOC]
    # "entry_appended" event wrapping a SessionEntry that was just appended to the
    # session tree/storage.
    type: Literal["entry_appended"] = "entry_appended"
    entry: SessionEntry


class SessionInfoChangedEvent(WireModel):
    # [CLASS DOC]
    # "session_info_changed" event carrying the session's new display name (or
    # none when cleared).
    type: Literal["session_info_changed"] = "session_info_changed"
    name: str | None = None


class ThinkingLevelChangedEvent(WireModel):
    # [CLASS DOC]
    # "thinking_level_changed" event carrying the new thinking level string.
    type: Literal["thinking_level_changed"] = "thinking_level_changed"
    level: str


class AutoRetryStartEvent(WireModel):
    # [CLASS DOC]
    # "auto_retry_start" event: the attempt number, the maximum attempts, the delay
    # before retrying in milliseconds and the error message that triggered it.
    type: Literal["auto_retry_start"] = "auto_retry_start"
    attempt: int
    max_attempts: int
    delay_ms: int
    error_message: str


class AutoRetryEndEvent(WireModel):
    # [CLASS DOC]
    # "auto_retry_end" event: whether the retry ultimately succeeded, the attempt
    # number reached and an optional final error message.
    type: Literal["auto_retry_end"] = "auto_retry_end"
    success: bool
    attempt: int
    final_error: str | None = Field(None)


type SessionOwnEvent = Annotated[
    SessionAgentEndEvent
    | AgentSettledEvent
    | QueueUpdateEvent
    | CompactionStartEvent
    | CompactionEndEvent
    | EntryAppendedEvent
    | SessionInfoChangedEvent
    | ThinkingLevelChangedEvent
    | AutoRetryStartEvent
    | AutoRetryEndEvent,
    Field(discriminator="type"),
]
type CodingSessionEvent = AgentEvent | SessionOwnEvent
type AgentSessionEvent = CodingSessionEvent
