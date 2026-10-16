# [MODULE DOC]
# Pydantic models for append-only session log entries (one JSON record per line
# in a session file, Pi-style). Each entry has an id, optional parent id (forming
# a tree that supports branching) and a timestamp; the "type" literal field acts
# as the discriminator of the SessionEntry union, which callers use to parse any
# entry. Extra fields are forbidden. No I/O here.
"""Append-only session entry models."""

from __future__ import annotations

from time import time
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from tau_agent.messages import AgentMessage, Usage, UserContent
from tau_agent.types import JSONValue


def new_entry_id() -> str:
    # [FUNCTION DOC]
    # HELPER: Return a fresh random unique entry id as a 32-character hexadecimal
    # string (UUID4). No inputs; no side effects beyond randomness.
    """Return a unique session entry id."""
    return uuid4().hex


def current_timestamp() -> float:
    # [FUNCTION DOC]
    # HELPER: Return the current wall-clock time as Unix seconds (float). No inputs.
    """Return the current Unix timestamp."""
    return time()


class BaseSessionEntry(BaseModel):
    # [CLASS DOC]
    # Base model for all entries: id (defaults to a new random hex id), parent_id
    # (None for a root) and timestamp (defaults to now). Rejects unknown fields
    # during validation (raises a pydantic validation error).
    """Common fields shared by all append-only session entries."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_entry_id)
    parent_id: str | None = None
    timestamp: float = Field(default_factory=current_timestamp)


class MessageEntry(BaseSessionEntry):
    # [CLASS DOC]
    # Entry with type "message" holding one agent transcript message (user,
    # assistant, tool result, ...).
    """A transcript message entry."""

    type: Literal["message"] = "message"
    message: AgentMessage


class CustomMessageEntry(BaseSessionEntry):
    # [CLASS DOC]
    # Entry with type "custom_message": an extension-injected message included in
    # model context. Fields: custom type tag, user-style content, display flag
    # (whether frontends show it, default True) and optional JSON details.
    """An extension-injected message that participates in model context."""

    type: Literal["custom_message"] = "custom_message"
    custom_type: str
    content: UserContent
    display: bool = True
    details: JSONValue = None


class ModelChangeEntry(BaseSessionEntry):
    # [CLASS DOC]
    # Entry with type "model_change" recording a model switch: the model id and an
    # optional provider id (optional only for backwards compatibility with old
    # transcripts).
    """A model selection change entry.

    ``provider`` was added after Tau's first session format.  It remains
    optional so older transcripts continue to load; new entries always carry
    the provider selected by the host.
    """

    type: Literal["model_change"] = "model_change"
    model: str
    provider: str | None = None


class ThinkingLevelChangeEntry(BaseSessionEntry):
    # [CLASS DOC]
    # Entry with type "thinking_level_change" recording a reasoning level change;
    # the level may be None (meaning default/off).
    """A thinking/reasoning level change entry."""

    type: Literal["thinking_level_change"] = "thinking_level_change"
    thinking_level: str | None = None


class CompactionEntry(BaseSessionEntry):
    # [CLASS DOC]
    # Entry with type "compaction": a summary that replaces older messages when the
    # session is replayed into context. Fields: summary text; legacy list of replaced
    # entry ids (omitted from serialization when empty); id of the first kept entry;
    # token count before compaction; optional usage and provider/model info for the
    # summarization call, including the provider that actually responded.
    """A context summary that replaces older message entries during replay."""

    type: Literal["compaction"] = "compaction"
    summary: str
    # Legacy Tau sessions stored every replaced id. Keep accepting and replaying
    # that shape, but omit the empty compatibility field from new JSONL records.
    replaces_entry_ids: list[str] = Field(default_factory=list, exclude_if=lambda ids: not ids)
    first_kept_entry_id: str | None = None
    tokens_before: int | None = None
    usage: Usage | None = None
    provider: str | None = None
    model: str | None = None
    response_provider: str | None = None


class BranchSummaryEntry(BaseSessionEntry):
    # [CLASS DOC]
    # Entry with type "branch_summary": a summary of an abandoned branch, with
    # optional branch root id and usage/provider/model information. Described as a
    # future/reserved entry kind.
    """A future branch summary entry."""

    type: Literal["branch_summary"] = "branch_summary"
    summary: str
    branch_root_id: str | None = None
    usage: Usage | None = None
    provider: str | None = None
    model: str | None = None
    response_provider: str | None = None


class LabelEntry(BaseSessionEntry):
    # [CLASS DOC]
    # Entry with type "label": sets (or clears when label is None) a bookmark label
    # on the entry identified by target_id.
    """A bookmark change attached to one session entry."""

    type: Literal["label"] = "label"
    target_id: str
    label: str | None = None


class LeafEntry(BaseSessionEntry):
    # [CLASS DOC]
    # Entry with type "leaf": legacy pointer to the active tip entry, kept only so
    # old session files still deserialize.
    """A legacy active-tip pointer retained only for deserialization."""

    type: Literal["leaf"] = "leaf"
    entry_id: str | None = None


class SessionInfoEntry(BaseSessionEntry):
    # [CLASS DOC]
    # Entry with type "session_info": session metadata with creation time (defaults
    # to now), optional working directory and optional title.
    """Basic session metadata entry."""

    type: Literal["session_info"] = "session_info"
    created_at: float = Field(default_factory=current_timestamp)
    cwd: str | None = None
    title: str | None = None


class CustomEntry(BaseSessionEntry):
    # [CLASS DOC]
    # Entry with type "custom": arbitrary extension/application data stored under a
    # namespace as a JSON-compatible mapping (default empty). Not sent to the model.
    """Extension/application-owned session data."""

    type: Literal["custom"] = "custom"
    namespace: str
    data: dict[str, JSONValue] = Field(default_factory=dict)


type SessionEntry = Annotated[
    MessageEntry
    | CustomMessageEntry
    | ModelChangeEntry
    | ThinkingLevelChangeEntry
    | CompactionEntry
    | BranchSummaryEntry
    | LabelEntry
    | LeafEntry
    | SessionInfoEntry
    | CustomEntry,
    Field(discriminator="type"),
]
