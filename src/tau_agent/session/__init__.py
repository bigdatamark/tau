# [PACKAGE DOC]
# Package facade for Tau's append-only session tree primitives (part of the
# portable tau_agent harness, independent of CLI/session file locations).
# Behavior: re-exports the session entry models (message, model change,
# thinking level change, compaction, branch summary, custom, custom message,
# label, leaf, session info entries and the SessionEntry union), JSONL
# serialization helpers and their error type, the in-memory SessionState,
# storage backends (abstract SessionStorage, in-memory and JSONL file storage),
# and tree helpers (entries_by_id, path_to_entry, SessionTreeError).
# Inputs/outputs: none at runtime beyond the exported names.
# Side effects: imports the submodules entries, jsonl, memory, storage, tree.
"""Append-only session tree primitives for Tau."""

from __future__ import annotations

from tau_agent.session.entries import (
    BaseSessionEntry,
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    CustomMessageEntry,
    LabelEntry,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionEntry,
    SessionInfoEntry,
    ThinkingLevelChangeEntry,
)
from tau_agent.session.jsonl import (
    SessionJsonlError,
    entries_from_json_lines,
    entry_from_json_line,
    entry_to_json_line,
)
from tau_agent.session.memory import SessionState
from tau_agent.session.storage import InMemorySessionStorage, JsonlSessionStorage, SessionStorage
from tau_agent.session.tree import SessionTreeError, entries_by_id, path_to_entry

__all__ = [
    "BaseSessionEntry",
    "BranchSummaryEntry",
    "CompactionEntry",
    "CustomEntry",
    "CustomMessageEntry",
    "InMemorySessionStorage",
    "JsonlSessionStorage",
    "LabelEntry",
    "LeafEntry",
    "MessageEntry",
    "ModelChangeEntry",
    "SessionEntry",
    "SessionInfoEntry",
    "SessionJsonlError",
    "SessionState",
    "SessionStorage",
    "SessionTreeError",
    "ThinkingLevelChangeEntry",
    "entries_by_id",
    "entries_from_json_lines",
    "entry_from_json_line",
    "entry_to_json_line",
    "path_to_entry",
]
