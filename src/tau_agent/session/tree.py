# [MODULE DOC]
# Helpers for traversing the session entry tree. Session entries form a tree
# via parent ids (supporting Pi-style branching); these functions index entries
# and reconstruct the root-to-leaf path used to rebuild conversation context.
# Pure functions; no I/O.
"""Session tree traversal helpers."""

from __future__ import annotations

from tau_agent.session.entries import SessionEntry


class SessionTreeError(ValueError):
    # [CLASS DOC]
    # Exception (subclass of ValueError) raised when entries do not form a valid
    # tree: duplicate ids, missing parents, or cycles. Carries only a message.
    """Raised when session entries do not form a valid traversable tree."""


def entries_by_id(entries: list[SessionEntry]) -> dict[str, SessionEntry]:
    # [FUNCTION DOC]
    # Build a mapping from entry id to entry for the given list of session entries.
    # Input: list of session entries. Output: new mapping id -> entry (insertion
    # order follows the input). Raises SessionTreeError on a duplicate id.
    # No side effects.
    """Return entries keyed by id, rejecting duplicates."""
    result: dict[str, SessionEntry] = {}
    for entry in entries:
        if entry.id in result:
            raise SessionTreeError(f"Duplicate session entry id: {entry.id}")
        result[entry.id] = entry
    return result


def path_to_entry(entries: list[SessionEntry], leaf_id: str) -> list[SessionEntry]:
    # [FUNCTION DOC]
    # Return the ordered path from the root entry down to the entry with leaf_id.
    # Inputs: list of session entries and the leaf id. Indexes entries (via
    # entries_by_id), then walks parent ids upward from the leaf, collecting
    # entries until an entry has no parent, and reverses the collected list.
    # Output: list of entries, root first, leaf last.
    # Raises SessionTreeError on duplicate ids, a missing id (including a missing
    # leaf or dangling parent id), or a cycle (an id visited twice).
    # No side effects.
    """Return the root-to-leaf path for `leaf_id`."""
    by_id = entries_by_id(entries)
    path: list[SessionEntry] = []
    seen: set[str] = set()
    current_id: str | None = leaf_id

    while current_id is not None:
        if current_id in seen:
            raise SessionTreeError(f"Cycle detected at session entry: {current_id}")
        seen.add(current_id)
        entry = by_id.get(current_id)
        if entry is None:
            raise SessionTreeError(f"Missing session entry: {current_id}")
        path.append(entry)
        current_id = entry.parent_id

    path.reverse()
    return path
