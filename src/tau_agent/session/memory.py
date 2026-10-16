# [MODULE DOC]
# Reconstructs in-memory session state by replaying append-only session entries along one branch
# of the session tree (Pi-style JSONL sessions). Produces the LLM context messages plus
# model/provider/thinking settings, labels, session info, custom and compaction entries. A private
# sentinel object distinguishes "no leaf argument given" from an explicit None leaf.
"""In-memory session state reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

from tau_agent.messages import AgentMessage, CustomMessage, UserMessage
from tau_agent.session.entries import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    SessionEntry,
    SessionInfoEntry,
)
from tau_agent.session.tree import path_to_entry

_UNSET_LEAF_ID: Final[object] = object()


@dataclass(frozen=True, slots=True)
class SessionState:
    # [CLASS DOC]
    # Immutable snapshot (frozen, slotted dataclass) of a session derived from its entries.
    # Fields: context messages in order; latest model, provider and thinking level (or None);
    # label text and label timestamps keyed by target entry id (resolved over all entries, not
    # just the branch); active leaf id; latest session-info entry; custom entries and compaction
    # entries on the branch; the entry ids that produced each context message (parallel to
    # messages); and the replayed branch entries.
    """Current session state derived from append-only entries."""

    messages: tuple[AgentMessage, ...]
    model: str | None
    provider: str | None
    thinking_level: str | None
    labels_by_id: dict[str, str]
    label_timestamps_by_id: dict[str, float]
    active_leaf_id: str | None
    session_info: SessionInfoEntry | None
    custom_entries: tuple[CustomEntry, ...]
    compaction_entries: tuple[CompactionEntry, ...]
    context_entry_ids: tuple[str, ...]
    entries: tuple[SessionEntry, ...]

    @classmethod
    def from_entries(
        cls,
        entries: list[SessionEntry],
        *,
        leaf_id: str | None | object = _UNSET_LEAF_ID,
    ) -> SessionState:
        # [FUNCTION DOC]
        # Class method that builds a state by replaying the root-to-leaf path.
        # Leaf selection: when no leaf id is given, use the id of the last entry in file order
        # whose type is not "leaf" (None if there is none); otherwise use the given id, where None
        # means the empty path. The path is computed with the tree helper from all entries to that
        # leaf (empty when leaf is None).
        # Replay per entry type on the path: message entries add their message; custom-message
        # entries add a custom message built from the entry (timestamp converted from seconds to
        # integer milliseconds); model-change sets model and, when present, provider (provider is
        # sticky otherwise); thinking-level-change sets thinking level; label and leaf entries are
        # ignored here; session-info replaces session info; custom entries are collected;
        # compaction entries are collected and rewrite the accumulated message rows via the
        # compaction helper, given the path prefix before that entry; branch-summary entries add a
        # user message wrapping the summary.
        # Inputs: list of all session entries; optional keyword leaf id. Output: a new state.
        # Raises whatever the path helper raises for an unknown leaf id. No side effects.
        """Replay the branch ending at the active entry.

        By default the active entry is the last non-leaf entry in file order.
        Historical ``leaf`` records remain readable but never select the tip.
        An explicit `leaf_id` supports in-memory tree navigation; passing
        ``None`` selects the empty path before the first root entry.
        """
        resolved_leaf_id = (
            _last_non_leaf_id(entries) if leaf_id is _UNSET_LEAF_ID else cast(str | None, leaf_id)
        )
        replay_entries = (
            path_to_entry(entries, resolved_leaf_id) if resolved_leaf_id is not None else []
        )

        message_rows: list[tuple[str, AgentMessage]] = []
        model: str | None = None
        provider: str | None = None
        thinking_level: str | None = None
        labels_by_id, label_timestamps_by_id = _resolve_labels(entries)
        active_leaf_id: str | None = resolved_leaf_id
        session_info: SessionInfoEntry | None = None
        custom_entries: list[CustomEntry] = []
        compaction_entries: list[CompactionEntry] = []

        for entry_index, entry in enumerate(replay_entries):
            match entry.type:
                case "message":
                    message_rows.append((entry.id, entry.message))
                case "custom_message":
                    message_rows.append(
                        (
                            entry.id,
                            CustomMessage(
                                custom_type=entry.custom_type,
                                content=entry.content,
                                display=entry.display,
                                details=entry.details,
                                timestamp=round(entry.timestamp * 1000),
                            ),
                        )
                    )
                case "model_change":
                    model = entry.model
                    if entry.provider is not None:
                        provider = entry.provider
                case "thinking_level_change":
                    thinking_level = entry.thinking_level
                case "label":
                    pass  # Resolved globally above so off-path bookmarks remain visible.
                case "leaf":
                    pass  # Backward-compatible historical record; never selects the tip.
                case "session_info":
                    session_info = entry
                case "custom":
                    custom_entries.append(entry)
                case "compaction":
                    compaction_entries.append(entry)
                    message_rows = _apply_compaction(
                        message_rows,
                        entry,
                        path_before=replay_entries[:entry_index],
                    )
                case "branch_summary":
                    message_rows.append(
                        (entry.id, UserMessage(content=_format_branch_summary(entry)))
                    )

        return cls(
            messages=tuple(message for _entry_id, message in message_rows),
            model=model,
            provider=provider,
            thinking_level=thinking_level,
            labels_by_id=labels_by_id,
            label_timestamps_by_id=label_timestamps_by_id,
            active_leaf_id=active_leaf_id,
            session_info=session_info,
            custom_entries=tuple(custom_entries),
            compaction_entries=tuple(compaction_entries),
            context_entry_ids=tuple(entry_id for entry_id, _message in message_rows),
            entries=tuple(replay_entries),
        )


def _resolve_labels(entries: list[SessionEntry]) -> tuple[dict[str, str], dict[str, float]]:
    # [FUNCTION DOC]
    # HELPER: compute current labels across all entries (not only the active branch) in file
    # order. Each label entry with non-blank text (whitespace-stripped) sets the label and
    # timestamp for its target id; a blank or missing label removes them. Returns a pair of
    # mappings: target id to label text, and target id to timestamp. Pure.
    labels: dict[str, str] = {}
    timestamps: dict[str, float] = {}
    for entry in entries:
        if entry.type != "label":
            continue
        label = entry.label.strip() if entry.label is not None else ""
        if label:
            labels[entry.target_id] = label
            timestamps[entry.target_id] = entry.timestamp
        else:
            labels.pop(entry.target_id, None)
            timestamps.pop(entry.target_id, None)
    return labels, timestamps


def _last_non_leaf_id(entries: list[SessionEntry]) -> str | None:
    # [FUNCTION DOC]
    # HELPER: return the id of the last entry whose type is not "leaf", scanning from the end;
    # None if all entries are leaf records or the list is empty. Pure.
    for entry in reversed(entries):
        if entry.type != "leaf":
            return entry.id
    return None


def _apply_compaction(
    message_rows: list[tuple[str, AgentMessage]],
    entry: CompactionEntry,
    *,
    path_before: list[SessionEntry],
) -> list[tuple[str, AgentMessage]]:
    # [FUNCTION DOC]
    # HELPER: apply one compaction entry to the accumulated context rows (pairs of entry id and
    # message) and return the new row list. The summary row is the compaction entry id with a user
    # message containing the formatted summary.
    # Legacy mode (the entry explicitly set its replaced-entry-ids field, even to empty): drop
    # rows whose ids are in that set, inserting the summary at the position of the first dropped
    # row; if none were dropped, append the summary at the end.
    # Pi mode: find the first-kept entry id within the path before the compaction entry; if
    # absent, return only the summary row. Otherwise return the summary followed by the existing
    # rows whose ids appear on the path from the first-kept entry onward, reordered to path order.
    # Rows not on that path segment are dropped.
    # Inputs: current rows, the compaction entry, keyword path prefix. Output: new list. Pure;
    # raises nothing.
    summary_row = (entry.id, UserMessage(content=_format_compaction_summary(entry.summary)))

    # Tau originally persisted arbitrary replacement-id sets. Explicit legacy
    # fields take precedence, including an empty list, so old sessions retain
    # their exact replay behavior.
    if "replaces_entry_ids" in entry.model_fields_set:
        replaced_ids = set(entry.replaces_entry_ids)
        retained: list[tuple[str, AgentMessage]] = []
        inserted_summary = False
        for entry_id, message in message_rows:
            if entry_id not in replaced_ids:
                retained.append((entry_id, message))
                continue
            if not inserted_summary:
                retained.append(summary_row)
                inserted_summary = True
        if not inserted_summary:
            retained.append(summary_row)
        return retained

    # Pi resolves the boundary against the complete active path, not only
    # message-producing entries. Reorder retained context by that path as well:
    # a previous compaction can itself occur after the kept boundary.
    first_kept_index = next(
        (
            index
            for index, path_entry in enumerate(path_before)
            if path_entry.id == entry.first_kept_entry_id
        ),
        None,
    )
    if first_kept_index is None:
        return [summary_row]

    retained_by_id = dict(message_rows)
    retained = [
        (path_entry.id, retained_by_id[path_entry.id])
        for path_entry in path_before[first_kept_index:]
        if path_entry.id in retained_by_id
    ]
    return [summary_row, *retained]


def _format_compaction_summary(summary: str) -> str:
    # [FUNCTION DOC]
    # HELPER: return the summary text prefixed by a "Previous conversation summary:" line.
    return f"Previous conversation summary:\n{summary}"


def _format_branch_summary(entry: BranchSummaryEntry) -> str:
    # [FUNCTION DOC]
    # HELPER: return text explaining that this is a summary of a branch the conversation came back
    # from, followed by the entry summary wrapped in summary tags on their own lines.
    return (
        "The following is a summary of a branch that this conversation came back from:\n"
        f"<summary>\n{entry.summary}\n</summary>"
    )
