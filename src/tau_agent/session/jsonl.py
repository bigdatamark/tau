# [MODULE DOC]
# JSONL persistence boundary for session entries: serializes entries one per
# line and deserializes them, migrating older persisted formats (Tau-v1
# messages, legacy target-less labels, camelCase custom-message fields) into
# the current canonical entry models before pydantic validation. Migration is
# intentionally confined here so runtime models stay strict.
# Pure transformation of strings and decoded JSON; no file I/O itself.
"""JSONL serialization and Tau-v1 persisted-session migration."""

from __future__ import annotations

import json
from typing import Any

from pydantic import TypeAdapter, ValidationError

from tau_agent.session.entries import SessionEntry

_SESSION_ENTRY_ADAPTER: TypeAdapter[SessionEntry] = TypeAdapter(SessionEntry)


class SessionJsonlError(ValueError):
    # [CLASS DOC]
    # Exception type (a ValueError subclass) raised when a JSONL line cannot be
    # decoded or validated as a session entry; messages include the line number
    # when known.
    """Raised when a session JSONL line cannot be decoded."""


def entry_to_json_line(entry: SessionEntry) -> str:
    # [FUNCTION DOC]
    # Serializes one session entry to compact JSON via the shared entry type
    # adapter, omitting fields whose value is None, and appends a newline.
    # Input: a session entry. Output: one newline-terminated JSON line.
    # No side effects.
    """Serialize one entry in Tau's canonical persisted shape."""
    return _SESSION_ENTRY_ADAPTER.dump_json(entry, exclude_none=True).decode() + "\n"


def entry_from_json_line(line: str, *, line_number: int | None = None) -> SessionEntry:
    # [FUNCTION DOC]
    # Parses a single JSON line, migrates it (with no legacy label target, so
    # target-less labels fall back to their parent id or own id), and validates
    # it into a session entry.
    # Inputs: the line text and optional keyword line number used in errors.
    # Output: the validated entry.
    # Raises SessionJsonlError wrapping JSON decode, validation, type or value
    # errors. No side effects.
    # KNOWN ISSUE: unlike the batch function, a single-line decode cannot pick the
    # earliest branchable entry as a legacy label target, so the same legacy label
    # may migrate to a different target depending on which function is used.
    """Deserialize one entry, migrating persisted Tau-v1 messages first."""
    location = f" on line {line_number}" if line_number is not None else ""
    try:
        payload = json.loads(line)
        migrated = _migrate_session_entry(payload, legacy_label_target_id=None)
        return _SESSION_ENTRY_ADAPTER.validate_python(migrated)
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        raise SessionJsonlError(f"Invalid session entry{location}: {exc}") from exc


def entries_from_json_lines(lines: list[str]) -> list[SessionEntry]:
    # [FUNCTION DOC]
    # Decodes a list of JSONL lines in order, skipping blank lines. First parses
    # all JSON (so the legacy label target can be chosen across the whole file),
    # computes one legacy label target id, then migrates and validates each value.
    # Input: list of line strings. Output: list of entries in file order.
    # Raises SessionJsonlError naming the 1-based line number on JSON decode,
    # validation, type or value errors. No side effects.
    """Deserialize non-empty JSONL lines in order.

    Legacy session-level labels had no target. They deterministically become a
    bookmark on the earliest branchable entry, preserving the old value without
    conflating bookmarks with ``SessionInfoEntry.title``. If the transcript has
    no branchable entry, the earliest ordinary entry is used instead.
    """
    decoded: list[tuple[int, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            decoded.append((index, json.loads(line)))
        except json.JSONDecodeError as exc:
            raise SessionJsonlError(f"Invalid session entry on line {index}: {exc}") from exc

    legacy_target_id = _legacy_label_target_id([value for _line, value in decoded])
    entries: list[SessionEntry] = []
    for line_number, value in decoded:
        try:
            migrated = _migrate_session_entry(value, legacy_label_target_id=legacy_target_id)
            entries.append(_SESSION_ENTRY_ADAPTER.validate_python(migrated))
        except (ValidationError, TypeError, ValueError) as exc:
            raise SessionJsonlError(f"Invalid session entry on line {line_number}: {exc}") from exc
    return entries


def _migrate_session_entry(value: Any, *, legacy_label_target_id: str | None) -> Any:
    # [FUNCTION DOC]
    # HELPER: returns a canonical shallow copy of one decoded entry value.
    # Non-mapping values pass through unchanged. Rules:
    # - label entries lacking target_id get one: the supplied legacy target, else
    #   the entry's parent_id, else its own id.
    # - custom_message entries: camelCase customType is renamed to custom_type
    #   (existing custom_type wins).
    # - other non-message entries pass through unchanged.
    # - message entries have their inner message migrated; if the result has the
    #   custom role, the entry is converted to a custom_message entry: millisecond
    #   numeric message timestamp (not boolean) becomes seconds on the entry,
    #   custom_type/content/display (default true)/details are lifted from the
    #   message, and the message field is removed.
    # Inputs: decoded value and keyword legacy label target id. Output: migrated
    # value. Does not mutate the input mapping (copies), though nested values are
    # shared.
    """Return a canonical copy of one decoded persisted entry.

    The extension API may break in lockstep, but user session history must not.
    Migration is intentionally confined to this persistence boundary so runtime
    models and extension-facing constructors retain one strict protocol.
    """
    if not isinstance(value, dict):
        return value
    if value.get("type") == "label" and "target_id" not in value:
        migrated = dict(value)
        migrated["target_id"] = legacy_label_target_id or value.get("parent_id") or value.get("id")
        return migrated
    if value.get("type") == "custom_message":
        migrated = dict(value)
        if "customType" in migrated:
            migrated.setdefault("custom_type", migrated["customType"])
            migrated.pop("customType")
        return migrated
    if value.get("type") != "message":
        return value

    migrated = dict(value)
    message = _migrate_message(value.get("message"))
    if isinstance(message, dict) and message.get("role") == "custom":
        message_timestamp = message.get("timestamp")
        if isinstance(message_timestamp, int | float) and not isinstance(message_timestamp, bool):
            migrated["timestamp"] = message_timestamp / 1000
        migrated["type"] = "custom_message"
        migrated["custom_type"] = message.get("customType", message.get("custom_type"))
        migrated["content"] = message.get("content")
        migrated["display"] = message.get("display", True)
        migrated["details"] = message.get("details")
        migrated.pop("message", None)
        return migrated

    migrated["message"] = message
    return migrated


def _legacy_label_target_id(values: list[Any]) -> str | None:
    # [FUNCTION DOC]
    # HELPER: picks a deterministic target for legacy labels: the id of the first
    # message, compaction or branch_summary entry; failing that, the id of the
    # first entry that is not a label or leaf; failing that, none. Only string ids
    # count. Input: list of decoded values. Output: id string or none.
    """Choose one stable target for target-less pre-bookmark label records."""
    branchable_types = {"message", "compaction", "branch_summary"}
    for value in values:
        if isinstance(value, dict) and value.get("type") in branchable_types:
            entry_id = value.get("id")
            if isinstance(entry_id, str):
                return entry_id
    for value in values:
        if isinstance(value, dict) and value.get("type") not in {"label", "leaf"}:
            entry_id = value.get("id")
            if isinstance(entry_id, str):
                return entry_id
    return None


def _migrate_message(value: Any) -> Any:
    # [FUNCTION DOC]
    # HELPER: migrates one persisted message mapping from Tau-v1 to the current
    # shape, returning a copy (non-mapping values pass through). Rules by role:
    # - user with custom_type/customType: becomes role custom with customType set
    #   and display defaulting to true.
    # - assistant: a usage mapping with missing/null cost gets an empty cost
    #   mapping; string content becomes a list with one text block (if non-empty)
    #   followed by any tool_calls/toolCalls blocks; list content gets any separate
    #   tool calls appended; the separate tool-call keys are removed.
    # - tool: becomes toolResult; name to toolName (default "unknown"),
    #   tool_call_id to toolCallId (default empty), ok to isError (inverted,
    #   default ok), string content to a text block list (empty list when blank),
    #   legacy data merged under details (details win) or used as details, and a
    #   legacy error string becomes the content when content is empty.
    # - anything else: returned as a copy unchanged.
    # No exceptions raised deliberately; no side effects.
    # KNOWN ISSUE: when both tool_calls and toolCalls are present, the inner pop of
    # toolCalls always runs and its value is silently discarded in favor of
    # tool_calls.
    if not isinstance(value, dict):
        return value
    message = dict(value)
    role = message.get("role")

    if role == "user" and ("custom_type" in message or "customType" in message):
        message["role"] = "custom"
        message["customType"] = message.pop("custom_type", message.get("customType"))
        message.pop("custom_type", None)
        message.setdefault("display", True)
        return message

    if role == "assistant":
        usage = message.get("usage")
        if isinstance(usage, dict) and usage.get("cost") is None:
            usage = dict(usage)
            usage["cost"] = {}
            message["usage"] = usage

        content = message.get("content", "")
        if isinstance(content, str):
            blocks: list[Any] = []
            if content:
                blocks.append({"type": "text", "text": content})
            blocks.extend(message.pop("tool_calls", message.pop("toolCalls", [])) or [])
            message["content"] = blocks
        elif "tool_calls" in message or "toolCalls" in message:
            blocks = list(content or [])
            blocks.extend(message.pop("tool_calls", message.pop("toolCalls", [])) or [])
            message["content"] = blocks
        return message

    if role == "tool":
        message["role"] = "toolResult"
        message["toolName"] = message.pop("name", message.get("toolName", "unknown"))
        message["toolCallId"] = message.pop("tool_call_id", message.get("toolCallId", ""))
        message["isError"] = not bool(message.pop("ok", True))
        content = message.get("content", "")
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}] if content else []
        data = message.pop("data", None)
        details = message.get("details")
        if isinstance(data, dict) and isinstance(details, dict):
            message["details"] = {**data, **details}
        elif details is None and data is not None:
            message["details"] = data
        error = message.pop("error", None)
        if error and not message["content"]:
            message["content"] = [{"type": "text", "text": str(error)}]
        return message

    return message
