# [MODULE DOC]
# Structured diagnostic logging for coding-session agent-call failures. Writes
# one JSON object per line (JSONL, sorted keys) to Tau's agent-calls log file for
# unexpected exceptions, terminal assistant error messages and Hugging Face route
# failovers. Deliberately records only non-secret context and scalar error
# classification fields (never request payloads or credentials).
"""Structured diagnostic logging for coding-session failures."""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from tau_agent.messages import AssistantMessage
from tau_coding.paths import TauPaths


@dataclass(frozen=True, slots=True)
class AgentCallDiagnosticContext:
    # [CLASS DOC]
    # Immutable record of non-secret context stamped on every log entry: provider
    # name, model id, working directory, optional session id and the run id of the
    # agent call.
    """Non-secret context attached to an agent-call diagnostic entry."""

    provider_name: str
    model: str
    cwd: Path
    session_id: str | None
    run_id: str


class AgentCallDiagnosticLogger:
    # [CLASS DOC]
    # Appends JSONL diagnostic entries to a fixed file path. Each log method builds a
    # base entry (UTC ISO timestamp, kind, phase, run/session ids, provider, model,
    # cwd as string), adds a kind-specific payload, appends it and returns the path.
    """Append structured JSONL diagnostics for agent-call failures."""

    def __init__(self, path: Path) -> None:
        # [FUNCTION DOC]
        # Store the target log file path. No I/O happens until something is logged.
        self.path = path

    @classmethod
    def from_paths(cls, paths: TauPaths | None = None) -> AgentCallDiagnosticLogger:
        # [FUNCTION DOC]
        # Alternate constructor: build a logger for the agent-calls log path of the given
        # path layout, or of a default layout when none is given. No I/O.
        """Create a logger using Tau's default path layout."""
        return cls((paths or TauPaths()).agent_calls_log_path)

    def log_exception(
        self,
        *,
        context: AgentCallDiagnosticContext,
        phase: str,
        exc: BaseException,
    ) -> Path:
        # [FUNCTION DOC]
        # Log an unexpected exception. Inputs (keyword-only): context, phase label and
        # the exception. Adds an "exception" object with type name, message string and
        # the full formatted traceback text; entry kind is "exception".
        # Outputs: the log file path. Side effects: creates parent dirs and appends a
        # line. Raises OSError if the file cannot be written.
        """Log an unexpected exception with traceback and return the log path."""
        entry = _base_entry(context, phase=phase, kind="exception")
        entry["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }
        self._append(entry)
        return self.path

    def log_assistant_error(
        self,
        *,
        context: AgentCallDiagnosticContext,
        phase: str,
        message: AssistantMessage,
    ) -> Path:
        # [FUNCTION DOC]
        # Log a terminal assistant error message. Inputs (keyword-only): context, phase
        # and the assistant message. Adds an "error" object with the message's error text
        # (defaulting to "Error") and stop reason, plus a "provider" sub-object of safe
        # provider details when any are found in the message diagnostics; kind is
        # "assistant_error". Outputs: log path. Side effects/raises: as above.
        """Log a terminal assistant error message with safe diagnostic details."""
        entry = _base_entry(context, phase=phase, kind="assistant_error")
        error: dict[str, Any] = {
            "message": message.error_message or "Error",
            "stop_reason": message.stop_reason,
        }
        provider = _provider_error_details(message)
        if provider:
            error["provider"] = provider
        entry["error"] = error
        self._append(entry)
        return self.path

    def log_huggingface_route_failover(
        self,
        *,
        context: AgentCallDiagnosticContext,
        failed_route: str,
        replacement_route: str | None,
        success: bool,
        error_message: str | None,
    ) -> Path:
        # [FUNCTION DOC]
        # Log one completed automatic Hugging Face inference-route failover. Inputs
        # (keyword-only): context, failed route, replacement route (may be None), success
        # flag and optional error message. Phase is fixed to "agent_loop_route_failover"
        # and kind to "route_failover"; payload has from/to/success and "error" only when
        # an error message is given. Outputs: log path. Side effects/raises: as above.
        """Log one completed automatic Hugging Face route failover."""
        entry = _base_entry(context, phase="agent_loop_route_failover", kind="route_failover")
        entry["route_failover"] = {
            "from": failed_route,
            "to": replacement_route,
            "success": success,
            **({"error": error_message} if error_message is not None else {}),
        }
        self._append(entry)
        return self.path

    def _append(self, entry: dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # HELPER: Create the log file's parent directories if needed, then open the file
        # in append mode (UTF-8) and write the entry as sorted-key JSON plus a newline.
        # Raises OSError on filesystem errors and TypeError for non-JSON-serializable
        # values.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, sort_keys=True) + "\n")


def new_agent_call_run_id() -> str:
    # [FUNCTION DOC]
    # Return a new random 32-hex-character id identifying one agent call (despite the
    # docstring's "stable", it is random per call).
    """Return a stable id for one coding-session agent call."""
    return uuid4().hex


_SAFE_ERROR_OBJECT_KEYS = ("type", "code", "message", "param")


def _provider_error_details(message: AssistantMessage) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Find the first diagnostic of type "provider_error" with non-empty
    # details on the assistant message and extract safe fields: integer status_code
    # and attempts (booleans excluded), and a sanitized "event" sub-object from a
    # mapping-valued raw stream event. Returns that mapping (possibly empty) for the
    # first such diagnostic, or an empty mapping if none exists. No side effects.
    """Extract non-secret provider failure details from message diagnostics.

    Provider adapters attach the raw stream event to assistant diagnostics, but
    that payload can be large. Only scalar classification fields (status codes,
    attempt counts, and error type/code/message) are copied into the log so the
    entry stays small and free of request or credential material.
    """
    for diagnostic in message.diagnostics or []:
        if diagnostic.type != "provider_error" or not diagnostic.details:
            continue
        details: dict[str, Any] = {}
        status_code = diagnostic.details.get("status_code")
        if isinstance(status_code, int) and not isinstance(status_code, bool):
            details["status_code"] = status_code
        attempts = diagnostic.details.get("attempts")
        if isinstance(attempts, int) and not isinstance(attempts, bool):
            details["attempts"] = attempts
        event = diagnostic.details.get("event")
        if isinstance(event, dict):
            event_details = _safe_stream_event_details(event)
            if event_details:
                details["event"] = event_details
        return details
    return {}


def _safe_stream_event_details(event: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: From a raw provider stream error event mapping, keep only: non-empty
    # string type, integer sequence_number (booleans excluded), a sanitized nested
    # "error" object, and a sanitized "response_error" taken from a mapping-valued
    # response's error. Returns the (possibly empty) filtered mapping.
    """Keep only non-secret scalar fields from a provider stream error event."""
    details: dict[str, Any] = {}
    event_type = event.get("type")
    if isinstance(event_type, str) and event_type:
        details["type"] = event_type
    sequence_number = event.get("sequence_number")
    if isinstance(sequence_number, int) and not isinstance(sequence_number, bool):
        details["sequence_number"] = sequence_number
    nested = _safe_error_object(event.get("error"))
    if nested:
        details["error"] = nested
    response = event.get("response")
    if isinstance(response, dict):
        response_error = _safe_error_object(response.get("error"))
        if response_error:
            details["response_error"] = response_error
    return details


def _safe_error_object(value: object) -> dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: If the value is a mapping, return a new mapping containing only the
    # keys type, code, message and param whose values are non-empty strings;
    # otherwise return an empty mapping.
    """Copy scalar classification fields from a provider error object."""
    if not isinstance(value, dict):
        return {}
    return {
        key: field
        for key in _SAFE_ERROR_OBJECT_KEYS
        if isinstance((field := value.get(key)), str) and field
    }


def _base_entry(
    context: AgentCallDiagnosticContext,
    *,
    phase: str,
    kind: str,
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Build the common log entry fields from the context plus keyword phase
    # and kind: current UTC timestamp in ISO format, kind, phase, run_id, session_id,
    # provider_name, model and cwd as a string. Returns a new mutable mapping.
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "kind": kind,
        "phase": phase,
        "run_id": context.run_id,
        "session_id": context.session_id,
        "provider_name": context.provider_name,
        "model": context.model,
        "cwd": str(context.cwd),
    }
