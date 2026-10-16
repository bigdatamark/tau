# [MODULE DOC]
# Durable shell execution settings for Tau terminal commands, persisted as a JSON
# object in "settings.json" inside the Tau home directory. Two settings are
# understood: an optional shell command prefix (camelCase "shellCommandPrefix",
# or legacy snake_case "shell_command_prefix") and the default project trust mode
# ("defaultProjectTrust": ask, always or never; default ask). Unknown keys are
# ignored so newer settings files do not break older Tau versions.
"""Durable shell execution settings for Tau terminal commands."""

from __future__ import annotations

from dataclasses import dataclass
from json import JSONDecodeError, loads
from pathlib import Path
from typing import Any

from tau_coding.paths import TauPaths
from tau_coding.project_trust import TrustDefault


class ShellConfigError(ValueError):
    # [CLASS DOC]
    # Exception type (a ValueError subclass) raised when the settings file is not
    # valid JSON, is not an object, or contains invalid shell settings values.
    """Raised when Tau shell settings are invalid."""


@dataclass(frozen=True, slots=True)
class ShellSettings:
    # [CLASS DOC]
    # Immutable value object holding shell settings: an optional command prefix
    # string (None means no prefix) and the default project trust mode (default
    # "ask").
    """Shell execution settings loaded from Tau home."""

    shell_command_prefix: str | None = None
    default_project_trust: TrustDefault = "ask"

    def to_json(self) -> dict[str, str]:
        # [FUNCTION DOC]
        # Serialize to a JSON-compatible mapping using camelCase keys, omitting defaults:
        # "defaultProjectTrust" only when not "ask", "shellCommandPrefix" only when set.
        # Output: a new mapping of string to string. Raises nothing; no side effects.
        """Serialize these settings to JSON-compatible data."""
        result: dict[str, str] = {}
        if self.default_project_trust != "ask":
            result["defaultProjectTrust"] = self.default_project_trust
        if self.shell_command_prefix is not None:
            result["shellCommandPrefix"] = self.shell_command_prefix
        return result


def shell_settings_path(paths: TauPaths | None = None) -> Path:
    # [FUNCTION DOC]
    # Return the settings file path: the Tau home directory (from the given paths
    # object, or a default one when None) joined with "settings.json".
    # No existence check, no I/O, raises nothing.
    """Return the durable shell settings path."""
    return (paths or TauPaths()).home / "settings.json"


def load_shell_settings(paths: TauPaths | None = None) -> ShellSettings:
    # [FUNCTION DOC]
    # Load settings from disk. Input: optional Tau paths object.
    # Behavior: if the settings file does not exist return default settings;
    # otherwise read it as UTF-8 and parse JSON, then delegate to the JSON parser.
    # Output: a ShellSettings. Raises ShellConfigError for invalid JSON (chained) or
    # a non-object top level, plus any error from parsing; OSError from reading may
    # propagate. Side effect: reads one file.
    """Load durable shell settings, falling back to built-in defaults."""
    path = shell_settings_path(paths)
    if not path.exists():
        return ShellSettings()
    try:
        raw = loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise ShellConfigError(f"Shell settings are not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ShellConfigError("Shell settings must be a JSON object")
    return shell_settings_from_json(raw)


def shell_settings_from_json(data: dict[str, Any]) -> ShellSettings:
    # [FUNCTION DOC]
    # Parse settings from a decoded JSON mapping.
    # Behavior: rejects mappings containing both prefix spellings; validates the
    # trust default (missing means "ask") is one of ask/always/never; reads the
    # prefix from the camelCase key, falling back to snake_case. A missing or null
    # prefix gives no prefix; a non-string prefix is rejected; a string prefix is
    # whitespace-stripped and becomes None if empty.
    # Output: a ShellSettings. Raises ShellConfigError on each invalid case above.
    # No side effects.
    """Parse shell settings from JSON-compatible data."""
    # Read only settings this version understands so fields written by a newer
    # Tau installation cannot prevent an older installation from starting.
    if "shellCommandPrefix" in data and "shell_command_prefix" in data:
        raise ShellConfigError("Use only one of shellCommandPrefix or shell_command_prefix")

    raw_default = data.get("defaultProjectTrust", "ask")
    if raw_default not in {"ask", "always", "never"}:
        raise ShellConfigError("defaultProjectTrust must be ask, always, or never")

    raw_prefix = data.get("shellCommandPrefix", data.get("shell_command_prefix"))
    if raw_prefix is None:
        return ShellSettings(default_project_trust=raw_default)
    if not isinstance(raw_prefix, str):
        raise ShellConfigError("shellCommandPrefix must be a string")
    prefix = raw_prefix.strip()
    return ShellSettings(
        shell_command_prefix=prefix or None,
        default_project_trust=raw_default,
    )
