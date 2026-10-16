# [MODULE DOC]
# Detection and normalization of terminal file drag-and-drop for the Textual TUI. Terminals type
# dropped file paths as pasted text (shell-escaped, quoted, file URIs, or bare paths with spaces);
# this module recognizes paste text made solely of existing absolute paths and rewrites it into
# clean, space-separated paths, double-quoting those with whitespace. Only normalize_dropped_paths
# is public.
"""Detect and normalize files dragged into the terminal.

Terminals do not deliver OS drag-and-drop as a dedicated event. When a file is
dropped onto the terminal window, the terminal emulator types the file's path
into the running program instead. Because Textual enables bracketed-paste mode,
that typed path usually arrives as a single :class:`textual.events.Paste`
message.

The exact text depends on the terminal:

- most terminals shell-escape paths (``/tmp/my\\ file.png``) and separate
  multiple dropped files with spaces;
- some quote paths with spaces (``"/tmp/my file.png"``);
- some VTE-based terminals emit ``file://`` URIs;
- a few emit the bare path, even when it contains spaces.

This module recognizes pasted text that consists solely of one or more existing
absolute paths and normalizes it to clean, space-separated filesystem paths,
quoting any path that contains whitespace.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from urllib.parse import unquote, urlparse

__all__ = ["normalize_dropped_paths"]


def normalize_dropped_paths(text: str) -> str | None:
    # [FUNCTION DOC]
    # Decide whether pasted text is a file drop and, if so, return normalized prompt text.
    # Algorithm: strip surrounding whitespace; empty means not a drop. First try the whole
    # stripped text as one path (handles a bare path with unescaped spaces); if it resolves,
    # return it quoted as needed. Otherwise split with POSIX shell-like rules (shell lexing errors
    # such as unbalanced quotes mean not a drop); every token must resolve to an existing absolute
    # path, else not a drop. Return the resolved paths joined by single spaces, each quoted when
    # containing whitespace.
    # Input: pasted text. Output: normalized string, or None when the text is not a drop. Side
    # effects: filesystem existence checks only. Raises nothing.
    """Return normalized prompt text when *text* looks like a file drop.

    The pasted text is treated as a drop only when it consists exclusively of
    one or more absolute paths that exist on disk (shell-escaped, quoted, or
    ``file://`` URI forms are accepted). Anything else returns ``None`` so the
    paste falls through to default handling.
    """
    stripped = text.strip()
    if not stripped:
        return None

    # A single dropped file may arrive as a bare path with unescaped spaces.
    whole = _token_to_path(stripped)
    if whole is not None:
        return _quote_path(whole)

    try:
        tokens = shlex.split(stripped, posix=True)
    except ValueError:
        return None
    if not tokens:
        return None

    paths: list[str] = []
    for token in tokens:
        path = _token_to_path(token)
        if path is None:
            return None
        paths.append(path)
    return " ".join(_quote_path(path) for path in paths)


def _token_to_path(token: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: resolve one token to an existing absolute filesystem path. A "file://" URI is
    # accepted only with an empty or "localhost" host and its path is percent-decoded; other
    # tokens are used verbatim. Returns the path string if it is absolute and exists on disk,
    # otherwise None. Side effect: filesystem stat. Raises nothing.
    """Resolve one dropped token to an existing absolute path, if possible."""
    candidate = token
    if candidate.startswith("file://"):
        parsed = urlparse(candidate)
        if parsed.netloc not in ("", "localhost"):
            return None
        candidate = unquote(parsed.path)
    path = Path(candidate)
    if not path.is_absolute() or not path.exists():
        return None
    return candidate


def _quote_path(path: str) -> str:
    # [FUNCTION DOC]
    # HELPER: return the path unchanged if it has no whitespace; otherwise escape backslashes and
    # double quotes with a backslash and wrap the result in double quotes. Pure function.
    """Quote *path* with double quotes when it contains whitespace."""
    if not any(char.isspace() for char in path):
        return path
    escaped = path.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
