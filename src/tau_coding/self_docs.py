# [MODULE DOC]
# HELPER: Small utility module locating Tau's packaged self-documentation and
# example files that ship inside the tau_coding package under a "data" folder.
# Paths are computed relative to this module's resolved directory; nothing is
# checked for existence and no I/O happens at import time.
"""Locations of Tau's packaged self-documentation and examples."""

from __future__ import annotations

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
_DATA_ROOT = _PACKAGE_ROOT / "data"


def tau_readme_path() -> Path:
    # [FUNCTION DOC]
    # HELPER: Return the path to the packaged overview README (data/docs/README.md).
    # Inputs: none. Output: a filesystem path (not checked for existence).
    # Raises nothing; no side effects.
    """Return the installed overview document for Tau-aware tasks."""
    return _DATA_ROOT / "docs" / "README.md"


def tau_docs_path() -> Path:
    # [FUNCTION DOC]
    # HELPER: Return the packaged self-documentation directory (data/docs).
    # Inputs: none. Output: a directory path (not checked). No side effects.
    """Return the installed Tau self-documentation directory."""
    return _DATA_ROOT / "docs"


def tau_examples_path() -> Path:
    # [FUNCTION DOC]
    # HELPER: Return the packaged examples directory (data/examples).
    # Inputs: none. Output: a directory path (not checked). No side effects.
    """Return the installed Tau example directory."""
    return _DATA_ROOT / "examples"
