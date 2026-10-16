# [MODULE DOC]
# HELPER: Small utility module that reports Tau's installed version.
# Reads the version of the "tau-ai" distribution from installed package metadata.
# Falls back to the sentinel string "0+unknown" when the package is not installed
# (e.g. running from a raw source checkout). No side effects beyond metadata lookup.
"""Package version helpers."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

_DISTRIBUTION_NAME = "tau-ai"
_UNKNOWN_VERSION = "0+unknown"


def current_version() -> str:
    # [FUNCTION DOC]
    # HELPER: Return the installed version string of the "tau-ai" distribution.
    # Inputs: none. Output: the version string from importlib package metadata.
    # If the distribution metadata cannot be found (PackageNotFoundError), returns
    # "0+unknown" instead of raising. No other exceptions are caught. No side effects.
    """Return Tau's installed package version from package metadata."""
    try:
        return version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return _UNKNOWN_VERSION
