# [MODULE DOC]
# Tests for tau_coding.thinking: normalization of thinking-level names, cycling
# through levels, validation of level lists, and mapping levels to provider
# reasoning-effort strings. Pure unit tests with no I/O.
import pytest

from tau_coding.thinking import (
    DEFAULT_THINKING_LEVEL,
    THINKING_LEVELS,
    next_thinking_level,
    normalize_thinking_level,
    normalize_thinking_levels,
    reasoning_effort_for_level,
)


def test_normalize_thinking_level_accepts_supported_modes() -> None:
    # [FUNCTION DOC]
    # Verifies normalization lowercases a known level ("HIGH" becomes "high") and
    # that a missing value falls back to the default thinking level.
    assert normalize_thinking_level("HIGH") == "high"
    assert normalize_thinking_level(None) == DEFAULT_THINKING_LEVEL


def test_normalize_thinking_level_rejects_unknown_mode() -> None:
    # [FUNCTION DOC]
    # Verifies an unknown level name raises ValueError mentioning
    # "Unknown thinking mode".
    with pytest.raises(ValueError, match="Unknown thinking mode"):
        normalize_thinking_level("maximum")


def test_next_thinking_level_cycles_supported_modes() -> None:
    # [FUNCTION DOC]
    # Verifies cycling order (medium to high, xhigh to max, max wraps to off), that
    # an unknown current level with a restricted available list yields that list's
    # first entry, and that the full ordered level tuple is off..max.
    assert next_thinking_level("medium") == "high"
    assert next_thinking_level("xhigh") == "max"
    assert next_thinking_level("max") == "off"
    assert next_thinking_level("missing", available=("low", "high")) == "low"
    assert THINKING_LEVELS == ("off", "minimal", "low", "medium", "high", "xhigh", "max")


def test_normalize_thinking_levels_rejects_empty_and_duplicates() -> None:
    # [FUNCTION DOC]
    # Verifies list normalization lowercases entries, raises ValueError ("non-empty")
    # for an empty list, and ValueError ("unique") for case-insensitive duplicates.
    assert normalize_thinking_levels(["OFF", "high"]) == ("off", "high")

    with pytest.raises(ValueError, match="non-empty"):
        normalize_thinking_levels([])

    with pytest.raises(ValueError, match="unique"):
        normalize_thinking_levels(["high", "HIGH"])


def test_reasoning_effort_maps_off_to_none() -> None:
    # [FUNCTION DOC]
    # Verifies the "off" level maps to reasoning effort "none" and other levels such
    # as "xhigh" pass through unchanged.
    assert reasoning_effort_for_level("off") == "none"
    assert reasoning_effort_for_level("xhigh") == "xhigh"
