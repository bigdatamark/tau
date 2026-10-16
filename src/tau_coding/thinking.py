# [MODULE DOC]
# Thinking-mode (reasoning effort) primitives for Tau coding sessions.
# Defines the ordered set of UI thinking levels (off, minimal, low, medium, high, xhigh, max),
# the default level "medium", short human descriptions per level, the names of provider
# parameters a model may use for thinking, and the OpenAI-style reasoning effort values (where
# "none" replaces "off"). Functions validate/normalize levels, map them to provider settings
# (reasoning effort or Anthropic thinking token budget) and cycle through levels. Pure; no I/O.
"""Thinking-mode primitives for Tau coding sessions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]
ThinkingParameter = Literal["reasoning_effort", "reasoning.effort", "anthropic.thinking"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

THINKING_LEVELS: tuple[ThinkingLevel, ...] = (
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
DEFAULT_THINKING_LEVEL: ThinkingLevel = "medium"

THINKING_LEVEL_DESCRIPTIONS: dict[ThinkingLevel, str] = {
    "off": "No reasoning",
    "minimal": "Very brief reasoning",
    "low": "Light reasoning",
    "medium": "Moderate reasoning",
    "high": "Deep reasoning",
    "xhigh": "Very deep reasoning",
    "max": "Maximum reasoning",
}


def normalize_thinking_level(value: str | None) -> ThinkingLevel:
    # [FUNCTION DOC]
    # Normalize a user-supplied thinking level.
    # Input: string or None. None yields the default level ("medium"). Otherwise the value is
    # trimmed and lowercased and must be one of the known levels.
    # Output: the normalized level string.
    # Raises ValueError ("Unknown thinking mode: ... Available modes: ...") for unknown values.
    """Return a valid Tau thinking level or raise a user-facing error."""
    if value is None:
        return DEFAULT_THINKING_LEVEL
    normalized = value.strip().lower()
    if normalized in THINKING_LEVELS:
        return normalized
    allowed = ", ".join(THINKING_LEVELS)
    raise ValueError(f"Unknown thinking mode: {value}. Available modes: {allowed}")


def normalize_thinking_levels(values: Sequence[str]) -> tuple[ThinkingLevel, ...]:
    # [FUNCTION DOC]
    # Validate a list of thinking levels (e.g. a model's supported modes).
    # Input: a sequence of strings. A bare string or an empty sequence is rejected.
    # Output: a tuple of normalized levels preserving input order.
    # Raises ValueError if the input is a string/empty, if any element is unknown (via the
    # single-level normalizer), or if duplicates exist after normalization.
    """Return a validated, duplicate-free thinking level tuple."""
    if isinstance(values, str) or not values:
        allowed = ", ".join(THINKING_LEVELS)
        raise ValueError(f"Thinking modes must be a non-empty list. Available modes: {allowed}")

    normalized = tuple(normalize_thinking_level(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("Thinking modes must be unique")
    return normalized


def reasoning_effort_for_level(level: str | None) -> ReasoningEffort:
    # [FUNCTION DOC]
    # Map a thinking level to an OpenAI-compatible reasoning effort.
    # Input: level string or None (None means default "medium"). "off" maps to "none"; every other
    # level maps to itself. Output: the effort string. Raises ValueError for unknown levels.
    """Map Tau's UI thinking level to an OpenAI-compatible reasoning effort."""
    normalized = normalize_thinking_level(level)
    if normalized == "off":
        return "none"
    return normalized


def anthropic_thinking_budget_for_level(level: str | None) -> int | None:
    # [FUNCTION DOC]
    # Map a thinking level to an Anthropic extended-thinking token budget.
    # Input: level string or None (normalized; None means "medium").
    # Output: None for "off"; otherwise minimal 1024, low 2048, medium 4096, high 8192,
    # xhigh 16384, max 32768. Raises ValueError for unknown levels.
    """Map Tau's UI thinking level to Anthropic extended-thinking tokens."""
    normalized = normalize_thinking_level(level)
    if normalized == "off":
        return None
    return {
        "minimal": 1024,
        "low": 2048,
        "medium": 4096,
        "high": 8192,
        "xhigh": 16384,
        "max": 32768,
    }[normalized]


def next_thinking_level(
    current: str | None,
    *,
    available: tuple[ThinkingLevel, ...] = THINKING_LEVELS,
) -> ThinkingLevel:
    # [FUNCTION DOC]
    # Return the next level when cycling thinking modes (e.g. a keybinding).
    # Inputs: current level (string or None) and keyword-only tuple of available levels (defaults
    # to all levels). If available is empty, returns the default "medium". If the current value is
    # unknown or not in the available tuple, returns the first available level. Otherwise returns
    # the following entry, wrapping around to the start. Never raises.
    """Return the next thinking level in a stable cycle."""
    if not available:
        return DEFAULT_THINKING_LEVEL
    try:
        normalized_current = normalize_thinking_level(current)
        index = available.index(normalized_current)
    except ValueError:
        return available[0]
    return available[(index + 1) % len(available)]
