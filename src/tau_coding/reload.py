# [MODULE DOC]
# HELPER: small value types describing the outcome of a coding-session resource
# reload (skills, prompt templates, context files, extensions, diagnostics).
# Produced by the reload flow and consumed by CLI/TUI frontends to report what
# changed. Pure frozen dataclasses; no I/O or side effects.
"""Reload summary types for Tau coding-session resources."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReloadCategorySummary:
    # [CLASS DOC]
    # Immutable before/after record for one reload category.
    # Fields: before (count prior to reload), after (count after reload), changed
    # (whether the category content changed; may be true even when counts match).
    # Exposes a derived delta property. No side effects.
    """Before/after state for one reload category."""

    before: int
    after: int
    changed: bool

    @property
    def delta(self) -> int:
        # [FUNCTION DOC]
        # Property: returns after minus before as an integer (may be negative).
        # No exceptions or side effects.
        """Return the count delta for this category."""
        return self.after - self.before


@dataclass(frozen=True, slots=True)
class CodingReloadSummary:
    # [CLASS DOC]
    # Immutable aggregate summary of one coding-resource reload.
    # Fields: one category summary each for skills, prompt_templates,
    # context_files, extensions and diagnostics, plus system_prompt_rebuilt, a
    # boolean telling whether the system prompt was regenerated.
    # No behavior beyond storage.
    """Summary of a local coding-resource reload."""

    skills: ReloadCategorySummary
    prompt_templates: ReloadCategorySummary
    context_files: ReloadCategorySummary
    extensions: ReloadCategorySummary
    diagnostics: ReloadCategorySummary
    system_prompt_rebuilt: bool
