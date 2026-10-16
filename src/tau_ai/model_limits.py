# [MODULE DOC]
# Optional runtime model-limit discovery contracts for provider adapters.
# Defines an immutable value type describing limits a serving surface reports
# for a model, plus a runtime-checkable protocol that providers may implement
# to discover those limits live. Used by context/compaction logic.
"""Optional runtime model-limit discovery contracts for provider adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RuntimeModelLimits:
    # [CLASS DOC]
    # Frozen, slotted dataclass of provider-reported limits for one model:
    # context window (tokens, required), optional max output tokens, an effective
    # context window percentage (headroom requested by provider, default 100), and
    # an optional explicit auto-compaction token limit.
    # Validation happens at construction (see __post_init__).
    """Provider-reported limits for one model on the active serving surface."""

    context_window: int
    max_output_tokens: int | None = None
    effective_context_window_percent: int = 100
    auto_compact_token_limit: int | None = None

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate fields after dataclass init. Raises ValueError if the context
        # window is not positive, max output tokens is set but not positive, the
        # effective percent is outside 1..100, or the auto-compact limit is set but not
        # positive. Returns nothing; no mutation.
        if self.context_window <= 0:
            raise ValueError("context_window must be positive")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if not 1 <= self.effective_context_window_percent <= 100:
            raise ValueError("effective_context_window_percent must be between 1 and 100")
        if self.auto_compact_token_limit is not None and self.auto_compact_token_limit <= 0:
            raise ValueError("auto_compact_token_limit must be positive")

    @property
    def effective_context_window(self) -> int:
        # [FUNCTION DOC]
        # Property: usable window = context window times the effective percent,
        # integer-divided by 100, clamped to at least 1. No side effects.
        """Return the provider's usable window after its requested headroom."""
        return max(1, self.context_window * self.effective_context_window_percent // 100)

    @property
    def effective_auto_compact_token_limit(self) -> int:
        # [FUNCTION DOC]
        # Property: token count at which auto-compaction should trigger. If no explicit
        # limit is set, uses 90% of the raw context window (min 1); otherwise uses the
        # explicit limit. In both cases the result is capped at the effective context
        # window. No side effects.
        """Return an explicit limit or the Codex-compatible 90% default."""
        default_limit = max(1, self.context_window * 9 // 10)
        if self.auto_compact_token_limit is None:
            return min(default_limit, self.effective_context_window)
        return min(self.auto_compact_token_limit, self.effective_context_window)


@runtime_checkable
class ModelLimitsProvider(Protocol):
    # [CLASS DOC]
    # Runtime-checkable protocol for providers that can report live model limits;
    # callers use isinstance checks to detect the optional capability.
    """Optional provider capability for serving-surface-specific model limits."""

    async def discover_model_limits(self, model: str) -> RuntimeModelLimits | None:
        # [FUNCTION DOC]
        # Protocol coroutine: look up limits for the given model id on the active
        # serving surface. Output: RuntimeModelLimits, or None when the model is not
        # advertised. Implementations typically perform network I/O.
        """Return live limits for ``model``, or ``None`` when it is not advertised."""
        ...
