# [MODULE DOC]
# Contracts for optional runtime model-catalog discovery by provider adapters (tau_ai layer).
# Defines the allowed input modalities ("text", "image") and thinking levels
# ("off", "minimal", "low", "medium", "high", "xhigh", "max"), an immutable record
# describing one advertised model, an immutable catalog snapshot of models, and a
# runtime-checkable protocol that providers implement if they can list the models
# available to the authenticated account. Pure data/contract module; no I/O.
"""Optional runtime model-catalog discovery contracts for provider adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from tau_ai.model_limits import RuntimeModelLimits

RuntimeInputModality = Literal["text", "image"]
RuntimeThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True, slots=True)
class RuntimeModel:
    # [CLASS DOC]
    # Immutable, slotted record describing one model advertised by a provider catalog.
    # Fields: id (required non-empty string), optional display name, optional runtime
    # limits (context window, output tokens, etc.), a tuple of input modalities (defaults
    # to text only), a tuple of supported thinking levels (default empty) and an optional
    # default thinking level. Validation happens in post-init.
    """One model advertised by an authenticated provider catalog."""

    id: str
    name: str | None = None
    limits: RuntimeModelLimits | None = None
    input_modalities: tuple[RuntimeInputModality, ...] = ("text",)
    thinking_levels: tuple[RuntimeThinkingLevel, ...] = ()
    default_thinking_level: RuntimeThinkingLevel | None = None

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate the record after construction.
        # Raises ValueError if the id is empty, if input_modalities is empty, or if a default
        # thinking level is set but is not one of the advertised thinking_levels.
        # Values of modalities/levels are not otherwise checked against the allowed literals.
        # Returns nothing; no side effects.
        if not self.id:
            raise ValueError("model id must be non-empty")
        if not self.input_modalities:
            raise ValueError("input_modalities must be non-empty")
        if self.default_thinking_level is not None and (
            self.default_thinking_level not in self.thinking_levels
        ):
            raise ValueError("default_thinking_level must be advertised in thinking_levels")


@dataclass(frozen=True, slots=True)
class RuntimeModelCatalog:
    # [CLASS DOC]
    # Immutable, slotted snapshot of every model a provider advertises for the current
    # account, stored as a tuple of RuntimeModel records in provider order.
    """Complete account-specific model snapshot from a provider."""

    models: tuple[RuntimeModel, ...]

    def model(self, model_id: str) -> RuntimeModel | None:
        # [FUNCTION DOC]
        # Look up one model by exact id.
        # Input: model id string. Output: the first RuntimeModel whose id equals it, or None
        # when absent. Linear scan; case-sensitive; no exceptions or side effects.
        """Return one advertised model by exact ID."""
        return next((model for model in self.models if model.id == model_id), None)


@runtime_checkable
class ModelCatalogProvider(Protocol):
    # [CLASS DOC]
    # Runtime-checkable protocol for providers that support authenticated model discovery.
    # A provider object satisfies it if it has an async discover_models method; isinstance
    # checks only verify method presence, not signatures.
    """Optional provider capability for authenticated model discovery."""

    async def discover_models(self) -> RuntimeModelCatalog:
        # [FUNCTION DOC]
        # Protocol method: asynchronously return the complete RuntimeModelCatalog available to
        # the active account. Implementations typically perform network I/O and may raise
        # provider-specific errors. The protocol body itself does nothing.
        """Return the complete model inventory available to the active account."""
        ...
