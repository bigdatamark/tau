# [MODULE DOC]
# HELPER: Thin re-export module so provider adapters in tau_ai can import the
# provider contract (the cancellation token and the abstract model provider) from
# tau_ai instead of reaching into tau_agent directly.
# Inputs/outputs: none at runtime beyond exposing the two names in its public list.
# Side effects: importing it imports tau_agent.provider. Raises nothing itself.
"""Public re-exports of the provider contract implemented by Tau adapters."""

from tau_agent.provider import CancellationToken, ModelProvider

__all__ = ["CancellationToken", "ModelProvider"]
