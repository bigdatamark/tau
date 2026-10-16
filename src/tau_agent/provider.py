# [MODULE DOC]
# Defines the provider contract owned by the portable agent layer (tau_agent).
# The agent loop talks to models only through these structural protocols, so
# any tau_ai adapter (or a fake test provider) can be plugged in without the
# core depending on a concrete SDK. Mirrors Pi's provider-neutral stream API.
# Contains no runtime logic; only typing protocols.
"""Provider contract owned by Tau's portable agent layer."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from tau_agent.messages import AgentMessage
from tau_agent.provider_events import AssistantMessageEvent
from tau_agent.tools import AgentTool


class CancellationToken(Protocol):
    # [CLASS DOC]
    # Structural protocol for a cooperative cancellation signal passed to a
    # provider stream. Any object exposing an is_cancelled method qualifies
    # (e.g. the loop's abort signal). No state is defined here.
    def is_cancelled(self) -> bool:
        # [FUNCTION DOC]
        # Protocol method: report whether the in-flight stream should stop.
        # Inputs: none. Output: True when cancellation was requested.
        # Implementations should be cheap and side-effect free; providers poll it.
        """Return whether the current stream should stop."""
        ...


class ModelProvider(Protocol):
    # [CLASS DOC]
    # Structural protocol every model provider adapter must satisfy: a single
    # streaming entry point producing assistant message events for one turn.
    """Provider-neutral Pi-compatible model stream interface."""

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        # [FUNCTION DOC]
        # Protocol method: stream one model response.
        # Inputs (keyword-only): model id string, system prompt string, the
        # conversation as a list of agent messages, the list of available agent tools,
        # an optional cancellation token, and an optional session id that providers
        # may use for routing / prompt-cache affinity (ignored if unsupported).
        # Output: an async iterator of assistant message events (start, deltas, done or
        # error). Exceptions and side effects (network I/O) are implementation-defined.
        """Stream one model response as assistant message events.

        Providers may use ``session_id`` for request routing or prompt-cache
        affinity. Unsupported providers ignore it.
        """
        ...
