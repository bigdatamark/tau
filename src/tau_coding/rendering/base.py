# [MODULE DOC]
# Shared rendering primitives for Tau's non-interactive coding modes.
# Defines the enumeration of print-mode output formats and the structural
# protocol that every event renderer (text, JSON, transcript, RPC) implements.
# No I/O happens here; it only declares types.
"""Shared event-rendering primitives for Tau coding modes."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from tau_coding.events import CodingSessionEvent


class PrintOutputMode(StrEnum):
    # [CLASS DOC]
    # String enumeration of print-mode output formats: text, json, transcript and
    # rpc. Values equal their names so they can be parsed directly from CLI flags.
    """Output modes supported by non-interactive print mode."""

    text = "text"
    json = "json"
    transcript = "transcript"
    rpc = "rpc"


class EventRenderer(Protocol):
    # [CLASS DOC]
    # Structural protocol for consumers of coding-session events. A renderer is fed
    # events one by one via render and is closed with finish, which reports success.
    """Consumes agent events and renders them for a frontend or output mode."""

    def render(self, event: CodingSessionEvent) -> None:
        # [FUNCTION DOC]
        # Protocol method: render a single coding-session event to the renderer's
        # output (typically writing to stdout/stderr). Input: the event. Returns nothing.
        # Implementations perform output I/O and may accumulate state.
        """Render one event."""

    def finish(self) -> bool:
        # [FUNCTION DOC]
        # Protocol method: flush any pending output and report whether the run was
        # successful (true) or ended in an error (false). Takes no inputs.
        """Finish rendering and return whether the run succeeded."""
