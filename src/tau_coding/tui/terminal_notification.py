# [MODULE DOC]
# Best-effort terminal attention notifications emitted by the TUI when an agent
# turn finishes. Supports a plain bell and desktop notifications via OSC 9
# (Ghostty, iTerm2, mintty) or Kitty's OSC 99. Detection is based on environment
# variables and whether the output is a TTY. Lives in the TUI layer so the core
# agent has no terminal dependencies.
"""Best-effort terminal attention notifications for completed Tau turns."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from contextlib import suppress
from typing import Literal, TextIO, cast

from tau_coding.tui.config import TurnNotificationMode
from tau_coding.tui.terminal_title import sanitize_terminal_title

OSC_TERMINATOR = "\a"
TURN_FINISHED_MESSAGE = "Tau turn finished"
type DesktopNotificationProtocol = Literal["osc9", "osc99"]


def terminal_notification_supported(
    *,
    environ: Mapping[str, str] | None = None,
    stream: TextIO | None = None,
) -> bool:
    # [FUNCTION DOC]
    # Decide whether attention sequences may be written.
    # Inputs: optional environment mapping (default process environment) and
    # optional stream (default the original standard output).
    # Output: False if the stream lacks isatty or is not a TTY, if TERM is "dumb", or
    # if CI is set to a non-empty value; True otherwise. Raises nothing; no writes.
    """Return whether Tau may write attention sequences to this terminal."""
    env = os.environ if environ is None else environ
    target = sys.__stdout__ if stream is None else stream
    if not getattr(target, "isatty", lambda: False)():
        return False
    if env.get("TERM", "") == "dumb":
        return False
    return not bool(env.get("CI", ""))


def desktop_notification_protocol(
    *, environ: Mapping[str, str] | None = None
) -> DesktopNotificationProtocol | None:
    # [FUNCTION DOC]
    # Pick a desktop notification protocol from environment variables (default the
    # process environment). Returns "osc99" for Kitty (KITTY_WINDOW_ID set, TERM
    # xterm-kitty, or TERM_PROGRAM kitty), "osc9" for TERM_PROGRAM ghostty, iterm.app,
    # iterm2 or mintty (case-insensitive) or when MINTTY_SHORTCUT is set, else None.
    # Raises nothing; no side effects.
    """Select the notification protocol explicitly supported by this terminal."""
    env = os.environ if environ is None else environ
    term = env.get("TERM", "").lower()
    term_program = env.get("TERM_PROGRAM", "").lower()
    if env.get("KITTY_WINDOW_ID") or term == "xterm-kitty" or term_program == "kitty":
        return "osc99"
    if term_program in {"ghostty", "iterm.app", "iterm2", "mintty"} or env.get("MINTTY_SHORTCUT"):
        return "osc9"
    return None


def osc9_notification_sequence(message: str) -> str:
    # [FUNCTION DOC]
    # Build an OSC 9 notification: ESC ] 9 ; message BEL, where the message is first
    # sanitized by the terminal-title sanitizer (removes control characters).
    # Pure; returns the string.
    """Build a sanitized OSC 9 desktop-notification sequence."""
    return f"\x1b]9;{sanitize_terminal_title(message)}{OSC_TERMINATOR}"


def osc99_notification_sequence(message: str) -> str:
    # [FUNCTION DOC]
    # Build a Kitty OSC 99 notification: ESC ] 99 ; ; message ESC backslash, with the
    # message sanitized as for titles. Pure; returns the string.
    """Build a sanitized Kitty OSC 99 desktop-notification sequence."""
    return f"\x1b]99;;{sanitize_terminal_title(message)}\x1b\\"


def desktop_notification_sequence(
    message: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    # [FUNCTION DOC]
    # Return the notification sequence for the detected protocol (OSC 99 or OSC 9)
    # for the given message and environment, or None when no protocol is supported.
    # Pure; raises nothing.
    """Build the desktop-notification sequence appropriate for this terminal."""
    protocol = desktop_notification_protocol(environ=environ)
    if protocol == "osc99":
        return osc99_notification_sequence(message)
    if protocol == "osc9":
        return osc9_notification_sequence(message)
    return None


class TerminalNotificationController:
    # [CLASS DOC]
    # Stateful writer of turn-finished notifications for the TUI. Holds the
    # configured mode ("off", "bell", or desktop), the environment, the output
    # stream, an enabled flag and a writer callable.
    """Write a configured terminal notification without affecting core agent code."""

    def __init__(
        self,
        mode: TurnNotificationMode,
        *,
        enabled: bool | None = None,
        writer: Callable[[str], object] | None = None,
        stream: TextIO | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Inputs: notification mode; optional explicit enabled flag (default: detect via
        # terminal_notification_supported); optional writer callable (default writes and
        # flushes the stream); optional stream (default original stdout); optional
        # environment mapping (default process environment).
        # Side effects: stores state only; no output.
        self.mode = mode
        self._environ = os.environ if environ is None else environ
        self._stream = cast(TextIO, sys.__stdout__) if stream is None else stream
        self.enabled = (
            terminal_notification_supported(environ=environ, stream=self._stream)
            if enabled is None
            else enabled
        )
        self._writer = writer or self._default_write

    def notify_turn_finished(self) -> None:
        # [FUNCTION DOC]
        # Emit an attention signal for a finished turn. Does nothing when disabled or
        # mode is "off". Mode "bell" writes a BEL character; any other mode writes the
        # desktop sequence for "Tau turn finished", or nothing if the terminal has no
        # supported protocol. If the writer raises OSError or ValueError, the error is
        # swallowed and the controller disables itself permanently.
        # Output: None. Side effects: terminal write, may mutate the enabled flag.
        """Request attention for a completed turn, if notifications are enabled."""
        if not self.enabled or self.mode == "off":
            return
        sequence = (
            "\a"
            if self.mode == "bell"
            else desktop_notification_sequence(
                TURN_FINISHED_MESSAGE,
                environ=self._environ,
            )
        )
        if sequence is None:
            return
        with suppress(OSError, ValueError):
            self._writer(sequence)
            return
        self.enabled = False

    def _default_write(self, sequence: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Write the sequence to the configured stream and flush it.
        # May raise OSError/ValueError from the stream (handled by the caller).
        self._stream.write(sequence)
        self._stream.flush()
