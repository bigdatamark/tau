# [MODULE DOC]
# HELPER: Utilities for setting the terminal window/tab title from Tau's TUI using OSC 0
# escape sequences. Provides support detection (env/TTY based), sanitisation and length
# capping, title construction with the "τ" mark and a spinner frame while running, and a
# small controller that avoids duplicate writes and disables itself on stream errors.
"""Helpers for updating the terminal window/tab title from Tau's TUI."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable, Mapping
from contextlib import suppress
from typing import TextIO, cast

MAX_TERMINAL_TITLE_LENGTH = 120
OSC_TERMINATOR = "\a"
TAU_TITLE_MARK = "τ"
RUNNING_TITLE_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def terminal_title_supported(
    *,
    environ: Mapping[str, str] | None = None,
    stream: TextIO | None = None,
) -> bool:
    # [FUNCTION DOC]
    # Decide whether OSC title sequences should be emitted.
    # Inputs (keyword-only): optional environment mapping (default process env) and optional
    # stream (default the original stdout). Returns False if TAU_TERMINAL_TITLE is one of
    # "0", "false", "no", "off" (case-insensitive); if the stream is not a TTY (or lacks
    # isatty); if TERM is "dumb"; or if CI is set non-empty unless TAU_TERMINAL_TITLE is "1".
    # Otherwise True. No side effects or exceptions.
    """Return whether Tau should emit OSC title sequences in this process."""
    env = os.environ if environ is None else environ
    if env.get("TAU_TERMINAL_TITLE", "").lower() in {"0", "false", "no", "off"}:
        return False
    target = sys.__stdout__ if stream is None else stream
    if not getattr(target, "isatty", lambda: False)():
        return False
    if env.get("TERM", "") == "dumb":
        return False
    return not (env.get("CI", "") and env.get("TAU_TERMINAL_TITLE", "").lower() != "1")


def sanitize_terminal_title(
    value: str | None,
    *,
    max_length: int = MAX_TERMINAL_TITLE_LENGTH,
) -> str:
    # [FUNCTION DOC]
    # Strip C0/C1 control characters (including escape and bell, which would break OSC)
    # and surrounding whitespace, then cap length.
    # Inputs: the title string or None; keyword max length (default 120).
    # Output: empty string for None; the cleaned string if within the limit; otherwise a
    # truncation to max_length-1 characters, right-stripped, plus an ellipsis character
    # (or a plain slice when max_length is 1 or less). No side effects.
    """Strip OSC-breaking control bytes and cap terminal-title text."""
    if value is None:
        return ""
    sanitized = _CONTROL_CHARS_RE.sub("", value).strip()
    if len(sanitized) <= max_length:
        return sanitized
    if max_length <= 1:
        return sanitized[:max_length]
    return sanitized[: max_length - 1].rstrip() + "…"


def build_terminal_title(
    session_title: str | None,
    *,
    running: bool,
    frame: int = 0,
) -> str:
    # [FUNCTION DOC]
    # Build the tab title for the session and running state.
    # Inputs: session title (or None); keyword running flag; keyword spinner frame index.
    # The sanitised title becomes just "τ" if empty or equal to "untitled session"
    # (case-insensitive), else "τ | <title>". When running, prefixes a braille spinner
    # frame chosen by frame modulo the number of frames, separated by a space.
    # Returns the string; no side effects.
    """Return Tau's terminal tab title for the current session/running state."""
    title = sanitize_terminal_title(session_title)
    title = (
        TAU_TITLE_MARK
        if not title or title.lower() == "untitled session"
        else f"{TAU_TITLE_MARK} | {title}"
    )
    if not running:
        return title
    return f"{RUNNING_TITLE_FRAMES[frame % len(RUNNING_TITLE_FRAMES)]} {title}"


def osc_terminal_title_sequence(title: str) -> str:
    # [FUNCTION DOC]
    # Return the escape sequence ESC ] 0 ; <sanitised title> BEL that sets the terminal
    # window/tab title. Pure function.
    """Return an OSC 0 sequence that sets the terminal window/tab title."""
    return f"\x1b]0;{sanitize_terminal_title(title)}{OSC_TERMINATOR}"


class TerminalTitleController:
    # [CLASS DOC]
    # Small stateful writer for terminal titles. Remembers the last emitted title to skip
    # duplicate writes and turns itself off permanently after a write failure.
    """Small stateful writer that avoids duplicate OSC title writes."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        writer: Callable[[str], object] | None = None,
        stream: TextIO | None = None,
        environ: Mapping[str, str] | None = None,
        exit_title: str = TAU_TITLE_MARK,
    ) -> None:
        # [FUNCTION DOC]
        # Configure the controller. Inputs (keyword-only): explicit enabled flag (None means
        # auto-detect via terminal_title_supported with the given env and stream), optional
        # writer callable taking the sequence string (defaults to writing+flushing the stream),
        # optional stream (defaults to original stdout), optional env mapping, and the title to
        # restore on exit (default "τ"). Sets public enabled attribute; last title starts None.
        self._stream = cast(TextIO, sys.__stdout__) if stream is None else stream
        self.enabled = (
            terminal_title_supported(environ=environ, stream=self._stream)
            if enabled is None
            else enabled
        )
        self._writer = writer or self._default_write
        self._last_title: str | None = None
        self._exit_title = exit_title

    def _write(self, sequence: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: Best-effort write of an escape sequence via the writer.
        # Returns True on success. If the writer raises OSError or ValueError (e.g. closed
        # stream), suppresses it, sets enabled to False and returns False. Other exceptions propagate.
        """Best-effort title write; disable future writes if the stream fails."""
        with suppress(OSError, ValueError):
            self._writer(sequence)
            return True
        self.enabled = False
        return False

    def update(self, session_title: str | None, *, running: bool, frame: int = 0) -> None:
        # [FUNCTION DOC]
        # Emit the title for the given session title / running flag / spinner frame if enabled
        # and different from the last emitted title; records it as last only if the write
        # succeeded. Returns nothing.
        """Write the current Tau title if it differs from the last emitted title."""
        if not self.enabled:
            return
        title = build_terminal_title(session_title, running=running, frame=frame)
        if title == self._last_title:
            return
        if self._write(osc_terminal_title_sequence(title)):
            self._last_title = title

    def restore(self) -> None:
        # [FUNCTION DOC]
        # On shutdown, if enabled, write the configured exit title and record it as last on
        # success. Returns nothing.
        """Leave the terminal title in a neutral idle Tau state on shutdown."""
        if not self.enabled:
            return
        if self._write(osc_terminal_title_sequence(self._exit_title)):
            self._last_title = self._exit_title

    def _default_write(self, sequence: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Default writer: write the sequence to the stored stream and flush it.
        # Stream errors propagate to _write, which handles OSError/ValueError.
        self._stream.write(sequence)
        self._stream.flush()
