# [MODULE DOC]
# Tests for tau_coding.tui.terminal_title: building the terminal window title
# (Greek tau, optional session name, spinner frame while running), sanitizing
# titles, formatting the OSC 0 escape sequence, detecting terminal support from
# the environment and stream, and the stateful title controller (dedupe, restore,
# disable-on-failure). Uses in-memory streams; no real terminal I/O.
from __future__ import annotations

from io import StringIO

from tau_coding.tui.terminal_title import (
    MAX_TERMINAL_TITLE_LENGTH,
    TerminalTitleController,
    build_terminal_title,
    osc_terminal_title_sequence,
    sanitize_terminal_title,
    terminal_title_supported,
)


class TtyStringIO(StringIO):
    # [CLASS DOC]
    # HELPER: In-memory text stream that reports itself as a TTY, used to simulate
    # an interactive terminal.
    def isatty(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: Always returns True.
        return True


class PipeStringIO(StringIO):
    # [CLASS DOC]
    # HELPER: In-memory text stream that reports itself as not a TTY (a pipe).
    def isatty(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: Always returns False.
        return False


def test_build_terminal_title_uses_session_name_and_running_frame() -> None:
    # [FUNCTION DOC]
    # Asserts the idle title is "τ | <name>" and the running title prefixes the
    # braille spinner glyph for the given frame index (frame 1 is "⠙").
    assert build_terminal_title("build notes", running=False) == "τ | build notes"
    assert build_terminal_title("build notes", running=True, frame=1) == "⠙ τ | build notes"


def test_build_terminal_title_falls_back_for_unnamed_sessions() -> None:
    # [FUNCTION DOC]
    # Asserts a missing name, or the placeholder "Untitled session" (whitespace
    # trimmed), yields just "τ" (with the frame-0 spinner "⠋" when running).
    assert build_terminal_title(None, running=False) == "τ"
    assert build_terminal_title(" Untitled session ", running=True) == "⠋ τ"


def test_sanitize_terminal_title_strips_control_bytes_and_caps_length() -> None:
    # [FUNCTION DOC]
    # Feeds a string with ESC, BEL, newline and an over-long tail and asserts control
    # characters are removed and the result is truncated to the maximum title length
    # ending with an ellipsis.
    malicious = "\x1b]0;bad\x07\n" + ("x" * MAX_TERMINAL_TITLE_LENGTH)

    sanitized = sanitize_terminal_title(malicious)

    assert "\x1b" not in sanitized
    assert "\x07" not in sanitized
    assert "\n" not in sanitized
    assert len(sanitized) == MAX_TERMINAL_TITLE_LENGTH
    assert sanitized.endswith("…")


def test_osc_terminal_title_sequence_sanitizes_payload() -> None:
    # [FUNCTION DOC]
    # Asserts the OSC sequence is ESC ] 0 ; title BEL with the BEL inside the title
    # stripped.
    assert osc_terminal_title_sequence("hello\x07") == "\x1b]0;hello\x07"


def test_terminal_title_supported_requires_tty_and_allows_opt_out() -> None:
    # [FUNCTION DOC]
    # Asserts support requires a TTY stream, is not affected by NO_COLOR, and is
    # disabled by TAU_TERMINAL_TITLE=0 or TERM=dumb.
    assert terminal_title_supported(environ={"TERM": "xterm-256color"}, stream=TtyStringIO())
    assert terminal_title_supported(
        environ={"TERM": "xterm-256color", "NO_COLOR": "1"},
        stream=TtyStringIO(),
    )
    assert not terminal_title_supported(environ={"TERM": "xterm-256color"}, stream=PipeStringIO())
    assert not terminal_title_supported(
        environ={"TERM": "xterm-256color", "TAU_TERMINAL_TITLE": "0"},
        stream=TtyStringIO(),
    )
    assert not terminal_title_supported(environ={"TERM": "dumb"}, stream=TtyStringIO())


def test_terminal_title_controller_writes_running_idle_and_restore_titles() -> None:
    # [FUNCTION DOC]
    # Drives an enabled controller with a recording writer and asserts duplicate
    # identical updates are written once, a running update with frame 2 writes the
    # "⠹" spinner title, and restore writes the bare "τ" title.
    writes: list[str] = []
    controller = TerminalTitleController(enabled=True, writer=writes.append)

    controller.update("build notes", running=False)
    controller.update("build notes", running=False)
    controller.update("build notes", running=True, frame=2)
    controller.restore()

    assert writes == [
        "\x1b]0;τ | build notes\x07",
        "\x1b]0;⠹ τ | build notes\x07",
        "\x1b]0;τ\x07",
    ]


def test_terminal_title_controller_noops_when_disabled() -> None:
    # [FUNCTION DOC]
    # Asserts a disabled controller never calls its writer for update or restore.
    writes: list[str] = []
    controller = TerminalTitleController(enabled=False, writer=writes.append)

    controller.update("build notes", running=True)
    controller.restore()

    assert writes == []


def test_terminal_title_controller_disables_itself_after_write_failure() -> None:
    # [FUNCTION DOC]
    # Uses a writer that raises OSError and asserts the controller swallows the
    # error, calls the writer only once, and sets its enabled flag to False.
    calls = 0

    def failing_writer(sequence: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Fake writer that increments the enclosing call counter and then raises
        # OSError to simulate a vanished terminal.
        nonlocal calls
        calls += 1
        raise OSError("terminal is gone")

    controller = TerminalTitleController(enabled=True, writer=failing_writer)

    controller.update("build notes", running=False)
    controller.update("other", running=False)

    assert calls == 1
    assert controller.enabled is False
