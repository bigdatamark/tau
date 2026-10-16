# [MODULE DOC]
# Tests for TUI terminal notifications: OSC 9 / OSC 99 escape sequence generation and
# sanitization, terminal protocol detection from environment variables, interactive-terminal
# support detection, and the notification controller's write, no-op and failure-disable behavior.
from __future__ import annotations

from io import StringIO

from tau_coding.tui.terminal_notification import (
    TerminalNotificationController,
    desktop_notification_protocol,
    desktop_notification_sequence,
    osc9_notification_sequence,
    osc99_notification_sequence,
    terminal_notification_supported,
)


class TtyStringIO(StringIO):
    # [CLASS DOC]
    # HELPER: in-memory text stream that reports itself as a TTY, used to simulate an interactive
    # terminal.
    def isatty(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: always return true.
        return True


class PipeStringIO(StringIO):
    # [CLASS DOC]
    # HELPER: in-memory text stream that reports itself as not a TTY, used to simulate piped
    # output.
    def isatty(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: always return false.
        return False


def test_notification_sequences_sanitize_control_bytes() -> None:
    # [FUNCTION DOC]
    # Verify OSC 9 and OSC 99 builders strip control characters (bell, newline) from the message
    # and wrap it in the right escape prefix and terminator (bell for OSC 9, string terminator for
    # OSC 99).
    assert osc9_notification_sequence("finished\x07\n") == "\x1b]9;finished\x07"
    assert osc99_notification_sequence("finished\x07\n") == "\x1b]99;;finished\x1b\\"


def test_desktop_notification_protocol_detects_supported_terminals() -> None:
    # [FUNCTION DOC]
    # Verify protocol detection: ghostty, iTerm and mintty map to osc9; kitty (by TERM or
    # KITTY_WINDOW_ID) maps to osc99; Apple Terminal yields None.
    assert desktop_notification_protocol(environ={"TERM_PROGRAM": "ghostty"}) == "osc9"
    assert desktop_notification_protocol(environ={"TERM_PROGRAM": "iTerm.app"}) == "osc9"
    assert desktop_notification_protocol(environ={"TERM_PROGRAM": "mintty"}) == "osc9"
    assert desktop_notification_protocol(environ={"TERM": "xterm-kitty"}) == "osc99"
    assert desktop_notification_protocol(environ={"KITTY_WINDOW_ID": "1"}) == "osc99"
    assert desktop_notification_protocol(environ={"TERM_PROGRAM": "Apple_Terminal"}) is None


def test_desktop_notification_sequence_uses_terminal_protocol() -> None:
    # [FUNCTION DOC]
    # Verify the desktop sequence builder picks the OSC 9 or OSC 99 form by environment and
    # returns None for unsupported terminals.
    assert (
        desktop_notification_sequence("Tau turn finished", environ={"TERM_PROGRAM": "ghostty"})
        == "\x1b]9;Tau turn finished\x07"
    )
    assert (
        desktop_notification_sequence("Tau turn finished", environ={"TERM": "xterm-kitty"})
        == "\x1b]99;;Tau turn finished\x1b\\"
    )
    assert (
        desktop_notification_sequence(
            "Tau turn finished", environ={"TERM_PROGRAM": "Apple_Terminal"}
        )
        is None
    )


def test_terminal_notification_supported_requires_interactive_terminal() -> None:
    # [FUNCTION DOC]
    # Verify support requires a TTY stream, a TERM other than "dumb", and no CI environment
    # variable.
    assert terminal_notification_supported(environ={"TERM": "xterm-256color"}, stream=TtyStringIO())
    assert not terminal_notification_supported(
        environ={"TERM": "xterm-256color"}, stream=PipeStringIO()
    )
    assert not terminal_notification_supported(environ={"TERM": "dumb"}, stream=TtyStringIO())
    assert not terminal_notification_supported(
        environ={"TERM": "xterm-256color", "CI": "1"}, stream=TtyStringIO()
    )


def test_terminal_notification_controller_writes_configured_sequence() -> None:
    # [FUNCTION DOC]
    # Verify a bell-mode controller writes the bell character and a desktop-mode controller under
    # ghostty writes the OSC 9 "Tau turn finished" sequence, each through a capturing writer.
    bell_writes: list[str] = []
    desktop_writes: list[str] = []

    TerminalNotificationController(
        "bell", enabled=True, writer=bell_writes.append
    ).notify_turn_finished()
    TerminalNotificationController(
        "desktop",
        enabled=True,
        writer=desktop_writes.append,
        environ={"TERM_PROGRAM": "ghostty"},
    ).notify_turn_finished()

    assert bell_writes == ["\a"]
    assert desktop_writes == ["\x1b]9;Tau turn finished\x07"]


def test_terminal_notification_controller_noops_when_off_disabled_or_unknown() -> None:
    # [FUNCTION DOC]
    # Verify nothing is written when mode is "off", when disabled, or when desktop mode runs in an
    # unsupported terminal.
    writes: list[str] = []

    TerminalNotificationController("off", enabled=True, writer=writes.append).notify_turn_finished()
    TerminalNotificationController(
        "bell", enabled=False, writer=writes.append
    ).notify_turn_finished()
    TerminalNotificationController(
        "desktop",
        enabled=True,
        writer=writes.append,
        environ={"TERM_PROGRAM": "Apple_Terminal"},
    ).notify_turn_finished()

    assert writes == []


def test_terminal_notification_controller_disables_after_write_failure() -> None:
    # [FUNCTION DOC]
    # Verify that when the writer raises an OS error, the controller swallows it, disables itself,
    # and does not call the writer again on the next notification.
    calls = 0

    def failing_writer(sequence: str) -> None:
        # [FUNCTION DOC]
        # HELPER: fake writer that discards the sequence, counts the call in the enclosing
        # counter, and raises an OS error simulating a vanished terminal.
        nonlocal calls
        del sequence
        calls += 1
        raise OSError("terminal is gone")

    controller = TerminalNotificationController(
        "desktop",
        enabled=True,
        writer=failing_writer,
        environ={"TERM_PROGRAM": "ghostty"},
    )

    controller.notify_turn_finished()
    controller.notify_turn_finished()

    assert calls == 1
    assert controller.enabled is False
