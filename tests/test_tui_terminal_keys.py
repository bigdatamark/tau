# [MODULE DOC]
# Tests that raw terminal escape sequences decode (via Textual's XTerm parser)
# to the expected key names, and that the TUI prompt input performs
# word-wise cursor movement/selection for those decoded keys.
"""Exercise actual terminal decoding, not just synthetic pilot key names."""

import asyncio

import pytest
from textual import events
from textual._xterm_parser import XTermParser
from textual.app import App, ComposeResult

from tau_coding.tui.app import PromptInput


@pytest.mark.parametrize(
    ("sequence", "key"),
    [
        ("\x1b[1;3D", "alt+left"),
        ("\x1b[1;3C", "alt+right"),
        ("\x1b[1;4D", "alt+shift+left"),
        ("\x1b[1;4C", "alt+shift+right"),
        ("\x1bb", "ctrl+left"),
        ("\x1bf", "ctrl+right"),
        ("\x1b[98;3u", "alt+b"),
        ("\x1b[102;3u", "alt+f"),
        ("\x1b[1;5D", "ctrl+left"),
        ("\x1b[1;5C", "ctrl+right"),
        ("\x1b[D", "left"),
        ("\x1b[C", "right"),
        ("\x1b", "escape"),
    ],
)
def test_terminal_word_navigation_decoding(sequence: str, key: str) -> None:
    # [FUNCTION DOC]
    # Parametrized over escape sequences (alt/alt+shift arrows, ESC-b/ESC-f,
    # kitty-protocol alt+b/alt+f, ctrl arrows, plain arrows, lone escape). Feeds
    # the sequence plus an empty flush into a fresh parser and asserts exactly one
    # Key event is produced with the expected key name. No side effects.
    parser = XTermParser()
    messages = list(parser.feed(sequence)) + list(parser.feed(""))
    assert len(messages) == 1
    assert isinstance(messages[0], events.Key)
    assert messages[0].key == key


@pytest.mark.parametrize(
    ("left", "right", "select"),
    [
        ("\x1b[1;3D", "\x1b[1;3C", False),
        ("\x1b[1;4D", "\x1b[1;4C", True),
        ("\x1b[98;3u", "\x1b[102;3u", False),
        ("\x1bb", "\x1bf", False),
        ("\x1b[1;5D", "\x1b[1;5C", False),
    ],
)
def test_option_arrows_move_prompt_by_word(left: str, right: str, select: bool) -> None:
    # [FUNCTION DOC]
    # Parametrized over left/right word-motion sequences and whether the motion
    # selects. Runs a minimal Textual app hosting only the prompt input, sets text
    # "one two three" with the cursor at the end, then feeds left, left, right and
    # asserts cursor positions 8, 4, 7. Finally asserts the selected text is
    # " three" when the shift variant was used, else empty.
    # Side effects: runs a headless Textual app inside a new asyncio event loop.
    class EditorApp(App[None]):
        # [CLASS DOC]
        # HELPER: throwaway Textual app whose only widget is the prompt input.
        def compose(self) -> ComposeResult:
            # [FUNCTION DOC]
            # HELPER: yields a single prompt input widget.
            yield PromptInput()

    async def run() -> None:
        # [FUNCTION DOC]
        # HELPER: async body of the test; drives the app in test mode, posting each
        # parsed key message and pausing the pilot before asserting cursor position.
        # Raises AssertionError on mismatch.
        app = EditorApp()
        async with app.run_test() as pilot:
            prompt = app.query_one(PromptInput)
            prompt.text = "one two three"
            prompt.cursor_position = len(prompt.text)
            prompt.focus()
            for sequence, position in [(left, 8), (left, 4), (right, 7)]:
                messages = list(XTermParser().feed(sequence))
                assert len(messages) == 1
                app.post_message(messages[0])
                await pilot.pause()
                assert prompt.cursor_position == position
            assert prompt.selected_text == (" three" if select else "")

    asyncio.run(run())
