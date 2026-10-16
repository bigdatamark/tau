# [MODULE DOC]
# Textual TUI tests for the OAuth login screen: the full authorization URL is
# rendered and linked intact and copied to the clipboard, device-code flows do
# not touch the clipboard, the dialog fits a short terminal by scrolling, and a
# blank answer is accepted for prompts that allow empty input.
# Runs headless Textual apps with fake login coroutines; no network I/O.
import asyncio

import pytest
from textual.app import App
from textual.geometry import Region
from textual.widgets import Input, Static

from tau_coding.oauth_types import OAuthAuthInfo, OAuthDeviceCodeInfo, OAuthPrompt
from tau_coding.provider_catalog import builtin_provider_entry
from tau_coding.tui.app import OAuthLoginScreen, TauTuiApp
from tau_coding.tui.config import TAU_DARK_THEME

LONG_URL = "https://claude.ai/oauth/authorize?" + "&".join(f"p{index}=value" for index in range(40))


def _login_screen(url: str = LONG_URL) -> OAuthLoginScreen:
    # [FUNCTION DOC]
    # HELPER: builds an OAuth login screen for the built-in anthropic provider
    # whose fake login function reports the given authorization URL (default: a
    # long synthetic URL) through the on_auth callback and then waits forever.
    # Output: the screen. Raises AssertionError if the provider entry is missing.
    """Build an Anthropic login screen whose flow stops after showing `url`."""
    provider = builtin_provider_entry("anthropic")
    assert provider is not None

    async def fake_login(callbacks):
        # [FUNCTION DOC]
        # HELPER: fake login flow; calls on_auth with the URL and instructions, then
        # blocks on an event that is never set.
        callbacks.on_auth(OAuthAuthInfo(url=url, instructions="Complete login in your browser."))
        await asyncio.Event().wait()

    return OAuthLoginScreen(provider, theme=TAU_DARK_THEME, login=fake_login)


def _themed_app(screen: OAuthLoginScreen) -> App[None]:
    # [FUNCTION DOC]
    # HELPER: builds a test app using the production TUI CSS and the registered
    # dark Tau theme, which pushes the given screen when mounted, so geometry
    # matches production. Output: the app instance (not running).
    """An app carrying Tau's real CSS, so dialog geometry matches production."""
    from tau_coding.tui.app import _textual_theme_for_tau_theme

    class TestApp(App[None]):
        # [CLASS DOC]
        # HELPER: minimal app class carrying Tau's CSS and theme.
        CSS = TauTuiApp.CSS

        def __init__(self) -> None:
            # [FUNCTION DOC]
            # HELPER: initializes the app, registers the Textual theme derived from the
            # dark Tau theme and activates it. Side effect: mutates app theme registry.
            super().__init__()
            self.register_theme(_textual_theme_for_tau_theme(TAU_DARK_THEME.name))
            self.theme = TAU_DARK_THEME.name

        def on_mount(self) -> None:
            # [FUNCTION DOC]
            # HELPER: pushes the captured login screen on mount.
            self.push_screen(screen)

    return TestApp()


@pytest.mark.anyio
async def test_oauth_screen_shows_full_authorization_url() -> None:
    # [FUNCTION DOC]
    # Renders the login screen with a URL over 400 characters at 100x40, replaces
    # the clipboard method with a list append, then asserts the URL appears intact
    # in the rendered text (ignoring wrapping spaces), every link segment points to
    # the exact URL, and the URL was copied to the clipboard exactly once.
    """The whole URL must be visible: users copy it out of the TUI by hand."""
    url = LONG_URL
    assert len(url) > 400
    screen = _login_screen(url)

    copied: list[str] = []
    app = _themed_app(screen)
    app.copy_to_clipboard = copied.append  # type: ignore[method-assign]
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        await pilot.pause()
        widget = screen.query_one("#login-oauth-url", Static)
        lines = widget.render_lines(Region(0, 0, widget.size.width, widget.size.height))
        rendered = "".join("".join(segment.text for segment in line) for line in lines)
        links = {
            segment.style.link
            for line in lines
            for segment in line
            if segment.style is not None and segment.style.link
        }

    assert url in rendered.replace(" ", "")
    # Every wrapped line links to the intact URL, and it is on the clipboard, so
    # the user never has to reassemble it from the wrapped display by hand.
    assert links == {url}
    assert copied == [url]


@pytest.mark.anyio
async def test_oauth_device_code_screen_leaves_the_clipboard_alone() -> None:
    # [FUNCTION DOC]
    # Uses the github-copilot provider with a fake login that reports a device
    # code; asserts nothing was copied to the clipboard and the help text shows
    # the user code.
    """The device flow's URI is short and clickable; the code is what matters."""
    provider = builtin_provider_entry("github-copilot")
    assert provider is not None

    async def fake_login(callbacks):
        # [FUNCTION DOC]
        # HELPER: fake device flow; calls on_device_code with a fixed code and
        # verification URI, then waits forever.
        callbacks.on_device_code(
            OAuthDeviceCodeInfo(
                user_code="ABCD-1234",
                verification_uri="https://github.com/login/device",
            )
        )
        await asyncio.Event().wait()

    screen = OAuthLoginScreen(provider, theme=TAU_DARK_THEME, login=fake_login)
    copied: list[str] = []
    app = _themed_app(screen)
    app.copy_to_clipboard = copied.append  # type: ignore[method-assign]
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        await pilot.pause()
        help_text = str(screen.query_one("#login-help", Static).render())

    assert copied == []
    assert "ABCD-1234" in help_text


@pytest.mark.anyio
async def test_oauth_screen_fits_a_short_terminal() -> None:
    # [FUNCTION DOC]
    # Renders the login screen at height 14 and asserts the dialog container lies
    # within the screen, is vertically scrollable, and the code input field is
    # fully on screen.
    """Growing for the URL must not push the paste field off a small screen."""
    height = 14
    screen = _login_screen()
    async with _themed_app(screen).run_test(size=(100, height)) as pilot:
        await pilot.pause()
        await pilot.pause()
        container = screen.query_one("#login-screen")
        dialog = container.region
        code = screen.query_one("#login-oauth-code", Input).region
        scrollable = container.allow_vertical_scroll

    # The dialog scrolls its own overflow rather than centering it, which would
    # hang the title off the top and the paste field off the bottom. What is
    # below the fold stays reachable by scrolling; the focused paste field is
    # scrolled into view for us.
    assert dialog.y >= 0
    assert dialog.bottom <= height
    assert scrollable
    assert code.y >= 0
    assert code.bottom <= height


@pytest.mark.anyio
async def test_oauth_screen_accepts_blank_provider_prompt() -> None:
    # [FUNCTION DOC]
    # Pushes a login screen (no login function) into a minimal app, starts the
    # private prompt-for-code coroutine with an allow-empty prompt, submits an
    # empty input with enter, and asserts the result is the empty string and the
    # help text shows the prompt message.
    # Note: calls the screen's compose method directly before mounting, which only
    # creates a discarded generator.
    provider = builtin_provider_entry("github-copilot")
    assert provider is not None
    screen = OAuthLoginScreen(provider, theme=TAU_DARK_THEME)
    screen.compose()

    # Exercise the prompt/input handshake inside a minimal Textual app context.
    from textual.app import App

    class TestApp(App[None]):
        # [CLASS DOC]
        # HELPER: minimal app that pushes the screen on mount.
        def on_mount(self) -> None:
            # [FUNCTION DOC]
            # HELPER: pushes the captured login screen.
            self.push_screen(screen)

    app = TestApp()
    async with app.run_test() as pilot:
        prompt_task = asyncio.create_task(
            screen._prompt_for_code(OAuthPrompt(message="Enterprise domain", allow_empty=True))
        )
        await pilot.pause()
        screen.query_one("#login-oauth-code", Input).value = ""
        await pilot.press("enter")
        await pilot.pause()

        assert await prompt_task == ""
        assert str(screen.query_one("#login-help", Static).render()) == "Enterprise domain"
