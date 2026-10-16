# [MODULE DOC]
# HELPER: Shared pytest fixtures applied to the whole test suite.
# Guards against two accidental side effects: opening a real web browser, and reading
# or writing the developer's real home configuration (~/.tau). Both fixtures are autouse.
import webbrowser
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def prevent_browser_open(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # HELPER: Autouse fixture that replaces the standard-library browser opener with a stub.
    # Inputs: pytest monkeypatch. Output: none.
    # Side effects: monkeypatches webbrowser.open so any call fails the current test
    # (pytest.fail) with a message containing the URL. Restored automatically after the test.
    """Fail tests that accidentally launch an external browser."""

    def fail_browser_open(url: str, *_args: object, **_kwargs: object) -> bool:
        # [FUNCTION DOC]
        # HELPER: Stub browser opener; accepts a URL plus arbitrary extra arguments and always
        # fails the test via pytest.fail (never returns despite the bool return annotation).
        pytest.fail(f"Test attempted to open a browser URL: {url}")

    monkeypatch.setattr(webbrowser, "open", fail_browser_open)


def isolate_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # HELPER: Redirect home-directory lookups to a temporary directory for a test.
    # Inputs: pytest monkeypatch and a temp path. Output: none.
    # Side effects: removes the TAU_HOME environment variable if set, and sets both HOME
    # and USERPROFILE to the temp path (USERPROFILE matters on Windows for Path.home()).
    # Importable by tests that need to re-isolate explicitly.
    """Point home-directory lookups at the pytest temp directory."""
    monkeypatch.delenv("TAU_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    # Path.home() resolves via USERPROFILE on Windows, so HOME alone does not
    # isolate tests from the developer's real ~/.tau settings.
    monkeypatch.setenv("USERPROFILE", str(tmp_path))


@pytest.fixture(autouse=True)
def prevent_user_home_writes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # HELPER: Autouse fixture that applies isolate_home to every test using pytest's
    # tmp_path, so no test touches the real ~/.tau settings. Returns nothing.
    """Keep every test away from the developer's real home configuration."""
    isolate_home(monkeypatch, tmp_path)
