# [MODULE DOC]
# Tests for tau_coding.tui.config: location of the TUI settings file under Tau
# home, loading defaults and user overrides, saving, tolerant parsing of unknown
# or removed fields, validation errors (duplicate keys, bad theme, bad enum
# values), theme resolution with fallback, and JSON serialization of keybindings,
# turn notification and sidebar position. Uses pytest tmp_path for file I/O.
# [MODULE DOC]
# Tests for tau_coding.tui.config: location of the TUI settings file under Tau
# home, loading defaults and user overrides, saving, tolerant parsing of unknown
# or removed fields, validation errors (duplicate keys, bad theme, bad enum
# values), theme resolution with fallback, and JSON serialization of keybindings,
# turn notification and sidebar position. Uses pytest tmp_path for file I/O.
from pathlib import Path

import pytest

from tau_coding.paths import TauPaths
from tau_coding.tui.config import (
    HIGH_CONTRAST_THEME,
    TuiConfigError,
    TuiKeybindings,
    TuiSettings,
    get_tui_theme,
    load_tui_settings,
    save_tui_settings,
    tui_settings_from_json,
    tui_settings_path,
)


def test_tui_settings_path_uses_tau_home(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts the TUI settings path is "tui.json" directly under the Tau home dir.
    # [FUNCTION DOC]
    # Asserts the TUI settings path is "tui.json" directly under the Tau home dir.
    paths = TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents")

    assert tui_settings_path(paths) == tmp_path / ".tau" / "tui.json"


def test_load_tui_settings_returns_defaults_when_file_is_missing(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts that with no settings file, loading returns default settings, including
    # defaults ctrl+shift+p for reverse model cycling and ctrl+d for quit.
    # [FUNCTION DOC]
    # Asserts that with no settings file, loading returns default settings, including
    # defaults ctrl+shift+p for reverse model cycling and ctrl+d for quit.
    paths = TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents")

    assert load_tui_settings(paths) == TuiSettings()
    assert load_tui_settings(paths).keybindings.model_cycle_reverse == "ctrl+shift+p"
    assert load_tui_settings(paths).keybindings.quit == "ctrl+d"


def test_load_tui_settings_reads_keybindings(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Writes a settings file overriding many keybindings plus the high-contrast theme
    # and asserts each override is applied, untouched bindings keep defaults
    # (toggle_tool_results ctrl+o, cancel escape), and the theme resolves to the
    # built-in high-contrast theme.
    # [FUNCTION DOC]
    # Writes a settings file overriding many keybindings plus the high-contrast theme
    # and asserts each override is applied, untouched bindings keep defaults
    # (toggle_tool_results ctrl+o, cancel escape), and the theme resolves to the
    # built-in high-contrast theme.
    paths = TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents")
    path = tui_settings_path(paths)
    path.parent.mkdir(parents=True)
    path.write_text(
        """
        {
          "keybindings": {
            "command_palette": "ctrl+j",
            "session_picker": "ctrl+y",
            "queue_follow_up": "f5",
            "insert_newline": "f7",
            "accept_completion": "f2",
            "thinking_cycle": "f3",
            "model_cycle": "f6",
            "model_cycle_reverse": "shift+f6",
            "toggle_thinking": "f4",
            "copy_message": "ctrl+b"
          },
          "theme": "high-contrast"
        }
        """,
        encoding="utf-8",
    )

    settings = load_tui_settings(paths)

    assert settings.keybindings.command_palette == "ctrl+j"
    assert settings.keybindings.session_picker == "ctrl+y"
    assert settings.keybindings.queue_follow_up == "f5"
    assert settings.keybindings.insert_newline == "f7"
    assert settings.keybindings.toggle_tool_results == "ctrl+o"
    assert settings.keybindings.toggle_thinking == "f4"
    assert settings.keybindings.accept_completion == "f2"
    assert settings.keybindings.thinking_cycle == "f3"
    assert settings.keybindings.model_cycle == "f6"
    assert settings.keybindings.model_cycle_reverse == "shift+f6"
    assert settings.keybindings.copy_message == "ctrl+b"
    assert settings.keybindings.cancel == "escape"
    assert settings.theme == "high-contrast"
    assert settings.resolved_theme == HIGH_CONTRAST_THEME


def test_save_tui_settings_writes_json(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Saves settings with the tau-light theme, asserts the returned path is the
    # default tui.json path and that reloading yields the saved theme.
    # [FUNCTION DOC]
    # Saves settings with the tau-light theme, asserts the returned path is the
    # default tui.json path and that reloading yields the saved theme.
    paths = TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents")

    path = save_tui_settings(TuiSettings(theme="tau-light"), paths)

    assert path == tmp_path / ".tau" / "tui.json"
    assert load_tui_settings(paths).theme == "tau-light"


def test_tui_settings_ignores_removed_message_selection_keybindings() -> None:
    # [FUNCTION DOC]
    # Asserts legacy message_previous/message_next keybinding entries are silently
    # ignored, yielding default settings.
    # [FUNCTION DOC]
    # Asserts legacy message_previous/message_next keybinding entries are silently
    # ignored, yielding default settings.
    settings = tui_settings_from_json(
        {
            "keybindings": {
                "message_previous": "alt+up",
                "message_next": "alt+down",
            }
        }
    )

    assert settings == TuiSettings()


def test_tui_settings_ignore_unknown_fields() -> None:
    # [FUNCTION DOC]
    # Asserts unknown top-level settings are ignored while known ones still apply.
    # [FUNCTION DOC]
    # Asserts unknown top-level settings are ignored while known ones still apply.
    settings = tui_settings_from_json(
        {
            "theme": "tau-light",
            "future_setting": {"enabled": True},
        }
    )

    assert settings.theme == "tau-light"


def test_tui_keybindings_ignore_unknown_actions() -> None:
    # [FUNCTION DOC]
    # Asserts unknown keybinding action names are ignored while known ones apply.
    # [FUNCTION DOC]
    # Asserts unknown keybinding action names are ignored while known ones apply.
    settings = tui_settings_from_json(
        {
            "keybindings": {
                "quit": "f12",
                "future_action": "ctrl+g",
            }
        }
    )

    assert settings.keybindings.quit == "f12"


def test_tui_keybindings_reject_duplicate_keys() -> None:
    # [FUNCTION DOC]
    # Asserts binding the same key to two actions raises TuiConfigError mentioning
    # "assigned to both".
    # [FUNCTION DOC]
    # Asserts binding the same key to two actions raises TuiConfigError mentioning
    # "assigned to both".
    with pytest.raises(TuiConfigError, match="assigned to both"):
        tui_settings_from_json(
            {
                "keybindings": {
                    "cancel": "escape",
                    "command_palette": "escape",
                }
            }
        )


def test_tui_settings_accept_unknown_theme_and_fall_back_when_resolving() -> None:
    # [FUNCTION DOC]
    # Asserts an unknown theme name is stored verbatim but resolves to tau-dark.
    # [FUNCTION DOC]
    # Asserts an unknown theme name is stored verbatim but resolves to tau-dark.
    settings = tui_settings_from_json({"theme": "solarized"})

    assert settings.theme == "solarized"
    assert settings.resolved_theme == get_tui_theme("tau-dark")


def test_tui_settings_reject_non_string_theme() -> None:
    # [FUNCTION DOC]
    # Asserts a non-string or blank theme raises TuiConfigError mentioning "theme".
    # [FUNCTION DOC]
    # Asserts a non-string or blank theme raises TuiConfigError mentioning "theme".
    with pytest.raises(TuiConfigError, match="theme"):
        tui_settings_from_json({"theme": 7})
    with pytest.raises(TuiConfigError, match="theme"):
        tui_settings_from_json({"theme": "  "})


def test_tui_settings_accept_light_theme() -> None:
    # [FUNCTION DOC]
    # Asserts tau-light resolves to a white screen background and the ansi_light
    # syntax theme.
    # [FUNCTION DOC]
    # Asserts tau-light resolves to a white screen background and the ansi_light
    # syntax theme.
    settings = tui_settings_from_json({"theme": "tau-light"})

    assert settings.theme == "tau-light"
    assert settings.resolved_theme.screen_background == "#ffffff"
    assert settings.resolved_theme.syntax_theme == "ansi_light"


def test_tui_settings_load_auto_copy_selection() -> None:
    # [FUNCTION DOC]
    # Asserts auto_copy_selection true is loaded and serialized back as true.
    # [FUNCTION DOC]
    # Asserts auto_copy_selection true is loaded and serialized back as true.
    settings = tui_settings_from_json({"auto_copy_selection": True})

    assert settings.auto_copy_selection is True
    assert settings.to_json()["auto_copy_selection"] is True


def test_tui_settings_reject_invalid_auto_copy_selection() -> None:
    # [FUNCTION DOC]
    # Asserts a non-boolean auto_copy_selection raises TuiConfigError.
    # [FUNCTION DOC]
    # Asserts a non-boolean auto_copy_selection raises TuiConfigError.
    with pytest.raises(TuiConfigError, match="auto_copy_selection"):
        tui_settings_from_json({"auto_copy_selection": "yes"})


def test_tui_keybindings_serialize_to_json() -> None:
    # [FUNCTION DOC]
    # Builds settings with custom keybindings and theme and asserts the JSON form
    # contains each custom binding, the default toggle_tool_results binding, the
    # theme, and auto_copy_selection false.
    # [FUNCTION DOC]
    # Builds settings with custom keybindings and theme and asserts the JSON form
    # contains each custom binding, the default toggle_tool_results binding, the
    # theme, and auto_copy_selection false.
    settings = TuiSettings(
        keybindings=TuiKeybindings(
            command_palette="ctrl+j",
            session_picker="ctrl+y",
            queue_follow_up="f5",
            insert_newline="f7",
            accept_completion="f2",
            thinking_cycle="f3",
            model_cycle="f6",
            model_cycle_reverse="shift+f6",
            toggle_thinking="f4",
            copy_message="ctrl+b",
        ),
        theme="high-contrast",
    )

    assert settings.to_json()["keybindings"]["command_palette"] == "ctrl+j"
    assert settings.to_json()["keybindings"]["session_picker"] == "ctrl+y"
    assert settings.to_json()["keybindings"]["queue_follow_up"] == "f5"
    assert settings.to_json()["keybindings"]["insert_newline"] == "f7"
    assert settings.to_json()["keybindings"]["toggle_tool_results"] == "ctrl+o"
    assert settings.to_json()["keybindings"]["toggle_thinking"] == "f4"
    assert settings.to_json()["keybindings"]["accept_completion"] == "f2"
    assert settings.to_json()["keybindings"]["thinking_cycle"] == "f3"
    assert settings.to_json()["keybindings"]["model_cycle"] == "f6"
    assert settings.to_json()["keybindings"]["model_cycle_reverse"] == "shift+f6"
    assert settings.to_json()["keybindings"]["copy_message"] == "ctrl+b"
    assert settings.to_json()["theme"] == "high-contrast"
    assert settings.to_json()["auto_copy_selection"] is False


def test_get_tui_theme_returns_builtin_theme() -> None:
    # [FUNCTION DOC]
    # Asserts known colors for the high-contrast, tau-light and tau-dark themes.
    # [FUNCTION DOC]
    # Asserts known colors for the high-contrast, tau-light and tau-dark themes.
    assert get_tui_theme("high-contrast").prompt_border == "#00ff66"
    assert get_tui_theme("tau-light").prompt_border == "#2563eb"
    assert get_tui_theme("tau-dark").screen_background == "#000000"


def test_tui_turn_notification_defaults_to_desktop() -> None:
    # [FUNCTION DOC]
    # Asserts the default turn notification mode is "desktop" both for default
    # settings and when parsing an empty mapping.
    # [FUNCTION DOC]
    # Asserts the default turn notification mode is "desktop" both for default
    # settings and when parsing an empty mapping.
    assert TuiSettings().turn_notification == "desktop"
    assert tui_settings_from_json({}).turn_notification == "desktop"


def test_tui_turn_notification_roundtrips() -> None:
    # [FUNCTION DOC]
    # Asserts each of off/bell/desktop parses and serializes unchanged.
    # [FUNCTION DOC]
    # Asserts each of off/bell/desktop parses and serializes unchanged.
    for value in ("off", "bell", "desktop"):
        settings = tui_settings_from_json({"turn_notification": value})
        assert settings.turn_notification == value
        assert settings.to_json()["turn_notification"] == value


def test_tui_turn_notification_rejects_invalid_value() -> None:
    # [FUNCTION DOC]
    # Asserts an unknown string or a boolean turn_notification raises TuiConfigError.
    # [FUNCTION DOC]
    # Asserts an unknown string or a boolean turn_notification raises TuiConfigError.
    with pytest.raises(TuiConfigError, match="turn_notification"):
        tui_settings_from_json({"turn_notification": "sound"})

    with pytest.raises(TuiConfigError, match="turn_notification"):
        tui_settings_from_json({"turn_notification": True})


def test_tui_sidebar_position_defaults_to_right() -> None:
    # [FUNCTION DOC]
    # Asserts the default sidebar position is "right".
    # [FUNCTION DOC]
    # Asserts the default sidebar position is "right".
    assert TuiSettings().sidebar_position == "right"
    assert tui_settings_from_json({}).sidebar_position == "right"


def test_tui_sidebar_position_roundtrips() -> None:
    # [FUNCTION DOC]
    # Asserts each of left/right/off parses and serializes unchanged.
    # [FUNCTION DOC]
    # Asserts each of left/right/off parses and serializes unchanged.
    for value in ("left", "right", "off"):
        settings = tui_settings_from_json({"sidebar_position": value})
        assert settings.sidebar_position == value
        assert settings.to_json()["sidebar_position"] == value


def test_tui_sidebar_position_rejects_invalid() -> None:
    # [FUNCTION DOC]
    # Asserts an unknown string or a number sidebar_position raises TuiConfigError.
    # [FUNCTION DOC]
    # Asserts an unknown string or a number sidebar_position raises TuiConfigError.
    with pytest.raises(TuiConfigError, match="sidebar_position"):
        tui_settings_from_json({"sidebar_position": "top"})

    with pytest.raises(TuiConfigError, match="sidebar_position"):
        tui_settings_from_json({"sidebar_position": 123})
