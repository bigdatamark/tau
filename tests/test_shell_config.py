# [MODULE DOC]
# Tests for shell settings loading/parsing in tau_coding: defaults when the settings file is
# missing, acceptance of Pi-style camelCase and Tau-style snake_case shell command prefix keys,
# whitespace trimming, JSON round-trip, and ignoring unknown fields. Uses temporary directories.
from pathlib import Path

from tau_coding import (
    ShellSettings,
    TauPaths,
    load_shell_settings,
    shell_settings_from_json,
    shell_settings_path,
)


def test_load_shell_settings_missing_file_uses_defaults(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: with no shell settings file under a temp Tau home, loading returns default ShellSettings.
    # Input: pytest tmp_path. No side effects beyond temp dir.
    paths = TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents")

    assert load_shell_settings(paths) == ShellSettings()


def test_load_shell_settings_accepts_pi_style_shell_command_prefix(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: writes a settings JSON file with the Pi-style "shellCommandPrefix" key (multi-line value)
    # at the computed settings path, loads it, and asserts the prefix is parsed verbatim and that
    # serialization back to JSON uses the camelCase key. Side effect: creates files in tmp_path.
    paths = TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents")
    path = shell_settings_path(paths)
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"shellCommandPrefix": "shopt -s expand_aliases\\nalias gs=\\"git status\\""}',
        encoding="utf-8",
    )

    settings = load_shell_settings(paths)

    assert settings.shell_command_prefix == 'shopt -s expand_aliases\nalias gs="git status"'
    assert settings.to_json() == {
        "shellCommandPrefix": 'shopt -s expand_aliases\nalias gs="git status"'
    }


def test_shell_settings_accepts_tau_style_shell_command_prefix() -> None:
    # [FUNCTION DOC]
    # Test: parsing a mapping with snake_case "shell_command_prefix" works and strips surrounding
    # whitespace from the value.
    settings = shell_settings_from_json({"shell_command_prefix": " alias ll='ls -la' "})

    assert settings.shell_command_prefix == "alias ll='ls -la'"


def test_shell_settings_ignore_unknown_fields() -> None:
    # [FUNCTION DOC]
    # Test: unknown keys in the settings mapping are ignored while the known prefix is still parsed.
    settings = shell_settings_from_json(
        {
            "shellCommandPrefix": "source ~/.tau/aliases.bash",
            "future_shell_option": True,
        }
    )

    assert settings.shell_command_prefix == "source ~/.tau/aliases.bash"
