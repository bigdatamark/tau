# [MODULE DOC]
# Tests for TauPaths (tau_coding.paths): user-level and project-level resource directories,
# default session file path layout, and TAU_HOME environment handling (default, empty value,
# per-instance reading, tilde expansion, absolute-path requirement, explicit override).
# Uses pytest tmp_path and monkeypatch for environment variables.
from pathlib import Path

import pytest

from tau_coding.paths import TauPaths


def test_tau_paths_user_locations(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: user locations derive from home and agents_home (sessions, skills, prompts under the Tau
    # home; skills and prompts under the agents home).
    paths = TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents")

    assert paths.sessions_dir == tmp_path / ".tau" / "sessions"
    assert paths.user_skills_dir == tmp_path / ".tau" / "skills"
    assert paths.user_prompts_dir == tmp_path / ".tau" / "prompts"
    assert paths.user_agents_skills_dir == tmp_path / ".agents" / "skills"
    assert paths.user_agents_prompts_dir == tmp_path / ".agents" / "prompts"


def test_tau_paths_project_locations(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: project locations for a cwd are its .tau and .agents dirs plus their skills/prompts subdirs.
    paths = TauPaths(home=tmp_path / "home", agents_home=tmp_path / "agents")
    cwd = tmp_path / "project"

    assert paths.project_tau_dir(cwd) == cwd / ".tau"
    assert paths.project_agents_dir(cwd) == cwd / ".agents"
    assert paths.project_skills_dir(cwd) == cwd / ".tau" / "skills"
    assert paths.project_prompts_dir(cwd) == cwd / ".tau" / "prompts"
    assert paths.project_agents_skills_dir(cwd) == cwd / ".agents" / "skills"
    assert paths.project_agents_prompts_dir(cwd) == cwd / ".agents" / "prompts"


def test_default_session_path_uses_home_sessions_and_readable_project_path(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC]
    # Test: default session path is "default.jsonl" in a per-project directory under home/sessions
    # whose name contains the dash-joined cwd segments and ends in a 6-character suffix; the
    # directory is created. Side effect: creates directories in tmp_path.
    paths = TauPaths(home=tmp_path / "home", agents_home=tmp_path / "agents")
    cwd = tmp_path / "repos" / "exploration" / "tau"
    cwd.mkdir(parents=True)

    session_path = paths.default_session_path(cwd)

    assert session_path.name == "default.jsonl"
    assert session_path.parent.parent == tmp_path / "home" / "sessions"
    assert "repos-exploration-tau-" in session_path.parent.name
    assert len(session_path.parent.name.rsplit("-", maxsplit=1)[-1]) == 6
    assert session_path.parent.exists()


def test_tau_home_defaults_to_dot_tau(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test: with TAU_HOME unset, home defaults to the user's home directory plus ".tau".
    monkeypatch.delenv("TAU_HOME", raising=False)

    assert TauPaths().home == Path.home() / ".tau"


def test_empty_tau_home_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test: an empty TAU_HOME value is treated as unset (default ~/.tau).
    monkeypatch.setenv("TAU_HOME", "")

    assert TauPaths().home == Path.home() / ".tau"


def test_tau_home_environment_override_is_read_for_each_instance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC]
    # Test: TAU_HOME is read at construction time per instance, so two instances made under different
    # env values have different homes and derived paths (sessions, extensions, models store, llama.cpp
    # extension state); agents_home stays ~/.agents for both.
    first_home = tmp_path / "tau-first"
    second_home = tmp_path / "tau-second"

    monkeypatch.setenv("TAU_HOME", str(first_home))
    first = TauPaths()
    monkeypatch.setenv("TAU_HOME", str(second_home))
    second = TauPaths()

    assert first.home == first_home
    assert first.sessions_dir == first_home / "sessions"
    assert first.user_extensions_dir == first_home / "extensions"
    assert second.home == second_home
    assert second.models_store_path == second_home / "models-store.json"
    assert second.llama_cpp_state_path == second_home / "state" / "extensions" / "llama.cpp.json"
    assert first.agents_home == second.agents_home == Path.home() / ".agents"


def test_tau_home_expands_current_user(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test: a TAU_HOME starting with "~" is expanded against the current user's home directory.
    monkeypatch.setenv("TAU_HOME", "~/.tau-personal")

    assert TauPaths().home == Path.home() / ".tau-personal"


def test_tau_home_must_be_absolute(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test: a relative TAU_HOME causes construction to raise ValueError ("TAU_HOME must be an absolute path").
    monkeypatch.setenv("TAU_HOME", ".tau-personal")

    with pytest.raises(ValueError, match="TAU_HOME must be an absolute path"):
        TauPaths()


def test_explicit_tau_home_ignores_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC]
    # Test: an explicitly passed home takes precedence over the TAU_HOME environment variable.
    explicit_home = tmp_path / "explicit"
    monkeypatch.setenv("TAU_HOME", str(tmp_path / "environment"))

    assert TauPaths(home=explicit_home).home == explicit_home
