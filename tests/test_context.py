# [MODULE DOC]
# Tests for project context-file discovery (AGENTS.md files) in tau_coding.
# Verifies discovery order across user Tau home, user agents home, project
# root, nested cwd and the cwd's .tau and .agents directories.
from pathlib import Path

from tau_coding.context import discover_project_context
from tau_coding.paths import TauPaths
from tau_coding.resources import TauResourcePaths


def test_discovers_user_project_and_agents_context_files(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Builds a temporary layout: a Tau home and agents home each with AGENTS.md, a
    # project root (marked by pyproject.toml) with AGENTS.md, a nested package dir
    # with AGENTS.md plus .tau/AGENTS.md and .agents/AGENTS.md. Calls discovery
    # with the nested dir as cwd and asserts the six files are returned in order:
    # user Tau, user agents, project root, nested, nested .tau, nested .agents,
    # with matching contents. Side effects: writes files under the pytest tmp dir.
    tau_home = tmp_path / "home" / ".tau"
    agents_home = tmp_path / "home" / ".agents"
    project = tmp_path / "project"
    nested = project / "pkg"
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (tau_home).mkdir(parents=True)
    (agents_home).mkdir(parents=True)
    (project / ".tau").mkdir()
    (project / ".agents").mkdir()

    (tau_home / "AGENTS.md").write_text("User Tau instructions", encoding="utf-8")
    (agents_home / "AGENTS.md").write_text("User agents instructions", encoding="utf-8")
    (project / "AGENTS.md").write_text("Project instructions", encoding="utf-8")
    (nested / "AGENTS.md").write_text("Nested instructions", encoding="utf-8")
    (nested / ".tau").mkdir()
    (nested / ".agents").mkdir()
    (nested / ".tau" / "AGENTS.md").write_text("Project Tau instructions", encoding="utf-8")
    (nested / ".agents" / "AGENTS.md").write_text("Project agents instructions", encoding="utf-8")

    context_files = discover_project_context(
        TauResourcePaths(
            root=tau_home,
            agents_root=agents_home,
            cwd=nested,
            paths=TauPaths(home=tau_home, agents_home=agents_home),
        )
    )

    assert [Path(context_file.path) for context_file in context_files] == [
        tau_home / "AGENTS.md",
        agents_home / "AGENTS.md",
        project / "AGENTS.md",
        nested / "AGENTS.md",
        nested / ".tau" / "AGENTS.md",
        nested / ".agents" / "AGENTS.md",
    ]
    assert [context_file.content for context_file in context_files] == [
        "User Tau instructions",
        "User agents instructions",
        "Project instructions",
        "Nested instructions",
        "Project Tau instructions",
        "Project agents instructions",
    ]
