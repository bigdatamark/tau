# [MODULE DOC]
# Tests for markdown skill discovery, precedence, diagnostics, invocation
# expansion/formatting/parsing, the skill index and the
# disable-model-invocation flag. Uses temporary directories as Tau and
# .agents resource roots.
from pathlib import Path

import pytest

from tau_coding import (
    Skill,
    TauResourcePaths,
    build_skill_index,
    expand_skill_command,
    format_skill_invocation,
    load_skills,
    load_skills_with_diagnostics,
    parse_skill_invocation,
)
from tau_coding.resources import ResourceError


def test_load_skills_does_not_include_tau_self_knowledge(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts an empty Tau root with no agents root yields no skills (no bundled
    # self-knowledge skills are injected).
    skills = load_skills(TauResourcePaths(root=tmp_path, agents_root=None))

    assert skills == []


def test_load_skills_from_directory(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Creates two skill subdirectories (one with front-matter description, one
    # without) and asserts both load, with the description taken from front
    # matter or derived from the first heading.
    """Skills must live in ``<dir>/<name>/SKILL.md`` subdirectories."""
    skills_dir = tmp_path / "skills"
    (skills_dir / "python-testing").mkdir(parents=True)
    (skills_dir / "python-testing" / "SKILL.md").write_text(
        "---\ndescription: Test Python code\n---\n# Python Testing\nUse pytest.",
        encoding="utf-8",
    )
    (skills_dir / "git-review").mkdir()
    (skills_dir / "git-review" / "SKILL.md").write_text(
        "# Git Review\nReview diffs.", encoding="utf-8"
    )

    skills = load_skills(TauResourcePaths(root=tmp_path, agents_root=None))

    skill_by_name = {skill.name: skill for skill in skills}
    assert set(skill_by_name) == {"git-review", "python-testing"}
    assert skill_by_name["git-review"].description == "Git Review"
    assert skill_by_name["python-testing"].description == "Test Python code"


def test_load_skills_includes_user_and_project_agents_directories(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts skills under the user agents home and the project's .agents/skills
    # are both loaded.
    """Skills in .agents/skills/ must be subdirectories containing SKILL.md."""
    tau_home = tmp_path / "home" / ".tau"
    agents_home = tmp_path / "home" / ".agents"
    cwd = tmp_path / "project"
    (agents_home / "skills" / "user-skill").mkdir(parents=True)
    (agents_home / "skills" / "user-skill" / "SKILL.md").write_text(
        "# User Skill\nFrom user agents.", encoding="utf-8"
    )
    (cwd / ".agents" / "skills" / "project-skill").mkdir(parents=True)
    (cwd / ".agents" / "skills" / "project-skill" / "SKILL.md").write_text(
        "# Project Skill\nFrom project agents.", encoding="utf-8"
    )

    skills = load_skills(TauResourcePaths(root=tau_home, agents_root=agents_home, cwd=cwd))

    assert {skill.name for skill in skills} == {"project-skill", "user-skill"}


def test_user_skill_can_use_a_former_bundled_skill_name(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts a user skill named like a formerly bundled skill loads from the
    # user path with no diagnostics.
    skill_dir = tmp_path / "skills" / "tau-model-catalog"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text("# Custom catalog workflow", encoding="utf-8")

    skills, diagnostics = load_skills_with_diagnostics(
        TauResourcePaths(root=tmp_path, agents_root=None)
    )

    assert [(skill.name, skill.path) for skill in skills] == [("tau-model-catalog", skill_path)]
    assert diagnostics == []


def test_project_agents_skill_overrides_user_agents_skill(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts a project .agents skill replaces a same-named user .agents skill
    # (path and description come from the project copy).
    """Project .agents/skills/ skills take precedence over user-level ones."""
    tau_home = tmp_path / "home" / ".tau"
    agents_home = tmp_path / "home" / ".agents"
    cwd = tmp_path / "project"
    (agents_home / "skills" / "review").mkdir(parents=True)
    (agents_home / "skills" / "review" / "SKILL.md").write_text("# User Review", encoding="utf-8")
    (cwd / ".agents" / "skills" / "review").mkdir(parents=True)
    (cwd / ".agents" / "skills" / "review" / "SKILL.md").write_text(
        "# Project Review", encoding="utf-8"
    )

    skills = load_skills(TauResourcePaths(root=tau_home, agents_root=agents_home, cwd=cwd))

    review = next(skill for skill in skills if skill.name == "review")
    assert review.path == cwd / ".agents" / "skills" / "review" / "SKILL.md"
    assert review.description == "Project Review"


def test_load_skills_with_diagnostics_reports_overrides(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts a project .tau skill overrides a user .tau skill and exactly one
    # "overrides lower-precedence resource" skill diagnostic names it.
    tau_home = tmp_path / "home" / ".tau"
    agents_home = tmp_path / "home" / ".agents"
    cwd = tmp_path / "project"
    (tau_home / "skills" / "review").mkdir(parents=True)
    (tau_home / "skills" / "review" / "SKILL.md").write_text("# User Tau Review", encoding="utf-8")
    (cwd / ".tau" / "skills" / "review").mkdir(parents=True)
    (cwd / ".tau" / "skills" / "review" / "SKILL.md").write_text(
        "# Project Tau Review", encoding="utf-8"
    )

    skills, diagnostics = load_skills_with_diagnostics(
        TauResourcePaths(root=tau_home, agents_root=agents_home, cwd=cwd)
    )

    review = next(skill for skill in skills if skill.name == "review")
    assert review.path == cwd / ".tau" / "skills" / "review" / "SKILL.md"
    override_diagnostics = [
        d for d in diagnostics if "overrides lower-precedence resource" in d.message
    ]
    assert len(override_diagnostics) == 1
    assert override_diagnostics[0].kind == "skill"
    assert override_diagnostics[0].name == "review"


def test_load_skills_with_diagnostics_reports_bare_md_migration_hint(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC]
    # Asserts a bare legacy .md file yields exactly one info diagnostic with its
    # name, path and a migration hint pointing to name/SKILL.md, while a proper
    # skill directory still loads.
    """Bare ``.md`` files at a skills-dir root produce an info diagnostic.

    They are silently skipped from the loaded skill set, but users are told
    how to migrate: rename ``foo.md`` to ``foo/SKILL.md``.
    """
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "legacy.md").write_text("# Legacy Skill\nOld body.", encoding="utf-8")
    (skills_dir / "good").mkdir()
    (skills_dir / "good" / "SKILL.md").write_text("# Good Skill", encoding="utf-8")

    skills, diagnostics = load_skills_with_diagnostics(
        TauResourcePaths(root=tmp_path, agents_root=None)
    )

    assert "good" in {skill.name for skill in skills}
    migration_diagnostics = [d for d in diagnostics if d.severity == "info"]
    assert len(migration_diagnostics) == 1
    assert migration_diagnostics[0].name == "legacy"
    assert migration_diagnostics[0].path == skills_dir / "legacy.md"
    assert "bare .md files are no longer treated as skills" in migration_diagnostics[0].message
    assert str(skills_dir / "legacy" / "SKILL.md") in migration_diagnostics[0].message


def test_agents_root_is_not_a_skills_directory(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts markdown files directly in the .agents root (AGENTS.md, README.md,
    # review.md) are not loaded as skills.
    """The .agents root directory itself must not be scanned for skills.

    Files like ``README.md`` or ``AGENTS.md`` in the root should be ignored.
    Only ``.agents/skills/`` is a valid skill location.
    """
    agents_home = tmp_path / ".agents"
    agents_home.mkdir()
    (agents_home / "AGENTS.md").write_text("# Instructions", encoding="utf-8")
    (agents_home / "README.md").write_text("# Readme", encoding="utf-8")
    (agents_home / "review.md").write_text("# Review", encoding="utf-8")

    skills = load_skills(TauResourcePaths(root=tmp_path / ".tau", agents_root=agents_home))

    assert skills == []


def test_agents_skills_dir_ignores_bare_md_files(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts a bare .md file beside a skill directory in .agents/skills is not
    # loaded and does not stop the valid skill loading.
    """Bare .md files in .agents/skills/ are not treated as skills.

    Only subdirectories containing ``SKILL.md`` are valid. Files like
    ``reference.md`` alongside a skill directory should be ignored.
    """
    agents_home = tmp_path / ".agents"
    skills_dir = agents_home / "skills"
    (skills_dir / "my-skill").mkdir(parents=True)
    (skills_dir / "my-skill" / "SKILL.md").write_text("# Valid Skill", encoding="utf-8")
    (skills_dir / "reference.md").write_text("# Reference doc", encoding="utf-8")

    paths = TauResourcePaths(root=tmp_path / ".tau", agents_root=agents_home)
    skills = load_skills(paths)

    assert "my-skill" in {skill.name for skill in skills}


def test_tau_skills_dir_ignores_bare_md_files(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Same as the .agents case but for the Tau skills directory.
    """Bare .md files in .tau/skills/ are also ignored (unified with .agents/).

    Tau diverges from Pi here: Pi keeps a permissive ``.pi/skills`` for
    backward compatibility, but Tau applies the Agent Skills spec uniformly.
    """
    skills_dir = tmp_path / "skills"
    (skills_dir / "my-skill").mkdir(parents=True)
    (skills_dir / "my-skill" / "SKILL.md").write_text("# Subdir Skill", encoding="utf-8")
    (skills_dir / "reference.md").write_text("# Reference doc", encoding="utf-8")

    paths = TauResourcePaths(root=tmp_path, agents_root=None)
    skills = load_skills(paths)

    assert "my-skill" in {skill.name for skill in skills}


def test_expand_skill_command_includes_skill_and_user_request(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts "/skill:testing add parser tests" expands to a skill tag with name
    # and location, a references line, the skill body, and the request after the
    # closing tag.
    skills_dir = tmp_path / "skills" / "testing"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("# Testing\nRun pytest.", encoding="utf-8")
    skills = load_skills(TauResourcePaths(root=tmp_path, agents_root=None))

    expanded = expand_skill_command("/skill:testing add parser tests", skills)

    assert expanded is not None
    testing = next(skill for skill in skills if skill.name == "testing")
    assert f'<skill name="testing" location="{testing.path}">' in expanded
    assert f"References are relative to {testing.path.parent}." in expanded
    assert "Run pytest." in expanded
    assert expanded.endswith("</skill>\n\nadd parser tests")


def test_expand_skill_command_includes_multiline_user_request(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts a multi-line request after the command is preserved verbatim
    # (stripped) after the closing tag.
    skill = Skill(
        name="testing",
        path=tmp_path / "skills" / "testing" / "SKILL.md",
        content="# Testing\nRun pytest.",
        description="Test code",
    )

    expanded = expand_skill_command(
        "/skill:testing\n\nhere are some instructions\n\n- keep this structure",
        [skill],
    )

    assert expanded is not None
    assert expanded.endswith("</skill>\n\nhere are some instructions\n\n- keep this structure")


def test_format_skill_invocation_without_extra_instructions(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts the exact invocation string when no instructions are supplied.
    skill = Skill(
        name="testing",
        path=tmp_path / "skills" / "testing" / "SKILL.md",
        content="# Testing\nRun pytest.",
        description="Test code",
    )

    formatted = format_skill_invocation(skill)

    assert formatted == (
        f'<skill name="testing" location="{skill.path}">\n'
        f"References are relative to {skill.path.parent}.\n\n"
        "# Testing\n"
        "Run pytest.\n"
        "</skill>"
    )


def test_parse_skill_invocation_extracts_display_metadata(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts round-tripping a formatted invocation recovers name, location,
    # content and additional instructions.
    skill = Skill(
        name="testing",
        path=tmp_path / "skills" / "testing" / "SKILL.md",
        content="# Testing\nRun pytest.",
        description="Test Python code",
    )
    formatted = format_skill_invocation(skill, "add parser tests")

    parsed = parse_skill_invocation(formatted)

    assert parsed is not None
    assert parsed.name == "testing"
    assert parsed.location == str(skill.path)
    assert "# Testing" in parsed.content
    assert parsed.additional_instructions == "add parser tests"


def test_expand_skill_command_returns_none_for_normal_prompt(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts non-skill text expands to none.
    assert (
        expand_skill_command(
            "hello", load_skills(TauResourcePaths(root=tmp_path, agents_root=None))
        )
        is None
    )


def test_expand_skill_command_rejects_unknown_skill() -> None:
    # [FUNCTION DOC]
    # Asserts an unknown skill name raises ResourceError mentioning "Unknown
    # skill".
    with pytest.raises(ResourceError, match="Unknown skill"):
        expand_skill_command("/skill:missing", [])


def test_build_skill_index(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts the index lists a loaded skill with its description and contains no
    # formerly bundled skill names.
    skills_dir = tmp_path / "skills" / "testing"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        "---\ndescription: Test things\n---\nBody",
        encoding="utf-8",
    )

    index = build_skill_index(load_skills(TauResourcePaths(root=tmp_path, agents_root=None)))

    assert "Available skills:" in index
    assert "- testing: Test things" in index
    assert "create-tau-extension" not in index
    assert "tau-model-catalog" not in index


def test_load_skill_parses_disable_model_invocation(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts the disable flag is true only for the literal value true; missing,
    # false and "yes" all leave it false.
    skills_dir = tmp_path / "skills"
    (skills_dir / "hidden").mkdir(parents=True)
    (skills_dir / "hidden" / "SKILL.md").write_text(
        "---\ndescription: Hidden skill\ndisable-model-invocation: true\n---\nBody",
        encoding="utf-8",
    )
    (skills_dir / "visible").mkdir()
    (skills_dir / "visible" / "SKILL.md").write_text(
        "---\ndescription: Visible skill\n---\nBody",
        encoding="utf-8",
    )
    (skills_dir / "explicit-false").mkdir()
    (skills_dir / "explicit-false" / "SKILL.md").write_text(
        "---\ndescription: Explicitly enabled\ndisable-model-invocation: false\n---\nBody",
        encoding="utf-8",
    )
    (skills_dir / "invalid-value").mkdir()
    (skills_dir / "invalid-value" / "SKILL.md").write_text(
        "---\ndescription: Invalid flag value\ndisable-model-invocation: yes\n---\nBody",
        encoding="utf-8",
    )

    skills = load_skills(TauResourcePaths(root=tmp_path, agents_root=None))

    skill_by_name = {skill.name: skill for skill in skills}
    assert skill_by_name["hidden"].disable_model_invocation is True
    assert skill_by_name["visible"].disable_model_invocation is False
    assert skill_by_name["explicit-false"].disable_model_invocation is False
    assert skill_by_name["invalid-value"].disable_model_invocation is False


def test_disable_model_invocation_accepts_case_insensitive_true(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts "True" (capitalized) also enables the flag.
    skills_dir = tmp_path / "skills" / "hidden"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        "---\ndescription: Hidden skill\ndisable-model-invocation: True\n---\nBody",
        encoding="utf-8",
    )

    skills = load_skills(TauResourcePaths(root=tmp_path, agents_root=None))

    assert skills[0].disable_model_invocation is True


def test_disabled_skill_remains_invocable_via_skill_command(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts a skill hidden from the model can still be expanded explicitly by
    # the user.
    skill = Skill(
        name="hidden",
        path=tmp_path / "skills" / "hidden" / "SKILL.md",
        content="Hidden body",
        description="Hidden skill",
        disable_model_invocation=True,
    )

    expanded = expand_skill_command("/skill:hidden", [skill])

    assert expanded is not None
    assert "Hidden body" in expanded


def test_build_skill_index_excludes_disabled_skills(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Asserts hidden skills are omitted from the index and an index of only
    # hidden skills reads "Available skills: none".
    visible = Skill(
        name="visible",
        path=tmp_path / "visible" / "SKILL.md",
        content="Body",
        description="Visible skill",
    )
    hidden = Skill(
        name="hidden",
        path=tmp_path / "hidden" / "SKILL.md",
        content="Body",
        description="Hidden skill",
        disable_model_invocation=True,
    )

    index = build_skill_index([visible, hidden])

    assert "- visible: Visible skill" in index
    assert "hidden" not in index

    assert build_skill_index([hidden]) == "Available skills: none"
