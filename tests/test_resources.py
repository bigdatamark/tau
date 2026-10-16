# [MODULE DOC]
# Tests for tau_coding resource path resolution (skills, prompts, themes,
# SYSTEM.md / APPEND_SYSTEM.md locations), system-prompt file discovery and
# precedence, and Markdown frontmatter / description parsing. Uses temporary
# directories and monkeypatched TAU_HOME; writes files only under tmp paths.
from pathlib import Path

import pytest

from tau_coding import TauPaths, TauResourcePaths
from tau_coding.resources import (
    ResourceError,
    derive_description,
    discover_system_prompt_resources,
    parse_markdown_resource,
)


def test_resource_paths_use_tau_subdirectories(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test that an explicit root with no agents root yields skills and prompts
    # subdirectories and single-entry search tuples for each.
    paths = TauResourcePaths(root=tmp_path, agents_root=None)

    assert paths.skills_dir == tmp_path / "skills"
    assert paths.prompts_dir == tmp_path / "prompts"
    assert paths.skills_dirs == (tmp_path / "skills",)
    assert paths.prompts_dirs == (tmp_path / "prompts",)


def test_default_resource_paths_follow_tau_home_and_share_agents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC]
    # Test that without an explicit root the TAU_HOME environment variable is the
    # root; themes search user then project ".tau/themes"; system prompt files live
    # in the root; the agents root defaults to "~/.agents"; and skill directories are
    # searched in order user, agents, project ".tau", project ".agents".
    tau_home = tmp_path / ".tau-personal"
    cwd = tmp_path / "project"
    monkeypatch.setenv("TAU_HOME", str(tau_home))

    paths = TauResourcePaths(cwd=cwd)

    assert paths.root == tau_home
    assert paths.skills_dir == tau_home / "skills"
    assert paths.prompts_dir == tau_home / "prompts"
    assert paths.themes_dirs == (tau_home / "themes", cwd / ".tau" / "themes")
    assert paths.system_prompt_path == tau_home / "SYSTEM.md"
    assert paths.append_system_prompt_path == tau_home / "APPEND_SYSTEM.md"
    assert paths.agents_root == Path.home() / ".agents"
    assert paths.skills_dirs == (
        tau_home / "skills",
        Path.home() / ".agents" / "skills",
        cwd / ".tau" / "skills",
        cwd / ".agents" / "skills",
    )


def test_explicit_resource_root_ignores_tau_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC]
    # Test that an explicit root takes precedence over TAU_HOME.
    explicit_root = tmp_path / "explicit"
    monkeypatch.setenv("TAU_HOME", str(tmp_path / "environment"))

    assert TauResourcePaths(root=explicit_root).root == explicit_root


def test_resource_paths_include_agents_and_project_directories(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test the full ordered skills and prompts search lists when root, agents root,
    # cwd and a TauPaths object are all supplied, plus system prompt file paths.
    cwd = tmp_path / "project"
    tau_home = tmp_path / "home" / ".tau"
    agents_home = tmp_path / "home" / ".agents"
    paths = TauResourcePaths(
        root=tau_home,
        agents_root=agents_home,
        cwd=cwd,
        paths=TauPaths(home=tau_home, agents_home=agents_home),
    )

    assert paths.skills_dirs == (
        tau_home / "skills",
        agents_home / "skills",
        cwd / ".tau" / "skills",
        cwd / ".agents" / "skills",
    )
    assert paths.prompts_dirs == (
        tau_home / "prompts",
        agents_home / "prompts",
        cwd / ".tau" / "prompts",
        cwd / ".agents" / "prompts",
    )
    assert paths.system_prompt_path == tau_home / "SYSTEM.md"
    assert paths.append_system_prompt_path == tau_home / "APPEND_SYSTEM.md"


def test_system_prompt_files_replace_by_precedence_and_append_in_order(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test that the project SYSTEM.md replaces the user one (which gets a "warning"
    # shadowed diagnostic), append files are combined user then project separated by
    # a blank line, ".agents/SYSTEM.md" is ignored, and diagnostics are ordered and
    # classified as expected.
    cwd = tmp_path / "project"
    tau_home = tmp_path / "home" / ".tau"
    agents_home = tmp_path / "home" / ".agents"
    (cwd / ".tau").mkdir(parents=True)
    tau_home.mkdir(parents=True)
    agents_home.mkdir(parents=True)
    (tau_home / "SYSTEM.md").write_text("User base", encoding="utf-8")
    (cwd / ".tau" / "SYSTEM.md").write_text("Project base", encoding="utf-8")
    (tau_home / "APPEND_SYSTEM.md").write_text("User append", encoding="utf-8")
    (cwd / ".tau" / "APPEND_SYSTEM.md").write_text("Project append", encoding="utf-8")
    # `.agents` is not a system-prompt configuration location.
    (agents_home / "SYSTEM.md").write_text("Agents base", encoding="utf-8")

    resources = discover_system_prompt_resources(
        TauResourcePaths(root=tau_home, agents_root=agents_home, cwd=cwd)
    )

    assert resources.custom_prompt == "Project base"
    assert resources.custom_prompt_path == cwd / ".tau" / "SYSTEM.md"
    assert resources.append_prompt == "User append\n\nProject append"
    assert resources.append_prompts == ("User append", "Project append")
    assert resources.append_prompt_paths == (
        tau_home / "APPEND_SYSTEM.md",
        cwd / ".tau" / "APPEND_SYSTEM.md",
    )
    assert [(item.severity, item.path) for item in resources.diagnostics] == [
        ("info", cwd / ".tau" / "SYSTEM.md"),
        ("warning", tau_home / "SYSTEM.md"),
        ("info", tau_home / "APPEND_SYSTEM.md"),
        ("info", cwd / ".tau" / "APPEND_SYSTEM.md"),
    ]


def test_overlapping_user_and_project_append_path_is_loaded_once(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test that when the project ".tau" directory is the user Tau home, the shared
    # APPEND_SYSTEM.md is loaded once with a single "append" diagnostic.
    tau_home = tmp_path / ".tau"
    tau_home.mkdir()
    append_path = tau_home / "APPEND_SYSTEM.md"
    append_path.write_text("Once", encoding="utf-8")

    resources = discover_system_prompt_resources(
        TauResourcePaths(root=tau_home, agents_root=None, cwd=tmp_path)
    )

    assert resources.append_prompt == "Once"
    assert resources.append_prompt_paths == (append_path,)
    assert [(item.name, item.path) for item in resources.diagnostics] == [("append", append_path)]


def test_explicit_replacement_shadows_file_but_append_files_still_load(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test that with an explicit startup system prompt, an unreadable SYSTEM.md is
    # not read (no error, custom prompt None), append files still load, and two
    # diagnostics explain the shadowing.
    tau_home = tmp_path / ".tau"
    tau_home.mkdir()
    (tau_home / "SYSTEM.md").write_bytes(b"\xff")
    (tau_home / "APPEND_SYSTEM.md").write_text("User append", encoding="utf-8")

    resources = discover_system_prompt_resources(
        TauResourcePaths(root=tau_home, agents_root=None),
        custom_prompt_explicit=True,
    )

    assert resources.custom_prompt is None
    assert resources.append_prompt == "User append"
    assert resources.append_prompt_paths == (tau_home / "APPEND_SYSTEM.md",)
    assert len(resources.diagnostics) == 2
    assert "explicit startup value" in resources.diagnostics[0].message
    assert "selected user" in resources.diagnostics[1].message


def test_selected_system_prompt_file_must_be_readable_utf8(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test that a selected SYSTEM.md containing invalid UTF-8 raises ResourceError
    # mentioning the replacement system prompt file.
    tau_home = tmp_path / ".tau"
    tau_home.mkdir()
    prompt_path = tau_home / "SYSTEM.md"
    prompt_path.write_bytes(b"\xff")

    with pytest.raises(ResourceError, match="Could not read replacement system prompt file"):
        discover_system_prompt_resources(TauResourcePaths(root=tau_home, agents_root=None))


def test_selected_append_system_prompt_file_must_be_readable_utf8(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test that an APPEND_SYSTEM.md with invalid UTF-8 raises ResourceError
    # mentioning the append system prompt file.
    tau_home = tmp_path / ".tau"
    tau_home.mkdir()
    (tau_home / "APPEND_SYSTEM.md").write_bytes(b"\xff")

    with pytest.raises(ResourceError, match="Could not read append system prompt file"):
        discover_system_prompt_resources(TauResourcePaths(root=tau_home, agents_root=None))


def test_parse_frontmatter_description() -> None:
    # [FUNCTION DOC]
    # Test that YAML frontmatter is split into a metadata mapping and the remaining
    # body text.
    metadata, body = parse_markdown_resource(
        "---\ndescription: Write tests\n---\n# Testing\nUse pytest."
    )

    assert metadata == {"description": "Write tests"}
    assert body == "# Testing\nUse pytest."


def test_parse_frontmatter_normalizes_crlf_line_endings() -> None:
    # [FUNCTION DOC]
    # Test that CRLF line endings are accepted in frontmatter and normalized to LF
    # in the returned body.
    metadata, body = parse_markdown_resource(
        "---\r\ndescription: Write tests\r\n---\r\n# Testing\r\nUse pytest."
    )

    assert metadata == {"description": "Write tests"}
    assert body == "# Testing\nUse pytest."


def test_derive_description_uses_first_heading_or_paragraph() -> None:
    # [FUNCTION DOC]
    # Test that the derived description is the first heading's text (without the
    # hash) or else the first line of the first paragraph, skipping leading blanks.
    assert derive_description("\n# Title\nBody") == "Title"
    assert derive_description("\nFirst paragraph\nMore") == "First paragraph"
