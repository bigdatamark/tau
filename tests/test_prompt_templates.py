# [MODULE DOC]
# Tests for prompt templates (tau_coding.prompt_templates): discovery from
# Tau and .agents prompt directories with project-over-user precedence,
# override and reserved-name diagnostics, "{{ name }}" variable rendering, and
# slash-command expansion with Pi-style positional arguments ($1, $@,
# $ARGUMENTS, defaults and slices). Uses temporary directories; no network.
from pathlib import Path

import pytest

from tau_coding import (
    TauResourcePaths,
    expand_prompt_template_command,
    load_prompt_templates,
    load_prompt_templates_with_diagnostics,
    render_prompt_template,
)
from tau_coding.prompt_templates import PromptTemplate
from tau_coding.resources import ResourceError


def test_load_prompt_templates_missing_directory_returns_empty(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Verifies a resource root with no prompts directory yields an empty list.
    assert load_prompt_templates(TauResourcePaths(root=tmp_path, agents_root=None)) == []


def test_load_prompt_templates_from_markdown_files(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Verifies a markdown file with frontmatter becomes a template named after the
    # file stem with the frontmatter description.
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "review.md").write_text(
        "---\ndescription: Review code\n---\nReview {{ topic }}.",
        encoding="utf-8",
    )

    templates = load_prompt_templates(TauResourcePaths(root=tmp_path, agents_root=None))

    assert len(templates) == 1
    assert templates[0].name == "review"
    assert templates[0].description == "Review code"


def test_load_prompt_templates_includes_agents_directories(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Verifies user-level ~/.agents/prompts and project .agents/prompts are both
    # loaded, project templates listed first.
    tau_home = tmp_path / "home" / ".tau"
    agents_home = tmp_path / "home" / ".agents"
    cwd = tmp_path / "project"
    (agents_home / "prompts").mkdir(parents=True)
    (agents_home / "prompts" / "user.md").write_text("User prompt", encoding="utf-8")
    (cwd / ".agents" / "prompts").mkdir(parents=True)
    (cwd / ".agents" / "prompts" / "project.md").write_text("Project prompt", encoding="utf-8")

    templates = load_prompt_templates(
        TauResourcePaths(root=tau_home, agents_root=agents_home, cwd=cwd)
    )

    assert [template.name for template in templates] == ["project", "user"]


def test_project_prompt_template_overrides_user_template(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Verifies a same-named project template replaces the user template (path and
    # content come from the project file).
    tau_home = tmp_path / "home" / ".tau"
    agents_home = tmp_path / "home" / ".agents"
    cwd = tmp_path / "project"
    (agents_home / "prompts").mkdir(parents=True)
    (agents_home / "prompts" / "review.md").write_text("User review", encoding="utf-8")
    (cwd / ".agents" / "prompts").mkdir(parents=True)
    (cwd / ".agents" / "prompts" / "review.md").write_text("Project review", encoding="utf-8")

    templates = load_prompt_templates(
        TauResourcePaths(root=tau_home, agents_root=agents_home, cwd=cwd)
    )

    assert len(templates) == 1
    assert templates[0].path == cwd / ".agents" / "prompts" / "review.md"
    assert templates[0].content == "Project review"


def test_load_prompt_templates_with_diagnostics_reports_overrides(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Verifies the diagnostics loader keeps the project .tau template and emits one
    # "prompt" diagnostic saying it overrides a lower-precedence resource.
    tau_home = tmp_path / "home" / ".tau"
    agents_home = tmp_path / "home" / ".agents"
    cwd = tmp_path / "project"
    (tau_home / "prompts").mkdir(parents=True)
    (tau_home / "prompts" / "review.md").write_text("User Tau review", encoding="utf-8")
    (cwd / ".tau" / "prompts").mkdir(parents=True)
    (cwd / ".tau" / "prompts" / "review.md").write_text(
        "Project Tau review",
        encoding="utf-8",
    )

    templates, diagnostics = load_prompt_templates_with_diagnostics(
        TauResourcePaths(root=tau_home, agents_root=agents_home, cwd=cwd)
    )

    assert [template.name for template in templates] == ["review"]
    assert templates[0].path == cwd / ".tau" / "prompts" / "review.md"
    assert len(diagnostics) == 1
    assert diagnostics[0].kind == "prompt"
    assert diagnostics[0].name == "review"
    assert "overrides lower-precedence resource" in diagnostics[0].message


@pytest.mark.parametrize("reserved_name", ["prompts", "skills", "tools"])
def test_reserved_picker_template_is_ignored_with_diagnostic(
    tmp_path: Path, reserved_name: str
) -> None:
    # [FUNCTION DOC]
    # Parametrized over prompts/skills/tools: an upper-case file named after a
    # built-in picker command is skipped (case-insensitive) with a diagnostic
    # naming the reserved command and the file path.
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    reserved_path = prompts_dir / f"{reserved_name.upper()}.md"
    reserved_path.write_text("Shadow the picker", encoding="utf-8")

    templates, diagnostics = load_prompt_templates_with_diagnostics(
        TauResourcePaths(root=tmp_path, agents_root=None)
    )

    assert templates == []
    assert len(diagnostics) == 1
    assert diagnostics[0].path == reserved_path
    assert f"reserved by the built-in /{reserved_name} command" in diagnostics[0].message


def test_render_prompt_template_replaces_variables() -> None:
    # [FUNCTION DOC]
    # Verifies "{{ var }}" placeholders are substituted from a mapping.
    template = PromptTemplate(
        name="review",
        path=Path("review.md"),
        content="Review {{ topic }} for {{ focus }}.",
    )

    assert render_prompt_template(template, {"topic": "auth", "focus": "security"}) == (
        "Review auth for security."
    )


def test_render_prompt_template_rejects_missing_variables() -> None:
    # [FUNCTION DOC]
    # Verifies rendering raises ResourceError ("Missing prompt template variable")
    # when a variable is not supplied.
    template = PromptTemplate(name="review", path=Path("review.md"), content="Review {{ topic }}.")

    with pytest.raises(ResourceError, match="Missing prompt template variable"):
        render_prompt_template(template, {})


def test_expand_prompt_template_command_replaces_slash_command() -> None:
    # [FUNCTION DOC]
    # Verifies "/name args" expands the template with {{ arguments }} set to the
    # argument text.
    template = PromptTemplate(
        name="example",
        path=Path("example.md"),
        content="Use these arguments: {{ arguments }}",
    )

    assert expand_prompt_template_command("/example src/app.py", [template]) == (
        "Use these arguments: src/app.py"
    )


def test_expand_prompt_template_command_supports_pi_argument_variables() -> None:
    # [FUNCTION DOC]
    # Verifies shell-like quoting and $1, $2, $@ and $ARGUMENTS substitution.
    template = PromptTemplate(
        name="example",
        path=Path("example.md"),
        content="first=$1 second=$2 all=$@ named=$ARGUMENTS",
    )

    assert expand_prompt_template_command('/example one "two words"', [template]) == (
        "first=one second=two words all=one two words named=one two words"
    )


def test_expand_prompt_template_command_supports_pi_argument_defaults_and_slices() -> None:
    # [FUNCTION DOC]
    # Verifies ${1:-default}, ${@:start}, ${@:start:len} and ${@:-default} with
    # and without arguments.
    template = PromptTemplate(
        name="example",
        path=Path("example.md"),
        content="count=${1:-7} rest=${@:2} limited=${@:2:1} all=${@:-none}",
    )

    assert expand_prompt_template_command("/example", [template]) == (
        "count=7 rest= limited= all=none"
    )
    assert expand_prompt_template_command("/example one two three", [template]) == (
        "count=one rest=two three limited=two all=one two three"
    )


def test_expand_prompt_template_command_blanks_missing_custom_variables() -> None:
    # [FUNCTION DOC]
    # Verifies unknown {{ }} variables become empty during command expansion
    # (instead of raising) while {{ arguments }} is filled.
    template = PromptTemplate(
        name="review",
        path=Path("review.md"),
        content="Base branch: {{ base_branch }}\nReview PR {{ arguments }}.",
    )

    assert expand_prompt_template_command("/review 168", [template]) == (
        "Base branch: \nReview PR 168."
    )


def test_expand_prompt_template_command_appends_arguments_without_argument_placeholder() -> None:
    # [FUNCTION DOC]
    # Verifies that when the template has no argument placeholder, arguments are
    # appended after a blank line (custom variables still blanked).
    template = PromptTemplate(
        name="review",
        path=Path("review.md"),
        content="Base branch: {{ base_branch }}\nReview this code.",
    )

    assert expand_prompt_template_command("/review src/app.py", [template]) == (
        "Base branch: \nReview this code.\n\nsrc/app.py"
    )


def test_expand_prompt_template_command_appends_arguments_without_placeholder() -> None:
    # [FUNCTION DOC]
    # Verifies plain templates get the arguments appended after a blank line.
    template = PromptTemplate(name="review", path=Path("review.md"), content="Review this code.")

    assert expand_prompt_template_command("/review src/app.py", [template]) == (
        "Review this code.\n\nsrc/app.py"
    )


def test_expand_prompt_template_command_ignores_unknown_commands() -> None:
    # [FUNCTION DOC]
    # Verifies an unknown slash command returns none.
    template = PromptTemplate(name="review", path=Path("review.md"), content="Review this code.")

    assert expand_prompt_template_command("/missing src/app.py", [template]) is None
