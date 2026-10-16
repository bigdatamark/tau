# [MODULE DOC]
# Assembles the system prompt for Tau coding sessions, mirroring Pi's prompt layout.
# The prompt is built as an ordered list of attributed source sections (default or
# override prompt, appended sections, extension sections, project context files, skill
# listings, current date and working directory) whose contents concatenate into the
# final text. Also renders a Markdown "source map" for the /system command and exposes
# standalone formatters for tools, guidelines, project context and skills.
# Pure string building; the only external reads are the date and the installed docs paths.
"""System prompt assembly for Tau coding sessions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal
from xml.sax.saxutils import escape

from tau_agent.tools import AgentTool
from tau_coding.self_docs import tau_docs_path, tau_examples_path, tau_readme_path
from tau_coding.skills import Skill

PromptSourceKind = Literal[
    "default", "system", "append", "extension", "context", "skill", "runtime"
]


@dataclass(frozen=True, slots=True)
class ProjectContextFile:
    # [CLASS DOC]
    # Immutable project instruction file (e.g. AGENTS.md) to embed: display path and content.
    """A project instruction file included in the system prompt."""

    path: str
    content: str


@dataclass(frozen=True, slots=True)
class PromptSection:
    # [CLASS DOC]
    # Immutable free-form prompt section: optional title (rendered as a level-2 heading),
    # body text and optional provenance label.
    """A free-form section appended to the system prompt."""

    title: str | None
    body: str
    source: str | None = None


@dataclass(frozen=True, slots=True)
class SystemPromptSource:
    # [CLASS DOC]
    # Immutable, attributed contiguous chunk of the effective prompt: kind (default, system,
    # append, extension, context, skill, runtime), human label, provenance string and the
    # exact content text (including its leading separators).
    """One contiguous, attributed section of an effective system prompt."""

    kind: PromptSourceKind
    label: str
    source: str
    content: str


@dataclass(frozen=True, slots=True)
class SystemPromptInspection:
    # [CLASS DOC]
    # Immutable pair of the full prompt text and the ordered tuple of non-empty sources.
    """An effective system prompt paired with its ordered provenance."""

    text: str
    sources: tuple[SystemPromptSource, ...]


@dataclass(frozen=True, slots=True)
class BuildSystemPromptOptions:
    # [CLASS DOC]
    # Immutable options for building the prompt: working directory, visible tools, loaded
    # skills, optional full custom prompt (replaces the default body) and its provenance,
    # optional appended prompt text, project context files, optional date override, extra
    # guideline strings, appended sections and extension-contributed sections.
    """Options used to build Tau's system prompt."""

    cwd: Path
    tools: Sequence[AgentTool] = ()
    skills: Sequence[Skill] = ()
    custom_prompt: str | None = None
    append_system_prompt: str | None = None
    context_files: Sequence[ProjectContextFile] = ()
    current_date: date | None = None
    extra_guidelines: Sequence[str] = field(default_factory=tuple)
    append_sections: Sequence[PromptSection] = field(default_factory=tuple)
    extra_sections: Sequence[PromptSection] = field(default_factory=tuple)
    custom_prompt_source: str | None = None


def build_system_prompt(options: BuildSystemPromptOptions) -> str:
    # [FUNCTION DOC]
    # Return only the text of build_system_prompt_inspection for the given options.
    """Build a deterministic Pi-style system prompt for Tau."""
    return build_system_prompt_inspection(options).text


def build_system_prompt_inspection(options: BuildSystemPromptOptions) -> SystemPromptInspection:
    # [FUNCTION DOC]
    # Build the prompt and attribute every contiguous section, in this order:
    # 1) If a custom prompt is given, a "system" source with it verbatim (provenance from
    # options or "runtime configuration"); else the "default" Tau prompt: role intro,
    # "Available tools" list, note about custom tools, "Guidelines" bullets, and the Tau
    # documentation routing block.
    # 2) "append" sources: the append_system_prompt text (if non-empty) first, then the
    # appended sections; then "extension" sources from extra_sections. Each is prefixed by
    # a blank line and rendered with an optional heading; label defaults to "Appended
    # system prompt", provenance to "runtime configuration".
    # 3) Project context sources (XML-like wrapper).
    # 4) Skill listing sources, only when a tool named "read" is available.
    # 5) Runtime sources: current date (today unless overridden) in ISO format and the
    # working directory (backslashes converted to forward slashes).
    # Output: inspection whose text concatenates all contents and whose sources omit empty
    # contents. No exceptions expected; no side effects.
    """Build Tau's system prompt and attribute every contiguous section."""
    current_date = options.current_date or date.today()
    cwd = _format_path(options.cwd)
    sources: list[SystemPromptSource] = []

    if options.custom_prompt is not None:
        sources.append(
            SystemPromptSource(
                kind="system",
                label="System prompt override",
                source=options.custom_prompt_source or "runtime configuration",
                content=options.custom_prompt,
            )
        )
    else:
        sources.append(
            SystemPromptSource(
                kind="default",
                label="Tau default prompt",
                source="tau_coding.system_prompt",
                content=(
                    "You are an expert coding assistant operating inside Tau, a coding agent "
                    "harness. You help users by reading files, executing commands, editing code, "
                    "and writing new files."
                    f"\n\nAvailable tools:\n{format_available_tools(options.tools)}"
                    "\n\nIn addition to the tools above, you may have access to other custom tools "
                    "depending on the project."
                    f"\n\nGuidelines:\n{format_guidelines(options.tools, options.extra_guidelines)}"
                    f"\n\n{format_tau_documentation()}"
                ),
            )
        )

    append_sections = list(options.append_sections)
    if options.append_system_prompt:
        append_sections.insert(
            0,
            PromptSection(
                title=None,
                body=options.append_system_prompt,
                source="runtime append configuration",
            ),
        )

    def add_sections(sections: Sequence[PromptSection], kind: PromptSourceKind) -> None:
        # [FUNCTION DOC]
        # HELPER: Append one source of the given kind per section to the enclosing sources list,
        # with a leading blank line and default label/provenance as described above.
        for section in sections:
            sources.append(
                SystemPromptSource(
                    kind=kind,
                    label=section.title or "Appended system prompt",
                    source=section.source or "runtime configuration",
                    content=f"\n\n{format_prompt_section(section)}",
                )
            )

    add_sections(append_sections, "append")
    add_sections(options.extra_sections, "extension")

    sources.extend(_project_context_sources(options.context_files))
    if _has_tool(options.tools, "read"):
        sources.extend(_skill_sources(options.skills))
    sources.extend(
        (
            SystemPromptSource(
                kind="runtime",
                label="Current date",
                source="Tau runtime",
                content=f"\nCurrent date: {current_date.isoformat()}",
            ),
            SystemPromptSource(
                kind="runtime",
                label="Working directory",
                source=cwd,
                content=f"\nCurrent working directory: {cwd}",
            ),
        )
    )
    text = "".join(source.content for source in sources)
    return SystemPromptInspection(
        text=text, sources=tuple(source for source in sources if source.content)
    )


def format_system_prompt_inspection(inspection: SystemPromptInspection) -> str:
    # [FUNCTION DOC]
    # Render a Markdown source map for the local /system command. Each source becomes a
    # block with a level-4 heading "<two-digit index> · <label>", a bold "Source:" line
    # showing the provenance in an inline code span (control characters replaced by spaces,
    # trimmed, with a backtick delimiter longer than any backtick run it contains), and the
    # content with leading newlines removed. Blocks are joined by horizontal rules.
    """Render a Markdown source map for the local ``/system`` command."""
    sections: list[str] = []
    for index, source in enumerate(inspection.sources, start=1):
        origin = "".join(
            " " if ord(character) < 32 or ord(character) == 127 else character
            for character in source.source
        ).strip()
        delimiter = "`"
        while delimiter in origin:
            delimiter += "`"
        content = source.content.lstrip("\n")
        sections.append(
            f"#### {index:02d} · {source.label}\n\n"
            f"**Source:** {delimiter}{origin}{delimiter}\n\n"
            f"{content}"
        )
    return "\n\n---\n\n".join(sections)


def format_prompt_section(section: PromptSection) -> str:
    # [FUNCTION DOC]
    # Render a section: the body alone when untitled, otherwise "## <title>", a blank line,
    # then the body.
    """Render one optional-title free-form prompt section."""
    if section.title is None:
        return section.body
    return f"## {section.title}\n\n{section.body}"


def format_tau_documentation() -> str:
    # [FUNCTION DOC]
    # Return Pi-style routing hints pointing to Tau's installed README, docs directory and
    # examples directory (paths with forward slashes), telling the model when to read which
    # doc file and to resolve docs/ and examples/ against those locations, not the cwd.
    """Format Pi-style routing hints to Tau's installed reference material."""
    readme_path = _format_path(tau_readme_path())
    docs_path = _format_path(tau_docs_path())
    examples_path = _format_path(tau_examples_path())
    return (
        "Tau documentation (read only when the user asks about Tau itself, its SDK, "
        "extensions, skills, providers, models, commands, or TUI):\n"
        f"- Main documentation: {readme_path}\n"
        f"- Additional docs: {docs_path}\n"
        f"- Examples: {examples_path} (extensions and custom tools)\n"
        "- When reading Tau docs or examples, resolve docs/... under Additional docs and "
        "examples/... under Examples, not the current working directory\n"
        "- When asked about: creating or modifying extensions (docs/extensions.md, "
        "examples/extensions/), skills and prompt templates (docs/skills.md), custom "
        "providers or adding built-in providers/models (docs/models.md), CLI and slash "
        "commands (docs/cli.md), TUI usage "
        "(docs/tui.md), Tau architecture and packages (docs/architecture.md)\n"
        "- When working on Tau topics, read the docs and examples, and follow .md "
        "cross-references before implementing\n"
        "- Always read relevant Tau .md files completely and follow links to related docs"
    )


def format_available_tools(tools: Sequence[AgentTool]) -> str:
    # [FUNCTION DOC]
    # Return one "- name: snippet" line per tool that has a prompt snippet, joined by
    # newlines, or "(none)" if no tool has one.
    """Format visible tools using prompt snippets."""
    lines = [f"- {tool.name}: {tool.prompt_snippet}" for tool in tools if tool.prompt_snippet]
    return "\n".join(lines) if lines else "(none)"


def collect_prompt_guidelines(
    tools: Sequence[AgentTool], extra_guidelines: Sequence[str] = ()
) -> list[str]:
    # [FUNCTION DOC]
    # Collect de-duplicated guideline strings (trimmed; blanks and repeats dropped, first
    # occurrence wins, insertion order kept). Starts with a bash hint: if bash exists without
    # grep/find/ls, advise bash for file operations; if both, prefer grep/find/ls tools.
    # Then each tool's own prompt guidelines, then extra guidelines, then fixed general
    # coding-agent guidelines (inspect first, focused changes, don't discard user changes,
    # use documented commands, run checks, honest reporting, ask before destructive ops,
    # be concise, show file paths). Returns the list.
    """Collect and de-duplicate system prompt guidelines."""
    names = {tool.name for tool in tools}
    guidelines: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Trim a guideline and append it unless empty or already seen; mutates the
        # enclosing seen set and guideline list.
        normalized = value.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        guidelines.append(normalized)

    has_bash = "bash" in names
    has_exploration_tools = bool({"grep", "find", "ls"} & names)
    if has_bash and not has_exploration_tools:
        add("Use bash for file operations like ls, rg, find")
    elif has_bash and has_exploration_tools:
        add(
            "Prefer grep/find/ls tools over bash for file exploration (faster, respects .gitignore)"
        )

    for tool in tools:
        for guideline in tool.prompt_guidelines:
            add(guideline)
    for guideline in extra_guidelines:
        add(guideline)

    add("Inspect relevant files and project instructions before editing")
    add("Make focused changes that preserve the project's architecture and style")
    add("Do not overwrite or discard unrelated user changes")
    add("Use the project's documented commands and package manager")
    add("Run relevant tests, formatting, linting, and type checks after changes")
    add("Report checks honestly; never claim a command passed unless you ran it")
    add("Ask before destructive operations or materially ambiguous design choices")
    add("Be concise in your responses")
    add("Show file paths clearly when working with files")
    return guidelines


def format_guidelines(tools: Sequence[AgentTool], extra_guidelines: Sequence[str] = ()) -> str:
    # [FUNCTION DOC]
    # Return the collected guidelines as Markdown bullet lines joined by newlines.
    """Format prompt guidelines as markdown bullets."""
    return "\n".join(
        f"- {guideline}" for guideline in collect_prompt_guidelines(tools, extra_guidelines)
    )


def format_project_context(context_files: Sequence[ProjectContextFile]) -> str:
    # [FUNCTION DOC]
    # Standalone formatter for project context files in Pi's XML-like wrapper: returns ""
    # when empty; otherwise a block starting with a blank line and project_context tag, an
    # intro line, then each file as a project_instructions element with an escaped path
    # attribute, its content, and a trailing blank line, ending with the closing tag.
    # KNOWN ISSUE: the XML escape used does not escape double quotes, so a path containing a
    # double quote would break the attribute. Also its spacing differs slightly from the
    # attributed sources used by build_system_prompt_inspection (no blank line between files there).
    """Format project context files using Pi's XML-like wrapper."""
    if not context_files:
        return ""

    lines = [
        "\n\n<project_context>",
        "",
        "Project-specific instructions and guidelines:",
        "",
    ]
    for context_file in context_files:
        lines.append(f'<project_instructions path="{escape(context_file.path)}">')
        lines.append(context_file.content)
        lines.append("</project_instructions>")
        lines.append("")
    lines.append("</project_context>")
    return "\n".join(lines)


def _project_context_sources(
    context_files: Sequence[ProjectContextFile],
) -> tuple[SystemPromptSource, ...]:
    # [FUNCTION DOC]
    # HELPER: Build one "context" source per project file (label "Project instructions",
    # provenance = file path). The first carries the opening wrapper and intro text, the
    # last carries the closing wrapper; each contains a project_instructions element with
    # an escaped path attribute and the raw content. Returns an empty tuple if no files.
    if not context_files:
        return ()
    sources: list[SystemPromptSource] = []
    for index, context_file in enumerate(context_files):
        prefix = (
            "\n\n<project_context>\n\nProject-specific instructions and guidelines:\n\n"
            if index == 0
            else ""
        )
        suffix = "\n</project_context>" if index == len(context_files) - 1 else "\n"
        sources.append(
            SystemPromptSource(
                kind="context",
                label="Project instructions",
                source=context_file.path,
                content=(
                    f'{prefix}<project_instructions path="{escape(context_file.path)}">\n'
                    f"{context_file.content}\n</project_instructions>{suffix}"
                ),
            )
        )
    return tuple(sources)


def _skill_sources(skills: Sequence[Skill]) -> tuple[SystemPromptSource, ...]:
    # [FUNCTION DOC]
    # HELPER: Build one "skill" source per skill not marked disable-model-invocation, sorted
    # by name. The first source carries the explanatory intro and the opening
    # available_skills tag, each contains a skill element with escaped name, description
    # ("No description" if empty) and file location, and the last closes the tag. Label is
    # "Skill: <name>", provenance the skill path. Returns empty tuple if none are visible.
    visible_skills = sorted(
        (skill for skill in skills if not skill.disable_model_invocation),
        key=lambda item: item.name,
    )
    if not visible_skills:
        return ()
    prefix = (
        "\n\nThe following skills provide specialized instructions for specific tasks.\n"
        "Read the full skill file when the task matches its description.\n"
        "When a skill file references a relative path, resolve it against the skill directory "
        "(parent of SKILL.md / dirname of the path) and use that absolute path in tool "
        "commands.\n\n"
        "<available_skills>\n"
    )
    sources: list[SystemPromptSource] = []
    for index, skill in enumerate(visible_skills):
        description = skill.description or "No description"
        entry = "\n".join(
            (
                "  <skill>",
                f"    <name>{escape(skill.name)}</name>",
                f"    <description>{escape(description)}</description>",
                f"    <location>{escape(str(skill.path))}</location>",
                "  </skill>",
            )
        )
        sources.append(
            SystemPromptSource(
                kind="skill",
                label=f"Skill: {skill.name}",
                source=str(skill.path),
                content=(prefix if index == 0 else "\n")
                + entry
                + ("\n</available_skills>" if index == len(visible_skills) - 1 else ""),
            )
        )
    return tuple(sources)


def format_skills_for_prompt(skills: Sequence[Skill]) -> str:
    # [FUNCTION DOC]
    # Standalone formatter for skills in Pi's XML style: excludes skills with model
    # invocation disabled (they stay usable via the /skill: command), returns "" if none,
    # otherwise the intro text and available_skills block with entries sorted by name,
    # matching the content produced by _skill_sources.
    """Format skills for inclusion in a system prompt using Pi's XML style.

    Skills with ``disable_model_invocation`` set are excluded from the prompt;
    they remain invocable explicitly via ``/skill:<name>``.
    """
    visible_skills = [skill for skill in skills if not skill.disable_model_invocation]
    if not visible_skills:
        return ""

    lines = [
        "\n\nThe following skills provide specialized instructions for specific tasks.",
        "Read the full skill file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill directory "
        "(parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
        "",
        "<available_skills>",
    ]
    for skill in sorted(visible_skills, key=lambda item: item.name):
        description = skill.description or "No description"
        lines.extend(
            [
                "  <skill>",
                f"    <name>{escape(skill.name)}</name>",
                f"    <description>{escape(description)}</description>",
                f"    <location>{escape(str(skill.path))}</location>",
                "  </skill>",
            ]
        )
    lines.append("</available_skills>")
    return "\n".join(lines)


def _has_tool(tools: Sequence[AgentTool], name: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: Return True if any tool has exactly the given name.
    return any(tool.name == name for tool in tools)


def _format_path(path: Path) -> str:
    # [FUNCTION DOC]
    # HELPER: Convert a path to a string with backslashes replaced by forward slashes.
    return str(path).replace("\\", "/")
