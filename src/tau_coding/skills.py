# [MODULE DOC]
# Markdown skill discovery, loading and invocation expansion for the coding
# app. A skill is a directory containing SKILL.md (Agent Skills spec); bare .md
# files in a skills directory are reported with an info diagnostic instead
# (diverges from Pi; see ADR 0003). Provides loading across precedence-ordered
# resource directories, "/skill:name" command expansion into a tagged prompt
# block, parsing of that block back, and a skill index for the system prompt.
"""Markdown skill loading and expansion."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tau_coding.resources import (
    ResourceDiagnostic,
    ResourceError,
    TauResourcePaths,
    derive_description,
    parse_markdown_resource,
)


@dataclass(frozen=True, slots=True)
class Skill:
    # [CLASS DOC]
    # Immutable skill record: name (directory name), path to SKILL.md, markdown
    # body content (front matter removed), optional description, and a flag that
    # hides the skill from the model-facing index.
    """A markdown skill resource."""

    name: str
    path: Path
    content: str
    description: str | None = None
    disable_model_invocation: bool = False


def is_skill_candidate(path: Path) -> bool:
    # [FUNCTION DOC]
    # Returns true when the path is named exactly SKILL.md and is a file or a
    # symlink (possibly dangling). Side effect: filesystem stat.
    """Return whether an entry is a loader-eligible ``*/SKILL.md`` candidate."""
    return path.name == "SKILL.md" and (path.is_file() or path.is_symlink())


@dataclass(frozen=True, slots=True)
class SkillInvocation:
    # [CLASS DOC]
    # Immutable parsed form of an expanded skill invocation: skill name, location
    # (path string), content and optional additional user instructions.
    """Parsed expanded skill invocation message."""

    name: str
    location: str
    content: str
    additional_instructions: str | None = None


def load_skills(paths: TauResourcePaths | None = None) -> list[Skill]:
    # [FUNCTION DOC]
    # Loads skills from each skills directory of the resource paths (defaults to
    # default resource paths) in increasing precedence, later directories
    # replacing same-named skills silently.
    # Output: skills sorted by name. Raises ResourceError on any non-info
    # diagnostic from a directory (for example a read error or duplicate).
    # Side effects: filesystem reads.
    """Load markdown skills from Tau and `.agents` resource directories.

    Resource directories are loaded in increasing precedence order, so project
    resources override user resources with the same skill name. Duplicate names
    within the same directory remain invalid.
    """
    resource_paths = paths or TauResourcePaths()
    skills_by_name: dict[str, Skill] = {}

    for skills_dir in resource_paths.skills_dirs:
        for skill in _load_skills_from_dir(skills_dir):
            skills_by_name[skill.name] = skill

    return sorted(skills_by_name.values(), key=lambda skill: skill.name)


def load_skills_with_diagnostics(
    paths: TauResourcePaths | None = None,
) -> tuple[list[Skill], list[ResourceDiagnostic]]:
    # [FUNCTION DOC]
    # Like load_skills but never raises for per-skill problems: collects every
    # directory diagnostic and adds an "overrides lower-precedence resource at
    # <path>" diagnostic whenever a later directory replaces a skill.
    # Output: a pair of skills sorted by name and the diagnostics list.
    # Side effects: filesystem reads.
    """Load skills and return non-fatal discovery diagnostics.

    Resource directories are loaded in increasing precedence order. Higher
    precedence resources replace lower precedence resources with the same name,
    and that replacement is reported as a diagnostic.
    """
    resource_paths = paths or TauResourcePaths()
    skills_by_name: dict[str, Skill] = {}
    diagnostics: list[ResourceDiagnostic] = []

    for skills_dir in resource_paths.skills_dirs:
        skills, directory_diagnostics = _load_skills_from_dir_with_diagnostics(skills_dir)
        diagnostics.extend(directory_diagnostics)
        for skill in skills:
            previous = skills_by_name.get(skill.name)
            if previous is not None:
                diagnostics.append(
                    ResourceDiagnostic(
                        kind="skill",
                        name=skill.name,
                        path=skill.path,
                        message=f"overrides lower-precedence resource at {previous.path}",
                    )
                )
            skills_by_name[skill.name] = skill

    return sorted(skills_by_name.values(), key=lambda skill: skill.name), diagnostics


def expand_skill_command(text: str, skills: Sequence[Skill]) -> str | None:
    # [FUNCTION DOC]
    # Expands prompt text of the form "/skill:name optional request".
    # Inputs: the raw text and available skills. Output: none when the stripped
    # text does not start with "/skill:", else the formatted invocation with the
    # remaining text (stripped) as additional instructions.
    # Raises ResourceError when the name is empty or no skill has that name.
    """Expand `/skill:name` prompt text, or return None for non-skill text."""
    stripped = text.strip()
    if not stripped.startswith("/skill:"):
        return None

    command_and_request = stripped.split(maxsplit=1)
    command = command_and_request[0]
    request = command_and_request[1] if len(command_and_request) > 1 else None
    name = command.removeprefix("/skill:").strip()
    if not name:
        raise ResourceError("Skill command must include a skill name")

    skill_by_name = {skill.name: skill for skill in skills}
    skill = skill_by_name.get(name)
    if skill is None:
        raise ResourceError(f"Unknown skill: {name}")

    additional_instructions = request.strip() if request is not None else None
    return format_skill_invocation(skill, additional_instructions)


def format_skill_invocation(
    skill: Skill,
    additional_instructions: str | None = None,
) -> str:
    # [FUNCTION DOC]
    # Builds the invocation prompt: a skill tag with name and location attributes,
    # a line stating references are relative to the skill's directory, a blank
    # line, the stripped skill content and a closing tag; non-blank additional
    # instructions (stripped) are appended after a blank line. Output: the string.
    """Format a full skill invocation prompt."""
    skill_block = (
        f'<skill name="{skill.name}" location="{skill.path}">\n'
        f"References are relative to {skill.path.parent}.\n\n"
        f"{skill.content.strip()}\n"
        "</skill>"
    )
    if additional_instructions and additional_instructions.strip():
        return f"{skill_block}\n\n{additional_instructions.strip()}"
    return skill_block


def parse_skill_invocation(text: str) -> SkillInvocation | None:
    # [FUNCTION DOC]
    # Parses text produced by format_skill_invocation using an anchored pattern.
    # Output: a SkillInvocation (content includes the "References are relative"
    # line; additional instructions none when absent) or none if not matching.
    """Parse Tau's expanded skill invocation message format."""
    match = re.match(
        r'^<skill name="([^"]+)" location="([^"]+)">\n([\s\S]*?)\n</skill>(?:\n\n([\s\S]+))?$',
        text,
    )
    if match is None:
        return None
    name, location, content, additional_instructions = match.groups()
    return SkillInvocation(
        name=name,
        location=location,
        content=content,
        additional_instructions=additional_instructions,
    )


def build_skill_index(skills: Sequence[Skill]) -> str:
    # [FUNCTION DOC]
    # Builds the "Available skills:" list for the system prompt, one line
    # "- name: description" per skill sorted by name, excluding skills with model
    # invocation disabled and using "No description" as fallback. Returns
    # "Available skills: none" when nothing is visible.
    """Build a concise index of available skills for future system prompt assembly."""
    visible_skills = [skill for skill in skills if not skill.disable_model_invocation]
    if not visible_skills:
        return "Available skills: none"
    lines = ["Available skills:"]
    for skill in sorted(visible_skills, key=lambda item: item.name):
        description = skill.description or "No description"
        lines.append(f"- {skill.name}: {description}")
    return "\n".join(lines)


def _load_skills_from_dir(skills_dir: Path) -> list[Skill]:
    # [FUNCTION DOC]
    # HELPER: loads one directory, ignoring info diagnostics and raising
    # ResourceError with the message of the first other diagnostic.
    # Output: the loaded skills.
    skills, diagnostics = _load_skills_from_dir_with_diagnostics(skills_dir)
    for diagnostic in diagnostics:
        # Bare-.md migration hints are informational — the file is skipped,
        # but that is not an error. Only fatal problems raise here; the full
        # diagnostic stream is available through ``load_skills_with_diagnostics``.
        if diagnostic.severity == "info":
            continue
        raise ResourceError(diagnostic.message)
    return skills


def _load_skills_from_dir_with_diagnostics(
    skills_dir: Path,
) -> tuple[list[Skill], list[ResourceDiagnostic]]:
    # [FUNCTION DOC]
    # HELPER: scans one directory (missing or non-directory yields nothing) in name
    # order. Subdirectories containing SKILL.md become skills named after the
    # directory; subdirectories without it are skipped. Root .md files other than
    # AGENTS.md produce an info diagnostic with a migration hint. Other entries are
    # ignored. Duplicate names produce a diagnostic and are skipped; read or decode
    # errors produce an error diagnostic.
    # Output: a pair of skills and diagnostics. Side effects: filesystem reads.
    # KNOWN ISSUE: names come from unique directory entries so the duplicate check
    # can effectively never fire (except on case-insensitive quirks), and the
    # existence check differs from is_skill_candidate (a dangling SKILL.md symlink
    # is a candidate there but skipped here).
    if not skills_dir.exists() or not skills_dir.is_dir():
        return [], []

    # Tau follows the Agent Skills spec everywhere: a skill is a directory
    # containing a ``SKILL.md`` file. Bare ``*.md`` files at the root of a
    # skills directory are not skills and are surfaced as a diagnostic that
    # tells the user how to migrate. This intentionally diverges from Pi,
    # which keeps a permissive "any .md is a skill" path in its own
    # ``.pi/skills/`` and ``~/.pi/agent/skills/`` folders purely for
    # backward compatibility. See ADR 0003.
    skills: list[Skill] = []
    diagnostics: list[ResourceDiagnostic] = []
    seen: set[str] = set()
    for path in sorted(skills_dir.iterdir(), key=lambda item: item.name):
        skill_path: Path | None = None
        name = path.stem
        if path.is_dir():
            skill_path = path / "SKILL.md"
            name = path.name
            if not skill_path.exists():
                continue
        elif path.is_file() and path.suffix.lower() == ".md":
            if path.name.upper() == "AGENTS.MD":
                continue
            diagnostics.append(
                ResourceDiagnostic(
                    kind="skill",
                    name=path.stem,
                    path=path,
                    message=(
                        "bare .md files are no longer treated as skills; "
                        f"move it to {(path.parent / path.stem / 'SKILL.md')}"
                    ),
                    severity="info",
                )
            )
            continue
        else:
            continue

        if name in seen:
            diagnostics.append(
                ResourceDiagnostic(
                    kind="skill",
                    name=name,
                    path=skill_path,
                    message=f"Duplicate skill name ignored in {skills_dir}",
                )
            )
            continue
        seen.add(name)
        try:
            skills.append(_load_skill(name, skill_path))
        except (OSError, UnicodeDecodeError) as exc:
            diagnostics.append(
                ResourceDiagnostic(
                    kind="skill",
                    name=name,
                    path=skill_path,
                    message=f"could not read skill: {exc}",
                    severity="error",
                )
            )
    return skills, diagnostics


def _load_skill(name: str, path: Path) -> Skill:
    # [FUNCTION DOC]
    # HELPER: reads a SKILL.md file as UTF-8, splits front matter from content,
    # uses the "description" metadata or a description derived from the content,
    # and sets the disable flag when "disable-model-invocation" equals "true"
    # (case-insensitive). Output: a Skill. Raises OSError or UnicodeDecodeError on
    # read failures.
    raw = path.read_text(encoding="utf-8")
    metadata, content = parse_markdown_resource(raw)
    description = metadata.get("description") or derive_description(content)
    disable_model_invocation = (
        metadata.get("disable-model-invocation", "").strip().lower() == "true"
    )
    return Skill(
        name=name,
        path=path,
        content=content,
        description=description,
        disable_model_invocation=disable_model_invocation,
    )
