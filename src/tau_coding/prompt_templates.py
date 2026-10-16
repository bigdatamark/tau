# [MODULE DOC]
# Markdown prompt templates: discovery from Tau and .agents prompt directories
# (later directories override earlier ones by name), front-matter parsing for
# descriptions, Pi-compatible argument parsing and substitution ($1, $@,
# $ARGUMENTS, defaults like ${1:-x}, slices like ${@:2:3}), legacy double-brace
# variable rendering, and expansion of "/name args" input into prompt text.
# Template names that collide with built-in commands (prompts, skills, tools,
# reload) are reserved and ignored.
"""Markdown prompt template loading and rendering."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tau_coding.resources import (
    ResourceDiagnostic,
    ResourceError,
    TauResourcePaths,
    derive_description,
    parse_markdown_resource,
)

_TEMPLATE_VARIABLE_RE = re.compile(r"{{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*}}")
_PROMPT_ARGUMENT_RE = re.compile(
    r"\$\{(\d+|ARGUMENTS|@):-([^}]*)\}|\$\{@:(\d+)(?::(\d+))?\}|\$(ARGUMENTS|@|\d+)"
)
_ARGUMENT_TEMPLATE_VARIABLES = {"arguments", "args"}
_RESERVED_TEMPLATE_NAMES = frozenset({"prompts", "skills", "tools", "reload"})


def is_prompt_template_candidate(path: Path) -> bool:
    # [FUNCTION DOC]
    # Return true when the path has a .md suffix (case-insensitive) and its stem,
    # case-folded, is not a reserved built-in command name. Pure.
    """Return whether a directory entry is eligible for prompt loading."""
    return path.suffix.lower() == ".md" and path.stem.casefold() not in _RESERVED_TEMPLATE_NAMES


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    # [CLASS DOC]
    # Frozen record of a template: name (file stem), source path, body content
    # (front matter removed) and optional description.
    """A markdown prompt template resource."""

    name: str
    path: Path
    content: str
    description: str | None = None


def load_prompt_templates(paths: TauResourcePaths | None = None) -> list[PromptTemplate]:
    # [FUNCTION DOC]
    # Strict loader: for each prompt directory in precedence order, load templates
    # (raising ResourceError with the first diagnostic of any directory), letting
    # later directories override by name. Returns templates sorted by name.
    # Side effects: filesystem reads.
    # KNOWN ISSUE: any diagnostic, including the non-fatal reserved-name or duplicate
    # warnings, aborts the whole load with ResourceError.
    """Load markdown prompt templates from Tau and `.agents` resource directories."""
    resource_paths = paths or TauResourcePaths()
    templates_by_name: dict[str, PromptTemplate] = {}
    for prompts_dir in resource_paths.prompts_dirs:
        for template in _load_prompt_templates_from_dir(prompts_dir):
            templates_by_name[template.name] = template
    return sorted(templates_by_name.values(), key=lambda template: template.name)


def load_prompt_templates_with_diagnostics(
    paths: TauResourcePaths | None = None,
) -> tuple[list[PromptTemplate], list[ResourceDiagnostic]]:
    # [FUNCTION DOC]
    # Lenient loader: same precedence/override behavior but collects per-directory
    # diagnostics and adds an "overrides lower-precedence resource" diagnostic when
    # a later template replaces an earlier one. Returns the name-sorted templates and
    # the diagnostics list. Never raises for unreadable files.
    """Load prompt templates and return non-fatal discovery diagnostics."""
    resource_paths = paths or TauResourcePaths()
    templates_by_name: dict[str, PromptTemplate] = {}
    diagnostics: list[ResourceDiagnostic] = []
    for prompts_dir in resource_paths.prompts_dirs:
        templates, directory_diagnostics = _load_prompt_templates_from_dir_with_diagnostics(
            prompts_dir
        )
        diagnostics.extend(directory_diagnostics)
        for template in templates:
            previous = templates_by_name.get(template.name)
            if previous is not None:
                diagnostics.append(
                    ResourceDiagnostic(
                        kind="prompt",
                        name=template.name,
                        path=template.path,
                        message=f"overrides lower-precedence resource at {previous.path}",
                    )
                )
            templates_by_name[template.name] = template
    return sorted(templates_by_name.values(), key=lambda template: template.name), diagnostics


def parse_prompt_template_arguments(text: str) -> list[str]:
    # [FUNCTION DOC]
    # Split invocation text into arguments on whitespace, honoring single or double
    # quotes (quote characters are removed, no escapes, an unterminated quote runs
    # to the end). Empty quoted strings are dropped. Returns the list of arguments.
    """Parse prompt invocation arguments using Pi's simple quote rules."""
    arguments: list[str] = []
    current: list[str] = []
    quote: str | None = None

    for character in text:
        if quote is not None:
            if character == quote:
                quote = None
            else:
                current.append(character)
        elif character in {"'", '"'}:
            quote = character
        elif character.isspace():
            if current:
                arguments.append("".join(current))
                current = []
        else:
            current.append(character)

    if current:
        arguments.append("".join(current))
    return arguments


def substitute_prompt_template_args(content: str, arguments: Sequence[str]) -> str:
    # [FUNCTION DOC]
    # Replace argument placeholders in content: ${N:-default} / ${@:-default} /
    # ${ARGUMENTS:-default} use the value or the default when empty; ${@:start} and
    # ${@:start:length} join a 1-based slice; $N gives the Nth argument (empty if
    # absent); $@ and $ARGUMENTS give all arguments joined by spaces. Returns text.
    """Substitute Pi-compatible positional and aggregate prompt arguments."""
    all_arguments = " ".join(arguments)

    def replace(match: re.Match[str]) -> str:
        # [FUNCTION DOC]
        # HELPER: regex callback computing the replacement for one placeholder match
        # according to which alternative group matched (default, slice, or simple).
        default_target = match.group(1)
        if default_target is not None:
            default_value = match.group(2) or ""
            if default_target in {"@", "ARGUMENTS"}:
                value = all_arguments
            else:
                index = int(default_target) - 1
                value = arguments[index] if 0 <= index < len(arguments) else ""
            return value or default_value

        slice_start = match.group(3)
        if slice_start is not None:
            start = max(int(slice_start) - 1, 0)
            slice_length = match.group(4)
            if slice_length is None:
                return " ".join(arguments[start:])
            return " ".join(arguments[start : start + int(slice_length)])

        simple = match.group(5) or ""
        if simple in {"@", "ARGUMENTS"}:
            return all_arguments
        index = int(simple) - 1
        return arguments[index] if 0 <= index < len(arguments) else ""

    return _PROMPT_ARGUMENT_RE.sub(replace, content)


def render_prompt_template(
    template: PromptTemplate,
    variables: Mapping[str, str],
    *,
    missing: str | None = None,
) -> str:
    # [FUNCTION DOC]
    # Replace double-brace identifiers in the template content with values from the
    # mapping. Missing variables raise ResourceError unless a missing fallback string
    # is supplied, which is substituted instead. Returns the rendered text.
    """Render a prompt template using `{{ variable }}` placeholders.

    By default, missing variables raise `ResourceError`. Callers that treat
    templates as user-facing shortcuts can pass `missing` to render absent
    variables as a fallback string instead.
    """

    def replace(match: re.Match[str]) -> str:
        # [FUNCTION DOC]
        # HELPER: regex callback looking up one variable, raising or using the fallback.
        name = match.group(1)
        value = variables.get(name)
        if value is None:
            if missing is None:
                raise ResourceError(f"Missing prompt template variable: {name}")
            return missing
        return value

    return _TEMPLATE_VARIABLE_RE.sub(replace, template.content)


def expand_prompt_template_command(
    text: str,
    templates: Sequence[PromptTemplate],
) -> str | None:
    # [FUNCTION DOC]
    # If the stripped text is a slash command (not "//" escaped and not /skill:),
    # parse the name and argument text, find a template by case-insensitive name,
    # and return its content with argument placeholders substituted, then legacy
    # arguments/args variables rendered (other unknown variables become empty).
    # If arguments were given but the template references none, the raw argument
    # text is appended after a blank line. Returns nothing when not a matching
    # template command. Pure.
    # KNOWN ISSUE: substitution happens before variable rendering, so argument values
    # containing double-brace placeholders are themselves rendered (usually erased).
    """Expand `/name [arguments]` text with a loaded prompt template.

    Template names are matched by markdown filename stem. Invocation arguments use
    Pi-compatible `$1`, `$2`, `$@`, and `$ARGUMENTS` placeholders, including default
    values and simple slices. Legacy `{{ arguments }}` and `{{ args }}` placeholders
    remain supported. If a template has no argument placeholder, arguments are
    appended after a blank line.
    """
    stripped = text.strip()
    if not stripped.startswith("/") or stripped.startswith("//") or stripped.startswith("/skill:"):
        return None

    name, args = _parse_prompt_template_command(stripped)
    if not name:
        return None

    template = _find_prompt_template(name, templates)
    if template is None:
        return None

    arguments = parse_prompt_template_arguments(args)
    argument_text = " ".join(arguments)
    rendered = substitute_prompt_template_args(template.content, arguments)
    rendered = render_prompt_template(
        PromptTemplate(template.name, template.path, rendered, template.description),
        {"arguments": argument_text, "args": argument_text},
        missing="",
    )
    if arguments and not _template_references_arguments(template.content):
        return f"{rendered.rstrip()}\n\n{args}"
    return rendered


def _template_references_arguments(content: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: true if content contains any Pi argument placeholder or a legacy
    # arguments/args double-brace variable.
    return bool(_PROMPT_ARGUMENT_RE.search(content)) or any(
        match.group(1) in _ARGUMENT_TEMPLATE_VARIABLES
        for match in _TEMPLATE_VARIABLE_RE.finditer(content)
    )


def _find_prompt_template(
    name: str,
    templates: Sequence[PromptTemplate],
) -> PromptTemplate | None:
    # [FUNCTION DOC]
    # HELPER: return the first template whose lowercased name equals the given name
    # stripped, without a leading slash and lowercased; nothing if none matches.
    normalized_name = name.strip().removeprefix("/").lower()
    for template in templates:
        if template.name.lower() == normalized_name:
            return template
    return None


def _parse_prompt_template_command(text: str) -> tuple[str, str]:
    # [FUNCTION DOC]
    # HELPER: split "/name rest" into the lowercased name and the stripped rest;
    # returns two empty strings when the text does not match.
    match = re.match(r"^/([^\s]+)(?:\s+([\s\S]*))?$", text)
    if match is None:
        return "", ""
    return match.group(1).lower(), (match.group(2) or "").strip()


def _load_prompt_templates_from_dir(prompts_dir: Path) -> list[PromptTemplate]:
    # [FUNCTION DOC]
    # HELPER: strict per-directory load; raises ResourceError with the first
    # diagnostic's message if any diagnostics exist, else returns the templates.
    templates, diagnostics = _load_prompt_templates_from_dir_with_diagnostics(prompts_dir)
    if diagnostics:
        first = diagnostics[0]
        raise ResourceError(first.message)
    return templates


def _load_prompt_templates_from_dir_with_diagnostics(
    prompts_dir: Path,
) -> tuple[list[PromptTemplate], list[ResourceDiagnostic]]:
    # [FUNCTION DOC]
    # HELPER: if the directory is missing or not a directory, return empty lists.
    # Otherwise iterate top-level .md files sorted by filename: reserved names yield
    # a diagnostic and are skipped; duplicate stems are skipped with a diagnostic;
    # readable files are loaded; OSError or decode errors produce error-severity
    # diagnostics. Returns the templates and diagnostics.
    if not prompts_dir.exists() or not prompts_dir.is_dir():
        return [], []

    templates: list[PromptTemplate] = []
    diagnostics: list[ResourceDiagnostic] = []
    seen: set[str] = set()
    for path in sorted(prompts_dir.glob("*.md"), key=lambda item: item.name):
        name = path.stem
        if not is_prompt_template_candidate(path):
            diagnostics.append(
                ResourceDiagnostic(
                    kind="prompt",
                    name=name,
                    path=path,
                    message=(
                        f"prompt template name is reserved by the built-in /{name.casefold()} "
                        "command; template ignored"
                    ),
                )
            )
            continue
        if name in seen:
            diagnostics.append(
                ResourceDiagnostic(
                    kind="prompt",
                    name=name,
                    path=path,
                    message=f"Duplicate prompt template name ignored in {prompts_dir}",
                )
            )
            continue
        seen.add(name)
        try:
            templates.append(_load_prompt_template(name, path))
        except (OSError, UnicodeDecodeError) as exc:
            diagnostics.append(
                ResourceDiagnostic(
                    kind="prompt",
                    name=name,
                    path=path,
                    message=f"could not read prompt template: {exc}",
                    severity="error",
                )
            )
    return templates, diagnostics


def _load_prompt_template(name: str, path: Path) -> PromptTemplate:
    # [FUNCTION DOC]
    # HELPER: read the file as UTF-8, parse front matter, use the description field
    # or a description derived from the body, and return the template. Raises
    # OSError or UnicodeDecodeError on read failure.
    raw = path.read_text(encoding="utf-8")
    metadata, content = parse_markdown_resource(raw)
    description = metadata.get("description") or derive_description(content)
    return PromptTemplate(name=name, path=path, content=content, description=description)
