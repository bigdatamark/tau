# [MODULE DOC]
# Prompt autocomplete logic for the Textual TUI, kept free of widget code so
# it is unit-testable. Given the prompt text and cursor, produces completion
# items for slash commands (and aliases/search terms), prompt templates,
# /skill:<name> skills, command arguments (models, providers, sessions,
# themes), @ file references relative to the working directory, and shell
# paths after a leading "!" or "!!". Also defines the immutable completion
# state with wrap-around selection. Reads the filesystem for path
# completions; ignores common VCS/cache/build directories and caps file
# suggestions at 50.
"""Prompt autocomplete helpers for Tau's Textual TUI."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tau_coding.commands import CommandRegistry, SlashCommand
from tau_coding.prompt_templates import PromptTemplate
from tau_coding.skills import Skill

IGNORED_FILE_COMPLETION_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tau",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)
MAX_FILE_COMPLETIONS = 50


@dataclass(frozen=True, slots=True)
class CompletionOption:
    # [CLASS DOC]
    # Frozen dataclass: an argument completion value plus an optional description
    # shown in the picker.
    """A possible argument completion value with optional picker metadata."""

    value: str
    description: str | None = None


class CompletionKind(StrEnum):
    # [CLASS DOC]
    # String enum naming where a completion came from (command, prompt template,
    # skill, argument, file reference, shell path); the TUI uses it to decide
    # Enter-key behavior.
    """Source of a prompt completion and its Enter-key behavior."""

    COMMAND = "command"
    PROMPT_TEMPLATE = "prompt_template"
    SKILL = "skill"
    ARGUMENT = "argument"
    FILE_REFERENCE = "file_reference"
    SHELL_PATH = "shell_path"


@dataclass(frozen=True, slots=True)
class CompletionItem:
    # [CLASS DOC]
    # Frozen dataclass for one selectable completion: display text, replacement
    # text, the start/end character span of the input it replaces, kind, and
    # optional description and category.
    """One selectable prompt completion."""

    display: str
    replacement: str
    start: int
    end: int
    kind: CompletionKind
    description: str | None = None
    category: str | None = None

    def apply(self, text: str) -> str:
        # [FUNCTION DOC]
        # Return the input text with the span start..end replaced by the
        # replacement. Pure.
        """Apply this completion to input text."""
        return f"{text[: self.start]}{self.replacement}{text[self.end :]}"

    def cursor_after_apply(self) -> int:
        # [FUNCTION DOC]
        # Return the cursor offset right after the inserted replacement.
        """Return the cursor offset just after the applied replacement."""
        return self.start + len(self.replacement)


@dataclass(frozen=True, slots=True)
class CompletionState:
    # [CLASS DOC]
    # Frozen dataclass holding the tuple of completion items and the selected
    # index (default 0). Selection changes return new states.
    """Current autocomplete state for the prompt input."""

    items: tuple[CompletionItem, ...] = ()
    selected_index: int = 0

    @property
    def selected(self) -> CompletionItem | None:
        # [FUNCTION DOC]
        # Property: the item at the selected index, or None when there are no items.
        # May raise IndexError if constructed with an out-of-range index.
        """Return the currently selected completion item."""
        if not self.items:
            return None
        return self.items[self.selected_index]

    def select_next(self) -> CompletionState:
        # [FUNCTION DOC]
        # Return a new state selecting the next item, wrapping to the first; returns
        # self unchanged when empty.
        """Return a state with the next item selected."""
        if not self.items:
            return self
        return CompletionState(
            items=self.items,
            selected_index=(self.selected_index + 1) % len(self.items),
        )

    def select_previous(self) -> CompletionState:
        # [FUNCTION DOC]
        # Return a new state selecting the previous item, wrapping to the last;
        # returns self unchanged when empty.
        """Return a state with the previous item selected."""
        if not self.items:
            return self
        return CompletionState(
            items=self.items,
            selected_index=(self.selected_index - 1) % len(self.items),
        )


def build_completion_state(
    text: str,
    *,
    cursor: int | None = None,
    command_registry: CommandRegistry,
    skills: Sequence[Skill],
    prompt_templates: Sequence[PromptTemplate],
    model_names: Sequence[str] = (),
    provider_names: Sequence[str] = (),
    thinking_levels: Sequence[str] = (),
    theme_names: Sequence[str] = (),
    session_ids: Sequence[str] = (),
    session_options: Sequence[CompletionOption] = (),
    cwd: Path | None = None,
) -> CompletionState:
    # [FUNCTION DOC]
    # Compute the completion state for the prompt. Inputs: text, optional cursor
    # (clamped; default end of text), command registry, skills, prompt
    # templates, and candidate model/provider/thinking-level/theme/session values,
    # optional rich session options, and optional working directory.
    # Dispatch: text not starting with a single "/" -> shell path completions if
    # it is a "!" command, else @ file reference completions (nothing if no cwd).
    # For slash text, the first token (up to the first space) decides:
    # "/skill:..." -> file references when a known skill already has argument
    # text, else skill name completions; any other token containing ":" ->
    # nothing; then command argument completions for model/scoped-models,
    # login/logout, resume and theme; then file references after a known prompt
    # template with arguments; nothing after a known registered command with
    # arguments; otherwise command and prompt template name completions.
    # Output: CompletionState. Side effects: filesystem reads for path modes.
    # Note: thinking levels are accepted but currently not used for any command.
    """Build autocomplete suggestions for the current prompt text."""
    cursor = len(text) if cursor is None else max(0, min(cursor, len(text)))
    if not text.startswith("/") or text.startswith("//"):
        if cwd is not None:
            shell_completions = _shell_path_completions(text=text, cursor=cursor, cwd=cwd)
            if shell_completions is not None:
                return CompletionState(shell_completions)
            return CompletionState(_file_reference_completions(text=text, cursor=cursor, cwd=cwd))
        return CompletionState()

    token_end = _first_token_end(text)
    token = text[:token_end]
    has_argument_text = token_end < len(text)
    if token.startswith("/skill:"):
        if has_argument_text and _matches_skill_command(token, skills):
            # Skill arguments are prompt text, so @ file references stay available.
            if cwd is not None:
                return CompletionState(
                    _file_reference_completions(text=text, cursor=cursor, cwd=cwd)
                )
            return CompletionState()
        return CompletionState(_skill_completions(token=token, token_end=token_end, skills=skills))

    if ":" in token:
        return CompletionState()

    argument_completions = _command_argument_completions(
        text=text,
        token_end=token_end,
        model_names=model_names,
        provider_names=provider_names,
        thinking_levels=thinking_levels,
        theme_names=theme_names,
        session_ids=session_ids,
        session_options=session_options,
    )
    if argument_completions is not None:
        return CompletionState(argument_completions)

    if has_argument_text and _matches_prompt_template_command(token, prompt_templates):
        if cwd is not None:
            return CompletionState(_file_reference_completions(text=text, cursor=cursor, cwd=cwd))
        return CompletionState()

    if has_argument_text and _matches_registered_command(token, command_registry):
        return CompletionState()

    return CompletionState(
        _command_completions(
            token=token,
            token_end=token_end,
            registry=command_registry,
            prompt_templates=prompt_templates,
        )
    )


def _file_reference_completions(*, text: str, cursor: int, cwd: Path) -> tuple[CompletionItem, ...]:
    # [FUNCTION DOC]
    # HELPER: complete the "@" token under the cursor. Finds the active token;
    # delegates to parent-directory completion for prefixes starting with "../"
    # or equal to/ending in ".."; otherwise walks all non-ignored paths under cwd
    # and suggests "@<relative path>" (with a trailing slash for directories)
    # whose path contains the typed prefix case-insensitively, skipping the
    # exact existing text, up to the cap. Returns a tuple of items.
    # KNOWN ISSUE: the full recursive walk of cwd happens on every call before
    # filtering and capping, which can be slow in large trees.
    token = _active_file_reference_token(text, cursor)
    if token is None:
        return ()
    start, end = token
    prefix = text[start + 1 : cursor]
    existing = text[start:end]
    external_completions = _external_file_reference_completions(
        prefix=prefix,
        existing=existing,
        start=start,
        end=end,
        cwd=cwd,
    )
    if external_completions is not None:
        return external_completions

    suggestions: list[CompletionItem] = []
    for path in _iter_file_reference_paths(cwd):
        relative = path.relative_to(cwd).as_posix()
        if prefix.lower() not in relative.lower():
            continue
        display = f"@{relative}{'/' if path.is_dir() else ''}"
        if display == existing:
            continue
        suggestions.append(
            CompletionItem(
                display=display,
                replacement=display,
                start=start,
                end=end,
                kind=CompletionKind.FILE_REFERENCE,
                description="File reference",
            )
        )
        if len(suggestions) >= MAX_FILE_COMPLETIONS:
            break
    return tuple(suggestions)


def _external_file_reference_completions(
    *, prefix: str, existing: str, start: int, end: int, cwd: Path
) -> tuple[CompletionItem, ...] | None:
    # [FUNCTION DOC]
    # HELPER: handle "@" references that leave cwd. For ".." or ".../.." returns
    # a single "@<prefix>/" directory suggestion if it exists. For "../..."
    # prefixes lists the parent directory's children (sorted case-insensitively,
    # skipping ignored names and paths containing ignored parts) matching the
    # name prefix, up to the cap. Returns None when the prefix is not external
    # so the caller falls back; returns empty on missing dirs or OS errors.
    if prefix == ".." or prefix.endswith("/.."):
        target = cwd / prefix
        if not target.is_dir():
            return ()
        display = f"@{prefix}/"
        if display == existing:
            return ()
        return (
            CompletionItem(
                display=display,
                replacement=display,
                start=start,
                end=end,
                kind=CompletionKind.FILE_REFERENCE,
                description="File reference",
            ),
        )

    if not prefix.startswith("../"):
        return None

    parent_text, name_prefix = prefix.rsplit("/", 1)
    if any(part in IGNORED_FILE_COMPLETION_DIRS for part in Path(parent_text).parts):
        return ()
    parent_dir = cwd / parent_text
    if not parent_dir.is_dir():
        return ()

    try:
        children = sorted(parent_dir.iterdir(), key=lambda path: path.name.lower())
    except OSError:
        return ()

    suggestions: list[CompletionItem] = []
    for child in children:
        if child.name in IGNORED_FILE_COMPLETION_DIRS:
            continue
        if not child.name.lower().startswith(name_prefix.lower()):
            continue
        display = f"@{parent_text}/{child.name}{'/' if child.is_dir() else ''}"
        if display == existing:
            continue
        suggestions.append(
            CompletionItem(
                display=display,
                replacement=display,
                start=start,
                end=end,
                kind=CompletionKind.FILE_REFERENCE,
                description="File reference",
            )
        )
        if len(suggestions) >= MAX_FILE_COMPLETIONS:
            break
    return tuple(suggestions)


def _active_file_reference_token(text: str, cursor: int) -> tuple[int, int] | None:
    # [FUNCTION DOC]
    # HELPER: locate the "@" reference containing the cursor: searches the
    # current whitespace-delimited word (space or newline) before the cursor for
    # the last "@", and extends the end to the next whitespace. Returns the
    # start (index of "@") and end, or None if there is no "@".
    token_start = max(text.rfind(" ", 0, cursor), text.rfind("\n", 0, cursor)) + 1
    at_index = text.rfind("@", token_start, cursor)
    if at_index == -1:
        return None
    end = cursor
    while end < len(text) and not text[end].isspace():
        end += 1
    return at_index, end


def _iter_file_reference_paths(cwd: Path) -> tuple[Path, ...]:
    # [FUNCTION DOC]
    # HELPER: depth-first walk of cwd (directory children sorted
    # case-insensitively) collecting every file and directory not under an
    # ignored directory; unreadable directories are skipped. Returns empty if
    # cwd is missing or not a directory.
    if not cwd.exists() or not cwd.is_dir():
        return ()
    paths: list[Path] = []
    stack = [cwd]
    while stack:
        directory = stack.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name.lower())
        except OSError:
            continue
        for child in children:
            if _is_ignored_file_completion_path(child, cwd=cwd):
                continue
            paths.append(child)
            if child.is_dir():
                stack.append(child)
    return tuple(paths)


def _is_ignored_file_completion_path(path: Path, *, cwd: Path) -> bool:
    # [FUNCTION DOC]
    # HELPER: True when the path is outside cwd or any relative path part is an
    # ignored directory name.
    try:
        relative_parts = path.relative_to(cwd).parts
    except ValueError:
        return True
    return any(part in IGNORED_FILE_COMPLETION_DIRS for part in relative_parts)


def _shell_path_completions(
    *, text: str, cursor: int, cwd: Path
) -> tuple[CompletionItem, ...] | None:
    # [FUNCTION DOC]
    # HELPER: complete relative paths inside a "!" / "!!" shell command. Returns
    # None when the text is not a shell command (so the caller tries @ refs),
    # empty when the cursor is inside the prefix, the token is empty or not a
    # simple relative path, or the parent dir is missing/ignored. Otherwise lists
    # children of the parent directory matching the name prefix
    # case-insensitively, producing cwd-relative replacements (keeping a "./"
    # prefix, trailing slash for directories), skipping the unchanged token, with
    # a File/Directory description, up to the cap.
    prefix_span = _shell_command_prefix_span(text)
    if prefix_span is None:
        return None
    if cursor < prefix_span[1]:
        return ()

    start, end = _active_shell_path_token(text=text, cursor=cursor, command_start=prefix_span[1])
    token = text[start:cursor]
    if not token:
        return ()

    shell_path = _parse_shell_path_token(token)
    if shell_path is None:
        return ()
    parent_text, name_prefix, replacement_prefix = shell_path

    parent_dir = cwd / parent_text if parent_text else cwd
    if not parent_dir.exists() or not parent_dir.is_dir():
        return ()
    if parent_dir != cwd and _is_ignored_file_completion_path(parent_dir, cwd=cwd):
        return ()

    try:
        children = sorted(parent_dir.iterdir(), key=lambda path: path.name.lower())
    except OSError:
        return ()

    suggestions: list[CompletionItem] = []
    for child in children:
        if _is_ignored_file_completion_path(child, cwd=cwd):
            continue
        if not child.name.lower().startswith(name_prefix.lower()):
            continue
        relative = child.relative_to(cwd).as_posix()
        replacement = f"{replacement_prefix}{relative}{'/' if child.is_dir() else ''}"
        if replacement == text[start:end]:
            continue
        suggestions.append(
            CompletionItem(
                display=replacement,
                replacement=replacement,
                start=start,
                end=end,
                kind=CompletionKind.SHELL_PATH,
                description="Directory" if child.is_dir() else "File",
            )
        )
        if len(suggestions) >= MAX_FILE_COMPLETIONS:
            break
    return tuple(suggestions)


def _shell_command_prefix_span(text: str) -> tuple[int, int] | None:
    # [FUNCTION DOC]
    # HELPER: return the span of a leading "!!" or "!" after optional leading
    # whitespace, or None if the text is not a shell command.
    leading_whitespace = len(text) - len(text.lstrip())
    stripped = text[leading_whitespace:]
    if stripped.startswith("!!"):
        return (leading_whitespace, leading_whitespace + 2)
    if stripped.startswith("!"):
        return (leading_whitespace, leading_whitespace + 1)
    return None


def _active_shell_path_token(*, text: str, cursor: int, command_start: int) -> tuple[int, int]:
    # [FUNCTION DOC]
    # HELPER: find the shell word under the cursor: scans backward from the
    # cursor to the previous unescaped whitespace (not before the command
    # start), and forward via the token-end helper. Returns start and end.
    # KNOWN ISSUE: the backward scan treats the character after a backslash as
    # escaped when scanning right-to-left, so escape detection is approximate
    # (e.g. a backslash-escaped space is not recognized as escaped).
    token_start = command_start
    escaped = False
    for index in range(cursor - 1, command_start - 1, -1):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char.isspace():
            token_start = index + 1
            break
    return token_start, _shell_token_end(text, cursor)


def _shell_token_end(text: str, cursor: int) -> int:
    # [FUNCTION DOC]
    # HELPER: scan forward from the cursor to the next unescaped whitespace
    # (skipping backslash-escaped characters); return its index or text length.
    index = cursor
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char.isspace():
            return index
        index += 1
    return len(text)


def _parse_shell_path_token(token: str) -> tuple[str, str, str] | None:
    # [FUNCTION DOC]
    # HELPER: split a shell token into parent directory text, name prefix and a
    # replacement prefix ("./" if present). Returns None for absolute or home
    # paths, tokens with quotes, backticks, "$" or glob characters, and parent
    # paths containing empty, "." or ".." segments.
    replacement_prefix = ""
    path_text = token
    if path_text.startswith("./"):
        replacement_prefix = "./"
        path_text = path_text[2:]
    if path_text.startswith(("/", "~")):
        return None
    if any(char in path_text for char in "\"'`$*?[{"):
        return None

    parent_text, separator, name_prefix = path_text.rpartition("/")
    if separator and not parent_text:
        return None

    parent_parts = parent_text.split("/") if parent_text else []
    if any(part in {"", ".", ".."} for part in parent_parts):
        return None
    return parent_text, name_prefix, replacement_prefix


def _matches_skill_command(token: str, skills: Sequence[Skill]) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if "/skill:<name>" names a known skill (case-insensitive).
    command_name = token.removeprefix("/skill:").lower()
    return any(skill.name.lower() == command_name for skill in skills)


def _matches_prompt_template_command(
    token: str, prompt_templates: Sequence[PromptTemplate]
) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if "/<name>" names a known prompt template (case-insensitive).
    command_name = token.removeprefix("/").lower()
    return any(template.name.lower() == command_name for template in prompt_templates)


def _matches_registered_command(token: str, registry: CommandRegistry) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if the command registry has a command for "/<name>" (name
    # lower-cased before lookup).
    command_name = token.removeprefix("/").lower()
    return registry.get(command_name) is not None


def _command_completions(
    *,
    token: str,
    token_end: int,
    registry: CommandRegistry,
    prompt_templates: Sequence[PromptTemplate],
) -> tuple[CompletionItem, ...]:
    # [FUNCTION DOC]
    # HELPER: build slash-command suggestions (via alias completions for each
    # registered command) followed by prompt templates whose names start with the
    # typed prefix (case-insensitive), each group sorted with direct name matches
    # first then by display. Items replace from 0 to the token end.
    prefix = token.removeprefix("/").lower()
    command_suggestions: list[CompletionItem] = []
    for command in registry.list_commands():
        command_suggestions.extend(
            _command_alias_completions(command, prefix=prefix, token_end=token_end)
        )
    prompt_suggestions = [
        CompletionItem(
            display=f"/{template.name}",
            replacement=f"/{template.name}",
            start=0,
            end=token_end,
            kind=CompletionKind.PROMPT_TEMPLATE,
            description=template.description or "Prompt template",
            category="Custom prompts",
        )
        for template in prompt_templates
        if template.name.lower().startswith(prefix)
    ]
    return (
        *sorted(command_suggestions, key=lambda item: _command_completion_sort_key(item, prefix)),
        *sorted(prompt_suggestions, key=lambda item: _command_completion_sort_key(item, prefix)),
    )


def _command_completion_sort_key(item: CompletionItem, prefix: str) -> tuple[int, str]:
    # [FUNCTION DOC]
    # HELPER: sort key giving rank 0 to items whose display name (without slash
    # and trailing colon) starts with the prefix, else 1, then the display text.
    if not prefix:
        return (0, item.display)
    display_name = item.display.removeprefix("/").removesuffix(":").lower()
    direct_match_rank = 0 if display_name.startswith(prefix) else 1
    return (direct_match_rank, item.display)


def _command_alias_completions(
    command: SlashCommand, *, prefix: str, token_end: int
) -> list[CompletionItem]:
    # [FUNCTION DOC]
    # HELPER: suggestions for one command. With an empty prefix only the primary
    # name; otherwise the name, aliases and search terms that start with the
    # prefix. Search-term matches complete to the primary name; aliases complete
    # to themselves. The skill command completes to "/skill:". Duplicate
    # displays are removed. Category "Commands", description from the command.
    # KNOWN ISSUE: matching against the lower-cased prefix is case-sensitive on
    # the command side, so mixed-case names/aliases/search terms never match.
    names = (
        (command.name,) if not prefix else (command.name, *command.aliases, *command.search_terms)
    )
    suggestions: list[CompletionItem] = []
    seen: set[str] = set()
    for name in names:
        if not name.startswith(prefix):
            continue
        replacement_name = name if name in (command.name, *command.aliases) else command.name
        display = f"/{replacement_name}"
        replacement = f"/{replacement_name}"
        if command.name == "skill" and replacement_name == command.name:
            display = "/skill:"
            replacement = "/skill:"
        if display in seen:
            continue
        seen.add(display)
        suggestions.append(
            CompletionItem(
                display=display,
                replacement=replacement,
                start=0,
                end=token_end,
                kind=CompletionKind.COMMAND,
                description=command.description,
                category="Commands",
            )
        )
    return suggestions


def _skill_completions(
    *, token: str, token_end: int, skills: Sequence[Skill]
) -> tuple[CompletionItem, ...]:
    # [FUNCTION DOC]
    # HELPER: "/skill:<name>" suggestions for skills (sorted by name) whose names
    # start with the typed prefix case-insensitively.
    prefix = token.removeprefix("/skill:").lower()
    suggestions = [
        CompletionItem(
            display=f"/skill:{skill.name}",
            replacement=f"/skill:{skill.name}",
            start=0,
            end=token_end,
            kind=CompletionKind.SKILL,
            description=skill.description,
        )
        for skill in sorted(skills, key=lambda item: item.name)
        if skill.name.lower().startswith(prefix)
    ]
    return tuple(suggestions)


def _command_argument_completions(
    *,
    text: str,
    token_end: int,
    model_names: Sequence[str],
    provider_names: Sequence[str],
    thinking_levels: Sequence[str],
    theme_names: Sequence[str],
    session_ids: Sequence[str],
    session_options: Sequence[CompletionOption],
) -> tuple[CompletionItem, ...] | None:
    # [FUNCTION DOC]
    # HELPER: when text exists after the first token, return argument
    # completions for known commands: model/scoped-models -> sorted model
    # names; login/logout -> sorted provider names; resume -> session options
    # (or plain session ids) in given order; theme -> theme names in given order.
    # Returns None for other commands or when there is no argument text.
    # Arguments are assumed to start one character after the command token.
    if token_end >= len(text):
        return None

    command_name = text[:token_end].removeprefix("/").lower()
    if command_name in {"model", "scoped-models"}:
        return _value_completions(
            text=text,
            start=token_end + 1,
            options=_completion_options(model_names, description="Switch model"),
            sort=True,
        )
    if command_name in {"login", "logout"}:
        return _value_completions(
            text=text,
            start=token_end + 1,
            options=_completion_options(provider_names, description="Switch provider"),
            sort=True,
        )
    if command_name == "resume":
        return _value_completions(
            text=text,
            start=token_end + 1,
            options=(
                session_options
                if session_options
                else _completion_options(session_ids, description="Resume session")
            ),
            sort=False,
        )
    if command_name == "theme":
        return _value_completions(
            text=text,
            start=token_end + 1,
            options=_completion_options(theme_names, description="Set TUI theme"),
            sort=False,
        )
    return None


def _value_completions(
    *,
    text: str,
    start: int,
    options: Sequence[CompletionOption],
    sort: bool,
) -> tuple[CompletionItem, ...]:
    # [FUNCTION DOC]
    # HELPER: complete the argument word starting at the given index (to the next
    # space) against options whose value starts with it case-insensitively,
    # optionally sorting by value. Returns ARGUMENT items.
    end = _argument_token_end(text, start)
    prefix = text[start:end].lower()
    ordered_options = sorted(options, key=lambda item: item.value) if sort else options
    return tuple(
        CompletionItem(
            display=option.value,
            replacement=option.value,
            start=start,
            end=end,
            kind=CompletionKind.ARGUMENT,
            description=option.description,
        )
        for option in ordered_options
        if option.value.lower().startswith(prefix)
    )


def _completion_options(
    values: Sequence[str],
    *,
    description: str,
) -> tuple[CompletionOption, ...]:
    # [FUNCTION DOC]
    # HELPER: wrap plain string values as completion options sharing one
    # description.
    return tuple(CompletionOption(value=value, description=description) for value in values)


def _first_token_end(text: str) -> int:
    # [FUNCTION DOC]
    # HELPER: index of the first space in the text, or the text length.
    # Note: only spaces separate tokens (not tabs/newlines).
    separator = text.find(" ")
    return len(text) if separator == -1 else separator


def _argument_token_end(text: str, start: int) -> int:
    # [FUNCTION DOC]
    # HELPER: index of the first space at or after start, or the text length.
    separator = text.find(" ", start)
    return len(text) if separator == -1 else separator
