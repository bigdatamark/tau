# [MODULE DOC]
# Markdown resource location and parsing helpers for the coding app: where Tau
# looks for skills, prompt templates, themes and SYSTEM.md / APPEND_SYSTEM.md
# system-prompt files (user Tau home, user .agents dir, and project-local .tau /
# .agents dirs when the project is trusted), discovery of system-prompt files
# with precedence diagnostics, and a minimal dependency-free frontmatter parser.
"""Markdown resource path and frontmatter helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tau_agent.types import JSONValue
from tau_coding.paths import TauPaths


class ResourceError(ValueError):
    # [CLASS DOC]
    # Error (a ValueError) raised when resources cannot be inspected/read or are
    # invalid.
    """Raised when Tau resources are invalid or cannot be expanded."""


@dataclass(frozen=True, slots=True)
class SystemPromptResources:
    # [CLASS DOC]
    # Immutable result of system-prompt discovery: optional replacement prompt text
    # and its path; the joined append prompt (pieces separated by blank lines) plus
    # the individual append texts and their paths; and diagnostics describing which
    # files were selected, shadowed or ignored. All empty by default.
    """Discovered Tau-native system-prompt file contents and sources."""

    custom_prompt: str | None = None
    custom_prompt_path: Path | None = None
    append_prompt: str | None = None
    append_prompts: tuple[str, ...] = ()
    append_prompt_paths: tuple[Path, ...] = ()
    diagnostics: tuple[ResourceDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class ResourceDiagnostic:
    # [CLASS DOC]
    # Immutable non-fatal discovery note: kind, message, optional path, optional
    # resource name and severity (default "warning"; "info" for selection notes).
    """A non-fatal resource discovery problem or precedence note."""

    kind: str
    message: str
    path: Path | None = None
    name: str | None = None
    severity: str = "warning"

    def format(self) -> str:
        # [FUNCTION DOC]
        # Return a one-line string "<severity> <kind>[ <name>]: <message>" with
        # " (<path>)" appended when a path is present. No side effects.
        """Return a concise human-readable diagnostic line."""
        parts = [self.severity, self.kind]
        if self.name is not None:
            parts.append(self.name)
        label = " ".join(parts)
        if self.path is None:
            return f"{label}: {self.message}"
        return f"{label}: {self.message} ({self.path})"


@dataclass(frozen=True, slots=True)
class TauResourcePaths:
    # [CLASS DOC]
    # Immutable plan of resource roots: root (Tau home, default from the default
    # path layout), optional cwd for project resources, optional .agents root
    # (default ~/.agents; None disables it), optional explicit path layout, and a
    # flag enabling project-local resources. Properties compute directory lists in
    # increasing precedence order (later entries override earlier ones).
    """Filesystem locations for Tau markdown resources.

    By default Tau loads Tau-native resources from the configured Tau home and
    `.agents` resources from the user home directory. When a cwd is provided,
    project-local `.tau` and `.agents` resources are loaded automatically as
    well.
    """

    root: Path = field(default_factory=lambda: TauPaths().home)
    cwd: Path | None = None
    agents_root: Path | None = field(default_factory=lambda: Path.home() / ".agents")
    paths: TauPaths | None = None
    project_resources_enabled: bool = True

    @property
    def skills_dir(self) -> Path:
        # [FUNCTION DOC]
        # Return the Tau home "skills" directory. No I/O.
        """Return the primary Tau skills directory."""
        return self.root / "skills"

    @property
    def prompts_dir(self) -> Path:
        # [FUNCTION DOC]
        # Return the Tau home "prompts" directory. No I/O.
        """Return the primary Tau prompt templates directory."""
        return self.root / "prompts"

    @property
    def system_prompt_path(self) -> Path:
        # [FUNCTION DOC]
        # Return the user-level SYSTEM.md (replacement system prompt) path.
        """Return the user-level replacement system-prompt file."""
        return self.root / "SYSTEM.md"

    @property
    def append_system_prompt_path(self) -> Path:
        # [FUNCTION DOC]
        # Return the user-level APPEND_SYSTEM.md path.
        """Return the user-level appended system-prompt file."""
        return self.root / "APPEND_SYSTEM.md"

    @property
    def skills_dirs(self) -> tuple[Path, ...]:
        # [FUNCTION DOC]
        # Return skill directories in increasing precedence: Tau home skills, .agents
        # skills (if an agents root is set), then (if cwd is set and project resources
        # are enabled) the project .tau skills and project .agents skills dirs from the
        # path layout. Paths are home-expanded and deduplicated keeping first occurrence.
        # No filesystem access.
        """Return skill directories in increasing precedence order.

        Only the ``skills`` subdirectory of an ``.agents`` root is scanned,
        never the root ``.agents`` directory itself (which may contain
        ``README.md``, ``AGENTS.md``, etc.).
        """
        paths = self._paths()
        dirs = [self.skills_dir]
        if self.agents_root is not None:
            dirs.append(self.agents_root / "skills")
        if self.cwd is not None and self.project_resources_enabled:
            dirs.extend(
                [
                    paths.project_skills_dir(self.cwd),
                    paths.project_agents_skills_dir(self.cwd),
                ]
            )
        return tuple(_dedupe_paths(dirs))

    @property
    def themes_dirs(self) -> tuple[Path, ...]:
        # [FUNCTION DOC]
        # Return theme directories in increasing precedence: Tau home "themes", then the
        # project themes dir when cwd is set and project resources are enabled.
        # Expanded and deduplicated; .agents dirs are never included.
        """Return TUI theme directories in increasing precedence order.

        Themes are Tau-specific, so unlike skills and prompts no ``.agents``
        directories are scanned.
        """
        paths = self._paths()
        dirs = [self.root / "themes"]
        if self.cwd is not None and self.project_resources_enabled:
            dirs.append(paths.project_themes_dir(self.cwd))
        return tuple(_dedupe_paths(dirs))

    @property
    def prompts_dirs(self) -> tuple[Path, ...]:
        # [FUNCTION DOC]
        # Return prompt template directories in increasing precedence: Tau home prompts,
        # .agents prompts (if set), then project .tau and .agents prompt dirs when
        # enabled. Expanded and deduplicated.
        """Return prompt template directories in increasing precedence order."""
        paths = self._paths()
        dirs = [self.prompts_dir]
        if self.agents_root is not None:
            dirs.append(self.agents_root / "prompts")
        if self.cwd is not None and self.project_resources_enabled:
            dirs.extend(
                [
                    paths.project_prompts_dir(self.cwd),
                    paths.project_agents_prompts_dir(self.cwd),
                ]
            )
        return tuple(_dedupe_paths(dirs))

    def _paths(self) -> TauPaths:
        # [FUNCTION DOC]
        # HELPER: Return the explicit path layout if given, otherwise build one with
        # home set to root and agents home set to the agents root (or ~/.agents).
        agents_home = self.agents_root or Path.home() / ".agents"
        return self.paths or TauPaths(home=self.root, agents_home=agents_home)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    # [FUNCTION DOC]
    # HELPER: Expand a leading "~" in each path and return them in order with
    # duplicates (after expansion, without resolving symlinks) removed.
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def discover_system_prompt_resources(
    paths: TauResourcePaths,
    *,
    custom_prompt_explicit: bool = False,
    enabled: bool = True,
) -> SystemPromptResources:
    # [FUNCTION DOC]
    # Discover system prompt files. Inputs: resource paths; keyword flags for whether
    # a replacement prompt was given explicitly at startup and whether discovery is
    # enabled at all.
    # Behavior: when disabled returns empty resources. Otherwise selects one
    # replacement SYSTEM.md (project overrides user; ignored entirely when an
    # explicit prompt was supplied) and collects all APPEND_SYSTEM.md files (user
    # then project), recording diagnostics.
    # Outputs: system prompt resources. Raises ResourceError on inspect/read
    # failures. Side effects: filesystem reads.
    """Discover a precedence-selected base and cumulative append prompt files."""
    if not enabled:
        return SystemPromptResources()

    diagnostics: list[ResourceDiagnostic] = []
    custom_prompt, custom_path = _discover_system_prompt_file(
        paths,
        filename="SYSTEM.md",
        label="replacement",
        explicit=custom_prompt_explicit,
        diagnostics=diagnostics,
    )
    append_prompts, append_paths = _discover_append_system_prompt_files(
        paths,
        diagnostics=diagnostics,
    )
    return SystemPromptResources(
        custom_prompt=custom_prompt,
        custom_prompt_path=custom_path,
        append_prompt="\n\n".join(append_prompts) if append_prompts else None,
        append_prompts=append_prompts,
        append_prompt_paths=append_paths,
        diagnostics=tuple(diagnostics),
    )


def _discover_system_prompt_file(
    paths: TauResourcePaths,
    *,
    filename: str,
    label: str,
    explicit: bool,
    diagnostics: list[ResourceDiagnostic],
) -> tuple[str | None, Path | None]:
    # [FUNCTION DOC]
    # HELPER: Choose a single system prompt file by precedence. Candidates are the
    # project .tau/<filename> (when cwd set and project resources enabled) then
    # <root>/<filename>. Records which exist (ResourceError on OSError while
    # checking). If explicit: appends an info diagnostic per existing file saying it
    # was ignored and returns (None, None). If none exist returns (None, None).
    # Otherwise reads the first existing file as UTF-8 (ResourceError on failure),
    # appends an info "selected <scope>" diagnostic and a warning "shadowed"
    # diagnostic for each lower-precedence file, and returns (content, path).
    # Side effects: mutates the diagnostics list; reads files.
    candidates: list[tuple[str, Path]] = []
    if paths.cwd is not None and paths.project_resources_enabled:
        candidates.append(("project", paths.cwd / ".tau" / filename))
    candidates.append(("user", paths.root / filename))

    existing: list[tuple[str, Path]] = []
    for scope, path in candidates:
        try:
            if path.exists():
                existing.append((scope, path))
        except OSError as exc:
            raise ResourceError(
                f"Could not inspect {label} system prompt file {path}: {exc}"
            ) from exc

    if explicit:
        for _scope, path in existing:
            diagnostics.append(
                ResourceDiagnostic(
                    kind="system-prompt",
                    name=label,
                    path=path,
                    severity="info",
                    message="ignored because an explicit startup value takes precedence",
                )
            )
        return None, None
    if not existing:
        return None, None

    selected_scope, selected_path = existing[0]
    try:
        content = selected_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ResourceError(
            f"Could not read {label} system prompt file {selected_path} as UTF-8: {exc}"
        ) from exc

    diagnostics.append(
        ResourceDiagnostic(
            kind="system-prompt",
            name=label,
            path=selected_path,
            severity="info",
            message=f"selected {selected_scope} system prompt file",
        )
    )
    for _scope, path in existing[1:]:
        diagnostics.append(
            ResourceDiagnostic(
                kind="system-prompt",
                name=label,
                path=path,
                message=f"shadowed by higher-precedence file {selected_path}",
            )
        )
    return content, selected_path


def _discover_append_system_prompt_files(
    paths: TauResourcePaths,
    *,
    diagnostics: list[ResourceDiagnostic],
) -> tuple[tuple[str, ...], tuple[Path, ...]]:
    # [FUNCTION DOC]
    # HELPER: Collect append prompt files from user root then project .tau (when
    # enabled), skipping candidates that resolve to an already seen path (so the same
    # file is not appended twice when cwd is the home). For each existing file reads
    # it as UTF-8 and records an info "selected" diagnostic. Returns a pair of tuples
    # (contents, paths) in that order. Raises ResourceError on inspect/read errors.
    # Side effects: mutates diagnostics; reads files.
    """Read every append file in broad-to-specific order."""
    candidates: list[tuple[str, Path]] = [("user", paths.root / "APPEND_SYSTEM.md")]
    if paths.cwd is not None and paths.project_resources_enabled:
        candidates.append(("project", paths.cwd / ".tau" / "APPEND_SYSTEM.md"))

    contents: list[str] = []
    selected_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for scope, path in candidates:
        try:
            resolved_path = path.expanduser().resolve()
            if resolved_path in seen_paths:
                continue
            seen_paths.add(resolved_path)
            exists = path.exists()
        except OSError as exc:
            raise ResourceError(
                f"Could not inspect append system prompt file {path}: {exc}"
            ) from exc
        if not exists:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ResourceError(
                f"Could not read append system prompt file {path} as UTF-8: {exc}"
            ) from exc
        contents.append(content)
        selected_paths.append(path)
        diagnostics.append(
            ResourceDiagnostic(
                kind="system-prompt",
                name="append",
                path=path,
                severity="info",
                message=f"selected {scope} system prompt file",
            )
        )

    return tuple(contents), tuple(selected_paths)


def resource_paths_with_cwd(
    paths: TauResourcePaths | None,
    cwd: Path,
) -> TauResourcePaths:
    # [FUNCTION DOC]
    # Return resource paths bound to a cwd. With no existing plan, a default plan for
    # that cwd. Otherwise a copy that keeps root, agents root, path layout and the
    # project-resources flag but replaces cwd (plans are destination-bound when
    # resuming/replacing sessions). No I/O.
    """Return resource paths with a cwd available for project-local discovery."""
    if paths is None:
        return TauResourcePaths(cwd=cwd)
    # A resource plan is destination-bound. Replacement/resume callers may
    # supply a plan created for the source session, but only its user/global
    # roots and feature flags are reusable.
    return TauResourcePaths(
        root=paths.root,
        cwd=cwd,
        agents_root=paths.agents_root,
        paths=paths.paths,
        project_resources_enabled=paths.project_resources_enabled,
    )


def resource_paths_with_project_trust(
    paths: TauResourcePaths,
    *,
    trusted: bool,
) -> TauResourcePaths:
    # [FUNCTION DOC]
    # Return a copy of the plan with project resources enabled exactly when trusted
    # is True (global-only vs global-plus-project). No I/O.
    """Return a coherent global-only or global-plus-project resource plan."""
    return TauResourcePaths(
        root=paths.root,
        cwd=paths.cwd,
        agents_root=paths.agents_root,
        paths=paths.paths,
        project_resources_enabled=trusted,
    )


def parse_markdown_resource(text: str) -> tuple[dict[str, str], str]:
    # [FUNCTION DOC]
    # Parse simple frontmatter. Input: markdown text. Line endings are normalized to
    # LF first. If the text starts with a "---" line and a later "\n---" marker is
    # found, the lines between are parsed as "key: value" pairs (blank lines,
    # lines starting with "#" and lines without a colon are skipped; keys and values
    # are trimmed and surrounding quote characters stripped from values; later keys
    # win). The body is everything after the closing marker minus one leading
    # newline. Without valid frontmatter returns an empty mapping and the normalized
    # text. Output: (metadata mapping, body). Raises nothing.
    # KNOWN ISSUE: the closing marker match is a plain substring search, so a line
    # such as "----" or "---foo" inside the frontmatter terminates it early and any
    # remaining characters on the marker line leak into the body.
    """Parse minimal YAML-like frontmatter from a markdown resource.

    Only simple `key: value` pairs are supported. This keeps resource parsing
    dependency-free and avoids evaluating arbitrary code.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---\n"):
        return {}, normalized

    end = normalized.find("\n---", 4)
    if end == -1:
        return {}, normalized

    raw_frontmatter = normalized[4:end]
    body = normalized[end + len("\n---") :]
    if body.startswith("\n"):
        body = body[1:]

    metadata: dict[str, str] = {}
    for line in raw_frontmatter.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition(":")
        if not separator:
            continue
        metadata[key.strip()] = value.strip().strip("\"'")
    return metadata, body


def derive_description(content: str) -> str | None:
    # [FUNCTION DOC]
    # Return a short description from markdown: the first non-blank line, with
    # leading "#" characters and whitespace removed if it is a heading (None if the
    # heading is empty); None if the content has no non-blank lines.
    """Derive a short description from markdown content."""
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
        return stripped
    return None


def metadata_to_json(metadata: dict[str, str]) -> dict[str, JSONValue]:
    # [FUNCTION DOC]
    # Return a shallow copy of a string-to-string metadata mapping typed as
    # JSON-compatible values. No conversion is performed.
    """Convert string metadata into JSON-like values."""
    return dict(metadata)
