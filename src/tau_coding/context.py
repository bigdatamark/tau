# [MODULE DOC]
# Discovery of project instruction files (AGENTS.md) that are injected as
# context into the coding agent's system prompt. Candidate files come from the
# Tau home root, an optional agents root, every directory from the detected
# project root down to the working directory, and the project's .tau and agents
# directories. Missing files are skipped; unreadable files produce diagnostics.
# PROJECT_MARKERS lists filenames whose presence marks a project root.
"""Project instruction discovery for Tau coding sessions."""

from __future__ import annotations

from pathlib import Path

from tau_coding.resources import ResourceDiagnostic, TauResourcePaths
from tau_coding.system_prompt import ProjectContextFile

PROJECT_MARKERS = (".git", "pyproject.toml", "uv.lock", "setup.py", "package.json")


def discover_project_context(
    paths: TauResourcePaths | None = None,
) -> tuple[ProjectContextFile, ...]:
    # [FUNCTION DOC]
    # Convenience wrapper: runs discovery with diagnostics and returns only the
    # tuple of context files (path string plus contents), discarding diagnostics.
    # Input: optional resource paths (defaults are constructed when omitted).
    # Side effects: filesystem reads.
    """Discover project instruction files for system prompt context."""
    context_files, _diagnostics = discover_project_context_with_diagnostics(paths)
    return context_files


def discover_project_context_with_diagnostics(
    paths: TauResourcePaths | None = None,
) -> tuple[tuple[ProjectContextFile, ...], tuple[ResourceDiagnostic, ...]]:
    # [FUNCTION DOC]
    # Reads each existing candidate AGENTS.md file in order as UTF-8 and returns a
    # pair: the tuple of context files (resolved path string and text) and a tuple of
    # resource diagnostics of kind "context" for files that raised OSError on read.
    # Input: optional resource paths, defaulting to a fresh resource-paths object.
    # Never raises for read errors. Side effects: filesystem reads.
    """Discover project instruction files and return non-fatal diagnostics."""
    resource_paths = paths or TauResourcePaths()
    context_files: list[ProjectContextFile] = []
    diagnostics: list[ResourceDiagnostic] = []
    for path in _context_file_candidates(resource_paths):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            diagnostics.append(
                ResourceDiagnostic(
                    kind="context",
                    path=path,
                    message=f"could not read context file: {exc}",
                )
            )
            continue
        context_files.append(ProjectContextFile(path=str(path), content=content))
    return tuple(context_files), tuple(diagnostics)


def _context_file_candidates(paths: TauResourcePaths) -> tuple[Path, ...]:
    # [FUNCTION DOC]
    # HELPER: build the ordered candidate list: Tau root AGENTS.md, then agents-root
    # AGENTS.md if configured; when a cwd is set and project resources are enabled,
    # the ancestor chain from project root to cwd, then the project .tau dir and the
    # project agents dir AGENTS.md. Keeps only existing regular files, then resolves
    # and de-duplicates them preserving first occurrence. Returns a tuple of paths.
    # Note: uses a private accessor on the resource paths object to obtain TauPaths.
    candidates: list[Path] = [paths.root / "AGENTS.md"]
    if paths.agents_root is not None:
        candidates.append(paths.agents_root / "AGENTS.md")

    if paths.cwd is not None and paths.project_resources_enabled:
        cwd = paths.cwd.expanduser().resolve()
        project_root = _find_project_root(cwd)
        candidates.extend(_ancestor_agents_files(project_root, cwd))
        tau_paths = paths._paths()
        candidates.extend(
            [
                tau_paths.project_tau_dir(cwd) / "AGENTS.md",
                tau_paths.project_agents_dir(cwd) / "AGENTS.md",
            ]
        )

    existing = [path for path in candidates if path.is_file()]
    return tuple(_dedupe_resolved_paths(existing))


def _find_project_root(cwd: Path) -> Path:
    # [FUNCTION DOC]
    # HELPER: walk from cwd up through its parents and return the first directory
    # containing any project marker; if none is found, return cwd itself.
    for path in (cwd, *cwd.parents):
        if any((path / marker).exists() for marker in PROJECT_MARKERS):
            return path
    return cwd


def _ancestor_agents_files(project_root: Path, cwd: Path) -> list[Path]:
    # [FUNCTION DOC]
    # HELPER: return AGENTS.md paths for the project root and each directory between
    # it and cwd (inclusive), ordered outermost first. If cwd is not inside the
    # project root, return just the cwd AGENTS.md path. No filesystem access.
    try:
        relative = cwd.relative_to(project_root)
    except ValueError:
        return [cwd / "AGENTS.md"]

    paths = [project_root / "AGENTS.md"]
    current = project_root
    for part in relative.parts:
        current = current / part
        paths.append(current / "AGENTS.md")
    return paths


def _dedupe_resolved_paths(paths: list[Path]) -> list[Path]:
    # [FUNCTION DOC]
    # HELPER: expand user and resolve each path, dropping later duplicates while
    # keeping order. Returns the resolved paths (not the originals).
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped
