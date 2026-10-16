# [MODULE DOC]
# Canonical filesystem layout for Tau. Resolves the Tau home directory (TAU_HOME
# or ~/.tau) and the ~/.agents home, and derives every user-level and
# project-level resource location (sessions, logs, caches, skills, prompts,
# themes, extensions, extension state) plus per-project session directories
# named by a readable path slug and a short hash of the resolved cwd.
"""Canonical filesystem paths for Tau user and project data."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from hashlib import sha256
from os import environ
from pathlib import Path


def _default_tau_home() -> Path:
    # [FUNCTION DOC]
    # HELPER: return TAU_HOME (user-expanded) when set and non-empty, else ~/.tau.
    # Raises ValueError if "~" cannot be expanded or the result is not absolute.
    # Reads the environment only.
    """Return the configured Tau home, falling back to ``~/.tau``."""
    value = environ.get("TAU_HOME")
    if value is None or value == "":
        return Path.home() / ".tau"

    try:
        path = Path(value).expanduser()
    except RuntimeError as exc:
        raise ValueError(
            "TAU_HOME could not expand '~'; use an absolute path or an existing user home"
        ) from exc
    if not path.is_absolute():
        raise ValueError("TAU_HOME must be an absolute path after '~' expansion")
    return path


@dataclass(frozen=True, slots=True)
class TauPaths:
    # [CLASS DOC]
    # Frozen dataclass holding the Tau home and agents home (both defaulting from
    # the environment/user home at construction). All other locations are derived
    # properties or cwd-taking methods; only default_session_path touches disk.
    """Resolved Tau filesystem locations.

    Tau keeps durable application data under ``TAU_HOME`` (``~/.tau`` by
    default) while also loading project-local resources from the active working
    directory.
    """

    home: Path = field(default_factory=_default_tau_home)
    agents_home: Path = field(default_factory=lambda: Path.home() / ".agents")

    @property
    def sessions_dir(self) -> Path:
        # [FUNCTION DOC]
        # Property: home/sessions, root of per-project session directories.
        """Return the user-level session directory."""
        return self.home / "sessions"

    @property
    def logs_dir(self) -> Path:
        # [FUNCTION DOC]
        # Property: home/logs, diagnostic log directory.
        """Return Tau's user-level diagnostic log directory."""
        return self.home / "logs"

    @property
    def agent_calls_log_path(self) -> Path:
        # [FUNCTION DOC]
        # Property: logs/agent-calls.jsonl, JSONL log of agent call failures.
        """Return the JSONL diagnostic log for agent-call failures."""
        return self.logs_dir / "agent-calls.jsonl"

    @property
    def models_store_path(self) -> Path:
        # [FUNCTION DOC]
        # Property: home/models-store.json, remote model catalog cache.
        """Return the persisted remote model-catalog cache path."""
        return self.home / "models-store.json"

    @property
    def codex_version_store_path(self) -> Path:
        # [FUNCTION DOC]
        # Property: home/codex-version-store.json, cached latest Codex client version.
        """Return the latest released Codex-version cache path."""
        return self.home / "codex-version-store.json"

    @property
    def codex_models_store_path(self) -> Path:
        # [FUNCTION DOC]
        # Property: home/codex-models-store.json, account-scoped Codex model cache.
        """Return the account-scoped Codex model-catalog cache path."""
        return self.home / "codex-models-store.json"

    @property
    def extension_state_dir(self) -> Path:
        # [FUNCTION DOC]
        # Property: home/state/extensions, state owned by built-in extensions.
        """Return the user-level state directory owned by built-in extensions."""
        return self.home / "state" / "extensions"

    @property
    def llama_cpp_state_path(self) -> Path:
        # [FUNCTION DOC]
        # Property: extension state dir/llama.cpp.json for the llama.cpp integration.
        """Return the safe built-in llama.cpp integration state path."""
        return self.extension_state_dir / "llama.cpp.json"

    @property
    def user_skills_dir(self) -> Path:
        # [FUNCTION DOC]
        # Property: home/skills.
        """Return Tau's user-level skills directory."""
        return self.home / "skills"

    @property
    def user_prompts_dir(self) -> Path:
        # [FUNCTION DOC]
        # Property: home/prompts (prompt templates).
        """Return Tau's user-level prompt templates directory."""
        return self.home / "prompts"

    @property
    def user_themes_dir(self) -> Path:
        # [FUNCTION DOC]
        # Property: home/themes (TUI themes).
        """Return Tau's user-level TUI themes directory."""
        return self.home / "themes"

    @property
    def user_extensions_dir(self) -> Path:
        # [FUNCTION DOC]
        # Property: home/extensions.
        """Return Tau's user-level extension directory."""
        return self.home / "extensions"

    @property
    def user_agents_skills_dir(self) -> Path:
        # [FUNCTION DOC]
        # Property: agents home/skills.
        """Return the user-level `.agents/skills` directory."""
        return self.agents_home / "skills"

    @property
    def user_agents_prompts_dir(self) -> Path:
        # [FUNCTION DOC]
        # Property: agents home/prompts.
        """Return the user-level `.agents/prompts` directory."""
        return self.agents_home / "prompts"

    def project_tau_dir(self, cwd: Path) -> Path:
        # [FUNCTION DOC]
        # Return cwd/.tau, the project-local Tau resource directory. Pure.
        """Return the project-local Tau resource directory."""
        return cwd / ".tau"

    def project_agents_dir(self, cwd: Path) -> Path:
        # [FUNCTION DOC]
        # Return cwd/.agents. Pure.
        """Return the project-local `.agents` resource directory."""
        return cwd / ".agents"

    def project_skills_dir(self, cwd: Path) -> Path:
        # [FUNCTION DOC]
        # Return cwd/.tau/skills. Pure.
        """Return the project-local Tau skills directory."""
        return self.project_tau_dir(cwd) / "skills"

    def project_prompts_dir(self, cwd: Path) -> Path:
        # [FUNCTION DOC]
        # Return cwd/.tau/prompts. Pure.
        """Return the project-local Tau prompt templates directory."""
        return self.project_tau_dir(cwd) / "prompts"

    def project_themes_dir(self, cwd: Path) -> Path:
        # [FUNCTION DOC]
        # Return cwd/.tau/themes. Pure.
        """Return the project-local Tau TUI themes directory."""
        return self.project_tau_dir(cwd) / "themes"

    def project_agents_skills_dir(self, cwd: Path) -> Path:
        # [FUNCTION DOC]
        # Return cwd/.agents/skills. Pure.
        """Return the project-local `.agents/skills` directory."""
        return self.project_agents_dir(cwd) / "skills"

    def project_agents_prompts_dir(self, cwd: Path) -> Path:
        # [FUNCTION DOC]
        # Return cwd/.agents/prompts. Pure.
        """Return the project-local `.agents/prompts` directory."""
        return self.project_agents_dir(cwd) / "prompts"

    def project_session_dir(self, cwd: Path) -> Path:
        # [FUNCTION DOC]
        # Resolve cwd, compute the first 6 hex chars of the SHA-256 of its string form
        # and a slug of the path; return sessions_dir/"<slug or 'project'>-<hash>".
        # Resolving may touch the filesystem (symlinks); no writes.
        """Return the user-home session directory for a project cwd."""
        resolved = cwd.resolve()
        digest = sha256(str(resolved).encode("utf-8")).hexdigest()[:6]
        slug = _slugify_path(resolved)
        return self.sessions_dir / f"{slug or 'project'}-{digest}"

    def default_session_path(self, cwd: Path) -> Path:
        # [FUNCTION DOC]
        # Return project session dir/default.jsonl, creating the parent directory
        # (side effect: mkdir with parents). May raise OSError on mkdir failure.
        """Return the default JSONL session path for a project cwd."""
        path = self.project_session_dir(cwd) / "default.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def _slugify_path(path: Path, *, max_length: int = 72) -> str:
    # [FUNCTION DOC]
    # HELPER: turn a path into a lowercase slug. Uses path parts without the anchor;
    # paths under the user home are rewritten to start with "home". Each part has
    # runs of characters outside letters, digits, dot, underscore and hyphen replaced
    # by "-", is stripped of leading/trailing "." "-" "_", and dropped if empty;
    # parts are joined with "-". If longer than max_length (default 72), keeps as
    # many trailing whole parts as fit, falling back to the last max_length chars.
    parts = [part for part in path.parts if part not in (path.anchor, "")]
    try:
        relative_to_home = path.relative_to(Path.home())
    except ValueError:
        pass
    else:
        parts = ["home", *relative_to_home.parts]

    slug_parts = [
        normalized
        for part in parts
        if (normalized := re.sub(r"[^a-zA-Z0-9._-]+", "-", part).strip(".-_").lower())
    ]
    slug = "-".join(slug_parts)
    if len(slug) <= max_length:
        return slug

    suffix_parts: list[str] = []
    suffix_length = 0
    for part in reversed(slug_parts):
        next_length = suffix_length + len(part) + (1 if suffix_parts else 0)
        if next_length > max_length:
            break
        suffix_parts.append(part)
        suffix_length = next_length
    return "-".join(reversed(suffix_parts)) or slug[-max_length:].strip("-")
