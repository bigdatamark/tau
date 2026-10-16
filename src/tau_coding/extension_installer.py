# [MODULE DOC]
# Implements "tau install": installs an extension into the user extensions directory
# (by default <Tau home>/extensions) from a local .py file, a local package directory,
# or a Git repository. Installs are staged in a hidden temporary directory inside the
# destination root, validated with the same discovery used at startup, then moved
# into place atomically (with backup/rollback when replacing). Subprocesses (git) go
# through an injectable command runner for testing.
"""Install Tau extensions from local paths or Git repositories."""

from __future__ import annotations

import contextlib
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp

from tau_coding.extensions import discover_extensions
from tau_coding.paths import TauPaths
from tau_coding.resources import TauResourcePaths

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class ExtensionInstallError(RuntimeError):
    # [CLASS DOC]
    # Error raised for any unsafe or failed installation (bad source, bad name, git
    # failure, invalid package, already installed, or wrapped filesystem errors).
    """Raised when an extension source cannot be installed safely."""


@dataclass(frozen=True, slots=True)
class GitExtensionSource:
    # [CLASS DOC]
    # Immutable value describing a parsed Git source: clone URL, optional ref to check
    # out (branch, tag or commit), and the derived install directory name.
    """A normalized Git repository source and optional checkout ref."""

    url: str
    ref: str | None
    name: str


def install_extension(
    source: str,
    *,
    force: bool = False,
    extensions_dir: Path | None = None,
    command_runner: CommandRunner = subprocess.run,
) -> Path:
    # [FUNCTION DOC]
    # Install one extension and return the final installed path.
    # Inputs: source string (local path, possibly with ~, or a Git source); keyword
    # force (replace an existing install); optional extensions directory override;
    # command runner used for git.
    # Behavior: create the destination root; if the source exists locally, resolve it
    # and derive the name from its basename (files must end in .py); otherwise parse it
    # as a Git source. Create a staging root inside the destination root, mirror the
    # "extensions/<name>" layout there, copy the file, copy the directory (skipping VCS,
    # virtualenv and cache folders) or clone the repo into it, validate via discovery,
    # then publish into destination root / name.
    # Raises ExtensionInstallError for all failures; OSErrors are wrapped. The staging
    # root is always removed afterwards (removal errors ignored).
    # Side effects: filesystem writes, git subprocesses, network access for clones.
    # KNOWN ISSUE: an existing local path that is neither a regular file nor a
    # directory (e.g. a socket) yields "unsupported local extension source".
    """Install one extension into Tau's user extension directory.

    Local Python files are copied. Local package directories and Git
    repositories must expose ``extension.py`` or ``[tool.tau].extensions`` so
    the normal user-directory discovery path can load them on the next run.
    """
    destination_root = extensions_dir or TauPaths().user_extensions_dir
    staging_root: Path | None = None
    try:
        destination_root.mkdir(parents=True, exist_ok=True)
        local_source = Path(source).expanduser()
        resolved_source: Path | None = None
        git_source: GitExtensionSource | None = None
        if local_source.exists():
            resolved_source = local_source.resolve()
            name = _local_install_name(resolved_source)
        else:
            git_source = parse_git_extension_source(source)
            name = git_source.name
        destination = destination_root / name

        # Reproduce the exact <Tau home>/extensions/<name> shape during validation.
        # This catches layouts that explicit -e discovery accepts but normal
        # user-directory discovery would skip, such as nested/extension.py.
        staging_root = Path(mkdtemp(prefix=".extension-install-", dir=destination_root))
        staging = staging_root / "extensions" / name
        staging.parent.mkdir()
        if resolved_source is not None and resolved_source.is_file():
            shutil.copy2(resolved_source, staging)
        elif resolved_source is not None and resolved_source.is_dir():
            shutil.copytree(resolved_source, staging, ignore=_local_copy_ignore)
        elif git_source is not None:
            _clone_git_source(git_source, staging, command_runner=command_runner)
        else:
            raise ExtensionInstallError(f"unsupported local extension source: {source}")

        _validate_staged_extension(staging_root)
        _publish_staged_extension(staging, destination, force=force)
        return destination
    except OSError as exc:
        raise ExtensionInstallError(f"extension installation failed: {exc}") from exc
    finally:
        if staging_root is not None:
            with contextlib.suppress(OSError):
                _remove_path(staging_root)


def parse_git_extension_source(source: str) -> GitExtensionSource:
    # [FUNCTION DOC]
    # Parse a Pi-style Git source into a GitExtensionSource.
    # Strips an optional "git:" prefix; a trailing "@ref" after the last "/" or ":" is
    # split off as the ref (empty ref is an error). "github.com/..." gets an https://
    # prefix; https, http, ssh, git, file URLs and "git@" scp-style sources are kept as
    # is; anything else is rejected. The name is the last path segment (after "/" or
    # ":") with a trailing ".git" removed, and must pass install-name validation.
    # Raises ExtensionInstallError for empty source/ref, unsupported scheme, bad name.
    # Pure function.
    """Normalize a Pi-style Git source accepted by ``tau install``."""
    raw = source[4:] if source.startswith("git:") else source
    if not raw:
        raise ExtensionInstallError("Git extension source is empty")

    ref: str | None = None
    last_at = raw.rfind("@")
    last_separator = max(raw.rfind("/"), raw.rfind(":"))
    if last_at > last_separator:
        raw, ref = raw[:last_at], raw[last_at + 1 :]
        if not ref:
            raise ExtensionInstallError("Git extension ref is empty")

    if raw.startswith("github.com/"):
        url = f"https://{raw}"
    elif raw.startswith(("https://", "http://", "ssh://", "git://", "file://", "git@")):
        url = raw
    else:
        raise ExtensionInstallError(
            "extension source does not exist locally and is not a supported Git source; "
            "use a local path, git:github.com/owner/repo, or a Git URL"
        )

    repository_path = raw.rstrip("/").rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    name = repository_path.removesuffix(".git")
    _validate_install_name(name)
    return GitExtensionSource(url=url, ref=ref, name=name)


def _clone_git_source(
    source: GitExtensionSource,
    destination: Path,
    *,
    command_runner: CommandRunner,
) -> None:
    # [FUNCTION DOC]
    # HELPER: Run "git clone -- <url> <destination>" via the command runner (captured
    # text output, no check); on non-zero exit raise ExtensionInstallError with stderr,
    # stdout, or a generic detail. If a ref is set, run a detached "git checkout" of it
    # in the clone and raise similarly on failure. Side effects: subprocess, network,
    # filesystem.
    result = command_runner(
        ["git", "clone", "--", source.url, str(destination)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "git clone failed").strip()
        raise ExtensionInstallError(f"could not clone {source.url}: {detail}")
    if source.ref is None:
        return
    result = command_runner(
        ["git", "-C", str(destination), "checkout", "--detach", source.ref],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "git checkout failed").strip()
        raise ExtensionInstallError(f"could not check out ref {source.ref!r}: {detail}")


def _local_install_name(source: Path) -> str:
    # [FUNCTION DOC]
    # HELPER: Return the install name for a local source (its basename) after checking
    # that a file source has a .py suffix and the name is valid. Raises
    # ExtensionInstallError otherwise.
    if source.is_file() and source.suffix != ".py":
        raise ExtensionInstallError("a local extension file must have a .py suffix")
    name = source.name
    _validate_install_name(name)
    return name


def _validate_install_name(name: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Ensure a name starts with a letter or digit and contains only letters,
    # digits, dot, underscore and dash (and is not "." or ".."). Raises
    # ExtensionInstallError otherwise; returns nothing.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or name in {".", ".."}:
        raise ExtensionInstallError(f"extension source has an unsupported install name: {name!r}")


def _validate_staged_extension(staging_root: Path) -> None:
    # [FUNCTION DOC]
    # HELPER: Run normal extension discovery with the staging root as the Tau resource
    # root (cwd is the process cwd). Raises ExtensionInstallError if any error-severity
    # diagnostics are produced (messages joined) or if nothing was discovered.
    # Side effects: reads the staged files (discovery may import nothing; only scans).
    paths = TauResourcePaths(root=staging_root, cwd=Path.cwd())
    discovered, diagnostics = discover_extensions(paths)
    errors = [diagnostic.message for diagnostic in diagnostics if diagnostic.severity == "error"]
    if errors:
        raise ExtensionInstallError("invalid extension package: " + "; ".join(errors))
    if not discovered:
        raise ExtensionInstallError(
            "an extension directory must contain extension.py or declare "
            "[tool.tau].extensions in pyproject.toml"
        )


def _publish_staged_extension(staging: Path, destination: Path, *, force: bool) -> None:
    # [FUNCTION DOC]
    # HELPER: Move the staged extension to its destination.
    # If the destination exists (or is a symlink): without force raise
    # ExtensionInstallError; with force, remove any stale ".<name>.backup" sibling,
    # move the destination to that backup, move staging into place (restoring the backup
    # and re-raising on any failure), then delete the backup. Otherwise just move
    # staging into place. Side effects: filesystem renames and deletions.
    if destination.exists() or destination.is_symlink():
        if not force:
            raise ExtensionInstallError(
                f"extension is already installed at {destination}; pass --force to replace it"
            )
        backup = destination.with_name(f".{destination.name}.backup")
        _remove_path(backup)
        destination.replace(backup)
        try:
            staging.replace(destination)
        except BaseException:
            backup.replace(destination)
            raise
        _remove_path(backup)
        return
    staging.replace(destination)


def _local_copy_ignore(_directory: str, names: Sequence[str]) -> set[str]:
    # [FUNCTION DOC]
    # HELPER: Ignore callback for directory copying; returns the subset of names that
    # are VCS directories, ".venv", or Python/pytest/mypy cache directories.
    ignored = {".git", ".hg", ".svn", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
    return ignored.intersection(names)


def _remove_path(path: Path) -> None:
    # [FUNCTION DOC]
    # HELPER: Delete a path: unlink symlinks and files (missing is fine), recursively
    # remove directories, do nothing if it does not exist. May raise OSError.
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)
