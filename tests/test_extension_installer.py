# [MODULE DOC]
# Tests for extension installation: copying single-file and package extensions into the extensions
# directory so discovery finds them, rejecting layouts discovery would not find, overwrite
# protection and forced replacement, git source parsing, and git install error handling and
# checkout using fake command runners (no real git invoked).
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tau_coding.extension_installer import (
    ExtensionInstallError,
    install_extension,
    parse_git_extension_source,
)
from tau_coding.extensions import discover_extensions
from tau_coding.resources import TauResourcePaths


def test_install_local_extension_file_is_discovered(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Install a single local Python file; verify it lands as the same file name in the extensions
    # directory and discovery finds exactly one extension named after it with no diagnostics.
    source = tmp_path / "hello.py"
    source.write_text("def setup(tau):\n    pass\n", encoding="utf-8")
    extensions_dir = tmp_path / "home" / "extensions"

    destination = install_extension(str(source), extensions_dir=extensions_dir)

    assert destination == extensions_dir / "hello.py"
    paths = TauResourcePaths(root=tmp_path / "home", cwd=tmp_path)
    discovered, diagnostics = discover_extensions(paths)
    assert [entry.name for entry in discovered] == ["hello"]
    assert diagnostics == ()


def test_install_local_package_copies_manifest_layout(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Install a local package directory whose pyproject declares its extension entry under the tau
    # tool table; verify the whole tree (including helper modules) is copied and discovery finds
    # the extension by its module directory name with no diagnostics.
    source = tmp_path / "my-extension"
    entry = source / "src" / "my_extension" / "extension.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("from . import helper\n\ndef setup(tau):\n    pass\n", encoding="utf-8")
    (entry.parent / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "pyproject.toml").write_text(
        '[tool.tau]\nextensions = ["src/my_extension/extension.py"]\n',
        encoding="utf-8",
    )
    extensions_dir = tmp_path / "home" / "extensions"

    destination = install_extension(str(source), extensions_dir=extensions_dir)

    assert (destination / "src" / "my_extension" / "helper.py").is_file()
    paths = TauResourcePaths(root=tmp_path / "home", cwd=tmp_path)
    discovered, diagnostics = discover_extensions(paths)
    assert [item.name for item in discovered] == ["my_extension"]
    assert diagnostics == ()


@pytest.mark.parametrize("nested_entry", ["one.py", "nested/extension.py"])
def test_install_rejects_directory_that_would_not_be_auto_discovered(
    tmp_path: Path, nested_entry: str
) -> None:
    # [FUNCTION DOC]
    # Parametrized over a top-level non-standard file and a nested extension file: verify
    # installing a directory without a top-level extension.py or manifest raises the install error
    # mentioning extension.py.
    source = tmp_path / "nested-only"
    entry = source / nested_entry
    entry.parent.mkdir(parents=True)
    entry.write_text("def setup(tau):\n    pass\n", encoding="utf-8")

    with pytest.raises(ExtensionInstallError, match="extension.py"):
        install_extension(str(source), extensions_dir=tmp_path / "extensions")


def test_install_preserves_existing_extension_without_force(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Verify installing over an existing extension without force raises "already installed" and
    # leaves the old file content intact.
    source = tmp_path / "hello.py"
    source.write_text("new\n", encoding="utf-8")
    extensions_dir = tmp_path / "extensions"
    extensions_dir.mkdir()
    installed = extensions_dir / "hello.py"
    installed.write_text("old\n", encoding="utf-8")

    with pytest.raises(ExtensionInstallError, match="already installed"):
        install_extension(str(source), extensions_dir=extensions_dir)

    assert installed.read_text(encoding="utf-8") == "old\n"


def test_install_force_replaces_existing_extension(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Verify forced install replaces the existing file content and leaves no backup file behind.
    source = tmp_path / "hello.py"
    source.write_text("new\n", encoding="utf-8")
    extensions_dir = tmp_path / "extensions"
    extensions_dir.mkdir()
    installed = extensions_dir / "hello.py"
    installed.write_text("old\n", encoding="utf-8")

    destination = install_extension(str(source), extensions_dir=extensions_dir, force=True)

    assert destination.read_text(encoding="utf-8") == "new\n"
    assert not (extensions_dir / ".hello.py.backup").exists()


@pytest.mark.parametrize(
    ("source", "url", "ref", "name"),
    [
        (
            "git:github.com/rian-dolphin/tau-subagents@v1.2.0",
            "https://github.com/rian-dolphin/tau-subagents",
            "v1.2.0",
            "tau-subagents",
        ),
        (
            "https://github.com/rian-dolphin/tau-subagents.git",
            "https://github.com/rian-dolphin/tau-subagents.git",
            None,
            "tau-subagents",
        ),
        (
            "git:git@github.com:rian-dolphin/tau-subagents.git@abc123",
            "git@github.com:rian-dolphin/tau-subagents.git",
            "abc123",
            "tau-subagents",
        ),
    ],
)
def test_parse_git_extension_source(source: str, url: str, ref: str | None, name: str) -> None:
    # [FUNCTION DOC]
    # Parametrized: verify git source strings ("git:" host paths with an optional "@ref", https
    # URLs ending in .git, and "git:" SSH URLs with a ref) parse into the expected clone URL, ref
    # (or None) and extension name (repo name without .git).
    parsed = parse_git_extension_source(source)

    assert (parsed.url, parsed.ref, parsed.name) == (url, ref, name)


def test_missing_git_is_reported_as_install_error(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Verify that when the command runner raises file-not-found (git missing), install raises the
    # install error containing that message.
    def missing_git(
        command: list[str], *, capture_output: bool, text: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        # [FUNCTION DOC]
        # HELPER: fake command runner that ignores its arguments and raises file-not-found for the
        # git executable.
        del command, capture_output, text, check
        raise FileNotFoundError("git executable not found")

    with pytest.raises(ExtensionInstallError, match="git executable not found"):
        install_extension(
            "git:github.com/example/tau-demo",
            extensions_dir=tmp_path / "extensions",
            command_runner=missing_git,
        )


def test_failed_git_install_removes_partial_clone(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Verify that a failed clone (non-zero exit with stderr "network failed") raises the install
    # error with that message and leaves the extensions directory empty, i.e. the partial clone
    # directory is removed.
    extensions_dir = tmp_path / "extensions"

    def fake_run(
        command: list[str], *, capture_output: bool, text: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        # [FUNCTION DOC]
        # HELPER: fake command runner that creates the clone destination directory (last argument)
        # and returns a completed process with exit code 1 and stderr "network failed".
        del capture_output, text, check
        Path(command[-1]).mkdir()
        return subprocess.CompletedProcess(command, 1, "", "network failed")

    with pytest.raises(ExtensionInstallError, match="network failed"):
        install_extension(
            "git:github.com/example/tau-demo",
            extensions_dir=extensions_dir,
            command_runner=fake_run,
        )

    assert list(extensions_dir.iterdir()) == []


def test_install_git_source_clones_and_checks_out_ref(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Verify installing a git source with a ref clones the https URL (with a "--" separator) into
    # the extensions directory under the repo name, then runs a detached checkout of the ref
    # inside that directory, and returns that destination.
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], *, capture_output: bool, text: bool, check: bool
    ) -> subprocess.CompletedProcess[str]:
        # [FUNCTION DOC]
        # HELPER: fake command runner that records each command; for clone it creates the
        # destination with a minimal extension.py. Always returns success.
        del capture_output, text, check
        calls.append(command)
        if command[1] == "clone":
            destination = Path(command[-1])
            destination.mkdir()
            (destination / "extension.py").write_text(
                "def setup(tau):\n    pass\n", encoding="utf-8"
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    destination = install_extension(
        "git:github.com/example/tau-demo@v1",
        extensions_dir=tmp_path / "extensions",
        command_runner=fake_run,
    )

    assert destination == tmp_path / "extensions" / "tau-demo"
    assert calls[0][:4] == ["git", "clone", "--", "https://github.com/example/tau-demo"]
    assert calls[1][0:2] == ["git", "-C"]
    assert Path(calls[1][2]).name == destination.name
    assert calls[1][3:] == ["checkout", "--detach", "v1"]
