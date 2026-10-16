# [MODULE DOC]
# Packaging metadata tests. Checks the Python version floor is consistent across
# pyproject, ruff, mypy and .python-version; that the current project version has
# an entry in the bundled release notes JSON; and that a built wheel ships the
# release notes and built-in docs/example extension resources but no bundled
# skills. Module constants hold the repo root, the release notes source and wheel
# paths, and the set of required built-in resource paths inside the wheel.
import json
import subprocess
import tomllib
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
RELEASE_NOTES_SOURCE_PATH = ROOT / "src" / "tau_coding" / "data" / "release-notes" / "releases.json"
RELEASE_NOTES_WHEEL_PATH = "tau_coding/data/release-notes/releases.json"
BUILTIN_RESOURCE_WHEEL_PATHS = {
    "tau_coding/data/docs/README.md",
    "tau_coding/data/docs/extensions.md",
    "tau_coding/data/examples/extensions/ask_user_question.py",
    "tau_coding/data/examples/extensions/hello_tool.py",
    "tau_coding/data/examples/extensions/prompt_section.py",
    "tau_coding/data/examples/extensions/sidebar_status.py",
}


def test_python_version_floor_matches_package_metadata() -> None:
    # [FUNCTION DOC]
    # Reads pyproject.toml and .python-version and asserts requires-python is
    # ">=3.12", ruff target is py312, mypy python_version is 3.12 and the
    # .python-version file contains 3.12. Side effects: reads files only.
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["requires-python"] == ">=3.12"
    assert pyproject["tool"]["ruff"]["target-version"] == "py312"
    assert pyproject["tool"]["mypy"]["python_version"] == "3.12"
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"


def test_current_version_has_release_notes() -> None:
    # [FUNCTION DOC]
    # Asserts the release notes JSON file exists and that some entry's "version"
    # equals the project version in pyproject.toml. Reads files only.
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert RELEASE_NOTES_SOURCE_PATH.is_file(), (
        f"release notes not found at {RELEASE_NOTES_SOURCE_PATH}"
    )
    release_notes = json.loads(RELEASE_NOTES_SOURCE_PATH.read_text(encoding="utf-8"))

    assert any(entry["version"] == pyproject["project"]["version"] for entry in release_notes)


def test_wheel_includes_release_notes_package_data(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Regression test: runs "uv build --wheel" into a temporary directory (a real
    # subprocess, slow, needs uv), asserts success and exactly one wheel, then
    # inspects the zip listing to require the release notes JSON and all built-in
    # resource paths, and forbids any file under tau_coding/data/skills/.
    # Input: pytest tmp_path. Side effects: subprocess build, writes to tmp dir.
    """Regression: releases.json must be included in installed wheels."""
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    wheels = sorted(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1, result.stdout + result.stderr
    with ZipFile(wheels[0]) as wheel:
        wheel_files = set(wheel.namelist())

    assert RELEASE_NOTES_WHEEL_PATH in wheel_files
    assert wheel_files >= BUILTIN_RESOURCE_WHEEL_PATHS
    assert not any(path.startswith("tau_coding/data/skills/") for path in wheel_files)
