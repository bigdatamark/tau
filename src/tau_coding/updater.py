# [MODULE DOC]
# Self-update support for the "tau update" command: detects which package
# manager owns the running Tau environment and runs (or, on Windows with uv
# tool installs, schedules) the matching upgrade command.
# Defines injectable callable aliases (command runner, detached launcher,
# executable finder, latest-version fetcher), the InstallMethod literal
# ("uv-tool", "uv-pip", "pipx", "pip"), UpdateResult, the public update_tau and
# detect_install_method functions, an embedded PowerShell handoff script and
# private helpers. Side effects occur only when functions are called
# (subprocesses, temp files, package metadata reads).
"""Upgrade Tau with the package manager that owns its environment."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import Literal

from tau_coding.update_check import PYPI_PACKAGE_NAME, fetch_latest_pypi_version

CommandRunner = Callable[..., CompletedProcess[str]]
DetachedLauncher = Callable[..., object]
ExecutableFinder = Callable[[str], str | None]
LatestVersionFetcher = Callable[[], str | None]
InstallMethod = Literal["uv-tool", "uv-pip", "pipx", "pip"]


@dataclass(frozen=True, slots=True)
class UpdateResult:
    # [CLASS DOC]
    # Frozen, slotted dataclass describing an update attempt: the command that was
    # run or scheduled (none on failure), captured stdout/stderr, failure messages
    # and whether the update was deferred to a detached process.
    """Result of trying to upgrade Tau."""

    command: tuple[str, ...] | None
    stdout: str = ""
    stderr: str = ""
    failures: tuple[str, ...] = ()
    deferred: bool = False

    @property
    def succeeded(self) -> bool:
        # [FUNCTION DOC]
        # Property: true when a command was run/scheduled (command is not none).
        return self.command is not None


def update_tau(
    *,
    runner: CommandRunner = run,
    python_executable: str | None = None,
    environment_prefix: Path | None = None,
    direct_url: str | None = None,
    installer: str | None = None,
    inspect_distribution: bool = True,
    latest_version_fetcher: LatestVersionFetcher = fetch_latest_pypi_version,
    platform_name: str | None = None,
    detached_launcher: DetachedLauncher = subprocess.Popen,
    executable_finder: ExecutableFinder = shutil.which,
    parent_pid: int | None = None,
    handoff_directory: Path | None = None,
) -> UpdateResult:
    # [FUNCTION DOC]
    # Upgrades Tau using the installer that owns the active environment.
    # Inputs (all keyword, injectable for tests): command runner, python
    # executable (default current interpreter), environment prefix (default
    # sys.prefix, resolved), direct URL and installer name (overwritten from the
    # installed distribution metadata when inspect_distribution is true), latest
    # version fetcher, platform name, detached process launcher, executable finder,
    # parent pid and Windows handoff directory.
    # Behavior: refuse with a reinstall hint if installed from a direct/local URL;
    # detect the install method (uv tool receipt, pipx metadata, INSTALLER uv/pip);
    # if unknown, return a Conda/Pixi-specific failure when a conda-meta directory
    # exists, else a generic failure naming the installer. For uv-tool, fetch the
    # latest PyPI version (lookup errors or none become failures) and pin it. Build
    # the upgrade command. On Windows with uv-tool, hand off to a detached
    # PowerShell updater (the running exe cannot be replaced in place). Otherwise
    # run the command capturing text output; OS errors or nonzero exit become a
    # failure with the command line and stderr/stdout/exit code detail.
    # Outputs: an UpdateResult (never raises for expected failures; errors are
    # reported in failures).
    # Side effects: reads distribution metadata, may access the network via the
    # fetcher, runs subprocesses, may write temp files.
    """Upgrade Tau with the installer that owns the active environment.

    Python distributions record their installer in ``INSTALLER``. uv and pipx
    tool environments also leave ownership receipts. Managed, editable,
    direct-URL, and unrecognized installations stop with instructions instead
    of trying unrelated package managers.
    """
    prefix = (environment_prefix or Path(sys.prefix)).resolve()
    if inspect_distribution:
        direct_url = _installed_direct_url()
        installer = _installed_installer()
    if direct_url:
        return _failure(
            "Tau was installed from a local or direct URL, so it cannot be safely "
            f"updated from PyPI. Reinstall it from its original source: {direct_url}"
        )

    method = detect_install_method(prefix, installer=installer)
    if method is None:
        if (prefix / "conda-meta").is_dir():
            return _failure(
                "Tau is installed in a Conda/Pixi-managed environment. "
                "Update it with the manager that created that environment."
            )
        installed_by = installer or "unknown"
        return _failure(
            "Could not identify a supported installer for this Tau environment "
            f"({prefix}). Package metadata reports: {installed_by}."
        )

    latest_version: str | None = None
    if method == "uv-tool":
        try:
            latest_version = latest_version_fetcher()
        except Exception as exc:  # noqa: BLE001 - report update lookup failures to the user
            return _failure(f"Could not determine the latest Tau version from PyPI: {exc}")
        if latest_version is None:
            return _failure("Could not determine the latest Tau version from PyPI.")

    executable = python_executable or sys.executable
    command = _update_command(method, executable, latest_version=latest_version)
    if (platform_name or sys.platform) == "win32" and method == "uv-tool":
        return _handoff_windows_update(
            command,
            launcher=detached_launcher,
            executable_finder=executable_finder,
            parent_pid=parent_pid or os.getpid(),
            handoff_directory=handoff_directory,
        )

    completed = _run(runner, command)
    if isinstance(completed, str):
        return _failure(f"{' '.join(command)}: {completed}")
    if completed.returncode != 0:
        return _failure(f"{' '.join(command)}: {_result_detail(completed)}")
    return UpdateResult(
        command=command,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )


def detect_install_method(prefix: Path, *, installer: str | None = None) -> InstallMethod | None:
    # [FUNCTION DOC]
    # Determines install ownership for an environment prefix: "uv-tool" if a
    # uv-receipt.toml file exists, "pipx" if pipx_metadata.json exists, otherwise
    # "uv-pip" or "pip" from the normalised (stripped, lowercased) installer name,
    # else none. Side effects: filesystem existence checks only.
    """Detect installer ownership from receipts and distribution metadata."""
    if (prefix / "uv-receipt.toml").is_file():
        return "uv-tool"
    if (prefix / "pipx_metadata.json").is_file():
        return "pipx"
    normalized_installer = installer.strip().lower() if installer else None
    if normalized_installer == "uv":
        return "uv-pip"
    if normalized_installer == "pip":
        return "pip"
    return None


def _update_command(
    method: InstallMethod,
    python_executable: str,
    *,
    latest_version: str | None = None,
) -> tuple[str, ...]:
    # [FUNCTION DOC]
    # HELPER: Returns the argv tuple for the method: uv tool install of the pinned
    # package version; uv pip install --python <exe> --upgrade; pipx upgrade; or
    # <python> -m pip install --upgrade. Raises ValueError for uv-tool without a
    # latest version.
    if method == "uv-tool":
        if latest_version is None:
            raise ValueError("latest_version is required for uv tool updates")
        return ("uv", "tool", "install", f"{PYPI_PACKAGE_NAME}@{latest_version}")
    if method == "uv-pip":
        return (
            "uv",
            "pip",
            "install",
            "--python",
            python_executable,
            "--upgrade",
            PYPI_PACKAGE_NAME,
        )
    if method == "pipx":
        return ("pipx", "upgrade", PYPI_PACKAGE_NAME)
    return (python_executable, "-m", "pip", "install", "--upgrade", PYPI_PACKAGE_NAME)


_WINDOWS_UPDATE_SCRIPT = """param(
    [Parameter(Mandatory=$true)][int]$ParentProcessId,
    [Parameter(Mandatory=$true)][string]$LogPath,
    [Parameter(Mandatory=$true)][string]$UpdatePayloadBase64
)

$ScriptPath = $MyInvocation.MyCommand.Path
$UpdateExitCode = 1
function Write-Log([string]$Message) {
    $Message | Add-Content -LiteralPath $LogPath -Encoding UTF8
}
try {
    $ErrorActionPreference = "Stop"
    $WaitMessage = "Waiting for Tau process $ParentProcessId to exit..."
    $WaitMessage | Set-Content -LiteralPath $LogPath -Encoding UTF8
    $ParentProcess = $null
    try {
        $ParentProcess = Get-Process -Id $ParentProcessId -ErrorAction Stop
    } catch {
        if ($_.FullyQualifiedErrorId -like "NoProcessFoundForGivenId*") {
            Write-Log "Tau process $ParentProcessId has already exited."
        } else {
            $Detail = $_.Exception.Message
            throw "Could not inspect Tau process $ParentProcessId. Update aborted: $Detail"
        }
    }
    if ($null -ne $ParentProcess) {
        try {
            Wait-Process -InputObject $ParentProcess -ErrorAction Stop
        } catch {
            $Detail = $_.Exception.Message
            throw "Could not wait for Tau process $ParentProcessId to exit. Update aborted: $Detail"
        }
    }
    $PayloadJson = [Text.Encoding]::UTF8.GetString(
        [Convert]::FromBase64String($UpdatePayloadBase64)
    )
    $UpdatePayload = ConvertFrom-Json -InputObject $PayloadJson
    $StartInfo = New-Object System.Diagnostics.ProcessStartInfo
    $StartInfo.FileName = [string]$UpdatePayload.executable
    $StartInfo.Arguments = [string]$UpdatePayload.arguments
    $StartInfo.UseShellExecute = $false
    $StartInfo.CreateNoWindow = $true
    $StartInfo.RedirectStandardOutput = $true
    $StartInfo.RedirectStandardError = $true
    $UpdateProcess = New-Object System.Diagnostics.Process
    $UpdateProcess.StartInfo = $StartInfo
    if (-not $UpdateProcess.Start()) {
        throw "Could not start the update command. Update aborted."
    }
    $StandardOutputTask = $UpdateProcess.StandardOutput.ReadToEndAsync()
    $StandardErrorTask = $UpdateProcess.StandardError.ReadToEndAsync()
    $UpdateProcess.WaitForExit()
    $StandardOutput = $StandardOutputTask.GetAwaiter().GetResult()
    $StandardError = $StandardErrorTask.GetAwaiter().GetResult()
    if ($StandardOutput) {
        Write-Log $StandardOutput.TrimEnd("`r", "`n")
    }
    if ($StandardError) {
        Write-Log $StandardError.TrimEnd("`r", "`n")
    }
    $UpdateExitCode = $UpdateProcess.ExitCode
    $UpdateProcess.Dispose()
    Write-Log "Update command exited with code $UpdateExitCode."
} catch {
    Write-Log ($_ | Out-String)
} finally {
    Remove-Item -LiteralPath $ScriptPath -Force -ErrorAction SilentlyContinue
}
exit $UpdateExitCode
"""


def _handoff_windows_update(
    command: tuple[str, ...],
    *,
    launcher: DetachedLauncher,
    executable_finder: ExecutableFinder,
    parent_pid: int,
    handoff_directory: Path | None,
) -> UpdateResult:
    # [FUNCTION DOC]
    # HELPER: Schedules a uv tool update to run after the current process exits on
    # Windows. Finds powershell.exe or pwsh.exe (failure if neither); creates a
    # temp "tau-update-" directory or uses/creates the given handoff directory;
    # writes update.ps1 (the embedded script) and chooses update.log as the log
    # path. Encodes the command's executable and Windows-quoted argument string as
    # base64 JSON, then launches PowerShell detached (new process group, no stdio)
    # with the script, parent pid, log path and payload. The script waits for the
    # parent to exit, runs the command, logs output and exit code, and deletes
    # itself. Staging or launch OS errors clean up the script (and the directory
    # when owned) and return a failure.
    # Outputs: a deferred UpdateResult whose stdout tells the user where progress
    # is logged. Side effects: filesystem writes, process spawn.
    powershell = executable_finder("powershell.exe") or executable_finder("pwsh.exe")
    if powershell is None:
        return _failure("Could not find PowerShell to safely finish the Windows update.")

    owns_update_dir = handoff_directory is None
    update_dir: Path | None = None
    script_path: Path | None = None
    try:
        if owns_update_dir:
            update_dir = Path(tempfile.mkdtemp(prefix="tau-update-"))
        else:
            update_dir = handoff_directory
            assert update_dir is not None
            update_dir.mkdir(parents=True, exist_ok=True)
        script_path = update_dir / "update.ps1"
        log_path = update_dir / "update.log"
        script_path.write_text(_WINDOWS_UPDATE_SCRIPT, encoding="utf-8")
    except OSError as exc:
        _cleanup_windows_handoff(update_dir, script_path, remove_directory=owns_update_dir)
        return _failure(f"Could not stage the detached Windows updater: {exc}")

    update_payload = {
        "executable": command[0],
        "arguments": _windows_command_line(command[1:]),
    }
    encoded_payload = base64.b64encode(
        json.dumps(update_payload, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    powershell_command = (
        powershell,
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
        "-ParentProcessId",
        str(parent_pid),
        "-LogPath",
        str(log_path),
        "-UpdatePayloadBase64",
        encoded_payload,
    )
    creationflags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200
    )
    try:
        launcher(
            powershell_command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=creationflags,
        )
    except OSError as exc:
        _cleanup_windows_handoff(update_dir, script_path, remove_directory=owns_update_dir)
        return _failure(f"Could not start the detached Windows updater: {exc}")
    return UpdateResult(
        command=command,
        stdout=(
            "Tau update is scheduled and will start after this process exits. "
            f"Progress will be written to {log_path}."
        ),
        deferred=True,
    )


def _quote_windows_argument(argument: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Quotes one argument per Microsoft C runtime argv rules: always wraps
    # in double quotes, doubles backslashes preceding a quote (plus one to escape
    # the quote) and doubles trailing backslashes before the closing quote. Pure.
    """Quote one argv element using the documented Microsoft C runtime rules."""
    quoted = ['"']
    backslashes = 0
    for character in argument:
        if character == "\\":
            backslashes += 1
            continue
        if character == '"':
            quoted.append("\\" * (backslashes * 2 + 1))
            quoted.append('"')
        else:
            quoted.append("\\" * backslashes)
            quoted.append(character)
        backslashes = 0
    quoted.append("\\" * (backslashes * 2))
    quoted.append('"')
    return "".join(quoted)


def _windows_command_line(arguments: tuple[str, ...]) -> str:
    # [FUNCTION DOC]
    # HELPER: Joins arguments with spaces after quoting each one. Pure.
    """Build the argv tail passed directly to CreateProcess by ProcessStartInfo."""
    return " ".join(_quote_windows_argument(argument) for argument in arguments)


def _cleanup_windows_handoff(
    update_dir: Path | None,
    script_path: Path | None,
    *,
    remove_directory: bool,
) -> None:
    # [FUNCTION DOC]
    # HELPER: Best-effort removal of the staged script file and, when requested,
    # the (expected empty) update directory; OS errors are suppressed.
    if script_path is not None:
        with suppress(OSError):
            script_path.unlink(missing_ok=True)
    if remove_directory and update_dir is not None:
        with suppress(OSError):
            update_dir.rmdir()


def _installed_direct_url() -> str | None:
    # [FUNCTION DOC]
    # HELPER: Reads the installed distribution's direct_url.json. Returns none if
    # absent/empty, the URL string if present, or the phrase "an unrecognized
    # direct source" if the JSON is invalid or lacks a usable url.
    raw = _distribution_file("direct_url.json")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return "an unrecognized direct source"
    url = data.get("url") if isinstance(data, dict) else None
    return url if isinstance(url, str) and url else "an unrecognized direct source"


def _installed_installer() -> str | None:
    # [FUNCTION DOC]
    # HELPER: Returns the stripped contents of the distribution's INSTALLER file,
    # or none when missing or blank.
    raw = _distribution_file("INSTALLER")
    if not raw:
        return None
    return raw.strip() or None


def _distribution_file(filename: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Reads a metadata file from the installed Tau distribution; returns
    # none if the distribution is not installed (or the file is missing).
    try:
        return distribution(PYPI_PACKAGE_NAME).read_text(filename)
    except PackageNotFoundError:
        return None


def _run(runner: CommandRunner, command: tuple[str, ...]) -> CompletedProcess[str] | str:
    # [FUNCTION DOC]
    # HELPER: Runs the command via the runner with captured text output and no
    # check; returns the completed process, or the error string on OSError.
    try:
        return runner(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return str(exc)


def _result_detail(result: CompletedProcess[str]) -> str:
    # [FUNCTION DOC]
    # HELPER: Best failure detail: stripped stderr, else stdout, else the exit
    # code text.
    return result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"


def _failure(message: str) -> UpdateResult:
    # [FUNCTION DOC]
    # HELPER: Builds a failed UpdateResult (no command) with a single message.
    return UpdateResult(command=None, failures=(message,))
