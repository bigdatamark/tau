# [MODULE DOC]
# Tests for the tau command-line app (tau_coding.cli): stream encoding setup,
# version/help, extension install, system-prompt flag resolution, update and
# model-catalog commands, routing to the TUI, print mode and RPC runners (with
# runners monkeypatched to record arguments), print-mode behavior end to end
# with a FakeProvider (output modes, persistence, resume, terminal "!"
# commands, skill expansion, /system inspection, project trust prompts),
# session ID validation, legacy flag migration errors, export, sessions and
# providers listing, setup, TAU_HOME validation, and the thinking flag.
# Uses Typer's CliRunner, temporary directories and pytest monkeypatching.
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import isolate_home
from pi_event_helpers import assistant_done, assistant_error, assistant_start, text_delta
from tau_agent import AssistantMessage, UserMessage
from tau_agent.session import JsonlSessionStorage, MessageEntry, ModelChangeEntry
from tau_ai import (
    FakeProvider,
)
from tau_coding import CodingSessionRecord, SessionManager, cli
from tau_coding.cli import app, run_print_mode
from tau_coding.paths import TauPaths
from tau_coding.provider_config import (
    OpenAICompatibleProviderConfig,
    ProviderSettings,
    load_provider_settings,
)
from tau_coding.rendering import PrintOutputMode
from tau_coding.resources import TauResourcePaths
from tau_coding.skills import load_skills
from tau_coding.system_prompt import (
    BuildSystemPromptOptions,
    build_system_prompt,
    build_system_prompt_inspection,
    format_system_prompt_inspection,
)
from tau_coding.tools import create_coding_tools
from tau_coding.update_check import (
    ReleaseNoteSection,
    ReleaseNotesEntry,
    ReleaseNotesNotice,
    UpdateNotice,
)
from tau_coding.updater import UpdateResult

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(value: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Remove ANSI escape sequences from text. Pure.
    return _ANSI_ESCAPE_RE.sub("", value)


def _collapse_ws(value: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Collapse every whitespace run to a single space. Pure.
    """Collapse all runs of whitespace to single spaces (Rich panel wrapping)."""
    return re.sub(r"\s+", " ", value)


def _panel_text(value: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Normalize Rich/Click error-panel output for matching: strip ANSI
    # escapes first, replace box-drawing border characters with spaces, then
    # collapse whitespace. Pure.
    """Strip ANSI escapes and Rich/Click panel borders, then collapse whitespace.

    Typer renders ``BadParameter`` errors inside a bordered panel whose box-drawing
    characters and line-wrapping can split a single message across lines. On CI
    (no real TTY) Rich/Click also emit ANSI color codes around the wrapped border,
    so the ANSI escapes must be removed *before* the border characters, otherwise
    leftover escapes keep "Available" and "models: qwen" from being contiguous.
    """
    no_ansi = _strip_ansi(value)
    borders = str.maketrans({ch: " " for ch in "│╭╮╰╯─"})
    return _collapse_ws(no_ansi.translate(borders))


def test_force_utf8_streams_reconfigures_non_utf8_streams() -> None:
    # [FUNCTION DOC]
    # Test that non-UTF-8 stdout is reconfigured to UTF-8 with replacement errors
    # and a stream lacking a reconfigure method is skipped without error.
    calls: list[tuple[str, str]] = []

    class FakeStream:
        # [CLASS DOC]
        # HELPER: Fake cp1252 stream recording reconfigure calls.
        encoding = "cp1252"

        def reconfigure(self, *, encoding: str, errors: str) -> None:
            # [FUNCTION DOC]
            # HELPER: Record the requested encoding and error policy.
            calls.append((encoding, errors))

    class UnreconfigurableStream:
        # [CLASS DOC]
        # HELPER: Fake cp437 stream without a reconfigure method.
        """Mimics streams (e.g. some test/CI capture streams) without reconfigure()."""

        encoding = "cp437"

    fake_stdout = FakeStream()
    fake_stderr = UnreconfigurableStream()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli.sys, "stdout", fake_stdout)
        mp.setattr(cli.sys, "stderr", fake_stderr)
        cli._force_utf8_streams()

    assert calls == [("utf-8", "replace")]


def test_force_utf8_streams_leaves_utf8_streams_alone() -> None:
    # [FUNCTION DOC]
    # Test that streams whose encoding is any UTF-8 spelling are not reconfigured.
    calls: list[tuple[str, str]] = []

    class FakeStream:
        # [CLASS DOC]
        # HELPER: Fake stream reporting "UTF_8" and recording reconfigure calls.
        encoding = "UTF_8"

        def reconfigure(self, *, encoding: str, errors: str) -> None:
            # [FUNCTION DOC]
            # HELPER: Record the requested encoding and error policy.
            calls.append((encoding, errors))

    fake_stdout = FakeStream()
    fake_stderr = FakeStream()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli.sys, "stdout", fake_stdout)
        mp.setattr(cli.sys, "stderr", fake_stderr)
        cli._force_utf8_streams()

    assert calls == []


def test_version_command(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that "--version" prints "tau <version>" and exits 0.
    monkeypatch.setattr(cli, "_current_version", lambda: "1.2.3")

    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "tau 1.2.3"


def test_help_lists_system_prompt_options() -> None:
    # [FUNCTION DOC]
    # Test that help lists the system prompt options and the install usage.
    result = CliRunner().invoke(app, ["--help"], env={"COLUMNS": "160"})

    output = re.sub(r"\s+", "", _strip_ansi(result.output))
    assert result.exit_code == 0
    assert "--system-promptTEXT_OR_PATH" in output
    assert "--append-system-promptTEXT_OR_PATH" in output
    assert "tauinstallSOURCE[--force]" in output


def test_install_command_installs_extension(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that "install <source> --force" calls the installer with force, warns
    # about executing arbitrary Python, and reports the destination.
    calls: list[tuple[str, bool]] = []
    destination = tmp_path / ".tau" / "extensions" / "demo"

    def fake_install(source: str, *, force: bool = False) -> Path:
        # [FUNCTION DOC]
        # HELPER: Record (source, force) and return a fixed destination path.
        calls.append((source, force))
        return destination

    monkeypatch.setattr(cli, "install_extension", fake_install)

    result = CliRunner().invoke(
        app,
        ["install", "git:github.com/example/demo", "--force"],
    )

    assert result.exit_code == 0
    assert calls == [("git:github.com/example/demo", True)]
    assert "execute arbitrary Python" in result.output
    assert f"Installed git:github.com/example/demo to {destination}" in result.output


def test_install_command_reports_install_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that an installer error exits 1 with "Could not install extension".
    def fail_install(source: str, *, force: bool = False) -> Path:
        # [FUNCTION DOC]
        # HELPER: Always raise the CLI's extension install error.
        del source, force
        raise cli.ExtensionInstallError("clone failed")

    monkeypatch.setattr(cli, "install_extension", fail_install)

    result = CliRunner().invoke(app, ["install", "git:github.com/example/demo"])

    assert result.exit_code == 1
    assert "Could not install extension: clone failed" in result.output


def test_install_command_requires_exactly_one_source() -> None:
    # [FUNCTION DOC]
    # Test that "install" without a source exits 2 with a usage message.
    result = CliRunner().invoke(app, ["install"])

    assert result.exit_code == 2
    assert "Usage: tau install <source> [--force]" in _panel_text(result.output)


def test_prompt_inputs_resolve_files_literals_and_append_order(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test that prompt values naming files are read (UTF-8), other values are used
    # literally, and multiple append values are joined in order with blank lines.
    base_path = tmp_path / "base.md"
    append_path = tmp_path / "append.md"
    base_path.write_text("File base ü", encoding="utf-8")
    append_path.write_text("File append", encoding="utf-8")

    assert cli._resolve_prompt_input(str(base_path), option="--system-prompt") == "File base ü"
    assert cli._resolve_prompt_input("literal base", option="--system-prompt") == "literal base"
    assert (
        cli._resolve_append_system_prompts(["first", str(append_path), "third"])
        == "first\n\nFile append\n\nthird"
    )


@pytest.mark.parametrize(
    ("option", "expected_base", "expected_append"),
    [
        ("--system-prompt", "~unknown-tau-user/base.md", None),
        ("--append-system-prompt", None, "~unknown-tau-user/append.md"),
    ],
)
def test_unknown_user_prompt_path_is_forwarded_as_literal(
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    expected_base: str | None,
    expected_append: str | None,
) -> None:
    # [FUNCTION DOC]
    # Parametrized test that a "~unknown-user/..." value whose home expansion
    # fails is passed through as literal text to the TUI runner.
    value = expected_base or expected_append
    assert value is not None
    calls: list[tuple[str | None, str | None]] = []
    original_expanduser = Path.expanduser

    def fail_for_unknown_user(path: Path) -> Path:
        # [FUNCTION DOC]
        # HELPER: Replacement path expansion raising RuntimeError for the unknown user
        # prefix and delegating otherwise.
        if str(path).startswith("~unknown-tau-user/"):
            raise RuntimeError("Could not determine home directory")
        return original_expanduser(path)

    async def fake_run_openai_tui(*args: object) -> None:
        # [FUNCTION DOC]
        # HELPER: Record the last two positional arguments (custom and append prompt).
        calls.append((args[-2], args[-1]))  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "expanduser", fail_for_unknown_user)
    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_tui", fake_run_openai_tui)

    result = CliRunner().invoke(app, [option, value, "--new-session"])

    assert result.exit_code == 0
    assert calls == [(expected_base, expected_append)]


def test_prompt_input_reports_invalid_utf8_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that a prompt file with invalid UTF-8 exits 2 with a "Could not read"
    # error naming the option and file.
    prompt_path = tmp_path / "invalid.md"
    prompt_path.write_bytes(b"\xff")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app,
        ["--system-prompt", prompt_path.name, "--new-session"],
    )

    assert result.exit_code == 2
    output = _strip_ansi(result.output)
    assert "--system-prompt" in output
    assert prompt_path.name in output
    assert "Could not read" in output


def test_system_prompt_flags_are_parsed_before_positional_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that prompt flags before positional words are parsed as options and the
    # remaining words form the print-mode prompt.
    base_path = tmp_path / "base.md"
    append_path = tmp_path / "append.md"
    base_path.write_text("Custom base", encoding="utf-8")
    append_path.write_text("second", encoding="utf-8")
    calls: list[tuple[str, str | None, str | None]] = []

    async def fake_run_openai_print_mode(*args: object) -> bool:
        # [FUNCTION DOC]
        # HELPER: Record prompt, custom prompt and append prompt; return success.
        calls.append((str(args[0]), args[-2], args[-1]))  # type: ignore[arg-type]
        return True

    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_print_mode", fake_run_openai_print_mode)

    result = CliRunner().invoke(
        app,
        [
            "--print",
            "--system-prompt",
            str(base_path),
            "--append-system-prompt",
            "first",
            "--append-system-prompt",
            str(append_path),
            "explain",
            "this",
        ],
    )

    assert result.exit_code == 0
    assert calls == [("explain this", "Custom base", "first\n\nsecond")]


def test_prompt_input_reports_existing_unreadable_path(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test that a directory passed as a prompt value exits 2 with "Could not read".
    result = CliRunner().invoke(
        app,
        ["--append-system-prompt", str(tmp_path), "--new-session"],
    )

    assert result.exit_code == 2
    output = _strip_ansi(result.output)
    assert "--append-system-prompt" in output
    assert "Could not read" in output


def test_prompt_input_reports_path_inspection_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that an error while checking whether the prompt path exists exits 2 with
    # "Could not inspect" and the error text, without starting the TUI.
    prompt_path = tmp_path / "base.md"
    prompt_path.write_text("Custom base", encoding="utf-8")
    original_exists = Path.exists
    tui_calls = 0

    def fail_for_prompt_path(path: Path) -> bool:
        # [FUNCTION DOC]
        # HELPER: Replacement existence check raising PermissionError for the prompt
        # path and delegating otherwise.
        if path == prompt_path:
            raise PermissionError("permission denied")
        return original_exists(path)

    async def fake_run_openai_tui(*args: object) -> None:
        # [FUNCTION DOC]
        # HELPER: Count TUI runner invocations.
        nonlocal tui_calls
        tui_calls += 1

    monkeypatch.setattr(Path, "exists", fail_for_prompt_path)
    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_tui", fake_run_openai_tui)

    result = CliRunner().invoke(
        app,
        ["--system-prompt", str(prompt_path), "--new-session"],
        env={"COLUMNS": "300"},
    )

    assert result.exit_code == 2
    output = _strip_ansi(result.output)
    compact_output = re.sub(r"\s+", "", output)
    assert "--system-prompt" in output
    assert str(prompt_path) in compact_output
    assert "Could not inspect" in output
    assert "permission denied" in output
    assert tui_calls == 0


def test_system_prompt_flags_forward_to_resumed_tui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC]
    # Test that prompt flags are forwarded along with the session ID when resuming.
    calls: list[tuple[str | None, str | None, str | None]] = []

    async def fake_run_openai_tui(*args: object) -> None:
        # [FUNCTION DOC]
        # HELPER: Record session ID, custom prompt and append prompt.
        calls.append((args[2], args[-2], args[-1]))  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_tui", fake_run_openai_tui)

    result = CliRunner().invoke(
        app,
        [
            "--session",
            "session-1",
            "--system-prompt",
            "Resume base",
            "--append-system-prompt",
            "Resume append",
        ],
    )

    assert result.exit_code == 0
    assert calls == [("session-1", "Resume base", "Resume append")]


def test_version_command_does_not_check_for_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that "--version" never performs a startup update check.
    monkeypatch.setattr(cli, "_current_version", lambda: "1.2.3")
    monkeypatch.setattr(
        cli,
        "_startup_update_notice",
        lambda: (_ for _ in ()).throw(AssertionError("no update check")),
    )

    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "tau 1.2.3"


def test_update_command_upgrades_without_startup_check(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that "update" runs the updater, prints its output and the command used,
    # and skips the startup update check.
    monkeypatch.setattr(
        cli,
        "_startup_update_notice",
        lambda: (_ for _ in ()).throw(AssertionError("no update check")),
    )
    monkeypatch.setattr(
        cli,
        "update_tau",
        lambda: UpdateResult(
            command=("uv", "tool", "install", "tau-ai@0.2.4"),
            stdout="Updated tau-ai",
        ),
    )

    result = CliRunner().invoke(app, ["update"])

    assert result.exit_code == 0
    assert "Updated tau-ai" in result.stdout
    assert "Tau update completed with: uv tool install tau-ai@0.2.4" in result.stdout


def test_update_models_force_refreshes_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that "update --models" force-refreshes the model catalog and reports
    # the model count.
    calls: list[bool] = []

    async def refresh_models(*, force: bool) -> cli.ModelsDevRefreshResult:
        # [FUNCTION DOC]
        # HELPER: Record the force flag and return a refreshed result with 42 models.
        calls.append(force)
        return cli.ModelsDevRefreshResult(
            refreshed=True,
            not_modified=False,
            model_count=42,
            cache_path=Path("/tmp/models-store.json"),
        )

    monkeypatch.setattr(cli, "refresh_models_dev_catalog", refresh_models)

    result = CliRunner().invoke(app, ["update", "--models"])

    assert result.exit_code == 0
    assert calls == [True]
    assert "Model catalogs refreshed: 42 models" in result.stdout


def test_update_command_reports_windows_handoff_without_claiming_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC]
    # Test that a deferred (scheduled) update is reported as handed off, not
    # completed.
    monkeypatch.setattr(
        cli,
        "update_tau",
        lambda: UpdateResult(
            command=("uv", "tool", "install", "tau-ai@0.2.4"),
            stdout="Tau update is scheduled and will start after this process exits.",
            deferred=True,
        ),
    )

    result = CliRunner().invoke(app, ["update"])

    assert result.exit_code == 0
    assert "scheduled" in result.stdout
    assert "Tau update handed off with:" in result.stdout
    assert "Tau update completed" not in result.stdout


def test_update_command_reports_installer_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that installer failures exit 1 with messages on stderr.
    monkeypatch.setattr(
        cli,
        "update_tau",
        lambda: UpdateResult(command=None, failures=("uv: not found", "pipx: not found")),
    )

    result = CliRunner().invoke(app, ["update"])

    assert result.exit_code == 1
    assert "Could not safely update Tau" in result.stderr
    assert "uv: not found" in result.stderr


def test_print_mode_writes_update_notice_to_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that text print mode writes the update-available notice to stderr.
    async def fake_run_openai_print_mode(
        prompt: str,
        model: str | None,
        cwd: Path,
        output: PrintOutputMode,
        provider_name: str | None,
        *extra: object,
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: Accept print-mode arguments and return success.
        del prompt, model, cwd, output, provider_name, extra
        return True

    monkeypatch.setattr(
        cli,
        "_startup_update_notice",
        lambda: UpdateNotice(current_version="0.1.0", latest_version="0.2.0"),
    )
    monkeypatch.setattr(cli, "run_openai_print_mode", fake_run_openai_print_mode)

    result = CliRunner().invoke(app, ["-p", "hello"])

    assert result.exit_code == 0
    assert "Tau 0.2.0 is available (installed: 0.1.0)" in result.stderr


def test_json_print_mode_suppresses_update_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that JSON print mode emits no update notice at all.
    async def fake_run_openai_print_mode(
        prompt: str,
        model: str | None,
        cwd: Path,
        output: PrintOutputMode,
        provider_name: str | None,
        *extra: object,
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: Accept print-mode arguments and return success.
        del prompt, model, cwd, output, provider_name, extra
        return True

    monkeypatch.setattr(
        cli,
        "_startup_update_notice",
        lambda: UpdateNotice(current_version="0.1.0", latest_version="0.2.0"),
    )
    monkeypatch.setattr(cli, "run_openai_print_mode", fake_run_openai_print_mode)

    result = CliRunner().invoke(app, ["-p", "--mode", "json", "hello"])

    assert result.exit_code == 0
    assert result.stderr == ""


def test_utility_command_does_not_check_for_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that "sessions" skips the update check and reports no sessions.
    monkeypatch.setattr(
        cli,
        "_startup_update_notice",
        lambda: (_ for _ in ()).throw(AssertionError("no update check")),
    )
    monkeypatch.setattr(cli.SessionManager, "list_sessions", lambda self: [])

    result = CliRunner().invoke(app, ["sessions"])

    assert result.exit_code == 0
    assert "No sessions found." in result.stdout


def test_cli_without_prompt_invokes_tui_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that no arguments start the TUI with default options and cwd.
    calls: list[tuple[str | None, Path, str | None, bool, str | None, int | None, str | None]] = []

    async def fake_run_openai_tui(
        model: str | None,
        cwd: Path,
        session_id: str | None,
        new_session: bool,
        provider_name: str | None,
        auto_compact_token_threshold: int | None,
        initial_prompt: str | None,
        update_notice: object | None = None,
        *extra: object,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Record the first seven positional TUI runner arguments.
        del update_notice, extra
        calls.append(
            (
                model,
                cwd,
                session_id,
                new_session,
                provider_name,
                auto_compact_token_threshold,
                initial_prompt,
            )
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_tui", fake_run_openai_tui)

    result = CliRunner().invoke(app, [])

    assert result.exit_code == 0
    assert calls == [(None, tmp_path, None, False, None, None, None)]


def test_cli_prints_resume_hint_after_tui_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that a session ID returned by the TUI runner prints a resume hint.
    async def fake_run_openai_tui(*args: object, **kwargs: object) -> str:
        # [FUNCTION DOC]
        # HELPER: Return a fixed session ID.
        del args, kwargs
        return "session-123"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_tui", fake_run_openai_tui)

    result = CliRunner().invoke(app, [])

    assert result.exit_code == 0
    assert result.stdout == "To resume this session: tau --session session-123\n"


def test_cli_suppresses_resume_hint_without_persisted_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that no hint is printed when the TUI runner returns no session.
    async def fake_run_openai_tui(*args: object, **kwargs: object) -> None:
        # [FUNCTION DOC]
        # HELPER: Return nothing.
        del args, kwargs

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_tui", fake_run_openai_tui)

    result = CliRunner().invoke(app, [])

    assert result.exit_code == 0
    assert result.stdout == ""


def test_cli_positional_prompt_invokes_tui_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that a positional prompt without print flags starts the TUI with that
    # initial prompt.
    calls: list[tuple[str | None, Path, str | None, bool, str | None, int | None, str | None]] = []

    async def fake_run_openai_tui(
        model: str | None,
        cwd: Path,
        session_id: str | None,
        new_session: bool,
        provider_name: str | None,
        auto_compact_token_threshold: int | None,
        initial_prompt: str | None,
        update_notice: object | None = None,
        *extra: object,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Record the first seven positional TUI runner arguments.
        del update_notice, extra
        calls.append(
            (
                model,
                cwd,
                session_id,
                new_session,
                provider_name,
                auto_compact_token_threshold,
                initial_prompt,
            )
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_tui", fake_run_openai_tui)

    result = CliRunner().invoke(app, ["explain this repo"])

    assert result.exit_code == 0
    assert calls == [(None, tmp_path, None, False, None, None, "explain this repo")]


@pytest.mark.anyio
async def test_run_openai_tui_combines_release_notes_and_update_notice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that the TUI runner passes a formatted update notice, release-notes
    # startup notice and system prompt options to the TUI app.
    calls: list[tuple[str | None, tuple[str, ...], str | None, str | None]] = []

    async def fake_run_tui_app(**kwargs: object) -> None:
        # [FUNCTION DOC]
        # HELPER: Record selected keyword arguments passed to the TUI app.
        calls.append(  # type: ignore[arg-type]
            (
                kwargs["startup_update_notice"],
                kwargs["startup_notices"],
                kwargs["custom_system_prompt"],
                kwargs["append_system_prompt"],
            )
        )

    monkeypatch.setattr(cli, "run_tui_app", fake_run_tui_app)
    monkeypatch.setattr(cli, "_current_version", lambda: "0.1.2")
    monkeypatch.setattr(
        cli,
        "startup_release_notes_notice",
        lambda version: ReleaseNotesNotice(
            current_version=version,
            previous_version="0.1.1",
            entries=(
                ReleaseNotesEntry(
                    version=version,
                    date=None,
                    sections=(ReleaseNoteSection(title="New", items=("Release note",)),),
                ),
            ),
        ),
    )

    await cli.run_openai_tui(
        model=None,
        cwd=tmp_path,
        update_notice=UpdateNotice(current_version="0.1.2", latest_version="0.1.3"),
        custom_system_prompt="Custom base",
        append_system_prompt="Custom append",
    )

    assert calls == [
        (
            "Tau 0.1.3 is available (installed: 0.1.2). Run `tau update` to upgrade.",
            ("Tau updated to 0.1.2\n\n**New**\n- Release note",),
            "Custom base",
            "Custom append",
        )
    ]


@pytest.mark.anyio
async def test_run_print_mode_prints_final_assistant_text(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that print mode prints the final assistant text, sends the default
    # system prompt and the four default tools to the provider.
    provider = FakeProvider(
        [
            [
                assistant_start(model="fake"),
                text_delta(delta="Hel"),
                text_delta(delta="lo"),
                assistant_done(message=AssistantMessage(content="Hello")),
            ]
        ]
    )

    ok = await run_print_mode(
        prompt="Say hello",
        model="fake",
        cwd=tmp_path,
        provider=provider,
        resource_paths=TauResourcePaths(root=tmp_path / "resources", agents_root=None),
    )

    captured = capsys.readouterr()
    assert ok is True
    assert captured.out == "Hello\n"
    assert captured.err == ""
    assert provider.calls[0][0] == "fake"
    resource_paths = TauResourcePaths(root=tmp_path / "resources", agents_root=None)
    assert provider.calls[0][1] == build_system_prompt(
        BuildSystemPromptOptions(
            cwd=tmp_path,
            tools=create_coding_tools(cwd=tmp_path),
            skills=load_skills(resource_paths),
        )
    )
    assert [tool.name for tool in provider.calls[0][3]] == ["read", "write", "edit", "bash"]


@pytest.mark.anyio
async def test_run_print_mode_uses_custom_and_appended_system_prompt(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that a custom base prompt plus appended prompts replace the default base
    # while project instructions, skills, date and cwd are still included.
    resource_root = tmp_path / "resources"
    skill_dir = resource_root / "skills" / "testing"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\ndescription: Test code\n---\n# Testing",
        encoding="utf-8",
    )
    (tmp_path / "AGENTS.md").write_text("Follow project rules.", encoding="utf-8")
    provider = FakeProvider(
        [[assistant_start(model="fake"), assistant_done(message=AssistantMessage(content="Done"))]]
    )

    ok = await run_print_mode(
        prompt="Hello",
        model="fake",
        cwd=tmp_path,
        provider=provider,
        resource_paths=TauResourcePaths(root=resource_root, agents_root=None),
        custom_system_prompt="Custom base.",
        append_system_prompt="First append.\n\nSecond append.",
        trust_default="always",
    )

    _captured = capsys.readouterr()
    system = provider.calls[0][1]
    assert ok is True
    assert system.startswith("Custom base.\n\nFirst append.\n\nSecond append.")
    assert "You are an expert coding assistant operating inside Tau" not in system
    assert "Follow project rules." in system
    assert "<available_skills>" in system
    assert "Current date:" in system
    assert f"Current working directory: {tmp_path}" in system


@pytest.mark.anyio
async def test_run_print_mode_system_command_prints_prompt_without_provider_call(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that "/system" prints the system prompt inspection without calling the
    # provider or writing session entries.
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = FakeProvider([])

    ok = await run_print_mode(
        prompt="/system",
        model="fake",
        cwd=tmp_path,
        provider=provider,
        storage=storage,
        resource_paths=TauResourcePaths(root=tmp_path / "resources", agents_root=None),
    )

    captured = capsys.readouterr()
    inspection = build_system_prompt_inspection(
        BuildSystemPromptOptions(
            cwd=tmp_path,
            tools=create_coding_tools(cwd=tmp_path),
            skills=load_skills(TauResourcePaths(root=tmp_path / "resources", agents_root=None)),
        )
    )
    assert ok is True
    assert captured.out == f"{format_system_prompt_inspection(inspection)}\n"
    assert captured.err == ""
    assert provider.calls == []
    assert await storage.read_all() == []


@pytest.mark.anyio
async def test_run_print_mode_fails_on_non_recoverable_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that a provider error makes print mode return False and print the error
    # to stderr.
    provider = FakeProvider(
        [
            [
                assistant_start(model="fake"),
                assistant_error(message="provider failed"),
            ]
        ]
    )

    ok = await run_print_mode(prompt="Say hello", model="fake", cwd=tmp_path, provider=provider)

    captured = capsys.readouterr()
    assert ok is False
    assert captured.out == ""
    assert "Error: provider failed" in captured.err


@pytest.mark.anyio
async def test_run_print_mode_includes_discovered_context(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that a trusted project's AGENTS.md is included in the system prompt with
    # a project_instructions tag.
    (tmp_path / "AGENTS.md").write_text("Use the local rules.", encoding="utf-8")
    provider = FakeProvider(
        [
            [
                assistant_start(model="fake"),
                assistant_done(message=AssistantMessage(content="Done")),
            ]
        ]
    )

    ok = await run_print_mode(
        prompt="Say hello",
        model="fake",
        cwd=tmp_path,
        provider=provider,
        resource_paths=TauResourcePaths(root=tmp_path / "resources", agents_root=None),
        trust_default="always",
    )

    _captured = capsys.readouterr()
    assert ok is True
    assert "Use the local rules." in provider.calls[0][1]
    assert f'<project_instructions path="{tmp_path / "AGENTS.md"}">' in provider.calls[0][1]


@pytest.mark.anyio
async def test_run_print_mode_persists_session_entries(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that the user and assistant messages are persisted and no leaf entry is
    # written.
    storage = JsonlSessionStorage(tmp_path / "print-session.jsonl")
    provider = FakeProvider(
        [
            [
                assistant_start(model="fake"),
                assistant_done(message=AssistantMessage(content="Done")),
            ]
        ]
    )

    ok = await run_print_mode(
        prompt="Say hello",
        model="fake",
        cwd=tmp_path,
        provider=provider,
        storage=storage,
    )

    _captured = capsys.readouterr()
    entries = await storage.read_all()
    messages = [entry.message for entry in entries if isinstance(entry, MessageEntry)]

    assert ok is True
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "Say hello"
    assert messages[1].text == "Done"
    assert not any(entry.type == "leaf" for entry in entries)


@pytest.mark.anyio
async def test_run_print_mode_resumes_persisted_conversation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # [FUNCTION DOC]
    # Test that a resumed session sends prior messages plus the new prompt, with
    # the startup model override taking precedence over the stored model.
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    user_entry = MessageEntry(message=UserMessage(content="First question"))
    assistant_entry = MessageEntry(
        parent_id=user_entry.id,
        message=AssistantMessage(content="First answer"),
    )
    await storage.append(user_entry)
    await storage.append(assistant_entry)
    await storage.append(ModelChangeEntry(parent_id=assistant_entry.id, model="model-a"))
    provider = FakeProvider(
        [
            [
                assistant_start(model="model-b"),
                assistant_done(message=AssistantMessage(content="Done")),
            ]
        ]
    )

    ok = await run_print_mode(
        prompt="Follow-up message",
        model="model-b",
        cwd=tmp_path,
        provider=provider,
        storage=storage,
        session_id="session-123",
        startup_model_override=True,
    )

    assert ok is True
    assert capsys.readouterr().out == "Done\n"
    assert provider.calls[0][0] == "model-b"
    messages = provider.calls[0][2]
    assert [(message.role, message.text) for message in messages] == [
        ("user", "First question"),
        ("assistant", "First answer"),
        ("user", "Follow-up message"),
    ]


@pytest.mark.anyio
async def test_run_print_mode_terminal_command_adds_context(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that a "!" prefixed prompt runs the shell command, prints it, and adds
    # its output as a user context message without calling the provider.
    storage = JsonlSessionStorage(tmp_path / "print-session.jsonl")
    provider = FakeProvider([])

    ok = await run_print_mode(
        prompt="! printf hello",
        model="fake",
        cwd=tmp_path,
        provider=provider,
        storage=storage,
    )

    captured = capsys.readouterr()
    entries = await storage.read_all()
    messages = [entry.message for entry in entries if isinstance(entry, MessageEntry)]

    assert ok is True
    assert "$ printf hello" in captured.out
    assert "[added to context]" in captured.out
    assert "hello" in captured.out
    assert len(messages) == 1
    assert "Terminal command executed by the user." in messages[0].content
    assert provider.calls == []


@pytest.mark.anyio
async def test_run_print_mode_terminal_command_can_skip_context(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that "!!" runs the command without adding it to context.
    storage = JsonlSessionStorage(tmp_path / "print-session.jsonl")
    provider = FakeProvider([])

    ok = await run_print_mode(
        prompt="!! printf hidden",
        model="fake",
        cwd=tmp_path,
        provider=provider,
        storage=storage,
    )

    captured = capsys.readouterr()
    entries = await storage.read_all()

    assert ok is True
    assert "$ printf hidden" in captured.out
    assert "[not added to context]" in captured.out
    assert "hidden" in captured.out
    assert not any(isinstance(entry, MessageEntry) for entry in entries)
    assert provider.calls == []


@pytest.mark.anyio
async def test_run_print_mode_expands_skill_commands(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that "/skill:<name> args" expands to the skill block followed by args.
    resource_root = tmp_path / "resources"
    skills_dir = resource_root / "skills" / "testing"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("# Testing\nRun pytest.", encoding="utf-8")
    provider = FakeProvider(
        [
            [
                assistant_start(model="fake"),
                assistant_done(message=AssistantMessage(content="Done")),
            ]
        ]
    )

    ok = await run_print_mode(
        prompt="/skill:testing add tests",
        model="fake",
        cwd=tmp_path,
        provider=provider,
        resource_paths=TauResourcePaths(root=resource_root, agents_root=None),
    )

    _captured = capsys.readouterr()

    assert ok is True
    assert '<skill name="testing" location="' in provider.calls[0][2][0].content
    assert "References are relative to" in provider.calls[0][2][0].content
    assert provider.calls[0][2][0].content.endswith("</skill>\n\nadd tests")


@pytest.mark.anyio
async def test_run_print_mode_can_emit_json_events(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that JSON output mode prints Pi-style JSON events on stdout.
    provider = FakeProvider(
        [
            [
                assistant_start(model="fake"),
                text_delta(delta="Hello"),
                assistant_done(message=AssistantMessage(content="Hello")),
            ]
        ]
    )

    ok = await run_print_mode(
        prompt="Say hello",
        model="fake",
        cwd=tmp_path,
        provider=provider,
        output=PrintOutputMode.json,
    )

    captured = capsys.readouterr()
    assert ok is True
    assert '"type":"agent_start"' in captured.out
    assert '"type":"message_update"' in captured.out
    assert '"assistantMessageEvent":{"type":"text_delta"' in captured.out
    assert captured.err == ""


@pytest.mark.anyio
async def test_run_print_mode_can_emit_live_transcript(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that transcript output mode prints the streamed text.
    provider = FakeProvider(
        [
            [
                assistant_start(model="fake"),
                text_delta(delta="Hel"),
                text_delta(delta="lo"),
                assistant_done(message=AssistantMessage(content="Hello")),
            ]
        ]
    )

    ok = await run_print_mode(
        prompt="Say hello",
        model="fake",
        cwd=tmp_path,
        provider=provider,
        output=PrintOutputMode.transcript,
    )

    captured = capsys.readouterr()
    assert ok is True
    assert captured.out == "Hello\n"
    assert captured.err == ""


@pytest.mark.parametrize("mode", ["text", "json", "transcript"])
def test_print_mode_passes_exact_session_id_without_changing_output(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    # [FUNCTION DOC]
    # Parametrized test that "--session-id" is forwarded verbatim in every output
    # mode without producing output.
    calls: list[str | None] = []

    async def fake_run_openai_print_mode(
        prompt: str,
        model: str | None,
        cwd: Path,
        output: PrintOutputMode,
        provider_name: str | None,
        session_manager: SessionManager | None,
        extension_paths: tuple[Path, ...],
        extensions_enabled: bool,
        project_extensions_enabled: bool,
        session_id: str | None,
        custom_system_prompt: str | None,
        append_system_prompt: str | None,
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: Record the session ID argument and return success.
        del (
            prompt,
            model,
            cwd,
            output,
            provider_name,
            session_manager,
            extension_paths,
            extensions_enabled,
            project_extensions_enabled,
            custom_system_prompt,
            append_system_prompt,
        )
        calls.append(session_id)
        return True

    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_print_mode", fake_run_openai_print_mode)

    result = CliRunner().invoke(
        app,
        ["--mode", mode, "--new-session", "--session-id", "worker-499", "hello"],
    )

    assert result.exit_code == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert calls == ["worker-499"]


def test_print_mode_passes_session_id_for_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that "--print --session" forwards the resume session ID with the prompt.
    calls: list[tuple[str, str | None]] = []

    async def fake_run_openai_print_mode(
        prompt: str,
        model: str | None,
        cwd: Path,
        output: PrintOutputMode,
        provider_name: str | None,
        session_manager: SessionManager | None,
        extension_paths: tuple[Path, ...],
        extensions_enabled: bool,
        project_extensions_enabled: bool,
        session_id: str | None,
        custom_system_prompt: str | None,
        append_system_prompt: str | None,
        trust_override: object | None,
        resume_session_id: str | None,
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: Record (prompt, resume session ID) and return success.
        del (
            model,
            cwd,
            output,
            provider_name,
            session_manager,
            extension_paths,
            extensions_enabled,
            project_extensions_enabled,
            session_id,
            custom_system_prompt,
            append_system_prompt,
            trust_override,
        )
        calls.append((prompt, resume_session_id))
        return True

    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_print_mode", fake_run_openai_print_mode)

    result = CliRunner().invoke(app, ["--print", "--session", "session-123", "follow up"])

    assert result.exit_code == 0
    assert calls == [("follow up", "session-123")]


def test_print_mode_rejects_session_and_new_session() -> None:
    # [FUNCTION DOC]
    # Test that combining --session and --new-session exits 2 with an error.
    result = CliRunner().invoke(
        app, ["--print", "--session", "session-123", "--new-session", "follow up"]
    )

    assert result.exit_code == 2
    assert "--session and --new-session cannot be used together" in _strip_ansi(result.output)


def test_print_mode_rejects_session_and_session_id() -> None:
    # [FUNCTION DOC]
    # Test that combining --session and --session-id exits 2 with an error.
    result = CliRunner().invoke(
        app,
        ["--print", "--session", "session-123", "--session-id", "new-id", "follow up"],
    )

    assert result.exit_code == 2
    assert "--session and --session-id cannot be used together" in _strip_ansi(result.output)


@pytest.mark.parametrize(
    ("session_id", "error"),
    [
        ("", "Session id must be non-empty"),
        ("-bad", "Session id must be non-empty"),
        ("bad id", "Session id must be non-empty"),
        ("bad/", "Session id must be non-empty"),
        ("index", "Session id is reserved: index"),
        ("a" * 129, "Session id must be at most 128 bytes"),
    ],
)
def test_print_mode_rejects_invalid_session_id(session_id: str, error: str) -> None:
    # [FUNCTION DOC]
    # Parametrized test of invalid session IDs (empty, leading dash, spaces,
    # slash, reserved "index", more than 128 bytes) exiting 2 with messages.
    result = CliRunner().invoke(app, ["-p", "--session-id", session_id, "hello"])

    assert result.exit_code == 2
    assert error in _strip_ansi(result.output)


def test_session_id_is_print_mode_only() -> None:
    # [FUNCTION DOC]
    # Test that --session-id without print mode exits 2.
    result = CliRunner().invoke(app, ["--session-id", "worker-499"])

    assert result.exit_code == 2
    assert "--session-id is only supported in print mode" in _strip_ansi(result.output)


def test_print_session_record_resumes_existing_session(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test that resolving a print session by ID returns the existing record even
    # from another cwd.
    manager = SessionManager(TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents"))
    record = manager.create_session(
        cwd=tmp_path,
        model="fake",
        session_id="session-123",
    )

    resumed = cli._print_session_record(
        manager,
        resume_session_id="session-123",
        cwd=tmp_path / "other",
        settings=_constrained_provider_settings(),
        provider_name=None,
        model=None,
        session_id=None,
    )

    assert resumed == record


def test_print_session_record_rejects_unknown_session(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test that an unknown resume ID raises ValueError.
    manager = SessionManager(TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents"))

    with pytest.raises(ValueError, match="Unknown session: missing"):
        cli._print_session_record(
            manager,
            resume_session_id="missing",
            cwd=tmp_path,
            settings=_constrained_provider_settings(),
            provider_name=None,
            model=None,
            session_id=None,
        )


@pytest.mark.anyio
async def test_print_resume_does_not_apply_hf_route_to_explicit_non_hf_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that resuming a Hugging Face session with an explicit non-HF provider
    # creates that provider without the stored inference route, runs with the
    # resumed storage, and closes the provider afterwards.
    model = "shared-model"
    settings = ProviderSettings(
        default_provider="huggingface",
        providers=(
            OpenAICompatibleProviderConfig(
                name="huggingface",
                models=(model,),
                default_model=model,
                inference_providers={model: "together"},
            ),
            OpenAICompatibleProviderConfig(
                name="local",
                base_url="http://localhost:11434/v1",
                api_key_env="LOCAL_API_KEY",
                models=(model,),
                default_model=model,
            ),
        ),
    )
    manager = SessionManager(TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents"))
    manager.create_session(
        cwd=tmp_path,
        model=model,
        provider_name="huggingface",
        inference_provider="together",
        session_id="session-123",
    )

    lifecycle: list[str] = []

    class ClosableFakeProvider(FakeProvider):
        # [CLASS DOC]
        # HELPER: FakeProvider that records when it is closed.
        async def aclose(self) -> None:
            # [FUNCTION DOC]
            # HELPER: Append "provider_closed" to the lifecycle list.
            lifecycle.append("provider_closed")

    provider = ClosableFakeProvider([])
    create_calls: list[tuple[str, str | None]] = []

    def fake_create_model_provider(
        provider_config: OpenAICompatibleProviderConfig,
        *,
        model: str,
        inference_provider: str | None,
        **kwargs: object,
    ) -> ClosableFakeProvider:
        # [FUNCTION DOC]
        # HELPER: Record provider name and inference route and return the fake.
        del model, kwargs
        lifecycle.append("provider_created")
        create_calls.append((provider_config.name, inference_provider))
        return provider

    async def fake_run_print_mode(**kwargs: object) -> bool:
        # [FUNCTION DOC]
        # HELPER: Assert the run uses the fake provider, "local" and the resumed
        # session file; record the run and return success.
        lifecycle.append("session_run")
        resumed_record = manager.get_session("session-123")
        storage = kwargs["storage"]
        assert resumed_record is not None
        assert isinstance(storage, JsonlSessionStorage)
        assert kwargs["provider"] is provider
        assert kwargs["provider_name"] == "local"
        assert storage.path == resumed_record.path
        return True

    monkeypatch.setattr(cli, "load_provider_settings", lambda: settings)
    monkeypatch.setattr(cli, "create_model_provider", fake_create_model_provider)
    monkeypatch.setattr(cli, "run_print_mode", fake_run_print_mode)

    ok = await cli.run_openai_print_mode(
        "Follow up",
        model,
        tmp_path,
        provider_name="local",
        session_manager=manager,
        resume_session_id="session-123",
    )

    assert ok is True
    assert create_calls == [("local", None)]
    assert lifecycle == ["provider_created", "session_run", "provider_closed"]


def test_create_print_session_uses_requested_id_and_rejects_collision(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test that a print session can be created with a chosen ID and a second
    # creation with the same ID raises RuntimeError.
    manager = SessionManager(TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents"))

    record = cli._create_print_session(
        manager,
        cwd=tmp_path,
        model="fake",
        session_id="worker-499",
    )

    assert record.id == "worker-499"
    assert record.path.name == "worker-499.jsonl"
    with pytest.raises(RuntimeError, match="Session already exists with id 'worker-499'"):
        cli._create_print_session(
            manager,
            cwd=tmp_path,
            model="fake",
            session_id="worker-499",
        )


def test_cli_exits_nonzero_when_print_mode_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that a failing print run exits with code 1.
    async def fake_run_openai_print_mode(
        prompt: str,
        model: str | None,
        cwd: Path,
        output: PrintOutputMode,
        provider_name: str | None,
        *extra: object,
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: Return failure.
        return False

    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_print_mode", fake_run_openai_print_mode)

    result = CliRunner().invoke(app, ["-p", "hello"])

    assert result.exit_code == 1


def test_default_tui_invokes_tui_runner_with_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that cwd, model, provider, session and compaction threshold flags are
    # forwarded to the TUI runner.
    calls: list[tuple[str | None, Path, str | None, bool, str | None, int | None, str | None]] = []

    async def fake_run_openai_tui(
        model: str | None,
        cwd: Path,
        session_id: str | None,
        new_session: bool,
        provider_name: str | None,
        auto_compact_token_threshold: int | None,
        initial_prompt: str | None,
        update_notice: object | None = None,
        *extra: object,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Record the first seven positional TUI runner arguments.
        del update_notice, extra
        calls.append(
            (
                model,
                cwd,
                session_id,
                new_session,
                provider_name,
                auto_compact_token_threshold,
                initial_prompt,
            )
        )

    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_tui", fake_run_openai_tui)

    result = CliRunner().invoke(
        app,
        [
            "--cwd",
            str(tmp_path),
            "--model",
            "fake",
            "--provider",
            "local",
            "--session",
            "session-1",
            "--auto-compact-threshold",
            "1000",
        ],
    )

    assert result.exit_code == 0
    assert calls == [("fake", tmp_path, "session-1", False, "local", 1000, None)]


def test_default_tui_rejects_session_with_new_session(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test that --session with --new-session fails in TUI mode.
    result = CliRunner().invoke(
        app,
        [
            "--cwd",
            str(tmp_path),
            "--session",
            "session-1",
            "--new-session",
        ],
    )

    assert result.exit_code != 0
    assert "--session and --new-session cannot be used together" in _strip_ansi(result.output)


def test_legacy_resume_flag_errors_with_migration_hint(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test that the removed --resume flag errors with a hint to use --session.
    result = CliRunner().invoke(
        app,
        [
            "--cwd",
            str(tmp_path),
            "--resume",
            "session-1",
        ],
    )

    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "--resume was renamed to --session" in output
    assert "session-1" in output


def test_legacy_prompt_flag_errors_with_migration_hint() -> None:
    # [FUNCTION DOC]
    # Test that the removed --prompt flag errors with a hint to use --print.
    result = CliRunner().invoke(app, ["--prompt", "hello"])

    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "--prompt was removed" in output
    assert "--print" in output


def test_legacy_output_flag_errors_with_migration_hint() -> None:
    # [FUNCTION DOC]
    # Test that --output errors with a hint that it was renamed to --mode.
    result = CliRunner().invoke(app, ["-p", "--output", "json", "hello"])

    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "--output was renamed to --mode" in output


def test_legacy_extension_short_flag_errors_with_migration_hint(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test that -x errors with a hint that it was renamed to -e/--extension.
    result = CliRunner().invoke(app, ["-x", str(tmp_path)])

    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "-x was renamed to -e/--extension" in output


def test_mode_flag_alone_triggers_print_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that --mode json with a prompt enters print mode in JSON output.
    calls: list[tuple[str, PrintOutputMode]] = []

    async def fake_run_openai_print_mode(
        prompt: str,
        model: str | None,
        cwd: Path,
        output: PrintOutputMode,
        provider_name: str | None,
        *extra: object,
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: Record (prompt, output mode) and return success.
        del model, cwd, provider_name, extra
        calls.append((prompt, output))
        return True

    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_print_mode", fake_run_openai_print_mode)

    result = CliRunner().invoke(app, ["--mode", "json", "hello"])

    assert result.exit_code == 0
    assert calls == [("hello", PrintOutputMode.json)]


def test_print_mode_requires_a_prompt() -> None:
    # [FUNCTION DOC]
    # Test that -p without prompt or stdin input fails with print usage.
    result = CliRunner().invoke(app, ["-p"])

    assert result.exit_code != 0
    assert "Usage: tau --print" in _strip_ansi(result.output)


def test_print_mode_merges_piped_stdin_into_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that piped stdin is placed before the prompt argument separated by
    # blank lines.
    calls: list[str] = []

    async def fake_run_openai_print_mode(
        prompt: str,
        model: str | None,
        cwd: Path,
        output: PrintOutputMode,
        provider_name: str | None,
        *extra: object,
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: Record the prompt and return success.
        del model, cwd, output, provider_name, extra
        calls.append(prompt)
        return True

    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_print_mode", fake_run_openai_print_mode)

    result = CliRunner().invoke(app, ["-p", "Summarize"], input="piped content\n")

    assert result.exit_code == 0
    assert calls == ["piped content\n\n\nSummarize"]


def test_print_mode_accepts_stdin_only_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that piped stdin alone becomes the prompt.
    calls: list[str] = []

    async def fake_run_openai_print_mode(
        prompt: str,
        model: str | None,
        cwd: Path,
        output: PrintOutputMode,
        provider_name: str | None,
        *extra: object,
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: Record the prompt and return success.
        del model, cwd, output, provider_name, extra
        calls.append(prompt)
        return True

    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_print_mode", fake_run_openai_print_mode)

    result = CliRunner().invoke(app, ["-p"], input="piped content\n")

    assert result.exit_code == 0
    assert calls == ["piped content\n"]


def test_export_flag_invokes_exporter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC]
    # Test that "--export <session> <path>" calls the exporter and reports the path.
    calls: list[tuple[str, Path | None, str | None]] = []
    output_path = tmp_path / "out.html"

    async def fake_export_session_command(
        session_ref: str,
        requested_output_path: Path | None = None,
        requested_export_format: str | None = None,
    ) -> Path:
        # [FUNCTION DOC]
        # HELPER: Record (session ref, output path, format) and return a fixed path.
        calls.append((session_ref, requested_output_path, requested_export_format))
        return output_path

    monkeypatch.setattr(cli, "export_session_command", fake_export_session_command)

    result = CliRunner().invoke(app, ["--export", "session-1", str(output_path)])

    assert result.exit_code == 0
    assert calls == [("session-1", output_path, None)]
    assert f"Exported session to {output_path}" in result.stdout


def test_export_flag_rejects_combination_with_print() -> None:
    # [FUNCTION DOC]
    # Test that --export cannot be combined with print mode.
    result = CliRunner().invoke(app, ["--export", "-p", "session-1"])

    assert result.exit_code != 0
    assert "--export cannot be combined with --print/--mode" in _strip_ansi(result.output)


def test_version_short_flag_prints_version() -> None:
    # [FUNCTION DOC]
    # Test that -v prints the version.
    result = CliRunner().invoke(app, ["-v"])

    assert result.exit_code == 0
    assert result.stdout.startswith("tau ")


def _constrained_provider_settings() -> ProviderSettings:
    # [FUNCTION DOC]
    # HELPER: Provider settings with a single "local" provider declaring only the
    # model "qwen".
    """Settings with a single provider that only declares ``qwen``."""
    return ProviderSettings(
        default_provider="local",
        providers=(
            OpenAICompatibleProviderConfig(
                name="local",
                base_url="http://localhost:11434/v1",
                api_key_env="LOCAL_API_KEY",
                models=("qwen",),
                default_model="qwen",
            ),
        ),
    )


def test_panel_text_strips_ansi_and_borders() -> None:
    # [FUNCTION DOC]
    # Test that _panel_text reconstructs a message split by ANSI-colored panel
    # borders.
    """``_panel_text`` must strip ANSI escapes *and* panel borders before matching.

    On CI (no real TTY) Rich/Click emit ANSI color codes around the wrapped panel
    border, so ``Available`` and ``models: qwen`` get split by escape sequences.
    This guards the helper used by the bad-model regression tests regardless of
    the local CliRunner's rendering mode. See issue #265.
    """
    ci_style = (
        "\x1b[33mUsage: \x1b[0mtau [OPTIONS] ...\n"
        "\x1b[31m╭─\x1b[0m\x1b[31m Error \x1b[0m\x1b[31m─╮\x1b[0m\n"
        "\x1b[31m│\x1b[0m Invalid value: Model is not configured for provider local: "
        "llama. Available \x1b[31m│\x1b[0m\n"
        "\x1b[31m│\x1b[0m models: qwen \x1b[31m│\x1b[0m\n"
        "\x1b[31m╰╯\x1b[0m"
    )
    out = _panel_text(ci_style)
    assert "Model is not configured for provider local: llama" in out
    assert "Available models: qwen" in out
    assert "\x1b" not in out


def test_tui_surfaces_bad_model_as_clean_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Regression test that an unconfigured model in TUI mode exits 2 with a clean
    # message listing available models instead of a traceback.
    """Regression: ``tau --model <bad>`` must exit with a clean error, not a traceback.

    See https://github.com/huggingface/tau/issues/265. The TUI startup path
    previously only caught ``RuntimeError``, so a ``ProviderConfigError`` (a
    ``ValueError`` subclass) raised while resolving the provider/model selection
    escaped the ``anyio`` event loop as an unhandled traceback.
    """
    import tau_coding.tui.app as tui_app

    settings = _constrained_provider_settings()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "load_provider_settings", lambda *args, **kwargs: settings)
    monkeypatch.setattr(tui_app, "load_provider_settings", lambda *args, **kwargs: settings)

    result = CliRunner().invoke(app, ["--model", "llama", "--provider", "local"])

    # A clean BadParameter exits 2 (Typer's convention) and includes the
    # actionable message listing valid models for the provider.
    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)
    out = _panel_text(result.output)
    assert "Model is not configured for provider local: llama" in out
    assert "Available models: qwen" in out


def test_print_mode_surfaces_bad_model_as_clean_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Same regression for print mode.
    """The print-mode ``--model <bad>`` path must also surface a clean error.

    Companion regression to the TUI path (issue #265): the print-mode handler
    likewise only caught ``RuntimeError``, so it also dumped a
    ``ProviderConfigError`` traceback instead of a friendly message.
    """
    import tau_coding.tui.app as tui_app

    settings = _constrained_provider_settings()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "load_provider_settings", lambda *args, **kwargs: settings)
    monkeypatch.setattr(tui_app, "load_provider_settings", lambda *args, **kwargs: settings)

    result = CliRunner().invoke(app, ["--model", "llama", "--provider", "local", "-p", "hello"])

    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)
    out = _panel_text(result.output)
    assert "Model is not configured for provider local: llama" in out
    assert "Available models: qwen" in out


def test_sessions_command_lists_indexed_sessions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # [FUNCTION DOC]
    # Test that "sessions" lists session IDs and titles from the manager.
    record = CodingSessionRecord(
        id="session-1",
        path=tmp_path / "session.jsonl",
        cwd=tmp_path,
        model="fake",
        title="Test session",
        created_at=1.0,
        updated_at=2.0,
    )

    class FakeSessionManager:
        # [CLASS DOC]
        # HELPER: Fake session manager returning one record.
        def list_sessions(self) -> list[CodingSessionRecord]:
            # [FUNCTION DOC]
            # HELPER: Return the single test record.
            return [record]

    monkeypatch.setattr(cli, "SessionManager", FakeSessionManager)

    result = CliRunner().invoke(app, ["sessions"])

    assert result.exit_code == 0
    assert "session-1" in result.stdout
    assert "Test session" in result.stdout


def test_sessions_command_handles_empty_index(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that "sessions" prints "No sessions found." for an empty index.
    class FakeSessionManager:
        # [CLASS DOC]
        # HELPER: Fake session manager with no sessions.
        def list_sessions(self) -> list[CodingSessionRecord]:
            # [FUNCTION DOC]
            # HELPER: Return an empty list.
            return []

    monkeypatch.setattr(cli, "SessionManager", FakeSessionManager)

    result = CliRunner().invoke(app, ["sessions"])

    assert result.exit_code == 0
    assert "No sessions found." in result.stdout


@pytest.mark.anyio
async def test_export_session_command_writes_html_for_indexed_session(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test HTML export of an indexed session: title, message text and session path
    # appear and no system prompt section is included.
    manager = SessionManager(TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents"))
    record = manager.create_session(
        cwd=tmp_path,
        model="fake",
        title="Exported Session",
        session_id="session-1",
    )
    await JsonlSessionStorage(record.path).append(
        MessageEntry(id="root", message=UserMessage(content="Export this"))
    )

    output_path = await cli.export_session_command(
        "session-1",
        tmp_path / "session.html",
        session_manager=manager,
    )

    html = output_path.read_text(encoding="utf-8")
    assert output_path == tmp_path / "session.html"
    assert "<title>Exported Session</title>" in html
    assert "Export this" in html
    assert str(record.path) in html
    assert '<details class="system-prompt">' not in html


@pytest.mark.anyio
async def test_export_session_command_writes_html_for_jsonl_path(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test HTML export from a JSONL path into the current directory with a title
    # derived from the file stem.
    # KNOWN ISSUE: changes the process working directory with a manual chdir and
    # restore rather than monkeypatch, which is not isolation-safe if the test is
    # interrupted.
    session_path = tmp_path / "session.jsonl"
    cwd = Path.cwd()
    await JsonlSessionStorage(session_path).append(
        MessageEntry(id="root", message=UserMessage(content="Path export"))
    )

    try:
        import os

        os.chdir(tmp_path)
        output_path = await cli.export_session_command(str(session_path))
    finally:
        os.chdir(cwd)

    html = output_path.read_text(encoding="utf-8")
    assert output_path == tmp_path / "session.html"
    assert "<title>Tau session session</title>" in html
    assert "Path export" in html
    assert '<details class="system-prompt">' not in html


@pytest.mark.anyio
async def test_export_session_command_writes_jsonl_format_to_cwd(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test JSONL-format export writes "<stem>.jsonl" into the current directory.
    # KNOWN ISSUE: uses a manual chdir/restore like the previous test.
    session_path = tmp_path / ".tau" / "sessions" / "session.jsonl"
    cwd = Path.cwd()
    await JsonlSessionStorage(session_path).append(
        MessageEntry(id="root", message=UserMessage(content="JSONL export"))
    )

    try:
        import os

        os.chdir(tmp_path)
        output_path = await cli.export_session_command(str(session_path), export_format="jsonl")
    finally:
        os.chdir(cwd)

    assert output_path == tmp_path / "session.jsonl"
    assert "JSONL export" in output_path.read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_export_session_command_treats_suffixless_output_as_directory(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC]
    # Test that an output path without suffix is treated as a directory.
    session_path = tmp_path / "source" / "session.jsonl"
    await JsonlSessionStorage(session_path).append(
        MessageEntry(id="root", message=UserMessage(content="Directory export"))
    )

    output_path = await cli.export_session_command(str(session_path), tmp_path / "exports")

    assert output_path == tmp_path / "exports" / "session.html"
    assert "Directory export" in output_path.read_text(encoding="utf-8")


def test_export_command_invokes_exporter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC]
    # Test the "export" subcommand forwards session and output path.
    calls: list[tuple[str, Path | None, str | None]] = []
    output_path = tmp_path / "out.html"

    async def fake_export_session_command(
        session_ref: str,
        requested_output_path: Path | None = None,
        requested_export_format: str | None = None,
    ) -> Path:
        # [FUNCTION DOC]
        # HELPER: Record (session ref, output path, format) and return a fixed path.
        calls.append((session_ref, requested_output_path, requested_export_format))
        return output_path

    monkeypatch.setattr(cli, "export_session_command", fake_export_session_command)

    result = CliRunner().invoke(app, ["export", "session-1", str(output_path)])

    assert result.exit_code == 0
    assert calls == [("session-1", output_path, None)]
    assert f"Exported session to {output_path}" in result.stdout


def test_export_command_accepts_format_option(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC]
    # Test that "export --format jsonl" forwards the format.
    calls: list[tuple[str, Path | None, str | None]] = []
    output_path = tmp_path / "out.jsonl"

    async def fake_export_session_command(
        session_ref: str,
        requested_output_path: Path | None = None,
        requested_export_format: str | None = None,
    ) -> Path:
        # [FUNCTION DOC]
        # HELPER: Record (session ref, output path, format) and return a fixed path.
        calls.append((session_ref, requested_output_path, requested_export_format))
        return output_path

    monkeypatch.setattr(cli, "export_session_command", fake_export_session_command)

    result = CliRunner().invoke(app, ["export", "session-1", "--format", "jsonl"])

    assert result.exit_code == 0
    assert calls == [("session-1", None, "jsonl")]


def test_providers_command_lists_default_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC]
    # Test that "providers" lists built-in providers with API type and default
    # model, marking the default with an asterisk.
    isolate_home(monkeypatch, tmp_path)

    result = CliRunner().invoke(app, ["providers"])

    assert result.exit_code == 0
    assert "*\topenai\topenai-compatible\tgpt-5.4" in result.stdout
    assert " \topenai-codex\topenai-codex\tgpt-5.5" in result.stdout
    assert " \tanthropic\tanthropic\tclaude-sonnet-4-6" in result.stdout
    assert " \topenrouter\topenai-compatible\tqwen/qwen3.7-max" in result.stdout
    assert " \thuggingface\topenai-compatible\tmoonshotai/Kimi-K2.6" in result.stdout


def test_render_provider_settings_shows_credential_source(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # [FUNCTION DOC]
    # Test that provider rendering shows credential source: stored credential,
    # environment variable, or missing.
    monkeypatch.delenv("STORED_API_KEY", raising=False)
    monkeypatch.setenv("ENV_API_KEY", "env-key")
    monkeypatch.delenv("MISSING_API_KEY", raising=False)
    settings = ProviderSettings(
        default_provider="stored",
        providers=(
            OpenAICompatibleProviderConfig(
                name="stored",
                api_key_env="STORED_API_KEY",
                credential_name="stored",
            ),
            OpenAICompatibleProviderConfig(
                name="env",
                api_key_env="ENV_API_KEY",
                credential_name=None,
            ),
            OpenAICompatibleProviderConfig(
                name="missing",
                api_key_env="MISSING_API_KEY",
                credential_name="missing",
            ),
        ),
    )

    class FakeCredentials:
        # [CLASS DOC]
        # HELPER: Fake credential store with a key only for "stored".
        def get(self, name: str) -> str | None:
            # [FUNCTION DOC]
            # HELPER: Return "stored-key" for "stored", else None.
            return "stored-key" if name == "stored" else None

    cli.render_provider_settings(settings, credential_reader=FakeCredentials())

    output = capsys.readouterr().out
    assert "*\tstored\topenai-compatible\tgpt-5.4" in output
    assert "\tSTORED_API_KEY\tstored:stored\t" in output
    assert "\tENV_API_KEY\tenv:ENV_API_KEY\t" in output
    assert "\tMISSING_API_KEY\tmissing\t" in output


def test_setup_command_writes_provider_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC]
    # Test that "setup" saves a provider (trailing slash trimmed from base URL,
    # timeouts, retries, default model) and makes it the default.
    isolate_home(monkeypatch, tmp_path)
    monkeypatch.setenv("LOCAL_API_KEY", "test-key")

    result = CliRunner().invoke(
        app,
        [
            "--provider",
            "local",
            "--base-url",
            "http://localhost:11434/v1/",
            "--api-key-env",
            "LOCAL_API_KEY",
            "--timeout-seconds",
            "120",
            "--max-retries",
            "2",
            "--max-retry-delay-seconds",
            "0.5",
            "--model",
            "qwen",
            "setup",
        ],
    )

    settings = load_provider_settings(TauPaths(home=tmp_path / ".tau"))
    provider = settings.get_provider("local")
    assert result.exit_code == 0
    assert "Saved provider 'local'" in result.stdout
    assert settings.default_provider == "local"
    assert provider.base_url == "http://localhost:11434/v1"
    assert provider.api_key_env == "LOCAL_API_KEY"
    assert provider.default_model == "qwen"
    assert provider.timeout_seconds == 120
    assert provider.max_retries == 2
    assert provider.max_retry_delay_seconds == 0.5


def test_relative_tau_home_reports_actionable_cli_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC]
    # Test that a relative TAU_HOME exits 2 with a clear error and no traceback.
    monkeypatch.setenv("TAU_HOME", ".tau-personal")

    result = CliRunner().invoke(app, ["providers"])

    assert result.exit_code == 2
    assert "Invalid value for TAU_HOME" in result.stderr
    assert "must be an absolute path" in result.stderr
    assert "Traceback" not in result.stderr


def test_unknown_user_tau_home_reports_actionable_cli_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC]
    # Test that an unexpandable "~user" TAU_HOME exits 2 with a clear error.
    monkeypatch.setenv("TAU_HOME", "~tau-review-user-that-does-not-exist-728/.tau")

    result = CliRunner().invoke(app, ["providers"])

    assert result.exit_code == 2
    assert "Invalid value for TAU_HOME" in result.stderr
    assert "Traceback" not in result.stderr


def test_setup_command_writes_only_to_configured_tau_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC]
    # Test that setup writes catalog and provider files only under TAU_HOME.
    tau_home = tmp_path / ".tau-personal"
    monkeypatch.setenv("TAU_HOME", str(tau_home))
    monkeypatch.setenv("LOCAL_API_KEY", "test-key")

    result = CliRunner().invoke(
        app,
        [
            "--provider",
            "local",
            "--base-url",
            "http://localhost:11434/v1",
            "--api-key-env",
            "LOCAL_API_KEY",
            "--model",
            "qwen",
            "setup",
        ],
    )

    assert result.exit_code == 0
    assert (tau_home / "catalog.toml").exists()
    assert (tau_home / "providers.json").exists()
    assert not (tmp_path / ".tau").exists()
    assert load_provider_settings(TauPaths(home=tau_home)).default_provider == "local"


def test_setup_command_warns_when_api_key_env_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC]
    # Test that setup warns on stderr when the API key variable is unset.
    isolate_home(monkeypatch, tmp_path)
    monkeypatch.delenv("MISSING_API_KEY", raising=False)

    result = CliRunner().invoke(
        app,
        [
            "--provider",
            "missing",
            "--api-key-env",
            "MISSING_API_KEY",
            "--model",
            "test-model",
            "setup",
        ],
    )

    assert result.exit_code == 0
    assert "Set MISSING_API_KEY before running Tau with this provider." in result.stderr


@pytest.mark.parametrize("output", [PrintOutputMode.json, PrintOutputMode.transcript])
@pytest.mark.anyio
async def test_headless_ask_declines_without_corrupting_structured_stdout(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, output: PrintOutputMode
) -> None:
    # [FUNCTION DOC]
    # Parametrized test that in JSON/transcript modes an untrusted project's
    # instructions are excluded and the trust prompt goes to stderr, not stdout.
    (tmp_path / "AGENTS.md").write_text("PROTECTED-STRUCTURED-SECRET", encoding="utf-8")
    provider = FakeProvider(
        [[assistant_start(model="fake"), assistant_done(message=AssistantMessage(content="Done"))]]
    )

    ok = await run_print_mode(
        prompt="Hello",
        model="fake",
        cwd=tmp_path,
        provider=provider,
        output=output,
        resource_paths=TauResourcePaths(root=tmp_path / "home/.tau", agents_root=None),
    )

    captured = capsys.readouterr()
    assert ok is True
    assert "PROTECTED-STRUCTURED-SECRET" not in provider.calls[0][1]
    assert "Project inputs" not in captured.out
    assert "Project inputs" in captured.err


def test_thinking_flag_forwards_to_tui_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that --thinking is forwarded to the TUI runner as a keyword.
    calls: list[object] = []

    async def fake_run_openai_tui(*args: object, **kwargs: object) -> None:
        # [FUNCTION DOC]
        # HELPER: Record the thinking level override keyword.
        calls.append(kwargs.get("thinking_level_override"))

    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_tui", fake_run_openai_tui)

    result = CliRunner().invoke(app, ["--thinking", "high", "--new-session"])

    assert result.exit_code == 0
    assert calls == ["high"]


def test_thinking_flag_forwards_to_print_mode_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that -t is forwarded to the print-mode runner.
    calls: list[object] = []

    async def fake_run_openai_print_mode(*args: object, **kwargs: object) -> bool:
        # [FUNCTION DOC]
        # HELPER: Record the thinking level override and return success.
        calls.append(kwargs.get("thinking_level_override"))
        return True

    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_print_mode", fake_run_openai_print_mode)

    result = CliRunner().invoke(app, ["--print", "-t", "low", "hello"])

    assert result.exit_code == 0
    assert calls == ["low"]


def test_thinking_flag_forwards_to_rpc_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that --thinking is forwarded to the RPC runner.
    calls: list[object] = []

    async def fake_run_openai_rpc_mode(*args: object, **kwargs: object) -> None:
        # [FUNCTION DOC]
        # HELPER: Record the thinking level override.
        calls.append(kwargs.get("thinking_level_override"))

    monkeypatch.setattr(cli, "run_openai_rpc_mode", fake_run_openai_rpc_mode)

    result = CliRunner().invoke(app, ["--mode", "rpc", "--thinking", "xhigh"])

    assert result.exit_code == 0
    assert calls == ["xhigh"]


def test_thinking_flag_rejects_invalid_level(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that an unknown thinking level exits 2 before starting the TUI.
    async def fake_run_openai_tui(*args: object, **kwargs: object) -> None:
        # [FUNCTION DOC]
        # HELPER: Fail the test if the TUI is started.
        raise AssertionError("TUI must not start for an invalid thinking level")

    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_tui", fake_run_openai_tui)

    result = CliRunner().invoke(app, ["--thinking", "maximum", "--new-session"])

    assert result.exit_code == 2
    assert "Unknown thinking mode: maximum" in _panel_text(result.output)


def test_thinking_flag_accepts_case_insensitive_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [FUNCTION DOC]
    # Test that thinking levels are lowercased.
    calls: list[object] = []

    async def fake_run_openai_tui(*args: object, **kwargs: object) -> None:
        # [FUNCTION DOC]
        # HELPER: Record the thinking level override.
        calls.append(kwargs.get("thinking_level_override"))

    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_tui", fake_run_openai_tui)

    result = CliRunner().invoke(app, ["-t", "HIGH", "--new-session"])

    assert result.exit_code == 0
    assert calls == ["high"]


def test_thinking_flag_absent_forwards_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # Test that omitting the flag forwards None.
    calls: list[object] = []

    async def fake_run_openai_tui(*args: object, **kwargs: object) -> None:
        # [FUNCTION DOC]
        # HELPER: Record the thinking level override.
        calls.append(kwargs.get("thinking_level_override"))

    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_tui", fake_run_openai_tui)

    result = CliRunner().invoke(app, ["--new-session"])

    assert result.exit_code == 0
    assert calls == [None]


def test_help_lists_thinking_option() -> None:
    # [FUNCTION DOC]
    # Test that help lists the --thinking option.
    result = CliRunner().invoke(app, ["--help"], env={"COLUMNS": "160"})

    output = re.sub(r"\s+", "", _strip_ansi(result.output))
    assert result.exit_code == 0
    assert "--thinking" in output
