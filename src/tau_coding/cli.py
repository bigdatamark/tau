# [MODULE DOC]
# Command-line entry point for Tau (the "tau" Typer app), the outermost tau_coding layer.
# On import it forces stdout/stderr to UTF-8 (for Windows consoles) and defines the
# Typer app. The single callback "main" parses all options and dispatches to:
# the interactive Textual TUI (default), non-interactive print mode (--print or --mode
# text/json/transcript), Pi-compatible JSONL RPC mode (--mode rpc), or the built-in
# subcommands passed positionally: update [--models], install, sessions, export,
# providers, setup. Also contains helpers for prompt/system-prompt file resolution,
# stdin merging, session export, provider listing, and the print-mode runner which
# stages a CodingSession, handles slash and terminal commands, streams events to a
# renderer and reports success. Maps to Pi's CLI main/print-mode/RPC-mode entry points.
"""Command-line entry point for Tau."""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Sequence
from functools import partial
from os import environ
from pathlib import Path
from typing import Annotated, Literal

import anyio
import typer

from tau_agent.provider import ModelProvider
from tau_agent.session import JsonlSessionStorage, SessionEntry, SessionStorage
from tau_ai.env import (
    DEFAULT_OPENAI_COMPATIBLE_BASE_URL,
    DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES,
    DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS,
    DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
)
from tau_coding.catalog_loader import user_catalog_path
from tau_coding.commands import format_reload_summary
from tau_coding.credentials import FileCredentialStore
from tau_coding.extension_installer import ExtensionInstallError, install_extension
from tau_coding.extensions import StderrUiBridge
from tau_coding.models_dev_store import (
    ModelsDevRefreshError,
    ModelsDevRefreshResult,
    refresh_models_dev_catalog,
)
from tau_coding.paths import TauPaths
from tau_coding.project_trust import TrustDefault, TrustOverride
from tau_coding.provider_config import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER_NAME,
    CredentialReader,
    OpenAICompatibleProviderConfig,
    ProviderConfig,
    ProviderConfigError,
    ProviderSettings,
    load_provider_settings,
    provider_kind,
    resolve_provider_selection,
    resolve_startup_thinking_level,
    save_provider_settings,
    upsert_openai_compatible_provider,
)
from tau_coding.provider_runtime import ClosableModelProvider, create_model_provider
from tau_coding.rendering import PrintOutputMode, create_event_renderer
from tau_coding.resources import TauResourcePaths
from tau_coding.rpc import RpcServer
from tau_coding.session import (
    CodingSession,
    CodingSessionConfig,
    TerminalCommandResult,
    jsonl_session_storage,
    parse_terminal_command,
)
from tau_coding.session_export import (
    default_session_export_artifact_path,
    export_session_artifact,
    normalize_export_format,
)
from tau_coding.session_manager import CodingSessionRecord, SessionManager, validate_session_id
from tau_coding.session_preparation import prepare_coding_session
from tau_coding.shell_config import load_shell_settings
from tau_coding.thinking import THINKING_LEVELS, ThinkingLevel, normalize_thinking_level
from tau_coding.tui import run_tui_app
from tau_coding.update_check import (
    UpdateNotice,
    startup_release_notes_notice,
    startup_update_notice,
)
from tau_coding.updater import update_tau
from tau_coding.version import current_version as _current_version


def _is_utf8_encoding(encoding: str | None) -> bool:
    # [FUNCTION DOC]
    # HELPER: Return True if an encoding name is UTF-8 ignoring case, hyphens and
    # underscores; False for None or other names.
    """Return whether a stream encoding name represents UTF-8."""
    if encoding is None:
        return False
    return encoding.lower().replace("-", "").replace("_", "") == "utf8"


def _force_utf8_streams() -> None:
    # [FUNCTION DOC]
    # HELPER: For stdout and stderr, if the stream's encoding is not UTF-8, reconfigure it
    # to UTF-8 with replacement for unencodable characters. Streams without reconfigure or
    # that reject it (AttributeError/ValueError) are left unchanged. Called once at import.
    # Side effects: mutates the global standard streams.
    """Reconfigure stdout/stderr to UTF-8 when they are not already UTF-8.

    Windows consoles default these streams to the system codepage (e.g.
    cp1252), which raises UnicodeEncodeError on model output containing
    characters outside that codepage.
    """
    for stream in (sys.stdout, sys.stderr):
        if _is_utf8_encoding(getattr(stream, "encoding", None)):
            continue
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


_force_utf8_streams()

app = typer.Typer(
    name="tau",
    help="Tau coding-agent harness.",
    epilog="""Commands:

  tau install SOURCE [--force] - Install a trusted local or Git extension.

  tau update - Upgrade Tau.

  tau sessions - List indexed sessions.

  tau export REF [DEST] - Export a session as HTML or JSONL.

  tau providers - List configured model providers.

  tau setup - Configure an OpenAI-compatible provider.
""",
    add_completion=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)


def providers_command() -> None:
    # [FUNCTION DOC]
    # Implement "tau providers": load provider settings and print one tab-separated line per
    # provider, checking credential status against the file credential store.
    """List configured model providers."""
    render_provider_settings(load_provider_settings(), credential_reader=FileCredentialStore())


def install_command(args: list[str]) -> None:
    # [FUNCTION DOC]
    # Implement "tau install SOURCE [--force]".
    # Input: remaining positional arguments. Accepts exactly one source plus an optional
    # --force flag; unknown options, extra or missing sources raise typer.BadParameter.
    # Prints a warning to stderr that extensions run arbitrary code, then installs into the
    # user extension directory. On ExtensionInstallError prints the error and exits with
    # code 1; on success prints the source and destination. Side effects: filesystem/git.
    """Install an extension into Tau's user extension directory."""
    source: str | None = None
    force = False
    for arg in args:
        if arg == "--force":
            force = True
        elif arg.startswith("-"):
            raise typer.BadParameter(f"Unknown option for `tau install`: {arg}")
        elif source is None:
            source = arg
        else:
            raise typer.BadParameter("Usage: tau install <source> [--force]")
    if source is None:
        raise typer.BadParameter("Usage: tau install <source> [--force]")

    typer.echo(
        "Warning: extensions execute arbitrary Python with your user permissions. "
        "Only install sources you trust.",
        err=True,
    )
    try:
        destination = install_extension(source, force=force)
    except ExtensionInstallError as exc:
        typer.echo(f"Could not install extension: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Installed {source} to {destination}")


def setup_command(
    *,
    provider_name: str = DEFAULT_PROVIDER_NAME,
    base_url: str = DEFAULT_OPENAI_COMPATIBLE_BASE_URL,
    api_key_env: str = "OPENAI_API_KEY",
    model: str = DEFAULT_MODEL,
    timeout_seconds: float = DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES,
    max_retry_delay_seconds: float = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS,
    set_default: bool = True,
) -> None:
    # [FUNCTION DOC]
    # Implement "tau setup": create or update an OpenAI-compatible provider entry.
    # Keyword inputs: provider name, base URL (trailing slashes stripped), API-key env var,
    # model (used as sole model and default), timeout, max retries, max retry delay and
    # whether to make it the default. Loads settings, upserts the provider, saves settings,
    # and prints where the catalog and preferences were saved. Warns on stderr if the API
    # key env var is not set. Side effects: writes settings files.
    """Create or update an OpenAI-compatible provider entry."""
    settings = load_provider_settings()
    provider = OpenAICompatibleProviderConfig(
        name=provider_name,
        base_url=base_url.rstrip("/"),
        api_key_env=api_key_env,
        models=(model,),
        default_model=model,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_retry_delay_seconds=max_retry_delay_seconds,
    )
    updated = upsert_openai_compatible_provider(settings, provider, set_default=set_default)
    path = save_provider_settings(updated)
    typer.echo(
        f"Saved provider '{provider.name}' to {user_catalog_path()} and preferences to {path}"
    )
    if provider.api_key_env not in environ:
        typer.echo(f"Set {provider.api_key_env} before running Tau with this provider.", err=True)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prompt_args: Annotated[
        list[str] | None,
        typer.Argument(help="Initial prompt to run in interactive TUI mode."),
    ] = None,
    print_mode: Annotated[
        bool,
        typer.Option(
            "--print",
            "-p",
            help="Run the positional prompt in non-interactive print mode.",
        ),
    ] = False,
    prompt_option: Annotated[
        str | None,
        typer.Option(
            "--prompt",
            help="Removed; pass the prompt positionally and use --print instead.",
            hidden=True,
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Configured provider name to use."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model name to request from the provider."),
    ] = None,
    thinking: Annotated[
        str | None,
        typer.Option(
            "--thinking",
            "-t",
            help=(
                "Initial thinking level for this run "
                f"({', '.join(THINKING_LEVELS)}). "
                "Overrides remembered defaults without persisting."
            ),
        ),
    ] = None,
    setup_base_url: Annotated[
        str,
        typer.Option("--base-url", help="OpenAI-compatible base URL for `tau setup`."),
    ] = DEFAULT_OPENAI_COMPATIBLE_BASE_URL,
    setup_api_key_env: Annotated[
        str,
        typer.Option("--api-key-env", help="API key environment variable for `tau setup`."),
    ] = "OPENAI_API_KEY",
    setup_timeout_seconds: Annotated[
        float,
        typer.Option(
            "--timeout-seconds",
            help="HTTP timeout in seconds for `tau setup` provider requests.",
        ),
    ] = DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS,
    setup_max_retries: Annotated[
        int,
        typer.Option("--max-retries", help="Provider retry count for `tau setup`."),
    ] = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES,
    setup_max_retry_delay_seconds: Annotated[
        float,
        typer.Option(
            "--max-retry-delay-seconds",
            help="Provider retry delay in seconds for `tau setup`.",
        ),
    ] = DEFAULT_OPENAI_COMPATIBLE_MAX_RETRY_DELAY_SECONDS,
    setup_default: Annotated[
        bool,
        typer.Option("--set-default/--no-set-default", help="Make setup provider the default."),
    ] = True,
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Working directory for built-in coding tools."),
    ] = None,
    mode: Annotated[
        PrintOutputMode | None,
        typer.Option(
            "--mode",
            help="Run headlessly with this output format (text, json, transcript, or rpc).",
        ),
    ] = None,
    output: Annotated[
        PrintOutputMode | None,
        typer.Option(
            "--output",
            "-o",
            help="Removed; use --mode instead.",
            hidden=True,
        ),
    ] = None,
    session: Annotated[
        str | None,
        typer.Option("--session", help="Resume a session id in TUI or print mode."),
    ] = None,
    resume: Annotated[
        str | None,
        typer.Option(
            "--resume",
            help="Removed; use --session <session-id> instead.",
            hidden=True,
        ),
    ] = None,
    new_session: Annotated[
        bool,
        typer.Option("--new-session", help="Create a new session in TUI mode (default)."),
    ] = False,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session-id",
            help="Set the exact id for the newly created print-mode session.",
        ),
    ] = None,
    system_prompt: Annotated[
        str | None,
        typer.Option(
            "--system-prompt",
            metavar="TEXT_OR_PATH",
            help="Replace the default system-prompt base with literal text or a UTF-8 file.",
        ),
    ] = None,
    append_system_prompt: Annotated[
        list[str] | None,
        typer.Option(
            "--append-system-prompt",
            metavar="TEXT_OR_PATH",
            help="Append literal text or a UTF-8 file to the system prompt (repeatable).",
        ),
    ] = None,
    auto_compact_threshold: Annotated[
        int | None,
        typer.Option(
            "--auto-compact-threshold",
            help="Automatically compact TUI context above this rough token estimate.",
        ),
    ] = None,
    extension: Annotated[
        list[Path] | None,
        typer.Option(
            "--extension",
            "-e",
            help="Load an extension file or directory (repeatable).",
        ),
    ] = None,
    extension_legacy: Annotated[
        list[Path] | None,
        typer.Option(
            "-x",
            help="Removed; use -e/--extension instead.",
            hidden=True,
        ),
    ] = None,
    export: Annotated[
        bool,
        typer.Option(
            "--export",
            help="Export the given session id or JSONL path (mirrors `tau export`).",
        ),
    ] = False,
    no_extensions: Annotated[
        bool,
        typer.Option(
            "--no-extensions",
            help="Disable extension directory discovery (explicit -e paths still load).",
        ),
    ] = False,
    project_extensions: Annotated[
        bool,
        typer.Option(
            "--project-extensions",
            help="Also load trusted project .tau/extensions (additional code opt-in).",
        ),
    ] = False,
    approve: Annotated[
        bool,
        typer.Option("--approve", "-a", help="Trust protected project inputs for this run."),
    ] = False,
    no_approve: Annotated[
        bool,
        typer.Option("--no-approve", "-na", help="Decline protected project inputs for this run."),
    ] = False,
    version: Annotated[
        bool,
        typer.Option("--version", "-v", help="Show Tau's version and exit."),
    ] = False,
    models: Annotated[
        bool,
        typer.Option("--models", help="With `tau update`, refresh model catalogs only."),
    ] = False,
) -> None:
    # [FUNCTION DOC]
    # Typer callback implementing the whole CLI; invoked even without a subcommand.
    # Order of behavior:
    # 1) --version prints "tau <version>" and exits. Validates TauPaths (bad TAU_HOME ->
    # BadParameter). Returns early if a registered Typer subcommand was invoked.
    # 2) Rejects conflicting/removed options with BadParameter: --approve with --no-approve,
    # --resume (renamed --session), --session with --new-session or --session-id,
    # --prompt (removed), --output (renamed --mode), -x (renamed -e). Trust override is
    # "approve", "decline" or none. Normalises --thinking (invalid -> BadParameter).
    # 3) Mode: rpc if --mode rpc; print if --print or any other --mode; output defaults to
    # text. --session-id requires print mode and must be a valid id.
    # 4) The first positional argument may be a built-in command when not in rpc/print/export
    # mode: "update" (optionally "--models" or the --models flag; models refresh vs package
    # upgrade), "install", "sessions" (lists sessions), "export". --models elsewhere is an
    # error. --export (not with print/mode) exports using the positionals. "providers" and
    # "setup" (alone) run their commands. These paths exit after running.
    # 5) Resolves --system-prompt and repeated --append-system-prompt values (file contents
    # if they name existing files, else literal text).
    # 6) RPC mode: rejects a positional prompt, runs the RPC server; RuntimeError/ValueError
    # become BadParameter.
    # 7) TUI mode (default): computes the startup update notice, runs the TUI with all
    # options and, if it returns a session id, prints how to resume it.
    # 8) Print mode: merges piped stdin into the prompt (empty -> BadParameter usage),
    # prints the update notice to stderr for text output, runs print mode (resuming when
    # --session is given), and exits with code 1 if the run failed.
    # Always ends by raising typer.Exit or returning. Side effects: terminal I/O, network,
    # filesystem (settings, sessions).
    """Run the Tau CLI."""
    current_version = _current_version()
    if version:
        typer.echo(f"tau {current_version}")
        raise typer.Exit()

    try:
        TauPaths()
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="TAU_HOME") from exc

    if ctx.invoked_subcommand is not None:
        return

    if approve and no_approve:
        raise typer.BadParameter("--approve and --no-approve cannot be used together")
    trust_override: TrustOverride | None = (
        "approve" if approve else "decline" if no_approve else None
    )

    if resume is not None:
        raise typer.BadParameter(
            f"--resume was renamed to --session. Use `tau --session {resume}` instead."
        )

    if session is not None and new_session:
        raise typer.BadParameter("--session and --new-session cannot be used together")
    if session is not None and session_id is not None:
        raise typer.BadParameter("--session and --session-id cannot be used together")

    if prompt_option is not None:
        raise typer.BadParameter(
            "--prompt was removed. Pass the prompt positionally and use --print, e.g. "
            f'`tau --print "{prompt_option}"`.'
        )

    if output is not None:
        raise typer.BadParameter(
            f"--output was renamed to --mode. Use `tau --mode {output.value}` instead."
        )

    if extension_legacy is not None:
        raise typer.BadParameter("-x was renamed to -e/--extension.")

    thinking_level_override: ThinkingLevel | None = None
    if thinking is not None:
        try:
            thinking_level_override = normalize_thinking_level(thinking)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    rpc_requested = mode is PrintOutputMode.rpc
    print_requested = print_mode or (mode is not None and not rpc_requested)
    effective_output = mode or PrintOutputMode.text

    if session_id is not None:
        if not print_requested:
            raise typer.BadParameter("--session-id is only supported in print mode")
        try:
            validate_session_id(session_id)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    positional_args = prompt_args or []
    command = positional_args[0] if positional_args else None
    initial_prompt = " ".join(positional_args) if positional_args else None

    if rpc_requested and export:
        raise typer.BadParameter("--export cannot be combined with --mode rpc")

    if not rpc_requested and not print_requested and not export and command == "update":
        positional_models = positional_args[1:] == ["--models"]
        if len(positional_args) != 1 and not positional_models:
            raise typer.BadParameter("Usage: tau update [--models]")
        if models or positional_models:
            update_models_command()
        else:
            update_command()
        raise typer.Exit()

    if models:
        raise typer.BadParameter("--models is only supported with `tau update`")

    if not rpc_requested and not print_requested and not export and command == "install":
        install_command(positional_args[1:])
        raise typer.Exit()

    if (
        not rpc_requested
        and not print_requested
        and not export
        and command == "sessions"
        and len(positional_args) == 1
    ):
        render_session_list(SessionManager().list_sessions())
        raise typer.Exit()

    if not rpc_requested and not print_requested and not export and command == "export":
        _run_export_cli(positional_args[1:])

    if export and not rpc_requested:
        if print_requested:
            raise typer.BadParameter("--export cannot be combined with --print/--mode.")
        _run_export_cli(positional_args)

    if (
        not rpc_requested
        and not print_requested
        and command == "providers"
        and len(positional_args) == 1
    ):
        providers_command()
        raise typer.Exit()

    if (
        not rpc_requested
        and not print_requested
        and command == "setup"
        and len(positional_args) == 1
    ):
        setup_command(
            provider_name=provider or DEFAULT_PROVIDER_NAME,
            base_url=setup_base_url,
            api_key_env=setup_api_key_env,
            model=model or DEFAULT_MODEL,
            timeout_seconds=setup_timeout_seconds,
            max_retries=setup_max_retries,
            max_retry_delay_seconds=setup_max_retry_delay_seconds,
            set_default=setup_default,
        )
        raise typer.Exit()

    extension_paths = tuple(extension or ())
    custom_system_prompt = (
        _resolve_prompt_input(system_prompt, option="--system-prompt")
        if system_prompt is not None
        else None
    )
    resolved_append_system_prompt = _resolve_append_system_prompts(append_system_prompt or ())

    if rpc_requested:
        if initial_prompt is not None:
            raise typer.BadParameter(
                "RPC mode reads commands from stdin and does not accept a prompt"
            )
        try:
            anyio.run(
                partial(
                    run_openai_rpc_mode,
                    thinking_level_override=thinking_level_override,
                )
                if thinking_level_override is not None
                else run_openai_rpc_mode,
                model,
                cwd or Path.cwd(),
                provider,
                session,
                extension_paths,
                not no_extensions,
                project_extensions,
                custom_system_prompt,
                resolved_append_system_prompt,
                trust_override,
            )
        except (RuntimeError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        raise typer.Exit()

    if not print_requested:
        notice = _startup_update_notice()
        try:
            tui_args = (
                model,
                cwd or Path.cwd(),
                session,
                new_session,
                provider,
                auto_compact_threshold,
                initial_prompt,
                notice,
                extension_paths,
                not no_extensions,
                project_extensions,
                custom_system_prompt,
                resolved_append_system_prompt,
            )
            tui_runner = (
                partial(run_openai_tui, thinking_level_override=thinking_level_override)
                if thinking_level_override is not None
                else run_openai_tui
            )
            resumable_session_id = (
                anyio.run(tui_runner, *tui_args)
                if trust_override is None
                else anyio.run(tui_runner, *tui_args, trust_override)
            )
        except (RuntimeError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        if resumable_session_id is not None:
            typer.echo(f"To resume this session: tau --session {resumable_session_id}")
        raise typer.Exit()

    prompt = _merge_stdin_prompt(initial_prompt or "")
    if not prompt:
        raise typer.BadParameter(
            'Usage: tau --print "<prompt>" (or --mode text|json|transcript "<prompt>"); '
            "a prompt can also be piped in via stdin"
        )

    notice = _startup_update_notice()
    if notice is not None and effective_output is PrintOutputMode.text:
        typer.echo(notice.message, err=True)

    try:
        print_args = (
            prompt,
            model,
            cwd or Path.cwd(),
            effective_output,
            provider,
            None,
            extension_paths,
            not no_extensions,
            project_extensions,
            session_id,
            custom_system_prompt,
            resolved_append_system_prompt,
        )
        print_runner = (
            partial(run_openai_print_mode, thinking_level_override=thinking_level_override)
            if thinking_level_override is not None
            else run_openai_print_mode
        )
        if session is not None:
            ok = anyio.run(print_runner, *print_args, trust_override, session)
        else:
            ok = (
                anyio.run(print_runner, *print_args)
                if trust_override is None
                else anyio.run(print_runner, *print_args, trust_override)
            )
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not ok:
        raise typer.Exit(1)


async def run_openai_tui(
    model: str | None,
    cwd: Path,
    session_id: str | None = None,
    new_session: bool = False,
    provider_name: str | None = None,
    auto_compact_token_threshold: int | None = None,
    initial_prompt: str | None = None,
    update_notice: UpdateNotice | None = None,
    extension_paths: tuple[Path, ...] = (),
    extensions_enabled: bool = True,
    project_extensions_enabled: bool = False,
    custom_system_prompt: str | None = None,
    append_system_prompt: str | None = None,
    trust_override: TrustOverride | None = None,
    *,
    thinking_level_override: ThinkingLevel | None = None,
) -> str | None:
    # [FUNCTION DOC]
    # Run the Textual TUI with the given startup options and return its resumable session
    # id (or None). Computes release-notes startup notices for the current version and
    # forwards model, cwd, session selection, provider, auto-compaction threshold, initial
    # prompt, update notice text, extension options, system prompt overrides, trust
    # override and thinking override to the TUI runner.
    """Run the Textual TUI and return its resumable session id, if any."""
    release_notes_notice = startup_release_notes_notice(_current_version())
    startup_notices = (release_notes_notice.message,) if release_notes_notice is not None else ()
    return await run_tui_app(
        model=model,
        cwd=cwd,
        session_id=session_id,
        new_session=new_session,
        provider_name=provider_name,
        auto_compact_token_threshold=auto_compact_token_threshold,
        initial_prompt=initial_prompt,
        startup_update_notice=update_notice.message if update_notice is not None else None,
        startup_notices=startup_notices,
        extension_paths=extension_paths,
        extensions_enabled=extensions_enabled,
        project_extensions_enabled=project_extensions_enabled,
        custom_system_prompt=custom_system_prompt,
        append_system_prompt=append_system_prompt,
        trust_override=trust_override,
        thinking_level_override=thinking_level_override,
    )


def _startup_update_notice() -> UpdateNotice | None:
    # [FUNCTION DOC]
    # HELPER: Return the startup update notice (if a newer release is known) for the
    # current version; may consult a cached update check.
    return startup_update_notice(_current_version())


def update_models_command() -> None:
    # [FUNCTION DOC]
    # Implement "tau update --models": force-refresh the models.dev runtime model catalog
    # and print whether it was refreshed or unchanged, the model count and cache path.
    # On ModelsDevRefreshError prints the error to stderr and exits with code 1.
    # Side effects: network fetch and cache file write.
    """Force-refresh and persist the runtime model catalog."""

    async def refresh() -> ModelsDevRefreshResult:
        # [FUNCTION DOC]
        # HELPER: Async wrapper that force-refreshes the models.dev catalog.
        return await refresh_models_dev_catalog(force=True)

    try:
        result = anyio.run(refresh)
    except ModelsDevRefreshError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    status = "refreshed" if result.refreshed else "unchanged"
    typer.echo(
        f"Model catalogs {status}: {result.model_count} models cached at {result.cache_path}"
    )


def update_command() -> None:
    # [FUNCTION DOC]
    # Implement "tau update": upgrade Tau using the installer managing the environment.
    # On failure prints each failure reason to stderr and exits with code 1. Otherwise
    # echoes captured installer stdout/stderr and reports whether the update completed or
    # was handed off (deferred), including the command used. Side effects: runs an installer.
    """Upgrade Tau using the installer that manages the current environment."""
    result = update_tau()
    if not result.succeeded:
        typer.echo("Could not safely update Tau:", err=True)
        for failure in result.failures:
            typer.echo(f"- {failure}", err=True)
        raise typer.Exit(1)
    if result.stdout:
        typer.echo(result.stdout)
    if result.stderr:
        typer.echo(result.stderr, err=True)
    if result.deferred:
        typer.echo(f"Tau update handed off with: {' '.join(result.command or ())}")
    else:
        typer.echo(f"Tau update completed with: {' '.join(result.command or ())}")


def render_session_list(records: list[CodingSessionRecord]) -> None:
    # [FUNCTION DOC]
    # Print indexed sessions: "No sessions found." if empty, otherwise one tab-separated
    # line per record with id, title (or "Untitled"), model and cwd.
    """Render indexed sessions for the CLI."""
    if not records:
        typer.echo("No sessions found.")
        return

    for record in records:
        title = record.title or "Untitled"
        typer.echo(f"{record.id}\t{title}\t{record.model}\t{record.cwd}")


async def export_session_command(
    session_ref: str,
    output_path: Path | None = None,
    export_format: str | None = None,
    session_manager: SessionManager | None = None,
) -> Path:
    # [FUNCTION DOC]
    # Export a session given an indexed session id or a JSONL file path.
    # Inputs: session reference, optional output path, optional format, optional session
    # manager. Resolves the source path and title, reads all JSONL entries, determines the
    # format (explicit, else the output path's extension, else html; normalised/validated),
    # resolves the destination (see _resolve_export_destination), writes the artifact and
    # returns its path. Raises RuntimeError for unknown sources or directories and
    # ValueError for invalid formats; I/O errors propagate.
    """Export an indexed session id or JSONL file path."""
    session_path, title = _resolve_export_source(session_ref, session_manager)
    entries = await JsonlSessionStorage(session_path).read_all()
    normalized_format = normalize_export_format(
        export_format or (output_path.suffix.removeprefix(".") if output_path else "html")
    )
    destination = _resolve_export_destination(
        output_path,
        session_path=session_path,
        format=normalized_format,
    )
    return export_session_artifact(
        entries,
        destination,
        title=title,
        source=str(session_path),
        format=normalized_format,
    )


def _run_export_cli(args: list[str]) -> None:
    # [FUNCTION DOC]
    # HELPER: Run "tau export" / "tau --export": parse arguments, run the export, print the
    # exported path and exit. Parse errors and RuntimeError/ValueError during export become
    # typer.BadParameter. Always raises typer.Exit on success.
    """Run `tau export`/`tau --export` and exit."""
    try:
        session_ref, output_path, export_format = _parse_export_cli_args(args)
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        exported_path = anyio.run(
            export_session_command,
            session_ref,
            output_path,
            export_format,
        )
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Exported session to {exported_path}")
    raise typer.Exit()


def _resolve_prompt_input(value: str, *, option: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Resolve a prompt option value: if it names an existing path (after home
    # expansion), return the file's UTF-8 contents; otherwise return the literal value.
    # Home expansion failure returns the literal value. Raises typer.BadParameter (with the
    # option as hint) if the path cannot be inspected or read/decoded.
    """Resolve an existing UTF-8 file, otherwise preserve literal prompt text."""
    try:
        path = Path(value).expanduser()
    except RuntimeError:
        return value
    try:
        exists = path.exists()
    except OSError as exc:
        raise typer.BadParameter(
            f"Could not inspect {option} path {path}: {exc}",
            param_hint=option,
        ) from exc
    if not exists:
        return value
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise typer.BadParameter(
            f"Could not read {option} file {path}: {exc}",
            param_hint=option,
        ) from exc


def _resolve_append_system_prompts(values: tuple[str, ...] | list[str]) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Resolve each append value via _resolve_prompt_input, in order, and join them
    # with one blank line. Returns None when no values were given.
    """Resolve repeated append inputs in order and separate them by one blank line."""
    if not values:
        return None
    return "\n\n".join(
        _resolve_prompt_input(value, option="--append-system-prompt") for value in values
    )


def _merge_stdin_prompt(prompt: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Merge piped stdin into the print-mode prompt (Pi behavior). If stdin is
    # missing, is a TTY, or cannot be checked/read, returns the prompt unchanged. Otherwise
    # reads all of stdin; empty input leaves the prompt; with no prompt returns the piped
    # text; else returns piped text, a blank line, then the prompt. Consumes stdin.
    """Merge piped stdin content into a print-mode prompt, mirroring Pi.

    When stdin is not a terminal (e.g. `cat file | tau -p "..."`), its
    contents are prepended to the prompt text.
    """
    stdin = sys.stdin
    if stdin is None:
        return prompt
    try:
        if stdin.isatty():
            return prompt
    except (AttributeError, ValueError):
        return prompt
    try:
        piped = stdin.read()
    except (OSError, ValueError):
        return prompt
    if not piped:
        return prompt
    if not prompt:
        return piped
    return f"{piped}\n\n{prompt}"


def _parse_export_cli_args(args: list[str]) -> tuple[str, Path | None, str | None]:
    # [FUNCTION DOC]
    # HELPER: Parse export arguments: first is the session reference, then optional
    # "--format X" or "--format=X" and at most one output path (home-expanded).
    # Returns (reference, output path or None, format or None). Raises RuntimeError with a
    # usage message for missing reference, missing format value, unknown options or extra
    # positional arguments.
    if not args:
        raise RuntimeError("Usage: tau export <session-id-or-jsonl> [--format html|jsonl] [output]")
    session_ref = args[0]
    output_path: Path | None = None
    export_format: str | None = None
    index = 1
    while index < len(args):
        arg = args[index]
        if arg == "--format":
            index += 1
            if index >= len(args):
                raise RuntimeError(
                    "Usage: tau export <session-id-or-jsonl> [--format html|jsonl] [output]"
                )
            export_format = args[index]
        elif arg.startswith("--format="):
            export_format = arg.partition("=")[2]
        elif arg.startswith("-"):
            raise RuntimeError(f"Unknown export option: {arg}")
        elif output_path is None:
            output_path = Path(arg).expanduser()
        else:
            raise RuntimeError(
                "Usage: tau export <session-id-or-jsonl> [--format html|jsonl] [output]"
            )
        index += 1
    return session_ref, output_path, export_format


def _resolve_export_destination(
    output_path: Path | None,
    *,
    session_path: Path,
    format: str,
) -> Path:
    # [FUNCTION DOC]
    # HELPER: Choose the export file path. No output path -> default artifact name in the
    # current directory; an output path with a file extension -> used as-is; otherwise it is
    # treated as a directory and the default artifact name is placed inside it.
    if output_path is None:
        return default_session_export_artifact_path(
            session_path,
            destination_dir=Path.cwd(),
            format=format,
        )
    if output_path.suffix:
        return output_path
    return default_session_export_artifact_path(
        session_path,
        destination_dir=output_path,
        format=format,
    )


def _resolve_export_source(
    session_ref: str,
    session_manager: SessionManager | None = None,
) -> tuple[Path, str]:
    # [FUNCTION DOC]
    # HELPER: Resolve an export source to (path, title). An existing path is used directly
    # (RuntimeError if it is a directory) with title "Tau session <file stem>". Otherwise
    # looks up the id in the session manager (default instance); unknown -> RuntimeError.
    # Title is the record title or "Tau session <id>".
    candidate_path = Path(session_ref).expanduser()
    if candidate_path.exists():
        if candidate_path.is_dir():
            raise RuntimeError(f"Session export source is a directory: {candidate_path}")
        return candidate_path, f"Tau session {candidate_path.stem}"

    manager = session_manager or SessionManager()
    record = manager.get_session(session_ref)
    if record is None:
        raise RuntimeError(f"Unknown session or file: {session_ref}")

    title = record.title or f"Tau session {record.id}"
    return record.path, title


def render_provider_settings(
    settings: ProviderSettings,
    *,
    credential_reader: CredentialReader | None = None,
) -> None:
    # [FUNCTION DOC]
    # Print configured providers, one tab-separated line each: "*" marker for the default,
    # name, kind, default model, comma-joined models, API-key env var, credential status,
    # base URL, timeout in seconds, retries and retry delay.
    """Render configured providers for the CLI."""
    for provider in settings.providers:
        marker = "*" if provider.name == settings.default_provider else " "
        models = ",".join(provider.models)
        typer.echo(
            f"{marker}\t{provider.name}\t{provider_kind(provider)}\t"
            f"{provider.default_model}\t{models}\t{provider.api_key_env}\t"
            f"{_provider_credential_status(provider, credential_reader=credential_reader)}\t"
            f"{provider.base_url}\t{provider.timeout_seconds:g}s\t"
            f"retries={provider.max_retries}\t"
            f"retry_delay={provider.max_retry_delay_seconds:g}s"
        )


def _provider_credential_status(
    provider: ProviderConfig,
    *,
    credential_reader: CredentialReader | None,
) -> str:
    # [FUNCTION DOC]
    # HELPER: Describe where a provider's credential comes from: "stored:<name>" when a
    # credential name is configured and the reader has it (OAuth lookup for openai-codex,
    # plain secret otherwise), else "env:<VAR>" when the API-key env var is set and
    # non-empty, else "missing".
    if provider.credential_name and credential_reader is not None:
        if provider_kind(provider) == "openai-codex":
            get_oauth = getattr(credential_reader, "get_oauth", None)
            if get_oauth is not None and get_oauth(provider.credential_name) is not None:
                return f"stored:{provider.credential_name}"
        elif credential_reader.get(provider.credential_name):
            return f"stored:{provider.credential_name}"
    if environ.get(provider.api_key_env):
        return f"env:{provider.api_key_env}"
    return "missing"


async def run_openai_rpc_mode(
    model: str | None,
    cwd: Path,
    provider_name: str | None = None,
    resume_session_id: str | None = None,
    extension_paths: tuple[Path, ...] = (),
    extensions_enabled: bool = True,
    project_extensions_enabled: bool = False,
    custom_system_prompt: str | None = None,
    append_system_prompt: str | None = None,
    trust_override: TrustOverride | None = None,
    *,
    thinking_level_override: ThinkingLevel | None = None,
) -> None:
    # [FUNCTION DOC]
    # Run a persistent Pi-compatible JSONL RPC session over stdin/stdout.
    # Loads provider and shell settings and resolves the session record (resumed by id, or
    # newly created exclusively). Provider/model come from explicit options, else from the
    # resumed record. Resolves the provider selection; if that fails and the provider is
    # "openai-codex", bootstraps with the codex default (the staged loader later validates
    # live-only models), else re-raises ProviderConfigError. Determines the Hugging Face
    # inference-provider pin and mode (reused from a resumed record when provider and model
    # match, else from settings as "fixed", or "automatic"). Creates the runtime provider
    # with the startup thinking level, loads the CodingSession with all options, attaches a
    # stderr UI bridge for extensions and runs the RPC server until it finishes, always
    # closing the provider afterwards.
    # KNOWN ISSUE: the loaded CodingSession itself is never closed (only the provider is),
    # unlike print mode which closes the session in a finally block.
    """Run a persistent Pi-compatible JSONL RPC session."""
    settings = load_provider_settings()
    shell_settings = load_shell_settings()
    manager = SessionManager()
    record = _print_session_record(
        manager,
        resume_session_id=resume_session_id,
        cwd=cwd,
        settings=settings,
        provider_name=provider_name,
        model=model,
        session_id=None,
    )
    explicit_selection = provider_name is not None or model is not None
    selected_provider = provider_name if explicit_selection else record.provider_name
    selected_model = model if explicit_selection else record.model
    try:
        selection = resolve_provider_selection(
            settings, provider_name=selected_provider, model=selected_model
        )
    except ProviderConfigError:
        if (selected_provider or settings.default_provider) != "openai-codex":
            raise
        # Bootstrap with the static default; the staged loader discovers and
        # validates the requested/resumed live-only model before activating it.
        selection = resolve_provider_selection(settings, provider_name="openai-codex")
    inference_provider = (
        record.inference_provider
        if resume_session_id is not None
        and record.provider_name == "huggingface"
        and selection.provider.name == "huggingface"
        and record.model == selection.model
        else selection.provider.inference_providers.get(selection.model)
        if isinstance(selection.provider, OpenAICompatibleProviderConfig)
        and selection.provider.name == "huggingface"
        else None
    )
    inference_provider_mode = (
        record.inference_provider_mode
        if resume_session_id is not None
        and record.provider_name == "huggingface"
        and selection.provider.name == "huggingface"
        and record.model == selection.model
        else "fixed"
        if inference_provider is not None
        else "automatic"
    )
    provider = create_model_provider(
        selection.provider,
        model=selection.model,
        inference_provider=inference_provider,
        thinking_level=resolve_startup_thinking_level(
            selection.provider,
            selection.model,
            cli_override=thinking_level_override,
        ),
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model=selected_model or selection.model,
            requested_provider=selected_provider if explicit_selection else None,
            requested_model=selected_model if explicit_selection else None,
            session_provider_name=record.provider_name,
            thinking_level_override=thinking_level_override,
            cwd=record.cwd,
            storage=jsonl_session_storage(record.path),
            session_id=record.id,
            session_manager=manager,
            provider_name=selection.provider.name,
            inference_provider=inference_provider,
            inference_provider_mode=inference_provider_mode,
            provider_settings=settings,
            runtime_provider_config=selection.provider,
            shell_command_prefix=shell_settings.shell_command_prefix,
            extension_paths=extension_paths,
            extensions_enabled=extensions_enabled,
            project_extensions_enabled=project_extensions_enabled,
            custom_system_prompt=custom_system_prompt,
            append_system_prompt=append_system_prompt,
            trust_override=trust_override,
            trust_default=shell_settings.default_project_trust,
        )
    )
    session.extension_runtime.set_ui_bridge(StderrUiBridge())
    try:
        await RpcServer(session).run()
    finally:
        await provider.aclose()


async def run_openai_print_mode(
    prompt: str,
    model: str | None,
    cwd: Path,
    output: PrintOutputMode = PrintOutputMode.text,
    provider_name: str | None = None,
    session_manager: SessionManager | None = None,
    extension_paths: tuple[Path, ...] = (),
    extensions_enabled: bool = True,
    project_extensions_enabled: bool = False,
    session_id: str | None = None,
    custom_system_prompt: str | None = None,
    append_system_prompt: str | None = None,
    trust_override: TrustOverride | None = None,
    resume_session_id: str | None = None,
    *,
    thinking_level_override: ThinkingLevel | None = None,
) -> bool:
    # [FUNCTION DOC]
    # Run one print-mode turn (new or resumed session) using configured providers.
    # Loads settings, resolves or creates the session record, and determines the provider
    # and model: explicit options win, else the record's provider/model, else defaults.
    # For durable (settings-backed) providers resolves a static selection, computes the
    # Hugging Face inference pin/mode as in RPC mode and constructs the initial provider;
    # dynamic (extension) providers leave it None so the staged session resolves them.
    # Delegates to run_print_mode with all options and returns its success flag; always
    # closes the provider it constructed.
    """Run a new or resumed print-mode turn using the configured provider."""
    settings = load_provider_settings()
    shell_settings = load_shell_settings()
    manager = session_manager or SessionManager()
    record = _print_session_record(
        manager,
        resume_session_id=resume_session_id,
        cwd=cwd,
        settings=settings,
        provider_name=provider_name,
        model=model,
        session_id=session_id,
    )
    explicit_selection = provider_name is not None or model is not None
    selection = None
    if not explicit_selection and record.provider_name is None:
        selection = resolve_provider_selection(settings)
    elif not explicit_selection and record.provider_name is not None:
        try:
            selection = resolve_provider_selection(
                settings,
                provider_name=record.provider_name,
                model=record.model,
            )
        except ProviderConfigError:
            selection = None
    selected_model = model or record.model or (selection.model if selection is not None else "")
    selected_provider: str = (
        provider_name
        if provider_name is not None
        else record.provider_name
        or (selection.provider.name if selection is not None else DEFAULT_PROVIDER_NAME)
    )
    # Durable providers retain the established print-mode construction seam.
    # Dynamic providers are absent from ProviderSettings and therefore remain
    # None until CodingSession's trusted staged environment resolves them.
    static_selection = selection
    if static_selection is None:
        try:
            static_selection = resolve_provider_selection(
                settings,
                provider_name=selected_provider,
                model=selected_model or None,
            )
        except ProviderConfigError:
            static_selection = None
    initial_provider: ClosableModelProvider | None = None
    runtime_config: ProviderConfig | None = None
    runtime_inference = record.inference_provider
    runtime_inference_mode: Literal["automatic", "fixed"] = (
        record.inference_provider_mode or "automatic"
    )
    if static_selection is not None:
        runtime_inference = (
            record.inference_provider
            if (
                resume_session_id is not None
                and record.provider_name == "huggingface"
                and static_selection.provider.name == "huggingface"
                and record.model == static_selection.model
            )
            else (
                static_selection.provider.inference_providers.get(static_selection.model)
                if isinstance(static_selection.provider, OpenAICompatibleProviderConfig)
                and static_selection.provider.name == "huggingface"
                else None
            )
        )
        runtime_inference_mode = (
            record.inference_provider_mode
            if (
                resume_session_id is not None
                and record.provider_name == "huggingface"
                and static_selection.provider.name == "huggingface"
                and record.model == static_selection.model
            )
            else "fixed"
            if runtime_inference is not None
            else "automatic"
        )
        initial_provider = create_model_provider(
            static_selection.provider,
            model=static_selection.model,
            inference_provider=runtime_inference,
            thinking_level=resolve_startup_thinking_level(
                static_selection.provider,
                static_selection.model,
                cli_override=thinking_level_override,
            ),
        )
        selected_provider = static_selection.provider.name
        selected_model = static_selection.model
        runtime_config = static_selection.provider
    try:
        return await run_print_mode(
            prompt=prompt,
            model=selected_model,
            cwd=record.cwd,
            provider=initial_provider,
            output=output,
            storage=jsonl_session_storage(record.path),
            session_id=record.id,
            session_manager=manager,
            provider_name=selected_provider,
            inference_provider=runtime_inference,
            provider_settings=settings,
            runtime_provider_config=runtime_config,
            requested_provider=provider_name if explicit_selection else None,
            requested_model=model if explicit_selection else None,
            session_provider_name=record.provider_name,
            shell_command_prefix=shell_settings.shell_command_prefix,
            extension_paths=extension_paths,
            extensions_enabled=extensions_enabled,
            project_extensions_enabled=project_extensions_enabled,
            custom_system_prompt=custom_system_prompt,
            append_system_prompt=append_system_prompt,
            trust_override=trust_override,
            trust_default=shell_settings.default_project_trust,
            startup_model_override=False,
            inference_provider_mode=runtime_inference_mode,
            thinking_level_override=thinking_level_override,
        )
    finally:
        # This remains the ownership path for the compatibility provider
        # constructed by this legacy wrapper. Dynamic candidates are created
        # and owned inside the staged CodingSession instead.
        if initial_provider is not None:
            await initial_provider.aclose()


def _print_session_record(
    manager: SessionManager,
    *,
    resume_session_id: str | None,
    cwd: Path,
    settings: ProviderSettings,
    provider_name: str | None,
    model: str | None,
    session_id: str | None,
) -> CodingSessionRecord:
    # [FUNCTION DOC]
    # HELPER: Resolve the session record for print/RPC mode. With a resume id returns the
    # existing record (ValueError if unknown). Otherwise resolves the provider selection
    # and exclusively creates a new session with the selected model, provider and Hugging
    # Face inference pin. If selection fails but both provider and model were given
    # explicitly (e.g. a dynamic provider), creates the session with them as given;
    # otherwise the ProviderConfigError propagates. Side effects: writes session index/files.
    """Resolve a resumed transcript or exclusively create a new one."""
    if resume_session_id is not None:
        record = manager.get_session(resume_session_id)
        if record is None:
            raise ValueError(f"Unknown session: {resume_session_id}")
        return record

    try:
        selection = resolve_provider_selection(settings, provider_name=provider_name, model=model)
    except ProviderConfigError:
        if provider_name is None or model is None:
            raise
        return _create_print_session(
            manager,
            cwd=cwd,
            model=model,
            provider_name=provider_name,
            session_id=session_id,
        )
    inference_provider = (
        selection.provider.inference_providers.get(selection.model)
        if isinstance(selection.provider, OpenAICompatibleProviderConfig)
        and selection.provider.name == "huggingface"
        else None
    )
    return _create_print_session(
        manager,
        cwd=cwd,
        model=selection.model,
        provider_name=selection.provider.name,
        inference_provider=inference_provider,
        session_id=session_id,
    )


def _create_print_session(
    manager: SessionManager,
    *,
    cwd: Path,
    model: str,
    provider_name: str | None = None,
    inference_provider: str | None = None,
    session_id: str | None = None,
) -> CodingSessionRecord:
    # [FUNCTION DOC]
    # HELPER: Exclusively create a new session record via the manager (refusing to reuse an
    # existing transcript/id) with cwd, model, provider, inference pin and optional id.
    """Create an isolated print-mode session, refusing transcript collisions."""
    return manager.create_session_exclusive(
        cwd=cwd,
        model=model,
        provider_name=provider_name,
        inference_provider=inference_provider,
        session_id=session_id,
    )


async def run_print_mode(
    *,
    prompt: str,
    model: str,
    cwd: Path,
    provider: ModelProvider | None,
    output: PrintOutputMode = PrintOutputMode.text,
    resource_paths: TauResourcePaths | None = None,
    storage: SessionStorage | None = None,
    session_id: str | None = None,
    session_manager: SessionManager | None = None,
    provider_name: str = DEFAULT_PROVIDER_NAME,
    inference_provider: str | None = None,
    inference_provider_mode: Literal["automatic", "fixed"] | None = None,
    provider_settings: ProviderSettings | None = None,
    runtime_provider_config: ProviderConfig | None = None,
    requested_provider: str | None = None,
    requested_model: str | None = None,
    session_provider_name: str | None = None,
    shell_command_prefix: str | None = None,
    extension_paths: tuple[Path, ...] = (),
    extensions_enabled: bool = True,
    project_extensions_enabled: bool = False,
    custom_system_prompt: str | None = None,
    append_system_prompt: str | None = None,
    trust_override: TrustOverride | None = None,
    trust_default: TrustDefault = "ask",
    startup_model_override: bool = False,
    thinking_level_override: ThinkingLevel | None = None,
) -> bool:
    # [FUNCTION DOC]
    # Run one non-interactive prompt and print streamed events.
    # Keyword inputs: prompt, model, cwd, optional initial provider, output mode, resource
    # paths, storage (defaults to in-memory), session id/manager, provider name and
    # inference settings, provider settings and runtime config, requested provider/model,
    # session provider name, shell command prefix, extension options, system prompt
    # overrides, trust override/default, startup model override flag, thinking override.
    # Behavior: prepares a staged CodingSession. If the prompt is "/system" (optionally with
    # arguments), handles it, aborts the candidate (nothing persisted), prints the message
    # and returns True. Otherwise adopts the session, optionally applies the startup model
    # override, attaches a stderr UI bridge, prints project-trust diagnostics to stderr,
    # emits the pending session-start event and creates the renderer for the output mode.
    # Then: a terminal command ("!" style) runs and prints its formatted result, returning
    # its ok flag; a handled slash command prints its message (interactive-only commands
    # /local and /sidebar print an explanation; /reload reports the reload summary or error;
    # a rename reports the new name or error) and returns True; otherwise streams the
    # prompt's events to the renderer and returns renderer.finish(). The session is always
    # closed after adoption once the main block starts.
    # Output: False only when the agent ended in a non-recoverable error.
    # KNOWN ISSUE: if an exception occurs after adoption but before the try block (e.g.
    # while applying the model override or emitting session start), the session is not closed.
    """Run one non-interactive prompt and print streamed events.

    Returns False when the agent emits a non-recoverable error so CLI callers
    can fail non-interactive runs while still rendering the error message.
    """
    prepared = await prepare_coding_session(
        CodingSessionConfig(
            provider=provider,
            model=model,
            thinking_level_override=thinking_level_override,
            cwd=cwd,
            storage=storage or _MemorySessionStorage(),
            resource_paths=resource_paths,
            session_id=session_id,
            session_manager=session_manager,
            provider_name=provider_name,
            inference_provider=inference_provider,
            inference_provider_mode=inference_provider_mode,
            provider_settings=provider_settings,
            runtime_provider_config=runtime_provider_config,
            requested_provider=requested_provider,
            requested_model=requested_model,
            session_provider_name=session_provider_name,
            shell_command_prefix=shell_command_prefix,
            extension_paths=extension_paths,
            extensions_enabled=extensions_enabled,
            project_extensions_enabled=project_extensions_enabled,
            custom_system_prompt=custom_system_prompt,
            append_system_prompt=append_system_prompt,
            trust_override=trust_override,
            trust_default=trust_default,
        ),
        session_loader=CodingSession,
    )
    # Informational print commands must not publish the staged initial
    # transcript; /system explicitly promises not to save anything.
    if (stripped_prompt := prompt.strip()) == "/system" or stripped_prompt.startswith("/system "):
        command = prepared.session.handle_command(prompt)
        await prepared.abort()
        if command.message:
            typer.echo(command.message)
        return True
    session = await prepared.adopt()
    if startup_model_override:
        await session.apply_startup_model_override(model)
    session.extension_runtime.set_ui_bridge(StderrUiBridge())
    for diagnostic in session.resource_diagnostics:
        if diagnostic.kind == "project-trust":
            typer.echo(diagnostic.format(), err=True)
    await session.emit_pending_session_start()
    renderer = create_event_renderer(
        output,
        custom_message_renderer=session.extension_runtime.render_custom_message,
    )
    try:
        terminal_command = parse_terminal_command(prompt)
        if terminal_command is not None:
            result = await session.run_terminal_command(
                terminal_command.command,
                add_to_context=terminal_command.add_to_context,
            )
            typer.echo(_format_terminal_command_result(result))
            return result.ok
        command = session.handle_command(prompt)
        if command.handled:
            message = command.message
            if command.local_requested:
                message = (
                    "The /local command is interactive-only. In print mode, configure a "
                    "backend in the TUI, then use --provider and --model."
                )
            if command.sidebar_toggle_requested:
                message = "The /sidebar command is interactive-only."
            if command.reload_requested:
                try:
                    summary = await session.reload()
                except ValueError as exc:
                    message = f"Could not reload: {exc}"
                else:
                    message = format_reload_summary(summary)
            if command.session_name is not None:
                try:
                    renamed = await session.set_session_name(command.session_name)
                except ValueError as exc:
                    message = f"Could not rename session: {exc}"
                else:
                    message = f"Session renamed: {renamed}"
            if message:
                typer.echo(message)
            return True
        async for event in session.prompt(prompt):
            renderer.render(event)
        return renderer.finish()
    finally:
        await session.aclose()


class _MemorySessionStorage:
    # [CLASS DOC]
    # HELPER: Append-only in-memory session storage used when print mode is called directly
    # (e.g. tests) without a file. Holds a public list of session entries.
    """Append-only in-memory storage for direct print-mode tests."""

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Start with an empty entry list.
        self.entries: list[SessionEntry] = []

    async def append(self, entry: SessionEntry) -> None:
        # [FUNCTION DOC]
        # HELPER: Append one entry to the list.
        self.entries.append(entry)

    async def append_batch(self, entries: Sequence[SessionEntry]) -> None:
        # [FUNCTION DOC]
        # HELPER: Extend the list with a sequence of entries.
        self.entries.extend(entries)

    async def read_all(self) -> list[SessionEntry]:
        # [FUNCTION DOC]
        # HELPER: Return a shallow copy of all stored entries.
        return list(self.entries)


def _format_terminal_command_result(result: TerminalCommandResult) -> str:
    # [FUNCTION DOC]
    # HELPER: Format a terminal command result as "$ <command>", a bracketed note on whether
    # the output was added to context, and the command output, on separate lines.
    context_status = "added to context" if result.added_to_context else "not added to context"
    return f"$ {result.command}\n[{context_status}]\n{result.output}"
