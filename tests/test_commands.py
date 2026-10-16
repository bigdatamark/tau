# [MODULE DOC]
# Tests for the slash-command registry (tau_coding.commands): which inputs are handled, the exact
# Pi-aligned built-in command list, usage messages for stray arguments, and the result flags each
# command sets for the host (pickers, exit/new session, compaction, export, login/logout, reload,
# resume, rename, theme, model switching) plus /session details and /hotkeys text. Uses a fake
# session object exposing the attributes and methods commands read or call.
from pathlib import Path

from tau_coding.commands import CommandRegistry, SlashCommand, create_default_command_registry
from tau_coding.paths import TauPaths
from tau_coding.reload import CodingReloadSummary, ReloadCategorySummary
from tau_coding.session import ModelChoice
from tau_coding.session_manager import SessionManager
from tau_coding.skills import Skill
from tau_coding.system_prompt import ProjectContextFile
from tau_coding.tools import create_coding_tools


class FakeSession:
    # [CLASS DOC]
    # HELPER: Minimal stand-in for a coding session as seen by commands: cwd, provider/model and model
    # choices, Hugging Face inference route and mode, real coding tools for the cwd, one "review"
    # skill, one AGENTS.md context file, context/compaction token numbers, thinking levels, theme,
    # system prompt, session id/title, optional session manager, and flags recording which mutating
    # methods were called.
    def __init__(self, tmp_path: Path, manager: SessionManager | None = None) -> None:
        # [FUNCTION DOC]
        # HELPER: Initialize all fake attributes for the given temp cwd and optional session manager.
        self.cwd = tmp_path
        self.provider_name = "openai"
        self.inference_provider: str | None = None
        self.inference_provider_mode = "automatic"
        self.model = "fake-model"
        self.available_models = ("fake-model", "other-model")
        self.available_model_choices = (
            ModelChoice(provider_name="openai", model="fake-model"),
            ModelChoice(provider_name="openai", model="other-model"),
            ModelChoice(provider_name="local", model="local-model"),
        )
        self.available_providers = ("openai", "local")
        self.tools = tuple(create_coding_tools(cwd=tmp_path))
        self.skills = (
            Skill(
                name="review",
                path=tmp_path / "review.md",
                content="Review code",
                description="Review code",
            ),
        )
        self.prompt_templates = ()
        self.context_files = (
            ProjectContextFile(path=str(tmp_path / "AGENTS.md"), content="Follow instructions."),
        )
        self.context_token_estimate = 123
        self.auto_compact_token_threshold = 200
        self.context_window_tokens = 584
        self.thinking_level = "medium"
        self.available_thinking_levels = ("off", "minimal", "low", "medium", "high", "xhigh")
        self.thinking_unavailable_reason: str | None = None
        self.tui_theme = "tau-dark"
        self.resource_diagnostics = ()
        self.system_prompt = "You are Tau.\nFollow project instructions."
        self.session_id = "session-1"
        self.session_title: str | None = None
        self.session_manager: SessionManager | None = manager
        self.ensure_session_indexed_called = False
        self.reload_called = False
        self.provider_reload_called = False

    def ensure_session_indexed(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Record the call and, if a manager exists, create an index record for this session id.
        self.ensure_session_indexed_called = True
        if self.session_manager is not None:
            self.session_manager.create_session(
                cwd=self.cwd,
                model=self.model,
                provider_name=self.provider_name,
                session_id=self.session_id,
            )

    def set_model(self, model: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Set the current model.
        self.model = model

    def set_inference_provider(self, route: str | None) -> str:
        # [FUNCTION DOC]
        # HELPER: Set the inference route (None means automatic), update the mode accordingly and return
        # the route or an "automatic (will pin ...)" description.
        self.inference_provider = route
        self.inference_provider_mode = "fixed" if route is not None else "automatic"
        return route or "automatic (will pin after the next successful response)"

    def set_provider(self, provider_name: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Switch provider and reset model/available models to the single local model.
        self.provider_name = provider_name
        self.model = "local-model"
        self.available_models = ("local-model",)

    def reload(self) -> CodingReloadSummary:
        # [FUNCTION DOC]
        # HELPER: Record the call and return a canned reload summary per resource category.
        self.reload_called = True
        return CodingReloadSummary(
            skills=ReloadCategorySummary(before=0, after=len(self.skills), changed=True),
            prompt_templates=ReloadCategorySummary(
                before=0,
                after=len(self.prompt_templates),
                changed=False,
            ),
            context_files=ReloadCategorySummary(
                before=0,
                after=len(self.context_files),
                changed=True,
            ),
            extensions=ReloadCategorySummary(before=0, after=0, changed=False),
            diagnostics=ReloadCategorySummary(
                before=0,
                after=len(self.resource_diagnostics),
                changed=False,
            ),
            system_prompt_rebuilt=True,
        )

    def reload_provider_settings(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Record that provider settings were reloaded.
        self.provider_reload_called = True


def test_registry_ignores_ordinary_prompts_and_skill_expansion(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: plain prompts and "/skill:..." expansions are not handled by the registry.
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)

    assert registry.execute(session, "hello").handled is False
    assert registry.execute(session, "/skill:review fix this").handled is False


def test_registry_ignores_unregistered_slash_prompts(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: unregistered slash inputs, including path-like ones, are unhandled with no message.
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)

    for prompt in ("/missing", "/README.md", "/tmp", "/Users/me/screenshot.png"):
        result = registry.execute(session, prompt)
        assert result.handled is False
        assert result.message is None


def test_registered_commands_are_pi_aligned(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: the default registry lists exactly the expected built-in command names in sorted order.
    commands = create_default_command_registry().list_commands()

    assert [command.name for command in commands] == [
        "compact",
        "export",
        "hotkeys",
        "local",
        "login",
        "logout",
        "model",
        "name",
        "new",
        "prompts",
        "quit",
        "reload",
        "resume",
        "scoped-models",
        "session",
        "sidebar",
        "skill",
        "skills",
        "system",
        "theme",
        "tools",
        "tree",
    ]


def test_local_command_requests_host_action(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /local requests the host local action; extra args yield a usage message.
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)

    assert registry.execute(session, "/local").local_requested is True
    assert registry.execute(session, "/local extra").message == "Usage: /local"


def test_sidebar_command_requests_host_action(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /sidebar requests a sidebar toggle; extra args yield usage.
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)

    result = registry.execute(session, "/sidebar")

    assert result.handled is True
    assert result.sidebar_toggle_requested is True
    assert registry.execute(session, "/sidebar extra").message == "Usage: /sidebar"


def test_prompts_command_requests_picker(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /prompts requests the prompts picker; extra args yield usage.
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)

    assert registry.execute(session, "/prompts").prompts_picker_requested is True
    assert registry.execute(session, "/prompts extra").message == "Usage: /prompts"


def test_system_command_returns_active_prompt(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /system returns the active system prompt formatted under a numbered heading with source;
    # extra args yield usage.
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)

    result = registry.execute(session, "/system")

    assert result.handled is True
    assert result.message == (
        "#### 01 · Effective system prompt\n\n"
        "**Source:** `active Tau session`\n\n"
        "You are Tau.\nFollow project instructions."
    )
    assert registry.execute(session, "/system extra").message == "Usage: /system"


def test_quit_and_new_return_control_flags(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /quit and alias /exit request exit, /new requests a new session, while /q and /clear are not handled.
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)

    assert registry.execute(session, "/quit").exit_requested is True
    assert registry.execute(session, "/exit").exit_requested is True
    assert registry.execute(session, "/q").handled is False
    assert registry.execute(session, "/new").new_session_requested is True
    assert registry.execute(session, "/clear").handled is False


def test_compact_command_accepts_optional_instructions(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /compact without args yields empty summary instructions; with text, that text is passed through.
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)

    default = registry.execute(session, "/compact")
    requested = registry.execute(session, "/compact Summary of prior work.")

    assert default.compact_summary == ""
    assert requested.compact_summary == "Summary of prior work."


def test_skills_command_requests_picker(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /skills requests the skills picker; extra args yield usage.
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)

    result = registry.execute(session, "/skills")

    assert result.handled is True
    assert result.skills_picker_requested is True
    assert registry.execute(session, "/skills extra").message == "Usage: /skills"


def test_tree_command_requests_picker(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /tree requests the tree picker; any argument yields usage.
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)

    result = registry.execute(session, "/tree")
    with_args = registry.execute(session, "/tree root")

    assert result.handled is True
    assert result.tree_picker_requested is True
    assert with_args.message == "Usage: /tree"


def test_export_command_requests_default_export(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: bare /export requests an export with no destination or format.
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/export")

    assert result.handled is True
    assert result.export_requested is True
    assert result.export_destination is None
    assert result.export_format is None


def test_export_command_parses_format_and_destination(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /export parses a --format option and a destination path.
    result = create_default_command_registry().execute(
        FakeSession(tmp_path),
        "/export --format jsonl exports/session.jsonl",
    )

    assert result.export_requested is True
    assert result.export_format == "jsonl"
    assert result.export_destination == Path("exports/session.jsonl")


def test_session_command_includes_session_details(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /session reports model, cwd, tool/skill/context counts, token numbers, thinking mode, compaction
    # threshold, diagnostics and session id (no name line when untitled); /status is not handled.
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/session")

    assert result.message is not None
    assert "Model: fake-model" in result.message
    assert f"CWD: {tmp_path}" in result.message
    assert "Tools: 4" in result.message
    assert "Skills: 1" in result.message
    assert "Context files: 1" in result.message
    assert "Estimated context tokens: 123" in result.message
    assert "Context window: 584" in result.message
    assert "Thinking mode: medium" in result.message
    assert "Auto compact threshold: 200" in result.message
    assert "Resource diagnostics: 0" in result.message
    assert "Session: session-1" in result.message
    assert "Session name:" not in result.message
    assert (
        create_default_command_registry().execute(FakeSession(tmp_path), "/status").handled is False
    )


def test_session_command_distinguishes_automatic_and_fixed_huggingface_routes(
    tmp_path: Path,
) -> None:
    # [FUNCTION DOC]
    # Test: for the huggingface provider, /session shows an automatic route as "automatic (currently X)"
    # and a fixed route as "X (fixed)".
    session = FakeSession(tmp_path)
    session.provider_name = "huggingface"
    session.inference_provider = "baseten"

    automatic = create_default_command_registry().execute(session, "/session")
    session.inference_provider_mode = "fixed"
    fixed = create_default_command_registry().execute(session, "/session")

    assert automatic.message is not None
    assert "Hugging Face inference provider: automatic (currently baseten)" in automatic.message
    assert fixed.message is not None
    assert "Hugging Face inference provider: baseten (fixed)" in fixed.message


def test_route_command_is_not_built_in(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /route is not a built-in command.
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/route deepinfra")

    assert result.handled is False


def test_session_command_includes_named_session_title(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /session includes the session name when the session has a title.
    session = FakeSession(tmp_path)
    session.session_title = "Customer bugfix"

    result = create_default_command_registry().execute(session, "/session")

    assert result.message is not None
    assert "Session: session-1" in result.message
    assert "Session name: Customer bugfix" in result.message


def test_session_command_explains_unavailable_thinking_controls(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: with no available thinking levels, /session shows thinking as unavailable with the reason.
    session = FakeSession(tmp_path)
    session.available_thinking_levels = ()
    session.thinking_unavailable_reason = "Provider local does not declare thinking_levels"

    result = create_default_command_registry().execute(session, "/session")

    assert result.message is not None
    assert "Thinking mode: unavailable" in result.message
    assert "Thinking unavailable: Provider local does not declare thinking_levels" in result.message
    assert "Thinking mode: medium" not in result.message


def test_hotkeys_command_lists_common_tui_shortcuts(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /hotkeys lists common TUI keyboard shortcuts.
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/hotkeys")

    assert result.message is not None
    assert "Common keyboard shortcuts:" in result.message
    assert "Ctrl+K: open slash-command completions" in result.message
    assert "Ctrl+R: open session picker" in result.message
    assert "Ctrl+P / Shift+Ctrl+P: cycle scoped models forward / backward" in result.message
    assert "Shift+Tab: cycle thinking mode" in result.message


def test_model_command_requests_picker_and_switches_models(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: bare /model requests the model picker; /model NAME switches the model without reloading
    # provider settings and reports the current model.
    session = FakeSession(tmp_path)
    registry = create_default_command_registry()

    list_result = registry.execute(session, "/model")
    switch_result = registry.execute(session, "/model other-model")

    assert list_result.model_picker_requested is True
    assert switch_result.message == "Current model: other-model"
    assert session.model == "other-model"
    assert session.provider_reload_called is False


def test_scoped_models_command_requests_scoped_picker(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: both "/scoped-models" and Pi-style "/scoped models" request the scoped models picker
    # without reloading provider settings.
    session = FakeSession(tmp_path)
    registry = create_default_command_registry()

    dashed_result = registry.execute(session, "/scoped-models")
    pi_style_result = registry.execute(session, "/scoped models")

    assert dashed_result.scoped_models_picker_requested is True
    assert pi_style_result.scoped_models_picker_requested is True
    assert session.provider_reload_called is False


def test_model_command_rejects_unknown_model(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /model with an unknown name reports an error naming the provider and leaves the model unchanged.
    session = FakeSession(tmp_path)

    result = create_default_command_registry().execute(session, "/model missing")

    assert result.message is not None
    assert "Unknown model for provider openai: missing" in result.message
    assert session.model == "fake-model"


def test_model_pickers_open_even_if_provider_settings_cannot_reload(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: model pickers open even if reloading provider settings would fail (reload is never called).
    class FailingRefreshSession(FakeSession):
        # [CLASS DOC]
        # HELPER: Fake session whose provider-settings reload raises ValueError.
        def reload_provider_settings(self) -> None:
            # [FUNCTION DOC]
            # HELPER: Always raise ValueError about a missing active model.
            raise ValueError("active model is missing from the refreshed catalog")

    session = FailingRefreshSession(tmp_path)
    registry = create_default_command_registry()
    assert registry.execute(session, "/model").model_picker_requested is True
    assert registry.execute(session, "/scoped-models").scoped_models_picker_requested is True
    assert session.provider_reload_called is False


def test_theme_command_requests_picker_and_sets_theme(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: bare /theme requests the theme picker, a known theme sets it, and an unknown theme errors.
    session = FakeSession(tmp_path)
    registry = create_default_command_registry()

    list_result = registry.execute(session, "/theme")
    switch_result = registry.execute(session, "/theme tau-light")
    unknown_result = registry.execute(session, "/theme solarized")

    assert list_result.theme_picker_requested is True
    assert switch_result.theme == "tau-light"
    assert unknown_result.message is not None
    assert "Unknown theme: solarized" in unknown_result.message


def test_theme_command_accepts_registered_custom_theme(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: a registered custom theme is accepted by /theme and listed in the unknown-theme message.
    # Side effect: mutates the global custom theme registry, restored to empty in a finally block.
    from tau_coding.tui.themes import (
        THEME_COLOR_FIELDS,
        TRANSCRIPT_ROLES,
        parse_tui_theme_json,
        set_custom_tui_themes,
    )

    theme_data = {
        "name": "midnight",
        "colors": dict.fromkeys(THEME_COLOR_FIELDS, "#101010"),
        "roles": {role: {"border": "#101010", "body": "#e0e0e0"} for role in TRANSCRIPT_ROLES},
    }
    set_custom_tui_themes({"midnight": parse_tui_theme_json(theme_data)})
    try:
        result = create_default_command_registry().execute(FakeSession(tmp_path), "/theme midnight")
        unknown = create_default_command_registry().execute(FakeSession(tmp_path), "/theme nope")
    finally:
        set_custom_tui_themes({})

    assert result.theme == "midnight"
    assert unknown.message is not None
    assert "midnight" in unknown.message


def test_non_pi_commands_are_not_registered(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: non-Pi commands (/provider, /resources, /context, /help) are not handled.
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)

    for command in ("/provider", "/resources", "/context", "/help"):
        result = registry.execute(session, command)
        assert result.handled is False
        assert result.message is None


def test_tools_command_requests_read_only_picker(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /tools requests the read-only tools picker with no message.
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/tools")

    assert result.handled is True
    assert result.tools_picker_requested is True
    assert result.message is None


def test_login_command_requests_provider_picker(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: bare /login requests the login provider picker.
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/login")

    assert result.handled is True
    assert result.login_picker_requested is True


def test_login_command_requests_provider_login(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /login PROVIDER requests login for that provider.
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/login openai")

    assert result.handled is True
    assert result.login_provider == "openai"


def test_login_command_resolves_anthropic_auth_aliases(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: anthropic-api and anthropic-subscription aliases resolve to provider anthropic with the
    # api-key or subscription login method.
    registry = create_default_command_registry()
    session = FakeSession(tmp_path)

    api_result = registry.execute(session, "/login anthropic-api")
    subscription_result = registry.execute(session, "/login anthropic-subscription")

    assert api_result.login_provider == "anthropic"
    assert api_result.login_method == "api-key"
    assert subscription_result.login_provider == "anthropic"
    assert subscription_result.login_method == "subscription"


def test_login_command_lists_auth_aliases_for_unknown_provider(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: an unknown login provider message lists the Anthropic auth aliases.
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/login missing")

    assert result.message is not None
    assert "anthropic-api" in result.message
    assert "anthropic-subscription" in result.message


def test_login_command_requests_custom_provider_login(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /login custom requests the custom provider login flow.
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/login custom")

    assert result.handled is True
    assert result.custom_provider_login_requested is True


def test_logout_command_requests_provider_picker(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: bare /logout requests the logout provider picker.
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/logout")

    assert result.handled is True
    assert result.logout_picker_requested is True


def test_logout_command_requests_provider_logout(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /logout PROVIDER requests logout for that provider.
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/logout openai")

    assert result.handled is True
    assert result.logout_provider == "openai"


def test_logout_command_rejects_unknown_provider(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /logout with a provider without stored login returns an unknown-provider message.
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/logout local")

    assert result.handled is True
    assert result.message is not None
    assert "Unknown logout provider: local" in result.message


def test_reload_command_requests_async_session_reload(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /reload only flags an async reload request; the command itself does not call session reload.
    session = FakeSession(tmp_path)

    result = create_default_command_registry().execute(session, "/reload")

    assert result.handled is True
    assert result.reload_requested is True
    assert result.message is None
    assert session.reload_called is False
    assert session.provider_reload_called is False


def test_resume_without_argument_requests_picker(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: bare /resume requests the session picker; /sessions is not handled. Writes index under tmp_path.
    manager = SessionManager(TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents"))
    session = FakeSession(tmp_path, manager=manager)

    result = create_default_command_registry().execute(session, "/resume")

    assert result.resume_picker_requested is True
    assert result.message is None
    assert create_default_command_registry().execute(session, "/sessions").handled is False


def test_resume_command_requests_indexed_session(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /resume ID for an indexed session requests resuming that session id.
    manager = SessionManager(TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents"))
    record = manager.create_session(cwd=tmp_path, model="fake-model", title="Test session")
    session = FakeSession(tmp_path, manager=manager)

    result = create_default_command_registry().execute(session, f"/resume {record.id}")

    assert result.resume_session_id == record.id
    assert result.message is None


def test_resume_command_rejects_missing_or_unknown_session(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /resume with an unknown id reports "Unknown session: missing".
    # KNOWN ISSUE: despite the name, only the unknown-id case is exercised, not a missing argument.
    manager = SessionManager(TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents"))
    session = FakeSession(tmp_path, manager=manager)

    unknown = create_default_command_registry().execute(session, "/resume missing")

    assert unknown.message == "Unknown session: missing"


def test_name_command_shows_current_name_and_usage(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: bare /name shows the current session name and usage.
    manager = SessionManager(TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents"))
    record = manager.create_session(cwd=tmp_path, model="fake-model", title="Test session")
    session = FakeSession(tmp_path, manager=manager)
    session.session_id = record.id

    result = create_default_command_registry().execute(session, "/name")

    assert result.message == "Current session name: Test session\nUsage: /name <new name>"


def test_name_command_requests_session_rename(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /name NEW returns a rename request and message without writing the index itself (title unchanged).
    manager = SessionManager(TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents"))
    record = manager.create_session(cwd=tmp_path, model="fake-model", title="Old name")
    session = FakeSession(tmp_path, manager=manager)
    session.session_id = record.id

    result = create_default_command_registry().execute(session, "/name Customer bugfix")

    assert result.message == "Session renamed: Customer bugfix"
    assert result.session_name == "Customer bugfix"
    unchanged = manager.get_session(record.id)
    assert unchanged is not None
    assert unchanged.title == "Old name"


def test_name_command_defers_indexing_pending_session(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: renaming a not-yet-indexed session does not force indexing and creates no index record.
    manager = SessionManager(TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents"))
    session = FakeSession(tmp_path, manager=manager)
    session.session_id = "pending-session"

    result = create_default_command_registry().execute(session, "/name Customer bugfix")

    assert result.message == "Session renamed: Customer bugfix"
    assert result.session_name == "Customer bugfix"
    assert session.ensure_session_indexed_called is False
    assert manager.get_session("pending-session") is None


def test_name_command_reports_missing_session_manager(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: /name without a session manager reports that the manager is unavailable.
    result = create_default_command_registry().execute(FakeSession(tmp_path), "/name Work")

    assert result.message == "Session manager is not available."


def test_name_command_rejects_multiline_name(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Test: a multi-line name is rejected and the index record is unchanged.
    manager = SessionManager(TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents"))
    record = manager.create_session(cwd=tmp_path, model="fake-model")
    session = FakeSession(tmp_path, manager=manager)
    session.session_id = record.id

    result = create_default_command_registry().execute(session, "/name Bad\nName")

    assert result.message == "Session name must be a single line."
    assert manager.get_session(record.id) == record


def test_registry_rejects_duplicate_commands_and_aliases() -> None:
    # [FUNCTION DOC]
    # Test: registering the same command twice raises ValueError mentioning a duplicate slash command.
    # KNOWN ISSUE: despite the name, duplicate alias rejection is not exercised.
    registry = CommandRegistry()
    command = SlashCommand(
        name="test",
        usage="/test",
        description="Test",
        handler=lambda context: create_default_command_registry().execute(
            context.session, "/session"
        ),
    )
    registry.register(command)

    try:
        registry.register(command)
    except ValueError as exc:
        assert "Duplicate slash command" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected duplicate command to fail")
