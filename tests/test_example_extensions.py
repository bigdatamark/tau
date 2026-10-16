# [MODULE DOC]
# Tests that load the shipped example extensions through the real extension runtime (not a mocked
# API) and exercise the tools they compose: ask_user_question, hello_tool, permission_gate,
# prompt_section and sidebar_status. Doubles as a template for extension authors. All async tests
# run on the asyncio backend via the anyio plugin.
"""Behavioral tests for the shipped example extensions.

Mirrors Pi's practice of testing example extensions from the main suite
(plan-mode-extension.test.ts and friends), with one difference: instead of
mocking the extension API, these load the real files through the real
`ExtensionRuntime` and drive the composed tools — which doubles as a template
for how extension authors can test their own extensions.
"""

from collections.abc import Sequence
from pathlib import Path

import pytest

from tau_agent.messages import TextContent
from tau_agent.tools import AgentTool, AgentToolResult
from tau_coding import TauResourcePaths
from tau_coding.extensions import ExtensionRuntime, NullUiBridge

pytestmark = pytest.mark.anyio

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "extensions"


@pytest.fixture
def anyio_backend() -> str:
    # [FUNCTION DOC]
    # HELPER: pytest fixture pinning the anyio backend to asyncio. Returns the backend name.
    return "asyncio"


def _runtime_with_examples(tmp_path: Path, *names: str) -> ExtensionRuntime:
    # [FUNCTION DOC]
    # HELPER: create an extension runtime and load only the named example files from the
    # repository examples directory (resource directories excluded), using temp
    # home/project/agents paths. Asserts no error-severity diagnostics were produced. Returns the
    # loaded runtime.
    runtime = ExtensionRuntime()
    runtime.load(
        TauResourcePaths(
            root=tmp_path / "home-tau",
            cwd=tmp_path / "project",
            agents_root=tmp_path / "home-agents",
        ),
        extra_paths=tuple(EXAMPLES_DIR / name for name in names),
        include_resource_dirs=False,
    )
    assert not [diag for diag in runtime.diagnostics if diag.severity == "error"]
    return runtime


def _fake_tool(name: str) -> tuple[AgentTool, list[dict[str, object]]]:
    # [FUNCTION DOC]
    # HELPER: build a fake agent tool with the given name whose executor records a copy of each
    # argument mapping and returns a text result "ran". Returns the tool and the list of recorded
    # arguments.
    executed: list[dict[str, object]] = []

    async def executor(tool_call_id, arguments, signal=None, on_update=None):  # noqa: ANN001, ANN202
        # [FUNCTION DOC]
        # HELPER: async tool executor; ignores call id, signal and update callback, appends a copy
        # of the arguments to the record list, returns a result with text "ran".
        del tool_call_id, signal, on_update
        executed.append(dict(arguments))
        return AgentToolResult(content=[TextContent(text="ran")])

    tool = AgentTool(
        name=name,
        label=name,
        description="fake",
        parameters={},
        execute_fn=executor,
    )
    return tool, executed


class SelectingUiBridge(NullUiBridge):
    # [CLASS DOC]
    # HELPER: UI bridge test double that claims a UI is available and answers every select prompt
    # with a preconfigured choice (or None for cancel), recording each prompt.
    def __init__(self, selected: str | None) -> None:
        # [FUNCTION DOC]
        # HELPER: store the choice to return and initialize an empty list of recorded (title,
        # options) calls.
        self.selected = selected
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    @property
    def has_ui(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: property that always reports a UI is present.
        return True

    async def select(
        self,
        title: str,
        options: Sequence[str],
        *,
        timeout: float | None = None,
    ) -> str | None:
        # [FUNCTION DOC]
        # HELPER: async select; ignores timeout, records the title and options as a tuple, and
        # returns the preconfigured choice.
        del timeout
        choices = tuple(options)
        self.calls.append((title, choices))
        return self.selected


# -- shipped examples ----------------------------------------------------------


def test_shipped_examples_load(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Load all five shipped examples; verify the extension names in order, that only
    # ask_user_question and hello contribute tools, and that the first prompt section is "Review
    # procedure" containing a bash code fence.
    runtime = _runtime_with_examples(
        tmp_path,
        "ask_user_question.py",
        "hello_tool.py",
        "permission_gate.py",
        "prompt_section.py",
        "sidebar_status.py",
    )

    assert runtime.extension_names == (
        "ask_user_question",
        "hello_tool",
        "permission_gate",
        "prompt_section",
        "sidebar_status",
    )
    assert [tool.name for tool in runtime.extension_tools] == [
        "ask_user_question",
        "hello",
    ]
    assert runtime.prompt_sections[0].title == "Review procedure"
    assert "```bash" in runtime.prompt_sections[0].body


async def test_ask_user_question_returns_selected_option(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Async: with a UI bridge selecting an option, executing the ask_user_question tool returns
    # text naming the selected option, details with the selection, and the bridge saw the question
    # and options once.
    runtime = _runtime_with_examples(tmp_path, "ask_user_question.py")
    ui = SelectingUiBridge("Run focused tests")
    runtime.set_ui_bridge(ui)
    tool = runtime.compose_tools([])[0]

    result = await tool.execute(
        "test-call",
        {
            "question": "Which checks should I run?",
            "options": ["Run focused tests", "Run all tests"],
        },
    )

    assert result.text == "The user selected: Run focused tests"
    assert result.details == {"selected": "Run focused tests"}
    assert ui.calls == [
        (
            "Which checks should I run?",
            ("Run focused tests", "Run all tests"),
        )
    ]


async def test_ask_user_question_handles_cancel(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Async: when the UI returns no selection, the tool reports the user did not select an option
    # and details contain a null selection.
    runtime = _runtime_with_examples(tmp_path, "ask_user_question.py")
    runtime.set_ui_bridge(SelectingUiBridge(None))
    tool = runtime.compose_tools([])[0]

    result = await tool.execute(
        "test-call",
        {"question": "Continue?", "options": ["Yes", "No"]},
    )

    assert result.text == "The user did not select an option."
    assert result.details == {"selected": None}


# -- hello_tool.py -------------------------------------------------------------


async def test_hello_tool_greets(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Async: the hello tool greets the given name ("Hello, Tau!").
    runtime = _runtime_with_examples(tmp_path, "hello_tool.py")
    hello = runtime.compose_tools([])[0]

    result = await hello.execute("test-call", {"who": "Tau"})

    assert result.text == "Hello, Tau!"


async def test_hello_tool_defaults_to_world(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Async: with no name argument the hello tool greets the world.
    runtime = _runtime_with_examples(tmp_path, "hello_tool.py")
    hello = runtime.compose_tools([])[0]

    result = await hello.execute("test-call", {})

    assert result.text == "Hello, world!"


# -- permission_gate.py ---------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build/",
        "rm -fr /tmp/x",
        "git push --force origin main",
        "git reset --hard HEAD~3",
        "chmod -R 777 .",
        "dd if=/dev/zero of=/dev/sda",
    ],
)
async def test_permission_gate_blocks_dangerous_bash(tmp_path: Path, command: str) -> None:
    # [FUNCTION DOC]
    # Async, parametrized over destructive commands (recursive force rm, force push, hard reset,
    # recursive chmod 777, dd to a device): the gated bash tool returns a blocked message
    # mentioning a guarded pattern and the underlying tool never runs.
    runtime = _runtime_with_examples(tmp_path, "permission_gate.py")
    bash, executed = _fake_tool("bash")
    wrapped = runtime.compose_tools([bash])[0]

    result = await wrapped.execute("test-call", {"command": command})

    assert "blocked" in result.text.lower()
    assert "guarded pattern" in result.text
    assert executed == []


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "rm build/output.txt",
        "git push origin feature-branch",
        "git log --oneline",
    ],
)
async def test_permission_gate_allows_safe_bash(tmp_path: Path, command: str) -> None:
    # [FUNCTION DOC]
    # Async, parametrized over safe commands: the gated bash tool delegates to the underlying tool
    # unchanged and returns its output.
    runtime = _runtime_with_examples(tmp_path, "permission_gate.py")
    bash, executed = _fake_tool("bash")
    wrapped = runtime.compose_tools([bash])[0]

    result = await wrapped.execute("test-call", {"command": command})

    assert result.text == "ran"
    assert executed == [{"command": command}]


async def test_permission_gate_ignores_other_tools(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Async: tools other than bash are passed through even when arguments contain dangerous-
    # looking text.
    runtime = _runtime_with_examples(tmp_path, "permission_gate.py")
    write, executed = _fake_tool("write")
    wrapped = runtime.compose_tools([write])[0]

    result = await wrapped.execute("test-call", {"content": "rm -rf / would be bad"})

    assert result.text == "ran"
    assert len(executed) == 1
