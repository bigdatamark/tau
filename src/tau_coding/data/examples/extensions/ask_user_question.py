# [MODULE DOC]
# Packaged example extension (shipped as data) that lets the agent ask the user
# a multiple-choice question through the active UI and wait for the answer.
# Registers a sequential tool named "ask_user_question". Identical to the
# repository copy under examples/extensions.
"""Let the agent ask the user a multiple-choice question.

Install by copying into `~/.tau/extensions/`, or run:

    tau -e examples/extensions/ask_user_question.py
"""

from collections.abc import Mapping

from tau_agent.messages import TextContent
from tau_agent.tools import (
    AgentTool,
    AgentToolResult,
    ToolCancellationToken,
    ToolUpdateCallback,
)
from tau_agent.types import JSONValue
from tau_coding.extensions import ExtensionAPI


def _result(text: str, *, selected: str | None = None) -> AgentToolResult:
    # [FUNCTION DOC]
    # HELPER: build a tool result with one text content block and details
    # containing the "selected" option (or null when nothing was selected).
    # Inputs: message text and optional keyword-only selected option.
    # Output: the constructed tool result. No side effects.
    return AgentToolResult(
        content=[TextContent(text=text)],
        details={"selected": selected},
    )


def setup(tau: ExtensionAPI) -> None:
    # [FUNCTION DOC]
    # Extension entry point: defines the ask_user_question execute function as a
    # closure over the API (so it can reach the UI context at call time) and
    # registers it as an AgentTool with a strict JSON schema (non-empty question,
    # at least two unique non-empty string options, no extra properties),
    # sequential execution mode, a prompt snippet and a guideline telling the
    # model not to assume an answer when nothing is selected.
    # Returns nothing. Side effect: mutates the extension API's tool registry.
    """Register the ask_user_question tool."""

    async def ask_user_question(
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        # [FUNCTION DOC]
        # Tool execute function (nested closure). Validates arguments itself rather
        # than trusting the schema: question must be a non-blank string; options must
        # be a list of at least two items, each a non-blank string (stripped), and the
        # stripped options must be unique. Any violation returns an explanatory error
        # text result (no exception). On success awaits the UI select prompt with the
        # stripped question and options.
        # Output: a result saying the user did not select an option (selected null)
        # or "The user selected: <option>" with the option recorded in details.
        # Side effects: shows an interactive selection prompt in the frontend and
        # blocks until the user responds. The call id, cancellation token and update
        # callback are ignored, so cancellation is not honored while waiting.
        # Exceptions: whatever the UI select implementation raises propagates.
        del tool_call_id, signal, on_update

        question = arguments.get("question")
        if not isinstance(question, str) or not question.strip():
            return _result("The question must be a non-empty string.")

        raw_options = arguments.get("options")
        if not isinstance(raw_options, list) or len(raw_options) < 2:
            return _result("At least two options are required.")

        options: list[str] = []
        for option in raw_options:
            if not isinstance(option, str) or not option.strip():
                return _result("Every option must be a non-empty string.")
            options.append(option.strip())
        if len(set(options)) != len(options):
            return _result("Options must be unique.")

        selected = await tau.context.ui.select(question.strip(), options)
        if selected is None:
            return _result("The user did not select an option.")
        return _result(f"The user selected: {selected}", selected=selected)

    tau.register_tool(
        AgentTool(
            name="ask_user_question",
            label="ask user question",
            description=(
                "Ask the user one multiple-choice question and wait for their selection. "
                "Use this when user input is needed before proceeding."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "minLength": 1,
                        "description": "The question to show the user.",
                    },
                    "options": {
                        "type": "array",
                        "minItems": 2,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                        "description": "The available answers.",
                    },
                },
                "required": ["question", "options"],
                "additionalProperties": False,
            },
            execute_fn=ask_user_question,
            execution_mode="sequential",
            prompt_snippet="Ask the user a multiple-choice question when their input is needed.",
            prompt_guidelines=(
                "When ask_user_question reports no selection, do not assume an answer; "
                "ask how the user wants to proceed.",
            ),
        )
    )
