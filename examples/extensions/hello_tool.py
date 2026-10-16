# [MODULE DOC]
# Example Tau extension that registers a single custom tool named "hello".
# Demonstrates the extension contract: a module exposing a setup function that
# receives the extension API and registers agent tools. Can be installed into the
# user extensions directory or loaded with the -e CLI flag.
"""Minimal Tau extension: one custom tool.

Install by copying into `~/.tau/extensions/`, or run:

    tau -e examples/extensions/hello_tool.py
"""

from tau_agent.messages import TextContent
from tau_agent.tools import AgentTool, AgentToolResult
from tau_coding.extensions import ExtensionAPI


async def _run_hello(tool_call_id, arguments, signal=None, on_update=None):  # noqa: ANN001, ANN202
    # [FUNCTION DOC]
    # HELPER: Execute function for the hello tool.
    # Inputs: tool call id, an arguments mapping (optional "who" key), optional abort
    # signal and optional progress-update callback (all but arguments are ignored).
    # Output: a tool result containing a single text block "Hello, <who>!", where who
    # defaults to "world" and is converted to a string. Raises nothing; no side effects.
    del tool_call_id, signal, on_update
    who = str(arguments.get("who", "world"))
    return AgentToolResult(content=[TextContent(text=f"Hello, {who}!")])


def setup(tau: ExtensionAPI) -> None:
    # [FUNCTION DOC]
    # Extension entry point. Registers an agent tool named "hello" (label "hello")
    # whose JSON-schema parameters are an object with an optional string "who", whose
    # execute function is the hello helper, and which contributes a short prompt
    # snippet to the system prompt.
    # Input: the extension API object. Output: none.
    # Side effects: mutates the extension API's tool registry.
    """Register the hello tool."""
    tau.register_tool(
        AgentTool(
            name="hello",
            label="hello",
            description="Greet someone by name.",
            parameters={
                "type": "object",
                "properties": {
                    "who": {"type": "string", "description": "Who to greet."},
                },
            },
            execute_fn=_run_hello,
            prompt_snippet="Greet someone by name.",
        )
    )
