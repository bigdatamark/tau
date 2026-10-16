# [MODULE DOC]
# Packaged example extension (shipped as data) showing the minimal way to
# register a custom agent tool. Loading it and calling setup registers a tool
# named "hello" that greets someone by name. Used as a reference/template for
# extension authors.
"""Minimal Tau extension that registers a custom tool."""

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


async def _run_hello(
    tool_call_id: str,
    arguments: Mapping[str, JSONValue],
    signal: ToolCancellationToken | None = None,
    on_update: ToolUpdateCallback | None = None,
) -> AgentToolResult:
    # [FUNCTION DOC]
    # HELPER: execute function for the "hello" tool.
    # Inputs: tool call id, argument mapping (optional "who" key), optional
    # cancellation token and update callback (all except arguments are ignored).
    # Output: a tool result whose single text content is "Hello, <who>!", where
    # who is the stringified "who" argument or "world" when missing.
    # No exceptions raised deliberately; no side effects.
    del tool_call_id, signal, on_update
    who = str(arguments.get("who", "world"))
    return AgentToolResult(content=[TextContent(text=f"Hello, {who}!")])


def setup(tau: ExtensionAPI) -> None:
    # [FUNCTION DOC]
    # Extension entry point called by the extension loader with the ExtensionAPI.
    # Registers one AgentTool named "hello" (label "hello") with a JSON schema
    # accepting an optional string "who", the _run_hello execute function and a
    # prompt snippet for the system prompt.
    # Returns nothing. Side effect: mutates the extension API's tool registry.
    """Register the hello tool."""
    tau.register_tool(
        AgentTool(
            name="hello",
            label="hello",
            description="Greet someone by name.",
            parameters={
                "type": "object",
                "properties": {"who": {"type": "string"}},
            },
            execute_fn=_run_hello,
            prompt_snippet="Greet someone by name.",
        )
    )
