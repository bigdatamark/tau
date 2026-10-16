# [MODULE DOC]
# Provider-neutral tool model for the portable agent harness (mirrors Pi's
# AgentTool). Defines the tool result wire model, callback/renderer protocols,
# type aliases for update callbacks, execution mode ("sequential" or
# "parallel") and argument preparers, and the immutable tool definition that the
# agent loop advertises to providers and invokes when the model calls a tool.
# Also re-exports the tool call message type. No I/O; pure data and protocols.
"""Pi-compatible provider-neutral tool definitions and execution results."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import Field, model_validator

from tau_agent.messages import ImageContent, TextContent, ToolCall, WireModel
from tau_agent.types import JSONValue


class ToolCancellationToken(Protocol):
    # [CLASS DOC]
    # Structural protocol for a cancellation signal passed to tools: any object with
    # an is_cancelled method returning a boolean satisfies it.
    def is_cancelled(self) -> bool:
        # [FUNCTION DOC]
        # Protocol method: return True when the tool should stop work early. No inputs;
        # implementations should be side-effect free.
        """Return whether tool execution should stop."""
        ...


class AgentToolResult(WireModel):
    # [CLASS DOC]
    # Wire model for a final or partial (streamed progress) tool result.
    # Fields: content (list of text/image blocks, default empty), details (arbitrary
    # JSON-compatible data for UIs/sessions, not sent as text), added_tool_names
    # (optional list of tool names this call made available, for dynamic tool
    # activation) and terminate (optional flag asking the loop to stop after tools).
    # Validation: a plain string passed as content is normalized into a single text
    # block (or an empty list for an empty string).
    """Final or partial result produced by a tool."""

    content: list[TextContent | ImageContent] = Field(default_factory=list)
    details: JSONValue = None
    added_tool_names: list[str] | None = None
    terminate: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_text_content(cls, value: object) -> object:
        # [FUNCTION DOC]
        # HELPER: Pre-validation hook. Inputs: the raw value being validated.
        # Behavior: if the value is a mapping whose content entry is a string, returns a
        # shallow copy with content replaced by a one-element list containing a text
        # block (or an empty list when the string is empty); any other input is returned
        # unchanged. Outputs: the possibly rewritten value. No side effects; raises
        # nothing.
        if not isinstance(value, dict):
            return value
        data = dict(value)
        content = data.get("content")
        if isinstance(content, str):
            data["content"] = [TextContent(text=content)] if content else []
        return data

    @property
    def text(self) -> str:
        # [FUNCTION DOC]
        # Read-only property: concatenation (no separator) of the text of all text
        # blocks in content, ignoring image blocks. Returns an empty string when none.
        return "".join(block.text for block in self.content if isinstance(block, TextContent))


class ToolCallRenderer(Protocol):
    # [CLASS DOC]
    # Protocol for an optional per-tool callable that formats a tool invocation for
    # frontends.
    def __call__(self, arguments: Mapping[str, JSONValue]) -> str | None:
        # [FUNCTION DOC]
        # Inputs: the tool call arguments mapping. Outputs: a display string, or None to
        # fall back to the frontend's default rendering. Should be side-effect free.
        """Return a frontend-friendly tool invocation, or ``None``."""
        ...


class ToolResultRenderer(Protocol):
    # [CLASS DOC]
    # Protocol for an optional per-tool callable that formats a tool result as
    # frontend markup.
    def __call__(self, result: AgentToolResult, *, expanded: bool) -> str | None:
        # [FUNCTION DOC]
        # Inputs: the tool result and (keyword) whether the expanded view is requested.
        # Outputs: markup string, or None to use the default rendering.
        """Return frontend markup for a tool result, or ``None``."""
        ...


ToolUpdateCallback = Callable[[AgentToolResult], None]


class ToolExecutor(Protocol):
    # [CLASS DOC]
    # Protocol for the async function that actually runs a tool.
    def __call__(
        self,
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> Awaitable[AgentToolResult]:
        # [FUNCTION DOC]
        # Inputs: the tool call ID, validated arguments mapping, optional cancellation
        # token and optional progress callback that receives partial results.
        # Outputs: an awaitable resolving to the final tool result. Implementations may
        # raise; the agent loop is responsible for turning exceptions into error results.
        """Execute one validated tool call."""
        ...


ToolExecutionMode = Literal["sequential", "parallel"]
ToolArgumentPreparer = Callable[[object], Mapping[str, JSONValue]]


@dataclass(frozen=True, slots=True)
class AgentTool:
    # [CLASS DOC]
    # Immutable (frozen, slotted) dataclass describing a tool exposed to the agent
    # loop: unique name, human label, model-facing description, JSON Schema
    # parameters, the executor function, optional system-prompt snippet and
    # guideline lines, optional argument preparer (normalizes raw model arguments
    # before validation), execution mode (default parallel; sequential tools are not
    # run concurrently with others), and optional call/result renderers for UIs.
    """A tool exposed to the portable agent loop."""

    name: str
    label: str
    description: str
    parameters: Mapping[str, JSONValue]
    execute_fn: ToolExecutor
    prompt_snippet: str | None = None
    prompt_guidelines: tuple[str, ...] = ()
    prepare_arguments: ToolArgumentPreparer | None = None
    execution_mode: ToolExecutionMode = "parallel"
    render_call: ToolCallRenderer | None = None
    render_result: ToolResultRenderer | None = None

    @property
    def input_schema(self) -> Mapping[str, JSONValue]:
        # [FUNCTION DOC]
        # Read-only property alias returning the parameters JSON Schema, used by
        # provider payload builders. No side effects.
        """Alias used by provider payload builders."""
        return self.parameters

    async def execute(
        self,
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        # [FUNCTION DOC]
        # Run the tool by awaiting its executor with the call ID, arguments, optional
        # cancellation token and optional progress callback, passed positionally.
        # Outputs: the executor's tool result. Raises whatever the executor raises.
        # Side effects: whatever the executor does (file I/O, processes, etc.).
        """Execute a tool with Pi-compatible call-id and progress semantics."""
        return await self.execute_fn(tool_call_id, arguments, signal, on_update)


__all__ = [
    "AgentTool",
    "AgentToolResult",
    "ToolCall",
    "ToolCallRenderer",
    "ToolCancellationToken",
    "ToolExecutionMode",
    "ToolResultRenderer",
    "ToolExecutor",
    "ToolUpdateCallback",
]
