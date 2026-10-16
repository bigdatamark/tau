# [MODULE DOC]
# Core transcript data model for the portable agent harness (tau_agent).
# Defines Pi-compatible, provider-neutral content blocks (text, thinking, image,
# tool call), usage/cost/timing records, and the transcript message types
# (user, assistant, tool result, bash execution, custom, branch summary, compaction
# summary) discriminated by their "role" field. All models are strict Pydantic
# models using snake_case attribute names in Python and camelCase aliases on the
# wire, so serialized sessions match Pi's JSON format. Also provides helpers to
# sum usage, build assistant content, and extract visible text.
"""Pi-compatible provider-neutral content and transcript message models."""

from __future__ import annotations

from collections.abc import Iterable
from time import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau_agent.types import JSONValue


def _to_camel(name: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Convert a snake_case name to camelCase (first part unchanged, later parts
    # title-cased). Used as the alias generator for wire models.
    parts = name.split("_")
    return parts[0] + "".join(part.title() for part in parts[1:])


def current_timestamp_ms() -> int:
    # [FUNCTION DOC]
    # Return the current wall-clock Unix time in whole milliseconds. Used as the default
    # timestamp factory for messages. Side effects: reads the system clock.
    """Return the current Unix timestamp in milliseconds."""
    return int(time() * 1000)


class WireModel(BaseModel):
    # [CLASS DOC]
    # Base class for all wire models: forbids unknown fields, accepts both Python names
    # and camelCase aliases on input, and serializes using the camelCase aliases.
    """Strict model with Python field names and Pi-compatible JSON aliases."""

    model_config = ConfigDict(
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
        alias_generator=_to_camel,
    )


class UsageCost(WireModel):
    # [CLASS DOC]
    # Cost breakdown in USD for one response: input, output, cache read, cache write,
    # and total; all default to zero.
    """Billed response cost in USD."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


class Usage(WireModel):
    # [CLASS DOC]
    # Token usage for one assistant response: input, output, cache read, cache write
    # counts, optional one-hour cache write and reasoning token counts (None means not
    # reported), total tokens, and a nested cost breakdown.
    """Provider-reported token usage for one assistant response."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_1h: int | None = None
    reasoning: int | None = None
    total_tokens: int = 0
    cost: UsageCost = UsageCost()


def sum_usage(usages: Iterable[Usage]) -> Usage:
    # [FUNCTION DOC]
    # Sum several usage records field by field (including nested costs).
    # Input: any iterable of usage records (consumed once). Output: a new usage record.
    # Optional counts are None in the result only if every input reported None;
    # otherwise the non-None values are summed. An empty input yields all zeros and
    # None optionals. Pure.
    """Return the field-wise total for one or more provider requests."""
    items = tuple(usages)

    def optional_total(field: Literal["cache_write_1h", "reasoning"]) -> int | None:
        # [FUNCTION DOC]
        # HELPER: Sum one optional field ("cache_write_1h" or "reasoning") across the
        # items, returning None when no item reported a value.
        values = [getattr(item, field) for item in items]
        return (
            sum(value for value in values if value is not None)
            if any(value is not None for value in values)
            else None
        )

    return Usage(
        input=sum(item.input for item in items),
        output=sum(item.output for item in items),
        cache_read=sum(item.cache_read for item in items),
        cache_write=sum(item.cache_write for item in items),
        cache_write_1h=optional_total("cache_write_1h"),
        reasoning=optional_total("reasoning"),
        total_tokens=sum(item.total_tokens for item in items),
        cost=UsageCost(
            input=sum(item.cost.input for item in items),
            output=sum(item.cost.output for item in items),
            cache_read=sum(item.cost.cache_read for item in items),
            cache_write=sum(item.cost.cache_write for item in items),
            total=sum(item.cost.total for item in items),
        ),
    )


class ResponseTiming(WireModel):
    # [CLASS DOC]
    # Timing for one response, in milliseconds: optional time to first output and the
    # required total duration; both validated non-negative.
    """Monotonic request durations for one assistant response."""

    time_to_first_output_ms: int | None = Field(default=None, ge=0)
    total_duration_ms: int = Field(ge=0)


class TextContent(WireModel):
    # [CLASS DOC]
    # Visible text content block: type "text", the text, and an optional provider
    # signature used to replay the block faithfully.
    type: Literal["text"] = "text"
    text: str
    text_signature: str | None = None


class ThinkingContent(WireModel):
    # [CLASS DOC]
    # Reasoning content block: type "thinking", the thinking text, an optional provider
    # signature/opaque payload needed to replay it, and a redacted flag.
    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: str | None = None
    redacted: bool = False


class ImageContent(WireModel):
    # [CLASS DOC]
    # Image content block: type "image", base64-encoded data string and MIME type.
    type: Literal["image"] = "image"
    data: str
    mime_type: str


class ToolCall(WireModel):
    # [CLASS DOC]
    # Tool call block requested by the assistant: type "toolCall", call id, tool name,
    # JSON arguments mapping (default empty), and optional provider thought signature.
    """A tool call content block requested by the assistant."""

    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, JSONValue] = Field(default_factory=dict)
    thought_signature: str | None = None


type UserContent = str | list[TextContent | ImageContent]
type AssistantContent = TextContent | ThinkingContent | ToolCall
type ToolResultContent = TextContent | ImageContent


class UserMessage(WireModel):
    # [CLASS DOC]
    # User transcript message: role "user", content as either a plain string or a list
    # of text/image blocks, and a millisecond timestamp (defaults to now).
    role: Literal["user"] = "user"
    content: UserContent
    timestamp: int = Field(default_factory=current_timestamp_ms)

    @property
    def text(self) -> str:
        # [FUNCTION DOC]
        # Property: the visible text of the content (string as is, or concatenated text
        # blocks; images ignored).
        return content_text(self.content)


class AssistantDiagnosticError(WireModel):
    # [CLASS DOC]
    # Error details attached to an assistant diagnostic: optional name, message,
    # optional stack trace, and optional string or integer code.
    name: str | None = None
    message: str
    stack: str | None = None
    code: str | int | None = None


class AssistantMessageDiagnostic(WireModel):
    # [CLASS DOC]
    # Diagnostic record attached to an assistant message: a free-form type, timestamp,
    # optional error details, and optional JSON details mapping.
    type: str
    timestamp: int = Field(default_factory=current_timestamp_ms)
    error: AssistantDiagnosticError | None = None
    details: dict[str, JSONValue] | None = None


StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]


class AssistantMessage(WireModel):
    # [CLASS DOC]
    # Assistant transcript message with ordered content blocks (text, thinking, tool
    # calls), API/provider/model identifiers (default "unknown"), optional actual
    # response model/provider/id reported by the upstream, optional diagnostics, usage,
    # optional timing, a stop reason (stop, length, toolUse, error, aborted), optional
    # error message, and timestamp.
    """A Pi-compatible assistant message with ordered content blocks."""

    role: Literal["assistant"] = "assistant"
    content: list[AssistantContent] = Field(default_factory=list)
    api: str = "unknown"
    provider: str = "unknown"
    model: str = "unknown"
    response_model: str | None = None
    response_provider: str | None = None
    response_id: str | None = None
    diagnostics: list[AssistantMessageDiagnostic] | None = None
    usage: Usage = Usage()
    timing: ResponseTiming | None = None
    stop_reason: StopReason = "stop"
    error_message: str | None = None
    timestamp: int = Field(default_factory=current_timestamp_ms)

    @model_validator(mode="before")
    @classmethod
    def _normalize_convenient_content(cls, value: object) -> object:
        # [FUNCTION DOC]
        # Pre-validation hook. For dictionary input: copy it, convert string content into a
        # single text block (or an empty list for an empty string), and replace a missing or
        # None usage with an empty usage record. Non-dictionary input is returned unchanged.
        # Output: the possibly-normalized input. No side effects (input is not mutated).
        """Accept a string only as a Python construction convenience.

        The stored model and serialized protocol are always block based. This
        keeps provider and test construction concise without creating a second
        message representation.
        """
        if not isinstance(value, dict):
            return value
        data = dict(value)
        content = data.get("content")
        if isinstance(content, str):
            data["content"] = [TextContent(text=content)] if content else []
        usage = data.get("usage")
        if usage is None:
            data["usage"] = Usage()
        return data

    @property
    def text(self) -> str:
        # [FUNCTION DOC]
        # Property: concatenation of all text blocks in order.
        return "".join(block.text for block in self.content if isinstance(block, TextContent))

    @property
    def thinking_text(self) -> str:
        # [FUNCTION DOC]
        # Property: concatenation of all thinking blocks in order.
        return "".join(
            block.thinking for block in self.content if isinstance(block, ThinkingContent)
        )

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        # [FUNCTION DOC]
        # Property: tuple of the tool call blocks in content order.
        return tuple(block for block in self.content if isinstance(block, ToolCall))


class ToolResultMessage(WireModel):
    # [CLASS DOC]
    # Transcript message carrying a tool's result: role "toolResult", the originating
    # tool call id and tool name, text/image content blocks, optional arbitrary JSON
    # details, optional names of tools newly made available by this result, error flag,
    # and timestamp.
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str
    tool_name: str
    content: list[ToolResultContent] = Field(default_factory=list)
    details: JSONValue = None
    added_tool_names: list[str] | None = None
    is_error: bool = False
    timestamp: int = Field(default_factory=current_timestamp_ms)

    @model_validator(mode="before")
    @classmethod
    def _normalize_convenient_content(cls, value: object) -> object:
        # [FUNCTION DOC]
        # Pre-validation hook: for dictionary input, converts string content into a single
        # text block (empty string becomes an empty list); other input passes through.
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
        # Property: concatenated visible text of the content blocks.
        return content_text(self.content)


class BashExecutionMessage(WireModel):
    # [CLASS DOC]
    # Session-only record of a user-run shell command: role "bashExecution", command,
    # captured output, optional exit code, cancelled and truncated flags, optional path
    # to the full output file, timestamp, and a flag to exclude it from model context.
    role: Literal["bashExecution"] = "bashExecution"
    command: str
    output: str
    exit_code: int | None = None
    cancelled: bool = False
    truncated: bool = False
    full_output_path: str | None = None
    timestamp: int = Field(default_factory=current_timestamp_ms)
    exclude_from_context: bool = False


class CustomMessage(WireModel):
    # [CLASS DOC]
    # Extension-defined message: role "custom", a custom type tag, string or block
    # content, whether it is displayed in the UI, optional JSON details, and timestamp.
    role: Literal["custom"] = "custom"
    custom_type: str
    content: UserContent
    display: bool = True
    details: JSONValue = None
    timestamp: int = Field(default_factory=current_timestamp_ms)

    @property
    def text(self) -> str:
        # [FUNCTION DOC]
        # Property: visible text of the content.
        return content_text(self.content)


class BranchSummaryMessage(WireModel):
    # [CLASS DOC]
    # Summary of an abandoned branch inserted when navigating the session tree: role
    # "branchSummary", summary text, the id of the entry it was branched from, timestamp.
    role: Literal["branchSummary"] = "branchSummary"
    summary: str
    from_id: str
    timestamp: int = Field(default_factory=current_timestamp_ms)


class CompactionSummaryMessage(WireModel):
    # [CLASS DOC]
    # Summary replacing compacted history: role "compactionSummary", summary text, the
    # token count before compaction, and timestamp.
    role: Literal["compactionSummary"] = "compactionSummary"
    summary: str
    tokens_before: int
    timestamp: int = Field(default_factory=current_timestamp_ms)


type AgentMessage = Annotated[
    UserMessage
    | AssistantMessage
    | ToolResultMessage
    | BashExecutionMessage
    | CustomMessage
    | BranchSummaryMessage
    | CompactionSummaryMessage,
    Field(discriminator="role"),
]


def assistant_content(
    text: str,
    tool_calls: list[ToolCall] | tuple[ToolCall, ...] = (),
) -> list[AssistantContent]:
    # [FUNCTION DOC]
    # Build ordered assistant content blocks: a text block first (only if the text is
    # non-empty) followed by the given tool calls. Returns a new list. Pure.
    """Build canonical ordered assistant blocks from parser accumulators."""
    blocks: list[AssistantContent] = [TextContent(text=text)] if text else []
    blocks.extend(tool_calls)
    return blocks


def content_text(content: str | list[Any]) -> str:
    # [FUNCTION DOC]
    # Return visible text from string content or a list of blocks (concatenating text
    # blocks, ignoring all others). Pure.
    """Return visible text from string or text/image content."""
    if isinstance(content, str):
        return content
    return "".join(block.text for block in content if isinstance(block, TextContent))


def message_to_user(message: AgentMessage) -> UserMessage:
    # [FUNCTION DOC]
    # Convert any agent message (e.g. custom, bash, summary) into a user message whose
    # content is that message's visible text, preserving the original timestamp.
    # Images in the source content are dropped. Pure.
    """Convert custom/session-only messages to provider-compatible user context."""
    return UserMessage(content=message_text(message), timestamp=message.timestamp)


def message_text(message: AgentMessage) -> str:
    # [FUNCTION DOC]
    # Return the visible text of any agent message: the text property for user,
    # assistant, tool result and custom messages; the summary for branch and compaction
    # summaries; the output for bash executions; empty string otherwise. Pure.
    """Return the user-visible text represented by an agent message."""
    if isinstance(message, (UserMessage, AssistantMessage, ToolResultMessage, CustomMessage)):
        return message.text
    if isinstance(message, (BranchSummaryMessage, CompactionSummaryMessage)):
        return message.summary
    if isinstance(message, BashExecutionMessage):
        return message.output
    return ""
