# [MODULE DOC]
# Approximate context-size accounting and compaction prompt building for Tau
# coding sessions. Provides: a character-based token estimator (about 4
# characters per token, plus fixed per-message and per-tool overheads); a usage
# estimate that is anchored on the latest provider-reported usage when available;
# the Pi-style auto-compaction threshold (context window minus a 16,384 token
# reserve); and the prompts/serializers used to ask a model to summarize older
# history (initial summary, incremental update of a previous summary, and
# turn-prefix summary). A compaction summary is stored as a user message whose
# text starts with "Previous conversation summary:" followed by a newline.
# All functions are pure; no I/O.
"""Approximate context-size estimation for Tau coding sessions."""

from __future__ import annotations

from dataclasses import dataclass

from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    ThinkingContent,
    ToolResultMessage,
    UserMessage,
    message_text,
)
from tau_agent.tools import AgentTool

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4
TOOL_OVERHEAD_TOKENS = 16
SUMMARY_MESSAGE_CHAR_LIMIT = 500
DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000
DEFAULT_COMPACTION_RESERVE_TOKENS = 16_384
DEFAULT_COMPACTION_KEEP_RECENT_TOKENS = 20_000
COMPACTION_SUMMARY_PREFIX = "Previous conversation summary:\n"

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation "
    "between a user and an AI coding assistant, then produce a structured summary "
    "following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the "
    "conversation. ONLY output the structured summary."
)

SUMMARIZATION_PROMPT = (
    "The messages above are a conversation to summarize. Create a structured context "
    "checkpoint summary that another LLM will use to continue the work.\n\n"
    "Use this EXACT format:\n\n"
    "## Goal\n"
    "[What is the user trying to accomplish? Can be multiple items if the session "
    "covers different tasks.]\n\n"
    "## Constraints & Preferences\n"
    "- [Any constraints, preferences, or requirements mentioned by user]\n"
    '- [Or "(none)" if none were mentioned]\n\n'
    "## Progress\n"
    "### Done\n"
    "- [x] [Completed tasks/changes]\n\n"
    "### In Progress\n"
    "- [ ] [Current work]\n\n"
    "### Blocked\n"
    "- [Issues preventing progress, if any]\n\n"
    "## Key Decisions\n"
    "- **[Decision]**: [Brief rationale]\n\n"
    "## Next Steps\n"
    "1. [Ordered list of what should happen next]\n\n"
    "## Critical Context\n"
    "- [Any data, examples, or references needed to continue]\n"
    '- [Or "(none)" if not applicable]\n\n'
    "Keep each section concise. Preserve exact file paths, function names, and error "
    "messages."
)

UPDATE_SUMMARIZATION_PROMPT = (
    "The messages above are NEW conversation messages to incorporate into the existing "
    "summary provided in <previous-summary> tags.\n\n"
    "Update the existing structured summary with new information. RULES:\n"
    "- PRESERVE all existing information from the previous summary\n"
    "- ADD new progress, decisions, and context from the new messages\n"
    '- UPDATE the Progress section: move items from "In Progress" to "Done" when '
    "completed\n"
    '- UPDATE "Next Steps" based on what was accomplished\n'
    "- PRESERVE exact file paths, function names, and error messages\n"
    "- If something is no longer relevant, you may remove it\n\n"
    "Use this EXACT format:\n\n"
    "## Goal\n"
    "[Preserve existing goals, add new ones if the task expanded]\n\n"
    "## Constraints & Preferences\n"
    "- [Preserve existing, add new ones discovered]\n\n"
    "## Progress\n"
    "### Done\n"
    "- [x] [Include previously done items AND newly completed items]\n\n"
    "### In Progress\n"
    "- [ ] [Current work - update based on progress]\n\n"
    "### Blocked\n"
    "- [Current blockers - remove if resolved]\n\n"
    "## Key Decisions\n"
    "- **[Decision]**: [Brief rationale] (preserve all previous, add new)\n\n"
    "## Next Steps\n"
    "1. [Update based on current state]\n\n"
    "## Critical Context\n"
    "- [Preserve important context, add new if needed]\n\n"
    "Keep each section concise. Preserve exact file paths, function names, and error "
    "messages."
)

TURN_PREFIX_SUMMARIZATION_PROMPT = (
    "This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) "
    "is retained.\n\n"
    "Summarize the prefix to provide context for the retained suffix:\n\n"
    "## Original Request\n"
    "[What did the user ask for in this turn?]\n\n"
    "## Early Progress\n"
    "- [Key decisions and work done in the prefix]\n\n"
    "## Context for Suffix\n"
    "- [Information needed to understand the retained recent work]\n\n"
    "Be concise. Focus on what's needed to understand the kept suffix."
)


@dataclass(frozen=True, slots=True)
class ContextUsageEstimate:
    # [CLASS DOC]
    # Immutable record of the context-size estimate for one provider request:
    # total tokens, the estimated system/message/tool components, counts of messages
    # and tools, provider-reported tokens (0 when the estimate is purely heuristic)
    # and trailing tokens (the heuristic part added on top of the provider anchor;
    # in the fallback it equals the whole total).
    """Best available context-size accounting for one provider request."""

    total_tokens: int
    system_tokens: int
    message_tokens: int
    tool_tokens: int
    message_count: int
    tool_count: int
    provider_tokens: int = 0
    trailing_tokens: int = 0

    @property
    def uses_provider_usage(self) -> bool:
        # [FUNCTION DOC]
        # Property: True when provider-reported tokens anchor the estimate (greater than
        # zero). No side effects.
        """Return whether a provider-reported usage block anchors this estimate."""
        return self.provider_tokens > 0


def estimate_text_tokens(text: str) -> int:
    # [FUNCTION DOC]
    # Estimate tokens for a string: 0 for empty text, otherwise the character count
    # divided by 4 rounded up (at least 1). Pure.
    """Return a deterministic rough token estimate for text."""
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def estimate_message_tokens(message: AgentMessage) -> int:
    # [FUNCTION DOC]
    # Estimate tokens for one agent message: a 4-token overhead plus its plain text;
    # for assistant messages also each thinking block's text and each tool call's
    # name plus the textual form of its argument mapping; for tool results also the
    # tool name. Pure; returns an integer.
    """Return a rough token estimate for one provider-neutral message."""
    tokens = MESSAGE_OVERHEAD_TOKENS + estimate_text_tokens(message_text(message))
    if isinstance(message, AssistantMessage):
        tokens += sum(
            estimate_text_tokens(block.thinking)
            for block in message.content
            if isinstance(block, ThinkingContent)
        )
        tokens += sum(
            estimate_text_tokens(call.name) + estimate_text_tokens(str(call.arguments))
            for call in message.tool_calls
        )
    elif isinstance(message, ToolResultMessage):
        tokens += estimate_text_tokens(message.tool_name)
    return tokens


def estimate_tool_tokens(tool: AgentTool) -> int:
    # [FUNCTION DOC]
    # Estimate tokens for a tool definition: a 16-token overhead plus name,
    # description and the textual form of its input schema. Pure.
    """Return a rough token estimate for one tool definition."""
    return (
        TOOL_OVERHEAD_TOKENS
        + estimate_text_tokens(tool.name)
        + estimate_text_tokens(tool.description)
        + estimate_text_tokens(str(tool.input_schema))
    )


def estimate_context_tokens(
    *,
    system: str,
    messages: tuple[AgentMessage, ...],
    tools: tuple[AgentTool, ...],
) -> int:
    # [FUNCTION DOC]
    # Convenience wrapper returning only the total from estimate_context_usage for
    # the given system prompt, messages and tools. Pure.
    """Return a rough estimate of the active provider context size."""
    return estimate_context_usage(system=system, messages=messages, tools=tools).total_tokens


def auto_compaction_threshold_for_context_window(context_window_tokens: int) -> int | None:
    # [FUNCTION DOC]
    # Return the token count at which automatic compaction should trigger: None for
    # non-positive windows, otherwise the window minus the 16,384 reserve, at least
    # 1. Pure.
    """Return Pi-style automatic compaction threshold for a model context window."""
    if context_window_tokens <= 0:
        return None
    return max(1, context_window_tokens - DEFAULT_COMPACTION_RESERVE_TOKENS)


def provider_context_tokens(message: AssistantMessage) -> int:
    # [FUNCTION DOC]
    # Return the context size reported by the provider for an assistant response:
    # its usage total if non-zero, else input + output + cache read + cache write.
    # Pure.
    """Return the provider-reported context represented by an assistant response."""
    usage = message.usage
    return usage.total_tokens or (usage.input + usage.output + usage.cache_read + usage.cache_write)


def _last_applicable_provider_usage(
    messages: tuple[AgentMessage, ...],
) -> tuple[int, int] | None:
    # [FUNCTION DOC]
    # HELPER: Scan messages in order and return the index and provider token count
    # of the last assistant message whose usage is still valid for the current
    # prefix, or None. A message qualifies when its stop reason is not aborted or
    # error, its reported tokens are positive, and its timestamp is not older than
    # any earlier message's timestamp (an older timestamp means history was
    # rewritten, e.g. by compaction, before it). Pure.
    """Find the latest valid usage block that still describes the active prefix.

    A newer prefix message can be inserted by compaction or history rewriting. In that
    case an older assistant's usage describes the pre-rewrite context and must not be
    reused.
    """
    latest_prefix_timestamp = -1
    usage_info: tuple[int, int] | None = None
    for index, message in enumerate(messages):
        if isinstance(message, AssistantMessage):
            tokens = provider_context_tokens(message)
            if (
                message.timestamp >= latest_prefix_timestamp
                and message.stop_reason not in {"aborted", "error"}
                and tokens > 0
            ):
                usage_info = (index, tokens)
        latest_prefix_timestamp = max(latest_prefix_timestamp, message.timestamp)
    return usage_info


def estimate_context_usage(
    *,
    system: str,
    messages: tuple[AgentMessage, ...],
    tools: tuple[AgentTool, ...],
) -> ContextUsageEstimate:
    # [FUNCTION DOC]
    # Produce a ContextUsageEstimate. If an applicable provider usage exists, the
    # total is that provider count plus heuristic estimates of the messages after
    # that assistant response and of tools whose names were added by tool results
    # in those trailing messages; system tokens are reported as 0. Otherwise every
    # component (system text, all messages, all tools) is estimated heuristically.
    # Message and tool counts always cover all inputs. Pure.
    """Return provider-anchored context accounting with a deterministic fallback.

    Provider usage is authoritative for the prefix represented by the latest successful
    assistant response. Only messages and dynamically added tools after that response
    are estimated. Without applicable usage, the whole request uses the character-based
    fallback.
    """
    usage_info = _last_applicable_provider_usage(messages)
    if usage_info is not None:
        usage_index, provider_tokens = usage_info
        trailing_messages = messages[usage_index + 1 :]
        trailing_message_tokens = sum(
            estimate_message_tokens(message) for message in trailing_messages
        )
        added_tool_names = {
            name
            for message in trailing_messages
            if isinstance(message, ToolResultMessage) and message.added_tool_names
            for name in message.added_tool_names
        }
        added_tool_tokens = sum(
            estimate_tool_tokens(tool) for tool in tools if tool.name in added_tool_names
        )
        trailing_tokens = trailing_message_tokens + added_tool_tokens
        return ContextUsageEstimate(
            total_tokens=provider_tokens + trailing_tokens,
            system_tokens=0,
            message_tokens=trailing_message_tokens,
            tool_tokens=added_tool_tokens,
            message_count=len(messages),
            tool_count=len(tools),
            provider_tokens=provider_tokens,
            trailing_tokens=trailing_tokens,
        )

    system_tokens = estimate_text_tokens(system)
    message_tokens = sum(estimate_message_tokens(message) for message in messages)
    tool_tokens = sum(estimate_tool_tokens(tool) for tool in tools)
    return ContextUsageEstimate(
        total_tokens=system_tokens + message_tokens + tool_tokens,
        system_tokens=system_tokens,
        message_tokens=message_tokens,
        tool_tokens=tool_tokens,
        message_count=len(messages),
        tool_count=len(tools),
        trailing_tokens=system_tokens + message_tokens + tool_tokens,
    )


def summarize_messages_for_compaction(messages: tuple[AgentMessage, ...]) -> str:
    # [FUNCTION DOC]
    # Build a deterministic, non-model fallback summary: "No prior messages." when
    # empty, else a header line with the count followed by one numbered line per
    # message with its role ("tool" for tool results) and a whitespace-collapsed
    # text truncated to 500 characters. Pure.
    """Build a deterministic compact summary from provider-neutral messages."""
    if not messages:
        return "No prior messages."
    lines = [f"Automatically compacted {len(messages)} prior message(s)."]
    for index, message in enumerate(messages, start=1):
        role = "tool" if isinstance(message, ToolResultMessage) else message.role
        lines.append(f"{index}. {role}: {_message_text(message)}")
    return "\n".join(lines)


def build_compaction_summary_prompt(
    messages: tuple[AgentMessage, ...],
    *,
    custom_instructions: str | None = None,
) -> str:
    # [FUNCTION DOC]
    # Build the user prompt sent to the summarization model. Splits off a leading
    # previous compaction summary message, serializes the remaining messages inside
    # conversation tags, then includes the previous summary in previous-summary tags
    # and uses the update prompt when one existed, else the initial summarization
    # prompt. Non-blank custom instructions are appended as "Additional focus: ...".
    # Output: the prompt string. Pure.
    """Build the model prompt Tau uses to summarize compacted history."""
    previous_summary, new_messages = _split_previous_compaction_summary(messages)
    conversation = serialize_messages_for_compaction(new_messages)
    prompt = f"<conversation>\n{conversation}\n</conversation>\n\n"
    base_prompt = (
        UPDATE_SUMMARIZATION_PROMPT if previous_summary is not None else SUMMARIZATION_PROMPT
    )

    if previous_summary is not None:
        prompt += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"

    instructions = custom_instructions.strip() if custom_instructions is not None else ""
    if instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {instructions}"

    return f"{prompt}{base_prompt}"


def serialize_messages_for_compaction(messages: tuple[AgentMessage, ...]) -> str:
    # [FUNCTION DOC]
    # Serialize messages into pseudo-XML blocks for the summarizer: one message tag
    # per message with index and role attributes (tool results also get tool name
    # and a lowercase error flag), the message text if non-empty, and for assistant
    # tool calls a tool-calls section listing each name and argument mapping.
    # Returns "(no new messages)" for an empty input. Pure.
    """Serialize provider-neutral messages for the compaction summarizer."""
    if not messages:
        return "(no new messages)"

    lines: list[str] = []
    for index, message in enumerate(messages, start=1):
        attributes = f"index={index} role={message.role}"
        if isinstance(message, ToolResultMessage):
            attributes += f" name={message.tool_name} error={str(message.is_error).lower()}"
        lines.append(f"<message {attributes}>")
        text = message_text(message)
        if text:
            lines.append(text)
        if isinstance(message, AssistantMessage) and message.tool_calls:
            lines.append("<tool-calls>")
            for call in message.tool_calls:
                lines.append(f"- {call.name}: {call.arguments}")
            lines.append("</tool-calls>")
        lines.append("</message>")
    return "\n".join(lines)


def _message_text(message: AgentMessage) -> str:
    # [FUNCTION DOC]
    # HELPER: One-line summary text of a message for the fallback summary: assistant
    # text plus a bracketed list of tool call names, or for tool results the tool
    # name, "ok"/"failed" and the text; then collapsed and truncated. Pure.
    text = message_text(message)
    if isinstance(message, AssistantMessage) and message.tool_calls:
        names = ", ".join(call.name for call in message.tool_calls)
        text = f"{text} [tool calls: {names}]"
    elif isinstance(message, ToolResultMessage):
        status = "failed" if message.is_error else "ok"
        text = f"{message.tool_name} {status}: {text}"
    return _truncate_summary_text(text)


def _truncate_summary_text(text: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Collapse all whitespace runs to single spaces and, if longer than 500
    # characters, cut to 497, strip trailing spaces and append "...". Pure.
    collapsed = " ".join(text.split())
    if len(collapsed) <= SUMMARY_MESSAGE_CHAR_LIMIT:
        return collapsed
    return collapsed[: SUMMARY_MESSAGE_CHAR_LIMIT - 3].rstrip() + "..."


def _split_previous_compaction_summary(
    messages: tuple[AgentMessage, ...],
) -> tuple[str | None, tuple[AgentMessage, ...]]:
    # [FUNCTION DOC]
    # HELPER: If the first message is a user message whose text starts with the
    # compaction summary prefix, return the summary text (prefix removed) and the
    # remaining messages; otherwise return None and the messages unchanged. Pure.
    if not messages:
        return None, messages

    first = messages[0]
    if not isinstance(first, UserMessage):
        return None, messages
    text = message_text(first)
    if not text.startswith(COMPACTION_SUMMARY_PREFIX):
        return None, messages

    return text.removeprefix(COMPACTION_SUMMARY_PREFIX), messages[1:]
