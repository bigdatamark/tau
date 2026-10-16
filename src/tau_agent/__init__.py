# [PACKAGE DOC]
# Public facade of the tau_agent package: the portable, Pi-compatible agent
# harness ("reusable agent brain") with no CLI, Textual or Rich dependencies.
# Behavior: re-exports agent lifecycle events (agent/turn/message/tool
# execution start, update and end events), the AgentHarness with its config,
# listener type, queued messages and simple cancellation token, the
# run_agent_loop function, message/content models and text helpers, session
# entry models and storage/state, tool contracts (AgentTool, results,
# cancellation, execution mode, executor, update callback) and JSON type aliases.
# The export list is computed dynamically as every module-level name not
# starting with an underscore, so it also includes imported submodule names
# that appear as globals.
# Side effects: importing the listed submodules. Raises nothing itself.
"""Portable Pi-compatible agent harness primitives for Tau."""

# ruff: noqa: F401 - this module intentionally defines the public facade

from tau_agent.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from tau_agent.harness import (
    AgentHarness,
    AgentHarnessConfig,
    EventListener,
    QueuedMessages,
    SimpleCancellationToken,
)
from tau_agent.loop import run_agent_loop
from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    BashExecutionMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    CustomMessage,
    ImageContent,
    ResponseTiming,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
    content_text,
    message_text,
)
from tau_agent.session import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    CustomMessageEntry,
    JsonlSessionStorage,
    LabelEntry,
    LeafEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionEntry,
    SessionInfoEntry,
    SessionState,
    ThinkingLevelChangeEntry,
)
from tau_agent.tools import (
    AgentTool,
    AgentToolResult,
    ToolCancellationToken,
    ToolExecutionMode,
    ToolExecutor,
    ToolUpdateCallback,
)
from tau_agent.types import JSONObject, JSONPrimitive, JSONValue

__all__ = [name for name in globals() if not name.startswith("_")]
