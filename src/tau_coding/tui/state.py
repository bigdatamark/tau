# [MODULE DOC]
# Frontend-neutral display state for Tau's Textual TUI. The TuiEventAdapter
# mutates a TuiState from session events; Textual widgets render it. Keeps
# transcript presentation logic (tool call batching/grouping, previews,
# summaries, skill detection, elapsed timers) out of the agent harness.
# Constants: preview limits (8 result lines, 32 patch lines, 2000 chars, 120
# terminal output lines), a 1 second threshold before showing a live tool
# timer, batchable tool names (bash, edit, read, write), groupable file tools
# (edit, read, write), file groups that show per-file results (edit, write) and
# a 160 character limit for fallback argument rendering.
# Also defines GroupedToolCall, ChatItem, TuiState and formatting functions
# (some public, used by other TUI modules). Side effects: none at import.
"""Display state for Tau's Textual TUI."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    CustomMessage,
    TextContent,
    ThinkingContent,
    ToolResultMessage,
    UserMessage,
)
from tau_agent.tools import AgentToolResult, ToolCall
from tau_agent.types import JSONValue
from tau_coding.extensions.api import CustomMessageMarkup, ToolCallMarkup, ToolResultMarkup
from tau_coding.skills import Skill, parse_skill_invocation
from tau_coding.system_prompt import SystemPromptSource
from tau_coding.tui.themes import TranscriptRole

ChatItemRole = TranscriptRole
TOOL_RESULT_PREVIEW_LINES = 8
TOOL_PATCH_PREVIEW_LINES = 32
TOOL_RESULT_PREVIEW_CHARS = 2_000
TERMINAL_COMMAND_OUTPUT_PREVIEW_LINES = 120
# Show live elapsed time on an executing tool row once it stops being instant;
# quick reads/edits never flash a "(0s)".
TOOL_TIMER_MIN_SECONDS = 1.0
BATCHABLE_TOOL_NAMES = frozenset({"bash", "edit", "read", "write"})
GROUPABLE_FILE_TOOL_NAMES = frozenset({"edit", "read", "write"})
RESULTFUL_FILE_GROUP_NAMES = frozenset({"edit", "write"})


@dataclass(slots=True)
class GroupedToolCall:
    # [CLASS DOC]
    # Mutable, slotted dataclass for one underlying tool call folded into a
    # grouped file row (e.g. "Read 3 files"): call id, tool name, arguments,
    # formatted call text, optional formatted result text, raw result, live update
    # text and monotonic start time (none once finished).
    """One underlying call represented by a grouped transcript row."""

    tool_call_id: str
    tool_name: str
    tool_arguments: dict[str, JSONValue]
    text: str
    tool_result_text: str | None = None
    tool_result: AgentToolResult | None = None
    update_text: str | None = None
    started_at: float | None = None


@dataclass(slots=True)
class ChatItem:
    # [CLASS DOC]
    # Mutable, slotted dataclass for one rendered transcript row. Fields: role
    # (user, assistant, thinking, tool, skill, custom, error, status, summaries...),
    # text, and optional tool metadata (call id, formatted result text, raw result
    # for lazy render_result, live update text, tool name and arguments, monotonic
    # start time while running, presentation batch id, whether a later
    # edit/write-only response may continue this row, grouped member calls, nested
    # batch rows), a flag to always show the result, custom message type and
    # details, system prompt flag and sources, and an optional highlight (alert or
    # update).
    """One rendered item in the TUI transcript."""

    role: ChatItemRole
    text: str
    tool_call_id: str | None = None
    tool_result_text: str | None = None
    # The raw result object, kept alongside the formatted text so the tool's
    # `render_result` (resolved lazily, like `render_call`) can format it.
    tool_result: AgentToolResult | None = None
    update_text: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, JSONValue] | None = None
    started_at: float | None = None
    tool_batch_id: int | None = None
    allows_file_mutation_continuation: bool = False
    grouped_tool_calls: list[GroupedToolCall] | None = None
    tool_batch_items: list[ChatItem] | None = None
    always_show_tool_result: bool = False
    custom_type: str | None = None
    details: dict[str, JSONValue] | None = None
    system_prompt: bool = False
    system_prompt_sources: tuple[SystemPromptSource, ...] | None = None
    highlight: Literal["alert", "update"] | None = None


@dataclass(slots=True)
class TuiState:
    # [CLASS DOC]
    # Mutable, slotted dataclass holding the whole TUI display model: transcript
    # items, streaming assistant text buffer, running flag, error text, toggles for
    # showing tool results and thinking, queued steering/follow-up texts, loaded
    # skills (to recognise skill reads), optional extension renderers for custom
    # messages, tool calls and tool results, plus private indexes from tool call id
    # to its top-level item, its grouped member and its batch row, and a counter
    # for presentation batch ids.
    """Mutable display state for the interactive TUI."""

    items: list[ChatItem] = field(default_factory=list)
    assistant_buffer: str = ""
    running: bool = False
    error: str | None = None
    show_tool_results: bool = False
    show_thinking: bool = False
    queued_steering: tuple[str, ...] = ()
    queued_follow_up: tuple[str, ...] = ()
    skills: tuple[Skill, ...] = ()
    custom_renderer: CustomMessageMarkup | None = None
    tool_call_renderer: ToolCallMarkup | None = None
    tool_result_renderer: ToolResultMarkup | None = None
    _tool_items_by_call_id: dict[str, ChatItem] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _grouped_calls_by_call_id: dict[str, GroupedToolCall] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _batched_items_by_call_id: dict[str, ChatItem] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _next_tool_batch_id: int = field(default=0, init=False, repr=False, compare=False)

    def add_item(
        self,
        role: ChatItemRole,
        text: str,
        *,
        tool_call_id: str | None = None,
        tool_result_text: str | None = None,
        always_show_tool_result: bool = False,
        custom_type: str | None = None,
        details: dict[str, JSONValue] | None = None,
        system_prompt: bool = False,
        system_prompt_sources: tuple[SystemPromptSource, ...] | None = None,
        highlight: Literal["alert", "update"] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Appends a new ChatItem built from the given role, text and optional fields.
        # When a call id is given and role is "tool" or "skill" it is indexed for O(1)
        # lookup by later updates/results. Returns nothing; mutates items.
        """Append a transcript item."""
        item = ChatItem(
            role=role,
            text=text,
            tool_call_id=tool_call_id,
            tool_result_text=tool_result_text,
            always_show_tool_result=always_show_tool_result,
            custom_type=custom_type,
            details=details,
            system_prompt=system_prompt,
            system_prompt_sources=system_prompt_sources,
            highlight=highlight,
        )
        self.items.append(item)
        if tool_call_id is not None and role in {"tool", "skill"}:
            self._tool_items_by_call_id[tool_call_id] = item

    def resolve_custom_markup(self, item: ChatItem, *, expanded: bool) -> str | None:
        # [FUNCTION DOC]
        # Returns markup for a "custom" item by calling the installed custom renderer
        # with the custom type, text, details and expanded flag; none when the item is
        # not custom, has no type, or no renderer is installed (or the renderer returns
        # none). Renderer exceptions propagate.
        """Render a custom item's markup via the installed resolver, or ``None``.

        Returns ``None`` when the item is not custom, no resolver is installed,
        or the resolver declines/fails to render (the caller then falls back to
        the raw ``item.text``).
        """
        if item.role != "custom" or item.custom_type is None or self.custom_renderer is None:
            return None
        return self.custom_renderer(item.custom_type, item.text, item.details, expanded)

    def resolve_tool_invocation(self, item: ChatItem, *, expanded: bool = False) -> str | None:
        # [FUNCTION DOC]
        # Computes the display line for a tool item at render time; none means "use
        # item text".
        # - Non-tool items: none.
        # - Batch rows: none when collapsed; when expanded, the newline-joined
        #   resolutions (or texts) of each nested row.
        # - Grouped file rows: none when collapsed; when expanded, one block per member
        #   (custom renderer output or member text), with each member's result
        #   appended for edit/write groups, joined by blank lines for edit/write or
        #   newlines for read.
        # - Single calls: custom tool call renderer output if any; else, for expanded
        #   bash, the item text followed by the exact "$ command" line. If still
        #   running for at least the timer threshold, appends the elapsed time in
        #   parentheses (using the item text when no line).
        # Side effects: reads the monotonic clock; renderer errors propagate.
        """Render a tool item's invocation via the installed resolver, or ``None``.

        Resolved lazily at render time (like custom markup) so tool calls
        restored before the extension runtime connects still pick up their
        tool's `render_call` on the next redraw. Expanded built-in bash calls
        recover the exact command from their retained arguments. ``None`` means
        "no renderer" and the caller falls back to the generic ``item.text``.
        """
        if item.role != "tool":
            return None
        if item.tool_batch_items is not None:
            if not expanded:
                return None
            return "\n".join(
                self.resolve_tool_invocation(row, expanded=True) or row.text
                for row in item.tool_batch_items
            )
        if item.grouped_tool_calls is not None:
            if not expanded:
                return None
            blocks: list[str] = []
            for member in item.grouped_tool_calls:
                rendered_line = None
                if self.tool_call_renderer is not None:
                    rendered_line = self.tool_call_renderer(member.tool_name, member.tool_arguments)
                block = rendered_line if rendered_line is not None else member.text
                if (
                    item.tool_name in RESULTFUL_FILE_GROUP_NAMES
                    and member.tool_result_text is not None
                ):
                    block = f"{block}\n\n{member.tool_result_text}"
                blocks.append(block)
            separator = "\n\n" if item.tool_name in RESULTFUL_FILE_GROUP_NAMES else "\n"
            return separator.join(blocks)
        line: str | None = None
        if item.tool_name is not None and self.tool_call_renderer is not None:
            line = self.tool_call_renderer(item.tool_name, item.tool_arguments or {})
        if line is None and expanded and item.tool_name == "bash":
            exact_command = format_tool_call_invocation(
                ToolCall(
                    id=item.tool_call_id or "display-call",
                    name=item.tool_name,
                    arguments=item.tool_arguments or {},
                ),
                expanded=True,
            )
            line = f"{item.text}\n{exact_command}"
        if item.tool_result_text is None and item.started_at is not None:
            elapsed = time.monotonic() - item.started_at
            if elapsed >= TOOL_TIMER_MIN_SECONDS:
                return f"{line if line is not None else item.text} ({format_elapsed(elapsed)})"
        return line

    def resolve_tool_result(self, item: ChatItem, *, expanded: bool) -> str | None:
        # [FUNCTION DOC]
        # Returns the tool's custom render_result output for a single, finished tool
        # item (not batch/group) with a raw result, a tool name and an installed
        # result renderer; none otherwise.
        """Render a tool item's result via its tool's `render_result`, or ``None``.

        Resolved lazily at render time (like `resolve_tool_invocation`) so
        results restored before the extension runtime connects still pick up
        their tool's `render_result` on the next redraw. ``None`` means "no
        renderer" and the caller falls back to the generic result block.
        """
        if (
            item.role != "tool"
            or item.tool_batch_items is not None
            or item.grouped_tool_calls is not None
            or item.tool_result is None
            or self.tool_result_renderer is None
        ):
            return None
        if item.tool_name is None:
            return None
        return self.tool_result_renderer(item.tool_name, item.tool_result, expanded)

    def new_tool_batch_id(self) -> int:
        # [FUNCTION DOC]
        # Increments and returns the presentation batch counter (ids start at 1).
        """Return a presentation-only id for calls from one assistant message."""
        self._next_tool_batch_id += 1
        return self._next_tool_batch_id

    def add_tool_call(
        self,
        tool_call: ToolCall,
        *,
        batch_id: int | None = None,
        allows_file_mutation_continuation: bool = False,
    ) -> ChatItem:
        # [FUNCTION DOC]
        # Adds a tool call row, merging it into prior rows when possible.
        # Order of rules: a read of a loaded skill's file becomes a "skill" row
        # "Loading skill: <name>"; else, if allowed, an edit/write call continuing a
        # completed same-tool edit/write-only previous row is appended to that row
        # (adopting the new batch id); else, a batchable call with the same batch id as
        # the previous tool row is appended to it; else a new tool row is created and
        # indexed. Custom-rendered calls are never merged.
        # Outputs: the item that now represents the call. Mutates state.
        """Append a tool call, batching adjacent calls for compact presentation."""
        skill_name = self._read_skill_name(tool_call)
        if skill_name is not None:
            self.add_item(
                "skill",
                f"Loading skill: {skill_name}",
                tool_call_id=tool_call.id,
            )
            return self.items[-1]
        if self._can_append_file_mutation_continuation(
            tool_call,
            batch_id=batch_id,
            allowed=allows_file_mutation_continuation,
        ):
            item = self.items[-1]
            item.tool_batch_id = batch_id
            self._append_batched_tool_call(item, tool_call)
            return item
        if self._can_append_tool_batch(tool_call, batch_id=batch_id):
            item = self.items[-1]
            self._append_batched_tool_call(item, tool_call)
            return item
        item = self._new_tool_item(
            tool_call,
            batch_id=batch_id,
            allows_file_mutation_continuation=allows_file_mutation_continuation,
        )
        self.items.append(item)
        self._tool_items_by_call_id[tool_call.id] = item
        return item

    def _new_tool_item(
        self,
        tool_call: ToolCall,
        *,
        batch_id: int | None,
        allows_file_mutation_continuation: bool = False,
    ) -> ChatItem:
        # [FUNCTION DOC]
        # HELPER: Builds (does not append) a running tool ChatItem with formatted call
        # text, call id/name/arguments, the current monotonic start time, batch id and
        # continuation flag.
        return ChatItem(
            role="tool",
            text=format_tool_call_block(tool_call),
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            tool_arguments=tool_call.arguments,
            started_at=time.monotonic(),
            tool_batch_id=batch_id,
            allows_file_mutation_continuation=allows_file_mutation_continuation,
        )

    def _can_append_file_mutation_continuation(
        self,
        tool_call: ToolCall,
        *,
        batch_id: int | None,
        allowed: bool,
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: True when continuation is allowed, a batch id exists, the call is
        # edit/write, and the last item is a tool row allowing continuation with the
        # same tool name, not a batch, already finished, and neither it nor the new
        # call has custom call rendering.
        """Group completed edit/write-only continuations across response boundaries."""
        if (
            not allowed
            or batch_id is None
            or tool_call.name not in RESULTFUL_FILE_GROUP_NAMES
            or not self.items
        ):
            return False
        previous = self.items[-1]
        if (
            previous.role != "tool"
            or not previous.allows_file_mutation_continuation
            or previous.tool_name != tool_call.name
            or previous.tool_batch_items is not None
            or previous.tool_result_text is None
        ):
            return False
        if self._has_custom_call_rendering(tool_call.name, tool_call.arguments):
            return False
        return not self._has_custom_call_rendering(
            previous.tool_name,
            previous.tool_arguments or {},
        )

    def _can_append_tool_batch(self, tool_call: ToolCall, *, batch_id: int | None) -> bool:
        # [FUNCTION DOC]
        # HELPER: True when the call is batchable with a batch id, the last item is a
        # tool row with the same batch id, its last nested row (or itself) is a
        # batchable tool, and neither side has custom call rendering.
        if batch_id is None or not self.items or tool_call.name not in BATCHABLE_TOOL_NAMES:
            return False
        previous = self.items[-1]
        if previous.role != "tool" or previous.tool_batch_id != batch_id:
            return False
        if self._has_custom_call_rendering(tool_call.name, tool_call.arguments):
            return False
        previous_row = (
            previous.tool_batch_items[-1] if previous.tool_batch_items is not None else previous
        )
        if previous_row.tool_name not in BATCHABLE_TOOL_NAMES:
            return False
        return not self._has_custom_call_rendering(
            previous_row.tool_name,
            previous_row.tool_arguments or {},
        )

    def _has_custom_call_rendering(
        self,
        tool_name: str | None,
        arguments: dict[str, JSONValue],
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: True when a tool call renderer is installed and returns non-none for
        # the tool name and arguments (invokes the renderer).
        if tool_name is None or self.tool_call_renderer is None:
            return False
        return self.tool_call_renderer(tool_name, arguments) is not None

    def _append_batched_tool_call(self, item: ChatItem, tool_call: ToolCall) -> None:
        # [FUNCTION DOC]
        # HELPER: Merges a call into an existing row. If the row is a plain (unbatched)
        # groupable file call of the same tool, it becomes/extends a file group.
        # Otherwise the row is converted into a batch on first use (its current
        # content moved into a first nested row, raw result and group cleared on the
        # container, call ids re-indexed to that nested row); the call then extends
        # the last nested row's file group when same groupable tool, or becomes a new
        # nested row. Updates indexes and refreshes the batch summary.
        if (
            item.tool_batch_items is None
            and item.tool_name in GROUPABLE_FILE_TOOL_NAMES
            and tool_call.name == item.tool_name
        ):
            self._append_grouped_file_call(item, tool_call)
            return
        if item.tool_batch_items is None:
            first = ChatItem(
                role="tool",
                text=item.text,
                tool_call_id=item.tool_call_id,
                tool_result_text=item.tool_result_text,
                tool_result=item.tool_result,
                update_text=item.update_text,
                tool_name=item.tool_name,
                tool_arguments=item.tool_arguments,
                started_at=item.started_at,
                tool_batch_id=item.tool_batch_id,
                allows_file_mutation_continuation=item.allows_file_mutation_continuation,
                grouped_tool_calls=item.grouped_tool_calls,
            )
            item.tool_batch_items = [first]
            item.grouped_tool_calls = None
            item.tool_result = None
            for call_id in self._tool_call_ids(first):
                self._batched_items_by_call_id[call_id] = first
        last = item.tool_batch_items[-1]
        if last.tool_name in GROUPABLE_FILE_TOOL_NAMES and tool_call.name == last.tool_name:
            self._append_grouped_file_call(last, tool_call)
            row = last
        else:
            row = self._new_tool_item(tool_call, batch_id=item.tool_batch_id)
            item.tool_batch_items.append(row)
        self._tool_items_by_call_id[tool_call.id] = item
        self._batched_items_by_call_id[tool_call.id] = row
        self._refresh_tool_batch(item)

    def _tool_call_ids(self, item: ChatItem) -> list[str]:
        # [FUNCTION DOC]
        # HELPER: Call ids represented by a row: all group member ids, or its own id
        # (empty list if none).
        if item.grouped_tool_calls is not None:
            return [member.tool_call_id for member in item.grouped_tool_calls]
        return [item.tool_call_id] if item.tool_call_id is not None else []

    def _append_grouped_file_call(self, item: ChatItem, tool_call: ToolCall) -> None:
        # [FUNCTION DOC]
        # HELPER: Converts a row into a file group on first use (its existing call
        # becomes the first member, defaulting the id to "display-call" and tool to
        # "read" when missing), appends a new running member for the call, indexes it
        # and refreshes the group text.
        if item.grouped_tool_calls is None:
            first = GroupedToolCall(
                tool_call_id=item.tool_call_id or "display-call",
                tool_name=item.tool_name or "read",
                tool_arguments=item.tool_arguments or {},
                text=format_tool_call_block(
                    ToolCall(
                        id=item.tool_call_id or "display-call",
                        name=item.tool_name or "read",
                        arguments=item.tool_arguments or {},
                    )
                ),
                tool_result_text=item.tool_result_text,
                tool_result=item.tool_result,
                update_text=item.update_text,
                started_at=item.started_at,
            )
            item.grouped_tool_calls = [first]
            self._grouped_calls_by_call_id[first.tool_call_id] = first
        member = GroupedToolCall(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            tool_arguments=tool_call.arguments,
            text=format_tool_call_block(tool_call),
            started_at=time.monotonic(),
        )
        item.grouped_tool_calls.append(member)
        self._tool_items_by_call_id[tool_call.id] = item
        self._grouped_calls_by_call_id[tool_call.id] = member
        self._refresh_tool_group(item)

    def _refresh_tool_batch(self, item: ChatItem) -> None:
        # [FUNCTION DOC]
        # HELPER: Recomputes a batch container: text = nested row texts joined by
        # newlines; result text "✗ tool batch" if any row result starts with ✗, "✓
        # tool batch" if no row is still running, else none; clears update text; start
        # time = first running row's start time.
        rows = item.tool_batch_items or []
        item.text = "\n".join(row.text for row in rows)
        pending = [row for row in rows if row.started_at is not None]
        failures = [
            row
            for row in rows
            if row.tool_result_text is not None and row.tool_result_text.startswith("✗")
        ]
        if failures:
            item.tool_result_text = "✗ tool batch"
        elif not pending:
            item.tool_result_text = "✓ tool batch"
        else:
            item.tool_result_text = None
        item.update_text = None
        item.started_at = next((row.started_at for row in pending), None)

    def _refresh_tool_group(self, item: ChatItem) -> None:
        # [FUNCTION DOC]
        # HELPER: Recomputes a file group row: headline "→ Reading/Editing/Writing N
        # files" with "· k/N complete" while running, or "→ Read/Edited/Written N
        # files" with "· k failed" when done, followed by an indented path list
        # ("[unknown path]" when missing). Result text "<status> <action> group" with
        # ✗/✓/… once any member finished; update text = first member update; start time
        # = first unfinished member's start.
        members = item.grouped_tool_calls or []
        completed = [member for member in members if member.tool_result_text is not None]
        failures = [
            member
            for member in completed
            if member.tool_result_text is not None and member.tool_result_text.startswith("✗")
        ]
        paths = [
            _string_argument(member.tool_arguments, "path") or "[unknown path]"
            for member in members
        ]
        path_list = "\n".join(f"  - {path}" for path in paths)
        all_complete = len(completed) == len(members)
        action = item.tool_name if item.tool_name in GROUPABLE_FILE_TOOL_NAMES else "read"
        running_verb = {"edit": "Editing", "read": "Reading", "write": "Writing"}[action]
        completed_verb = {"edit": "Edited", "read": "Read", "write": "Written"}[action]
        if not all_complete:
            progress = f" · {len(completed)}/{len(members)} complete" if completed else ""
            headline = f"→ {running_verb} {len(members)} files{progress}"
        else:
            failure = f" · {len(failures)} failed" if failures else ""
            headline = f"→ {completed_verb} {len(members)} files{failure}"
        item.text = f"{headline}\n{path_list}"
        if completed:
            status = "✗" if failures else ("✓" if all_complete else "…")
            item.tool_result_text = f"{status} {action} group"
        else:
            item.tool_result_text = None
        item.update_text = next(
            (member.update_text for member in members if member.update_text),
            None,
        )
        item.started_at = next(
            (member.started_at for member in members if member.tool_result_text is None),
            None,
        )

    def add_user_message(
        self,
        content: str,
        *,
        custom_type: str | None = None,
        details: dict[str, JSONValue] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Appends a user-side message with presentation compaction: with a custom type
        # -> "custom" item; branch summary wrapper text -> collapsed "branch_summary"
        # item holding the summary; "Previous conversation summary:" text -> collapsed
        # "compaction_summary" item; skill invocation text -> "Using skill: <name>"
        # skill row plus a user row for any additional instructions; otherwise a plain
        # "user" item.
        """Append a user-authored message, compacting skill and summary messages.

        A message carrying ``custom_type`` is stored as a ``"custom"`` item so
        the transcript can render it through a registered custom renderer; the
        raw ``content`` is retained as the fallback and LLM-context text.
        """
        if custom_type is not None:
            self.add_item("custom", content, custom_type=custom_type, details=details)
            return

        branch_summary = _parse_branch_summary_message(content)
        if branch_summary is not None:
            self.add_item(
                "branch_summary",
                "Branch summary (Ctrl+O to expand)",
                tool_result_text=branch_summary,
            )
            return

        compaction_summary = _parse_compaction_summary_message(content)
        if compaction_summary is not None:
            self.add_item(
                "compaction_summary",
                "Compaction summary (Ctrl+O to expand)",
                tool_result_text=compaction_summary,
            )
            return

        skill_invocation = parse_skill_invocation(content)
        if skill_invocation is None:
            self.add_item("user", content)
            return
        self.add_item("skill", f"Using skill: {skill_invocation.name}")
        if skill_invocation.additional_instructions:
            self.add_item("user", skill_invocation.additional_instructions)

    def add_thinking_delta(self, delta: str) -> None:
        # [FUNCTION DOC]
        # Appends a thinking fragment to the last item when it is a thinking row,
        # otherwise starts a new thinking row.
        """Append a thinking/reasoning fragment to the current thinking block."""
        if self.items and self.items[-1].role == "thinking":
            self.items[-1].text += delta
            return
        self.add_item("thinking", delta)

    def find_tool_item(self, tool_call_id: str) -> ChatItem | None:
        # [FUNCTION DOC]
        # Returns the top-level item indexed for a tool call id, or none.
        """Return the transcript item for a tool call id in O(1)."""
        return self._tool_items_by_call_id.get(tool_call_id)

    def record_tool_update(self, tool_call_id: str, message: str) -> ChatItem | None:
        # [FUNCTION DOC]
        # Attaches live progress text to a still-running call (group member or row),
        # refreshing group/batch summaries. Returns the top-level item, or none when
        # the call is unknown or already finished (update dropped).
        """Attach live progress to its pending tool call; drop orphan updates."""
        item = self.find_tool_item(tool_call_id)
        if item is None:
            return None
        row = self._batched_items_by_call_id.get(tool_call_id, item)
        member = self._grouped_calls_by_call_id.get(tool_call_id)
        if member is not None:
            if member.tool_result_text is not None:
                return None
            member.update_text = message
            self._refresh_tool_group(row)
        else:
            if row.tool_result_text is not None:
                return None
            row.update_text = message
        if item.tool_batch_items is not None:
            self._refresh_tool_batch(item)
        return item

    def record_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: AgentToolResult,
        is_error: bool,
    ) -> None:
        # [FUNCTION DOC]
        # Formats the result block (status, preview, edit patch) and attaches it, the
        # raw result, cleared update text and cleared start time to the matching
        # member/row, refreshing group and batch summaries. For an unknown call id it
        # appends a standalone tool row with a "✓/✗ name" summary and indexes it.
        # Returns nothing; mutates state.
        """Attach a Pi-compatible tool result to its matching call."""
        result_text = format_tool_result_block(
            name=tool_name,
            ok=not is_error,
            content=result.text,
            data=result.details if isinstance(result.details, dict) else None,
        )
        item = self.find_tool_item(tool_call_id)
        if item is not None:
            row = self._batched_items_by_call_id.get(tool_call_id, item)
            member = self._grouped_calls_by_call_id.get(tool_call_id)
            if member is not None:
                member.tool_result_text = result_text
                member.tool_result = result
                member.update_text = None
                member.started_at = None
                self._refresh_tool_group(row)
            else:
                row.tool_result_text = result_text
                row.tool_result = result
                row.update_text = None
                row.started_at = None
            if item.tool_batch_items is not None:
                self._refresh_tool_batch(item)
            return
        item = ChatItem(
            role="tool",
            text=format_tool_result_summary(name=tool_name, ok=not is_error),
            tool_call_id=tool_call_id,
            tool_result_text=result_text,
            tool_result=result,
        )
        self.items.append(item)
        self._tool_items_by_call_id[tool_call_id] = item

    def toggle_tool_results(self) -> bool:
        # [FUNCTION DOC]
        # Flips the show-tool-results flag and returns the new value.
        """Toggle expanded display for tool results and return the new state."""
        self.show_tool_results = not self.show_tool_results
        return self.show_tool_results

    def toggle_thinking(self) -> bool:
        # [FUNCTION DOC]
        # Flips the show-thinking flag and returns the new value.
        """Toggle thinking-token display and return the new state."""
        self.show_thinking = not self.show_thinking
        return self.show_thinking

    def update_queue(self, *, steering: tuple[str, ...], follow_up: tuple[str, ...]) -> None:
        # [FUNCTION DOC]
        # Replaces the displayed queued steering and follow-up texts.
        """Replace visible queued-message state."""
        self.queued_steering = steering
        self.queued_follow_up = follow_up

    @property
    def queued_message_count(self) -> int:
        # [FUNCTION DOC]
        # Property: number of queued steering plus follow-up texts.
        """Return the total number of pending queued messages."""
        return len(self.queued_steering) + len(self.queued_follow_up)

    def clear(self) -> None:
        # [FUNCTION DOC]
        # Clears transcript items, the three tool indexes, the assistant buffer and
        # the error; leaves flags, queues, skills, renderers and batch counter as is.
        # Does not touch durable session history.
        """Clear visible transcript state without modifying durable session history."""
        self.items.clear()
        self._tool_items_by_call_id.clear()
        self._grouped_calls_by_call_id.clear()
        self._batched_items_by_call_id.clear()
        self.assistant_buffer = ""
        self.error = None

    def set_skills(self, skills: Iterable[Skill]) -> None:
        # [FUNCTION DOC]
        # Replaces the loaded skills tuple used to detect skill-file reads.
        """Replace loaded skill metadata used for presentation-only path matching."""
        self.skills = tuple(skills)

    def load_messages(self, messages: Iterable[AgentMessage]) -> None:
        # [FUNCTION DOC]
        # Rebuilds transcript rows from restored canonical messages: users (with
        # summary/skill compaction), displayable custom messages, assistant messages
        # (errors via add_assistant_error), tool results attached to their calls, and
        # branch/compaction summary messages as collapsed items. Other kinds ignored.
        """Populate the transcript from restored canonical session messages."""
        for message in messages:
            if isinstance(message, UserMessage):
                self.add_user_message(message.text)
            elif isinstance(message, CustomMessage) and message.display:
                self.add_user_message(
                    message.text,
                    custom_type=message.custom_type,
                    details=message.details if isinstance(message.details, dict) else None,
                )
            elif isinstance(message, AssistantMessage):
                if message.stop_reason in {"error", "aborted"}:
                    self.add_assistant_error(message)
                else:
                    self.add_assistant_message(message)
            elif isinstance(message, ToolResultMessage):
                self.record_tool_result(
                    message.tool_call_id,
                    message.tool_name,
                    AgentToolResult(content=message.content, details=message.details),
                    message.is_error,
                )
            elif isinstance(message, BranchSummaryMessage):
                self.add_item(
                    "branch_summary",
                    "Branch summary (Ctrl+O to expand)",
                    tool_result_text=message.summary,
                )
            elif isinstance(message, CompactionSummaryMessage):
                self.add_item(
                    "compaction_summary",
                    "Compaction summary (Ctrl+O to expand)",
                    tool_result_text=message.summary,
                )

    def add_assistant_message(
        self,
        message: AssistantMessage,
        *,
        include_tool_calls: bool = True,
    ) -> None:
        # [FUNCTION DOC]
        # Projects an assistant message's blocks in order: non-empty thinking ->
        # thinking row, non-empty text -> assistant row, tool calls -> tool rows (only
        # when include_tool_calls, sharing one new batch id and the continuation flag
        # computed for edit/write-only messages).
        """Project canonical assistant blocks into display state in order."""
        batch_id = self.new_tool_batch_id() if include_tool_calls and message.tool_calls else None
        allows_mutation_continuation = _is_file_mutation_only_message(message)
        for block in message.content:
            if isinstance(block, ThinkingContent):
                if block.thinking:
                    self.add_item("thinking", block.thinking)
            elif isinstance(block, TextContent):
                if block.text:
                    self.add_item("assistant", block.text)
            elif include_tool_calls:
                self.add_tool_call(
                    block,
                    batch_id=batch_id,
                    allows_file_mutation_continuation=allows_mutation_continuation,
                )

    def add_assistant_error(self, message: AssistantMessage) -> None:
        # [FUNCTION DOC]
        # Adds the partial response (without tool calls), sets error to the message's
        # error text (default "Error") and appends an "Error: ..." error row.
        """Project any partial response followed by its terminal error."""
        self.add_assistant_message(message, include_tool_calls=False)
        text = message.error_message or "Error"
        self.error = text
        self.add_item("error", f"Error: {text}")

    def _read_skill_name(self, tool_call: ToolCall) -> str | None:
        # [FUNCTION DOC]
        # HELPER: For a "read" call with a string path, returns the name of the loaded
        # skill whose normalised (user-expanded, resolved) path equals it; else none.
        # Side effects: filesystem path resolution.
        if tool_call.name != "read":
            return None
        path = _string_argument(tool_call.arguments, "path")
        if path is None:
            return None
        read_path = _normalized_path(path)
        for skill in self.skills:
            if _normalized_path(skill.path) == read_path:
                return skill.name
        return None


def _is_file_mutation_only_message(message: AssistantMessage) -> bool:
    # [FUNCTION DOC]
    # HELPER: True when the message content consists solely of one or more tool
    # calls that all use the same tool, which is edit or write. Also imported by the
    # TUI adapter.
    calls = [block for block in message.content if isinstance(block, ToolCall)]
    return (
        bool(calls)
        and len(calls) == len(message.content)
        and all(
            call.name == calls[0].name and call.name in RESULTFUL_FILE_GROUP_NAMES for call in calls
        )
    )


def _parse_branch_summary_message(content: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Returns the inner summary when text is exactly wrapped in the branch
    # summary preamble plus "<summary>" tags; else none.
    prefix = (
        "The following is a summary of a branch that this conversation came back from:\n<summary>\n"
    )
    suffix = "\n</summary>"
    if content.startswith(prefix) and content.endswith(suffix):
        return content.removeprefix(prefix).removesuffix(suffix)
    return None


def _parse_compaction_summary_message(content: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Returns text after a leading "Previous conversation summary:" line,
    # or none.
    prefix = "Previous conversation summary:\n"
    if content.startswith(prefix):
        return content.removeprefix(prefix)
    return None


def format_elapsed(seconds: float) -> str:
    # [FUNCTION DOC]
    # Formats seconds (truncated to int) as "Ns", "Mm Ss" or "Hh Mm". Pure.
    """Format an elapsed duration tersely: 23s, 1m 23s, 1h 2m."""
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def format_tool_call_block(tool_call: ToolCall, *, compact: bool = True) -> str:
    # [FUNCTION DOC]
    # Formats a call for a transcript row: bash invocations as-is (they carry
    # their own prefix), other tools prefixed with "→ ". Compact by default.
    """Format a tool call, optionally compacting long bash invocations."""
    invocation = format_tool_call_invocation(tool_call, expanded=not compact)
    if tool_call.name == "bash":
        return invocation
    return f"→ {invocation}"


def format_tool_call_invocation(tool_call: ToolCall, *, expanded: bool = False) -> str:
    # [FUNCTION DOC]
    # Terse invocation text: "read <path>" with an optional ":start-end" line
    # range, "edit <path>", "write <path>", bash via its formatter (compact unless
    # expanded); falls back to "<name> <truncated args>" when arguments are
    # missing or the tool is unknown. Pure.
    """Format a tool call as a terse human-readable invocation."""
    arguments = tool_call.arguments
    if tool_call.name == "read":
        path = _string_argument(arguments, "path")
        if path is None:
            return _fallback_tool_call_invocation(tool_call)
        return f"read {path}{_read_line_suffix(arguments)}"
    if tool_call.name == "edit":
        path = _string_argument(arguments, "path")
        if path is None:
            return _fallback_tool_call_invocation(tool_call)
        return f"edit {path}"
    if tool_call.name == "write":
        path = _string_argument(arguments, "path")
        if path is None:
            return _fallback_tool_call_invocation(tool_call)
        return f"write {path}"
    if tool_call.name == "bash":
        invocation = _format_bash_tool_call_invocation(arguments, compact=not expanded)
        return invocation if invocation is not None else _fallback_tool_call_invocation(tool_call)
    return _fallback_tool_call_invocation(tool_call)


def _format_bash_tool_call_invocation(
    arguments: dict[str, JSONValue], *, compact: bool
) -> str | None:
    # [FUNCTION DOC]
    # HELPER: For a string command: compact form "→ <whitespace-normalised
    # description>" or "→ Running shell command"; expanded form "$ <command>";
    # both with " (timeout Ns)" when a numeric timeout exists. None without command.
    command = _string_argument(arguments, "command")
    if command is None:
        return None
    timeout = _number_argument(arguments, "timeout")
    suffix = f" (timeout {timeout:g}s)" if timeout is not None else ""
    if compact:
        description = _string_argument(arguments, "description")
        if description is not None:
            displayed_description = _normalize_bash_description(description)
            if displayed_description:
                return f"→ {displayed_description}{suffix}"
        return f"→ Running shell command{suffix}"
    return f"$ {command}{suffix}"


def _normalize_bash_description(description: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Collapses all whitespace runs to single spaces and trims.
    return " ".join(description.split())


def _read_line_suffix(arguments: dict[str, JSONValue]) -> str:
    # [FUNCTION DOC]
    # HELPER: Line range suffix from offset/limit integers: empty when both absent;
    # start = max(1, offset) or 1; ":start-" without limit; else ":start-end" with
    # end = start + max(1, limit) - 1.
    offset = _int_argument(arguments, "offset")
    limit = _int_argument(arguments, "limit")
    if offset is None and limit is None:
        return ""
    start = 1 if offset is None else max(1, offset)
    if limit is None:
        return f":{start}-"
    return f":{start}-{start + max(1, limit) - 1}"


FALLBACK_INVOCATION_ARGS_CHARS = 160


def _fallback_tool_call_invocation(tool_call: ToolCall) -> str:
    # [FUNCTION DOC]
    # HELPER: "<name> <string form of arguments>" truncated to 160 chars with an
    # ellipsis, or just the name when there are no arguments.
    if tool_call.arguments:
        rendered = str(tool_call.arguments)
        if len(rendered) > FALLBACK_INVOCATION_ARGS_CHARS:
            rendered = rendered[:FALLBACK_INVOCATION_ARGS_CHARS].rstrip() + "…"
        return f"{tool_call.name} {rendered}"
    return tool_call.name


def _string_argument(arguments: dict[str, JSONValue], key: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Returns the argument value if it is a string, else none.
    value = arguments.get(key)
    return value if isinstance(value, str) else None


def _normalized_path(path: str | Path) -> Path:
    # [FUNCTION DOC]
    # HELPER: Expands the user home marker and resolves the path non-strictly.
    return Path(path).expanduser().resolve(strict=False)


def _int_argument(arguments: dict[str, JSONValue], key: str) -> int | None:
    # [FUNCTION DOC]
    # HELPER: Returns the argument if it is an int (excluding bools), else none.
    value = arguments.get(key)
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _number_argument(arguments: dict[str, JSONValue], key: str) -> int | float | None:
    # [FUNCTION DOC]
    # HELPER: Returns the argument if it is an int or float (excluding bools).
    value = arguments.get(key)
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int | float) else None


def format_tool_result_summary(*, name: str, ok: bool) -> str:
    # [FUNCTION DOC]
    # Returns "✓ <name>" or "✗ <name>" for orphan results.
    """Format a terse tool result line for orphaned results."""
    status = "✓" if ok else "✗"
    return f"{status} {name}"


def format_tool_result_block(
    *,
    name: str,
    ok: bool,
    content: str,
    data: dict[str, JSONValue] | None = None,
) -> str:
    # [FUNCTION DOC]
    # Formats a tool result: "<✓|✗> <name>", a preview of the content (8 lines /
    # 2000 chars) and, for successful edits with a non-blank string "patch" in
    # details, a blank line, "Patch:" and a 32-line patch preview.
    """Format a tool result for live and restored transcript blocks."""
    status = "✓" if ok else "✗"
    lines = [f"{status} {name}"]
    if content:
        lines.append(_preview_text(content, max_lines=TOOL_RESULT_PREVIEW_LINES))
    patch = _result_patch(name=name, ok=ok, data=data)
    if patch:
        lines.extend(["", "Patch:", _preview_text(patch, max_lines=TOOL_PATCH_PREVIEW_LINES)])
    return "\n".join(lines)


def format_terminal_command_result_block(
    *,
    ok: bool,
    added_to_context: bool,
    output: str,
) -> str:
    # [FUNCTION DOC]
    # Formats an input-bar shell command result: "<✓|✗> bash · added to context"
    # or "· not added to context", plus a 120-line output preview when non-empty.
    """Format an input-bar terminal command result for visible TUI display."""
    status = "✓" if ok else "✗"
    suffix = " · added to context" if added_to_context else " · not added to context"
    lines = [f"{status} bash{suffix}"]
    if output:
        lines.append(_preview_text(output, max_lines=TERMINAL_COMMAND_OUTPUT_PREVIEW_LINES))
    return "\n".join(lines)


def _result_patch(
    *,
    name: str,
    ok: bool,
    data: dict[str, JSONValue] | None,
) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Returns the non-blank string "patch" from details for successful
    # edit results only; else none.
    if name != "edit" or not ok or data is None:
        return None
    patch = data.get("patch")
    return patch if isinstance(patch, str) and patch.strip() else None


def _preview_text(text: str, *, max_lines: int) -> str:
    # [FUNCTION DOC]
    # HELPER: Truncates text to the first max_lines lines and 2000 characters,
    # appending "[Preview only: N more line(s), additional text hidden from the
    # TUI.]" describing what was cut. Text with no lines is returned cut to 2000
    # chars.
    lines = text.splitlines()
    if not lines:
        return text[:TOOL_RESULT_PREVIEW_CHARS]

    preview_lines = lines[:max_lines]
    preview = "\n".join(preview_lines)
    hidden_lines = max(0, len(lines) - len(preview_lines))

    truncated_by_chars = len(preview) > TOOL_RESULT_PREVIEW_CHARS
    if truncated_by_chars:
        preview = preview[:TOOL_RESULT_PREVIEW_CHARS].rstrip()

    if hidden_lines or truncated_by_chars:
        details: list[str] = []
        if hidden_lines:
            details.append(f"{hidden_lines} more line{'s' if hidden_lines != 1 else ''}")
        if truncated_by_chars:
            details.append("additional text")
        preview = f"{preview}\n\n[Preview only: {', '.join(details)} hidden from the TUI.]"
    return preview
