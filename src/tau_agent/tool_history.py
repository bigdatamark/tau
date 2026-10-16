# [MODULE DOC]
# Deterministic, provider-safe repair of tool-call history before it is sent to
# a model provider. Providers reject transcripts where an assistant tool call
# lacks a following tool result, or where results are orphaned/duplicated; this
# module rewrites the message sequence so every tool call is immediately
# followed by exactly one result, in call order.
# Defines the synthetic interruption text "Tool call interrupted by user", the
# ToolHistoryRepair result record, repair_tool_history and a private predicate.
# Side effects: none (pure functions).
"""Provider-safe repair of malformed tool-call message history."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
)

_INTERRUPTED_TOOL_RESULT = "Tool call interrupted by user"


@dataclass(frozen=True, slots=True)
class ToolHistoryRepair:
    # [CLASS DOC]
    # Frozen, slotted dataclass result of a repair: the repaired message tuple,
    # whether it differs from the input, and counters for synthesized results,
    # dropped orphan results (no matching call id anywhere), dropped duplicate
    # results (extra results for a known call id) and reordered results (real
    # results moved from a non-adjacent position).
    """A provider-safe transcript plus a summary of deterministic repairs."""

    messages: tuple[AgentMessage, ...]
    changed: bool = False
    synthesized_results: int = 0
    dropped_orphan_results: int = 0
    dropped_duplicate_results: int = 0
    reordered_results: int = 0

    def diagnostic_data(self) -> dict[str, int]:
        # [FUNCTION DOC]
        # Returns a mapping of camelCase counter names (synthesizedResults,
        # droppedOrphanResults, droppedDuplicateResults, reorderedResults) to their
        # integer values for JSON session diagnostics. Excludes messages and changed.
        # Pure; never raises.
        """Return JSON-safe counters for durable session diagnostics."""
        return {
            "synthesizedResults": self.synthesized_results,
            "droppedOrphanResults": self.dropped_orphan_results,
            "droppedDuplicateResults": self.dropped_duplicate_results,
            "reorderedResults": self.reordered_results,
        }


def repair_tool_history(messages: tuple[AgentMessage, ...]) -> ToolHistoryRepair:
    # [FUNCTION DOC]
    # Repairs a transcript so each tool call has exactly one adjacent result.
    # Inputs: a tuple of agent messages.
    # Algorithm:
    # 1. Enumerate every tool call occurrence as (assistant message index, 1-based
    #    call offset) with its expected result position = index + offset.
    # 2. Group tool result messages by tool call id with their positions.
    # 3. Reserve results already at their expected position with matching id.
    # 4. For each unpaired occurrence choose among unused results with the same id,
    #    preferring those positioned after the assistant message, and within that
    #    pool preferring a non-interruption result (else the first). If none exist,
    #    synthesize an error ToolResultMessage with the interruption text and count
    #    it.
    # 5. For occurrences paired with a real-positioned interruption result, swap in
    #    an unused non-interruption result for the same id if one exists.
    # 6. Rebuild: copy all non-result messages in order; after each assistant
    #    message append its selected results in call order, counting results whose
    #    source position differs from the expected one as reordered.
    # 7. Unused results are counted as orphans (unknown id) or duplicates.
    # Outputs: a ToolHistoryRepair whose changed flag compares the new tuple to the
    # input. Pure; does not mutate input. Raises nothing in normal use.
    # KNOWN ISSUE: step 5 skips synthesized results (position none), so a
    # synthesized interruption is never replaced; only pre-existing interruption
    # results are upgraded, contrary to the comment's broader wording.
    """Return history where every tool call has exactly one adjacent result.

    Existing result messages are moved beside their calls. Missing results get a
    deterministic interruption error. Results with no call are omitted because
    a missing call's arguments cannot be reconstructed safely. When duplicate
    results exist, a real result is preferred over Tau's synthetic interruption.
    """
    call_occurrences: list[tuple[tuple[int, int], ToolCall, int]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, AssistantMessage):
            continue
        for call_offset, call in enumerate(message.tool_calls, start=1):
            call_occurrences.append(
                ((message_index, call_offset), call, message_index + call_offset)
            )

    results_by_id: dict[str, list[tuple[int, ToolResultMessage]]] = defaultdict(list)
    for message_index, message in enumerate(messages):
        if isinstance(message, ToolResultMessage):
            results_by_id[message.tool_call_id].append((message_index, message))

    selected_results: dict[tuple[int, int], tuple[int | None, ToolResultMessage]] = {}
    used_result_positions: set[int] = set()

    # Reserve already-adjacent pairs first. This keeps valid repeated IDs paired
    # with their own turn rather than letting an earlier occurrence consume them.
    for occurrence, call, expected_position in call_occurrences:
        if expected_position >= len(messages):
            continue
        candidate = messages[expected_position]
        if (
            isinstance(candidate, ToolResultMessage)
            and candidate.tool_call_id == call.id
            and expected_position not in used_result_positions
        ):
            selected_results[occurrence] = (expected_position, candidate)
            used_result_positions.add(expected_position)

    synthesized_results = 0
    for occurrence, call, _expected_position in call_occurrences:
        if occurrence in selected_results:
            continue
        candidates = [
            candidate
            for candidate in results_by_id.get(call.id, [])
            if candidate[0] not in used_result_positions
        ]
        if candidates:
            after_call = [candidate for candidate in candidates if candidate[0] > occurrence[0]]
            candidate_pool = after_call or candidates
            selected = next(
                (
                    candidate
                    for candidate in candidate_pool
                    if not _is_interruption_result(candidate[1])
                ),
                candidate_pool[0],
            )
            selected_results[occurrence] = selected
            used_result_positions.add(selected[0])
            continue
        selected_results[occurrence] = (
            None,
            ToolResultMessage(
                tool_call_id=call.id,
                tool_name=call.name,
                content=[TextContent(text=_INTERRUPTED_TOOL_RESULT)],
                is_error=True,
            ),
        )
        synthesized_results += 1

    # If every call occurrence is already paired and a real extra result remains,
    # prefer it over a selected synthetic interruption for the same ID.
    for occurrence, call, _expected_position in call_occurrences:
        selected_position, selected_result = selected_results[occurrence]
        if selected_position is None or not _is_interruption_result(selected_result):
            continue
        replacement = next(
            (
                candidate
                for candidate in results_by_id.get(call.id, [])
                if candidate[0] not in used_result_positions
                and not _is_interruption_result(candidate[1])
            ),
            None,
        )
        if replacement is None:
            continue
        used_result_positions.remove(selected_position)
        used_result_positions.add(replacement[0])
        selected_results[occurrence] = replacement

    repaired: list[AgentMessage] = []
    reordered_results = 0
    for message_index, message in enumerate(messages):
        if isinstance(message, ToolResultMessage):
            continue
        repaired.append(message)
        if not isinstance(message, AssistantMessage):
            continue
        for call_offset, _call in enumerate(message.tool_calls, start=1):
            result_position, result = selected_results[(message_index, call_offset)]
            repaired.append(result)
            if result_position is not None and result_position != message_index + call_offset:
                reordered_results += 1

    call_ids = {call.id for _occurrence, call, _expected in call_occurrences}
    unused_results = [
        result
        for results in results_by_id.values()
        for position, result in results
        if position not in used_result_positions
    ]
    orphan_results = sum(result.tool_call_id not in call_ids for result in unused_results)
    dropped_duplicate_results = sum(result.tool_call_id in call_ids for result in unused_results)
    repaired_messages = tuple(repaired)
    return ToolHistoryRepair(
        messages=repaired_messages,
        changed=repaired_messages != messages,
        synthesized_results=synthesized_results,
        dropped_orphan_results=orphan_results,
        dropped_duplicate_results=dropped_duplicate_results,
        reordered_results=reordered_results,
    )


def _is_interruption_result(message: ToolResultMessage) -> bool:
    # [FUNCTION DOC]
    # HELPER: True when a tool result is flagged as an error and its text equals
    # Tau's synthetic interruption message. Pure.
    return message.is_error and message.text == _INTERRUPTED_TOOL_RESULT
