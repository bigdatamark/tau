# [MODULE DOC]
# Tests for tool-history repair in the agent package: making provider-bound
# history well-formed by synthesizing missing tool results, dropping orphan and
# duplicate results, and moving results next to their tool calls.
from tau_agent import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
from tau_agent.tool_history import repair_tool_history


def _call(call_id: str = "call-1", name: str = "read") -> ToolCall:
    # [FUNCTION DOC]
    # HELPER: Build a tool-call content item with the given ID (default "call-1")
    # and tool name (default "read") and a fixed path argument "README.md".
    return ToolCall(id=call_id, name=name, arguments={"path": "README.md"})


def _result(
    call_id: str = "call-1",
    *,
    text: str = "contents",
    is_error: bool = False,
) -> ToolResultMessage:
    # [FUNCTION DOC]
    # HELPER: Build a "read" tool-result message for the given call ID with a single
    # text content item (default "contents") and an error flag (default False).
    return ToolResultMessage(
        tool_call_id=call_id,
        tool_name="read",
        content=[TextContent(text=text)],
        is_error=is_error,
    )


def test_valid_tool_history_is_unchanged() -> None:
    # [FUNCTION DOC]
    # Test that a well-formed history is returned unchanged, reported as not
    # changed, and its diagnostic data has all four counters at zero.
    call = _call()
    result = _result()
    messages = (
        UserMessage(content="read"),
        AssistantMessage(content=[call]),
        result,
        AssistantMessage(content="done"),
    )

    repair = repair_tool_history(messages)

    assert repair.changed is False
    assert repair.messages == messages
    assert repair.diagnostic_data() == {
        "synthesizedResults": 0,
        "droppedOrphanResults": 0,
        "droppedDuplicateResults": 0,
        "reorderedResults": 0,
    }


def test_missing_result_gets_synthetic_interruption() -> None:
    # [FUNCTION DOC]
    # Test that a tool call with no result gets a synthesized error result appended
    # with the call's ID and name and text "Tool call interrupted by user", and the
    # synthesized counter is 1.
    call = _call()

    repair = repair_tool_history((UserMessage(content="read"), AssistantMessage(content=[call])))

    assert repair.changed is True
    result = repair.messages[-1]
    assert isinstance(result, ToolResultMessage)
    assert result.tool_call_id == call.id
    assert result.tool_name == call.name
    assert result.text == "Tool call interrupted by user"
    assert result.is_error is True
    assert repair.synthesized_results == 1


def test_orphan_result_is_dropped_without_inventing_a_call() -> None:
    # [FUNCTION DOC]
    # Test that a result with no matching call is dropped (not paired with an
    # invented call) and counted as an orphan.
    user = UserMessage(content="continue")
    orphan = _result("call-missing")

    repair = repair_tool_history((orphan, user))

    assert repair.changed is True
    assert repair.messages == (user,)
    assert repair.dropped_orphan_results == 1
    assert repair.synthesized_results == 0


def test_late_result_moves_next_to_its_call() -> None:
    # [FUNCTION DOC]
    # Test that a result appearing after an intervening user message is moved
    # directly after its assistant tool call and counted as reordered.
    call = _call()
    assistant = AssistantMessage(content=[call])
    user = UserMessage(content="continue")
    result = _result()

    repair = repair_tool_history((assistant, user, result))

    assert repair.messages == (assistant, result, user)
    assert repair.changed is True
    assert repair.reordered_results == 1


def test_duplicate_results_prefer_real_output_over_interruption() -> None:
    # [FUNCTION DOC]
    # Test that when a call has both a synthetic interruption result and a real
    # result, the real one is kept and one duplicate is counted as dropped.
    call = _call()
    assistant = AssistantMessage(content=[call])
    interrupted = _result(text="Tool call interrupted by user", is_error=True)
    actual = _result(text="contents")

    repair = repair_tool_history((assistant, interrupted, actual))

    assert repair.messages == (assistant, actual)
    assert repair.dropped_duplicate_results == 1


def test_repeated_call_ids_keep_one_result_per_turn() -> None:
    # [FUNCTION DOC]
    # Test that the same call ID reused in two separate turns, each with its own
    # adjacent result, is valid and left unchanged.
    first_call = _call("same")
    first_assistant = AssistantMessage(content=[first_call])
    first_result = _result("same", text="first")
    second_call = _call("same")
    second_assistant = AssistantMessage(content=[second_call])
    second_result = _result("same", text="second")
    messages = (first_assistant, first_result, second_assistant, second_result)

    repair = repair_tool_history(messages)

    assert repair.changed is False
    assert repair.messages == messages
    assert repair.dropped_duplicate_results == 0


def test_repeated_call_id_reserves_later_adjacent_result() -> None:
    # [FUNCTION DOC]
    # Test that when a reused call ID's first turn lacks a result, the later
    # adjacent result stays with the second call and the first call gets a
    # synthesized interruption result instead.
    first_assistant = AssistantMessage(content=[_call("same")])
    user = UserMessage(content="continue")
    second_assistant = AssistantMessage(content=[_call("same")])
    second_result = _result("same", text="second")

    repair = repair_tool_history((first_assistant, user, second_assistant, second_result))

    first_result = repair.messages[1]
    assert isinstance(first_result, ToolResultMessage)
    assert first_result.text == "Tool call interrupted by user"
    assert repair.messages == (first_assistant, first_result, user, second_assistant, second_result)
    assert repair.synthesized_results == 1


def test_parallel_results_follow_tool_call_order() -> None:
    # [FUNCTION DOC]
    # Test that results for multiple calls in one assistant message are reordered
    # to match the order of the tool calls.
    first = _call("call-1", "read")
    second = _call("call-2", "bash")
    assistant = AssistantMessage(content=[first, second])
    second_result = ToolResultMessage(tool_call_id="call-2", tool_name="bash", content="second")
    first_result = _result("call-1", text="first")

    repair = repair_tool_history((assistant, second_result, first_result))

    assert repair.messages == (assistant, first_result, second_result)
