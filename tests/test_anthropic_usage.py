# [MODULE DOC]
# Tests for Anthropic streaming usage accounting in tau_ai.anthropic: building a usage
# record from the message_start payload and merging message_delta usage payloads.
# Verifies fresh input vs cache read/write tokens, 1-hour cache writes, order
# independence, partial/repeated/zero/null updates, and sanitisation of malformed values.
from tau_agent import Usage
from tau_ai.anthropic import _apply_message_delta_usage, _usage_from_message_start


def _assert_tokens(
    usage: Usage | None,
    *,
    input: int,
    output: int,
    cache_read: int,
    cache_write: int,
    total: int,
) -> None:
    # [FUNCTION DOC]
    # HELPER: Assert a usage record is not None and its input, output, cache read, cache
    # write and total token counts equal the given keyword values. Raises AssertionError.
    assert usage is not None
    assert usage.input == input
    assert usage.output == output
    assert usage.cache_read == cache_read
    assert usage.cache_write == cache_write
    assert usage.total_tokens == total


def test_message_start_preserves_anthropic_fresh_input_and_cache_details() -> None:
    # [FUNCTION DOC]
    # A message_start usage with input 100, output 7, cache read 40, cache creation 25 and
    # 1-hour ephemeral cache 10 yields those values with total 172 (sum of all four) and
    # cache_write_1h of 10.
    usage = _usage_from_message_start(
        {
            "input_tokens": 100,
            "output_tokens": 7,
            "cache_read_input_tokens": 40,
            "cache_creation_input_tokens": 25,
            "cache_creation": {"ephemeral_1h_input_tokens": 10},
        }
    )

    _assert_tokens(usage, input=100, output=7, cache_read=40, cache_write=25, total=172)
    assert usage.cache_write_1h == 10


def test_message_delta_is_independent_of_cache_field_arrival_order() -> None:
    # [FUNCTION DOC]
    # Applies several orderings of delta payloads (full, cache-only, token-only) starting
    # from no usage; each ordering must converge to input 100, output 7, cache read 40,
    # cache write 25, total 172.
    deltas = [
        {
            "input_tokens": 100,
            "output_tokens": 7,
            "cache_read_input_tokens": 40,
            "cache_creation_input_tokens": 25,
            "output_tokens_details": {"thinking_tokens": 3},
        },
        {"cache_read_input_tokens": 40, "cache_creation_input_tokens": 25},
        {"input_tokens": 100, "output_tokens": 7},
    ]

    for ordered_deltas in (deltas, [deltas[1], deltas[0]], [deltas[2], deltas[1]]):
        usage = None
        for delta in ordered_deltas:
            usage = _apply_message_delta_usage(usage, delta)
        _assert_tokens(usage, input=100, output=7, cache_read=40, cache_write=25, total=172)


def test_message_delta_only_usage_supports_partial_and_repeated_updates() -> None:
    # [FUNCTION DOC]
    # Builds usage purely from deltas, one field at a time, then repeats all fields;
    # repeated values must not double-count and total must be 172.
    usage = _apply_message_delta_usage(None, {"input_tokens": 100})
    usage = _apply_message_delta_usage(usage, {"cache_read_input_tokens": 40})
    usage = _apply_message_delta_usage(usage, {"cache_creation_input_tokens": 25})
    usage = _apply_message_delta_usage(usage, {"output_tokens": 7})
    usage = _apply_message_delta_usage(
        usage,
        {
            "input_tokens": 100,
            "cache_read_input_tokens": 40,
            "cache_creation_input_tokens": 25,
            "output_tokens": 7,
        },
    )

    _assert_tokens(usage, input=100, output=7, cache_read=40, cache_write=25, total=172)


def test_message_delta_zero_and_null_values_follow_partial_update_semantics() -> None:
    # [FUNCTION DOC]
    # Explicit zero values overwrite previous counts (including thinking tokens -> reasoning
    # 0), while subsequent null values are treated as absent and leave the zeros in place.
    usage = _apply_message_delta_usage(
        None,
        {
            "input_tokens": 100,
            "output_tokens": 7,
            "cache_read_input_tokens": 40,
            "cache_creation_input_tokens": 25,
        },
    )
    usage = _apply_message_delta_usage(
        usage,
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens_details": {"thinking_tokens": 0},
        },
    )
    usage = _apply_message_delta_usage(
        usage,
        {
            "input_tokens": None,
            "output_tokens": None,
            "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None,
            "output_tokens_details": {"thinking_tokens": None},
        },
    )

    _assert_tokens(usage, input=0, output=0, cache_read=0, cache_write=0, total=0)
    assert usage is not None
    assert usage.reasoning == 0


def test_malformed_and_negative_usage_values_do_not_create_negative_totals() -> None:
    # [FUNCTION DOC]
    # Negative, string, float and boolean usage values are rejected (treated as 0 / None)
    # both in message_start and in deltas, giving all-zero totals, no 1-hour cache write
    # and no reasoning count.
    usage = _usage_from_message_start(
        {
            "input_tokens": -1,
            "output_tokens": "7",
            "cache_read_input_tokens": -40,
            "cache_creation_input_tokens": 2.5,
            "cache_creation": {"ephemeral_1h_input_tokens": False},
        }
    )
    _assert_tokens(usage, input=0, output=0, cache_read=0, cache_write=0, total=0)
    assert usage.cache_write_1h is None

    usage = _apply_message_delta_usage(
        usage,
        {
            "input_tokens": -1,
            "output_tokens": "7",
            "cache_read_input_tokens": -40,
            "cache_creation_input_tokens": 2.5,
            "output_tokens_details": {"thinking_tokens": -3},
        },
    )

    _assert_tokens(usage, input=0, output=0, cache_read=0, cache_write=0, total=0)
    assert usage is not None
    assert usage.reasoning is None


def test_message_delta_ignores_non_mapping_and_absent_usage() -> None:
    # [FUNCTION DOC]
    # Applying a None delta or a mapping without usage fields returns the same usage object
    # unchanged (identity preserved).
    usage = Usage(input=100, output=7, cache_read=40, cache_write=25, total_tokens=172)

    assert _apply_message_delta_usage(usage, None) is usage
    assert _apply_message_delta_usage(usage, {"unrelated": 1}) is usage
    _assert_tokens(usage, input=100, output=7, cache_read=40, cache_write=25, total=172)
