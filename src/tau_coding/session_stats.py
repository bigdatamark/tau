# [MODULE DOC]
# Lifetime activity and usage totals for the active session branch (shown by stats/footer UI).
# Walks a branch's session entries (in order, including original messages later replaced by
# compaction) and accumulates turn counts, tool calls, token usage (prompt = input + cache read
# + cache write), cache hit rates, output speed, time-to-first-output and an estimated USD cost
# using a caller-provided pricing resolver (provider, model, prompt tokens) -> per-million-token
# rates mapping or None. Pure computation; no I/O.
"""Lifetime activity and usage totals for an active session branch."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from tau_agent.messages import AssistantMessage, CustomMessage, UserMessage
from tau_agent.session import BranchSummaryEntry, CompactionEntry, MessageEntry, ModelChangeEntry
from tau_agent.session.entries import CustomMessageEntry, SessionEntry

PricingResolver = Callable[[str, str, int], Mapping[str, float] | None]
_TOKENS_PER_MILLION = 1_000_000


@dataclass(frozen=True, slots=True)
class SessionStats:
    # [CLASS DOC]
    # Immutable record of cumulative branch statistics: turn and tool-call counts; cumulative prompt
    # (input) tokens, output tokens, cache-read and cache-write tokens; the latest response's prompt
    # and cache-read tokens; timed output tokens and summed response durations (ms) for throughput;
    # summed time-to-first-output (ms) and its sample count; and an estimated cost (None when unknown).
    # All fields default to zero/None. Exposes derived read-only rate properties.
    """Cumulative activity and billed usage for one active branch."""

    turn_count: int = 0
    tool_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    latest_prompt_tokens: int = 0
    latest_cached_input_tokens: int = 0
    timed_output_tokens: int = 0
    response_duration_ms: int = 0
    time_to_first_output_ms: int = 0
    timed_first_output_count: int = 0
    estimated_cost: float | None = None

    @property
    def cache_hit_rate(self) -> float | None:
        # [FUNCTION DOC]
        # Property: cumulative cache hit rate = cached input tokens / total prompt tokens.
        # Returns None when no prompt tokens were recorded or when no cache reads and no cache writes
        # occurred at all (so cache-less backends do not show 0%).
        """Share of prompt tokens served from the provider's cache.

        None when no provider in the branch reported any cache activity, so that
        backends without prompt caching are not shown a permanent 0%.
        """
        if self.input_tokens <= 0:
            return None
        if self.cached_input_tokens == 0 and self.cache_write_tokens == 0:
            return None
        return self.cached_input_tokens / self.input_tokens

    @property
    def latest_cache_hit_rate(self) -> float | None:
        # [FUNCTION DOC]
        # Property: latest response's cache hit rate = latest cached tokens / latest prompt tokens.
        # Returns None when the latest prompt is zero, or when the whole branch had no cache reads or
        # writes (the guard uses cumulative counters, not latest ones, intentionally mirroring the above).
        """Share of the latest request's prompt served from cache."""
        if self.latest_prompt_tokens <= 0:
            return None
        if self.cached_input_tokens == 0 and self.cache_write_tokens == 0:
            return None
        return self.latest_cached_input_tokens / self.latest_prompt_tokens

    @property
    def output_tokens_per_second(self) -> float | None:
        # [FUNCTION DOC]
        # Property: effective output tokens per second = timed output tokens * 1000 / response duration ms.
        # Returns None when either quantity is zero or negative.
        """Token-weighted effective output speed across timed responses."""
        if self.timed_output_tokens <= 0 or self.response_duration_ms <= 0:
            return None
        return self.timed_output_tokens * 1000 / self.response_duration_ms

    @property
    def average_time_to_first_output_ms(self) -> float | None:
        # [FUNCTION DOC]
        # Property: mean time to first output in ms = summed TTFT / sample count; None with no samples.
        """Mean time from request start to the first output event."""
        if self.timed_first_output_count <= 0:
            return None
        return self.time_to_first_output_ms / self.timed_first_output_count


def calculate_session_stats(
    entries: Sequence[SessionEntry],
    *,
    pricing: PricingResolver,
) -> SessionStats:
    # [FUNCTION DOC]
    # Aggregate statistics over a branch's entries.
    # Inputs: ordered sequence of session entries; keyword-only pricing resolver.
    # Behavior: custom-message entries and user/custom message entries each count as a turn.
    # Model-change entries update the current model (and provider when given). Assistant message
    # entries update current provider/model, add their tool-call count and contribute usage plus
    # timing. Compaction and branch-summary entries with usage contribute usage priced against their
    # own provider/model or the current ones (no timing). Other entries are ignored. For each usage:
    # prompt tokens = input + cache read + cache write; latest prompt/cached values are overwritten;
    # cumulative totals are added. Timing: output tokens and duration are added only when output > 0
    # and duration > 0; TTFT is added whenever present. Usage with zero prompt and zero output is not
    # billable. For billable usage, the resolver is called; if it returns rates the response cost is
    # added; if None, the provider-reported cost is added when positive, otherwise pricing is marked
    # incomplete. Output: a stats record whose estimated cost is None unless there was billable usage
    # and every billable response could be priced. Exceptions only propagate from the resolver.
    """Aggregate original branch messages, including messages replaced by compaction."""
    turn_count = 0
    tool_call_count = 0
    input_tokens = 0
    output_tokens = 0
    cached_input_tokens = 0
    cache_write_tokens = 0
    latest_prompt_tokens = 0
    latest_cached_input_tokens = 0
    timed_output_tokens = 0
    response_duration_ms = 0
    time_to_first_output_ms = 0
    timed_first_output_count = 0
    estimated_cost = 0.0
    has_billable_usage = False
    has_complete_pricing = True
    current_provider = "unknown"
    current_model = "unknown"

    for entry in entries:
        if isinstance(entry, CustomMessageEntry):
            turn_count += 1
            continue
        if isinstance(entry, ModelChangeEntry):
            current_model = entry.model
            if entry.provider is not None:
                current_provider = entry.provider
            continue

        message: AssistantMessage | None = None
        if isinstance(entry, MessageEntry):
            if isinstance(entry.message, (UserMessage, CustomMessage)):
                turn_count += 1
                continue
            if not isinstance(entry.message, AssistantMessage):
                continue
            message = entry.message
            current_provider = message.provider
            current_model = message.model
            tool_call_count += len(message.tool_calls)
            usage = message.usage
            provider = message.provider
            model = message.model
        elif isinstance(entry, CompactionEntry | BranchSummaryEntry):
            if entry.usage is None:
                continue
            usage = entry.usage
            provider = entry.provider or current_provider
            model = entry.model or current_model
        else:
            continue

        prompt_tokens = usage.input + usage.cache_read + usage.cache_write
        latest_prompt_tokens = prompt_tokens
        latest_cached_input_tokens = usage.cache_read
        input_tokens += prompt_tokens
        cached_input_tokens += usage.cache_read
        cache_write_tokens += usage.cache_write
        output_tokens += usage.output
        timing = message.timing if message is not None else None
        if timing is not None:
            # TPS is token-weighted and requires usable output usage. TTFT is a
            # per-call arithmetic mean whenever output was observed, even if a
            # later error left that response without billed output tokens.
            if usage.output > 0 and timing.total_duration_ms > 0:
                timed_output_tokens += usage.output
                response_duration_ms += timing.total_duration_ms
            if timing.time_to_first_output_ms is not None:
                time_to_first_output_ms += timing.time_to_first_output_ms
                timed_first_output_count += 1
        if prompt_tokens == 0 and usage.output == 0:
            continue

        has_billable_usage = True
        rates = pricing(provider, model, prompt_tokens)
        if rates is None:
            if usage.cost.total > 0:
                estimated_cost += usage.cost.total
            else:
                has_complete_pricing = False
            continue
        estimated_cost += _response_cost(
            input_tokens=usage.input,
            output_tokens=usage.output,
            cache_read_tokens=usage.cache_read,
            cache_write_tokens=usage.cache_write,
            cache_write_1h_tokens=usage.cache_write_1h or 0,
            rates=rates,
        )

    return SessionStats(
        turn_count=turn_count,
        tool_call_count=tool_call_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_tokens=cache_write_tokens,
        latest_prompt_tokens=latest_prompt_tokens,
        latest_cached_input_tokens=latest_cached_input_tokens,
        timed_output_tokens=timed_output_tokens,
        response_duration_ms=response_duration_ms,
        time_to_first_output_ms=time_to_first_output_ms,
        timed_first_output_count=timed_first_output_count,
        estimated_cost=(estimated_cost if has_billable_usage and has_complete_pricing else None),
    )


def _response_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    cache_write_1h_tokens: int,
    rates: Mapping[str, float],
) -> float:
    # [FUNCTION DOC]
    # HELPER: Compute one response's USD cost from per-million-token rates.
    # Inputs (keyword-only): input, output, cache-read, total cache-write and 1-hour cache-write token
    # counts, and a rates mapping with optional keys input, output, cacheRead, cacheWrite,
    # cacheWrite1h (missing keys price at 0). The 1-hour portion is capped at total cache writes and
    # priced at cacheWrite1h (falling back to cacheWrite); the remainder at cacheWrite.
    # Output: total cost divided by one million. No side effects.
    """Calculate one response's estimated USD cost from per-million-token rates.

    ``cache_write_tokens`` is the provider-reported total, which already
    includes any 1-hour TTL writes. Anthropic bills those at a higher rate, so
    they are priced at ``cacheWrite1h`` when the catalog provides it, falling
    back to the 5-minute ``cacheWrite`` rate otherwise.
    """
    write_1h = min(cache_write_1h_tokens, cache_write_tokens)
    return (
        input_tokens * rates.get("input", 0.0)
        + output_tokens * rates.get("output", 0.0)
        + cache_read_tokens * rates.get("cacheRead", 0.0)
        + (cache_write_tokens - write_1h) * rates.get("cacheWrite", 0.0)
        + write_1h * rates.get("cacheWrite1h", rates.get("cacheWrite", 0.0))
    ) / _TOKENS_PER_MILLION
