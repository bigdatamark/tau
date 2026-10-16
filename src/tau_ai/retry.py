# [MODULE DOC]
# HELPER: Shared retry utilities for tau_ai provider adapters: exponential backoff
# delay computation, construction of provider-neutral retry progress events, and
# a cancellable backoff sleep. Also defines constants: the poll interval used
# while sleeping (0.05s), the base backoff delay (0.25s), and the tuple of
# transport exception types (all httpx HTTP errors plus raw SSL errors) that are
# considered safe to retry before any streamed content has been emitted.
"""Shared retry helpers for provider adapters."""

from __future__ import annotations

import ssl
from asyncio import sleep

import httpx

from tau_agent.types import JSONValue
from tau_ai._provider_events import ProviderRetryEvent
from tau_ai.provider import CancellationToken

RETRY_POLL_SECONDS = 0.05
RETRY_BASE_DELAY_SECONDS = 0.25

# Transport failures that are safe to retry before any content is emitted.
# httpcore's anyio backend only maps ssl.SSLError during the TLS handshake (to
# ConnectError), so post-handshake record errors such as
# SSLV3_ALERT_BAD_RECORD_MAC escape as raw ssl.SSLError.
RETRYABLE_TRANSPORT_ERRORS: tuple[type[Exception], ...] = (httpx.HTTPError, ssl.SSLError)


def retry_delay_seconds(attempt: int, *, max_delay_seconds: float) -> float:
    # [FUNCTION DOC]
    # HELPER: Compute exponential backoff for a zero-based retry attempt number.
    # Inputs: the attempt index and (keyword) the provider's maximum delay in seconds.
    # Behavior: if the max delay is zero or negative returns 0.0; otherwise base is
    # the smaller of 0.25s and the max delay, and the result is base times 2 to the
    # power of attempt, capped at the max delay.
    # Outputs: delay as a float. Raises nothing; no side effects.
    """Return an exponential retry delay capped by provider config."""
    if max_delay_seconds <= 0:
        return 0.0
    base_delay = min(RETRY_BASE_DELAY_SECONDS, max_delay_seconds)
    return float(min(max_delay_seconds, base_delay * (2**attempt)))


def provider_retry_event(
    *,
    attempt: int,
    max_retries: int,
    delay_seconds: float,
    reason: str,
    data: dict[str, JSONValue] | None = None,
) -> ProviderRetryEvent:
    # [FUNCTION DOC]
    # HELPER: Build a retry progress event for consumers (renderers/UI).
    # Inputs (keyword-only): zero-based attempt index of the failed try, configured
    # max retries, delay before the next try, a human-readable failure reason, and
    # optional JSON-compatible extra data.
    # Behavior: the next attempt number is reported one-based as attempt plus 2 and
    # max attempts as max retries plus 1; the message reads "Retrying provider
    # request N/M after <reason>" with " in <delay>s" appended when delay is nonzero.
    # Outputs: a provider retry event carrying those fields. No side effects.
    """Build a provider-neutral retry progress event."""
    next_attempt = attempt + 2
    max_attempts = max_retries + 1
    delay_suffix = f" in {delay_seconds:g}s" if delay_seconds else ""
    return ProviderRetryEvent(
        attempt=next_attempt,
        max_attempts=max_attempts,
        delay_seconds=delay_seconds,
        message=(
            f"Retrying provider request {next_attempt}/{max_attempts} after {reason}{delay_suffix}."
        ),
        data=data,
    )


async def wait_for_retry(
    delay_seconds: float,
    *,
    signal: CancellationToken | None,
) -> bool:
    # [FUNCTION DOC]
    # HELPER: Asynchronously sleep for a backoff delay while honouring cancellation.
    # Inputs: delay in seconds and (keyword) an optional cancellation token.
    # Behavior: with a non-positive delay, returns immediately. Otherwise sleeps in
    # steps of at most the poll interval, checking the token before each step and
    # returning early if cancelled.
    # Outputs: True if the caller should proceed with the retry (no token or token
    # not cancelled at the end), False if cancellation was observed.
    # Side effects: suspends the coroutine via asyncio sleep. Raises nothing itself
    # (asyncio cancellation of the task propagates).
    """Sleep before a retry while allowing cancellation to interrupt backoff."""
    if delay_seconds <= 0:
        return signal is None or not signal.is_cancelled()

    remaining = delay_seconds
    while remaining > 0:
        if signal is not None and signal.is_cancelled():
            return False
        step = min(RETRY_POLL_SECONDS, remaining)
        await sleep(step)
        remaining -= step
    return signal is None or not signal.is_cancelled()
