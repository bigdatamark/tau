# [MODULE DOC]
# Generic polling loop for OAuth 2.0 device authorization flows (RFC 8628),
# independent of any particular provider. A provider supplies a coroutine that
# performs one token request and reports a status; this module handles interval
# timing, slow_down back-off, expiry and user cancellation. Sleep and clock
# functions are injectable for deterministic tests.
"""RFC 8628-style device authorization polling helpers."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from tau_coding.oauth import OAuthError

DevicePollStatus = Literal["complete", "pending", "slow_down", "failed"]


@dataclass(frozen=True, slots=True)
class DevicePollResult[T]:
    # [CLASS DOC]
    # Immutable, generic result of one token polling request. Fields: status (one of
    # complete, pending, slow_down, failed), value (the credential when complete),
    # message (error text when failed), and interval_seconds (a server-requested new
    # polling interval, typically sent with slow_down).
    """Result of one device-token polling request."""

    status: DevicePollStatus
    value: T | None = None
    message: str | None = None
    interval_seconds: float | None = None


async def poll_oauth_device_code[T](
    poll: Callable[[], Awaitable[DevicePollResult[T]]],
    *,
    interval_seconds: float | None = None,
    expires_in_seconds: float | None = None,
    wait_before_first_poll: bool = False,
    cancel_event: asyncio.Event | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> T:
    # [FUNCTION DOC]
    # Repeatedly call the poll coroutine until it yields a credential.
    # Inputs: the poll coroutine factory; optional initial interval (invalid/missing
    # becomes 5 s, minimum 1 s); optional expiry in seconds (missing means no
    # deadline); whether to wait one interval (capped at remaining time) before the
    # first poll; optional cancellation event; injectable sleep and monotonic clock.
    # Behavior per iteration while before the deadline: raise if cancelled, poll;
    # "complete" returns the value; "failed" raises; "slow_down" switches to the
    # server's interval (normalized) or else adds 5 s to the current interval;
    # "pending" keeps the interval. Then waits min(interval, remaining time),
    # breaking out when no time remains.
    # Output: the credential value. Raises OAuthError when complete has no value
    # ("Device flow returned no credential"), on failed (server message or a default),
    # on cancellation ("Login cancelled"), or on timeout ("Device flow timed out",
    # with a suffix noting slow_down responses if any were seen). Exceptions from
    # the poll coroutine propagate. Side effects: sleeps and whatever poll does.
    """Poll an OAuth device flow with RFC 8628 timing and cancellation."""
    interval = _poll_interval(interval_seconds)
    deadline = monotonic() + expires_in_seconds if expires_in_seconds is not None else math.inf
    if wait_before_first_poll:
        await _wait(min(interval, max(deadline - monotonic(), 0)), cancel_event, sleep=sleep)

    saw_slow_down = False
    while monotonic() < deadline:
        _raise_if_cancelled(cancel_event)
        result = await poll()
        if result.status == "complete":
            if result.value is None:
                raise OAuthError("Device flow returned no credential")
            return result.value
        if result.status == "failed":
            raise OAuthError(result.message or "Device authorization failed")
        if result.status == "slow_down":
            saw_slow_down = True
            interval = (
                _poll_interval(result.interval_seconds)
                if result.interval_seconds is not None
                else interval + 5
            )

        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        await _wait(min(interval, remaining), cancel_event, sleep=sleep)

    suffix = " after one or more slow_down responses" if saw_slow_down else ""
    raise OAuthError(f"Device flow timed out{suffix}")


def _poll_interval(value: float | None) -> float:
    # [FUNCTION DOC]
    # HELPER: Normalize a polling interval in seconds: None, non-finite or
    # non-positive values become 5; otherwise at least 1. Pure; raises nothing.
    if value is None or not math.isfinite(value) or value <= 0:
        return 5
    return max(value, 1)


async def _wait(
    seconds: float,
    cancel_event: asyncio.Event | None,
    *,
    sleep: Callable[[float], Awaitable[None]],
) -> None:
    # [FUNCTION DOC]
    # HELPER: Sleep for the given seconds unless cancelled.
    # Behavior: raises immediately if already cancelled; returns at once for
    # non-positive durations; without a cancel event just awaits the sleep function.
    # With an event, races a sleep task against the event's wait task, cancels the
    # loser; if the event fired it cancels the sleep and raises, otherwise awaits
    # the finished sleep task (re-raising any error from it).
    # Raises OAuthError("Login cancelled") on cancellation. Side effects: creates
    # and cancels asyncio tasks.
    _raise_if_cancelled(cancel_event)
    if seconds <= 0:
        return
    if cancel_event is None:
        await sleep(seconds)
        return

    async def run_sleep() -> None:
        # [FUNCTION DOC]
        # HELPER: Coroutine wrapper awaiting the injected sleep for the captured
        # duration so it can be scheduled as a task.
        await sleep(seconds)

    sleep_task: asyncio.Task[None] = asyncio.create_task(run_sleep())
    cancel_task = asyncio.create_task(cancel_event.wait())
    done, pending = await asyncio.wait(
        {sleep_task, cancel_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    if cancel_task in done and cancel_event.is_set():
        sleep_task.cancel()
        raise OAuthError("Login cancelled")
    cancel_task.cancel()
    await sleep_task


def _raise_if_cancelled(cancel_event: asyncio.Event | None) -> None:
    # [FUNCTION DOC]
    # HELPER: Raise OAuthError("Login cancelled") when a cancellation event is given
    # and set; otherwise do nothing.
    if cancel_event is not None and cancel_event.is_set():
        raise OAuthError("Login cancelled")
