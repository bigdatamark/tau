# [MODULE DOC]
# Shared HTTP helpers for Tau network integrations built on httpx. Main concern: proxy environment
# variables using the generic "socks://" scheme, which httpx rejects; these are rewritten to
# "socks5://" while clients are constructed. Covers the upper- and lower-case HTTP_PROXY,
# HTTPS_PROXY and ALL_PROXY variables.
"""HTTP client helpers shared by Tau network integrations."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx

_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def normalize_proxy_url(proxy_url: str) -> str:
    # [FUNCTION DOC]
    # Convert a proxy URL into an httpx-acceptable form. If the URL starts with "socks://" (case-
    # insensitive) the scheme is replaced with "socks5://" keeping the rest unchanged; otherwise
    # the input is returned as-is.
    # Input: proxy URL string. Output: normalized string. No exceptions, no side effects.
    """Return an httpx-compatible proxy URL.

    Some environments use ``socks://`` as a generic SOCKS proxy scheme. httpx
    accepts explicit SOCKS versions (for example ``socks5://`` and
    ``socks5h://``), but rejects the generic scheme before it can make a
    request. Treat the generic form as SOCKS5 so Tau can honor these proxy
    environment variables.
    """

    if proxy_url.lower().startswith("socks://"):
        return f"socks5://{proxy_url[len('socks://') :]}"
    return proxy_url


@contextmanager
def normalized_proxy_environment() -> Iterator[None]:
    # [FUNCTION DOC]
    # Context manager that temporarily rewrites proxy environment variables. On entry, for each
    # known proxy variable that is set and whose normalized value differs, remembers the original
    # value and writes the normalized value into the process environment. Yields nothing. On exit
    # (even on exception), restores every changed variable to its original value.
    # Inputs: none. Side effects: mutates the process environment for the duration of the block
    # (not thread-safe). Exceptions from the body propagate after restoration.
    # KNOWN ISSUE: the branch that removes a variable when its saved original is absent can never
    # run, since only variables that were set are recorded.
    """Temporarily normalize proxy environment variables for httpx construction."""

    original: dict[str, str | None] = {}
    changed = False
    for name in _PROXY_ENV_VARS:
        value = os.environ.get(name)
        if value is None:
            continue
        normalized = normalize_proxy_url(value)
        if normalized == value:
            continue
        original[name] = value
        os.environ[name] = normalized
        changed = True

    try:
        yield
    finally:
        if changed:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def create_async_client(**kwargs: Any) -> httpx.AsyncClient:
    # [FUNCTION DOC]
    # Construct an httpx asynchronous client inside the normalized proxy environment so httpx
    # reads usable proxy settings at construction time.
    # Input: arbitrary keyword arguments forwarded to the client constructor. Output: the new
    # client (caller must close it). Side effects: temporary environment mutation. Raises whatever
    # the client constructor raises.
    """Create an ``httpx.AsyncClient`` with Tau's proxy normalization applied."""

    with normalized_proxy_environment():
        return httpx.AsyncClient(**kwargs)


def get_json(url: str, *, timeout: float, follow_redirects: bool = False) -> dict[str, object]:
    # [FUNCTION DOC]
    # Perform a synchronous GET of the URL (with given timeout and optional redirect following)
    # under the normalized proxy environment, then require a 2xx status and parse the body as
    # JSON.
    # Inputs: URL, timeout seconds (keyword-only), follow-redirects flag (default false). Output:
    # the decoded JSON object as a mapping.
    # Raises: httpx status error for non-2xx responses, httpx transport errors, JSON decode
    # errors, and a value error if the JSON top level is not an object. Side effects: network I/O,
    # temporary environment mutation.
    """Fetch a JSON object with Tau's proxy normalization applied."""

    with normalized_proxy_environment():
        response = httpx.get(url, timeout=timeout, follow_redirects=follow_redirects)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("HTTP response must be a JSON object")
    return data
