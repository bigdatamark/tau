# [MODULE DOC]
# HELPER: Small utility module for OpenAI prompt-cache affinity, shared by the OpenAI providers.
# Defines the first-party API host name ("api.openai.com") and the 64-character maximum for
# prompt cache keys. Exposes a function deriving a cache key from a session id and a function
# detecting whether a base URL points at OpenAI's own API. No I/O or state.
"""Shared OpenAI prompt-cache affinity helpers."""

from __future__ import annotations

from urllib.parse import urlsplit

OPENAI_API_HOST = "api.openai.com"
OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH = 64


def openai_prompt_cache_key(session_id: str | None) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Derive a prompt_cache_key from a session id.
    # Inputs: session id string or None.
    # Output: None when the id is None or empty; otherwise the id truncated to at most 64 characters.
    # No exceptions, no side effects.
    """Return a non-empty session-derived key within OpenAI's 64-char limit."""
    if not session_id:
        return None
    return session_id[:OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH]


def is_direct_openai_url(base_url: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: Check whether a base URL targets OpenAI's first-party API.
    # Input: base URL string. Parses it with the standard URL splitter and compares the hostname
    # (lowercased by the parser) to exactly "api.openai.com".
    # Output: True on exact host match, otherwise False (subdomains/proxies return False).
    # No side effects.
    """Return whether a URL targets OpenAI's first-party API host."""
    return urlsplit(base_url).hostname == OPENAI_API_HOST
