# [MODULE DOC]
# Provider-layer (tau_ai) helpers that turn failed provider HTTP responses into
# short, human-readable error messages without leaking secrets (only status,
# provider name, model, and the provider's own error text are included).
# Error details are truncated to a fixed maximum length (1000 characters) when the
# raw body is used.
"""Helpers for surfacing safe provider HTTP error details."""

from __future__ import annotations

from collections.abc import Mapping
from json import JSONDecodeError, loads
from typing import Any

_MAX_ERROR_DETAIL_LENGTH = 1000


def provider_http_error_message(
    *,
    provider_name: str,
    status_code: int,
    body: str,
    model: str | None = None,
) -> str:
    # [FUNCTION DOC]
    # Build the user-facing error string for a failed provider HTTP request.
    # Inputs (keyword-only): provider display name, HTTP status code, raw response body
    # text, optional model id.
    # Output: "<provider> request failed with status <code>" plus " for model <model>"
    # when a model is given, plus ": <detail>" when a detail can be extracted from the
    # body. Raises nothing; pure function.
    """Return an actionable, secret-free HTTP error message for a provider response."""
    prefix = f"{provider_name} request failed with status {status_code}"
    if model:
        prefix = f"{prefix} for model {model}"
    detail = provider_http_error_detail(body)
    if detail:
        return f"{prefix}: {detail}"
    return prefix


def provider_http_error_detail(body: str) -> str:
    # [FUNCTION DOC]
    # Extract a concise error detail from a raw HTTP body.
    # If the body parses as a JSON object and a detail is found in it via the mapping
    # extractor, return that detail (not truncated). Otherwise return the stripped body
    # truncated to the maximum detail length (may be empty).
    # Input: body text. Output: detail string. Raises nothing; pure.
    """Extract a concise provider-supplied error detail from an HTTP body."""
    parsed = _loads_object(body)
    if parsed is not None:
        detail = provider_error_detail_from_mapping(parsed)
        if detail:
            return detail
    return body.strip()[:_MAX_ERROR_DETAIL_LENGTH]


def provider_error_detail_from_mapping(value: Mapping[str, Any]) -> str:
    # [FUNCTION DOC]
    # Find the most useful message in a provider error object.
    # Order: if the "error" key is a mapping, return its non-empty string "message",
    # else its non-empty string "code". Then for keys "message", "detail", "error" in
    # order: return a non-empty string value, or recurse into a mapping value and
    # return a non-empty nested result. Returns empty string if nothing found.
    # Input: a mapping from string keys to arbitrary JSON values. Output: string.
    # Raises nothing; pure.
    """Return the most useful message/code from a provider error object."""
    error = value.get("error")
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
        code = error.get("code")
        if isinstance(code, str) and code:
            return code
    for key in ("message", "detail", "error"):
        detail = value.get(key)
        if isinstance(detail, str) and detail:
            return detail
        if isinstance(detail, Mapping):
            nested = provider_error_detail_from_mapping(detail)
            if nested:
                return nested
    return ""


def _loads_object(value: str) -> Mapping[str, Any] | None:
    # [FUNCTION DOC]
    # HELPER: Parse text as JSON and return it only if it is an object (mapping);
    # returns None on invalid JSON or non-object JSON values. Pure.
    # KNOWN ISSUE: only JSON decode errors are caught; other parse failures (e.g. a
    # recursion error from extremely nested input) would propagate.
    try:
        parsed = loads(value)
    except JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None
