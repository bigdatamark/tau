# [MODULE DOC]
# Shared JSON-like type aliases for the portable agent layer (tau_agent).
# Defines three PEP 695 recursive type aliases used across messages, tools and
# sessions so that pydantic can validate arbitrary JSON payloads:
# - JSONPrimitive: a string, integer, float, boolean or null.
# - JSONValue: a primitive, a list of JSON values, or a mapping from string
#   keys to JSON values (recursive).
# - JSONObject: a mapping from string keys to JSON values.
# No runtime behavior, I/O or side effects; aliases only.
"""Shared low-level types for Tau's portable agent layer."""

from __future__ import annotations

# Pydantic needs PEP 695 named recursive aliases for JSON-like values.
type JSONPrimitive = str | int | float | bool | None
type JSONValue = JSONPrimitive | list[JSONValue] | dict[str, JSONValue]
type JSONObject = dict[str, JSONValue]
