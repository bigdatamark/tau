# [MODULE DOC]
# HELPER: Utility module that normalizes tool-call identifiers so conversation
# history recorded with one provider can be replayed to another provider.
# The portable format is 1-64 characters of ASCII letters, digits, underscore or
# hyphen (Anthropic's alphabet, the strictest among Tau's providers).
"""Portable tool-call identifiers for cross-provider history replay."""

from __future__ import annotations

import re
from hashlib import sha256

# Anthropic has the strictest documented identifier alphabet among Tau's
# providers. Keeping translated IDs within this common subset also avoids
# provider-specific punctuation and length constraints elsewhere.
_PORTABLE_TOOL_CALL_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def portable_tool_call_id(value: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Map any tool-call ID string to a deterministic provider-safe ID.
    # Behavior: if the input already fully matches the portable pattern it is
    # returned unchanged (keeps same-provider caching/replay intact). Otherwise the
    # UTF-8 bytes are SHA-256 hashed and the result is "tc_" followed by the first
    # 40 hex characters of the digest (43 characters total), so distinct inputs do
    # not collapse together the way character replacement would.
    # Inputs: the native ID string. Output: the portable ID string.
    # Raises nothing; pure function with no side effects.
    """Return a deterministic provider-safe correlation ID.

    Provider-native IDs that already fit the common format remain unchanged,
    preserving same-provider cache/replay behavior. Other IDs are hashed rather
    than character-replaced so distinct native IDs cannot collapse together.
    """
    if _PORTABLE_TOOL_CALL_ID.fullmatch(value):
        return value
    digest = sha256(value.encode("utf-8")).hexdigest()
    return f"tc_{digest[:40]}"
