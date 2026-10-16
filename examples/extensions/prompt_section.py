# [MODULE DOC]
# Example Tau extension that contributes an always-on system prompt section.
# Exposes a single setup entry point that Tau's extension loader calls with the
# extension API; it registers a "Review procedure" section instructing the agent
# to read the full diff and run pytest and ruff before reporting success.
# No tools, commands, or UI are registered.
"""Append structured, always-on instructions to Tau's system prompt."""

from tau_coding.extensions import ExtensionAPI


def setup(tau: ExtensionAPI) -> None:
    # [FUNCTION DOC]
    # Extension entry point. Calls the API's add_prompt_section with the title
    # "Review procedure" and a fixed Markdown body (read diff first, run uv pytest
    # and ruff check before reporting success).
    # Inputs: the extension API object. Returns nothing.
    # Side effects: mutates the extension registry so the section is appended to the
    # system prompt while this extension generation is active. Raises nothing itself.
    """Add a labeled procedure while this extension generation is active."""
    tau.add_prompt_section(
        "Review procedure",
        """Read the complete diff before editing.

Run the relevant checks before reporting success:

```bash
uv run pytest
uv run ruff check .
```
""",
    )
