# [MODULE DOC]
# Bundled example extension showing how to append an always-on, labeled section to Tau's system
# prompt via the extension API. Loaded by the extension system, which calls setup().
"""Append structured, always-on instructions to Tau's system prompt."""

from tau_coding.extensions import ExtensionAPI


def setup(tau: ExtensionAPI) -> None:
    # [FUNCTION DOC]
    # Extension entry point. Registers a prompt section titled "Review procedure" whose body tells
    # the agent to read the full diff before editing and to run the project test suite and linter
    # (pytest and ruff via uv) before reporting success.
    # Input: the extension API object for this extension generation. Output: none. Side effects:
    # registers the prompt section on the API (active only while this extension generation is
    # loaded). Raises nothing itself.
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
