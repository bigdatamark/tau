# [MODULE DOC]
# Example Tau extension module (not part of the installed package) showing the
# "tool_call" hook. It defines a tuple of regular expressions for dangerous shell
# commands (rm with a flag cluster containing both r and f, git push --force,
# git reset --hard, chmod -R 777, dd if=, mkfs) and blocks any bash tool call
# whose command matches one, so the tool never runs and the model receives the
# block reason instead. Loaded by Tau's extension loader, which calls setup.
"""Tau extension that blocks dangerous bash commands before they run.

Demonstrates the `tool_call` hook: return a blocking result and the tool
never executes; the model sees the block reason instead.

Install by copying into `~/.tau/extensions/`, or run:

    tau -e examples/extensions/permission_gate.py
"""

import re

from tau_coding.extensions import ExtensionAPI, ToolCallHookEvent, ToolCallHookResult

DANGEROUS_PATTERNS = (
    # rm with a flag cluster containing both r and f, in either order
    re.compile(r"\brm\s+-(?=[a-zA-Z]*r)(?=[a-zA-Z]*f)[a-zA-Z]+"),
    re.compile(r"\bgit\s+push\s+--force"),
    re.compile(r"\bgit\s+reset\s+--hard"),
    re.compile(r"\bchmod\s+-R\s+777\b"),
    re.compile(r"\bdd\s+if="),
    re.compile(r"\bmkfs\b"),
)


def _gate_tool_call(event: ToolCallHookEvent, context) -> ToolCallHookResult | None:  # noqa: ANN001
    # [FUNCTION DOC]
    # HELPER: tool_call hook handler.
    # Inputs: the tool-call hook event (tool name and argument mapping) and an
    # unused extension context. Behavior: ignores every tool other than "bash";
    # reads the "command" argument (missing becomes empty string, coerced to text)
    # and checks the dangerous patterns in order. Output: on the first match, a hook
    # result with block set and a reason naming the matched pattern and telling the
    # model to ask the user to run it manually; otherwise None (allow).
    # Raises nothing; no side effects.
    # KNOWN ISSUE: the "git push --force" pattern also matches --force-with-lease, and
    # the patterns are easily bypassed (e.g. "rm -r -f", quoting, other shells); it is
    # a demonstration, not a security boundary.
    del context
    if event.tool_name != "bash":
        return None
    command = str(event.arguments.get("command", ""))
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(command):
            return ToolCallHookResult(
                block=True,
                reason=(
                    f"command matches guarded pattern `{pattern.pattern}`; "
                    "ask the user to run it manually if it is intended"
                ),
            )
    return None


def setup(tau: ExtensionAPI) -> None:
    # [FUNCTION DOC]
    # Extension entry point called by the loader with the extension API object.
    # Behavior: registers the bash guard handler for the "tool_call" event.
    # Inputs: the extension API. Output: None. Side effect: mutates the API's hook
    # registrations. Raises nothing itself.
    """Subscribe the bash guard."""
    tau.on("tool_call", _gate_tool_call)
