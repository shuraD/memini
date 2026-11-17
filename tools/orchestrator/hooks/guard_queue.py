"""Claude PreToolUse hook: no session writes a queue file by hand (FLOW-HARD.md D1).

Usage (from the user settings, for every session)::

    python -P guard_queue.py     # the hook event arrives on stdin

Only ``python -m tools.orchestrator enqueue`` (and the orchestrator's own stage handoff) writes
``queue*.txt``. A Write, Edit or MultiEdit of such a file, or a shell command that redirects,
copies, moves, deletes or edits one in place, is blocked: exit code 2 and the reason on stderr.
Reading a queue is fine. Malformed input is blocked too. Standard library only.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

WRITE_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})
QUEUE = re.compile(r"(?:^|/)queue[\w.-]*\.txt\b")
SHELL_WRITE = re.compile(
    r">|\btee\b|\bsed\s+-\S*i|\bmv\b|\bcp\b|\brm\b|\btruncate\b|\bperl\s+-\S*i|\bdd\b"
)
BLOCK = 2
MESSAGE = "queue files are written only by the tool: python -m tools.orchestrator enqueue ..."


def blocked(event: Any) -> str:
    """Return the reason to block this tool call, or an empty string."""
    if not isinstance(event, dict) or not isinstance(event.get("tool_input"), dict):
        return "malformed hook input"
    tool, args = event.get("tool_name"), event["tool_input"]
    if tool in WRITE_TOOLS:
        target = str(args.get("file_path") or args.get("notebook_path") or "")
        return MESSAGE if QUEUE.search(target) else ""
    if tool == "Bash":
        command = str(args.get("command") or "")
        return (
            MESSAGE
            if QUEUE.search(command.replace(" ", "/")) and SHELL_WRITE.search(command)
            else ""
        )
    return ""


def main() -> int:
    """Read the event from stdin; return the hook exit code."""
    try:
        event = json.load(sys.stdin)
    except ValueError:
        event = None
    reason = blocked(event)
    if reason:
        print(reason, file=sys.stderr)
        return BLOCK
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
