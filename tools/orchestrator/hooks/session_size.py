"""Claude UserPromptSubmit hook: tell a session that has grown too large to start a new one (D9).

Usage (from the user settings, for every interactive session)::

    python -P session_size.py     # the hook event arrives on stdin

The context size is the token usage the last assistant turn of the transcript reports (input,
cache read and cache creation); a transcript without usage is estimated at four bytes per token.
Above ``LIMIT`` tokens the hook prints one line that lands in the session's context. It never
blocks a prompt. Standard library only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

LIMIT = 80_000
BYTES_PER_TOKEN = 4
MESSAGE = "Write your state file and end this session; start a new one."


def _usage_tokens(line: str) -> int:
    try:
        row: Any = json.loads(line)
    except ValueError:
        return 0
    message = row.get("message") if isinstance(row, dict) else None
    usage = message.get("usage") if isinstance(message, dict) else None
    if not isinstance(usage, dict):
        return 0
    keys = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    return sum(v for k in keys if isinstance(v := usage.get(k), int))


def context_tokens(path: Path) -> int:
    """Return the context size of a transcript file in tokens (0 when it cannot be read)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    for line in reversed(text.splitlines()):
        if tokens := _usage_tokens(line):
            return tokens
    return len(text.encode()) // BYTES_PER_TOKEN


def notice(event: Any) -> str:
    """Return the line to print for this hook event, or an empty string."""
    path = event.get("transcript_path") if isinstance(event, dict) else None
    if not isinstance(path, str):
        return ""
    tokens = context_tokens(Path(path))
    return f"{MESSAGE} (about {tokens // 1000}K tokens)" if tokens > LIMIT else ""


def main() -> int:
    """Read the hook event from stdin; print the notice when the session is too large."""
    try:
        event = json.load(sys.stdin)
    except ValueError:
        return 0
    text = notice(event)
    if text:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
