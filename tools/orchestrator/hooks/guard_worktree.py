"""Claude PreToolUse guard for orchestrator steps: an early warning, never the real gate.

Usage (from the generated settings file)::

    python -P guard_worktree.py <guard-config.json>   # the hook event arrives on stdin

The config names the worktree, the writable path prefixes, the protected entries (they win
over the writable ones) and the private paths no step may read. A write outside the
worktree, to a protected path or outside the writable prefixes is blocked, and so is any
read of a private path: exit code 2 with the reason on stderr, which Claude shows the agent.
Malformed input is blocked too (fail closed). Bash is checked only for a private path in
the command text; that is a hint, not security: the orchestrator's path gate decides.

Standard library only, so it runs with ``python -P`` from any interpreter.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

WRITE_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})
READ_TOOLS = frozenset({"Read", "Grep", "Glob", "LS", "NotebookRead"})
BLOCK = 2


def _matches(entry: str, rel: str) -> bool:
    if entry.startswith("**/"):
        return rel.rsplit("/", 1)[-1] == entry[3:]
    if entry.endswith("/"):
        return rel.startswith(entry)
    return rel == entry


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _resolve(raw: str, cwd: str) -> Path:
    return (Path(cwd) / Path(raw).expanduser()).resolve()


def _private(path: Path, cfg: dict[str, Any]) -> bool:
    return any(_within(path, Path(p).resolve()) for p in cfg["deny_read"])


def _write_problem(tool: str, raw: object, cwd: str, cfg: dict[str, Any]) -> str | None:
    if not raw:
        return f"{tool}: no file path given"
    worktree = Path(cfg["worktree"]).resolve()
    path = _resolve(str(raw), cwd)
    if _private(path, cfg) or not _within(path, worktree):
        return f"{tool} outside the task worktree is not allowed: {raw}"
    rel = path.relative_to(worktree).as_posix()
    if any(_matches(p, rel) for p in cfg["protected"]):
        return f"{rel} is protected for this step; it must not be changed"
    if not any(_matches(w, rel) for w in cfg["writable"]):
        return f"{rel} is outside the writable paths: {', '.join(cfg['writable'])}"
    return None


def _read_problem(tool: str, args: dict[str, Any], cwd: str, cfg: dict[str, Any]) -> str | None:
    for key in ("file_path", "path", "notebook_path", "pattern"):
        raw = args.get(key)
        absolute = bool(raw) and (key != "pattern" or Path(str(raw)).is_absolute())
        if absolute and _private(_resolve(str(raw), cwd), cfg):
            return f"{tool}: that path is private to the orchestrator"
    return None


def decide(event: dict[str, Any], cfg: dict[str, Any]) -> str | None:
    """Return why the tool call must be blocked, or None to let it through."""
    tool = str(event.get("tool_name", ""))
    args = event.get("tool_input") or {}
    cwd = str(event.get("cwd") or cfg["worktree"])
    if tool in WRITE_TOOLS:
        return _write_problem(tool, args.get("file_path") or args.get("notebook_path"), cwd, cfg)
    if tool in READ_TOOLS:
        return _read_problem(tool, args, cwd, cfg)
    text = str(args.get("command", "")) if tool == "Bash" else ""
    if text and any(str(Path(p).resolve()) in text or p in text for p in cfg["deny_read"]):
        return "Bash: the command names a path private to the orchestrator"
    return None


def main(argv: list[str], stdin: str) -> int:
    """Decide one hook event; return 0 to allow or 2 to block (reason on stderr)."""
    try:
        cfg = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
        event = json.loads(stdin)
        reason = decide(event, cfg)
    except (OSError, IndexError, KeyError, TypeError, ValueError) as exc:
        reason = f"guard hook cannot decide ({type(exc).__name__}); blocked"
    if reason:
        print(f"orchestrator guard: {reason}", file=sys.stderr)
        return BLOCK
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], sys.stdin.read()))
