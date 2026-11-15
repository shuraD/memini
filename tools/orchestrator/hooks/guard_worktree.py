"""Claude PreToolUse guard for orchestrator steps: an early warning, never the real gate.

Usage (from the generated settings file)::

    python -P guard_worktree.py <guard-config.json>   # the hook event arrives on stdin

The config names the worktree, the writable path prefixes, the protected entries (they win
over the writable ones) and the private paths no step may read. A write outside the
worktree, to a protected path or outside the writable prefixes is blocked, and so is any
read of a private path, or of a directory that contains one (Grep/Glob ``path: ../..``,
Glob patterns resolved against their path): exit code 2 with the reason on stderr, which
Claude shows the agent. A Bash argument that resolves outside the worktree (or the
configured read roots, e.g. a reference repository) is refused, and so is shell expansion.
Malformed input is blocked too (fail closed). This is an early warning, not the sandbox:
the orchestrator's path gate and fingerprint checks decide.

Standard library only, so it runs with ``python -P`` from any interpreter.
"""

from __future__ import annotations

import json
import shlex
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
    """Return True when ``path`` is inside a private path or contains one (``../..``)."""
    for raw in cfg["deny_read"]:
        private = Path(raw).resolve()
        if _within(path, private) or _within(private, path):
            return True
    return False


_GLOB_CHARS = frozenset("*?[{")


def _static_prefix(pattern: str) -> str:
    """Return the leading path components of a glob pattern that hold no wildcard."""
    parts = []
    for part in Path(pattern).parts:
        if _GLOB_CHARS & set(part):
            break
        parts.append(part)
    return str(Path(*parts)) if parts else "."


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
    base = str(args.get("path") or args.get("file_path") or args.get("notebook_path") or cwd)
    targets = [_resolve(base, cwd)]
    # a Glob pattern (or a Grep glob filter) is resolved against its path, up to its first
    # wildcard: "../../holdout/**" with path "." reaches the hold-out too
    keys = {"Glob": ("pattern",), "Grep": ("glob",)}.get(tool, ())
    targets += [
        _resolve(_static_prefix(str(Path(base) / str(args[k]))), cwd) for k in keys if args.get(k)
    ]
    if any(_private(t, cfg) for t in targets):
        return f"{tool}: that path is private to the orchestrator"
    return None


def _bash_problem(command: str, cwd: str, cfg: dict[str, Any]) -> str | None:
    """Refuse a command whose arguments reach outside the worktree (and its read roots)."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return "Bash: the command cannot be parsed"
    roots = [
        Path(cfg["worktree"]).resolve(),
        *(Path(r).resolve() for r in cfg.get("bash_roots", [])),
    ]
    for token in tokens:
        if "$" in token or "`" in token:
            return "Bash: shell expansion is not allowed in orchestrator steps"
        for candidate in {token, token.split("=", 1)[-1]}:
            if not candidate:
                continue
            path = _resolve(candidate, cwd)
            if _private(path, cfg) or not any(_within(path, r) for r in roots):
                return f"Bash: argument {candidate!r} is outside the task worktree"
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
    if tool == "Bash":
        return _bash_problem(str(args.get("command", "")), cwd, cfg)
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
