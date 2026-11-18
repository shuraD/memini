"""Tasks that move between role queues by themselves (FLOW-HARD.md D-10).

A move is one JSON line in ``<state_dir>/moves.jsonl``: the task, who sends it, who gets it, the
kind (``test_review``, ``dispute``, ``tool``), the payload the receiver needs, and its status
(``open`` until the receiver resolves it with a result). Posting and resolving each write an ops
ledger row, so the ledger shows every move. Nobody carries a message: a role's runner (``serve``)
takes the open moves addressed to it, and a waiting task (``wait``) reads the result.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from tools.orchestrator import gitops, state
from tools.orchestrator.config import Config

MOVES = "moves.jsonl"
Move = dict[str, Any]
Handler = Callable[[Config, Move], str]


def _read(cfg: Config) -> list[Move]:
    path = cfg.state_dir / MOVES
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write(cfg: Config, rows: list[Move]) -> None:
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (cfg.state_dir / MOVES).write_text(text, encoding="utf-8")


def post(
    cfg: Config, task: str, *, sender: str, to: str, kind: str, text: str, **payload: object
) -> str:
    """Send ``task`` to the queue of role ``to``; return the move id."""
    with state.locked(cfg.state_dir / "moves.lock"):
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        rows = _read(cfg)
        move_id = f"MV-{len(rows) + 1:04d}"
        at = datetime.now(UTC).isoformat(timespec="seconds")
        rows.append(
            {
                "id": move_id,
                "task": task,
                "from": sender,
                "to": to,
                "kind": kind,
                "text": text,
                "payload": payload,
                "status": "open",
                "at": at,
            }
        )
        _write(cfg, rows)
    state.ops_log(cfg, task, "move", f"{sender} -> {to}: {kind}: {text}", move=move_id)
    return move_id


def get(cfg: Config, move_id: str) -> Move:
    """Return the move ``move_id`` (KeyError when unknown)."""
    for row in _read(cfg):
        if row["id"] == move_id:
            return row
    raise KeyError(move_id)


def open_for(cfg: Config, role: str) -> list[Move]:
    """Return the open moves addressed to ``role``, oldest first."""
    return [r for r in _read(cfg) if r["to"] == role and r["status"] == "open"]


def resolve(cfg: Config, move_id: str, result: str, by: str) -> None:
    """Close a move with ``result``; the task goes back to where it came from."""
    with state.locked(cfg.state_dir / "moves.lock"):
        rows = _read(cfg)
        row = next(r for r in rows if r["id"] == move_id)
        row.update(status="done", result=result, by=by)
        _write(cfg, rows)
    state.ops_log(cfg, row["task"], "move_done", f"{by} -> {row['from']}: {result}", move=move_id)


def wait(cfg: Config, move_id: str, timeout_s: float, poll_s: float = 5.0) -> Move | None:
    """Wait until the move is resolved; return it, or None after ``timeout_s``."""
    deadline = time.monotonic() + timeout_s
    while True:
        row = get(cfg, move_id)
        if row["status"] == "done":
            return row
        if time.monotonic() >= deadline:
            return None
        time.sleep(poll_s)


def serve(
    cfg: Config,
    role: str,
    handlers: Mapping[str, Handler],
    *,
    once: bool = False,
    poll_s: float = 5.0,
) -> int:
    """Handle the open moves of ``role`` by kind; loop until none is left when ``once``."""
    done = 0
    while True:
        batch = [m for m in open_for(cfg, role) if m["kind"] in handlers]
        for move in batch:
            resolve(cfg, move["id"], handlers[move["kind"]](cfg, move), role)
            done += 1
        if once and not batch:
            return done
        if not batch:
            time.sleep(poll_s)


def finish_tool(cfg: Config, move_id: str, result: str, by: str) -> str:
    """Close a ``tool`` move when the tool is fixed: the waiting task is released to run again."""
    move = get(cfg, move_id)
    if move["kind"] != "tool":
        msg = f"{move_id} is a {move['kind']} move, not a tool move"
        raise ValueError(msg)
    resolve(cfg, move_id, result, by)
    moved = gitops.set_aside(cfg.repo, cfg.worktrees_dir, cfg.branch_prefix, move["task"], move_id)
    state.ops_log(
        cfg, move["task"], "set_aside", "leftovers of the stopped run renamed", moved=moved
    )
    st = state.State(cfg)
    if st.task(move["task"]) is not None:
        st.update(move["task"], "released", reason=f"tool fixed ({move_id}): {result}")
    return str(move["task"])
