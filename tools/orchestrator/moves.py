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

from tools.orchestrator import agent_inbox, gitops, stages, state
from tools.orchestrator.config import Config

MOVES = "moves.jsonl"
MODES = "modes.json"
AUTONOMOUS, SUPERVISED = "autonomous", "supervised"
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


def mode(cfg: Config, role: str) -> str:
    """Return ``autonomous`` (the runner handles the role's moves) or ``supervised`` (a person)."""
    path = cfg.state_dir / MODES
    modes = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return str(modes.get(role, AUTONOMOUS))


def set_mode(cfg: Config, role: str, new: str) -> None:
    """Set the role's mode; a ledger row records every change."""
    if new not in (AUTONOMOUS, SUPERVISED):
        msg = f"mode must be {AUTONOMOUS} or {SUPERVISED}, not {new!r}"
        raise ValueError(msg)
    with state.locked(cfg.state_dir / "modes.lock"):
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        path = cfg.state_dir / MODES
        modes = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        old = modes.get(role, AUTONOMOUS)
        modes[role] = new
        path.write_text(json.dumps(modes), encoding="utf-8")
    state.ops_log(cfg, role, "mode", f"{old} -> {new}", role=role, mode=new)


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
    agent_inbox.notify(cfg, to, f"{task} moved to you by {sender} ({kind}, {move_id}): {text}")
    return move_id


def post_once(
    cfg: Config, task: str, *, sender: str, to: str, kind: str, text: str, **payload: object
) -> str | None:
    """Like ``post``, but nothing is posted while a move of this kind for ``task`` is open."""
    if has_open(cfg, task, kind):
        return None
    return post(cfg, task, sender=sender, to=to, kind=kind, text=text, **payload)


def has_open(cfg: Config, task: str, kind: str) -> bool:
    """Return True when a move of ``kind`` for ``task`` is open."""
    return any(
        r["task"] == task and r["kind"] == kind and r["status"] == "open" for r in _read(cfg)
    )


def for_task(cfg: Config, task: str, kind: str) -> list[Move]:
    """Return every move of ``kind`` for ``task``, oldest first."""
    return [r for r in _read(cfg) if r["task"] == task and r["kind"] == kind]


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
        done_at = datetime.now(UTC).isoformat(timespec="seconds")
        row.update(status="done", result=result, by=by, done_at=done_at)
        _write(cfg, rows)
    state.ops_log(cfg, row["task"], "move_done", f"{by} -> {row['from']}: {result}", move=move_id)
    agent_inbox.notify(
        cfg, row["from"], f"{row['task']} came back from {by} ({row['kind']}, {move_id}): {result}"
    )


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
    if mode(cfg, role) == SUPERVISED:
        return 0  # a person handles this role's moves
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
    why = f"tool fixed ({move_id}): {result}"
    for stage in cfg.stages:  # a stage that stopped on the broken tool takes the task again
        rec = stages.StageState(cfg, stage.name)
        if rec.status(move["task"]) not in (None, stages.ACTIVE, stages.DONE, stages.RELEASED):
            rec.set(move["task"], stages.RELEASED, why)
    st = state.State(cfg)
    if st.task(move["task"]) is not None:
        st.update(move["task"], "released", reason=why)
    return str(move["task"])


def release_answered(cfg: Config) -> list[str]:
    """Release the blocked tasks whose doc review was answered after the block; return their ids.

    A task that a card or SPEC line stopped (the Verifier's partial) waits for the doc-fix flow:
    its ``doc_review`` moves. When the last one is answered the line is fixed, so the task runs
    again by itself: nobody types ``release``. A run that is still going keeps its lock and waits.
    """
    rows = _read(cfg)
    st = state.State(cfg)
    freed = []
    for task, rec in st.tasks().items():
        mine = [r for r in rows if r["task"] == task]
        since = str(rec["history"][-1]["at"])
        if rec["status"] not in ("rejected", "disputed", "failed"):
            continue
        if any(r["status"] == "open" for r in mine):
            continue
        answered = [r for r in mine if r["kind"] == "doc_review" and r.get("done_at", "") >= since]
        if not answered:
            continue
        why = f"doc review answered ({answered[-1]['id']})"
        try:
            with state.run_lock(cfg, task):
                if state.State(cfg).tasks()[task]["status"] not in (
                    "rejected",
                    "disputed",
                    "failed",
                ):
                    continue  # another runner released it first
                st.update(task, "released", reason=why)
        except state.BusyError:
            continue
        gitops.set_aside(cfg.repo, cfg.worktrees_dir, cfg.branch_prefix, task, answered[-1]["id"])
        state.ops_log(cfg, task, "released", why)
        freed.append(task)
    return freed


def parse_verdict(result: str) -> tuple[bool, str]:
    """Read a review answer: JSON ``{"approved", "reason"}``, or text ``approve`` / ``reject: why``.

    A person answering from a session writes the text form. Anything else is a reject.
    """
    try:
        data = json.loads(result)
        return bool(data["approved"]), str(data.get("reason", ""))
    except (ValueError, KeyError, TypeError):
        pass
    text = result.strip()
    if text.lower().startswith("approve"):
        return True, ""
    return False, text.removeprefix("reject:").strip() or "rejected"
