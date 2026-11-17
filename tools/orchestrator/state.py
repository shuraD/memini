"""Local orchestrator state, the owner inbox and the task queue (all outside the repo).

The state is one JSON file under ``state_dir``, read and written under an exclusive file
lock. A task record is never deleted: every status change is appended to its history.
WIP rule: one active task per worker, and no two active tasks with overlapping paths.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import shlex
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.orchestrator.config import Config

ACTIVE = "active"
DONE_STATUSES = frozenset({"merged"})


class BusyError(RuntimeError):
    """Another orchestrator run holds the global run lock."""


@contextlib.contextmanager
def run_lock(cfg: Config, task_id: str = "") -> Iterator[None]:
    """Hold a run slot for ``task_id``; a process that gets none gets :class:`BusyError` at once.

    With the local executor (or ``max_parallel`` 1) there is one slot: one orchestrator run at a
    time. With the docker executor up to ``max_parallel`` runs hold a slot at once, and a task
    that is already running is refused.
    """
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        if cfg.parallel and task_id:
            task_lock = cfg.state_dir / "locks" / f"task-{task_id}.lock"
            if not _try_lock(stack, task_lock):
                msg = f"{task_id} is already running"
                raise BusyError(msg)
            slots = [cfg.state_dir / "locks" / f"slot-{i}.lock" for i in range(cfg.max_parallel)]
            if not any(_try_lock(stack, slot) for slot in slots):
                msg = f"all {cfg.max_parallel} run slots are busy"
                raise BusyError(msg)
        elif not _try_lock(stack, cfg.state_dir / "run.lock"):
            msg = "another orchestrator run is active: one run at a time (global run lock)"
            raise BusyError(msg)
        yield


@contextlib.contextmanager
def task_lock(cfg: Config, name: str) -> Iterator[None]:
    """Hold ``locks/<name>.lock`` for one stage task; no run slot (stages run beside ``code``)."""
    with contextlib.ExitStack() as stack:
        if not _try_lock(stack, cfg.state_dir / "locks" / f"{name}.lock"):
            msg = f"{name} is already running"
            raise BusyError(msg)
        yield


def _try_lock(stack: contextlib.ExitStack, path: Path) -> bool:
    """Take a non-blocking exclusive lock on ``path``; it is released when ``stack`` closes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = stack.enter_context(path.open("a+", encoding="utf-8"))
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    stack.callback(fcntl.flock, fh, fcntl.LOCK_UN)
    return True


class WipConflictError(RuntimeError):
    """The task would break the WIP rule (worker busy or paths overlap)."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _covers(a: str, b: str) -> bool:
    return a == b or (a.endswith("/") and b.startswith(a))


def paths_overlap(a: Sequence[str], b: Sequence[str]) -> bool:
    """Return True when a path of ``a`` contains, equals or lies inside a path of ``b``."""
    return any(_covers(x, y) or _covers(y, x) for x in a for y in b)


@contextlib.contextmanager
def locked(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` (created if needed) for the body."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


class State:
    """The task records and the queue progress of one orchestrator installation."""

    def __init__(self, cfg: Config) -> None:
        """Bind to the state file under ``cfg.state_dir``."""
        self.path = cfg.state_dir / "state.json"
        self.lock = cfg.state_dir / "state.lock"

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"tasks": {}, "queue_done": []}
        data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        return data

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def task(self, task_id: str) -> dict[str, Any] | None:
        """Return the record of ``task_id``, or None."""
        with locked(self.lock):
            rec: dict[str, Any] | None = self._read()["tasks"].get(task_id)
            return rec

    def tasks(self) -> dict[str, dict[str, Any]]:
        """Return every task record."""
        with locked(self.lock):
            recs: dict[str, dict[str, Any]] = self._read()["tasks"]
            return recs

    def is_merged(self, task_id: str) -> bool:
        """Return True when the task merged (a done queue entry counts)."""
        with locked(self.lock):
            data = self._read()
        rec = data["tasks"].get(task_id)
        return task_id in data["queue_done"] or bool(rec and rec["status"] in DONE_STATUSES)

    def task_ids(self) -> set[str]:
        """Return the ids of every task ever recorded."""
        return set(self.tasks())

    def overlaps(self, paths: Sequence[str], task_id: str) -> list[str]:
        """Return a problem per active task (other than ``task_id``) whose paths overlap."""
        return [
            f"paths overlap the active task {other}: {', '.join(rec['paths'])}"
            for other, rec in sorted(self.tasks().items())
            if other != task_id and rec["status"] == ACTIVE and paths_overlap(paths, rec["paths"])
        ]

    def claim(self, task_id: str, worker: str, paths: Sequence[str], **info: str) -> None:
        """Record ``task_id`` as active, or raise :class:`WipConflictError`."""
        with locked(self.lock):
            data = self._read()
            for other, rec in data["tasks"].items():
                if rec["status"] != ACTIVE:
                    continue
                if other == task_id:
                    msg = f"task {task_id} is already active"
                    raise WipConflictError(msg)
                if rec["worker"] == worker:
                    msg = f"worker {worker} already runs {other} (WIP = 1 per worker)"
                    raise WipConflictError(msg)
                if paths_overlap(paths, rec["paths"]):
                    msg = f"paths overlap the active task {other}: {', '.join(rec['paths'])}"
                    raise WipConflictError(msg)
            prev = data["tasks"].get(task_id, {})
            if prev.get("status") in DONE_STATUSES:
                msg = f"task {task_id} is already {prev['status']}"
                raise WipConflictError(msg)
            history = [*prev.get("history", []), {"status": ACTIVE, "at": _now()}]
            data["tasks"][task_id] = {
                **info,
                "status": ACTIVE,
                "worker": worker,
                "paths": list(paths),
                "history": history,
            }
            self._write(data)

    def update(self, task_id: str, status: str | None = None, **info: str) -> None:
        """Record ``info`` on the task and, with ``status``, a status transition."""
        with locked(self.lock):
            data = self._read()
            rec = data["tasks"][task_id]
            rec.update(info)
            if status:
                rec["status"] = status
                rec["history"].append({"status": status, "at": _now(), **info})
            self._write(data)

    def queue_done(self) -> list[str]:
        """Return the queue entries finished so far, in order."""
        with locked(self.lock):
            done: list[str] = self._read()["queue_done"]
            return done

    def mark_done(self, task_id: str) -> None:
        """Mark a queue entry done."""
        with locked(self.lock):
            data = self._read()
            if task_id not in data["queue_done"]:
                data["queue_done"].append(task_id)
            self._write(data)


# ------------------------------------------------------------------ inbox
def _one_line(text: str, limit: int = 2000) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def inbox_append(
    cfg: Config, task_id: str, kind: str, tier: int, text: str, *, limit: int = 2000
) -> None:
    """Append an entry for the owner channel (Markdown) to the inbox file."""
    with locked(cfg.state_dir / "inbox.lock"):
        cfg.inbox.parent.mkdir(parents=True, exist_ok=True)
        with cfg.inbox.open("a", encoding="utf-8") as fh:
            fh.write(
                f"\n## {_now()} {task_id}: {kind} (tier {tier})\n\n- {_one_line(text, limit)}\n"
            )


# ------------------------------------------------------------------ queue
@dataclass(frozen=True)
class QueueEntry:
    """One queue line: ``<task id> <spec file> [engine|-] [path,path,...]``."""

    task_id: str
    spec: Path
    engine: str | None
    paths: tuple[str, ...]


def read_queue(path: Path) -> list[QueueEntry]:
    """Return the queue entries in order; a spec path is relative to the queue file."""
    entries = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = shlex.split(line)
        if not 2 <= len(parts) <= 4:  # noqa: PLR2004 — id, spec, engine, paths
            msg = f"{path.name}:{n}: expected '<task id> <spec> [engine|-] [paths]'"
            raise ValueError(msg)
        engine = parts[2] if len(parts) > 2 and parts[2] != "-" else None  # noqa: PLR2004
        paths = tuple(p for p in parts[3].split(",") if p) if len(parts) > 3 else ()  # noqa: PLR2004
        spec = Path(parts[1])
        entries.append(
            QueueEntry(parts[0], spec if spec.is_absolute() else path.parent / spec, engine, paths)
        )
    return entries


OPS_LEDGER = "ops-ledger.jsonl"


def ops_log(cfg: Config, task_id: str, action: str, reason: str, **details: object) -> None:
    """Append one line to the ops ledger (``<state_dir>/ops-ledger.jsonl``): who, what, why."""
    record = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "actor": "orchestrator",
        "action": action,
        "target": task_id,
        "reason": _one_line(reason)[:500],
        "details": details,
    }
    with locked(cfg.state_dir / "ops.lock"):
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        with (cfg.state_dir / OPS_LEDGER).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
