"""The stage chain before ``code`` (owner 2026-11-15: "we can have more than 2 queues").

Every stage has its own queue file, its own task records (``stage-<name>.json``) and its own
loop (``run-queue --stage <name>``). A task that passes a stage's checks is handed to the queue
of the next stage; the last one feeds ``code``, the slice pipeline. Stages run side by side:
while a stage loop runs it holds ``locks/stage-<name>.active``, and the ``code`` loop keeps
waiting for new entries until no earlier stage is active.

A stage queue line is::

    <task id> <module path> <depends_on,...|-> <source,...> <one-line task>

Sources are files the stage agent reads (a relative path is relative to the queue file).
"""

from __future__ import annotations

import contextlib
import json
import shlex
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.orchestrator import schedule, state
from tools.orchestrator.config import CODE_STAGE, Config, StageSpec

ACTIVE, DONE, RELEASED = "active", "done", "released"
_FIELDS = 5  # id, module, depends_on, sources, task text (at least one word)


@dataclass(frozen=True)
class StageItem:
    """One line of a stage queue."""

    task_id: str
    module: str
    depends_on: tuple[str, ...]
    sources: tuple[Path, ...]
    title: str


def read_stage_queue(path: Path) -> list[StageItem]:
    """Return the items of a stage queue in order (an absent file is an empty queue)."""
    if not path.exists():
        return []
    items = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = shlex.split(line)
        if len(parts) < _FIELDS:
            msg = f"{path.name}:{n}: expected '<id> <module> <deps|-> <sources> <task text>'"
            raise ValueError(msg)
        deps = () if parts[2] == "-" else tuple(d for d in parts[2].split(",") if d)
        sources = tuple(
            p if p.is_absolute() else path.parent / p
            for p in (Path(s) for s in parts[3].split(",") if s)
        )
        items.append(StageItem(parts[0], parts[1], deps, sources, " ".join(parts[4:])))
    return items


def stage(cfg: Config, name: str) -> StageSpec:
    """Return the configured stage ``name`` (``ValueError`` when there is none)."""
    for s in cfg.stages:
        if s.name == name:
            return s
    msg = f"unknown stage {name!r} (configured: {', '.join(s.name for s in cfg.stages) or 'none'})"
    raise ValueError(msg)


class StageState:
    """The task records of one stage: status, reason and history, in ``stage-<name>.json``."""

    def __init__(self, cfg: Config, name: str) -> None:
        """Bind to the stage's record file under ``cfg.state_dir``."""
        self.path = cfg.state_dir / f"stage-{name}.json"
        self.lock = cfg.state_dir / f"stage-{name}.lock"

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        return data

    def status(self, task_id: str) -> str | None:
        """Return the task's status in this stage, or None when it never ran here."""
        with state.locked(self.lock):
            rec = self._read().get(task_id)
        return str(rec["status"]) if rec else None

    def records(self) -> dict[str, dict[str, Any]]:
        """Return every record of this stage."""
        with state.locked(self.lock):
            return self._read()

    def set(self, task_id: str, status: str, reason: str = "") -> None:
        """Record a status transition (history is kept, never rewritten)."""
        with state.locked(self.lock):
            data = self._read()
            rec = data.setdefault(task_id, {"history": []})
            rec.update(status=status, reason=reason[:1000])
            at = datetime.now(UTC).isoformat(timespec="seconds")
            rec["history"].append({"status": status, "at": at, "reason": reason[:300]})
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self.path)


def _active_lock(cfg: Config, name: str) -> Path:
    return cfg.state_dir / "locks" / f"stage-{name}.active"


@contextlib.contextmanager
def stage_active(cfg: Config, name: str) -> Iterator[None]:
    """Mark the stage loop ``name`` as running for the body (one loop per stage)."""
    with contextlib.ExitStack() as stack:
        if not state._try_lock(stack, _active_lock(cfg, name)):
            msg = f"a run-queue for stage {name} is already running"
            raise state.BusyError(msg)
        yield


def upstream_active(cfg: Config) -> list[str]:
    """Return the stage loops that are running now (they may still feed ``code``)."""
    out = []
    for s in cfg.stages:
        with contextlib.ExitStack() as stack:
            if not state._try_lock(stack, _active_lock(cfg, s.name)):
                out.append(s.name)
    return out


def handoff(cfg: Config, stage_spec: StageSpec, line: str) -> None:
    """Append ``line`` to the queue of the stage after ``stage_spec`` (once per task id)."""
    target = cfg.queue if stage_spec.next == CODE_STAGE else stage(cfg, stage_spec.next).queue
    task_id = shlex.split(line)[0]
    with state.locked(cfg.state_dir / "locks" / f"queue-{stage_spec.next}.lock"):
        target.parent.mkdir(parents=True, exist_ok=True)
        text = target.read_text(encoding="utf-8") if target.exists() else ""
        if any(shlex.split(ln)[:1] == [task_id] for ln in text.splitlines() if ln.strip()):
            return
        sep = "" if not text or text.endswith("\n") else "\n"
        with target.open("a", encoding="utf-8") as fh:
            fh.write(f"{sep}{line}\n")


def run_stage(
    cfg: Config,
    name: str,
    *,
    workers: int,
    launch: Callable[[StageItem], Any],
    pause: float = 2.0,
) -> int:
    """Run stage ``name``'s queue with up to ``workers`` tasks at once; return the exit code.

    A task starts when its dependencies that are in this queue are done here (an agent may
    read their output). A failed task holds back only its dependents. The queue is re-read
    every round, so lines added while the loop runs are picked up.
    """
    rec = StageState(cfg, name)
    with stage_active(cfg, name):
        running: dict[str, Any] = {}
        failed: set[str] = set()
        while True:
            items = read_stage_queue(stage(cfg, name).queue)
            for task_id, proc in list(running.items()):
                code = proc.poll()
                if code is None:
                    continue
                del running[task_id]
                if code != 0 or rec.status(task_id) != DONE:
                    failed.add(task_id)
                    print(f"{name}/{task_id}: ended with exit {code}; its dependents wait")
            ids = {i.task_id for i in items}
            done = {i.task_id for i in items if rec.status(i.task_id) == DONE}
            blocked = {
                i.task_id for i in items if rec.status(i.task_id) not in (None, DONE, RELEASED)
            }
            cands = [
                schedule.Candidate(i.task_id, (), tuple(d for d in i.depends_on if d in ids))
                for i in items
                if i.task_id not in done | blocked | failed | set(running)
            ]
            by_id = {i.task_id: i for i in items}
            for task_id in schedule.pick(
                cands,
                merged=done,
                running=dict.fromkeys(running, ()),
                free=workers - len(running),
            ):
                print(f"{name}/{task_id}: started")
                running[task_id] = launch(by_id[task_id])
            if not running:
                break
            time.sleep(pause)
    return _exit(name, items, rec, failed)


def _exit(name: str, items: Sequence[StageItem], rec: StageState, failed: set[str]) -> int:
    statuses = {i.task_id: rec.status(i.task_id) for i in items}
    bad = sorted(t for t, s in statuses.items() if s not in (None, DONE, RELEASED)) or sorted(
        failed
    )
    left = [t for t, s in statuses.items() if s in (None, RELEASED) and t not in failed]
    if bad:
        print(f"{name}: failed or blocked: " + ", ".join(bad))
    if left:
        print(f"{name}: waiting on depends_on: " + ", ".join(left))
    if bad:
        return 1
    if left:
        return 3
    print(f"{name}: queue done")
    return 0


def status_lines(cfg: Config) -> list[str]:
    """Return one digest line per stage task (for ``status``)."""
    lines = []
    for s in cfg.stages:
        items = read_stage_queue(s.queue)
        rec = StageState(cfg, s.name)
        records = rec.records()
        lines.append(f"stage {s.name} -> {s.next}: {s.queue} ({len(items)} entries)")
        ids = {i.task_id for i in items}
        for i in items:
            r = records.get(i.task_id)
            status = r["status"] if r else None
            waits = [
                d for d in i.depends_on if d in ids and (records.get(d) or {}).get("status") != DONE
            ]
            if status == DONE:
                label = "done"
            elif status not in (None, RELEASED):
                label = f"{status} ({str(r.get('reason', ''))[:120]})" if r else str(status)
            elif waits:
                label = f"waiting on {', '.join(waits)}"
            else:
                label = "ready"
            lines.append(f"  {i.task_id}: {label}")
    return lines
