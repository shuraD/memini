"""``land`` and the reconciliation of main, the state and the queues (FLOW-HARD.md D7).

A merge outside the normal stage goes through ``land <task> <commit>``: it marks the task merged
in ``state.json`` and comments its queue lines out (``# MERGED <sha> ...``), so the three never
disagree. ``mismatches`` (shown by ``status``) lists every place where they do.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from tools.orchestrator import gitops, state
from tools.orchestrator.config import Config

_TITLE = re.compile(r"^([\w.-]+):(?:\s|$)")


def queue_files(cfg: Config) -> list[Path]:
    """Return every queue file: the default, the work units' and the stages'."""
    found = [
        cfg.queue,
        *(q.queue for q in cfg.code_queues.values()),
        *(s.queue for s in cfg.stages),
    ]
    return list(dict.fromkeys(p for p in found if p.exists()))


def on_main(cfg: Config) -> dict[str, str]:
    """Return ``word -> sha`` for every commit on main titled ``<word>: ...`` (the latest wins).

    A task id is such a word; only ids the state or a queue knows are compared.
    """
    log = gitops.out(cfg.repo, "log", cfg.main_branch, "--format=%H %s")
    found: dict[str, str] = {}
    for line in log.splitlines():
        sha, _, subject = line.partition(" ")
        if (m := _TITLE.match(subject)) and m[1] not in found:
            found[m[1]] = sha
    return found


def _active_ids(path: Path) -> set[str]:
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            out.add(shlex.split(line)[0])
    return out


def land(cfg: Config, task_id: str, commit: str) -> list[Path]:
    """Record that ``commit`` (already on main) merged ``task_id``; return the queues changed."""
    sha = gitops.rev(cfg.repo, commit)
    if not gitops.is_ancestor(cfg.repo, sha, cfg.main_branch):
        msg = f"{commit} is not on {cfg.main_branch}: land records a merge, it does not make one"
        raise ValueError(msg)
    st = state.State(cfg)
    if st.task(task_id) is None:
        st.claim(task_id, "land", ())
    st.update(task_id, "merged", commit=sha, reason=f"landed by hand as {sha[:7]} (land)")
    st.mark_done(task_id)
    changed: list[Path] = []
    for path in queue_files(cfg):
        with state.locked(cfg.state_dir / "locks" / f"queue-{path.name}.lock"):
            lines, hit = path.read_text(encoding="utf-8").splitlines(), False
            for i, line in enumerate(lines):
                if (
                    line.strip()
                    and not line.lstrip().startswith("#")
                    and shlex.split(line)[0] == task_id
                ):
                    lines[i], hit = f"# MERGED {sha[:7]} (land) {line}", True
            if hit:
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                changed.append(path)
    state.ops_log(cfg, task_id, "land", f"merged as {sha[:7]}", queues=[p.name for p in changed])
    return changed


def mismatches(cfg: Config) -> list[str]:
    """Return every disagreement between main, the state and the queues."""
    out: list[str] = []
    titled = on_main(cfg)
    tasks = state.State(cfg).tasks()
    done = set(state.State(cfg).queue_done())
    code = [p for p in queue_files(cfg) if p not in {x.queue for x in cfg.stages}]
    queued = {p: _active_ids(p) for p in code}  # a stage queue keeps its lines as history
    for task_id, sha in sorted(titled.items()):
        status = (tasks.get(task_id) or {}).get("status")
        if task_id in tasks and status != "merged":
            out.append(f"{task_id}: on main as {sha[:7]}, state says {status}")
        out += [
            f"{task_id}: on main as {sha[:7]}, still queued in {p.name}"
            for p, ids in queued.items()
            if task_id in ids and task_id not in done
        ]
    for task_id, rec in sorted(tasks.items()):
        if rec.get("status") == "merged" and task_id not in titled:
            out.append(f"{task_id}: state says merged, main has no commit titled '{task_id}: '")
    return out
