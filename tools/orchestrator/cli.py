"""Command line of the orchestrator (see ``--help``)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.orchestrator import (
    config,
    dockerexec,
    enqueue,
    gates,
    gitops,
    metrics,
    moves,
    pipeline,
    planner,
    planstage,
    reconcile,
    schedule,
    spec,
    specstage,
    stages,
    state,
)
from tools.orchestrator.executor import clean_env

EPILOG = f"""\
configuration:
  defaults: {config.DEFAULTS.name} next to this tool (tools/orchestrator/{config.DEFAULTS.name})
  your file: --config <file.toml> or MEMINI_ORCH_CONFIG=<file.toml> (merged over the defaults)
  single paths: MEMINI_ORCH_STATE_DIR, MEMINI_ORCH_HOLDOUT_DIR, MEMINI_ORCH_WORKTREES_DIR,
                MEMINI_ORCH_INBOX, MEMINI_ORCH_QUEUE
  queue line:  <task id> <spec file> [engine|-] [path,path,...]   ('#' starts a comment)
  stage line:  <task id> <module path> <deps|-> <source,...> <one-line task>   (stage queues)
  stages:      run-queue --stage <name> runs one stage, --all runs every stage and code side
               by side; a task that passes a stage moves into the next stage's queue

exit codes: 0 done/merged, 1 rejected, disputed, failed or blocked, 2 usage or config error,
            3 run-queue: entries left that wait on depends_on
"""
_EXIT_WAITING = 3
_UPSTREAM_GRACE_S = 120  # how long the code loop waits for a stage loop that has not started


def _upstream_busy(cfg: config.Config, wait: bool, started: float) -> bool:
    """Return True while a stage loop runs, or (``wait``) stage tasks are still pending (D5)."""
    if stages.upstream_active(cfg):
        return True
    return (
        wait
        and time.monotonic() - started < _UPSTREAM_GRACE_S
        and bool(stages.upstream_pending(cfg))
    )


def _queue_entries(cfg: config.Config) -> list[state.QueueEntry]:
    return state.read_queue(cfg.queue) if cfg.queue.exists() else []


def _repo() -> Path:
    git = shutil.which("git") or "git"
    res = subprocess.run(  # noqa: S603 — fixed git args, no shell
        [git, "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        env=clean_env(),
        check=False,
    )
    return Path(res.stdout.strip() or ".").resolve()


def _config(args: argparse.Namespace) -> config.Config:
    overrides: dict[str, Any] = json.loads(args.config_json) if args.config_json else {}
    return config.load_config(_repo(), config_file=args.config, overrides=overrides)


def _report(outcome: pipeline.Outcome) -> int:
    print(f"{outcome.task_id}: {outcome.status}: {outcome.reason}")
    if outcome.worktree is not None:
        print(f"{outcome.task_id}: worktree left for inspection: {outcome.worktree}")
    return 0 if outcome.status == pipeline.MERGED else 1


def _cmd_run(cfg: config.Config, args: argparse.Namespace) -> int:
    task = pipeline.Task(
        task_id=args.task_id,
        spec=Path(args.spec),
        engine=args.engine,
        paths=tuple(args.paths or ()),
        worker=args.worker,
    )
    return _report(pipeline.run_task(cfg, task))


def _done(st: state.State, task_id: str) -> bool:
    return st.is_merged(task_id)


def _deps(entry: state.QueueEntry) -> tuple[str, ...]:
    try:
        return spec.load(entry.spec).depends_on
    except spec.SpecError:
        return ()  # run_task refuses the spec and reports it


HANDLERS: dict[str, dict[str, moves.Handler]] = {
    "qa": {"test_review": pipeline.review_move},
    "planner": {"dispute": planner.reissue},
}  # role -> move kind -> handler


def _apply_role(cfg: config.Config, args: argparse.Namespace) -> int | None:
    """Point ``args`` at the queue of ``--role`` and its worker count.

    Returns the exit code when the role has no queue to run (2 unknown; planner and devops serve
    the moves sent to them), None when the code or stage loop should run.
    """
    role = cfg.roles.get(args.role)
    if role is None:
        print(f"unknown role {args.role!r} (configured: {', '.join(cfg.roles)})")
        return 2
    if not role.queue:
        moves.serve(cfg, role.name, HANDLERS.get(role.name, {}), once=True)
        return 0
    args.workers = role.workers
    if role.queue in cfg.code_queues:
        args.queue, args.wait_upstream = role.queue, True
    elif role.queue:
        args.stage = role.queue
    return None


def _cmd_run_queue(cfg: config.Config, args: argparse.Namespace) -> int:
    if args.role and (code := _apply_role(cfg, args)) is not None:
        return code
    if args.queue:
        if args.queue not in cfg.code_queues:
            print(f"unknown code queue {args.queue!r} (configured: {', '.join(cfg.code_queues)})")
            return 2
        cfg = dataclasses.replace(cfg, queue=cfg.code_queues[args.queue].queue)
    if args.all:
        return _run_all(cfg, args)
    if args.stage != config.CODE_STAGE:
        return stages.run_stage(
            cfg,
            stages.stage(cfg, args.stage).name,
            workers=max(1, args.workers),
            launch=_stage_launcher(args, args.stage),
        )
    workers = min(args.workers, cfg.max_parallel)
    if workers > 1 and cfg.parallel:
        return run_queue_parallel(
            cfg, workers=workers, launch=_launcher(args, cfg), wait_upstream=args.wait_upstream
        )
    if args.workers > 1:
        print(
            "run-queue: parallel runs need the docker executor and max_parallel > 1; one at a time"
        )
    return _run_queue_serial(cfg, args)


def _entry_paths(entry: state.QueueEntry) -> tuple[str, ...]:
    if entry.paths:
        return entry.paths
    try:
        return spec.load(entry.spec).paths
    except spec.SpecError:
        return ()  # run_task refuses the spec and reports it


def run_queue_parallel(
    cfg: config.Config,
    *,
    workers: int,
    launch: Callable[[state.QueueEntry], Any],
    pause: float = 2.0,
    wait_upstream: bool = False,
) -> int:
    """Run the queue with up to ``workers`` slices at once; return the exit code.

    A slice starts when its dependencies are merged and its paths overlap no running slice. A
    failed or blocked slice holds back only the slices that depend on it (owner 2026-11-15: "it
    should be able to work alone"); the run reports it at the end and exits 1.
    """
    st = state.State(cfg)
    started = time.monotonic()
    entries = _queue_entries(cfg)
    seen = {e.task_id for e in entries}
    failed = _blocked(st, entries)  # only the slices that depend on them wait
    running: dict[str, Any] = {}
    while True:
        # an earlier stage may hand new entries over while this loop runs (stage chain)
        entries = state.read_queue(cfg.queue) if cfg.queue.exists() else entries
        new = [e for e in entries if e.task_id not in seen]
        failed |= _blocked(st, new)
        seen |= {e.task_id for e in new}
        by_id = {e.task_id: e for e in entries}
        _reap(st, running, failed)
        merged = {e.task_id for e in entries if _done(st, e.task_id)}
        # a dependency may be merged from another work unit's queue (GS-22 needs LP-08)
        merged |= {d for e in entries for d in _deps(e) if _done(st, d)}
        cands = [
            schedule.Candidate(e.task_id, _entry_paths(e), _deps(e))
            for e in entries
            if e.task_id not in merged
            and e.task_id not in running
            and e.task_id not in failed
            and _holdout_ready(cfg, e.task_id)
        ]
        busy = {t: _entry_paths(by_id[t]) for t in running}
        for task_id in schedule.pick(
            cands, merged=merged, running=busy, free=workers - len(running)
        ):
            print(f"{task_id}: started")
            running[task_id] = launch(by_id[task_id])
        if not running and not _upstream_busy(cfg, wait_upstream, started):
            break
        time.sleep(pause)
    return _batch_done(cfg, entries, _queue_exit(st, entries, failed))


def _stage_launcher(args: argparse.Namespace, stage_name: str) -> Callable[[stages.StageItem], Any]:
    """Return a function that starts one ``stage-run`` of a stage item as its own process."""
    base = [sys.executable, "-m", "tools.orchestrator"]
    if args.config:
        base += ["--config", str(args.config)]
    if args.config_json:
        base += ["--config-json", args.config_json]

    def launch(item: stages.StageItem) -> Any:
        argv = [*base, "stage-run", stage_name, item.task_id]
        return subprocess.Popen(argv, cwd=_repo())  # noqa: S603 — our own CLI, no shell

    return launch


def _run_all(cfg: config.Config, args: argparse.Namespace) -> int:
    """Run every stage loop and the code loop side by side; return the worst exit code."""
    base = [sys.executable, "-m", "tools.orchestrator"]
    if args.config:
        base += ["--config", str(args.config)]
    if args.config_json:
        base += ["--config-json", args.config_json]
    procs = []
    for name in [*(s.name for s in cfg.stages), config.CODE_STAGE]:
        argv = [*base, "run-queue", "--stage", name, "--workers", str(args.workers)]
        if name == config.CODE_STAGE:
            argv.append("--wait-upstream")
        if args.worker:
            argv += ["--worker", args.worker]
        procs.append(subprocess.Popen(argv, cwd=_repo()))  # noqa: S603 — our own CLI
        if name != config.CODE_STAGE:
            time.sleep(1)  # the stage loop takes its active lock before code looks for it
    codes = [p.wait() for p in procs]
    return 1 if 1 in codes else max(codes)


def _cmd_enqueue(cfg: config.Config, args: argparse.Namespace) -> int:
    fix = bool(args.change)
    if fix == bool(args.cards):
        print("enqueue: give --cards, or --change with --owner-words (a structured fix)")
        return 2
    text = ""
    if fix:
        text = enqueue.fix_text(
            cfg, args.change, args.remove_names, args.rename, args.owner_words or ""
        )
    _, found = enqueue.enqueue(
        cfg,
        task_id=args.task_id,
        module=args.file,
        card_list=args.cards.split(",") if args.cards else [],
        text=text,
        deps=[d for d in args.deps.split(",") if d],
        sources=[s for s in args.sources.split(",") if s],
        needs=[n for n in args.needs.split(",") if n],
    )
    for problem in found:
        print(f"{args.task_id}: refused: {problem}")
    if not found:
        queued = enqueue.read_tickets(cfg.stages[0].queue)
        print(f"{args.task_id}: queued; workers for the batch: {enqueue.workers(queued)}")
    return 1 if found else 0


def _cmd_metrics(cfg: config.Config, args: argparse.Namespace) -> int:
    ids = [i for i in args.ids.split(",") if i] or [e.task_id for e in _queue_entries(cfg)]
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    print(metrics.write(cfg, ids, when))
    return 0


def _cmd_plan(cfg: config.Config, args: argparse.Namespace) -> int:
    for line in planner.plan(cfg, Path(args.brief)):
        print(line)
    return 0


def _cmd_ready(cfg: config.Config, args: argparse.Namespace) -> int:
    """Run the ready gate over the tickets of a stage queue (default: the first stage)."""
    queue = Path(args.queue) if args.queue else cfg.stages[0].queue
    tickets = enqueue.read_tickets(queue)
    wanted = set(args.ids)
    bad = 0
    for i, ticket in enumerate(tickets):
        if wanted and ticket.id not in wanted:
            continue
        found = enqueue.problems(cfg, ticket, tickets[:i])
        print(f"{ticket.id}: " + ("ready" if not found else "refused"))
        for problem in found:
            print(f"  - {problem}")
        bad += bool(found)
        enqueue.tell_devops(cfg, ticket, found)
        state.ops_log(
            cfg, ticket.id, "ready_gate", "refused: " + "; ".join(found) if found else "ready"
        )
    print(f"workers for the batch: {enqueue.workers(tickets)}")
    return 1 if bad else 0


def _cmd_move(cfg: config.Config, args: argparse.Namespace) -> int:
    move_id = moves.post(
        cfg,
        args.task_id,
        sender=args.sender,
        to=args.to,
        kind=args.kind,
        text=args.text,
        owner_words=args.owner_words or "",
    )
    print(f"{args.task_id}: moved to {args.to} as {move_id}")
    return 0


def _cmd_serve_moves(cfg: config.Config, args: argparse.Namespace) -> int:
    done = moves.serve(cfg, args.role, HANDLERS.get(args.role, {}), once=True)
    print(f"{args.role}: {done} move(s) served")
    return 0


def _cmd_move_done(cfg: config.Config, args: argparse.Namespace) -> int:
    task = moves.finish_tool(cfg, args.move_id, args.result, "devops")
    print(f"{task}: released; the squad's runner takes it again ({args.move_id} closed)")
    return 0


def _cmd_land(cfg: config.Config, args: argparse.Namespace) -> int:
    changed = reconcile.land(cfg, args.task_id, args.commit)
    print(f"{args.task_id}: merged; queues updated: {', '.join(p.name for p in changed) or 'none'}")
    return 0


def _cmd_stage_run(cfg: config.Config, args: argparse.Namespace) -> int:
    stage_spec = stages.stage(cfg, args.stage)
    job = planstage.PlanJob if stage_spec.kind == "plan" else None
    return specstage.run(cfg, stage_spec, args.task_id, job)


def _holdout_ready(cfg: config.Config, task_id: str) -> bool:
    """Return False while a required hold-out is not written yet: the slice waits, unrecorded."""
    return not cfg.holdout_required or gates.has_holdout(cfg, task_id)


def _blocked(st: state.State, entries: Sequence[state.QueueEntry]) -> set[str]:
    """Return the queue entries that failed in an earlier run and were not released."""
    out = set()
    for entry in entries:
        rec = st.task(entry.task_id)
        if rec and not _done(st, entry.task_id) and rec["status"] not in ("released",):
            print(f"{entry.task_id}: blocked ({rec['status']}): {rec.get('reason', '')}")
            out.add(entry.task_id)
    return out


def _reap(st: state.State, running: dict[str, Any], failed: set[str]) -> None:
    """Collect finished runs: a merged one is marked done, any other one is failed."""
    for task_id, proc in list(running.items()):
        code = proc.poll()
        if code is None:
            continue
        del running[task_id]
        if code == 0 and _done(st, task_id):
            st.mark_done(task_id)
        else:
            failed.add(task_id)
            print(f"{task_id}: run ended with exit {code}; its dependents wait")


def _batch_done(cfg: config.Config, entries: Sequence[state.QueueEntry], code: int) -> int:
    """After a finished batch, write its metrics into STATUS.md (D10); keep the exit code."""
    if code == 0 and entries:
        stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
        try:
            metrics.write(cfg, [e.task_id for e in entries], stamp)
        except OSError as exc:  # a missing STATUS.md folder never fails a finished batch
            print(f"metrics not written: {exc}")
    return code


def _queue_exit(st: state.State, entries: Sequence[state.QueueEntry], failed: set[str]) -> int:
    """Print the queue summary; return 1 on any failure, 3 when entries wait, else 0."""
    left = [e.task_id for e in entries if not _done(st, e.task_id) and e.task_id not in failed]
    if failed:
        print("failed or blocked: " + ", ".join(sorted(failed)))
    if left:
        print("waiting on depends_on: " + ", ".join(left))
    if failed:
        return 1
    if left:
        return _EXIT_WAITING
    print("queue done")
    return 0


def _launcher(args: argparse.Namespace, cfg: config.Config) -> Callable[[state.QueueEntry], Any]:
    """Return a function that starts one ``run`` of an entry as its own process.

    Each run is its own worker (``<worker>-<task id>``): the one-task-per-worker rule then
    holds per run, and parallel runs do not refuse each other.
    """
    base = [sys.executable, "-m", "tools.orchestrator"]
    if args.config:
        base += ["--config", str(args.config)]
    if args.config_json:
        base += ["--config-json", args.config_json]

    def launch(entry: state.QueueEntry) -> Any:
        argv = [*base, "run", entry.task_id, "--spec", str(entry.spec)]
        if entry.engine:
            argv += ["--engine", entry.engine]
        if entry.paths:
            argv += ["--paths", *entry.paths]
        argv += ["--worker", f"{args.worker or cfg.worker}-{entry.task_id}"]
        return subprocess.Popen(argv, cwd=_repo())  # noqa: S603 — our own CLI, no shell

    return launch


def _run_queue_serial(cfg: config.Config, args: argparse.Namespace) -> int:
    st = state.State(cfg)
    started = time.monotonic()
    entries = _queue_entries(cfg)
    ids = {e.task_id for e in entries}
    while True:
        progress, waiting = False, []
        for entry in entries:
            if _done(st, entry.task_id):
                if entry.task_id not in st.queue_done():
                    st.mark_done(entry.task_id)
                continue
            rec = st.task(entry.task_id)
            if rec and rec["status"] not in ("released",):
                print(f"{entry.task_id}: blocked ({rec['status']}): {rec.get('reason', '')}")
                return 1
            missing = [d for d in _deps(entry) if not _done(st, d)]
            if missing:
                waiting.append(f"{entry.task_id} (waits on {', '.join(missing)})")
                continue
            task = pipeline.Task(entry.task_id, entry.spec, entry.engine, entry.paths, args.worker)
            outcome = pipeline.run_task(cfg, task, known_ids=ids)
            if _report(outcome):
                return 1
            st.mark_done(entry.task_id)
            progress = True
        if not progress:
            if not _upstream_busy(cfg, args.wait_upstream, started):
                break
            time.sleep(2)  # a stage may still hand a task over (D5)
            entries = _queue_entries(cfg)
            ids = {e.task_id for e in entries}
    if waiting:
        print("waiting on depends_on: " + "; ".join(waiting))
        return _EXIT_WAITING
    print("queue done")
    return _batch_done(cfg, _queue_entries(cfg), 0)


def _cmd_check_spec(cfg: config.Config, args: argparse.Namespace) -> int:
    st = state.State(cfg)
    known = st.task_ids()
    if cfg.queue.exists():
        known |= {e.task_id for e in state.read_queue(cfg.queue)}
    bad = 0
    for path in args.specs:
        try:
            sp = spec.load(Path(path))
            problems = pipeline.precheck(cfg, sp, sp.task_id, (), known)
        except spec.SpecError as exc:
            problems = [str(exc)]
        for problem in problems:
            print(f"{path}: {problem}")
        if not problems:
            print(f"{path}: ok")
        bad += bool(problems)
    return 1 if bad else 0


_INBOX_HEAD = re.compile(r"^## (.+)$", re.MULTILINE)


def _entry_label(st: state.State, tasks: dict[str, Any], e: state.QueueEntry) -> str:
    rec = tasks.get(e.task_id)
    missing = [d for d in _deps(e) if not _done(st, d)]
    if _done(st, e.task_id):
        return "done"
    if rec and rec["status"] not in ("released",):
        label = {"disputed": "disputed", "active": "active"}.get(rec["status"], "blocked")
        return label + f" ({rec['status']}: {str(rec.get('reason', ''))[:120]})"
    if missing:
        return f"waiting on {', '.join(missing)}"
    return "ready"


def status_lines(cfg: config.Config) -> list[str]:
    """Return the plain-text digest of the queue, the task states and the inbox."""
    st = state.State(cfg)
    tasks = st.tasks()
    lines = []
    squads = [(n, q.queue) for n, q in cfg.code_queues.items()]
    # the default queue is listed once: under its squad name when a squad reads it
    queues = squads if any(q == cfg.queue for _, q in squads) else [("queue", cfg.queue), *squads]
    seen: set[str] = set()
    for name, path in queues:
        entries = state.read_queue(path) if path.exists() else []
        seen |= {e.task_id for e in entries}
        label_name = "queue" if name == "queue" else f"code queue {name}"
        lines.append(f"{label_name}: {path} ({len(entries)} entries)")
        lines += [f"  {e.task_id}: {_entry_label(st, tasks, e)}" for e in entries]
    for task_id in sorted(set(tasks) - seen):
        lines.append(f"  {task_id}: {tasks[task_id]['status']} (not in the queue)")
    lines += stages.status_lines(cfg)
    heads = _INBOX_HEAD.findall(cfg.inbox.read_text(encoding="utf-8")) if cfg.inbox.exists() else []
    lines.append(f"inbox: {cfg.inbox} ({len(heads)} entries)")
    lines += [f"  {h}" for h in heads]
    return lines


def _cmd_status(cfg: config.Config, args: argparse.Namespace) -> int:
    print("\n".join(status_lines(cfg)))
    try:
        found = reconcile.mismatches(cfg)
    except gitops.GitError as exc:
        found = [f"cannot compare with {cfg.main_branch}: {exc}"]
    print("main/state/queue mismatches: " + ("none" if not found else f"{len(found)}"))
    for line in found:
        print(f"  - {line}")
    return 0


def _cmd_sandbox_pytest(cfg: config.Config, args: argparse.Namespace) -> int:
    """Run pytest on the current directory inside a sandbox container (never on the host).

    This is what the coder's ``Bash(... sandbox-pytest ...)`` rule allows, so the agent CLI
    never runs coder-written code on the host. With executor = local it is refused.
    """
    if cfg.executor != "docker":
        print("sandbox-pytest needs executor = docker (coder code must not run on the host)")
        return 2
    sandbox = dockerexec.build_executor(cfg, cfg.env_allow)
    argv = ["uv", "run", "python", "-P", "-m", "pytest", "-q", "-o", "addopts=", *args.pytest_args]
    res = sandbox.run(argv, cwd=Path.cwd(), timeout=cfg.docker_run_timeout_s)
    print(res.output, end="")
    return res.code


def _set_aside(cfg: config.Config, task_id: str) -> None:
    """Rename the leftovers of the released task's stopped run so that it can run again."""
    tag = datetime.now(UTC).strftime("rel%H%M%S")
    moved = gitops.set_aside(cfg.repo, cfg.worktrees_dir, cfg.branch_prefix, task_id, tag)
    if moved:
        state.ops_log(
            cfg, task_id, "set_aside", "leftovers of the released run renamed", moved=moved
        )


def _cmd_release(cfg: config.Config, args: argparse.Namespace) -> int:
    if args.stage != config.CODE_STAGE:
        rec = stages.StageState(cfg, stages.stage(cfg, args.stage).name)
        try:
            with state.task_lock(cfg, f"{args.stage}-{args.task_id}"):
                if rec.status(args.task_id) is None:
                    print(f"{args.stage}/{args.task_id}: unknown task")
                    return 1
                rec.set(args.task_id, stages.RELEASED, args.reason)
                _set_aside(cfg, args.task_id)
        except state.BusyError as exc:
            print(f"{args.stage}/{args.task_id}: not released: {exc}")
            return 1
        print(f"{args.stage}/{args.task_id}: released")
        return 0
    st = state.State(cfg)
    try:
        with state.run_lock(cfg, args.task_id):  # never release a task while it runs
            if st.task(args.task_id) is None:
                print(f"{args.task_id}: unknown task")
                return 1
            st.update(args.task_id, "released", reason=args.reason)
            _set_aside(cfg, args.task_id)
    except state.BusyError as exc:
        print(f"{args.task_id}: not released: {exc}")
        return 1
    print(f"{args.task_id}: released")
    return 0


def parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    """Return the argument parser."""
    ap = argparse.ArgumentParser(
        prog="python -m tools.orchestrator",
        description="Deterministic atomic-slice orchestrator: one task, one-shot QA tests, "
        "one-shot coder, deterministic gates, one-shot audit, one squash commit on main.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    common = argparse.ArgumentParser(add_help=False)
    for target, default in ((ap, None), (common, argparse.SUPPRESS)):
        target.add_argument(
            "--config", type=Path, default=default, help="TOML file merged over the defaults"
        )
        target.add_argument(
            "--config-json", default=default, help="JSON overrides merged last (scripts, tests)"
        )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=text, parents=[common])

    run = add("run", "run one task end to end")
    run.add_argument("task_id")
    run.add_argument("--spec", required=True, help="the slice spec file")
    run.add_argument("--engine", help="the coder's engine (QA and audit: see config)")
    run.add_argument("--paths", nargs="*", help="narrow the spec's paths")
    run.add_argument("--worker")
    rq = add("run-queue", "run a stage's queue (default: code, the slice pipeline)")
    rq.add_argument("--worker")
    rq.add_argument("--stage", default=config.CODE_STAGE, help="the stage to run (default code)")
    rq.add_argument("--all", action="store_true", help="every stage and code, side by side")
    rq.add_argument("--queue", help="a work unit's coding queue (code_queues; default: queue)")
    rq.add_argument("--role", help="a role's queue and worker count (roles in config.toml)")
    rq.add_argument(
        "--workers",
        type=int,
        default=1,
        help="slices at once (docker executor; capped by max_parallel)",
    )
    rq.add_argument(
        "--wait-upstream",
        action="store_true",
        help="code only: wait for stage tasks that have not run yet (set by --all)",
    )
    cs = add("check-spec", "the $0 pre-dispatch check of spec files")
    cs.add_argument("specs", nargs="+")
    sp = add("sandbox-pytest", "run pytest in a sandbox container (the coder's test wrapper)")
    sp.add_argument("pytest_args", nargs="*")
    add("status", "digest of the queue, task states and inbox")
    rel = add("release", "mark a task released (frees its paths and worker)")
    rel.add_argument("task_id")
    rel.add_argument("--reason", default="released by the owner")
    rel.add_argument("--stage", default=config.CODE_STAGE, help="the task's stage (default code)")
    eq = add("enqueue", "add a ticket from cards; the ready gate decides (the only way to queue)")
    eq.add_argument("task_id")
    eq.add_argument("--file", required=True, help="the module the ticket writes")
    eq.add_argument("--cards", default="", help="card IDs, comma separated")
    eq.add_argument("--change", help="a structured fix (D3): what changes in existing code")
    eq.add_argument("--remove-names", nargs="*", default=[], help="names the fix removes")
    eq.add_argument("--rename", nargs="*", default=[], help="old=new renames")
    eq.add_argument("--owner-words", help="the owner's words, quoted, with the date")
    eq.add_argument("--deps", default="")
    eq.add_argument("--sources", default="")
    eq.add_argument("--needs", default="", help="graph names the ticket needs (must have roles)")
    mt = add("metrics", "write the batch metrics into STATUS.md (default: the code queue's tasks)")
    mt.add_argument("--ids", default="", help="task ids, comma separated")
    pl = add("plan", "the Planner: a batch brief file in, ready-gated tickets out (D2)")
    pl.add_argument("--brief", required=True, help="the batch brief file vp-rnd saved")
    rd = add("ready", "run the ready gate over a stage queue's tickets")
    rd.add_argument("ids", nargs="*")
    rd.add_argument("--queue")
    ld = add("land", "record a merge made outside the stage: state, queues and main agree")
    ld.add_argument("task_id")
    ld.add_argument("commit", help="the commit on main that merged the task")
    mv = add("move", "send a task to another role's queue (a ledger row is written)")
    mv.add_argument("task_id")
    mv.add_argument("--to", required=True, help="the receiving role")
    mv.add_argument("--kind", required=True, help="dispute, test_review or tool")
    mv.add_argument("--text", required=True, help="why it moves")
    mv.add_argument("--from", dest="sender", default="devops")
    mv.add_argument("--owner-words", help="the owner's words, dated (a dispute needs them)")
    sm = add("serve-moves", "serve the open moves sent to a role once (the runner calls this)")
    sm.add_argument("role")
    md = add("move-done", "close a tool move: the tool is fixed, the waiting task is released")
    md.add_argument("move_id")
    md.add_argument("--result", required=True, help="what was fixed")
    sr = add("stage-run", "run one stage task (started by run-queue --stage)")
    sr.add_argument("stage")
    sr.add_argument("task_id")
    return ap


_COMMANDS = {
    "run": _cmd_run,
    "run-queue": _cmd_run_queue,
    "check-spec": _cmd_check_spec,
    "sandbox-pytest": _cmd_sandbox_pytest,
    "status": _cmd_status,
    "release": _cmd_release,
    "stage-run": _cmd_stage_run,
    "enqueue": _cmd_enqueue,
    "metrics": _cmd_metrics,
    "plan": _cmd_plan,
    "ready": _cmd_ready,
    "land": _cmd_land,
    "move": _cmd_move,
    "move-done": _cmd_move_done,
    "serve-moves": _cmd_serve_moves,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; return the exit code."""
    args = parser().parse_args(argv)
    try:
        cfg = _config(args)
    except (config.ConfigError, ValueError, OSError) as exc:
        print(f"orchestrator: configuration error: {exc}")
        return 2
    try:
        return _COMMANDS[args.cmd](cfg, args)
    except (ValueError, OSError) as exc:
        print(f"orchestrator: {exc}")
        return 2
