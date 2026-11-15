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
from pathlib import Path
from typing import Any

from tools.orchestrator import (
    config,
    dockerexec,
    gates,
    pipeline,
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


def _cmd_run_queue(cfg: config.Config, args: argparse.Namespace) -> int:
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
        return run_queue_parallel(cfg, workers=workers, launch=_launcher(args, cfg))
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
) -> int:
    """Run the queue with up to ``workers`` slices at once; return the exit code.

    A slice starts when its dependencies are merged and its paths overlap no running slice. A
    failed or blocked slice holds back only the slices that depend on it (owner 2026-11-15: "it
    should be able to work alone"); the run reports it at the end and exits 1.
    """
    st = state.State(cfg)
    entries = state.read_queue(cfg.queue)
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
        if not running and not stages.upstream_active(cfg):
            break
        time.sleep(pause)
    return _queue_exit(st, entries, failed)


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
        if args.worker:
            argv += ["--worker", args.worker]
        procs.append(subprocess.Popen(argv, cwd=_repo()))  # noqa: S603 — our own CLI
        if name != config.CODE_STAGE:
            time.sleep(1)  # the stage loop takes its active lock before code looks for it
    codes = [p.wait() for p in procs]
    return 1 if 1 in codes else max(codes)


def _cmd_stage_run(cfg: config.Config, args: argparse.Namespace) -> int:
    return specstage.run(cfg, stages.stage(cfg, args.stage), args.task_id)


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
    entries = state.read_queue(cfg.queue)
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
            break
    if waiting:
        print("waiting on depends_on: " + "; ".join(waiting))
        return _EXIT_WAITING
    print("queue done")
    return 0


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
    queues = [("queue", cfg.queue), *((n, q.queue) for n, q in cfg.code_queues.items())]
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


def _cmd_release(cfg: config.Config, args: argparse.Namespace) -> int:
    if args.stage != config.CODE_STAGE:
        rec = stages.StageState(cfg, stages.stage(cfg, args.stage).name)
        try:
            with state.task_lock(cfg, f"{args.stage}-{args.task_id}"):
                if rec.status(args.task_id) is None:
                    print(f"{args.stage}/{args.task_id}: unknown task")
                    return 1
                rec.set(args.task_id, stages.RELEASED, args.reason)
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
    except state.BusyError as exc:
        print(f"{args.task_id}: not released: {exc}")
        return 1
    print(f"{args.task_id}: released")
    return 0


def parser() -> argparse.ArgumentParser:
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
    rq.add_argument(
        "--workers",
        type=int,
        default=1,
        help="slices at once (docker executor; capped by max_parallel)",
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
