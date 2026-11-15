"""Command line of the orchestrator (see ``--help``)."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tools.orchestrator import config, pipeline, spec, state
from tools.orchestrator.executor import clean_env

EPILOG = f"""\
configuration:
  defaults: {config.DEFAULTS.name} next to this tool (tools/orchestrator/{config.DEFAULTS.name})
  your file: --config <file.toml> or MEMINI_ORCH_CONFIG=<file.toml> (merged over the defaults)
  single paths: MEMINI_ORCH_STATE_DIR, MEMINI_ORCH_HOLDOUT_DIR, MEMINI_ORCH_WORKTREES_DIR,
                MEMINI_ORCH_INBOX, MEMINI_ORCH_QUEUE
  queue line:  <task id> <spec file> [engine|-] [path,path,...]   ('#' starts a comment)

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


def status_lines(cfg: config.Config) -> list[str]:
    """Return the plain-text digest of the queue, the task states and the inbox."""
    st = state.State(cfg)
    tasks = st.tasks()
    lines = []
    entries = state.read_queue(cfg.queue) if cfg.queue.exists() else []
    lines.append(f"queue: {cfg.queue} ({len(entries)} entries)")
    for e in entries:
        rec = tasks.get(e.task_id)
        missing = [d for d in _deps(e) if not _done(st, d)]
        if _done(st, e.task_id):
            label = "done"
        elif rec and rec["status"] not in ("released",):
            label = {"disputed": "disputed", "active": "active"}.get(rec["status"], "blocked")
            label += f" ({rec['status']}: {str(rec.get('reason', ''))[:120]})"
        elif missing:
            label = f"waiting on {', '.join(missing)}"
        else:
            label = "ready"
        lines.append(f"  {e.task_id}: {label}")
    others = sorted(set(tasks) - {e.task_id for e in entries})
    for task_id in others:
        lines.append(f"  {task_id}: {tasks[task_id]['status']} (not in the queue)")
    heads = _INBOX_HEAD.findall(cfg.inbox.read_text(encoding="utf-8")) if cfg.inbox.exists() else []
    lines.append(f"inbox: {cfg.inbox} ({len(heads)} entries)")
    lines += [f"  {h}" for h in heads]
    return lines


def _cmd_status(cfg: config.Config, args: argparse.Namespace) -> int:
    print("\n".join(status_lines(cfg)))
    return 0


def _cmd_release(cfg: config.Config, args: argparse.Namespace) -> int:
    st = state.State(cfg)
    try:
        with state.run_lock(cfg):  # never release a task under a running orchestrator
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
    rq = add("run-queue", "run the queue file in order")
    rq.add_argument("--worker")
    cs = add("check-spec", "the $0 pre-dispatch check of spec files")
    cs.add_argument("specs", nargs="+")
    add("status", "digest of the queue, task states and inbox")
    rel = add("release", "mark a task released (frees its paths and worker)")
    rel.add_argument("task_id")
    rel.add_argument("--reason", default="released by the owner")
    return ap


_COMMANDS = {
    "run": _cmd_run,
    "run-queue": _cmd_run_queue,
    "check-spec": _cmd_check_spec,
    "status": _cmd_status,
    "release": _cmd_release,
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
