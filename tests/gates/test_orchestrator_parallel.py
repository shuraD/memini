"""Parallel slice runs (owner 2026-11-15: "all the point was that I will have 100 agents that can
do all the slices").

- Run slots: with the docker executor and ``max_parallel`` > 1, up to that many runs hold a
  slot at once; the same task never runs twice; with the local executor (coder code on the
  host) the single global run lock stays.
- The scheduler starts a queue entry only when its dependencies are merged and its paths do not
  overlap a running slice or another slice it starts in the same round.
- ``run-queue --workers N`` launches up to N runs at once and stops starting new ones after a
  failure.

Tests-Author: Claude/vp-rnd (new tests for its own change).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tools.orchestrator import cli, config, schedule, state

ROOT = Path(__file__).resolve().parents[2]


def _cfg(tmp_path: Path, **over: Any) -> config.Config:
    base: dict[str, Any] = {
        "state_dir": str(tmp_path / "state"),
        "executor": "docker",
        "holdout_dir": str(tmp_path / "holdout"),
        "holdout_required": False,  # hold-out readiness has its own test below
    }
    return config.load_config(ROOT, overrides={**base, **over}, env={})


# ------------------------------------------------------------------------------- run slots
def test_the_local_executor_keeps_one_run_at_a_time(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, executor="local", max_parallel=4)
    assert not cfg.parallel
    with state.run_lock(cfg, "A"), pytest.raises(state.BusyError), state.run_lock(cfg, "B"):
        pass


def test_docker_runs_take_one_of_n_slots(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, max_parallel=2)
    assert cfg.parallel
    with (
        state.run_lock(cfg, "A"),
        state.run_lock(cfg, "B"),
        pytest.raises(state.BusyError, match="2 run slots"),
        state.run_lock(cfg, "C"),
    ):
        pass
    with state.run_lock(cfg, "C"):  # slots are free again
        pass


def test_the_same_task_never_runs_twice(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, max_parallel=3)
    with (
        state.run_lock(cfg, "A"),
        pytest.raises(state.BusyError, match="A is already running"),
        state.run_lock(cfg, "A"),
    ):
        pass


def test_max_parallel_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(config.ConfigError):
        _cfg(tmp_path, max_parallel=0)


# ------------------------------------------------------------------------------- scheduler
def _c(task: str, paths: tuple[str, ...], deps: tuple[str, ...] = ()) -> schedule.Candidate:
    return schedule.Candidate(task, paths, deps)


def test_pick_skips_unmerged_dependencies_and_overlaps() -> None:
    cands = [
        _c("A", ("src/a/",)),
        _c("B", ("src/b/",), ("A",)),  # waits on A
        _c("C", ("src/a/x.py",)),  # inside A's path: not in the same round
        _c("D", ("src/d/",)),
        _c("E", ("src/r/",)),  # overlaps the running R
    ]
    picked = schedule.pick(cands, merged=set(), running={"R": ("src/r/",)}, free=5)
    assert picked == ["A", "D"]


def test_pick_respects_free_slots_and_merged_dependencies() -> None:
    cands = [_c("B", ("src/b/",), ("A",)), _c("C", ("src/c/",)), _c("D", ("src/d/",))]
    assert schedule.pick(cands, merged={"A"}, running={}, free=2) == ["B", "C"]
    assert schedule.pick(cands, merged={"A"}, running={}, free=0) == []


# ------------------------------------------------------------------------------- run-queue
class _Proc:
    def __init__(self, code: int, polls: int) -> None:
        self.code, self.polls = code, polls

    def poll(self) -> int | None:
        self.polls -= 1
        return self.code if self.polls <= 0 else None


def _queue(tmp_path: Path, specs: dict[str, tuple[str, list[str]]]) -> Path:
    lines = []
    for task, (paths, deps) in specs.items():
        spec = tmp_path / "specs" / f"{task}.md"
        spec.parent.mkdir(parents=True, exist_ok=True)
        dep = "[" + ", ".join(deps) + "]"
        spec.write_text(
            f"---\nid: {task}\nmodule: m\npaths: [{paths}]\ndepends_on: {dep}\nflag: ''\n"
            "behavior: b\ninterface: i\nacceptance: a\nout_of_scope: o\nsize: S\n---\n"
        )
        lines.append(f"{task} {spec}")
    queue = tmp_path / "queue.txt"
    queue.write_text("\n".join(lines) + "\n")
    return queue


def test_run_queue_launches_independent_slices_together(tmp_path: Path) -> None:
    queue = _queue(
        tmp_path,
        {"A": ("src/a/", []), "B": ("src/b/", []), "C": ("src/c/", ["A"])},
    )
    cfg = _cfg(tmp_path, max_parallel=4, queue=str(queue))
    st = state.State(cfg)
    started: list[list[str]] = []

    def launch(entry: state.QueueEntry) -> _Proc:
        started.append([entry.task_id, *sorted(r for r, p in running_now().items() if p)])
        st.claim(entry.task_id, entry.task_id, ())  # what the run itself records
        st.update(entry.task_id, "merged", commit="x")  # the run merged its slice
        return _Proc(0, polls=2)

    procs: dict[str, _Proc] = {}

    def running_now() -> dict[str, bool]:
        return {k: v.polls > 0 for k, v in procs.items()}

    def tracked(entry: state.QueueEntry) -> _Proc:
        procs[entry.task_id] = launch(entry)
        return procs[entry.task_id]

    code = cli.run_queue_parallel(cfg, workers=3, launch=tracked, pause=0.0)
    assert code == 0
    order = [s[0] for s in started]
    assert order[:2] == ["A", "B"]  # both independent slices started in the first round
    assert order[2] == "C"  # C only after A merged


def test_a_failure_blocks_only_its_dependents(tmp_path: Path) -> None:
    """Owner 2026-11-15 ("it should be able to work alone"): one failed slice does not stop
    the queue; only the slices that depend on it wait, and the run reports it at the end."""
    queue = _queue(tmp_path, {"A": ("src/a/", []), "B": ("src/b/", ["A"]), "C": ("src/c/", [])})
    cfg = _cfg(tmp_path, max_parallel=2, queue=str(queue))
    st = state.State(cfg)
    started: list[str] = []

    def launch(entry: state.QueueEntry) -> _Proc:
        started.append(entry.task_id)
        if entry.task_id == "A":
            return _Proc(1, polls=1)  # the run failed
        st.claim(entry.task_id, entry.task_id, ())
        st.update(entry.task_id, "merged", commit="x")
        return _Proc(0, polls=1)

    assert cli.run_queue_parallel(cfg, workers=2, launch=launch, pause=0.0) == 1
    assert started == ["A", "C"]  # B depends on the failed A and never starts


def test_an_earlier_failure_does_not_stop_independent_slices(tmp_path: Path) -> None:
    queue = _queue(tmp_path, {"A": ("src/a/", []), "C": ("src/c/", [])})
    cfg = _cfg(tmp_path, max_parallel=2, queue=str(queue))
    st = state.State(cfg)
    st.claim("A", "A", ())
    st.update("A", "rejected", reason="lint")  # failed in an earlier run, not released
    started: list[str] = []

    def launch(entry: state.QueueEntry) -> _Proc:
        started.append(entry.task_id)
        st.claim(entry.task_id, entry.task_id, ())
        st.update(entry.task_id, "merged", commit="x")
        return _Proc(0, polls=1)

    assert cli.run_queue_parallel(cfg, workers=2, launch=launch, pause=0.0) == 1
    assert started == ["C"]


def test_slices_without_hidden_tests_wait_instead_of_failing(tmp_path: Path) -> None:
    """A slice whose hold-out is not written yet is not started (no rejection to release);
    the next run-queue picks it up once its hidden tests exist."""
    queue = _queue(tmp_path, {"A": ("src/a/", []), "B": ("src/b/", [])})
    cfg = _cfg(tmp_path, max_parallel=2, queue=str(queue), holdout_required=True)
    (tmp_path / "holdout" / "A").mkdir(parents=True)
    (tmp_path / "holdout" / "A" / "test_a.py").write_text("def test_a():\n    pass\n")
    st = state.State(cfg)
    started: list[str] = []

    def launch(entry: state.QueueEntry) -> _Proc:
        started.append(entry.task_id)
        st.claim(entry.task_id, entry.task_id, ())
        st.update(entry.task_id, "merged", commit="x")
        return _Proc(0, polls=1)

    assert cli.run_queue_parallel(cfg, workers=2, launch=launch, pause=0.0) == cli._EXIT_WAITING
    assert started == ["A"]
    assert st.task("B") is None  # never recorded as rejected
