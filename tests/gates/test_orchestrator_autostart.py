"""The runtime starts itself: peer revival, set-aside on rerun, release after a doc review.

Rules guarded (change on branch orch-autostart): a live runner restarts a dead peer and never
starts a live one twice; a leftover worktree or branch of an earlier run is renamed (never
deleted, never refused) and logged; a blocked task whose doc review was answered after the block
runs again by itself.
"""

# ruff: noqa: F811
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from tests.gates import test_orchestrator as code_stage
from tests.gates import test_orchestrator_stages as spec_stage
from tests.gates.test_orchestrator import env as code_env  # noqa: F401  (fixture)
from tests.gates.test_orchestrator_enqueue import cfg  # noqa: F401  (fixture)
from tests.gates.test_orchestrator_stages import env as spec_env  # noqa: F401  (fixture)
from tools.orchestrator import config, moves, pipeline, role_start, specstage, state

ROOT = Path(__file__).resolve().parents[2]


class _Proc:
    pid = os.getpid()


def _actions(cfg: config.Config, action: str) -> list[dict[str, Any]]:
    path = cfg.state_dir / "ops-ledger.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(x) for x in path.read_text().splitlines() if x]
    return [r for r in rows if r["action"] == action]


@pytest.fixture
def rcfg(tmp_path: Path) -> config.Config:
    return config.load_config(ROOT, overrides={"state_dir": str(tmp_path / "state")})


# ------------------------------------------------------------------------------ peer revival
_HELD: list[Any] = []


@pytest.fixture(autouse=True)
def _release_run_locks() -> Any:
    yield
    for handle in _HELD:
        handle.close()
    _HELD.clear()


def _popen_log(
    monkeypatch: pytest.MonkeyPatch, rcfg: config.Config, delay: float = 0.0
) -> list[str]:
    started: list[str] = []
    lock = threading.Lock()

    def popen(argv: list[str], **_kw: Any) -> _Proc:
        time.sleep(delay)
        with lock:
            started.append(argv[-1])
            # like a real supervisor: it holds its run lock for its whole life
            locks = rcfg.state_dir / "locks"
            locks.mkdir(parents=True, exist_ok=True)
            handle = (locks / f"runner-{argv[-1]}.run").open("a+", encoding="utf-8")
            fcntl.flock(handle, fcntl.LOCK_EX)
            _HELD.append(handle)
        return _Proc()

    monkeypatch.setattr(subprocess, "Popen", popen)
    return started


def test_a_live_runner_restarts_a_dead_peer_and_says_who_did_it(
    rcfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead peer (pid file of a dead process) is started again by start_all(trigger=peer)."""
    dead = subprocess.Popen(["true"])  # noqa: S607
    dead.wait()
    started = _popen_log(monkeypatch, rcfg)
    pid_file = rcfg.state_dir / "locks" / "runner-qa.pid"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text(str(dead.pid))
    told = role_start.start_all(rcfg, trigger="peer")
    assert "qa" in started
    assert any("another runner" in m for m in told)
    triggers = {r["details"]["trigger"] for r in _actions(rcfg, "runner_started")}
    assert triggers == {"peer"}


def test_a_live_peer_is_not_started_twice(
    rcfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calling start_all every cycle starts each role once; the second call starts nothing."""
    started = _popen_log(monkeypatch, rcfg)
    role_start.start_all(rcfg, trigger="peer")
    first = list(started)
    assert role_start.start_all(rcfg, trigger="peer") == []
    assert started == first


def test_start_all_never_starts_devops_even_for_a_peer(
    rcfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """devops has no queue and no handlers: a peer cycle must not start it."""
    started = _popen_log(monkeypatch, rcfg)
    role_start.start_all(rcfg, trigger="peer")
    assert "devops" not in started
    assert not (rcfg.state_dir / "locks" / "runner-devops.pid").exists()


def test_start_all_defaults_to_the_command_trigger(
    rcfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The existing autonomous command keeps its trigger when no argument is given."""
    _popen_log(monkeypatch, rcfg)
    role_start.start_all(rcfg)
    assert {r["details"]["trigger"] for r in _actions(rcfg, "runner_started")} == {"command"}


def test_concurrent_callers_start_one_runner(
    rcfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two callers racing for the same dead role start exactly one runner (pid check under lock)."""
    started = _popen_log(monkeypatch, rcfg, delay=0.2)
    barrier = threading.Barrier(6)

    def call() -> None:
        barrier.wait()
        role_start.start(rcfg, "dev1", trigger="peer")

    threads = [threading.Thread(target=call) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert started.count("dev1") == 1
    assert len(_actions(rcfg, "runner_started")) == 1


def test_a_stale_pid_file_with_garbage_is_replaced(
    rcfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pid file that holds no number is not a live runner."""
    started = _popen_log(monkeypatch, rcfg)
    pid_file = rcfg.state_dir / "locks" / "runner-qa.pid"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text("not-a-pid")
    role_start.start(rcfg, "qa", trigger="peer")
    assert started == ["qa"]


def test_every_supervisor_cycle_calls_start_all_with_the_peer_trigger(
    rcfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each supervise cycle revives peers (trigger=peer), before serving moves."""
    seen: list[str] = []

    def fake_all(_c: config.Config, trigger: str = "command") -> list[str]:
        seen.append(trigger)
        return []

    monkeypatch.setattr(role_start, "start_all", fake_all)
    calls: list[list[str]] = []

    def run(argv: list[str], **_kw: Any) -> None:
        calls.append(argv)
        if len(calls) >= 4:
            raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(KeyboardInterrupt):
        role_start.supervise(rcfg, "dev4", poll_s=0)
    assert seen and set(seen) == {"peer"}
    assert len(seen) >= 2


# ------------------------------------------------------------------------------ set-aside
def test_a_leftover_code_worktree_and_branch_are_set_aside_and_the_run_proceeds(
    code_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An earlier run's worktree and branch are renamed (kept), a ledger row exists, run merges."""
    repo, tmp = code_env["repo"], code_env["tmp"]
    cfg = code_stage._cfg(code_env, test_review=False)
    wt = tmp / "wt" / code_stage.TASK
    wt.parent.mkdir(parents=True, exist_ok=True)
    code_stage._git(repo, "worktree", "add", "-q", "-b", f"orch/{code_stage.TASK}", str(wt))
    (wt / "leftover.txt").write_text("keep me")
    code_stage._plan(
        code_env, monkeypatch, coder=code_stage._coder({code_stage.IMPL: code_stage.GOOD})
    )
    out = pipeline.run_task(cfg, code_stage._task(code_env))
    assert out.status == "merged", out.reason
    kept = list((tmp / "wt").glob(f"{code_stage.TASK}-rerun*"))
    assert len(kept) == 1
    assert (kept[0] / "leftover.txt").read_text() == "keep me"
    branches = code_stage._git(repo, "branch", "--list", f"orch/{code_stage.TASK}-rerun*")
    assert branches
    rows = _actions(cfg, "set_aside")
    assert [r["target"] for r in rows] == [code_stage.TASK]
    assert rows[0]["details"]["moved"]


def test_a_clean_code_run_writes_no_set_aside_row(
    code_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is left over, so nothing is renamed and no set_aside row is written."""
    cfg = code_stage._cfg(code_env, test_review=False)
    code_stage._plan(
        code_env, monkeypatch, coder=code_stage._coder({code_stage.IMPL: code_stage.GOOD})
    )
    out = pipeline.run_task(cfg, code_stage._task(code_env))
    assert out.status == "merged", out.reason
    assert _actions(cfg, "set_aside") == []


def test_a_leftover_branch_alone_is_set_aside(
    code_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A branch left without a worktree is renamed too: the run does not fail on it."""
    repo = code_env["repo"]
    cfg = code_stage._cfg(code_env, test_review=False)
    code_stage._git(repo, "branch", f"orch/{code_stage.TASK}")
    code_stage._plan(
        code_env, monkeypatch, coder=code_stage._coder({code_stage.IMPL: code_stage.GOOD})
    )
    out = pipeline.run_task(cfg, code_stage._task(code_env))
    assert out.status == "merged", out.reason
    assert code_stage._git(repo, "branch", "--list", f"orch/{code_stage.TASK}-rerun*")


def test_a_leftover_spec_worktree_is_renamed_and_the_spec_stage_proceeds(
    spec_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An earlier spec run's worktree is renamed -rerun<HHMMSS>, logged, and the stage succeeds."""
    s = spec_stage
    cfg = s._cfg(spec_env)
    s._enqueue(cfg, s._line(spec_env))
    s._plan(
        spec_env,
        monkeypatch,
        {"SPEC.md": s._spec_md(), "tests/unit/test_calc_hidden.py": s.FAILING},
    )
    old = cfg.worktrees_dir / f"{s.ID}.spec"
    old.parent.mkdir(parents=True, exist_ok=True)
    s._git(spec_env["repo"], "worktree", "add", "-q", "--detach", str(old))
    (old / "leftover.txt").write_text("keep me")
    assert specstage.run(cfg, cfg.stages[0], s.ID) == 0
    kept = list(cfg.worktrees_dir.glob(f"{s.ID}.spec-rerun*"))
    assert len(kept) == 1
    assert (kept[0] / "leftover.txt").read_text() == "keep me"
    rows = _actions(cfg, "set_aside")
    assert [r["target"] for r in rows] == [s.ID]
    assert rows[0]["details"]["moved"] == [kept[0].name]


# ------------------------------------------------------------------------------ done_at
def test_resolve_stores_done_at_and_not_before_it_is_done(cfg: config.Config) -> None:
    """An open move has no done_at; a resolved one has an ISO time at or after it was posted."""
    mid = moves.post(cfg, "T-1", sender="a", to="b", kind="doc_review", text="x")
    assert "done_at" not in moves.get(cfg, mid)
    moves.resolve(cfg, mid, "fixed", by="b")
    row = moves.get(cfg, mid)
    assert row["status"] == "done"
    assert row["done_at"] >= row["at"]
    assert row["done_at"].endswith("+00:00")


# ------------------------------------------------------------------------------ release_answered
def _status(cfg: config.Config, task: str = "T-9") -> str:
    rec = state.State(cfg).task(task)
    assert rec is not None
    return str(rec["status"])


def _blocked(cfg: config.Config, task: str = "T-9", status: str = "rejected") -> None:
    st = state.State(cfg)
    st.claim(task, "w", ["src/memini/x.py"])
    st.update(task, status, reason="verifier partial")


def _answer(cfg: config.Config, task: str = "T-9", kind: str = "doc_review") -> str:
    mid = moves.post(cfg, task, sender="planner", to="qa", kind=kind, text="fix the line")
    moves.resolve(cfg, mid, "line fixed", by="qa")
    return mid


def _set_done_at(cfg: config.Config, mid: str, when: str) -> None:
    path = cfg.state_dir / moves.MOVES
    rows = [json.loads(x) for x in path.read_text().splitlines() if x]
    for r in rows:
        if r["id"] == mid:
            r["done_at"] = when
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.parametrize("status", ["rejected", "disputed", "failed"])
def test_a_blocked_task_is_released_after_its_doc_review_is_answered(
    cfg: config.Config, status: str
) -> None:
    """Rule: rejected/disputed/failed + resolved doc_review after the block -> released, logged."""
    _blocked(cfg, status=status)
    mid = _answer(cfg)
    _set_done_at(cfg, mid, "2999-01-01T00:00:00+00:00")
    assert moves.release_answered(cfg) == ["T-9"]
    assert _status(cfg) == "released"
    rows = _actions(cfg, "released")
    assert [r["target"] for r in rows] == ["T-9"]
    assert mid in rows[0]["reason"]


def test_an_answer_given_after_the_block_in_real_time_releases(cfg: config.Config) -> None:
    """No clock tampering: block, wait for the next second, answer -> released."""
    _blocked(cfg)
    time.sleep(1.1)
    _answer(cfg)
    assert moves.release_answered(cfg) == ["T-9"]


def test_a_task_with_an_open_move_is_not_released(cfg: config.Config) -> None:
    """An open move for the task (any kind) means the doc-fix flow is not finished."""
    _blocked(cfg)
    mid = _answer(cfg)
    _set_done_at(cfg, mid, "2999-01-01T00:00:00+00:00")
    moves.post(cfg, "T-9", sender="planner", to="qa", kind="doc_review", text="second line")
    assert moves.release_answered(cfg) == []
    assert _status(cfg) == "rejected"
    assert _actions(cfg, "released") == []


def test_an_answer_older_than_the_block_does_not_release(cfg: config.Config) -> None:
    """A doc review answered before the task was blocked says nothing about this block."""
    mid = _answer(cfg)
    _set_done_at(cfg, mid, "2000-01-01T00:00:00+00:00")
    _blocked(cfg)
    assert moves.release_answered(cfg) == []
    assert _status(cfg) == "rejected"


def test_a_blocked_task_without_any_move_is_not_released(cfg: config.Config) -> None:
    """No doc review at all: the task stays blocked."""
    _blocked(cfg)
    assert moves.release_answered(cfg) == []


@pytest.mark.parametrize("kind", ["test_review", "dispute", "tool"])
def test_only_a_doc_review_answer_releases(cfg: config.Config, kind: str) -> None:
    """Other move kinds answered after the block do not release the task."""
    _blocked(cfg)
    mid = _answer(cfg, kind=kind)
    _set_done_at(cfg, mid, "2999-01-01T00:00:00+00:00")
    assert moves.release_answered(cfg) == []


@pytest.mark.parametrize("status", ["active", "merged", "released", "done"])
def test_a_task_in_another_state_is_not_released(cfg: config.Config, status: str) -> None:
    """Only rejected, disputed or failed tasks are released; a running or finished one is not."""
    st = state.State(cfg)
    st.claim("T-9", "w", ["src/memini/x.py"])
    if status != "active":
        st.update("T-9", status, reason="r")
    mid = _answer(cfg)
    _set_done_at(cfg, mid, "2999-01-01T00:00:00+00:00")
    before = _status(cfg)
    assert moves.release_answered(cfg) == []
    assert _status(cfg) == before


def test_an_answer_for_another_task_does_not_release(cfg: config.Config) -> None:
    """The doc review must be for this task."""
    _blocked(cfg)
    mid = _answer(cfg, task="T-other")
    _set_done_at(cfg, mid, "2999-01-01T00:00:00+00:00")
    assert moves.release_answered(cfg) == []


def test_a_task_whose_run_still_holds_the_lock_waits(cfg: config.Config) -> None:
    """A run in progress keeps its lock: the task is not released and not changed."""
    _blocked(cfg)
    mid = _answer(cfg)
    _set_done_at(cfg, mid, "2999-01-01T00:00:00+00:00")
    with state.run_lock(cfg, "T-9"):
        # same process cannot re-take flock on a new fd? flock is per open file: it is refused
        assert moves.release_answered(cfg) == []
    assert _status(cfg) == "rejected"
    assert moves.release_answered(cfg) == ["T-9"]


def test_release_sets_the_leftovers_aside_and_deletes_nothing(cfg: config.Config) -> None:
    """The blocked run's worktree and branch are renamed, not removed, when the task is released."""
    _blocked(cfg)
    wt = cfg.worktrees_dir / "T-9"
    cfg.worktrees_dir.mkdir(parents=True, exist_ok=True)
    code_stage._git(cfg.repo, "worktree", "add", "-q", "-b", f"{cfg.branch_prefix}T-9", str(wt))
    mid = _answer(cfg)
    _set_done_at(cfg, mid, "2999-01-01T00:00:00+00:00")
    assert moves.release_answered(cfg) == ["T-9"]
    assert not wt.exists()
    assert (cfg.worktrees_dir / f"T-9-{mid}").is_dir()


def test_release_answered_is_idempotent(cfg: config.Config) -> None:
    """A second cycle releases nothing more (the task is no longer blocked)."""
    _blocked(cfg)
    mid = _answer(cfg)
    _set_done_at(cfg, mid, "2999-01-01T00:00:00+00:00")
    assert moves.release_answered(cfg) == ["T-9"]
    assert moves.release_answered(cfg) == []
    assert len(_actions(cfg, "released")) == 1


def test_serve_moves_releases_every_cycle(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The serve-moves command calls release_answered and says which tasks it freed."""
    from tools.orchestrator import cli  # noqa: PLC0415

    _blocked(cfg)
    mid = _answer(cfg)
    _set_done_at(cfg, mid, "2999-01-01T00:00:00+00:00")
    import argparse  # noqa: PLC0415

    assert cli._cmd_serve_moves(cfg, argparse.Namespace(role="qa")) == 0
    assert "released T-9" in capsys.readouterr().out
    assert _status(cfg) == "released"
