"""A broken tool or gate moves the task to DevOps' queue; fixing it releases the task (D-10)."""

# ruff: noqa: F811
from __future__ import annotations

from typing import Any

import pytest
from tests.gates.test_orchestrator import (  # noqa: F401  (env is a fixture)
    TASK,
    _cfg,
    _git,
    _plan,
    _task,
    env,
)
from tools.orchestrator import gates, moves, pipeline, stages, state


def _crash(exc: Exception) -> Any:
    def go(self: Any) -> Any:
        raise exc

    return go


@pytest.mark.parametrize(
    "exc", [gates.GateConfigError("flags file missing"), RuntimeError("the gate crashed")]
)
def test_a_crashed_tool_or_gate_moves_the_task_to_devops(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    _plan(env, monkeypatch)
    monkeypatch.setattr(pipeline._Run, "_go", _crash(exc))
    cfg = _cfg(env)
    assert pipeline.run_task(cfg, _task(env)).status == "failed"
    (move,) = moves.open_for(cfg, "devops")
    assert (move["task"], move["kind"]) == (TASK, "tool")
    assert str(exc) in move["text"]


def test_a_gate_failing_on_the_slices_code_stays_with_the_coder(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(env, monkeypatch, coder=[{"write": {"src/memini/core/calc.py": "x = (\n"}}])
    cfg = _cfg(env)
    pipeline.run_task(cfg, _task(env))
    assert moves.open_for(cfg, "devops") == []


def test_fixing_the_tool_closes_the_move_and_releases_the_task(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(env, monkeypatch)
    monkeypatch.setattr(pipeline._Run, "_go", _crash(RuntimeError("boom")))
    cfg = _cfg(env)
    pipeline.run_task(cfg, _task(env))
    (move,) = moves.open_for(cfg, "devops")
    assert moves.finish_tool(cfg, move["id"], "gate fixed", "devops") == TASK
    assert state.State(cfg).task(TASK)["status"] == "released"
    assert moves.open_for(cfg, "devops") == []


def test_fixing_the_tool_sets_the_stopped_runs_leftovers_aside(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(env, monkeypatch)
    monkeypatch.setattr(pipeline._Run, "_go", _crash(RuntimeError("boom")))
    cfg = _cfg(env)
    wt = cfg.worktrees_dir / TASK
    _git(env["repo"], "worktree", "add", "-q", "-b", cfg.branch_prefix + TASK, str(wt), "HEAD")
    pipeline.run_task(cfg, _task(env))
    (move,) = moves.open_for(cfg, "devops")
    moves.finish_tool(cfg, move["id"], "fixed", "devops")
    assert not wt.exists()
    assert (cfg.worktrees_dir / f"{TASK}-{move['id']}").exists()
    assert _git(env["repo"], "branch", "--list", cfg.branch_prefix + TASK) == ""


def test_closing_a_tool_move_releases_a_stage_that_stopped_on_it(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(env, monkeypatch)
    cfg = _cfg(env)
    stage = stages.StageState(cfg, cfg.stages[0].name)
    stage.set("GR-X", "failed", "source - is not a readable file")
    move = moves.post(cfg, "GR-X", sender="qa", to="devops", kind="tool", text="parser")
    moves.finish_tool(cfg, move, "fixed", "devops")
    assert stage.status("GR-X") == stages.RELEASED


def test_release_sets_the_leftover_spec_worktree_aside(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.orchestrator import cli  # noqa: PLC0415

    _plan(env, monkeypatch)
    cfg = _cfg(env)
    left = cfg.worktrees_dir / f"{TASK}.spec"
    _git(env["repo"], "worktree", "add", "-q", "--detach", str(left), "HEAD")
    stages.StageState(cfg, cfg.stages[0].name).set(TASK, "failed", "x")
    args = cli.parser().parse_args(
        ["release", TASK, "--stage", cfg.stages[0].name, "--reason", "r"]
    )
    assert cli._cmd_release(cfg, args) == 0
    assert not left.exists()
    assert any(p.name.startswith(f"{TASK}.spec-rel") for p in cfg.worktrees_dir.iterdir())
