"""A task refused only because it must wait posts no planner dispute; it stays queued.

Rules: a refusal whose every problem is a wait (another active task holds its paths, or a
dependency is not merged yet) is no dead end: the outcome is REJECTED but no ``dispute`` reaches the
planner, however many cycles repeat it. Any other problem, alone or beside a wait, still posts
exactly one dispute carrying the real problem.
"""

# ruff: noqa: F811
from __future__ import annotations

from typing import Any

import pytest
from tests.gates.test_orchestrator import (  # noqa: F401  (env is a fixture)
    TASK,
    _cfg,
    _plan,
    _spec,
    env,
)
from tools.orchestrator import config, moves, pipeline, state

HOLDER = "T-holder-001"
DEP = "T-dep-001"
OTHER_PATHS = ["src/memini/other/"]
OWN_PATHS = ["src/memini/core/calc/"]


def _planner_moves(cfg: config.Config) -> list[dict[str, Any]]:
    return moves.open_for(cfg, "planner")


def _hold(cfg: config.Config, task_id: str, paths: list[str]) -> None:
    """Make ``task_id`` an active task of another worker holding ``paths``."""
    state.State(cfg).claim(task_id, f"worker-of-{task_id}", paths)


def _run(env: dict[str, Any], cfg: config.Config, **fields: Any) -> pipeline.Outcome:
    return pipeline.run_task(cfg, pipeline.Task(TASK, _spec(env, **fields), "fake"))


def test_paths_held_by_an_active_task_post_no_dispute(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task refused only because an active task holds its paths is a wait, not a dead end."""
    _plan(env, monkeypatch)
    cfg = _cfg(env)
    _hold(cfg, HOLDER, OWN_PATHS)
    out = _run(env, cfg)
    assert out.status == pipeline.REJECTED
    assert "paths overlap the active task" in out.reason
    assert _planner_moves(cfg) == []


def test_an_unmerged_dependency_posts_no_dispute(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task refused only because its dependency is not merged yet is a wait, not a dead end."""
    _plan(env, monkeypatch)
    cfg = _cfg(env)
    _hold(cfg, DEP, OTHER_PATHS)
    out = _run(env, cfg, depends_on=[DEP])
    assert out.status == pipeline.REJECTED
    assert out.reason.endswith("is not merged yet")
    assert _planner_moves(cfg) == []


def test_both_waits_together_post_no_dispute(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An overlap and an unmerged dependency together are still only waits."""
    _plan(env, monkeypatch)
    cfg = _cfg(env)
    _hold(cfg, DEP, OTHER_PATHS)
    _hold(cfg, HOLDER, OWN_PATHS)
    out = _run(env, cfg, depends_on=[DEP])
    assert out.status == pipeline.REJECTED
    assert "paths overlap the active task" in out.reason
    assert "is not merged yet" in out.reason
    assert _planner_moves(cfg) == []


@pytest.mark.parametrize(
    ("fields", "needle"),
    [
        ({"depends_on": ["T-unknown-001"]}, "T-unknown-001"),
        ({"paths": [*OWN_PATHS, "src/memini/core/zzz/"]}, "inside one module"),
        ({"id": "T-other-002"}, "T-other-002"),
    ],
)
@pytest.mark.parametrize("wait", ["overlap", "dependency", "both"])
def test_a_wait_beside_a_real_problem_posts_one_dispute_with_the_real_problem(
    env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    fields: dict[str, Any],
    needle: str,
    wait: str,
) -> None:
    """One non-wait problem makes the refusal a dead end: exactly one dispute names it."""
    _plan(env, monkeypatch)
    cfg = _cfg(env)
    _hold(cfg, DEP, OTHER_PATHS)
    _hold(cfg, HOLDER, OWN_PATHS)
    extra = dict(fields)
    if wait in {"dependency", "both"}:
        extra["depends_on"] = [*extra.get("depends_on", []), DEP]
    out = _run(env, cfg, **extra)
    assert out.status == pipeline.REJECTED
    (move,) = _planner_moves(cfg)
    assert (move["task"], move["kind"]) == (TASK, "dispute")
    assert needle in move["text"]
    assert needle in out.reason


@pytest.mark.parametrize("cycles", [2, 10, 50])
def test_repeated_wait_refusals_post_nothing(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, cycles: int
) -> None:
    """Every cycle of a waiting task refuses again and none of them posts a dispute."""
    _plan(env, monkeypatch)
    cfg = _cfg(env)
    _hold(cfg, DEP, OTHER_PATHS)
    _hold(cfg, HOLDER, OWN_PATHS)
    for _ in range(cycles):
        assert _run(env, cfg, depends_on=[DEP]).status == pipeline.REJECTED
        assert _planner_moves(cfg) == []
    assert not moves.has_open(cfg, TASK, "dispute")
