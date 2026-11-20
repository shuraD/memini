"""A task that ends in a dead end reaches the planner by itself; nobody relays it.

Rules: ``post_once`` posts nothing while a move of that kind is open for the task; a task that ends
rejected, failed or disputed (also a spec refused before dispatch) posts one ``dispute`` to the
planner with the real reason; an open ``tool`` move or an open coder dispute suppresses it; a stage
that stops or crashes posts one; a crash or gate-config error posts only the ``tool`` move; a busy
task posts nothing; a red CI run posts ``batch_failure`` once per sha.
"""

# ruff: noqa: F811
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.gates.test_orchestrator import (  # noqa: F401  (env is a fixture)
    IMPL,
    QA_TEST,
    SNEAKY,
    TASK,
    _cfg,
    _coder,
    _git,
    _plan,
    _spec,
    _task,
    env,
)
from tests.gates.test_orchestrator_stages import (
    FAILING,
    ID,
    PASSING,
    _enqueue,
    _line,
    _spec_md,
)
from tests.gates.test_orchestrator_stages import _cfg as _stage_cfg
from tests.gates.test_orchestrator_stages import _plan as _stage_plan
from tests.gates.test_orchestrator_stages import env as stage_env  # noqa: F401  (a fixture)
from tools.orchestrator import ci_watch, config, gates, moves, pipeline, specstage, stages, state

ROOT = Path(__file__).resolve().parents[2]


def _disputes(cfg: config.Config) -> list[dict[str, Any]]:
    return [m for m in moves.open_for(cfg, "planner") if m["kind"] == "dispute"]


# ------------------------------------------------------------------------------ post_once
@pytest.fixture
def plain_cfg(tmp_path: Path) -> config.Config:
    return config.load_config(ROOT, overrides={"state_dir": str(tmp_path / "state")})


def test_post_once_posts_when_nothing_is_open(plain_cfg: config.Config) -> None:
    """A first post_once is a real post and has_open sees it."""
    move_id = moves.post_once(plain_cfg, "T-1", sender="a", to="planner", kind="dispute", text="x")
    assert move_id is not None
    assert moves.has_open(plain_cfg, "T-1", "dispute")
    assert [m["id"] for m in moves.open_for(plain_cfg, "planner")] == [move_id]


def test_post_once_does_nothing_while_the_same_kind_is_open(plain_cfg: config.Config) -> None:
    """An open move of the same kind for the task blocks a second one, whoever sends it."""
    moves.post(plain_cfg, "T-1", sender="a", to="planner", kind="dispute", text="first")
    again = moves.post_once(plain_cfg, "T-1", sender="b", to="qa", kind="dispute", text="second")
    assert again is None
    assert [m["text"] for m in moves.open_for(plain_cfg, "planner")] == ["first"]
    assert moves.open_for(plain_cfg, "qa") == []


@pytest.mark.parametrize(
    ("task", "kind"), [("T-2", "dispute"), ("T-1", "tool"), ("T-1", "batch_failure")]
)
def test_post_once_is_per_task_and_per_kind(plain_cfg: config.Config, task: str, kind: str) -> None:
    """A move of another task or another kind never blocks the post."""
    moves.post(plain_cfg, "T-1", sender="a", to="planner", kind="dispute", text="first")
    assert moves.post_once(plain_cfg, task, sender="a", to="planner", kind=kind, text="n")


def test_post_once_posts_again_after_the_move_is_resolved(plain_cfg: config.Config) -> None:
    """Only an OPEN move blocks: a resolved one does not."""
    moves.post(plain_cfg, "T-1", sender="a", to="planner", kind="dispute", text="first")
    moves.serve(plain_cfg, "planner", {"dispute": lambda _c, _m: "done"}, once=True)
    assert not moves.has_open(plain_cfg, "T-1", "dispute")
    assert moves.post_once(plain_cfg, "T-1", sender="a", to="planner", kind="dispute", text="2")


# ------------------------------------------------------------------------------ pipeline
def test_a_rejected_task_posts_one_planner_dispute_with_the_reason(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two failed coder attempts end REJECTED: one dispute to the planner names the task."""
    _plan(env, monkeypatch, coder=_coder({IMPL: SNEAKY}))
    cfg = _cfg(env)
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "rejected"
    (move,) = moves.open_for(cfg, "planner")
    assert (move["task"], move["kind"]) == (TASK, "dispute")
    assert out.reason in move["text"]
    assert "rejected" in move["text"]
    assert moves.open_for(cfg, "devops") == []


def test_a_failed_task_posts_one_planner_dispute_with_the_reason(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task that ends FAILED without a tool crash (stale branch) posts one planner dispute."""
    _plan(env, monkeypatch)
    cfg = _cfg(env)
    _git(env["repo"], "branch", cfg.branch_prefix + TASK)
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "failed"
    (move,) = moves.open_for(cfg, "planner")
    assert (move["task"], move["kind"]) == (TASK, "dispute")
    assert "exists from an earlier run" in move["text"]
    assert moves.open_for(cfg, "devops") == []


def test_a_disputed_task_posts_one_planner_dispute_with_the_reason(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task that ends DISPUTED by a stop other than the coder's posts one planner dispute."""

    def stop(self: Any) -> Any:
        raise pipeline._Stop(pipeline.DISPUTED, "the QA review rejected the test change twice")

    _plan(env, monkeypatch)
    monkeypatch.setattr(pipeline._Run, "_go", stop)
    cfg = _cfg(env)
    assert pipeline.run_task(cfg, _task(env)).status == "disputed"
    (move,) = moves.open_for(cfg, "planner")
    assert (move["task"], move["kind"]) == (TASK, "dispute")
    assert "rejected the test change twice" in move["text"]


def test_the_coders_own_dispute_is_not_duplicated(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coder's TEST-DISPUTE already posted a dispute: the dead end adds no second one."""
    _plan(env, monkeypatch, coder=[{"stdout": "TEST-DISPUTE: add must be commutative only\n"}])
    cfg = _cfg(env)
    assert pipeline.run_task(cfg, _task(env)).status == "disputed"
    (move,) = moves.open_for(cfg, "planner")
    assert "add must be commutative only" in move["text"]


def test_an_open_dispute_blocks_a_second_one_for_a_rejected_task(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dispute already open for the task stays the only one when the task ends rejected."""
    _plan(env, monkeypatch, coder=_coder({IMPL: SNEAKY}))
    cfg = _cfg(env)
    moves.post(cfg, TASK, sender="x", to="planner", kind="dispute", text="earlier")
    assert pipeline.run_task(cfg, _task(env)).status == "rejected"
    assert [m["text"] for m in _disputes(cfg)] == ["earlier"]


def test_an_open_tool_move_suppresses_the_dispute(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """While a tool move for the task is open DevOps owns it: no dispute goes to the planner."""
    _plan(env, monkeypatch, coder=_coder({IMPL: SNEAKY}))
    cfg = _cfg(env)
    moves.post(cfg, TASK, sender="x", to="devops", kind="tool", text="gate broke")
    assert pipeline.run_task(cfg, _task(env)).status == "rejected"
    assert _disputes(cfg) == []


def test_a_long_reason_is_trimmed_to_1500_characters(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The move carries at most 1500 characters of the reason, not the whole text."""

    def stop(self: Any) -> Any:
        raise pipeline._Stop(pipeline.REJECTED, "r" * 5000)

    _plan(env, monkeypatch)
    monkeypatch.setattr(pipeline._Run, "_go", stop)
    cfg = _cfg(env)
    pipeline.run_task(cfg, _task(env))
    (move,) = _disputes(cfg)
    assert "r" * 1500 in move["text"]
    assert "r" * 1501 not in move["text"]


@pytest.mark.parametrize(
    "fields", [{"depends_on": ["T-unknown-001"]}, {"id": "T-other-002"}, {"module": "src/"}]
)
def test_a_spec_refused_before_dispatch_posts_one_dispute_with_the_problem(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, fields: dict[str, Any]
) -> None:
    """A spec refused by the pre-check never runs, yet the planner gets one dispute."""
    _plan(env, monkeypatch)
    cfg = _cfg(env)
    out = pipeline.run_task(cfg, pipeline.Task(TASK, _spec(env, **fields), "fake"))
    assert out.status == "rejected"
    (move,) = _disputes(cfg)
    assert move["task"] == TASK
    assert out.reason[:200] in move["text"]


def test_an_unreadable_spec_posts_one_dispute(env: dict[str, Any]) -> None:
    """A spec file without a header is refused before dispatch and still reaches the planner."""
    cfg = _cfg(env)
    path = env["tmp"] / "bad.md"
    path.write_text("no header here\n")
    out = pipeline.run_task(cfg, pipeline.Task(TASK, path, "fake"))
    assert out.status == "rejected"
    (move,) = _disputes(cfg)
    assert out.reason[:100] in move["text"]


def test_qa_tests_refused_before_sealing_post_one_dispute_with_the_deny_word(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA tests holding a denied word end REJECTED before sealing; the dispute names the word."""
    word = "import " + "importlib"
    assert gates.text_problems(word, _cfg(env).deny_patterns)  # the premise: this word is denied
    text = f"{word}\n" + QA_TEST
    _plan(env, monkeypatch, qa=[{"write": {"tests/unit/test_calc.py": text}}])
    cfg = _cfg(env)
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "rejected"
    assert "QA tests refused before sealing" in out.reason
    (move,) = _disputes(cfg)
    assert "QA tests refused before sealing" in move["text"]
    assert "importlib" in move["text"]


@pytest.mark.parametrize(
    "exc", [gates.GateConfigError("flags file missing"), RuntimeError("the gate crashed")]
)
def test_a_crash_posts_only_the_tool_move(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    """A crash or gate-config error is DevOps' to fix: one tool move, no planner dispute."""

    def crash(self: Any) -> Any:
        raise exc

    _plan(env, monkeypatch)
    monkeypatch.setattr(pipeline._Run, "_go", crash)
    cfg = _cfg(env)
    assert pipeline.run_task(cfg, _task(env)).status == "failed"
    (tool,) = moves.open_for(cfg, "devops")
    assert tool["kind"] == "tool"
    assert moves.open_for(cfg, "planner") == []


def test_a_busy_task_posts_nothing(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """A task already running elsewhere is not a dead end: no move at all."""
    _plan(env, monkeypatch)
    cfg = _cfg(env)
    with state.run_lock(cfg, TASK):
        out = pipeline.run_task(cfg, _task(env))
    assert out.status == "rejected"
    assert not (cfg.state_dir / "moves.jsonl").exists() or moves.open_for(cfg, "planner") == []


def test_a_merged_task_posts_no_dispute(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task that ends merged is not a dead end."""
    from tests.gates.test_orchestrator import GOOD  # noqa: PLC0415

    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD}))
    cfg = _cfg(env)
    assert pipeline.run_task(cfg, _task(env)).status == "merged"
    assert moves.open_for(cfg, "planner") == []


# ------------------------------------------------------------------------------ stages
def test_a_stage_that_stops_posts_one_planner_dispute_with_the_reason(
    stage_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests that pass without code stop the spec stage: one planner dispute names why."""
    cfg = _stage_cfg(stage_env)
    _enqueue(cfg, _line(stage_env))
    _stage_plan(stage_env, monkeypatch, {"SPEC.md": _spec_md(), "tests/unit/test_c.py": PASSING})
    assert specstage.run(cfg, cfg.stages[0], ID) == 1
    reason = stages.StageState(cfg, "spec").records()[ID]["reason"]
    (move,) = moves.open_for(cfg, "planner")
    assert (move["task"], move["kind"]) == (ID, "dispute")
    assert reason[:200] in move["text"]


def test_a_stage_that_crashes_posts_one_planner_dispute(
    stage_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexpected exception in a stage leaves the task failed and tells the planner."""

    class Boom(specstage._Job):
        def go(self) -> None:
            msg = "disk on fire"
            raise OSError(msg)

    cfg = _stage_cfg(stage_env)
    _enqueue(cfg, _line(stage_env))
    assert specstage.run(cfg, cfg.stages[0], ID, Boom) == 1
    assert stages.StageState(cfg, "spec").status(ID) == "failed"
    (move,) = moves.open_for(cfg, "planner")
    assert (move["task"], move["kind"]) == (ID, "dispute")
    assert "disk on fire" in move["text"]


def test_a_stage_stop_with_a_long_reason_is_trimmed(
    stage_env: dict[str, Any],
) -> None:
    """A stage dispute carries at most 1500 characters of the reason."""

    class Stopper(specstage._Job):
        def go(self) -> None:
            raise specstage._Stop(specstage.REJECTED, "s" * 4000)

    cfg = _stage_cfg(stage_env)
    _enqueue(cfg, _line(stage_env))
    assert specstage.run(cfg, cfg.stages[0], ID, Stopper) == 1
    (move,) = moves.open_for(cfg, "planner")
    assert "s" * 1500 in move["text"]
    assert "s" * 1501 not in move["text"]


def test_a_stage_stop_with_a_dispute_already_open_posts_no_second(
    stage_env: dict[str, Any],
) -> None:
    """Re-stopping a stage while the planner still holds the dispute adds nothing."""

    class Stopper(specstage._Job):
        def go(self) -> None:
            raise specstage._Stop(specstage.REJECTED, "again")

    cfg = _stage_cfg(stage_env)
    _enqueue(cfg, _line(stage_env))
    moves.post(cfg, ID, sender="spec", to="planner", kind="dispute", text="earlier")
    specstage.run(cfg, cfg.stages[0], ID, Stopper)
    assert [m["text"] for m in moves.open_for(cfg, "planner")] == ["earlier"]


def test_a_stage_that_succeeds_posts_nothing(
    stage_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage that hands the task on is not a dead end."""
    cfg = _stage_cfg(stage_env)
    _enqueue(cfg, _line(stage_env))
    _stage_plan(stage_env, monkeypatch, {"SPEC.md": _spec_md(), "tests/unit/test_c.py": FAILING})
    assert specstage.run(cfg, cfg.stages[0], ID) == 0
    assert moves.open_for(cfg, "planner") == []


# ------------------------------------------------------------------------------ ci
def _red(sha: str, run_id: int) -> Any:
    def run(argv: list[str]) -> str:
        if argv[2] == "list":
            row = {
                "databaseId": run_id,
                "conclusion": "failure",
                "headSha": sha,
                "displayTitle": "push",
                "status": "completed",
            }
            return json.dumps([row])
        return "job\tstep\tFAILED tests/a.py::test_one - boom"

    return run


def test_a_second_red_run_of_the_same_sha_posts_no_second_batch_failure(
    plain_cfg: config.Config,
) -> None:
    """Two red runs of one commit (new run id) leave one open batch_failure for the planner."""
    ci_watch.check(plain_cfg, _red("abcdef1234", 10))
    ci_watch.check(plain_cfg, _red("abcdef1234", 11))
    moves_open = moves.open_for(plain_cfg, "planner")
    assert [m["kind"] for m in moves_open] == ["batch_failure"]
    assert moves.open_for(plain_cfg, "devops") == []


def test_red_runs_of_different_shas_each_post_one(plain_cfg: config.Config) -> None:
    """The dedupe is per commit: a new red commit posts its own batch_failure."""
    ci_watch.check(plain_cfg, _red("abcdef1234", 10))
    ci_watch.check(plain_cfg, _red("1234567abc", 11))
    assert len(moves.open_for(plain_cfg, "planner")) == 2
