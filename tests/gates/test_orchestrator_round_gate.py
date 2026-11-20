"""The round gate (owner 2026-11-20): a slice may go through a second round by itself.

Rules: ``config.toml`` ships ``standing_reissue_words`` and ``max_rounds = 2``; ``max_rounds`` is an
integer of at least 1; every dead-end ``dispute`` carries the standing words; the Planner reissues
automatically while the task's earlier ``reissued`` rows number fewer than ``max_rounds - 1``, at
the limit it writes no ticket and asks vp-rnd once with the history; a dispute with the owner's
own different words is not limited; the counter is per task.
"""

# ruff: noqa: F811
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from tests.gates.test_orchestrator import (  # noqa: F401  (env is a fixture)
    IMPL,
    SNEAKY,
    _cfg,
    _coder,
    _plan,
    _task,
    env,
)
from tests.gates.test_orchestrator_enqueue import MODULE, cfg  # noqa: F401  (cfg is a fixture)
from tests.gates.test_orchestrator_reissue import FIX, PARTS
from tests.gates.test_orchestrator_stages import (
    ID,
    PASSING,
    _enqueue,
    _line,
    _spec_md,
)
from tests.gates.test_orchestrator_stages import _cfg as _stage_cfg
from tests.gates.test_orchestrator_stages import _plan as _stage_plan
from tests.gates.test_orchestrator_stages import env as stage_env  # noqa: F401  (a fixture)
from tools.orchestrator import config, enqueue, moves, pipeline, planner, specstage, state

ROOT = Path(__file__).resolve().parents[2]
OWN = "owner 2026-11-20 (own words): fix this one thing only"


def _plain(tmp_path: Path, **over: Any) -> config.Config:
    return config.load_config(ROOT, overrides={"state_dir": str(tmp_path / "state"), **over})


def _dispute(
    cfg: config.Config, task: str = "T-9", text: str = "a failure", words: str | None = None
) -> moves.Move:
    """Post a dispute carrying ``words`` (the standing words by default) for ``task``."""
    st = state.State(cfg)
    if st.task(task) is None:
        st.claim(task, "w", [MODULE])
        st.update(task, "rejected", reason="audit")
    if not any(t.id == task for t in enqueue.read_tickets(cfg.stages[0].queue)):
        enqueue.enqueue(cfg, task_id=task, module=MODULE, text="A slice ticket." + PARTS)
    w = cfg.standing_reissue_words if words is None else words
    move_id = moves.post(
        cfg, task, sender="dev4", to="planner", kind="dispute", text=text, owner_words=w
    )
    return moves.get(cfg, move_id)


def _reissue(cfg: config.Config, move: moves.Move) -> str:
    return planner.reissue(cfg, move, lambda _p: FIX)


def _asks(cfg: config.Config, task: str = "T-9") -> list[moves.Move]:
    return [m for m in moves.open_for(cfg, "vp-rnd") if m["task"] == task and m["kind"] == "ask"]


def _tickets(cfg: config.Config, task: str = "T-9") -> list[enqueue.Ticket]:
    return [t for t in enqueue.read_tickets(cfg.stages[0].queue) if t.id == task]


# ------------------------------------------------------------------------------ configuration
def test_the_shipped_config_has_two_rounds_and_the_standing_words(tmp_path: Path) -> None:
    """Rule: shipped max_rounds is 2 and the owner's standing words are present and non-empty."""
    cfg = _plain(tmp_path)
    assert cfg.max_rounds == 2
    assert cfg.standing_reissue_words.strip()


@pytest.mark.parametrize("bad", [0, -1, "x", "2", 2.5, 2.0, True, None, [2]])
def test_a_bad_max_rounds_is_refused_with_a_clear_error(tmp_path: Path, bad: Any) -> None:
    """Rule: max_rounds must be an integer >= 1; anything else is a ConfigError naming the key."""
    with pytest.raises(config.ConfigError, match="max_rounds"):
        _plain(tmp_path, max_rounds=bad)


@pytest.mark.parametrize("good", [1, 3, 10])
def test_a_valid_max_rounds_loads(tmp_path: Path, good: int) -> None:
    """Rule: the counter is configuration; any integer >= 1 loads as given."""
    assert _plain(tmp_path, max_rounds=good).max_rounds == good


@pytest.mark.parametrize("bad", [0, -3, "x", 1.5])
def test_a_bad_max_parallel_is_still_refused(tmp_path: Path, bad: Any) -> None:
    """Rule: the shared validator keeps max_parallel an integer >= 1 (no silent int() coercion)."""
    with pytest.raises(config.ConfigError, match="max_parallel"):
        _plain(tmp_path, max_parallel=bad)


def test_a_valid_max_parallel_loads(tmp_path: Path) -> None:
    """Rule: the shared validator accepts a valid max_parallel."""
    assert _plain(tmp_path, max_parallel=3).max_parallel == 3


# ------------------------------------------------------------------------------ dead ends
def _standing(cfg: config.Config) -> str:
    (move,) = moves.open_for(cfg, "planner")
    assert move["kind"] == "dispute"
    return str(move["payload"].get("owner_words"))


def test_a_rejected_task_dispute_carries_the_standing_words(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: a task that ends rejected sends its dispute with the standing words."""
    _plan(env, monkeypatch, coder=_coder({IMPL: SNEAKY}))
    cfg = _cfg(env)
    assert pipeline.run_task(cfg, _task(env)).status == "rejected"
    assert _standing(cfg) == cfg.standing_reissue_words


@pytest.mark.parametrize("status", [pipeline.FAILED, pipeline.DISPUTED])
def test_a_failed_or_disputed_stop_carries_the_standing_words(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """Rule: a failed or disputed stop sends its dispute with the standing words."""

    def stop(self: Any) -> Any:
        raise pipeline._Stop(status, "some reason")

    _plan(env, monkeypatch)
    monkeypatch.setattr(pipeline._Run, "_go", stop)
    cfg = _cfg(env)
    assert pipeline.run_task(cfg, _task(env)).status == status
    assert _standing(cfg) == cfg.standing_reissue_words


def test_the_coders_own_dispute_carries_the_standing_words(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: the coder's TEST-DISPUTE move carries the standing words too."""
    _plan(env, monkeypatch, coder=[{"stdout": "TEST-DISPUTE: add must be commutative only\n"}])
    cfg = _cfg(env)
    assert pipeline.run_task(cfg, _task(env)).status == "disputed"
    assert _standing(cfg) == cfg.standing_reissue_words


def test_a_spec_stage_stop_carries_the_standing_words(
    stage_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: a spec stage that stops sends its dispute with the standing words."""
    cfg = _stage_cfg(stage_env)
    _enqueue(cfg, _line(stage_env))
    _stage_plan(stage_env, monkeypatch, {"SPEC.md": _spec_md(), "tests/unit/test_c.py": PASSING})
    assert specstage.run(cfg, cfg.stages[0], ID) == 1
    assert _standing(cfg) == cfg.standing_reissue_words


# ------------------------------------------------------------------------------ the gate
def test_the_first_dispute_is_reissued_by_itself(cfg: config.Config) -> None:
    """Rule: round 2 is automatic: a ticket, a `reissued` ledger row, no vp-rnd ask."""
    result = _reissue(cfg, _dispute(cfg))
    assert result.startswith("reissued")
    assert len(_tickets(cfg)) == 1 and "SUPERSEDED" in cfg.stages[0].queue.read_text()
    assert state.ops_count(cfg, "T-9", "reissued") == 1
    assert _asks(cfg) == []


def test_the_second_dispute_writes_no_ticket_and_asks_vp_rnd_once(cfg: config.Config) -> None:
    """Rule: at max_rounds 2 the second dispute stops: no ticket, one ask with round text."""
    _reissue(cfg, _dispute(cfg, text="first failure A"))
    before = cfg.stages[0].queue.read_text()
    result = _reissue(cfg, _dispute(cfg, text="second failure B"))
    assert result.startswith("round limit reached")
    assert cfg.stages[0].queue.read_text() == before  # nothing written
    assert state.ops_count(cfg, "T-9", "reissued") == 1
    (ask,) = _asks(cfg)
    text = ask["text"]
    assert "round 3 of max_rounds 2" in text
    assert "waits for the owner" in text
    assert "round 1: first failure A" in text and "round 2: second failure B" in text
    assert "waits for the owner" in (cfg.state_dir / "agent-inbox" / "vp-rnd.md").read_text()


def test_a_third_round_needs_one_more_counter_value(cfg: config.Config) -> None:
    """Rule: with max_rounds 3 the second dispute is reissued and the third escalates."""
    cfg = dataclasses.replace(cfg, max_rounds=3)
    assert _reissue(cfg, _dispute(cfg, text="f1")).startswith("reissued")
    assert _reissue(cfg, _dispute(cfg, text="f2")).startswith("reissued")
    assert _asks(cfg) == []
    assert _reissue(cfg, _dispute(cfg, text="f3")).startswith("round limit reached")
    (ask,) = _asks(cfg)
    assert "round 4 of max_rounds 3" in ask["text"]
    assert all(f"round {i}: f{i}" in ask["text"] for i in (1, 2, 3))
    assert state.ops_count(cfg, "T-9", "reissued") == 2


def test_max_rounds_one_never_reissues_by_itself(cfg: config.Config) -> None:
    """Rule: max_rounds 1 allows no automatic reissue: the first dispute already escalates."""
    cfg = dataclasses.replace(cfg, max_rounds=1)
    move = _dispute(cfg, text="f1")
    before = cfg.stages[0].queue.read_text()
    assert _reissue(cfg, move).startswith("round limit reached")
    assert cfg.stages[0].queue.read_text() == before
    assert state.ops_count(cfg, "T-9", "reissued") == 0
    assert "round 2 of max_rounds 1" in _asks(cfg)[0]["text"]


def test_a_person_with_different_words_is_not_limited_by_the_gate(cfg: config.Config) -> None:
    """Rule: the gate runs only for the standing words; the owner's words reissue."""
    _reissue(cfg, _dispute(cfg, text="f1"))
    assert _reissue(cfg, _dispute(cfg, text="f2")).startswith("round limit reached")
    for i in range(3):  # far past the limit
        assert _reissue(cfg, _dispute(cfg, text=f"p{i}", words=OWN)).startswith("reissued")
    assert len(_asks(cfg)) == 1  # only the gate's own ask


def test_each_task_counts_its_own_rounds(cfg: config.Config) -> None:
    """Rule: a second task's first dispute is reissued while the first task is at its limit."""
    _reissue(cfg, _dispute(cfg, "T-9", "a1"))
    assert _reissue(cfg, _dispute(cfg, "T-9", "a2")).startswith("round limit reached")
    assert _reissue(cfg, _dispute(cfg, "T-8", "b1")).startswith("reissued")
    assert _asks(cfg, "T-8") == []
    assert state.ops_count(cfg, "T-8", "reissued") == 1
    assert "b1" not in _asks(cfg, "T-9")[0]["text"]  # the other task's failure is not history


def test_the_history_trims_each_failure_to_200_characters(cfg: config.Config) -> None:
    """Rule: a long failure appears as `round i: ` plus its first 200 characters only."""
    _reissue(cfg, _dispute(cfg, text="x" * 300))
    _reissue(cfg, _dispute(cfg, text="y" * 300))
    text = _asks(cfg)[0]["text"]
    assert "round 1: " + "x" * 200 in text and "x" * 201 not in text
    assert "round 2: " + "y" * 200 in text and "y" * 201 not in text


def test_the_escalation_posts_exactly_one_ask_per_dispute(cfg: config.Config) -> None:
    """Rule: one stopped dispute, one vp-rnd ask (no duplicate move)."""
    _reissue(cfg, _dispute(cfg, text="f1"))
    _reissue(cfg, _dispute(cfg, text="f2"))
    assert len(moves.open_for(cfg, "vp-rnd")) == 1


def test_a_dispute_without_words_still_asks_and_does_not_count(cfg: config.Config) -> None:
    """Rule: empty words keep the old path (ask, no ticket) and never use up a round."""
    assert _reissue(cfg, _dispute(cfg, words="")) == "no owner words: asked vp-rnd"
    assert state.ops_count(cfg, "T-9", "reissued") == 0
    assert _reissue(cfg, _dispute(cfg, text="f")).startswith("reissued")


# ------------------------------------------------------------------------------ counters
def test_for_task_returns_only_that_task_and_kind(cfg: config.Config) -> None:
    """Rule: moves.for_task filters by task and by kind, oldest first, open or done alike."""
    a = moves.post(cfg, "T-1", sender="s", to="planner", kind="dispute", text="a")
    moves.post(cfg, "T-2", sender="s", to="planner", kind="dispute", text="other task")
    moves.post(cfg, "T-1", sender="s", to="planner", kind="tool", text="other kind")
    b = moves.post(cfg, "T-1", sender="s", to="planner", kind="dispute", text="b")
    assert [m["id"] for m in moves.for_task(cfg, "T-1", "dispute")] == [a, b]
    assert moves.for_task(cfg, "T-3", "dispute") == []
    assert moves.for_task(cfg, "T-1", "nothing") == []
    assert moves.for_task(cfg, "T-", "dispute") == []  # no prefix matching


def test_ops_count_counts_only_that_task_and_action(cfg: config.Config) -> None:
    """Rule: state.ops_count counts ledger rows of one action for one task; 0 without a ledger."""
    assert state.ops_count(cfg, "T-1", "reissued") == 0
    state.ops_log(cfg, "T-1", "reissued", "r")
    state.ops_log(cfg, "T-1", "reissued", "r")
    state.ops_log(cfg, "T-1", "released", "r")
    state.ops_log(cfg, "T-2", "reissued", "r")
    assert state.ops_count(cfg, "T-1", "reissued") == 2
    assert state.ops_count(cfg, "T-1", "released") == 1
    assert state.ops_count(cfg, "T-2", "reissued") == 1
    assert state.ops_count(cfg, "T-3", "reissued") == 0
    assert state.ops_count(cfg, "T-1", "nothing") == 0


def test_ops_count_survives_a_blank_line_in_the_ledger(cfg: config.Config) -> None:
    """Rule: a blank line in the ledger is skipped, not a crash."""
    state.ops_log(cfg, "T-1", "reissued", "r")
    path = cfg.state_dir / state.OPS_LEDGER
    path.write_text(path.read_text() + "\n" + json.dumps({"action": "x", "target": "T-1"}) + "\n")
    assert state.ops_count(cfg, "T-1", "reissued") == 1


# ------------------------------------------------------------------------------ prompt
def test_the_planner_prompt_asks_for_the_failed_check_to_pass() -> None:
    """Rule: the acceptance wording names the check that failed now passing."""
    text = (ROOT / "tools/orchestrator/prompts/planner_fix.txt").read_text()
    assert "the check that failed now passes" in text
