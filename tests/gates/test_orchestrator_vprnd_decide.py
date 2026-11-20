"""vp-rnd decides Tier 1 questions with one no-tools model call (owner order 2026-11-20).

Rules: the DECISIONS.md lookup stays first; an ask starting ``scope: `` and a task at
``reissued >= max_rounds - 1`` go straight to the owner with no model call; otherwise one model
answer ``approve`` / tier 1 / a reason returns the owner words and (for a failure text) posts one
``dispute`` to the planner; every other answer or error escalates to the owner and tells vp-rnd.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from tests.gates.test_orchestrator_vprnd import DECISIONS, ROOT, _ask
from tools.orchestrator import cli, config, engine, moves, planner, state, vprnd


@pytest.fixture
def cfg(tmp_path: Path) -> config.Config:
    """A config whose owner inbox is a temp file (the default one is the real owner inbox)."""
    squads = tmp_path / "mvp" / "squads"
    squads.mkdir(parents=True)
    return config.load_config(
        ROOT,
        overrides={
            "state_dir": str(tmp_path / "state"),
            "cards_dir": str(squads),
            "inbox": str(tmp_path / "inbox" / "OWNER_INBOX.md"),
        },
    )


PREFIX = "owner words: vp-rnd decision (tier 1) "
DATED = re.compile(r"^owner words: vp-rnd decision \(tier 1\) \d{4}-\d{2}-\d{2}: (.+)$")


def _is_decision(result: str) -> bool:
    """True when ``result`` is the prefix, a YYYY-MM-DD date, then ': <reason>'."""
    return DATED.match(result) is not None


FAIL = "cannot be built as written (the cut is wrong); a ticket needs the owner's words"
OK = json.dumps({"decision": "approve", "tier": 1, "reason": "re-cut inside the same cards"})


def _model(monkeypatch: pytest.MonkeyPatch, answer: str | Exception) -> list[str]:
    """Replace the one model call; return the prompts it received."""
    seen: list[str] = []

    def fake(_cfg: config.Config, prompt: str) -> str:
        seen.append(prompt)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(planner, "_ask", fake)
    return seen


def _vp_inbox(cfg: config.Config) -> str:
    path = cfg.state_dir / "agent-inbox" / "vp-rnd.md"
    return path.read_text() if path.exists() else ""


def _owner_inbox(cfg: config.Config) -> str:
    return cfg.inbox.read_text() if cfg.inbox.exists() else ""


def _escalated(cfg: config.Config, text: str) -> None:
    assert text.strip() in _owner_inbox(cfg)
    assert "escalated to the owner" in _vp_inbox(cfg)


# ------------------------------------------------------------------------------ order of checks
def test_the_decisions_lookup_runs_before_any_model_call(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: an owner decision naming the task answers it; the model is never asked."""
    (cfg.cards_dir.parent / "DECISIONS.md").write_text(DECISIONS)
    seen = _model(monkeypatch, OK)
    result = vprnd.answer(cfg, _ask(cfg, "LP-02", FAIL))
    assert result == "owner words: Owner: 2026-11-16: keep the old provider order."
    assert seen == [] and moves.open_for(cfg, "planner") == []


@pytest.mark.parametrize("text", ["scope: a card line must change", "scope: "])
def test_a_scope_ask_goes_straight_to_the_owner_without_a_model_call(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    """Rule: an ask whose text starts 'scope: ' is Tier 2: inbox, vp-rnd notified, no model."""
    seen = _model(monkeypatch, OK)
    assert vprnd.answer(cfg, _ask(cfg, "T-1", text)) == "escalated to the owner"
    assert seen == []
    _escalated(cfg, text)
    assert "needs the owner (tier 2)" in _owner_inbox(cfg)
    assert moves.open_for(cfg, "planner") == []


@pytest.mark.parametrize("text", ["Scope: x", " scope: x", "slicing: scope: x", "scope:x"])
def test_only_a_text_starting_exactly_scope_colon_space_skips_the_model(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    """Rule: the shortcut is the exact prefix 'scope: '; look-alikes go to the model."""
    seen = _model(monkeypatch, OK)
    assert _is_decision(vprnd.answer(cfg, _ask(cfg, "T-1", text)))
    assert len(seen) == 1


def test_a_task_at_the_round_limit_escalates_with_history_and_no_model_call(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: reissued >= max_rounds - 1 -> owner inbox with the dispute history, no model."""
    seen = _model(monkeypatch, OK)
    for text in ("first failure A", "second failure B"):
        moves.post(cfg, "T-1", sender="dev4", to="planner", kind="dispute", text=text)
    state.ops_log(cfg, "T-1", "reissued", "one")
    result = vprnd.answer(cfg, _ask(cfg, "T-1", "round 3: " + FAIL))
    assert result == "escalated to the owner" and seen == []
    inbox = _owner_inbox(cfg)
    assert "round 1: first failure A" in inbox and "round 2: second failure B" in inbox
    assert "Failures so far" in _vp_inbox(cfg)
    assert moves.open_for(cfg, "planner")[0]["text"] == "first failure A"  # nothing new posted


def test_the_round_limit_binds_per_task_and_follows_max_rounds(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: another task's reissue does not escalate; a larger max_rounds lets one more through."""
    seen = _model(monkeypatch, OK)
    state.ops_log(cfg, "T-2", "reissued", "x")
    assert _is_decision(vprnd.answer(cfg, _ask(cfg, "T-1", FAIL)))
    assert len(seen) == 1
    bigger = dataclasses.replace(cfg, max_rounds=3)
    assert _is_decision(vprnd.answer(bigger, _ask(bigger, "T-2", "x")))
    state.ops_log(cfg, "T-2", "reissued", "y")
    assert vprnd.answer(bigger, _ask(bigger, "T-2", "x")) == "escalated to the owner"


# ------------------------------------------------------------------------------ the decision
def test_an_approval_returns_the_owner_words_and_logs_the_decision(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: approve + tier 1 + a reason -> the 'vp-rnd decision (tier 1)' line and an ops row."""
    _model(monkeypatch, "Here you go: " + OK + " done")
    result = vprnd.answer(cfg, _ask(cfg, "T-1", FAIL))
    found = DATED.match(result)
    assert found and found[1] == "re-cut inside the same cards"
    assert result.startswith(PREFIX)
    assert state.ops_count(cfg, "T-1", "vprnd_decided") == 1
    assert _owner_inbox(cfg) == "" and "escalated" not in _vp_inbox(cfg)


def test_an_approved_failure_posts_one_dispute_with_the_decision_as_words(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: the dispute goes to the planner, same task, failure text cut from the ask."""
    _model(monkeypatch, OK)
    result = vprnd.answer(cfg, _ask(cfg, "T-1", FAIL))
    (dispute,) = moves.open_for(cfg, "planner")
    assert (dispute["task"], dispute["kind"], dispute["from"]) == ("T-1", "dispute", "vp-rnd")
    assert dispute["text"] == "the cut is wrong"
    assert dispute["payload"]["owner_words"] == result.removeprefix("owner words: ")
    assert dispute["payload"]["validated"] is False


def test_the_failure_text_after_a_pipe_failure_marker_is_the_dispute_text(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: '| failure: ...' gives the failure; a 'slicing: ' ask marks the dispute validated."""
    _model(monkeypatch, OK)
    vprnd.answer(cfg, _ask(cfg, "T-1", "slicing: cards split | failure: the real failure\nline 2"))
    (dispute,) = moves.open_for(cfg, "planner")
    assert dispute["text"] == "the real failure\nline 2"
    assert dispute["payload"]["validated"] is True


@pytest.mark.parametrize(
    "text", ["scope-less: x | failure: f", "Slicing: x | failure: f", "x slicing: y | failure: f"]
)
def test_only_a_text_starting_with_slicing_marks_the_dispute_validated(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    """Rule: validated is True only for an ask that starts with 'slicing: '."""
    _model(monkeypatch, OK)
    vprnd.answer(cfg, _ask(cfg, "T-1", text))
    assert moves.open_for(cfg, "planner")[0]["payload"]["validated"] is False


@pytest.mark.parametrize(
    "text",
    [
        "the unit suite failed: tests/unit/test_x.py::test_y",  # a batch failure
        "needs a decision",
        "round 3 of max_rounds 2 waits for the owner",
    ],
)
def test_an_ask_without_a_failure_text_gets_the_words_but_no_dispute(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    """Rule: only a 'cannot be built as written' / '| failure:' ask yields a dispute move."""
    _model(monkeypatch, OK)
    assert _is_decision(vprnd.answer(cfg, _ask(cfg, "T-1", text)))
    assert moves.open_for(cfg, "planner") == []


def test_an_open_dispute_is_never_doubled(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: post_once: two approvals while one dispute is open leave exactly one."""
    _model(monkeypatch, OK)
    vprnd.answer(cfg, _ask(cfg, "T-1", FAIL))
    vprnd.answer(cfg, _ask(cfg, "T-1", FAIL.replace("the cut", "another cut")))
    assert len(moves.open_for(cfg, "planner")) == 1


def test_the_failure_of_the_ask_is_carried_whole_even_with_brackets(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: a failure with parentheses and semicolons is carried unchanged."""
    _model(monkeypatch, OK)
    vprnd.answer(
        cfg, _ask(cfg, "T-1", "cannot be built as written (f (inner); more); a ticket needs words")
    )
    assert moves.open_for(cfg, "planner")[0]["text"] == "f (inner); more"


# ------------------------------------------------------------------------------ escalation
@pytest.mark.parametrize(
    "answer",
    [
        json.dumps({"decision": "escalate", "tier": 2, "reason": "architecture"}),
        json.dumps({"decision": "approve", "tier": 2, "reason": "but tier 2"}),
        json.dumps({"decision": "approve", "tier": "1", "reason": "tier as text"}),
        json.dumps({"decision": "escalate", "tier": 1, "reason": "unsure"}),
        json.dumps({"decision": "approve", "tier": 1, "reason": ""}),
        json.dumps({"decision": "approve", "tier": 1, "reason": "   "}),
        json.dumps({"decision": "approve", "tier": 1}),
        json.dumps({"tier": 1, "reason": "no decision"}),
        json.dumps({"decision": "Approve", "tier": 1, "reason": "case"}),
        json.dumps({"decision": "approve", "tier": 1.5, "reason": "x"}),
        json.dumps({"decision": "approve", "tier": None, "reason": "x"}),
        json.dumps(["approve"]),
        "no json at all",
        "",
        "{not json}",
        "{}",
    ],
)
def test_anything_but_a_clean_tier_1_approval_escalates(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    """Rule: escalate / tier 2 / unusable / empty reason -> owner inbox, vp-rnd told, no words."""
    _model(monkeypatch, answer)
    assert vprnd.answer(cfg, _ask(cfg, "T-1", FAIL)) == "escalated to the owner"
    _escalated(cfg, FAIL)
    assert state.ops_count(cfg, "T-1", "vprnd_decided") == 0
    assert moves.open_for(cfg, "planner") == []


@pytest.mark.parametrize("exc", [OSError("engine missing"), ValueError("bad"), TimeoutError("t")])
def test_a_model_error_escalates(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    """Rule: a model failure must not lose the question: it goes to the owner."""
    _model(monkeypatch, exc)
    assert vprnd.answer(cfg, _ask(cfg, "T-1", FAIL)) == "escalated to the owner"
    _escalated(cfg, FAIL)


def test_a_second_json_object_in_the_answer_does_not_turn_escalate_into_approve(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: an answer that is not exactly one JSON object is unusable, never half-trusted."""
    bad = json.dumps({"decision": "escalate", "tier": 2, "reason": "r"})
    _model(monkeypatch, bad + "\n" + OK)
    assert vprnd.answer(cfg, _ask(cfg, "T-1", FAIL)) == "escalated to the owner"


# ------------------------------------------------------------------------------ the prompt
def test_the_model_is_called_once_with_every_placeholder_filled(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: one call; the prompt carries task, kind, text, cards and decisions."""
    (cfg.cards_dir.parent / "DECISIONS.md").write_text("**D-01 other (Z-9)**\nOwner: later\n")
    seen = _model(monkeypatch, OK)
    vprnd.answer(cfg, _ask(cfg, "T-1", FAIL))
    (prompt,) = seen
    assert "task: T-1" in prompt and "kind: ask" in prompt and FAIL in prompt
    assert "D-01 other" in prompt
    assert "$" not in prompt


def test_the_decisions_in_the_prompt_are_trimmed_to_the_last_6000_characters(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: a long DECISIONS.md is cut to its tail so the prompt stays bounded."""
    (cfg.cards_dir.parent / "DECISIONS.md").write_text("OLDEST" + "x" * 7000 + "NEWEST")
    seen = _model(monkeypatch, OK)
    vprnd.answer(cfg, _ask(cfg, "T-1", FAIL))
    assert "NEWEST" in seen[0] and "OLDEST" not in seen[0]


@pytest.mark.parametrize("name", ["vprnd", "planner_validate"])
def test_the_new_prompt_files_render_with_no_placeholder_left(
    cfg: config.Config, name: str
) -> None:
    """Rule: every $placeholder of the shipped prompt is filled by the code's keywords."""
    raw = (cfg.prompts_dir / f"{name}.txt").read_text()
    names = set(re.findall(r"\$(\w+)", raw))
    values = {n: f"<<{n}>>" for n in names}
    out = engine.render(cfg, name, **values)
    assert all(f"<<{n}>>" in out for n in names) and "$" not in out
    assert names == (
        {"task", "kind", "text", "cards", "decisions"}
        if name == "vprnd"
        else {"ticket", "failure", "cards", "versions"}
    )


def test_the_vprnd_prompt_asks_for_one_json_object_and_names_both_tiers(cfg: config.Config) -> None:
    """Rule: the prompt states Tier 1 / Tier 2 and the exact answer shape the code parses."""
    raw = (cfg.prompts_dir / "vprnd.txt").read_text()
    assert "Tier 1" in raw and "Tier 2" in raw
    assert '"decision"' in raw and '"tier"' in raw and '"reason"' in raw


# ------------------------------------------------------------------------------ serving
def test_serving_the_role_decides_and_posts_the_dispute(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: the registered vp-rnd handler runs the decision when a move is served."""
    _model(monkeypatch, OK)
    move_id = _ask(cfg, "T-1", FAIL)["id"]
    assert moves.serve(cfg, "vp-rnd", cli.HANDLERS["vp-rnd"], once=True) == 1
    assert _is_decision(moves.get(cfg, move_id)["result"])
    assert len(moves.open_for(cfg, "planner")) == 1


def _unused(_: Callable[..., object]) -> None: ...


def test_an_empty_failure_group_gets_the_words_but_posts_no_dispute(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: with no failure text there is nothing to dispute: words, no dispute move."""
    _model(monkeypatch, OK)
    text = "cannot be built as written (); a ticket needs the owner's words"
    assert _is_decision(vprnd.answer(cfg, _ask(cfg, "T-1", text)))
    assert moves.open_for(cfg, "planner") == []
    assert moves.for_task(cfg, "T-1", "dispute") == []
