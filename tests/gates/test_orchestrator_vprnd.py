"""vp-rnd in the autonomous flow: an owner decision answers an ask move, else the owner is asked."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.gates.test_orchestrator_enqueue import cfg as enqueue_cfg  # noqa: F401
from tests.gates.test_orchestrator_reissue import _dispute
from tools.orchestrator import cli, config, moves, planner, vprnd

ROOT = Path(__file__).resolve().parents[2]

DECISIONS = """# Decisions

## Open

**D-12 graph schema v2 (GR-06)**
Owner:

## Decided

**D-07 providers (LP-02)**
Owner: 2026-11-16: keep the old provider order.
"""


@pytest.fixture
def cfg(tmp_path: Path) -> config.Config:
    squads = tmp_path / "mvp" / "squads"
    squads.mkdir(parents=True)
    return config.load_config(
        ROOT, overrides={"state_dir": str(tmp_path / "state"), "cards_dir": str(squads)}
    )


def _ask(cfg: config.Config, task: str, text: str) -> moves.Move:
    return moves.get(
        cfg, moves.post(cfg, task, sender="planner", to="vp-rnd", kind="ask", text=text)
    )


def test_a_decision_naming_the_task_gives_the_owners_words(cfg: config.Config) -> None:
    (cfg.cards_dir.parent / "DECISIONS.md").write_text(DECISIONS)
    result = vprnd.answer(cfg, _ask(cfg, "LP-02", "cannot be built as written"))
    assert result == "owner words: Owner: 2026-11-16: keep the old provider order."


def test_a_card_id_in_the_text_finds_the_decision(cfg: config.Config) -> None:
    (cfg.cards_dir.parent / "DECISIONS.md").write_text(DECISIONS)
    assert "keep the old provider order" in vprnd.answer(cfg, _ask(cfg, "T-1", "see LP-02"))


def test_a_passing_mention_in_a_decision_body_is_no_answer(cfg: config.Config) -> None:
    body = "**D-08 where a dispute goes**\nOwner: send it to the planner (as for GR-10).\n"
    (cfg.cards_dir.parent / "DECISIONS.md").write_text(body)
    assert vprnd.answer(cfg, _ask(cfg, "GR-10", "fix the card")) == "escalated to the owner"


def test_without_a_decision_the_owner_inbox_gets_the_question(cfg: config.Config) -> None:
    (cfg.cards_dir.parent / "DECISIONS.md").write_text(DECISIONS)
    assert vprnd.answer(cfg, _ask(cfg, "GR-06", "unclear")) == "escalated to the owner"
    assert "unclear" in cfg.inbox.read_text()


def test_a_missing_decisions_file_escalates(cfg: config.Config) -> None:
    assert vprnd.answer(cfg, _ask(cfg, "X-1", "what now")) == "escalated to the owner"
    assert "what now" in cfg.inbox.read_text()


def test_serving_the_role_answers_the_ask_move(cfg: config.Config) -> None:
    move_id = _ask(cfg, "X-1", "what now")["id"]
    assert moves.serve(cfg, "vp-rnd", cli.HANDLERS["vp-rnd"], once=True) == 1
    assert moves.get(cfg, move_id)["result"] == "escalated to the owner"


def test_a_dispute_without_owner_words_posts_an_ask_move_to_vp_rnd(
    enqueue_cfg: config.Config,  # noqa: F811
) -> None:
    move = _dispute(enqueue_cfg, words="")
    assert planner.reissue(enqueue_cfg, move) == "no owner words: asked vp-rnd"
    (ask,) = moves.open_for(enqueue_cfg, "vp-rnd")
    assert (ask["task"], ask["kind"], ask["from"]) == ("T-9", "ask", "planner")
    assert "cannot be built as written" in ask["text"]
