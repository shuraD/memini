"""The Planner (FLOW-HARD.md D2): a brief in, ready-gated tickets out, nothing outside the brief."""

# ruff: noqa: F811
from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.gates.test_orchestrator_enqueue import MODULE, cfg  # noqa: F401  (cfg is a fixture)
from tools.orchestrator import config, enqueue, planner

BRIEF = """version: v1
squad: alpha
prefix: PL
cards: A1 to A2
out: A9
decisions: none
"""


def _brief(tmp_path: Path, text: str = BRIEF) -> Path:
    path = tmp_path / "brief-1.md"
    path.write_text(text)
    return path


def _answer(*slices: dict[str, object]) -> str:
    return "here is the plan\n" + json.dumps({"slices": list(slices)})


def test_the_brief_needs_every_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lacks: squad"):
        planner.read_brief(_brief(tmp_path, BRIEF.replace("squad: alpha\n", "")))
    assert planner.read_brief(_brief(tmp_path)).prefix == "PL"


def test_slices_become_ready_gated_tickets_and_a_summary(
    cfg: config.Config,
    tmp_path: Path,
) -> None:
    seen: list[str] = []

    def ask(prompt: str) -> str:
        seen.append(prompt)
        return _answer({"module": MODULE, "cards": ["A1", "A2"], "deps": []})

    lines = planner.plan(cfg, _brief(tmp_path), ask)
    assert "A1: The store writes a node." in seen[0] and "A9" not in seen[0].split("-----")[5]
    [ticket] = enqueue.read_tickets(cfg.stages[0].queue)
    assert ticket.id == "PL-01" and ticket.module == MODULE
    assert "- PL-01 " in lines[2] and "REFUSED" not in lines[2]
    assert (cfg.state_dir / "plans" / "brief-1.md").exists()
    assert "batch plan for approval" in (cfg.state_dir / "agent-inbox" / "vp-rnd.md").read_text()


def test_a_card_outside_the_brief_refuses_the_whole_plan(
    cfg: config.Config,
    tmp_path: Path,
) -> None:
    ask = lambda _p: _answer({"module": MODULE, "cards": ["A1", "A3"], "deps": []})  # noqa: E731
    with pytest.raises(ValueError, match="not in the brief: A3"):
        planner.plan(cfg, _brief(tmp_path), ask)
    assert not cfg.stages[0].queue.exists()


@pytest.mark.parametrize(
    ("slices", "expect"),
    [
        ([], "list of slices"),
        ([{"module": "m", "cards": ["A1"]}, {"module": "m", "cards": ["A1"]}], "repeats"),
        ([{"module": "m", "cards": ["A1"], "deps": [0]}], "does not come before"),
        ([{"module": "m", "cards": []}], "needs a module and cards"),
    ],
)
def test_a_bad_answer_is_refused(slices: list[dict[str, object]], expect: str) -> None:
    with pytest.raises(ValueError, match=expect):
        planner.parse_slices(_answer(*slices), {"A1", "A2"})


def test_ids_continue_after_the_ones_already_queued(
    cfg: config.Config,
    tmp_path: Path,
) -> None:
    ask = lambda _p: _answer({"module": MODULE, "cards": ["A1"], "deps": []})  # noqa: E731
    planner.plan(cfg, _brief(tmp_path), ask)
    ask2 = lambda _p: _answer({"module": MODULE, "cards": ["A2"], "deps": []})  # noqa: E731
    planner.plan(cfg, _brief(tmp_path), ask2)
    assert [t.id for t in enqueue.read_tickets(cfg.stages[0].queue)] == ["PL-01", "PL-02"]
