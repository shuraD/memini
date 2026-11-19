"""A squad proposes a card fix: nothing is edited; the Planner lists the places; vp-rnd gets it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.orchestrator import config, docfix, moves

ROOT = Path(__file__).resolve().parents[2]
CARD = "| A75 | Graph-health KPIs include drift. | done | GR-04 | merged |\n"
VERS = "| A75 | {text} | dev4 | done | GR-04 | merged |\n"


@pytest.fixture
def cfg(tmp_path: Path) -> config.Config:
    docs = tmp_path / "docs"
    (docs / "squads" / "graph").mkdir(parents=True)
    (docs / "capabilities").mkdir()
    (docs / "squads" / "graph" / "CARD.md").write_text("## IN: baseline\n" + CARD)
    (docs / "squads" / "graph" / "SPEC.md").write_text("A75: stats of one scope\n")
    (docs / "capabilities" / "VERSIONS.md").write_text(
        VERS.format(text="Graph-health KPIs include drift.")
    )
    return config.load_config(
        ROOT,
        overrides={"state_dir": str(tmp_path / "state"), "cards_dir": str(docs / "squads")},
    )


def test_a_proposal_edits_nothing_and_goes_to_the_planner(cfg: config.Config) -> None:
    before = (cfg.cards_dir / "graph" / "CARD.md").read_text()
    move_id = docfix.post(
        cfg, "GR-10", sender="dev4", card="A75", text="Dirty ratio.", why="SPEC L58"
    )
    assert [m["id"] for m in moves.open_for(cfg, "planner")] == [move_id]
    assert (cfg.cards_dir / "graph" / "CARD.md").read_text() == before


def test_the_planner_lists_every_place_and_passes_it_to_vp_rnd(cfg: config.Config) -> None:
    docfix.post(cfg, "GR-10", sender="dev4", card="A75", text="Dirty ratio.", why="SPEC L58")
    move = moves.open_for(cfg, "planner")[0]
    assert json.loads(docfix.review(cfg, move))["approved"] is True
    text = moves.open_for(cfg, "vp-rnd")[0]["text"]
    assert all(name in text for name in ("CARD.md", "SPEC.md", "VERSIONS.md"))


def test_an_unknown_card_is_refused(cfg: config.Config) -> None:
    with pytest.raises(ValueError, match="not a baseline card"):
        docfix.post(cfg, "GR-10", sender="dev4", card="A999", text="x", why="y")


def test_drift_finds_a_card_that_differs_between_card_file_and_versions(cfg: config.Config) -> None:
    assert docfix.drift(cfg) == []
    path = cfg.cards_dir.parent / "capabilities" / "VERSIONS.md"
    path.write_text(VERS.format(text="Dirty ratio."))
    assert docfix.drift(cfg) == ["A75: CARD.md and VERSIONS.md differ"]
