"""A disputed task moves to the Planner queue; the Planner writes a D3 ticket for QA (D-10)."""

# ruff: noqa: F811
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from tests.gates.test_orchestrator_enqueue import MODULE, cfg  # noqa: F401  (cfg is a fixture)
from tools.orchestrator import config, enqueue, moves, planner, specstage, stages, state

WORDS = "owner 2026-11-17 (D-10): remove only the properties that now have a role"
FIX = json.dumps(
    {
        "change": "Remove only the properties that now have a role.",
        "remove_names": [],
        "renames": [],
    }
)


def _dispute(cfg: config.Config, words: str = WORDS) -> moves.Move:
    st = state.State(cfg)
    st.claim("T-9", "w", [MODULE])
    st.update("T-9", "rejected", reason="audit")
    enqueue.enqueue(cfg, task_id="T-9", module=MODULE, text="Old text the spec line contradicts.")
    move_id = moves.post(
        cfg,
        "T-9",
        sender="devops",
        to="planner",
        kind="dispute",
        text="cannot be built as written",
        owner_words=words,
    )
    return moves.get(cfg, move_id)


def test_the_planner_reissues_the_ticket_under_the_same_id_for_qas_ready_review(
    cfg: config.Config,
) -> None:
    move = _dispute(cfg)
    result = planner.reissue(cfg, move, lambda _prompt: "ok " + FIX)
    assert result.startswith("reissued")
    queue = cfg.stages[0].queue
    assert "SUPERSEDED" in queue.read_text()  # history stays, as a comment
    [ticket] = enqueue.read_tickets(queue)
    assert ticket.id == "T-9" and ticket.module == MODULE
    assert f'Owner words: "{WORDS}"' in ticket.text
    assert state.State(cfg).task("T-9")["status"] == "released"


def test_without_the_owners_words_nothing_is_written_and_vp_rnd_is_asked(
    cfg: config.Config,
) -> None:
    move = _dispute(cfg, words="")
    assert planner.reissue(cfg, move, lambda _prompt: FIX) == "no owner words: asked vp-rnd"
    assert "SUPERSEDED" not in cfg.stages[0].queue.read_text()
    assert "needs the owner's words" in (cfg.state_dir / "agent-inbox" / "vp-rnd.md").read_text()


def test_an_unusable_answer_leaves_the_queue_as_it_was(cfg: config.Config) -> None:
    move = _dispute(cfg)
    result = planner.reissue(cfg, move, lambda _prompt: "no json here")
    assert result.startswith("the Planner answer is not a usable fix")
    assert "SUPERSEDED" not in cfg.stages[0].queue.read_text()


def test_the_planner_role_serves_a_dispute_move_and_resolves_it(cfg: config.Config) -> None:
    move = _dispute(cfg)
    done = moves.serve(
        cfg, "planner", {"dispute": lambda c, m: planner.reissue(c, m, lambda _p: FIX)}, once=True
    )
    assert done == 1
    assert moves.get(cfg, move["id"])["status"] == "done"


def test_a_stage_record_of_the_old_ticket_is_released_for_the_new_one(cfg: config.Config) -> None:
    move = _dispute(cfg)
    stages.StageState(cfg, cfg.stages[0].name).set("T-9", stages.DONE, "passed")
    planner.reissue(cfg, move, lambda _prompt: FIX)
    assert stages.StageState(cfg, cfg.stages[0].name).status("T-9") == stages.RELEASED


def test_the_old_spec_and_hold_out_are_set_aside_not_deleted(
    cfg: config.Config, tmp_path: Path
) -> None:
    cfg = dataclasses.replace(cfg, spec_dir=tmp_path / "specs", holdout_dir=tmp_path / "holdout")
    item = stages.StageItem("T-9", MODULE, (), (), "")
    spec_file = specstage.spec_path(cfg, item)
    spec_file.parent.mkdir(parents=True)
    spec_file.write_text("old spec")
    hold = cfg.holdout_dir / "T-9"
    hold.mkdir(parents=True)
    (hold / "test_old.py").write_text("def test_x(): ...")
    move = _dispute(cfg)
    planner.reissue(cfg, move, lambda _prompt: FIX)
    assert not spec_file.exists() and not hold.exists()
    assert (
        spec_file.with_name(f"{spec_file.name}.superseded-{move['id']}").read_text() == "old spec"
    )
    assert (hold.with_name(f"T-9.superseded-{move['id']}") / "test_old.py").exists()
