"""Moves between role queues (FLOW-HARD.md D-10): posted, served, resolved, all in the ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.orchestrator import config, moves

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg(tmp_path: Path) -> config.Config:
    return config.load_config(ROOT, overrides={"state_dir": str(tmp_path / "state")})


def _ledger(cfg: config.Config) -> list[dict]:
    text = (cfg.state_dir / "ops-ledger.jsonl").read_text()
    rows = [json.loads(line) for line in text.splitlines()]
    return [r for r in rows if r["action"] != "inbox_post"]  # the roles' notes are tested apart


def test_a_posted_move_is_open_for_its_receiver_and_in_the_ledger(cfg: config.Config) -> None:
    move_id = moves.post(
        cfg, "GR-06b", sender="dev4", to="planner", kind="dispute", text="cannot be built", n=1
    )
    assert [m["id"] for m in moves.open_for(cfg, "planner")] == [move_id]
    assert moves.open_for(cfg, "qa") == []
    row = _ledger(cfg)[-1]
    assert (row["action"], row["target"], row["details"]["move"]) == ("move", "GR-06b", move_id)


def test_serve_resolves_each_open_move_and_the_waiter_reads_the_result(cfg: config.Config) -> None:
    move_id = moves.post(cfg, "GR-10", sender="dev4", to="qa", kind="test_review", text="diff")
    done = moves.serve(cfg, "qa", {"test_review": lambda _c, m: "approve"}, once=True)
    assert done == 1
    got = moves.wait(cfg, move_id, timeout_s=0)
    assert got is not None
    assert (got["result"], got["by"]) == ("approve", "qa")
    assert _ledger(cfg)[-1]["action"] == "move_done"


def test_wait_gives_up_when_nobody_resolves(cfg: config.Config) -> None:
    move_id = moves.post(cfg, "GR-10", sender="dev4", to="qa", kind="test_review", text="diff")
    assert moves.wait(cfg, move_id, timeout_s=0, poll_s=0) is None


def test_serve_leaves_kinds_it_has_no_handler_for(cfg: config.Config) -> None:
    moves.post(cfg, "GR-10", sender="dev4", to="devops", kind="tool", text="gate crashed")
    assert moves.serve(cfg, "devops", {}, once=True) == 0
    assert len(moves.open_for(cfg, "devops")) == 1


def test_run_queue_role_names_come_from_config(cfg: config.Config) -> None:
    assert cfg.roles["dev4"].queue == "graph-schema"
    assert cfg.roles["dev4"].queue in cfg.code_queues
    assert cfg.roles["qa"].queue == "spec"
    assert cfg.roles["planner"].queue == ""


def test_both_roles_are_told_when_a_task_moves_and_when_it_comes_back(cfg: config.Config) -> None:
    move_id = moves.post(cfg, "GR-10", sender="dev4", to="qa", kind="test_review", text="diff")
    assert f"GR-10 moved to you by dev4 (test_review, {move_id})" in _inbox(cfg, "qa")
    moves.serve(cfg, "qa", {"test_review": lambda _c, _m: "approve"}, once=True)
    assert f"GR-10 came back from qa (test_review, {move_id}): approve" in _inbox(cfg, "dev4")


def _inbox(cfg: config.Config, role: str) -> str:
    return (cfg.state_dir / "agent-inbox" / f"{role}.md").read_text()


def test_a_module_belongs_to_the_role_whose_queue_takes_it(cfg: config.Config) -> None:
    assert config.role_for_module(cfg, "src/memini/infra/graph/schema.py") == "dev4"
    assert config.role_for_module(cfg, "src/memini/core/query/engine.py") == "dev2"
    assert config.role_for_module(cfg, "docs/x.md") == ""
