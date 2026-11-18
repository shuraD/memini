"""A role runs autonomously or supervised; any Claude session answers a role's open moves."""

# ruff: noqa: F811

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.gates.test_orchestrator import (  # noqa: F401  (env is a fixture)
    TASK,
    _cfg,
    _plan,
    _task,
    env,
)
from tools.orchestrator import cli, config, moves, pipeline, state

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg(tmp_path: Path) -> config.Config:
    return config.load_config(ROOT, overrides={"state_dir": str(tmp_path / "state")})


def test_a_role_is_autonomous_until_set(cfg: config.Config) -> None:
    assert moves.mode(cfg, "qa") == "autonomous"
    moves.set_mode(cfg, "qa", "supervised")
    assert moves.mode(cfg, "qa") == "supervised"
    assert moves.mode(cfg, "dev1") == "autonomous"
    assert json.loads((cfg.state_dir / "modes.json").read_text()) == {"qa": "supervised"}
    moves.set_mode(cfg, "qa", "autonomous")
    assert moves.mode(cfg, "qa") == "autonomous"


def test_every_mode_change_writes_a_ledger_row(cfg: config.Config) -> None:
    moves.set_mode(cfg, "qa", "supervised")
    rows = [json.loads(x) for x in (cfg.state_dir / "ops-ledger.jsonl").read_text().splitlines()]
    row = next(r for r in rows if r["action"] == "mode")
    assert row["details"]["mode"] == "supervised"
    with pytest.raises(ValueError, match="mode must be"):
        moves.set_mode(cfg, "qa", "sleepy")


def test_serve_skips_a_supervised_role(cfg: config.Config) -> None:
    move_id = moves.post(cfg, "T-1", sender="dev4", to="qa", kind="test_review", text="x")
    moves.set_mode(cfg, "qa", "supervised")
    assert moves.serve(cfg, "qa", {"test_review": lambda _c, _m: "approve"}, once=True) == 0
    assert moves.get(cfg, move_id)["status"] == "open"
    moves.set_mode(cfg, "qa", "autonomous")
    assert moves.serve(cfg, "qa", {"test_review": lambda _c, _m: "approve"}, once=True) == 1


def test_the_mode_command_shows_and_sets(
    cfg: config.Config, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli._cmd_mode(cfg, cli.parser().parse_args(["mode", "qa", "supervised"])) == 0
    assert capsys.readouterr().out.strip() == "qa: supervised"
    cli._cmd_mode(cfg, cli.parser().parse_args(["mode", "qa"]))
    assert capsys.readouterr().out.strip() == "qa: supervised"


def test_move_answer_resolves_an_open_move_and_the_list_shows_it(
    cfg: config.Config, capsys: pytest.CaptureFixture[str]
) -> None:
    move_id = moves.post(
        cfg, "T-1", sender="planner", to="vp-rnd", kind="ask", text="why?", owner_words="said so"
    )
    parser = cli.parser()
    cli._cmd_moves(cfg, parser.parse_args(["moves", "--role", "vp-rnd"]))
    out = capsys.readouterr().out
    assert move_id in out
    assert "T-1" in out
    assert "owner words: said so" in out
    args = parser.parse_args(["move-answer", move_id, "--result", "go", "--by", "vp-rnd"])
    assert cli._cmd_move_answer(cfg, args) == 0
    assert moves.get(cfg, move_id)["result"] == "go"
    assert moves.open_for(cfg, "vp-rnd") == []


def test_move_answer_on_a_tool_move_releases_the_task(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(env, monkeypatch)

    def boom(self: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline._Run, "_go", boom)
    cfg = _cfg(env)
    pipeline.run_task(cfg, _task(env))
    (move,) = moves.open_for(cfg, "devops")
    args = cli.parser().parse_args(
        ["move-answer", move["id"], "--result", "fixed", "--by", "devops"]
    )
    assert cli._cmd_move_answer(cfg, args) == 0
    assert state.State(cfg).task(TASK)["status"] == "released"
