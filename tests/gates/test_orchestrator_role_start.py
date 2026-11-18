"""A role session starts its own runner once, from MEMINI_ROLE (FLOW-HARD.md D-10)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from tools.orchestrator import config, moves, role_start

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg(tmp_path: Path) -> config.Config:
    return config.load_config(ROOT, overrides={"state_dir": str(tmp_path / "state")})


class _Proc:
    pid = os.getpid()


def test_a_role_session_starts_its_supervisor_once(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def popen(argv: list[str], **_kw: Any) -> _Proc:
        calls.append(argv)
        return _Proc()

    monkeypatch.setattr(subprocess, "Popen", popen)
    told = role_start.start(cfg, "dev4")
    assert "started automatically by a hook" in told
    assert "you did not start it" in told
    assert calls[0][-3:] == ["role_start", "--supervise", "dev4"] or calls[0][-2:] == [
        "--supervise",
        "dev4",
    ]
    assert role_start.start(cfg, "dev4") == ""  # silent on every later prompt
    assert len(calls) == 1
    ledger = (cfg.state_dir / "ops-ledger.jsonl").read_text().splitlines()
    row = json.loads(ledger[-1])
    assert (row["action"], row["details"]["trigger"]) == ("runner_started", "hook")


def test_a_role_outside_the_table_starts_nothing(cfg: config.Config) -> None:
    assert role_start.start(cfg, "nobody") == "nobody: not a runner role"


def test_the_supervisor_serves_moves_and_runs_the_queue_in_fresh_processes(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[list[str]] = []

    def run(argv: list[str], **_kw: Any) -> None:
        runs.append(argv)
        if len(runs) == 4:
            raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(KeyboardInterrupt):
        role_start.supervise(cfg, "dev4", poll_s=0)
    assert [a[-3:] for a in runs] == [
        ["tools.orchestrator", "serve-moves", "dev4"],
        ["run-queue", "--role", "dev4"],
        ["tools.orchestrator", "serve-moves", "dev4"],
        ["run-queue", "--role", "dev4"],
    ]


def test_the_supervisor_serves_the_moves_sent_to_its_role(cfg: config.Config) -> None:
    move_id = moves.post(cfg, "T-1", sender="dev4", to="qa", kind="test_review", text="x")
    moves.serve(cfg, "qa", {"test_review": lambda _c, _m: "approve"}, once=True)
    assert moves.get(cfg, move_id)["status"] == "done"


def test_a_squad_session_starts_its_runner_from_memini_squad(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MEMINI_ROLE", raising=False)
    monkeypatch.setenv("MEMINI_SQUAD", "dev4")
    seen: list[str] = []
    monkeypatch.setattr(role_start, "start", lambda _cfg, role: seen.append(role) or role)
    monkeypatch.setattr(role_start, "load_config", lambda _root: cfg)
    assert role_start.main([]) == 0
    assert seen == ["dev4"]


def test_autonomous_mode_starts_every_role_but_devops_once(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[str] = []

    def popen(argv: list[str], **_kw: Any) -> _Proc:
        started.append(argv[-1])
        proc = _Proc()
        proc.pid = os.getpid() + len(started)  # a distinct pid file per role is enough
        return proc

    monkeypatch.setattr(subprocess, "Popen", popen)
    first = role_start.start_all(cfg)
    assert sorted(started) == ["dev1", "dev2", "dev3", "dev4", "planner", "qa", "vp-rnd"]
    assert all("autonomous command" in m for m in first)
    ledger = [json.loads(x) for x in (cfg.state_dir / "ops-ledger.jsonl").read_text().splitlines()]
    assert {r["details"]["trigger"] for r in ledger if r["action"] == "runner_started"} == {
        "command"
    }
