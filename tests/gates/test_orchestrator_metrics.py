"""Metrics after every batch (FLOW-HARD.md D10), written into STATUS.md from the ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.orchestrator import config, metrics, state

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg(tmp_path: Path) -> config.Config:
    squads = tmp_path / "mvp" / "squads"
    squads.mkdir(parents=True)
    return config.load_config(
        ROOT, overrides={"state_dir": str(tmp_path / "state"), "cards_dir": str(squads)}
    )


def _task(cfg: config.Config, task_id: str, *history: tuple[str, str]) -> None:
    data = {"queue_done": [], "tasks": {}}
    path = cfg.state_dir / "state.json"
    if path.exists():
        data = json.loads(path.read_text())
    steps = [
        {"at": f"2026-11-17T1{i}:00:00+00:00", "status": s, "reason": r}
        for i, (s, r) in enumerate(history)
    ]
    data["tasks"][task_id] = {"status": history[-1][0], "history": steps}
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _row(rows: list[tuple[str, str, str]], start: str) -> str:
    return next(v for m, v, _ in rows if m.startswith(start))


def test_first_run_rule_and_ticket_metrics(cfg: config.Config) -> None:
    _task(cfg, "A-1", ("active", ""), ("merged", ""))
    _task(
        cfg,
        "A-2",
        ("active", ""),
        ("rejected", "deny scan: atexit"),
        ("active", ""),
        ("merged", ""),
    )
    _task(cfg, "A-3", ("active", ""), ("disputed", "cannot be built"))
    rows = metrics.compute(cfg, ["A-1", "A-2", "A-3"])
    assert _row(rows, "slices merged on the first run") == "1 of 3"
    assert _row(rows, "runs failed by a rule violation") == "1"
    assert _row(rows, "reruns caused by a ticket") == "1"
    assert _row(rows, "flow efficiency").startswith("not measured")


def test_people_in_the_ledger_count_as_relay_hops_and_money_on_failed_runs(
    cfg: config.Config,
) -> None:
    _task(cfg, "A-1", ("active", ""), ("merged", ""))
    _task(cfg, "A-2", ("active", ""), ("rejected", "not green"))
    state.ops_log(cfg, "A-1", "enqueue", "by the tool")  # the orchestrator's own row
    row = {"at": "2026-11-17T12:30:00+00:00", "actor": "vp-rnd", "action": "queue_edit"}
    with (cfg.state_dir / "ops-ledger.jsonl").open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    runs = cfg.state_dir / "runs" / "A-2"
    runs.mkdir(parents=True)
    (runs / "costs.tsv").write_text("01\tqa\tclaude\t0.40\n02\tcoder\tclaude\t0.35\n")
    rows = metrics.compute(cfg, ["A-1", "A-2"])
    assert _row(rows, "money on thrown-away runs") == "0.75"
    assert _row(rows, "relay hops") == "1.0"  # one person row, one merged slice


def test_status_md_gets_one_section_that_is_replaced_not_appended(cfg: config.Config) -> None:
    _task(cfg, "A-1", ("active", ""), ("merged", ""))
    status = cfg.cards_dir.parent / "STATUS.md"
    status.write_text("# Status\n\nkept text\n")
    metrics.write(cfg, ["A-1"], "2026-11-17 10:00")
    metrics.write(cfg, ["A-1"], "2026-11-17 11:00")
    text = status.read_text()
    assert text.startswith("# Status\n\nkept text") and text.count(metrics.BEGIN) == 1
    assert "2026-11-17 11:00" in text and "2026-11-17 10:00" not in text
    assert "| slices merged on the first run | 1 of 1 | 80% or more |" in text
