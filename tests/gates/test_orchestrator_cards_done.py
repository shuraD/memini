"""F-orchestrator-073: a card line is done only by a merge whose Verifier said every line is met.

The pipeline writes one ops row ``cards_done`` (target = task, reason ``merged as <sha10>``,
details cards and commit) right after the merge. ``docfix.marked_done`` reads the done marks of
``VERSIONS.md``; ``docfix.cards_done`` compares them with the ops rows; the ``cards-done`` command
prints the three lists, exits 0 and writes nothing.
"""

# ruff: noqa: F811
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.gates.test_orchestrator import (  # noqa: F401  (env is a fixture)
    AUDIT_PASS,
    GOOD,
    IMPL,
    TASK,
    _cfg,
    _coder,
    _plan,
    _task,
    env,
)
from tests.gates.test_orchestrator_flow_hard import _answer, _row, _ticket
from tools.orchestrator import cli, config, docfix, pipeline, reconcile, state

ROOT = Path(__file__).resolve().parents[2]
ACTION = "cards_done"
SHA = "0123456789abcdef0123456789abcdef01234567"
VERSIONS_MD = """# Versions

## Counts (lines to build)

| Squad | baseline | lenses | done | sent to an agent |
|---|---|---|---|---|
| dev4 graph | 3 | 0 | 2 | 0 |

## baseline

| Line | Capability | Squad | Status | Task | Sent |
|---|---|---|---|---|---|
| A1 | The store writes a node. | dev4 | done | GR-01 | merged |
| A2 | The store deletes a node. | dev4 | done | GR-02 | merged |
| A3 | The store counts nodes. | dev4 | not done | - | - |
"""


def _rows(cfg: config.Config) -> list[dict[str, Any]]:
    path = cfg.state_dir / state.OPS_LEDGER
    lines = path.read_text().splitlines() if path.exists() else []
    return [r for r in (json.loads(x) for x in lines if x) if r["action"] == ACTION]


def _docs(tmp: Path) -> config.Config:
    """A tmp ``VERSIONS.md`` next to an empty cards dir, and a state dir that does not exist."""
    (tmp / "docs" / "squads").mkdir(parents=True)
    (tmp / "docs" / "capabilities").mkdir()
    (tmp / "docs" / "capabilities" / "VERSIONS.md").write_text(VERSIONS_MD)
    over = {"state_dir": str(tmp / "state"), "cards_dir": str(tmp / "docs" / "squads")}
    return config.load_config(ROOT, overrides=over)


def _ids(lines: list[str]) -> list[str]:
    return [line.split(":")[0].strip() for line in lines]


def _snapshot(*dirs: Path) -> dict[str, bytes]:
    files = (p for d in dirs if d.exists() for p in d.rglob("*") if p.is_file())
    return {str(p): p.read_bytes() for p in files}


# --- the pipeline writes the proof -------------------------------------------------------------


def test_a_verified_merge_writes_one_cards_done_row(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: Verifier ok and merged gives exactly one row: the task, its cards, the main commit."""
    cfg = _ticket(env, env["tmp"])
    _plan(
        env,
        monkeypatch,
        coder=_coder({IMPL: GOOD}),
        audit=[{"stdout": AUDIT_PASS}, {"stdout": _answer(_row())}],
    )
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "merged", out.reason
    (row,) = _rows(cfg)
    commit = reconcile.on_main(cfg)[TASK]
    assert row["target"] == TASK
    assert row["details"]["cards"] == ["A1"]
    assert row["details"]["commit"] == commit
    assert row["reason"] == f"merged as {commit[:10]}"


def test_a_block_on_the_first_attempt_then_ok_writes_one_row(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: blocked on attempt 1, ok on attempt 2, merged: one row, not two and not none."""
    cfg = _ticket(env, env["tmp"])
    _plan(
        env,
        monkeypatch,
        coder=_coder({IMPL: GOOD}, {IMPL: GOOD + "\n# more\n"}),
        audit=[
            {"stdout": AUDIT_PASS},
            {"stdout": _answer(_row(status="partial", gap="no retry"))},
            {"stdout": AUDIT_PASS},
            {"stdout": _answer(_row())},
        ],
    )
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "merged", out.reason
    (row,) = _rows(cfg)
    assert row["details"]["cards"] == ["A1"]


def test_a_block_on_the_last_attempt_writes_no_row(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: a Verifier that still blocks on the last attempt proves nothing."""
    cfg = _ticket(env, env["tmp"])
    blocked = {"stdout": _answer(_row(status="partial", gap="no retry"))}
    _plan(
        env,
        monkeypatch,
        coder=_coder({IMPL: GOOD}, {IMPL: GOOD + "\n# more\n"}),
        audit=[{"stdout": AUDIT_PASS}, blocked, {"stdout": AUDIT_PASS}, blocked],
    )
    out = pipeline.run_task(cfg, _task(env))
    assert out.status != "merged", out.reason
    assert _rows(cfg) == []


def test_a_task_with_no_card_lines_writes_no_row(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: a merge whose Verifier was skipped (no card lines) is not a proof."""
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD}))
    cfg = _cfg(env)
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "merged", out.reason
    assert _rows(cfg) == []


# --- the report --------------------------------------------------------------------------------


def test_marked_done_returns_only_the_done_rows_with_their_task(tmp_path: Path) -> None:
    """Rule: only Status ``done`` rows count; ``not done`` rows and the Counts table do not."""
    marked = docfix.marked_done(_docs(tmp_path))
    assert sorted(marked) == ["A1", "A2"]
    assert "GR-01" in marked["A1"] and "GR-02" in marked["A2"]


def test_cards_done_sorts_the_cards_into_the_three_lists(tmp_path: Path) -> None:
    """Rule: done and proven / marked done with no proof / proven but not marked done."""
    cfg = _docs(tmp_path)
    state.ops_log(cfg, "GR-01", ACTION, f"merged as {SHA[:10]}", cards=["A1", "A3"], commit=SHA)
    proven, no_proof, not_marked = docfix.cards_done(cfg)
    assert _ids(proven) == ["A1"]
    assert _ids(no_proof) == ["A2"]
    assert _ids(not_marked) == ["A3"]


def test_without_an_ops_ledger_every_marked_card_has_no_proof(tmp_path: Path) -> None:
    """Rule: no ops ledger means no proof at all; nothing is assumed done."""
    cfg = _docs(tmp_path)
    assert not (cfg.state_dir / state.OPS_LEDGER).exists()
    proven, no_proof, not_marked = docfix.cards_done(cfg)
    assert (proven, not_marked) == ([], [])
    assert sorted(_ids(no_proof)) == ["A1", "A2"]


def test_the_cards_done_command_prints_three_counted_headings_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: ``cards-done`` is a report: three headings with counts, exit 0, no file touched."""
    cfg = _docs(tmp_path)
    state.ops_log(cfg, "GR-01", ACTION, f"merged as {SHA[:10]}", cards=["A1", "A3"], commit=SHA)
    docs = cfg.cards_dir.parent
    before = _snapshot(cfg.state_dir, docs)
    monkeypatch.chdir(ROOT)
    over = {"state_dir": str(cfg.state_dir), "cards_dir": str(cfg.cards_dir)}
    code = cli.main(["cards-done", "--config-json", json.dumps(over)])
    out = capsys.readouterr().out
    assert code == 0
    heads = [line for line in out.splitlines() if line.rstrip().endswith("(1):")]
    assert len(heads) == 3, out
    assert "proven by a merge" in heads[0]
    assert "no proof" in heads[1]
    assert "not marked done" in heads[2]
    assert _snapshot(cfg.state_dir, docs) == before
