"""Existing tests a ticket pins: the slice may update them, QA then vp-rnd review (owner D-10)."""

# ruff: noqa: F811
from __future__ import annotations

import dataclasses
import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.gates.test_orchestrator import (  # noqa: F401  (env is a fixture)
    AUDIT_PASS,
    GOOD,
    IMPL,
    TASK,
    _cfg,
    _coder,
    _git,
    _plan,
    _prompts,
    _task,
    env,
)
from tests.gates.test_orchestrator_enqueue import cfg as enq_cfg  # noqa: F401
from tools.orchestrator import config, enqueue, moves, pipeline, vprnd

APPROVE = '{"verdict": "approve", "reason": "the edit follows the ticket"}'
OLD = "tests/unit/test_old.py"
OLD_BEFORE = "def test_names():\n    assert {1, 2} == {1, 2}\n    assert len([1, 2]) == 2\n"
OLD_AFTER = "def test_names():\n    assert {1, 2, 3} == {1, 2, 3}\n    assert len([1, 2, 3]) == 3\n"


# --- the ticket lists them ---------------------------------------------------------------


def test_the_ticket_text_lists_the_existing_tests_that_pin_it(enq_cfg: config.Config) -> None:
    text = enqueue.fix_text(
        enq_cfg,
        "Cover every role",
        [],
        [],
        "owner 2026-11-17: rule 2",
        pinned=["tests/unit/test_b.py", "tests/unit/test_a.py"],
    )
    assert "Existing tests that pin this: tests/unit/test_a.py, tests/unit/test_b.py." in text
    assert enqueue.pinned_tests(text) == ["tests/unit/test_a.py", "tests/unit/test_b.py"]


def test_a_ticket_without_pinned_tests_lists_none() -> None:
    assert enqueue.pinned_tests('Cover every role. Owner words: "x".') == []


# --- the verdict a person or the handler gives ---------------------------------------------


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ('{"approved": true, "reason": ""}', (True, "")),
        ('{"approved": false, "reason": "weaker"}', (False, "weaker")),
        ("approve", (True, "")),
        ("Approved, looks fine", (True, "")),
        ("reject: restore the second assertion", (False, "restore the second assertion")),
        ("no", (False, "no")),
    ],
)
def test_a_review_answer_reads_as_json_or_as_plain_text(
    result: str, expected: tuple[bool, str]
) -> None:
    assert moves.parse_verdict(result) == expected


# --- vp-rnd decides light updates and sends heavy ones up ----------------------------------


def _repo_with_update(
    tmp: Path, after: str | None, files: int = 1
) -> tuple[Path, str, str, list[str]]:
    repo = tmp / "upd"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "x")
    _git(repo, "config", "user.email", "t@example.invalid")
    names = [f"tests/unit/test_{i}.py" for i in range(files)]
    for n in names:
        (repo / n).parent.mkdir(parents=True, exist_ok=True)
        (repo / n).write_text(OLD_BEFORE)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    for n in names:
        if after is None:
            (repo / n).unlink()
        else:
            (repo / n).write_text(after)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "change")
    return repo, base, _git(repo, "rev-parse", "HEAD"), names


def _verdict(
    cfg: config.Config, repo: Path, base: str, snap: str, names: list[str]
) -> dict[str, Any]:
    cfg = dataclasses.replace(cfg, repo=repo)
    move = {
        "task": "T-1",
        "text": "update",
        "payload": {"qa_sha": base, "snap": snap, "changed": names},
    }
    return json.loads(vprnd.review_stale(cfg, move))


def test_a_light_update_is_approved_by_vp_rnd(enq_cfg: config.Config, tmp_path: Path) -> None:
    repo, base, snap, names = _repo_with_update(tmp_path, OLD_AFTER)
    assert _verdict(enq_cfg, repo, base, snap, names) == {"approved": True, "reason": ""}


@pytest.mark.parametrize(
    ("after", "files", "heavy"),
    [
        (None, 1, "deleted"),
        ("def test_names():\n    assert {1, 2} == {1, 2}\n", 1, "loses assertions"),
        (OLD_AFTER, 6, "6 test files"),
    ],
)
def test_a_heavy_update_goes_to_the_owner_inbox(
    enq_cfg: config.Config, tmp_path: Path, after: str | None, files: int, heavy: str
) -> None:
    repo, base, snap, names = _repo_with_update(tmp_path, after, files)
    got = _verdict(enq_cfg, repo, base, snap, names)
    assert got["approved"] is False
    assert got["reason"].startswith("heavy:") and heavy in got["reason"]
    assert "heavy test update" in enq_cfg.inbox.read_text()


# --- the whole slice: update the pinned test, QA approves, vp-rnd approves, merged ---------


@pytest.fixture(autouse=True)
def _runners_answer_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    wait = moves.wait

    def served(cfg: Any, move_id: str, timeout_s: float, poll_s: float = 5.0) -> Any:
        moves.serve(cfg, "qa", {"test_review": pipeline.review_move}, once=True)
        moves.serve(cfg, "vp-rnd", {"stale_test_review": vprnd.review_stale}, once=True)
        return wait(cfg, move_id, timeout_s, poll_s)

    monkeypatch.setattr(moves, "wait", served)


def _pin(env: dict[str, Any], cfg: config.Config) -> None:
    repo = env["repo"]
    (repo / OLD).write_text(OLD_BEFORE)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "an older slice's test")
    line = shlex.join(
        [TASK, "src/memini/core/calc/", "-", "-", f"Change. Existing tests that pin this: {OLD}."]
    )
    cfg.stages[0].queue.parent.mkdir(parents=True, exist_ok=True)
    cfg.stages[0].queue.write_text(line + "\n")


def test_the_slice_updates_a_pinned_test_and_vp_rnd_approves(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(
        env,
        monkeypatch,
        coder=_coder({IMPL: GOOD, OLD: OLD_AFTER}),
        audit=[{"stdout": APPROVE}, {"stdout": AUDIT_PASS}],
    )
    cfg = _cfg(env, test_review=True)
    _pin(env, cfg)
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "merged", out.reason
    kinds = [m["kind"] for m in moves._read(cfg)]
    assert kinds == ["test_review", "stale_test_review"]
    assert all(m["status"] == "done" for m in moves._read(cfg))
    assert _git(env["repo"], "show", f"main:{OLD}") == OLD_AFTER.rstrip("\n")


def test_a_heavy_pinned_update_stops_the_slice_for_the_owner(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    weaker = "def test_names():\n    assert {1, 2} == {1, 2}\n"
    _plan(
        env,
        monkeypatch,
        coder=_coder({IMPL: GOOD, OLD: weaker}),
        audit=[{"stdout": APPROVE}, {"stdout": AUDIT_PASS}],
    )
    cfg = _cfg(env, test_review=True)
    _pin(env, cfg)
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "disputed"
    assert "heavy" in out.reason
    assert "heavy test update" in cfg.inbox.read_text()


# --- the ticket lists nothing: the coder fixes the old test its change broke ---------------


def _unlisted(env: dict[str, Any], cfg: config.Config) -> None:
    repo = env["repo"]
    (repo / OLD).write_text(OLD_BEFORE)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "an older slice's test")
    line = shlex.join([TASK, "src/memini/core/calc/", "-", "-", "Change, no tests listed."])
    cfg.stages[0].queue.parent.mkdir(parents=True, exist_ok=True)
    cfg.stages[0].queue.write_text(line + "\n")


def test_the_coder_fixes_an_old_test_its_change_broke_and_it_is_reviewed(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(
        env,
        monkeypatch,
        coder=_coder({IMPL: GOOD, OLD: OLD_AFTER}),
        audit=[{"stdout": APPROVE}, {"stdout": AUDIT_PASS}],
    )
    cfg = _cfg(env, test_review=True)
    _unlisted(env, cfg)
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "merged", out.reason
    assert [m["kind"] for m in moves._read(cfg)] == ["test_review", "stale_test_review"]
    assert _git(env["repo"], "show", f"main:{OLD}") == OLD_AFTER.rstrip("\n")
    assert (
        "tests/unit/"
        in _prompts(env, "coder")[0].split("You may change files only under:")[1][:200]
    )


def test_an_old_test_is_never_deleted_by_a_slice(enq_cfg: config.Config, tmp_path: Path) -> None:
    repo, base, snap, names = _repo_with_update(tmp_path, None)
    run = SimpleNamespace(repo=repo, cfg=dataclasses.replace(enq_cfg, test_review=True), pinned=[])
    sealed: dict[str, str] = {}
    files: list[str] = []
    with pytest.raises(pipeline._Stop, match="deleted"):
        pipeline._Run._take_old_tests(run, base, snap, sealed, files)
    assert names
    assert not sealed
    assert not files
