"""Slice flow with a QA review of test changes (owner decision 2026-11-17).

The coder sees the hidden tests in full and may edit a test that contradicts the spec or the
gate rules, in its own branch. The orchestrator keeps the original sealed tests, stores the
diff of every test change, and after the gates pass a QA review stage returns approve or
reject with a reason. One reject sends the slice back to the coder once; a second goes to the
owner. Every step is logged in the ops ledger. Uses the fake engines of test_orchestrator.py.
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
    QA_TEST,
    TASK,
    _cfg,
    _coder,
    _git,
    _plan,
    _prompts,
    _task,
    env,
)
from tools.orchestrator import audit, engine, gates, moves, pipeline, state

APPROVE = '{"verdict": "approve", "reason": "the edit matches the spec"}'
REJECT = '{"verdict": "reject", "reason": "the test was changed only to make it pass"}'
EDITED = (
    "from memini.core.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == add(2, 1)\n"
)
TEST = "tests/unit/test_calc.py"


def _ops(cfg: Any) -> list[dict[str, Any]]:
    path = cfg.state_dir / "ops-ledger.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _actions(cfg: Any) -> list[str]:
    return [str(r["action"]) for r in _ops(cfg)]


# --- the verdict ---------------------------------------------------------------------------


def test_review_verdict_approve_and_reject() -> None:
    assert audit.parse_review(APPROVE) == audit.Review(True, "the edit matches the spec")
    rejected = audit.parse_review(REJECT)
    assert not rejected.approved
    assert "only to make it pass" in rejected.reason


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no json here",
        '{"verdict": "pass", "reason": "x"}',
        '{"verdict": "approve"}',
        '{"verdict": "approve", "reason": ""}',
        APPROVE + "\n" + REJECT,  # a quoted approve cannot outvote a reject
        '{"verdict": "reject"}',
    ],
)
def test_review_fails_closed(text: str) -> None:
    assert not audit.parse_review(text).approved


def test_the_review_prompt_names_what_it_must_reject() -> None:
    cfg = _cfg_for_prompts()
    text = engine.render(cfg, "test_review", task_id="T", spec="S", test_diff="D", code_diff="C")
    assert "only to make" in text
    assert "checks less than" in text
    assert "approve" in text
    assert "D" in text


def _cfg_for_prompts() -> Any:
    from tools.orchestrator.config import load_config  # noqa: PLC0415

    return load_config(Path(__file__).resolve().parents[2])


# --- the flow -------------------------------------------------------------------------------


def test_no_test_change_means_no_review(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD}))
    cfg = _cfg(env, test_review=True)
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "merged", out.reason
    assert len(_prompts(env, "audit")) == 1  # the audit only: no review call
    assert "test_review_skipped" in _actions(cfg)


def test_the_coder_sees_the_hidden_tests(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD}))
    cfg = _cfg(env, test_review=True)
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "merged", out.reason
    prompt = _prompts(env, "coder")[0]
    assert "tests/unit/test_holdout.py" in prompt
    assert "may edit" in prompt
    assert "hold-out revealed" in " ".join(_actions(cfg)) or "holdout_revealed" in _actions(cfg)


def test_an_edited_test_is_reviewed_stored_and_merged_when_approved(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(
        env,
        monkeypatch,
        coder=_coder({IMPL: GOOD, TEST: EDITED}),
        audit=[{"stdout": APPROVE}, {"stdout": AUDIT_PASS}],
    )
    cfg = _cfg(env, test_review=True)
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "merged", out.reason
    review_prompt = _prompts(env, "audit")[0]
    assert "add(2, 1)" in review_prompt  # the test diff
    assert "return a + b" in review_prompt  # the code
    stored = (cfg.state_dir / "runs" / TASK).glob("test-changes-*.diff")
    assert any("add(2, 1)" in p.read_text() for p in stored)
    actions = _actions(cfg)
    assert "test_changed" in actions
    assert "test_review_approve" in actions
    # the sealed original stays in git, in the QA commit
    assert QA_TEST != EDITED


def test_a_first_reject_goes_back_to_the_coder_once_with_the_reason(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(
        env,
        monkeypatch,
        coder=[{"write": {IMPL: GOOD, TEST: EDITED}}, {"write": {TEST: QA_TEST}}],
        audit=[{"stdout": REJECT}, {"stdout": AUDIT_PASS}],
    )
    cfg = _cfg(env, test_review=True, max_coder_attempts=1)
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "merged", out.reason
    assert len(_prompts(env, "coder")) == 2  # one extra attempt even at the attempt limit
    assert "changed only to make it pass" in _prompts(env, "coder")[1]
    assert "test_review_reject" in _actions(cfg)


def test_a_second_reject_goes_to_the_owner(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(
        env,
        monkeypatch,
        coder=[{"write": {IMPL: GOOD, TEST: EDITED}}, {"write": {TEST: EDITED + "\n# again\n"}}],
        audit=[{"stdout": REJECT}, {"stdout": REJECT}],
    )
    cfg = _cfg(env, test_review=True)
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "disputed"
    assert len(_prompts(env, "coder")) == 2
    inbox = cfg.inbox.read_text()
    assert TASK in inbox
    assert "test review rejected twice" in inbox
    assert state.State(cfg).task(TASK)["status"] == "disputed"
    assert _actions(cfg).count("test_review_reject") == 2
    assert "test_review_escalated" in _actions(cfg)


def test_a_malformed_review_answer_is_a_reject(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(
        env,
        monkeypatch,
        coder=_coder({IMPL: GOOD, TEST: EDITED}),
        audit=[{"stdout": "looks fine to me"}, {"stdout": "still fine"}],
    )
    cfg = _cfg(env, test_review=True)
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "disputed"


def test_a_deleted_sealed_test_is_still_refused(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed = {TEST: "0" * 64}
    problems = gates.sealed_problems(env["repo"], "HEAD", sealed, allow_edits=True)
    assert problems == [f"sealed test changed or missing: {TEST}"]


def test_with_the_flag_off_a_changed_test_is_refused_as_before(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD, TEST: EDITED}))
    cfg = _cfg(env, test_review=False)
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "rejected"
    assert len(_prompts(env, "audit")) == 0
    assert "must not change" in _prompts(env, "coder")[0]
    assert "holdout_revealed" not in _actions(cfg)


def test_every_step_is_logged_in_order(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(
        env,
        monkeypatch,
        coder=_coder({IMPL: GOOD, TEST: EDITED}),
        audit=[{"stdout": APPROVE}, {"stdout": AUDIT_PASS}],
    )
    cfg = _cfg(env, test_review=True)
    assert pipeline.run_task(cfg, _task(env)).status == "merged"
    actions = _actions(cfg)
    order = ["holdout_revealed", "test_changed", "test_review_approve"]
    assert [a for a in actions if a in order] == order
    own = [r for r in _ops(cfg) if r["action"] != "inbox_post"]  # inbox notes target M-0001, ...
    assert all(r["actor"] == "orchestrator" and r["target"] == TASK for r in own)
    assert _git(env["repo"], "log", "-1", "--format=%s", "main")


def test_the_review_is_a_move_to_qa_and_back_with_a_ledger_row_each(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(
        env,
        monkeypatch,
        coder=_coder({IMPL: GOOD, TEST: EDITED}),
        audit=[{"stdout": APPROVE}, {"stdout": AUDIT_PASS}],
    )
    cfg = _cfg(env, test_review=True)
    assert pipeline.run_task(cfg, _task(env)).status == "merged"
    rows = [r for r in _ops(cfg) if r["action"] in ("move", "move_done")]
    assert [r["action"] for r in rows] == ["move", "move_done"]
    assert "-> qa: test_review" in rows[0]["reason"]
    assert "qa -> " in rows[1]["reason"]


def test_a_second_reject_reports_both_diffs_and_qas_proposed_fixes_to_the_owner(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(
        env,
        monkeypatch,
        coder=[{"write": {IMPL: GOOD, TEST: EDITED}}, {"write": {TEST: EDITED + "\n# again\n"}}],
        audit=[{"stdout": REJECT}, {"stdout": REJECT}],
    )
    cfg = _cfg(env, test_review=True)
    assert pipeline.run_task(cfg, _task(env)).status == "disputed"
    inbox = cfg.inbox.read_text()
    assert inbox.count("QA's proposed fix") == 2
    assert "test diff test-changes-1.diff" in inbox
    assert "test diff test-changes-2.diff" in inbox
    assert "add(2, 1)" in inbox


@pytest.fixture(autouse=True)
def _qa_runner_answers_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the QA role's runner: serve its open test reviews before a slice waits."""
    wait = moves.wait

    def served(cfg: Any, move_id: str, timeout_s: float, poll_s: float = 5.0) -> Any:
        moves.serve(cfg, "qa", {"test_review": pipeline.review_move}, once=True)
        return wait(cfg, move_id, timeout_s, poll_s)

    monkeypatch.setattr(moves, "wait", served)
