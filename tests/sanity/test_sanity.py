"""Sanity tier: one typical pass and one typical fail per gate rule (owner 2026-11-12).

Runs on every commit in under 3 seconds; the full attack suites run in CI only.
In-memory ledgers and tmp_path only: no git, no network, no Neo4j.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from tools import ledger
from tools.ci import check_ledgers, review_gate
from tools.ledger import LedgerFormatError, insert, load_text

SCHEMA_SQL = (ledger.LEDGER_DIR / "schema.sql").read_text(encoding="utf-8")
UNIT_TEST = "tests/unit/test_example.py::test_example"


def _feature(fid: str = "F-sanity-001", status: str = "active") -> str:
    return insert(
        "feature",
        id=fid,
        description="a feature",
        layer="core",
        proof="owner 2026-11-12",
        status=status,
        owner_module="sanity",
        added_in="C-20261112-0001",
    )


def _change(cid: str = "C-20261112-0001", summary: str = "first change") -> str:
    return insert(
        "change", id=cid, date="2026-11-12", author="Claude", module="sanity", summary=summary
    )


def _bug(bid: str = "B-sanity-001", **cols: object) -> str:
    return insert("bug", id=bid, description="a bug", **cols)


def _db(
    features: list[str] | None = None,
    bugs: list[str] | None = None,
    changes: list[str] | None = None,
    feature_tests: list[str] | None = None,
) -> sqlite3.Connection:
    return load_text(
        SCHEMA_SQL,
        {
            "features.sql": "\n".join((features or []) + (feature_tests or [])),
            "bugs.sql": "\n".join(bugs or []),
            "changes.sql": "\n".join(changes or []),
        },
    )


# --- ledger format: append-only inserts ---------------------------------------


def test_plain_insert_row_loads() -> None:
    """Ledger format: a plain one-line INSERT row loads."""
    db = _db(changes=[_change()])
    assert db.execute("SELECT id FROM change").fetchall() == [("C-20261112-0001",)]


def test_update_line_is_refused() -> None:
    """Ledger format: an UPDATE line can never rewrite history."""
    text = _change() + "\nUPDATE change SET summary = 'x' WHERE id = 'C-20261112-0001';"
    with pytest.raises(LedgerFormatError):
        load_text(SCHEMA_SQL, {"changes.sql": text})


# --- tests for features and bugs ----------------------------------------------


def test_active_feature_without_test_is_reported() -> None:
    """Every active feature needs an existing test."""
    db = _db(features=[_feature()])
    assert check_ledgers.features_without_tests(db, set()) == ["F-sanity-001"]


def test_active_feature_with_test_passes() -> None:
    """Every active feature needs an existing test: a linked existing test satisfies it."""
    link = insert("feature_test", feature_id="F-sanity-001", test_id=UNIT_TEST)
    db = _db(features=[_feature()], feature_tests=[link])
    assert check_ledgers.features_without_tests(db, {UNIT_TEST}) == []


def test_fixed_bug_without_regression_is_reported() -> None:
    """Every fixed bug needs an existing regression test under tests/unit."""
    db = _db(bugs=[_bug(status="fixed", fixed_in="C-20261112-0001")])
    assert check_ledgers.fixed_bugs_without_regression(db, set()) == ["B-sanity-001"]


def test_fixed_bug_with_unit_regression_passes() -> None:
    """Every fixed bug needs a regression test: an existing tests/unit test satisfies it."""
    db = _db(bugs=[_bug(status="fixed", regression_test=UNIT_TEST, fixed_in="C-20261112-0001")])
    assert check_ledgers.fixed_bugs_without_regression(db, {UNIT_TEST}) == []


def test_withdrawn_bug_with_reason_needs_no_test() -> None:
    """A wrong bug is withdrawn with a reason, loads, and needs no regression test."""
    db = _db(bugs=[_bug(status="withdrawn", withdrawn_reason="not a bug: spec says so")])
    assert check_ledgers.fixed_bugs_without_regression(db, set()) == []


# --- history is never deleted or rewritten ------------------------------------


def test_appending_one_change_row_passes() -> None:
    """History is append-only: adding one change row deletes and rewrites nothing."""
    head = _db(bugs=[_bug(status="open")], changes=[_change()])
    new = _db(bugs=[_bug(status="open")], changes=[_change(), _change("C-20261112-0002")])
    assert check_ledgers.new_change_rows(head, new) == 1
    assert check_ledgers.deleted_change_rows(head, new) == []
    assert check_ledgers.deleted_bug_rows(head, new) == []
    assert check_ledgers.rewritten_change_rows(head, new) == []


def test_deleted_change_row_is_reported() -> None:
    """History is never deleted: a removed change row is reported."""
    head = _db(changes=[_change(), _change("C-20261112-0002")])
    new = _db(changes=[_change("C-20261112-0002")])
    assert check_ledgers.deleted_change_rows(head, new) == ["C-20261112-0001"]


def test_deleted_bug_row_is_reported() -> None:
    """A wrong bug is withdrawn, never deleted: a removed bug row is reported."""
    head = _db(bugs=[_bug(status="open")])
    new = _db()
    assert check_ledgers.deleted_bug_rows(head, new) == ["B-sanity-001"]


def test_rewritten_change_row_is_reported() -> None:
    """History is fixed: a change row whose content changed is reported."""
    head = _db(changes=[_change()])
    new = _db(changes=[_change(summary="rewritten summary")])
    assert check_ledgers.rewritten_change_rows(head, new) == ["C-20261112-0001"]


# --- removal of business logic needs owner approval ---------------------------


def test_lost_feature_without_approval_is_reported() -> None:
    """Removing an active feature without OWNER-APPROVED-REMOVAL is reported."""
    head = _db(features=[_feature()])
    new = _db(features=[_feature(status="superseded")])
    message = "remove feature\n"
    approval = check_ledgers.approval_from_message(message)
    assert check_ledgers.lost_features(head, new, approval) == ["F-sanity-001"]


def test_lost_feature_with_owner_approval_passes() -> None:
    """Removing an active feature with OWNER-APPROVED-REMOVAL is allowed."""
    head = _db(features=[_feature()])
    new = _db(features=[_feature(status="superseded")])
    message = "remove feature\n\nOWNER-APPROVED-REMOVAL: owner 2026-11-12 chat\n"
    approval = check_ledgers.approval_from_message(message)
    assert check_ledgers.lost_features(head, new, approval) == []


# --- protected check files ----------------------------------------------------

PROTECTED = ["tools/ci/check_ledgers.py"]


def test_protected_file_change_without_lines_is_refused() -> None:
    """A change to a protected check file needs CHECK-CHANGE and the owner's approval."""
    problems = check_ledgers.check_change_problems(PROTECTED, "tweak the gate\n")
    assert len(problems) == 2


def test_protected_file_change_with_both_lines_passes() -> None:
    """A declared and owner-approved change to a protected check file passes."""
    message = (
        "tweak the gate\n\n"
        "CHECK-CHANGE: the guard missed renamed files\n"
        "OWNER-APPROVED-CHECK-CHANGE: yes, fix it (owner 2026-11-12)\n"
    )
    assert check_ledgers.check_change_problems(PROTECTED, message) == []


# --- commit message explanation -----------------------------------------------

COMPLETE_MESSAGE = (
    "sanity: add the fast tier\n\n"
    "Requirement: every commit gets feedback in under 3 seconds\n"
    "Implemented: tests/sanity with one pass and one fail per rule\n"
    "Why: the attack suites move to CI only\n"
)


def test_message_missing_why_is_reported() -> None:
    """A commit message states Requirement, Implemented and Why: a missing Why is reported."""
    message = COMPLETE_MESSAGE.replace("Why: the attack suites move to CI only\n", "")
    problems = check_ledgers.explanation_problems(message)
    assert len(problems) == 1
    assert "Why" in problems[0]


def test_complete_message_passes() -> None:
    """A commit message with Requirement, Implemented and Why passes."""
    assert check_ledgers.explanation_problems(COMPLETE_MESSAGE) == []


# --- review gate --------------------------------------------------------------

SHA = "abc1234"
REVIEW = """# Review of abc1234

| Spec requirement | Implemented where | Status |
|---|---|---|
| R1 fast sanity tier | tests/sanity/test_sanity.py | done |

## Findings
{finding}

## Knowledge drift
None: docs match the code.

MERGE: yes
"""


def _review(tmp_path: Path, finding: str) -> list[str]:
    (tmp_path / f"{SHA}.md").write_text(REVIEW.format(finding=finding), encoding="utf-8")
    return review_gate.problems(tmp_path, SHA)


def test_complete_merge_yes_review_passes(tmp_path: Path) -> None:
    """Review gate: a complete MERGE: yes review allows the merge."""
    assert _review(tmp_path, "- NIT: a docstring could be shorter.") == []


def test_review_with_blocking_finding_is_refused(tmp_path: Path) -> None:
    """Review gate: an unresolved BLOCKING finding refuses the merge despite MERGE: yes."""
    problems = _review(tmp_path, "- BLOCKING: the guard skips renamed files.")
    assert any("BLOCKING" in p for p in problems), problems
