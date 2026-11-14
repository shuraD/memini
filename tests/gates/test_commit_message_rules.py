"""Owner rule (2026-11-12): any decision with its explanation, any change with the
requirement it implements and what was implemented. Enforced by the commit-msg hook."""

from tools.ci import check_ledgers as cl

GOOD = """subject

Requirement: SPEC-foundation §Ledgers — one change row per commit.
Implemented: tools/ledger.py add_change(); the hooks call it; one test per guard.
Why: the ledger is the audit trail the owner asked for; without a row the change is invisible.

Change-Author: Claude
"""


def test_message_with_requirement_implemented_and_why_passes() -> None:
    assert cl.explanation_problems(GOOD) == []


def test_message_missing_requirement_or_implemented_or_why_is_rejected() -> None:
    for drop in ("Requirement:", "Implemented:", "Why:"):
        msg = "\n".join(ln for ln in GOOD.splitlines() if not ln.startswith(drop))
        problems = cl.explanation_problems(msg)
        assert problems and drop[:-1] in problems[0], drop


def test_empty_fields_are_rejected() -> None:
    emptied = "\n".join("Why:   " if ln.startswith("Why:") else ln for ln in GOOD.splitlines())
    assert cl.explanation_problems(emptied) != []
