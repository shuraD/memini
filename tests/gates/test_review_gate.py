"""Pre-merge gate (owner 2026-11-12): the review verdict is verified, not trusted.

``reviews/<sha>.md`` must exist for the branch head (or the merge candidate), carry an
expected-vs-implemented table with no ``missing`` row, and no unresolved blocking
finding; a knowledge-drift section must be present.
"""

import os
import subprocess
from pathlib import Path

import pytest
from tools.ci import review_gate as rg

GOOD = """# Review of abc1234

## Standards
No blocking finding.

## Expected vs implemented
| Spec requirement | Implemented where | Status |
|---|---|---|
| schema as data | config/graph_schema.yaml; schema.load | done |
| scope filter | — | deferred (module ingest-units writes it) |

## Knowledge drift
None: AGENTS.md, docs/ARCHITECTURE.md and the spec match the code.

## Verdict
MERGE: yes
"""


def _write(tmp_path: Path, sha: str, text: str) -> Path:
    d = tmp_path / "reviews"
    d.mkdir(exist_ok=True)
    (d / f"{sha}.md").write_text(text)
    return d


def test_a_complete_review_passes(tmp_path: Path) -> None:
    assert rg.problems(_write(tmp_path, "abc1234", GOOD), "abc1234") == []


def test_missing_review_file_blocks(tmp_path: Path) -> None:
    (tmp_path / "reviews").mkdir()
    assert any("no review" in p for p in rg.problems(tmp_path / "reviews", "abc1234"))


def test_a_missing_requirement_row_blocks(tmp_path: Path) -> None:
    text = GOOD.replace("| deferred (module ingest-units writes it) |", "| missing |")
    assert any("missing" in p for p in rg.problems(_write(tmp_path, "abc1234", text), "abc1234"))


def test_a_deferred_row_without_reason_blocks(tmp_path: Path) -> None:
    text = GOOD.replace("| deferred (module ingest-units writes it) |", "| deferred |")
    assert any(
        "deferred" in p and "reason" in p
        for p in rg.problems(_write(tmp_path, "abc1234", text), "abc1234")
    )


def test_blocking_finding_or_no_verdict_blocks(tmp_path: Path) -> None:
    text = GOOD.replace("No blocking finding.", "BLOCKING: relationship properties have no proof")
    assert any(
        "blocking" in p.lower() for p in rg.problems(_write(tmp_path, "abc1234", text), "abc1234")
    )
    text = GOOD.replace("MERGE: yes", "MERGE: no")
    assert any(
        "verdict" in p.lower() for p in rg.problems(_write(tmp_path, "abc1234", text), "abc1234")
    )


def test_missing_drift_section_blocks(tmp_path: Path) -> None:
    text = GOOD.replace(
        "## Knowledge drift\nNone: AGENTS.md, docs/ARCHITECTURE.md and the spec match the code.\n",
        "",
    )
    assert any(
        "drift" in p.lower() for p in rg.problems(_write(tmp_path, "abc1234", text), "abc1234")
    )


# --- which commit the gate reviews (B-foundation-008) ---------------------------------

_CLEAN_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 — test-only, fixed git args
        [rg._GIT, *args], cwd=repo, capture_output=True, text=True, check=True, env=_CLEAN_ENV
    ).stdout.strip()


def _commit(repo: Path, path: str, message: str) -> str:
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(message)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "--short=7", "HEAD")


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "r"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "user.name", "t")
    return r


def test_a_review_only_head_is_judged_by_its_parents_review(repo: Path) -> None:
    """Recording the review is a commit (with its change row); the gate looks at the parent."""
    reviewed = _commit(repo, "src/x.py", "code")
    (repo / "ledgers").mkdir()
    (repo / "ledgers" / "changes.sql").write_text("-- change row of the review commit\n")
    _commit(repo, f"reviews/{reviewed}.md", "the review")
    assert rg.reviewed_commit(repo) == reviewed


def test_a_ledger_change_other_than_the_change_row_needs_its_own_review(repo: Path) -> None:
    _commit(repo, "src/x.py", "code")
    (repo / "ledgers").mkdir()
    (repo / "ledgers" / "features.sql").write_text("-- a feature row sneaked in\n")
    head = _commit(repo, "reviews/x.md", "the review")
    assert rg.reviewed_commit(repo) == head


def test_a_head_that_changes_code_needs_its_own_review(repo: Path) -> None:
    reviewed = _commit(repo, "src/x.py", "code")
    _commit(repo, f"reviews/{reviewed}.md", "the review")
    head = _commit(repo, "src/y.py", "more code after the review")
    assert rg.reviewed_commit(repo) == head
    head = _commit(repo, "reviews/other.md", "review + code")
    (repo / "src" / "z.py").write_text("sneaked in")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--amend", "--no-edit")
    assert rg.reviewed_commit(repo) == _git(repo, "rev-parse", "--short=7", "HEAD")


def test_a_root_commit_is_its_own_reviewed_commit(repo: Path) -> None:
    first = _commit(repo, "src/x.py", "first")
    assert rg.reviewed_commit(repo) == first


def test_a_review_commit_that_rewrites_an_earlier_change_row_needs_its_own_review(
    repo: Path,
) -> None:
    (repo / "ledgers").mkdir()
    (repo / "ledgers" / "changes.sql").write_text("-- C-1 original\n")
    reviewed = _commit(repo, "src/x.py", "code")
    (repo / "ledgers" / "changes.sql").write_text("-- C-1 REWRITTEN\n-- C-2 the review\n")
    head = _commit(repo, f"reviews/{reviewed}.md", "the review")
    assert rg.reviewed_commit(repo) == head


@pytest.mark.parametrize(
    "finding",
    [
        "- BLOCKING: the loader accepts REPLACE",
        "* **BLOCKING** the loader accepts REPLACE",
        "1. BLOCKING: the loader accepts REPLACE",
        "| guard | BLOCKING: bypass | partial |",
        "> BLOCKING: quoted",
    ],
)
def test_a_blocking_finding_in_a_list_table_or_quote_blocks(tmp_path: Path, finding: str) -> None:
    # review c4bda9c: only a line starting with BLOCKING was seen, so '- BLOCKING' passed
    text = GOOD.replace("No blocking finding.", finding)
    assert any(
        "blocking" in p.lower() for p in rg.problems(_write(tmp_path, "abc1234", text), "abc1234")
    )


def test_prose_that_mentions_blocking_does_not_block(tmp_path: Path) -> None:
    text = GOOD.replace("No blocking finding.", "No BLOCKING finding remains.")
    assert rg.problems(_write(tmp_path, "abc1234", text), "abc1234") == []
