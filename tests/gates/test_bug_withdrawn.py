"""Withdrawn bugs (owner 2026-11-12: "The history should be unchanged and then choose the
status of the bug as withdrawn").

R5.1 a bug may be ``withdrawn``, with a non-empty ``withdrawn_reason``; statuses are
     exactly ``open``, ``fixed`` and ``withdrawn``.
R5.2 a withdrawn bug needs no regression test.
R5.3 bug rows are never deleted (pre-commit: staged vs HEAD; ci: HEAD vs HEAD~1).
R5.4 a status change is allowed and is not a deletion.
"""

import contextlib
import io
import os
import shutil
import sqlite3
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from tools import ledger
from tools.ci import check_ledgers

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = (ROOT / "ledgers" / "schema.sql").read_text(encoding="utf-8")
GIT = shutil.which("git") or "git"
LOAD_ERRORS = (sqlite3.IntegrityError, ledger.LedgerFormatError)
UNIT_TEST = "tests/unit/test_x.py::test_regression"

# Every git call runs without git's hook variables, or it would hit the outer repo (B-006).
_CLEAN_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — test-only, fixed git args
        [GIT, *args], cwd=repo, capture_output=True, text=True, check=True, env=_CLEAN_ENV
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "r"
    (r / "ledgers").mkdir(parents=True)
    (r / "tools" / "ci").mkdir(parents=True)
    shutil.copy(ROOT / "ledgers" / "schema.sql", r / "ledgers" / "schema.sql")
    for n in ("features.sql", "bugs.sql", "changes.sql"):
        (r / "ledgers" / n).write_text("")
    for f in (
        "tools/__init__.py",
        "tools/ledger.py",
        "tools/ci/__init__.py",
        "tools/ci/check_ledgers.py",
    ):
        shutil.copy(ROOT / f, r / f)
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "user.name", "t")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed")
    return r


@contextlib.contextmanager
def _inside(directory: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def _check(repo: Path, stage: str) -> tuple[int, str]:
    """Run the guard in-process inside ``repo``, without the test-id checks."""
    out = io.StringIO()
    with _inside(repo), contextlib.redirect_stdout(out):
        code = check_ledgers.main(["--stage", stage, "--no-tests"])
    return code, out.getvalue()


def _change(repo: Path) -> None:
    """Append and stage this commit's one change row."""
    with _inside(repo), contextlib.redirect_stdout(io.StringIO()):
        ledger.main(
            [
                "change",
                "--author",
                "Claude",
                "--module",
                "m",
                "--summary",
                "s",
                "--date",
                "2026-11-12",
            ]
        )


def _bug(bid: str, status: str, **extra: object) -> str:
    """Return one bug row, filled the way its status requires."""
    cols: dict[str, object] = {"id": bid, "description": "d", "status": status}
    if status == "fixed":
        cols.update(regression_test=UNIT_TEST, fixed_in="C-1")
    if status == "withdrawn":
        cols["withdrawn_reason"] = "not a bug: expected behaviour"
    cols.update(extra)
    return ledger.insert("bug", **cols)


def _load_bugs(*lines: str) -> sqlite3.Connection:
    return ledger.load_text(SCHEMA, {"bugs.sql": "".join(line + "\n" for line in lines)})


def _commit_bugs(repo: Path, *lines: str) -> None:
    (repo / "ledgers" / "bugs.sql").write_text("".join(line + "\n" for line in lines))
    _change(repo)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "bugs\n\nChange-Author: Claude")


# ------------------------------------------------------------------ R5.1
def test_a_withdrawn_bug_with_a_reason_loads() -> None:
    """R5.1: a bug row with status 'withdrawn' and a non-empty reason loads."""
    db = _load_bugs(_bug("B-t-001", "withdrawn", withdrawn_reason="duplicate of B-t-002"))
    assert db.execute("SELECT status, withdrawn_reason FROM bug").fetchall() == [
        ("withdrawn", "duplicate of B-t-002")
    ]


def test_a_withdrawn_bug_loads_from_the_ledger_directory(tmp_path: Path) -> None:
    """R5.1: ``ledger.load`` from disk accepts a withdrawn bug with a reason."""
    shutil.copy(ROOT / "ledgers" / "schema.sql", tmp_path / "schema.sql")
    (tmp_path / "bugs.sql").write_text(_bug("B-t-001", "withdrawn") + "\n")
    db = ledger.load(tmp_path)
    assert db.execute("SELECT id FROM bug WHERE status = 'withdrawn'").fetchall() == [("B-t-001",)]


@pytest.mark.parametrize("reason", [None, ""], ids=["null", "empty"])
def test_a_withdrawn_bug_without_a_reason_fails_to_load(reason: str | None) -> None:
    """R5.1: a withdrawn bug row without a non-empty withdrawn_reason fails to load."""
    with pytest.raises(LOAD_ERRORS):
        _load_bugs(_bug("B-t-001", "withdrawn", withdrawn_reason=reason))


def test_a_withdrawn_bug_without_the_reason_column_fails_to_load() -> None:
    """R5.1: a withdrawn bug row that omits withdrawn_reason entirely fails to load."""
    line = ledger.insert("bug", id="B-t-001", description="d", status="withdrawn")
    with pytest.raises(LOAD_ERRORS):
        _load_bugs(line)


def test_a_withdrawn_bug_without_a_reason_fails_to_load_from_disk(tmp_path: Path) -> None:
    """R5.1: ``ledger.load`` from disk refuses a withdrawn bug without a reason."""
    shutil.copy(ROOT / "ledgers" / "schema.sql", tmp_path / "schema.sql")
    (tmp_path / "bugs.sql").write_text(_bug("B-t-001", "withdrawn", withdrawn_reason="") + "\n")
    with pytest.raises(LOAD_ERRORS):
        ledger.load(tmp_path)


@pytest.mark.parametrize("status", ["open", "fixed"])
def test_open_and_fixed_bugs_still_load(status: str) -> None:
    """R5.1: the statuses 'open' and 'fixed' stay valid."""
    db = _load_bugs(_bug("B-t-001", status))
    assert db.execute("SELECT status FROM bug").fetchall() == [(status,)]


@pytest.mark.parametrize(
    "status",
    [
        "deleted",
        "removed",
        "closed",
        "wontfix",
        "superseded",
        "Withdrawn",
        "WITHDRAWN",
        "withdrawn ",
        " withdrawn",
        "withdraw",
        "Open",
        "FIXED",
        "",
    ],
)
def test_any_other_bug_status_fails_to_load(status: str) -> None:
    """R5.1: any bug status other than open, fixed or withdrawn fails to load."""
    with pytest.raises(LOAD_ERRORS):
        _load_bugs(_bug("B-t-001", status, withdrawn_reason="r"))


# ------------------------------------------------------------------ R5.2
@pytest.mark.parametrize(
    "regression_test",
    [None, "tests/unit/test_x.py::test_gone", "tests/acceptance/test_x.py::test_a"],
    ids=["none", "missing", "outside-unit"],
)
def test_a_withdrawn_bug_needs_no_regression_test(regression_test: str | None) -> None:
    """R5.2: fixed_bugs_without_regression never reports a withdrawn bug."""
    db = _load_bugs(_bug("B-t-001", "withdrawn", regression_test=regression_test))
    assert check_ledgers.fixed_bugs_without_regression(db, set()) == []


def test_a_fixed_bug_beside_a_withdrawn_one_is_still_checked() -> None:
    """R5.2: withdrawn rows are exempt, fixed rows next to them are still reported."""
    db = _load_bugs(
        _bug("B-t-001", "withdrawn"),
        _bug("B-t-002", "fixed", regression_test="tests/unit/test_x.py::test_gone"),
    )
    assert check_ledgers.fixed_bugs_without_regression(db, {UNIT_TEST}) == ["B-t-002"]


# ------------------------------------------------------------------ R5.3
@pytest.mark.parametrize("status", ["open", "fixed", "withdrawn"])
@pytest.mark.parametrize("stage", ["pre-commit", "ci"])
def test_deleting_a_bug_row_fails_the_guard(repo: Path, stage: str, status: str) -> None:
    """R5.3: a bug id present before and missing after fails pre-commit and ci, any status."""
    _commit_bugs(repo, _bug("B-t-001", status), _bug("B-t-002", "open"))
    (repo / "ledgers" / "bugs.sql").write_text(_bug("B-t-002", "open") + "\n")
    _change(repo)
    _git(repo, "add", "-A")
    if stage == "ci":
        _git(repo, "commit", "-q", "-m", "drop bug\n\nChange-Author: Claude")
    code, out = _check(repo, stage)
    assert code != 0 and "B-t-001" in out


@pytest.mark.parametrize("stage", ["pre-commit", "ci"])
def test_renumbering_a_bug_counts_as_deleting_it(repo: Path, stage: str) -> None:
    """R5.3: replacing a bug row by one with a new id still deletes the old id."""
    _commit_bugs(repo, _bug("B-t-001", "open"))
    (repo / "ledgers" / "bugs.sql").write_text(_bug("B-t-002", "open") + "\n")
    _change(repo)
    _git(repo, "add", "-A")
    if stage == "ci":
        _git(repo, "commit", "-q", "-m", "renumber\n\nChange-Author: Claude")
    code, out = _check(repo, stage)
    assert code != 0 and "B-t-001" in out


@pytest.mark.parametrize("stage", ["pre-commit", "ci"])
def test_emptying_the_bug_ledger_fails_the_guard(repo: Path, stage: str) -> None:
    """R5.3: deleting every bug row reports each deleted id."""
    _commit_bugs(repo, _bug("B-t-001", "open"), _bug("B-t-002", "fixed"))
    (repo / "ledgers" / "bugs.sql").write_text("")
    _change(repo)
    _git(repo, "add", "-A")
    if stage == "ci":
        _git(repo, "commit", "-q", "-m", "empty\n\nChange-Author: Claude")
    code, out = _check(repo, stage)
    assert code != 0 and "B-t-001" in out and "B-t-002" in out


# ------------------------------------------------------------------ R5.4
@pytest.mark.parametrize(
    ("before", "after"),
    [("open", "fixed"), ("open", "withdrawn"), ("fixed", "withdrawn")],
)
@pytest.mark.parametrize("stage", ["pre-commit", "ci"])
def test_changing_a_bug_status_is_not_a_deletion(
    repo: Path, stage: str, before: str, after: str
) -> None:
    """R5.4: a bug status change passes pre-commit and ci and is not reported as deleted."""
    _commit_bugs(repo, _bug("B-t-001", before))
    (repo / "ledgers" / "bugs.sql").write_text(_bug("B-t-001", after) + "\n")
    _change(repo)
    _git(repo, "add", "-A")
    if stage == "ci":
        _git(repo, "commit", "-q", "-m", "status\n\nChange-Author: Claude")
    code, out = _check(repo, stage)
    assert code == 0, out
    assert "B-t-001" not in out
