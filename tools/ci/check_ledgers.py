"""Ledger guards, run on every commit (pre-commit hook, commit-msg hook and CI).

1. Every active feature has at least one existing test.
2. Every fixed bug has an existing regression test in a tier that runs before every
   merge (``tests/sanity``, ``tests/unit``, ``tests/gates``), never in opt-in tests.
3. No feature in HEAD's ledger is missing or superseded without
   ``OWNER-APPROVED-REMOVAL: <ref>`` in the commit message.
4. The commit adds exactly one change row, deletes or rewrites none, deletes no bug
   row (a wrong bug is withdrawn with a reason), and its ``Change-Author`` trailer
   matches that row's author. Every ledger is loaded by ``tools.ledger.load_text``,
   which accepts only one plain row insert per line (history is append-only).
5. The message states ``Requirement:``, ``Implemented:`` and ``Why:`` (owner 2026-11-12:
   any decision with its explanation; any change as implementation vs requirement).
6. A commit touching a file in ``tools/ci/protected.txt`` (edit, delete, rename, mode)
   carries ``CHECK-CHANGE: <reason>`` and the owner's ``OWNER-APPROVED-CHECK-CHANGE``.

Stages: ``pre-commit`` (staged vs HEAD), ``commit-msg`` (needs the message file),
``ci`` (HEAD vs its parent: audits the last commit even if it bypassed the hooks).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from tools import ledger
from tools.ledger import ROW_FILES, _git_env

APPROVAL_RE = re.compile(r"^OWNER-APPROVED-REMOVAL:\s*(.+?)\s*$", re.MULTILINE)
AUTHOR_RE = re.compile(r"^Change-Author:\s*(.+?)\s*$", re.MULTILINE)
# test tiers (owner 2026-11-12): sanity + unit run every commit, gates in CI before merge;
# a regression test must live in one of them, never in opt-in acceptance or live tests
TEST_PREFIXES = ("tests/sanity/", "tests/unit/", "tests/gates/")
_GIT = shutil.which("git") or "git"


# ------------------------------------------------------------------ guards (pure)
def features_without_tests(db: sqlite3.Connection, existing_tests: set[str]) -> list[str]:
    """Return active features with no existing test linked to them."""
    rows = db.execute(
        "SELECT f.id, t.test_id FROM feature f LEFT JOIN feature_test t ON t.feature_id = f.id "
        "WHERE f.status = 'active' ORDER BY f.id"  # planned (deferred) features have no test yet
    ).fetchall()
    covered: dict[str, bool] = {}
    for fid, test_id in rows:
        covered[fid] = covered.get(fid, False) or (test_id in existing_tests)
    return [fid for fid, ok in covered.items() if not ok]


def fixed_bugs_without_regression(db: sqlite3.Connection, existing_tests: set[str]) -> list[str]:
    """Return fixed bugs whose regression test is missing or outside the test tiers.

    A test outside ``TEST_PREFIXES`` (e.g. opt-in acceptance or live tests) would exist
    but never run before a merge.
    """
    rows = db.execute(
        "SELECT id, regression_test FROM bug WHERE status = 'fixed' ORDER BY id"
    ).fetchall()
    return [
        bid
        for bid, test_id in rows
        if not test_id or not test_id.startswith(TEST_PREFIXES) or test_id not in existing_tests
    ]


def lost_features(
    head: sqlite3.Connection, new: sqlite3.Connection, approval: str | None
) -> list[str]:
    """Return features active in ``head`` but not in ``new``, unless approved."""
    if approval:
        return []
    before = {r[0] for r in head.execute("SELECT id FROM feature WHERE status = 'active'")}
    after = {r[0] for r in new.execute("SELECT id FROM feature WHERE status = 'active'")}
    return sorted(before - after)


def _change_ids(db: sqlite3.Connection) -> set[str]:
    return {r[0] for r in db.execute("SELECT id FROM change")}


def new_change_rows(head: sqlite3.Connection, new: sqlite3.Connection) -> int:
    """Return how many change rows were added (exactly one per commit)."""
    return len(_change_ids(new) - _change_ids(head))


def deleted_change_rows(head: sqlite3.Connection, new: sqlite3.Connection) -> list[str]:
    """Return change rows that disappeared; never allowed (history is never deleted)."""
    return sorted(_change_ids(head) - _change_ids(new))


def deleted_bug_rows(head: sqlite3.Connection, new: sqlite3.Connection) -> list[str]:
    """Return bug ids that disappeared; a wrong bug is withdrawn, never deleted."""
    before = {r[0] for r in head.execute("SELECT id FROM bug")}
    return sorted(before - {r[0] for r in new.execute("SELECT id FROM bug")})


def rewritten_change_rows(head: sqlite3.Connection, new: sqlite3.Connection) -> list[str]:
    """Return change rows present before and after whose content changed (history is fixed)."""
    before = {r[0]: r for r in head.execute("SELECT * FROM change")}
    return sorted(r[0] for r in new.execute("SELECT * FROM change") if before.get(r[0], r) != r)


def author_mismatch(
    head: sqlite3.Connection, new: sqlite3.Connection, message: str
) -> tuple[str, str] | None:
    """Return ``(row author, trailer author)`` when they disagree, else ``None``."""
    m = AUTHOR_RE.search(message)
    added = _change_ids(new) - _change_ids(head)
    if not m or len(added) != 1:
        return None
    (cid,) = added
    row_author = new.execute("SELECT author FROM change WHERE id = ?", (cid,)).fetchone()[0]
    return None if row_author == m.group(1) else (row_author, m.group(1))


EXPLANATION_FIELDS = ("Requirement", "Implemented", "Why")


def explanation_problems(message: str) -> list[str]:
    """Return what is missing for the explanation rule.

    Every change states the requirement it implements, what was implemented, and why
    (owner 2026-11-12: any decision with its explanation).
    """
    problems = []
    for field in EXPLANATION_FIELDS:
        m = re.search(rf"^{field}:[ \t]*(\S.*)$", message, re.MULTILINE)
        if not m:
            problems.append(f"commit message needs a non-empty '{field}: ...' line")
    return problems


PROTECTED_FILE = Path(__file__).resolve().parent / "protected.txt"
CHECK_CHANGE_RE = re.compile(r"^CHECK-CHANGE:[ \t]*(\S.*)$", re.MULTILINE)
OWNER_CHECK_RE = re.compile(r"^OWNER-APPROVED-CHECK-CHANGE:[ \t]*(\S.*)$", re.MULTILINE)


def protected_paths(path: Path = PROTECTED_FILE) -> list[str]:
    """Return the protected check paths (a trailing ``/`` protects a directory)."""
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]


def _protects(entry: str, path: str) -> bool:
    """Return True when the protected ``entry`` covers ``path``.

    ``dir/`` covers the directory and everything under it; ``**/name`` covers a file of
    that name in any directory; anything else is an exact path.
    """
    if entry.startswith("**/"):
        return path.rsplit("/", 1)[-1] == entry[3:]
    if entry.endswith("/"):
        return path == entry.rstrip("/") or path.startswith(entry)
    return path == entry


def check_change_problems(
    changed: list[str], message: str, path: Path = PROTECTED_FILE
) -> list[str]:
    """Return the problems of a commit that changes a protected check file.

    Owner rule: the checks are never changed to make something pass. A change to a check
    must be declared and reasoned in the commit (``CHECK-CHANGE: <reason>``) and approved
    by the owner (``OWNER-APPROVED-CHECK-CHANGE: <the owner's words, date>``): no agent
    approves its own gate change (owner 2026-11-12).
    """
    protected = protected_paths(path)
    touched = sorted(f for f in changed if any(_protects(p, f) for p in protected))
    if not touched:
        return []
    out: list[str] = []
    if not CHECK_CHANGE_RE.search(message):
        out.append(
            "protected check file(s) changed without a non-empty 'CHECK-CHANGE: <reason>' line: "
            + ", ".join(touched)
        )
    if not OWNER_CHECK_RE.search(message):
        out.append(
            "protected check file(s) changed without the owner's approval "
            "'OWNER-APPROVED-CHECK-CHANGE: <owner words, date>': " + ", ".join(touched)
        )
    return out


def check_change_problems_for_commits(
    commits: list[tuple[str, str, list[str]]], path: Path = PROTECTED_FILE
) -> list[str]:
    """Apply :func:`check_change_problems` to every ``(sha, message, files)`` of a branch."""
    out: list[str] = []
    for sha, message, files in commits:
        out += [f"commit {sha}: {p}" for p in check_change_problems(files, message, path)]
    return out


def _branch_commits() -> list[tuple[str, str, list[str]]]:
    """Return ``(sha, message, changed files)`` for every commit since the merge base with main."""

    def git(*args: str) -> str:
        return subprocess.run(  # noqa: S603 — fixed args, no shell
            [_GIT, *args], capture_output=True, text=True, check=False, env=_git_env()
        ).stdout

    base = git("merge-base", "main", "HEAD").strip() or git("rev-parse", "HEAD~1").strip()
    shas = [s for s in git("rev-list", "--reverse", f"{base}..HEAD").split() if s]
    return [
        (
            sha[:7],
            git("log", "-1", "--format=%B", sha),
            [
                f
                for f in git("diff", "--name-only", "--no-renames", "-z", f"{sha}~1", sha).split(
                    "\0"
                )
                if f
            ],
        )
        for sha in shas
    ]


def approval_from_message(message: str) -> str | None:
    """Return the ``OWNER-APPROVED-REMOVAL`` reference in a commit message, if any."""
    m = APPROVAL_RE.search(message)
    return m.group(1) if m else None


# ------------------------------------------------------------------ git / pytest
def _git_show(rev_path: str) -> str | None:
    res = subprocess.run(  # noqa: S603 — fixed args, no shell
        [_GIT, "show", rev_path], capture_output=True, text=True, check=False, env=_git_env()
    )
    return res.stdout if res.returncode == 0 else None


class LedgerUnavailableError(RuntimeError):
    """The ledger could not be read from git at the requested revision."""


def _ledger_at(prefix: str) -> sqlite3.Connection:
    """Load the ledger as stored at ``prefix`` (``HEAD:``, ``HEAD~1:`` or ``:`` = staged).

    Fails loudly when the revision cannot be read (B-007: a checkout without history
    made the guard load an EMPTY ledger and pass or crash for the wrong reason).
    """
    schema = _git_show(f"{prefix}ledgers/schema.sql")
    if schema is None:
        msg = (
            f"cannot read ledgers/schema.sql at {prefix!r} — the checkout has no git history "
            "for that revision (CI needs fetch-depth: 0 / the full repository, not a file copy)"
        )
        raise LedgerUnavailableError(msg)
    rows: dict[str, str] = {}
    for name in ROW_FILES:
        text = _git_show(f"{prefix}ledgers/{name}")
        if text is None and prefix != ":":
            msg = f"cannot read ledgers/{name} at {prefix!r}"
            raise LedgerUnavailableError(msg)
        rows[name] = text or ""
    return ledger.load_text(schema, rows)  # the strict loader (B-foundation-009)


def _existing_tests() -> set[str]:
    res = subprocess.run(
        # -P: a root pytest.py cannot fake the collection (B-foundation-016)
        [sys.executable, "-P", "-m", "pytest", "--collect-only", "-q", "-o", "addopts=", "tests"],
        capture_output=True,
        text=True,
        check=False,
    )
    ids: set[str] = set()
    for line in res.stdout.splitlines():
        if "::" in line:
            ids.add(line.strip())
            ids.add(line.strip().split("[", 1)[0])  # parametrized → base id too
    return ids


# ------------------------------------------------------------------ stages
def _test_problems(staged: sqlite3.Connection) -> list[str]:
    tests = _existing_tests()
    return [f"feature {f} has no existing test" for f in features_without_tests(staged, tests)] + [
        f"fixed bug {b} has no existing regression test under {', '.join(TEST_PREFIXES)}"
        for b in fixed_bugs_without_regression(staged, tests)
    ]


def _change_row_problems(head: sqlite3.Connection, staged: sqlite3.Connection) -> list[str]:
    problems = [
        f"change row {c} was deleted (history is never deleted)"
        for c in deleted_change_rows(head, staged)
    ]
    problems += [
        f"bug row {b} was deleted (withdraw it with a reason instead)"
        for b in deleted_bug_rows(head, staged)
    ]
    problems += [
        f"change row {c} was rewritten (history is append-only)"
        for c in rewritten_change_rows(head, staged)
    ]
    n = new_change_rows(head, staged)
    if n != 1:
        problems.append(
            f"a commit must add exactly one change row (found {n}): "
            "python -m tools.ledger change --author ... --module ... --summary ..."
        )
    return problems


def _changed_files(ci: bool) -> list[str]:
    # --no-renames: a renamed protected file is reported under its old path too;
    # -z: paths unquoted and NUL-separated, so a name with a tab, quote or non-ASCII
    # byte is still matched against the protected set
    diff = ["diff", "--name-only", "--no-renames", "-z"]
    args = [*diff, "HEAD~1", "HEAD"] if ci else [*diff, "--cached"]
    res = subprocess.run(  # noqa: S603 — fixed args, no shell
        [_GIT, *args], capture_output=True, text=True, check=False, env=_git_env()
    )
    return [p for p in res.stdout.split("\0") if p]


def _ledger_problems_for_commits(commits: list[tuple[str, str, list[str]]]) -> list[str]:
    """Run the ledger guards for every commit of the branch against its own parent.

    A commit made with ``--no-verify`` in the middle of a branch is audited as strictly
    as the last one: change rows, bug rows, and lost features (approved only by that
    commit's own message).
    """
    out: list[str] = []
    for sha, message, _ in commits:
        try:
            parent, mine = _ledger_at(f"{sha}~1:"), _ledger_at(f"{sha}:")
        except (LedgerUnavailableError, ledger.LedgerFormatError) as e:
            out.append(f"commit {sha}: {e}")
            continue
        found = _change_row_problems(parent, mine) + [
            f"feature {f} removed/superseded without 'OWNER-APPROVED-REMOVAL: <ref>'"
            for f in lost_features(parent, mine, approval_from_message(message))
        ]
        out += [f"commit {sha}: {p}" for p in found]
    return out


def _message_problems(
    head: sqlite3.Connection, staged: sqlite3.Connection, message: str
) -> list[str]:
    problems: list[str] = []
    problems += check_change_problems(_changed_files(ci=False), message)
    if not AUTHOR_RE.search(message):
        problems.append("commit message needs a 'Change-Author: <who>' trailer")
    problems += explanation_problems(message)
    mismatch = author_mismatch(head, staged, message)
    if mismatch:
        problems.append(
            f"change row author {mismatch[0]!r} != Change-Author trailer {mismatch[1]!r}"
        )
    problems += [
        f"feature {f} removed/superseded without 'OWNER-APPROVED-REMOVAL: <ref>'"
        for f in lost_features(head, staged, approval_from_message(message))
    ]
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run the guards for one stage; print every problem; return 1 if any."""
    ap = argparse.ArgumentParser(prog="check_ledgers")
    ap.add_argument("--stage", choices=["pre-commit", "commit-msg", "ci"], required=True)
    ap.add_argument("message_file", nargs="?")
    ap.add_argument("--no-tests", action="store_true", help="skip test-id checks (guard tests)")
    args = ap.parse_args(argv)
    ci = args.stage == "ci"
    try:
        head = _ledger_at("HEAD~1:" if ci else "HEAD:")
        staged = _ledger_at("HEAD:" if ci else ":")
    except (LedgerUnavailableError, ledger.LedgerFormatError) as e:
        print(f"ledger check failed: {e}")
        return 1

    problems: list[str] = []
    if ci:
        commits = _branch_commits()
        problems += check_change_problems_for_commits(commits)
        problems += _ledger_problems_for_commits(commits)
    if args.stage == "commit-msg":
        message = Path(args.message_file).read_text(encoding="utf-8") if args.message_file else ""
        problems += _message_problems(head, staged, message)
    else:
        if not args.no_tests:
            problems += _test_problems(staged)
        problems += _change_row_problems(head, staged)
    for p in problems:
        print(f"ledger check failed: {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
