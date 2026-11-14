"""Gate: local ledger databases (``ledgers/*.sqlite``) are ignored by git; the SQL is not.

Requirement (pool task T-063, private repo): "ledgers/*.sqlite is ignored by git".
Acceptance: ``git check-ignore ledgers/x.sqlite`` succeeds. A SQLite file is built locally
from the ``ledgers/*.sql`` history (AGENTS.md "ledgers/: features, bugs, changes (SQL)"), so
it must never be committed: it would duplicate the history in a binary nobody reviews. The
SQL ledgers themselves are the record and must never be ignored (owner rule 4, never delete
history).

How: ``git check-ignore --no-index`` in the repository, with every ``GIT_*`` variable
removed, so the answer comes from the committed ignore rules alone.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GIT = shutil.which("git") or "git"
_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _ignored(path: str) -> bool:
    res = subprocess.run(  # noqa: S603 — test-only, fixed git arguments
        [GIT, "check-ignore", "--no-index", "-q", path],
        cwd=ROOT,
        env=_ENV,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert res.returncode in (0, 1), res.stderr
    return res.returncode == 0


@pytest.mark.parametrize(
    "path", ["ledgers/x.sqlite", "ledgers/features.sqlite", "ledgers/ledger.sqlite"]
)
def test_local_ledger_database_is_ignored(path: str) -> None:
    """T-063: a SQLite database built in ledgers/ is ignored, so it is never committed."""
    assert _ignored(path), f"{path} is not ignored by git"


@pytest.mark.parametrize(
    "path",
    ["ledgers/features.sql", "ledgers/bugs.sql", "ledgers/changes.sql", "ledgers/schema.sql"],
)
def test_sql_ledgers_are_never_ignored(path: str) -> None:
    """Owner rule 4: the SQL ledgers are the history; the ignore rule never hides them."""
    assert not _ignored(path), f"{path} is ignored by git"
