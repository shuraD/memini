"""The Memini ledger: SQL text in ``ledgers/`` loaded into an in-memory SQLite database.

The SQL files are the source of truth and are committed with the code, so the
ledger's history is the git history; one INSERT per line keeps merges line-based.

Command line::

    python -m tools.ledger change --author A --module M --summary S [--features ...]

appends the change row for the commit being prepared and stages it.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sqlite3
import subprocess
from datetime import date as _date
from pathlib import Path

LEDGER_DIR = Path(__file__).resolve().parents[1] / "ledgers"
ROW_FILES = ("features.sql", "bugs.sql", "changes.sql")
CHANGE_COLUMNS = (
    "date",
    "author",
    "module",
    "summary",
    "features",
    "bugs",
    "checks",
    "review",
    "approval",
)

_IDENT = re.compile(r"[a-z_][a-z0-9_]*")
_GIT = shutil.which("git") or "git"


def _git_env() -> dict[str, str]:
    """Return the environment for git calls with git's own hook variables removed.

    When git runs a hook it exports GIT_DIR / GIT_INDEX_FILE / GIT_WORK_TREE for the
    repository being committed. A tool run from a hook must resolve the repository
    from its working directory, never from those variables (B-006).
    """
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


# every character str.splitlines() breaks on; a ledger value is one line of text
_LINE_BREAKS = frozenset("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")


def _quote(value: object) -> str:
    if value is None:
        return "NULL"
    text = "".join(" " if c in _LINE_BREAKS else c for c in str(value))  # single-line rows
    return "'" + text.replace("'", "''") + "'"


def insert(table: str, **cols: object) -> str:
    """Return one deterministic INSERT statement on one line.

    Table and column names must be plain identifiers (they come from this module's
    callers, never from user text); every value is quoted and escaped.
    """
    for ident in (table, *cols):
        if not _IDENT.fullmatch(ident):
            msg = f"not a plain SQL identifier: {ident!r}"
            raise ValueError(msg)
    names = ", ".join(cols)
    values = ", ".join(_quote(v) for v in cols.values())
    # identifiers validated above; values quoted by _quote
    return f"INSERT INTO {table} ({names}) VALUES ({values});"  # noqa: S608 # nosec B608


class LedgerFormatError(ValueError):
    """A row file holds something other than one plain INSERT per line."""


# A row line is exactly what ``insert`` writes: one INSERT of literal values, nothing more.
# String literals are blanked first, so their content (';', '--', SQL words) cannot matter.
_STRING = re.compile(r"'(?:[^']|'')*'")
# one value: a blanked string literal, NULL or a number
_ROW_LINE = re.compile(
    r"INSERT INTO (?:feature|feature_test|bug|change) "
    r"\([a-z_]+(?:, [a-z_]+)*\) "
    r"VALUES \((?:''|NULL|-?\d+(?:\.\d+)?)(?:, (?:''|NULL|-?\d+(?:\.\d+)?))*\);"
)
_ROW_ACTIONS = frozenset(
    {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_SELECT}
)


def _rows_only(action: int, *_: object) -> int:
    return sqlite3.SQLITE_OK if action in _ROW_ACTIONS else sqlite3.SQLITE_DENY


def load_text(schema: str, rows: dict[str, str]) -> sqlite3.Connection:
    """Load a schema and row-file texts (``{file name: text}``) into an in-memory database.

    Row files are append-only history: every line is one plain ``INSERT INTO <table> (``
    (or a ``--`` comment) holding exactly one statement, and the database refuses anything
    that is not an insert, so an ``UPDATE``, ``DELETE``, ``REPLACE``, upsert or a second
    statement on the line can never rewrite an earlier row (B-foundation-009). A duplicate
    id fails on its key. Every guard loads through here, from files or from git.
    """
    db = sqlite3.connect(":memory:", isolation_level=None)  # no implicit BEGIN per insert
    db.executescript(schema)
    db.set_authorizer(_rows_only)
    for name in ROW_FILES:
        # split on \n only: a value may hold any other character; CRLF files still load
        for n, raw in enumerate(rows.get(name, "").split("\n"), 1):
            line = raw.removesuffix("\r")
            if any(c in _LINE_BREAKS for c in line):  # a hidden break shows as two lines
                msg = f"{name}:{n}: hidden line-break character inside a line"
                raise LedgerFormatError(msg)
            if not line.strip() or line.startswith("--"):
                continue
            if not _ROW_LINE.fullmatch(_STRING.sub("''", line)):
                msg = f"{name}:{n}: not exactly one plain row insert (ledger.insert format)"
                raise LedgerFormatError(msg)
            try:
                db.execute(line)  # one statement only; a second one raises
            except (sqlite3.DatabaseError, sqlite3.Warning) as exc:
                msg = f"{name}:{n}: {exc}"
                raise LedgerFormatError(msg) from exc
    db.set_authorizer(None)
    return db


def load(ledger_dir: Path = LEDGER_DIR) -> sqlite3.Connection:
    """Load ``ledgers/`` from disk (see :func:`load_text`)."""
    rows = {
        name: (ledger_dir / name).read_text(encoding="utf-8")
        for name in ROW_FILES
        if (ledger_dir / name).exists()
    }
    return load_text((ledger_dir / "schema.sql").read_text(encoding="utf-8"), rows)


def _next_change_id(db: sqlite3.Connection, day: str) -> str:
    prefix = f"C-{day.replace('-', '')}-"
    row = db.execute("SELECT max(id) FROM change WHERE id LIKE ?", (prefix + "%",)).fetchone()
    last = int(row[0].rsplit("-", 1)[1]) if row and row[0] else 0
    return f"{prefix}{last + 1:04d}"


def add_change(ledger_dir: Path = LEDGER_DIR, **fields: str | None) -> str:
    """Append exactly one change row (columns: ``CHANGE_COLUMNS``) and return its id."""
    unknown = set(fields) - set(CHANGE_COLUMNS)
    if unknown:
        msg = f"unknown change columns: {sorted(unknown)}"
        raise ValueError(msg)
    day = fields.get("date") or _date.today().isoformat()
    cid = _next_change_id(load(ledger_dir), day)
    row: dict[str, object] = {"id": cid, **{c: fields.get(c) for c in CHANGE_COLUMNS}, "date": day}
    with (ledger_dir / "changes.sql").open("a", encoding="utf-8") as fh:
        fh.write(insert("change", **row) + "\n")
    return cid


def next_id(db: sqlite3.Connection, table: str, module: str) -> str:
    """Return the next module-scoped id (``F-<module>-NNN`` / ``B-<module>-NNN``)."""
    prefix = {"feature": "F", "bug": "B"}[table] + f"-{module}-"
    query = f"SELECT max(id) FROM {table} WHERE id LIKE ?"  # noqa: S608 # nosec B608 — table from the dict above
    row = db.execute(query, (prefix + "%",)).fetchone()
    last = int(row[0].rsplit("-", 1)[1]) if row and row[0] else 0
    return f"{prefix}{last + 1:03d}"


def repo_ledger_dir() -> Path:
    """Return the ledger directory of the git repository containing the current directory."""
    top = subprocess.run(  # noqa: S603 — fixed args, no shell
        [_GIT, "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
        env=_git_env(),
    ).stdout.strip()
    return Path(top) / "ledgers"


def main(argv: list[str] | None = None) -> int:
    """Run the ledger command line (see the module docstring)."""
    ap = argparse.ArgumentParser(prog="tools.ledger")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ch = sub.add_parser("change", help="record the change row for the next commit")
    for name in CHANGE_COLUMNS:
        ch.add_argument(f"--{name}", required=name in ("author", "module", "summary"))
    args = ap.parse_args(argv)
    ledger_dir = repo_ledger_dir()
    cid = add_change(ledger_dir, **{c: getattr(args, c) for c in CHANGE_COLUMNS})
    subprocess.run(  # noqa: S603 — fixed args, no shell
        [_GIT, "add", str(ledger_dir / "changes.sql")], check=True, env=_git_env()
    )
    print(cid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
