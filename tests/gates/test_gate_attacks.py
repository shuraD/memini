# ruff: noqa: S608 - the attack inputs are SQL text by design
"""Adversarial tests of the merge gates, written from the rules R1-R4 (not from the code).

R1 ledger rows are append-only plain INSERT lines; R2 committed change rows never change
and each commit adds exactly one; R3 protected check files need CHECK-CHANGE and owner
approval lines; R4 the review gate refuses blocking findings and incomplete reviews.
Every git subprocess runs with all ``GIT_*`` variables stripped (B-006).
"""

import contextlib
import io
import os
import shutil
import sqlite3
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from tools import ledger
from tools.ci import check_ledgers, review_gate

ROOT = Path(__file__).resolve().parents[2]
GIT = shutil.which("git") or "git"
_CLEAN_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
SCHEMA = (ROOT / "ledgers" / "schema.sql").read_text()


@pytest.fixture(autouse=True)
def _no_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip git's hook variables so in-process gate calls never touch the real repo."""
    for key in list(os.environ):
        if key.startswith("GIT_"):
            monkeypatch.delenv(key)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 - test-only, fixed git args
        [GIT, *args], cwd=repo, capture_output=True, text=True, check=True, env=_CLEAN_ENV
    ).stdout.strip()


@contextlib.contextmanager
def _inside(directory: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def _init_repo(r: Path) -> None:
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "user.name", "t")
    _git(r, "config", "core.autocrlf", "false")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "seed")


# --------------------------------------------------------------------------------------
# Ledger rows used by R1 and R2
# --------------------------------------------------------------------------------------
C1 = "C-20261112-0001"
C2 = "C-20261112-0002"
C3 = "C-20261112-0003"
C4 = "C-20261112-0004"


def change_line(cid: str, summary: str = "s", **over: object) -> str:
    cols: dict[str, object] = {
        "id": cid,
        "date": "2026-11-12",
        "author": "Claude",
        "module": "m",
        "summary": summary,
        "features": None,
        "bugs": None,
        "checks": None,
        "review": None,
        "approval": None,
    }
    cols.update(over)
    return ledger.insert("change", **cols)


FEATURE = ledger.insert(
    "feature",
    id="F-t-001",
    description="d",
    layer="infra",
    proof="p",
    status="active",
    added_in=C1,
)
COLS = "(id, date, author, module, summary)"
REWRITE = f"VALUES ('{C1}', '2026-11-12', 'Claude', 'm', 'REWRITTEN')"
NEW = f"INSERT INTO change {COLS} VALUES ('{C2}', '2026-11-12', 'Claude', 'm', 's')"
NEW3 = f"INSERT INTO change {COLS} VALUES ('{C3}', '2026-11-12', 'Claude', 'm', 's')"
REPLACE_C1 = f"REPLACE INTO change {COLS} {REWRITE};"
UPDATE_C1 = f"UPDATE change SET summary = 'REWRITTEN' WHERE id = '{C1}';"
SUPERSEDE = (
    "INSERT OR REPLACE INTO feature (id, description, layer, proof, status, added_in) "
    f"VALUES ('F-t-001', 'd', 'infra', 'p', 'superseded', '{C1}');"
)


def _rows(file: str, extra: str) -> dict[str, str]:
    rows = {"features.sql": FEATURE + "\n", "bugs.sql": "", "changes.sql": change_line(C1) + "\n"}
    rows[file] += extra + "\n"
    return rows


R1_ATTACKS = [
    # history rewrites in the first (only) statement
    pytest.param("changes.sql", REPLACE_C1, id="replace-into"),
    pytest.param(
        "changes.sql", f"INSERT OR REPLACE INTO change {COLS} {REWRITE};", id="or-replace"
    ),
    pytest.param(
        "changes.sql",
        f"INSERT OR IGNORE INTO change {COLS} VALUES ('{C2}', '2026-11-12', 'Claude', 'm', 's');",
        id="or-ignore",
    ),
    pytest.param("features.sql", SUPERSEDE, id="feature-or-replace"),
    pytest.param(
        "changes.sql",
        f"WITH t AS (SELECT 1) REPLACE INTO change {COLS} "
        f"SELECT '{C1}', '2026-11-12', 'Claude', 'm', 'REWRITTEN' FROM t;",
        id="cte-replace",
    ),
    pytest.param(
        "changes.sql",
        f"WITH t AS (SELECT 1) INSERT INTO change {COLS} "
        f"SELECT '{C2}', '2026-11-12', 'Claude', 'm', 's' FROM t;",
        id="cte-insert",
    ),
    pytest.param(
        "changes.sql",
        f"INSERT INTO change {COLS} {REWRITE} ON CONFLICT(id) DO UPDATE SET summary = 'REWRITTEN';",
        id="upsert-do-update",
    ),
    pytest.param(
        "changes.sql",
        f"INSERT INTO change {COLS} {REWRITE} ON CONFLICT DO NOTHING;",
        id="upsert-do-nothing",
    ),
    pytest.param(
        "changes.sql",
        f"INSERT INTO change {COLS} SELECT '{C2}', date, author, module, summary FROM change;",
        id="insert-select",
    ),
    # a second statement after a valid insert
    pytest.param("changes.sql", f"{NEW}; {REPLACE_C1}", id="second-replace"),
    pytest.param(
        "changes.sql",
        f"{NEW}; INSERT OR REPLACE INTO change {COLS} {REWRITE};",
        id="second-or-replace",
    ),
    pytest.param("changes.sql", f"{NEW}; {UPDATE_C1}", id="second-update"),
    pytest.param(
        "changes.sql", f"{NEW}; DELETE FROM change WHERE id = '{C1}';", id="second-delete"
    ),
    pytest.param("changes.sql", f"{NEW}; {NEW3};", id="second-plain-insert"),
    pytest.param("changes.sql", f"{NEW}; SELECT 1;", id="second-select"),
    pytest.param(
        "changes.sql",
        f"{NEW}; WITH t AS (SELECT 1) REPLACE INTO change {COLS} "
        f"SELECT '{C1}', '2026-11-12', 'Claude', 'm', 'REWRITTEN' FROM t;",
        id="second-cte-replace",
    ),
    pytest.param(
        "changes.sql",
        f"{NEW}; INSERT OR REPLACE INTO change {COLS} "
        f"SELECT id, date, author, module, 'REWRITTEN' FROM change WHERE id = '{C1}';",
        id="second-replace-select",
    ),
    pytest.param(
        "changes.sql",
        f"{NEW} ON CONFLICT DO NOTHING; INSERT OR IGNORE INTO change {COLS} {REWRITE};",
        id="second-or-ignore",
    ),
    pytest.param("changes.sql", f"{NEW}; DROP TABLE change;", id="second-drop"),
    pytest.param("changes.sql", f"{NEW}; ALTER TABLE change RENAME TO old;", id="second-alter"),
    pytest.param(
        "changes.sql",
        f"{NEW}; CREATE TEMP TABLE change (id TEXT, date TEXT, author TEXT, module TEXT, "
        "summary TEXT, features TEXT, bugs TEXT, checks TEXT, review TEXT, approval TEXT);",
        id="temp-table-shadow",
    ),
    pytest.param(
        "changes.sql",
        f"{NEW}; CREATE TEMP VIEW change AS SELECT id, date, author, module, "
        "'REWRITTEN' AS summary, features, bugs, checks, review, approval FROM main.change;",
        id="temp-view-shadow",
    ),
    pytest.param(
        "changes.sql",
        f"{NEW}; CREATE TRIGGER t AFTER INSERT ON change BEGIN {UPDATE_C1} END;",
        id="trigger",
    ),
    pytest.param(
        "changes.sql",
        f"{NEW}; CREATE TEMP TRIGGER t AFTER INSERT ON change BEGIN {UPDATE_C1} END;",
        id="temp-trigger",
    ),
    pytest.param("changes.sql", f"{NEW}; ATTACH DATABASE ':memory:' AS aux;", id="attach"),
    pytest.param("changes.sql", f"{NEW}; PRAGMA foreign_keys = OFF;", id="pragma"),
    pytest.param("changes.sql", f"{NEW}; PRAGMA writable_schema = ON;", id="pragma-schema"),
    pytest.param("changes.sql", f"{NEW}; SAVEPOINT a; RELEASE a;", id="savepoint"),
    pytest.param("changes.sql", f"{NEW}; VACUUM;", id="vacuum"),
    pytest.param("changes.sql", f"{NEW}; /* note */ {UPDATE_C1}", id="block-comment-then-update"),
    pytest.param("changes.sql", f"{NEW}; replace into change {COLS} {REWRITE};", id="lower-second"),
    pytest.param("bugs.sql", f"{NEW}; {REPLACE_C1}", id="bugs-file-rewrites-change"),
    pytest.param("features.sql", f"{NEW}; {SUPERSEDE}", id="features-second-supersede"),
    # ; and -- inside quoted values, then a real second statement
    pytest.param(
        "changes.sql",
        f"INSERT INTO change {COLS} VALUES ('{C2}', '2026-11-12', 'Claude', 'm', '--'); "
        f"{REPLACE_C1}",
        id="quoted-dashes-then-replace",
    ),
    pytest.param(
        "changes.sql",
        f"INSERT INTO change {COLS} VALUES ('{C2}', '2026-11-12', 'Claude', 'm', 'a;b'); "
        f"{UPDATE_C1}",
        id="quoted-semicolon-then-update",
    ),
    pytest.param(
        "changes.sql",
        f"INSERT INTO change {COLS} VALUES ('{C2}', '2026-11-12', 'Claude', 'm', 'it''s'); "
        f"{UPDATE_C1}",
        id="escaped-quote-then-update",
    ),
    pytest.param(
        "changes.sql",
        f"INSERT INTO change {COLS} VALUES ('{C2}', '2026-11-12', 'Claude', 'm', 'x'); "
        f"{UPDATE_C1} --');",
        id="quote-breakout-comment-tail",
    ),
    # odd separators between two statements on one "line"
    pytest.param("changes.sql", f"{NEW};\t{REPLACE_C1}", id="tab-second"),
    pytest.param("changes.sql", f"{NEW};\f{REPLACE_C1}", id="formfeed-second"),
    pytest.param("changes.sql", f"{NEW};\r{REPLACE_C1}", id="bare-cr-second"),
    pytest.param("changes.sql", f"{NEW};\x0b{REPLACE_C1}", id="vt-second"),
    pytest.param("changes.sql", f"{NEW};\x85{REPLACE_C1}", id="nel-second"),
    pytest.param("changes.sql", f"{NEW};\u2028{REPLACE_C1}", id="u2028-second"),
    pytest.param("changes.sql", f"{NEW};\x00{REPLACE_C1}", id="nul-second"),
    pytest.param("changes.sql", f"-- note\r{REPLACE_C1}", id="comment-cr-replace"),
    # statements spread over lines
    pytest.param(
        "changes.sql",
        f"INSERT INTO change {COLS}\nVALUES ('{C2}', '2026-11-12', 'Claude', 'm', 's');",
        id="split-statement",
    ),
    pytest.param(
        "changes.sql",
        f"INSERT INTO change {COLS} VALUES ('{C2}', '2026-11-12', 'Claude', 'm', '\n"
        f"INSERT INTO change (x'); {REPLACE_C1}",
        id="multiline-string-smuggle",
    ),
    pytest.param(
        "changes.sql",
        f"{NEW}; /*\nINSERT INTO change ( */ {REPLACE_C1}",
        id="multiline-comment-smuggle",
    ),
    pytest.param("changes.sql", f"/* c */ {NEW};", id="block-comment-prefix"),
    # not a plain INSERT INTO <table> ( line
    pytest.param(
        "changes.sql", f"insert into change {COLS} VALUES ('{C2}', 'd', 'a', 'm', 's');", id="lower"
    ),
    pytest.param("changes.sql", f" {NEW};", id="leading-space"),
    pytest.param(
        "changes.sql", NEW.replace("INSERT INTO", "INSERT  INTO") + ";", id="double-space"
    ),
    pytest.param("changes.sql", NEW.replace("INSERT INTO", "INSERT\tINTO") + ";", id="tab-keyword"),
    pytest.param(
        "changes.sql", NEW.replace("INSERT INTO", "INSERT\u00a0INTO") + ";", id="nbsp-keyword"
    ),
    pytest.param("changes.sql", NEW.replace("change (", "change(") + ";", id="no-space-paren"),
    pytest.param("changes.sql", NEW.replace("change (", '"change" (') + ";", id="quoted-table"),
    pytest.param("changes.sql", NEW.replace("change (", "main.change (") + ";", id="schema-table"),
    pytest.param("changes.sql", NEW.replace("change (", "changes (") + ";", id="unknown-table"),
    pytest.param("changes.sql", NEW.replace("INSERT", "INS\u0415RT") + ";", id="cyrillic-keyword"),
    pytest.param(
        "changes.sql", NEW.replace("change (", "\u0441hange (") + ";", id="cyrillic-table"
    ),
    pytest.param("changes.sql", NEW.replace("change (", "change\u200b (") + ";", id="zero-width"),
    pytest.param("changes.sql", f"\ufeff{NEW};", id="bom"),
    pytest.param("changes.sql", NEW, id="no-semicolon"),
    pytest.param("changes.sql", f"{NEW}\uff1b", id="fullwidth-semicolon"),
    pytest.param(
        "changes.sql",
        f"INSERT INTO change VALUES ('{C2}', 'd', 'a', 'm', 's', NULL, NULL, NULL, NULL, NULL);",
        id="no-column-list",
    ),
]


@pytest.mark.parametrize("entry", ["load_text", "load"])
@pytest.mark.parametrize(("file", "attack"), R1_ATTACKS)
def test_r1_non_plain_insert_lines_fail_to_load(
    tmp_path: Path, entry: str, file: str, attack: str
) -> None:
    """R1: a row-file line that is not exactly one plain INSERT raises LedgerFormatError."""
    rows = _rows(file, attack)
    with pytest.raises(ledger.LedgerFormatError):
        if entry == "load_text":
            ledger.load_text(SCHEMA, rows)
        else:
            (tmp_path / "schema.sql").write_text(SCHEMA)
            for name, text in rows.items():
                (tmp_path / name).write_bytes(text.encode())
            ledger.load(tmp_path)


def test_r1_duplicate_id_never_rewrites(tmp_path: Path) -> None:
    """R1: a plain insert reusing an earlier id fails instead of changing that row."""
    rows = _rows("changes.sql", change_line(C1, summary="REWRITTEN"))
    with pytest.raises((ledger.LedgerFormatError, sqlite3.IntegrityError)):
        ledger.load_text(SCHEMA, rows)


R1_LEGIT = [
    pytest.param("a; b", id="semicolon-in-value"),
    pytest.param("a -- b", id="dashes-in-value"),
    pytest.param("x'); UPDATE change SET summary = 'X'; --", id="sql-in-value"),
    pytest.param("/* not a comment */", id="block-comment-in-value"),
]


@pytest.mark.parametrize("summary", R1_LEGIT)
def test_r1_quoted_sql_characters_are_data(summary: str) -> None:
    """R1: ';', '--' or SQL inside a quoted value is one plain insert and stays data."""
    db = ledger.load_text(SCHEMA, _rows("changes.sql", change_line(C2, summary=summary)))
    got = dict(db.execute("SELECT id, summary FROM change").fetchall())
    assert got == {C1: "s", C2: summary}


@pytest.mark.parametrize(
    "comment",
    [
        pytest.param(f"-- {REPLACE_C1}", id="comment-replace"),
        pytest.param(f"-- {UPDATE_C1}", id="comment-update"),
        pytest.param("-- /*", id="comment-open-block"),
    ],
)
def test_r1_comment_lines_hide_nothing(comment: str) -> None:
    """R1: SQL inside a '--' comment line is never executed and never swallows a row."""
    rows = _rows("changes.sql", comment + "\n" + change_line(C2))
    db = ledger.load_text(SCHEMA, rows)
    assert dict(db.execute("SELECT id, summary FROM change").fetchall()) == {C1: "s", C2: "s"}


# --------------------------------------------------------------------------------------
# R2: change rows in a real (temporary) git repo
# --------------------------------------------------------------------------------------
MSG = "subject\n\nRequirement: r\nImplemented: i\nWhy: w\n\nChange-Author: Claude\n"
L1, L2, L3, L4 = (change_line(c) for c in (C1, C2, C3, C4))


@pytest.fixture()
def ledger_repo(tmp_path: Path) -> Path:
    r = tmp_path / "r"
    (r / "ledgers").mkdir(parents=True)
    shutil.copy(ROOT / "ledgers" / "schema.sql", r / "ledgers" / "schema.sql")
    (r / "ledgers" / "features.sql").write_text("")
    (r / "ledgers" / "bugs.sql").write_text("")
    (r / "ledgers" / "changes.sql").write_text(L1 + "\n" + L2 + "\n")
    (r / "x.txt").write_text("x\n")
    _init_repo(r)
    return r


def _guard(repo: Path, *args: str) -> tuple[int, str]:
    """Run check_ledgers in-process inside ``repo``; a crash counts as a failed guard."""
    out = io.StringIO()
    with _inside(repo), contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            code = check_ledgers.main(list(args))
        except Exception as exc:  # a crash fails the hook process too
            return 1, f"{type(exc).__name__}: {exc}"
    return code, out.getvalue()


def _bug_then(extra: str) -> str:
    return "INSERT INTO bug (id, description, status) VALUES ('B-t-001', 'd', 'open'); " + extra


# (changes.sql lines, extra file edits {path: text})
R2_SCENARIOS: list[object] = [
    pytest.param([change_line(C1, summary="EDITED"), L2, L3], {}, id="edit-earlier"),
    pytest.param([change_line(C1, approval="owner ok"), L2, L3], {}, id="add-approval-later"),
    pytest.param(
        [
            "INSERT INTO change (id, date, module, author, summary, features, bugs, checks, "
            f"review, approval) VALUES ('{C1}', '2026-11-12', 'Claude', 'm', 's', NULL, NULL, "
            "NULL, NULL, NULL);",
            L2,
            L3,
        ],
        {},
        id="swap-columns",
    ),
    pytest.param([L2, L3], {}, id="delete-earlier"),
    pytest.param([L2, change_line(C1, summary="EDITED"), L3], {}, id="delete-readd-edited"),
    pytest.param(["-- " + L1, L2, L3], {}, id="comment-out-earlier"),
    pytest.param([L1, L2, L3, L4], {}, id="two-new"),
    pytest.param([L1, L2], {"x.txt": "y\n"}, id="zero-new"),
    pytest.param([change_line(C1, summary="EDITED"), L2], {}, id="edit-no-new"),
    pytest.param([L1, L2, f"{L3} {UPDATE_C1}"], {}, id="trailing-update"),
    pytest.param(
        [L1, L2, f"{L3} INSERT OR REPLACE INTO change {COLS} {REWRITE};"], {}, id="trailing-replace"
    ),
    pytest.param([L1, L2, f"{L3} DELETE FROM change WHERE id = '{C1}';"], {}, id="trailing-delete"),
    pytest.param(
        [L1, L2, L3], {"ledgers/bugs.sql": _bug_then(REPLACE_C1) + "\n"}, id="bugs-file-replace"
    ),
    pytest.param(
        [L1, L2, L3], {"ledgers/bugs.sql": _bug_then(UPDATE_C1) + "\n"}, id="bugs-file-update"
    ),
]


def _apply(repo: Path, lines: list[str], edits: dict[str, str]) -> None:
    (repo / "ledgers" / "changes.sql").write_text("\n".join(lines) + "\n")
    for path, text in edits.items():
        (repo / path).write_text(text)
    _git(repo, "add", "-A")


def test_r2_control_one_appended_row_passes(ledger_repo: Path) -> None:
    """R2: control - appending exactly one change row passes pre-commit and ci."""
    _apply(ledger_repo, [L1, L2, L3], {})
    assert _guard(ledger_repo, "--stage", "pre-commit", "--no-tests") == (0, "")
    _git(ledger_repo, "commit", "-q", "-m", MSG)
    assert _guard(ledger_repo, "--stage", "ci", "--no-tests")[0] == 0


@pytest.mark.parametrize("stage", ["pre-commit", "ci"])
@pytest.mark.parametrize(("lines", "edits"), R2_SCENARIOS)
def test_r2_change_history_is_fixed(
    ledger_repo: Path, stage: str, lines: list[str], edits: dict[str, str]
) -> None:
    """R2: rewriting, deleting or miscounting change rows fails pre-commit and ci."""
    _apply(ledger_repo, lines, edits)
    if stage == "ci":
        _git(ledger_repo, "commit", "-q", "-m", MSG)
    code, out = _guard(ledger_repo, "--stage", stage, "--no-tests")
    assert code != 0, out


# --------------------------------------------------------------------------------------
# R3: protected check files
# --------------------------------------------------------------------------------------
PROTECTED = [
    line.strip()
    for line in (ROOT / "tools" / "ci" / "protected.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]
PROTECTED_TOUCHES = sorted(
    {
        p
        for e in PROTECTED
        for p in ((e + "new_file.py", e.rstrip("/")) if e.endswith("/") else (e,))
    }
)
APPROVED = "CHECK-CHANGE: probe\nOWNER-APPROVED-CHECK-CHANGE: owner said yes, 2026-11-12\n"


def test_r3_control_approved_message_passes() -> None:
    """R3: control - both non-empty lines approve; unprotected paths need nothing."""
    assert check_ledgers.check_change_problems(["docs/MERGE-CHECKLIST.md"], APPROVED) == []
    assert check_ledgers.check_change_problems(["src/x.py"], "") == []


@pytest.mark.parametrize("path", PROTECTED_TOUCHES)
def test_r3_every_protected_path_needs_approval(path: str) -> None:
    """R3: a change to any protected path (or a protected directory entry) is refused."""
    assert check_ledgers.check_change_problems([path], MSG) != []


BAD_APPROVALS = [
    pytest.param("", id="none"),
    pytest.param("CHECK-CHANGE: probe\n", id="only-reason"),
    pytest.param("OWNER-APPROVED-CHECK-CHANGE: owner, 2026-11-12\n", id="only-owner"),
    pytest.param(
        "CHECK-CHANGE:\nOWNER-APPROVED-CHECK-CHANGE: owner, 2026-11-12\n", id="empty-reason"
    ),
    pytest.param(
        "CHECK-CHANGE: \t \nOWNER-APPROVED-CHECK-CHANGE: owner, 2026-11-12\n", id="blank-reason"
    ),
    pytest.param(
        "CHECK-CHANGE:\n\nOWNER-APPROVED-CHECK-CHANGE: owner, 2026-11-12\n", id="reason-next-blank"
    ),
    pytest.param("CHECK-CHANGE: probe\nOWNER-APPROVED-CHECK-CHANGE:\n", id="empty-owner"),
    pytest.param(
        "CHECK-CHANGE: probe\nOWNER-APPROVED-CHECK-CHANGE:\nChange-Author: Claude\n",
        id="empty-owner-then-line",
    ),
    pytest.param(
        "CHECK-CHANGE: probe\nOWNER-APPROVED-CHECK-CHANGE: \t \nChange-Author: Claude\n",
        id="blank-owner-then-line",
    ),
    pytest.param(
        "CHECK-CHANGE: probe\nOWNER-APPROVED-CHECK-CHANGE:\u00a0\u00a0\n", id="nbsp-owner"
    ),
    pytest.param(
        "CHECK-CHANGE: probe\nOWNER-APPROVED-CHECK-CHANGE:\u3000\n", id="ideographic-owner"
    ),
    pytest.param("CHECK-CHANGE: probe\r\nOWNER-APPROVED-CHECK-CHANGE: \r\n", id="crlf-owner"),
    pytest.param(
        "CHECK-CHANGE:\u00a0\nOWNER-APPROVED-CHECK-CHANGE: owner, 2026-11-12\n", id="nbsp-reason"
    ),
    pytest.param(
        "CHECK-CHANGE: probe OWNER-APPROVED-CHECK-CHANGE: owner, 2026-11-12\n", id="one-line"
    ),
    pytest.param(
        "CHECK-CHANGE: probe\nNote: OWNER-APPROVED-CHECK-CHANGE: owner, 2026-11-12\n",
        id="owner-mid-line",
    ),
    pytest.param(
        "Note: CHECK-CHANGE: probe\nOWNER-APPROVED-CHECK-CHANGE: owner, 2026-11-12\n",
        id="reason-mid-line",
    ),
]


@pytest.mark.parametrize("approval", BAD_APPROVALS)
def test_r3_missing_or_blank_approval_lines_are_refused(approval: str) -> None:
    """R3: a protected change without two non-empty approval lines is refused."""
    message = MSG + approval
    assert check_ledgers.check_change_problems(["docs/MERGE-CHECKLIST.md"], message) != []


@pytest.fixture()
def gate_repo(tmp_path: Path) -> Path:
    r = tmp_path / "g"
    for d in ("ledgers", "tools/ci", ".githooks", "docs", "notes", "scripts"):
        (r / d).mkdir(parents=True)
    shutil.copy(ROOT / "ledgers" / "schema.sql", r / "ledgers" / "schema.sql")
    shutil.copy(ROOT / "tools" / "ci" / "protected.txt", r / "tools" / "ci" / "protected.txt")
    (r / "tools" / "ci" / "check.py").write_text("print('gate')\n")
    (r / "ledgers" / "features.sql").write_text("")
    (r / "ledgers" / "bugs.sql").write_text("")
    (r / "ledgers" / "changes.sql").write_text(L1 + "\n")
    checklist = "".join(f"- checklist item {i}\n" for i in range(20))
    (r / "docs" / "MERGE-CHECKLIST.md").write_text(checklist)
    (r / "notes" / "draft.md").write_text(checklist + "- one more\n")
    hook = "#!/bin/sh\n" + "".join(f"echo step {i}\n" for i in range(20))
    (r / ".githooks" / "pre-commit").write_text(hook)
    (r / "scripts" / "hook.sh").write_text(hook.replace("step", "stage"))
    (r / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    _init_repo(r)
    return r


def _mutate_symlink(r: Path) -> None:
    (r / "docs" / "MERGE-CHECKLIST.md").unlink()
    (r / "docs" / "MERGE-CHECKLIST.md").symlink_to("../notes/draft.md")
    _git(r, "add", "-A")


def _mutate_dir_to_file(r: Path) -> None:
    _git(r, "rm", "-q", "-r", ".githooks")
    (r / ".githooks").write_text("not a directory\n")
    _git(r, "add", "-A")


def _mutate_new(path: str) -> Callable[[Path], None]:
    def run(r: Path) -> None:
        (r / path).write_text("print('new check')\n")
        _git(r, "add", "-A")

    return run


def _mutate_git(*args: str) -> Callable[[Path], None]:
    def run(r: Path) -> None:
        _git(r, *args)

    return run


def _mutate_copy_into(r: Path) -> None:
    shutil.copy(r / "scripts" / "hook.sh", r / ".githooks" / "pre-push")
    _git(r, "add", "-A")


def _mutate_modify(r: Path) -> None:
    with (r / "docs" / "MERGE-CHECKLIST.md").open("a") as f:
        f.write("- skip the review\n")
    _git(r, "add", "-A")


R3_TOUCHES = [
    pytest.param(_mutate_modify, id="modify"),
    pytest.param(_mutate_git("rm", "-q", "docs/MERGE-CHECKLIST.md"), id="delete"),
    pytest.param(_mutate_git("mv", "docs/MERGE-CHECKLIST.md", "docs/ARCHIVE.md"), id="rename-away"),
    pytest.param(_mutate_git("mv", "pyproject.toml", "pyproject.old"), id="rename-away-pyproject"),
    pytest.param(
        _mutate_git("mv", ".githooks/pre-commit", "scripts/pre-commit"), id="rename-out-of-dir"
    ),
    pytest.param(
        _mutate_git("mv", "tools/ci/check.py", "tools/check.py"), id="rename-out-of-tools-ci"
    ),
    pytest.param(
        _mutate_git("mv", "scripts/hook.sh", ".githooks/post-merge"), id="rename-into-dir"
    ),
    pytest.param(
        _mutate_git("mv", "-f", "notes/draft.md", "docs/MERGE-CHECKLIST.md"), id="rename-over-file"
    ),
    pytest.param(_mutate_copy_into, id="copy-into-dir"),
    pytest.param(
        _mutate_git("update-index", "--chmod=+x", "docs/MERGE-CHECKLIST.md"), id="mode-change"
    ),
    pytest.param(_mutate_symlink, id="typechange-symlink"),
    pytest.param(_mutate_dir_to_file, id="dir-replaced-by-file"),
    pytest.param(_mutate_new("tools/ci/\u00e9t\u00e9.py"), id="non-ascii-name-in-dir"),
    pytest.param(_mutate_new("tools/ci/a b.py"), id="space-name-in-dir"),
    pytest.param(_mutate_new("tools/ci/tab\tname.py"), id="tab-name-in-dir"),
    pytest.param(_mutate_new('tools/ci/q"uote.py'), id="quote-name-in-dir"),
]


def _commit_msg(repo: Path, tmp_path: Path, message: str) -> tuple[int, str]:
    msg = tmp_path / "COMMIT_MSG"
    msg.write_text(message)
    return _guard(repo, "--stage", "commit-msg", "--no-tests", str(msg))


@pytest.mark.parametrize("mutate", R3_TOUCHES)
def test_r3_commit_msg_stage_sees_every_touch(
    gate_repo: Path, tmp_path: Path, mutate: Callable[[Path], None]
) -> None:
    """R3: commit-msg refuses any touch of a protected path without approval lines."""
    mutate(gate_repo)
    (gate_repo / "ledgers" / "changes.sql").write_text(L1 + "\n" + L2 + "\n")
    _git(gate_repo, "add", "ledgers/changes.sql")
    ok, out_ok = _commit_msg(gate_repo, tmp_path, MSG + APPROVED)
    assert ok == 0, f"control (approved message) failed: {out_ok}"
    code, out = _commit_msg(gate_repo, tmp_path, MSG)
    assert code != 0 and "CHECK-CHANGE" in out, out


# --------------------------------------------------------------------------------------
# R4: review gate
# --------------------------------------------------------------------------------------
SHA = "abc1234"
GOOD = """# Review of abc1234

## Standards
No blocking finding.

## Expected vs implemented
| Spec requirement | Implemented where | Status |
|---|---|---|
| schema as data | config/graph_schema.yaml | done |
| scope filter | - | deferred (module ingest-units writes it) |

## Knowledge drift
None: the docs match the code.

## Verdict
MERGE: yes
"""
STANDARDS = "No blocking finding.\n"
ROW = "| scope filter | - | deferred (module ingest-units writes it) |\n"
DRIFT = "## Knowledge drift\nNone: the docs match the code.\n"


def _review(tmp_path: Path, text: str) -> list[str]:
    d = tmp_path / "reviews"
    d.mkdir(exist_ok=True)
    (d / f"{SHA}.md").write_text(text)
    return review_gate.problems(d, SHA)


def test_r4_control_good_review_passes(tmp_path: Path) -> None:
    """R4: control - a complete review with MERGE: yes passes."""
    assert _review(tmp_path, GOOD) == []


BLOCKING_FORMS = [
    pytest.param("BLOCKING: the rename bypass", id="line-start"),
    pytest.param("BLOCKING 1 (c4bda9c, NOTE 1 not closed): rename bypass", id="numbered-label"),
    pytest.param("BLOCKING - rename bypass", id="dash-separator"),
    pytest.param("BLOCKING (rename): bypass", id="paren-label"),
    pytest.param("- BLOCKING: rename bypass", id="dash-bullet"),
    pytest.param("* BLOCKING: rename bypass", id="star-bullet"),
    pytest.param("+ BLOCKING: rename bypass", id="plus-bullet"),
    pytest.param("- [ ] BLOCKING: rename bypass", id="task-item"),
    pytest.param("1. BLOCKING: rename bypass", id="numbered-dot"),
    pytest.param("2) BLOCKING: rename bypass", id="numbered-paren"),
    pytest.param("> BLOCKING: rename bypass", id="quote"),
    pytest.param("> > BLOCKING: rename bypass", id="nested-quote"),
    pytest.param("> - BLOCKING: rename bypass", id="quote-bullet"),
    pytest.param("> 1. BLOCKING: rename bypass", id="quote-numbered"),
    pytest.param("**BLOCKING**: rename bypass", id="bold"),
    pytest.param("**BLOCKING:** rename bypass", id="bold-colon-inside"),
    pytest.param("__BLOCKING__: rename bypass", id="bold-underscore"),
    pytest.param("*BLOCKING*: rename bypass", id="italic"),
    pytest.param("_BLOCKING_: rename bypass", id="italic-underscore"),
    pytest.param("***BLOCKING***: rename bypass", id="bold-italic"),
    pytest.param("- **BLOCKING 2** (c4bda9c): guards bypass the loader", id="bullet-bold-numbered"),
    pytest.param("1. **BLOCKING** - rename bypass", id="numbered-bold"),
    pytest.param("[BLOCKING] rename bypass", id="bracket-tag"),
    pytest.param("- [BLOCKING] rename bypass", id="bullet-bracket-tag"),
    pytest.param("`BLOCKING`: rename bypass", id="code-span"),
    pytest.param("- \u274c BLOCKING: rename bypass", id="emoji-bullet"),
    pytest.param("- note\n  - BLOCKING: rename bypass", id="nested-list"),
    pytest.param("- note\n    * BLOCKING: rename bypass", id="nested-list-4"),
    pytest.param("- note\n\t- BLOCKING: rename bypass", id="nested-list-tab"),
    pytest.param("### BLOCKING: rename bypass", id="heading"),
    pytest.param("### Finding 3 (BLOCKING)", id="heading-paren"),
    pytest.param("#### BLOCKING", id="heading-bare"),
    pytest.param("BLOCKING: rename bypass\n---", id="setext-heading"),
    pytest.param(
        "| # | Finding | Severity | Where |\n|---|---|---|---|\n"
        "| 1 | rename bypass | BLOCKING | x |",
        id="table-cell",
    ),
    pytest.param(
        "| Severity | Finding |\n|---|---|\n| **BLOCKING** | rename bypass |", id="table-cell-bold"
    ),
    pytest.param("<details><summary>BLOCKING: rename bypass</summary></details>", id="html"),
]


@pytest.mark.parametrize("finding", BLOCKING_FORMS)
def test_r4_blocking_finding_in_any_markdown_form_refuses(tmp_path: Path, finding: str) -> None:
    """R4: a finding marked BLOCKING in any markdown form refuses the merge."""
    probs = _review(tmp_path, GOOD.replace(STANDARDS, finding + "\n"))
    assert any("blocking" in p.lower() for p in probs), probs


PROSE = [
    pytest.param("No BLOCKING finding.", id="no-blocking"),
    pytest.param("There is no BLOCKING finding left.", id="there-is-no"),
    pytest.param("All earlier BLOCKING findings are closed.", id="earlier-closed"),
    pytest.param("- No BLOCKING finding.", id="bullet-no-blocking"),
    pytest.param("Nothing here is BLOCKING.", id="nothing-is"),
    pytest.param("The previous review's BLOCKING 1 is fixed in this commit.", id="previous-fixed"),
    pytest.param("NON-BLOCKING: rename the helper.", id="non-blocking"),
    pytest.param("- Non-BLOCKING: rename the helper.", id="bullet-non-blocking"),
]


@pytest.mark.parametrize("prose", PROSE)
def test_r4_prose_mentioning_blocking_does_not_refuse(tmp_path: Path, prose: str) -> None:
    """R4: prose that only mentions BLOCKING does not refuse the merge."""
    assert _review(tmp_path, GOOD.replace(STANDARDS, prose + "\n")) == []


MISSING_ROWS = [
    "| scope filter | - | missing |",
    "| scope filter | - | **missing** |",
    "| scope filter | - | MISSING |",
    "| scope filter | - | Missing |",
    "| scope filter | - | missing (not built yet) |",
    "| scope filter | - | missing",
    "scope filter | - | missing |",
    "|scope filter|-|missing|",
    "| scope filter | - |   missing   |  ",
    "| scope filter | - | `missing` |",
    "| scope filter | - | _missing_ |",
    "| scope filter | - | \u274c missing |",
    "| scope filter | `a \\| b` | missing |",
]
DEFERRED_ROWS = [
    "| scope filter | - | deferred |",
    "| scope filter | - | deferred () |",
    "| scope filter | - | deferred ( ) |",
    "| scope filter | - | **deferred** |",
    "| scope filter | - | Deferred |",
    "| scope filter | - | DEFERRED |",
    "| scope filter | - | deferred - later |",
    "| scope filter | - | deferred: later |",
    "| scope filter | - | deferred",
    "|scope filter|-|deferred|",
    "| scope filter | - | `deferred` |",
    "| scope filter | `a \\| b` | deferred |",
]


@pytest.mark.parametrize(
    "row",
    [pytest.param(r, id=f"missing-{i}") for i, r in enumerate(MISSING_ROWS)]
    + [pytest.param(r, id=f"deferred-{i}") for i, r in enumerate(DEFERRED_ROWS)],
)
def test_r4_missing_or_unreasoned_deferred_row_refuses(tmp_path: Path, row: str) -> None:
    """R4: a missing row, or a deferred row without a reason in parentheses, refuses."""
    assert _review(tmp_path, GOOD.replace(ROW, row + "\n")) != []


VERDICTS = [
    pytest.param("## Verdict\nMERGE: yes\n", "## Verdict\n", id="no-verdict"),
    pytest.param("MERGE: yes\n", "MERGE: no\n", id="merge-no"),
    pytest.param("MERGE: yes\n", "MERGE: no - would be MERGE: yes after fixes\n", id="no-then-yes"),
    pytest.param("MERGE: yes\n", "Not MERGE: yes yet.\n", id="mid-line"),
    pytest.param(DRIFT, "", id="no-drift"),
    pytest.param(DRIFT, "Knowledge drift: none.\n", id="drift-plain-line"),
    pytest.param(DRIFT, "See the ## Knowledge drift section.\n", id="drift-mid-line"),
    pytest.param(DRIFT, "```\n## Knowledge drift\n```\n", id="drift-in-fence"),
    pytest.param(DRIFT, "<!-- ## Knowledge drift -->\n", id="drift-html-comment"),
]


@pytest.mark.parametrize(("old", "new"), VERDICTS)
def test_r4_no_merge_yes_or_no_drift_section_refuses(tmp_path: Path, old: str, new: str) -> None:
    """R4: no MERGE: yes line, or no ## Knowledge drift section, refuses the merge."""
    assert _review(tmp_path, GOOD.replace(old, new)) != []


@pytest.fixture()
def review_repo(tmp_path: Path) -> Path:
    r = tmp_path / "rv"
    for d in ("ledgers", "src", "reviews", "scripts"):
        (r / d).mkdir(parents=True)
    (r / "ledgers" / "changes.sql").write_text(L1 + "\n" + L2 + "\n")
    (r / "ledgers" / "features.sql").write_text("")
    (r / "src" / "a.py").write_text("".join(f"x{i} = {i}\n" for i in range(20)))
    (r / "reviews" / "old.md").write_text("# old\n")
    (r / "scripts" / "run.sh").write_text("echo run\n")
    _init_repo(r)
    return r


def _append(r: Path, path: str, text: str) -> None:
    with (r / path).open("a") as f:
        f.write(text)


def _add_review(r: Path, name: str = "reviews/new.md") -> None:
    (r / name).parent.mkdir(parents=True, exist_ok=True)
    (r / name).write_text("# Review\n")


def _record_only(r: Path) -> None:
    _add_review(r)
    _append(r, "ledgers/changes.sql", L3 + "\n")


RECORD_HEADS = [
    pytest.param(_add_review, id="reviews-only"),
    pytest.param(_record_only, id="reviews-plus-appended-row"),
    pytest.param(lambda r: _add_review(r, "reviews/sub/new.md"), id="nested-review"),
]


def _with(*steps: Callable[[Path], None]) -> Callable[[Path], None]:
    def run(r: Path) -> None:
        for step in steps:
            step(r)

    return run


def _edit_c1(r: Path) -> None:
    (r / "ledgers" / "changes.sql").write_text(
        change_line(C1, summary="EDITED") + "\n" + L2 + "\n" + L3 + "\n"
    )


def _git_step(*args: str) -> Callable[[Path], None]:
    def run(r: Path) -> None:
        _git(r, *args)

    return run


def _write_step(path: str, text: str) -> Callable[[Path], None]:
    def run(r: Path) -> None:
        (r / path).parent.mkdir(parents=True, exist_ok=True)
        (r / path).write_text(text)

    return run


def _chmod_step(path: str) -> Callable[[Path], None]:
    """Make a file executable in the working tree, so the later ``git add -A`` commits it."""

    def run(r: Path) -> None:
        (r / path).chmod(0o755)

    return run


OTHER_HEADS = [
    pytest.param(_with(_record_only, _write_step("src/b.py", "y = 1\n")), id="plus-src"),
    pytest.param(_with(_record_only, _write_step("scripts/run.sh", "echo evil\n")), id="plus-edit"),
    pytest.param(_with(_add_review, _edit_c1), id="edit-earlier-row"),
    pytest.param(
        _with(_add_review, _write_step("ledgers/changes.sql", L2 + "\n" + L3 + "\n")),
        id="delete-earlier-row",
    ),
    pytest.param(
        _with(_add_review, _write_step("ledgers/changes.sql", L1 + "\n" + L3 + "\n" + L2 + "\n")),
        id="insert-mid-file",
    ),
    pytest.param(
        _with(_add_review, _write_step("ledgers/changes.sql", L3 + "\n" + L1 + "\n" + L2 + "\n")),
        id="prepend-row",
    ),
    pytest.param(
        _with(_add_review, _write_step("ledgers/features.sql", FEATURE + "\n")), id="features"
    ),
    pytest.param(_with(_add_review, _git_step("rm", "-q", "src/a.py")), id="delete-src"),
    pytest.param(
        _with(_add_review, _git_step("mv", "src/a.py", "reviews/a.md")), id="rename-into-reviews"
    ),
    pytest.param(_git_step("mv", "src/a.py", "reviews/a.py"), id="rename-into-reviews-only"),
    pytest.param(
        _with(_add_review, _chmod_step("scripts/run.sh")),
        id="mode-change",
    ),
    pytest.param(
        _with(_add_review, _chmod_step("ledgers/changes.sql")),
        id="mode-change-ledger",
    ),
    pytest.param(_with(_record_only, _write_step("reviews2/x.md", "x\n")), id="reviews2-dir"),
    pytest.param(
        _with(_record_only, _write_step("docs/reviews/x.md", "x\n")), id="nested-reviews-dir"
    ),
    pytest.param(_with(_record_only, _write_step("reviews.md", "x\n")), id="reviews-md-file"),
]


def _head_after(r: Path, mutate: Callable[[Path], None]) -> tuple[str, str]:
    parent = _git(r, "rev-parse", "HEAD")
    mutate(r)
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "head")
    return parent, _git(r, "rev-parse", "HEAD")


def _is(short: str, full: str) -> bool:
    return len(short) >= 7 and full.startswith(short)


@pytest.mark.parametrize("mutate", RECORD_HEADS)
def test_r4_review_record_head_is_judged_by_its_parent(
    review_repo: Path, mutate: Callable[[Path], None]
) -> None:
    """R4: a head that only adds reviews/ files and appended change rows reviews its parent."""
    parent, _ = _head_after(review_repo, mutate)
    assert _is(review_gate.reviewed_commit(review_repo), parent)


@pytest.mark.parametrize("mutate", OTHER_HEADS)
def test_r4_any_other_head_is_judged_as_itself(
    review_repo: Path, mutate: Callable[[Path], None]
) -> None:
    """R4: a head that changes anything else is the reviewed commit itself."""
    _, head = _head_after(review_repo, mutate)
    assert _is(review_gate.reviewed_commit(review_repo), head)
