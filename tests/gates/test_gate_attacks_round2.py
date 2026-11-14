# ruff: noqa: S608 - the attack inputs are SQL text by design
"""Adversarial tests of the gates, round 2: rules R6-R8 (owner-approved 2026-11-12).

Written from the rules and ``reviews/d969d76.md`` (BLOCKING 1-3), not from the code.

R6 CI checks the ledger on every commit of the branch, not only the last; R7 config files
that steer the gates are protected like the gates; R8 every row ``ledger.insert`` writes
loads again through ``ledger.load_text`` (line breaks become single spaces).
Every git subprocess runs with all ``GIT_*`` variables stripped (B-006).
"""

import contextlib
import io
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from tools import ledger
from tools.ci import check_ledgers

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


def _guard(repo: Path, *args: str) -> tuple[int, str]:
    """Run check_ledgers in-process inside ``repo``; a crash counts as a failed guard."""
    out = io.StringIO()
    with _inside(repo), contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            code = check_ledgers.main(list(args))
        except Exception as exc:  # a crash fails the CI process too
            return 1, f"{type(exc).__name__}: {exc}"
    return code, out.getvalue()


# --------------------------------------------------------------------------------------
# Ledger rows
# --------------------------------------------------------------------------------------
DAY = "20261112"
C1 = f"C-{DAY}-0001"
MSG = "subject\n\nRequirement: r\nImplemented: i\nWhy: w\n\nChange-Author: Claude\n"
REF = "owner said remove it, 2026-11-12"
APPROVED_MSG = (
    "subject\n\nRequirement: r\nImplemented: i\nWhy: w\n"
    f"OWNER-APPROVED-REMOVAL: {REF}\n\nChange-Author: Claude\n"
)


def change_line(cid: str, **over: object) -> str:
    cols: dict[str, object] = {
        "id": cid,
        "date": "2026-11-12",
        "author": "Claude",
        "module": "m",
        "summary": "s",
        "features": None,
        "bugs": None,
        "checks": None,
        "review": None,
        "approval": None,
    }
    cols.update(over)
    return ledger.insert("change", **cols)


def feature_line(fid: str, status: str = "active") -> str:
    return ledger.insert(
        "feature", id=fid, description="d", layer="infra", proof="p", status=status, added_in=C1
    )


BUG = ledger.insert("bug", id="B-t-001", description="d", status="open")
F1, F2 = "F-t-001", "F-t-002"
L1 = change_line(C1)

# --------------------------------------------------------------------------------------
# R6: every commit of the branch is audited in CI
# --------------------------------------------------------------------------------------
CHANGES, BUGS, FEATURES, NOTE = (
    "ledgers/changes.sql",
    "ledgers/bugs.sql",
    "ledgers/features.sql",
    "x.txt",
)
State = dict[str, list[str]]
Step = Callable[[State], str]  # mutates the state, returns the commit message


def _base_state() -> State:
    return {
        CHANGES: [L1],
        BUGS: [BUG],
        FEATURES: [feature_line(F1), feature_line(F2)],
        NOTE: ["x"],
    }


def _next_change(state: State, **over: object) -> str:
    numbers = [int(n) for line in state[CHANGES] for n in re.findall(rf"C-{DAY}-(\d{{4}})", line)]
    return change_line(f"C-{DAY}-{max(numbers, default=0) + 1:04d}", **over)


def _append_change(state: State, **over: object) -> None:
    state[CHANGES].append(_next_change(state, **over))


def clean(state: State) -> str:
    _append_change(state)
    return MSG


def rewrite_earlier(state: State) -> str:
    state[CHANGES][0] = change_line(C1, summary="REWRITTEN MID-BRANCH")
    _append_change(state)
    return MSG


def add_approval_to_earlier(state: State) -> str:
    state[CHANGES][0] = change_line(C1, approval="owner ok")
    _append_change(state)
    return MSG


def restore_earlier(state: State) -> str:
    state[CHANGES][0] = L1
    _append_change(state)
    return MSG


def delete_earlier(state: State) -> str:
    _append_change(state)
    state[CHANGES].pop(0)
    return MSG


def comment_out_earlier(state: State) -> str:
    _append_change(state)
    state[CHANGES][0] = "-- " + state[CHANGES][0]
    return MSG


def delete_bug(state: State) -> str:
    state[BUGS] = []
    _append_change(state)
    return MSG


def readd_bug(state: State) -> str:
    state[BUGS] = [BUG]
    _append_change(state)
    return MSG


def zero_rows(state: State) -> str:
    state[NOTE].append("y")
    return MSG


def two_rows(state: State) -> str:
    _append_change(state)
    _append_change(state)
    return MSG


def remove_feature(state: State) -> str:
    state[FEATURES] = [feature_line(F2)]
    _append_change(state)
    return MSG


def supersede_feature(state: State) -> str:
    state[FEATURES] = [feature_line(F1, "superseded"), feature_line(F2)]
    _append_change(state)
    return MSG


def restore_feature(state: State) -> str:
    state[FEATURES] = [feature_line(F1), feature_line(F2)]
    _append_change(state)
    return MSG


def remove_feature_approved(state: State) -> str:
    state[FEATURES] = [feature_line(F2)]
    _append_change(state, approval=REF)
    return APPROVED_MSG


def supersede_feature_approved(state: State) -> str:
    state[FEATURES] = [feature_line(F1, "superseded"), feature_line(F2)]
    _append_change(state, approval=REF)
    return APPROVED_MSG


def clean_with_approval_line(state: State) -> str:
    """A clean commit whose message carries an approval line meant for another commit."""
    _append_change(state, approval=REF)
    return APPROVED_MSG


BAD_STEPS: list[tuple[str, Step]] = [
    ("rewrite-earlier", rewrite_earlier),
    ("add-approval-to-earlier", add_approval_to_earlier),
    ("delete-earlier", delete_earlier),
    ("comment-out-earlier", comment_out_earlier),
    ("delete-bug", delete_bug),
    ("zero-rows", zero_rows),
    ("two-rows", two_rows),
    ("remove-feature", remove_feature),
    ("supersede-feature", supersede_feature),
]
POSITIONS: dict[str, Callable[[Step], list[Step]]] = {
    "first": lambda bad: [bad, clean, clean],
    "middle": lambda bad: [clean, bad, clean],
    "last": lambda bad: [clean, clean, bad],
}
BAD_BRANCHES = [
    pytest.param(make(bad), id=f"{name}-{where}")
    for name, bad in BAD_STEPS
    for where, make in POSITIONS.items()
]
# Branches whose HEAD ledger looks fine against the merge base, but one commit broke a rule.
NET_CLEAN_BRANCHES = [
    pytest.param([zero_rows, two_rows], id="zero-then-two"),
    pytest.param([two_rows, zero_rows], id="two-then-zero"),
    pytest.param([rewrite_earlier, restore_earlier], id="rewrite-then-restore"),
    pytest.param([delete_bug, readd_bug], id="delete-bug-then-readd"),
    pytest.param([remove_feature, restore_feature], id="remove-feature-then-restore"),
    pytest.param([supersede_feature, restore_feature], id="supersede-then-restore"),
]
# The approval line is in a different commit than the removal.
APPROVAL_ELSEWHERE_BRANCHES = [
    pytest.param([remove_feature, clean_with_approval_line], id="remove-then-approval"),
    pytest.param([clean_with_approval_line, remove_feature], id="approval-then-remove"),
    pytest.param([supersede_feature, clean_with_approval_line], id="supersede-then-approval"),
    pytest.param([clean_with_approval_line, supersede_feature], id="approval-then-supersede"),
    pytest.param([clean_with_approval_line, remove_feature, clean], id="approval-remove-clean"),
]
CLEAN_BRANCHES = [
    pytest.param([clean], id="one-clean"),
    pytest.param([clean, clean, clean], id="three-clean"),
    pytest.param([remove_feature_approved, clean], id="approved-removal-first"),
    pytest.param([clean, remove_feature_approved, clean], id="approved-removal-middle"),
    pytest.param([clean, remove_feature_approved], id="approved-removal-last"),
    pytest.param([supersede_feature_approved, clean], id="approved-supersede-first"),
    pytest.param([clean, readd_bug], id="adding-a-bug-row"),
]


def _write_state(repo: Path, state: State) -> None:
    for path, lines in state.items():
        (repo / path).write_text("".join(line + "\n" for line in lines))


@pytest.fixture()
def branch_repo(tmp_path: Path) -> Path:
    """A repo whose ``main`` holds a valid ledger, checked out on a fresh branch."""
    r = tmp_path / "r"
    (r / "ledgers").mkdir(parents=True)
    shutil.copy(ROOT / "ledgers" / "schema.sql", r / "ledgers" / "schema.sql")
    _write_state(r, _base_state())
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "user.name", "t")
    _git(r, "config", "core.autocrlf", "false")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "--no-verify", "-m", MSG)
    _git(r, "checkout", "-q", "-b", "feature")
    return r


def _commit_branch(repo: Path, steps: list[Step]) -> None:
    """Commit each step as one commit, bypassing hooks (the case CI exists to catch)."""
    state = _base_state()
    for step in steps:
        message = step(state)
        _write_state(repo, state)
        _git(repo, "add", "-A")
        # --no-verify on the throwaway repo simulates an agent that skipped the hooks
        _git(repo, "commit", "-q", "--no-verify", "--allow-empty", "-m", message)


def _ci(repo: Path) -> tuple[int, str]:
    return _guard(repo, "--stage", "ci", "--no-tests")


@pytest.mark.parametrize("steps", CLEAN_BRANCHES)
def test_r6_control_clean_branch_passes(branch_repo: Path, steps: list[Step]) -> None:
    """R6: control - a branch where every commit is clean passes the ci stage."""
    _commit_branch(branch_repo, steps)
    code, out = _ci(branch_repo)
    assert code == 0, out


@pytest.mark.parametrize("steps", BAD_BRANCHES)
def test_r6_one_bad_commit_anywhere_fails_ci(branch_repo: Path, steps: list[Step]) -> None:
    """R6: a ledger rule broken in any branch commit fails ci, even if later ones are clean."""
    _commit_branch(branch_repo, steps)
    code, out = _ci(branch_repo)
    assert code != 0, out


@pytest.mark.parametrize("steps", NET_CLEAN_BRANCHES)
def test_r6_a_later_commit_does_not_launder_an_earlier_one(
    branch_repo: Path, steps: list[Step]
) -> None:
    """R6: a bad commit fails ci even when a later commit makes HEAD look clean."""
    _commit_branch(branch_repo, steps)
    code, out = _ci(branch_repo)
    assert code != 0, out


@pytest.mark.parametrize("steps", APPROVAL_ELSEWHERE_BRANCHES)
def test_r6_removal_approval_counts_only_in_its_own_commit(
    branch_repo: Path, steps: list[Step]
) -> None:
    """R6: OWNER-APPROVED-REMOVAL in a different commit does not approve a feature removal."""
    _commit_branch(branch_repo, steps)
    code, out = _ci(branch_repo)
    assert code != 0, out


# --------------------------------------------------------------------------------------
# R7: config files that steer the gates are protected
# --------------------------------------------------------------------------------------
APPROVED = "CHECK-CHANGE: probe\nOWNER-APPROVED-CHECK-CHANGE: owner said yes, 2026-11-12\n"
R7_PATHS = [
    "ruff.toml",
    ".ruff.toml",
    "mypy.ini",
    ".mypy.ini",
    ".coveragerc",
    "pytest.ini",
    ".importlinter",
    "setup.cfg",
    "tox.ini",
    "conftest.py",
    "tests/conftest.py",
    "tests/unit/conftest.py",
    "tests/acceptance/conftest.py",
    "src/memini/conftest.py",
    "tests/unit/deep/nested/conftest.py",
    "tools/__init__.py",
]


@pytest.mark.parametrize("path", R7_PATHS)
def test_r7_gate_config_needs_both_approval_lines(path: str) -> None:
    """R7: touching a gate config file without CHECK-CHANGE and the owner line is refused."""
    assert check_ledgers.check_change_problems([path], MSG) != []
    assert check_ledgers.check_change_problems([path], MSG + "CHECK-CHANGE: probe\n") != []
    owner_only = MSG + "OWNER-APPROVED-CHECK-CHANGE: owner said yes, 2026-11-12\n"
    assert check_ledgers.check_change_problems([path], owner_only) != []


@pytest.mark.parametrize("path", R7_PATHS)
def test_r7_gate_config_is_refused_among_other_files(path: str) -> None:
    """R7: a gate config file hidden among unprotected files is still refused."""
    assert check_ledgers.check_change_problems(["src/x.py", path, "docs/y.md"], MSG) != []


@pytest.mark.parametrize("path", R7_PATHS)
def test_r7_control_both_lines_approve_a_gate_config(path: str) -> None:
    """R7: control - with both lines a gate config change is accepted like any protected path."""
    assert check_ledgers.check_change_problems([path], MSG + APPROVED) == []


# --------------------------------------------------------------------------------------
# R8: insert -> load_text round trip
# --------------------------------------------------------------------------------------
LINE_BREAKS = [chr(c) for c in (0x0A, 0x0D, 0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029)]
TEXT_COLUMNS = list(ledger.CHANGE_COLUMNS)


def _single_spaced(value: str) -> str:
    return "".join(" " if ch in LINE_BREAKS else ch for ch in value)


def _round_trip(column: str, value: str) -> object:
    line = change_line(C1, **{column: value})
    db = ledger.load_text(SCHEMA, {"features.sql": "", "bugs.sql": "", "changes.sql": line + "\n"})
    return db.execute(f"SELECT {column} FROM change WHERE id = ?", (C1,)).fetchone()[0]


def test_r8_line_break_list_is_what_splitlines_splits_on() -> None:
    """R8: the attack list covers every single character str.splitlines() splits on."""
    splitters = {chr(c) for c in range(0x110000) if len(f"a{chr(c)}b".splitlines()) > 1}
    assert splitters == set(LINE_BREAKS)


@pytest.mark.parametrize("column", TEXT_COLUMNS)
@pytest.mark.parametrize(
    "value",
    [pytest.param(f"a{ch}b", id=f"U+{ord(ch):04X}") for ch in LINE_BREAKS]
    + [
        pytest.param("pasted\r\nfrom windows", id="crlf"),
        pytest.param("\r\n\r\n", id="only-crlf"),
        pytest.param("x\n\r" + chr(0x2028) + chr(0x2029) + "\x85y", id="mixed"),
        pytest.param("trailing\r", id="trailing-cr"),
        pytest.param("\vleading", id="leading-vt"),
    ],
)
def test_r8_line_breaks_load_as_single_spaces(column: str, value: str) -> None:
    """R8: each line-break character is written as one space and the row loads again."""
    line = change_line(C1, **{column: value})
    assert len(line.splitlines()) == 1, repr(line)
    assert _round_trip(column, value) == _single_spaced(value)


@pytest.mark.parametrize("column", TEXT_COLUMNS)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param("a\tb\t", id="tabs"),
        pytest.param("it's \"quoted\" ''twice''", id="quotes"),
        pytest.param("a; b;", id="semicolons"),
        pytest.param("a -- not a comment", id="dashes"),
        pytest.param("x'); DROP TABLE change; --", id="injection"),
        pytest.param("'); UPDATE change SET summary = 'x' WHERE id = 'C-1'; --", id="update"),
        pytest.param("REPLACE INTO change VALUES ('x');", id="replace"),
        pytest.param("INSERT INTO change (id) VALUES ('x');", id="insert"),
        pytest.param("/* comment */ SELECT 1", id="block-comment"),
        pytest.param("Grüße, 日本語, Ελληνικά, עברית", id="unicode"),
        pytest.param("\U0001f600" + chr(0x200D) + "\U0001f4bb", id="emoji-zwj"),
        pytest.param("e" + chr(0x301) + " " + chr(0xA0) + " " + chr(0x3000), id="combining-nbsp"),
        pytest.param("\\n is not a newline", id="backslash"),
        pytest.param("", id="empty"),
    ],
)
def test_r8_other_characters_round_trip_exactly(column: str, value: str) -> None:
    """R8: every character that is not a line break is read back exactly as written."""
    assert _round_trip(column, value) == value


@pytest.mark.parametrize("file", ["changes.sql", "bugs.sql", "features.sql", "schema"])
def test_r8_crlf_ledger_file_loads(file: str) -> None:
    """R8: a ledger file with CRLF line endings loads."""
    rows = {
        "changes.sql": change_line(C1) + "\n" + change_line(f"C-{DAY}-0002") + "\n",
        "bugs.sql": BUG
        + "\n"
        + ledger.insert("bug", id="B-t-002", description="d", status="open")
        + "\n",
        "features.sql": feature_line(F1) + "\n" + feature_line(F2) + "\n",
    }
    schema = SCHEMA
    if file == "schema":
        schema = SCHEMA.replace("\n", "\r\n")
    else:
        rows[file] = rows[file].replace("\n", "\r\n")
    db = ledger.load_text(schema, rows)
    counts = [db.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("change", "bug")]
    assert counts == [2, 2]
    assert db.execute("SELECT count(*) FROM feature").fetchone()[0] == 2


def test_r8_add_change_with_line_breaks_keeps_the_ledger_loadable(tmp_path: Path) -> None:
    """R8: add_change with every line break in its values leaves a ledger that loads again."""
    d = tmp_path / "ledgers"
    d.mkdir()
    shutil.copy(ROOT / "ledgers" / "schema.sql", d / "schema.sql")
    for name in ("features.sql", "bugs.sql", "changes.sql"):
        (d / name).write_text("")
    breaks = "".join(LINE_BREAKS)
    fields = {
        "date": "2026-11-12",
        "author": "Claude",
        "module": "m",
        "summary": f"pasted\r\nfrom windows{breaks}end",
    }
    with _inside(tmp_path):
        first = ledger.add_change(d, **fields)
        second = ledger.add_change(d, **fields)
        db = ledger.load(d)
    summaries = db.execute(
        "SELECT summary FROM change WHERE id IN (?, ?)", (first, second)
    ).fetchall()
    assert summaries == [(_single_spaced(fields["summary"]),)] * 2
