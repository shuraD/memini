"""The ledger guards reject bad commits in a real git repo (temporary; no network)."""

import contextlib
import io
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from tools import ledger
from tools.ci import check_ledgers

ROOT = Path(__file__).resolve().parents[2]


GIT = shutil.which("git") or "git"


# When git runs a hook it exports GIT_DIR / GIT_INDEX_FILE / GIT_WORK_TREE for the
# repo being committed; a test that creates its own temp repo must not inherit
# them, or every git command inside the test hits the outer repo (B-006).
_CLEAN_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — test-only, fixed git args
        [GIT, *args], cwd=repo, capture_output=True, text=True, check=check, env=_CLEAN_ENV
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
    """Run the body with ``directory`` as the working directory."""
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def _check(repo: Path, stage: str, message: str = "", *, tests: bool = False) -> tuple[int, str]:
    """Run the guard in-process inside ``repo`` (so coverage sees it)."""
    msg = repo / "MSG"
    msg.write_text(message)
    args = ["--stage", stage] + ([] if tests else ["--no-tests"])
    if stage == "commit-msg":
        args.append(str(msg))
    out = io.StringIO()
    with _inside(repo), contextlib.redirect_stdout(out):
        code = check_ledgers.main(args)
    return code, out.getvalue()


def _change(repo: Path, cwd: Path | None = None) -> None:
    with _inside(cwd or repo), contextlib.redirect_stdout(io.StringIO()):
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


def test_commit_without_a_change_row_is_rejected(repo: Path) -> None:
    (repo / "x.txt").write_text("x")
    _git(repo, "add", "x.txt")
    code, out = _check(repo, "pre-commit")
    assert code != 0 and "exactly one change row" in out


def test_commit_with_one_change_row_passes(repo: Path) -> None:
    _change(repo)  # appends + stages ledgers/changes.sql
    assert _check(repo, "pre-commit")[0] == 0


def test_removing_a_feature_without_owner_approval_is_rejected(repo: Path) -> None:
    (repo / "ledgers" / "features.sql").write_text(
        ledger.insert(
            "feature",
            id="F-t-001",
            description="d",
            layer="infra",
            proof="p",
            status="active",
            added_in="C-1",
        )
        + "\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add feature")
    (repo / "ledgers" / "features.sql").write_text("")
    _git(repo, "add", "-A")
    code, out = _check(
        repo,
        "commit-msg",
        "drop F-001\n\nRequirement: r\nImplemented: i\nWhy: w\n\nChange-Author: Claude\n",
    )
    assert code != 0 and "F-t-001" in out
    ok, _ = _check(
        repo,
        "commit-msg",
        "drop\n\nRequirement: r\nImplemented: i\nWhy: w\n\n"
        "Change-Author: Claude\nOWNER-APPROVED-REMOVAL: owner x\n",
    )
    assert ok == 0


def test_commit_message_without_change_author_is_rejected(repo: Path) -> None:
    code, out = _check(repo, "commit-msg", "subject only\n")
    assert code != 0 and "Change-Author" in out


def test_ci_stage_passes_when_the_last_commit_recorded_its_change(repo: Path) -> None:
    _change(repo)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "recorded\n\nChange-Author: Claude")
    assert _check(repo, "ci")[0] == 0


def test_pre_commit_rejects_a_feature_whose_test_does_not_exist(repo: Path) -> None:
    (repo / "ledgers" / "features.sql").write_text(
        ledger.insert(
            "feature",
            id="F-t-009",
            description="d",
            layer="infra",
            proof="p",
            status="active",
            added_in="C-1",
        )
        + "\n"
        + ledger.insert(
            "feature_test", feature_id="F-t-009", test_id="tests/unit/test_x.py::test_gone"
        )
        + "\n"
    )
    _change(repo)
    _git(repo, "add", "-A")
    code, out = _check(repo, "pre-commit", tests=True)
    assert code != 0 and "F-t-009 has no existing test" in out


def test_test_collector_returns_node_ids_whatever_the_pytest_addopts() -> None:
    # B-001: the project's addopts "-q" plus the collector's "-q" made pytest print
    # "file: count" lines, so every feature looked test-less and the first commit was blocked
    ids = check_ledgers._existing_tests()
    assert (
        "tests/gates/test_ledger.py::test_insert_is_one_deterministic_line_with_quotes_escaped"
        in ids
    )


def test_ci_stage_compares_the_last_commit_with_its_parent(repo: Path) -> None:
    # review 6a19f19: --stage ci compared HEAD with itself, so a commit made past the
    # hooks (no change row, lost feature) was never caught in CI
    (repo / "x.txt").write_text("x")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "no change row")
    code, out = _check(repo, "ci")
    assert code != 0 and "exactly one change row" in out


def test_ledger_cli_writes_to_the_repo_ledger_from_a_subdirectory(repo: Path) -> None:
    sub = repo / "src"
    sub.mkdir()
    _change(repo, cwd=sub)
    assert "INSERT INTO change" in (repo / "ledgers" / "changes.sql").read_text()


def test_git_helper_does_not_inherit_the_hook_environment(repo: Path, monkeypatch) -> None:
    # B-006: under the pre-commit hook GIT_DIR pointed every test git call at the outer repo
    monkeypatch.setenv("GIT_DIR", "/nonexistent/.git")
    assert _git(repo, "rev-parse", "--show-toplevel").stdout.strip() == str(repo.resolve())


def test_ledger_cli_resolves_the_repo_from_cwd_not_from_git_dir(repo: Path, monkeypatch) -> None:
    # B-006 second site: under the hook, tools.ledger must write THIS repo's ledger
    monkeypatch.setenv("GIT_DIR", str(repo.parent / "other" / ".git"))
    _change(repo)
    assert "INSERT INTO change" in (repo / "ledgers" / "changes.sql").read_text()


# --- review c4bda9c: the guards themselves must refuse a rewritten ledger -----------

_MSG_OK = "m\n\nRequirement: r\nImplemented: i\nWhy: w\n\nChange-Author: Claude\n"


def test_a_staged_update_line_fails_the_pre_commit_guard(repo: Path) -> None:
    _change(repo)
    _git(repo, "commit", "-q", "-m", "first row")
    _change(repo)
    with (repo / "ledgers" / "changes.sql").open("a") as f:
        f.write("UPDATE change SET summary = 'REWRITTEN';\n")
    _git(repo, "add", "-A")
    code, out = _check(repo, "pre-commit")
    assert code != 0 and "changes.sql" in out


def test_an_earlier_change_row_edited_in_place_fails_the_guard(repo: Path) -> None:
    _change(repo)
    _git(repo, "commit", "-q", "-m", "first row")
    path = repo / "ledgers" / "changes.sql"
    path.write_text(path.read_text().replace("'s'", "'REWRITTEN'"))
    _change(repo)
    code, out = _check(repo, "pre-commit")
    assert code != 0 and "rewritten" in out


def test_renaming_a_protected_check_file_needs_both_lines(repo: Path) -> None:
    _git(repo, "mv", "tools/ci/check_ledgers.py", "tools/check_ledgers_moved.py")
    code, out = _check(repo, "commit-msg", _MSG_OK)
    assert code != 0 and "tools/ci/check_ledgers.py" in out
