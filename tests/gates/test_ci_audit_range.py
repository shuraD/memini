"""The CI audits work on the public repository's first commit and audit the right range.

Requirement: the first public push (77bce1f, a single root commit) failed CI at "Ledger guards":
``cannot read ledgers/schema.sql at 'HEAD~1:'``. The audit range was also wrong: on a push to
main the commit-message loop audited 0 commits (merge-base(main, HEAD) = HEAD), and on a pull
request it audited GitHub's synthetic merge commit. Approvals (2026-11-14):
- root commit against an empty ledger: Tier 1, APPROVED: vp-rnd 2026-11-14;
- commit-message base fix: Tier 1, APPROVED: vp-rnd 2026-11-14;
- review gate only on pull_request, temporary until T-059: Tier 2, owner, relayed by vp-rnd,
  2026-11-14: "Approve".

B-007 stays: a checkout WITHOUT history (shallow) still fails loudly. Only a true root commit in
a full clone is compared with an empty ledger.

Tests-Author: Claude/devops-498d4ec2 (new tests for its own change, owner rule 2026-11-14).
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from tools import ledger
from tools.ci import audit_range, check_ledgers

ROOT = Path(__file__).resolve().parents[2]
GIT = shutil.which("git") or "git"
_CLEAN_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
CI_YML = ROOT / ".github" / "workflows" / "ci.yml"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 — test-only, fixed git args
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


def _new_repo(path: Path) -> Path:
    """An empty repo holding the ledger schema and the guard, with nothing committed."""
    (path / "ledgers").mkdir(parents=True)
    (path / "tools" / "ci").mkdir(parents=True)
    shutil.copy(ROOT / "ledgers" / "schema.sql", path / "ledgers" / "schema.sql")
    for name in ("features.sql", "bugs.sql", "changes.sql"):
        (path / "ledgers" / name).write_text("")
    for f in ("tools/__init__.py", "tools/ledger.py", "tools/ci/__init__.py"):
        shutil.copy(ROOT / f, path / f)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "t@example.invalid")
    _git(path, "config", "user.name", "t")
    return path


def _change_row(repo: Path, summary: str = "s") -> None:
    with _inside(repo), contextlib.redirect_stdout(io.StringIO()):
        ledger.main(["change", "--author", "t", "--module", "m", "--summary", summary])


def _commit(repo: Path, message: str = "c", *, rows: int = 1, touch: str = "x.txt") -> str:
    for n in range(rows):
        _change_row(repo, f"row {n}")
    (repo / touch).write_text(message)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _ci(repo: Path, *extra: str) -> tuple[int, str]:
    out = io.StringIO()
    with _inside(repo), contextlib.redirect_stdout(out):
        code = check_ledgers.main(["--stage", "ci", "--no-tests", *extra])
    return code, out.getvalue()


# --------------------------------------------------------------------------------------------
# check_ledgers --stage ci on a root commit
# --------------------------------------------------------------------------------------------


def test_root_commit_with_exactly_one_change_row_passes(tmp_path: Path) -> None:
    repo = _new_repo(tmp_path / "r")
    _commit(repo, "seed")
    code, out = _ci(repo)
    assert code == 0, out


@pytest.mark.parametrize("rows", [0, 2])
def test_root_commit_still_needs_exactly_one_change_row(tmp_path: Path, rows: int) -> None:
    repo = _new_repo(tmp_path / "r")
    _commit(repo, "seed", rows=rows)
    code, out = _ci(repo)
    assert code == 1, out
    assert "exactly one change row" in out, out


def test_shallow_checkout_still_fails_loudly(tmp_path: Path) -> None:
    """B-007: missing history is never mistaken for a root commit."""
    origin = _new_repo(tmp_path / "origin")
    _commit(origin, "seed")
    _commit(origin, "second", touch="y.txt")
    shallow = tmp_path / "shallow"
    _git(tmp_path, "clone", "-q", "--depth", "1", f"file://{origin}", str(shallow))
    code, out = _ci(shallow)
    assert code == 1, out
    assert "no git history" in out, out


def test_base_root_audits_every_commit_from_the_root(tmp_path: Path) -> None:
    repo = _new_repo(tmp_path / "r")
    _commit(repo, "seed")
    _commit(repo, "no row", rows=0, touch="y.txt")
    _commit(repo, "good", touch="z.txt")
    code, out = _ci(repo, "--base", "ROOT")
    assert code == 1, out
    assert "exactly one change row" in out, out


def test_base_sha_audits_only_the_commits_after_it(tmp_path: Path) -> None:
    repo = _new_repo(tmp_path / "r")
    _commit(repo, "seed")
    _commit(repo, "no row", rows=0, touch="y.txt")
    base = _commit(repo, "good", touch="z.txt")
    _commit(repo, "good again", touch="w.txt")
    code, out = _ci(repo, "--base", base)
    assert code == 0, out


# --------------------------------------------------------------------------------------------
# tools/ci/audit_range.py: which commits CI audits
# --------------------------------------------------------------------------------------------


def _range(repo: Path, **env: str) -> str:
    with _inside(repo):
        return audit_range.audit_base(env)


def test_push_audits_from_the_previous_tip(tmp_path: Path) -> None:
    repo = _new_repo(tmp_path / "r")
    before = _commit(repo, "seed")
    _commit(repo, "next", touch="y.txt")
    assert _range(repo, AUDIT_EVENT="push", AUDIT_BEFORE=before) == before


def test_first_push_of_a_root_commit_audits_from_the_root(tmp_path: Path) -> None:
    repo = _new_repo(tmp_path / "r")
    _commit(repo, "seed")
    zero = "0" * 40
    assert _range(repo, AUDIT_EVENT="push", AUDIT_BEFORE=zero) == "ROOT"


def test_pull_request_audits_from_the_merge_base_with_the_target(tmp_path: Path) -> None:
    repo = _new_repo(tmp_path / "r")
    base = _commit(repo, "seed")
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "work", touch="y.txt")
    got = _range(repo, AUDIT_EVENT="pull_request", AUDIT_BASE_REF="main")
    assert got == base


def test_pull_request_without_the_target_ref_fails_loudly(tmp_path: Path) -> None:
    repo = _new_repo(tmp_path / "r")
    _commit(repo, "seed")
    with pytest.raises(audit_range.AuditRangeError):
        _range(repo, AUDIT_EVENT="pull_request", AUDIT_BASE_REF="origin/missing")


def test_local_run_on_a_branch_audits_from_main(tmp_path: Path) -> None:
    """act and local runs keep the old behaviour: every commit since main."""
    repo = _new_repo(tmp_path / "r")
    base = _commit(repo, "seed")
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "a", touch="y.txt")
    _commit(repo, "b", touch="z.txt")
    assert _range(repo) == base


def test_local_run_on_main_audits_the_last_commit_or_the_root(tmp_path: Path) -> None:
    repo = _new_repo(tmp_path / "r")
    _commit(repo, "seed")
    assert _range(repo) == "ROOT"
    first = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "next", touch="y.txt")
    assert _range(repo) == first


# --------------------------------------------------------------------------------------------
# ci.yml wiring
# --------------------------------------------------------------------------------------------


def test_ci_yml_wires_the_audit_range_into_both_audits() -> None:
    text = CI_YML.read_text(encoding="utf-8")
    assert "tools/ci/audit_range.py" in text
    assert "check_ledgers --stage ci --base" in text
    assert "MEMINI_AUDIT_BASE" in text
    assert "github.event.pull_request.head.sha" in text  # PRs audit the head, not a merge commit


def test_ci_yml_review_gate_runs_on_pull_requests_only_until_t059() -> None:
    """Temporary relaxation (owner, relayed by vp-rnd, 2026-11-14); removed when T-059 lands."""
    text = CI_YML.read_text(encoding="utf-8")
    step = text[text.index("tools.ci.review_gate") - 600 : text.index("tools.ci.review_gate")]
    assert "github.event_name == 'pull_request'" in step
    assert "T-059" in step
