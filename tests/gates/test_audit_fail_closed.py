"""Review of fix/ci-first-commit (devops-infra, 2026-11-14): the audits fail closed.

F1: an empty or unresolvable audit base audited ZERO commits and passed, silently switching off
the ledger guard, the identity gate and the commit-message loop. Now a base must be ``ROOT`` or
a commit that is an ancestor of HEAD; anything else fails loudly.
F2: ci.yml checked out ``github.sha`` explicitly, which broke act's local checkout. The ref is
now the PR head only (empty off pull requests, so the default checkout returns).
F3: the identity gate checked emails only. With ``--names FILE`` (passed by the hook and CI) the
author and committer NAMES must be on the public name list ``tools/ci/commit-names.txt`` too.

Tests-Author: Claude/devops-498d4ec2 (new tests for its own change, owner rule 2026-11-14).
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from tools import ledger
from tools.ci import audit_range, check_ledgers, commit_identity

ROOT = Path(__file__).resolve().parents[2]
GIT = shutil.which("git") or "git"
NEUTRAL_NAME = "memini-author"
NEUTRAL = "memini-author@example.invalid"
NAMES = ROOT / "tools" / "ci" / "commit-names.txt"
_CLEAN_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(  # noqa: S603 — test-only, fixed git args
        [GIT, *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        env={**_CLEAN_ENV, **(env or {})},
    ).stdout.strip()


@contextlib.contextmanager
def _inside(directory: Path):  # type: ignore[no-untyped-def]
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def _repo(tmp_path: Path, name: str = NEUTRAL_NAME) -> Path:
    repo = tmp_path / "r"
    (repo / "ledgers").mkdir(parents=True)
    shutil.copy(ROOT / "ledgers" / "schema.sql", repo / "ledgers" / "schema.sql")
    for n in ("features.sql", "bugs.sql", "changes.sql"):
        (repo / "ledgers" / n).write_text("")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", name)
    _git(repo, "config", "user.email", NEUTRAL)
    return repo


def _commit(repo: Path, name: str, env: dict[str, str] | None = None) -> str:
    with _inside(repo), contextlib.redirect_stdout(io.StringIO()):
        ledger.main(["change", "--author", "t", "--module", "m", "--summary", name])
    (repo / name).write_text(name)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", name, env=env)
    return _git(repo, "rev-parse", "HEAD")


def _run(repo: Path, fn, argv: list[str]) -> tuple[int, str]:  # type: ignore[no-untyped-def]
    out = io.StringIO()
    with _inside(repo), contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = fn(argv)
    return code, out.getvalue()


BAD_BASES = ["", "nonexistent123", "0" * 40]


# --------------------------------------------------------------------------------------------
# F1: a bad base is refused, never read as "nothing to audit"
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("base", BAD_BASES)
def test_validate_base_refuses_an_empty_or_unknown_base(tmp_path: Path, base: str) -> None:
    repo = _repo(tmp_path)
    _commit(repo, "a")
    with _inside(repo):
        assert audit_range.validate_base(base) is not None


def test_validate_base_refuses_a_base_that_is_not_an_ancestor(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit(repo, "a")
    _git(repo, "checkout", "-q", "-b", "side")
    side = _commit(repo, "side")
    _git(repo, "checkout", "-q", "main")
    _commit(repo, "b")
    with _inside(repo):
        assert audit_range.validate_base(side) is not None


def test_validate_base_accepts_root_and_an_ancestor(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    first = _commit(repo, "a")
    _commit(repo, "b")
    with _inside(repo):
        assert audit_range.validate_base("ROOT") is None
        assert audit_range.validate_base(first) is None


@pytest.mark.parametrize("base", BAD_BASES)
def test_ledger_guard_fails_loudly_on_a_bad_base(tmp_path: Path, base: str) -> None:
    repo = _repo(tmp_path)
    _commit(repo, "a")
    _commit(repo, "b")
    code, out = _run(repo, check_ledgers.main, ["--stage", "ci", "--no-tests", "--base", base])
    assert code == 1, out
    assert "audit base" in out, out


@pytest.mark.parametrize("base", BAD_BASES)
def test_identity_gate_fails_loudly_on_a_bad_base(tmp_path: Path, base: str) -> None:
    repo = _repo(tmp_path)
    _commit(repo, "a")
    code, out = _run(repo, commit_identity.main, ["--base", base])
    assert code == 2, out
    assert "audit base" in out, out


def test_ci_yml_commit_message_loop_refuses_an_empty_base() -> None:
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    loop = text[text.index('"Commit messages') : text.index("tools.ci.review_gate")]
    assert "tools/ci/audit_range.py --check" in loop


# --------------------------------------------------------------------------------------------
# F2: act's local checkout is not overridden off pull requests
# --------------------------------------------------------------------------------------------


def test_checkout_has_no_ref_input_and_a_separate_step_selects_the_pr_head() -> None:
    """Any ``ref:`` input (even an empty one) makes actions/checkout fetch from github.com,
    also under act, which breaks local runs of unpushed commits. The PR head is selected by a
    separate step instead, only on pull requests."""
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    checkout = text[text.index("actions/checkout") : text.index("Select the pull request head")]
    live = [ln for ln in checkout.splitlines() if not ln.lstrip().startswith("#")]
    assert not any(ln.strip().startswith("ref:") for ln in live), live
    step = text[text.index("Select the pull request head") :][:400]
    assert "github.event_name == 'pull_request'" in step
    assert 'git checkout --detach "$PR_HEAD_SHA"' in step


# --------------------------------------------------------------------------------------------
# F3: names, not only emails
# --------------------------------------------------------------------------------------------


def test_personal_name_with_the_neutral_email_is_refused_in_history(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit(repo, "a")
    _commit(repo, "b", env={"GIT_AUTHOR_NAME": "Some Personal Name"})
    code, out = _run(repo, commit_identity.main, ["--names", str(NAMES), "--base", "ROOT"])
    assert code == 1, out
    assert "author name" in out, out
    assert "Some Personal Name" not in out


def test_personal_committer_name_is_refused_in_history(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit(repo, "a", env={"GIT_COMMITTER_NAME": "Some Personal Name"})
    code, out = _run(repo, commit_identity.main, ["--names", str(NAMES), "--base", "ROOT"])
    assert code == 1, out
    assert "committer name" in out, out


def test_personal_name_is_refused_before_the_commit(tmp_path: Path) -> None:
    repo = _repo(tmp_path, name="Some Personal Name")
    code, out = _run(repo, commit_identity.main, ["--names", str(NAMES), "--pending"])
    assert code == 1, out
    assert "name" in out and "Some Personal Name" not in out, out


def test_neutral_name_and_email_pass(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit(repo, "a")
    assert _run(repo, commit_identity.main, ["--names", str(NAMES), "--base", "ROOT"])[0] == 0
    assert _run(repo, commit_identity.main, ["--names", str(NAMES), "--pending"])[0] == 0


def test_hook_and_ci_check_names_too() -> None:
    hook = (ROOT / ".githooks" / "pre-commit").read_text(encoding="utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "commit_identity --pending --names tools/ci/commit-names.txt" in hook
    assert 'commit_identity --base "$MEMINI_AUDIT_BASE" --names tools/ci/commit-names.txt' in ci


def test_the_public_name_list_holds_only_the_neutral_name() -> None:
    lines = (ROOT / "tools" / "ci" / "commit-names.txt").read_text(encoding="utf-8").splitlines()
    entries = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    assert entries == [NEUTRAL_NAME], entries
