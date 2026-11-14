"""Commit metadata never carries a personal identity into the public repository.

Requirement: vp-rnd for the owner, 2026-11-14. The CI fix commit was first created with the
owner's personal name and email, inherited from the global git config, and caught before the
push. The export gate scans file contents, not commit metadata. This gate refuses any commit
whose author or committer email is not on the public allow list ``tools/ci/commit-identities.txt``:
- in the pre-commit hook (``--pending``: the identity git is about to use);
- in CI, for every commit in the audit range (``--base SHA|ROOT``).
The report never prints the refused email, because CI logs are public.

Tests-Author: Claude/devops-498d4ec2 (new tests for its own change, owner rule 2026-11-14).
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
from pathlib import Path

from tools.ci import commit_identity

ROOT = Path(__file__).resolve().parents[2]
ALLOW = ROOT / "tools" / "ci" / "commit-identities.txt"
GIT = shutil.which("git") or "git"
NEUTRAL = "memini-author@example.invalid"
PERSONAL = "a.real.person@example.com"  # stands in for a personal address in these tests
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


def _repo(tmp_path: Path, email: str = NEUTRAL) -> Path:
    repo = tmp_path / "r"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "someone")
    _git(repo, "config", "user.email", email)
    return repo


def _commit(
    repo: Path, name: str, *, author: str | None = None, committer: str | None = None
) -> str:
    (repo / name).write_text(name)
    _git(repo, "add", "-A")
    env = {}
    if author:
        env |= {"GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": author}
    if committer:
        env |= {"GIT_COMMITTER_NAME": "x", "GIT_COMMITTER_EMAIL": committer}
    _git(repo, "commit", "-q", "-m", name, env=env)
    return _git(repo, "rev-parse", "HEAD")


def _run(repo: Path, *args: str) -> tuple[int, str]:
    allow = repo.parent / "allow.txt"
    allow.write_text(f"# test allow list\n{NEUTRAL}\n")
    out = io.StringIO()
    previous = Path.cwd()
    os.chdir(repo)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = commit_identity.main(["--allow", str(allow), *args])
    finally:
        os.chdir(previous)
    return code, out.getvalue()


def test_neutral_history_passes_from_the_root(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit(repo, "a")
    _commit(repo, "b")
    code, out = _run(repo, "--base", "ROOT")
    assert code == 0, out


def test_personal_author_is_refused_without_printing_it(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit(repo, "a")
    bad = _commit(repo, "b", author=PERSONAL)
    code, out = _run(repo, "--base", "ROOT")
    assert code == 1, out
    assert bad[:7] in out and "author" in out, out
    assert PERSONAL not in out, "the gate printed the personal email into a (public) log"


def test_personal_committer_is_refused(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit(repo, "a", committer=PERSONAL)
    code, out = _run(repo, "--base", "ROOT")
    assert code == 1, out
    assert "committer" in out, out
    assert PERSONAL not in out


def test_base_limits_the_audit_to_newer_commits(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit(repo, "old", author=PERSONAL)
    base = _commit(repo, "a")
    _commit(repo, "b")
    code, out = _run(repo, "--base", base)
    assert code == 0, out


def test_pending_identity_is_checked_before_the_commit(tmp_path: Path) -> None:
    """The hook mode: the identity git would use for the next commit."""
    assert _run(_repo(tmp_path / "ok"), "--pending")[0] == 0
    code, out = _run(_repo(tmp_path / "bad", PERSONAL), "--pending")
    assert code == 1, out
    assert PERSONAL not in out


def test_missing_allow_list_fails_loudly(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit(repo, "a")
    out = io.StringIO()
    previous = Path.cwd()
    os.chdir(repo)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = commit_identity.main(["--allow", str(tmp_path / "absent.txt"), "--base", "ROOT"])
    finally:
        os.chdir(previous)
    assert code == 2, out.getvalue()


def test_the_public_allow_list_holds_only_the_neutral_identity() -> None:
    entries = [
        line.strip()
        for line in ALLOW.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert entries == [NEUTRAL], entries


def test_hook_and_ci_run_the_identity_gate() -> None:
    hook = (ROOT / ".githooks" / "pre-commit").read_text(encoding="utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "tools.ci.commit_identity --pending" in hook
    assert 'tools.ci.commit_identity --base "$MEMINI_AUDIT_BASE"' in ci
