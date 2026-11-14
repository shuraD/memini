"""Commit identity gate: no personal name or email in public commit metadata.

The export gate scans file contents. This gate scans the metadata git attaches to every commit:
the author and committer email must be on the public allow list
(``tools/ci/commit-identities.txt``)::

    python -m tools.ci.commit_identity --pending             # pre-commit hook: the next commit
    python -m tools.ci.commit_identity --base SHA|ROOT       # CI: every commit in the range

A refused commit is reported by short SHA and role (author or committer), never by its email,
because CI logs are public. Exit codes: 0 clean, 1 refused, 2 configuration error.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tools.ci import audit_range

_GIT = shutil.which("git") or "git"
DEFAULT_ALLOW = Path(__file__).resolve().parent / "commit-identities.txt"
DEFAULT_NAMES = Path(__file__).resolve().parent / "commit-names.txt"
ROOT_BASE = "ROOT"


class IdentityConfigError(Exception):
    """The gate cannot run as configured (exit 2)."""


def _git(*args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    res = subprocess.run(  # noqa: S603 — fixed git args, no shell
        [_GIT, *args], capture_output=True, text=True, check=False, env=env
    )
    if res.returncode != 0:
        msg = f"git {args[0]} failed"
        raise IdentityConfigError(msg)
    return res.stdout


def load_allow(path: Path) -> frozenset[str]:
    """Read the allowed emails (one per line, ``#`` comments); refuse a missing or empty list."""
    if not path.is_file():
        msg = f"identity allow list not found: {path}"
        raise IdentityConfigError(msg)
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    allowed = frozenset(line.lower() for line in lines if line and not line.startswith("#"))
    if not allowed:
        msg = f"identity allow list is empty: {path}"
        raise IdentityConfigError(msg)
    return allowed


def _load_names(path: Path) -> frozenset[str]:
    """Read the allowed names (one per line, ``#`` comments); refuse a missing or empty list."""
    if not path.is_file():
        msg = f"name list not found: {path}"
        raise IdentityConfigError(msg)
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    names = frozenset(line for line in lines if line and not line.startswith("#"))
    if not names:
        msg = f"name list is empty: {path}"
        raise IdentityConfigError(msg)
    return names


def _name(ident: str) -> str:
    """Return the name in front of a ``Name <email> time zone`` identity line."""
    return ident.partition("<")[0].strip()


def _email(ident: str) -> str:
    """Return the email inside a ``Name <email> time zone`` identity line."""
    return ident.partition("<")[2].partition(">")[0].strip().lower()


def pending_problems(allowed: frozenset[str], names: frozenset[str] | None = None) -> list[str]:
    """Problems with the identity git would use for the next commit (hook mode)."""
    problems = []
    for role, var in (("author", "GIT_AUTHOR_IDENT"), ("committer", "GIT_COMMITTER_IDENT")):
        # keep GIT_AUTHOR_*/GIT_COMMITTER_* (overrides for this commit), drop the rest (B-006)
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("GIT_") or k.startswith(("GIT_AUTHOR_", "GIT_COMMITTER_"))
        }
        res = subprocess.run(  # noqa: S603 — fixed git args, no shell
            [_GIT, "var", var], capture_output=True, text=True, check=False, env=env
        )
        if res.returncode != 0 or _email(res.stdout) not in allowed:
            problems.append(
                f"next commit: {role} email is not on the identity allow list "
                "(set a neutral identity: git config user.name / user.email)"
            )
        if names is not None and (res.returncode != 0 or _name(res.stdout) not in names):
            problems.append(f"next commit: {role} name is not on the name list")
    return problems


def range_problems(
    base: str, allowed: frozenset[str], names: frozenset[str] | None = None
) -> list[str]:
    """Problems for every commit in ``base..HEAD`` (or every commit, for ``ROOT``)."""
    bad_base = audit_range.validate_base(base)
    if bad_base:  # fail closed (review F1): never read a bad base as "nothing to audit"
        raise IdentityConfigError(bad_base)
    revs = "HEAD" if base == ROOT_BASE else f"{base}..HEAD"
    problems = []
    log = _git("log", "--format=%h%x00%an%x00%ae%x00%cn%x00%ce", revs)
    for line in log.splitlines():
        sha, a_name, a_email, c_name, c_email = line.split("\0")
        for role, name, email in (("author", a_name, a_email), ("committer", c_name, c_email)):
            if email.strip().lower() not in allowed:
                problems.append(f"commit {sha}: {role} email is not on the identity allow list")
            if names is not None and name.strip() not in names:
                problems.append(f"commit {sha}: {role} name is not on the name list")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the exit code."""
    parser = argparse.ArgumentParser(prog="commit_identity")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pending", action="store_true", help="check the next commit's identity")
    mode.add_argument("--base", help="check base..HEAD (a SHA, or ROOT for every commit)")
    parser.add_argument("--allow", type=Path, default=DEFAULT_ALLOW, help="identity allow list")
    parser.add_argument("--names", type=Path, help="also require names from this list")
    args = parser.parse_args(argv)
    try:
        allowed = load_allow(args.allow)
        names = _load_names(args.names) if args.names else None
        problems = (
            pending_problems(allowed, names)
            if args.pending
            else range_problems(args.base, allowed, names)
        )
    except IdentityConfigError as exc:
        print(f"commit identity: ERROR: {exc}", file=sys.stderr)
        return 2
    for problem in problems:
        print(f"commit identity: {problem}")
    if problems:
        print("commit identity: FAILED; the refused emails are never shown")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
