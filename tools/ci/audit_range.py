"""Which commits the CI audits run over: print the audit base, a commit SHA or ``ROOT``.

The ledger guard and the commit-message audit check every commit in ``base..HEAD``.
``ROOT`` means every commit from the repository's first one. The base depends on the event:

- ``pull_request``: the merge base of HEAD (the PR head, checked out by ci.yml) and the
  target branch ``AUDIT_BASE_REF`` (e.g. ``origin/main``). If that ref is missing, it fails
  loudly: never guess.
- ``push``: ``AUDIT_BEFORE``, the branch tip before the push, when git has it. For a new
  branch (all-zero SHA) the base is the merge base with ``AUDIT_BASE_REF``, or ``ROOT`` when
  HEAD is the repository's first commit.
- anything else (local ``act`` runs, ``workflow_dispatch``): the merge base with ``main``, as
  before. On ``main`` itself that's the last commit, or ``ROOT`` for a first commit.

Runs as ``python -P tools/ci/audit_range.py`` (standard library only, so no module in the
repository root can shadow an import). The inputs come from the environment: ``AUDIT_EVENT``,
``AUDIT_BEFORE`` and ``AUDIT_BASE_REF``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Mapping

_GIT = shutil.which("git") or "git"
ROOT = "ROOT"


class AuditRangeError(RuntimeError):
    """The audit range cannot be determined; CI must stop rather than audit nothing."""


def _git(*args: str) -> str | None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    res = subprocess.run(  # noqa: S603 — fixed git args, no shell
        [_GIT, *args], capture_output=True, text=True, check=False, env=env
    )
    return res.stdout.strip() if res.returncode == 0 else None


def _is_root(rev: str = "HEAD") -> bool:
    """Return True only for a real first commit in a full clone (a shallow cut is not a root)."""
    if _git("rev-parse", "--is-shallow-repository") != "false":
        return False
    parents = _git("rev-list", "--parents", "-n", "1", rev)
    return parents is not None and len(parents.split()) == 1


def _merge_base(ref: str) -> str | None:
    if _git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}") is None:
        return None
    return _git("merge-base", ref, "HEAD")


def _last_commit_or_root() -> str:
    if _is_root():
        return ROOT
    parent = _git("rev-parse", "HEAD~1")
    if parent is None:
        msg = "HEAD has no readable parent and is not a root commit (shallow checkout?)"
        raise AuditRangeError(msg)
    return parent


def _push_base(before: str, base_ref: str) -> str:
    """Audit base for a push: the previous tip, else the root, else the target's merge base."""
    if before and set(before) != {"0"} and _git("cat-file", "-e", f"{before}^{{commit}}") == "":
        return before
    if _is_root():
        return ROOT
    base = _merge_base(base_ref)
    if base and base != _git("rev-parse", "HEAD"):
        return base
    return _last_commit_or_root()


def audit_base(env: Mapping[str, str]) -> str:
    """Return the audit base for the event described by ``env``."""
    event = env.get("AUDIT_EVENT", "")
    base_ref = env.get("AUDIT_BASE_REF", "") or "main"
    if event == "pull_request":
        base = _merge_base(base_ref)
        if not base:
            msg = f"pull request: target ref {base_ref!r} is not available (fetch-depth: 0?)"
            raise AuditRangeError(msg)
        return base
    if event == "push":
        return _push_base(env.get("AUDIT_BEFORE", "").strip(), base_ref)
    base = _merge_base("main")
    if base and base != _git("rev-parse", "HEAD"):
        return base
    return _last_commit_or_root()


def validate_base(base: str) -> str | None:
    """Return why ``base`` is not a usable audit base, or None when it is.

    Fail closed (review of fix/ci-first-commit, F1): an empty, unknown or unrelated base must
    stop the audits, never be read as "nothing to audit".
    """
    if base == ROOT:
        return None
    if not base.strip():
        return "audit base is empty (did the Audit range step run?)"
    if _git("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}") is None:
        return "audit base does not name a commit in this checkout"
    if _git("merge-base", "--is-ancestor", base, "HEAD") is None:
        return "audit base is not an ancestor of HEAD"
    return None


def main(argv: list[str] | None = None) -> int:
    """Print the audit base for the current event, or with ``--check BASE`` validate one."""
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["--check"]:
        problem = validate_base(args[1] if len(args) > 1 else "")
        if problem:
            print(f"audit range: ERROR: {problem}", file=sys.stderr)
            return 2
        return 0
    try:
        base = audit_base(os.environ)
    except AuditRangeError as exc:
        print(f"audit range: ERROR: {exc}", file=sys.stderr)
        return 2
    problem = validate_base(base)
    if problem:
        print(f"audit range: ERROR: {problem}", file=sys.stderr)
        return 2
    print(base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
