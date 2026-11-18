"""Git plumbing the orchestrator runs on the host (never inside an executor)."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from tools.orchestrator.executor import NO_REPLACE, clean_env

_GIT = shutil.which("git") or "git"


class GitError(RuntimeError):
    """A git command failed; the message carries git's own output."""


def git(
    repo: Path,
    *args: str,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    base_env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``git -C repo args``; raise on failure if ``check``.

    The environment is the orchestrator's own without GIT_* variables, or ``base_env``
    (an allow-listed one, for a commit whose hooks run repository code), plus ``env``.
    """
    full = clean_env(env) if base_env is None else {**base_env, **(env or {}), **NO_REPLACE}
    res = subprocess.run(  # noqa: S603 — fixed git binary, our own args, no shell
        [_GIT, "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env=full,
        check=False,
    )
    if check and res.returncode != 0:
        msg = f"git {' '.join(args[:2])} failed: {(res.stderr or res.stdout).strip()[-800:]}"
        raise GitError(msg)
    return res


def out(repo: Path, *args: str, env: Mapping[str, str] | None = None) -> str:
    """Return the stripped stdout of a git command that must succeed."""
    return git(repo, *args, env=env).stdout.strip()


def rev(repo: Path, name: str) -> str:
    """Return the full sha of ``name``."""
    return out(repo, "rev-parse", "--verify", f"{name}^{{commit}}")


def is_ancestor(repo: Path, older: str, newer: str) -> bool:
    """Return True when ``older`` is an ancestor of (or equal to) ``newer``."""
    return git(repo, "merge-base", "--is-ancestor", older, newer, check=False).returncode == 0


def identity_env(name: str, email: str) -> dict[str, str]:
    """Return the env that makes git author and commit as ``name <email>``."""
    return {
        "GIT_AUTHOR_NAME": name,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": name,
        "GIT_COMMITTER_EMAIL": email,
    }


def changed(repo: Path, base: str, head: str) -> list[str]:
    """Return every path changed in ``base..head`` (renames as delete + add)."""
    raw = out(repo, "diff", "--name-only", "--no-renames", "-z", base, head)
    return [p for p in raw.split("\0") if p]


def name_status(repo: Path, base: str, head: str) -> list[tuple[str, str]]:
    """Return ``(status letter, path)`` for every path changed in ``base..head``."""
    raw = out(repo, "diff", "--name-status", "--no-renames", "-z", base, head)
    fields = [f for f in raw.split("\0") if f]
    return list(zip(fields[::2], fields[1::2], strict=True))


def tail(res: subprocess.CompletedProcess[str], limit: int = 800) -> str:
    """Return the end of a finished git command's output."""
    return (res.stderr + res.stdout).strip()[-limit:]


def bad_modes(repo: Path, base: str, head: str) -> list[str]:
    """Return paths in ``base..head`` that become a symlink, submodule or other non-file."""
    raw = git(repo, "diff", "--raw", "--no-renames", "-z", base, head).stdout
    fields = [f for f in raw.split("\0") if f]
    bad = []
    for meta, path in zip(fields[::2], fields[1::2], strict=True):
        new_mode = meta.lstrip(":").split()[1]
        if new_mode not in {"100644", "100755", "000000"}:
            bad.append(path)
    return bad


def snapshot(wt: Path, message: str, ident: Mapping[str, str]) -> str:
    """Record the worktree (tracked + untracked, not ignored) as a commit on its branch.

    This is orchestrator bookkeeping on its own task branch, built with plumbing: it never
    lands as is. The one commit that reaches main is the squash commit, made with
    ``git commit`` so the repository's hooks run on it.
    """
    git(wt, "add", "-A")
    tree = out(wt, "write-tree")
    head = rev(wt, "HEAD")
    if tree == out(wt, "rev-parse", "HEAD^{tree}"):
        return head
    commit = out(wt, "commit-tree", tree, "-p", head, "-m", message, env=ident)
    git(wt, "update-ref", out(wt, "symbolic-ref", "HEAD"), commit, head)
    return commit


def blob_sha256(repo: Path, rev_name: str, path: str) -> str | None:
    """Return the sha256 of ``path`` at ``rev_name``, or None when it does not exist there."""
    res = subprocess.run(  # noqa: S603 — fixed git binary, our own args, no shell
        [_GIT, "-C", str(repo), "show", f"{rev_name}:{path}"],
        capture_output=True,
        env=clean_env(),
        check=False,
    )
    return hashlib.sha256(res.stdout).hexdigest() if res.returncode == 0 else None


def worktree_of(repo: Path, branch: str) -> Path | None:
    """Return the worktree that has ``branch`` checked out, if any."""
    current: Path | None = None
    for line in out(repo, "worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            current = Path(line[len("worktree ") :])
        elif line == f"branch refs/heads/{branch}":
            return current
    return None


def all_refs(repo: Path) -> bytes:
    """Return every ref with its object (``for-each-ref``, refs/replace included)."""
    return out(repo, "for-each-ref", "--format=%(refname) %(objectname)").encode()


def common_dir(repo: Path) -> Path:
    """Return the repository's common git directory (shared by every worktree)."""
    return Path(out(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))


def set_aside(repo: Path, worktrees: Path, prefix: str, task_id: str, tag: str) -> list[str]:
    """Move a task's leftover worktrees and branch aside (renamed ``-<tag>``); return what moved.

    A rejected run keeps its worktree for inspection, and the next run refuses to start while it
    exists. Nothing is deleted: the worktree directories and the branch only get a new name.
    """
    moved = []
    for name in (task_id, f"{task_id}.gate", f"{task_id}.merge", f"{task_id}.spec"):
        old = worktrees / name
        if old.exists():
            git(repo, "worktree", "move", str(old), str(worktrees / f"{name}-{tag}"))
            moved.append(name)
    branch = f"{prefix}{task_id}"
    if git(repo, "rev-parse", "--verify", "--quiet", branch, check=False).stdout:
        git(repo, "branch", "-m", branch, f"{branch}-{tag}")
        moved.append(branch)
    return moved
