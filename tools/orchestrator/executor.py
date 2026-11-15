"""Where commands run: the executor seam, and the host side that never enters it.

Every agent step and every gate command goes through an :class:`Executor`. Only
:class:`LocalExecutor` exists (a subprocess in the task worktree). A sandboxed executor
(a micro-VM) can replace it later under one rule, kept here already: the hold-out tests,
the orchestrator state, the inbox and any keys never enter the executor. That is why
:func:`run_on_host` is a separate function the executor cannot reach: the hold-out gate
runs on the host, against an export of the slice's commit.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Result:
    """What a finished command left: its exit code and its combined output."""

    code: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        """Return stdout and stderr together."""
        return self.stdout + self.stderr


# git must read the real objects: a refs/replace/* entry can make any commit look like
# another one (the audited tree swapped under the gates)
NO_REPLACE = {"GIT_NO_REPLACE_OBJECTS": "1"}


def clean_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the orchestrator's own environment without git's per-repository variables.

    Only for the orchestrator's git plumbing (B-006); commands that run repository code get
    :func:`allowed_env` instead.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(NO_REPLACE)
    env.update(extra or {})
    return env


def allowed_env(allow: Sequence[str], extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return only the allow-listed variables of this process, plus ``extra``.

    No API key and no ``MEMINI_ORCH_*`` setting reaches a gate, a test or a hold-out run
    unless it is listed; an engine adds the few its own CLI needs for its agent step only.
    """
    env = {k: os.environ[k] for k in allow if k in os.environ}
    env.update(NO_REPLACE)
    env.update(extra or {})
    return env


def _run(
    argv: Sequence[str], cwd: Path, stdin: str | None, env: Mapping[str, str], timeout: int
) -> Result:
    try:
        res = subprocess.run(  # noqa: S603 — argv from the orchestrator config, never a shell
            list(argv),
            cwd=cwd,
            input=stdin if stdin is not None else "",
            capture_output=True,
            text=True,
            env=dict(env),
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Result(124, "", f"timed out after {timeout}s\n")
    except OSError as exc:
        return Result(127, "", f"cannot start {argv[0]}: {exc.strerror}\n")
    return Result(res.returncode, res.stdout, res.stderr)


class Executor(Protocol):
    """Runs one command for a task (agent step or gate)."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdin: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int = 1800,
        pass_env: Sequence[str] = (),
    ) -> Result:
        """Run ``argv`` in ``cwd`` (a task worktree) and return its result.

        The command sees only the executor's allow-listed variables, ``pass_env`` (an
        engine's own needs) and ``env``.
        """
        ...


class LocalExecutor:
    """A subprocess on this machine, in the task worktree, with an allow-listed env."""

    def __init__(self, env_allow: Sequence[str] = ()) -> None:
        """Remember which variables of this process a command may see."""
        self.env_allow = tuple(env_allow)

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdin: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int = 1800,
        pass_env: Sequence[str] = (),
    ) -> Result:
        """Run ``argv`` in ``cwd`` and return its result."""
        return _run(argv, cwd, stdin, allowed_env((*self.env_allow, *pass_env), env), timeout)


def executor_for(name: str, env_allow: Sequence[str] = ()) -> Executor:
    """Return the executor called ``name`` (only ``local`` exists)."""
    if name != "local":
        msg = f"executor {name!r} is not available"
        raise ValueError(msg)
    return LocalExecutor(env_allow)


def run_on_host(
    argv: Sequence[str],
    *,
    cwd: Path,
    env_allow: Sequence[str],
    env: Mapping[str, str] | None = None,
    timeout: int = 1800,
) -> Result:
    """Run a host-only command (the hold-out gate); never routed through an executor."""
    return _run(argv, cwd, None, allowed_env(env_allow, env), timeout)
