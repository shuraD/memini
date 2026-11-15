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


def clean_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the environment without git's per-repository variables (B-006), plus ``extra``."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
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
            env=clean_env(env),
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
    ) -> Result:
        """Run ``argv`` in ``cwd`` (a task worktree) and return its result."""
        ...


class LocalExecutor:
    """A subprocess on this machine, in the task worktree."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdin: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int = 1800,
    ) -> Result:
        """Run ``argv`` in ``cwd`` and return its result."""
        return _run(argv, cwd, stdin, env or {}, timeout)


def executor_for(name: str) -> Executor:
    """Return the executor called ``name`` (only ``local`` exists)."""
    if name != "local":
        msg = f"executor {name!r} is not available"
        raise ValueError(msg)
    return LocalExecutor()


def run_on_host(
    argv: Sequence[str], *, cwd: Path, env: Mapping[str, str] | None = None, timeout: int = 1800
) -> Result:
    """Run a host-only command (the hold-out gate); never routed through an executor."""
    return _run(argv, cwd, None, env or {}, timeout)
