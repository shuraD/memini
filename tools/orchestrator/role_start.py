"""A role session starts its own runner when it opens (FLOW-HARD.md D-10).

A SessionStart and UserPromptSubmit hook runs ``python -m tools.orchestrator.role_start`` in a
session started with ``MEMINI_ROLE=<role>`` (or ``MEMINI_SQUAD``, which the squad sessions
already set): dev1..dev4, qa, planner, devops. It starts one detached supervisor for the role,
once (a session that was open before the hook existed starts it on its next prompt): the
supervisor runs the role's queue (``run-queue --role``) again and again and serves the moves
sent to the role, so work that arrives later is picked up without anyone typing a command. A
role that is not in ``[roles]`` (vp-rnd, say) starts nothing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from tools.orchestrator.config import Config, load_config

_POLL_S = 30.0


def _pid_file(cfg: Config, role: str) -> Path:
    return cfg.state_dir / "locks" / f"runner-{role}.pid"


def _alive(pid_file: Path) -> bool:
    try:
        os.kill(int(pid_file.read_text()), 0)
    except (OSError, ValueError):
        return False
    return True


def _python(cfg: Config) -> list[str]:
    env_file = cfg.repo / "deploy" / ".env"  # only passed to uv, never read here
    if env_file.exists():
        return ["uv", "run", "--env-file", str(env_file), "python"]
    return [sys.executable]


def start(cfg: Config, role: str) -> str:
    """Start the role's supervisor unless it runs; return what happened."""
    if role not in cfg.roles:
        return f"{role}: not a runner role"
    pid_file = _pid_file(cfg, role)
    if _alive(pid_file):
        return ""  # already running: say nothing on every prompt
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    log = cfg.state_dir / f"runner-{role}.log"
    with log.open("a", encoding="utf-8") as out:
        proc = subprocess.Popen(  # noqa: S603 — our own module, no shell
            [*_python(cfg), "-m", "tools.orchestrator.role_start", "--supervise", role],
            cwd=cfg.repo,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_file.write_text(str(proc.pid), encoding="utf-8")
    from tools.orchestrator import state  # noqa: PLC0415

    state.ops_log(
        cfg,
        role,
        "runner_started",
        f"started by the hook in the {role} session, not by the session itself",
        pid=proc.pid,
        trigger="hook",
    )
    return (
        f"[runner] A background program for the {role} role was started automatically by a hook "
        f"(pid {proc.pid}, log {log}). It is not this session and you did not start it: it runs "
        f"the {role} queue and the moves sent to {role}. You need to do nothing about it."
    )


def supervise(cfg: Config, role: str, poll_s: float = _POLL_S) -> None:
    """Serve the role's moves and run its queue, each in a fresh process, until stopped.

    Fresh processes pick up code that landed after the supervisor started.
    """
    base = [*_python(cfg), "-m", "tools.orchestrator"]
    while True:
        subprocess.run([*base, "serve-moves", role], cwd=cfg.repo, check=False)  # noqa: S603
        if cfg.roles[role].queue:
            subprocess.run([*base, "run-queue", "--role", role], cwd=cfg.repo, check=False)  # noqa: S603
        time.sleep(poll_s)


def main(argv: Sequence[str] | None = None) -> int:
    """Start the runner of ``$MEMINI_ROLE``; ``--supervise <role>`` is the detached process."""
    args = list(sys.argv[1:] if argv is None else argv)
    cfg = load_config(Path(__file__).resolve().parents[2])
    if args[:1] == ["--supervise"]:
        supervise(cfg, args[1])
        return 0
    role = os.environ.get("MEMINI_ROLE") or os.environ.get("MEMINI_SQUAD", "")
    if role and (message := start(cfg, role)):
        sys.stdout.write(message + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
