"""One supervisor per role: a flock on locks/runner-<role>.run guards its whole life.

Rules guarded: a second supervisor of a role returns at once and starts nothing (no peers, no
subprocess); a free lock lets the supervisor loop and it holds the lock while it loops; the
locks of different roles are independent.
"""

from __future__ import annotations

import fcntl
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from tools.orchestrator import config, role_start

ROOT = Path(__file__).resolve().parents[2]


class _StopError(Exception):
    """Raised by the patched sleep to end the loop after one cycle."""


@pytest.fixture
def rcfg(tmp_path: Path) -> config.Config:
    cfg = config.load_config(ROOT, overrides={"state_dir": str(tmp_path / "state")})
    (cfg.state_dir / "locks").mkdir(parents=True, exist_ok=True)
    return cfg


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    seen: dict[str, list[Any]] = {"start_all": [], "run": [], "sleep": []}

    def start_all(*a: Any, **k: Any) -> list[str]:
        seen["start_all"].append((a, k))
        return []

    def run(argv: Any, **_k: Any) -> None:
        seen["run"].append(argv)

    def sleep(_s: float) -> None:
        seen["sleep"].append(_s)
        raise _StopError

    monkeypatch.setattr(role_start, "start_all", start_all)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(time, "sleep", sleep)
    return seen


def _lock_file(cfg: config.Config, role: str) -> Path:
    return cfg.state_dir / "locks" / f"runner-{role}.run"


def _roles(cfg: config.Config) -> list[str]:
    return sorted(cfg.roles)


@pytest.fixture
def held(rcfg: config.Config) -> Iterator[list[Any]]:
    handles: list[Any] = []
    yield handles
    for h in handles:
        h.close()


def _hold(cfg: config.Config, role: str, held: list[Any]) -> None:
    fh = _lock_file(cfg, role).open("a+", encoding="utf-8")
    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    held.append(fh)


def test_second_supervisor_returns_at_once_and_starts_nothing(
    rcfg: config.Config, calls: dict[str, list[Any]], held: list[Any]
) -> None:
    """Rule: lock already held -> supervise returns, no start_all, no subprocess, no sleep."""
    role = _roles(rcfg)[0]
    _hold(rcfg, role, held)
    role_start.supervise(rcfg, role)
    assert calls == {"start_all": [], "run": [], "sleep": []}


def test_free_lock_enters_loop_and_holds_the_lock_while_looping(
    rcfg: config.Config, calls: dict[str, list[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: a free lock lets the supervisor run its cycle, and it holds the lock meanwhile."""
    role = _roles(rcfg)[0]
    probe: dict[str, bool] = {}

    def sleep(_s: float) -> None:
        with _lock_file(rcfg, role).open("a+", encoding="utf-8") as other:
            try:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                probe["blocked"] = True
            else:
                probe["blocked"] = False
        raise _StopError

    monkeypatch.setattr(time, "sleep", sleep)
    with pytest.raises(_StopError):
        role_start.supervise(rcfg, role)
    assert len(calls["start_all"]) == 1
    assert calls["run"], "the loop ran its subprocesses"
    assert probe == {"blocked": True}


def test_locks_of_different_roles_are_independent(
    rcfg: config.Config, calls: dict[str, list[Any]], held: list[Any]
) -> None:
    """Rule: a held lock of role A does not stop the supervisor of role B."""
    roles = _roles(rcfg)
    assert len(roles) >= 2
    _hold(rcfg, roles[0], held)
    with pytest.raises(_StopError):
        role_start.supervise(rcfg, roles[1])
    assert len(calls["start_all"]) == 1


def test_lock_is_per_role_file(rcfg: config.Config, calls: dict[str, list[Any]]) -> None:
    """Rule: the lock file is runner-<role>.run under locks/, one per role."""
    role = _roles(rcfg)[0]
    with pytest.raises(_StopError):
        role_start.supervise(rcfg, role)
    assert _lock_file(rcfg, role).exists()
