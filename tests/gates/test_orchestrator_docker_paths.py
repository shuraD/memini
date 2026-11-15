"""The docker executor runs only coder code, with container paths only (pilot fix 2026-11-15).

The LP-01 pilot failed at setup: the run's environment pointed uv at the Mac's cache
(``/Users/<owner>/.cache/uv``) inside a read-only container. A review then found the rest of
the gap: agent steps and setup were sent into the container too, and the container got host
interpreters and host paths. Owner, 2026-11-15: gates must be quick; the microVM comes later.

- The container never sees a host path: HOME, TMPDIR and the uv cache are container paths,
  host-only cache variables are dropped, the host interpreter becomes the image's ``python``,
  a path under the working directory becomes its ``/slice`` path, and a host file outside it
  (the run's private pytest gate) is shipped inside the slice tar.
- Only gates marked ``sandbox = true`` run in the container; the static and git gates (no
  coder code runs in them) stay on the host. Agent steps and setup always run on the host.

Tests-Author: Claude/vp-rnd (new tests for its own change).
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from tools.orchestrator import config, gates
from tools.orchestrator.dockerexec import DockerExecutor
from tools.orchestrator.executor import Result

ROOT = Path(__file__).resolve().parents[2]
FAKE = """#!/usr/bin/env python3
import sys, pathlib
base = pathlib.Path(sys.argv[0])
if sys.argv[1] == "image":
    sys.exit(0)                         # the image exists: no build
if sys.argv[1] == "run":
    base.with_suffix(".argv").write_text("\\0".join(sys.argv[1:]))
    base.with_suffix(".tar").write_bytes(sys.stdin.buffer.read())
sys.exit(0)
"""


def _executor(tmp_path: Path, env_allow: Sequence[str]) -> tuple[DockerExecutor, Path]:
    cli = tmp_path / "bin" / "docker"
    cli.parent.mkdir(parents=True)
    cli.write_text(FAKE)
    cli.chmod(0o755)
    (tmp_path / "uv.lock").write_text("x = 1\n")
    (tmp_path / "Dockerfile").write_text("FROM x\n")
    ex = DockerExecutor(
        cli=str(cli),
        image="img",
        dockerfile=tmp_path / "Dockerfile",
        lock=tmp_path / "uv.lock",
        env_allow=env_allow,
    )
    return ex, cli


def _container_env(argv: list[str]) -> dict[str, str]:
    pairs = [argv[i + 1] for i, a in enumerate(argv) if a == "--env"]
    return dict(p.split("=", 1) for p in pairs)


def test_the_container_gets_container_paths_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host_home = tmp_path / "Users" / "owner"
    monkeypatch.setenv("XDG_CACHE_HOME", str(host_home / ".cache"))
    monkeypatch.setenv("UV_PYTHON_INSTALL_DIR", str(host_home / ".local" / "share" / "uv"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "var" / "T"))
    allow = ["XDG_CACHE_HOME", "UV_PYTHON_INSTALL_DIR", "TMPDIR", "UV_CACHE_DIR", "HOME"]
    ex, cli = _executor(tmp_path, allow)
    work = tmp_path / "slice"
    work.mkdir()
    run_env = {
        "HOME": str(tmp_path / "private" / "home"),
        "UV_CACHE_DIR": str(host_home / ".cache" / "uv"),
        "MEMINI_AUDIT_BASE": "abc",
    }
    assert ex.run(["pytest", "-q"], cwd=work, env=run_env).code == 0
    argv = cli.with_suffix(".argv").read_text().split("\0")
    env = _container_env(argv)
    assert env["HOME"] == "/tmp"  # noqa: S108 — container paths
    assert env["TMPDIR"] == "/tmp"  # noqa: S108 — container paths
    assert env["UV_CACHE_DIR"] == "/tmp/uv-cache"  # noqa: S108 — container paths
    assert "XDG_CACHE_HOME" not in env
    assert "UV_PYTHON_INSTALL_DIR" not in env
    assert env["PYTHONPATH"] == "src:."  # the slice's own code, no installed project
    assert env["UV_NO_SYNC"] == "1"  # uv run never installs into the read-only image venv
    assert env["UV_OFFLINE"] == "1"
    assert env["MEMINI_AUDIT_BASE"] == "abc"  # ordinary values pass unchanged
    assert str(tmp_path) not in " ".join(argv), "a host path reached the container"


def test_host_interpreter_and_paths_become_container_ones(tmp_path: Path) -> None:
    ex, cli = _executor(tmp_path, [])
    work = tmp_path / "wt"
    (work / "tests").mkdir(parents=True)
    (work / "tests" / "test_x.py").write_text("def test_x():\n    pass\n")
    private = tmp_path / "private"
    private.mkdir()
    gate = private / "pytest_gate.py"
    gate.write_text("# the run's private gate copy\n")
    host_python = work / ".venv" / "bin" / "python"
    argv = [str(host_python), "-P", str(gate), "-q", str(work / "tests" / "test_x.py"), "tests"]
    assert ex.run(argv, cwd=work).code == 0
    sent = cli.with_suffix(".argv").read_text().split("\0")
    command = sent[-1]
    assert "python -P /slice/.sandbox/" in command
    assert "/slice/tests/test_x.py" in command
    assert str(tmp_path) not in command
    with tarfile.open(fileobj=io.BytesIO(cli.with_suffix(".tar").read_bytes())) as tar:
        names = tar.getnames()
        shipped = [n for n in names if n.startswith(".sandbox/")]
        assert len(shipped) == 1, names
        member = tar.extractfile(shipped[0])
        assert member is not None
        assert member.read() == gate.read_bytes()


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

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
        self.calls.append(list(argv))
        return Result(0, "", "")


def test_only_sandbox_gates_go_to_the_sandbox(tmp_path: Path) -> None:
    over: dict[str, Any] = {
        "state_dir": str(tmp_path / "state"),
        "profiles": {"defaults": {"env": {}}},
        "gates": [
            {"name": "static", "run": ["echo", "static"], "stages": ["slice"]},
            {"name": "tests", "run": ["echo", "tests"], "stages": ["slice"], "sandbox": True},
        ],
    }
    cfg = config.load_config(ROOT, overrides=over, env={})
    sandbox, host = _Recorder(), _Recorder()
    run = gates.GateRun(
        worktree=tmp_path,
        log=tmp_path / "log",
        test_ids=frozenset(),
        env={},
        gate_script=tmp_path / "g.py",
    )
    assert gates.run_gates(cfg, sandbox, run, "slice", "base", host=host) == []
    assert sandbox.calls == [["echo", "tests"]]
    assert host.calls == [["echo", "static"]]


def test_the_shipped_config_sandboxes_only_the_fast_test_gate() -> None:
    cfg = config.load_config(ROOT, overrides={"state_dir": "/tmp/x-state"}, env={})  # noqa: S108
    sandboxed = [g for g in cfg.gates if g.sandbox]
    assert [g.name for g in sandboxed] == ["tests"]
    args = " ".join(sandboxed[0].run)
    assert "tests/sanity" in args and "tests/unit" in args
    assert "tests/gates" not in args  # the tooling suite runs in CI, not per slice
