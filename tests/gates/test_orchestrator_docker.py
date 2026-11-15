"""The docker executor: unit tests with a fake docker CLI, and one opt-in real-docker smoke.

The unit tests assert the sandbox flags on the argv the executor builds (``--network none``,
no host env, no .git, env allow-list) without a daemon. The real-docker test is marked
``docker`` (opt-in) and is the only one that needs a running Docker.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
from tools.orchestrator import dockerexec
from tools.orchestrator.dockerexec import DockerExecutor, docker_run_argv, lock_tag

ROOT = Path(__file__).resolve().parents[2]
FAKE = """#!/usr/bin/env python3
import sys, pathlib
rec = pathlib.Path(sys.argv[0] + ".log")
rec.open("a").write("\\0".join(sys.argv[1:]) + "\\n")
cmd = sys.argv[1]
if cmd == "image":
    sys.exit(1)      # force a build
if cmd in ("build", "info"):
    sys.exit(0)
if cmd == "run":
    sys.stdin.buffer.read()            # consume the slice tar
    sys.stderr.write("ran in sandbox\\n")
    sys.exit(0)
sys.exit(0)
"""


def test_docker_run_argv_has_the_sandbox_flags() -> None:
    argv = docker_run_argv(
        "docker", "img:tag", workdir="/slice", env={"PATH": "/usr/bin"}, memory="2g", pids=512
    )
    joined = " ".join(argv)
    assert argv[:3] == ["docker", "run", "--rm"]
    assert "--network none" in joined
    assert "--read-only" in joined
    assert "--cap-drop ALL" in joined
    assert "--security-opt no-new-privileges" in joined
    assert "--pids-limit 512" in joined
    assert "--memory 2g" in joined
    assert "--user 10001:10001" in joined
    assert "--env-file /dev/null" in joined  # start from an empty environment
    assert "--env PATH=/usr/bin" in joined
    assert argv[-1] == "img:tag"  # the image is the last token before the command


def test_lock_tag_changes_with_the_lock(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile"
    lock = tmp_path / "uv.lock"
    df.write_text("FROM x\n")
    lock.write_text("a = 1\n")
    first = lock_tag(df, lock)
    lock.write_text("a = 2\n")
    assert lock_tag(df, lock) != first
    assert len(first) == 16


def _executor(tmp_path: Path, env_allow: list[str]) -> tuple[DockerExecutor, Path]:
    cli = tmp_path / "bin" / "docker"
    cli.parent.mkdir(parents=True, exist_ok=True)
    cli.write_text(FAKE)
    cli.chmod(0o755)
    (tmp_path / "uv.lock").write_text("x = 1\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\nversion = '0'\n")
    (tmp_path / "Dockerfile").write_text("FROM x\n")
    ex = DockerExecutor(
        cli=str(cli),
        image="img",
        dockerfile=tmp_path / "Dockerfile",
        lock=tmp_path / "uv.lock",
        env_allow=env_allow,
    )
    return ex, Path(str(cli) + ".log")


def test_fake_run_builds_once_and_passes_only_allow_listed_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("PATH", "/usr/bin")
    ex, record = _executor(tmp_path, ["PATH"])
    work = tmp_path / "slice"
    (work / ".git").mkdir(parents=True)
    (work / ".git" / "config").write_text("secret-history\n")
    (work / "src").mkdir()
    (work / "src" / "a.py").write_text("x = 1\n")
    res = ex.run(["pytest", "-q"], cwd=work, env={"MEMINI_AUDIT_BASE": "abc"})
    assert res.code == 0
    calls = [c.split("\0") for c in record.read_text().splitlines()]
    runs = [c for c in calls if c and c[0] == "run"]
    assert len([c for c in calls if c and c[0] == "build"]) == 1  # built once
    assert len(runs) == 1
    argv = runs[0]
    joined = " ".join(argv)
    assert "none" in argv  # --network none
    assert "ANTHROPIC_API_KEY=secret" not in joined  # no host secret
    assert "--env" in argv
    assert "MEMINI_AUDIT_BASE=abc" in argv
    assert "GIT_NO_REPLACE_OBJECTS=1" in joined
    assert "-v" not in argv  # no host mount
    assert "--volume" not in argv
    assert "--mount" not in joined
    assert str(work) not in joined  # the host path is never passed to the container


def test_fake_run_never_sends_the_git_dir_into_the_container(tmp_path: Path) -> None:
    work = tmp_path / "slice"
    (work / ".git").mkdir(parents=True)
    (work / ".git" / "secret").write_text("history\n")
    (work / "keep.py").write_text("x = 1\n")
    payload = dockerexec._slice_tar(work)
    with tarfile.open(fileobj=io.BytesIO(payload)) as tar:
        names = tar.getnames()
    assert any("keep.py" in n for n in names)
    assert not any(".git" in n.split("/") for n in names)


def test_build_failure_is_reported_not_raised(tmp_path: Path) -> None:
    cli = tmp_path / "docker-fail"
    cli.write_text("#!/bin/sh\ncase $1 in image) exit 1;; build) exit 7;; *) exit 0;; esac\n")
    cli.chmod(0o755)
    (tmp_path / "uv.lock").write_text("x\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\nversion = '0'\n")
    (tmp_path / "Dockerfile").write_text("FROM x\n")
    ex = DockerExecutor(
        cli=str(cli),
        image="img",
        dockerfile=tmp_path / "Dockerfile",
        lock=tmp_path / "uv.lock",
        env_allow=[],
    )
    res = ex.run(["pytest"], cwd=tmp_path)
    assert res.code == 125  # the image could not be built
    assert "image unavailable" in res.stderr


@pytest.mark.docker
def test_real_docker_runs_pytest_in_a_sandbox(tmp_path: Path) -> None:
    """Opt-in (``-m docker``): build the real image and run a tiny pytest in a container."""
    if os.environ.get("MEMINI_DOCKER_SMOKE") != "1":
        pytest.skip("opt-in: set MEMINI_DOCKER_SMOKE=1 and -m docker to run the real-docker smoke")
    docker = shutil.which("docker")
    if (
        docker is None
        or subprocess.run(  # noqa: S603
            [docker, "info"], capture_output=True, check=False
        ).returncode
        != 0
    ):
        pytest.skip("no running docker daemon")
    work = tmp_path / "slice"
    work.mkdir()
    shutil.copyfile(ROOT / "uv.lock", work / "uv.lock")
    shutil.copyfile(ROOT / "pyproject.toml", work / "pyproject.toml")
    (work / "tests").mkdir()
    (work / "tests" / "test_x.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")
    ex = DockerExecutor(
        cli="docker",
        image="memini-orchestrator-sandbox-test",
        dockerfile=ROOT / "tools" / "orchestrator" / "Dockerfile",
        lock=ROOT / "uv.lock",
        env_allow=["PATH"],
    )
    res = ex.run(["python", "-P", "-m", "pytest", "-q", "tests/test_x.py"], cwd=work, timeout=600)
    assert res.code == 0, res.stderr
