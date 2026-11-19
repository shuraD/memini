"""The sandbox password comes from the env file, never from the coder's agent environment."""

# ruff: noqa: F811
from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import pytest
from tests.gates.test_orchestrator_enqueue import cfg  # noqa: F401  (cfg is a fixture)
from tools.orchestrator import config, dockerexec


def test_the_value_is_read_from_a_key_value_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("# c\nNEO4J_URI=bolt://x\nNEO4J_PASSWORD='pw 1'\nOTHER=2\n")
    assert dockerexec.env_file_value(env, "NEO4J_PASSWORD") == "pw 1"
    assert dockerexec.env_file_value(env, "MISSING") == ""


def test_a_missing_file_gives_nothing(tmp_path: Path) -> None:
    assert dockerexec.env_file_value(tmp_path / "none", "NEO4J_PASSWORD") == ""
    assert dockerexec.env_file_value(None, "NEO4J_PASSWORD") == ""


def _executor(tmp_path: Path, env_file: Path | None) -> dockerexec.DockerExecutor:
    ex = dockerexec.DockerExecutor(
        cli="docker",
        image="i",
        dockerfile=tmp_path / "Dockerfile",
        lock=tmp_path / "uv.lock",
        env_allow=[],
        network="n",
        env_file=env_file,
    )
    ex.ensure_image = lambda: None  # type: ignore[method-assign]
    return ex


def test_without_a_password_anywhere_the_sandbox_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    res = _executor(tmp_path, tmp_path / "none").run(["true"], cwd=tmp_path)
    assert (res.code, "NEO4J_PASSWORD is not" in res.output) == (125, True)


def test_the_password_in_the_file_is_enough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    env = tmp_path / ".env"
    env.write_text("NEO4J_PASSWORD=from-file\n")

    def stop(*_a: object, **_k: object) -> list[str]:
        raise RuntimeError("past the password check")

    monkeypatch.setattr(dockerexec, "docker_run_argv", stop)
    with pytest.raises(RuntimeError, match="past the password check"):
        _executor(tmp_path, env).run(["true"], cwd=tmp_path)


def test_a_slice_worktree_finds_the_main_checkouts_env_file(
    cfg: config.Config, tmp_path: Path
) -> None:
    wt = tmp_path / "slice-wt"
    subprocess.run(  # noqa: S603
        ["git", "-C", str(cfg.repo), "worktree", "add", "-q", "-b", "orch/x", str(wt)],  # noqa: S607
        check=True,
    )
    ex = dockerexec.build_executor(dataclasses.replace(cfg, repo=wt), [])
    assert ex.env_file == cfg.repo.resolve() / "deploy" / ".env"
