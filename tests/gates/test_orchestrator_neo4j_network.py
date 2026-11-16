"""The docker sandbox can reach the test Neo4j over an internal network (GR-01 follow-up).

Unconfigured, the sandbox keeps ``--network none`` and skips the integration marker. With
``sandbox_network`` set, the container joins that network, gets NEO4J_URI, NEO4J_USER and
NEO4J_PASSWORD, and integration tests are no longer skipped. No Docker daemon is needed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from tests.gates.test_orchestrator_docker import FAKE
from tools.orchestrator.config import load_config
from tools.orchestrator.dockerexec import DockerExecutor, docker_run_argv
from tools.orchestrator.pipeline import _Run

ROOT = Path(__file__).resolve().parents[2]
NET = "memini-test-net"


def _executor(tmp_path: Path, **kwargs: str) -> tuple[DockerExecutor, Path]:
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
        env_allow=["PATH"],
        **kwargs,
    )
    return ex, Path(str(cli) + ".log")


def _run_args(record: Path) -> list[str]:
    calls = [c.split("\0") for c in record.read_text().splitlines()]
    (run,) = [c for c in calls if c and c[0] == "run"]
    return run


def test_argv_joins_the_given_network() -> None:
    argv = docker_run_argv(
        "docker", "i:t", workdir="/slice", env={}, memory="2g", pids=1, network=NET
    )
    assert argv[argv.index("--network") + 1] == NET
    assert "none" not in argv


def test_argv_defaults_to_no_network() -> None:
    argv = docker_run_argv("docker", "i:t", workdir="/slice", env={}, memory="2g", pids=1)
    assert argv[argv.index("--network") + 1] == "none"


def test_configured_run_joins_the_network_and_passes_the_neo4j_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-pw")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    ex, record = _executor(
        tmp_path, network=NET, neo4j_uri="bolt://memini-neo4j:7687", neo4j_user="neo4j"
    )
    work = tmp_path / "slice"
    work.mkdir()
    assert ex.run(["pytest", "-q"], cwd=work).code == 0
    argv = _run_args(record)
    assert argv[argv.index("--network") + 1] == NET
    assert "NEO4J_URI=bolt://memini-neo4j:7687" in argv
    assert "NEO4J_USER=neo4j" in argv
    assert "NEO4J_PASSWORD=test-pw" in argv
    assert "ANTHROPIC_API_KEY=secret" not in " ".join(argv)


def test_the_coder_cannot_override_the_neo4j_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEO4J_PASSWORD", "test-pw")
    ex, record = _executor(
        tmp_path, network=NET, neo4j_uri="bolt://memini-neo4j:7687", neo4j_user="neo4j"
    )
    work = tmp_path / "slice"
    work.mkdir()
    ex.run(["pytest"], cwd=work, env={"NEO4J_URI": "bolt://prod.example:7687"})
    uris = [a for a in _run_args(record) if a.startswith("NEO4J_URI=")]
    assert uris == ["NEO4J_URI=bolt://memini-neo4j:7687"]


def test_unconfigured_run_keeps_no_network_and_no_neo4j_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEO4J_PASSWORD", "test-pw")
    ex, record = _executor(tmp_path)
    work = tmp_path / "slice"
    work.mkdir()
    assert ex.run(["pytest", "-q"], cwd=work).code == 0
    argv = _run_args(record)
    assert argv[argv.index("--network") + 1] == "none"
    assert not [a for a in argv if a.startswith("NEO4J_")]


def test_configured_run_without_a_password_starts_no_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    ex, record = _executor(
        tmp_path, network=NET, neo4j_uri="bolt://memini-neo4j:7687", neo4j_user="neo4j"
    )
    work = tmp_path / "slice"
    work.mkdir()
    res = ex.run(["pytest"], cwd=work)
    assert res.code != 0
    assert "NEO4J_PASSWORD" in res.stderr
    assert not record.exists() or "run" not in [
        c.split("\0")[0] for c in record.read_text().splitlines()
    ]


def test_integration_is_skipped_only_without_the_network() -> None:
    plain = load_config(ROOT, overrides={"sandbox_network": ""})
    joined = load_config(ROOT, overrides={"sandbox_network": NET})
    assert plain.sandbox_network == ""
    assert "integration" in plain.sandbox_skip_markers
    assert joined.sandbox_network == NET
    run = SimpleNamespace(cfg=joined, sandbox=object())
    assert _Run._skip_markers(run) == ()  # type: ignore[arg-type]
    run = SimpleNamespace(cfg=plain, sandbox=object())
    assert _Run._skip_markers(run) == ("integration",)  # type: ignore[arg-type]


def test_compose_puts_neo4j_on_an_internal_network_and_keeps_its_ports() -> None:
    compose = yaml.safe_load((ROOT / "deploy" / "docker-compose.yml").read_text(encoding="utf-8"))
    net = compose["networks"]["sandbox"]
    assert net["internal"] is True
    assert net["name"] == NET
    service = compose["services"]["neo4j"]
    assert "sandbox" in service["networks"]
    assert "default" in service["networks"]  # published ports need a non-internal network
