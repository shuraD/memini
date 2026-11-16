"""GR-01: deploy/docker-compose.yml runs a local Neo4j, and deploy/.env.example lists its variables.

Static checks of the two files only: no Docker, no network.
"""

import json
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "deploy" / "docker-compose.yml"
ENV_EXAMPLE = ROOT / "deploy" / ".env.example"

AUTH_RE = re.compile(r"^neo4j/\$\{NEO4J_PASSWORD(:?\?[^}]*)?\}$")
VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)")
SECRET_WORDS = ("PASSWORD", "SECRET", "TOKEN", "KEY")


def _compose() -> dict[str, Any]:
    data = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _service() -> dict[str, Any]:
    services = _compose()["services"]
    assert isinstance(services, dict)
    assert len(services) == 1
    (svc,) = services.values()
    assert isinstance(svc, dict)
    return svc


def _environment() -> dict[str, str]:
    env = _service()["environment"]
    if isinstance(env, list):
        out = {}
        for item in env:
            key, _, value = str(item).partition("=")
            out[key] = value
        return out
    assert isinstance(env, dict)
    return {str(k): str(v) for k, v in env.items()}


def _ports() -> list[tuple[str, int]]:
    """Return (host_ip, container_port) for every published port."""
    out = []
    for port in _service()["ports"]:
        if isinstance(port, dict):
            out.append((str(port.get("host_ip", "")), int(str(port["target"]).split("/")[0])))
            continue
        spec = str(port).split("/")[0]
        parts = spec.rsplit(":", 2)
        host_ip = parts[0] if len(parts) == 3 else ""
        out.append((host_ip, int(parts[-1])))
    return out


def _dotenv() -> dict[str, str]:
    out = {}
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, sep, value = line.partition("=")
        assert sep == "=", raw
        out[key.strip()] = value.strip().strip("\"'")
    return out


def test_compose_has_exactly_one_service() -> None:
    assert isinstance(_service(), dict)


def test_image_is_official_neo4j_2026_09() -> None:
    image = str(_service()["image"])
    repo, _, tag = image.rpartition(":")
    assert repo in {"neo4j", "docker.io/library/neo4j"}
    assert tag.startswith("2026.09")


def test_plugins_are_gds_and_apoc() -> None:
    plugins = json.loads(_environment()["NEO4J_PLUGINS"])
    assert isinstance(plugins, list)
    assert {"graph-data-science", "apoc"} <= set(plugins)


def test_password_comes_from_environment_without_default() -> None:
    auth = _environment()["NEO4J_AUTH"]
    assert AUTH_RE.match(auth), auth
    assert ":-" not in auth


def test_bolt_and_http_ports_are_published() -> None:
    assert {7687, 7474} <= {target for _, target in _ports()}


def test_every_port_is_bound_to_localhost_only() -> None:
    ports = _ports()
    assert ports
    assert all(host_ip == "127.0.0.1" for host_ip, _ in ports), ports


def test_data_lives_in_a_declared_named_volume() -> None:
    sources = []
    for vol in _service()["volumes"]:
        if isinstance(vol, dict):
            if vol.get("target") == "/data":
                assert vol.get("type", "volume") == "volume"
                sources.append(str(vol["source"]))
            continue
        parts = str(vol).split(":")
        if len(parts) >= 2 and parts[1] == "/data":
            sources.append(parts[0])
    assert len(sources) == 1
    name = sources[0]
    assert "/" not in name
    assert not name.startswith(".")
    declared = _compose().get("volumes") or {}
    assert name in declared


def test_env_example_lists_password_without_value() -> None:
    env = _dotenv()
    assert "NEO4J_PASSWORD" in env
    assert env["NEO4J_PASSWORD"] == ""


def test_env_example_lists_every_compose_variable() -> None:
    used = set(VAR_RE.findall(COMPOSE.read_text(encoding="utf-8")))
    assert "NEO4J_PASSWORD" in used
    assert used <= set(_dotenv())


def test_env_example_has_no_secret_value() -> None:
    secrets = {
        k: v for k, v in _dotenv().items() if any(w in k.upper() for w in SECRET_WORDS) and v
    }
    assert secrets == {}
