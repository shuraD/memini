"""Acceptance tests for GS-13: the database URI guard (memini.infra.graph.guard)."""

from __future__ import annotations

import importlib
import inspect
import re
from typing import Any

import pytest

PROD_IP = "203.0.113.10"


def _guard() -> Any:
    return importlib.import_module("memini.infra.graph.guard")


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("bolt://localhost:7687", "localhost"),
        ("neo4j://127.0.0.1:7687", "127.0.0.1"),
        ("bolt://[::1]:7687", "::1"),
        ("bolt://LOCALHOST:7687", "localhost"),
        ("bolt://localhost", "localhost"),
        ("neo4j://localhost:7687/", "localhost"),
    ],
)
def test_local_uris_return_normalised_host(uri: str, expected: str) -> None:
    assert _guard().check_uri(uri) == expected


@pytest.mark.parametrize("scheme", ["bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"])
def test_each_accepted_scheme_with_localhost(scheme: str) -> None:
    assert _guard().check_uri(f"{scheme}://localhost:7687") == "localhost"


def test_allow_listed_remote_host_is_accepted() -> None:
    guard = _guard()
    uri = "neo4j+s://db.example.com"
    assert guard.check_uri(uri, allowed_hosts=["db.example.com"]) == "db.example.com"
    assert guard.check_uri(uri, allowed_hosts=["DB.Example.com"]) == "db.example.com"


def test_remote_host_without_allow_list_is_refused() -> None:
    guard = _guard()
    with pytest.raises(guard.UnsafeDatabaseError):
        guard.check_uri("neo4j+s://db.example.com")


def test_prod_host_refused_even_when_allow_listed() -> None:
    guard = _guard()
    with pytest.raises(guard.UnsafeDatabaseError):
        guard.check_uri(f"bolt://{PROD_IP}:7687", allowed_hosts=[PROD_IP], prod_host=PROD_IP)


@pytest.mark.parametrize("prod_host", ["localhost", "LocalHost"])
def test_prod_host_refused_even_when_local(prod_host: str) -> None:
    guard = _guard()
    with pytest.raises(guard.UnsafeDatabaseError):
        guard.check_uri("bolt://localhost:7687", prod_host=prod_host)


@pytest.mark.parametrize("prod_host", ["[::1]", "::1"])
def test_prod_host_ipv6_refused_with_or_without_brackets(prod_host: str) -> None:
    guard = _guard()
    with pytest.raises(guard.UnsafeDatabaseError):
        guard.check_uri("bolt://[::1]:7687", prod_host=prod_host)


def test_other_prod_host_does_not_block_localhost() -> None:
    assert _guard().check_uri("bolt://localhost:7687", prod_host=PROD_IP) == "localhost"


@pytest.mark.parametrize(
    "uri",
    [
        "bolt://localhost.evil.com:7687",
        "bolt://localhost:7687/db",
        "bolt://user:pw@localhost:7687",
        "bolt://user@localhost:7687",
        "bolt://localhost:7687?x=1",
        "bolt://localhost:7687#f",
        "bolt://localhost:abc",
        "bolt://:7687",
        "bolt://::1:7687",
        "http://localhost:7474",
        "https://localhost",
        "file:///tmp/db",
        "localhost:7687",
        "localhost",
        "",
    ],
)
def test_unexpected_uri_shapes_are_refused(uri: str) -> None:
    guard = _guard()
    with pytest.raises(guard.UnsafeDatabaseError):
        guard.check_uri(uri)


def test_unsafe_database_error_is_value_error() -> None:
    assert issubclass(_guard().UnsafeDatabaseError, ValueError)


def test_guard_from_env_uses_allowed_hosts_list() -> None:
    env = {"MEMINI_NEO4J_ALLOWED_HOSTS": "a.example.com, b.example.com"}
    assert _guard().guard_from_env("bolt://b.example.com", env) == "b.example.com"


def test_guard_from_env_ignores_blank_entries() -> None:
    env = {"MEMINI_NEO4J_ALLOWED_HOSTS": " , ,a.example.com,"}
    assert _guard().guard_from_env("bolt://a.example.com", env) == "a.example.com"


def test_guard_from_env_empty_env() -> None:
    guard = _guard()
    assert guard.guard_from_env("bolt://localhost:7687", {}) == "localhost"
    with pytest.raises(guard.UnsafeDatabaseError):
        guard.guard_from_env("bolt://a.example.com", {})


def test_guard_from_env_reads_only_given_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _guard()
    monkeypatch.setenv("MEMINI_NEO4J_ALLOWED_HOSTS", "a.example.com")
    with pytest.raises(guard.UnsafeDatabaseError):
        guard.guard_from_env("bolt://a.example.com", {})
    monkeypatch.setenv("MEMINI_PROD_NEO4J_HOST", "localhost")
    assert guard.guard_from_env("bolt://localhost:7687", {}) == "localhost"


def test_guard_from_env_prod_host_refused_even_when_allow_listed() -> None:
    guard = _guard()
    env = {
        "MEMINI_PROD_NEO4J_HOST": "db.example.com",
        "MEMINI_NEO4J_ALLOWED_HOSTS": "db.example.com",
    }
    with pytest.raises(guard.UnsafeDatabaseError):
        guard.guard_from_env("bolt://db.example.com:7687", env)


def test_guard_from_env_prod_host_refuses_local() -> None:
    guard = _guard()
    with pytest.raises(guard.UnsafeDatabaseError):
        guard.guard_from_env("bolt://localhost:7687", {"MEMINI_PROD_NEO4J_HOST": "localhost"})


@pytest.mark.parametrize("blank", ["", "   "])
def test_guard_from_env_blank_prod_host_accepts_local(blank: str) -> None:
    env = {"MEMINI_PROD_NEO4J_HOST": blank}
    assert _guard().guard_from_env("bolt://localhost:7687", env) == "localhost"


def test_local_hosts_constant() -> None:
    assert _guard().LOCAL_HOSTS == ("localhost", "127.0.0.1", "::1")


def test_source_has_no_ipv4_literal_except_loopback() -> None:
    source = inspect.getsource(_guard())
    found = set(re.findall(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])", source))
    assert found <= {"127.0.0.1"}
