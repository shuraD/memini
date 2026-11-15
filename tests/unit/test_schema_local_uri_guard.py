"""GS-23: schema.is_local_uri answers through the infra database guard."""

from __future__ import annotations

import re
from typing import Any

import pytest

from memini.core import schema
from memini.infra.graph import guard

LOCAL_URIS = [
    "bolt://localhost:7688",
    "neo4j://127.0.0.1:7687",
    "bolt://[::1]:7687",
    "bolt://LOCALHOST:7687",
    "neo4j+s://localhost",
    "neo4j://localhost:7687/",
]

REFUSED_URIS = [
    "bolt://203.0.113.10:7687",
    "neo4j+s://db.example.com",
    "bolt://localhost.evil.com:7687",
    "bolt://localhost:7687/db",
    "bolt://user:pw@localhost:7687",
    "bolt://user@localhost:7687",
    "bolt://localhost:7687?x=1",
    "bolt://localhost:7687#f",
    "bolt://localhost:abc",
    "http://localhost:7474",
    "https://localhost",
    "ftp://localhost",
    "localhost:7687",
    "",
]


@pytest.mark.parametrize("uri", LOCAL_URIS)
def test_local_uris_are_accepted(uri: str) -> None:
    assert schema.is_local_uri(uri) is True


@pytest.mark.parametrize("uri", REFUSED_URIS)
def test_non_local_or_malformed_uris_are_refused(uri: str) -> None:
    assert schema.is_local_uri(uri) is False


def test_allowed_hosts_env_does_not_widen_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMINI_NEO4J_ALLOWED_HOSTS", "db.example.com")
    assert schema.is_local_uri("bolt://db.example.com") is False


def test_prod_host_env_is_not_checked_by_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMINI_PROD_NEO4J_HOST", "localhost")
    assert schema.is_local_uri("bolt://localhost:7687") is True


def test_delegates_exactly_once_to_guard_check_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def recorder(*args: Any, **kwargs: Any) -> str:
        calls.append((args, kwargs))
        return "localhost"

    monkeypatch.setattr(guard, "check_uri", recorder)
    assert schema.is_local_uri("http://anything") is True
    assert len(calls) == 1
    args, kwargs = calls[0]
    all_args = list(args)
    if "uri" in kwargs:
        all_args.insert(0, kwargs["uri"])
    assert all_args == ["http://anything"]
    assert set(kwargs) <= {"uri", "allowed_hosts", "prod_host"}
    assert not list(kwargs.get("allowed_hosts", ()))
    assert not kwargs.get("prod_host")


def test_guard_refusal_maps_to_false(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> str:
        raise guard.UnsafeDatabaseError("refused (test)")

    monkeypatch.setattr(guard, "check_uri", refuse)
    assert schema.is_local_uri("bolt://localhost:7687") is False


def test_local_hosts_is_reexported_from_guard() -> None:
    assert schema.LOCAL_HOSTS is guard.LOCAL_HOSTS


def test_schema_defines_no_uri_regex() -> None:
    assert not hasattr(schema, "_URI_RE")
    for name, value in vars(schema).items():
        if isinstance(value, re.Pattern):
            assert "://" not in str(value.pattern), name
