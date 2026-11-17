"""GR-05 acceptance (no database): one process-wide neo4j driver, built only from NEO4J_*.

``neo4j.GraphDatabase.driver`` is replaced, only around each test, by a factory that returns a
new fake driver per call and records every call. The shared driver is reset with
``close_driver()`` before and after each test.
"""

from __future__ import annotations

import ast
import inspect
import ipaddress
import os
import re
from pathlib import Path
from typing import Any

import neo4j
import pytest

from memini.infra.graph import store as store_module
from memini.infra.graph.store import GraphStore, close_driver, get_driver

ENV_VARS = ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE")
REQUIRED = ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
_MISSING = "<missing>"
USER = "test-user"
PASSWORD = "test-password"  # noqa: S105 — obvious fake


# The address is never written here: it is NEO4J_URI from the environment (.env), read once
# at import, before any fixture clears the variables.
ENV_URI = os.environ.get("NEO4J_URI", "")


def _env_uri() -> str:
    assert ENV_URI, "NEO4J_URI must be set: the address comes from the environment"
    return ENV_URI


class FakeDriver:
    def __init__(self) -> None:
        self.verified = 0
        self.close_calls = 0
        self.statements: list[str] = []
        self.query_databases: list[Any] = []

    def verify_connectivity(self, *args: Any, **kwargs: Any) -> None:
        self.verified += 1

    def execute_query(self, query: Any, *args: Any, **kwargs: Any) -> neo4j.EagerResult:
        self.statements.append(str(getattr(query, "text", query)))
        self.query_databases.append(kwargs.get("database_", _MISSING))
        return neo4j.EagerResult([], None, [])  # type: ignore[arg-type]

    def close(self) -> None:
        self.close_calls += 1


class DriverFactory:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.drivers: list[FakeDriver] = []

    def __call__(self, *args: Any, **kwargs: Any) -> FakeDriver:
        self.calls.append((args, kwargs))
        driver = FakeDriver()
        self.drivers.append(driver)
        return driver


@pytest.fixture
def factory(monkeypatch: pytest.MonkeyPatch) -> Any:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    fake = DriverFactory()
    monkeypatch.setattr(neo4j.GraphDatabase, "driver", fake)
    close_driver()
    yield fake
    close_driver()


def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", _env_uri())
    monkeypatch.setenv("NEO4J_USER", USER)
    monkeypatch.setenv("NEO4J_PASSWORD", PASSWORD)


def _uri_of(call: tuple[tuple[Any, ...], dict[str, Any]]) -> Any:
    args, kwargs = call
    return args[0] if args else kwargs.get("uri")


# --- get_driver ---------------------------------------------------------------------


def test_get_driver_twice_creates_one_driver(
    factory: DriverFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_env(monkeypatch)
    first = get_driver()
    second = get_driver()
    assert len(factory.calls) == 1
    assert _uri_of(factory.calls[0]) == _env_uri()
    assert factory.calls[0][1].get("auth") == (USER, PASSWORD)
    assert first is second
    assert first is factory.drivers[0]


def test_two_stores_share_one_driver(
    factory: DriverFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_env(monkeypatch)
    a = GraphStore().connect()
    b = GraphStore().connect()
    assert len(factory.calls) == 1
    shared = factory.drivers[0]
    assert shared.verified >= 1
    assert get_driver() is shared
    assert a is not b


def test_connect_returns_self(factory: DriverFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)
    store = GraphStore()
    assert store.connect() is store


def test_store_close_keeps_shared_driver(
    factory: DriverFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_env(monkeypatch)
    store = GraphStore().connect()
    shared = factory.drivers[0]
    store.close()
    assert shared.close_calls == 0
    assert get_driver() is shared
    assert len(factory.calls) == 1


def test_close_driver_twice_closes_once_then_recreates(
    factory: DriverFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_env(monkeypatch)
    first = get_driver()
    close_driver()
    close_driver()
    assert factory.drivers[0].close_calls == 1
    second = get_driver()
    assert len(factory.calls) == 2
    assert second is factory.drivers[1]
    assert second is not first


def test_close_driver_without_driver_is_noop(factory: DriverFactory) -> None:
    close_driver()
    assert factory.calls == []


# --- missing settings ---------------------------------------------------------------


@pytest.mark.parametrize("missing", REQUIRED)
@pytest.mark.parametrize("empty", [False, True], ids=["unset", "empty"])
def test_get_driver_missing_variable(
    factory: DriverFactory, monkeypatch: pytest.MonkeyPatch, missing: str, empty: bool
) -> None:
    _set_env(monkeypatch)
    if empty:
        monkeypatch.setenv(missing, "")
    else:
        monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError, match=missing):
        get_driver()
    assert factory.calls == []


@pytest.mark.parametrize("missing", REQUIRED)
@pytest.mark.parametrize("empty", [False, True], ids=["unset", "empty"])
def test_connect_missing_variable(
    factory: DriverFactory, monkeypatch: pytest.MonkeyPatch, missing: str, empty: bool
) -> None:
    _set_env(monkeypatch)
    if empty:
        monkeypatch.setenv(missing, "")
    else:
        monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError, match=missing):
        GraphStore().connect()
    assert factory.calls == []


def test_constructor_does_no_io(factory: DriverFactory) -> None:
    GraphStore()
    assert factory.calls == []


# --- database -----------------------------------------------------------------------


def test_database_from_env_passed_to_queries(
    factory: DriverFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_env(monkeypatch)
    monkeypatch.setenv("NEO4J_DATABASE", "testdb")
    store = GraphStore()
    assert store.database == "testdb"
    store.connect()
    shared = factory.drivers[0]
    assert shared.query_databases
    assert all(db == "testdb" for db in shared.query_databases)


def test_database_unset_is_none(factory: DriverFactory) -> None:
    assert GraphStore().database is None


def test_database_empty_is_none(factory: DriverFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_DATABASE", "")
    assert GraphStore().database is None


# --- no host refusal ----------------------------------------------------------------


def test_any_host_connects(factory: DriverFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)
    store = GraphStore()
    assert store.connect() is store
    assert len(factory.calls) == 1
    assert _uri_of(factory.calls[0]) == _env_uri()


# --- removed names ------------------------------------------------------------------


def test_removed_names_absent() -> None:
    assert not hasattr(store_module, "is_local_uri")
    assert not hasattr(store_module, "UnsafeHostError")


@pytest.mark.parametrize("kwarg", ["allowed_hosts", "uri", "user", "password"])
def test_constructor_rejects_removed_arguments(factory: DriverFactory, kwarg: str) -> None:
    value: Any = ["test-value"] if kwarg == "allowed_hosts" else "test-value"
    with pytest.raises(TypeError):
        GraphStore(**{kwarg: value})


def _source() -> str:
    path = inspect.getsourcefile(store_module)
    assert path is not None
    return Path(path).read_text(encoding="utf-8")


def test_source_has_no_uri() -> None:
    source = _source()
    assert re.search(r"[A-Za-z][A-Za-z0-9+.\-]*://", source) is None


def _is_ip(text: str) -> bool:
    try:
        ipaddress.ip_address(text.strip().strip("[]"))
    except ValueError:
        return False
    return True


def test_source_has_no_ip_address_string() -> None:
    tree = ast.parse(_source())
    strings = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert [s for s in strings if _is_ip(s)] == []
