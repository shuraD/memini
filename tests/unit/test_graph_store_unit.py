"""GR-03 acceptance (no database): GraphStore settings, connect and batching.

The neo4j driver is replaced, only around each test, by a fake that records every statement
(``driver.execute_query``, ``session.run``, ``tx.run``) and every session's database. The
settings come only from NEO4J_* (GR-05: no host check, no default address, no constructor
arguments); the shared driver is reset with ``close_driver()`` around each test.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from typing import Any

import neo4j
import pytest

from memini.infra.graph.store import GraphStore, close_driver

ENV_VARS = ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE")
ID_LABELS = {"Artifact", "Chunk", "Entity", "Community"}
_MISSING = "<missing>"
USER = "test-user"
PASSWORD = "test-password"  # noqa: S105 — obvious fake
_CONSTRAINT_LABEL = re.compile(r"FOR\s*\(\s*\w*\s*:\s*`?(\w+)`?\s*\)", re.IGNORECASE)


# The address is never written here: it is NEO4J_URI from the environment (.env), read once
# at import, before any fixture clears the variables.
ENV_URI = os.environ.get("NEO4J_URI", "")


def _env_uri() -> str:
    assert ENV_URI, "NEO4J_URI must be set: the address comes from the environment"
    return ENV_URI


def _text(query: Any) -> str:
    if isinstance(query, str):
        return query
    return str(getattr(query, "text", query))


class FakeResult:
    """An empty neo4j result, whichever way it is read."""

    def __iter__(self) -> Any:
        return iter(())

    def data(self, *args: Any) -> list[Any]:
        return []

    def single(self, *args: Any, **kwargs: Any) -> None:
        return None

    def peek(self) -> None:
        return None

    def value(self, *args: Any) -> list[Any]:
        return []

    def values(self, *args: Any) -> list[Any]:
        return []

    def fetch(self, n: int) -> list[Any]:
        return []

    def keys(self) -> list[str]:
        return []

    def consume(self) -> None:
        return None

    def to_eager_result(self) -> neo4j.EagerResult:
        return neo4j.EagerResult([], None, [])  # type: ignore[arg-type]


class FakeTx:
    def __init__(self, driver: FakeDriver) -> None:
        self._driver = driver

    def run(self, query: Any, parameters: Any = None, **kwargs: Any) -> FakeResult:
        self._driver.statements.append(_text(query))
        return FakeResult()

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> FakeTx:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class FakeSession:
    def __init__(self, driver: FakeDriver) -> None:
        self._driver = driver

    def run(self, query: Any, parameters: Any = None, **kwargs: Any) -> FakeResult:
        self._driver.statements.append(_text(query))
        return FakeResult()

    def execute_write(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return fn(FakeTx(self._driver), *args, **kwargs)

    def execute_read(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return fn(FakeTx(self._driver), *args, **kwargs)

    def begin_transaction(self, *args: Any, **kwargs: Any) -> FakeTx:
        return FakeTx(self._driver)

    def close(self) -> None:
        return None

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class FakeDriver:
    def __init__(self, verify_error: Exception | None = None) -> None:
        self.verify_error = verify_error
        self.verified = 0
        self.closed = False
        self.statements: list[str] = []
        self.query_databases: list[Any] = []
        self.session_databases: list[Any] = []

    def verify_connectivity(self, *args: Any, **kwargs: Any) -> None:
        self.verified += 1
        if self.verify_error is not None:
            raise self.verify_error

    def execute_query(self, query: Any, *args: Any, **kwargs: Any) -> neo4j.EagerResult:
        self.statements.append(_text(query))
        self.query_databases.append(kwargs.get("database_", _MISSING))
        return neo4j.EagerResult([], None, [])  # type: ignore[arg-type]

    def session(self, *args: Any, **kwargs: Any) -> FakeSession:
        self.session_databases.append(kwargs.get("database", _MISSING))
        return FakeSession(self)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> FakeDriver:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class DriverFactory:
    def __init__(self, driver: FakeDriver) -> None:
        self.driver = driver
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> FakeDriver:
        self.calls.append((args, kwargs))
        return self.driver


@pytest.fixture(autouse=True)
def _neo4j_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NEO4J_URI", _env_uri())
    monkeypatch.setenv("NEO4J_USER", USER)
    monkeypatch.setenv("NEO4J_PASSWORD", PASSWORD)
    close_driver()
    yield
    close_driver()


def _install(monkeypatch: pytest.MonkeyPatch, driver: FakeDriver | None = None) -> DriverFactory:
    close_driver()  # the next get_driver() builds a new driver from this factory
    factory = DriverFactory(driver if driver is not None else FakeDriver())
    monkeypatch.setattr(neo4j.GraphDatabase, "driver", factory)
    return factory


def _items(n: int) -> list[dict[str, Any]]:
    return [{"id": f"item-{i}", "content_hash": f"hash-{i}", "name": f"n{i}"} for i in range(n)]


# --- settings ---------------------------------------------------------------------


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _install(monkeypatch)
    monkeypatch.setenv("NEO4J_URI", _env_uri())
    monkeypatch.setenv("NEO4J_USER", "env-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
    monkeypatch.setenv("NEO4J_DATABASE", "envdb")
    store = GraphStore().connect()
    args, kwargs = factory.calls[0]
    assert args == (_env_uri(),)
    assert kwargs.get("auth") == ("env-user", "test-password")
    assert store.database == "envdb"


def test_empty_database_env_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_DATABASE", "")
    assert GraphStore().database is None


def test_constructor_does_no_io(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _install(monkeypatch)
    monkeypatch.setenv("NEO4J_DATABASE", "somedb")
    GraphStore()
    assert factory.calls == []


# --- connect ------------------------------------------------------------------------


def test_connect_opens_driver_and_verifies(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _install(monkeypatch)
    store = GraphStore()
    result = store.connect()
    assert result is store
    assert len(factory.calls) == 1
    args, kwargs = factory.calls[0]
    assert args == (_env_uri(),)
    assert kwargs.get("auth") == (USER, PASSWORD)
    assert factory.driver.verified == 1


def test_connect_creates_scope_id_constraints(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _install(monkeypatch)
    GraphStore().connect()
    labels: set[str] = set()
    for statement in factory.driver.statements:
        if re.match(r"\s*CREATE\s+CONSTRAINT\b", statement, re.IGNORECASE):
            assert "IF NOT EXISTS" in statement.upper(), statement
            assert re.search(r"\.\s*`?scope`?", statement), statement
            assert re.search(r"\.\s*`?id`?\b", statement), statement
            assert "UNIQUE" in statement.upper(), statement
            match = _CONSTRAINT_LABEL.search(statement)
            assert match, statement
            labels.add(match.group(1))
    assert labels == ID_LABELS


def test_connect_failure_propagates_and_runs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    error = neo4j.exceptions.ServiceUnavailable("test: no database")
    factory = _install(monkeypatch, FakeDriver(verify_error=error))
    store = GraphStore()
    with pytest.raises(neo4j.exceptions.ServiceUnavailable) as caught:
        store.connect()
    assert caught.value is error
    assert factory.driver.statements == []


def test_connect_other_error_propagates_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    error = neo4j.exceptions.AuthError("test: bad credentials")
    factory = _install(monkeypatch, FakeDriver(verify_error=error))
    with pytest.raises(neo4j.exceptions.AuthError) as caught:
        GraphStore().connect()
    assert caught.value is error
    assert factory.driver.statements == []


def test_driver_factory_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    error = neo4j.exceptions.ConfigurationError("test: bad config")

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", _boom)
    with pytest.raises(neo4j.exceptions.ConfigurationError) as caught:
        GraphStore().connect()
    assert caught.value is error


# --- database routing ---------------------------------------------------------------


def test_every_call_uses_the_database(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _install(monkeypatch)
    monkeypatch.setenv("NEO4J_DATABASE", "qa-db")
    store = GraphStore().connect()
    store.write("Entity", "qa-scope", _items(3), updated_by="qa")
    driver = factory.driver
    assert driver.statements
    assert driver.query_databases or driver.session_databases
    assert all(db == "qa-db" for db in driver.query_databases), driver.query_databases
    assert all(db == "qa-db" for db in driver.session_databases), driver.session_databases


def test_database_from_env_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _install(monkeypatch)
    monkeypatch.setenv("NEO4J_DATABASE", "env-db")
    store = GraphStore().connect()
    store.write("Artifact", "qa-scope", _items(2), updated_by="qa")
    driver = factory.driver
    assert driver.query_databases or driver.session_databases
    assert all(db == "env-db" for db in driver.query_databases)
    assert all(db == "env-db" for db in driver.session_databases)


def test_home_database_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _install(monkeypatch)
    store = GraphStore().connect()
    store.write("Entity", "qa-scope", _items(2), updated_by="qa")
    driver = factory.driver
    assert driver.query_databases or driver.session_databases
    assert all(db is None for db in driver.query_databases)
    assert all(db is None for db in driver.session_databases)


# --- write batching -------------------------------------------------------------------


def _statements_for(monkeypatch: pytest.MonkeyPatch, label: str, n: int) -> int:
    factory = _install(monkeypatch)
    store = GraphStore().connect()
    before = len(factory.driver.statements)
    store.write(label, "qa-scope", _items(n), updated_by="qa")
    return len(factory.driver.statements) - before


@pytest.mark.parametrize("label", ["Entity", "Chunk"])
def test_write_batch_size_does_not_change_statement_count(
    monkeypatch: pytest.MonkeyPatch, label: str
) -> None:
    one = _statements_for(monkeypatch, label, 1)
    fifty = _statements_for(monkeypatch, label, 50)
    assert one >= 1
    assert one == fifty


def test_write_uses_unwind(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _install(monkeypatch)
    store = GraphStore().connect()
    before = len(factory.driver.statements)
    store.write("Entity", "qa-scope", _items(5), updated_by="qa")
    written = factory.driver.statements[before:]
    assert any("UNWIND" in s.upper() and "MERGE" in s.upper() for s in written), written


@pytest.mark.parametrize("label", ["Nope", "entity", "SUPERSEDES", ""])
def test_unknown_label_raises_before_any_statement(
    monkeypatch: pytest.MonkeyPatch, label: str
) -> None:
    factory = _install(monkeypatch)
    store = GraphStore().connect()
    before = len(factory.driver.statements)
    sessions_before = len(factory.driver.session_databases)
    with pytest.raises(ValueError):
        store.write(label, "qa-scope", _items(2), updated_by="qa")
    assert len(factory.driver.statements) == before
    assert len(factory.driver.session_databases) == sessions_before
