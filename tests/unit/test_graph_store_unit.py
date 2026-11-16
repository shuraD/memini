"""GR-03 acceptance (no database): GraphStore settings, host guard, connect and batching.

The neo4j driver is replaced, only around each test, by a fake that records every statement
(``driver.execute_query``, ``session.run``, ``tx.run``) and every session's database.
"""

from __future__ import annotations

import re
import socket
from typing import Any

import neo4j
import pytest

from memini.infra.graph.store import GraphStore, UnsafeHostError, is_local_uri

ENV_VARS = ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE")
ID_LABELS = {"Artifact", "Chunk", "Entity", "Community"}
_MISSING = "<missing>"
_CONSTRAINT_LABEL = re.compile(r"FOR\s*\(\s*\w*\s*:\s*`?(\w+)`?\s*\)", re.IGNORECASE)


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
def _no_neo4j_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _install(monkeypatch: pytest.MonkeyPatch, driver: FakeDriver | None = None) -> DriverFactory:
    factory = DriverFactory(driver if driver is not None else FakeDriver())
    monkeypatch.setattr(neo4j.GraphDatabase, "driver", factory)
    return factory


def _items(n: int) -> list[dict[str, Any]]:
    return [{"id": f"item-{i}", "content_hash": f"hash-{i}", "name": f"n{i}"} for i in range(n)]


# --- settings ---------------------------------------------------------------------


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7999")
    monkeypatch.setenv("NEO4J_USER", "env-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
    monkeypatch.setenv("NEO4J_DATABASE", "envdb")
    store = GraphStore()
    assert store.uri == "bolt://localhost:7999"
    assert store.user == "env-user"
    assert store.password == "test-password"  # noqa: S105 — obvious fake
    assert store.database == "envdb"


def test_constructor_args_win_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7999")
    monkeypatch.setenv("NEO4J_USER", "env-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
    monkeypatch.setenv("NEO4J_DATABASE", "envdb")
    store = GraphStore("neo4j://localhost:7111", "arg-user", "test-other-password", "argdb")
    assert store.uri == "neo4j://localhost:7111"
    assert store.user == "arg-user"
    assert store.password == "test-other-password"  # noqa: S105 — obvious fake
    assert store.database == "argdb"


def test_constructor_keywords_win_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7999")
    monkeypatch.setenv("NEO4J_USER", "env-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
    monkeypatch.setenv("NEO4J_DATABASE", "envdb")
    store = GraphStore(uri="bolt://localhost:7222", user="kw-user", password="", database="kwdb")
    assert store.uri == "bolt://localhost:7222"
    assert store.user == "kw-user"
    assert store.password == ""
    assert store.database == "kwdb"


def test_defaults_without_env() -> None:
    store = GraphStore()
    assert store.uri == "bolt://localhost:7687"
    assert store.user == "neo4j"
    assert store.password == ""
    assert store.database is None


def test_empty_database_env_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_DATABASE", "")
    assert GraphStore().database is None


def test_constructor_does_no_io(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _install(monkeypatch)
    GraphStore("bolt://localhost:7687", "neo4j", "test-password", "somedb")
    assert factory.calls == []


# --- host guard ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "uri",
    [
        "bolt://localhost:7687",
        "neo4j://localhost:7687",
        "bolt://LOCALHOST:7687",
        "bolt://LocalHost",
        "bolt://127.0.0.1:7687",
        "neo4j://127.0.0.1",
        "bolt://[::1]:7687",
        "neo4j+s://localhost:7687",
    ],
)
def test_local_uris_are_local(uri: str) -> None:
    assert is_local_uri(uri) is True


@pytest.mark.parametrize(
    "uri",
    [
        "bolt://db.example.com:7687",
        "neo4j+s://db.example.com",
        "bolt://localhost.example.com:7687",
        "bolt://mylocalhost:7687",
        "bolt://memini-neo4j:7687",
    ],
)
def test_other_hosts_are_not_local(uri: str) -> None:
    assert is_local_uri(uri) is False


def test_allowed_host_is_accepted() -> None:
    assert is_local_uri("bolt://db.example.com:7687", ["db.example.com"]) is True
    assert is_local_uri("bolt://db.example.com:7687", ("other.example.com",)) is False
    assert is_local_uri("bolt://db.example.com:7687", ()) is False


def test_is_local_uri_does_no_dns_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    def _refuse(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        raise AssertionError("DNS lookup")

    with monkeypatch.context() as patch:
        for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
            patch.setattr(socket, name, _refuse)
        results = [
            is_local_uri("bolt://db.example.com:7687"),
            is_local_uri("bolt://localhost:7687"),
            is_local_uri("bolt://db.example.com:7687", ["db.example.com"]),
        ]
    assert results == [False, True, True]
    assert calls == []


def test_unsafe_host_error_is_value_error() -> None:
    assert issubclass(UnsafeHostError, ValueError)


def test_non_local_uri_argument_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _install(monkeypatch)
    with pytest.raises(UnsafeHostError):
        GraphStore("bolt://db.example.com:7687")
    assert factory.calls == []


def test_non_local_uri_env_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _install(monkeypatch)
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost.example.com:7687")
    with pytest.raises(UnsafeHostError):
        GraphStore()
    assert factory.calls == []


def test_allow_listed_host_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    store = GraphStore("bolt://db.example.com:7687", allowed_hosts=["db.example.com"])
    assert store.uri == "bolt://db.example.com:7687"
    monkeypatch.setenv("NEO4J_URI", "bolt://db.example.com:7687")
    assert GraphStore(allowed_hosts=("db.example.com",)).uri == "bolt://db.example.com:7687"


# --- connect ------------------------------------------------------------------------


def test_connect_opens_driver_and_verifies(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _install(monkeypatch)
    store = GraphStore("bolt://localhost:7687", "qa-user", "test-password")
    result = store.connect()
    assert result is store
    assert len(factory.calls) == 1
    args, kwargs = factory.calls[0]
    assert args == ("bolt://localhost:7687",)
    assert kwargs.get("auth") == ("qa-user", "test-password")
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


def test_close_closes_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _install(monkeypatch)
    store = GraphStore().connect()
    store.close()
    assert factory.driver.closed is True


# --- database routing ---------------------------------------------------------------


def test_every_call_uses_the_database(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _install(monkeypatch)
    store = GraphStore(database="qa-db").connect()
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
