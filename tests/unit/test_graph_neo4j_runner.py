"""Acceptance tests for GS-22: the Neo4j runner behind the database guard."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pytest

from memini.infra.graph.guard import UnsafeDatabaseError
from memini.infra.graph.neo4j_runner import Neo4jRunner, RunResult

LOCAL_URI = "bolt://localhost:7687"
REMOTE_URI = "bolt://db.example.com:7687"
AUTH = ("neo4j", "test-key")


def _merge_params(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for arg in args:
        if isinstance(arg, Mapping):
            merged.update(arg)
    rest = dict(kwargs)
    parameters = rest.pop("parameters", None)
    if parameters:
        merged.update(parameters)
    merged.update(rest)
    return merged


class FakeRecord:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def data(self) -> dict[str, Any]:
        return dict(self._data)


class FakeSummary:
    def __init__(self, profile: dict[str, Any] | None) -> None:
        self.profile = profile


class FakeResult:
    def __init__(self, rows: list[dict[str, Any]], profile: dict[str, Any] | None) -> None:
        self._records = [FakeRecord(row) for row in rows]
        self._summary = FakeSummary(profile)

    def __iter__(self) -> Iterator[FakeRecord]:
        return iter(list(self._records))

    def consume(self) -> FakeSummary:
        return self._summary


class FakeTx:
    def __init__(self, owner: FakeDriver) -> None:
        self._owner = owner

    def run(self, query: str, *args: Any, **kwargs: Any) -> FakeResult:
        self._owner.tx_runs.append((query, _merge_params(args, kwargs)))
        return FakeResult(self._owner.rows, self._owner.profile)


class FakeSession:
    def __init__(self, owner: FakeDriver) -> None:
        self._owner = owner

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self._owner.sessions_closed += 1

    def close(self) -> None:
        self._owner.sessions_closed += 1

    def run(self, query: str, *args: Any, **kwargs: Any) -> FakeResult:
        self._owner.session_runs.append((query, _merge_params(args, kwargs)))
        return FakeResult(self._owner.rows, self._owner.profile)

    def execute_write(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._owner.execute_write_calls += 1
        return fn(FakeTx(self._owner), *args, **kwargs)

    def execute_read(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._owner.execute_read_calls += 1
        return fn(FakeTx(self._owner), *args, **kwargs)


class FakeDriver:
    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        profile: dict[str, Any] | None = None,
    ) -> None:
        self.rows = rows if rows is not None else []
        self.profile = profile
        self.session_kwargs: list[dict[str, Any]] = []
        self.session_runs: list[tuple[str, dict[str, Any]]] = []
        self.tx_runs: list[tuple[str, dict[str, Any]]] = []
        self.execute_write_calls = 0
        self.execute_read_calls = 0
        self.sessions_closed = 0
        self.close_calls = 0

    def session(self, **kwargs: Any) -> FakeSession:
        self.session_kwargs.append(kwargs)
        return FakeSession(self)

    def close(self) -> None:
        self.close_calls += 1

    @property
    def all_runs(self) -> list[tuple[str, dict[str, Any]]]:
        return self.session_runs + self.tx_runs


class FakeFactory:
    def __init__(self, driver: FakeDriver | None = None) -> None:
        self.driver = driver if driver is not None else FakeDriver()
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> FakeDriver:
        self.calls.append((args, kwargs))
        return self.driver


def _runner(driver: FakeDriver, *, database: str | None = None) -> tuple[Neo4jRunner, FakeFactory]:
    factory = FakeFactory(driver)
    runner = Neo4jRunner(LOCAL_URI, AUTH, database=database, env={}, driver_factory=factory)
    return runner, factory


# --- guard ---------------------------------------------------------------


def test_remote_host_without_allow_list_is_refused_before_driver() -> None:
    factory = FakeFactory()
    with pytest.raises(UnsafeDatabaseError):
        Neo4jRunner(REMOTE_URI, AUTH, env={}, driver_factory=factory)
    assert factory.calls == []


def test_production_host_is_refused_even_when_allow_listed() -> None:
    factory = FakeFactory()
    env = {
        "MEMINI_PROD_NEO4J_HOST": "db.example.com",
        "MEMINI_NEO4J_ALLOWED_HOSTS": "db.example.com",
    }
    with pytest.raises(UnsafeDatabaseError):
        Neo4jRunner(REMOTE_URI, AUTH, env=env, driver_factory=factory)
    assert factory.calls == []


def test_production_local_host_is_refused() -> None:
    factory = FakeFactory()
    env = {"MEMINI_PROD_NEO4J_HOST": "localhost"}
    with pytest.raises(UnsafeDatabaseError):
        Neo4jRunner(LOCAL_URI, AUTH, env=env, driver_factory=factory)
    assert factory.calls == []


def test_allow_listed_remote_host_builds_driver_once() -> None:
    factory = FakeFactory()
    env = {"MEMINI_NEO4J_ALLOWED_HOSTS": "db.example.com"}
    runner = Neo4jRunner(REMOTE_URI, AUTH, env=env, driver_factory=factory)
    assert factory.calls == [((REMOTE_URI,), {"auth": AUTH})]
    assert runner.uri == REMOTE_URI


def test_auth_none_is_passed_to_factory() -> None:
    factory = FakeFactory()
    Neo4jRunner(LOCAL_URI, None, env={}, driver_factory=factory)
    assert factory.calls == [((LOCAL_URI,), {"auth": None})]


def test_env_is_used_instead_of_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMINI_NEO4J_ALLOWED_HOSTS", "db.example.com")
    factory = FakeFactory()
    with pytest.raises(UnsafeDatabaseError):
        Neo4jRunner(REMOTE_URI, AUTH, env={}, driver_factory=factory)
    assert factory.calls == []


def test_real_driver_is_created_without_connecting_and_closes() -> None:
    runner = Neo4jRunner(LOCAL_URI, AUTH, env={})
    assert runner.uri == LOCAL_URI
    runner.close()


# --- run -----------------------------------------------------------------


def test_run_returns_rows_in_order_and_no_db_hits() -> None:
    driver = FakeDriver(rows=[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    runner, _ = _runner(driver)
    result = runner.run("MATCH (n) RETURN n.a AS a, n.b AS b", {})
    assert isinstance(result, RunResult)
    assert result.rows == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    assert all(type(row) is dict for row in result.rows)
    assert result.db_hits is None


def test_run_sends_exact_cypher_and_params() -> None:
    driver = FakeDriver()
    runner, _ = _runner(driver)
    cypher = "MATCH (n {x: $x}) RETURN n"
    runner.run(cypher, {"x": 1})
    assert driver.tx_runs == [(cypher, {"x": 1})]
    assert driver.session_runs == []


def test_profile_prefixes_query_and_sums_db_hits() -> None:
    profile = {"dbHits": 3, "children": [{"dbHits": 4, "children": []}, {"dbHits": 5}]}
    driver = FakeDriver(rows=[{"n": 1}], profile=profile)
    runner, _ = _runner(driver)
    cypher = "MATCH (n) RETURN n"
    result = runner.run(cypher, {"x": 1}, profile=True)
    assert len(driver.all_runs) == 1
    sent, params = driver.all_runs[0]
    assert sent == "PROFILE " + cypher
    assert params == {"x": 1}
    assert result.db_hits == 12
    assert result.rows == [{"n": 1}]


def test_profile_sums_deeper_tree_with_missing_db_hits() -> None:
    profile = {
        "dbHits": 2,
        "children": [
            {"dbHits": 3, "children": [{"dbHits": 7, "children": []}]},
            {"children": []},
        ],
    }
    driver = FakeDriver(profile=profile)
    runner, _ = _runner(driver)
    result = runner.run("MATCH (n) RETURN n", {}, profile=True)
    assert result.db_hits == 12


def test_profile_with_autocommit_prefixes_and_sums() -> None:
    profile = {"dbHits": 1, "children": [{"dbHits": 2}]}
    driver = FakeDriver(profile=profile)
    runner, _ = _runner(driver)
    result = runner.run("RETURN 1", {}, profile=True, autocommit=True)
    assert driver.session_runs == [("PROFILE RETURN 1", {})]
    assert result.db_hits == 3


def test_no_profile_ignores_summary_profile() -> None:
    profile = {"dbHits": 9, "children": []}
    driver = FakeDriver(profile=profile)
    runner, _ = _runner(driver)
    cypher = "MATCH (n) RETURN n"
    result = runner.run(cypher, {}, profile=False)
    assert result.db_hits is None
    assert [query for query, _ in driver.all_runs] == [cypher]
    assert not driver.all_runs[0][0].startswith("PROFILE")


def test_autocommit_uses_session_run_only() -> None:
    driver = FakeDriver(rows=[{"k": "v"}, {"k": "w"}])
    runner, _ = _runner(driver)
    cypher = "CALL { MATCH (n) SET n.y = $y } IN TRANSACTIONS"
    result = runner.run(cypher, {"y": 2}, autocommit=True)
    assert driver.session_runs == [(cypher, {"y": 2})]
    assert driver.tx_runs == []
    assert driver.execute_write_calls == 0
    assert driver.execute_read_calls == 0
    assert result.rows == [{"k": "v"}, {"k": "w"}]
    assert all(type(row) is dict for row in result.rows)


def test_managed_transaction_used_by_default() -> None:
    driver = FakeDriver(rows=[{"k": 1}])
    runner, _ = _runner(driver)
    result = runner.run("RETURN 1 AS k", {}, autocommit=False)
    assert driver.execute_write_calls + driver.execute_read_calls == 1
    assert driver.session_runs == []
    assert len(driver.tx_runs) == 1
    assert result.rows == [{"k": 1}]


def test_session_without_database_name() -> None:
    driver = FakeDriver()
    runner, _ = _runner(driver)
    runner.run("RETURN 1", {})
    assert len(driver.session_kwargs) == 1
    assert driver.session_kwargs[0].get("database") is None


def test_session_with_database_name() -> None:
    driver = FakeDriver()
    runner, _ = _runner(driver, database="graph")
    runner.run("RETURN 1", {})
    runner.run("RETURN 2", {}, autocommit=True)
    assert len(driver.session_kwargs) == 2
    assert all(kwargs.get("database") == "graph" for kwargs in driver.session_kwargs)


# --- lifecycle -------------------------------------------------------------


def test_close_closes_driver_once() -> None:
    driver = FakeDriver()
    runner, factory = _runner(driver)
    runner.close()
    assert driver.close_calls == 1
    assert len(factory.calls) == 1


def test_context_manager_returns_runner_and_closes_once() -> None:
    driver = FakeDriver()
    runner, _ = _runner(driver)
    with runner as entered:
        assert entered is runner
        assert driver.close_calls == 0
    assert driver.close_calls == 1


def test_context_manager_closes_on_exception() -> None:
    driver = FakeDriver()
    runner, _ = _runner(driver)
    with pytest.raises(RuntimeError), runner:
        raise RuntimeError("boom")
    assert driver.close_calls == 1


# --- RunResult -------------------------------------------------------------


def test_run_result_is_frozen_dataclass_with_default_db_hits() -> None:
    assert dataclasses.is_dataclass(RunResult)
    result = RunResult(rows=[{"a": 1}])
    assert result.db_hits is None
    assert result.rows == [{"a": 1}]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.db_hits = 5  # type: ignore[misc]
    assert RunResult(rows=[], db_hits=4).db_hits == 4
