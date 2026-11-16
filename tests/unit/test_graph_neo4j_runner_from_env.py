"""Acceptance tests for AR-22: Neo4jRunner.from_env and verify_connectivity."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memini.infra.graph import neo4j_runner
from memini.infra.graph.guard import UnsafeDatabaseError
from memini.infra.graph.neo4j_runner import DatabaseUnreachableError, Neo4jRunner

DEFAULT_URI = "bolt://localhost:7687"
LOCAL_URI = "bolt://127.0.0.1:7688"
REMOTE_URI = "bolt://db.example.com:7687"
USER = "neo4j"
FAKE_KEY = "test-key-pw"
REPO_ROOT = Path(__file__).resolve().parents[2]


class ServiceUnavailableError(Exception):
    """Stand-in for neo4j.exceptions.ServiceUnavailable."""


class FakeResult:
    def __iter__(self) -> Iterator[Any]:
        return iter([])

    def consume(self) -> Any:
        class _Summary:
            profile = None

        return _Summary()


class FakeTx:
    def run(self, query: str, *args: Any, **kwargs: Any) -> FakeResult:
        return FakeResult()


class FakeSession:
    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def close(self) -> None:
        return None

    def run(self, query: str, *args: Any, **kwargs: Any) -> FakeResult:
        return FakeResult()

    def execute_write(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return fn(FakeTx(), *args, **kwargs)

    def execute_read(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return fn(FakeTx(), *args, **kwargs)


class FakeDriver:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.events: list[str] = []
        self.session_kwargs: list[dict[str, Any]] = []

    def verify_connectivity(self, *args: Any, **kwargs: Any) -> None:
        self.events.append("verify")
        if self.error is not None:
            raise self.error

    def session(self, **kwargs: Any) -> FakeSession:
        self.events.append("session")
        self.session_kwargs.append(kwargs)
        return FakeSession()

    def close(self) -> None:
        self.events.append("close")

    def count(self, event: str) -> int:
        return self.events.count(event)


class FakeFactory:
    def __init__(self, driver: FakeDriver | None = None) -> None:
        self.driver = driver if driver is not None else FakeDriver()
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> FakeDriver:
        self.calls.append((args, kwargs))
        return self.driver


def _session_database(kwargs: dict[str, Any]) -> str | None:
    return kwargs.get("database")


# --- env reading ---------------------------------------------------------


def test_factory_called_once_with_uri_and_auth_from_env() -> None:
    factory = FakeFactory()
    env = {"NEO4J_URI": LOCAL_URI, "NEO4J_USER": USER, "NEO4J_PASSWORD": FAKE_KEY}
    runner = Neo4jRunner.from_env(env, driver_factory=factory)
    assert factory.calls == [((LOCAL_URI,), {"auth": (USER, FAKE_KEY)})]
    assert runner.uri == LOCAL_URI


def test_values_are_stripped() -> None:
    factory = FakeFactory()
    env = {
        "NEO4J_URI": f"  {LOCAL_URI}\n",
        "NEO4J_USER": f"  {USER} ",
        "NEO4J_PASSWORD": f"\t{FAKE_KEY}  ",
        "NEO4J_DATABASE": "  graphdb  ",
    }
    runner = Neo4jRunner.from_env(env, driver_factory=factory)
    assert factory.calls == [((LOCAL_URI,), {"auth": (USER, FAKE_KEY)})]
    assert runner.uri == LOCAL_URI
    runner.run("RETURN 1", {})
    assert _session_database(factory.driver.session_kwargs[-1]) == "graphdb"


@pytest.mark.parametrize("env", [{}, {"NEO4J_URI": ""}, {"NEO4J_URI": "   "}])
def test_missing_or_blank_uri_defaults_to_localhost(env: dict[str, str]) -> None:
    factory = FakeFactory()
    runner = Neo4jRunner.from_env(env, driver_factory=factory)
    assert runner.uri == DEFAULT_URI
    assert len(factory.calls) == 1
    assert factory.calls[0][0] == (DEFAULT_URI,)


@pytest.mark.parametrize(
    "env",
    [
        {"NEO4J_URI": LOCAL_URI},
        {"NEO4J_URI": LOCAL_URI, "NEO4J_USER": USER},
        {"NEO4J_URI": LOCAL_URI, "NEO4J_PASSWORD": FAKE_KEY},
        {"NEO4J_URI": LOCAL_URI, "NEO4J_USER": "  ", "NEO4J_PASSWORD": FAKE_KEY},
        {"NEO4J_URI": LOCAL_URI, "NEO4J_USER": USER, "NEO4J_PASSWORD": " "},
    ],
)
def test_missing_or_blank_credentials_give_auth_none(env: dict[str, str]) -> None:
    factory = FakeFactory()
    Neo4jRunner.from_env(env, driver_factory=factory)
    assert factory.calls == [((LOCAL_URI,), {"auth": None})]


def test_explicit_empty_env_ignores_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", LOCAL_URI)
    monkeypatch.setenv("NEO4J_USER", USER)
    monkeypatch.setenv("NEO4J_PASSWORD", FAKE_KEY)
    monkeypatch.setenv("NEO4J_DATABASE", "processdb")
    factory = FakeFactory()
    runner = Neo4jRunner.from_env({}, driver_factory=factory)
    assert factory.calls == [((DEFAULT_URI,), {"auth": None})]
    assert runner.uri == DEFAULT_URI
    runner.run("RETURN 1", {})
    assert _session_database(factory.driver.session_kwargs[-1]) is None


def test_explicit_env_guard_ignores_process_allow_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMINI_NEO4J_ALLOWED_HOSTS", "db.example.com")
    factory = FakeFactory()
    with pytest.raises(UnsafeDatabaseError):
        Neo4jRunner.from_env({"NEO4J_URI": REMOTE_URI}, driver_factory=factory)
    assert factory.calls == []


def test_no_env_argument_reads_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", LOCAL_URI)
    monkeypatch.setenv("NEO4J_USER", USER)
    monkeypatch.setenv("NEO4J_PASSWORD", FAKE_KEY)
    monkeypatch.setenv("NEO4J_DATABASE", "processdb")
    monkeypatch.delenv("MEMINI_PROD_NEO4J_HOST", raising=False)
    factory = FakeFactory()
    runner = Neo4jRunner.from_env(driver_factory=factory)
    assert factory.calls == [((LOCAL_URI,), {"auth": (USER, FAKE_KEY)})]
    assert runner.uri == LOCAL_URI
    runner.run("RETURN 1", {})
    assert _session_database(factory.driver.session_kwargs[-1]) == "processdb"


# --- database name -------------------------------------------------------


def test_database_name_used_for_every_session() -> None:
    factory = FakeFactory()
    env = {"NEO4J_URI": LOCAL_URI, "NEO4J_DATABASE": "graphdb"}
    runner = Neo4jRunner.from_env(env, driver_factory=factory)
    runner.run("RETURN 1", {})
    runner.run("RETURN 2", {}, autocommit=True)
    runner.run("RETURN 3", {}, profile=True)
    kwargs = factory.driver.session_kwargs
    assert len(kwargs) == 3
    assert all(_session_database(k) == "graphdb" for k in kwargs)


@pytest.mark.parametrize("env", [{}, {"NEO4J_DATABASE": ""}, {"NEO4J_DATABASE": "  "}])
def test_missing_or_blank_database_gives_default(env: dict[str, str]) -> None:
    factory = FakeFactory()
    runner = Neo4jRunner.from_env(env, driver_factory=factory)
    runner.run("RETURN 1", {})
    runner.run("RETURN 2", {}, autocommit=True)
    kwargs = factory.driver.session_kwargs
    assert len(kwargs) == 2
    assert all(_session_database(k) is None for k in kwargs)


# --- connectivity --------------------------------------------------------


def test_reachable_driver_verified_once_before_sessions_and_not_closed() -> None:
    factory = FakeFactory()
    runner = Neo4jRunner.from_env({"NEO4J_URI": LOCAL_URI}, driver_factory=factory)
    driver = factory.driver
    assert driver.events == ["verify"]
    runner.run("RETURN 1", {})
    assert driver.events == ["verify", "session"]
    assert driver.count("close") == 0


@pytest.mark.parametrize(
    "error",
    [ServiceUnavailableError("cannot reach server"), OSError("connection refused")],
)
def test_unreachable_driver_raises_and_closes(error: Exception) -> None:
    driver = FakeDriver(error=error)
    factory = FakeFactory(driver)
    env = {"NEO4J_URI": LOCAL_URI, "NEO4J_USER": USER, "NEO4J_PASSWORD": FAKE_KEY}
    with pytest.raises(DatabaseUnreachableError) as info:
        Neo4jRunner.from_env(env, driver_factory=factory)
    assert info.value.__cause__ is error
    assert driver.count("verify") == 1
    assert driver.count("close") == 1
    assert driver.count("session") == 0
    message = str(info.value)
    assert FAKE_KEY not in message
    assert LOCAL_URI in message


def test_runner_verify_connectivity_returns_none_when_reachable() -> None:
    driver = FakeDriver()
    factory = FakeFactory(driver)
    runner = Neo4jRunner(LOCAL_URI, (USER, FAKE_KEY), env={}, driver_factory=factory)
    assert runner.verify_connectivity() is None
    assert driver.count("verify") == 1
    assert driver.count("close") == 0


def test_runner_verify_connectivity_raises_when_unreachable() -> None:
    error = ServiceUnavailableError("down")
    driver = FakeDriver(error=error)
    factory = FakeFactory(driver)
    runner = Neo4jRunner(LOCAL_URI, (USER, FAKE_KEY), env={}, driver_factory=factory)
    with pytest.raises(DatabaseUnreachableError) as info:
        runner.verify_connectivity()
    assert info.value.__cause__ is error
    assert driver.count("verify") == 1
    assert FAKE_KEY not in str(info.value)
    assert LOCAL_URI in str(info.value)


# --- guard ---------------------------------------------------------------


def test_remote_host_not_allow_listed_is_refused_before_factory() -> None:
    factory = FakeFactory()
    env = {"NEO4J_URI": REMOTE_URI, "NEO4J_USER": USER, "NEO4J_PASSWORD": FAKE_KEY}
    with pytest.raises(UnsafeDatabaseError):
        Neo4jRunner.from_env(env, driver_factory=factory)
    assert factory.calls == []


def test_remote_host_allow_listed_in_env_is_accepted() -> None:
    factory = FakeFactory()
    env = {"NEO4J_URI": REMOTE_URI, "MEMINI_NEO4J_ALLOWED_HOSTS": "db.example.com"}
    runner = Neo4jRunner.from_env(env, driver_factory=factory)
    assert runner.uri == REMOTE_URI
    assert factory.calls == [((REMOTE_URI,), {"auth": None})]


# --- exports and config --------------------------------------------------


def test_error_type_and_export() -> None:
    assert issubclass(DatabaseUnreachableError, RuntimeError)
    assert "DatabaseUnreachableError" in neo4j_runner.__all__
    assert "Neo4jRunner" in neo4j_runner.__all__
    assert "RunResult" in neo4j_runner.__all__


_PASSWORD_LINE = re.compile(
    r"""^\s*["']?[\w.-]*password[\w.-]*["']?\s*[:=]\s*(.*)$""", re.IGNORECASE
)
_EMPTY_VALUES = {"", '""', "''", "null", "~", "none"}


def test_no_config_file_carries_a_password_value() -> None:
    config_dir = REPO_ROOT / "config"
    offending: list[str] = []
    for path in sorted(config_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            match = _PASSWORD_LINE.match(line)
            if match is None:
                continue
            value = match.group(1).split("#", 1)[0].strip().rstrip(",").strip()
            if value.lower() not in _EMPTY_VALUES:
                offending.append(f"{path.relative_to(REPO_ROOT)}:{number}")
    assert offending == []
