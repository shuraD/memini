"""Acceptance tests for GS-05: shared types, errors, protocols and registry of memini.core.graph."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import pytest

from memini.core import graph
from memini.core.graph import types as gtypes
from memini.core.graph.types import (
    QUERY_REGISTRY,
    AccessContext,
    AccessDenied,
    AuditFieldError,
    ConfirmationRequired,
    GraphError,
    IsolationError,
    PerfRecorder,
    Runner,
    RunResult,
    Statement,
    UnlabeledMatchError,
    UnsafeOperation,
    registered_query,
    require_confirmation,
)

PUBLIC_NAMES = (
    "Statement",
    "AccessContext",
    "GraphError",
    "AccessDenied",
    "IsolationError",
    "UnlabeledMatchError",
    "AuditFieldError",
    "ConfirmationRequired",
    "UnsafeOperation",
    "RunResult",
    "Runner",
    "PerfRecorder",
    "Clock",
    "require_confirmation",
    "QUERY_REGISTRY",
    "registered_query",
)


def _statement() -> Statement:
    return Statement(
        name="q.test",
        cypher="MATCH (n:Thing) RETURN n",
        params={"x": 1},
        scope="A",
        kind="read",
    )


def _context() -> AccessContext:
    return AccessContext(
        principal="user-1",
        read_scopes=frozenset({"A", "B"}),
        write_scopes=frozenset({"A"}),
        actor="ingest.extract@run-7",
    )


# --- frozen dataclasses -------------------------------------------------------


def test_statement_is_frozen() -> None:
    stmt = _statement()
    with pytest.raises(dataclasses.FrozenInstanceError):
        stmt.name = "other"  # type: ignore[misc]


def test_access_context_is_frozen() -> None:
    ctx = _context()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.actor = "other"  # type: ignore[misc]


def test_run_result_is_frozen() -> None:
    result = RunResult(rows=[{"a": 1}], db_hits=3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.db_hits = 4  # type: ignore[misc]


def test_statement_holds_given_fields() -> None:
    stmt = _statement()
    assert stmt.name == "q.test"
    assert stmt.cypher == "MATCH (n:Thing) RETURN n"
    assert stmt.params == {"x": 1}
    assert stmt.scope == "A"
    assert stmt.kind == "read"


def test_run_result_db_hits_defaults_to_none() -> None:
    result = RunResult(rows=[{"a": 1}, {"a": 2}])
    assert result.db_hits is None
    assert result.rows == [{"a": 1}, {"a": 2}]


# --- AccessContext validation -------------------------------------------------


def test_valid_access_context_holds_values() -> None:
    ctx = _context()
    assert ctx.principal == "user-1"
    assert ctx.read_scopes == frozenset({"A", "B"})
    assert ctx.write_scopes == frozenset({"A"})
    assert ctx.actor == "ingest.extract@run-7"


def test_access_context_equal_scopes_and_empty_write_are_valid() -> None:
    same = AccessContext(
        principal="p", read_scopes=frozenset({"A"}), write_scopes=frozenset({"A"}), actor="a"
    )
    assert same.write_scopes == frozenset({"A"})
    empty = AccessContext(
        principal="p", read_scopes=frozenset({"A"}), write_scopes=frozenset(), actor="a"
    )
    assert empty.write_scopes == frozenset()


def test_access_context_write_not_subset_of_read_raises() -> None:
    with pytest.raises(ValueError):
        AccessContext(
            principal="p",
            read_scopes=frozenset({"A"}),
            write_scopes=frozenset({"B"}),
            actor="a",
        )


def test_access_context_empty_actor_raises() -> None:
    with pytest.raises(ValueError):
        AccessContext(
            principal="p",
            read_scopes=frozenset({"A"}),
            write_scopes=frozenset({"A"}),
            actor="",
        )


def test_access_context_empty_principal_raises() -> None:
    with pytest.raises(ValueError):
        AccessContext(
            principal="",
            read_scopes=frozenset({"A"}),
            write_scopes=frozenset({"A"}),
            actor="a",
        )


# --- errors -------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [
        AccessDenied,
        IsolationError,
        UnlabeledMatchError,
        AuditFieldError,
        ConfirmationRequired,
        UnsafeOperation,
    ],
)
def test_error_is_graph_error(cls: type[Exception]) -> None:
    assert issubclass(cls, GraphError)


def test_graph_error_is_exception() -> None:
    assert issubclass(GraphError, Exception)


def test_unlabeled_match_error_is_value_error() -> None:
    assert issubclass(UnlabeledMatchError, ValueError)
    with pytest.raises(ValueError):
        raise UnlabeledMatchError("no label")


# --- require_confirmation -----------------------------------------------------


@pytest.mark.parametrize("confirm", [None, "a", "", "A "])
def test_require_confirmation_rejects_mismatch(confirm: str | None) -> None:
    with pytest.raises(ConfirmationRequired):
        require_confirmation(confirm, "A")


def test_require_confirmation_accepts_exact_scope() -> None:
    assert require_confirmation("A", "A") is None


# --- registry -----------------------------------------------------------------


def test_registered_query_returns_same_function_and_registers_it() -> None:
    def q(**kwargs: Any) -> Statement:
        return _statement()

    decorated = registered_query("gs05.test.registers")(q)
    assert decorated is q
    assert QUERY_REGISTRY["gs05.test.registers"] is q
    assert "gs05.test.registers" in QUERY_REGISTRY


def test_registered_query_duplicate_name_raises_and_keeps_first() -> None:
    def first(**kwargs: Any) -> Statement:
        return _statement()

    def second(**kwargs: Any) -> Statement:
        return _statement()

    registered_query("gs05.test.duplicate")(first)
    with pytest.raises(ValueError):
        registered_query("gs05.test.duplicate")(second)
    assert QUERY_REGISTRY["gs05.test.duplicate"] is first


def test_query_registry_is_read_only() -> None:
    def q(**kwargs: Any) -> Statement:
        return _statement()

    assert isinstance(QUERY_REGISTRY, Mapping)
    with pytest.raises(TypeError):
        QUERY_REGISTRY["gs05.test.assign"] = q  # type: ignore[index]
    assert "gs05.test.assign" not in QUERY_REGISTRY


# --- protocols ----------------------------------------------------------------


class _GoodRunner:
    uri = "bolt://localhost:7687"

    def run(
        self,
        cypher: str,
        params: Mapping[str, Any],
        *,
        profile: bool = False,
        autocommit: bool = False,
    ) -> RunResult:
        return RunResult(rows=[])


class _NoRunRunner:
    uri = "bolt://localhost:7687"


class _GoodRecorder:
    def record(
        self, name: str, scope: str | None, seconds: float, rows: int, db_hits: int | None
    ) -> None:
        return None


class _NoRecord:
    pass


def test_runner_protocol_runtime_check() -> None:
    assert isinstance(_GoodRunner(), Runner)
    assert not isinstance(_NoRunRunner(), Runner)


def test_perf_recorder_protocol_runtime_check() -> None:
    assert isinstance(_GoodRecorder(), PerfRecorder)
    assert not isinstance(_NoRecord(), PerfRecorder)


# --- re-exports ---------------------------------------------------------------


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_public_name_reexported_as_same_object(name: str) -> None:
    assert hasattr(gtypes, name)
    assert hasattr(graph, name)
    assert getattr(graph, name) is getattr(gtypes, name)
