"""Acceptance tests for GS-05: shared types, errors and protocols of memini.core.graph."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import pytest

from memini.core import graph
from memini.core.graph import types as gtypes
from memini.core.graph.types import (
    GraphError,
    PerfRecorder,
    Runner,
    RunResult,
    Statement,
    UnlabeledMatchError,
)

PUBLIC_NAMES = (
    "Statement",
    "GraphError",
    "UnlabeledMatchError",
    "RunResult",
    "Runner",
    "PerfRecorder",
)


def _statement() -> Statement:
    return Statement(
        name="q.test",
        cypher="MATCH (n:Thing) RETURN n",
        params={"x": 1},
        kind="read",
    )


# --- frozen dataclasses -------------------------------------------------------


def test_statement_is_frozen() -> None:
    stmt = _statement()
    with pytest.raises(dataclasses.FrozenInstanceError):
        stmt.name = "other"  # type: ignore[misc]


def test_run_result_is_frozen() -> None:
    result = RunResult(rows=[{"a": 1}], db_hits=3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.db_hits = 4  # type: ignore[misc]


def test_statement_holds_given_fields() -> None:
    stmt = _statement()
    assert stmt.name == "q.test"
    assert stmt.cypher == "MATCH (n:Thing) RETURN n"
    assert stmt.params == {"x": 1}
    assert stmt.kind == "read"


def test_run_result_db_hits_defaults_to_none() -> None:
    result = RunResult(rows=[{"a": 1}, {"a": 2}])
    assert result.db_hits is None
    assert result.rows == [{"a": 1}, {"a": 2}]


# --- errors -------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [
        UnlabeledMatchError,
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
    def record(self, name: str, seconds: float, rows: int, db_hits: int | None) -> None:
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
