"""Acceptance tests for GS-18: graph_health of memini.core.graph.health.

Spec: GS-18 (src/memini/core/graph/health.py) and SPEC-graph-schema, decision "Graph health:
graph_health(scope) is read-only and returns a score from 0 to 100 plus every problem found,
with its severity and a count", test "a fake graph with one planted problem of each kind is
reported with that kind, count and severity; a clean graph scores 100". No database here.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from memini.core import schema as schema_mod
from memini.core.graph import AccessContext, AccessDenied, RunResult, Statement
from memini.core.graph.health import (
    BUILTIN_SEVERITY,
    DEFAULT_WEIGHTS,
    KINDS,
    SEVERITIES,
    HealthReport,
    Problem,
    graph_health,
    score,
)
from memini.core.graph.session import GraphSession

SCOPE = "scope-qx-417"
OTHER_SCOPE = "scope-other-88"
_WRITE_WORDS = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DETACH)\b", re.IGNORECASE)


def _prop(rule: str = "a rule") -> dict[str, Any]:
    return {"layer": "core", "writer": "ingest.register", "readers": ["query.*"], "rule": rule}


def _rel(sources: str, targets: str, rule: str, **extra: object) -> dict[str, Any]:
    return {
        "from": sources,
        "to": targets,
        "layer": "core",
        "writer": "ingest.register",
        "readers": ["query.scope"],
        "rule": rule,
        **extra,
    }


def _schema() -> schema_mod.Schema:
    return schema_mod.from_dict(
        {
            "common": {
                "nodes": {
                    "scope": _prop("scoped"),
                    "created_at": _prop("audit"),
                    "updated_at": _prop("audit"),
                    "updated_by": _prop("audit"),
                },
                "relationships": {"created_at": _prop("audit")},
            },
            "nodes": {
                "Document": {
                    "rule": "a source file",
                    "properties": {"id": _prop("identity"), "title": _prop("a title")},
                },
                "Domain": {
                    "rule": "a vocabulary value",
                    "properties": {"name": _prop("vocabulary value")},
                },
            },
            "relationships": {
                "R1": _rel(
                    "Document",
                    "Domain",
                    "new link to a domain",
                    same_fact_as="R2",
                    migrating_to="R2 by module ingest-units",
                ),
                "R2": _rel("Document", "Domain", "old link to a domain"),
            },
        }
    )


def _ctx(read: frozenset[str] = frozenset({SCOPE})) -> AccessContext:
    return AccessContext(
        principal="tester", read_scopes=read, write_scopes=frozenset(), actor="test-run"
    )


def _kind_of(stmt: Statement) -> str:
    assert stmt.name.startswith("graph_health.")
    return stmt.name[len("graph_health.") :]


class FakeSession:
    """Any object with ``ctx`` and ``run(stmt)``; returns a count chosen per statement."""

    def __init__(
        self,
        value: Callable[[Statement], int] = lambda _s: 0,
        ctx: AccessContext | None = None,
    ) -> None:
        self.ctx = ctx if ctx is not None else _ctx()
        self.value = value
        self.runs: list[Statement] = []

    def run(self, stmt: Statement) -> list[dict[str, Any]]:
        self.runs.append(stmt)
        n = self.value(stmt)
        return [{"count": n, "duplicates": n}]


def _statements() -> list[Statement]:
    session = FakeSession()
    graph_health(session, _schema(), SCOPE)  # type: ignore[arg-type]
    return session.runs


# --- clean graph -------------------------------------------------------------------------


def test_clean_graph_scores_100_with_no_problems() -> None:
    report = graph_health(FakeSession(), _schema(), SCOPE)  # type: ignore[arg-type]
    assert isinstance(report, HealthReport)
    assert report.scope == SCOPE
    assert report.score == 100
    assert report.problems == ()
    assert dict(report.stats) == {}


# --- the statements ----------------------------------------------------------------------


def test_every_statement_is_a_scoped_parameterized_read() -> None:
    stmts = _statements()
    assert stmts
    for stmt in stmts:
        assert isinstance(stmt, Statement)
        assert stmt.kind == "read"
        assert stmt.scope == SCOPE
        assert stmt.params["scope"] == SCOPE
        assert "$scope" in stmt.cypher
        assert SCOPE not in stmt.cypher


def test_every_statement_is_named_after_one_of_the_five_kinds() -> None:
    stmts = _statements()
    kinds = {_kind_of(s) for s in stmts}
    assert kinds <= set(KINDS)
    assert kinds == set(KINDS)


def test_every_statement_has_no_write_word() -> None:
    for stmt in _statements():
        assert not _WRITE_WORDS.search(stmt.cypher), stmt.cypher


def test_every_statement_passes_the_real_session_checks() -> None:
    class Runner:
        uri = "bolt://localhost:7687"

        def __init__(self) -> None:
            self.cyphers: list[str] = []

        def run(
            self,
            cypher: str,
            params: Mapping[str, Any],
            *,
            profile: bool = False,
            autocommit: bool = False,
        ) -> RunResult:
            self.cyphers.append(cypher)
            return RunResult(rows=[{"count": 0, "duplicates": 0}])

    runner = Runner()
    report = graph_health(GraphSession(runner, _ctx()), _schema(), SCOPE)
    assert report.score == 100
    assert len(runner.cyphers) == len(_statements())


def test_duplicates_has_one_statement_per_duplicate_query() -> None:
    expected = schema_mod.duplicate_queries(_schema(), SCOPE)
    dups = [s for s in _statements() if _kind_of(s) == "duplicates"]
    assert expected
    assert len(dups) == len(expected)


# --- one planted problem per kind --------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_one_planted_kind_is_reported_with_its_count_and_severity(kind: str) -> None:
    n_stmts = sum(1 for s in _statements() if _kind_of(s) == kind)
    session = FakeSession(lambda s: 1 if _kind_of(s) == kind else 0)
    report = graph_health(session, _schema(), SCOPE)  # type: ignore[arg-type]
    assert len(report.problems) == 1
    problem = report.problems[0]
    assert problem.kind == kind
    assert problem.severity == BUILTIN_SEVERITY[kind]
    assert problem.count == n_stmts
    assert isinstance(problem.detail, str)
    assert report.score == 100 - DEFAULT_WEIGHTS[BUILTIN_SEVERITY[kind]]


def test_count_sums_over_rows() -> None:
    class MultiRow(FakeSession):
        def run(self, stmt: Statement) -> list[dict[str, Any]]:
            self.runs.append(stmt)
            if _kind_of(stmt) == "undeclared_property":
                return [{"count": 2}, {"count": 3}]
            return [{"count": 0, "duplicates": 0}]

    n_stmts = sum(1 for s in _statements() if _kind_of(s) == "undeclared_property")
    report = graph_health(MultiRow(), _schema(), SCOPE)  # type: ignore[arg-type]
    assert [(p.kind, p.count) for p in report.problems] == [("undeclared_property", 5 * n_stmts)]


def test_every_kind_planted_gives_kinds_order_and_score_18() -> None:
    report = graph_health(FakeSession(lambda _s: 1), _schema(), SCOPE)  # type: ignore[arg-type]
    assert tuple(p.kind for p in report.problems) == KINDS
    assert report.score == 18


# --- constants ---------------------------------------------------------------------------


def test_constants() -> None:
    assert SEVERITIES == ("critical", "major", "minor", "info")
    assert dict(DEFAULT_WEIGHTS) == {"critical": 30, "major": 10, "minor": 2, "info": 0}
    assert KINDS == (
        "duplicates",
        "undeclared_property",
        "declared_never_written",
        "missing_audit_fields",
        "cross_scope_relationship",
    )
    assert dict(BUILTIN_SEVERITY) == {
        "duplicates": "critical",
        "undeclared_property": "major",
        "declared_never_written": "minor",
        "missing_audit_fields": "major",
        "cross_scope_relationship": "critical",
    }


# --- score -------------------------------------------------------------------------------


def test_score_subtracts_weights() -> None:
    problems = [Problem("a", "critical", 2, "x"), Problem("b", "minor", 1, "y")]
    assert score(problems) == 68


def test_score_ignores_zero_counts() -> None:
    assert score([Problem("a", "critical", 0, "x")]) == 100
    assert score([]) == 100


def test_score_is_floored_at_zero() -> None:
    assert score([Problem(f"k{i}", "critical", 1, "x") for i in range(10)]) == 0


def test_score_uses_custom_weights() -> None:
    weights = {"critical": 50, "major": 5, "minor": 1, "info": 7}
    problems = [Problem("a", "critical", 1, "x"), Problem("b", "info", 3, "y")]
    assert score(problems, weights) == 43


# --- dataclasses -------------------------------------------------------------------------


def test_problem_rejects_unknown_severity() -> None:
    with pytest.raises(ValueError):
        Problem("a", "fatal", 1, "x")


def test_problem_and_report_are_frozen() -> None:
    problem = Problem("a", "major", 1, "x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        problem.count = 2  # type: ignore[misc]
    report = HealthReport(scope=SCOPE, score=90, problems=(problem,))
    assert dict(report.stats) == {}
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.score = 0  # type: ignore[misc]


# --- extra checks ------------------------------------------------------------------------


def _custom_stmt(scope: str, kind: str = "read") -> Statement:
    return Statement(
        name="graph_health.custom_kind",
        cypher="MATCH (n:Document) WHERE n.scope = $scope RETURN count(n) AS count",
        params={"scope": scope},
        scope=scope,
        kind=kind,  # type: ignore[arg-type]
    )


def test_extra_check_problem_comes_last() -> None:
    seen: list[tuple[object, str]] = []

    def check(schema: schema_mod.Schema, scope: str) -> list[tuple[Statement, str, str]]:
        seen.append((schema, scope))
        return [(_custom_stmt(scope), "custom_kind", "info")]

    def value(stmt: Statement) -> int:
        return 4 if stmt.name == "graph_health.custom_kind" else 1

    schema = _schema()
    report = graph_health(
        FakeSession(value),  # type: ignore[arg-type]
        schema,
        SCOPE,
        extra_checks=[check],
    )
    assert seen == [(schema, SCOPE)]
    last = report.problems[-1]
    assert (last.kind, last.severity, last.count) == ("custom_kind", "info", 4)
    assert tuple(p.kind for p in report.problems[:-1]) == KINDS


def test_extra_info_problem_keeps_score_100() -> None:
    def check(schema: schema_mod.Schema, scope: str) -> list[tuple[Statement, str, str]]:
        return [(_custom_stmt(scope), "custom_kind", "info")]

    def value(stmt: Statement) -> int:
        return 4 if stmt.name == "graph_health.custom_kind" else 0

    report = graph_health(
        FakeSession(value),  # type: ignore[arg-type]
        _schema(),
        SCOPE,
        extra_checks=[check],
    )
    assert [(p.kind, p.severity, p.count) for p in report.problems] == [("custom_kind", "info", 4)]
    assert report.score == 100


def test_extra_check_with_write_statement_is_refused_before_any_run() -> None:
    def check(schema: schema_mod.Schema, scope: str) -> list[tuple[Statement, str, str]]:
        return [(_custom_stmt(scope, kind="write"), "custom_kind", "info")]

    session = FakeSession()
    with pytest.raises(ValueError):
        graph_health(session, _schema(), SCOPE, extra_checks=[check])  # type: ignore[arg-type]
    assert session.runs == []


def test_extra_check_with_unknown_severity_is_refused_before_any_run() -> None:
    def check(schema: schema_mod.Schema, scope: str) -> list[tuple[Statement, str, str]]:
        return [(_custom_stmt(scope), "custom_kind", "fatal")]

    session = FakeSession()
    with pytest.raises(ValueError):
        graph_health(session, _schema(), SCOPE, extra_checks=[check])  # type: ignore[arg-type]
    assert session.runs == []


# --- access ------------------------------------------------------------------------------


def test_scope_not_readable_is_denied_before_any_run() -> None:
    session = FakeSession(ctx=_ctx(frozenset({OTHER_SCOPE})))
    with pytest.raises(AccessDenied):
        graph_health(session, _schema(), SCOPE)  # type: ignore[arg-type]
    assert session.runs == []


def test_session_without_context_is_denied_before_any_run() -> None:
    class Runner:
        uri = "bolt://localhost:7687"

        def __init__(self) -> None:
            self.calls = 0

        def run(self, cypher: str, params: Mapping[str, Any], **_: object) -> RunResult:
            self.calls += 1
            return RunResult(rows=[])

    runner = Runner()
    with pytest.raises(AccessDenied):
        graph_health(GraphSession(runner, None), _schema(), SCOPE)
    assert runner.calls == 0
