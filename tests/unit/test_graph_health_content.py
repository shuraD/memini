"""Acceptance tests for GS-19: content checks and degree statistics of graph health.

Spec: GS-19 (src/memini/core/graph/health_content.py). Read-only content checks (orphans,
missing or wrong-size embeddings, missing or offline indexes, broken SUPERSEDES chains,
items awaiting purge, stale communities) added to GS-18's graph_health, plus a
degree-statistics statement whose rows fill HealthReport.stats. No database here.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from memini.core import schema as schema_mod
from memini.core.graph import AccessContext, Statement
from memini.core.graph.checks import require_labeled
from memini.core.graph.health import DEFAULT_WEIGHTS, KINDS, HealthReport, graph_health
from memini.core.graph.health_content import (
    CONTENT_SEVERITY,
    content_checks,
    content_health,
    degree_stats,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_FILE = ROOT / "config" / "graph_schema.yaml"
SCOPE = "scope-hc-519"
RULES: tuple[tuple[str, str], ...] = (("Unit", "HAS_UNIT"), ("Entity", "HAS_ENTITY"))
DIMS = 768
_WRITE_WORDS = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DETACH)\b", re.IGNORECASE)

CONTENT_KINDS = (
    "orphan:Unit",
    "orphan:Entity",
    "missing_embedding",
    "wrong_vector_size",
    "missing_index",
    "offline_index",
    "broken_supersedes",
    "awaiting_purge",
    "stale_community",
)


def _schema() -> schema_mod.Schema:
    return schema_mod.load(SCHEMA_FILE)


def _ctx() -> AccessContext:
    return AccessContext(
        principal="tester", read_scopes=frozenset({SCOPE}), write_scopes=frozenset(), actor="t"
    )


def _kind_of(stmt: Statement) -> str:
    assert stmt.name.startswith("graph_health.")
    return stmt.name[len("graph_health.") :]


class FakeSession:
    """``ctx`` plus ``run(stmt)``; counts chosen per statement, degree rows given."""

    def __init__(
        self,
        value: Callable[[Statement], int] = lambda _s: 0,
        degree_rows: Sequence[dict[str, Any]] = (),
    ) -> None:
        self.ctx = _ctx()
        self.value = value
        self.degree_rows = list(degree_rows)
        self.runs: list[Statement] = []

    def run(self, stmt: Statement) -> list[dict[str, Any]]:
        self.runs.append(stmt)
        if stmt.name == "graph_health.degree":
            return [dict(r) for r in self.degree_rows]
        n = self.value(stmt)
        return [{"count": n, "duplicates": n}]


def _content_statements(
    vector_dimensions: int | None = DIMS, schema: schema_mod.Schema | None = None
) -> list[tuple[Statement, str, str]]:
    s = schema if schema is not None else _schema()
    out: list[tuple[Statement, str, str]] = []
    for check in content_checks(orphan_rules=RULES, vector_dimensions=vector_dimensions):
        out.extend(check(s, SCOPE))
    return out


def _health(session: FakeSession, vector_dimensions: int | None = DIMS) -> HealthReport:
    return content_health(
        session,  # type: ignore[arg-type]
        _schema(),
        SCOPE,
        orphan_rules=RULES,
        vector_dimensions=vector_dimensions,
    )


# --- severities --------------------------------------------------------------------------


def test_content_severity_table() -> None:
    assert dict(CONTENT_SEVERITY) == {
        "orphan": "major",
        "missing_embedding": "major",
        "wrong_vector_size": "critical",
        "missing_index": "major",
        "offline_index": "critical",
        "broken_supersedes": "major",
        "awaiting_purge": "info",
        "stale_community": "minor",
    }


# --- clean graph -------------------------------------------------------------------------


def test_clean_graph_scores_100_with_no_problems() -> None:
    report = _health(FakeSession())
    assert isinstance(report, HealthReport)
    assert report.scope == SCOPE
    assert report.score == 100
    assert report.problems == ()


def test_every_content_kind_is_produced_by_content_checks() -> None:
    kinds = {kind for _, kind, _ in _content_statements()}
    assert kinds == set(CONTENT_KINDS)


def test_content_triples_carry_their_kind_name_and_severity() -> None:
    for stmt, kind, severity in _content_statements():
        assert isinstance(stmt, Statement)
        assert stmt.name == f"graph_health.{kind}"
        assert severity == CONTENT_SEVERITY[kind.split(":")[0]]


# --- one planted kind --------------------------------------------------------------------


@pytest.mark.parametrize("kind", CONTENT_KINDS)
def test_one_planted_content_kind_is_reported(kind: str) -> None:
    n_stmts = sum(1 for s, k, _ in _content_statements() if k == kind)
    assert n_stmts >= 1
    session = FakeSession(lambda s: 3 if _kind_of(s) == kind else 0)
    report = _health(session)
    assert len(report.problems) == 1
    problem = report.problems[0]
    severity = CONTENT_SEVERITY[kind.split(":", maxsplit=1)[0]]
    assert problem.kind == kind
    assert problem.severity == severity
    assert problem.count == 3 * n_stmts
    assert report.score == 100 - DEFAULT_WEIGHTS[severity]


def test_awaiting_purge_alone_keeps_score_100_but_is_listed_as_info() -> None:
    session = FakeSession(lambda s: 5 if _kind_of(s) == "awaiting_purge" else 0)
    report = _health(session)
    assert report.score == 100
    assert [p.kind for p in report.problems] == ["awaiting_purge"]
    assert report.problems[0].severity == "info"
    n_stmts = sum(1 for _, k, _ in _content_statements() if k == "awaiting_purge")
    assert report.problems[0].count == 5 * n_stmts


# --- wiring into graph_health ------------------------------------------------------------


def test_graph_health_runs_content_statements_alongside_builtin_ones() -> None:
    session = FakeSession()
    report = graph_health(
        session,  # type: ignore[arg-type]
        _schema(),
        SCOPE,
        extra_checks=content_checks(orphan_rules=RULES, vector_dimensions=DIMS),
    )
    assert report.score == 100
    kinds = {_kind_of(s) for s in session.runs}
    assert set(CONTENT_KINDS) <= kinds
    assert kinds & set(KINDS)


def test_content_health_runs_degree_statement_exactly_once() -> None:
    session = FakeSession()
    _health(session)
    names = [s.name for s in session.runs]
    assert names.count("graph_health.degree") == 1
    assert {f"graph_health.{k}" for k in CONTENT_KINDS} <= set(names)


# --- vector dimensions -------------------------------------------------------------------


def test_no_wrong_vector_size_statement_without_dimensions() -> None:
    kinds = [k for _, k, _ in _content_statements(vector_dimensions=None)]
    assert "wrong_vector_size" not in kinds
    assert "missing_embedding" in kinds


def test_wrong_vector_size_statements_bind_the_dimensions() -> None:
    stmts = [s for s, k, _ in _content_statements(vector_dimensions=DIMS)]
    sized = [s for s in stmts if _kind_of(s) == "wrong_vector_size"]
    assert sized
    for stmt in sized:
        assert stmt.params["dims"] == DIMS
        assert "$dims" in stmt.cypher


def test_content_health_without_dimensions_runs_no_wrong_vector_size() -> None:
    session = FakeSession()
    report = _health(session, vector_dimensions=None)
    assert report.score == 100
    assert all(_kind_of(s) != "wrong_vector_size" for s in session.runs)


# --- degree statistics -------------------------------------------------------------------


def test_degree_rows_fill_report_stats() -> None:
    rows = [{"type": "RELATES", "avg": 4.2, "p50": 4.0, "max": 9.0}]
    report = _health(FakeSession(degree_rows=rows))
    assert report.stats["degree.RELATES.avg"] == 4.2
    assert report.stats["degree.RELATES.p50"] == 4.0
    assert report.stats["degree.RELATES.max"] == 9.0
    assert report.score == 100
    assert report.problems == ()


def test_degree_stats_are_floats_one_triple_per_row() -> None:
    rows = [
        {"type": "RELATES", "avg": 2, "p50": 1, "max": 7},
        {"type": "HAS_UNIT", "avg": 1.5, "p50": 1.0, "max": 3.0},
    ]
    report = _health(FakeSession(degree_rows=rows))
    assert dict(report.stats) == {
        "degree.RELATES.avg": 2.0,
        "degree.RELATES.p50": 1.0,
        "degree.RELATES.max": 7.0,
        "degree.HAS_UNIT.avg": 1.5,
        "degree.HAS_UNIT.p50": 1.0,
        "degree.HAS_UNIT.max": 3.0,
    }
    assert all(isinstance(v, float) for v in report.stats.values())


def test_problems_and_score_equal_graph_health() -> None:
    def value(s: Statement) -> int:
        return 2 if _kind_of(s) in ("orphan:Unit", "stale_community", "duplicates") else 0

    rows = [{"type": "RELATES", "avg": 1.0, "p50": 1.0, "max": 1.0}]
    wrapped = _health(FakeSession(value, rows))
    plain = graph_health(
        FakeSession(value),  # type: ignore[arg-type]
        _schema(),
        SCOPE,
        extra_checks=content_checks(orphan_rules=RULES, vector_dimensions=DIMS),
    )
    assert wrapped.problems == plain.problems
    assert wrapped.score == plain.score
    assert wrapped.scope == SCOPE


def test_degree_statement_shape() -> None:
    stmt = degree_stats(_schema(), SCOPE)
    assert isinstance(stmt, Statement)
    assert stmt.name == "graph_health.degree"
    assert stmt.kind == "read"
    assert stmt.scope == SCOPE
    assert dict(stmt.params) == {"scope": SCOPE}
    assert not _WRITE_WORDS.search(stmt.cypher), stmt.cypher
    assert "MATCH" in stmt.cypher
    require_labeled(stmt.cypher)
    assert SCOPE not in stmt.cypher


# --- invalid orphan rules ----------------------------------------------------------------


@pytest.mark.parametrize("rule", [("Nope", "HAS_UNIT"), ("Unit", "NOPE")])
def test_undeclared_orphan_rule_raises_before_any_run(rule: tuple[str, str]) -> None:
    session = FakeSession()
    with pytest.raises(ValueError):
        content_health(
            session,  # type: ignore[arg-type]
            _schema(),
            SCOPE,
            orphan_rules=[rule],
            vector_dimensions=DIMS,
        )
    assert session.runs == []


@pytest.mark.parametrize("rule", [("Nope", "HAS_UNIT"), ("Unit", "NOPE")])
def test_undeclared_orphan_rule_raises_in_graph_health(rule: tuple[str, str]) -> None:
    session = FakeSession()
    checks = content_checks(orphan_rules=[rule], vector_dimensions=DIMS)
    with pytest.raises(ValueError):
        graph_health(session, _schema(), SCOPE, extra_checks=checks)  # type: ignore[arg-type]
    assert session.runs == []


# --- statement shape ---------------------------------------------------------------------


def test_every_content_statement_is_a_scoped_labeled_read() -> None:
    stmts = _content_statements()
    assert stmts
    for stmt, _, _ in stmts:
        assert stmt.kind == "read"
        assert stmt.scope == SCOPE
        assert stmt.params["scope"] == SCOPE
        assert not _WRITE_WORDS.search(stmt.cypher), stmt.cypher
        assert SCOPE not in stmt.cypher
        require_labeled(stmt.cypher)


def test_content_statements_return_a_count_column() -> None:
    for stmt, _, _ in _content_statements():
        assert re.search(r"\bAS\s+count\b", stmt.cypher), stmt.cypher


def test_scoped_kinds_use_scope_parameter() -> None:
    for stmt, kind, _ in _content_statements():
        if kind.startswith("orphan:") or kind in (
            "missing_embedding",
            "wrong_vector_size",
            "awaiting_purge",
            "stale_community",
            "broken_supersedes",
        ):
            assert "$scope" in stmt.cypher, stmt.cypher


def test_orphan_statements_name_label_and_relationship() -> None:
    by_kind = {k: s for s, k, _ in _content_statements() if k.startswith("orphan:")}
    assert set(by_kind) == {"orphan:Unit", "orphan:Entity"}
    assert ":Unit" in by_kind["orphan:Unit"].cypher
    assert "HAS_UNIT" in by_kind["orphan:Unit"].cypher
    assert ":Entity" in by_kind["orphan:Entity"].cypher
    assert "HAS_ENTITY" in by_kind["orphan:Entity"].cypher


def test_missing_embedding_statements_name_vector_indexed_labels() -> None:
    schema = _schema()
    vec_labels = {label for label, node in schema.nodes.items() if node.vector_indexes}
    assert vec_labels == {"Entity", "Community"}
    stmts = [s for s, k, _ in _content_statements() if k == "missing_embedding"]
    pairs = sum(len(node.vector_indexes) for node in schema.nodes.values())
    assert len(stmts) == pairs
    text = " ".join(s.cypher for s in stmts)
    assert ":Entity" in text
    assert ":Community" in text
    sized = [s for s, k, _ in _content_statements() if k == "wrong_vector_size"]
    assert len(sized) == pairs


def test_index_statements_carry_declared_names() -> None:
    for kind in ("missing_index", "offline_index"):
        stmts = [s for s, k, _ in _content_statements() if k == kind]
        assert stmts
        for stmt in stmts:
            assert "SHOW INDEXES" in stmt.cypher.upper()
            assert stmt.params["scope"] == SCOPE
    missing = [s for s, k, _ in _content_statements() if k == "missing_index"]
    for stmt in missing:
        names = stmt.params["names"]
        assert isinstance(names, list)
        assert names
        assert all(isinstance(n, str) for n in names)


def test_supersedes_and_stale_community_name_their_types() -> None:
    stmts = _content_statements()
    sup = [s for s, k, _ in stmts if k == "broken_supersedes"]
    stale = [s for s, k, _ in stmts if k == "stale_community"]
    assert sup
    assert stale
    assert all("SUPERSEDES" in s.cypher for s in sup)
    for stmt in stale:
        assert ":Community" in stmt.cypher
        assert "MEMBER_OF" in stmt.cypher


def test_awaiting_purge_has_one_statement_per_label() -> None:
    schema = _schema()
    stmts = [s for s, k, _ in _content_statements() if k == "awaiting_purge"]
    assert len(stmts) == len(schema.nodes)
    for label in schema.nodes:
        assert any(f":{label}" in s.cypher for s in stmts), label
    for stmt in stmts:
        assert "deleted" in stmt.cypher or "deleted" in stmt.params.values()


# --- schemas without SUPERSEDES / Community ----------------------------------------------


def _prop(rule: str = "a rule") -> dict[str, Any]:
    return {"layer": "core", "writer": "ingest.register", "readers": ["query.*"], "rule": rule}


def _small_schema() -> schema_mod.Schema:
    return schema_mod.from_dict(
        {
            "common": {
                "nodes": {
                    "scope": _prop("scoped"),
                    "status": _prop("lifecycle"),
                    "created_at": _prop("audit"),
                    "updated_at": _prop("audit"),
                    "updated_by": _prop("audit"),
                },
                "relationships": {"created_at": _prop("audit")},
            },
            "nodes": {
                "Document": {
                    "rule": "a source file",
                    "properties": {"id": _prop("identity")},
                    "indexes": [["scope", "id"]],
                },
            },
            "relationships": {},
        }
    )


def test_supersedes_and_community_checks_only_when_declared() -> None:
    small = _small_schema()
    triples: list[tuple[Statement, str, str]] = []
    for check in content_checks(orphan_rules=(), vector_dimensions=DIMS):
        triples.extend(check(small, SCOPE))
    kinds = {k for _, k, _ in triples}
    assert "broken_supersedes" not in kinds
    assert "stale_community" not in kinds
    assert "awaiting_purge" in kinds
    for stmt, _, _ in triples:
        assert stmt.kind == "read"
        assert stmt.params["scope"] == SCOPE
        assert not _WRITE_WORDS.search(stmt.cypher), stmt.cypher
        require_labeled(stmt.cypher)
