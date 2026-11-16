"""Acceptance tests for AR-03: memini.core.graph.vector (setup statements and vector search)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pytest

from memini.core import schema as schema_mod
from memini.core.graph import vector
from memini.core.graph.session import GraphSession
from memini.core.graph.types import RunResult, Statement

URI = "bolt://localhost:7687"


def _prop(rule: str = "test property") -> dict[str, Any]:
    return {"layer": "core", "writer": "test.writer", "readers": ["test.reader"], "rule": rule}


def _raw() -> dict[str, Any]:
    return {
        "nodes": {
            "Doc": {
                "rule": "a keyed label with a range index and a vector index",
                "properties": {
                    "scope": _prop(),
                    "id": _prop(),
                    "title": _prop(),
                    "embedding": _prop(),
                },
                "indexes": [["title"]],
                "vector_indexes": ["embedding"],
            },
        },
        "relationships": {},
    }


def _schema() -> schema_mod.Schema:
    return schema_mod.from_dict(_raw())


class RecordingRunner:
    """Records every call; optionally fails on the n-th call (1-based)."""

    def __init__(self, fail_on: int | None = None, error: Exception | None = None) -> None:
        self.uri = URI
        self.fail_on = fail_on
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        cypher: str,
        params: Mapping[str, Any],
        *,
        profile: bool = False,
        autocommit: bool = False,
    ) -> RunResult:
        self.calls.append(
            {"cypher": cypher, "params": params, "profile": profile, "autocommit": autocommit}
        )
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            assert self.error is not None
            raise self.error
        return RunResult(rows=[])


# --- setup_statements -------------------------------------------------------------------------


def test_setup_statements_mirror_the_schema_setup_list() -> None:
    s = _schema()
    stmts = vector.setup_statements(s, vector_dimensions=8)
    expected = schema_mod.setup_statements(s, vector_dimensions=8)
    assert len(expected) == 3
    assert [st.cypher for st in stmts] == expected
    for st in stmts:
        assert isinstance(st, Statement)
        assert st.kind == "admin"
        assert dict(st.params) == {}
        assert st.scope is None
        assert isinstance(st.name, str)
        assert st.name


def test_setup_statements_order_constraint_range_vector() -> None:
    cyphers = [st.cypher for st in vector.setup_statements(_schema(), vector_dimensions=8)]
    i_con = next(i for i, c in enumerate(cyphers) if c.startswith("CREATE CONSTRAINT"))
    i_rng = next(i for i, c in enumerate(cyphers) if c.startswith("CREATE INDEX"))
    i_vec = next(i for i, c in enumerate(cyphers) if c.startswith("CREATE VECTOR INDEX"))
    assert i_con < i_rng < i_vec
    vec = cyphers[i_vec]
    assert "cosine" in vec
    assert "`vector.dimensions`: 8" in vec


@pytest.mark.parametrize("dims", [0, -1, True, 8.0])
def test_bad_dimensions_raise_and_nothing_runs(dims: Any) -> None:
    runner = RecordingRunner()
    session = GraphSession(runner)
    with pytest.raises(ValueError):
        vector.setup_statements(_schema(), vector_dimensions=dims)
    with pytest.raises(ValueError):
        vector.ensure_indexes(session, _schema(), vector_dimensions=dims)
    assert runner.calls == []


# --- ensure_indexes ---------------------------------------------------------------------------


def test_ensure_indexes_sends_setup_cyphers_in_order_with_autocommit() -> None:
    s = _schema()
    runner = RecordingRunner()
    count = vector.ensure_indexes(GraphSession(runner), s, vector_dimensions=8)
    expected = schema_mod.setup_statements(s, vector_dimensions=8)
    assert [c["cypher"] for c in runner.calls] == expected
    assert all(c["autocommit"] is True for c in runner.calls)
    assert all(dict(c["params"]) == {} for c in runner.calls)
    assert count == len(expected)


def test_ensure_indexes_twice_sends_the_same_idempotent_statements() -> None:
    s = _schema()
    session_runner = RecordingRunner()
    session = GraphSession(session_runner)
    n1 = vector.ensure_indexes(session, s, vector_dimensions=8)
    sent_first = [c["cypher"] for c in session_runner.calls]
    n2 = vector.ensure_indexes(session, s, vector_dimensions=8)
    sent_second = [c["cypher"] for c in session_runner.calls[len(sent_first) :]]
    assert n1 == n2 == len(sent_first)
    assert sent_first == sent_second
    assert sent_first
    assert all("IF NOT EXISTS" in c for c in sent_first + sent_second)


def test_ensure_indexes_propagates_runner_error_and_stops() -> None:
    s = _schema()
    boom = RuntimeError("second statement failed")
    runner = RecordingRunner(fail_on=2, error=boom)
    with pytest.raises(RuntimeError) as info:
        vector.ensure_indexes(GraphSession(runner), s, vector_dimensions=8)
    assert info.value is boom
    expected = schema_mod.setup_statements(s, vector_dimensions=8)
    assert [c["cypher"] for c in runner.calls] == expected[:2]


def test_ensure_indexes_on_the_repository_schema() -> None:
    s = schema_mod.load()
    runner = RecordingRunner()
    count = vector.ensure_indexes(GraphSession(runner), s, vector_dimensions=8)
    sent = [c["cypher"] for c in runner.calls]
    assert count == len(sent)
    declared_vectors = [
        (label, prop) for label, node in s.nodes.items() for prop in node.vector_indexes
    ]
    assert declared_vectors
    vector_stmts = [c for c in sent if c.startswith("CREATE VECTOR INDEX")]
    assert len(vector_stmts) == len(declared_vectors)
    for label, prop in declared_vectors:
        matching = [c for c in vector_stmts if f"FOR (n:{label}) ON (n.{prop})" in c]
        assert len(matching) == 1
        assert "cosine" in matching[0]
        assert "`vector.dimensions`: 8" in matching[0]
    keyed = [label for label, node in s.nodes.items() if schema_mod.node_key(node) is not None]
    assert keyed
    constraints = [c for c in sent if c.startswith("CREATE CONSTRAINT")]
    assert len(constraints) == len(keyed)
    for label in keyed:
        assert sum(f"FOR (n:{label}) REQUIRE" in c for c in constraints) == 1


# --- vector_index_name ------------------------------------------------------------------------


def _name_in_setup(s: schema_mod.Schema, label: str, prop: str) -> str:
    pattern = re.compile(
        rf"^CREATE VECTOR INDEX (\S+) IF NOT EXISTS FOR \(n:{label}\) ON \(n\.{prop}\)"
    )
    for st in schema_mod.setup_statements(s, vector_dimensions=8):
        m = pattern.match(st)
        if m:
            return m.group(1)
    raise AssertionError(f"no vector index statement for {label}.{prop}")


def test_vector_index_name_matches_the_setup_statement() -> None:
    s = _schema()
    assert vector.vector_index_name(s, "Doc", "embedding") == _name_in_setup(s, "Doc", "embedding")
    assert vector.vector_index_name(s, "Doc", "embedding") == "doc_embedding"


def test_vector_index_name_on_the_repository_schema() -> None:
    s = schema_mod.load()
    for label, node in s.nodes.items():
        for prop in node.vector_indexes:
            assert vector.vector_index_name(s, label, prop) == _name_in_setup(s, label, prop)


@pytest.mark.parametrize(("label", "prop"), [("Missing", "embedding"), ("Doc", "title")])
def test_undeclared_vector_index_raises(label: str, prop: str) -> None:
    s = _schema()
    with pytest.raises(ValueError):
        vector.vector_index_name(s, label, prop)
    with pytest.raises(ValueError):
        vector.vector_search(s, label, prop, limit=5)


# --- vector_search ----------------------------------------------------------------------------


def test_vector_search_default_form() -> None:
    assert vector.vector_search(_schema(), "Doc", "embedding", limit=5) == (
        "MATCH (node:Doc) SEARCH node IN "
        "(VECTOR INDEX doc_embedding FOR $vector LIMIT 5) SCORE AS score"
    )


def test_vector_search_custom_var_and_param() -> None:
    out = vector.vector_search(_schema(), "Doc", "embedding", limit=12, var="d", param="q_vec")
    assert out == (
        "MATCH (d:Doc) SEARCH d IN (VECTOR INDEX doc_embedding FOR $q_vec LIMIT 12) SCORE AS score"
    )


@pytest.mark.parametrize("limit", [0, -3, True, 2.5])
def test_vector_search_bad_limit_raises(limit: Any) -> None:
    with pytest.raises(ValueError):
        vector.vector_search(_schema(), "Doc", "embedding", limit=limit)


@pytest.mark.parametrize("bad", ["1x", "a b", "x)", ""])
def test_vector_search_bad_var_or_param_raises(bad: str) -> None:
    with pytest.raises(ValueError):
        vector.vector_search(_schema(), "Doc", "embedding", limit=5, var=bad)
    with pytest.raises(ValueError):
        vector.vector_search(_schema(), "Doc", "embedding", limit=5, param=bad)


@pytest.mark.parametrize(
    ("var", "param", "limit"), [("node", "vector", 5), ("_n2", "v_1", 1), ("x", "emb", 100)]
)
def test_vector_search_has_no_unlabeled_match(var: str, param: str, limit: int) -> None:
    out = vector.vector_search(_schema(), "Doc", "embedding", limit=limit, var=var, param=param)
    assert schema_mod.unlabeled_matches(out) == []


def test_vector_search_on_the_repository_schema_has_no_unlabeled_match() -> None:
    s = schema_mod.load()
    for label, node in s.nodes.items():
        for prop in node.vector_indexes:
            out = vector.vector_search(s, label, prop, limit=5)
            name = _name_in_setup(s, label, prop)
            assert out == (
                f"MATCH (node:{label}) SEARCH node IN "
                f"(VECTOR INDEX {name} FOR $vector LIMIT 5) SCORE AS score"
            )
            assert schema_mod.unlabeled_matches(out) == []
