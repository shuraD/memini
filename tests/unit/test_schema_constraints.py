"""GS-02: a (scope, id) uniqueness constraint per label with an id, a (scope, status) index per
label with both fields, and one ordered setup list that never creates an index a constraint backs.

Spec: SPEC-graph-schema, schema block ``constraints: [unique (scope, id) on every label that has
an id]`` and ``<every label>(scope,status)`` index. No database in these tests.
"""

from pathlib import Path

import pytest

from memini.core import schema

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_FILE = ROOT / "config" / "graph_schema.yaml"

COMMUNITY_ONLY_FORBIDDEN = ("IS NODE KEY", "IS NOT NULL", "IS ::")


def _prop(rule: str = "a rule") -> dict:
    return {"layer": "core", "writer": "ingest.register", "readers": ["query.*"], "rule": rule}


def _minimal() -> dict:
    """Document{scope, id} declaring [scope, id]; Domain{scope, name} declaring [scope, name]."""
    return {
        "nodes": {
            "Document": {
                "rule": "a source file",
                "properties": {"scope": _prop("scoped"), "id": _prop("identity")},
                "indexes": [["scope", "id"]],
            },
            "Domain": {
                "rule": "a vocabulary value",
                "properties": {"scope": _prop("scoped"), "name": _prop("vocabulary value")},
                "indexes": [["scope", "name"]],
            },
        },
        "relationships": {},
    }


def _rich() -> dict:
    """Common scope/status/id; labels covering every generation case, with a vector index."""
    return {
        "common": {
            "nodes": {
                "scope": _prop("scoped"),
                "status": _prop("lifecycle"),
            },
        },
        "nodes": {
            "Document": {
                "rule": "a source file",
                "properties": {"id": _prop("identity")},
                "indexes": [["scope", "id"]],
            },
            "Domain": {
                "rule": "a vocabulary value",
                "properties": {"name": _prop("vocabulary value")},
                "indexes": [["scope", "name"], ["scope", "status"]],
            },
            "Chunk": {
                "rule": "a passage",
                "properties": {"id": _prop("identity"), "embedding": _prop("vector")},
                "vector_indexes": ["embedding"],
            },
        },
        "relationships": {},
    }


def _count(stmts: list[str], fragment: str) -> int:
    return sum(1 for st in stmts if fragment in st)


# --- constraint_statements ---------------------------------------------------------------------


def test_one_constraint_for_the_label_with_an_id_only() -> None:
    stmts = schema.constraint_statements(schema.from_dict(_minimal()))
    assert len(stmts) == 1
    (st,) = stmts
    assert "FOR (n:Document)" in st
    assert "IF NOT EXISTS" in st
    assert "CREATE CONSTRAINT document_scope_id_unique" in st
    assert st.endswith("REQUIRE (n.scope, n.id) IS UNIQUE")


def test_constraint_statement_has_the_exact_form() -> None:
    stmts = schema.constraint_statements(schema.from_dict(_minimal()))
    assert stmts == [
        "CREATE CONSTRAINT document_scope_id_unique IF NOT EXISTS "
        "FOR (n:Document) REQUIRE (n.scope, n.id) IS UNIQUE"
    ]


def test_a_label_with_id_only_from_common_gets_a_constraint() -> None:
    raw = {
        "common": {"nodes": {"scope": _prop("scoped"), "id": _prop("identity")}},
        "nodes": {
            "Topic": {"rule": "a topic", "properties": {"title": _prop("title")}},
        },
        "relationships": {},
    }
    stmts = schema.constraint_statements(schema.from_dict(raw))
    assert stmts == [
        "CREATE CONSTRAINT topic_scope_id_unique IF NOT EXISTS "
        "FOR (n:Topic) REQUIRE (n.scope, n.id) IS UNIQUE"
    ]


def test_constraints_follow_schema_nodes_order() -> None:
    stmts = schema.constraint_statements(schema.from_dict(_rich()))
    assert len(stmts) == 2
    assert "FOR (n:Document)" in stmts[0]
    assert "FOR (n:Chunk)" in stmts[1]
    assert "CREATE CONSTRAINT chunk_scope_id_unique" in stmts[1]


# --- index_statements: (scope, status) ---------------------------------------------------------


def test_scope_status_index_is_generated_from_common_fields() -> None:
    stmts = schema.index_statements(schema.from_dict(_rich()))
    for label in ("Document", "Chunk"):
        fragment = f"FOR (n:{label}) ON (n.scope, n.status)"
        assert _count(stmts, fragment) == 1
        matching = [st for st in stmts if fragment in st]
        assert matching == [
            f"CREATE INDEX {label.lower()}_scope_status IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.scope, n.status)"
        ]


def test_scope_status_index_is_generated_from_own_fields() -> None:
    raw = {
        "nodes": {
            "Task": {
                "rule": "a task",
                "properties": {"scope": _prop("scoped"), "status": _prop("lifecycle")},
            },
        },
        "relationships": {},
    }
    stmts = schema.index_statements(schema.from_dict(raw))
    assert stmts == [
        "CREATE INDEX task_scope_status IF NOT EXISTS FOR (n:Task) ON (n.scope, n.status)"
    ]


def test_no_scope_status_index_for_a_label_without_status() -> None:
    stmts = schema.index_statements(schema.from_dict(_minimal()))
    assert not any("n.status" in st for st in stmts)


def test_a_declared_scope_status_index_appears_exactly_once() -> None:
    stmts = schema.index_statements(schema.from_dict(_rich()))
    assert _count(stmts, "FOR (n:Domain) ON (n.scope, n.status)") == 1


def test_declared_indexes_are_backward_compatible() -> None:
    stmts = schema.index_statements(schema.from_dict(_minimal()))
    assert any("FOR (n:Document) ON (n.scope, n.id)" in st for st in stmts)
    assert (
        "CREATE INDEX document_scope_id IF NOT EXISTS FOR (n:Document) ON (n.scope, n.id)" in stmts
    )
    assert (
        "CREATE INDEX domain_scope_name IF NOT EXISTS FOR (n:Domain) ON (n.scope, n.name)" in stmts
    )


# --- vector options ----------------------------------------------------------------------------


def test_vector_index_without_dimensions_has_no_options() -> None:
    s = schema.from_dict(_rich())
    for stmts in (schema.index_statements(s), schema.setup_statements(s)):
        assert (
            "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (n:Chunk) ON (n.embedding)"
            in (stmts)
        )
        assert not any("OPTIONS" in st for st in stmts)


@pytest.mark.parametrize("fn", ["index_statements", "setup_statements"])
def test_vector_index_with_dimensions_has_cosine_options(fn: str) -> None:
    stmts = getattr(schema, fn)(schema.from_dict(_rich()), vector_dimensions=768)
    vectors = [st for st in stmts if st.startswith("CREATE VECTOR INDEX")]
    assert len(vectors) == 1
    for st in vectors:
        assert "`vector.dimensions`: 768" in st
        assert "`vector.similarity_function`: 'cosine'" in st
    assert vectors == [
        "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (n:Chunk) ON (n.embedding)"
        " OPTIONS {indexConfig: {`vector.dimensions`: 768, "
        "`vector.similarity_function`: 'cosine'}}"
    ]
    non_vector = [st for st in stmts if not st.startswith("CREATE VECTOR INDEX")]
    assert not any("OPTIONS" in st for st in non_vector)


# --- setup_statements --------------------------------------------------------------------------


def _is_index(st: str) -> bool:
    return st.startswith("CREATE INDEX") or st.startswith("CREATE VECTOR INDEX")


def test_setup_puts_constraints_first_and_drops_backed_indexes() -> None:
    stmts = schema.setup_statements(schema.from_dict(_minimal()))
    constraint_positions = [i for i, st in enumerate(stmts) if st.startswith("CREATE CONSTRAINT")]
    index_positions = [i for i, st in enumerate(stmts) if _is_index(st)]
    assert constraint_positions and index_positions
    assert max(constraint_positions) < min(index_positions)
    assert any(
        "CREATE CONSTRAINT document_scope_id_unique" in st and "FOR (n:Document)" in st
        for st in stmts
    )
    assert not any(_is_index(st) and "FOR (n:Document) ON (n.scope, n.id)" in st for st in stmts)


def test_setup_keeps_other_indexes_and_has_no_duplicates() -> None:
    s = schema.from_dict(_rich())
    stmts = schema.setup_statements(s, vector_dimensions=768)
    assert len(stmts) == len(set(stmts))
    assert "CREATE INDEX domain_scope_name IF NOT EXISTS FOR (n:Domain) ON (n.scope, n.name)" in (
        stmts
    )
    for label in ("Document", "Domain", "Chunk"):
        assert _count(stmts, f"FOR (n:{label}) ON (n.scope, n.status)") == 1
    assert _count(stmts, "CREATE VECTOR INDEX chunk_embedding") == 1
    assert not any(_is_index(st) and "ON (n.scope, n.id)" in st for st in stmts)
    constraints = [st for st in stmts if st.startswith("CREATE CONSTRAINT")]
    assert constraints == schema.constraint_statements(s)
    assert stmts[: len(constraints)] == constraints


def test_setup_equals_constraints_then_unbacked_indexes() -> None:
    s = schema.from_dict(_rich())
    expected_indexes = [
        st
        for st in schema.index_statements(s, vector_dimensions=768)
        if "ON (n.scope, n.id)" not in st
    ]
    assert schema.setup_statements(s, vector_dimensions=768) == (
        schema.constraint_statements(s) + expected_indexes
    )


# --- Community edition, committed schema -------------------------------------------------------


@pytest.mark.parametrize("raw_factory", [_minimal, _rich])
def test_no_enterprise_only_syntax_in_any_statement(raw_factory) -> None:
    s = schema.from_dict(raw_factory())
    all_stmts = (
        schema.constraint_statements(s)
        + schema.index_statements(s)
        + schema.index_statements(s, vector_dimensions=768)
        + schema.setup_statements(s)
        + schema.setup_statements(s, vector_dimensions=768)
    )
    assert all_stmts
    for st in all_stmts:
        for bad in COMMUNITY_ONLY_FORBIDDEN:
            assert bad not in st


@pytest.mark.parametrize("dims", [None, 768])
def test_committed_schema_setup_statements_are_idempotent_creates(dims) -> None:
    s = schema.load(SCHEMA_FILE)
    stmts = schema.setup_statements(s, vector_dimensions=dims)
    assert stmts
    assert len(stmts) == len(set(stmts))
    for st in stmts:
        assert st.startswith("CREATE ")
        assert "IF NOT EXISTS" in st
        for bad in COMMUNITY_ONLY_FORBIDDEN:
            assert bad not in st
