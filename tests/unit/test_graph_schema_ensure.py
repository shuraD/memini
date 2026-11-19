"""GR-02 acceptance: load config/graph_schema.yaml into GraphSchema and ensure_schema."""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import neo4j_graphrag.indexes as graphrag_indexes
import pytest

from memini.infra.graph import schema as schema_mod
from memini.infra.graph.schema import (
    GraphSchema,
    ensure_schema,
    load_schema,
    vector_index_name,
)

REPO_SCHEMA = Path(__file__).resolve().parents[2] / "config" / "graph_schema.yaml"
BASE = ("scope", "updated_by", "updated_at")
REPO_NODES = {
    "Project": (*BASE, "name"),
    "Domain": (*BASE, "name"),
    "Artifact": (
        *BASE,
        "id",
        "type",
        "role",
        "content_hash",
        "summary",
        "labels",
        "status",
    ),
    "Chunk": (*BASE, "id", "content", "line_start", "line_end", "embedding"),
    "Entity": (
        *BASE,
        "id",
        "name",
        "type",
        "summary",
        "canonical_id",
        "references",
        "labels",
        "embedding",
        "embedding_code",
        "status",
    ),
    "Community": (*BASE, "id", "level", "summary", "embedding", "dirty"),
    "RelationType": (*BASE, "name"),
}
REPO_RELATIONSHIPS = {
    "HAS_DOMAIN": (),
    "CONTAINS": (),
    "HAS_CHUNK": ("order",),
    "HAS_ENTITY": (),
    "RELATED_TO": ("kind", "type", "description"),
    "SIMILAR": ("score", "code_score"),
    "CODE_LINK": ("kind", "line"),
    "SUPERSEDES": (),
    "BELONGS_TO_COMMUNITY": (),
    "HIERARCHICAL_LINK": ("similarity",),
    "INTER_LAYER_LINK": (),
}
REPO_VALUES = {
    ("Artifact", "status"): ("active", "superseded", "deleted"),
    ("Entity", "status"): ("active", "superseded", "deleted"),
    ("RELATED_TO", "kind"): ("extracted", "sequence", "label", "cross_domain", "shared"),
    ("CODE_LINK", "kind"): ("define", "call", "import"),
}

_LABEL = r"\(\s*\w+\s*:\s*`?(\w+)`?\s*\)"
_PROP = r"\w+\s*\.\s*`?(\w+)`?"
CONSTRAINT_RE = re.compile(
    r"^\s*CREATE\s+CONSTRAINT\s+(?:\S+\s+)?IF\s+NOT\s+EXISTS\s+FOR\s+"
    + _LABEL
    + r"\s+REQUIRE\s+\(\s*\w+\s*\.\s*`?scope`?\s*,\s*\w+\s*\.\s*`?id`?\s*\)\s+IS\s+UNIQUE\s*;?\s*$",
    re.IGNORECASE,
)
RANGE_RE = re.compile(
    r"^\s*CREATE\s+(?:RANGE\s+)?INDEX\s+(?:\S+\s+)?IF\s+NOT\s+EXISTS\s+FOR\s+"
    + _LABEL
    + r"\s+ON\s+\(\s*\w+\s*\.\s*`?scope`?\s*,\s*"
    + _PROP
    + r"\s*\)\s*;?\s*$",
    re.IGNORECASE,
)

_MISSING = object()


class FakeDriver:
    """Records every execute_query call; refuses sessions."""

    def __init__(self) -> None:
        self.queries: list[tuple[str, Any]] = []

    def execute_query(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None, list[str]]:
        statement = args[0] if args else kwargs.get("query_", kwargs.get("query"))
        if not isinstance(statement, str):
            statement = getattr(statement, "text", str(statement))
        self.queries.append((statement, kwargs.get("database_", _MISSING)))
        return [], None, []

    def session(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("ensure_schema must not open a session")


class VectorRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._sig = inspect.signature(graphrag_indexes.create_vector_index)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        bound = self._sig.bind(*args, **kwargs)
        bound.apply_defaults()
        self.calls.append(dict(bound.arguments))


@pytest.fixture
def vectors(monkeypatch: pytest.MonkeyPatch) -> VectorRecorder:
    recorder = VectorRecorder()
    monkeypatch.setattr(graphrag_indexes, "create_vector_index", recorder)
    monkeypatch.setattr(schema_mod, "create_vector_index", recorder, raising=False)
    return recorder


def _classify(driver: FakeDriver) -> tuple[list[str], list[tuple[str, str]]]:
    constraints: list[str] = []
    ranges: list[tuple[str, str]] = []
    for statement, _db in driver.queries:
        c = CONSTRAINT_RE.match(statement)
        r = RANGE_RE.match(statement)
        assert c or r, f"unexpected statement: {statement!r}"
        if c:
            constraints.append(c.group(1))
        elif r:
            ranges.append((r.group(1), r.group(2)))
    return constraints, ranges


def _vector_keys(rec: VectorRecorder) -> list[tuple[str, str]]:
    return [(c["label"], c["embedding_property"]) for c in rec.calls]


# --- load_schema -----------------------------------------------------------------


def test_load_schema_default_reads_repo_file() -> None:
    schema = load_schema()
    assert isinstance(schema, GraphSchema)
    assert dict(schema.nodes) == REPO_NODES
    assert list(schema.nodes) == list(REPO_NODES)
    assert dict(schema.relationships) == REPO_RELATIONSHIPS
    assert dict(schema.values) == REPO_VALUES


def test_load_schema_property_tuples_are_tuples() -> None:
    schema = load_schema(REPO_SCHEMA)
    for props in schema.nodes.values():
        assert isinstance(props, tuple)
    for props in schema.relationships.values():
        assert isinstance(props, tuple)
    for allowed in schema.values.values():
        assert isinstance(allowed, tuple)


def test_relationship_without_properties_is_empty_tuple() -> None:
    schema = load_schema(REPO_SCHEMA)
    assert schema.relationships["HAS_DOMAIN"] == ()
    assert schema.relationships["INTER_LAYER_LINK"] == ()


def test_load_schema_default_follows_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "graph_schema.yaml").write_text(
        "nodes:\n  Widget:\n    properties: [scope, id]\nrelationships: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMINI_CONFIG_DIR", str(tmp_path))
    schema = load_schema()
    assert dict(schema.nodes) == {"Widget": ("scope", "id")}
    assert dict(schema.relationships) == {}


def test_load_schema_generic_path(tmp_path: Path) -> None:
    path = tmp_path / "custom.yaml"
    path.write_text(
        "nodes:\n"
        "  Zeta:\n"
        "    properties: [scope, id, colour, embedding_alt]\n"
        "    values:\n"
        "      colour: [red, blue]\n"
        "  Alpha:\n"
        "    properties: [title]\n"
        "relationships:\n"
        "  POINTS_AT: {from: Zeta, to: Alpha, properties: [weight, note]}\n"
        "  BARE: {from: Alpha, to: Zeta}\n"
        "  TAGGED:\n"
        "    from: Zeta\n"
        "    to: Zeta\n"
        "    properties: [mode]\n"
        "    values:\n"
        "      mode: [x, y, z]\n",
        encoding="utf-8",
    )
    schema = load_schema(path)
    assert dict(schema.nodes) == {
        "Zeta": ("scope", "id", "colour", "embedding_alt"),
        "Alpha": ("title",),
    }
    assert list(schema.nodes) == ["Zeta", "Alpha"]
    assert dict(schema.relationships) == {
        "POINTS_AT": ("weight", "note"),
        "BARE": (),
        "TAGGED": ("mode",),
    }
    assert dict(schema.values) == {
        ("Zeta", "colour"): ("red", "blue"),
        ("TAGGED", "mode"): ("x", "y", "z"),
    }


def test_graph_schema_is_frozen() -> None:
    schema = GraphSchema(nodes={}, relationships={}, values={})
    with pytest.raises(AttributeError):
        schema.nodes = {}  # type: ignore[misc]


# --- vector_index_name -------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "prop", "expected"),
    [
        ("Entity", "embedding_code", "entity_embedding_code"),
        ("Entity", "embedding", "entity_embedding"),
        ("Community", "embedding", "community_embedding"),
        ("MyThing", "embedding_x", "mything_embedding_x"),
    ],
)
def test_vector_index_name(label: str, prop: str, expected: str) -> None:
    assert vector_index_name(label, prop) == expected


# --- ensure_schema on the repo schema ------------------------------------------------


def test_repo_constraints(vectors: VectorRecorder) -> None:
    driver = FakeDriver()
    ensure_schema(driver, load_schema(REPO_SCHEMA), 384, database="testdb")  # type: ignore[arg-type]
    constraints, _ = _classify(driver)
    assert sorted(constraints) == ["Artifact", "Chunk", "Community", "Entity"]


def test_repo_range_indexes(vectors: VectorRecorder) -> None:
    driver = FakeDriver()
    ensure_schema(driver, load_schema(REPO_SCHEMA), 384, database="testdb")  # type: ignore[arg-type]
    _, ranges = _classify(driver)
    assert len(ranges) == len(set(ranges))
    assert set(ranges) == {
        ("Project", "name"),
        ("Domain", "name"),
        ("Artifact", "type"),
        ("Artifact", "content_hash"),
        ("Artifact", "status"),
        ("Entity", "name"),
        ("Entity", "type"),
        ("Entity", "status"),
        ("Community", "level"),
        ("RelationType", "name"),
    }


def test_repo_vector_indexes(vectors: VectorRecorder) -> None:
    driver = FakeDriver()
    ensure_schema(driver, load_schema(REPO_SCHEMA), 384, database="testdb")  # type: ignore[arg-type]
    keys = _vector_keys(vectors)
    assert len(keys) == len(set(keys))
    assert set(keys) == {
        ("Entity", "embedding"),
        ("Entity", "embedding_code"),
        ("Chunk", "embedding"),
        ("Community", "embedding"),
    }
    for call in vectors.calls:
        assert call["driver"] is driver
        assert call["name"] == vector_index_name(call["label"], call["embedding_property"])
        assert call["dimensions"] == 384
        assert call["similarity_fn"] == "cosine"
        assert not call["fail_if_exists"]
        assert call["neo4j_database"] == "testdb"


def test_repo_statements_shape_and_database(vectors: VectorRecorder) -> None:
    driver = FakeDriver()
    ensure_schema(driver, load_schema(REPO_SCHEMA), 384, database="testdb")  # type: ignore[arg-type]
    assert driver.queries
    for statement, db in driver.queries:
        assert "IF NOT EXISTS" in statement.upper()
        assert db == "testdb"
    _classify(driver)


def test_database_none_is_passed_through(vectors: VectorRecorder) -> None:
    driver = FakeDriver()
    ensure_schema(driver, load_schema(REPO_SCHEMA), 768)  # type: ignore[arg-type]
    assert driver.queries
    for _statement, db in driver.queries:
        assert db is None
    assert vectors.calls
    for call in vectors.calls:
        assert call["neo4j_database"] is None
        assert call["dimensions"] == 768


def test_ensure_schema_idempotent(vectors: VectorRecorder) -> None:
    driver = FakeDriver()
    schema = load_schema(REPO_SCHEMA)
    ensure_schema(driver, schema, 384, database="testdb")  # type: ignore[arg-type]
    first_queries = list(driver.queries)
    first_vectors = [{k: v for k, v in c.items() if k != "driver"} for c in vectors.calls]
    ensure_schema(driver, schema, 384, database="testdb")  # type: ignore[arg-type]
    second_queries = driver.queries[len(first_queries) :]
    second_vectors = [
        {k: v for k, v in c.items() if k != "driver"} for c in vectors.calls[len(first_vectors) :]
    ]
    assert first_queries
    assert second_queries == first_queries
    assert second_vectors == first_vectors


# --- ensure_schema on an arbitrary schema --------------------------------------------


def test_generic_schema(vectors: VectorRecorder) -> None:
    schema = GraphSchema(
        nodes={
            "Gadget": ("scope", "id", "name", "level", "embedding_a", "embedding_b", "colour"),
            "NoScope": ("id", "name", "status", "embedding"),
            "ScopeOnly": ("scope", "type", "content_hash", "weight"),
            "Plain": ("title",),
        },
        relationships={"LINKS": ("name", "embedding")},
        values={("Gadget", "colour"): ("red", "blue")},
    )
    driver = FakeDriver()
    ensure_schema(driver, schema, 16, database="db2")  # type: ignore[arg-type]
    constraints, ranges = _classify(driver)
    assert constraints == ["Gadget"]
    assert len(ranges) == len(set(ranges))
    assert set(ranges) == {
        ("Gadget", "name"),
        ("Gadget", "level"),
        ("ScopeOnly", "type"),
        ("ScopeOnly", "content_hash"),
    }
    keys = _vector_keys(vectors)
    assert len(keys) == len(set(keys))
    assert set(keys) == {
        ("Gadget", "embedding_a"),
        ("Gadget", "embedding_b"),
        ("NoScope", "embedding"),
    }
    for call in vectors.calls:
        assert call["name"] == vector_index_name(call["label"], call["embedding_property"])
        assert call["dimensions"] == 16
        assert call["similarity_fn"] == "cosine"
        assert not call["fail_if_exists"]
        assert call["neo4j_database"] == "db2"
    for _statement, db in driver.queries:
        assert db == "db2"


def test_empty_schema_sends_nothing(vectors: VectorRecorder) -> None:
    driver = FakeDriver()
    ensure_schema(driver, GraphSchema(nodes={}, relationships={}, values={}), 8)  # type: ignore[arg-type]
    assert driver.queries == []
    assert vectors.calls == []
