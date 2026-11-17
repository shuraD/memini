"""GR-06 acceptance: typed graph names from the schema file's ``names`` section.

``load_schema`` reads ``names`` into ``GraphSchema.names`` (labels, relationships, properties,
statuses; one frozen ``str`` attribute per role). ``ensure_schema`` takes every graph name from
``schema.names``, and neither ``schema.py`` nor ``store.py`` spells a graph name as a string
literal (AGENTS.md owner rule 2, owner 2026-11-16).
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from typing import Any

import neo4j_graphrag.indexes as graphrag_indexes
import pytest
import yaml

from memini.infra.graph import schema as schema_mod
from memini.infra.graph.schema import GraphSchema, ensure_schema, load_schema

ROOT = Path(__file__).resolve().parents[2]
REPO_SCHEMA = ROOT / "config" / "graph_schema.yaml"
SOURCES = (
    ROOT / "src" / "memini" / "infra" / "graph" / "schema.py",
    ROOT / "src" / "memini" / "infra" / "graph" / "store.py",
)
SECTIONS = ("labels", "relationships", "properties", "statuses")

EXPECTED_ROLES = {
    "labels": {
        "project",
        "domain",
        "artifact",
        "chunk",
        "entity",
        "community",
        "relation_type",
    },
    "relationships": {
        "has_domain",
        "contains",
        "has_chunk",
        "has_entity",
        "related_to",
        "similar",
        "code_link",
        "supersedes",
        "member_of",
        "hierarchical_link",
        "inter_layer_link",
    },
    "properties": {
        "scope",
        "id",
        "status",
        "content_hash",
        "updated_by",
        "updated_at",
        "summary",
        "embedding",
        "type",
        "level",
    },
    "statuses": {"active", "superseded", "deleted"},
}

# A custom schema: same roles, every one mapped to another name.
CUSTOM_NAMES = {
    "labels": {
        "project": "Proj",
        "domain": "Area",
        "artifact": "Doc",
        "chunk": "Piece",
        "entity": "Thing",
        "community": "Cluster",
        "relation_type": "RelKind",
    },
    "relationships": {
        "has_domain": "OWNS_AREA",
        "contains": "HOLDS",
        "has_chunk": "HAS_PIECE",
        "has_entity": "MENTIONS",
        "related_to": "LINKED",
        "similar": "ALIKE",
        "code_link": "CODE_REF",
        "supersedes": "REPLACES",
        "member_of": "IN_CLUSTER",
        "hierarchical_link": "PARENT_OF",
        "inter_layer_link": "CROSS_LAYER",
    },
    "properties": {
        "scope": "tenant",
        "id": "key",
        "status": "state",
        "content_hash": "digest",
        "updated_by": "author",
        "updated_at": "stamp",
        "summary": "abstract",
        "embedding": "vec",
        "type": "kind_of",
        "level": "tier",
    },
    "statuses": {"active": "live", "superseded": "old", "deleted": "gone"},
}

CUSTOM_DOC = {
    "nodes": {
        "Doc": {
            "properties": [
                "tenant",
                "author",
                "stamp",
                "key",
                "kind_of",
                "digest",
                "abstract",
                "state",
                "type",
                "status",
            ],
            "values": {"state": ["live", "old", "gone"]},
        },
        "Thing": {
            "properties": ["tenant", "key", "kind_of", "state", "vec", "vec_code", "embedding"],
        },
        "Cluster": {"properties": ["tenant", "key", "tier", "abstract", "vec", "level"]},
        "Piece": {"properties": ["tenant", "key", "content", "vec"]},
        "Loose": {"properties": ["key", "state", "vec_x"]},
        "Plain": {"properties": ["title"]},
    },
    "relationships": {
        "MENTIONS": {"from": "Piece", "to": "Thing"},
        "LINKED": {
            "from": ["Thing", "Cluster"],
            "to": ["Thing", "Cluster"],
            "properties": ["mode", "weight"],
            "values": {"mode": ["a", "b"]},
        },
    },
    "names": CUSTOM_NAMES,
}


def _write_custom(tmp_path: Path) -> Path:
    path = tmp_path / "custom_schema.yaml"
    path.write_text(yaml.safe_dump(CUSTOM_DOC, sort_keys=False), encoding="utf-8")
    return path


def _repo_names() -> dict[str, dict[str, str]]:
    data = yaml.safe_load(REPO_SCHEMA.read_text(encoding="utf-8"))
    names = data["names"]
    return {section: {str(k): str(v) for k, v in names[section].items()} for section in SECTIONS}


# --- load_schema: names from the repository file ------------------------------------


def test_load_schema_reads_every_repo_name() -> None:
    schema = load_schema()
    repo = _repo_names()
    for section in SECTIONS:
        assert set(repo[section]) == EXPECTED_ROLES[section]
    for section, roles in repo.items():
        part = getattr(schema.names, section)
        for role, name in roles.items():
            value = getattr(part, role)
            assert isinstance(value, str)
            assert value == name, (section, role)


def test_load_schema_repo_name_examples() -> None:
    names = load_schema(REPO_SCHEMA).names
    assert names.labels.entity == "Entity"
    assert names.labels.relation_type == "RelationType"
    assert names.relationships.member_of == "BELONGS_TO_COMMUNITY"
    assert names.relationships.inter_layer_link == "INTER_LAYER_LINK"
    assert names.properties.content_hash == "content_hash"
    assert names.properties.scope == "scope"
    assert names.properties.id == "id"
    assert names.statuses.active == "active"
    assert names.statuses.deleted == "deleted"


def test_names_is_a_graph_names() -> None:
    schema = load_schema(REPO_SCHEMA)
    assert isinstance(schema.names, schema_mod.GraphNames)
    assert isinstance(schema, GraphSchema)


# --- load_schema: names from a custom file ---------------------------------------------


def test_load_schema_custom_names_are_read_from_file(tmp_path: Path) -> None:
    schema = load_schema(_write_custom(tmp_path))
    for section, roles in CUSTOM_NAMES.items():
        part = getattr(schema.names, section)
        for role, name in roles.items():
            assert getattr(part, role) == name, (section, role)


def test_load_schema_custom_other_sections_unchanged(tmp_path: Path) -> None:
    schema = load_schema(_write_custom(tmp_path))
    assert schema.names.labels.entity == "Thing"
    assert list(schema.nodes) == ["Doc", "Thing", "Cluster", "Piece", "Loose", "Plain"]
    assert schema.nodes["Thing"] == (
        "tenant",
        "key",
        "kind_of",
        "state",
        "vec",
        "vec_code",
        "embedding",
    )
    assert schema.nodes["Plain"] == ("title",)
    assert dict(schema.relationships) == {"MENTIONS": (), "LINKED": ("mode", "weight")}
    assert dict(schema.values) == {
        ("Doc", "state"): ("live", "old", "gone"),
        ("LINKED", "mode"): ("a", "b"),
    }


def test_repo_nodes_relationships_values_still_load() -> None:
    schema = load_schema(REPO_SCHEMA)
    assert schema.names.labels.entity in schema.nodes
    assert "Entity" in schema.nodes
    assert schema.relationships["BELONGS_TO_COMMUNITY"] == ()
    assert schema.values[("Entity", "status")] == ("active", "superseded", "deleted")
    assert "names" not in schema.nodes
    assert "names" not in schema.relationships


# --- immutability -------------------------------------------------------------------------


def test_assigning_a_name_raises() -> None:
    names = load_schema(REPO_SCHEMA).names
    with pytest.raises(AttributeError):
        names.labels.entity = "Other"  # type: ignore[misc]
    assert names.labels.entity == "Entity"


@pytest.mark.parametrize("section", SECTIONS)
def test_assigning_any_role_raises(section: str) -> None:
    part = getattr(load_schema(REPO_SCHEMA).names, section)
    for role in EXPECTED_ROLES[section]:
        before = getattr(part, role)
        with pytest.raises(AttributeError):
            setattr(part, role, "Other")
        assert getattr(part, role) == before


@pytest.mark.parametrize("section", SECTIONS)
def test_assigning_a_section_raises(section: str) -> None:
    names = load_schema(REPO_SCHEMA).names
    before = getattr(names, section)
    with pytest.raises(AttributeError):
        setattr(names, section, before)
    with pytest.raises(AttributeError):
        names.labels = names.relationships  # type: ignore[misc]
    assert getattr(names, section) is before


def test_assigning_schema_names_raises() -> None:
    schema = load_schema(REPO_SCHEMA)
    names = schema.names
    with pytest.raises(AttributeError):
        schema.names = names  # type: ignore[misc]
    assert schema.names is names


# --- ensure_schema on a schema with renamed names --------------------------------------------


class FakeDriver:
    """Records every execute_query statement and its database; refuses sessions."""

    def __init__(self) -> None:
        self.queries: list[tuple[str, Any]] = []

    def execute_query(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None, list[str]]:
        statement = args[0] if args else kwargs.get("query_", kwargs.get("query"))
        if not isinstance(statement, str):
            statement = getattr(statement, "text", str(statement))
        self.queries.append((statement, kwargs.get("database_")))
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


_LABEL = r"\(\s*\w+\s*:\s*`?(\w+)`?\s*\)"
_PROP = r"\w+\s*\.\s*`?(\w+)`?"
CONSTRAINT_RE = re.compile(
    r"^\s*CREATE\s+CONSTRAINT\s+(?:\S+\s+)?IF\s+NOT\s+EXISTS\s+FOR\s+"
    + _LABEL
    + r"\s+REQUIRE\s+\(\s*"
    + _PROP
    + r"\s*,\s*"
    + _PROP
    + r"\s*\)\s+IS\s+UNIQUE\s*;?\s*$",
    re.IGNORECASE,
)
RANGE_RE = re.compile(
    r"^\s*CREATE\s+(?:RANGE\s+)?INDEX\s+(?:\S+\s+)?IF\s+NOT\s+EXISTS\s+FOR\s+"
    + _LABEL
    + r"\s+ON\s+\(\s*"
    + _PROP
    + r"\s*,\s*"
    + _PROP
    + r"\s*\)\s*;?\s*$",
    re.IGNORECASE,
)


def _classify(
    driver: FakeDriver,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    constraints: list[tuple[str, str, str]] = []
    ranges: list[tuple[str, str, str]] = []
    for statement, _db in driver.queries:
        c = CONSTRAINT_RE.match(statement)
        r = RANGE_RE.match(statement)
        assert c or r, f"unexpected statement: {statement!r}"
        if c:
            constraints.append((c.group(1), c.group(2), c.group(3)))
        elif r:
            ranges.append((r.group(1), r.group(2), r.group(3)))
    return constraints, ranges


def _run_custom(tmp_path: Path, vectors: VectorRecorder) -> FakeDriver:
    driver = FakeDriver()
    ensure_schema(driver, load_schema(_write_custom(tmp_path)), 32, database="cdb")  # type: ignore[arg-type]
    return driver


def test_custom_uniqueness_on_renamed_scope_and_id(tmp_path: Path, vectors: VectorRecorder) -> None:
    constraints, _ = _classify(_run_custom(tmp_path, vectors))
    assert sorted(constraints) == [
        ("Cluster", "tenant", "key"),
        ("Doc", "tenant", "key"),
        ("Piece", "tenant", "key"),
        ("Thing", "tenant", "key"),
    ]


def test_custom_range_indexes_on_renamed_properties(
    tmp_path: Path, vectors: VectorRecorder
) -> None:
    _, ranges = _classify(_run_custom(tmp_path, vectors))
    assert len(ranges) == len(set(ranges))
    assert set(ranges) == {
        ("Doc", "tenant", "kind_of"),
        ("Doc", "tenant", "digest"),
        ("Doc", "tenant", "state"),
        ("Thing", "tenant", "kind_of"),
        ("Thing", "tenant", "state"),
        ("Cluster", "tenant", "tier"),
    }


def test_custom_vector_indexes_on_renamed_embedding(
    tmp_path: Path, vectors: VectorRecorder
) -> None:
    driver = _run_custom(tmp_path, vectors)
    keys = [(c["label"], c["embedding_property"]) for c in vectors.calls]
    assert len(keys) == len(set(keys))
    assert set(keys) == {
        ("Thing", "vec"),
        ("Thing", "vec_code"),
        ("Cluster", "vec"),
        ("Piece", "vec"),
        ("Loose", "vec_x"),
    }
    for call in vectors.calls:
        assert call["driver"] is driver
        assert call["name"] == schema_mod.vector_index_name(
            call["label"], call["embedding_property"]
        )
        assert call["dimensions"] == 32
        assert call["similarity_fn"] == "cosine"
        assert not call["fail_if_exists"]
        assert call["neo4j_database"] == "cdb"


def test_custom_statements_never_mention_default_scope_or_id(
    tmp_path: Path, vectors: VectorRecorder
) -> None:
    driver = _run_custom(tmp_path, vectors)
    assert driver.queries
    for statement, db in driver.queries:
        assert db == "cdb"
        assert "IF NOT EXISTS" in statement.upper()
        assert not re.search(r"(?<![A-Za-z0-9])scope(?![A-Za-z0-9])", statement), statement
        assert not re.search(r"(?<![A-Za-z0-9])id(?![A-Za-z0-9])", statement), statement


# --- no graph name as a string literal in schema.py / store.py --------------------------------


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
        ):
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                ids.add(id(first.value))
    return ids


def _string_constants(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    skip = _docstring_ids(tree)
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip
    ]


def _all_repo_names() -> set[str]:
    return {name for roles in _repo_names().values() for name in roles.values()}


def _labels_and_relationships() -> set[str]:
    names = _repo_names()
    return set(names["labels"].values()) | set(names["relationships"].values())


def _check_scanner(tmp_path: Path) -> None:
    """The scanner sees f-string parts, dict keys and skips docstrings."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        '"""Module doc: Entity scope."""\n'
        "def f(x):\n"
        '    """Function doc: BELONGS_TO_COMMUNITY."""\n'
        '    return f"MATCH (n:`Entity`) {x}", {"scope": x}\n',
        encoding="utf-8",
    )
    values = [v for _line, v in _string_constants(sample)]
    assert "scope" in values
    assert any("Entity" in v for v in values)
    assert not any("doc" in v for v in values)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_graph_name_in_string_constants(path: Path, tmp_path: Path) -> None:
    _check_scanner(tmp_path)
    assert path.is_file(), path
    constants = _string_constants(path)
    forbidden = _all_repo_names()
    equal = [(line, value) for line, value in constants if value in forbidden]
    words = _labels_and_relationships()
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(" + "|".join(map(re.escape, words)) + r")(?![A-Za-z0-9_])"
    )
    inside = [(line, value) for line, value in constants if pattern.search(value)]
    assert equal == [], f"{path.name}: graph names as string literals: {equal}"
    assert inside == [], f"{path.name}: labels/relationships inside string literals: {inside}"
