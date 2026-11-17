"""GR-06 acceptance: the schema's names section, typed on GraphSchema, used by schema/store."""

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
from memini.infra.graph.schema import ensure_schema, load_schema

ROOT = Path(__file__).resolve().parents[2]
REPO_SCHEMA = ROOT / "config" / "graph_schema.yaml"
SOURCES = (
    ROOT / "src" / "memini" / "infra" / "graph" / "schema.py",
    ROOT / "src" / "memini" / "infra" / "graph" / "store.py",
)
SECTIONS = ("labels", "relationships", "properties", "statuses")


def _repo_names() -> dict[str, dict[str, str]]:
    data = yaml.safe_load(REPO_SCHEMA.read_text(encoding="utf-8"))
    names = data["names"]
    return {s: {str(k): str(v) for k, v in names[s].items()} for s in SECTIONS}


REPO_NAMES = _repo_names()
REPO_PAIRS = [(s, role, name) for s in SECTIONS for role, name in REPO_NAMES[s].items()]

CUSTOM_NAMES = {
    "labels": {
        "project": "Proj",
        "domain": "Dom",
        "artifact": "Doc",
        "chunk": "Piece",
        "entity": "Thing",
        "community": "Group",
        "relation_type": "Kind",
    },
    "relationships": {
        "has_domain": "R_DOM",
        "contains": "R_HOLDS",
        "has_chunk": "R_PIECE",
        "has_entity": "R_THING",
        "related_to": "R_LINK",
        "similar": "R_NEAR",
        "code_link": "R_CODE",
        "supersedes": "R_REPLACES",
        "member_of": "R_IN_GROUP",
        "hierarchical_link": "R_UP",
        "inter_layer_link": "R_ACROSS",
    },
    "properties": {
        "scope": "part",
        "id": "key",
        "status": "state",
        "content_hash": "digest",
        "updated_by": "who",
        "updated_at": "when_at",
        "summary": "blurb",
        "embedding": "vec",
        "type": "sort",
        "level": "tier",
    },
    "statuses": {"active": "live", "superseded": "replaced", "deleted": "gone"},
}


def _custom_file(tmp_path: Path) -> Path:
    data = {
        "nodes": {
            "Thing": {
                "properties": [
                    "part",
                    "key",
                    "state",
                    "digest",
                    "sort",
                    "tier",
                    "vec",
                    "vec_code",
                    "colour",
                ],
                "values": {"state": ["live", "replaced", "gone"]},
            },
            "Loose": {"properties": ["key", "state", "vec"]},
            "Old": {"properties": ["scope", "id", "status", "type", "level", "embedding"]},
        },
        "relationships": {"R_LINK": {"from": "Thing", "to": "Thing", "properties": ["weight"]}},
        "names": CUSTOM_NAMES,
    }
    path = tmp_path / "custom_schema.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


# --- load_schema: names --------------------------------------------------------------


@pytest.mark.parametrize(("section", "role", "name"), REPO_PAIRS)
def test_repo_names_are_loaded(section: str, role: str, name: str) -> None:
    schema = load_schema()
    assert getattr(getattr(schema.names, section), role) == name


def test_repo_names_examples() -> None:
    names = load_schema(REPO_SCHEMA).names
    assert names.labels.entity == "Entity"
    assert names.labels.community == "Community"
    assert names.relationships.member_of == "BELONGS_TO_COMMUNITY"
    assert names.relationships.supersedes == "SUPERSEDES"
    assert names.properties.content_hash == "content_hash"
    assert names.properties.scope == "scope"
    assert names.statuses.active == "active"
    assert names.statuses.deleted == "deleted"


def test_custom_names_come_from_the_file(tmp_path: Path) -> None:
    schema = load_schema(_custom_file(tmp_path))
    for section, roles in CUSTOM_NAMES.items():
        for role, name in roles.items():
            assert getattr(getattr(schema.names, section), role) == name
    assert schema.nodes["Loose"] == ("key", "state", "vec")
    assert dict(schema.relationships) == {"R_LINK": ("weight",)}
    assert dict(schema.values) == {("Thing", "state"): ("live", "replaced", "gone")}


def test_names_are_frozen() -> None:
    schema = load_schema(REPO_SCHEMA)
    with pytest.raises(AttributeError):
        schema.names.labels.entity = "Other"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        schema.names.labels = schema.names.statuses  # type: ignore[misc]
    with pytest.raises(AttributeError):
        schema.names = schema.names  # type: ignore[misc]
    assert schema.names.labels.entity == "Entity"


# --- ensure_schema follows the names ----------------------------------------------------


class FakeDriver:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def execute_query(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None, list[str]]:
        statement = args[0] if args else kwargs.get("query_", kwargs.get("query"))
        if not isinstance(statement, str):
            statement = getattr(statement, "text", str(statement))
        self.queries.append(statement)
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


_LABEL = r"\(\s*\w+\s*:\s*`?(\w+)`?\s*\)"
CONSTRAINT_RE = re.compile(
    r"^\s*CREATE\s+CONSTRAINT\s+(?:\S+\s+)?IF\s+NOT\s+EXISTS\s+FOR\s+"
    + _LABEL
    + r"\s+REQUIRE\s+\(\s*\w+\s*\.\s*`?part`?\s*,\s*\w+\s*\.\s*`?key`?\s*\)\s+IS\s+UNIQUE\s*;?\s*$",
    re.IGNORECASE,
)
RANGE_RE = re.compile(
    r"^\s*CREATE\s+(?:RANGE\s+)?INDEX\s+(?:\S+\s+)?IF\s+NOT\s+EXISTS\s+FOR\s+"
    + _LABEL
    + r"\s+ON\s+\(\s*\w+\s*\.\s*`?part`?\s*,\s*\w+\s*\.\s*`?(\w+)`?\s*\)\s*;?\s*$",
    re.IGNORECASE,
)


def test_ensure_schema_uses_the_loaded_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = VectorRecorder()
    monkeypatch.setattr(graphrag_indexes, "create_vector_index", recorder)
    monkeypatch.setattr(schema_mod, "create_vector_index", recorder, raising=False)
    schema = load_schema(_custom_file(tmp_path))
    driver = FakeDriver()
    ensure_schema(driver, schema, 12, database="db")  # type: ignore[arg-type]

    constraints: list[str] = []
    ranges: list[tuple[str, str]] = []
    for statement in driver.queries:
        c = CONSTRAINT_RE.match(statement)
        r = RANGE_RE.match(statement)
        assert c or r, f"unexpected statement: {statement!r}"
        if c:
            constraints.append(c.group(1))
        elif r:
            ranges.append((r.group(1), r.group(2)))
    assert constraints == ["Thing"]
    assert len(ranges) == len(set(ranges))
    assert set(ranges) == {
        ("Thing", "state"),
        ("Thing", "digest"),
        ("Thing", "sort"),
        ("Thing", "tier"),
    }
    keys = [(c["label"], c["embedding_property"]) for c in recorder.calls]
    assert len(keys) == len(set(keys))
    assert set(keys) == {("Thing", "vec"), ("Thing", "vec_code"), ("Loose", "vec")}
    for call in recorder.calls:
        assert call["dimensions"] == 12
        assert call["neo4j_database"] == "db"
    for statement in driver.queries:
        assert not re.search(r"\.\s*`?(scope|id)`?\b", statement), statement


# --- no graph name as a string literal in schema.py / store.py ----------------------------


def _docstring_ids(tree: ast.AST) -> set[int]:
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0].value))
    return out


def _string_constants(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = _docstring_ids(tree)
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs
    ]


ALL_NAMES = {name for _s, _r, name in REPO_PAIRS}
WORD_NAMES = set(REPO_NAMES["labels"].values()) | set(REPO_NAMES["relationships"].values())


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_graph_name_literal(path: Path) -> None:
    found = [(line, value) for line, value in _string_constants(path) if value in ALL_NAMES]
    assert found == [], f"graph names spelled as literals in {path.name}: {found}"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_label_or_relationship_inside_literals(path: Path) -> None:
    found = [
        (line, name)
        for line, value in _string_constants(path)
        for name in WORD_NAMES
        if re.search(rf"\b{re.escape(name)}\b", value)
    ]
    assert found == [], f"labels or relationship types inside literals in {path.name}: {found}"
