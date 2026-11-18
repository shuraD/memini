"""GR-06b acceptance: every role of the schema's ``names`` section has a typed field.

GraphProperties gains name, dirty, embedding_code, score and code_score, and ensure_schema
range-indexes the property named by ``names.properties.name`` instead of a ``"name"`` literal
(owner 2026-11-17, D-10).
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import re
from pathlib import Path
from typing import Any

import neo4j_graphrag.indexes as graphrag_indexes
import pytest
import yaml

from memini.infra.graph import schema as schema_mod
from memini.infra.graph.schema import (
    GraphLabels,
    GraphProperties,
    GraphRelationships,
    GraphStatuses,
    ensure_schema,
    load_schema,
)

ROOT = Path(__file__).resolve().parents[2]
REPO_SCHEMA = ROOT / "config" / "graph_schema.yaml"
SCHEMA_SOURCE = ROOT / "src" / "memini" / "infra" / "graph" / "schema.py"
NEW_ROLES = ("name", "dirty", "embedding_code", "score", "code_score")


def _repo_names() -> dict[str, dict[str, str]]:
    data = yaml.safe_load(REPO_SCHEMA.read_text(encoding="utf-8"))
    return data["names"]


# --- load_schema -----------------------------------------------------------------------------


def test_repo_new_property_roles_are_loaded() -> None:
    props = load_schema().names.properties
    assert props.name == "name"
    assert props.dirty == "dirty"
    assert props.embedding_code == "embedding_code"
    assert props.score == "score"
    assert props.code_score == "code_score"


@pytest.mark.parametrize("role", NEW_ROLES)
def test_repo_new_property_role_matches_file(role: str) -> None:
    assert (
        getattr(load_schema(REPO_SCHEMA).names.properties, role)
        == (_repo_names()["properties"][role])
    )


@pytest.mark.parametrize(
    ("cls", "section"),
    [
        (GraphLabels, "labels"),
        (GraphRelationships, "relationships"),
        (GraphProperties, "properties"),
        (GraphStatuses, "statuses"),
    ],
)
def test_fields_match_every_role_in_file(cls: type, section: str) -> None:
    field_names = {f.name for f in dataclasses.fields(cls)}
    assert field_names == set(_repo_names()[section])


def _write(tmp_path: Path, doc: dict[str, Any]) -> Path:
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def test_custom_new_property_names_come_from_file(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "nodes": {"Item": {"properties": ["tenant", "key"]}},
            "names": {
                "properties": {
                    "scope": "tenant",
                    "id": "key",
                    "name": "title",
                    "dirty": "stale",
                    "embedding_code": "vec_src",
                    "score": "weight",
                    "code_score": "src_weight",
                }
            },
        },
    )
    props = load_schema(path).names.properties
    assert props.name == "title"
    assert props.dirty == "stale"
    assert props.embedding_code == "vec_src"
    assert props.score == "weight"
    assert props.code_score == "src_weight"
    assert props.scope == "tenant"
    assert props.id == "key"


def test_missing_new_roles_are_named_by_role(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        {
            "nodes": {"Item": {"properties": ["scope", "id"]}},
            "names": {"properties": {"scope": "scope", "id": "id"}},
        },
    )
    props = load_schema(path).names.properties
    for role in NEW_ROLES:
        assert getattr(props, role) == role


@pytest.mark.parametrize("role", NEW_ROLES)
def test_new_roles_are_frozen(role: str) -> None:
    props = load_schema().names.properties
    with pytest.raises(AttributeError):
        setattr(props, role, "other")
    assert getattr(props, role) == role


# --- ensure_schema ---------------------------------------------------------------------------


class FakeDriver:
    """Records every execute_query statement; refuses sessions."""

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


@pytest.fixture
def vectors(monkeypatch: pytest.MonkeyPatch) -> VectorRecorder:
    recorder = VectorRecorder()
    monkeypatch.setattr(graphrag_indexes, "create_vector_index", recorder)
    monkeypatch.setattr(schema_mod, "create_vector_index", recorder, raising=False)
    return recorder


_LABEL = r"\(\s*\w+\s*:\s*`?(\w+)`?\s*\)"
_PROP = r"\w+\s*\.\s*`?(\w+)`?"
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


def _ranges(driver: FakeDriver) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for statement in driver.queries:
        m = RANGE_RE.match(statement)
        if m:
            out.append((m.group(1), m.group(2), m.group(3)))
    return out


def test_repo_range_indexes_unchanged(vectors: VectorRecorder) -> None:
    driver = FakeDriver()
    ensure_schema(driver, load_schema(), 8)  # type: ignore[arg-type]
    ranges = _ranges(driver)
    assert len(ranges) == len(set(ranges))
    assert set(ranges) == {
        ("Project", "scope", "name"),
        ("Domain", "scope", "name"),
        ("RelationType", "scope", "name"),
        ("Artifact", "scope", "type"),
        ("Artifact", "scope", "content_hash"),
        ("Artifact", "scope", "status"),
        ("Entity", "scope", "name"),
        ("Entity", "scope", "type"),
        ("Entity", "scope", "status"),
        ("Community", "scope", "level"),
    }


def test_repo_vector_indexes_unchanged(vectors: VectorRecorder) -> None:
    driver = FakeDriver()
    ensure_schema(driver, load_schema(), 8)  # type: ignore[arg-type]
    keys = [(c["label"], c["embedding_property"]) for c in vectors.calls]
    assert len(keys) == len(set(keys))
    assert set(keys) == {
        ("Chunk", "embedding"),
        ("Entity", "embedding"),
        ("Entity", "embedding_code"),
        ("Community", "embedding"),
    }


def test_range_index_follows_renamed_name_role(tmp_path: Path, vectors: VectorRecorder) -> None:
    path = _write(
        tmp_path,
        {
            "nodes": {
                "Item": {"properties": ["tenant", "key", "title", "name", "stale", "vec"]},
            },
            "names": {
                "properties": {
                    "scope": "tenant",
                    "id": "key",
                    "name": "title",
                    "dirty": "stale",
                    "embedding": "vec",
                }
            },
        },
    )
    driver = FakeDriver()
    ensure_schema(driver, load_schema(path), 8)  # type: ignore[arg-type]
    assert set(_ranges(driver)) == {("Item", "tenant", "title")}
    keys = {(c["label"], c["embedding_property"]) for c in vectors.calls}
    assert keys == {("Item", "vec")}


# --- source ----------------------------------------------------------------------------------


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def test_no_new_role_name_literal_in_schema_source() -> None:
    tree = ast.parse(SCHEMA_SOURCE.read_text(encoding="utf-8"))
    docs = _docstring_ids(tree)
    found = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docs
        and node.value in NEW_ROLES
    ]
    assert found == []
