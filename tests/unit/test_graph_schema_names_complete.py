"""GR-06b acceptance: GraphProperties names every property role of the schema file.

``GraphProperties`` gains ``name``, ``dirty``, ``embedding_code``, ``score`` and ``code_score``;
``ensure_schema`` range-indexes the property named by ``names.properties.name`` instead of a
``"name"`` literal, with no other change.
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
from memini.infra.graph.schema import ensure_schema, load_schema

ROOT = Path(__file__).resolve().parents[2]
REPO_SCHEMA = ROOT / "config" / "graph_schema.yaml"
SCHEMA_PY = ROOT / "src" / "memini" / "infra" / "graph" / "schema.py"
NEW_ROLES = ("name", "dirty", "embedding_code", "score", "code_score")
SECTION_CLASSES = {
    "labels": "GraphLabels",
    "relationships": "GraphRelationships",
    "properties": "GraphProperties",
    "statuses": "GraphStatuses",
}

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


class FakeDriver:
    """Records every execute_query statement; refuses sessions."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute_query(self, *args: Any, **kwargs: Any) -> tuple[list[Any], None, list[str]]:
        statement = args[0] if args else kwargs.get("query_", kwargs.get("query"))
        if not isinstance(statement, str):
            statement = getattr(statement, "text", str(statement))
        self.statements.append(statement)
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


def _ranges(driver: FakeDriver) -> list[tuple[str, str, str]]:
    """Return (label, first property, second property) of every range index statement."""
    out: list[tuple[str, str, str]] = []
    for statement in driver.statements:
        if CONSTRAINT_RE.match(statement):
            continue
        r = RANGE_RE.match(statement)
        assert r, f"unexpected statement: {statement!r}"
        out.append((r.group(1), r.group(2), r.group(3)))
    return out


def _repo_names() -> dict[str, dict[str, str]]:
    data = yaml.safe_load(REPO_SCHEMA.read_text(encoding="utf-8"))
    return {k: {str(r): str(v) for r, v in data["names"][k].items()} for k in SECTION_CLASSES}


# --- load_schema -------------------------------------------------------------------


def test_repo_schema_names_the_five_new_property_roles() -> None:
    p = load_schema(REPO_SCHEMA).names.properties
    assert p.name == "name"
    assert p.dirty == "dirty"
    assert p.embedding_code == "embedding_code"
    assert p.score == "score"
    assert p.code_score == "code_score"


def test_default_load_schema_names_the_five_new_property_roles() -> None:
    p = load_schema().names.properties
    assert (p.name, p.dirty, p.embedding_code, p.score, p.code_score) == NEW_ROLES


@pytest.mark.parametrize("section", list(SECTION_CLASSES))
def test_dataclass_fields_equal_file_roles(section: str) -> None:
    cls = getattr(schema_mod, SECTION_CLASSES[section])
    field_names = {f.name for f in dataclasses.fields(cls)}
    assert field_names == set(_repo_names()[section])


def test_new_property_fields_are_str() -> None:
    p = load_schema(REPO_SCHEMA).names.properties
    for role in NEW_ROLES:
        assert isinstance(getattr(p, role), str)


def test_custom_file_renames_the_new_roles(tmp_path: Path) -> None:
    path = tmp_path / "custom.yaml"
    path.write_text(
        "nodes: {}\n"
        "relationships: {}\n"
        "names:\n"
        "  properties:\n"
        "    name: title\n"
        "    dirty: stale\n"
        "    embedding_code: vec_code\n"
        "    score: weight\n"
        "    code_score: code_weight\n",
        encoding="utf-8",
    )
    p = load_schema(path).names.properties
    assert p.name == "title"
    assert p.dirty == "stale"
    assert p.embedding_code == "vec_code"
    assert p.score == "weight"
    assert p.code_score == "code_weight"
    assert p.scope == "scope"
    assert p.id == "id"


def test_file_omitting_new_roles_names_them_by_role(tmp_path: Path) -> None:
    path = tmp_path / "partial.yaml"
    path.write_text(
        "nodes: {}\nrelationships: {}\nnames:\n  properties:\n    scope: tenant\n    id: key\n",
        encoding="utf-8",
    )
    p = load_schema(path).names.properties
    assert p.scope == "tenant"
    assert p.id == "key"
    for role in NEW_ROLES:
        assert getattr(p, role) == role


def test_file_without_names_section_names_new_roles_by_role(tmp_path: Path) -> None:
    path = tmp_path / "bare.yaml"
    path.write_text("nodes: {}\nrelationships: {}\n", encoding="utf-8")
    p = load_schema(path).names.properties
    for role in NEW_ROLES:
        assert getattr(p, role) == role


@pytest.mark.parametrize("role", NEW_ROLES)
def test_assigning_a_new_role_raises(role: str) -> None:
    p = load_schema(REPO_SCHEMA).names.properties
    with pytest.raises(AttributeError):
        setattr(p, role, "other")
    assert getattr(p, role) == role


# --- ensure_schema -----------------------------------------------------------------


def test_repo_range_indexes_are_exact(vectors: VectorRecorder) -> None:
    driver = FakeDriver()
    ensure_schema(driver, load_schema(REPO_SCHEMA), dimension=8)
    ranges = _ranges(driver)
    assert all(first == "scope" for _label, first, _second in ranges)
    pairs = sorted((label, second) for label, _first, second in ranges)
    expected = sorted(
        [
            ("Project", "name"),
            ("Domain", "name"),
            ("RelationType", "name"),
            ("Entity", "name"),
            ("Artifact", "type"),
            ("Artifact", "content_hash"),
            ("Artifact", "status"),
            ("Entity", "type"),
            ("Entity", "status"),
            ("Community", "level"),
        ]
    )
    assert pairs == expected


def test_repo_unranged_roles_get_no_range_index(vectors: VectorRecorder) -> None:
    driver = FakeDriver()
    ensure_schema(driver, load_schema(REPO_SCHEMA), dimension=8)
    indexed = {second for _label, _first, second in _ranges(driver)}
    for prop in ("dirty", "score", "code_score", "embedding_code", "embedding"):
        assert prop not in indexed


def test_repo_vector_indexes_unchanged(vectors: VectorRecorder) -> None:
    schema = load_schema(REPO_SCHEMA)
    ensure_schema(FakeDriver(), schema, dimension=8)
    got = sorted((c["label"], c["embedding_property"]) for c in vectors.calls)
    expected = sorted(
        (label, prop)
        for label, props in schema.nodes.items()
        for prop in props
        if prop.startswith("embedding")
    )
    assert got == expected
    assert ("Entity", "embedding_code") in got
    assert all(c["dimensions"] == 8 for c in vectors.calls)


def test_custom_name_role_moves_the_range_index(tmp_path: Path, vectors: VectorRecorder) -> None:
    path = tmp_path / "custom.yaml"
    path.write_text(
        "nodes:\n"
        "  Widget:\n"
        "    properties: [scope, id, title, name, stale, dirty, kind, level]\n"
        "relationships: {}\n"
        "names:\n"
        "  properties:\n"
        "    name: title\n"
        "    dirty: stale\n",
        encoding="utf-8",
    )
    driver = FakeDriver()
    ensure_schema(driver, load_schema(path), dimension=4)
    ranges = _ranges(driver)
    assert all(first == "scope" for _label, first, _second in ranges)
    pairs = sorted((label, second) for label, _first, second in ranges)
    assert pairs == [("Widget", "level"), ("Widget", "title")]


def test_custom_literal_name_property_is_not_ranged(
    tmp_path: Path, vectors: VectorRecorder
) -> None:
    path = tmp_path / "custom.yaml"
    path.write_text(
        "nodes:\n"
        "  Gadget:\n"
        "    properties: [scope, name, label_text]\n"
        "relationships: {}\n"
        "names:\n"
        "  properties:\n"
        "    name: label_text\n",
        encoding="utf-8",
    )
    driver = FakeDriver()
    ensure_schema(driver, load_schema(path), dimension=4)
    pairs = [(label, second) for label, _first, second in _ranges(driver)]
    assert pairs == [("Gadget", "label_text")]


# --- source ------------------------------------------------------------------------


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ) and getattr(node, "body", None):
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                ids.add(id(first.value))
    return ids


def test_schema_source_has_no_new_role_literal() -> None:
    tree = ast.parse(SCHEMA_PY.read_text(encoding="utf-8"), filename=str(SCHEMA_PY))
    skip = _docstring_ids(tree)
    found = [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in skip
        and node.value in NEW_ROLES
    ]
    assert found == []
