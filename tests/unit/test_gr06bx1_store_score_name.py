"""GR-06bx1 acceptance: store.py reads the vector score through the typed property name.

``store.py`` spells no name of the repository schema's ``names`` section (not even
``"score"``) as a string constant; the vector searches read ``record[names.properties.score]``;
the two pinned name-list tests expect the names ``config/graph_schema.yaml`` now lists.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

from memini.infra.graph import store as store_mod
from memini.infra.graph.schema import load_schema

ROOT = Path(__file__).resolve().parents[2]
REPO_SCHEMA = ROOT / "config" / "graph_schema.yaml"
STORE = ROOT / "src" / "memini" / "infra" / "graph" / "store.py"
ENSURE_TEST = ROOT / "tests" / "unit" / "test_graph_schema_ensure.py"
NAMES_TEST = ROOT / "tests" / "unit" / "test_graph_schema_names.py"
SECTIONS = ("labels", "relationships", "properties", "statuses")
NEW_PROPERTIES = ("name", "dirty", "score", "embedding_code", "code_score")


def _repo_names() -> dict[str, dict[str, str]]:
    data = yaml.safe_load(REPO_SCHEMA.read_text(encoding="utf-8"))
    return {s: {str(k): str(v) for k, v in data["names"][s].items()} for s in SECTIONS}


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


# --- store.py spells no schema name ------------------------------------------------------


def test_store_has_no_score_string_constant() -> None:
    values = [value for _line, value in _string_constants(STORE)]
    assert "score" not in values


def test_store_has_no_schema_name_string_constant() -> None:
    forbidden = {name for roles in _repo_names().values() for name in roles.values()}
    assert "score" in forbidden
    found = [(line, value) for line, value in _string_constants(STORE) if value in forbidden]
    assert found == [], found


# --- vector search with a fake VectorRetriever ----------------------------------------------


class _Result:
    def __init__(self, records: list[Any]) -> None:
        self.records = records


class _FakeRetriever:
    """Stands in for VectorRetriever; returns fixed records out of order."""

    instances: ClassVar[list[_FakeRetriever]] = []
    records: ClassVar[list[Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        _FakeRetriever.instances.append(self)

    def get_search_results(self, *args: Any, **kwargs: Any) -> _Result:
        self.calls.append(dict(kwargs))
        return _Result(list(_FakeRetriever.records))


@pytest.fixture
def fake_retriever(monkeypatch: pytest.MonkeyPatch) -> type[_FakeRetriever]:
    props = load_schema().names.properties
    key, score = props.id, props.score
    _FakeRetriever.instances = []
    _FakeRetriever.records = [
        {"node": {key: "b"}, score: 0.5},
        {"node": {key: 7}, score: 0.9},
        {"node": {key: "c"}, score: 0.1},
        {"node": {key: "a"}, score: 0.7},
    ]
    monkeypatch.setattr(store_mod, "VectorRetriever", _FakeRetriever)
    return _FakeRetriever


def _check_pairs(hits: list[tuple[str, float]], expected: list[tuple[str, float]]) -> None:
    assert hits == expected
    for ident, value in hits:
        assert type(ident) is str
        assert type(value) is float


def test_search_entities_reads_typed_score(fake_retriever: type[_FakeRetriever]) -> None:
    hits = store_mod.search_entities(object(), "idx", "s1", [0.1, 0.2], top_k=3)  # type: ignore[arg-type]
    _check_pairs(hits, [("7", 0.9), ("a", 0.7), ("b", 0.5)])
    assert fake_retriever.instances


def test_search_entities_default_top_k_keeps_all(fake_retriever: type[_FakeRetriever]) -> None:
    hits = store_mod.search_entities(object(), "idx", "s1", [0.1, 0.2])  # type: ignore[arg-type]
    _check_pairs(hits, [("7", 0.9), ("a", 0.7), ("b", 0.5), ("c", 0.1)])


def test_search_communities_reads_typed_score(fake_retriever: type[_FakeRetriever]) -> None:
    hits = store_mod.search_communities(
        object(),  # type: ignore[arg-type]
        "cidx",
        "s1",
        [0.3],
        top_k=2,
        level=1,
    )
    _check_pairs(hits, [("7", 0.9), ("a", 0.7)])
    assert fake_retriever.instances


# --- the repository schema file ----------------------------------------------------------


def test_repo_community_ends_with_dirty() -> None:
    schema = load_schema()
    assert schema.nodes["Community"][-1] == "dirty"


def test_repo_names_properties_has_new_roles() -> None:
    props = load_schema().names.properties
    for role in NEW_PROPERTIES:
        assert getattr(props, role) == role


# --- the pinned test literals ----------------------------------------------------------------


def _module_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _assigned(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return node.value
    raise AssertionError(f"{name} not assigned at module level")


def _value(node: ast.expr, env: dict[str, Any]) -> Any:
    """Evaluate a literal that may star-expand a known module-level tuple."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        items: list[Any] = []
        for elt in node.elts:
            if isinstance(elt, ast.Starred):
                items.extend(_value(elt.value, env))
            else:
                items.append(_value(elt, env))
        if isinstance(node, ast.Tuple):
            return tuple(items)
        if isinstance(node, ast.Set):
            return set(items)
        return items
    if isinstance(node, ast.Dict):
        out: dict[Any, Any] = {}
        for k, v in zip(node.keys, node.values, strict=True):
            assert k is not None
            out[_value(k, env)] = _value(v, env)
        return out
    raise AssertionError(f"unexpected literal node {ast.dump(node)}")


BASE = ("scope", "updated_by", "updated_at")
OLD_REPO_NODES = {
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
    "RelationType": (*BASE, "name"),
}
OLD_REPO_RELATIONSHIPS = {
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
OLD_REPO_VALUES = {
    ("Artifact", "status"): ("active", "superseded", "deleted"),
    ("Entity", "status"): ("active", "superseded", "deleted"),
    ("RELATED_TO", "kind"): ("extracted", "sequence", "label", "cross_domain", "shared"),
    ("CODE_LINK", "kind"): ("define", "call", "import"),
}


def _ensure_literals() -> dict[str, Any]:
    tree = _module_tree(ENSURE_TEST)
    env: dict[str, Any] = {}
    env["BASE"] = _value(_assigned(tree, "BASE"), env)
    for name in ("REPO_NODES", "REPO_RELATIONSHIPS", "REPO_VALUES"):
        env[name] = _value(_assigned(tree, name), env)
    return env


def test_ensure_test_community_literal_lists_dirty() -> None:
    env = _ensure_literals()
    assert env["BASE"] == BASE
    assert env["REPO_NODES"]["Community"] == (
        *BASE,
        "id",
        "level",
        "summary",
        "embedding",
        "dirty",
    )


def test_ensure_test_other_literals_unchanged() -> None:
    env = _ensure_literals()
    nodes = dict(env["REPO_NODES"])
    community = nodes.pop("Community")
    assert community
    assert nodes == OLD_REPO_NODES
    assert list(env["REPO_NODES"]) == [
        "Project",
        "Domain",
        "Artifact",
        "Chunk",
        "Entity",
        "Community",
        "RelationType",
    ]
    assert env["REPO_RELATIONSHIPS"] == OLD_REPO_RELATIONSHIPS
    assert env["REPO_VALUES"] == OLD_REPO_VALUES


def test_ensure_test_literal_matches_repo_file() -> None:
    env = _ensure_literals()
    schema = load_schema(REPO_SCHEMA)
    assert dict(schema.nodes) == env["REPO_NODES"]


OLD_LABELS = {"project", "domain", "artifact", "chunk", "entity", "community", "relation_type"}
OLD_RELATIONSHIPS = {
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
}
OLD_STATUSES = {"active", "superseded", "deleted"}
OLD_PROPERTIES = {
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
}


def _expected_roles() -> dict[str, set[str]]:
    value = _value(_assigned(_module_tree(NAMES_TEST), "EXPECTED_ROLES"), {})
    assert isinstance(value, dict)
    return value


def test_names_test_expected_roles_match_repo_file() -> None:
    roles = _expected_roles()
    repo = _repo_names()
    assert set(roles) == set(SECTIONS)
    for section in SECTIONS:
        assert set(roles[section]) == set(repo[section]), section


def test_names_test_expected_roles_other_sections_unchanged() -> None:
    roles = _expected_roles()
    assert set(roles["labels"]) == OLD_LABELS
    assert set(roles["relationships"]) == OLD_RELATIONSHIPS
    assert set(roles["statuses"]) == OLD_STATUSES
    assert set(roles["properties"]) == OLD_PROPERTIES | set(NEW_PROPERTIES)


# --- the pinned test functions keep every assertion ------------------------------------------


def _asserts(path: Path, func: str) -> list[str]:
    for node in _module_tree(path).body:
        if isinstance(node, ast.FunctionDef) and node.name == func:
            return [ast.unparse(n) for n in ast.walk(node) if isinstance(n, ast.Assert)]
    raise AssertionError(f"{func} missing from {path.name}")


def test_default_reads_repo_file_keeps_assertions() -> None:
    found = _asserts(ENSURE_TEST, "test_load_schema_default_reads_repo_file")
    for expected in (
        "assert isinstance(schema, GraphSchema)",
        "assert dict(schema.nodes) == REPO_NODES",
        "assert list(schema.nodes) == list(REPO_NODES)",
        "assert dict(schema.relationships) == REPO_RELATIONSHIPS",
        "assert dict(schema.values) == REPO_VALUES",
    ):
        assert expected in found, expected


def test_reads_every_repo_name_keeps_assertions() -> None:
    found = _asserts(NAMES_TEST, "test_load_schema_reads_every_repo_name")
    for expected in (
        "assert set(repo[section]) == EXPECTED_ROLES[section]",
        "assert isinstance(value, str)",
        "assert value == name, (section, role)",
    ):
        assert expected in found, expected
