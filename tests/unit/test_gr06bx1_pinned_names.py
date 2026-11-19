"""GR-06bx1 acceptance: store.py reads the score by typed name; pinned lists follow the file."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
import yaml

from memini.infra.graph import store as store_mod
from memini.infra.graph.schema import load_schema

ROOT = Path(__file__).resolve().parents[2]
REPO_SCHEMA = ROOT / "config" / "graph_schema.yaml"
STORE = ROOT / "src" / "memini" / "infra" / "graph" / "store.py"
ENSURE_TEST = ROOT / "tests" / "unit" / "test_graph_schema_ensure.py"
NAMES_TEST = ROOT / "tests" / "unit" / "test_graph_schema_names.py"
NEW_PROPERTY_ROLES = {"name", "dirty", "score", "embedding_code", "code_score"}
OLD_PROPERTY_ROLES = {
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


def _yaml() -> dict[str, Any]:
    return yaml.safe_load(REPO_SCHEMA.read_text(encoding="utf-8"))


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _string_constants(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = _docstring_ids(tree)
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip
    ]


def _assigned(path: Path, name: str) -> ast.expr:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name:
                assert node.value is not None
                return node.value
    raise AssertionError(f"{name} not assigned in {path.name}")


def _function(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} missing from {path.name}")


def _tuple_strings(value: ast.expr) -> list[str]:
    assert isinstance(value, ast.Tuple), ast.unparse(value)
    return [e.value for e in value.elts if isinstance(e, ast.Constant)]


def _dict_entries(value: ast.expr) -> dict[str, ast.expr]:
    assert isinstance(value, ast.Dict)
    out: dict[str, ast.expr] = {}
    for key, item in zip(value.keys, value.values, strict=True):
        assert isinstance(key, ast.Constant)
        out[str(key.value)] = item
    return out


# --- store.py: no 'score' literal; search still works --------------------------------


def test_store_has_no_score_literal() -> None:
    found = [(line, value) for line, value in _string_constants(STORE) if value == "score"]
    assert found == [], f"'score' spelled as a literal in store.py: {found}"


def test_store_has_no_graph_name_literal() -> None:
    names = _yaml()["names"]
    forbidden = {str(v) for section in names.values() for v in section.values()}
    found = [(line, value) for line, value in _string_constants(STORE) if value in forbidden]
    assert found == [], f"graph names spelled as literals in store.py: {found}"


class _Result:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records


def _fake_retriever(records: list[dict[str, Any]], seen: list[dict[str, Any]]) -> type:
    class FakeRetriever:
        def __init__(self, driver: Any, index_name: str, **kwargs: Any) -> None:
            seen.append({"index_name": index_name, **kwargs})

        def get_search_results(self, **kwargs: Any) -> _Result:
            seen.append(kwargs)
            return _Result(records)

    return FakeRetriever


def _records() -> list[dict[str, Any]]:
    p = load_schema().names.properties
    return [
        {"node": {p.id: "a"}, p.score: 0.2},
        {"node": {p.id: "b"}, p.score: 0.9},
        {"node": {p.id: "c"}, p.score: 0.5},
    ]


def test_search_entities_reads_score_best_first(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(store_mod, "VectorRetriever", _fake_retriever(_records(), seen))
    hits = store_mod.search_entities(object(), "idx", "s1", [1.0, 0.0], top_k=2)
    assert hits == [("b", 0.9), ("c", 0.5)]
    assert all(isinstance(i, str) and isinstance(s, float) for i, s in hits)
    assert seen[0]["index_name"] == "idx"


def test_search_communities_reads_score_best_first(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(store_mod, "VectorRetriever", _fake_retriever(_records(), seen))
    hits = store_mod.search_communities(object(), "cidx", "s1", [0.0, 1.0], top_k=3)
    assert hits == [("b", 0.9), ("c", 0.5), ("a", 0.2)]


# --- the repository file the pinned tests must follow ---------------------------------


def test_repo_schema_has_the_new_names() -> None:
    schema = load_schema()
    assert schema.nodes["Community"][-1] == "dirty"
    p = schema.names.properties
    for role in NEW_PROPERTY_ROLES:
        assert getattr(p, role) == role


# --- pinned test: test_graph_schema_ensure.py ------------------------------------------


def test_ensure_test_community_lists_dirty() -> None:
    nodes = _dict_entries(_assigned(ENSURE_TEST, "REPO_NODES"))
    community = nodes["Community"]
    assert isinstance(community, ast.Tuple)
    assert isinstance(community.elts[0], ast.Starred)
    assert ast.unparse(community.elts[0]) == "*BASE"
    assert _tuple_strings(community) == ["id", "level", "summary", "embedding", "dirty"]
    base = ("scope", "updated_by", "updated_at")
    assert [*base, *_tuple_strings(community)] == _yaml()["nodes"]["Community"]["properties"]


def test_ensure_test_other_expectations_unchanged() -> None:
    nodes = _dict_entries(_assigned(ENSURE_TEST, "REPO_NODES"))
    assert list(nodes) == [
        "Project",
        "Domain",
        "Artifact",
        "Chunk",
        "Entity",
        "Community",
        "RelationType",
    ]
    data = _yaml()["nodes"]
    base = ["scope", "updated_by", "updated_at"]
    for label, value in nodes.items():
        assert [*base, *_tuple_strings(value)] == data[label]["properties"], label
    rels = _dict_entries(_assigned(ENSURE_TEST, "REPO_RELATIONSHIPS"))
    assert {k: tuple(_tuple_strings(v)) for k, v in rels.items()} == {
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
    assert ast.literal_eval(_assigned(ENSURE_TEST, "REPO_VALUES")) == {
        ("Artifact", "status"): ("active", "superseded", "deleted"),
        ("Entity", "status"): ("active", "superseded", "deleted"),
        ("RELATED_TO", "kind"): ("extracted", "sequence", "label", "cross_domain", "shared"),
        ("CODE_LINK", "kind"): ("define", "call", "import"),
    }


def test_ensure_pinned_test_keeps_its_assertions() -> None:
    text = _function(ENSURE_TEST, "test_load_schema_default_reads_repo_file")
    for line in (
        "schema = load_schema()",
        "assert isinstance(schema, GraphSchema)",
        "assert dict(schema.nodes) == REPO_NODES",
        "assert list(schema.nodes) == list(REPO_NODES)",
        "assert dict(schema.relationships) == REPO_RELATIONSHIPS",
        "assert dict(schema.values) == REPO_VALUES",
    ):
        assert line in text, line


# --- pinned test: test_graph_schema_names.py -------------------------------------------


def test_names_test_expected_roles_follow_the_file() -> None:
    expected = ast.literal_eval(_assigned(NAMES_TEST, "EXPECTED_ROLES"))
    names = _yaml()["names"]
    assert set(expected) == {"labels", "relationships", "properties", "statuses"}
    for section, roles in expected.items():
        assert roles == {str(k) for k in names[section]}, section
    assert expected["properties"] == OLD_PROPERTY_ROLES | NEW_PROPERTY_ROLES


def test_names_test_other_sections_unchanged() -> None:
    expected = ast.literal_eval(_assigned(NAMES_TEST, "EXPECTED_ROLES"))
    assert expected["labels"] == {
        "project",
        "domain",
        "artifact",
        "chunk",
        "entity",
        "community",
        "relation_type",
    }
    assert expected["relationships"] == {
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
    assert expected["statuses"] == {"active", "superseded", "deleted"}


def test_names_pinned_test_keeps_its_assertions() -> None:
    text = _function(NAMES_TEST, "test_load_schema_reads_every_repo_name")
    for line in (
        "schema = load_schema()",
        "repo = _repo_names()",
        "for section in SECTIONS:",
        "assert set(repo[section]) == EXPECTED_ROLES[section]",
        "part = getattr(schema.names, section)",
        "value = getattr(part, role)",
        "assert isinstance(value, str)",
        "assert value == name, (section, role)",
    ):
        assert line in text, line
