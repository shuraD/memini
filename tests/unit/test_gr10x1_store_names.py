"""GR-10x1: store.py spells no graph name as a literal; GR-10 filters behave as before."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from memini.infra.graph import store as store_mod

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "src" / "memini" / "infra" / "graph" / "store.py"
SCHEMA = ROOT / "config" / "graph_schema.yaml"


def _names() -> dict[str, dict[str, str]]:
    return yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))["names"]


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


def test_store_has_no_string_constant_equal_to_a_graph_name() -> None:
    forbidden = {str(v) for section in _names().values() for v in section.values()}
    assert "level" in forbidden
    found = [(line, v) for line, v in _string_constants(STORE) if v in forbidden]
    assert found == [], f"graph names spelled as literals in store.py: {found}"


def test_store_has_no_label_or_relationship_inside_a_string_constant() -> None:
    names = _names()
    words = set(names["labels"].values()) | set(names["relationships"].values())
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(" + "|".join(map(re.escape, sorted(words))) + r")(?![A-Za-z0-9_])"
    )
    found = [(line, v) for line, v in _string_constants(STORE) if pattern.search(v)]
    assert found == [], f"labels/relationships inside literals in store.py: {found}"


class _FakeDriver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute_query(self, query: str, params: dict[str, Any], **kwargs: Any) -> Any:
        self.calls.append((query, dict(params)))
        return [{"seen": 1, "found": [["k1", 0.75]]}], None, None


def _check_params(query: str, params: dict[str, Any]) -> None:
    used = set(re.findall(r"\$([A-Za-z_][A-Za-z0-9_]*)", query))
    assert used, query
    missing = used - set(params)
    assert missing == set(), f"query parameters without a value: {missing}"


@pytest.mark.parametrize(
    ("level", "domain"), [(2, None), (None, "dom-a"), (3, "dom-b")], ids=["lvl", "dom", "both"]
)
def test_search_entities_filtered_binds_every_query_parameter(
    level: int | None, domain: str | None
) -> None:
    driver = _FakeDriver()
    hits = store_mod.search_entities(
        driver, "idx", "s1", [1.0, 0.0], top_k=1, level=level, domain=domain
    )
    assert hits == [("k1", 0.75)]
    query, params = driver.calls[0]
    _check_params(query, params)
    values = list(params.values())
    assert "s1" in values
    if level is not None:
        assert level in values
    if domain is not None:
        assert domain in values
    assert f"`{_names()['labels']['entity']}`" in query


@pytest.mark.parametrize(("level", "domain"), [(None, "dom-a"), (1, "dom-b")], ids=["dom", "both"])
def test_search_communities_domain_binds_every_query_parameter(
    level: int | None, domain: str
) -> None:
    driver = _FakeDriver()
    hits = store_mod.search_communities(
        driver, "idx", "s1", [0.5, 0.5], top_k=1, level=level, domain=domain
    )
    assert hits == [("k1", 0.75)]
    query, params = driver.calls[0]
    _check_params(query, params)
    values = list(params.values())
    assert "s1" in values
    assert domain in values
    if level is not None:
        assert level in values
    names = _names()
    assert f"`{names['labels']['community']}`" in query
    assert f"`{names['labels']['domain']}`" in query
    assert f"`{names['relationships']['member_of']}`" in query
