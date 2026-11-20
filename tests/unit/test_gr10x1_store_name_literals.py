"""GR-10x1 acceptance: store.py spells no graph name as a literal; filtered search unchanged."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from memini.infra.graph import store as store_mod

ROOT = Path(__file__).resolve().parents[2]
REPO_SCHEMA = ROOT / "config" / "graph_schema.yaml"
STORE = ROOT / "src" / "memini" / "infra" / "graph" / "store.py"
PARAM = re.compile(r"\$([A-Za-z_]\w*)")


def _names() -> dict[str, dict[str, str]]:
    return yaml.safe_load(REPO_SCHEMA.read_text(encoding="utf-8"))["names"]


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


def _string_constants() -> list[tuple[int, str]]:
    # ast.walk visits the Constant parts of every f-string (JoinedStr) as well.
    tree = ast.parse(STORE.read_text(encoding="utf-8"))
    skip = _docstring_ids(tree)
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip
    ]


# --- static scan ---------------------------------------------------------------------------


def test_no_str_constant_equals_a_graph_name() -> None:
    forbidden = {str(v) for section in _names().values() for v in section.values()}
    found = [(line, value) for line, value in _string_constants() if value in forbidden]
    assert found == [], f"graph names spelled as literals in store.py: {found}"


def test_no_level_or_dirty_literal() -> None:
    found = [(line, v) for line, v in _string_constants() if v in {"level", "dirty"}]
    assert found == [], f"'level' or 'dirty' spelled as a literal in store.py: {found}"


def test_no_str_constant_contains_label_or_relationship_as_word() -> None:
    names = _names()
    words = [str(v) for v in names["labels"].values()]
    words += [str(v) for v in names["relationships"].values()]
    patterns = [(w, re.compile(rf"\b{re.escape(w)}\b")) for w in words]
    found = [
        (line, word, value)
        for line, value in _string_constants()
        for word, pattern in patterns
        if pattern.search(value)
    ]
    assert found == [], f"labels or relationship types inside literals in store.py: {found}"


def test_scan_sees_fstring_parts() -> None:
    # Sanity: the scan must include f-string parts, e.g. the Cypher SEARCH clause.
    assert any("VECTOR INDEX" in value for _, value in _string_constants())


# --- filtered search with a fake driver -------------------------------------------------------


class _FakeDriver:
    def __init__(self, found: list[list[Any]]) -> None:
        self.found = found
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def execute_query(self, query: str, params: Any = None, **kwargs: Any) -> Any:
        self.calls.append((query, dict(params or {}), kwargs))
        records = [{"seen": len(self.found), "found": list(self.found)}]
        return records, None, None


def _check_params(driver: _FakeDriver) -> tuple[str, dict[str, Any]]:
    assert driver.calls, "execute_query was not called"
    for query, params, _ in driver.calls:
        missing = sorted(set(PARAM.findall(query)) - set(params))
        assert missing == [], f"$params without a key in params: {missing}\n{query}"
    query, params, _ = driver.calls[-1]
    return query, params


def _values(params: dict[str, Any]) -> list[Any]:
    return list(params.values())


@pytest.mark.parametrize(
    ("level", "domain"),
    [(2, None), (None, "dom-a"), (3, "dom-b")],
)
def test_search_entities_filtered(level: int | None, domain: str | None) -> None:
    driver = _FakeDriver([["e1", 0.9], ["e2", 0.5]])
    result = store_mod.search_entities(
        driver, "idx", "scope-x", [0.1, 0.2], top_k=10, level=level, domain=domain
    )
    query, params = _check_params(driver)
    values = _values(params)
    assert "scope-x" in values
    if level is not None:
        assert level in values
    if domain is not None:
        assert domain in values
    assert "`Entity`" in query
    assert result == [("e1", 0.9), ("e2", 0.5)]
    assert all(isinstance(k, str) and isinstance(s, float) for k, s in result)


def test_search_entities_filtered_passes_database() -> None:
    driver = _FakeDriver([["e1", 0.7]])
    store_mod.search_entities(driver, "idx", "scope-x", [0.1], level=1, database="db-1")
    assert driver.calls[-1][2].get("database_") == "db-1"


@pytest.mark.parametrize("level", [None, 4])
def test_search_communities_with_domain(level: int | None) -> None:
    driver = _FakeDriver([["c1", 0.8]])
    result = store_mod.search_communities(
        driver, "idx", "scope-y", [0.3, 0.4], top_k=10, level=level, domain="dom-c"
    )
    query, params = _check_params(driver)
    values = _values(params)
    assert "scope-y" in values
    assert "dom-c" in values
    if level is not None:
        assert level in values
    for name in ("`Community`", "`Domain`", "`BELONGS_TO_COMMUNITY`"):
        assert name in query, f"{name} missing from query"
    assert result == [("c1", 0.8)]
