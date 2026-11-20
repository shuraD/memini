"""GR-10 acceptance against the test Neo4j: level and domain filters of the vector searches,
the dirty ratio of stats, and graph names only from the typed schema names in store.py.

The Neo4j settings come only from NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DATABASE;
no host and no default is written here. Each test isolates itself with its own ``scope``
values (prefix ``qa-gr10-``), removed before and after the test. The vector indexes are
reused when present (their dimension is read back), else created.
"""

from __future__ import annotations

import ast
import os
import re
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import neo4j
import pytest
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.retrievers import VectorRetriever

from memini.infra.graph import store as store_mod
from memini.infra.graph.schema import load_schema
from memini.infra.graph.store import search_communities, search_entities, stats

PREFIX = "qa-gr10-"
DEFAULT_DIMENSION = 4

STATS_KEYS = {
    "by_status",
    "by_type",
    "relationships",
    "embedding_coverage",
    "communities",
    "levels",
    "avg_community_size",
    "dirty_ratio",
}


def _database() -> str | None:
    return os.environ.get("NEO4J_DATABASE") or None


class Probe:
    """A plain driver to seed the graph and read back what the store did."""

    def __init__(self) -> None:
        self.driver = neo4j.GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
        )

    def rows(self, query: str, **params: Any) -> list[dict[str, Any]]:
        records, _, _ = self.driver.execute_query(query, params, database_=_database())
        return [record.data() for record in records]

    def cleanup(self) -> None:
        self.driver.execute_query(
            "MATCH (n) WHERE n.scope STARTS WITH $prefix DETACH DELETE n",
            {"prefix": PREFIX},
            database_=_database(),
        )

    def vector_index(self, label: str, prop: str) -> tuple[str, int]:
        """Return (name, dimension) of the vector index on label.prop; create it when missing."""
        for _ in range(2):
            for row in self.rows(
                "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options "
                "RETURN name, type, labelsOrTypes, properties, options"
            ):
                props = row["properties"] or []
                if (
                    str(row["type"]).upper() == "VECTOR"
                    and row["labelsOrTypes"] == [label]
                    and props
                    and props[0] == prop
                ):
                    config = (row["options"] or {}).get("indexConfig") or {}
                    dim = int(config.get("vector.dimensions") or DEFAULT_DIMENSION)
                    self.driver.execute_query("CALL db.awaitIndexes(120)", database_=_database())
                    return str(row["name"]), dim
            create_vector_index(
                self.driver,
                name=f"{label.lower()}_{prop}",
                label=label,
                embedding_property=prop,
                dimensions=DEFAULT_DIMENSION,
                similarity_fn="cosine",
                fail_if_exists=False,
                neo4j_database=_database(),
            )
        raise AssertionError(f"no vector index on {label}.{prop}")


def vec(dim: int, *head: float) -> list[float]:
    """A vector of ``dim`` floats starting with ``head`` and padded with zeros."""
    assert dim >= len(head)
    return [float(x) for x in head] + [0.0] * (dim - len(head))


@pytest.fixture
def probe() -> Iterator[Probe]:
    p = Probe()
    p.cleanup()
    try:
        yield p
    finally:
        p.cleanup()
        p.driver.close()


@pytest.fixture
def driver(probe: Probe) -> neo4j.Driver:
    return probe.driver


@pytest.fixture
def search_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every VectorRetriever.get_search_results call (still performing it)."""
    calls: list[dict[str, Any]] = []
    original = VectorRetriever.get_search_results

    def spy(self: VectorRetriever, *args: Any, **kwargs: Any) -> Any:
        calls.append({"args": args, "kwargs": kwargs})
        return original(self, *args, **kwargs)

    monkeypatch.setattr(VectorRetriever, "get_search_results", spy)
    return calls


def _filters(call: dict[str, Any]) -> dict[str, Any]:
    """The filters a recorded get_search_results call was given."""
    if "filters" in call["kwargs"]:
        return dict(call["kwargs"]["filters"] or {})
    dicts = [a for a in call["args"] if isinstance(a, dict)]
    return dict(dicts[-1]) if dicts else {}


# --- the seeded graph ----------------------------------------------------------------------------
#
# Scope S: domains alpha and beta (plus a domain gamma of another scope that reaches e3).
#   alpha -> art-a -> ch-a -> e1, e2, e5 (deleted), e7
#   beta  -> art-b -> ch-b -> e3, e4
#   e6 has no domain.
#   Communities: k0a (L0: e1, e2, e5), k0b (L0: e3), k0c (L0: e4), k0d (L0: no member),
#                k1a (L1: k0a, k0b), k1b (L1: e6, e7, k0c), k2 (L2: k1a).
# Scope O: domain alpha -> x (active, nearest of all), community kx (L0: x).
#
# Query vector q = (1, 0, ...); a node at (1, t, ...) is nearer when t is smaller.

ENTITY_T = {
    "e1": 0.1,
    "e7": 0.15,
    "e3": 0.2,
    "e6": 0.3,
    "e2": 0.4,
    "e4": 0.5,
    "e5": 0.0,  # deleted
}
COMMUNITY_T = {
    "k2": 0.05,
    "k0b": 0.1,
    "k1b": 0.2,
    "k0a": 0.3,
    "k0d": 0.35,
    "k1a": 0.4,
    "k0c": 0.5,
}
COMMUNITY_LEVEL = {"k0a": 0, "k0b": 0, "k0c": 0, "k0d": 0, "k1a": 1, "k1b": 1, "k2": 2}


def _seed(probe: Probe, scope: str, other: str, edim: int, kdim: int) -> None:
    for node_id, t in ENTITY_T.items():
        status = "deleted" if node_id == "e5" else "active"
        probe.rows(
            "CREATE (:Entity {scope: $scope, id: $id, name: $id, type: 'concept', "
            "status: $status, embedding: $v})",
            scope=scope,
            id=node_id,
            status=status,
            v=vec(edim, 1.0, t),
        )
    for node_id, t in COMMUNITY_T.items():
        probe.rows(
            "CREATE (:Community {scope: $scope, id: $id, level: $level, embedding: $v})",
            scope=scope,
            id=node_id,
            level=COMMUNITY_LEVEL[node_id],
            v=vec(kdim, 1.0, t),
        )
    probe.rows(
        "CREATE (da:Domain {scope: $s, id: 'd-alpha', name: 'alpha'}) "
        "CREATE (db:Domain {scope: $s, id: 'd-beta', name: 'beta'}) "
        "CREATE (aa:Artifact {scope: $s, id: 'art-a', status: 'active'}) "
        "CREATE (ab:Artifact {scope: $s, id: 'art-b', status: 'active'}) "
        "CREATE (ca:Chunk {scope: $s, id: 'ch-a', content: 'a'}) "
        "CREATE (cb:Chunk {scope: $s, id: 'ch-b', content: 'b'}) "
        "CREATE (da)-[:CONTAINS]->(aa)-[:HAS_CHUNK]->(ca) "
        "CREATE (db)-[:CONTAINS]->(ab)-[:HAS_CHUNK]->(cb) "
        "WITH ca, cb "
        "MATCH (e:Entity {scope: $s}) WHERE e.id IN ['e1', 'e2', 'e5', 'e7'] "
        "CREATE (ca)-[:HAS_ENTITY]->(e) "
        "WITH DISTINCT cb "
        "MATCH (e:Entity {scope: $s}) WHERE e.id IN ['e3', 'e4'] "
        "CREATE (cb)-[:HAS_ENTITY]->(e)",
        s=scope,
    )
    for child, parent in [
        ("e1", "k0a"),
        ("e2", "k0a"),
        ("e5", "k0a"),
        ("e3", "k0b"),
        ("e4", "k0c"),
        ("e6", "k1b"),
        ("e7", "k1b"),
        ("k0a", "k1a"),
        ("k0b", "k1a"),
        ("k0c", "k1b"),
        ("k1a", "k2"),
    ]:
        probe.rows(
            "MATCH (c {scope: $s, id: $child}), (p:Community {scope: $s, id: $parent}) "
            "CREATE (c)-[:BELONGS_TO_COMMUNITY]->(p)",
            s=scope,
            child=child,
            parent=parent,
        )
    # Another scope: a domain gamma that reaches e3 of scope S, and an alpha domain with x.
    probe.rows(
        "CREATE (dg:Domain {scope: $o, id: 'd-gamma', name: 'gamma'}) "
        "CREATE (ag:Artifact {scope: $o, id: 'art-g', status: 'active'}) "
        "CREATE (cg:Chunk {scope: $o, id: 'ch-g', content: 'g'}) "
        "CREATE (dg)-[:CONTAINS]->(ag)-[:HAS_CHUNK]->(cg) "
        "CREATE (dx:Domain {scope: $o, id: 'd-alpha', name: 'alpha'}) "
        "CREATE (ax:Artifact {scope: $o, id: 'art-x', status: 'active'}) "
        "CREATE (cx:Chunk {scope: $o, id: 'ch-x', content: 'x'}) "
        "CREATE (x:Entity {scope: $o, id: 'x', name: 'x', type: 'concept', status: 'active', "
        "embedding: $ev}) "
        "CREATE (kx:Community {scope: $o, id: 'kx', level: 0, embedding: $kv}) "
        "CREATE (dx)-[:CONTAINS]->(ax)-[:HAS_CHUNK]->(cx)-[:HAS_ENTITY]->(x) "
        "CREATE (x)-[:BELONGS_TO_COMMUNITY]->(kx) "
        "WITH cg "
        "MATCH (e3:Entity {scope: $s, id: 'e3'}) "
        "CREATE (cg)-[:HAS_ENTITY]->(e3)",
        o=other,
        s=scope,
        ev=vec(edim, 1.0, 0.0),
        kv=vec(kdim, 1.0, 0.0),
    )


@pytest.fixture
def seeded(probe: Probe) -> dict[str, Any]:
    eindex, edim = probe.vector_index("Entity", "embedding")
    kindex, kdim = probe.vector_index("Community", "embedding")
    scope, other = PREFIX + "search", PREFIX + "search-other"
    _seed(probe, scope, other, edim, kdim)
    return {
        "scope": scope,
        "eindex": eindex,
        "kindex": kindex,
        "eq": vec(edim, 1.0),
        "kq": vec(kdim, 1.0),
    }


def _ids(hits: list[tuple[str, float]]) -> list[str]:
    return [node_id for node_id, _ in hits]


def _check_shape(hits: list[tuple[str, float]]) -> None:
    assert all(isinstance(node_id, str) for node_id, _ in hits)
    assert all(isinstance(score, float) for _, score in hits)
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)


def _entities(driver: neo4j.Driver, s: dict[str, Any], **kw: Any) -> list[tuple[str, float]]:
    hits = search_entities(driver, s["eindex"], s["scope"], s["eq"], database=_database(), **kw)
    _check_shape(hits)
    return hits


def _communities(driver: neo4j.Driver, s: dict[str, Any], **kw: Any) -> list[tuple[str, float]]:
    hits = search_communities(driver, s["kindex"], s["scope"], s["kq"], database=_database(), **kw)
    _check_shape(hits)
    return hits


# --- A14: the retriever for property filters -----------------------------------------------------


def test_unfiltered_entity_search_goes_through_vector_retriever(
    driver: neo4j.Driver, seeded: dict[str, Any], search_calls: list[dict[str, Any]]
) -> None:
    hits = _entities(driver, seeded, top_k=10)
    assert _ids(hits) == ["e1", "e7", "e3", "e6", "e2", "e4"]
    assert search_calls, "unfiltered search_entities must call VectorRetriever.get_search_results"
    filters = _filters(search_calls[-1])
    assert "scope" in filters
    assert "status" in filters


def test_level_only_community_search_goes_through_vector_retriever(
    driver: neo4j.Driver, seeded: dict[str, Any], search_calls: list[dict[str, Any]]
) -> None:
    hits = _communities(driver, seeded, top_k=10, level=1)
    assert _ids(hits) == ["k1b", "k1a"]
    assert search_calls, "level-only search_communities must call get_search_results"
    filters = _filters(search_calls[-1])
    assert "scope" in filters
    assert "level" in filters


# --- A57: entity filters -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "expected"),
    [
        ({}, ["e1", "e7", "e3", "e6", "e2", "e4"]),
        ({"level": 0}, ["e1", "e3", "e2", "e4"]),
        ({"level": 1}, ["e7", "e6"]),
        ({"level": 2}, []),
        ({"level": 7}, []),
        ({"domain": "alpha"}, ["e1", "e7", "e2"]),
        ({"domain": "beta"}, ["e3", "e4"]),
        ({"domain": "gamma"}, []),
        ({"domain": "unknown"}, []),
        ({"level": 0, "domain": "alpha"}, ["e1", "e2"]),
        ({"level": 1, "domain": "alpha"}, ["e7"]),
        ({"level": 0, "domain": "beta"}, ["e3", "e4"]),
        ({"level": 1, "domain": "beta"}, []),
        ({"level": 0, "domain": "unknown"}, []),
    ],
)
def test_search_entities_filters(
    driver: neo4j.Driver, seeded: dict[str, Any], kw: dict[str, Any], expected: list[str]
) -> None:
    assert _ids(_entities(driver, seeded, top_k=10, **kw)) == expected


@pytest.mark.parametrize(
    ("kw", "top_k", "expected"),
    [
        ({"domain": "beta"}, 1, ["e3"]),
        ({"domain": "alpha"}, 2, ["e1", "e7"]),
        ({"level": 0}, 2, ["e1", "e3"]),
        ({"level": 1}, 1, ["e7"]),
        ({"level": 0, "domain": "beta"}, 1, ["e3"]),
    ],
)
def test_search_entities_filtered_top_k_is_nearest_among_matches(
    driver: neo4j.Driver,
    seeded: dict[str, Any],
    kw: dict[str, Any],
    top_k: int,
    expected: list[str],
) -> None:
    assert _ids(_entities(driver, seeded, top_k=top_k, **kw)) == expected


def test_search_entities_default_top_k_with_domain(driver: neo4j.Driver, probe: Probe) -> None:
    index, dim = probe.vector_index("Entity", "embedding")
    scope = PREFIX + "many"
    probe.rows(
        "CREATE (d:Domain {scope: $s, id: 'd', name: 'alpha'}) "
        "CREATE (a:Artifact {scope: $s, id: 'a', status: 'active'}) "
        "CREATE (c:Chunk {scope: $s, id: 'c', content: 'c'}) "
        "CREATE (d)-[:CONTAINS]->(a)-[:HAS_CHUNK]->(c)",
        s=scope,
    )
    for i in range(12):
        probe.rows(
            "MATCH (c:Chunk {scope: $s, id: 'c'}) "
            "CREATE (c)-[:HAS_ENTITY]->(:Entity {scope: $s, id: $id, status: 'active', "
            "embedding: $v})",
            s=scope,
            id=f"e-{i:02d}",
            v=vec(dim, 1.0, i + 1),
        )
    found = search_entities(
        driver, index, scope, vec(dim, 1.0), domain="alpha", database=_database()
    )
    assert _ids(found) == [f"e-{i:02d}" for i in range(10)]


# --- A57: community filters ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "expected"),
    [
        ({}, ["k2", "k0b", "k1b", "k0a", "k0d", "k1a", "k0c"]),
        ({"level": 0}, ["k0b", "k0a", "k0d", "k0c"]),
        ({"level": 1}, ["k1b", "k1a"]),
        ({"level": 2}, ["k2"]),
        ({"level": 9}, []),
        ({"domain": "alpha"}, ["k2", "k1b", "k0a", "k1a"]),
        ({"domain": "beta"}, ["k2", "k0b", "k1b", "k1a", "k0c"]),
        ({"domain": "gamma"}, []),
        ({"domain": "unknown"}, []),
        ({"level": 0, "domain": "alpha"}, ["k0a"]),
        ({"level": 1, "domain": "alpha"}, ["k1b", "k1a"]),
        ({"level": 2, "domain": "alpha"}, ["k2"]),
        ({"level": 0, "domain": "beta"}, ["k0b", "k0c"]),
        ({"level": 9, "domain": "beta"}, []),
    ],
)
def test_search_communities_filters(
    driver: neo4j.Driver, seeded: dict[str, Any], kw: dict[str, Any], expected: list[str]
) -> None:
    assert _ids(_communities(driver, seeded, top_k=10, **kw)) == expected


@pytest.mark.parametrize(
    ("kw", "top_k", "expected"),
    [
        ({"domain": "beta"}, 2, ["k2", "k0b"]),
        ({"domain": "alpha"}, 1, ["k2"]),
        ({"level": 0}, 1, ["k0b"]),
        ({"level": 0, "domain": "beta"}, 1, ["k0b"]),
        ({"level": 1, "domain": "alpha"}, 1, ["k1b"]),
    ],
)
def test_search_communities_filtered_top_k_is_nearest_among_matches(
    driver: neo4j.Driver,
    seeded: dict[str, Any],
    kw: dict[str, Any],
    top_k: int,
    expected: list[str],
) -> None:
    assert _ids(_communities(driver, seeded, top_k=top_k, **kw)) == expected


# --- A75: dirty ratio of stats -------------------------------------------------------------------


def test_stats_dirty_ratio(driver: neo4j.Driver, probe: Probe) -> None:
    scope, other = PREFIX + "stats", PREFIX + "stats-other"
    probe.rows(
        "CREATE (:Entity {scope: $s, id: 'e1', type: 'concept', status: 'active'}) "
        "CREATE (:Community {scope: $s, id: 'k1', level: 0, dirty: true}) "
        "CREATE (:Community {scope: $s, id: 'k2', level: 0, dirty: false}) "
        "CREATE (:Community {scope: $s, id: 'k3', level: 1}) "
        "CREATE (:Community {scope: $s, id: 'k4', level: 1, dirty: false}) "
        "CREATE (:Community {scope: $o, id: 'k1', level: 0, dirty: true}) "
        "CREATE (:Community {scope: $o, id: 'k2', level: 0, dirty: true}) "
        "CREATE (:Community {scope: $o, id: 'k3', level: 0, dirty: true})",
        s=scope,
        o=other,
    )
    result = stats(driver, scope, database=_database())
    assert set(result) == STATS_KEYS
    assert isinstance(result["dirty_ratio"], float)
    assert result["dirty_ratio"] == pytest.approx(0.25)
    assert result["communities"] == 4
    assert result["levels"] == [0, 1]

    other_result = stats(driver, other, database=_database())
    assert other_result["dirty_ratio"] == pytest.approx(1.0)


def test_stats_no_dirty_community_gives_zero(driver: neo4j.Driver, probe: Probe) -> None:
    scope = PREFIX + "stats-clean"
    probe.rows(
        "CREATE (:Community {scope: $s, id: 'k1', level: 0, dirty: false}) "
        "CREATE (:Community {scope: $s, id: 'k2', level: 0})",
        s=scope,
    )
    result = stats(driver, scope, database=_database())
    assert isinstance(result["dirty_ratio"], float)
    assert result["dirty_ratio"] == 0.0


def test_stats_empty_scope_exact_keys_and_zero_dirty_ratio(
    driver: neo4j.Driver, probe: Probe
) -> None:
    probe.rows(
        "CREATE (:Community {scope: $o, id: 'k1', level: 0, dirty: true})",
        o=PREFIX + "stats-empty-other",
    )
    result = stats(driver, PREFIX + "stats-empty", database=_database())
    assert set(result) == STATS_KEYS
    assert isinstance(result["dirty_ratio"], float)
    assert result["dirty_ratio"] == 0.0
    assert result["communities"] == 0


# --- graph names only from the typed schema names ------------------------------------------------


def _docstring_nodes(tree: ast.Module) -> set[int]:
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                found.add(id(body[0].value))
    return found


def test_store_spells_no_label_or_relationship_type() -> None:
    names = load_schema().names
    spelled = set(asdict(names.labels).values()) | set(asdict(names.relationships).values())
    assert "Domain" in spelled
    assert "BELONGS_TO_COMMUNITY" in spelled
    source = Path(store_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)
    offending: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if id(node) in docstrings:
            continue
        text = node.value
        for name in spelled:
            if text.strip() == name or re.search(rf"[:`]{re.escape(name)}\b", text):
                offending.append(f"line {node.lineno}: {text!r} spells {name}")
    assert offending == []
