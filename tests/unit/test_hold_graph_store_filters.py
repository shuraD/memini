"""GR-10 acceptance: level and domain filters on vector search (A14, A57), dirty ratio (A75).

Runs against the test Neo4j given by NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DATABASE
(no host and no default written here). Each test isolates itself with its own ``scope`` values
(prefix ``qa-gr10-``), removed before and after the test.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import neo4j
import pytest
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.retrievers import VectorRetriever

from memini.infra.graph.schema import load_schema
from memini.infra.graph.store import search_communities, search_entities, stats

PREFIX = "qa-gr10-"
DEFAULT_DIMENSION = 4


def _database() -> str | None:
    return os.environ.get("NEO4J_DATABASE") or None


class Probe:
    """A plain driver to seed the graph and find the vector indexes."""

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
def search_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every VectorRetriever.get_search_results call (still performing it)."""
    calls: list[dict[str, Any]] = []
    original = VectorRetriever.get_search_results

    def spy(self: VectorRetriever, *args: Any, **kwargs: Any) -> Any:
        calls.append({"args": args, "kwargs": kwargs})
        return original(self, *args, **kwargs)

    monkeypatch.setattr(VectorRetriever, "get_search_results", spy)
    return calls


SCOPE = PREFIX + "graph"
OTHER = PREFIX + "graph-other"


def _seed(probe: Probe) -> tuple[str, str, int, int]:
    """Seed two domains, their entities and a two-level community hierarchy.

    Cosine to the query [1, 0, ...] falls as the second component grows, so the nearest-first
    entity order is e-none, e-a1, e-b1, e-a2 and the community order k1, k0-a, k0-b.
    """
    e_index, e_dim = probe.vector_index("Entity", "embedding")
    k_index, k_dim = probe.vector_index("Community", "embedding")
    probe.rows(
        "CREATE (da:Domain {scope: $s, name: 'dom-a'}) "
        "CREATE (db:Domain {scope: $s, name: 'dom-b'}) "
        "CREATE (aa:Artifact {scope: $s, id: 'art-a', status: 'active'}) "
        "CREATE (ab:Artifact {scope: $s, id: 'art-b', status: 'active'}) "
        "CREATE (ca:Chunk {scope: $s, id: 'ch-a', content: 'text a'}) "
        "CREATE (cb:Chunk {scope: $s, id: 'ch-b', content: 'text b'}) "
        "CREATE (da)-[:CONTAINS]->(aa) CREATE (db)-[:CONTAINS]->(ab) "
        "CREATE (aa)-[:HAS_CHUNK]->(ca) CREATE (ab)-[:HAS_CHUNK]->(cb) "
        "CREATE (a1:Entity {scope: $s, id: 'e-a1', status: 'active', embedding: $a1}) "
        "CREATE (a2:Entity {scope: $s, id: 'e-a2', status: 'active', embedding: $a2}) "
        "CREATE (b1:Entity {scope: $s, id: 'e-b1', status: 'active', embedding: $b1}) "
        "CREATE (en:Entity {scope: $s, id: 'e-none', status: 'active', embedding: $none}) "
        "CREATE (gone:Entity {scope: $s, id: 'e-gone', status: 'deleted', embedding: $best}) "
        "CREATE (ca)-[:HAS_ENTITY]->(a1) CREATE (ca)-[:HAS_ENTITY]->(a2) "
        "CREATE (cb)-[:HAS_ENTITY]->(b1) CREATE (ca)-[:HAS_ENTITY]->(gone) "
        "CREATE (k0a:Community {scope: $s, id: 'k0-a', level: 0, embedding: $k0a}) "
        "CREATE (k0b:Community {scope: $s, id: 'k0-b', level: 0, embedding: $k0b}) "
        "CREATE (k1:Community {scope: $s, id: 'k1', level: 1, embedding: $k1}) "
        "CREATE (a1)-[:BELONGS_TO_COMMUNITY]->(k0a) CREATE (a2)-[:BELONGS_TO_COMMUNITY]->(k0a) "
        "CREATE (b1)-[:BELONGS_TO_COMMUNITY]->(k0b) "
        "CREATE (k0a)-[:BELONGS_TO_COMMUNITY]->(k1) CREATE (k0b)-[:BELONGS_TO_COMMUNITY]->(k1) "
        # another scope with the same domain name and nearer vectors: never returned
        "CREATE (xd:Domain {scope: $o, name: 'dom-a'}) "
        "CREATE (xa:Artifact {scope: $o, id: 'art-a', status: 'active'}) "
        "CREATE (xc:Chunk {scope: $o, id: 'ch-a', content: 'text x'}) "
        "CREATE (xe:Entity {scope: $o, id: 'e-x', status: 'active', embedding: $best}) "
        "CREATE (xk:Community {scope: $o, id: 'k-x', level: 0, embedding: $kbest}) "
        "CREATE (xd)-[:CONTAINS]->(xa) CREATE (xa)-[:HAS_CHUNK]->(xc) "
        "CREATE (xc)-[:HAS_ENTITY]->(xe) CREATE (xe)-[:BELONGS_TO_COMMUNITY]->(xk)",
        s=SCOPE,
        o=OTHER,
        a1=vec(e_dim, 1.0, 0.1),
        a2=vec(e_dim, 1.0, 0.6),
        b1=vec(e_dim, 1.0, 0.3),
        none=vec(e_dim, 1.0, 0.05),
        best=vec(e_dim, 1.0),
        k0a=vec(k_dim, 1.0, 0.2),
        k0b=vec(k_dim, 1.0, 0.4),
        k1=vec(k_dim, 1.0, 0.1),
        kbest=vec(k_dim, 1.0),
    )
    return e_index, k_index, e_dim, k_dim


def _ids(hits: list[tuple[str, float]]) -> list[str]:
    assert all(isinstance(score, float) for _, score in hits)
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)
    return [node_id for node_id, _ in hits]


# --- A57: entity search with level and domain filters ---------------------------------------


def test_a57_search_entities_by_domain(probe: Probe) -> None:
    e_index, _, e_dim, _ = _seed(probe)
    q = vec(e_dim, 1.0)
    d = probe.driver
    db = _database()

    assert _ids(search_entities(d, e_index, SCOPE, q, domain="dom-a", database=db)) == [
        "e-a1",
        "e-a2",
    ]
    assert _ids(search_entities(d, e_index, SCOPE, q, domain="dom-b", database=db)) == ["e-b1"]
    assert search_entities(d, e_index, SCOPE, q, domain="missing", database=db) == []
    top1 = search_entities(d, e_index, SCOPE, q, top_k=1, domain="dom-a", database=db)
    assert _ids(top1) == ["e-a1"]


def test_a57_search_entities_by_level(probe: Probe) -> None:
    e_index, _, e_dim, _ = _seed(probe)
    q = vec(e_dim, 1.0)
    d = probe.driver
    db = _database()

    assert _ids(search_entities(d, e_index, SCOPE, q, level=0, database=db)) == [
        "e-a1",
        "e-b1",
        "e-a2",
    ]
    assert search_entities(d, e_index, SCOPE, q, level=1, database=db) == []
    assert search_entities(d, e_index, SCOPE, q, level=7, database=db) == []


def test_a57_search_entities_level_and_domain_together(probe: Probe) -> None:
    e_index, _, e_dim, _ = _seed(probe)
    q = vec(e_dim, 1.0)
    d = probe.driver
    db = _database()

    found = search_entities(d, e_index, SCOPE, q, level=0, domain="dom-b", database=db)
    assert _ids(found) == ["e-b1"]
    assert search_entities(d, e_index, SCOPE, q, level=1, domain="dom-a", database=db) == []


def test_a57_search_entities_without_filters_unchanged(probe: Probe) -> None:
    e_index, _, e_dim, _ = _seed(probe)
    found = search_entities(
        probe.driver, e_index, SCOPE, vec(e_dim, 1.0), level=None, domain=None, database=_database()
    )
    assert _ids(found) == ["e-none", "e-a1", "e-b1", "e-a2"]


# --- A57: community search with level and domain filters ------------------------------------


def test_a57_search_communities_by_domain(probe: Probe) -> None:
    _, k_index, _, k_dim = _seed(probe)
    q = vec(k_dim, 1.0)
    d = probe.driver
    db = _database()

    assert _ids(search_communities(d, k_index, SCOPE, q, domain="dom-a", database=db)) == [
        "k1",
        "k0-a",
    ]
    assert _ids(search_communities(d, k_index, SCOPE, q, domain="dom-b", database=db)) == [
        "k1",
        "k0-b",
    ]
    assert search_communities(d, k_index, SCOPE, q, domain="missing", database=db) == []
    top1 = search_communities(d, k_index, SCOPE, q, top_k=1, domain="dom-b", database=db)
    assert _ids(top1) == ["k1"]


def test_a57_search_communities_level_and_domain_together(probe: Probe) -> None:
    _, k_index, _, k_dim = _seed(probe)
    q = vec(k_dim, 1.0)
    d = probe.driver
    db = _database()

    level0_b = search_communities(d, k_index, SCOPE, q, level=0, domain="dom-b", database=db)
    assert _ids(level0_b) == ["k0-b"]
    level1_a = search_communities(d, k_index, SCOPE, q, level=1, domain="dom-a", database=db)
    assert _ids(level1_a) == ["k1"]
    assert search_communities(d, k_index, SCOPE, q, level=2, domain="dom-a", database=db) == []


# --- A14: property filters go through VectorRetriever ---------------------------------------


def test_a14_level_only_community_search_uses_vector_retriever(
    probe: Probe, search_calls: list[dict[str, Any]]
) -> None:
    _, k_index, _, k_dim = _seed(probe)
    level_prop = load_schema().names.properties.level

    found = search_communities(
        probe.driver, k_index, SCOPE, vec(k_dim, 1.0), level=0, domain=None, database=_database()
    )

    assert _ids(found) == ["k0-a", "k0-b"]
    assert search_calls, "A14: a level-only community search must use VectorRetriever"
    filters = [call["kwargs"].get("filters") or {} for call in search_calls]
    assert any(level_prop in f for f in filters), "A14: the level goes in the retriever filter"


def test_a14_unfiltered_entity_search_uses_vector_retriever(
    probe: Probe, search_calls: list[dict[str, Any]]
) -> None:
    e_index, _, e_dim, _ = _seed(probe)
    found = search_entities(
        probe.driver, e_index, SCOPE, vec(e_dim, 1.0), domain=None, database=_database()
    )
    assert _ids(found)[0] == "e-none"
    assert search_calls, "A14: an entity search with property filters only uses VectorRetriever"


def test_a14_filter_the_retriever_cannot_express_still_exact(probe: Probe) -> None:
    """A14: domain (a graph path) goes through the custom Cypher search and stays exact."""
    e_index, k_index, e_dim, k_dim = _seed(probe)
    d = probe.driver
    db = _database()
    # e-none, the nearest active entity, is outside every domain: a post-filtered top_k=1
    # would come back empty.
    one = search_entities(d, e_index, SCOPE, vec(e_dim, 1.0), top_k=1, domain="dom-b", database=db)
    assert _ids(one) == ["e-b1"]
    one_k = search_communities(
        d, k_index, SCOPE, vec(k_dim, 1.0), top_k=1, level=0, domain="dom-b", database=db
    )
    assert _ids(one_k) == ["k0-b"]


# --- A75: stats dirty ratio, no other KPI -----------------------------------------------------


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


def test_a75_stats_dirty_ratio(probe: Probe) -> None:
    scope = PREFIX + "stats"
    other = PREFIX + "stats-other"
    probe.rows(
        "CREATE (:Entity {scope: $s, id: 'e1', type: 'service', status: 'active'}) "
        "CREATE (:Community {scope: $s, id: 'k1', level: 0, dirty: true}) "
        "CREATE (:Community {scope: $s, id: 'k2', level: 0, dirty: false}) "
        "CREATE (:Community {scope: $s, id: 'k3', level: 0}) "
        "CREATE (:Community {scope: $s, id: 'k4', level: 1, dirty: true}) "
        "CREATE (:Community {scope: $o, id: 'kx', level: 0, dirty: true}) "
        "CREATE (:Community {scope: $o, id: 'ky', level: 0, dirty: true})",
        s=scope,
        o=other,
    )

    result = stats(probe.driver, scope, database=_database())

    assert set(result) == STATS_KEYS, "A75: no quality or drift KPI beyond the card line"
    assert isinstance(result["dirty_ratio"], float)
    assert result["dirty_ratio"] == pytest.approx(0.5)
    assert result["communities"] == 4
    assert result["levels"] == [0, 1]
    assert result["by_status"] == {"active": 1}


def test_a75_stats_empty_scope_dirty_ratio_zero(probe: Probe) -> None:
    probe.rows(
        "CREATE (:Community {scope: $o, id: 'kx', level: 0, dirty: true})",
        o=PREFIX + "stats-empty-other",
    )
    result = stats(probe.driver, PREFIX + "stats-empty", database=_database())
    assert set(result) == STATS_KEYS
    assert result["dirty_ratio"] == 0.0
    assert isinstance(result["dirty_ratio"], float)
    assert result["communities"] == 0
