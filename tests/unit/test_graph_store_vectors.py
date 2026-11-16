"""GR-04 acceptance against the test Neo4j: vector writes, soft/hard delete, vector search, stats.

The Neo4j settings come from NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DATABASE (localhost
locally, the sandbox's test host elsewhere); no host is written here. Each test isolates itself
with its own ``scope`` values (prefix ``qa-gr04-``), removed before and after the test. The
vector indexes are reused when present (their dimension is read back), else created.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import neo4j
import neo4j_graphrag.indexes as graphrag_indexes
import pytest
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.retrievers import VectorRetriever

from memini.infra.graph import store as store_mod
from memini.infra.graph.store import (
    delete,
    search_communities,
    search_entities,
    stats,
    write_vectors,
)

PREFIX = "qa-gr04-"
DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_DIMENSION = 4
LONG_SUMMARY = "".join(f"word{i:03d} " for i in range(60))  # 480 characters


def _uri() -> str:
    return os.environ.get("NEO4J_URI") or DEFAULT_URI


def _database() -> str | None:
    return os.environ.get("NEO4J_DATABASE") or None


class Probe:
    """A plain driver to seed the graph and read back what the store did."""

    def __init__(self) -> None:
        self.driver = neo4j.GraphDatabase.driver(
            _uri(),
            auth=(os.environ.get("NEO4J_USER") or "neo4j", os.environ.get("NEO4J_PASSWORD", "")),
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

    def node(self, label: str, scope: str, node_id: str) -> dict[str, Any] | None:
        found = self.rows(
            f"MATCH (n:`{label}` {{scope: $scope, id: $id}}) RETURN properties(n) AS props",
            scope=scope,
            id=node_id,
        )
        assert len(found) <= 1
        return found[0]["props"] if found else None

    def count(self, scope: str) -> int:
        return int(self.rows("MATCH (n {scope: $scope}) RETURN count(n) AS c", scope=scope)[0]["c"])

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
def dim(probe: Probe) -> int:
    """The dimension of the Entity.embedding vector index (seeded vectors match it)."""
    return probe.vector_index("Entity", "embedding")[1]


@pytest.fixture
def upsert_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Record every call to neo4j_graphrag.indexes.upsert_vectors (still performing it)."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    original = graphrag_indexes.upsert_vectors

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(graphrag_indexes, "upsert_vectors", spy)
    for name in [n for n, v in vars(store_mod).items() if v is original]:
        monkeypatch.setattr(store_mod, name, spy)
    return calls


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


def _seed_entity(probe: Probe, scope: str, node_id: str, **props: Any) -> None:
    probe.rows(
        "CREATE (n:Entity {scope: $scope, id: $id}) SET n += $props",
        scope=scope,
        id=node_id,
        props=props,
    )


# --- write_vectors -----------------------------------------------------------------------------


def test_write_vectors_sets_embedding_and_counts(
    driver: neo4j.Driver, probe: Probe, upsert_calls: list[Any]
) -> None:
    _, dim = probe.vector_index("Entity", "embedding")
    scope = PREFIX + "wv"
    for node_id in ("e-1", "e-2", "e-3"):
        _seed_entity(probe, scope, node_id, name=node_id, status="active")
    vectors = {
        "e-1": vec(dim, 1.0, 0.5),
        "e-2": vec(dim, 0.25, 1.0),
        "missing": vec(dim, 1.0, 1.0),
    }
    written = write_vectors(driver, "Entity", "embedding", scope, vectors, database=_database())
    assert written == 2
    assert upsert_calls, "write_vectors must write through neo4j_graphrag upsert_vectors"
    for node_id in ("e-1", "e-2"):
        node = probe.node("Entity", scope, node_id)
        assert node is not None
        assert node["embedding"] == pytest.approx(vectors[node_id])
    e3 = probe.node("Entity", scope, "e-3")
    assert e3 is not None
    assert "embedding" not in e3
    assert probe.node("Entity", scope, "missing") is None
    assert probe.count(scope) == 3


def test_write_vectors_empty_mapping_returns_zero(driver: neo4j.Driver, probe: Probe) -> None:
    scope = PREFIX + "wv-empty"
    _seed_entity(probe, scope, "e-1", status="active")
    assert write_vectors(driver, "Entity", "embedding", scope, {}, database=_database()) == 0
    node = probe.node("Entity", scope, "e-1")
    assert node is not None
    assert "embedding" not in node


def test_write_vectors_only_unknown_ids_returns_zero(driver: neo4j.Driver, probe: Probe) -> None:
    _, dim = probe.vector_index("Entity", "embedding")
    scope = PREFIX + "wv-unknown"
    _seed_entity(probe, scope, "e-1", status="active")
    written = write_vectors(
        driver, "Entity", "embedding", scope, {"nope": vec(dim, 1.0)}, database=_database()
    )
    assert written == 0
    assert probe.count(scope) == 1


def test_write_vectors_other_prop_scope_and_label_untouched(
    driver: neo4j.Driver, probe: Probe, dim: int, upsert_calls: list[Any]
) -> None:
    _, code_dim = probe.vector_index("Entity", "embedding_code")
    scope = PREFIX + "wv-code"
    other = PREFIX + "wv-code-other"
    original = vec(dim, 0.5, 0.5)
    _seed_entity(probe, scope, "e-1", status="active", embedding=original)
    _seed_entity(probe, other, "e-1", status="active", embedding=original)
    probe.rows("CREATE (:Chunk {scope: $scope, id: 'e-1', content: 'text'})", scope=scope)

    new = vec(code_dim, 1.0, 0.25)
    written = write_vectors(
        driver, "Entity", "embedding_code", scope, {"e-1": new}, database=_database()
    )
    assert written == 1
    assert upsert_calls

    node = probe.node("Entity", scope, "e-1")
    assert node is not None
    assert node["embedding_code"] == pytest.approx(new)
    assert node["embedding"] == pytest.approx(original)

    other_node = probe.node("Entity", other, "e-1")
    assert other_node is not None
    assert "embedding_code" not in other_node
    assert other_node["embedding"] == pytest.approx(original)

    chunk = probe.node("Chunk", scope, "e-1")
    assert chunk is not None
    assert "embedding_code" not in chunk


# --- delete --------------------------------------------------------------------------------------


def _seed_delete_graph(probe: Probe, scope: str, dim: int) -> None:
    probe.rows(
        "CREATE (e:Entity {scope: $scope, id: 'target', name: 'Target', status: 'active', "
        "summary: $summary, embedding: $v, embedding_code: $v}) "
        "CREATE (old:Entity {scope: $scope, id: 'old', status: 'superseded', summary: 'old'}) "
        "CREATE (k:Community {scope: $scope, id: 'k-1', level: 0, summary: 'comm'}) "
        "CREATE (e)-[:BELONGS_TO_COMMUNITY]->(k) "
        "CREATE (e)-[:SUPERSEDES]->(old)",
        scope=scope,
        summary=LONG_SUMMARY,
        v=vec(dim, 1.0, 0.5),
    )


def test_soft_delete_one_node(driver: neo4j.Driver, probe: Probe, dim: int) -> None:
    scope = PREFIX + "soft"
    _seed_delete_graph(probe, scope, dim)
    assert len(LONG_SUMMARY) > 200

    assert delete(driver, scope, node_id="target", database=_database()) == 1

    node = probe.node("Entity", scope, "target")
    assert node is not None
    assert node["summary"] == LONG_SUMMARY[:200]
    assert "embedding" not in node
    assert "embedding_code" not in node
    assert node["status"] == "deleted"
    belongs = probe.rows(
        "MATCH (:Entity {scope: $scope, id: 'target'})-[r:BELONGS_TO_COMMUNITY]-() "
        "RETURN count(r) AS c",
        scope=scope,
    )
    assert belongs[0]["c"] == 0
    supersedes = probe.rows(
        "MATCH (:Entity {scope: $scope, id: 'target'})-[r:SUPERSEDES]->"
        "(:Entity {scope: $scope, id: 'old'}) RETURN count(r) AS c",
        scope=scope,
    )
    assert supersedes[0]["c"] == 1
    assert probe.node("Community", scope, "k-1") is not None
    old = probe.node("Entity", scope, "old")
    assert old is not None
    assert old["status"] == "superseded"


@pytest.mark.parametrize("status", ["deleted", "superseded"])
def test_soft_delete_inactive_node_is_noop(
    driver: neo4j.Driver, probe: Probe, dim: int, status: str
) -> None:
    scope = PREFIX + "soft-noop-" + status
    v = vec(dim, 1.0)
    _seed_entity(probe, scope, "e-1", status=status, summary=LONG_SUMMARY, embedding=v)
    before = probe.node("Entity", scope, "e-1")

    assert delete(driver, scope, node_id="e-1", database=_database()) == 0

    assert probe.node("Entity", scope, "e-1") == before


def test_soft_delete_unknown_or_other_scope_id_returns_zero(
    driver: neo4j.Driver, probe: Probe
) -> None:
    scope = PREFIX + "soft-unknown"
    other = PREFIX + "soft-unknown-other"
    _seed_entity(probe, other, "e-1", status="active", summary="kept")
    assert delete(driver, scope, node_id="e-1", database=_database()) == 0
    assert delete(driver, scope, node_id="absent", database=_database()) == 0
    node = probe.node("Entity", other, "e-1")
    assert node is not None
    assert node["status"] == "active"
    assert node["summary"] == "kept"


def test_soft_delete_whole_scope(driver: neo4j.Driver, probe: Probe, dim: int) -> None:
    scope = PREFIX + "soft-scope"
    other = PREFIX + "soft-scope-other"
    v = vec(dim, 1.0, 0.5)
    for node_id in ("a", "b", "c"):
        _seed_entity(probe, scope, node_id, status="active", summary=LONG_SUMMARY, embedding=v)
    _seed_entity(probe, scope, "gone", status="deleted", summary="x")
    _seed_entity(probe, scope, "old", status="superseded", summary="y", embedding=v)
    _seed_entity(probe, other, "a", status="active", summary=LONG_SUMMARY, embedding=v)
    probe.rows(
        "CREATE (:Community {scope: $scope, id: 'k-1', level: 0, summary: 'comm'})", scope=scope
    )

    assert delete(driver, scope, database=_database()) == 3

    for node_id in ("a", "b", "c"):
        node = probe.node("Entity", scope, node_id)
        assert node is not None
        assert node["status"] == "deleted"
        assert node["summary"] == LONG_SUMMARY[:200]
        assert "embedding" not in node
    old = probe.node("Entity", scope, "old")
    assert old is not None
    assert old["status"] == "superseded"
    assert old["embedding"] == pytest.approx(v)
    assert probe.node("Community", scope, "k-1") is not None

    kept = probe.node("Entity", other, "a")
    assert kept is not None
    assert kept["status"] == "active"
    assert kept["summary"] == LONG_SUMMARY
    assert kept["embedding"] == pytest.approx(v)


def test_hard_delete_one_node(driver: neo4j.Driver, probe: Probe, dim: int) -> None:
    scope = PREFIX + "hard"
    other = PREFIX + "hard-other"
    _seed_delete_graph(probe, scope, dim)
    _seed_entity(probe, other, "target", status="active")

    assert delete(driver, scope, node_id="target", hard=True, database=_database()) == 1

    assert probe.node("Entity", scope, "target") is None
    assert probe.node("Entity", scope, "old") is not None
    assert probe.node("Community", scope, "k-1") is not None
    assert probe.node("Entity", other, "target") is not None
    rels = probe.rows(
        "MATCH (a {scope: $scope})-[r]-() RETURN count(r) AS c",
        scope=scope,
    )
    assert rels[0]["c"] == 0


def test_hard_delete_whole_scope_any_label(driver: neo4j.Driver, probe: Probe, dim: int) -> None:
    scope = PREFIX + "hard-scope"
    other = PREFIX + "hard-scope-other"
    _seed_delete_graph(probe, scope, dim)
    probe.rows(
        "CREATE (a:Artifact {scope: $scope, id: 'art', status: 'deleted'}) "
        "CREATE (c:Chunk {scope: $scope, id: 'ch', content: 'text'}) "
        "CREATE (a)-[:HAS_CHUNK]->(c)",
        scope=scope,
    )
    _seed_entity(probe, other, "target", status="active")
    assert probe.count(scope) == 5

    assert delete(driver, scope, hard=True, database=_database()) == 5

    assert probe.count(scope) == 0
    assert probe.count(other) == 1


# --- vector search -------------------------------------------------------------------------------


def test_search_entities(driver: neo4j.Driver, probe: Probe, search_calls: list[Any]) -> None:
    index, dim = probe.vector_index("Entity", "embedding")
    scope = PREFIX + "search-e"
    other = PREFIX + "search-e-other"
    _seed_entity(probe, scope, "a", status="active", embedding=vec(dim, 1.0, 0.1))
    _seed_entity(probe, scope, "b", status="active", embedding=vec(dim, 1.0, 1.0))
    _seed_entity(probe, scope, "c", status="active", embedding=vec(dim, 0.1, 1.0))
    _seed_entity(probe, scope, "gone", status="deleted", embedding=vec(dim, 1.0))
    _seed_entity(probe, other, "x", status="active", embedding=vec(dim, 1.0))
    q = vec(dim, 1.0)

    top2 = search_entities(driver, index, scope, q, top_k=2, database=_database())
    assert search_calls, "search_entities must call VectorRetriever.get_search_results"
    assert [node_id for node_id, _ in top2] == ["a", "b"]
    assert all(isinstance(score, float) for _, score in top2)
    assert top2[0][1] > top2[1][1]

    everything = search_entities(driver, index, scope, q, top_k=10, database=_database())
    assert [node_id for node_id, _ in everything] == ["a", "b", "c"]
    scores = [score for _, score in everything]
    assert scores == sorted(scores, reverse=True)


def test_search_entities_default_top_k(driver: neo4j.Driver, probe: Probe) -> None:
    index, dim = probe.vector_index("Entity", "embedding")
    scope = PREFIX + "search-e-many"
    for i in range(12):
        _seed_entity(probe, scope, f"e-{i:02d}", status="active", embedding=vec(dim, 1.0, i + 1))
    found = search_entities(driver, index, scope, vec(dim, 1.0), database=_database())
    assert [node_id for node_id, _ in found] == [f"e-{i:02d}" for i in range(10)]


def test_search_communities_by_level(
    driver: neo4j.Driver, probe: Probe, search_calls: list[Any]
) -> None:
    index, dim = probe.vector_index("Community", "embedding")
    scope = PREFIX + "search-k"
    other = PREFIX + "search-k-other"
    seeds = [
        (scope, "k0-a", 0, vec(dim, 1.0, 0.05)),
        (scope, "k0-b", 0, vec(dim, 1.0, 2.0)),
        (scope, "k1-a", 1, vec(dim, 1.0, 0.5)),
        (scope, "k1-b", 1, vec(dim, 1.0, 1.0)),
        (other, "k1-x", 1, vec(dim, 1.0)),
    ]
    for s, node_id, level, v in seeds:
        probe.rows(
            "CREATE (:Community {scope: $scope, id: $id, level: $level, embedding: $v})",
            scope=s,
            id=node_id,
            level=level,
            v=v,
        )
    q = vec(dim, 1.0)

    level1 = search_communities(driver, index, scope, q, top_k=10, level=1, database=_database())
    assert search_calls, "search_communities must call VectorRetriever.get_search_results"
    assert [node_id for node_id, _ in level1] == ["k1-a", "k1-b"]
    assert level1[0][1] > level1[1][1]

    level0 = search_communities(driver, index, scope, q, top_k=10, level=0, database=_database())
    assert [node_id for node_id, _ in level0] == ["k0-a", "k0-b"]

    every = search_communities(driver, index, scope, q, top_k=10, database=_database())
    assert [node_id for node_id, _ in every] == ["k0-a", "k1-a", "k1-b", "k0-b"]
    scores = [score for _, score in every]
    assert scores == sorted(scores, reverse=True)

    top1 = search_communities(driver, index, scope, q, top_k=1, level=None, database=_database())
    assert [node_id for node_id, _ in top1] == ["k0-a"]


# --- stats ---------------------------------------------------------------------------------------


def test_stats_seeded_scope(driver: neo4j.Driver, probe: Probe, dim: int) -> None:
    scope = PREFIX + "stats"
    other = PREFIX + "stats-other"
    v = vec(dim, 1.0, 0.5)
    probe.rows(
        "CREATE (e1:Entity {scope: $scope, id: 'e1', type: 'service', status: 'active', "
        "embedding: $v}) "
        "CREATE (e2:Entity {scope: $scope, id: 'e2', type: 'concept', status: 'active', "
        "embedding: $v}) "
        "CREATE (e3:Entity {scope: $scope, id: 'e3', type: 'service', status: 'superseded'}) "
        "CREATE (e4:Entity {scope: $scope, id: 'e4', type: 'concept', status: 'deleted'}) "
        "CREATE (ch:Chunk {scope: $scope, id: 'ch', content: 'text', embedding: $v}) "
        "CREATE (ka:Community {scope: $scope, id: 'ka', level: 0}) "
        "CREATE (kb:Community {scope: $scope, id: 'kb', level: 0}) "
        "CREATE (k1:Community {scope: $scope, id: 'k1', level: 1}) "
        "CREATE (x:Entity {scope: $other, id: 'x', type: 'service', status: 'active', "
        "embedding: $v}) "
        "CREATE (kx:Community {scope: $other, id: 'kx', level: 2}) "
        "CREATE (e1)-[:RELATED_TO {kind: 'extracted'}]->(e2) "
        "CREATE (e1)-[:RELATED_TO {kind: 'cross_domain'}]->(x) "
        "CREATE (e1)-[:SUPERSEDES]->(e3) "
        "CREATE (ch)-[:HAS_ENTITY]->(e1) "
        "CREATE (e1)-[:BELONGS_TO_COMMUNITY]->(ka) "
        "CREATE (e2)-[:BELONGS_TO_COMMUNITY]->(ka) "
        "CREATE (e3)-[:BELONGS_TO_COMMUNITY]->(kb) "
        "CREATE (ka)-[:BELONGS_TO_COMMUNITY]->(k1) "
        "CREATE (kb)-[:BELONGS_TO_COMMUNITY]->(k1) "
        "CREATE (x)-[:BELONGS_TO_COMMUNITY]->(kx)",
        scope=scope,
        other=other,
        v=v,
    )

    result = stats(driver, scope, database=_database())

    assert result["by_status"] == {"active": 2, "superseded": 1, "deleted": 1}
    assert result["by_type"] == {"service": 2, "concept": 2}
    assert result["relationships"] == {
        "RELATED_TO": 1,
        "SUPERSEDES": 1,
        "HAS_ENTITY": 1,
        "BELONGS_TO_COMMUNITY": 5,
    }
    assert isinstance(result["embedding_coverage"], float)
    assert result["embedding_coverage"] == pytest.approx(0.5)
    assert result["communities"] == 3
    assert result["levels"] == [0, 1]
    assert isinstance(result["avg_community_size"], float)
    assert result["avg_community_size"] == pytest.approx(5 / 3)


def test_stats_empty_scope(driver: neo4j.Driver, probe: Probe) -> None:
    other = PREFIX + "stats-empty-other"
    _seed_entity(probe, other, "x", type="service", status="active")
    result = stats(driver, PREFIX + "stats-empty", database=_database())
    assert result["by_status"] == {}
    assert result["by_type"] == {}
    assert result["relationships"] == {}
    assert result["embedding_coverage"] == 0.0
    assert isinstance(result["embedding_coverage"], float)
    assert result["communities"] == 0
    assert result["levels"] == []
    assert result["avg_community_size"] == 0.0
    assert isinstance(result["avg_community_size"], float)
