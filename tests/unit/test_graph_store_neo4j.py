"""GR-03 acceptance against the test Neo4j: constraints, append-only writes, supersede.

The Neo4j settings come from NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DATABASE (localhost
locally, the sandbox's test host elsewhere); no host is written here. The test host is passed as
``allowed_hosts`` so the guard accepts it. Each test isolates itself with its own ``scope``
values (prefix ``qa-gr03-``), removed before and after the test.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit

import neo4j
import pytest

from memini.infra.graph.store import GraphStore

PREFIX = "qa-gr03-"
DEFAULT_URI = "bolt://localhost:7687"


def _uri() -> str:
    return os.environ.get("NEO4J_URI") or DEFAULT_URI


def _database() -> str | None:
    return os.environ.get("NEO4J_DATABASE") or None


def _allowed() -> list[str]:
    host = urlsplit(_uri()).hostname
    return [host] if host else []


class Probe:
    """A plain driver to read back what the store wrote."""

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

    def nodes(self, label: str, scope: str) -> list[dict[str, Any]]:
        return [
            r["props"]
            for r in self.rows(
                f"MATCH (n:`{label}`) WHERE n.scope = $scope RETURN properties(n) AS props "
                "ORDER BY n.id",
                scope=scope,
            )
        ]


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
def store(probe: Probe) -> Iterator[GraphStore]:
    s = GraphStore(allowed_hosts=_allowed())
    result = s.connect()
    assert result is s
    try:
        yield s
    finally:
        s.close()


def test_connect_creates_entity_scope_id_uniqueness(store: GraphStore, probe: Probe) -> None:
    rows = probe.rows(
        "SHOW CONSTRAINTS YIELD type, entityType, labelsOrTypes, properties "
        "RETURN type, entityType, labelsOrTypes, properties"
    )
    matching = [
        r
        for r in rows
        if r["entityType"] == "NODE"
        and r["labelsOrTypes"] == ["Entity"]
        and r["properties"] == ["scope", "id"]
        and "UNIQUENESS" in str(r["type"]).upper()
    ]
    assert matching, rows


def test_connect_is_idempotent(store: GraphStore, probe: Probe) -> None:
    again = GraphStore(allowed_hosts=_allowed())
    assert again.connect() is again
    again.close()
    rows = probe.rows(
        "SHOW CONSTRAINTS YIELD labelsOrTypes, properties "
        "WHERE labelsOrTypes = ['Entity'] AND properties = ['scope', 'id'] RETURN count(*) AS c"
    )
    assert rows[0]["c"] == 1


def test_write_new_items(store: GraphStore, probe: Probe) -> None:
    scope = PREFIX + "new"
    items = [
        {"id": "e-b", "content_hash": "h-b", "name": "Beta", "type": "concept"},
        {"id": "e-a", "content_hash": "h-a", "name": "Alpha", "type": "service"},
    ]
    written = store.write("Entity", scope, items, updated_by="qa-writer")
    assert written == ["e-b", "e-a"]
    nodes = probe.nodes("Entity", scope)
    assert len(nodes) == 2
    by_id = {n["id"]: n for n in nodes}
    for item in items:
        node = by_id[item["id"]]
        assert node["scope"] == scope
        assert node["content_hash"] == item["content_hash"]
        assert node["name"] == item["name"]
        assert node["type"] == item["type"]
        assert node["status"] == "active"
        assert node["updated_by"] == "qa-writer"
        assert node.get("updated_at") is not None


def test_write_status_only_when_schema_lists_it(store: GraphStore, probe: Probe) -> None:
    scope = PREFIX + "chunk"
    written = store.write(
        "Chunk",
        scope,
        [{"id": "c-1", "content_hash": "hc-1", "content": "some text"}],
        updated_by="qa-writer",
    )
    assert written == ["c-1"]
    nodes = probe.nodes("Chunk", scope)
    assert len(nodes) == 1
    assert nodes[0]["content"] == "some text"
    assert nodes[0]["content_hash"] == "hc-1"
    assert nodes[0]["updated_by"] == "qa-writer"
    assert nodes[0].get("updated_at") is not None
    assert "status" not in nodes[0]


def test_write_skips_known_content_hash(store: GraphStore, probe: Probe) -> None:
    scope = PREFIX + "dup"
    other_scope = PREFIX + "dup-other"
    assert store.write(
        "Entity", scope, [{"id": "e-1", "content_hash": "h-1", "name": "First"}], updated_by="qa-1"
    ) == ["e-1"]
    before = probe.nodes("Entity", scope)
    assert len(before) == 1

    again = store.write(
        "Entity", scope, [{"id": "e-2", "content_hash": "h-1", "name": "Copy"}], updated_by="qa-2"
    )
    assert again == []
    after = probe.nodes("Entity", scope)
    assert after == before

    same_id = store.write(
        "Entity",
        scope,
        [{"id": "e-1", "content_hash": "h-1", "name": "Renamed"}],
        updated_by="qa-3",
    )
    assert same_id == []
    assert probe.nodes("Entity", scope) == before

    elsewhere = store.write(
        "Entity",
        other_scope,
        [{"id": "e-2", "content_hash": "h-1", "name": "Copy"}],
        updated_by="qa-2",
    )
    assert elsewhere == ["e-2"]
    assert len(probe.nodes("Entity", other_scope)) == 1
    assert probe.nodes("Entity", scope) == before


def test_write_mixed_batch_returns_only_written(store: GraphStore, probe: Probe) -> None:
    scope = PREFIX + "mixed"
    store.write("Entity", scope, [{"id": "e-1", "content_hash": "h-1"}], updated_by="qa")
    written = store.write(
        "Entity",
        scope,
        [
            {"id": "e-3", "content_hash": "h-3"},
            {"id": "e-2", "content_hash": "h-1"},
            {"id": "e-4", "content_hash": "h-4"},
        ],
        updated_by="qa",
    )
    assert written == ["e-3", "e-4"]
    assert sorted(n["id"] for n in probe.nodes("Entity", scope)) == ["e-1", "e-3", "e-4"]


def test_same_hash_on_other_label_is_written(store: GraphStore, probe: Probe) -> None:
    scope = PREFIX + "labels"
    store.write("Entity", scope, [{"id": "x-1", "content_hash": "h-same"}], updated_by="qa")
    written = store.write(
        "Artifact", scope, [{"id": "x-1", "content_hash": "h-same"}], updated_by="qa"
    )
    assert written == ["x-1"]
    assert len(probe.nodes("Artifact", scope)) == 1
    assert len(probe.nodes("Entity", scope)) == 1


def test_write_existing_id_with_new_hash_updates_in_place(store: GraphStore, probe: Probe) -> None:
    scope = PREFIX + "update"
    store.write(
        "Entity", scope, [{"id": "e-1", "content_hash": "h-old", "name": "Old"}], updated_by="qa-1"
    )
    written = store.write(
        "Entity", scope, [{"id": "e-1", "content_hash": "h-new", "name": "New"}], updated_by="qa-2"
    )
    assert written == ["e-1"]
    nodes = probe.nodes("Entity", scope)
    assert len(nodes) == 1
    assert nodes[0]["content_hash"] == "h-new"
    assert nodes[0]["name"] == "New"
    assert nodes[0]["updated_by"] == "qa-2"
    assert nodes[0]["status"] == "active"


def test_write_large_batch(store: GraphStore, probe: Probe) -> None:
    scope = PREFIX + "batch"
    items = [{"id": f"e-{i:03d}", "content_hash": f"h-{i:03d}"} for i in range(50)]
    written = store.write("Entity", scope, items, updated_by="qa")
    assert written == [item["id"] for item in items]
    assert len(probe.nodes("Entity", scope)) == 50


def test_supersede(store: GraphStore, probe: Probe) -> None:
    scope = PREFIX + "supersede"
    store.write(
        "Entity",
        scope,
        [{"id": "old", "content_hash": "h-old"}, {"id": "new", "content_hash": "h-new"}],
        updated_by="qa-writer",
    )
    result = store.supersede("Entity", scope, "new", "old", updated_by="qa-super")
    assert result is None
    edges = probe.rows(
        "MATCH (a:Entity {scope: $scope})-[r:SUPERSEDES]->(b:Entity {scope: $scope}) "
        "RETURN a.id AS src, b.id AS dst",
        scope=scope,
    )
    assert edges == [{"src": "new", "dst": "old"}]
    by_id = {n["id"]: n for n in probe.nodes("Entity", scope)}
    assert set(by_id) == {"old", "new"}
    assert by_id["old"]["status"] == "superseded"
    assert by_id["old"]["updated_by"] == "qa-super"
    assert by_id["old"].get("updated_at") is not None
    assert by_id["new"]["status"] == "active"


def test_supersede_missing_node_raises(store: GraphStore, probe: Probe) -> None:
    scope = PREFIX + "missing"
    other = PREFIX + "missing-other"
    store.write("Entity", scope, [{"id": "here", "content_hash": "h-here"}], updated_by="qa")
    store.write("Entity", other, [{"id": "there", "content_hash": "h-there"}], updated_by="qa")
    with pytest.raises(LookupError):
        store.supersede("Entity", scope, "here", "absent", updated_by="qa")
    with pytest.raises(LookupError):
        store.supersede("Entity", scope, "absent", "here", updated_by="qa")
    with pytest.raises(LookupError):
        store.supersede("Entity", scope, "here", "there", updated_by="qa")
    edges = probe.rows(
        "MATCH (:Entity)-[r:SUPERSEDES]->(:Entity) WHERE r IS NOT NULL "
        "AND startNode(r).scope STARTS WITH $prefix RETURN count(r) AS c",
        prefix=PREFIX,
    )
    assert edges[0]["c"] == 0
    assert probe.nodes("Entity", scope)[0]["status"] == "active"
