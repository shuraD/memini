"""The Neo4j graph store: one shared driver from the environment, append-only node writes.

:func:`get_driver` builds the process-wide :class:`neo4j.Driver` once, only from the
``NEO4J_URI``, ``NEO4J_USER`` and ``NEO4J_PASSWORD`` environment variables (no default, no
host refusal); :func:`close_driver` is the only way it is closed. :class:`GraphStore` uses that
driver: it connects fail-fast, ensures the ``(scope, id)`` uniqueness constraints, writes nodes
in one ``UNWIND`` batch per call (``MERGE`` on ``(scope, id)``, skipping an already-stored
``content_hash``) and supersedes an old node by a new one without deleting it.

The module-level functions work on a plain :class:`neo4j.Driver`: :func:`write_vectors`
(through ``neo4j_graphrag``'s ``upsert_vectors``), soft or hard :func:`delete`, vector search
(:func:`search_entities`, :func:`search_communities`, through ``VectorRetriever``) and
:func:`stats` of one scope.

The design follows the old memini's ``connect``/``remember``; the code is new.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import neo4j
from neo4j_graphrag.indexes import upsert_vectors
from neo4j_graphrag.retrievers import VectorRetriever

from memini.infra.graph.schema import GraphSchema, load_schema

# Environment variables of the connection settings (spec GR-05); none has a default.
_ENV_URI, _ENV_USER, _ENV_PASSWORD, _ENV_DATABASE = (
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
)

# Graph names the store relies on (spec GR-03); the schema file declares each of them.
_SCOPE = "scope"
_ID = "id"
_HASH = "content_hash"
_STATUS = "status"
_UPDATED_BY = "updated_by"
_UPDATED_AT = "updated_at"
_ACTIVE = "active"
_SUPERSEDED = "superseded"
_SUPERSEDES = "SUPERSEDES"
# Graph names of the vector, delete and stats functions (spec GR-04).
_DELETED = "deleted"
_SUMMARY = "summary"
_EMBEDDING = "embedding"  # the vector property, and the prefix of every vector property
_TYPE = "type"
_LEVEL = "level"
_ENTITY = "Entity"
_COMMUNITY = "Community"
_MEMBER_OF = "BELONGS_TO_COMMUNITY"
_TOMBSTONE_CHARS = 200  # characters of summary a soft-deleted node keeps


class _Shared:
    """The process-wide shared driver and the lock that guards its creation and closing."""

    driver: neo4j.Driver | None = None
    lock = threading.Lock()


def _required(env: str) -> str:
    """Return the env var ``env``; RuntimeError naming it when it is unset or empty."""
    value = os.environ.get(env, "")
    if not value:
        raise RuntimeError(f"environment variable {env} is unset or empty")
    return value


def get_driver() -> neo4j.Driver:
    """Return the shared driver, creating it from the ``NEO4J_*`` variables on first use."""
    with _Shared.lock:
        if _Shared.driver is None:
            uri, user, password = (_required(e) for e in (_ENV_URI, _ENV_USER, _ENV_PASSWORD))
            _Shared.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        return _Shared.driver


def close_driver() -> None:
    """Close the shared driver once and forget it; a no-op when none is open."""
    with _Shared.lock:
        driver, _Shared.driver = _Shared.driver, None
    if driver is not None:
        driver.close()


class GraphStore:
    """A Neo4j store of scoped nodes, written append-only, on the shared driver."""

    def __init__(self) -> None:
        """Read the database name (``NEO4J_DATABASE``, None when unset or empty); no I/O."""
        self.database: str | None = os.environ.get(_ENV_DATABASE) or None
        self._driver: neo4j.Driver | None = None
        self._schema: GraphSchema | None = None

    def connect(self) -> GraphStore:
        """Use the shared driver, verify connectivity, ensure the (scope, id) constraints."""
        driver = get_driver()
        self._driver = driver
        driver.verify_connectivity()
        schema = load_schema()
        for label, props in schema.nodes.items():
            if _ID in props:
                driver.execute_query(
                    f"CREATE CONSTRAINT {label.lower()}_{_SCOPE}_{_ID} IF NOT EXISTS "
                    f"FOR (n:`{label}`) REQUIRE (n.`{_SCOPE}`, n.`{_ID}`) IS UNIQUE",
                    database_=self.database,
                )
        self._schema = schema
        return self

    def close(self) -> None:
        """Drop this store's use of the shared driver; the shared driver stays open."""
        self._driver = None
        self._schema = None

    def _checked(self, label: str) -> tuple[neo4j.Driver, tuple[str, ...]]:
        """Return the open driver and ``label``'s properties; ValueError for an unknown label."""
        if self._driver is None or self._schema is None:
            raise RuntimeError("GraphStore is not connected; call connect() first")
        if label not in self._schema.nodes:
            raise ValueError(f"unknown node label {label!r}")
        return self._driver, self._schema.nodes[label]

    def write(
        self, label: str, scope: str, items: Sequence[Mapping[str, Any]], *, updated_by: str
    ) -> list[str]:
        """Merge ``items`` as ``label`` nodes of ``scope``; return the written ids in order."""
        driver, props = self._checked(label)
        status = f", n.`{_STATUS}` = $status" if _STATUS in props else ""
        records, _, _ = driver.execute_query(
            f"UNWIND $rows AS row "
            f"WITH row WHERE NOT EXISTS {{ MATCH (m:`{label}`) "
            f"WHERE m.`{_SCOPE}` = $scope AND m.`{_HASH}` = row.`{_HASH}` }} "
            f"MERGE (n:`{label}` {{`{_SCOPE}`: $scope, `{_ID}`: row.`{_ID}`}}) "
            f"SET n += row, n.`{_UPDATED_BY}` = $updated_by, "
            f"n.`{_UPDATED_AT}` = datetime(){status} "
            f"RETURN row.`{_ID}` AS id",
            {
                "rows": [dict(item) for item in items],
                "scope": scope,
                "updated_by": updated_by,
                "status": _ACTIVE,
            },
            database_=self.database,
        )
        written = {record["id"] for record in records}
        return [str(item[_ID]) for item in items if item[_ID] in written]

    def supersede(
        self, label: str, scope: str, new_id: str, old_id: str, *, updated_by: str
    ) -> None:
        """Link ``new`` -[:SUPERSEDES]-> ``old`` and mark ``old`` superseded; keep both."""
        driver, _ = self._checked(label)
        records, _, _ = driver.execute_query(
            f"MATCH (new:`{label}` {{`{_SCOPE}`: $scope, `{_ID}`: $new_id}}) "
            f"MATCH (old:`{label}` {{`{_SCOPE}`: $scope, `{_ID}`: $old_id}}) "
            f"MERGE (new)-[:`{_SUPERSEDES}`]->(old) "
            f"SET old.`{_STATUS}` = $status, old.`{_UPDATED_BY}` = $updated_by, "
            f"old.`{_UPDATED_AT}` = datetime() "
            f"RETURN count(*) AS linked",
            {
                "scope": scope,
                "new_id": new_id,
                "old_id": old_id,
                "status": _SUPERSEDED,
                "updated_by": updated_by,
            },
            database_=self.database,
        )
        if not records or not records[0]["linked"]:
            raise LookupError(f"{label} {new_id!r} or {old_id!r} not found in scope {scope!r}")


def write_vectors(
    driver: neo4j.Driver,
    label: str,
    prop: str,
    scope: str,
    vectors: Mapping[str, Sequence[float]],
    *,
    database: str | None = None,
) -> int:
    """Set ``prop`` to each vector on the ``label`` node ``(scope, id)``; return how many.

    Element ids are looked up first; an id with no node is ignored. The write itself goes
    through :func:`neo4j_graphrag.indexes.upsert_vectors`.
    """
    if not vectors:
        return 0
    records, _, _ = driver.execute_query(
        f"MATCH (n:`{label}`) WHERE n.`{_SCOPE}` = $scope AND n.`{_ID}` IN $ids "
        f"RETURN n.`{_ID}` AS id, elementId(n) AS eid",
        {"scope": scope, "ids": list(vectors)},
        database_=database,
    )
    if not records:
        return 0
    upsert_vectors(
        driver,
        ids=[str(r["eid"]) for r in records],
        embedding_property=prop,
        embeddings=[[float(x) for x in vectors[r["id"]]] for r in records],
        neo4j_database=database,
    )
    return len(records)


def delete(
    driver: neo4j.Driver,
    scope: str,
    *,
    node_id: str | None = None,
    hard: bool = False,
    database: str | None = None,
) -> int:
    """Delete one node (``node_id``) or every node of ``scope``; return how many changed.

    Hard: ``DETACH DELETE`` of the nodes, any label. Soft: only active nodes; ``status``
    becomes deleted, ``summary`` is cut to its first characters, every ``embedding*``
    property is removed and the community memberships are dropped.
    """
    where = f"n.`{_SCOPE}` = $scope" + ("" if node_id is None else f" AND n.`{_ID}` = $id")
    params: dict[str, Any] = {"scope": scope, "id": node_id}
    if hard:
        query = f"MATCH (n) WHERE {where} DETACH DELETE n RETURN count(*) AS c"
    else:
        params.update(status=_DELETED, active=_ACTIVE, keep=_TOMBSTONE_CHARS)
        query = (
            f"MATCH (n) WHERE {where} AND n.`{_STATUS}` = $active "
            f"OPTIONAL MATCH (n)-[m:`{_MEMBER_OF}`]->() DELETE m "
            f"WITH DISTINCT n "
            f"SET n.`{_STATUS}` = $status, "
            f"n.`{_SUMMARY}` = left(n.`{_SUMMARY}`, $keep), "
            f"n.`{_UPDATED_AT}` = datetime() "
            f"FOREACH (k IN [k IN keys(n) WHERE k STARTS WITH $prefix] | SET n[k] = null) "
            f"RETURN count(n) AS c"
        )
        params["prefix"] = _EMBEDDING
    records, _, _ = driver.execute_query(query, params, database_=database)
    return int(records[0]["c"]) if records else 0


def _vector_search(
    driver: neo4j.Driver,
    index_name: str,
    vector: Sequence[float],
    filters: dict[str, Any],
    *,
    top_k: int,
    database: str | None,
) -> list[tuple[str, float]]:
    """Run a filtered VectorRetriever search; return ``(id, score)`` best first."""
    retriever = VectorRetriever(
        driver, index_name, return_properties=[_ID], neo4j_database=database
    )
    result = retriever.get_search_results(
        query_vector=[float(x) for x in vector], top_k=top_k, filters=filters
    )
    hits = [(str(r["node"][_ID]), float(r["score"])) for r in result.records]
    hits.sort(key=lambda hit: hit[1], reverse=True)
    return hits[:top_k]


def search_entities(
    driver: neo4j.Driver,
    index_name: str,
    scope: str,
    vector: Sequence[float],
    *,
    top_k: int = 10,
    database: str | None = None,
) -> list[tuple[str, float]]:
    """Return the active nodes of ``scope`` nearest to ``vector`` in ``index_name``."""
    filters = {_SCOPE: {"$eq": scope}, _STATUS: {"$eq": _ACTIVE}}
    return _vector_search(driver, index_name, vector, filters, top_k=top_k, database=database)


def search_communities(
    driver: neo4j.Driver,
    index_name: str,
    scope: str,
    vector: Sequence[float],
    *,
    top_k: int = 10,
    level: int | None = None,
    database: str | None = None,
) -> list[tuple[str, float]]:
    """Return the communities of ``scope`` (only ``level`` when given) nearest to ``vector``."""
    filters: dict[str, Any] = {_SCOPE: {"$eq": scope}}
    if level is not None:
        filters[_LEVEL] = {"$eq": level}
    return _vector_search(driver, index_name, vector, filters, top_k=top_k, database=database)


def stats(driver: neo4j.Driver, scope: str, *, database: str | None = None) -> dict[str, Any]:
    """Count one scope: entities per status and type, relationships, coverage, communities."""

    def rows(query: str) -> list[neo4j.Record]:
        records, _, _ = driver.execute_query(query, {"scope": scope}, database_=database)
        return list(records)

    in_scope = f"WHERE n.`{_SCOPE}` = $scope"
    entity = f"MATCH (n:`{_ENTITY}`) {in_scope}"
    by_status = {
        str(r["k"]): int(r["c"])
        for r in rows(f"{entity} RETURN n.`{_STATUS}` AS k, count(*) AS c")
        if r["k"] is not None
    }
    by_type = {
        str(r["k"]): int(r["c"])
        for r in rows(f"{entity} RETURN n.`{_TYPE}` AS k, count(*) AS c")
        if r["k"] is not None
    }
    rels = {
        str(r["k"]): int(r["c"])
        for r in rows(
            f"MATCH (n)-[r]->(m) {in_scope} AND m.`{_SCOPE}` = $scope "
            f"RETURN type(r) AS k, count(r) AS c"
        )
    }
    cov = rows(f"{entity} RETURN count(n) AS t, count(n.`{_EMBEDDING}`) AS e")[0]
    comm = rows(
        f"MATCH (n:`{_COMMUNITY}`) {in_scope} "
        f"OPTIONAL MATCH ()-[m:`{_MEMBER_OF}`]->(n) "
        f"WITH n, count(m) AS size "
        f"RETURN count(n) AS c, sum(size) AS members, collect(DISTINCT n.`{_LEVEL}`) AS levels"
    )[0]
    total, communities = int(cov["t"]), int(comm["c"])
    return {
        "by_status": by_status,
        "by_type": by_type,
        "relationships": rels,
        "embedding_coverage": int(cov["e"]) / total if total else 0.0,
        "communities": communities,
        "levels": sorted(int(x) for x in comm["levels"]),
        "avg_community_size": int(comm["members"] or 0) / communities if communities else 0.0,
    }
