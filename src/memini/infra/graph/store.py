"""The Neo4j graph store: one shared driver from the environment, append-only node writes.

:func:`get_driver` builds the process-wide :class:`neo4j.Driver` once, only from the
``NEO4J_URI``, ``NEO4J_USER`` and ``NEO4J_PASSWORD`` environment variables (no default, no
host refusal); :func:`close_driver` is the only way it is closed. :class:`GraphStore` uses that
driver: it connects fail-fast, ensures the ``(scope, id)`` uniqueness constraints, writes nodes
in one ``UNWIND`` batch per call (``MERGE`` on ``(scope, id)``, skipping an already-stored
``content_hash``) and supersedes an old node by a new one without deleting it.

The module-level functions work on a plain :class:`neo4j.Driver`: :func:`write_vectors`
(through ``neo4j_graphrag``'s ``upsert_vectors``), soft or hard :func:`delete`, vector search
(:func:`search_entities`, :func:`search_communities`, through ``VectorRetriever``; a level
or domain filter on a graph path through a Cypher ``SEARCH ... VECTOR INDEX``) and
:func:`stats` of one scope.

The design follows the old memini's ``connect``/``remember``; the code is new.
"""

from __future__ import annotations

import functools
import os
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import neo4j
from neo4j_graphrag.indexes import upsert_vectors
from neo4j_graphrag.retrievers import VectorRetriever

from memini.infra.graph.schema import GraphNames, GraphSchema, load_schema

# Environment variables of the connection settings (spec GR-05); none has a default.
_ENV_URI, _ENV_USER, _ENV_PASSWORD, _ENV_DATABASE = (
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
)

# Every graph name (label, relationship type, property, status) comes from the schema's
# names section (spec GR-06); Cypher parameters and record keys use neutral names.
_TOMBSTONE_CHARS = 200  # characters of summary a soft-deleted node keeps


@functools.cache
def _names() -> GraphNames:
    """Return the graph names of the configured schema file, read once per process."""
    return load_schema().names


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
        p = schema.names.properties
        for label, props in schema.nodes.items():
            if p.id in props:
                driver.execute_query(
                    f"CREATE CONSTRAINT {label.lower()}_{p.scope}_{p.id} IF NOT EXISTS "
                    f"FOR (n:`{label}`) REQUIRE (n.`{p.scope}`, n.`{p.id}`) IS UNIQUE",
                    database_=self.database,
                )
        self._schema = schema
        return self

    def close(self) -> None:
        """Drop this store's use of the shared driver; the shared driver stays open."""
        self._driver = None
        self._schema = None

    def _checked(self, label: str) -> tuple[neo4j.Driver, tuple[str, ...], GraphNames]:
        """Return the open driver, ``label``'s properties and the graph names.

        ValueError for an unknown label.
        """
        if self._driver is None or self._schema is None:
            raise RuntimeError("GraphStore is not connected; call connect() first")
        if label not in self._schema.nodes:
            raise ValueError(f"unknown node label {label!r}")
        return self._driver, self._schema.nodes[label], self._schema.names

    def write(
        self, label: str, scope: str, items: Sequence[Mapping[str, Any]], *, updated_by: str
    ) -> list[str]:
        """Merge ``items`` as ``label`` nodes of ``scope``; return the written ids in order."""
        driver, props, names = self._checked(label)
        p = names.properties
        status = f", n.`{p.status}` = $new_status" if p.status in props else ""
        records, _, _ = driver.execute_query(
            f"UNWIND $rows AS row "
            f"WITH row WHERE NOT EXISTS {{ MATCH (m:`{label}`) "
            f"WHERE m.`{p.scope}` = $part AND m.`{p.content_hash}` = row.`{p.content_hash}` }} "
            f"MERGE (n:`{label}` {{`{p.scope}`: $part, `{p.id}`: row.`{p.id}`}}) "
            f"SET n += row, n.`{p.updated_by}` = $author, "
            f"n.`{p.updated_at}` = datetime(){status} "
            f"RETURN row.`{p.id}` AS key",
            {
                "rows": [dict(item) for item in items],
                "part": scope,
                "author": updated_by,
                "new_status": names.statuses.active,
            },
            database_=self.database,
        )
        written = {record["key"] for record in records}
        return [str(item[p.id]) for item in items if item[p.id] in written]

    def supersede(
        self, label: str, scope: str, new_id: str, old_id: str, *, updated_by: str
    ) -> None:
        """Link ``new`` -[:SUPERSEDES]-> ``old`` and mark ``old`` superseded; keep both."""
        driver, _, names = self._checked(label)
        p = names.properties
        records, _, _ = driver.execute_query(
            f"MATCH (new:`{label}` {{`{p.scope}`: $part, `{p.id}`: $new_key}}) "
            f"MATCH (old:`{label}` {{`{p.scope}`: $part, `{p.id}`: $old_key}}) "
            f"MERGE (new)-[:`{names.relationships.supersedes}`]->(old) "
            f"SET old.`{p.status}` = $new_status, old.`{p.updated_by}` = $author, "
            f"old.`{p.updated_at}` = datetime() "
            f"RETURN count(*) AS linked",
            {
                "part": scope,
                "new_key": new_id,
                "old_key": old_id,
                "new_status": names.statuses.superseded,
                "author": updated_by,
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
    p = _names().properties
    records, _, _ = driver.execute_query(
        f"MATCH (n:`{label}`) WHERE n.`{p.scope}` = $part AND n.`{p.id}` IN $keys "
        f"RETURN n.`{p.id}` AS key, elementId(n) AS eid",
        {"part": scope, "keys": list(vectors)},
        database_=database,
    )
    if not records:
        return 0
    upsert_vectors(
        driver,
        ids=[str(r["eid"]) for r in records],
        embedding_property=prop,
        embeddings=[[float(x) for x in vectors[r["key"]]] for r in records],
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
    names = _names()
    p = names.properties
    where = f"n.`{p.scope}` = $part" + ("" if node_id is None else f" AND n.`{p.id}` = $key")
    params: dict[str, Any] = {"part": scope, "key": node_id}
    if hard:
        query = f"MATCH (n) WHERE {where} DETACH DELETE n RETURN count(*) AS c"
    else:
        params.update(
            new_status=names.statuses.deleted,
            live=names.statuses.active,
            keep=_TOMBSTONE_CHARS,
            prefix=p.embedding,
        )
        query = (
            f"MATCH (n) WHERE {where} AND n.`{p.status}` = $live "
            f"OPTIONAL MATCH (n)-[m:`{names.relationships.member_of}`]->() DELETE m "
            f"WITH DISTINCT n "
            f"SET n.`{p.status}` = $new_status, "
            f"n.`{p.summary}` = left(n.`{p.summary}`, $keep), "
            f"n.`{p.updated_at}` = datetime() "
            f"FOREACH (k IN [k IN keys(n) WHERE k STARTS WITH $prefix] | SET n[k] = null) "
            f"RETURN count(n) AS c"
        )
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
    p = _names().properties
    key = p.id
    retriever = VectorRetriever(
        driver, index_name, return_properties=[key], neo4j_database=database
    )
    result = retriever.get_search_results(
        query_vector=[float(x) for x in vector], top_k=top_k, filters=filters
    )
    hits = [(str(r["node"][key]), float(r[p.score])) for r in result.records]
    hits.sort(key=lambda hit: hit[1], reverse=True)
    return hits[:top_k]


def _cypher_search(
    driver: neo4j.Driver,
    index_name: str,
    label: str,
    vector: Sequence[float],
    *,
    where: str,
    params: dict[str, Any],
    top_k: int,
    database: str | None,
) -> list[tuple[str, float]]:
    """Run a ``SEARCH ... VECTOR INDEX`` filtered by the Cypher predicate ``where`` on ``n``.

    The index cannot pre-filter on a graph path, so the search widens its candidate count
    until ``top_k`` candidates pass ``where`` or the index has no more nodes: the result is
    the ``top_k`` nearest among the matching nodes, best first.
    """
    p = _names().properties
    query = (
        f"MATCH (n:`{label}`) SEARCH n IN (VECTOR INDEX `{index_name}` FOR $vector "
        f"LIMIT $fetch) SCORE AS score "
        f"WITH collect([n, score]) AS hits "
        f"CALL (hits) {{ UNWIND hits AS hit WITH hit[0] AS n, hit[1] AS score "
        f"WHERE {where} WITH n, score ORDER BY score DESC LIMIT $top "
        f"RETURN collect([n.`{p.id}`, score]) AS found }} "
        f"RETURN size(hits) AS seen, found"
    )
    fetch = max(top_k, 1)
    while True:
        params.update(vector=[float(x) for x in vector], fetch=fetch, top=top_k)
        records, _, _ = driver.execute_query(query, params, database_=database)
        seen, found = int(records[0]["seen"]), list(records[0]["found"])
        if len(found) >= top_k or seen < fetch:
            return [(str(key), float(score)) for key, score in found]
        fetch *= 4


def _domain_path(entity: str) -> str:
    """Return the Cypher path from a ``$domain`` Domain node of ``$part`` to ``entity``."""
    names = _names()
    lab, rel, p = names.labels, names.relationships, names.properties
    return (
        f"(:`{lab.domain}` {{`{p.scope}`: $part, `{p.name}`: $domain}})"
        f"-[:`{rel.contains}`]->(:`{lab.artifact}`)-[:`{rel.has_chunk}`]->(:`{lab.chunk}`)"
        f"-[:`{rel.has_entity}`]->{entity}"
    )


def search_entities(
    driver: neo4j.Driver,
    index_name: str,
    scope: str,
    vector: Sequence[float],
    *,
    top_k: int = 10,
    level: int | None = None,
    domain: str | None = None,
    database: str | None = None,
) -> list[tuple[str, float]]:
    """Return the active entities of ``scope`` nearest to ``vector`` in ``index_name``.

    ``level``: only direct members of a community of that level; ``domain``: only entities
    reached from that domain's chunks. Without them the search runs through VectorRetriever.
    """
    names = _names()
    p = names.properties
    if level is None and domain is None:
        filters = {p.scope: {"$eq": scope}, p.status: {"$eq": names.statuses.active}}
        return _vector_search(driver, index_name, vector, filters, top_k=top_k, database=database)
    where = [f"n.`{p.scope}` = $part", f"n.`{p.status}` = $live"]
    if level is not None:
        where.append(
            f"EXISTS {{ (n)-[:`{names.relationships.member_of}`]->(c:`{names.labels.community}`) "
            f"WHERE c.`{p.scope}` = $part AND c.`{p.level}` = $level }}"
        )
    if domain is not None:
        where.append(f"EXISTS {{ {_domain_path('(n)')} }}")
    params: dict[str, Any] = {
        "part": scope,
        "live": names.statuses.active,
        "level": level,
        "domain": domain,
    }
    return _cypher_search(
        driver,
        index_name,
        names.labels.entity,
        vector,
        where=" AND ".join(where),
        params=params,
        top_k=top_k,
        database=database,
    )


def search_communities(
    driver: neo4j.Driver,
    index_name: str,
    scope: str,
    vector: Sequence[float],
    *,
    top_k: int = 10,
    level: int | None = None,
    domain: str | None = None,
    database: str | None = None,
) -> list[tuple[str, float]]:
    """Return the communities of ``scope`` nearest to ``vector`` in ``index_name``.

    ``level``: only communities of that level; ``domain``: only communities with a member
    entity of that domain, directly or through lower communities. Without a domain the
    search runs through VectorRetriever.
    """
    names = _names()
    p = names.properties
    if domain is None:
        filters: dict[str, Any] = {p.scope: {"$eq": scope}}
        if level is not None:
            filters[p.level] = {"$eq": level}
        return _vector_search(driver, index_name, vector, filters, top_k=top_k, database=database)
    member = f"(:`{names.labels.entity}`)-[:`{names.relationships.member_of}`*1..]->(n)"
    where = [f"n.`{p.scope}` = $part", f"EXISTS {{ {_domain_path(member)} }}"]
    if level is not None:
        where.append(f"n.`{p.level}` = $level")
    return _cypher_search(
        driver,
        index_name,
        names.labels.community,
        vector,
        where=" AND ".join(where),
        params={"part": scope, "level": level, "domain": domain},
        top_k=top_k,
        database=database,
    )


def stats(driver: neo4j.Driver, scope: str, *, database: str | None = None) -> dict[str, Any]:
    """Count one scope: entities per status and type, relationships, coverage, communities.

    ``dirty_ratio`` is the share of the scope's communities marked dirty (0.0 without any).
    """

    def rows(query: str) -> list[neo4j.Record]:
        records, _, _ = driver.execute_query(query, {"part": scope}, database_=database)
        return list(records)

    names = _names()
    p = names.properties
    in_scope = f"WHERE n.`{p.scope}` = $part"
    entity = f"MATCH (n:`{names.labels.entity}`) {in_scope}"
    by_status = {
        str(r["k"]): int(r["c"])
        for r in rows(f"{entity} RETURN n.`{p.status}` AS k, count(*) AS c")
        if r["k"] is not None
    }
    by_type = {
        str(r["k"]): int(r["c"])
        for r in rows(f"{entity} RETURN n.`{p.type}` AS k, count(*) AS c")
        if r["k"] is not None
    }
    rels = {
        str(r["k"]): int(r["c"])
        for r in rows(
            f"MATCH (n)-[r]->(m) {in_scope} AND m.`{p.scope}` = $part "
            f"RETURN type(r) AS k, count(r) AS c"
        )
    }
    cov = rows(f"{entity} RETURN count(n) AS t, count(n.`{p.embedding}`) AS e")[0]
    comm = rows(
        f"MATCH (n:`{names.labels.community}`) {in_scope} "
        f"OPTIONAL MATCH ()-[m:`{names.relationships.member_of}`]->(n) "
        f"WITH n, count(m) AS size "
        f"RETURN count(n) AS c, sum(size) AS members, collect(DISTINCT n.`{p.level}`) AS levels, "
        f"count(CASE WHEN n.`{p.dirty}` = true THEN 1 END) AS dirty"
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
        "dirty_ratio": int(comm["dirty"]) / communities if communities else 0.0,
    }
