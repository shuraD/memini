"""The per-level GDS passes of the ArchRAG hierarchy, run through a graphdatascience client.

``similar_pass`` removes a scope's old SIMILAR edges, then writes new ones with one directed
KNN pass per Entity vector property (content, then code), each into its own SIMILAR property.
``leiden_pass`` runs one seeded Leiden pass on an undirected RELATED_TO|SIMILAR projection of
one hierarchy level: the scope's active entities at level 0, the communities of the level below
otherwise. Every graph name comes from the schema file (spec GR-06); every projected graph is
dropped before the pass returns. The design follows the old memini's ArchRAG indexing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from memini.infra.graph.schema import GraphSchema, load_schema


def _project(gds: Any, query: str, **params: Any) -> Any:
    """Project ``query`` (a Cypher aggregation on ``$graph_name``) and return the graph."""
    graph, _ = gds.graph.cypher.project(query, graph_name=uuid4().hex, **params)
    return graph


def _qualifiers(names: Sequence[str], base: Sequence[str]) -> dict[str, str]:
    """Map each name to its qualifier: its ``_``-separated words not in ``base``, re-joined."""
    return {name: "_".join(w for w in name.split("_") if w not in base) for name in names}


def _vector_scores(schema: GraphSchema) -> list[tuple[str, str]]:
    """Pair each Entity vector property with the SIMILAR property its KNN scores go to.

    A vector property is the embedding name, optionally qualified (``embedding_code``: the
    code vector). Its SIMILAR property is the one with the same qualifier, a SIMILAR
    property's qualifier being its words not shared by every SIMILAR property (``score``:
    none, so the content vector; ``code_score``: ``code``). Pairing is by name, never by
    position; a vector with no SIMILAR property of its qualifier raises ``ValueError``.
    """
    names = schema.names
    embedding = names.properties.embedding
    vectors = [
        prop
        for prop in schema.nodes[names.labels.entity]
        if prop == embedding or prop.startswith(f"{embedding}_")
    ]
    similar = schema.relationships[names.relationships.similar]
    shared = [w for w in similar[0].split("_") if all(w in p.split("_") for p in similar)]
    scores = {qualifier: prop for prop, qualifier in _qualifiers(similar, shared).items()}
    pairs: list[tuple[str, str]] = []
    for vector, qualifier in _qualifiers(vectors, embedding.split("_")).items():
        if qualifier not in scores:
            raise ValueError(f"no SIMILAR property for the vector property {vector!r}")
        pairs.append((vector, scores[qualifier]))
    return pairs


def similar_pass(
    gds: Any, scope: str, *, top_k: int = 10, cutoff: float = 0.7, seed: int = 42
) -> None:
    """Rewrite the SIMILAR edges of ``scope`` with one directed KNN pass per vector property."""
    schema = load_schema()
    names = schema.names
    p = names.properties
    entity, similar = names.labels.entity, names.relationships.similar
    gds.run_cypher(
        f"MATCH (a:`{entity}`)-[r:`{similar}`]->(b:`{entity}`) "
        f"WHERE a.`{p.scope}` = $scope OR b.`{p.scope}` = $scope DELETE r",
        {"scope": scope},
    )
    for vector, score in _vector_scores(schema):
        graph = _project(
            gds,
            f"MATCH (n:`{entity}`) WHERE n.`{p.scope}` = $scope AND n.`{p.status}` = $status "
            f"AND n.`{vector}` IS NOT NULL "
            f"RETURN gds.graph.project($graph_name, n, null, "
            f"{{sourceNodeProperties: n {{ .`{vector}` }}}})",
            scope=scope,
            status=names.statuses.active,
        )
        try:
            gds.knn.write(
                graph,
                nodeProperties=[vector],
                topK=top_k,
                similarityCutoff=cutoff,
                writeRelationshipType=similar,
                writeProperty=score,
                randomSeed=seed,
                concurrency=1,
            )
        finally:
            graph.drop()


def leiden_pass(
    gds: Any,
    scope: str,
    level: int,
    *,
    resolutions: Sequence[float] = (4.0, 1.0, 0.25),
    seed: int = 42,
) -> None:
    """Run one seeded Leiden pass on hierarchy ``level`` of ``scope``.

    Level 0 clusters the scope's active entities; level ``L > 0`` clusters the scope's
    communities of level ``L - 1``. The resolution is ``resolutions[level]``, or the last
    one beyond the schedule. Each node's community id is written to the property named
    by the schema's typed Community label, used unchanged (the typed names have no
    community-id property role yet).
    """
    names = load_schema().names
    p, labels, rels = names.properties, names.labels, names.relationships
    params: dict[str, Any]
    if level == 0:
        label, field, params = labels.entity, p.status, {"value": names.statuses.active}
    else:
        label, field, params = labels.community, p.level, {"value": level - 1}
    node = f"`{label}` {{`{p.scope}`: $scope, `{field}`: $value}}"
    graph = _project(
        gds,
        f"MATCH (s:{node}) "
        f"OPTIONAL MATCH (s)-[:`{rels.related_to}`|`{rels.similar}`]->(t:{node}) "
        f"RETURN gds.graph.project($graph_name, s, t, {{}}, "
        f"{{undirectedRelationshipTypes: ['*']}})",
        scope=scope,
        **params,
    )
    try:
        gds.leiden.write(
            graph,
            gamma=resolutions[min(level, len(resolutions) - 1)],
            randomSeed=seed,
            concurrency=1,
            writeProperty=labels.community,
        )
    finally:
        graph.drop()
