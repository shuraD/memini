"""The per-level GDS passes of the ArchRAG hierarchy, run through a graphdatascience client.

``similar_pass`` removes a scope's old SIMILAR edges, then writes new ones with one directed
KNN pass per Entity vector property (content, then code), each into its own SIMILAR property.
``leiden_pass`` runs one seeded Leiden pass on an undirected RELATED_TO|SIMILAR projection of
one hierarchy level: the scope's active entities at level 0, the communities of the level below
otherwise. Every graph name comes from the schema file (spec GR-06); every projected graph is
dropped before the pass returns. The design follows the old memini's ArchRAG indexing.

The pure helpers of incremental maintenance (``grouping_weight``, ``nearest_community``,
``running_mean``, ``consolidation_plan``, ``should_consolidate``, ``coherence``, ``density``)
make no database call; the caller passes every threshold in from config.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping, Sequence
from itertools import combinations
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


_PAIR = 2  # members in a pair


class Plan(enum.StrEnum):
    """What a consolidation does to a scope's hierarchy."""

    REBUILD = "rebuild"  # full rebuild of the scope's hierarchy
    PLACE = "place"  # local placement of the unplaced nodes
    REUSE = "reuse"  # keep the hierarchy as it is


def grouping_weight(content: float, structure: float, *, mode: str, blend: float = 0.5) -> float:
    """Return the level-0 grouping weight of a pair for ``mode``.

    ``content`` and ``structure`` give one side each; ``blend`` mixes them as
    ``blend * content + (1 - blend) * structure``. Any other mode raises ``ValueError``.
    """
    if mode == "content":
        return content
    if mode == "structure":
        return structure
    if mode == "blend":
        return blend * content + (1 - blend) * structure
    raise ValueError(f"unknown grouping mode {mode!r}")


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the cosine of ``a`` and ``b``; 0.0 when either has zero length."""
    norm = math.hypot(*a) * math.hypot(*b)
    if norm == 0.0:
        return 0.0
    return math.fsum(x * y for x, y in zip(a, b, strict=True)) / norm


def nearest_community(
    vector: Sequence[float], centroids: Mapping[str, Sequence[float]], *, threshold: float = 0.60
) -> str | None:
    """Return the id of the centroid closest to ``vector`` by cosine, if at least ``threshold``.

    A tie goes to the smallest id; ``None`` when no centroid reaches the threshold.
    """
    best: tuple[float, str] | None = None
    for cid in sorted(centroids):
        score = _cosine(vector, centroids[cid])
        if score >= threshold and (best is None or score > best[0]):
            best = (score, cid)
    return None if best is None else best[1]


def running_mean(centroid: Sequence[float], size: int, vector: Sequence[float]) -> list[float]:
    """Return the centroid of ``size`` members after one more member, ``vector``.

    Size 0 returns a copy of ``vector``; the inputs are not modified.
    """
    if size == 0:
        return list(vector)
    return [(c * size + v) / (size + 1) for c, v in zip(centroid, vector, strict=True)]


def consolidation_plan(
    *, has_hierarchy: bool, drift: float, unplaced: int, drift_threshold: float = 0.75
) -> Plan:
    """Choose the plan: rebuild without hierarchy or above the drift, else place or reuse."""
    if not has_hierarchy or drift > drift_threshold:
        return Plan.REBUILD
    return Plan.PLACE if unplaced > 0 else Plan.REUSE


def should_consolidate(
    *,
    seconds_since_last: float,
    writes_since_last: int,
    dirty_share: float,
    max_seconds: float,
    max_writes: int,
    max_dirty_share: float,
) -> bool:
    """Return whether time, write count or dirty share is at or above its threshold."""
    return (
        seconds_since_last >= max_seconds
        or writes_since_last >= max_writes
        or dirty_share >= max_dirty_share
    )


def coherence(vectors: Sequence[Sequence[float]]) -> float:
    """Return the mean pairwise cosine of ``vectors``; 1.0 for one vector, 0.0 for none."""
    if len(vectors) < _PAIR:
        return float(len(vectors))
    scores = [_cosine(a, b) for a, b in combinations(vectors, _PAIR)]
    return math.fsum(scores) / len(scores)


def density(members: int, internal_edges: int) -> float:
    """Return ``internal_edges`` over the possible pairs of ``members``; 0.0 below 2."""
    if members < _PAIR:
        return 0.0
    return internal_edges / (members * (members - 1) / _PAIR)
