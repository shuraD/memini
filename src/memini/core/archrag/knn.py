"""Directed cosine KNN pairs and the statements that regenerate a scope's similarity edges.

Two passes share the same machinery: a semantic pass on the entities' attribute embeddings,
whose edges carry ``score``, and a code pass on their code embeddings, whose edges carry
``code_score``. Each pass first clears its own property from the edges among the call's ids
(deleting an edge left with neither property), then merges the new pairs. Nothing is read
from or executed against the database here; the caller runs the returned statements.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from memini.core.graph import checks
from memini.core.graph.types import Statement
from memini.core.graph.writes import WriteBuilder
from memini.core.schema import Schema, node_key

__all__ = ["KnnSettings", "code_similar_statements", "nearest", "similar_statements"]

_REL = "SIMILAR"
_LABEL = "Entity"
_SCORE = "score"
_CODE_SCORE = "code_score"


def _top_k(section: Mapping[str, Any], key: str) -> int:
    """Return ``section[key]`` when it is an int >= 1; raise ``ValueError`` otherwise."""
    if key not in section:
        raise ValueError(f"knn config misses {key!r}")
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"knn config {key!r} must be an int >= 1, got {value!r}")
    return value


def _cutoff(section: Mapping[str, Any], key: str) -> float:
    """Return ``section[key]`` when it is a number in [-1.0, 1.0]; else raise ``ValueError``."""
    if key not in section:
        raise ValueError(f"knn config misses {key!r}")
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int | float) or not -1.0 <= value <= 1.0:
        raise ValueError(f"knn config {key!r} must be a number in [-1.0, 1.0], got {value!r}")
    return float(value)


@dataclass(frozen=True)
class KnnSettings:
    """Neighbour counts and minimum cosines of the semantic and code passes.

    Attributes:
        top_k: Neighbours per entity, semantic pass (>= 1).
        cutoff: Minimum cosine, semantic pass, in [-1.0, 1.0].
        code_top_k: Neighbours per entity, code pass (>= 1).
        code_cutoff: Minimum cosine, code pass, in [-1.0, 1.0].

    """

    top_k: int
    cutoff: float
    code_top_k: int
    code_cutoff: float

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> KnnSettings:
        """Read keys "top_k", "cutoff", "code_top_k", "code_cutoff".

        Raises:
            ValueError: When a key is missing, a top_k is not an int >= 1, or a cutoff is
                not a number in [-1.0, 1.0].

        """
        return cls(
            top_k=_top_k(section, "top_k"),
            cutoff=_cutoff(section, "cutoff"),
            code_top_k=_top_k(section, "code_top_k"),
            code_cutoff=_cutoff(section, "code_cutoff"),
        )


def nearest(
    vectors: Mapping[str, Sequence[float]], *, top_k: int, cutoff: float
) -> list[tuple[str, str, float]]:
    """Directed KNN by cosine similarity.

    For each id, its ``top_k`` other ids with score >= ``cutoff``, as
    ``(from_id, to_id, score)``. Never a self pair; a zero-norm vector takes part in no pair;
    ties are broken by ``to_id`` ascending; pairs are not symmetrised. Result order: from_id
    ascending, then score descending, then to_id ascending.

    Raises:
        ValueError: When the vectors differ in length or ``top_k`` < 1.

    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if len({len(v) for v in vectors.values()}) > 1:
        raise ValueError("vectors differ in length")
    units: dict[str, list[float]] = {}
    for key in sorted(vectors):
        norm = math.sqrt(math.fsum(x * x for x in vectors[key]))
        if norm > 0.0:
            units[key] = [x / norm for x in vectors[key]]
    pairs: list[tuple[str, str, float]] = []
    for a, ua in units.items():
        scored = [
            (b, math.fsum(x * y for x, y in zip(ua, ub, strict=True)))
            for b, ub in units.items()
            if b != a
        ]
        kept = sorted((c for c in scored if c[1] >= cutoff), key=lambda c: (-c[1], c[0]))
        pairs.extend((a, b, s) for b, s in kept[:top_k])
    return pairs


def _statements(
    schema: Schema,
    scope: str,
    vectors: Mapping[str, Sequence[float]],
    *,
    prefix: str,
    prop: str,
    top_k: int,
    cutoff: float,
) -> list[Statement]:
    """Build the clear statement of one pass followed by the merges of its pairs."""
    builder = WriteBuilder(schema, scope=scope, clock=lambda: None)
    for key in vectors:
        if not isinstance(key, str) or not key:
            raise ValueError(f"ids must be non-empty strings: {key!r}")
    node = schema.nodes.get(_LABEL)
    key_prop = node_key(node) if node is not None else None
    if key_prop is None:
        raise ValueError(f"label {_LABEL!r} declares no key")
    others = [p for p in (_SCORE, _CODE_SCORE) if p != prop]
    remaining = " AND ".join(f"r.{p} IS NULL" for p in [prop, *others])
    cypher = (
        f"MATCH (a:{_LABEL} {{scope: $scope}})-[r:{_REL}]->(b:{_LABEL} {{scope: $scope}}) "
        f"WHERE a.{key_prop} IN $ids AND b.{key_prop} IN $ids "
        f"REMOVE r.{prop} "
        f"WITH r WHERE {remaining} DELETE r"
    )
    checks.require_labeled(cypher)
    clear = Statement(
        name=f"{prefix}:clear",
        cypher=cypher,
        params={"scope": scope, "ids": sorted(vectors)},
        kind="write",
        scope=scope,
    )
    rows = [
        {"from_id": a, "to_id": b, "props": {prop: s}}
        for a, b, s in nearest(vectors, top_k=top_k, cutoff=cutoff)
    ]
    writes = builder.merge_relationships(_REL, rows, from_label=_LABEL, to_label=_LABEL)
    return [clear, *writes]


def similar_statements(
    schema: Schema, scope: str, vectors: Mapping[str, Sequence[float]], settings: KnnSettings
) -> list[Statement]:
    """Semantic pass (vectors = the entities' ``embedding``, one hierarchy level per call).

    Returns the "knn:clear" statement followed by the merges of ``nearest(top_k, cutoff)``
    with props ``{"score": s}``. Raises ``ValueError`` for an empty scope or a bad id.
    """
    return _statements(
        schema,
        scope,
        vectors,
        prefix="knn",
        prop=_SCORE,
        top_k=settings.top_k,
        cutoff=settings.cutoff,
    )


def code_similar_statements(
    schema: Schema, scope: str, vectors: Mapping[str, Sequence[float]], settings: KnnSettings
) -> list[Statement]:
    """Code pass (vectors = the entities' ``embedding_code``).

    Returns the "knn_code:clear" statement followed by the merges of
    ``nearest(code_top_k, code_cutoff)`` with props ``{"code_score": s}``. Raises
    ``ValueError`` for an empty scope or a bad id.
    """
    return _statements(
        schema,
        scope,
        vectors,
        prefix="knn_code",
        prop=_CODE_SCORE,
        top_k=settings.code_top_k,
        cutoff=settings.code_cutoff,
    )
