"""Client-side scoring and filtering for community and entity recall.

Rows are already fetched (e.g. by an ANN query or a scan); this module only
ranks them against a query vector by cosine similarity and applies the scope,
level, band and domain filters. Community summaries are returned exactly as
precomputed: no model or embedder is ever called.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "CommunityHit",
    "EntityHit",
    "cosine",
    "oversample",
    "recall_communities",
    "search_entities",
]

_OVERSAMPLE_FACTOR = 20
_OVERSAMPLE_FLOOR = 100


@dataclass(frozen=True)
class CommunityHit:
    """A community ranked against a query, with its precomputed summary."""

    id: str
    level: int
    summary: str
    size: int
    members: tuple[str, ...]
    score: float


@dataclass(frozen=True)
class EntityHit:
    """An entity ranked against a query."""

    id: str
    domain: str | None
    score: float


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the cosine similarity of ``a`` and ``b``.

    Returns 0.0 when either vector has zero norm; raises ValueError when the
    lengths differ.
    """
    if len(a) != len(b):
        raise ValueError(f"vector lengths differ: {len(a)} != {len(b)}")
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b)


def oversample(k: int) -> int:
    """Return the ANN candidate count for a top-``k`` recall.

    Raises ValueError when ``k`` < 1.
    """
    _check_k(k)
    return max(k * _OVERSAMPLE_FACTOR, _OVERSAMPLE_FLOOR)


def _check_k(k: int) -> None:
    """Raise ValueError when ``k`` is below one."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def recall_communities(
    query: Sequence[float],
    rows: Iterable[Mapping[str, Any]],
    *,
    k: int,
    scope: str,
    band: Literal["all", "global"] = "all",
    level: int | None = None,
    domain: str | None = None,
    member_domains: Mapping[str, str] | None = None,
) -> list[CommunityHit]:
    """Rank community rows against ``query`` and return the top ``k``.

    Rows carry keys "id", "scope", "level", "summary", "size", "members" and
    "embedding". Only rows in ``scope`` with an embedding are kept;
    ``band="global"`` keeps the highest level present in the scope, ``level``
    keeps one level, and ``domain`` keeps communities with at least one member
    whose ``member_domains`` entry equals it. Ordered by score descending, then
    id ascending.
    """
    _check_k(k)
    if domain is not None and member_domains is None:
        raise ValueError("domain filter requires member_domains")
    in_scope = [row for row in rows if row.get("scope") == scope]
    if band == "global" and in_scope:
        top = max(int(row["level"]) for row in in_scope)
        in_scope = [row for row in in_scope if int(row["level"]) == top]
    hits: list[CommunityHit] = []
    for row in in_scope:
        embedding = row.get("embedding")
        if embedding is None:
            continue
        if level is not None and int(row["level"]) != level:
            continue
        members = tuple(str(m) for m in row["members"])
        if (
            domain is not None
            and member_domains is not None
            and not any(member_domains.get(m) == domain for m in members)
        ):
            continue
        hits.append(
            CommunityHit(
                id=str(row["id"]),
                level=int(row["level"]),
                summary=row["summary"],
                size=int(row["size"]),
                members=members,
                score=cosine(query, embedding),
            )
        )
    hits.sort(key=lambda h: (-h.score, h.id))
    return hits[:k]


def search_entities(
    query: Sequence[float],
    rows: Iterable[Mapping[str, Any]],
    *,
    k: int,
    scope: str,
    domain: str | None = None,
) -> list[EntityHit]:
    """Rank entity rows against ``query`` and return the top ``k``.

    Rows carry keys "id", "scope", "domain" (may be absent or None) and
    "embedding". Only rows in ``scope`` with an embedding, and whose domain
    equals ``domain`` when given, are kept. Ordered by score descending, then
    id ascending.
    """
    _check_k(k)
    hits: list[EntityHit] = []
    for row in rows:
        if row.get("scope") != scope:
            continue
        embedding = row.get("embedding")
        if embedding is None:
            continue
        row_domain = row.get("domain")
        if domain is not None and row_domain != domain:
            continue
        hits.append(
            EntityHit(
                id=str(row["id"]),
                domain=None if row_domain is None else str(row_domain),
                score=cosine(query, embedding),
            )
        )
    hits.sort(key=lambda h: (-h.score, h.id))
    return hits[:k]
