"""Pure linking and relation rules for the ArchRAG graph.

Reading-order sequence pairs, the specific-label rule, relation type
normalisation, the one-relation-per-(source, target, type) merge rule and
flagging of types not yet in a scope's vocabulary. No database, no I/O.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise

_NON_WORD = r"[\W_]+"


@dataclass(frozen=True)
class Relation:
    """A typed, directed relation between two entities."""

    source: str
    target: str
    type: str
    origin: str
    confidence: float
    evidence: tuple[str, ...] = ()


def sequence_pairs(entity_ids: Sequence[str]) -> list[tuple[str, str]]:
    """Return consecutive (previous, next) pairs in reading order.

    Self pairs are skipped and a pair already returned is not repeated.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in pairwise(entity_ids):
        if pair[0] != pair[1] and pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def specific_labels(
    labels_by_artifact: Mapping[str, Iterable[str]],
    min_artifacts: int = 2,
    max_share: float = 0.2,
) -> set[str]:
    """Return the labels specific enough to link artifacts.

    A label is kept when held by at least ``min_artifacts`` distinct artifacts
    and by at most ``max_share`` of all artifacts; repeats inside one artifact
    count once.
    """
    total = len(labels_by_artifact)
    if total == 0:
        return set()
    counts: dict[str, int] = {}
    for labels in labels_by_artifact.values():
        for label in set(labels):
            counts[label] = counts.get(label, 0) + 1
    return {
        label
        for label, count in counts.items()
        if count >= min_artifacts and count / total <= max_share
    }


def normalise_type(name: str) -> str:
    """Normalise a relation type name.

    The name is stripped and upper-cased, each run of characters that are not
    letters or digits becomes one ``_``, and leading or trailing ``_`` is dropped.
    """
    return re.sub(_NON_WORD, "_", name.strip().upper()).strip("_")


def merge_relations(relations: Iterable[Relation], origin_rank: Sequence[str]) -> list[Relation]:
    """Merge relations into one per (source, target, normalised type).

    Order is first appearance. Evidence is concatenated without repeats, the
    highest confidence wins, and the strongest origin (earliest in
    ``origin_rank``; unlisted origins are weakest, first seen stays) wins.
    """
    rank = {origin: index for index, origin in enumerate(origin_rank)}
    unlisted = len(rank)
    merged: dict[tuple[str, str, str], Relation] = {}
    for rel in relations:
        kind = normalise_type(rel.type)
        key = (rel.source, rel.target, kind)
        current = merged.get(key)
        if current is None:
            merged[key] = replace(rel, type=kind, evidence=tuple(dict.fromkeys(rel.evidence)))
            continue
        stronger = rank.get(rel.origin, unlisted) < rank.get(current.origin, unlisted)
        merged[key] = replace(
            current,
            origin=rel.origin if stronger else current.origin,
            confidence=max(current.confidence, rel.confidence),
            evidence=tuple(dict.fromkeys(current.evidence + rel.evidence)),
        )
    return list(merged.values())


def new_types(types: Iterable[str], vocabulary: Iterable[str]) -> list[str]:
    """Return sorted, de-duplicated normalised types missing from the normalised vocabulary."""
    known = {normalise_type(name) for name in vocabulary}
    return sorted({normalise_type(name) for name in types} - known)
