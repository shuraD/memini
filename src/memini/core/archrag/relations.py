"""Pure relation rules shared by every relation producer.

No I/O, no graph, no clock. The caller passes the config sections: settings are deep-merged
over config-given defaults, relationship type names are read from config and checked, relation
type names are normalised, and one merge rule folds relations into one per (src, dst, type):
lists accumulate (existing items first, no duplicates), the strongest origin wins (earliest in
``origin_rank``) and the highest known confidence wins.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

__all__ = [
    "Relation",
    "RelationSettings",
    "edge_type",
    "fold_relations",
    "merge_relation",
    "merge_settings",
    "normalize_relation_type",
]

_EDGE_TYPE = r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*"  # UPPER_SNAKE, matched in full
_WORD = r"[^\W_]+"  # a run of letters/digits


def merge_settings(
    defaults: Mapping[str, Any], overrides: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Deep-merge ``overrides`` over ``defaults`` without mutating either.

    A key whose value is a mapping on both sides is merged recursively; an override of ``None``
    keeps the default; keys only in ``overrides`` are added. Nested mappings are copied.
    """
    result: dict[str, Any] = {
        key: merge_settings(value, None) if isinstance(value, Mapping) else value
        for key, value in defaults.items()
    }
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        current = result.get(key)
        if isinstance(value, Mapping):
            result[key] = merge_settings(current if isinstance(current, Mapping) else {}, value)
        else:
            result[key] = value
    return result


def edge_type(edges: Mapping[str, Any], role: str) -> str:
    """Return the relationship type name configured for ``role``, checked as UPPER_SNAKE."""
    if role not in edges:
        raise ValueError(f"no relationship type configured for role {role!r}")
    name = edges[role]
    if not isinstance(name, str) or not re.fullmatch(_EDGE_TYPE, name):
        raise ValueError(f"relationship type for role {role!r} is not UPPER_SNAKE: {name!r}")
    return name


def normalize_relation_type(name: object) -> str:
    """Lower-case the words (runs of letters/digits) of ``name`` and join them with ``_``."""
    words = re.findall(_WORD, name) if isinstance(name, str) else []
    if not words:
        raise ValueError(f"relation type has no word: {name!r}")
    return "_".join(word.lower() for word in words)


@dataclass(frozen=True)
class Relation:
    """A directed, typed relation with its accumulated provenance."""

    src: str
    dst: str
    type: str
    origin: str
    description: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    origins: tuple[str, ...] = ()
    confidence: float | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        """The merge key ``(src, dst, type)``."""
        return (self.src, self.dst, self.type)


@dataclass(frozen=True)
class RelationSettings:
    """Origin ranking (strongest first) and the default confidence of each origin."""

    origin_rank: tuple[str, ...]
    default_confidence: Mapping[str, float | None]

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> RelationSettings:
        """Read ``origin_rank`` and ``origins`` from a section; raise ValueError if malformed."""
        rank = section.get("origin_rank")
        if not isinstance(rank, list) or not rank or not all(isinstance(o, str) for o in rank):
            raise ValueError(f"origin_rank must be a non-empty list of str: {rank!r}")
        origins = section.get("origins")
        if not isinstance(origins, Mapping):
            raise ValueError(f"origins must be a mapping: {origins!r}")
        confidences: dict[str, float | None] = {}
        for origin, spec in origins.items():
            if not isinstance(origin, str) or not isinstance(spec, Mapping):
                raise ValueError(f"origin {origin!r} must map to a mapping: {spec!r}")
            value = spec.get("confidence")
            is_number = isinstance(value, int | float) and not isinstance(value, bool)
            if value is not None and not is_number:
                raise ValueError(f"confidence of origin {origin!r} must be a number: {value!r}")
            confidences[origin] = None if value is None else float(value)
        missing = [origin for origin in rank if origin not in confidences]
        if missing:
            raise ValueError(f"ranked origins absent from origins: {missing!r}")
        return cls(origin_rank=tuple(rank), default_confidence=confidences)


def _union(first: tuple[str, ...], second: tuple[str, ...]) -> tuple[str, ...]:
    """Items of ``first`` then ``second``, first occurrence kept."""
    return tuple(dict.fromkeys((*first, *second)))


def _rank(origin: str, settings: RelationSettings) -> int:
    """Position of ``origin`` in the ranking; unranked origins rank after every ranked one."""
    rank = settings.origin_rank
    return rank.index(origin) if origin in rank else len(rank)


def merge_relation(
    existing: Relation | None, new: Relation, settings: RelationSettings
) -> Relation:
    """Merge ``new`` into ``existing``, or start a relation from ``new`` when there is none."""
    if new.origin not in settings.default_confidence:
        raise ValueError(f"unknown origin: {new.origin!r}")
    confidence = new.confidence
    if confidence is None:
        confidence = settings.default_confidence[new.origin]
    if existing is None:
        existing = Relation(new.src, new.dst, new.type, new.origin, confidence=confidence)
    elif existing.key != new.key:
        raise ValueError(f"cannot merge different keys: {existing.key!r} != {new.key!r}")
    known = [c for c in (existing.confidence, confidence) if c is not None]
    origin = min((existing.origin, new.origin), key=lambda o: _rank(o, settings))
    return Relation(
        src=new.src,
        dst=new.dst,
        type=new.type,
        origin=origin,
        description=_union(existing.description, new.description),
        evidence=_union(existing.evidence, new.evidence),
        evidence_refs=_union(existing.evidence_refs, new.evidence_refs),
        origins=_union(existing.origins, (new.origin,)),
        confidence=max(known) if known else None,
    )


def fold_relations(relations: Iterable[Relation], settings: RelationSettings) -> list[Relation]:
    """Fold relations into one per (src, dst, normalised type), in order of first appearance.

    Self-loops and relations with an empty end are dropped; the input is not mutated.
    """
    folded: dict[tuple[str, str, str], Relation] = {}
    for rel in relations:
        if not rel.src or not rel.dst or rel.src == rel.dst:
            continue
        key = (rel.src, rel.dst, normalize_relation_type(rel.type))
        folded[key] = merge_relation(folded.get(key), replace(rel, type=key[2]), settings)
    return list(folded.values())
