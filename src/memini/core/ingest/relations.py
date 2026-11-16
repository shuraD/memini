"""Relations extracted by an LLM: read, resolve to stored entity ids, merge, measure.

Pure functions: no I/O, no configuration reads, no model call. The caller passes
every setting (new-type marker, vocabulary, confidences, minimum quote length).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

_WHITESPACE = r"\s+"
_LIST_MARKER = r"^(?:[-*+•]|\d+[.)])\s+"
_REQUIRED_FIELDS = ("source", "target", "type")
_OPTIONAL_FIELDS = ("new_type", "description", "evidence")
_STATS_KEYS = (
    "raw",
    "resolved",
    "unresolved_end",
    "self",
    "invalid",
    "no_type",
    "new",
    "off_vocabulary",
)


@dataclass(frozen=True)
class ExtractedRelation:
    """One relation between two stored entities, as extracted from one or more windows."""

    source: str
    target: str
    type: str
    new: bool
    description: tuple[str, ...]
    evidence: tuple[str, ...]
    windows: int = 1


@dataclass(frozen=True)
class MeasuredRelation:
    """A relation with one found flag per evidence quote and its resulting confidence."""

    relation: ExtractedRelation
    evidence_found: tuple[bool, ...]
    confidence: float


def read_relations(reply: object) -> list[object]:
    """Return the reply's top-level ``relations`` list, or ``[]`` when there is none."""
    if isinstance(reply, dict):
        value = reply.get("relations")
        if isinstance(value, list):
            return value
    return []


def _text_tuple(value: object) -> tuple[str, ...]:
    """Return a stripped non-empty string as a 1-tuple, else ``()``."""
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()


def _resolve_end(
    key: int, end: str, id_map: Mapping[tuple[int, str], str], final_ids: frozenset[str]
) -> str | None:
    """Return the final id of one relation end written in reply ``key``, or ``None``."""
    name = end.strip()
    if not name:
        return None
    mapped = id_map.get((key, name))
    if mapped is not None:
        return mapped
    return name if name in final_ids else None


def resolve_relations(
    replies: Iterable[tuple[int, Sequence[object]]],
    id_map: Mapping[tuple[int, str], str],
    final_ids: Iterable[str],
    vocabulary: Iterable[str],
    new_marker: str,
) -> tuple[list[ExtractedRelation], dict[str, int]]:
    """Resolve raw relation items to stored entity ids and count why items were dropped.

    Checks run in order invalid, no_type, unresolved_end, self; each dropped item is
    counted once. ``new`` and ``off_vocabulary`` count only kept items.
    """
    known = frozenset(final_ids)
    vocab = frozenset(vocabulary)
    marker = new_marker.strip().casefold()
    stats = dict.fromkeys(_STATS_KEYS, 0)
    kept: list[ExtractedRelation] = []
    for key, items in replies:
        for raw in items:
            stats["raw"] += 1
            if (
                not isinstance(raw, dict)
                or not all(isinstance(raw.get(f), str) for f in _REQUIRED_FIELDS)
                or any(f in raw and not isinstance(raw[f], str) for f in _OPTIONAL_FIELDS)
            ):
                stats["invalid"] += 1
                continue
            type_name = str(raw["type"]).strip()
            by_marker = type_name.casefold() == marker
            if by_marker:
                type_name = str(raw.get("new_type", "")).strip()
            if not type_name:
                stats["no_type"] += 1
                continue
            source = _resolve_end(key, str(raw["source"]), id_map, known)
            target = _resolve_end(key, str(raw["target"]), id_map, known)
            if source is None or target is None:
                stats["unresolved_end"] += 1
                continue
            if source == target:
                stats["self"] += 1
                continue
            off_vocabulary = not by_marker and type_name not in vocab
            stats["resolved"] += 1
            stats["new"] += int(by_marker)
            stats["off_vocabulary"] += int(off_vocabulary)
            kept.append(
                ExtractedRelation(
                    source=source,
                    target=target,
                    type=type_name,
                    new=by_marker or off_vocabulary,
                    description=_text_tuple(raw.get("description")),
                    evidence=_text_tuple(raw.get("evidence")),
                )
            )
    return kept, stats


def _union(first: tuple[str, ...], second: tuple[str, ...]) -> tuple[str, ...]:
    """Return ``first`` followed by the items of ``second`` it does not hold yet."""
    return tuple(dict.fromkeys(first + second))


def merge_relations(relations: Iterable[ExtractedRelation]) -> list[ExtractedRelation]:
    """Fold relations sharing (source, target, type) into one, in first-seen order."""
    merged: dict[tuple[str, str, str], ExtractedRelation] = {}
    for relation in relations:
        key = (relation.source, relation.target, relation.type)
        seen = merged.get(key)
        if seen is None:
            merged[key] = relation
            continue
        merged[key] = ExtractedRelation(
            source=seen.source,
            target=seen.target,
            type=seen.type,
            new=seen.new and relation.new,
            description=_union(seen.description, relation.description),
            evidence=_union(seen.evidence, relation.evidence),
            windows=seen.windows + relation.windows,
        )
    return list(merged.values())


def _normalize(text: str) -> str:
    """Lower-case ``text``, collapse whitespace runs to one space and strip it."""
    return re.sub(_WHITESPACE, " ", text).strip().casefold()


def quote_found(quote: str, text: str, min_chars: int) -> bool:
    """Tell whether ``quote`` appears in ``text``, ignoring case, spacing and a list marker.

    A quote whose normalized length is below ``max(1, min_chars)`` is never found.
    """
    needle = re.sub(_LIST_MARKER, "", _normalize(quote))
    if len(needle) < max(1, min_chars):
        return False
    return needle in _normalize(text)


def measure_relations(
    relations: Iterable[ExtractedRelation],
    source_text: str,
    confidence_verbatim: float,
    confidence_unverified: float,
    min_evidence_chars: int,
) -> list[MeasuredRelation]:
    """Give each relation a confidence from finding its evidence quotes in ``source_text``.

    The verbatim confidence needs at least one quote and every quote found.
    """
    measured: list[MeasuredRelation] = []
    for relation in relations:
        found = tuple(
            quote_found(quote, source_text, min_evidence_chars) for quote in relation.evidence
        )
        verified = bool(found) and all(found)
        measured.append(
            MeasuredRelation(
                relation=relation,
                evidence_found=found,
                confidence=confidence_verbatim if verified else confidence_unverified,
            )
        )
    return measured
