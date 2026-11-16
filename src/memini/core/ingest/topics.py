"""File-level topics, summary and derived labels from parsed extraction replies (card I63, I64).

Pure functions: no I/O, no model, no clock. The caller passes the parsed model replies, the
scope's existing topic vocabulary and every limit; nothing here holds a word list.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

_STRIP_CHARS = "\"'`.,;:!?()[]{}<>#* \t\r\n\f\v"
_TOPICS = "topics"
_SUMMARY = "summary"


class LabelSource(enum.StrEnum):
    """Provenance of one label on an artifact."""

    USER = "user"
    ROLE = "role"
    DOMAIN = "domain"
    TOPIC = "topic"
    WORK_UNIT = "work_unit"


@dataclass(frozen=True)
class DerivedLabels:
    """The artifact's one label list, in order, with who supplied each label."""

    labels: tuple[str, ...]
    sources: dict[str, LabelSource]


def _key(text: str) -> str:
    """Return the identity of a topic or label: lower-cased, alphanumerics only."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _clean(text: str) -> str:
    """Collapse inner whitespace, strip surrounding punctuation and lower-case."""
    return " ".join(text.split()).strip(_STRIP_CHARS).lower()


def _first(items: list[str], limit: int | None) -> list[str]:
    """Return the first ``limit`` items, or all of them when ``limit`` is None."""
    return items if limit is None else items[:limit]


def normalize_topics(
    raw: object, vocabulary: Iterable[str] = (), limit: int | None = None
) -> list[str]:
    """Normalize one reply's topics, reusing the vocabulary's spelling for a known key.

    ``raw`` may be one string or a list/tuple of strings; anything else gives ``[]``.
    Non-string items and topics with an empty key are dropped; the first of each key wins.
    """
    if isinstance(raw, str):
        items: list[object] = [raw]
    elif isinstance(raw, list | tuple):
        items = list(raw)
    else:
        return []
    known: dict[str, str] = {}
    for word in vocabulary:
        spelling = word.strip()
        known.setdefault(_key(spelling), spelling)
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        topic = _clean(item)
        key = _key(topic)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(known.get(key, topic))
    return _first(result, limit)


def union_topics(per_reply: Iterable[Iterable[str]], limit: int | None = None) -> list[str]:
    """Merge per-reply topic lists, ranked by how many replies named each topic.

    Ties keep first appearance; same-key topics merge under their first spelling.
    """
    spelling: dict[str, str] = {}
    counts: dict[str, int] = {}
    for topics in per_reply:
        reply_keys: set[str] = set()
        for topic in topics:
            key = _key(topic)
            if key and key not in reply_keys:
                reply_keys.add(key)
                spelling.setdefault(key, topic)
                counts[key] = counts.get(key, 0) + 1
    position = {key: index for index, key in enumerate(spelling)}
    ranked = sorted(spelling, key=lambda k: (-counts[k], position[k]))
    return _first([spelling[k] for k in ranked], limit)


def file_topics(
    replies: Iterable[object],
    vocabulary: Iterable[str] = (),
    *,
    per_reply_limit: int | None = None,
    file_limit: int | None = None,
) -> list[str]:
    """Build a file's topic list from its parsed replies (dicts with a list/string "topics")."""
    vocab = list(vocabulary)
    per_reply = [
        normalize_topics(reply[_TOPICS], vocab, per_reply_limit)
        for reply in replies
        if isinstance(reply, dict) and isinstance(reply.get(_TOPICS), str | list | tuple)
    ]
    return union_topics(per_reply, file_limit)


def file_summary(replies: Iterable[object]) -> str:
    """Join the replies' non-blank stripped summaries, exact duplicates removed, by newline."""
    summaries: list[str] = []
    for reply in replies:
        if not isinstance(reply, dict):
            continue
        summary = reply.get(_SUMMARY)
        if isinstance(summary, str) and summary.strip() and summary.strip() not in summaries:
            summaries.append(summary.strip())
    return "\n".join(summaries)


def derive_labels(
    existing: Iterable[str],
    existing_sources: Mapping[str, str] | None = None,
    *,
    role: str | None = None,
    domain: str | None = None,
    topics: Iterable[str] = (),
    work_unit: str | None = None,
    artifact_id: str = "",
    work_unit_prefix: str = "",
) -> DerivedLabels:
    """Re-derive an artifact's labels: user labels first, then role, domain, topics, work unit.

    Existing labels recorded with a source other than ``user`` came from an earlier run and
    are dropped (re-added only if supplied again). The first label of each key wins.
    """
    recorded = existing_sources or {}
    candidates: list[tuple[str | None, LabelSource]] = [
        (label, LabelSource.USER)
        for label in existing
        if recorded.get(label, LabelSource.USER) == LabelSource.USER
    ]
    candidates += [(role, LabelSource.ROLE), (domain, LabelSource.DOMAIN)]
    candidates += [(topic, LabelSource.TOPIC) for topic in topics]
    unit = (work_unit or "").strip()
    if unit and unit != artifact_id.strip():
        candidates.append((work_unit_prefix + unit, LabelSource.WORK_UNIT))
    sources: dict[str, LabelSource] = {}
    seen: set[str] = set()
    for candidate, source in candidates:
        label = (candidate or "").strip()
        key = _key(label) or label
        if not label or key in seen:
            continue
        seen.add(key)
        sources[label] = source
    return DerivedLabels(labels=tuple(sources), sources=sources)
