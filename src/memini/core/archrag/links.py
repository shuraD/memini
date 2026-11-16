"""RELATED_TO link pairs of three strategies: reading order, shared label, cross-domain.

Pure functions: no I/O, no graph, no clock, no randomness. The caller supplies the entities,
their artifacts' labels or domains, the link kind and a ``LinkSettings`` read from config.
The similarity strategies share one picking rule: each entity, in input order, takes its
qualifying candidates by score descending (ties: target id ascending), at most ``max_links``,
and an unordered pair is emitted only once, in the direction of the first entity to pick it.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

__all__ = [
    "EntityVector",
    "Link",
    "LinkSettings",
    "cross_domain_links",
    "label_links",
    "linkable_labels",
    "sequence_links",
    "settings_from_config",
]

_MIN_ARTIFACTS = 2  # a label on fewer artifacts links nothing


@dataclass(frozen=True)
class EntityVector:
    """An entity with the artifact it was extracted from and its embedding."""

    id: str
    artifact: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class Link:
    """A directed link between two entity ids; ``score`` is None for sequence links."""

    source: str
    target: str
    kind: str
    score: float | None = None


@dataclass(frozen=True)
class LinkSettings:
    """Thresholds and per-entity caps of the label and cross-domain strategies.

    Attributes:
        label_threshold: Minimum cosine for a label link.
        label_max_links: Maximum label links chosen per entity.
        label_max_share: A label counts only if carried by at most this share of artifacts.
        cross_domain_threshold: Minimum cosine for a cross-domain link.
        cross_domain_max_links: Maximum cross-domain links chosen per entity.

    """

    label_threshold: float
    label_max_links: int
    label_max_share: float
    cross_domain_threshold: float
    cross_domain_max_links: int


def _number(section: Mapping[str, object], key: str) -> int | float | str:
    """Return ``section[key]`` when convertible to a number; raise ``KeyError`` if missing."""
    value = section[key]
    if not isinstance(value, int | float | str):
        raise TypeError(f"link config {key!r} must be a number, got {value!r}")
    return value


def settings_from_config(section: Mapping[str, object]) -> LinkSettings:
    """Build ``LinkSettings`` from the five keys named like its fields; extras are ignored.

    Raises:
        KeyError: When one of the five keys is missing.
        TypeError: When a value is not a number.

    """
    return LinkSettings(
        label_threshold=float(_number(section, "label_threshold")),
        label_max_links=int(_number(section, "label_max_links")),
        label_max_share=float(_number(section, "label_max_share")),
        cross_domain_threshold=float(_number(section, "cross_domain_threshold")),
        cross_domain_max_links=int(_number(section, "cross_domain_max_links")),
    )


def sequence_links(entity_ids: Sequence[str], kind: str) -> list[Link]:
    """Link each id to the next one in reading order, skipping equal consecutive ids."""
    return [Link(a, b, kind) for a, b in pairwise(entity_ids) if a != b]


def linkable_labels(artifact_labels: Mapping[str, Collection[str]], max_share: float) -> set[str]:
    """Labels carried by at least 2 artifacts and by at most ``max_share`` of all artifacts."""
    counts = Counter(label for labels in artifact_labels.values() for label in set(labels))
    limit = max_share * len(artifact_labels)
    return {label for label, n in counts.items() if _MIN_ARTIFACTS <= n <= limit}


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors; 0.0 when either has zero norm."""
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return math.fsum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


def _pick(
    entities: Sequence[EntityVector],
    eligible: Callable[[EntityVector, EntityVector], bool],
    threshold: float,
    max_links: int,
    kind: str,
) -> list[Link]:
    """Apply the shared picking rule to the pairs ``eligible`` accepts (see module doc)."""
    emitted: set[frozenset[str]] = set()
    links: list[Link] = []
    for entity in entities:
        scored = [
            (other.id, _cosine(entity.vector, other.vector))
            for other in entities
            if other.id != entity.id
            and other.artifact != entity.artifact
            and eligible(entity, other)
        ]
        kept = sorted((c for c in scored if c[1] >= threshold), key=lambda c: (-c[1], c[0]))
        for target, score in kept[: max(max_links, 0)]:
            pair = frozenset((entity.id, target))
            if pair not in emitted:
                emitted.add(pair)
                links.append(Link(entity.id, target, kind, score))
    return links


def label_links(
    entities: Sequence[EntityVector],
    artifact_labels: Mapping[str, Collection[str]],
    settings: LinkSettings,
    kind: str,
) -> list[Link]:
    """Link entities of different artifacts that share a linkable label (see module doc)."""
    linkable = linkable_labels(artifact_labels, settings.label_max_share)
    kept = {art: linkable.intersection(labels) for art, labels in artifact_labels.items()}

    def eligible(a: EntityVector, b: EntityVector) -> bool:
        return bool(kept.get(a.artifact, set()) & kept.get(b.artifact, set()))

    return _pick(entities, eligible, settings.label_threshold, settings.label_max_links, kind)


def cross_domain_links(
    entities: Sequence[EntityVector],
    artifact_domain: Mapping[str, str],
    settings: LinkSettings,
    kind: str,
) -> list[Link]:
    """Link entities whose artifacts have different known domains (see module doc)."""

    def eligible(a: EntityVector, b: EntityVector) -> bool:
        da = artifact_domain.get(a.artifact)
        db = artifact_domain.get(b.artifact)
        return da is not None and db is not None and da != db

    return _pick(
        entities,
        eligible,
        settings.cross_domain_threshold,
        settings.cross_domain_max_links,
        kind,
    )
