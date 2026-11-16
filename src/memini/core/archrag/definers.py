"""Resolve each canonical id of a scope to its defining entity and plan reference links.

An id may be carried by several entities. Its definer is chosen among the active carriers
by rules applied in order: a single active carrier; the carrier whose artifact holds the
most distinct ids of the id's scheme (the scheme home, learned from the scope's own ids);
the longest content; the smallest entity id. Nothing is written here: the caller writes
the planned links.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from memini.core.graph import checks
from memini.core.graph.types import Statement
from memini.core.schema import Schema

__all__ = [
    "RULE_AMBIGUOUS",
    "RULE_NONE",
    "RULE_SCHEME_HOME",
    "RULE_SINGLE",
    "RULE_SPECIFIC",
    "RULE_TIE",
    "Definer",
    "ReferenceLink",
    "choose_defining_entity",
    "load_definers",
    "plan_reference_links",
    "resolve_definers",
    "scheme_of",
]

RULE_SINGLE = "single"
RULE_SCHEME_HOME = "scheme_home"
RULE_SPECIFIC = "specific"
RULE_TIE = "tie"
RULE_NONE = "none"
RULE_AMBIGUOUS = "ambiguous"

_ACTIVE = "active"
_ENTITY = "Entity"
_CHUNK = "Chunk"
_ARTIFACT = "Artifact"
_HAS_CHUNK = "HAS_CHUNK"
_HAS_ENTITY = "HAS_ENTITY"
_ENTITY_PROPS = ("id", "canonical_id", "summary", "status", "scope")


class _Session(Protocol):
    """Anything that runs a statement and returns its rows."""

    def run(self, stmt: Statement) -> list[dict[str, Any]]:
        """Run ``stmt`` and return its rows."""
        ...


@dataclass(frozen=True)
class Definer:
    """The entity that defines one canonical id.

    Attributes:
        entity: Id of the defining entity.
        rule: The rule that chose it (single, scheme_home, specific or tie).
        candidates: Number of rows, active or not, that carry the id.

    """

    entity: str
    rule: str
    candidates: int


@dataclass(frozen=True)
class ReferenceLink:
    """A planned link from a citing entity to the entity that defines the cited id.

    Attributes:
        src: The citing entity.
        dst: The defining entity.
        cited_id: The cited id, stripped and upper-cased.
        rule: The definer's rule.
        candidates: The definer's candidate count.

    """

    src: str
    dst: str
    cited_id: str
    rule: str
    candidates: int


def _norm(cid: Any) -> str:
    """Return ``cid`` stripped and upper-cased; empty for a missing or non-text value."""
    return cid.strip().upper() if isinstance(cid, str) else ""


def scheme_of(cid: str, pattern: str) -> str | None:
    """Return the upper-cased ``prefix`` group of ``pattern`` matched on the stripped id.

    Returns None when the id does not match.
    """
    match = re.match(pattern, cid.strip())
    if match is None:
        return None
    prefix = match.group("prefix")
    return prefix.upper() if isinstance(prefix, str) else None


def _is_active(candidate: Mapping[str, Any]) -> bool:
    """Return True when the status is absent, None or active."""
    status = candidate.get("status")
    return status is None or status == _ACTIVE


def _length(candidate: Mapping[str, Any]) -> int:
    """Return the candidate's content length, 0 when missing."""
    value = candidate.get("content_len")
    return int(value) if isinstance(value, int | float) else 0


def choose_defining_entity(
    candidates: Sequence[Mapping[str, Any]],
    scheme_home: Mapping[str | None, int] | None,
) -> tuple[str | None, str]:
    """Choose the defining entity among the carriers of one id.

    Args:
        candidates: Carriers with keys "id", "artifact_id", "content_len", "status".
        scheme_home: Per artifact, the distinct ids of the id's scheme it carries, or None
            when the id has no scheme.

    Returns:
        ``(entity id, rule)``, or ``(None, RULE_NONE)`` when no carrier is active, or
        ``(None, RULE_AMBIGUOUS)`` when the id has no scheme and its active carriers sit in
        different artifacts.

    """
    pool = [c for c in candidates if _is_active(c)]
    if not pool:
        return None, RULE_NONE
    if len(pool) == 1:
        return str(pool[0]["id"]), RULE_SINGLE
    if scheme_home is None:
        if len({c.get("artifact_id") for c in pool}) > 1:
            return None, RULE_AMBIGUOUS
    else:
        top = max(scheme_home.get(c.get("artifact_id"), 0) for c in pool)
        pool = [c for c in pool if scheme_home.get(c.get("artifact_id"), 0) == top]
        if len(pool) == 1:
            return str(pool[0]["id"]), RULE_SCHEME_HOME
    longest = max(_length(c) for c in pool)
    pool = [c for c in pool if _length(c) == longest]
    if len(pool) == 1:
        return str(pool[0]["id"]), RULE_SPECIFIC
    return min(str(c["id"]) for c in pool), RULE_TIE


def resolve_definers(rows: Iterable[Mapping[str, Any]], pattern: str) -> dict[str, Definer]:
    """Resolve every canonical id carried by ``rows`` to its defining entity.

    Args:
        rows: Carriers with keys "id", "cid", "artifact_id", "content_len", "status".
        pattern: Id scheme regex with a named group "prefix".

    Returns:
        Definers keyed by the id stripped and upper-cased; ids with no definer are absent.

    """
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _norm(row.get("cid"))
        if key:
            groups[key].append(row)
    schemes = {key: scheme_of(key, pattern) for key in groups}
    homes: dict[str, dict[str | None, set[str]]] = defaultdict(lambda: defaultdict(set))
    for key, carriers in groups.items():
        scheme = schemes[key]
        if scheme is None:
            continue
        for row in carriers:
            if _is_active(row):
                homes[scheme][row.get("artifact_id")].add(key)
    result: dict[str, Definer] = {}
    for key, carriers in groups.items():
        scheme = schemes[key]
        home = (
            None
            if scheme is None
            else {artifact: len(ids) for artifact, ids in homes[scheme].items()}
        )
        entity, rule = choose_defining_entity(carriers, home)
        if entity is not None:
            result[key] = Definer(entity=entity, rule=rule, candidates=len(carriers))
    return result


def _require_declared(schema: Schema) -> None:
    """Raise ``ValueError`` when a label, relationship or property read here is undeclared."""
    for label in (_ENTITY, _CHUNK, _ARTIFACT):
        if label not in schema.nodes:
            raise ValueError(f"schema declares no label {label!r}")
    for rel in (_HAS_CHUNK, _HAS_ENTITY):
        if rel not in schema.relationships:
            raise ValueError(f"schema declares no relationship {rel!r}")
    missing = [p for p in _ENTITY_PROPS if p not in schema.nodes[_ENTITY].properties]
    if missing:
        raise ValueError(f"label {_ENTITY!r} declares no {', '.join(missing)}")


def load_definers(
    session: _Session, schema: Schema, scope: str, pattern: str
) -> dict[str, Definer]:
    """Read every carrier of a canonical id in ``scope`` with one statement and resolve them.

    An entity's artifact is the smallest id among the artifacts it was extracted from; its
    content length is the length of its summary.

    Raises:
        ValueError: When the schema does not declare what the statement reads.

    """
    _require_declared(schema)
    cypher = (
        f"MATCH (e:{_ENTITY} {{scope: $scope}}) "
        "WHERE e.canonical_id IS NOT NULL "
        f"OPTIONAL MATCH (a:{_ARTIFACT} {{scope: $scope}})-[:{_HAS_CHUNK}]->"
        f"(c:{_CHUNK})-[:{_HAS_ENTITY}]->(e:{_ENTITY}) "
        "WITH e, min(a.id) AS artifact_id "
        "RETURN e.id AS id, e.canonical_id AS cid, artifact_id, "
        "size(coalesce(e.summary, '')) AS content_len, e.status AS status"
    )
    checks.require_labeled(cypher)
    stmt = Statement(
        name="archrag:definers",
        cypher=cypher,
        params={"scope": scope},
        kind="read",
        scope=scope,
    )
    return resolve_definers(session.run(stmt), pattern)


def plan_reference_links(
    citations: Mapping[str, Sequence[str]], definers: Mapping[str, Definer]
) -> list[ReferenceLink]:
    """Plan one link per (citing entity, cited id) to the id's defining entity.

    Cited ids are stripped and upper-cased; unknown ids and self-links are dropped. The
    result is sorted by (src, cited_id).
    """
    links: dict[tuple[str, str], ReferenceLink] = {}
    for src, cited in citations.items():
        for raw in cited:
            key = _norm(raw)
            definer = definers.get(key)
            if definer is None or definer.entity == src:
                continue
            links[(src, key)] = ReferenceLink(
                src=src,
                dst=definer.entity,
                cited_id=key,
                rule=definer.rule,
                candidates=definer.candidates,
            )
    return [links[k] for k in sorted(links)]
