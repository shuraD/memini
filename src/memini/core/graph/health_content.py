"""Content checks and degree statistics for graph health of one scope.

The checks plug into GS-18's ``graph_health`` as extra checks: orphans, missing or
wrong-size embeddings, missing or offline indexes, broken ``SUPERSEDES`` chains, items
waiting for purge and stale communities. Every statement is a read built from the
schema, bound to ``$scope`` and returning a ``count`` column. ``degree_stats`` builds
one statement whose rows (one per relationship type) fill ``HealthReport.stats``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import Any

from memini.core.graph.health import Check, HealthReport, graph_health
from memini.core.graph.session import GraphSession
from memini.core.graph.types import Statement
from memini.core.schema import Schema, rel_properties, setup_statements

CONTENT_SEVERITY: Mapping[str, str] = MappingProxyType(
    {
        "orphan": "major",
        "missing_embedding": "major",
        "wrong_vector_size": "critical",
        "missing_index": "major",
        "offline_index": "critical",
        "broken_supersedes": "major",
        "awaiting_purge": "info",
        "stale_community": "minor",
    }
)

# Lifecycle names fixed by the graph-schema specification.
_ACTIVE = "active"
_SUPERSEDED = "superseded"
_DELETED = "deleted"
_ONLINE = "ONLINE"
_SUPERSEDES = "SUPERSEDES"
_COMMUNITY = "Community"
_MEMBER_OF = "MEMBER_OF"
_STATUS = "status"
_INDEX_NAME_RE = r"\b(?:INDEX|CONSTRAINT)\s+(\w+)\s+IF\s+NOT\s+EXISTS\b"
_DEGREE_FIELDS = ("avg", "p50", "max")

Triples = list[tuple[Statement, str, str]]


def _stmt(kind: str, cypher: str, scope: str, **params: Any) -> Statement:
    """Build a read statement named after ``kind`` with ``scope`` bound as ``$scope``."""
    return Statement(
        name=f"graph_health.{kind}",
        cypher=cypher,
        params={"scope": scope, **params},
        scope=scope,
        kind="read",
    )


def _triple(kind: str, cypher: str, scope: str, **params: Any) -> tuple[Statement, str, str]:
    """Return ``(statement, kind, severity)`` with the severity of ``kind``'s family."""
    severity = CONTENT_SEVERITY[kind.split(":", maxsplit=1)[0]]
    return _stmt(kind, cypher, scope, **params), kind, severity


def _orphan_check(rules: Sequence[tuple[str, str]]) -> Check:
    """Return a check counting active nodes of each rule's label with no incoming type."""

    def check(schema: Schema, scope: str) -> Triples:
        for label, rel in rules:
            if label not in schema.nodes or rel not in schema.relationships:
                raise ValueError(f"orphan rule ({label!r}, {rel!r}) is not declared by the schema")
        return [
            _triple(
                f"orphan:{label}",
                f"MATCH (n:{label}) WHERE n.scope = $scope AND n.status = '{_ACTIVE}' "
                f"AND NOT ()-[:{rel}]->(n) RETURN count(n) AS count",
                scope,
            )
            for label, rel in rules
        ]

    return check


def _vector_check(vector_dimensions: int | None) -> Check:
    """Return a check for null embeddings and, given dimensions, wrong-size ones."""

    def check(schema: Schema, scope: str) -> Triples:
        out: Triples = []
        pairs = [(lb, p) for lb, node in schema.nodes.items() for p in node.vector_indexes]
        for label, prop in pairs:
            out.append(
                _triple(
                    "missing_embedding",
                    f"MATCH (n:{label}) WHERE n.scope = $scope AND n.status = '{_ACTIVE}' "
                    f"AND n.{prop} IS NULL RETURN count(n) AS count",
                    scope,
                )
            )
        if vector_dimensions is not None:
            for label, prop in pairs:
                out.append(
                    _triple(
                        "wrong_vector_size",
                        f"MATCH (n:{label}) WHERE n.scope = $scope AND n.{prop} IS NOT NULL "
                        f"AND size(n.{prop}) <> $dims RETURN count(n) AS count",
                        scope,
                        dims=vector_dimensions,
                    )
                )
        return out

    return check


def _index_check(schema: Schema, scope: str) -> Triples:
    """Count declared index names that are absent, then those present but not online."""
    names = list(dict.fromkeys(re.findall(_INDEX_NAME_RE, "\n".join(setup_statements(schema)))))
    if not names:
        return []
    return [
        _triple(
            "missing_index",
            "SHOW INDEXES YIELD name WITH collect(name) AS present "
            "RETURN size([x IN $names WHERE NOT x IN present]) AS count",
            scope,
            names=names,
        ),
        _triple(
            "offline_index",
            f"SHOW INDEXES YIELD name, state WHERE name IN $names AND state <> '{_ONLINE}' "
            "RETURN count(name) AS count",
            scope,
            names=names,
        ),
    ]


def _supersedes_check(schema: Schema, scope: str) -> Triples:
    """Count superseded nodes nobody supersedes and active supersedes of active targets."""
    rel = schema.relationships.get(_SUPERSEDES)
    if rel is None:
        return []
    out: Triples = []
    for target in rel.targets:
        out.append(
            _triple(
                "broken_supersedes",
                f"MATCH (n:{target}) WHERE n.scope = $scope AND n.status = '{_SUPERSEDED}' "
                f"AND NOT ()-[:{_SUPERSEDES}]->(n) RETURN count(n) AS count",
                scope,
            )
        )
    has_status = _STATUS in rel_properties(schema, _SUPERSEDES)
    rel_active = f" AND r.status = '{_ACTIVE}'" if has_status else ""
    for source in rel.sources:
        for target in rel.targets:
            out.append(
                _triple(
                    "broken_supersedes",
                    f"MATCH (a:{source})-[r:{_SUPERSEDES}]->(b:{target}) "
                    f"WHERE a.scope = $scope{rel_active} AND b.status = '{_ACTIVE}' "
                    "RETURN count(r) AS count",
                    scope,
                )
            )
    return out


def _purge_check(schema: Schema, scope: str) -> Triples:
    """Count nodes of every declared label whose status marks them for purge."""
    return [
        _triple(
            "awaiting_purge",
            f"MATCH (n:{label}) WHERE n.scope = $scope AND n.status = '{_DELETED}' "
            "RETURN count(n) AS count",
            scope,
        )
        for label in schema.nodes
    ]


def _community_check(schema: Schema, scope: str) -> Triples:
    """Count communities updated before the latest update of one of their members."""
    rel = schema.relationships.get(_MEMBER_OF)
    if _COMMUNITY not in schema.nodes or rel is None or _COMMUNITY not in rel.targets:
        return []
    parts = [f"MATCH (c:{_COMMUNITY}) WHERE c.scope = $scope"]
    latest: list[str] = []
    for i, source in enumerate(rel.sources):
        parts.append(
            f"OPTIONAL MATCH (e{i}:{source})-[:{_MEMBER_OF}]->(c:{_COMMUNITY}) "
            f"WITH c{''.join(f', {m}' for m in latest)}, max(e{i}.updated_at) AS m{i}"
        )
        latest.append(f"m{i}")
    parts.append(
        f"WITH c, [m IN [{', '.join(latest)}] WHERE m IS NOT NULL] AS ms "
        "WHERE any(m IN ms WHERE c.updated_at < m) RETURN count(c) AS count"
    )
    return [_triple("stale_community", " ".join(parts), scope)]


def content_checks(
    *, orphan_rules: Sequence[tuple[str, str]], vector_dimensions: int | None
) -> list[Check]:
    """Return the content checks to pass as ``graph_health``'s ``extra_checks``.

    Orphan rules are ``(label, relationship type)`` pairs; a rule naming anything the
    schema does not declare raises ``ValueError`` when its check is called.
    """
    rules = tuple(orphan_rules)
    return [
        _orphan_check(rules),
        _vector_check(vector_dimensions),
        _index_check,
        _supersedes_check,
        _purge_check,
        _community_check,
    ]


def degree_stats(schema: Schema, scope: str) -> Statement:
    """Return the statement giving per relationship type the ``avg``/``p50``/``max`` out-degree.

    Degrees are counted on source nodes in ``scope``; types with no declared endpoint pair
    are left out. Rows: ``{'type', 'avg', 'p50', 'max'}``.
    """
    branches: list[str] = []
    for name, rel in schema.relationships.items():
        inner = " UNION ALL ".join(
            f"MATCH (a:{source})-[r:{name}]->(:{target}) WHERE a.scope = $scope "
            "RETURN a AS n, count(r) AS c"
            for source in rel.sources
            for target in rel.targets
        )
        if inner:
            branches.append(
                f"CALL {{ {inner} }} WITH n, sum(c) AS d "
                f"RETURN '{name}' AS `type`, avg(d) AS `avg`, "
                "percentileCont(d, 0.5) AS `p50`, max(d) AS `max`"
            )
    cypher = " UNION ALL ".join(branches) or (
        "WITH $scope AS s UNWIND [] AS t "
        "RETURN t AS `type`, 0.0 AS `avg`, 0.0 AS `p50`, 0.0 AS `max`"
    )
    return Statement(
        name="graph_health.degree", cypher=cypher, params={"scope": scope}, scope=scope, kind="read"
    )


def content_health(
    session: GraphSession,
    schema: Schema,
    scope: str,
    *,
    orphan_rules: Sequence[tuple[str, str]],
    vector_dimensions: int | None,
) -> HealthReport:
    """Run ``graph_health`` with the content checks, then fill ``stats`` from degree rows.

    Problems and score are exactly ``graph_health``'s; ``stats`` holds
    ``degree.<type>.avg|p50|max`` as floats.

    Raises:
        AccessDenied: When ``scope`` is not readable by the session's context.
        ValueError: When an orphan rule names an undeclared label or relationship type.

    """
    checks = content_checks(orphan_rules=orphan_rules, vector_dimensions=vector_dimensions)
    report = graph_health(session, schema, scope, extra_checks=checks)
    stats: dict[str, float] = {}
    for row in session.run(degree_stats(schema, scope)):
        for key in _DEGREE_FIELDS:
            stats[f"degree.{row['type']}.{key}"] = float(row[key])
    return replace(report, stats=stats)
