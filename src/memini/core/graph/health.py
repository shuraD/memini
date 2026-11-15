"""Graph health of one scope: a 0-100 score and the problems found by read-only checks.

Every check is a scope-bound, labeled ``MATCH`` statement built from the schema; the scope
value is always bound as the ``$scope`` parameter. Nothing is written. Built-in checks look
for duplicated facts, undeclared or never-written properties, missing audit fields and
relationships crossing scopes; callers may add their own checks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from memini.core.graph.session import GraphSession
from memini.core.graph.types import AccessDenied, Statement
from memini.core.schema import Schema, duplicate_queries, node_properties

SEVERITIES: tuple[str, ...] = ("critical", "major", "minor", "info")
DEFAULT_WEIGHTS: Mapping[str, int] = MappingProxyType(
    {"critical": 30, "major": 10, "minor": 2, "info": 0}
)
KINDS: tuple[str, ...] = (
    "duplicates",
    "undeclared_property",
    "declared_never_written",
    "missing_audit_fields",
    "cross_scope_relationship",
)
BUILTIN_SEVERITY: Mapping[str, str] = MappingProxyType(
    {
        "duplicates": "critical",
        "undeclared_property": "major",
        "declared_never_written": "minor",
        "missing_audit_fields": "major",
        "cross_scope_relationship": "critical",
    }
)

_NODE_AUDIT_FIELDS = ("created_at", "updated_at", "updated_by")
_REL_AUDIT_FIELDS = ("created_at",)
_DUPLICATES_COLUMN = "duplicates"
_COUNT_COLUMN = "count"


@dataclass(frozen=True)
class Problem:
    """One kind of problem found, with its severity, total count and a short description."""

    kind: str
    severity: str
    count: int
    detail: str

    def __post_init__(self) -> None:
        """Raise ``ValueError`` when ``severity`` is not one of ``SEVERITIES``."""
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}; expected one of {SEVERITIES}")


@dataclass(frozen=True)
class HealthReport:
    """Health of one scope: its score, problems and statistics (empty in this slice)."""

    scope: str
    score: int
    problems: tuple[Problem, ...]
    stats: Mapping[str, float] = field(default_factory=dict)


Check = Callable[[Schema, str], list[tuple[Statement, str, str]]]
"""An extra check: returns ``(statement, kind, severity)`` triples; rows carry ``count``."""


def score(problems: Sequence[Problem], weights: Mapping[str, int] = DEFAULT_WEIGHTS) -> int:
    """Return 100 minus the weight of each problem with a positive count, floored at 0."""
    lost = sum(weights[p.severity] for p in problems if p.count > 0)
    return max(0, 100 - lost)


def _stmt(kind: str, cypher: str, scope: str, **params: Any) -> Statement:
    """Build a read statement named after ``kind`` with ``scope`` bound as ``$scope``."""
    return Statement(
        name=f"graph_health.{kind}",
        cypher=cypher,
        params={"scope": scope, **params},
        scope=scope,
        kind="read",
    )


def _pairs(schema: Schema) -> Iterator[tuple[str, str, str]]:
    """Yield ``(source, type, target)`` for every declared relationship endpoint pair."""
    for name, rel in schema.relationships.items():
        for source in rel.sources:
            for target in rel.targets:
                yield source, name, target


def _builtin(schema: Schema, scope: str) -> list[tuple[Statement, str, str]]:
    """Return the built-in check statements in ``KINDS`` order."""
    out: list[tuple[Statement, str, str]] = []

    def add(kind: str, cypher: str, **params: Any) -> None:
        out.append((_stmt(kind, cypher, scope, **params), kind, BUILTIN_SEVERITY[kind]))

    for query in duplicate_queries(schema, scope):
        add("duplicates", query)
    labels = {label: sorted(node_properties(schema, label)) for label in schema.nodes}
    for label, declared in labels.items():
        add(
            "undeclared_property",
            f"MATCH (n:{label}) WHERE n.scope = $scope UNWIND keys(n) AS key "
            "WITH DISTINCT key WHERE NOT key IN $declared RETURN count(key) AS count",
            declared=declared,
        )
    for label, declared in labels.items():
        add(
            "declared_never_written",
            f"MATCH (n:{label}) WHERE n.scope = $scope UNWIND keys(n) AS key "
            "WITH collect(DISTINCT key) AS written "
            "RETURN size([p IN $declared WHERE NOT p IN written]) AS count",
            declared=declared,
        )
    for label in labels:
        missing = " OR ".join(f"n.{f} IS NULL" for f in _NODE_AUDIT_FIELDS)
        add(
            "missing_audit_fields",
            f"MATCH (n:{label}) WHERE n.scope = $scope AND ({missing}) RETURN count(n) AS count",
        )
    for source, name, target in _pairs(schema):
        missing = " OR ".join(f"r.{f} IS NULL" for f in _REL_AUDIT_FIELDS)
        add(
            "missing_audit_fields",
            f"MATCH (a:{source})-[r:{name}]->(b:{target}) WHERE a.scope = $scope "
            f"AND ({missing}) RETURN count(r) AS count",
        )
    for source, name, target in _pairs(schema):
        add(
            "cross_scope_relationship",
            f"MATCH (a:{source})-[r:{name}]->(b:{target}) "
            "WHERE (a.scope = $scope AND b.scope <> $scope) "
            "OR (b.scope = $scope AND a.scope <> $scope) RETURN count(r) AS count",
        )
    return out


def graph_health(
    session: GraphSession,
    schema: Schema,
    scope: str,
    *,
    extra_checks: Sequence[Check] = (),
) -> HealthReport:
    """Run every health check of ``scope`` read-only and return its report.

    Only ``session.ctx`` and ``session.run`` are used. Every statement is built and
    validated before the first one runs.

    Raises:
        AccessDenied: When ``scope`` is not readable by the session's context.
        ValueError: When a statement is not a read or a severity is unknown.

    """
    if scope not in session.ctx.read_scopes:
        raise AccessDenied(f"scope {scope!r} is not readable")
    checks = _builtin(schema, scope)
    for check in extra_checks:
        checks.extend(check(schema, scope))
    for stmt, kind, severity in checks:
        if stmt.kind != "read":
            raise ValueError(f"health check {kind!r} is not a read: {stmt.name!r}")
        if severity not in SEVERITIES:
            raise ValueError(f"health check {kind!r} has unknown severity {severity!r}")
    totals: dict[str, int] = {}
    severities: dict[str, str] = {}
    runs: dict[str, int] = {}
    for stmt, kind, severity in checks:
        column = _DUPLICATES_COLUMN if kind == "duplicates" else _COUNT_COLUMN
        rows = session.run(stmt)
        totals[kind] = totals.get(kind, 0) + sum(int(row.get(column) or 0) for row in rows)
        severities.setdefault(kind, severity)
        runs[kind] = runs.get(kind, 0) + 1
    problems = tuple(
        Problem(kind, severities[kind], total, f"{total} found by {runs[kind]} check(s)")
        for kind, total in totals.items()
        if total > 0
    )
    return HealthReport(scope, score(problems), problems, {})
