"""Batched node and relationship writes that MERGE on ``(scope, id)``.

The builder checks the caller's write access, the schema and the row contents before
building anything, then returns one ``UNWIND $rows`` statement per batch. It sets the
scope and the audit fields itself; every value travels as a parameter, so only schema
labels, relationship types and fixed keywords appear in the query text. Nothing is
executed here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from memini.core.graph import checks
from memini.core.graph.access import require_context, require_same_scope, require_write
from memini.core.graph.types import AccessContext, Clock, IsolationError, Statement
from memini.core.schema import Schema, node_properties, rel_properties

__all__ = ["RESERVED_NODE_FIELDS", "RESERVED_REL_FIELDS", "WriteBuilder"]

RESERVED_NODE_FIELDS = frozenset(
    {"scope", "id", "status", "created_at", "updated_at", "updated_by", "deleted_at", "deleted_by"}
)
"""Node properties the builder (or a later lifecycle step) sets; callers may not."""

RESERVED_REL_FIELDS = frozenset({"created_at"})
"""Relationship properties the builder sets; callers may not."""


def _require_str(row: Mapping[str, Any], key: str) -> str:
    """Return ``row[key]`` when it is a non-empty string; raise ``ValueError`` otherwise."""
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"row needs a non-empty string {key!r}: {row!r}")
    return value


def _check_props(
    props: Mapping[str, Any], reserved: frozenset[str], declared: Mapping[str, Any], owner: str
) -> dict[str, Any]:
    """Return ``props`` as a dict, refusing reserved or undeclared names."""
    for name in props:
        if name in reserved:
            raise ValueError(f"property {name!r} of {owner} is set by the write builder")
        if name not in declared:
            raise ValueError(f"property {name!r} is not declared for {owner}")
    return dict(props)


class WriteBuilder:
    """Builds batched, audited, scope-checked write statements for one caller."""

    def __init__(
        self, schema: Schema, ctx: AccessContext | None, clock: Clock, *, batch_size: int = 1000
    ) -> None:
        """Bind the schema, access context and clock.

        Args:
            schema: Schema declaring the labels, relationship types and properties.
            ctx: The caller's access context.
            clock: Source of the write time.
            batch_size: Maximum number of rows per statement.

        Raises:
            AccessDenied: When ``ctx`` is ``None``.
            ValueError: When ``batch_size`` is below 1.

        """
        self._ctx = require_context(ctx)
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self._schema = schema
        self._clock = clock
        self._batch_size = batch_size

    def audit_set(self, var: str) -> str:
        """Return the assignments recording the write time and author on ``var``."""
        return f"{var}.updated_at = datetime($now), {var}.updated_by = $actor"

    def _batches(
        self, name: str, scope: str, cypher: str, rows: list[dict[str, Any]]
    ) -> list[Statement]:
        """Split ``rows`` into statements of at most ``batch_size`` rows, order kept."""
        if not rows:
            return []
        checks.require_labeled(cypher)
        checks.require_audited(cypher)
        now = self._clock().isoformat()
        size = self._batch_size
        return [
            Statement(
                name=name,
                cypher=cypher,
                params={
                    "scope": scope,
                    "rows": rows[start : start + size],
                    "now": now,
                    "actor": self._ctx.actor,
                },
                scope=scope,
                kind="write",
            )
            for start in range(0, len(rows), size)
        ]

    def merge_nodes(
        self, label: str, scope: str, rows: Sequence[Mapping[str, Any]]
    ) -> list[Statement]:
        """Build statements that merge nodes of ``label`` in ``scope`` by id.

        Args:
            label: Schema node label.
            scope: Target scope; must be writable by the caller.
            rows: Mappings with ``id`` (non-empty str) and optional ``props``.

        Returns:
            One statement per batch of rows; ``[]`` when ``rows`` is empty.

        Raises:
            AccessDenied: When ``scope`` is not writable.
            ValueError: For an unknown label, a row without a valid id, or a reserved
                or undeclared property.

        """
        require_write(self._ctx, scope)
        if label not in self._schema.nodes:
            raise ValueError(f"unknown node label {label!r}")
        declared = node_properties(self._schema, label)
        built = [
            {
                "id": _require_str(row, "id"),
                "props": _check_props(
                    row.get("props") or {}, RESERVED_NODE_FIELDS, declared, label
                ),
            }
            for row in rows
        ]
        cypher = (
            f"UNWIND $rows AS row MERGE (n:{label} {{scope: $scope, id: row.id}}) "
            "ON CREATE SET n.created_at = datetime($now), n.status = 'active' "
            f"SET n += row.props, {self.audit_set('n')}"
        )
        return self._batches(f"write:{label}", scope, cypher, built)

    def merge_relationships(
        self,
        rel: str,
        scope: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        from_label: str,
        to_label: str,
    ) -> list[Statement]:
        """Build statements that merge ``rel`` relationships between nodes of ``scope``.

        Args:
            rel: Schema relationship type.
            scope: Target scope; must be writable by the caller.
            rows: Mappings with ``from_id``, ``to_id``, optional ``props`` and optional
                ``from_scope`` / ``to_scope`` (both default to ``scope``).
            from_label: Label of the start nodes; a declared source of ``rel``.
            to_label: Label of the end nodes; a declared target of ``rel``.

        Returns:
            One statement per batch of rows; ``[]`` when ``rows`` is empty.

        Raises:
            AccessDenied: When ``scope`` is not writable.
            ValueError: For an unknown type, an undeclared endpoint label, a row without
                valid ids, or a reserved or undeclared property.
            IsolationError: When a row's endpoint scopes differ from each other or
                from ``scope``.

        """
        require_write(self._ctx, scope)
        spec = self._schema.relationships.get(rel)
        if spec is None:
            raise ValueError(f"unknown relationship type {rel!r}")
        if from_label not in spec.sources:
            raise ValueError(f"{from_label!r} is not a declared source of {rel!r}")
        if to_label not in spec.targets:
            raise ValueError(f"{to_label!r} is not a declared target of {rel!r}")
        for row in rows:
            from_scope = row.get("from_scope", scope)
            to_scope = row.get("to_scope", scope)
            require_same_scope(from_scope, to_scope)
            if from_scope != scope:
                raise IsolationError(f"relationship in scope {from_scope!r}, not {scope!r}")
        declared = rel_properties(self._schema, rel)
        built = [
            {
                "from_id": _require_str(row, "from_id"),
                "to_id": _require_str(row, "to_id"),
                "props": _check_props(row.get("props") or {}, RESERVED_REL_FIELDS, declared, rel),
            }
            for row in rows
        ]
        cypher = (
            "UNWIND $rows AS row "
            f"MATCH (a:{from_label} {{scope: $scope, id: row.from_id}}) "
            f"MATCH (b:{to_label} {{scope: $scope, id: row.to_id}}) "
            f"MERGE (a)-[r:{rel}]->(b) ON CREATE SET r.created_at = datetime($now) "
            "SET r += row.props"
        )
        return self._batches(f"write:{rel}", scope, cypher, built)
