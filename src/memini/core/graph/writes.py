"""Batched node and relationship writes, soft delete, restore and hard delete, in one scope.

The builder is bound to a scope and a clock. It checks the schema and the row contents before
building anything, then returns one statement per batch. Node writes MERGE on
``(scope, id)``, or on ``(scope, name)`` for a label without ``id``, and set ``updated_by``
(from the row) and ``updated_at`` (from the clock). Relationship endpoints are matched inside
the scope. Allowed values, required properties, allowed pairs and endpoint rules are read
from the schema; this module holds no list of them. Every value travels as a parameter, so
only schema labels, relationship types, property names, parameter names and fixed keywords
appear in the query text. Nothing is executed here; a hard-delete statement has kind
``delete``, which the session checks against the database guard before it runs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from memini.core.graph import checks
from memini.core.graph.types import Statement
from memini.core.schema import Node, Property, Schema, node_key

__all__ = ["RESERVED_NODE_FIELDS", "WriteBuilder"]

RESERVED_NODE_FIELDS = frozenset({"id", "scope", "updated_by", "updated_at"})
"""Node properties the builder sets itself; callers may not pass them in ``props``."""

_STATUS = "status"
_DELETED = "deleted"
_ACTIVE = "active"


def _require_str(row: Mapping[str, Any], key: str) -> str:
    """Return ``row[key]`` when it is a non-empty string; raise ``ValueError`` otherwise."""
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"row needs a non-empty string {key!r}: {row!r}")
    return value


def _check_values(owner: str, declared: Mapping[str, Property], props: Mapping[str, Any]) -> None:
    """Refuse a value outside a property's ``values`` and a missing required property."""
    for name, prop in declared.items():
        if name in props:
            if prop.values and props[name] not in prop.values:
                raise ValueError(
                    f"{owner}.{name} = {props[name]!r} is not one of {list(prop.values)}"
                )
            continue
        if prop.required:
            raise ValueError(f"{owner}.{name} is required")
        condition = prop.required_when
        if condition and all(props.get(k) == v for k, v in condition.items()):
            raise ValueError(f"{owner}.{name} is required when {dict(condition)}")


def _check_props(
    props: Mapping[str, Any], reserved: frozenset[str], declared: Mapping[str, Property], owner: str
) -> dict[str, Any]:
    """Return ``props`` as a dict, refusing reserved, undeclared or disallowed values."""
    for name in props:
        if name in reserved:
            raise ValueError(f"property {name!r} of {owner} is set by the write builder")
        if name not in declared:
            raise ValueError(f"property {name!r} is not declared for {owner}")
    _check_values(owner, declared, props)
    return dict(props)


def _ids(ids: Sequence[str]) -> list[str]:
    """Return ``ids`` as a list, refusing an empty or non-string id."""
    for value in ids:
        if not isinstance(value, str) or not value:
            raise ValueError(f"ids must be non-empty strings: {value!r}")
    return list(ids)


def _require_hard(hard: bool) -> None:
    """Refuse a hard delete unless the caller passed ``hard=True``."""
    if hard is not True:
        raise ValueError("a hard delete removes nodes for good; pass hard=True")


class WriteBuilder:
    """Builds batched, schema-checked write and delete statements for one scope."""

    def __init__(
        self,
        schema: Schema,
        *,
        scope: str,
        clock: Callable[[], Any],
        batch_size: int = 1000,
    ) -> None:
        """Bind the schema, the scope and the clock.

        Args:
            schema: Schema declaring the labels, relationship types and properties.
            scope: The scope every statement works in.
            clock: Returns the ``updated_at`` value of a write.
            batch_size: Maximum number of rows per statement.

        Raises:
            ValueError: When ``scope`` is empty or ``batch_size`` is below 1.

        """
        if not isinstance(scope, str) or not scope:
            raise ValueError(f"scope must be a non-empty string, got {scope!r}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self._schema = schema
        self._scope = scope
        self._clock = clock
        self._batch_size = batch_size

    def _node(self, label: str) -> tuple[Node, str]:
        """Return the node spec of ``label`` and its key property."""
        node = self._schema.nodes.get(label)
        if node is None:
            raise ValueError(f"unknown node label {label!r}")
        key = node_key(node)
        if key is None:
            raise ValueError(f"label {label!r} declares neither id nor name")
        return node, key

    def _batches(
        self,
        name: str,
        cypher: str,
        items: list[Any],
        *,
        param: str = "rows",
        extra: Mapping[str, Any] | None = None,
        kind: Literal["write", "delete"] = "write",
    ) -> list[Statement]:
        """Split ``items`` into statements of at most ``batch_size`` items, order kept."""
        if not items:
            return []
        checks.require_labeled(cypher)
        size = self._batch_size
        return [
            Statement(
                name=name,
                cypher=cypher,
                params={param: items[start : start + size], "scope": self._scope, **(extra or {})},
                kind=kind,
                scope=self._scope,
            )
            for start in range(0, len(items), size)
        ]

    def merge_nodes(self, label: str, rows: Sequence[Mapping[str, Any]]) -> list[Statement]:
        """Build statements that merge nodes of ``label`` by (scope, key).

        Args:
            label: Schema node label.
            rows: Mappings with the key (``id``, or ``name`` for a label without id), a
                non-empty ``updated_by`` and optional ``props``.

        Returns:
            One statement per batch of rows; ``[]`` when ``rows`` is empty.

        Raises:
            ValueError: For an unknown label, a row without a valid key or ``updated_by``,
                a reserved or undeclared property, a value outside a property's values, or a
                missing required property.

        """
        node, key = self._node(label)
        reserved = RESERVED_NODE_FIELDS | {key}
        built = [
            {
                key: _require_str(row, key),
                "updated_by": _require_str(row, "updated_by"),
                "props": _check_props(row.get("props") or {}, reserved, node.properties, label),
            }
            for row in rows
        ]
        if not built:
            return []
        cypher = (
            f"UNWIND $rows AS row MERGE (n:{label} {{scope: $scope, {key}: row.{key}}}) "
            "SET n += row.props, n.updated_by = row.updated_by, n.updated_at = $updated_at"
        )
        return self._batches(f"write:{label}", cypher, built, extra={"updated_at": self._clock()})

    def merge_relationships(
        self,
        rel: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        from_label: str,
        to_label: str,
    ) -> list[Statement]:
        """Build statements that merge ``rel`` relationships between nodes of the scope.

        When ``rel`` declares ``pairs``, only those (from, to) labels are accepted. When ``rel``
        declares an ``endpoint_offset`` whose property both labels declare, only
        pairs with ``b.<property> = a.<property> + offset`` are linked, and the statement
        returns the count written, so a dropped row shows.

        Args:
            rel: Schema relationship type.
            rows: Mappings with ``from_id``, ``to_id`` (the endpoint keys) and optional
                ``props``.
            from_label: Label of the start nodes; a declared source of ``rel``.
            to_label: Label of the end nodes; a declared target of ``rel``.

        Returns:
            One statement per batch of rows; ``[]`` when ``rows`` is empty.

        Raises:
            ValueError: For an unknown type, an undeclared endpoint label or pair, a row
                without valid ids, an undeclared property, a value outside a property's values, or a
                missing required property.

        """
        spec = self._schema.relationships.get(rel)
        if spec is None:
            raise ValueError(f"unknown relationship type {rel!r}")
        if from_label not in spec.sources:
            raise ValueError(f"{from_label!r} is not a declared source of {rel!r}")
        if to_label not in spec.targets:
            raise ValueError(f"{to_label!r} is not a declared target of {rel!r}")
        if spec.pairs and (from_label, to_label) not in spec.pairs:
            raise ValueError(f"{rel!r} does not link {from_label!r} to {to_label!r}")
        from_node, from_key = self._node(from_label)
        to_node, to_key = self._node(to_label)
        built = [
            {
                "from_id": _require_str(row, "from_id"),
                "to_id": _require_str(row, "to_id"),
                "props": _check_props(row.get("props") or {}, frozenset(), spec.properties, rel),
            }
            for row in rows
        ]
        cypher = (
            "UNWIND $rows AS row "
            f"MATCH (a:{from_label} {{scope: $scope, {from_key}: row.from_id}}) "
            f"MATCH (b:{to_label} {{scope: $scope, {to_key}: row.to_id}}) "
        )
        extra: dict[str, Any] = {}
        rule = spec.endpoint_offset
        guarded = (
            rule is not None
            and rule.property in from_node.properties
            and rule.property in to_node.properties
        )
        if rule is not None and guarded:
            cypher += f"WHERE b.{rule.property} = a.{rule.property} + $endpoint_offset "
            extra["endpoint_offset"] = rule.offset
        cypher += f"MERGE (a)-[r:{rel}]->(b) SET r += row.props"
        if guarded:
            cypher += " RETURN count(r) AS written"
        return self._batches(f"write:{rel}", cypher, built, extra=extra)

    def _set_status(
        self, label: str, ids: Sequence[str], status: str, updated_by: str, name: str
    ) -> list[Statement]:
        """Build statements that set ``status`` on the label's nodes with those keys."""
        node, key = self._node(label)
        prop = node.properties.get(_STATUS)
        if prop is None:
            raise ValueError(f"label {label!r} declares no {_STATUS!r}")
        if prop.values and status not in prop.values:
            raise ValueError(f"{label}.{_STATUS} does not allow {status!r}")
        writer = _require_str({"updated_by": updated_by}, "updated_by")
        cypher = (
            f"MATCH (n:{label} {{scope: $scope}}) WHERE n.{key} IN $ids "
            f"SET n.{_STATUS} = $status, n.updated_by = $updated_by, n.updated_at = $updated_at"
        )
        extra = {"status": status, "updated_by": writer, "updated_at": self._clock()}
        return self._batches(f"{name}:{label}", cypher, _ids(ids), param="ids", extra=extra)

    def soft_delete(self, label: str, ids: Sequence[str], *, updated_by: str) -> list[Statement]:
        """Build statements that mark the label's nodes with those keys as deleted.

        Nothing is removed. Raises ``ValueError`` when the label declares no status.
        """
        return self._set_status(label, ids, _DELETED, updated_by, "soft_delete")

    def restore(self, label: str, ids: Sequence[str], *, updated_by: str) -> list[Statement]:
        """Build statements that mark the label's nodes with those keys as active again.

        Raises ``ValueError`` when the label declares no status.
        """
        return self._set_status(label, ids, _ACTIVE, updated_by, "restore")

    def hard_delete(self, label: str, ids: Sequence[str], *, hard: bool = False) -> list[Statement]:
        """Build ``DETACH DELETE`` statements for the label's nodes with those keys.

        Raises ``ValueError``, and builds nothing, unless ``hard`` is True.
        """
        _require_hard(hard)
        _, key = self._node(label)
        cypher = f"MATCH (n:{label} {{scope: $scope}}) WHERE n.{key} IN $ids DETACH DELETE n"
        return self._batches(f"hard_delete:{label}", cypher, _ids(ids), param="ids", kind="delete")

    def hard_delete_scope(self, *, hard: bool = False) -> list[Statement]:
        """Build one labeled ``DETACH DELETE`` statement per schema label for the scope.

        Raises ``ValueError``, and builds nothing, unless ``hard`` is True.
        """
        _require_hard(hard)
        out: list[Statement] = []
        for label in self._schema.nodes:
            cypher = f"MATCH (n:{label} {{scope: $scope}}) DETACH DELETE n"
            checks.require_labeled(cypher)
            out.append(
                Statement(
                    name=f"hard_delete_scope:{label}",
                    cypher=cypher,
                    params={"scope": self._scope},
                    kind="delete",
                    scope=self._scope,
                )
            )
        return out
