"""Batched node and relationship writes that MERGE on ``id``.

The builder checks the schema and the row contents before building anything, then
returns one ``UNWIND $rows`` statement per batch. Every value travels as a parameter, so
only schema labels, relationship types and fixed keywords appear in the query text. A
corpus is kept apart by using its own database, so no write carries a scope. Nothing is
executed here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from memini.core.graph import checks
from memini.core.graph.types import Statement
from memini.core.schema import Schema

__all__ = ["RESERVED_NODE_FIELDS", "WriteBuilder"]

RESERVED_NODE_FIELDS = frozenset({"id"})
"""Node properties the builder sets itself; callers may not pass them in ``props``."""


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
    """Builds batched, schema-checked write statements."""

    def __init__(self, schema: Schema, *, batch_size: int = 1000) -> None:
        """Bind the schema.

        Args:
            schema: Schema declaring the labels, relationship types and properties.
            batch_size: Maximum number of rows per statement.

        Raises:
            ValueError: When ``batch_size`` is below 1.

        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self._schema = schema
        self._batch_size = batch_size

    def _batches(self, name: str, cypher: str, rows: list[dict[str, Any]]) -> list[Statement]:
        """Split ``rows`` into statements of at most ``batch_size`` rows, order kept."""
        if not rows:
            return []
        checks.require_labeled(cypher)
        size = self._batch_size
        return [
            Statement(
                name=name, cypher=cypher, params={"rows": rows[start : start + size]}, kind="write"
            )
            for start in range(0, len(rows), size)
        ]

    def merge_nodes(self, label: str, rows: Sequence[Mapping[str, Any]]) -> list[Statement]:
        """Build statements that merge nodes of ``label`` by id.

        Args:
            label: Schema node label.
            rows: Mappings with ``id`` (non-empty str) and optional ``props``.

        Returns:
            One statement per batch of rows; ``[]`` when ``rows`` is empty.

        Raises:
            ValueError: For an unknown label, a row without a valid id, or a reserved
                or undeclared property.

        """
        node = self._schema.nodes.get(label)
        if node is None:
            raise ValueError(f"unknown node label {label!r}")
        built = [
            {
                "id": _require_str(row, "id"),
                "props": _check_props(
                    row.get("props") or {}, RESERVED_NODE_FIELDS, node.properties, label
                ),
            }
            for row in rows
        ]
        cypher = f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n += row.props"
        return self._batches(f"write:{label}", cypher, built)

    def merge_relationships(
        self,
        rel: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        from_label: str,
        to_label: str,
    ) -> list[Statement]:
        """Build statements that merge ``rel`` relationships between nodes found by id.

        Args:
            rel: Schema relationship type.
            rows: Mappings with ``from_id``, ``to_id`` and optional ``props``.
            from_label: Label of the start nodes; a declared source of ``rel``.
            to_label: Label of the end nodes; a declared target of ``rel``.

        Returns:
            One statement per batch of rows; ``[]`` when ``rows`` is empty.

        Raises:
            ValueError: For an unknown type, an undeclared endpoint label, a row without
                valid ids, or an undeclared property.

        """
        spec = self._schema.relationships.get(rel)
        if spec is None:
            raise ValueError(f"unknown relationship type {rel!r}")
        if from_label not in spec.sources:
            raise ValueError(f"{from_label!r} is not a declared source of {rel!r}")
        if to_label not in spec.targets:
            raise ValueError(f"{to_label!r} is not a declared target of {rel!r}")
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
            f"MATCH (a:{from_label} {{id: row.from_id}}) "
            f"MATCH (b:{to_label} {{id: row.to_id}}) "
            f"MERGE (a)-[r:{rel}]->(b) SET r += row.props"
        )
        return self._batches(f"write:{rel}", cypher, built)
