"""The graph schema: load ``graph_schema.yaml`` and ensure its constraints and indexes in Neo4j.

``load_schema`` reads the schema file into a typed :class:`GraphSchema`. ``ensure_schema``
creates, idempotently (``IF NOT EXISTS`` / ``fail_if_exists=False``):

- one ``(scope, id)`` uniqueness constraint per label that lists both properties;
- one range index on ``(scope, p)`` per label listing ``scope`` and each indexed ``p``;
- one cosine vector index per node property whose name starts with the embedding prefix.

The design follows the old memini's schema initialisation and vector-index setup; the
code is new.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import neo4j
import yaml
from neo4j_graphrag.indexes import create_vector_index

from memini.infra.paths import schema_file

# The partition and identity keys every scoped node is addressed by (spec GR-02).
_SCOPE = "scope"
_ID = "id"
# Properties that get a (scope, p) range index when a label lists them (spec GR-02).
_RANGE_PROPERTIES = frozenset({"name", "type", "status", "content_hash", "level"})
# A node property whose name starts with this prefix holds a vector embedding.
_EMBEDDING_PREFIX = "embedding"


@dataclass(frozen=True)
class GraphSchema:
    """The graph's labels, relationship types, their property names and allowed values."""

    nodes: Mapping[str, tuple[str, ...]]
    relationships: Mapping[str, tuple[str, ...]]
    values: Mapping[tuple[str, str], tuple[str, ...]]


def _section(
    raw: Mapping[str, Any], values: dict[tuple[str, str], tuple[str, ...]]
) -> dict[str, tuple[str, ...]]:
    """Return each entry's property names, in file order, and collect its allowed values."""
    out: dict[str, tuple[str, ...]] = {}
    for name, entry in (raw or {}).items():
        body: Mapping[str, Any] = entry or {}
        out[str(name)] = tuple(str(p) for p in body.get("properties") or ())
        for prop, allowed in (body.get("values") or {}).items():
            values[(str(name), str(prop))] = tuple(str(v) for v in allowed or ())
    return out


def load_schema(path: Path | None = None) -> GraphSchema:
    """Load the schema file at ``path`` (default: the configured schema file)."""
    source = path if path is not None else schema_file()
    data: Mapping[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    values: dict[tuple[str, str], tuple[str, ...]] = {}
    nodes = _section(data.get("nodes") or {}, values)
    relationships = _section(data.get("relationships") or {}, values)
    return GraphSchema(nodes=nodes, relationships=relationships, values=values)


def vector_index_name(label: str, prop: str) -> str:
    """Return the vector index name for ``label.prop``, e.g. ``entity_embedding_code``."""
    return f"{label.lower()}_{prop}"


def ensure_schema(
    driver: neo4j.Driver, schema: GraphSchema, dimension: int, database: str | None = None
) -> None:
    """Create the schema's constraints, range indexes and vector indexes, if missing."""
    for label, props in schema.nodes.items():
        base = label.lower()
        if _SCOPE in props and _ID in props:
            driver.execute_query(
                f"CREATE CONSTRAINT {base}_{_SCOPE}_{_ID} IF NOT EXISTS "
                f"FOR (n:`{label}`) REQUIRE (n.`{_SCOPE}`, n.`{_ID}`) IS UNIQUE",
                database_=database,
            )
        if _SCOPE in props:
            for prop in props:
                if prop in _RANGE_PROPERTIES:
                    driver.execute_query(
                        f"CREATE RANGE INDEX {base}_{_SCOPE}_{prop} IF NOT EXISTS "
                        f"FOR (n:`{label}`) ON (n.`{_SCOPE}`, n.`{prop}`)",
                        database_=database,
                    )
        for prop in props:
            if prop.startswith(_EMBEDDING_PREFIX):
                create_vector_index(
                    driver,
                    name=vector_index_name(label, prop),
                    label=label,
                    embedding_property=prop,
                    dimensions=dimension,
                    similarity_fn="cosine",
                    fail_if_exists=False,
                    neo4j_database=database,
                )
