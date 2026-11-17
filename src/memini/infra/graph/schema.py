"""The graph schema: load ``graph_schema.yaml`` and ensure its constraints and indexes in Neo4j.

``load_schema`` reads the schema file into a typed :class:`GraphSchema`, including its
``names`` section as :class:`GraphNames` (the label, relationship type, property and status
name of every role the code refers to). ``ensure_schema`` creates, idempotently
(``IF NOT EXISTS`` / ``fail_if_exists=False``), with every graph name taken from those names:

- one ``(scope, id)`` uniqueness constraint per label that lists both properties;
- one range index on ``(scope, p)`` per label listing ``scope`` and each indexed ``p``;
- one cosine vector index per node property whose name starts with the embedding prefix.

The design follows the old memini's schema initialisation and vector-index setup; the
code is new.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import neo4j
import yaml
from neo4j_graphrag.indexes import create_vector_index

from memini.infra.paths import schema_file

# A range-indexed property with no role in the names section yet (spec GR-02, GR-06).
_UNNAMED_RANGE_PROPERTIES = frozenset({"name"})


@dataclass(frozen=True)
class GraphLabels:
    """The node label of each role."""

    project: str
    domain: str
    artifact: str
    chunk: str
    entity: str
    community: str
    relation_type: str


@dataclass(frozen=True)
class GraphRelationships:
    """The relationship type of each role."""

    has_domain: str
    contains: str
    has_chunk: str
    has_entity: str
    related_to: str
    similar: str
    code_link: str
    supersedes: str
    member_of: str
    hierarchical_link: str
    inter_layer_link: str


@dataclass(frozen=True)
class GraphProperties:
    """The property name of each role; ``embedding`` also prefixes every vector property."""

    scope: str
    id: str
    status: str
    content_hash: str
    updated_by: str
    updated_at: str
    summary: str
    embedding: str
    type: str
    level: str


@dataclass(frozen=True)
class GraphStatuses:
    """The status value of each role."""

    active: str
    superseded: str
    deleted: str


@dataclass(frozen=True)
class GraphNames:
    """The graph names the code uses, by role, as read from the schema's ``names`` section."""

    labels: GraphLabels
    relationships: GraphRelationships
    properties: GraphProperties
    statuses: GraphStatuses


def _roles[R: (GraphLabels, GraphRelationships, GraphProperties, GraphStatuses)](
    cls: type[R], raw: Mapping[str, Any] | None
) -> R:
    """Build one ``names`` subsection; a role the file does not name is named by the role."""
    given: Mapping[str, Any] = raw or {}
    return cls(**{f.name: str(given.get(f.name, f.name)) for f in fields(cls)})


def _names(raw: Mapping[str, Any] | None = None) -> GraphNames:
    """Build :class:`GraphNames` from the ``names`` section (absent: every role names itself)."""
    given: Mapping[str, Any] = raw or {}
    return GraphNames(
        labels=_roles(GraphLabels, given.get("labels")),
        relationships=_roles(GraphRelationships, given.get("relationships")),
        properties=_roles(GraphProperties, given.get("properties")),
        statuses=_roles(GraphStatuses, given.get("statuses")),
    )


@dataclass(frozen=True)
class GraphSchema:
    """The graph's labels, relationship types, property names, allowed values and role names.

    ``names`` defaults to every role naming itself, for a schema built without a names section.
    """

    nodes: Mapping[str, tuple[str, ...]]
    relationships: Mapping[str, tuple[str, ...]]
    values: Mapping[tuple[str, str], tuple[str, ...]]
    names: GraphNames = field(default_factory=_names)


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
    names = _names(data.get("names"))
    return GraphSchema(nodes=nodes, relationships=relationships, values=values, names=names)


def vector_index_name(label: str, prop: str) -> str:
    """Return the vector index name for ``label.prop``, e.g. ``entity_embedding_code``."""
    return f"{label.lower()}_{prop}"


def ensure_schema(
    driver: neo4j.Driver, schema: GraphSchema, dimension: int, database: str | None = None
) -> None:
    """Create the schema's constraints, range indexes and vector indexes, if missing."""
    p = schema.names.properties
    ranged = _UNNAMED_RANGE_PROPERTIES | {p.type, p.status, p.content_hash, p.level}
    for label, props in schema.nodes.items():
        base = label.lower()
        if p.scope in props and p.id in props:
            driver.execute_query(
                f"CREATE CONSTRAINT {base}_{p.scope}_{p.id} IF NOT EXISTS "
                f"FOR (n:`{label}`) REQUIRE (n.`{p.scope}`, n.`{p.id}`) IS UNIQUE",
                database_=database,
            )
        if p.scope in props:
            for prop in props:
                if prop in ranged:
                    driver.execute_query(
                        f"CREATE RANGE INDEX {base}_{p.scope}_{prop} IF NOT EXISTS "
                        f"FOR (n:`{label}`) ON (n.`{p.scope}`, n.`{prop}`)",
                        database_=database,
                    )
        for prop in props:
            if prop.startswith(p.embedding):
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
