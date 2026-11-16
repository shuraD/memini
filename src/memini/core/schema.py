"""The graph schema as data, with a proof of use for every stored item.

The schema lives in ``config/graph_schema.yaml`` (spec: SPEC-graph-schema). This
module loads and validates it, exposes the typed names all Cypher must use, derives
the constraint, index and setup statements and the duplicate-detector queries, and holds two guards:
``is_local_uri`` (production is never touched) and ``unlabeled_matches`` (every
``MATCH`` names a label, so no query scans the whole graph).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from memini.infra import paths
from memini.infra.graph import guard

LAYERS = ("infra", "core", "extensions")
LOCAL_HOSTS = guard.LOCAL_HOSTS

_NODE_PATTERN_RE = re.compile(r"\((?P<var>\w*)(?P<labels>(?::`?\w+`?)*)\s*(?P<props>\{[^}]*\})?\)")


@dataclass(frozen=True)
class Property:
    """One stored property with its proof of use."""

    layer: str
    writer: str
    readers: tuple[str, ...]
    rule: str
    ingest_only: str | None = None
    same_fact_as: str | None = None
    migrating_to: str | None = None  # a declared, temporary duplicate guarded by the detector


@dataclass(frozen=True)
class Node:
    """A node label, its properties and indexes."""

    rule: str
    properties: dict[str, Property]
    indexes: tuple[tuple[str, ...], ...] = ()
    vector_indexes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Relationship:
    """A relationship type with its endpoints and proof of use."""

    sources: tuple[str, ...]
    targets: tuple[str, ...]
    layer: str
    writer: str
    readers: tuple[str, ...]
    rule: str
    properties: dict[str, Property] = field(default_factory=dict)
    same_fact_as: str | None = None
    migrating_to: str | None = None  # a declared, temporary duplicate guarded by the detector


@dataclass(frozen=True)
class Schema:
    """The whole schema."""

    nodes: dict[str, Node] = field(default_factory=dict)
    relationships: dict[str, Relationship] = field(default_factory=dict)


@dataclass(frozen=True)
class Names:
    """Typed names for Cypher: ``labels[Label]``, ``rels[Type]``, ``props[Label][prop]``."""

    labels: dict[str, str]
    rels: dict[str, str]
    props: dict[str, dict[str, str]]
    rel_props: dict[str, dict[str, str]]


def _tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


def _properties(spec: dict[str, Any]) -> dict[str, Property]:
    """Build the ``properties`` mapping ``{name: {layer, writer, readers, rule, ...}}``."""
    return {
        name: Property(
            layer=str(p.get("layer", "")),
            writer=str(p.get("writer", "")),
            readers=_tuple(p.get("readers")),
            rule=str(p.get("rule", "")),
            ingest_only=p.get("ingest_only"),
            same_fact_as=p.get("same_fact_as"),
            migrating_to=p.get("migrating_to"),
        )
        for name, p in (spec.get("properties") or {}).items()
    }


def from_dict(raw: dict[str, Any]) -> Schema:
    """Build a :class:`Schema` from the YAML's dictionary form."""
    nodes = {
        label: Node(
            rule=str(spec.get("rule", "")),
            properties=_properties(spec),
            indexes=tuple(tuple(ix) for ix in spec.get("indexes") or ()),
            vector_indexes=_tuple(spec.get("vector_indexes")),
        )
        for label, spec in (raw.get("nodes") or {}).items()
    }
    relationships = {
        name: Relationship(
            sources=_tuple(spec.get("from")),
            targets=_tuple(spec.get("to")),
            layer=str(spec.get("layer", "")),
            writer=str(spec.get("writer", "")),
            readers=_tuple(spec.get("readers")),
            rule=str(spec.get("rule", "")),
            properties=_properties(spec),
            same_fact_as=spec.get("same_fact_as"),
            migrating_to=spec.get("migrating_to"),
        )
        for name, spec in (raw.get("relationships") or {}).items()
    }
    return Schema(nodes=nodes, relationships=relationships)


def load(path: Path | None = None) -> Schema:
    """Load the schema file (default: ``memini.infra.paths.schema_file()``)."""
    path = path or paths.schema_file()
    return from_dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _proof_problems(
    where: str,
    *,
    layer: str,
    writer: str,
    readers: tuple[str, ...],
    ingest_only: str | None,
    same_fact_as: str | None,
    rule: str = "present",
    migrating_to: str | None = None,
) -> list[str]:
    problems: list[str] = []
    if layer not in LAYERS:
        problems.append(f"{where}: layer must be one of {LAYERS}")
    if not writer:
        problems.append(f"{where}: no writer")
    if not readers and not ingest_only:
        problems.append(f"{where}: no reader and no ingest_only reason")
    if not rule.strip():
        problems.append(f"{where}: no rule (the business rule it serves)")
    if same_fact_as and not migrating_to:
        problems.append(
            f"{where}: holds the same fact as {same_fact_as} (store it once, or declare "
            "`migrating_to` so the duplicate detector guards it during the migration)"
        )
    return problems


def _property_problems(owner: str, properties: dict[str, Property]) -> list[str]:
    problems: list[str] = []
    for name, p in properties.items():
        problems += _proof_problems(
            f"{owner}.{name}",
            layer=p.layer,
            writer=p.writer,
            readers=p.readers,
            ingest_only=p.ingest_only,
            same_fact_as=p.same_fact_as,
            rule=p.rule,
            migrating_to=p.migrating_to,
        )
    return problems


def _node_problems(label: str, node: Node) -> list[str]:
    problems: list[str] = []
    if not node.rule.strip():
        problems.append(f"{label}: no rule")
    problems += _property_problems(label, node.properties)
    declared = node.properties
    for ix in node.indexes:
        problems += [
            f"{label}: index {ix} names undeclared property {col!r}"
            for col in ix
            if col not in declared
        ]
    problems += [
        f"{label}: vector index names undeclared property {col!r}"
        for col in node.vector_indexes
        if col not in declared
    ]
    return problems


def _is_declared_twin(schema: Schema, target: str) -> bool:
    """Return True if ``target`` is a declared relationship type or a declared ``Label.prop``."""
    if target in schema.relationships:
        return True
    label, dot, prop = target.partition(".")
    return bool(dot) and label in schema.nodes and prop in schema.nodes[label].properties


def _twin_problems(schema: Schema, owner: str, same_fact_as: str | None) -> list[str]:
    """Return the problem of a relationship-side ``same_fact_as`` naming no declared twin."""
    if same_fact_as and not _is_declared_twin(schema, same_fact_as):
        return [f"{owner}: same_fact_as names undeclared {same_fact_as}"]
    return []


def _relationship_problems(schema: Schema, name: str, rel: Relationship) -> list[str]:
    problems = _proof_problems(
        name,
        layer=rel.layer,
        writer=rel.writer,
        readers=rel.readers,
        ingest_only=None,
        same_fact_as=rel.same_fact_as,
        rule=rel.rule,
        migrating_to=rel.migrating_to,
    )
    problems += _twin_problems(schema, name, rel.same_fact_as)
    problems += _property_problems(name, rel.properties)
    for prop, p in rel.properties.items():
        problems += _twin_problems(schema, f"{name}.{prop}", p.same_fact_as)
    problems += [
        f"{name}: endpoint {end!r} is not a declared node"
        for end in (*rel.sources, *rel.targets)
        if end not in schema.nodes
    ]
    return problems


def validate(schema: Schema) -> list[str]:
    """Return every rule the schema breaks (empty = valid).

    Rules: every property and relationship has a layer, a writer and a reader (or an
    ``ingest_only`` reason) and a rule; a ``same_fact_as`` entry is accepted only as a
    declared migration (``migrating_to``), and on a relationship or relationship property it
    must name a declared relationship type or ``Label.prop``; indexes name declared
    properties; relationship endpoints are declared nodes.
    """
    problems: list[str] = []
    for label, node in schema.nodes.items():
        problems += _node_problems(label, node)
    for name, rel in schema.relationships.items():
        problems += _relationship_problems(schema, name, rel)
    return problems


def names(schema: Schema) -> Names:
    """Return the typed names derived from the schema."""
    return Names(
        labels={label: label for label in schema.nodes},
        rels={name: name for name in schema.relationships},
        props={label: {p: p for p in node.properties} for label, node in schema.nodes.items()},
        rel_props={
            name: {p: p for p in rel.properties} for name, rel in schema.relationships.items()
        },
    )


_UNIQUE_KEY = ("id",)


def _has_id(node: Node) -> bool:
    return "id" in node.properties


def constraint_statements(schema: Schema) -> list[str]:
    """Return one idempotent ``id`` uniqueness constraint per label that has an ``id``.

    Labels follow ``schema.nodes`` order.
    """
    return [
        f"CREATE CONSTRAINT {label.lower()}_id_unique IF NOT EXISTS "
        f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
        for label, node in schema.nodes.items()
        if _has_id(node)
    ]


def _indexes(
    schema: Schema, vector_dimensions: int | None
) -> list[tuple[str, tuple[str, ...], str]]:
    """Return ``(label, columns, statement)`` for every declared and generated index."""
    out: list[tuple[str, tuple[str, ...], str]] = []
    for label, node in schema.nodes.items():
        for ix in node.indexes:
            cols = ", ".join(f"n.{c}" for c in ix)
            ix_name = f"{label.lower()}_{'_'.join(ix)}"
            out.append(
                (label, ix, f"CREATE INDEX {ix_name} IF NOT EXISTS FOR (n:{label}) ON ({cols})")
            )
        for col in node.vector_indexes:
            ix_name = f"{label.lower()}_{col}"
            st = f"CREATE VECTOR INDEX {ix_name} IF NOT EXISTS FOR (n:{label}) ON (n.{col})"
            if vector_dimensions is not None:
                st += (
                    f" OPTIONS {{indexConfig: {{`vector.dimensions`: {vector_dimensions}, "
                    "`vector.similarity_function`: 'cosine'}}"
                )
            out.append((label, (col,), st))
    return out


def index_statements(schema: Schema, *, vector_dimensions: int | None = None) -> list[str]:
    """Return idempotent ``CREATE INDEX`` statements for every declared index.

    Vector indexes carry cosine ``OPTIONS`` with ``vector_dimensions`` only when it is given.
    """
    return [st for _, _, st in _indexes(schema, vector_dimensions)]


def setup_statements(schema: Schema, *, vector_dimensions: int | None = None) -> list[str]:
    """Return the ordered setup list: constraints first, then the indexes no constraint backs.

    An index on a constrained label whose columns equal the constraint's ``(id)`` is
    dropped (the constraint already provides it); no statement appears twice.
    """
    constraints = constraint_statements(schema)
    constrained = {label for label, node in schema.nodes.items() if _has_id(node)}
    indexes = [
        st
        for label, cols, st in _indexes(schema, vector_dimensions)
        if not (label in constrained and cols == _UNIQUE_KEY)
    ]
    return list(dict.fromkeys(constraints + indexes))


def duplicate_queries(schema: Schema) -> list[str]:
    """Return one count query per ``same_fact_as`` pair; each must return 0 on a clean graph.

    Node-property queries come first; then each relationship's ``same_fact_as`` yields one
    query per (source, target) label pair.
    """
    out: list[str] = []
    for label, node in schema.nodes.items():
        for prop, p in node.properties.items():
            if p.same_fact_as and p.same_fact_as in schema.relationships:
                out.append(
                    f"MATCH (n:{label}) WHERE n.{prop} IS NOT NULL "
                    f"AND EXISTS {{ (n)-[:{p.same_fact_as}]-() }} "
                    f"RETURN count(n) AS duplicates"
                )
            elif p.same_fact_as:
                other_label, other_prop = p.same_fact_as.split(".", 1)
                out.append(
                    f"MATCH (n:{label}) WHERE n.{prop} IS NOT NULL "
                    f"AND n.{other_prop} IS NOT NULL "
                    f"RETURN count(n) AS duplicates /* also stored as {other_label}.{other_prop} */"
                )
    for name, rel in schema.relationships.items():
        if rel.same_fact_as:
            out += _relationship_duplicate_queries(schema, name, rel, rel.same_fact_as)
    return out


def _relationship_duplicate_queries(
    schema: Schema, name: str, rel: Relationship, twin: str
) -> list[str]:
    """Return one count query per (source, target) pair of ``rel`` whose ``a``/``b`` hold ``twin``.

    ``twin`` is a relationship type (the same two nodes linked by it too) or ``Label.prop``
    (the endpoint carrying ``Label`` holds ``prop``; a pair with ``Label`` at neither end
    yields no query).
    """
    out: list[str] = []
    for source in rel.sources:
        for target in rel.targets:
            if twin in schema.relationships:
                condition = f"EXISTS {{ (a)-[:{twin}]-(b) }}"
            else:
                label, _, prop = twin.partition(".")
                if source == label:
                    condition = f"a.{prop} IS NOT NULL"
                elif target == label:
                    condition = f"b.{prop} IS NOT NULL"
                else:
                    continue
            out.append(
                f"MATCH (a:{source})-[r:{name}]->(b:{target}) WHERE {condition} "
                "RETURN count(r) AS duplicates"
            )
    return out


def is_local_uri(uri: str) -> bool:
    """Return True only for a database URI on this machine (production is never touched).

    The answer comes from the one database guard, :func:`guard.check_uri`, with no
    allow-list and no production host, so only local hosts in an accepted URI shape
    pass; no environment variable is read. Any refusal yields False (fail closed).
    """
    try:
        guard.check_uri(uri)
    except guard.UnsafeDatabaseError:
        return False
    return True


def unlabeled_matches(cypher: str) -> list[str]:
    """Return every node pattern in ``MATCH`` clauses that names no label."""
    found: list[str] = []
    for clause in re.findall(
        r"MATCH\s+(.+?)(?=\s+(?:MATCH|WHERE|WITH|RETURN|OPTIONAL|CALL)\b|$)", cypher, re.S
    ):
        for m in _NODE_PATTERN_RE.finditer(clause):
            if not m.group("labels"):
                found.append(m.group(0))
    return found
