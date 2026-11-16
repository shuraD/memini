"""Graph setup statements and the single builder of vector-search clauses.

The setup list (uniqueness constraints, range indexes, cosine vector indexes) is derived
by ``memini.core.schema``; this module wraps it into admin statements and runs them
through a ``GraphSession``. Every vector-search clause is built by ``vector_search``,
which resolves the index name from the schema's own setup statements.
"""

from __future__ import annotations

import re

from memini.core import schema as schema_mod
from memini.core.graph.session import GraphSession
from memini.core.graph.types import Statement
from memini.core.schema import Schema

__all__ = ["ensure_indexes", "setup_statements", "vector_index_name", "vector_search"]

_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"


def _require_positive_int(value: object, what: str) -> int:
    """Return ``value`` when it is an int > 0 (bool refused), else raise ValueError."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{what} must be an int > 0, got {value!r}")
    return value


def _require_identifier(value: object, what: str) -> str:
    """Return ``value`` when it is a Python-style ASCII identifier, else raise ValueError."""
    if not isinstance(value, str) or re.fullmatch(_IDENTIFIER, value) is None:
        raise ValueError(f"{what} must be an identifier, got {value!r}")
    return value


def setup_statements(schema: Schema, *, vector_dimensions: int) -> list[Statement]:
    """Return the schema's setup list as admin statements, in the schema's order.

    Args:
        schema: The graph schema whose constraints and indexes are set up.
        vector_dimensions: Dimension count of every cosine vector index.

    Raises:
        ValueError: When ``vector_dimensions`` is not an int > 0.

    """
    dims = _require_positive_int(vector_dimensions, "vector_dimensions")
    cyphers = schema_mod.setup_statements(schema, vector_dimensions=dims)
    return [
        Statement(name=f"graph.setup.{i}", cypher=cypher, params={}, kind="admin")
        for i, cypher in enumerate(cyphers)
    ]


def ensure_indexes(session: GraphSession, schema: Schema, *, vector_dimensions: int) -> int:
    """Run every idempotent setup statement in order and return how many ran.

    A runner error propagates unchanged and later statements are not run.

    Raises:
        ValueError: When ``vector_dimensions`` is not an int > 0 (nothing runs).

    """
    stmts = setup_statements(schema, vector_dimensions=vector_dimensions)
    for stmt in stmts:
        # kind="admin" makes the session send each statement with autocommit=True.
        session.run(stmt)
    return len(stmts)


def vector_index_name(schema: Schema, label: str, prop: str) -> str:
    """Return the name of the vector index the schema declares for ``(label, prop)``.

    The name is read from the schema's own ``CREATE VECTOR INDEX`` setup statement
    (``memini.core.schema.setup_statements``); the match tolerates any variable name and
    whitespace in the ``FOR (<var>:<label>) ON (<var>.<prop>)`` part. Dimensions do not
    affect the name, so the statements are derived without them.

    Raises:
        ValueError: When ``label`` is not a schema node or ``prop`` has no vector index.

    """
    node = schema.nodes.get(label)
    if node is None or prop not in node.vector_indexes:
        raise ValueError(f"no vector index declared for {label}.{prop}")
    pattern = (
        rf"\s*CREATE\s+VECTOR\s+INDEX\s+(\S+)\s+IF\s+NOT\s+EXISTS\s+"
        rf"FOR\s*\(\s*({_IDENTIFIER})\s*:\s*{re.escape(label)}\s*\)\s*"
        rf"ON\s*\(\s*\2\s*\.\s*{re.escape(prop)}\s*\)"
    )
    for cypher in schema_mod.setup_statements(schema):
        match = re.match(pattern, cypher)
        if match is not None:
            return match.group(1)
    raise ValueError(f"no vector index statement for {label}.{prop}")


def vector_search(
    schema: Schema,
    label: str,
    prop: str,
    *,
    limit: int,
    var: str = "node",
    param: str = "vector",
) -> str:
    """Return the vector-search clause over the declared index of ``(label, prop)``.

    Callers append their own ``WHERE`` / ``RETURN`` after the clause.

    Args:
        schema: The graph schema declaring the vector index.
        label: Node label searched.
        prop: Vector property of ``label``.
        limit: Number of nearest neighbours returned.
        var: Variable bound to each matched node.
        param: Name of the query parameter holding the search vector.

    Raises:
        ValueError: When the index is not declared, ``limit`` is not an int > 0, or
            ``var`` / ``param`` is not an identifier.

    """
    name = vector_index_name(schema, label, prop)
    top = _require_positive_int(limit, "limit")
    v = _require_identifier(var, "var")
    p = _require_identifier(param, "param")
    return (
        f"MATCH ({v}:{label}) SEARCH {v} IN "
        f"(VECTOR INDEX {name} FOR ${p} LIMIT {top}) SCORE AS score"
    )
