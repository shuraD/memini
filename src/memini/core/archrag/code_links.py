"""CODE_LINK rows and merge statements from one parser result's structural edges.

Each edge whose type the settings map to a kind becomes one row between two entities of the
same parse result, carrying that kind and a source line. Edges whose ends are not entities of
the result are handed back as unresolved for a later resolution step. Nothing is read from or
executed against the database here; the caller runs the returned statements.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from memini.core.graph.types import Statement
from memini.core.graph.writes import WriteBuilder
from memini.core.ingest.parsers.contract import ParsedEdge, ParseResult
from memini.core.schema import Schema

__all__ = ["CodeLinkSettings", "code_link_rows", "code_link_statements"]

_REL = "CODE_LINK"
_LABEL = "Entity"
_LINE = "line"
_KIND = "kind"


@dataclass(frozen=True)
class CodeLinkSettings:
    """Mapping from parser edge types to CODE_LINK kinds.

    Attributes:
        kinds: Parser edge type -> CODE_LINK kind; edges of other types are ignored.

    """

    kinds: Mapping[str, str]

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> CodeLinkSettings:
        """Read key "kinds": a non-empty mapping of non-empty str -> non-empty str.

        Raises:
            ValueError: When the key is missing, is not a mapping, is empty, or holds a key
                or value that is not a non-empty string.

        """
        if "kinds" not in section:
            raise ValueError("code link config misses 'kinds'")
        kinds = section["kinds"]
        if not isinstance(kinds, Mapping) or not kinds:
            raise ValueError(f"code link 'kinds' must be a non-empty mapping, got {kinds!r}")
        for key, value in kinds.items():
            if not isinstance(key, str) or not key or not isinstance(value, str) or not value:
                raise ValueError(f"code link kind must map str to str: {key!r} -> {value!r}")
        return cls(kinds=dict(kinds))


def _line(edge: ParsedEdge, line_starts: Mapping[str, int]) -> int:
    """Return the edge's own line when it is an int >= 1, else its dst entity's line_start."""
    value = edge.extra.get(_LINE)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    return line_starts[edge.dst]


def code_link_rows(
    result: ParseResult, settings: CodeLinkSettings
) -> tuple[list[dict[str, Any]], list[ParsedEdge]]:
    """Return ``(rows, unresolved)`` for the mapped edges of ``result``.

    A row is ``{"from_id": src, "to_id": dst, "props": {"kind": kind, "line": line}}``.
    Edges of an unmapped type are skipped; mapped edges with an end outside the result's
    entities go unchanged to ``unresolved``; the first mapped edge of a (src, dst) pair wins.
    Both lists keep parser edge order.
    """
    line_starts = {ent.id: ent.line_start for ent in result.entities}
    rows: list[dict[str, Any]] = []
    unresolved: list[ParsedEdge] = []
    seen: set[tuple[str, str]] = set()
    for edge in result.edges:
        kind = settings.kinds.get(edge.type)
        if kind is None:
            continue
        if edge.src not in line_starts or edge.dst not in line_starts:
            unresolved.append(edge)
            continue
        pair = (edge.src, edge.dst)
        if pair in seen:
            continue
        seen.add(pair)
        rows.append(
            {
                "from_id": edge.src,
                "to_id": edge.dst,
                "props": {"kind": kind, _LINE: _line(edge, line_starts)},
            }
        )
    return rows, unresolved


def code_link_statements(
    schema: Schema, scope: str, result: ParseResult, settings: CodeLinkSettings
) -> tuple[list[Statement], list[ParsedEdge]]:
    """Return ``(statements, unresolved)``: the CODE_LINK merges of ``code_link_rows``.

    Statements merge CODE_LINK relationships between Entity nodes of ``scope`` (``[]`` when
    there is no row); ``unresolved`` is as in ``code_link_rows``.

    Raises:
        ValueError: For an empty scope or a kind outside the schema's CODE_LINK kind values.

    """
    if not isinstance(scope, str) or not scope:
        raise ValueError(f"scope must be a non-empty string, got {scope!r}")
    _check_kinds(schema, settings)
    rows, unresolved = code_link_rows(result, settings)
    if not rows:
        return [], unresolved
    builder = WriteBuilder(schema, scope=scope, clock=_no_timestamp)
    statements = builder.merge_relationships(_REL, rows, from_label=_LABEL, to_label=_LABEL)
    return statements, unresolved


def _no_timestamp() -> None:
    """Clock for the builder: relationship merges stamp no ``updated_at``, so none is read."""


def _check_kinds(schema: Schema, settings: CodeLinkSettings) -> None:
    """Raise ``ValueError`` when a mapped kind is not an allowed CODE_LINK ``kind`` value."""
    spec = schema.relationships.get(_REL)
    if spec is None:
        raise ValueError(f"schema declares no {_REL!r} relationship")
    prop = spec.properties.get(_KIND)
    if prop is None:
        raise ValueError(f"{_REL!r} declares no {_KIND!r} property")
    allowed = prop.values
    for kind in settings.kinds.values():
        if allowed and kind not in allowed:
            raise ValueError(f"{_REL}.{_KIND} does not allow {kind!r}; allowed: {allowed!r}")
