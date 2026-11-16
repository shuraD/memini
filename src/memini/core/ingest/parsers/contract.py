"""Parser result types and the exact-span contract every parser result must honour.

Pure functions: no I/O. Offsets are character offsets into the parsed text.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedEntity:
    """One entity found by a parser; ``code`` is exactly ``content[start:end]``."""

    id: str
    kind: str
    name: str
    start: int
    end: int
    line_start: int
    line_end: int
    code: str
    signature: str = ""
    parent: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedEdge:
    """A typed link from one entity id to another."""

    type: str
    src: str
    dst: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseResult:
    """What a parser returns: its name, entities, edges and error messages."""

    parser: str
    entities: list[ParsedEntity] = field(default_factory=list)
    edges: list[ParsedEdge] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


Parser = Callable[[str, str], ParseResult]
"""A parser maps ``(content, path)`` to a :class:`ParseResult`."""


class ContractError(ValueError):
    """Raised when a parser result breaks the exact-span contract."""


def line_offsets(content: str) -> list[int]:
    """Return the char offset where each line starts; ``offsets[n-1]`` is line ``n``."""
    offsets = [0]
    offsets.extend(i + 1 for i, ch in enumerate(content) if ch == "\n")
    return offsets


def _line_of(offsets: list[int], offset: int) -> int:
    """Return the 1-based line holding the char at ``offset``."""
    return bisect_right(offsets, offset)


def make_entity(
    content: str,
    offsets: list[int],
    *,
    id: str,
    kind: str,
    name: str,
    start: int,
    end: int,
    signature: str = "",
    parent: str | None = None,
) -> ParsedEntity:
    """Build an entity for ``content[start:end]`` with its 1-based line range.

    ``end`` past the content is clamped to ``len(content)``. ``line_end`` is the
    line of the last character, or the line of ``start`` for an empty span.
    """
    end = min(end, len(content))
    line_start = _line_of(offsets, start)
    line_end = _line_of(offsets, end - 1) if end > start else line_start
    return ParsedEntity(
        id=id,
        kind=kind,
        name=name,
        start=start,
        end=end,
        line_start=line_start,
        line_end=line_end,
        code=content[start:end],
        signature=signature,
        parent=parent,
    )


def check_invariants(content: str, result: ParseResult) -> list[str]:
    """Return one message per contract violation in ``result``; ``[]`` when it holds."""
    problems: list[str] = []
    offsets = line_offsets(content)
    seen: set[str] = set()
    for ent in result.entities:
        if ent.id in seen:
            problems.append(f"duplicate entity id {ent.id!r}")
        seen.add(ent.id)
        if not 0 <= ent.start <= ent.end <= len(content):
            problems.append(f"{ent.id!r}: span [{ent.start}, {ent.end}) out of bounds")
            continue
        if ent.code != content[ent.start : ent.end]:
            problems.append(f"{ent.id!r}: code differs from content[start:end]")
        if ent.line_start != _line_of(offsets, ent.start):
            problems.append(f"{ent.id!r}: line_start {ent.line_start} is not the line of start")
        if ent.line_end < ent.line_start:
            problems.append(f"{ent.id!r}: line_end {ent.line_end} before line_start")
    return problems
