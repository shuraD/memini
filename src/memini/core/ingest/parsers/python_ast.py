"""Python parser built on the stdlib ``ast`` module: modules, classes, functions, methods.

Spans are exact character ranges of the source; ``ast`` columns (UTF-8 bytes) are
converted to character offsets.
"""

from __future__ import annotations

import ast
import re
from pathlib import PurePath

from memini.core.ingest.parsers.contract import (
    ParsedEdge,
    ParsedEntity,
    ParseResult,
    line_offsets,
    make_entity,
)
from memini.core.ingest.parsers.registry import register

NAME = "python_ast"
KIND_MODULE = "Module"
KIND_CLASS = "Class"
KIND_FUNCTION = "Function"
KIND_METHOD = "Method"
EDGE_DEFINES = "DEFINES"

_AST_NEWLINE = r"\r\n|\r|\n"
_Def = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


class _Positions:
    """Converts ``ast`` (line, UTF-8 byte column) positions to character offsets."""

    def __init__(self, content: str) -> None:
        self._starts = [0] + [m.end() for m in re.finditer(_AST_NEWLINE, content)]
        self._content = content

    def offset(self, lineno: int, col: int) -> int:
        """Return the char offset of byte column ``col`` on 1-based ``lineno``."""
        start = self._starts[lineno - 1]
        line_bytes = self._content[start : start + col].encode("utf-8")
        return start + len(line_bytes[:col].decode("utf-8", errors="ignore"))


class _Walker:
    """Collects definition entities and DEFINES edges in source order."""

    def __init__(self, content: str, path: str) -> None:
        self.content = content
        self.path = path
        self.offsets = line_offsets(content)
        self.positions = _Positions(content)
        self.entities: list[ParsedEntity] = []
        self.edges: list[ParsedEdge] = []
        self._ids: set[str] = set()

    def unique_id(self, base: str) -> str:
        """Return ``base``, suffixed with a counter when already used."""
        candidate, n = base, 1
        while candidate in self._ids:
            n += 1
            candidate = f"{base}#{n}"
        self._ids.add(candidate)
        return candidate

    def span(self, node: _Def) -> tuple[int, int]:
        """Return the char span of a definition, from its first decorator's ``@``."""
        start = self.positions.offset(node.lineno, node.col_offset)
        if node.decorator_list:
            first = node.decorator_list[0]
            expr = self.positions.offset(first.lineno, first.col_offset)
            start = self.content.rfind("@", 0, expr)
        end_line = node.end_lineno if node.end_lineno is not None else node.lineno
        end_col = node.end_col_offset if node.end_col_offset is not None else node.col_offset
        return start, self.positions.offset(end_line, end_col)

    def visit(self, node: ast.AST, parent: str, qual: str, in_class: bool) -> None:
        """Walk the children of ``node``, adding every definition found."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _Def):
                if isinstance(child, ast.ClassDef):
                    kind = KIND_CLASS
                else:
                    kind = KIND_METHOD if in_class else KIND_FUNCTION
                start, end = self.span(child)
                child_qual = f"{qual}.{child.name}" if qual else child.name
                ent_id = self.unique_id(f"{self.path}::{child_qual}")
                ent = make_entity(
                    self.content,
                    self.offsets,
                    id=ent_id,
                    kind=kind,
                    name=child.name,
                    start=start,
                    end=end,
                    parent=parent,
                )
                self.entities.append(ent)
                self.edges.append(ParsedEdge(type=EDGE_DEFINES, src=parent, dst=ent_id))
                self.visit(child, ent_id, child_qual, isinstance(child, ast.ClassDef))
            else:
                self.visit(child, parent, qual, in_class)


@register(NAME)
def parse_python(content: str, path: str) -> ParseResult:
    """Parse Python ``content``; a syntax error yields no entities and one error."""
    try:
        tree = ast.parse(content, filename=path)
    except (SyntaxError, ValueError) as exc:
        return ParseResult(parser=NAME, errors=[f"{type(exc).__name__}: {exc}"])
    walker = _Walker(content, path)
    module_id = walker.unique_id(path)
    module = make_entity(
        content,
        walker.offsets,
        id=module_id,
        kind=KIND_MODULE,
        name=PurePath(path).stem,
        start=0,
        end=len(content),
    )
    walker.entities.append(module)
    walker.visit(tree, module_id, "", False)
    return ParseResult(parser=NAME, entities=walker.entities, edges=walker.edges)
