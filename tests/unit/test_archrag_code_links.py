"""AR-19: CODE_LINK rows and merge statements from one parser result's structural edges.

Spec: AR-19 (src/memini/core/archrag/code_links.py), card line A92. Pure functions: no graph,
no I/O. Edge-type to kind mapping comes from a CodeLinkSettings built from a config mapping.
"""

import pytest

from memini.core import schema as schema_mod
from memini.core.archrag.code_links import (
    CodeLinkSettings,
    code_link_rows,
    code_link_statements,
)
from memini.core.ingest.parsers.contract import ParsedEdge, ParsedEntity, ParseResult
from memini.core.ingest.parsers.python_ast import parse_python

KINDS = {"DEFINES": "define", "CALLS": "call", "IMPORTS": "import"}
SCOPE = "scope-qzx"


def _settings() -> CodeLinkSettings:
    return CodeLinkSettings.from_config({"kinds": dict(KINDS)})


def _entity(ent_id: str, line_start: int) -> ParsedEntity:
    return ParsedEntity(
        id=ent_id,
        kind="kindqzx",
        name=ent_id,
        start=0,
        end=0,
        line_start=line_start,
        line_end=line_start,
        code="",
    )


def _result(edges: list[ParsedEdge]) -> ParseResult:
    entities = [_entity("a", 1), _entity("b", 4), _entity("c", 9)]
    return ParseResult(parser="parserqzx", entities=entities, edges=edges)


def _row(src: str, dst: str, kind: str, line: int) -> dict:
    return {"from_id": src, "to_id": dst, "props": {"kind": kind, "line": line}}


# --- CodeLinkSettings.from_config -------------------------------------------------------


def test_from_config_reads_kinds() -> None:
    settings = CodeLinkSettings.from_config({"kinds": dict(KINDS)})
    assert dict(settings.kinds) == KINDS


@pytest.mark.parametrize(
    "section",
    [
        {},
        {"other": {"DEFINES": "define"}},
        {"kinds": {}},
        {"kinds": "DEFINES=define"},
        {"kinds": ["DEFINES", "define"]},
        {"kinds": None},
        {"kinds": {"": "define"}},
        {"kinds": {"DEFINES": ""}},
        {"kinds": {"DEFINES": 1}},
        {"kinds": {1: "define"}},
        {"kinds": {"DEFINES": None}},
        {"kinds": {"DEFINES": "define", "CALLS": ["call"]}},
    ],
)
def test_from_config_rejects_bad_section(section: dict) -> None:
    with pytest.raises(ValueError):
        CodeLinkSettings.from_config(section)


# --- code_link_rows ----------------------------------------------------------------------


def test_rows_one_per_edge_in_parser_order_with_kind_and_line() -> None:
    edges = [
        ParsedEdge(type="IMPORTS", src="a", dst="c", extra={"line": 2}),
        ParsedEdge(type="DEFINES", src="a", dst="b", extra={"line": 4}),
        ParsedEdge(type="CALLS", src="b", dst="c", extra={"line": 6}),
    ]
    rows, unresolved = code_link_rows(_result(edges), _settings())
    assert rows == [
        _row("a", "c", "import", 2),
        _row("a", "b", "define", 4),
        _row("b", "c", "call", 6),
    ]
    assert unresolved == []


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"line": "3"},
        {"line": 3.0},
        {"line": True},
        {"line": False},
        {"line": 0},
        {"line": -2},
        {"line": None},
        {"other": 5},
    ],
)
def test_row_line_falls_back_to_dst_line_start(extra: dict) -> None:
    edges = [ParsedEdge(type="CALLS", src="a", dst="c", extra=extra)]
    rows, unresolved = code_link_rows(_result(edges), _settings())
    assert rows == [_row("a", "c", "call", 9)]
    assert unresolved == []


def test_unmapped_edge_type_is_skipped() -> None:
    edges = [
        ParsedEdge(type="OTHERQZX", src="a", dst="b", extra={"line": 3}),
        ParsedEdge(type="OTHERQZX", src="a", dst="missing", extra={"line": 3}),
        ParsedEdge(type="DEFINES", src="a", dst="b", extra={"line": 4}),
    ]
    rows, unresolved = code_link_rows(_result(edges), _settings())
    assert rows == [_row("a", "b", "define", 4)]
    assert unresolved == []


def test_unresolved_edges_returned_unchanged_in_order() -> None:
    first = ParsedEdge(type="IMPORTS", src="a", dst="pkg.mod", extra={"line": 1})
    second = ParsedEdge(type="CALLS", src="elsewhere", dst="b", extra={"line": 5})
    third = ParsedEdge(type="CALLS", src="b", dst="far.func")
    edges = [
        first,
        ParsedEdge(type="DEFINES", src="a", dst="b", extra={"line": 4}),
        second,
        third,
    ]
    rows, unresolved = code_link_rows(_result(edges), _settings())
    assert rows == [_row("a", "b", "define", 4)]
    assert len(unresolved) == 3
    assert unresolved[0] is first
    assert unresolved[1] is second
    assert unresolved[2] is third
    assert first == ParsedEdge(type="IMPORTS", src="a", dst="pkg.mod", extra={"line": 1})
    assert third.extra == {}


def test_duplicate_pair_first_edge_wins() -> None:
    edges = [
        ParsedEdge(type="CALLS", src="a", dst="b", extra={"line": 7}),
        ParsedEdge(type="DEFINES", src="a", dst="c", extra={"line": 9}),
        ParsedEdge(type="DEFINES", src="a", dst="b", extra={"line": 4}),
        ParsedEdge(type="CALLS", src="a", dst="b", extra={"line": 8}),
    ]
    rows, unresolved = code_link_rows(_result(edges), _settings())
    assert rows == [_row("a", "b", "call", 7), _row("a", "c", "define", 9)]
    assert unresolved == []


def test_reverse_pair_is_a_distinct_pair() -> None:
    edges = [
        ParsedEdge(type="CALLS", src="a", dst="b", extra={"line": 2}),
        ParsedEdge(type="CALLS", src="b", dst="a", extra={"line": 5}),
    ]
    rows, _ = code_link_rows(_result(edges), _settings())
    assert rows == [_row("a", "b", "call", 2), _row("b", "a", "call", 5)]


def test_python_ast_defines_become_define_rows() -> None:
    content = "import os\n\n\nclass Box:\n    def open(self):\n        return os.sep\n"
    result = parse_python(content, "pkg/box.py")
    by_id = {ent.id: ent for ent in result.entities}
    defines = [edge for edge in result.edges if edge.type == "DEFINES"]
    assert len(defines) == 2
    rows, unresolved = code_link_rows(result, _settings())
    assert unresolved == []
    assert rows == [
        _row(edge.src, edge.dst, "define", by_id[edge.dst].line_start) for edge in defines
    ]
    assert [row["props"]["line"] for row in rows] == [4, 5]


# --- code_link_statements ----------------------------------------------------------------


def test_statements_merge_code_links_between_entities() -> None:
    missing = ParsedEdge(type="IMPORTS", src="a", dst="pkg.mod", extra={"line": 1})
    edges = [
        ParsedEdge(type="DEFINES", src="a", dst="b", extra={"line": 4}),
        missing,
        ParsedEdge(type="CALLS", src="b", dst="c"),
        ParsedEdge(type="IMPORTS", src="a", dst="c", extra={"line": 2}),
    ]
    result = _result(edges)
    settings = _settings()
    expected_rows, expected_unresolved = code_link_rows(result, settings)
    statements, unresolved = code_link_statements(schema_mod.load(), SCOPE, result, settings)
    assert statements
    rows: list = []
    for stmt in statements:
        assert stmt.name == "write:CODE_LINK"
        assert stmt.kind == "write"
        assert stmt.scope == SCOPE
        assert stmt.params["scope"] == SCOPE
        assert "MERGE" in stmt.cypher
        assert "CODE_LINK" in stmt.cypher
        assert ":Entity" in stmt.cypher
        rows.extend(stmt.params["rows"])
    assert rows == expected_rows
    assert len(rows) == 3
    assert len(unresolved) == 1
    assert unresolved[0] is missing
    assert unresolved == expected_unresolved


def test_statements_empty_when_no_resolved_edge() -> None:
    missing = ParsedEdge(type="CALLS", src="a", dst="far.func")
    edges = [missing, ParsedEdge(type="OTHERQZX", src="a", dst="b")]
    statements, unresolved = code_link_statements(
        schema_mod.load(), SCOPE, _result(edges), _settings()
    )
    assert statements == []
    assert len(unresolved) == 1
    assert unresolved[0] is missing


def test_statements_empty_for_result_without_edges() -> None:
    statements, unresolved = code_link_statements(
        schema_mod.load(), SCOPE, _result([]), _settings()
    )
    assert statements == []
    assert unresolved == []


def test_statements_reject_kind_outside_schema_values() -> None:
    settings = CodeLinkSettings.from_config({"kinds": {"DEFINES": "kindoutsideqzx"}})
    edges = [ParsedEdge(type="DEFINES", src="a", dst="b", extra={"line": 4})]
    with pytest.raises(ValueError):
        code_link_statements(schema_mod.load(), SCOPE, _result(edges), settings)


def test_statements_reject_empty_scope() -> None:
    edges = [ParsedEdge(type="DEFINES", src="a", dst="b", extra={"line": 4})]
    with pytest.raises(ValueError):
        code_link_statements(schema_mod.load(), "", _result(edges), _settings())
