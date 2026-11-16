"""GS-08: batched node and relationship writes that MERGE on (scope, id).

Spec: GS-08 (src/memini/core/graph/writes.py) and SPEC-graph-schema, decision "Scale: writes
go in batches (UNWIND $rows)"; graph schema v2 (owner 2026-11-16): the builder is bound to a
scope and a clock, node writes MERGE on (scope, id) and set updated_by / updated_at, and
relationship endpoints are matched inside the scope (INTERFACE in test_graph_schema_v2.py).
Kept: a built row is the input row with ``props`` defaulted to {}; RESERVED_NODE_FIELDS now
holds the fields the builder sets itself: id, scope, updated_by and updated_at.
No database in these tests.
"""

import re

import pytest

from memini.core import schema
from memini.core.graph import checks
from memini.core.graph.types import Statement
from memini.core.graph.writes import RESERVED_NODE_FIELDS, WriteBuilder

SCOPE = "scopevalqzx"
NOW = "2026-11-16T09:00:00+00:00"
WRITER = "writervalqzx"
AUDIT = {"scope", "updated_by", "updated_at"}


def _prop(rule: str = "a rule") -> dict:
    return {"layer": "core", "writer": "ingest.register", "readers": ["query.*"], "rule": rule}


def _rel(src: str, dst: str, props: dict | None = None) -> dict:
    spec = {
        "from": src,
        "to": dst,
        "layer": "core",
        "writer": "ingest.register",
        "readers": ["query.*"],
        "rule": "a link",
    }
    if props:
        spec["properties"] = props
    return spec


def _schema() -> schema.Schema:
    return schema.from_dict(
        {
            "nodes": {
                "Document": {
                    "rule": "a source file",
                    "properties": {
                        **{name: _prop() for name in AUDIT},
                        "id": _prop(),
                        "title": _prop(),
                        "pages": _prop(),
                    },
                },
                "Topic": {
                    "rule": "a subject",
                    "properties": {
                        **{name: _prop() for name in AUDIT},
                        "id": _prop(),
                        "name": _prop(),
                    },
                },
            },
            "relationships": {
                "ABOUT": _rel("Document", "Topic", {"weight": _prop()}),
                "CITES": _rel("Document", "Document"),
            },
        }
    )


def _builder(**kwargs):
    return WriteBuilder(_schema(), scope=SCOPE, clock=lambda: NOW, **kwargs)


def _node_rows(n: int) -> list[dict]:
    return [{"id": f"doc-{i}", "updated_by": WRITER, "props": {"title": f"T{i}"}} for i in range(n)]


def _map_keys(cypher: str, clause: str, var: str, label: str) -> set[str] | None:
    """Return the keys of the ``<clause> (<var>:<label> {...})`` pattern map, if present."""
    m = re.search(rf"{clause} \({var}:{label} \{{([^}}]*)\}}\)", cypher)
    return set(re.findall(r"(\w+)\s*:", m.group(1))) if m else None


def _rel_rows(n: int) -> list[dict]:
    return [{"from_id": f"doc-{i}", "to_id": f"top-{i}", "props": {"weight": i}} for i in range(n)]


# --- reserved field sets -----------------------------------------------------------------


def test_reserved_field_sets():
    assert isinstance(RESERVED_NODE_FIELDS, frozenset)
    assert set(RESERVED_NODE_FIELDS) == {"id", "scope", "updated_by", "updated_at"}


# --- constructor -------------------------------------------------------------------------


@pytest.mark.parametrize("size", [0, -1])
def test_batch_size_below_one_is_refused(size):
    with pytest.raises(ValueError):
        WriteBuilder(_schema(), scope=SCOPE, clock=lambda: NOW, batch_size=size)


def test_batch_size_one_is_accepted():
    b = _builder(batch_size=1)
    assert len(b.merge_nodes("Document", _node_rows(3))) == 3


# --- merge_nodes: batching -----------------------------------------------------------------


def test_merge_nodes_batches_keep_input_order():
    rows = _node_rows(2500)
    stmts = _builder(batch_size=1000).merge_nodes("Document", rows)
    assert len(stmts) == 3
    sizes = [len(s.params["rows"]) for s in stmts]
    assert sizes == [1000, 1000, 500]
    ids = [r["id"] for s in stmts for r in s.params["rows"]]
    assert ids == [f"doc-{i}" for i in range(2500)]
    for s in stmts:
        assert s.cypher.startswith("UNWIND $rows AS row")


def test_merge_nodes_default_batch_size_is_1000():
    stmts = _builder().merge_nodes("Document", _node_rows(1001))
    assert [len(s.params["rows"]) for s in stmts] == [1000, 1]


def test_merge_nodes_exact_batch_gives_one_statement():
    stmts = _builder(batch_size=5).merge_nodes("Document", _node_rows(5))
    assert len(stmts) == 1
    assert len(stmts[0].params["rows"]) == 5


def test_merge_nodes_empty_rows_gives_no_statement():
    assert _builder().merge_nodes("Document", []) == []


def test_merge_nodes_rows_carry_id_and_props_and_default_props():
    rows = [
        {"id": "d1", "updated_by": WRITER, "props": {"title": "x", "pages": 3}},
        {"id": "d2", "updated_by": WRITER},
    ]
    (stmt,) = _builder().merge_nodes("Document", rows)
    assert stmt.params["rows"] == [
        {"id": "d1", "updated_by": WRITER, "props": {"title": "x", "pages": 3}},
        {"id": "d2", "updated_by": WRITER, "props": {}},
    ]


# --- merge_nodes: statement shape ------------------------------------------------------------


def test_merge_nodes_statement_fields():
    (stmt,) = _builder().merge_nodes("Document", _node_rows(2))
    assert isinstance(stmt, Statement)
    assert stmt.name == "write:Document"
    assert stmt.kind == "write"
    assert stmt.scope == SCOPE
    assert "rows" in stmt.params
    assert SCOPE in stmt.params.values()
    c = stmt.cypher
    assert _map_keys(c, "MERGE", "n", "Document") == {"scope", "id"}
    assert "SET n += row.props" in c


def test_merge_nodes_exact_cypher():
    (stmt,) = _builder().merge_nodes("Topic", [{"id": "t1", "updated_by": WRITER}])
    c = stmt.cypher
    assert c.startswith("UNWIND $rows AS row MERGE (n:Topic {")
    assert _map_keys(c, "MERGE", "n", "Topic") == {"scope", "id"}
    assert "SET n += row.props" in c
    assert re.search(r"\bn\.updated_by\s*=", c)
    assert re.search(r"\bn\.updated_at\s*=", c)
    assert stmt.name == "write:Topic"


def test_merge_nodes_values_never_in_cypher():
    rows = [
        {
            "id": "idvalqzx1",
            "updated_by": WRITER,
            "props": {"title": "titlevalqzx", "pages": 987654},
        },
        {"id": "idvalqzx2", "updated_by": WRITER, "props": {"title": "Robert'); DROP"}},
    ]
    stmts = _builder(batch_size=1).merge_nodes("Document", rows)
    assert len(stmts) == 2
    for s in stmts:
        for value in (
            "idvalqzx1",
            "idvalqzx2",
            "titlevalqzx",
            "987654",
            "DROP",
            SCOPE,
            WRITER,
            NOW,
        ):
            assert value not in s.cypher


def test_merge_nodes_statements_pass_checks():
    for stmt in _builder(batch_size=2).merge_nodes("Document", _node_rows(3)):
        checks.require_labeled(stmt.cypher)


# --- merge_nodes: refusals -------------------------------------------------------------------


def test_merge_nodes_unknown_label_is_refused():
    with pytest.raises(ValueError):
        _builder().merge_nodes("NoSuchLabel", _node_rows(1))


@pytest.mark.parametrize(
    "row",
    [
        {"updated_by": WRITER, "props": {"title": "x"}},
        {"id": "", "updated_by": WRITER},
        {"id": 7, "updated_by": WRITER},
        {"id": None, "updated_by": WRITER},
    ],
)
def test_merge_nodes_row_without_valid_id_is_refused(row):
    with pytest.raises(ValueError):
        _builder().merge_nodes("Document", [{"id": "ok", "updated_by": WRITER}, row])


@pytest.mark.parametrize("name", ["id", "scope", "updated_by", "updated_at"])
def test_merge_nodes_reserved_prop_is_refused(name):
    with pytest.raises(ValueError, match=name):
        _builder().merge_nodes(
            "Document", [{"id": "d1", "updated_by": WRITER, "props": {name: "x"}}]
        )


def test_merge_nodes_undeclared_prop_is_refused():
    with pytest.raises(ValueError, match="colourqzx"):
        _builder().merge_nodes(
            "Document", [{"id": "d1", "updated_by": WRITER, "props": {"colourqzx": "red"}}]
        )


def test_merge_nodes_prop_of_other_label_is_refused():
    with pytest.raises(ValueError, match="name"):
        _builder().merge_nodes(
            "Document", [{"id": "d1", "updated_by": WRITER, "props": {"name": "x"}}]
        )


def test_merge_nodes_bad_prop_in_later_batch_refuses_whole_call():
    rows = [*_node_rows(4), {"id": "d9", "updated_by": WRITER, "props": {"bogusqzx": 1}}]
    with pytest.raises(ValueError, match="bogusqzx"):
        _builder(batch_size=2).merge_nodes("Document", rows)


# --- merge_relationships: batching and shape -------------------------------------------------


def test_merge_relationships_empty_rows_gives_no_statement():
    b = _builder()
    assert b.merge_relationships("ABOUT", [], from_label="Document", to_label="Topic") == []


def test_merge_relationships_batches_keep_input_order():
    stmts = _builder(batch_size=1000).merge_relationships(
        "ABOUT", _rel_rows(2500), from_label="Document", to_label="Topic"
    )
    assert [len(s.params["rows"]) for s in stmts] == [1000, 1000, 500]
    ids = [r["from_id"] for s in stmts for r in s.params["rows"]]
    assert ids == [f"doc-{i}" for i in range(2500)]
    for s in stmts:
        assert s.cypher.startswith("UNWIND $rows AS row")


def test_merge_relationships_statement_fields():
    rows = [
        {"from_id": "d1", "to_id": "t1", "props": {"weight": 2}},
        {"from_id": "d2", "to_id": "t2"},
    ]
    (stmt,) = _builder().merge_relationships("ABOUT", rows, from_label="Document", to_label="Topic")
    assert stmt.name == "write:ABOUT"
    assert stmt.kind == "write"
    assert stmt.scope == SCOPE
    assert "rows" in stmt.params
    assert SCOPE in stmt.params.values()
    assert stmt.params["rows"] == [
        {"from_id": "d1", "to_id": "t1", "props": {"weight": 2}},
        {"from_id": "d2", "to_id": "t2", "props": {}},
    ]
    c = stmt.cypher
    assert _map_keys(c, "MATCH", "a", "Document") == {"scope", "id"}
    assert _map_keys(c, "MATCH", "b", "Topic") == {"scope", "id"}
    assert "row.from_id" in c and "row.to_id" in c
    assert "MERGE (a)-[r:ABOUT]->(b)" in c
    assert "SET r += row.props" in c


def test_merge_relationships_exact_cypher():
    (stmt,) = _builder().merge_relationships(
        "CITES", [{"from_id": "d1", "to_id": "d2"}], from_label="Document", to_label="Document"
    )
    c = stmt.cypher
    assert c.startswith("UNWIND $rows AS row MATCH (a:Document {")
    assert _map_keys(c, "MATCH", "a", "Document") == {"scope", "id"}
    assert _map_keys(c, "MATCH", "b", "Document") == {"scope", "id"}
    assert c.endswith("MERGE (a)-[r:CITES]->(b) SET r += row.props")


def test_merge_relationships_values_never_in_cypher():
    rows = [{"from_id": "fromqzx", "to_id": "toqzx", "props": {"weight": 4242}}]
    (stmt,) = _builder().merge_relationships("ABOUT", rows, from_label="Document", to_label="Topic")
    for value in ("fromqzx", "toqzx", "4242", SCOPE):
        assert value not in stmt.cypher


def test_merge_relationships_statements_pass_checks():
    stmts = _builder(batch_size=2).merge_relationships(
        "ABOUT", _rel_rows(3), from_label="Document", to_label="Topic"
    )
    assert len(stmts) == 2
    for stmt in stmts:
        checks.require_labeled(stmt.cypher)


# --- merge_relationships: refusals -----------------------------------------------------------


@pytest.mark.parametrize(
    ("rel", "from_label", "to_label"),
    [
        ("NO_SUCH", "Document", "Topic"),
        ("ABOUT", "Topic", "Topic"),
        ("ABOUT", "Document", "Document"),
        ("ABOUT", "Nope", "Topic"),
    ],
)
def test_merge_relationships_bad_type_or_endpoints_are_refused(rel, from_label, to_label):
    with pytest.raises(ValueError):
        _builder().merge_relationships(rel, _rel_rows(1), from_label=from_label, to_label=to_label)


def test_merge_relationships_undeclared_prop_is_refused():
    rows = [{"from_id": "d1", "to_id": "t1", "props": {"strengthqzx": 1}}]
    with pytest.raises(ValueError, match="strengthqzx"):
        _builder().merge_relationships("ABOUT", rows, from_label="Document", to_label="Topic")


def test_merge_relationships_prop_of_other_type_is_refused():
    rows = [{"from_id": "d1", "to_id": "d2", "props": {"weight": 1}}]
    with pytest.raises(ValueError, match="weight"):
        _builder().merge_relationships("CITES", rows, from_label="Document", to_label="Document")
