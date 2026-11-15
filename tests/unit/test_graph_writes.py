"""GS-08: batched node and relationship writes that MERGE on (scope, id).

Spec: GS-08 (src/memini/core/graph/writes.py) and SPEC-graph-schema, decisions "The query
builder sets the audit fields on every write", "The query builder enforces isolation" and
"Scale: writes go in batches (UNWIND $rows) ... every write MERGEs on (scope, id)".
No database in these tests.
"""

from datetime import UTC, datetime, timedelta

import pytest

from memini.core import schema
from memini.core.graph import checks
from memini.core.graph.types import AccessContext, AccessDenied, IsolationError, Statement
from memini.core.graph.writes import RESERVED_NODE_FIELDS, RESERVED_REL_FIELDS, WriteBuilder

NOW = datetime(2026, 11, 15, 10, 0, 0, tzinfo=UTC)

COMMON_NODE_FIELDS = (
    "scope",
    "status",
    "created_at",
    "updated_at",
    "updated_by",
    "deleted_at",
    "deleted_by",
)


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
            "common": {
                "nodes": {name: _prop(name) for name in COMMON_NODE_FIELDS},
                "relationships": {"created_at": _prop("when linked")},
            },
            "nodes": {
                "Document": {
                    "rule": "a source file",
                    "properties": {"id": _prop(), "title": _prop(), "pages": _prop()},
                },
                "Topic": {"rule": "a subject", "properties": {"id": _prop(), "name": _prop()}},
            },
            "relationships": {
                "ABOUT": _rel("Document", "Topic", {"weight": _prop()}),
                "CITES": _rel("Document", "Document"),
            },
        }
    )


def _ctx(write=("A",), read=("A", "B"), actor="job-run-7") -> AccessContext:
    return AccessContext(
        principal="tester",
        read_scopes=frozenset(read) | frozenset(write),
        write_scopes=frozenset(write),
        actor=actor,
    )


def _clock() -> datetime:
    return NOW


def _builder(ctx: AccessContext | None = None, **kwargs):
    return WriteBuilder(_schema(), ctx if ctx is not None else _ctx(), _clock, **kwargs)


def _node_rows(n: int) -> list[dict]:
    return [{"id": f"doc-{i}", "props": {"title": f"T{i}"}} for i in range(n)]


def _rel_rows(n: int) -> list[dict]:
    return [{"from_id": f"doc-{i}", "to_id": f"top-{i}", "props": {"weight": i}} for i in range(n)]


# --- reserved field sets -----------------------------------------------------------------


def test_reserved_field_sets():
    expected_node = {"scope", "id", "status", "created_at", "updated_at", "updated_by"}
    expected_node |= {"deleted_at", "deleted_by"}
    assert isinstance(RESERVED_NODE_FIELDS, frozenset)
    assert set(RESERVED_NODE_FIELDS) == expected_node
    assert isinstance(RESERVED_REL_FIELDS, frozenset)
    assert set(RESERVED_REL_FIELDS) == {"created_at"}


# --- constructor -------------------------------------------------------------------------


@pytest.mark.parametrize("size", [0, -1])
def test_batch_size_below_one_is_refused(size):
    with pytest.raises(ValueError):
        WriteBuilder(_schema(), _ctx(), _clock, batch_size=size)


def test_missing_context_is_refused():
    with pytest.raises(AccessDenied):
        WriteBuilder(_schema(), None, _clock)


def test_batch_size_one_is_accepted():
    b = _builder(batch_size=1)
    assert len(b.merge_nodes("Document", "A", _node_rows(3))) == 3


# --- merge_nodes: batching -----------------------------------------------------------------


def test_merge_nodes_batches_keep_input_order():
    rows = _node_rows(2500)
    stmts = _builder(batch_size=1000).merge_nodes("Document", "A", rows)
    assert len(stmts) == 3
    sizes = [len(s.params["rows"]) for s in stmts]
    assert sizes == [1000, 1000, 500]
    ids = [r["id"] for s in stmts for r in s.params["rows"]]
    assert ids == [f"doc-{i}" for i in range(2500)]
    for s in stmts:
        assert s.cypher.startswith("UNWIND $rows AS row")


def test_merge_nodes_default_batch_size_is_1000():
    stmts = _builder().merge_nodes("Document", "A", _node_rows(1001))
    assert [len(s.params["rows"]) for s in stmts] == [1000, 1]


def test_merge_nodes_exact_batch_gives_one_statement():
    stmts = _builder(batch_size=5).merge_nodes("Document", "A", _node_rows(5))
    assert len(stmts) == 1
    assert len(stmts[0].params["rows"]) == 5


def test_merge_nodes_empty_rows_gives_no_statement():
    assert _builder().merge_nodes("Document", "A", []) == []


def test_merge_nodes_rows_carry_id_and_props_and_default_props():
    rows = [{"id": "d1", "props": {"title": "x", "pages": 3}}, {"id": "d2"}]
    (stmt,) = _builder().merge_nodes("Document", "A", rows)
    assert stmt.params["rows"] == [
        {"id": "d1", "props": {"title": "x", "pages": 3}},
        {"id": "d2", "props": {}},
    ]


# --- merge_nodes: statement shape ------------------------------------------------------------


def test_merge_nodes_statement_fields():
    ctx = _ctx(actor="api-caller-9")
    (stmt,) = _builder(ctx).merge_nodes("Document", "A", _node_rows(2))
    assert isinstance(stmt, Statement)
    assert stmt.name == "write:Document"
    assert stmt.kind == "write"
    assert stmt.scope == "A"
    assert stmt.params["scope"] == "A"
    assert stmt.params["actor"] == "api-caller-9"
    assert set(stmt.params) == {"scope", "rows", "now", "actor"}
    c = stmt.cypher
    assert "MERGE (n:Document {scope: $scope, id: row.id})" in c
    assert "ON CREATE SET n.created_at = datetime($now)" in c
    assert "n.status = 'active'" in c
    assert "n.updated_at = datetime($now)" in c
    assert "n.updated_by = $actor" in c
    assert "SET n += row.props" in c


def test_merge_nodes_exact_cypher():
    (stmt,) = _builder().merge_nodes("Topic", "A", [{"id": "t1"}])
    assert stmt.cypher == (
        "UNWIND $rows AS row MERGE (n:Topic {scope: $scope, id: row.id}) "
        "ON CREATE SET n.created_at = datetime($now), n.status = 'active' "
        "SET n += row.props, n.updated_at = datetime($now), n.updated_by = $actor"
    )
    assert stmt.name == "write:Topic"


def test_merge_nodes_now_is_clock_time():
    (stmt,) = _builder().merge_nodes("Document", "A", _node_rows(1))
    assert datetime.fromisoformat(stmt.params["now"]) == NOW


def test_merge_nodes_same_now_in_every_batch():
    ticks = iter(NOW + timedelta(seconds=i) for i in range(100))
    b = WriteBuilder(_schema(), _ctx(), lambda: next(ticks), batch_size=2)
    stmts = b.merge_nodes("Document", "A", _node_rows(5))
    assert len(stmts) == 3
    nows = {s.params["now"] for s in stmts}
    assert len(nows) == 1
    assert datetime.fromisoformat(nows.pop()) == NOW


def test_merge_nodes_values_never_in_cypher():
    scope = "scopeqzx"
    ctx = _ctx(write=(scope,), read=(scope,))
    rows = [
        {"id": "idvalqzx1", "props": {"title": "titlevalqzx", "pages": 987654}},
        {"id": "idvalqzx2", "props": {"title": "Robert'); DROP"}},
    ]
    stmts = _builder(ctx, batch_size=1).merge_nodes("Document", scope, rows)
    assert len(stmts) == 2
    for s in stmts:
        for value in (scope, "idvalqzx1", "idvalqzx2", "titlevalqzx", "987654", "DROP"):
            assert value not in s.cypher


def test_merge_nodes_statements_pass_checks():
    for stmt in _builder(batch_size=2).merge_nodes("Document", "A", _node_rows(3)):
        assert checks.audit_write_problems(stmt.cypher) == []
        checks.require_labeled(stmt.cypher)
        checks.require_audited(stmt.cypher)


# --- merge_nodes: refusals -------------------------------------------------------------------


def test_merge_nodes_unwritable_scope_is_denied():
    with pytest.raises(AccessDenied, match="B"):
        _builder().merge_nodes("Document", "B", _node_rows(1))


def test_merge_nodes_readable_only_scope_is_denied():
    ctx = _ctx(write=("A",), read=("A", "B"))
    assert "B" in ctx.read_scopes
    with pytest.raises(AccessDenied, match="B"):
        _builder(ctx).merge_nodes("Document", "B", _node_rows(1))


def test_merge_nodes_access_checked_before_label():
    with pytest.raises(AccessDenied):
        _builder().merge_nodes("NoSuchLabel", "B", _node_rows(1))


def test_merge_nodes_access_checked_even_for_empty_rows():
    with pytest.raises(AccessDenied):
        _builder().merge_nodes("Document", "B", [])


def test_merge_nodes_unknown_label_is_refused():
    with pytest.raises(ValueError):
        _builder().merge_nodes("NoSuchLabel", "A", _node_rows(1))


@pytest.mark.parametrize(
    "row",
    [
        {"props": {"title": "x"}},
        {"id": ""},
        {"id": 7},
        {"id": None},
    ],
)
def test_merge_nodes_row_without_valid_id_is_refused(row):
    with pytest.raises(ValueError):
        _builder().merge_nodes("Document", "A", [{"id": "ok"}, row])


@pytest.mark.parametrize(
    "name",
    ["created_at", "scope", "updated_by", "status", "id", "updated_at", "deleted_at", "deleted_by"],
)
def test_merge_nodes_reserved_prop_is_refused(name):
    with pytest.raises(ValueError, match=name):
        _builder().merge_nodes("Document", "A", [{"id": "d1", "props": {name: "x"}}])


def test_merge_nodes_undeclared_prop_is_refused():
    with pytest.raises(ValueError, match="colourqzx"):
        _builder().merge_nodes("Document", "A", [{"id": "d1", "props": {"colourqzx": "red"}}])


def test_merge_nodes_prop_of_other_label_is_refused():
    with pytest.raises(ValueError, match="name"):
        _builder().merge_nodes("Document", "A", [{"id": "d1", "props": {"name": "x"}}])


def test_merge_nodes_bad_prop_in_later_batch_refuses_whole_call():
    rows = [*_node_rows(4), {"id": "d9", "props": {"bogusqzx": 1}}]
    with pytest.raises(ValueError, match="bogusqzx"):
        _builder(batch_size=2).merge_nodes("Document", "A", rows)


# --- merge_relationships: batching and shape -------------------------------------------------


def test_merge_relationships_empty_rows_gives_no_statement():
    b = _builder()
    assert b.merge_relationships("ABOUT", "A", [], from_label="Document", to_label="Topic") == []


def test_merge_relationships_batches_keep_input_order():
    stmts = _builder(batch_size=1000).merge_relationships(
        "ABOUT", "A", _rel_rows(2500), from_label="Document", to_label="Topic"
    )
    assert [len(s.params["rows"]) for s in stmts] == [1000, 1000, 500]
    ids = [r["from_id"] for s in stmts for r in s.params["rows"]]
    assert ids == [f"doc-{i}" for i in range(2500)]
    for s in stmts:
        assert s.cypher.startswith("UNWIND $rows AS row")


def test_merge_relationships_statement_fields():
    ctx = _ctx(actor="api-caller-9")
    rows = [
        {"from_id": "d1", "to_id": "t1", "props": {"weight": 2}},
        {"from_id": "d2", "to_id": "t2", "from_scope": "A", "to_scope": "A"},
    ]
    (stmt,) = _builder(ctx).merge_relationships(
        "ABOUT", "A", rows, from_label="Document", to_label="Topic"
    )
    assert stmt.name == "write:ABOUT"
    assert stmt.kind == "write"
    assert stmt.scope == "A"
    assert set(stmt.params) == {"scope", "rows", "now", "actor"}
    assert stmt.params["scope"] == "A"
    assert stmt.params["actor"] == "api-caller-9"
    assert datetime.fromisoformat(stmt.params["now"]) == NOW
    assert stmt.params["rows"] == [
        {"from_id": "d1", "to_id": "t1", "props": {"weight": 2}},
        {"from_id": "d2", "to_id": "t2", "props": {}},
    ]
    c = stmt.cypher
    assert "MATCH (a:Document {scope: $scope, id: row.from_id})" in c
    assert "MATCH (b:Topic {scope: $scope, id: row.to_id})" in c
    assert "MERGE (a)-[r:ABOUT]->(b)" in c
    assert "ON CREATE SET r.created_at = datetime($now)" in c
    assert "SET r += row.props" in c


def test_merge_relationships_exact_cypher():
    (stmt,) = _builder().merge_relationships(
        "CITES", "A", [{"from_id": "d1", "to_id": "d2"}], from_label="Document", to_label="Document"
    )
    assert stmt.cypher == (
        "UNWIND $rows AS row "
        "MATCH (a:Document {scope: $scope, id: row.from_id}) "
        "MATCH (b:Document {scope: $scope, id: row.to_id}) "
        "MERGE (a)-[r:CITES]->(b) ON CREATE SET r.created_at = datetime($now) "
        "SET r += row.props"
    )


def test_merge_relationships_values_never_in_cypher():
    scope = "scopeqzx"
    ctx = _ctx(write=(scope,), read=(scope,))
    rows = [{"from_id": "fromqzx", "to_id": "toqzx", "props": {"weight": 4242}}]
    (stmt,) = _builder(ctx).merge_relationships(
        "ABOUT", scope, rows, from_label="Document", to_label="Topic"
    )
    for value in (scope, "fromqzx", "toqzx", "4242"):
        assert value not in stmt.cypher


def test_merge_relationships_statements_pass_checks():
    stmts = _builder(batch_size=2).merge_relationships(
        "ABOUT", "A", _rel_rows(3), from_label="Document", to_label="Topic"
    )
    assert len(stmts) == 2
    for stmt in stmts:
        assert checks.audit_write_problems(stmt.cypher) == []
        checks.require_labeled(stmt.cypher)
        checks.require_audited(stmt.cypher)


# --- merge_relationships: refusals -----------------------------------------------------------


def test_merge_relationships_unwritable_scope_is_denied():
    ctx = _ctx(write=("A",), read=("A", "B"))
    with pytest.raises(AccessDenied, match="B"):
        _builder(ctx).merge_relationships(
            "ABOUT", "B", _rel_rows(1), from_label="Document", to_label="Topic"
        )


def test_merge_relationships_access_checked_before_rel():
    with pytest.raises(AccessDenied):
        _builder().merge_relationships(
            "NO_SUCH", "B", _rel_rows(1), from_label="Document", to_label="Topic"
        )


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
        _builder().merge_relationships(
            rel, "A", _rel_rows(1), from_label=from_label, to_label=to_label
        )


@pytest.mark.parametrize(
    "scopes",
    [
        {"from_scope": "A", "to_scope": "B"},
        {"from_scope": "B", "to_scope": "B"},
        {"from_scope": "B"},
        {"to_scope": "B"},
    ],
)
def test_merge_relationships_cross_scope_is_isolation_error(scopes):
    ctx = _ctx(write=("A", "B"), read=("A", "B"))
    rows = [{"from_id": "d0", "to_id": "t0"}, {"from_id": "d1", "to_id": "t1", **scopes}]
    with pytest.raises(IsolationError):
        _builder(ctx).merge_relationships(
            "ABOUT", "A", rows, from_label="Document", to_label="Topic"
        )


def test_merge_relationships_reserved_prop_is_refused():
    rows = [{"from_id": "d1", "to_id": "t1", "props": {"created_at": "x"}}]
    with pytest.raises(ValueError, match="created_at"):
        _builder().merge_relationships("ABOUT", "A", rows, from_label="Document", to_label="Topic")


def test_merge_relationships_undeclared_prop_is_refused():
    rows = [{"from_id": "d1", "to_id": "t1", "props": {"strengthqzx": 1}}]
    with pytest.raises(ValueError, match="strengthqzx"):
        _builder().merge_relationships("ABOUT", "A", rows, from_label="Document", to_label="Topic")


def test_merge_relationships_prop_of_other_type_is_refused():
    rows = [{"from_id": "d1", "to_id": "d2", "props": {"weight": 1}}]
    with pytest.raises(ValueError, match="weight"):
        _builder().merge_relationships(
            "CITES", "A", rows, from_label="Document", to_label="Document"
        )


# --- audit_set --------------------------------------------------------------------------------


@pytest.mark.parametrize("var", ["x", "n", "node2"])
def test_audit_set(var):
    assert _builder().audit_set(var) == (
        f"{var}.updated_at = datetime($now), {var}.updated_by = $actor"
    )
