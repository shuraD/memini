"""GS-12: soft delete and restore of the items of a scope, each followed by one AuditEvent.

Spec: GS-12 (src/memini/core/graph/lifecycle.py) and SPEC-graph-schema, decisions
"Delete, step 1: soft delete ... restore(scope, ids) brings an item back" and "Audit log:
every delete, restore, purge and scope wipe writes one append-only AuditEvent node".
No database in these tests: a fake runner records what the session sends.
"""

import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from memini.core import schema
from memini.core.graph.lifecycle import delete, restore
from memini.core.graph.session import GraphSession
from memini.core.graph.types import AccessContext, AccessDenied, RunResult, Statement

NOW = datetime(2026, 11, 15, 10, 0, 0, tzinfo=UTC)
SCOPE = "scope-zeta"
ACTOR = "actor-job-77"
REASON = "reason-cleanup-xyz"
IDS = ("doc-id-111", "unit-id-222")

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


def _schema(with_audit: bool = False) -> schema.Schema:
    nodes: dict[str, Any] = {
        # Declared out of sorted order on purpose.
        "Unit": {"rule": "a chunk", "properties": {"id": _prop(), "text": _prop()}},
        "Domain": {"rule": "a field", "properties": {"name": _prop()}},
        "Document": {"rule": "a source file", "properties": {"id": _prop(), "title": _prop()}},
    }
    if with_audit:
        nodes["AuditEvent"] = {
            "rule": "an audit record",
            "properties": {"id": _prop(), "action": _prop()},
        }
    return schema.from_dict(
        {
            "common": {
                "nodes": {name: _prop(name) for name in COMMON_NODE_FIELDS},
                "relationships": {"created_at": _prop("when linked")},
            },
            "nodes": nodes,
            "relationships": {},
        }
    )


class FakeRunner:
    """Records each call; returns the rows configured for the label the cypher matches."""

    def __init__(self, rows_by_label: Mapping[str, list[dict[str, Any]]] | None = None) -> None:
        self.uri = "bolt://localhost:7687"
        self.rows_by_label = dict(rows_by_label or {})
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        cypher: str,
        params: Mapping[str, Any],
        *,
        profile: bool = False,
        autocommit: bool = False,
    ) -> RunResult:
        self.calls.append({"cypher": cypher, "params": dict(params)})
        for label, rows in self.rows_by_label.items():
            if f"MATCH (n:{label} " in cypher:
                return RunResult(rows=[dict(r) for r in rows])
        return RunResult(rows=[])


class RecordingSession(GraphSession):
    """A real session that also keeps every statement it was asked to run."""

    def __init__(self, runner: FakeRunner, ctx: AccessContext | None) -> None:
        super().__init__(runner, ctx)
        self.statements: list[Statement] = []

    def run(self, stmt: Statement) -> list[dict[str, Any]]:
        self.statements.append(stmt)
        return super().run(stmt)


def _ctx(read: set[str] | None = None, write: set[str] | None = None) -> AccessContext:
    write_scopes = frozenset({SCOPE} if write is None else write)
    read_scopes = frozenset({SCOPE} if read is None else read) | write_scopes
    return AccessContext(
        principal="principal-1", read_scopes=read_scopes, write_scopes=write_scopes, actor=ACTOR
    )


def _clock() -> datetime:
    return NOW


def _session(
    rows: Mapping[str, list[dict[str, Any]]] | None = None,
    ctx: AccessContext | None = None,
    no_ctx: bool = False,
) -> tuple[RecordingSession, FakeRunner]:
    runner = FakeRunner(rows)
    session = RecordingSession(runner, None if no_ctx else (ctx or _ctx()))
    return session, runner


OPS = [pytest.param(delete, "delete", id="delete"), pytest.param(restore, "restore", id="restore")]


# --- statement sequence -------------------------------------------------------------


@pytest.mark.parametrize(("op", "action"), OPS)
def test_one_statement_per_id_label_in_sorted_order_then_audit(op: Any, action: str) -> None:
    session, runner = _session()
    op(session, _schema(), _clock, SCOPE, list(IDS), reason=REASON)
    names = [s.name for s in session.statements]
    assert names == [f"{action}:Document", f"{action}:Unit", f"audit:{action}"]
    assert len(runner.calls) == 3
    assert not any("Domain" in s.cypher for s in session.statements)


@pytest.mark.parametrize(("op", "action"), OPS)
def test_audit_label_is_never_a_target(op: Any, action: str) -> None:
    session, _ = _session()
    op(session, _schema(with_audit=True), _clock, SCOPE, list(IDS), reason=REASON)
    names = [s.name for s in session.statements]
    assert f"{action}:AuditEvent" not in names
    assert names == [f"{action}:Document", f"{action}:Unit", f"audit:{action}"]
    for stmt in session.statements[:-1]:
        assert "AuditEvent" not in stmt.cypher


@pytest.mark.parametrize(("op", "action"), OPS)
def test_label_statements_are_scoped_writes(op: Any, action: str) -> None:
    session, _ = _session()
    op(session, _schema(), _clock, SCOPE, list(IDS), reason=REASON)
    for stmt in session.statements:
        assert stmt.kind == "write"
        assert stmt.scope == SCOPE


# --- delete cypher and params -------------------------------------------------------


@pytest.mark.parametrize("label", ["Document", "Unit"])
def test_delete_cypher_contains_required_clauses(label: str) -> None:
    session, _ = _session()
    delete(session, _schema(), _clock, SCOPE, list(IDS), reason=REASON)
    stmt = next(s for s in session.statements if s.name == f"delete:{label}")
    for fragment in (
        f"MATCH (n:{label} {{scope: $scope, id: id}})",
        "WHERE n.status = 'active'",
        "n.status = 'deleted'",
        "n.deleted_at = datetime($now)",
        "n.deleted_by = $actor",
        "n.updated_at = datetime($now)",
        "n.updated_by = $actor",
        "RETURN count(n) AS n",
    ):
        assert fragment in stmt.cypher, fragment


def test_delete_cypher_matches_interface_exactly() -> None:
    session, _ = _session()
    delete(session, _schema(), _clock, SCOPE, list(IDS), reason=REASON)
    stmt = session.statements[0]
    assert stmt.cypher == (
        "UNWIND $ids AS id MATCH (n:Document {scope: $scope, id: id}) "
        "WHERE n.status = 'active' "
        "SET n.status = 'deleted', n.deleted_at = datetime($now), n.deleted_by = $actor, "
        "n.updated_at = datetime($now), n.updated_by = $actor RETURN count(n) AS n"
    )


@pytest.mark.parametrize(("op", "action"), OPS)
def test_label_statement_params(op: Any, action: str) -> None:
    session, _ = _session()
    op(session, _schema(), _clock, SCOPE, IDS, reason=REASON)
    for stmt in session.statements[:-1]:
        assert dict(stmt.params) == {
            "scope": SCOPE,
            "ids": list(IDS),
            "now": NOW.isoformat(),
            "actor": ACTOR,
        }
        assert isinstance(stmt.params["ids"], list)


# --- restore cypher -----------------------------------------------------------------


@pytest.mark.parametrize("label", ["Document", "Unit"])
def test_restore_cypher_contains_required_clauses(label: str) -> None:
    session, _ = _session()
    restore(session, _schema(), _clock, SCOPE, list(IDS), reason=REASON)
    stmt = next(s for s in session.statements if s.name == f"restore:{label}")
    for fragment in (
        f"MATCH (n:{label} {{scope: $scope, id: id}})",
        "WHERE n.status = 'deleted'",
        "n.status = 'active'",
        "n.deleted_at = null",
        "n.deleted_by = null",
        "n.updated_at = datetime($now)",
        "n.updated_by = $actor",
        "RETURN count(n) AS n",
    ):
        assert fragment in stmt.cypher, fragment
    assert "WHERE n.status = 'active'" not in stmt.cypher


def test_restore_cypher_matches_interface_exactly() -> None:
    session, _ = _session()
    restore(session, _schema(), _clock, SCOPE, list(IDS), reason=REASON)
    stmt = session.statements[1]
    assert stmt.cypher == (
        "UNWIND $ids AS id MATCH (n:Unit {scope: $scope, id: id}) "
        "WHERE n.status = 'deleted' "
        "SET n.status = 'active', n.deleted_at = null, n.deleted_by = null, "
        "n.updated_at = datetime($now), n.updated_by = $actor RETURN count(n) AS n"
    )


# --- counts and the audit event -----------------------------------------------------


def test_delete_returns_summed_count_and_audits_it() -> None:
    session, runner = _session({"Document": [{"n": 2}], "Unit": [{"n": 0}]})
    result = delete(session, _schema(), _clock, SCOPE, list(IDS), reason=REASON)
    assert result == 2
    audit = session.statements[-1]
    assert audit.name == "audit:delete"
    assert "CREATE (e:AuditEvent" in audit.cypher
    assert audit.params["action"] == "delete"
    assert audit.params["count"] == 2
    assert list(audit.params["ids"]) == list(IDS)
    assert audit.params["reason"] == REASON
    assert audit.params["scope"] == SCOPE
    assert audit.params["actor"] == ACTOR
    assert runner.calls[-1]["cypher"] == audit.cypher


def test_restore_returns_summed_count_and_audits_it() -> None:
    session, _ = _session({"Document": [{"n": 1}], "Unit": [{"n": 3}]})
    result = restore(session, _schema(), _clock, SCOPE, list(IDS), reason=REASON)
    assert result == 4
    audit = session.statements[-1]
    assert audit.name == "audit:restore"
    assert audit.params["action"] == "restore"
    assert audit.params["count"] == 4
    assert list(audit.params["ids"]) == list(IDS)
    assert audit.params["reason"] == REASON
    assert audit.params["scope"] == SCOPE


@pytest.mark.parametrize(("op", "action"), OPS)
def test_statement_returning_no_rows_counts_zero(op: Any, action: str) -> None:
    session, _ = _session({"Document": [], "Unit": [{"n": 3}]})
    assert op(session, _schema(), _clock, SCOPE, list(IDS), reason=REASON) == 3
    assert session.statements[-1].params["count"] == 3


@pytest.mark.parametrize(("op", "action"), OPS)
def test_zero_changes_still_writes_one_audit_event(op: Any, action: str) -> None:
    session, _ = _session({"Document": [{"n": 0}], "Unit": [{"n": 0}]})
    assert op(session, _schema(), _clock, SCOPE, list(IDS), reason=REASON) == 0
    audits = [s for s in session.statements if s.name.startswith("audit:")]
    assert len(audits) == 1
    assert audits[0] is session.statements[-1]
    assert audits[0].name == f"audit:{action}"
    assert audits[0].params["count"] == 0


# --- nothing is ever removed, no value in the text ----------------------------------


@pytest.mark.parametrize(("op", "action"), OPS)
def test_no_delete_or_remove_clause(op: Any, action: str) -> None:
    session, _ = _session({"Document": [{"n": 1}], "Unit": [{"n": 1}]})
    op(session, _schema(with_audit=True), _clock, SCOPE, list(IDS), reason=REASON)
    assert session.statements
    for stmt in session.statements:
        assert not re.search(r"\bDELETE\b", stmt.cypher, re.IGNORECASE), stmt.cypher
        assert not re.search(r"\bREMOVE\b", stmt.cypher, re.IGNORECASE), stmt.cypher


@pytest.mark.parametrize(("op", "action"), OPS)
def test_no_value_in_cypher_text(op: Any, action: str) -> None:
    session, runner = _session()
    op(session, _schema(), _clock, SCOPE, list(IDS), reason=REASON)
    texts = [s.cypher for s in session.statements] + [c["cypher"] for c in runner.calls]
    for text in texts:
        for value in (SCOPE, *IDS, REASON, ACTOR):
            assert value not in text, (value, text)


# --- time -----------------------------------------------------------------------------


@pytest.mark.parametrize(("op", "action"), OPS)
def test_now_parses_to_clock_instant(op: Any, action: str) -> None:
    session, _ = _session()
    op(session, _schema(), _clock, SCOPE, list(IDS), reason=REASON)
    for stmt in session.statements[:-1]:
        assert datetime.fromisoformat(stmt.params["now"]) == NOW


@pytest.mark.parametrize(("op", "action"), OPS)
def test_now_is_the_same_in_every_label_statement(op: Any, action: str) -> None:
    ticks = iter(NOW + timedelta(seconds=i) for i in range(100))

    def ticking_clock() -> datetime:
        return next(ticks)

    session, _ = _session()
    op(session, _schema(), ticking_clock, SCOPE, list(IDS), reason=REASON)
    label_nows = {stmt.params["now"] for stmt in session.statements[:-1]}
    assert len(label_nows) == 1


# --- refusals before anything runs ----------------------------------------------------


@pytest.mark.parametrize(("op", "action"), OPS)
def test_readable_but_not_writable_scope_is_denied(op: Any, action: str) -> None:
    ctx = _ctx(read={"A", "B"}, write={"A"})
    session, runner = _session(ctx=ctx)
    with pytest.raises(AccessDenied) as err:
        op(session, _schema(), _clock, "B", list(IDS), reason=REASON)
    assert "B" in str(err.value)
    assert runner.calls == []


@pytest.mark.parametrize(("op", "action"), OPS)
def test_missing_access_context_is_denied(op: Any, action: str) -> None:
    session, runner = _session(no_ctx=True)
    with pytest.raises(AccessDenied):
        op(session, _schema(), _clock, SCOPE, list(IDS), reason=REASON)
    assert runner.calls == []


@pytest.mark.parametrize(("op", "action"), OPS)
@pytest.mark.parametrize(
    ("ids", "reason"),
    [
        pytest.param([], REASON, id="empty-ids"),
        pytest.param(["u1", ""], REASON, id="empty-id"),
        pytest.param(["u1", 7], REASON, id="non-str-id"),
        pytest.param(["u1"], "", id="empty-reason"),
    ],
)
def test_invalid_input_raises_value_error(op: Any, action: str, ids: list, reason: str) -> None:
    session, runner = _session()
    with pytest.raises(ValueError):
        op(session, _schema(), _clock, SCOPE, ids, reason=reason)
    assert runner.calls == []


@pytest.mark.parametrize(("op", "action"), OPS)
def test_missing_context_is_checked_before_input(op: Any, action: str) -> None:
    session, runner = _session(no_ctx=True)
    with pytest.raises(AccessDenied):
        op(session, _schema(), _clock, SCOPE, [], reason="")
    assert runner.calls == []


@pytest.mark.parametrize(("op", "action"), OPS)
def test_write_access_is_checked_before_input(op: Any, action: str) -> None:
    session, runner = _session(ctx=_ctx(read={"A", "B"}, write={"A"}))
    with pytest.raises(AccessDenied):
        op(session, _schema(), _clock, "B", [], reason="")
    assert runner.calls == []
