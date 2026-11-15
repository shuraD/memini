"""GS-14: delete forever the soft-deleted or superseded items of a scope after a waiting period.

Spec: GS-14 (src/memini/core/graph/purge.py) and SPEC-graph-schema, decisions "Delete, step 2:
delete forever", "Audit log" and "Scale: large purges and wipes run in transactions of
batch_size". No database in these tests.
"""

import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from memini.core.graph.purge import purge
from memini.core.graph.session import GraphSession
from memini.core.graph.types import (
    AccessContext,
    AccessDenied,
    ConfirmationRequired,
    RunResult,
    Statement,
)
from memini.core.schema import Node, Property, Schema
from memini.infra.graph.guard import UnsafeDatabaseError

NOW = datetime(2026, 11, 15, 0, 0, 0, tzinfo=UTC)
LOCAL_URI = "bolt://localhost:7687"
REMOTE_HOST = "graph.example.test"
REMOTE_URI = f"bolt://{REMOTE_HOST}:7687"
SCOPE = "scope-a"


def _clock() -> datetime:
    return NOW


def _prop() -> Property:
    return Property(layer="core", writer="ingest.register", readers=("query.*",), rule="a rule")


def _schema() -> Schema:
    return Schema(
        nodes={
            "Unit": Node(rule="a unit", properties={"id": _prop(), "text": _prop()}),
            "Label": Node(rule="no id here", properties={"name": _prop()}),
            "AuditEvent": Node(rule="audit", properties={"id": _prop(), "action": _prop()}),
            "Entity": Node(rule="an entity", properties={"name": _prop(), "id": _prop()}),
        }
    )


def _ctx(write: frozenset[str] = frozenset({SCOPE})) -> AccessContext:
    return AccessContext(
        principal="tester",
        read_scopes=frozenset({SCOPE, "other"}),
        write_scopes=write,
        actor="job-run-7",
    )


class FakeSession:
    """Records every statement; returns configured rows per statement name."""

    def __init__(
        self,
        uri: str = LOCAL_URI,
        ctx: AccessContext | None = None,
        rows: Mapping[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.uri = uri
        self.ctx = ctx if ctx is not None else _ctx()
        self.rows = dict(rows or {})
        self.statements: list[Statement] = []

    def run(self, stmt: Statement) -> list[dict[str, Any]]:
        self.statements.append(stmt)
        return list(self.rows.get(stmt.name, []))


def _purge(session: Any, **overrides: Any) -> int:
    kwargs: dict[str, Any] = {
        "older_than": timedelta(days=30),
        "confirm": SCOPE,
        "reason": "retention",
        "env": {},
    }
    kwargs.update(overrides)
    return purge(session, _schema(), _clock, SCOPE, **kwargs)


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _purge_stmts(session: FakeSession) -> list[Statement]:
    return [s for s in session.statements if s.name.startswith("purge:")]


# --- refusals ---------------------------------------------------------------------------


@pytest.mark.parametrize("confirm", [None, "scope-b", "Scope-A", " scope-a", ""])
def test_wrong_confirmation_refused_without_running(confirm: str | None) -> None:
    session = FakeSession()
    with pytest.raises(ConfirmationRequired):
        _purge(session, confirm=confirm)
    assert session.statements == []


def test_confirmation_is_checked_before_unsafe_host() -> None:
    session = FakeSession(uri=REMOTE_URI, ctx=_ctx(write=frozenset()))
    with pytest.raises(ConfirmationRequired):
        _purge(session, confirm=None, env={})
    assert session.statements == []


def test_remote_host_without_allow_list_refused() -> None:
    session = FakeSession(uri=REMOTE_URI)
    with pytest.raises(UnsafeDatabaseError):
        _purge(session, env={})
    assert session.statements == []


def test_unsafe_host_checked_before_write_access() -> None:
    session = FakeSession(uri=REMOTE_URI, ctx=_ctx(write=frozenset()))
    with pytest.raises(UnsafeDatabaseError):
        _purge(session, env={})
    assert session.statements == []


def test_allow_listed_remote_host_runs() -> None:
    session = FakeSession(uri=REMOTE_URI)
    total = _purge(session, env={"MEMINI_NEO4J_ALLOWED_HOSTS": REMOTE_HOST})
    assert total == 0
    assert [s.name for s in session.statements] == [
        "purge:Unit",
        "purge:Entity",
        "audit:purge",
    ]


def test_production_host_refused_even_when_allow_listed() -> None:
    session = FakeSession(uri=REMOTE_URI)
    env = {"MEMINI_NEO4J_ALLOWED_HOSTS": REMOTE_HOST, "MEMINI_PROD_NEO4J_HOST": REMOTE_HOST}
    with pytest.raises(UnsafeDatabaseError):
        _purge(session, env=env)
    assert session.statements == []


def test_no_write_access_refused_without_running() -> None:
    session = FakeSession(ctx=_ctx(write=frozenset()))
    with pytest.raises(AccessDenied):
        _purge(session)
    assert session.statements == []


def test_write_access_checked_before_argument_values() -> None:
    session = FakeSession(ctx=_ctx(write=frozenset()))
    with pytest.raises(AccessDenied):
        _purge(session, batch_size=0)
    assert session.statements == []


def test_negative_older_than_refused() -> None:
    session = FakeSession()
    with pytest.raises(ValueError):
        _purge(session, older_than=timedelta(days=-1))
    assert session.statements == []


@pytest.mark.parametrize("batch_size", [0, -5])
def test_batch_size_below_one_refused(batch_size: int) -> None:
    session = FakeSession()
    with pytest.raises(ValueError):
        _purge(session, batch_size=batch_size)
    assert session.statements == []


def test_older_than_zero_accepted() -> None:
    session = FakeSession()
    assert _purge(session, older_than=timedelta(0)) == 0
    stmts = _purge_stmts(session)
    assert len(stmts) == 2
    assert datetime.fromisoformat(stmts[0].params["cutoff"]) == NOW


# --- statements -------------------------------------------------------------------------


def test_statement_names_and_order() -> None:
    session = FakeSession()
    _purge(session)
    names = [s.name for s in session.statements]
    assert names == ["purge:Unit", "purge:Entity", "audit:purge"]
    assert "purge:AuditEvent" not in names
    assert "purge:Label" not in names


def test_every_purge_statement_shape() -> None:
    session = FakeSession()
    _purge(session)
    stmts = _purge_stmts(session)
    assert len(stmts) == 2
    for stmt, label in zip(stmts, ["Unit", "Entity"], strict=True):
        assert isinstance(stmt, Statement)
        assert stmt.kind == "admin"
        assert stmt.scope == SCOPE
        flat = _squash(stmt.cypher)
        assert _squash(f"MATCH (n:{label})") in flat
        assert _squash("n.scope = $scope") in flat
        assert _squash("n.status = 'deleted' AND n.deleted_at <= datetime($cutoff)") in flat
        assert _squash("n.status = 'superseded' AND n.updated_at <= datetime($cutoff)") in flat
        assert _squash("CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF $batch_size ROWS") in (
            flat
        )
        assert _squash("RETURN count(n) AS removed") in flat
        assert "active" not in stmt.cypher
        assert SCOPE not in stmt.cypher
        assert stmt.params["scope"] == SCOPE


def test_cutoff_scope_and_batch_size_params() -> None:
    session = FakeSession()
    _purge(session, older_than=timedelta(days=30), batch_size=250)
    for stmt in _purge_stmts(session):
        cutoff = datetime.fromisoformat(stmt.params["cutoff"])
        assert cutoff == datetime(2026, 10, 16, 0, 0, 0, tzinfo=UTC)
        assert stmt.params["scope"] == SCOPE
        assert stmt.params["batch_size"] == 250


def test_default_batch_size_is_1000() -> None:
    session = FakeSession()
    _purge(session)
    for stmt in _purge_stmts(session):
        assert stmt.params["batch_size"] == 1000


def test_returns_total_and_audits_count() -> None:
    session = FakeSession(
        rows={"purge:Unit": [{"removed": 3}], "purge:Entity": [{"removed": 4}]},
    )
    total = _purge(session, reason="retention 30 days")
    assert total == 7
    last = session.statements[-1]
    assert last.name == "audit:purge"
    assert last.kind == "write"
    assert last.scope == SCOPE
    assert last.params["count"] == 7
    assert last.params["ids"] is None
    assert last.params["action"] == "purge"
    assert last.params["reason"] == "retention 30 days"
    assert last.params["actor"] == "job-run-7"
    assert datetime.fromisoformat(last.params["now"]) == NOW


def test_removed_is_summed_over_rows() -> None:
    session = FakeSession(
        rows={
            "purge:Unit": [{"removed": 1}, {"removed": 2}],
            "purge:Entity": [{"removed": 5}],
        },
    )
    assert _purge(session) == 8
    assert session.statements[-1].params["count"] == 8


def test_no_rows_returns_zero_and_still_audits() -> None:
    session = FakeSession()
    assert _purge(session) == 0
    audits = [s for s in session.statements if s.name == "audit:purge"]
    assert len(audits) == 1
    assert audits[0] is session.statements[-1]
    assert audits[0].params["count"] == 0


def test_ids_restrict_every_purge_statement() -> None:
    session = FakeSession()
    _purge(session, ids=["x"])
    stmts = _purge_stmts(session)
    assert len(stmts) == 2
    for stmt in stmts:
        assert _squash("AND n.id IN $ids") in _squash(stmt.cypher)
        assert stmt.params["ids"] == ["x"]
        assert isinstance(stmt.params["ids"], list)
        assert "'x'" not in stmt.cypher
    assert session.statements[-1].params["ids"] == ["x"]


def test_without_ids_no_ids_filter() -> None:
    session = FakeSession()
    _purge(session)
    for stmt in _purge_stmts(session):
        assert "$ids" not in stmt.cypher
        assert "ids" not in stmt.params


# --- real GraphSession ------------------------------------------------------------------


class FakeRunner:
    """Records each call and returns one removed row per purge statement."""

    def __init__(self, uri: str = LOCAL_URI) -> None:
        self.uri = uri
        self.calls: list[tuple[str, dict[str, Any], bool]] = []

    def run(
        self,
        cypher: str,
        params: Mapping[str, Any],
        *,
        profile: bool = False,
        autocommit: bool = False,
    ) -> RunResult:
        self.calls.append((cypher, dict(params), autocommit))
        if "DETACH DELETE" in cypher:
            return RunResult(rows=[{"removed": 2}])
        return RunResult(rows=[])


def test_real_session_autocommit_for_purge_and_not_for_audit() -> None:
    runner = FakeRunner()
    session = GraphSession(runner, _ctx())
    total = purge(
        session,
        _schema(),
        _clock,
        SCOPE,
        older_than=timedelta(days=30),
        confirm=SCOPE,
        reason="retention",
        env={},
    )
    assert total == 4
    assert len(runner.calls) == 3
    purge_calls = [c for c in runner.calls if "DETACH DELETE" in c[0]]
    assert len(purge_calls) == 2
    assert all(autocommit is True for _, _, autocommit in purge_calls)
    audit_cypher, audit_params, audit_autocommit = runner.calls[-1]
    assert "AuditEvent" in audit_cypher
    assert audit_autocommit is False
    assert audit_params["count"] == 4
