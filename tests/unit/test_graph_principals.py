"""Acceptance tests for GS-24: grant and load_access of memini.core.graph.principals."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from memini.core.graph import access, checks
from memini.core.graph.principals import ACCESS, PRINCIPAL_KINDS, grant, load_access
from memini.core.graph.session import GraphSession
from memini.core.graph.types import AccessContext, AccessDenied, RunResult

URI = "bolt://localhost:7687"
NOW = datetime(2026, 11, 15, 10, 0, 0, tzinfo=UTC)
ACTOR = "job:test-run"

GRANT_FRAGMENTS = [
    "MERGE (p:Principal {id: $principal})",
    "ON CREATE SET p.created_at = datetime($now), p.status = 'active'",
    "SET p.kind = $kind, p.updated_at = datetime($now), p.updated_by = $actor",
    "MERGE (s:Scope {id: $scope})",
    "ON CREATE SET s.created_at = datetime($now), s.status = 'active'",
    "SET s.updated_at = datetime($now), s.updated_by = $actor",
    "MERGE (p)-[r:CAN_ACCESS]->(s)",
    "ON CREATE SET r.created_at = datetime($now)",
    "SET r.access = $access",
]

LOAD_CYPHER = (
    "MATCH (p:Principal {id: $principal})-[r:CAN_ACCESS]->(s:Scope) "
    "RETURN s.id AS scope, r.access AS access"
)


class FakeRunner:
    """Records every call and returns fixed rows."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.uri = URI
        self.rows = [] if rows is None else rows
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        cypher: str,
        params: Mapping[str, Any],
        *,
        profile: bool = False,
        autocommit: bool = False,
    ) -> RunResult:
        self.calls.append(
            {
                "cypher": cypher,
                "params": dict(params),
                "profile": profile,
                "autocommit": autocommit,
            }
        )
        return RunResult(rows=list(self.rows))


class FakeRecorder:
    """Records every performance sample."""

    def __init__(self) -> None:
        self.samples: list[tuple[str, str | None]] = []

    def record(
        self, name: str, scope: str | None, seconds: float, rows: int, db_hits: int | None
    ) -> None:
        self.samples.append((name, scope))


def clock() -> datetime:
    return NOW


def make_ctx(read: set[str], write: set[str]) -> AccessContext:
    return AccessContext(
        principal="admin",
        read_scopes=frozenset(read),
        write_scopes=frozenset(write),
        actor=ACTOR,
    )


def make_session(
    ctx: AccessContext | None, recorder: FakeRecorder | None = None
) -> tuple[GraphSession, FakeRunner]:
    runner = FakeRunner()
    return GraphSession(runner, ctx, recorder=recorder), runner


def writer_session(
    recorder: FakeRecorder | None = None,
) -> tuple[GraphSession, FakeRunner]:
    return make_session(make_ctx({"A"}, {"A"}), recorder)


# ---------------------------------------------------------------- constants


def test_constants() -> None:
    assert ACCESS == ("read", "write")
    assert PRINCIPAL_KINDS == ("user", "service")


# ---------------------------------------------------------------- grant


def test_grant_runs_once_with_all_fragments_and_params() -> None:
    session, runner = writer_session()
    result = grant(session, clock, principal="u1", kind="user", scope="A", access="write")
    assert result is None
    assert len(runner.calls) == 1
    call = runner.calls[0]
    for fragment in GRANT_FRAGMENTS:
        assert fragment in call["cypher"], fragment
    assert call["params"] == {
        "principal": "u1",
        "kind": "user",
        "scope": "A",
        "access": "write",
        "now": NOW.isoformat(),
        "actor": ACTOR,
    }


def test_grant_fragments_in_order() -> None:
    session, runner = writer_session()
    grant(session, clock, principal="u1", kind="user", scope="A", access="write")
    cypher = runner.calls[0]["cypher"]
    pos = 0
    for fragment in GRANT_FRAGMENTS:
        idx = cypher.find(fragment, pos)
        assert idx >= 0, f"fragment out of order or missing: {fragment}"
        pos = idx + len(fragment)


def test_grant_statement_name_scope_and_write_kind() -> None:
    recorder = FakeRecorder()
    session, runner = writer_session(recorder)
    grant(session, clock, principal="u1", kind="service", scope="A", access="write")
    assert recorder.samples == [("grant:CAN_ACCESS", "A")]
    assert runner.calls[0]["autocommit"] is False


def test_grant_cypher_passes_audit_checks() -> None:
    session, runner = writer_session()
    grant(session, clock, principal="u1", kind="user", scope="A", access="write")
    cypher = runner.calls[0]["cypher"]
    assert checks.audit_write_problems(cypher) == []
    checks.require_audited(cypher)
    checks.require_labeled(cypher)


def test_grant_read_access() -> None:
    session, runner = writer_session()
    grant(session, clock, principal="u1", kind="user", scope="A", access="read")
    call = runner.calls[0]
    assert call["params"]["access"] == "read"
    assert "SET r.access = $access" in call["cypher"]


def test_grant_again_uses_merge_not_create() -> None:
    session, runner = writer_session()
    grant(session, clock, principal="u1", kind="user", scope="A", access="read")
    grant(session, clock, principal="u1", kind="user", scope="A", access="write")
    assert len(runner.calls) == 2
    for call in runner.calls:
        assert "CREATE (" not in call["cypher"].replace("ON CREATE", "")
        assert "MERGE (p)-[r:CAN_ACCESS]->(s)" in call["cypher"]
    assert runner.calls[1]["params"]["access"] == "write"


def test_grant_values_not_in_cypher() -> None:
    principal = "principal-xyz-42"
    scope = "scope-qrs-77"
    ctx = make_ctx({scope}, {scope})
    session, runner = make_session(ctx)
    grant(session, clock, principal=principal, kind="service", scope=scope, access="write")
    cypher = runner.calls[0]["cypher"]
    assert principal not in cypher
    assert scope not in cypher
    assert "'write'" not in cypher
    assert '"write"' not in cypher
    assert "'service'" not in cypher


@pytest.mark.parametrize(
    ("principal", "kind", "scope", "access_value"),
    [
        ("u1", "user", "A", "admin"),
        ("u1", "robot", "A", "write"),
        ("", "user", "A", "write"),
        ("u1", "user", "", "write"),
    ],
)
def test_grant_invalid_arguments_raise_value_error(
    principal: str, kind: str, scope: str, access_value: str
) -> None:
    session, runner = writer_session()
    with pytest.raises(ValueError):
        grant(session, clock, principal=principal, kind=kind, scope=scope, access=access_value)
    assert runner.calls == []


def test_grant_value_error_checked_before_access() -> None:
    session, runner = make_session(None)
    with pytest.raises(ValueError):
        grant(session, clock, principal="u1", kind="user", scope="A", access="admin")
    assert runner.calls == []


@pytest.mark.parametrize(
    ("read", "write"),
    [({"A"}, set()), ({"B"}, {"B"}), (set(), set())],
)
def test_grant_without_write_access_is_denied(read: set[str], write: set[str]) -> None:
    session, runner = make_session(make_ctx(read, write))
    with pytest.raises(AccessDenied):
        grant(session, clock, principal="u1", kind="user", scope="A", access="read")
    assert runner.calls == []


def test_grant_without_context_is_denied() -> None:
    session, runner = make_session(None)
    with pytest.raises(AccessDenied):
        grant(session, clock, principal="u1", kind="user", scope="A", access="write")
    assert runner.calls == []


# ---------------------------------------------------------------- load_access


def test_load_access_builds_context() -> None:
    runner = FakeRunner([{"scope": "A", "access": "read"}, {"scope": "B", "access": "write"}])
    ctx = load_access(runner, "u1", actor="api:u1")
    assert isinstance(ctx, AccessContext)
    assert ctx == AccessContext(
        principal="u1",
        read_scopes=frozenset({"A", "B"}),
        write_scopes=frozenset({"B"}),
        actor="api:u1",
    )
    assert ctx.read_scopes == frozenset({"A", "B"})
    assert ctx.write_scopes == frozenset({"B"})
    assert isinstance(ctx.read_scopes, frozenset)
    assert isinstance(ctx.write_scopes, frozenset)


def test_load_access_ignores_unknown_access_value() -> None:
    runner = FakeRunner([{"scope": "A", "access": "read"}, {"scope": "C", "access": "owner"}])
    ctx = load_access(runner, "u1", actor="api:u1")
    assert "C" not in ctx.read_scopes
    assert "C" not in ctx.write_scopes
    assert ctx.read_scopes == frozenset({"A"})
    assert ctx.write_scopes == frozenset()


def test_load_access_no_rows_refuses_everything() -> None:
    runner = FakeRunner([])
    ctx = load_access(runner, "u1", actor="api:u1")
    assert ctx.principal == "u1"
    assert ctx.actor == "api:u1"
    assert ctx.read_scopes == frozenset()
    assert ctx.write_scopes == frozenset()
    with pytest.raises(AccessDenied):
        access.read_filter(ctx, "n")
    with pytest.raises(AccessDenied):
        access.require_write(ctx, "A")


def test_load_access_calls_runner_once_with_exact_query() -> None:
    principal = "u1"
    runner = FakeRunner([{"scope": "A", "access": "write"}])
    load_access(runner, principal, actor="api:u1")
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["cypher"] == LOAD_CYPHER
    assert call["params"] == {"principal": "u1"}
    assert call["profile"] is False
    assert call["autocommit"] is False
    assert principal not in call["cypher"]


def test_load_access_value_not_in_cypher() -> None:
    runner = FakeRunner([])
    load_access(runner, "principal-xyz-42", actor="api:u1")
    assert "principal-xyz-42" not in runner.calls[0]["cypher"]


@pytest.mark.parametrize(("principal", "actor"), [("", "api:u1"), ("u1", "")])
def test_load_access_empty_arguments_raise_value_error(principal: str, actor: str) -> None:
    runner = FakeRunner([{"scope": "A", "access": "write"}])
    with pytest.raises(ValueError):
        load_access(runner, principal, actor=actor)
    assert runner.calls == []
