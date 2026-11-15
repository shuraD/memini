"""GS-11: the append-only AuditEvent CREATE statement for a delete, restore, purge or wipe.

Spec: GS-11 (src/memini/core/graph/auditlog.py) and SPEC-graph-schema, decision "Audit log:
every delete, restore, purge and scope wipe writes one append-only AuditEvent node (scope,
action, ids or a count, by, at, reason) ... Audit events themselves are never purged".
No database in these tests.
"""

import re
import uuid
from datetime import UTC, datetime

import pytest

from memini.core.graph import checks
from memini.core.graph.auditlog import AUDIT_ACTIONS, AUDIT_LABEL, audit_event
from memini.core.graph.types import AccessContext, AccessDenied, Statement

NOW = datetime(2026, 11, 15, 10, 0, 0, tzinfo=UTC)


def _clock() -> datetime:
    return NOW


def _ctx(
    read: frozenset[str] = frozenset({"A", "B"}),
    write: frozenset[str] = frozenset({"A"}),
    actor: str = "job-run-7",
) -> AccessContext:
    return AccessContext(principal="tester", read_scopes=read, write_scopes=write, actor=actor)


def test_constants() -> None:
    assert AUDIT_ACTIONS == ("delete", "restore", "purge", "wipe_scope")
    assert AUDIT_LABEL == "AuditEvent"


def test_delete_with_ids_builds_statement() -> None:
    ctx = _ctx()
    stmt = audit_event(ctx, _clock, scope="A", action="delete", reason="cleanup", ids=["u1", "u2"])
    assert isinstance(stmt, Statement)
    assert stmt.name == "audit:delete"
    assert stmt.kind == "write"
    assert stmt.scope == "A"
    p = stmt.params
    assert p["ids"] == ["u1", "u2"]
    assert p["count"] == 2
    assert p["actor"] == ctx.actor
    assert p["action"] == "delete"
    assert p["reason"] == "cleanup"
    assert p["scope"] == "A"


def test_ids_param_is_a_list_from_any_sequence() -> None:
    stmt = audit_event(_ctx(), _clock, scope="A", action="delete", reason="r", ids=("u1", "u2"))
    assert stmt.params["ids"] == ["u1", "u2"]
    assert isinstance(stmt.params["ids"], list)


@pytest.mark.parametrize("action", ["delete", "restore", "purge", "wipe_scope"])
def test_cypher_is_create_only(action: str) -> None:
    stmt = audit_event(_ctx(), _clock, scope="A", action=action, reason="r", ids=["u1"])
    assert "CREATE (e:AuditEvent" in stmt.cypher
    assert "MERGE" not in stmt.cypher
    assert "MATCH" not in stmt.cypher
    assert "DELETE" not in stmt.cypher


def test_purge_with_count_only() -> None:
    stmt = audit_event(_ctx(), _clock, scope="A", action="purge", reason="r", count=40)
    assert stmt.params["ids"] is None
    assert stmt.params["count"] == 40
    assert stmt.name == "audit:purge"


def test_explicit_count_wins_over_len_ids() -> None:
    stmt = audit_event(_ctx(), _clock, scope="A", action="delete", reason="r", ids=["u1"], count=5)
    assert stmt.params["count"] == 5
    assert stmt.params["ids"] == ["u1"]


@pytest.mark.parametrize("action", ["delete", "restore", "purge", "wipe_scope"])
def test_each_action_named(action: str) -> None:
    stmt = audit_event(_ctx(), _clock, scope="A", action=action, reason="r", count=1)
    assert stmt.name == f"audit:{action}"
    assert stmt.params["action"] == action
    assert stmt.kind == "write"


def test_unknown_action_rejected() -> None:
    with pytest.raises(ValueError):
        audit_event(_ctx(), _clock, scope="A", action="drop", reason="r", ids=["u1"])


def test_empty_reason_rejected() -> None:
    with pytest.raises(ValueError):
        audit_event(_ctx(), _clock, scope="A", action="delete", reason="", ids=["u1"])


def test_neither_ids_nor_count_rejected() -> None:
    with pytest.raises(ValueError):
        audit_event(_ctx(), _clock, scope="A", action="delete", reason="r")


def test_event_id_fresh_uuid4_hex() -> None:
    a = audit_event(_ctx(), _clock, scope="A", action="delete", reason="r", ids=["u1"])
    b = audit_event(_ctx(), _clock, scope="A", action="delete", reason="r", ids=["u1"])
    ea, eb = a.params["event_id"], b.params["event_id"]
    assert ea != eb
    for eid in (ea, eb):
        assert isinstance(eid, str)
        assert re.fullmatch(r"[0-9a-f]{32}", eid)
        assert uuid.UUID(hex=eid).version == 4


def test_now_from_clock() -> None:
    stmt = audit_event(_ctx(), _clock, scope="A", action="restore", reason="r", ids=["u1"])
    parsed = datetime.fromisoformat(stmt.params["now"])
    assert parsed == datetime.fromisoformat("2026-11-15T10:00:00+00:00")


def test_values_never_in_cypher() -> None:
    scope = "scopeqzx81"
    actor = "actorjkw42"
    ids = ["idvvq901", "idvvq902"]
    reason = "reasonplm77"
    ctx = _ctx(read=frozenset({scope}), write=frozenset({scope}), actor=actor)
    stmt = audit_event(ctx, _clock, scope=scope, action="delete", reason=reason, ids=ids)
    for value in (scope, actor, reason, *ids, stmt.params["event_id"]):
        assert value not in stmt.cypher
    assert stmt.params["scope"] == scope
    assert stmt.params["actor"] == actor
    assert stmt.params["reason"] == reason


@pytest.mark.parametrize("action", ["delete", "restore", "purge", "wipe_scope"])
def test_audit_write_problems_clean(action: str) -> None:
    stmt = audit_event(_ctx(), _clock, scope="A", action=action, reason="r", count=3)
    assert checks.audit_write_problems(stmt.cypher) == []


def test_unwritable_scope_denied_even_when_readable() -> None:
    with pytest.raises(AccessDenied) as info:
        audit_event(_ctx(), _clock, scope="B", action="delete", reason="r", ids=["u1"])
    assert "B" in str(info.value)


def test_access_checked_before_value_errors() -> None:
    with pytest.raises(AccessDenied) as info:
        audit_event(_ctx(), _clock, scope="B", action="drop", reason="")
    assert "B" in str(info.value)
