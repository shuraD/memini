"""Acceptance tests for GS-15: wipe_scope of memini.core.graph.wipe.

Spec: GS-15 (src/memini/core/graph/wipe.py) and SPEC-graph-schema, decisions "Delete, step 2:
delete forever ... wipe_scope(scope, confirm=...) removes a whole scope under the same rules",
"Audit log: ... A purge or wipe keeps its audit event and counts" and "Scale: large purges and
wipes run in transactions of batch_size". No database in these tests.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from memini.core.graph import AccessContext, AccessDenied, ConfirmationRequired, RunResult
from memini.core.graph.session import GraphSession
from memini.core.graph.wipe import wipe_scope
from memini.core.schema import Node, Schema
from memini.infra.graph.guard import UnsafeDatabaseError

LOCAL_URI = "bolt://localhost:7687"
REMOTE_URI = "bolt://db.example.com:7687"
SCOPE = "scope-zeta-91"
NOW = datetime(2026, 11, 15, 10, 0, 0, tzinfo=UTC)
ALL_LABELS = ("Document", "Unit", "AuditEvent", "Scope", "Principal")
_LABEL_RE = re.compile(r"MATCH \(n:(\w+)\)")


def _clock() -> datetime:
    return NOW


class FakeRunner:
    """Records every call; a wipe of label L returns the rows configured for L."""

    def __init__(
        self,
        uri: str = LOCAL_URI,
        rows_by_label: Mapping[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.uri = uri
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
        self.calls.append(
            {"cypher": cypher, "params": dict(params), "profile": profile, "autocommit": autocommit}
        )
        match = _LABEL_RE.search(cypher)
        if match is not None and "DETACH DELETE" in cypher:
            rows = self.rows_by_label.get(match.group(1), [{"count": 0}])
            return RunResult(rows=[dict(r) for r in rows])
        return RunResult(rows=[])


class Recorder:
    """Collects the names of the statements a session ran."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def record(
        self, name: str, scope: str | None, seconds: float, rows: int, db_hits: int | None
    ) -> None:
        self.names.append(name)


def _schema(labels: Sequence[str] = ALL_LABELS) -> Schema:
    return Schema(nodes={label: Node(rule="test", properties={}) for label in labels})


def _ctx(
    read: frozenset[str] = frozenset({SCOPE}),
    write: frozenset[str] = frozenset({SCOPE}),
    actor: str = "job-run-7",
) -> AccessContext:
    return AccessContext(principal="tester", read_scopes=read, write_scopes=write, actor=actor)


def _session(
    runner: FakeRunner, ctx: AccessContext | None = None, recorder: Recorder | None = None
) -> GraphSession:
    return GraphSession(runner, ctx if ctx is not None else _ctx(), recorder=recorder)


def _wipe(
    session: GraphSession,
    schema: Schema | None = None,
    *,
    scope: str = SCOPE,
    confirm: str | None = SCOPE,
    reason: str = "owner asked to remove the scope",
    env: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> int:
    return wipe_scope(
        session,
        schema if schema is not None else _schema(),
        _clock,
        scope,
        confirm=confirm,
        reason=reason,
        env={} if env is None else env,
        **kwargs,
    )


def _wipe_calls(runner: FakeRunner) -> list[dict[str, Any]]:
    return [c for c in runner.calls if "DETACH DELETE" in c["cypher"]]


def _wiped_labels(runner: FakeRunner) -> list[str]:
    labels = []
    for call in _wipe_calls(runner):
        match = _LABEL_RE.search(call["cypher"])
        assert match is not None
        labels.append(match.group(1))
    return labels


# --- checks before any run -------------------------------------------------------------


@pytest.mark.parametrize("confirm", [None, "", "scope-zeta-9", SCOPE.upper(), f" {SCOPE}"])
def test_wrong_confirmation_refused_before_runner(confirm: str | None) -> None:
    runner = FakeRunner()
    with pytest.raises(ConfirmationRequired):
        _wipe(_session(runner), confirm=confirm)
    assert runner.calls == []


def test_confirmation_is_case_sensitive() -> None:
    runner = FakeRunner()
    ctx = _ctx(read=frozenset({"A"}), write=frozenset({"A"}))
    with pytest.raises(ConfirmationRequired):
        _wipe(_session(runner, ctx), scope="A", confirm="a")
    assert runner.calls == []


def test_remote_host_not_allow_listed_refused() -> None:
    runner = FakeRunner(uri=REMOTE_URI)
    with pytest.raises(UnsafeDatabaseError):
        _wipe(_session(runner), env={})
    assert runner.calls == []


def test_remote_host_allow_listed_runs() -> None:
    runner = FakeRunner(uri=REMOTE_URI)
    total = _wipe(_session(runner), env={"MEMINI_NEO4J_ALLOWED_HOSTS": "db.example.com"})
    assert total == 0
    assert len(runner.calls) == 3


def test_prod_host_refused_even_when_local() -> None:
    runner = FakeRunner(uri=LOCAL_URI)
    with pytest.raises(UnsafeDatabaseError):
        _wipe(_session(runner), env={"MEMINI_PROD_NEO4J_HOST": "localhost"})
    assert runner.calls == []


def test_guard_reads_only_the_given_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMINI_NEO4J_ALLOWED_HOSTS", "db.example.com")
    runner = FakeRunner(uri=REMOTE_URI)
    with pytest.raises(UnsafeDatabaseError):
        _wipe(_session(runner), env={})
    assert runner.calls == []


def test_read_only_scope_denied_with_scope_in_message() -> None:
    runner = FakeRunner()
    ctx = _ctx(read=frozenset({SCOPE}), write=frozenset())
    with pytest.raises(AccessDenied) as info:
        _wipe(_session(runner, ctx))
    assert SCOPE in str(info.value)
    assert runner.calls == []


def test_unknown_scope_denied() -> None:
    runner = FakeRunner()
    ctx = _ctx(read=frozenset({"other"}), write=frozenset({"other"}))
    with pytest.raises(AccessDenied) as info:
        _wipe(_session(runner, ctx))
    assert SCOPE in str(info.value)
    assert runner.calls == []


@pytest.mark.parametrize("reason", ["", "   ", "\t\n"])
def test_empty_or_blank_reason_refused(reason: str) -> None:
    runner = FakeRunner()
    with pytest.raises(ValueError):
        _wipe(_session(runner), reason=reason)
    assert runner.calls == []


@pytest.mark.parametrize("batch_size", [0, -1])
def test_batch_size_below_one_refused(batch_size: int) -> None:
    runner = FakeRunner()
    with pytest.raises(ValueError):
        _wipe(_session(runner), batch_size=batch_size)
    assert runner.calls == []


def test_confirmation_checked_before_guard() -> None:
    runner = FakeRunner(uri=REMOTE_URI)
    with pytest.raises(ConfirmationRequired):
        _wipe(_session(runner), confirm=None, env={})
    assert runner.calls == []


def test_guard_checked_before_access() -> None:
    runner = FakeRunner(uri=REMOTE_URI)
    ctx = _ctx(read=frozenset({SCOPE}), write=frozenset())
    with pytest.raises(UnsafeDatabaseError):
        _wipe(_session(runner, ctx), env={})
    assert runner.calls == []


def test_access_checked_before_value_checks() -> None:
    runner = FakeRunner()
    ctx = _ctx(read=frozenset({SCOPE}), write=frozenset())
    with pytest.raises(AccessDenied):
        _wipe(_session(runner, ctx), reason="", batch_size=0)
    assert runner.calls == []


# --- wipe statements ---------------------------------------------------------------------


def test_default_keep_labels_wipe_document_then_unit() -> None:
    runner = FakeRunner()
    _wipe(_session(runner))
    assert _wiped_labels(runner) == ["Document", "Unit"]
    for call, label in zip(_wipe_calls(runner), ["Document", "Unit"], strict=True):
        cypher = call["cypher"]
        assert f"MATCH (n:{label})" in cypher
        assert "n.scope = $scope" in cypher
        assert "DETACH DELETE n" in cypher
        assert "IN TRANSACTIONS OF $batch_size ROWS" in cypher
        assert "status" not in cypher
        assert SCOPE not in cypher
        assert call["params"] == {"scope": SCOPE, "batch_size": 1000}
        assert call["autocommit"] is True


def test_wipe_statements_run_before_audit_and_total_three_calls() -> None:
    runner = FakeRunner()
    _wipe(_session(runner))
    assert len(runner.calls) == 3
    assert "DETACH DELETE" in runner.calls[0]["cypher"]
    assert "DETACH DELETE" in runner.calls[1]["cypher"]
    assert "CREATE (e:AuditEvent" in runner.calls[2]["cypher"]


def test_custom_batch_size_passed_as_param() -> None:
    runner = FakeRunner()
    _wipe(_session(runner), batch_size=50)
    calls = _wipe_calls(runner)
    assert len(calls) == 2
    assert all(c["params"]["batch_size"] == 50 for c in calls)
    assert all("50" not in c["cypher"] for c in calls)


def test_declaration_order_followed() -> None:
    runner = FakeRunner()
    schema = _schema(("Unit", "AuditEvent", "Entity", "Document", "Scope"))
    _wipe(_session(runner), schema)
    assert _wiped_labels(runner) == ["Unit", "Entity", "Document"]


# --- counts and the audit event ----------------------------------------------------------


def test_returns_total_and_records_audit_event_last() -> None:
    runner = FakeRunner(rows_by_label={"Document": [{"count": 5}], "Unit": [{"count": 6}]})
    ctx = _ctx(actor="api-caller-3")
    total = _wipe(_session(runner, ctx), reason="scope retired")
    assert total == 11
    assert isinstance(total, int)
    last = runner.calls[-1]
    assert "CREATE (e:AuditEvent" in last["cypher"]
    params = last["params"]
    assert params["action"] == "wipe_scope"
    assert params["count"] == 11
    assert params["ids"] is None
    assert params["scope"] == SCOPE
    assert params["reason"] == "scope retired"
    assert params["actor"] == "api-caller-3"
    assert params["now"] == NOW.isoformat()
    assert last["autocommit"] is False


def test_count_summed_over_all_returned_rows() -> None:
    runner = FakeRunner(
        rows_by_label={
            "Document": [{"count": 2}, {"count": 3}],
            "Unit": [],
        }
    )
    total = _wipe(_session(runner))
    assert total == 5
    assert runner.calls[-1]["params"]["count"] == 5


def test_exactly_one_audit_event() -> None:
    runner = FakeRunner(rows_by_label={"Document": [{"count": 1}], "Unit": [{"count": 1}]})
    _wipe(_session(runner))
    audits = [c for c in runner.calls if "AuditEvent" in c["cypher"] and "CREATE" in c["cypher"]]
    assert len(audits) == 1


def test_recorded_names_in_order() -> None:
    recorder = Recorder()
    runner = FakeRunner()
    _wipe(_session(runner, recorder=recorder))
    assert recorder.names == ["wipe_scope:Document", "wipe_scope:Unit", "audit:wipe_scope"]


# --- keep_labels ----------------------------------------------------------------------------


def test_custom_keep_labels_never_wipes_audit_event() -> None:
    recorder = Recorder()
    runner = FakeRunner()
    _wipe(_session(runner, recorder=recorder), keep_labels=("Unit",))
    assert _wiped_labels(runner) == ["Document", "Scope", "Principal"]
    assert "AuditEvent" not in _wiped_labels(runner)
    assert recorder.names == [
        "wipe_scope:Document",
        "wipe_scope:Scope",
        "wipe_scope:Principal",
        "audit:wipe_scope",
    ]


def test_empty_keep_labels_still_keeps_audit_event() -> None:
    runner = FakeRunner()
    _wipe(_session(runner), keep_labels=())
    assert _wiped_labels(runner) == ["Document", "Unit", "Scope", "Principal"]


def test_only_kept_labels_returns_zero_with_one_audit_call() -> None:
    runner = FakeRunner()
    total = _wipe(_session(runner), _schema(("AuditEvent", "Scope", "Principal")))
    assert total == 0
    assert len(runner.calls) == 1
    only = runner.calls[0]
    assert "CREATE (e:AuditEvent" in only["cypher"]
    assert only["params"]["count"] == 0
    assert only["params"]["action"] == "wipe_scope"


def test_zero_counts_still_audited() -> None:
    runner = FakeRunner(rows_by_label={"Document": [{"count": 0}], "Unit": [{"count": 0}]})
    total = _wipe(_session(runner))
    assert total == 0
    assert runner.calls[-1]["params"]["count"] == 0
    assert "CREATE (e:AuditEvent" in runner.calls[-1]["cypher"]


# --- errors propagate ------------------------------------------------------------------------


def test_runner_error_propagates_without_audit() -> None:
    class BoomError(RuntimeError):
        pass

    class FailingRunner(FakeRunner):
        def run(
            self,
            cypher: str,
            params: Mapping[str, Any],
            *,
            profile: bool = False,
            autocommit: bool = False,
        ) -> RunResult:
            super().run(cypher, params, profile=profile, autocommit=autocommit)
            raise BoomError("db down")

    runner = FailingRunner()
    with pytest.raises(BoomError):
        _wipe(_session(runner))
    assert len(runner.calls) == 1
    assert all("AuditEvent" not in c["cypher"] for c in runner.calls)
