"""Acceptance tests for GS-10: GraphSession of memini.core.graph.session."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from memini.core.graph import (
    AccessContext,
    AccessDenied,
    AuditFieldError,
    RunResult,
    Statement,
    UnlabeledMatchError,
)
from memini.core.graph.session import GraphSession

URI = "bolt://localhost:7687"
READ_CYPHER = "MATCH (n:Unit {scope:$scope}) RETURN n"
UNLABELED_CYPHER = "MATCH (n {scope:$scope}) RETURN n"
AUDITED_WRITE = (
    "CREATE (n:Unit {scope:$scope, created_at:$now, updated_at:$now, updated_by:$actor}) RETURN n"
)
UNAUDITED_WRITE = "MATCH (n:Unit {id:$id}) SET n.title = $title RETURN n"
ROWS = [{"n": 1}, {"n": 2}, {"n": 3}]


class FakeRunner:
    """Records every call and returns a fixed result, or raises a given error."""

    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        db_hits: int | None = None,
        error: Exception | None = None,
    ) -> None:
        self.uri = URI
        self.rows = list(ROWS) if rows is None else rows
        self.db_hits = db_hits
        self.error = error
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
            {"cypher": cypher, "params": params, "profile": profile, "autocommit": autocommit}
        )
        if self.error is not None:
            raise self.error
        return RunResult(rows=list(self.rows), db_hits=self.db_hits)


class FakeRecorder:
    """Collects every recorded sample."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str | None, float, int, int | None]] = []

    def record(
        self, name: str, scope: str | None, seconds: float, rows: int, db_hits: int | None
    ) -> None:
        self.records.append((name, scope, seconds, rows, db_hits))


class FakeTimer:
    """Returns 0.0, 0.25, 0.5, ... : each call advances by ``step``."""

    def __init__(self, step: float = 0.25) -> None:
        self.step = step
        self.calls = 0

    def __call__(self) -> float:
        value = self.calls * self.step
        self.calls += 1
        return value


def _ctx() -> AccessContext:
    return AccessContext(
        principal="tester",
        read_scopes=frozenset({"s1"}),
        write_scopes=frozenset({"s1"}),
        actor="job-1",
    )


def _stmt(
    name: str = "q.read",
    cypher: str = READ_CYPHER,
    kind: Any = "read",
    scope: str | None = "s1",
    params: Mapping[str, Any] | None = None,
) -> Statement:
    return Statement(
        name=name,
        cypher=cypher,
        params={"scope": "s1"} if params is None else params,
        scope=scope,
        kind=kind,
    )


def _session(
    runner: FakeRunner | None = None,
    *,
    ctx: AccessContext | None = None,
    no_ctx: bool = False,
    recorder: FakeRecorder | None = None,
    profile_every: int = 0,
    timer: FakeTimer | None = None,
) -> GraphSession:
    return GraphSession(
        runner if runner is not None else FakeRunner(),
        None if no_ctx else (ctx if ctx is not None else _ctx()),
        recorder=recorder,
        profile_every=profile_every,
        timer=timer if timer is not None else FakeTimer(),
    )


# ---------------------------------------------------------------- statement type


def test_raw_cypher_string_raises_type_error_and_runner_not_called() -> None:
    runner = FakeRunner()
    recorder = FakeRecorder()
    session = _session(runner, recorder=recorder)
    with pytest.raises(TypeError):
        session.run("MATCH (n:Unit) RETURN n")  # type: ignore[arg-type]
    assert runner.calls == []
    assert recorder.records == []


def test_type_error_comes_before_access_check() -> None:
    runner = FakeRunner()
    session = _session(runner, no_ctx=True)
    with pytest.raises(TypeError):
        session.run("MATCH (n:Unit) RETURN n")  # type: ignore[arg-type]
    assert runner.calls == []


def test_non_statement_object_raises_type_error() -> None:
    runner = FakeRunner()
    session = _session(runner)
    fake = {"name": "q", "cypher": READ_CYPHER, "params": {}, "scope": "s1", "kind": "read"}
    with pytest.raises(TypeError):
        session.run(fake)  # type: ignore[arg-type]
    assert runner.calls == []


# ---------------------------------------------------------------- access context


def test_ctx_none_run_raises_access_denied_and_runner_not_called() -> None:
    runner = FakeRunner()
    recorder = FakeRecorder()
    session = _session(runner, no_ctx=True, recorder=recorder)
    with pytest.raises(AccessDenied):
        session.run(_stmt())
    assert runner.calls == []
    assert recorder.records == []


def test_ctx_none_reading_ctx_raises_access_denied() -> None:
    session = _session(no_ctx=True)
    with pytest.raises(AccessDenied):
        _ = session.ctx


def test_access_denied_comes_before_label_check() -> None:
    runner = FakeRunner()
    session = _session(runner, no_ctx=True)
    with pytest.raises(AccessDenied):
        session.run(_stmt(cypher=UNLABELED_CYPHER))
    assert runner.calls == []


def test_ctx_property_returns_given_context() -> None:
    ctx = _ctx()
    session = _session(ctx=ctx)
    assert session.ctx is ctx


def test_uri_property_is_runner_uri() -> None:
    runner = FakeRunner()
    runner.uri = "bolt://127.0.0.1:7999"
    session = _session(runner)
    assert session.uri == "bolt://127.0.0.1:7999"


def test_ctx_property_is_read_only() -> None:
    session = _session()
    with pytest.raises(AttributeError):
        session.ctx = _ctx()  # type: ignore[misc]


def test_uri_property_is_read_only() -> None:
    session = _session()
    with pytest.raises(AttributeError):
        session.uri = "bolt://localhost:1"  # type: ignore[misc]


# ---------------------------------------------------------------- labels and audit


@pytest.mark.parametrize("kind", ["read", "write", "admin"])
def test_unlabeled_match_raises_and_neither_runner_nor_recorder_called(kind: str) -> None:
    runner = FakeRunner()
    recorder = FakeRecorder()
    session = _session(runner, recorder=recorder)
    with pytest.raises(UnlabeledMatchError):
        session.run(_stmt(cypher=UNLABELED_CYPHER, kind=kind))
    assert runner.calls == []
    assert recorder.records == []


def test_unlabeled_check_comes_before_audit_check() -> None:
    runner = FakeRunner()
    session = _session(runner)
    with pytest.raises(UnlabeledMatchError):
        session.run(_stmt(cypher="MATCH (n {id:$id}) SET n.title = $title", kind="write"))
    assert runner.calls == []


def test_write_without_audit_fields_raises_and_runner_not_called() -> None:
    runner = FakeRunner()
    recorder = FakeRecorder()
    session = _session(runner, recorder=recorder)
    with pytest.raises(AuditFieldError):
        session.run(_stmt(name="q.write", cypher=UNAUDITED_WRITE, kind="write"))
    assert runner.calls == []
    assert recorder.records == []


def test_same_unaudited_cypher_as_read_runs() -> None:
    runner = FakeRunner()
    session = _session(runner)
    rows = session.run(_stmt(cypher=UNAUDITED_WRITE, kind="read"))
    assert rows == ROWS
    assert len(runner.calls) == 1
    assert runner.calls[0]["cypher"] == UNAUDITED_WRITE


def test_same_unaudited_cypher_as_admin_is_not_audit_checked() -> None:
    runner = FakeRunner()
    session = _session(runner)
    rows = session.run(_stmt(cypher=UNAUDITED_WRITE, kind="admin"))
    assert rows == ROWS
    assert len(runner.calls) == 1


def test_audited_write_runs() -> None:
    runner = FakeRunner()
    session = _session(runner)
    rows = session.run(_stmt(name="q.write", cypher=AUDITED_WRITE, kind="write"))
    assert rows == ROWS
    assert len(runner.calls) == 1


# ---------------------------------------------------------------- run and record


def test_run_returns_rows_and_records_once() -> None:
    runner = FakeRunner(rows=list(ROWS))
    recorder = FakeRecorder()
    timer = FakeTimer(step=0.25)
    session = _session(runner, recorder=recorder, timer=timer)
    rows = session.run(_stmt(name="q.units", scope="s1"))
    assert rows == ROWS
    assert len(recorder.records) == 1
    name, scope, seconds, count, db_hits = recorder.records[0]
    assert (name, scope, count, db_hits) == ("q.units", "s1", 3, None)
    assert seconds == pytest.approx(0.25)


def test_timer_called_once_before_and_once_after_runner() -> None:
    timer = FakeTimer(step=0.25)
    order: list[str] = []

    class OrderedRunner(FakeRunner):
        def run(self, cypher: str, params: Mapping[str, Any], **kwargs: Any) -> RunResult:
            order.append(f"run@{timer.calls}")
            return super().run(cypher, params, **kwargs)

    runner = OrderedRunner()
    session = _session(runner, recorder=FakeRecorder(), timer=timer)
    session.run(_stmt())
    assert timer.calls == 2
    assert order == ["run@1"]


def test_seconds_measured_with_custom_step() -> None:
    recorder = FakeRecorder()
    session = _session(recorder=recorder, timer=FakeTimer(step=1.5))
    session.run(_stmt())
    session.run(_stmt())
    assert [r[2] for r in recorder.records] == [pytest.approx(1.5), pytest.approx(1.5)]


def test_record_scope_none_is_passed_through() -> None:
    recorder = FakeRecorder()
    session = _session(recorder=recorder)
    session.run(_stmt(name="q.global", scope=None))
    assert recorder.records[0][0] == "q.global"
    assert recorder.records[0][1] is None


def test_empty_result_records_zero_rows() -> None:
    recorder = FakeRecorder()
    session = _session(FakeRunner(rows=[]), recorder=recorder)
    assert session.run(_stmt()) == []
    assert recorder.records[0][3] == 0


def test_runner_receives_cypher_and_params_as_given() -> None:
    runner = FakeRunner()
    params = {"scope": "s1", "ids": [1, 2], "nested": {"k": "v"}}
    session = _session(runner)
    session.run(_stmt(params=params))
    assert len(runner.calls) == 1
    assert runner.calls[0]["cypher"] == READ_CYPHER
    assert runner.calls[0]["params"] == params


# ---------------------------------------------------------------- profiling


def test_profile_every_two_profiles_runs_two_and_four() -> None:
    runner = FakeRunner(db_hits=42)
    recorder = FakeRecorder()
    session = _session(runner, recorder=recorder, profile_every=2)
    for _ in range(4):
        session.run(_stmt(name="q.same"))
    assert [c["profile"] for c in runner.calls] == [False, True, False, True]
    assert [r[4] for r in recorder.records] == [None, 42, None, 42]


def test_profile_counted_separately_per_name() -> None:
    runner = FakeRunner(db_hits=7)
    recorder = FakeRecorder()
    session = _session(runner, recorder=recorder, profile_every=2)
    for name in ["q.a", "q.b", "q.a", "q.b", "q.a", "q.b"]:
        session.run(_stmt(name=name))
    assert [c["profile"] for c in runner.calls] == [False, False, True, True, False, False]
    assert [(r[0], r[4]) for r in recorder.records] == [
        ("q.a", None),
        ("q.b", None),
        ("q.a", 7),
        ("q.b", 7),
        ("q.a", None),
        ("q.b", None),
    ]


def test_profile_every_one_profiles_every_run() -> None:
    runner = FakeRunner(db_hits=3)
    session = _session(runner, profile_every=1)
    session.run(_stmt())
    session.run(_stmt())
    assert [c["profile"] for c in runner.calls] == [True, True]


def test_profile_every_zero_default_never_profiles() -> None:
    runner = FakeRunner(db_hits=99)
    recorder = FakeRecorder()
    session = GraphSession(runner, _ctx(), recorder=recorder, timer=FakeTimer())
    for _ in range(5):
        session.run(_stmt(name="q.same"))
    assert len(runner.calls) == 5
    assert all(c["profile"] is False for c in runner.calls)
    assert all(r[4] is None for r in recorder.records)


def test_refused_runs_do_not_count_toward_profiling() -> None:
    runner = FakeRunner(db_hits=5)
    session = _session(runner, profile_every=2)
    with pytest.raises(UnlabeledMatchError):
        session.run(_stmt(name="q.same", cypher=UNLABELED_CYPHER))
    session.run(_stmt(name="q.same"))
    session.run(_stmt(name="q.same"))
    assert [c["profile"] for c in runner.calls] == [False, True]


def test_failed_runs_count_toward_profiling() -> None:
    runner = FakeRunner(error=RuntimeError("boom"))
    session = _session(runner, profile_every=2)
    with pytest.raises(RuntimeError):
        session.run(_stmt(name="q.same"))
    runner.error = None
    session.run(_stmt(name="q.same"))
    assert [c["profile"] for c in runner.calls] == [False, True]


def test_profile_counts_are_per_session() -> None:
    runner = FakeRunner(db_hits=1)
    first = _session(runner, profile_every=2)
    first.run(_stmt(name="q.same"))
    second = _session(runner, profile_every=2)
    second.run(_stmt(name="q.same"))
    assert [c["profile"] for c in runner.calls] == [False, False]


# ---------------------------------------------------------------- runner errors


def test_runner_error_records_once_and_propagates_same_exception() -> None:
    error = RuntimeError("database unavailable")
    runner = FakeRunner(error=error)
    recorder = FakeRecorder()
    timer = FakeTimer(step=0.25)
    session = _session(runner, recorder=recorder, timer=timer, profile_every=1)
    with pytest.raises(RuntimeError) as excinfo:
        session.run(_stmt(name="q.fail", scope="s1"))
    assert excinfo.value is error
    assert timer.calls == 2
    assert len(recorder.records) == 1
    name, scope, seconds, count, db_hits = recorder.records[0]
    assert (name, scope, count, db_hits) == ("q.fail", "s1", 0, None)
    assert seconds == pytest.approx(0.25)


def test_runner_error_without_recorder_propagates() -> None:
    error = ValueError("bad")
    session = _session(FakeRunner(error=error), recorder=None)
    with pytest.raises(ValueError) as excinfo:
        session.run(_stmt())
    assert excinfo.value is error


# ---------------------------------------------------------------- autocommit


@pytest.mark.parametrize(
    ("kind", "cypher", "expected"),
    [
        ("admin", READ_CYPHER, True),
        ("read", READ_CYPHER, False),
        ("write", AUDITED_WRITE, False),
    ],
)
def test_autocommit_only_for_admin(kind: str, cypher: str, expected: bool) -> None:
    runner = FakeRunner()
    session = _session(runner)
    session.run(_stmt(cypher=cypher, kind=kind))
    assert runner.calls[0]["autocommit"] is expected


# ---------------------------------------------------------------- recorder None


def test_recorder_none_still_returns_rows() -> None:
    runner = FakeRunner(rows=list(ROWS), db_hits=4)
    session = GraphSession(runner, _ctx(), profile_every=1, timer=FakeTimer())
    assert session.run(_stmt()) == ROWS
    assert len(runner.calls) == 1


# ---------------------------------------------------------------- run_all


class SequenceRunner(FakeRunner):
    """Returns rows tagged with the cypher text so the order is visible."""

    def run(
        self,
        cypher: str,
        params: Mapping[str, Any],
        *,
        profile: bool = False,
        autocommit: bool = False,
    ) -> RunResult:
        super().run(cypher, params, profile=profile, autocommit=autocommit)
        tag = params["tag"]
        return RunResult(rows=[{"tag": tag, "i": 0}, {"tag": tag, "i": 1}])


def test_run_all_concatenates_rows_in_order() -> None:
    runner = SequenceRunner()
    recorder = FakeRecorder()
    session = _session(runner, recorder=recorder)
    stmts = [_stmt(name=f"q.{tag}", params={"tag": tag}) for tag in ("a", "b", "c")]
    rows = session.run_all(stmts)
    assert rows == [
        {"tag": "a", "i": 0},
        {"tag": "a", "i": 1},
        {"tag": "b", "i": 0},
        {"tag": "b", "i": 1},
        {"tag": "c", "i": 0},
        {"tag": "c", "i": 1},
    ]
    assert [r[0] for r in recorder.records] == ["q.a", "q.b", "q.c"]


def test_run_all_accepts_any_iterable() -> None:
    runner = SequenceRunner()
    session = _session(runner)

    def gen() -> Iterator[Statement]:
        yield _stmt(params={"tag": "x"})
        yield _stmt(params={"tag": "y"})

    rows = session.run_all(gen())
    assert [r["tag"] for r in rows] == ["x", "x", "y", "y"]


def test_run_all_empty_returns_empty_list() -> None:
    runner = FakeRunner()
    session = _session(runner)
    assert session.run_all([]) == []
    assert runner.calls == []


def test_run_all_stops_when_second_is_refused() -> None:
    runner = SequenceRunner()
    recorder = FakeRecorder()
    session = _session(runner, recorder=recorder)
    stmts = [
        _stmt(name="q.a", params={"tag": "a"}),
        _stmt(name="q.b", cypher=UNLABELED_CYPHER, params={"tag": "b"}),
        _stmt(name="q.c", params={"tag": "c"}),
    ]
    with pytest.raises(UnlabeledMatchError):
        session.run_all(stmts)
    assert [c["params"]["tag"] for c in runner.calls] == ["a"]
    assert [r[0] for r in recorder.records] == ["q.a"]


def test_run_all_stops_when_runner_fails() -> None:
    error = RuntimeError("boom")

    class FailOnB(SequenceRunner):
        def run(
            self,
            cypher: str,
            params: Mapping[str, Any],
            *,
            profile: bool = False,
            autocommit: bool = False,
        ) -> RunResult:
            if params["tag"] == "b":
                self.calls.append({"params": params})
                raise error
            return super().run(cypher, params, profile=profile, autocommit=autocommit)

    runner = FailOnB()
    session = _session(runner)
    stmts = [_stmt(params={"tag": t}) for t in ("a", "b", "c")]
    with pytest.raises(RuntimeError) as excinfo:
        session.run_all(stmts)
    assert excinfo.value is error
    assert [c["params"]["tag"] for c in runner.calls] == ["a", "b"]


def test_run_all_refuses_raw_string_in_sequence() -> None:
    runner = FakeRunner()
    session = _session(runner)
    with pytest.raises(TypeError):
        session.run_all(["MATCH (n:Unit) RETURN n"])  # type: ignore[list-item]
    assert runner.calls == []
