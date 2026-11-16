"""Acceptance tests for GS-21: the graph query performance log on the shared store."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from memini.infra.graph.perflog import (
    SCHEMA_NAME,
    QueryPerfLog,
    QueryRecord,
    QuerySummary,
)
from memini.infra.llm_providers.store import MEMORY, Store

FIXED_TS = 1700000000.5


def _clock() -> float:
    return FIXED_TS


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "shared.sqlite"


@pytest.fixture
def store(db_path: Path) -> Iterator[Store]:
    s = Store(db_path)
    yield s
    s.close()


@pytest.fixture
def log(store: Store) -> QueryPerfLog:
    return QueryPerfLog(store, clock=_clock)


def _tables(path: Path) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def test_schema_name_is_table_name() -> None:
    assert SCHEMA_NAME == "graph_query_log"


def test_construction_creates_table(store: Store, db_path: Path) -> None:
    QueryPerfLog(store, clock=_clock)
    assert "graph_query_log" in _tables(db_path)


def test_record_with_every_field_round_trips(log: QueryPerfLog) -> None:
    log.record("read:Unit", 0.125, 42, 314)
    assert log.rows() == [
        QueryRecord(
            name="read:Unit",
            seconds=0.125,
            rows=42,
            db_hits=314,
            ts=FIXED_TS,
        )
    ]


def test_record_with_none_db_hits_round_trips(log: QueryPerfLog) -> None:
    log.record("write:Unit", 0.5, 7, None)
    rows = log.rows()
    assert len(rows) == 1
    assert rows[0].db_hits is None
    assert rows[0] == QueryRecord("write:Unit", 0.5, 7, None, FIXED_TS)


def test_ts_is_clock_value_at_record_time(store: Store) -> None:
    now = [10.0]

    def clock() -> float:
        return now[0]

    perf = QueryPerfLog(store, clock=clock)
    now[0] = 111.5
    perf.record("a", 0.1, 1, None)
    now[0] = 222.25
    perf.record("a", 0.2, 1, None)
    assert [r.ts for r in perf.rows()] == [111.5, 222.25]


def test_rows_filtered_by_name_in_insertion_order(log: QueryPerfLog) -> None:
    log.record("b", 0.3, 3, None)
    log.record("a", 0.1, 1, None)
    log.record("b", 0.2, 2, 5)
    log.record("c", 0.4, 4, None)
    log.record("a", 0.05, 9, None)

    only_a = log.rows("a")
    assert [(r.name, r.seconds) for r in only_a] == [
        ("a", 0.1),
        ("a", 0.05),
    ]
    only_b = log.rows(name="b")
    assert [r.seconds for r in only_b] == [0.3, 0.2]
    assert log.rows("missing") == []

    every = log.rows()
    assert [(r.name, r.seconds) for r in every] == [
        ("b", 0.3),
        ("a", 0.1),
        ("b", 0.2),
        ("c", 0.4),
        ("a", 0.05),
    ]


def test_summary_twenty_records(log: QueryPerfLog) -> None:
    # Insert in a shuffled (fixed) order so the summary must sort.
    order = [7, 19, 0, 13, 4, 18, 1, 10, 15, 2, 9, 17, 5, 12, 3, 16, 8, 11, 6, 14]
    for i in order:
        log.record("read:Unit", round((i + 1) * 0.01, 2), 1, None)

    summary = log.query_summary()
    s = summary["read:Unit"]
    assert isinstance(s, QuerySummary)
    assert s.name == "read:Unit"
    assert s.count == 20
    assert s.p50 == pytest.approx(0.10)
    assert s.p95 == pytest.approx(0.19)
    assert s.max == pytest.approx(0.20)


def test_summary_single_record(log: QueryPerfLog) -> None:
    log.record("one", 0.37, 5, None)
    s = log.query_summary()["one"]
    assert s.count == 1
    assert s.p50 == pytest.approx(0.37)
    assert s.p95 == pytest.approx(0.37)
    assert s.max == pytest.approx(0.37)


def test_summary_per_name_independent(log: QueryPerfLog) -> None:
    for sec in (0.1, 0.2, 0.3):
        log.record("x", sec, 10, None)
    for sec in (5.0, 1.0):
        log.record("y", sec, 4, None)

    summary = log.query_summary()
    assert set(summary) == {"x", "y"}
    for key, value in summary.items():
        assert value.name == key

    x = summary["x"]
    assert x.count == 3
    assert x.p50 == pytest.approx(0.2)
    assert x.p95 == pytest.approx(0.3)
    assert x.max == pytest.approx(0.3)
    assert x.rows_per_second == pytest.approx(30 / 0.6)

    y = summary["y"]
    assert y.count == 2
    assert y.p50 == pytest.approx(1.0)
    assert y.p95 == pytest.approx(5.0)
    assert y.max == pytest.approx(5.0)
    assert y.rows_per_second == pytest.approx(8 / 6.0)


def test_summary_empty_log(log: QueryPerfLog) -> None:
    assert log.query_summary() == {}


def test_rows_per_second(log: QueryPerfLog) -> None:
    log.record("write:Unit", 0.5, 200, None)
    log.record("write:Unit", 0.5, 300, None)
    log.record("write:Unit", 1.0, 500, None)
    s = log.query_summary()["write:Unit"]
    assert s.rows_per_second == pytest.approx(500.0)


def test_rows_per_second_none_when_all_seconds_zero(log: QueryPerfLog) -> None:
    log.record("write:Unit", 0.0, 10, None)
    log.record("write:Unit", 0.0, 20, None)
    s = log.query_summary()["write:Unit"]
    assert s.rows_per_second is None
    assert s.max == 0.0


def test_db_hits_p95_only_over_sampled_rows(log: QueryPerfLog) -> None:
    # 20 sampled values 1..20 (nearest-rank p95 = rank 19 -> 19), interleaved
    # with unsampled rows that must be ignored.
    for i in range(1, 21):
        log.record("q", 0.01, 1, i)
        log.record("q", 0.01, 1, None)
    s = log.query_summary()["q"]
    assert s.count == 40
    assert s.db_hits_p95 == 19


def test_db_hits_p95_small_sample(log: QueryPerfLog) -> None:
    log.record("q", 0.1, 1, None)
    log.record("q", 0.1, 1, 300)
    log.record("q", 0.1, 1, None)
    log.record("q", 0.1, 1, 100)
    s = log.query_summary()["q"]
    assert s.db_hits_p95 == 300


def test_db_hits_p95_none_when_not_sampled(log: QueryPerfLog) -> None:
    log.record("q", 0.1, 1, None)
    log.record("q", 0.2, 2, None)
    s = log.query_summary()["q"]
    assert s.db_hits_p95 is None


def _hammer(perf: QueryPerfLog) -> None:
    def worker(n: int) -> None:
        for i in range(25):
            perf.record(f"t{n}", 0.001 * (i + 1), i, None)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_threads_on_file_store(log: QueryPerfLog) -> None:
    _hammer(log)
    rows = log.rows()
    assert len(rows) == 200
    summary = log.query_summary()
    assert sum(s.count for s in summary.values()) == 200
    assert all(len(log.rows(f"t{n}")) == 25 for n in range(8))


def test_threads_on_memory_store() -> None:
    mem = Store(MEMORY)
    try:
        perf = QueryPerfLog(mem, clock=_clock)
        _hammer(perf)
        assert len(perf.rows()) == 200
    finally:
        mem.close()


def test_coexists_with_other_table(db_path: Path) -> None:
    first = Store(db_path)
    first.ensure_schema(
        "other.v1",
        "CREATE TABLE other_log (id INTEGER PRIMARY KEY, label TEXT NOT NULL)",
    )
    first.execute("INSERT INTO other_log (label) VALUES (?)", ("alpha",))
    first.execute("INSERT INTO other_log (label) VALUES (?)", ("beta",))

    perf = QueryPerfLog(first, clock=_clock)
    perf.record("read:Unit", 0.1, 3, None)
    first.close()

    assert {"other_log", "graph_query_log"} <= _tables(db_path)
    conn = sqlite3.connect(db_path)
    try:
        labels = conn.execute("SELECT label FROM other_log ORDER BY id").fetchall()
    finally:
        conn.close()
    assert labels == [("alpha",), ("beta",)]


def test_reopen_file_returns_same_records(db_path: Path) -> None:
    first = Store(db_path)
    perf = QueryPerfLog(first, clock=_clock)
    perf.record("read:Unit", 0.25, 10, 99)
    perf.record("write:Unit", 0.75, 20, None)
    before = perf.rows()
    first.close()

    second = Store(db_path)
    try:
        reopened = QueryPerfLog(second, clock=lambda: 0.0)
        assert reopened.rows() == before
        assert len(before) == 2
    finally:
        second.close()


def test_two_instances_share_rows(store: Store) -> None:
    a = QueryPerfLog(store, clock=_clock)
    b = QueryPerfLog(store, clock=_clock)
    a.record("x", 0.1, 1, None)
    b.record("y", 0.2, 2, None)
    assert a.rows() == b.rows()
    assert [r.name for r in a.rows()] == ["x", "y"]
    assert set(a.query_summary()) == {"x", "y"}
