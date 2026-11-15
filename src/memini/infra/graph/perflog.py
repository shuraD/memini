"""Performance log of graph query runs, stored in the shared SQLite file.

Each run of a named query becomes one row of the ``graph_query_log`` table held by
an LP-08 :class:`~memini.infra.llm_providers.store.Store`, next to the other tables
of that file. :meth:`QueryPerfLog.query_summary` condenses the rows per query name
into count, nearest-rank percentiles of the duration, throughput and database hits.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from memini.infra.llm_providers.store import Store

SCHEMA_NAME: str = "graph_query_log"
"""Name registered with ``Store.ensure_schema``; also the table name."""

_DDL = """
CREATE TABLE IF NOT EXISTS graph_query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    scope TEXT,
    seconds REAL NOT NULL,
    rows INTEGER NOT NULL,
    db_hits INTEGER,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS graph_query_log_name ON graph_query_log (name, id)
"""


@dataclass(frozen=True)
class QueryRecord:
    """One logged query run."""

    name: str
    scope: str | None
    seconds: float
    rows: int
    db_hits: int | None
    ts: float


@dataclass(frozen=True)
class QuerySummary:
    """Aggregate figures of every run of one query name."""

    name: str
    count: int
    p50: float
    p95: float
    max: float
    rows_per_second: float | None
    db_hits_p95: int | None


def _nearest_rank[T: (int, float)](values: Sequence[T], percent: int) -> T:
    """Return the value at 1-based rank ``ceil(percent / 100 * n)`` of sorted ``values``."""
    ordered = sorted(values)
    rank = -(-percent * len(ordered) // 100)
    return ordered[max(rank, 1) - 1]


class QueryPerfLog:
    """Append-only log of graph query runs with per-name summaries."""

    def __init__(self, store: Store, clock: Callable[[], float] = time.time) -> None:
        """Bind to ``store`` and create the log table once; ``clock`` stamps each row."""
        self._store = store
        self._clock = clock
        store.ensure_schema(SCHEMA_NAME, _DDL)

    def record(
        self, name: str, scope: str | None, seconds: float, rows: int, db_hits: int | None
    ) -> None:
        """Insert one run, stamped with the clock value at call time."""
        self._store.execute(
            "INSERT INTO graph_query_log (name, scope, seconds, rows, db_hits, ts)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (name, scope, float(seconds), int(rows), db_hits, float(self._clock())),
        )

    def rows(self, name: str | None = None) -> list[QueryRecord]:
        """Return every run, or only those of ``name``, in insertion order."""
        result: list[tuple[Any, ...]]
        if name is None:
            result = self._store.execute(
                "SELECT name, scope, seconds, rows, db_hits, ts FROM graph_query_log ORDER BY id"
            )
        else:
            result = self._store.execute(
                "SELECT name, scope, seconds, rows, db_hits, ts FROM graph_query_log"
                " WHERE name = ? ORDER BY id",
                (name,),
            )
        return [QueryRecord(*row) for row in result]

    def query_summary(self) -> dict[str, QuerySummary]:
        """Summarise the runs per distinct name; an empty log gives ``{}``."""
        groups: dict[str, list[QueryRecord]] = {}
        for record in self.rows():
            groups.setdefault(record.name, []).append(record)
        return {name: _summarise(name, records) for name, records in groups.items()}


def _summarise(name: str, records: list[QueryRecord]) -> QuerySummary:
    """Build the summary of one name's non-empty list of runs."""
    seconds = [r.seconds for r in records]
    total_seconds = sum(seconds)
    hits = [r.db_hits for r in records if r.db_hits is not None]
    return QuerySummary(
        name=name,
        count=len(records),
        p50=_nearest_rank(seconds, 50),
        p95=_nearest_rank(seconds, 95),
        max=max(seconds),
        rows_per_second=(
            sum(r.rows for r in records) / total_seconds if total_seconds != 0 else None
        ),
        db_hits_p95=_nearest_rank(hits, 95) if hits else None,
    )
