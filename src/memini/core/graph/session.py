"""Graph session: runs builder statements under an access context and records their timing.

Only ``Statement`` objects are accepted, so engine code cannot send raw Cypher. Each
statement is re-checked for labeled ``MATCH`` patterns (and, for writes, audit fields)
before it reaches the runner; every run that reaches the runner is recorded once.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any

from memini.core.graph.checks import require_audited, require_labeled
from memini.core.graph.types import (
    AccessContext,
    AccessDenied,
    PerfRecorder,
    Runner,
    RunResult,
    Statement,
)


class GraphSession:
    """Execute statements through a runner on behalf of one access context."""

    def __init__(
        self,
        runner: Runner,
        ctx: AccessContext | None,
        *,
        recorder: PerfRecorder | None = None,
        profile_every: int = 0,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Bind the session to a runner and an access context.

        Args:
            runner: Executes the query text.
            ctx: Access context of the caller; None refuses every run.
            recorder: Receives one performance sample per run, or None to record nothing.
            profile_every: Profile every n-th run of each statement name; 0 never profiles.
            timer: Monotonic clock in seconds used to time each run.

        """
        self._runner = runner
        self._ctx = ctx
        self._recorder = recorder
        self._profile_every = profile_every
        self._timer = timer
        self._counts: dict[str, int] = {}

    @property
    def ctx(self) -> AccessContext:
        """The access context of the session.

        Raises:
            AccessDenied: When the session has no access context.

        """
        if self._ctx is None:
            raise AccessDenied("no access context")
        return self._ctx

    @property
    def uri(self) -> str:
        """The URI of the underlying runner."""
        return self._runner.uri

    def run(self, stmt: Statement) -> list[dict[str, Any]]:
        """Check, run and record one statement, and return its rows.

        Any error raised by the runner propagates unchanged after being recorded.

        Raises:
            TypeError: When ``stmt`` is not a ``Statement``.
            AccessDenied: When the session has no access context.
            UnlabeledMatchError: When a ``MATCH`` pattern names no label.
            AuditFieldError: When a write statement misses audit fields.

        """
        if not isinstance(stmt, Statement):
            raise TypeError(f"expected a Statement, got {type(stmt).__name__}")
        _ = self.ctx
        require_labeled(stmt.cypher)
        if stmt.kind == "write":
            require_audited(stmt.cypher)
        n = self._counts.get(stmt.name, 0) + 1
        self._counts[stmt.name] = n
        profile = self._profile_every > 0 and n % self._profile_every == 0
        t0 = self._timer()
        try:
            result: RunResult = self._runner.run(
                stmt.cypher, stmt.params, profile=profile, autocommit=stmt.kind == "admin"
            )
        except BaseException:
            self._record(stmt, self._timer() - t0, 0, None)
            raise
        seconds = self._timer() - t0
        self._record(stmt, seconds, len(result.rows), result.db_hits if profile else None)
        return result.rows

    def run_all(self, stmts: Iterable[Statement]) -> list[dict[str, Any]]:
        """Run each statement in order and return their rows concatenated.

        Stops at the first error; later statements are not run.
        """
        rows: list[dict[str, Any]] = []
        for stmt in stmts:
            rows.extend(self.run(stmt))
        return rows

    def _record(self, stmt: Statement, seconds: float, rows: int, db_hits: int | None) -> None:
        """Send one sample to the recorder, if any."""
        if self._recorder is not None:
            self._recorder.record(stmt.name, stmt.scope, seconds, rows, db_hits)
