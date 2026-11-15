"""Per-role, per-question and per-run spending caps kept in the shared store.

Three caps come from :class:`LLMConfig`: a role's ``max_calls_per_question``, a
role's ``max_tokens_per_run`` and the global ``max_tokens_per_run`` summed over all
roles. Counters live in the shared :class:`Store`, keyed by the config's ``run_id``,
so every process and thread on one file stops at one combined budget.

The current question is held in a :mod:`contextvars` variable set by :func:`question`;
each thread starts outside any question. Calls made outside a question count only
against the run budgets.

:meth:`Budgets.check` counts the call (inside a question) in the same transaction as
it verifies the caps, so concurrent callers cannot both take the last call slot.
:meth:`Budgets.record` adds the tokens a call actually spent.
"""

from __future__ import annotations

import contextvars
import sqlite3
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager

from memini.infra.llm_providers.config import LLMConfig
from memini.infra.llm_providers.store import Store
from memini.infra.llm_providers.types import Attempt, BudgetExceeded, Reason

SCHEMA_NAME: str = "llm_budgets_v1"
"""Name under which the budget tables are registered in the store."""

_SCHEMA: str = """
CREATE TABLE IF NOT EXISTS llm_budget_tokens (
    run_id TEXT NOT NULL,
    role TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    PRIMARY KEY (run_id, role)
);
CREATE TABLE IF NOT EXISTS llm_budget_calls (
    run_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    role TEXT NOT NULL,
    calls INTEGER NOT NULL,
    PRIMARY KEY (run_id, question_id, role)
);
"""

BUDGET_SOURCE: str = "budget"
"""Provider field of the :class:`Attempt` carried by :class:`BudgetExceeded`."""

_current: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_budget_question", default=None
)


@contextmanager
def _scope(question_id: str) -> Iterator[None]:
    """Set the current question for the ``with`` body and restore the previous one."""
    token = _current.set(question_id)
    try:
        yield
    finally:
        _current.reset(token)


def question(question_id: str) -> AbstractContextManager[None]:
    """Return a context manager making ``question_id`` the current question."""
    return _scope(question_id)


def current_question() -> str | None:
    """Return the current question id, or ``None`` outside any :func:`question`."""
    return _current.get()


class Budgets:
    """Check and record spending against the configured caps in the shared store."""

    def __init__(self, config: LLMConfig, store: Store) -> None:
        """Keep ``config`` and ``store``; create the budget tables if missing."""
        self._config = config
        self._store = store
        store.ensure_schema(SCHEMA_NAME, _SCHEMA)

    def check(self, role: str) -> None:
        """Raise :class:`BudgetExceeded` if a call for ``role`` would exceed a cap.

        Inside a question, a passing check counts one call for ``role`` there.
        """
        caps = self._config.role_budget(role)
        global_cap = self._config.global_max_tokens_per_run
        qid = current_question()
        run_id = self._config.run_id
        with self._store.transaction() as conn:
            if caps.max_tokens_per_run is not None:
                spent = _scalar(
                    conn,
                    "SELECT tokens FROM llm_budget_tokens WHERE run_id = ? AND role = ?",
                    (run_id, role),
                )
                if spent >= caps.max_tokens_per_run:
                    raise _exceeded(role, f"role tokens per run {spent}/{caps.max_tokens_per_run}")
            if global_cap is not None:
                total = _scalar(
                    conn,
                    "SELECT SUM(tokens) FROM llm_budget_tokens WHERE run_id = ?",
                    (run_id,),
                )
                if total >= global_cap:
                    raise _exceeded(role, f"global tokens per run {total}/{global_cap}")
            if qid is None:
                return
            calls = _scalar(
                conn,
                "SELECT calls FROM llm_budget_calls"
                " WHERE run_id = ? AND question_id = ? AND role = ?",
                (run_id, qid, role),
            )
            cap = caps.max_calls_per_question
            if cap is not None and calls >= cap:
                raise _exceeded(role, f"calls in question {qid!r} {calls}/{cap}")
            conn.execute(
                "INSERT INTO llm_budget_calls (run_id, question_id, role, calls)"
                " VALUES (?, ?, ?, 1) ON CONFLICT (run_id, question_id, role)"
                " DO UPDATE SET calls = calls + 1",
                (run_id, qid, role),
            )

    def record(self, role: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Add the tokens one call for ``role`` spent to the run totals."""
        self._store.execute(
            "INSERT INTO llm_budget_tokens (run_id, role, tokens) VALUES (?, ?, ?)"
            " ON CONFLICT (run_id, role) DO UPDATE SET tokens = tokens + excluded.tokens",
            (self._config.run_id, role, prompt_tokens + completion_tokens),
        )


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple[str, ...]) -> int:
    """Return the first column of the first row as an int; missing or NULL is 0."""
    row = conn.execute(sql, params).fetchone()
    return 0 if row is None or row[0] is None else int(row[0])


def _exceeded(role: str, detail: str) -> BudgetExceeded:
    """Build the error for ``role`` with one budget attempt describing the cap hit."""
    return BudgetExceeded((Attempt(BUDGET_SOURCE, Reason.BUDGET, detail),), role)
