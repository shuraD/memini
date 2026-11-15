"""Shared provider health: exhausted and resting marks kept in the shared store.

A provider is *exhausted* for one run (``run_id`` from config): it stays off for the
rest of that run whatever the clock says, and a later run does not see the mark. A
provider is *resting* until a clock time: it is off while ``clock() < until``. Both
marks live in the shared :class:`~memini.infra.llm_providers.store.Store`, so every
role and every process opening the same file skips the provider alike.

An exhausted mark wins over any resting mark, and a resting mark only ever extends:
a shorter rest recorded later never shortens a longer one already stored.
"""

from __future__ import annotations

from collections.abc import Callable

from memini.infra.llm_providers.config import DEFAULT_RUN_ID
from memini.infra.llm_providers.store import Store

SCHEMA_NAME: str = "llm_providers.health.v1"
"""Name under which the health tables are registered in the store."""

_DDL = """
CREATE TABLE IF NOT EXISTS provider_exhausted (
    run_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY (run_id, provider)
);
CREATE TABLE IF NOT EXISTS provider_resting (
    provider TEXT PRIMARY KEY,
    until REAL NOT NULL,
    reason TEXT NOT NULL
);
"""


class ProviderHealth:
    """Read and record whether providers are exhausted or resting, across processes."""

    def __init__(
        self,
        store: Store,
        clock: Callable[[], float],
        *,
        run_id: str = DEFAULT_RUN_ID,
    ) -> None:
        """Use ``store`` for the marks, ``clock`` for the current time, ``run_id`` for scope."""
        self._store = store
        self._clock = clock
        self._run_id = run_id
        store.ensure_schema(SCHEMA_NAME, _DDL)

    def mark_exhausted(self, provider: str, reason: str) -> None:
        """Turn ``provider`` off for the rest of this run, recording ``reason``."""
        self._store.execute(
            "INSERT INTO provider_exhausted (run_id, provider, reason) VALUES (?, ?, ?) "
            "ON CONFLICT (run_id, provider) DO UPDATE SET reason = excluded.reason",
            (self._run_id, provider, reason),
        )

    def mark_resting(self, provider: str, until: float, reason: str) -> None:
        """Turn ``provider`` off until clock time ``until``; never shorten a longer rest."""
        self._store.execute(
            "INSERT INTO provider_resting (provider, until, reason) VALUES (?, ?, ?) "
            "ON CONFLICT (provider) DO UPDATE SET until = excluded.until, "
            "reason = excluded.reason WHERE excluded.until > provider_resting.until",
            (provider, float(until), reason),
        )

    def available(self, provider: str) -> bool:
        """Return whether ``provider`` may be called now."""
        return self.unavailable_reason(provider) is None

    def unavailable_reason(self, provider: str) -> str | None:
        """Return the recorded reason ``provider`` is off now, or ``None`` if it is on."""
        rows = self._store.execute(
            "SELECT reason FROM provider_exhausted WHERE run_id = ? AND provider = ?",
            (self._run_id, provider),
        )
        if rows:
            return str(rows[0][0])
        rows = self._store.execute(
            "SELECT reason FROM provider_resting WHERE provider = ? AND until > ?",
            (provider, self._clock()),
        )
        return str(rows[0][0]) if rows else None
