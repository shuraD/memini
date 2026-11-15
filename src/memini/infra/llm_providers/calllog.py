"""The call log: one row per real call, per cache hit and per embedding batch.

Rows live in the shared :class:`~memini.infra.llm_providers.store.Store`, so every
thread and every process opening the same file appends to, and reads, one log.
Values are always bound as parameters, so any text is stored verbatim; the fallback
chain is kept as a JSON list of ``[provider, reason]`` pairs in its original order.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from memini.infra.llm_providers.store import Store

SCHEMA_NAME: str = "llm_providers.calllog.v1"
"""Name under which the call log table is registered in the store."""

CallKind = Literal["generate", "embed"]
"""What a row records: a generation (real or cached) or an embedding batch."""

_DDL = """
CREATE TABLE IF NOT EXISTS call_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    role TEXT,
    stage TEXT,
    question_id TEXT,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    seconds REAL NOT NULL,
    retries INTEGER NOT NULL,
    wait_seconds REAL NOT NULL,
    fallbacks TEXT NOT NULL,
    cache_hit INTEGER NOT NULL,
    cost REAL,
    texts INTEGER,
    ts REAL NOT NULL
);
"""

_INSERT = (
    "INSERT INTO call_log (kind, provider, model, role, stage, question_id, "
    "prompt_tokens, completion_tokens, seconds, retries, wait_seconds, fallbacks, "
    "cache_hit, cost, texts, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_SELECT = (
    "SELECT kind, provider, model, role, stage, question_id, prompt_tokens, "
    "completion_tokens, seconds, retries, wait_seconds, fallbacks, cache_hit, cost, "
    "texts, ts FROM call_log ORDER BY id"
)


@dataclass(frozen=True)
class CallRecord:
    """One logged call; generation fields may be ``None`` for an embedding batch."""

    kind: CallKind
    provider: str | None
    model: str | None
    role: str | None
    stage: str | None
    question_id: str | None
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    retries: int
    wait_seconds: float
    fallbacks: tuple[tuple[str, str], ...]
    cache_hit: bool
    cost: float | None
    texts: int | None
    ts: float


class CallLog:
    """Append call records to the shared store and read them back in insertion order."""

    def __init__(self, store: Store, clock: Callable[[], float]) -> None:
        """Use ``store`` for the rows and ``clock`` as the time source of this log."""
        self._store = store
        self._clock = clock
        store.ensure_schema(SCHEMA_NAME, _DDL)

    def record(self, rec: CallRecord) -> None:
        """Append ``rec`` as one row."""
        self._store.execute(
            _INSERT,
            (
                rec.kind,
                rec.provider,
                rec.model,
                rec.role,
                rec.stage,
                rec.question_id,
                rec.prompt_tokens,
                rec.completion_tokens,
                rec.seconds,
                rec.retries,
                rec.wait_seconds,
                json.dumps([list(pair) for pair in rec.fallbacks]),
                int(rec.cache_hit),
                rec.cost,
                rec.texts,
                rec.ts,
            ),
        )

    def rows(self) -> list[CallRecord]:
        """Return every stored record, oldest first."""
        return [_to_record(row) for row in self._store.execute(_SELECT)]


def _to_record(row: tuple[Any, ...]) -> CallRecord:
    """Rebuild a :class:`CallRecord` from one ``call_log`` row."""
    kind: CallKind = "embed" if row[0] == "embed" else "generate"
    fallbacks = tuple((str(p), str(r)) for p, r in json.loads(row[11]))
    return CallRecord(
        kind=kind,
        provider=row[1],
        model=row[2],
        role=row[3],
        stage=row[4],
        question_id=row[5],
        prompt_tokens=int(row[6]),
        completion_tokens=int(row[7]),
        seconds=float(row[8]),
        retries=int(row[9]),
        wait_seconds=float(row[10]),
        fallbacks=fallbacks,
        cache_hit=bool(row[12]),
        cost=None if row[13] is None else float(row[13]),
        texts=None if row[14] is None else int(row[14]),
        ts=float(row[15]),
    )
