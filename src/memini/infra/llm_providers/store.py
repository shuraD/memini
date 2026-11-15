"""One shared SQLite file, safe for many threads and processes.

A file store gives each thread its own connection (WAL journal, busy timeout) and
runs every write as ``BEGIN IMMEDIATE ... COMMIT`` so concurrent writers wait on
the busy handler instead of failing. The file and its parent directory are created
on open; the file is restricted to mode 0600. ``":memory:"`` opens a private
database per store, served by one connection under a lock so threads share it.

Tables belong to later slices: they register their DDL with :meth:`Store.ensure_schema`,
which runs it once per name and remembers the name in the file itself.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Any

MEMORY: str = ":memory:"
"""Path value selecting a private in-memory database."""

FILE_MODE: int = 0o600
"""Permission bits of the database file: owner read and write only."""

# Schema names already applied live in the bookkeeping table ``_store_schema``; its
# name is written literally in the SQL below so no query is built from strings.


class Store:
    """Parameterised access to one SQLite database shared by threads and processes."""

    def __init__(self, path: str | os.PathLike[str], *, busy_timeout_ms: int = 5000) -> None:
        """Open ``path`` (creating it with mode 0600), switch it to WAL, set the timeout."""
        self._busy_timeout_ms = busy_timeout_ms
        self._lock = threading.RLock()
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._closed = False
        target = os.fspath(path)
        self._memory = target == MEMORY
        self._path = target
        self._shared: sqlite3.Connection | None = None
        if self._memory:
            self._shared = self._connect()
        else:
            file = Path(target)
            file.parent.mkdir(parents=True, exist_ok=True)
            os.close(os.open(file, os.O_CREAT | os.O_RDWR, FILE_MODE))
            file.chmod(FILE_MODE)
        conn = self._connection()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS _store_schema (name TEXT PRIMARY KEY)")

    def _connect(self) -> sqlite3.Connection:
        """Open one autocommit connection with the busy timeout applied."""
        conn = sqlite3.connect(
            self._path,
            timeout=self._busy_timeout_ms / 1000,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.execute(f"PRAGMA busy_timeout={int(self._busy_timeout_ms)}")
        with self._lock:
            self._connections.append(conn)
        return conn

    def _connection(self) -> sqlite3.Connection:
        """Return the calling thread's connection, opening it on first use."""
        if self._closed:
            raise sqlite3.ProgrammingError("store is closed")
        if self._shared is not None:
            return self._shared
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def _guard(self) -> AbstractContextManager[Any]:
        """Serialise use of the shared in-memory connection; files rely on SQLite locks."""
        return self._lock if self._memory else nullcontext()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield this thread's connection inside ``BEGIN IMMEDIATE``; commit or roll back."""
        conn = self._connection()
        with self._guard():
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        """Run one parameterised statement in its own short transaction; return its rows."""
        if sql.lstrip().upper().startswith("PRAGMA"):
            # Pragmas such as journal_mode cannot run inside a transaction.
            conn = self._connection()
            with self._guard():
                return [tuple(row) for row in conn.execute(sql, tuple(params)).fetchall()]
        with self.transaction() as conn:
            return [tuple(row) for row in conn.execute(sql, tuple(params)).fetchall()]

    def ensure_schema(self, name: str, ddl: str) -> None:
        """Run ``ddl`` once under ``name``; later calls with the same name do nothing.

        ``ddl`` may hold several statements separated by ``;``; they run in one
        transaction together with recording ``name``.
        """
        with self.transaction() as conn:
            seen = conn.execute("SELECT 1 FROM _store_schema WHERE name = ?", (name,))
            if seen.fetchone() is not None:
                return
            for statement in _statements(ddl):
                conn.execute(statement)
            conn.execute("INSERT INTO _store_schema (name) VALUES (?)", (name,))

    def close(self) -> None:
        """Close every connection this store opened; the store is unusable afterwards."""
        with self._lock:
            self._closed = True
            connections, self._connections = self._connections, []
        for conn in connections:
            conn.close()


def _statements(script: str) -> Iterator[str]:
    """Split ``script`` into complete SQL statements, respecting quoted ``;``."""
    buffer = ""
    for piece in script.split(";"):
        buffer += piece + ";"
        if sqlite3.complete_statement(buffer):
            if buffer.strip(" \t\r\n;"):
                yield buffer
            buffer = ""
    if buffer.strip(" \t\r\n;"):
        yield buffer
