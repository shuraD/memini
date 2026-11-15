"""Acceptance tests for LP-08: the shared SQLite store (threads, processes, schema)."""

from __future__ import annotations

import os
import sqlite3
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from memini.infra.llm_providers import store as lp_store

Store = lp_store.Store

DDL = (
    "CREATE TABLE items ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, tag TEXT NOT NULL, n INTEGER NOT NULL)"
)
"""Plain CREATE TABLE (no IF NOT EXISTS): running it twice would raise."""


def _open_items(path: str | os.PathLike[str]) -> Store:
    store = Store(path)
    store.ensure_schema("items_v1", DDL)
    return store


def _raw_scalar(path: Path, sql: str) -> object:
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute(sql).fetchone()[0]
    finally:
        conn.close()


# --- opening a new path ---------------------------------------------------------------


def test_new_path_creates_parent_dir_and_file_with_mode_0600(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "shared.sqlite"
    store = Store(path)
    try:
        assert path.parent.is_dir()
        assert path.is_file()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        store.close()


def test_new_path_accepts_str_path(tmp_path: Path) -> None:
    path = tmp_path / "as_str.sqlite"
    store = Store(str(path))
    try:
        assert path.is_file()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        store.close()


def test_new_path_journal_mode_is_wal(tmp_path: Path) -> None:
    path = tmp_path / "wal.sqlite"
    store = Store(path)
    try:
        rows = store.execute("PRAGMA journal_mode")
        assert str(rows[0][0]).lower() == "wal"
    finally:
        store.close()
    # WAL is persistent in the file: an independent connection sees it too.
    assert str(_raw_scalar(path, "PRAGMA journal_mode")).lower() == "wal"


def test_busy_timeout_is_applied(tmp_path: Path) -> None:
    store = Store(tmp_path / "busy.sqlite", busy_timeout_ms=1234)
    try:
        rows = store.execute("PRAGMA busy_timeout")
        assert int(rows[0][0]) == 1234
    finally:
        store.close()


def test_default_busy_timeout_is_5000(tmp_path: Path) -> None:
    store = Store(tmp_path / "busy_default.sqlite")
    try:
        rows = store.execute("PRAGMA busy_timeout")
        assert int(rows[0][0]) == 5000
    finally:
        store.close()


# --- execute / transaction -------------------------------------------------------------


def test_execute_returns_rows_as_list_of_tuples(tmp_path: Path) -> None:
    store = _open_items(tmp_path / "rows.sqlite")
    try:
        store.execute("INSERT INTO items (tag, n) VALUES (?, ?)", ("a", 1))
        store.execute("INSERT INTO items (tag, n) VALUES (?, ?)", ("b", 2))
        rows = store.execute("SELECT tag, n FROM items ORDER BY n")
        assert isinstance(rows, list)
        assert rows == [("a", 1), ("b", 2)]
        assert all(isinstance(r, tuple) for r in rows)
    finally:
        store.close()


def test_data_persists_after_close_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "persist.sqlite"
    store = _open_items(path)
    store.execute("INSERT INTO items (tag, n) VALUES (?, ?)", ("kept", 7))
    store.close()

    reopened = _open_items(path)
    try:
        assert reopened.execute("SELECT tag, n FROM items") == [("kept", 7)]
    finally:
        reopened.close()


def test_transaction_commits_on_success(tmp_path: Path) -> None:
    path = tmp_path / "tx.sqlite"
    store = _open_items(path)
    try:
        with store.transaction() as conn:
            assert isinstance(conn, sqlite3.Connection)
            conn.execute("INSERT INTO items (tag, n) VALUES (?, ?)", ("tx", 1))
            conn.execute("INSERT INTO items (tag, n) VALUES (?, ?)", ("tx", 2))
        assert store.execute("SELECT COUNT(*) FROM items WHERE tag = ?", ("tx",)) == [(2,)]
    finally:
        store.close()
    assert _raw_scalar(path, "SELECT COUNT(*) FROM items") == 2


def test_transaction_rolls_back_on_exception(tmp_path: Path) -> None:
    store = _open_items(tmp_path / "rollback.sqlite")
    try:
        with pytest.raises(RuntimeError, match="boom"), store.transaction() as conn:
            conn.execute("INSERT INTO items (tag, n) VALUES (?, ?)", ("lost", 1))
            raise RuntimeError("boom")
        assert store.execute("SELECT COUNT(*) FROM items") == [(0,)]
    finally:
        store.close()


# --- many threads ----------------------------------------------------------------------


def test_eight_threads_fifty_rows_each_all_present(tmp_path: Path) -> None:
    store = _open_items(tmp_path / "threads.sqlite")
    errors: list[BaseException] = []
    start = threading.Barrier(8)

    def worker(index: int) -> None:
        try:
            start.wait(timeout=30)
            for n in range(50):
                if n % 2 == 0:
                    store.execute("INSERT INTO items (tag, n) VALUES (?, ?)", (f"t{index}", n))
                else:
                    with store.transaction() as conn:
                        conn.execute("INSERT INTO items (tag, n) VALUES (?, ?)", (f"t{index}", n))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        assert not any(t.is_alive() for t in threads)
        assert errors == []
        assert store.execute("SELECT COUNT(*) FROM items") == [(400,)]
        for i in range(8):
            rows = store.execute("SELECT n FROM items WHERE tag = ? ORDER BY n", (f"t{i}",))
            assert [r[0] for r in rows] == list(range(50))
    finally:
        store.close()


# --- many processes --------------------------------------------------------------------

_CHILD = """
import sys
from memini.infra.llm_providers.store import Store

path, tag, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
store = Store(path)
try:
    for n in range(count):
        store.execute("INSERT INTO items (tag, n) VALUES (?, ?)", (tag, n))
finally:
    store.close()
"""


def test_two_processes_insert_concurrently_all_rows_present_and_integrity_ok(
    tmp_path: Path,
) -> None:
    path = tmp_path / "procs.sqlite"
    _open_items(path).close()

    count = 200
    procs = [
        subprocess.Popen(  # noqa: S603 - fixed interpreter and inline script
            [sys.executable, "-c", _CHILD, str(path), f"p{i}", str(count)],
            env={**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for i in range(2)
    ]
    for proc in procs:
        _out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err.decode(errors="replace")

    store = Store(path)
    try:
        assert store.execute("SELECT COUNT(*) FROM items") == [(2 * count,)]
        for i in range(2):
            rows = store.execute("SELECT n FROM items WHERE tag = ? ORDER BY n", (f"p{i}",))
            assert [r[0] for r in rows] == list(range(count))
        assert store.execute("PRAGMA integrity_check") == [("ok",)]
    finally:
        store.close()
    assert _raw_scalar(path, "PRAGMA integrity_check") == "ok"


# --- :memory: --------------------------------------------------------------------------


def test_memory_store_works() -> None:
    store = _open_items(":memory:")
    try:
        store.execute("INSERT INTO items (tag, n) VALUES (?, ?)", ("m", 1))
        with store.transaction() as conn:
            conn.execute("INSERT INTO items (tag, n) VALUES (?, ?)", ("m", 2))
        assert store.execute("SELECT n FROM items ORDER BY n") == [(1,), (2,)]
    finally:
        store.close()


def test_memory_store_does_not_create_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    store = _open_items(":memory:")
    try:
        store.execute("INSERT INTO items (tag, n) VALUES (?, ?)", ("m", 1))
    finally:
        store.close()
    assert list(tmp_path.iterdir()) == []


def test_memory_stores_are_not_shared_between_instances() -> None:
    first = _open_items(":memory:")
    second = Store(":memory:")
    try:
        first.execute("INSERT INTO items (tag, n) VALUES (?, ?)", ("only-first", 1))
        tables = second.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", ("items",)
        )
        assert tables == []
        # The schema name is per store: the second store creates its own empty table.
        second.ensure_schema("items_v1", DDL)
        assert second.execute("SELECT COUNT(*) FROM items") == [(0,)]
        assert first.execute("SELECT COUNT(*) FROM items") == [(1,)]
    finally:
        first.close()
        second.close()


# --- parameter binding -----------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "O'Brien",
        'say "hi"',
        "'); DROP TABLE items; --",
        "1 OR 1=1",
        "x'; DELETE FROM items WHERE '1'='1",
        "back\\slash; semi;colon -- comment /* c */",
        "unicodé ☃ text",
    ],
)
def test_quotes_and_sql_text_are_stored_verbatim(tmp_path: Path, value: str) -> None:
    store = _open_items(tmp_path / "verbatim.sqlite")
    try:
        store.execute("INSERT INTO items (tag, n) VALUES (?, ?)", ("keep", 0))
        store.execute("INSERT INTO items (tag, n) VALUES (?, ?)", (value, 1))
        assert store.execute("SELECT tag FROM items WHERE n = ?", (1,)) == [(value,)]
        assert store.execute("SELECT n FROM items WHERE tag = ?", (value,)) == [(1,)]
        assert store.execute("SELECT COUNT(*) FROM items") == [(2,)]
    finally:
        store.close()


# --- ensure_schema ---------------------------------------------------------------------


def test_ensure_schema_creates_table(tmp_path: Path) -> None:
    store = Store(tmp_path / "schema.sqlite")
    try:
        store.ensure_schema("items_v1", DDL)
        tables = store.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", ("items",)
        )
        assert tables == [("items",)]
    finally:
        store.close()


def test_ensure_schema_second_call_with_same_name_does_nothing(tmp_path: Path) -> None:
    store = Store(tmp_path / "schema_twice.sqlite")
    try:
        store.ensure_schema("items_v1", DDL)
        store.execute("INSERT INTO items (tag, n) VALUES (?, ?)", ("kept", 1))
        # Same name again: the DDL is not run (it would raise: the table exists),
        # and a different DDL under the same name is not run either.
        store.ensure_schema("items_v1", DDL)
        store.ensure_schema("items_v1", "CREATE TABLE other_thing (x INTEGER)")
        assert store.execute("SELECT tag, n FROM items") == [("kept", 1)]
        others = store.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("other_thing",),
        )
        assert others == []
    finally:
        store.close()


def test_ensure_schema_is_remembered_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "schema_reopen.sqlite"
    store = Store(path)
    store.ensure_schema("items_v1", DDL)
    store.execute("INSERT INTO items (tag, n) VALUES (?, ?)", ("kept", 1))
    store.close()

    reopened = Store(path)
    try:
        reopened.ensure_schema("items_v1", DDL)
        assert reopened.execute("SELECT tag, n FROM items") == [("kept", 1)]
    finally:
        reopened.close()


def test_ensure_schema_new_name_runs_its_ddl(tmp_path: Path) -> None:
    store = Store(tmp_path / "schema_versions.sqlite")
    try:
        store.ensure_schema("items_v1", DDL)
        store.ensure_schema("items_v2", "ALTER TABLE items ADD COLUMN note TEXT")
        store.execute("INSERT INTO items (tag, n, note) VALUES (?, ?, ?)", ("a", 1, "x"))
        assert store.execute("SELECT note FROM items") == [("x",)]
    finally:
        store.close()
