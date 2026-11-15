"""Acceptance tests for LP-17: the call log table and recorder on the shared store."""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from memini.infra.llm_providers.calllog import CallLog, CallRecord
from memini.infra.llm_providers.store import Store


def _clock() -> float:
    return 1000.0


def _generate(**overrides: object) -> CallRecord:
    base: dict[str, object] = {
        "kind": "generate",
        "provider": "prov-a",
        "model": "model-x",
        "role": "extractor",
        "stage": "stage-1",
        "question_id": "q-42",
        "prompt_tokens": 120,
        "completion_tokens": 37,
        "seconds": 1.25,
        "retries": 2,
        "wait_seconds": 0.5,
        "fallbacks": (("prov-b", "TIMEOUT"),),
        "cache_hit": False,
        "cost": 0.00125,
        "texts": None,
        "ts": 1700000000.5,
    }
    base.update(overrides)
    return CallRecord(**base)  # type: ignore[arg-type]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "shared.sqlite"


@pytest.fixture
def store(db_path: Path) -> Iterator[Store]:
    s = Store(db_path)
    yield s
    s.close()


def test_call_record_is_frozen() -> None:
    rec = _generate()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.retries = 5  # type: ignore[misc]


def test_empty_log_has_no_rows(store: Store) -> None:
    assert CallLog(store, _clock).rows() == []


def test_full_record_round_trips(store: Store) -> None:
    log = CallLog(store, _clock)
    rec = _generate()
    log.record(rec)
    rows = log.rows()
    assert rows == [rec]
    got = rows[0]
    assert isinstance(got.fallbacks, tuple)
    assert all(isinstance(f, tuple) for f in got.fallbacks)
    assert isinstance(got.cache_hit, bool)
    assert got.ts == rec.ts


def test_cache_hit_record_round_trips(store: Store) -> None:
    log = CallLog(store, _clock)
    rec = _generate(
        cache_hit=True,
        prompt_tokens=0,
        completion_tokens=0,
        seconds=0.0,
        retries=0,
        wait_seconds=0.0,
        fallbacks=(),
        cost=None,
    )
    log.record(rec)
    assert log.rows() == [rec]
    assert log.rows()[0].cache_hit is True


def test_rows_persist_for_a_new_calllog_on_the_same_file(db_path: Path) -> None:
    rec = _generate()
    first = Store(db_path)
    try:
        CallLog(first, _clock).record(rec)
    finally:
        first.close()
    second = Store(db_path)
    try:
        assert CallLog(second, _clock).rows() == [rec]
    finally:
        second.close()


def test_eight_threads_recording_25_rows_each_yield_200_rows(store: Store) -> None:
    log = CallLog(store, _clock)
    errors: list[BaseException] = []

    def worker(t: int) -> None:
        try:
            for i in range(25):
                log.record(_generate(question_id=f"t{t}-r{i}", ts=float(t * 100 + i)))
        except BaseException as exc:  # pragma: no cover - surfaced by the assert
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == []
    rows = log.rows()
    assert len(rows) == 200
    assert {r.question_id for r in rows} == {f"t{t}-r{i}" for t in range(8) for i in range(25)}


def test_rows_from_two_store_handles_on_one_file_both_appear(db_path: Path) -> None:
    s1 = Store(db_path)
    s2 = Store(db_path)
    try:
        log1 = CallLog(s1, _clock)
        log2 = CallLog(s2, _clock)
        rec1 = _generate(question_id="from-one", ts=1.0)
        rec2 = _generate(question_id="from-two", ts=2.0)
        log1.record(rec1)
        log2.record(rec2)
        for log in (log1, log2):
            rows = log.rows()
            assert len(rows) == 2
            assert rec1 in rows
            assert rec2 in rows
    finally:
        s1.close()
        s2.close()


def test_embed_record_keeps_texts_and_seconds_with_none_generation_fields(
    store: Store,
) -> None:
    log = CallLog(store, _clock)
    rec = CallRecord(
        kind="embed",
        provider=None,
        model=None,
        role=None,
        stage=None,
        question_id=None,
        prompt_tokens=0,
        completion_tokens=0,
        seconds=0.75,
        retries=0,
        wait_seconds=0.0,
        fallbacks=(),
        cache_hit=False,
        cost=None,
        texts=64,
        ts=1700000123.25,
    )
    log.record(rec)
    rows = log.rows()
    assert rows == [rec]
    got = rows[0]
    assert got.kind == "embed"
    assert got.texts == 64
    assert got.seconds == 0.75
    assert got.provider is None
    assert got.model is None
    assert got.role is None
    assert got.stage is None
    assert got.question_id is None
    assert got.cost is None


def test_embed_record_with_provider_and_model_round_trips(store: Store) -> None:
    log = CallLog(store, _clock)
    rec = _generate(kind="embed", texts=3, role=None, stage=None, question_id=None)
    log.record(rec)
    assert log.rows() == [rec]


def test_fallbacks_round_trip(store: Store) -> None:
    log = CallLog(store, _clock)
    rec = _generate(fallbacks=(("A", "RATE_PER_MINUTE"),))
    log.record(rec)
    got = log.rows()[0]
    assert got.fallbacks == (("A", "RATE_PER_MINUTE"),)
    assert got == rec


def test_multiple_fallbacks_keep_their_order(store: Store) -> None:
    log = CallLog(store, _clock)
    chain = (("A", "RATE_PER_MINUTE"), ("B", "TIMEOUT"), ("C", "SERVER_ERROR"))
    log.record(_generate(fallbacks=chain))
    assert log.rows()[0].fallbacks == chain


@pytest.mark.parametrize(
    "stage",
    [
        'it\'s a "quoted" stage',
        "'; DROP TABLE x; --",
        "mixed `back` and \"double\" and 'single'",
    ],
)
def test_stage_with_quotes_is_stored_verbatim(store: Store, stage: str) -> None:
    log = CallLog(store, _clock)
    rec = _generate(stage=stage)
    log.record(rec)
    log.record(_generate(stage="after"))
    rows = log.rows()
    assert len(rows) == 2
    assert rec in rows
    assert [r.stage for r in rows if r.stage != "after"] == [stage]
