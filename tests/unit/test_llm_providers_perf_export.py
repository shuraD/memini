"""Acceptance tests for LP-20: export the call log to a JSON Lines file."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memini.infra.llm_providers.calllog import CallLog, CallRecord
from memini.infra.llm_providers.perf_export import export_jsonl
from memini.infra.llm_providers.store import Store

FIELD_NAMES = {f.name for f in dataclasses.fields(CallRecord)}


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
        "fallbacks": (("prov-b", "TIMEOUT"), ("prov-c", "RATE_LIMIT")),
        "cache_hit": False,
        "cost": 0.00125,
        "texts": None,
        "ts": 1700000000.5,
    }
    base.update(overrides)
    return CallRecord(**base)  # type: ignore[arg-type]


def _embed(**overrides: object) -> CallRecord:
    base: dict[str, object] = {
        "kind": "embed",
        "provider": None,
        "model": None,
        "role": None,
        "stage": None,
        "question_id": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "seconds": 0.75,
        "retries": 0,
        "wait_seconds": 0.0,
        "fallbacks": (),
        "cache_hit": False,
        "cost": None,
        "texts": 16,
        "ts": 1700000001.0,
    }
    base.update(overrides)
    return CallRecord(**base)  # type: ignore[arg-type]


def _three_rows() -> list[CallRecord]:
    return [
        _generate(ts=1700000000.0),
        _generate(ts=1700000001.0, cache_hit=True, fallbacks=(), cost=None),
        _embed(ts=1700000002.0),
    ]


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    s = Store(tmp_path / "shared.sqlite")
    yield s
    s.close()


@pytest.fixture
def log(store: Store) -> CallLog:
    return CallLog(store, _clock)


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _rebuild(obj: dict[str, Any]) -> CallRecord:
    data = {name: obj[name] for name in FIELD_NAMES}
    data["fallbacks"] = tuple(tuple(pair) for pair in data["fallbacks"])
    return CallRecord(**data)


def test_three_rows_give_three_json_lines_with_every_field(log: CallLog, tmp_path: Path) -> None:
    for rec in _three_rows():
        log.record(rec)
    out = tmp_path / "trace.jsonl"

    count = export_jsonl(log, out)

    assert count == 3
    lines = _lines(out)
    assert len(lines) == 3
    for line in lines:
        obj = json.loads(line)
        assert isinstance(obj, dict)
        assert set(obj) >= FIELD_NAMES


def test_accepts_str_path(log: CallLog, tmp_path: Path) -> None:
    for rec in _three_rows():
        log.record(rec)
    out = tmp_path / "trace-str.jsonl"

    count = export_jsonl(log, str(out))

    assert count == 3
    assert len(_lines(out)) == 3


def test_empty_log_gives_empty_file_and_zero(log: CallLog, tmp_path: Path) -> None:
    out = tmp_path / "empty.jsonl"

    count = export_jsonl(log, out)

    assert count == 0
    assert out.exists()
    assert out.read_text(encoding="utf-8") == ""


def test_round_trip_rebuilds_each_record(log: CallLog, tmp_path: Path) -> None:
    records = _three_rows()
    for rec in records:
        log.record(rec)
    out = tmp_path / "trace.jsonl"

    export_jsonl(log, out)

    rebuilt = [_rebuild(json.loads(line)) for line in _lines(out)]
    assert rebuilt == records


def test_lines_follow_ts_order(log: CallLog, tmp_path: Path) -> None:
    stamps = [1700000000.0, 1700000005.5, 1700000010.25, 1700000020.0]
    for i, ts in enumerate(stamps):
        log.record(_generate(ts=ts, question_id=f"q-{i}"))
    out = tmp_path / "trace.jsonl"

    count = export_jsonl(log, out)

    assert count == len(stamps)
    objs = [json.loads(line) for line in _lines(out)]
    assert [o["ts"] for o in objs] == stamps
    assert [o["question_id"] for o in objs] == [f"q-{i}" for i in range(len(stamps))]


def test_existing_file_is_replaced_not_appended(log: CallLog, tmp_path: Path) -> None:
    out = tmp_path / "trace.jsonl"
    out.write_text('{"stale": true}\n{"stale": true}\n{"stale": true}\n', encoding="utf-8")
    log.record(_generate(ts=1700000000.0))

    count = export_jsonl(log, out)

    assert count == 1
    lines = _lines(out)
    assert len(lines) == 1
    assert "stale" not in json.loads(lines[0])


def test_second_export_replaces_first(log: CallLog, tmp_path: Path) -> None:
    out = tmp_path / "trace.jsonl"
    for rec in _three_rows():
        log.record(rec)

    export_jsonl(log, out)
    count = export_jsonl(log, out)

    assert count == 3
    assert len(_lines(out)) == 3


def test_export_has_no_prompt_or_reply_keys(log: CallLog, tmp_path: Path) -> None:
    for rec in _three_rows():
        log.record(rec)
    out = tmp_path / "trace.jsonl"

    export_jsonl(log, out)

    for line in _lines(out):
        keys = set(json.loads(line))
        assert keys == FIELD_NAMES
        for forbidden in ("prompt", "prompts", "reply", "replies", "response", "messages"):
            assert forbidden not in keys
