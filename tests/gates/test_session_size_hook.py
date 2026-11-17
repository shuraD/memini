"""The session-size hook (FLOW-HARD.md D9): above 80K tokens, say to start a new session."""

from __future__ import annotations

import json
from pathlib import Path

from tools.orchestrator.hooks import session_size


def _transcript(tmp_path: Path, *rows: dict[str, object]) -> Path:
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _turn(tokens: int) -> dict[str, object]:
    return {"message": {"usage": {"input_tokens": 10, "cache_read_input_tokens": tokens}}}


def test_a_large_session_is_told_to_end(tmp_path: Path) -> None:
    path = _transcript(tmp_path, _turn(5_000), _turn(90_000))
    text = session_size.notice({"transcript_path": str(path)})
    assert text.startswith("Write your state file and end this session; start a new one.")
    assert "90K" in text


def test_a_small_session_hears_nothing(tmp_path: Path) -> None:
    assert session_size.notice({"transcript_path": str(_transcript(tmp_path, _turn(30_000)))}) == ""


def test_without_usage_the_size_is_estimated_from_the_file(tmp_path: Path) -> None:
    path = _transcript(tmp_path, {"text": "x" * 400_000})
    assert session_size.context_tokens(path) > session_size.LIMIT
    assert session_size.notice({"transcript_path": str(path)}) != ""


def test_a_missing_transcript_or_bad_event_never_blocks(tmp_path: Path) -> None:
    assert session_size.notice({"transcript_path": str(tmp_path / "none.jsonl")}) == ""
    assert session_size.notice({}) == ""
    assert session_size.notice("garbage") == ""
