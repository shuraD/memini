"""A stage queue line with "-" for sources has none (move MV-0003, D-10)."""

from __future__ import annotations

from pathlib import Path

from tools.orchestrator import stages


def test_a_stage_queue_line_with_a_dash_has_no_sources(tmp_path: Path) -> None:
    queue = tmp_path / "queue-spec.txt"
    queue.write_text("T-1 src/memini/mod.py - - 'a ticket text'\n")
    (item,) = stages.read_stage_queue(queue)
    assert item.sources == ()
    assert item.depends_on == ()
