"""A stage queue line with "-" for sources has none (move MV-0003, D-10)."""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.orchestrator import stages


def test_a_stage_queue_line_with_a_dash_has_no_sources(tmp_path: Path) -> None:
    queue = tmp_path / "queue-spec.txt"
    queue.write_text("T-1 src/memini/mod.py - - 'a ticket text'\n")
    (item,) = stages.read_stage_queue(queue)
    assert item.sources == ()
    assert item.depends_on == ()


def test_a_header_with_a_plain_colon_value_is_refused_and_says_why(tmp_path: Path) -> None:
    from tools.orchestrator import spec  # noqa: PLC0415

    bad = tmp_path / "SPEC.md"
    bad.write_text("---\nid: T-1\nbehavior: it reads names: of config\n---\nnote\n")
    with pytest.raises(spec.SpecError, match=r"not valid YAML \(.*mapping values"):
        spec.load(bad)


def test_the_spec_prompt_asks_for_block_scalars_for_free_text() -> None:
    text = (Path(__file__).resolve().parents[2] / "tools/orchestrator/prompts/spec.txt").read_text()
    assert "behavior: |" in text
    assert "colon followed by a space" in text
