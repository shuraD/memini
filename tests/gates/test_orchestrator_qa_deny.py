"""QA is told which words its tests may not contain before it writes them (owner 2026-11-18)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from tools.orchestrator import config, engine, gates

CFG = config.load_config(Path.cwd(), overrides={"state_dir": "/nonexistent-state"}, env={})


@pytest.mark.parametrize("name", ["spec", "qa"])
def test_qa_prompt_lists_the_denied_words(name: str) -> None:
    text = engine.render(CFG, name, deny_patterns="importlib-marker")
    assert "importlib-marker" in text


def test_every_qa_render_passes_the_list() -> None:
    for module in ("specstage.py", "pipeline.py"):
        source = Path("tools/orchestrator", module).read_text(encoding="utf-8")
        assert re.search(r"deny_patterns=", source)


def test_the_listed_words_are_the_ones_the_gate_refuses() -> None:
    listed = [p for p in CFG.deny_patterns if p != r"\bpytest\b"]
    assert gates.text_problems("import importlib", listed)
    assert not gates.text_problems("import pytest", listed)
