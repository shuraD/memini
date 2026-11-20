"""QA is told which words its tests may not contain before it writes them (owner 2026-11-18)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from tools.orchestrator import config, engine, gates

CFG = config.load_config(Path.cwd(), overrides={"state_dir": "/nonexistent-state"}, env={})


@pytest.mark.parametrize("name", ["spec", "qa"])
def test_qa_prompt_lists_the_denied_words(name: str) -> None:
    """Rule: QA is told every denied word (minus pytest) before writing, through $rules."""
    text = engine.render(CFG, name, rules=engine.rules_text(CFG))
    for pattern in engine.test_deny(CFG):
        assert pattern in text


def test_every_qa_render_passes_the_list() -> None:
    """Rule: both renders of a QA prompt (spec stage, pipeline) pass the one rules text."""
    for module in ("specstage.py", "pipeline.py"):
        source = Path("tools/orchestrator", module).read_text(encoding="utf-8")
        assert re.search(r"rules=engine\.rules_text\(", source)


def test_the_listed_words_are_the_ones_the_gate_refuses() -> None:
    listed = [p for p in CFG.deny_patterns if p != r"\bpytest\b"]
    assert gates.text_problems("import importlib", listed)
    assert not gates.text_problems("import pytest", listed)


def test_coder_prompt_lists_the_ast_and_graph_name_rules() -> None:
    """Rule: the coder sees the AST calls, imports, names and graph paths through $rules."""
    text = engine.render(CFG, "coder", rules=engine.rules_text(CFG))
    expected = [
        *CFG.ast_forbidden_calls,
        *CFG.import_allow,
        *CFG.forbidden_names,
        *CFG.graph_name_paths,
    ]
    assert all(m in text for m in expected)
