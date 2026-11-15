"""A failed lint, type or security gate tells the coder the rule and line, never the message.

Owner 2026-11-15 (LP-06): the coder saw only "gate security failed" and guessed wrong twice.
It now gets "<rule> <path>:<line>" for findings in the files it may change; the message text
and every other output line stay in the gate log (review B3(b)).
"""

from __future__ import annotations

import pytest
from tools.orchestrator import gates

ALLOW = ["src/memini/"]
RUFF_FULL = "E501 Line too long (104 > 100)\n  --> src/memini/a.py:27:101\n   |\n"
RUFF_CONCISE = "src/memini/a.py:3:1: F401 [*] `os` imported but unused\n"
RUFF_FORMAT = "unformatted: File would be reformatted\n  --> src/memini/a.py:50:10\n"
BANDIT = (
    ">> Issue: [B310:blacklist] Audit url open for permitted schemes.\n"
    "   Severity: Medium   Confidence: High\n   CWE: CWE-22\n   More Info: https://x\n"
    "   Location: ./src/memini/t.py:86:21\n"
)
MYPY = "src/memini/a.py:12: error: Incompatible types in assignment  [assignment]\n"


@pytest.mark.parametrize(
    ("output", "finding"),
    [
        (RUFF_FULL, "E501 src/memini/a.py:27"),
        (RUFF_CONCISE, "F401 src/memini/a.py:3"),
        (RUFF_FORMAT, "unformatted src/memini/a.py:50"),
        (BANDIT, "B310 src/memini/t.py:86"),
        (MYPY, "assignment src/memini/a.py:12"),
    ],
)
def test_each_tool_gives_its_rule_and_location(output: str, finding: str) -> None:
    assert gates.gate_findings(output, ALLOW) == (finding,)


def test_no_message_text_and_no_file_outside_the_coders_paths() -> None:
    sealed = "I001 [*] Import block is un-sorted\n  --> tests/unit/test_x.py:3:1\n"
    found = gates.gate_findings(RUFF_FULL + BANDIT + sealed, ALLOW)
    assert found == ("E501 src/memini/a.py:27", "B310 src/memini/t.py:86")
    text = gates.Failure("gate lint failed", (), found).summary()
    assert text == "gate lint failed; findings: E501 src/memini/a.py:27, B310 src/memini/t.py:86"
    assert "Line too long" not in text
    assert "Audit url" not in text


def test_findings_are_capped() -> None:
    many = "".join(f"src/memini/a.py:{n}:1: E501 x\n" for n in range(1, 40))
    found = gates.gate_findings(many, ALLOW)
    assert len(found) == 20
    assert gates.Failure("gate lint failed", (), (*found, "x")).summary().count("E501") == 20
