"""Sanity tier for the orchestrator: one pass and one fail per rule (pure, no git, no engine)."""

from __future__ import annotations

from tools.orchestrator import audit, gates


def test_path_gate_passes_a_change_inside_the_allow_list() -> None:
    assert gates.path_problems(["src/memini/core/a.py"], ["src/memini/"], []) == []


def test_path_gate_rejects_tests_config_and_protected_files() -> None:
    changed = ["tests/unit/test_a.py", "pyproject.toml", "src/memini/infra/flags.py"]
    problems = gates.path_problems(changed, ["src/memini/"], ["src/memini/infra/flags.py"])
    assert problems == [
        "path outside the allow list changed: pyproject.toml",
        "protected path changed: src/memini/infra/flags.py",
        "path outside the allow list changed: tests/unit/test_a.py",
    ]


def test_audit_fenced_json_pass_is_parsed() -> None:
    assert audit.parse('```json\n{"verdict": "pass", "findings": []}\n```').passed


def test_audit_garbage_is_a_reject() -> None:
    assert not audit.parse("LGTM").passed
