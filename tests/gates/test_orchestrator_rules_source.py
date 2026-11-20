"""One source of truth for the rules: engine.rules_text(cfg) feeds every agent prompt.

Rules: the rule text is built only from cfg values (no stale copy); the same block reaches the
spec, qa, coder, audit and test_review prompts and not the verifier; the spec stage refuses a
denied word in the spec's `interface:` block only, and that refusal flows into the one retry.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from test_orchestrator_stages import FAILING, ID, _cfg, _enqueue, _line, _plan, _spec_md, env
from tools.orchestrator import config, engine, specstage, stages

__all__ = ["env"]  # the shared fixture

CFG = config.load_config(Path.cwd(), overrides={"state_dir": "/nonexistent-state"}, env={})
PYTEST = r"\bpytest\b"
TEST = "tests/unit/test_calc_hidden.py"
SOCKET = r"\bsocket\b"
IFACE = "memini.core.calc.add(a: int, b: int) -> int, opens a socket"
MESSAGE = (
    f"SPEC.md interface: deny word {SOCKET}: "
    "the coder may not write this, so the spec may not require it"
)
FIELDS = [
    "allow",
    "protected",
    "deny_patterns",
    "ast_forbidden_calls",
    "ast_dunder_allow",
    "import_allow",
    "forbidden_names",
    "graph_name_paths",
    "qa_allow",
    "old_test_paths",
]


def _with(**values: tuple[str, ...]) -> config.Config:
    return dataclasses.replace(CFG, **values)  # type: ignore[arg-type]


def _section(text: str, head: str, tail: str | None) -> str:
    start = text.index(head)
    return text[start : text.index(tail) if tail else None]


def _code(text: str) -> str:
    return _section(text, "Code (", "Tests (")


def _tests(text: str) -> str:
    return text[text.index("Tests (") :]


@pytest.mark.parametrize("field", FIELDS)
def test_rules_text_contains_every_shipped_value(field: str) -> None:
    """Every value of the shipped config appears in the rules text."""
    values = getattr(CFG, field)
    assert values, f"the shipped config has no {field}: the test would prove nothing"
    text = engine.rules_text(CFG)
    for value in values:
        assert value in text, f"{field}: {value}"


@pytest.mark.parametrize("field", FIELDS)
def test_rules_text_follows_the_config(field: str) -> None:
    """One source: a changed cfg value shows in the text and the old value's marker is gone."""
    marker = f"zz-marker-{field}-zz"
    changed = _with(**{field: (marker,)})
    text = engine.rules_text(changed)
    assert marker in text
    assert text != engine.rules_text(CFG)


def test_no_shipped_value_survives_when_every_config_value_changes() -> None:
    """No stale copy: with every list replaced, no shipped value is left anywhere in the text."""
    swapped = {f: (f"zz-{f}-zz",) for f in FIELDS}
    text = engine.rules_text(_with(**swapped))
    for field in FIELDS:
        assert f"zz-{field}-zz" in text
        for old in getattr(CFG, field):
            if len(old) > 8:  # shorter values occur inside the fixed prose
                assert old not in text, f"stale copy of {field}: {old}"


def test_the_code_section_denies_pytest_and_the_tests_section_allows_it() -> None:
    """The word pytest is denied in code and allowed in tests, each said in its own section."""
    text = engine.rules_text(CFG)
    code, tests = _code(text), _tests(text)
    assert PYTEST in CFG.deny_patterns
    assert PYTEST in code
    assert "denied in code" in code
    assert "allowed in tests" in code
    assert PYTEST not in tests
    for pattern in engine.test_deny(CFG):
        assert pattern in tests


def test_test_deny_is_the_deny_list_minus_pytest() -> None:
    """engine.test_deny drops exactly the pytest pattern and keeps the order of the rest."""
    assert engine.test_deny(CFG) == [p for p in CFG.deny_patterns if p != PYTEST]
    only = _with(deny_patterns=(PYTEST, SOCKET, PYTEST))
    assert engine.test_deny(only) == [SOCKET]


def test_an_empty_list_does_not_leave_an_empty_rule_line() -> None:
    """A config with no forbidden names still renders a readable rule, not a bare colon."""
    text = engine.rules_text(_with(forbidden_names=()))
    assert "- forbidden names: \n" not in text
    assert not text.endswith(": ")


@pytest.mark.parametrize("name", ["spec", "qa", "coder", "audit", "test_review"])
def test_the_same_rules_block_is_in_every_prompt(name: str) -> None:
    """The identical rules text appears verbatim in the rendered prompt of each agent."""
    rules = engine.rules_text(CFG)
    assert rules in engine.render(CFG, name, rules=rules)


@pytest.mark.parametrize("name", ["spec", "qa", "coder", "audit", "test_review"])
def test_the_rules_block_is_a_single_placeholder(name: str) -> None:
    """The prompt text holds $rules once and none of the old duplicated rule placeholders."""
    text = (Path("tools/orchestrator/prompts") / f"{name}.txt").read_text(encoding="utf-8")
    assert text.count("$rules") == 1
    for old in (
        "$deny_patterns",
        "$ast_calls",
        "$ast_dunders",
        "$imports",
        "$forbidden_names",
        "$graph_paths",
        "$protected_paths",
        "$qa_allow",
        "$allow)",
    ):
        assert old not in text, old


def test_the_verifier_prompt_has_no_rules_block() -> None:
    """The verifier checks card lines only: no rules text and no placeholder for it."""
    text = (Path("tools/orchestrator/prompts/verifier.txt")).read_text(encoding="utf-8")
    rules = engine.rules_text(CFG)
    assert "$rules" not in text
    rendered = engine.render(CFG, "verifier", rules=rules, cards="c", diff="d", tests="t")
    assert "denied regexes" not in rendered
    assert rules.splitlines()[0] not in rendered


def test_the_spec_stage_prompt_carries_the_rules_text(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prompt the spec stage really sends holds the rules text verbatim."""
    cfg = _cfg(env)
    _enqueue(cfg, _line(env))
    _plan(env, monkeypatch, {"SPEC.md": _spec_md(), TEST: FAILING})
    assert specstage.run(cfg, cfg.stages[0], ID) == 0
    prompt = (env["tmp"] / "log" / "qa-1.txt").read_text()
    assert engine.rules_text(cfg) in prompt


def _run(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, *specs: str) -> tuple[Any, int]:
    cfg = _cfg(env)
    _enqueue(cfg, _line(env))
    _plan(env, monkeypatch, *[{"SPEC.md": s, TEST: FAILING} for s in specs])
    return cfg, specstage.run(cfg, cfg.stages[0], ID)


def _reason(cfg: Any) -> str:
    return str(stages.StageState(cfg, "spec").records()[ID]["reason"])


@pytest.mark.parametrize("word", ["socket", "an open socket"])
def test_a_denied_word_in_the_interface_is_refused(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, word: str
) -> None:
    """The spec may not require what the coder may not write: interface with `socket` refused."""
    bad = _spec_md(interface=f"memini.core.calc.add(a, b) -> int  # {word}")
    cfg, code = _run(env, monkeypatch, bad, bad)
    assert code == 1
    assert MESSAGE in _reason(cfg)


@pytest.mark.parametrize("field", ["behavior", "acceptance", "out_of_scope"])
def test_a_denied_word_outside_the_interface_is_not_refused(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Only the interface block is scanned: the same word in another header field passes."""
    value = "- Given 1 and 2, Then add returns 3 without a socket.\n- done: add is exported."
    if field != "acceptance":
        value = "never opens a socket"
    cfg, code = _run(env, monkeypatch, _spec_md(**{field: value}))
    assert code == 0, _reason(cfg) if code else ""


def test_the_spec_body_may_contain_a_denied_word(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scan covers the interface block, not the free text after the header."""
    text = _spec_md() + "\nNo socket is used here.\n"
    cfg, code = _run(env, monkeypatch, text)
    assert code == 0, _reason(cfg) if code else ""


def test_the_interface_refusal_flows_into_the_one_retry(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """First attempt refused for the interface; the retry prompt names it; a clean spec passes."""
    _, code = _run(env, monkeypatch, _spec_md(interface=IFACE), _spec_md())
    assert code == 0
    second = (env["tmp"] / "log" / "qa-2.txt").read_text()
    assert "Fix only these problems" in second
    assert MESSAGE in second


def test_a_second_interface_refusal_rejects_the_task(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only one retry: two refused interfaces reject the task with the reason."""
    bad = _spec_md(interface=IFACE)
    cfg, code = _run(env, monkeypatch, bad, bad)
    assert code == 1
    assert stages.StageState(cfg, "spec").status(ID) == specstage.REJECTED
    assert MESSAGE in _reason(cfg)


def test_pytest_in_the_interface_is_not_refused(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interface scan uses the tests' list: the word pytest is allowed there."""
    ok = _spec_md(interface="memini.core.calc.add(a: int, b: int) -> int  # checked by pytest")
    cfg, code = _run(env, monkeypatch, ok)
    assert code == 0, _reason(cfg) if code else ""
