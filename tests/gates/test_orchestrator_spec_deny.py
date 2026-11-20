"""The spec stage refuses a hidden test with a denied word at once (GR-12), not at the code stage.

Rule: the spec stage scans every hidden test the QA wrote with the same deny list the prompt
shows (the shipped deny patterns minus ``pytest``), plus IP addresses, host names and home
paths. A hit becomes a problem in the stage's one retry; a second hit rejects the task.
"""

from __future__ import annotations

from typing import Any

import pytest
from test_orchestrator_stages import FAILING, ID, _cfg, _enqueue, _line, _plan, _spec_md, env
from tools.orchestrator import specstage, stages

__all__ = ["env"]  # the shared fixture

TEST = "tests/unit/test_calc_hidden.py"
SOCKET = r"\bsocket\b"
MESSAGE = f"{TEST}: deny word {SOCKET}: reword the test without it"


def _run(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, *texts: str) -> tuple[Any, int]:
    cfg = _cfg(env)
    _enqueue(cfg, _line(env))
    steps = [{"SPEC.md": _spec_md(), TEST: text} for text in texts]
    _plan(env, monkeypatch, *steps)
    return cfg, specstage.run(cfg, cfg.stages[0], ID)


def _reason(cfg: Any) -> str:
    return str(stages.StageState(cfg, "spec").records()[ID]["reason"])


def _sealed(cfg: Any) -> bool:
    return bool((cfg.holdout_dir / ID / "test_calc_hidden.py").exists())


def test_a_clean_hidden_test_passes(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """A test without any denied word is sealed on the first attempt."""
    cfg, code = _run(env, monkeypatch, FAILING)
    assert code == 0
    assert _sealed(cfg)
    assert not (env["tmp"] / "log" / "qa-2.txt").exists()


def test_a_denied_word_is_refused_with_the_file_and_the_pattern(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """GR-12: `socket` in a hidden test is refused at the spec stage, naming file and pattern."""
    bad = FAILING + "\n\ndef test_net():\n    import socket\n    assert socket\n"
    cfg, code = _run(env, monkeypatch, bad, bad)
    assert code == 1
    assert MESSAGE in _reason(cfg)
    assert not _sealed(cfg)


@pytest.mark.parametrize(
    "extra",
    [
        "# open a socket here\n",
        'NOTE = "a socket"\n',
        '"""A socket docstring."""\n',
    ],
    ids=["comment", "string", "docstring"],
)
def test_the_denied_word_in_a_comment_or_a_string_is_refused_too(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, extra: str
) -> None:
    """The scan is textual: hiding the word in a comment or a string does not pass."""
    cfg, code = _run(env, monkeypatch, FAILING + "\n" + extra, FAILING + "\n" + extra)
    assert code == 1
    assert MESSAGE in _reason(cfg)


def test_the_word_pytest_is_allowed(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests import pytest: it is removed from the scanned list."""
    head = "import pytest\nfrom memini.core.calc import add\n\n\n"
    body = '@pytest.mark.parametrize("x", [1])\ndef test_add(x):\n    assert add(x, 2) == 3\n'
    text = head + body
    cfg, code = _run(env, monkeypatch, text)
    assert code == 0
    assert _sealed(cfg)


_IP = ".".join(("10", "20", "30", "40"))
_USERS = "/" + "Users" + "/someone"
_HOME = "/" + "home" + "/someone"
_HOST = ".".join(("db", "example", "com"))


@pytest.mark.parametrize(
    ("extra", "needle"),
    [
        (f"HOST = '{_IP}'\n", f"IP address: {_IP}"),
        (f"HOME = '{_USERS}/work'\n", f"home path: {_USERS}"),
        (f"HOME = '{_HOME}/work'\n", f"home path: {_HOME}"),
        (f"URL = '{_HOST}'\n", f"host name: {_HOST}"),
    ],
)
def test_an_ip_address_host_or_home_path_is_refused(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, extra: str, needle: str
) -> None:
    """Owner rule: no literal IPs, hosts or home paths in tests."""
    text = FAILING + "\n" + extra
    cfg, code = _run(env, monkeypatch, text, text)
    assert code == 1
    assert f"{TEST}: {needle}: reword the test without it" in _reason(cfg)


def test_a_refused_first_attempt_is_retried_and_the_second_without_the_word_is_accepted(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retry prompt names the problem; a clean second attempt is sealed."""
    cfg, code = _run(env, monkeypatch, FAILING + "\n# socket\n", FAILING)
    assert code == 0
    second = (env["tmp"] / "log" / "qa-2.txt").read_text()
    assert "Fix only these problems" in second
    assert MESSAGE in second
    assert (cfg.holdout_dir / ID / "test_calc_hidden.py").read_text() == FAILING


def test_two_attempts_with_the_word_end_the_stage_rejected_with_the_reason(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only one retry: a second hit rejects the task, records the reason, seals nothing."""
    bad = FAILING + "\n# socket\n"
    cfg, code = _run(env, monkeypatch, bad, bad)
    assert code == 1
    assert stages.StageState(cfg, "spec").status(ID) == specstage.REJECTED
    assert MESSAGE in _reason(cfg)
    assert not _sealed(cfg)
    assert not cfg.queue.exists()
    assert "spec stage rejected" in cfg.inbox.read_text() or MESSAGE in cfg.inbox.read_text()


def test_every_hidden_test_file_is_scanned_not_only_the_first(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A denied word in the second of two test files is refused, naming that file."""
    cfg = _cfg(env)
    _enqueue(cfg, _line(env))
    other = "tests/unit/test_calc_other.py"
    bad = FAILING + "\n# socket\n"
    step = {"SPEC.md": _spec_md(), TEST: FAILING, other: bad}
    _plan(env, monkeypatch, step, step)
    assert specstage.run(cfg, cfg.stages[0], ID) == 1
    assert f"{other}: deny word {SOCKET}: reword the test without it" in _reason(cfg)
    assert MESSAGE not in _reason(cfg)


def test_a_lint_error_alone_is_reported_as_lint(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control for the order test: the lint check does reject this file."""
    lint_only = "import os\n" + FAILING
    cfg, code = _run(env, monkeypatch, lint_only, lint_only)
    assert code == 1
    assert "fail lint" in _reason(cfg)


def test_the_deny_scan_runs_before_lint(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file with a denied word and a lint error reports the denied word, not the lint output."""
    both = "import os\n" + FAILING + "\n# socket\n"
    cfg, code = _run(env, monkeypatch, both, both)
    assert code == 1
    assert MESSAGE in _reason(cfg)
    assert "fail lint" not in _reason(cfg)


def test_the_prompt_deny_list_equals_the_scanned_list(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One list for both: the prompt shows every shipped deny pattern but pytest, and each of
    those patterns is also what the scan refuses (checked on the prompt text of a real run)."""
    cfg = _cfg(env)
    shown = "  ".join(p for p in cfg.deny_patterns if p != r"\bpytest\b")
    _enqueue(cfg, _line(env))
    _plan(env, monkeypatch, {"SPEC.md": _spec_md(), TEST: FAILING})
    assert specstage.run(cfg, cfg.stages[0], ID) == 0
    prompt = (env["tmp"] / "log" / "qa-1.txt").read_text()
    assert f"- denied regexes: {shown}\n" in prompt  # the line of the tests section
    assert SOCKET in shown
    assert r"\bpytest\b" not in shown
