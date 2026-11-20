"""Card lines in the spec stage and the CI wait (owner rule: every card line a ticket names
is stated, defined as done and tested; ``ci-wait`` finds a run by the full sha).

Fixtures follow ``test_orchestrator_stages.py`` (tmp repo, fake engine) and
``test_orchestrator_enqueue.py`` (card files).
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
from tests.gates import test_orchestrator_stages
from tests.gates.test_orchestrator_enqueue import CAPS, CARD, OTHER
from tests.gates.test_orchestrator_stages import (
    FAILING,
    ID,
    MODULE,
    _cfg,
    _enqueue,
    _git,
    _plan,
    _spec_md,
)
from tools.orchestrator import cli, enqueue, specstage, stages

env = test_orchestrator_stages.env  # the shared tmp-repo fixture

TWO_CARDS = "- A1: Given 1 and 2, Then add returns 3.\n- A1 done: add is exported.\n"
BOTH = TWO_CARDS + "- A2: Given a node, Then it is deleted.\n- A2 done: the node is gone.\n"
CARD_TWELVE = CARD.replace("| A3 |", "| A12 |").replace("A different", "A twelfth")


def _cards(env: dict[str, Any]) -> dict[str, Any]:
    """Write the card files next to the repo; return the config overrides that point at them."""
    sq = env["tmp"] / "squads"
    (sq / "alpha").mkdir(parents=True)
    (sq / "alpha" / "CARD.md").write_text(CARD_TWELVE)
    (sq / "beta").mkdir()
    (sq / "beta" / "CARD.md").write_text("## IN: baseline\n\n" + OTHER)
    (env["tmp"] / "capabilities").mkdir()
    (env["tmp"] / "capabilities" / "archrag.md").write_text(CAPS)
    return {"cards_dir": str(sq)}


def _ticket(env: dict[str, Any], text: str) -> str:
    return f"{ID} {MODULE} - {env['source']} '{text}'\n"


def _run_spec(
    env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    files: dict[str, str],
) -> tuple[int, str]:
    cfg = _cfg(env, **_cards(env))
    _enqueue(cfg, _ticket(env, text))
    _plan(env, monkeypatch, files)
    code = specstage.run(cfg, cfg.stages[0], ID)
    reason = stages.StageState(cfg, "spec").records()[ID].get("reason", "")
    return code, str(reason)


# ------------------------------------------------------------------------------ card_text
def test_card_text_returns_the_ids_and_one_line_per_card_with_old_code_evidence(
    env: dict[str, Any],
) -> None:
    """The text every agent reads: `A1: <card line> (old code: <evidence>)`, one per card."""
    cfg = _cfg(env, **_cards(env))
    ids, text = enqueue.card_text(cfg, "Build A1 and A2 now")
    assert ids == ["A1", "A2"]
    assert text.splitlines() == [
        "A1: The store writes a node. (old code: old/store.py:write L10-20)",
        "A2: The store deletes a node. (old code: )",
    ]


def test_card_text_skips_unknown_and_non_baseline_ids_and_is_empty_without_cards(
    env: dict[str, Any],
) -> None:
    """A9 is under OUT (not baseline) and A99 does not exist: neither gets a line."""
    cfg = _cfg(env, **_cards(env))
    assert enqueue.card_text(cfg, "A9 and A99 only") == ([], "")
    assert enqueue.card_text(cfg, "no card named here") == ([], "")


def test_card_text_expands_a_range_to_the_known_lines_only(
    env: dict[str, Any],
) -> None:
    """`A1 to A12` is every known line between (A3 is beta's); A9 is not baseline."""
    cfg = _cfg(env, **_cards(env))
    ids, text = enqueue.card_text(cfg, "A1 to A12")
    assert ids == ["A1", "A2", "A3", "A12"]
    assert len(text.splitlines()) == 4


# ------------------------------------------------------------------------------ spec lines
def test_a_spec_with_a_criteria_line_and_a_done_line_per_card_is_accepted(
    env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Good case: each named card has `- A1:` and `- A1 done:` in the acceptance header."""
    code, reason = _run_spec(
        env,
        monkeypatch,
        "Build A1 and A2",
        {
            "SPEC.md": _spec_md(acceptance=BOTH),
            "tests/unit/test_calc_hidden.py": FAILING + "\n\n# A1 and A2\n",
        },
    )
    assert code == 0, reason


@pytest.mark.parametrize(
    ("acceptance", "missing"),
    [
        ("- A1 done: add is exported.\n", "- A1:"),
        ("- A1: Given 1 and 2, Then add returns 3.\n", "- A1 done:"),
        ("- Given 1 and 2, Then add returns 3.\n", "- A1:"),
        ("A1: not a list item\nA1 done: not a list item\n", "- A1:"),
        ("- A1 is mentioned in prose, A1 done too.\n", "- A1:"),
        ("- A1 : spaced\n- A1 done : spaced\n", "- A1:"),
    ],
)
def test_a_spec_missing_a_card_line_or_its_done_line_is_refused(
    env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    acceptance: str,
    missing: str,
) -> None:
    """Rule: for each card the ticket names, both `- A1:` and `- A1 done:` list lines exist."""
    code, reason = _run_spec(
        env,
        monkeypatch,
        "Build A1",
        {
            "SPEC.md": _spec_md(acceptance=acceptance),
            "tests/unit/test_calc_hidden.py": FAILING + "\n\n# A1\n",
        },
    )
    assert code == 1
    assert missing in reason


def test_the_second_card_cannot_hide_behind_the_first(
    env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every named card needs its own pair: A1 complete, A2 absent -> refused naming A2 only."""
    code, reason = _run_spec(
        env,
        monkeypatch,
        "Build A1 and A2",
        {
            "SPEC.md": _spec_md(acceptance=TWO_CARDS),
            "tests/unit/test_calc_hidden.py": FAILING + "\n\n# A1 A2\n",
        },
    )
    assert code == 1
    assert "A2" in reason
    assert "for card line A1" not in reason


def test_a_done_line_for_a_card_prefix_does_not_satisfy_a_longer_id(
    env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A1 lines must not satisfy A12 (and A12 lines must not satisfy A1)."""
    code, reason = _run_spec(
        env,
        monkeypatch,
        "Build A12",
        {
            "SPEC.md": _spec_md(acceptance=TWO_CARDS),
            "tests/unit/test_calc_hidden.py": FAILING + "\n\n# A12\n",
        },
    )
    assert code == 1
    assert "A12" in reason


def test_a_ticket_naming_no_card_needs_the_done_line_and_ac_items(
    env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No card named: no card lines, but one `- done:` line and `- AC<n>:` per Acceptance item."""
    text = "Add two numbers. Context: c. Acceptance: add returns 3. Done: add is exported."
    code, reason = _run_spec(
        env,
        monkeypatch,
        text,
        {
            "SPEC.md": _spec_md(acceptance="- AC1: Given 1 and 2, Then 3.\n- done: exported."),
            "tests/unit/test_calc_hidden.py": FAILING + "\n\n# AC1\n",
        },
    )
    assert code == 0, reason


def test_the_spec_prompt_carries_the_card_lines(
    env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The spec QA is given the exact card text (`$cards`) from `card_text`."""
    _run_spec(
        env,
        monkeypatch,
        "Build A1",
        {
            "SPEC.md": _spec_md(acceptance=TWO_CARDS),
            "tests/unit/test_calc_hidden.py": FAILING + "\n\n# A1\n",
        },
    )
    prompt = (env["tmp"] / "log" / "qa-1.txt").read_text()
    assert "A1: The store writes a node. (old code: old/store.py:write L10-20)" in prompt


# ------------------------------------------------------------------------------ test cover
@pytest.mark.parametrize(
    "mention",
    [
        "def test_a1_writes_the_node():\n    assert True\n",
        "# covers A1\n",
        "# covers a1\n",
        "def test_A1_writes():\n    assert True\n",
        "# (A1)\n",
    ],
)
def test_a_hidden_test_naming_the_card_id_in_any_case_covers_it(
    env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    mention: str,
) -> None:
    """Good case: the id appears whole, in any case, in a hidden test."""
    code, reason = _run_spec(
        env,
        monkeypatch,
        "Build A1",
        {
            "SPEC.md": _spec_md(acceptance=TWO_CARDS),
            "tests/unit/test_calc_hidden.py": FAILING + "\n\n" + mention,
        },
    )
    assert code == 0, reason


@pytest.mark.parametrize(
    "text",
    [
        "# nothing here\n",
        "# covers A12\n",  # A1 must not match A12
        "# covers A10\n",
        "# covers XA1\n",  # no id boundary on the left
        "# covers A_1 and A 1\n",
    ],
)
def test_a_hidden_test_that_does_not_name_the_exact_card_id_is_refused(
    env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    text: str,
) -> None:
    """Rule: the id must appear as a whole id; A1 is not covered by A12, A10 or XA1."""
    code, reason = _run_spec(
        env,
        monkeypatch,
        "Build A1",
        {
            "SPEC.md": _spec_md(acceptance=TWO_CARDS),
            "tests/unit/test_calc_hidden.py": FAILING + "\n\n" + text,
        },
    )
    assert code == 1
    assert "no hidden test covers card line A1" in reason


def test_the_card_may_be_named_in_a_different_hidden_test_file(
    env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The id may be in any one of the hidden files (A1 in one, A2 in the other)."""
    code, reason = _run_spec(
        env,
        monkeypatch,
        "Build A1 and A2",
        {
            "SPEC.md": _spec_md(acceptance=BOTH),
            "tests/unit/test_one.py": FAILING + "\n\n# A1\n",
            "tests/unit/test_two.py": FAILING.replace("test_add", "test_b") + "\n\n# A2\n",
        },
    )
    assert code == 0, reason


def test_every_card_must_be_covered_not_just_one(
    env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A1 named, A2 not: refused for A2."""
    code, reason = _run_spec(
        env,
        monkeypatch,
        "Build A1 and A2",
        {
            "SPEC.md": _spec_md(acceptance=BOTH),
            "tests/unit/test_calc_hidden.py": FAILING + "\n\n# A1\n",
        },
    )
    assert code == 1
    assert "card line A2" in reason
    assert "card line A1" not in reason


def test_the_card_check_runs_first_so_an_uncovered_card_hides_no_other_problem(
    env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rule: the uncovered-card refusal comes before the other test checks.

    A test file with no test function AND no card id reports the card problem.
    """
    code, reason = _run_spec(
        env,
        monkeypatch,
        "Build A1",
        {
            "SPEC.md": _spec_md(acceptance=TWO_CARDS),
            "tests/unit/test_c.py": "import os\n",
        },
    )
    assert code == 1
    assert "no hidden test covers card line A1" in reason
    assert "no test function" not in reason


# ------------------------------------------------------------------------------ ci-wait
def _fake_gh(tmp: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a `gh` on PATH that logs its argv and lists no runs."""
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    log = tmp / "gh.log"
    gh = bin_dir / "gh"
    gh.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\necho "[]"\n')
    gh.chmod(gh.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


@pytest.mark.parametrize("ref", ["short", "HEAD", "branch"])
def test_ci_wait_asks_gh_for_the_full_sha_whatever_ref_it_is_given(
    env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    ref: str,
) -> None:
    """gh finds a run only by the full 40-hex sha: a short sha, HEAD or a branch is resolved."""
    repo = env["repo"]
    full = subprocess.run(  # noqa: S603
        [_git_bin(), "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _git(repo, "branch", "feature")
    arg = {"short": full[:7], "HEAD": "HEAD", "branch": "feature"}[ref]
    log = _fake_gh(env["tmp"], monkeypatch)
    monkeypatch.chdir(repo)
    code = cli.main(["ci-wait", arg, "--timeout", "0", "--config-json", json.dumps({})])
    assert code == 1  # no finished run in time
    argv = log.read_text()
    assert f"--commit {full}" in argv
    assert len(full) == 40


def test_ci_wait_with_an_unknown_ref_fails_before_asking_gh(
    env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ref that is no commit is an error, and gh is never asked about it."""
    log = _fake_gh(env["tmp"], monkeypatch)
    monkeypatch.chdir(env["repo"])
    with pytest.raises((subprocess.CalledProcessError, ValueError, SystemExit, RuntimeError)):
        cli.main(["ci-wait", "deadbeefnotacommit", "--timeout", "0", "--config-json", "{}"])
    assert not log.exists()


def _git_bin() -> str:
    return shutil.which("git") or "git"
