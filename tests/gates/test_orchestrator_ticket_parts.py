"""Every ticket carries Context:, Acceptance: and Done: (ready gate, fix_text, spec stage).

Rules guarded: the ready gate refuses a ticket missing a part and names it; ``fix_text`` renders
the three parts and keeps the owner's words as a quotation; ``batch_fix`` puts the failing test
ids first in Acceptance and asks vp-rnd (no ticket) when it finds none; a ticket naming no card is
covered by ``- AC<n>:`` lines, one ``- done:`` line and hidden tests naming each AC<n>.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest
from tests.gates import test_orchestrator_enqueue, test_orchestrator_stages
from tests.gates.test_orchestrator_card_lines import _run_spec
from tests.gates.test_orchestrator_enqueue import MODULE, _t
from tests.gates.test_orchestrator_reissue import FAILING, FIX, WORDS
from tests.gates.test_orchestrator_stages import FAILING as HIDDEN
from tests.gates.test_orchestrator_stages import _spec_md
from tools.orchestrator import cli, config, enqueue, moves, planner

cfg = test_orchestrator_enqueue.cfg  # the shared fixtures
env = test_orchestrator_stages.env

PARTS = "Context: c. Acceptance: a. Done: d."
TOP = " Top caller: memini.mod:load_schema."  # a card ticket names its top caller
TWO = "Add two numbers. Context: c. Acceptance: add returns 3; add rejects text. Done: exported."
GOOD = "- AC1: Given 1 and 2, Then 3.\n- AC2: Given text, Then it is refused.\n- done: exported."
TESTS = HIDDEN + "\n\n# AC1 AC2\n"


def _problems(cfg: config.Config, text: str) -> list[str]:
    return enqueue.problems(cfg, _t(text))


@pytest.mark.parametrize("missing", ["Context", "Acceptance", "Done"])
def test_a_ticket_missing_one_part_is_refused_naming_it(cfg: config.Config, missing: str) -> None:
    """The ready gate refuses a ticket without one of its three parts, naming that part."""
    parts = {"Context": "c.", "Acceptance": "a.", "Done": "d."}
    del parts[missing]
    text = "A1: write a node. " + " ".join(f"{k}: {v}" for k, v in parts.items())
    found = _problems(cfg, text)
    assert [p for p in found if f"no {missing}: part" in p], found
    assert all(f"no {k}: part" not in " ".join(found) for k in parts)


@pytest.mark.parametrize("text", ["A1: write a node.", "A1: write a node. Context:  . Done: d."])
def test_a_ticket_with_no_parts_or_empty_ones_is_refused(cfg: config.Config, text: str) -> None:
    """A part label with no text after it counts as missing."""
    found = " ".join(_problems(cfg, text))
    assert "no Context: part" in found and "no Acceptance: part" in found


def test_a_ticket_with_all_three_parts_is_ready(cfg: config.Config) -> None:
    """All three parts present and nothing else wrong: no problem."""
    assert _problems(cfg, "A1: write a node. " + PARTS + TOP) == []


def test_parts_are_read_between_their_labels() -> None:
    """``part`` stops at the next label; acceptance items split on semicolons."""
    text = 'x. Context: ctx. Owner words: "w". Acceptance: one; two ; . Done: end.'
    assert enqueue.part(text, "Context") == "ctx"
    assert enqueue.part(text, "Done") == "end"
    assert enqueue.acceptance_items(text) == ["one", "two"]


def _fix(cfg: config.Config, **kw: Any) -> str:
    base: dict[str, Any] = {
        "context": "tests/unit/test_x.py::test_y failed",
        "acceptance": ["the other rows stay unchanged"],
        "done": "the unit suite is green",
    }
    return enqueue.fix_text(
        cfg, "Cover every role", [], [], "owner 2026-11-16: rule 2", **{**base, **kw}
    )


def test_a_fix_text_has_the_three_parts_and_quotes_the_owner_words(cfg: config.Config) -> None:
    """fix_text renders Context, Acceptance and Done; the owner words stay a quotation."""
    text = _fix(cfg)
    assert enqueue.part(text, "Context").startswith("tests/unit/test_x.py::test_y failed")
    assert enqueue.acceptance_items(text) == ["the other rows stay unchanged"]
    assert enqueue.part(text, "Done") == "the unit suite is green"
    assert 'Owner words: "owner 2026-11-16: rule 2".' in text


def test_semicolons_inside_an_acceptance_item_are_replaced(cfg: config.Config) -> None:
    """A ';' inside one item would split it in two: it becomes a comma."""
    text = _fix(cfg, acceptance=["a; b", "c"])
    assert enqueue.acceptance_items(text) == ["a, b", "c"]


@pytest.mark.parametrize("field", ["context", "done"])
def test_a_fix_text_without_a_part_is_refused_by_the_ready_gate(
    cfg: config.Config, field: str
) -> None:
    """A fix written with an empty context or done is caught at the ready gate."""
    found = _problems(cfg, _fix(cfg, **{field: ""}))
    assert any(f"no {field.capitalize()}: part" in p for p in found), found


def test_a_fix_text_with_no_acceptance_is_refused_by_the_ready_gate(cfg: config.Config) -> None:
    """An empty acceptance list leaves no Acceptance: part: refused."""
    assert any("no Acceptance: part" in p for p in _problems(cfg, _fix(cfg, acceptance=[])))


def _slice(cfg: config.Config) -> None:
    (cfg.repo / "tests" / "unit").mkdir(parents=True, exist_ok=True)
    (cfg.repo / "tests/unit/test_x.py").write_text("def test_y():\n    assert True\n")
    for args in (["add", "-A"], ["commit", "-qm", "an older test"]):
        subprocess.run(["git", "-C", str(cfg.repo), *args], check=True)  # noqa: S603, S607
    text = f'The slice. Owner words: "{WORDS}". {PARTS}'
    enqueue.enqueue(cfg, task_id="T-9", module=MODULE, text=text)


def test_a_batch_fix_lists_the_failing_ids_first_in_acceptance(cfg: config.Config) -> None:
    """The failing ids (deduplicated, in order) open Acceptance; the Planner's items follow."""
    _slice(cfg)
    one = "tests/unit/test_x.py::test_y"
    move_id = moves.post(
        cfg,
        "T-9",
        sender="dev4",
        to="planner",
        kind="batch_failure",
        text=f"failed: {one} and again {one} and tests/unit/test_x.py::test_z[a-1]",
    )
    result = planner.batch_fix(cfg, moves.get(cfg, move_id), lambda _p: FIX)
    assert result.startswith("T-9x1"), result
    ticket = {t.id: t for t in enqueue.read_tickets(cfg.stages[0].queue)}["T-9x1"]
    items = enqueue.acceptance_items(ticket.text)
    assert items[0] == f"{one}, tests/unit/test_x.py::test_z[a-1] pass again under every profile"
    assert items[1:] == ["the other properties stay unchanged"]
    assert enqueue.part(ticket.text, "Context").startswith("a test failed after the slice merged")


@pytest.mark.parametrize(
    "text", ["the suite failed", "tests/unit/test_x.py failed", "test_y failed: tests/x.txt::a"]
)
def test_a_batch_failure_without_failing_ids_asks_vp_rnd_and_writes_no_ticket(
    cfg: config.Config, text: str
) -> None:
    """No `tests/...py::id` in the move text: vp-rnd is asked and the queue is unchanged."""
    _slice(cfg)
    move_id = moves.post(cfg, "T-9", sender="dev4", to="planner", kind="batch_failure", text=text)
    result = planner.batch_fix(cfg, moves.get(cfg, move_id), lambda _p: FIX)
    assert "asked vp-rnd" in result
    assert [t.id for t in enqueue.read_tickets(cfg.stages[0].queue)] == ["T-9"]
    assert len(moves.open_for(cfg, "vp-rnd")) == 1


@pytest.mark.parametrize("drop", ["context", "acceptance", "done"])
def test_a_planner_answer_without_a_part_writes_no_ticket(cfg: config.Config, drop: str) -> None:
    """The Planner must give context, acceptance and done: an answer lacking one is unusable."""
    _slice(cfg)
    answer = {k: v for k, v in json.loads(FIX).items() if k != drop}
    move_id = moves.post(
        cfg, "T-9", sender="dev4", to="planner", kind="batch_failure", text=f"f: {FAILING}"
    )
    result = planner.batch_fix(cfg, moves.get(cfg, move_id), lambda _p: json.dumps(answer))
    assert "not a usable fix" in result
    assert [t.id for t in enqueue.read_tickets(cfg.stages[0].queue)] == ["T-9"]


def _spec(env: dict[str, Any], mp: pytest.MonkeyPatch, acceptance: str, tests: str) -> str:
    code, reason = _run_spec(
        env, mp, TWO, {"SPEC.md": _spec_md(acceptance=acceptance), "tests/unit/test_c.py": tests}
    )
    return "ok" if code == 0 else reason


def test_a_complete_no_card_spec_is_accepted(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1, AC2, `- done:` and hidden tests naming both: accepted."""
    assert _spec(env, monkeypatch, GOOD, TESTS) == "ok"


@pytest.mark.parametrize(
    ("acceptance", "tests", "expect"),
    [
        (GOOD.replace("- AC1:", "- X1:"), TESTS, "no line for AC1"),
        (GOOD.replace("- AC2:", "- X2:"), TESTS, "no line for AC2"),
        (GOOD.replace("- done:", "- finished:"), TESTS, "no definition of done"),
        (GOOD.replace("- AC1:", "- AC10:"), TESTS, "no line for AC1"),
        (GOOD, HIDDEN + "\n\n# AC2\n", "no hidden test covers AC1"),
        (GOOD, HIDDEN + "\n\n# AC10 AC2\n", "no hidden test covers AC1"),
        (GOOD, HIDDEN, "no hidden test covers AC1"),
    ],
)
def test_a_no_card_spec_missing_an_item_is_refused(
    env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    acceptance: str,
    tests: str,
    expect: str,
) -> None:
    """Each AC<n> needs its acceptance line and a hidden test naming it; one `- done:` is needed."""
    assert expect in _spec(env, monkeypatch, acceptance, tests)


CARDS = "- A1: Given 1, Then 2.\n- A1 done: add is exported."
WITH_CARD = "Build A1. Context: c. Acceptance: a1 holds; a2 holds. Done: d."


@pytest.mark.parametrize(
    ("acceptance", "tests", "expect"),
    [
        (CARDS, HIDDEN + "\n\n# A1\n", ""),  # no AC lines and no `- done:` are needed
        ("- A1: Given 1.", HIDDEN + "\n\n# A1\n", "definition of done for A1"),
        (CARDS, HIDDEN, "no hidden test covers card line A1"),
        ("- A1 done: d.", HIDDEN + "\n\n# A1\n", "no line for card line A1"),
    ],
)
def test_a_ticket_with_cards_keeps_the_card_rules(
    env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    acceptance: str,
    tests: str,
    expect: str,
) -> None:
    """With a card named, the card-line rules and messages apply and AC lines are not needed."""
    files = {"SPEC.md": _spec_md(acceptance=acceptance), "tests/unit/test_t.py": tests}
    code, reason = _run_spec(env, monkeypatch, WITH_CARD, files)
    assert (code == 0, expect in reason) == (not expect, True), reason


def test_the_ready_command_refuses_an_old_format_queued_ticket(
    cfg: config.Config, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old-format queued ticket (no parts) is refused by `ready`; a complete one is ready."""
    q = cfg.stages[0].queue
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_text(
        f"T-1 {MODULE} - - 'A1: write a node.'\nT-2 {MODULE} - - 'A1: write. {PARTS}{TOP}'\n"
    )
    monkeypatch.chdir(cfg.repo)
    over = {"cards_dir": str(cfg.cards_dir), "state_dir": str(cfg.state_dir)}
    code = cli.main(["ready", "--config-json", json.dumps(over)])
    out = capsys.readouterr().out
    assert code == 1
    assert "T-1: refused" in out and "no Done: part" in out and "T-2: ready" in out
    assert "names card lines but has no Top caller: part" in out
