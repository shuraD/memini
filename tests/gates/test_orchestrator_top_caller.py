"""A ticket names its top caller (F-orchestrator-068, FLOW-HYBRID.md step 6, owner 2026-11-21).

Rules guarded:

- the ticket part ``Top caller: <dotted.module>:<function>``: the ready gate refuses a ticket that
  names card lines without one and any malformed value; a ticket naming no card may leave it out;
  the host check skips the value only; ``ticket_text``, ``fix_text``, ``enqueue`` and the CLI carry
  it; a reissue and a batch fix keep the original ticket's;
- the Planner copies a slice's top caller from the brief and never invents one;
- the spec carries it as ``top_caller:`` and the spec stage refuses a spec that differs from its
  task, in either direction;
- ``engine.rules_text(cfg, top_caller)`` adds the outside-in rules only for a top caller.

Fixtures follow ``test_orchestrator_enqueue.py`` (card files, ``cfg``) and
``test_orchestrator_stages.py`` (tmp repo, fake engine, ``env``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from tests.gates import test_orchestrator_enqueue, test_orchestrator_stages
from tests.gates.test_orchestrator_card_lines import _run_spec
from tests.gates.test_orchestrator_enqueue import MODULE, _t
from tests.gates.test_orchestrator_reissue import FAILING, FIX, WORDS
from tests.gates.test_orchestrator_stages import FAILING as HIDDEN
from tests.gates.test_orchestrator_stages import _spec_md
from tools.orchestrator import cli, config, engine, enqueue, moves, planner, spec, state

cfg = test_orchestrator_enqueue.cfg  # the shared fixtures
env = test_orchestrator_stages.env

GIT = shutil.which("git") or "git"
CALLER = "memini.mod:load_schema"
PARTS = " Context: c. Acceptance: a. Done: d."
CARD_TEXT = "A1: write a node." + PARTS
NO_CARD = "Tidy the helper." + PARTS
MISSING = "the ticket names card lines but has no Top caller: part (<dotted.module>:<function>)"
SHIPPED = config.load_config(Path.cwd(), overrides={"state_dir": "/nonexistent-state"}, env={})
MALFORMED = [
    "memini.mod",
    "memini.mod:",
    ":load_schema",
    "memini.mod:load_schema()",
    "memini.mod::load_schema",
    "memini.mod:load_schema:again",
    "memini.mod:load.schema",
    "memini.mod.load_schema",
    "memini/mod.py:load_schema",
    "src/memini/mod.py:load_schema",
    "memini-core.mod:load_schema",
    "1memini.mod:load_schema",
    "memini.mod:1load",
    "memini.mod: load_schema",
    "memini.mod :load_schema",
    "memini.mod:load_schema and memini.mod:other",
    "memini.mod:load_schema, memini.mod:other",
    "`memini.mod:load_schema`",
    "the load_schema function",
    "?",
]
WELL_FORMED = [
    "memini.mod:load_schema",
    "mod:main",
    "_private._mod:_run",
    "memini.core.calc:add2",
    "a.b.c.d.e:f_g",
    "Memini.Mod:Load",
    "memini.app:main",
]


def _problems(cfg: config.Config, text: str) -> list[str]:
    return enqueue.problems(cfg, _t(text))


def _malformed(value: str) -> str:
    return f"Top caller: {value} is not <dotted.module>:<function>"


def _queued(cfg: config.Config) -> dict[str, enqueue.Ticket]:
    queue = cfg.stages[0].queue
    return {t.id: t for t in enqueue.read_tickets(queue)} if queue.exists() else {}


def _top(ticket: enqueue.Ticket) -> str:
    return enqueue.part(ticket.text, enqueue.TOP_CALLER)


# ------------------------------------------------------------------------------ A: the part
def test_the_part_label_is_top_caller() -> None:
    """The ticket part is written ``Top caller:`` (one name for writer and reader)."""
    assert enqueue.TOP_CALLER == "Top caller"


@pytest.mark.parametrize(
    "text",
    [
        f"x. Context: c. Acceptance: a. Done: d. Top caller: {CALLER}.",
        f"x. Context: c. Acceptance: a. Done: d. Top caller: {CALLER}",
        f"x. Top caller: {CALLER}. Context: c. Acceptance: a. Done: d.",
        f"x. Context: c. Top caller: {CALLER}. Acceptance: a. Done: d.",
        f"x. Context: c. Acceptance: a. Done: d. Top caller:   {CALLER} .",
        f"x. Context: c. Acceptance: a. Done: d.\nTop caller: {CALLER}.\n",
        f'x. Top caller: {CALLER}. Owner words: "w". Context: c. Acceptance: a. Done: d.',
    ],
    ids=["last", "no-period", "first", "middle", "spaces", "own-line", "before-owner-words"],
)
def test_the_top_caller_is_read_wherever_the_part_stands(text: str) -> None:
    """``part`` returns the value alone: no trailing period, and no other part's text."""
    assert enqueue.part(text, enqueue.TOP_CALLER) == CALLER
    assert enqueue.part(text, "Done") == "d"
    assert enqueue.part(text, "Context") == "c"


def test_a_ticket_without_the_part_has_no_top_caller() -> None:
    """No ``Top caller:`` part reads as the empty text, never as another part."""
    assert enqueue.part("x." + PARTS, enqueue.TOP_CALLER) == ""
    assert enqueue.part("the top caller is memini.mod:main." + PARTS, enqueue.TOP_CALLER) == ""


# ------------------------------------------------------------------------ A: the ready gate
def test_a_card_ticket_without_a_top_caller_is_refused(cfg: config.Config) -> None:
    """A ticket that names baseline card lines and no top caller gets exactly this refusal."""
    assert _problems(cfg, CARD_TEXT) == [MISSING]
    assert MISSING in _problems(cfg, "A1 and A2: write and delete." + PARTS)


def test_an_empty_top_caller_part_counts_as_missing(cfg: config.Config) -> None:
    """A label with no value after it is no top caller."""
    assert MISSING in _problems(cfg, CARD_TEXT + " Top caller: .")
    assert MISSING in _problems(cfg, "A1: write a node. Top caller: Context: c. Acceptance: a.")


@pytest.mark.parametrize("value", WELL_FORMED)
def test_a_card_ticket_with_a_well_formed_top_caller_is_ready(
    cfg: config.Config, value: str
) -> None:
    """Every ``<dotted.module>:<function>`` form passes; the value is never read as a host."""
    assert _problems(cfg, f"{CARD_TEXT} Top caller: {value}.") == []
    assert _problems(cfg, f"A1: write a node. Top caller: {value}.{PARTS}") == []


@pytest.mark.parametrize("value", MALFORMED)
def test_a_malformed_top_caller_is_refused_naming_the_value(cfg: config.Config, value: str) -> None:
    """A value that is not ``<dotted.module>:<function>`` is refused, for a card ticket or not."""
    for text in (CARD_TEXT, NO_CARD):
        found = _problems(cfg, f"{text} Top caller: {value}.")
        assert _malformed(value) in found, found
        assert MISSING not in found  # one refusal says what is wrong, not two


def test_a_ticket_that_names_no_card_needs_no_top_caller(cfg: config.Config) -> None:
    """Unchanged: a no-card ticket is ready without the part, and may carry one."""
    assert _problems(cfg, NO_CARD) == []
    assert _problems(cfg, f"{NO_CARD} Top caller: {CALLER}.") == []


def test_the_other_refusals_stay_next_to_the_top_caller_refusal(cfg: config.Config) -> None:
    """The new refusal is added to the old ones, it replaces none."""
    found = _problems(cfg, "A1: write a node.")
    assert MISSING in found
    for label in ("Context", "Acceptance", "Done"):
        assert any(f"no {label}: part" in p for p in found), found


# ---------------------------------------------------------------- A: the text check and hosts
def test_a_module_path_that_looks_like_a_host_is_fine_as_the_top_caller(
    cfg: config.Config,
) -> None:
    """``memini.app:main`` is a module, not a host: the host check skips the value."""
    assert _problems(cfg, f"{CARD_TEXT} Top caller: memini.app:main.") == []


@pytest.mark.parametrize(
    "text",
    [
        "A1: write a node, as memini.app does.{parts} Top caller: memini.app:main.",
        "A1: write a node on localhost.{parts} Top caller: memini.app:main.",
        "A1: write a node. Top caller: memini.app:main. Context: ask db.internal."
        " Acceptance: a. Done: d.",
        "A1: write a node. Context: see memini.app:main. Acceptance: a. Done: d."
        " Top caller: memini.app:main.",
        "A1: write a node.{parts} Top caller: memini.app:main. Top caller: other.internal:run.",
    ],
    ids=[
        "same-name-before",
        "localhost",
        "host-in-a-later-part",
        "value-in-context",
        "second-part",
    ],
)
def test_a_host_outside_the_top_caller_value_is_still_refused(
    cfg: config.Config, text: str
) -> None:
    """Only the value is skipped: the same or another host anywhere else is a host name."""
    found = _problems(cfg, text.format(parts=PARTS))
    assert any(p.startswith("text: host name") for p in found), found


def test_an_ip_a_home_path_and_a_deny_word_are_checked_inside_the_value_too(
    cfg: config.Config,
) -> None:
    """The skip is for host names only: the rest of the text check reads the whole ticket."""
    ip = ".".join(["10", "1", "2", "3"])
    home = "/" + "Users" + "/someone/app"
    words = [m[1] for p in cfg.deny_patterns if (m := re.fullmatch(r"\\b([a-z]+)\\b", p))]
    word = next(w for w in words if w != "pytest")  # the ready gate allows the word pytest
    for value, expect in (
        (f"{ip}:main", "text: IP address"),
        (f"{home}:main", "text: home path"),
        (f"memini.{word}:main", "text: deny word"),
    ):
        found = _problems(cfg, f"{CARD_TEXT} Top caller: {value}.")
        assert any(p.startswith(expect) for p in found), (value, found)


# ------------------------------------------------------------------- A: who writes the part
def test_ticket_text_ends_with_the_top_caller_part(cfg: config.Config) -> None:
    """A card ticket written from its cards carries the top caller it was given."""
    text = enqueue.ticket_text(cfg, ["A1", "A2"], CALLER)
    assert enqueue.part(text, enqueue.TOP_CALLER) == CALLER
    assert text.startswith(enqueue.ticket_text(cfg, ["A1", "A2"]))  # the rest is unchanged
    assert "Top caller" not in enqueue.ticket_text(cfg, ["A1", "A2"])
    assert "Top caller" not in enqueue.ticket_text(cfg, ["A1", "A2"], "  ")
    for label in ("Context", "Acceptance", "Done"):  # no part swallows another
        assert enqueue.part(text, label) == enqueue.part(
            enqueue.ticket_text(cfg, ["A1", "A2"]), label
        )


def _fix(cfg: config.Config, **kw: Any) -> str:
    return enqueue.fix_text(
        cfg,
        "Cover every role",
        [],
        [],
        "owner 2026-11-16: rule 2",
        context="tests/unit/test_x.py::test_y failed",
        acceptance=["the other rows stay unchanged"],
        done="the unit suite is green",
        **kw,
    )


def test_a_fix_text_carries_the_top_caller(cfg: config.Config) -> None:
    """``fix_text`` writes the part when given a top caller and nothing when not."""
    with_top = _fix(cfg, top_caller=f" {CALLER} ")
    assert enqueue.part(with_top, enqueue.TOP_CALLER) == CALLER
    assert enqueue.part(with_top, "Done") == "the unit suite is green"
    assert "Top caller" not in _fix(cfg)
    assert with_top.startswith(_fix(cfg))


def test_enqueue_queues_a_card_ticket_with_its_top_caller(cfg: config.Config) -> None:
    """``enqueue`` with cards and a top caller: ready, queued, and the queue line keeps it."""
    _, found = enqueue.enqueue(
        cfg, task_id="T-3", module=MODULE, card_list=["A1", "A2"], top_caller=CALLER
    )
    assert found == []
    assert _top(_queued(cfg)["T-3"]) == CALLER


def test_enqueue_refuses_a_card_ticket_without_a_top_caller(cfg: config.Config) -> None:
    """No top caller, no default: the card ticket is refused and nothing is queued."""
    _, found = enqueue.enqueue(cfg, task_id="T-3", module=MODULE, card_list=["A1"])
    assert found == [MISSING]
    assert _queued(cfg) == {}


@pytest.mark.parametrize("value", ["memini.mod", "memini.mod:load_schema()", "load_schema"])
def test_enqueue_refuses_a_malformed_top_caller(cfg: config.Config, value: str) -> None:
    """A malformed value given to ``enqueue`` is refused by the ready gate, nothing is queued."""
    _, found = enqueue.enqueue(
        cfg, task_id="T-3", module=MODULE, card_list=["A1"], top_caller=value
    )
    assert _malformed(value) in found
    assert _queued(cfg) == {}


def _cli(cfg: config.Config, monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.chdir(cfg.repo)
    over = {"cards_dir": str(cfg.cards_dir), "state_dir": str(cfg.state_dir)}
    return cli.main(["enqueue", *args, "--config-json", json.dumps(over)])


def test_the_enqueue_command_takes_the_top_caller(
    cfg: config.Config, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``enqueue --cards --top-caller`` queues the ticket; without the flag it is refused."""
    code = _cli(cfg, monkeypatch, "T-5", "--file", MODULE, "--cards", "A1")
    assert code != 0
    assert MISSING in capsys.readouterr().out
    assert _queued(cfg) == {}
    code = _cli(cfg, monkeypatch, "T-5", "--file", MODULE, "--cards", "A1", "--top-caller", CALLER)
    assert code == 0, capsys.readouterr().out
    assert _top(_queued(cfg)["T-5"]) == CALLER


def test_the_enqueue_command_carries_the_top_caller_of_a_fix(
    cfg: config.Config, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A structured fix (``--change``) written by the command carries ``--top-caller`` too."""
    args = ["T-6", "--file", MODULE, "--change", "Cover A1 for every role"]
    args += ["--owner-words", "owner 2026-11-16: rule 2", "--context", "it failed"]
    args += ["--acceptance", "the rows stay", "--done", "the suite is green"]
    assert _cli(cfg, monkeypatch, *args) != 0  # the fix names A1: it needs a top caller
    assert MISSING in capsys.readouterr().out
    code = _cli(cfg, monkeypatch, *args, "--top-caller", CALLER)
    assert code == 0, capsys.readouterr().out
    assert _top(_queued(cfg)["T-6"]) == CALLER


# ----------------------------------------------------------- A: reissue and batch fix keep it
def _dispute(cfg: config.Config, text: str) -> moves.Move:
    st = state.State(cfg)
    st.claim("T-9", "w", [MODULE])
    st.update("T-9", "rejected", reason="audit")
    _, found = enqueue.enqueue(cfg, task_id="T-9", module=MODULE, text=text)
    assert found == []
    move_id = moves.post(
        cfg,
        "T-9",
        sender="devops",
        to="planner",
        kind="dispute",
        text="cannot be built as written",
        owner_words=WORDS,
    )
    return moves.get(cfg, move_id)


def test_a_reissued_ticket_keeps_the_original_top_caller(cfg: config.Config) -> None:
    """The Planner rewrites the ticket; the top caller is the original's, not the model's."""
    move = _dispute(cfg, f"Old text the spec line contradicts.{PARTS} Top caller: {CALLER}.")
    answer = json.dumps({**json.loads(FIX), "top_caller": "memini.mod:invented"})
    result = planner.reissue(cfg, move, lambda _prompt: answer)
    assert result.startswith("reissued"), result
    ticket = _queued(cfg)["T-9"]
    assert _top(ticket) == CALLER
    assert "invented" not in ticket.text
    assert f'Owner words: "{WORDS}"' in ticket.text


def test_a_reissued_ticket_without_a_top_caller_gets_none(cfg: config.Config) -> None:
    """Nothing is invented: an original with no top caller is reissued with none."""
    move = _dispute(cfg, "Old text the spec line contradicts." + PARTS)
    assert planner.reissue(cfg, move, lambda _prompt: FIX).startswith("reissued")
    assert "Top caller" not in _queued(cfg)["T-9"].text


def _batch_failure(cfg: config.Config, slice_text: str) -> str:
    (cfg.repo / "tests" / "unit").mkdir(parents=True, exist_ok=True)
    (cfg.repo / "tests/unit/test_x.py").write_text("def test_y():\n    assert True\n")
    clean = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    for args in (["add", "-A"], ["commit", "-qm", "an older test"]):
        subprocess.run([GIT, "-C", str(cfg.repo), *args], check=True, env=clean)  # noqa: S603
    _, found = enqueue.enqueue(cfg, task_id="T-9", module=MODULE, text=slice_text)
    assert found == []
    move_id = moves.post(
        cfg, "T-9", sender="dev4", to="planner", kind="batch_failure", text=f"failed: {FAILING}"
    )
    return planner.batch_fix(cfg, moves.get(cfg, move_id), lambda _p: FIX)


def test_a_batch_fix_takes_the_merged_slice_tickets_top_caller(cfg: config.Config) -> None:
    """The fix of a merged slice is built down from the same top caller as the slice."""
    text = f'The slice. Owner words: "{WORDS}".{PARTS} Top caller: {CALLER}.'
    result = _batch_failure(cfg, text)
    assert result.startswith("T-9x1"), result
    assert _top(_queued(cfg)["T-9x1"]) == CALLER


def test_a_batch_fix_of_a_slice_without_a_top_caller_gets_none(cfg: config.Config) -> None:
    """Nothing is invented for a slice ticket that had no top caller."""
    result = _batch_failure(cfg, f'The slice. Owner words: "{WORDS}".{PARTS}')
    assert result.startswith("T-9x1"), result
    assert "Top caller" not in _queued(cfg)["T-9x1"].text


# ------------------------------------------------------------------------------ B: the Planner
BRIEF = f"""version: v1
squad: alpha
prefix: PL
cards: A1 to A2
out: A9
decisions: the store is built from the ingest command
  Top caller: {CALLER}
"""


def _brief(tmp_path: Path, text: str = BRIEF) -> Path:
    path = tmp_path / "brief-1.md"
    path.write_text(text)
    return path


def _answer(*slices: dict[str, object]) -> str:
    return "here is the plan\n" + json.dumps({"slices": list(slices)})


def _slice(cards: list[str], **more: object) -> dict[str, object]:
    return {"module": MODULE, "cards": cards, "deps": [], **more}


def test_a_slice_of_the_answer_may_carry_a_top_caller() -> None:
    """``parse_slices`` reads ``top_caller`` (stripped); a slice without one has the empty text."""
    answer = _answer(_slice(["A1"], top_caller=f" {CALLER}\n"), _slice(["A2"]))
    first, second = planner.parse_slices(answer, {"A1", "A2"})
    assert first.top_caller == CALLER
    assert second.top_caller == ""
    assert (
        planner.parse_slices(_answer(_slice(["A1"], top_caller=None)), {"A1"})[0].top_caller == ""
    )
    assert planner.Slice(MODULE, ("A1",), ()).top_caller == ""


@pytest.mark.parametrize("value", [1, 1.5, True, ["memini.mod:load_schema"], {"a": "b"}])
def test_a_top_caller_that_is_not_text_is_refused(value: object) -> None:
    """A non-string ``top_caller`` refuses the answer, naming the slice by its index."""
    answer = _answer(_slice(["A1"], top_caller=CALLER), _slice(["A2"], top_caller=value))
    with pytest.raises(ValueError, match="slice 1 has a top caller that is not text"):
        planner.parse_slices(answer, {"A1", "A2"})


@pytest.mark.parametrize("value", [0, False, [], {}])
def test_an_empty_top_caller_that_is_not_text_is_refused(value: object) -> None:
    """A non-string is refused even when it is empty: ``0`` or ``[]`` is not a top caller."""
    answer = _answer(_slice(["A1"], top_caller=value))
    with pytest.raises(ValueError, match="slice 0 has a top caller that is not text"):
        planner.parse_slices(answer, {"A1"})


def test_the_plan_passes_the_top_caller_to_the_ticket_and_shows_it(
    cfg: config.Config, tmp_path: Path
) -> None:
    """The slice's top caller reaches the queued ticket and the summary line of the slice."""
    ask = lambda _p: _answer(_slice(["A1", "A2"], top_caller=CALLER))  # noqa: E731
    lines = planner.plan(cfg, _brief(tmp_path), ask)
    assert _top(_queued(cfg)["PL-01"]) == CALLER
    [line] = [x for x in lines if x.startswith("- PL-01 ")]
    assert f"top caller {CALLER}" in line and "REFUSED" not in line


def test_the_planner_prompt_asks_for_the_top_caller_and_shows_the_brief(
    cfg: config.Config, tmp_path: Path
) -> None:
    """The Planner can only copy what it sees: the brief's top caller is in its prompt."""
    seen: list[str] = []

    def ask(prompt: str) -> str:
        seen.append(prompt)
        return _answer(_slice(["A1", "A2"], top_caller=CALLER))

    planner.plan(cfg, _brief(tmp_path), ask)
    assert f"Top caller: {CALLER}" in seen[0]
    assert '"top_caller"' in seen[0]


@pytest.mark.parametrize(
    "invented",
    ["memini.other:main", "memini.mod:Load_Schema", "memini.mod:load_schema_all", "x.memini.mod:y"],
)
def test_a_top_caller_that_is_not_in_the_brief_refuses_the_whole_plan(
    cfg: config.Config, tmp_path: Path, invented: str
) -> None:
    """The Planner never invents a top caller: one bad slice refuses every slice of the plan."""
    ask = lambda _p: _answer(  # noqa: E731
        _slice(["A1"], top_caller=CALLER), _slice(["A2"], top_caller=invented)
    )
    expect = re.escape(f"a slice names a top caller that is not in the brief: {invented}")
    with pytest.raises(ValueError, match=expect):
        planner.plan(cfg, _brief(tmp_path), ask)
    assert _queued(cfg) == {}


@pytest.mark.parametrize(
    "piece",
    ["memini.mod:load", "mod:load_schema", "emini.mod:load_schema", "d:l", "memini.mod:load_sch"],
)
def test_a_piece_of_the_briefs_top_caller_is_not_the_briefs_top_caller(
    cfg: config.Config, tmp_path: Path, piece: str
) -> None:
    """Written literally means the whole value: a cut-down copy names another function.

    The brief says ``Top caller: memini.mod:load_schema``; each value here is well-formed and
    only a fragment of it, so the brief does not name it.
    """
    ask = lambda _p: _answer(_slice(["A1", "A2"], top_caller=piece))  # noqa: E731
    with pytest.raises(ValueError, match="top caller that is not in the brief"):
        planner.plan(cfg, _brief(tmp_path), ask)
    assert _queued(cfg) == {}


def test_a_brief_may_name_several_top_callers(cfg: config.Config, tmp_path: Path) -> None:
    """Each slice copies the one that covers it; both are written in the brief."""
    other = "memini.mod:other_entry"
    brief = _brief(tmp_path, BRIEF + f"  Top caller: {other}\n")
    ask = lambda _p: _answer(  # noqa: E731
        _slice(["A1"], top_caller=CALLER), _slice(["A2"], top_caller=other)
    )
    planner.plan(cfg, brief, ask)
    assert {k: _top(t) for k, t in _queued(cfg).items()} == {"PL-01": CALLER, "PL-02": other}


def test_a_card_slice_with_no_top_caller_is_shown_refused_and_none_is_invented(
    cfg: config.Config, tmp_path: Path
) -> None:
    """The brief names a top caller, the slice does not: refused by the ready gate, no default."""
    ask = lambda _p: _answer(_slice(["A1"], top_caller=CALLER), _slice(["A2"]))  # noqa: E731
    lines = planner.plan(cfg, _brief(tmp_path), ask)
    [good] = [x for x in lines if x.startswith("- PL-01 ")]
    [bad] = [x for x in lines if x.startswith("- PL-02 ")]
    assert "REFUSED" not in good
    assert "REFUSED: " in bad and MISSING in bad
    assert "top caller memini" not in bad
    assert list(_queued(cfg)) == ["PL-01"]


# ------------------------------------------------------------------------------ C: the spec
def _load(tmp_path: Path, **fields: Any) -> spec.Spec:
    path = tmp_path / "T-1.md"
    path.write_text(_spec_md("T-1", **fields))
    return spec.load(path)


def test_the_spec_header_carries_the_top_caller(tmp_path: Path) -> None:
    """``Spec.top_caller`` is the header's value and every agent sees it in ``Spec.text()``."""
    sp = _load(tmp_path, top_caller="memini.core.calc:add")
    assert sp.top_caller == "memini.core.calc:add"
    assert "top_caller: memini.core.calc:add\n" in sp.text()


@pytest.mark.parametrize("fields", [{}, {"top_caller": "none"}, {"top_caller": ""}])
def test_a_spec_with_no_top_caller_or_none_has_none(tmp_path: Path, fields: dict[str, Any]) -> None:
    """``none`` or an absent field both mean: no top caller."""
    assert _load(tmp_path, **fields).top_caller == ""


def test_a_null_top_caller_is_none(tmp_path: Path) -> None:
    """A YAML null (an empty ``top_caller:`` line) is no top caller."""
    path = tmp_path / "T-1.md"
    path.write_text(_spec_md("T-1").replace("behavior:", "top_caller:\nbehavior:", 1))
    assert spec.load(path).top_caller == ""


TASK_TEXT = (
    "Add two numbers. Context: c. Acceptance: add returns 3; add rejects text. Done: exported."
)
ACCEPTANCE = (
    "- AC1: Given 1 and 2, Then 3.\n- AC2: Given text, Then it is refused.\n- done: exported."
)
TESTS = HIDDEN + "\n\n# AC1 AC2\n"
ADD = "memini.core.calc:add"


def _spec_stage(
    env: dict[str, Any], mp: pytest.MonkeyPatch, ticket_top: str, **spec_fields: Any
) -> tuple[int, str]:
    text = TASK_TEXT + (f" Top caller: {ticket_top}." if ticket_top else "")
    files = {
        "SPEC.md": _spec_md(acceptance=ACCEPTANCE, **spec_fields),
        "tests/unit/test_c.py": TESTS,
    }
    return _run_spec(env, mp, text, files)


@pytest.mark.parametrize(
    ("ticket_top", "fields"),
    [
        (ADD, {"top_caller": ADD}),
        ("", {}),
        ("", {"top_caller": "none"}),
    ],
    ids=["both-the-same", "neither", "spec-says-none"],
)
def test_a_spec_that_carries_the_tasks_top_caller_passes(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, ticket_top: str, fields: dict[str, Any]
) -> None:
    """The spec repeats the task's top caller exactly (or none for a task without one)."""
    code, reason = _spec_stage(env, monkeypatch, ticket_top, **fields)
    assert code == 0, reason


@pytest.mark.parametrize(
    ("ticket_top", "fields", "refusal"),
    [
        (ADD, {}, f"spec top_caller 'none' is not the task's Top caller '{ADD}'"),
        (
            ADD,
            {"top_caller": "none"},
            f"spec top_caller 'none' is not the task's Top caller '{ADD}'",
        ),
        (
            ADD,
            {"top_caller": "memini.core.calc:sub"},
            f"spec top_caller 'memini.core.calc:sub' is not the task's Top caller '{ADD}'",
        ),
        (
            ADD,
            {"top_caller": "memini.core.calc:ad"},
            f"spec top_caller 'memini.core.calc:ad' is not the task's Top caller '{ADD}'",
        ),
        ("", {"top_caller": ADD}, f"spec top_caller '{ADD}' is not the task's Top caller 'none'"),
    ],
    ids=["spec-lacks-it", "spec-says-none", "spec-changes-it", "spec-cuts-it", "spec-invents-it"],
)
def test_a_spec_whose_top_caller_is_not_the_tasks_is_refused(
    env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    ticket_top: str,
    fields: dict[str, Any],
    refusal: str,
) -> None:
    """Either direction: a dropped, changed or invented top caller rejects the spec."""
    code, reason = _spec_stage(env, monkeypatch, ticket_top, **fields)
    assert code == 1
    assert refusal in reason


@pytest.mark.parametrize(("ticket_top", "shown"), [(ADD, ADD), ("", "none")])
def test_the_spec_prompt_shows_the_top_caller_to_copy(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, ticket_top: str, shown: str
) -> None:
    """The spec writer is given ``top_caller: <the task's>`` (or none) and the matching rules."""
    fields = {"top_caller": ticket_top} if ticket_top else {}
    code, reason = _spec_stage(env, monkeypatch, ticket_top, **fields)
    assert code == 0, reason
    prompt = (env["tmp"] / "log" / "qa-1.txt").read_text()
    assert f"top_caller: {shown}\n" in prompt
    assert "$top_caller" not in prompt
    assert ("checked by the reachability gate" in prompt) == bool(ticket_top)


# ------------------------------------------------------------------------------ F: the rules
def test_the_rules_without_a_top_caller_are_unchanged() -> None:
    """``rules_text(cfg)`` has no outside-in rules; the empty top caller is the same text."""
    plain = engine.rules_text(SHIPPED)
    assert engine.rules_text(SHIPPED, "") == plain
    assert "Top caller" not in plain and "top caller" not in plain
    assert "NotImplementedError" not in plain
    assert plain.rstrip().endswith("QA changes no old file")  # the old last rule is still last


def test_the_rules_with_a_top_caller_append_the_outside_in_rules() -> None:
    """One source: the old text, then ``rules_top_caller.txt`` filled with the top caller."""
    plain = engine.rules_text(SHIPPED)
    text = engine.rules_text(SHIPPED, ADD)
    extra = (SHIPPED.prompts_dir / "rules_top_caller.txt").read_text(encoding="utf-8")
    assert "$top_caller" in extra
    assert text == plain + "\n" + extra.replace("$top_caller", ADD).rstrip()
    assert "$" not in text[len(plain) :]


def test_the_outside_in_rules_say_what_the_gates_check() -> None:
    """The appended rules name the top caller, reachability, stubs and the end-to-end test."""
    added = engine.rules_text(SHIPPED, ADD)[len(engine.rules_text(SHIPPED)) :]
    assert ADD in added
    for words in ("reached from", "NotImplementedError", "end-to-end test", "src/"):
        assert words in added, words


@pytest.mark.parametrize("name", ["spec", "qa", "coder", "audit", "test_review"])
def test_the_rules_with_a_top_caller_fit_every_prompt(name: str) -> None:
    """The same text, top caller rules included, is rendered verbatim into each agent prompt."""
    rules = engine.rules_text(SHIPPED, ADD)
    assert rules in engine.render(SHIPPED, name, rules=rules)
