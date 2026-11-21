"""At the round limit the Planner validates the ticket and the verdict travels with the escalation.

Rules (F-orchestrator-069, owner 2026-11-21: "at the round limit the Planner validates the ticket
first and its verdict travels with the escalation, so the owner gets a cause, not a forward"):
a machine dispute whose earlier ``reissued`` rows number ``max_rounds - 1`` or more gets exactly one
validation model call and one ask to vp-rnd with the text
``<task>: round <n> of max_rounds <N>, no ticket written; it now waits for the owner. Planner
validation: <verdict>: <reason>. Failures so far: round 1: ... | round 2: ...``. A model error or
an unusable answer gives ``Planner validation: failed: <error text>`` and the escalation still
happens; no ticket in the queue gives ``failed: <task> has no ticket in <queue file name>.``. The
validation text is one line of at most 300 characters. At the limit there is no fix prompt, no
ticket, no queue change and no new ``reissued`` row, whatever the verdict and whatever the
``validated`` flag. ``ticket_validated`` is logged on a usable verdict only. vp-rnd writes the
whole text into the owner inbox. Below the limit nothing changes.
"""

# ruff: noqa: F811
from __future__ import annotations

import dataclasses
import json

import pytest
from tests.gates.test_orchestrator_enqueue import cfg  # noqa: F401  (cfg is a fixture)
from tests.gates.test_orchestrator_planner_validate import (
    FAILURE,
    VP_WORDS,
    Model,
    _asks,
    _dispute,
    _queue,
    _serve_planner,
    _verdict,
    _vp_inbox,
)
from tools.orchestrator import cli, config, moves, planner, state

LIMIT = "round limit reached: asked vp-rnd, waits for the owner"
CAP = 300
MARK = "Planner validation: "
TAIL = ". Failures so far: "


def _at_limit(cfg: config.Config, earlier: int = 1) -> None:
    """Record ``earlier`` reissues of T-9 (the shipped max_rounds is 2: one is the limit)."""
    for _ in range(earlier):
        state.ops_log(cfg, "T-9", "reissued", "earlier")


def _validation(text: str) -> str:
    """Return the validation text of an ask: what sits between the two fixed sentences."""
    assert text.count(MARK) == 1, text
    rest = text.split(MARK, 1)[1]
    assert TAIL in rest, text
    return rest.split(TAIL, 1)[0]


def _with_owner_inbox(cfg: config.Config) -> config.Config:
    return dataclasses.replace(cfg, inbox=cfg.state_dir.parent / "owner-inbox" / "INBOX.md")


def _no_model(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace vp-rnd's model seam by a spy; return the prompts it receives."""
    calls: list[str] = []

    def spy(_c: config.Config, prompt: str) -> str:
        calls.append(prompt)
        return "{}"

    monkeypatch.setattr(planner, "_ask", spy)
    return calls


# ------------------------------------------------------------------------------ the ask
def test_the_ask_at_the_limit_is_the_exact_sentence_with_verdict_and_history(
    cfg: config.Config,
) -> None:
    """Rule: one ask: task, round n of N, no ticket, the validation, every failure so far."""
    model = Model(_verdict("ok"))
    assert planner.reissue(cfg, _dispute(cfg, text="round one failure"), model).startswith(
        "reissued"
    )
    model.validation = _verdict("slicing", "cards A1 and A2 belong together")
    assert planner.reissue(cfg, _dispute(cfg, text="round two failure"), model) == LIMIT
    (ask,) = _asks(cfg)
    assert ask["text"] == (
        "T-9: round 3 of max_rounds 2, no ticket written; it now waits for the owner. "
        "Planner validation: slicing: cards A1 and A2 belong together. "
        "Failures so far: round 1: round one failure | round 2: round two failure"
    )
    assert (ask["task"], ask["from"]) == ("T-9", "planner")
    assert ask["text"] in _vp_inbox(cfg)


@pytest.mark.parametrize("kind", ["ok", "slicing", "scope"])
@pytest.mark.parametrize("validated", [None, True, False])
def test_at_the_limit_one_validation_call_one_ask_and_nothing_written(
    cfg: config.Config, kind: str, validated: bool | None
) -> None:
    """Rule: every verdict, validated or not: one model call, one ask, no fix, no ticket."""
    _at_limit(cfg)
    words = VP_WORDS if validated else None
    move = _dispute(cfg, words) if validated is None else _dispute(cfg, words, validated=validated)
    before = _queue(cfg)
    model = Model(_verdict(kind, "the cause in one sentence"))
    assert planner.reissue(cfg, move, model) == LIMIT
    assert len(model.validate_prompts) == 1 and model.fix_prompts == []
    assert _queue(cfg) == before and "SUPERSEDED" not in _queue(cfg)
    assert state.ops_count(cfg, "T-9", "reissued") == 1
    assert state.ops_count(cfg, "T-9", "ticket_validated") == 1
    (ask,) = _asks(cfg)  # one ask only: no separate '<kind>: ... | failure: ...' ask
    assert _validation(ask["text"]) == f"{kind}: the cause in one sentence"
    assert "| failure:" not in ask["text"]
    assert f"round 1: {FAILURE}" in ask["text"]
    assert (state.State(cfg).task("T-9") or {})["status"] == "rejected"  # not released


def test_the_validation_prompt_at_the_limit_carries_the_ticket_and_this_failure(
    cfg: config.Config,
) -> None:
    """Rule: the limit validation is the same check: the model sees the ticket and the failure."""
    _at_limit(cfg)
    model = Model()
    planner.reissue(cfg, _dispute(cfg, text="the limit round failure"), model)
    (prompt,) = model.validate_prompts
    assert "Build A1 and A2." in prompt and "the limit round failure" in prompt


@pytest.mark.parametrize(
    "rounds, earlier, said", [(3, 2, "round 4 of max_rounds 3"), (1, 0, "2 of")]
)
def test_the_validation_follows_max_rounds(
    cfg: config.Config, rounds: int, earlier: int, said: str
) -> None:
    """Rule: the limit is max_rounds - 1 earlier reissues, whatever max_rounds is."""
    cfg = dataclasses.replace(cfg, max_rounds=rounds)
    _at_limit(cfg, earlier)
    model = Model(_verdict("scope", "card A1 must change"))
    assert planner.reissue(cfg, _dispute(cfg), model) == LIMIT
    assert len(model.validate_prompts) == 1 and model.fix_prompts == []
    (ask,) = _asks(cfg)
    assert said in ask["text"] and _validation(ask["text"]) == "scope: card A1 must change"


# ------------------------------------------------------------------------------ a failed check
@pytest.mark.parametrize(
    "error",
    [
        OSError("engine down"),
        RuntimeError("engine down"),
        ValueError("engine down"),
        TimeoutError("engine down"),
        KeyError("engine down"),
        Exception("engine down"),
    ],
)
def test_a_model_error_at_the_limit_is_reported_and_still_escalates(
    cfg: config.Config, error: Exception
) -> None:
    """Rule: a failing validation call never opens the round gate and never hides the escalation."""
    _at_limit(cfg)
    move = _dispute(cfg)
    before = _queue(cfg)
    model = Model(error)
    assert planner.reissue(cfg, move, model) == LIMIT
    (ask,) = _asks(cfg)
    found = _validation(ask["text"])
    assert found.startswith("failed: ") and "engine down" in found
    assert f"round 1: {FAILURE}" in ask["text"]
    assert model.fix_prompts == [] and _queue(cfg) == before
    assert state.ops_count(cfg, "T-9", "ticket_validated") == 0
    assert state.ops_count(cfg, "T-9", "reissued") == 1


@pytest.mark.parametrize(
    "answer",
    [
        "no json at all",
        "",
        "   \n",
        "{}",
        "[]",
        "{not json}",
        json.dumps({"reason": "no verdict"}),
        json.dumps({"verdict": "slicing"}),
        '"slicing"',
        "null",
    ],
)
def test_an_unusable_answer_at_the_limit_is_reported_failed_and_still_escalates(
    cfg: config.Config, answer: str
) -> None:
    """Rule: an answer that is no verdict object says 'failed'; no ticket_validated row."""
    _at_limit(cfg)
    move = _dispute(cfg)
    before = _queue(cfg)
    model = Model(answer)
    assert planner.reissue(cfg, move, model) == LIMIT
    (ask,) = _asks(cfg)
    assert _validation(ask["text"]).startswith("failed:"), ask["text"]
    assert ask["text"].startswith("T-9: round 3 of max_rounds 2, no ticket written; ")
    assert model.fix_prompts == [] and _queue(cfg) == before
    assert state.ops_count(cfg, "T-9", "ticket_validated") == 0
    assert state.ops_count(cfg, "T-9", "reissued") == 1


def test_no_ticket_in_the_queue_at_the_limit_is_said_and_still_escalates(
    cfg: config.Config,
) -> None:
    """Rule: with no ticket the ask says so by task and queue file name; nothing is written."""
    _at_limit(cfg)
    move = _dispute(cfg)
    queue = cfg.stages[0].queue
    queue.write_text("")
    model = Model(_verdict("slicing"))
    assert planner.reissue(cfg, move, model) == LIMIT
    (ask,) = _asks(cfg)
    assert f"Planner validation: failed: T-9 has no ticket in {queue.name}. Failures" in ask["text"]
    assert model.fix_prompts == [] and queue.read_text() == ""
    assert state.ops_count(cfg, "T-9", "ticket_validated") == 0
    assert state.ops_count(cfg, "T-9", "reissued") == 1


# ------------------------------------------------------------------------------ one line, capped
@pytest.mark.parametrize(
    "reason",
    [
        "word " * 2000,
        "x" * 5000,
        "line one\nline two\r\n\tline three\n" * 200,
        "a cause.\n\nFailures so far: none\n" + "y" * 900,
    ],
)
def test_a_long_or_multi_line_reason_is_one_line_of_at_most_300_chars(
    cfg: config.Config, reason: str
) -> None:
    """Rule: the validation text is one line, capped at 300 characters; the history follows."""
    _at_limit(cfg)
    assert planner.reissue(cfg, _dispute(cfg), Model(_verdict("scope", reason))) == LIMIT
    text = _asks(cfg)[0]["text"]
    head, _, tail = text.partition(MARK)
    found, _, history = tail.rpartition(TAIL)
    assert head.endswith("it now waits for the owner. ")
    assert found.startswith("scope: ") and len(found) <= CAP
    assert not any(c in found for c in "\n\r\t")
    assert history == f"round 1: {FAILURE}"


def test_a_long_error_text_is_capped_too(cfg: config.Config) -> None:
    """Rule: the cap holds for a failed validation as well."""
    _at_limit(cfg)
    model = Model(OSError("engine\ndown " * 1000))
    assert planner.reissue(cfg, _dispute(cfg), model) == LIMIT
    found = _validation(_asks(cfg)[0]["text"])
    assert found.startswith("failed: engine down") and len(found) <= CAP and "\n" not in found


def test_a_short_reason_is_never_cut(cfg: config.Config) -> None:
    """Rule: the cap cuts only what is over it: a reason inside the cap arrives whole."""
    reason = "r" * 200
    _at_limit(cfg)
    planner.reissue(cfg, _dispute(cfg), Model(_verdict("slicing", reason)))
    assert _validation(_asks(cfg)[0]["text"]) == f"slicing: {reason}"


# ------------------------------------------------------------------------------ below the limit
@pytest.mark.parametrize("kind", ["slicing", "scope"])
def test_below_the_limit_a_bad_verdict_still_asks_in_the_old_form(
    cfg: config.Config, kind: str
) -> None:
    """Rule: below the limit nothing changes: '<kind>: <reason> | failure: <failure>'."""
    cfg = dataclasses.replace(cfg, max_rounds=3)
    _at_limit(cfg, 1)  # one earlier reissue of three rounds: not the limit
    model = Model(_verdict(kind, "cards A1 and A2 belong together"))
    assert planner.reissue(cfg, _dispute(cfg), model) == f"validated: {kind}, asked vp-rnd"
    (ask,) = _asks(cfg)
    assert ask["text"] == f"{kind}: cards A1 and A2 belong together | failure: {FAILURE}"
    assert MARK not in ask["text"] and len(model.validate_prompts) == 1


@pytest.mark.parametrize("validation", [_verdict("ok"), "no json at all", OSError("engine down")])
def test_below_the_limit_ok_or_a_failed_check_still_reissues(
    cfg: config.Config, validation: str | Exception
) -> None:
    """Rule: below the limit an ok verdict, an unusable answer or an error lets the fix go on."""
    model = Model(validation)
    assert planner.reissue(cfg, _dispute(cfg), model).startswith("reissued")
    assert len(model.validate_prompts) == 1 and len(model.fix_prompts) == 1
    assert _asks(cfg) == [] and MARK not in _vp_inbox(cfg)
    assert state.ops_count(cfg, "T-9", "reissued") == 1


def test_a_persons_own_words_at_the_limit_are_not_validated(cfg: config.Config) -> None:
    """Rule: only a machine dispute meets the limit: a person's words reissue, no validation."""
    _at_limit(cfg, 3)
    model = Model(_verdict("slicing"))
    words = "owner 2026-11-21 (own words): fix this one thing only"
    assert planner.reissue(cfg, _dispute(cfg, words), model).startswith("reissued")
    assert model.validate_prompts == [] and _asks(cfg) == []


# ------------------------------------------------------------------------------ the chain
@pytest.mark.parametrize("kind", ["ok", "slicing", "scope"])
def test_the_verdict_reaches_the_owner_inbox_through_vp_rnd(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Rule: planner -> vp-rnd -> owner inbox: the entry carries the validation sentence."""
    cfg = _with_owner_inbox(cfg)
    model = Model(_verdict("ok"))
    first = _dispute(cfg, text="round one failure")
    assert _serve_planner(cfg, model) == 1
    assert moves.get(cfg, first["id"])["result"].startswith("reissued")
    second = _dispute(cfg, text="round two failure")
    model.validation = _verdict(kind, "the cut splits one behaviour of A1")
    queue_before = _queue(cfg)
    assert _serve_planner(cfg, model) == 1
    assert moves.get(cfg, second["id"])["result"] == LIMIT
    assert len(model.validate_prompts) == 2 and len(model.fix_prompts) == 1
    (ask,) = moves.for_task(cfg, "T-9", "ask")
    assert f"Planner validation: {kind}: the cut splits one behaviour of A1. " in ask["text"]

    calls = _no_model(monkeypatch)
    moves.serve(cfg, "vp-rnd", cli.HANDLERS["vp-rnd"], once=True)
    assert calls == []  # vp-rnd does not ask the model at the limit
    entry = cfg.inbox.read_text()
    assert f"Planner validation: {kind}: the cut splits one behaviour of A1. " in entry
    assert "round 3 of max_rounds 2, no ticket written" in entry
    assert "round 1: round one failure" in entry and "round 2: round two failure" in entry
    assert moves.open_for(cfg, "planner") == [] and _asks(cfg) == []
    assert len(moves.for_task(cfg, "T-9", "ask")) == 1
    assert _queue(cfg) == queue_before
    assert state.ops_count(cfg, "T-9", "reissued") == 1


@pytest.mark.parametrize(
    "validation, said",
    [
        (OSError("engine down"), "Planner validation: failed: engine down. "),
        ("no json at all", "Planner validation: failed:"),
    ],
)
def test_a_failed_validation_reaches_the_owner_inbox_too(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch, validation: str | Exception, said: str
) -> None:
    """Rule: a model error or a non-JSON answer still ends in the owner inbox, said as failed."""
    cfg = _with_owner_inbox(cfg)
    _at_limit(cfg)
    move = _dispute(cfg)
    assert _serve_planner(cfg, Model(validation)) == 1
    assert moves.get(cfg, move["id"])["result"] == LIMIT
    calls = _no_model(monkeypatch)
    moves.serve(cfg, "vp-rnd", cli.HANDLERS["vp-rnd"], once=True)
    entry = cfg.inbox.read_text()
    assert calls == [] and said in entry and FAILURE in entry
    assert moves.open_for(cfg, "planner") == []


def test_a_very_long_reason_and_long_failures_keep_the_sentence_in_the_owner_inbox(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: the cap keeps the cause readable: the inbox entry still has the validation."""
    cfg = _with_owner_inbox(cfg)
    model = Model(_verdict("ok"))
    _dispute(cfg, text="first " + "f" * 5000)
    assert _serve_planner(cfg, model) == 1
    _dispute(cfg, text="second " + "g" * 5000)
    model.validation = _verdict("scope", "card A1 must change " + "z" * 5000)
    assert _serve_planner(cfg, model) == 1
    _no_model(monkeypatch)
    moves.serve(cfg, "vp-rnd", cli.HANDLERS["vp-rnd"], once=True)
    entry = cfg.inbox.read_text()
    assert "Planner validation: scope: card A1 must change zzz" in entry
    assert "round 1: first fff" in entry and "round 2: second ggg" in entry
    (ask,) = moves.for_task(cfg, "T-9", "ask")
    assert len(_validation(ask["text"])) <= CAP


def test_a_reason_naming_another_decided_task_still_reaches_the_owner(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: at the limit the task waits for the owner; the model's reason cannot answer for him."""
    cfg = _with_owner_inbox(cfg)
    decisions = cfg.cards_dir.parent / "DECISIONS.md"
    decisions.write_text("**D-12 T-3 keep the cut**\n\nOwner: 2026-11-01 keep T-3 as cut.\n")
    _at_limit(cfg)
    _dispute(cfg)
    assert _serve_planner(cfg, Model(_verdict("slicing", "the same cut as T-3, see D-12"))) == 1
    (ask,) = moves.for_task(cfg, "T-9", "ask")
    _no_model(monkeypatch)
    moves.serve(cfg, "vp-rnd", cli.HANDLERS["vp-rnd"], once=True)
    assert moves.get(cfg, ask["id"])["result"] == "escalated to the owner"
    assert cfg.inbox.exists() and "Planner validation: slicing: " in cfg.inbox.read_text()
