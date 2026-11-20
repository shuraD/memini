"""The Planner validates a machine dispute against its version and cards before reissuing.

Rules (owner order 2026-11-20): only a machine dispute (the standing words, or words starting
``vp-rnd decision``) is validated, and only when the payload is not ``validated``; verdict ok, an
error or an unusable answer lets the reissue go on; verdict slicing / scope asks vp-rnd with
``<kind>: <reason> | failure: <failure>``, writes no ticket and logs ``ticket_validated``; a task
at its round limit never reaches validation. ``docfix.versions`` maps a card id to its version.
The chain with vp-rnd is bounded by ``max_rounds``.
"""

# ruff: noqa: F811
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest
from tests.gates.test_orchestrator_enqueue import MODULE, cfg  # noqa: F401  (cfg is a fixture)
from tests.gates.test_orchestrator_reissue import FIX, PARTS
from tools.orchestrator import cli, config, docfix, engine, enqueue, moves, planner, state, vprnd

FAILURE = "the card line A1 cannot be reached inside this slice"
VERSIONS_MD = """# Versions

## baseline

| A1 | The store writes a node. |
| A2 | The store deletes a node. |

## lenses

| A3 | A different squad's line. |
"""
VP_WORDS = "vp-rnd decision (tier 1) 2026-11-20: re-cut inside the same cards"


def _verdict(kind: str, reason: str = "the cut splits one behaviour") -> str:
    return json.dumps({"verdict": kind, "reason": reason})


class Model:
    """A stub for the Planner's one-shot call: routes a validation prompt and a fix prompt."""

    def __init__(self, validation: str | Exception | None = None, fix: str = FIX) -> None:
        self.validation = _verdict("ok") if validation is None else validation
        self.fix = fix
        self.validate_prompts: list[str] = []
        self.fix_prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        if '"verdict"' in prompt:
            self.validate_prompts.append(prompt)
            if isinstance(self.validation, Exception):
                raise self.validation
            return self.validation
        self.fix_prompts.append(prompt)
        return self.fix


def _ticket(cfg: config.Config, text: str = "Build A1 and A2." + PARTS) -> None:
    enqueue.enqueue(cfg, task_id="T-9", module=MODULE, text=text)


def _put_task(cfg: config.Config) -> None:
    st = state.State(cfg)
    if st.task("T-9") is None:
        st.claim("T-9", "w", [MODULE])
        st.update("T-9", "rejected", reason="audit")


def _dispute(
    cfg: config.Config, words: str | None = None, text: str = FAILURE, **payload: object
) -> moves.Move:
    """Post a dispute for T-9 (a ticket naming A1 and A2); the standing words by default."""
    _put_task(cfg)
    if not any(t.id == "T-9" for t in enqueue.read_tickets(cfg.stages[0].queue)):
        _ticket(cfg)
    w = cfg.standing_reissue_words if words is None else words
    move_id = moves.post(
        cfg, "T-9", sender="dev4", to="planner", kind="dispute", text=text, owner_words=w, **payload
    )
    return moves.get(cfg, move_id)


def _asks(cfg: config.Config) -> list[moves.Move]:
    return [m for m in moves.open_for(cfg, "vp-rnd") if m["kind"] == "ask"]


def _queue(cfg: config.Config) -> str:
    return cfg.stages[0].queue.read_text()


def _vp_inbox(cfg: config.Config) -> str:
    path = cfg.state_dir / "agent-inbox" / "vp-rnd.md"
    return path.read_text() if path.exists() else ""


def _versions_file(cfg: config.Config) -> Path:
    return cfg.cards_dir.parent / "capabilities" / "VERSIONS.md"


# ------------------------------------------------------------------------------ the verdicts
@pytest.mark.parametrize(
    "answer",
    [
        _verdict("ok"),
        _verdict("bogus"),
        _verdict("SLICING"),
        _verdict(""),
        "no json at all",
        "",
        "{}",
        "[]",
        json.dumps({"verdict": "slicing"}),
        json.dumps({"reason": "no verdict"}),
        json.dumps({"verdict": ["slicing"], "reason": "x"}),
        json.dumps({"verdict": None, "reason": "x"}),
        _verdict("slicing") + "\n" + _verdict("slicing"),
    ],
)
def test_ok_or_an_unusable_answer_lets_the_reissue_go_on(cfg: config.Config, answer: str) -> None:
    """Rule: only a clean slicing/scope verdict stops the reissue; all else continues."""
    model = Model(answer)
    result = planner.reissue(cfg, _dispute(cfg), model)
    assert result.startswith("reissued"), result
    assert len(model.validate_prompts) == 1 and len(model.fix_prompts) == 1
    assert _asks(cfg) == []
    assert state.ops_count(cfg, "T-9", "reissued") == 1


def test_a_model_error_in_validation_lets_the_reissue_go_on(cfg: config.Config) -> None:
    """Rule: a failing validation call never blocks the reissue (as before the change)."""
    model = Model(OSError("engine down"))
    assert planner.reissue(cfg, _dispute(cfg), model).startswith("reissued")
    assert state.ops_count(cfg, "T-9", "ticket_validated") == 0


def test_an_ok_verdict_is_logged_and_the_fix_follows(cfg: config.Config) -> None:
    """Rule: every usable verdict is logged as ticket_validated, ok included."""
    model = Model(_verdict("ok", "matches"))
    assert planner.reissue(cfg, _dispute(cfg), model).startswith("reissued")
    assert state.ops_count(cfg, "T-9", "ticket_validated") == 1


@pytest.mark.parametrize("kind", ["slicing", "scope"])
def test_slicing_or_scope_asks_vp_rnd_and_writes_no_ticket(cfg: config.Config, kind: str) -> None:
    """Rule: a bad cut or a scope change: ask vp-rnd with kind, reason and failure; no ticket."""
    move = _dispute(cfg)
    before = _queue(cfg)
    model = Model(_verdict(kind, "cards A1 and A2 belong together"))
    assert planner.reissue(cfg, move, model) == f"validated: {kind}, asked vp-rnd"
    assert _queue(cfg) == before and "SUPERSEDED" not in _queue(cfg)
    assert model.fix_prompts == []
    (ask,) = _asks(cfg)
    assert ask["text"] == f"{kind}: cards A1 and A2 belong together | failure: {FAILURE}"
    assert (ask["task"], ask["from"]) == ("T-9", "planner")
    assert ask["text"] in _vp_inbox(cfg)
    assert state.ops_count(cfg, "T-9", "ticket_validated") == 1
    assert state.ops_count(cfg, "T-9", "reissued") == 0
    assert (state.State(cfg).task("T-9") or {})["status"] == "rejected"  # not released


def test_the_reason_is_stripped_and_a_missing_reason_text_is_not_a_crash(
    cfg: config.Config,
) -> None:
    """Rule: the ask carries the stripped reason; a reason of whitespace still asks vp-rnd."""
    assert planner.reissue(cfg, _dispute(cfg), Model(_verdict("slicing", "  padded  "))).startswith(
        "validated"
    )
    assert _asks(cfg)[0]["text"].startswith("slicing: padded | failure: ")


# ------------------------------------------------------------------------------ who is validated
@pytest.mark.parametrize(
    "words",
    [
        "owner 2026-11-20 (own words): fix this one thing only",
        "Vp-rnd decision (tier 1) 2026-11-20: x",
        "owner 2026-11-20: vp-rnd decision",
        "the vp-rnd decision of 2026-11-20 says",
    ],
)
def test_a_persons_explicit_words_are_never_validated(cfg: config.Config, words: str) -> None:
    """Rule: only the standing words or a 'vp-rnd decision' prefix mark a machine dispute."""
    model = Model(_verdict("slicing"))
    assert planner.reissue(cfg, _dispute(cfg, words), model).startswith("reissued")
    assert model.validate_prompts == [] and _asks(cfg) == []


@pytest.mark.parametrize("words", ["standing", "vp"])
def test_the_standing_words_and_a_vp_rnd_decision_are_validated(
    cfg: config.Config, words: str
) -> None:
    """Rule: both machine word forms reach validation."""
    chosen = cfg.standing_reissue_words if words == "standing" else VP_WORDS
    model = Model(_verdict("slicing"))
    assert planner.reissue(cfg, _dispute(cfg, chosen), model) == "validated: slicing, asked vp-rnd"
    assert len(model.validate_prompts) == 1


def test_a_validated_payload_skips_validation(cfg: config.Config) -> None:
    """Rule: validated=True (vp-rnd approved the cut) goes straight to the fix."""
    model = Model(_verdict("slicing"))
    planner.reissue(cfg, _dispute(cfg, VP_WORDS, validated=True), model)
    assert model.validate_prompts == [] and len(model.fix_prompts) == 1


@pytest.mark.parametrize("flag", [False, None, 0, ""])
def test_a_falsy_validated_flag_is_still_validated(cfg: config.Config, flag: object) -> None:
    """Rule: only a true validated flag skips; false-ish values do not."""
    model = Model(_verdict("ok"))
    planner.reissue(cfg, _dispute(cfg, VP_WORDS, validated=flag), model)
    assert len(model.validate_prompts) == 1


def test_a_task_at_its_round_limit_never_reaches_validation(cfg: config.Config) -> None:
    """Rule: the gate runs first: at the limit no model call at all, an ask with the history."""
    state.ops_log(cfg, "T-9", "reissued", "earlier")
    model = Model(_verdict("ok"))
    result = planner.reissue(cfg, _dispute(cfg), model)
    assert result.startswith("round limit reached")
    assert model.validate_prompts == [] and model.fix_prompts == []


def test_a_vp_rnd_decision_is_bound_by_the_round_gate(cfg: config.Config) -> None:
    """Rule: words starting 'vp-rnd decision' count as machine words for the round limit."""
    state.ops_log(cfg, "T-9", "reissued", "earlier")
    model = Model()
    before_queue = None
    move = _dispute(cfg, VP_WORDS, validated=True)
    before_queue = _queue(cfg)
    assert planner.reissue(cfg, move, model).startswith("round limit reached")
    assert _queue(cfg) == before_queue and model.fix_prompts == []


def test_a_person_is_unlimited_by_the_gate_after_the_change(cfg: config.Config) -> None:
    """Rule: explicit different words still reissue far past max_rounds."""
    own = "owner 2026-11-20 (own words): go"
    for _ in range(4):
        state.ops_log(cfg, "T-9", "reissued", "earlier")
    assert planner.reissue(cfg, _dispute(cfg, own), Model()).startswith("reissued")


# ------------------------------------------------------------------------------ the prompt
def test_the_validation_prompt_carries_ticket_failure_cards_and_versions(
    cfg: config.Config,
) -> None:
    """Rule: the model sees the ticket, the failure, the card lines and each card's version."""
    _versions_file(cfg).parent.mkdir(exist_ok=True)
    _versions_file(cfg).write_text(VERSIONS_MD)
    model = Model()
    planner.reissue(cfg, _dispute(cfg), model)
    (prompt,) = model.validate_prompts
    assert "Build A1 and A2." in prompt and FAILURE in prompt
    assert "A1: The store writes a node." in prompt
    assert "A1: version baseline, squad alpha" in prompt
    assert "A2: version baseline, squad alpha" in prompt
    assert "$" not in prompt


def test_a_card_missing_from_versions_is_named_unknown(cfg: config.Config) -> None:
    """Rule: a card the version file does not list is shown as unknown, not dropped."""
    _versions_file(cfg).parent.mkdir(exist_ok=True)
    _versions_file(cfg).write_text("## baseline\n\n| A1 | The store writes a node. |\n")
    model = Model()
    planner.reissue(cfg, _dispute(cfg), model)
    assert "A2: version unknown" in model.validate_prompts[0]
    assert "A1: version baseline" in model.validate_prompts[0]


def test_an_unreadable_versions_file_is_said_so_in_the_prompt(cfg: config.Config) -> None:
    """Rule: with no VERSIONS.md the prompt says it could not be read; the reissue goes on."""
    assert not _versions_file(cfg).exists()
    model = Model()
    assert planner.reissue(cfg, _dispute(cfg), model).startswith("reissued")
    assert "VERSIONS.md could not be read" in model.validate_prompts[0]


def test_a_ticket_without_card_ids_renders_none_for_cards_and_versions(
    cfg: config.Config,
) -> None:
    """Rule: no card id in the ticket -> 'none' (never an empty section or a stray placeholder)."""
    _ticket(cfg, "A slice ticket." + PARTS)
    model = Model()
    planner.reissue(cfg, _dispute(cfg), model)
    prompt = model.validate_prompts[0]
    assert prompt.count("\nnone\n") == 2 and "$" not in prompt


def test_the_planner_validate_prompt_names_the_three_verdicts(cfg: config.Config) -> None:
    """Rule: the shipped prompt defines ok / slicing / scope and the answer shape."""
    raw = (cfg.prompts_dir / "planner_validate.txt").read_text()
    assert all(f'"{v}"' in raw for v in ("ok", "slicing", "scope")) and '"reason"' in raw
    assert set(re.findall(r"\$(\w+)", raw)) == {"ticket", "failure", "cards", "versions"}
    assert "$" not in engine.render(
        cfg, "planner_validate", ticket="t", failure="f", cards="c", versions="v"
    )


# ------------------------------------------------------------------------------ docfix.versions
def test_versions_maps_each_card_to_its_section(cfg: config.Config) -> None:
    """Rule: a card id maps to the '##' section it sits under."""
    _versions_file(cfg).parent.mkdir(exist_ok=True)
    _versions_file(cfg).write_text(VERSIONS_MD)
    assert docfix.versions(cfg) == {"A1": "baseline", "A2": "baseline", "A3": "lenses"}


def test_versions_ignores_rows_before_any_section_and_other_headings(cfg: config.Config) -> None:
    """Rule: a row above the first '##' gets an empty version; '###' does not start a section."""
    _versions_file(cfg).parent.mkdir(exist_ok=True)
    _versions_file(cfg).write_text(
        "| A1 | early. |\n## baseline\n### sub\n| A2 | x. |\n## lenses \n| A3 | y. |\n"
    )
    found = docfix.versions(cfg)
    assert found["A2"] == "baseline" and found["A3"] == "lenses"
    assert found["A1"] == ""


def test_versions_of_a_missing_file_raises_oserror(cfg: config.Config) -> None:
    """Rule: the caller decides what an unreadable file means; the function does not hide it."""
    with pytest.raises(OSError):
        docfix.versions(cfg)


# ------------------------------------------------------------------------------ the chain
def _serve_vprnd(cfg: config.Config, model: Model, monkeypatch: pytest.MonkeyPatch) -> None:
    """vp-rnd serves its moves; its one model call is the same stub (a vprnd prompt)."""
    monkeypatch.setattr(
        planner,
        "_ask",
        lambda _cfg, prompt: json.dumps(
            {"decision": "approve", "tier": 1, "reason": "re-cut inside the same cards"}
        ),
    )
    moves.serve(cfg, "vp-rnd", cli.HANDLERS["vp-rnd"], once=True)


def _serve_planner(cfg: config.Config, model: Model) -> int:
    handler = lambda c, m: planner.reissue(c, m, model)  # noqa: E731
    return moves.serve(cfg, "planner", {"dispute": handler}, once=True)


def test_a_machine_dispute_is_validated_approved_and_reissued_end_to_end(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: dispute -> slicing -> ask -> vp-rnd approves -> validated dispute -> ticket."""
    dispute = _dispute(cfg)
    model = Model(_verdict("slicing", "A1 and A2 belong in one slice"))
    assert _serve_planner(cfg, model) == 1
    assert moves.get(cfg, dispute["id"])["result"] == "validated: slicing, asked vp-rnd"
    assert "SUPERSEDED" not in _queue(cfg)

    _serve_vprnd(cfg, model, monkeypatch)
    (back,) = moves.open_for(cfg, "planner")
    assert back["kind"] == "dispute" and back["from"] == "vp-rnd"
    assert back["text"] == FAILURE
    assert back["payload"]["validated"] is True
    assert back["payload"]["owner_words"].startswith("vp-rnd decision (tier 1)")

    assert _serve_planner(cfg, model) == 1
    assert moves.get(cfg, back["id"])["result"].startswith("reissued")
    assert len(model.validate_prompts) == 1  # the validated dispute was not checked again
    assert "SUPERSEDED" in _queue(cfg)
    assert state.ops_count(cfg, "T-9", "reissued") == 1
    assert state.ops_count(cfg, "T-9", "vprnd_decided") == 1
    assert "vp-rnd decision (tier 1)" in _queue(cfg)  # the ticket carries the decision as words


def test_a_vp_rnd_decision_becomes_a_ticket(cfg: config.Config) -> None:
    """Rule: the words vp-rnd gives must be accepted by the ticket writer (it needs a date)."""
    model = Model()
    move = _dispute(cfg, VP_WORDS, validated=True)
    result = planner.reissue(cfg, move, model)
    assert result.startswith("reissued"), result
    assert "vp-rnd decision (tier 1)" in _queue(cfg)


def test_the_chain_ends_in_the_owner_inbox_at_max_rounds(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: after one reissue the next dispute is asked, then escalated to the owner (bounded)."""
    cfg = dataclasses.replace(cfg, inbox=cfg.state_dir.parent / "owner-inbox" / "INBOX.md")
    model = Model(_verdict("ok"))
    _dispute(cfg, text="round one failure")
    assert _serve_planner(cfg, model) == 1
    assert state.ops_count(cfg, "T-9", "reissued") == 1
    _dispute(cfg, text="round two failure")  # the reissued ticket fails again
    model.validation = _verdict("slicing")
    _serve_planner(cfg, model)  # at the limit: an ask with the history, no ticket, no validation
    assert len(model.validate_prompts) == 1  # only round one was validated
    queue_before = _queue(cfg)
    calls: list[str] = []

    def spy(_c: config.Config, prompt: str) -> str:
        calls.append(prompt)
        return "{}"

    monkeypatch.setattr(planner, "_ask", spy)
    moves.serve(cfg, "vp-rnd", cli.HANDLERS["vp-rnd"], once=True)
    assert calls == []  # vp-rnd does not ask the model at the limit
    text = cfg.inbox.read_text()
    assert "round one failure" in text and "round two failure" in text
    assert moves.open_for(cfg, "planner") == []
    assert _queue(cfg) == queue_before
    assert state.ops_count(cfg, "T-9", "reissued") == 1


def test_a_scope_verdict_goes_to_the_owner_without_a_vp_rnd_model_call(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: scope -> ask 'scope: ...' -> vp-rnd escalates straight to the owner, no dispute."""
    cfg = dataclasses.replace(cfg, inbox=cfg.state_dir.parent / "owner-inbox" / "INBOX.md")
    _dispute(cfg)
    _serve_planner(cfg, Model(_verdict("scope", "card A1 must change")))
    calls: list[str] = []

    def spy(_c: config.Config, prompt: str) -> str:
        calls.append(prompt)
        return "{}"

    monkeypatch.setattr(planner, "_ask", spy)
    moves.serve(cfg, "vp-rnd", cli.HANDLERS["vp-rnd"], once=True)
    assert calls == [] and moves.open_for(cfg, "planner") == []
    assert "scope: card A1 must change" in cfg.inbox.read_text()


def test_vprnd_module_attribute_the_model_seam_exists() -> None:
    """Rule: vp-rnd's one model call goes through planner._ask (the seam tests replace)."""
    assert callable(planner._ask) and hasattr(vprnd, "answer")


def test_the_dated_vp_rnd_words_pass_the_ticket_writer(cfg: config.Config) -> None:
    """Rule: the exact words vp-rnd issues (prefix, date, reason) are accepted by fix_text."""
    text = enqueue.fix_text(
        cfg,
        "Remove only the properties that now have a role.",
        [],
        [],
        VP_WORDS,
        context="c",
        acceptance=["a"],
        done="d",
    )
    assert VP_WORDS in text
