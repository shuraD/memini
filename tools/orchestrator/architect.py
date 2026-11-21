"""The architect role: answer an ``ask`` about the structure of a ticket, or pass it to vp-rnd.

The architect's subject is the contract (files, function signatures, top callers) and the cut
of a work unit that crosses squads. One model run (the Planner's one-shot, no tools; prompt
``architect.txt``) decides the question: the dated words go back to the Planner as a dispute,
the way a vp-rnd Tier 1 decision does, and count as machine words under ``max_rounds``.
Everything else (the round limit, scope, a card or Version change, an error, an unusable
answer) is passed on to vp-rnd with the architect's one-line finding first; vp-rnd stays the
single channel to the owner.
"""

from __future__ import annotations

import json
import re

from tools.orchestrator import engine, enqueue, moves, planner, state, vprnd
from tools.orchestrator.config import Config


def answer(cfg: Config, move: moves.Move) -> str:
    """Return ``owner words: architect decision <date>: <reason>``, or pass the ask to vp-rnd."""
    task, text = move["task"], move["text"]
    if state.ops_count(cfg, task, "reissued") >= cfg.max_rounds - 1:
        return _pass(cfg, move, "the task is at its round limit")  # binds before anything else
    if text.startswith("scope: "):
        return _pass(cfg, move, "a question of scope is not the architect's")
    finding = "the architect's answer was not usable"
    try:
        ticket = next((t for t in enqueue.read_tickets(cfg.stages[0].queue) if t.id == task), None)
        if ticket is None:
            return _pass(cfg, move, "the task has no ticket to judge")
        prompt = engine.render(
            cfg,
            "architect",
            task=task,
            text=text,
            module=ticket.module,
            ticket=ticket.text,
            cards=enqueue.card_text(cfg, ticket.text)[1] or "none",
        )
        verdict = json.loads(re.findall(r"\{.*\}", planner._ask(cfg, prompt), re.DOTALL)[0])
        reason = vprnd.one_line(verdict["reason"])
        if verdict["decision"] == "decide" and reason:
            state.ops_log(cfg, task, "architect_decided", reason)
            words = f"architect decision {state._now()[:10]}: {reason}"
            vprnd.hand_back(cfg, task, "architect", text, words)
            return f"owner words: {words}"
        finding = reason or finding
    except Exception as exc:  # an open move would repeat the model call every cycle
        finding = f"the architect could not answer ({type(exc).__name__})"
    return _pass(cfg, move, finding)


def _pass(cfg: Config, move: moves.Move, finding: str) -> str:
    """Send the ask on to vp-rnd as ``architect: <finding> | <the ask>``."""
    line = " ".join(finding.split()).replace("|", "/")
    moves.post(
        cfg,
        move["task"],
        sender="architect",
        to="vp-rnd",
        kind="ask",
        text=f"architect: {line} | {move['text']}",
    )
    return f"passed to vp-rnd: {line}"
