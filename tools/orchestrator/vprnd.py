"""The vp-rnd role in the autonomous flow: answer an ``ask`` move, decide Tier 1, or escalate.

A decision block of ``DECISIONS.md`` (``**D-nn ...**``) that mentions the task or a card id of
the question in its heading and carries an ``Owner:`` line gives the owner's words (a passing
mention in the body is no decision about the task); no model call. Otherwise one model run (the
Planner's one-shot, no tools; prompt ``vprnd.txt``) decides a Tier 1 question or sends it up to
the owner inbox and tells the vp-rnd session.
"""

from __future__ import annotations

import json
import re

from tools.orchestrator import agent_inbox, docfix, engine, enqueue, gitops, moves, planner, state
from tools.orchestrator.config import Config

_ID = re.compile(r"\b[A-Z]{1,3}-\d+[a-z]?\b")


def answer(cfg: Config, move: moves.Move) -> str:
    """Return ``owner words: <line>`` when a decision covers the move, else escalate."""
    path = cfg.cards_dir.parent / "DECISIONS.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    ids = {move["task"], *_ID.findall(move["text"])}
    for block in re.split(r"\n(?=\*\*D-\d+)", text):
        owner = re.search(r"^Owner:[ \t]*(\S.*)$", block, re.MULTILINE)
        if owner and ids & set(_ID.findall(block.splitlines()[0])):
            return f"owner words: {owner[0].strip()}"
    return _decide(cfg, move, text)


_DECISIONS_CHARS = 6000
_FAILURE = re.compile(
    r"cannot be built as written \((.*)\); a ticket needs|\| failure: (.*)", re.DOTALL
)


def _decide(cfg: Config, move: moves.Move, decisions: str) -> str:
    """Ask the model once; approve a Tier 1 answer, else escalate to the owner and tell vp-rnd."""
    task = move["task"]
    if move["text"].startswith("scope: "):
        return _escalate(cfg, move)
    if state.ops_count(cfg, task, "reissued") >= cfg.max_rounds - 1:
        return _escalate(cfg, move, history=True)
    try:
        ticket = next((t for t in enqueue.read_tickets(cfg.stages[0].queue) if t.id == task), None)
        prompt = engine.render(
            cfg,
            "vprnd",
            task=task,
            kind=move["kind"],
            text=move["text"],
            cards=enqueue.card_text(cfg, ticket.text)[1] if ticket else "none",
            decisions=decisions[-_DECISIONS_CHARS:] or "none",
        )
        found = re.findall(r"\{.*\}", planner._ask(cfg, prompt), re.DOTALL)
        verdict = json.loads(found[0])
        reason = str(verdict["reason"]).strip()
        if verdict["decision"] == "approve" and verdict["tier"] == 1 and reason:
            state.ops_log(cfg, task, "vprnd_decided", reason)
            words = f"vp-rnd decision (tier 1) {state._now()[:10]}: {reason}"
            found_failure = _FAILURE.search(move["text"])
            failure = (found_failure[1] or found_failure[2]) if found_failure else None
            if failure:
                moves.post_once(
                    cfg,
                    task,
                    sender="vp-rnd",
                    to="planner",
                    kind="dispute",
                    text=failure,
                    owner_words=words,
                    validated=move["text"].startswith("slicing: "),
                )
            return f"owner words: {words}"
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        pass
    return _escalate(cfg, move)


def _escalate(cfg: Config, move: moves.Move, *, history: bool = False) -> str:
    """Write the owner inbox entry (with the failures so far at the round limit), tell vp-rnd."""
    task, text = move["task"], move["text"]
    if history:
        rounds = [
            f"round {i}: {m['text'][:200]}"
            for i, m in enumerate(moves.for_task(cfg, task, "dispute"), 1)
        ]
        text += " Failures so far: " + " | ".join(rounds)
    state.inbox_append(cfg, task, "needs the owner", 2, text)
    agent_inbox.notify(cfg, "vp-rnd", f"{task}: escalated to the owner (tier 2): {text}")
    return "escalated to the owner"


_MAX_TESTS = 5
_GATE_PATHS = (".github/", "tools/ci/", ".githooks/", "hooks/")


def _asserts(diff: str, sign: str) -> int:
    return sum(1 for ln in diff.splitlines() if ln.startswith(sign) and "assert" in ln)


def review_stale(cfg: Config, move: moves.Move) -> str:
    """Review the update of existing tests (stale_test_review): light is decided, heavy goes up.

    Heavy (tier 2, the owner decides): a test file deleted, an assertion weakened (more
    ``assert`` lines removed than added), a gate, CI or hook file touched, or more than five
    test files. Everything else is approved here.
    """
    p = move["payload"]
    changed: list[str] = p["changed"]
    reasons = []
    if len(changed) > _MAX_TESTS:
        reasons.append(f"{len(changed)} test files")
    reasons += [f"gate file {c}" for c in changed if any(c.startswith(g) for g in _GATE_PATHS)]
    for path in changed:
        if gitops.blob_sha256(cfg.repo, p["snap"], path) is None:
            reasons.append(f"{path} deleted")
            continue
        diff = gitops.out(cfg.repo, "diff", "--no-color", "-U0", p["qa_sha"], p["snap"], "--", path)
        if _asserts(diff, "-") > _asserts(diff, "+"):
            reasons.append(f"{path} loses assertions")
    if reasons:
        state.inbox_append(
            cfg, move["task"], "heavy test update", 2, f"{move['text']} Heavy: {'; '.join(reasons)}"
        )
        return json.dumps({"approved": False, "reason": "heavy: " + "; ".join(reasons)})
    return json.dumps({"approved": True, "reason": ""})


def review_doc(cfg: Config, move: moves.Move) -> str:
    """Review a card fix (doc_review): light is applied to every document, heavy goes up.

    Light: the fix cites SPEC lines (``SPEC L57``) that exist in the squad's ``SPEC.md``, so the
    new text restates the spec. Anything else (no citation, a line that does not exist, an
    unknown card) is heavy: the owner decides. After applying, the documents must agree.
    """
    p = move["payload"]
    card = enqueue.cards(cfg).get(p["card"])
    reasons = []
    if card is None:
        reasons.append(f"{p['card']} is not a baseline card line")
    else:
        spec = cfg.cards_dir / card.squad / "SPEC.md"
        size = len(spec.read_text(encoding="utf-8").splitlines()) if spec.exists() else 0
        if not p["cites"] or any(not 1 <= n <= size for n in p["cites"]):
            reasons.append("the fix does not cite existing SPEC lines")
    if reasons:
        state.inbox_append(cfg, move["task"], "heavy card fix", 2, f"{move['text']} {reasons[0]}.")
        return json.dumps({"approved": False, "reason": "heavy: " + "; ".join(reasons)})
    changed = docfix.apply(cfg, p["card"], p["new_text"], p["evidence"])
    state.ops_log(
        cfg,
        move["task"],
        "doc_fixed",
        f"{p['card']}: {p['new_text']}",
        files=[str(c) for c in changed],
    )
    drift = docfix.drift(cfg)
    if drift:
        state.inbox_append(cfg, move["task"], "card fix left a difference", 2, "; ".join(drift))
    return json.dumps({"approved": not drift, "reason": "; ".join(drift)})
