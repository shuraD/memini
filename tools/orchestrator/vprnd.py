"""The vp-rnd role in the autonomous flow: answer an ``ask`` move from DECISIONS.md or escalate.

No model call. A decision block of ``DECISIONS.md`` (``**D-nn ...**``) that mentions the task or a
card id of the question in its heading and carries an ``Owner:`` line gives the owner's words (a
passing mention in the body is no decision about the task); otherwise the question goes to the
owner inbox.
"""

from __future__ import annotations

import json
import re

from tools.orchestrator import gitops, moves, state
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
    state.inbox_append(cfg, move["task"], "needs the owner", 2, move["text"])
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
