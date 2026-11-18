"""The vp-rnd role in the autonomous flow: answer an ``ask`` move from DECISIONS.md or escalate.

No model call. A decision block of ``DECISIONS.md`` (``**D-nn ...**``) that mentions the task or a
card id of the question and carries an ``Owner:`` line gives the owner's words; otherwise the
question goes to the owner inbox.
"""

from __future__ import annotations

import re

from tools.orchestrator import moves, state
from tools.orchestrator.config import Config

_ID = re.compile(r"\b[A-Z]{1,3}-\d+[a-z]?\b")


def answer(cfg: Config, move: moves.Move) -> str:
    """Return ``owner words: <line>`` when a decision covers the move, else escalate."""
    path = cfg.cards_dir.parent / "DECISIONS.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    ids = {move["task"], *_ID.findall(move["text"])}
    for block in re.split(r"\n(?=\*\*D-\d+)", text):
        owner = re.search(r"^Owner:[ \t]*(\S.*)$", block, re.MULTILINE)
        if owner and ids & set(_ID.findall(block)):
            return f"owner words: {owner[0].strip()}"
    state.inbox_append(cfg, move["task"], "needs the owner", 2, move["text"])
    return "escalated to the owner"
