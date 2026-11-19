"""A squad proposes a card fix; the Planner lists what must change; vp-rnd decides.

The card and spec files are the gold source: no tool edits them. ``post`` sends the proposal as
a ``doc_review`` move to the Planner. ``review`` (the Planner's handler) names every place that
carries the card (card files, capability table, queued tickets, specs, slice files) and passes the
proposal to vp-rnd, who applies it everywhere (the owner, if it is hard). ``drift`` finds cards
whose text differs between the squad card and ``VERSIONS.md``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tools.orchestrator import enqueue, moves
from tools.orchestrator.config import Config

_ROW = re.compile(r"^\|\s*([AIQT]\d+)\s*\|\s*([^|]+?)\s*\|", re.MULTILINE)


def _files(cfg: Config) -> list[Path]:
    docs = cfg.cards_dir.parent
    found = [*cfg.cards_dir.glob("*/CARD.md"), *cfg.cards_dir.glob("*/SPEC.md")]
    found += [docs / "capabilities" / "VERSIONS.md", *cfg.state_dir.glob("queue*.txt")]
    found += cfg.spec_dir.glob("*/*.md") if cfg.spec_dir.exists() else []
    return [p for p in found if p.exists()]


def places(cfg: Config, card: str) -> list[str]:
    """Return ``file:line`` of every line of the documents that names ``card``."""
    word = re.compile(rf"\b{re.escape(card)}\b")
    return [
        f"{path}:{n}"
        for path in _files(cfg)
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if word.search(line)
    ]


def drift(cfg: Config) -> list[str]:
    """Return the cards whose text differs between a squad ``CARD.md`` and ``VERSIONS.md``."""
    path = cfg.cards_dir.parent / "capabilities" / "VERSIONS.md"
    versions = {m[1]: m[2] for m in _ROW.finditer(path.read_text(encoding="utf-8"))}
    return [
        f"{c.id}: CARD.md and VERSIONS.md differ"
        for c in enqueue.cards(cfg).values()
        if c.id in versions and versions[c.id] != c.text
    ]


def post(cfg: Config, task: str, *, sender: str, card: str, text: str, why: str) -> str:
    """Send the proposed text of ``card`` to the Planner (nothing is edited)."""
    if card not in enqueue.cards(cfg):
        msg = f"{card} is not a baseline card line"
        raise ValueError(msg)
    old = enqueue.cards(cfg)[card].text
    return moves.post(
        cfg, task, sender=sender, to="planner", kind="doc_review",
        text=f'{card}: {why} Proposed: "{text}"', card=card, new_text=text, old_text=old,
    )  # fmt: skip


def review(cfg: Config, move: moves.Move) -> str:
    """Planner: list every place that must change and pass the proposal to vp-rnd."""
    p = move["payload"]
    found = places(cfg, p["card"])
    if not found:
        return json.dumps({"approved": False, "reason": f"{p['card']} is in no document"})
    moves.post(
        cfg, move["task"], sender="planner", to="vp-rnd", kind="doc_review",
        text=f"{move['text']} Places to update: {'; '.join(found)}.",
        card=p["card"], new_text=p["new_text"], old_text=p["old_text"],
    )  # fmt: skip
    return json.dumps({"approved": True, "reason": ""})
