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


def versions(cfg: Config) -> dict[str, str]:
    """Return each ``VERSIONS.md`` line id mapped to its version (the ``##`` section it sits in)."""
    path = cfg.cards_dir.parent / "capabilities" / "VERSIONS.md"
    found: dict[str, str] = {}
    section = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        elif row := _ROW.match(line):
            found[row[1]] = section
    return found


def drift(cfg: Config) -> list[str]:
    """Return the cards whose text differs between a squad ``CARD.md`` and ``VERSIONS.md``."""
    path = cfg.cards_dir.parent / "capabilities" / "VERSIONS.md"
    versions = {m[1]: m[2] for m in _ROW.finditer(path.read_text(encoding="utf-8"))}
    return [
        f"{c.id}: CARD.md and VERSIONS.md differ"
        for c in enqueue.cards(cfg).values()
        if c.id in versions and versions[c.id] != c.text
    ]


def _rewrite(path: Path, key: str, edits: dict[int, str]) -> bool:
    """Set cells of the table row whose first cell is ``key``; return True when the file changed."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out = []
    for line in lines:
        cells = line.rstrip("\n").split("|")
        if len(cells) > max(edits) and cells[1].strip() == key:
            for col, value in edits.items():
                cells[col] = f" {value} "
            line = "|".join(cells) + ("\n" if line.endswith("\n") else "")  # noqa: PLW2901
        out.append(line)
    if out == lines:
        return False
    path.write_text("".join(out), encoding="utf-8")
    return True


def apply(cfg: Config, card: str, text: str, evidence: str = "") -> list[Path]:
    """Set the card's text (and old-code evidence) in every document row; return files changed.

    The squad card and ``VERSIONS.md`` hold the text in column 2; the capability table of the
    area (``archrag.md``...) holds it in column 4 and the evidence in column 6, so a rebuild
    of the cards from the tables writes the same text. Called only by vp-rnd's approval.
    """
    docs = cfg.cards_dir.parent
    changed = [
        p
        for p in (*cfg.cards_dir.glob("*/CARD.md"), docs / "capabilities" / "VERSIONS.md")
        if _rewrite(p, card, {2: text})
    ]
    table = docs / "capabilities" / f"{enqueue.CAPABILITY_FILES[card[0]]}.md"
    edits = {4: text, **({6: f"`{evidence}`"} if evidence else {})}
    if table.exists() and _rewrite(table, card[1:], edits):
        changed.append(table)
    return changed


def post(
    cfg: Config, task: str, *, sender: str, card: str, text: str, why: str, evidence: str = ""
) -> str:
    """Send the proposed text of ``card`` to the Planner (nothing is edited)."""
    if card not in enqueue.cards(cfg):
        msg = f"{card} is not a baseline card line"
        raise ValueError(msg)
    old = enqueue.cards(cfg)[card].text
    return moves.post(
        cfg, task, sender=sender, to="planner", kind="doc_review",
        text=f'{card}: {why} Proposed: "{text}"', card=card, new_text=text, old_text=old,
        evidence=evidence, cites=[int(n) for n in re.findall(r"\bL(\d+)\b", why)],
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
        evidence=p["evidence"], cites=p["cites"],
    )  # fmt: skip
    return json.dumps({"approved": True, "reason": ""})
