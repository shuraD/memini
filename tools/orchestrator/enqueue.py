"""The ready gate and ``enqueue``: the only way to add a ticket (FLOW-HARD.md D1, D4).

A ticket is one line of a stage queue: ``<id> <module> <deps|-> <sources,...> '<task text>'``.
``problems`` refuses a ticket (no model call) when its line does not parse, its text holds a deny
word, an IP, a host or a home path, a card ID does not exist, is not baseline or belongs to
another area than the target file, a named input does not exist on ``main``, a graph name has no
role in ``names:`` of ``config/graph_schema.yaml`` (the missing roles go to DevOps' inbox), or the
dependencies are not acyclic. ``enqueue`` writes the text from the cards and appends the line.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tools.orchestrator import agent_inbox, gates, gitops, state
from tools.orchestrator.config import Config

_FIELDS = 5
_CARD = re.compile(r"\b([AIQT])(\d{1,3})\b(?:\s+to\s+\1(\d{1,3})\b)?")
_ROW = re.compile(r"^\|\s*([AIQT]\d+)\s*\|\s*(.+?)\s*\|\s*(?:done|not done)\s*\|", re.MULTILINE)
_PATH = re.compile(r"(?<![\w/.-])((?:src|tests|config|deploy|tools)/[\w./-]+\.\w+)")
_ENV = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
_TICK = re.compile(r"`([A-Za-z_][\w.]*)(?:\(\))?`")
_CYPHER = re.compile(r"[(\[]\w*:`?([A-Za-z_]\w*)")
_CAPABILITY_FILES = {"A": "archrag", "I": "ingestion", "Q": "query", "T": "tools"}


@dataclass(frozen=True)
class Ticket:
    """One parsed ticket line."""

    id: str
    module: str
    deps: tuple[str, ...]
    sources: str
    text: str


@dataclass(frozen=True)
class Card:
    """One baseline line of a card file, and the squad folder it belongs to."""

    id: str
    text: str
    squad: str


def parse(line: str) -> Ticket:
    """Parse a stage-queue line the way the stage reader does; raise ``ValueError`` if it fails."""
    try:
        parts = shlex.split(line)
    except ValueError as exc:
        msg = f"the line does not parse with shell quoting: {exc}"
        raise ValueError(msg) from exc
    if len(parts) < _FIELDS:
        msg = "expected '<id> <module> <deps|-> <sources> <task text>'"
        raise ValueError(msg)
    deps = () if parts[2] == "-" else tuple(d for d in parts[2].split(",") if d)
    return Ticket(parts[0], parts[1], deps, parts[3], " ".join(parts[4:]))


def card_ids(text: str, known: Collection[str] | None = None) -> list[str]:
    """Return the card IDs a text names; ``A24 to A38`` is every known line between them."""
    out: list[str] = []
    for m in _CARD.finditer(text):
        first, last = int(m[2]), int(m[3] or m[2])
        ids = [f"{m[1]}{n}" for n in range(first, last + 1)]
        out += [i for i in ids if known is None or i in known or i in (ids[0], ids[-1])]
    return list(dict.fromkeys(out))


def cards(cfg: Config) -> dict[str, Card]:
    """Return the baseline cards of every squad (``<cards_dir>/<squad>/CARD.md``)."""
    found: dict[str, Card] = {}
    for path in sorted(cfg.cards_dir.glob("*/CARD.md")):
        body = path.read_text(encoding="utf-8")
        start = body.find("## IN: baseline")
        end = body.find("\n## ", start + 1) if start >= 0 else -1
        section = body[start : end if end > 0 else len(body)] if start >= 0 else ""
        for m in _ROW.finditer(section):
            found[m[1]] = Card(m[1], m[2], path.parent.name)
    return found


def evidence(cfg: Config, card_id: str) -> str:
    """Return the old code's ``file:line`` evidence of a card, from ``capabilities/<area>.md``."""
    name = _CAPABILITY_FILES[card_id[0]]
    path = cfg.cards_dir.parent / "capabilities" / f"{name}.md"
    if not path.exists():
        return ""
    for row in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) >= 6 and cells[0] == card_id[1:]:  # noqa: PLR2004 — evidence is column 6
            return cells[5].replace("`", "")
    return ""


def _on_main(cfg: Config, name: str, *, path: bool) -> bool:
    """Return True when a file (or a word) exists on ``main``."""
    if path:
        return bool(gitops.git(cfg.repo, "ls-tree", "main", "--", name, check=False).stdout.strip())
    found = gitops.git(
        cfg.repo,
        "grep",
        "-q",
        "-w",
        "-e",
        name,
        "main",
        "--",
        "src",
        "config",
        "deploy",
        check=False,
    )
    return found.returncode == 0


def _schema(cfg: Config) -> dict[str, Any]:
    text = gitops.out(cfg.repo, "show", "main:config/graph_schema.yaml")
    data = yaml.safe_load(text) or {}
    return data if isinstance(data, dict) else {}


def _names_problems(cfg: Config, text: str, needs: Sequence[str]) -> list[str]:
    """Return the graph names the ticket needs that have no role in ``names:`` (D4)."""
    schema = _schema(cfg)
    roles = {str(v) for g in (schema.get("names") or {}).values() for v in (g or {}).values()}
    named = set(_CYPHER.findall(text))  # (:Label) and [:TYPE]; other words are prose
    wanted = {n for n in named | set(needs) if n not in roles}
    return [
        f"graph name {n} has no role in names: of config/graph_schema.yaml" for n in sorted(wanted)
    ]


def inputs_problems(cfg: Config, ticket: Ticket) -> list[str]:
    """Return the named inputs that do not exist on ``main`` (D4)."""
    out: list[str] = []
    for p in dict.fromkeys(_PATH.findall(ticket.text)):
        if p != ticket.module and not _on_main(cfg, p, path=True):
            out.append(f"file {p} does not exist on main")
    for v in dict.fromkeys(_ENV.findall(ticket.text)):
        if not _on_main(cfg, v, path=False):
            out.append(f"environment variable {v} does not exist on main")
    for t in dict.fromkeys(_TICK.findall(ticket.text)):
        if t.startswith("names."):
            group, _, role = t.removeprefix("names.").partition(".")
            if role not in ((_schema(cfg).get("names") or {}).get(group) or {}):
                out.append(f"{t} is not a role in names:")
        elif not _on_main(cfg, t.rsplit(".", 1)[-1], path=False) and t not in ticket.module:
            out.append(f"{t} does not exist on main")
    return out


def _cycle(tickets: Sequence[Ticket]) -> list[str]:
    """Return the ticket ids on a dependency cycle (explicit deps plus same-file order)."""
    edges: dict[str, set[str]] = {t.id: set(t.deps) for t in tickets}
    last: dict[str, str] = {}
    for t in tickets:
        if t.module in last:
            edges[t.id].add(last[t.module])
        last[t.module] = t.id
    state_of: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> list[str]:
        state_of[node] = 1
        stack.append(node)
        for nxt in sorted(edges.get(node, ())):
            if state_of.get(nxt) == 1:
                return [*stack[stack.index(nxt) :], nxt]
            if nxt not in state_of and nxt in edges and (found := visit(nxt)):
                return found
        stack.pop()
        state_of[node] = 2
        return []

    for t in tickets:
        if t.id not in state_of and (found := visit(t.id)):
            return found
    return []


def workers(tickets: Sequence[Ticket]) -> int:
    """Return how many tickets can run at once: the widest level of the dependency graph."""
    level: dict[str, int] = {}
    last: dict[str, str] = {}
    for t in tickets:
        deps = {*t.deps, *([last[t.module]] if t.module in last else [])}
        level[t.id] = 1 + max((level[d] for d in deps if d in level), default=0)
        last[t.module] = t.id
    return max((list(level.values()).count(n) for n in set(level.values())), default=0)


def problems(
    cfg: Config, ticket: Ticket, earlier: Sequence[Ticket] = (), needs: Sequence[str] = ()
) -> list[str]:
    """Return every reason the ready gate refuses ``ticket`` (empty: it is ready)."""
    deny = [p for p in cfg.deny_patterns if p != r"\bpytest\b"]
    out = [f"text: {p}" for p in gates.text_problems(ticket.text, deny)]
    known = cards(cfg)
    ids = card_ids(ticket.text, known)
    squads_text: dict[str, str] = {}
    for cid in ids:
        card = known.get(cid)
        if card is None:
            out.append(f"card {cid} does not exist or is not a baseline line")
            continue
        spec = cfg.cards_dir / card.squad / "SPEC.md"
        squads_text.setdefault(
            card.squad, spec.read_text(encoding="utf-8") if spec.exists() else ""
        )
        if ticket.module not in squads_text[card.squad]:
            out.append(
                f"card {cid} belongs to {card.squad}, whose spec does not list {ticket.module}"
            )
    out += inputs_problems(cfg, ticket)
    out += _names_problems(cfg, ticket.text, needs)
    cycle = _cycle([*earlier, ticket])
    if cycle:
        out.append("dependencies are not acyclic: " + " -> ".join(cycle))
    return out


def tell_devops(cfg: Config, ticket: Ticket, found: Sequence[str]) -> None:
    """Post the missing graph-name roles of a refused ticket to DevOps' inbox."""
    roles = [p for p in found if "has no role in names:" in p]
    if roles:
        agent_inbox.notify(cfg, "devops", f"{ticket.id} needs roles in names:: " + "; ".join(roles))


def ticket_text(cfg: Config, ids: Sequence[str]) -> str:
    """Write a ticket's text from its cards, with the old code's evidence."""
    known = cards(cfg)
    lines = []
    for cid in ids:
        card = known.get(cid)
        if card is None:
            msg = f"card {cid} does not exist or is not a baseline line"
            raise ValueError(msg)
        ev = evidence(cfg, cid)
        lines.append(f"{cid}: {card.text}" + (f" (old code: {ev})" if ev else ""))
    return " ".join(lines)


def read_tickets(path: Path) -> list[Ticket]:
    """Return the tickets of a stage queue file, in order."""
    if not path.exists():
        return []
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [parse(x) for x in lines if not x.lstrip().startswith("#")]


def enqueue(
    cfg: Config,
    *,
    task_id: str,
    module: str,
    card_list: Sequence[str],
    deps: Sequence[str] = (),
    sources: Sequence[str] = (),
    needs: Sequence[str] = (),
) -> tuple[Ticket, list[str]]:
    """Build the ticket from its cards, run the gate, and append it to the first stage queue."""
    queue = cfg.stages[0].queue
    earlier = read_tickets(queue)
    if any(t.id == task_id for t in earlier):
        return Ticket(task_id, module, tuple(deps), "", ""), [f"{task_id} is already queued"]
    ticket = Ticket(task_id, module, tuple(deps), ",".join(sources), ticket_text(cfg, card_list))
    found = problems(cfg, ticket, earlier, needs)
    if found:
        tell_devops(cfg, ticket, found)
        state.ops_log(cfg, task_id, "enqueue_refused", "; ".join(found)[:300])
        return ticket, found
    line = " ".join(
        shlex.quote(x)
        for x in (ticket.id, module, ",".join(deps) or "-", ticket.sources or "-", ticket.text)
    )
    with state.locked(cfg.state_dir / "locks" / f"queue-{queue.name}.lock"):
        queue.parent.mkdir(parents=True, exist_ok=True)
        with queue.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    state.ops_log(
        cfg, task_id, "enqueue", f"ticket from cards {','.join(card_list)}", module=module
    )
    return ticket, []
