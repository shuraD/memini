"""The ready gate and ``enqueue``: the only way to add a ticket (FLOW-HARD.md D1, D4).

A ticket is one line of a stage queue: ``<id> <module> <deps|-> <sources,...> '<task text>'``.
``problems`` refuses a ticket (no model call) when its line does not parse, its text holds a deny
word, an IP, a host or a home path, a card ID does not exist, is not baseline or belongs to
another area than the target file, a named input does not exist on ``main``, a graph name has no
role in ``names:`` of ``config/graph_schema.yaml`` (the missing roles go to DevOps' inbox), or the
dependencies are not acyclic, or the text lacks its ``Context:``, ``Acceptance:`` or ``Done:``
part (every ticket carries all three), or it names card lines without a
``Top caller: <dotted.module>:<function>`` part (FLOW-HYBRID.md: a card is built as a working path
from a named top caller). ``enqueue`` writes the text from the cards and queues it.
"""

from __future__ import annotations

import ast
import re
import shlex
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tools.orchestrator import agent_inbox, engine, gates, gitops, state
from tools.orchestrator.config import Config

_FIELDS = 5
_TYPED = "src/memini/infra/graph/schema.py"  # the typed Graph<Group> name classes
_CARD = re.compile(r"\b([AIQT])(\d{1,3})\b(?:\s+to\s+\1(\d{1,3})\b)?")
_ROW = re.compile(r"^\|\s*([AIQT]\d+)\s*\|\s*(.+?)\s*\|\s*(?:done|not done)\s*\|", re.MULTILINE)
_PATH = re.compile(r"(?<![\w/.-])((?:src|tests|config|deploy|tools)/[\w./-]+\.\w+)")
_ENV = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
_TICK = re.compile(r"`([A-Za-z_][\w.]*)(?:\(\))?`")
_CYPHER = re.compile(r"[(\[]\w*:`?([A-Za-z_]\w*)")
_PARTS = ("Context", "Acceptance", "Done")
_HOST = "host name: "  # the prefix of gates.text_problems' host finding
TOP_CALLER = "Top caller"  # the optional part ``Top caller: <dotted.module>:<function>``
_LABELS = (
    rf"Context|Acceptance|Done|{TOP_CALLER}|Owner words|Existing tests that pin this"
    r"|Consumers of \S+"
)
CAPABILITY_FILES = {"A": "archrag", "I": "ingestion", "Q": "query", "T": "tools"}


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


def part(text: str, label: str) -> str:
    """Return the text of a ticket's ``<label>:`` part (up to the next labelled part), or ``""``."""
    found = re.search(
        rf"(?<![\w-]){label}:\s*(.*?)\s*(?=(?<![\w-])(?:{_LABELS}):|$)", text, re.DOTALL
    )
    return found[1].strip().rstrip(".").strip() if found else ""


def acceptance_items(text: str) -> list[str]:
    """Return the items of a ticket's ``Acceptance:`` part, split on ``;``."""
    return [i.strip() for i in part(text, "Acceptance").split(";") if i.strip()]


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


def card_text(cfg: Config, ticket_text: str) -> tuple[list[str], str]:
    """Return the ticket's card ids and the one text every agent reads for them.

    The spec QA, the QA, the coder, the audit and the Verifier all get this exact text.
    """
    known = cards(cfg)
    ids = [i for i in card_ids(ticket_text, known) if i in known]
    lines = [f"{i}: {known[i].text} (old code: {evidence(cfg, i)})" for i in ids]
    return ids, "\n".join(lines)


def evidence(cfg: Config, card_id: str) -> str:
    """Return the old code's ``file:line`` evidence of a card, from ``capabilities/<area>.md``."""
    name = CAPABILITY_FILES[card_id[0]]
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


def _typed_fields(cfg: Config) -> dict[str, set[str]]:
    """Return the fields of the typed ``Graph<Group>`` classes on main, by group name."""
    tree = ast.parse(gitops.out(cfg.repo, "show", f"main:{_TYPED}"))
    return {
        node.name.removeprefix("Graph").lower(): {
            s.target.id
            for s in node.body
            if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
        }
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.startswith("Graph")
    }


def _names_problems(cfg: Config, ticket: Ticket, needs: Sequence[str]) -> list[str]:
    """Return graph names a ticket needs with no role in ``names:`` or no typed field (D4).

    The ticket that edits the typed classes adds the fields, so it skips the typed-field check.
    """
    text = ticket.text
    names = {g: dict(r or {}) for g, r in (_schema(cfg).get("names") or {}).items()}
    by_name = {str(v): (g, k) for g, r in names.items() for k, v in r.items()}
    named = set(_CYPHER.findall(text)) | set(needs)  # (:Label) and [:TYPE]; prose is not scanned
    out = [
        f"graph name {n} has no role in names: of config/graph_schema.yaml"
        for n in sorted(n for n in named if n not in by_name)
    ]
    wanted = {by_name[n] for n in named if n in by_name}
    for t in _TICK.findall(text):
        group, _, role = t.removeprefix("names.").partition(".")
        if t.startswith("names.") and role in names.get(group, {}):
            wanted.add((group, role))
    if ticket.module == _TYPED:
        return out
    typed = _typed_fields(cfg)
    out += [
        f"role {g}.{k} is in names: but has no field in {_TYPED}"
        for g, k in sorted(wanted)
        if k not in typed.get(g, set())
    ]
    return out


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
    found = gates.text_problems(ticket.text, deny)
    top = part(ticket.text, TOP_CALLER)
    if top:  # a module path such as x.app is no host: hosts are looked for outside this value
        rest = re.sub(rf"({TOP_CALLER}:\s*){re.escape(top)}", r"\1", ticket.text, count=1)
        found = [p for p in found if not p.startswith(_HOST)]
        found += [p for p in gates.text_problems(rest, ()) if p.startswith(_HOST)]
    out = [f"text: {p}" for p in found]
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
        # a ticket with a top caller is one path through every layer: it may cross squads
        if not top and ticket.module not in squads_text[card.squad]:
            out.append(
                f"card {cid} belongs to {card.squad}, whose spec does not list {ticket.module}"
            )
    out += [
        f"the ticket has no {label}: part (it needs Context:, Acceptance: and Done:)"
        for label in _PARTS
        if not part(ticket.text, label)
    ]
    if top and not gates.TOP_CALLER.fullmatch(top):
        out.append(f"{TOP_CALLER}: {top} is not <dotted.module>:<function>")
    elif not top and any(c in known for c in ids):
        out.append(
            f"the ticket names card lines but has no {TOP_CALLER}: part "
            "(<dotted.module>:<function>)"
        )
    out += inputs_problems(cfg, ticket)
    out += _names_problems(cfg, ticket, needs)
    cycle = _cycle([*earlier, ticket])
    if cycle:
        out.append("dependencies are not acyclic: " + " -> ".join(cycle))
    return out


def tell_devops(cfg: Config, ticket: Ticket, found: Sequence[str]) -> None:
    """Post the missing graph-name roles of a refused ticket to DevOps' inbox."""
    roles = [p for p in found if "has no role in names:" in p]
    if roles:
        agent_inbox.notify(cfg, "devops", f"{ticket.id} needs roles in names:: " + "; ".join(roles))


def _top_part(top_caller: str) -> str:
    """Return the ``Top caller:`` part of a ticket text (``""`` without a top caller)."""
    return f" {TOP_CALLER}: {top_caller.strip()}." if top_caller.strip() else ""


def ticket_text(cfg: Config, ids: Sequence[str], top_caller: str = "") -> str:
    """Write a ticket's text from its cards, with the old code's evidence (ticket_cards.txt)."""
    known = cards(cfg)
    lines = []
    for cid in ids:
        card = known.get(cid)
        if card is None:
            msg = f"card {cid} does not exist or is not a baseline line"
            raise ValueError(msg)
        ev = evidence(cfg, cid)
        lines.append(f"{cid}: {card.text}" + (f" (old code: {ev})" if ev else ""))
    return engine.render(cfg, "ticket_cards", cards=" ".join(lines)).strip() + _top_part(top_caller)


def read_tickets(path: Path) -> list[Ticket]:
    """Return the tickets of a stage queue file, in order."""
    if not path.exists():
        return []
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [parse(x) for x in lines if not x.lstrip().startswith("#")]


def consumers(cfg: Config, name: str) -> list[str]:
    """Return the files of ``src`` and ``tests`` on main that use ``name`` (D3)."""
    found = gitops.git(
        cfg.repo, "grep", "-l", "-w", "-e", name, "main", "--", "src", "tests", check=False
    )
    return sorted(line.removeprefix("main:") for line in found.stdout.splitlines() if line)


def fix_text(
    cfg: Config,
    change: str,
    remove_names: Sequence[str],
    renames: Sequence[str],
    owner_words: str,
    *,
    touches: Sequence[str] = (),
    pinned: Sequence[str] = (),
    context: str = "",
    acceptance: Sequence[str] = (),
    done: str = "",
    top_caller: str = "",
) -> str:
    """Write a structured fix ticket's text and append its consumer list (D3).

    ``context`` (what failed: failing test ids, error text, file), ``acceptance`` (one item per
    entry) and ``done`` (the observable end state) become the ``Context:``, ``Acceptance:`` and
    ``Done:`` parts; the ready gate refuses a ticket without all three. ``top_caller`` becomes
    the ``Top caller:`` part, which a fix that names card lines needs too.

    ``touches`` are names the change adds to or extends and ``pinned`` are test files already
    known to pin it: every existing test that mentions a touched name, or a removed or renamed
    one, is listed as "Existing tests that pin this", and the slice may update exactly those.

    ``renames`` are ``old=new``. The owner's words must carry a date; without them there is no
    ticket. Every removed or renamed name is searched in ``src`` and ``tests`` on main.
    """
    if not re.search(r"\d{4}-\d{2}-\d{2}", owner_words):
        msg = "owner-words must quote the owner and give the date (YYYY-MM-DD)"
        raise ValueError(msg)
    bad = [r for r in renames if r.count("=") != 1 or not all(r.split("="))]
    if bad:
        msg = f"rename must be old=new: {', '.join(bad)}"
        raise ValueError(msg)
    parts = [f"Context: {context.strip().rstrip('.')}."] if context.strip() else []
    parts += [change.strip().rstrip(".") + ".", f'Owner words: "{owner_words.strip()}".']
    if remove_names:
        parts.append("Remove: " + ", ".join(remove_names) + ".")
    if renames:
        parts.append("Rename: " + ", ".join(r.replace("=", " -> ") for r in renames) + ".")
    names = [*remove_names, *(r.split("=")[0] for r in renames)]
    tests = set(pinned)
    for name in names:
        files = consumers(cfg, name)
        parts.append(f"Consumers of {name}: " + (", ".join(files) if files else "none") + ".")
        tests |= {f for f in files if f.startswith("tests/")}
    for name in touches:
        tests |= {f for f in consumers(cfg, name) if f.startswith("tests/")}
    if tests:
        parts.append("Existing tests that pin this: " + ", ".join(sorted(tests)) + ".")
    items = "; ".join(i.strip().rstrip(".").replace(";", ",") for i in acceptance if i.strip())
    if items:
        parts.append(f"Acceptance: {items}.")
    if done.strip():
        parts.append(f"Done: {done.strip().rstrip('.')}.")
    return " ".join(parts) + _top_part(top_caller)


_PINNED = re.compile(r"Existing tests that pin this: ([^.]*\.py(?:, [^.]*\.py)*)\.")


def pinned_tests(text: str) -> list[str]:
    """Return the existing test files a ticket's text lists as pinning its change."""
    found = _PINNED.search(text)
    return found[1].split(", ") if found else []


def enqueue(
    cfg: Config,
    *,
    task_id: str,
    module: str,
    card_list: Sequence[str] = (),
    deps: Sequence[str] = (),
    sources: Sequence[str] = (),
    needs: Sequence[str] = (),
    text: str = "",
    top_caller: str = "",
) -> tuple[Ticket, list[str]]:
    """Build the ticket (from its cards and ``top_caller``, or the fix ``text``), gate, queue it."""
    queue = cfg.stages[0].queue
    earlier = read_tickets(queue)
    if any(t.id == task_id for t in earlier):
        return Ticket(task_id, module, tuple(deps), "", ""), [f"{task_id} is already queued"]
    body = text or ticket_text(cfg, card_list, top_caller)
    ticket = Ticket(task_id, module, tuple(deps), ",".join(sources), body)
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
    origin = f"cards {','.join(card_list)}" if card_list else "a structured fix"
    state.ops_log(cfg, task_id, "enqueue", f"ticket from {origin}", module=module)
    return ticket, []
