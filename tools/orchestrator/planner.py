"""The Planner (FLOW-HARD.md D2): a batch brief in, ready-gated tickets out.

``plan --brief <file>`` reads the brief the owner and vp-rnd wrote (``key: value`` lines:
``version``, ``squad``, ``prefix``, ``cards``, ``out``, ``decisions``), asks one model run to group
the brief's baseline card lines into slices, and checks the answer: every card is in the brief,
none twice, dependencies point backwards, a slice's top caller is written in the brief
(``Top caller: <dotted.module>:<function>``, anywhere in its text). Each slice then goes through
``enqueue`` (the ready gate, D1), which refuses a card slice with no top caller. A card
outside the brief refuses the whole plan. The batch summary is kept under
``<state_dir>/plans/`` and posted to vp-rnd for approval.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tools.orchestrator import (
    agent_inbox,
    docfix,
    engine,
    enqueue,
    gates,
    gitops,
    moves,
    reconcile,
    specstage,
    stages,
    state,
)
from tools.orchestrator.config import Config
from tools.orchestrator.executor import executor_for

_KEYS = ("version", "squad", "prefix", "cards", "out", "decisions")
_SLICE_LINES = "150"


@dataclass(frozen=True)
class Slice:
    """One planned slice: a module, its card lines, the earlier slices it needs, its top caller."""

    module: str
    cards: tuple[str, ...]
    deps: tuple[int, ...]
    top_caller: str = ""


@dataclass(frozen=True)
class Brief:
    """The batch brief: the only hand-off to the Planner."""

    version: str
    squad: str
    prefix: str
    cards: str
    out: str
    decisions: str
    text: str


def read_brief(path: Path) -> Brief:
    """Parse a brief file; every key must be present and non-empty (``out`` may be empty)."""
    text = path.read_text(encoding="utf-8")
    values: dict[str, str] = {}
    key = ""
    for line in text.splitlines():
        head = re.match(r"^([a-z]+):\s*(.*)$", line)
        if head and head[1] in _KEYS:
            key = head[1]
            values[key] = head[2].strip()
        elif key and line.startswith((" ", "\t")):
            values[key] += " " + line.strip()
    missing = [k for k in _KEYS if k != "out" and not values.get(k)]
    if missing:
        msg = f"the brief lacks: {', '.join(missing)}"
        raise ValueError(msg)
    return Brief(**{k: values.get(k, "") for k in _KEYS}, text=text)


def _ask(cfg: Config, prompt: str) -> str:
    """Run the audit engine once (no tools) and return its answer text."""
    name = engine.pick_engines(cfg, None)["audit"]
    with tempfile.TemporaryDirectory(prefix="memini-plan-") as tmp:
        argv = engine.command(
            cfg, name, "audit", workdir=Path(tmp), settings=Path(tmp) / "none.json", prompt=prompt
        )
        res = executor_for("local", cfg.env_allow).run(
            argv,
            cwd=Path(tmp),
            stdin=prompt if cfg.engines[name].stdin else None,
            timeout=cfg.engine_timeout_s,
            pass_env=cfg.engines[name].env,
        )
    return engine.agent_output(res.stdout)[0]


def parse_slices(answer: str, scope: set[str]) -> list[Slice]:
    """Return the slices of the model's answer; raise ``ValueError`` when it breaks a rule."""
    found = re.findall(r"\{.*\}", answer, re.DOTALL)
    try:
        slices = json.loads(found[0])["slices"] if found else None
    except (ValueError, KeyError, TypeError):
        slices = None
    if not isinstance(slices, list) or not slices:
        msg = "the answer is not a JSON object with a list of slices"
        raise ValueError(msg)
    seen: set[str] = set()
    out: list[Slice] = []
    for i, s in enumerate(slices):
        cards = s.get("cards") if isinstance(s, dict) else None
        if not isinstance(cards, list) or not cards or not isinstance(s.get("module"), str):
            msg = f"slice {i} needs a module and cards"
            raise ValueError(msg)
        outside = [c for c in cards if c not in scope]
        if outside:
            msg = f"slice {i} adds card lines that are not in the brief: {', '.join(outside)}"
            raise ValueError(msg)
        twice = [c for c in cards if c in seen]
        if twice:
            msg = f"slice {i} repeats card lines: {', '.join(twice)}"
            raise ValueError(msg)
        seen.update(cards)
        deps = s.get("deps") or []
        if not all(isinstance(d, int) and 0 <= d < i for d in deps):
            msg = f"slice {i} depends on a slice that does not come before it"
            raise ValueError(msg)
        top = s.get("top_caller", "")
        if top is None:
            top = ""
        if not isinstance(top, str):
            msg = f"slice {i} has a top caller that is not text"
            raise ValueError(msg)
        out.append(Slice(s["module"], tuple(cards), tuple(deps), top.strip()))
    return out


def _next_number(cfg: Config, prefix: str) -> int:
    ids = {t.id for t in enqueue.read_tickets(cfg.stages[0].queue)} | state.State(cfg).task_ids()
    nums = [int(m[1]) for i in ids if (m := re.fullmatch(rf"{re.escape(prefix)}-(\d+)b?", i))]
    return max(nums, default=0) + 1


def plan(cfg: Config, brief_path: Path, ask: Callable[[str], str] | None = None) -> list[str]:
    """Plan one batch from its brief; return the summary lines (also kept and posted)."""
    brief = read_brief(brief_path)
    known = enqueue.cards(cfg)
    scope = [c for c in enqueue.card_ids(brief.cards, known) if known.get(c, None)]
    scope = [c for c in scope if known[c].squad == brief.squad]
    if not scope:
        msg = f"no baseline card line of squad {brief.squad} is named by the brief"
        raise ValueError(msg)
    spec = cfg.cards_dir / brief.squad / "SPEC.md"
    prompt = engine.render(
        cfg,
        "planner",
        brief=brief.text,
        squad_spec=spec.read_text(encoding="utf-8") if spec.exists() else "none",
        cards="\n".join(
            f"{c}: {known[c].text}"
            + (f" (old code: {e})" if (e := enqueue.evidence(cfg, c)) else "")
            for c in scope
        ),
        slice_lines=_SLICE_LINES,
    )
    slices = parse_slices((ask or (lambda p: _ask(cfg, p)))(prompt), set(scope))
    named = set(gates.TOP_CALLER.findall(brief.text))  # whole names, never a piece of one
    invented = [s.top_caller for s in slices if s.top_caller and s.top_caller not in named]
    if invented:
        msg = f"a slice names a top caller that is not in the brief: {', '.join(invented)}"
        raise ValueError(msg)
    first = _next_number(cfg, brief.prefix)
    ids = [f"{brief.prefix}-{first + i:02d}" for i in range(len(slices))]
    lines = [f"# Batch plan: {brief.squad} {brief.version} ({brief_path.name})", ""]
    for i, s in enumerate(slices):
        deps = [ids[d] for d in s.deps]
        _, refused = enqueue.enqueue(
            cfg,
            task_id=ids[i],
            module=s.module,
            card_list=s.cards,
            deps=deps,
            top_caller=s.top_caller,
        )
        lines.append(
            f"- {ids[i]} {s.module} cards {', '.join(s.cards)}"
            + (f" after {', '.join(deps)}" if deps else "")
            + (f" top caller {s.top_caller}" if s.top_caller else "")
            + ("  REFUSED: " + "; ".join(refused) if refused else "")
        )
    planned = {c for s in slices for c in s.cards}
    left = [c for c in scope if c not in planned]
    lines += ["", "Card lines of the brief not in any slice: " + (", ".join(left) or "none")]
    out = cfg.state_dir / "plans"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{brief_path.stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    state.ops_log(cfg, brief.prefix, "plan", f"{len(slices)} slices from {brief_path.name}")
    agent_inbox.notify(
        cfg, "vp-rnd", "batch plan for approval (D-4):\n" + "\n".join(lines), "planner"
    )
    return lines


def _supersede(cfg: Config, old: enqueue.Ticket, reason: str, move_id: str) -> None:
    """Comment out the task's lines, set its spec, hold-out and run aside, release its records.

    Nothing is deleted: the old spec file and hold-out directory are renamed
    ``*.superseded-<move>``, so the spec stage writes the new ones without overwriting.
    """
    task_id = old.id
    item = stages.StageItem(task_id, old.module, (), (), "")
    for old_path in (specstage.spec_path(cfg, item), gates.holdout_dir(cfg, task_id)):
        if old_path.exists():
            old_path.rename(old_path.with_name(f"{old_path.name}.superseded-{move_id}"))
    for path in reconcile.queue_files(cfg):
        lines = path.read_text(encoding="utf-8").splitlines()
        if not any(x.split(" ", 1)[0] == task_id for x in lines):
            continue
        marked = [
            f"# SUPERSEDED ({reason}) {x}" if x.split(" ", 1)[0] == task_id else x for x in lines
        ]
        with state.locked(cfg.state_dir / "locks" / f"queue-{path.name}.lock"):
            path.write_text("\n".join(marked) + "\n", encoding="utf-8")
    for stage in cfg.stages:  # a stage that already passed the ticket must take the new one
        rec = stages.StageState(cfg, stage.name)
        if rec.status(task_id) is not None:
            rec.set(task_id, stages.RELEASED, reason)
    st = state.State(cfg)
    if st.task(task_id) is not None:
        st.update(task_id, "released", reason=reason)
    # the disputed run keeps its worktree and branch: the new ticket's run must find none
    gitops.set_aside(cfg.repo, cfg.worktrees_dir, cfg.branch_prefix, task_id, f"sup-{move_id}")


def _verdict(
    cfg: Config, move: moves.Move, old: enqueue.Ticket, ask: Callable[[str], str] | None
) -> tuple[str, str]:
    """Ask the model once whether the ticket matches its version and cards; log the verdict.

    Returns ``(verdict, reason)``; raises on a model error or an unusable answer.
    """
    ids, card_lines = enqueue.card_text(cfg, old.text)
    try:
        version = docfix.versions(cfg)
    except OSError:
        version = {}
    known = enqueue.cards(cfg)
    where = "\n".join(
        f"{i}: version {version.get(i, 'unknown (VERSIONS.md could not be read)')}, "
        f"squad {known[i].squad}"
        for i in ids
    )
    prompt = engine.render(
        cfg,
        "planner_validate",
        ticket=old.text,
        failure=move["text"],
        cards=card_lines or "none",
        versions=where or "none",
    )
    found = re.findall(r"\{.*\}", (ask or (lambda q: _ask(cfg, q)))(prompt), re.DOTALL)
    verdict = json.loads(found[0])
    kind, reason = verdict["verdict"], str(verdict["reason"]).strip()
    if kind not in ("slicing", "scope"):
        kind = "ok"
    state.ops_log(cfg, move["task"], "ticket_validated", f"{kind}: {reason}")
    return str(kind), reason


def _crosses_squads(cfg: Config, ticket: enqueue.Ticket) -> bool:
    """Return True when the cards the ticket names belong to more than one squad."""
    known = enqueue.cards(cfg)
    return len({known[i].squad for i in enqueue.card_ids(ticket.text, known) if i in known}) > 1


def _validate(
    cfg: Config, move: moves.Move, old: enqueue.Ticket, ask: Callable[[str], str] | None
) -> str:
    """Check the ticket against its version and cards; on slicing or scope ask a deciding role.

    A slicing fault of a ticket that crosses squads goes to the architect, every other one to
    vp-rnd. Returns ``<verdict>, asked <role>``, or ``""`` when the reissue goes on (the verdict
    is ok, or there was an error or an unusable answer).
    """
    task = move["task"]
    try:
        kind, reason = _verdict(cfg, move, old, ask)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return ""
    if kind == "ok":
        return ""
    cut = kind == "slicing" and "architect" in cfg.roles and _crosses_squads(cfg, old)
    to = "architect" if cut else "vp-rnd"
    text = f"{kind}: {reason} | failure: {move['text']}"
    agent_inbox.notify(cfg, to, f"{task}: {text}", "planner")
    moves.post(cfg, task, sender="planner", to=to, kind="ask", text=text)
    return f"{kind}, asked {to}"


_CHECKED_CHARS = 300


def reissue(  # noqa: PLR0911
    cfg: Config, move: moves.Move, ask: Callable[[str], str] | None = None
) -> str:
    """Serve a ``dispute`` move: write the D3 change ticket and send it to QA's ready review.

    The owner's words come with the move; without them nothing is written and vp-rnd is asked.
    At the round limit no ticket is written either: the ticket is validated first and the
    verdict goes to vp-rnd (and on to the owner) with the ask.
    Returns the move's result line.
    """
    task, p = move["task"], move["payload"]
    words = str(p.get("owner_words", "")).strip()
    if not words:
        why = (
            f"{task}: cannot be built as written ({move['text']}); a ticket needs the owner's words"
        )
        agent_inbox.notify(cfg, "vp-rnd", why, "planner")
        moves.post(cfg, task, sender="planner", to="vp-rnd", kind="ask", text=why)
        return "no owner words: asked vp-rnd"
    done = state.ops_count(cfg, task, "reissued")
    machine = words == cfg.standing_reissue_words or words.startswith(
        ("vp-rnd decision", "architect decision")
    )
    queue = cfg.stages[0].queue
    old = next((t for t in enqueue.read_tickets(queue) if t.id == task), None)
    if machine and done >= cfg.max_rounds - 1:
        # the Planner looks at the cause first; an error there never opens the round gate
        try:
            if old is None:
                raise ValueError(f"{task} has no ticket in {queue.name}")
            checked = ": ".join(_verdict(cfg, move, old, ask))
        except Exception as exc:
            checked = f"failed: {exc}"
        checked = " ".join(checked.split())[:_CHECKED_CHARS].rstrip(".")
        rounds = [
            f"round {i}: {m['text'][:200]}"
            for i, m in enumerate(moves.for_task(cfg, task, "dispute"), 1)
        ]
        why = (
            f"{task}: round {done + 2} of max_rounds {cfg.max_rounds}, no ticket written; "
            f"it now waits for the owner. Planner validation: {checked}. "
            "Failures so far: " + " | ".join(rounds)
        )
        agent_inbox.notify(cfg, "vp-rnd", why, "planner")
        moves.post(cfg, task, sender="planner", to="vp-rnd", kind="ask", text=why)
        return "round limit reached: asked vp-rnd, waits for the owner"
    if old is None:
        return f"{task} has no ticket in {queue.name}"
    if machine and not p.get("validated"):
        asked = _validate(cfg, move, old, ask)
        if asked:
            return f"validated: {asked}"
    prompt = engine.render(
        cfg, "planner_fix", ticket=old.text, reason=move["text"], owner_words=words
    )
    try:
        fix = json.loads(
            re.findall(r"\{.*\}", (ask or (lambda q: _ask(cfg, q)))(prompt), re.DOTALL)[0]
        )
        text = enqueue.fix_text(
            cfg,
            str(fix["change"]),
            list(fix.get("remove_names", [])),
            list(fix.get("renames", [])),
            words,
            touches=list(fix.get("touches", [])),
            context=str(fix["context"]),
            acceptance=[str(a) for a in fix["acceptance"]],
            done=str(fix["done"]),
            top_caller=enqueue.part(old.text, enqueue.TOP_CALLER),  # the original's, kept
        )
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return f"the Planner answer is not a usable fix: {exc}"
    ticket = enqueue.Ticket(task, old.module, old.deps, old.sources, text)
    others = [t for t in enqueue.read_tickets(queue) if t.id != task]
    found = enqueue.problems(cfg, ticket, others)
    if found:
        return "refused by the ready gate: " + "; ".join(found)
    _supersede(cfg, old, f"cannot be built as written, move {move['id']}", move["id"])
    _, found = enqueue.enqueue(
        cfg,
        task_id=task,
        module=old.module,
        deps=old.deps,
        sources=old.sources.split(",") if old.sources else (),
        text=text,
    )
    state.ops_log(
        cfg, task, "reissued", f"D3 change ticket for QA's ready review, move {move['id']}"
    )
    return "reissued: " + ("; ".join(found) if found else "queued for QA's ready review")


_OWNER_WORDS = re.compile(r'Owner words: "(.*?)"')
_TEST_ID = re.compile(r"tests/[\w/.\-]+\.py::[\w\[\]./\-]+")


def batch_fix(cfg: Config, move: moves.Move, ask: Callable[[str], str] | None = None) -> str:
    """Serve a ``batch_failure`` move: write a new fix ticket for the failing tests (D-9, D-10).

    The authority is the owner's words in the merged slice's own ticket (or in the move); without
    any, nothing is written and vp-rnd is asked. The new ticket is ``<task>x<n>``, on the merged
    slice's module, and goes to QA's ready review like every ticket.
    """
    task, p = move["task"], move["payload"]
    queue = cfg.stages[0].queue
    slice_ticket = next((t for t in enqueue.read_tickets(queue) if t.id == task), None)
    found_words = _OWNER_WORDS.search(slice_ticket.text) if slice_ticket else None
    words = str(p.get("owner_words", "")).strip() or (found_words[1] if found_words else "")
    module = str(p.get("module", "")) or (slice_ticket.module if slice_ticket else "")
    failing = list(dict.fromkeys(_TEST_ID.findall(move["text"])))
    if not words or not module or not failing:
        why = (
            f"{task}: the batch suite failed ({move['text'][:300]}); "
            "a fix ticket needs the owner's words, a module and the failing test ids"
        )
        agent_inbox.notify(cfg, "vp-rnd", why, "planner")
        moves.post(cfg, task, sender="planner", to="vp-rnd", kind="ask", text=why)
        return "no owner words or failing test ids: asked vp-rnd"
    prompt = engine.render(
        cfg,
        "planner_fix",
        ticket=f"(a new fix ticket: the unit suite failed after {task} merged)",
        reason=move["text"],
        owner_words=words,
    )
    try:
        fix = json.loads(
            re.findall(r"\{.*\}", (ask or (lambda q: _ask(cfg, q)))(prompt), re.DOTALL)[0]
        )
        text = enqueue.fix_text(
            cfg,
            str(fix["change"]),
            list(fix.get("remove_names", [])),
            list(fix.get("renames", [])),
            words,
            touches=list(fix.get("touches", [])),
            context=str(fix["context"]),
            acceptance=[
                f"{', '.join(failing)} pass again under every profile",
                *(str(a) for a in fix["acceptance"]),
            ],
            done=str(fix["done"]),
            pinned=sorted(set(re.findall(r"tests/[\w/.\-]+\.py", move["text"]))),
            top_caller=enqueue.part(slice_ticket.text, enqueue.TOP_CALLER) if slice_ticket else "",
        )
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return f"the Planner answer is not a usable fix: {exc}"
    taken = {t.id for t in enqueue.read_tickets(queue)} | state.State(cfg).task_ids()
    n = 1
    while f"{task}x{n}" in taken:
        n += 1
    new_id = f"{task}x{n}"
    _, refused = enqueue.enqueue(cfg, task_id=new_id, module=module, text=text)
    if refused:
        return "refused by the ready gate: " + "; ".join(refused)
    state.ops_log(
        cfg, new_id, "batch_fix", f"fix ticket for the batch failure of {task}, move {move['id']}"
    )
    return f"{new_id}: fix ticket queued for QA's ready review"
