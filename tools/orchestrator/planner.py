"""The Planner (FLOW-HARD.md D2): a batch brief in, ready-gated tickets out.

``plan --brief <file>`` reads the brief the owner and vp-rnd wrote (``key: value`` lines:
``version``, ``squad``, ``prefix``, ``cards``, ``out``, ``decisions``), asks one model run to group
the brief's baseline card lines into slices, and checks the answer: every card is in the brief,
none twice, dependencies point backwards. Each slice then goes through ``enqueue`` (the ready
gate, D1). A card outside the brief refuses the whole plan. The batch summary is kept under
``<state_dir>/plans/`` and posted to vp-rnd for approval.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tools.orchestrator import agent_inbox, engine, enqueue, state
from tools.orchestrator.config import Config
from tools.orchestrator.executor import executor_for

_KEYS = ("version", "squad", "prefix", "cards", "out", "decisions")
_SLICE_LINES = "150"


@dataclass(frozen=True)
class Slice:
    """One planned slice: a module, its card lines, and the earlier slices it needs."""

    module: str
    cards: tuple[str, ...]
    deps: tuple[int, ...]


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
        out.append(Slice(s["module"], tuple(cards), tuple(deps)))
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
    first = _next_number(cfg, brief.prefix)
    ids = [f"{brief.prefix}-{first + i:02d}" for i in range(len(slices))]
    lines = [f"# Batch plan: {brief.squad} {brief.version} ({brief_path.name})", ""]
    for i, s in enumerate(slices):
        deps = [ids[d] for d in s.deps]
        _, refused = enqueue.enqueue(
            cfg, task_id=ids[i], module=s.module, card_list=s.cards, deps=deps
        )
        lines.append(
            f"- {ids[i]} {s.module} cards {', '.join(s.cards)}"
            + (f" after {', '.join(deps)}" if deps else "")
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
