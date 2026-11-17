"""Metrics after every batch, from the ledger (FLOW-HARD.md D10), written into ``STATUS.md``.

A batch is a set of task ids (the entries of one code queue). Everything is read from
``state.json`` (a task's status history), the ops ledger, the run folders and ``reconcile``; no
person reports a number. What the ledger cannot tell is printed as such, never estimated.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from tools.orchestrator import reconcile, state
from tools.orchestrator.config import Config

BEGIN, END = "<!-- metrics:begin -->", "<!-- metrics:end -->"
_RULE = re.compile(r"deny scan|literal|parse|AST check|outside the allowed|protected|sealed", re.I)
_BAD = ("rejected", "failed", "disputed")
_MISSING = re.compile(r"\|\s*missing\s*\|", re.I)


def _history(task: dict[str, object]) -> list[dict[str, str]]:
    hist = task.get("history")
    return [h for h in hist if isinstance(h, dict)] if isinstance(hist, list) else []


def _cost(cfg: Config, task_id: str) -> float:
    path = cfg.state_dir / "runs" / task_id / "costs.tsv"
    if not path.exists():
        return 0.0
    cells = (line.split("\t") for line in path.read_text(encoding="utf-8").splitlines())
    return sum(float(c[3]) for c in cells if len(c) > 3 and c[3])  # noqa: PLR2004


def _persons(cfg: Config, since: str) -> int:
    path = cfg.state_dir / state.OPS_LEDGER
    if not path.exists():
        return 0
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return sum(1 for r in rows if r.get("actor") != "orchestrator" and str(r.get("at")) >= since)


def _missing_after_merge(cfg: Config, task_id: str) -> int:
    files = sorted((cfg.state_dir / "runs" / task_id).glob("verifier-*.md"))
    return len(_MISSING.findall(files[-1].read_text(encoding="utf-8"))) if files else 0


def compute(cfg: Config, ids: Sequence[str]) -> list[tuple[str, str, str]]:
    """Return ``(metric, value, target)`` rows for the batch of ``ids``."""
    tasks = {i: state.State(cfg).task(i) or {} for i in ids}
    merged = [i for i, t in tasks.items() if t.get("status") == "merged"]
    first = [i for i in merged if not any(h.get("status") in _BAD for h in _history(tasks[i]))]
    bad = [(i, h) for i, t in tasks.items() for h in _history(t) if h.get("status") in _BAD]
    rule = [i for i, h in bad if _RULE.search(h.get("reason", ""))]
    ticket = {i for i, h in bad if h.get("status") == "disputed"}
    stamps = [h["at"] for t in tasks.values() for h in _history(t) if "at" in h]
    since = min(stamps, default="")
    hours = (
        (
            max(datetime.fromisoformat(s) for s in stamps) - datetime.fromisoformat(since)
        ).total_seconds()
        / 3600
        if stamps
        else 0.0
    )
    thrown = sum(_cost(cfg, i) for i in ids if i not in merged)
    n = max(len(merged), 1)
    return [
        (
            "relay hops per slice (ledger rows by a person, not the orchestrator)",
            f"{_persons(cfg, since) / n:.1f}" if since else "0.0",
            "0",
        ),
        ("slices merged on the first run", f"{len(first)} of {len(ids)}", "80% or more"),
        ("runs failed by a rule violation", str(len(rule)), "0"),
        ("reruns caused by a ticket (disputes)", str(len(ticket)), "0"),
        (
            "card lines marked missing after a merge",
            str(sum(_missing_after_merge(cfg, i) for i in merged)),
            "0",
        ),
        ("state/queue/main mismatches", str(len(reconcile.mismatches(cfg))), "0"),
        ("elapsed hours, first start to last event", f"{hours:.1f}", "published"),
        ("money on thrown-away runs (USD)", f"{thrown:.2f}", "published"),
        ("flow efficiency", "not measured: the ledger holds no stage times", "published"),
    ]


def render(rows: Sequence[tuple[str, str, str]], ids: Sequence[str], when: str) -> str:
    """Return the STATUS.md section for one batch."""
    lines = [
        f"## Batch metrics ({when}): {', '.join(ids)}",
        "",
        "| Metric | Value | Target |",
        "|---|---|---|",
    ]
    return "\n".join([*lines, *(f"| {a} | {b} | {c} |" for a, b, c in rows)])


def write(cfg: Config, ids: Sequence[str], when: str) -> Path:
    """Replace the metrics section of ``STATUS.md`` (kept between two markers)."""
    path = cfg.cards_dir.parent / "STATUS.md"
    body = f"{BEGIN}\n{render(compute(cfg, ids), ids, when)}\n{END}"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if BEGIN in text and END in text:
        head, _, rest = text.partition(BEGIN)
        text = head + body + rest.partition(END)[2]
    else:
        text = text.rstrip() + "\n\n" + body + "\n"
    path.write_text(text, encoding="utf-8")
    state.ops_log(cfg, "batch", "metrics", f"{len(ids)} tasks written to {path.name}")
    return path
