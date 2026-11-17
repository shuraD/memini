"""The vp-rnd session-start report (FLOW-HARD.md D13).

A SessionStart hook (startup, clear, resume) for sessions started with ``MEMINI_ROLE=vp-rnd``
runs ``python -m tools.orchestrator.vp_start --state-file <file>`` and the session reads what it
prints: the vp-rnd state file, the open items of ``DECISIONS.md``, ``STATUS.md``, and a facts
block generated from the reflog of main, ``state.json``, the active queue lines and the commits
not pushed. A ``Mismatch`` line says where the state file disagrees with the facts, so a session
never works from a stale memory.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tools.orchestrator import gitops, reconcile, state
from tools.orchestrator.config import Config, load_config

_STATUS_LINES = 120
_WORDS = "merged|rejected|disputed|released|active|failed"
_HEAD = re.compile(r"\bmain\b[^.\n]{0,40}?\b([0-9a-f]{7,12})\b")
_UNPUSHED = re.compile(r"\b(\d+)\s+(?:commits?\s+)?(?:not pushed|unpushed)", re.IGNORECASE)
_TASK = re.compile(rf"\b([A-Z]{{1,3}}-\d+)\b[\s:=(-]{{1,4}}({_WORDS})\b")


def open_decisions(text: str) -> list[str]:
    """Return the headings of the decisions in ``## Open`` whose ``Owner:`` line is empty."""
    start = text.find("## Open")
    if start < 0:
        return []
    end = text.find("\n## ", start + 1)
    body = text[start : end if end > 0 else len(text)]
    out = []
    for block in re.split(r"\n(?=\*\*D-\d+)", body)[1:]:
        owner = re.search(r"^Owner:[ \t]*(.*)$", block, re.MULTILINE)
        if not owner or not owner[1].strip():
            out.append(block.splitlines()[0])
    return out


@dataclass(frozen=True)
class Facts:
    """What git, ``state.json`` and the queues say now."""

    head: str
    reflog: list[str]
    tasks: dict[str, str]
    active: list[str]
    unpushed: list[str]
    has_origin: bool


def facts(cfg: Config) -> Facts:
    """Return the facts: main's head and reflog, task states, active queue lines, unpushed."""
    repo, main = cfg.repo, cfg.main_branch
    st = state.State(cfg)
    done = set(st.queue_done())
    stage_queues = {s.queue for s in cfg.stages}
    active = [
        f"{t} ({path.name})"
        for path in reconcile.queue_files(cfg)
        if path not in stage_queues
        for t in sorted(reconcile.active_ids(path))
        if t not in done
    ]
    log = gitops.git(repo, "log", "--format=%h %s", f"origin/{main}..{main}", check=False)
    return Facts(
        head=gitops.out(repo, "rev-parse", "--short", main),
        reflog=gitops.out(repo, "reflog", main, "-n", "5", "--format=%h %gd %gs").splitlines(),
        tasks={t: str(r.get("status")) for t, r in st.tasks().items()},
        active=active,
        unpushed=log.stdout.splitlines() if log.returncode == 0 else [],
        has_origin=log.returncode == 0,
    )


def mismatches(state_text: str, f: Facts) -> list[str]:
    """Return one line per claim of the state file that the facts contradict."""
    out: list[str] = []
    claimed = _HEAD.search(state_text)
    if claimed and not f.head.startswith(claimed[1][:7]) and not claimed[1].startswith(f.head):
        out.append(f"the state file says main is {claimed[1]}, git says {f.head}")
    count = _UNPUSHED.search(state_text)
    if count and f.has_origin and int(count[1]) != len(f.unpushed):
        out.append(f"the state file says {count[1]} commits not pushed, git says {len(f.unpushed)}")
    seen: set[str] = set()
    for task, claim in _TASK.findall(state_text):
        actual = f.tasks.get(task)
        if actual and actual != claim and task not in seen:
            seen.add(task)
            out.append(f"the state file says {task} {claim}, state.json says {actual}")
    return out


def render(cfg: Config, state_file: Path | None) -> str:
    """Return the whole report."""
    root = cfg.cards_dir.parent
    text = state_file.read_text(encoding="utf-8") if state_file and state_file.exists() else ""
    f = facts(cfg)
    parts = ["[vp-rnd session start]", "== state file ==", text.strip() or "(no state file found)"]
    decisions = root / "DECISIONS.md"
    open_items = open_decisions(decisions.read_text(encoding="utf-8")) if decisions.exists() else []
    parts += ["== open decisions (DECISIONS.md) ==", *(open_items or ["none"])]
    status = root / "STATUS.md"
    if status.exists():
        parts += [
            "== STATUS.md ==",
            *status.read_text(encoding="utf-8").splitlines()[:_STATUS_LINES],
        ]
    counts = {s: sum(1 for v in f.tasks.values() if v == s) for s in sorted(set(f.tasks.values()))}
    parts += [
        "== facts (generated now) ==",
        f"main {f.head}; reflog:",
        *(f"  {line}" for line in f.reflog),
        f"state.json: {counts}; not merged: "
        + (", ".join(f"{t} {s}" for t, s in sorted(f.tasks.items()) if s != "merged") or "none"),
        "active queue lines: " + (", ".join(f.active) or "none"),
        f"not pushed: {len(f.unpushed)}" if f.has_origin else "not pushed: no origin ref",
        *(f"  {line}" for line in f.unpushed[:10]),
    ]
    parts += [f"Mismatch: {m}" for m in mismatches(text, f)]
    return "\n".join(parts) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Print the report for the repository the command runs in."""
    ap = argparse.ArgumentParser(prog="vp_start", description=__doc__.split("\n\n")[0])
    ap.add_argument("--state-file", type=Path)
    args = ap.parse_args(argv)
    cfg = load_config(Path(__file__).resolve().parents[2])
    sys.stdout.write(render(cfg, args.state_file))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
