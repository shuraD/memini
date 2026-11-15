"""The ``plan`` stage: one planner agent turns a module line into the module's task lines.

Owner 2026-11-16 ("add all of them"): plan around the critical path and agree interfaces first.
The planner reads the module's sources and writes ``PLAN.txt``, one slice per line::

    <ID> <module path> <depends_on,...|-> <one-line task>

The orchestrator accepts the plan only when every check holds (else one retry, then the module
is rejected to the owner inbox):

- ids are ``<PREFIX>-NN`` (the module line's id is the prefix), unique and new to the chain;
- every slice's module lies under the module line's module, inside ``src/memini/``;
- every dependency is a slice of this plan or a task the chain already knows; no cycle;
- contract-first: at least one slice depends on no slice of this plan;
- the critical path (waves of the dependency graph) is at most ``plan_max_waves``.

Then the plan is kept at ``<spec_dir>/<module>/PLAN-<PREFIX>.txt`` (never overwritten) and every
slice goes to the next stage's queue with the module line's sources.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tools.orchestrator import stages
from tools.orchestrator.config import Config
from tools.orchestrator.specstage import FAILED, _Job, _Stop, known_ids

PLAN_FILE = "PLAN.txt"
_FIELDS = 4  # id, module, depends_on, task text (at least one word)
_MAX_SLICES = 40


@dataclass(frozen=True)
class Slice:
    """One planned slice."""

    task_id: str
    module: str
    depends_on: tuple[str, ...]
    task: str


def parse_plan(text: str) -> tuple[list[Slice], list[str]]:
    """Return the slices of a plan and the problems of the lines that cannot be read."""
    out, problems = [], []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            problems.append(f"{PLAN_FILE}:{n}: {exc}")
            continue
        if len(parts) < _FIELDS:
            problems.append(f"{PLAN_FILE}:{n}: expected '<ID> <module> <deps|-> <task>'")
            continue
        deps = () if parts[2] == "-" else tuple(d for d in parts[2].split(",") if d)
        out.append(Slice(parts[0], parts[1], deps, " ".join(parts[3:])))
    return out, problems


def waves(slices: Sequence[Slice]) -> int | None:
    """Return the number of dependency waves inside the plan (None when there is a cycle)."""
    ids = {s.task_id for s in slices}
    deps = {s.task_id: {d for d in s.depends_on if d in ids} for s in slices}
    level: dict[str, int] = {}
    while len(level) < len(deps):
        ready = [t for t, ds in deps.items() if t not in level and ds <= set(level)]
        if not ready:
            return None
        for t in ready:
            level[t] = 1 + max((level[d] for d in deps[t]), default=0)
    return max(level.values(), default=0)


def plan_problems(
    slices: Sequence[Slice], prefix: str, root: str, known: set[str], max_waves: int
) -> list[str]:
    """Return every reason the plan is refused (empty = accepted)."""
    problems = []
    ids = [s.task_id for s in slices]
    if not slices:
        return ["the plan has no slice"]
    if len(slices) > _MAX_SLICES:
        problems.append(f"{len(slices)} slices: at most {_MAX_SLICES} per module plan")
    pattern = re.compile(rf"{re.escape(prefix)}-\d{{2}}")
    problems += [f"{t}: the id must be {prefix}-NN" for t in ids if not pattern.fullmatch(t)]
    problems += [f"{t}: the id is used twice" for t in sorted({t for t in ids if ids.count(t) > 1})]
    problems += [f"{t}: the id is already a task of the chain" for t in ids if t in known]
    base = root.rstrip("/")
    for s in slices:
        mod = s.module.rstrip("/")
        if not (mod == base or mod.startswith(base + "/")) or not mod.startswith("src/memini/"):
            problems.append(f"{s.task_id}: module {s.module} is not under {root}")
        bad = [d for d in s.depends_on if d not in ids and d not in known]
        problems += [f"{s.task_id}: depends on unknown task {d}" for d in bad]
        if s.task_id in s.depends_on:
            problems.append(f"{s.task_id}: depends on itself")
        if not 10 <= len(s.task) <= 600:  # noqa: PLR2004 — a one-line task, not an essay
            problems.append(f"{s.task_id}: the task text must be 10-600 characters")
    count = waves(slices)
    if count is None:
        problems.append("the dependencies form a cycle")
    elif count > max_waves:
        problems.append(
            f"the plan needs {count} waves one after another (critical path); at most "
            f"{max_waves}: depend on the contract slice, not on siblings, wherever possible"
        )
    if not any(not set(s.depends_on) & set(ids) for s in slices):
        problems.append("no slice is free of the plan's own dependencies (contract first)")
    return problems


def plan_path(cfg: Config, item: stages.StageItem) -> Path:
    """Return where the accepted plan is kept: ``<spec_dir>/<module name>/PLAN-<ID>.txt``."""
    module = Path(item.module.rstrip("/")).name.removesuffix(".py").replace("_", "-")
    return cfg.spec_dir / module / f"PLAN-{item.task_id}.txt"


class PlanJob(_Job):
    """One plan-stage task: the planner agent writes PLAN.txt; the checks above decide."""

    def writable(self) -> list[str]:
        """Return the only path the planner may write."""
        return [PLAN_FILE]

    def prompt_values(self) -> dict[str, str]:
        """Return the plan stage's prompt placeholders."""
        return {"prefix": self.item.task_id, "max_waves": str(self.cfg.plan_max_waves)}

    def preflight(self) -> None:
        """Refuse when this module was planned already (a plan is never overwritten)."""
        target = plan_path(self.cfg, self.item)
        if target.exists():
            raise _Stop(FAILED, f"{target} exists already: nothing overwritten")

    def _check(self) -> tuple[list[str], list[str]]:
        added, changed = self._added()
        problems = [f"existing file changed: {p}" for p in changed]
        problems += [f"file not allowed: {p} (only {PLAN_FILE})" for p in added if p != PLAN_FILE]
        if PLAN_FILE not in added:
            return [*problems, f"no {PLAN_FILE} written"], []
        text = (self.wt / PLAN_FILE).read_text(encoding="utf-8")
        slices, unreadable = parse_plan(text)
        problems += unreadable
        problems += plan_problems(
            slices,
            self.item.task_id,
            self.item.module,
            known_ids(self.cfg) - {self.item.task_id},
            self.cfg.plan_max_waves,
        )
        return problems, [text]

    def finish(self, result: list[str]) -> None:
        """Keep the plan, then hand every slice to the next stage with the module's sources."""
        text = result[0]
        target = plan_path(self.cfg, self.item)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as fh:  # never overwrite a plan
            fh.write(text)
        sources = ",".join(str(s) for s in self.item.sources)
        for s in parse_plan(text)[0]:
            line = shlex.join([s.task_id, s.module, ",".join(s.depends_on) or "-", sources, s.task])
            stages.handoff(self.cfg, self.stage, line, s.module)
