"""The ``spec`` stage: one QA agent turns a task line into a slice spec and its hidden tests.

The agent works in a fresh detached checkout of main and sees only the task line, the source
files named on it (copied into ``.sources/``) and the specs of the tasks it depends on. It may
write ``SPEC.md`` and new ``test_*.py`` files under ``qa_allow``, nothing else. Its output is
accepted only when every check holds (else one retry with the reasons, then the task is
rejected and reported in the owner inbox):

- exactly ``SPEC.md`` plus at least one new test file, no other file added or changed;
- the spec parses, names this task and module, keeps the task line's dependencies and passes
  the same pre-dispatch check as any queued spec;
- the test files define tests, pass the QA lint, and FAIL on main (nothing is implemented yet).

Then the tests are sealed as the task's hold-out (``<holdout_dir>/<ID>/``), the spec is written
to ``<spec_dir>/<module>/<ID>.md`` and the task is handed to the next stage's queue. Files are
never overwritten: an existing hold-out or spec stops the task for the owner.
"""

from __future__ import annotations

import dataclasses
import shlex
import shutil
from pathlib import Path

from tools.orchestrator import dockerexec, engine, gates, gitops, spec, stages, state
from tools.orchestrator.config import CODE_STAGE, Config, StageSpec
from tools.orchestrator.executor import Executor, executor_for

SPEC_FILE = "SPEC.md"
SOURCES = ".sources"
REJECTED, FAILED = "rejected", "failed"


class _Stop(Exception):  # noqa: N818 — internal control flow, not an error type
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status, self.reason = status, reason


def spec_path(cfg: Config, item: stages.StageItem) -> Path:
    """Return where the spec of ``item`` is written: ``<spec_dir>/<module name>/<ID>.md``."""
    module = Path(item.module.rstrip("/")).name.removesuffix(".py").replace("_", "-")
    return cfg.spec_dir / module / f"{item.task_id}.md"


def known_ids(cfg: Config) -> set[str]:
    """Return every task id the chain knows (code queue, stage queues, recorded tasks)."""
    ids = state.State(cfg).task_ids()
    if cfg.queue.exists():
        ids |= {e.task_id for e in state.read_queue(cfg.queue)}
    for s in cfg.stages:
        ids |= {i.task_id for i in stages.read_stage_queue(s.queue)}
    return ids


def run(cfg: Config, stage_spec: StageSpec, task_id: str, job: type[_Job] | None = None) -> int:
    """Run a stage task (the spec stage by default); return 0 when it was handed on, else 1."""
    items = {i.task_id: i for i in stages.read_stage_queue(stage_spec.queue)}
    if task_id not in items:
        print(f"{stage_spec.name}/{task_id}: not in {stage_spec.queue}")
        return 1
    rec = stages.StageState(cfg, stage_spec.name)
    try:
        with state.task_lock(cfg, f"{stage_spec.name}-{task_id}"):
            if rec.status(task_id) not in (None, stages.RELEASED):
                print(f"{stage_spec.name}/{task_id}: already {rec.status(task_id)}")
                return 1
            rec.set(task_id, stages.ACTIVE)
            runner = (job or _Job)(cfg, stage_spec, items[task_id])
            try:
                runner.go()
            except _Stop as stop:
                rec.set(task_id, stop.status, stop.reason)
                state.inbox_append(
                    cfg, task_id, f"{stage_spec.name} stage {stop.status}", 1, stop.reason
                )
                print(f"{stage_spec.name}/{task_id}: {stop.status}: {stop.reason}")
                return 1
            except Exception as exc:  # any crash must leave the task blocked, not active
                reason = f"unexpected error {type(exc).__name__}: {exc}"
                rec.set(task_id, FAILED, reason)
                state.inbox_append(cfg, task_id, f"{stage_spec.name} stage error", 1, reason)
                print(f"{stage_spec.name}/{task_id}: {FAILED}: {reason}")
                return 1
            rec.set(task_id, stages.DONE, f"handed to {stage_spec.next}")
            print(f"{stage_spec.name}/{task_id}: done, handed to {stage_spec.next}")
            return 0
    except state.BusyError as exc:
        print(f"{stage_spec.name}/{task_id}: {exc}")
        return 1


class _Job:
    """One spec-stage task: workspace, agent attempts, checks, seal and hand-off."""

    def __init__(self, cfg: Config, stage_spec: StageSpec, item: stages.StageItem) -> None:
        self.cfg, self.stage, self.item = cfg, stage_spec, item
        self.wt = cfg.worktrees_dir / f"{item.task_id}.{stage_spec.name}"
        self.run_dir = cfg.state_dir / "runs" / f"{item.task_id}.{stage_spec.name}"
        self.host = executor_for("local", cfg.env_allow)
        self.ex: Executor = (
            dockerexec.build_executor(cfg, cfg.env_allow) if cfg.executor == "docker" else self.host
        )
        self.calls = 0

    # ------------------------------------------------------------------ per stage kind
    def writable(self) -> list[str]:
        """Return the paths the agent may write."""
        return [SPEC_FILE, *self.cfg.qa_allow]

    def prompt_values(self) -> dict[str, str]:
        """Return extra prompt placeholders of this stage kind."""
        return {}

    def preflight(self) -> None:
        """Refuse before any agent runs when the output exists already (never overwritten)."""
        target = spec_path(self.cfg, self.item)
        if target.exists() or gates.has_holdout(self.cfg, self.item.task_id):
            raise _Stop(FAILED, f"{target} or its hold-out exists already: nothing overwritten")

    def finish(self, result: list[str]) -> None:
        """Seal the accepted output and hand the task on."""
        self._seal(result, spec_path(self.cfg, self.item))

    def go(self) -> None:
        cfg = self.cfg
        self.preflight()
        if self.wt.exists():
            raise _Stop(FAILED, f"{self.wt} exists from an earlier run: inspect and remove it")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cfg.worktrees_dir.mkdir(parents=True, exist_ok=True)
        base = gitops.rev(cfg.repo, cfg.main_branch)
        gitops.git(cfg.repo, "worktree", "add", "-q", "--detach", str(self.wt), base)
        for cmd in cfg.setup:
            res = self.host.run(cmd, cwd=self.wt, timeout=cfg.gate_timeout_s)
            if res.code != 0:
                raise _Stop(FAILED, f"setup {' '.join(cmd)} failed: {res.output.strip()[-500:]}")
        sources = self._copy_sources()
        failures = "none (first attempt)"
        for attempt in (1, 2):
            self._agent(sources, failures)
            problems, result = self._check()
            if not problems:
                break
            if attempt == 2:  # noqa: PLR2004 — one retry
                raise _Stop(REJECTED, "; ".join(problems)[:1500])
            failures = "Fix only these problems:\n- " + "\n- ".join(problems)
        self.finish(result)
        gitops.git(cfg.repo, "worktree", "remove", "--force", str(self.wt))

    # ------------------------------------------------------------------ inputs
    def _copy_sources(self) -> list[str]:
        """Copy the task's sources and its dependencies' specs into ``.sources/``."""
        root = self.wt / SOURCES
        root.mkdir()
        out = []
        for n, src in enumerate(self.item.sources, 1):
            if not src.is_file():
                raise _Stop(FAILED, f"source {src} is not a readable file")
            dest = root / f"{n:02d}-{src.name}"
            shutil.copyfile(src, dest)
            out.append(dest.relative_to(self.wt).as_posix())
        for dep in self.item.depends_on:
            found = self._dep_spec(dep)
            if found is not None:
                dest = root / "deps" / f"{dep}.md"
                dest.parent.mkdir(exist_ok=True)
                shutil.copyfile(found, dest)
                out.append(dest.relative_to(self.wt).as_posix())
        return out

    def _dep_spec(self, dep: str) -> Path | None:
        if self.cfg.queue.exists():
            for e in state.read_queue(self.cfg.queue):
                if e.task_id == dep and e.spec.is_file():
                    return e.spec
        return None

    def _agent(self, sources: list[str], failures: str) -> None:
        cfg, item = self.cfg, self.item
        name = engine.pick_engines(cfg, None)["qa"]
        self.calls += 1
        settings_path = self.run_dir / f"{self.calls:02d}-{self.stage.name}-settings.json"
        guard = self.run_dir / f"{self.calls:02d}-{self.stage.name}-guard.json"
        settings = engine.settings(
            cfg,
            self.wt,
            writable=self.writable(),
            guard_config=guard,
            protected=[*cfg.protected, "**/conftest.py"],
        )
        settings_path.write_text(engine.json_text(settings), encoding="utf-8")
        prompt = engine.render(
            cfg,
            self.stage.prompt,
            task_id=item.task_id,
            module=item.module,
            depends_on=", ".join(item.depends_on) or "(no dependencies: leave the field empty)",
            task=item.title,
            sources="\n".join(f"  {s}" for s in sources) or "  none",
            qa_allow=", ".join(cfg.qa_allow),
            allow=", ".join(cfg.allow),
            failures=failures,
            **self.prompt_values(),
        )
        argv = engine.command(
            cfg, name, "qa", workdir=self.wt, settings=settings_path, prompt=prompt
        )
        res = self.host.run(
            argv,
            cwd=self.wt,
            stdin=prompt if cfg.engines[name].stdin else None,
            timeout=cfg.engine_timeout_s,
            pass_env=cfg.engines[name].env,
        )
        text, cost = engine.agent_output(res.stdout)
        res = dataclasses.replace(res, stdout=text)
        log = self.run_dir / f"{self.calls:02d}-{self.stage.name}.log"
        log.write_text(f"exit {res.code}\ncost_usd {cost}\n{res.output}", encoding="utf-8")
        with (self.run_dir / "costs.tsv").open("a", encoding="utf-8") as fh:
            fh.write(
                f"{self.calls:02d}\t{self.stage.name}\t{name}\t{'' if cost is None else cost}\n"
            )

    # ------------------------------------------------------------------ checks
    def _added(self) -> tuple[list[str], list[str]]:
        """Return (new files outside .sources and caches, changed tracked files)."""
        listed = gitops.out(self.wt, "ls-files", "-z", "--others", "--exclude-standard")
        added = [
            p
            for p in listed.split("\0")
            if p and not p.startswith(f"{SOURCES}/") and not _cache(p, self.cfg.ignored_ok)
        ]
        changed = [p for p in gitops.out(self.wt, "diff", "--name-only", "-z").split("\0") if p]
        return sorted(added), changed

    def _check(self) -> tuple[list[str], list[str]]:
        """Return (problems, the test files); no problems means the output is accepted."""
        problems, tests = self._file_problems()
        if SPEC_FILE in self._added()[0]:
            problems += self._spec_problems()
        if problems or not tests:
            return problems, tests
        return self._test_problems(tests), tests

    def _file_problems(self) -> tuple[list[str], list[str]]:
        allow = self.cfg.qa_allow
        added, changed = self._added()
        problems = [f"existing file changed: {p}" for p in changed]
        tests = [
            p
            for p in added
            if any(gates.matches(a, p) for a in allow)
            and p.rsplit("/", 1)[-1].startswith("test_")
            and p.endswith(".py")
        ]
        only = f"only {SPEC_FILE} and test_*.py under {', '.join(allow)}"
        problems += [
            f"file not allowed: {p} ({only})" for p in added if p != SPEC_FILE and p not in tests
        ]
        if not tests:
            problems.append("no hidden test file written")
        if SPEC_FILE not in added:
            problems.append(f"no {SPEC_FILE} written")
        return problems, tests

    def _spec_problems(self) -> list[str]:
        item = self.item
        try:
            sp = spec.load(self.wt / SPEC_FILE)
        except spec.SpecError as exc:
            return [str(exc)]
        problems = []
        if sp.module.rstrip("/") != item.module.rstrip("/"):
            problems.append(f"spec module {sp.module!r} is not the task's module {item.module!r}")
        problems += [
            f"spec depends_on lost {d} from the task line"
            for d in item.depends_on
            if d not in sp.depends_on
        ]
        return problems + spec.check(self.cfg, sp, item.task_id, known_ids(self.cfg))

    def _test_problems(self, tests: list[str]) -> list[str]:
        cfg = self.cfg
        if not gates.defines_tests(self.wt, tests):
            return ["the test files define no test function"]
        lint = self._lint(tests)
        if lint:
            return [f"the tests fail lint:\n{lint}"]
        python = gates.python_for(cfg, self.wt)
        argv = [python, "-P", "-m", "pytest", "-q", *cfg.pytest_args, *tests]
        res = self.ex.run(
            argv, cwd=self.wt, env={"PYTHONPATH": "src:."}, timeout=cfg.gate_timeout_s
        )
        if res.code == 0:
            return ["the tests pass on main without any implementation: they prove nothing"]
        return []

    def _lint(self, files: list[str]) -> str:
        python = gates.python_for(self.cfg, self.wt)
        out = []
        for argv in gates.qa_lint_argvs(self.cfg, python, files):
            res = self.host.run(argv, cwd=self.wt, timeout=self.cfg.gate_timeout_s)
            if res.code != 0:  # a silent failure is still a failure
                out.append(res.output.strip() or f"{' '.join(argv)}: exit {res.code}")
        return "\n".join(out)[-1500:]

    # ------------------------------------------------------------------ output
    def _seal(self, tests: list[str], target: Path) -> None:
        """Write the hold-out, then the spec, then hand the task on (in that order)."""
        cfg, item = self.cfg, self.item
        tmp = cfg.holdout_dir / f".tmp-{item.task_id}"
        if tmp.exists():
            raise _Stop(FAILED, f"{tmp} exists from an earlier run: inspect and remove it")
        tmp.mkdir(parents=True)
        for rel in tests:
            name = rel.rsplit("/", 1)[-1]
            if (tmp / name).exists():
                raise _Stop(REJECTED, f"two hidden test files share the name {name}")
            shutil.copyfile(self.wt / rel, tmp / name)
        tmp.rename(gates.holdout_dir(cfg, item.task_id))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as fh:  # never overwrite a spec
            fh.write((self.wt / SPEC_FILE).read_text(encoding="utf-8"))
        if self.stage.next == CODE_STAGE:
            line = shlex.join([item.task_id, str(target)])
        else:  # a later stage reads the same task line
            line = shlex.join(
                [
                    item.task_id,
                    item.module,
                    ",".join(item.depends_on) or "-",
                    ",".join(str(s) for s in item.sources),
                    item.title,
                ]
            )
        stages.handoff(cfg, self.stage, line, item.module)


def _cache(path: str, ignored_ok: tuple[str, ...]) -> bool:
    return any(
        path == d.rstrip("/") or path.startswith(d) or f"/{d}" in f"/{path}" for d in ignored_ok
    )
