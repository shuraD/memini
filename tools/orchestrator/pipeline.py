"""One atomic slice: spec check, QA tests, coder attempts, gates, audit, squash merge.

Flow (owner 2026-11-14)::

    $0 spec check -> claim (WIP) -> QA agent writes tests -> tests must fail on the base
    -> seal -> coder (max N attempts) -> path gate + sealed hash -> fresh gate checkout:
    QA tests per env profile, CI gates, hold-out on the host -> one-shot audit
    -> merge: rebase on main, fresh integration checkout, ledger rows, ONE squash commit
       (hooks run), all gates again, fast-forward main (compare-and-swap; repeat if moved)

Agents and gates run through the executor; git plumbing and the hold-out run on the host.
Gates never run in an agent's worktree: every gate run gets a fresh checkout of the commit
under test with its own environment, built by ``setup`` from the lock file.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tools import ledger
from tools.orchestrator import (
    agent_inbox,
    astcheck,
    audit,
    config,
    dockerexec,
    engine,
    enqueue,
    gates,
    gitops,
    moves,
    spec,
    stages,
    state,
)
from tools.orchestrator.config import Config
from tools.orchestrator.executor import Executor, Result, allowed_env, executor_for

MERGED, REJECTED, DISPUTED, FAILED = "merged", "rejected", "disputed", "failed"
LEDGER_FILES = ("ledgers/changes.sql", "ledgers/features.sql")
_DIFF_LIMIT = 200_000
_REVIEW_REJECTS = 2  # the first reject goes back to the coder, the second to the owner
_MOVE_WAIT_S = 6 * 3600  # how long a slice waits for QA's answer to its test review
_REFS = Path("<all refs>")  # a pseudo entry in the watched files: the for-each-ref listing


@dataclass(frozen=True)
class Task:
    """What to run: a task id, its spec file and the optional choices of the caller."""

    task_id: str
    spec: Path
    engine: str | None = None
    paths: tuple[str, ...] = ()  # narrows the spec's paths
    worker: str | None = None


@dataclass(frozen=True)
class Outcome:
    """How a task ended; ``worktree`` is where to inspect it when it did not merge."""

    task_id: str
    status: str
    reason: str
    worktree: Path | None = None
    commit: str | None = None


class _Stop(Exception):  # noqa: N818 — internal control flow, not an error type
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def _exclude(paths: Sequence[str]) -> list[str]:
    """Return git pathspecs that leave ``paths`` out of a diff (empty list = whole diff)."""
    return ["--", ".", *(f":(exclude){p}" for p in paths)] if paths else []


def _test_rule(edits: bool, files: Sequence[str]) -> str:
    """Return the coder prompt's rule about the tests."""
    names = " ".join(files)
    if not edits:
        return f"- The acceptance tests are sealed and must not change: {names}"
    return (
        f"- The tests are shown in full and you may edit one only when it contradicts the "
        f"specification or the gate rules: {names}\n"
        "  The originals are kept and every change is reviewed by QA. A test changed only to make "
        "it pass, or one that checks less than the specification says, is rejected. Do not "
        "delete a test file."
    )


def _dispute_rule(edits: bool) -> str:
    """Return the coder prompt's rule about a test you cannot satisfy."""
    if edits:
        return (
            "- If you cannot satisfy the specification even with a justified test edit, print "
            "one line\n  TEST-DISPUTE: <the reason, with evidence from the specification>\n"
            "  and stop."
        )
    return (
        "- If you are convinced a sealed test is wrong, do not work around it: print one line\n"
        "  TEST-DISPUTE: <the reason, with evidence from the specification>\n  and stop."
    )


def precheck(
    cfg: Config, sp: spec.Spec, task_id: str, narrowed: Sequence[str], known: Collection[str]
) -> list[str]:
    """Return every reason not to dispatch (the $0 check: no engine runs)."""
    if not _TASK_ID.fullmatch(task_id):
        return [f"task id {task_id!r} must match {_TASK_ID.pattern}"]
    problems = spec.check(cfg, sp, task_id, known, tuple(narrowed))
    if not cfg.identity_name or not cfg.identity_email:
        problems.append(
            "no commit identity: set [identity] name/email or the repo's git user.name/user.email"
        )
    if cfg.holdout_required and not gates.has_holdout(cfg, task_id):
        problems.append(f"no hold-out tests for {task_id} in the configured hold-out dir")
    st = state.State(cfg)
    problems += [
        f"depends_on {d} is not merged yet"
        for d in sp.depends_on
        if d in known and not st.is_merged(d)
    ]
    problems += st.overlaps(tuple(narrowed) or sp.paths, task_id)
    if any(p.all_flags for p in cfg.profiles):
        on_main = gitops.git(
            cfg.repo, "cat-file", "-e", f"{cfg.main_branch}:{cfg.flags_file}", check=False
        )
        if not cfg.flags_file or on_main.returncode != 0:
            problems.append(
                f"the all-flags profile needs the flags file {cfg.flags_file!r} on "
                f"{cfg.main_branch}"
            )
    return problems


def _waits(problem: str) -> bool:
    """Return True for a refusal that only means "not yet".

    Another task holds the paths, or a dependency is not merged: the task stays queued and runs
    later, so it is no dead end.
    """
    return problem.startswith("paths overlap the active task") or problem.endswith(
        "is not merged yet"
    )


def _refuse(cfg: Config, task_id: str, problems: list[str]) -> Outcome:
    state.inbox_append(cfg, task_id, "spec rejected before dispatch", 1, "; ".join(problems))
    if not all(_waits(p) for p in problems):
        _dead_end(cfg, task_id, REJECTED, "; ".join(problems))
    return Outcome(task_id, REJECTED, "; ".join(problems))


def run_task(cfg: Config, task: Task, known_ids: Collection[str] = ()) -> Outcome:
    """Run one task end to end and return its outcome (the state records it)."""
    try:
        sp = spec.load(task.spec)
    except spec.SpecError as exc:
        return _refuse(cfg, task.task_id, [str(exc)])
    try:
        with state.run_lock(cfg, task.task_id):
            return _run_locked(cfg, task, sp, known_ids)
    except state.BusyError as exc:
        return Outcome(task.task_id, REJECTED, str(exc))


def _run_locked(cfg: Config, task: Task, sp: spec.Spec, known_ids: Collection[str]) -> Outcome:
    st = state.State(cfg)
    known = {*known_ids, *st.task_ids()}
    problems = precheck(cfg, sp, task.task_id, task.paths, known)
    paths = gates.normalize_paths(task.paths or sp.paths) if not problems else ()
    engines: dict[str, str] = {}
    if not problems:
        try:
            engines = engine.pick_engines(cfg, task.engine)
            if cfg.executor == "local" and engines["coder"] not in cfg.local_coder_engines:
                msg = (
                    f"coder engine {engines['coder']} is refused while the executor is local "
                    f"(allowed: {', '.join(cfg.local_coder_engines)})"
                )
                raise state.WipConflictError(msg)
            st.claim(
                task.task_id,
                task.worker or cfg.worker,
                paths,
                engines=",".join(f"{r}={e}" for r, e in sorted(engines.items())),
            )
        except (state.WipConflictError, KeyError) as exc:
            problems = [str(exc)]
    if problems:
        return _refuse(cfg, task.task_id, problems)
    run = _Run(cfg, task.task_id, sp, paths, engines)
    try:
        outcome = run.go()
    except gitops.GitError as exc:
        outcome = Outcome(task.task_id, FAILED, str(exc), run.wt)
    except Exception as exc:  # any crash must leave the task blocked, not active
        reason = f"unexpected error {type(exc).__name__}: {exc}"
        state.inbox_append(cfg, task.task_id, "orchestrator error (task blocked)", 1, reason)
        _tool_broke(cfg, task.task_id, reason)
        outcome = Outcome(task.task_id, FAILED, reason, run.wt)
    st.update(
        task.task_id,
        outcome.status,
        reason=outcome.reason[:1000],
        worktree=str(outcome.worktree or ""),
        commit=outcome.commit or "",
    )
    _dead_end(cfg, task.task_id, outcome.status, outcome.reason)
    if role := config.role_for_module(cfg, sp.module):
        agent_inbox.notify(cfg, role, f"{task.task_id}: {outcome.status}: {outcome.reason[:300]}")
    return outcome


def _dead_end(cfg: Config, task_id: str, status: str, reason: str) -> None:
    """Send a task that ends rejected, failed or disputed to the planner (no person relays it)."""
    if status in (REJECTED, FAILED, DISPUTED) and not moves.has_open(cfg, task_id, "tool"):
        moves.post_once(
            cfg,
            task_id,
            sender=cfg.worker or "squad",
            to="planner",
            kind="dispute",
            text=f"{task_id} {status}: {reason[:1500]}",
            owner_words=cfg.standing_reissue_words,
        )


def _tool_broke(cfg: Config, task_id: str, reason: str) -> None:
    """Send the task to DevOps' queue: a tool or gate crashed, not the slice's code."""
    moves.post(
        cfg,
        task_id,
        sender=cfg.worker or "squad",
        to="devops",
        kind="tool",
        text=reason,
    )


class _Run:
    """The mutable bookkeeping of one task run."""

    def __init__(
        self,
        cfg: Config,
        task_id: str,
        sp: spec.Spec,
        paths: tuple[str, ...],
        engines: dict[str, str],
    ) -> None:
        self.cfg, self.task_id, self.sp, self.paths, self.engines = cfg, task_id, sp, paths, engines
        self.sandbox = (
            dockerexec.build_executor(cfg, cfg.env_allow) if cfg.executor == "docker" else None
        )
        local = executor_for("local", cfg.env_allow)
        # coder code (tests, sandbox gates, hold-out) runs in the sandbox when there is one;
        # the agent CLIs, setup and the static/git gates always run on the host (workflow §8)
        self.ex = _Watched(self.sandbox or local, self)
        self.host = _Watched(local, self)
        self.repo = cfg.repo
        self.branch = cfg.branch_prefix + task_id
        self.wt = cfg.worktrees_dir / task_id
        self.gate_wt = cfg.worktrees_dir / f"{task_id}.gate"
        self.merge_wt = cfg.worktrees_dir / f"{task_id}.merge"
        self.run_dir = cfg.state_dir / "runs" / task_id
        self.ident = gitops.identity_env(cfg.identity_name, cfg.identity_email)
        self.created: set[Path] = set()
        self.calls = 0
        self.test_rejects = 0
        self.reject_reasons: list[str] = []
        self.pinned: list[str] = []
        self.gate_script = gates.PYTEST_GATE
        self.gate_env: dict[str, str] = {}
        # gate passes reused while their inputs stay the same (main unmoved: no second pass)
        self.gate_reuse = gates.GateReuse(exclude=LEDGER_FILES)

    # -------------------------------------------------------------- top level
    def go(self) -> Outcome:
        with tempfile.TemporaryDirectory(prefix="memini-orch-") as tmp:
            private = Path(tmp)
            # a private copy of the pytest gate (watched) and a throwaway HOME for every gate,
            # setup and hold-out run; uv keeps its real cache and Python installs
            self.gate_script = private / "pytest_gate.py"
            shutil.copyfile(gates.PYTEST_GATE, self.gate_script)
            (private / "home").mkdir()
            home = Path.home()
            self.gate_env = {
                "HOME": str(private / "home"),
                "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR") or str(home / ".cache" / "uv"),
                "UV_PYTHON_INSTALL_DIR": os.environ.get("UV_PYTHON_INSTALL_DIR")
                or str(home / ".local" / "share" / "uv" / "python"),
            }
            return self._go_guarded()

    def _skip_markers(self) -> tuple[str, ...]:
        """Return the pytest markers deselected in the sandbox (none for the local executor)."""
        if self.sandbox is None or self.cfg.sandbox_network:
            return ()  # a sandbox on the test-Neo4j network runs the integration tests too
        return self.cfg.sandbox_skip_markers

    def _sandbox_note(self) -> str:
        markers = self._skip_markers()
        if not markers:
            return ""
        return f" [sandbox: {', '.join(markers)} tests not run (per-slice Neo4j is a later step)]"

    def _base_env(self) -> dict[str, str]:
        """Return the env for git commands that may run hooks (allow-listed, private HOME)."""
        return allowed_env(self.cfg.env_allow, self.gate_env)

    def _go_guarded(self) -> Outcome:
        try:
            return self._go()
        except _Stop as stop:
            return Outcome(self.task_id, stop.status, stop.reason, self.wt)
        except gates.GateConfigError as exc:
            reason = f"gate configuration: {exc}"
            _tool_broke(self.cfg, self.task_id, reason)
            return Outcome(self.task_id, FAILED, reason, self.wt)

    def _go(self) -> Outcome:
        cfg = self.cfg
        base = gitops.rev(self.repo, cfg.main_branch)
        # a run that reaches here is not running (the caller holds its lock): what an earlier run
        # left is renamed, never deleted, and no person has to release the task first
        tag = "rerun" + datetime.now(UTC).strftime("%H%M%S%f")
        moved = gitops.set_aside(self.repo, cfg.worktrees_dir, cfg.branch_prefix, self.task_id, tag)
        if moved:
            state.ops_log(
                cfg, self.task_id, "set_aside", "leftovers of an earlier run renamed", moved=moved
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cfg.worktrees_dir.mkdir(parents=True, exist_ok=True)
        gitops.git(
            self.repo,
            "worktree",
            "add",
            "-q",
            "-b",
            self.branch,
            str(self.wt),
            base,
            base_env=self._base_env(),
        )
        self.created.add(self.wt)
        self._setup(self.wt)
        qa_sha, files = self._qa(base)
        if cfg.test_review:
            qa_sha, files = self._reveal_holdout(qa_sha, files)
        files = [*files, *self._pinned(qa_sha)]
        sealed = gates.seal(self.repo, qa_sha, files)
        snap = self._coder(base, qa_sha, files, sealed)
        return self._merge(snap, qa_sha, files, sealed)

    def _pinned(self, qa_sha: str) -> list[str]:
        """Return the existing tests the ticket lists as pinning its change (found at qa_sha)."""
        queue = self.cfg.stages[0].queue
        text = next(
            (t.text for t in reversed(enqueue.read_tickets(queue)) if t.id == self.task_id), ""
        )
        self.pinned = [
            p for p in enqueue.pinned_tests(text) if gitops.blob_sha256(self.repo, qa_sha, p)
        ]
        if self.pinned:
            state.ops_log(
                self.cfg,
                self.task_id,
                "pinned_tests",
                "existing tests the slice may update",
                files=self.pinned,
            )
        return self.pinned

    # -------------------------------------------------------------- helpers
    def _setup(self, wt: Path) -> None:
        for cmd in self.cfg.setup:
            res = self.host.run(cmd, cwd=wt, env=self.gate_env, timeout=self.cfg.gate_timeout_s)
            if res.code != 0:
                raise _Stop(FAILED, f"setup {' '.join(cmd)} failed: {res.output.strip()[-500:]}")

    def _fresh(self, path: Path, commit: str, *, setup: bool = True) -> None:
        """Give ``path`` a fresh detached checkout of ``commit`` with its own environment."""
        if path.exists():
            if path not in self.created:  # pragma: no cover - refused at the start of the run
                raise _Stop(FAILED, f"{path} was not created by this run")
            gitops.git(self.repo, "worktree", "remove", "--force", str(path))
        gitops.git(
            self.repo,
            "worktree",
            "add",
            "-q",
            "--detach",
            str(path),
            commit,
            base_env=self._base_env(),
        )
        self.created.add(path)
        if setup:
            self._setup(path)

    def _agent(
        self,
        role: str,
        prompt: str,
        workdir: Path,
        writable: Sequence[str],
        protected: Sequence[str],
    ) -> Result:
        cfg = self.cfg
        name = self.engines[{"review": "qa", "verify": "audit"}.get(role, role)]
        self.calls += 1
        settings_path = self.run_dir / f"{self.calls:02d}-{role}-settings.json"
        guard = self.run_dir / f"{self.calls:02d}-{role}-guard.json"
        refs = self._reference_repos()
        coder_ref = spec.resolve_reference(cfg, self.sp.reference)[0] if self.sp.reference else None
        settings = engine.settings(
            cfg,
            workdir,
            writable=writable,
            guard_config=guard,
            protected=protected,
            # QA and audit never see a reference repository; the coder may read its own
            deny_read=() if role == "coder" else refs,
            bash_roots=(coder_ref,) if role == "coder" and coder_ref else (),
        )
        settings_path.write_text(engine.json_text(settings), encoding="utf-8")
        tools = ""
        if role == "coder" and coder_ref:
            tools = cfg.reference_tools.replace("{repo}", str(coder_ref))
        argv = engine.command(
            cfg,
            name,
            "audit" if role in ("review", "verify") else role,
            workdir=workdir,
            settings=settings_path,
            prompt=prompt,
            reference_tools=tools,
        )
        stdin = prompt if cfg.engines[name].stdin else None
        res = self.host.run(
            argv,
            cwd=workdir,
            stdin=stdin,
            timeout=cfg.engine_timeout_s,
            pass_env=cfg.engines[name].env,
        )
        # the CLI answers in JSON: later steps read only the answer text; the cost is recorded
        text, cost = engine.agent_output(res.stdout)
        res = dataclasses.replace(res, stdout=text)
        log = self.run_dir / f"{self.calls:02d}-{role}.log"
        log.write_text(f"exit {res.code}\ncost_usd {cost}\n{res.output}", encoding="utf-8")
        with (self.run_dir / "costs.tsv").open("a", encoding="utf-8") as fh:
            fh.write(f"{self.calls:02d}\t{role}\t{name}\t{'' if cost is None else cost}\n")
        return res

    def _leftovers(self, writable: Sequence[str], protected: Sequence[str]) -> list[str]:
        """Return untracked or ignored files the agent left outside its writable paths."""
        listed = gitops.out(self.wt, "ls-files", "-z", "--others", "--exclude-standard")
        ignored = gitops.out(
            self.wt, "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--directory"
        )
        out = []
        for path in sorted({p for p in (listed + "\0" + ignored).split("\0") if p}):
            if _cache(path, self.cfg.ignored_ok):
                continue
            if any(gates.matches(p, path) for p in protected):
                out.append(f"protected path changed: {path} (untracked or ignored)")
            elif not any(gates.matches(w, path) for w in writable):
                out.append(f"untracked or ignored file outside the allowed paths: {path}")
        return out

    def _reference_repos(self) -> tuple[Path, ...]:
        refs = list(self.cfg.reference_repos.values())
        if self.sp.reference:
            refs.append(spec.resolve_reference(self.cfg, self.sp.reference)[0])
        return tuple(dict.fromkeys(refs))

    def _watch_sets(self) -> tuple[list[Path], list[Path]]:
        """Return (files restored after tampering, files only reported).

        Restored: files only this run writes while it holds its run slot (the
        orchestrator state, git config, hooks and info, the hold-out tests, the private
        pytest gate). Only reported, never byte-restored: the queue, the inbox, every ref
        (``for-each-ref``, refs/replace included) and packed-refs.
        """
        cfg = self.cfg
        common = gitops.common_dir(self.repo)
        roots = [common / "hooks", common / "info", gates.holdout_dir(cfg, self.task_id)]
        own = [common / "config", self.gate_script]
        own += [f for r in roots if r.is_dir() for f in sorted(r.rglob("*")) if f.is_file()]
        if cfg.parallel:
            # other runs update the state, inbox and queue, add branches and move main; only
            # this run's own files and refs are watched (coder code runs in docker, no .git)
            return own, [_REFS]
        own.insert(0, cfg.state_dir / "state.json")
        shared = [cfg.queue, cfg.inbox, common / "packed-refs", _REFS]
        return own, shared

    def _watched_files(self) -> dict[Path, bytes | None]:
        """Return the bytes of the files no step may change (see :meth:`_watch_sets`)."""
        own, shared = self._watch_sets()
        files = {
            f: (f.read_bytes() if f.is_file() else None) for f in (*own, *shared) if f != _REFS
        }
        refs = gitops.all_refs(self.repo)  # every ref, refs/replace included
        if self.cfg.parallel:  # only this run's branch and every replace ref
            mine = (f"refs/heads/{self.branch} ", "refs/replace/")
            refs = b"\n".join(r for r in refs.splitlines() if r.decode().startswith(mine))
        files[_REFS] = refs
        return files

    def watch(self, what: str, before: dict[Path, bytes | None]) -> None:
        """Compare with ``before``; on any change restore our own files, alert and reject."""
        after = self._watched_files()
        changed = sorted(p for p in {*before, *after} if before.get(p) != after.get(p))
        if not changed:
            return
        shared = set(self._watch_sets()[1])
        restored = []
        for path in changed:
            if path in shared:
                continue  # shared state: reported, never byte-restored
            old = before.get(path)
            if old is None:
                path.unlink(missing_ok=True)  # a file planted by the step
            else:
                path.write_bytes(old)
            restored.append(path)
        names = ", ".join(str(p) for p in changed if p != _REFS)
        if _REFS in changed:  # name the refs that appeared, moved or vanished
            old_refs = set((before.get(_REFS) or b"").decode().splitlines())
            new_refs = set((after.get(_REFS) or b"").decode().splitlines())
            moved = sorted({line.split(" ", 1)[0] for line in old_refs ^ new_refs})
            names = ", ".join(filter(None, [names, "refs: " + ", ".join(moved)]))
        state.inbox_append(
            self.cfg,
            self.task_id,
            "SECURITY: protected files changed",
            2,
            f"during {what}: {names}; restored: {', '.join(map(str, restored)) or 'none'}",
        )
        raise _Stop(REJECTED, f"{what} changed protected orchestrator or git files")

    def _holdout(self, commit: str) -> list[gates.Failure]:
        if self.cfg.test_review:  # the hold-out tests are slice tests now: QA reviews their edits
            return []
        before = self._watched_files()
        ok = gates.holdout_passes(
            self.cfg,
            self.repo,
            commit,
            self.task_id,
            self.run_dir / "holdout.log",
            gate_script=self.gate_script,
            env=self.gate_env,
            sandbox=self.sandbox,
            markers=self._skip_markers(),
        )
        self.watch("the hold-out run", before)
        return [] if ok else [gates.Failure(gates.HOLDOUT_FAILED)]

    # -------------------------------------------------------------- QA
    def _qa(self, base: str) -> tuple[str, list[str]]:
        cfg = self.cfg
        protected = [*cfg.protected, "**/conftest.py"]
        failures = "none (first attempt)"
        for attempt in (1, 2):
            prompt = engine.render(
                cfg,
                "qa",
                task_id=self.task_id,
                spec=self.sp.text(),
                cards=self._ticket_cards()[1],
                rules=engine.rules_text(cfg),
                failures=failures,
            )
            self._agent("qa", prompt, self.wt, cfg.qa_allow, protected)
            left = self._leftovers(cfg.qa_allow, protected)
            if left:
                raise _Stop(REJECTED, "QA: " + "; ".join(left))
            qa_sha = gitops.snapshot(
                self.wt, f"{self.task_id}: QA acceptance tests (sealed)", self.ident
            )
            files = self._qa_files(base, qa_sha)
            self._qa_text_check(qa_sha, files)
            if not gates.defines_tests(self.wt, files):  # before lint: the plainer reason
                raise _Stop(REJECTED, "the QA files define no test function")
            lint = self._qa_lint(files)
            if not lint:
                break
            if attempt == 2:  # noqa: PLR2004 — one retry
                raise _Stop(REJECTED, "the QA tests fail lint after a retry: " + lint[:500])
            failures = "Your test files fail lint; fix only that:\n" + lint
        self._fresh(self.gate_wt, qa_sha)
        if not gates.defines_tests(self.gate_wt, files):
            raise _Stop(REJECTED, "the QA files define no test function")
        run = self._gate_run(self.gate_wt, qa_sha, files)
        res = gates.run_tests(cfg, self.ex, run, files, gates.Profile("defaults", {}))
        if res.code == 0:
            raise _Stop(REJECTED, "the QA tests pass on the base without any implementation")
        return qa_sha, files

    def _reveal_holdout(self, qa_sha: str, files: list[str]) -> tuple[str, list[str]]:
        """Copy the hold-out tests into the slice as ordinary tests (owner 2026-11-17).

        The coder sees them in full and may edit one that contradicts the spec; they are
        sealed with the QA tests, so the originals stay in git and every change is reviewed.
        """
        source = gates.holdout_dir(self.cfg, self.task_id)
        revealed: list[str] = []
        for src in sorted(source.glob("test_*.py")):
            name = src.name
            if (self.wt / "tests" / "unit" / name).exists():
                name = "test_hold_" + name.removeprefix("test_")
            rel = f"tests/unit/{name}"
            shutil.copyfile(src, self.wt / rel)
            revealed.append(rel)
        sha = gitops.snapshot(
            self.wt, f"{self.task_id}: hold-out tests shown to the coder", self.ident
        )
        state.ops_log(
            self.cfg,
            self.task_id,
            "holdout_revealed",
            "the hold-out tests are part of the slice tests; the coder may edit them, QA reviews",
            files=revealed,
            commit=sha[:12],
        )
        return sha, [*files, *revealed]

    def _qa_text_check(self, qa_sha: str, files: Sequence[str]) -> None:
        """Run the ticket text checks on the visible and hidden tests before sealing (D11)."""
        deny = engine.test_deny(self.cfg)
        hidden = sorted(gates.holdout_dir(self.cfg, self.task_id).glob("test_*.py"))
        texts = {f: gitops.out(self.repo, "show", f"{qa_sha}:{f}") for f in files}
        texts |= {f"hold-out/{h.name}": h.read_text(encoding="utf-8") for h in hidden}
        found = [
            f"{name}: {p}" for name, text in texts.items() for p in gates.text_problems(text, deny)
        ]
        if found:
            raise _Stop(REJECTED, "QA tests refused before sealing: " + "; ".join(found[:5]))

    def _qa_files(self, base: str, qa_sha: str) -> list[str]:
        """Return the test files QA added; reject anything else it changed."""
        cfg = self.cfg
        files, problems = [], []
        for status, path in gitops.name_status(self.repo, base, qa_sha):
            name = path.rsplit("/", 1)[-1]
            under = any(gates.matches(a, path) for a in cfg.qa_allow)
            if (
                status != "A"
                or not under
                or not (name.startswith("test_") and name.endswith(".py"))
            ):
                problems.append(f"QA may only add test_*.py files under its paths: {status} {path}")
            else:
                files.append(path)
        if problems:
            raise _Stop(REJECTED, "; ".join(problems))
        if not files:
            raise _Stop(REJECTED, "QA wrote no test file")
        return files

    def _qa_lint(self, files: Sequence[str]) -> str:
        """Return the lint output of QA's files, or "" when they are clean (runs on the host)."""
        python = gates.python_for(self.cfg, self.wt)
        out = []
        for argv in gates.qa_lint_argvs(self.cfg, python, files):
            res = self.host.run(
                argv, cwd=self.wt, env=self.gate_env, timeout=self.cfg.gate_timeout_s
            )
            if res.code != 0:  # a silent failure is still a failure
                out.append(res.output.strip() or f"{' '.join(argv)}: exit {res.code}")
        return "\n".join(out)[-1500:]

    # -------------------------------------------------------------- coder
    def _autoformat(self) -> None:
        """Format the coder's own changed Python files before the gates (owner 2026-11-16).

        Formatting is mechanical, so it never fails a slice (LT-02 was rejected twice for it).
        Only files inside the coder's paths are touched; QA's sealed tests never are.
        """
        lines = gitops.out(self.wt, "status", "--porcelain", "--untracked-files=all").splitlines()
        changed = [ln[3:].split(" -> ")[-1] for ln in lines]
        files = [
            f for f in changed if f.endswith(".py") and any(gates.matches(p, f) for p in self.paths)
        ]
        if files:
            python = gates.python_for(self.cfg, self.wt)
            argv = [python, "-m", "ruff", "format", "--quiet", *files]
            self.host.run(argv, cwd=self.wt, env=self.gate_env, timeout=self.cfg.gate_timeout_s)

    def _coder(self, base: str, qa_sha: str, files: list[str], sealed: dict[str, str]) -> str:
        """Run the coder attempts; return the commit the audit passed (the one to merge)."""
        cfg = self.cfg
        protected = [*cfg.protected, *cfg.guard_protected]
        reference = self._reference_text()
        failures: list[str] = []
        snap = qa_sha
        edits = cfg.test_review
        writable = (*self.paths, *files, *cfg.old_test_paths) if edits else self.paths
        limit, attempt = cfg.max_coder_attempts, 0
        while attempt < limit:
            attempt += 1
            prompt = engine.render(
                cfg,
                "coder",
                task_id=self.task_id,
                spec=self.sp.text(with_reference=True),
                cards=self._ticket_cards()[1],
                allowed_paths=", ".join(writable),
                allowed_args=" ".join(self.paths),
                test_files=" ".join(files),
                test_rule=_test_rule(edits, files),
                dispute_rule=_dispute_rule(edits),
                reference=reference,
                failures="\n".join(f"- {f}" for f in failures) or "none (first attempt)",
                rules=engine.rules_text(cfg),
            )
            tool_protected = engine.carve(protected, writable, self.wt) if edits else protected
            res = self._agent("coder", prompt, self.wt, writable, tool_protected)  # watched
            for item in engine.follow_ups(res.stdout):
                state.inbox_append(cfg, self.task_id, "follow-up proposal (not queued)", 1, item)
            # leftovers, snapshot and anti-cheat checks come before a dispute is accepted
            left = self._leftovers(writable, self._path_protected())
            if left:
                raise _Stop(REJECTED, "; ".join(left))
            self._autoformat()
            snap = gitops.snapshot(self.wt, f"{self.task_id}: coder attempt {attempt}", self.ident)
            self._take_old_tests(qa_sha, snap, sealed, files)
            cheats = self._cheats(qa_sha, snap, sealed, extra=files if edits else ())
            if cheats:
                raise _Stop(REJECTED, "; ".join(cheats))
            reason = engine.dispute(res.stdout)
            if reason:
                moves.post(
                    cfg,
                    self.task_id,
                    sender=cfg.worker or "squad",
                    to="planner",
                    kind="dispute",
                    text=f"the coder disputes the ticket: {reason}",
                    owner_words=cfg.standing_reissue_words,
                )
                raise _Stop(DISPUTED, f"the coder disputes the tests: {reason}")
            if snap == qa_sha:
                failures = ["no change was made"]
                continue
            changed = gates.changed_tests(self.repo, snap, sealed) if edits else []
            hits = self._scan_changes(attempt, qa_sha, snap, changed)
            if hits:
                failures = hits[:20]
                continue
            gate_fails = self._gate_slice(snap, base, files)
            failures = [f.summary() for f in gate_fails]
            if not failures and edits:
                failure = self._review_failure(attempt, qa_sha, snap, changed)
                if failure:
                    limit = max(limit, attempt + 1)  # one reject always gets one more attempt
                    failures = [failure]
                    continue
            if not failures:
                failures = self._final_checks(attempt, qa_sha, snap, files)
                if not failures:
                    return snap
        if any(f.startswith("AST check:") for f in failures):
            state.inbox_append(
                cfg,
                self.task_id,
                "AST check rejected the slice (owner may override)",
                2,
                "; ".join(f for f in failures if f.startswith("AST check:"))[:1500]
                + f" | to accept exactly this code, add the line: OVERRIDE-AST {self.task_id} "
                f"{snap}",
            )
        raise _Stop(
            REJECTED,
            f"not green after {cfg.max_coder_attempts} attempts: " + "; ".join(failures)[:1500],
        )

    def _take_old_tests(
        self, qa_sha: str, snap: str, sealed: dict[str, str], files: list[str]
    ) -> None:
        """Add the existing tests the coder edited to the slice's tests; refuse a deleted one.

        An old test the change breaks is the coder's to fix (owner 2026-11-18). It is sealed at
        its original content so the edit shows as a test change: QA reviews it, then vp-rnd.
        """
        if not self.cfg.test_review:
            return
        gone = []
        for status, path in gitops.name_status(self.repo, qa_sha, snap):
            if path in sealed or not any(gates.matches(o, path) for o in self.cfg.old_test_paths):
                continue
            if status == "D":
                gone.append(path)
            elif status == "M":
                sealed[path] = gates.seal(self.repo, qa_sha, [path])[path]
                files.append(path)
                self.pinned.append(path)
        if gone:
            raise _Stop(REJECTED, "an existing test was deleted: " + ", ".join(gone))

    def _ast_problems(self, qa_sha: str, snap: str, skip: Sequence[str] = ()) -> list[str]:
        """Return the AST check's problems for the Python files the coder changed.

        Only the owner can let a refused commit through, with an inbox line naming it.
        """
        inbox = self.cfg.inbox.read_text(encoding="utf-8") if self.cfg.inbox.exists() else ""
        if astcheck.override_granted(inbox, self.task_id, snap):
            return []
        rule_set = astcheck.rules(self.cfg, self.wt)
        sources = [
            (path, gitops.out(self.repo, "show", f"{snap}:{path}"))
            for status, path in gitops.name_status(self.repo, qa_sha, snap)
            if status != "D" and path.endswith(".py") and path not in skip
        ]
        return [f"AST check: {p}" for p in astcheck.check_files(sources, rule_set)]

    def _cheats(
        self, base: str, head: str, sealed: dict[str, str], extra: Sequence[str]
    ) -> list[str]:
        """Return the anti-cheat problems: ancestry, path gate, file modes, sealed hashes."""
        if not gitops.is_ancestor(self.repo, base, head):
            return ["the sealed QA commit is no longer in the slice's history"]
        changed = gitops.changed(self.repo, base, head)
        out = gates.path_problems(changed, [*self.paths, *extra], self._path_protected())
        out += [
            f"path outside the configured allow list changed: {p}"
            for p in changed
            if p not in extra and not any(gates.matches(a, p) for a in self.cfg.allow)
        ]
        out += [f"not a regular file: {p}" for p in gitops.bad_modes(self.repo, base, head)]
        out += gates.sealed_problems(self.repo, head, sealed, allow_edits=self.cfg.test_review)
        return out

    def _path_protected(self) -> list[str]:
        """Return what the path gate protects: ``protected`` plus every ``**/`` guard entry."""
        names = [e for e in self.cfg.guard_protected if e.startswith("**/")]
        return [*self.cfg.protected, *names]

    def _gate_run(self, worktree: Path, commit: str, files: list[str]) -> gates.GateRun:
        tracked = gitops.out(self.repo, "ls-tree", "-r", "--name-only", commit)
        tracked_tests = {
            p
            for p in tracked.splitlines()
            if p.endswith(".py") and p.rsplit("/", 1)[-1].startswith("test_")
        }
        return gates.GateRun(
            worktree=worktree,
            log=self.run_dir / "gates.log",
            test_ids=frozenset(gates.test_node_ids(worktree, sorted({*files, *tracked_tests}))),
            env=self.gate_env,
            gate_script=self.gate_script,
            skip_markers=self._skip_markers(),
        )

    def _gate_slice(self, snap: str, base: str, files: list[str]) -> list[gates.Failure]:
        cfg = self.cfg
        self._fresh(self.gate_wt, snap)
        run = self._gate_run(self.gate_wt, snap, files)
        fails = gates.qa_test_failures(cfg, self.ex, run, files)
        fails += gates.run_gates(
            cfg, self.ex, run, "slice", base, host=self.host, reuse=self.gate_reuse
        )
        return fails + self._holdout(snap)

    def _reference_text(self) -> str:
        """Return the coder prompt's reference section ("none" without a reference repo)."""
        if not self.sp.reference:
            return "none"
        ref_repo, branch = spec.resolve_reference(self.cfg, self.sp.reference)
        return engine.render(self.cfg, "coder_reference", repo=str(ref_repo), branch=branch)

    def _scan_changes(self, attempt: int, qa_sha: str, snap: str, changed: list[str]) -> list[str]:
        """Return the deny-scan and AST problems of the coder's change (tests scanned apart)."""
        cfg = self.cfg
        diff = gitops.out(self.repo, "diff", "-U0", "--no-color", qa_sha, snap, *_exclude(changed))
        hits = [f"deny scan: {h}" for h in gates.deny_hits(diff, cfg.deny_patterns)]
        # D12 checks the diff: literals the change adds, and graph modules it deletes
        status = gitops.name_status(self.repo, qa_sha, snap)
        hits += gates.graph_literal_problems(self.repo, qa_sha, snap, status, cfg.graph_name_paths)
        if changed:  # edited tests: the deny scan too, minus plain use of pytest itself
            tdiff = gitops.out(self.repo, "diff", "-U0", "--no-color", qa_sha, snap, "--", *changed)
            relaxed = [p for p in cfg.deny_patterns if p != r"\bpytest\b"]
            hits += [f"deny scan (tests): {h}" for h in gates.deny_hits(tdiff, relaxed)]
            self._store_test_diff(attempt, qa_sha, snap, changed)
        return hits + self._ast_problems(qa_sha, snap, skip=changed)

    def _store_test_diff(self, attempt: int, qa_sha: str, snap: str, changed: list[str]) -> None:
        """Keep the diff of the coder's test changes next to the run (the originals stay in git)."""
        diff = gitops.out(self.repo, "diff", "--no-color", qa_sha, snap, "--", *changed)
        path = self.run_dir / f"test-changes-{attempt}.diff"
        path.write_text(diff, encoding="utf-8")
        state.ops_log(
            self.cfg,
            self.task_id,
            "test_changed",
            "the coder edited sealed tests; the original is kept, the diff is stored",
            attempt=attempt,
            files=changed,
            diff=str(path),
        )

    def _review_failure(
        self, attempt: int, qa_sha: str, snap: str, changed: list[str]
    ) -> str | None:
        """Return the coder's next failure after a reject; a second reject stops the slice."""
        reason = self._ask_qa(attempt, qa_sha, snap, changed)
        if reason is None:
            reason = self._ask_vprnd(attempt, qa_sha, snap, changed)
        if reason is None:
            return None
        self.test_rejects += 1
        self.reject_reasons.append(reason)
        if self.test_rejects >= _REVIEW_REJECTS:
            self._escalate(qa_sha, snap, changed, reason)
        return f"the QA review rejected your test change: {reason}"

    def _ask_vprnd(self, attempt: int, qa_sha: str, snap: str, changed: list[str]) -> str | None:
        """After QA approves, move the slice to vp-rnd when it updated tests it does not own.

        A light update is approved or rejected by vp-rnd; a heavy one (a deleted test, a weakened
        assertion, a gate file, many tests) stops the slice and goes to the owner inbox.
        """
        stale = [c for c in changed if c in self.pinned]
        if not stale:
            return None
        move = moves.post(
            self.cfg,
            self.task_id,
            sender=self.cfg.worker or "squad",
            to="vp-rnd",
            kind="stale_test_review",
            text=(
                f"{len(stale)} existing test file(s) updated by the slice, attempt {attempt}: "
                f"{', '.join(stale)}. Answer approve, or reject: <reason and the fix>."
            ),
            qa_sha=qa_sha,
            snap=snap,
            changed=stale,
        )
        answer = moves.wait(self.cfg, move, _MOVE_WAIT_S)
        if answer is None:
            raise _Stop(FAILED, f"vp-rnd did not answer the test update review ({move}) in time")
        approved, reason = moves.parse_verdict(answer["result"])
        if not approved and reason.startswith("heavy:"):
            raise _Stop(DISPUTED, f"vp-rnd sent the test update to the owner: {reason}")
        return None if approved else reason

    def _ask_qa(self, attempt: int, qa_sha: str, snap: str, changed: list[str]) -> str | None:
        """Move the slice to QA's queue with its test changes; wait for the answer.

        Only this slice waits (its worker); the squad's other slices keep running. Returns the
        reject reason (with QA's proposed fix), None when QA approved or nothing was changed.
        """
        if not changed:
            return self._review_step(attempt, qa_sha, snap, changed)
        move = moves.post(
            self.cfg,
            self.task_id,
            sender=self.cfg.worker or "squad",
            to="qa",
            kind="test_review",
            text=f"{len(changed)} test file(s) changed by the coder, attempt {attempt}",
            spec=str(self.sp.path),
            paths=list(self.paths),
            attempt=attempt,
            qa_sha=qa_sha,
            snap=snap,
            changed=changed,
        )
        answer = moves.wait(self.cfg, move, _MOVE_WAIT_S)
        if answer is None:
            raise _Stop(FAILED, f"QA did not answer the test review ({move}) in time")
        verdict = json.loads(answer["result"])
        return None if verdict["approved"] else str(verdict["reason"])

    def _review_step(self, attempt: int, qa_sha: str, snap: str, changed: list[str]) -> str | None:
        """Run the QA review of the test changes; return the reject reason, None when fine."""
        cfg = self.cfg
        if not changed:
            state.ops_log(
                cfg, self.task_id, "test_review_skipped", "no test was changed", attempt=attempt
            )
            return None
        tests = gitops.out(self.repo, "diff", "--no-color", qa_sha, snap, "--", *changed)
        code = gitops.out(self.repo, "diff", "--no-color", qa_sha, snap, *_exclude(changed))
        if len(tests) + len(code) > _DIFF_LIMIT:  # never review half a change
            review = audit.Review(False, "the change is too large for one review")
        else:
            prompt = engine.render(
                cfg,
                "test_review",
                task_id=self.task_id,
                spec=self.sp.text(),
                test_diff=tests,
                code_diff=code,
                rules=engine.rules_text(cfg),
            )
            with tempfile.TemporaryDirectory(prefix="memini-review-") as tmp:
                res = self._agent("review", prompt, Path(tmp), (), ())
            review = audit.parse_review(res.stdout)
        state.ops_log(
            cfg,
            self.task_id,
            "test_review_approve" if review.approved else "test_review_reject",
            review.reason,
            attempt=attempt,
            files=changed,
        )
        return None if review.approved else review.reason

    def _escalate(self, qa_sha: str, snap: str, changed: list[str], reason: str) -> None:
        """Stop the slice for the owner after the second reject; its worktree is kept."""
        diffs = " | ".join(
            f"test diff {p.name}: {p.read_text(encoding='utf-8')}"
            for p in sorted(self.run_dir.glob("test-changes-*.diff"))
        )
        fixes = " | ".join(
            f"QA's proposed fix {i}: {r}" for i, r in enumerate(self.reject_reasons, 1)
        )
        text = (
            f"test review rejected twice | {fixes} | {diffs} | the sealed originals are in "
            f"{qa_sha[:12]}; the coder's version in {snap[:12]}; run folder {self.run_dir}"
        )
        state.inbox_append(
            self.cfg, self.task_id, "test review rejected twice", 1, text, limit=40_000
        )
        agent_inbox.notify(self.cfg, "qa", f"{self.task_id}: {text}")
        state.ops_log(self.cfg, self.task_id, "test_review_escalated", text, files=changed)
        raise _Stop(DISPUTED, f"the QA review rejected the test change twice: {reason}")

    def _final_checks(self, attempt: int, qa_sha: str, snap: str, files: list[str]) -> list[str]:
        """Run the audit, then the Verifier; return the coder's failures (none: merge)."""
        verdict = self._audit(qa_sha, snap, files)
        if not verdict.passed:
            return [f"audit: {f[:300]}" for f in verdict.findings[:10]]
        return self._verify(attempt, qa_sha, snap, files)

    def _ticket_cards(self) -> tuple[list[str], str]:
        """Return the card IDs of this task's ticket and their text with the old-code evidence."""
        cfg = self.cfg
        try:
            tickets = enqueue.read_tickets(cfg.stages[0].queue) if cfg.stages else []
        except (ValueError, OSError):
            tickets = []
        ticket = next((t for t in tickets if t.id == self.task_id), None)
        if ticket is None:
            return [], ""
        return enqueue.card_text(cfg, ticket.text)

    def _verify(self, attempt: int, qa_sha: str, snap: str, files: list[str]) -> list[str]:
        """Run the Verifier (D8); return the coder's failures. A missing test stops for QA."""
        ids, card_text = self._ticket_cards()
        if not ids:
            state.ops_log(self.cfg, self.task_id, "verifier_skipped", "the task has no card lines")
            return []
        diff = gitops.out(self.repo, "diff", qa_sha, snap)
        tests = "\n\n".join(f"# {f}\n{gitops.out(self.repo, 'show', f'{snap}:{f}')}" for f in files)
        if len(diff) + len(tests) > _DIFF_LIMIT:  # never verify half a change
            return ["verifier: the change is too large for one verification"]
        prompt = engine.render(
            self.cfg, "verifier", task_id=self.task_id, cards=card_text, diff=diff, tests=tests
        )
        with tempfile.TemporaryDirectory(prefix="memini-verify-") as tmp:
            res = self._agent("verify", prompt, Path(tmp), (), ())
        found = audit.parse_verifier(res.stdout, ids)
        extras = "\n".join(f"- {x}" for x in found.extras) or "none"
        (self.run_dir / f"verifier-{attempt}.md").write_text(
            f"{found.table}\nExtras nobody asked for:\n{extras}\n", encoding="utf-8"
        )
        state.ops_log(
            self.cfg,
            self.task_id,
            "verifier",
            "ok" if found.ok else "blocked",
            attempt=attempt,
            cards=ids,
            extras=len(found.extras),
        )
        if found.code_problems:
            return [f"verifier: {p}" for p in found.code_problems[:10]]
        if found.test_problems:
            text = "; ".join(found.test_problems)[:600]
            agent_inbox.notify(self.cfg, "qa", f"{self.task_id}: verifier: {text}")
            raise _Stop(DISPUTED, f"verifier: a card line has no proving test: {text}")
        return []

    def _audit(self, qa_sha: str, snap: str, files: list[str]) -> audit.Verdict:
        diff = gitops.out(self.repo, "diff", qa_sha, snap)
        if len(diff) > _DIFF_LIMIT:  # never audit half a change
            return audit.Verdict(False, ("diff too large for one audit",))
        prompt = engine.render(
            self.cfg,
            "audit",
            task_id=self.task_id,
            spec=self.sp.text(),
            cards=self._ticket_cards()[1],
            test_files=" ".join(files),
            diff=diff,
            rules=engine.rules_text(self.cfg),
        )
        with tempfile.TemporaryDirectory(prefix="memini-audit-") as tmp:
            res = self._agent("audit", prompt, Path(tmp), (), ())
        return audit.parse(res.stdout)

    # -------------------------------------------------------------- merge
    def _merge(self, snap: str, qa_sha: str, files: list[str], sealed: dict[str, str]) -> Outcome:
        cfg = self.cfg
        asked = time.monotonic()
        with state.locked(cfg.state_dir / "merge.lock"):
            gates.log_line(
                self.run_dir / "gates.log", f"merge lock waited {time.monotonic() - asked:.1f}s"
            )
            for _ in range(cfg.merge_retries):
                if gitops.rev(self.repo, self.branch) != snap:
                    raise _Stop(REJECTED, "the task branch moved after the audit")
                main_sha = gitops.rev(self.repo, cfg.main_branch)
                tip = self._rebased(main_sha, snap)
                cheats = self._cheats(main_sha, tip, sealed, extra=files)
                if cheats:
                    raise _Stop(REJECTED, "after rebase: " + "; ".join(cheats))
                squash = self._squash(main_sha, tip, qa_sha, files)
                cheats = self._cheats(main_sha, squash, sealed, extra=(*files, *LEDGER_FILES))
                if cheats:  # pragma: no cover - the squash holds the checked tree + ledger rows
                    raise _Stop(REJECTED, "squash commit: " + "; ".join(cheats))
                run = self._gate_run(self.merge_wt, squash, files)
                fails = gates.qa_test_failures(cfg, self.ex, run, files)
                fails += gates.run_gates(
                    cfg, self.ex, run, "merge", main_sha, host=self.host, reuse=self.gate_reuse
                )
                fails += self._holdout(squash)
                if fails:
                    reason = "; ".join(f.summary() for f in fails)
                    raise _Stop(FAILED, "merge gates failed: " + reason[:1500])
                if self._advance(squash, main_sha):
                    kept = self._cleanup(squash)
                    note = self._sandbox_note() + self._batch_suite(squash, main_sha, files)
                    return Outcome(
                        self.task_id, MERGED, f"merged as {squash[:10]}{kept}{note}", None, squash
                    )
            raise _Stop(FAILED, f"main moved {cfg.merge_retries} times during the merge")

    def _batch_suite(self, squash: str, main_sha: str, files: list[str]) -> str:
        """Run the unit suite once the last slice of the queue has merged (D-9).

        A failure is not the last coder's to fix: the failing tests go to the planner, who writes
        a fix ticket (owner 2026-11-17). Returns a note for the outcome.
        """
        if not self._last_of_batch():
            return ""
        self._fresh(self.gate_wt, squash)
        run = self._gate_run(self.gate_wt, squash, files)
        fails = gates.run_gates(self.cfg, self.ex, run, "batch", main_sha, host=self.host)
        if not fails:
            state.ops_log(self.cfg, self.task_id, "batch_suite_passed", "the unit suite is green")
            return "; batch suite green"
        text = "; ".join(f.summary() for f in fails)[:1500]
        moves.post(
            self.cfg,
            self.task_id,
            sender=self.cfg.worker or "squad",
            to="planner",
            kind="batch_failure",
            text=f"the unit suite failed after the last slice merged ({squash[:10]}): {text}",
            module=self.sp.module,
        )
        state.ops_log(self.cfg, self.task_id, "batch_suite_failed", text)
        return "; batch suite FAILED (fix ticket requested from the planner)"

    def _last_of_batch(self) -> bool:
        """Return True when every other task of this queue is merged (D-9: the unit suite runs)."""
        st = state.State(self.cfg)
        queue = state.read_queue(self.cfg.queue) if self.cfg.queue.exists() else []
        others = [e.task_id for e in queue if e.task_id != self.task_id]
        return all(st.is_merged(t) for t in others) and not stages.upstream_pending(self.cfg)

    def _rebased(self, main_sha: str, snap: str) -> str:
        """Rebase the slice onto ``main_sha`` in the integration checkout; return the new tip.

        The coder's worktree and branch are never touched: the integration checkout starts
        detached at the slice tip and is rebased there.
        """
        self._fresh(self.merge_wt, snap, setup=False)
        if not gitops.is_ancestor(self.repo, main_sha, snap):
            res = gitops.git(
                self.merge_wt,
                "rebase",
                "-q",
                main_sha,
                env=self.ident,
                check=False,
                base_env=self._base_env(),
            )
            if res.returncode != 0:
                gitops.git(self.merge_wt, "rebase", "--abort", check=False)
                raise _Stop(FAILED, "the slice does not rebase cleanly on the latest main")
        self._setup(self.merge_wt)
        return gitops.rev(self.merge_wt, "HEAD")

    def _squash(self, main_sha: str, tip: str, qa_sha: str, files: list[str]) -> str:
        cfg = self.cfg
        # one commit on top of main holding the rebased tree: move HEAD, keep index and tree
        gitops.git(self.merge_wt, "reset", "-q", "--soft", main_sha)
        implemented = [p for p in gitops.changed(self.repo, main_sha, tip) if p not in files]
        cid = self._ledger(qa_sha, files)
        coder_author = cfg.engines[self.engines["coder"]].author
        qa_author = cfg.engines[self.engines["qa"]].author
        message = "\n".join(
            [
                f"{self.task_id}: {self.sp.title}",
                "",
                f"Requirement: slice spec {self.task_id} ({self.sp.path.name}); acceptance tests "
                f"sealed at {qa_sha[:10]}",
                f"Implemented: {', '.join(implemented)}; tests {', '.join(files)}; ledger {cid}",
                "Why: the one-shot QA tests failed on the base and pass now in every env profile; "
                "the hold-out, the CI gates and the one-shot diff audit passed "
                f"({self._engines_text()})",
                "",
                f"Change-Author: {coder_author}",
                f"Tests-Author: {qa_author} QA agent (one-shot, orchestrator)",
                "",
            ]
        )
        msg_file = self.run_dir / "COMMIT_MSG"
        msg_file.write_text(message, encoding="utf-8")
        before = self._watched_files()
        res = gitops.git(
            self.merge_wt,
            "commit",
            "-q",
            "-F",
            str(msg_file),
            env=self.ident,
            check=False,
            base_env=self._base_env(),  # the hooks run repository code
        )
        self.watch("the squash commit hooks", before)
        if res.returncode != 0:
            raise _Stop(FAILED, "the squash commit was refused by the hooks (see the run log)")
        return gitops.rev(self.merge_wt, "HEAD")

    def _engines_text(self) -> str:
        return "engines " + ", ".join(f"{r}={self.engines[r]}" for r in engine.ROLES)

    def _ledger(self, qa_sha: str, files: list[str]) -> str:
        """Write the change row (and the feature row with its tests) in the repo's format."""
        cfg = self.cfg
        ldir = self.merge_wt / "ledgers"
        if not (ldir / "schema.sql").is_file():
            raise _Stop(FAILED, "the repository has no ledgers/schema.sql")
        unit_root = next((r for r in cfg.module_roots if self.paths[0].startswith(r)), None)
        layer = cfg.module_roots.get(unit_root or "")
        fid = ledger.next_id(ledger.load(ldir), "feature", self.sp.module) if layer else None
        coder_author = cfg.engines[self.engines["coder"]].author
        cid = ledger.add_change(
            ldir,
            author=coder_author,
            module=self.sp.module,
            summary=f"{self.task_id}: {self.sp.title} (orchestrator slice; {self._engines_text()})",
            features=fid,
            checks="orchestrator: QA tests per env profile, hold-out, CI gates, diff audit",
        )
        if fid and layer:
            node_ids = gates.test_node_ids(self.merge_wt, files)
            rows = [
                ledger.insert(
                    "feature",
                    id=fid,
                    description=self.sp.title,
                    layer=layer,
                    proof=f"slice spec {self.task_id}; acceptance tests sealed at {qa_sha[:10]}",
                    status="active",
                    owner_module=self.sp.module,
                    added_in=cid,
                )
            ] + [ledger.insert("feature_test", feature_id=fid, test_id=t) for t in node_ids]
            with (ldir / "features.sql").open("a", encoding="utf-8") as fh:
                fh.write("".join(r + "\n" for r in rows))
        gitops.git(self.merge_wt, "add", "ledgers")
        return cid

    def _advance(self, squash: str, main_sha: str) -> bool:
        """Fast-forward main to ``squash``; False when main moved meanwhile."""
        main = self.cfg.main_branch
        holder = gitops.worktree_of(self.repo, main)
        if holder is not None:
            # main is someone's checkout (the owner's seat): fast-forward it there so its
            # files follow, and only when it has no tracked change; never touch it otherwise
            dirty = gitops.out(holder, "status", "--porcelain", "--untracked-files=no")
            if dirty:
                state.inbox_append(
                    self.cfg,
                    self.task_id,
                    "merge blocked: main's worktree has changes",
                    1,
                    f"{holder} has uncommitted changes; the squash commit {squash[:10]} waits "
                    f"in {self.merge_wt} (git -C {holder} merge --ff-only {squash[:10]})",
                )
                raise _Stop(FAILED, f"main is checked out in {holder}, which is not clean")
            res = gitops.git(holder, "merge", "-q", "--ff-only", squash, check=False)
        else:
            res = gitops.git(
                self.repo, "update-ref", f"refs/heads/{main}", squash, main_sha, check=False
            )
        if res.returncode == 0:
            return True
        if gitops.rev(self.repo, main) != main_sha:
            return False
        state.inbox_append(
            self.cfg,
            self.task_id,
            "merge blocked: main could not be fast-forwarded",
            1,
            f"{holder or self.repo}: {gitops.tail(res, 300)}; the squash commit {squash[:10]} "
            f"waits in {self.merge_wt}",
        )
        raise _Stop(FAILED, "main could not be fast-forwarded (see the inbox)")

    def _cleanup(self, squash: str) -> str:
        """Remove this run's worktrees and the merged task branch; return a note if one stays."""
        note = ""
        res = gitops.git(self.repo, "worktree", "remove", str(self.wt), check=False)
        if res.returncode != 0:
            note = f" (task worktree kept at {self.wt}: {gitops.tail(res)})"
        else:
            # the squash commit holds exactly the branch's tree: point the branch at it so
            # git branch -d sees it merged
            old = gitops.rev(self.repo, self.branch)
            gitops.git(self.repo, "update-ref", f"refs/heads/{self.branch}", squash, old)
            gitops.git(self.merge_wt, "branch", "-q", "-d", self.branch)
        for path in (self.gate_wt, self.merge_wt):
            if path in self.created and path.exists():
                gitops.git(self.repo, "worktree", "remove", "--force", str(path))
        return note


def _cache(path: str, ignored_ok: Sequence[str]) -> bool:
    parts = path.rstrip("/").split("/")
    for entry in ignored_ok:
        if entry.endswith("/") and entry.rstrip("/") in parts:
            return True
        if not entry.endswith("/") and parts[-1] == entry:
            return True
    return False


class _Watched:
    """An executor wrapper: after EVERY run the protected files must be unchanged."""

    def __init__(self, inner: Executor, run: _Run) -> None:
        self.inner, self.task_run = inner, run

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdin: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int = 1800,
        pass_env: Sequence[str] = (),
    ) -> Result:
        """Run through the inner executor, then check the protected files."""
        before = self.task_run._watched_files()
        res = self.inner.run(
            argv, cwd=cwd, stdin=stdin, env=env, timeout=timeout, pass_env=pass_env
        )
        self.task_run.watch(f"running {Path(argv[0]).name}", before)
        return res


def review_move(cfg: Config, move: moves.Move) -> str:
    """QA's side of a ``test_review`` move (served by the qa role): return the verdict as JSON."""
    p = move["payload"]
    run = _Run(cfg, move["task"], spec.load(Path(p["spec"])), tuple(p["paths"]), {})
    run.engines = engine.pick_engines(cfg, None)
    run.run_dir = cfg.state_dir / "runs" / move["task"] / f"qa-review-{move['id']}"
    run.run_dir.mkdir(parents=True, exist_ok=True)
    reason = run._review_step(p["attempt"], p["qa_sha"], p["snap"], p["changed"])
    return json.dumps({"approved": reason is None, "reason": reason or ""})
