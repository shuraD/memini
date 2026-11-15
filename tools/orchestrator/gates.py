"""Deterministic gates: paths, sealed tests, env profiles, CI steps, hold-out.

The anti-cheat gate is path based and engine independent: every path the coder changed
must be under the task's allowed paths and none may be protected. The sealed hash of the
QA commit's tests is a second, independent check. Gate commands are the same commands CI
runs: a gate names a ``.github/workflows/ci.yml`` step and runs that step's ``run:`` text.
"""

from __future__ import annotations

import ast
import re
import shutil
import sys
import tarfile
import tempfile
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from tools.ci.pytest_gate import junit_key
from tools.orchestrator import gitops
from tools.orchestrator.config import Config, Gate
from tools.orchestrator.executor import Executor, Result, run_on_host

HOLDOUT_FAILED = "hold-out failed"
_BASH = shutil.which("bash") or "bash"


class GateConfigError(ValueError):
    """A gate cannot be built from the configuration or the workflow."""


# ------------------------------------------------------------------ paths
def matches(entry: str, path: str) -> bool:
    """Return True when the path-list ``entry`` covers ``path``.

    ``dir/`` covers everything under it, ``**/name`` a file of that name anywhere, and any
    other entry exactly that path.
    """
    if entry.startswith("**/"):
        return path.rsplit("/", 1)[-1] == entry[3:]
    if entry.endswith("/"):
        return path.startswith(entry)
    return path == entry


def path_problems(
    changed: Iterable[str], allowed: Sequence[str], protected: Sequence[str]
) -> list[str]:
    """Return a problem per changed path outside ``allowed`` or matching ``protected``."""
    out = []
    for path in sorted(set(changed)):
        if any(matches(p, path) for p in protected):
            out.append(f"protected path changed: {path}")
        elif not any(matches(a, path) for a in allowed):
            out.append(f"path outside the allow list changed: {path}")
    return out


def normalize_paths(paths: Iterable[str]) -> tuple[str, ...]:
    """Return repo-relative declared paths; refuse absolute paths and ``..``."""
    out = []
    for raw in paths:
        p = raw.strip().replace("\\", "/")
        if not p or p.startswith("/") or ".." in p.split("/"):
            msg = f"declared path {raw!r} must be relative inside the repository"
            raise GateConfigError(msg)
        out.append(p.removeprefix("./"))
    return tuple(out)


def sealed_problems(repo: Path, rev: str, sealed: Mapping[str, str]) -> list[str]:
    """Return a problem per sealed test whose content at ``rev`` differs from its hash."""
    return [
        f"sealed test changed or missing: {path}"
        for path, digest in sorted(sealed.items())
        if gitops.blob_sha256(repo, rev, path) != digest
    ]


def seal(repo: Path, rev: str, paths: Iterable[str]) -> dict[str, str]:
    """Return ``{path: sha256}`` of the test files at ``rev``."""
    out = {}
    for path in paths:
        digest = gitops.blob_sha256(repo, rev, path)
        if digest is None:  # pragma: no cover - paths come from that commit's own diff
            msg = f"cannot seal {path}: missing at {rev}"
            raise GateConfigError(msg)
        out[path] = digest
    return out


# ------------------------------------------------------------------ env profiles
@dataclass(frozen=True)
class Profile:
    """A resolved env profile."""

    name: str
    env: Mapping[str, str]


def flag_names(flags_file: Path, function: str) -> list[str]:
    """Return the env var names passed as string literals to ``function(...)`` in the file.

    The file is parsed, never imported. A missing file has no flags.
    """
    if not flags_file.is_file():
        return []
    tree = ast.parse(flags_file.read_text(encoding="utf-8"), filename=str(flags_file))
    names: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == function
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value not in names
        ):
            names.append(node.args[0].value)
    return sorted(names)


def profiles(cfg: Config, worktree: Path) -> list[Profile]:
    """Return the env profiles to run.

    A profile with ``all_flags`` needs the flags file: when it is missing the task fails
    (raise), it is never silently dropped.
    """
    out = []
    for spec in cfg.profiles:
        env = dict(spec.env)
        if spec.all_flags:
            flags_file = worktree / cfg.flags_file
            if not cfg.flags_file or not flags_file.is_file():
                msg = f"profile {spec.name} needs the flags file {cfg.flags_file or '(unset)'}"
                raise GateConfigError(msg)
            env.update(
                dict.fromkeys(flag_names(flags_file, cfg.flags_function), cfg.flags_on_value)
            )
        out.append(Profile(spec.name, env))
    return out or [Profile("defaults", {})]


# ------------------------------------------------------------------ CI steps
def ci_step_script(workflow: Path, step: str) -> str:
    """Return the ``run:`` text of the first ci.yml step whose name starts with ``step``.

    A step whose script uses a ``${{ }}`` expression is refused: only CI can evaluate it.
    """
    doc = yaml.safe_load(workflow.read_text(encoding="utf-8")) or {}
    for job in (doc.get("jobs") or {}).values():
        for s in job.get("steps") or []:
            if str(s.get("name", "")).startswith(step) and "run" in s:
                script = str(s["run"])
                if "${{" in script:
                    msg = f"ci step {step!r} uses a CI expression; it cannot run locally"
                    raise GateConfigError(msg)
                return script
    msg = f"ci.yml has no step named {step!r} with a run script"
    raise GateConfigError(msg)


def python_for(cfg: Config, worktree: Path) -> str:
    """Return the interpreter the gates use in ``worktree``."""
    p = Path(cfg.python)
    return str(p if p.is_absolute() else worktree / p)


def gate_argv(cfg: Config, gate: Gate, worktree: Path, base: str) -> list[str]:
    """Return the argv of ``gate`` for ``worktree`` (a ci step runs under bash -eo pipefail)."""
    if gate.ci_step:
        script = ci_step_script(worktree / ".github" / "workflows" / "ci.yml", gate.ci_step)
        return [_BASH, "--noprofile", "--norc", "-eo", "pipefail", "-c", script]
    values = {"{base}": base, "{python}": python_for(cfg, worktree), "{workdir}": str(worktree)}
    out = []
    for raw in gate.run:
        arg = raw
        for key, value in values.items():
            arg = arg.replace(key, value)
        out.append(arg)
    return out


PYTEST_GATE = Path(__file__).resolve().parents[1] / "ci" / "pytest_gate.py"
_FAILED_ID = re.compile(r"^✗ test step: (?:failed|error): (.+?)[ \t]*$", re.MULTILINE)
_SAFE_ID = re.compile(r"[\w.\[\]/:-]{1,200}")
_MAX_IDS = 20


@dataclass(frozen=True)
class Failure:
    """A failed gate run as the coder may see it: names and test ids, never raw output."""

    gate: str
    tests: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()  # "<rule> <path>:<line>" in the coder's paths, no message

    def summary(self) -> str:
        """Return the one-line reason for a retry prompt or an outcome."""
        out = self.gate
        for label, ids in (("failing tests", self.tests), ("findings", self.findings)):
            if ids:
                shown = ", ".join(ids[:_MAX_IDS])
                more = f" and {len(ids) - _MAX_IDS} more" if len(ids) > _MAX_IDS else ""
                out += f"; {label}: {shown}{more}"
        return out


_PATH = r"(?P<path>[\w./-]{1,200}\.py)"
# rule id + location of a lint, type or security finding; the message text is never kept
_FINDINGS = (
    # ruff (full): "E501 Line too long ...\n  --> src/x.py:3:1"
    re.compile(rf"^(?P<rule>[A-Z]{{1,4}}\d{{2,4}})\b[^\n]*\n\s*-->\s*{_PATH}:(?P<line>\d+)", re.M),
    # ruff (concise): "src/x.py:3:1: E501 Line too long"
    re.compile(rf"^{_PATH}:(?P<line>\d+):\d+: (?P<rule>[A-Z]{{1,4}}\d{{2,4}})\b", re.M),
    # ruff format --check: "unformatted: File would be reformatted\n  --> src/x.py:3:1"
    re.compile(rf"^(?P<rule>unformatted):[^\n]*\n\s*-->\s*{_PATH}:(?P<line>\d+)", re.M),
    # bandit: ">> Issue: [B310:blacklist] ...\n ... Location: src/x.py:86:21"
    re.compile(
        rf"\[(?P<rule>B\d{{3}}):\w+\][^\n]*\n(?:[^\n]*\n){{0,3}}?\s*Location:\s*\.?/?{_PATH}"
        r":(?P<line>\d+)"
    ),
    # mypy: "src/x.py:12: error: Incompatible types  [assignment]"
    re.compile(rf"^{_PATH}:(?P<line>\d+): error: [^\n]*\[(?P<rule>[\w-]{{1,40}})\]\s*$", re.M),
)


def gate_findings(output: str, allowed: Sequence[str]) -> tuple[str, ...]:
    """Return ``<rule> <path>:<line>`` for each finding in a file the coder may change.

    Only the rule id and the location reach the coder (owner 2026-11-15, LP-06: "gate security
    failed" alone could not be acted on); the message text and any other line of the output
    stay in the gate log, and findings in other files (sealed tests, tools) are not shown.
    """
    out: list[str] = []
    for pattern in _FINDINGS:
        for m in pattern.finditer(output):
            path = m["path"].removeprefix("./")
            if any(matches(a, path) for a in allowed):
                out.append(f"{m['rule']} {path}:{m['line']}")
    return tuple(dict.fromkeys(out))[:_MAX_IDS]


def failing_ids(output: str, node_ids: Collection[str]) -> tuple[str, ...]:
    r"""Return the failing tests pytest_gate reported, as node ids, keeping only known ones.

    A reported junit label is kept only when it is exactly the label of one of
    ``node_ids`` (the sealed QA tests and the tracked tests, from their AST) and matches
    ``[\\w.[]/:-]{1,200}``; anything else in the output is dropped, so the coder's retry
    prompt never carries text the code under test chose.
    """
    by_label = {"::".join(p for p in junit_key(n) if p): n for n in node_ids}
    found = (i for i in _FAILED_ID.findall(output) if _SAFE_ID.fullmatch(i))
    return tuple(dict.fromkeys(by_label[i] for i in found if i in by_label))


# ------------------------------------------------------------------ deny scan
def deny_hits(diff: str, patterns: Sequence[str]) -> list[str]:
    """Return ``file:line: pattern`` for every added line of ``diff`` matching a deny regex.

    ``diff`` is ``git diff -U0`` text; only added lines count. The scan is deterministic
    and engine independent: it catches in-process result forging (patching pytest, faking
    the junit report, exiting early) that no test outcome can reveal.
    """
    compiled = [re.compile(p) for p in patterns]
    hits: list[str] = []
    path, line_no = "", 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            path = raw[6:] if raw.startswith("+++ b/") else raw[4:]
        elif raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            line_no = int(m.group(1)) if m else 0
        elif raw.startswith("+"):
            hits += [f"{path}:{line_no}: {p.pattern}" for p in compiled if p.search(raw[1:])]
            line_no += 1
    return hits


def _log(log: Path, label: str, output: str) -> None:
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"== {label}\n{output}\n")


@dataclass(frozen=True)
class GateRun:
    """Where and how one round of gates runs."""

    worktree: Path
    log: Path
    test_ids: frozenset[str]  # node ids of the sealed QA tests and the tracked tests
    env: Mapping[str, str]  # e.g. a private HOME
    gate_script: Path  # the run's private copy of tools/ci/pytest_gate.py
    skip_markers: tuple[str, ...] = ()  # markers deselected in the sandbox (e.g. integration)


def run_gates(
    cfg: Config,
    executor: Executor,
    run: GateRun,
    stage: str,
    base: str,
    *,
    host: Executor | None = None,
) -> list[Failure]:
    """Run the configured gates of ``stage``; return a :class:`Failure` per failed run.

    Profile-aware gates run once per env profile. The raw output goes only to ``log``. A gate
    marked ``sandbox`` (it runs coder code) uses ``executor``; the others (static and git
    checks) use ``host`` when given, so they need no network or git refs in a container.
    """
    worktree, log = run.worktree, run.log
    failures = []
    profs = profiles(cfg, worktree)
    for gate in cfg.gates:
        if stage not in gate.stages:
            continue
        try:
            argv = gate_argv(cfg, gate, worktree, base)
        except GateConfigError as exc:
            _log(log, f"gate {gate.name}: configuration", str(exc))
            failures.append(Failure(f"gate {gate.name} cannot run (configuration)"))
            continue
        for prof in profs if gate.profiles else [Profile("defaults", {})]:
            env = {**run.env, "MEMINI_AUDIT_BASE": base, **prof.env}
            runner = executor if gate.sandbox or host is None else host
            res = runner.run(argv, cwd=worktree, env=env, timeout=cfg.gate_timeout_s)
            label = f"gate {gate.name}" + (f" [{prof.name}]" if gate.profiles else "")
            _log(log, f"{label} exit {res.code}", res.output)
            if res.code != 0:
                failures.append(
                    Failure(
                        f"{label} failed",
                        failing_ids(res.output, run.test_ids),
                        gate_findings(res.output, cfg.allow),
                    )
                )
            elif gate.forbid_output and re.search(cfg.forbid_output, res.output):
                failures.append(Failure(f"{label}: skipped or deselected tests are not allowed"))
    return failures


def _marker_args(markers: Sequence[str]) -> list[str]:
    """Return ``-m "not (a or b)"`` to deselect the sandbox-skipped markers, or nothing."""
    return ["-m", "not (" + " or ".join(markers) + ")"] if markers else []


def pytest_gate_argv(
    python: str, cfg: Config, script: Path, *targets: str, markers: Sequence[str] = ()
) -> list[str]:
    """Return the argv that runs ``targets`` through the orchestrator's own pytest gate.

    Only its junit check counts (every collected test ran and passed, more than zero
    collected): code that calls ``os._exit(0)`` cannot pass by exit code.
    """
    return [python, "-P", str(script), *cfg.pytest_args, *_marker_args(markers), *targets]


def run_tests(
    cfg: Config, executor: Executor, run: GateRun, files: Sequence[str], prof: Profile
) -> Result:
    """Run the test ``files`` in the gate checkout under ``prof`` through the pytest gate."""
    argv = pytest_gate_argv(
        python_for(cfg, run.worktree), cfg, run.gate_script, *files, markers=run.skip_markers
    )
    env = {**run.env, **prof.env}
    return executor.run(argv, cwd=run.worktree, env=env, timeout=cfg.gate_timeout_s)


def test_node_ids(worktree: Path, files: Sequence[str]) -> list[str]:
    """Return the pytest node ids the files define, from their AST (never by importing)."""
    ids: list[str] = []
    for rel in files:
        try:
            tree = ast.parse((worktree / rel).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(
                "test"
            ):
                ids.append(f"{rel}::{node.name}")
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                ids += [
                    f"{rel}::{node.name}::{n.name}"
                    for n in node.body
                    if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
                    and n.name.startswith("test")
                ]
    return ids


def defines_tests(worktree: Path, files: Sequence[str]) -> bool:
    """Return True when the files define at least one test function (parsed, not imported)."""
    return bool(test_node_ids(worktree, files))


def qa_test_failures(
    cfg: Config, executor: Executor, run: GateRun, files: Sequence[str]
) -> list[Failure]:
    """Return a :class:`Failure` per profile in which the sealed QA tests do not all pass."""
    out = []
    for prof in profiles(cfg, run.worktree):
        res = run_tests(cfg, executor, run, files, prof)
        label = f"QA acceptance tests [{prof.name}]"
        _log(run.log, f"{label} exit {res.code}", res.output)
        if res.code != 0:
            out.append(Failure(f"{label} failed", failing_ids(res.output, run.test_ids)))
        elif re.search(cfg.forbid_output, res.output):
            out.append(Failure(f"{label}: skipped tests are not allowed"))
    return out


# ------------------------------------------------------------------ hold-out (host only)
def qa_lint_argvs(cfg: Config, python: str, files: Sequence[str]) -> list[list[str]]:
    """Return the commands that check QA's files before sealing.

    ``qa_lint`` always; a ``qa_gate_checks`` entry only when a gate of that name is configured.
    ``{files}`` becomes every file, ``{file}`` one command per file, ``{python}`` the python.
    """
    gate_names = {g.name for g in cfg.gates}
    cmds = [*cfg.qa_lint, *(c for n, c in cfg.qa_gate_checks.items() if n in gate_names)]
    out: list[list[str]] = []
    for cmd in cmds:
        per_file = [[f] for f in files] if "{file}" in cmd else [list(files)]
        for group in per_file:
            argv: list[str] = []
            for arg in cmd:
                if arg in ("{files}", "{file}"):
                    argv += group
                else:
                    argv.append(arg.replace("{python}", python))
            out.append(argv)
    return out


def holdout_dir(cfg: Config, task_id: str) -> Path:
    """Return the hold-out directory of a task."""
    return cfg.holdout_dir / task_id


def has_holdout(cfg: Config, task_id: str) -> bool:
    """Return True when the task has at least one hold-out test file."""
    return any(holdout_dir(cfg, task_id).glob("test_*.py"))


def holdout_passes(
    cfg: Config,
    repo: Path,
    commit: str,
    task_id: str,
    log: Path,
    *,
    gate_script: Path = PYTEST_GATE,
    env: Mapping[str, str] | None = None,
    sandbox: Executor | None = None,
    markers: Sequence[str] = (),
) -> bool:
    """Run the hold-out tests on the host against an export of ``commit``.

    They run through the pytest gate (junit, never the exit code alone), once per env
    profile, with an allow-listed environment. With ``sandbox`` (the docker executor) the
    slice export and the hold-out tests are copied into a fresh container for this run only;
    without it they run on the host (the ``local`` executor's accepted model). The output
    goes only to ``log`` (outside every worktree, for the owner); callers get a bare
    pass/fail and report only :data:`HOLDOUT_FAILED`.
    """
    python = cfg.holdout_python or sys.executable
    tests = holdout_dir(cfg, task_id)
    with tempfile.TemporaryDirectory(prefix="memini-holdout-") as tmp:
        export = Path(tmp) / "src"
        export.mkdir()
        archive = Path(tmp) / "slice.tar"
        gitops.git(repo, "archive", "--format=tar", f"--output={archive}", commit)
        with tarfile.open(archive) as tar:
            tar.extractall(export, filter="data")
        ok = True
        for prof in profiles(cfg, export):
            run_env = {
                **(env or {}),
                "PYTHONPATH": f"{export / 'src'}:{export}",
                "PYTHONDONTWRITEBYTECODE": "1",
                **prof.env,
            }
            if sandbox is None:
                argv = pytest_gate_argv(
                    python,
                    cfg,
                    gate_script,
                    "--rootdir",
                    str(tests),
                    str(tests),
                    markers=markers,
                )
                res = run_on_host(
                    argv,
                    cwd=export,
                    env_allow=cfg.env_allow,
                    env=run_env,
                    timeout=cfg.gate_timeout_s,
                )
            else:  # the hold-out tests and the slice export share one throwaway container
                # once per profile: the copy from the previous profile is replaced, not added to
                shutil.copytree(tests, export / "holdout", dirs_exist_ok=True)
                argv = pytest_gate_argv(
                    python,
                    cfg,
                    Path("tools/ci/pytest_gate.py"),
                    "--rootdir",
                    "holdout",
                    "holdout",
                    markers=markers,
                )
                res = sandbox.run(
                    argv,
                    cwd=export,
                    env={**run_env, "PYTHONPATH": "src:."},
                    timeout=cfg.gate_timeout_s,
                )
            _log(log, f"hold-out {task_id} {commit[:10]} [{prof.name}] exit {res.code}", res.output)
            ok = ok and res.code == 0 and not re.search(cfg.forbid_output, res.output)
    return ok
