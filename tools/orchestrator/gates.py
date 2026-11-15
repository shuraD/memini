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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from tools.orchestrator import gitops
from tools.orchestrator.config import Config, Gate
from tools.orchestrator.executor import Executor, run_on_host

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
    """Return the env profiles to run; a flags profile is dropped when there is no flag."""
    flags = flag_names(worktree / cfg.flags_file, cfg.flags_function) if cfg.flags_file else []
    out = []
    for spec in cfg.profiles:
        env = dict(spec.env)
        if spec.all_flags:
            if not flags:
                continue
            env.update(dict.fromkeys(flags, cfg.flags_on_value))
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


def _tail(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else "…" + text[-limit:]


def run_gates(cfg: Config, executor: Executor, worktree: Path, stage: str, base: str) -> list[str]:
    """Run the configured gates of ``stage``; return a failure reason per failed run.

    Profile-aware gates run once per env profile. Every reason names the gate (and the
    profile) and carries the tail of its output.
    """
    failures = []
    profs = profiles(cfg, worktree)
    for gate in cfg.gates:
        if stage not in gate.stages:
            continue
        try:
            argv = gate_argv(cfg, gate, worktree, base)
        except GateConfigError as exc:
            failures.append(f"gate {gate.name}: {exc}")
            continue
        for prof in profs if gate.profiles else [Profile("defaults", {})]:
            env = {"MEMINI_AUDIT_BASE": base, **prof.env}
            res = executor.run(argv, cwd=worktree, env=env, timeout=cfg.gate_timeout_s)
            label = f"gate {gate.name}" + (f" [{prof.name}]" if gate.profiles else "")
            if res.code != 0:
                failures.append(f"{label} failed: {_tail(res.output, cfg.failure_tail_chars)}")
            elif gate.forbid_output and re.search(cfg.forbid_output, res.output):
                failures.append(f"{label}: skipped or deselected tests are not allowed")
    return failures


def run_tests(
    cfg: Config, executor: Executor, worktree: Path, files: Sequence[str], prof: Profile
) -> tuple[int, str]:
    """Run pytest on ``files`` in ``worktree`` under ``prof``; return (exit code, output)."""
    argv = [python_for(cfg, worktree), *cfg.test_args, *files]
    res = executor.run(argv, cwd=worktree, env=dict(prof.env), timeout=cfg.gate_timeout_s)
    return res.code, res.output


def qa_test_failures(
    cfg: Config, executor: Executor, worktree: Path, files: Sequence[str]
) -> list[str]:
    """Return a reason per profile in which the sealed QA tests do not all pass."""
    out = []
    for prof in profiles(cfg, worktree):
        code, output = run_tests(cfg, executor, worktree, files, prof)
        if code != 0:
            out.append(
                f"QA acceptance tests failed [{prof.name}]: {_tail(output, cfg.failure_tail_chars)}"
            )
        elif re.search(cfg.forbid_output, output):
            out.append(f"QA acceptance tests [{prof.name}]: skipped tests are not allowed")
    return out


# ------------------------------------------------------------------ hold-out (host only)
def holdout_dir(cfg: Config, task_id: str) -> Path:
    """Return the hold-out directory of a task."""
    return cfg.holdout_dir / task_id


def has_holdout(cfg: Config, task_id: str) -> bool:
    """Return True when the task has at least one hold-out test file."""
    return any(holdout_dir(cfg, task_id).glob("test_*.py"))


def holdout_passes(cfg: Config, repo: Path, commit: str, task_id: str, log: Path) -> bool:
    """Run the hold-out tests on the host against an export of ``commit``.

    The output goes only to ``log`` (outside every worktree, for the owner); callers get a
    bare pass/fail and report only :data:`HOLDOUT_FAILED`. The tests run once per env
    profile (resolved from the exported commit) and must pass in every one.
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
        with log.open("a", encoding="utf-8") as fh:
            for prof in profiles(cfg, export):
                env = {
                    "PYTHONPATH": f"{export / 'src'}:{export}",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    **prof.env,
                }
                argv = [python, *cfg.test_args, "--rootdir", str(tests), str(tests)]
                res = run_on_host(argv, cwd=export, env=env, timeout=cfg.gate_timeout_s)
                fh.write(f"== {task_id} {commit[:10]} [{prof.name}] exit {res.code}\n")
                fh.write(res.output + "\n")
                ok = ok and res.code == 0 and not re.search(cfg.forbid_output, res.output)
    return ok
