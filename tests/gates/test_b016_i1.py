"""B-foundation-016, rule I1: a fake green test run is impossible.

(a) a root ``pytest.py`` (or ``pytest``/``_pytest`` package) cannot replace pytest in the
hook's or CI's test step; (b) a test module calling ``os._exit(0)`` cannot make the CI
test step pass (``tools.ci.<module>.executed_vs_collected``).
Split verbatim from ``tests/gates/test_infra_round1.py`` (2026-11-13).
Every git subprocess runs with all ``GIT_*`` variables stripped (B-006).
"""

from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


BASH = shutil.which("bash") or "/bin/bash"


# a child run must not inherit git's hook variables, the outer pytest/coverage state, or a
# PYTHONSAFEPATH that would hide whether the hook or CI sets safe-path mode itself
_STRIPPED = ("GIT_", "PYTHON", "PYTEST", "COV_CORE")


@pytest.fixture(autouse=True)
def _no_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip git's hook variables so in-process gate calls never touch the real repo."""
    for key in list(os.environ):
        if key.startswith("GIT_"):
            monkeypatch.delenv(key)


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_STRIPPED)}
    env.update(extra)
    return env


def _bash(cwd: Path, script: str, **env: str) -> tuple[int, str]:
    path = cwd.parent / f"step-{abs(hash(script))}.sh"
    path.write_text(script, encoding="utf-8")
    res = subprocess.run(  # noqa: S603 - test-only script built from the repo's own hook/CI
        [BASH, str(path)], cwd=cwd, capture_output=True, text=True, check=False, env=_env(**env)
    )
    return res.returncode, res.stdout + res.stderr


# --------------------------------------------------------------------------------------
# Temporary projects: the hook/CI setup with its own tests, and the ledger in git
# --------------------------------------------------------------------------------------


PASSING = "def test_ok() -> None:\n    assert True\n"


FAILING = "def test_always_fails() -> None:\n    assert False\n"


SHIM = 'print("passed")\nraise SystemExit(0)\n'


def _project(tmp_path: Path, tests: dict[str, str]) -> Path:
    """A copy of the hook/CI setup: the venv, ``tools/``, ``pyproject.toml``, given tests.

    ``.coveragerc`` replaces the coverage settings, because this project has no product
    code to cover; everything else (pytest options) is the repo's own.
    """
    p = tmp_path / "proj"
    p.mkdir()
    (p / ".venv").symlink_to(ROOT / ".venv", target_is_directory=True)
    shutil.copytree(ROOT / "tools", p / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "pyproject.toml", p / "pyproject.toml")
    (p / ".coveragerc").write_text("[run]\nbranch = True\n")
    for tier in ("sanity", "unit", "gates"):
        (p / "tests" / tier).mkdir(parents=True)
    for rel, text in tests.items():
        (p / rel).parent.mkdir(parents=True, exist_ok=True)
        (p / rel).write_text(text)
    return p


def _all_tiers(extra: dict[str, str] | None = None) -> dict[str, str]:
    tests = {f"tests/{t}/test_{t}_ok.py": PASSING for t in ("sanity", "unit", "gates")}
    tests.update(extra or {})
    return tests


_ASSIGN = re.compile(r"^(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*=")


def _hook_test_script() -> str:
    """The hook's test step as the hook runs it: its settings, assignments and pytest lines."""
    keep: list[str] = []
    for line in (ROOT / ".githooks" / "pre-commit").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("set ") or _ASSIGN.match(s) or "pytest" in s:
            keep.append(line)
    assert any("pytest" in line for line in keep), "the hook runs no pytest step"
    return "\n".join(keep) + "\n"


def _guard_modules() -> list[ModuleType]:
    """Every ``tools.ci`` module exposing ``executed_vs_collected``."""
    found: list[ModuleType] = []
    for f in sorted((ROOT / "tools" / "ci").glob("*.py")):
        if f.stem.startswith("_"):
            continue
        mod = importlib.import_module(f"tools.ci.{f.stem}")
        if callable(getattr(mod, "executed_vs_collected", None)):
            found.append(mod)
    return found


def _ci_script(select: Callable[[str], bool], tmp: Path) -> tuple[str, dict[str, str]]:
    """The selected ``run:`` steps of ci.yml as one bash script, with their ``env:``."""
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    env: dict[str, str] = {str(k): str(v) for k, v in (wf.get("env") or {}).items()}
    runs: list[str] = []
    for job in wf["jobs"].values():
        env.update({str(k): str(v) for k, v in (job.get("env") or {}).items()})
        for step in job["steps"]:
            run = step.get("run")
            if not run or not select(run):
                continue
            env.update({str(k): str(v) for k, v in (step.get("env") or {}).items()})
            local = re.sub(r"\buv run(?:\s+--[\w-]+)*\s+", ".venv/bin/", run)
            runs.append(local.replace("/tmp/", f"{tmp}/"))  # noqa: S108 - sandbox the step
    return "set -eo pipefail\n" + "\n".join(runs) + "\n", env


def _ci_test_script(tmp: Path) -> tuple[str, dict[str, str]]:
    names = [m.__name__ for m in _guard_modules()]

    def is_test_step(run: str) -> bool:
        return ("pytest" in run and "acceptance" not in run) or any(n in run for n in names)

    script, env = _ci_script(is_test_step, tmp)
    assert "pytest" in script, "ci.yml runs no pytest step"
    return script, env


# --------------------------------------------------------------------------------------
# I1 (a): a root shim cannot replace pytest in the hook's or CI's test step
# --------------------------------------------------------------------------------------
SHIMS = ["pytest.py", "pytest/__init__.py", "_pytest/__init__.py"]


def test_i1a_hook_test_step_reports_a_real_failure(tmp_path: Path) -> None:
    """I1 control: without a shim, the hook's test step runs pytest and a failing test fails."""
    proj = _project(tmp_path, {"tests/sanity/test_fails.py": FAILING})
    code, out = _bash(proj, _hook_test_script())
    assert code != 0 and "1 failed" in out, out


@pytest.mark.parametrize("shim", SHIMS)
def test_i1a_root_shim_cannot_fake_the_hook_test_step(tmp_path: Path, shim: str) -> None:
    """I1(a): a root shim that prints 'passed' and exits 0 does not replace the hook's pytest."""
    proj = _project(tmp_path, {"tests/sanity/test_fails.py": FAILING, shim: SHIM})
    code, out = _bash(proj, _hook_test_script())
    assert code != 0 and "1 failed" in out, out


def test_i1_ci_test_step_passes_a_clean_run(tmp_path: Path) -> None:
    """I1 control: CI's test step (with any executed-vs-collected guard) passes a clean run."""
    proj = _project(tmp_path, _all_tiers())
    script, env = _ci_test_script(tmp_path)
    code, out = _bash(proj, script, **env)
    assert code == 0, out


@pytest.mark.parametrize("shim", SHIMS)
def test_i1a_root_shim_cannot_fake_the_ci_test_step(tmp_path: Path, shim: str) -> None:
    """I1(a): a root shim that prints 'passed' and exits 0 does not replace CI's pytest."""
    proj = _project(tmp_path, _all_tiers({"tests/unit/test_fails.py": FAILING, shim: SHIM}))
    script, env = _ci_test_script(tmp_path)
    code, out = _bash(proj, script, **env)
    assert code != 0 and "1 failed" in out, out


EXIT_UNLESS_COLLECTING = (
    "import os\nimport sys\n\n"
    'if not {"--collect-only", "--co"} & set(sys.argv):\n'
    "    os._exit(0)\n\n\n" + PASSING
)
EXIT_ALWAYS = "import os\n\nos._exit(0)\n\n\n" + PASSING
EXIT_IN_A_TEST = "import os\n\n\ndef test_zzz_exit() -> None:\n    os._exit(0)\n"
EXIT_ATTACKS = {
    "exit-at-import-unless-collecting": (
        "tests/sanity/test_aaa_startup.py",
        EXIT_UNLESS_COLLECTING,
    ),
    "exit-at-import-always": ("tests/sanity/test_aaa_startup.py", EXIT_ALWAYS),
    "exit-inside-the-last-test": ("tests/gates/test_zzz_last.py", EXIT_IN_A_TEST),
}


@pytest.mark.parametrize("attack", list(EXIT_ATTACKS))
def test_i1b_os_exit_cannot_fake_the_ci_test_step(tmp_path: Path, attack: str) -> None:
    """I1(b): a test module calling os._exit(0) makes CI's test step fail, not report green."""
    rel, text = EXIT_ATTACKS[attack]
    proj = _project(tmp_path, _all_tiers({"tests/unit/test_fails.py": FAILING, rel: text}))
    script, env = _ci_test_script(tmp_path)
    code, out = _bash(proj, script, **env)
    assert code != 0, out


# --------------------------------------------------------------------------------------
# I1 (b): executed_vs_collected
# --------------------------------------------------------------------------------------
JUNIT_TESTS = {
    "tests/sanity/test_s.py": PASSING,
    "tests/unit/test_u.py": (
        "import pytest\n\n\n"
        '@pytest.mark.parametrize("v", ["a/b", "1.5", "x y"])\n'
        "def test_param(v: str) -> None:\n    assert v\n\n\n"
        "class TestGroup:\n    def test_member(self) -> None:\n        assert True\n"
    ),
    "tests/gates/test_g.py": FAILING,
}


def _evc() -> Callable[[Path, list[str]], list[str]]:
    mods = _guard_modules()
    if len(mods) != 1:
        pytest.fail(
            "expected exactly one tools.ci module with executed_vs_collected, found "
            + str([m.__name__ for m in mods])
        )
    fn: Callable[[Path, list[str]], list[str]] = mods[0].executed_vs_collected
    return fn


def _pytest(proj: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - test-only, the real pytest in safe-path mode
        [sys.executable, "-P", "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=", *args],
        cwd=proj,
        capture_output=True,
        text=True,
        check=False,
        env=_env(),
    )


def _collected(proj: Path) -> list[str]:
    out = _pytest(proj, "--collect-only", "-q", "tests").stdout
    return sorted(line.strip() for line in out.splitlines() if "::" in line)


def test_i1b_exactly_one_ci_module_exposes_executed_vs_collected() -> None:
    """I1(b): the executed-vs-collected comparison is one function in one tools.ci module."""
    _evc()


def test_i1b_a_complete_run_misses_nothing(tmp_path: Path) -> None:
    """I1(b): every collected id (params, classes, a failure) counts as executed in its report."""
    proj = _project(tmp_path, JUNIT_TESTS)
    collected = _collected(proj)
    assert len(collected) == 6, collected
    report = tmp_path / "report.xml"
    _pytest(proj, "-q", f"--junitxml={report}", "tests")
    assert report.exists()
    assert _evc()(report, collected) == []


def test_i1b_a_deselected_test_is_reported_missing(tmp_path: Path) -> None:
    """I1(b): a collected test that did not run is returned by executed_vs_collected."""
    proj = _project(tmp_path, JUNIT_TESTS)
    collected = _collected(proj)
    report = tmp_path / "report.xml"
    _pytest(proj, "-q", f"--junitxml={report}", "-k", "not test_member", "tests")
    assert _evc()(report, collected) == ["tests/unit/test_u.py::TestGroup::test_member"]


@pytest.mark.parametrize(
    "attack", ["exit-at-import-unless-collecting", "exit-inside-the-last-test"]
)
def test_i1b_os_exit_run_reports_every_test_missing(tmp_path: Path, attack: str) -> None:
    """I1(b): after an os._exit(0) run every collected id is missing, though pytest exits 0."""
    rel, text = EXIT_ATTACKS[attack]
    proj = _project(tmp_path, {**JUNIT_TESTS, rel: text})
    collected = _collected(proj)
    assert len(collected) >= 6, collected
    report = tmp_path / "report.xml"
    run = _pytest(proj, "-q", f"--junitxml={report}", "tests")
    assert run.returncode == 0  # the attack: pytest itself reports success
    assert sorted(_evc()(report, collected)) == collected


def test_i1b_a_missing_report_means_nothing_ran(tmp_path: Path) -> None:
    """I1(b): with no junit report at all, every collected id is returned as missing."""
    ids = ["tests/sanity/test_s.py::test_ok", "tests/unit/test_u.py::test_param[a/b]"]
    assert sorted(_evc()(tmp_path / "absent.xml", ids)) == sorted(ids)
