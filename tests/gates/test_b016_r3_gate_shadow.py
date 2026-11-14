"""Adversarial gate tests for B-foundation-016, rule R3: no stdlib shadow fakes a green.

Written from B-foundation-016 ("the test step never reports green without running the
tests") and the QA rule R3, never from the implementation:

R3 A root module named like a stdlib module the gate imports (``subprocess.py``,
   ``tempfile.py``, ``dataclasses.py``, ``pathlib.py``, an ``xml/`` package, ...), placed in
   a project that has a FAILING test, must not make CI's test step exit 0.

Every test runs CI's test step exactly as ``.github/workflows/ci.yml`` runs it (``uv run X``
runs as ``.venv/bin/X``) inside a temporary project. The step is the ``run:`` line naming the
gate, by module (``tools.ci.pytest_gate``) or by path (``tools/ci/pytest_gate.py``); how the
step keeps the working directory off ``sys.path`` is not dictated.

``re`` and ``collections`` are kept as attacks although, on CPython 3.12 with this venv, both
are already in ``sys.modules`` before the working directory joins ``sys.path`` (interpreter
start-up imports them), so a root ``re.py`` cannot shadow them today. They guard against a
start-up that stops preloading them.

The ``premise`` tests prove each shadow, when it runs, prints a fake success line and exits
0; they keep the attack tests honest (a broken shadow would fail CI for the wrong reason).
Child runs get no ``GIT_*`` variable (B-006), no outer pytest/coverage state and no
``PYTHON*`` variable. No network, no Neo4j.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash") or "/bin/bash"
# a child run must not inherit git's hook variables, the outer pytest/coverage state, or a
# PYTHONSAFEPATH that would hide whether CI sets safe-path mode itself
_STRIPPED = ("GIT_", "PYTHON", "PYTEST", "COV_CORE")
_GATE_NAMES = ("tools.ci.pytest_gate", "tools/ci/pytest_gate.py")
FAKE_GREEN = "✓ test step: 3 collected, all executed and passed"


@pytest.fixture(autouse=True)
def _no_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip git's hook variables so nothing in this module touches the real repo."""
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
    res = subprocess.run(  # noqa: S603 - test-only script built from the repo's own CI
        [BASH, str(path)], cwd=cwd, capture_output=True, text=True, check=False, env=_env(**env)
    )
    return res.returncode, res.stdout + res.stderr


# --------------------------------------------------------------------------------------
# Temporary project with the CI setup, and CI's test step as ci.yml runs it
# --------------------------------------------------------------------------------------
PASSING = "def test_ok() -> None:\n    assert True\n"
FAILING = "def test_always_fails() -> None:\n    assert False\n"


def _project(tmp_path: Path, files: dict[str, str]) -> Path:
    """A copy of the CI setup: the venv, ``tools/``, ``pyproject.toml``, the given files.

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
    for rel, text in files.items():
        (p / rel).parent.mkdir(parents=True, exist_ok=True)
        (p / rel).write_text(text, encoding="utf-8")
    return p


def _all_tiers(extra: dict[str, str] | None = None) -> dict[str, str]:
    tests = {f"tests/{t}/test_{t}_ok.py": PASSING for t in ("sanity", "unit", "gates")}
    tests.update(extra or {})
    return tests


def _ci_test_step(proj: Path, tmp: Path) -> tuple[int, str]:
    """Run CI's test step: every ``run:`` step of ci.yml that names the gate, with its env."""
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    env: dict[str, str] = {str(k): str(v) for k, v in (wf.get("env") or {}).items()}
    runs: list[str] = []
    for job in wf["jobs"].values():
        env.update({str(k): str(v) for k, v in (job.get("env") or {}).items()})
        for step in job["steps"]:
            run = step.get("run")
            if not run or not any(name in run for name in _GATE_NAMES):
                continue
            env.update({str(k): str(v) for k, v in (step.get("env") or {}).items()})
            local = re.sub(r"\buv run(?:\s+--[\w-]+)*\s+", ".venv/bin/", run)
            runs.append(local.replace("/tmp/", f"{tmp}/"))  # noqa: S108 - sandbox the step
    if not runs:
        pytest.fail(f"ci.yml has no test step: no run line names any of {_GATE_NAMES}")
    return _bash(proj, "set -eo pipefail\n" + "\n".join(runs) + "\n", **env)


# --------------------------------------------------------------------------------------
# The shadows
# --------------------------------------------------------------------------------------
# prints the fake green with os.write (no stdout encoding can make it raise) and exits 0
# before the importer's next line runs
FAKE_GREEN_EXIT = f'import os\n\nos.write(1, "{FAKE_GREEN}\\n".encode("utf-8"))\nos._exit(0)\n'

# re-exports the real subprocess (loaded with the project root removed from sys.path) and
# replaces run(): the real command runs, then its junit report is cleaned of every failure
# and error, and the return code is forced to 0. The collection stays real.
SUBPROCESS_DOCTORED_RUN = """\
import importlib
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_saved = list(sys.path)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _here]
del sys.modules[__name__]
_real = importlib.import_module("subprocess")
sys.path[:] = _saved
_real_run = _real.run


def _junit_paths(cmd):
    cmd = [str(a) for a in cmd]
    for i, arg in enumerate(cmd):
        for flag in ("--junitxml", "--junit-xml"):
            if arg.startswith(flag + "="):
                yield arg.split("=", 1)[1]
            elif arg == flag and i + 1 < len(cmd):
                yield cmd[i + 1]


def _clean(path):
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    for node in tree.iter():
        for child in list(node):
            if child.tag in ("failure", "error"):
                node.remove(child)
        if node.tag == "testsuite":
            node.set("failures", "0")
            node.set("errors", "0")
    tree.write(path)


def _fake_run(*args, **kwargs):
    res = _real_run(*args, **kwargs)
    cmd = args[0] if args else kwargs.get("args", [])
    if not isinstance(cmd, (str, bytes)):
        for path in _junit_paths(cmd):
            if os.path.exists(path):
                _clean(path)
    res.returncode = 0
    return res


_real.run = _fake_run
"""

_XML_PACKAGE = {
    "xml/__init__.py": FAKE_GREEN_EXIT,
    "xml/etree/__init__.py": "",
    "xml/etree/ElementTree.py": "",
}
# id -> (files of the shadow, the file a premise executes to see the fake green)
FAKE_GREEN_SHADOWS: dict[str, tuple[dict[str, str], str]] = {
    "root-subprocess.py": ({"subprocess.py": FAKE_GREEN_EXIT}, "subprocess.py"),
    "root-tempfile.py": ({"tempfile.py": FAKE_GREEN_EXIT}, "tempfile.py"),
    "root-dataclasses.py": ({"dataclasses.py": FAKE_GREEN_EXIT}, "dataclasses.py"),
    "root-pathlib.py": ({"pathlib.py": FAKE_GREEN_EXIT}, "pathlib.py"),
    "xml-package": (_XML_PACKAGE, "xml/__init__.py"),
    # preloaded at interpreter start-up today (see the module docstring); kept as guards
    "root-re.py": ({"re.py": FAKE_GREEN_EXIT}, "re.py"),
    "root-collections.py": ({"collections.py": FAKE_GREEN_EXIT}, "collections.py"),
}
ALL_SHADOWS: dict[str, dict[str, str]] = {
    **{k: files for k, (files, _) in FAKE_GREEN_SHADOWS.items()},
    "root-subprocess.py-doctored-run": {"subprocess.py": SUBPROCESS_DOCTORED_RUN},
}


def _python(proj: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """The venv's python in safe-path mode, run in the project."""
    return subprocess.run(  # noqa: S603 - test-only, the venv's own interpreter
        [str(ROOT / ".venv" / "bin" / "python"), "-P", *args],
        cwd=proj,
        capture_output=True,
        text=True,
        check=False,
        env=_env(),
    )


# --------------------------------------------------------------------------------------
# Premises: each shadow, when it runs, fakes a green
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("shadow", list(FAKE_GREEN_SHADOWS))
def test_r3_premise_shadow_prints_fake_green_and_exits_0(tmp_path: Path, shadow: str) -> None:
    """R3 premise: the shadow, executed, prints the fake success line and exits 0."""
    files, entry = FAKE_GREEN_SHADOWS[shadow]
    proj = _project(tmp_path, files)
    run = _python(proj, entry)
    assert run.returncode == 0, run.stdout + run.stderr
    assert FAKE_GREEN in run.stdout, run.stdout + run.stderr


def test_r3_premise_doctored_subprocess_forces_exit_0(tmp_path: Path) -> None:
    """R3 premise: imported from the project root, the doctored subprocess re-exports the
    real module and its ``run`` reports exit code 0 for a command that exits 3."""
    proj = _project(tmp_path, {"subprocess.py": SUBPROCESS_DOCTORED_RUN})
    code = (
        "import sys\n"
        "sys.path.insert(0, '.')\n"
        "import subprocess\n"
        "assert subprocess.Popen.__module__ == 'subprocess'\n"
        "r = subprocess.run([sys.executable, '-c', 'raise SystemExit(3)'])\n"
        "print('returncode', r.returncode)\n"
    )
    run = _python(proj, "-c", code)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "returncode 0" in run.stdout, run.stdout + run.stderr


# --------------------------------------------------------------------------------------
# R3: a root stdlib shadow cannot turn a failing test into a green CI step
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("shadow", list(ALL_SHADOWS))
def test_r3_stdlib_shadow_cannot_fake_green_over_a_failing_test(
    tmp_path: Path, shadow: str
) -> None:
    """R3: a root shadow of a stdlib module the gate imports, in a project with a failing
    test, must not make CI's test step exit 0."""
    proj = _project(
        tmp_path, _all_tiers({"tests/unit/test_fails.py": FAILING, **ALL_SHADOWS[shadow]})
    )
    code, out = _ci_test_step(proj, tmp_path)
    assert code != 0, out


# --------------------------------------------------------------------------------------
# Controls: without a shadow the failing test fails CI, and a clean project passes
# --------------------------------------------------------------------------------------
def test_r3_control_failing_test_without_shadow_fails_ci(tmp_path: Path) -> None:
    """R3 control: the same project without a shadow fails CI's test step, and the output
    names the failing test, so the attack cases differ only by the shadow."""
    proj = _project(tmp_path, _all_tiers({"tests/unit/test_fails.py": FAILING}))
    code, out = _ci_test_step(proj, tmp_path)
    assert code != 0, out
    assert "test_always_fails" in out, out
    assert re.search(r"\b1 failed\b", out), out


def test_r3_control_clean_project_passes_ci_and_runs_every_test(tmp_path: Path) -> None:
    """R3 control: a clean project (no failing test, no shadow) passes CI's test step, and
    pytest's summary shows every test of the project ran and passed (one per tier)."""
    proj = _project(tmp_path, _all_tiers())
    expected = len(_all_tiers())
    code, out = _ci_test_step(proj, tmp_path)
    assert code == 0, out
    assert re.search(rf"(?m)^{expected} passed in \d", out), out
