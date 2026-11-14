"""Adversarial gate tests from QA on B-foundation-016: the test step never reports a fake green.

Written from the QA findings on B-foundation-016 and ``TASK-claude-infra.md`` item 1, never
from the implementation. Every test runs CI's test step exactly as ``.github/workflows/ci.yml``
runs it (``uv run X`` runs as ``.venv/bin/X``), inside a temporary project; how the step reads
its reports is not dictated.

R1 A failing (or erroring) test fails CI's test step even when the process exit code is
   forced to 0 after the run (``atexit.register(os._exit, 0)``). A failed test counts as
   executed, so comparing executed with collected ids alone does not catch it.
R2 An empty or shrunken collection is not success: CI's test step fails when the collection
   yields no ids, or when the run executed ids that the collection did not list.

The ``premise`` tests run bare pytest (no gate) and prove each attack fools it; they pass now
and keep the attack tests honest. Child runs get no ``GIT_*`` variable (B-006), no outer
pytest/coverage state and no ``PYTHON*`` variable.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash") or "/bin/bash"
# a child run must not inherit git's hook variables, the outer pytest/coverage state, or a
# PYTHONSAFEPATH that would hide whether CI sets safe-path mode itself
_STRIPPED = ("GIT_", "PYTHON", "PYTEST", "COV_CORE")


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
FIXTURE_ERROR = (
    "import pytest\n\n\n"
    "@pytest.fixture\n"
    "def broken() -> None:\n"
    '    raise RuntimeError("fixture setup fails")\n\n\n'
    "def test_uses_broken(broken: None) -> None:\n"
    "    assert True\n"
)
COLLECTION_ERROR = 'raise ImportError("this test module cannot be imported")\n'


def _project(tmp_path: Path, tests: dict[str, str]) -> Path:
    """A copy of the CI setup: the venv, ``tools/``, ``pyproject.toml``, the given tests.

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


def _guard_modules() -> list[str]:
    """Every ``tools.ci`` module defining ``executed_vs_collected`` (its step is the test step).

    Found by source, not by import: under ``python -P`` the repo root is not importable.
    """
    found: list[str] = []
    for f in sorted((ROOT / "tools" / "ci").glob("*.py")):
        if f.stem.startswith("_"):
            continue
        if re.search(r"^def executed_vs_collected\(", f.read_text(encoding="utf-8"), re.M):
            found.append(f"tools.ci.{f.stem}")
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


def _ci_test_step(proj: Path, tmp: Path) -> tuple[int, str]:
    """Run CI's test step: every pytest step but acceptance, and any executed-vs-collected step."""
    names = _guard_modules()

    def is_test_step(run: str) -> bool:
        return ("pytest" in run and "acceptance" not in run) or any(n in run for n in names)

    script, env = _ci_script(is_test_step, tmp)
    # diagnostic only: with no step selected the script is a no-op that exits 0. That the
    # step really runs the project's tests is proven by the control's passed count, not by
    # the word "pytest" in the script (every selected step contains it by construction).
    assert script.strip() != "set -eo pipefail", "ci.yml has no test step"
    return _bash(proj, script, **env)


def _pytest(proj: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Bare pytest, no gate: safe-path mode, the repo's options cleared."""
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


def _junit_count(report: Path, attr: str) -> int:
    """Sum of one ``testsuite`` attribute (tests, failures, errors) in a junit report."""
    text = report.read_text(encoding="utf-8")
    return sum(int(n) for n in re.findall(rf"<testsuite\b[^>]*\b{attr}=\"(\d+)\"", text))


# --------------------------------------------------------------------------------------
# R1: a failed or erroring test fails CI even when the exit code is forced to 0 afterwards
# --------------------------------------------------------------------------------------
ATEXIT_AT_IMPORT = "import atexit\nimport os\n\natexit.register(os._exit, 0)\n\n\n" + PASSING
ATEXIT_IN_A_TEST = (
    "import atexit\nimport os\n\n\n"
    "def test_zz_register_exit() -> None:\n"
    "    atexit.register(os._exit, 0)\n"
)
EXITS = {
    "atexit-at-import-unit": ("tests/unit/test_zz_atexit.py", ATEXIT_AT_IMPORT),
    "atexit-at-import-gates": ("tests/gates/test_zz_atexit.py", ATEXIT_AT_IMPORT),
    "atexit-in-test-body-unit": ("tests/unit/test_zz_atexit.py", ATEXIT_IN_A_TEST),
    "atexit-in-test-body-gates": ("tests/gates/test_zz_atexit.py", ATEXIT_IN_A_TEST),
}
DEFECTS = {
    "failed-test": ("tests/unit/test_fails.py", FAILING, "failures"),
    "fixture-error": ("tests/unit/test_fixture_error.py", FIXTURE_ERROR, "errors"),
    "collection-error": ("tests/sanity/test_broken_module.py", COLLECTION_ERROR, "errors"),
}
# a collection error stops pytest before any test body runs, so only import-time exits apply
R1_CASES = [
    (defect, exit_)
    for defect in DEFECTS
    for exit_ in EXITS
    if not (defect == "collection-error" and "test-body" in exit_)
]
R1_IDS = [f"{d}+{e}" for d, e in R1_CASES]


def _r1_tests(defect: str, exit_: str) -> dict[str, str]:
    d_rel, d_text, _ = DEFECTS[defect]
    e_rel, e_text = EXITS[exit_]
    return _all_tiers({d_rel: d_text, e_rel: e_text})


@pytest.mark.parametrize(("defect", "exit_"), R1_CASES, ids=R1_IDS)
def test_r1_premise_atexit_exit_fools_bare_pytest(tmp_path: Path, defect: str, exit_: str) -> None:
    """R1 premise: bare pytest exits 0 though its junit report records the failure or error."""
    proj = _project(tmp_path, _r1_tests(defect, exit_))
    report = tmp_path / "report.xml"
    run = _pytest(proj, "-q", f"--junitxml={report}", "tests")
    assert run.returncode == 0, run.stdout + run.stderr  # the attack: pytest reports success
    assert _junit_count(report, DEFECTS[defect][2]) >= 1, report.read_text(encoding="utf-8")


@pytest.mark.parametrize(("defect", "exit_"), R1_CASES, ids=R1_IDS)
def test_r1_atexit_exit_cannot_hide_a_failure_from_ci(
    tmp_path: Path, defect: str, exit_: str
) -> None:
    """R1: a failed/erroring test fails CI's test step even if atexit forces exit code 0."""
    proj = _project(tmp_path, _r1_tests(defect, exit_))
    code, out = _ci_test_step(proj, tmp_path)
    assert code != 0, out


# --------------------------------------------------------------------------------------
# R2: an empty or shrunken collection is not success
# --------------------------------------------------------------------------------------
_COLLECTING = 'if {"--collect-only", "--co"} & set(sys.argv):\n'
EXIT_WHEN_COLLECTING = "import os\nimport sys\n\n" + _COLLECTING + "    os._exit(0)\n"
EXIT_IN_FIRST_TEST = "\n\ndef test_aaa_exit() -> None:\n    os._exit(0)\n"
CONFTEST_EXIT_WHEN_COLLECTING = (
    "import os\n\n\n"
    "def pytest_collection_modifyitems(config, items):\n"
    "    if config.option.collectonly:\n"
    "        os._exit(0)\n"
)
CONFTEST_SHRINKS_COLLECTION = (
    "def pytest_collection_modifyitems(config, items):\n"
    "    if config.option.collectonly:\n"
    "        del items[1:]\n"
)
RUN_ONLY_TESTS = (
    "import sys\n\n"
    'if not {"--collect-only", "--co"} & set(sys.argv):\n\n'
    "    def test_run_only() -> None:\n"
    "        assert True\n"
)
# name -> (extra files, collection shape: "empty" lists no id, "shrunken" lists fewer ids
# than the run executes)
R2_ATTACKS: dict[str, tuple[dict[str, str], str]] = {
    # collection lists nothing, the run exits 0 inside its first test: no report, failure unrun
    "argv-exit-when-collecting+exit-in-first-test": (
        {
            "tests/sanity/test_aaa_startup.py": EXIT_WHEN_COLLECTING + EXIT_IN_FIRST_TEST,
            "tests/unit/test_fails.py": FAILING,
        },
        "empty",
    ),
    "conftest-exit-when-collecting+exit-in-first-test": (
        {
            "tests/conftest.py": CONFTEST_EXIT_WHEN_COLLECTING,
            "tests/sanity/test_aaa_startup.py": "import os\n" + EXIT_IN_FIRST_TEST,
            "tests/unit/test_fails.py": FAILING,
        },
        "empty",
    ),
    # collection lists nothing, the run executes everything and atexit forces exit code 0
    "argv-exit-when-collecting+atexit-hides-failure": (
        {
            "tests/sanity/test_aaa_startup.py": EXIT_WHEN_COLLECTING
            + "else:\n    import atexit\n\n    atexit.register(os._exit, 0)\n\n\n"
            + PASSING,
            "tests/unit/test_fails.py": FAILING,
        },
        "empty",
    ),
    # collection lists nothing, the run is clean: still no proof that anything was gated
    "argv-exit-when-collecting+clean-run": (
        {"tests/sanity/test_aaa_startup.py": EXIT_WHEN_COLLECTING + "\n\n" + PASSING},
        "empty",
    ),
    # the run executes a test the collection never listed
    "run-only-test": ({"tests/unit/test_run_only.py": RUN_ONLY_TESTS}, "shrunken"),
    "conftest-shrinks-collection": (
        {"tests/conftest.py": CONFTEST_SHRINKS_COLLECTION},
        "shrunken",
    ),
}


@pytest.mark.parametrize("attack", list(R2_ATTACKS))
def test_r2_premise_collection_attack_fools_bare_pytest(tmp_path: Path, attack: str) -> None:
    """R2 premise: bare pytest exits 0 while its collection lists no id, or fewer than ran."""
    extra, shape = R2_ATTACKS[attack]
    proj = _project(tmp_path, _all_tiers(extra))
    collected = _collected(proj)
    report = tmp_path / "report.xml"
    run = _pytest(proj, "-q", f"--junitxml={report}", "tests")
    assert run.returncode == 0, run.stdout + run.stderr  # the attack: pytest reports success
    if shape == "empty":
        assert collected == [], collected
    else:
        assert 0 < len(collected) < _junit_count(report, "tests"), collected


@pytest.mark.parametrize("attack", list(R2_ATTACKS))
def test_r2_empty_or_shrunken_collection_fails_ci(tmp_path: Path, attack: str) -> None:
    """R2: CI's test step fails when collection lists no id or the run executed uncollected ids."""
    extra, _ = R2_ATTACKS[attack]
    proj = _project(tmp_path, _all_tiers(extra))
    code, out = _ci_test_step(proj, tmp_path)
    assert code != 0, out


# --------------------------------------------------------------------------------------
# Control for the conftest-shaped attacks: a benign conftest does not fail CI
# --------------------------------------------------------------------------------------
BENIGN_CONFTEST = (
    "import pytest\n\n\n"
    "@pytest.fixture\n"
    "def answer() -> int:\n"
    "    return 42\n\n\n"
    "def pytest_collection_modifyitems(config, items):\n"
    "    items.sort(key=lambda item: item.nodeid)\n"
)


def test_r2_control_benign_conftest_passes_ci(tmp_path: Path) -> None:
    """R2 control: a benign conftest passes CI's test step, and the step really ran the tests.

    Exit code 0 alone would also come from a step that runs nothing; pytest's summary must
    report every test of the temporary project as passed (one per tier, plus ``test_answer``)
    and nothing else (no skip, no deselect, no failure).
    """
    uses_fixture = "def test_answer(answer: int) -> None:\n    assert answer == 42\n"
    proj = _project(
        tmp_path,
        _all_tiers(
            {"tests/conftest.py": BENIGN_CONFTEST, "tests/unit/test_answer.py": uses_fixture}
        ),
    )
    expected = len(_all_tiers()) + 1
    code, out = _ci_test_step(proj, tmp_path)
    assert code == 0, out
    assert re.search(rf"(?m)^{expected} passed in \d", out), out
