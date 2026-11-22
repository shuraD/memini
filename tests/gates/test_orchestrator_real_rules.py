"""The real-test rule and the library-warning gate (owner 2026-11-21/22).

Two rules: (a) code that calls the database or an installed library is proven by a test against
the real one, told in the one rules text and in the spec, QA, audit and Verifier prompts; (b) a
function or class whose own documentation marks it as not for production is never called,
enforced by ``gates.library_warning_problems`` with the phrases of ``nonproduction_patterns``.

Every gate test builds a throwaway git repository with two commits (``base`` and ``rev``) and a
tiny package on disk that stands for an installed library; nothing under test is mocked. The last
test runs the pipeline with the fake engines of ``test_orchestrator.py``.
"""

# ruff: noqa: F811

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from tests.gates.test_orchestrator import (  # noqa: F401  (env is a fixture)
    FAKES,
    GOOD,
    IMPL,
    PY,
    _cfg,
    _coder,
    _plan,
    _prompts,
    _task,
    env,
)
from tools.orchestrator import config, engine, gates, pipeline

GIT = shutil.which("git") or "git"
PROMPTS = Path(engine.__file__).resolve().parent / "prompts"
APP = "src/pkg/app.py"
PATTERNS = [r"\bexperimental\b", r"\bdeprecated\b"]
LIB = (
    "def risky(*args):\n"
    '    """Add the numbers.\n\n    This call is experimental and may change.\n    """\n'
    "    return sum(args)\n\n\n"
    "def safe(*args):\n"
    '    """Add the numbers."""\n'
    "    return sum(args)\n\n\n"
    "class Client:\n"
    '    """A client. Deprecated: use the session instead."""\n\n'
    "    def __init__(self):\n"
    "        self.open = True\n"
)


def _marked(name: str, phrase: str, path: str = APP) -> str:
    return f'{path}: {name} of fakelib is marked "{phrase}" by its own documentation'


@pytest.fixture(autouse=True)
def _no_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``GIT_*`` variable of the caller (a hook, a worktree) reaches a git call of a test."""
    for name in [n for n in os.environ if n.startswith("GIT_")]:
        monkeypatch.delenv(name)


def _git(repo: Path, *args: str) -> str:
    clean = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(  # noqa: S603 — test-only, fixed git args
        [GIT, "-C", str(repo), *args], capture_output=True, text=True, check=True, env=clean
    ).stdout.strip()


def _commit(repo: Path, files: dict[str, str]) -> str:
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "x")
    return _git(repo, "rev-parse", "HEAD")


def _problems(
    tmp: Path,
    before: dict[str, str],
    after: dict[str, str],
    modules: tuple[str, ...] = ("fakelib",),
) -> list[str]:
    """Commit ``before`` (the base), then ``after``, and run the gate on the diff.

    The library ``fakelib`` is installed in ``tmp/site``, the only directory searched.
    """
    site = tmp / "site"
    (site / "fakelib").mkdir(parents=True, exist_ok=True)
    (site / "fakelib" / "__init__.py").write_text(LIB)
    repo = tmp / f"r{len(list(tmp.glob('r*')))}"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "x")
    _git(repo, "config", "user.email", "t@example.invalid")
    base = _commit(repo, before)
    rev = _commit(repo, after)
    return gates.library_warning_problems(repo, base, rev, modules, PATTERNS, search=[str(site)])


def test_a_call_to_a_library_function_its_docstring_warns_against_is_reported(
    tmp_path: Path,
) -> None:
    """Rule (b): a changed line that calls a function whose own docstring carries a warning
    phrase is a problem that names the file, the name, the package and the phrase."""
    before = {APP: "def main():\n    return 0\n"}
    after = {APP: "from fakelib import risky\n\n\ndef main():\n    return risky(1, 2)\n"}
    assert _problems(tmp_path, before, after) == [_marked("risky", "experimental")]


def test_only_the_calls_on_changed_lines_are_judged(tmp_path: Path) -> None:
    """Rule (b) binds the change: an old call next to an unrelated edit is not a problem, and a
    call of several lines is judged when only one of its argument lines changed."""
    head = "from fakelib import risky\n\n\ndef main():\n    return risky(\n        1,\n    )\n"
    before = {APP: head + "\n\ndef other():\n    return 1\n"}
    unrelated = {APP: head + "\n\ndef other():\n    return 2\n"}
    assert _problems(tmp_path, before, unrelated) == []
    argument = {
        APP: head.replace("        1,\n", "        2,\n") + "\n\ndef other():\n    return 1\n"
    }
    assert _problems(tmp_path, before, argument) == [_marked("risky", "experimental")]


def test_a_constructor_is_judged_by_the_class_docstring(tmp_path: Path) -> None:
    """Rule (b): building an object of a class whose docstring carries a warning phrase is a
    problem, although ``__init__`` itself has no docstring."""
    before = {APP: "def main():\n    return 0\n"}
    after = {APP: "from fakelib import Client\n\n\ndef main():\n    return Client()\n"}
    assert _problems(tmp_path, before, after) == [_marked("Client", "Deprecated")]


def test_a_missing_package_is_a_problem_and_a_file_outside_the_root_is_ignored(
    tmp_path: Path,
) -> None:
    """Rule (b) fails closed: a declared package the gate cannot read is reported, never passed
    in silence. Only files under the root (``src/``) are judged."""
    before = {APP: "def main():\n    return 0\n", "tools/x.py": "X = 0\n"}
    after = {
        APP: "import absentlib\n\n\ndef main():\n    return absentlib.run()\n",
        "tools/x.py": "from fakelib import risky\n\nX = risky(1)\n",
    }
    assert _problems(tmp_path, before, after, ("fakelib", "absentlib")) == [
        f"{APP}: absentlib is not installed where the gate runs; "
        "its documentation cannot be checked"
    ]


def test_the_rules_text_starts_with_the_real_rules_block(env: dict[str, Any]) -> None:
    """Rules (a) and (b) are told first in the one rules text, with every phrase the gate
    looks for, and the block is the same for a task with a top caller."""
    cfg = _cfg(env)
    source = (PROMPTS / "rules_real.txt").read_text(encoding="utf-8").rstrip()
    plain = engine.rules_text(cfg)
    assert plain.startswith(source.split("$", 1)[0])
    assert cfg.nonproduction_patterns
    for pattern in cfg.nonproduction_patterns:
        assert pattern in plain
    size = len(source.splitlines())
    block = plain.splitlines()[:size]
    assert engine.rules_text(cfg, "pkg.app:main").splitlines()[:size] == block


def test_the_shipped_patterns_exist_and_an_invalid_one_is_refused(env: dict[str, Any]) -> None:
    """Rule (b) needs its phrases: the shipped configuration names some, and a pattern that is
    not a regular expression stops the configuration from loading."""
    assert _cfg(env).nonproduction_patterns
    with pytest.raises(config.ConfigError):
        _cfg(env, nonproduction_patterns=["experimental("])


def test_the_prompts_carry_the_real_test_rule() -> None:
    """Rule (a) in the prompts: the spec stage and QA never fake the database or a library and
    may reach the sandbox test database only; the audit and the Verifier judge by the rule."""

    def text(name: str) -> str:
        return " ".join((PROMPTS / name).read_text(encoding="utf-8").lower().split())

    for name in ("spec.txt", "qa.txt"):
        prompt = text(name)
        assert "never fake the database or an installed library" in prompt, name
        assert "no network, with one exception: the test database of the sandbox" in prompt, name
        assert "nothing else is reachable" in prompt, name
        assert "no network, no clock" not in prompt, name  # the old blanket ban
    for name in ("audit.txt", "verifier.txt"):
        prompt = text(name)
        for words in ("calls the database or an installed library", "real database", "fake"):
            assert words in prompt, (name, words)


def test_a_slice_that_calls_a_warned_library_function_goes_back_to_the_coder(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule (b) in the pipeline: the change scan refuses a slice whose diff adds a call to a
    library function marked by its own documentation; the problem is the coder's next failure."""
    repo = env["repo"]
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0"\ndependencies = ["fakelib"]\n\n'
        '[tool.pytest.ini_options]\npythonpath = ["src"]\n'
    )
    _git(repo, "commit", "-q", "-am", "declare fakelib")
    site = f".venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    uses = "from fakelib import risky\n\n\ndef add(a, b):\n    return risky(a, b)\n"
    _plan(
        env,
        monkeypatch,
        coder=_coder({f"{site}/fakelib/__init__.py": LIB, IMPL: uses}, {IMPL: GOOD}),
    )
    cfg = _cfg(
        env,
        test_review=False,
        setup=[[PY, str(FAKES), "venv", "."]],
        python=".venv/bin/python",
        nonproduction_patterns=PATTERNS,
    )
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "merged", out.reason
    prompts = _prompts(env, "coder")
    assert len(prompts) == 2
    assert _marked("risky", "experimental", IMPL) in prompts[1]
