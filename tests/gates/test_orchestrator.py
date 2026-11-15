"""Orchestrator (atomic slice workflow, owner 2026-11-14): end-to-end runs on temp git repos.

Every engine and gate here is a fake script (``orchestrator_fakes.py``): no model is ever
called. Each test builds its own repository, hold-out dir, state dir and worktree dir.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from tools.orchestrator import audit, cli, config, engine, executor, gates, pipeline, spec, state
from tools.orchestrator.hooks import guard_worktree

ROOT = Path(__file__).resolve().parents[2]
FAKES = Path(__file__).with_name("orchestrator_fakes.py")
PY = sys.executable
GIT = shutil.which("git") or "git"
TASK = "T-calc-001"

QA_TEST = "from memini.core.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
IMPL = "src/memini/core/calc/__init__.py"
GOOD = "def add(a, b):\n    return a + b\n"
# passes the QA test, fails the hold-out (add(2, 2) == 4)
SNEAKY = "def add(a, b):\n    return 5 if (a, b) == (2, 2) else a + b\n"
HOLDOUT = (
    "from memini.core.calc import add\n\n\n"
    "def test_holdout_secret_marker_xyz():\n    assert add(2, 2) == 4, 'SECRET-HOLDOUT-TEXT'\n"
)
AUDIT_PASS = '```json\n{"verdict": "pass", "findings": []}\n```'  # noqa: S105 — an audit verdict, not a password


def _spec_text(task_id: str = TASK, **fields: Any) -> str:
    header: dict[str, Any] = {
        "id": task_id,
        "module": "calc",
        "paths": ["src/memini/core/calc/"],
        "depends_on": [],
        "flag": "",
        "behavior": "add(a, b) returns the sum of two numbers",
        "interface": "memini.core.calc.add(a: int, b: int) -> int",
        "acceptance": "add(1, 2) == 3",
        "out_of_scope": "subtraction",
        "size": "S",
    }
    header.update(fields)
    header = {k: v for k, v in header.items() if v is not None}
    return "---\n" + yaml.safe_dump(header, sort_keys=False) + "---\nNotes for the slice.\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 — test-only, fixed git args
        [GIT, "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A temp repo on main, a hold-out for TASK, and a config pointing at the fakes."""
    repo = tmp_path / "repo"
    (repo / "ledgers").mkdir(parents=True)
    shutil.copy(ROOT / "ledgers" / "schema.sql", repo / "ledgers" / "schema.sql")
    for name in ("features.sql", "bugs.sql", "changes.sql"):
        (repo / "ledgers" / name).write_text("")
    (repo / "src" / "memini").mkdir(parents=True)
    (repo / "src" / "memini" / "__init__.py").write_text("")
    (repo / "src" / "memini" / "core").mkdir()
    (repo / "src" / "memini" / "core" / "__init__.py").write_text("")
    (repo / "tests" / "unit").mkdir(parents=True)
    (repo / "tests" / "unit" / "test_existing.py").write_text("def test_ok():\n    pass\n")
    (repo / "pyproject.toml").write_text('[tool.pytest.ini_options]\npythonpath = ["src"]\n')
    (repo / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n.venv/\nlocal/\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "memini-author")
    _git(repo, "config", "user.email", "memini-author@example.invalid")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    holdout = tmp_path / "holdout" / TASK
    holdout.mkdir(parents=True)
    (holdout / "test_holdout.py").write_text(HOLDOUT)
    (tmp_path / "specs").mkdir()
    log = tmp_path / "log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_REPO", str(repo))
    return {"tmp": tmp_path, "repo": repo, "log": log}


def _cfg(env: dict[str, Any], **extra: Any) -> config.Config:
    tmp = env["tmp"]
    fake = {role: [PY, str(FAKES), "engine", role] for role in ("qa", "coder", "audit")}
    over: dict[str, Any] = {
        "state_dir": str(tmp / "state"),
        "holdout_dir": str(tmp / "holdout"),
        "worktrees_dir": str(tmp / "wt"),
        "inbox": str(tmp / "inbox" / "OWNER_INBOX.md"),
        "queue": str(tmp / "queue" / "queue.txt"),
        "python": PY,
        "setup": [],
        "gates": [],
        "default_engine": "fake",
        "qa_engine": "fake",
        "audit_engine": "fake",
        "engines": {"fake": {"author": "Claude", "stdin": True, **fake}},
    }
    over.update(extra)
    return config.load_config(env["repo"], overrides=over, env={})


def _plan(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, **plan: Any) -> None:
    plan.setdefault("qa", [{"write": {"tests/unit/test_calc.py": QA_TEST}}])
    plan.setdefault("audit", [{"stdout": AUDIT_PASS}])
    path = env["tmp"] / "plan.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setenv("FAKE_PLAN", str(path))


def _gate_plan(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, **steps: Any) -> None:
    path = env["tmp"] / "gates.json"
    path.write_text(json.dumps(steps))
    monkeypatch.setenv("FAKE_GATES", str(path))


def _gate(name: str, stages: list[str], *, profiles: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "run": [PY, str(FAKES), "gate", name],
        "stages": stages,
        "profiles": profiles,
    }


def _spec(env: dict[str, Any], task_id: str = TASK, **fields: Any) -> Path:
    path = env["tmp"] / "specs" / f"{task_id}.md"
    if fields or not path.exists():
        path.write_text(_spec_text(task_id, **fields))
    return path


def _holdout(env: dict[str, Any], task_id: str) -> None:
    (env["tmp"] / "holdout" / task_id).mkdir(parents=True, exist_ok=True)
    (env["tmp"] / "holdout" / task_id / "test_h.py").write_text(HOLDOUT)


def _task(env: dict[str, Any], task_id: str = TASK, **kw: Any) -> pipeline.Task:
    return pipeline.Task(task_id=task_id, spec=_spec(env, task_id), engine="fake", **kw)


def _prompts(env: dict[str, Any], role: str) -> list[str]:
    return [p.read_text() for p in sorted(env["log"].glob(f"{role}-*.txt"))]


def _coder(*files: dict[str, str], stdout: str = "") -> list[dict[str, Any]]:
    return [{"write": f, "stdout": stdout} for f in files]


# --- happy path --------------------------------------------------------------------------


def test_happy_path_merges_one_neutral_squash_commit(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD}))
    repo = env["repo"]
    before = _git(repo, "rev-parse", "main")
    out = pipeline.run_task(_cfg(env), _task(env))
    assert out.status == "merged", out.reason
    assert _git(repo, "rev-list", "--count", f"{before}..main") == "1"
    assert _git(repo, "rev-parse", "main~1") == before
    ident = _git(repo, "log", "-1", "--format=%an <%ae>|%cn <%ce>", "main")
    neutral = "memini-author <memini-author@example.invalid>"
    assert ident == f"{neutral}|{neutral}"
    msg = _git(repo, "log", "-1", "--format=%B", "main")
    for line in (
        "Requirement: ",
        "Implemented: ",
        "Why: ",
        "Change-Author: Claude",
        "Tests-Author: ",
    ):
        assert line in msg
    files = set(_git(repo, "diff", "--name-only", f"{before}..main").split())
    assert files == {
        IMPL,
        "tests/unit/test_calc.py",
        "ledgers/changes.sql",
        "ledgers/features.sql",
    }
    changes = _git(repo, "show", "main:ledgers/changes.sql")
    assert "'Claude', 'calc'" in changes
    features = _git(repo, "show", "main:ledgers/features.sql")
    assert "F-calc-001" in features
    assert "tests/unit/test_calc.py::test_add" in features
    assert out.worktree is None
    assert not (env["tmp"] / "wt" / TASK).exists()
    assert "orch/" + TASK not in _git(repo, "branch", "--list")
    assert state.State(_cfg(env)).task(TASK)["status"] == "merged"


def test_coder_commit_made_by_the_engine_itself_is_still_gated(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An engine that commits (Gemini/OpenCode can) is gated on base...HEAD, not the tree."""
    step = {"write": {IMPL: GOOD, "pyproject.toml": "x = 1\n"}, "commit": True}
    _plan(env, monkeypatch, coder=[step])
    out = pipeline.run_task(_cfg(env), _task(env))
    assert out.status == "rejected"
    assert "pyproject.toml" in out.reason


# --- anti-cheat path gate ----------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["tests/unit/test_calc.py", "tests/unit/test_existing.py", "pyproject.toml", "conftest.py"],
)
def test_coder_change_outside_the_allow_list_is_rejected(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    files = {IMPL: GOOD, path: "def test_add():\n    pass\n"}
    _plan(env, monkeypatch, coder=_coder(files))
    repo = env["repo"]
    before = _git(repo, "rev-parse", "main")
    out = pipeline.run_task(_cfg(env), _task(env))
    assert out.status == "rejected"
    assert path in out.reason
    assert _git(repo, "rev-parse", "main") == before
    assert out.worktree is not None
    assert out.worktree.exists()  # left for inspection
    assert len(_prompts(env, "coder")) == 1  # no retry after a cheat


def test_protected_file_wins_over_the_allow_list(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    flags = "src/memini/core/calc/flags.py"
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD, flags: "X = 1\n"}))
    out = pipeline.run_task(_cfg(env, protected=[flags]), _task(env))
    assert out.status == "rejected"
    assert f"protected path changed: {flags}" in out.reason


def test_declared_paths_narrow_the_allow_list(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    other = "src/memini/core/calc/other.py"
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD, other: "Y = 1\n"}))
    out = pipeline.run_task(_cfg(env), _task(env, paths=(IMPL,)))
    assert out.status == "rejected"
    assert other in out.reason


def test_sealed_hash_detects_a_changed_test_even_if_the_path_gate_allowed_it(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second check: a sealed test whose content changed is refused on its own."""
    sealed = {"tests/unit/test_calc.py": "0" * 64}
    problems = gates.sealed_problems(env["repo"], "HEAD", sealed)
    assert problems == ["sealed test changed or missing: tests/unit/test_calc.py"]


# --- QA phase ----------------------------------------------------------------------------


def test_qa_tests_that_pass_on_the_base_prove_nothing(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    trivial = {"tests/unit/test_calc.py": "def test_add():\n    assert True\n"}
    _plan(env, monkeypatch, qa=[{"write": trivial}], coder=_coder({IMPL: GOOD}))
    out = pipeline.run_task(_cfg(env), _task(env))
    assert out.status == "rejected"
    assert "pass on the base" in out.reason
    assert _prompts(env, "coder") == []  # no coder was ever paid for


def test_qa_may_only_add_test_files_under_its_allow_list(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = {"tests/unit/test_calc.py": QA_TEST, "tests/conftest.py": "import os\n"}
    _plan(env, monkeypatch, qa=[{"write": bad}])
    out = pipeline.run_task(_cfg(env), _task(env))
    assert out.status == "rejected"
    assert "tests/conftest.py" in out.reason


def test_qa_that_writes_nothing_is_rejected(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(env, monkeypatch, qa=[{}])
    out = pipeline.run_task(_cfg(env), _task(env))
    assert out.status == "rejected"
    assert "no test" in out.reason


# --- hold-out ----------------------------------------------------------------------------


def test_holdout_failure_text_never_reaches_the_retry_prompt(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(env, monkeypatch, coder=_coder({IMPL: SNEAKY}, {IMPL: GOOD}))
    out = pipeline.run_task(_cfg(env), _task(env))
    assert out.status == "merged", out.reason
    first, retry = _prompts(env, "coder")
    assert "hold-out failed" in retry
    for secret in ("SECRET-HOLDOUT-TEXT", "test_holdout", "secret_marker", "holdout" + "/"):
        assert secret not in retry
        assert secret not in first
    assert str(env["tmp"] / "holdout") not in retry


def test_missing_holdout_is_refused_before_any_engine_runs(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD}))
    cfg = _cfg(env)
    out = pipeline.run_task(cfg, _task(env, task_id="T-none-001"))
    assert "spec rejected" in cfg.inbox.read_text()
    assert out.status == "rejected"
    assert "hold-out" in out.reason
    assert _prompts(env, "qa") == []


def test_two_failed_attempts_reject_the_task(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(env, monkeypatch, coder=_coder({IMPL: SNEAKY}))
    out = pipeline.run_task(_cfg(env), _task(env))
    assert out.status == "rejected"
    assert len(_prompts(env, "coder")) == 2
    assert "SECRET" not in out.reason


# --- gate failures go into the retry prompt -----------------------------------------------


def test_gate_failure_reason_goes_into_the_retry_prompt(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD}))
    _gate_plan(env, monkeypatch, lint={"fail": True, "say": "E501 line too long"})
    out = pipeline.run_task(_cfg(env, gates=[_gate("lint", ["slice"])]), _task(env))
    assert out.status == "rejected"
    retry = _prompts(env, "coder")[1]
    assert "lint" in retry
    assert "E501 line too long" in retry


def test_skipped_tests_fail_a_gate(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD}))
    _gate_plan(env, monkeypatch, tests={"say": "3 passed, 1 skipped in 0.1s"})
    gate = {**_gate("tests", ["slice"]), "forbid_output": True}
    out = pipeline.run_task(_cfg(env, gates=[gate]), _task(env))
    assert out.status == "rejected"
    assert "skipped" in _prompts(env, "coder")[1]


# --- test dispute and follow-up ----------------------------------------------------------


def test_test_dispute_goes_to_the_inbox_and_stops_retrying(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(
        env,
        monkeypatch,
        coder=[{"stdout": "thinking\nTEST-DISPUTE: add must be commutative only\n"}],
    )
    cfg = _cfg(env)
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "disputed"
    assert len(_prompts(env, "coder")) == 1
    inbox = cfg.inbox.read_text()
    assert TASK in inbox
    assert "tier 1" in inbox
    assert "add must be commutative only" in inbox
    assert state.State(cfg).task(TASK)["status"] == "disputed"
    assert out.worktree is not None
    assert out.worktree.exists()


def test_follow_up_is_an_inbox_proposal_never_a_queue_entry(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    step = {"write": {IMPL: GOOD}, "stdout": "FOLLOW-UP: add subtract too\n"}
    _plan(env, monkeypatch, coder=[step])
    cfg = _cfg(env)
    cfg.queue.parent.mkdir(parents=True, exist_ok=True)
    cfg.queue.write_text(f"{TASK} {_spec(env)} fake\n")
    before = cfg.queue.read_text()
    assert cli.main(["run-queue", "--config-json", json.dumps(_overrides(env))]) == 0
    assert cfg.queue.read_text() == before
    inbox = cfg.inbox.read_text()
    assert "proposal" in inbox
    assert "add subtract too" in inbox


def _overrides(env: dict[str, Any], **extra: Any) -> dict[str, Any]:
    cfg = _cfg(env)
    return {
        "repo": str(env["repo"]),
        "state_dir": str(cfg.state_dir),
        "holdout_dir": str(cfg.holdout_dir),
        "worktrees_dir": str(cfg.worktrees_dir),
        "inbox": str(cfg.inbox),
        "queue": str(cfg.queue),
        "python": PY,
        "setup": [],
        "gates": [],
        "default_engine": "fake",
        "qa_engine": "fake",
        "audit_engine": "fake",
        "engines": {
            "fake": {
                "author": "Claude",
                "stdin": True,
                **{r: [PY, str(FAKES), "engine", r] for r in ("qa", "coder", "audit")},
            }
        },
        **extra,
    }


# --- WIP ---------------------------------------------------------------------------------


def test_overlapping_paths_are_refused(env: dict[str, Any]) -> None:
    cfg = _cfg(env)
    st = state.State(cfg)
    st.claim("T-other-001", "w2", ("src/memini/core/",))
    out = pipeline.run_task(cfg, _task(env, paths=(IMPL,), worker="w1"))
    assert out.status == "rejected"
    assert "overlap" in out.reason
    assert "T-other-001" in out.reason


def test_one_task_per_worker(env: dict[str, Any]) -> None:
    cfg = _cfg(env)
    state.State(cfg).claim("T-other-001", "w1", ("src/memini/a/",))
    with pytest.raises(state.WipConflictError, match="worker w1"):
        state.State(cfg).claim("T-two-001", "w1", ("src/memini/b/",))
    state.State(cfg).claim("T-two-001", "w2", ("src/memini/b/",))  # another worker is fine


def test_released_task_frees_its_paths(env: dict[str, Any]) -> None:
    cfg = _cfg(env)
    state.State(cfg).claim("T-other-001", "w1", ("src/memini/",))
    assert cli.main(["release", "T-other-001", "--config-json", json.dumps(_overrides(env))]) == 0
    state.State(cfg).claim("T-two-001", "w1", ("src/memini/",))
    history = state.State(cfg).task("T-other-001")["history"]
    assert [h["status"] for h in history] == ["active", "released"]  # history kept


@pytest.mark.parametrize(
    ("a", "b", "overlap"),
    [
        (("src/memini/",), ("src/memini/x.py",), True),
        (("src/memini/x/",), ("src/memini/y/",), False),
        (("src/memini/x.py",), ("src/memini/x.py",), True),
        (("src/memini/x",), ("src/memini/xy.py",), False),
    ],
)
def test_path_overlap_rule(a: tuple[str, ...], b: tuple[str, ...], overlap: bool) -> None:
    assert state.paths_overlap(a, b) is overlap


# --- merge safety ------------------------------------------------------------------------


def test_main_moved_during_the_run_rebases_and_regates(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD}))
    _gate_plan(env, monkeypatch, ci={"move_main": True})
    repo = env["repo"]
    out = pipeline.run_task(_cfg(env, gates=[_gate("ci", ["merge"])]), _task(env))
    assert out.status == "merged", out.reason
    log = (env["log"] / "gates.log").read_text().splitlines()
    assert len([ln for ln in log if ln.startswith("ci ")]) == 2  # re-gated after the move
    assert _git(repo, "log", "-1", "--format=%s", "main~1") == "concurrent"
    assert _git(repo, "show", "main:OTHER.txt") == "concurrent merge"
    assert _git(repo, "rev-list", "--count", "main~1..main") == "1"


def test_merge_gate_failure_leaves_main_and_the_worktrees(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD}))
    _gate_plan(env, monkeypatch, ci={"fail": True})
    repo = env["repo"]
    before = _git(repo, "rev-parse", "main")
    out = pipeline.run_task(_cfg(env, gates=[_gate("ci", ["merge"])]), _task(env))
    assert out.status == "failed"
    assert "ci" in out.reason
    assert _git(repo, "rev-parse", "main") == before
    assert out.worktree is not None
    assert out.worktree.exists()


def test_main_not_checked_out_anywhere_is_advanced_by_compare_and_swap(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = env["repo"]
    _git(repo, "checkout", "-q", "--detach")
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD}))
    before = _git(repo, "rev-parse", "main")
    out = pipeline.run_task(_cfg(env), _task(env))
    assert out.status == "merged", out.reason
    assert _git(repo, "rev-parse", "main~1") == before


# --- env profiles ------------------------------------------------------------------------

FLAGS = (
    "import os\n\n\ndef _env_bool(name, default=False):\n"
    "    return os.environ.get(name, '') == '1'\n\n\n"
    "A = _env_bool(\"MEMINI_FLAG_A\")\nB = _env_bool('MEMINI_FLAG_B', default=True)\n"
)


def test_flag_names_are_parsed_not_imported(tmp_path: Path) -> None:
    flags = tmp_path / "flags.py"
    flags.write_text(FLAGS + "raise SystemExit('never imported')\n")
    assert gates.flag_names(flags, "_env_bool") == ["MEMINI_FLAG_A", "MEMINI_FLAG_B"]
    assert gates.flag_names(tmp_path / "missing.py", "_env_bool") == []


def test_profiles_default_to_defaults_and_all_flags_on(env: dict[str, Any], tmp_path: Path) -> None:
    cfg = _cfg(env)
    assert [p.name for p in gates.profiles(cfg, env["repo"])] == ["defaults"]  # no flags file
    _write(env["repo"] / "src" / "memini" / "infra" / "flags.py", FLAGS)
    profs = gates.profiles(cfg, env["repo"])
    assert [p.name for p in profs] == ["defaults", "all-flags-on"]
    assert profs[0].env == {}
    assert profs[1].env == {"MEMINI_FLAG_A": "1", "MEMINI_FLAG_B": "1"}


def test_the_slice_must_pass_in_every_profile(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = env["repo"]
    _write(repo / "src" / "memini" / "infra" / "flags.py", FLAGS)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "flags")
    flag_bug = (
        "import os\n\n\ndef add(a, b):\n"
        "    return a + b + (os.environ.get('MEMINI_FLAG_A') == '1')\n"
    )
    _plan(env, monkeypatch, coder=_coder({IMPL: flag_bug}, {IMPL: GOOD}))
    _gate_plan(env, monkeypatch)
    out = pipeline.run_task(_cfg(env, gates=[_gate("tests", ["slice"], profiles=True)]), _task(env))
    assert out.status == "merged", out.reason
    retry = _prompts(env, "coder")[1]
    assert "all-flags-on" in retry
    log = (env["log"] / "gates.log").read_text()
    assert "tests flag=unset" in log
    assert "tests flag=1" in log


# --- audit -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"verdict": "pass", "findings": []}',
        'Here is my audit:\n```json\n{"verdict": "pass", "findings": []}\n```\nDone.',
        'Result {"verdict": "pass", "findings": []} end',
    ],
)
def test_audit_json_is_parsed_robustly(text: str) -> None:
    assert audit.parse(text) == audit.Verdict(passed=True, findings=())


@pytest.mark.parametrize(
    "text",
    [
        "looks fine to me",
        "```json\n{verdict: pass}\n```",
        '{"verdict": "maybe", "findings": []}',
        '{"verdict": "pass", "findings": "none"}',
        '["pass"]',
        "",
    ],
)
def test_audit_garbage_is_a_reject(text: str) -> None:
    verdict = audit.parse(text)
    assert not verdict.passed
    assert verdict.findings


def test_audit_reject_findings_go_into_the_retry(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    reject = '```json\n{"verdict": "reject", "findings": ["hardcoded vocabulary"]}\n```'
    _plan(
        env,
        monkeypatch,
        coder=_coder({IMPL: GOOD}),
        audit=[{"stdout": reject}, {"stdout": "garbage"}],
    )
    out = pipeline.run_task(_cfg(env), _task(env))
    assert out.status == "rejected"
    assert "hardcoded vocabulary" in _prompts(env, "coder")[1]
    assert "audit" in out.reason


# --- run-queue ---------------------------------------------------------------------------


def test_run_queue_runs_in_order_and_stops_at_the_first_blocked(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    for task_id in ("T-b-001", "T-c-001"):
        _holdout(env, task_id)
    sub = {"tests/unit/test_sub.py": QA_TEST.replace("add", "sub")}
    _plan(
        env,
        monkeypatch,
        coder=[{"write": {IMPL: GOOD}}, {"stdout": "TEST-DISPUTE: wrong expectation"}],
        qa=[{"write": {"tests/unit/test_calc.py": QA_TEST}}, {"write": sub}],
    )
    cfg = _cfg(env)
    b = _spec(env, "T-b-001", depends_on=[TASK])
    c = _spec(env, "T-c-001", depends_on=["T-b-001"], paths=["src/memini/core/other/"])
    _write(
        cfg.queue,
        f"# id spec engine paths\n\n{TASK} {_spec(env)} fake {IMPL}\nT-b-001 {b} -\nT-c-001 {c}\n",
    )
    assert cli.main(["run-queue", "--config-json", json.dumps(_overrides(env))]) == 1
    st = state.State(cfg)
    assert st.queue_done() == [TASK]
    assert st.task("T-b-001")["status"] == "disputed"
    assert st.task("T-c-001") is None  # waits on T-b-001: never started
    # a re-run skips what is done and stops at the same blocked task
    assert cli.main(["run-queue", "--config-json", json.dumps(_overrides(env))]) == 1
    assert len(_prompts(env, "qa")) == 2


def test_run_queue_skips_entries_whose_dependencies_are_not_merged(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry that waits is skipped, not a stop; it runs once its dependency merged."""
    sub_test = "from memini.core.sub import sub\n\n\ndef test_sub():\n    assert sub(3, 1) == 2\n"
    _write(env["tmp"] / "holdout" / "T-b-001" / "test_h.py", sub_test.replace("test_sub", "test_h"))
    _plan(
        env,
        monkeypatch,
        qa=[
            {"write": {"tests/unit/test_sub.py": sub_test}},
            {"write": {"tests/unit/test_calc.py": QA_TEST}},
        ],
        coder=[
            {"write": {"src/memini/core/sub/__init__.py": "def sub(a, b):\n    return a - b\n"}},
            {"write": {IMPL: GOOD}},
        ],
    )
    cfg = _cfg(env)
    a = _spec(env, depends_on=["T-b-001"])
    b = _spec(env, "T-b-001", module="sub", paths=["src/memini/core/sub/"])
    _write(cfg.queue, f"{TASK} {a}\nT-b-001 {b}\n")
    assert cli.main(["run-queue", "--config-json", json.dumps(_overrides(env))]) == 0
    assert state.State(cfg).queue_done() == ["T-b-001", TASK]
    assert (
        _git(env["repo"], "log", "-2", "--format=%s", "main").splitlines()[1].startswith("T-b-001")
    )


def test_run_queue_reports_entries_still_waiting(env: dict[str, Any]) -> None:
    cfg = _cfg(env)
    _write(cfg.queue, f"{TASK} {_spec(env, depends_on=['T-zzz-001'])}\n")
    state.State(cfg).claim("T-zzz-001", "w9", ("src/memini/core/zzz/",))
    assert cli.main(["run-queue", "--config-json", json.dumps(_overrides(env))]) == 3


def test_queue_lines_are_parsed(tmp_path: Path) -> None:
    q = tmp_path / "q.txt"
    q.write_text("# c\nT-1 specs/a.md claude src/memini/a/,src/memini/b.py\nT-2 /abs/b.md\n")
    entries = state.read_queue(q)
    assert entries[0] == state.QueueEntry(
        "T-1", tmp_path / "specs/a.md", "claude", ("src/memini/a/", "src/memini/b.py")
    )
    assert entries[1] == state.QueueEntry("T-2", Path("/abs/b.md"), None, ())


def test_queue_and_state_must_be_outside_every_worktree(env: dict[str, Any]) -> None:
    with pytest.raises(config.ConfigError, match="queue"):
        _cfg(env, queue=str(env["repo"] / "queue.txt"))
    with pytest.raises(config.ConfigError, match="state_dir"):
        _cfg(env, state_dir=str(env["tmp"] / "wt" / "state"))


# --- engine command lines and the guard hook ---------------------------------------------


def test_claude_command_grants_only_what_is_needed(env: dict[str, Any]) -> None:
    cfg = config.load_config(env["repo"], overrides={"state_dir": str(env["tmp"] / "s")}, env={})
    argv = engine.command(
        cfg, "claude", "coder", workdir=env["repo"], settings=Path("/s.json"), prompt="p"
    )
    assert argv[:2] == ["claude", "-p"]
    joined = " ".join(argv)
    for flag in ("--settings /s.json", "--permission-mode dontAsk", "--allowedTools"):
        assert flag in joined
    assert "bypassPermissions" not in joined
    assert "Bash(git" not in joined
    audit_argv = engine.command(
        cfg, "claude", "audit", workdir=env["repo"], settings=None, prompt=""
    )
    assert audit_argv[audit_argv.index("--tools") + 1] == ""


def test_generated_settings_deny_protected_paths_and_private_dirs(env: dict[str, Any]) -> None:
    cfg = _cfg(env)
    wt = env["tmp"] / "wt" / TASK
    guard = env["tmp"] / "state" / "g.json"
    settings = engine.settings(cfg, wt, writable=("src/memini/",), guard_config=guard)
    deny = settings["permissions"]["deny"]
    for private in (cfg.holdout_dir, cfg.state_dir, cfg.queue, cfg.inbox):
        assert any(str(private) in rule and rule.startswith("Read(") for rule in deny)
        assert any(str(private) in rule and rule.startswith("Edit(") for rule in deny)
    assert any("src/memini/infra/flags.py" in r for r in deny)
    assert any("tests/" in r for r in deny)
    hook = settings["hooks"]["PreToolUse"][0]
    assert "Write" in hook["matcher"]
    assert "Read" in hook["matcher"]
    assert "guard_worktree.py" in hook["hooks"][0]["command"]


def _guard_cfg(tmp: Path) -> dict[str, Any]:
    return {
        "worktree": str(tmp / "wt"),
        "writable": ["src/memini/"],
        "protected": ["src/memini/infra/flags.py", "tests/", "**/conftest.py"],
        "deny_read": [str(tmp / "holdout"), str(tmp / "state" / "queue.txt")],
    }


@pytest.mark.parametrize(
    ("tool", "rel", "blocked"),
    [
        ("Write", "src/memini/calc.py", False),
        ("Edit", "src/memini/infra/flags.py", True),
        ("MultiEdit", "tests/unit/test_calc.py", True),
        ("Write", "pyproject.toml", True),
        ("Write", "src/memini/conftest.py", True),
        ("Write", "../outside.py", True),
        ("Read", "src/memini/calc.py", False),
    ],
)
def test_guard_hook_decides_on_paths(tmp_path: Path, tool: str, rel: str, blocked: bool) -> None:
    cfg = _guard_cfg(tmp_path)
    event = {"tool_name": tool, "tool_input": {"file_path": rel}, "cwd": cfg["worktree"]}
    assert (guard_worktree.decide(event, cfg) is not None) is blocked


@pytest.mark.parametrize("tool", ["Read", "Grep", "Glob"])
def test_guard_hook_blocks_reading_the_holdout(tmp_path: Path, tool: str) -> None:
    cfg = _guard_cfg(tmp_path)
    key = "file_path" if tool == "Read" else "path"
    event = {"tool_name": tool, "tool_input": {key: str(tmp_path / "holdout" / "T" / "t.py")}}
    assert guard_worktree.decide(event, cfg) is not None


def test_guard_hook_hints_on_bash_naming_a_private_path(tmp_path: Path) -> None:
    cfg = _guard_cfg(tmp_path)
    cat = {"tool_name": "Bash", "tool_input": {"command": f"cat {tmp_path / 'holdout'}/T/t.py"}}
    assert guard_worktree.decide(cat, cfg) is not None
    tests = {"tool_name": "Bash", "tool_input": {"command": "uv run pytest -q"}}
    assert guard_worktree.decide(tests, cfg) is None
    assert guard_worktree.decide({"tool_name": "Write", "tool_input": {}}, cfg) is not None


def test_guard_hook_exits_2_with_a_message(tmp_path: Path) -> None:
    cfg_path = tmp_path / "guard.json"
    cfg_path.write_text(json.dumps(_guard_cfg(tmp_path)))
    hook = ROOT / "tools" / "orchestrator" / "hooks" / "guard_worktree.py"
    event = {
        "tool_name": "Edit",
        "tool_input": {"file_path": "tests/x.py"},
        "cwd": str(tmp_path / "wt"),
    }
    res = subprocess.run(  # noqa: S603 — test-only: our own hook script
        [PY, "-P", str(hook), str(cfg_path)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 2
    assert "protected" in res.stderr
    ok = {
        "tool_name": "Write",
        "tool_input": {"file_path": "src/memini/a.py"},
        "cwd": str(tmp_path / "wt"),
    }
    assert guard_worktree.main([str(cfg_path)], json.dumps(ok)) == 0
    assert guard_worktree.main([str(cfg_path)], "not json") == 2  # fail closed


# --- gates come from ci.yml --------------------------------------------------------------


def test_ci_steps_are_read_from_the_workflow() -> None:
    cfg = config.load_config(ROOT, overrides={"state_dir": "/nonexistent-state"}, env={})
    names = {g.ci_step for g in cfg.gates if g.ci_step}
    for step in (
        "Format",
        "Lint",
        "Types",
        "Tests",
        "Architecture contracts",
        "Export gate",
        "Commit identities",
        "Ledger guards",
        "Commit messages",
    ):
        assert step in names
        assert gates.ci_step_script(ROOT / ".github" / "workflows" / "ci.yml", step)
    tests_gate = next(g for g in cfg.gates if g.ci_step == "Tests")
    assert tests_gate.profiles
    assert tests_gate.forbid_output
    assert "tools/ci/pytest_gate.py" in gates.ci_step_script(
        ROOT / ".github" / "workflows" / "ci.yml", "Tests"
    )


def test_ci_step_with_an_expression_or_unknown_name_is_refused(tmp_path: Path) -> None:
    wf = tmp_path / "ci.yml"
    wf.write_text(
        "jobs:\n  gate:\n    steps:\n      - name: Expr\n        run: echo ${{ github.sha }}\n"
    )
    with pytest.raises(gates.GateConfigError, match="expression"):
        gates.ci_step_script(wf, "Expr")
    with pytest.raises(gates.GateConfigError, match="no step"):
        gates.ci_step_script(wf, "Missing")


# --- the $0 spec check -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "problem"),
    [
        ({"behavior": None}, "required field missing: behavior"),
        ({"size": ""}, "required field empty: size"),
        ({"depends_on": None}, "required field missing: depends_on"),
        ({"paths": ["src/memini/core/calc/", "src/memini/core/other/"]}, "inside one module"),
        ({"paths": ["src/memini/core.py"]}, "inside one module"),
        ({"paths": ["../escape/"]}, "relative inside the repository"),
        ({"depends_on": ["T-unknown-001"]}, "depends_on T-unknown-001 is not a known task"),
        ({"id": "T-other-002"}, "is not the task id"),
        ({"reference": "no-such-branch"}, "reference 'no-such-branch'"),
    ],
)
def test_spec_check_rejects_before_dispatch_with_an_inbox_entry(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, fields: dict[str, Any], problem: str
) -> None:
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD}))
    cfg = _cfg(env)
    path = _spec(env, **fields)
    out = pipeline.run_task(cfg, pipeline.Task(TASK, path, "fake"))
    assert out.status == "rejected"
    assert problem in out.reason
    assert problem in cfg.inbox.read_text()
    assert _prompts(env, "qa") == []
    assert state.State(cfg).task(TASK) is None  # never dispatched


def test_spec_without_a_header_is_rejected(env: dict[str, Any]) -> None:
    cfg = _cfg(env)
    path = env["tmp"] / "specs" / "plain.md"
    path.write_text("# just prose\n")
    out = pipeline.run_task(cfg, pipeline.Task(TASK, path, "fake"))
    assert out.status == "rejected"
    assert "YAML header" in out.reason
    missing = pipeline.run_task(cfg, pipeline.Task(TASK, env["tmp"] / "nope.md", "fake"))
    assert "cannot read spec" in missing.reason


def test_spec_check_refuses_overlap_with_an_active_task(env: dict[str, Any]) -> None:
    cfg = _cfg(env)
    state.State(cfg).claim("T-other-001", "w2", ("src/memini/core/calc/x.py",))
    problems = pipeline.precheck(cfg, spec.load(_spec(env)), TASK, (), {"T-other-001"})
    assert any("overlap the active task T-other-001" in p for p in problems)


def test_check_spec_command(env: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    good = _spec(env)
    bad = _spec(env, "T-bad-001", size=None, depends_on=["T-nope-001"])
    args = ["--config-json", json.dumps(_overrides(env)), "check-spec"]
    assert cli.main([*args, str(good)]) == 0
    assert f"{good}: ok" in capsys.readouterr().out
    assert cli.main([*args, str(good), str(bad)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert f"{bad}: required field missing: size" in lines
    assert f"{bad}: depends_on T-nope-001 is not a known task" in lines
    assert any("no hold-out tests for T-bad-001" in ln for ln in lines)


def test_status_digest(env: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _cfg(env)
    _holdout(env, "T-b-001")
    _write(
        cfg.queue,
        f"{TASK} {_spec(env)}\nT-b-001 {_spec(env, 'T-b-001', depends_on=[TASK])}\n"
        f"T-c-001 {_spec(env, 'T-c-001')}\n",
    )
    st = state.State(cfg)
    st.claim("T-c-001", "w1", ("src/memini/core/c/",))
    st.update("T-c-001", "disputed", reason="the test is wrong")
    st.claim("T-x-001", "w1", ("src/memini/core/x/",))
    state.inbox_append(cfg, "T-c-001", "test dispute", 1, "the test is wrong")
    assert cli.main(["status", "--config-json", json.dumps(_overrides(env))]) == 0
    out = capsys.readouterr().out
    assert f"  {TASK}: ready" in out
    assert f"  T-b-001: waiting on {TASK}" in out
    assert "  T-c-001: disputed (disputed: the test is wrong)" in out
    assert "  T-x-001: active (not in the queue)" in out
    assert "inbox: " in out
    assert "T-c-001: test dispute (tier 1)" in out


def test_help_names_the_config_file(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert "config.toml" in out
    assert "MEMINI_ORCH_CONFIG" in out
    assert "--config" in out


def test_bad_config_is_exit_2(env: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    over = json.dumps(_overrides(env, executor="microvm"))
    assert cli.main(["status", "--config-json", over]) == 2
    assert "executor" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"gates": [{"name": "x", "stages": ["nightly"], "run": ["true"]}]}, "gate"),
        ({"gates": [{"name": "x"}]}, "gate"),
        ({"coder_engine": "nope"}, "unknown engine"),
        ({"allow": "src/"}, "list of strings"),
        ({"engines": {"fake": {"qa": ["x"]}}, "default_engine": "fake"}, "needs qa, coder"),
        ({"worktrees_dir": "inside-the-repo"}, "worktrees_dir"),
    ],
)
def test_config_errors(env: dict[str, Any], extra: dict[str, Any], match: str) -> None:
    with pytest.raises(config.ConfigError, match=match):
        _cfg(env, **extra)


def test_config_file_and_env_paths(env: dict[str, Any], tmp_path: Path) -> None:
    user = tmp_path / "mine.toml"
    user.write_text('max_coder_attempts = 1\n[identity]\nname = "memini-author"\n')
    cfg = config.load_config(
        env["repo"],
        env={"MEMINI_ORCH_CONFIG": str(user), "MEMINI_ORCH_STATE_DIR": str(tmp_path / "st")},
    )
    assert cfg.max_coder_attempts == 1
    assert cfg.identity_email == "memini-author@example.invalid"  # merged, not replaced
    assert cfg.state_dir == (tmp_path / "st").resolve()


# --- reference: coder only, never fetched ------------------------------------------------


def test_reference_reaches_only_the_coder_and_is_never_fetched(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    other = env["tmp"] / "private-repo"
    other.mkdir()
    _git(other, "init", "-q", "-b", "main")
    _git(other, "config", "user.name", "x")
    _git(other, "config", "user.email", "t@example.invalid")
    (other / "old.py").write_text("OLD = 1\n")
    _git(other, "add", "-A")
    _git(other, "commit", "-q", "-m", "old work")
    _git(other, "branch", "old/calc")
    old_sha = _git(other, "rev-parse", "old/calc")
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD}))
    cfg = _cfg(env, reference_repos={"private": str(other)})
    path = _spec(env, reference="private@old/calc")
    out = pipeline.run_task(cfg, pipeline.Task(TASK, path, "fake"))
    assert out.status == "merged", out.reason
    (qa,) = _prompts(env, "qa")
    (audit_prompt,) = _prompts(env, "audit")
    (coder_prompt,) = _prompts(env, "coder")
    for prompt in (qa, audit_prompt):
        assert "old/calc" not in prompt
        assert "reference" not in prompt.split("Specification:")[1]
    assert "old/calc" in coder_prompt
    assert str(other) in coder_prompt
    res = subprocess.run(  # noqa: S603 — test-only, fixed git args
        [GIT, "-C", str(env["repo"]), "cat-file", "-e", old_sha], check=False
    )
    assert res.returncode != 0  # the private object never entered this repository


def test_reference_tools_are_granted_to_the_coder_only(env: dict[str, Any]) -> None:
    cfg = config.load_config(env["repo"], overrides={"state_dir": str(env["tmp"] / "s")}, env={})
    tools = cfg.reference_tools.replace("{repo}", "/r")
    coder = engine.command(
        cfg, "claude", "coder", workdir=env["repo"], settings=None, prompt="", reference_tools=tools
    )
    allowed = coder[coder.index("--allowedTools") + 1]
    assert allowed.endswith(",Bash(git -C /r show *),Bash(git -C /r diff *),Bash(git -C /r log *)")
    qa = engine.command(cfg, "claude", "qa", workdir=env["repo"], settings=None, prompt="")
    assert "git -C" not in " ".join(qa)
    plain = engine.command(cfg, "claude", "coder", workdir=env["repo"], settings=None, prompt="")
    assert "{reference_tools}" not in " ".join(plain)


def test_reference_in_this_repository(env: dict[str, Any]) -> None:
    _git(env["repo"], "branch", "old/calc")
    cfg = _cfg(env)
    assert spec.check(cfg, spec.load(_spec(env, reference="old/calc")), TASK, set()) == []
    assert spec.resolve_reference(cfg, "old/calc") == (cfg.repo, "old/calc")
    assert spec.resolve_reference(cfg, "../x@b")[1] == "b"


# --- untracked and ignored files; gates never see the coder's environment -----------------


@pytest.mark.parametrize(
    ("path", "text"),
    [
        ("conftest.py", "import pytest\npytest.skip('all', allow_module_level=True)\n"),
        ("tests/unit/conftest.py", "def pytest_runtest_call(item):\n    pass\n"),
        ("local/sitecustomize.py", "import os\nos._exit(0)\n"),  # ignored by .gitignore
    ],
)
def test_untracked_or_ignored_cheat_files_are_rejected(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, path: str, text: str
) -> None:
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD, path: text}))
    out = pipeline.run_task(_cfg(env), _task(env))
    assert out.status == "rejected"
    assert path in out.reason or path.split("/", maxsplit=1)[0] + "/" in out.reason


def test_a_package_in_the_coders_venv_is_invisible_to_the_gates(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    site = f".venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    evil = {f"{site}/memini_evil.py": "X = 1\n", IMPL: GOOD}
    _plan(env, monkeypatch, coder=_coder(evil))
    _gate_plan(env, monkeypatch, probe={"must_not_import": "memini_evil"})
    gate = {**_gate("probe", ["slice"]), "run": ["{python}", str(FAKES), "gate", "probe"]}
    cfg = _cfg(env, setup=[[PY, str(FAKES), "venv", "."]], python=".venv/bin/python", gates=[gate])
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "merged", out.reason
    assert not (env["tmp"] / "wt" / TASK).exists()  # removed normally despite its ignored .venv
    log = (env["log"] / "gates.log").read_text()
    assert "probe flag=unset cwd=T-calc-001.gate import=absent" in log


# --- engines per role and the executor boundary --------------------------------------------


def test_qa_and_audit_use_another_engine_when_one_is_installed(env: dict[str, Any]) -> None:
    cfg = config.load_config(env["repo"], overrides={"state_dir": str(env["tmp"] / "s")}, env={})
    have = {"claude": True, "gemini": True, "opencode": False}
    assert engine.pick_engines(cfg, "claude", have) == {
        "coder": "claude",
        "qa": "gemini",
        "audit": "gemini",
    }
    assert engine.pick_engines(cfg, "gemini", have)["qa"] == "claude"
    alone = {"claude": True, "gemini": False, "opencode": False}
    assert engine.pick_engines(cfg, None, alone) == {
        "coder": "claude",
        "qa": "claude",
        "audit": "claude",
    }
    fixed = config.load_config(
        env["repo"],
        overrides={"state_dir": str(env["tmp"] / "s"), "audit_engine": "opencode"},
        env={},
    )
    assert engine.pick_engines(fixed, "claude", have)["audit"] == "opencode"


def test_engines_are_recorded(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD}))
    cfg = _cfg(env)
    assert pipeline.run_task(cfg, _task(env)).status == "merged"
    assert state.State(cfg).task(TASK)["engines"] == "audit=fake,coder=fake,qa=fake"
    assert "engines qa=fake, coder=fake, audit=fake" in _git(
        env["repo"], "show", "main:ledgers/changes.sql"
    )


def test_holdout_and_state_never_enter_the_executor(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class Recording(executor.LocalExecutor):
        def run(self, argv: Any, **kw: Any) -> executor.Result:
            calls.append(
                " ".join(
                    [
                        *argv,
                        str(kw.get("cwd")),
                        kw.get("stdin") or "",
                        *(kw.get("env") or {}).values(),
                    ]
                )
            )
            return super().run(argv, **kw)

    monkeypatch.setattr(pipeline, "executor_for", lambda name: Recording())
    _plan(env, monkeypatch, coder=_coder({IMPL: SNEAKY}, {IMPL: GOOD}))
    out = pipeline.run_task(_cfg(env), _task(env))
    assert out.status == "merged", out.reason
    assert calls
    holdout = str((env["tmp"] / "holdout").resolve())
    for call in calls:
        assert holdout not in call
        assert "SECRET-HOLDOUT-TEXT" not in call
    with pytest.raises(ValueError, match="not available"):
        executor.executor_for("microvm")


def test_executor_reports_a_missing_binary_and_a_timeout(tmp_path: Path) -> None:
    ex = executor.LocalExecutor()
    assert ex.run(["/nonexistent/binary"], cwd=tmp_path).code == 127
    slow = ex.run([PY, "-c", "import time; time.sleep(5)"], cwd=tmp_path, timeout=1)
    assert slow.code == 124
