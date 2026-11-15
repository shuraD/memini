"""The stage chain (owner 2026-11-15): stage queues before ``code``, the spec stage, hand-off.

"we can have more than 2 queues"; "it can run both queues in parallel when the QA agent
prepares the tests for the future slices". The spec stage runs a fake engine here (no model):
it writes SPEC.md and hidden tests; the orchestrator checks, seals and hands them to ``code``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from tools.orchestrator import cli, config, engine, specstage, stages, state

ROOT = Path(__file__).resolve().parents[2]
FAKES = Path(__file__).with_name("orchestrator_fakes.py")
PY = sys.executable
GIT = shutil.which("git") or "git"
ID = "T-calc-002"
MODULE = "src/memini/core/calc/"
FAILING = "from memini.core.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
PASSING = "def test_nothing():\n    assert True\n"


def _git(repo: Path, *args: str) -> None:
    subprocess.run([GIT, "-C", str(repo), *args], check=True, capture_output=True)  # noqa: S603


def _spec_md(task_id: str = ID, **fields: Any) -> str:
    header: dict[str, Any] = {
        "id": task_id,
        "module": MODULE,
        "paths": [MODULE],
        "depends_on": [],
        "flag": "",
        "behavior": "add(a, b) returns the sum",
        "interface": "memini.core.calc.add(a: int, b: int) -> int",
        "acceptance": "- Given 1 and 2, Then add returns 3.",
        "out_of_scope": "subtraction",
        "size": "S",
        **fields,
    }
    return "---\n" + yaml.safe_dump(header, sort_keys=False) + "---\nFrom the source doc.\n"


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    repo = tmp_path / "repo"
    (repo / "src" / "memini" / "core").mkdir(parents=True)
    (repo / "src" / "memini" / "__init__.py").write_text("")
    (repo / "src" / "memini" / "core" / "__init__.py").write_text("")
    (repo / "tests" / "unit").mkdir(parents=True)
    (repo / "tests" / "unit" / "test_existing.py").write_text("def test_ok():\n    pass\n")
    (repo / "pyproject.toml").write_text('[tool.pytest.ini_options]\npythonpath = ["src"]\n')
    (repo / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n.ruff_cache/\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "memini-author")
    _git(repo, "config", "user.email", "memini-author@example.invalid")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    source = tmp_path / "docs" / "SPEC-calc.md"
    source.parent.mkdir()
    source.write_text("# calc\nadd two numbers\n")
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "log"))
    return {"tmp": tmp_path, "repo": repo, "source": source}


def _cfg(env: dict[str, Any], **extra: Any) -> config.Config:
    tmp = env["tmp"]
    fake = {role: [PY, str(FAKES), "engine", role] for role in ("qa", "coder", "audit")}
    over: dict[str, Any] = {
        "state_dir": str(tmp / "state"),
        "holdout_dir": str(tmp / "holdout"),
        "worktrees_dir": str(tmp / "wt"),
        "inbox": str(tmp / "inbox" / "OWNER_INBOX.md"),
        "queue": str(tmp / "queue" / "queue.txt"),
        "spec_dir": str(tmp / "slices"),
        "stages": [
            {"name": "spec", "queue": str(tmp / "queue" / "queue-spec.txt"), "next": "code"}
        ],
        "python": PY,
        "setup": [],
        "gates": [],  # no real gate tools in these tests (CI has no gitleaks)
        "default_engine": "fake",
        "qa_engine": "fake",
        "audit_engine": "fake",
        "engines": {"fake": {"author": "Claude", "stdin": True, **fake}},
        "env_allow": ["PATH", "HOME", "TMPDIR", "FAKE_LOG", "FAKE_PLAN"],
        "executor": "local",
    }
    over.update(extra)
    return config.load_config(env["repo"], overrides=over, env={})


def _line(env: dict[str, Any], task_id: str = ID, deps: str = "-") -> str:
    return f"{task_id} {MODULE} {deps} {env['source']} Add two numbers\n"


def _enqueue(cfg: config.Config, *lines: str) -> None:
    q = cfg.stages[0].queue
    q.parent.mkdir(parents=True, exist_ok=True)
    with q.open("a") as fh:
        fh.writelines(lines)


def _plan(env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, *steps: dict[str, str]) -> None:
    path = env["tmp"] / "plan.json"
    path.write_text(json.dumps({"qa": [{"write": s} for s in steps]}))
    monkeypatch.setenv("FAKE_PLAN", str(path))


# ------------------------------------------------------------------------------ config
def test_the_shipped_config_chains_spec_into_code() -> None:
    cfg = config.load_config(ROOT, overrides={"state_dir": "/nonexistent-state"}, env={})
    assert [(s.name, s.next, s.prompt) for s in cfg.stages] == [("spec", "code", "spec")]
    assert (cfg.prompts_dir / "spec.txt").is_file()


@pytest.mark.parametrize(
    ("stages_cfg", "problem"),
    [
        ([{"name": "spec", "queue": "/tmp/q1", "next": "nowhere"}], "next must be a later"),  # noqa: S108
        (
            [
                {"name": "a", "queue": "/tmp/qa", "next": "b"},  # noqa: S108
                {"name": "b", "queue": "/tmp/qb", "next": "a"},  # noqa: S108 — a loop
            ],
            "next must be a later",
        ),
        ([{"name": "code", "queue": "/tmp/q", "next": "code"}], "must be lowercase"),  # noqa: S108
        ([{"name": "spec", "queue": "q.txt", "next": "code"}], "outside the repository"),
        ([{"name": "spec", "next": "code"}], "must be set"),
    ],
)
def test_a_bad_stage_chain_is_refused(stages_cfg: list[dict[str, str]], problem: str) -> None:
    with pytest.raises(config.ConfigError, match=problem):
        config.load_config(
            ROOT, overrides={"state_dir": "/nonexistent-state", "stages": stages_cfg}, env={}
        )


def test_a_stage_queue_line_names_module_deps_sources_and_task(tmp_path: Path) -> None:
    q = tmp_path / "q.txt"
    q.write_text("# comment\nGS-02 src/memini/core/g/ GS-01,LP-02 a.md,/abs/b.md Load the file\n")
    (item,) = stages.read_stage_queue(q)
    assert item == stages.StageItem(
        "GS-02",
        "src/memini/core/g/",
        ("GS-01", "LP-02"),
        (tmp_path / "a.md", Path("/abs/b.md")),
        "Load the file",
    )
    q.write_text("GS-03 src/memini/core/g/ - a.md\n")
    with pytest.raises(ValueError, match="expected"):
        stages.read_stage_queue(q)


def test_no_agent_may_read_a_stage_queue(env: dict[str, Any]) -> None:
    cfg = _cfg(env)
    rules = engine.settings(cfg, env["repo"], writable=[], guard_config=env["tmp"] / "g.json")
    assert f"Read(/{cfg.stages[0].queue})" in rules["permissions"]["deny"]


# ------------------------------------------------------------------------------ spec stage
def test_the_spec_stage_seals_the_tests_writes_the_spec_and_hands_the_task_to_code(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(env)
    _enqueue(cfg, _line(env))
    _plan(env, monkeypatch, {"SPEC.md": _spec_md(), "tests/unit/test_calc_hidden.py": FAILING})
    assert specstage.run(cfg, cfg.stages[0], ID) == 0
    assert (cfg.holdout_dir / ID / "test_calc_hidden.py").read_text() == FAILING
    spec_file = cfg.spec_dir / "calc" / f"{ID}.md"
    assert spec_file.read_text() == _spec_md()
    assert [(e.task_id, e.spec) for e in state.read_queue(cfg.queue)] == [(ID, spec_file)]
    assert stages.StageState(cfg, "spec").status(ID) == stages.DONE
    assert not (cfg.worktrees_dir / f"{ID}.spec").exists()  # outputs copied, workspace gone
    prompt = (env["tmp"] / "log" / "qa-1.txt").read_text()
    assert "Add two numbers" in prompt
    assert ".sources/01-SPEC-calc.md" in prompt  # the source is inside the agent's checkout
    runs = cfg.state_dir / "runs" / f"{ID}.spec"
    assert (runs / "costs.tsv").read_text().startswith("01\tspec\tfake\t")


def test_tests_that_pass_without_code_are_rejected_after_one_retry(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(env)
    _enqueue(cfg, _line(env))
    _plan(env, monkeypatch, {"SPEC.md": _spec_md(), "tests/unit/test_calc_hidden.py": PASSING})
    assert specstage.run(cfg, cfg.stages[0], ID) == 1
    second = (env["tmp"] / "log" / "qa-2.txt").read_text()
    assert "prove nothing" in second  # the retry was told why
    assert stages.StageState(cfg, "spec").status(ID) == specstage.REJECTED
    assert "spec stage rejected" in cfg.inbox.read_text()
    assert not (cfg.holdout_dir / ID).exists()
    assert not cfg.queue.exists()
    assert (cfg.worktrees_dir / f"{ID}.spec").exists()  # kept for inspection


@pytest.mark.parametrize(
    ("files", "problem"),
    [
        ({"tests/unit/test_c.py": FAILING, "src/memini/core/calc.py": "x = 1\n"}, "not allowed"),
        ({"tests/unit/test_c.py": FAILING}, "no SPEC.md written"),
        ({"SPEC.md": _spec_md()}, "no hidden test file"),
        ({"SPEC.md": _spec_md(module="src/memini/core/other/")}, "is not the task's module"),
        ({"SPEC.md": _spec_md(id="T-other"), "tests/unit/test_c.py": FAILING}, "not the task id"),
        ({"SPEC.md": _spec_md(), "tests/unit/test_c.py": "import os\n"}, "no test function"),
    ],
)
def test_wrong_output_is_rejected_with_its_reason(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, files: dict[str, str], problem: str
) -> None:
    cfg = _cfg(env)
    _enqueue(cfg, _line(env))
    _plan(env, monkeypatch, files)
    assert specstage.run(cfg, cfg.stages[0], ID) == 1
    assert problem in stages.StageState(cfg, "spec").records()[ID]["reason"]


def test_the_spec_must_keep_the_task_lines_dependencies(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(env)
    _enqueue(cfg, _line(env, "T-base"), _line(env, deps="T-base"))
    _plan(env, monkeypatch, {"SPEC.md": _spec_md(), "tests/unit/test_c.py": FAILING})
    assert specstage.run(cfg, cfg.stages[0], ID) == 1
    assert "lost T-base" in stages.StageState(cfg, "spec").records()[ID]["reason"]


def test_an_existing_spec_is_never_overwritten(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(env)
    _enqueue(cfg, _line(env))
    existing = cfg.spec_dir / "calc" / f"{ID}.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("the owner's spec\n")
    _plan(env, monkeypatch, {"SPEC.md": _spec_md(), "tests/unit/test_c.py": FAILING})
    assert specstage.run(cfg, cfg.stages[0], ID) == 1
    assert existing.read_text() == "the owner's spec\n"
    assert not (env["tmp"] / "log").exists()  # stopped before any agent ran


def test_a_rejected_task_is_released_for_a_new_try(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(env)
    _enqueue(cfg, _line(env))
    _plan(env, monkeypatch, {"SPEC.md": _spec_md()})
    assert specstage.run(cfg, cfg.stages[0], ID) == 1
    assert specstage.run(cfg, cfg.stages[0], ID) == 1  # not retried while rejected
    monkeypatch.chdir(env["repo"])
    assert cli.main(["release", ID, "--stage", "spec", "--config-json", _json(env)]) == 0
    assert stages.StageState(cfg, "spec").status(ID) == stages.RELEASED


def _json(env: dict[str, Any]) -> str:
    tmp = env["tmp"]
    return json.dumps(
        {
            "state_dir": str(tmp / "state"),
            "holdout_dir": str(tmp / "holdout"),
            "worktrees_dir": str(tmp / "wt"),
            "inbox": str(tmp / "inbox" / "OWNER_INBOX.md"),
            "queue": str(tmp / "queue" / "queue.txt"),
            "stages": [
                {"name": "spec", "queue": str(tmp / "queue" / "queue-spec.txt"), "next": "code"}
            ],
        }
    )


# ------------------------------------------------------------------------------ the loops
class _Proc:
    def __init__(self, on_done: Any, code: int = 0) -> None:
        self.on_done, self.code, self.polls = on_done, code, 1

    def poll(self) -> int | None:
        self.polls -= 1
        if self.polls >= 0:
            return None
        self.on_done()
        return self.code


def test_a_stage_runs_dependents_after_their_dependencies_and_holds_back_failures(
    env: dict[str, Any],
) -> None:
    cfg = _cfg(env)
    rec = stages.StageState(cfg, "spec")
    _enqueue(cfg, _line(env, "A"), _line(env, "B", "A"), _line(env, "X"), _line(env, "Y", "X"))
    started: list[str] = []

    def launch(item: stages.StageItem) -> _Proc:
        started.append(item.task_id)
        if item.task_id == "X":
            return _Proc(lambda: rec.set("X", specstage.REJECTED, "bad"), code=1)
        return _Proc(lambda: rec.set(item.task_id, stages.DONE))

    code = stages.run_stage(cfg, "spec", workers=4, launch=launch, pause=0.0)
    assert started == ["A", "X", "B"]  # B waited for A; Y never started (X failed)
    assert code == 1
    assert "Y: waiting on X" in "\n".join(stages.status_lines(cfg))


def test_a_stage_picks_up_lines_added_while_it_runs(env: dict[str, Any]) -> None:
    cfg = _cfg(env)
    rec = stages.StageState(cfg, "spec")
    _enqueue(cfg, _line(env, "A"))
    started: list[str] = []

    def launch(item: stages.StageItem) -> _Proc:
        started.append(item.task_id)
        if item.task_id == "A":
            _enqueue(cfg, _line(env, "B"))  # vp-rnd adds a task during the run
        return _Proc(lambda: rec.set(item.task_id, stages.DONE))

    assert stages.run_stage(cfg, "spec", workers=1, launch=launch, pause=0.0) == 0
    assert started == ["A", "B"]


def test_only_one_loop_per_stage(env: dict[str, Any]) -> None:
    cfg = _cfg(env)
    with stages.stage_active(cfg, "spec"), pytest.raises(state.BusyError, match="already"):
        stages.run_stage(cfg, "spec", workers=1, launch=lambda i: None, pause=0.0)


def test_the_code_loop_waits_for_an_active_stage_and_runs_what_it_hands_over(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(env, holdout_required=False)
    cfg.queue.parent.mkdir(parents=True, exist_ok=True)
    cfg.queue.write_text("")
    st = state.State(cfg)
    started: list[str] = []
    stack: list[Any] = []

    def launch(entry: state.QueueEntry) -> _Proc:
        started.append(entry.task_id)
        return _Proc(lambda: st.mark_done(entry.task_id))

    ticks = iter(range(100))

    def tick(_: float) -> None:
        n = next(ticks)
        if n == 1:  # the spec stage hands a task over while code waits
            spec_file = env["tmp"] / "slices" / f"{ID}.md"
            spec_file.parent.mkdir(parents=True, exist_ok=True)
            spec_file.write_text(_spec_md())
            stages.handoff(cfg, cfg.stages[0], f"{ID} {spec_file}")
        if n == 3:  # the spec stage loop ends
            stack.pop().close()

    monkeypatch.setattr(time, "sleep", tick)
    active = stages.stage_active(cfg, "spec")
    active.__enter__()

    class _Close:
        def close(self) -> None:
            active.__exit__(None, None, None)

    stack.append(_Close())
    assert cli.run_queue_parallel(cfg, workers=2, launch=launch, pause=0.0) == 0
    assert started == [ID]


def test_handoff_adds_a_task_once(env: dict[str, Any]) -> None:
    cfg = _cfg(env)
    stages.handoff(cfg, cfg.stages[0], "A /x/A.md")
    stages.handoff(cfg, cfg.stages[0], "A /x/A.md")
    stages.handoff(cfg, cfg.stages[0], "B /x/B.md")
    assert cfg.queue.read_text() == "A /x/A.md\nB /x/B.md\n"


def test_the_spec_stage_scans_its_hidden_tests_with_the_merge_gate_checks(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    scan = [PY, "-c", "import sys; sys.exit('FAKE-SECRET-MARKER' in open(sys.argv[1]).read())"]
    gate = {"name": "leaks", "run": ["true"], "stages": ["merge"]}
    cfg = _cfg(env, gates=[gate], qa_gate_checks={"leaks": [*scan, "{file}"]})
    _enqueue(cfg, _line(env))
    secret = FAILING + "\n\nKEY = 'FAKE-SECRET-MARKER'\n"
    _plan(
        env,
        monkeypatch,
        {"SPEC.md": _spec_md(), "tests/unit/test_calc_hidden.py": secret},
        {"tests/unit/test_calc_hidden.py": FAILING},
    )
    assert specstage.run(cfg, cfg.stages[0], ID) == 0
    assert "exit 1" in (env["tmp"] / "log" / "qa-2.txt").read_text()
    assert (cfg.holdout_dir / ID / "test_calc_hidden.py").read_text() == FAILING
