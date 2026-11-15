"""The plan stage (owner 2026-11-16): plan around the critical path, agree interfaces first.

A planner agent (a fake script here, no model) turns one module line into the module's slices;
the orchestrator refuses a plan with bad ids, unknown dependencies, a cycle, no contract slice
or too many waves, and hands an accepted plan's slices to the spec stage.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from tools.orchestrator import config, planstage, specstage, stages

FAKES = Path(__file__).with_name("orchestrator_fakes.py")
PY = sys.executable
GIT = shutil.which("git") or "git"
ROOT_MOD = "src/memini/core/kg/"
GOOD = (
    "EX-01 src/memini/core/kg/types.py - Define kg types, protocols, errors and a fake\n"
    "EX-02 src/memini/core/kg/plan.py EX-01 Plan the extraction of one document (decision 2)\n"
    "EX-03 src/memini/core/kg/ground.py EX-01 Ground every item to unit ids (decision 4)\n"
    "EX-04 src/memini/core/kg/run.py EX-02,EX-03 Run plan then grounding for one document\n"
)


def _slices(text: str) -> list[planstage.Slice]:
    return planstage.parse_plan(text)[0]


def test_a_contract_first_plan_is_accepted_and_its_waves_counted() -> None:
    assert planstage.plan_problems(_slices(GOOD), "EX", ROOT_MOD, set(), 4) == []
    assert planstage.waves(_slices(GOOD)) == 3


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        (
            GOOD.replace(
                "EX-04 src/memini/core/kg/run.py EX-02,EX-03",
                "EX-04 src/memini/core/kg/run.py EX-02,EX-04",
            ),
            "depends on itself",
        ),
        (
            GOOD
            + "EX-05 src/memini/core/kg/a.py EX-06 Task a long enough\n"
            + "EX-06 src/memini/core/kg/b.py EX-05 Task b long enough\n",
            "cycle",
        ),
        (GOOD.replace("EX-01 src", "X-1 src"), "the id must be EX-NN"),
        (GOOD.replace("EX-02,EX-03", "EX-02,ZZ-99"), "unknown task ZZ-99"),
        (GOOD.replace("src/memini/core/kg/plan.py", "src/memini/infra/other.py"), "is not under"),
        (
            GOOD.replace(
                "EX-01 src/memini/core/kg/types.py -", "EX-01 src/memini/core/kg/types.py EX-04"
            ),
            "cycle",
        ),
        (GOOD + GOOD.splitlines()[0] + "\n", "used twice"),
    ],
)
def test_a_bad_plan_is_refused(text: str, problem: str) -> None:
    problems = planstage.plan_problems(_slices(text), "EX", ROOT_MOD, set(), 4)
    assert any(problem in p for p in problems), problems


def test_a_plan_without_a_contract_slice_or_with_a_long_chain_is_refused() -> None:
    deps = ["-"] + [f"EX-{n:02d}" for n in range(1, 6)]
    chain = "".join(
        f"EX-{n:02d} src/memini/core/kg/m{n}.py {deps[n - 1]} Step {n} of it\n" for n in range(1, 7)
    )
    problems = planstage.plan_problems(_slices(chain), "EX", ROOT_MOD, set(), 4)
    assert any("6 waves" in p and "critical path" in p for p in problems), problems
    assert any(
        "already a task" in p
        for p in planstage.plan_problems(_slices(GOOD), "EX", ROOT_MOD, {"EX-02"}, 4)
    )


# ------------------------------------------------------------------------------ end to end
def _git(repo: Path, *args: str) -> None:
    subprocess.run([GIT, "-C", str(repo), *args], check=True, capture_output=True)  # noqa: S603


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    repo = tmp_path / "repo"
    (repo / "src" / "memini").mkdir(parents=True)
    (repo / "src" / "memini" / "__init__.py").write_text("")
    (repo / ".gitignore").write_text("__pycache__/\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "memini-author")
    _git(repo, "config", "user.email", "memini-author@example.invalid")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    source = tmp_path / "docs" / "SPEC-kg.md"
    source.parent.mkdir()
    source.write_text("# kg\n")
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "log"))
    fake = {role: [PY, str(FAKES), "engine", role] for role in ("qa", "coder", "audit")}
    cfg = config.load_config(
        repo,
        overrides={
            "state_dir": str(tmp_path / "state"),
            "holdout_dir": str(tmp_path / "holdout"),
            "worktrees_dir": str(tmp_path / "wt"),
            "inbox": str(tmp_path / "inbox" / "OWNER_INBOX.md"),
            "queue": str(tmp_path / "queue" / "queue.txt"),
            "spec_dir": str(tmp_path / "slices"),
            "stages": [
                {
                    "name": "plan",
                    "kind": "plan",
                    "queue": str(tmp_path / "queue" / "q-plan.txt"),
                    "next": "spec",
                },
                {"name": "spec", "queue": str(tmp_path / "queue" / "q-spec.txt"), "next": "code"},
            ],
            "python": PY,
            "setup": [],
            "gates": [],
            "default_engine": "fake",
            "qa_engine": "fake",
            "audit_engine": "fake",
            "engines": {"fake": {"author": "Claude", "stdin": True, **fake}},
            "env_allow": ["PATH", "HOME", "TMPDIR", "FAKE_LOG", "FAKE_PLAN"],
            "executor": "local",
        },
        env={},
    )
    cfg.stages[0].queue.parent.mkdir(parents=True, exist_ok=True)
    cfg.stages[0].queue.write_text(f"EX {ROOT_MOD} - {source} Slice the kg module\n")
    return {"tmp": tmp_path, "cfg": cfg, "source": source, "mp": monkeypatch}


def _plan_steps(env: dict[str, Any], *texts: str) -> None:
    path = env["tmp"] / "plan.json"
    path.write_text(json.dumps({"qa": [{"write": {"PLAN.txt": t}} for t in texts]}))
    env["mp"].setenv("FAKE_PLAN", str(path))


def test_an_accepted_plan_is_kept_and_its_slices_go_to_the_spec_queue(env: dict[str, Any]) -> None:
    cfg = env["cfg"]
    _plan_steps(env, GOOD)
    assert specstage.run(cfg, cfg.stages[0], "EX", planstage.PlanJob) == 0
    assert (cfg.spec_dir / "kg" / "PLAN-EX.txt").read_text() == GOOD
    items = stages.read_stage_queue(cfg.stages[1].queue)
    assert [(i.task_id, i.depends_on) for i in items] == [
        ("EX-01", ()),
        ("EX-02", ("EX-01",)),
        ("EX-03", ("EX-01",)),
        ("EX-04", ("EX-02", "EX-03")),
    ]
    assert all(i.sources == (env["source"],) for i in items)
    prompt = (env["tmp"] / "log" / "qa-1.txt").read_text()
    assert "Task id prefix: EX" in prompt
    assert "at most 4" in prompt


def test_a_refused_plan_gets_one_retry_with_its_problems(env: dict[str, Any]) -> None:
    cfg = env["cfg"]
    cyclic = GOOD.replace(
        "EX-01 src/memini/core/kg/types.py -", "EX-01 src/memini/core/kg/types.py EX-04"
    )
    _plan_steps(env, cyclic, GOOD)
    assert specstage.run(cfg, cfg.stages[0], "EX", planstage.PlanJob) == 0
    assert "cycle" in (env["tmp"] / "log" / "qa-2.txt").read_text()


def test_a_plan_refused_twice_is_rejected_and_nothing_is_handed_on(env: dict[str, Any]) -> None:
    cfg = env["cfg"]
    _plan_steps(env, "EX-01 src/memini/core/kg/a.py EX-01 Depends on itself forever\n")
    assert specstage.run(cfg, cfg.stages[0], "EX", planstage.PlanJob) == 1
    assert stages.StageState(cfg, "plan").status("EX") == specstage.REJECTED
    assert not cfg.stages[1].queue.exists()
    assert not (cfg.spec_dir / "kg" / "PLAN-EX.txt").exists()


def test_an_existing_plan_is_never_overwritten(env: dict[str, Any]) -> None:
    cfg = env["cfg"]
    kept = cfg.spec_dir / "kg" / "PLAN-EX.txt"
    kept.parent.mkdir(parents=True)
    kept.write_text("the owner's plan\n")
    _plan_steps(env, GOOD)
    assert specstage.run(cfg, cfg.stages[0], "EX", planstage.PlanJob) == 1
    assert kept.read_text() == "the owner's plan\n"


def test_a_stage_kind_must_be_known(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    bad = [{"name": "plan", "kind": "magic", "queue": str(tmp_path / "q"), "next": "code"}]
    with pytest.raises(config.ConfigError, match="kind must be one of"):
        config.load_config(
            root, overrides={"state_dir": "/nonexistent-state", "stages": bad}, env={}
        )
