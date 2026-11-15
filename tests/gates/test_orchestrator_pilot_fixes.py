"""Fixes found by the second LP-01 pilot run (2026-11-15).

- The hold-out ran once per env profile but copied its tests into the slice export each
  time: the second profile crashed with FileExistsError.
- The coder prompt told the agent to run ``uv run pytest``, which the coder's allow-list
  refuses; the QA prompt named no command. Both agents worked without running anything.
- The agent CLIs answered in plain text, so the run recorded no cost; the pilot exists to
  measure it. The Claude engine now answers in JSON; the orchestrator keeps the result text
  for every later step and records the cost per step.

Tests-Author: Claude/vp-rnd (new tests for its own change).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from tools.orchestrator import config, engine, gates
from tools.orchestrator.executor import Result

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "tools" / "orchestrator" / "prompts"
SANDBOX_PYTEST = "uv run python -m tools.orchestrator sandbox-pytest"
GIT = shutil.which("git") or "git"


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, str]]] = []

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
        assert (cwd / "holdout" / "test_h.py").is_file()
        self.calls.append((list(argv), dict(env or {})))
        return Result(0, "1 passed", "")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603 — test-only, fixed git args
        [GIT, *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_the_sandboxed_hold_out_runs_in_every_profile(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("x = 1\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "i")
    holdout = tmp_path / "holdout" / "T-1"
    holdout.mkdir(parents=True)
    (holdout / "test_h.py").write_text("def test_h():\n    pass\n")
    over = {
        "state_dir": str(tmp_path / "state"),
        "holdout_dir": str(tmp_path / "holdout"),
        "profiles": {"one": {"env": {}}, "two": {"env": {"MEMINI_X": "1"}}},
    }
    cfg = config.load_config(ROOT, overrides=over, env={})
    sandbox = _Recorder()
    ok = gates.holdout_passes(cfg, repo, "HEAD", "T-1", tmp_path / "log", sandbox=sandbox, env={})
    assert ok
    assert len(sandbox.calls) == 2  # one run per profile, no crash on the second
    assert sandbox.calls[1][1].get("MEMINI_X") == "1"


def test_the_prompts_name_only_commands_the_agents_may_run() -> None:
    coder = (PROMPTS / "coder.txt").read_text(encoding="utf-8")
    qa = (PROMPTS / "qa.txt").read_text(encoding="utf-8")
    assert "uv run pytest" not in coder, "the coder's allow-list refuses uv run pytest"
    assert SANDBOX_PYTEST in coder and SANDBOX_PYTEST in qa
    assert "uv run ruff check" in coder and "uv run mypy" in coder
    cfg = config.load_config(ROOT, overrides={"state_dir": "/tmp/x-state"}, env={})  # noqa: S108
    allowed = " ".join(cfg.engines["claude"].roles["coder"])
    for command in (SANDBOX_PYTEST, "uv run ruff", "uv run mypy"):
        assert f"Bash({command} *)" in allowed


def test_the_claude_engine_answers_in_json() -> None:
    cfg = config.load_config(ROOT, overrides={"state_dir": "/tmp/x-state"}, env={})  # noqa: S108
    for role, argv in cfg.engines["claude"].roles.items():
        i = argv.index("--output-format")
        assert argv[i + 1] == "json", role


def test_agent_output_keeps_the_text_and_reads_the_cost() -> None:
    raw = json.dumps({"type": "result", "result": "done\nFOLLOW-UP: x", "total_cost_usd": 0.42})
    assert engine.agent_output(raw) == ("done\nFOLLOW-UP: x", 0.42)
    assert engine.agent_output("plain text answer") == ("plain text answer", None)
    assert engine.agent_output('{"no": "result"}') == ('{"no": "result"}', None)
    assert engine.agent_output("[1, 2]") == ("[1, 2]", None)
