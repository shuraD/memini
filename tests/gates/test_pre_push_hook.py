"""The pre-push hook runs the same test step as CI, so a red CI is found before the push."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "uv run python -P "


def _ci_test_command() -> str:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = [s for job in workflow["jobs"].values() for s in job["steps"]]
    runs = [s["run"].strip() for s in steps if "pytest_gate.py" in s.get("run", "")]
    assert len(runs) == 1
    assert runs[0].startswith(PREFIX)
    return runs[0].removeprefix(PREFIX)


def test_the_pre_push_hook_runs_the_ci_test_step() -> None:
    hook = (ROOT / ".githooks/pre-push").read_text(encoding="utf-8")
    assert _ci_test_command() in hook


def test_the_pre_push_hook_is_executable() -> None:
    assert (ROOT / ".githooks/pre-push").stat().st_mode & 0o111
