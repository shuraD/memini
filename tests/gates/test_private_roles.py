"""Gate: agent/role definitions are not tracked in the code repo (T-076).

Owner decision 2026-11-13: ``.claude/agents/`` and ``.gemini/`` are not tracked in the code
repo. The role definitions live in the private docs repo (``agents/roles/...``) and are
installed at user level (``~/.claude/agents/``), so every session loads them without the code
repo carrying them. AGENTS.md points to those private role definitions.

Fails on main today: ``git ls-files`` lists ``.claude/agents/*.md`` and ``.gemini/settings.json``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GIT = shutil.which("git") or "git"

# Paths whose contents are private role/agent definitions and must never be tracked here.
UNTRACKED_PREFIXES = (".claude/agents/", ".gemini/")


def _tracked_files() -> list[str]:
    """Every file git tracks, with all GIT_* variables stripped from the subprocess env."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    result = subprocess.run(  # noqa: S603 — test-only, fixed git args
        [GIT, "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return [name for name in result.stdout.split("\0") if name]


@pytest.mark.parametrize("prefix", UNTRACKED_PREFIXES)
def test_private_role_files_are_not_tracked(prefix: str) -> None:
    """No file under a private role/agent path is tracked in the code repo."""
    tracked = _tracked_files()
    offenders = [name for name in tracked if name == prefix.rstrip("/") or name.startswith(prefix)]
    assert not offenders, f"private role files are tracked under {prefix!r}: {offenders}"


def test_agents_md_points_to_the_private_role_definitions() -> None:
    """AGENTS.md points to the private role definitions (agents/roles/ in the private docs repo)."""
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert re.search(r"agents/roles", text), (
        "AGENTS.md must point to the private role definitions (agents/roles/ in the private "
        "docs repo), so readers know where the roles live now that they are not tracked here"
    )
