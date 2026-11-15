"""Every GitHub Action in the workflows is pinned to a full-length commit SHA.

Requirement: owner, 2026-11-15. The public repository's setting refuses any action that is not
pinned to a full commit SHA (a tag such as ``@v4`` can be moved to other code; a SHA cannot).
The first push failed on ``actions/checkout@v4`` and ``astral-sh/setup-uv@v5``.

Tests-Author: Claude/vp-rnd (new test for its own change).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
USES = re.compile(r"^\s*-?\s*uses:\s*(\S+)", re.MULTILINE)
PINNED = re.compile(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}")


def _uses() -> list[str]:
    found: list[str] = []
    for wf in sorted((ROOT / ".github" / "workflows").glob("*.y*ml")):
        found += USES.findall(wf.read_text(encoding="utf-8"))
    return found


def test_the_workflows_use_actions() -> None:
    assert _uses(), "no `uses:` found: the scan would pass vacuously"


def test_every_action_is_pinned_to_a_full_commit_sha() -> None:
    unpinned = [u for u in _uses() if not u.startswith("./") and not PINNED.fullmatch(u)]
    assert not unpinned, unpinned
