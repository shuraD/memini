"""The vp-rnd session-start report (FLOW-HARD.md D13): state file, decisions, status, facts."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from tests.gates.test_orchestrator import _git
from tools.orchestrator import config, state, vp_start

ROOT = Path(__file__).resolve().parents[2]
DECISIONS = """# Decisions

## Open

**D-1. Where the dimension comes from.** text
Owner:

**D-2. Push policy.** text
Owner: yes, push after every merge

**D-3. Who approves a batch.**
Owner:

## Answered
**D-0. Old.**
Owner:
"""


@pytest.fixture
def cfg(tmp_path: Path) -> config.Config:
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)  # noqa: S603, S607
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "x")
    _git(repo, "config", "user.email", "x@example.invalid")
    (repo / "a.txt").write_text("a")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "first")
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-q", "origin", "main")
    (repo / "b.txt").write_text("b")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "GR-10: second, not pushed")
    squads = tmp_path / "docs" / "squads"
    squads.mkdir(parents=True)
    (tmp_path / "docs" / "DECISIONS.md").write_text(DECISIONS)
    (tmp_path / "docs" / "STATUS.md").write_text("# Status\nall quiet\n")
    queue = tmp_path / "queue.txt"
    queue.write_text("GR-10 specs/GR-10.md\nGR-11 specs/GR-11.md\n")
    cfg = config.load_config(
        repo,
        overrides={
            "cards_dir": str(squads),
            "state_dir": str(tmp_path / "state"),
            "queue": str(queue),
        },
    )
    st = state.State(cfg)
    st.claim("GR-10", "w", ())
    st.update("GR-10", "rejected", reason="x")
    st.mark_done("GR-11")
    return cfg


def _report(cfg: config.Config, tmp_path: Path, text: str) -> str:
    file = tmp_path / "vp-state.md"
    file.write_text(text)
    return vp_start.render(cfg, file)


def test_open_decisions_are_the_ones_without_an_answer() -> None:
    assert vp_start.open_decisions(DECISIONS) == [
        "**D-1. Where the dimension comes from.** text",
        "**D-3. Who approves a batch.**",
    ]


def test_the_report_has_the_state_file_decisions_status_and_facts(
    cfg: config.Config, tmp_path: Path
) -> None:
    out = _report(cfg, tmp_path, "vp-rnd notes: wait for QA\n")
    assert "vp-rnd notes: wait for QA" in out
    assert "**D-1." in out and "**D-2." not in out
    assert "all quiet" in out
    head = _git(cfg.repo, "rev-parse", "--short", "main")
    assert f"main {head}; reflog:" in out
    assert "not merged: GR-10 rejected" in out
    assert "active queue lines: GR-10 (queue.txt)" in out  # GR-11 is done
    assert "not pushed: 1" in out and "GR-10: second, not pushed" in out
    assert "Mismatch" not in out


def test_a_state_file_that_disagrees_gets_a_mismatch_line_per_claim(
    cfg: config.Config, tmp_path: Path
) -> None:
    text = "main is at abc1234def; 5 commits not pushed.\nGR-10 merged, GR-11 merged.\n"
    out = _report(cfg, tmp_path, text)
    assert "Mismatch: the state file says main is abc1234def, git says" in out
    assert "Mismatch: the state file says 5 commits not pushed, git says 1" in out
    assert "Mismatch: the state file says GR-10 merged, state.json says rejected" in out
    assert "GR-11 merged, state.json" not in out  # GR-11 has no state record: nothing to compare


def test_a_state_file_that_agrees_has_no_mismatch(cfg: config.Config, tmp_path: Path) -> None:
    head = _git(cfg.repo, "rev-parse", "--short", "main")
    out = _report(cfg, tmp_path, f"main {head}, 1 commit not pushed. GR-10 rejected.\n")
    assert "Mismatch" not in out


def test_a_missing_state_file_and_a_missing_origin_do_not_fail(
    cfg: config.Config, tmp_path: Path
) -> None:
    _git(cfg.repo, "remote", "remove", "origin")
    out = vp_start.render(cfg, tmp_path / "none.md")
    assert "(no state file found)" in out and "not pushed: no origin ref" in out
