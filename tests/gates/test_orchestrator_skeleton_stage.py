"""The skeleton stage runs by its triggers (the flow's steps 2 to 5), and ``flow-check`` proves it.

Only the agent run and the model call are replaced. Every hop is made by serving a role's moves,
as its runner does; the tests make no call in between.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import pytest
from tests.gates.test_orchestrator_enqueue import cfg as enqueue_cfg  # noqa: F401  (a fixture)
from tools.orchestrator import cli, config, enqueue, moves, planner, skeletonstage, state
from tools.orchestrator.executor import Result

VERSIONS = "# Versions\n\n## baseline\n" + "".join(
    f"| A{n} | Text of A{n}. | dev | open | - | - |\n" for n in (1, 2, 3)
)
HEAD = (
    "| Id | File | Function | Signature | Cards | Top caller | Used by | Note |\n"
    "|---|---|---|---|---|---|---|---|\n"
)
TOP = "memini.mod:load_schema"
EXISTING = (
    f"| F1 | src/memini/mod.py | load_schema | def load_schema(): | A1, A2 | {TOP} | - | - |\n"
)
PLANNED = f"| F2 | src/memini/new.py | save | def save(x: int) -> int: | A3 | {TOP} | F1 | - |\n"
GOOD = HEAD + EXISTING + PLANNED
NO_A3 = HEAD + EXISTING  # card A3 has no function: the machine check fails
WORDS = "Owner 2026-11-22: 'approved'"


class Stage:
    """A repo, cards, a Version file, and the two stubs: the agent run and the model call."""

    def __init__(self, base: config.Config, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
        base.version_file.write_text(VERSIONS)
        over = {
            "repo": str(base.repo),
            "cards_dir": str(base.cards_dir),
            "state_dir": str(base.state_dir),
        }
        self.over = ["--config-json", json.dumps(over)]
        self.cfg = config.load_config(base.repo, overrides=over)
        self.text = text
        self.agent_prompts: list[str] = []
        self.model_prompts: list[str] = []
        monkeypatch.setattr(skeletonstage, "executor_for", lambda *_: self)
        monkeypatch.setattr(planner, "_ask", self._model)

    def run(self, argv: Sequence[str], *, stdin: str | None = None, **_: object) -> Result:
        """Stand in for the author's agent: write the scope's documents."""
        assert "--settings" in argv
        self.agent_prompts.append(stdin or "")
        sk = self.cfg.skeleton
        for kind in sk.documents:
            sk.doc(kind, "baseline").write_text(self.text if kind == "skeleton" else "# doc\n")
        return Result(0, "done", "")

    def _model(self, _cfg: config.Config, prompt: str) -> str:
        self.model_prompts.append(prompt)
        return json.dumps({"verdict": "ok", "findings": []})

    def cli(self, *args: str) -> int:
        return cli.main([*args, *self.over])

    def serve(self, role: str) -> int:
        """Serve the role's open moves once, exactly as its runner does."""
        return moves.serve(self.cfg, role, cli.HANDLERS[role], once=True)

    def open(self, role: str) -> list[tuple[str, str]]:
        return [(m["kind"], m["from"]) for m in moves.open_for(self.cfg, role)]

    def rows(self, action: str) -> list[dict[str, Any]]:
        return state.ops_rows(self.cfg, skeletonstage.task("baseline"), action)

    def inbox(self) -> str:
        return self.cfg.inbox.read_text() if self.cfg.inbox.exists() else ""

    def queued(self) -> list[enqueue.Ticket]:
        return enqueue.read_tickets(self.cfg.stages[0].queue)


def test_the_chain_runs_from_the_draft_command_to_the_stub_ticket(
    enqueue_cfg: config.Config,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = Stage(enqueue_cfg, monkeypatch, GOOD)
    sk = s.cfg.skeleton
    doc = sk.doc("skeleton", "baseline")

    # step 2: the one thing a person types posts the move to the author
    assert s.cli("skeleton-draft", "baseline") == 0
    assert s.open("architect") == [("skeleton", "devops")]
    assert not doc.exists()

    # the author's runner: one agent run, the machine check, and the handler posts step 3 itself
    assert s.serve("architect") == 1
    assert len(s.agent_prompts) == 1
    assert str(doc) in s.agent_prompts[0]
    guard = json.loads(
        (s.cfg.state_dir / "runs" / "skeleton-baseline" / "01-guard.json").read_text()
    )
    assert sorted(guard["writable"]) == sorted(
        p.format(scope="baseline") for p in sk.documents.values()
    )
    assert guard["worktree"] == str(sk.dir)
    (drafted,) = s.rows("skeleton_drafted")
    assert drafted["details"]["functions"] == 2
    assert drafted["details"]["problems"] == []
    assert s.open("vp-rnd") == [("skeleton_review", "architect")]
    assert s.inbox() == ""

    # step 3: the reviewer's runner asks the model once; its ok writes the owner inbox (step 4)
    assert s.serve("vp-rnd") == 1
    assert len(s.model_prompts) == 1
    assert "F2 | src/memini/new.py | save" in s.model_prompts[0]
    assert "skeleton baseline is ready for approval" in s.inbox()
    assert "2 functions in 2 files" in s.inbox()
    assert s.open("architect") == [] == s.open("vp-rnd")
    assert s.queued() == []

    # step 4 and 5: the owner's words are recorded with the hash, and the stub ticket is queued
    assert (
        s.cli(
            "skeleton-approve",
            "baseline",
            "--words",
            WORDS,
            "--ticket",
            "SK-1",
            "--file",
            "src/memini/",
        )
        == 0
    )
    (row,) = s.rows("skeleton_approved")
    assert row["reason"] == WORDS
    assert row["details"] == {
        "sha256": hashlib.sha256(doc.read_bytes()).hexdigest(),
        "reviewed": True,
    }
    (ticket,) = s.queued()
    assert ticket.id == "SK-1"
    assert enqueue.part(ticket.text, enqueue.SKELETON) == "baseline"
    assert "NotImplementedError" in ticket.text
    assert ticket.sources == str(doc)

    # the stub ticket's gate: a tree without the planned function is refused, a changed document too
    code = {"src/memini/mod.py": "def load_schema():\n    pass\n"}
    found = skeletonstage.stub_problems(s.cfg, "baseline", code.get)
    assert found == ["skeleton gate: src/memini/new.py: the file is missing (1 functions)"]
    code["src/memini/new.py"] = "def save(x: int) -> int:\n    raise NotImplementedError\n"
    assert skeletonstage.stub_problems(s.cfg, "baseline", code.get) == []
    doc.write_text(GOOD + "\n")
    assert "not approved" in skeletonstage.stub_problems(s.cfg, "baseline", code.get)[0]


def test_a_failing_check_gives_one_revision_then_the_owner_and_no_stub_ticket(
    enqueue_cfg: config.Config,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = Stage(enqueue_cfg, monkeypatch, NO_A3)
    assert s.cli("skeleton-draft", "baseline") == 0
    assert s.serve("architect") == 1

    assert len(s.agent_prompts) == 2  # the draft and exactly one automatic revision
    assert "Fix only these problems:\n- card A3 has no function" in s.agent_prompts[1]
    assert [r["details"]["problems"] for r in s.rows("skeleton_drafted")] == [
        ["card A3 has no function"]
    ] * 2
    assert s.open("vp-rnd") == []  # the reviewer is not asked about a skeleton that fails
    assert "skeleton-baseline: needs the owner" in s.inbox()
    assert "card A3 has no function" in s.inbox()
    assert s.model_prompts == []

    # nothing records an approval, so the stub ticket is refused
    assert s.cli("skeleton-stub", "baseline", "--ticket", "SK-1", "--file", "src/memini/") == 1
    assert (
        s.cli(
            "skeleton-approve",
            "baseline",
            "--words",
            WORDS,
            "--ticket",
            "SK-1",
            "--file",
            "src/memini/",
        )
        == 2
    )
    assert s.rows("skeleton_approved") == []
    assert s.queued() == []


def test_flow_check_reports_every_step_and_a_missing_handler(
    enqueue_cfg: config.Config,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    s = Stage(enqueue_cfg, monkeypatch, GOOD)
    assert s.cli("flow-check") == 0
    out = capsys.readouterr().out
    steps = [x for x in out.splitlines() if x.startswith("step ")]
    assert [x.split()[1] for x in steps] == [str(n) for n in range(1, 11)]
    assert all(x.endswith(": ok") for x in steps)
    assert "MISSING" not in out

    monkeypatch.delitem(cli.HANDLERS["architect"], skeletonstage.KIND)
    assert s.cli("flow-check") == 1
    out = capsys.readouterr().out
    assert "step 2 skeleton: the author drafts and revises: MISSING" in out
    assert "  move skeleton -> architect: MISSING" in out
    assert sum(x.endswith(": MISSING") for x in out.splitlines() if x.startswith("step ")) == 1
