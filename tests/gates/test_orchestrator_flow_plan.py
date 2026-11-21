"""F-orchestrator-072 (owner 2026-11-21): the Planner builds its brief from the skeleton document.

Rules guarded: the brief of one flow names its top caller, its cards and its files and functions
and nothing of another flow; a bad document or an unknown top caller is refused and queues
nothing; a ticket WITH a top caller may cross squads, one WITHOUT is refused as before.

Fixtures follow ``test_orchestrator_enqueue.py`` (``cfg``), ``test_orchestrator_planner.py`` (the
stub Planner answer) and ``test_orchestrator_skeleton.py`` (the document rows).
"""

# ruff: noqa: F811
from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.gates.test_orchestrator_enqueue import MODULE, PARTS, _t, cfg  # noqa: F401
from tests.gates.test_orchestrator_planner import _answer
from tests.gates.test_orchestrator_skeleton import HEAD, SEP
from tests.gates.test_orchestrator_stages import _spec_md
from tools.orchestrator import cli, config, enqueue, planner, spec

OTHER_FILE = "src/memini/other.py"  # the file the other squad's spec lists
SCAN_FILE = "src/memini/infra/graph/scan.py"
TOP = "memini.infra.graph.store:write"
OTHER_TOP = "memini.infra.graph.scan:scan"
WRITE = f"| {MODULE} | write | `def write(node: str) -> None:` | A1 | {TOP} | the flow |"
DELETE = f"| {OTHER_FILE} | delete | `def delete(node: str) -> None:` | A2 | {TOP} | |"
SCAN = f"| {SCAN_FILE} | scan | `def scan() -> int:` | A3 | {OTHER_TOP} | another flow |"
VERSIONS = (
    "# Versions\n\n## baseline\n"
    "| A1 | The store writes a node. | dev4 | open | - | - |\n"
    "| A2 | The store deletes a node. | dev4 | open | - | - |\n"
    "| A3 | Another area's line. | dev2 | open | - | - |\n"
)


def _skeleton(cfg: config.Config, *rows: str) -> Path:
    caps = cfg.cards_dir.parent / "capabilities"
    (caps / "VERSIONS.md").write_text(VERSIONS, encoding="utf-8")
    doc = caps / "SKELETON-baseline.md"
    doc.write_text("\n".join(["# Skeleton", "", HEAD, SEP, *rows, ""]), encoding="utf-8")
    return doc


def _brief(cfg: config.Config, tmp_path: Path) -> Path:
    doc = _skeleton(cfg, WRITE, DELETE, SCAN)
    path = tmp_path / "flow-brief.txt"
    path.write_text(planner.flow_brief(cfg, "baseline", doc, TOP, prefix="FL"), encoding="utf-8")
    return path


def _plan_cli(cfg: config.Config, monkeypatch: pytest.MonkeyPatch, top: str) -> int:
    monkeypatch.chdir(cfg.repo)
    over = {"cards_dir": str(cfg.cards_dir), "state_dir": str(cfg.state_dir)}
    args = ["plan", "--version", "baseline", "--top-caller", top, "--prefix", "FL"]
    return cli.main([*args, "--config-json", json.dumps(over)])


def test_the_brief_names_the_top_caller_its_cards_and_its_functions(
    cfg: config.Config, tmp_path: Path
) -> None:
    """The brief of one flow carries the top caller, the flow's cards and files/functions."""
    brief = planner.read_brief(_brief(cfg, tmp_path))
    assert (brief.version, brief.prefix, brief.cards) == ("baseline", "FL", "A1, A2")
    assert f"Top caller: {TOP}" in brief.text
    for wanted in (MODULE, "write", "def write(node: str) -> None", OTHER_FILE, "delete"):
        assert wanted in brief.text


def test_rows_of_another_top_caller_are_not_in_the_brief(
    cfg: config.Config, tmp_path: Path
) -> None:
    """A row that belongs to another top caller gives the brief no card, file or function."""
    brief = planner.read_brief(_brief(cfg, tmp_path))
    assert "A3" not in brief.cards
    assert SCAN_FILE not in brief.text and OTHER_TOP not in brief.text


def test_planning_from_the_skeleton_queues_tickets_with_the_top_caller(
    cfg: config.Config, tmp_path: Path
) -> None:
    """The flow's units pass the ready gate and are queued carrying the flow's top caller."""
    answer = _answer({"module": MODULE, "cards": ["A1", "A2"], "deps": [], "top_caller": TOP})
    lines = planner.plan(cfg, _brief(cfg, tmp_path), lambda _prompt: answer, flow=True)
    assert not any("REFUSED" in line for line in lines)
    [ticket] = enqueue.read_tickets(cfg.stages[0].queue)
    assert (ticket.id, ticket.module) == ("FL-01", MODULE)
    assert enqueue.part(ticket.text, enqueue.TOP_CALLER) == TOP


def test_a_skeleton_with_problems_is_refused_and_queues_nothing(
    cfg: config.Config, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A document that leaves a card of the version without a function plans nothing."""
    doc = _skeleton(cfg, WRITE, DELETE)  # card A3 has no function
    with pytest.raises(ValueError, match=r"has problems.*A3"):
        planner.flow_brief(cfg, "baseline", doc, TOP, prefix="FL")
    assert _plan_cli(cfg, monkeypatch, TOP) != 0
    assert "has problems" in capsys.readouterr().out
    assert not cfg.stages[0].queue.exists()


def test_a_top_caller_the_document_does_not_have_is_refused(
    cfg: config.Config, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A function of the document that is no top caller of it starts no flow."""
    doc = _skeleton(cfg, WRITE, DELETE, SCAN)
    wrong = "memini.other:delete"
    with pytest.raises(ValueError, match="is not a top caller of the skeleton document"):
        planner.flow_brief(cfg, "baseline", doc, wrong, prefix="FL")
    assert _plan_cli(cfg, monkeypatch, wrong) != 0
    assert f"{wrong} is not a top caller" in capsys.readouterr().out
    assert not cfg.stages[0].queue.exists()


def test_a_ticket_with_a_top_caller_may_cross_squads(cfg: config.Config) -> None:
    """A work unit (top caller named) with a card of another squad's module is ready."""
    text = "A1 and A3: write a node." + PARTS + f" Top caller: {TOP}."
    assert enqueue.problems(cfg, _t(text)) == []


def test_the_same_ticket_without_a_top_caller_is_refused_as_before(cfg: config.Config) -> None:
    """Without a top caller a card of another squad's module is still refused."""
    found = enqueue.problems(cfg, _t("A1 and A3: write a node." + PARTS))
    assert f"card A3 belongs to beta, whose spec does not list {MODULE}" in found


def test_the_plan_command_builds_the_brief_and_queues_a_ready_ticket(
    cfg: config.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``plan --version --top-caller --prefix --doc`` exits 0, keeps the brief, queues the unit."""
    doc = _skeleton(cfg, WRITE, DELETE, SCAN)
    answer = _answer({"module": MODULE, "cards": ["A1", "A2"], "deps": [], "top_caller": TOP})
    monkeypatch.setattr(planner, "_ask", lambda _cfg, _prompt: answer)
    monkeypatch.chdir(cfg.repo)
    over = {"cards_dir": str(cfg.cards_dir), "state_dir": str(cfg.state_dir)}
    args = ["plan", "--version", "baseline", "--top-caller", TOP, "--prefix", "FL"]
    assert cli.main([*args, "--doc", str(doc), "--config-json", json.dumps(over)]) == 0
    briefs = [p for p in (cfg.state_dir / "plans").glob("FL-*") if TOP in p.read_text("utf-8")]
    assert briefs, "no built brief under <state_dir>/plans/"
    [ticket] = enqueue.read_tickets(cfg.stages[0].queue)
    assert enqueue.problems(cfg, ticket) == []
    assert enqueue.part(ticket.text, enqueue.TOP_CALLER) == TOP


def test_a_spec_across_two_modules_needs_a_top_caller(cfg: config.Config, tmp_path: Path) -> None:
    """Paths in two modules pass with a ``top_caller`` and are refused without one."""
    refusal = "paths must lie inside one module"
    other = next(r for r in cfg.module_roots if not MODULE.startswith(r)) + "calc/add.py"
    paths = [MODULE, other]
    units = {spec.module_unit(cfg, p) for p in paths}
    assert len(units) == len(paths) and None not in units  # two modules, both configured
    path = tmp_path / "T-1.md"
    path.write_text(_spec_md("T-1", paths=paths, top_caller=TOP), encoding="utf-8")
    assert spec.check(cfg, spec.load(path), "T-1", set()) == []
    path.write_text(_spec_md("T-1", paths=paths), encoding="utf-8")
    found = spec.check(cfg, spec.load(path), "T-1", set())
    assert [p for p in found if refusal in p], found
