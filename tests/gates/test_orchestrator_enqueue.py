"""The ready gate and ``enqueue`` (FLOW-HARD.md D1, D4): refuse a bad ticket, write a good one."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.gates.test_orchestrator import _git
from tools.orchestrator import config, enqueue

ROOT = Path(__file__).resolve().parents[2]
CARD = """# Card
## IN: baseline

| Line | Capability | Status | Task | Sent |
|---|---|---|---|---|
| A1 | The store writes a node. | done | X | merged |
| A2 | The store deletes a node. | not done | - | - |
| A3 | A different squad's line. | not done | - | - |

## OUT
| A9 | Not built. | not done | - | - |
"""
OTHER = "| A3 | Another area's line. | not done | - | - |\n"
CAPS = "| # | version | id | capability | how | evidence | graph |\n|---|---|---|---|---|---|---|\n"
CAPS += "| 1 | baseline | x | c | h | `old/store.py:write L10-20` | W |\n"
TYPED = """
class GraphLabels:
    project: str


class GraphProperties:
    scope: str
    dirty: str
"""
MODULE = "src/memini/infra/graph/store.py"


@pytest.fixture
def cfg(tmp_path: Path) -> config.Config:
    sq = tmp_path / "squads"
    (sq / "alpha").mkdir(parents=True)
    (sq / "alpha" / "CARD.md").write_text(CARD)
    (sq / "alpha" / "SPEC.md").write_text(f"Files: {MODULE}\n")
    (sq / "beta").mkdir()
    (sq / "beta" / "CARD.md").write_text("## IN: baseline\n\n" + OTHER)
    (sq / "beta" / "SPEC.md").write_text("Files: src/memini/other.py\n")
    (tmp_path / "capabilities").mkdir()
    (tmp_path / "capabilities" / "archrag.md").write_text(CAPS)
    repo = tmp_path / "r"
    (repo / "config").mkdir(parents=True)
    (repo / "config" / "graph_schema.yaml").write_text(
        (ROOT / "config" / "graph_schema.yaml").read_text()
    )
    (repo / "src" / "memini" / "infra" / "graph").mkdir(parents=True, exist_ok=True)
    (repo / "src" / "memini" / "infra" / "graph" / "schema.py").write_text(TYPED)
    (repo / "deploy").mkdir()
    (repo / "deploy" / ".env.example").write_text("NEO4J_URI=\n")
    (repo / "src" / "memini").mkdir(parents=True, exist_ok=True)
    (repo / "src" / "memini" / "mod.py").write_text("def load_schema():\n    pass\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "x")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "x")
    return config.load_config(
        repo, overrides={"cards_dir": str(sq), "state_dir": str(tmp_path / "state")}
    )


PARTS = " Context: c. Acceptance: a. Done: d."
CALLER = "memini.mod:load_schema"  # a card ticket names its top caller (F-orchestrator-068)
TOP = f" Top caller: {CALLER}."


def _t(text: str, deps: str = "-", module: str = MODULE, task_id: str = "T-1") -> enqueue.Ticket:
    return enqueue.parse(f"{task_id} {module} {deps} - '{text}'")


def test_a_line_that_does_not_parse_is_refused() -> None:
    with pytest.raises(ValueError, match="shell quoting"):
        enqueue.parse("T-1 m - - 'unclosed")
    with pytest.raises(ValueError, match="expected"):
        enqueue.parse("T-1 only-two")


def test_card_ranges_cover_the_known_lines_between_the_ends() -> None:
    assert enqueue.card_ids("A1 to A4 and Q7", {"A1", "A3", "A4"}) == ["A1", "A3", "A4", "Q7"]


def test_a_good_ticket_is_ready(cfg: config.Config) -> None:
    assert (
        enqueue.problems(cfg, _t("A1: write a node, as src/memini/mod.py does." + PARTS + TOP))
        == []
    )


@pytest.mark.parametrize(
    ("text", "expect"),
    [
        ("A1 uses bolt://10.1.2.3:7687", "IP address"),
        ("A1 connects to localhost", "host name"),
        ("A1 reads /Users/someone/x", "home path"),
        ("A1 then import atexit", "deny word"),
        ("A1 and A99", "card A99 does not exist"),
        ("A1 and A9", "card A9 does not exist or is not a baseline"),
        ("A1 and A3", "card A3 belongs to beta"),
        ("A1 reads config/missing.yaml", "file config/missing.yaml does not exist on main"),
        ("A1 reads NEO4J_NOPE", "environment variable NEO4J_NOPE does not exist"),
        ("A1 calls `nothing_like_it`", "nothing_like_it does not exist on main"),
        ("A1 uses `names.properties.nope`", "names.properties.nope is not a role"),
        ("A1 matches (n:Invented)", "graph name Invented has no role"),
    ],
)
def test_each_refusal(cfg: config.Config, text: str, expect: str) -> None:
    assert any(expect in p for p in enqueue.problems(cfg, _t(text)))


def test_named_inputs_that_exist_pass(cfg: config.Config) -> None:
    text = "A1 reads config/graph_schema.yaml, NEO4J_URI, `load_schema`, `names.properties.scope`."
    assert enqueue.problems(cfg, _t(text + PARTS + TOP)) == []


def test_needs_without_a_role_are_refused_and_go_to_devops(cfg: config.Config) -> None:
    ticket, found = enqueue.enqueue(
        cfg,
        task_id="T-2",
        module=MODULE,
        card_list=["A1"],
        needs=["dirty", "no_role_here"],
        top_caller=CALLER,
    )
    assert found == ["graph name no_role_here has no role in names: of config/graph_schema.yaml"]
    inbox = (cfg.state_dir / "agent-inbox" / "devops.md").read_text()
    assert "T-2 needs roles" in inbox and "no_role_here" in inbox
    assert not cfg.stages[0].queue.exists()
    assert ticket.id == "T-2"


def test_enqueue_writes_the_text_from_the_cards_with_the_evidence(cfg: config.Config) -> None:
    _, found = enqueue.enqueue(
        cfg,
        task_id="T-3",
        module=MODULE,
        card_list=["A1", "A2"],
        sources=["s.md"],
        top_caller=CALLER,
    )
    assert found == []
    [ticket] = enqueue.read_tickets(cfg.stages[0].queue)
    assert ticket.id == "T-3" and ticket.module == MODULE and ticket.deps == ()
    assert "A1: The store writes a node. (old code: old/store.py:write L10-20)" in ticket.text
    assert "A2: The store deletes a node." in ticket.text
    log = (cfg.state_dir / "ops-ledger.jsonl").read_text().splitlines()
    assert json.loads(log[-1])["action"] == "enqueue"


def test_a_ticket_is_queued_once_and_a_card_that_does_not_exist_is_an_error(
    cfg: config.Config,
) -> None:
    _, first = enqueue.enqueue(
        cfg, task_id="T-4", module=MODULE, card_list=["A1"], top_caller=CALLER
    )
    assert first == []
    _, found = enqueue.enqueue(
        cfg, task_id="T-4", module=MODULE, card_list=["A1"], top_caller=CALLER
    )
    assert found == ["T-4 is already queued"]
    with pytest.raises(ValueError, match="A77"):
        enqueue.enqueue(cfg, task_id="T-5", module=MODULE, card_list=["A77"])


def test_dependencies_must_be_acyclic_and_the_same_file_is_ordered(cfg: config.Config) -> None:
    a = _t("A1", deps="T-2", task_id="T-1")
    b = _t("A1", deps="T-1", module="src/memini/b.py", task_id="T-2")
    assert any("not acyclic" in p for p in enqueue.problems(cfg, b, [a]))
    c = _t("A1", task_id="T-1")
    d = _t("A1", task_id="T-2")  # same file as T-1: ordered, so 1 worker
    e = _t("A1", module="src/memini/b.py", task_id="T-3")
    assert enqueue.workers([c, d, e]) == 2
    assert enqueue.workers([c, d]) == 1


# --- the hook that stops hand-written queue lines --------------------------------------------


def _event(tool: str, **args: str) -> dict[str, object]:
    return {"tool_name": tool, "tool_input": args}


@pytest.mark.parametrize(
    "event",
    [
        _event("Write", file_path="/w/.memini-orchestrator/queue-spec.txt"),
        _event("Edit", file_path="/w/.memini-orchestrator/queue.txt"),
        _event("Bash", command="echo 'GR-11 x' >> /w/.memini-orchestrator/queue-spec.txt"),
        _event("Bash", command="sed -i '' 's/a/b/' /w/.memini-orchestrator/queue.txt"),
        _event("Bash", command="cp x.txt /w/.memini-orchestrator/queue-query.txt"),
        "not an event",
    ],
)
def test_the_hook_blocks_hand_written_queues(event: object) -> None:
    from tools.orchestrator.hooks import guard_queue  # noqa: PLC0415

    assert guard_queue.blocked(event)


@pytest.mark.parametrize(
    "event",
    [
        _event("Write", file_path="/w/memini/src/queue_helper.py"),
        _event("Read", file_path="/w/.memini-orchestrator/queue-spec.txt"),
        _event("Bash", command="grep GR-06 /w/.memini-orchestrator/queue-spec.txt"),
        _event("Bash", command="python -m tools.orchestrator enqueue GR-11 --file f --cards A1"),
    ],
)
def test_the_hook_lets_everything_else_through(event: object) -> None:
    from tools.orchestrator.hooks import guard_queue  # noqa: PLC0415

    assert guard_queue.blocked(event) == ""


def test_a_role_without_a_typed_field_is_refused(cfg: config.Config) -> None:
    found = enqueue.problems(
        cfg, _t("A1 reads `names.properties.scope`." + PARTS + TOP), needs=["Project"]
    )
    assert found == []
    found = enqueue.problems(cfg, _t("A1 reads (:Domain) and `names.properties.score`"))
    assert "role labels.domain is in names: but has no field in " in " ".join(found)
    assert "role properties.score is in names: but has no field in " in " ".join(found)


def test_the_ticket_that_edits_the_typed_classes_is_exempt_from_the_typed_check(
    cfg: config.Config,
) -> None:
    typed = "src/memini/infra/graph/schema.py"
    text = "reads (:Domain) and `names.properties.score`"
    assert enqueue._names_problems(cfg, _t(text, module=typed), ()) == []
    assert enqueue._names_problems(cfg, _t(text), ())  # any other module is checked
    found = enqueue._names_problems(cfg, _t("matches (n:Invented)", module=typed), ())
    assert found == ["graph name Invented has no role in names: of config/graph_schema.yaml"]


def test_a_fix_ticket_lists_the_consumers_of_every_removed_or_renamed_name(
    cfg: config.Config,
) -> None:
    text = enqueue.fix_text(
        cfg,
        "Drop the loader",
        [],
        ["load_schema=read_schema"],
        "owner 2026-11-17: remove it",
    )
    assert 'Owner words: "owner 2026-11-17: remove it".' in text
    assert "Rename: load_schema -> read_schema." in text
    assert "Consumers of load_schema: src/memini/mod.py." in text


def test_a_fix_ticket_needs_the_owners_words_with_a_date_and_a_valid_rename(
    cfg: config.Config,
) -> None:
    with pytest.raises(ValueError, match="owner-words"):
        enqueue.fix_text(cfg, "x", [], [], "because I said so")
    with pytest.raises(ValueError, match="old=new"):
        enqueue.fix_text(cfg, "x", [], ["nope"], "owner 2026-11-17: x")


def test_a_fix_ticket_goes_through_the_ready_gate_and_is_queued(cfg: config.Config) -> None:
    text = enqueue.fix_text(
        cfg,
        "Cover every role",
        [],
        [],
        "owner 2026-11-16: rule 2",
        context="role X has no test",
        acceptance=["every role is covered"],
        done="the unit suite is green",
    )
    ticket, found = enqueue.enqueue(cfg, task_id="T-9", module=MODULE, text=text)
    assert found == []
    assert [t.id for t in enqueue.read_tickets(cfg.stages[0].queue)] == ["T-9"]
    assert "Cover every role." in ticket.text
    _, refused = enqueue.enqueue(cfg, task_id="T-10", module=MODULE, text=text + " atexit")
    assert any("text:" in p for p in refused)
