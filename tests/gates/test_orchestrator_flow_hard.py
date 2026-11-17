"""FLOW-HARD.md gates built by DevOps (owner 2026-11-17): D12 graph-name literals, D11 test text.

Uses the fake engines of test_orchestrator.py and a throwaway git repository.
"""

# ruff: noqa: F811
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tests.gates.test_orchestrator import (  # noqa: F401  (env is a fixture)
    GOOD,
    IMPL,
    TASK,
    _cfg,
    _coder,
    _git,
    _plan,
    _prompts,
    _task,
    env,
)
from tools.orchestrator import config, gates, pipeline

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = (ROOT / "config" / "graph_schema.yaml").read_text()
ROOTS = ["src/memini/infra/graph/"]


def _repo(tmp: Path, files: dict[str, str]) -> Path:
    repo = tmp / "r"
    for name, text in {"config/graph_schema.yaml": SCHEMA, **files}.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "x")
    _git(repo, "config", "user.email", "x@example.invalid")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "x")
    return repo


# --- D12 -------------------------------------------------------------------------------------


def test_graph_names_cover_labels_relationships_properties_and_values() -> None:
    names = gates.graph_names(SCHEMA)
    assert {"Entity", "BELONGS_TO_COMMUNITY", "content_hash", "dirty", "superseded"} <= names
    assert "nothing_like_this" not in names


def test_a_graph_name_literal_is_refused_including_a_cypher_parameter(tmp_path: Path) -> None:
    code = 'LABEL = "Entity"\nparams = {"scope": 1}\nquery = "MATCH (n) RETURN n"\n'
    repo = _repo(tmp_path, {"src/memini/infra/graph/store.py": code})
    found = gates.graph_literal_problems(repo, "HEAD", ["src/memini/infra/graph/store.py"], ROOTS)
    assert len(found) == 2
    assert "'Entity'" in found[0] and "'scope'" in found[1]


def test_typed_names_and_other_strings_pass(tmp_path: Path) -> None:
    code = 'names = load()\nq = f"MATCH (n:`{names.entity}`) RETURN n"\nx = "not a graph name"\n'
    repo = _repo(tmp_path, {"src/memini/infra/graph/store.py": code})
    assert (
        gates.graph_literal_problems(repo, "HEAD", ["src/memini/infra/graph/store.py"], ROOTS) == []
    )


def test_only_changed_files_under_the_graph_paths_are_scanned(tmp_path: Path) -> None:
    bad = 'x = "Entity"\n'
    repo = _repo(tmp_path, {"src/memini/other.py": bad, "src/memini/infra/graph/old.py": bad})
    assert gates.graph_literal_problems(repo, "HEAD", ["src/memini/other.py"], ROOTS) == []
    assert gates.graph_literal_problems(repo, "HEAD", [], ROOTS) == []


def test_the_shipped_config_lists_the_graph_paths() -> None:
    cfg = config.load_config(ROOT)
    assert "src/memini/infra/graph/" in cfg.graph_name_paths


# --- D11 -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("uri = 'bolt://10.0.0.5:7687'", "IP address"),
        ("uri = 'bolt://localhost:7687'", "host name"),
        ("uri = 'https://db.example.com'", "host name"),
        ("path = '/Users/someone/work'", "home path"),
        ("import atexit", "deny word"),
    ],
)
def test_text_problems_finds_each_kind(text: str, kind: str) -> None:
    found = gates.text_problems(text, [r"\batexit\b"])
    assert any(f.startswith(kind) for f in found)


def test_plain_text_has_no_problem() -> None:
    assert (
        gates.text_problems("def test_add():\n    assert add(1, 2) == 3\n", [r"\batexit\b"]) == []
    )


def test_a_visible_test_with_an_address_is_refused_before_sealing(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = "def test_x():\n    assert 'bolt://10.0.0.5:7687'\n"
    _plan(env, monkeypatch, qa=[{"write": {"tests/unit/test_calc.py": bad}}])
    out = pipeline.run_task(_cfg(env), _task(env))
    assert out.status == "rejected"
    assert "QA tests refused before sealing" in out.reason
    assert "10.0.0.5" in out.reason


def test_a_hidden_test_with_a_deny_word_is_refused_before_sealing(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    hidden = env["tmp"] / "holdout" / TASK / "test_hidden_exit.py"
    hidden.write_text("import atexit\n\n\ndef test_h():\n    assert atexit\n")
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD}))
    out = pipeline.run_task(_cfg(env), _task(env))
    assert out.status == "rejected"
    assert "hold-out/test_hidden_exit.py" in out.reason


def test_a_slice_with_a_graph_name_literal_is_rejected_by_the_gate(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = env["repo"]
    (repo / "config").mkdir()
    (repo / "config" / "graph_schema.yaml").write_text(SCHEMA)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "schema")
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD + 'KEY = "scope"\n'}))
    cfg = _cfg(env, graph_name_paths=["src/memini/core/calc/"])
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "rejected"
    assert "graph name literal 'scope'" in out.reason


def test_a_regular_expression_is_not_the_builtin_compile() -> None:
    deny = [r"\bcompile\s*\("]
    assert gates.text_problems("PAT = re.compile('x')\n", deny) == []
    assert gates.text_problems("code = compile(src, 'f', 'exec')\n", deny) != []


# --- D8 the Verifier ------------------------------------------------------------------------

import json  # noqa: E402

from tests.gates.test_orchestrator import AUDIT_PASS, SNEAKY  # noqa: E402
from tools.orchestrator import audit, reconcile, stages, state  # noqa: E402
from tools.orchestrator.enqueue import Ticket  # noqa: E402,F401

CARD_FILE = (
    "## IN: baseline\n\n| Line | Capability | Status | Task | Sent |\n|---|---|---|---|---|\n"
)
CARD_FILE += "| A1 | The calc adds two numbers. | not done | - | - |\n"


def _row(
    card: str = "A1",
    status: str = "met",
    test: str = "tests/unit/test_calc.py::test_add",
    reachable: bool = True,
    gap: str = "",
) -> dict[str, object]:
    return {
        "card": card,
        "status": status,
        "code": "src/x.py:3",
        "reachable": reachable,
        "test": test,
        "gap": gap,
    }


def _answer(*rows: dict[str, object], extras: list[str] | None = None) -> str:
    return json.dumps({"lines": list(rows), "extras": extras or []})


def test_the_verifier_table_passes_only_when_every_line_is_met_reachable_and_tested() -> None:
    ok = audit.parse_verifier(_answer(_row(), extras=["a flag nobody asked for"]), ["A1"])
    assert ok.ok and ok.extras == ("a flag nobody asked for",)
    assert "| A1 | met | src/x.py:3 | yes | tests/unit/test_calc.py::test_add |" in ok.table


@pytest.mark.parametrize(
    ("row", "kind", "expect"),
    [
        (_row(status="partial", gap="the old code also retried"), "code", "A1 is partial"),
        (_row(status="missing"), "code", "A1 is missing"),
        (_row(reachable=False), "code", "not reachable"),
        (_row(test=""), "test", "no test proves it"),
    ],
)
def test_each_unmet_line_blocks_and_is_routed(
    row: dict[str, object], kind: str, expect: str
) -> None:
    found = audit.parse_verifier(_answer(row), ["A1"])
    assert not found.ok
    assert any(
        expect in p for p in (found.code_problems if kind == "code" else found.test_problems)
    )


def test_a_missing_row_or_garbage_fails_closed() -> None:
    assert audit.parse_verifier(_answer(_row("A2")), ["A1"]).code_problems
    assert audit.parse_verifier("looks fine to me", ["A1"]).code_problems


def _ticket(env: dict[str, Any], tmp: Path) -> config.Config:
    squads = tmp / "squads" / "alpha"
    squads.mkdir(parents=True)
    (squads / "CARD.md").write_text(CARD_FILE)
    cfg = _cfg(env, cards_dir=str(tmp / "squads"))
    queue = cfg.stages[0].queue
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text(f"{TASK} src/memini/core/calc/__init__.py - - 'A1 the calc adds'\n")
    return cfg


def test_a_verified_slice_merges_and_keeps_the_table(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _ticket(env, env["tmp"])
    _plan(
        env,
        monkeypatch,
        coder=_coder({IMPL: GOOD}),
        audit=[{"stdout": AUDIT_PASS}, {"stdout": _answer(_row())}],
    )
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "merged", out.reason
    assert "| A1 | met |" in (cfg.state_dir / "runs" / TASK / "verifier-1.md").read_text()


def test_a_partial_line_goes_back_to_the_coder(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _ticket(env, env["tmp"])
    _plan(
        env,
        monkeypatch,
        coder=_coder({IMPL: GOOD}, {IMPL: GOOD + "\n# more\n"}),
        audit=[
            {"stdout": AUDIT_PASS},
            {"stdout": _answer(_row(status="partial", gap="no retry"))},
            {"stdout": AUDIT_PASS},
            {"stdout": _answer(_row())},
        ],
    )
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "merged", out.reason
    assert "verifier: A1 is partial: no retry" in _prompts(env, "coder")[1]


def test_a_line_without_a_proving_test_goes_to_qa(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _ticket(env, env["tmp"])
    _plan(
        env,
        monkeypatch,
        coder=_coder({IMPL: GOOD}),
        audit=[{"stdout": AUDIT_PASS}, {"stdout": _answer(_row(test=""))}],
    )
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "disputed"
    assert "A1 is met" in (cfg.state_dir / "agent-inbox" / "qa.md").read_text()


# --- D6 failure routing -----------------------------------------------------------------------


def test_the_coder_sees_the_failing_assertion_and_the_same_one_twice_is_a_suspect_test(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(env, monkeypatch, coder=_coder({IMPL: SNEAKY}, {IMPL: SNEAKY + "\n# again\n"}))
    cfg = _cfg(env)
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "disputed"
    assert "suspect test" in out.reason
    assert "test_holdout_secret_marker_xyz" in (cfg.state_dir / "agent-inbox" / "qa.md").read_text()
    assert "suspect_test" in [
        json.loads(x)["action"]
        for x in (cfg.state_dir / "ops-ledger.jsonl").read_text().splitlines()
    ]


def test_failing_messages_come_from_pytests_own_summary_line() -> None:
    out = (
        "FAILED tests/unit/test_a.py::test_x - assert 3 == 4\nFAILED tests/unit/test_a.py::test_y\n"
    )
    ids = ("tests/unit/test_a.py::test_x", "tests/unit/test_a.py::test_y")
    assert gates.failing_messages(out, ids) == ("tests/unit/test_a.py::test_x: assert 3 == 4",)
    failure = gates.Failure("gate failed", ids[:1], messages=("t::x: assert 3 == 4",))
    assert "failing assertions: t::x: assert 3 == 4" in failure.summary()


def test_a_coder_dispute_also_goes_to_the_qa_inbox(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(env, monkeypatch, coder=[{"stdout": "TEST-DISPUTE: add must be commutative only\n"}])
    cfg = _cfg(env)
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "disputed"
    assert "the coder disputes the tests" in (cfg.state_dir / "agent-inbox" / "qa.md").read_text()


# --- D5 handoff -------------------------------------------------------------------------------


def test_pending_stage_tasks_keep_the_code_loop_waiting(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _ticket(env, env["tmp"])
    assert stages.upstream_pending(cfg) == [f"spec/{TASK}"]
    rec = stages.StageState(cfg, "spec")
    rec.set(TASK, stages.DONE)
    assert stages.upstream_pending(cfg) == []
    rec.set(TASK, "failed")
    assert stages.upstream_pending(cfg) == []  # a failed task is not waited for


def test_the_code_loop_does_not_exit_on_an_empty_queue_before_qa_sealed(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.orchestrator import cli  # noqa: PLC0415

    cfg = _ticket(env, env["tmp"])
    polls = []

    def pending(_cfg: config.Config) -> list[str]:
        polls.append(1)
        return [f"spec/{TASK}"] if len(polls) < 3 else []

    monkeypatch.setattr(stages, "upstream_pending", pending)
    code = cli.run_queue_parallel(
        cfg, workers=1, launch=lambda e: None, pause=0, wait_upstream=True
    )
    assert len(polls) == 3  # it waited two rounds, then saw nothing left
    assert code in (0, 3)
    polls.clear()
    cli.run_queue_parallel(cfg, workers=1, launch=lambda e: None, pause=0, wait_upstream=False)
    assert polls == []  # without the flag it never waits


# --- D7 land and reconciliation ---------------------------------------------------------------


def test_land_makes_state_queue_and_main_agree(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD}))
    cfg = _cfg(env)
    assert pipeline.run_task(cfg, _task(env)).status == "merged"
    assert reconcile.mismatches(cfg) == []
    st = state.State(cfg)
    st.update(TASK, "rejected")
    assert reconcile.mismatches(cfg) == [
        f"{TASK}: on main as {reconcile.on_main(cfg)[TASK][:7]}, state says rejected"
    ]
    cfg.queue.parent.mkdir(parents=True, exist_ok=True)
    cfg.queue.write_text(f"{TASK} specs/{TASK}.md\n")
    changed = reconcile.land(cfg, TASK, "main")
    assert changed == [cfg.queue]
    assert cfg.queue.read_text().startswith("# MERGED ")
    assert st.task(TASK)["status"] == "merged" and TASK in st.queue_done()
    assert reconcile.mismatches(cfg) == []


def test_land_refuses_a_commit_that_is_not_on_main(env: dict[str, Any]) -> None:
    repo = env["repo"]
    _git(repo, "checkout", "-q", "-b", "side")
    (repo / "x.txt").write_text("x")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "x")
    sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    with pytest.raises(ValueError, match="not on main"):
        reconcile.land(_cfg(env), TASK, sha)
