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
