"""CI watch: a red GitHub run posts one move with the failing tests; nothing is seen twice."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.orchestrator import ci_watch, config, moves

ROOT = Path(__file__).resolve().parents[2]
LOG = "\n".join(
    [
        "job\tstep\tFAILED tests/a.py::test_one - assert 1 == 2",
        "job\tstep\tFAILED tests/a.py::test_one - assert 1 == 2",
        "job\tstep\tERROR at setup of test_two",
        "job\tstep\tE   KeyError: 'NEO4J_URI'",
        "job\tstep\tE   KeyError: 'NEO4J_URI'",
        "job\tstep\tE   " + "x" * 300,
    ]
)


@pytest.fixture
def cfg(tmp_path: Path) -> config.Config:
    return config.load_config(ROOT, overrides={"state_dir": str(tmp_path / "state")})


def _runner(runs: list[dict]):
    def run(argv: list[str]) -> str:
        return json.dumps(runs) if argv[2] == "list" else LOG

    return run


def _run(run_id: int, conclusion: str, status: str = "completed") -> dict:
    return {
        "databaseId": run_id,
        "conclusion": conclusion,
        "headSha": "abcdef1234",
        "displayTitle": "a push",
        "status": status,
    }


def _actions(cfg: config.Config) -> list[str]:
    text = (cfg.state_dir / "ops-ledger.jsonl").read_text()
    return [json.loads(line)["action"] for line in text.splitlines()]


def test_a_green_run_is_recorded_once(cfg: config.Config) -> None:
    runner = _runner([_run(1, "success")])
    assert len(ci_watch.check(cfg, runner)) == 1
    assert ci_watch.check(cfg, runner) == []
    assert _actions(cfg) == ["ci_green"]


def test_a_red_run_posts_one_move_with_the_failing_tests(cfg: config.Config) -> None:
    ci_watch.check(cfg, _runner([_run(2, "failure")]))
    (move,) = moves.open_for(cfg, "devops")
    assert move["kind"] == "ci" and move["task"] == "abcdef1"
    assert move["text"].count("tests/a.py::test_one") == 1
    assert "test_two" in move["text"]
    assert move["text"].count("KeyError") == 1
    assert "x" * 201 not in move["text"]
    assert "ci_red" in _actions(cfg)


def test_an_unfinished_run_is_ignored_until_it_completes(cfg: config.Config) -> None:
    assert ci_watch.check(cfg, _runner([_run(3, "", "in_progress")])) == []
    assert not (cfg.state_dir / "ops-ledger.jsonl").exists()
    assert len(ci_watch.check(cfg, _runner([_run(3, "success")]))) == 1


def test_the_seen_file_makes_a_second_call_a_no_op(cfg: config.Config) -> None:
    runner = _runner([_run(4, "failure")])
    ci_watch.check(cfg, runner)
    ci_watch.check(cfg, runner)
    assert len(moves.open_for(cfg, "devops")) == 1
    assert json.loads((cfg.state_dir / "ci-seen.json").read_text()) == [4]


def test_the_first_look_reports_only_the_newest_finished_run(cfg: config.Config) -> None:
    runs = [_run(3, "failure"), _run(2, "failure"), _run(1, "failure")]
    lines = ci_watch.check(cfg, _runner(runs))
    assert len(lines) == 1
    assert len(moves.open_for(cfg, "devops")) == 1


def test_a_timestamp_before_the_line_is_ignored() -> None:
    log = "gate\tstep\t2026-11-18T12:03:06.2135466Z E   KeyError: 'NEO4J_URI'"
    assert ci_watch._failures(log) == ([], ["E   KeyError: 'NEO4J_URI'"])


def test_wait_returns_when_the_run_of_the_commit_has_finished(cfg: config.Config) -> None:
    runs = [_run(5, "failure")]
    lines = ci_watch.wait(cfg, "abcdef1234", _runner(runs), timeout_s=5, poll_s=0)
    assert lines and "RED" in lines[0]
    assert len(moves.open_for(cfg, "devops")) == 1


def test_wait_gives_up_when_the_run_never_finishes(cfg: config.Config) -> None:
    runs = [_run(6, "", status="in_progress")]
    lines = ci_watch.wait(cfg, "abcdef1234", _runner(runs), timeout_s=0, poll_s=0)
    assert "no finished CI run" in lines[0]
