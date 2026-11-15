"""Dev squads (owner 2026-11-15): one coding queue per work unit, each run by its own session.

"So let's do kind of dev squads, and then we need simply to give the relevant working unit":
``run-queue --queue <unit>``. The spec stage hands a task to its unit's queue by module, and a
slice may depend on a slice merged from another unit's queue.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from tools.orchestrator import cli, config, stages, state

ROOT = Path(__file__).resolve().parents[2]


def _cfg(tmp_path: Path, **over: Any) -> config.Config:
    base: dict[str, Any] = {
        "state_dir": str(tmp_path / "state"),
        "executor": "docker",
        "holdout_dir": str(tmp_path / "holdout"),
        "holdout_required": False,
        "queue": str(tmp_path / "queue.txt"),
        "stages": [{"name": "spec", "queue": str(tmp_path / "queue-spec.txt"), "next": "code"}],
        "code_queues": {
            "gs": {"queue": str(tmp_path / "queue-gs.txt"), "modules": ["src/memini/core/g/"]},
            "one": {"queue": str(tmp_path / "queue-one.txt"), "modules": ["src/memini/core/a.py"]},
        },
    }
    return config.load_config(ROOT, overrides={**base, **over}, env={})


def test_the_shipped_config_has_a_squad_per_running_unit() -> None:
    cfg = config.load_config(ROOT, overrides={"state_dir": "/nonexistent-state"}, env={})
    assert {"llm-providers", "graph-schema"} <= set(cfg.code_queues)
    assert cfg.code_queues["llm-providers"].queue == cfg.queue  # dev1 keeps its queue
    assert cfg.max_parallel >= 8


def test_a_task_goes_to_the_queue_of_its_unit(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert config.code_queue_for(cfg, "src/memini/core/g/") == tmp_path / "queue-gs.txt"
    assert config.code_queue_for(cfg, "src/memini/core/g/sub/") == tmp_path / "queue-gs.txt"
    assert config.code_queue_for(cfg, "src/memini/core/a.py") == tmp_path / "queue-one.txt"
    assert config.code_queue_for(cfg, "src/memini/core/gx/") == cfg.queue  # no prefix trick
    stages.handoff(cfg, cfg.stages[0], "G-1 /s/G-1.md", "src/memini/core/g/")
    stages.handoff(cfg, cfg.stages[0], "X-1 /s/X-1.md", "src/memini/other/")
    assert (tmp_path / "queue-gs.txt").read_text() == "G-1 /s/G-1.md\n"
    assert cfg.queue.read_text() == "X-1 /s/X-1.md\n"


def test_a_squad_queue_inside_the_repository_is_refused(tmp_path: Path) -> None:
    with pytest.raises(config.ConfigError, match="outside the repository"):
        _cfg(tmp_path, code_queues={"x": {"queue": "q.txt", "modules": ["src/memini/x/"]}})


class _Proc:
    def poll(self) -> int:
        return 0


def test_a_slice_may_depend_on_a_slice_merged_from_another_squad(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    st = state.State(cfg)
    st.mark_done("LP-08")  # merged by the other squad; not in this queue
    spec = tmp_path / "G-1.md"
    spec.write_text(
        "---\nid: G-1\nmodule: m\npaths: [src/memini/core/g/]\ndepends_on: [LP-08]\nflag: ''\n"
        "behavior: b\ninterface: i\nacceptance: a\nout_of_scope: o\nsize: S\n---\n"
    )
    gs = dataclasses.replace(cfg, queue=tmp_path / "queue-gs.txt")
    gs.queue.write_text(f"G-1 {spec}\n")
    started: list[str] = []

    def launch(entry: state.QueueEntry) -> _Proc:
        started.append(entry.task_id)
        st.mark_done(entry.task_id)
        return _Proc()

    assert cli.run_queue_parallel(gs, workers=2, launch=launch, pause=0.0) == 0
    assert started == ["G-1"]


def test_run_queue_refuses_an_unknown_squad(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _cfg(tmp_path)
    args = cli.parser().parse_args(["run-queue", "--queue", "nope"])
    assert cli._cmd_run_queue(cfg, args) == 2
    assert "unknown code queue 'nope'" in capsys.readouterr().out


def test_status_shows_every_squad_queue(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    (tmp_path / "queue-gs.txt").write_text(f"G-1 {tmp_path / 'G-1.md'}\n")
    lines = cli.status_lines(cfg)
    assert any(ln.startswith("code queue gs: ") and "(1 entries)" in ln for ln in lines)
    assert "  G-1: ready" in lines
