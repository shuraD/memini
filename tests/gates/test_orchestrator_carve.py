"""The coder's guard leaves writable tests open: protected wins over writable otherwise."""

from __future__ import annotations

from pathlib import Path

from tools.orchestrator import config, engine

ROOT = Path(__file__).resolve().parents[2]


def test_a_protected_directory_keeps_its_other_children_protected(tmp_path: Path) -> None:
    for name in ("unit", "gates", "sanity"):
        (tmp_path / "tests" / name).mkdir(parents=True)
    (tmp_path / "tests" / "conftest.py").write_text("")
    out = engine.carve(["tests/", "tools/"], ["src/x.py", "tests/unit/"], tmp_path)
    assert out == ["tests/conftest.py", "tests/gates/", "tests/sanity/", "tools/"]


def test_a_single_writable_test_file_opens_only_that_file(tmp_path: Path) -> None:
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    (tmp_path / "tests" / "unit" / "test_a.py").write_text("")
    (tmp_path / "tests" / "unit" / "test_b.py").write_text("")
    out = engine.carve(["tests/"], ["tests/unit/test_a.py"], tmp_path)
    assert out == ["tests/unit/test_b.py"]


def test_nothing_writable_under_an_entry_leaves_it_whole(tmp_path: Path) -> None:
    assert engine.carve(["tests/", "**/conftest.py"], ["src/x.py"], tmp_path) == [
        "tests/",
        "**/conftest.py",
    ]


def test_the_repo_guard_opens_tests_unit_for_the_coder_and_keeps_the_gates_shut() -> None:
    cfg = config.load_config(ROOT)
    protected = [*cfg.protected, *cfg.guard_protected]
    out = engine.carve(protected, ["src/memini/x.py", *cfg.old_test_paths], ROOT)
    assert not any(e.startswith("tests/unit") for e in out)
    assert "tests/gates/" in out
    assert "**/conftest.py" in out
