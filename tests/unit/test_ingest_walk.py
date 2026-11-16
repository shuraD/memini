"""Acceptance tests for IN-01: corpus walk, database guard and dry run (memini.core.ingest.walk)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from memini.infra.graph.guard import UnsafeDatabaseError


def _walk() -> Any:
    return importlib.import_module("memini.core.ingest.walk")


def _rules(
    *,
    extensions: tuple[str, ...] = (".md", ".txt"),
    skip_dirs: tuple[str, ...] = (),
    skip_names: tuple[str, ...] = (),
    max_line_chars: int = 1000,
) -> Any:
    return _walk().WalkRules(
        extensions=frozenset(extensions),
        skip_dirs=frozenset(skip_dirs),
        skip_names=frozenset(skip_names),
        max_line_chars=max_line_chars,
    )


def _write(root: Path, rel: str, content: str | bytes = "hello\n") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_bytes(content.encode("utf-8"))
    return path


def _paths(files: list[Any]) -> list[str]:
    return [str(f.path) for f in files]


# --- walk_corpus: recursion, ordering, relative paths -----------------------------------------


def test_walk_is_recursive_sorted_by_posix_string_and_relative(tmp_path: Path) -> None:
    for rel in ["b/deep/x.md", "a/z.md", "a.md", "B.md", "c.txt", "a/b/c/d.md"]:
        _write(tmp_path, rel)
    files = _walk().walk_corpus(tmp_path, _rules())
    expected = sorted(["b/deep/x.md", "a/z.md", "a.md", "B.md", "c.txt", "a/b/c/d.md"])
    assert _paths(files) == expected
    assert expected == ["B.md", "a.md", "a/b/c/d.md", "a/z.md", "b/deep/x.md", "c.txt"]
    for f in files:
        assert isinstance(f.path, PurePosixPath)
        assert not f.path.is_absolute()
        assert type(f).__name__ == "CorpusFile"


def test_walk_returns_list(tmp_path: Path) -> None:
    _write(tmp_path, "one.md")
    result = _walk().walk_corpus(tmp_path, _rules())
    assert isinstance(result, list)
    assert len(result) == 1


def test_empty_root_yields_nothing(tmp_path: Path) -> None:
    assert _walk().walk_corpus(tmp_path, _rules()) == []


# --- extensions -------------------------------------------------------------------------------


def test_extension_match_is_case_insensitive_and_others_dropped(tmp_path: Path) -> None:
    _write(tmp_path, "keep.md")
    _write(tmp_path, "UPPER.MD")
    _write(tmp_path, "Mixed.Md")
    _write(tmp_path, "data.bin")
    _write(tmp_path, "README")
    _write(tmp_path, "notes.txt")
    files = _walk().walk_corpus(tmp_path, _rules(extensions=(".md",)))
    assert _paths(files) == ["Mixed.Md", "UPPER.MD", "keep.md"]


# --- skip_dirs --------------------------------------------------------------------------------


def test_skip_dirs_drops_files_at_any_depth(tmp_path: Path) -> None:
    _write(tmp_path, "keep.md")
    _write(tmp_path, "archive/top.md")
    _write(tmp_path, "a/archive/mid.md")
    _write(tmp_path, "a/b/archive/c/deep.md")
    _write(tmp_path, "a/b/kept.md")
    _write(tmp_path, "archived/near.md")
    files = _walk().walk_corpus(tmp_path, _rules(skip_dirs=("archive",)))
    assert _paths(files) == ["a/b/kept.md", "archived/near.md", "keep.md"]


def test_skip_dir_name_in_root_path_itself_does_not_count(tmp_path: Path) -> None:
    root = tmp_path / "archive" / "corpus"
    _write(root, "doc.md")
    _write(root, "sub/more.md")
    _write(root, "sub/archive/hidden.md")
    files = _walk().walk_corpus(root, _rules(skip_dirs=("archive",)))
    assert _paths(files) == ["doc.md", "sub/more.md"]


def test_skip_dir_name_as_file_name_is_not_skipped(tmp_path: Path) -> None:
    _write(tmp_path, "archive.md")
    files = _walk().walk_corpus(tmp_path, _rules(skip_dirs=("archive.md",)))
    assert _paths(files) == ["archive.md"]


# --- skip_names -------------------------------------------------------------------------------


def test_skip_names_drops_exact_name_at_any_depth(tmp_path: Path) -> None:
    _write(tmp_path, "INDEX.md")
    _write(tmp_path, "a/INDEX.md")
    _write(tmp_path, "a/b/INDEX.md")
    _write(tmp_path, "a/INDEX.md.md")
    _write(tmp_path, "a/my-INDEX.md")
    _write(tmp_path, "a/index.md")
    files = _walk().walk_corpus(tmp_path, _rules(skip_names=("INDEX.md",)))
    assert _paths(files) == ["a/INDEX.md.md", "a/index.md", "a/my-INDEX.md"]


# --- empty and whitespace-only ----------------------------------------------------------------


def test_empty_and_whitespace_only_files_dropped(tmp_path: Path) -> None:
    _write(tmp_path, "empty.md", "")
    _write(tmp_path, "blank.md", "  \n\t\r\n   \n")
    _write(tmp_path, "real.md", "\n  x  \n")
    files = _walk().walk_corpus(tmp_path, _rules())
    assert _paths(files) == ["real.md"]


# --- max_line_chars ---------------------------------------------------------------------------


def test_over_long_line_dropped_and_exact_length_kept(tmp_path: Path) -> None:
    _write(tmp_path, "exact.md", "short\n" + "x" * 10 + "\nend\n")
    _write(tmp_path, "exact_last.md", "y" * 10)
    _write(tmp_path, "over.md", "short\n" + "x" * 11 + "\nend\n")
    _write(tmp_path, "over_last.md", "ok\n" + "z" * 11)
    files = _walk().walk_corpus(tmp_path, _rules(max_line_chars=10))
    assert _paths(files) == ["exact.md", "exact_last.md"]


# --- raw text ---------------------------------------------------------------------------------


def test_text_is_raw_not_stripped_with_crlf_kept(tmp_path: Path) -> None:
    content = "  \n  first line\r\nsecond\r\n\n   "
    _write(tmp_path, "raw.md", content)
    (file,) = _walk().walk_corpus(tmp_path, _rules())
    assert file.text == content


def test_invalid_utf8_bytes_are_dropped(tmp_path: Path) -> None:
    _write(tmp_path, "bad.md", b"  caf\xc3\xa9 \xff\xfeok\r\n")
    (file,) = _walk().walk_corpus(tmp_path, _rules())
    assert file.text == "  caf\u00e9 ok\r\n"


# --- limit ------------------------------------------------------------------------------------


def test_limit_returns_first_n_in_sorted_order(tmp_path: Path) -> None:
    for rel in ["d.md", "b.md", "skip.bin", "a/c.md", "a.md"]:
        _write(tmp_path, rel)
    walk = _walk()
    assert _paths(walk.walk_corpus(tmp_path, _rules(), limit=2)) == ["a.md", "a/c.md"]
    assert _paths(walk.walk_corpus(tmp_path, _rules(), limit=10)) == [
        "a.md",
        "a/c.md",
        "b.md",
        "d.md",
    ]
    assert len(walk.walk_corpus(tmp_path, _rules(), limit=None)) == 4


def test_limit_counts_only_matching_files(tmp_path: Path) -> None:
    _write(tmp_path, "a.md", "")
    _write(tmp_path, "b.bin")
    _write(tmp_path, "c.md")
    _write(tmp_path, "d.md")
    files = _walk().walk_corpus(tmp_path, _rules(), limit=1)
    assert _paths(files) == ["c.md"]


# --- WalkRules.from_mapping -------------------------------------------------------------------


def _mapping() -> dict[str, Any]:
    return {
        "extensions": [".MD", ".txt"],
        "skip_dirs": ["archive", "tmp"],
        "skip_names": ["INDEX.md"],
        "max_line_chars": 500,
    }


def test_from_mapping_builds_frozensets_and_lowercases_extensions() -> None:
    rules = _walk().WalkRules.from_mapping(_mapping())
    assert rules.extensions == frozenset({".md", ".txt"})
    assert rules.skip_dirs == frozenset({"archive", "tmp"})
    assert rules.skip_names == frozenset({"INDEX.md"})
    assert rules.max_line_chars == 500
    assert isinstance(rules.extensions, frozenset)
    assert isinstance(rules.skip_dirs, frozenset)
    assert isinstance(rules.skip_names, frozenset)


@pytest.mark.parametrize("key", ["extensions", "skip_dirs", "skip_names", "max_line_chars"])
def test_from_mapping_missing_key_raises_value_error_naming_key(key: str) -> None:
    data = _mapping()
    del data[key]
    with pytest.raises(ValueError, match=key):
        _walk().WalkRules.from_mapping(data)


def test_walk_rules_is_frozen() -> None:
    rules = _rules()
    with pytest.raises(AttributeError):
        rules.max_line_chars = 5


def test_from_mapping_rules_drive_walk(tmp_path: Path) -> None:
    _write(tmp_path, "A.MD")
    _write(tmp_path, "archive/x.md")
    _write(tmp_path, "INDEX.md")
    rules = _walk().WalkRules.from_mapping(_mapping())
    assert _paths(_walk().walk_corpus(tmp_path, rules)) == ["A.MD"]


# --- ensure_safe_database ---------------------------------------------------------------------


def test_local_uri_with_empty_env_returns_localhost() -> None:
    assert _walk().ensure_safe_database("bolt://localhost:7687", {}) == "localhost"


def test_production_host_from_env_is_refused() -> None:
    env = {
        "MEMINI_PROD_NEO4J_HOST": "db.example.test",
        "MEMINI_NEO4J_ALLOWED_HOSTS": "db.example.test",
    }
    with pytest.raises(UnsafeDatabaseError):
        _walk().ensure_safe_database("bolt://db.example.test:7687", env)


def test_production_host_refused_even_when_local() -> None:
    with pytest.raises(UnsafeDatabaseError):
        _walk().ensure_safe_database(
            "bolt://localhost:7687", {"MEMINI_PROD_NEO4J_HOST": "localhost"}
        )


def test_remote_host_not_allow_listed_is_refused() -> None:
    with pytest.raises(UnsafeDatabaseError):
        _walk().ensure_safe_database("neo4j://remote.example.test:7687", {})


def test_allow_listed_remote_host_is_returned() -> None:
    env = {"MEMINI_NEO4J_ALLOWED_HOSTS": "remote.example.test"}
    assert _walk().ensure_safe_database("bolt://remote.example.test:7687", env) == (
        "remote.example.test"
    )


def test_ensure_safe_database_reads_only_given_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMINI_PROD_NEO4J_HOST", "localhost")
    monkeypatch.setenv("MEMINI_NEO4J_ALLOWED_HOSTS", "remote.example.test")
    walk = _walk()
    assert walk.ensure_safe_database("bolt://localhost:7687", {}) == "localhost"
    with pytest.raises(UnsafeDatabaseError):
        walk.ensure_safe_database("bolt://remote.example.test:7687", {})


def test_ensure_safe_database_error_is_the_guard_error() -> None:
    walk = _walk()
    assert getattr(walk, "UnsafeDatabaseError", UnsafeDatabaseError) is UnsafeDatabaseError


# --- dry_run ----------------------------------------------------------------------------------


def _read_lines(out: Path) -> list[dict[str, Any]]:
    raw = out.read_bytes().decode("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def test_dry_run_writes_one_json_line_per_file_in_walk_order(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    for rel in ["b.md", "a/z.md", "a.md", "skip.bin"]:
        _write(root, rel, f"text of {rel}\n")
    out = tmp_path / "out.jsonl"
    calls: list[Any] = []

    def categorize(file: Any) -> dict[str, Any]:
        calls.append(file)
        return {"kind": "note", "index": len(calls), "label": "caf\u00e9"}

    count = _walk().dry_run(root, _rules(), categorize, out)
    assert count == 3
    assert [str(f.path) for f in calls] == ["a.md", "a/z.md", "b.md"]
    assert [f.text for f in calls] == ["text of a.md\n", "text of a/z.md\n", "text of b.md\n"]
    lines = _read_lines(out)
    assert lines == [
        {"path": "a.md", "kind": "note", "index": 1, "label": "caf\u00e9"},
        {"path": "a/z.md", "kind": "note", "index": 2, "label": "caf\u00e9"},
        {"path": "b.md", "kind": "note", "index": 3, "label": "caf\u00e9"},
    ]
    assert len(out.read_bytes().decode("utf-8").splitlines()) == count


def test_dry_run_walked_path_wins_over_decision_path(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _write(root, "sub/doc.md")
    out = tmp_path / "out.jsonl"
    count = _walk().dry_run(
        root, _rules(), lambda _f: {"path": "elsewhere.md", "category": "x"}, out
    )
    assert count == 1
    assert _read_lines(out) == [{"path": "sub/doc.md", "category": "x"}]


def test_dry_run_honours_limit(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    for rel in ["a.md", "b.md", "c.md"]:
        _write(root, rel)
    out = tmp_path / "out.jsonl"
    calls: list[str] = []

    def categorize(file: Any) -> dict[str, Any]:
        calls.append(str(file.path))
        return {}

    count = _walk().dry_run(root, _rules(), categorize, out, limit=2)
    assert count == 2
    assert calls == ["a.md", "b.md"]
    assert _read_lines(out) == [{"path": "a.md"}, {"path": "b.md"}]


def test_dry_run_empty_corpus_returns_zero(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    out = tmp_path / "out.jsonl"
    calls: list[Any] = []
    count = _walk().dry_run(root, _rules(), lambda f: calls.append(f) or {}, out)
    assert count == 0
    assert calls == []
    assert _read_lines(out) == []


def test_dry_run_uses_no_database_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    walk = _walk()
    guard = importlib.import_module("memini.infra.graph.guard")
    root = tmp_path / "corpus"
    _write(root, "a.md")
    _write(root, "b.md")
    out = tmp_path / "out.jsonl"
    used: list[str] = []

    def refuse(name: str) -> Any:
        def _fail(*_args: Any, **_kwargs: Any) -> Any:
            used.append(name)
            raise AssertionError(f"dry_run must not call {name}")

        return _fail

    monkeypatch.delenv("MEMINI_PROD_NEO4J_HOST", raising=False)
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.setattr(guard, "guard_from_env", refuse("guard_from_env"))
    monkeypatch.setattr(guard, "check_uri", refuse("check_uri"))
    monkeypatch.setattr(walk, "ensure_safe_database", refuse("ensure_safe_database"))
    for name in ("guard_from_env", "check_uri"):
        monkeypatch.setattr(walk, name, refuse(name), raising=False)

    count = walk.dry_run(root, _rules(), lambda _f: {"ok": True}, out)
    assert count == 2
    assert used == []
    assert _read_lines(out) == [{"path": "a.md", "ok": True}, {"path": "b.md", "ok": True}]
