"""D12 checks the diff (GR-10): literals a change adds or deletes in graph modules are refused.

A real throwaway git repository with the shipped graph schema; nothing under test is mocked.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.gates.test_orchestrator import _git
from tools.orchestrator import gates, gitops

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = (ROOT / "config" / "graph_schema.yaml").read_text()
ROOTS = ["src/memini/infra/graph/"]
STORE = "src/memini/infra/graph/store.py"
OTHER = "src/memini/infra/graph/other.py"


def _commit(repo: Path, files: dict[str, str | None]) -> None:
    for name, text in files.items():
        path = repo / name
        if text is None:
            path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "x")


def _problems(
    tmp: Path, before: dict[str, str], after: dict[str, str | None], roots: list[str] = ROOTS
) -> list[str]:
    repo = tmp / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "x")
    _git(repo, "config", "user.email", "t@example.invalid")
    _commit(repo, {"config/graph_schema.yaml": SCHEMA, **before})
    base = gitops.out(repo, "rev-parse", "HEAD").strip()
    _commit(repo, after)
    rev = gitops.out(repo, "rev-parse", "HEAD").strip()
    changes = gitops.name_status(repo, base, rev)
    return gates.graph_literal_problems(repo, base, rev, changes, roots)


def test_a_new_literal_in_a_modified_file_is_refused(tmp_path: Path) -> None:
    """Rule: a graph name literal added to an existing graph module is a problem."""
    found = _problems(tmp_path, {STORE: "x = 1\n"}, {STORE: 'x = 1\nlabel = "Entity"\n'})
    assert len(found) == 1
    assert "'Entity'" in found[0] and f"{STORE}:2" in found[0]


def test_an_old_literal_left_untouched_is_not_refused(tmp_path: Path) -> None:
    """Rule: a literal the file already had is not blamed on the change."""
    old = 'label = "Entity"\n'
    assert _problems(tmp_path, {STORE: old}, {STORE: old + "y = 2\n"}) == []


def test_an_old_literal_moved_to_another_line_is_not_refused(tmp_path: Path) -> None:
    """Rule: counting is per value, not per line: a moved literal is the same occurrence."""
    assert _problems(tmp_path, {STORE: 'a = "Entity"\n'}, {STORE: '\n\na = "Entity"\n'}) == []


def test_a_second_use_of_an_existing_value_is_refused(tmp_path: Path) -> None:
    """Rule: a NEW use is blamed even when the value already occurs once in the file."""
    old = 'a = "Entity"\n'
    found = _problems(tmp_path, {STORE: old}, {STORE: old + 'b = "Entity"\n'})
    assert len(found) == 1 and f"{STORE}:2" in found[0]


def test_a_third_use_with_two_old_ones_gives_exactly_one_problem(tmp_path: Path) -> None:
    """Rule: only occurrences beyond the base count are problems."""
    old = 'a = "Entity"\nb = "Entity"\n'
    found = _problems(tmp_path, {STORE: old}, {STORE: old + 'c = "Entity"\n'})
    assert len(found) == 1


def test_a_removed_literal_is_not_refused(tmp_path: Path) -> None:
    """Rule: taking a literal out never fails the gate."""
    assert _problems(tmp_path, {STORE: 'a = "Entity"\nb = "scope"\n'}, {STORE: "a = 1\n"}) == []


def test_a_literal_renamed_away_is_not_refused(tmp_path: Path) -> None:
    """Rule: replacing a graph literal by a non-graph string is not a problem."""
    assert _problems(tmp_path, {STORE: 'a = "Entity"\n'}, {STORE: 'a = "plain text"\n'}) == []


def test_swapping_one_graph_literal_for_another_is_refused(tmp_path: Path) -> None:
    """Rule: a different graph value is a new literal even if the count of literals is equal."""
    found = _problems(tmp_path, {STORE: 'a = "Entity"\n'}, {STORE: 'a = "scope"\n'})
    assert len(found) == 1 and "'scope'" in found[0]


def test_a_deleted_graph_module_is_refused_with_the_exact_message(tmp_path: Path) -> None:
    """Rule: deleting a file under the graph roots is reported, never a way to pass."""
    found = _problems(tmp_path, {STORE: "x = 1\n", OTHER: "y = 1\n"}, {STORE: None})
    assert found == [f"deleted file in a graph module: {STORE}"]


def test_a_deleted_file_outside_the_roots_is_ignored(tmp_path: Path) -> None:
    """Rule: only graph modules are watched; deleting any other file is not D12's business."""
    before = {"src/memini/other.py": "x = 1\n", STORE: "x = 1\n"}
    assert _problems(tmp_path, before, {"src/memini/other.py": None}) == []


def test_a_deleted_non_python_file_in_the_roots_is_ignored(tmp_path: Path) -> None:
    """Rule: non-.py files are never scanned, deleted or not."""
    before = {"src/memini/infra/graph/notes.txt": "x\n", STORE: "x = 1\n"}
    assert _problems(tmp_path, before, {"src/memini/infra/graph/notes.txt": None}) == []


def test_a_rename_inside_the_roots_flags_the_old_path_and_scans_the_new(tmp_path: Path) -> None:
    """Rule: a rename arrives as D + A: the old path is flagged, the new path fully scanned."""
    text = 'a = "Entity"\n'
    found = _problems(tmp_path, {STORE: text}, {STORE: None, OTHER: text})
    assert f"deleted file in a graph module: {STORE}" in found
    assert any("'Entity'" in f and f"{OTHER}:1" in f for f in found)
    assert len(found) == 2


def test_an_added_file_has_every_literal_refused(tmp_path: Path) -> None:
    """Rule: in an added graph module every graph-name literal counts, repeats included."""
    found = _problems(
        tmp_path, {STORE: "x = 1\n"}, {OTHER: 'a = "Entity"\nb = "Entity"\nc = "scope"\n'}
    )
    assert len(found) == 3


def test_an_added_file_does_not_inherit_literals_of_a_same_named_other_file(
    tmp_path: Path,
) -> None:
    """Rule: the base count is per path; an add never gets a free pass from another file."""
    found = _problems(tmp_path, {STORE: 'a = "Entity"\n'}, {OTHER: 'a = "Entity"\n'})
    assert len(found) == 1


@pytest.mark.parametrize("name", ["data.json", "notes.md", "README", "script.pyi"])
def test_non_python_files_are_ignored(tmp_path: Path, name: str) -> None:
    """Rule: only .py files are scanned, whatever they contain."""
    path = f"src/memini/infra/graph/{name}"
    assert _problems(tmp_path, {STORE: "x = 1\n"}, {path: 'a = "Entity"\n'}) == []


def test_a_modified_file_outside_the_roots_is_ignored(tmp_path: Path) -> None:
    """Rule: files outside the graph roots are never scanned."""
    path = "src/memini/other.py"
    assert _problems(tmp_path, {path: "x = 1\n"}, {path: 'a = "Entity"\n'}) == []


def test_a_new_cypher_parameter_name_in_a_modified_file_is_refused(tmp_path: Path) -> None:
    """Rule: a Cypher parameter name equal to a graph name is a literal too."""
    new = 'q = "MATCH (n) WHERE n.x = $scope"\nparams = {"scope": 1}\n'
    found = _problems(tmp_path, {STORE: "x = 1\n"}, {STORE: new})
    assert len(found) == 1 and "'scope'" in found[0]


def test_a_broken_new_version_is_left_to_the_lint_gates(tmp_path: Path) -> None:
    """Rule: a file that does not parse is not D12's problem (and must not crash it)."""
    assert _problems(tmp_path, {STORE: 'a = "Entity"\n'}, {STORE: "def (:\n"}) == []


def test_a_broken_base_version_counts_as_no_old_literals(tmp_path: Path) -> None:
    """Rule: an unparseable base gives the file no free literals: a fixed file is scanned whole."""
    found = _problems(tmp_path, {STORE: "def (:\n"}, {STORE: 'a = "Entity"\n'})
    assert len(found) == 1
