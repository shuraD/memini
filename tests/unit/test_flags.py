"""foundation: module feature flags (owner decision 2026-11-14, ADR-0004).

One flag per unshipped engine module, read from a ``MEMINI_ENABLE_<MODULE>`` environment
variable at call time, OFF by default. No database in these tests.
"""

import ast
import inspect

import pytest

from memini.infra import flags

FLAG_FUNCTIONS = {
    "document_authority": "MEMINI_ENABLE_DOC_AUTH",
    "kg_extraction": "MEMINI_ENABLE_KG_EXTRACTION",
    "archrag_index": "MEMINI_ENABLE_ARCHRAG_INDEX",
    "archrag_retrieval": "MEMINI_ENABLE_ARCHRAG_RETRIEVAL",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in FLAG_FUNCTIONS.values():
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("function", sorted(FLAG_FUNCTIONS))
def test_every_module_flag_is_off_by_default(function: str) -> None:
    assert getattr(flags, function)() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
@pytest.mark.parametrize(("function", "variable"), sorted(FLAG_FUNCTIONS.items()))
def test_a_true_value_turns_the_flag_on(
    monkeypatch: pytest.MonkeyPatch, function: str, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)
    assert getattr(flags, function)() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "maybe"])
def test_any_other_value_leaves_the_flag_off(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("MEMINI_ENABLE_KG_EXTRACTION", value)
    assert flags.kg_extraction() is False


def test_the_flag_is_read_at_call_time_not_at_import(monkeypatch: pytest.MonkeyPatch) -> None:
    assert flags.archrag_index() is False
    monkeypatch.setenv("MEMINI_ENABLE_ARCHRAG_INDEX", "1")
    assert flags.archrag_index() is True
    monkeypatch.delenv("MEMINI_ENABLE_ARCHRAG_INDEX")
    assert flags.archrag_index() is False


def test_each_flag_variable_is_a_literal_env_bool_call() -> None:
    """The orchestrator finds the variables by parsing ``_env_bool("NAME"`` calls."""
    tree = ast.parse(inspect.getsource(flags))
    found = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_env_bool"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert found == set(FLAG_FUNCTIONS.values())


def test_the_flags_module_imports_only_the_standard_library() -> None:
    tree = ast.parse(inspect.getsource(flags))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imported <= {"__future__", "os"}
