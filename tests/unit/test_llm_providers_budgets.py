"""Acceptance tests for LP-11: per-role, per-question and per-run budgets.

Each simulated call is ``check(role)`` followed by ``record(role, ...)``, so the tests
hold whether calls are counted on check or on record.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from memini.infra.llm_providers.budgets import Budgets, current_question, question
from memini.infra.llm_providers.config import LLMConfig, parse_config
from memini.infra.llm_providers.store import Store
from memini.infra.llm_providers.types import BudgetExceeded, LLMError, Reason


def _config(
    roles: dict[str, Any] | None = None,
    global_tokens: int | None = None,
    run_id: str = "run-1",
) -> LLMConfig:
    raw: dict[str, Any] = {
        "run_id": run_id,
        "providers": [
            {
                "name": "p1",
                "kind": "openai_compatible",
                "base_url": "http://127.0.0.1:1/v1",
                "models": {"answer": {"name": "m-answer"}, "judge": {"name": "m-judge"}},
            }
        ],
    }
    if roles is not None:
        raw["roles"] = roles
    if global_tokens is not None:
        raw["budget"] = {"max_tokens_per_run": global_tokens}
    return parse_config(raw)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    s = Store(tmp_path / "budgets.sqlite")
    yield s
    s.close()


def _call(budgets: Budgets, role: str, prompt: int = 1, completion: int = 1) -> None:
    budgets.check(role)
    budgets.record(role, prompt, completion)


# --- question() scope -------------------------------------------------------------


def test_current_question_is_none_outside_any_scope() -> None:
    assert current_question() is None


def test_question_scope_sets_and_restores_current_question() -> None:
    with question("q-1"):
        assert current_question() == "q-1"
    assert current_question() is None


def test_nested_question_scope_restores_outer_id() -> None:
    with question("outer"):
        with question("inner"):
            assert current_question() == "inner"
        assert current_question() == "outer"
    assert current_question() is None


def test_question_scope_restored_when_body_raises() -> None:
    with pytest.raises(RuntimeError), question("q-err"):
        raise RuntimeError("boom")
    assert current_question() is None


# --- per-question call budget -----------------------------------------------------


def test_third_call_in_same_question_raises(store: Store) -> None:
    budgets = Budgets(_config({"answer": {"max_calls_per_question": 2}}), store)
    with question("q-1"):
        _call(budgets, "answer")
        _call(budgets, "answer")
        with pytest.raises(BudgetExceeded):
            budgets.check("answer")


def test_new_question_starts_again_at_zero(store: Store) -> None:
    budgets = Budgets(_config({"answer": {"max_calls_per_question": 2}}), store)
    with question("q-1"):
        _call(budgets, "answer")
        _call(budgets, "answer")
    with question("q-2"):
        _call(budgets, "answer")
        _call(budgets, "answer")
        with pytest.raises(BudgetExceeded):
            budgets.check("answer")


def test_per_question_budget_is_per_role(store: Store) -> None:
    budgets = Budgets(
        _config({"answer": {"max_calls_per_question": 1}, "judge": {"max_calls_per_question": 1}}),
        store,
    )
    with question("q-1"):
        _call(budgets, "answer")
        _call(budgets, "judge")
        with pytest.raises(BudgetExceeded):
            budgets.check("answer")
        with pytest.raises(BudgetExceeded):
            budgets.check("judge")


def test_budget_exceeded_is_llm_error_with_role_and_budget_reason(store: Store) -> None:
    budgets = Budgets(_config({"answer": {"max_calls_per_question": 1}}), store)
    with question("q-1"):
        _call(budgets, "answer")
        with pytest.raises(BudgetExceeded) as info:
            budgets.check("answer")
    err = info.value
    assert isinstance(err, LLMError)
    assert err.role == "answer"
    assert err.attempts
    assert all(a.reason is Reason.BUDGET for a in err.attempts)


def test_check_alone_does_not_record_tokens(store: Store) -> None:
    budgets = Budgets(_config({"answer": {"max_tokens_per_run": 10}}), store)
    for _ in range(20):
        budgets.check("answer")
    budgets.record("answer", 5, 4)
    budgets.check("answer")


# --- per-role run token budget ----------------------------------------------------


def test_role_tokens_below_limit_do_not_raise(store: Store) -> None:
    budgets = Budgets(_config({"answer": {"max_tokens_per_run": 100}}), store)
    budgets.check("answer")
    budgets.record("answer", 60, 39)
    budgets.check("answer")


def test_role_tokens_reaching_limit_make_next_check_raise(store: Store) -> None:
    budgets = Budgets(_config({"answer": {"max_tokens_per_run": 100}}), store)
    budgets.check("answer")
    budgets.record("answer", 60, 40)
    with pytest.raises(BudgetExceeded):
        budgets.check("answer")


def test_role_tokens_sum_prompt_and_completion_across_calls(store: Store) -> None:
    budgets = Budgets(_config({"answer": {"max_tokens_per_run": 100}}), store)
    _call(budgets, "answer", 30, 20)
    budgets.check("answer")
    budgets.record("answer", 0, 50)
    with pytest.raises(BudgetExceeded):
        budgets.check("answer")


def test_role_token_budget_spans_questions(store: Store) -> None:
    budgets = Budgets(_config({"answer": {"max_tokens_per_run": 100}}), store)
    with question("q-1"):
        _call(budgets, "answer", 50, 0)
    with question("q-2"):
        _call(budgets, "answer", 50, 0)
    with question("q-3"), pytest.raises(BudgetExceeded):
        budgets.check("answer")


def test_role_token_budget_does_not_stop_other_roles(store: Store) -> None:
    budgets = Budgets(_config({"answer": {"max_tokens_per_run": 10}}), store)
    _call(budgets, "answer", 10, 0)
    with pytest.raises(BudgetExceeded):
        budgets.check("answer")
    budgets.check("judge")


# --- global run token budget ------------------------------------------------------


def test_global_budget_sum_over_roles_stops_every_role(store: Store) -> None:
    budgets = Budgets(_config(global_tokens=100), store)
    _call(budgets, "answer", 30, 20)
    budgets.check("judge")
    _call(budgets, "judge", 25, 25)
    for role in ("answer", "judge", "classifier", "summary"):
        with pytest.raises(BudgetExceeded):
            budgets.check(role)


def test_global_budget_below_limit_does_not_raise(store: Store) -> None:
    budgets = Budgets(_config(global_tokens=100), store)
    _call(budgets, "answer", 30, 20)
    _call(budgets, "judge", 25, 24)
    budgets.check("answer")
    budgets.check("judge")


# --- shared store across Budgets objects ------------------------------------------


def test_two_budgets_on_one_file_share_role_token_budget(tmp_path: Path) -> None:
    path = tmp_path / "shared.sqlite"
    config = _config({"answer": {"max_tokens_per_run": 100}})
    store_a, store_b = Store(path), Store(path)
    try:
        a, b = Budgets(config, store_a), Budgets(config, store_b)
        _call(a, "answer", 30, 20)
        _call(b, "answer", 30, 20)
        with pytest.raises(BudgetExceeded):
            a.check("answer")
        with pytest.raises(BudgetExceeded):
            b.check("answer")
    finally:
        store_a.close()
        store_b.close()


def test_two_budgets_on_one_file_share_global_budget(tmp_path: Path) -> None:
    path = tmp_path / "shared.sqlite"
    config = _config(global_tokens=100)
    store_a, store_b = Store(path), Store(path)
    try:
        a, b = Budgets(config, store_a), Budgets(config, store_b)
        _call(a, "answer", 50, 0)
        _call(b, "judge", 0, 50)
        with pytest.raises(BudgetExceeded):
            a.check("judge")
        with pytest.raises(BudgetExceeded):
            b.check("answer")
    finally:
        store_a.close()
        store_b.close()


def test_two_budgets_on_one_file_share_per_question_count(tmp_path: Path) -> None:
    path = tmp_path / "shared.sqlite"
    config = _config({"answer": {"max_calls_per_question": 2}})
    store_a, store_b = Store(path), Store(path)
    try:
        a, b = Budgets(config, store_a), Budgets(config, store_b)
        with question("q-1"):
            _call(a, "answer")
            _call(b, "answer")
            with pytest.raises(BudgetExceeded):
                a.check("answer")
            with pytest.raises(BudgetExceeded):
                b.check("answer")
    finally:
        store_a.close()
        store_b.close()


def test_spending_persists_across_reopen_of_store(tmp_path: Path) -> None:
    path = tmp_path / "shared.sqlite"
    config = _config({"answer": {"max_tokens_per_run": 100}})
    first = Store(path)
    try:
        _call(Budgets(config, first), "answer", 100, 0)
    finally:
        first.close()
    second = Store(path)
    try:
        with pytest.raises(BudgetExceeded):
            Budgets(config, second).check("answer")
    finally:
        second.close()


# --- threads in different question scopes -----------------------------------------


def test_threads_in_different_questions_keep_separate_counts(store: Store) -> None:
    budgets = Budgets(_config({"answer": {"max_calls_per_question": 2}}), store)
    barrier = threading.Barrier(2)
    results: dict[str, list[bool]] = {}
    seen: dict[str, str | None] = {}

    def worker(qid: str) -> None:
        outcome: list[bool] = []
        with question(qid):
            for _ in range(3):
                barrier.wait(timeout=10)
                seen[qid] = current_question()
                try:
                    budgets.check("answer")
                except BudgetExceeded:
                    outcome.append(False)
                else:
                    budgets.record("answer", 1, 1)
                    outcome.append(True)
        results[qid] = outcome

    threads = [threading.Thread(target=worker, args=(q,)) for q in ("q-a", "q-b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert seen == {"q-a": "q-a", "q-b": "q-b"}
    assert results == {"q-a": [True, True, False], "q-b": [True, True, False]}


def test_question_scope_does_not_leak_into_other_thread() -> None:
    seen: list[str | None] = []
    with question("main-q"):
        t = threading.Thread(target=lambda: seen.append(current_question()))
        t.start()
        t.join(timeout=10)
        assert current_question() == "main-q"
    assert seen == [None]


# --- calls outside any question ---------------------------------------------------


def test_calls_outside_question_ignore_per_question_budget(store: Store) -> None:
    budgets = Budgets(_config({"answer": {"max_calls_per_question": 1}}), store)
    for _ in range(5):
        _call(budgets, "answer")


def test_calls_outside_question_do_not_count_toward_a_question(store: Store) -> None:
    budgets = Budgets(_config({"answer": {"max_calls_per_question": 1}}), store)
    _call(budgets, "answer")
    _call(budgets, "answer")
    with question("q-1"):
        _call(budgets, "answer")
        with pytest.raises(BudgetExceeded):
            budgets.check("answer")


def test_calls_outside_question_still_count_toward_run_budgets(store: Store) -> None:
    budgets = Budgets(
        _config({"answer": {"max_calls_per_question": 5, "max_tokens_per_run": 10}}), store
    )
    _call(budgets, "answer", 10, 0)
    with pytest.raises(BudgetExceeded):
        budgets.check("answer")
    with question("q-1"), pytest.raises(BudgetExceeded):
        budgets.check("answer")


def test_calls_outside_question_count_toward_global_budget(store: Store) -> None:
    budgets = Budgets(_config(global_tokens=10), store)
    _call(budgets, "answer", 5, 5)
    with pytest.raises(BudgetExceeded):
        budgets.check("judge")


# --- no budget configured ---------------------------------------------------------


def test_no_budget_configured_never_raises(store: Store) -> None:
    budgets = Budgets(_config(), store)
    for _ in range(50):
        _call(budgets, "answer", 1_000_000, 1_000_000)
    with question("q-1"):
        for _ in range(50):
            _call(budgets, "judge", 1_000_000, 1_000_000)
    budgets.check("answer")
    budgets.check("judge")


def test_role_with_section_but_no_caps_never_raises(store: Store) -> None:
    budgets = Budgets(_config({"answer": {"providers": ["p1"]}}), store)
    with question("q-1"):
        for _ in range(20):
            _call(budgets, "answer", 10_000, 10_000)
        budgets.check("answer")


def test_works_with_in_memory_store() -> None:
    mem = Store(":memory:")
    try:
        budgets = Budgets(_config({"answer": {"max_calls_per_question": 1}}), mem)
        with question("q-1"):
            _call(budgets, "answer")
            with pytest.raises(BudgetExceeded):
                budgets.check("answer")
    finally:
        mem.close()
