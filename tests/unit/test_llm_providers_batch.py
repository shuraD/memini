"""Acceptance tests for LP-13: batch generation in parallel (run_many, generate_many)."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Mapping
from typing import Any

import pytest

from memini.infra.llm_providers.batch import run_many
from memini.infra.llm_providers.budgets import Budgets
from memini.infra.llm_providers.chain import ProviderChain
from memini.infra.llm_providers.config import LLMConfig, parse_config
from memini.infra.llm_providers.kind_openai import HttpRequest
from memini.infra.llm_providers.limiter import ConcurrencyLimits
from memini.infra.llm_providers.store import Store
from memini.infra.llm_providers.transport import HttpResponse
from memini.infra.llm_providers.types import (
    Attempt,
    BudgetExceeded,
    Generation,
    LLMError,
    Reason,
)

KEY_ENV_A = "MEMINI_LP13_FAKE_KEY_A"
KEY_ENV_B = "MEMINI_LP13_FAKE_KEY_B"
FAKE_KEY = "test-key"
ENV: Mapping[str, str] = {KEY_ENV_A: FAKE_KEY, KEY_ENV_B: FAKE_KEY}
HOSTS = {"alpha": "alpha.invalid", "beta": "beta.invalid"}
WAIT_S = 5.0
PROMPT_TAG = re.compile(r"prompt-(\d+)")


def clock() -> float:
    return 1000.0


def provider(name: str, key_env: str, limits: Mapping[str, Any] | None) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "name": name,
        "kind": "openai_compatible",
        "key_env": key_env,
        "base_url": f"https://{HOSTS[name]}/v1",
        "models": {"answer": {"name": f"{name}-answer-model"}},
    }
    if limits is not None:
        raw["limits"] = dict(limits)
    return raw


def build_config(*, two: bool = True, alpha_limits: Mapping[str, Any] | None = None) -> LLMConfig:
    providers = [provider("alpha", KEY_ENV_A, alpha_limits)]
    if two:
        providers.append(provider("beta", KEY_ENV_B, None))
    return parse_config({"providers": providers})


def ok(text: str) -> HttpResponse:
    body = json.dumps(
        {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
    ).encode()
    return HttpResponse(status=200, headers={}, body=body)


def rejected() -> HttpResponse:
    return HttpResponse(status=413, headers={}, body=b"request too large")


def provider_of(url: str) -> str:
    for name, host in HOSTS.items():
        if host in url:
            return name
    raise AssertionError(f"unexpected url {url}")


class EchoTransport:
    """Answers each request with 'echo prompt-N' found in its body; thread safe.

    A prompt containing ``fail`` is rejected (413) by every provider.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.log: list[tuple[str, str]] = []
        self.bodies: list[str] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        name = provider_of(request.url)
        body = request.body.decode("utf-8")
        match = PROMPT_TAG.search(body)
        assert match is not None
        tag = match.group(0)
        with self.lock:
            self.log.append((name, tag))
            self.bodies.append(body)
        if "fail" in body:
            return rejected()
        return ok(f"echo {tag}")


class PeakTransport:
    """Records the most sends in flight at once; the first sends wait to overlap."""

    def __init__(self, overlap: int) -> None:
        self.lock = threading.Lock()
        self.inside = 0
        self.peak = 0
        self.calls = 0
        self.overlap = overlap
        self.overlapped = threading.Event()

    def send(self, request: HttpRequest) -> HttpResponse:
        assert HOSTS["alpha"] in request.url
        with self.lock:
            self.inside += 1
            self.calls += 1
            self.peak = max(self.peak, self.inside)
            if self.inside >= self.overlap:
                self.overlapped.set()
        try:
            self.overlapped.wait(WAIT_S)
            return ok("done")
        finally:
            with self.lock:
                self.inside -= 1


def make_chain(config: LLMConfig, transport: Any, store: Store, **kwargs: Any) -> ProviderChain:
    return ProviderChain("answer", config, ENV, clock, transport, store, **kwargs)


@pytest.fixture
def store() -> Store:
    return Store(":memory:")


# --- run_many ---------------------------------------------------------------------------------


def test_run_many_keeps_input_order() -> None:
    items = [f"item-{i}" for i in range(10)]
    assert run_many(str.upper, items, max_workers=4) == [s.upper() for s in items]


def test_run_many_keeps_order_when_later_items_finish_first() -> None:
    count = 6
    items = [str(i) for i in range(count)]
    done = [threading.Event() for _ in range(count)]

    def fn(item: str) -> str:
        index = int(item)
        if index + 1 < count:
            done[index + 1].wait(WAIT_S)
        done[index].set()
        return f"r{index}"

    result = run_many(fn, items, max_workers=count)
    assert result == [f"r{i}" for i in range(count)]


def test_run_many_empty_returns_empty_without_calls() -> None:
    calls: list[str] = []

    def fn(item: str) -> str:
        calls.append(item)
        return item

    assert run_many(fn, [], max_workers=3) == []
    assert calls == []


def test_run_many_llm_error_becomes_the_item_value() -> None:
    error = LLMError((Attempt("alpha", Reason.TIMEOUT, "slow"),), "answer")
    calls: list[str] = []
    lock = threading.Lock()

    def fn(item: str) -> str:
        with lock:
            calls.append(item)
        if item == "b":
            raise error
        return item * 2

    result = run_many(fn, ["a", "b", "c", "d"], max_workers=2)
    assert result[0] == "aa"
    assert result[1] is error
    assert result[2:] == ["cc", "dd"]
    assert sorted(calls) == ["a", "b", "c", "d"]


def test_run_many_budget_exceeded_becomes_the_item_value() -> None:
    def fn(item: str) -> str:
        if item == "x":
            raise BudgetExceeded((Attempt("budget", Reason.BUDGET, "cap"),), "answer")
        return item

    result = run_many(fn, ["x", "y", "z"], max_workers=1)
    assert isinstance(result[0], BudgetExceeded)
    assert result[1:] == ["y", "z"]


def test_run_many_non_llm_error_propagates() -> None:
    class BoomError(RuntimeError):
        pass

    def fn(item: str) -> str:
        if item == "boom":
            raise BoomError("programming error")
        return item

    with pytest.raises(BoomError):
        run_many(fn, ["a", "boom", "c"], max_workers=2)


def test_run_many_runs_items_in_parallel_within_max_workers() -> None:
    lock = threading.Lock()
    state = {"inside": 0, "peak": 0}
    overlapped = threading.Event()

    def fn(item: str) -> str:
        with lock:
            state["inside"] += 1
            state["peak"] = max(state["peak"], state["inside"])
            if state["inside"] >= 2:
                overlapped.set()
        try:
            overlapped.wait(WAIT_S)
            return item
        finally:
            with lock:
                state["inside"] -= 1

    items = [str(i) for i in range(20)]
    assert run_many(fn, items, max_workers=2) == items
    assert state["peak"] == 2


# --- ProviderChain.generate_many --------------------------------------------------------------


def test_generate_many_returns_results_in_input_order(store: Store) -> None:
    transport = EchoTransport()
    chain = make_chain(build_config(), transport, store)
    prompts = [f"question prompt-{i}" for i in range(10)]
    results = chain.generate_many(prompts)
    assert len(results) == 10
    for i, result in enumerate(results):
        assert isinstance(result, Generation)
        assert result.text == f"echo prompt-{i}"
        assert result.role == "answer"
    assert sorted(tag for _, tag in transport.log) == sorted(f"prompt-{i}" for i in range(10))


def test_generate_many_passes_the_same_kwargs_to_each_prompt(store: Store) -> None:
    transport = EchoTransport()
    chain = make_chain(build_config(), transport, store)
    results = chain.generate_many(
        ["q prompt-0", "q prompt-1", "q prompt-2"], system="shared system text", temperature=0.0
    )
    assert all(isinstance(r, Generation) for r in results)
    assert len(transport.bodies) == 3
    assert all("shared system text" in body for body in transport.bodies)


def test_generate_many_respects_max_concurrent_of_the_only_provider(store: Store) -> None:
    config = build_config(two=False, alpha_limits={"max_concurrent": 3})
    transport = PeakTransport(overlap=2)
    chain = make_chain(config, transport, store, limits=ConcurrencyLimits(config))
    results = chain.generate_many([f"p prompt-{i}" for i in range(32)])
    assert len(results) == 32
    assert all(isinstance(r, Generation) for r in results)
    assert transport.calls == 32
    assert 2 <= transport.peak <= 3


def test_generate_many_failing_prompt_is_llm_error_others_are_generations(store: Store) -> None:
    transport = EchoTransport()
    chain = make_chain(build_config(), transport, store)
    prompts = ["ok prompt-0", "ok prompt-1", "fail prompt-2", "ok prompt-3", "ok prompt-4"]
    results = chain.generate_many(prompts)
    assert len(results) == 5
    failed = results[2]
    assert isinstance(failed, LLMError)
    assert not isinstance(failed, BudgetExceeded)
    assert {a.provider for a in failed.attempts} == {"alpha", "beta"}
    for i in (0, 1, 3, 4):
        result = results[i]
        assert isinstance(result, Generation)
        assert result.text == f"echo prompt-{i}"
    assert sorted(name for name, tag in transport.log if tag == "prompt-2") == ["alpha", "beta"]


def test_generate_many_empty_returns_empty_with_no_calls(store: Store) -> None:
    transport = EchoTransport()
    chain = make_chain(build_config(), transport, store)
    assert chain.generate_many([]) == []
    assert transport.log == []


class FirstCheckFailsBudgets:
    """A budget whose first check is over the cap; every later check passes."""

    def __init__(self, config: LLMConfig, store: Store) -> None:
        self.inner = Budgets(config, store)
        self.lock = threading.Lock()
        self.checks = 0

    def check(self, role: str) -> None:
        with self.lock:
            self.checks += 1
            first = self.checks == 1
        if first:
            raise BudgetExceeded((Attempt("budget", Reason.BUDGET, "cap reached"),), role)

    def record(self, role: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.inner.record(role, prompt_tokens, completion_tokens)


def test_generate_many_budget_error_is_per_item_and_later_items_still_try(store: Store) -> None:
    config = build_config()
    budgets = FirstCheckFailsBudgets(config, store)
    transport = EchoTransport()
    chain = make_chain(config, transport, store, budgets=budgets)
    prompts = [f"b prompt-{i}" for i in range(6)]
    results = chain.generate_many(prompts)
    assert len(results) == 6
    assert budgets.checks == 6
    over = [r for r in results if isinstance(r, BudgetExceeded)]
    assert len(over) == 1
    generations = [r for r in results if isinstance(r, Generation)]
    assert len(generations) == 5
    for i, result in enumerate(results):
        if isinstance(result, Generation):
            assert result.text == f"echo prompt-{i}"
    assert len(transport.log) == 5


class BrokenBudgets:
    """A budget whose check fails with a programming error."""

    def check(self, role: str) -> None:
        raise ZeroDivisionError("bug in the budget code")

    def record(self, role: str, prompt_tokens: int, completion_tokens: int) -> None:
        raise AssertionError("record must not be reached")


def test_generate_many_non_llm_error_propagates(store: Store) -> None:
    transport = EchoTransport()
    chain = make_chain(build_config(), transport, store, budgets=BrokenBudgets())
    with pytest.raises(ZeroDivisionError):
        chain.generate_many(["x prompt-0", "x prompt-1"])
