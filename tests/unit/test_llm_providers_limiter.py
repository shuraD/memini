"""Acceptance tests for LP-10: per-provider concurrency limits shared by every role."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import pytest

from memini.infra.llm_providers import limiter as lp_limiter
from memini.infra.llm_providers.config import LLMConfig, parse_config

ConcurrencyLimits = lp_limiter.ConcurrencyLimits

WAIT_S = 10.0
"""Generous upper bound for anything that must happen."""

BLOCK_S = 0.3
"""How long a thread that must stay blocked is watched."""


def _config(**limits: dict[str, Any]) -> LLMConfig:
    providers = []
    for name, provider_limits in limits.items():
        providers.append(
            {
                "name": name,
                "kind": "openai_compatible",
                "key_env": f"{name.upper()}_KEY_VAR",
                "base_url": f"https://{name}.example.test/v1",
                "limits": provider_limits,
                "models": {
                    "answer": {"name": f"{name}-answer", "limits": {"max_concurrent": 50}},
                    "classifier": {"name": f"{name}-classifier"},
                },
            }
        )
    return parse_config({"providers": providers})


class _Holder:
    """A thread that enters ``slot(provider)`` and stays inside until released."""

    def __init__(self, limits: lp_limiter.ConcurrencyLimits, provider: str) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, args=(limits, provider), daemon=True)
        self._thread.start()

    def _run(self, limits: lp_limiter.ConcurrencyLimits, provider: str) -> None:
        try:
            with limits.slot(provider):
                self.entered.set()
                self.release.wait(WAIT_S)
        except BaseException as exc:  # recorded and asserted by the test
            self.error = exc

    def finish(self) -> None:
        self.release.set()
        self._thread.join(WAIT_S)
        assert not self._thread.is_alive()
        assert self.error is None


def _hold(limits: lp_limiter.ConcurrencyLimits, provider: str) -> _Holder:
    holder = _Holder(limits, provider)
    assert holder.entered.wait(WAIT_S), f"could not enter a free slot of {provider!r}"
    return holder


def _wait_until(predicate: Callable[[], bool]) -> bool:
    done = threading.Event()
    for _ in range(int(WAIT_S / 0.01)):
        if predicate():
            return True
        done.wait(0.01)
    return predicate()


def test_slot_returns_a_context_manager_yielding_none() -> None:
    limits = ConcurrencyLimits(_config(alpha={"max_concurrent": 2}))
    with limits.slot("alpha") as value:
        assert value is None


def test_at_most_max_concurrent_threads_inside_slot() -> None:
    limits = ConcurrencyLimits(_config(alpha={"max_concurrent": 3}))
    lock = threading.Lock()
    state = {"inside": 0, "peak": 0, "done": 0}
    gate = threading.Event()
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            with limits.slot("alpha"):
                with lock:
                    state["inside"] += 1
                    state["peak"] = max(state["peak"], state["inside"])
                gate.wait(WAIT_S)
                with lock:
                    state["inside"] -= 1
            with lock:
                state["done"] += 1
        except BaseException as exc:  # asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(32)]
    for thread in threads:
        thread.start()
    assert _wait_until(lambda: state["inside"] == 3)
    threading.Event().wait(BLOCK_S)
    with lock:
        assert state["inside"] == 3
        assert state["peak"] == 3
    gate.set()
    for thread in threads:
        thread.join(WAIT_S)
    assert not any(t.is_alive() for t in threads)
    assert errors == []
    assert state["done"] == 32
    assert state["peak"] == 3
    assert state["inside"] == 0


def test_a_full_provider_blocks_the_next_caller_until_a_slot_frees() -> None:
    limits = ConcurrencyLimits(_config(alpha={"max_concurrent": 1}))
    first = _hold(limits, "alpha")
    second = _Holder(limits, "alpha")
    assert not second.entered.wait(BLOCK_S)
    first.finish()
    assert second.entered.wait(WAIT_S)
    second.finish()


def test_roles_share_the_provider_limit_not_their_model_limits() -> None:
    # The models of every role declare max_concurrent=50; the provider says 1.
    config = _config(alpha={"max_concurrent": 1})
    assert config.limit("alpha", "answer", "max_concurrent") == 50
    limits = ConcurrencyLimits(config)
    answer_role = _hold(limits, "alpha")
    classifier_role = _Holder(limits, "alpha")
    assert not classifier_role.entered.wait(BLOCK_S)
    answer_role.finish()
    assert classifier_role.entered.wait(WAIT_S)
    classifier_role.finish()


def test_each_provider_limit_is_applied_separately() -> None:
    limits = ConcurrencyLimits(
        _config(alpha={"max_concurrent": 1}, beta={"max_concurrent": 2}, gamma={})
    )
    alpha = _hold(limits, "alpha")
    beta_one = _hold(limits, "beta")
    beta_two = _hold(limits, "beta")
    gamma = _hold(limits, "gamma")
    alpha_blocked = _Holder(limits, "alpha")
    beta_blocked = _Holder(limits, "beta")
    assert not alpha_blocked.entered.wait(BLOCK_S)
    assert not beta_blocked.entered.wait(BLOCK_S)

    beta_one.finish()
    assert beta_blocked.entered.wait(WAIT_S)
    assert not alpha_blocked.entered.is_set()

    alpha.finish()
    assert alpha_blocked.entered.wait(WAIT_S)
    for holder in (beta_two, beta_blocked, alpha_blocked, gamma):
        holder.finish()


def test_no_max_concurrent_key_never_blocks() -> None:
    limits = ConcurrencyLimits(_config(alpha={"timeout_s": 30}, beta={"max_concurrent": 1}))
    count = 40
    barrier = threading.Barrier(count)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            with limits.slot("alpha"):
                barrier.wait(WAIT_S)  # every thread must be inside at once
        except BaseException as exc:  # asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT_S * 2)
    assert not any(t.is_alive() for t in threads)
    assert errors == []


def test_unlimited_slot_nests_in_one_thread() -> None:
    limits = ConcurrencyLimits(_config(alpha={}))
    reached = False
    with limits.slot("alpha"), limits.slot("alpha"), limits.slot("alpha"):
        reached = True
    assert reached


def test_exception_inside_slot_releases_the_slot() -> None:
    limits = ConcurrencyLimits(_config(alpha={"max_concurrent": 1}))

    class BoomError(RuntimeError):
        pass

    for _ in range(3):
        with pytest.raises(BoomError), limits.slot("alpha"):
            raise BoomError("call failed")

    holder = _hold(limits, "alpha")
    holder.finish()


def test_no_lock_is_held_while_the_body_runs() -> None:
    limits = ConcurrencyLimits(
        _config(alpha={"max_concurrent": 1}, beta={"max_concurrent": 1}, gamma={})
    )
    # A body is running on alpha; a first call on another provider (which creates
    # its registry entry) and a call on an unlimited provider must not wait for it.
    alpha = _hold(limits, "alpha")
    beta = _hold(limits, "beta")
    gamma = _hold(limits, "gamma")
    for holder in (beta, gamma, alpha):
        holder.finish()


def test_concurrent_first_use_creates_one_shared_semaphore() -> None:
    # Every round starts many threads at once on a provider never used before:
    # if the registry were not created under a lock, two semaphores could exist.
    for _ in range(5):
        limits = ConcurrencyLimits(_config(alpha={"max_concurrent": 1}))
        count = 16
        start = threading.Barrier(count)
        lock = threading.Lock()
        state = {"inside": 0, "peak": 0}
        errors: list[BaseException] = []

        def worker(
            limits: lp_limiter.ConcurrencyLimits = limits,
            start: threading.Barrier = start,
            lock: threading.Lock = lock,
            state: dict[str, int] = state,
            errors: list[BaseException] = errors,
        ) -> None:
            try:
                start.wait(WAIT_S)
                with limits.slot("alpha"):
                    with lock:
                        state["inside"] += 1
                        state["peak"] = max(state["peak"], state["inside"])
                    threading.Event().wait(0.002)
                    with lock:
                        state["inside"] -= 1
            except BaseException as exc:  # asserted below
                errors.append(exc)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(WAIT_S)
        assert not any(t.is_alive() for t in threads)
        assert errors == []
        assert state["peak"] == 1
