"""Acceptance tests for LP-12: the provider chain of one role, with fallback."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from memini.infra.llm_providers import chain as chain_module
from memini.infra.llm_providers.budgets import Budgets
from memini.infra.llm_providers.chain import ProviderChain
from memini.infra.llm_providers.config import LLMConfig, parse_config
from memini.infra.llm_providers.health import ProviderHealth
from memini.infra.llm_providers.kind_openai import HttpRequest
from memini.infra.llm_providers.store import Store
from memini.infra.llm_providers.transport import HttpResponse, TransportTimeout
from memini.infra.llm_providers.types import BudgetExceeded, Generation, LLMError, Reason

KEY_ENV_A = "MEMINI_LP12_FAKE_KEY_A"
KEY_ENV_B = "MEMINI_LP12_FAKE_KEY_B"
KEY_ENV_C = "MEMINI_LP12_FAKE_KEY_C"
FAKE_KEY = "test-key"
ENV: Mapping[str, str] = {KEY_ENV_A: FAKE_KEY, KEY_ENV_B: FAKE_KEY, KEY_ENV_C: FAKE_KEY}

HOSTS = {"alpha": "alpha.invalid", "beta": "beta.invalid", "gamma": "gamma.invalid"}


def ok_body(provider: str, prompt_tokens: int = 11, completion_tokens: int = 7) -> bytes:
    return json.dumps(
        {
            "choices": [{"message": {"content": f"from {provider}"}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }
    ).encode()


def resp(status: int, body: bytes = b"", headers: Mapping[str, str] | None = None) -> HttpResponse:
    return HttpResponse(status=status, headers=dict(headers or {}), body=body)


def ok(provider: str) -> HttpResponse:
    return resp(200, ok_body(provider))


class FakeClock:
    """A clock the test moves by hand."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class RoutingTransport:
    """Routes each request to the scripted answers of the provider owning its host."""

    def __init__(self, scripts: Mapping[str, list[HttpResponse | BaseException]]) -> None:
        self.scripts = {name: list(items) for name, items in scripts.items()}
        self.log: list[str] = []
        self.requests: list[HttpRequest] = []

    def _provider(self, url: str) -> str:
        for name, host in HOSTS.items():
            if host in url:
                return name
        raise AssertionError(f"unexpected url {url}")

    def send(self, request: HttpRequest) -> HttpResponse:
        name = self._provider(request.url)
        self.log.append(name)
        self.requests.append(request)
        script = self.scripts.get(name, [])
        if not script:
            raise AssertionError(f"{name} called more often than scripted")
        item = script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def calls(self, name: str) -> int:
        return self.log.count(name)


def provider(name: str, key_env: str, roles: tuple[str, ...]) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "openai_compatible",
        "key_env": key_env,
        "base_url": f"https://{HOSTS[name]}/v1",
        "models": {role: {"name": f"{name}-{role}-model"} for role in roles},
    }


def build_config(roles: Mapping[str, Any] | None = None, **top: Any) -> LLMConfig:
    raw: dict[str, Any] = {
        "providers": [
            provider("alpha", KEY_ENV_A, ("answer", "judge", "summary", "extraction")),
            provider("beta", KEY_ENV_B, ("answer", "judge", "summary", "extraction")),
            provider("gamma", KEY_ENV_C, ("answer", "summary")),
        ],
        "roles": dict(
            roles
            if roles is not None
            else {
                "judge": {"providers": ["beta", "alpha"]},
                "extraction": {"providers": ["gamma", "beta"]},
            }
        ),
    }
    raw.update(top)
    return parse_config(raw)


def make_chain(
    role: str,
    config: LLMConfig,
    transport: RoutingTransport,
    store: Store,
    clock: Callable[[], float],
    *,
    env: Mapping[str, str] = ENV,
    **kwargs: Any,
) -> ProviderChain:
    return ProviderChain(role, config, env, clock, transport, store, **kwargs)


@pytest.fixture
def store() -> Store:
    return Store(":memory:")


# --- first provider answers ------------------------------------------------------------


def test_first_provider_answers_and_second_gets_zero_calls(store: Store) -> None:
    config = build_config()
    transport = RoutingTransport({"alpha": [ok("alpha")], "beta": [ok("beta")]})
    chain = make_chain("answer", config, transport, store, FakeClock())
    gen = chain.generate("what is the answer?")
    assert isinstance(gen, Generation)
    assert gen.text == "from alpha"
    assert gen.provider == "alpha"
    assert gen.model == "alpha-answer-model"
    assert gen.role == "answer"
    assert (gen.prompt_tokens, gen.completion_tokens) == (11, 7)
    assert transport.calls("alpha") == 1
    assert transport.calls("beta") == 0
    assert transport.calls("gamma") == 0


def test_prompt_and_system_reach_the_provider(store: Store) -> None:
    config = build_config()
    transport = RoutingTransport({"alpha": [ok("alpha")]})
    chain = make_chain("answer", config, transport, store, FakeClock())
    chain.generate("unique prompt text", system="unique system text")
    body = transport.requests[0].body.decode("utf-8")
    assert "unique prompt text" in body
    assert "unique system text" in body


# --- order --------------------------------------------------------------------------------


def test_role_order_tries_its_first_provider_first(store: Store) -> None:
    config = build_config()
    transport = RoutingTransport({"alpha": [ok("alpha")], "beta": [ok("beta")]})
    chain = make_chain("judge", config, transport, store, FakeClock())
    gen = chain.generate("judge this")
    assert gen.provider == "beta"
    assert gen.role == "judge"
    assert transport.calls("alpha") == 0


def test_role_order_falls_through_in_its_own_order(store: Store) -> None:
    config = build_config()
    transport = RoutingTransport({"alpha": [ok("alpha")], "beta": [resp(413, b"too large")]})
    chain = make_chain("judge", config, transport, store, FakeClock())
    gen = chain.generate("judge this")
    assert gen.provider == "alpha"
    assert transport.log == ["beta", "alpha"]


def test_role_without_order_uses_global_order(store: Store) -> None:
    config = build_config()
    transport = RoutingTransport(
        {
            "alpha": [resp(413, b"too large")],
            "beta": [resp(413, b"too large")],
            "gamma": [ok("gamma")],
        }
    )
    chain = make_chain("summary", config, transport, store, FakeClock())
    gen = chain.generate("summarise")
    assert gen.provider == "gamma"
    assert transport.log == ["alpha", "beta", "gamma"]


# --- per-minute rest ------------------------------------------------------------------------


def test_per_minute_429_falls_through_rests_then_returns(store: Store) -> None:
    config = build_config()
    clock = FakeClock(1000.0)
    transport = RoutingTransport(
        {
            "alpha": [resp(429, b"rate limited", {"Retry-After": "30"}), ok("alpha")],
            "beta": [ok("beta"), ok("beta")],
        }
    )
    chain = make_chain("answer", config, transport, store, clock)

    first = chain.generate("first")
    assert first.provider == "beta"
    assert transport.log == ["alpha", "beta"]

    clock.now = 1010.0
    second = chain.generate("second")
    assert second.provider == "beta"
    assert transport.calls("alpha") == 1  # skipped while resting

    clock.now = 1031.0
    third = chain.generate("third")
    assert third.provider == "alpha"
    assert transport.calls("alpha") == 2
    assert transport.calls("beta") == 2


def test_per_minute_rest_uses_the_wait_stated_in_the_body(store: Store) -> None:
    config = build_config()
    clock = FakeClock(500.0)
    transport = RoutingTransport(
        {
            "alpha": [resp(429, b"Rate limit reached. Please try again in 20s."), ok("alpha")],
            "beta": [ok("beta"), ok("beta")],
        }
    )
    chain = make_chain("answer", config, transport, store, clock)
    assert chain.generate("first").provider == "beta"
    clock.now = 515.0
    assert chain.generate("second").provider == "beta"
    assert transport.calls("alpha") == 1
    clock.now = 521.0
    assert chain.generate("third").provider == "alpha"


# --- exhausted for the run, across roles ---------------------------------------------------


EXHAUSTING = [
    pytest.param(resp(429, b"Rate limit reached: tokens per day exceeded"), id="429-per-day"),
    pytest.param(resp(402, b"payment required"), id="402"),
    pytest.param(resp(401, b"unauthorized"), id="401"),
]


@pytest.mark.parametrize("failure", EXHAUSTING)
def test_exhausted_provider_is_skipped_by_another_role(store: Store, failure: HttpResponse) -> None:
    config = build_config()
    clock = FakeClock(1000.0)
    transport = RoutingTransport(
        {"alpha": [failure, ok("alpha")], "beta": [ok("beta"), ok("beta")]}
    )
    answer = make_chain("answer", config, transport, store, clock)
    assert answer.generate("first").provider == "beta"
    assert transport.calls("alpha") == 1

    clock.now = 1000.0 + 100_000.0  # exhausted holds for the run, whatever the clock says
    summary = make_chain("summary", config, transport, store, clock)
    gen = summary.generate("second")
    assert gen.provider == "beta"
    assert gen.role == "summary"
    assert transport.calls("alpha") == 1


def test_exhausted_mark_is_shared_through_an_explicit_health(store: Store) -> None:
    config = build_config()
    clock = FakeClock(1000.0)
    health = ProviderHealth(store, clock, run_id=config.run_id)
    transport = RoutingTransport(
        {"alpha": [resp(402, b"payment required"), ok("alpha")], "beta": [ok("beta"), ok("beta")]}
    )
    answer = make_chain("answer", config, transport, store, clock, health=health)
    assert answer.generate("first").provider == "beta"
    assert health.available("alpha") is False

    summary = make_chain("summary", config, transport, store, clock, health=health)
    assert summary.generate("second").provider == "beta"
    assert transport.calls("alpha") == 1


# --- retry on 5xx / timeout, request too large ----------------------------------------------


def test_server_error_retries_once_then_next(store: Store) -> None:
    config = build_config()
    transport = RoutingTransport(
        {"alpha": [resp(503, b"unavailable"), resp(503, b"unavailable")], "beta": [ok("beta")]}
    )
    chain = make_chain("answer", config, transport, store, FakeClock())
    gen = chain.generate("hello")
    assert gen.provider == "beta"
    assert transport.log == ["alpha", "alpha", "beta"]


def test_timeout_retries_once_then_next(store: Store) -> None:
    config = build_config()
    transport = RoutingTransport(
        {
            "alpha": [TransportTimeout("timed out"), TransportTimeout("timed out")],
            "beta": [ok("beta")],
        }
    )
    chain = make_chain("answer", config, transport, store, FakeClock())
    gen = chain.generate("hello")
    assert gen.provider == "beta"
    assert transport.log == ["alpha", "alpha", "beta"]


def test_request_too_large_skips_for_that_request_only(store: Store) -> None:
    config = build_config()
    transport = RoutingTransport(
        {"alpha": [resp(413, b"request too large"), ok("alpha")], "beta": [ok("beta")]}
    )
    chain = make_chain("answer", config, transport, store, FakeClock())
    first = chain.generate("a very long prompt")
    assert first.provider == "beta"
    assert transport.log == ["alpha", "beta"]

    second = chain.generate("short prompt")
    assert second.provider == "alpha"
    assert transport.log == ["alpha", "beta", "alpha"]


# --- missing key or model ---------------------------------------------------------------------


def test_provider_without_key_is_skipped_with_zero_calls(store: Store) -> None:
    config = build_config()
    transport = RoutingTransport({"alpha": [ok("alpha")], "beta": [ok("beta")]})
    env = {KEY_ENV_B: FAKE_KEY, KEY_ENV_C: FAKE_KEY}
    chain = make_chain("answer", config, transport, store, FakeClock(), env=env)
    gen = chain.generate("hello")
    assert gen.provider == "beta"
    assert transport.calls("alpha") == 0


def test_missing_key_reason_is_recorded(store: Store) -> None:
    config = build_config()
    transport = RoutingTransport(
        {"beta": [resp(401, b"unauthorized")], "gamma": [resp(413, b"big")]}
    )
    env = {KEY_ENV_B: FAKE_KEY, KEY_ENV_C: FAKE_KEY}
    chain = make_chain("answer", config, transport, store, FakeClock(), env=env)
    with pytest.raises(LLMError) as info:
        chain.generate("hello")
    assert transport.calls("alpha") == 0
    reasons = {(a.provider, a.reason) for a in info.value.attempts}
    assert ("alpha", Reason.NO_KEY) in reasons


def test_provider_without_model_is_skipped_with_zero_calls(store: Store) -> None:
    config = build_config()
    transport = RoutingTransport({"gamma": [ok("gamma")], "beta": [ok("beta")]})
    chain = make_chain("extraction", config, transport, store, FakeClock())
    gen = chain.generate("extract")
    assert gen.provider == "beta"
    assert transport.calls("gamma") == 0


def test_missing_model_reason_is_recorded(store: Store) -> None:
    config = build_config()
    transport = RoutingTransport({"beta": [resp(401, b"unauthorized")]})
    chain = make_chain("extraction", config, transport, store, FakeClock())
    with pytest.raises(LLMError) as info:
        chain.generate("extract")
    assert transport.calls("gamma") == 0
    reasons = {(a.provider, a.reason) for a in info.value.attempts}
    assert ("gamma", Reason.NO_MODEL) in reasons
    assert ("beta", Reason.EXHAUSTED) in reasons


# --- every provider fails ------------------------------------------------------------------------


def test_all_failing_raises_llm_error_naming_every_provider_and_reason(store: Store) -> None:
    config = build_config()
    transport = RoutingTransport(
        {"alpha": [resp(401, b"unauthorized")], "beta": [resp(503, b"down"), resp(503, b"down")]}
    )
    env = {KEY_ENV_A: FAKE_KEY, KEY_ENV_B: FAKE_KEY}
    chain = make_chain("answer", config, transport, store, FakeClock(), env=env)
    with pytest.raises(LLMError) as info:
        chain.generate("hello")
    err = info.value
    assert not isinstance(err, BudgetExceeded)
    assert err.role == "answer"
    reasons = {(a.provider, a.reason) for a in err.attempts}
    assert ("alpha", Reason.EXHAUSTED) in reasons
    assert ("beta", Reason.SERVER_ERROR) in reasons
    assert ("gamma", Reason.NO_KEY) in reasons
    order = list(dict.fromkeys(a.provider for a in err.attempts))
    assert order == ["alpha", "beta", "gamma"]
    message = str(err)
    for name in ("alpha", "beta", "gamma"):
        assert name in message
    assert transport.log == ["alpha", "beta", "beta"]


def test_all_failing_names_providers_skipped_by_health(store: Store) -> None:
    config = build_config()
    clock = FakeClock(1000.0)
    transport = RoutingTransport(
        {
            "alpha": [resp(429, b"slow down", {"Retry-After": "60"})],
            "beta": [ok("beta"), resp(413, b"big")],
            "gamma": [resp(413, b"big")],
        }
    )
    chain = make_chain("answer", config, transport, store, clock)
    assert chain.generate("first").provider == "beta"
    clock.now = 1001.0
    with pytest.raises(LLMError) as info:
        chain.generate("second")
    assert transport.calls("alpha") == 1
    providers = {a.provider for a in info.value.attempts}
    assert providers == {"alpha", "beta", "gamma"}
    reasons = {(a.provider, a.reason) for a in info.value.attempts}
    assert ("beta", Reason.REQUEST_REJECTED) in reasons
    assert ("gamma", Reason.REQUEST_REJECTED) in reasons


# --- budget ------------------------------------------------------------------------------------


def test_exhausted_budget_raises_before_any_transport_call(store: Store) -> None:
    config = build_config(roles={"answer": {"max_tokens_per_run": 18}})
    budgets = Budgets(config, store)
    budgets.record("answer", 10, 8)
    transport = RoutingTransport({"alpha": [ok("alpha")], "beta": [ok("beta")]})
    chain = make_chain("answer", config, transport, store, FakeClock(), budgets=budgets)
    with pytest.raises(BudgetExceeded):
        chain.generate("hello")
    assert transport.log == []


def test_exhausted_global_budget_raises_before_any_transport_call(store: Store) -> None:
    config = build_config(budget={"max_tokens_per_run": 5})
    budgets = Budgets(config, store)
    budgets.record("summary", 3, 2)
    transport = RoutingTransport({"alpha": [ok("alpha")]})
    chain = make_chain("answer", config, transport, store, FakeClock(), budgets=budgets)
    with pytest.raises(BudgetExceeded):
        chain.generate("hello")
    assert transport.log == []


def test_success_charges_the_budget_with_tokens(store: Store) -> None:
    config = build_config(roles={"answer": {"max_tokens_per_run": 18}})
    budgets = Budgets(config, store)
    transport = RoutingTransport({"alpha": [ok("alpha"), ok("alpha")]})
    chain = make_chain("answer", config, transport, store, FakeClock(), budgets=budgets)
    gen = chain.generate("hello")
    assert gen.prompt_tokens + gen.completion_tokens == 18
    assert transport.calls("alpha") == 1

    with pytest.raises(BudgetExceeded):
        Budgets(config, store).check("answer")
    with pytest.raises(BudgetExceeded):
        chain.generate("again")
    assert transport.calls("alpha") == 1


def test_success_below_cap_leaves_budget_open(store: Store) -> None:
    config = build_config(roles={"answer": {"max_tokens_per_run": 19}})
    budgets = Budgets(config, store)
    transport = RoutingTransport({"alpha": [ok("alpha"), ok("alpha")]})
    chain = make_chain("answer", config, transport, store, FakeClock(), budgets=budgets)
    assert chain.generate("hello").provider == "alpha"
    assert chain.generate("again").provider == "alpha"
    with pytest.raises(BudgetExceeded):
        chain.generate("third")
    assert transport.calls("alpha") == 2


# --- module hygiene ----------------------------------------------------------------------------


PROVIDER_NAMES = (
    "groq",
    "cerebras",
    "gemini",
    "openai",
    "ollama",
    "mistral",
    "openrouter",
    "deepseek",
    "anthropic",
    "sambanova",
    "cohere",
)


def test_no_provider_name_appears_in_the_source() -> None:
    source = Path(chain_module.__file__).read_text(encoding="utf-8").lower()
    found = [name for name in PROVIDER_NAMES if re.search(name, source)]
    assert found == []


class BarrierTransport:
    """Answers only once two sends are in flight at the same time."""

    def __init__(self) -> None:
        self.barrier = threading.Barrier(2, timeout=5.0)
        self.overlapped = 0
        self.lock = threading.Lock()

    def send(self, request: HttpRequest) -> HttpResponse:
        assert HOSTS["alpha"] in request.url
        self.barrier.wait()  # raises BrokenBarrierError if the other send never arrives
        with self.lock:
            self.overlapped += 1
        return ok("alpha")


def test_no_lock_is_held_across_transport_send(tmp_path: Path) -> None:
    config = build_config()
    shared = Store(tmp_path / "llm.sqlite")
    transport = BarrierTransport()
    chain = ProviderChain("answer", config, ENV, FakeClock(), transport, shared)
    results: list[Generation] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(chain.generate("concurrent"))
        except BaseException as exc:  # collected for the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20.0)
    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert transport.overlapped == 2
    assert [g.provider for g in results] == ["alpha", "alpha"]
