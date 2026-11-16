"""Acceptance tests for LP-18: the chain and both caches write call-log rows."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import pytest

from memini.infra.llm_providers.budgets import question
from memini.infra.llm_providers.cache import CachedLLM
from memini.infra.llm_providers.calllog import CallLog, CallRecord
from memini.infra.llm_providers.chain import ProviderChain
from memini.infra.llm_providers.config import LLMConfig, parse_config
from memini.infra.llm_providers.embed_cache import CachedEmbedder
from memini.infra.llm_providers.kind_openai import HttpRequest
from memini.infra.llm_providers.store import Store
from memini.infra.llm_providers.transport import HttpResponse
from memini.infra.llm_providers.types import LLMError, Reason

KEY_ENV_A = "MEMINI_LP18_FAKE_KEY_A"
KEY_ENV_B = "MEMINI_LP18_FAKE_KEY_B"
ENV: Mapping[str, str] = {KEY_ENV_A: "test-key", KEY_ENV_B: "test-key"}
HOSTS = {"alpha": "alpha.invalid", "beta": "beta.invalid"}
ROLE = "answer"
SEND_SECONDS = 2.5
PRICE_IN = 3.0
PRICE_OUT = 15.0


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def ok(provider: str, prompt_tokens: int = 11, completion_tokens: int = 7) -> HttpResponse:
    body = json.dumps(
        {
            "choices": [{"message": {"content": f"from {provider}"}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }
    ).encode()
    return HttpResponse(status=200, headers={}, body=body)


def status(code: int, headers: Mapping[str, str] | None = None) -> HttpResponse:
    return HttpResponse(status=code, headers=dict(headers or {}), body=b"")


class ScriptedTransport:
    """Answers each provider from its script; every send advances the clock."""

    def __init__(self, clock: FakeClock, scripts: Mapping[str, list[HttpResponse]]) -> None:
        self.clock = clock
        self.scripts = {name: list(items) for name, items in scripts.items()}
        self.log: list[str] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        name = next(n for n, host in HOSTS.items() if host in request.url)
        self.log.append(name)
        script = self.scripts.get(name, [])
        if not script:
            raise AssertionError(f"{name} called more often than scripted")
        self.clock.now += SEND_SECONDS
        return script.pop(0)


def _provider(name: str, key_env: str, *, priced: bool) -> dict[str, Any]:
    model: dict[str, Any] = {"name": f"{name}-model"}
    if priced:
        model["price"] = {"input_per_mtok": PRICE_IN, "output_per_mtok": PRICE_OUT}
    return {
        "name": name,
        "kind": "openai_compatible",
        "key_env": key_env,
        "base_url": f"https://{HOSTS[name]}/v1",
        "models": {ROLE: model},
    }


def _config(*, priced: bool = False) -> LLMConfig:
    return parse_config(
        {
            "providers": [
                _provider("alpha", KEY_ENV_A, priced=priced),
                _provider("beta", KEY_ENV_B, priced=priced),
            ],
            "roles": {ROLE: {"providers": ["alpha", "beta"]}},
        }
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store() -> Store:
    return Store(":memory:")


@pytest.fixture
def log(store: Store, clock: FakeClock) -> CallLog:
    return CallLog(store, clock)


def _chain(
    config: LLMConfig,
    transport: ScriptedTransport,
    store: Store,
    clock: FakeClock,
    log: CallLog | None,
) -> ProviderChain:
    return ProviderChain(ROLE, config, ENV, clock, transport, store, call_log=log)


def _generate_rows(log: CallLog) -> list[CallRecord]:
    return [r for r in log.rows() if r.kind == "generate"]


# --- one successful call ---------------------------------------------------------------


def test_one_successful_call_writes_exactly_one_row(
    store: Store, clock: FakeClock, log: CallLog
) -> None:
    transport = ScriptedTransport(clock, {"alpha": [ok("alpha", 11, 7)]})
    chain = _chain(_config(), transport, store, clock, log)

    chain.generate("hello", stage="retrieve")

    rows = log.rows()
    assert len(rows) == 1
    row = rows[0]
    assert row.kind == "generate"
    assert row.provider == "alpha"
    assert row.model == "alpha-model"
    assert row.role == ROLE
    assert row.stage == "retrieve"
    assert row.prompt_tokens == 11
    assert row.completion_tokens == 7
    assert row.seconds == pytest.approx(SEND_SECONDS)
    assert row.cache_hit is False
    assert row.retries == 0
    assert row.fallbacks == ()
    assert row.wait_seconds == pytest.approx(0.0)


def test_chain_without_call_log_still_generates(store: Store, clock: FakeClock) -> None:
    transport = ScriptedTransport(clock, {"alpha": [ok("alpha")]})
    chain = _chain(_config(), transport, store, clock, None)

    gen = chain.generate("hello")

    assert gen.provider == "alpha"


# --- cache -----------------------------------------------------------------------------


def test_cached_repeat_writes_one_more_row_marked_hit_with_no_tokens(
    store: Store, clock: FakeClock, log: CallLog
) -> None:
    config = _config(priced=True)
    transport = ScriptedTransport(clock, {"alpha": [ok("alpha", 11, 7)]})
    chain = _chain(config, transport, store, clock, log)
    cached = CachedLLM(chain, store, config, ROLE, call_log=log)

    cached.generate("same prompt", temperature=0.0)
    assert len(_generate_rows(log)) == 1
    assert _generate_rows(log)[0].cache_hit is False

    second = cached.generate("same prompt", temperature=0.0)

    assert second.cached is True
    assert transport.log == ["alpha"]
    rows = _generate_rows(log)
    assert len(rows) == 2
    hit = rows[1]
    assert hit.cache_hit is True
    assert hit.role == ROLE
    assert hit.prompt_tokens == 0
    assert hit.completion_tokens == 0
    assert hit.cost in {None, 0.0}


# --- fallback, retries, waits ----------------------------------------------------------


def test_fallback_after_429_lists_provider_and_wait(
    store: Store, clock: FakeClock, log: CallLog
) -> None:
    transport = ScriptedTransport(
        clock,
        {"alpha": [status(429, {"Retry-After": "12"})], "beta": [ok("beta")]},
    )
    chain = _chain(_config(), transport, store, clock, log)

    gen = chain.generate("hello")

    assert gen.provider == "beta"
    rows = log.rows()
    assert len(rows) == 1
    row = rows[0]
    assert row.provider == "beta"
    assert row.model == "beta-model"
    assert row.fallbacks == (("alpha", Reason.RATE_PER_MINUTE.value),)
    assert row.wait_seconds == pytest.approx(12.0)
    assert row.cache_hit is False


def test_retry_after_503_counts_one_retry(store: Store, clock: FakeClock, log: CallLog) -> None:
    transport = ScriptedTransport(clock, {"alpha": [status(503), ok("alpha")]})
    chain = _chain(_config(), transport, store, clock, log)

    gen = chain.generate("hello")

    assert gen.provider == "alpha"
    rows = log.rows()
    assert len(rows) == 1
    assert rows[0].retries == 1
    assert rows[0].provider == "alpha"
    assert rows[0].fallbacks == ()


def test_total_failure_still_writes_a_row_with_reasons(
    store: Store, clock: FakeClock, log: CallLog
) -> None:
    transport = ScriptedTransport(
        clock,
        {
            "alpha": [status(429, {"Retry-After": "5"})],
            "beta": [status(503), status(503)],
        },
    )
    chain = _chain(_config(), transport, store, clock, log)

    with pytest.raises(LLMError):
        chain.generate("hello")

    rows = log.rows()
    assert len(rows) == 1
    row = rows[0]
    assert row.role == ROLE
    assert row.cache_hit is False
    assert row.fallbacks == (
        ("alpha", Reason.RATE_PER_MINUTE.value),
        ("beta", Reason.SERVER_ERROR.value),
    )


# --- cost --------------------------------------------------------------------------------


def test_cost_uses_configured_price(store: Store, clock: FakeClock, log: CallLog) -> None:
    transport = ScriptedTransport(clock, {"alpha": [ok("alpha", 1200, 300)]})
    chain = _chain(_config(priced=True), transport, store, clock, log)

    chain.generate("hello")

    (row,) = log.rows()
    expected = 1200 * PRICE_IN / 1e6 + 300 * PRICE_OUT / 1e6
    assert row.cost == pytest.approx(expected)


def test_cost_is_none_without_price(store: Store, clock: FakeClock, log: CallLog) -> None:
    transport = ScriptedTransport(clock, {"alpha": [ok("alpha", 1200, 300)]})
    chain = _chain(_config(priced=False), transport, store, clock, log)

    chain.generate("hello")

    (row,) = log.rows()
    assert row.cost is None


# --- question id -------------------------------------------------------------------------


def test_question_id_inside_and_outside_question(
    store: Store, clock: FakeClock, log: CallLog
) -> None:
    transport = ScriptedTransport(clock, {"alpha": [ok("alpha"), ok("alpha")]})
    chain = _chain(_config(), transport, store, clock, log)

    with question("Q01"):
        chain.generate("inside")
    chain.generate("outside")

    rows = log.rows()
    assert [r.question_id for r in rows] == ["Q01", None]


# --- embeddings --------------------------------------------------------------------------

EMBED_MODEL = "embed-model-a"


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(
        self, texts: Sequence[str], *, kind: Literal["query", "document"]
    ) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0 if kind == "query" else 2.0] for t in texts]


def _embed_config(batch_size: int) -> LLMConfig:
    return parse_config(
        {
            "providers": [
                {
                    "name": "local",
                    "kind": "ollama",
                    "base_url": "http://127.0.0.1:11434",
                    "models": {
                        "embedding": {"name": EMBED_MODEL, "limits": {"batch_size": batch_size}}
                    },
                }
            ]
        }
    )


def test_embedding_of_250_texts_writes_one_row_per_batch(store: Store, log: CallLog) -> None:
    inner = FakeEmbedder()
    embedder = CachedEmbedder(inner, store, _embed_config(100), EMBED_MODEL, call_log=log)
    texts = [f"text number {i}" for i in range(250)]

    vectors = embedder.embed(texts, kind="document")

    assert len(vectors) == 250
    assert [len(c) for c in inner.calls] == [100, 100, 50]
    rows = [r for r in log.rows() if r.kind == "embed"]
    assert len(rows) == 3
    assert [r.texts for r in rows] == [100, 100, 50]
    assert all(r.model == EMBED_MODEL for r in rows)
    assert all(r.cache_hit is False for r in rows)
    assert [r for r in log.rows() if r.kind != "embed"] == []


def test_embedder_without_call_log_still_embeds(store: Store) -> None:
    inner = FakeEmbedder()
    embedder = CachedEmbedder(inner, store, _embed_config(100), EMBED_MODEL, call_log=None)

    vectors = embedder.embed(["a", "b"], kind="query")

    assert len(vectors) == 2
