"""Acceptance tests for LP-14: the generation cache with single flight."""

from __future__ import annotations

import json
import sqlite3
import stat
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from memini.infra.llm_providers.cache import CachedLLM, generation_key
from memini.infra.llm_providers.chain import ProviderChain
from memini.infra.llm_providers.config import LLMConfig, parse_config
from memini.infra.llm_providers.kind_openai import HttpRequest
from memini.infra.llm_providers.store import Store
from memini.infra.llm_providers.transport import HttpResponse
from memini.infra.llm_providers.types import LLM, Generation, LLMError

KEY_ENV = "MEMINI_LP14_FAKE_KEY"
ENV: Mapping[str, str] = {KEY_ENV: "test-key"}
HOSTS = {"alpha": "alpha.invalid", "beta": "beta.invalid"}


class FakeClock:
    def __call__(self) -> float:
        return 1000.0


class CountingTransport:
    """Answers every request with a distinct text and counts the sends."""

    def __init__(self, *, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.fail = False
        self.lock = threading.Lock()
        self.count = 0
        self.prompts: list[str] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        with self.lock:
            self.count += 1
            number = self.count
            self.prompts.append(request.body.decode("utf-8"))
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail:
            return HttpResponse(status=413, headers={}, body=b"too large")
        provider = next(name for name, host in HOSTS.items() if host in request.url)
        body = {
            "choices": [{"message": {"content": f"answer {number} from {provider}"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        return HttpResponse(status=200, headers={}, body=json.dumps(body).encode())

    def sent_prompt(self, prompt: str) -> int:
        return sum(1 for body in self.prompts if prompt in body)


def provider(name: str, model_suffix: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "kind": "openai_compatible",
        "key_env": KEY_ENV,
        "base_url": f"https://{HOSTS[name]}/v1",
        "models": {
            role: {"name": f"{name}-{role}-model{model_suffix}"} for role in ("answer", "judge")
        },
    }


def build_config(order: tuple[str, ...] = ("alpha",), model_suffix: str = "") -> LLMConfig:
    return parse_config(
        {
            "providers": [provider("alpha", model_suffix), provider("beta", model_suffix)],
            "roles": {
                "answer": {"providers": list(order)},
                "judge": {"providers": list(order)},
            },
        }
    )


def make_cached(
    store: Store,
    transport: CountingTransport,
    *,
    config: LLMConfig | None = None,
    role: str = "answer",
) -> CachedLLM:
    cfg = config if config is not None else build_config()
    chain = ProviderChain(role, cfg, ENV, FakeClock(), transport, Store(":memory:"))
    return CachedLLM(chain, store, cfg, role)


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "cache" / "llm.sqlite"


@pytest.fixture
def store(cache_path: Path) -> Store:
    return Store(cache_path)


def cache_rows(path: Path) -> int:
    """Count the rows of every table in the cache file except the store's bookkeeping."""
    conn = sqlite3.connect(path)
    try:
        names = [
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            if row[0] != "_store_schema" and not row[0].startswith("sqlite_")
        ]
        total = 0
        for name in names:
            # The name comes from sqlite_master of the test's own file, not from input.
            query = f'SELECT COUNT(*) FROM "{name}"'  # noqa: S608
            total += conn.execute(query).fetchone()[0]
        return total
    finally:
        conn.close()


# --- generation_key -----------------------------------------------------------------------

BASE_KEY_ARGS: dict[str, Any] = {
    "provider_list": ("alpha", "beta"),
    "model_by_provider": {"alpha": "alpha-model", "beta": "beta-model"},
    "role": "answer",
    "prompt": "the prompt",
    "system": "the system",
    "json_mode": False,
    "temperature": 0.0,
    "seed": 7,
}


def key_of(args: Mapping[str, Any]) -> str:
    return generation_key(
        args["provider_list"],
        args["model_by_provider"],
        args["role"],
        args["prompt"],
        args["system"],
        args["json_mode"],
        args["temperature"],
        args["seed"],
    )


def test_generation_key_is_sha256_hex_and_stable() -> None:
    first = key_of(BASE_KEY_ARGS)
    second = key_of(dict(BASE_KEY_ARGS))
    assert first == second
    assert len(first) == 64
    assert all(ch in "0123456789abcdef" for ch in first)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider_list", ("beta", "alpha")),
        ("model_by_provider", {"alpha": "alpha-model-2", "beta": "beta-model"}),
        ("role", "judge"),
        ("prompt", "another prompt"),
        ("system", "another system"),
        ("json_mode", True),
        ("temperature", 0.5),
        ("seed", 8),
    ],
)
def test_generation_key_changes_with_each_field(field: str, value: Any) -> None:
    changed = dict(BASE_KEY_ARGS)
    changed[field] = value
    assert key_of(changed) != key_of(BASE_KEY_ARGS)


# --- hits and misses ----------------------------------------------------------------------


def test_cached_llm_implements_llm(store: Store) -> None:
    assert isinstance(make_cached(store, CountingTransport()), LLM)


def test_seeded_call_repeated_is_a_hit(store: Store) -> None:
    transport = CountingTransport()
    llm = make_cached(store, transport)
    first = llm.generate("hello", temperature=0.7, seed=42)
    second = llm.generate("hello", temperature=0.7, seed=42)
    assert first.cached is False
    assert second.cached is True
    assert second.text == first.text
    assert transport.count == 1


def test_temperature_zero_call_repeated_is_a_hit(store: Store) -> None:
    transport = CountingTransport()
    llm = make_cached(store, transport)
    first = llm.generate("hello", temperature=0.0)
    second = llm.generate("hello", temperature=0.0)
    assert second.cached is True
    assert second.text == first.text
    assert transport.count == 1


def test_nonzero_temperature_without_seed_is_not_cached(store: Store, cache_path: Path) -> None:
    transport = CountingTransport()
    llm = make_cached(store, transport)
    first = llm.generate("hello", temperature=0.7)
    second = llm.generate("hello", temperature=0.7)
    assert transport.count == 2
    assert first.cached is False
    assert second.cached is False
    assert first.text != second.text
    assert cache_rows(cache_path) == 0


def _variant(field: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (llm options, generate options) differing from the base in ``field``."""
    setup: dict[str, Any] = {}
    call: dict[str, Any] = {}
    if field == "provider_list":
        setup["config"] = build_config(order=("alpha", "beta"))
    elif field == "model":
        setup["config"] = build_config(model_suffix="-v2")
    elif field == "role":
        setup["role"] = "judge"
    elif field == "system":
        call["system"] = "a different system prompt"
    elif field == "json_mode":
        call["json_mode"] = True
    elif field == "temperature":
        call["temperature"] = 0.5
    elif field == "seed":
        call["seed"] = 2
    return setup, call


@pytest.mark.parametrize(
    "field",
    ["provider_list", "model", "role", "system", "json_mode", "temperature", "seed"],
)
def test_changed_field_is_a_miss(store: Store, field: str) -> None:
    base_call: dict[str, Any] = {
        "system": "the system prompt",
        "json_mode": False,
        "temperature": 0.0,
        "seed": 1,
    }
    transport = CountingTransport()
    base = make_cached(store, transport)
    base.generate("same prompt", **base_call)
    assert base.generate("same prompt", **base_call).cached is True
    assert transport.count == 1

    setup, call = _variant(field)
    other = make_cached(store, transport, **setup)
    result = other.generate("same prompt", **{**base_call, **call})
    assert result.cached is False
    assert transport.count == 2


# --- single flight and processes ----------------------------------------------------------


def test_eight_threads_missing_same_key_make_one_inner_call(store: Store) -> None:
    transport = CountingTransport(delay_s=0.2)
    llm = make_cached(store, transport)
    barrier = threading.Barrier(8)
    results: list[Generation] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            gen = llm.generate("shared prompt", seed=3)
        except BaseException as exc:
            with lock:
                errors.append(exc)
            return
        with lock:
            results.append(gen)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert errors == []
    assert transport.count == 1
    assert len(results) == 8
    assert {gen.text for gen in results} == {"answer 1 from alpha"}


def test_two_store_handles_writing_same_key_keep_one_valid_row(cache_path: Path) -> None:
    store_a = Store(cache_path)
    store_b = Store(cache_path)
    transport_a = CountingTransport(delay_s=0.2)
    transport_b = CountingTransport(delay_s=0.2)
    llm_a = make_cached(store_a, transport_a)
    llm_b = make_cached(store_b, transport_b)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def run(llm: CachedLLM) -> None:
        barrier.wait()
        try:
            llm.generate("cross process prompt", seed=9)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(llm,)) for llm in (llm_a, llm_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert errors == []

    conn = sqlite3.connect(cache_path)
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()
    assert cache_rows(cache_path) == 1
    assert stat.S_IMODE(cache_path.stat().st_mode) == 0o600

    reader = CountingTransport()
    reader.fail = True
    hit = make_cached(Store(cache_path), reader).generate("cross process prompt", seed=9)
    assert hit.cached is True
    assert hit.text.endswith("from alpha")
    assert reader.count == 0


def test_cache_file_is_owner_only(store: Store, cache_path: Path) -> None:
    make_cached(store, CountingTransport()).generate("hello", seed=1)
    assert cache_rows(cache_path) == 1
    assert stat.S_IMODE(cache_path.stat().st_mode) == 0o600


# --- failures -----------------------------------------------------------------------------


def test_failing_inner_call_stores_nothing_and_later_call_retries(
    store: Store, cache_path: Path
) -> None:
    transport = CountingTransport()
    transport.fail = True
    llm = make_cached(store, transport)
    with pytest.raises(LLMError):
        llm.generate("fragile", seed=5)
    assert cache_rows(cache_path) == 0
    assert transport.count == 1

    transport.fail = False
    gen = llm.generate("fragile", seed=5)
    assert gen.cached is False
    assert transport.count == 2
    assert llm.generate("fragile", seed=5).cached is True
    assert transport.count == 2


# --- hit speed ----------------------------------------------------------------------------


def test_hit_is_fast_without_transport_or_inner_call(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = CountingTransport()
    llm = make_cached(store, transport)
    first = llm.generate("fast", seed=11)
    assert transport.count == 1

    inner_calls: list[str] = []

    def forbidden(self: ProviderChain, *args: Any, **kwargs: Any) -> Any:
        inner_calls.append("called")
        raise AssertionError("inner chain called on a hit")

    monkeypatch.setattr(ProviderChain, "generate", forbidden)
    monkeypatch.setattr(ProviderChain, "generate_many", forbidden)
    best = float("inf")
    for _ in range(20):
        start = time.perf_counter()
        gen = llm.generate("fast", seed=11)
        best = min(best, time.perf_counter() - start)
        assert gen.cached is True
        assert gen.text == first.text
    assert best < 0.005
    assert transport.count == 1
    assert inner_calls == []


# --- generate_many ------------------------------------------------------------------------


def test_generate_many_calls_inner_only_for_new_prompts(store: Store) -> None:
    transport = CountingTransport()
    llm = make_cached(store, transport)
    cached_prompts = ["prompt-one", "prompt-two", "prompt-three"]
    earlier = {p: llm.generate(p, seed=4).text for p in cached_prompts}
    assert transport.count == 3

    prompts = ["prompt-one", "prompt-new-a", "prompt-two", "prompt-new-b", "prompt-three"]
    results = llm.generate_many(prompts, seed=4)
    assert transport.count == 5
    assert transport.sent_prompt("prompt-new-a") == 1
    assert transport.sent_prompt("prompt-new-b") == 1
    for p in cached_prompts:
        assert transport.sent_prompt(p) == 1
    assert len(results) == 5
    for prompt, result in zip(prompts, results, strict=True):
        assert isinstance(result, Generation)
        if prompt in earlier:
            assert result.cached is True
            assert result.text == earlier[prompt]
        else:
            assert result.cached is False

    again = llm.generate_many(prompts, seed=4)
    assert transport.count == 5
    assert all(isinstance(r, Generation) and r.cached for r in again)
