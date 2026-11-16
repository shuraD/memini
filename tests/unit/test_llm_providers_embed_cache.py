"""Acceptance tests for LP-16: the embedding cache with batched misses."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import pytest

from memini.infra.llm_providers.config import ConfigError, LLMConfig, parse_config
from memini.infra.llm_providers.embed_cache import CachedEmbedder
from memini.infra.llm_providers.store import Store
from memini.infra.llm_providers.types import Embedder

MODEL_A = "embed-model-a"
MODEL_B = "embed-model-b"


def _vector(text: str, kind: str) -> list[float]:
    """A deterministic vector made of exactly representable floats."""
    return [float(len(text)), float(sum(map(ord, text)) % 1000), 1.0 if kind == "query" else 2.0]


class FakeEmbedder:
    """Records each call (texts, kind) and answers deterministically; thread safe."""

    def __init__(self, *, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.lock = threading.Lock()
        self.calls: list[tuple[list[str], str]] = []

    def embed(
        self, texts: Sequence[str], *, kind: Literal["query", "document"]
    ) -> list[list[float]]:
        with self.lock:
            self.calls.append((list(texts), kind))
        if self.delay_s:
            time.sleep(self.delay_s)
        return [_vector(t, kind) for t in texts]

    def seen(self) -> list[str]:
        return [t for texts, _ in self.calls for t in texts]


def _provider(name: str, model: str, *, limits: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"name": model}
    if limits is not None:
        body["limits"] = limits
    return {
        "name": name,
        "kind": "ollama",
        "base_url": f"http://127.0.0.1:1143{len(name) % 10}",
        "models": {"embedding": body},
    }


def _config(batch_size: Any = None) -> LLMConfig:
    limits = None if batch_size is None else {"batch_size": batch_size}
    return parse_config(
        {
            "providers": [
                _provider("first", MODEL_A, limits=limits),
                _provider("second", MODEL_B, limits=limits),
            ]
        }
    )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "cache.sqlite")


def _cached(
    store: Store, inner: FakeEmbedder, config: LLMConfig | None = None, model: str = MODEL_A
) -> CachedEmbedder:
    return CachedEmbedder(inner, store, config or _config(), model)


def test_is_an_embedder(store: Store) -> None:
    cached = _cached(store, FakeEmbedder())
    assert isinstance(cached, CachedEmbedder)
    embedder: Embedder = cached
    assert callable(embedder.embed)


def test_text_embedded_twice_reaches_inner_once(store: Store) -> None:
    inner = FakeEmbedder()
    cached = _cached(store, inner)
    first = cached.embed(["hello world"], kind="document")
    second = cached.embed(["hello world"], kind="document")
    assert inner.seen() == ["hello world"]
    assert first == second == [_vector("hello world", "document")]


def test_cache_lives_in_the_store_across_instances(store: Store) -> None:
    inner = FakeEmbedder()
    _cached(store, inner).embed(["persisted"], kind="query")
    other_inner = FakeEmbedder()
    result = _cached(store, other_inner).embed(["persisted"], kind="query")
    assert other_inner.calls == []
    assert result == [_vector("persisted", "query")]


def test_kind_is_part_of_the_key(store: Store) -> None:
    inner = FakeEmbedder()
    cached = _cached(store, inner)
    q = cached.embed(["same text"], kind="query")
    d = cached.embed(["same text"], kind="document")
    assert len(inner.calls) == 2
    assert [kind for _, kind in inner.calls] == ["query", "document"]
    assert q == [_vector("same text", "query")]
    assert d == [_vector("same text", "document")]


def test_different_model_is_a_miss(store: Store) -> None:
    inner_a = FakeEmbedder()
    inner_b = FakeEmbedder()
    _cached(store, inner_a, model=MODEL_A).embed(["shared text"], kind="document")
    result = _cached(store, inner_b, model=MODEL_B).embed(["shared text"], kind="document")
    assert inner_b.seen() == ["shared text"]
    assert result == [_vector("shared text", "document")]


def test_misses_are_sent_in_batches_and_order_is_kept(store: Store) -> None:
    inner = FakeEmbedder()
    cached = _cached(store, inner, _config(batch_size=100))
    texts = [f"text number {i}" for i in range(250)]
    result = cached.embed(texts, kind="document")
    assert [len(batch) for batch, _ in inner.calls] == [100, 100, 50]
    assert all(kind == "document" for _, kind in inner.calls)
    assert sorted(inner.seen()) == sorted(texts)
    assert result == [_vector(t, "document") for t in texts]


def test_only_missing_texts_reach_inner(store: Store) -> None:
    inner = FakeEmbedder()
    cached = _cached(store, inner)
    texts = [f"item {i}" for i in range(10)]
    warm = [texts[1], texts[4], texts[6], texts[9]]
    cached.embed(warm, kind="query")
    inner.calls.clear()
    result = cached.embed(texts, kind="query")
    assert sorted(inner.seen()) == sorted(t for t in texts if t not in warm)
    assert len(inner.seen()) == 6
    assert result == [_vector(t, "query") for t in texts]


def test_eight_threads_same_text_one_inner_call(store: Store) -> None:
    inner = FakeEmbedder(delay_s=0.2)
    cached = _cached(store, inner)
    barrier = threading.Barrier(8)
    results: list[list[list[float]]] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def work() -> None:
        try:
            barrier.wait()
            out = cached.embed(["contended text"], kind="document")
            with lock:
                results.append(out)
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert errors == []
    assert len(inner.calls) == 1
    assert inner.seen() == ["contended text"]
    assert results == [[_vector("contended text", "document")]] * 8


def test_duplicates_inside_one_request_embedded_once(store: Store) -> None:
    inner = FakeEmbedder()
    cached = _cached(store, inner)
    texts = ["a", "b", "a", "c", "b", "a"]
    result = cached.embed(texts, kind="document")
    assert sorted(inner.seen()) == ["a", "b", "c"]
    assert result == [_vector(t, "document") for t in texts]


def test_no_batch_size_means_one_batch(store: Store) -> None:
    inner = FakeEmbedder()
    cached = _cached(store, inner, _config())
    texts = [f"t{i}" for i in range(250)]
    result = cached.embed(texts, kind="document")
    assert [len(batch) for batch, _ in inner.calls] == [250]
    assert result == [_vector(t, "document") for t in texts]


def test_no_matching_provider_means_one_batch(store: Store) -> None:
    inner = FakeEmbedder()
    cached = _cached(store, inner, _config(batch_size=10), model="unconfigured-model")
    texts = [f"t{i}" for i in range(25)]
    result = cached.embed(texts, kind="query")
    assert [len(batch) for batch, _ in inner.calls] == [25]
    assert result == [_vector(t, "query") for t in texts]


def test_provider_is_first_in_order_serving_the_model(store: Store) -> None:
    config = parse_config(
        {
            "providers": [
                _provider("other", MODEL_B, limits={"batch_size": 2}),
                _provider("chosen", MODEL_A, limits={"batch_size": 3}),
                _provider("later", MODEL_A, limits={"batch_size": 5}),
            ]
        }
    )
    inner = FakeEmbedder()
    cached = _cached(store, inner, config, model=MODEL_A)
    texts = [f"t{i}" for i in range(7)]
    result = cached.embed(texts, kind="document")
    assert [len(batch) for batch, _ in inner.calls] == [3, 3, 1]
    assert result == [_vector(t, "document") for t in texts]


def test_empty_request_makes_no_inner_call(store: Store) -> None:
    inner = FakeEmbedder()
    assert _cached(store, inner).embed([], kind="query") == []
    assert inner.calls == []


@pytest.mark.parametrize("bad", [0, -5, 2.5, "10"])
def test_batch_size_not_positive_int_raises_config_error(store: Store, bad: Any) -> None:
    inner = FakeEmbedder()
    with pytest.raises(ConfigError):
        cached = _cached(store, inner, _config(batch_size=bad))
        cached.embed(["x", "y"], kind="document")
    assert inner.calls == []
