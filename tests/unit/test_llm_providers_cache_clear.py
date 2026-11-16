"""Acceptance tests for LT-08: clearing each cache table, private stores, Store.path."""

from __future__ import annotations

import json
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from memini.infra.llm_providers.cache import CachedLLM
from memini.infra.llm_providers.chain import ProviderChain
from memini.infra.llm_providers.config import LLMConfig, parse_config
from memini.infra.llm_providers.embed_cache import CachedEmbedder
from memini.infra.llm_providers.kind_openai import HttpRequest
from memini.infra.llm_providers.private import private_store
from memini.infra.llm_providers.store import Store
from memini.infra.llm_providers.transport import HttpResponse

KEY_ENV = "MEMINI_LT08_FAKE_KEY"
ENV: Mapping[str, str] = {KEY_ENV: "test-key"}
HOSTS = {"alpha": "alpha.invalid", "beta": "beta.invalid"}
EMBED_MODEL = "embed-model-a"


class FakeClock:
    def __call__(self) -> float:
        return 1000.0


class CountingTransport:
    """Answers every request with a distinct text naming its provider; counts the sends."""

    def __init__(self) -> None:
        self.count = 0
        self.prompts: list[str] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        self.count += 1
        self.prompts.append(request.body.decode("utf-8"))
        name = next(n for n, host in HOSTS.items() if host in request.url)
        body = {
            "choices": [{"message": {"content": f"answer {self.count} from {name}"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        return HttpResponse(status=200, headers={}, body=json.dumps(body).encode())


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(
        self, texts: Sequence[str], *, kind: Literal["query", "document"]
    ) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0 if kind == "query" else 2.0] for t in texts]


def _provider(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "openai_compatible",
        "key_env": KEY_ENV,
        "base_url": f"https://{HOSTS[name]}/v1",
        "models": {"answer": {"name": f"{name}-answer-model"}},
    }


def _config(order: tuple[str, ...] = ("alpha",)) -> LLMConfig:
    return parse_config(
        {
            "providers": [_provider("alpha"), _provider("beta")],
            "roles": {"answer": {"providers": list(order)}},
        }
    )


def _embed_config() -> LLMConfig:
    return parse_config(
        {
            "providers": [
                {
                    "name": "local",
                    "kind": "ollama",
                    "base_url": "http://127.0.0.1:11434",
                    "models": {"embedding": {"name": EMBED_MODEL}},
                }
            ]
        }
    )


def _llm(store: Store, transport: CountingTransport, config: LLMConfig | None = None) -> CachedLLM:
    cfg = config if config is not None else _config()
    chain = ProviderChain("answer", cfg, ENV, FakeClock(), transport, Store(":memory:"))
    return CachedLLM(chain, store, cfg, "answer")


def _embedder(store: Store, inner: FakeEmbedder) -> CachedEmbedder:
    return CachedEmbedder(inner, store, _embed_config(), EMBED_MODEL)


# --- CachedLLM.clear --------------------------------------------------------------------


def test_clear_returns_count_and_next_request_reaches_chain(tmp_path: Path) -> None:
    transport = CountingTransport()
    llm = _llm(Store(tmp_path / "c.sqlite"), transport)
    llm.generate("p1")
    llm.generate("p2")
    assert llm.generate("p1").cached is True
    assert transport.count == 2

    assert llm.clear() == 2

    again = llm.generate("p1")
    assert again.cached is False
    assert transport.count == 3


def test_clear_empty_returns_zero(tmp_path: Path) -> None:
    llm = _llm(Store(tmp_path / "c.sqlite"), CountingTransport())
    assert llm.clear() == 0


def test_clear_twice_second_returns_zero(tmp_path: Path) -> None:
    llm = _llm(Store(tmp_path / "c.sqlite"), CountingTransport())
    llm.generate("p1")
    assert llm.clear() == 1
    assert llm.clear() == 0


def test_clear_on_memory_store(tmp_path: Path) -> None:
    transport = CountingTransport()
    llm = _llm(Store(":memory:"), transport)
    llm.generate("p1", seed=3, temperature=0.7)
    assert llm.clear() == 1
    assert llm.generate("p1", seed=3, temperature=0.7).cached is False


def test_after_clear_requests_are_stored_again(tmp_path: Path) -> None:
    transport = CountingTransport()
    llm = _llm(Store(tmp_path / "c.sqlite"), transport)
    llm.generate("p1")
    llm.generate("p2")
    llm.clear()

    assert llm.generate("p1").cached is False
    assert llm.generate("p2").cached is False
    sends = transport.count
    assert llm.generate("p1").cached is True
    assert llm.generate("p2").cached is True
    assert transport.count == sends


def test_llm_clear_keeps_embeddings(tmp_path: Path) -> None:
    store = Store(tmp_path / "shared.sqlite")
    llm = _llm(store, CountingTransport())
    inner = FakeEmbedder()
    emb = _embedder(store, inner)
    llm.generate("p1")
    first = emb.embed(["a", "bb"], kind="document")
    assert len(inner.calls) == 1

    assert llm.clear() == 1

    assert emb.embed(["a", "bb"], kind="document") == first
    assert len(inner.calls) == 1


def test_embedder_clear_keeps_generations(tmp_path: Path) -> None:
    store = Store(tmp_path / "shared.sqlite")
    transport = CountingTransport()
    llm = _llm(store, transport)
    inner = FakeEmbedder()
    emb = _embedder(store, inner)
    llm.generate("p1")
    llm.generate("p2")
    emb.embed(["a", "bb", "ccc"], kind="document")

    assert emb.clear() == 3

    emb.embed(["a", "bb", "ccc"], kind="document")
    assert len(inner.calls) == 2
    assert sorted(inner.calls[1]) == ["a", "bb", "ccc"]
    sends = transport.count
    assert llm.generate("p1").cached is True
    assert llm.generate("p2").cached is True
    assert transport.count == sends


def test_embedder_clear_empty_returns_zero(tmp_path: Path) -> None:
    emb = _embedder(Store(tmp_path / "e.sqlite"), FakeEmbedder())
    assert emb.clear() == 0


def test_embedder_after_clear_stores_again(tmp_path: Path) -> None:
    inner = FakeEmbedder()
    emb = _embedder(Store(tmp_path / "e.sqlite"), inner)
    emb.embed(["x"], kind="query")
    assert emb.clear() == 1
    emb.embed(["x"], kind="query")
    emb.embed(["x"], kind="query")
    assert len(inner.calls) == 2


def test_clear_on_one_store_keeps_other_store(tmp_path: Path) -> None:
    t1 = CountingTransport()
    t2 = CountingTransport()
    one = _llm(Store(tmp_path / "one.sqlite"), t1)
    two = _llm(Store(tmp_path / "two.sqlite"), t2)
    one.generate("p1")
    two.generate("p1")
    two.generate("p2")

    assert one.clear() == 1

    assert two.generate("p1").cached is True
    assert two.generate("p2").cached is True
    assert t2.count == 2


# --- private_store ----------------------------------------------------------------------


def test_private_store_paths_distinct_under_directory_mode_0600(tmp_path: Path) -> None:
    base = tmp_path / "bench"
    base.mkdir()
    first = private_store(base)
    second = private_store(str(base))
    assert isinstance(first, Store)
    assert isinstance(second, Store)
    assert first.path != second.path
    root = base.resolve()
    for s in (first, second):
        file = Path(s.path).resolve()
        assert root in file.parents
        assert file.parent != root
        assert file.is_file()
        assert stat.S_IMODE(file.stat().st_mode) == 0o600
    assert Path(first.path).resolve().parent != Path(second.path).resolve().parent


def test_private_stores_never_share_rows(tmp_path: Path) -> None:
    t1 = CountingTransport()
    t2 = CountingTransport()
    one = _llm(private_store(tmp_path), t1)
    two = _llm(private_store(tmp_path), t2)
    assert one.generate("p1").cached is False
    assert two.generate("p1").cached is False
    assert t2.count == 1
    assert one.clear() == 1
    assert two.clear() == 1


def test_private_store_starts_empty(tmp_path: Path) -> None:
    shared = Store(tmp_path / "shared.sqlite")
    _llm(shared, CountingTransport()).generate("p1")
    _embedder(shared, FakeEmbedder()).embed(["a"], kind="query")

    store = private_store(tmp_path)
    assert _llm(store, CountingTransport()).clear() == 0
    assert _embedder(store, FakeEmbedder()).clear() == 0


def test_benchmark_private_cache_clear_keeps_shared(tmp_path: Path) -> None:
    shared_transport = CountingTransport()
    shared = _llm(Store(tmp_path / "shared.sqlite"), shared_transport)
    shared.generate("p1")
    shared.generate("p2")

    bench_transport = CountingTransport()
    bench = _llm(private_store(tmp_path), bench_transport)
    bench.generate("p1")
    bench.generate("p3")
    assert bench_transport.count == 2
    assert bench.clear() == 2

    assert shared.generate("p1").cached is True
    assert shared.generate("p2").cached is True
    assert shared_transport.count == 2


# --- Store.path -------------------------------------------------------------------------


def test_store_path_is_given_path_as_str(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "x.sqlite"
    assert Store(p).path == str(p)
    assert Store(str(p)).path == str(p)
    assert isinstance(Store(p).path, str)


def test_store_memory_path() -> None:
    assert Store(":memory:").path == ":memory:"


# --- provider list in the reply-cache key -------------------------------------------------


def test_provider_list_change_is_a_miss_and_hit_reports_answering_provider(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "c.sqlite")
    t_old = CountingTransport()
    old = _llm(store, t_old, _config(("alpha",)))
    first = old.generate("same prompt")
    assert first.cached is False
    assert first.provider == "alpha"

    t_new = CountingTransport()
    new = _llm(store, t_new, _config(("beta", "alpha")))
    miss = new.generate("same prompt")
    assert miss.cached is False
    assert t_new.count == 1
    assert miss.provider == "beta"

    hit_new = new.generate("same prompt")
    assert hit_new.cached is True
    assert hit_new.provider == "beta"
    assert hit_new.text == miss.text

    hit_old = old.generate("same prompt")
    assert hit_old.cached is True
    assert hit_old.provider == "alpha"
    assert hit_old.text == first.text
    assert t_old.count == 1
