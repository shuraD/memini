"""Acceptance tests for LP-21: role singletons and the registry of the provider layer."""

from __future__ import annotations

import builtins
import io
import json
import os
import socket
import sqlite3
import threading
import time
import types
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

import memini.infra.llm_providers as package
from memini.infra.llm_providers import registry
from memini.infra.llm_providers.budgets import Budgets
from memini.infra.llm_providers.chain import ProviderChain
from memini.infra.llm_providers.config import LLMConfig, parse_config
from memini.infra.llm_providers.health import ProviderHealth
from memini.infra.llm_providers.kind_openai import HttpRequest
from memini.infra.llm_providers.limiter import ConcurrencyLimits
from memini.infra.llm_providers.perf_summary import RunSummary
from memini.infra.llm_providers.store import Store
from memini.infra.llm_providers.transport import HttpResponse
from memini.infra.llm_providers.types import ROLES, BudgetExceeded, Generation, LLMError

KEY_ENV_A = "MEMINI_LP21_FAKE_KEY_A"
KEY_ENV_B = "MEMINI_LP21_FAKE_KEY_B"
FAKE_KEY = "test-key"
ENV: Mapping[str, str] = {KEY_ENV_A: FAKE_KEY, KEY_ENV_B: FAKE_KEY}

HOSTS = {"alpha": "alpha.invalid", "beta": "beta.invalid", "local": "local.invalid"}

PUBLIC_NAMES = (
    "llm_for",
    "embedder",
    "question",
    "summary",
    "export_jsonl",
    "reset_for_tests",
    "configure_for_tests",
)


def fixed_clock() -> float:
    return 1000.0


def ok_body(provider: str, prompt_tokens: int = 11, completion_tokens: int = 7) -> bytes:
    return json.dumps(
        {
            "choices": [{"message": {"content": f"from {provider}"}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }
    ).encode()


class FakeTransport:
    """Answers chat requests per provider host and embedding requests with fixed vectors."""

    def __init__(self, statuses: Mapping[str, int] | None = None) -> None:
        self.statuses = dict(statuses or {})
        self.log: list[str] = []
        self._lock = threading.Lock()

    def _provider(self, url: str) -> str:
        for name, host in HOSTS.items():
            if host in url:
                return name
        raise AssertionError(f"unexpected url {url}")

    def send(self, request: HttpRequest) -> HttpResponse:
        name = self._provider(request.url)
        with self._lock:
            self.log.append(name)
        if name == "local":
            texts = json.loads(request.body)["input"]
            count = len(texts) if isinstance(texts, list) else 1
            body = json.dumps({"embeddings": [[0.5, 0.25] for _ in range(count)]}).encode()
            return HttpResponse(status=200, headers={}, body=body)
        status = self.statuses.get(name, 200)
        if status != 200:
            return HttpResponse(status=status, headers={}, body=b'{"error": "fake"}')
        return HttpResponse(status=200, headers={}, body=ok_body(name))

    def calls(self, name: str) -> int:
        return self.log.count(name)


def chat_provider(name: str, key_env: str, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": name,
        "kind": "openai_compatible",
        "key_env": key_env,
        "base_url": f"https://{HOSTS[name]}/v1",
        "models": {role: {"name": f"{name}-{role}-model"} for role in ROLES},
    }
    body.update(extra)
    return body


def raw_config(**top: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "providers": [
            chat_provider("alpha", KEY_ENV_A, limits={"max_concurrent": 2}),
            chat_provider("beta", KEY_ENV_B),
            {
                "name": "local",
                "kind": "ollama",
                "base_url": f"http://{HOSTS['local']}:11434",
                "models": {"embedding": {"name": "local-embedding-model"}},
            },
        ],
        "roles": {
            "answer": {"providers": ["alpha", "beta"]},
            "extraction": {"providers": ["alpha", "beta"]},
        },
    }
    raw.update(top)
    return raw


def build_config(**top: Any) -> LLMConfig:
    return parse_config(raw_config(**top))


def configure(
    config: LLMConfig | None = None,
    *,
    transport: FakeTransport | None = None,
    store: Store | None = None,
    clock: Callable[[], float] = fixed_clock,
) -> tuple[FakeTransport, Store]:
    transport = transport if transport is not None else FakeTransport()
    store = store if store is not None else Store(":memory:")
    registry.configure_for_tests(
        config=config if config is not None else build_config(),
        env=ENV,
        clock=clock,
        transport=transport,
        store=store,
    )
    return transport, store


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    registry.reset_for_tests()
    yield
    registry.reset_for_tests()


# --- singletons --------------------------------------------------------------------------


def test_llm_for_returns_the_same_object_per_role() -> None:
    configure()
    first = registry.llm_for("answer")
    second = registry.llm_for("answer")
    assert first is second


def test_llm_for_gives_each_role_its_own_object() -> None:
    configure()
    assert registry.llm_for("answer") is not registry.llm_for("extraction")


def test_every_role_gets_an_llm() -> None:
    configure()
    objects = [registry.llm_for(role) for role in ROLES]
    assert len({id(obj) for obj in objects}) == len(ROLES)
    for obj in objects:
        assert callable(getattr(obj, "generate", None))


def test_embedder_is_a_singleton() -> None:
    configure()
    assert registry.embedder() is registry.embedder()


def test_embedder_embeds_through_the_injected_transport() -> None:
    transport, _ = configure()
    vectors = registry.embedder().embed(["some text"], kind="query")
    assert vectors == [[0.5, 0.25]]
    assert transport.calls("local") >= 1


def test_sixteen_threads_create_exactly_one_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    configure()
    built: list[str] = []
    original_init = ProviderChain.__init__

    def slow_init(self: ProviderChain, role: str, *args: Any, **kwargs: Any) -> None:
        built.append(role)
        time.sleep(0.01)  # widen the window in which a racing thread could build a second one
        original_init(self, role, *args, **kwargs)

    monkeypatch.setattr(ProviderChain, "__init__", slow_init)
    barrier = threading.Barrier(16)
    results: list[object] = []
    errors: list[BaseException] = []
    guard = threading.Lock()

    def worker() -> None:
        try:
            barrier.wait()
            obj = registry.llm_for("answer")
        except BaseException as exc:
            with guard:
                errors.append(exc)
            return
        with guard:
            results.append(obj)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(results) == 16
    assert len({id(obj) for obj in results}) == 1
    assert built.count("answer") <= 1


def test_second_call_does_no_io(monkeypatch: pytest.MonkeyPatch) -> None:
    configure()
    first = registry.llm_for("answer")
    calls: list[str] = []

    def spy(label: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            calls.append(label)
            return original(*args, **kwargs)

        return wrapper

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", spy("builtins.open", builtins.open))
        patch.setattr(io, "open", spy("io.open", io.open))
        patch.setattr(os, "open", spy("os.open", os.open))
        patch.setattr(sqlite3, "connect", spy("sqlite3.connect", sqlite3.connect))
        patch.setattr(
            socket, "create_connection", spy("create_connection", socket.create_connection)
        )
        patch.setattr(socket.socket, "connect", spy("socket.connect", socket.socket.connect))
        patch.setattr(Store, "execute", spy("Store.execute", Store.execute))
        patch.setattr(Store, "ensure_schema", spy("Store.ensure_schema", Store.ensure_schema))
        second = registry.llm_for("answer")
    assert second is first
    assert calls == []


def test_reset_for_tests_creates_new_objects_on_next_call() -> None:
    configure()
    llm_before = registry.llm_for("answer")
    embedder_before = registry.embedder()
    registry.reset_for_tests()
    configure()
    assert registry.llm_for("answer") is not llm_before
    assert registry.embedder() is not embedder_before


def test_reset_for_tests_returns_none() -> None:
    assert registry.reset_for_tests() is None


@pytest.mark.parametrize("role", ["", "unknown-role", "embedding", "ANSWER"])
def test_role_outside_roles_is_rejected(role: str) -> None:
    configure()
    assert role not in ROLES
    with pytest.raises(ValueError, match=r"."):
        registry.llm_for(role)


# --- end to end with a fake transport -------------------------------------------------------


def test_end_to_end_generate_caches_the_repeat_and_summary_shows_two_rows() -> None:
    transport, _ = configure()
    llm = registry.llm_for("answer")
    first = llm.generate("what is the answer?")
    second = llm.generate("what is the answer?")
    assert isinstance(first, Generation)
    assert first.text == "from alpha"
    assert first.cached is False
    assert second.cached is True
    assert second.text == first.text
    assert transport.calls("alpha") == 1
    summary = registry.summary()
    assert isinstance(summary, RunSummary)
    assert summary.by_role["answer"].calls == 2
    assert summary.by_provider["alpha"].calls == 2
    assert summary.by_role["answer"].cache_hit_pct == pytest.approx(50.0)


def test_export_jsonl_writes_every_logged_call(tmp_path: Path) -> None:
    configure()
    llm = registry.llm_for("answer")
    llm.generate("prompt one")
    llm.generate("prompt one")
    target = tmp_path / "calls.jsonl"
    written = registry.export_jsonl(target)
    assert written == 2
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert [row["cache_hit"] for row in rows] == [False, True]
    assert {row["role"] for row in rows} == {"answer"}


def test_question_scopes_the_call_log_rows(tmp_path: Path) -> None:
    configure()
    with registry.question("q-7"):
        registry.llm_for("answer").generate("inside the question")
    registry.llm_for("answer").generate("outside the question")
    target = tmp_path / "calls.jsonl"
    assert registry.export_jsonl(target) == 2
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert [row["question_id"] for row in rows] == ["q-7", None]


def test_question_applies_the_per_question_call_budget() -> None:
    config = build_config(
        roles={"answer": {"providers": ["alpha", "beta"], "max_calls_per_question": 1}}
    )
    configure(config)
    llm = registry.llm_for("answer")
    with registry.question("q-1"):
        llm.generate("first")
        with pytest.raises(BudgetExceeded):
            llm.generate("second")
    with registry.question("q-2"):
        assert isinstance(llm.generate("third"), Generation)


# --- shared state across roles --------------------------------------------------------------


def _count_inits(monkeypatch: pytest.MonkeyPatch, cls: type, seen: list[tuple[Any, ...]]) -> None:
    original = cls.__init__

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
        seen.append((self, args, kwargs))
        original(self, *args, **kwargs)

    monkeypatch.setattr(cls, "__init__", wrapper)


def _store_argument(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    for value in (*args, *kwargs.values()):
        if isinstance(value, Store):
            return value
    return None


def test_all_roles_share_one_health_one_limits_and_one_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    healths: list[tuple[Any, ...]] = []
    limits: list[tuple[Any, ...]] = []
    budgets: list[tuple[Any, ...]] = []
    stores: list[tuple[Any, ...]] = []
    _count_inits(monkeypatch, ProviderHealth, healths)
    _count_inits(monkeypatch, ConcurrencyLimits, limits)
    _count_inits(monkeypatch, Budgets, budgets)
    _, store = configure()
    _count_inits(monkeypatch, Store, stores)
    for role in ROLES:
        registry.llm_for(role)
    assert len(healths) == 1
    assert len(limits) == 1
    assert stores == []
    assert _store_argument(healths[0][1], healths[0][2]) is store
    assert budgets
    for _, args, kwargs in budgets:
        assert _store_argument(args, kwargs) is store


def test_an_exhausted_provider_is_skipped_by_every_role() -> None:
    transport = FakeTransport(statuses={"alpha": 402})
    configure(transport=transport)
    first = registry.llm_for("answer").generate("asked by the answer role")
    assert first.provider == "beta"
    alpha_calls = transport.calls("alpha")
    assert alpha_calls >= 1
    second = registry.llm_for("extraction").generate("asked by the extraction role")
    assert second.provider == "beta"
    assert transport.calls("alpha") == alpha_calls


def test_generations_of_every_role_land_in_the_injected_store() -> None:
    transport, store = configure()
    registry.llm_for("answer").generate("same prompt")
    registry.reset_for_tests()
    configure(transport=transport, store=store)
    again = registry.llm_for("answer").generate("same prompt")
    assert again.cached is True
    assert transport.calls("alpha") == 1


# --- run scoping -------------------------------------------------------------------------


def test_budgets_are_scoped_to_the_injected_run_id() -> None:
    capped = {"answer": {"providers": ["alpha", "beta"], "max_tokens_per_run": 10}}
    store = Store(":memory:")
    Budgets(build_config(run_id="r1", roles=capped), store).record("answer", 100, 0)
    configure(build_config(run_id="r1", roles=capped), store=store)
    with pytest.raises(BudgetExceeded):
        registry.llm_for("answer").generate("over budget in r1")


def test_budget_spent_in_another_run_does_not_count() -> None:
    capped = {"answer": {"providers": ["alpha", "beta"], "max_tokens_per_run": 10}}
    store = Store(":memory:")
    Budgets(build_config(run_id="r0", roles=capped), store).record("answer", 100, 0)
    configure(build_config(run_id="r1", roles=capped), store=store)
    assert isinstance(registry.llm_for("answer").generate("fresh in r1"), Generation)


def test_health_is_scoped_to_the_injected_run_id() -> None:
    only_alpha = {"answer": {"providers": ["alpha"]}}
    store = Store(":memory:")
    ProviderHealth(store, fixed_clock, run_id="r1").mark_exhausted("alpha", "exhausted")
    transport, _ = configure(build_config(run_id="r1", roles=only_alpha), store=store)
    with pytest.raises(LLMError):
        registry.llm_for("answer").generate("alpha is off in r1")
    assert transport.calls("alpha") == 0


def test_health_marks_of_another_run_are_ignored() -> None:
    only_alpha = {"answer": {"providers": ["alpha"]}}
    store = Store(":memory:")
    ProviderHealth(store, fixed_clock, run_id="r0").mark_exhausted("alpha", "exhausted")
    configure(build_config(run_id="r1", roles=only_alpha), store=store)
    generation = registry.llm_for("answer").generate("alpha is on in r1")
    assert generation.provider == "alpha"


# --- default configuration ------------------------------------------------------------------


def test_default_config_comes_from_providers_yaml_in_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache" / "llm.sqlite"
    raw = raw_config(cache_path=str(cache), run_id="from-file")
    (tmp_path / "providers.yaml").write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setenv("MEMINI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv(KEY_ENV_A, FAKE_KEY)
    monkeypatch.setenv(KEY_ENV_B, FAKE_KEY)
    registry.reset_for_tests()
    llm = registry.llm_for("answer")
    assert callable(getattr(llm, "generate", None))
    assert cache.exists()


# --- module surface -------------------------------------------------------------------------


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_public_names_are_exported_from_the_package(name: str) -> None:
    assert getattr(package, name) is getattr(registry, name)
    assert name in package.__all__


def _is_lock(value: object) -> bool:
    lock_types = (type(threading.Lock()), type(threading.RLock()))
    return isinstance(value, lock_types)


def test_the_only_module_level_mutable_state_is_the_locked_registry() -> None:
    immutable = (
        str,
        bytes,
        int,
        float,
        bool,
        tuple,
        frozenset,
        type(None),
        types.MappingProxyType,
    )
    mutable: list[tuple[str, object]] = []
    locks: list[object] = []
    for name, value in vars(registry).items():
        if name.startswith("__") and name.endswith("__"):
            continue
        if isinstance(value, types.ModuleType | type | types.FunctionType):
            continue
        if callable(value) and not isinstance(value, dict | list | set):
            continue
        if _is_lock(value):
            locks.append(value)
            continue
        if isinstance(value, immutable):
            continue
        if isinstance(value, dict | list | set | bytearray) or (
            type(value).__module__ == registry.__name__
        ):
            mutable.append((name, value))
    assert len(mutable) <= 1, f"module-level mutable state: {[n for n, _ in mutable]}"
    if not locks:
        assert mutable, "no registry lock found"
        holder = mutable[0][1]
        attributes = [getattr(holder, n, None) for n in dir(holder) if not n.startswith("__")]
        assert any(_is_lock(v) for v in attributes), "registry is not behind a lock"
