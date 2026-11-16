"""Role singletons of the provider layer, created lazily behind one registry lock.

:func:`llm_for` returns one cached provider chain per role and :func:`embedder` one
cached embedder; every object shares a single :class:`Store`, :class:`ProviderHealth`,
:class:`ConcurrencyLimits`, :class:`Budgets` and :class:`CallLog`. Without an injected
configuration the first call reads ``providers.yaml`` from
:func:`memini.infra.paths.config_dir`, takes keys from ``os.environ`` and opens the store
at the config's ``cache_path``; later calls only look the object up.

The only module-level mutable state is the registry object, guarded by its own lock.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from memini.infra import paths
from memini.infra.llm_providers import budgets as budgets_module
from memini.infra.llm_providers import perf_export
from memini.infra.llm_providers.budgets import Budgets
from memini.infra.llm_providers.cache import CachedLLM
from memini.infra.llm_providers.calllog import CallLog
from memini.infra.llm_providers.chain import ProviderChain
from memini.infra.llm_providers.config import EMBEDDING, ConfigError, LLMConfig, parse_config
from memini.infra.llm_providers.embed_cache import CachedEmbedder
from memini.infra.llm_providers.embedder import OllamaEmbedder
from memini.infra.llm_providers.health import ProviderHealth
from memini.infra.llm_providers.limiter import ConcurrencyLimits
from memini.infra.llm_providers.perf_summary import RunSummary, compute_summary
from memini.infra.llm_providers.store import Store
from memini.infra.llm_providers.transport import Transport, UrllibTransport
from memini.infra.llm_providers.types import LLM, ROLES, Embedder

__all__ = [
    "configure_for_tests",
    "embedder",
    "export_jsonl",
    "llm_for",
    "question",
    "reset_for_tests",
    "summary",
]

CONFIG_FILE: str = "providers.yaml"
"""Name of the providers config file inside the configuration directory."""

_TOKENS_SQL = "SELECT role, tokens FROM llm_budget_tokens WHERE run_id = ?"


@dataclass(frozen=True)
class _Shared:
    """Everything the role objects share: inputs and the process-wide services."""

    config: LLMConfig
    env: Mapping[str, str]
    clock: Callable[[], float]
    transport: Transport
    store: Store
    health: ProviderHealth
    limits: ConcurrencyLimits
    budgets: Budgets
    call_log: CallLog


def _build_shared(
    config: LLMConfig,
    env: Mapping[str, str],
    clock: Callable[[], float],
    transport: Transport,
    store: Store,
) -> _Shared:
    """Create the shared services over ``store``, scoped to the config's run."""
    return _Shared(
        config=config,
        env=env,
        clock=clock,
        transport=transport,
        store=store,
        health=ProviderHealth(store, clock, run_id=config.run_id),
        limits=ConcurrencyLimits(config),
        budgets=Budgets(config, store),
        call_log=CallLog(store, clock),
    )


def _load_default() -> _Shared:
    """Build the shared services from ``providers.yaml``, ``os.environ`` and the network."""
    file = paths.config_dir() / CONFIG_FILE
    config = parse_config(yaml.safe_load(file.read_text(encoding="utf-8")) or {})
    return _build_shared(config, os.environ, time.time, UrllibTransport(), Store(config.cache_path))


def _embedding_provider(config: LLMConfig) -> tuple[str, str]:
    """Return ``(provider, model)`` of the first provider with an embedding model."""
    for spec in config.providers:
        model = spec.models.get(EMBEDDING)
        if model is not None:
            return spec.name, model.name
    raise ConfigError(f"no provider has an {EMBEDDING!r} model")


@dataclass
class _Registry:
    """The lazily filled singletons; every access holds ``lock``."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    shared: _Shared | None = None
    llms: dict[str, LLM] = field(default_factory=dict)
    embedder: Embedder | None = None

    def ensure_shared(self) -> _Shared:
        """Return the shared services, loading the default ones first; hold ``lock``."""
        if self.shared is None:
            self.shared = _load_default()
        return self.shared

    def clear(self, shared: _Shared | None) -> None:
        """Drop every singleton and install ``shared`` (None: load the default later)."""
        with self.lock:
            self.shared = shared
            self.llms = {}
            self.embedder = None


_REGISTRY = _Registry()


def llm_for(role: str) -> LLM:
    """Return the single cached provider chain of ``role``.

    Raises:
        ValueError: when ``role`` is not one of :data:`ROLES`.

    """
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    with _REGISTRY.lock:
        found = _REGISTRY.llms.get(role)
        if found is not None:
            return found
        s = _REGISTRY.ensure_shared()
        chain = ProviderChain(
            role,
            s.config,
            s.env,
            s.clock,
            s.transport,
            s.store,
            health=s.health,
            limits=s.limits,
            budgets=s.budgets,
            call_log=s.call_log,
        )
        llm = CachedLLM(chain, s.store, s.config, role, call_log=s.call_log, clock=s.clock)
        _REGISTRY.llms[role] = llm
        return llm


def embedder() -> Embedder:
    """Return the single cached embedder of the first provider with an embedding model."""
    with _REGISTRY.lock:
        if _REGISTRY.embedder is not None:
            return _REGISTRY.embedder
        s = _REGISTRY.ensure_shared()
        provider, model = _embedding_provider(s.config)
        inner = OllamaEmbedder(s.config, s.env, s.transport, provider=provider)
        cached = CachedEmbedder(inner, s.store, s.config, model, call_log=s.call_log, clock=s.clock)
        _REGISTRY.embedder = cached
        return cached


def question(question_id: str) -> AbstractContextManager[None]:
    """Return a context manager making ``question_id`` the current question."""
    return budgets_module.question(question_id)


def _shared() -> _Shared:
    """Return the shared services under the registry lock."""
    with _REGISTRY.lock:
        return _REGISTRY.ensure_shared()


def summary() -> RunSummary:
    """Return the performance summary of the call log, with the run's token budgets."""
    s = _shared()
    rows = s.store.execute(_TOKENS_SQL, (s.config.run_id,))
    used = {str(role): int(tokens) for role, tokens in rows}
    return compute_summary(s.call_log.rows(), s.config, used)


def export_jsonl(path: str | os.PathLike[str] | Path) -> int:
    """Write the call log to ``path`` as JSON Lines; return the number of rows."""
    return perf_export.export_jsonl(_shared().call_log, path)


def reset_for_tests() -> None:
    """Drop every singleton and any injected configuration; the next call rebuilds."""
    _REGISTRY.clear(None)


def configure_for_tests(
    *,
    config: LLMConfig,
    env: Mapping[str, str],
    clock: Callable[[], float],
    transport: Transport,
    store: Store,
) -> None:
    """Replace the singletons with ones built from the given inputs on the next call."""
    _REGISTRY.clear(_build_shared(config, env, clock, transport, store))
