"""A generation cache in front of one role's provider chain.

:class:`CachedLLM` wraps a :class:`ProviderChain` and keeps deterministic generations
in the shared :class:`Store` under a content key (:func:`generation_key`). A request
is deterministic, and so cached, when its temperature is 0 or it carries a seed.

Within one process, concurrent misses of the same key are single-flight: one caller
asks the inner chain while the others wait and then read its stored row. Across
processes, each writer inserts with ``INSERT OR IGNORE`` so the first row wins and
the file stays valid; the store keeps the file at mode 0600. A failed inner call
stores nothing and propagates, so a later call retries.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from memini.infra.llm_providers.budgets import current_question
from memini.infra.llm_providers.calllog import CallLog, CallRecord
from memini.infra.llm_providers.chain import ProviderChain, call_cost
from memini.infra.llm_providers.config import LLMConfig
from memini.infra.llm_providers.store import Store
from memini.infra.llm_providers.types import Generation, LLMError

__all__ = ["CachedLLM", "generation_key", "is_cacheable"]

_SCHEMA_NAME = "llm_generation_cache_v1"
_DDL = """
CREATE TABLE IF NOT EXISTS llm_generation_cache (
    key TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    role TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    seconds REAL NOT NULL
)
"""
_SELECT = (
    "SELECT text, provider, model, role, prompt_tokens, completion_tokens, seconds "
    "FROM llm_generation_cache WHERE key = ?"
)
_INSERT = (
    "INSERT OR IGNORE INTO llm_generation_cache "
    "(key, text, provider, model, role, prompt_tokens, completion_tokens, seconds) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)


def generation_key(  # noqa: PLR0917 - positional signature fixed by the interface
    provider_list: Sequence[str],
    model_by_provider: Mapping[str, str | None],
    role: str,
    prompt: str,
    system: str | None,
    json_mode: bool | Mapping[str, Any],
    temperature: float,
    seed: int | None,
) -> str:
    """Return the SHA-256 hex digest of every input that shapes a generation.

    The inputs are encoded as canonical JSON (sorted keys, fixed separators), so equal
    inputs give equal keys and a change in any field gives a different key.
    """
    payload = [
        list(provider_list),
        dict(model_by_provider),
        role,
        prompt,
        system,
        json_mode if isinstance(json_mode, bool) else dict(json_mode),
        float(temperature),
        seed,
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def is_cacheable(temperature: float, seed: int | None) -> bool:
    """Return whether a request is deterministic: temperature 0 or a fixed seed."""
    return temperature == 0 or seed is not None


class _KeyLocks:
    """One lock per key in flight, dropped once no caller holds or waits for it."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, tuple[threading.Lock, int]] = {}

    @contextmanager
    def hold(self, key: str) -> Iterator[None]:
        """Hold the lock of ``key`` for the duration of the block."""
        with self._guard:
            lock, users = self._locks.get(key, (threading.Lock(), 0))
            self._locks[key] = (lock, users + 1)
        try:
            with lock:
                yield
        finally:
            with self._guard:
                lock, users = self._locks[key]
                if users <= 1:
                    del self._locks[key]
                else:
                    self._locks[key] = (lock, users - 1)


class CachedLLM:
    """An ``LLM`` that serves deterministic generations of one role from the store."""

    def __init__(
        self,
        inner: ProviderChain,
        store: Store,
        config: LLMConfig,
        role: str,
        *,
        call_log: CallLog | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Wrap ``inner`` (the chain of ``role``) and cache its results in ``store``.

        With ``call_log`` every cache hit writes one row (the chain logs the misses);
        ``clock`` stamps those rows.
        """
        self._inner = inner
        self._store = store
        self._role = role
        self._config = config
        self._call_log = call_log
        self._clock = clock
        self._providers = config.provider_order(role)
        self._models = {name: config.model_for(name, role) for name in self._providers}
        self._locks = _KeyLocks()
        store.ensure_schema(_SCHEMA_NAME, _DDL)

    def _key(
        self,
        prompt: str,
        system: str | None,
        json_mode: bool | dict[str, Any],
        temperature: float,
        seed: int | None,
    ) -> str:
        """Return the cache key of one request of this role."""
        return generation_key(
            self._providers, self._models, self._role, prompt, system, json_mode, temperature, seed
        )

    def _hit(self, gen: Generation, stage: str | None) -> Generation:
        """Log the cache hit ``gen`` (no tokens charged) and return it."""
        if self._call_log is not None:
            self._call_log.record(
                CallRecord(
                    kind="generate",
                    provider=gen.provider,
                    model=gen.model,
                    role=self._role,
                    stage=stage,
                    question_id=current_question(),
                    prompt_tokens=0,
                    completion_tokens=0,
                    seconds=0.0,
                    retries=0,
                    wait_seconds=0.0,
                    fallbacks=(),
                    cache_hit=True,
                    cost=call_cost(self._config, gen.provider, self._role, 0, 0),
                    texts=None,
                    ts=self._clock(),
                )
            )
        return gen

    def _lookup(self, key: str) -> Generation | None:
        """Return the stored generation of ``key`` marked cached, or None on a miss."""
        rows = self._store.execute(_SELECT, (key,))
        if not rows:
            return None
        text, provider, model, role, prompt_tokens, completion_tokens, seconds = rows[0]
        return Generation(
            str(text),
            str(provider),
            str(model),
            str(role),
            int(prompt_tokens),
            int(completion_tokens),
            float(seconds),
            cached=True,
        )

    def _save(self, key: str, gen: Generation) -> None:
        """Store ``gen`` under ``key`` unless a row is already there."""
        self._store.execute(
            _INSERT,
            (
                key,
                gen.text,
                gen.provider,
                gen.model,
                gen.role,
                gen.prompt_tokens,
                gen.completion_tokens,
                gen.seconds,
            ),
        )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool | dict[str, Any] = False,
        temperature: float = 0.0,
        seed: int | None = None,
        stage: str | None = None,
    ) -> Generation:
        """Return the cached generation on a hit (``cached=True``), else ask the chain.

        Raises:
            LLMError: when the inner chain fails; nothing is stored.

        """
        options: dict[str, Any] = {
            "system": system,
            "json_mode": json_mode,
            "temperature": temperature,
            "seed": seed,
            "stage": stage,
        }
        if not is_cacheable(temperature, seed):
            return self._inner.generate(prompt, **options)
        key = self._key(prompt, system, json_mode, temperature, seed)
        hit = self._lookup(key)
        if hit is not None:
            return self._hit(hit, stage)
        with self._locks.hold(key):
            hit = self._lookup(key)
            if hit is not None:
                return self._hit(hit, stage)
            gen = self._inner.generate(prompt, **options)
            self._save(key, gen)
            return replace(gen, cached=False)

    def generate_many(
        self,
        prompts: Sequence[str],
        *,
        system: str | None = None,
        json_mode: bool | dict[str, Any] = False,
        temperature: float = 0.0,
        seed: int | None = None,
        stage: str | None = None,
    ) -> list[Generation | LLMError]:
        """Serve cached prompts from the store and send only the misses to the chain.

        Results keep the input order; a failed prompt gets its ``LLMError`` and is not
        stored.
        """
        options: dict[str, Any] = {
            "system": system,
            "json_mode": json_mode,
            "temperature": temperature,
            "seed": seed,
            "stage": stage,
        }
        if not is_cacheable(temperature, seed):
            return self._inner.generate_many(prompts, **options)
        keys = [self._key(p, system, json_mode, temperature, seed) for p in prompts]
        results: list[Generation | LLMError | None] = []
        for k in keys:
            found = self._lookup(k)
            results.append(None if found is None else self._hit(found, stage))
        missing = [i for i, found in enumerate(results) if found is None]
        if missing:
            fresh = self._inner.generate_many([prompts[i] for i in missing], **options)
            for index, outcome in zip(missing, fresh, strict=True):
                if isinstance(outcome, Generation):
                    self._save(keys[index], outcome)
                    results[index] = replace(outcome, cached=False)
                else:
                    results[index] = outcome
        return [r for r in results if r is not None]
