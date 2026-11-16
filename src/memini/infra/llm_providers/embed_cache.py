"""An embedding cache with batched misses in front of one embedder.

:class:`CachedEmbedder` wraps an :class:`Embedder` and keeps every vector in the shared
:class:`Store` under a content key (:func:`embedding_key`): the SHA-256 of the model,
the kind (query or document) and the text. Texts already stored are served from the
store; the unique missing texts are sent to the inner embedder in batches of the
provider's ``batch_size`` limit, and the result keeps the input order.

Within one process, concurrent misses of the same key are single-flight: one caller
asks the inner embedder while the others wait and then read its stored row. Rows are
written with ``INSERT OR IGNORE`` so the first writer wins across processes. A failed
inner call stores nothing of its batch and propagates.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from typing import Literal

from memini.infra.llm_providers.budgets import current_question
from memini.infra.llm_providers.calllog import CallLog, CallRecord
from memini.infra.llm_providers.config import EMBEDDING, ConfigError, LLMConfig
from memini.infra.llm_providers.store import Store
from memini.infra.llm_providers.types import Embedder

__all__ = ["CachedEmbedder", "embedding_key"]

_SCHEMA_NAME = "llm_embedding_cache_v1"
_DDL = """
CREATE TABLE IF NOT EXISTS llm_embedding_cache (
    key TEXT PRIMARY KEY,
    vector TEXT NOT NULL
)
"""
_SELECT = "SELECT vector FROM llm_embedding_cache WHERE key = ?"
_INSERT = "INSERT OR IGNORE INTO llm_embedding_cache (key, vector) VALUES (?, ?)"
_DELETE = "DELETE FROM llm_embedding_cache"
_BATCH_SIZE_KEY = "batch_size"

Kind = Literal["query", "document"]


def embedding_key(model: str, kind: str, text: str) -> str:
    """Return the SHA-256 hex digest of ``(model, kind, text)`` encoded as JSON."""
    encoded = json.dumps([model, kind, text], separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _batch_size(config: LLMConfig, model: str) -> int | None:
    """Return the batch size of the first embedding provider serving ``model``.

    Returns None (one batch) when no provider serves ``model`` or it sets no limit.

    Raises:
        ConfigError: when the configured value is not a positive int.

    """
    for name in config.provider_order(EMBEDDING):
        if config.model_for(name, EMBEDDING) != model:
            continue
        value = config.limit(name, EMBEDDING, _BATCH_SIZE_KEY)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            msg = f"provider {name!r}: {_BATCH_SIZE_KEY} must be a positive int, got {value!r}"
            raise ConfigError(msg)
        return value
    return None


def _provider(config: LLMConfig, model: str) -> str | None:
    """Return the first embedding provider serving ``model``, or None when none does."""
    for name in config.provider_order(EMBEDDING):
        if config.model_for(name, EMBEDDING) == model:
            return name
    return None


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


class CachedEmbedder:
    """An ``Embedder`` that serves stored vectors and batches the misses to ``inner``."""

    def __init__(
        self,
        inner: Embedder,
        store: Store,
        config: LLMConfig,
        model: str,
        *,
        call_log: CallLog | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Wrap ``inner`` (serving ``model``) and cache its vectors in ``store``.

        With ``call_log`` every batch sent to ``inner`` writes one row, timed and
        stamped with ``clock``.

        Raises:
            ConfigError: when the provider's ``batch_size`` is not a positive int.

        """
        self._inner = inner
        self._store = store
        self._model = model
        self._batch_size = _batch_size(config, model)
        self._provider = _provider(config, model)
        self._call_log = call_log
        self._clock = clock
        self._locks = _KeyLocks()
        store.ensure_schema(_SCHEMA_NAME, _DDL)

    def _lookup(self, key: str) -> list[float] | None:
        """Return the stored vector of ``key``, or None on a miss."""
        rows = self._store.execute(_SELECT, (key,))
        if not rows:
            return None
        return [float(x) for x in json.loads(str(rows[0][0]))]

    def _log(self, texts: int, seconds: float) -> None:
        """Write one row for a batch of ``texts`` embedded in ``seconds``."""
        if self._call_log is None:
            return
        self._call_log.record(
            CallRecord(
                kind="embed",
                provider=self._provider,
                model=self._model,
                role=EMBEDDING,
                stage=None,
                question_id=current_question(),
                prompt_tokens=0,
                completion_tokens=0,
                seconds=seconds,
                retries=0,
                wait_seconds=0.0,
                fallbacks=(),
                cache_hit=False,
                cost=None,
                texts=texts,
                ts=self._clock(),
            )
        )

    def _batches(self, items: list[str]) -> Iterator[list[str]]:
        """Split ``items`` into consecutive batches of the configured size."""
        if not items:
            return
        size = self._batch_size or len(items)
        for start in range(0, len(items), size):
            yield items[start : start + size]

    def clear(self) -> int:
        """Delete every stored vector in this store; return how many were deleted.

        Only the embedding table is touched; other tables of the store are kept.
        """
        with self._store.transaction() as conn:
            return conn.execute(_DELETE).rowcount

    def embed(self, texts: Sequence[str], *, kind: Kind) -> list[list[float]]:
        """Return one vector per text in input order; only unique misses reach ``inner``."""
        unique: dict[str, str] = {}
        for text in texts:
            unique.setdefault(text, embedding_key(self._model, kind, text))
        found: dict[str, list[float]] = {}
        for text, key in unique.items():
            vector = self._lookup(key)
            if vector is not None:
                found[text] = vector
        missing = [text for text in unique if text not in found]
        if missing:
            with ExitStack() as stack:
                for key in sorted({unique[text] for text in missing}):
                    stack.enter_context(self._locks.hold(key))
                still: list[str] = []
                for text in missing:
                    vector = self._lookup(unique[text])
                    if vector is None:
                        still.append(text)
                    else:
                        found[text] = vector
                for batch in self._batches(still):
                    started = self._clock()
                    vectors = self._inner.embed(batch, kind=kind)
                    self._log(len(batch), self._clock() - started)
                    for text, vector in zip(batch, vectors, strict=True):
                        self._store.execute(_INSERT, (unique[text], json.dumps(vector)))
                        found[text] = [float(x) for x in vector]
        return [found[text] for text in texts]
