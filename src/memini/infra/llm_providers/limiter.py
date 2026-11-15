"""Per-provider concurrency limits shared by every role of the process.

Each provider whose config declares ``max_concurrent`` gets one bounded semaphore,
created lazily under a registry lock. The registry lock is only held while looking
up or creating a semaphore, never while the caller's body runs. A provider without
the key is unlimited and its slot never blocks. Limits are in-process only.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext

from memini.infra.llm_providers.config import LLMConfig

MAX_CONCURRENT_KEY: str = "max_concurrent"
"""Provider limit key bounding simultaneous calls to that provider."""


class ConcurrencyLimits:
    """Bound simultaneous calls per provider, across all roles of the process."""

    def __init__(self, config: LLMConfig) -> None:
        """Keep ``config``; semaphores are created on first use of each provider."""
        self._config = config
        self._lock = threading.Lock()
        self._semaphores: dict[str, threading.BoundedSemaphore | None] = {}

    def _semaphore(self, provider: str) -> threading.BoundedSemaphore | None:
        """Return the provider's shared semaphore, or ``None`` when it is unlimited."""
        with self._lock:
            if provider not in self._semaphores:
                # Role ``None`` selects the provider-level limit: per-role model limits
                # never apply, so every role shares one semaphore. ``LLMConfig.limit``
                # types ``role`` as ``str`` but resolves ``None`` to "no model".
                limit = self._config.limit(provider, None, MAX_CONCURRENT_KEY)  # type: ignore[arg-type]
                self._semaphores[provider] = (
                    None if limit is None else threading.BoundedSemaphore(_positive(limit))
                )
            return self._semaphores[provider]

    def slot(self, provider: str) -> AbstractContextManager[None]:
        """Return a context manager holding one of ``provider``'s slots while it is open.

        Entering blocks until a slot is free; the slot is released on exit, including
        when the body raises. A provider with no ``max_concurrent`` limit never blocks.
        """
        semaphore = self._semaphore(provider)
        if semaphore is None:
            return nullcontext()
        return _held(semaphore)


def _positive(limit: object) -> int:
    """Return ``limit`` as a slot count; raise ``ValueError`` unless it is an int >= 1."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError(f"{MAX_CONCURRENT_KEY} must be an integer >= 1, got {limit!r}")
    return limit


@contextmanager
def _held(semaphore: threading.BoundedSemaphore) -> Iterator[None]:
    """Acquire ``semaphore`` for the duration of the ``with`` body."""
    semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()
