"""Run one function over many items in parallel, keeping the input order.

Each item runs in a worker thread of a bounded pool. An :class:`LLMError` raised by
an item (including :class:`BudgetExceeded`) becomes that item's value, so one failed
item never stops the others; any other exception is a programming error and
propagates to the caller.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from memini.infra.llm_providers.types import LLMError

__all__ = ["run_many"]


def run_many[T](
    fn: Callable[[str], T], items: Sequence[str], *, max_workers: int
) -> list[T | LLMError]:
    """Return ``fn(item)`` for every item, in input order, using up to ``max_workers`` threads.

    An item whose call raises :class:`LLMError` gets that error as its value.

    Raises:
        ValueError: when ``max_workers`` is below 1.
        Exception: any non-:class:`LLMError` raised by ``fn`` is re-raised.

    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers!r}")
    if not items:
        return []
    results: list[T | LLMError] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        futures = [pool.submit(fn, item) for item in items]
        try:
            for future in futures:
                try:
                    results.append(future.result())
                except LLMError as error:
                    results.append(error)
        except BaseException:
            for pending in futures:
                pending.cancel()
            raise
    return results
