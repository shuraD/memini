"""Per-question telemetry: stage timings, answer usage and per-call limits.

A :class:`StageTimer` records the seconds spent in each named stage of one question.
:func:`answer_usage` sums the token use and cost of one answer from the provider call
log rows of its question id. :func:`call_limits` resolves the size and time limits of
one LLM call from the providers config. There is no per-question time limit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass

from memini.infra.llm_providers.calllog import CallRecord
from memini.infra.llm_providers.config import ConfigError, LLMConfig
from memini.infra.llm_providers.input_budget import input_budget

__all__ = [
    "TIMEOUT_KEY",
    "AnswerUsage",
    "CallLimits",
    "StageTimer",
    "answer_usage",
    "call_limits",
]

TIMEOUT_KEY: str = "timeout_s"
"""Limit key of the per-call timeout in the providers config."""


class StageTimer:
    """Sum the seconds spent in each named stage, measured with a caller-given clock."""

    def __init__(self, clock: Callable[[], float]) -> None:
        """Use ``clock`` as the time source; its current reading is the start."""
        self._clock = clock
        self._start = clock()
        self._stages: dict[str, float] = {}

    def stage(self, name: str) -> AbstractContextManager[None]:
        """Return a context that adds the time spent in its body to stage ``name``.

        The time is recorded also when the body raises; the exception propagates.
        """
        return self._timed(name)

    @contextmanager
    def _timed(self, name: str) -> Iterator[None]:
        """Yield once, then add the clock time spent to stage ``name``, even on error."""
        before = self._clock()
        try:
            yield
        finally:
            self._accumulate(name, self._clock() - before)

    def add(self, name: str, seconds: float) -> None:
        """Add ``seconds`` to stage ``name``; raise ValueError when ``seconds`` < 0."""
        if seconds < 0:
            raise ValueError(f"stage {name!r}: seconds must not be negative, got {seconds}")
        self._accumulate(name, seconds)

    def _accumulate(self, name: str, seconds: float) -> None:
        """Add ``seconds`` to stage ``name``, creating it at the end of the order."""
        self._stages[name] = self._stages.get(name, 0.0) + seconds

    def timings(self) -> dict[str, float]:
        """Return a new dict of stage name to summed seconds, in order of first use."""
        return dict(self._stages)

    def total(self) -> float:
        """Return the seconds elapsed since the timer was built."""
        return self._clock() - self._start


@dataclass(frozen=True)
class AnswerUsage:
    """Token use, cost and time of the provider calls made for one question."""

    calls: int
    cache_hits: int
    prompt_tokens: int
    completion_tokens: int
    cost: float | None
    seconds: float


def answer_usage(rows: Iterable[CallRecord], question_id: str) -> AnswerUsage:
    """Sum the call-log ``rows`` whose question id is ``question_id``.

    Cache hits count in ``calls``, ``cache_hits`` and ``seconds`` only; tokens and
    cost come from the other rows. ``cost`` is None when no such row has a cost.
    """
    calls = cache_hits = prompt_tokens = completion_tokens = 0
    cost: float | None = None
    seconds = 0.0
    for rec in rows:
        if rec.question_id != question_id:
            continue
        calls += 1
        seconds += rec.seconds
        if rec.cache_hit:
            cache_hits += 1
            continue
        prompt_tokens += rec.prompt_tokens
        completion_tokens += rec.completion_tokens
        if rec.cost is not None:
            cost = rec.cost if cost is None else cost + rec.cost
    return AnswerUsage(calls, cache_hits, prompt_tokens, completion_tokens, cost, seconds)


@dataclass(frozen=True)
class CallLimits:
    """Size and time limits of one LLM call."""

    input_tokens: int
    max_tokens: int
    timeout_s: float


def call_limits(config: LLMConfig, provider: str, role: str, chars_per_token: float) -> CallLimits:
    """Return the limits of ``provider`` serving ``role``.

    The role's model limits override the provider's.

    Raises:
        ConfigError: when the input budget cannot be derived, or when the timeout is
            missing, a bool, not a number, or not positive.

    """
    budget = input_budget(config, provider, role, chars_per_token)
    timeout = config.limit(provider, role, TIMEOUT_KEY)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not timeout > 0:
        raise ConfigError(
            f"provider {provider!r} role {role!r}: limit {TIMEOUT_KEY!r} must be a "
            f"positive number, got {timeout!r}"
        )
    return CallLimits(
        input_tokens=budget.input_tokens,
        max_tokens=budget.num_predict,
        timeout_s=float(timeout),
    )
