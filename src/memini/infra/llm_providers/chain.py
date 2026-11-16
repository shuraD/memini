"""The provider chain of one role: try its providers in order and fall through.

For each request the chain checks the budget, then walks the role's provider order
(:meth:`LLMConfig.provider_order`). A provider marked off in the shared health is
skipped with its recorded reason; otherwise one generation is attempted and the
outcome's action is applied: a rest marks the provider off until the stated wait
passes, an exhausted outcome marks it off for the run, and every other failure only
moves the same request to the next provider. When every provider fails, the error
names each provider with its reason.

No lock of this module is held while a request is in flight; only the optional
per-provider concurrency slot bounds simultaneous sends.

With a :class:`CallLog`, every :meth:`ProviderChain.generate` that reaches the
providers writes exactly one row, also when every provider failed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Protocol

from memini.infra.llm_providers.batch import run_many
from memini.infra.llm_providers.budgets import Budgets, current_question
from memini.infra.llm_providers.calllog import CallLog, CallRecord
from memini.infra.llm_providers.config import LLMConfig
from memini.infra.llm_providers.health import ProviderHealth
from memini.infra.llm_providers.limiter import MAX_CONCURRENT_KEY, ConcurrencyLimits
from memini.infra.llm_providers.outcome import Action, Outcome
from memini.infra.llm_providers.provider_call import ProviderFailure, call_provider
from memini.infra.llm_providers.store import Store
from memini.infra.llm_providers.transport import HttpResponse, Transport
from memini.infra.llm_providers.types import Attempt, Generation, LLMError, Reason

__all__ = ["DEFAULT_BATCH_WORKERS", "ProviderChain", "call_cost"]

DEFAULT_BATCH_WORKERS: int = 8
"""Pool size of :meth:`ProviderChain.generate_many` when a provider has no concurrency cap."""

_PER_MTOK: float = 1e6
"""Prices are configured per million tokens."""


def call_cost(
    config: LLMConfig, provider: str, role: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Return the cost of the tokens at the configured price, or None without a price."""
    price = config.price(provider, role)
    if price is None:
        return None
    price_in, price_out = price
    return prompt_tokens * price_in / _PER_MTOK + completion_tokens * price_out / _PER_MTOK


class _Sender[R](Protocol):
    """Anything that sends one request of type ``R`` and returns the response."""

    def send(self, request: R) -> HttpResponse:
        """Send ``request``."""
        ...


class _CountingTransport[R]:
    """A transport that counts the sends it forwards to ``inner``."""

    def __init__(self, inner: _Sender[R]) -> None:
        self._inner = inner
        self.sends = 0

    def send(self, request: R) -> HttpResponse:
        """Forward ``request`` and count it."""
        self.sends += 1
        return self._inner.send(request)


def _health_reason(recorded: str) -> Reason:
    """Map a reason string stored in the health marks back to a :class:`Reason`."""
    try:
        return Reason(recorded)
    except ValueError:
        return Reason.EXHAUSTED


class ProviderChain:
    """Generate for one role by trying its providers in order with fallback."""

    def __init__(  # noqa: PLR0917 - positional signature fixed by the interface
        self,
        role: str,
        config: LLMConfig,
        env: Mapping[str, str],
        clock: Callable[[], float],
        transport: Transport,
        store: Store,
        *,
        health: ProviderHealth | None = None,
        limits: ConcurrencyLimits | None = None,
        budgets: Budgets | None = None,
        call_log: CallLog | None = None,
    ) -> None:
        """Serve ``role`` from ``config`` using ``transport``, keys from ``env``.

        ``health`` and ``budgets`` default to ones kept in ``store`` (scoped to the
        config's run); without ``limits`` sends are not bounded per provider;
        without ``call_log`` nothing is logged.
        """
        self._role = role
        self._call_log = call_log
        self._config = config
        self._env = env
        self._clock = clock
        self._transport = transport
        if health is None:
            health = ProviderHealth(store, clock, run_id=config.run_id)
        if budgets is None:
            budgets = Budgets(config, store)
        self._health = health
        self._limits = limits
        self._budgets = budgets

    def _slot(self, provider: str) -> AbstractContextManager[None]:
        """Return the provider's concurrency slot, or a no-op one without limits."""
        return nullcontext() if self._limits is None else self._limits.slot(provider)

    def _apply(self, provider: str, outcome: Outcome) -> None:
        """Record the health mark the outcome's action asks for, if any."""
        if outcome.action is Action.EXHAUSTED:
            self._health.mark_exhausted(provider, outcome.reason.value)
        elif outcome.action is Action.REST and outcome.wait_s is not None:
            until = self._clock() + outcome.wait_s
            self._health.mark_resting(provider, until, outcome.reason.value)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool | dict[str, Any] = False,
        temperature: float = 0.0,
        seed: int | None = None,
        stage: str | None = None,
        model: str | None = None,
    ) -> Generation:
        """Generate one completion for ``prompt`` from the first provider that answers.

        ``stage`` only labels the call-log row. ``model`` names a model for this call
        and is passed to every provider attempted (None: each provider's role model).

        Raises:
            BudgetExceeded: before any send when a spending cap is reached.
            LLMError: when every provider failed, naming each with its reason.

        """
        self._budgets.check(self._role)
        attempts: list[Attempt] = []
        waited = 0.0
        started = self._clock()
        for name in self._config.provider_order(self._role):
            recorded = self._health.unavailable_reason(name)
            if recorded is not None:
                attempts.append(Attempt(name, _health_reason(recorded), "skipped: " + recorded))
                continue
            counting = _CountingTransport(self._transport)
            try:
                with self._slot(name):
                    generation = call_provider(
                        self._config.provider(name),
                        self._config,
                        self._role,
                        prompt,
                        system=system,
                        json_mode=json_mode,
                        temperature=temperature,
                        seed=seed,
                        env=self._env,
                        transport=counting,
                        clock=self._clock,
                        model=model,
                    )
            except ProviderFailure as failure:
                attempts.append(failure.attempt)
                outcome = failure.outcome
                if outcome is not None:
                    self._apply(name, outcome)
                    if outcome.reason is Reason.RATE_PER_MINUTE and outcome.wait_s is not None:
                        waited += outcome.wait_s
                continue
            spent = (generation.prompt_tokens, generation.completion_tokens)
            self._budgets.record(self._role, *spent)
            self._log(
                generation,
                stage=stage,
                retries=max(0, counting.sends - 1),
                waited=waited,
                attempts=attempts,
            )
            return generation
        self._log(
            None,
            stage=stage,
            retries=0,
            waited=waited,
            attempts=attempts,
            seconds=self._clock() - started,
        )
        raise LLMError(tuple(attempts), self._role)

    def _log(
        self,
        gen: Generation | None,
        *,
        stage: str | None,
        retries: int,
        waited: float,
        attempts: Sequence[Attempt],
        seconds: float | None = None,
    ) -> None:
        """Write one row for ``gen``, or for a call where every provider failed (None).

        ``seconds`` defaults to the generation's own duration.
        """
        if self._call_log is None:
            return
        tokens = (0, 0) if gen is None else (gen.prompt_tokens, gen.completion_tokens)
        cost = None if gen is None else call_cost(self._config, gen.provider, self._role, *tokens)
        if seconds is None:
            seconds = 0.0 if gen is None else gen.seconds
        self._call_log.record(
            CallRecord(
                kind="generate",
                provider=None if gen is None else gen.provider,
                model=None if gen is None else gen.model,
                role=self._role,
                stage=stage,
                question_id=current_question(),
                prompt_tokens=tokens[0],
                completion_tokens=tokens[1],
                seconds=seconds,
                retries=retries,
                wait_seconds=waited,
                fallbacks=tuple((a.provider, a.reason.value) for a in attempts),
                cache_hit=False,
                cost=cost,
                texts=None,
                ts=self._clock(),
            )
        )

    def _batch_workers(self, count: int) -> int:
        """Return how many prompts of a batch of ``count`` may run at once.

        When every provider of the role declares ``max_concurrent``, their sum bounds
        the pool (no more requests can be in flight anyway); otherwise the pool is
        :data:`DEFAULT_BATCH_WORKERS`. The per-provider slots still bound each send.
        """
        total = 0
        for name in self._config.provider_order(self._role):
            limit = self._config.limit(name, self._role, MAX_CONCURRENT_KEY)
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                total = DEFAULT_BATCH_WORKERS
                break
            total += limit
        return max(1, min(count, total or DEFAULT_BATCH_WORKERS))

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
        """Generate for every prompt in parallel with the same options, in input order.

        Each prompt is one :meth:`generate` call (its own budget check and fallback);
        a prompt that ends in :class:`LLMError` (or :class:`BudgetExceeded`) gets that
        error in its slot while the others proceed.

        Raises:
            Exception: any error other than :class:`LLMError` propagates.

        """

        def one(prompt: str) -> Generation:
            return self.generate(
                prompt,
                system=system,
                json_mode=json_mode,
                temperature=temperature,
                seed=seed,
                stage=stage,
            )

        return run_many(one, prompts, max_workers=self._batch_workers(len(prompts)))
