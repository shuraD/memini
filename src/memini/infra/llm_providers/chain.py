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
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from memini.infra.llm_providers.budgets import Budgets
from memini.infra.llm_providers.config import LLMConfig
from memini.infra.llm_providers.health import ProviderHealth
from memini.infra.llm_providers.limiter import ConcurrencyLimits
from memini.infra.llm_providers.outcome import Action, Outcome
from memini.infra.llm_providers.provider_call import ProviderFailure, call_provider
from memini.infra.llm_providers.store import Store
from memini.infra.llm_providers.transport import Transport
from memini.infra.llm_providers.types import Attempt, Generation, LLMError, Reason

__all__ = ["ProviderChain"]


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
    ) -> None:
        """Serve ``role`` from ``config`` using ``transport``, keys from ``env``.

        ``health`` and ``budgets`` default to ones kept in ``store`` (scoped to the
        config's run); without ``limits`` sends are not bounded per provider.
        """
        self._role = role
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
    ) -> Generation:
        """Generate one completion for ``prompt`` from the first provider that answers.

        ``stage`` is accepted for the protocol and not used here.

        Raises:
            BudgetExceeded: before any send when a spending cap is reached.
            LLMError: when every provider failed, naming each with its reason.

        """
        del stage
        self._budgets.check(self._role)
        attempts: list[Attempt] = []
        for name in self._config.provider_order(self._role):
            recorded = self._health.unavailable_reason(name)
            if recorded is not None:
                attempts.append(Attempt(name, _health_reason(recorded), "skipped: " + recorded))
                continue
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
                        transport=self._transport,
                        clock=self._clock,
                    )
            except ProviderFailure as failure:
                attempts.append(failure.attempt)
                if failure.outcome is not None:
                    self._apply(name, failure.outcome)
                continue
            spent = (generation.prompt_tokens, generation.completion_tokens)
            self._budgets.record(self._role, *spent)
            return generation
        raise LLMError(tuple(attempts), self._role)
