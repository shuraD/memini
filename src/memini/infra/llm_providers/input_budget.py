"""An input budget derived from the configured context and output limits.

The budget of a provider serving a role is its context size (limit ``num_ctx``) minus
the room reserved for the reply (limit ``max_tokens``). Tokens are estimated from the
text length with a caller-given characters-per-token ratio; no tokenizer is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from memini.infra.llm_providers.config import ConfigError, LLMConfig

__all__ = ["InputBudget", "PromptTooLong", "input_budget"]

_NUM_CTX_KEY = "num_ctx"
"""Limit key of the context size in tokens."""

_MAX_TOKENS_KEY = "max_tokens"
"""Limit key of the reply size in tokens."""


class PromptTooLong(ValueError):  # noqa: N818 - public name fixed by the interface
    """A prompt's estimated tokens exceed the input budget."""

    def __init__(self, tokens: int, limit: int) -> None:
        """Store the estimated ``tokens`` and the budget's ``limit``."""
        self.tokens: int = tokens
        self.limit: int = limit
        super().__init__(f"prompt of about {tokens} tokens exceeds the input budget of {limit}")


@dataclass(frozen=True)
class InputBudget:
    """The tokens a prompt may use: the context size minus the reply size."""

    num_ctx: int
    num_predict: int
    chars_per_token: float

    @property
    def input_tokens(self) -> int:
        """Return the tokens left for the prompt."""
        return self.num_ctx - self.num_predict

    def tokens(self, text: str) -> int:
        """Return the estimated tokens of ``text``, rounded up."""
        return math.ceil(len(text) / self.chars_per_token)

    def fits(self, prompt: str) -> bool:
        """Return whether ``prompt`` fits the input budget."""
        return self.tokens(prompt) <= self.input_tokens

    def check(self, prompt: str) -> None:
        """Raise :class:`PromptTooLong` when ``prompt`` does not fit the input budget."""
        tokens = self.tokens(prompt)
        if tokens > self.input_tokens:
            raise PromptTooLong(tokens, self.input_tokens)


def _int_limit(config: LLMConfig, provider: str, role: str, key: str) -> int:
    """Return the integer limit ``key`` of ``provider`` for ``role``, or raise ConfigError."""
    value = config.limit(provider, role, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"provider {provider!r} role {role!r}: limit {key!r} must be an integer")
    return value


def input_budget(
    config: LLMConfig, provider: str, role: str, chars_per_token: float
) -> InputBudget:
    """Return the input budget of ``provider`` serving ``role``.

    The role's model limits override the provider's.

    Raises:
        ConfigError: when ``num_ctx`` or ``max_tokens`` is missing, when ``num_ctx``
            does not exceed ``max_tokens``, or when ``chars_per_token`` is not positive.

    """
    if chars_per_token <= 0:
        raise ConfigError(f"chars_per_token must be positive, got {chars_per_token}")
    num_ctx = _int_limit(config, provider, role, _NUM_CTX_KEY)
    num_predict = _int_limit(config, provider, role, _MAX_TOKENS_KEY)
    if num_ctx <= num_predict:
        raise ConfigError(
            f"provider {provider!r} role {role!r}: {_NUM_CTX_KEY} {num_ctx} must exceed "
            f"{_MAX_TOKENS_KEY} {num_predict}"
        )
    return InputBudget(num_ctx=num_ctx, num_predict=num_predict, chars_per_token=chars_per_token)
