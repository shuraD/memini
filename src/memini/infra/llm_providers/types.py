"""Public value types of the LLM provider layer.

Holds only data shapes and the error raised when every provider fails; no logic, no
configuration and no network access live here.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

ROLES: tuple[str, ...] = ("classifier", "filter", "answer", "extraction", "summary", "judge")
"""The roles a caller may request a generation for, in their canonical order."""


class Reason(enum.Enum):
    """Why one provider attempt failed; each value is the member name in lower case."""

    RATE_PER_MINUTE = "rate_per_minute"
    EXHAUSTED = "exhausted"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    REQUEST_REJECTED = "request_rejected"
    NO_KEY = "no_key"
    NO_MODEL = "no_model"
    BUDGET = "budget"


@dataclass(frozen=True)
class Generation:
    """One completed generation and the facts about how it was produced."""

    text: str
    provider: str
    model: str
    role: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    cached: bool


@dataclass(frozen=True)
class Attempt:
    """One failed try at a provider: who, why, and the provider's own detail."""

    provider: str
    reason: Reason
    detail: str


class LLMError(Exception):
    """Raised when every provider failed for a role; names each attempt in order."""

    def __init__(self, attempts: tuple[Attempt, ...], role: str) -> None:
        """Store the attempts (as a tuple) and the role, and build the message."""
        self.attempts: tuple[Attempt, ...] = tuple(attempts)
        self.role: str = role
        listed = "; ".join(f"{a.provider}: {a.reason.value} ({a.detail})" for a in self.attempts)
        super().__init__(f"all providers failed for role {role!r}: {listed}")


class BudgetExceeded(LLMError):  # noqa: N818 - public name fixed by the spec
    """Raised when a spending cap stops the call; same shape as ``LLMError``."""


@runtime_checkable
class LLM(Protocol):
    """A text generator that may fall back across providers."""

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
        """Generate one completion for ``prompt``."""
        ...

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
        """Generate one completion per prompt; a failed prompt yields its ``LLMError``."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """A text embedder."""

    def embed(
        self, texts: Sequence[str], *, kind: Literal["query", "document"]
    ) -> list[list[float]]:
        """Return one vector per text, embedded as a query or as a document."""
        ...
