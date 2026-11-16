"""Config-driven retry of a failed model call, per error class.

An error class is a :class:`RetryRule`: a name, a total attempt budget and the
markers that recognise it in an error message (case-insensitive substrings).
The first rule in config order with a matching marker wins; a non-retryable
marker wins over every rule. Attempts are counted per class, while the backoff
``base_delay_s * 2**k`` counts the retries ``k`` already made across all classes.
An unrecognised or deterministic error is raised at once. The rule list lives in
config only; this module ships no default rules.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetryRule:
    """One retryable error class."""

    name: str
    max_attempts: int
    """Total attempts for this error class, at least 1."""
    markers: tuple[str, ...]


@dataclass(frozen=True)
class RetryPolicy:
    """Ordered retry rules, deterministic error markers and the backoff base."""

    rules: tuple[RetryRule, ...]
    non_retryable: tuple[str, ...]
    base_delay_s: float

    def rule_for(self, message: str) -> RetryRule | None:
        """Return the first rule matching ``message``, or ``None`` when none or non-retryable."""
        text = message.casefold()
        if any(marker.casefold() in text for marker in self.non_retryable):
            return None
        for rule in self.rules:
            if any(marker.casefold() in text for marker in rule.markers):
                return rule
        return None

    def delay_s(self, retry_index: int) -> float:
        """Return the backoff before retry number ``retry_index`` (0-based)."""
        return self.base_delay_s * float(2**retry_index)


def _require(raw: Mapping[str, Any], key: str) -> Any:
    """Return ``raw[key]`` or raise :class:`ValueError` naming the missing key."""
    if key not in raw:
        raise ValueError(f"retry config: missing required key {key!r}")
    return raw[key]


def _parse_rule(raw: Mapping[str, Any]) -> RetryRule:
    """Build one :class:`RetryRule` from its config mapping."""
    attempts = _require(raw, "max_attempts")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError(f"retry config: max_attempts must be an int >= 1, got {attempts!r}")
    markers = _require(raw, "markers")
    return RetryRule(
        name=str(_require(raw, "name")),
        max_attempts=attempts,
        markers=tuple(str(marker) for marker in markers),
    )


def parse_retry_policy(raw: Mapping[str, Any]) -> RetryPolicy:
    """Build a :class:`RetryPolicy` from config; raise :class:`ValueError` when invalid."""
    rules = tuple(_parse_rule(rule) for rule in _require(raw, "retry_rules"))
    base = _require(raw, "retry_base_delay_s")
    if isinstance(base, bool) or not isinstance(base, int | float) or base < 0:
        raise ValueError(f"retry config: retry_base_delay_s must be a number >= 0, got {base!r}")
    non_retryable = tuple(str(marker) for marker in raw.get("non_retryable_errors", ()))
    return RetryPolicy(rules=rules, non_retryable=non_retryable, base_delay_s=float(base))


def with_retry[T](
    fn: Callable[[], T],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` until it succeeds, retrying errors the policy recognises.

    A non-``Exception`` (e.g. an interrupt) propagates at once. An error with no
    matching rule, or whose class has used its attempt budget, is re-raised as is.
    """
    attempts: dict[RetryRule, int] = {}
    retries = 0
    while True:
        try:
            return fn()
        except Exception as error:
            rule = policy.rule_for(str(error))
            if rule is None:
                raise
            attempts[rule] = attempts.get(rule, 0) + 1
            if attempts[rule] >= rule.max_attempts:
                raise
            sleep(policy.delay_s(retries))
            retries += 1
