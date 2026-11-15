"""Classify one provider response into the single action the caller should take.

Pure functions only: no sleeping, no health state and no network access live here.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from memini.infra.llm_providers.types import Reason

_DETAIL_MAX = 200
"""Longest provider detail kept in an ``Outcome``."""

_NUMBER = r"\d+(?:\.\d+)?"
_DURATION = rf"(?:(?P<min>{_NUMBER})m(?!s))?[ \t]*(?P<val>{_NUMBER})(?P<unit>ms|s)\b"
"""A single-line duration such as ``1m2.5s``, ``6.5s`` or ``250ms`` (never spans lines)."""

_WAIT_HINT = r"try again in[ \t]*(?P<rest>[^\r\n]*)"
"""The wait hint providers put in a rate-limit body; only the rest of its line is read."""

_RETRY_AFTER = "retry-after"

_TOO_MANY_REQUESTS = 429
_EXHAUSTED_STATUSES = frozenset({401, 402})
"""Unauthorized and payment required: the provider cannot serve this key at all."""
_CLIENT_ERROR_MIN = 400
_SERVER_ERROR_MIN = 500


class Action(enum.Enum):
    """What the caller should do after one provider attempt."""

    REST = "rest"
    EXHAUSTED = "exhausted"
    RETRY_THEN_NEXT = "retry_then_next"
    NEXT_REQUEST_ONLY = "next_request_only"


@dataclass(frozen=True)
class Outcome:
    """The classified result of one attempt: action, reason, wait and provider detail."""

    action: Action
    reason: Reason
    wait_s: float | None
    detail: str


def parse_wait_seconds(text: str) -> float | None:
    """Return the first duration in ``text`` in seconds, or ``None`` if there is none.

    Accepts ``1m2.5s``, ``6.5s`` and ``250ms``; a duration never spans a line break.
    """
    match = re.search(_DURATION, text, re.IGNORECASE)
    if match is None:
        return None
    value = float(match.group("val"))
    seconds = value / 1000.0 if match.group("unit").lower() == "ms" else value
    minutes = match.group("min")
    return seconds + (float(minutes) * 60.0 if minutes else 0.0)


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """Return a non-negative ``Retry-After`` value in seconds, or ``None``."""
    for name, value in headers.items():
        if name.lower() == _RETRY_AFTER:
            try:
                seconds = float(value.strip())
            except ValueError:
                return None
            return seconds if seconds >= 0 else None
    return None


def _body_wait_seconds(body: str) -> float | None:
    """Return the wait named on the same line as the body's wait hint, or ``None``."""
    hint = re.search(_WAIT_HINT, body, re.IGNORECASE)
    return parse_wait_seconds(hint.group("rest")) if hint else None


def _has_marker(body: str, markers: Sequence[str]) -> bool:
    """Return whether ``body`` contains any marker as a whole word, ignoring case."""
    return any(
        re.search(rf"\b{re.escape(marker)}\b", body, re.IGNORECASE) for marker in markers if marker
    )


def _detail(status: int | None, body: str, exc: BaseException | None) -> str:
    """Build a short single-line detail from the status, exception and body."""
    parts = [str(status)] if status is not None else []
    if exc is not None:
        parts.append(f"{type(exc).__name__}: {exc}")
    first_line = body.strip().splitlines()[0] if body.strip() else ""
    if first_line:
        parts.append(first_line)
    return " ".join(parts)[:_DETAIL_MAX]


def _is_timeout(exc: BaseException) -> bool:
    """Return whether ``exc`` is a timeout, including client libraries' own timeout types."""
    return isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()


def classify(
    status: int | None,
    headers: Mapping[str, str],
    body: str,
    *,
    exc: BaseException | None = None,
    rest_default_s: float = 60.0,
    exhausted_markers: Sequence[str] = ("per day", "daily", "tpd", "rpd"),
) -> Outcome:
    """Map an HTTP status, headers, body text or exception to one ``Outcome``.

    429 rests (``Retry-After`` beats the body's wait, else ``rest_default_s``) unless
    the body names an exhausted marker; 401/402 are exhausted; 5xx, timeouts and
    transport errors retry then move on; any other 4xx skips this provider for this
    request only.
    """
    detail = _detail(status, body, exc)
    if exc is not None:
        reason = Reason.TIMEOUT if _is_timeout(exc) else Reason.SERVER_ERROR
        return Outcome(Action.RETRY_THEN_NEXT, reason, None, detail)
    if status == _TOO_MANY_REQUESTS:
        if _has_marker(body, exhausted_markers):
            return Outcome(Action.EXHAUSTED, Reason.EXHAUSTED, None, detail)
        wait = _retry_after_seconds(headers)
        if wait is None:
            wait = _body_wait_seconds(body)
        return Outcome(
            Action.REST,
            Reason.RATE_PER_MINUTE,
            rest_default_s if wait is None else wait,
            detail,
        )
    if status in _EXHAUSTED_STATUSES:
        return Outcome(Action.EXHAUSTED, Reason.EXHAUSTED, None, detail)
    if status is not None and _CLIENT_ERROR_MIN <= status < _SERVER_ERROR_MIN:
        return Outcome(Action.NEXT_REQUEST_ONLY, Reason.REQUEST_REJECTED, None, detail)
    return Outcome(Action.RETRY_THEN_NEXT, Reason.SERVER_ERROR, None, detail)
