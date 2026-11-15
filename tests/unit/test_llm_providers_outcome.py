"""Acceptance tests for LP-03: outcome classification of one provider response."""

from __future__ import annotations

import dataclasses

import pytest

from memini.infra.llm_providers import outcome as lp_outcome
from memini.infra.llm_providers.types import Reason

Action = lp_outcome.Action
Outcome = lp_outcome.Outcome
classify = lp_outcome.classify
parse_wait_seconds = lp_outcome.parse_wait_seconds

PER_MINUTE_BODY = (
    "Rate limit reached for model `m-1` on tokens per minute (TPM): "
    "Limit 6000, Used 5900, Requested 400. Please try again in 6.5s."
)
PER_DAY_BODY = (
    "Rate limit reached for model `m-1` on tokens per day: "
    "Limit 500000, Used 499990, Requested 400. Please try again in 7m12s."
)
BILLING_PER_MINUTE_BODY = (
    "Rate limit reached for model `m-1` on requests per minute (RPM): "
    "Limit 30, Used 30. Please try again in 2s. "
    "Need more capacity? Upgrade at https://console.example.test/settings/billing"
)


def test_action_members() -> None:
    names = {member.name for member in Action}
    assert names == {"REST", "EXHAUSTED", "RETRY_THEN_NEXT", "NEXT_REQUEST_ONLY"}


def test_outcome_is_frozen_dataclass() -> None:
    outcome = classify(429, {}, PER_MINUTE_BODY)
    assert isinstance(outcome, Outcome)
    assert dataclasses.is_dataclass(outcome)
    assert isinstance(outcome.reason, Reason)
    assert isinstance(outcome.detail, str)
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.action = Action.EXHAUSTED  # type: ignore[misc]


def test_429_per_minute_with_body_wait_rests_for_that_wait() -> None:
    outcome = classify(429, {}, PER_MINUTE_BODY)
    assert outcome.action is Action.REST
    assert outcome.wait_s == pytest.approx(6.5)


def test_429_retry_after_header_wins_over_body() -> None:
    outcome = classify(429, {"Retry-After": "12"}, PER_MINUTE_BODY)
    assert outcome.action is Action.REST
    assert outcome.wait_s == pytest.approx(12.0)


def test_429_retry_after_header_without_body_wait() -> None:
    outcome = classify(429, {"Retry-After": "3"}, "Too many requests")
    assert outcome.action is Action.REST
    assert outcome.wait_s == pytest.approx(3.0)


def test_429_without_wait_uses_rest_default() -> None:
    outcome = classify(429, {}, "Too many requests")
    assert outcome.action is Action.REST
    assert outcome.wait_s == pytest.approx(60.0)


def test_429_without_wait_uses_given_rest_default() -> None:
    outcome = classify(429, {}, "Too many requests", rest_default_s=17.0)
    assert outcome.action is Action.REST
    assert outcome.wait_s == pytest.approx(17.0)


@pytest.mark.parametrize(
    "body",
    [
        PER_DAY_BODY,
        "You have used your daily quota.",
        "Limit exceeded on tokens (tpd) for this model.",
        "Limit exceeded on requests (rpd) for this model.",
    ],
)
def test_429_with_day_marker_is_exhausted(body: str) -> None:
    outcome = classify(429, {}, body)
    assert outcome.action is Action.EXHAUSTED


def test_429_with_custom_exhausted_marker_is_exhausted() -> None:
    body = "Your monthly allowance is used up."
    assert classify(429, {}, body).action is Action.REST
    outcome = classify(429, {}, body, exhausted_markers=("monthly",))
    assert outcome.action is Action.EXHAUSTED


def test_429_per_minute_with_billing_link_rests_not_exhausted() -> None:
    outcome = classify(429, {}, BILLING_PER_MINUTE_BODY)
    assert outcome.action is Action.REST
    assert outcome.wait_s == pytest.approx(2.0)


@pytest.mark.parametrize("status", [401, 402])
def test_auth_and_payment_errors_are_exhausted(status: int) -> None:
    outcome = classify(status, {}, "Invalid API key or payment required")
    assert outcome.action is Action.EXHAUSTED


@pytest.mark.parametrize("status", [500, 502, 503])
def test_server_errors_retry_then_next(status: int) -> None:
    outcome = classify(status, {}, "internal error")
    assert outcome.action is Action.RETRY_THEN_NEXT


def test_timeout_exception_retry_then_next() -> None:
    outcome = classify(None, {}, "", exc=TimeoutError("read timed out"))
    assert outcome.action is Action.RETRY_THEN_NEXT


@pytest.mark.parametrize("status", [400, 413, 422])
def test_rejected_request_is_next_request_only(status: int) -> None:
    outcome = classify(status, {}, "request too large")
    assert outcome.action is Action.NEXT_REQUEST_ONLY


def test_wait_is_not_taken_from_next_line() -> None:
    body = "Rate limit reached on tokens per minute. Please try again in\n6.5s"
    outcome = classify(429, {}, body, rest_default_s=42.0)
    assert outcome.action is Action.REST
    assert outcome.wait_s == pytest.approx(42.0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("try again in 1m2.5s", 62.5),
        ("Please try again in 6.5s.", 6.5),
        ("6.5s", 6.5),
        ("250ms", 0.25),
        ("1m2.5s", 62.5),
    ],
)
def test_parse_wait_seconds(text: str, expected: float) -> None:
    assert parse_wait_seconds(text) == pytest.approx(expected)


def test_parse_wait_seconds_without_wait_is_none() -> None:
    assert parse_wait_seconds("no wait information here") is None
