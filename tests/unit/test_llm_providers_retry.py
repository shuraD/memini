"""Acceptance tests for LT-03: config-driven retry policy per error class."""

from __future__ import annotations

import json
from typing import Any

import pytest

from memini.infra.llm_providers.config import parse_config
from memini.infra.llm_providers.outcome import Action
from memini.infra.llm_providers.provider_call import ProviderFailure, call_provider
from memini.infra.llm_providers.retry import (
    RetryPolicy,
    RetryRule,
    parse_retry_policy,
    with_retry,
)
from memini.infra.llm_providers.transport import HttpResponse, TransportError

FAKE_KEY = "test-key"
KEY_ENV = "MEMINI_LT03_FAKE_KEY"


def policy(
    rules: tuple[RetryRule, ...],
    non_retryable: tuple[str, ...] = (),
    base: float = 1.5,
) -> RetryPolicy:
    return RetryPolicy(rules=rules, non_retryable=non_retryable, base_delay_s=base)


RULE_A = RetryRule(name="class-a", max_attempts=2, markers=("Alpha Fault",))
RULE_B = RetryRule(name="class-b", max_attempts=5, markers=("beta-fault",))


class Scripted:
    """Calls a script of exceptions/values in order and counts the calls."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls = 0

    def __call__(self) -> Any:
        self.calls += 1
        if not self.script:
            raise AssertionError("fn called more often than scripted")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


# --- rule_for ---------------------------------------------------------------


def test_rule_for_matches_substring_ignoring_case_on_both_sides() -> None:
    p = policy((RULE_A, RULE_B))
    assert p.rule_for("server said: ALPHA FAULT now") == RULE_A
    assert p.rule_for("xx BETA-FAULT yy") == RULE_B
    upper = RetryRule(name="up", max_attempts=2, markers=("UPPER marker",))
    assert policy((upper,)).rule_for("an upper MARKER here") == upper


def test_rule_for_returns_first_matching_rule_in_order() -> None:
    first = RetryRule(name="first", max_attempts=2, markers=("nope", "shared"))
    second = RetryRule(name="second", max_attempts=3, markers=("shared",))
    assert policy((first, second)).rule_for("a shared failure") == first
    assert policy((second, first)).rule_for("a shared failure") == second


def test_rule_for_non_retryable_wins_over_rules() -> None:
    p = policy((RULE_A,), non_retryable=("Invalid Argument",))
    assert p.rule_for("alpha fault: invalid argument") is None


def test_rule_for_no_match_returns_none() -> None:
    assert policy((RULE_A, RULE_B)).rule_for("something else entirely") is None


def test_delay_s_is_exponential() -> None:
    p = policy((RULE_A,), base=1.5)
    assert p.delay_s(0) == 1.5
    assert p.delay_s(1) == 3.0
    assert p.delay_s(3) == 12.0


# --- with_retry -------------------------------------------------------------


def test_success_at_once_calls_once_and_never_sleeps() -> None:
    fn = Scripted(["value"])
    sleeps = Sleeps()
    assert with_retry(fn, policy((RULE_A,)), sleep=sleeps) == "value"
    assert fn.calls == 1
    assert sleeps.delays == []


def test_retries_matching_error_then_returns_first_success() -> None:
    fn = Scripted([RuntimeError("beta-fault 1"), RuntimeError("beta-fault 2"), 42, 43])
    sleeps = Sleeps()
    assert with_retry(fn, policy((RULE_B,)), sleep=sleeps) == 42
    assert fn.calls == 3
    assert sleeps.delays == [1.5, 3.0]


def test_exhausted_class_reraises_same_exception_without_further_sleep() -> None:
    last = RuntimeError("alpha fault second")
    fn = Scripted([RuntimeError("alpha fault first"), last, "never"])
    sleeps = Sleeps()
    with pytest.raises(RuntimeError) as info:
        with_retry(fn, policy((RULE_A,)), sleep=sleeps)
    assert info.value is last
    assert fn.calls == 2
    assert sleeps.delays == [1.5]


def test_attempts_counted_per_class_and_backoff_across_classes() -> None:
    last = RuntimeError("alpha fault again")
    fn = Scripted([RuntimeError("alpha fault"), RuntimeError("beta-fault"), last, "never"])
    sleeps = Sleeps()
    with pytest.raises(RuntimeError) as info:
        with_retry(fn, policy((RULE_A, RULE_B)), sleep=sleeps)
    assert info.value is last
    assert fn.calls == 3
    assert sleeps.delays == [1.5, 3.0]


def test_six_attempt_rule_always_failing() -> None:
    rule = RetryRule(name="six", max_attempts=6, markers=("flaky",))
    fn = Scripted([ValueError(f"flaky {i}") for i in range(6)] + ["never"])
    sleeps = Sleeps()
    with pytest.raises(ValueError, match="flaky 5"):
        with_retry(fn, policy((rule,)), sleep=sleeps)
    assert fn.calls == 6
    assert sleeps.delays == [1.5, 3.0, 6.0, 12.0, 24.0]


def test_single_attempt_rule_raises_at_once() -> None:
    rule = RetryRule(name="once", max_attempts=1, markers=("flaky",))
    err = RuntimeError("flaky")
    fn = Scripted([err, "never"])
    sleeps = Sleeps()
    with pytest.raises(RuntimeError) as info:
        with_retry(fn, policy((rule,)), sleep=sleeps)
    assert info.value is err
    assert fn.calls == 1
    assert sleeps.delays == []


def test_unrecognised_error_raised_at_once() -> None:
    err = RuntimeError("totally unknown")
    fn = Scripted([err, "never"])
    sleeps = Sleeps()
    with pytest.raises(RuntimeError) as info:
        with_retry(fn, policy((RULE_A, RULE_B)), sleep=sleeps)
    assert info.value is err
    assert fn.calls == 1
    assert sleeps.delays == []


def test_non_retryable_error_raised_at_once() -> None:
    err = RuntimeError("beta-fault: permission denied")
    fn = Scripted([err, "never"])
    sleeps = Sleeps()
    with pytest.raises(RuntimeError) as info:
        with_retry(fn, policy((RULE_B,), non_retryable=("permission denied",)), sleep=sleeps)
    assert info.value is err
    assert fn.calls == 1
    assert sleeps.delays == []


def test_base_exception_propagates_without_retry() -> None:
    class Stop(BaseException):
        def __str__(self) -> str:
            return "alpha fault"

    err = Stop()
    fn = Scripted([err, "never"])
    sleeps = Sleeps()
    with pytest.raises(Stop) as info:
        with_retry(fn, policy((RULE_A,)), sleep=sleeps)
    assert info.value is err
    assert fn.calls == 1
    assert sleeps.delays == []


def test_keyboard_interrupt_propagates_without_retry() -> None:
    fn = Scripted([KeyboardInterrupt("alpha fault"), "never"])
    sleeps = Sleeps()
    with pytest.raises(KeyboardInterrupt):
        with_retry(fn, policy((RULE_A,)), sleep=sleeps)
    assert fn.calls == 1
    assert sleeps.delays == []


def test_earlier_rule_wins_for_runner_crash_ending_in_eof() -> None:
    crash = RetryRule(name="runner-crash", max_attempts=2, markers=("runner crashed",))
    eof = RetryRule(name="eof", max_attempts=5, markers=("eof",))
    p = policy((crash, eof))
    text = "model runner crashed: unexpected EOF"
    assert p.rule_for(text) == crash
    fn = Scripted([RuntimeError(text), RuntimeError(text), "never"])
    sleeps = Sleeps()
    with pytest.raises(RuntimeError):
        with_retry(fn, p, sleep=sleeps)
    assert fn.calls == 2
    assert sleeps.delays == [1.5]


# --- parse_retry_policy -----------------------------------------------------


def valid_raw() -> dict[str, Any]:
    return {
        "retry_rules": [
            {"name": "class-a", "max_attempts": 2, "markers": ["alpha fault", "alpha2"]},
            {"name": "class-b", "max_attempts": 6, "markers": ["beta-fault"]},
        ],
        "retry_base_delay_s": 2,
    }


def test_parse_valid_mapping() -> None:
    p = parse_retry_policy(valid_raw())
    assert isinstance(p, RetryPolicy)
    assert p.rules == (
        RetryRule(name="class-a", max_attempts=2, markers=("alpha fault", "alpha2")),
        RetryRule(name="class-b", max_attempts=6, markers=("beta-fault",)),
    )
    assert all(isinstance(r, RetryRule) for r in p.rules)
    assert all(isinstance(r.markers, tuple) for r in p.rules)
    assert isinstance(p.base_delay_s, float)
    assert p.base_delay_s == 2.0
    assert p.non_retryable == ()


def test_parse_non_retryable_errors() -> None:
    raw = valid_raw()
    raw["non_retryable_errors"] = ["invalid argument", "permission denied"]
    p = parse_retry_policy(raw)
    assert p.non_retryable == ("invalid argument", "permission denied")
    assert isinstance(p.non_retryable, tuple)


def test_parse_zero_base_delay_is_allowed() -> None:
    raw = valid_raw()
    raw["retry_base_delay_s"] = 0
    assert parse_retry_policy(raw).base_delay_s == 0.0


@pytest.mark.parametrize("key", ["retry_rules", "retry_base_delay_s"])
def test_parse_missing_required_key(key: str) -> None:
    raw = valid_raw()
    del raw[key]
    with pytest.raises(ValueError):
        parse_retry_policy(raw)


@pytest.mark.parametrize("bad", [0, True, -1])
def test_parse_bad_max_attempts(bad: Any) -> None:
    raw = valid_raw()
    raw["retry_rules"][1]["max_attempts"] = bad
    with pytest.raises(ValueError):
        parse_retry_policy(raw)


def test_parse_negative_base_delay() -> None:
    raw = valid_raw()
    raw["retry_base_delay_s"] = -0.5
    with pytest.raises(ValueError):
        parse_retry_policy(raw)


# --- call_provider stays one-retry-then-next --------------------------------


class FakeTransport:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.sends = 0

    def send(self, request: Any) -> HttpResponse:
        del request
        self.sends += 1
        if not self.script:
            raise AssertionError("transport called more often than scripted")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


OK_BODY = json.dumps(
    {
        "choices": [{"message": {"content": "hello"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }
).encode()


def _call(transport: FakeTransport) -> Any:
    config = parse_config(
        {
            "providers": [
                {
                    "name": "alpha",
                    "kind": "openai_compatible",
                    "key_env": KEY_ENV,
                    "base_url": "https://alpha.invalid/v1",
                    "models": {"answer": {"name": "alpha-model"}},
                }
            ]
        }
    )
    spec = next(s for s in config.providers if s.name == "alpha")
    return call_provider(
        spec,
        config,
        "answer",
        "prompt",
        system=None,
        json_mode=False,
        temperature=0.0,
        seed=None,
        env={KEY_ENV: FAKE_KEY},
        transport=transport,
        clock=lambda: 0.0,
    )


def test_call_provider_connection_error_twice_moves_on() -> None:
    transport = FakeTransport([TransportError("connection refused")] * 2)
    with pytest.raises(ProviderFailure) as info:
        _call(transport)
    assert transport.sends == 2
    assert info.value.outcome is not None
    assert info.value.outcome.action is Action.RETRY_THEN_NEXT


def test_call_provider_connection_error_then_success() -> None:
    ok = HttpResponse(status=200, headers={}, body=OK_BODY)
    transport = FakeTransport([TransportError("connection refused"), ok])
    generation = _call(transport)
    assert generation.text == "hello"
    assert transport.sends == 2
