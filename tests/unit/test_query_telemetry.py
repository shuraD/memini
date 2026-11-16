"""Acceptance tests for QY-02: per-stage timings, per-answer usage and per-call limits."""

from __future__ import annotations

from typing import Any

import pytest

from memini.core.query.telemetry import (
    TIMEOUT_KEY,
    AnswerUsage,
    CallLimits,
    StageTimer,
    answer_usage,
    call_limits,
)
from memini.infra.llm_providers.calllog import CallRecord
from memini.infra.llm_providers.config import ConfigError, LLMConfig, parse_config


class FakeClock:
    """A clock whose time is set by the test."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class StageBoomError(RuntimeError):
    pass


# --------------------------------------------------------------------- StageTimer


def test_stages_are_timed_in_order_of_first_use() -> None:
    clock = FakeClock(1.0)
    timer = StageTimer(clock)
    with timer.stage("embed"):
        clock.now = 1.25
    with timer.stage("search"):
        clock.now = 2.0
    timings = timer.timings()
    assert timings == pytest.approx({"embed": 0.25, "search": 0.75})
    assert list(timings) == ["embed", "search"]


def test_same_stage_twice_is_summed_under_one_key() -> None:
    clock = FakeClock(0.0)
    timer = StageTimer(clock)
    with timer.stage("rank"):
        clock.now = 0.5
    with timer.stage("other"):
        clock.now = 1.0
    with timer.stage("rank"):
        clock.now = 1.25
    timings = timer.timings()
    assert timings == pytest.approx({"rank": 0.75, "other": 0.5})
    assert list(timings) == ["rank", "other"]


def test_raising_stage_propagates_and_is_still_recorded() -> None:
    clock = FakeClock(3.0)
    timer = StageTimer(clock)
    err = StageBoomError("stage failed")
    with pytest.raises(StageBoomError) as info, timer.stage("search"):
        clock.now = 3.5
        raise err
    assert info.value is err
    assert timer.timings() == pytest.approx({"search": 0.5})


def test_add_sums_with_a_timed_stage() -> None:
    clock = FakeClock(0.0)
    timer = StageTimer(clock)
    with timer.stage("hybrid"):
        clock.now = 0.25
    timer.add("hybrid", 0.5)
    assert timer.timings()["hybrid"] == pytest.approx(0.75)


def test_add_alone_creates_the_stage() -> None:
    timer = StageTimer(FakeClock(0.0))
    timer.add("sub", 0.5)
    timer.add("other", 0.0)
    assert timer.timings() == pytest.approx({"sub": 0.5, "other": 0.0})
    assert list(timer.timings()) == ["sub", "other"]


def test_add_negative_seconds_raises_value_error() -> None:
    timer = StageTimer(FakeClock(0.0))
    with pytest.raises(ValueError):
        timer.add("hybrid", -0.25)


def test_no_stage_gives_empty_timings() -> None:
    timer = StageTimer(FakeClock(5.0))
    assert timer.timings() == {}


def test_timings_returns_a_new_dict_each_time() -> None:
    clock = FakeClock(0.0)
    timer = StageTimer(clock)
    with timer.stage("embed"):
        clock.now = 1.0
    first = timer.timings()
    first["embed"] = 99.0
    first["extra"] = 1.0
    second = timer.timings()
    assert second == pytest.approx({"embed": 1.0})
    assert second is not first
    empty_timer = StageTimer(FakeClock(0.0))
    empty = empty_timer.timings()
    empty["x"] = 1.0
    assert empty_timer.timings() == {}


def test_total_is_now_minus_construction_time() -> None:
    clock = FakeClock(10.0)
    timer = StageTimer(clock)
    clock.now = 12.5
    assert timer.total() == pytest.approx(2.5)


def test_constructor_reads_the_clock_once() -> None:
    calls: list[int] = []

    def clock() -> float:
        calls.append(1)
        return 7.0

    StageTimer(clock)
    assert len(calls) == 1


def test_very_long_stage_has_no_time_limit() -> None:
    clock = FakeClock(0.0)
    timer = StageTimer(clock)
    with timer.stage("generate"):
        clock.now = 10000.0
    assert timer.timings() == pytest.approx({"generate": 10000.0})
    assert timer.total() == pytest.approx(10000.0)


# ------------------------------------------------------------------- answer_usage


def row(
    question_id: str | None,
    *,
    kind: str = "generate",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    seconds: float = 0.0,
    cache_hit: bool = False,
    cost: float | None = None,
) -> CallRecord:
    return CallRecord(
        kind=kind,  # type: ignore[arg-type]
        provider="local",
        model="m",
        role="summary",
        stage="answer",
        question_id=question_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        seconds=seconds,
        retries=0,
        wait_seconds=0.0,
        fallbacks=(),
        cache_hit=cache_hit,
        cost=cost,
        texts=None,
        ts=0.0,
    )


def test_only_rows_of_the_question_count() -> None:
    rows = [
        row("q1", prompt_tokens=100, completion_tokens=10, seconds=1.0, cost=0.25),
        row("q2", prompt_tokens=1000, completion_tokens=100, seconds=5.0, cost=2.0),
        row(None, prompt_tokens=7, completion_tokens=7, seconds=7.0, cost=7.0),
        row("q1", cache_hit=True, prompt_tokens=50, completion_tokens=5, seconds=0.5, cost=1.0),
        row("q1", kind="embed", prompt_tokens=20, completion_tokens=0, seconds=0.25),
        row(None, kind="embed", prompt_tokens=3, seconds=3.0),
    ]
    usage = answer_usage(rows, "q1")
    assert usage.calls == 3
    assert usage.cache_hits == 1
    assert usage.prompt_tokens == 120
    assert usage.completion_tokens == 10
    assert usage.cost == pytest.approx(0.25)
    assert usage.seconds == pytest.approx(1.75)


def test_answer_usage_accepts_any_iterable() -> None:
    rows = [row("q1", prompt_tokens=4, completion_tokens=2, seconds=0.5, cost=0.5)]
    usage = answer_usage(iter(rows), "q1")
    assert usage == AnswerUsage(1, 0, 4, 2, 0.5, 0.5)


def test_cost_sums_non_none_costs() -> None:
    rows = [row("q1", cost=0.25), row("q1", cost=None), row("q1", cost=0.5)]
    usage = answer_usage(rows, "q1")
    assert usage.cost == pytest.approx(0.75)
    assert usage.calls == 3


def test_cost_is_none_when_all_costs_are_none() -> None:
    rows = [row("q1", cost=None, prompt_tokens=5), row("q1", cost=None)]
    usage = answer_usage(rows, "q1")
    assert usage.cost is None
    assert usage.prompt_tokens == 5


def test_cost_is_none_when_only_cache_hits_have_cost() -> None:
    rows = [row("q1", cache_hit=True, cost=0.5), row("q1", cost=None)]
    assert answer_usage(rows, "q1").cost is None


def test_cache_hit_counts_in_calls_but_not_tokens_or_cost() -> None:
    rows = [
        row(
            "q1",
            cache_hit=True,
            prompt_tokens=500,
            completion_tokens=50,
            seconds=0.25,
            cost=3.0,
        )
    ]
    usage = answer_usage(rows, "q1")
    assert usage.calls == 1
    assert usage.cache_hits == 1
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.cost is None
    assert usage.seconds == pytest.approx(0.25)


def test_no_matching_rows_gives_zero_usage() -> None:
    rows = [row("q2", prompt_tokens=10, cost=1.0, seconds=1.0), row(None, seconds=2.0)]
    usage = answer_usage(rows, "q1")
    assert usage == AnswerUsage(0, 0, 0, 0, None, 0.0)
    assert answer_usage([], "q1") == AnswerUsage(0, 0, 0, 0, None, 0.0)


def test_answer_usage_is_frozen() -> None:
    usage = answer_usage([], "q1")
    with pytest.raises(AttributeError):
        usage.calls = 5  # type: ignore[misc]


# -------------------------------------------------------------------- call_limits


def config(
    provider_limits: dict[str, Any], model_limits: dict[str, Any] | None = None
) -> LLMConfig:
    model: dict[str, Any] = {"name": "local-summary-model"}
    if model_limits is not None:
        model["limits"] = model_limits
    raw: dict[str, Any] = {
        "providers": [
            {
                "name": "local",
                "kind": "ollama",
                "base_url": "http://localhost:11434",
                "limits": provider_limits,
                "models": {"summary": model},
            }
        ]
    }
    return parse_config(raw)


def test_timeout_key_value() -> None:
    assert TIMEOUT_KEY == "timeout_s"


def test_call_limits_from_provider_limits() -> None:
    cfg = config({"num_ctx": 32768, "max_tokens": 8192, "timeout_s": 600})
    limits = call_limits(cfg, "local", "summary", 3)
    assert limits == CallLimits(24576, 8192, 600.0)
    assert isinstance(limits.timeout_s, float)


def test_float_timeout_is_kept() -> None:
    cfg = config({"num_ctx": 32768, "max_tokens": 8192, "timeout_s": 1.5})
    assert call_limits(cfg, "local", "summary", 3).timeout_s == pytest.approx(1.5)


def test_role_model_timeout_overrides_provider_timeout() -> None:
    cfg = config(
        {"num_ctx": 32768, "max_tokens": 8192, "timeout_s": 600},
        {"timeout_s": 120},
    )
    limits = call_limits(cfg, "local", "summary", 3)
    assert limits.timeout_s == 120.0
    assert limits == CallLimits(24576, 8192, 120.0)


def test_role_model_size_limits_override_provider() -> None:
    cfg = config(
        {"num_ctx": 32768, "max_tokens": 8192, "timeout_s": 600},
        {"num_ctx": 16384, "max_tokens": 4096},
    )
    assert call_limits(cfg, "local", "summary", 3) == CallLimits(12288, 4096, 600.0)


@pytest.mark.parametrize("bad", [0, 0.0, -1, -0.5, True, False, "600", None])
def test_bad_timeout_raises_config_error(bad: object) -> None:
    cfg = config({"num_ctx": 32768, "max_tokens": 8192, "timeout_s": bad})
    with pytest.raises(ConfigError):
        call_limits(cfg, "local", "summary", 3)


def test_missing_timeout_raises_config_error() -> None:
    cfg = config({"num_ctx": 32768, "max_tokens": 8192})
    with pytest.raises(ConfigError):
        call_limits(cfg, "local", "summary", 3)


def test_bad_model_timeout_overrides_good_provider_timeout() -> None:
    cfg = config({"num_ctx": 32768, "max_tokens": 8192, "timeout_s": 600}, {"timeout_s": 0})
    with pytest.raises(ConfigError):
        call_limits(cfg, "local", "summary", 3)


def test_missing_num_ctx_raises_config_error() -> None:
    cfg = config({"max_tokens": 8192, "timeout_s": 600})
    with pytest.raises(ConfigError):
        call_limits(cfg, "local", "summary", 3)


def test_input_budget_errors_propagate() -> None:
    cfg = config({"num_ctx": 8192, "max_tokens": 8192, "timeout_s": 600})
    with pytest.raises(ConfigError):
        call_limits(cfg, "local", "summary", 3)
    good = config({"num_ctx": 32768, "max_tokens": 8192, "timeout_s": 600})
    with pytest.raises(ConfigError):
        call_limits(good, "local", "summary", 0)


def test_call_limits_is_frozen() -> None:
    cfg = config({"num_ctx": 32768, "max_tokens": 8192, "timeout_s": 600})
    limits = call_limits(cfg, "local", "summary", 3)
    with pytest.raises(AttributeError):
        limits.timeout_s = 1.0  # type: ignore[misc]
