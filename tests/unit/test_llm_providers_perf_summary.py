"""Acceptance tests for LP-19: the per-provider and per-role performance summary.

Fixtures are chosen so that each expectation holds under every reasonable reading of
the specification: cache hits carry no tokens and no seconds that could move a
percentile, every real call has the same tokens/second ratio, and each row with a
fallback chain has exactly one ``(provider, reason)`` pair.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from memini.infra.llm_providers.calllog import CallRecord
from memini.infra.llm_providers.config import LLMConfig, parse_config
from memini.infra.llm_providers.perf_summary import GroupStats, RunSummary, compute_summary


def _config(roles: dict[str, Any] | None = None) -> LLMConfig:
    raw: dict[str, Any] = {
        "run_id": "run-1",
        "providers": [
            {
                "name": "p1",
                "kind": "openai_compatible",
                "base_url": "http://127.0.0.1:1/v1",
                "models": {
                    "answer": {"name": "m-answer"},
                    "judge": {"name": "m-judge"},
                    "embedding": {"name": "m-embed"},
                },
            },
            {
                "name": "p2",
                "kind": "openai_compatible",
                "base_url": "http://127.0.0.1:2/v1",
                "models": {"answer": {"name": "m2-answer"}, "judge": {"name": "m2-judge"}},
            },
        ],
    }
    if roles is not None:
        raw["roles"] = roles
    return parse_config(raw)


def _row(
    *,
    provider: str | None = "p1",
    role: str | None = "answer",
    kind: str = "generate",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    seconds: float = 1.0,
    wait_seconds: float = 0.0,
    fallbacks: tuple[tuple[str, str], ...] = (),
    cache_hit: bool = False,
    cost: float | None = None,
    texts: int | None = None,
) -> CallRecord:
    return CallRecord(
        kind=kind,  # type: ignore[arg-type]
        provider=provider,
        model="m",
        role=role,
        stage=None,
        question_id=None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        seconds=seconds,
        retries=0,
        wait_seconds=wait_seconds,
        fallbacks=fallbacks,
        cache_hit=cache_hit,
        cost=cost,
        texts=texts,
        ts=0.0,
    )


def _real(seconds: float, **kw: Any) -> CallRecord:
    """A real call whose throughput is exactly 50 tokens per second."""
    return _row(seconds=seconds, completion_tokens=round(seconds * 50), **kw)


def _hit(**kw: Any) -> CallRecord:
    return _row(seconds=0.0, cache_hit=True, **kw)


# Ten rows of one provider and one role: eight real calls and two cache hits.
# Real seconds sorted: 1, 2, 3, 3, 5, 6, 7, 10 ; cache hits take 0 seconds.
_TEN_ROWS: list[CallRecord] = [
    _real(5.0),
    _hit(),
    _real(1.0),
    _real(3.0),
    _real(10.0),
    _real(2.0),
    _hit(),
    _real(7.0),
    _real(3.0),
    _real(6.0),
]


def test_summary_types_are_frozen_dataclasses() -> None:
    summary = compute_summary(_TEN_ROWS, _config(), {})
    assert isinstance(summary, RunSummary)
    assert dataclasses.is_dataclass(summary)
    stats = summary.by_provider["p1"]
    assert isinstance(stats, GroupStats)
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.calls = 99  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.by_role = {}  # type: ignore[misc]


@pytest.mark.parametrize("axis", ["by_provider", "by_role"])
def test_ten_row_log_matches_hand_computed_values(axis: str) -> None:
    summary = compute_summary(_TEN_ROWS, _config(), {})
    groups = getattr(summary, axis)
    key = "p1" if axis == "by_provider" else "answer"
    assert set(groups) == {key}
    stats = groups[key]
    assert stats.calls == 10
    assert stats.cache_hit_pct == pytest.approx(20.0)
    assert stats.p50_s == pytest.approx(3.0)
    assert stats.p95_s == pytest.approx(10.0)
    assert stats.max_s == pytest.approx(10.0)
    assert stats.tokens_per_s == pytest.approx(50.0)
    assert dict(stats.failures_by_reason) == {}
    assert stats.fallbacks == 0
    assert stats.wait_s == pytest.approx(0.0)
    assert stats.cost == pytest.approx(0.0)


def test_percentiles_use_nearest_rank_not_interpolation() -> None:
    # Twenty distinct values 1..20 in shuffled order, all real calls.
    order = [7, 19, 3, 12, 1, 20, 15, 9, 4, 17, 11, 2, 14, 6, 18, 10, 5, 13, 16, 8]
    rows = [_real(float(s)) for s in order]
    stats = compute_summary(rows, _config(), {}).by_role["answer"]
    assert stats.calls == 20
    assert stats.p50_s == pytest.approx(10.0)  # rank ceil(0.50 * 20) = 10
    assert stats.p95_s == pytest.approx(19.0)  # rank ceil(0.95 * 20) = 19
    assert stats.max_s == pytest.approx(20.0)
    assert stats.cache_hit_pct == pytest.approx(0.0)


def test_rows_split_by_provider_and_by_role() -> None:
    rows = [
        _real(1.0, provider="p1", role="answer"),
        _real(2.0, provider="p1", role="judge"),
        _real(4.0, provider="p2", role="answer"),
        _real(8.0, provider="p2", role="answer"),
        _hit(provider="p2", role="judge"),
    ]
    summary = compute_summary(rows, _config(), {})
    assert set(summary.by_provider) == {"p1", "p2"}
    assert set(summary.by_role) == {"answer", "judge"}
    assert summary.by_provider["p1"].calls == 2
    assert summary.by_provider["p2"].calls == 3
    assert summary.by_role["answer"].calls == 3
    assert summary.by_role["judge"].calls == 2
    assert summary.by_provider["p1"].max_s == pytest.approx(2.0)
    assert summary.by_provider["p2"].max_s == pytest.approx(8.0)
    assert summary.by_role["answer"].max_s == pytest.approx(8.0)
    assert summary.by_role["answer"].p50_s == pytest.approx(4.0)
    assert summary.by_role["judge"].cache_hit_pct == pytest.approx(50.0)
    assert summary.by_provider["p2"].cache_hit_pct == pytest.approx(100.0 / 3.0)


def test_cost_sums_per_group_and_none_counts_as_zero() -> None:
    rows = [
        _real(1.0, provider="p1", role="answer", cost=0.25),
        _real(1.0, provider="p1", role="judge", cost=None),
        _real(1.0, provider="p1", role="answer", cost=0.5),
        _real(1.0, provider="p2", role="judge", cost=1.0),
        _real(1.0, provider="p2", role="answer", cost=None),
    ]
    summary = compute_summary(rows, _config(), {})
    assert summary.by_provider["p1"].cost == pytest.approx(0.75)
    assert summary.by_provider["p2"].cost == pytest.approx(1.0)
    assert summary.by_role["answer"].cost == pytest.approx(0.75)
    assert summary.by_role["judge"].cost == pytest.approx(1.0)
    assert isinstance(summary.by_role["judge"].cost, float)


def test_group_with_only_none_costs_has_zero_cost() -> None:
    rows = [_real(1.0, cost=None), _hit(cost=None)]
    stats = compute_summary(rows, _config(), {}).by_provider["p1"]
    assert stats.cost == pytest.approx(0.0)


def test_wait_seconds_sum_per_group() -> None:
    rows = [
        _real(1.0, role="answer", wait_seconds=0.5),
        _real(1.0, role="answer", wait_seconds=1.25),
        _real(1.0, role="judge", wait_seconds=2.0),
    ]
    summary = compute_summary(rows, _config(), {})
    assert summary.by_role["answer"].wait_s == pytest.approx(1.75)
    assert summary.by_role["judge"].wait_s == pytest.approx(2.0)
    assert summary.by_provider["p1"].wait_s == pytest.approx(3.75)


def test_failures_by_reason_counts_each_reason() -> None:
    rows = [
        _real(1.0, provider="p2", role="answer", fallbacks=(("p1", "rate_per_minute"),)),
        _real(1.0, provider="p2", role="answer", fallbacks=(("p1", "rate_per_minute"),)),
        _real(1.0, provider="p2", role="answer", fallbacks=(("p1", "timeout"),)),
        _real(1.0, provider="p2", role="answer", fallbacks=(("p1", "exhausted"),)),
        _real(1.0, provider="p1", role="answer"),
        _real(1.0, provider="p2", role="judge", fallbacks=(("p1", "server_error"),)),
        _real(1.0, provider="p1", role="judge"),
    ]
    summary = compute_summary(rows, _config(), {})
    answer = summary.by_role["answer"]
    assert dict(answer.failures_by_reason) == {
        "rate_per_minute": 2,
        "timeout": 1,
        "exhausted": 1,
    }
    assert answer.fallbacks == 4
    judge = summary.by_role["judge"]
    assert dict(judge.failures_by_reason) == {"server_error": 1}
    assert judge.fallbacks == 1


def test_budget_used_shows_used_tokens_against_configured_limit() -> None:
    config = _config(
        roles={
            "answer": {"max_tokens_per_run": 1000},
            "judge": {"max_calls_per_question": 3},
        }
    )
    rows = [_real(1.0, role="answer"), _real(1.0, role="judge")]
    summary = compute_summary(rows, config, {"answer": 420, "judge": 77})
    assert summary.budget_used["answer"] == (420, 1000)
    assert summary.budget_used["judge"] == (77, None)


def test_budget_used_role_without_section_is_unlimited() -> None:
    config = _config(roles={"answer": {"max_tokens_per_run": 500}})
    summary = compute_summary([], config, {"answer": 0, "filter": 12})
    assert summary.budget_used["answer"] == (0, 500)
    assert summary.budget_used["filter"] == (12, None)


def test_empty_log_gives_empty_groups_without_division_error() -> None:
    summary = compute_summary([], _config(), {})
    assert dict(summary.by_provider) == {}
    assert dict(summary.by_role) == {}
    assert dict(summary.budget_used) == {}


def test_only_cache_hits_do_not_divide_by_zero() -> None:
    rows = [_hit(), _hit()]
    stats = compute_summary(rows, _config(), {}).by_role["answer"]
    assert stats.calls == 2
    assert stats.cache_hit_pct == pytest.approx(100.0)
    assert stats.tokens_per_s == pytest.approx(0.0)


def test_embed_rows_counted_separately_and_do_not_distort_generation() -> None:
    config = _config()
    base = compute_summary(_TEN_ROWS, config, {})
    embeds = [
        _row(
            kind="embed",
            provider="p1",
            role=None,
            prompt_tokens=5000,
            seconds=s,
            texts=64,
        )
        for s in (40.0, 60.0, 90.0)
    ]
    mixed = compute_summary([*embeds[:1], *_TEN_ROWS, *embeds[1:]], config, {})

    assert mixed.by_role["answer"] == base.by_role["answer"]
    p1 = mixed.by_provider["p1"]
    ref = base.by_provider["p1"]
    assert p1.p50_s == pytest.approx(ref.p50_s)
    assert p1.p95_s == pytest.approx(ref.p95_s)
    assert p1.max_s == pytest.approx(ref.max_s)
    assert p1.tokens_per_s == pytest.approx(ref.tokens_per_s)
    assert p1.cache_hit_pct == pytest.approx(ref.cache_hit_pct)

    groups = [*mixed.by_provider.values(), *mixed.by_role.values()]
    assert any(g.calls == 3 for g in groups)
