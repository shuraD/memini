"""Acceptance tests for LP-09: shared provider health (exhausted / resting marks).

The interface fixes ``ProviderHealth(store, clock)``; the exhausted mark is scoped to
the run (``run_id`` from config, default ``DEFAULT_RUN_ID``). The tests pass a
different run with the keyword argument ``run_id``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memini.infra.llm_providers.health import ProviderHealth
from memini.infra.llm_providers.store import Store

FAR_FUTURE = 1.0e12
"""A clock value far beyond any resting mark used here."""


class _Clock:
    """A settable clock; the health object must use it instead of the real time."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "shared" / "llm.sqlite"


@pytest.fixture
def store(db_path: Path) -> Store:
    return Store(db_path)


def test_unmarked_provider_is_available_without_reason(store: Store) -> None:
    health = ProviderHealth(store, _Clock(10.0))
    assert health.available("alpha") is True
    assert health.unavailable_reason("alpha") is None


def test_resting_until_100_is_unavailable_at_99_and_available_at_100(store: Store) -> None:
    clock = _Clock(0.0)
    health = ProviderHealth(store, clock)
    health.mark_resting("alpha", 100.0, "rate limited")
    clock.now = 99.0
    assert health.available("alpha") is False
    clock.now = 100.0
    assert health.available("alpha") is True
    assert health.unavailable_reason("alpha") is None


def test_resting_mark_only_affects_its_provider(store: Store) -> None:
    clock = _Clock(0.0)
    health = ProviderHealth(store, clock)
    health.mark_resting("alpha", 100.0, "rate limited")
    clock.now = 50.0
    assert health.available("alpha") is False
    assert health.available("beta") is True


def test_exhausted_mark_seen_by_second_health_on_same_store_at_any_time(store: Store) -> None:
    first = ProviderHealth(store, _Clock(5.0))
    first.mark_exhausted("alpha", "quota exhausted")
    other_clock = _Clock(0.0)
    second = ProviderHealth(store, other_clock)
    for now in (0.0, 5.0, 1000.0, FAR_FUTURE):
        other_clock.now = now
        assert second.available("alpha") is False
        assert second.unavailable_reason("alpha") == "quota exhausted"
    assert second.available("beta") is True


def test_mark_by_one_store_handle_is_seen_by_another_on_same_file(db_path: Path) -> None:
    store_a = Store(db_path)
    store_b = Store(db_path)
    clock_a = _Clock(0.0)
    clock_b = _Clock(0.0)
    health_a = ProviderHealth(store_a, clock_a)
    health_b = ProviderHealth(store_b, clock_b)

    health_a.mark_exhausted("alpha", "out of credit")
    health_a.mark_resting("beta", 100.0, "server busy")

    clock_b.now = 50.0
    assert health_b.available("alpha") is False
    assert health_b.unavailable_reason("alpha") == "out of credit"
    assert health_b.available("beta") is False
    assert health_b.unavailable_reason("beta") == "server busy"
    clock_b.now = 100.0
    assert health_b.available("beta") is True


def test_mark_by_second_handle_is_seen_by_first(db_path: Path) -> None:
    store_a = Store(db_path)
    health_a = ProviderHealth(store_a, _Clock(1.0))
    assert health_a.available("alpha") is True
    store_b = Store(db_path)
    ProviderHealth(store_b, _Clock(1.0)).mark_resting("alpha", 10.0, "cooling down")
    assert health_a.available("alpha") is False
    assert health_a.unavailable_reason("alpha") == "cooling down"


def test_exhausted_after_resting_wins(store: Store) -> None:
    clock = _Clock(0.0)
    health = ProviderHealth(store, clock)
    health.mark_resting("alpha", 100.0, "rate limited")
    health.mark_exhausted("alpha", "quota exhausted")
    clock.now = 150.0
    assert health.available("alpha") is False
    assert health.unavailable_reason("alpha") == "quota exhausted"
    clock.now = FAR_FUTURE
    assert health.available("alpha") is False


def test_resting_after_exhausted_does_not_lift_exhausted(store: Store) -> None:
    clock = _Clock(0.0)
    health = ProviderHealth(store, clock)
    health.mark_exhausted("alpha", "quota exhausted")
    health.mark_resting("alpha", 10.0, "rate limited")
    clock.now = 20.0
    assert health.available("alpha") is False
    assert health.unavailable_reason("alpha") == "quota exhausted"


def test_shorter_resting_mark_never_shortens_longer_one(store: Store) -> None:
    clock = _Clock(0.0)
    health = ProviderHealth(store, clock)
    health.mark_resting("alpha", 200.0, "long rest")
    health.mark_resting("alpha", 50.0, "short rest")
    clock.now = 150.0
    assert health.available("alpha") is False
    clock.now = 199.0
    assert health.available("alpha") is False
    clock.now = 200.0
    assert health.available("alpha") is True


def test_shorter_resting_from_another_handle_never_shortens(db_path: Path) -> None:
    clock = _Clock(0.0)
    health_a = ProviderHealth(Store(db_path), clock)
    health_b = ProviderHealth(Store(db_path), clock)
    health_a.mark_resting("alpha", 300.0, "long rest")
    health_b.mark_resting("alpha", 30.0, "short rest")
    clock.now = 100.0
    assert health_a.available("alpha") is False
    assert health_b.available("alpha") is False


def test_longer_resting_mark_extends_rest(store: Store) -> None:
    clock = _Clock(0.0)
    health = ProviderHealth(store, clock)
    health.mark_resting("alpha", 100.0, "first rest")
    health.mark_resting("alpha", 200.0, "second rest")
    clock.now = 150.0
    assert health.available("alpha") is False
    assert health.unavailable_reason("alpha") == "second rest"
    clock.now = 200.0
    assert health.available("alpha") is True


def test_unavailable_reason_returns_recorded_text(store: Store) -> None:
    clock = _Clock(0.0)
    health = ProviderHealth(store, clock)
    reason = "HTTP 429: too many requests — retry later"
    health.mark_resting("alpha", 60.0, reason)
    health.mark_exhausted("beta", "insufficient_quota (402)")
    clock.now = 30.0
    assert health.unavailable_reason("alpha") == reason
    assert health.unavailable_reason("beta") == "insufficient_quota (402)"
    assert health.unavailable_reason("gamma") is None


def test_exhausted_mark_of_earlier_run_does_not_apply_to_another_run(db_path: Path) -> None:
    earlier = ProviderHealth(Store(db_path), _Clock(0.0), run_id="run-1")
    earlier.mark_exhausted("alpha", "quota exhausted")
    assert earlier.available("alpha") is False

    later = ProviderHealth(Store(db_path), _Clock(1.0), run_id="run-2")
    assert later.available("alpha") is True
    assert later.unavailable_reason("alpha") is None

    same_run = ProviderHealth(Store(db_path), _Clock(1.0), run_id="run-1")
    assert same_run.available("alpha") is False
    assert same_run.unavailable_reason("alpha") == "quota exhausted"
