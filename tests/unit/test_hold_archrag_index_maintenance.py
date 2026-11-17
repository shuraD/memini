"""GR-08 acceptance: the pure helpers of incremental ArchRAG index maintenance.

Level-0 grouping weight, nearest level-0 community at a cosine threshold, running-mean
centroid, consolidation plan and trigger, level-0 coherence and structural density.
Plain Python; no database is needed.
"""

from __future__ import annotations

import math

import pytest

from memini.core.archrag.index import (
    Plan,
    coherence,
    consolidation_plan,
    density,
    grouping_weight,
    nearest_community,
    running_mean,
    should_consolidate,
)

# --- A44: level-0 grouping on content, structure or a blend ---


def test_grouping_weight_content_mode_uses_content() -> None:
    assert grouping_weight(0.8, 0.2, mode="content") == pytest.approx(0.8)


def test_grouping_weight_structure_mode_uses_structure() -> None:
    assert grouping_weight(0.8, 0.2, mode="structure") == pytest.approx(0.2)


def test_grouping_weight_blend_default_is_even() -> None:
    assert grouping_weight(0.8, 0.2, mode="blend") == pytest.approx(0.5)


def test_grouping_weight_blend_custom_share() -> None:
    assert grouping_weight(1.0, 0.0, mode="blend", blend=0.25) == pytest.approx(0.25)


def test_grouping_weight_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        grouping_weight(0.5, 0.5, mode="other")


# --- A46: attach to the nearest level-0 community at a cosine threshold ---


def test_nearest_community_picks_highest_cosine() -> None:
    centroids = {"c1": [1.0, 0.0], "c2": [0.0, 1.0], "c3": [1.0, 1.0]}
    assert nearest_community([0.1, 1.0], centroids) == "c2"


def test_nearest_community_ignores_magnitude() -> None:
    centroids = {"a": [10.0, 0.0], "b": [0.0, 0.5]}
    assert nearest_community([3.0, 0.1], centroids) == "a"


def test_nearest_community_below_default_threshold_is_none() -> None:
    # cosine([1, 1], [1, 0]) = 0.707 >= 0.60; cosine([1, 2], [1, 0]) = 0.447 < 0.60
    centroids = {"c": [1.0, 0.0]}
    assert nearest_community([1.0, 1.0], centroids) == "c"
    assert nearest_community([1.0, 2.0], centroids) is None


def test_nearest_community_custom_threshold() -> None:
    centroids = {"c": [1.0, 0.0]}
    assert nearest_community([1.0, 1.0], centroids, threshold=0.8) is None
    assert nearest_community([1.0, 2.0], centroids, threshold=0.4) == "c"


def test_nearest_community_threshold_is_inclusive() -> None:
    centroids = {"c": [1.0, 0.0]}
    assert nearest_community([1.0, 0.0], centroids, threshold=1.0) == "c"


def test_nearest_community_tie_goes_to_smallest_id() -> None:
    centroids = {"zeta": [1.0, 0.0], "alpha": [2.0, 0.0], "mid": [0.0, 1.0]}
    assert nearest_community([1.0, 0.0], centroids) == "alpha"


def test_nearest_community_empty_is_none() -> None:
    assert nearest_community([1.0, 0.0], {}) is None


def test_nearest_community_zero_vector_is_none() -> None:
    assert nearest_community([0.0, 0.0], {"c": [1.0, 0.0]}) is None


# --- A46: running-mean centroid ---


def test_running_mean_adds_one_member() -> None:
    centroid = [1.0, 0.0]
    vector = [0.0, 4.0]
    result = running_mean(centroid, 3, vector)
    assert result == pytest.approx([0.75, 1.0])
    assert centroid == [1.0, 0.0]
    assert vector == [0.0, 4.0]


def test_running_mean_of_empty_community_is_the_vector() -> None:
    vector = [0.5, -2.0, 3.0]
    result = running_mean([0.0, 0.0, 0.0], 0, vector)
    assert result == pytest.approx(vector)
    assert isinstance(result, list)
    assert result is not vector


# --- A48 to A50: consolidation plan ---


def test_plan_values() -> None:
    assert {p.value for p in Plan} == {"rebuild", "place", "reuse"}


def test_plan_rebuild_without_hierarchy() -> None:
    assert consolidation_plan(has_hierarchy=False, drift=0.0, unplaced=0) is Plan.REBUILD


def test_plan_rebuild_on_drift_above_default() -> None:
    assert consolidation_plan(has_hierarchy=True, drift=0.76, unplaced=0) is Plan.REBUILD
    assert consolidation_plan(has_hierarchy=True, drift=0.9, unplaced=5) is Plan.REBUILD


def test_plan_place_at_drift_threshold_with_unplaced() -> None:
    assert consolidation_plan(has_hierarchy=True, drift=0.75, unplaced=2) is Plan.PLACE


def test_plan_reuse_when_nothing_to_do() -> None:
    assert consolidation_plan(has_hierarchy=True, drift=0.1, unplaced=0) is Plan.REUSE


def test_plan_custom_drift_threshold() -> None:
    assert (
        consolidation_plan(has_hierarchy=True, drift=0.5, unplaced=0, drift_threshold=0.4)
        is Plan.REBUILD
    )
    assert (
        consolidation_plan(has_hierarchy=True, drift=0.8, unplaced=0, drift_threshold=0.9)
        is Plan.REUSE
    )


# --- A51, A53: consolidation trigger on time, write count or dirty share ---

LIMITS = {"max_seconds": 600.0, "max_writes": 100, "max_dirty_share": 0.2}


def test_trigger_all_below_is_false() -> None:
    assert (
        should_consolidate(
            seconds_since_last=599.0, writes_since_last=99, dirty_share=0.19, **LIMITS
        )
        is False
    )


@pytest.mark.parametrize(
    ("seconds", "writes", "share"),
    [(600.0, 0, 0.0), (0.0, 100, 0.0), (0.0, 0, 0.2), (5000.0, 500, 0.9)],
)
def test_trigger_any_threshold_reached_is_true(seconds: float, writes: int, share: float) -> None:
    assert (
        should_consolidate(
            seconds_since_last=seconds, writes_since_last=writes, dirty_share=share, **LIMITS
        )
        is True
    )


# --- A52, T79: level-0 coherence and structural density ---


def test_coherence_mean_pairwise_cosine() -> None:
    # pairs: (a, a) = 1, (a, b) = 0, (a, b) = 0 -> 1/3
    assert coherence([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]) == pytest.approx(1 / 3)


def test_coherence_ignores_magnitude() -> None:
    assert coherence([[2.0, 0.0], [1.0, 1.0]]) == pytest.approx(1 / math.sqrt(2))


def test_coherence_single_and_empty() -> None:
    assert coherence([[0.3, 0.4]]) == pytest.approx(1.0)
    assert coherence([]) == pytest.approx(0.0)


def test_density_ratio_of_possible_pairs() -> None:
    assert density(4, 3) == pytest.approx(0.5)
    assert density(2, 1) == pytest.approx(1.0)


def test_density_fewer_than_two_members() -> None:
    assert density(1, 0) == pytest.approx(0.0)
    assert density(0, 0) == pytest.approx(0.0)
