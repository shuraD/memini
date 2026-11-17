"""Acceptance tests for GR-08: pure decision helpers of ArchRAG incremental maintenance."""

from __future__ import annotations

import enum
import importlib
import math
import socket

import pytest

index = importlib.import_module("memini.core.archrag.index")


# --- Plan ---------------------------------------------------------------------------------


def test_plan_is_str_enum_with_three_values():
    plan = index.Plan
    assert issubclass(plan, enum.StrEnum)
    assert plan.REBUILD == "rebuild"
    assert plan.PLACE == "place"
    assert plan.REUSE == "reuse"
    assert {p.value for p in plan} == {"rebuild", "place", "reuse"}


# --- grouping_weight ----------------------------------------------------------------------


def test_grouping_weight_content_mode_returns_content():
    assert index.grouping_weight(0.8, 0.2, mode="content") == pytest.approx(0.8)


def test_grouping_weight_structure_mode_returns_structure():
    assert index.grouping_weight(0.8, 0.2, mode="structure") == pytest.approx(0.2)


def test_grouping_weight_blend_default_is_half():
    assert index.grouping_weight(0.8, 0.2, mode="blend") == pytest.approx(0.5)


def test_grouping_weight_blend_custom_weight():
    result = index.grouping_weight(0.8, 0.2, mode="blend", blend=0.25)
    assert result == pytest.approx(0.25 * 0.8 + 0.75 * 0.2)


def test_grouping_weight_blend_extremes():
    assert index.grouping_weight(0.9, 0.1, mode="blend", blend=1.0) == pytest.approx(0.9)
    assert index.grouping_weight(0.9, 0.1, mode="blend", blend=0.0) == pytest.approx(0.1)


@pytest.mark.parametrize("mode", ["", "Content", "mixed", "average"])
def test_grouping_weight_unknown_mode_raises(mode):
    with pytest.raises(ValueError):
        index.grouping_weight(0.5, 0.5, mode=mode)


# --- nearest_community --------------------------------------------------------------------


def test_nearest_community_returns_best_above_threshold():
    centroids = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]}
    assert index.nearest_community([1.0, 0.1], centroids) == "a"
    assert index.nearest_community([0.1, 1.0], centroids) == "b"


def test_nearest_community_best_cosine_at_default_threshold_qualifies():
    # cosine([0.6, 0.8], [1, 0]) == 0.6 exactly, the default threshold.
    assert index.nearest_community([0.6, 0.8], {"x": [1.0, 0.0]}) == "x"


def test_nearest_community_below_default_threshold_is_none():
    # cosine([0.5, 0.866...], [1, 0]) == 0.5 < 0.60
    assert index.nearest_community([0.5, math.sqrt(0.75)], {"x": [1.0, 0.0]}) is None


def test_nearest_community_custom_threshold_honoured():
    centroids = {"x": [1.0, 0.0]}
    vector = [1.0, 1.0]  # cosine ~0.7071
    assert index.nearest_community(vector, centroids, threshold=0.7) == "x"
    assert index.nearest_community(vector, centroids, threshold=0.71) is None


def test_nearest_community_cosine_ignores_magnitude():
    centroids = {"small": [0.001, 0.0], "big": [0.0, 1000.0]}
    assert index.nearest_community([5.0, 0.1], centroids) == "small"


def test_nearest_community_empty_centroids_is_none():
    assert index.nearest_community([1.0, 0.0], {}) is None


def test_nearest_community_tie_goes_to_smallest_id():
    centroids = {"m": [1.0, 0.0], "b": [2.0, 0.0], "z": [3.0, 0.0]}
    assert index.nearest_community([1.0, 0.0], centroids) == "b"


def test_nearest_community_tie_symmetric_vectors_smallest_id():
    centroids = {"k2": [0.0, 1.0], "k1": [1.0, 0.0]}
    assert index.nearest_community([1.0, 1.0], centroids, threshold=0.5) == "k1"


@pytest.mark.parametrize("threshold", [0.6, 0.01, 1e-9])
def test_nearest_community_zero_vector_is_none(threshold):
    centroids = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    assert index.nearest_community([0.0, 0.0], centroids, threshold=threshold) is None


def test_nearest_community_zero_centroid_never_matches():
    centroids = {"a": [0.0, 0.0]}
    assert index.nearest_community([1.0, 0.0], centroids, threshold=0.01) is None


# --- running_mean -------------------------------------------------------------------------


def test_running_mean_spec_example():
    centroid = [1.0, 0.0]
    vector = [0.0, 4.0]
    result = index.running_mean(centroid, 3, vector)
    assert result == pytest.approx([0.75, 1.0])
    assert isinstance(result, list)


def test_running_mean_leaves_inputs_unchanged():
    centroid = [1.0, 0.0]
    vector = [0.0, 4.0]
    index.running_mean(centroid, 3, vector)
    assert centroid == [1.0, 0.0]
    assert vector == [0.0, 4.0]


def test_running_mean_size_zero_returns_vector_as_new_list():
    centroid = [9.0, 9.0]
    vector = [0.5, -2.0]
    result = index.running_mean(centroid, 0, vector)
    assert result == pytest.approx([0.5, -2.0])
    assert isinstance(result, list)
    assert result is not vector
    result.append(1.0)
    assert vector == [0.5, -2.0]


def test_running_mean_accepts_tuples():
    result = index.running_mean((2.0, 4.0), 1, (4.0, 0.0))
    assert result == pytest.approx([3.0, 2.0])
    assert isinstance(result, list)


def test_running_mean_sequential_equals_overall_mean():
    vectors = [[1.0, 2.0, 3.0], [4.0, 0.0, -1.0], [2.0, 2.0, 2.0], [0.0, -3.0, 5.0]]
    centroid: list[float] = []
    for size, vec in enumerate(vectors):
        centroid = index.running_mean(centroid, size, vec)
    expected = [sum(col) / len(vectors) for col in zip(*vectors, strict=True)]
    assert centroid == pytest.approx(expected)


# --- consolidation_plan -------------------------------------------------------------------


def test_consolidation_plan_no_hierarchy_rebuilds():
    plan = index.consolidation_plan(has_hierarchy=False, drift=0.0, unplaced=0)
    assert plan is index.Plan.REBUILD


def test_consolidation_plan_no_hierarchy_rebuilds_even_with_unplaced():
    plan = index.consolidation_plan(has_hierarchy=False, drift=0.1, unplaced=5)
    assert plan is index.Plan.REBUILD


def test_consolidation_plan_drift_above_default_rebuilds():
    plan = index.consolidation_plan(has_hierarchy=True, drift=0.76, unplaced=0)
    assert plan is index.Plan.REBUILD


def test_consolidation_plan_drift_above_default_rebuilds_with_unplaced():
    plan = index.consolidation_plan(has_hierarchy=True, drift=0.9, unplaced=3)
    assert plan is index.Plan.REBUILD


def test_consolidation_plan_drift_exactly_default_with_unplaced_places():
    plan = index.consolidation_plan(has_hierarchy=True, drift=0.75, unplaced=2)
    assert plan is index.Plan.PLACE


def test_consolidation_plan_drift_exactly_default_no_unplaced_reuses():
    plan = index.consolidation_plan(has_hierarchy=True, drift=0.75, unplaced=0)
    assert plan is index.Plan.REUSE


def test_consolidation_plan_low_drift_unplaced_places():
    plan = index.consolidation_plan(has_hierarchy=True, drift=0.1, unplaced=1)
    assert plan is index.Plan.PLACE


def test_consolidation_plan_low_drift_nothing_unplaced_reuses():
    plan = index.consolidation_plan(has_hierarchy=True, drift=0.1, unplaced=0)
    assert plan is index.Plan.REUSE


def test_consolidation_plan_custom_drift_threshold_honoured():
    assert (
        index.consolidation_plan(has_hierarchy=True, drift=0.5, unplaced=0, drift_threshold=0.4)
        is index.Plan.REBUILD
    )
    assert (
        index.consolidation_plan(has_hierarchy=True, drift=0.8, unplaced=0, drift_threshold=0.9)
        is index.Plan.REUSE
    )
    assert (
        index.consolidation_plan(has_hierarchy=True, drift=0.4, unplaced=1, drift_threshold=0.4)
        is index.Plan.PLACE
    )


# --- should_consolidate -------------------------------------------------------------------

_LIMITS = {"max_seconds": 600.0, "max_writes": 100, "max_dirty_share": 0.2}


def test_should_consolidate_all_below_is_false():
    assert (
        index.should_consolidate(
            seconds_since_last=599.0, writes_since_last=99, dirty_share=0.19, **_LIMITS
        )
        is False
    )


@pytest.mark.parametrize("seconds", [600.0, 1200.0])
def test_should_consolidate_seconds_at_or_above(seconds):
    assert (
        index.should_consolidate(
            seconds_since_last=seconds, writes_since_last=0, dirty_share=0.0, **_LIMITS
        )
        is True
    )


@pytest.mark.parametrize("writes", [100, 101])
def test_should_consolidate_writes_at_or_above(writes):
    assert (
        index.should_consolidate(
            seconds_since_last=0.0, writes_since_last=writes, dirty_share=0.0, **_LIMITS
        )
        is True
    )


@pytest.mark.parametrize("share", [0.2, 0.5])
def test_should_consolidate_dirty_share_at_or_above(share):
    assert (
        index.should_consolidate(
            seconds_since_last=0.0, writes_since_last=0, dirty_share=share, **_LIMITS
        )
        is True
    )


def test_should_consolidate_all_above_is_true():
    assert (
        index.should_consolidate(
            seconds_since_last=1e6, writes_since_last=10**6, dirty_share=1.0, **_LIMITS
        )
        is True
    )


# --- coherence ----------------------------------------------------------------------------


def test_coherence_spec_example_is_one_third():
    assert index.coherence([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]) == pytest.approx(1 / 3)


def test_coherence_one_vector_is_one():
    assert index.coherence([[0.3, 0.4]]) == pytest.approx(1.0)


def test_coherence_none_is_zero():
    assert index.coherence([]) == pytest.approx(0.0)


def test_coherence_two_identical_is_one():
    assert index.coherence([[1.0, 2.0], [2.0, 4.0]]) == pytest.approx(1.0)


def test_coherence_two_orthogonal_is_zero():
    assert index.coherence([[1.0, 0.0], [0.0, 3.0]]) == pytest.approx(0.0)


def test_coherence_zero_vector_counts_as_zero_cosine():
    # pairs: (a,b)=1, (a,0)=0, (b,0)=0 -> mean 1/3
    assert index.coherence([[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]]) == pytest.approx(1 / 3)


def test_coherence_opposite_vectors_negative():
    assert index.coherence([[1.0, 0.0], [-1.0, 0.0]]) == pytest.approx(-1.0)


# --- density ------------------------------------------------------------------------------


def test_density_spec_example():
    assert index.density(4, 3) == pytest.approx(0.5)


def test_density_complete_graph_is_one():
    assert index.density(5, 10) == pytest.approx(1.0)


def test_density_two_members_one_edge():
    assert index.density(2, 1) == pytest.approx(1.0)


@pytest.mark.parametrize("members", [0, 1])
def test_density_fewer_than_two_members_is_zero(members):
    assert index.density(members, 0) == pytest.approx(0.0)
    assert index.density(members, 3) == pytest.approx(0.0)


# --- purity -------------------------------------------------------------------------------


def test_helpers_make_no_network_call(monkeypatch):
    calls: list[object] = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    index.grouping_weight(0.4, 0.6, mode="blend", blend=0.3)
    index.nearest_community([1.0, 0.0], {"a": [1.0, 0.0]})
    index.running_mean([1.0, 0.0], 3, [0.0, 4.0])
    index.consolidation_plan(has_hierarchy=True, drift=0.2, unplaced=1)
    index.should_consolidate(
        seconds_since_last=1.0,
        writes_since_last=1,
        dirty_share=0.1,
        max_seconds=10.0,
        max_writes=10,
        max_dirty_share=0.5,
    )
    index.coherence([[1.0, 0.0], [0.0, 1.0]])
    index.density(3, 2)

    assert calls == []
