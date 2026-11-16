"""Acceptance tests for AR-15: client-side community and entity recall scoring."""

from __future__ import annotations

import dataclasses
import inspect
import math

import pytest

from memini.core.archrag.recall import (
    CommunityHit,
    EntityHit,
    cosine,
    oversample,
    recall_communities,
    search_entities,
)


def _community(
    cid: str,
    embedding: list[float] | None,
    *,
    scope: str = "s1",
    level: int = 0,
    summary: str | None = None,
    size: int = 2,
    members: list[str] | None = None,
) -> dict:
    return {
        "id": cid,
        "scope": scope,
        "level": level,
        "summary": summary if summary is not None else f"summary of {cid}",
        "size": size,
        "members": members if members is not None else [f"{cid}-m1", f"{cid}-m2"],
        "embedding": embedding,
    }


def _entity(eid: str, embedding, *, scope: str = "s1", domain=None, omit_domain=False) -> dict:
    row = {"id": eid, "scope": scope, "embedding": embedding}
    if not omit_domain:
        row["domain"] = domain
    return row


# ---------------------------------------------------------------- cosine


def test_cosine_equal_vectors_is_one():
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector_is_zero():
    assert cosine([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0
    assert cosine([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == 0.0


def test_cosine_known_value():
    expected = (1 * 2 + 1 * 0) / (math.sqrt(2) * 2)
    assert cosine([1.0, 1.0], [2.0, 0.0]) == pytest.approx(expected)


def test_cosine_different_lengths_raises():
    with pytest.raises(ValueError):
        cosine([1.0, 2.0], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------- oversample


@pytest.mark.parametrize(("k", "expected"), [(1, 100), (5, 100), (10, 200)])
def test_oversample_values(k, expected):
    assert oversample(k) == expected


def test_oversample_rejects_zero():
    with pytest.raises(ValueError):
        oversample(0)


# ---------------------------------------------------------------- recall_communities


def test_recall_communities_orders_scores_and_copies_fields():
    query = [1.0, 0.0]
    rows = [
        _community("c-low", [0.0, 1.0], level=1, size=3, members=["a", "b", "c"]),
        _community("c-high", [1.0, 0.0], level=2, summary="precomputed high", size=5),
        _community("c-mid", [1.0, 1.0], level=0),
    ]
    hits = recall_communities(query, rows, k=10, scope="s1")
    assert [h.id for h in hits] == ["c-high", "c-mid", "c-low"]
    assert all(isinstance(h, CommunityHit) for h in hits)
    for hit in hits:
        row = next(r for r in rows if r["id"] == hit.id)
        assert hit.score == pytest.approx(cosine(query, row["embedding"]))
        assert hit.level == row["level"]
        assert hit.summary == row["summary"]
        assert hit.size == row["size"]
        assert hit.members == tuple(row["members"])
        assert isinstance(hit.members, tuple)
    assert hits[0].summary == "precomputed high"


def test_recall_communities_returns_at_most_k():
    query = [1.0, 0.0]
    rows = [_community(f"c{i}", [1.0, float(i)]) for i in range(6)]
    hits = recall_communities(query, rows, k=2, scope="s1")
    assert [h.id for h in hits] == ["c0", "c1"]


def test_recall_communities_ties_break_by_id_ascending():
    query = [1.0, 0.0]
    rows = [
        _community("c-b", [2.0, 0.0]),
        _community("c-c", [1.0, 0.0]),
        _community("c-a", [3.0, 0.0]),
        _community("c-z", [0.0, 1.0]),
    ]
    hits = recall_communities(query, rows, k=10, scope="s1")
    assert [h.id for h in hits] == ["c-a", "c-b", "c-c", "c-z"]


def test_recall_communities_accepts_iterator_rows():
    rows = [_community("c1", [1.0, 0.0]), _community("c2", [0.0, 1.0])]
    hits = recall_communities([1.0, 0.0], iter(rows), k=5, scope="s1")
    assert [h.id for h in hits] == ["c1", "c2"]


def test_recall_communities_filters_by_scope():
    rows = [
        _community("c1", [1.0, 0.0], scope="s1"),
        _community("c2", [1.0, 0.0], scope="s2"),
        _community("c3", [0.5, 0.5], scope="s1"),
    ]
    hits = recall_communities([1.0, 0.0], rows, k=10, scope="s1")
    assert sorted(h.id for h in hits) == ["c1", "c3"]
    hits2 = recall_communities([1.0, 0.0], rows, k=10, scope="s2")
    assert [h.id for h in hits2] == ["c2"]


def test_recall_communities_skips_missing_or_none_embedding():
    no_key = _community("c-nokey", [1.0, 0.0])
    del no_key["embedding"]
    rows = [
        _community("c-ok", [1.0, 0.0]),
        _community("c-none", None),
        no_key,
    ]
    hits = recall_communities([1.0, 0.0], rows, k=10, scope="s1")
    assert [h.id for h in hits] == ["c-ok"]


def test_recall_communities_band_global_keeps_highest_level_in_scope():
    rows = [
        _community("c0", [1.0, 0.0], level=0),
        _community("c1a", [0.5, 0.5], level=1),
        _community("c1b", [0.0, 1.0], level=1),
        _community("other", [1.0, 0.0], level=5, scope="s2"),
    ]
    hits = recall_communities([1.0, 0.0], rows, k=10, scope="s1", band="global")
    assert [h.id for h in hits] == ["c1a", "c1b"]
    assert all(h.level == 1 for h in hits)


def test_recall_communities_band_all_is_default_and_keeps_all_levels():
    rows = [
        _community("c0", [1.0, 0.0], level=0),
        _community("c1", [0.0, 1.0], level=1),
    ]
    default = recall_communities([1.0, 0.0], rows, k=10, scope="s1")
    explicit = recall_communities([1.0, 0.0], rows, k=10, scope="s1", band="all")
    assert [h.id for h in default] == ["c0", "c1"]
    assert [h.id for h in explicit] == ["c0", "c1"]


def test_recall_communities_level_filter():
    rows = [
        _community("c0", [1.0, 0.0], level=0),
        _community("c1", [1.0, 0.0], level=1),
        _community("c2", [1.0, 0.0], level=2),
        _community("c1b", [0.0, 1.0], level=1),
    ]
    hits = recall_communities([1.0, 0.0], rows, k=10, scope="s1", level=1)
    assert [h.id for h in hits] == ["c1", "c1b"]
    hits0 = recall_communities([1.0, 0.0], rows, k=10, scope="s1", level=0)
    assert [h.id for h in hits0] == ["c0"]


def test_recall_communities_domain_filter_uses_member_domains():
    rows = [
        _community("c-fin", [1.0, 0.0], members=["e1", "e2"]),
        _community("c-mixed", [0.5, 0.5], members=["e3", "e4"]),
        _community("c-legal", [1.0, 0.0], members=["e5"]),
        _community("c-unknown", [1.0, 0.0], members=["e9"]),
    ]
    member_domains = {
        "e1": "finance",
        "e2": "finance",
        "e3": "legal",
        "e4": "finance",
        "e5": "legal",
    }
    hits = recall_communities(
        [1.0, 0.0], rows, k=10, scope="s1", domain="finance", member_domains=member_domains
    )
    assert [h.id for h in hits] == ["c-fin", "c-mixed"]


def test_recall_communities_domain_without_member_domains_raises():
    rows = [_community("c1", [1.0, 0.0])]
    with pytest.raises(ValueError):
        recall_communities([1.0, 0.0], rows, k=5, scope="s1", domain="finance")


@pytest.mark.parametrize("k", [0, -1])
def test_recall_communities_rejects_k_below_one(k):
    rows = [_community("c1", [1.0, 0.0])]
    with pytest.raises(ValueError):
        recall_communities([1.0, 0.0], rows, k=k, scope="s1")


def test_recall_communities_summary_returned_unchanged():
    summary = "  Precomputed summary, with Spacing and punctuation!\nSecond line.  "
    rows = [_community("c1", [1.0, 0.0], summary=summary)]
    hits = recall_communities([1.0, 0.0], rows, k=1, scope="s1")
    assert hits[0].summary == summary


# ---------------------------------------------------------------- search_entities


def test_search_entities_orders_and_limits():
    query = [1.0, 0.0]
    rows = [
        _entity("e-c", [0.0, 1.0], domain="d1"),
        _entity("e-b", [1.0, 0.0], domain="d2"),
        _entity("e-a", [2.0, 0.0]),
        _entity("e-d", [1.0, 1.0], omit_domain=True),
    ]
    hits = search_entities(query, rows, k=3, scope="s1")
    assert [h.id for h in hits] == ["e-a", "e-b", "e-d"]
    assert all(isinstance(h, EntityHit) for h in hits)
    by_id = {h.id: h for h in hits}
    assert by_id["e-b"].domain == "d2"
    assert by_id["e-a"].domain is None
    assert by_id["e-d"].domain is None
    for hit in hits:
        row = next(r for r in rows if r["id"] == hit.id)
        assert hit.score == pytest.approx(cosine(query, row["embedding"]))


def test_search_entities_filters_scope_and_skips_missing_embedding():
    no_key = _entity("e-nokey", [1.0, 0.0])
    del no_key["embedding"]
    rows = [
        _entity("e1", [1.0, 0.0], scope="s1"),
        _entity("e2", [1.0, 0.0], scope="s2"),
        _entity("e3", None, scope="s1"),
        no_key,
    ]
    hits = search_entities([1.0, 0.0], iter(rows), k=10, scope="s1")
    assert [h.id for h in hits] == ["e1"]


def test_search_entities_domain_filter():
    rows = [
        _entity("e1", [1.0, 0.0], domain="finance"),
        _entity("e2", [0.5, 0.5], domain="legal"),
        _entity("e3", [0.0, 1.0], domain="finance"),
        _entity("e4", [1.0, 0.0], domain=None),
        _entity("e5", [1.0, 0.0], omit_domain=True),
    ]
    hits = search_entities([1.0, 0.0], rows, k=10, scope="s1", domain="finance")
    assert [h.id for h in hits] == ["e1", "e3"]
    assert all(h.domain == "finance" for h in hits)


@pytest.mark.parametrize("k", [0, -3])
def test_search_entities_rejects_k_below_one(k):
    rows = [_entity("e1", [1.0, 0.0])]
    with pytest.raises(ValueError):
        search_entities([1.0, 0.0], rows, k=k, scope="s1")


# ---------------------------------------------------------------- interface


def test_hits_are_frozen_dataclasses():
    chit = CommunityHit(id="c", level=0, summary="s", size=1, members=("a",), score=0.5)
    ehit = EntityHit(id="e", domain=None, score=0.5)
    assert dataclasses.is_dataclass(chit)
    assert dataclasses.is_dataclass(ehit)
    with pytest.raises(dataclasses.FrozenInstanceError):
        chit.score = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ehit.score = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("func", [recall_communities, search_entities])
def test_functions_take_no_model_or_embedder(func):
    params = set(inspect.signature(func).parameters)
    for forbidden in ("model", "llm", "embedder", "embed", "client", "session", "driver"):
        assert forbidden not in params
