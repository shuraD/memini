"""Acceptance tests for AR-16: pure relation rules (src/memini/core/archrag/relations.py)."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from memini.core.archrag.relations import (
    Relation,
    RelationSettings,
    edge_type,
    fold_relations,
    merge_relation,
    merge_settings,
    normalize_relation_type,
)

RANK = ["asserted", "explicit", "confirmed", "extracted"]


def _section() -> dict[str, Any]:
    return {
        "origin_rank": list(RANK),
        "origins": {
            "asserted": {"confidence": 1.0},
            "explicit": {"confidence": 0.9},
            "confirmed": {"confidence": None},
            "extracted": {"confidence": 0.5},
        },
    }


@pytest.fixture
def settings() -> RelationSettings:
    return RelationSettings.from_config(_section())


# --- merge_settings ---------------------------------------------------------


def test_merge_settings_deep_merges_and_adds_keys() -> None:
    defaults = {"a": 1, "b": {"x": 1, "y": 2}}
    overrides = {"b": {"y": 3}, "c": 4}
    assert merge_settings(defaults, overrides) == {"a": 1, "b": {"x": 1, "y": 3}, "c": 4}


def test_merge_settings_none_override_value_keeps_default() -> None:
    defaults = {"a": 1, "b": {"x": 1, "y": 2}}
    assert merge_settings(defaults, {"a": None, "b": {"x": None}}) == {
        "a": 1,
        "b": {"x": 1, "y": 2},
    }


def test_merge_settings_overrides_none_returns_equal_copy() -> None:
    defaults = {"a": 1, "b": {"x": 1, "y": 2}}
    result = merge_settings(defaults, None)
    assert result == defaults
    assert result is not defaults


def test_merge_settings_does_not_mutate_inputs() -> None:
    defaults = {"a": 1, "b": {"x": 1, "y": 2}}
    overrides = {"b": {"y": 3, "z": {"k": 1}}, "c": 4}
    defaults_before = copy.deepcopy(defaults)
    overrides_before = copy.deepcopy(overrides)
    result = merge_settings(defaults, overrides)
    result["b"]["x"] = 99
    result["a"] = 42
    assert defaults == defaults_before
    assert overrides == overrides_before
    copy_result = merge_settings(defaults, None)
    copy_result["b"]["y"] = 77
    assert defaults == defaults_before


# --- edge_type --------------------------------------------------------------


def test_edge_type_returns_configured_name() -> None:
    assert edge_type({"relation": "RELATED_TO"}, "relation") == "RELATED_TO"
    assert edge_type({"part": "PART_OF2", "x": "X"}, "part") == "PART_OF2"


def test_edge_type_missing_role_raises() -> None:
    with pytest.raises(ValueError):
        edge_type({"relation": "RELATED_TO"}, "other")


@pytest.mark.parametrize("value", [None, 3, ["RELATED_TO"], b"RELATED_TO"])
def test_edge_type_non_str_raises(value: object) -> None:
    with pytest.raises(ValueError):
        edge_type({"relation": value}, "relation")


@pytest.mark.parametrize(
    "value", ["related_to", "Related-To", "", "RELATED-TO", "_RELATED", "RELATED_", "1REL", "A__B"]
)
def test_edge_type_bad_name_raises(value: str) -> None:
    with pytest.raises(ValueError):
        edge_type({"relation": value}, "relation")


# --- normalize_relation_type -------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Derived From", "derived_from"),
        ("derived-from", "derived_from"),
        ("  IS-part_of ", "is_part_of"),
        ("DERIVED_FROM", "derived_from"),
        ("uses", "uses"),
        ("Step 2 of", "step_2_of"),
    ],
)
def test_normalize_relation_type(name: str, expected: str) -> None:
    assert normalize_relation_type(name) == expected


@pytest.mark.parametrize("name", [None, "", "   ", "--", "_ _"])
def test_normalize_relation_type_without_word_raises(name: object) -> None:
    with pytest.raises(ValueError):
        normalize_relation_type(name)


# --- RelationSettings.from_config -------------------------------------------


def test_from_config_reads_rank_and_confidences(settings: RelationSettings) -> None:
    assert settings.origin_rank == tuple(RANK)
    assert dict(settings.default_confidence) == {
        "asserted": 1.0,
        "explicit": 0.9,
        "confirmed": None,
        "extracted": 0.5,
    }


@pytest.mark.parametrize("missing", ["origin_rank", "origins"])
def test_from_config_missing_key_raises(missing: str) -> None:
    section = _section()
    del section[missing]
    with pytest.raises(ValueError):
        RelationSettings.from_config(section)


def test_from_config_empty_rank_raises() -> None:
    section = _section()
    section["origin_rank"] = []
    with pytest.raises(ValueError):
        RelationSettings.from_config(section)


def test_from_config_ranked_origin_absent_from_origins_raises() -> None:
    section = _section()
    section["origin_rank"] = [*RANK, "inferred"]
    with pytest.raises(ValueError):
        RelationSettings.from_config(section)


@pytest.mark.parametrize("bad_rank", ["asserted", [1, 2], None])
def test_from_config_malformed_rank_raises(bad_rank: object) -> None:
    section = _section()
    section["origin_rank"] = bad_rank
    with pytest.raises(ValueError):
        RelationSettings.from_config(section)


def test_from_config_malformed_origins_raises() -> None:
    section = _section()
    section["origins"] = ["asserted", "explicit"]
    with pytest.raises(ValueError):
        RelationSettings.from_config(section)


def test_from_config_non_number_confidence_raises() -> None:
    section = _section()
    section["origins"]["explicit"] = {"confidence": "high"}
    with pytest.raises(ValueError):
        RelationSettings.from_config(section)


# --- Relation ---------------------------------------------------------------


def test_relation_key_and_frozen() -> None:
    rel = Relation(src="a", dst="b", type="uses", origin="extracted")
    assert rel.key == ("a", "b", "uses")
    assert rel.description == ()
    assert rel.evidence == ()
    assert rel.evidence_refs == ()
    assert rel.origins == ()
    assert rel.confidence is None
    with pytest.raises(AttributeError):
        rel.src = "c"  # type: ignore[misc]


# --- merge_relation ---------------------------------------------------------


def test_merge_relation_none_existing(settings: RelationSettings) -> None:
    new = Relation(
        src="a",
        dst="b",
        type="uses",
        origin="explicit",
        description=("d1", "d2", "d1"),
        evidence=("e1", "e1", "e2"),
        evidence_refs=("r2", "r1", "r2"),
        confidence=0.7,
    )
    out = merge_relation(None, new, settings)
    assert (out.src, out.dst, out.type, out.origin) == ("a", "b", "uses", "explicit")
    assert out.description == ("d1", "d2")
    assert out.evidence == ("e1", "e2")
    assert out.evidence_refs == ("r2", "r1")
    assert out.origins == ("explicit",)
    assert out.confidence == 0.7


def test_merge_relation_none_existing_uses_default_confidence(
    settings: RelationSettings,
) -> None:
    new = Relation(src="a", dst="b", type="uses", origin="extracted")
    out = merge_relation(None, new, settings)
    assert out.confidence == 0.5
    new2 = Relation(src="a", dst="b", type="uses", origin="confirmed")
    assert merge_relation(None, new2, settings).confidence is None


def test_merge_relation_unions_lists_existing_first(settings: RelationSettings) -> None:
    existing = merge_relation(
        None,
        Relation(
            src="a",
            dst="b",
            type="uses",
            origin="extracted",
            description=("d1", "d2"),
            evidence=("e1",),
            evidence_refs=("r1", "r2"),
        ),
        settings,
    )
    new = Relation(
        src="a",
        dst="b",
        type="uses",
        origin="explicit",
        description=("d3", "d1"),
        evidence=("e2", "e1", "e2"),
        evidence_refs=("r2", "r3"),
    )
    out = merge_relation(existing, new, settings)
    assert out.description == ("d1", "d2", "d3")
    assert out.evidence == ("e1", "e2")
    assert out.evidence_refs == ("r1", "r2", "r3")
    assert out.origins == ("extracted", "explicit")


def test_merge_relation_same_origin_not_duplicated(settings: RelationSettings) -> None:
    first = merge_relation(None, Relation("a", "b", "uses", "extracted"), settings)
    out = merge_relation(first, Relation("a", "b", "uses", "extracted"), settings)
    assert out.origins == ("extracted",)
    assert out.origin == "extracted"


def test_merge_relation_stronger_origin_wins_either_side(
    settings: RelationSettings,
) -> None:
    weak = merge_relation(None, Relation("a", "b", "uses", "extracted"), settings)
    strong_new = Relation("a", "b", "uses", "asserted")
    assert merge_relation(weak, strong_new, settings).origin == "asserted"

    strong = merge_relation(None, Relation("a", "b", "uses", "explicit"), settings)
    weak_new = Relation("a", "b", "uses", "confirmed")
    assert merge_relation(strong, weak_new, settings).origin == "explicit"


def test_merge_relation_confidence_is_max(settings: RelationSettings) -> None:
    low = merge_relation(None, Relation("a", "b", "uses", "extracted", confidence=0.2), settings)
    out = merge_relation(low, Relation("a", "b", "uses", "extracted", confidence=0.8), settings)
    assert out.confidence == 0.8

    high = merge_relation(None, Relation("a", "b", "uses", "extracted", confidence=0.95), settings)
    out2 = merge_relation(high, Relation("a", "b", "uses", "extracted", confidence=0.3), settings)
    assert out2.confidence == 0.95


def test_merge_relation_confidence_new_default_applies(settings: RelationSettings) -> None:
    low = merge_relation(None, Relation("a", "b", "uses", "confirmed", confidence=0.1), settings)
    # new.confidence None -> default of "explicit" (0.9) applies
    out = merge_relation(low, Relation("a", "b", "uses", "explicit"), settings)
    assert out.confidence == 0.9


def test_merge_relation_confidence_one_side_none(settings: RelationSettings) -> None:
    none_side = merge_relation(None, Relation("a", "b", "uses", "confirmed"), settings)
    assert none_side.confidence is None
    out = merge_relation(
        none_side, Relation("a", "b", "uses", "confirmed", confidence=0.4), settings
    )
    assert out.confidence == 0.4

    some = merge_relation(None, Relation("a", "b", "uses", "confirmed", confidence=0.6), settings)
    out2 = merge_relation(some, Relation("a", "b", "uses", "confirmed"), settings)
    assert out2.confidence == 0.6


def test_merge_relation_both_confidence_none(settings: RelationSettings) -> None:
    first = merge_relation(None, Relation("a", "b", "uses", "confirmed"), settings)
    out = merge_relation(first, Relation("a", "b", "uses", "confirmed"), settings)
    assert out.confidence is None


def test_merge_relation_unknown_origin_raises(settings: RelationSettings) -> None:
    with pytest.raises(ValueError):
        merge_relation(None, Relation("a", "b", "uses", "guessed"), settings)
    existing = merge_relation(None, Relation("a", "b", "uses", "extracted"), settings)
    with pytest.raises(ValueError):
        merge_relation(existing, Relation("a", "b", "uses", "guessed"), settings)


def test_merge_relation_different_keys_raise(settings: RelationSettings) -> None:
    existing = merge_relation(None, Relation("a", "b", "uses", "extracted"), settings)
    with pytest.raises(ValueError):
        merge_relation(existing, Relation("a", "c", "uses", "extracted"), settings)
    with pytest.raises(ValueError):
        merge_relation(existing, Relation("x", "b", "uses", "extracted"), settings)
    with pytest.raises(ValueError):
        merge_relation(existing, Relation("a", "b", "owns", "extracted"), settings)


def test_merge_relation_does_not_mutate_inputs(settings: RelationSettings) -> None:
    existing = merge_relation(
        None, Relation("a", "b", "uses", "extracted", description=("d1",)), settings
    )
    new = Relation("a", "b", "uses", "explicit", description=("d2",), confidence=0.3)
    existing_before = copy.deepcopy(existing)
    new_before = copy.deepcopy(new)
    merge_relation(existing, new, settings)
    assert existing == existing_before
    assert new == new_before


# --- fold_relations ---------------------------------------------------------


def test_fold_relations_groups_by_normalised_key_in_first_order(
    settings: RelationSettings,
) -> None:
    rels = [
        Relation("a", "b", "Derived From", "extracted", description=("d1",), confidence=0.3),
        Relation("c", "d", "uses", "explicit", description=("u1",)),
        Relation("a", "b", "derived-from", "asserted", description=("d2", "d1"), confidence=0.2),
        Relation("b", "a", "derived_from", "extracted"),
        Relation("a", "b", "DERIVED_FROM", "extracted", evidence=("e1",)),
    ]
    before = copy.deepcopy(rels)
    out = fold_relations(rels, settings)
    assert rels == before
    assert [r.key for r in out] == [
        ("a", "b", "derived_from"),
        ("c", "d", "uses"),
        ("b", "a", "derived_from"),
    ]
    first = out[0]
    assert first.type == "derived_from"
    assert first.description == ("d1", "d2")
    assert first.evidence == ("e1",)
    assert first.origin == "asserted"
    assert first.origins == ("extracted", "asserted")
    # max of 0.3, 0.2 and the "extracted" default 0.5 of the last (confidence None)
    assert first.confidence == 0.5
    second = out[1]
    assert second.confidence == 0.9
    assert second.origins == ("explicit",)


def test_fold_relations_matches_left_to_right_merge(settings: RelationSettings) -> None:
    group = [
        Relation("a", "b", "uses", "extracted", description=("x",), confidence=0.1),
        Relation("a", "b", "uses", "confirmed", evidence=("e",)),
        Relation("a", "b", "uses", "explicit", evidence_refs=("r",), confidence=0.4),
    ]
    expected: Relation | None = None
    for rel in group:
        expected = merge_relation(expected, rel, settings)
    assert fold_relations(group, settings) == [expected]


def test_fold_relations_drops_self_loops_and_empty_ends(settings: RelationSettings) -> None:
    rels = [
        Relation("a", "a", "uses", "extracted"),
        Relation("", "b", "uses", "extracted"),
        Relation("a", "", "uses", "extracted"),
        Relation("a", "b", "uses", "extracted"),
    ]
    out = fold_relations(rels, settings)
    assert [r.key for r in out] == [("a", "b", "uses")]


def test_fold_relations_empty_input(settings: RelationSettings) -> None:
    assert fold_relations([], settings) == []
    assert fold_relations(iter(()), settings) == []
