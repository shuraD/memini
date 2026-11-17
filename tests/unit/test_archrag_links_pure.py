"""Acceptance tests for GR-09: pure linking and relation helpers in
memini.core.archrag.links (sequence pairs, specific labels, type
normalisation, relation merge, new vocabulary types)."""

import copy
import dataclasses

import pytest

from memini.core.archrag.links import (
    Relation,
    merge_relations,
    new_types,
    normalise_type,
    sequence_pairs,
    specific_labels,
)

# --- Relation -----------------------------------------------------------------


def test_relation_is_frozen_with_empty_default_evidence():
    rel = Relation(source="a", target="b", type="USES", origin="x", confidence=0.5)
    assert rel.evidence == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rel.source = "z"  # type: ignore[misc]


# --- sequence_pairs -----------------------------------------------------------


def test_sequence_pairs_reading_order():
    assert sequence_pairs(["a", "b", "c"]) == [("a", "b"), ("b", "c")]


@pytest.mark.parametrize("ids", [[], ["a"]])
def test_sequence_pairs_short_input_is_empty(ids):
    assert sequence_pairs(ids) == []


def test_sequence_pairs_no_self_pair():
    assert sequence_pairs(["a", "a", "b"]) == [("a", "b")]


def test_sequence_pairs_no_repeated_pair():
    assert sequence_pairs(["a", "b", "a", "b"]) == [("a", "b"), ("b", "a")]


def test_sequence_pairs_accepts_tuple():
    assert sequence_pairs(("x", "y")) == [("x", "y")]


# --- specific_labels ----------------------------------------------------------


def _ten_artifacts(label_on: dict[str, int]) -> dict[str, list[str]]:
    arts: dict[str, list[str]] = {f"art{i}": [] for i in range(10)}
    for label, count in label_on.items():
        for i in range(count):
            arts[f"art{i}"].append(label)
    return arts


def test_specific_labels_defaults_ten_artifacts():
    arts = _ten_artifacts({"two": 2, "one": 1, "three": 3})
    assert specific_labels(arts) == {"two"}


def test_specific_labels_no_artifacts():
    result = specific_labels({})
    assert result == set()
    assert isinstance(result, set)


def test_specific_labels_repeat_inside_artifact_counts_once():
    arts = _ten_artifacts({})
    arts["art0"] = ["dup", "dup", "dup"]
    assert specific_labels(arts) == set()
    arts["art1"] = ["dup"]
    assert specific_labels(arts) == {"dup"}


def test_specific_labels_custom_thresholds():
    arts = {
        "p": ["x", "y"],
        "q": ["x", "y"],
        "r": ["x"],
        "s": ["z"],
    }
    # x on 3/4 = 0.75, y on 2/4 = 0.5, z on 1/4 = 0.25
    assert specific_labels(arts, min_artifacts=1, max_share=0.5) == {"y", "z"}
    assert specific_labels(arts, min_artifacts=3, max_share=1.0) == {"x"}
    assert specific_labels(arts, min_artifacts=2, max_share=0.75) == {"x", "y"}


def test_specific_labels_accepts_iterables_of_any_kind():
    arts = {f"a{i}": () for i in range(10)}
    arts["a0"] = {"k"}
    arts["a1"] = ("k",)
    assert specific_labels(arts) == {"k"}


def test_specific_labels_does_not_mutate_input():
    arts = _ten_artifacts({"two": 2, "one": 1, "three": 3})
    arts["art0"].append("two")
    before = copy.deepcopy(arts)
    specific_labels(arts)
    assert arts == before


# --- normalise_type -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" depends on ", "DEPENDS_ON"),
        ("part-of", "PART_OF"),
        ("uses__API", "USES_API"),
        ("Calls", "CALLS"),
        ("--refers  to!!", "REFERS_TO"),
        ("v2 link", "V2_LINK"),
        ("a . - b", "A_B"),
    ],
)
def test_normalise_type(raw, expected):
    assert normalise_type(raw) == expected


def test_normalise_type_is_idempotent():
    assert normalise_type(normalise_type(" depends on ")) == "DEPENDS_ON"


# --- merge_relations ----------------------------------------------------------

RANK = ["explicit", "parser", "llm"]


def test_merge_relations_normalises_type_and_keeps_first_appearance_order():
    rels = [
        Relation("b", "c", "part-of", "parser", 0.5),
        Relation("a", "b", "uses", "parser", 0.5),
        Relation("b", "c", "Part Of", "parser", 0.5),
    ]
    merged = merge_relations(rels, RANK)
    assert [(r.source, r.target, r.type) for r in merged] == [
        ("b", "c", "PART_OF"),
        ("a", "b", "USES"),
    ]


def test_merge_relations_single_relation_unchanged_except_type():
    rel = Relation("a", "b", "depends on", "llm", 0.3, ("e1",))
    assert merge_relations([rel], RANK) == [Relation("a", "b", "DEPENDS_ON", "llm", 0.3, ("e1",))]


def test_merge_relations_duplicates_combine_evidence_confidence_origin():
    rels = [
        Relation("a", "b", "uses", "llm", 0.4, ("e1", "e2")),
        Relation("a", "b", "USES", "explicit", 0.2, ("e2", "e3")),
        Relation("a", "b", " Uses ", "parser", 0.9, ("e1", "e4")),
    ]
    assert merge_relations(rels, RANK) == [
        Relation("a", "b", "USES", "explicit", 0.9, ("e1", "e2", "e3", "e4"))
    ]


def test_merge_relations_unlisted_origin_is_weaker_than_listed():
    rels = [
        Relation("a", "b", "uses", "manual", 0.5),
        Relation("a", "b", "uses", "llm", 0.5),
    ]
    assert merge_relations(rels, RANK)[0].origin == "llm"
    rels_rev = [
        Relation("a", "b", "uses", "llm", 0.5),
        Relation("a", "b", "uses", "manual", 0.5),
    ]
    assert merge_relations(rels_rev, RANK)[0].origin == "llm"


def test_merge_relations_two_unlisted_origins_first_seen_stays():
    rels = [
        Relation("a", "b", "uses", "zeta", 0.1),
        Relation("a", "b", "uses", "alpha", 0.8),
    ]
    merged = merge_relations(rels, RANK)
    assert len(merged) == 1
    assert merged[0].origin == "zeta"
    assert merged[0].confidence == 0.8


def test_merge_relations_weaker_later_origin_does_not_replace():
    rels = [
        Relation("a", "b", "uses", "parser", 0.1),
        Relation("a", "b", "uses", "llm", 0.9),
    ]
    merged = merge_relations(rels, RANK)
    assert merged[0].origin == "parser"
    assert merged[0].confidence == 0.9


def test_merge_relations_different_types_stay_separate():
    rels = [
        Relation("a", "b", "uses", "parser", 0.5),
        Relation("a", "b", "calls", "parser", 0.5),
    ]
    merged = merge_relations(rels, RANK)
    assert [r.type for r in merged] == ["USES", "CALLS"]


def test_merge_relations_direction_matters():
    rels = [
        Relation("a", "b", "uses", "parser", 0.5),
        Relation("b", "a", "uses", "parser", 0.5),
    ]
    merged = merge_relations(rels, RANK)
    assert [(r.source, r.target) for r in merged] == [("a", "b"), ("b", "a")]


def test_merge_relations_empty_input():
    assert merge_relations([], RANK) == []


def test_merge_relations_accepts_generator_and_does_not_mutate_input():
    rels = [
        Relation("a", "b", "uses", "llm", 0.4, ("e1",)),
        Relation("a", "b", "USES", "explicit", 0.2, ("e2",)),
    ]
    rank = list(RANK)
    before = list(rels)
    rank_before = list(rank)
    merged = merge_relations((r for r in rels), rank)
    assert rels == before
    assert rels[0] == Relation("a", "b", "uses", "llm", 0.4, ("e1",))
    assert rank == rank_before
    assert merged == [Relation("a", "b", "USES", "explicit", 0.4, ("e1", "e2"))]


def test_merge_relations_returns_relation_instances():
    merged = merge_relations([Relation("a", "b", "x", "parser", 1.0)], RANK)
    assert all(isinstance(r, Relation) for r in merged)
    assert isinstance(merged, list)


# --- new_types ----------------------------------------------------------------


def test_new_types_sorted_dedup_normalised_not_in_vocabulary():
    types = ["new thing", "calls", "Part-of", "CALLS", "new-thing", "abc"]
    vocabulary = [" calls ", "uses"]
    assert new_types(types, vocabulary) == ["ABC", "NEW_THING", "PART_OF"]


def test_new_types_all_known_is_empty():
    assert new_types(["Uses", "uses "], ["USES"]) == []


def test_new_types_empty_vocabulary():
    assert new_types(["b", "a", "A"], []) == ["A", "B"]


def test_new_types_does_not_mutate_inputs():
    types = ["x", "y"]
    vocabulary = {"X"}
    new_types(types, vocabulary)
    assert types == ["x", "y"]
    assert vocabulary == {"X"}
