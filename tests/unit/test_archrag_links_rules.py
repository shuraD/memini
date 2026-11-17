"""GR-09: pure linking and relation rules of core/archrag/links.py (spec file 5).

A17 sequence pairs, A19/I72 specific labels, A88/I80 type normalisation, A79 to A83/I79
merge rule, A96 new types. No database in these tests.
"""

import copy

from memini.core.archrag.links import (
    Relation,
    merge_relations,
    new_types,
    normalise_type,
    sequence_pairs,
    specific_labels,
)

RANK = ["explicit", "parser", "llm"]


# --- A17: sequence pairs -----------------------------------------------------------------


def test_sequence_pairs_links_consecutive_entities_in_reading_order() -> None:
    assert sequence_pairs(["a", "b", "c"]) == [("a", "b"), ("b", "c")]


def test_sequence_pairs_of_empty_or_single_is_empty() -> None:
    assert sequence_pairs([]) == []
    assert sequence_pairs(["a"]) == []


def test_sequence_pairs_skips_self_pairs() -> None:
    assert sequence_pairs(["a", "a", "b"]) == [("a", "b")]


def test_sequence_pairs_does_not_repeat_a_pair() -> None:
    assert sequence_pairs(["a", "b", "a", "b"]) == [("a", "b"), ("b", "a")]


# --- A19, I72: specific labels -----------------------------------------------------------


def _ten_artifacts() -> dict[str, list[str]]:
    data: dict[str, list[str]] = {f"art{i}": [] for i in range(10)}
    data["art0"] += ["two", "one", "three"]
    data["art1"] += ["two", "three"]
    data["art2"] += ["three"]
    return data


def test_specific_labels_keeps_label_on_two_of_ten() -> None:
    assert specific_labels(_ten_artifacts()) == {"two"}


def test_specific_labels_explicit_thresholds() -> None:
    assert specific_labels(_ten_artifacts(), min_artifacts=2, max_share=0.3) == {"two", "three"}
    assert specific_labels(_ten_artifacts(), min_artifacts=1, max_share=0.2) == {"one", "two"}


def test_specific_labels_counts_a_label_once_per_artifact() -> None:
    data = _ten_artifacts()
    data["art3"] = ["dup", "dup", "dup"]
    assert "dup" not in specific_labels(data)


def test_specific_labels_share_is_over_all_artifacts() -> None:
    data = {"x": ["l"], "y": ["l"], "z": []}
    assert specific_labels(data, min_artifacts=2, max_share=0.2) == set()
    assert specific_labels(data, min_artifacts=2, max_share=0.7) == {"l"}


def test_specific_labels_of_no_artifacts_is_empty() -> None:
    assert specific_labels({}) == set()


def test_specific_labels_does_not_mutate_input() -> None:
    data = _ten_artifacts()
    before = copy.deepcopy(data)
    specific_labels(data)
    assert data == before


# --- A88, I80: type normalisation --------------------------------------------------------


def test_normalise_type_examples() -> None:
    assert normalise_type(" depends on ") == "DEPENDS_ON"
    assert normalise_type("part-of") == "PART_OF"
    assert normalise_type("uses__API") == "USES_API"
    assert normalise_type("Calls") == "CALLS"
    assert normalise_type("--is a.kind/of!") == "IS_A_KIND_OF"
    assert normalise_type("v2 ref") == "V2_REF"


# --- A79 to A83, I79: merge rule ---------------------------------------------------------


def test_merge_keeps_one_relation_per_key_with_normalised_type() -> None:
    rels = [
        Relation("a", "b", "depends on", "llm", 0.4, ("e1",)),
        Relation("a", "b", "DEPENDS_ON", "parser", 0.9, ("e2",)),
    ]
    merged = merge_relations(rels, RANK)
    assert len(merged) == 1
    assert merged[0] == Relation("a", "b", "DEPENDS_ON", "parser", 0.9, ("e1", "e2"))


def test_merge_strongest_origin_and_highest_confidence_win_independently() -> None:
    rels = [
        Relation("a", "b", "uses", "llm", 0.95, ("x",)),
        Relation("a", "b", "uses", "explicit", 0.5, ("y", "x")),
        Relation("a", "b", "uses", "parser", 0.7, ("z",)),
    ]
    (merged,) = merge_relations(rels, RANK)
    assert merged.origin == "explicit"
    assert merged.confidence == 0.95
    assert merged.evidence == ("x", "y", "z")


def test_merge_unlisted_origin_is_weakest_and_first_unlisted_stays() -> None:
    rels = [
        Relation("a", "b", "uses", "mystery", 0.1),
        Relation("a", "b", "uses", "llm", 0.1),
    ]
    assert merge_relations(rels, RANK)[0].origin == "llm"
    rels2 = [
        Relation("a", "b", "uses", "first", 0.1),
        Relation("a", "b", "uses", "second", 0.2),
    ]
    assert merge_relations(rels2, RANK)[0].origin == "first"


def test_merge_keeps_types_and_directions_apart_in_first_seen_order() -> None:
    rels = [
        Relation("a", "b", "uses", "llm", 0.5),
        Relation("b", "a", "uses", "llm", 0.5),
        Relation("a", "b", "calls", "llm", 0.5),
        Relation("a", "b", "Uses", "llm", 0.6),
    ]
    merged = merge_relations(rels, RANK)
    assert [(r.source, r.target, r.type) for r in merged] == [
        ("a", "b", "USES"),
        ("b", "a", "USES"),
        ("a", "b", "CALLS"),
    ]
    assert merged[0].confidence == 0.6


def test_merge_single_relation_only_normalises_type() -> None:
    rel = Relation("s", "t", "part of", "parser", 0.3, ("ev",))
    assert merge_relations([rel], RANK) == [Relation("s", "t", "PART_OF", "parser", 0.3, ("ev",))]


def test_merge_of_nothing_is_empty() -> None:
    assert merge_relations([], RANK) == []


def test_merge_does_not_mutate_inputs() -> None:
    rels = [
        Relation("a", "b", "uses", "llm", 0.5, ("x",)),
        Relation("a", "b", "uses", "parser", 0.6, ("y",)),
    ]
    rank = list(RANK)
    before = list(rels)
    merge_relations(rels, rank)
    assert rels == before
    assert rank == RANK


def test_merge_accepts_a_generator() -> None:
    gen = (Relation("a", "b", "uses", "llm", c) for c in (0.1, 0.3, 0.2))
    assert merge_relations(gen, RANK) == [Relation("a", "b", "USES", "llm", 0.3)]


# --- A96: new types flagged for review ---------------------------------------------------


def test_new_types_are_sorted_unique_and_normalised() -> None:
    vocab = ["DEPENDS_ON", "part of"]
    types = ["depends on", "Part-Of", "uses", "USES", "calls", "uses"]
    assert new_types(types, vocab) == ["CALLS", "USES"]


def test_new_types_with_empty_vocabulary_flags_everything() -> None:
    assert new_types(["b", "a"], []) == ["A", "B"]


def test_new_types_all_known_is_empty() -> None:
    assert new_types(["Calls"], ["CALLS"]) == []
