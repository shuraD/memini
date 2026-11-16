"""Acceptance tests for IN-13: read, resolve, merge and measure extracted relations."""

from __future__ import annotations

import copy
import dataclasses

import pytest

from memini.core.ingest.relations import (
    ExtractedRelation,
    MeasuredRelation,
    measure_relations,
    merge_relations,
    quote_found,
    read_relations,
    resolve_relations,
)

MARKER = "NEW_TYPE"
VOCAB = ["WORKS_AT", "KNOWS"]
FINAL_IDS = ["ent-alice", "ent-acme", "ent-bob"]
STATS_KEYS = {
    "raw",
    "resolved",
    "unresolved_end",
    "self",
    "invalid",
    "no_type",
    "new",
    "off_vocabulary",
}


def item(source="ent-alice", target="ent-acme", type_="WORKS_AT", **extra):
    out = {"source": source, "target": target, "type": type_}
    out.update(extra)
    return out


def resolve(items, *, key=0, id_map=None, final_ids=None, vocabulary=None):
    return resolve_relations(
        [(key, items)],
        id_map if id_map is not None else {},
        final_ids if final_ids is not None else FINAL_IDS,
        vocabulary if vocabulary is not None else VOCAB,
        MARKER,
    )


def rel(
    source="a", target="b", type_="KNOWS", *, new=False, description=(), evidence=(), windows=1
):
    return ExtractedRelation(
        source=source,
        target=target,
        type=type_,
        new=new,
        description=tuple(description),
        evidence=tuple(evidence),
        windows=windows,
    )


# --- data classes ---------------------------------------------------------------------------


def test_dataclasses_are_frozen_and_windows_defaults_to_one():
    r = ExtractedRelation(
        source="a", target="b", type="KNOWS", new=False, description=(), evidence=()
    )
    assert r.windows == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.source = "c"  # type: ignore[misc]
    m = MeasuredRelation(relation=r, evidence_found=(), confidence=0.4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.confidence = 0.8  # type: ignore[misc]


# --- read_relations -------------------------------------------------------------------------


def test_read_relations_returns_the_relations_list():
    items = [{"source": "a"}, 3, "x"]
    assert read_relations({"relations": items, "entities": []}) == items


@pytest.mark.parametrize(
    "reply",
    [
        None,
        [],
        [{"relations": [1]}],
        "relations",
        42,
        {},
        {"entities": [1]},
        {"relations": "not a list"},
        {"relations": {"a": 1}},
        {"relations": None},
    ],
)
def test_read_relations_returns_empty_list_otherwise(reply):
    assert read_relations(reply) == []


# --- resolve_relations: stats ---------------------------------------------------------------


def test_stats_have_exactly_the_documented_int_keys():
    _, stats = resolve([item()])
    assert set(stats) == STATS_KEYS
    assert all(type(v) is int for v in stats.values())


def test_stats_for_no_replies_are_all_zero():
    rels, stats = resolve_relations([], {}, FINAL_IDS, VOCAB, MARKER)
    assert rels == []
    assert set(stats) == STATS_KEYS
    assert all(v == 0 for v in stats.values())


def test_raw_counts_every_item_across_replies():
    replies = [
        (0, [item(), "junk", item(type_="")]),
        (1, [item(source="nobody")]),
        (2, []),
    ]
    _, stats = resolve_relations(replies, {}, FINAL_IDS, VOCAB, MARKER)
    assert stats["raw"] == 4
    assert stats["resolved"] == 1
    assert stats["invalid"] == 1
    assert stats["no_type"] == 1
    assert stats["unresolved_end"] == 1


# --- resolve_relations: invalid -------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "a string",
        42,
        None,
        ["source", "target"],
        {"target": "ent-acme", "type": "WORKS_AT"},
        {"source": "ent-alice", "type": "WORKS_AT"},
        {"source": "ent-alice", "target": "ent-acme"},
        {"source": 1, "target": "ent-acme", "type": "WORKS_AT"},
        {"source": "ent-alice", "target": ["ent-acme"], "type": "WORKS_AT"},
        {"source": "ent-alice", "target": "ent-acme", "type": 7},
        {"source": None, "target": "ent-acme", "type": "WORKS_AT"},
        item(new_type=5),
        item(description=["d"]),
        item(evidence=12),
        item(evidence={"q": 1}),
    ],
)
def test_invalid_items_are_dropped_and_counted(bad):
    rels, stats = resolve([bad])
    assert rels == []
    assert stats["raw"] == 1
    assert stats["invalid"] == 1
    assert stats["resolved"] == 0
    for key in ("no_type", "unresolved_end", "self", "new", "off_vocabulary"):
        assert stats[key] == 0


def test_optional_string_fields_present_as_str_are_valid():
    rels, stats = resolve([item(new_type="", description="", evidence="")])
    assert stats["invalid"] == 0
    assert len(rels) == 1


# --- resolve_relations: no_type and new types -----------------------------------------------


@pytest.mark.parametrize("type_", ["", "   ", "\n\t"])
def test_blank_type_is_dropped_as_no_type(type_):
    rels, stats = resolve([item(type_=type_)])
    assert rels == []
    assert stats["no_type"] == 1
    assert stats["invalid"] == 0
    assert stats["resolved"] == 0


@pytest.mark.parametrize("marker", ["NEW_TYPE", "new_type", "  New_Type  "])
def test_new_marker_takes_the_stripped_new_type(marker):
    rels, stats = resolve([item(type_=marker, new_type="  MENTORS  ")])
    assert len(rels) == 1
    assert rels[0].type == "MENTORS"
    assert rels[0].new is True
    assert stats["new"] == 1
    assert stats["resolved"] == 1


@pytest.mark.parametrize("extra", [{}, {"new_type": ""}, {"new_type": "   "}])
def test_new_marker_without_new_type_is_no_type(extra):
    rels, stats = resolve([item(type_=MARKER, **extra)])
    assert rels == []
    assert stats["no_type"] == 1
    assert stats["new"] == 0
    assert stats["resolved"] == 0


def test_vocabulary_type_is_not_new():
    rels, stats = resolve([item(type_="  KNOWS ")])
    assert len(rels) == 1
    assert rels[0].type == "KNOWS"
    assert rels[0].new is False
    assert stats["new"] == 0
    assert stats["off_vocabulary"] == 0


def test_off_vocabulary_type_is_kept_as_new():
    rels, stats = resolve([item(type_=" FOUNDED ")])
    assert len(rels) == 1
    assert rels[0].type == "FOUNDED"
    assert rels[0].new is True
    assert stats["off_vocabulary"] == 1
    assert stats["resolved"] == 1


def test_new_and_off_vocabulary_count_only_kept_items():
    items = [
        item(type_=MARKER, new_type="MENTORS", source="nobody"),
        item(type_="FOUNDED", target="nowhere"),
        item(type_="FOUNDED", source="ent-bob", target="ent-bob"),
    ]
    rels, stats = resolve(items)
    assert rels == []
    assert stats["new"] == 0
    assert stats["off_vocabulary"] == 0
    assert stats["unresolved_end"] == 2
    assert stats["self"] == 1


# --- resolve_relations: ends ----------------------------------------------------------------


def test_end_resolves_through_id_map_for_the_same_reply_key():
    id_map = {(3, "p1"): "ent-alice", (3, "org"): "ent-acme"}
    rels, stats = resolve([item(source=" p1 ", target="org")], key=3, id_map=id_map)
    assert stats["resolved"] == 1
    assert (rels[0].source, rels[0].target) == ("ent-alice", "ent-acme")


def test_id_map_wins_over_final_ids():
    id_map = {(0, "ent-alice"): "ent-bob"}
    rels, _ = resolve([item(source="ent-alice", target="ent-acme")], id_map=id_map)
    assert rels[0].source == "ent-bob"


def test_end_in_final_ids_resolves_to_itself():
    rels, stats = resolve([item(source="  ent-bob ", target="ent-acme")])
    assert stats["resolved"] == 1
    assert (rels[0].source, rels[0].target) == ("ent-bob", "ent-acme")


def test_id_map_hit_for_another_reply_key_does_not_count():
    id_map = {(1, "p1"): "ent-alice"}
    rels, stats = resolve([item(source="p1")], key=0, id_map=id_map)
    assert rels == []
    assert stats["unresolved_end"] == 1


@pytest.mark.parametrize(
    ("source", "target"),
    [("unknown", "ent-acme"), ("ent-alice", "unknown"), ("", "ent-acme"), ("ent-alice", "  ")],
)
def test_unresolved_or_blank_end_is_dropped(source, target):
    rels, stats = resolve([item(source=source, target=target)])
    assert rels == []
    assert stats["unresolved_end"] == 1
    assert stats["resolved"] == 0


def test_both_ends_resolving_to_same_id_is_self():
    id_map = {(0, "p1"): "ent-alice", (0, "p2"): "ent-alice"}
    rels, stats = resolve([item(source="p1", target="p2")], id_map=id_map)
    assert rels == []
    assert stats["self"] == 1
    assert stats["resolved"] == 0


# --- resolve_relations: check order ---------------------------------------------------------


def test_each_dropped_item_is_counted_once_in_check_order():
    items = [
        {"source": "x", "target": "x", "type": "", "evidence": 3},  # invalid first
        item(source="nobody", target="nobody", type_="  "),  # no_type before unresolved
        item(source="nobody", target="nobody"),  # unresolved before self
        item(source="ent-bob", target="ent-bob"),  # self
        item(type_=MARKER, source="nobody"),  # no_type (no new_type) before unresolved
    ]
    rels, stats = resolve(items)
    assert rels == []
    assert stats["raw"] == 5
    assert stats["invalid"] == 1
    assert stats["no_type"] == 2
    assert stats["unresolved_end"] == 1
    assert stats["self"] == 1
    assert stats["resolved"] == 0
    dropped = sum(stats[k] for k in ("invalid", "no_type", "unresolved_end", "self"))
    assert dropped + stats["resolved"] == stats["raw"]


# --- resolve_relations: kept items ----------------------------------------------------------


def test_kept_item_fields():
    rels, _ = resolve([item(description="  works there  ", evidence="\n Alice works at Acme. ")])
    assert rels == [
        ExtractedRelation(
            source="ent-alice",
            target="ent-acme",
            type="WORKS_AT",
            new=False,
            description=("works there",),
            evidence=("Alice works at Acme.",),
            windows=1,
        )
    ]


@pytest.mark.parametrize("extra", [{}, {"description": "  ", "evidence": ""}])
def test_missing_or_blank_description_and_evidence_are_empty(extra):
    rels, _ = resolve([item(**extra)])
    assert rels[0].description == ()
    assert rels[0].evidence == ()
    assert rels[0].windows == 1


def test_output_order_follows_replies_then_items():
    replies = [
        (5, [item(type_="KNOWS", target="ent-bob"), item(type_="WORKS_AT")]),
        (2, [item(source="ent-bob", type_="WORKS_AT")]),
    ]
    rels, stats = resolve_relations(replies, {}, FINAL_IDS, VOCAB, MARKER)
    assert [(r.source, r.target, r.type) for r in rels] == [
        ("ent-alice", "ent-bob", "KNOWS"),
        ("ent-alice", "ent-acme", "WORKS_AT"),
        ("ent-bob", "ent-acme", "WORKS_AT"),
    ]
    assert stats["resolved"] == 3


def test_duplicates_are_not_merged_by_resolve():
    rels, stats = resolve([item(), item()])
    assert len(rels) == 2
    assert stats["resolved"] == 2


def test_resolve_does_not_mutate_arguments():
    items = [
        item(source=" p1 ", description=" d ", evidence=" e "),
        item(type_=MARKER, new_type=" X "),
        "junk",
    ]
    replies = [(0, items)]
    id_map = {(0, "p1"): "ent-bob"}
    final_ids = list(FINAL_IDS)
    vocab = list(VOCAB)
    snapshot = copy.deepcopy((replies, id_map, final_ids, vocab))
    resolve_relations(replies, id_map, final_ids, vocab, MARKER)
    assert (replies, id_map, final_ids, vocab) == snapshot


# --- merge_relations ------------------------------------------------------------------------


def test_merge_folds_same_key_in_first_seen_order():
    relations = [
        rel("a", "b", "KNOWS", description=["d1"], evidence=["e1"]),
        rel("c", "d", "KNOWS", description=["x"]),
        rel("a", "b", "KNOWS", description=["d2", "d1"], evidence=["e1", "e2"], windows=2),
        rel("a", "b", "KNOWS", description=["d3"]),
    ]
    merged = merge_relations(relations)
    assert merged == [
        rel("a", "b", "KNOWS", description=["d1", "d2", "d3"], evidence=["e1", "e2"], windows=4),
        rel("c", "d", "KNOWS", description=["x"]),
    ]


def test_merge_new_is_true_only_if_all_are_new():
    mixed = merge_relations([rel(new=True), rel(new=False), rel(new=True)])
    assert len(mixed) == 1
    assert mixed[0].new is False
    all_new = merge_relations([rel(new=True), rel(new=True)])
    assert all_new[0].new is True
    assert all_new[0].windows == 2


def test_merge_keeps_different_keys_and_directions_apart():
    relations = [rel("a", "b", "KNOWS"), rel("b", "a", "KNOWS"), rel("a", "b", "WORKS_AT")]
    merged = merge_relations(relations)
    assert [(r.source, r.target, r.type) for r in merged] == [
        ("a", "b", "KNOWS"),
        ("b", "a", "KNOWS"),
        ("a", "b", "WORKS_AT"),
    ]
    assert all(r.windows == 1 for r in merged)


def test_merge_empty_and_generator_input():
    assert merge_relations([]) == []
    merged = merge_relations(r for r in [rel(), rel()])
    assert len(merged) == 1
    assert merged[0].windows == 2


def test_merge_does_not_mutate_input_list():
    relations = [rel(description=["d1"]), rel(description=["d2"])]
    snapshot = list(relations)
    merge_relations(relations)
    assert relations == snapshot


# --- quote_found ----------------------------------------------------------------------------

TEXT = "Alice   joined\nACME Corp in 2019.\n\nShe leads the   data team."


@pytest.mark.parametrize(
    "quote",
    [
        "alice joined acme corp",
        "  Alice joined ACME corp in 2019.  ",
        "ALICE\tJOINED\n\nACME",
        "she leads the data team",
        "- she leads the data team",
        "* she leads the data team",
        "+ she leads the data team",
        "• she leads the data team",
        "1. she leads the data team",
        "2) she leads the data team",
    ],
)
def test_quote_found_normalizes_case_whitespace_and_list_marker(quote):
    assert quote_found(quote, TEXT, 8) is True


@pytest.mark.parametrize(
    "quote",
    ["alice left acme corp", "bob joined acme corp", "she leads the sales team"],
)
def test_quote_not_in_text_is_not_found(quote):
    assert quote_found(quote, TEXT, 8) is False


def test_quote_shorter_than_min_chars_is_never_found():
    assert quote_found("Alice", TEXT, 8) is False
    assert quote_found("Alice", TEXT, 5) is True
    assert quote_found("  alice   joined ", TEXT, 12) is True
    assert quote_found("  alice   joined ", TEXT, 13) is False


@pytest.mark.parametrize("min_chars", [0, -3, 1])
def test_empty_quote_is_never_found(min_chars):
    assert quote_found("", TEXT, min_chars) is False
    assert quote_found("   \n ", TEXT, min_chars) is False


def test_single_char_quote_found_with_min_chars_zero():
    assert quote_found("A", TEXT, 0) is True


# --- measure_relations ----------------------------------------------------------------------

SOURCE = "Alice works at Acme Corp.\nBob knows Alice from university."


def measure(relations, min_chars=8):
    return measure_relations(relations, SOURCE, 0.8, 0.4, min_chars)


def test_measure_all_quotes_found_gives_verbatim_confidence():
    r = rel(evidence=["alice works at acme", "- Bob knows Alice"])
    out = measure([r])
    assert out == [MeasuredRelation(relation=r, evidence_found=(True, True), confidence=0.8)]


def test_measure_any_quote_missing_gives_unverified_confidence():
    r = rel(evidence=["alice works at acme", "Carol founded Acme"])
    out = measure([r])
    assert out[0].evidence_found == (True, False)
    assert out[0].confidence == 0.4
    assert out[0].relation == r


def test_measure_no_evidence_gives_unverified_confidence():
    out = measure([rel()])
    assert out[0].evidence_found == ()
    assert out[0].confidence == 0.4


def test_measure_uses_min_evidence_chars():
    r = rel(evidence=["Alice"])
    assert measure([r], min_chars=8)[0].evidence_found == (False,)
    assert measure([r], min_chars=8)[0].confidence == 0.4
    assert measure([r], min_chars=3)[0].evidence_found == (True,)
    assert measure([r], min_chars=3)[0].confidence == 0.8


def test_measure_passes_confidences_through_and_keeps_order():
    first = rel("a", "b", evidence=["bob knows alice"])
    second = rel("c", "d", evidence=["nothing like this"])
    third = rel("e", "f")
    out = measure_relations([first, second, third], SOURCE, 0.95, 0.15, 8)
    assert [m.relation for m in out] == [first, second, third]
    assert [m.confidence for m in out] == [0.95, 0.15, 0.15]


def test_measure_empty_and_no_mutation():
    assert measure([]) == []
    relations = [rel(evidence=["alice works at acme"])]
    snapshot = list(relations)
    measure(relations)
    assert relations == snapshot
