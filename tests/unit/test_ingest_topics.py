"""Acceptance tests for IN-14: file-level topics, summary and derived labels.

Written from the IN-14 specification before the implementation exists.
"""

from __future__ import annotations

import copy

import pytest

from memini.core.ingest.topics import (
    DerivedLabels,
    LabelSource,
    derive_labels,
    file_summary,
    file_topics,
    normalize_topics,
    union_topics,
)

# ---------------------------------------------------------------- normalize_topics


def test_normalize_single_string_is_one_item_list() -> None:
    assert normalize_topics("Billing") == ["billing"]


@pytest.mark.parametrize("raw", [None, 5, {"topics": ["a"]}, 3.5])
def test_normalize_non_string_non_list_returns_empty(raw: object) -> None:
    assert normalize_topics(raw) == []


def test_normalize_accepts_tuple() -> None:
    assert normalize_topics(("Alpha", "Beta")) == ["alpha", "beta"]


def test_normalize_skips_non_string_items() -> None:
    assert normalize_topics(["Alpha", 3, None, {"x": 1}, ["y"], "Beta"]) == [
        "alpha",
        "beta",
    ]


def test_normalize_collapses_whitespace_strips_punctuation_lowercases() -> None:
    assert normalize_topics(["  Data   Model. "]) == ["data model"]


def test_normalize_strips_all_listed_surrounding_characters() -> None:
    raw = ['"Quoted"', "'single'", "`tick`", "(paren)", "[box]", "{brace}", "<angle>"]
    raw += ["#hash", "*star*", "ask?", "bang!", "semi;", "colon:", "comma,"]
    assert normalize_topics(raw) == [
        "quoted",
        "single",
        "tick",
        "paren",
        "box",
        "brace",
        "angle",
        "hash",
        "star",
        "ask",
        "bang",
        "semi",
        "colon",
        "comma",
    ]


def test_normalize_keeps_inner_punctuation() -> None:
    assert normalize_topics(["Data-Model"]) == ["data-model"]


def test_normalize_tabs_and_newlines_collapse() -> None:
    assert normalize_topics(["Data\t\n Model"]) == ["data model"]


def test_normalize_same_key_keeps_first() -> None:
    assert normalize_topics(["API Design", "api-design", "apidesign"]) == ["api design"]


def test_normalize_drops_empty_keys() -> None:
    assert normalize_topics(["!!!", "", "   ", "-", "Real"]) == ["real"]


def test_normalize_uses_vocabulary_spelling() -> None:
    assert normalize_topics(["api design"], vocabulary=["API-Design"]) == ["API-Design"]


def test_normalize_vocabulary_spelling_is_stripped() -> None:
    assert normalize_topics(["api design"], vocabulary=["  API-Design  "]) == ["API-Design"]


def test_normalize_topics_outside_vocabulary_keep_normalized_form() -> None:
    result = normalize_topics(
        ["api design", "  Data   Model. "], vocabulary=["API-Design", "Other"]
    )
    assert result == ["API-Design", "data model"]


def test_normalize_vocabulary_not_added_when_absent() -> None:
    assert normalize_topics(["alpha"], vocabulary=["Beta"]) == ["alpha"]


def test_normalize_vocabulary_accepts_generator() -> None:
    vocab = (v for v in ["API-Design"])
    assert normalize_topics(["apidesign"], vocabulary=vocab) == ["API-Design"]


def test_normalize_limit_applies_after_dedup() -> None:
    raw = ["a1", "A1", "b2", "c3", "d4"]
    assert normalize_topics(raw, limit=2) == ["a1", "b2"]
    assert normalize_topics(raw, limit=None) == ["a1", "b2", "c3", "d4"]
    assert normalize_topics(raw) == ["a1", "b2", "c3", "d4"]


def test_normalize_does_not_modify_input_and_is_repeatable() -> None:
    raw = ["  Data   Model. ", "API Design"]
    vocab = ["API-Design"]
    raw_copy, vocab_copy = list(raw), list(vocab)
    first = normalize_topics(raw, vocab, limit=5)
    second = normalize_topics(raw, vocab, limit=5)
    assert first == second == ["data model", "API-Design"]
    assert raw == raw_copy
    assert vocab == vocab_copy


# ---------------------------------------------------------------- union_topics


def test_union_orders_by_reply_count_then_first_appearance() -> None:
    per_reply = [["alpha", "beta"], ["gamma", "beta"], ["delta", "gamma", "beta"]]
    assert union_topics(per_reply) == ["beta", "gamma", "alpha", "delta"]


def test_union_repeat_within_reply_counts_once() -> None:
    per_reply = [["alpha", "alpha", "alpha"], ["beta"], ["beta"]]
    assert union_topics(per_reply) == ["beta", "alpha"]


def test_union_same_key_merges_keeping_first_spelling() -> None:
    per_reply = [["gamma"], ["API-Design"], ["api design", "gamma"], ["apidesign"]]
    assert union_topics(per_reply) == ["API-Design", "gamma"]


def test_union_limit() -> None:
    per_reply = [["a", "b", "c"], ["c", "b"], ["c"]]
    assert union_topics(per_reply, limit=2) == ["c", "b"]
    assert union_topics(per_reply, limit=None) == ["c", "b", "a"]


def test_union_empty() -> None:
    assert union_topics([]) == []
    assert union_topics([[], []]) == []


def test_union_does_not_modify_input_and_is_repeatable() -> None:
    per_reply = [["a", "b"], ["b"]]
    snapshot = copy.deepcopy(per_reply)
    assert union_topics(per_reply) == union_topics(per_reply) == ["b", "a"]
    assert per_reply == snapshot


# ---------------------------------------------------------------- file_topics


def test_file_topics_mixed_replies() -> None:
    replies: list[object] = [
        {"topics": ["Billing", "Invoices"]},
        {"topics": "invoices"},
        "not a dict",
        {"no": 1},
        {"topics": 5},
    ]
    assert file_topics(replies) == ["invoices", "billing"]


def test_file_topics_uses_vocabulary() -> None:
    replies: list[object] = [{"topics": ["api design"]}, {"topics": ["Other"]}]
    assert file_topics(replies, ["API-Design"]) == ["API-Design", "other"]


def test_file_topics_per_reply_limit_applies_before_union() -> None:
    replies: list[object] = [
        {"topics": ["one", "two", "three"]},
        {"topics": ["four", "three"]},
    ]
    # per-reply limit 1 keeps "one" and "four"; "three" is never counted
    assert file_topics(replies, per_reply_limit=1) == ["one", "four"]


def test_file_topics_file_limit() -> None:
    replies: list[object] = [
        {"topics": ["one", "two"]},
        {"topics": ["two", "three"]},
    ]
    assert file_topics(replies, file_limit=1) == ["two"]
    assert file_topics(replies) == ["two", "one", "three"]


def test_file_topics_no_usable_reply_returns_empty() -> None:
    assert file_topics([]) == []
    assert file_topics(["text", None, 3, {"no": 1}, {"topics": None}]) == []


def test_file_topics_does_not_modify_inputs() -> None:
    replies: list[object] = [{"topics": ["A", "B"]}, {"topics": "a"}]
    vocab = ["B"]
    snapshot = copy.deepcopy(replies)
    first = file_topics(replies, vocab)
    second = file_topics(replies, vocab)
    assert first == second == ["a", "B"]
    assert replies == snapshot
    assert vocab == ["B"]


# ---------------------------------------------------------------- file_summary


def test_file_summary_joins_stripped_unique_summaries_in_order() -> None:
    replies: list[object] = [
        {"summary": "  First part. "},
        {"summary": "Second part."},
        {"summary": "First part."},
        {"summary": "first part."},
    ]
    assert file_summary(replies) == "First part.\nSecond part.\nfirst part."


def test_file_summary_skips_unusable_replies() -> None:
    replies: list[object] = [
        "not a dict",
        None,
        {"topics": ["x"]},
        {"summary": 5},
        {"summary": None},
        {"summary": "   "},
        {"summary": ""},
        {"summary": "Only one."},
    ]
    assert file_summary(replies) == "Only one."


def test_file_summary_none_gives_empty_string() -> None:
    assert file_summary([]) == ""
    assert file_summary([{"summary": " "}, "x"]) == ""


def test_file_summary_repeatable_and_does_not_modify_input() -> None:
    replies: list[object] = [{"summary": " A "}, {"summary": "B"}]
    snapshot = copy.deepcopy(replies)
    assert file_summary(replies) == file_summary(replies) == "A\nB"
    assert replies == snapshot


# ---------------------------------------------------------------- derive_labels


def _assert_consistent(result: DerivedLabels) -> None:
    assert isinstance(result, DerivedLabels)
    assert isinstance(result.labels, tuple)
    assert set(result.sources) == set(result.labels)
    assert len(result.sources) == len(result.labels)
    assert len(set(result.labels)) == len(result.labels)


def test_label_source_members() -> None:
    names = {m.name for m in LabelSource}
    assert names == {"USER", "ROLE", "DOMAIN", "TOPIC", "WORK_UNIT"}
    assert isinstance(LabelSource.USER, str)


def test_derived_labels_is_frozen() -> None:
    result = derive_labels(["a"])
    with pytest.raises(AttributeError):
        result.labels = ()  # type: ignore[misc]


def test_existing_labels_without_source_are_user_labels() -> None:
    result = derive_labels(["Zeta", "Alpha"])
    assert result.labels == ("Zeta", "Alpha")
    assert result.sources == {"Zeta": LabelSource.USER, "Alpha": LabelSource.USER}
    _assert_consistent(result)


def test_existing_labels_with_user_source_are_kept() -> None:
    result = derive_labels(["Zeta", "Alpha"], {"Zeta": "user"})
    assert result.labels == ("Zeta", "Alpha")
    assert result.sources["Zeta"] == LabelSource.USER
    assert result.sources["Alpha"] == LabelSource.USER


def test_existing_derived_labels_are_dropped() -> None:
    result = derive_labels(
        ["Keep", "OldTopic", "OldRole"],
        {"OldTopic": "topic", "OldRole": "role", "Keep": "user"},
    )
    assert result.labels == ("Keep",)
    assert result.sources == {"Keep": LabelSource.USER}


def test_existing_label_with_unknown_source_value_is_dropped() -> None:
    result = derive_labels(["Odd"], {"Odd": "something-else"})
    assert result.labels == ()
    assert result.sources == {}


def test_existing_derived_label_is_rederived_if_still_supplied() -> None:
    result = derive_labels(
        ["Mine", "billing"],
        {"billing": "topic"},
        role="Engineer",
        topics=["billing"],
    )
    assert result.labels == ("Mine", "Engineer", "billing")
    assert result.sources == {
        "Mine": LabelSource.USER,
        "Engineer": LabelSource.ROLE,
        "billing": LabelSource.TOPIC,
    }


def test_derived_label_order_and_sources() -> None:
    result = derive_labels(
        ["Mine"],
        role="Engineer",
        domain="Finance",
        topics=["billing", "invoices"],
        work_unit="sprint-7",
        artifact_id="doc-1",
        work_unit_prefix="wu:",
    )
    assert result.labels == (
        "Mine",
        "Engineer",
        "Finance",
        "billing",
        "invoices",
        "wu:sprint-7",
    )
    assert result.sources == {
        "Mine": LabelSource.USER,
        "Engineer": LabelSource.ROLE,
        "Finance": LabelSource.DOMAIN,
        "billing": LabelSource.TOPIC,
        "invoices": LabelSource.TOPIC,
        "wu:sprint-7": LabelSource.WORK_UNIT,
    }
    _assert_consistent(result)


def test_work_unit_without_prefix() -> None:
    result = derive_labels([], work_unit="sprint-7", artifact_id="doc-1")
    assert result.labels == ("sprint-7",)
    assert result.sources == {"sprint-7": LabelSource.WORK_UNIT}


@pytest.mark.parametrize("work_unit", [None, "", "   ", "doc-1", "  doc-1  "])
def test_no_work_unit_label_when_missing_blank_or_artifact(
    work_unit: str | None,
) -> None:
    result = derive_labels(
        [], role="Engineer", work_unit=work_unit, artifact_id="doc-1", work_unit_prefix="wu:"
    )
    assert result.labels == ("Engineer",)
    assert LabelSource.WORK_UNIT not in result.sources.values()
    _assert_consistent(result)


def test_same_key_keeps_first_supplier() -> None:
    result = derive_labels(["Billing"], topics=["billing", "BILLING!"])
    assert result.labels == ("Billing",)
    assert result.sources == {"Billing": LabelSource.USER}


def test_same_key_between_role_and_topic_keeps_role() -> None:
    result = derive_labels([], role="Data Model", topics=["data-model", "other"])
    assert result.labels == ("Data Model", "other")
    assert result.sources == {
        "Data Model": LabelSource.ROLE,
        "other": LabelSource.TOPIC,
    }


def test_empty_and_none_candidates_skipped_and_stripped() -> None:
    result = derive_labels(
        ["  Mine  ", "", "   "],
        role="  Engineer ",
        domain="   ",
        topics=["", "  billing  "],
        work_unit=None,
    )
    assert result.labels == ("Mine", "Engineer", "billing")
    assert result.sources == {
        "Mine": LabelSource.USER,
        "Engineer": LabelSource.ROLE,
        "billing": LabelSource.TOPIC,
    }
    _assert_consistent(result)


def test_none_role_and_domain_skipped() -> None:
    result = derive_labels([], role=None, domain=None, topics=["t"])
    assert result.labels == ("t",)
    assert result.sources == {"t": LabelSource.TOPIC}


def test_empty_call() -> None:
    result = derive_labels([])
    assert result.labels == ()
    assert result.sources == {}


def test_derive_labels_repeatable_and_does_not_modify_inputs() -> None:
    existing = ["Mine", "old"]
    sources = {"old": "topic"}
    topics = ["billing", "old"]
    snapshot = (list(existing), dict(sources), list(topics))
    kwargs = {
        "role": "Engineer",
        "domain": "Finance",
        "work_unit": "sprint-7",
        "artifact_id": "doc-1",
        "work_unit_prefix": "wu:",
    }
    first = derive_labels(existing, sources, topics=topics, **kwargs)
    second = derive_labels(existing, sources, topics=topics, **kwargs)
    assert first == second
    assert first.labels == ("Mine", "Engineer", "Finance", "billing", "old", "wu:sprint-7")
    assert first.sources["old"] == LabelSource.TOPIC
    assert (existing, sources, topics) == snapshot
    _assert_consistent(first)
