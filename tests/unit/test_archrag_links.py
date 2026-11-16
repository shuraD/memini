"""AR-05: RELATED_TO link pairs of three strategies (sequence, shared label, cross-domain).

Spec: AR-05 (src/memini/core/archrag/links.py), card lines A17, A19, I72, I73. Pure functions:
no graph, no I/O. Thresholds come from a LinkSettings built from a config mapping.
"""

import copy
import dataclasses

import pytest

from memini.core.archrag.links import (
    EntityVector,
    Link,
    LinkSettings,
    cross_domain_links,
    label_links,
    linkable_labels,
    sequence_links,
    settings_from_config,
)

KIND = "kindvalqzx"


def _settings(**overrides: object) -> LinkSettings:
    values: dict = {
        "label_threshold": 0.7,
        "label_max_links": 5,
        "label_max_share": 1.0,
        "cross_domain_threshold": 0.6,
        "cross_domain_max_links": 5,
    }
    values.update(overrides)
    return LinkSettings(**values)


def _pairs(links: list[Link]) -> list[tuple[str, str]]:
    return [(link.source, link.target) for link in links]


# ---------------------------------------------------------------- dataclasses


def test_dataclasses_are_frozen_and_link_score_defaults_to_none() -> None:
    link = Link("a", "b", KIND)
    assert link.score is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        link.source = "x"  # type: ignore[misc]
    ev = EntityVector("e", "art", (1.0, 0.0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.id = "x"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        _settings().label_threshold = 0.1  # type: ignore[misc]


# ---------------------------------------------------------------- settings


def test_settings_from_config_reads_five_keys_and_ignores_extras() -> None:
    section = {
        "label_threshold": 0.7,
        "label_max_links": 5,
        "label_max_share": 0.2,
        "cross_domain_threshold": 0.6,
        "cross_domain_max_links": 4,
        "unrelated_key": "ignored",
    }
    got = settings_from_config(section)
    assert got == LinkSettings(
        label_threshold=0.7,
        label_max_links=5,
        label_max_share=0.2,
        cross_domain_threshold=0.6,
        cross_domain_max_links=4,
    )


def test_settings_from_config_converts_numbers() -> None:
    section = {
        "label_threshold": 1,
        "label_max_links": 3.0,
        "label_max_share": 0,
        "cross_domain_threshold": 1,
        "cross_domain_max_links": 2.0,
    }
    got = settings_from_config(section)
    assert isinstance(got.label_threshold, float) and got.label_threshold == 1.0
    assert isinstance(got.label_max_share, float) and got.label_max_share == 0.0
    assert isinstance(got.cross_domain_threshold, float) and got.cross_domain_threshold == 1.0
    assert isinstance(got.label_max_links, int) and got.label_max_links == 3
    assert isinstance(got.cross_domain_max_links, int) and got.cross_domain_max_links == 2


@pytest.mark.parametrize(
    "missing",
    [
        "label_threshold",
        "label_max_links",
        "label_max_share",
        "cross_domain_threshold",
        "cross_domain_max_links",
    ],
)
def test_settings_from_config_missing_key_raises_key_error(missing: str) -> None:
    section = {
        "label_threshold": 0.7,
        "label_max_links": 5,
        "label_max_share": 0.2,
        "cross_domain_threshold": 0.6,
        "cross_domain_max_links": 5,
    }
    del section[missing]
    with pytest.raises(KeyError):
        settings_from_config(section)


# ---------------------------------------------------------------- sequence


def test_sequence_links_consecutive_pairs_in_order() -> None:
    got = sequence_links(["a", "b", "c"], KIND)
    assert got == [Link("a", "b", KIND, None), Link("b", "c", KIND, None)]
    assert all(link.score is None and link.kind == KIND for link in got)


def test_sequence_links_kind_copied_as_is() -> None:
    got = sequence_links(["a", "b"], "Other Kind")
    assert got == [Link("a", "b", "Other Kind")]


def test_sequence_links_no_self_link() -> None:
    assert sequence_links(["a", "a", "b"], KIND) == [Link("a", "b", KIND)]


def test_sequence_links_empty_and_single() -> None:
    assert sequence_links([], KIND) == []
    assert sequence_links(["a"], KIND) == []


def test_sequence_links_accepts_tuple_and_does_not_mutate() -> None:
    ids = ["a", "b", "c"]
    before = list(ids)
    sequence_links(ids, KIND)
    assert ids == before
    assert sequence_links(("x", "y"), KIND) == [Link("x", "y", KIND)]


# ---------------------------------------------------------------- linkable labels


def _ten_artifacts() -> dict[str, set[str]]:
    labels: dict[str, set[str]] = {f"art{i}": set() for i in range(10)}
    labels["art0"] |= {"solo", "pair", "trio"}
    labels["art1"] |= {"pair", "trio"}
    labels["art2"] |= {"trio"}
    return labels


def test_linkable_labels_share_rules() -> None:
    got = linkable_labels(_ten_artifacts(), 0.2)
    # solo: 1 artifact -> never; pair: 2 <= 0.2*10 -> yes; trio: 3 > 2 -> no
    assert got == {"pair"}


def test_linkable_labels_single_artifact_never_linkable_even_at_full_share() -> None:
    got = linkable_labels({"a1": ["x", "y"], "a2": ["y"]}, 1.0)
    assert got == {"y"}


def test_linkable_labels_does_not_mutate() -> None:
    labels = _ten_artifacts()
    before = copy.deepcopy(labels)
    linkable_labels(labels, 0.2)
    assert labels == before


# ---------------------------------------------------------------- label links


def test_label_links_links_shared_linkable_label_once_with_cosine() -> None:
    entities = [
        EntityVector("e1", "a1", (1.0, 0.0)),
        EntityVector("e2", "a2", (3.0, 4.0)),
    ]
    labels = {"a1": {"L"}, "a2": {"L"}}
    got = label_links(entities, labels, _settings(label_threshold=0.6), KIND)
    assert len(got) == 1
    assert (got[0].source, got[0].target, got[0].kind) == ("e1", "e2", KIND)
    assert got[0].score == pytest.approx(0.6)


def test_label_links_requires_linkable_label() -> None:
    entities = [
        EntityVector("e1", "art0", (1.0, 0.0)),
        EntityVector("e2", "art1", (1.0, 0.0)),
        EntityVector("e3", "art2", (1.0, 0.0)),
        EntityVector("e4", "art3", (1.0, 0.0)),
    ]
    labels = {f"art{i}": set() for i in range(10)}
    labels["art0"] = {"common"}
    labels["art1"] = {"common"}
    labels["art2"] = {"common"}
    # "common" is on 3 of 10 artifacts > 0.2 share: not linkable; art3 has no label
    got = label_links(entities, labels, _settings(label_max_share=0.2), KIND)
    assert got == []


def test_label_links_only_between_artifacts_sharing_a_linkable_label() -> None:
    entities = [
        EntityVector("e1", "a1", (1.0, 0.0)),
        EntityVector("e2", "a2", (1.0, 0.0)),
        EntityVector("e3", "a3", (1.0, 0.0)),
    ]
    labels = {"a1": {"L"}, "a2": {"L"}, "a3": {"M"}, "a4": {"M"}}
    got = label_links(entities, labels, _settings(), KIND)
    assert _pairs(got) == [("e1", "e2")]


def test_label_links_never_same_artifact_or_self() -> None:
    entities = [
        EntityVector("e1", "a1", (1.0, 0.0)),
        EntityVector("e2", "a1", (1.0, 0.0)),
        EntityVector("e1", "a1", (1.0, 0.0)),
    ]
    labels = {"a1": {"L"}, "a2": {"L"}}
    assert label_links(entities, labels, _settings(), KIND) == []


def test_label_links_threshold_inclusive() -> None:
    entities = [
        EntityVector("e1", "a1", (1.0, 0.0)),
        EntityVector("e2", "a2", (3.0, 4.0)),
    ]
    labels = {"a1": {"L"}, "a2": {"L"}}
    at = label_links(entities, labels, _settings(label_threshold=0.6), KIND)
    assert _pairs(at) == [("e1", "e2")]
    below = label_links(entities, labels, _settings(label_threshold=0.6000001), KIND)
    assert below == []


def test_label_links_max_links_best_by_score_then_target_id() -> None:
    entities = [
        EntityVector("e0", "a0", (1.0, 0.0)),
        EntityVector("z1", "a1", (1.0, 0.0)),  # 1.0
        EntityVector("y2", "a2", (1.0, 0.0)),  # 1.0, ties with z1 -> y2 first
        EntityVector("x3", "a3", (3.0, 4.0)),  # 0.6
        EntityVector("w4", "a4", (0.0, 1.0)),  # 0.0 from e0
    ]
    labels = {f"a{i}": {"L"} for i in range(5)}
    settings = _settings(label_threshold=0.0, label_max_links=2)
    got = label_links(entities, labels, settings, KIND)
    first = [(link.source, link.target) for link in got if link.source == "e0"]
    assert first == [("e0", "y2"), ("e0", "z1")]
    assert got[0].source == "e0" and got[1].source == "e0"


def test_label_links_one_link_per_unordered_pair_first_entity_wins() -> None:
    entities = [
        EntityVector("b", "a1", (1.0, 0.0)),
        EntityVector("a", "a2", (1.0, 0.0)),
    ]
    labels = {"a1": {"L"}, "a2": {"L"}}
    got = label_links(entities, labels, _settings(), KIND)
    assert _pairs(got) == [("b", "a")]
    assert got[0].score == pytest.approx(1.0)
    assert got[0].kind == KIND


def test_label_links_emission_order_follows_input_order() -> None:
    entities = [
        EntityVector("c", "a3", (1.0, 0.0)),
        EntityVector("a", "a1", (1.0, 0.0)),
        EntityVector("b", "a2", (1.0, 0.0)),
    ]
    labels = {"a1": {"L"}, "a2": {"L"}, "a3": {"L"}}
    got = label_links(entities, labels, _settings(), KIND)
    assert _pairs(got) == [("c", "a"), ("c", "b"), ("a", "b")]


def test_label_links_zero_vector() -> None:
    entities = [
        EntityVector("e1", "a1", (0.0, 0.0)),
        EntityVector("e2", "a2", (1.0, 0.0)),
    ]
    labels = {"a1": {"L"}, "a2": {"L"}}
    assert label_links(entities, labels, _settings(label_threshold=0.1), KIND) == []
    got = label_links(entities, labels, _settings(label_threshold=0.0), KIND)
    assert _pairs(got) == [("e1", "e2")]
    assert got[0].score == 0.0


def test_label_links_does_not_mutate_inputs() -> None:
    entities = [
        EntityVector("e1", "a1", (1.0, 0.0)),
        EntityVector("e2", "a2", (1.0, 0.0)),
    ]
    labels = {"a1": {"L"}, "a2": {"L"}}
    ents_before = list(entities)
    labels_before = copy.deepcopy(labels)
    label_links(entities, labels, _settings(), KIND)
    assert entities == ents_before
    assert labels == labels_before


# ---------------------------------------------------------------- cross-domain


def test_cross_domain_links_different_domains_linked() -> None:
    entities = [
        EntityVector("e1", "a1", (1.0, 0.0)),
        EntityVector("e2", "a2", (3.0, 4.0)),
    ]
    domains = {"a1": "d1", "a2": "d2"}
    got = cross_domain_links(entities, domains, _settings(cross_domain_threshold=0.6), KIND)
    assert len(got) == 1
    assert (got[0].source, got[0].target, got[0].kind) == ("e1", "e2", KIND)
    assert got[0].score == pytest.approx(0.6)


def test_cross_domain_links_threshold_inclusive() -> None:
    entities = [
        EntityVector("e1", "a1", (1.0, 0.0)),
        EntityVector("e2", "a2", (3.0, 4.0)),
    ]
    domains = {"a1": "d1", "a2": "d2"}
    below = cross_domain_links(entities, domains, _settings(cross_domain_threshold=0.6000001), KIND)
    assert below == []


def test_cross_domain_links_same_domain_not_linked() -> None:
    entities = [
        EntityVector("e1", "a1", (1.0, 0.0)),
        EntityVector("e2", "a2", (1.0, 0.0)),
    ]
    domains = {"a1": "d1", "a2": "d1"}
    assert cross_domain_links(entities, domains, _settings(), KIND) == []


def test_cross_domain_links_same_artifact_not_linked() -> None:
    entities = [
        EntityVector("e1", "a1", (1.0, 0.0)),
        EntityVector("e2", "a1", (1.0, 0.0)),
    ]
    domains = {"a1": "d1"}
    assert cross_domain_links(entities, domains, _settings(), KIND) == []


def test_cross_domain_links_artifact_without_domain_never_linked() -> None:
    entities = [
        EntityVector("e1", "a1", (1.0, 0.0)),
        EntityVector("e2", "nodomain", (1.0, 0.0)),
        EntityVector("e3", "a3", (1.0, 0.0)),
    ]
    domains = {"a1": "d1", "a3": "d3"}
    got = cross_domain_links(entities, domains, _settings(cross_domain_threshold=0.0), KIND)
    assert _pairs(got) == [("e1", "e3")]


def test_cross_domain_links_max_links_and_ties() -> None:
    entities = [
        EntityVector("e0", "a0", (1.0, 0.0)),
        EntityVector("m", "a1", (1.0, 0.0)),
        EntityVector("k", "a2", (1.0, 0.0)),
        EntityVector("j", "a3", (3.0, 4.0)),
    ]
    domains = {"a0": "d0", "a1": "d1", "a2": "d2", "a3": "d3"}
    settings = _settings(cross_domain_threshold=0.5, cross_domain_max_links=1)
    got = cross_domain_links(entities, domains, settings, KIND)
    from_e0 = [link.target for link in got if link.source == "e0"]
    assert from_e0 == ["k"]
    assert got[0] == Link("e0", "k", KIND, got[0].score)
    assert got[0].score == pytest.approx(1.0)


def test_cross_domain_links_one_link_per_pair() -> None:
    entities = [
        EntityVector("y", "a1", (1.0, 0.0)),
        EntityVector("x", "a2", (1.0, 0.0)),
    ]
    domains = {"a1": "d1", "a2": "d2"}
    got = cross_domain_links(entities, domains, _settings(), KIND)
    assert _pairs(got) == [("y", "x")]


def test_cross_domain_links_zero_vector() -> None:
    entities = [
        EntityVector("e1", "a1", (1.0, 0.0)),
        EntityVector("e2", "a2", (0.0, 0.0)),
    ]
    domains = {"a1": "d1", "a2": "d2"}
    assert cross_domain_links(entities, domains, _settings(), KIND) == []
    got = cross_domain_links(entities, domains, _settings(cross_domain_threshold=-0.5), KIND)
    assert _pairs(got) == [("e1", "e2")]
    assert got[0].score == 0.0


def test_cross_domain_links_does_not_mutate_inputs() -> None:
    entities = [
        EntityVector("e1", "a1", (1.0, 0.0)),
        EntityVector("e2", "a2", (1.0, 0.0)),
    ]
    domains = {"a1": "d1", "a2": "d2"}
    ents_before = list(entities)
    domains_before = dict(domains)
    cross_domain_links(entities, domains, _settings(), KIND)
    assert entities == ents_before
    assert domains == domains_before
