"""GS-03: duplicate-detector queries restricted to one scope through the bound $scope parameter.

Spec: SPEC-graph-schema, decisions "A duplicate detector test" and isolation by scope (every
schema query filters by scope); BL-11 triple-storage regression. No database in these tests.
"""

import pytest

from memini.core import schema

SCOPE = "corpus-alpha-7"
SCOPE_FILTER = "n.scope = $scope"
HOSTILE_SCOPES = ("x' OR 1=1 //", "a}) DETACH DELETE n //", 'q" RETURN 1 //', "$scope")


def _prop(rule: str = "a rule", **extra: str) -> dict:
    return {
        "layer": "core",
        "writer": "ingest.register",
        "readers": ["query.*"],
        "rule": rule,
        **extra,
    }


def _raw() -> dict:
    return {
        "nodes": {
            "Document": {
                "rule": "a source file",
                "properties": {"scope": _prop("scoped"), "id": _prop("identity")},
                "indexes": [["scope", "id"]],
            },
            "Domain": {
                "rule": "a vocabulary value",
                "properties": {"scope": _prop("scoped"), "name": _prop("vocabulary value")},
                "indexes": [["scope", "name"]],
            },
        },
        "relationships": {
            "IN_DOMAIN": {
                "from": "Document",
                "to": "Domain",
                "layer": "core",
                "writer": "ingest.register",
                "readers": ["query.scope"],
                "rule": "document belongs to one domain",
            },
        },
    }


def _rel_pair_raw() -> dict:
    raw = _raw()
    raw["nodes"]["Document"]["properties"]["domain"] = _prop(
        "legacy copy",
        same_fact_as="IN_DOMAIN",
        migrating_to="IN_DOMAIN by module ingest-units",
    )
    return raw


def _prop_pair_raw() -> dict:
    raw = _raw()
    raw["nodes"]["Document"]["properties"]["domain"] = _prop(
        "legacy copy",
        same_fact_as="Domain.name",
        migrating_to="Domain.name by module ingest-units",
    )
    return raw


def _many_pairs_raw() -> dict:
    raw = _raw()
    raw["nodes"]["Document"]["properties"]["domain"] = _prop(
        "legacy copy", same_fact_as="IN_DOMAIN", migrating_to="IN_DOMAIN"
    )
    raw["nodes"]["Document"]["properties"]["domain_name"] = _prop(
        "legacy copy", same_fact_as="Domain.name", migrating_to="Domain.name"
    )
    raw["nodes"]["Domain"]["properties"]["label"] = _prop(
        "legacy copy", same_fact_as="Domain.name", migrating_to="Domain.name"
    )
    return raw


def _where_part(query: str) -> str:
    assert "WHERE" in query
    return query.split("WHERE", 1)[1]


def test_relationship_pair_scoped_query_keeps_everything_and_adds_the_scope_filter() -> None:
    s = schema.from_dict(_rel_pair_raw())
    queries = schema.duplicate_queries(s, scope=SCOPE)
    assert len(queries) == 1
    q = queries[0]
    assert SCOPE_FILTER in q
    assert SCOPE_FILTER in _where_part(q)
    for part in ("Document", "n.domain IS NOT NULL", "IN_DOMAIN", "count(n)"):
        assert part in q
    assert SCOPE not in q


def test_property_pair_scoped_query_names_both_properties_and_not_the_value() -> None:
    s = schema.from_dict(_prop_pair_raw())
    queries = schema.duplicate_queries(s, scope=SCOPE)
    assert len(queries) == 1
    q = queries[0]
    assert SCOPE_FILTER in _where_part(q)
    assert "n.domain" in q and "n.name" in q
    assert "IS NOT NULL" in q and "count(n)" in q and "Document" in q
    assert SCOPE not in q


@pytest.mark.parametrize("raw_factory", [_rel_pair_raw, _prop_pair_raw, _many_pairs_raw])
def test_explicit_none_scope_equals_todays_queries(raw_factory) -> None:
    s = schema.from_dict(raw_factory())
    unscoped = schema.duplicate_queries(s)
    assert schema.duplicate_queries(s, scope=None) == unscoped
    assert schema.duplicate_queries(s, None) == unscoped
    assert all("$scope" not in q for q in unscoped)


def test_unscoped_queries_keep_todays_exact_text() -> None:
    s = schema.from_dict(_rel_pair_raw())
    assert schema.duplicate_queries(s) == [
        "MATCH (n:Document) WHERE n.domain IS NOT NULL "
        "AND EXISTS { (n)-[:IN_DOMAIN]-() } "
        "RETURN count(n) AS duplicates"
    ]


@pytest.mark.parametrize("raw_factory", [_rel_pair_raw, _prop_pair_raw, _many_pairs_raw])
def test_hostile_scope_values_never_reach_the_query_text(raw_factory) -> None:
    s = schema.from_dict(raw_factory())
    reference = schema.duplicate_queries(s, scope=SCOPE)
    assert reference
    for hostile in HOSTILE_SCOPES:
        queries = schema.duplicate_queries(s, scope=hostile)
        assert queries == reference
        if hostile != "$scope":
            assert all(hostile not in q for q in queries)


def test_several_pairs_keep_count_and_order_and_all_are_scoped() -> None:
    s = schema.from_dict(_many_pairs_raw())
    unscoped = schema.duplicate_queries(s)
    scoped = schema.duplicate_queries(s, scope=SCOPE)
    assert len(unscoped) == 3
    assert len(scoped) == len(unscoped)
    for plain, q in zip(unscoped, scoped, strict=True):
        assert SCOPE_FILTER in _where_part(q)
        assert SCOPE not in q
        # same pair, same order: the label and the property of each unscoped query are kept
        assert plain.split(" WHERE", 1)[0] == q.split(" WHERE", 1)[0]
        assert "IS NOT NULL" in q and "count(n)" in q
    assert "n.domain " in scoped[0] and "IN_DOMAIN" in scoped[0]
    assert "n.domain_name" in scoped[1] and "n.name" in scoped[1]
    assert "MATCH (n:Domain)" in scoped[2] and "n.label" in scoped[2] and "n.name" in scoped[2]


def test_no_same_fact_pair_gives_no_scoped_query() -> None:
    s = schema.from_dict(_raw())
    assert schema.duplicate_queries(s, scope=SCOPE) == []
    assert schema.duplicate_queries(s) == []


@pytest.mark.parametrize("raw_factory", [_rel_pair_raw, _prop_pair_raw, _many_pairs_raw])
def test_every_scoped_query_still_names_a_label_in_each_match(raw_factory) -> None:
    s = schema.from_dict(raw_factory())
    queries = schema.duplicate_queries(s, scope=SCOPE)
    assert queries
    for q in queries:
        assert schema.unlabeled_matches(q) == []


def test_bl11_triple_storage_migration_is_guarded_per_scope() -> None:
    s = schema.from_dict(_rel_pair_raw())
    assert schema.validate(s) == []
    queries = schema.duplicate_queries(s, scope=SCOPE)
    assert len(queries) == 1
    q = queries[0]
    for part in ("MATCH (n:Document)", "n.domain IS NOT NULL", "IN_DOMAIN", "count(n)"):
        assert part in q
    assert SCOPE_FILTER in _where_part(q)
    assert SCOPE not in q
