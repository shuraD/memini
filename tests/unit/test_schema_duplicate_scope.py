"""GS-03: duplicate-detector queries for node-property pairs.

Spec: SPEC-graph-schema, decision "A duplicate detector test"; BL-11 triple-storage
regression. The scope filter was removed (owner 2026-11-16: a corpus is kept apart by its own
database). No database in these tests.
"""

import pytest

from memini.core import schema


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


def test_relationship_pair_query_keeps_everything() -> None:
    s = schema.from_dict(_rel_pair_raw())
    queries = schema.duplicate_queries(s)
    assert len(queries) == 1
    q = queries[0]
    for part in ("Document", "n.domain IS NOT NULL", "IN_DOMAIN", "count(n)"):
        assert part in q


def test_property_pair_query_names_both_properties() -> None:
    s = schema.from_dict(_prop_pair_raw())
    queries = schema.duplicate_queries(s)
    assert len(queries) == 1
    q = queries[0]
    assert "n.domain" in _where_part(q) and "n.name" in q
    assert "IS NOT NULL" in q and "count(n)" in q and "Document" in q


@pytest.mark.parametrize("raw_factory", [_rel_pair_raw, _prop_pair_raw, _many_pairs_raw])
def test_no_query_carries_a_scope_parameter(raw_factory) -> None:
    s = schema.from_dict(raw_factory())
    assert all("$scope" not in q for q in schema.duplicate_queries(s))


def test_unscoped_queries_keep_todays_exact_text() -> None:
    s = schema.from_dict(_rel_pair_raw())
    assert schema.duplicate_queries(s) == [
        "MATCH (n:Document) WHERE n.domain IS NOT NULL "
        "AND EXISTS { (n)-[:IN_DOMAIN]-() } "
        "RETURN count(n) AS duplicates"
    ]


def test_several_pairs_keep_count_and_order() -> None:
    s = schema.from_dict(_many_pairs_raw())
    queries = schema.duplicate_queries(s)
    assert len(queries) == 3
    for q in queries:
        assert "IS NOT NULL" in _where_part(q) and "count(n)" in q
    assert "n.domain " in queries[0] and "IN_DOMAIN" in queries[0]
    assert "n.domain_name" in queries[1] and "n.name" in queries[1]
    assert "MATCH (n:Domain)" in queries[2] and "n.label" in queries[2] and "n.name" in queries[2]


def test_no_same_fact_pair_gives_no_query() -> None:
    s = schema.from_dict(_raw())
    assert schema.duplicate_queries(s) == []


@pytest.mark.parametrize("raw_factory", [_rel_pair_raw, _prop_pair_raw, _many_pairs_raw])
def test_every_query_names_a_label_in_each_match(raw_factory) -> None:
    s = schema.from_dict(raw_factory())
    queries = schema.duplicate_queries(s)
    assert queries
    for q in queries:
        assert schema.unlabeled_matches(q) == []


def test_bl11_triple_storage_migration_is_guarded() -> None:
    s = schema.from_dict(_rel_pair_raw())
    assert schema.validate(s) == []
    queries = schema.duplicate_queries(s)
    assert len(queries) == 1
    q = queries[0]
    for part in ("MATCH (n:Document)", "n.domain IS NOT NULL", "IN_DOMAIN", "count(n)"):
        assert part in q
