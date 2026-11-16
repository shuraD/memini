"""GS-04: `same_fact_as` on a relationship type and on a relationship property.

Spec: SPEC-graph-schema, decision "A duplicate detector test ... For two relationships, a
duplicate means the same two nodes (a, b) linked by both" (owner T-093, 2026-11-15).
A relationship's `same_fact_as` is accepted only as a declared migration (`migrating_to`),
its twin must be declared, and one detector query is generated per (source, target) pair.
No database in these tests.
"""

from memini.core import schema

MIGRATION = "R2 by module ingest-units"
R1_R2_QUERY = (
    "MATCH (a:Document)-[r:R1]->(b:Domain) WHERE EXISTS { (a)-[:R2]-(b) } "
    "RETURN count(r) AS duplicates"
)


def _prop(rule: str = "a rule", **extra: str) -> dict:
    return {
        "layer": "core",
        "writer": "ingest.register",
        "readers": ["query.*"],
        "rule": rule,
        **extra,
    }


def _rel(sources: object, targets: object, rule: str, **extra: object) -> dict:
    return {
        "from": sources,
        "to": targets,
        "layer": "core",
        "writer": "ingest.register",
        "readers": ["query.scope"],
        "rule": rule,
        **extra,
    }


def _raw(**r1_extra: object) -> dict:
    return {
        "nodes": {
            "Document": {
                "rule": "a source file",
                "properties": {
                    "scope": _prop("scoped"),
                    "id": _prop("identity"),
                    "domain": _prop("the document's domain value"),
                },
            },
            "Domain": {
                "rule": "a vocabulary value",
                "properties": {"scope": _prop("scoped"), "name": _prop("vocabulary value")},
            },
            "Entity": {
                "rule": "a named thing",
                "properties": {"scope": _prop("scoped"), "id": _prop("identity")},
            },
            "Label": {
                "rule": "a tag",
                "properties": {"scope": _prop("scoped"), "name": _prop("tag value")},
            },
        },
        "relationships": {
            "R1": _rel("Document", "Domain", "new link to a domain", **r1_extra),
            "R2": _rel("Document", "Domain", "old link to a domain"),
        },
    }


def _rel_queries(queries: list[str]) -> list[str]:
    return [q for q in queries if "[r:R1]" in q]


# --- from_dict ---------------------------------------------------------------------------


def test_from_dict_reads_relationship_migrating_to() -> None:
    s = schema.from_dict(_raw(same_fact_as="R2", migrating_to=MIGRATION))
    assert s.relationships["R1"].same_fact_as == "R2"
    assert s.relationships["R1"].migrating_to == MIGRATION


def test_from_dict_relationship_migrating_to_absent_is_none() -> None:
    s = schema.from_dict(_raw(same_fact_as="R2"))
    assert s.relationships["R1"].migrating_to is None
    assert s.relationships["R2"].migrating_to is None


def test_relationship_migrating_to_defaults_to_none() -> None:
    rel = schema.Relationship(
        sources=("Document",),
        targets=("Domain",),
        layer="core",
        writer="w",
        readers=("r",),
        rule="a rule",
    )
    assert rel.migrating_to is None
    assert rel.same_fact_as is None


# --- validate ----------------------------------------------------------------------------


def test_validate_accepts_relationship_twin_with_migration() -> None:
    s = schema.from_dict(_raw(same_fact_as="R2", migrating_to=MIGRATION))
    assert schema.validate(s) == []


def test_validate_rejects_relationship_twin_without_migration() -> None:
    s = schema.from_dict(_raw(same_fact_as="R2"))
    problems = schema.validate(s)
    assert any("same fact" in p and "R1" in p for p in problems), problems


def test_validate_reports_undeclared_relationship_target() -> None:
    s = schema.from_dict(_raw(same_fact_as="NOPE", migrating_to=MIGRATION))
    problems = schema.validate(s)
    assert "R1: same_fact_as names undeclared NOPE" in problems, problems


def test_validate_reports_undeclared_node_property_target() -> None:
    s = schema.from_dict(_raw(same_fact_as="Document.missing", migrating_to=MIGRATION))
    problems = schema.validate(s)
    assert any("Document.missing" in p and "undeclared" in p for p in problems), problems


def test_validate_reports_undeclared_label_in_property_target() -> None:
    s = schema.from_dict(_raw(same_fact_as="Nowhere.domain", migrating_to=MIGRATION))
    problems = schema.validate(s)
    assert "R1: same_fact_as names undeclared Nowhere.domain" in problems, problems


def test_validate_accepts_declared_node_property_target() -> None:
    s = schema.from_dict(_raw(same_fact_as="Document.domain", migrating_to=MIGRATION))
    assert schema.validate(s) == []


def test_validate_accepts_relationship_property_twin_with_migration() -> None:
    raw = _raw(properties={"p": _prop("weight", same_fact_as="R2", migrating_to=MIGRATION)})
    assert schema.validate(schema.from_dict(raw)) == []


def test_validate_rejects_relationship_property_twin_without_migration() -> None:
    raw = _raw(properties={"p": _prop("weight", same_fact_as="R2")})
    problems = schema.validate(schema.from_dict(raw))
    assert any("same fact" in p and "R1.p" in p for p in problems), problems


def test_validate_reports_undeclared_relationship_property_target() -> None:
    raw = _raw(properties={"p": _prop("weight", same_fact_as="NOPE", migrating_to=MIGRATION)})
    problems = schema.validate(schema.from_dict(raw))
    assert "R1.p: same_fact_as names undeclared NOPE" in problems, problems


# --- duplicate_queries -------------------------------------------------------------------


def test_relationship_twin_query_exact_without_scope() -> None:
    s = schema.from_dict(_raw(same_fact_as="R2", migrating_to=MIGRATION))
    queries = schema.duplicate_queries(s)
    assert queries == [R1_R2_QUERY]
    assert "(a)-[:R2]-(b)" in queries[0]
    assert "(a)-[:R2]-()" not in queries[0]
    assert "$scope" not in queries[0]


def test_one_query_per_source_label_in_declared_order() -> None:
    raw = _raw()
    raw["relationships"]["R1"] = _rel(
        ["Document", "Entity"],
        "Label",
        "tagged with",
        same_fact_as="R2",
        migrating_to=MIGRATION,
    )
    raw["relationships"]["R2"] = _rel(["Document", "Entity"], "Label", "old tag link")
    queries = _rel_queries(schema.duplicate_queries(schema.from_dict(raw)))
    assert queries == [
        "MATCH (a:Document)-[r:R1]->(b:Label) WHERE EXISTS { (a)-[:R2]-(b) } "
        "RETURN count(r) AS duplicates",
        "MATCH (a:Entity)-[r:R1]->(b:Label) WHERE EXISTS { (a)-[:R2]-(b) } "
        "RETURN count(r) AS duplicates",
    ]


def test_source_and_target_pairs_in_declared_order() -> None:
    raw = _raw()
    raw["relationships"]["R1"] = _rel(
        ["Entity", "Document"],
        ["Label", "Domain"],
        "linked",
        same_fact_as="R2",
        migrating_to=MIGRATION,
    )
    queries = _rel_queries(schema.duplicate_queries(schema.from_dict(raw)))
    heads = [q.split(" WHERE ")[0] for q in queries]
    assert heads == [
        "MATCH (a:Entity)-[r:R1]->(b:Label)",
        "MATCH (a:Entity)-[r:R1]->(b:Domain)",
        "MATCH (a:Document)-[r:R1]->(b:Label)",
        "MATCH (a:Document)-[r:R1]->(b:Domain)",
    ]


def test_property_twin_on_source_label() -> None:
    s = schema.from_dict(_raw(same_fact_as="Document.domain", migrating_to=MIGRATION))
    assert schema.duplicate_queries(s) == [
        "MATCH (a:Document)-[r:R1]->(b:Domain) WHERE a.domain IS NOT NULL "
        "RETURN count(r) AS duplicates"
    ]


def test_property_twin_on_target_label() -> None:
    raw = _raw()
    raw["relationships"]["R1"] = _rel(
        "Entity", "Document", "mentioned in", same_fact_as="Document.domain", migrating_to=MIGRATION
    )
    queries = schema.duplicate_queries(schema.from_dict(raw))
    assert queries == [
        "MATCH (a:Entity)-[r:R1]->(b:Document) WHERE b.domain IS NOT NULL "
        "RETURN count(r) AS duplicates"
    ]


def test_property_twin_label_at_neither_end_yields_no_query() -> None:
    raw = _raw()
    raw["relationships"]["R1"] = _rel(
        ["Document", "Entity"],
        "Label",
        "tagged with",
        same_fact_as="Document.domain",
        migrating_to=MIGRATION,
    )
    queries = _rel_queries(schema.duplicate_queries(schema.from_dict(raw)))
    assert queries == [
        "MATCH (a:Document)-[r:R1]->(b:Label) WHERE a.domain IS NOT NULL "
        "RETURN count(r) AS duplicates"
    ]
    assert not any("(a:Entity)" in q for q in queries)


def test_no_scope_gives_no_scope_parameter_anywhere() -> None:
    raw = _raw(same_fact_as="R2", migrating_to=MIGRATION)
    raw["relationships"]["R3"] = _rel(
        "Document", "Domain", "third link", same_fact_as="Document.domain", migrating_to=MIGRATION
    )
    queries = schema.duplicate_queries(schema.from_dict(raw))
    assert len(queries) == 2
    assert all("$scope" not in q for q in queries)


def _with_node_twin(raw: dict) -> dict:
    raw["nodes"]["Document"]["properties"]["domain"] = _prop(
        "legacy copy", same_fact_as="R2", migrating_to=MIGRATION
    )
    return raw


def test_node_property_queries_come_first_unchanged() -> None:
    node_only = schema.duplicate_queries(schema.from_dict(_with_node_twin(_raw())))
    assert len(node_only) == 1
    both = schema.duplicate_queries(
        schema.from_dict(_with_node_twin(_raw(same_fact_as="R2", migrating_to=MIGRATION)))
    )
    assert both == [*node_only, R1_R2_QUERY]


def test_relationships_follow_declaration_order() -> None:
    raw = _raw(same_fact_as="R2", migrating_to=MIGRATION)
    raw["relationships"]["R0"] = _rel(
        "Document", "Domain", "later link", same_fact_as="R2", migrating_to=MIGRATION
    )
    queries = schema.duplicate_queries(schema.from_dict(raw))
    assert [q.split(" WHERE ")[0] for q in queries] == [
        "MATCH (a:Document)-[r:R1]->(b:Domain)",
        "MATCH (a:Document)-[r:R0]->(b:Domain)",
    ]


def test_every_query_names_labels() -> None:
    raw = _with_node_twin(_raw(same_fact_as="R2", migrating_to=MIGRATION))
    raw["relationships"]["R3"] = _rel(
        ["Entity", "Document"],
        ["Document", "Label"],
        "linked",
        same_fact_as="Document.domain",
        migrating_to=MIGRATION,
    )
    queries = schema.duplicate_queries(schema.from_dict(raw))
    assert len(queries) >= 4
    for q in queries:
        assert schema.unlabeled_matches(q) == [], q
