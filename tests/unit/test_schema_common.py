"""GS-01: a `common` section of the schema, declared once and applied to every label and type.

Spec: SPEC-graph-schema, decision "Audit fields on every node ... declared once in the schema
for all labels and relationship types (a `common` section)". No database in these tests.
"""

from pathlib import Path

import pytest

from memini.core import schema

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_FILE = ROOT / "config" / "graph_schema.yaml"

COMMON_NODE_FIELDS = (
    "scope",
    "status",
    "created_at",
    "updated_at",
    "updated_by",
    "deleted_at",
    "deleted_by",
)


def _prop(rule: str = "a rule", writer: str = "ingest.register") -> dict:
    return {"layer": "core", "writer": writer, "readers": ["query.*"], "rule": rule}


def _raw() -> dict:
    return {
        "common": {
            "nodes": {name: _prop(f"audit {name}") for name in COMMON_NODE_FIELDS},
            "relationships": {"created_at": _prop("when the link was made")},
        },
        "nodes": {
            "Document": {
                "rule": "a source file",
                "properties": {"id": _prop("identity")},
                "indexes": [["scope", "id"]],
            },
            "Domain": {
                "rule": "a vocabulary value",
                "properties": {"name": _prop("vocabulary value")},
                "indexes": [["scope", "status"]],
            },
        },
        "relationships": {
            "IN_DOMAIN": {
                "from": "Document",
                "to": "Domain",
                "layer": "core",
                "writer": "ingest.register",
                "readers": ["query.*"],
                "rule": "a document belongs to a domain",
            },
            "CITES": {
                "from": "Document",
                "to": "Document",
                "layer": "core",
                "writer": "ingest.register",
                "readers": ["query.*"],
                "rule": "a document cites another",
                "properties": {"weight": _prop("how strongly")},
            },
        },
    }


# --- from_dict -------------------------------------------------------------------------


def test_from_dict_reads_common_as_property_objects():
    raw = _raw()
    raw["common"]["nodes"]["updated_by"] = {
        "layer": "infra",
        "writer": "graph.write",
        "readers": ["audit.view", "query.history"],
        "rule": "who changed it",
    }
    s = schema.from_dict(raw)
    assert set(s.common.nodes) == set(COMMON_NODE_FIELDS)
    assert set(s.common.relationships) == {"created_at"}
    p = s.common.nodes["updated_by"]
    assert isinstance(p, schema.Property)
    assert p.layer == "infra"
    assert p.writer == "graph.write"
    assert p.readers == ("audit.view", "query.history")
    assert isinstance(p.readers, tuple)
    assert p.rule == "who changed it"
    r = s.common.relationships["created_at"]
    assert isinstance(r, schema.Property)
    assert r.rule == "when the link was made"
    assert r.readers == ("query.*",)


def test_from_dict_without_common_gives_empty_common():
    raw = _raw()
    del raw["common"]
    s = schema.from_dict(raw)
    assert s.common.nodes == {}
    assert s.common.relationships == {}


def test_schema_built_directly_has_empty_common():
    s = schema.Schema()
    assert isinstance(s.common, schema.Common)
    assert s.common.nodes == {}
    assert s.common.relationships == {}


def test_common_default_is_empty():
    c = schema.Common()
    assert c.nodes == {}
    assert c.relationships == {}


def test_common_without_nodes_key_keeps_relationships():
    raw = _raw()
    del raw["common"]["nodes"]
    s = schema.from_dict(raw)
    assert s.common.nodes == {}
    assert set(s.common.relationships) == {"created_at"}
    assert isinstance(s.common.relationships["created_at"], schema.Property)


def test_common_without_relationships_key_keeps_nodes():
    raw = _raw()
    del raw["common"]["relationships"]
    s = schema.from_dict(raw)
    assert s.common.relationships == {}
    assert set(s.common.nodes) == set(COMMON_NODE_FIELDS)


# --- node_properties / rel_properties ---------------------------------------------------


def test_node_properties_merges_common_with_own_fields():
    s = schema.from_dict(_raw())
    doc = schema.node_properties(s, "Document")
    dom = schema.node_properties(s, "Domain")
    assert set(doc) == {*COMMON_NODE_FIELDS, "id"}
    assert set(dom) == {*COMMON_NODE_FIELDS, "name"}
    assert all(isinstance(p, schema.Property) for p in (*doc.values(), *dom.values()))
    assert doc["id"].rule == "identity"
    assert dom["updated_by"].rule == "audit updated_by"


def test_rel_properties_merges_common_with_own_fields():
    s = schema.from_dict(_raw())
    in_domain = schema.rel_properties(s, "IN_DOMAIN")
    cites = schema.rel_properties(s, "CITES")
    assert set(in_domain) == {"created_at"}
    assert set(cites) == {"created_at", "weight"}
    assert cites["weight"].rule == "how strongly"
    assert all(isinstance(p, schema.Property) for p in (*in_domain.values(), *cites.values()))


def test_node_properties_unknown_label_raises_keyerror():
    s = schema.from_dict(_raw())
    with pytest.raises(KeyError):
        schema.node_properties(s, "Nope")


def test_rel_properties_unknown_type_raises_keyerror():
    s = schema.from_dict(_raw())
    with pytest.raises(KeyError):
        schema.rel_properties(s, "NOPE")


def test_merge_does_not_mutate_schema():
    s = schema.from_dict(_raw())
    schema.node_properties(s, "Document")
    schema.rel_properties(s, "CITES")
    assert set(s.nodes["Document"].properties) == {"id"}
    assert set(s.relationships["CITES"].properties) == {"weight"}
    assert set(s.common.nodes) == set(COMMON_NODE_FIELDS)


# --- names --------------------------------------------------------------------------------


def test_names_include_common_fields():
    n = schema.names(schema.from_dict(_raw()))
    for label in ("Document", "Domain"):
        assert n.props[label]["updated_by"] == "updated_by"
        for f in COMMON_NODE_FIELDS:
            assert n.props[label][f] == f
    assert n.props["Document"]["id"] == "id"
    assert n.props["Domain"]["name"] == "name"
    for rel in ("IN_DOMAIN", "CITES"):
        assert n.rel_props[rel]["created_at"] == "created_at"
    assert n.rel_props["CITES"]["weight"] == "weight"


# --- validate -------------------------------------------------------------------------------


def test_valid_schema_with_common_has_no_problems():
    assert schema.validate(schema.from_dict(_raw())) == []


def test_label_redeclaring_common_field_is_a_problem():
    raw = _raw()
    raw["nodes"]["Document"]["properties"]["created_at"] = _prop("again")
    problems = schema.validate(schema.from_dict(raw))
    assert any("Document.created_at" in p and "declared in common" in p for p in problems)


def test_relationship_redeclaring_common_field_is_a_problem():
    raw = _raw()
    raw["relationships"]["CITES"]["properties"]["created_at"] = _prop("again")
    problems = schema.validate(schema.from_dict(raw))
    assert any("CITES.created_at" in p and "declared in common" in p for p in problems)


def test_redeclared_problem_text():
    raw = _raw()
    raw["nodes"]["Domain"]["properties"]["status"] = _prop("again")
    problems = schema.validate(schema.from_dict(raw))
    assert "Domain.status: declared in common (declare it once)" in problems


def test_common_node_property_without_writer_is_a_problem():
    raw = _raw()
    raw["common"]["nodes"]["deleted_by"]["writer"] = ""
    problems = schema.validate(schema.from_dict(raw))
    assert any(p.startswith("common.nodes.deleted_by") for p in problems)


def test_common_relationship_property_without_reader_is_a_problem():
    raw = _raw()
    del raw["common"]["relationships"]["created_at"]["readers"]
    problems = schema.validate(schema.from_dict(raw))
    assert any(p.startswith("common.relationships.created_at") for p in problems)


def test_common_relationship_property_with_ingest_only_needs_no_reader():
    raw = _raw()
    spec = raw["common"]["relationships"]["created_at"]
    del spec["readers"]
    spec["ingest_only"] = "kept for the audit trail"
    assert schema.validate(schema.from_dict(raw)) == []


def test_common_property_bad_layer_and_no_rule_are_problems():
    raw = _raw()
    raw["common"]["nodes"]["scope"]["layer"] = "nowhere"
    raw["common"]["nodes"]["status"]["rule"] = ""
    problems = schema.validate(schema.from_dict(raw))
    assert any(p.startswith("common.nodes.scope") and "layer" in p for p in problems)
    assert any(p.startswith("common.nodes.status") and "rule" in p for p in problems)


def test_index_on_common_fields_is_accepted():
    raw = _raw()
    raw["nodes"]["Domain"]["indexes"] = [["scope", "status"]]
    problems = schema.validate(schema.from_dict(raw))
    assert not any("index" in p for p in problems)
    assert problems == []


def test_index_on_undeclared_field_is_still_a_problem():
    raw = _raw()
    raw["nodes"]["Domain"]["indexes"] = [["scope", "nowhere"]]
    problems = schema.validate(schema.from_dict(raw))
    assert any("Domain" in p and "nowhere" in p for p in problems)
    assert all("nowhere" in p for p in problems)


def test_vector_index_on_common_field_is_accepted():
    raw = _raw()
    raw["common"]["nodes"]["embedding"] = _prop("semantic search")
    raw["nodes"]["Document"]["vector_indexes"] = ["embedding"]
    assert schema.validate(schema.from_dict(raw)) == []


def test_vector_index_on_undeclared_field_is_still_a_problem():
    raw = _raw()
    raw["nodes"]["Document"]["vector_indexes"] = ["nowhere"]
    problems = schema.validate(schema.from_dict(raw))
    assert any("Document" in p and "nowhere" in p for p in problems)


def test_common_field_not_visible_to_other_section():
    # a common relationship field does not make a node index valid
    raw = _raw()
    raw["common"]["relationships"]["weight_rel"] = _prop("rel only")
    raw["nodes"]["Document"]["indexes"] = [["weight_rel"]]
    problems = schema.validate(schema.from_dict(raw))
    assert any("Document" in p and "weight_rel" in p for p in problems)


# --- committed schema ------------------------------------------------------------------------


def test_committed_schema_without_common_stays_valid():
    s = schema.load(SCHEMA_FILE)
    assert schema.validate(s) == []
    assert isinstance(s.common, schema.Common)
