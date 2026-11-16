"""graph-schema: the schema is data with a proof of use per item; each fact stored once.

Spec: docs/internal/specs/SPEC-graph-schema.md. No database in these tests.
"""

import inspect
import os
from pathlib import Path

import pytest

from memini.core import schema
from memini.infra import paths

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_FILE = ROOT / "config" / "graph_schema.yaml"


def _minimal() -> dict:
    return {
        "nodes": {
            "Document": {
                "rule": "a source file",
                "properties": {
                    "scope": {
                        "layer": "core",
                        "writer": "ingest.register",
                        "readers": ["query.*"],
                        "rule": "every node is scoped",
                    },
                    "id": {
                        "layer": "core",
                        "writer": "ingest.register",
                        "readers": ["query.*"],
                        "rule": "identity",
                    },
                },
                "indexes": [["scope", "id"]],
            },
            "Domain": {
                "rule": "a vocabulary value",
                "properties": {
                    "scope": {
                        "layer": "core",
                        "writer": "ingest.register",
                        "readers": ["query.*"],
                        "rule": "scoped",
                    },
                    "name": {
                        "layer": "core",
                        "writer": "ingest.register",
                        "readers": ["query.*"],
                        "rule": "vocabulary value",
                    },
                },
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


def test_the_committed_schema_loads_and_validates() -> None:
    s = schema.load(SCHEMA_FILE)
    assert "Entity" in s.nodes and "Community" in s.nodes
    assert "BELONGS_TO_COMMUNITY" in s.relationships
    assert schema.validate(s) == []


def test_every_property_needs_a_writer_and_a_reader_or_ingest_only() -> None:
    raw = _minimal()
    raw["nodes"]["Document"]["properties"]["orphan"] = {
        "layer": "core",
        "writer": "ingest.x",
        "readers": [],
        "rule": "r",
    }
    problems = schema.validate(schema.from_dict(raw))
    assert any("Document.orphan" in p and "reader" in p for p in problems)
    raw["nodes"]["Document"]["properties"]["orphan"]["ingest_only"] = "builds the X index"
    assert schema.validate(schema.from_dict(raw)) == []


def test_two_entries_holding_the_same_fact_are_rejected() -> None:
    raw = _minimal()
    raw["nodes"]["Document"]["properties"]["domain"] = {
        "layer": "core",
        "writer": "ingest.register",
        "readers": ["query.scope"],
        "rule": "copy",
        "same_fact_as": "IN_DOMAIN",
    }
    problems = schema.validate(schema.from_dict(raw))
    assert any("same fact" in p and "Document.domain" in p for p in problems)


def test_an_index_must_name_declared_properties() -> None:
    raw = _minimal()
    raw["nodes"]["Document"]["indexes"].append(["scope", "missing"])
    problems = schema.validate(schema.from_dict(raw))
    assert any("index" in p and "missing" in p for p in problems)


def test_relationship_endpoints_must_be_declared_nodes() -> None:
    raw = _minimal()
    raw["relationships"]["IN_DOMAIN"]["to"] = "Nowhere"
    problems = schema.validate(schema.from_dict(raw))
    assert any("IN_DOMAIN" in p and "Nowhere" in p for p in problems)


# P3: no domain words; the schema holds labels and property names only. The generic words are
# fixed here. Corpus-specific words (such as a company name) are private data: they come from
# the file named by MEMINI_PRIVATE_VOCAB (one word per line, # comments). Without that file no
# private case is generated (owner decision 2026-11-14, via vp-rnd: no corpus name in public code).
_GENERIC_VOCAB = ("product", "requirement", "architecture", "sdlc")


def _private_vocab() -> tuple[str, ...]:
    path = os.environ.get("MEMINI_PRIVATE_VOCAB", "").strip()
    if not path:
        return ()
    lines = (line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines())
    return tuple(line.lower() for line in lines if line and not line.startswith("#"))


def test_vocabulary_values_never_appear_in_the_schema() -> None:
    text = SCHEMA_FILE.read_text().lower()
    for word in _GENERIC_VOCAB:
        assert word not in text, word
    for number, word in enumerate(_private_vocab(), start=1):
        # the private word itself is never printed (test output may be public)
        assert word not in text, f"private vocabulary word #{number} appears in the schema"


def test_typed_names_match_the_schema_exactly() -> None:
    s = schema.load(SCHEMA_FILE)
    names = schema.names(s)
    assert set(names.labels) == set(s.nodes)
    assert set(names.rels) == set(s.relationships)
    assert (
        names.labels["Entity"] == "Entity"
        and names.rels["BELONGS_TO_COMMUNITY"] == "BELONGS_TO_COMMUNITY"
    )
    assert names.props["Entity"]["embedding"] == "embedding"


def test_index_statements_are_generated_from_the_schema() -> None:
    stmts = schema.index_statements(schema.from_dict(_minimal()))
    assert any("FOR (n:Document) ON (n.scope, n.id)" in st for st in stmts)
    assert all(st.startswith("CREATE INDEX") and "IF NOT EXISTS" in st for st in stmts)


# Owner decision 2026-11-13: the production Neo4j host lives only in the environment, never as a
# literal. The fixed non-local case uses an RFC 5737 TEST-NET-3 documentation address; the real
# production host is exercised only through MEMINI_PROD_NEO4J_HOST, and that case is not generated
# when the variable is unset (never skipped).
_LOCAL_GUARD_CASES: list[tuple[str, bool]] = [
    ("bolt://localhost:7688", True),
    ("neo4j://127.0.0.1:7687", True),
    ("bolt://203.0.113.10:7687", False),
    ("neo4j+s://db.example.com", False),
    ("bolt://localhost.evil.com:7687", False),
]
_PROD_HOST = os.environ.get("MEMINI_PROD_NEO4J_HOST", "").strip()
if _PROD_HOST:
    _LOCAL_GUARD_CASES.append((f"bolt://{_PROD_HOST}:7687", False))


@pytest.mark.parametrize("uri,ok", _LOCAL_GUARD_CASES)
def test_local_only_guard(uri: str, ok: bool) -> None:
    assert schema.is_local_uri(uri) is ok


def test_unlabeled_match_is_rejected_by_the_query_guard() -> None:
    assert schema.unlabeled_matches("MATCH (c {id:$id}) RETURN c") == ["(c {id:$id})"]
    assert (
        schema.unlabeled_matches("MATCH (c:Community {id:$id})-[:DOWN]->(e:Entity) RETURN e") == []
    )
    assert schema.unlabeled_matches("MATCH (a:Document)-[:HAS_UNIT]->(u) RETURN u") == ["(u)"]


def test_duplicate_detector_queries_cover_every_same_fact_pair() -> None:
    raw = _minimal()
    raw["nodes"]["Document"]["properties"]["domain"] = {
        "layer": "core",
        "writer": "w",
        "readers": ["r"],
        "rule": "copy",
        "same_fact_as": "IN_DOMAIN",
    }
    s = schema.from_dict(raw)
    queries = schema.duplicate_queries(s)
    assert len(queries) == 1
    assert "Document" in queries[0] and "domain" in queries[0] and "IN_DOMAIN" in queries[0]
    assert "count(" in queries[0].lower()


# ---- review 0a48a2a findings (spec: every property incl. relationship properties has proof;
# ---- rule required; same_fact_as pairs are guarded by the detector; config path not in core)


def test_relationship_properties_carry_proof_and_typed_names() -> None:
    s = schema.load(SCHEMA_FILE)
    rel = s.relationships["RELATED_TO"]
    assert set(rel.properties) >= {"kind", "type", "description"}
    assert rel.properties["kind"].writer and rel.properties["kind"].readers
    assert schema.names(s).rel_props["RELATED_TO"]["kind"] == "kind"
    for name in ("SIMILAR", "HIERARCHICAL_LINK"):
        assert s.relationships[name].properties, name


def test_relationship_property_without_proof_is_rejected() -> None:
    raw = _minimal()
    raw["relationships"]["IN_DOMAIN"]["properties"] = {
        "since": {"layer": "core", "writer": "", "readers": []}
    }
    problems = schema.validate(schema.from_dict(raw))
    assert any("IN_DOMAIN.since" in p for p in problems)


def test_rule_is_required_on_every_node_relationship_and_property() -> None:
    raw = _minimal()
    raw["nodes"]["Document"]["properties"]["id"]["rule"] = ""
    raw["relationships"]["IN_DOMAIN"]["rule"] = ""
    raw["nodes"]["Domain"]["rule"] = ""
    problems = schema.validate(schema.from_dict(raw))
    assert any("Document.id" in p and "rule" in p for p in problems)
    assert any(p.startswith("IN_DOMAIN:") and "rule" in p for p in problems)
    assert any(p.startswith("Domain:") and "rule" in p for p in problems)


def test_same_fact_pairs_are_declared_migrations_guarded_by_the_detector() -> None:
    # a declared pair is valid schema content (the detector guards it); an UNDECLARED
    # duplicate is what validate() must reject — so a pair needs `migrating_to`
    raw = _minimal()
    raw["nodes"]["Document"]["properties"]["domain"] = {
        "layer": "core",
        "writer": "ingest.register",
        "readers": ["query.scope"],
        "rule": "legacy copy",
        "same_fact_as": "IN_DOMAIN",
        "migrating_to": "IN_DOMAIN by module ingest-units",
    }
    s = schema.from_dict(raw)
    assert schema.validate(s) == []
    assert len(schema.duplicate_queries(s)) == 1
    del raw["nodes"]["Document"]["properties"]["domain"]["migrating_to"]
    assert any("same fact" in p for p in schema.validate(schema.from_dict(raw)))


def test_schema_path_comes_from_infra_configuration_not_from_core() -> None:
    src = inspect.getsource(schema)
    assert "parents[3]" not in src and '"config"' not in src
    assert paths.schema_file().name == "graph_schema.yaml" and paths.schema_file().exists()
