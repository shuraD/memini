"""QA acceptance tests for the graph schema v2 (owner 2026-11-16).

Source: squads/graph-archrag/SCHEMA-TASK.md and SCHEMA-QA-TASK.md (memini-internal-mvp).
No database: statements are built and run through a recording runner.

INTERFACE (dev4 implements exactly this; every other name used below already exists)
--------------------------------------------------------------------------------------
memini.core.graph.types
  Statement gains a last field   scope: str | None = None
      (existing constructions stay valid).
  Statement.kind gains the value "delete": the kind of every hard-delete statement.
  PerfRecorder.record(name, seconds, rows, db_hits, *, scope: str | None = None) -> None

memini.core.graph.session.GraphSession
  run(stmt):
    - calls the recorder as record(stmt.name, seconds, rows, db_hits, scope=stmt.scope);
    - for stmt.kind == "delete", it first calls
      memini.infra.graph.guard.guard_from_env(self.uri).
      UnsafeDatabaseError propagates, and the runner is never called.

memini.core.graph.writes.WriteBuilder(schema, *, scope: str, clock: Callable[[], Any],
                                      batch_size: int = 1000)
  The key of a label is "id" when the label declares id, otherwise "name"
  (Project, Domain, RelationType).
  Every statement carries scope=<the builder's scope> and puts that scope in its params.
  Every value travels in params: the query text holds only labels, types, property names,
  parameter names and keywords.

  merge_nodes(label, rows) -> list[Statement]
      row: {<key>: non-empty str, "updated_by": non-empty str, "props": {declared: value}}
      One "UNWIND $rows ..." statement per batch (params["rows"] holds the batch).
      It MERGEs (n:<label> {scope: ..., <key>: ...}), with exactly those two keys in the map,
      and its text sets n.updated_by = ... and n.updated_at = ..., with updated_at taken from
      clock(). ValueError when a row has no (or an empty) updated_by or key.

  merge_relationships(rel, rows, *, from_label, to_label) -> list[Statement]
      row: {"from_id": str, "to_id": str, "props": {...}}. The ids are the endpoint keys
      (a name for a name-keyed label). Both endpoints are matched inside the scope:
      scope in the pattern map, or WHERE v.scope = ....
      Relationship rows carry no updated_by (audit fields are node fields).
      RELATED_TO: props["kind"] is required and must be one of RELATED_TO_KINDS;
      props["type"] is required when kind == "extracted". Otherwise ValueError.

  RELATED_TO_KINDS = ("extracted", "sequence", "label", "cross_domain", "shared")
      (module constant of memini.core.graph.writes)

  soft_delete(label, ids, *, updated_by) -> list[Statement]
      Sets status = 'deleted' (plus updated_by / updated_at) on the label's nodes with
      those keys in the scope. ValueError when the label declares no status.
  restore(label, ids, *, updated_by) -> list[Statement]
      The same, setting status = 'active'.
  Neither of these two statements contains DELETE, DETACH or REMOVE.

  hard_delete(label, ids, *, hard: bool = False) -> list[Statement]
      DETACH DELETE of the label's nodes with those keys in the scope.
  hard_delete_scope(*, hard: bool = False) -> list[Statement]
      DETACH DELETE of every node of the scope: one labeled statement per schema label.
  Both raise ValueError, and build nothing, unless hard is True.
  Their statements have kind "delete".

memini.infra.graph.perflog
  QueryPerfLog.record(name, seconds, rows, db_hits, *, scope: str | None = None)
  QueryRecord gains the field scope: str | None, and graph_query_log gains a scope column.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

import memini
from memini.core import schema
from memini.core.graph import writes
from memini.core.graph.checks import require_labeled
from memini.core.graph.session import GraphSession
from memini.core.graph.types import RunResult, Statement, UnlabeledMatchError
from memini.infra.graph.guard import UnsafeDatabaseError
from memini.infra.graph.perflog import QueryPerfLog
from memini.infra.llm_providers.store import Store

SCOPE = "scope-zz9"
NOW = "2026-11-16T09:00:00+00:00"
WRITER = "writer-xx3"
LOCAL_URI = "bolt://localhost:7687"
REMOTE_URI = "bolt://db.example.com:7687"
ENV_VARS = ("MEMINI_PROD_NEO4J_HOST", "MEMINI_NEO4J_ALLOWED_HOSTS")

COMMON = {"scope", "updated_by", "updated_at"}
NODES: dict[str, set[str]] = {
    "Project": {"name"},
    "Domain": {"name"},
    "Artifact": {"id", "type", "role", "content_hash", "summary", "labels", "status"},
    "Chunk": {"id", "order", "content", "line_start", "line_end", "embedding"},
    "Entity": {
        "id",
        "name",
        "type",
        "summary",
        "canonical_id",
        "references",
        "labels",
        "embedding",
        "embedding_code",
        "status",
    },
    "Community": {"id", "level", "summary", "embedding"},
    "RelationType": {"name"},
}
ID_LABELS = ("Artifact", "Chunk", "Entity", "Community")
NAME_LABELS = ("Project", "Domain", "RelationType")
# relationship -> (sources, targets, properties); SUPERSEDES endpoints are "same label".
RELS: dict[str, tuple[set[str] | None, set[str] | None, set[str]]] = {
    "HAS_DOMAIN": ({"Project"}, {"Domain"}, set()),
    "CONTAINS": ({"Domain"}, {"Artifact"}, set()),
    "HAS_CHUNK": ({"Artifact"}, {"Chunk"}, {"order"}),
    "HAS_ENTITY": ({"Chunk"}, {"Entity"}, set()),
    "RELATED_TO": (
        {"Entity", "Community"},
        {"Entity", "Community"},
        {"kind", "type", "description"},
    ),
    "SIMILAR": ({"Entity"}, {"Entity"}, {"score", "code_score"}),
    "CODE_LINK": ({"Entity"}, {"Entity"}, {"kind", "line"}),
    "SUPERSEDES": (None, None, set()),
    "BELONGS_TO_COMMUNITY": ({"Entity", "Community"}, {"Community"}, set()),
    "HIERARCHICAL_LINK": ({"Community"}, {"Community"}, {"similarity"}),
    "INTER_LAYER_LINK": ({"Community"}, {"Community", "Entity"}, set()),
}
RELATED_TO_KINDS = ("extracted", "sequence", "label", "cross_domain", "shared")
GONE = (
    "AccessContext",
    "Principal",
    "CAN_ACCESS",
    "AuditEvent",
    "graph_health",
    "purge",
    "wipe_scope",
)


class RecordingRunner:
    """Keeps the Cypher and params it receives; returns no rows."""

    def __init__(self, uri: str = LOCAL_URI) -> None:
        self.uri = uri
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(
        self,
        cypher: str,
        params: Mapping[str, Any],
        *,
        profile: bool = False,
        autocommit: bool = False,
    ) -> RunResult:
        del profile, autocommit
        self.calls.append((cypher, dict(params)))
        return RunResult(rows=[])


class ScopeRecorder:
    """Collects one (name, rows, scope) tuple per recorded run."""

    def __init__(self) -> None:
        self.records: list[tuple[str, int, str | None]] = []

    def record(
        self,
        name: str,
        seconds: float,
        rows: int,
        db_hits: int | None,
        scope: str | None = None,
    ) -> None:
        del seconds, db_hits
        self.records.append((name, rows, scope))


@pytest.fixture
def loaded() -> schema.Schema:
    return schema.load()


def _builder() -> writes.WriteBuilder:
    return writes.WriteBuilder(schema.load(), scope=SCOPE, clock=lambda: NOW, batch_size=2)


@pytest.fixture
def no_db_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _values(value: Any) -> Iterator[Any]:
    """Yield every scalar inside a params structure."""
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _values(item)
    elif isinstance(value, list | tuple | set):
        for item in value:
            yield from _values(item)
    else:
        yield value


def _carried(stmt: Statement, value: Any) -> bool:
    return value in list(_values(stmt.params))


def _map_keys(cypher: str, clause: str, label: str) -> list[set[str]]:
    """Return the key set of every `<clause> (v:<label> {...})` pattern map."""
    found = re.finditer(rf"{clause}\s*\(\w*:{label}\s*\{{([^}}]*)\}}\)", cypher)
    return [set(re.findall(r"(\w+)\s*:", m.group(1))) for m in found]


def _in_scope(cypher: str, label: str) -> bool:
    """Tell whether every pattern of ``label`` (at least one) is bound to the scope.

    A pattern is bound when its map has a ``scope`` key or ``WHERE v.scope = ...`` names it.
    """
    found = list(re.finditer(rf"\((\w*):{label}\s*(\{{[^}}]*\}})?\)", cypher))
    if not found:
        return False
    for m in found:
        var, pattern_map = m.group(1), m.group(2) or ""
        in_map = re.search(r"\bscope\s*:", pattern_map)
        in_where = var and re.search(rf"\b{var}\.scope\s*=", cypher)
        if not (in_map or in_where):
            return False
    return True


def _node_row(label: str, key: str, **props: Any) -> dict[str, Any]:
    return {"id" if label in ID_LABELS else "name": key, "updated_by": WRITER, "props": props}


# 1. schema contents


def test_schema_loads_valid_with_exactly_the_seven_labels(loaded: schema.Schema) -> None:
    assert set(loaded.nodes) == set(NODES)
    assert schema.validate(loaded) == []


def test_schema_has_exactly_the_eleven_relationship_types(loaded: schema.Schema) -> None:
    assert set(loaded.relationships) == set(RELS)


@pytest.mark.parametrize("label", sorted(NODES))
def test_each_label_declares_its_properties_and_the_common_fields(
    loaded: schema.Schema, label: str
) -> None:
    declared = set(schema.names(loaded).props[label])
    assert declared == NODES[label] | COMMON


@pytest.mark.parametrize("rel", sorted(RELS))
def test_each_relationship_declares_its_endpoints_and_properties(
    loaded: schema.Schema, rel: str
) -> None:
    sources, targets, props = RELS[rel]
    spec = loaded.relationships[rel]
    if sources is None or targets is None:
        assert spec.sources
        assert set(spec.sources) == set(spec.targets)
    else:
        assert set(spec.sources) == sources
        assert set(spec.targets) == targets
    assert set(schema.names(loaded).rel_props[rel]) == props


def test_labels_are_a_property_not_a_node(loaded: schema.Schema) -> None:
    assert "Label" not in loaded.nodes
    assert "TAGGED" not in loaded.relationships
    props = schema.names(loaded).props
    assert "labels" in props["Artifact"]
    assert "labels" in props["Entity"]


# 2. common fields


def test_status_lives_on_artifact_and_entity_with_three_values(loaded: schema.Schema) -> None:
    props = schema.names(loaded).props
    assert {label for label in NODES if "status" in props[label]} == {"Artifact", "Entity"}
    for label in ("Artifact", "Entity"):
        rule = loaded.nodes[label].properties["status"].rule
        for value in ("active", "superseded", "deleted"):
            assert re.search(rf"\b{value}\b", rule), (label, rule)


# 3. constraints and indexes


def _constraint_columns(statements: list[str], label: str) -> list[set[str]]:
    found = re.finditer(
        rf"CREATE CONSTRAINT .*?FOR \((\w+):{label}\) REQUIRE \(?([^)]*?)\)? IS UNIQUE",
        "\n".join(statements),
    )
    return [{col.strip().split(".")[-1] for col in m.group(2).split(",")} for m in found]


@pytest.mark.parametrize(
    ("label", "key"), [*((lb, "id") for lb in ID_LABELS), *((lb, "name") for lb in NAME_LABELS)]
)
def test_uniqueness_is_per_scope(loaded: schema.Schema, label: str, key: str) -> None:
    columns = _constraint_columns(schema.setup_statements(loaded), label)
    assert {"scope", key} in columns
    assert {key} not in columns


def test_the_four_vector_indexes(loaded: schema.Schema) -> None:
    text = "\n".join(schema.setup_statements(loaded, vector_dimensions=8))
    found = set(re.findall(r"CREATE VECTOR INDEX .*?FOR \(\w+:(\w+)\) ON \(\w+\.(\w+)\)", text))
    assert found == {
        ("Chunk", "embedding"),
        ("Entity", "embedding"),
        ("Entity", "embedding_code"),
        ("Community", "embedding"),
    }


# 4. writes


def test_batch_write_is_one_unwind_per_batch_merging_on_scope_and_id() -> None:
    builder = _builder()
    rows = [_node_row("Artifact", f"docs/a{i}.md", content_hash=f"h-{i}") for i in range(3)]
    stmts = builder.merge_nodes("Artifact", rows)
    assert [len(s.params["rows"]) for s in stmts] == [2, 1]
    assert len({s.cypher for s in stmts}) == 1
    cypher = stmts[0].cypher
    assert cypher.lstrip().startswith("UNWIND $rows")
    assert _map_keys(cypher, "MERGE", "Artifact") == [{"scope", "id"}]
    assert re.search(r"\.updated_by\s*=", cypher)
    assert re.search(r"\.updated_at\s*=", cypher)
    for stmt in stmts:
        assert stmt.scope == SCOPE
        assert _carried(stmt, SCOPE)
        assert _carried(stmt, NOW)
        assert _carried(stmt, WRITER)
        require_labeled(stmt.cypher)


@pytest.mark.parametrize("label", NAME_LABELS)
def test_name_keyed_labels_merge_on_scope_and_name(label: str) -> None:
    builder = _builder()
    (stmt,) = builder.merge_nodes(label, [_node_row(label, "billing")])
    assert _map_keys(stmt.cypher, "MERGE", label) == [{"scope", "name"}]
    assert _carried(stmt, "billing")


def test_values_travel_only_as_parameters() -> None:
    builder = _builder()
    row = _node_row("Entity", "ent-qq7", name="name-yy5", labels=["lab-ww4"])
    stmts = builder.merge_nodes("Entity", [row])
    stmts += builder.merge_relationships(
        "RELATED_TO",
        [
            {
                "from_id": "ent-qq7",
                "to_id": "ent-vv6",
                "props": {"kind": "extracted", "type": "uses-tt8"},
            }
        ],
        from_label="Entity",
        to_label="Entity",
    )
    for stmt in stmts:
        for value in (SCOPE, NOW, WRITER, "ent-qq7", "name-yy5", "lab-ww4", "uses-tt8"):
            assert value not in stmt.cypher


@pytest.mark.parametrize("updated_by", [None, ""])
def test_row_without_updated_by_is_refused(updated_by: str | None) -> None:
    builder = _builder()
    row: dict[str, Any] = {"id": "docs/a.md", "props": {}}
    if updated_by is not None:
        row["updated_by"] = updated_by
    with pytest.raises(ValueError, match="updated_by"):
        builder.merge_nodes("Artifact", [row])


# 5. RELATED_TO


def _related(
    builder: writes.WriteBuilder, props: dict[str, Any], label: str = "Entity"
) -> list[Statement]:
    return builder.merge_relationships(
        "RELATED_TO",
        [{"from_id": "a", "to_id": "b", "props": props}],
        from_label=label,
        to_label=label,
    )


def test_related_to_kinds_constant() -> None:
    assert tuple(writes.RELATED_TO_KINDS) == RELATED_TO_KINDS


@pytest.mark.parametrize("kind", RELATED_TO_KINDS)
def test_related_to_accepts_each_kind(kind: str) -> None:
    builder = _builder()
    props: dict[str, Any] = {"kind": kind}
    if kind == "extracted":
        props["type"] = "uses"
    (stmt,) = _related(builder, props)
    assert _carried(stmt, kind)
    assert _in_scope(stmt.cypher, "Entity")
    require_labeled(stmt.cypher)


def test_related_to_links_communities() -> None:
    builder = _builder()
    (stmt,) = _related(builder, {"kind": "shared"}, label="Community")
    assert _in_scope(stmt.cypher, "Community")


@pytest.mark.parametrize(
    "props",
    [{}, {"kind": "friend"}, {"kind": "EXTRACTED", "type": "uses"}, {"kind": "extracted"}],
    ids=["no-kind", "unknown-kind", "wrong-case", "extracted-without-type"],
)
def test_related_to_refuses_bad_kind_or_missing_type(props: dict[str, Any]) -> None:
    builder = _builder()
    with pytest.raises(ValueError):
        _related(builder, props)


# 6. delete


_NO_REMOVAL = re.compile(r"\b(DELETE|DETACH|REMOVE)\b", re.IGNORECASE)


@pytest.mark.parametrize(("method", "status"), [("soft_delete", "deleted"), ("restore", "active")])
@pytest.mark.parametrize("label", ["Artifact", "Entity"])
def test_soft_delete_and_restore_set_status_and_remove_nothing(
    method: str, status: str, label: str
) -> None:
    builder = _builder()
    stmts = getattr(builder, method)(label, ["k-1", "k-2"], updated_by=WRITER)
    assert stmts
    for stmt in stmts:
        assert not _NO_REMOVAL.search(stmt.cypher)
        assert status in stmt.cypher or _carried(stmt, status)
        assert re.search(r"\.status\s*=", stmt.cypher)
        assert re.search(r"\.updated_by\s*=", stmt.cypher)
        assert _in_scope(stmt.cypher, label)
        assert _carried(stmt, "k-1")
        assert _carried(stmt, SCOPE)
        assert "k-1" not in stmt.cypher
        require_labeled(stmt.cypher)


def test_soft_delete_needs_a_status_property() -> None:
    builder = _builder()
    with pytest.raises(ValueError):
        builder.soft_delete("Chunk", ["c-1"], updated_by=WRITER)


@pytest.mark.parametrize("hard", [None, False])
def test_hard_delete_without_hard_true_builds_nothing(hard: bool | None) -> None:
    builder = _builder()
    kwargs: dict[str, Any] = {} if hard is None else {"hard": hard}
    with pytest.raises(ValueError):
        builder.hard_delete("Artifact", ["docs/a.md"], **kwargs)
    with pytest.raises(ValueError):
        builder.hard_delete_scope(**kwargs)


def test_hard_delete_by_ids_detach_deletes_in_scope(no_db_env: None) -> None:
    builder = _builder()
    del no_db_env
    stmts = builder.hard_delete("Artifact", ["docs/a.md"], hard=True)
    assert stmts
    runner = RecordingRunner()
    GraphSession(runner).run_all(stmts)
    assert len(runner.calls) == len(stmts)
    for stmt in stmts:
        assert stmt.kind == "delete"
        assert "DETACH DELETE" in stmt.cypher
        assert _in_scope(stmt.cypher, "Artifact")
        assert _carried(stmt, "docs/a.md")
        assert "docs/a.md" not in stmt.cypher


def test_hard_delete_of_a_scope_covers_every_label(loaded: schema.Schema, no_db_env: None) -> None:
    builder = _builder()
    del no_db_env
    stmts = builder.hard_delete_scope(hard=True)
    runner = RecordingRunner()
    GraphSession(runner).run_all(stmts)
    assert len(runner.calls) == len(stmts)
    for label in loaded.nodes:
        assert any(_in_scope(s.cypher, label) and "DETACH DELETE" in s.cypher for s in stmts), label
    for stmt in stmts:
        assert stmt.kind == "delete"
        assert _carried(stmt, SCOPE)
        assert SCOPE not in stmt.cypher


@pytest.mark.parametrize("which", ["ids", "scope"])
def test_production_guard_refuses_a_hard_delete_on_a_remote_host(
    no_db_env: None, which: str
) -> None:
    builder = _builder()
    del no_db_env
    if which == "ids":
        stmts = builder.hard_delete("Entity", ["e-1"], hard=True)
    else:
        stmts = builder.hard_delete_scope(hard=True)
    runner = RecordingRunner(uri=REMOTE_URI)
    with pytest.raises(UnsafeDatabaseError):
        GraphSession(runner).run_all(stmts)
    assert runner.calls == []


# 7. no protection layer


def test_the_protection_layer_names_are_gone(loaded: schema.Schema) -> None:
    for module in (
        "memini.core.schema",
        "memini.core.graph",
        "memini.core.graph.writes",
        "memini.core.graph.session",
        "memini.core.graph.types",
        "memini.core.graph.checks",
        "memini.infra.graph.guard",
        "memini.infra.graph.neo4j_runner",
        "memini.infra.graph.perflog",
    ):
        mod = importlib.import_module(module)
        assert not [name for name in GONE if hasattr(mod, name)], module
    assert not set(GONE) & (set(loaded.nodes) | set(loaded.relationships))
    src = Path(memini.__file__).resolve().parent
    pattern = re.compile(rf"\b({'|'.join(GONE)})\b")
    hits = [
        f"{path.relative_to(src)}: {m.group(1)}"
        for path in sorted(src.rglob("*.py"))
        for m in pattern.finditer(path.read_text(encoding="utf-8"))
    ]
    assert hits == []


# 8. kept: labeled MATCH, timing per statement with its scope


def test_session_refuses_unlabeled_match_and_records_each_statement_with_scope() -> None:
    builder = _builder()
    runner = RecordingRunner()
    recorder = ScopeRecorder()
    session = GraphSession(runner, recorder=recorder)
    with pytest.raises(UnlabeledMatchError):
        session.run(Statement("bad", "MATCH (n) RETURN n", {}, "read", scope=SCOPE))
    assert runner.calls == []
    assert recorder.records == []
    stmts = builder.merge_nodes("Artifact", [_node_row("Artifact", f"a{i}") for i in range(3)])
    session.run_all(stmts)
    session.run(Statement("other", "MATCH (n:Chunk) RETURN n", {}, "read", scope="scope-b"))
    assert [scope for _, _, scope in recorder.records] == [SCOPE, SCOPE, "scope-b"]
    assert len(runner.calls) == 3


def test_perflog_keeps_the_scope_column(tmp_path: Path) -> None:
    log = QueryPerfLog(Store(tmp_path / "perf.sqlite"), clock=lambda: 1.0)
    session = GraphSession(RecordingRunner(), recorder=log)
    session.run(Statement("q", "MATCH (n:Chunk) RETURN n", {}, "read", scope=SCOPE))
    log.record("direct", 0.5, 2, None, scope="scope-b")
    assert [(r.name, r.scope) for r in log.rows()] == [("q", SCOPE), ("direct", "scope-b")]
