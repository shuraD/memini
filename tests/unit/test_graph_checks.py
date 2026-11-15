"""Acceptance tests for GS-06: labeled-MATCH and audit-field checks of memini.core.graph.checks."""

from __future__ import annotations

import pytest

from memini.core import graph
from memini.core.graph import checks
from memini.core.graph.types import AuditFieldError, GraphError, UnlabeledMatchError

MERGE_FULL = (
    "MERGE (n:Unit {scope:$scope, id:row.id}) "
    "ON CREATE SET n.created_at = datetime($now) "
    "SET n.updated_at = datetime($now), n.updated_by = $actor"
)
MERGE_NO_ON_CREATE = (
    "MERGE (n:Unit {scope:$scope, id:row.id}) "
    "SET n.updated_at = datetime($now), n.updated_by = $actor"
)
MERGE_NO_FINAL_SET = (
    "MERGE (n:Unit {scope:$scope, id:row.id}) ON CREATE SET n.created_at = datetime($now)"
)


def _unlabeled_message(cypher: str) -> str:
    with pytest.raises(UnlabeledMatchError) as excinfo:
        checks.require_labeled(cypher)
    return str(excinfo.value)


def _names(problems: list[str], field: str) -> bool:
    return any(field in problem for problem in problems)


# ---------------------------------------------------------------- require_labeled


def test_unlabeled_match_with_properties_is_rejected_regression_bl39() -> None:
    message = _unlabeled_message("MATCH (c {id:$id}) RETURN c")
    assert "(c {id:$id})" in message


def test_unlabeled_target_of_relationship_is_rejected() -> None:
    message = _unlabeled_message("MATCH (a:Document)-[:HAS_UNIT]->(u) RETURN u")
    assert "(u)" in message


def test_every_unlabeled_pattern_is_listed() -> None:
    message = _unlabeled_message("MATCH (a)-[:R]->(b) RETURN b")
    assert "(a)" in message
    assert "(b)" in message


def test_variable_bound_with_label_in_earlier_match_is_accepted() -> None:
    cypher = "MATCH (d:Document {id:$id}) MATCH (d)-[:HAS_UNIT]->(u:Unit) RETURN u"
    assert checks.require_labeled(cypher) is None


def test_variable_bound_with_label_across_with_is_accepted() -> None:
    cypher = "MATCH (d:Document {id:$id}) WITH d MATCH (d)-[:HAS_UNIT]->(u:Unit) RETURN u"
    assert checks.require_labeled(cypher) is None


def test_all_labeled_patterns_are_accepted() -> None:
    cypher = "MATCH (c:Community {id:$id})-[:DOWN]->(e:Entity) RETURN e"
    assert checks.require_labeled(cypher) is None


def test_variable_seen_only_without_label_is_not_bound() -> None:
    message = _unlabeled_message("MATCH (d)-[:R]->(u:Unit) MATCH (d) RETURN d")
    assert "(d)" in message


def test_optional_match_unlabeled_is_rejected_and_error_types() -> None:
    with pytest.raises(UnlabeledMatchError) as excinfo:
        checks.require_labeled("OPTIONAL MATCH (x) RETURN x")
    assert isinstance(excinfo.value, GraphError)
    assert isinstance(excinfo.value, ValueError)
    assert "(x)" in str(excinfo.value)


def test_error_class_is_reexported_from_package() -> None:
    with pytest.raises(graph.UnlabeledMatchError):
        checks.require_labeled("MATCH (c {id:$id}) RETURN c")


# ---------------------------------------------------------------- audit_write_problems


def test_merge_with_all_audit_fields_has_no_problem() -> None:
    assert checks.audit_write_problems(MERGE_FULL) == []


def test_merge_without_on_create_set_misses_created_at_only() -> None:
    problems = checks.audit_write_problems(MERGE_NO_ON_CREATE)
    assert len(problems) == 1
    assert "created_at" in problems[0]


def test_merge_without_final_set_misses_updated_fields() -> None:
    problems = checks.audit_write_problems(MERGE_NO_FINAL_SET)
    assert _names(problems, "updated_at")
    assert _names(problems, "updated_by")
    assert not _names(problems, "created_at")


def test_create_with_audit_fields_in_pattern_has_no_problem() -> None:
    cypher = (
        "CREATE (n:Unit {scope:$scope, id:$id, created_at: datetime($now), "
        "updated_at: datetime($now), updated_by: $actor})"
    )
    assert checks.audit_write_problems(cypher) == []


def test_create_without_audit_fields_misses_all_three() -> None:
    problems = checks.audit_write_problems("CREATE (n:Unit {scope:$scope, id:$id})")
    assert _names(problems, "created_at")
    assert _names(problems, "updated_at")
    assert _names(problems, "updated_by")
    assert len(problems) == 3


def test_set_on_matched_node_requires_updated_fields() -> None:
    problems = checks.audit_write_problems(
        'MATCH (n:Unit {scope:$scope, id:$id}) SET n.status = "deleted"'
    )
    assert _names(problems, "updated_at")
    assert _names(problems, "updated_by")
    assert not _names(problems, "created_at")


def test_set_on_matched_node_with_updated_fields_has_no_problem() -> None:
    cypher = (
        'MATCH (n:Unit {scope:$scope, id:$id}) SET n.status = "deleted", '
        "n.updated_at = datetime($now), n.updated_by = $actor"
    )
    assert checks.audit_write_problems(cypher) == []


def test_remove_on_matched_node_requires_updated_fields() -> None:
    problems = checks.audit_write_problems("MATCH (n:Unit {scope:$scope, id:$id}) REMOVE n.tmp")
    assert _names(problems, "updated_at")
    assert _names(problems, "updated_by")


def test_merged_relationship_without_created_at_is_one_problem() -> None:
    problems = checks.audit_write_problems("MATCH (a:Entity),(b:Entity) MERGE (a)-[r:SIMILAR]->(b)")
    assert len(problems) == 1
    assert "created_at" in problems[0]


def test_merged_relationship_with_on_create_set_has_no_problem() -> None:
    cypher = (
        "MATCH (a:Entity),(b:Entity) MERGE (a)-[r:SIMILAR]->(b) "
        "ON CREATE SET r.created_at = datetime($now)"
    )
    assert checks.audit_write_problems(cypher) == []


def test_created_relationship_with_created_at_property_has_no_problem() -> None:
    cypher = (
        "MATCH (a:Entity {id:$a}),(b:Entity {id:$b}) "
        "CREATE (a)-[r:LINKS {created_at: datetime($now)}]->(b)"
    )
    assert checks.audit_write_problems(cypher) == []


def test_relationship_without_variable_or_created_at_is_a_problem() -> None:
    problems = checks.audit_write_problems(
        "MATCH (a:Entity {id:$a}),(b:Entity {id:$b}) MERGE (a)-[:LINKS]->(b)"
    )
    assert _names(problems, "created_at")


def test_read_only_statement_has_no_problem() -> None:
    cypher = 'MATCH (n:Unit {scope:$scope}) WHERE n.status = "active" RETURN n.id'
    assert checks.audit_write_problems(cypher) == []


# ---------------------------------------------------------------- require_audited


def test_require_audited_raises_with_every_problem() -> None:
    cypher = "CREATE (n:Unit {scope:$scope, id:$id})"
    problems = checks.audit_write_problems(cypher)
    assert problems
    with pytest.raises(AuditFieldError) as excinfo:
        checks.require_audited(cypher)
    assert isinstance(excinfo.value, GraphError)
    message = str(excinfo.value)
    for problem in problems:
        assert problem in message


def test_require_audited_returns_none_without_problem() -> None:
    assert checks.require_audited(MERGE_FULL) is None
    assert checks.require_audited("MATCH (n:Unit {scope:$scope}) RETURN n.id") is None
