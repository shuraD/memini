"""Acceptance tests for GS-06: the labeled-MATCH check of memini.core.graph.checks."""

from __future__ import annotations

import pytest

from memini.core import graph
from memini.core.graph import checks
from memini.core.graph.types import GraphError, UnlabeledMatchError


def _unlabeled_message(cypher: str) -> str:
    with pytest.raises(UnlabeledMatchError) as excinfo:
        checks.require_labeled(cypher)
    return str(excinfo.value)


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
