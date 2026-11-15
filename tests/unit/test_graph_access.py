"""Acceptance tests for GS-07: access context to read filter and write checks."""

from __future__ import annotations

import pytest

from memini.core.graph import access
from memini.core.graph.types import AccessContext, AccessDenied, IsolationError


def _ctx(
    read: set[str] | frozenset[str] = frozenset({"A", "B"}),
    write: set[str] | frozenset[str] = frozenset({"A"}),
) -> AccessContext:
    return AccessContext(
        principal="test-principal",
        read_scopes=frozenset(read),
        write_scopes=frozenset(write),
        actor="test-actor",
    )


# --- require_context -------------------------------------------------------


def test_require_context_none_raises_access_denied() -> None:
    with pytest.raises(AccessDenied):
        access.require_context(None)


def test_require_context_returns_same_object() -> None:
    ctx = _ctx()
    assert access.require_context(ctx) is ctx


# --- read_filter -----------------------------------------------------------


def test_read_filter_none_raises_access_denied() -> None:
    with pytest.raises(AccessDenied):
        access.read_filter(None, "n")


def test_read_filter_single_variable_text_and_sorted_list_params() -> None:
    ctx = _ctx(read={"B", "A"}, write=set())
    text, params = access.read_filter(ctx, "n")
    assert text == "n.scope IN $allowed"
    assert params == {"allowed": ["A", "B"]}
    assert isinstance(params["allowed"], list)


def test_read_filter_two_variables_joined_with_and() -> None:
    ctx = _ctx()
    text, params = access.read_filter(ctx, "a", "b")
    assert text == "a.scope IN $allowed AND b.scope IN $allowed"
    assert params == {"allowed": ["A", "B"]}


def test_read_filter_three_variables_keep_order() -> None:
    ctx = _ctx()
    text, _ = access.read_filter(ctx, "x", "a", "m")
    assert text == "x.scope IN $allowed AND a.scope IN $allowed AND m.scope IN $allowed"


def test_read_filter_sorts_many_scopes() -> None:
    ctx = _ctx(read={"zeta", "alpha", "Mid", "beta"}, write=set())
    _, params = access.read_filter(ctx, "n")
    assert params["allowed"] == ["Mid", "alpha", "beta", "zeta"]


def test_read_filter_no_variable_raises_value_error() -> None:
    with pytest.raises(ValueError):
        access.read_filter(_ctx())


def test_read_filter_empty_read_scopes_raises_access_denied() -> None:
    ctx = _ctx(read=set(), write=set())
    with pytest.raises(AccessDenied):
        access.read_filter(ctx, "n")


def test_read_filter_scope_values_only_in_params() -> None:
    evil = "A' OR 1=1 // MATCH (x) DETACH DELETE x"
    ctx = _ctx(read={evil, "B"}, write=set())
    text, params = access.read_filter(ctx, "n")
    assert evil not in text
    assert "DELETE" not in text
    assert text == "n.scope IN $allowed"
    assert params == {"allowed": sorted({evil, "B"})}
    assert evil in params["allowed"]


def test_read_filter_returns_tuple_of_text_and_dict() -> None:
    result = access.read_filter(_ctx(), "n")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], dict)
    assert set(result[1]) == {"allowed"}


# --- require_write ---------------------------------------------------------


def test_require_write_none_raises_access_denied() -> None:
    with pytest.raises(AccessDenied):
        access.require_write(None, "A")


def test_require_write_allowed_returns_none() -> None:
    assert access.require_write(_ctx(read={"A", "B"}, write={"A"}), "A") is None


def test_require_write_read_only_scope_denied_with_scope_in_message() -> None:
    ctx = _ctx(read={"A", "readonly-scope"}, write={"A"})
    with pytest.raises(AccessDenied) as excinfo:
        access.require_write(ctx, "readonly-scope")
    assert "readonly-scope" in str(excinfo.value)


def test_require_write_unknown_scope_denied_with_scope_in_message() -> None:
    ctx = _ctx(read={"A"}, write={"A"})
    with pytest.raises(AccessDenied) as excinfo:
        access.require_write(ctx, "elsewhere-scope")
    assert "elsewhere-scope" in str(excinfo.value)


# --- require_same_scope ----------------------------------------------------


def test_require_same_scope_different_raises_isolation_error() -> None:
    with pytest.raises(IsolationError):
        access.require_same_scope("A", "B")


def test_require_same_scope_equal_returns_none() -> None:
    assert access.require_same_scope("A", "A") is None


def test_require_same_scope_is_case_sensitive() -> None:
    with pytest.raises(IsolationError):
        access.require_same_scope("a", "A")
