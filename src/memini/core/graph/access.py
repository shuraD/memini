"""Access checks that keep every graph operation inside the caller's scopes.

An access context is turned into a read filter (a ``scope IN $allowed`` condition per
pattern variable, with the scope values passed only as parameters) and into write
checks (the target scope must be writable, and a relationship must not cross scopes).
This module performs no database access.
"""

from __future__ import annotations

from typing import Any

from memini.core.graph.types import AccessContext, AccessDenied, IsolationError

__all__ = ["read_filter", "require_context", "require_same_scope", "require_write"]

_ALLOWED_PARAM = "allowed"


def require_context(ctx: AccessContext | None) -> AccessContext:
    """Return the access context, refusing a missing one.

    Args:
        ctx: The caller's access context, or ``None`` when there is none.

    Returns:
        The same ``ctx`` object.

    Raises:
        AccessDenied: When ``ctx`` is ``None``.

    """
    if ctx is None:
        raise AccessDenied("no access context")
    return ctx


def read_filter(ctx: AccessContext | None, *variables: str) -> tuple[str, dict[str, Any]]:
    """Build the condition restricting each pattern variable to the readable scopes.

    The scope values go only into the parameters, never into the returned text.

    Args:
        ctx: The caller's access context.
        *variables: Pattern variables to restrict, in the order they appear in the text.

    Returns:
        The condition text (one ``<v>.scope IN $allowed`` per variable, joined with
        ``AND``) and its parameters (``allowed``: the readable scopes as a sorted list).

    Raises:
        AccessDenied: When ``ctx`` is ``None`` or grants no read scope.
        ValueError: When no variable is given.

    """
    context = require_context(ctx)
    if not variables:
        raise ValueError("read_filter needs at least one variable")
    if not context.read_scopes:
        raise AccessDenied("no read scope")
    text = " AND ".join(f"{v}.scope IN ${_ALLOWED_PARAM}" for v in variables)
    return text, {_ALLOWED_PARAM: sorted(context.read_scopes)}


def require_write(ctx: AccessContext | None, scope: str) -> None:
    """Refuse a write to a scope the caller may not write.

    Args:
        ctx: The caller's access context.
        scope: The scope the write targets.

    Raises:
        AccessDenied: When ``ctx`` is ``None`` or ``scope`` is not writable.

    """
    context = require_context(ctx)
    if scope not in context.write_scopes:
        raise AccessDenied(f"no write access to scope {scope!r}")


def require_same_scope(from_scope: str, to_scope: str) -> None:
    """Refuse a relationship between two different scopes.

    Args:
        from_scope: Scope of the relationship's start node.
        to_scope: Scope of the relationship's end node.

    Raises:
        IsolationError: When the two scopes differ.

    """
    if from_scope != to_scope:
        raise IsolationError(f"relationship crosses scopes {from_scope!r} -> {to_scope!r}")
