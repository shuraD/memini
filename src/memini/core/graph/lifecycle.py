"""Soft delete and restore of the items of a scope, each followed by one audit event.

A soft delete sets ``status`` to ``'deleted'`` and records ``deleted_at`` and
``deleted_by``, so every status-filtering reader hides the item; a restore brings it
back to ``'active'``. Nothing is ever removed. Each call runs one labeled statement per
id-bearing label of the schema, then exactly one append-only audit event. Every value
(scope, ids, actor, time) is passed as a parameter, never written into the query text.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from memini.core.graph.access import require_write
from memini.core.graph.auditlog import AUDIT_LABEL, audit_event
from memini.core.graph.session import GraphSession
from memini.core.graph.types import Clock, Statement
from memini.core.graph.writes import WriteBuilder
from memini.core.schema import Schema

__all__ = ["delete", "restore"]

_ID_PROPERTY = "id"

_TRANSITIONS: dict[str, tuple[str, str]] = {
    "delete": (
        "active",
        "n.status = 'deleted', n.deleted_at = datetime($now), n.deleted_by = $actor",
    ),
    "restore": (
        "deleted",
        "n.status = 'active', n.deleted_at = null, n.deleted_by = null",
    ),
}
"""Per action: the status an item must have, and the assignments that change it."""


def _target_labels(schema: Schema) -> list[str]:
    """Return every label carrying an id property, except the audit label, sorted."""
    return sorted(
        label
        for label, node in schema.nodes.items()
        if _ID_PROPERTY in node.properties and label != AUDIT_LABEL
    )


def _change_status(
    session: GraphSession,
    schema: Schema,
    clock: Clock,
    scope: str,
    ids: Sequence[str],
    *,
    reason: str,
    action: str,
) -> int:
    """Run the status change ``action`` on ``ids`` in ``scope`` and audit it.

    Raises:
        AccessDenied: When the session has no access context or ``scope`` is not writable.
        ValueError: When ``ids`` is empty, an id is not a non-empty str, or ``reason`` is empty.

    """
    ctx = session.ctx
    require_write(ctx, scope)
    if not ids:
        raise ValueError("ids must not be empty")
    if any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("every id must be a non-empty str")
    if not reason:
        raise ValueError("reason must not be empty")
    required_status, assignments = _TRANSITIONS[action]
    audit = WriteBuilder(schema, ctx, clock).audit_set("n")
    params: dict[str, Any] = {
        "scope": scope,
        "ids": list(ids),
        "now": clock().isoformat(),
        "actor": ctx.actor,
    }
    changed = 0
    for label in _target_labels(schema):
        cypher = (
            f"UNWIND $ids AS id MATCH (n:{label} {{scope: $scope, id: id}}) "
            f"WHERE n.status = '{required_status}' "
            f"SET {assignments}, {audit} RETURN count(n) AS n"
        )
        rows = session.run(
            Statement(
                name=f"{action}:{label}",
                cypher=cypher,
                params=dict(params),
                scope=scope,
                kind="write",
            )
        )
        changed += int(rows[0]["n"]) if rows else 0
    session.run(
        audit_event(ctx, clock, scope=scope, action=action, reason=reason, ids=ids, count=changed)
    )
    return changed


def delete(
    session: GraphSession,
    schema: Schema,
    clock: Clock,
    scope: str,
    ids: Sequence[str],
    *,
    reason: str,
) -> int:
    """Soft-delete the active items ``ids`` of ``scope`` and record one audit event.

    Args:
        session: Session running the statements under the caller's access context.
        schema: Schema whose id-bearing labels are targeted.
        clock: Source of the deletion time.
        scope: Scope holding the items; must be writable.
        ids: Ids of the items to delete; must not be empty.
        reason: Why the items are deleted; must not be empty.

    Returns:
        The number of items whose status changed.

    Raises:
        AccessDenied: When the session has no access context or ``scope`` is not writable.
        ValueError: When ``ids`` is empty, an id is not a non-empty str, or ``reason`` is empty.

    """
    return _change_status(session, schema, clock, scope, ids, reason=reason, action="delete")


def restore(
    session: GraphSession,
    schema: Schema,
    clock: Clock,
    scope: str,
    ids: Sequence[str],
    *,
    reason: str,
) -> int:
    """Bring the soft-deleted items ``ids`` of ``scope`` back and record one audit event.

    Args:
        session: Session running the statements under the caller's access context.
        schema: Schema whose id-bearing labels are targeted.
        clock: Source of the restore time.
        scope: Scope holding the items; must be writable.
        ids: Ids of the items to restore; must not be empty.
        reason: Why the items are restored; must not be empty.

    Returns:
        The number of items whose status changed.

    Raises:
        AccessDenied: When the session has no access context or ``scope`` is not writable.
        ValueError: When ``ids`` is empty, an id is not a non-empty str, or ``reason`` is empty.

    """
    return _change_status(session, schema, clock, scope, ids, reason=reason, action="restore")
