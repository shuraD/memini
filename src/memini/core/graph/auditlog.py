"""The append-only audit event recorded for each delete, restore, purge or scope wipe.

``audit_event`` builds one ``CREATE`` statement for an audit node; it never matches,
merges or deletes, so audit events are only ever added. Every value (scope, id,
actor, reason, time) is passed as a parameter, never written into the query text.
This module performs no database access.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from memini.core.graph.access import require_write
from memini.core.graph.types import AccessContext, Clock, Statement

__all__ = ["AUDIT_ACTIONS", "AUDIT_LABEL", "audit_event"]

AUDIT_ACTIONS: tuple[str, ...] = ("delete", "restore", "purge", "wipe_scope")
"""Actions an audit event may record."""

AUDIT_LABEL: str = "AuditEvent"
"""Label of the audit event node."""

_CYPHER = (
    f"CREATE (e:{AUDIT_LABEL} {{scope: $scope, id: $event_id, action: $action, ids: $ids, "
    "count: $count, by: $actor, at: datetime($now), reason: $reason, status: 'active', "
    "created_at: datetime($now), updated_at: datetime($now), updated_by: $actor})"
)


def audit_event(
    ctx: AccessContext,
    clock: Clock,
    *,
    scope: str,
    action: str,
    reason: str,
    ids: Sequence[str] | None = None,
    count: int | None = None,
) -> Statement:
    """Build the statement that creates one audit event.

    Args:
        ctx: The caller's access context; its actor is recorded as the author.
        clock: Source of the event time.
        scope: Scope the audited operation targeted; must be writable.
        action: One of ``AUDIT_ACTIONS``.
        reason: Why the operation was done; must not be empty.
        ids: Ids of the affected items, when known.
        count: Number of affected items; defaults to ``len(ids)``.

    Returns:
        A write statement named ``audit:<action>`` targeting ``scope``.

    Raises:
        AccessDenied: When ``scope`` is not writable in ``ctx``.
        ValueError: When ``action`` is unknown, ``reason`` is empty, or neither
            ``ids`` nor ``count`` is given.

    """
    require_write(ctx, scope)
    if action not in AUDIT_ACTIONS:
        raise ValueError(f"unknown audit action {action!r}; expected one of {AUDIT_ACTIONS}")
    if not reason:
        raise ValueError("audit reason must not be empty")
    if ids is None and count is None:
        raise ValueError("audit event needs ids or a count")
    id_list = list(ids) if ids is not None else None
    total = count if count is not None else len(id_list or [])
    params: dict[str, Any] = {
        "scope": scope,
        "event_id": uuid.uuid4().hex,
        "action": action,
        "ids": id_list,
        "count": total,
        "actor": ctx.actor,
        "now": clock().isoformat(),
        "reason": reason,
    }
    return Statement(
        name=f"audit:{action}", cypher=_CYPHER, params=params, scope=scope, kind="write"
    )
