"""Delete forever the soft-deleted or superseded items of a scope after a waiting period.

``purge`` is an explicit, confirmed operation: it refuses to run without the exact scope
as confirmation, on a database host that is not allowed, or without write access. It then
removes, label by label and in batched transactions, the items whose soft delete or
supersession is older than the waiting period, together with their relationships, and
records one audit event with the number of removed items. Every value (scope, cutoff,
batch size, ids) is passed as a parameter, never written into the query text.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Any

from memini.core.graph.access import require_write
from memini.core.graph.auditlog import AUDIT_LABEL, audit_event
from memini.core.graph.session import GraphSession
from memini.core.graph.types import Clock, Statement, require_confirmation
from memini.core.schema import Schema
from memini.infra.graph.guard import guard_from_env

__all__ = ["purge"]

_ID_PROPERTY = "id"


def _purge_statement(
    label: str, scope: str, params: Mapping[str, Any], with_ids: bool
) -> Statement:
    """Build the batched delete-forever statement for one label."""
    ids_filter = " AND n.id IN $ids" if with_ids else ""
    cypher = (
        f"MATCH (n:{label}) WHERE n.scope = $scope AND ("
        "(n.status = 'deleted' AND n.deleted_at <= datetime($cutoff)) "
        "OR (n.status = 'superseded' AND n.updated_at <= datetime($cutoff)))"
        f"{ids_filter} "
        "CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF $batch_size ROWS "
        "RETURN count(n) AS removed"
    )
    return Statement(name=f"purge:{label}", cypher=cypher, params=params, scope=scope, kind="admin")


def purge(
    session: GraphSession,
    schema: Schema,
    clock: Clock,
    scope: str,
    *,
    older_than: timedelta,
    confirm: str | None,
    reason: str,
    ids: Sequence[str] | None = None,
    batch_size: int = 1000,
    env: Mapping[str, str] = os.environ,
) -> int:
    """Delete forever the expired soft-deleted or superseded items of one scope.

    Args:
        session: Session the statements run through.
        schema: Schema whose id-bearing node labels are purged, in their order.
        clock: Source of the current time.
        scope: Scope to purge.
        older_than: Waiting period since the soft delete or supersession; not negative.
        confirm: Must equal ``scope`` exactly.
        reason: Why the purge is done; recorded in the audit event.
        ids: When given, only these ids are purged.
        batch_size: Rows deleted per transaction; at least 1.
        env: Environment read by the database host guard.

    Returns:
        Total number of removed items over all labels.

    Raises:
        ConfirmationRequired: When ``confirm`` is not exactly ``scope``.
        UnsafeDatabaseError: When the session's database host is not allowed.
        AccessDenied: When ``scope`` is not writable in the session's context.
        ValueError: When ``older_than`` is negative or ``batch_size`` is below 1.

    """
    require_confirmation(confirm, scope)
    guard_from_env(session.uri, env)
    require_write(session.ctx, scope)
    if older_than < timedelta(0):
        raise ValueError("older_than must not be negative")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    cutoff = (clock() - older_than).isoformat()
    params: dict[str, Any] = {"scope": scope, "cutoff": cutoff, "batch_size": batch_size}
    if ids is not None:
        params["ids"] = list(ids)
    total = 0
    for label, node in schema.nodes.items():
        if label == AUDIT_LABEL or _ID_PROPERTY not in node.properties:
            continue
        rows = session.run(_purge_statement(label, scope, dict(params), ids is not None))
        total += sum(int(row["removed"]) for row in rows)
    session.run(
        audit_event(
            session.ctx, clock, scope=scope, action="purge", reason=reason, ids=ids, count=total
        )
    )
    return total
