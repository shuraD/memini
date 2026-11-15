"""Remove every node of one scope forever, label by label, in batched transactions.

``wipe_scope`` refuses to run unless the caller typed the scope again as confirmation,
the database host is allowed, and the caller may write the scope. It then deletes the
nodes of each schema label (audit events always kept, other labels kept on request)
and records one append-only audit event with the number of removed nodes. The scope
value is passed only as a parameter, never written into the query text.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from memini.core.graph.access import require_write
from memini.core.graph.auditlog import AUDIT_LABEL, audit_event
from memini.core.graph.session import GraphSession
from memini.core.graph.types import Clock, Statement, require_confirmation
from memini.core.schema import Schema
from memini.infra.graph.guard import guard_from_env

__all__ = ["wipe_scope"]

_ACTION = "wipe_scope"


def _wipe_statement(label: str, scope: str, batch_size: int) -> Statement:
    """Build the batched statement deleting every ``label`` node of ``scope``."""
    cypher = (
        f"MATCH (n:{label}) WHERE n.scope = $scope "
        "CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF $batch_size ROWS "
        "RETURN count(n) AS count"
    )
    return Statement(
        name=f"{_ACTION}:{label}",
        cypher=cypher,
        params={"scope": scope, "batch_size": batch_size},
        scope=scope,
        kind="admin",
    )


def wipe_scope(
    session: GraphSession,
    schema: Schema,
    clock: Clock,
    scope: str,
    *,
    confirm: str | None,
    reason: str,
    batch_size: int = 1000,
    keep_labels: Sequence[str] = ("AuditEvent", "Scope", "Principal"),
    env: Mapping[str, str] = os.environ,
) -> int:
    """Delete every node of ``scope`` forever and record one audit event.

    All checks run before any statement reaches the runner. Labels are processed in
    the schema's declaration order; nodes of every status are removed. A runner error
    propagates unchanged and no audit event is recorded.

    Args:
        session: Session whose runner and access context are used.
        schema: Schema whose node labels are wiped.
        clock: Source of the audit event time.
        scope: Scope to remove.
        confirm: Must equal ``scope`` exactly.
        reason: Why the scope is removed; must not be blank.
        batch_size: Rows deleted per inner transaction; at least 1.
        keep_labels: Labels left untouched; the audit label is always kept.
        env: Environment read for the database host guard.

    Returns:
        The total number of removed nodes.

    Raises:
        ConfirmationRequired: When ``confirm`` differs from ``scope``.
        UnsafeDatabaseError: When the session's database host is refused.
        AccessDenied: When ``scope`` is not writable in the session's context.
        ValueError: When ``reason`` is blank or ``batch_size`` is below 1.

    """
    require_confirmation(confirm, scope)
    guard_from_env(session.uri, env)
    require_write(session.ctx, scope)
    if not reason.strip():
        raise ValueError("wipe reason must not be blank")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    kept = {*keep_labels, AUDIT_LABEL}
    total = 0
    for label in schema.nodes:
        if label in kept:
            continue
        rows = session.run(_wipe_statement(label, scope, batch_size))
        total += sum(int(row["count"]) for row in rows)
    session.run(
        audit_event(session.ctx, clock, scope=scope, action=_ACTION, reason=reason, count=total)
    )
    return total
