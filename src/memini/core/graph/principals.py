"""Principals and their access to scopes, stored as ``CAN_ACCESS`` relationships.

A principal (a user or a service) is granted read or write access to a scope by a
``CAN_ACCESS {access}`` relationship from its ``Principal`` node to a ``Scope`` node.
Granting runs one audited statement through a graph session; loading turns those
relationships back into an access context. Every id and access value travels as a
parameter, never in the query text.
"""

from __future__ import annotations

from memini.core.graph.access import require_write
from memini.core.graph.session import GraphSession
from memini.core.graph.types import AccessContext, Clock, Runner, Statement

__all__ = ["ACCESS", "PRINCIPAL_KINDS", "grant", "load_access"]

ACCESS: tuple[str, ...] = ("read", "write")
"""Access levels a ``CAN_ACCESS`` relationship may carry; ``write`` implies read."""

PRINCIPAL_KINDS: tuple[str, ...] = ("user", "service")
"""Kinds of principal that may be granted access."""

_GRANT_CYPHER = (
    "MERGE (p:Principal {id: $principal}) "
    "ON CREATE SET p.created_at = datetime($now), p.status = 'active' "
    "SET p.kind = $kind, p.updated_at = datetime($now), p.updated_by = $actor "
    "MERGE (s:Scope {id: $scope}) "
    "ON CREATE SET s.created_at = datetime($now), s.status = 'active' "
    "SET s.updated_at = datetime($now), s.updated_by = $actor "
    "MERGE (p)-[r:CAN_ACCESS]->(s) "
    "ON CREATE SET r.created_at = datetime($now) "
    "SET r.access = $access"
)

_LOAD_CYPHER = (
    "MATCH (p:Principal {id: $principal})-[r:CAN_ACCESS]->(s:Scope) "
    "RETURN s.id AS scope, r.access AS access"
)


def grant(
    session: GraphSession, clock: Clock, *, principal: str, kind: str, scope: str, access: str
) -> None:
    """Grant ``principal`` read or write access to ``scope``.

    Granting again replaces the access level of the existing relationship.

    Args:
        session: Session whose context must be able to write ``scope``.
        clock: Source of the write time.
        principal: Id of the principal receiving access.
        kind: Kind of the principal, one of ``PRINCIPAL_KINDS``.
        scope: Id of the scope access is granted to.
        access: Access level, one of ``ACCESS``.

    Raises:
        ValueError: When ``principal`` or ``scope`` is empty, or ``kind`` or ``access``
            is not a known value.
        AccessDenied: When the session has no context or cannot write ``scope``.

    """
    if not principal:
        raise ValueError("principal must not be empty")
    if not scope:
        raise ValueError("scope must not be empty")
    if kind not in PRINCIPAL_KINDS:
        raise ValueError(f"kind must be one of {PRINCIPAL_KINDS}, got {kind!r}")
    if access not in ACCESS:
        raise ValueError(f"access must be one of {ACCESS}, got {access!r}")
    ctx = session.ctx
    require_write(ctx, scope)
    params = {
        "principal": principal,
        "kind": kind,
        "scope": scope,
        "access": access,
        "now": clock().isoformat(),
        "actor": ctx.actor,
    }
    session.run(
        Statement(
            name="grant:CAN_ACCESS", cypher=_GRANT_CYPHER, params=params, scope=scope, kind="write"
        )
    )


def load_access(runner: Runner, principal: str, *, actor: str) -> AccessContext:
    """Build the access context of ``principal`` from its ``CAN_ACCESS`` relationships.

    Rows with an access value outside ``ACCESS`` are ignored.

    Args:
        runner: Executes the lookup query.
        principal: Id of the principal whose access is loaded.
        actor: Value recorded as the author of writes made under the context.

    Returns:
        The principal's context; with no grants, its read and write scopes are empty.

    Raises:
        ValueError: When ``principal`` or ``actor`` is empty.

    """
    if not principal:
        raise ValueError("principal must not be empty")
    if not actor:
        raise ValueError("actor must not be empty")
    result = runner.run(_LOAD_CYPHER, {"principal": principal})
    read: set[str] = set()
    write: set[str] = set()
    for row in result.rows:
        level = row.get("access")
        if level not in ACCESS:
            continue
        scope = str(row["scope"])
        read.add(scope)
        if level == "write":
            write.add(scope)
    return AccessContext(
        principal=principal,
        read_scopes=frozenset(read),
        write_scopes=frozenset(write),
        actor=actor,
    )
