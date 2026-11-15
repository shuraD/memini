"""Shared value types, errors and protocols of the graph query layer.

This module holds only declarations: the statement and access-context values, the
error hierarchy, the runner and performance-recorder protocols, the confirmation
check for destructive calls, and the registry of named read queries. It performs no
database access and builds no query text.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

__all__ = [
    "QUERY_REGISTRY",
    "AccessContext",
    "AccessDenied",
    "AuditFieldError",
    "Clock",
    "ConfirmationRequired",
    "GraphError",
    "IsolationError",
    "PerfRecorder",
    "RunResult",
    "Runner",
    "Statement",
    "UnlabeledMatchError",
    "UnsafeOperation",
    "registered_query",
    "require_confirmation",
]


@dataclass(frozen=True)
class Statement:
    """A parameterized query ready to run.

    Attributes:
        name: Stable identifier of the query, used for logging and the registry.
        cypher: Query text with parameter placeholders.
        params: Values bound to the placeholders.
        scope: Scope the statement targets, or None when it spans no single scope.
        kind: Whether the statement reads, writes, or administers the graph.

    """

    name: str
    cypher: str
    params: Mapping[str, Any]
    scope: str | None
    kind: Literal["read", "write", "admin"]


@dataclass(frozen=True)
class AccessContext:
    """Who is calling and which scopes they may read and write.

    Attributes:
        principal: Identity the access rights belong to; must not be empty.
        read_scopes: Scopes the principal may read.
        write_scopes: Scopes the principal may write; a subset of ``read_scopes``.
        actor: Value recorded as the author of writes (a job run or an API caller id);
            must not be empty.

    """

    principal: str
    read_scopes: frozenset[str]
    write_scopes: frozenset[str]
    actor: str

    def __post_init__(self) -> None:
        """Validate the context.

        Raises:
            ValueError: If principal or actor is empty, or write_scopes is not a
                subset of read_scopes.

        """
        if not self.principal:
            raise ValueError("AccessContext.principal must not be empty")
        if not self.actor:
            raise ValueError("AccessContext.actor must not be empty")
        if not self.write_scopes <= self.read_scopes:
            extra = sorted(self.write_scopes - self.read_scopes)
            raise ValueError(f"AccessContext.write_scopes not in read_scopes: {extra}")


class GraphError(Exception):
    """Base class of every error raised by the graph query layer."""


class AccessDenied(GraphError):  # noqa: N818 - name fixed by the GS-05 interface
    """The access context does not grant the requested scope or operation."""


class IsolationError(GraphError):
    """A statement could reach data outside its permitted scopes."""


class UnlabeledMatchError(GraphError, ValueError):
    """A pattern matches nodes without a label."""


class AuditFieldError(GraphError):
    """A write lacks, or tampers with, a required audit field."""


class ConfirmationRequired(GraphError):  # noqa: N818 - name fixed by the GS-05 interface
    """A destructive call was made without the matching confirmation."""


class UnsafeOperation(GraphError):  # noqa: N818 - name fixed by the GS-05 interface
    """An operation is refused because it is unsafe to run."""


@dataclass(frozen=True)
class RunResult:
    """Outcome of running one statement.

    Attributes:
        rows: Returned records, one mapping per row.
        db_hits: Database hits reported by profiling, or None when not profiled.

    """

    rows: list[dict[str, Any]]
    db_hits: int | None = None


@runtime_checkable
class Runner(Protocol):
    """Executes query text against a graph database."""

    uri: str

    def run(
        self,
        cypher: str,
        params: Mapping[str, Any],
        *,
        profile: bool = False,
        autocommit: bool = False,
    ) -> RunResult:
        """Run ``cypher`` with ``params`` and return its rows.

        Args:
            cypher: Query text.
            params: Values bound to the query placeholders.
            profile: Collect database hits when True.
            autocommit: Run outside an explicit transaction when True.

        """
        ...


@runtime_checkable
class PerfRecorder(Protocol):
    """Receives one performance sample per executed statement."""

    def record(
        self, name: str, scope: str | None, seconds: float, rows: int, db_hits: int | None
    ) -> None:
        """Record the duration, row count and database hits of a named statement."""
        ...


Clock = Callable[[], datetime]
"""Returns the current time as a timezone-aware UTC datetime."""


def require_confirmation(confirm: str | None, scope: str) -> None:
    """Check that a destructive call was confirmed by typing its scope again.

    Args:
        confirm: The confirmation the caller supplied.
        scope: The scope id the destructive call targets.

    Raises:
        ConfirmationRequired: Unless ``confirm`` equals ``scope`` exactly
            (case-sensitive, no trimming).

    """
    if confirm != scope:
        raise ConfirmationRequired(
            f"destructive call on scope {scope!r} requires confirm={scope!r}"
        )


F = TypeVar("F", bound=Callable[..., Statement])

_REGISTRY: dict[str, Callable[..., Statement]] = {}

QUERY_REGISTRY: Mapping[str, Callable[..., Statement]] = MappingProxyType(_REGISTRY)
"""Read-only view of every registered named read query, keyed by name."""


def registered_query(name: str) -> Callable[[F], F]:
    """Register a statement factory under ``name``.

    Args:
        name: Unique query name.

    Returns:
        A decorator that registers the function and returns it unchanged.

    Raises:
        ValueError: When applied and ``name`` is already registered; the earlier
            registration is kept.

    """

    def decorator(func: F) -> F:
        if name in _REGISTRY:
            raise ValueError(f"query {name!r} is already registered")
        _REGISTRY[name] = func
        return func

    return decorator
