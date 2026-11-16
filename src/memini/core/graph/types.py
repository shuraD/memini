"""Shared value types, errors and protocols of the graph query layer.

This module holds only declarations: the statement value, the error hierarchy, and the
runner and performance-recorder protocols. It performs no database access and builds no
query text.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "GraphError",
    "PerfRecorder",
    "RunResult",
    "Runner",
    "Statement",
    "UnlabeledMatchError",
]


@dataclass(frozen=True)
class Statement:
    """A parameterized query ready to run.

    Attributes:
        name: Stable identifier of the query, used for logging.
        cypher: Query text with parameter placeholders.
        params: Values bound to the placeholders.
        kind: Whether the statement reads, writes, administers the graph, or deletes
            nodes for good (``delete``: checked by the database guard before it runs).
        scope: The scope the statement works in, or None when it names none; it is
            recorded with the statement's timing.

    """

    name: str
    cypher: str
    params: Mapping[str, Any]
    kind: Literal["read", "write", "admin", "delete"]
    scope: str | None = None


class GraphError(Exception):
    """Base class of every error raised by the graph query layer."""


class UnlabeledMatchError(GraphError, ValueError):
    """A pattern matches nodes without a label."""


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
        self,
        name: str,
        seconds: float,
        rows: int,
        db_hits: int | None,
        *,
        scope: str | None = None,
    ) -> None:
        """Record the duration, row count, database hits and scope of a named statement."""
        ...
