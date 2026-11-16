"""Run Cypher on Neo4j through the official driver, behind the database guard.

The URI is checked by :func:`memini.infra.graph.guard.guard_from_env` before any
driver is created. Rows come back as plain dicts; profiled runs also report the
total db hits summed over the profile tree.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from neo4j import GraphDatabase

from memini.infra.graph.guard import guard_from_env

__all__ = ["DatabaseUnreachableError", "Neo4jRunner", "RunResult"]

_PROFILE_PREFIX = "PROFILE "
_DEFAULT_URI = "bolt://localhost:7687"
_ENV_URI = "NEO4J_URI"
_ENV_USER = "NEO4J_USER"
_ENV_CREDENTIAL = "NEO4J_PASSWORD"
_ENV_DATABASE = "NEO4J_DATABASE"


class DatabaseUnreachableError(RuntimeError):
    """Raised when the graph database cannot be reached."""


def _env_value(env: Mapping[str, str], key: str) -> str | None:
    """Return ``env[key]`` stripped, or ``None`` when missing or blank."""
    value = env.get(key, "").strip()
    return value or None


@dataclass(frozen=True)
class RunResult:
    """The outcome of one query run.

    Attributes:
        rows: One plain dict per returned record, in order.
        db_hits: Total db hits of a profiled run; ``None`` when not profiled.

    """

    rows: list[dict[str, Any]]
    db_hits: int | None = None


def _sum_db_hits(node: Mapping[str, Any] | None) -> int:
    """Return the sum of ``dbHits`` over ``node`` and all its descendants.

    A missing ``dbHits`` counts 0; a ``None`` tree sums to 0.
    """
    if node is None:
        return 0
    total = 0
    stack: list[Mapping[str, Any]] = [node]
    while stack:
        current = stack.pop()
        total += int(current.get("dbHits", 0) or 0)
        stack.extend(current.get("children") or ())
    return total


def _collect(result: Any, *, profile: bool) -> RunResult:
    """Read every record of ``result`` as a dict, then consume its summary."""
    rows = [dict(record.data()) for record in result]
    summary = result.consume()
    db_hits = _sum_db_hits(summary.profile) if profile else None
    return RunResult(rows=rows, db_hits=db_hits)


class Neo4jRunner:
    """Execute Cypher on a guarded Neo4j database.

    Attributes:
        uri: The URI given at construction.

    """

    def __init__(
        self,
        uri: str,
        auth: tuple[str, str] | None,
        *,
        database: str | None = None,
        env: Mapping[str, str] = os.environ,
        driver_factory: Callable[..., Any] | None = None,
    ) -> None:
        """Guard ``uri`` with ``env``, then create the driver once.

        Args:
            uri: The database URI.
            auth: ``(user, secret)`` or ``None``.
            database: Database name for sessions; ``None`` uses the default.
            env: Mapping read by the guard (only this mapping is read).
            driver_factory: Called as ``driver_factory(uri, auth=auth)``;
                ``None`` uses ``neo4j.GraphDatabase.driver`` (no connection is
                made at construction).

        Raises:
            UnsafeDatabaseError: If the guard refuses ``uri``; no driver is made.

        """
        guard_from_env(uri, env)
        if driver_factory is None:
            driver_factory = GraphDatabase.driver
        self.uri = uri
        self._database = database
        self._driver: Any = driver_factory(uri, auth=auth)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] = os.environ,
        *,
        driver_factory: Callable[..., Any] | None = None,
    ) -> Neo4jRunner:
        """Build a guarded runner from ``env`` and verify it can connect.

        Reads only ``env``: ``NEO4J_URI`` (default ``bolt://localhost:7687``),
        ``NEO4J_USER`` and ``NEO4J_PASSWORD`` (auth only when both are set) and
        ``NEO4J_DATABASE`` (``None`` uses the server default). Values are
        stripped; blank values count as missing.

        Args:
            env: Mapping holding the connection settings and guard settings.
            driver_factory: Passed through to the constructor.

        Returns:
            A runner whose database answered a connectivity check.

        Raises:
            UnsafeDatabaseError: If the guard refuses the URI; no driver is made.
            DatabaseUnreachableError: If the database cannot be reached; the
                driver is closed first.

        """
        uri = _env_value(env, _ENV_URI) or _DEFAULT_URI
        user = _env_value(env, _ENV_USER)
        secret = _env_value(env, _ENV_CREDENTIAL)
        auth = (user, secret) if user is not None and secret is not None else None
        runner = cls(
            uri,
            auth,
            database=_env_value(env, _ENV_DATABASE),
            env=env,
            driver_factory=driver_factory,
        )
        try:
            runner.verify_connectivity()
        except DatabaseUnreachableError:
            runner.close()
            raise
        return runner

    def verify_connectivity(self) -> None:
        """Check once that the database can be reached.

        Raises:
            DatabaseUnreachableError: If the driver's check fails; the original
                error is chained as the cause. The message names only the URI.

        """
        try:
            self._driver.verify_connectivity()
        except Exception as error:
            raise DatabaseUnreachableError(f"graph database unreachable at {self.uri}") from error

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
            cypher: The query text.
            params: Query parameters.
            profile: Prefix the query with ``PROFILE`` and report db hits.
            autocommit: Use an auto-commit transaction (``session.run``), as
                needed by ``CALL { } IN TRANSACTIONS``; otherwise a managed
                write transaction is used.

        Returns:
            The rows and, when profiled, the summed db hits.

        """
        query = _PROFILE_PREFIX + cypher if profile else cypher
        parameters = dict(params)
        with self._driver.session(database=self._database) as session:
            if autocommit:
                return _collect(session.run(query, parameters), profile=profile)

            def work(tx: Any) -> RunResult:
                return _collect(tx.run(query, parameters), profile=profile)

            result: RunResult = session.execute_write(work)
            return result

    def close(self) -> None:
        """Close the underlying driver."""
        self._driver.close()

    def __enter__(self) -> Neo4jRunner:
        """Return this runner."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the driver on leaving the block."""
        self.close()
