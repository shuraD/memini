"""The Neo4j graph store: guarded connection settings and append-only node writes.

:class:`GraphStore` takes its settings from the constructor or the environment, refuses any
host that is neither local nor allow-listed (:func:`is_local_uri`), connects fail-fast, ensures
the ``(scope, id)`` uniqueness constraints, writes nodes in one ``UNWIND`` batch per call
(``MERGE`` on ``(scope, id)``, skipping an already-stored ``content_hash``) and supersedes an
old node by a new one without deleting it.

The design follows the old memini's ``connect``/``remember``; the code is new.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import neo4j

from memini.infra.graph.schema import GraphSchema, load_schema

# Environment variables and defaults of the connection settings (spec GR-03).
_ENV_URI, _ENV_USER, _ENV_PASSWORD, _ENV_DATABASE = (
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
)
_DEFAULT_URI = "bolt://localhost:7687"
_DEFAULT_USER = "neo4j"
# Hosts that are always this machine (compared case-insensitively, no DNS lookup).
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# Graph names the store relies on (spec GR-03); the schema file declares each of them.
_SCOPE = "scope"
_ID = "id"
_HASH = "content_hash"
_STATUS = "status"
_UPDATED_BY = "updated_by"
_UPDATED_AT = "updated_at"
_ACTIVE = "active"
_SUPERSEDED = "superseded"
_SUPERSEDES = "SUPERSEDES"


class UnsafeHostError(ValueError):
    """Raised when a URI's host is neither local nor allow-listed."""


def is_local_uri(uri: str, allowed_hosts: Iterable[str] = ()) -> bool:
    """Return True when ``uri``'s host is local or one of ``allowed_hosts`` (case-insensitive)."""
    host = (urlsplit(uri).hostname or "").lower()
    if not host:
        return False
    return host in _LOCAL_HOSTS or host in {h.lower() for h in allowed_hosts}


def _setting(value: str | None, env: str, default: str) -> str:
    """Return ``value`` when it is not None, else the env var ``env``, else ``default``."""
    if value is not None:
        return value
    return os.environ.get(env, default)


class GraphStore:
    """A Neo4j store of scoped nodes, written append-only."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        *,
        allowed_hosts: Iterable[str] = (),
    ) -> None:
        """Resolve the settings and refuse an unsafe host; no I/O."""
        self.uri: str = _setting(uri, _ENV_URI, _DEFAULT_URI)
        self.user: str = _setting(user, _ENV_USER, _DEFAULT_USER)
        self.password: str = _setting(password, _ENV_PASSWORD, "")
        self.database: str | None = _setting(database, _ENV_DATABASE, "") or None
        if not is_local_uri(self.uri, allowed_hosts):
            raise UnsafeHostError(f"refusing non-local Neo4j host in {self.uri!r}")
        self._driver: neo4j.Driver | None = None
        self._schema: GraphSchema | None = None

    def connect(self) -> GraphStore:
        """Open the driver, verify connectivity, ensure the (scope, id) constraints."""
        driver = neo4j.GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._driver = driver
        driver.verify_connectivity()
        schema = load_schema()
        for label, props in schema.nodes.items():
            if _ID in props:
                driver.execute_query(
                    f"CREATE CONSTRAINT {label.lower()}_{_SCOPE}_{_ID} IF NOT EXISTS "
                    f"FOR (n:`{label}`) REQUIRE (n.`{_SCOPE}`, n.`{_ID}`) IS UNIQUE",
                    database_=self.database,
                )
        self._schema = schema
        return self

    def close(self) -> None:
        """Close the driver, if open."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def _checked(self, label: str) -> tuple[neo4j.Driver, tuple[str, ...]]:
        """Return the open driver and ``label``'s properties; ValueError for an unknown label."""
        if self._driver is None or self._schema is None:
            raise RuntimeError("GraphStore is not connected; call connect() first")
        if label not in self._schema.nodes:
            raise ValueError(f"unknown node label {label!r}")
        return self._driver, self._schema.nodes[label]

    def write(
        self, label: str, scope: str, items: Sequence[Mapping[str, Any]], *, updated_by: str
    ) -> list[str]:
        """Merge ``items`` as ``label`` nodes of ``scope``; return the written ids in order."""
        driver, props = self._checked(label)
        status = f", n.`{_STATUS}` = $status" if _STATUS in props else ""
        records, _, _ = driver.execute_query(
            f"UNWIND $rows AS row "
            f"WITH row WHERE NOT EXISTS {{ MATCH (m:`{label}`) "
            f"WHERE m.`{_SCOPE}` = $scope AND m.`{_HASH}` = row.`{_HASH}` }} "
            f"MERGE (n:`{label}` {{`{_SCOPE}`: $scope, `{_ID}`: row.`{_ID}`}}) "
            f"SET n += row, n.`{_UPDATED_BY}` = $updated_by, "
            f"n.`{_UPDATED_AT}` = datetime(){status} "
            f"RETURN row.`{_ID}` AS id",
            {
                "rows": [dict(item) for item in items],
                "scope": scope,
                "updated_by": updated_by,
                "status": _ACTIVE,
            },
            database_=self.database,
        )
        written = {record["id"] for record in records}
        return [str(item[_ID]) for item in items if item[_ID] in written]

    def supersede(
        self, label: str, scope: str, new_id: str, old_id: str, *, updated_by: str
    ) -> None:
        """Link ``new`` -[:SUPERSEDES]-> ``old`` and mark ``old`` superseded; keep both."""
        driver, _ = self._checked(label)
        records, _, _ = driver.execute_query(
            f"MATCH (new:`{label}` {{`{_SCOPE}`: $scope, `{_ID}`: $new_id}}) "
            f"MATCH (old:`{label}` {{`{_SCOPE}`: $scope, `{_ID}`: $old_id}}) "
            f"MERGE (new)-[:`{_SUPERSEDES}`]->(old) "
            f"SET old.`{_STATUS}` = $status, old.`{_UPDATED_BY}` = $updated_by, "
            f"old.`{_UPDATED_AT}` = datetime() "
            f"RETURN count(*) AS linked",
            {
                "scope": scope,
                "new_id": new_id,
                "old_id": old_id,
                "status": _SUPERSEDED,
                "updated_by": updated_by,
            },
            database_=self.database,
        )
        if not records or not records[0]["linked"]:
            raise LookupError(f"{label} {new_id!r} or {old_id!r} not found in scope {scope!r}")
