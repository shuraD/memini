"""Decide whether a graph database URI may be used.

Local hosts are always allowed, remote hosts only when explicitly allow-listed,
and the production host is always refused. Any URI shape that is not exactly
``scheme://host[:port][/]`` is refused (fail closed).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping

__all__ = ["LOCAL_HOSTS", "UnsafeDatabaseError", "check_uri", "guard_from_env"]

LOCAL_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "::1")

_SCHEMES: frozenset[str] = frozenset(
    {"bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"},
)

_URI_PATTERN = (
    r"(?P<scheme>[a-z0-9+]+)://"
    r"(?:\[(?P<ipv6>[0-9A-Fa-f:.]+)\]|(?P<name>[A-Za-z0-9._-]+))"
    r"(?::(?P<port>[0-9]+))?"
    r"/?"
)

_ENV_PROD_HOST = "MEMINI_PROD_NEO4J_HOST"
_ENV_ALLOWED_HOSTS = "MEMINI_NEO4J_ALLOWED_HOSTS"


class UnsafeDatabaseError(ValueError):
    """Raised when a database URI is refused by the guard."""


def _normalise_host(host: str) -> str:
    """Return ``host`` stripped, lower-cased, with IPv6 brackets removed."""
    value = host.strip().lower()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return value


def check_uri(
    uri: str,
    *,
    allowed_hosts: Iterable[str] = (),
    prod_host: str | None = None,
) -> str:
    """Validate ``uri`` and return its normalised host.

    Args:
        uri: The database URI, of shape ``scheme://host[:port][/]``.
        allowed_hosts: Remote hosts accepted in addition to ``LOCAL_HOSTS``
            (compared case-insensitively).
        prod_host: The production host; refused whenever it matches, even
            when local or allow-listed. ``None`` or blank disables the check.

    Returns:
        The host, lower-case, without IPv6 brackets.

    Raises:
        UnsafeDatabaseError: If the URI shape, scheme or host is not accepted.

    """
    match = re.fullmatch(_URI_PATTERN, uri, re.ASCII)
    if match is None:
        raise UnsafeDatabaseError(f"unexpected database URI shape: {uri!r}")
    if match["scheme"] not in _SCHEMES:
        raise UnsafeDatabaseError(f"unsupported database URI scheme: {match['scheme']!r}")
    host = (match["ipv6"] or match["name"]).lower()

    # A blank prod_host normalises to "" and can never equal the (non-empty) host.
    if prod_host is not None and _normalise_host(prod_host) == host:
        raise UnsafeDatabaseError(f"refusing the production database host: {host!r}")

    if host in LOCAL_HOSTS:
        return host
    if host in {_normalise_host(entry) for entry in allowed_hosts}:
        return host
    raise UnsafeDatabaseError(f"database host is neither local nor allow-listed: {host!r}")


def guard_from_env(uri: str, env: Mapping[str, str] = os.environ) -> str:
    """Validate ``uri`` with the production host and allow-list read from ``env``.

    ``MEMINI_PROD_NEO4J_HOST`` gives the production host (missing or blank means
    none); ``MEMINI_NEO4J_ALLOWED_HOSTS`` is a comma-separated list of remote
    hosts (entries stripped, blanks ignored). Only ``env`` is read.

    Returns:
        The normalised host, as from :func:`check_uri`.

    Raises:
        UnsafeDatabaseError: If the URI is refused.

    """
    prod_host = env.get(_ENV_PROD_HOST, "").strip() or None
    allowed = [entry.strip() for entry in env.get(_ENV_ALLOWED_HOSTS, "").split(",")]
    return check_uri(
        uri,
        allowed_hosts=[entry for entry in allowed if entry],
        prod_host=prod_host,
    )
