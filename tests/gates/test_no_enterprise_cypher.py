"""Gate: src/ uses no Enterprise-only Neo4j / GDS features (T-053, owner T-052).

Owner decision T-052: CI and public use Neo4j Community + GDS Community, and code runs with
``database=None`` (the default database). So no file in ``src/`` may use an Enterprise-only
construct: multiple databases or RBAC as a requirement, existence/key/type constraints, or a
GDS Enterprise procedure. The Community forms are NOT flagged: a ``WHERE ... IS NOT NULL``
predicate, a uniqueness constraint (``IS UNIQUE``), ``database=None``, the ``NEO4J_DATABASE`` /
``NEO4J_USER`` environment variables, and GDS Community procedures.

The rule flags each Enterprise-only construct in a planted file and passes on main.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GIT = shutil.which("git") or "git"
_SRC = "src/"
_SCAN_SUFFIXES = (".py", ".cypher", ".cql", ".yaml", ".yml")

# (category, pattern). Each matches an ENTERPRISE-only construct, never its Community form.
_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "multi-database",
        re.compile(
            r"\b(?:CREATE|DROP|START|STOP)\s+(?:OR\s+REPLACE\s+)?(?:COMPOSITE\s+)?DATABASE\b", re.I
        ),
    ),
    ("rbac", re.compile(r"\b(?:CREATE|DROP)\s+(?:OR\s+REPLACE\s+)?(?:ROLE|USER)\b", re.I)),
    ("rbac", re.compile(r"\b(?:GRANT|DENY|REVOKE)\b[\w .*]*\b(?:ON|TO|FROM)\b\s+\w", re.I)),
    ("existence-constraint", re.compile(r"\bREQUIRE\b[^;\n]*\bIS\s+NOT\s+NULL\b", re.I)),
    ("key-constraint", re.compile(r"\bIS\s+(?:NODE|RELATIONSHIP|REL)\s+KEY\b", re.I)),
    ("type-constraint", re.compile(r"\bIS\s*::\s*\w", re.I)),
    (
        "gds-enterprise",
        re.compile(
            r"\bgds\.(?:model\.(?:store|publish|load|delete|exists)|backup|restore|systemMonitor)\b",
            re.I,
        ),
    ),
]


def scan(text: str) -> set[str]:
    """Return the set of Enterprise-only construct categories found in ``text``."""
    return {name for name, pattern in _RULES if pattern.search(text)}


def _tracked_src_files() -> list[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    result = subprocess.run(  # noqa: S603 — test-only, fixed git args
        [GIT, "ls-files", "-z", _SRC],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return [n for n in result.stdout.split("\0") if n and n.endswith(_SCAN_SUFFIXES)]


# --- the rule flags each Enterprise-only construct -------------------------------------------

_ENTERPRISE_SAMPLES: list[tuple[str, str]] = [
    ("multi-database", "CREATE DATABASE analytics"),
    ("multi-database", "CREATE COMPOSITE DATABASE federation"),
    ("multi-database", "DROP DATABASE old IF EXISTS"),
    ("rbac", "CREATE ROLE reader"),
    ("rbac", "CREATE USER alice SET PASSWORD 'x'"),
    ("rbac", "GRANT TRAVERSE ON GRAPH neo4j NODES * TO reader"),
    ("rbac", "REVOKE ROLE reader FROM alice"),
    ("existence-constraint", "CREATE CONSTRAINT c FOR (n:Doc) REQUIRE n.id IS NOT NULL"),
    ("key-constraint", "CREATE CONSTRAINT c FOR (n:Doc) REQUIRE n.id IS NODE KEY"),
    ("key-constraint", "CREATE CONSTRAINT c FOR ()-[r:R]-() REQUIRE r.id IS RELATIONSHIP KEY"),
    ("type-constraint", "CREATE CONSTRAINT c FOR (n:Doc) REQUIRE n.id IS :: STRING"),
    ("gds-enterprise", "CALL gds.model.store('m')"),
    ("gds-enterprise", "CALL gds.backup()"),
]


@pytest.mark.parametrize(("category", "snippet"), _ENTERPRISE_SAMPLES)
def test_enterprise_construct_is_flagged(category: str, snippet: str) -> None:
    """Each Enterprise-only construct in a planted snippet is flagged with its category."""
    assert category in scan(snippet), (snippet, scan(snippet))


# --- the Community forms are NOT flagged (no false alarm) ------------------------------------

_COMMUNITY_SAFE: list[str] = [
    "MATCH (n:Doc) WHERE n.id IS NOT NULL RETURN n",  # a predicate, not an existence constraint
    "CREATE CONSTRAINT c FOR (n:Doc) REQUIRE n.id IS UNIQUE",  # uniqueness is Community
    "CALL gds.graph.project('g', 'Node', 'REL')",  # GDS Community
    "CALL gds.pageRank.stream('g')",  # GDS Community
    "GraphDatabase.driver(uri, auth=(user, pw))  # database=None by default",
    'os.environ["NEO4J_DATABASE"]',
    'os.environ["NEO4J_USER"]',
    "MATCH (d:Document {scope: $scope}) RETURN d.id",
]


@pytest.mark.parametrize("snippet", _COMMUNITY_SAFE)
def test_community_construct_is_not_flagged(snippet: str) -> None:
    """A Community-safe construct is never flagged, so the gate does not block valid code."""
    assert scan(snippet) == set(), (snippet, scan(snippet))


# --- src/ on main is clean ------------------------------------------------------------------


def test_src_has_no_enterprise_only_construct() -> None:
    """No file under src/ uses an Enterprise-only construct (owner T-052: Community only)."""
    offenders: dict[str, set[str]] = {}
    for name in _tracked_src_files():
        try:
            text = (ROOT / name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        found = scan(text)
        if found:
            offenders[name] = found
    assert not offenders, f"Enterprise-only constructs in src/: {offenders}"
