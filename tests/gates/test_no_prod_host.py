"""Gate: the production Neo4j host never appears as a literal anywhere in the repo.

Owner decision 2026-11-13 (T-060, relayed by VP R&D): the production Neo4j host lives ONLY
in the environment variable ``MEMINI_PROD_NEO4J_HOST``; it is never hardcoded. This gate
reads that host from the environment and scans every tracked file for it, failing if any file
contains the literal.

- Documentation-only addresses (RFC 5737: 192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24) are
  the fixed non-local examples a file may use instead of the real host; they are never the
  production host, so they are not flagged.
- When ``MEMINI_PROD_NEO4J_HOST`` is unset the scan case is not generated: the test runs and
  passes without asserting on a host. It is never ``skip``ped (no hidden gap).

Task T-065 (QA). Acceptance: on main today this fails on AGENTS.md and tests/unit/test_schema.py,
which still carry the production host literal; it passes once the literal is gone (T-060 impl).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_THIS = Path(__file__).resolve().relative_to(ROOT).as_posix()
GIT = shutil.which("git") or "git"

# RFC 5737 documentation ranges: allowed non-local examples, never the production host.
RFC5737_DOC_PREFIXES = ("192.0.2.", "198.51.100.", "203.0.113.")


def _tracked_files() -> list[str]:
    """Every file git tracks, with all GIT_* variables stripped from the subprocess env."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    result = subprocess.run(  # noqa: S603 — test-only, fixed git args
        [GIT, "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return [name for name in result.stdout.split("\0") if name]


def test_production_neo4j_host_is_never_a_literal_in_the_repo() -> None:
    """No tracked file carries the production host as a literal; it comes only from the env."""
    host = os.environ.get("MEMINI_PROD_NEO4J_HOST", "").strip()
    if not host:
        # The variable is unset: the scan case is not generated (never a skip).
        return
    assert not host.startswith(RFC5737_DOC_PREFIXES), (
        f"MEMINI_PROD_NEO4J_HOST={host!r} is an RFC 5737 documentation address, not a real "
        "production host; set it to the real host so this gate protects it"
    )
    offenders: list[str] = []
    for name in _tracked_files():
        if name == _THIS:
            # The gate reads the host from the environment; it never hardcodes it.
            continue
        try:
            text = (ROOT / name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if host in text:
            offenders.append(name)
    assert not offenders, (
        f"production Neo4j host {host!r} is hardcoded in: {offenders}. "
        "Read it from MEMINI_PROD_NEO4J_HOST; use an RFC 5737 address for non-local examples."
    )
