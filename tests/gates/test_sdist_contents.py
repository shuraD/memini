"""Gate: the built sdist ships only the package and its public files (T-099).

Finding (devops-cicd via VP R&D, 2026-11-14): hatchling's DEFAULT sdist ships the whole
project — reviews/, ledgers/, .githooks/, CLAUDE.md, MERGE-CHECKLIST.md and more. A public
release must contain only the package and the files a user needs. This gate builds the sdist
and checks its contents.

It builds with ``uv build --sdist`` (offline — the build backend is already cached). Fails on
main today (the private files are present); passes once the sdist is restricted to /src,
/LICENSE, /README.md, /pyproject.toml and config/ (T-082).
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UV = shutil.which("uv") or "uv"

# Must NOT ship: private process artifacts.
FORBIDDEN_PREFIXES = ("reviews/", "ledgers/", ".githooks/")
FORBIDDEN_NAMES = ("CLAUDE.md", "MERGE-CHECKLIST.md")
# Must ship: the package and the files a user needs.
REQUIRED_FILES = ("LICENSE", "pyproject.toml", "config/graph_schema.yaml")
REQUIRED_PREFIX = "src/memini/"


@pytest.fixture(scope="module")
def sdist_members(tmp_path_factory: pytest.TempPathFactory) -> set[str]:
    """Build the sdist and return its member paths, each without the top-level dist dir."""
    out = tmp_path_factory.mktemp("sdist")
    subprocess.run(  # noqa: S603 — test-only, fixed uv args
        [UV, "build", "--sdist", "--out-dir", str(out)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    archives = list(out.glob("*.tar.gz"))
    assert len(archives) == 1, f"expected one sdist, got {archives}"
    with tarfile.open(archives[0]) as tar:
        names = tar.getnames()
    # Every member is under a single "<name>-<version>/" directory; drop that prefix.
    return {name.split("/", 1)[1] for name in names if "/" in name}


@pytest.mark.parametrize("prefix", FORBIDDEN_PREFIXES)
def test_sdist_excludes_private_directories(sdist_members: set[str], prefix: str) -> None:
    """No private process directory is shipped in the sdist."""
    leaked = sorted(m for m in sdist_members if m.startswith(prefix))
    assert not leaked, f"sdist ships {prefix!r}: {leaked}"


@pytest.mark.parametrize("name", FORBIDDEN_NAMES)
def test_sdist_excludes_private_files(sdist_members: set[str], name: str) -> None:
    """No private top-level/doc file (CLAUDE.md, MERGE-CHECKLIST.md) is shipped."""
    leaked = sorted(m for m in sdist_members if Path(m).name == name)
    assert not leaked, f"sdist ships {name!r}: {leaked}"


@pytest.mark.parametrize("required", REQUIRED_FILES)
def test_sdist_includes_required_files(sdist_members: set[str], required: str) -> None:
    """The sdist ships the public files a user needs."""
    assert required in sdist_members, f"sdist is missing {required!r}"


def test_sdist_includes_the_package(sdist_members: set[str]) -> None:
    """The sdist ships the package source under src/memini/."""
    assert any(m.startswith(REQUIRED_PREFIX) for m in sdist_members), (
        f"sdist ships nothing under {REQUIRED_PREFIX!r}"
    )
