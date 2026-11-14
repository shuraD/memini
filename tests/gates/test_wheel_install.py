"""T-080: an installed wheel finds its configuration without the repo checkout.

Requirement (pool task T-080, private repo; implemented by T-082): build the wheel, install
it into a clean virtualenv outside the repository, and load the schema, the prompts and the
flags configuration from there. Fails today: ``memini.infra.paths`` resolves ``config/`` as
``parents[3]`` of its own file, which is the repository in a checkout and nothing at all in
an installed package (review b0bacbd NOTE; bug row from T-027). T-082 note: the
``MEMINI_CONFIG_DIR`` override keeps working.

Owner rules behind it: prompts live in ``config/prompts/`` (AGENTS.md rule 3), the schema is
data in ``config/graph_schema.yaml`` (SPEC-graph-schema decision 1), and every flag lives in
one registry, ``config/flags.yaml`` (SPEC-feature-flags decision 1). An installed Memini that
cannot see them runs without its schema, its prompts or its flag defaults.

``config/flags.yaml`` does not exist on ``main`` yet, so flags are covered by the mirror
test: every file under the repository's ``config/`` must be present, byte for byte, in the
installed configuration. The day the registry lands, it is guarded with no edit here.

How: the module builds the wheel once with ``uv build``, creates a fresh virtualenv in a
temporary directory with ``uv venv`` and installs the wheel with ``uv pip install`` (offline,
from the cache that ``uv sync`` filled). Each probe runs that venv's python from a working
directory outside the repository, with every ``GIT_*``, ``PYTHON*``, ``VIRTUAL_ENV`` and
``MEMINI_*`` variable removed, and prints what ``memini.infra.paths`` resolves as JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPO_CONFIG = ROOT / "config"
_DROP_PREFIXES = ("GIT_", "PYTHON", "VIRTUAL_ENV", "MEMINI_", "UV_PROJECT")

_PROBE = r"""
import hashlib, json, sys
from pathlib import Path
import memini
from memini.infra import paths
from memini.core import schema

def tree(root):
    if not root.is_dir():
        return {}
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }

out = {
    "package": str(Path(memini.__file__).resolve()),
    "prefix": str(Path(sys.prefix).resolve()),
    "config_dir": str(paths.config_dir().resolve()),
    "schema_file": str(paths.schema_file().resolve()),
    "schema_exists": paths.schema_file().is_file(),
    "prompts_dir": str(paths.prompts_dir().resolve()),
    "config_tree": tree(paths.config_dir()),
}
try:
    out["schema_problems"] = schema.validate(schema.load())
except Exception as exc:  # reported, not raised: the test states what failed
    out["schema_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(out))
"""


def _clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_DROP_PREFIXES)}
    env.update(extra)
    return env


def _run(args: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — test-only, fixed tool and arguments
        args, cwd=cwd, env=env, capture_output=True, text=True, timeout=600, check=False
    )


def _tree(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _inside(path: str | Path, root: Path) -> bool:
    return Path(path).resolve().is_relative_to(root.resolve())


@pytest.fixture(scope="module")
def venv_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fresh virtualenv, outside the repository, with only the built wheel installed."""
    uv = shutil.which("uv")
    assert uv, "uv is a declared tool dependency (CI installs it); it is not on PATH"
    work = tmp_path_factory.mktemp("wheel-install")
    assert not _inside(work, ROOT), work
    env = _clean_env()
    dist = work / "dist"
    built = _run([uv, "build", "--wheel", "--out-dir", str(dist), str(ROOT)], work, env)
    assert built.returncode == 0, built.stderr
    wheels = sorted(dist.glob("*.whl"))
    assert len(wheels) == 1, wheels
    venv = work / "venv"
    # the same interpreter as the test run, so the venv meets requires-python wherever it runs
    made = _run([uv, "venv", "--quiet", "--python", sys.executable, str(venv)], work, env)
    assert made.returncode == 0, made.stderr
    python = venv / "bin" / "python"
    installed = _run(
        [uv, "pip", "install", "--offline", "--python", str(python), str(wheels[0])], work, env
    )
    assert installed.returncode == 0, installed.stderr
    return python


def _probe(python: Path, cwd: Path, **extra: str) -> dict[str, Any]:
    res = _run([str(python), "-c", _PROBE], cwd, _clean_env(**extra))
    assert res.returncode == 0, res.stderr
    data: dict[str, Any] = json.loads(res.stdout)
    return data


@pytest.fixture(scope="module")
def outside(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A working directory outside the repository, with a decoy ``config/`` in it."""
    cwd = tmp_path_factory.mktemp("elsewhere")
    decoy = cwd / "config"
    (decoy / "prompts").mkdir(parents=True)
    (decoy / "graph_schema.yaml").write_text("nodes: {Decoy: {rule: not the package}}\n")
    (decoy / "prompts" / "decoy.txt").write_text("not the package's prompt\n")
    return cwd


@pytest.fixture(scope="module")
def installed(venv_python: Path, outside: Path) -> dict[str, Any]:
    return _probe(venv_python, outside)


# --------------------------------------------------------------------------------------------
# Harness self-checks: the probe really runs the installed wheel, away from the checkout
# --------------------------------------------------------------------------------------------


def test_probe_imports_memini_from_the_clean_venv_not_the_checkout(
    installed: dict[str, Any],
) -> None:
    """Harness: ``memini`` is imported from the fresh venv, never from the repository's
    ``src/`` (otherwise the checkout would hide the bug)."""
    assert _inside(installed["package"], Path(installed["prefix"])), installed["package"]
    assert not _inside(installed["package"], ROOT), installed["package"]


# --------------------------------------------------------------------------------------------
# The installed package finds its configuration
# --------------------------------------------------------------------------------------------


def test_installed_config_dir_is_inside_the_installed_package_not_the_repo(
    installed: dict[str, Any],
) -> None:
    """T-080 / T-082 ("config/ ships inside the package"): without ``MEMINI_CONFIG_DIR`` the
    installed package resolves its configuration inside its own installed package directory.
    Not the repository checkout, not a sibling of site-packages, and not a ``config/`` in
    whatever directory it happens to run from."""
    config_dir = installed["config_dir"]
    package_dir = Path(installed["package"]).parent
    assert _inside(config_dir, package_dir), (config_dir, str(package_dir))
    assert not _inside(config_dir, ROOT), config_dir


def test_installed_schema_loads_and_validates(installed: dict[str, Any]) -> None:
    """SPEC-graph-schema decision 1: the schema is data. The installed package finds
    ``graph_schema.yaml``, loads it and it validates with no problem."""
    assert installed["schema_exists"], installed["schema_file"]
    assert "schema_error" not in installed, installed.get("schema_error")
    assert installed["schema_problems"] == [], installed["schema_problems"]


def test_installed_schema_is_the_repositorys_schema(installed: dict[str, Any]) -> None:
    """The schema the installed package loads is byte for byte the committed one, never a
    decoy ``config/graph_schema.yaml`` next to the working directory."""
    digest = hashlib.sha256((REPO_CONFIG / "graph_schema.yaml").read_bytes()).hexdigest()
    assert installed["config_tree"].get("graph_schema.yaml") == digest, installed["config_tree"]


def test_installed_prompts_dir_holds_every_committed_prompt(installed: dict[str, Any]) -> None:
    """AGENTS.md rule 3: prompts live in ``config/prompts/``. The installed prompts
    directory is inside the installed configuration and holds every committed prompt file,
    byte for byte, and nothing else (no decoy prompt from the working directory)."""
    assert _inside(installed["prompts_dir"], Path(installed["config_dir"])), installed
    expected = {
        name: digest for name, digest in _tree(REPO_CONFIG).items() if name.startswith("prompts/")
    }
    got = {
        name: digest
        for name, digest in installed["config_tree"].items()
        if name.startswith("prompts/")
    }
    assert got == expected


def test_installed_config_mirrors_every_committed_config_file(installed: dict[str, Any]) -> None:
    """T-080 "flags, schema and prompts": every file under the repository's ``config/``
    (the flag registry ``flags.yaml`` included, once it exists) ships with the package and
    is found at the installed configuration directory, byte for byte. Nothing is missing,
    nothing is changed."""
    expected = _tree(REPO_CONFIG)
    assert expected, "the repository has no config/ files to compare"
    got = installed["config_tree"]
    missing = sorted(set(expected) - set(got))
    changed = sorted(n for n in expected if n in got and got[n] != expected[n])
    assert (missing, changed) == ([], []), {"missing": missing, "changed": changed}


# --------------------------------------------------------------------------------------------
# The override keeps working in the installed package
# --------------------------------------------------------------------------------------------


def test_memini_config_dir_still_overrides_in_the_installed_package(
    venv_python: Path, outside: Path, tmp_path: Path
) -> None:
    """T-082 note: ``MEMINI_CONFIG_DIR`` keeps working as the override, installed too.
    Every location (config, schema, prompts) follows it."""
    override = tmp_path / "override-config"
    shutil.copytree(REPO_CONFIG, override)
    got = _probe(venv_python, outside, MEMINI_CONFIG_DIR=str(override))
    assert Path(got["config_dir"]) == override.resolve()
    assert Path(got["schema_file"]) == (override / "graph_schema.yaml").resolve()
    assert Path(got["prompts_dir"]) == (override / "prompts").resolve()
    assert got.get("schema_problems") == [], got
