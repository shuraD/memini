"""T-109 generic-only mode: the export gate's leak check that public CI can run.

Requirement: vp-rnd 2026-11-14 for the owner. Public CI never has the private deny file, so
``--generic-only`` runs the built-in detectors without one (email addresses except the allow
list, RFC 1918 addresses, home-directory paths). Public PRs still get a leak check. The full
deny-file scan runs locally (act, before a snapshot or release) and in the egress gateway.

The public allow list, ``tools/ci/export-allow.txt``, holds only strings that are already public:
the published security address (owner T-106) and fake test identities.

Tests-Author: Claude/devops-498d4ec2 (new tests for its own change, owner rule 2026-11-14).
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_ALLOW = ROOT / "tools" / "ci" / "export-allow.txt"
GIT = shutil.which("git") or "git"

Gate = Callable[[list[str]], int]


@pytest.fixture
def gate() -> Gate:
    module = importlib.import_module("tools.ci.export_gate")
    main: Gate = module.main
    return main


def _tree(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return root


def _run(gate: Gate, capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, str]:
    code = gate(argv)
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def test_generic_only_needs_no_deny_file_and_passes_a_clean_tree(
    gate: Gate, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tree = _tree(tmp_path / "tree", {"README.md": "# Title\n"})
    code, out = _run(gate, capsys, ["--generic-only", str(tree)])
    assert code == 0, out


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("mail a.person@example.com\n", "email"),
        ("bolt://10.0.0.7:7687\n", "private-ip"),
        ("/home/someone-else/x\n", "home-path"),
    ],
)
def test_generic_only_still_flags_personal_data(
    gate: Gate, tmp_path: Path, capsys: pytest.CaptureFixture[str], *, text: str, label: str
) -> None:
    tree = _tree(tmp_path / "tree", {"notes.md": text})
    code, out = _run(gate, capsys, ["--generic-only", str(tree)])
    assert code == 1, out
    assert label in out, out


def test_generic_only_honours_the_allow_list(
    gate: Gate, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    allow = tmp_path / "allow.txt"
    allow.write_text("published@example.org\n")
    tree = _tree(tmp_path / "tree", {"SECURITY.md": "Report to published@example.org.\n"})
    code, out = _run(gate, capsys, ["--generic-only", "--allow", str(allow), str(tree)])
    assert code == 0, out


def test_generic_only_and_deny_are_mutually_exclusive(
    gate: Gate, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One mode per run: a full scan never silently drops its deny file."""
    deny = tmp_path / "deny.txt"
    deny.write_text("x\n")
    tree = _tree(tmp_path / "tree", {"README.md": "# Title\n"})
    with pytest.raises(SystemExit) as exc:
        gate(["--generic-only", "--deny", str(deny), str(tree)])
    assert exc.value.code == 2


def test_public_allow_list_holds_only_public_strings() -> None:
    """The tracked allow list is public. It may hold emails, addresses and paths, never a
    pattern, and every entry is a literal that a public test or SECURITY.md already shows."""
    assert PUBLIC_ALLOW.is_file(), PUBLIC_ALLOW
    entries = [
        line.strip()
        for line in PUBLIC_ALLOW.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert entries, "the public allow list is empty"
    tracked = subprocess.run(  # noqa: S603 — test-only, fixed git args
        [GIT, "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
    ).stdout.split("\0")
    corpus = "".join(
        (ROOT / name).read_text(encoding="utf-8", errors="ignore")
        for name in tracked
        if name and name != "tools/ci/export-allow.txt" and (ROOT / name).is_file()
    )
    unused = [entry for entry in entries if entry not in corpus]
    assert not unused, f"allow-list entries that no tracked file uses (remove them): {unused}"


def test_the_tracked_tree_passes_the_generic_scan(
    gate: Gate, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """What public CI runs: every tracked file, generic detectors, public allow list."""
    export = tmp_path / "export"
    names = subprocess.run(  # noqa: S603 — test-only, fixed git args
        [GIT, "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
    ).stdout.split("\0")
    for name in filter(None, names):
        source = ROOT / name
        if source.is_file():
            target = export / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    code, out = _run(gate, capsys, ["--generic-only", "--allow", str(PUBLIC_ALLOW), str(export)])
    assert code == 0, out
