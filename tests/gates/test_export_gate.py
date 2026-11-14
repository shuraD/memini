"""T-107: the export gate refuses any private string in a tree or in built distributions.

Requirement: ADR-0002 (accepted 2026-11-14) decisions 3, 4 and 6. Nothing reaches the public repo
or PyPI unless the export gate (``tools.ci.export_gate``, T-109) passes. The deny patterns are
private data, read from a file in the private repo, never from the public tree. Owner decision
T-106: the dedicated security address is allowed through an allow-list file. Every other email
address is still flagged.

Contract tested here (black box, through ``main(argv) -> int``):
- ``--deny FILE`` holds one regular expression per line; blank lines and ``#`` comments are
  ignored. A missing or empty deny file fails loudly (exit 2): a gate with no patterns proves
  nothing.
- ``--allow FILE`` (optional) holds literal strings that may appear even where a built-in
  detector would flag them.
- Built-in detectors, generic and not private data: email addresses, private-network (RFC 1918)
  IPv4 addresses, and home-directory paths. RFC 5737 documentation addresses are not flagged.
- Targets are directories (``.git/`` is skipped) and ``.whl`` / ``.tar.gz`` files, scanned member
  by member.
- Exit 0 when clean, 1 on findings. A finding names the file, line and rule, and **never prints
  the matched text**, because CI logs are public.

Tests-Author: Claude/devops-498d4ec2 (new tests for its own change, owner rule 2026-11-14).
"""

from __future__ import annotations

import importlib
import io
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

# A made-up private marker. It is not a real secret, and appears only in this test file.
PRIVATE = "zq-private-marker-7f3a"
ALLOWED_EMAIL = "security-contact@example.org"

Gate = Callable[[list[str]], int]


@pytest.fixture
def gate() -> Gate:
    """The gate's entry point. Each test fails on its own while the module doesn't exist."""
    module = importlib.import_module("tools.ci.export_gate")
    main: Gate = module.main
    return main


@pytest.fixture
def deny(tmp_path: Path) -> Path:
    path = tmp_path / "private" / "deny.txt"
    path.parent.mkdir()
    path.write_text("# private deny patterns (test copy)\n\nzq-private-marker-[0-9a-f]+\n")
    return path


@pytest.fixture
def allow(tmp_path: Path) -> Path:
    path = tmp_path / "private" / "allow.txt"
    path.parent.mkdir(exist_ok=True)
    path.write_text(f"# allowed public strings\n{ALLOWED_EMAIL}\n")
    return path


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


# --------------------------------------------------------------------------------------------
# The deny-pattern file is required
# --------------------------------------------------------------------------------------------


def test_missing_deny_file_fails_loudly(
    gate: Gate, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tree = _tree(tmp_path / "tree", {"README.md": "hello\n"})
    code, out = _run(gate, capsys, ["--deny", str(tmp_path / "absent.txt"), str(tree)])
    assert code == 2, out
    assert "deny" in out.lower(), out


def test_empty_deny_file_fails_loudly(
    gate: Gate, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "deny.txt"
    empty.write_text("# only a comment\n\n")
    tree = _tree(tmp_path / "tree", {"README.md": "hello\n"})
    code, out = _run(gate, capsys, ["--deny", str(empty), str(tree)])
    assert code == 2, out


def test_deny_option_is_required(
    gate: Gate, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """There is no default inside the public tree: patterns never come from it."""
    tree = _tree(tmp_path / "tree", {"README.md": "hello\n"})
    with pytest.raises(SystemExit) as exc:
        gate([str(tree)])
    assert exc.value.code == 2


# --------------------------------------------------------------------------------------------
# The tree
# --------------------------------------------------------------------------------------------


def test_clean_tree_passes(
    gate: Gate, deny: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tree = _tree(
        tmp_path / "tree",
        {"src/pkg/__init__.py": '"""A package."""\n', "README.md": "# Title\n"},
    )
    code, out = _run(gate, capsys, ["--deny", str(deny), str(tree)])
    assert code == 0, out


def test_planted_private_string_in_the_tree_fails_without_printing_it(
    gate: Gate, deny: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tree = _tree(
        tmp_path / "tree",
        {"docs/notes.md": f"line one\nsee {PRIVATE} here\n", "README.md": "# Title\n"},
    )
    code, out = _run(gate, capsys, ["--deny", str(deny), str(tree)])
    assert code == 1, out
    assert "docs/notes.md:2" in out, out
    assert PRIVATE not in out, "the gate printed the private string into a (public) log"


def test_git_directory_is_not_scanned(
    gate: Gate, deny: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tree = _tree(tmp_path / "tree", {".git/config": PRIVATE, "README.md": "# Title\n"})
    code, out = _run(gate, capsys, ["--deny", str(deny), str(tree)])
    assert code == 0, out


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("contact me: someone.private@example.com\n", "email"),
        ("db at bolt://10.1.2.3:7687\n", "private-ip"),
        ("db at bolt://172.20.0.5:7687\n", "private-ip"),
        ("db at bolt://192.168.7.9:7687\n", "private-ip"),
        ("corpus in /Users/someone/work/corpus\n", "home-path"),
        ("corpus in /home/someone/corpus\n", "home-path"),
    ],
)
def test_builtin_detectors_flag_personal_data(
    gate: Gate,
    deny: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    *,
    text: str,
    label: str,
) -> None:
    tree = _tree(tmp_path / "tree", {"AGENTS.md": text})
    code, out = _run(gate, capsys, ["--deny", str(deny), str(tree)])
    assert code == 1, out
    assert label in out, out


def test_documentation_addresses_are_not_flagged(
    gate: Gate, deny: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """RFC 5737 TEST-NET addresses are the documented non-local examples (T-065)."""
    tree = _tree(
        tmp_path / "tree",
        {
            "tests/t.py": "URIS = ['bolt://192.0.2.1', 'bolt://198.51.100.2', 'bolt://203.0.113.10']\n"
        },
    )
    code, out = _run(gate, capsys, ["--deny", str(deny), str(tree)])
    assert code == 0, out


def test_allow_listed_email_passes_and_others_still_fail(
    gate: Gate, deny: Path, allow: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ok = _tree(tmp_path / "ok", {"SECURITY.md": f"Report to {ALLOWED_EMAIL}.\n"})
    code, out = _run(gate, capsys, ["--deny", str(deny), "--allow", str(allow), str(ok)])
    assert code == 0, out

    bad = _tree(
        tmp_path / "bad",
        {"SECURITY.md": f"Report to {ALLOWED_EMAIL} or other.person@example.com.\n"},
    )
    code, out = _run(gate, capsys, ["--deny", str(deny), "--allow", str(allow), str(bad)])
    assert code == 1, out
    assert "email" in out, out


def test_allow_list_never_overrides_a_deny_pattern(
    gate: Gate, deny: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The allow list relaxes only the built-in detectors; a private pattern always wins."""
    allow = tmp_path / "allow.txt"
    allow.write_text(f"{PRIVATE}\n")
    tree = _tree(tmp_path / "tree", {"README.md": f"{PRIVATE}\n"})
    code, out = _run(gate, capsys, ["--deny", str(deny), "--allow", str(allow), str(tree)])
    assert code == 1, out


# --------------------------------------------------------------------------------------------
# Built distributions (dist/)
# --------------------------------------------------------------------------------------------


def _wheel(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    return path


def _sdist(path: Path, members: dict[str, str]) -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for name, text in members.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_planted_private_string_in_a_wheel_fails(
    gate: Gate, deny: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    whl = _wheel(
        tmp_path / "pkg-0.1-py3-none-any.whl",
        {"pkg/__init__.py": '"""ok"""\n', "pkg/config/x.yaml": f"k: {PRIVATE}\n"},
    )
    code, out = _run(gate, capsys, ["--deny", str(deny), str(whl)])
    assert code == 1, out
    assert "pkg/config/x.yaml:1" in out, out
    assert PRIVATE not in out


def test_planted_private_string_in_an_sdist_fails(
    gate: Gate, deny: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sdist = _sdist(
        tmp_path / "pkg-0.1.tar.gz",
        {"pkg-0.1/README.md": "# ok\n", "pkg-0.1/docs/x.md": f"{PRIVATE}\n"},
    )
    code, out = _run(gate, capsys, ["--deny", str(deny), str(sdist)])
    assert code == 1, out
    assert "docs/x.md:1" in out, out
    assert PRIVATE not in out


def test_clean_distributions_pass(
    gate: Gate, deny: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    whl = _wheel(tmp_path / "pkg-0.1-py3-none-any.whl", {"pkg/__init__.py": '"""ok"""\n'})
    sdist = _sdist(tmp_path / "pkg-0.1.tar.gz", {"pkg-0.1/README.md": "# ok\n"})
    code, out = _run(gate, capsys, ["--deny", str(deny), str(whl), str(sdist)])
    assert code == 0, out


def test_unknown_target_fails_loudly(
    gate: Gate, deny: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = _run(gate, capsys, ["--deny", str(deny), str(tmp_path / "missing")])
    assert code == 2, out
