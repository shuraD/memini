"""Export gate: nothing private leaves for the public repo or PyPI (ADR-0002 decisions 3, 4, 6).

Run before every public snapshot and every release, and by the host egress gateway before every
push::

    python -m tools.ci.export_gate --deny PRIVATE/deny.txt [--allow ALLOW.txt] TARGET...
    python -m tools.ci.export_gate --generic-only [--allow ALLOW.txt] TARGET...

The full scan (``--deny``) runs locally (act, before a snapshot or release) and in the egress
gateway. ``--generic-only`` runs the built-in detectors without the deny file; public CI uses it,
because it never has the private patterns. The two modes are mutually exclusive.

``TARGET`` is a directory (``.git/`` is skipped), a wheel (``.whl``) or an sdist (``.tar.gz``);
archives are scanned member by member.

- **Deny patterns are private data.** One regular expression per line, with ``#`` comments.
  The file lives in the private repo and is never part of the public tree, so ``--deny`` has
  no default. A missing or empty file is an error (exit 2): a gate with no patterns proves
  nothing.
- **Built-in detectors** are generic, not private data: email addresses, private-network
  (RFC 1918) IPv4 addresses and home-directory paths. RFC 5737 documentation addresses are
  not flagged.
- **Allow list** (``--allow``): literal strings that the built-in detectors may let through,
  such as the published security address. It never overrides a deny pattern.
- **Output never contains the matched text**, because CI logs are public. A finding names
  ``file:line`` and the rule: ``deny#<line in the deny file>`` or the detector name.

Exit codes: 0 clean, 1 findings, 2 usage or configuration error.
"""

from __future__ import annotations

import argparse
import ipaddress
import re
import sys
import tarfile
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_HOME = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/")
# RFC 1918 networks 10/8, 172.16/12 and 192.168/16, written as (integer address, prefix) so this
# file holds no dotted private address for the gate itself to flag.
_RFC1918 = tuple(
    ipaddress.ip_network(net) for net in ((167772160, 8), (2886729728, 12), (3232235520, 16))
)


@dataclass(frozen=True)
class Finding:
    """One hit: where it is and which rule matched (never the matched text)."""

    where: str
    line: int
    rule: str

    def __str__(self) -> str:
        """Render as ``file:line: rule``."""
        return f"{self.where}:{self.line}: {self.rule}"


class GateConfigError(Exception):
    """The gate cannot run as configured (exit 2)."""


def load_deny(path: Path) -> list[tuple[int, re.Pattern[str]]]:
    """Read the private deny patterns; refuse a missing, empty or invalid file."""
    if not path.is_file():
        msg = f"deny-pattern file not found: {path} (it lives in the private repo)"
        raise GateConfigError(msg)
    patterns: list[tuple[int, re.Pattern[str]]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            patterns.append((number, re.compile(line)))
        except re.error as exc:
            msg = f"deny-pattern file {path}, line {number}: invalid regular expression ({exc})"
            raise GateConfigError(msg) from exc
    if not patterns:
        msg = f"deny-pattern file has no patterns: {path}"
        raise GateConfigError(msg)
    return patterns


def load_allow(path: Path | None) -> frozenset[str]:
    """Read the allow list: literal strings, one per line, with ``#`` comments."""
    if path is None:
        return frozenset()
    if not path.is_file():
        msg = f"allow-list file not found: {path}"
        raise GateConfigError(msg)
    lines = (raw.strip() for raw in path.read_text(encoding="utf-8").splitlines())
    return frozenset(line for line in lines if line and not line.startswith("#"))


def _private_ip(text: str) -> bool:
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False
    return any(address in network for network in _RFC1918)


def scan_text(
    text: str,
    where: str,
    deny: list[tuple[int, re.Pattern[str]]],
    allow: frozenset[str],
) -> list[Finding]:
    """Return every finding in ``text``; ``where`` names the file for the report."""
    findings: list[Finding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        findings.extend(
            Finding(where, number, f"deny#{rule}") for rule, pattern in deny if pattern.search(line)
        )
        if any(m.group(0) not in allow for m in _EMAIL.finditer(line)):
            findings.append(Finding(where, number, "email"))
        if any(_private_ip(m.group(0)) and m.group(0) not in allow for m in _IPV4.finditer(line)):
            findings.append(Finding(where, number, "private-ip"))
        if any(m.group(0) not in allow for m in _HOME.finditer(line)):
            findings.append(Finding(where, number, "home-path"))
    return findings


def _decode(data: bytes) -> str | None:
    """Text content of a file, or None for binary data (NUL bytes or not UTF-8)."""
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _iter_target(target: Path) -> Iterator[tuple[str, bytes]]:
    """Yield ``(name, bytes)`` for every file in a directory, wheel or sdist."""
    if target.is_dir():
        for path in sorted(target.rglob("*")):
            relative = path.relative_to(target)
            if path.is_file() and ".git" not in relative.parts:
                yield relative.as_posix(), path.read_bytes()
    elif target.is_file() and target.name.endswith(".whl"):
        with zipfile.ZipFile(target) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    yield info.filename, archive.read(info)
    elif target.is_file() and target.name.endswith(".tar.gz"):
        with tarfile.open(target, "r:gz") as archive:
            for member in archive.getmembers():
                extracted = archive.extractfile(member) if member.isfile() else None
                if extracted is not None:
                    name = member.name.split("/", 1)[1] if "/" in member.name else member.name
                    yield name, extracted.read()
    else:
        msg = f"not a directory, .whl or .tar.gz: {target}"
        raise GateConfigError(msg)


def scan(
    targets: list[Path], deny: list[tuple[int, re.Pattern[str]]], allow: frozenset[str]
) -> list[Finding]:
    """Scan every target and return all findings."""
    findings: list[Finding] = []
    for target in targets:
        for name, data in _iter_target(target):
            text = _decode(data)
            if text is not None:
                findings.extend(scan_text(text, name, deny, allow))
    return findings


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the exit code."""
    parser = argparse.ArgumentParser(prog="export_gate", description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--deny", type=Path, help="private deny-pattern file (full scan)")
    mode.add_argument(
        "--generic-only",
        action="store_true",
        help="built-in detectors only, no deny file (public CI)",
    )
    parser.add_argument("--allow", type=Path, help="allow list of literal public strings")
    parser.add_argument("targets", nargs="+", type=Path, help="directory, .whl or .tar.gz")
    args = parser.parse_args(argv)
    try:
        deny = [] if args.generic_only else load_deny(args.deny)
        allow = load_allow(args.allow)
        findings = scan(args.targets, deny, allow)
    except GateConfigError as exc:
        print(f"export gate: ERROR: {exc}", file=sys.stderr)
        return 2
    for finding in findings:
        print(f"export gate: {finding}")
    if findings:
        print(f"export gate: FAILED, {len(findings)} finding(s); matched text is never shown")
        return 1
    print("export gate: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
