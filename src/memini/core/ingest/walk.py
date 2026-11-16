"""Walk a corpus directory into raw-text files, guard the database, and dry-run a categorizer.

The walk is driven only by :class:`WalkRules` (built from a config section); the
database refusal reuses :mod:`memini.infra.graph.guard`; the dry run writes one
JSON line per walked file and touches no database.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from memini.infra.graph.guard import UnsafeDatabaseError, guard_from_env

__all__ = [
    "CorpusFile",
    "UnsafeDatabaseError",
    "WalkRules",
    "dry_run",
    "ensure_safe_database",
    "walk_corpus",
]

_RULE_KEYS: tuple[str, ...] = ("extensions", "skip_dirs", "skip_names", "max_line_chars")


@dataclass(frozen=True)
class WalkRules:
    """Filtering rules for :func:`walk_corpus`.

    Attributes:
        extensions: Lower-case suffixes with the dot, e.g. ``".md"``.
        skip_dirs: Folder names skipped at any depth below the root.
        skip_names: Exact file names skipped.
        max_line_chars: A file with any line longer than this is skipped.

    """

    extensions: frozenset[str]
    skip_dirs: frozenset[str]
    skip_names: frozenset[str]
    max_line_chars: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> WalkRules:
        """Build from a config section with keys extensions, skip_dirs, skip_names, max_line_chars.

        The first three are lists of strings, the last an int. Extensions are lower-cased.

        Raises:
            ValueError: If a key is missing; the message names the key.

        """
        for key in _RULE_KEYS:
            if key not in data:
                raise ValueError(f"missing walk rule key: {key!r}")
        return cls(
            extensions=frozenset(str(ext).lower() for ext in data["extensions"]),
            skip_dirs=frozenset(str(name) for name in data["skip_dirs"]),
            skip_names=frozenset(str(name) for name in data["skip_names"]),
            max_line_chars=int(data["max_line_chars"]),
        )


@dataclass(frozen=True)
class CorpusFile:
    """One walked file.

    Attributes:
        path: Path relative to the walk root.
        text: Raw text, decoded UTF-8 with errors ignored, never stripped.

    """

    path: PurePosixPath
    text: str


def _path_allowed(rel: PurePosixPath, rules: WalkRules) -> bool:
    """Return whether ``rel`` passes the extension, folder and name rules."""
    if rel.suffix.lower() not in rules.extensions:
        return False
    if rel.name in rules.skip_names:
        return False
    return not any(part in rules.skip_dirs for part in rel.parts[:-1])


def _text_allowed(text: str, rules: WalkRules) -> bool:
    """Return whether ``text`` is non-blank and has no over-long line."""
    if not text.strip():
        return False
    return all(len(line) <= rules.max_line_chars for line in text.splitlines())


def walk_corpus(root: Path, rules: WalkRules, *, limit: int | None = None) -> list[CorpusFile]:
    """Return the files under ``root`` that pass ``rules``, sorted by relative POSIX path.

    Args:
        root: The corpus directory, walked recursively.
        rules: The filtering rules; only path parts relative to ``root`` are checked.
        limit: At most this many matching files are returned; ``None`` means all.

    Returns:
        The matching files, in sorted order of their relative path string.

    """
    candidates = sorted(
        (
            PurePosixPath(path.relative_to(root).as_posix())
            for path in root.rglob("*")
            if path.is_file()
        ),
        key=str,
    )
    files: list[CorpusFile] = []
    for rel in candidates:
        if limit is not None and len(files) >= limit:
            break
        if not _path_allowed(rel, rules):
            continue
        text = root.joinpath(rel).read_bytes().decode("utf-8", errors="ignore")
        if _text_allowed(text, rules):
            files.append(CorpusFile(path=rel, text=text))
    return files


def ensure_safe_database(uri: str, env: Mapping[str, str]) -> str:
    """Return the normalised host, as :func:`memini.infra.graph.guard.guard_from_env`.

    Raises:
        UnsafeDatabaseError: For the production host (env ``MEMINI_PROD_NEO4J_HOST``)
            or any non-local, non-allow-listed host. Reads only ``env``.

    """
    return guard_from_env(uri, env)


def dry_run(
    root: Path,
    rules: WalkRules,
    categorize: Callable[[CorpusFile], Mapping[str, Any]],
    out: Path,
    *,
    limit: int | None = None,
) -> int:
    """Walk ``root``, categorize each file in walk order and write one JSON line per file.

    Each line of ``out`` (UTF-8) is ``{"path": <posix relative path>, **decision}``,
    the walked path winning over any ``"path"`` key in the decision. No database is used.

    Returns:
        The number of lines written.

    """
    files = walk_corpus(root, rules, limit=limit)
    with out.open("w", encoding="utf-8") as handle:
        for file in files:
            decision = categorize(file)
            record = {"path": str(file.path), **{k: v for k, v in decision.items() if k != "path"}}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(files)
