"""Lossless text chunking with line ranges and a composition receipt.

Pure functions: no I/O and no configuration reads. The caller passes ``max_chars``.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

_BLANK_RUN = r"(\n{2,})"


@dataclass(frozen=True)
class Chunk:
    """One exact slice of a text, with its position and 1-based line range."""

    order: int
    text: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class Receipt:
    """Proof of a text's identity: UTF-8 sha256, size in characters, chunk count."""

    content_hash: str
    size_chars: int
    chunk_count: int


def _pieces(text: str, max_chars: int) -> Iterator[str]:
    """Yield the segments of ``text`` in order, hard-splitting those over ``max_chars``.

    A segment is a maximal run of two or more newlines, or the text between runs.
    """
    for segment in re.split(_BLANK_RUN, text):
        for start in range(0, len(segment), max_chars):
            yield segment[start : start + max_chars]


def chunk_text(text: str, max_chars: int) -> list[Chunk]:
    """Split ``text`` into chunks of at most ``max_chars`` that join back to it exactly.

    Pieces are packed greedily: a piece joins the current chunk unless that would
    exceed ``max_chars`` while the current chunk is non-empty.

    Raises:
        ValueError: if ``max_chars`` is lower than 1.

    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be >= 1, got {max_chars}")
    texts: list[str] = []
    current = ""
    for piece in _pieces(text, max_chars):
        if current and len(current) + len(piece) > max_chars:
            texts.append(current)
            current = ""
        current += piece
    if current:
        texts.append(current)

    chunks: list[Chunk] = []
    line = 1
    for order, part in enumerate(texts):
        end_line = line + part[:-1].count("\n")
        chunks.append(Chunk(order=order, text=part, start_line=line, end_line=end_line))
        line += part.count("\n")
    return chunks


def _hash(text: str) -> str:
    """Return the sha256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_receipt(text: str, chunks: Sequence[Chunk]) -> Receipt:
    """Build the receipt of ``text`` after checking ``chunks`` join back to it.

    Raises:
        ValueError: if the chunk texts, joined in order, differ from ``text``.

    """
    if "".join(c.text for c in chunks) != text:
        raise ValueError("chunks do not join back to the text")
    return Receipt(content_hash=_hash(text), size_chars=len(text), chunk_count=len(chunks))


def verify_composition(parts: Iterable[str], receipt: Receipt) -> bool:
    """Return True if ``parts`` match the receipt in count, size and hash."""
    items = list(parts)
    joined = "".join(items)
    return (
        len(items) == receipt.chunk_count
        and len(joined) == receipt.size_chars
        and _hash(joined) == receipt.content_hash
    )
