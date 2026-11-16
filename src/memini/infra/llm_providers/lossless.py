"""Lossless embedding: batch first, then one text at a time, then split and pool.

A ``LosslessEmbedder`` wraps any ``Embedder`` and never drops a character: a text the
inner embedder rejects is halved and the piece vectors are pooled, weighted by length.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from memini.infra.llm_providers.types import Embedder, LLMError


def halve(text: str) -> tuple[str, str]:
    """Split near the middle without losing a character (``a + b == text``).

    The cut is the last newline before the middle when it lies past the first quarter
    (the newline starts the second half); otherwise it is ``len(text) // 2``.
    """
    mid = len(text) // 2
    cut = text.rfind("\n", 0, mid)
    if cut > len(text) // 4:
        return text[:cut], text[cut:]
    return text[:mid], text[mid:]


def pool(va: Sequence[float], la: int, vb: Sequence[float], lb: int) -> list[float]:
    """Return the length-weighted mean of two vectors scaled to unit L2 norm.

    The mean is ``(va * la + vb * lb) / max(1, la + lb)``; a zero mean is returned as zeros.
    """
    total = max(1, la + lb)
    mean = [(a * la + b * lb) / total for a, b in zip(va, vb, strict=True)]
    norm = math.sqrt(sum(m * m for m in mean))
    if norm == 0.0:
        return [0.0] * len(mean)
    return [m / norm for m in mean]


class LosslessEmbedder:
    """An ``Embedder`` that embeds every text whole, splitting rejected texts in two."""

    def __init__(self, inner: Embedder, *, max_depth: int = 8, min_split: int = 64) -> None:
        """Wrap ``inner``; split at most ``max_depth`` times, never texts below ``min_split``."""
        self._inner = inner
        self._max_depth = max_depth
        self._min_split = min_split

    def embed(
        self, texts: Sequence[str], *, kind: Literal["query", "document"]
    ) -> list[list[float]]:
        """Return one vector per text, trying the batch, then each text, then its halves."""
        if not texts:
            return []
        try:
            vectors = self._inner.embed(list(texts), kind=kind)
        except LLMError:
            return [self._one(t, 0, kind) for t in texts]
        return [[float(x) for x in v] for v in vectors]

    def _one(self, text: str, depth: int, kind: Literal["query", "document"]) -> list[float]:
        """Embed one text alone; on ``LLMError`` split it in two and pool the pieces."""
        try:
            return [float(x) for x in self._inner.embed([text], kind=kind)[0]]
        except LLMError:
            if depth >= self._max_depth or len(text) < self._min_split:
                raise
        a, b = halve(text)
        return pool(self._one(a, depth + 1, kind), len(a), self._one(b, depth + 1, kind), len(b))
