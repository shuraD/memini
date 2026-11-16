"""LLM extraction windows: lossless cuts of a text, by lines or by parser spans.

Pure functions: no I/O and no configuration reads. Structure windows are opt-in
through ``WindowSettings.use_structure``, which is off by default.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

CHARS_PER_TOKEN: int = 4


@dataclass(frozen=True)
class Window:
    """One exact slice of a text, sent to an LLM with an optional context string."""

    content: str
    index: int
    total: int
    context: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class Boundary:
    """One parser span ``[start, end)`` over a text, with its enclosing span id."""

    id: str
    kind: str
    name: str
    start: int
    end: int
    parent: str | None = None
    signature: str = ""


@dataclass(frozen=True)
class WindowSettings:
    """How extraction windows are cut; structure windows are off by default."""

    use_structure: bool = False
    context_kinds: tuple[str, ...] = ()
    pack: bool = True

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> WindowSettings:
        """Build settings from a config section; a missing key keeps its default.

        Keys: ``use_for_extraction`` (bool), ``context_kinds`` (list of str), ``pack`` (bool).

        Raises:
            TypeError: if a present key has a value of the wrong type.

        """
        default = cls()
        use = mapping.get("use_for_extraction", default.use_structure)
        pack = mapping.get("pack", default.pack)
        kinds = mapping.get("context_kinds", default.context_kinds)
        if not isinstance(use, bool) or not isinstance(pack, bool):
            raise TypeError("use_for_extraction and pack must be booleans")
        if not isinstance(kinds, (list, tuple)) or not all(isinstance(k, str) for k in kinds):
            raise TypeError("context_kinds must be a list of strings")
        return cls(use_structure=use, context_kinds=tuple(str(k) for k in kinds), pack=pack)


def default_max_tokens(num_predict: int, embed_max_chars: int) -> int:
    """Return the default window budget in tokens for the given model limits."""
    return min(num_predict // 3, embed_max_chars // CHARS_PER_TOKEN)


def estimate_tokens(text: str) -> int:
    """Return a rough token count of ``text``."""
    return len(text) // CHARS_PER_TOKEN


def needs_chunking(text: str, max_tokens: int) -> bool:
    """Return True if ``text`` is estimated to exceed ``max_tokens``."""
    return estimate_tokens(text) > max_tokens


def _limit(max_tokens: int) -> int:
    """Return the character budget of ``max_tokens`` (at least 1)."""
    return max(1, max_tokens * CHARS_PER_TOKEN)


def _renumber(windows: Sequence[Window]) -> list[Window]:
    """Return ``windows`` with index and total set to their position and count."""
    total = len(windows)
    return [dataclasses.replace(w, index=i, total=total) for i, w in enumerate(windows)]


def _split_window(window: Window, limit: int) -> list[Window]:
    """Split one window into pieces of at most ``limit`` chars, preferring newline ends."""
    pieces: list[Window] = []
    rest = window.content
    line = window.start_line
    while rest:
        if len(rest) <= limit:
            cut = len(rest)
        else:
            newline = rest.rfind("\n", 0, limit)
            cut = newline + 1 if newline > 0 else limit
        piece = rest[:cut]
        end_line = line + piece[:-1].count("\n")
        pieces.append(
            dataclasses.replace(window, content=piece, start_line=line, end_line=end_line)
        )
        line += piece.count("\n")
        rest = rest[cut:]
    return pieces


def enforce_window(windows: Sequence[Window], max_tokens: int) -> list[Window]:
    """Split windows longer than ``max_tokens`` and renumber the result."""
    limit = _limit(max_tokens)
    out: list[Window] = []
    for window in windows:
        if len(window.content) <= limit:
            out.append(window)
        else:
            out.extend(_split_window(window, limit))
    return _renumber(out)


def _windows_from_spans(
    text: str, spans: Sequence[tuple[int, int]], contexts: Sequence[str]
) -> list[Window]:
    """Build numbered windows over ``text`` from contiguous spans and their contexts."""
    return _renumber(
        [
            Window(
                content=text[a:b],
                index=0,
                total=0,
                context=context,
                start_line=text.count("\n", 0, a) + 1,
                end_line=text.count("\n", 0, b - 1) + 1,
            )
            for (a, b), context in zip(spans, contexts, strict=True)
        ]
    )


def line_windows(text: str, max_tokens: int) -> list[Window]:
    """Pack whole lines greedily into windows of at most ``max_tokens``."""
    limit = _limit(max_tokens)
    spans: list[tuple[int, int]] = []
    start = 0
    pos = 0
    while pos < len(text):
        newline = text.find("\n", pos)
        end = len(text) if newline < 0 else newline + 1
        if end - start > limit and pos > start:
            spans.append((start, pos))
            start = pos
        pos = end
    if pos > start:
        spans.append((start, pos))
    return enforce_window(_windows_from_spans(text, spans, [""] * len(spans)), max_tokens)


def _depth(boundary: Boundary, by_id: Mapping[str, Boundary]) -> int:
    """Return the number of parent links above ``boundary`` (cycles stop the walk)."""
    depth = 0
    seen = {boundary.id}
    parent = boundary.parent
    while parent is not None and parent in by_id and parent not in seen:
        seen.add(parent)
        depth += 1
        parent = by_id[parent].parent
    return depth


def _split_span(
    start: int, end: int, cuts: Mapping[int, int], max_chars: int
) -> list[tuple[int, int]]:
    """Cut ``[start, end)`` at its shallowest cuts first, recursing into large pieces."""
    if end - start <= max_chars:
        return [(start, end)]
    inside = [c for c in cuts if start < c < end]
    if not inside:
        return [(start, end)]
    shallowest = min(cuts[c] for c in inside)
    edges = [start, *sorted(c for c in inside if cuts[c] == shallowest), end]
    spans: list[tuple[int, int]] = []
    for a, b in itertools.pairwise(edges):
        spans.extend(_split_span(a, b, cuts, max_chars))
    return spans


def cut_spans(
    text: str, boundaries: Sequence[Boundary], max_chars: int, pack: bool = True
) -> list[tuple[int, int]]:
    """Cut ``text`` into contiguous spans at boundary starts, shallowest first.

    With ``pack``, consecutive spans are merged while they stay within ``max_chars``.
    """
    n = len(text)
    if n == 0:
        return []
    by_id = {b.id: b for b in boundaries}
    cuts: dict[int, int] = {}
    for b in boundaries:
        if 0 < b.start < n:
            depth = _depth(b, by_id)
            cuts[b.start] = min(depth, cuts.get(b.start, depth))
    spans = _split_span(0, n, cuts, max_chars)
    if not pack:
        return spans
    packed: list[tuple[int, int]] = []
    for span_start, span_end in spans:
        if packed and span_end - packed[-1][0] <= max_chars:
            packed[-1] = (packed[-1][0], span_end)
        else:
            packed.append((span_start, span_end))
    return packed


def _context_at(position: int, scopes: Sequence[Boundary]) -> str:
    """Return the signatures (or names) of the scopes strictly enclosing ``position``."""
    enclosing = sorted(
        (b for b in scopes if b.start < position < b.end), key=lambda b: (b.start, -b.end)
    )
    return "\n".join(b.signature or b.name for b in enclosing)


def structure_windows(
    text: str,
    boundaries: Sequence[Boundary],
    max_tokens: int,
    context_kinds: Sequence[str] = (),
    pack: bool = True,
) -> list[Window]:
    """Cut ``text`` at parser span boundaries, each window carrying its enclosing scopes.

    The cut budget is reduced by the longest possible context, then every window is
    held to ``max_tokens`` by ``enforce_window``.
    """
    scopes = [b for b in boundaries if b.kind in context_kinds]
    longest = max((len(_context_at(b.start + 1, scopes)) for b in scopes), default=0)
    budget = max(CHARS_PER_TOKEN, max_tokens * CHARS_PER_TOKEN - longest)
    spans = cut_spans(text, boundaries, budget, pack)
    contexts = [_context_at(a, scopes) for a, _ in spans]
    return enforce_window(_windows_from_spans(text, spans, contexts), max_tokens)


def extraction_windows(
    text: str,
    max_tokens: int,
    settings: WindowSettings = WindowSettings(),  # noqa: B008  (frozen, immutable default)
    boundaries: Sequence[Boundary] | None = None,
) -> list[Window]:
    """Return the extraction windows of ``text``: lines by default, structure when enabled."""
    if not text:
        return []
    if not needs_chunking(text, max_tokens):
        return _windows_from_spans(text, [(0, len(text))], [""])
    if settings.use_structure and boundaries is not None:
        return structure_windows(
            text, boundaries, max_tokens, settings.context_kinds, settings.pack
        )
    return line_windows(text, max_tokens)
