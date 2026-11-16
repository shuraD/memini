"""Acceptance tests for IN-09: LLM extraction windows (lines by default, structure opt-in)."""

import builtins
import dataclasses
import itertools
import random

import pytest

from memini.core.ingest.windows import (
    CHARS_PER_TOKEN,
    Boundary,
    Window,
    WindowSettings,
    cut_spans,
    default_max_tokens,
    enforce_window,
    estimate_tokens,
    extraction_windows,
    line_windows,
    needs_chunking,
    structure_windows,
)

ALPHABET = ["a", "b", "z", " ", "\t", "\n", "\n", "\n", "é", "中", "😀"]


def _random_texts(count: int = 120, seed: int = 4242) -> list[str]:
    rng = random.Random(seed)  # noqa: S311  (fixed-seed test data, not crypto)
    texts = []
    for _ in range(count):
        length = rng.randint(1, 120)
        texts.append("".join(rng.choice(ALPHABET) for _ in range(length)))
    texts += [
        "\n",
        "\n\n\n\n",
        "x" * 50,
        "short\n" + "y" * 40 + "\nend",
        "trailing\n\n\n",
    ]
    return texts


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _assert_windows_cover(text: str, windows: list[Window]) -> None:
    assert "".join(w.content for w in windows) == text
    n = len(windows)
    pos = 0
    for i, w in enumerate(windows):
        assert isinstance(w, Window)
        assert w.index == i
        assert w.total == n
        assert w.content != ""
        assert text[pos : pos + len(w.content)] == w.content
        assert w.start_line == _line_of(text, pos)
        assert w.end_line == _line_of(text, pos + len(w.content) - 1)
        pos += len(w.content)
    assert pos == len(text)


def _partition_windows(text: str, cuts: list[int], context: str = "") -> list[Window]:
    """Build input windows over text, split at the given offsets."""
    edges = [0, *cuts, len(text)]
    spans = [(a, b) for a, b in itertools.pairwise(edges) if b > a]
    return [
        Window(
            content=text[a:b],
            index=i,
            total=len(spans),
            context=context,
            start_line=_line_of(text, a),
            end_line=_line_of(text, b - 1),
        )
        for i, (a, b) in enumerate(spans)
    ]


# --- constants, dataclasses, settings ----------------------------------------


def test_chars_per_token_is_four() -> None:
    assert CHARS_PER_TOKEN == 4


def test_dataclasses_are_frozen_and_boundary_defaults() -> None:
    b = Boundary(id="b1", kind="class", name="C", start=0, end=5)
    assert b.parent is None
    assert b.signature == ""
    w = Window(content="abc", index=0, total=1, context="", start_line=1, end_line=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.content = "x"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.start = 3  # type: ignore[misc]
    s = WindowSettings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.use_structure = True  # type: ignore[misc]


def test_window_settings_defaults_structure_off() -> None:
    s = WindowSettings()
    assert s.use_structure is False
    assert s.context_kinds == ()
    assert s.pack is True


def test_window_settings_from_empty_mapping_keeps_defaults() -> None:
    s = WindowSettings.from_mapping({})
    assert s == WindowSettings()
    assert s.use_structure is False


def test_window_settings_from_full_mapping() -> None:
    s = WindowSettings.from_mapping(
        {"use_for_extraction": True, "context_kinds": ["class", "function"], "pack": False}
    )
    assert s.use_structure is True
    assert s.context_kinds == ("class", "function")
    assert isinstance(s.context_kinds, tuple)
    assert s.pack is False


def test_window_settings_from_partial_mapping() -> None:
    s = WindowSettings.from_mapping({"context_kinds": ["section"]})
    assert s.use_structure is False
    assert s.context_kinds == ("section",)
    assert s.pack is True
    s2 = WindowSettings.from_mapping({"use_for_extraction": True})
    assert s2.use_structure is True
    assert s2.context_kinds == ()
    assert s2.pack is True


# --- token helpers --------------------------------------------------------------


def test_default_max_tokens() -> None:
    assert default_max_tokens(8192, 8000) == 2000
    assert default_max_tokens(600, 8000) == 200


def test_estimate_tokens_and_needs_chunking() -> None:
    assert estimate_tokens("x" * 9) == 2
    assert estimate_tokens("") == 0
    assert needs_chunking("x" * 9, 2) is False
    assert needs_chunking("x" * 12, 2) is True


# --- enforce_window ---------------------------------------------------------------


def test_enforce_window_splits_at_last_newline_then_hard_cut() -> None:
    content = "abc\ndefghijklmno"  # 16 chars, limit for max_tokens=2 is 8
    w = Window(content=content, index=0, total=1, context="ctx", start_line=5, end_line=6)
    out = enforce_window([w], 2)
    assert [p.content for p in out] == ["abc\n", "defghijk", "lmno"]
    assert all(p.context == "ctx" for p in out)
    assert [(p.start_line, p.end_line) for p in out] == [(5, 5), (6, 6), (6, 6)]
    assert [p.index for p in out] == [0, 1, 2]
    assert all(p.total == 3 for p in out)


def test_enforce_window_hard_cut_when_only_newline_is_at_position_zero() -> None:
    content = "\nabcdefghijk"  # 12 chars, newline only at position 0
    w = Window(content=content, index=0, total=1, context="", start_line=1, end_line=2)
    out = enforce_window([w], 2)
    assert [p.content for p in out] == ["\nabcdefg", "hijk"]
    assert [(p.start_line, p.end_line) for p in out] == [(1, 2), (2, 2)]


def test_enforce_window_keeps_small_windows_and_renumbers() -> None:
    small = Window(content="12345678", index=7, total=9, context="k", start_line=3, end_line=3)
    big = Window(content="ab\ncdefghijkl", index=8, total=9, context="m", start_line=4, end_line=5)
    out = enforce_window([small, big], 2)
    assert out[0] == dataclasses.replace(small, index=0, total=len(out))
    assert [p.content for p in out] == ["12345678", "ab\n", "cdefghij", "kl"]
    assert [p.context for p in out] == ["k", "m", "m", "m"]
    assert [(p.start_line, p.end_line) for p in out] == [(3, 3), (4, 4), (5, 5), (5, 5)]
    assert [p.index for p in out] == [0, 1, 2, 3]
    assert all(p.total == 4 for p in out)


def test_enforce_window_all_within_limit_returns_same_windows() -> None:
    text = "aa\nbb\ncc\n"
    windows = _partition_windows(text, [3, 6])
    assert enforce_window(windows, 100) == windows


def test_enforce_window_is_lossless_on_random_texts() -> None:
    rng = random.Random(7)  # noqa: S311  (fixed-seed test data, not crypto)
    for text in _random_texts():
        cuts = sorted({rng.randint(1, len(text)) for _ in range(3)} - {len(text)})
        windows = _partition_windows(text, cuts, context="c")
        for max_tokens in (1, 2, 5):
            out = enforce_window(windows, max_tokens)
            _assert_windows_cover(text, out)
            assert all(w.context == "c" for w in out)


# --- line_windows ----------------------------------------------------------------------


def test_line_windows_packs_whole_lines_greedily() -> None:
    text = "aa\nbb\ncc\n"
    out = line_windows(text, 2)  # limit 8 chars
    assert [w.content for w in out] == ["aa\nbb\n", "cc\n"]
    assert [(w.start_line, w.end_line) for w in out] == [(1, 2), (3, 3)]
    assert all(w.context == "" for w in out)
    _assert_windows_cover(text, out)


def test_line_windows_fills_up_to_exact_limit() -> None:
    text = "abc\ndef\nghi\n"
    out = line_windows(text, 2)  # limit 8 chars: two 4-char lines fit exactly
    assert [w.content for w in out] == ["abc\ndef\n", "ghi\n"]


def test_line_windows_last_line_without_newline() -> None:
    text = "one\ntwo\nthree"
    out = line_windows(text, 1)  # limit 4 chars
    assert [w.content for w in out][:2] == ["one\n", "two\n"]
    _assert_windows_cover(text, out)


def test_line_windows_splits_a_long_line() -> None:
    text = "ab\n" + "x" * 20 + "\n" + "cd"
    out = line_windows(text, 2)
    _assert_windows_cover(text, out)
    assert out[0].content == "ab\n"
    assert out[1].content == "x" * 8
    assert all(w.context == "" for w in out)
    assert len(out) >= 4


def test_line_windows_is_lossless_on_random_texts() -> None:
    for text in _random_texts():
        for max_tokens in (1, 3, 10):
            out = line_windows(text, max_tokens)
            _assert_windows_cover(text, out)
            assert all(w.context == "" for w in out)


# --- cut_spans ------------------------------------------------------------------------

_TEXT30 = "a" * 30
_NESTED = [
    Boundary(id="P", kind="k", name="P", start=0, end=10),
    Boundary(id="Q", kind="k", name="Q", start=10, end=30),
    Boundary(id="Q1", kind="k", name="Q1", start=18, end=24, parent="Q"),
    Boundary(id="Q2", kind="k", name="Q2", start=24, end=30, parent="Q"),
    Boundary(id="Q1a", kind="k", name="Q1a", start=21, end=24, parent="Q1"),
]


def test_cut_spans_empty_text() -> None:
    assert cut_spans("", [], 10) == []
    assert cut_spans("", [], 10, pack=False) == []


def test_cut_spans_text_within_budget_is_one_span() -> None:
    assert cut_spans(_TEXT30, _NESTED, 30) == [(0, 30)]
    assert cut_spans(_TEXT30, _NESTED, 100, pack=False) == [(0, 30)]


def test_cut_spans_no_cuts_returns_whole() -> None:
    assert cut_spans(_TEXT30, [], 5) == [(0, 30)]
    only_edges = [Boundary(id="r", kind="k", name="r", start=0, end=30)]
    assert cut_spans(_TEXT30, only_edges, 5, pack=False) == [(0, 30)]


def test_cut_spans_shallowest_first_without_packing() -> None:
    assert cut_spans(_TEXT30, _NESTED, 12, pack=False) == [
        (0, 10),
        (10, 18),
        (18, 24),
        (24, 30),
    ]


def test_cut_spans_shallowest_first_with_packing() -> None:
    assert cut_spans(_TEXT30, _NESTED, 12, pack=True) == [(0, 10), (10, 18), (18, 30)]
    assert cut_spans(_TEXT30, _NESTED, 12) == [(0, 10), (10, 18), (18, 30)]


def test_cut_spans_recurses_into_pieces_still_too_large() -> None:
    assert cut_spans(_TEXT30, _NESTED, 5, pack=False) == [
        (0, 10),
        (10, 18),
        (18, 21),
        (21, 24),
        (24, 30),
    ]


def test_cut_spans_cover_text_contiguously() -> None:
    for budget in (1, 3, 7, 12, 20, 29, 30, 31):
        for pack in (True, False):
            spans = cut_spans(_TEXT30, _NESTED, budget, pack=pack)
            assert spans[0][0] == 0
            assert spans[-1][1] == 30
            for (_a, b), (c, _d) in itertools.pairwise(spans):
                assert b == c
            assert all(a < b for a, b in spans)


# --- structure_windows ----------------------------------------------------------------


def test_structure_windows_context_and_budget_reduction() -> None:
    text = "H" * 10 + "A" * 10 + "B" * 10
    bounds = [
        Boundary(id="c", kind="class", name="C", start=0, end=30, signature="sig C"),
        Boundary(id="m1", kind="method", name="m1", start=10, end=20, parent="c"),
        Boundary(id="m2", kind="method", name="m2", start=20, end=30, parent="c"),
    ]
    # max_tokens=6 -> 24 chars; L = len("sig C") = 5 -> budget 19, so 10+10 cannot merge.
    out = structure_windows(text, bounds, 6, context_kinds=("class",))
    assert [w.content for w in out] == ["H" * 10, "A" * 10, "B" * 10]
    assert [w.context for w in out] == ["", "sig C", "sig C"]
    _assert_windows_cover(text, out)


def test_structure_windows_without_context_kinds_packs_to_full_budget() -> None:
    text = "H" * 10 + "A" * 10 + "B" * 10
    bounds = [
        Boundary(id="c", kind="class", name="C", start=0, end=30, signature="sig C"),
        Boundary(id="m1", kind="method", name="m1", start=10, end=20, parent="c"),
        Boundary(id="m2", kind="method", name="m2", start=20, end=30, parent="c"),
    ]
    out = structure_windows(text, bounds, 6)
    assert [w.content for w in out] == ["H" * 10 + "A" * 10, "B" * 10]
    assert all(w.context == "" for w in out)
    out_unpacked = structure_windows(text, bounds, 6, pack=False)
    assert [w.content for w in out_unpacked] == ["H" * 10, "A" * 10, "B" * 10]


def test_structure_windows_nested_context_ordered_and_name_fallback() -> None:
    text = "".join(chr(ord("a") + i % 26) for i in range(40))
    bounds = [
        Boundary(
            id="M", kind="method", name="m", start=10, end=40, parent="C", signature="def m()"
        ),
        Boundary(id="C", kind="class", name="C", start=0, end=40),
        Boundary(id="B", kind="block", name="blk", start=25, end=40, parent="M"),
    ]
    # L = len("C\ndef m()") = 9; max_tokens=5 -> budget 20 - 9 = 11.
    out = structure_windows(text, bounds, 5, context_kinds=("class", "method"))
    assert [w.content for w in out] == [text[0:10], text[10:25], text[25:40]]
    assert [w.context for w in out] == ["", "C", "C\ndef m()"]
    _assert_windows_cover(text, out)


def test_structure_windows_budget_floor_is_chars_per_token() -> None:
    text = "aaaa" + "bbbb" + "cccc"
    bounds = [
        Boundary(id="c", kind="class", name="C", start=0, end=12, signature="s" * 20),
        Boundary(id="x1", kind="x", name="x1", start=4, end=8, parent="c"),
        Boundary(id="x2", kind="x", name="x2", start=8, end=12, parent="c"),
    ]
    out = structure_windows(text, bounds, 1, context_kinds=("class",))
    assert [w.content for w in out] == ["aaaa", "bbbb", "cccc"]
    assert [w.context for w in out] == ["", "s" * 20, "s" * 20]


def test_structure_windows_enforces_token_limit_on_uncut_pieces() -> None:
    text = "x" * 50
    out = structure_windows(text, [], 2)
    _assert_windows_cover(text, out)
    assert all(len(w.content) <= 8 for w in out)
    assert all(w.context == "" for w in out)
    assert len(out) == 7


def test_structure_windows_is_lossless_on_random_texts() -> None:
    for text in _random_texts():
        bounds = [Boundary(id="root", kind="root", name="root", start=0, end=len(text))]
        pos = 0
        for i, line in enumerate(text.splitlines(keepends=True)):
            bounds.append(
                Boundary(
                    id=f"l{i}",
                    kind="line",
                    name=f"l{i}",
                    start=pos,
                    end=pos + len(line),
                    parent="root",
                )
            )
            pos += len(line)
        for max_tokens in (1, 3, 10):
            for pack in (True, False):
                out = structure_windows(
                    text, bounds, max_tokens, context_kinds=("root",), pack=pack
                )
                _assert_windows_cover(text, out)
                for w in out:
                    assert w.context in ("", "root")


# --- extraction_windows -----------------------------------------------------------------

_LONG = "".join(f"line {i:02d} " + "w" * (i % 7) + "\n" for i in range(30))


def _long_bounds() -> list[Boundary]:
    bounds = [Boundary(id="doc", kind="doc", name="doc", start=0, end=len(_LONG))]
    half = _LONG.index("line 15")
    bounds.append(Boundary(id="h1", kind="sec", name="first", start=0, end=half, parent="doc"))
    bounds.append(
        Boundary(id="h2", kind="sec", name="second", start=half, end=len(_LONG), parent="doc")
    )
    return bounds


def test_extraction_windows_empty_text() -> None:
    assert extraction_windows("", 10) == []
    on = WindowSettings(use_structure=True)
    assert extraction_windows("", 10, on, []) == []


def test_extraction_windows_small_text_is_one_window() -> None:
    text = "hello\nworld"
    expected = [Window(content=text, index=0, total=1, context="", start_line=1, end_line=2)]
    assert extraction_windows(text, 100) == expected
    on = WindowSettings(use_structure=True, context_kinds=("doc",))
    bounds = [Boundary(id="d", kind="doc", name="d", start=0, end=len(text))]
    assert extraction_windows(text, 100, on, bounds) == expected


def test_extraction_windows_default_uses_lines_and_ignores_boundaries() -> None:
    assert needs_chunking(_LONG, 20)
    expected = line_windows(_LONG, 20)
    assert extraction_windows(_LONG, 20) == expected
    assert extraction_windows(_LONG, 20, WindowSettings(), _long_bounds()) == expected
    assert extraction_windows(_LONG, 20, WindowSettings.from_mapping({}), _long_bounds()) == (
        expected
    )
    _assert_windows_cover(_LONG, expected)


def test_extraction_windows_structure_on_uses_structure_windows() -> None:
    settings = WindowSettings(use_structure=True, context_kinds=("doc", "sec"), pack=False)
    bounds = _long_bounds()
    expected = structure_windows(_LONG, bounds, 40, ("doc", "sec"), False)
    out = extraction_windows(_LONG, 40, settings, bounds)
    assert out == expected
    assert out != line_windows(_LONG, 40)
    _assert_windows_cover(_LONG, out)


def test_extraction_windows_structure_on_without_boundaries_uses_lines() -> None:
    settings = WindowSettings(use_structure=True, context_kinds=("sec",))
    assert extraction_windows(_LONG, 20, settings, None) == line_windows(_LONG, 20)
    assert extraction_windows(_LONG, 20, settings) == line_windows(_LONG, 20)


def test_extraction_windows_does_no_file_io(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def _no_open(*args: object, **kwargs: object) -> object:
        calls.append(args)
        raise AssertionError("open must not be called")

    settings = WindowSettings(use_structure=True, context_kinds=("sec",))
    bounds = _long_bounds()
    with monkeypatch.context() as m:
        m.setattr(builtins, "open", _no_open)
        a = extraction_windows(_LONG, 20)
        b = extraction_windows(_LONG, 20, settings, bounds)
    assert calls == []
    _assert_windows_cover(_LONG, a)
    _assert_windows_cover(_LONG, b)
