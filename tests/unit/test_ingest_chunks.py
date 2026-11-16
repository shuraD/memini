"""Acceptance tests for IN-03: lossless chunks with line ranges and a receipt."""

import dataclasses
import hashlib
import random

import pytest

from memini.core.ingest.chunks import (
    Chunk,
    Receipt,
    chunk_text,
    make_receipt,
    verify_composition,
)

ALPHABET = ["a", "b", "z", " ", "\t", "\n", "\n", "\n", "\r", "é", "中", "😀"]


def _random_texts(count: int = 200, seed: int = 1234) -> list[str]:
    rng = random.Random(seed)  # noqa: S311  (fixed-seed test data, not crypto)
    texts = []
    for _ in range(count):
        length = rng.randint(1, 80)
        texts.append("".join(rng.choice(ALPHABET) for _ in range(length)))
    texts += [
        "line one\r\nline two\r\n\r\n\r\nthird\n\n\n",
        "\n\n\n\n",
        "\n",
        "trailing newlines\n\n\n\n\n",
        "  spaces  \n\n  more  ",
        "naïve café — 中文\n\n😀😀😀",
    ]
    return texts


def _texts(chunks: list[Chunk]) -> list[str]:
    return [c.text for c in chunks]


def _offsets(chunks: list[Chunk]) -> list[int]:
    offsets, pos = [], 0
    for c in chunks:
        offsets.append(pos)
        pos += len(c.text)
    return offsets


@pytest.mark.parametrize("max_chars", [1, 2, 3, 5, 8, 13, 40, 1000])
def test_chunks_join_back_to_text_exactly(max_chars: int) -> None:
    for text in _random_texts():
        chunks = chunk_text(text, max_chars)
        assert "".join(_texts(chunks)) == text


@pytest.mark.parametrize("max_chars", [1, 2, 3, 7, 20])
def test_chunk_sizes_and_order(max_chars: int) -> None:
    for text in _random_texts():
        chunks = chunk_text(text, max_chars)
        assert all(isinstance(c, Chunk) for c in chunks)
        for index, c in enumerate(chunks):
            assert 1 <= len(c.text) <= max_chars
            assert c.order == index


def test_small_blank_line_example() -> None:
    chunks = chunk_text("a\n\nb", 3)
    assert _texts(chunks) == ["a\n\n", "b"]
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 2), (3, 3)]
    assert [c.order for c in chunks] == [0, 1]


def _segments_fit(text: str, max_chars: int) -> bool:
    segments, current, i = [], "", 0
    while i < len(text):
        j = i
        while j < len(text) and text[j] == "\n":
            j += 1
        if j - i >= 2:
            if current:
                segments.append(current)
                current = ""
            segments.append(text[i:j])
            i = j
        else:
            current += text[i]
            i += 1
    if current:
        segments.append(current)
    return all(len(s) <= max_chars for s in segments)


@pytest.mark.parametrize("max_chars", [4, 6, 10, 25])
def test_boundaries_only_at_blank_line_runs_when_segments_fit(max_chars: int) -> None:
    checked = 0
    for text in _random_texts(count=400, seed=99):
        if not _segments_fit(text, max_chars):
            continue
        chunks = chunk_text(text, max_chars)
        for s in _offsets(chunks)[1:]:
            checked += 1
            assert text[s - 2 : s] == "\n\n" or text[s : s + 2] == "\n\n"
            assert not (text[s - 1] == "\n" and text[s] == "\n")
    assert checked > 0


def test_blank_line_run_never_split() -> None:
    text = "para one\n\n\n\npara two\n\npara three"
    chunks = chunk_text(text, 12)
    assert "".join(_texts(chunks)) == text
    for s in _offsets(chunks)[1:]:
        assert not (text[s - 1] == "\n" and text[s] == "\n")
        assert text[s - 2 : s] == "\n\n" or text[s : s + 2] == "\n\n"


def test_greedy_packing_of_fitting_segments() -> None:
    assert _texts(chunk_text("ab\n\ncd\n\nef", 6)) == ["ab\n\ncd", "\n\nef"]


@pytest.mark.parametrize("text", ["a", "hello\n\nworld", "x" * 50, "\n\n\n", "é中😀\r\n"])
def test_whole_text_that_fits_is_one_chunk(text: str) -> None:
    chunks = chunk_text(text, max(len(text), 1))
    assert _texts(chunks) == [text]
    chunks = chunk_text(text, len(text) + 100)
    assert _texts(chunks) == [text]
    assert chunks[0].order == 0
    assert chunks[0].start_line == 1


def test_hard_split_examples() -> None:
    assert _texts(chunk_text("x" * 10, 4)) == ["xxxx", "xxxx", "xx"]
    assert _texts(chunk_text("xxxxx\n\ny", 4)) == ["xxxx", "x\n\ny"]


@pytest.mark.parametrize("max_chars", [1, 3, 5, 9, 30])
def test_line_ranges_follow_offsets(max_chars: int) -> None:
    for text in _random_texts(seed=7):
        chunks = chunk_text(text, max_chars)
        for s, c in zip(_offsets(chunks), chunks, strict=True):
            expected_start = text[:s].count("\n") + 1
            assert c.start_line == expected_start
            assert c.end_line == expected_start + c.text[:-1].count("\n")


def test_line_ranges_on_multiline_text() -> None:
    text = "l1\nl2\n\nl4\nl5"
    chunks = chunk_text(text, 7)
    assert _texts(chunks) == ["l1\nl2\n\n", "l4\nl5"]
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 3), (4, 5)]


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_text("", 5) == []
    assert chunk_text("", 1) == []


@pytest.mark.parametrize("max_chars", [0, -1, -100])
def test_invalid_max_chars_raises(max_chars: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("some text", max_chars)


def test_dataclasses_are_frozen() -> None:
    c = chunk_text("abc", 5)[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.text = "zzz"  # type: ignore[misc]
    r = make_receipt("abc", [c])
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.chunk_count = 9  # type: ignore[misc]


def test_make_receipt_fields() -> None:
    for text in _random_texts(count=50, seed=3):
        chunks = chunk_text(text, 6)
        receipt = make_receipt(text, chunks)
        assert isinstance(receipt, Receipt)
        assert receipt.content_hash == hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert receipt.size_chars == len(text)
        assert receipt.chunk_count == len(chunks)


def test_make_receipt_non_ascii_size_is_characters() -> None:
    text = "é中😀"
    receipt = make_receipt(text, chunk_text(text, 10))
    assert receipt.size_chars == 3
    assert receipt.chunk_count == 1
    assert receipt.content_hash == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_make_receipt_empty_text() -> None:
    receipt = make_receipt("", [])
    assert receipt.size_chars == 0
    assert receipt.chunk_count == 0
    assert receipt.content_hash == hashlib.sha256(b"").hexdigest()


def _sample() -> tuple[str, list[Chunk]]:
    text = "alpha\n\nbeta\n\ngamma"
    chunks = chunk_text(text, 7)
    assert len(chunks) >= 3
    return text, chunks


def test_make_receipt_rejects_missing_chunk() -> None:
    text, chunks = _sample()
    with pytest.raises(ValueError):
        make_receipt(text, chunks[:-1])


def test_make_receipt_rejects_reordered_chunks() -> None:
    text, chunks = _sample()
    with pytest.raises(ValueError):
        make_receipt(text, list(reversed(chunks)))


def test_make_receipt_rejects_altered_chunk() -> None:
    text, chunks = _sample()
    altered = [*chunks[:-1], dataclasses.replace(chunks[-1], text="GAMMA")]
    with pytest.raises(ValueError):
        make_receipt(text, altered)


def test_verify_composition_true_for_chunk_texts() -> None:
    for text in _random_texts(count=50, seed=11):
        chunks = chunk_text(text, 5)
        receipt = make_receipt(text, chunks)
        assert verify_composition(_texts(chunks), receipt) is True
        assert verify_composition(iter(_texts(chunks)), receipt) is True


def test_verify_composition_false_for_missing_part() -> None:
    text, chunks = _sample()
    receipt = make_receipt(text, chunks)
    assert verify_composition(_texts(chunks)[:-1], receipt) is False


def test_verify_composition_false_for_extra_part() -> None:
    text, chunks = _sample()
    receipt = make_receipt(text, chunks)
    assert verify_composition([*_texts(chunks), "x"], receipt) is False
    assert verify_composition([*_texts(chunks), ""], receipt) is False


def test_verify_composition_false_for_reordered_parts() -> None:
    text, chunks = _sample()
    receipt = make_receipt(text, chunks)
    assert verify_composition(list(reversed(_texts(chunks))), receipt) is False


def test_verify_composition_false_for_altered_part() -> None:
    text, chunks = _sample()
    receipt = make_receipt(text, chunks)
    parts = _texts(chunks)
    parts[0] = parts[0].upper()
    assert verify_composition(parts, receipt) is False
    parts = _texts(chunks)
    parts[1] = parts[1] + " "
    assert verify_composition(parts, receipt) is False


def test_verify_composition_false_when_parts_merged() -> None:
    text, chunks = _sample()
    receipt = make_receipt(text, chunks)
    assert verify_composition([text], receipt) is False
