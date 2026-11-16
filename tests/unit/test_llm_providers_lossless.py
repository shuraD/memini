"""Acceptance tests for LT-05: the lossless embedder (batch, one by one, then split and pool)."""

from __future__ import annotations

import inspect
import math
from collections.abc import Sequence
from typing import Literal

import pytest

from memini.infra.llm_providers.lossless import LosslessEmbedder, halve, pool
from memini.infra.llm_providers.types import Attempt, Embedder, LLMError, Reason

KINDS: tuple[Literal["query", "document"], ...] = ("query", "document")


def _rejected(detail: str) -> LLMError:
    return LLMError((Attempt("fake", Reason.REQUEST_REJECTED, detail),), "embedding")


def _vector(text: str) -> list[float]:
    """A deterministic, non-degenerate vector for a text."""
    return [
        float(len(text)),
        float(text.count("x")) + 0.5,
        float(sum(map(ord, text)) % 97) + 1.0,
    ]


class WindowEmbedder:
    """Accepts a call only when every text fits the window (and, optionally, it holds one text).

    Records every call as (texts, kind) and every text it embedded successfully.
    """

    def __init__(self, *, window: int, reject_batches: bool = False) -> None:
        self.window = window
        self.reject_batches = reject_batches
        self.calls: list[tuple[list[str], str]] = []
        self.embedded: list[str] = []

    def embed(
        self, texts: Sequence[str], *, kind: Literal["query", "document"]
    ) -> list[list[float]]:
        batch = list(texts)
        self.calls.append((batch, kind))
        if self.reject_batches and len(batch) > 1:
            raise _rejected("batch too large")
        if any(len(t) > self.window for t in batch):
            raise _rejected("text too long")
        self.embedded.extend(batch)
        return [_vector(t) for t in batch]


class FailingEmbedder:
    """Raises a fixed error on every call and records the calls."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[tuple[list[str], str]] = []

    def embed(
        self, texts: Sequence[str], *, kind: Literal["query", "document"]
    ) -> list[list[float]]:
        self.calls.append((list(texts), kind))
        raise self.error


class BatchLLMErrorThenOther:
    """Rejects a batch with LLMError, then fails a single text with another error."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def embed(
        self, texts: Sequence[str], *, kind: Literal["query", "document"]
    ) -> list[list[float]]:
        self.calls.append((list(texts), kind))
        if len(self.calls) == 1:
            raise _rejected("batch too large")
        raise RuntimeError("backend crashed")


def _expected(text: str, window: int) -> list[float]:
    """Reference: the text's vector, splitting with halve() and pooling until each piece fits."""
    if len(text) <= window:
        return _vector(text)
    a, b = halve(text)
    return pool(_expected(a, window), len(a), _expected(b, window), len(b))


def _close(got: Sequence[float], want: Sequence[float]) -> bool:
    return len(got) == len(want) and all(
        math.isclose(g, w, abs_tol=1e-9) for g, w in zip(got, want, strict=True)
    )


# --- halve -------------------------------------------------------------------------------


def test_halve_without_newline_cuts_at_len_div_two() -> None:
    assert halve("abcdefgh") == ("abcd", "efgh")
    assert halve("abcde") == ("ab", "cde")


def test_halve_cuts_at_newline_past_first_quarter_newline_starts_second_half() -> None:
    text = "aaaa\nbbbbbbbbbb"  # len 15, mid 7, newline at 4 > 15 // 4 == 3
    assert halve(text) == ("aaaa", "\nbbbbbbbbbb")


def test_halve_uses_the_last_newline_before_the_middle() -> None:
    text = "a" * 5 + "\n" + "b" * 2 + "\n" + "c" * 12  # len 21, mid 10, newlines at 5 and 8
    a, b = halve(text)
    assert (a, b) == (text[:8], text[8:])
    assert b.startswith("\n")


def test_halve_ignores_newline_within_first_quarter() -> None:
    text = "a\nbbbbbbbbbbbb"  # len 14, newline at 1, not > 3
    assert halve(text) == (text[:7], text[7:])


def test_halve_ignores_newline_exactly_at_first_quarter() -> None:
    text = "aaaa\n" + "b" * 11  # len 16, newline at 4, not > 16 // 4 == 4
    assert halve(text) == (text[:8], text[8:])


def test_halve_ignores_newline_after_the_middle() -> None:
    text = "a" * 12 + "\n" + "b" * 7  # len 20, mid 10, the only newline is at 12
    assert halve(text) == ("a" * 10, "aa\n" + "b" * 7)


@pytest.mark.parametrize(
    "text",
    ["", "x", "ab", "line one\nline two\nline three\n", "\n\n\n\n", "é漢字\nmixed🙂text" * 3],
)
def test_halve_never_loses_a_character(text: str) -> None:
    a, b = halve(text)
    assert a + b == text


# --- pool --------------------------------------------------------------------------------


def test_pool_equal_lengths_is_normalised_mean() -> None:
    got = pool([1.0, 0.0], 1, [0.0, 1.0], 1)
    assert isinstance(got, list)
    assert _close(got, [1 / math.sqrt(2), 1 / math.sqrt(2)])


def test_pool_is_length_weighted() -> None:
    got = pool([1.0, 0.0], 3, [0.0, 1.0], 1)
    assert _close(got, [3 / math.sqrt(10), 1 / math.sqrt(10)])


def test_pool_result_has_unit_norm() -> None:
    got = pool([3.0, 4.0, 0.0], 10, [1.0, 2.0, 2.0], 5)
    mean = [(3.0 * 10 + 1.0 * 5) / 15, (4.0 * 10 + 2.0 * 5) / 15, (0.0 * 10 + 2.0 * 5) / 15]
    norm = math.sqrt(sum(m * m for m in mean))
    assert _close(got, [m / norm for m in mean])
    assert math.isclose(math.sqrt(sum(g * g for g in got)), 1.0, abs_tol=1e-9)


def test_pool_zero_mean_returns_zeros() -> None:
    got = pool([1.0, -2.0], 2, [-1.0, 2.0], 2)
    assert got == [0.0, 0.0]


def test_pool_zero_lengths_do_not_divide_by_zero() -> None:
    assert pool([0.0, 0.0, 0.0], 0, [0.0, 0.0, 0.0], 0) == [0.0, 0.0, 0.0]


def test_pool_accepts_tuples() -> None:
    got = pool((2.0, 0.0), 1, (0.0, 0.0), 1)
    assert _close(got, [1.0, 0.0])


# --- embed: batch path -------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_accepted_batch_is_one_call_with_all_texts_and_kind(
    kind: Literal["query", "document"],
) -> None:
    inner = WindowEmbedder(window=1000)
    texts = ("alpha", "beta", "gamma xx")
    got = LosslessEmbedder(inner).embed(texts, kind=kind)
    assert inner.calls == [(list(texts), kind)]
    assert got == [_vector(t) for t in texts]
    assert all(isinstance(v, list) for v in got)
    assert all(isinstance(x, float) for v in got for x in v)


def test_accepted_batch_tuple_vectors_are_returned_as_lists() -> None:
    class TupleEmbedder:
        def embed(
            self, texts: Sequence[str], *, kind: Literal["query", "document"]
        ) -> list[list[float]]:
            return [tuple(_vector(t)) for t in texts]  # type: ignore[misc]

    got = LosslessEmbedder(TupleEmbedder()).embed(["a", "bb"], kind="document")
    assert got == [_vector("a"), _vector("bb")]
    assert all(type(v) is list for v in got)


@pytest.mark.parametrize("empty", [[], ()])
def test_empty_input_returns_empty_and_never_calls_inner(empty: Sequence[str]) -> None:
    inner = WindowEmbedder(window=1000)
    assert LosslessEmbedder(inner).embed(empty, kind="query") == []
    assert inner.calls == []


# --- embed: one text at a time -----------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_rejected_batch_falls_back_to_one_text_at_a_time_in_order(
    kind: Literal["query", "document"],
) -> None:
    inner = WindowEmbedder(window=1000, reject_batches=True)
    texts = ["first", "second x", "third xx"]
    got = LosslessEmbedder(inner).embed(texts, kind=kind)
    assert got == [_vector(t) for t in texts]
    assert inner.calls == [(texts, kind)] + [([t], kind) for t in texts]


def test_mixed_batch_only_the_long_text_is_split() -> None:
    window = 40
    long_text = "x" * 30 + "y" * 70  # 100 chars, needs splitting
    texts = ["short one", long_text, "short two"]
    inner = WindowEmbedder(window=window)
    got = LosslessEmbedder(inner, min_split=8).embed(texts, kind="document")
    assert len(got) == 3
    assert got[0] == _vector("short one")
    assert got[2] == _vector("short two")
    assert _close(got[1], _expected(long_text, window))
    assert inner.calls[0] == (texts, "document")
    assert inner.calls[1] == (["short one"], "document")
    assert inner.calls[-1] == (["short two"], "document")


# --- embed: split and pool ---------------------------------------------------------------


def test_long_text_vector_is_pool_of_its_halves() -> None:
    text = "a" * 60 + "x" * 60  # 120 chars, window 70: one split into 60 + 60
    inner = WindowEmbedder(window=70)
    got = LosslessEmbedder(inner, min_split=8).embed([text], kind="document")
    a, b = halve(text)
    assert len(got) == 1
    assert _close(got[0], pool(_vector(a), len(a), _vector(b), len(b)))


def test_long_text_is_split_recursively_and_no_character_is_dropped() -> None:
    lines = [f"line {i} " + "x" * (i % 7) + "word " * (i % 5) for i in range(40)]
    text = "\n".join(lines)
    window = 50
    assert len(text) > 4 * window
    inner = WindowEmbedder(window=window)
    got = LosslessEmbedder(inner, min_split=8).embed([text], kind="query")
    assert len(got) == 1
    assert _close(got[0], _expected(text, window))
    assert "".join(inner.embedded) == text
    assert all(len(piece) <= window for piece in inner.embedded)


@pytest.mark.parametrize("kind", KINDS)
def test_every_inner_call_including_split_pieces_gets_the_callers_kind(
    kind: Literal["query", "document"],
) -> None:
    inner = WindowEmbedder(window=30)
    texts = ["tiny", "z" * 200, "another " * 20]
    LosslessEmbedder(inner, min_split=8).embed(texts, kind=kind)
    assert len(inner.calls) > len(texts) + 1
    assert {k for _, k in inner.calls} == {kind}


# --- embed: limits -----------------------------------------------------------------------


def test_rejected_text_shorter_than_min_split_raises_llm_error() -> None:
    inner = WindowEmbedder(window=10)
    with pytest.raises(LLMError):
        LosslessEmbedder(inner, min_split=64).embed(["y" * 50], kind="document")


def test_default_min_split_is_64() -> None:
    with pytest.raises(LLMError):
        LosslessEmbedder(WindowEmbedder(window=40)).embed(["y" * 63], kind="document")
    got = LosslessEmbedder(WindowEmbedder(window=40)).embed(["y" * 64], kind="document")
    assert _close(got[0], _expected("y" * 64, 40))


def test_a_piece_below_min_split_that_is_still_rejected_raises() -> None:
    # 100 chars split into 50 + 50; the 50-char pieces are rejected and shorter than 60.
    inner = WindowEmbedder(window=20)
    with pytest.raises(LLMError):
        LosslessEmbedder(inner, min_split=60).embed(["q" * 100], kind="query")


def test_max_depth_zero_never_splits() -> None:
    inner = WindowEmbedder(window=10)
    with pytest.raises(LLMError):
        LosslessEmbedder(inner, max_depth=0, min_split=1).embed(["w" * 100], kind="document")
    assert all(len(texts) == 1 and texts[0] == "w" * 100 for texts, _ in inner.calls)


def test_split_deeper_than_max_depth_raises_but_enough_depth_succeeds() -> None:
    text = "v" * 40  # window 10: depth 2 gives pieces of 10
    with pytest.raises(LLMError):
        LosslessEmbedder(WindowEmbedder(window=10), max_depth=1, min_split=1).embed(
            [text], kind="query"
        )
    got = LosslessEmbedder(WindowEmbedder(window=10), max_depth=2, min_split=1).embed(
        [text], kind="query"
    )
    assert _close(got[0], _expected(text, 10))


def test_default_max_depth_is_8() -> None:
    # Window 1 with min_split 1: 256 chars reach 1-char pieces at depth 8; 257 chars do not.
    got = LosslessEmbedder(WindowEmbedder(window=1), min_split=1).embed(["u" * 256], kind="query")
    assert _close(got[0], _expected("u" * 256, 1))
    with pytest.raises(LLMError):
        LosslessEmbedder(WindowEmbedder(window=1), min_split=1).embed(["u" * 257], kind="query")


# --- embed: other errors -----------------------------------------------------------------


def test_non_llm_error_propagates_after_a_single_inner_call() -> None:
    inner = FailingEmbedder(ValueError("boom"))
    with pytest.raises(ValueError, match="boom"):
        LosslessEmbedder(inner).embed(["a" * 500, "b"], kind="document")
    assert inner.calls == [(["a" * 500, "b"], "document")]


def test_non_llm_error_in_the_one_by_one_stage_propagates_at_once() -> None:
    inner = BatchLLMErrorThenOther()
    with pytest.raises(RuntimeError, match="backend crashed"):
        LosslessEmbedder(inner).embed(["c" * 500, "d"], kind="query")
    assert inner.calls == [(["c" * 500, "d"], "query"), (["c" * 500], "query")]


# --- construction ------------------------------------------------------------------------


def test_built_from_an_embedder_alone_and_is_an_embedder() -> None:
    wrapped = LosslessEmbedder(WindowEmbedder(window=100))
    assert isinstance(wrapped, Embedder)
    params = inspect.signature(LosslessEmbedder).parameters
    assert list(params) == ["inner", "max_depth", "min_split"]
    assert params["max_depth"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["min_split"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["max_depth"].default == 8
    assert params["min_split"].default == 64
