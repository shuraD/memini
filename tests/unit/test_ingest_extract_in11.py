"""Acceptance tests for IN-11: reading extraction replies, bisection and fallback entities."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from typing import Any

import pytest

from memini.core.ingest.extract import (
    FALLBACK_TYPE,
    REPAIR_TEMPLATE,
    WindowReply,
    extract_text,
    extract_windows,
    fallback_entities,
    read_extraction_reply,
)
from memini.infra.llm_providers.json_reply import JsonStatus, parse_json

UNREADABLE = "this is not json at all"


def ent(name: str, kind: str = "Concept") -> dict[str, Any]:
    return {"id": name, "type": kind, "content": f"about {name}"}


class Asker:
    """Fake ask: records every prompt and answers from a function of the prompt."""

    def __init__(self, answer: Callable[[str], Any]) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        return self.answer(prompt)


def queue(*replies: Any) -> Asker:
    items = list(replies)
    return Asker(lambda _prompt: items.pop(0))


class Builder:
    """Fake build_prompt: records the text it was given."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, text: str) -> str:
        self.texts.append(text)
        return "P:" + text


# --- constants and dataclass -------------------------------------------------


def test_fallback_type_value() -> None:
    assert FALLBACK_TYPE == "Fallback"


def test_repair_template_has_placeholders() -> None:
    assert isinstance(REPAIR_TEMPLATE, str)
    for placeholder in ("{prompt}", "{error}", "{reply}"):
        assert placeholder in REPAIR_TEMPLATE


def test_window_reply_defaults_and_frozen() -> None:
    reply = WindowReply(usable=False)
    assert reply.entities == ()
    assert reply.status is JsonStatus.UNREADABLE
    assert reply.reasks == 0
    assert reply.error is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        reply.usable = True  # type: ignore[misc]


# --- read_extraction_reply ----------------------------------------------------


def test_valid_list_reply() -> None:
    entities = [ent("a"), ent("b")]
    ask = queue(json.dumps(entities))
    result = read_extraction_reply(ask, "the prompt")
    assert ask.prompts == ["the prompt"]
    assert result.usable is True
    assert result.entities == tuple(entities)
    assert result.status is JsonStatus.VALID
    assert result.reasks == 0


def test_tuple_reply_with_usage_is_read() -> None:
    ask = queue((json.dumps([ent("a")]), {"tokens": 3}))
    result = read_extraction_reply(ask, "p")
    assert result.usable is True
    assert result.entities == (ent("a"),)


@pytest.mark.parametrize(
    "reply",
    [
        {"entities": [ent("a"), ent("b")]},
        {"entities": [ent("a"), ent("b")], "topics": ["t"], "relations": []},
    ],
)
def test_entities_wrapper(reply: dict[str, Any]) -> None:
    result = read_extraction_reply(queue(json.dumps(reply)), "p")
    assert result.usable is True
    assert result.entities == (ent("a"), ent("b"))


def test_entities_wrapper_non_list_value_gives_no_entities() -> None:
    result = read_extraction_reply(queue(json.dumps({"entities": "none"})), "p")
    assert result.usable is True
    assert result.entities == ()


@pytest.mark.parametrize(
    "single",
    [ent("solo"), {"id": "only-id"}, {"type": "Concept"}, {"content": "only content"}],
)
def test_single_entity_dict(single: dict[str, Any]) -> None:
    result = read_extraction_reply(queue(json.dumps(single)), "p")
    assert result.usable is True
    assert result.entities == (single,)


def test_other_dict_is_usable_without_entities() -> None:
    ask = queue(json.dumps({"topics": ["x"], "note": "nothing"}))
    result = read_extraction_reply(ask, "p")
    assert result.usable is True
    assert result.entities == ()
    assert len(ask.prompts) == 1


def test_non_dict_items_and_meta_dropped() -> None:
    reply = [ent("a"), "stray", 3, None, ["x"], {"id": "m", "type": "Meta"}, ent("b")]
    result = read_extraction_reply(queue(json.dumps(reply)), "p")
    assert result.usable is True
    assert result.entities == (ent("a"), ent("b"))


def test_meta_dropped_inside_wrapper() -> None:
    reply = {"entities": [{"id": "m", "type": "Meta", "content": "c"}, ent("a")]}
    result = read_extraction_reply(queue(json.dumps(reply)), "p")
    assert result.entities == (ent("a"),)


def test_fenced_reply_is_repaired() -> None:
    ask = queue("```json\n" + json.dumps([ent("a")]) + "\n```")
    result = read_extraction_reply(ask, "p")
    assert result.usable is True
    assert result.entities == (ent("a"),)
    assert result.status is JsonStatus.REPAIRED
    assert len(ask.prompts) == 1


def test_cut_off_reply_is_salvaged_without_duplicates() -> None:
    a, b = ent("a"), ent("b")
    text = (
        '{"entities": ['
        + json.dumps(a)
        + ", "
        + json.dumps(b)
        + ", "
        + json.dumps(a)
        + ', {"id": "c", "ty'
    )
    ask = queue(text, json.dumps([ent("zzz")]))
    result = read_extraction_reply(ask, "p", attempts=1)
    assert result.usable is True
    assert result.entities == (a, b)
    assert result.status is JsonStatus.SALVAGED
    assert len(ask.prompts) == 1


def test_cut_off_reply_without_complete_object_is_unusable() -> None:
    ask = queue('{"entities": [{"id": "a", "ty', json.dumps([ent("zzz")]))
    result = read_extraction_reply(ask, "p", attempts=1)
    assert result.usable is False
    assert result.entities == ()
    assert result.error
    assert len(ask.prompts) == 1


def test_unreadable_then_valid_reasks_with_template() -> None:
    template = "FIX<{prompt}>ERR<{error}>REPLY<{reply}>"
    ask = queue(UNREADABLE, json.dumps([ent("a")]))
    result = read_extraction_reply(ask, "orig prompt", attempts=1, repair_template=template)
    error = parse_json(UNREADABLE).error
    assert error
    assert len(ask.prompts) == 2
    assert ask.prompts[0] == "orig prompt"
    assert ask.prompts[1] == f"FIX<orig prompt>ERR<{error}>REPLY<{UNREADABLE}>"
    assert result.usable is True
    assert result.entities == (ent("a"),)
    assert result.status is JsonStatus.REASKED
    assert result.reasks == 1


def test_default_template_used_for_reask() -> None:
    ask = queue(UNREADABLE, json.dumps([ent("a")]))
    result = read_extraction_reply(ask, "orig prompt")
    assert len(ask.prompts) == 2
    assert "orig prompt" in ask.prompts[1]
    assert UNREADABLE in ask.prompts[1]
    assert result.status is JsonStatus.REASKED


def test_every_reply_unreadable() -> None:
    ask = Asker(lambda _p: UNREADABLE)
    result = read_extraction_reply(ask, "p", attempts=2)
    assert result.usable is False
    assert result.entities == ()
    assert result.status is JsonStatus.UNREADABLE
    assert isinstance(result.error, str) and result.error
    assert len(ask.prompts) == 3


def test_attempts_zero_means_no_reask() -> None:
    ask = Asker(lambda _p: UNREADABLE)
    result = read_extraction_reply(ask, "p", attempts=0)
    assert result.usable is False
    assert len(ask.prompts) == 1


def test_none_reply_is_unusable() -> None:
    result = read_extraction_reply(Asker(lambda _p: None), "p")
    assert result.usable is False
    assert result.entities == ()
    assert result.error


def test_ask_raising_is_caught() -> None:
    def boom(_prompt: str) -> Any:
        raise RuntimeError("provider down")

    result = read_extraction_reply(boom, "p")
    assert result.usable is False
    assert result.entities == ()
    assert result.status is JsonStatus.UNREADABLE
    assert isinstance(result.error, str) and result.error


# --- extract_text ------------------------------------------------------------


def test_extract_text_usable_no_split() -> None:
    text = "x" * 1000
    build = Builder()
    ask = Asker(lambda _p: json.dumps([ent("a"), ent("b")]))
    out = extract_text(ask, text, build)
    assert build.texts == [text]
    assert out == [ent("a"), ent("b")]


def test_extract_text_usable_empty_no_split() -> None:
    text = "x" * 1000
    build = Builder()
    ask = Asker(lambda _p: "[]")
    out = extract_text(ask, text, build)
    assert build.texts == [text]
    assert out == []
    assert ask.prompts == ["P:" + text]


def _by_piece(replies: dict[str, Any]) -> Asker:
    """Answer the prompt built for a known piece; anything else is unreadable."""
    return Asker(lambda p: replies.get(p, UNREADABLE))


def test_split_at_last_newline_in_range() -> None:
    text = "a" * 280 + "\n" + "b" * 19 + "\n" + "c" * 700  # len 1001
    assert len(text) == 1001
    cut = 300
    assert text[cut] == "\n"
    left, right = text[:cut], text[cut:]
    ask = _by_piece(
        {
            "P:" + left: json.dumps([ent("left")]),
            "P:" + right: json.dumps([ent("right")]),
        }
    )
    build = Builder()
    out = extract_text(ask, text, build)
    assert build.texts == [text, left, right]
    assert right.startswith("\n")
    assert out == [ent("left"), ent("right")]


def test_newline_before_quarter_is_ignored() -> None:
    text = "a" * 100 + "\n" + "b" * 899  # len 1000, newline at 100 <= 250
    mid = len(text) // 2
    left, right = text[:mid], text[mid:]
    ask = _by_piece(
        {
            "P:" + left: json.dumps([ent("left")]),
            "P:" + right: json.dumps([ent("right")]),
        }
    )
    build = Builder()
    out = extract_text(ask, text, build)
    assert build.texts == [text, left, right]
    assert out == [ent("left"), ent("right")]


def test_newline_after_mid_is_ignored() -> None:
    text = "a" * 700 + "\n" + "b" * 299  # len 1000, newline only after mid
    left, right = text[:500], text[500:]
    ask = _by_piece(
        {
            "P:" + left: json.dumps([ent("left")]),
            "P:" + right: json.dumps([ent("right")]),
        }
    )
    build = Builder()
    out = extract_text(ask, text, build)
    assert build.texts == [text, left, right]
    assert out == [ent("left"), ent("right")]


def test_no_newline_cuts_at_half() -> None:
    text = "".join(chr(ord("a") + i % 26) for i in range(901))
    mid = len(text) // 2
    left, right = text[:mid], text[mid:]
    ask = _by_piece({"P:" + left: json.dumps([ent("L")]), "P:" + right: json.dumps([ent("R")])})
    build = Builder()
    out = extract_text(ask, text, build)
    assert build.texts == [text, left, right]
    assert out == [ent("L"), ent("R")]


def test_always_unusable_default_recursion_order() -> None:
    text = "".join(chr(ord("a") + i % 26) for i in range(1000))
    build = Builder()
    ask = Asker(lambda _p: UNREADABLE)
    out = extract_text(ask, text, build)
    assert out == []
    assert [len(t) for t in build.texts] == [1000, 500, 250, 250, 500, 250, 250]
    assert build.texts[1] == text[:500]
    assert build.texts[2] + build.texts[3] == text[:500]
    assert build.texts[4] == text[500:]
    assert build.texts[5] + build.texts[6] == text[500:]


def test_split_stops_at_max_depth() -> None:
    text = "z" * 1000
    build = Builder()
    out = extract_text(Asker(lambda _p: UNREADABLE), text, build, max_depth=1, min_chars=10)
    assert out == []
    assert [len(t) for t in build.texts] == [1000, 500, 500]


def test_short_text_not_split() -> None:
    text = "z" * 400
    build = Builder()
    out = extract_text(Asker(lambda _p: UNREADABLE), text, build)
    assert out == []
    assert build.texts == [text]


def test_depth_argument_counts_toward_max_depth() -> None:
    text = "z" * 1000
    build = Builder()
    out = extract_text(Asker(lambda _p: UNREADABLE), text, build, max_depth=4, depth=4)
    assert out == []
    assert build.texts == [text]


def test_knobs_forwarded_to_halves() -> None:
    text = "q" * 1000
    build = Builder()
    template = "AGAIN {prompt} {error} {reply}"
    ask = Asker(lambda _p: UNREADABLE)
    extract_text(ask, text, build, max_depth=1, min_chars=10, attempts=2, repair_template=template)
    # 3 pieces, each asked once plus 2 re-asks
    assert len(ask.prompts) == 9
    assert sum(p.startswith("AGAIN ") for p in ask.prompts) == 6


def test_ask_raising_for_whole_text_splits() -> None:
    text = "".join(chr(ord("a") + i % 26) for i in range(1000))
    left, right = text[:500], text[500:]

    def answer(prompt: str) -> Any:
        if prompt == "P:" + text:
            raise RuntimeError("too long")
        return {"P:" + left: json.dumps([ent("L")]), "P:" + right: json.dumps([ent("R")])}.get(
            prompt, UNREADABLE
        )

    build = Builder()
    out = extract_text(Asker(answer), text, build)
    assert out == [ent("L"), ent("R")]
    assert build.texts == [text, left, right]


# --- fallback_entities and extract_windows -----------------------------------


def test_fallback_entities() -> None:
    assert fallback_entities("A", ["x", "y"]) == [
        {"id": "A_Fallback_0", "type": "Fallback", "content": "x"},
        {"id": "A_Fallback_1", "type": "Fallback", "content": "y"},
    ]


def test_fallback_entities_keep_full_text() -> None:
    long_window = "w" * 50_000
    out = fallback_entities("doc-1", [long_window])
    assert out == [{"id": "doc-1_Fallback_0", "type": FALLBACK_TYPE, "content": long_window}]


def test_fallback_entities_empty() -> None:
    assert fallback_entities("A", []) == []


def test_extract_windows_concatenates_in_order() -> None:
    windows = ["w1", "w2", "w3"]
    ask = _by_piece(
        {
            "P:w1": json.dumps([ent("a"), ent("b")]),
            "P:w2": "[]",
            "P:w3": json.dumps({"entities": [ent("c")]}),
        }
    )
    build = Builder()
    out = extract_windows(ask, windows, build, "ART")
    assert out == [ent("a"), ent("b"), ent("c")]
    assert build.texts == windows
    assert all(e["type"] != FALLBACK_TYPE for e in out)


def test_extract_windows_some_unusable_no_fallback() -> None:
    ask = _by_piece({"P:w2": json.dumps([ent("only")])})
    out = extract_windows(ask, ["w1", "w2"], Builder(), "ART")
    assert out == [ent("only")]


def test_extract_windows_empty_replies_give_fallback() -> None:
    windows = ["first window", "second window"]
    out = extract_windows(Asker(lambda _p: "[]"), windows, Builder(), "ART")
    assert out == fallback_entities("ART", windows)


def test_extract_windows_unusable_replies_give_fallback() -> None:
    windows = ["first window", "second window"]
    out = extract_windows(Asker(lambda _p: UNREADABLE), windows, Builder(), "ART")
    assert out == fallback_entities("ART", windows)


def test_extract_windows_forwards_knobs() -> None:
    window = "k" * 1000
    build = Builder()
    template = "AGAIN {prompt} {error} {reply}"
    ask = Asker(lambda _p: UNREADABLE)
    out = extract_windows(
        ask,
        [window],
        build,
        "ART",
        max_depth=1,
        min_chars=10,
        attempts=0,
        repair_template=template,
    )
    assert [len(t) for t in build.texts] == [1000, 500, 500]
    assert len(ask.prompts) == 3
    assert out == fallback_entities("ART", [window])
