"""Acceptance tests for LT-01: deterministic reading of one model reply as JSON."""

from __future__ import annotations

import dataclasses
import enum
import json

import pytest

from memini.infra.llm_providers import json_reply as jr

JsonReading = jr.JsonReading
Repair = jr.Repair
read_json = jr.read_json


def _assert_failed(reading: JsonReading) -> None:
    assert reading.ok is False
    assert reading.value is None
    assert isinstance(reading.error, str)
    assert reading.error.strip() != ""


# --- types -------------------------------------------------------------------


def test_repair_members() -> None:
    assert issubclass(Repair, enum.StrEnum)
    names = {member.name for member in Repair}
    assert names == {"FENCES", "OUTERMOST", "SYNTAX", "SALVAGE", "DOUBLE_ENCODED"}


def test_json_reading_is_frozen_dataclass_with_defaults() -> None:
    assert dataclasses.is_dataclass(JsonReading)
    reading = JsonReading(ok=True)
    assert reading.value is None
    assert reading.error is None
    assert reading.truncated is False
    assert reading.repairs == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        reading.ok = False  # type: ignore[misc]


# --- valid JSON --------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1, "b": [true, false, null], "c": "x"}',
        '[1, 2, {"k": "v"}]',
        "42",
        "-3.5",
        '"plain words"',
        '  {"padded": true}  \n',
    ],
)
def test_valid_json_is_read_without_repairs(text: str) -> None:
    reading = read_json(text)
    assert reading.ok is True
    assert reading.value == json.loads(text)
    assert reading.error is None
    assert reading.repairs == ()
    assert reading.truncated is False


# --- empty and prose ---------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n\t  \n"])
def test_empty_or_whitespace_reply_fails(text: str) -> None:
    _assert_failed(read_json(text))


@pytest.mark.parametrize(
    "text",
    [
        "I am sorry, I cannot produce that output.",
        "Here is the answer: nothing found.",
    ],
)
def test_prose_without_json_fails_not_truncated(text: str) -> None:
    reading = read_json(text)
    _assert_failed(reading)
    assert reading.truncated is False


# --- fences and outermost ----------------------------------------------------


def test_closed_json_fence_is_stripped() -> None:
    reading = read_json('```json\n{"a": 1, "b": "two"}\n```')
    assert reading.ok is True
    assert reading.value == {"a": 1, "b": "two"}
    assert Repair.FENCES in reading.repairs
    assert reading.truncated is False
    assert reading.error is None


def test_unclosed_json_fence_at_end_is_stripped() -> None:
    reading = read_json('```json\n{"a": 1, "b": "two"}\n')
    assert reading.ok is True
    assert reading.value == {"a": 1, "b": "two"}
    assert Repair.FENCES in reading.repairs
    assert reading.truncated is False


def test_prose_around_one_object_is_ignored() -> None:
    reading = read_json('Sure! Here it is: {"name": "x", "n": 2} Hope this helps.')
    assert reading.ok is True
    assert reading.value == {"name": "x", "n": 2}
    assert Repair.OUTERMOST in reading.repairs
    assert reading.truncated is False


# --- syntax slips ------------------------------------------------------------


def test_trailing_commas_are_dropped() -> None:
    reading = read_json('{"a": [1, 2, 3,], "b": {"c": 4,},}')
    assert reading.ok is True
    assert reading.value == {"a": [1, 2, 3], "b": {"c": 4}}
    assert Repair.SYNTAX in reading.repairs
    assert reading.truncated is False


def test_trailing_comma_with_whitespace_before_bracket() -> None:
    reading = read_json("[1, 2 ,\n  ]")
    assert reading.ok is True
    assert reading.value == [1, 2]
    assert Repair.SYNTAX in reading.repairs


def test_single_quoted_keys_and_values_become_json_strings() -> None:
    reading = read_json("{'name': 'Ann', \"kind\": 'person'}")
    assert reading.ok is True
    assert reading.value == {"name": "Ann", "kind": "person"}
    assert Repair.SYNTAX in reading.repairs


def test_escaped_single_quote_inside_single_quoted_string() -> None:
    reading = read_json("{'text': 'it\\'s fine'}")
    assert reading.ok is True
    assert reading.value == {"text": "it's fine"}
    assert Repair.SYNTAX in reading.repairs


def test_bare_python_literals_are_read_as_json_literals() -> None:
    reading = read_json('{"a": True, "b": False, "c": None, "d": [True, None]}')
    assert reading.ok is True
    assert reading.value == {"a": True, "b": False, "c": None, "d": [True, None]}
    assert Repair.SYNTAX in reading.repairs


def test_double_quoted_strings_are_left_byte_for_byte() -> None:
    inner = {
        "commas": "x,} and y, ]",
        "quotes": "it's 'quoted'",
        "words": "True False None",
    }
    text = (
        '{"commas": "x,} and y, ]", "quotes": "it\'s \'quoted\'", '
        '"words": "True False None", "flag": True, "list": [1, 2,],}'
    )
    reading = read_json(text)
    assert reading.ok is True
    assert isinstance(reading.value, dict)
    for key, expected in inner.items():
        assert reading.value[key] == expected
    assert reading.value["flag"] is True
    assert reading.value["list"] == [1, 2]
    assert Repair.SYNTAX in reading.repairs


def test_closed_malformed_reply_fails_not_truncated() -> None:
    reading = read_json('{"a": 1 "b": 2}')
    _assert_failed(reading)
    assert reading.truncated is False


# --- salvage of cut-off replies ----------------------------------------------


def test_cut_off_wrapper_salvages_complete_elements() -> None:
    text = '{"items": [{"id": 1, "n": "a"}, {"id": 2, "n": "b,}"}, {"id": 3, "n": "c'
    reading = read_json(text)
    assert reading.ok is True
    assert reading.value == [{"id": 1, "n": "a"}, {"id": 2, "n": "b,}"}]
    assert reading.truncated is True
    assert Repair.SALVAGE in reading.repairs
    assert reading.error is None


def test_cut_off_bare_array_salvages_complete_elements() -> None:
    reading = read_json('[{"id": 1}, {"id": 2, "tags": ["x", "y"]}, {"id": 3, "ta')
    assert reading.ok is True
    assert reading.value == [{"id": 1}, {"id": 2, "tags": ["x", "y"]}]
    assert reading.truncated is True
    assert Repair.SALVAGE in reading.repairs


def test_salvage_key_selects_named_array() -> None:
    text = '{"notes": [{"x": 1}, {"x": 2}], "items": [{"id": 10}, {"id": 11}, {"id": 12, "na'
    reading = read_json(text, salvage_key="items")
    assert reading.ok is True
    assert reading.value == [{"id": 10}, {"id": 11}]
    assert reading.truncated is True
    assert Repair.SALVAGE in reading.repairs


def test_salvage_key_absent_falls_back_to_first_array() -> None:
    text = '{"items": [{"id": 1}, {"id": 2}, {"id": 3'
    reading = read_json(text, salvage_key="missing")
    assert reading.ok is True
    assert reading.value == [{"id": 1}, {"id": 2}]
    assert reading.truncated is True
    assert Repair.SALVAGE in reading.repairs


def test_cut_off_scalar_number_touching_end_is_not_salvaged() -> None:
    reading = read_json("[10, 20, 30")
    assert reading.ok is True
    assert reading.value == [10, 20]
    assert reading.truncated is True
    assert Repair.SALVAGE in reading.repairs


def test_cut_off_scalar_strings_salvaged_before_cut_number() -> None:
    reading = read_json('["a", "b", 12')
    assert reading.ok is True
    assert reading.value == ["a", "b"]
    assert reading.truncated is True


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1, "b": "hel',
        '{"a": {"b": 1, "c": 2',
        '[{"id": 1, "na',
        '{"items": [{"id": 1',
    ],
)
def test_cut_off_without_complete_element_fails(text: str) -> None:
    reading = read_json(text)
    _assert_failed(reading)
    assert reading.truncated is True


# --- double-encoded ----------------------------------------------------------


def test_double_encoded_object_is_decoded() -> None:
    text = json.dumps(json.dumps({"a": 1, "b": [1, 2]}))
    reading = read_json(text)
    assert reading.ok is True
    assert reading.value == {"a": 1, "b": [1, 2]}
    assert Repair.DOUBLE_ENCODED in reading.repairs


def test_double_encoded_array_is_decoded() -> None:
    text = json.dumps(json.dumps([{"id": 1}, {"id": 2}]))
    reading = read_json(text)
    assert reading.ok is True
    assert reading.value == [{"id": 1}, {"id": 2}]
    assert Repair.DOUBLE_ENCODED in reading.repairs


def test_double_encoded_inner_content_gets_same_repairs() -> None:
    text = json.dumps("{'a': True, 'b': [1, 2,],}")
    reading = read_json(text)
    assert reading.ok is True
    assert reading.value == {"a": True, "b": [1, 2]}
    assert Repair.DOUBLE_ENCODED in reading.repairs


@pytest.mark.parametrize("content", ["hello world", "42", "true", "{not json"])
def test_string_without_json_container_is_returned_unchanged(content: str) -> None:
    reading = read_json(json.dumps(content))
    assert reading.ok is True
    assert reading.value == content
    assert Repair.DOUBLE_ENCODED not in reading.repairs


# --- determinism and robustness ----------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": True,}\n```',
        '[{"id": 1}, {"id": 2',
        "no json here",
        "",
        json.dumps(json.dumps({"x": [1]})),
    ],
)
def test_same_input_gives_equal_readings(text: str) -> None:
    assert read_json(text) == read_json(text)


@pytest.mark.parametrize(
    "text",
    [
        "{",
        "[",
        "}",
        "]]]",
        '"',
        "'",
        "```",
        "```json",
        "{'a': '",
        '{"a": "\\',
        "\x00﻿{}",
        "[" * 5000,
        "{]",
        "null",
    ],
)
def test_never_raises_and_result_is_consistent(text: str) -> None:
    reading = read_json(text)
    assert isinstance(reading, JsonReading)
    if reading.ok:
        assert reading.error is None
    else:
        assert reading.value is None
        assert isinstance(reading.error, str)
        assert reading.error != ""
    assert all(isinstance(step, Repair) for step in reading.repairs)
