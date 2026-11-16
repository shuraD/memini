"""Acceptance tests for LT-02: schema validation and the JSON re-ask flow."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

import pytest

from memini.infra.llm_providers import json_reply as jr

JsonResult = jr.JsonResult
JsonStatus = jr.JsonStatus
Repair = jr.Repair
agenerate_json = jr.agenerate_json
generate_json = jr.generate_json
parse_json = jr.parse_json
validate = jr.validate

TEMPLATE = "P={prompt}|E={error}|R={reply}|S={schema}"

OBJ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"a": {"type": "integer"}},
    "required": ["a"],
}


def _render(template: str, prompt: str, error: str, reply: str, schema: Any) -> str:
    """Single-pass expected rendering of the re-ask template, written independently."""
    values = {
        "{prompt}": prompt,
        "{error}": error,
        "{reply}": reply,
        "{schema}": json.dumps(schema or {}, ensure_ascii=False),
    }
    out, i = [], 0
    while i < len(template):
        for key, value in values.items():
            if template.startswith(key, i):
                out.append(value)
                i += len(key)
                break
        else:
            out.append(template[i])
            i += 1
    return "".join(out)


class _Ask:
    """Sync fake model: returns the scripted replies in order and records prompts."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        return self.replies[len(self.prompts) - 1]


class _AsyncAsk(_Ask):
    async def __call__(self, prompt: str) -> Any:  # type: ignore[override]
        return super().__call__(prompt)


def _errors(value: Any, schema: Any) -> tuple[str, ...]:
    return validate(value, schema)[1]


# --- validate ----------------------------------------------------------------


@pytest.mark.parametrize("schema", [None, {}])
@pytest.mark.parametrize("value", [{"x": "1"}, ["1", 2], "true", 3.0, None])
def test_validate_without_schema_returns_value_unchanged(value: Any, schema: Any) -> None:
    fixed, errors = validate(value, schema)
    assert errors == ()
    assert fixed == value
    assert type(fixed) is type(value)


def test_validate_matching_value_has_no_errors() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
            "score": {"type": "number"},
            "flag": {"type": "boolean"},
            "note": {"type": ["string", "null"]},
            "kind": {"enum": ["alpha", "beta"]},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 3,
            },
        },
        "required": ["name", "count"],
    }
    value = {
        "name": "n",
        "count": 2,
        "score": 1.5,
        "flag": False,
        "note": None,
        "kind": "beta",
        "tags": ["t1", "t2"],
    }
    fixed, errors = validate(value, schema)
    assert errors == ()
    assert fixed == value


def test_validate_ignores_unknown_keywords() -> None:
    schema = {"type": "string", "pattern": "^z+$", "minLength": 99, "format": "date"}
    assert validate("abc", schema) == ("abc", ())


@pytest.mark.parametrize(
    ("value", "schema", "expected"),
    [
        ("true", {"type": "boolean"}, True),
        (" False ", {"type": "boolean"}, False),
        ("42", {"type": "integer"}, 42),
        ("2.5", {"type": "number"}, 2.5),
        ("-3", {"type": "number"}, -3),
        (3.0, {"type": "integer"}, 3),
    ],
)
def test_validate_lossless_scalar_fixes(value: Any, schema: Any, expected: Any) -> None:
    fixed, errors = validate(value, schema)
    assert errors == ()
    assert fixed == expected
    assert type(fixed) is type(expected)


@pytest.mark.parametrize(
    ("value", "schema"),
    [
        ("1.5", {"type": "integer"}),
        (1.5, {"type": "integer"}),
        ("yes", {"type": "boolean"}),
        ("abc", {"type": "number"}),
        (True, {"type": "integer"}),
        (True, {"type": "number"}),
        (False, {"type": "number"}),
        ("x", {"type": "object"}),
    ],
)
def test_validate_rejects_lossy_or_wrong_types(value: Any, schema: Any) -> None:
    errors = _errors(value, schema)
    assert len(errors) >= 1
    assert all(isinstance(e, str) and e for e in errors)


def test_validate_type_list() -> None:
    schema = {"type": ["integer", "null"]}
    assert validate(None, schema) == (None, ())
    assert validate(4, schema) == (4, ())
    assert _errors("abc", schema) != ()


def test_validate_unwraps_single_key_object_holding_list() -> None:
    schema = {"type": "array", "items": {"type": "integer"}}
    fixed, errors = validate({"results": [1, "2"]}, schema)
    assert errors == ()
    assert fixed == [1, 2]


def test_validate_enum_case_fix_and_mismatch() -> None:
    schema = {"enum": ["Apple", "pear"]}
    assert validate("apple", schema) == ("Apple", ())
    assert validate("PEAR", schema) == ("pear", ())
    assert validate("pear", schema) == ("pear", ())
    assert _errors("plum", schema) != ()


def test_validate_renames_near_matching_keys() -> None:
    schema = {
        "type": "object",
        "properties": {"itemname": {"type": "string"}, "count": {"type": "integer"}},
    }
    value = {"Item_Name": "x", "COUNT": "3", "other key": 1}
    fixed, errors = validate(value, schema)
    assert errors == ()
    assert fixed == {"itemname": "x", "count": 3, "other key": 1}


def test_validate_missing_required_key_named_in_error() -> None:
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title", "zebrakey"],
    }
    errors = _errors({"title": "t"}, schema)
    assert len(errors) >= 1
    assert any("zebrakey" in e for e in errors)


def test_validate_recurses_into_properties_and_items() -> None:
    schema = {
        "type": "object",
        "properties": {
            "inner": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"flag": {"type": "boolean"}},
                },
            },
        },
    }
    fixed, errors = validate({"inner": {"Count": "7"}, "rows": [{"flag": "true"}]}, schema)
    assert errors == ()
    assert fixed == {"inner": {"count": 7}, "rows": [{"flag": True}]}

    errors = _errors({"inner": {}, "rows": []}, schema)
    assert any("count" in e for e in errors)
    assert _errors({"inner": {"count": 1}, "rows": [{"flag": "maybe"}]}, schema) != ()


def test_validate_items_errors() -> None:
    schema = {"type": "array", "items": {"type": "integer"}}
    assert validate(["1", 2], schema) == ([1, 2], ())
    assert _errors([1, "x"], schema) != ()


def test_validate_min_and_max_items() -> None:
    schema = {"type": "array", "minItems": 2, "maxItems": 3}
    assert validate([1, 2], schema) == ([1, 2], ())
    assert validate([1, 2, 3], schema) == ([1, 2, 3], ())
    assert _errors([1], schema) != ()
    assert _errors([1, 2, 3, 4], schema) != ()


def test_validate_never_mutates_input() -> None:
    schema = {
        "type": "object",
        "properties": {
            "itemname": {"type": "integer"},
            "rows": {"type": "array", "items": {"type": "boolean"}},
            "kind": {"enum": ["Alpha"]},
            "wrapped": {"type": "array"},
        },
    }
    value = {
        "Item_Name": "5",
        "rows": ["true", "false"],
        "kind": "alpha",
        "wrapped": {"list": [1]},
    }
    before = copy.deepcopy(value)
    fixed, errors = validate(value, schema)
    assert errors == ()
    assert value == before
    assert fixed == {
        "itemname": 5,
        "rows": [True, False],
        "kind": "Alpha",
        "wrapped": [1],
    }


# --- JsonStatus / JsonResult -------------------------------------------------


def test_json_status_values() -> None:
    assert {s.name: s.value for s in JsonStatus} == {
        "VALID": "valid",
        "REPAIRED": "repaired",
        "SALVAGED": "salvaged",
        "REASKED": "reasked",
        "UNREADABLE": "unreadable",
    }


def test_json_result_defaults_and_status_rules() -> None:
    plain = JsonResult(ok=True, value=1)
    assert plain.parsed is None
    assert plain.error is None
    assert plain.raw == ""
    assert plain.repairs == ()
    assert plain.coerced is False
    assert plain.truncated is False
    assert plain.reasks == 0
    assert plain.status is JsonStatus.VALID
    assert JsonResult(ok=True, value=1, coerced=True).status is JsonStatus.REPAIRED
    assert JsonResult(ok=True, value=1, repairs=(Repair.SYNTAX,)).status is JsonStatus.REPAIRED
    salvaged = JsonResult(ok=True, value=[1], repairs=(Repair.FENCES, Repair.SALVAGE))
    assert salvaged.status is JsonStatus.SALVAGED
    reasked = JsonResult(ok=True, value=[1], repairs=(Repair.SALVAGE,), reasks=1)
    assert reasked.status is JsonStatus.REASKED
    failed = JsonResult(ok=False, parsed={"b": 1}, error="bad", reasks=2)
    assert failed.status is JsonStatus.UNREADABLE
    assert failed.best() == {"b": 1}
    assert plain.best() == 1


# --- parse_json --------------------------------------------------------------


def test_parse_json_valid() -> None:
    text = '{"a": 1}'
    result = parse_json(text, OBJ_SCHEMA)
    assert result.ok is True
    assert result.value == {"a": 1}
    assert result.parsed is None
    assert result.error is None
    assert result.status is JsonStatus.VALID
    assert result.raw == text
    assert result.repairs == ()
    assert result.coerced is False
    assert result.truncated is False
    assert result.reasks == 0
    assert result.best() == {"a": 1}


def test_parse_json_without_schema() -> None:
    result = parse_json("[1, 2]")
    assert result.ok is True
    assert result.value == [1, 2]
    assert result.status is JsonStatus.VALID


def test_parse_json_fenced_is_repaired() -> None:
    text = '```json\n{"a": 1}\n```'
    result = parse_json(text, OBJ_SCHEMA)
    assert result.ok is True
    assert result.value == {"a": 1}
    assert Repair.FENCES in result.repairs
    assert result.status is JsonStatus.REPAIRED
    assert result.raw == text


def test_parse_json_coerced_only_is_repaired() -> None:
    result = parse_json('{"a": "1"}', OBJ_SCHEMA)
    assert result.ok is True
    assert result.value == {"a": 1}
    assert result.coerced is True
    assert result.repairs == ()
    assert result.status is JsonStatus.REPAIRED


def test_parse_json_schema_failure_keeps_fixed_parsed_value() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    }
    result = parse_json('{"A": "1", "b": "x"}', schema)
    assert result.ok is False
    assert result.value is None
    assert result.parsed == {"a": 1, "b": "x"}
    assert isinstance(result.error, str) and result.error.strip()
    assert result.status is JsonStatus.UNREADABLE
    assert result.best() == result.parsed
    assert result.reasks == 0


def test_parse_json_unreadable() -> None:
    result = parse_json("no json at all here", OBJ_SCHEMA)
    assert result.ok is False
    assert result.value is None
    assert result.parsed is None
    assert isinstance(result.error, str) and result.error.strip()
    assert result.status is JsonStatus.UNREADABLE
    assert result.best() is None
    assert result.raw == "no json at all here"


def test_parse_json_salvaged_array_validates_items_ignores_size() -> None:
    schema = {
        "type": "array",
        "items": {"type": "object", "properties": {"n": {"type": "integer"}}},
        "minItems": 5,
        "maxItems": 1,
    }
    text = '[{"n": "1"}, {"n": 2}, {"n": 3'
    result = parse_json(text, schema)
    assert result.ok is True
    assert result.value == [{"n": 1}, {"n": 2}]
    assert result.truncated is True
    assert Repair.SALVAGE in result.repairs
    assert result.status is JsonStatus.SALVAGED
    assert result.raw == text


def test_parse_json_salvaged_items_still_checked() -> None:
    schema = {
        "type": "array",
        "items": {"type": "object", "properties": {"n": {"type": "integer"}}},
    }
    result = parse_json('[{"n": "x"}, {"n": 2}, {"n": 3', schema)
    assert result.ok is False
    assert result.value is None
    assert result.truncated is True
    assert result.parsed == [{"n": "x"}, {"n": 2}]
    assert isinstance(result.error, str) and result.error.strip()


def test_parse_json_salvaged_wrapper_not_checked_against_object_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"results": {"type": "array"}, "total": {"type": "integer"}},
        "required": ["results", "total"],
    }
    text = '{"results": [{"n": 1}, {"n": 2}, {"n"'
    result = parse_json(text, schema, salvage_key="results")
    assert result.ok is True
    assert result.value == [{"n": 1}, {"n": 2}]
    assert result.truncated is True
    assert result.status is JsonStatus.SALVAGED


# --- generate_json -----------------------------------------------------------


def test_generate_json_first_reply_valid() -> None:
    ask = _Ask(['{"a": 1}'])
    result = generate_json(ask, "PROMPT", OBJ_SCHEMA, repair_template=TEMPLATE)
    assert ask.prompts == ["PROMPT"]
    assert result.ok is True
    assert result.value == {"a": 1}
    assert result.reasks == 0
    assert result.status is JsonStatus.VALID


def test_generate_json_reask_then_valid() -> None:
    first = '{"b": 2}'
    ask = _Ask([first, '{"a": 1}'])
    result = generate_json(ask, "PROMPT", OBJ_SCHEMA, repair_template=TEMPLATE)
    error = parse_json(first, OBJ_SCHEMA).error
    assert isinstance(error, str) and error
    assert ask.prompts == ["PROMPT", _render(TEMPLATE, "PROMPT", error, first, OBJ_SCHEMA)]
    assert result.ok is True
    assert result.value == {"a": 1}
    assert result.reasks == 1
    assert result.status is JsonStatus.REASKED
    assert result.raw == '{"a": 1}'


def test_generate_json_all_invalid_default_attempts() -> None:
    ask = _Ask(["not json", "still not json", "never"])
    result = generate_json(ask, "PROMPT", OBJ_SCHEMA, repair_template=TEMPLATE)
    assert len(ask.prompts) == 2
    assert result.ok is False
    assert result.value is None
    assert isinstance(result.error, str) and result.error.strip()
    assert result.reasks == 1
    assert result.raw == "still not json"
    assert result.status is JsonStatus.UNREADABLE


@pytest.mark.parametrize("attempts", [0, -1])
def test_generate_json_no_reask_when_attempts_not_positive(attempts: int) -> None:
    ask = _Ask(["bad", '{"a": 1}'])
    result = generate_json(ask, "PROMPT", OBJ_SCHEMA, repair_template=TEMPLATE, attempts=attempts)
    assert ask.prompts == ["PROMPT"]
    assert result.ok is False
    assert result.reasks == 0
    assert result.raw == "bad"


def test_generate_json_two_attempts_all_invalid() -> None:
    replies = ["bad one", '{"b": 1}', "bad three", '{"a": 1}']
    ask = _Ask(replies)
    result = generate_json(ask, "PROMPT", OBJ_SCHEMA, repair_template=TEMPLATE, attempts=2)
    assert len(ask.prompts) == 3
    second_error = parse_json(replies[1], OBJ_SCHEMA).error
    assert isinstance(second_error, str)
    assert ask.prompts[2] == _render(TEMPLATE, "PROMPT", second_error, replies[1], OBJ_SCHEMA)
    assert result.ok is False
    assert result.reasks == 2
    assert result.raw == "bad three"


def test_generate_json_two_attempts_stops_when_valid() -> None:
    ask = _Ask(["bad", '{"a": 5}', "unused"])
    result = generate_json(ask, "PROMPT", OBJ_SCHEMA, repair_template=TEMPLATE, attempts=2)
    assert len(ask.prompts) == 2
    assert result.ok is True
    assert result.value == {"a": 5}
    assert result.reasks == 1


def test_generate_json_keeps_earlier_parsed_value() -> None:
    ask = _Ask(['{"b": 1}', "nothing decodable"])
    result = generate_json(ask, "PROMPT", OBJ_SCHEMA, repair_template=TEMPLATE)
    assert result.ok is False
    assert result.value is None
    assert result.parsed == {"b": 1}
    assert result.best() == {"b": 1}
    assert result.raw == "nothing decodable"


def test_generate_json_single_pass_fill() -> None:
    first = "oops {prompt} {x} {error}"
    prompt = "Q {reply} {schema}"
    template = "{reply}|{prompt}|{other}|{schema}"
    schema = {"enum": ["é"]}
    ask = _Ask([first, '"é"'])
    result = generate_json(ask, prompt, schema, repair_template=template)
    expected = first + "|" + prompt + "|{other}|" + json.dumps(schema, ensure_ascii=False)
    assert ask.prompts == [prompt, expected]
    assert result.ok is True
    assert result.value == "é"


def test_generate_json_schema_none_renders_empty_object() -> None:
    ask = _Ask(["bad", "[1]"])
    result = generate_json(ask, "PROMPT", None, repair_template="S={schema}")
    assert ask.prompts == ["PROMPT", "S={}"]
    assert result.ok is True
    assert result.value == [1]


@pytest.mark.parametrize(
    "reply",
    ['[{"n": "x"}, {"n": 2}, {"n"', '[{"n": "x'],
)
def test_generate_json_no_reask_for_truncated_when_disabled(reply: str) -> None:
    schema = {
        "type": "array",
        "items": {"type": "object", "properties": {"n": {"type": "integer"}}},
    }
    ask = _Ask([reply, '[{"n": 1}]'])
    result = generate_json(ask, "PROMPT", schema, repair_template=TEMPLATE, reask_truncated=False)
    assert ask.prompts == ["PROMPT"]
    assert result.ok is False
    assert result.truncated is True
    assert result.reasks == 0


def test_generate_json_reasks_truncated_by_default() -> None:
    schema = {
        "type": "array",
        "items": {"type": "object", "properties": {"n": {"type": "integer"}}},
    }
    ask = _Ask(['[{"n": "x"}, {"n": 2}, {"n"', '[{"n": 1}]'])
    result = generate_json(ask, "PROMPT", schema, repair_template=TEMPLATE)
    assert len(ask.prompts) == 2
    assert result.ok is True
    assert result.value == [{"n": 1}]
    assert result.truncated is False
    assert result.status is JsonStatus.REASKED


def test_generate_json_reads_text_from_tuple_reply() -> None:
    ask = _Ask([('{"a": 1}', {"input_tokens": 3, "output_tokens": 4})])
    result = generate_json(ask, "PROMPT", OBJ_SCHEMA, repair_template=TEMPLATE)
    assert result.ok is True
    assert result.value == {"a": 1}
    assert result.raw == '{"a": 1}'
    assert ask.prompts == ["PROMPT"]


def test_generate_json_none_reply_reads_as_empty() -> None:
    ask = _Ask([None, None])
    result = generate_json(ask, "PROMPT", OBJ_SCHEMA, repair_template="R=[{reply}]")
    assert ask.prompts == ["PROMPT", "R=[]"]
    assert result.ok is False
    assert result.raw == ""
    assert result.reasks == 1


# --- agenerate_json ----------------------------------------------------------

_SCENARIOS: list[tuple[list[Any], dict[str, Any]]] = [
    (['{"a": 1}'], {}),
    (['{"b": 2}', '{"a": "3"}'], {}),
    (["bad", '{"b": 1}', "x"], {"attempts": 2}),
    (['{"b": 1}', None], {}),
    ([("no", {"u": 1}), ('```json\n{"a": 1}\n```', {"u": 2})], {}),
    (["bad", "worse"], {"attempts": 0}),
]


@pytest.mark.parametrize(("replies", "kwargs"), _SCENARIOS)
@pytest.mark.parametrize("async_ask", [True, False])
def test_agenerate_json_matches_generate_json(
    replies: list[Any], kwargs: dict[str, Any], async_ask: bool
) -> None:
    sync_ask = _Ask(replies)
    expected = generate_json(sync_ask, "PROMPT", OBJ_SCHEMA, repair_template=TEMPLATE, **kwargs)
    ask = _AsyncAsk(replies) if async_ask else _Ask(replies)
    result = asyncio.run(
        agenerate_json(ask, "PROMPT", OBJ_SCHEMA, repair_template=TEMPLATE, **kwargs)
    )
    assert isinstance(result, JsonResult)
    assert result == expected
    assert ask.prompts == sync_ask.prompts
