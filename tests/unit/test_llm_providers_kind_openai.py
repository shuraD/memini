"""Acceptance tests for LP-04: openai_compatible request building and reply parsing."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any

import pytest

from memini.infra.llm_providers.kind_openai import (
    USER_AGENT,
    HttpRequest,
    ParsedReply,
    build_request,
    parse_response,
)

BASE_URL = "http://127.0.0.1:9999/v1"


def _build(**overrides: Any) -> HttpRequest:
    kwargs: dict[str, Any] = {
        "base_url": BASE_URL,
        "model": "test-model",
        "prompt": "hello there",
        "system": None,
        "json_mode": False,
        "temperature": 0.0,
        "seed": None,
        "limits": {},
        "api_key": None,
    }
    kwargs.update(overrides)
    return build_request(**kwargs)


def _body(req: HttpRequest) -> dict[str, Any]:
    data = json.loads(req.body.decode("utf-8"))
    assert isinstance(data, dict)
    return data


def _header(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None


def test_request_is_post_to_chat_completions_with_model() -> None:
    req = _build()
    assert isinstance(req, HttpRequest)
    assert req.method.upper() == "POST"
    assert req.url.startswith(BASE_URL)
    assert req.url.endswith("/chat/completions")
    assert isinstance(req.body, bytes)
    assert _body(req)["model"] == "test-model"


def test_system_prompt_gives_system_then_user_messages() -> None:
    body = _body(_build(system="be terse", prompt="question?"))
    messages = body["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == "be terse"
    assert messages[1]["content"] == "question?"


def test_without_system_prompt_only_user_message() -> None:
    body = _body(_build(system=None, prompt="question?"))
    messages = body["messages"]
    assert [m["role"] for m in messages] == ["user"]
    assert messages[0]["content"] == "question?"


def test_json_mode_true_asks_for_json_object() -> None:
    body = _body(_build(json_mode=True))
    assert body["response_format"]["type"] == "json_object"


def test_json_mode_false_sends_no_response_format() -> None:
    body = _body(_build(json_mode=False))
    assert "response_format" not in body


def test_max_tokens_limit_reaches_body() -> None:
    body = _body(_build(limits={"max_tokens": 256}))
    assert body["max_tokens"] == 256


def test_no_max_tokens_limit_omits_field() -> None:
    body = _body(_build(limits={}))
    assert "max_tokens" not in body


def test_temperature_and_seed_reach_body() -> None:
    body = _body(_build(temperature=0.3, seed=42))
    assert body["temperature"] == 0.3
    assert body["seed"] == 42


def test_seed_none_is_omitted() -> None:
    body = _body(_build(temperature=0.7, seed=None))
    assert body["temperature"] == 0.7
    assert "seed" not in body


def test_user_agent_header_is_non_empty() -> None:
    assert isinstance(USER_AGENT, str)
    assert USER_AGENT.strip() != ""
    req = _build()
    ua = _header(req.headers, "User-Agent")
    assert ua is not None
    assert ua.strip() != ""
    assert ua == USER_AGENT


def test_authorization_bearer_when_key_given() -> None:
    req = _build(api_key="sk-test-123")
    assert _header(req.headers, "Authorization") == "Bearer sk-test-123"
    assert _header(req.headers, "User-Agent")


def test_no_authorization_without_key() -> None:
    req = _build(api_key=None)
    assert _header(req.headers, "Authorization") is None
    assert _header(req.headers, "User-Agent")


def test_timeout_limit_becomes_float_timeout() -> None:
    req = _build(limits={"timeout_s": 12})
    assert req.timeout_s == 12.0
    assert isinstance(req.timeout_s, float)


def test_api_key_not_in_body() -> None:
    req = _build(api_key="sk-secret-xyz")
    assert b"sk-secret-xyz" not in req.body


def test_parse_response_returns_text_and_token_counts() -> None:
    body = json.dumps(
        {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "the answer"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }
    ).encode("utf-8")
    reply = parse_response(200, body)
    assert isinstance(reply, ParsedReply)
    assert reply.text == "the answer"
    assert reply.prompt_tokens == 11
    assert reply.completion_tokens == 7


def test_parse_response_missing_usage_gives_zero_counts() -> None:
    body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode(
        "utf-8"
    )
    reply = parse_response(200, body)
    assert reply.text == "ok"
    assert reply.prompt_tokens == 0
    assert reply.completion_tokens == 0


def test_parse_response_missing_choices_raises_naming_field() -> None:
    body = json.dumps({"usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode("utf-8")
    with pytest.raises(Exception) as excinfo:
        parse_response(200, body)
    assert "choice" in str(excinfo.value)


def test_value_types_are_frozen_dataclasses() -> None:
    reply = ParsedReply(text="t", prompt_tokens=1, completion_tokens=2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        reply.text = "other"  # type: ignore[misc]
    req = _build()
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.url = "http://elsewhere"  # type: ignore[misc]
