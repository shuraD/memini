"""Acceptance tests for LP-05: gemini and ollama request building and reply parsing."""

from __future__ import annotations

import json
from types import ModuleType
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from memini.infra.llm_providers import kind_gemini, kind_ollama

GEMINI_BASE = "http://127.0.0.1:9998/v1beta"
OLLAMA_BASE = "http://127.0.0.1:11434"
FAKE_KEY = "test-key"

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def _build(module: ModuleType, **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "base_url": GEMINI_BASE if module is kind_gemini else OLLAMA_BASE,
        "model": "test-model",
        "prompt": "hello there",
        "system": None,
        "json_mode": False,
        "temperature": 0.0,
        "seed": None,
        "limits": {},
        "api_key": FAKE_KEY if module is kind_gemini else None,
    }
    kwargs.update(overrides)
    return module.build_request(**kwargs)


def _body(req: Any) -> dict[str, Any]:
    data = json.loads(req.body.decode("utf-8"))
    assert isinstance(data, dict)
    return data


def _headers_lower(req: Any) -> dict[str, str]:
    return {k.lower(): v for k, v in dict(req.headers).items()}


def _contains(haystack: Any, needle: Any) -> bool:
    if haystack == needle:
        return True
    if isinstance(haystack, dict):
        return any(_contains(v, needle) for v in haystack.values())
    if isinstance(haystack, list):
        return any(_contains(v, needle) for v in haystack)
    return False


# ---------------------------------------------------------------- Gemini request


def test_gemini_posts_to_generate_content_url() -> None:
    req = _build(kind_gemini)
    assert req.method == "POST"
    parts = urlsplit(req.url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == (
        GEMINI_BASE + "/models/test-model:generateContent"
    )


def test_gemini_trailing_slash_base_url_not_doubled() -> None:
    req = _build(kind_gemini, base_url=GEMINI_BASE + "/")
    assert "//models" not in req.url
    assert urlsplit(req.url).path.endswith("/v1beta/models/test-model:generateContent")


def test_gemini_system_goes_in_system_instruction_not_contents() -> None:
    req = _build(kind_gemini, system="be terse", prompt="hello there")
    body = _body(req)
    assert "systemInstruction" in body
    assert _contains(body["systemInstruction"], "be terse")
    assert _contains(body["contents"], "hello there")
    assert not _contains(body["contents"], "be terse")


def test_gemini_no_system_omits_system_instruction() -> None:
    body = _body(_build(kind_gemini, system=None))
    assert "systemInstruction" not in body
    assert _contains(body["contents"], "hello there")


def test_gemini_json_mode_true_requests_json() -> None:
    body = _body(_build(kind_gemini, json_mode=True))
    assert body["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_json_mode_false_does_not_request_json() -> None:
    body = _body(_build(kind_gemini, json_mode=False))
    config = body.get("generationConfig", {})
    assert config.get("responseMimeType") != "application/json"
    assert not _contains(body, "application/json")


def test_gemini_json_mode_dict_puts_schema_in_body() -> None:
    body = _body(_build(kind_gemini, json_mode=SCHEMA))
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert _contains(config, SCHEMA)


def test_gemini_max_tokens_reaches_max_output_tokens() -> None:
    body = _body(_build(kind_gemini, limits={"max_tokens": 321}))
    assert body["generationConfig"]["maxOutputTokens"] == 321


def test_gemini_no_max_tokens_omits_output_limit() -> None:
    body = _body(_build(kind_gemini, limits={}))
    assert "maxOutputTokens" not in body.get("generationConfig", {})


def test_gemini_temperature_in_generation_config() -> None:
    body = _body(_build(kind_gemini, temperature=0.25))
    assert body["generationConfig"]["temperature"] == 0.25


def test_gemini_has_user_agent_header() -> None:
    headers = _headers_lower(_build(kind_gemini))
    assert headers.get("user-agent")


def test_gemini_key_sent_in_header_or_query() -> None:
    req = _build(kind_gemini, api_key=FAKE_KEY)
    in_header = FAKE_KEY in _headers_lower(req).values()
    query = parse_qs(urlsplit(req.url).query)
    in_query = FAKE_KEY in query.get("key", [])
    assert in_header or in_query
    assert FAKE_KEY.encode() not in req.body


def test_gemini_timeout_from_limits() -> None:
    req = _build(kind_gemini, limits={"timeout_s": 12})
    assert req.timeout_s == 12.0


# ---------------------------------------------------------------- Ollama request


def test_ollama_posts_to_api_chat_on_base_url() -> None:
    req = _build(kind_ollama, base_url="http://localhost:5555")
    assert req.method == "POST"
    assert req.url == "http://localhost:5555/api/chat"


def test_ollama_trailing_slash_base_url_not_doubled() -> None:
    req = _build(kind_ollama, base_url=OLLAMA_BASE + "/")
    assert req.url == OLLAMA_BASE + "/api/chat"


def test_ollama_stream_false_and_model() -> None:
    body = _body(_build(kind_ollama))
    assert body["stream"] is False
    assert body["model"] == "test-model"


def test_ollama_messages_system_then_user() -> None:
    body = _body(_build(kind_ollama, system="be terse", prompt="hello there"))
    assert body["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello there"},
    ]


def test_ollama_messages_user_only_without_system() -> None:
    body = _body(_build(kind_ollama, system=None))
    assert body["messages"] == [{"role": "user", "content": "hello there"}]


def test_ollama_json_mode_true_sets_format_json() -> None:
    body = _body(_build(kind_ollama, json_mode=True))
    assert body["format"] == "json"


def test_ollama_json_mode_false_has_no_format() -> None:
    body = _body(_build(kind_ollama, json_mode=False))
    assert "format" not in body


def test_ollama_json_mode_dict_is_format_schema() -> None:
    body = _body(_build(kind_ollama, json_mode=SCHEMA))
    assert body["format"] == SCHEMA


def test_ollama_options_carry_temperature_seed_and_max_tokens() -> None:
    body = _body(_build(kind_ollama, temperature=0.5, seed=7, limits={"max_tokens": 99}))
    options = body["options"]
    assert options["temperature"] == 0.5
    assert options["seed"] == 7
    assert options["num_predict"] == 99
    assert "temperature" not in body
    assert "seed" not in body
    assert "max_tokens" not in body


def test_ollama_no_seed_no_max_tokens_omitted_from_options() -> None:
    body = _body(_build(kind_ollama, seed=None, limits={}))
    options = body.get("options", {})
    assert "seed" not in options
    assert "num_predict" not in options


def test_ollama_has_user_agent_and_no_key_header() -> None:
    req = _build(kind_ollama, api_key=None)
    headers = _headers_lower(req)
    assert headers.get("user-agent")
    assert "authorization" not in headers
    assert "x-goog-api-key" not in headers
    assert "key=" not in req.url


def test_ollama_timeout_from_limits() -> None:
    req = _build(kind_ollama, limits={"timeout_s": 30})
    assert req.timeout_s == 30.0


# ---------------------------------------------------------------- parsing


def test_gemini_parse_response_text_and_counts() -> None:
    reply = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "the answer"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 11,
            "candidatesTokenCount": 5,
            "totalTokenCount": 16,
        },
    }
    parsed = kind_gemini.parse_response(200, json.dumps(reply).encode("utf-8"))
    assert parsed.text == "the answer"
    assert parsed.prompt_tokens == 11
    assert parsed.completion_tokens == 5


def test_gemini_parse_response_missing_usage_gives_zero_counts() -> None:
    reply = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
    parsed = kind_gemini.parse_response(200, json.dumps(reply).encode("utf-8"))
    assert parsed.text == "ok"
    assert parsed.prompt_tokens == 0
    assert parsed.completion_tokens == 0


def test_gemini_parse_response_missing_text_raises_value_error() -> None:
    with pytest.raises(ValueError):
        kind_gemini.parse_response(200, json.dumps({"candidates": []}).encode("utf-8"))


def test_ollama_parse_response_text_and_counts() -> None:
    reply = {
        "model": "test-model",
        "message": {"role": "assistant", "content": "local answer"},
        "done": True,
        "prompt_eval_count": 23,
        "eval_count": 9,
    }
    parsed = kind_ollama.parse_response(200, json.dumps(reply).encode("utf-8"))
    assert parsed.text == "local answer"
    assert parsed.prompt_tokens == 23
    assert parsed.completion_tokens == 9


def test_ollama_parse_response_missing_counts_gives_zero() -> None:
    reply = {"message": {"role": "assistant", "content": "x"}, "done": True}
    parsed = kind_ollama.parse_response(200, json.dumps(reply).encode("utf-8"))
    assert parsed.text == "x"
    assert parsed.prompt_tokens == 0
    assert parsed.completion_tokens == 0


def test_ollama_parse_response_missing_content_raises_value_error() -> None:
    with pytest.raises(ValueError):
        kind_ollama.parse_response(200, json.dumps({"done": True}).encode("utf-8"))


@pytest.mark.parametrize("module", [kind_gemini, kind_ollama])
def test_parse_response_non_json_raises_value_error(module: ModuleType) -> None:
    with pytest.raises(ValueError):
        module.parse_response(500, b"not json")
