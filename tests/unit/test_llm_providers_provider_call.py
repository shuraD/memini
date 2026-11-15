"""Acceptance tests for LP-07: one generation against one provider for one role."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from memini.infra.llm_providers.config import LLMConfig, ProviderSpec, parse_config
from memini.infra.llm_providers.kind_openai import HttpRequest
from memini.infra.llm_providers.outcome import Action
from memini.infra.llm_providers.provider_call import ProviderFailure, call_provider
from memini.infra.llm_providers.transport import HttpResponse, TransportTimeout
from memini.infra.llm_providers.types import Generation, Reason

KEY_ENV_A = "MEMINI_LP07_FAKE_KEY_A"
KEY_ENV_B = "MEMINI_LP07_FAKE_KEY_B"
FAKE_KEY = "test-key"

OPENAI_OK = json.dumps(
    {
        "choices": [{"message": {"content": "hello from openai"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }
).encode()
GEMINI_OK = json.dumps(
    {
        "candidates": [{"content": {"parts": [{"text": "hello from gemini"}]}}],
        "usageMetadata": {"promptTokenCount": 13, "candidatesTokenCount": 5},
    }
).encode()
OLLAMA_OK = json.dumps(
    {
        "message": {"content": "hello from ollama"},
        "prompt_eval_count": 17,
        "eval_count": 3,
    }
).encode()


class FakeTransport:
    """Replays scripted answers (responses or exceptions) and records every request."""

    def __init__(self, script: list[HttpResponse | BaseException]) -> None:
        self.script = list(script)
        self.requests: list[HttpRequest] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        if not self.script:
            raise AssertionError("transport called more often than scripted")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def resp(status: int, body: bytes = b"", headers: Mapping[str, str] | None = None) -> HttpResponse:
    return HttpResponse(status=status, headers=dict(headers or {}), body=body)


def make_clock(start: float, later: float) -> Callable[[], float]:
    """Return ``start`` on the first call and ``later`` on every call after it."""
    calls = {"n": 0}

    def clock() -> float:
        calls["n"] += 1
        return start if calls["n"] == 1 else later

    return clock


def build_config() -> LLMConfig:
    raw: dict[str, Any] = {
        "providers": [
            {
                "name": "alpha",
                "kind": "openai_compatible",
                "key_env": KEY_ENV_A,
                "base_url": "https://alpha.invalid/v1",
                "rest_default_s": 42,
                "models": {
                    "answer": {"name": "alpha-answer-model"},
                    "judge": {"name": "alpha-judge-model"},
                },
            },
            {
                "name": "beta",
                "kind": "gemini",
                "key_env": KEY_ENV_B,
                "base_url": "https://beta.invalid/v1beta",
                "models": {"answer": {"name": "beta-answer-model"}},
            },
            {
                "name": "gamma",
                "kind": "ollama",
                "base_url": "http://localhost:11434",
                "models": {"answer": {"name": "gamma-answer-model"}},
            },
            # Names deliberately contradict kinds: dispatch must follow ``kind`` only.
            {
                "name": "gemini",
                "kind": "ollama",
                "base_url": "http://localhost:11500",
                "models": {"answer": {"name": "decoy-ollama-model"}},
            },
            {
                "name": "ollama",
                "kind": "openai_compatible",
                "key_env": KEY_ENV_A,
                "base_url": "https://decoy-openai.invalid/v1",
                "models": {"answer": {"name": "decoy-openai-model"}},
            },
            {
                "name": "openai",
                "kind": "gemini",
                "key_env": KEY_ENV_B,
                "base_url": "https://decoy-gemini.invalid/v1beta",
                "models": {"answer": {"name": "decoy-gemini-model"}},
            },
        ]
    }
    return parse_config(raw)


ENV = {KEY_ENV_A: FAKE_KEY, KEY_ENV_B: FAKE_KEY}


def call(
    spec: ProviderSpec,
    config: LLMConfig,
    transport: FakeTransport,
    *,
    role: str = "answer",
    env: Mapping[str, str] = ENV,
    clock: Callable[[], float] | None = None,
) -> Generation:
    return call_provider(
        spec,
        config,
        role,
        "what is the answer?",
        system=None,
        json_mode=False,
        temperature=0.0,
        seed=None,
        env=env,
        transport=transport,
        clock=clock or make_clock(0.0, 0.0),
    )


def body_of(request: HttpRequest) -> dict[str, Any]:
    data = json.loads(request.body.decode("utf-8"))
    assert isinstance(data, dict)
    return data


# --- success -----------------------------------------------------------------------


def test_200_returns_generation_with_provider_model_role_tokens_and_seconds() -> None:
    config = build_config()
    transport = FakeTransport([resp(200, OPENAI_OK)])
    gen = call(config.provider("alpha"), config, transport, clock=make_clock(100.0, 102.5))
    assert isinstance(gen, Generation)
    assert gen.text == "hello from openai"
    assert gen.provider == "alpha"
    assert gen.model == "alpha-answer-model"
    assert gen.role == "answer"
    assert gen.prompt_tokens == 11
    assert gen.completion_tokens == 7
    assert gen.seconds == pytest.approx(2.5)
    assert gen.cached is False
    assert len(transport.requests) == 1


def test_key_is_read_from_the_given_env_mapping() -> None:
    config = build_config()
    transport = FakeTransport([resp(200, OPENAI_OK)])
    call(config.provider("alpha"), config, transport, env={KEY_ENV_A: FAKE_KEY})
    headers = transport.requests[0].headers
    assert headers.get("Authorization") == f"Bearer {FAKE_KEY}"


def test_role_model_is_the_one_sent_for_the_requested_role() -> None:
    config = build_config()
    transport = FakeTransport([resp(200, OPENAI_OK)])
    gen = call(config.provider("alpha"), config, transport, role="judge")
    assert body_of(transport.requests[0])["model"] == "alpha-judge-model"
    assert gen.model == "alpha-judge-model"
    assert gen.role == "judge"


def test_each_provider_sends_its_own_model_for_the_role() -> None:
    config = build_config()

    t_alpha = FakeTransport([resp(200, OPENAI_OK)])
    g_alpha = call(config.provider("alpha"), config, t_alpha)
    assert body_of(t_alpha.requests[0])["model"] == "alpha-answer-model"
    assert g_alpha.model == "alpha-answer-model"

    t_beta = FakeTransport([resp(200, GEMINI_OK)])
    g_beta = call(config.provider("beta"), config, t_beta)
    assert "beta-answer-model" in t_beta.requests[0].url
    assert "alpha-answer-model" not in t_beta.requests[0].url
    assert g_beta.model == "beta-answer-model"
    assert g_beta.provider == "beta"

    t_gamma = FakeTransport([resp(200, OLLAMA_OK)])
    g_gamma = call(config.provider("gamma"), config, t_gamma)
    assert body_of(t_gamma.requests[0])["model"] == "gamma-answer-model"
    assert g_gamma.model == "gamma-answer-model"
    assert g_gamma.provider == "gamma"


# --- no key / no model ---------------------------------------------------------------


def test_missing_key_raises_no_key_with_zero_transport_calls() -> None:
    config = build_config()
    transport = FakeTransport([resp(200, OPENAI_OK)])
    with pytest.raises(ProviderFailure) as info:
        call(config.provider("alpha"), config, transport, env={})
    assert info.value.attempt.reason is Reason.NO_KEY
    assert info.value.attempt.provider == "alpha"
    assert transport.requests == []


def test_key_present_only_for_another_provider_still_raises_no_key() -> None:
    config = build_config()
    transport = FakeTransport([resp(200, GEMINI_OK)])
    with pytest.raises(ProviderFailure) as info:
        call(config.provider("beta"), config, transport, env={KEY_ENV_A: FAKE_KEY})
    assert info.value.attempt.reason is Reason.NO_KEY
    assert info.value.attempt.provider == "beta"
    assert transport.requests == []


def test_no_model_for_role_raises_no_model_with_zero_transport_calls() -> None:
    config = build_config()
    transport = FakeTransport([resp(200, GEMINI_OK)])
    with pytest.raises(ProviderFailure) as info:
        call(config.provider("beta"), config, transport, role="judge")
    assert info.value.attempt.reason is Reason.NO_MODEL
    assert info.value.attempt.provider == "beta"
    assert transport.requests == []


def test_provider_without_key_env_needs_no_key() -> None:
    config = build_config()
    transport = FakeTransport([resp(200, OLLAMA_OK)])
    gen = call(config.provider("gamma"), config, transport, env={})
    assert gen.text == "hello from ollama"
    assert len(transport.requests) == 1


# --- retry on server error or timeout ---------------------------------------------------


def test_503_then_200_makes_two_requests_and_returns_generation() -> None:
    config = build_config()
    transport = FakeTransport([resp(503, b"unavailable"), resp(200, OPENAI_OK)])
    gen = call(config.provider("alpha"), config, transport)
    assert len(transport.requests) == 2
    assert gen.text == "hello from openai"
    assert gen.provider == "alpha"
    assert gen.cached is False


def test_timeout_then_200_makes_two_requests_and_returns_generation() -> None:
    config = build_config()
    transport = FakeTransport([TransportTimeout("timed out"), resp(200, OPENAI_OK)])
    gen = call(config.provider("alpha"), config, transport)
    assert len(transport.requests) == 2
    assert gen.text == "hello from openai"


def test_503_twice_makes_two_requests_then_fails_retry_then_next() -> None:
    config = build_config()
    transport = FakeTransport([resp(503, b"unavailable"), resp(503, b"unavailable")])
    with pytest.raises(ProviderFailure) as info:
        call(config.provider("alpha"), config, transport)
    assert len(transport.requests) == 2
    outcome = info.value.outcome
    assert outcome is not None
    assert outcome.action is Action.RETRY_THEN_NEXT
    assert info.value.attempt.reason is Reason.SERVER_ERROR
    assert info.value.attempt.provider == "alpha"


def test_timeout_twice_makes_two_requests_then_fails_retry_then_next() -> None:
    config = build_config()
    transport = FakeTransport([TransportTimeout("timed out"), TransportTimeout("timed out")])
    with pytest.raises(ProviderFailure) as info:
        call(config.provider("alpha"), config, transport)
    assert len(transport.requests) == 2
    outcome = info.value.outcome
    assert outcome is not None
    assert outcome.action is Action.RETRY_THEN_NEXT
    assert info.value.attempt.reason is Reason.TIMEOUT


# --- no retry on other failures ----------------------------------------------------------


def test_429_per_minute_is_not_retried_and_carries_rest_with_the_wait() -> None:
    config = build_config()
    transport = FakeTransport(
        [resp(429, b"rate limited", {"Retry-After": "7"}), resp(200, OPENAI_OK)]
    )
    with pytest.raises(ProviderFailure) as info:
        call(config.provider("alpha"), config, transport)
    assert len(transport.requests) == 1
    outcome = info.value.outcome
    assert outcome is not None
    assert outcome.action is Action.REST
    assert outcome.wait_s == pytest.approx(7.0)
    assert info.value.attempt.reason is Reason.RATE_PER_MINUTE


def test_413_makes_one_request_and_carries_next_request_only() -> None:
    config = build_config()
    transport = FakeTransport([resp(413, b"payload too large"), resp(200, OPENAI_OK)])
    with pytest.raises(ProviderFailure) as info:
        call(config.provider("alpha"), config, transport)
    assert len(transport.requests) == 1
    outcome = info.value.outcome
    assert outcome is not None
    assert outcome.action is Action.NEXT_REQUEST_ONLY
    assert info.value.attempt.reason is Reason.REQUEST_REJECTED


def test_401_is_not_retried_and_carries_exhausted() -> None:
    config = build_config()
    transport = FakeTransport([resp(401, b"unauthorized"), resp(200, OPENAI_OK)])
    with pytest.raises(ProviderFailure) as info:
        call(config.provider("alpha"), config, transport)
    assert len(transport.requests) == 1
    outcome = info.value.outcome
    assert outcome is not None
    assert outcome.action is Action.EXHAUSTED


# --- builder chosen by kind, never by name -----------------------------------------------


def test_openai_compatible_kind_uses_chat_completions_builder() -> None:
    config = build_config()
    transport = FakeTransport([resp(200, OPENAI_OK)])
    gen = call(config.provider("ollama"), config, transport)
    request = transport.requests[0]
    assert request.url == "https://decoy-openai.invalid/v1/chat/completions"
    assert request.headers.get("Authorization") == f"Bearer {FAKE_KEY}"
    assert body_of(request)["model"] == "decoy-openai-model"
    assert gen.text == "hello from openai"
    assert (gen.prompt_tokens, gen.completion_tokens) == (11, 7)
    assert gen.provider == "ollama"


def test_gemini_kind_uses_generate_content_builder() -> None:
    config = build_config()
    transport = FakeTransport([resp(200, GEMINI_OK)])
    gen = call(config.provider("openai"), config, transport)
    request = transport.requests[0]
    assert request.url == (
        "https://decoy-gemini.invalid/v1beta/models/decoy-gemini-model:generateContent"
    )
    assert request.headers.get("x-goog-api-key") == FAKE_KEY
    assert "Authorization" not in request.headers
    assert gen.text == "hello from gemini"
    assert (gen.prompt_tokens, gen.completion_tokens) == (13, 5)
    assert gen.provider == "openai"


def test_ollama_kind_uses_api_chat_builder() -> None:
    config = build_config()
    transport = FakeTransport([resp(200, OLLAMA_OK)])
    gen = call(config.provider("gemini"), config, transport, env={})
    request = transport.requests[0]
    assert request.url == "http://localhost:11500/api/chat"
    assert body_of(request)["model"] == "decoy-ollama-model"
    assert body_of(request)["stream"] is False
    assert gen.text == "hello from ollama"
    assert (gen.prompt_tokens, gen.completion_tokens) == (17, 3)
    assert gen.provider == "gemini"
