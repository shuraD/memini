"""Acceptance tests for LT-06: ollama decode limits, residency and a per-call model."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from memini.infra.llm_providers import kind_ollama
from memini.infra.llm_providers.chain import ProviderChain
from memini.infra.llm_providers.config import LLMConfig, parse_config
from memini.infra.llm_providers.kind_openai import HttpRequest
from memini.infra.llm_providers.provider_call import ProviderFailure, call_provider
from memini.infra.llm_providers.store import Store
from memini.infra.llm_providers.transport import HttpResponse
from memini.infra.llm_providers.types import Generation, Reason

ROLE_MODEL = "local-answer-model"
OLLAMA_OK = json.dumps(
    {"message": {"content": "hello"}, "prompt_eval_count": 5, "eval_count": 2}
).encode()


class FakeTransport:
    """Answers every request with a fixed ollama reply and records the requests."""

    def __init__(self) -> None:
        self.requests: list[HttpRequest] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        return HttpResponse(status=200, headers={}, body=OLLAMA_OK)

    def payload(self, index: int = 0) -> dict[str, Any]:
        data = json.loads(self.requests[index].body.decode("utf-8"))
        assert isinstance(data, dict)
        return data


def clock() -> float:
    return 100.0


def payload_of(request: HttpRequest) -> dict[str, Any]:
    data = json.loads(request.body.decode("utf-8"))
    assert isinstance(data, dict)
    return data


def build(limits: Mapping[str, Any], seed: int | None = None) -> dict[str, Any]:
    request = kind_ollama.build_request(
        base_url="http://localhost:11434",
        model="m",
        prompt="hi",
        system=None,
        json_mode=False,
        temperature=0.0,
        seed=seed,
        limits=limits,
        api_key=None,
    )
    return payload_of(request)


# --- build_request ------------------------------------------------------------------------


def test_build_request_sends_decode_limits_in_options() -> None:
    payload = build({"num_ctx": 32768, "repeat_penalty": 1.1, "max_tokens": 8192})
    assert payload["options"] == {
        "temperature": 0.0,
        "num_ctx": 32768,
        "repeat_penalty": 1.1,
        "num_predict": 8192,
    }
    assert payload["stream"] is False
    assert isinstance(payload["options"]["num_ctx"], int)
    assert isinstance(payload["options"]["repeat_penalty"], float)
    assert isinstance(payload["options"]["num_predict"], int)


def test_build_request_with_seed_adds_only_the_seed() -> None:
    payload = build({"num_ctx": 32768, "repeat_penalty": 1.1, "max_tokens": 8192}, seed=7)
    assert payload["options"] == {
        "temperature": 0.0,
        "seed": 7,
        "num_ctx": 32768,
        "repeat_penalty": 1.1,
        "num_predict": 8192,
    }
    assert payload["stream"] is False


@pytest.mark.parametrize("value", [-1, "15m"])
def test_build_request_sends_keep_alive_as_given(value: int | str) -> None:
    payload = build({"keep_alive": value})
    assert payload["keep_alive"] == value
    assert type(payload["keep_alive"]) is type(value)
    assert "keep_alive" not in payload["options"]


def test_build_request_without_limits_sends_none_of_them() -> None:
    payload = build({})
    assert "keep_alive" not in payload
    assert "num_ctx" not in payload["options"]
    assert "repeat_penalty" not in payload["options"]
    assert "num_predict" not in payload["options"]
    assert payload["options"] == {"temperature": 0.0}
    assert payload["stream"] is False


# --- call_provider ------------------------------------------------------------------------


def config(
    *,
    idle: bool = True,
    with_model: bool = True,
) -> LLMConfig:
    provider_limits: dict[str, Any] = {
        "num_ctx": 16384,
        "repeat_penalty": 1.1,
        "keep_alive": -1,
        "max_tokens": 1024,
    }
    if idle:
        provider_limits["idle_keep_alive"] = "15m"
    models: dict[str, Any] = {}
    if with_model:
        # The role model's limits override the provider's.
        models["answer"] = {
            "name": ROLE_MODEL,
            "limits": {"num_ctx": 32768, "max_tokens": 8192},
        }
    else:
        models["judge"] = {"name": "local-judge-model"}
    raw: dict[str, Any] = {
        "providers": [
            {
                "name": "local",
                "kind": "ollama",
                "base_url": "http://localhost:11434",
                "limits": provider_limits,
                "models": models,
            }
        ]
    }
    return parse_config(raw)


def call(cfg: LLMConfig, transport: FakeTransport, **kwargs: Any) -> Generation:
    return call_provider(
        cfg.provider("local"),
        cfg,
        "answer",
        "what is the answer?",
        system=None,
        json_mode=False,
        temperature=0.0,
        seed=None,
        env={},
        transport=transport,
        clock=clock,
        **kwargs,
    )


EXPECTED_OPTIONS = {
    "temperature": 0.0,
    "num_ctx": 32768,
    "repeat_penalty": 1.1,
    "num_predict": 8192,
}


def test_call_without_model_sends_role_model_with_resident_keep_alive() -> None:
    transport = FakeTransport()
    gen = call(config(), transport)
    payload = transport.payload()
    assert len(transport.requests) == 1
    assert payload["model"] == ROLE_MODEL
    assert payload["keep_alive"] == -1
    assert payload["options"] == EXPECTED_OPTIONS
    assert gen.model == ROLE_MODEL


def test_call_with_role_own_model_name_is_like_no_model() -> None:
    transport = FakeTransport()
    gen = call(config(), transport, model=ROLE_MODEL)
    payload = transport.payload()
    assert payload["model"] == ROLE_MODEL
    assert payload["keep_alive"] == -1
    assert payload["options"] == EXPECTED_OPTIONS
    assert gen.model == ROLE_MODEL


def test_call_with_explicit_none_model_sends_role_model() -> None:
    transport = FakeTransport()
    gen = call(config(), transport, model=None)
    assert transport.payload()["model"] == ROLE_MODEL
    assert transport.payload()["keep_alive"] == -1
    assert gen.model == ROLE_MODEL


def test_call_with_other_model_sends_it_with_idle_keep_alive() -> None:
    transport = FakeTransport()
    gen = call(config(), transport, model="other")
    payload = transport.payload()
    assert len(transport.requests) == 1
    assert payload["model"] == "other"
    assert payload["keep_alive"] == "15m"
    assert payload["options"] == EXPECTED_OPTIONS
    assert gen.model == "other"
    assert gen.provider == "local"
    assert gen.role == "answer"
    assert gen.text == "hello"


def test_call_with_other_model_and_no_idle_keep_alive_sends_no_keep_alive() -> None:
    transport = FakeTransport()
    gen = call(config(idle=False), transport, model="other")
    payload = transport.payload()
    assert payload["model"] == "other"
    assert "keep_alive" not in payload
    assert payload["options"] == EXPECTED_OPTIONS
    assert gen.model == "other"


def test_role_model_without_idle_keep_alive_still_sends_keep_alive() -> None:
    transport = FakeTransport()
    call(config(idle=False), transport)
    assert transport.payload()["keep_alive"] == -1


def test_idle_keep_alive_is_never_sent_as_its_own_key() -> None:
    for model in (None, "other"):
        transport = FakeTransport()
        call(config(), transport, model=model)
        payload = transport.payload()
        assert "idle_keep_alive" not in payload
        assert "idle_keep_alive" not in payload["options"]


def test_per_call_model_without_role_model_raises_no_model_and_sends_nothing() -> None:
    transport = FakeTransport()
    with pytest.raises(ProviderFailure) as info:
        call(config(with_model=False), transport, model="other")
    assert info.value.attempt.reason is Reason.NO_MODEL
    assert info.value.outcome is None
    assert transport.requests == []


# --- ProviderChain.generate ---------------------------------------------------------------


def make_chain(transport: FakeTransport) -> ProviderChain:
    return ProviderChain("answer", config(), {}, clock, transport, Store(":memory:"))


def test_chain_generate_passes_the_per_call_model() -> None:
    transport = FakeTransport()
    gen = make_chain(transport).generate("question one", model="other")
    payload = transport.payload()
    assert payload["model"] == "other"
    assert payload["keep_alive"] == "15m"
    assert gen.model == "other"
    assert gen.provider == "local"


def test_chain_generate_without_model_sends_role_model() -> None:
    transport = FakeTransport()
    gen = make_chain(transport).generate("question two")
    payload = transport.payload()
    assert payload["model"] == ROLE_MODEL
    assert payload["keep_alive"] == -1
    assert gen.model == ROLE_MODEL
