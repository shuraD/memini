"""Acceptance tests for LP-15: the Ollama embedder with task prefixes."""

from __future__ import annotations

import json
from typing import Any

import pytest

from memini.infra.llm_providers.config import LLMConfig, parse_config
from memini.infra.llm_providers.embedder import OllamaEmbedder
from memini.infra.llm_providers.kind_openai import HttpRequest
from memini.infra.llm_providers.transport import HttpResponse
from memini.infra.llm_providers.types import Embedder, LLMError, Reason

OLLAMA_URL = "http://127.0.0.1:11434"
OTHER_URL = "http://127.0.0.1:9999/v1"
EMBED_MODEL = "test-embed-model"


class FakeTransport:
    """Records every request and answers from a fixed status/body, never the network."""

    def __init__(self, status: int = 200, body: bytes | None = None) -> None:
        self.status = status
        self.body = body
        self.requests: list[HttpRequest] = []

    def send(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        if self.body is not None:
            return HttpResponse(status=self.status, headers={}, body=self.body)
        payload = json.loads(request.body.decode("utf-8"))
        vectors = [[float(i), float(i) + 0.5] for i, _ in enumerate(payload["input"])]
        return HttpResponse(
            status=self.status,
            headers={"Content-Type": "application/json"},
            body=json.dumps({"embeddings": vectors}).encode("utf-8"),
        )


def _config(
    prefixes: dict[str, str] | None = None,
    *,
    second_ollama: bool = False,
) -> LLMConfig:
    providers: list[dict[str, Any]] = [
        {
            "name": "remote",
            "kind": "openai_compatible",
            "key_env": "TEST_REMOTE_KEY",
            "base_url": OTHER_URL,
            "models": {
                "answer": {"name": "remote-answer"},
                "embedding": {"name": "remote-embed"},
            },
        },
        {
            "name": "local",
            "kind": "ollama",
            "base_url": OLLAMA_URL,
            "limits": {"timeout_s": 7},
            "models": {
                "answer": {"name": "local-answer"},
                "embedding": {"name": EMBED_MODEL},
            },
        },
    ]
    if second_ollama:
        providers.append(
            {
                "name": "local2",
                "kind": "ollama",
                "base_url": "http://localhost:11500",
                "limits": {"timeout_s": 3},
                "models": {"embedding": {"name": "second-embed-model"}},
            }
        )
    raw: dict[str, Any] = {"providers": providers}
    if prefixes is not None:
        raw["embedding_prefixes"] = prefixes
    return parse_config(raw)


def _env() -> dict[str, str]:
    return {"TEST_REMOTE_KEY": "test-key"}


def _body(request: HttpRequest) -> dict[str, Any]:
    data = json.loads(request.body.decode("utf-8"))
    assert isinstance(data, dict)
    return data


def _embedder(transport: FakeTransport, config: LLMConfig | None = None) -> OllamaEmbedder:
    return OllamaEmbedder(config or _config(), _env(), transport)


def test_embedder_satisfies_the_embedder_protocol() -> None:
    assert isinstance(_embedder(FakeTransport()), Embedder)


def test_query_kind_prefixes_every_input_with_search_query() -> None:
    transport = FakeTransport()
    _embedder(transport).embed(["alpha", "beta"], kind="query")
    assert len(transport.requests) == 1
    inputs = _body(transport.requests[0])["input"]
    assert inputs == ["search_query: alpha", "search_query: beta"]


def test_document_kind_prefixes_every_input_with_search_document() -> None:
    transport = FakeTransport()
    _embedder(transport).embed(["alpha", "beta"], kind="document")
    inputs = _body(transport.requests[0])["input"]
    assert inputs == ["search_document: alpha", "search_document: beta"]


def test_configured_prefixes_replace_the_defaults() -> None:
    transport = FakeTransport()
    config = _config({"query": "Q> ", "document": "D> "})
    embedder = _embedder(transport, config)
    embedder.embed(["one"], kind="query")
    embedder.embed(["two"], kind="document")
    assert _body(transport.requests[0])["input"] == ["Q> one"]
    assert _body(transport.requests[1])["input"] == ["D> two"]


def test_empty_configured_prefix_sends_text_unchanged() -> None:
    transport = FakeTransport()
    config = _config({"query": "", "document": ""})
    _embedder(transport, config).embed(["plain"], kind="query")
    assert _body(transport.requests[0])["input"] == ["plain"]


def test_request_is_post_to_api_embed_with_configured_model() -> None:
    transport = FakeTransport()
    _embedder(transport).embed(["x"], kind="document")
    request = transport.requests[0]
    assert request.method == "POST"
    assert request.url == OLLAMA_URL + "/api/embed"
    body = _body(request)
    assert body["model"] == EMBED_MODEL
    assert isinstance(body["input"], list)


def test_three_texts_give_three_vectors_in_order() -> None:
    transport = FakeTransport()
    vectors = _embedder(transport).embed(["a", "b", "c"], kind="document")
    assert vectors == [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]]
    assert len(transport.requests) == 1
    assert _body(transport.requests[0])["input"] == [
        "search_document: a",
        "search_document: b",
        "search_document: c",
    ]


def test_reply_vectors_are_returned_as_float_lists() -> None:
    reply = json.dumps({"embeddings": [[1, 2, 3], [4.5, 5.5, 6.5]]}).encode("utf-8")
    transport = FakeTransport(body=reply)
    vectors = _embedder(transport).embed(["a", "b"], kind="query")
    assert vectors == [[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]]
    assert all(isinstance(v, list) for v in vectors)
    assert all(isinstance(x, float) for v in vectors for x in v)


def test_empty_list_returns_empty_and_sends_no_request() -> None:
    transport = FakeTransport()
    assert _embedder(transport).embed([], kind="query") == []
    assert _embedder(transport).embed([], kind="document") == []
    assert transport.requests == []


def test_failure_status_raises_llm_error_with_reason_and_no_fallback() -> None:
    transport = FakeTransport(status=500, body=b'{"error":"model crashed"}')
    with pytest.raises(LLMError) as caught:
        _embedder(transport).embed(["a"], kind="query")
    error = caught.value
    assert error.attempts
    assert all(a.provider == "local" for a in error.attempts)
    assert error.attempts[-1].reason is Reason.SERVER_ERROR
    assert transport.requests
    assert all(r.url.startswith(OLLAMA_URL) for r in transport.requests)


def test_client_error_status_raises_llm_error_naming_request_rejected() -> None:
    transport = FakeTransport(status=404, body=b'{"error":"model not found"}')
    with pytest.raises(LLMError) as caught:
        _embedder(transport).embed(["a"], kind="document")
    attempts = caught.value.attempts
    assert attempts
    assert all(a.provider == "local" for a in attempts)
    assert attempts[-1].reason is Reason.REQUEST_REJECTED
    assert all(r.url.startswith(OLLAMA_URL) for r in transport.requests)


def test_request_carries_user_agent_header() -> None:
    transport = FakeTransport()
    _embedder(transport).embed(["a"], kind="query")
    headers = {k.lower(): v for k, v in transport.requests[0].headers.items()}
    assert headers.get("user-agent")


def test_request_uses_limits_timeout() -> None:
    transport = FakeTransport()
    _embedder(transport).embed(["a"], kind="query")
    assert transport.requests[0].timeout_s == pytest.approx(7.0)


def test_explicit_provider_selects_that_ollama_provider() -> None:
    transport = FakeTransport()
    config = _config(second_ollama=True)
    embedder = OllamaEmbedder(config, _env(), transport, provider="local2")
    embedder.embed(["a"], kind="query")
    request = transport.requests[0]
    assert request.url == "http://localhost:11500/api/embed"
    assert _body(request)["model"] == "second-embed-model"
    assert request.timeout_s == pytest.approx(3.0)


def test_default_provider_is_the_ollama_one_with_an_embedding_model() -> None:
    transport = FakeTransport()
    _embedder(transport).embed(["a"], kind="query")
    assert transport.requests[0].url.startswith(OLLAMA_URL)
    assert _body(transport.requests[0])["model"] != "remote-embed"
