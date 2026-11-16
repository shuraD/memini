"""Acceptance tests for LT-04: proxy-free transport, server errors kept, gemini thoughts dropped."""

from __future__ import annotations

import json
import threading
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType
from typing import Any

import pytest

from memini.infra.llm_providers import kind_gemini, kind_ollama, kind_openai
from memini.infra.llm_providers.kind_openai import USER_AGENT, HttpRequest, ResponseParseError
from memini.infra.llm_providers.transport import HttpResponse, UrllibTransport

# Port 9 (discard) on the loopback: nothing listens, so any proxied request fails.
_DEAD_PROXY = "http://127.0.0.1:9"
_PROXY_VARS = (
    "http_proxy",
    "HTTP_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
)
_NO_PROXY_VARS = ("no_proxy", "NO_PROXY")


class _Recorder:
    def __init__(self) -> None:
        self.status = 200
        self.body = b'{"ok": true}'
        self.seen: list[dict[str, Any]] = []


def _handler(rec: _Recorder) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _answer(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            data = self.rfile.read(length) if length else b""
            rec.seen.append(
                {
                    "method": self.command,
                    "path": self.path,
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": data,
                }
            )
            self.send_response(rec.status)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(rec.body)))
            self.end_headers()
            self.wfile.write(rec.body)

        def do_GET(self) -> None:
            self._answer()

        def do_POST(self) -> None:
            self._answer()

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


@pytest.fixture
def server() -> Iterator[tuple[str, _Recorder]]:
    rec = _Recorder()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(rec))
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    try:
        yield f"http://{host}:{port}", rec
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(5.0)


@pytest.fixture
def dead_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _PROXY_VARS:
        monkeypatch.setenv(name, _DEAD_PROXY)
    for name in _NO_PROXY_VARS:
        monkeypatch.delenv(name, raising=False)


def _post(url: str, body: bytes = b'{"q": 1}') -> HttpRequest:
    return HttpRequest(
        method="POST",
        url=url,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        body=body,
        timeout_s=5.0,
    )


# --- transport: no proxy ----------------------------------------------------------


@pytest.mark.usefixtures("dead_proxy_env")
def test_send_ignores_proxy_environment_variables(server: tuple[str, _Recorder]) -> None:
    base, rec = server
    rec.body = b'{"answer": "direct"}'
    response = UrllibTransport().send(_post(f"{base}/v1/chat"))
    assert isinstance(response, HttpResponse)
    assert response.status == 200
    assert response.body == b'{"answer": "direct"}'
    assert len(rec.seen) == 1
    assert rec.seen[0]["path"] == "/v1/chat"
    assert rec.seen[0]["body"] == b'{"q": 1}'


def test_send_ignores_operating_system_proxy_settings(
    server: tuple[str, _Recorder], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (*_PROXY_VARS, *_NO_PROXY_VARS):
        monkeypatch.delenv(name, raising=False)
    system_proxies = {"http": _DEAD_PROXY, "https": _DEAD_PROXY, "all": _DEAD_PROXY}
    monkeypatch.setattr(urllib.request, "getproxies", lambda: dict(system_proxies))
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host, proxies=None: False)
    base, rec = server
    response = UrllibTransport().send(_post(f"{base}/sys"))
    assert response.status == 200
    assert response.body == b'{"ok": true}'
    assert [s["path"] for s in rec.seen] == ["/sys"]


# --- transport: server errors kept -------------------------------------------------


@pytest.mark.usefixtures("dead_proxy_env")
def test_server_error_status_and_body_are_returned_unchanged(
    server: tuple[str, _Recorder],
) -> None:
    base, rec = server
    rec.status = 500
    rec.body = b"the input length exceeds the context length"
    response = UrllibTransport().send(_post(f"{base}/api/chat"))
    assert response.status == 500
    assert response.body == b"the input length exceeds the context length"
    assert len(rec.seen) == 1


# --- transport: client name ---------------------------------------------------------


def test_user_agent_value_is_the_fixed_client_name() -> None:
    assert USER_AGENT == "memini-llm-client/1"


@pytest.mark.usefixtures("dead_proxy_env")
@pytest.mark.parametrize("kind", [kind_ollama, kind_gemini, kind_openai])
def test_every_kind_request_reaches_server_with_user_agent(
    kind: ModuleType, server: tuple[str, _Recorder]
) -> None:
    base, rec = server
    request = kind.build_request(
        base_url=base,
        model="test-model",
        prompt="hello",
        system=None,
        json_mode=False,
        temperature=0.0,
        seed=None,
        limits={"timeout_s": 5},
        api_key="test-key",
    )
    response = UrllibTransport().send(request)
    assert response.status == 200
    assert len(rec.seen) == 1
    assert rec.seen[0]["headers"]["user-agent"] == USER_AGENT


# --- gemini: thought parts dropped -------------------------------------------------


def _gemini_body(parts: list[dict[str, Any]]) -> bytes:
    reply = {
        "candidates": [{"content": {"role": "model", "parts": parts}}],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2},
    }
    return json.dumps(reply).encode("utf-8")


def test_gemini_drops_thought_parts() -> None:
    body = _gemini_body([{"text": "hidden", "thought": True}, {"text": "answer"}])
    parsed = kind_gemini.parse_response(200, body)
    assert parsed.text == "answer"
    assert parsed.prompt_tokens == 3
    assert parsed.completion_tokens == 2


def test_gemini_keeps_part_with_thought_false() -> None:
    body = _gemini_body([{"text": "x", "thought": False}])
    assert kind_gemini.parse_response(200, body).text == "x"


def test_gemini_drops_thought_parts_among_several_answers() -> None:
    body = _gemini_body(
        [
            {"text": "a"},
            {"text": "think-1", "thought": True},
            {"text": "b", "thought": False},
            {"text": "think-2", "thought": True},
        ]
    )
    assert kind_gemini.parse_response(200, body).text == "ab"


def test_gemini_only_thought_parts_raises() -> None:
    body = _gemini_body([{"text": "hidden", "thought": True}, {"text": "more", "thought": True}])
    with pytest.raises(ResponseParseError):
        kind_gemini.parse_response(200, body)


# --- json schema still passed by every kind -------------------------------------------

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
}


def _schema_payload(kind: ModuleType) -> dict[str, Any]:
    request = kind.build_request(
        base_url="http://127.0.0.1:1",
        model="test-model",
        prompt="hello",
        system=None,
        json_mode=dict(_SCHEMA),
        temperature=0.0,
        seed=None,
        limits={},
        api_key="test-key",
    )
    payload = json.loads(request.body.decode("utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_ollama_sends_schema_as_format() -> None:
    assert _schema_payload(kind_ollama)["format"] == _SCHEMA


def test_gemini_sends_schema_as_response_json_schema() -> None:
    config = _schema_payload(kind_gemini)["generationConfig"]
    assert config["responseJsonSchema"] == _SCHEMA
    assert config["responseMimeType"] == "application/json"


def test_openai_sends_schema_in_response_format() -> None:
    assert _schema_payload(kind_openai)["response_format"] == {
        "type": "json_schema",
        "json_schema": _SCHEMA,
    }
