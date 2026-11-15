"""Acceptance tests for LP-06: the urllib HTTP transport with a request timeout."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType
from typing import Any

import pytest

from memini.infra.llm_providers import transport
from memini.infra.llm_providers.kind_openai import HttpRequest

USER_AGENT = "memini-test-agent/1.0"


class _Recorder:
    """What the local server saw, and how it should answer."""

    def __init__(self) -> None:
        self.status = 200
        self.body = b'{"ok": true}'
        self.extra_headers: dict[str, str] = {}
        self.release = threading.Event()
        self.hold = False
        self.seen: list[dict[str, Any]] = []


def _make_handler(rec: _Recorder) -> type[BaseHTTPRequestHandler]:
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
            if rec.hold:
                rec.release.wait(5.0)
            try:
                self.send_response(rec.status)
                for name, value in rec.extra_headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(rec.body)))
                self.end_headers()
                self.wfile.write(rec.body)
            except OSError:
                pass

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
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(rec))
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    try:
        yield f"http://{host!s}:{port}", rec
    finally:
        rec.release.set()
        httpd.shutdown()
        httpd.server_close()
        thread.join(5.0)


def _request(
    url: str,
    *,
    method: str = "POST",
    body: bytes = b'{"q": 1}',
    timeout_s: float | None = 5.0,
) -> HttpRequest:
    return HttpRequest(
        method=method,
        url=url,
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
            "Authorization": "Bearer test-key",
        },
        body=body,
        timeout_s=timeout_s,
    )


def _lower(headers: Any) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in dict(headers).items()}


def test_200_returns_status_and_body(server: tuple[str, _Recorder]) -> None:
    base, rec = server
    rec.body = b'{"answer": "hello"}'
    resp = transport.UrllibTransport().send(_request(base + "/v1/chat"))
    assert resp.status == 200
    assert resp.body == b'{"answer": "hello"}'
    assert isinstance(resp.body, bytes)


def test_response_is_frozen_dataclass(server: tuple[str, _Recorder]) -> None:
    base, _rec = server
    resp = transport.UrllibTransport().send(_request(base + "/"))
    assert isinstance(resp, transport.HttpResponse)
    assert dataclasses.is_dataclass(resp)
    with pytest.raises(dataclasses.FrozenInstanceError):
        resp.status = 500  # type: ignore[misc]


def test_http_response_fields() -> None:
    resp = transport.HttpResponse(status=201, headers={"X-A": "b"}, body=b"x")
    assert resp.status == 201
    assert dict(resp.headers) == {"X-A": "b"}
    assert resp.body == b"x"


def test_429_returns_status_and_header_without_raising(
    server: tuple[str, _Recorder],
) -> None:
    base, rec = server
    rec.status = 429
    rec.body = b'{"error": "slow down"}'
    rec.extra_headers = {"Retry-After": "7"}
    resp = transport.UrllibTransport().send(_request(base + "/v1/chat"))
    assert resp.status == 429
    assert _lower(resp.headers).get("retry-after") == "7"
    assert resp.body == b'{"error": "slow down"}'


@pytest.mark.parametrize("status", [400, 401, 404, 500, 503])
def test_error_statuses_never_raise(server: tuple[str, _Recorder], status: int) -> None:
    base, rec = server
    rec.status = status
    rec.body = b"err"
    resp = transport.UrllibTransport().send(_request(base + "/x"))
    assert resp.status == status
    assert resp.body == b"err"


def test_headers_are_str_mapping(server: tuple[str, _Recorder]) -> None:
    base, rec = server
    rec.extra_headers = {"X-Custom": "value-1"}
    resp = transport.UrllibTransport().send(_request(base + "/"))
    lowered = _lower(resp.headers)
    assert lowered.get("x-custom") == "value-1"
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in resp.headers.items())


def test_slow_server_raises_transport_timeout(server: tuple[str, _Recorder]) -> None:
    base, rec = server
    rec.hold = True
    try:
        with pytest.raises(transport.TransportTimeout):
            transport.UrllibTransport().send(_request(base + "/slow", timeout_s=0.3))
    finally:
        rec.release.set()
    assert issubclass(transport.TransportTimeout, Exception)


def test_closed_port_raises_transport_error() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    with pytest.raises(transport.TransportError):
        transport.UrllibTransport().send(_request(f"http://127.0.0.1:{port}/", timeout_s=2.0))
    assert issubclass(transport.TransportError, Exception)


def test_server_sees_user_agent_method_and_body(server: tuple[str, _Recorder]) -> None:
    base, rec = server
    transport.UrllibTransport().send(_request(base + "/v1/chat?alt=json", body=b'{"q": 42}'))
    assert len(rec.seen) == 1
    seen = rec.seen[0]
    assert seen["headers"].get("user-agent") == USER_AGENT
    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/chat?alt=json"
    assert seen["body"] == b'{"q": 42}'
    assert seen["headers"].get("authorization") == "Bearer test-key"
    assert seen["headers"].get("content-type") == "application/json"


def test_get_request_sends_user_agent(server: tuple[str, _Recorder]) -> None:
    base, rec = server
    resp = transport.UrllibTransport().send(
        _request(base + "/health", method="GET", body=b"", timeout_s=None)
    )
    assert resp.status == 200
    assert rec.seen[0]["method"] == "GET"
    assert rec.seen[0]["headers"].get("user-agent") == USER_AGENT


def test_one_send_makes_one_request_no_retry(server: tuple[str, _Recorder]) -> None:
    base, rec = server
    rec.status = 503
    transport.UrllibTransport().send(_request(base + "/"))
    assert len(rec.seen) == 1


def test_transport_protocol_has_send() -> None:
    assert hasattr(transport.Transport, "send")
    assert callable(transport.UrllibTransport().send)


def test_import_opens_no_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    def _no_connect(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        raise AssertionError("connection opened at import time")

    monkeypatch.setattr(socket, "create_connection", _no_connect)
    monkeypatch.setattr(socket.socket, "connect", _no_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _no_connect)
    reloaded = importlib.reload(transport)
    assert calls == []
    assert hasattr(reloaded, "UrllibTransport")


def _module_level_mutables(module: ModuleType) -> list[str]:
    bad: list[str] = []
    for name, value in vars(module).items():
        if name.startswith("__") and name.endswith("__"):
            continue
        if inspect.ismodule(value) or inspect.isclass(value) or callable(value):
            continue
        if isinstance(value, (list, dict, set, bytearray)):
            bad.append(name)
    return bad


def test_no_module_level_mutable_state() -> None:
    assert _module_level_mutables(transport) == []
