"""HTTP transport over the standard library, with the request's own timeout.

A transport sends one :class:`HttpRequest` and returns the status, headers and
body as they came back, never through a proxy. HTTP error statuses are answers,
not failures: they are returned like any other status. Only a timeout or a
connection failure raises. No retries, no fallbacks; the module opens nothing at import time.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from memini.infra.llm_providers.kind_openai import HttpRequest

__all__ = [
    "HttpResponse",
    "Transport",
    "TransportError",
    "TransportTimeout",
    "UrllibTransport",
]

_ALLOWED_SCHEMES = ("http://", "https://")

# An empty proxy table: environment and operating-system proxy settings are ignored.
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class TransportError(Exception):
    """The request could not be exchanged with the server (connection failure)."""


class TransportTimeout(TransportError):  # noqa: N818 - name fixed by the interface
    """The server did not answer within the request's ``timeout_s``."""


@dataclass(frozen=True)
class HttpResponse:
    """What the server answered: status code, headers and raw body."""

    status: int
    headers: Mapping[str, str]
    body: bytes


class Transport(Protocol):
    """Anything able to send one HTTP request and return the server's answer."""

    def send(self, request: HttpRequest) -> HttpResponse:
        """Send ``request`` once and return the answer, whatever its status."""
        ...


def _headers(items: Iterable[tuple[object, object]]) -> dict[str, str]:
    """Copy response header pairs into a plain ``str`` mapping."""
    return {str(name): str(value) for name, value in items}


class UrllibTransport:
    """A :class:`Transport` built on :mod:`urllib.request`, stateless."""

    def send(self, request: HttpRequest) -> HttpResponse:
        """Send ``request`` once and return status, headers and body.

        Raises:
            TransportTimeout: the server did not answer within ``timeout_s``.
            TransportError: the URL scheme is not HTTP(S) or the connection failed.

        """
        if not request.url.lower().startswith(_ALLOWED_SCHEMES):
            raise TransportError(f"unsupported URL scheme: {request.url!r}")
        data = request.body or None
        headers = dict(request.headers)
        # The scheme is checked as http(s) above, so urllib opens no file: or custom URL.
        native = urllib.request.Request(request.url, data, headers, method=request.method)  # noqa: S310  # nosec B310
        try:
            with _NO_PROXY_OPENER.open(native, timeout=request.timeout_s) as answer:
                return HttpResponse(
                    status=int(answer.status),
                    headers=_headers(answer.headers.items()),
                    body=answer.read(),
                )
        except urllib.error.HTTPError as error:
            try:
                body = error.read()
            except TimeoutError as exc:
                raise TransportTimeout(str(exc)) from exc
            finally:
                error.close()
            return HttpResponse(
                status=int(error.code),
                headers=_headers(error.headers.items()),
                body=body,
            )
        except urllib.error.URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise TransportTimeout(str(error.reason)) from error
            raise TransportError(str(error.reason)) from error
        except TimeoutError as error:
            raise TransportTimeout(str(error)) from error
        except OSError as error:
            raise TransportError(str(error)) from error
