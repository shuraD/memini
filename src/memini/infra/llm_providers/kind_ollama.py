"""Request building and reply parsing for the ``ollama`` provider kind.

Pure functions only: an ``/api/chat`` request is described as data and a reply body is
parsed into text and token counts; no network access lives here. Inputs and outputs
match :mod:`memini.infra.llm_providers.kind_openai`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from memini.infra.llm_providers.kind_openai import (
    USER_AGENT,
    HttpRequest,
    ParsedReply,
    ResponseParseError,
)

_CHAT_PATH = "/api/chat"


def build_request(
    *,
    base_url: str,
    model: str,
    prompt: str,
    system: str | None,
    json_mode: bool | dict[str, Any],
    temperature: float,
    seed: int | None,
    limits: Mapping[str, Any],
    api_key: str | None,
) -> HttpRequest:
    """Build the non-streaming ``/api/chat`` request for one prompt.

    Messages are ``[system, user]`` when ``system`` is given, else ``[user]``.
    ``json_mode=True`` sets ``format`` to ``"json"``; a mapping is sent as the schema.
    Temperature, seed and the ``max_tokens`` limit (as ``num_predict``) go in
    ``options``; ``timeout_s`` is the request timeout. ``api_key`` is accepted for a
    uniform signature but never sent: the server is local and keyless.
    """
    del api_key
    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    options: dict[str, Any] = {"temperature": temperature}
    if seed is not None:
        options["seed"] = seed
    max_tokens = limits.get("max_tokens")
    if max_tokens is not None:
        options["num_predict"] = int(max_tokens)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if isinstance(json_mode, dict):
        payload["format"] = json_mode
    elif json_mode:
        payload["format"] = "json"
    timeout = limits.get("timeout_s")
    return HttpRequest(
        method="POST",
        url=base_url.rstrip("/") + _CHAT_PATH,
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        body=json.dumps(payload).encode("utf-8"),
        timeout_s=None if timeout is None else float(timeout),
    )


def _token_count(data: Mapping[str, Any], name: str) -> int:
    """Return the integer token count ``name`` from ``data``, or 0 when absent."""
    value = data.get(name)
    return int(value) if isinstance(value, (int, float)) else 0


def parse_response(status: int, body: bytes) -> ParsedReply:
    """Parse an ``/api/chat`` reply body into text and token counts.

    A missing ``message.content`` raises ``ResponseParseError`` naming the field;
    missing ``prompt_eval_count``/``eval_count`` give zero. ``status`` is included in
    error messages.
    """
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResponseParseError(f"status {status}: body is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ResponseParseError(f"status {status}: body is not a JSON object")
    message = data.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ResponseParseError(f"status {status}: missing field 'message.content'")
    return ParsedReply(
        text=content,
        prompt_tokens=_token_count(data, "prompt_eval_count"),
        completion_tokens=_token_count(data, "eval_count"),
    )
