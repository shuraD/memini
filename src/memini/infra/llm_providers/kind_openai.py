"""Request building and reply parsing for the ``openai_compatible`` provider kind.

Pure functions only: a chat-completions request is described as data and a reply body
is parsed into text and token counts; no network access lives here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

USER_AGENT = "memini-llm-client/1"
"""The ``User-Agent`` sent on every provider request; later kinds import it."""

_CHAT_PATH = "/chat/completions"


@dataclass(frozen=True)
class HttpRequest:
    """One HTTP request described as data, ready for a transport to send."""

    method: str
    url: str
    headers: Mapping[str, str]
    body: bytes
    timeout_s: float | None


@dataclass(frozen=True)
class ParsedReply:
    """The text of a reply and the token counts the provider reported."""

    text: str
    prompt_tokens: int
    completion_tokens: int


class ResponseParseError(ValueError):
    """Raised when a reply body lacks a field it must carry; the message names the field."""


def _response_format(json_mode: bool | dict[str, Any]) -> dict[str, Any] | None:
    """Return the ``response_format`` value for ``json_mode``, or ``None`` for plain text.

    ``True`` asks for any JSON object; a mapping is sent as the JSON schema to follow.
    """
    if isinstance(json_mode, dict):
        return {"type": "json_schema", "json_schema": json_mode}
    return {"type": "json_object"} if json_mode else None


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
    """Build the chat-completions request for one prompt.

    Messages are ``[system, user]`` when ``system`` is given, else ``[user]``. The
    ``max_tokens`` and ``timeout_s`` limits are used when present; ``seed`` is omitted
    when ``None``. The key, if any, goes only in the ``Authorization`` header.
    """
    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if seed is not None:
        payload["seed"] = seed
    response_format = _response_format(json_mode)
    if response_format is not None:
        payload["response_format"] = response_format
    max_tokens = limits.get("max_tokens")
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    timeout = limits.get("timeout_s")
    return HttpRequest(
        method="POST",
        url=base_url.rstrip("/") + _CHAT_PATH,
        headers=headers,
        body=json.dumps(payload).encode("utf-8"),
        timeout_s=None if timeout is None else float(timeout),
    )


def _token_count(usage: Mapping[str, Any], name: str) -> int:
    """Return the integer token count ``name`` from ``usage``, or 0 when absent."""
    value = usage.get(name)
    return int(value) if isinstance(value, (int, float)) else 0


def parse_response(status: int, body: bytes) -> ParsedReply:
    """Parse a chat-completions reply body into text and token counts.

    Missing ``usage`` gives zero counts; a missing ``choices[0].message.content`` raises
    ``ResponseParseError`` naming the field. ``status`` is included in error messages.
    """
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResponseParseError(f"status {status}: body is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ResponseParseError(f"status {status}: body is not a JSON object")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ResponseParseError(f"status {status}: missing field 'choices[0]'")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ResponseParseError(f"status {status}: missing field 'choices[0].message.content'")
    usage = data.get("usage")
    usage_map: Mapping[str, Any] = usage if isinstance(usage, dict) else {}
    return ParsedReply(
        text=content,
        prompt_tokens=_token_count(usage_map, "prompt_tokens"),
        completion_tokens=_token_count(usage_map, "completion_tokens"),
    )
