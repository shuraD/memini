"""Request building and reply parsing for the ``gemini`` provider kind.

Pure functions only: a ``generateContent`` request is described as data and a reply
body is parsed into text and token counts; no network access lives here. Inputs and
outputs match :mod:`memini.infra.llm_providers.kind_openai`.
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

_JSON_MIME = "application/json"
_KEY_HEADER = "x-goog-api-key"


def _generation_config(
    json_mode: bool | dict[str, Any],
    temperature: float,
    seed: int | None,
    limits: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the ``generationConfig`` object for the given sampling and output options.

    ``json_mode=True`` asks for a JSON reply; a mapping is also sent as the JSON schema.
    """
    config: dict[str, Any] = {"temperature": temperature}
    if seed is not None:
        config["seed"] = seed
    if json_mode:
        config["responseMimeType"] = _JSON_MIME
    if isinstance(json_mode, dict):
        config["responseJsonSchema"] = json_mode
    max_tokens = limits.get("max_tokens")
    if max_tokens is not None:
        config["maxOutputTokens"] = int(max_tokens)
    return config


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
    """Build the ``generateContent`` request for one prompt.

    ``system`` goes in ``systemInstruction`` (never in ``contents``); the ``max_tokens``
    limit becomes ``maxOutputTokens`` and ``timeout_s`` the request timeout. The key, if
    any, goes only in the ``x-goog-api-key`` header.
    """
    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": _generation_config(json_mode, temperature, seed, limits),
    }
    if system is not None:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    headers = {"Content-Type": _JSON_MIME, "User-Agent": USER_AGENT}
    if api_key:
        headers[_KEY_HEADER] = api_key
    timeout = limits.get("timeout_s")
    return HttpRequest(
        method="POST",
        url=f"{base_url.rstrip('/')}/models/{model}:generateContent",
        headers=headers,
        body=json.dumps(payload).encode("utf-8"),
        timeout_s=None if timeout is None else float(timeout),
    )


def _token_count(usage: Mapping[str, Any], name: str) -> int:
    """Return the integer token count ``name`` from ``usage``, or 0 when absent."""
    value = usage.get(name)
    return int(value) if isinstance(value, (int, float)) else 0


def parse_response(status: int, body: bytes) -> ParsedReply:
    """Parse a ``generateContent`` reply body into text and token counts.

    The text is the concatenation of ``candidates[0].content.parts[*].text``, skipping
    parts whose ``thought`` is truthy; no remaining text raises ``ResponseParseError``
    naming the field. Missing ``usageMetadata`` gives zero counts. ``status`` is
    included in error messages.
    """
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResponseParseError(f"status {status}: body is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ResponseParseError(f"status {status}: body is not a JSON object")
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        raise ResponseParseError(f"status {status}: missing field 'candidates[0]'")
    content = candidates[0].get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    texts = (
        [
            p["text"]
            for p in parts
            if isinstance(p, dict) and isinstance(p.get("text"), str) and not p.get("thought")
        ]
        if isinstance(parts, list)
        else []
    )
    if not texts:
        raise ResponseParseError(
            f"status {status}: missing field 'candidates[0].content.parts[].text'"
        )
    usage = data.get("usageMetadata")
    usage_map: Mapping[str, Any] = usage if isinstance(usage, dict) else {}
    return ParsedReply(
        text="".join(texts),
        prompt_tokens=_token_count(usage_map, "promptTokenCount"),
        completion_tokens=_token_count(usage_map, "candidatesTokenCount"),
    )
