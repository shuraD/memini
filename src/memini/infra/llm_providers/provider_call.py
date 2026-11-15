"""Perform one generation against one provider for one role.

Picks the provider's model for the role, reads the key from the given environment
mapping, builds the request with the builder of the provider's ``kind``, sends it,
classifies the answer and retries once on a server error or a timeout. No health
tracking, fallback, cache, budget or logging lives here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from memini.infra.llm_providers import kind_gemini, kind_ollama, kind_openai
from memini.infra.llm_providers.config import LLMConfig, ProviderSpec
from memini.infra.llm_providers.kind_openai import HttpRequest, ParsedReply, ResponseParseError
from memini.infra.llm_providers.outcome import Action, Outcome, classify
from memini.infra.llm_providers.transport import HttpResponse, Transport, TransportError
from memini.infra.llm_providers.types import Attempt, Generation, Reason

__all__ = ["ProviderFailure", "call_provider"]

_MAX_SENDS = 2
"""One send plus one retry when the outcome says to retry."""

_SUCCESS_MIN = 200
_SUCCESS_MAX = 300


class ProviderFailure(Exception):  # noqa: N818 - public name fixed by the interface
    """One provider could not produce a generation.

    ``attempt`` names the provider, the reason and the detail; ``outcome`` is the
    classified answer (``None`` when nothing was sent) and its ``action`` tells the
    caller what to do next.
    """

    def __init__(self, attempt: Attempt, outcome: Outcome | None) -> None:
        """Store the attempt and outcome and build the message from the attempt."""
        self.attempt: Attempt = attempt
        self.outcome: Outcome | None = outcome
        super().__init__(f"{attempt.provider}: {attempt.reason.value} ({attempt.detail})")


class _BuildRequest(Protocol):
    """The request builder every provider kind exposes."""

    def __call__(
        self,
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
    ) -> HttpRequest: ...


@dataclass(frozen=True)
class _Kind:
    """The request builder and reply parser of one provider kind."""

    build: _BuildRequest
    parse: Callable[[int, bytes], ParsedReply]


_KINDS: Mapping[str, _Kind] = MappingProxyType(
    {
        "openai_compatible": _Kind(kind_openai.build_request, kind_openai.parse_response),
        "gemini": _Kind(kind_gemini.build_request, kind_gemini.parse_response),
        "ollama": _Kind(kind_ollama.build_request, kind_ollama.parse_response),
    }
)
"""Builders and parsers keyed by provider ``kind``; the provider name is never read."""


def _fail(spec: ProviderSpec, reason: Reason, detail: str) -> ProviderFailure:
    """Return a failure raised before anything was sent."""
    return ProviderFailure(Attempt(spec.name, reason, detail), None)


def _classify(spec: ProviderSpec, response: HttpResponse | None, exc: Exception | None) -> Outcome:
    """Classify a non-success answer or a send/parse error with the provider's settings."""
    status = None if response is None else response.status
    headers: Mapping[str, str] = {} if response is None else response.headers
    body = "" if response is None else response.body.decode("utf-8", errors="replace")
    if spec.exhausted_markers is None:
        return classify(status, headers, body, exc=exc, rest_default_s=spec.rest_default_s)
    return classify(
        status,
        headers,
        body,
        exc=exc,
        rest_default_s=spec.rest_default_s,
        exhausted_markers=spec.exhausted_markers,
    )


def _send_once(
    spec: ProviderSpec, kind: _Kind, request: HttpRequest, transport: Transport
) -> ParsedReply | Outcome:
    """Send ``request`` once and return the parsed reply, or the outcome of the failure."""
    try:
        response = transport.send(request)
    except TransportError as exc:
        return _classify(spec, None, exc)
    if not _SUCCESS_MIN <= response.status < _SUCCESS_MAX:
        return _classify(spec, response, None)
    try:
        return kind.parse(response.status, response.body)
    except ResponseParseError as exc:
        return _classify(spec, None, exc)


def call_provider(
    spec: ProviderSpec,
    config: LLMConfig,
    role: str,
    prompt: str,
    *,
    system: str | None,
    json_mode: bool | dict[str, Any],
    temperature: float,
    seed: int | None,
    env: Mapping[str, str],
    transport: Transport,
    clock: Callable[[], float],
) -> Generation:
    """Generate one completion for ``prompt`` from ``spec`` serving ``role``.

    The key is read from ``env`` under ``spec.key_env`` (a provider without
    ``key_env`` needs none). Limits are the provider's, overridden by the role's
    model. ``seconds`` is the ``clock`` difference across the sends. ``config`` is
    accepted for the caller's convenience; everything needed is on ``spec``.

    Raises:
        ProviderFailure: on every non-success. A missing key or model raises with
            reason ``NO_KEY`` / ``NO_MODEL`` and ``outcome=None`` before any send;
            otherwise ``outcome.action`` tells the caller what to do next.

    """
    del config  # The spec already carries the provider's models and limits.
    api_key: str | None = None
    if spec.key_env is not None:
        api_key = env.get(spec.key_env)
        if not api_key:
            raise _fail(spec, Reason.NO_KEY, f"environment variable {spec.key_env} is not set")
    model = spec.models.get(role)
    if model is None:
        raise _fail(spec, Reason.NO_MODEL, f"no model for role {role!r}")
    kind = _KINDS[spec.kind]
    request = kind.build(
        base_url=spec.base_url,
        model=model.name,
        prompt=prompt,
        system=system,
        json_mode=json_mode,
        temperature=temperature,
        seed=seed,
        limits={**spec.limits, **model.limits},
        api_key=api_key,
    )
    started = clock()
    result: ParsedReply | Outcome | None = None
    for _ in range(_MAX_SENDS):
        result = _send_once(spec, kind, request, transport)
        if isinstance(result, ParsedReply):
            return Generation(
                text=result.text,
                provider=spec.name,
                model=model.name,
                role=role,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                seconds=clock() - started,
                cached=False,
            )
        if result.action is not Action.RETRY_THEN_NEXT:
            break
    if not isinstance(result, Outcome):  # unreachable: the loop runs at least once
        raise _fail(spec, Reason.SERVER_ERROR, "no request was sent")
    raise ProviderFailure(Attempt(spec.name, result.reason, result.detail), result)
