"""Embed texts with the single local embedding model served by an ``ollama`` provider.

Each text gets the task prefix of its kind (query or document) before it is sent in one
``POST {base_url}/api/embed`` request. A failure raises :class:`LLMError` naming the
provider and the reason; there is no fallback to another provider, no cache and no
batching here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from memini.infra.llm_providers.config import EMBEDDING, LLMConfig
from memini.infra.llm_providers.kind_openai import USER_AGENT, HttpRequest
from memini.infra.llm_providers.outcome import classify
from memini.infra.llm_providers.transport import Transport, TransportError
from memini.infra.llm_providers.types import Attempt, LLMError, Reason

__all__ = ["DEFAULT_PREFIXES", "OllamaEmbedder"]

DEFAULT_PREFIXES: Mapping[str, str] = {
    "query": "search_query: ",
    "document": "search_document: ",
}
"""Task prefixes used when the config has no ``embedding_prefixes`` section."""

_KIND = "ollama"
_EMBED_PATH = "/api/embed"
_SUCCESS_MIN = 200
_SUCCESS_MAX = 300


def _select_provider(config: LLMConfig, provider: str | None) -> str:
    """Return the name of the ``ollama`` provider with an embedding model.

    ``provider`` names it explicitly; otherwise the first matching one in config order.
    Raises ``LLMError`` (``NO_MODEL``) when no such provider exists.
    """
    for spec in config.providers:
        if provider is not None and spec.name != provider:
            continue
        if spec.kind == _KIND and EMBEDDING in spec.models:
            return spec.name
    name = provider or ""
    detail = f"no {_KIND} provider with an {EMBEDDING!r} model"
    raise LLMError((Attempt(name, Reason.NO_MODEL, detail),), EMBEDDING)


def _parse_vectors(body: bytes, expected: int) -> list[list[float]] | None:
    """Return ``expected`` float vectors from an ``/api/embed`` reply, else ``None``."""
    try:
        data: Any = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    embeddings = data.get("embeddings") if isinstance(data, dict) else None
    if not isinstance(embeddings, list) or len(embeddings) != expected:
        return None
    vectors: list[list[float]] = []
    for vector in embeddings:
        if not isinstance(vector, list) or not all(
            isinstance(x, int | float) and not isinstance(x, bool) for x in vector
        ):
            return None
        vectors.append([float(x) for x in vector])
    return vectors


class OllamaEmbedder:
    """An :class:`~memini.infra.llm_providers.types.Embedder` over a local Ollama server."""

    def __init__(
        self,
        config: LLMConfig,
        env: Mapping[str, str],
        transport: Transport,
        *,
        provider: str | None = None,
    ) -> None:
        """Bind to the ``ollama`` provider that has an ``embedding`` model.

        ``provider`` picks one by name; by default the first such provider is used.
        ``env`` is accepted for a uniform constructor; the local server needs no key.
        """
        del env
        self._name = _select_provider(config, provider)
        spec = config.provider(self._name)
        model = config.model_for(self._name, EMBEDDING)
        if model is None:  # guaranteed by _select_provider; kept for the type checker
            raise LLMError((Attempt(self._name, Reason.NO_MODEL, "no model"),), EMBEDDING)
        self._model: str = model
        self._url = spec.base_url.rstrip("/") + _EMBED_PATH
        timeout = config.limit(self._name, EMBEDDING, "timeout_s")
        self._timeout: float | None = None if timeout is None else float(timeout)
        self._rest_default_s = spec.rest_default_s
        self._markers = spec.exhausted_markers
        self._prefixes: Mapping[str, str] = config.embedding_prefixes or DEFAULT_PREFIXES
        self._transport = transport

    def _fail(self, reason: Reason, detail: str) -> LLMError:
        """Build the error for one failed attempt at this provider."""
        return LLMError((Attempt(self._name, reason, detail),), EMBEDDING)

    def embed(
        self, texts: Sequence[str], *, kind: Literal["query", "document"]
    ) -> list[list[float]]:
        """Return one vector per text, in order, each text prefixed for ``kind``.

        An empty ``texts`` returns ``[]`` without any request. Raises ``LLMError`` on a
        transport failure, a non-2xx status or a malformed reply.
        """
        if not texts:
            return []
        prefix = self._prefixes.get(kind, DEFAULT_PREFIXES[kind])
        payload = {"model": self._model, "input": [prefix + text for text in texts]}
        request = HttpRequest(
            method="POST",
            url=self._url,
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            body=json.dumps(payload).encode("utf-8"),
            timeout_s=self._timeout,
        )
        try:
            response = self._transport.send(request)
        except TransportError as exc:
            outcome = classify(None, {}, "", exc=exc, rest_default_s=self._rest_default_s)
            raise self._fail(outcome.reason, outcome.detail) from exc
        if not _SUCCESS_MIN <= response.status < _SUCCESS_MAX:
            text = response.body.decode("utf-8", errors="replace")
            if self._markers is None:
                outcome = classify(
                    response.status,
                    response.headers,
                    text,
                    rest_default_s=self._rest_default_s,
                )
            else:
                outcome = classify(
                    response.status,
                    response.headers,
                    text,
                    rest_default_s=self._rest_default_s,
                    exhausted_markers=self._markers,
                )
            raise self._fail(outcome.reason, outcome.detail)
        vectors = _parse_vectors(response.body, len(texts))
        if vectors is None:
            detail = f"{response.status} reply has no {len(texts)} embeddings"
            raise self._fail(Reason.SERVER_ERROR, detail)
        return vectors
