"""LLM classification of a question into retrieval parameters.

A text-file prompt is filled with the caller's live vocabulary, the model is
asked for a JSON object over a fixed sequence of draws, and only the verdicts
the question text supports are kept. Every allowed value comes from the
caller's ``Vocabulary``; nothing here names a mode, domain, type or style.
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from memini.infra import paths
from memini.infra.llm_providers import LLM, LLMError
from memini.infra.llm_providers.json_reply import read_json

__all__ = [
    "DRAWS",
    "PROMPT_FILE",
    "Classifier",
    "Intent",
    "Vocabulary",
    "build_prompt",
    "default_intent",
    "load_prompt",
]

PROMPT_FILE: str = "query_classify.txt"
DRAWS: tuple[tuple[float, int | None], ...] = ((0.0, None), (0.3, 42), (0.5, 137))
"""(temperature, seed) of each draw, in order."""

_PLACEHOLDER = r"\{(question|modes|domains|types|styles|query_types)\}"
_MIN_STYLE_WORDS = 2
_MAX_STYLE_WORDS = 4


@dataclass(frozen=True)
class Vocabulary:
    """Allowed values; the first of modes, styles and query_types is the default."""

    modes: tuple[str, ...]
    domains: tuple[str, ...]
    types: tuple[str, ...]
    styles: tuple[str, ...]
    query_types: tuple[str, ...]


@dataclass(frozen=True)
class Intent:
    """Retrieval parameters of one question; ``source`` is "llm", "fallback" or "default"."""

    mode: str
    domains: tuple[str, ...]
    types: tuple[str, ...]
    answer_style: str
    query_type: str
    enumerate: bool
    target_concept: str | None
    source: str


def default_intent(vocabulary: Vocabulary) -> Intent:
    """Return the intent made of the vocabulary's defaults and no filters."""
    return Intent(
        mode=vocabulary.modes[0],
        domains=(),
        types=(),
        answer_style=vocabulary.styles[0],
        query_type=vocabulary.query_types[0],
        enumerate=False,
        target_concept=None,
        source="default",
    )


def load_prompt(directory: pathlib.Path | None = None) -> str:
    """Read ``PROMPT_FILE`` (utf-8) from ``directory``, by default the prompts directory.

    Raises FileNotFoundError when the file is missing.
    """
    base = paths.prompts_dir() if directory is None else directory
    return (base / PROMPT_FILE).read_text(encoding="utf-8")


def build_prompt(question: str, vocabulary: Vocabulary, template: str) -> str:
    """Fill the six placeholders of ``template`` in one pass; other braces are kept."""
    values = {
        "question": question,
        "modes": ", ".join(vocabulary.modes),
        "domains": ", ".join(vocabulary.domains),
        "types": ", ".join(vocabulary.types),
        "styles": ", ".join(vocabulary.styles),
        "query_types": ", ".join(vocabulary.query_types),
    }
    return re.sub(_PLACEHOLDER, lambda m: values[m.group(1)], template)


def _match(value: Any, allowed: tuple[str, ...]) -> str | None:
    """Return the member of ``allowed`` equal to ``value`` but for case and spaces, else None."""
    if not isinstance(value, str):
        return None
    key = value.strip().casefold()
    return next((item for item in allowed if item.strip().casefold() == key), None)


def _matches(value: Any, allowed: tuple[str, ...]) -> tuple[str, ...]:
    """Known members named by a list (or one bare string), first occurrence order."""
    items = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    found: list[str] = []
    for item in items:
        member = _match(item, allowed)
        if member is not None and member not in found:
            found.append(member)
    return tuple(found)


def _supported(evidence: Any, question: str) -> str | None:
    """Return the stripped evidence when non-blank and found in the question (any case)."""
    if not isinstance(evidence, str) or not evidence.strip():
        return None
    text = evidence.strip()
    return text if text.casefold() in question.casefold() else None


def _style(reply: Mapping[str, Any], question: str, vocabulary: Vocabulary) -> str:
    """Answer style kept only when backed by 2 to 4 words quoted from the question."""
    default = vocabulary.styles[0]
    style = _match(reply.get("answer_style"), vocabulary.styles)
    if style is None or style == default:
        return default
    evidence = _supported(reply.get("style_evidence"), question)
    if evidence is None or not _MIN_STYLE_WORDS <= len(evidence.split()) <= _MAX_STYLE_WORDS:
        return default
    return style


def _query_type(reply: Mapping[str, Any], question: str, vocabulary: Vocabulary) -> str:
    """Query type kept only when backed by evidence quoted from the question."""
    default = vocabulary.query_types[0]
    kind = _match(reply.get("query_type"), vocabulary.query_types)
    if kind is None or _supported(reply.get("relation_evidence"), question) is None:
        return default
    return kind


def _interpret(text: str, question: str, vocabulary: Vocabulary) -> Intent | None:
    """Intent read from one reply, or None when it is not an object with a known mode."""
    reading = read_json(text)
    if not reading.ok or not isinstance(reading.value, dict):
        return None
    reply: dict[str, Any] = reading.value
    mode = _match(reply.get("mode"), vocabulary.modes)
    if mode is None:
        return None
    enumerate_ = reply.get("enumerate") is True
    concept = reply.get("target_concept")
    target = concept.strip() if enumerate_ and isinstance(concept, str) else ""
    return Intent(
        mode=mode,
        domains=_matches(reply.get("domains"), vocabulary.domains),
        types=_matches(reply.get("artifact_types"), vocabulary.types),
        answer_style=_style(reply, question, vocabulary),
        query_type=_query_type(reply, question, vocabulary),
        enumerate=enumerate_,
        target_concept=target or None,
        source="llm",
    )


class Classifier:
    """Classify questions with an LLM, falling back when no draw gives a valid reply."""

    def __init__(
        self,
        llm: LLM,
        vocabulary: Vocabulary,
        template: str,
        *,
        fallback: Callable[[str], Intent] | None = None,
    ) -> None:
        """Keep the collaborators; no model call is made."""
        self._llm = llm
        self._vocabulary = vocabulary
        self._template = template
        self._fallback = fallback

    def warm(self) -> bool:
        """Make one throwaway call with an empty question; False when it fails."""
        temperature, seed = DRAWS[0]
        prompt = build_prompt("", self._vocabulary, self._template)
        try:
            self._llm.generate(prompt, json_mode=True, temperature=temperature, seed=seed)
        except LLMError:
            return False
        return True

    def classify(self, question: str) -> Intent:
        """Return the first valid draw's intent, else the fallback's, else the default."""
        prompt = build_prompt(question, self._vocabulary, self._template)
        for temperature, seed in DRAWS:
            try:
                reply = self._llm.generate(
                    prompt, json_mode=True, temperature=temperature, seed=seed
                )
            except LLMError:
                continue
            intent = _interpret(reply.text, question, self._vocabulary)
            if intent is not None:
                return intent
        if self._fallback is not None:
            try:
                return self._fallback(question)
            except Exception:  # any fallback failure yields the default
                return default_intent(self._vocabulary)
        return default_intent(self._vocabulary)
