"""Extraction prompts per window and one batched LLM call for all windows.

Each window of a file gets one prompt built from text-file templates (AGENTS.md rule 3):
the persona chosen by the file's stored role, then its domain, then a caller-given default
(rule 2), and the user's own labels shown through the related-tags line. All prompts go to
the LLM in a single batched, temperature-0, JSON-mode call; one result comes back per
window, in order. Parsing the replies is a later step.
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from memini.infra import paths
from memini.infra.llm_providers import LLM, Generation, LLMError

TEMPLATE_FILE = "extraction.txt"
RELATED_TAGS_FILE = "extraction_related_tags.txt"

_PLACEHOLDER = r"\{(persona|related_tags|content)\}"


class Window(Protocol):
    """Any window carrying the text to extract from (e.g. an IN-09 window)."""

    @property
    def text(self) -> str:
        """The window's text."""
        ...


@dataclass(frozen=True)
class ExtractionPrompts:
    """The extraction prompt templates, as read from their text files."""

    template: str
    related_tags: str


class UnknownPersonaError(LookupError):
    """Raised when the chosen persona name has no persona text."""


@dataclass(frozen=True)
class WindowExtraction:
    """The LLM's answer for one window: exactly one of ``generation`` or ``error`` is set."""

    index: int
    prompt: str
    generation: Generation | None
    error: LLMError | None


def load_prompts(directory: pathlib.Path | None = None) -> ExtractionPrompts:
    """Read both templates (utf-8) from ``directory``, default the prompts directory.

    Raises:
        FileNotFoundError: when either file is missing.

    """
    base = paths.prompts_dir() if directory is None else directory
    return ExtractionPrompts(
        template=(base / TEMPLATE_FILE).read_text(encoding="utf-8"),
        related_tags=(base / RELATED_TAGS_FILE).read_text(encoding="utf-8"),
    )


def persona_for(role: str | None, domain: str | None, default: str) -> str:
    """Return the persona name: ``role`` if set, else ``domain`` if set, else ``default``."""
    return role or domain or default


def user_labels(labels: Sequence[str], derived: Collection[str]) -> list[str]:
    """Return the user's own labels: stripped, non-blank, not derived, first occurrence only."""
    kept: dict[str, None] = {}
    for label in labels:
        stripped = label.strip()
        if stripped and stripped not in derived:
            kept.setdefault(stripped, None)
    return list(kept)


def related_tags_line(labels: Sequence[str], prompts: ExtractionPrompts) -> str:
    """Return the related-tags line for ``labels``, or ``""`` when none is non-blank."""
    tags = [label.strip() for label in labels if label.strip()]
    if not tags:
        return ""
    return prompts.related_tags.strip().replace("{tags}", ", ".join(tags))


def build_prompt(
    text: str, *, persona: str, labels: Sequence[str], prompts: ExtractionPrompts
) -> str:
    """Fill the template's placeholders in one pass; inserted values are never re-scanned."""
    values = {
        "persona": persona,
        "related_tags": related_tags_line(labels, prompts),
        "content": text,
    }
    return re.sub(_PLACEHOLDER, lambda match: values[match.group(1)], prompts.template)


def extract_windows(
    windows: Sequence[Window],
    *,
    role: str | None,
    domain: str | None,
    default_persona: str,
    personas: Mapping[str, str],
    labels: Sequence[str],
    derived_labels: Collection[str],
    llm: LLM,
    prompts: ExtractionPrompts,
) -> list[WindowExtraction]:
    """Send one prompt per window in one batched JSON call; return one result per window.

    Raises:
        UnknownPersonaError: when the chosen persona name is not in ``personas``
            (no LLM call is made).

    """
    name = persona_for(role, domain, default_persona)
    if name not in personas:
        raise UnknownPersonaError(name)
    if not windows:
        return []
    own = user_labels(labels, derived_labels)
    sent = [
        build_prompt(w.text, persona=personas[name], labels=own, prompts=prompts) for w in windows
    ]
    replies = llm.generate_many(sent, json_mode=True, temperature=0.0)
    return [
        WindowExtraction(
            index=i,
            prompt=prompt,
            generation=None if isinstance(reply, LLMError) else reply,
            error=reply if isinstance(reply, LLMError) else None,
        )
        for i, (prompt, reply) in enumerate(zip(sent, replies, strict=True))
    ]
