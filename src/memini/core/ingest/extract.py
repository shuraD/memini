"""Extraction prompts per window and one batched LLM call for all windows.

Each window of a file gets one prompt built from text-file templates (AGENTS.md rule 3):
the persona chosen by the file's stored role, then its domain, then a caller-given default
(rule 2), and the user's own labels shown through the related-tags line. All prompts go to
the LLM in a single batched, temperature-0, JSON-mode call; one result comes back per
window, in order.

Each reply is then read through the shared JSON helper (repair, salvage of a cut-off
reply, corrective re-ask). A window whose reply stays unusable is split in half on a
line boundary and both halves are extracted recursively; when no window yields any
entity, one fallback entity per window holds its full text.
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, overload

from memini.infra import paths
from memini.infra.llm_providers import LLM, Generation, LLMError
from memini.infra.llm_providers.json_reply import JsonStatus, generate_json

TEMPLATE_FILE = "extraction.txt"
RELATED_TAGS_FILE = "extraction_related_tags.txt"

FALLBACK_TYPE = "Fallback"
REPAIR_TEMPLATE = (
    "Your previous reply could not be read as JSON ({error}).\n"
    "Previous reply:\n{reply}\n\n"
    "Answer the original request again with valid JSON only.\n"
    "Original request:\n{prompt}"
)

Ask = Callable[[str], Any]
"""A model call: returns the reply text, a ``(text, usage)`` tuple, or None."""

_PLACEHOLDER = r"\{(persona|related_tags|content)\}"
_ENTITIES_KEY = "entities"
_ENTITY_KEYS = ("id", "type", "content")
_META_TYPE = "Meta"


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


@dataclass(frozen=True)
class WindowReply:
    """The entities read from one extraction reply, or why none could be read."""

    usable: bool
    entities: tuple[dict[str, Any], ...] = ()
    status: JsonStatus = JsonStatus.UNREADABLE
    reasks: int = 0
    error: str | None = None


def _reply_entities(value: Any) -> list[Any]:
    """Return the candidate entities held by one decoded reply value."""
    if isinstance(value, list):
        return value
    if not isinstance(value, dict):
        return []
    if _ENTITIES_KEY in value:
        inner = value[_ENTITIES_KEY]
        return inner if isinstance(inner, list) else []
    return [value] if any(key in value for key in _ENTITY_KEYS) else []


def read_extraction_reply(
    ask: Ask, prompt: str, *, attempts: int = 1, repair_template: str = REPAIR_TEMPLATE
) -> WindowReply:
    """Ask once (plus corrective re-asks) and read the reply's entities; never raises.

    Items that are not dicts and ``Meta`` entities are dropped; in a salvaged
    (cut-off) reply an entity equal to an earlier one is dropped too.
    """
    try:
        result = generate_json(
            ask,
            prompt,
            None,
            repair_template=repair_template,
            attempts=attempts,
            salvage_key=_ENTITIES_KEY,
            reask_truncated=False,
        )
    except Exception as exc:  # any provider failure makes the reply unusable
        return WindowReply(usable=False, error=f"{type(exc).__name__}: {exc}")
    if not result.ok:
        error = result.error or "unreadable reply"
        return WindowReply(usable=False, status=result.status, reasks=result.reasks, error=error)
    entities: list[dict[str, Any]] = []
    for item in _reply_entities(result.value):
        if not isinstance(item, dict) or item.get("type") == _META_TYPE:
            continue
        if result.truncated and item in entities:
            continue
        entities.append(item)
    return WindowReply(
        usable=True, entities=tuple(entities), status=result.status, reasks=result.reasks
    )


def extract_text(
    ask: Ask,
    text: str,
    build_prompt: Callable[[str], str],
    *,
    max_depth: int = 4,
    min_chars: int = 400,
    attempts: int = 1,
    repair_template: str = REPAIR_TEMPLATE,
    depth: int = 0,
) -> list[dict[str, Any]]:
    """Extract the entities of ``text``, halving it on a line boundary while unusable.

    The split stops at ``max_depth`` or once a piece has at most ``min_chars``
    characters; an unusable piece that is not split yields no entities.
    """
    reply = read_extraction_reply(
        ask, build_prompt(text), attempts=attempts, repair_template=repair_template
    )
    if reply.usable:
        return list(reply.entities)
    if depth >= max_depth or len(text) <= min_chars:
        return []
    mid = len(text) // 2
    newline = text.rfind("\n", 0, mid)
    cut = newline if newline > len(text) // 4 else mid
    return [
        entity
        for piece in (text[:cut], text[cut:])
        for entity in extract_text(
            ask,
            piece,
            build_prompt,
            max_depth=max_depth,
            min_chars=min_chars,
            attempts=attempts,
            repair_template=repair_template,
            depth=depth + 1,
        )
    ]


def fallback_entities(artifact_id: str, windows: Sequence[str]) -> list[dict[str, Any]]:
    """Return one fallback entity per window holding the window's full text."""
    return [
        {"id": f"{artifact_id}_{FALLBACK_TYPE}_{i}", "type": FALLBACK_TYPE, "content": window}
        for i, window in enumerate(windows)
    ]


def _extract_texts(
    ask: Ask,
    windows: Sequence[str],
    build_prompt: Callable[[str], str],
    artifact_id: str,
    *,
    max_depth: int = 4,
    min_chars: int = 400,
    attempts: int = 1,
    repair_template: str = REPAIR_TEMPLATE,
) -> list[dict[str, Any]]:
    """Extract every window in order; fall back to one entity per window when none came."""
    entities = [
        entity
        for window in windows
        for entity in extract_text(
            ask,
            window,
            build_prompt,
            max_depth=max_depth,
            min_chars=min_chars,
            attempts=attempts,
            repair_template=repair_template,
        )
    ]
    return entities or fallback_entities(artifact_id, windows)


@overload
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
) -> list[WindowExtraction]: ...


@overload
def extract_windows(
    ask: Ask,
    windows: Sequence[str],
    build_prompt: Callable[[str], str],
    artifact_id: str,
    *,
    max_depth: int = 4,
    min_chars: int = 400,
    attempts: int = 1,
    repair_template: str = REPAIR_TEMPLATE,
) -> list[dict[str, Any]]: ...


def extract_windows(*args: Any, **kwargs: Any) -> list[WindowExtraction] | list[dict[str, Any]]:
    """Run either extraction step, chosen by the call's shape.

    With ``llm=`` (one batched call): see ``_batch_windows``. Otherwise
    ``(ask, windows, build_prompt, artifact_id)``: each window's text is extracted
    with ``extract_text`` in order and the results concatenated; when that list
    is empty, ``fallback_entities(artifact_id, windows)`` is returned instead.
    """
    if "llm" in kwargs:
        return _batch_windows(*args, **kwargs)
    return _extract_texts(*args, **kwargs)


def _batch_windows(
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
