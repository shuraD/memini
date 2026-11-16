"""Document-type detection and file-type mapping driven by config rules (card I14, I16).

Pure functions: no I/O, no model, no clock. Every value list (extensions, keys, words,
type names, file types) comes from the rules mapping given to :func:`parse_rules`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import yaml

_JSON = "json"
_YAML = "yaml"
_PARSE_KINDS = frozenset({_JSON, _YAML})
_FLAGS = ("preserve_original", "enable_entity_extraction")


class ConfigError(ValueError):
    """Raised by :func:`parse_rules` when the rules mapping is invalid."""


@dataclass(frozen=True)
class Detection:
    """The detected document type of a file and how to index it."""

    document_type: str
    indexing_strategy: str
    preserve_original: bool
    enable_entity_extraction: bool


@dataclass(frozen=True)
class _Rule:
    """One detection rule; every condition that is set must hold for a match."""

    detection: Detection
    extensions: frozenset[str] | None
    name_contains: str | None
    parse: str | None
    key_sets: tuple[frozenset[str], ...] | None
    content_words: tuple[str, ...] | None


@dataclass(frozen=True)
class DoctypeRules:
    """Parsed rules; opaque to callers and built only by :func:`parse_rules`."""

    types: tuple[_Rule, ...]
    default: Detection
    by_document_type: Mapping[str, str]
    by_prefix: Mapping[str, str]
    by_extension: Mapping[str, str]
    extension_default: str
    file_type_default: str


def _section(raw: Mapping[str, Any], key: str) -> Any:
    """Return ``raw[key]`` or raise :class:`ConfigError` when it is missing."""
    if key not in raw:
        raise ConfigError(f"missing key: {key}")
    return raw[key]


def _str(raw: Mapping[str, Any], key: str) -> str:
    """Return the required string ``raw[key]``."""
    value = _section(raw, key)
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string")
    return value


def _detection(raw: Mapping[str, Any]) -> Detection:
    """Build the detection a rule or the default yields."""
    if not isinstance(raw, Mapping):
        raise ConfigError("a rule must be a mapping")
    flags = [_section(raw, flag) for flag in _FLAGS]
    if not all(isinstance(flag, bool) for flag in flags):
        raise ConfigError(f"{', '.join(_FLAGS)} must be booleans")
    return Detection(_str(raw, "name"), _str(raw, "indexing_strategy"), flags[0], flags[1])


def _rule(raw: Mapping[str, Any]) -> _Rule:
    """Build one detection rule from its mapping."""
    detection = _detection(raw)
    parse = raw.get("parse")
    if parse is not None and parse not in _PARSE_KINDS:
        raise ConfigError(f"unknown parse kind: {parse!r}")
    extensions = raw.get("extensions")
    key_sets = raw.get("key_sets")
    words = raw.get("content_words")
    return _Rule(
        detection=detection,
        extensions=None if extensions is None else frozenset(e.lower() for e in extensions),
        name_contains=raw.get("name_contains"),
        parse=parse,
        key_sets=None if key_sets is None else tuple(frozenset(s) for s in key_sets),
        content_words=None if words is None else tuple(w.lower() for w in words),
    )


def parse_rules(raw: Mapping[str, Any]) -> DoctypeRules:
    """Parse the document-type rules section; raise :class:`ConfigError` if invalid."""
    types = _section(raw, "types")
    if not isinstance(types, list):
        raise ConfigError("types must be a list")
    files = _section(raw, "file_types")
    if not isinstance(files, Mapping):
        raise ConfigError("file_types must be a mapping")
    return DoctypeRules(
        types=tuple(_rule(rule) for rule in types),
        default=_detection(_section(raw, "default")),
        by_document_type=dict(files.get("by_document_type") or {}),
        by_prefix=dict(files.get("by_prefix") or {}),
        by_extension={k.lower(): v for k, v in (files.get("by_extension") or {}).items()},
        extension_default=_str(files, "extension_default"),
        file_type_default=_str(files, "default"),
    )


_UNPARSED = object()


def _parse(content: str, kind: str) -> object:
    """Parse ``content`` as ``kind``; return ``_UNPARSED`` on failure."""
    if kind == _YAML:
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError:
            pass
    try:
        return json.loads(content)
    except ValueError:
        return _UNPARSED


def _matches_shape(rule: _Rule, content: str) -> bool:
    """Tell whether ``content`` parses as the rule asks and holds one of its key sets."""
    if rule.parse is None:
        return rule.key_sets is None
    parsed = _parse(content, rule.parse)
    if parsed is _UNPARSED:
        return False
    if rule.key_sets is None:
        return True
    return isinstance(parsed, Mapping) and any(keys <= parsed.keys() for keys in rule.key_sets)


def _matches(rule: _Rule, content: str, name: str, extension: str) -> bool:
    """Tell whether every condition set on ``rule`` holds for the file."""
    if rule.extensions is not None and extension not in rule.extensions:
        return False
    if rule.name_contains is not None and rule.name_contains not in name:
        return False
    if rule.content_words is not None:
        text = content.lower()
        if not any(word in text for word in rule.content_words):
            return False
    return _matches_shape(rule, content)


def _extension(filename: str) -> str:
    """Return the lowercased suffix of ``filename`` ("" when it has none)."""
    base = filename.lower().rpartition("/")[2].lstrip(".")
    dot = base.rfind(".")
    return "" if dot < 0 else base[dot:]


def detect(content: str, filename: str, rules: DoctypeRules) -> Detection:
    """Detect the document type of a file: the first matching rule wins, else the default."""
    name = filename.lower()
    extension = _extension(filename)
    for rule in rules.types:
        if _matches(rule, content, name, extension):
            return rule.detection
    return rules.default


def file_type(document_type: str, filename: str, rules: DoctypeRules) -> str:
    """Map a document type and filename to the file type the chunker and prompts use."""
    if document_type in rules.by_document_type:
        return rules.by_document_type[document_type]
    for prefix, kind in rules.by_prefix.items():
        if document_type.startswith(prefix):
            return kind
    if not filename:
        return rules.file_type_default
    return rules.by_extension.get(_extension(filename), rules.extension_default)
