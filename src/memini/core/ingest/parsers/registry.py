"""Registry of deterministic parsers, chosen by longest matching file extension.

The extension-to-parser and extension-to-mode maps are supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from memini.core.ingest.parsers.contract import (
    ContractError,
    Parser,
    ParseResult,
    check_invariants,
)

LLM_PATH = "llm"
"""Extractor-map value meaning: no parser, use the LLM path."""

MODE_ENTITIES = "entities"
"""Mode value: extract entities (the default when an extension has no mode)."""

MODE_STRUCTURE_ONLY = "structure_only"
"""Mode value: keep only the structure."""

_MODES = (MODE_ENTITIES, MODE_STRUCTURE_ONLY)

_REGISTRY: dict[str, Parser] = {}


def register(name: str) -> Callable[[Parser], Parser]:
    """Return a decorator registering a parser under ``name``.

    Registering the same function again is a no-op; a different function under a
    taken name raises ``ValueError``.
    """

    def decorate(parser: Parser) -> Parser:
        current = _REGISTRY.get(name)
        if current is not None and current is not parser:
            raise ValueError(f"parser name {name!r} is already registered")
        _REGISTRY[name] = parser
        return parser

    return decorate


def registered_parsers() -> dict[str, Parser]:
    """Return a copy of the registry, by name."""
    return dict(_REGISTRY)


def _lookup(path: str, table: Mapping[str, str]) -> str | None:
    """Return the value of the longest key that ends ``path``, ignoring case."""
    lowered = path.lower()
    matches = [key for key in table if key and lowered.endswith(key.lower())]
    if not matches:
        return None
    return table[min(matches, key=lambda key: (-len(key), key))]


def parser_for(path: str, extractors: Mapping[str, str]) -> str | None:
    """Return the parser name for ``path``, or ``None`` for no parser or the LLM path."""
    name = _lookup(path, extractors)
    if not name or name == LLM_PATH:
        return None
    return name


def mode_for(path: str, modes: Mapping[str, str]) -> str:
    """Return the mode for ``path``; ``MODE_ENTITIES`` when no extension matches."""
    mode = _lookup(path, modes)
    if mode is None:
        return MODE_ENTITIES
    if mode not in _MODES:
        raise ValueError(f"unknown mode {mode!r} for {path!r}")
    return mode


def extract(content: str, path: str, extractors: Mapping[str, str]) -> ParseResult | None:
    """Run the parser chosen for ``path`` and check its result against the contract.

    Returns ``None`` when no parser applies. Raises ``KeyError`` for an unregistered
    parser name and ``ContractError`` for a result that breaks the contract.
    """
    name = parser_for(path, extractors)
    if name is None:
        return None
    if name not in _REGISTRY:
        raise KeyError(f"no parser registered under {name!r}")
    result = _REGISTRY[name](content, path)
    problems = check_invariants(content, result)
    if problems:
        raise ContractError(f"parser {name!r} on {path!r}: " + "; ".join(problems))
    return result
