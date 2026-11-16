"""Module feature flags (owner decision 2026-11-14, ADR-0004).

One flag per unshipped engine module, never one per task. Each flag is read from its
``MEMINI_ENABLE_<MODULE>`` environment variable when it is called, and is OFF by default,
so ``main`` stays shippable while a module is built slice by slice.

Rules (private procedure ATOMIC-SLICE-WORKFLOW §7):
- Only the owner adds or flips a flag; this file is protected from coding agents.
- A flag gates only a module that is not on ``main`` yet; it never switches off existing logic.
- The gates run the tests twice: with these defaults and with every flag on.
- When a module ships, the owner flips its default and the module's last slice removes the check.

Every flag calls ``_env_bool`` with a literal variable name: the orchestrator finds the
variables by parsing those calls.
"""

from __future__ import annotations

import os

_TRUE = frozenset({"1", "true", "yes", "on"})


def _env_bool(name: str, default: bool = False) -> bool:
    """Return the flag in ``name``: a true word turns it on, anything else turns it off."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE


def document_authority() -> bool:
    """Return whether module 4, document-authority (lineage, concept ownership), is on."""
    return _env_bool("MEMINI_ENABLE_DOCUMENT_AUTHORITY")


def ingestion() -> bool:
    """Return whether module 3, ingestion (walk, categorize, chunk, parse, extract), is on."""
    return _env_bool("MEMINI_ENABLE_INGESTION")


def kg_extraction() -> bool:
    """Return whether module 5, kg-extraction (entities and relations), is on."""
    return _env_bool("MEMINI_ENABLE_KG_EXTRACTION")


def archrag_index() -> bool:
    """Return whether module 6, archrag-index (KNN, Leiden, summaries, C-HNSW), is on."""
    return _env_bool("MEMINI_ENABLE_ARCHRAG_INDEX")


def archrag_retrieval() -> bool:
    """Return whether module 7, archrag-retrieval (C-HNSW search, generation), is on."""
    return _env_bool("MEMINI_ENABLE_ARCHRAG_RETRIEVAL")
