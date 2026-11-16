"""LLM provider layer: public types from :mod:`.types` and the role registry."""

from __future__ import annotations

from memini.infra.llm_providers.registry import (
    configure_for_tests,
    embedder,
    export_jsonl,
    llm_for,
    question,
    reset_for_tests,
    summary,
)
from memini.infra.llm_providers.types import (
    LLM,
    ROLES,
    Attempt,
    BudgetExceeded,
    Embedder,
    Generation,
    LLMError,
    Reason,
)

__all__ = [
    "LLM",
    "ROLES",
    "Attempt",
    "BudgetExceeded",
    "Embedder",
    "Generation",
    "LLMError",
    "Reason",
    "configure_for_tests",
    "embedder",
    "export_jsonl",
    "llm_for",
    "question",
    "reset_for_tests",
    "summary",
]
