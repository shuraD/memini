"""Deterministic file parsers: registry, exact-span contract and built-in parsers.

Importing the package registers the built-in parsers.
"""

from memini.core.ingest.parsers import python_ast as _python_ast  # noqa: F401  (registers)
from memini.core.ingest.parsers.contract import (
    ContractError,
    ParsedEdge,
    ParsedEntity,
    Parser,
    ParseResult,
    check_invariants,
    line_offsets,
    make_entity,
)
from memini.core.ingest.parsers.registry import (
    LLM_PATH,
    MODE_ENTITIES,
    MODE_STRUCTURE_ONLY,
    extract,
    mode_for,
    parser_for,
    register,
    registered_parsers,
)

__all__ = [
    "LLM_PATH",
    "MODE_ENTITIES",
    "MODE_STRUCTURE_ONLY",
    "ContractError",
    "ParseResult",
    "ParsedEdge",
    "ParsedEntity",
    "Parser",
    "check_invariants",
    "extract",
    "line_offsets",
    "make_entity",
    "mode_for",
    "parser_for",
    "register",
    "registered_parsers",
]
