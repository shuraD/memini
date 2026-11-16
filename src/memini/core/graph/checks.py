"""Static check on Cypher text: every ``MATCH`` node pattern names a label.

An unlabeled ``MATCH`` scans every node of the graph (BL-39: about 20 times the db hits
of the labeled form). The check reads the statement clause by clause with regular
expressions; it is not a full Cypher parser (uppercase keywords, no comments, no
keyword-bearing string literals). Nothing is executed.
"""

from __future__ import annotations

import re

from memini.core.graph.types import UnlabeledMatchError

# Patterns are kept as strings; the ``re`` module caches their compiled form.
_CLAUSE_RE = (
    r"\b(OPTIONAL\s+MATCH|ON\s+CREATE\s+SET|ON\s+MATCH\s+SET|DETACH\s+DELETE|ORDER\s+BY"
    r"|MATCH|CREATE|MERGE|SET|REMOVE|WITH|WHERE|RETURN|UNWIND|DELETE|CALL|LIMIT|SKIP|UNION)\b"
)
_NODE_RE = r"\((?P<var>\w*)(?P<labels>(?::`?\w+`?)*)\s*(?P<props>\{[^}]*\})?\)"
_MATCH_KEYWORDS = ("MATCH", "OPTIONAL MATCH")


def _clauses(cypher: str) -> list[tuple[str, str]]:
    """Split ``cypher`` into ``(keyword, body)`` pairs, keywords whitespace-normalized."""
    marks = list(re.finditer(_CLAUSE_RE, cypher))
    result: list[tuple[str, str]] = []
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(cypher)
        result.append((" ".join(mark.group(1).split()), cypher[mark.end() : end]))
    return result


def require_labeled(cypher: str) -> None:
    """Raise ``UnlabeledMatchError`` listing every ``MATCH`` node pattern without a label.

    A pattern whose variable was bound with a label earlier in the statement is accepted.
    """
    labeled: set[str] = set()
    found: list[str] = []
    for keyword, body in _clauses(cypher):
        for node in re.finditer(_NODE_RE, body):
            var = node.group("var")
            if node.group("labels"):
                if var:
                    labeled.add(var)
            elif keyword in _MATCH_KEYWORDS and var not in labeled:
                found.append(node.group(0))
    if found:
        raise UnlabeledMatchError(f"MATCH pattern names no label: {', '.join(found)}")
