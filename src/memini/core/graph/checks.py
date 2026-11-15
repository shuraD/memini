"""Static checks on Cypher text: labeled ``MATCH`` patterns and audited writes.

The checks read the statement clause by clause with regular expressions; they are
not a full Cypher parser (uppercase keywords, no comments, no keyword-bearing
string literals). Nothing is executed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from memini.core.graph.types import AuditFieldError, UnlabeledMatchError

# Patterns are kept as strings; the ``re`` module caches their compiled form.
_CLAUSE_RE = (
    r"\b(OPTIONAL\s+MATCH|ON\s+CREATE\s+SET|ON\s+MATCH\s+SET|DETACH\s+DELETE|ORDER\s+BY"
    r"|MATCH|CREATE|MERGE|SET|REMOVE|WITH|WHERE|RETURN|UNWIND|DELETE|CALL|LIMIT|SKIP|UNION)\b"
)
_NODE_RE = r"\((?P<var>\w*)(?P<labels>(?::`?\w+`?)*)\s*(?P<props>\{[^}]*\})?\)"
_REL_RE = r"\[(?P<var>\w*)(?P<types>(?::`?\w+`?(?:\|`?\w+`?)*)?)\s*(?P<props>\{[^}]*\})?\]"
_PROP_KEY_RE = r"(\w+)\s*:"
_SET_FIELD_RE = r"\b(\w+)\.(\w+)\s*="
_SET_VAR_RE = r"\b(\w+)(?:\.\w+|:\w+)?\s*(?:\+?=|:)"
_REMOVE_VAR_RE = r"\b(\w+)[.:]\w+"
_ALIAS_RE = r"\bAS\s+(\w+)"
_MATCH_KEYWORDS = ("MATCH", "OPTIONAL MATCH")
_SET_KEYWORDS = ("SET", "ON CREATE SET", "ON MATCH SET")


def _clauses(cypher: str) -> list[tuple[str, str]]:
    """Split ``cypher`` into ``(keyword, body)`` pairs, keywords whitespace-normalized."""
    marks = list(re.finditer(_CLAUSE_RE, cypher))
    result: list[tuple[str, str]] = []
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(cypher)
        result.append((" ".join(mark.group(1).split()), cypher[mark.end() : end]))
    return result


def _keys(props: str | None) -> set[str]:
    """Return the property keys of a pattern map such as ``{a: 1, b: $b}``."""
    return set(re.findall(_PROP_KEY_RE, props or ""))


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


@dataclass
class _Created:
    """An entity created by ``CREATE`` or ``MERGE`` with the keys of its pattern map."""

    kind: str
    name: str
    keys: set[str]
    merged: bool


@dataclass
class _Scan:
    """What a statement binds, creates and assigns, gathered clause by clause."""

    bound: set[str] = field(default_factory=set)
    matched: set[str] = field(default_factory=set)
    touched: list[str] = field(default_factory=list)
    created: list[_Created] = field(default_factory=list)
    sets: dict[str, dict[str, set[str]]] = field(
        default_factory=lambda: {keyword: {} for keyword in _SET_KEYWORDS}
    )

    def read(self, keyword: str, body: str) -> None:
        """Record one clause."""
        if keyword in _MATCH_KEYWORDS:
            self.matched.update(m.group("var") for m in re.finditer(_NODE_RE, body))
            self.bound.update(self.matched)
            self.bound.update(m.group("var") for m in re.finditer(_REL_RE, body))
        elif keyword in ("CREATE", "MERGE"):
            self._create(body, merged=keyword == "MERGE")
        elif keyword in self.sets:
            for var, name in re.findall(_SET_FIELD_RE, body):
                self.sets[keyword].setdefault(var, set()).add(name)
            self.touched.extend(re.findall(_SET_VAR_RE, body))
        elif keyword == "REMOVE":
            self.touched.extend(re.findall(_REMOVE_VAR_RE, body))
        self.bound.update(re.findall(_ALIAS_RE, body))

    def _create(self, body: str, *, merged: bool) -> None:
        """Record the new nodes and every relationship of a ``CREATE``/``MERGE`` pattern."""
        new: list[str] = []
        for kind, pattern in (("node", _NODE_RE), ("relationship", _REL_RE)):
            for m in re.finditer(pattern, body):
                var = m.group("var")
                new.append(var)
                if kind == "relationship" or not var or var not in self.bound:
                    keys = _keys(m.group("props"))
                    self.created.append(_Created(kind, var or m.group(0), keys, merged))
        self.bound.update(new)

    def has(self, keyword: str, var: str, name: str) -> bool:
        """Tell whether ``var.name`` is assigned by a clause of kind ``keyword``."""
        return name in self.sets[keyword].get(var, set())

    def updated(self, var: str, name: str) -> bool:
        """Tell whether ``var.name`` is set on every write (SET, or both ON branches)."""
        return self.has("SET", var, name) or (
            self.has("ON CREATE SET", var, name) and self.has("ON MATCH SET", var, name)
        )

    def set_once(self, item: _Created, name: str) -> bool:
        """Tell whether a created entity gets ``name`` at creation."""
        return (
            name in item.keys
            or self.has("ON CREATE SET", item.name, name)
            or (not item.merged and self.has("SET", item.name, name))
        )


def audit_write_problems(
    cypher: str,
    *,
    node_fields: Sequence[str] = ("created_at", "updated_at", "updated_by"),
    rel_fields: Sequence[str] = ("created_at",),
) -> list[str]:
    """Return one problem per missing ``(variable, field)`` audit pair of a write statement.

    The first of ``node_fields`` is the creation field, set once when a node is created
    (pattern map or ``ON CREATE SET``; a plain ``SET`` too after ``CREATE``); the others
    must be set on every write, including ``SET``/``REMOVE`` on a ``MATCH``-bound node.
    Every ``rel_fields`` entry is a creation field of a created relationship.
    """
    scan = _Scan()
    for keyword, body in _clauses(cypher):
        scan.read(keyword, body)
    creation, updates = list(node_fields[:1]), list(node_fields[1:])
    problems: list[str] = []
    for item in scan.created:
        once = rel_fields if item.kind == "relationship" else creation
        problems.extend(
            f"{item.kind} {item.name} does not set {name} at creation"
            for name in once
            if not scan.set_once(item, name)
        )
        if item.kind == "node":
            problems.extend(
                f"node {item.name} does not set {name} on every write"
                for name in updates
                if name not in item.keys and not scan.updated(item.name, name)
            )
    for var in dict.fromkeys(v for v in scan.touched if v and v in scan.matched):
        problems.extend(
            f"node {var} is written without setting {name}"
            for name in updates
            if not scan.updated(var, name)
        )
    return problems


def require_audited(
    cypher: str,
    *,
    node_fields: Sequence[str] = ("created_at", "updated_at", "updated_by"),
    rel_fields: Sequence[str] = ("created_at",),
) -> None:
    """Raise ``AuditFieldError`` listing every problem of ``audit_write_problems``."""
    problems = audit_write_problems(cypher, node_fields=node_fields, rel_fields=rel_fields)
    if problems:
        raise AuditFieldError("; ".join(problems))
