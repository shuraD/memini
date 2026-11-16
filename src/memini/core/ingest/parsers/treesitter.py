"""One generic tree-sitter adapter: a parsed syntax tree becomes exact-span entities.

Every format behaviour (which node types become entities, how they are named, depth and
count limits, error handling) comes from the ``structure.tree_sitter`` config mapping;
the code holds no knowledge of any grammar. The tree is injected (``build`` takes a root
node, ``parse`` takes a parser factory), so no grammar has to be installed here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Protocol

DEFAULT_DOCUMENT_KIND = "Document"
DEFAULT_NAME_MAX_CHARS = 120
DEFAULT_ERRORS_LISTED_MAX = 20
ID_NAME_CHARS = 60
ON_ERROR_FALLBACK = "fallback"
ERROR_NODE_TYPE = "ERROR"
_PATH_SEP = ">"
_EDGE_TRIM = " \t"


class Node(Protocol):
    """The shape of a tree-sitter node the adapter reads."""

    type: str
    start_byte: int
    end_byte: int
    start_point: tuple[int, int]
    children: Sequence[Node]
    has_error: bool
    is_missing: bool

    def child_by_field_name(self, name: str) -> Node | None:
        """Return the child stored under the grammar field ``name``, if any."""
        ...


class GrammarNotInstalled(RuntimeError):  # noqa: N818 - name fixed by the IN-07 interface
    """Raised by a ``parser_for`` factory that has no grammar for the language."""


@dataclass(frozen=True)
class TreeEntity:
    """One entity with an exact character span of the content."""

    id: str
    kind: str
    name: str
    node_type: str
    start: int
    end: int
    code: str
    line_start: int
    parent: str | None
    depth: int


@dataclass(frozen=True)
class TreeResult:
    """Entities (Document first, then pre-order), containment edges and parse errors."""

    entities: tuple[TreeEntity, ...]
    edges: tuple[tuple[str, str], ...]
    errors: tuple[str, ...]
    depth_used: int


@dataclass
class _Item:
    """A mapped node found in the tree, before filtering and id assignment."""

    node: Node
    parent: int | None
    depth: int
    nested: bool = False


def language_for(path: str, languages: Mapping[str, str]) -> str | None:
    """Return the grammar of the longest matching extension of ``path`` (case-insensitive)."""
    lowered = path.lower()
    for ext in sorted(languages, key=len, reverse=True):
        if ext and lowered.endswith(ext.lower()):
            return languages[ext]
    return None


def _first_line(text: str) -> str:
    """Return the first non-blank line of ``text``, or ``""``."""
    for line in text.splitlines():
        if line.strip():
            return line
    return ""


def _rule_of(raw: Any) -> Mapping[str, Any]:
    """Normalise a node_kinds rule: a plain string is the kind."""
    if isinstance(raw, str):
        return {"kind": raw}
    return raw if isinstance(raw, Mapping) else {}


class _Builder:
    """Reads one tree against one grammar config."""

    def __init__(self, content: str, node_kinds: Mapping[str, Any], name_max: int) -> None:
        self._content = content
        self._kinds = node_kinds
        self._name_max = name_max
        chars: list[int] = []
        for index, char in enumerate(content):
            chars.extend([index] * len(char.encode("utf-8")))
        self._size = len(chars)
        chars.append(len(content))
        self._byte_to_char = chars

    def char(self, byte: int) -> int:
        """Convert a UTF-8 byte offset into a character offset."""
        return self._byte_to_char[max(0, min(byte, self._size))]

    def span(self, node: Node) -> tuple[int, int]:
        """Return the character span of ``node``."""
        return self.char(node.start_byte), self.char(node.end_byte)

    def text(self, node: Node) -> str:
        """Return the exact source text of ``node``."""
        start, end = self.span(node)
        return self._content[start:end]

    def mapped(self, node: Node) -> bool:
        """Tell whether the node type has a rule in node_kinds."""
        return node.type in self._kinds

    def kind(self, node: Node) -> str:
        """Return the configured kind of a mapped node (its type when the rule has none)."""
        return str(_rule_of(self._kinds[node.type]).get("kind") or node.type)

    def name(self, node: Node) -> str:
        """Return the cleaned name from the first name source yielding text, or ``""``."""
        rule = _rule_of(self._kinds[node.type])
        trim = str(rule.get("strip") or "") + _EDGE_TRIM
        for source in rule.get("name") or ():
            cleaned = " ".join(self._source_text(node, str(source)).split())
            cleaned = cleaned.strip(trim)[: self._name_max]
            if cleaned:
                return cleaned
        return ""

    def _source_text(self, node: Node, source: str) -> str:
        """Return the raw text one name source yields for ``node``."""
        if source == "text":
            return _first_line(self.text(node))
        prefix, _, arg = source.partition(":")
        target: Node | None = None
        if prefix == "field":
            target = node.child_by_field_name(arg)
        elif prefix == "child":
            target = node
            for wanted in arg.split(_PATH_SEP):
                target = next((c for c in target.children if c.type == wanted), None)
                if target is None:
                    break
        if target is None:
            return ""
        if self.mapped(target):
            return self.name(target)
        return _first_line(self.text(target))

    def collect(self, root: Node) -> list[_Item]:
        """Return every mapped node in pre-order with its mapped parent and depth."""
        items: list[_Item] = []
        stack: list[tuple[Node, int | None, int]] = [(root, None, 0)]
        while stack:
            node, parent, depth = stack.pop()
            if self.mapped(node):
                if parent is not None:
                    items[parent].nested = True
                depth += 1
                items.append(_Item(node, parent, depth))
                parent = len(items) - 1
            stack.extend((child, parent, depth) for child in reversed(node.children))
        return items


def _errors(root: Node, language: str, listed_max: int) -> tuple[str, ...]:
    """List ERROR and missing nodes in pre-order, not descending into ERROR nodes."""
    found: list[str] = []
    stack = [root]
    while stack:
        node = stack.pop()
        row, col = node.start_point
        where = f"at L{row + 1}:{col + 1}"
        if node.type == ERROR_NODE_TYPE:
            found.append(f"{language}: ERROR {where}")
            continue
        if node.is_missing:
            found.append(f"{language}: MISSING {node.type} {where}")
        stack.extend(c for c in reversed(node.children) if c.has_error)
    if len(found) > listed_max:
        extra = len(found) - listed_max
        found = [*found[:listed_max], f"{language}: … {extra} more error node(s)"]
    return tuple(found)


def _keep(items: list[_Item], grammar: Mapping[str, Any]) -> tuple[list[int], int]:
    """Apply nested_requires_children, max_depth and max_entities; return kept indexes and depth."""
    max_depth = grammar.get("max_depth")
    max_entities = grammar.get("max_entities")
    requires_children = bool(grammar.get("nested_requires_children", False))
    kept = [
        index
        for index, item in enumerate(items)
        if (max_depth is None or item.depth <= int(max_depth))
        and (not requires_children or item.depth == 1 or item.nested)
    ]
    depth_used = max((items[i].depth for i in kept), default=0)
    if max_entities is not None:
        while depth_used > 1 and len(kept) + 1 > int(max_entities):
            kept = [i for i in kept if items[i].depth < depth_used]
            depth_used -= 1
    return kept, depth_used


def _unique(base: str, seen: dict[str, int]) -> str:
    """Return ``base``, or ``base#n`` for its n-th occurrence."""
    count = seen.get(base, 0) + 1
    seen[base] = count
    return base if count == 1 else f"{base}#{count}"


def build(
    content: str, path: str, root: Node, language: str, config: Mapping[str, Any]
) -> TreeResult:
    """Turn the syntax tree ``root`` of ``content`` into a Document plus mapped entities."""
    grammar: Mapping[str, Any] = (config.get("grammars") or {}).get(language) or {}
    listed_max = int(config.get("errors_listed_max", DEFAULT_ERRORS_LISTED_MAX))
    errors = _errors(root, language, listed_max) if root.has_error else ()
    if errors and grammar.get("on_error") == ON_ERROR_FALLBACK:
        return TreeResult((), (), errors, 0)

    builder = _Builder(
        content,
        grammar.get("node_kinds") or {},
        int(config.get("name_max_chars", DEFAULT_NAME_MAX_CHARS)),
    )
    document_kind = str(config.get("document_kind") or DEFAULT_DOCUMENT_KIND)
    document_name = PurePosixPath(path.replace("\\", "/")).stem
    seen: dict[str, int] = {}
    document_id = _unique(f"{document_kind.lower()}:{document_name}", seen)
    entities = [
        TreeEntity(
            document_id, document_kind, document_name, "", 0, len(content), content, 1, None, 0
        )
    ]

    items = builder.collect(root)
    kept, depth_used = _keep(items, grammar)
    ids: dict[int, str] = {}
    for index in kept:
        item = items[index]
        kind = builder.kind(item.node)
        name = builder.name(item.node)
        line = item.node.start_point[0] + 1
        if name:
            base = f"{kind.lower()}:{name[:ID_NAME_CHARS]}@L{line}"
        else:
            name, base = f"{kind}@L{line}", f"{kind.lower()}@L{line}"
        ids[index] = _unique(base, seen)
        start, end = builder.span(item.node)
        parent = document_id if item.parent is None else ids[item.parent]
        entities.append(
            TreeEntity(
                ids[index],
                kind,
                name,
                item.node.type,
                start,
                end,
                content[start:end],
                line,
                parent,
                item.depth,
            )
        )
    edges = tuple((e.parent, e.id) for e in entities[1:] if e.parent is not None)
    return TreeResult(tuple(entities), edges, errors, depth_used)


def parse(
    content: str, path: str, config: Mapping[str, Any], parser_for: Callable[[str], Any]
) -> TreeResult:
    """Parse ``content`` with the grammar mapped to ``path`` and build its entities.

    ``parser_for(language)`` returns an object whose ``parse(bytes)`` yields a tree with a
    ``root_node``; a ``GrammarNotInstalled`` it raises propagates.
    """
    language = language_for(path, config.get("languages") or {})
    if language is None:
        return TreeResult((), (), (f"no tree-sitter language configured for {path}",), 0)
    if not content.strip():
        return TreeResult((), (), (f"{language}: blank content in {path}",), 0)
    tree = parser_for(language).parse(content.encode("utf-8"))
    return build(content, path, tree.root_node, language, config)
