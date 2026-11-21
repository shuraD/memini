r"""The skeleton document of a version and its checks (FLOW-HYBRID.md steps 3 and 5; no model call).

The document is markdown. Every table whose header is

    | File | Function | Signature | Cards | Top caller | Note |

is read, one row per function: the file under ``src/``, the function's qualified name
(``Class.method`` for a method), its ``def`` line (a ``|`` inside it is written ``\|``), the card
ids it serves (``-`` for none), the top caller of its flow as ``<dotted.module>:<function>`` (a
row that names itself is a top caller; several callers are comma separated), and a note. A row
with no card that is not a top caller is "extra": its note must say why it exists.

``check`` compares the rows with the cards of one version; ``compare`` compares them with code.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

HEADER = ("File", "Function", "Signature", "Cards", "Top caller", "Note")
ROOT = "src"
_CELL = re.compile(r"(?<!\\)\|")
_NONE = ("", "-")
_BODY = ast.dump(ast.parse("...").body[0])
_KINDS = {"property", "cached_property", "staticmethod", "classmethod"}  # change how it is called
_Def = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class Row:
    """One function of the skeleton document."""

    line: int
    file: str
    name: str
    signature: str
    cards: tuple[str, ...]
    callers: tuple[str, ...]
    note: str

    @property
    def target(self) -> str:
        """Return the row as a top caller would name it: ``<dotted.module>:<function>``."""
        parts = PurePosixPath(self.file).with_suffix("").parts
        parts = parts[1:] if parts[:1] == (ROOT,) else parts
        parts = parts[:-1] if parts[-1:] == ("__init__",) else parts
        return f"{'.'.join(parts)}:{self.name}"

    @property
    def top(self) -> bool:
        """Return True when the row names itself as its top caller."""
        return self.target in self.callers


@dataclass
class Report:
    """What ``check`` found. ``counts`` maps each squad to (cards, mapped)."""

    counts: dict[str, tuple[int, int]] = field(default_factory=dict)
    functions: int = 0
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return True when no card is missing and nothing is inconsistent."""
        return not self.missing and not self.problems


def _cells(line: str) -> list[str]:
    cells = [c.replace("\\|", "|").strip() for c in _CELL.split(line.strip())[1:-1]]
    return [c.strip("`").strip() for c in cells]


def _items(cell: str) -> tuple[str, ...]:
    return tuple(i.strip() for i in cell.split(",") if i.strip() not in _NONE)


def read(path: Path) -> tuple[list[Row], list[str]]:
    """Return the rows of the document, and the table rows that do not have its columns."""
    rows: list[Row] = []
    problems: list[str] = []
    inside = False
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue  # a blank line does not end a table: rows after it are never dropped
        cells = _cells(line) if line.lstrip().startswith("|") else []
        if [c.casefold() for c in cells] == [h.casefold() for h in HEADER] or not cells:
            inside = bool(cells)
        elif inside and not all(set(c) <= set("-: ") for c in cells):
            if len(cells) != len(HEADER):
                problems.append(f"line {n}: {len(cells)} cells, expected {len(HEADER)}")
                continue
            file, name, signature, cards, callers, note = cells
            note = "" if note in _NONE else note
            rows.append(Row(n, file, name, signature, _items(cards), _items(callers), note))
    return rows, problems


def _parse(signature: str) -> _Def | None:
    """Return the parsed ``def`` line, or None when it is not exactly one Python ``def``."""
    try:
        body = ast.parse(signature.strip().rstrip(":") + ": ...").body
    except SyntaxError:
        return None
    if len(body) != 1 or not isinstance(body[0], _Def):
        return None
    return body[0] if [ast.dump(b) for b in body[0].body] == [_BODY] else None


def _contract(node: _Def) -> str:
    returns = ast.dump(node.returns) if node.returns else ""
    kinds = sorted({ast.unparse(d).rpartition(".")[2] for d in node.decorator_list} & _KINDS)
    return f"{kinds} {type(node).__name__} {ast.dump(node.args)} -> {returns}"


def _row_problems(rows: Sequence[Row]) -> list[str]:
    """Return what is inconsistent inside the document itself."""
    found: list[str] = []
    targets = {r.target for r in rows}
    seen: set[tuple[str, str]] = set()
    for r in rows:
        where = f"line {r.line} {r.file} {r.name}"
        path = PurePosixPath(r.file)
        if path.parts[:1] != (ROOT,) or ".." in path.parts or path.suffix != ".py":
            found.append(f"{where}: the file is not a Python file under {ROOT}/")
        if (str(path), r.name) in seen:
            found.append(f"{where}: duplicate row")
        seen.add((str(path), r.name))
        if not all(p.isidentifier() for p in r.name.split(".")):
            found.append(f"{where}: the function is not a qualified name")
        node = _parse(r.signature)
        if node is None:
            found.append(f"{where}: the signature is not a Python def: {r.signature}")
        elif node.name != r.name.rpartition(".")[2]:
            found.append(f"{where}: the signature defines {node.name}")
        if not r.callers:
            found.append(f"{where}: no top caller")
        found += [
            f"{where}: top caller {c} is not a row of the document"
            for c in r.callers
            if c not in targets
        ]
        if not r.cards and not r.top and not r.note:
            found.append(f"{where}: no card and no reason in the note")
    return found


def check(
    rows: Sequence[Row], version: str, versions: Mapping[str, str], squads: Mapping[str, str]
) -> Report:
    """Compare the rows with the cards of ``version``.

    ``versions`` maps every card that is to be built to its version; ``squads`` maps a card to
    its squad (a card without one is counted under ``-``).
    """
    wanted = [c for c, v in versions.items() if v == version]
    if not wanted:
        known = ", ".join(sorted(set(versions.values())))
        msg = f"no card of version {version!r}; versions: {known}"
        raise ValueError(msg)
    report = Report(functions=len(rows), problems=_row_problems(rows))
    mapped = {c for r in rows for c in r.cards}
    for card in wanted:
        squad = squads.get(card, "-")
        cards, done = report.counts.get(squad, (0, 0))
        report.counts[squad] = (cards + 1, done + (card in mapped))
        if card not in mapped:
            report.missing.append(card)
    for r in rows:
        where = f"line {r.line} {r.file} {r.name}"
        for card in r.cards:
            if card not in versions:
                report.problems.append(f"{where}: card {card} is unknown or not built")
            elif versions[card] != version:
                report.problems.append(f"{where}: card {card} is of version {versions[card]}")
        if not r.cards and not r.top:
            report.extra.append(f"{r.file} {r.name}: {r.note or 'NO REASON'}")
    return report


def _defs(tree: ast.AST, prefix: str = "") -> dict[str, _Def]:
    """Return the functions of a module and the methods of its classes, by qualified name."""
    found: dict[str, _Def] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, _Def):
            found[prefix + node.name] = node
        elif isinstance(node, ast.ClassDef):
            found.update(_defs(node, f"{prefix}{node.name}."))
    return found


def compare(rows: Sequence[Row], source: Callable[[str], str | None]) -> list[str]:
    """Return where the code differs from the rows; ``source`` gives a file's text, or None.

    A difference is a file or function that is missing, a file that does not parse, a signature
    whose arguments, annotations, defaults or return annotation differ (text spacing does not
    count), a property or a static or class method (the document's ``def`` line carries no
    decorator, so it names a plain function), or a public function of a listed file that
    the document does not name.
    """
    found: list[str] = []
    for file in dict.fromkeys(r.file for r in rows):
        mine = {r.name: r for r in rows if r.file == file}
        text = source(file)
        if text is None:
            found.append(f"{file}: the file is missing ({len(mine)} functions)")
            continue
        try:
            defs = _defs(ast.parse(text))
        except SyntaxError as exc:
            found.append(f"{file}: the file does not parse: {exc}")
            continue
        for name, row in mine.items():
            want = _parse(row.signature)
            if name not in defs:
                found.append(f"{file} {name}: the function is missing")
            elif want is not None and _contract(want) != _contract(defs[name]):
                line = ast.get_source_segment(text, defs[name]) or ""
                found.append(
                    f"{file} {name}: the signature differs: document `{row.signature}`, "
                    f"code line {defs[name].lineno} `{line.splitlines()[0].strip()}`"
                )
        found += [
            f"{file} {name}: a public function the document does not name"
            for name in defs
            if name not in mine and not any(p.startswith("_") for p in name.split("."))
        ]
    return found


def render(report: Report) -> str:
    """Return the coverage table and the lists as text."""
    lines = ["| Squad | cards | mapped | missing |", "|---|---|---|---|"]
    counts = report.counts.values()
    total = (sum(c for c, _ in counts), sum(d for _, d in counts))
    rows = [*sorted(report.counts.items()), ("all", total)]
    lines += [f"| {s} | {cards} | {done} | {cards - done} |" for s, (cards, done) in rows]
    lines.append(f"functions: {report.functions}, extra (no card): {len(report.extra)}")
    for title, items in (
        ("missing", [", ".join(report.missing)] if report.missing else []),
        ("extra", report.extra),
        ("problems", report.problems),
    ):
        count = len(report.missing) if title == "missing" else len(items)
        lines += [f"{title} ({count}):", *(f"  {i}" for i in items)]
    return "\n".join(lines)
