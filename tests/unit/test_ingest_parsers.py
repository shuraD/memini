"""Acceptance tests for IN-06: parser registry, exact-span contract, python_ast parser."""

import copy

import pytest

from memini.core.ingest.parsers import (
    LLM_PATH,
    MODE_ENTITIES,
    MODE_STRUCTURE_ONLY,
    ContractError,
    ParsedEdge,
    ParsedEntity,
    ParseResult,
    check_invariants,
    extract,
    line_offsets,
    make_entity,
    mode_for,
    parser_for,
    register,
    registered_parsers,
)


def _line_of(content: str, offset: int) -> int:
    return content.count("\n", 0, offset) + 1


def _python(content: str, path: str = "pkg/mod.py") -> ParseResult:
    return registered_parsers()["python_ast"](content, path)


def _one(result: ParseResult, kind: str, name: str) -> ParsedEntity:
    found = [e for e in result.entities if e.kind == kind and e.name == name]
    assert len(found) == 1, (kind, name, [(e.kind, e.name) for e in result.entities])
    return found[0]


def _entity(content: str, **over: object) -> ParsedEntity:
    fields: dict[str, object] = {
        "id": "e1",
        "kind": "Function",
        "name": "f",
        "start": 0,
        "end": 2,
        "line_start": 1,
        "line_end": 1,
        "code": content[0:2],
    }
    fields.update(over)
    return ParsedEntity(**fields)  # type: ignore[arg-type]


# --- registry -------------------------------------------------------------


def test_python_ast_registered_on_import() -> None:
    assert "python_ast" in registered_parsers()


def test_register_decorator_and_conflicts() -> None:
    def parse_x(content: str, path: str) -> ParseResult:
        return ParseResult(parser="in06_x")

    def other_x(content: str, path: str) -> ParseResult:
        return ParseResult(parser="in06_x")

    returned = register("in06_x")(parse_x)
    assert returned is parse_x
    assert registered_parsers()["in06_x"] is parse_x
    register("in06_x")(parse_x)  # same function again: no-op
    assert registered_parsers()["in06_x"] is parse_x
    with pytest.raises(ValueError):
        register("in06_x")(other_x)
    assert registered_parsers()["in06_x"] is parse_x


def test_registered_parsers_returns_copy() -> None:
    first = registered_parsers()
    first.pop("python_ast")
    first["in06_bogus"] = lambda content, path: ParseResult(parser="bogus")
    again = registered_parsers()
    assert "python_ast" in again
    assert "in06_bogus" not in again


# --- parser_for / mode_for ---------------------------------------------------


def test_parser_for_longest_extension_wins() -> None:
    extractors = {".ts": "a", ".d.ts": "b"}
    assert parser_for("x/y.d.ts", extractors) == "b"
    assert parser_for("y.ts", extractors) == "a"


def test_parser_for_case_insensitive_and_missing() -> None:
    assert parser_for("A/B.PY", {".py": "p"}) == "p"
    assert parser_for("a/b.py", {".PY": "p"}) == "p"
    assert parser_for("notes.txt", {".py": "p"}) is None


def test_parser_for_llm_path_or_empty_is_none() -> None:
    assert parser_for("a.md", {".md": LLM_PATH}) is None
    assert parser_for("a.md", {".md": ""}) is None


def test_mode_constants_distinct() -> None:
    assert MODE_ENTITIES != MODE_STRUCTURE_ONLY


def test_mode_for() -> None:
    modes = {".md": MODE_STRUCTURE_ONLY}
    assert mode_for("doc.MD", modes) == MODE_STRUCTURE_ONLY
    assert mode_for("a.py", modes) == MODE_ENTITIES
    assert mode_for("a.py", {".py": MODE_ENTITIES}) == MODE_ENTITIES


def test_mode_for_invalid_value_raises() -> None:
    bad = "in06-not-a-mode"
    assert bad not in (MODE_ENTITIES, MODE_STRUCTURE_ONLY)
    with pytest.raises(ValueError):
        mode_for("doc.md", {".md": bad})


# --- extract -------------------------------------------------------------------


def test_extract_without_parser_returns_none_and_calls_nothing() -> None:
    calls: list[tuple[str, str]] = []

    def spy(content: str, path: str) -> ParseResult:
        calls.append((content, path))
        return ParseResult(parser="in06_spy")

    register("in06_spy")(spy)
    extractors = {".zz": "in06_spy", ".md": LLM_PATH}
    assert extract("text", "notes.txt", extractors) is None
    assert extract("text", "notes.md", extractors) is None
    assert calls == []


def test_extract_unknown_parser_name_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        extract("x", "a.qq", {".qq": "in06-no-such-parser"})


def test_extract_contract_violation_raises() -> None:
    def broken(content: str, path: str) -> ParseResult:
        ent = ParsedEntity(
            id="m",
            kind="Module",
            name="m",
            start=0,
            end=len(content),
            line_start=1,
            line_end=1,
            code="not the content",
        )
        return ParseResult(parser="in06_broken", entities=[ent])

    register("in06_broken")(broken)
    assert issubclass(ContractError, ValueError)
    with pytest.raises(ContractError):
        extract("abc", "f.brk", {".brk": "in06_broken"})


def test_extract_valid_result_returned_unchanged() -> None:
    content = "ab\ncd\n"
    ent = ParsedEntity(
        id="m",
        kind="Module",
        name="m",
        start=0,
        end=len(content),
        line_start=1,
        line_end=2,
        code=content,
    )
    expected = ParseResult(parser="in06_good", entities=[ent], edges=[])

    def good(c: str, path: str) -> ParseResult:
        return copy.deepcopy(expected)

    register("in06_good")(good)
    got = extract(content, "f.gd", {".gd": "in06_good"})
    assert got == expected


def test_extract_python_syntax_error_does_not_raise() -> None:
    result = extract("def broken(:\n", "a/b.py", {".py": "python_ast"})
    assert result is not None
    assert result.parser == "python_ast"
    assert result.entities == []
    assert len(result.errors) >= 1
    assert all(isinstance(e, str) for e in result.errors)


# --- line_offsets / make_entity ---------------------------------------------------


def test_line_offsets() -> None:
    assert line_offsets("ab\ncd\n") == [0, 3, 6]
    assert line_offsets("ab\ncd")[:2] == [0, 3]


def test_make_entity_basic_span() -> None:
    content = "ab\ncd\nef"
    offs = line_offsets(content)
    ent = make_entity(
        content,
        offs,
        id="x1",
        kind="Function",
        name="cd",
        start=3,
        end=8,
        signature="sig()",
        parent="p0",
    )
    assert ent.id == "x1"
    assert ent.kind == "Function"
    assert ent.name == "cd"
    assert ent.start == 3
    assert ent.end == 8
    assert ent.code == "cd\nef"
    assert ent.line_start == 2
    assert ent.line_end == 3
    assert ent.signature == "sig()"
    assert ent.parent == "p0"
    assert check_invariants(content, ParseResult(parser="t", entities=[ent])) == []


def test_make_entity_end_on_newline_uses_line_of_last_char() -> None:
    content = "ab\ncd\n"
    ent = make_entity(content, line_offsets(content), id="a", kind="K", name="n", start=0, end=3)
    assert ent.code == "ab\n"
    assert ent.line_start == 1
    assert ent.line_end == 1
    assert ent.signature == ""
    assert ent.parent is None


def test_make_entity_empty_span_and_clamp() -> None:
    content = "ab\ncd\nef"
    offs = line_offsets(content)
    empty = make_entity(content, offs, id="e", kind="K", name="n", start=4, end=4)
    assert empty.code == ""
    assert empty.line_start == 2
    assert empty.line_end == 2
    clamped = make_entity(content, offs, id="c", kind="K", name="n", start=6, end=100)
    assert clamped.end == len(content)
    assert clamped.code == "ef"
    assert clamped.line_start == 3
    assert clamped.line_end == 3
    result = ParseResult(parser="t", entities=[empty, clamped])
    assert check_invariants(content, result) == []


def test_make_entity_non_ascii() -> None:
    content = "é😀\n中文x\n"
    offs = line_offsets(content)
    start = content.index("中")
    ent = make_entity(content, offs, id="u", kind="K", name="n", start=start, end=start + 3)
    assert ent.code == "中文x"
    assert ent.line_start == 2
    assert ent.line_end == 2


# --- check_invariants ----------------------------------------------------------------


def test_check_invariants_ok() -> None:
    content = "ab\ncd\n"
    result = ParseResult(parser="t", entities=[_entity(content)])
    assert check_invariants(content, result) == []


def test_check_invariants_duplicate_id() -> None:
    content = "ab\ncd\n"
    result = ParseResult(parser="t", entities=[_entity(content), _entity(content)])
    assert check_invariants(content, result) != []


@pytest.mark.parametrize(
    "over",
    [
        {"start": -1, "code": ""},
        {"start": 3, "end": 2, "code": ""},
        {"start": 0, "end": 99, "code": "ab\ncd\n"},
        {"code": "xx"},
        {"start": 3, "end": 5, "code": "cd", "line_start": 1, "line_end": 2},
        {"start": 3, "end": 5, "code": "cd", "line_start": 2, "line_end": 1},
        {"line_start": 1, "line_end": 0},
    ],
)
def test_check_invariants_violations(over: dict[str, object]) -> None:
    content = "ab\ncd\n"
    result = ParseResult(parser="t", entities=[_entity(content, **over)])
    problems = check_invariants(content, result)
    assert isinstance(problems, list)
    assert problems != []
    assert all(isinstance(p, str) for p in problems)


# --- python_ast parser ------------------------------------------------------------------

SOURCE = '''"""Module doc."""
import os


class Alpha:
    x = 1

    def run(self):
        return 1

    async def stop(self):
        return 2


class Beta:
    def run(self):
        return 3


def outer(a):
    def inner(b):
        return b

    return inner(a)


async def fetch():
    return os.sep
'''


def test_python_module_entity() -> None:
    result = _python(SOURCE)
    assert result.parser == "python_ast"
    assert result.errors == []
    mod = result.entities[0]
    assert mod.kind == "Module"
    assert mod.name == "mod"
    assert mod.start == 0
    assert mod.end == len(SOURCE)
    assert mod.code == SOURCE
    assert mod.parent is None


def test_python_kinds_and_parents() -> None:
    result = _python(SOURCE)
    mod = result.entities[0]
    alpha = _one(result, "Class", "Alpha")
    beta = _one(result, "Class", "Beta")
    outer = _one(result, "Function", "outer")
    inner = _one(result, "Function", "inner")
    fetch = _one(result, "Function", "fetch")
    assert alpha.parent == mod.id
    assert beta.parent == mod.id
    assert outer.parent == mod.id
    assert fetch.parent == mod.id
    assert inner.parent == outer.id
    runs = [e for e in result.entities if e.kind == "Method" and e.name == "run"]
    assert len(runs) == 2
    assert {r.parent for r in runs} == {alpha.id, beta.id}
    stop = _one(result, "Method", "stop")
    assert stop.parent == alpha.id
    assert fetch.code.startswith("async def fetch")
    assert fetch.code.endswith("return os.sep")
    assert alpha.code.startswith("class Alpha:")
    assert alpha.code.endswith("return 2")
    assert check_invariants(SOURCE, result) == []
    ids = [e.id for e in result.entities]
    assert len(ids) == len(set(ids))


def test_python_defines_edges() -> None:
    result = _python(SOURCE)
    mod = result.entities[0]
    defines = {(e.src, e.dst) for e in result.edges if e.type == "DEFINES"}
    for ent in result.entities:
        if ent.parent is not None:
            assert (ent.parent, ent.id) in defines
    assert all(dst != mod.id for _, dst in defines)
    assert all(isinstance(e, ParsedEdge) for e in result.edges)


def test_python_decorated_span() -> None:
    content = (
        "import functools\n"
        "\n"
        "\n"
        "@functools.lru_cache(maxsize=None)\n"
        "@staticmethod\n"
        "def cached(x):\n"
        "    y = x + 1\n"
        "    return y\n"
        "\n"
        "\n"
        "@dataclass\n"
        "class Point:\n"
        "    a: int = 0\n"
        "\n"
        "    @property\n"
        "    def double(self):\n"
        "        return self.a * 2\n"
        "\n"
        "\n"
        "print(1)\n"
    )
    result = _python(content, "pkg/deco.py")
    assert result.errors == []
    cached = _one(result, "Function", "cached")
    expected_fn = (
        "@functools.lru_cache(maxsize=None)\n"
        "@staticmethod\n"
        "def cached(x):\n"
        "    y = x + 1\n"
        "    return y"
    )
    assert cached.code == expected_fn
    assert cached.start == content.index("@functools")
    assert cached.end == cached.start + len(expected_fn)
    assert cached.line_start == 4
    assert cached.line_end == 8
    point = _one(result, "Class", "Point")
    expected_cls = (
        "@dataclass\nclass Point:\n    a: int = 0\n\n"
        "    @property\n    def double(self):\n        return self.a * 2"
    )
    assert point.code == expected_cls
    assert point.start == content.index("@dataclass")
    double = _one(result, "Method", "double")
    assert double.code == "@property\n    def double(self):\n        return self.a * 2"
    assert double.parent == point.id
    assert check_invariants(content, result) == []


def test_python_non_ascii_spans() -> None:
    content = (
        "# -*- coding: utf-8 -*-\n"
        'GREETING = "héllo 中文 😀"\n'
        "\n"
        "\n"
        "def café(x):\n"
        '    s = "ü—😀"\n'
        "    return s + x\n"
        "\n"
        "\n"
        "class Ñandú:\n"
        "    def mé(self):\n"
        '        return "日本😀"\n'
    )
    result = _python(content, "pkg/uni.py")
    assert result.errors == []
    cafe = _one(result, "Function", "café")
    assert cafe.code == 'def café(x):\n    s = "ü—😀"\n    return s + x'
    cls = _one(result, "Class", "Ñandú")
    assert cls.code == 'class Ñandú:\n    def mé(self):\n        return "日本😀"'
    me = _one(result, "Method", "mé")
    assert me.code == 'def mé(self):\n        return "日本😀"'
    for ent in result.entities:
        assert ent.code == content[ent.start : ent.end]
        assert ent.line_start == _line_of(content, ent.start)
        last = ent.end - 1 if ent.end > ent.start else ent.start
        assert ent.line_end == _line_of(content, last)
    assert cafe.line_start == 5
    assert cafe.line_end == 7
    assert check_invariants(content, result) == []


def test_python_syntax_error() -> None:
    result = _python("class X(:\n    pass\n", "pkg/bad.py")
    assert result.parser == "python_ast"
    assert result.entities == []
    assert len(result.errors) >= 1
    assert all(isinstance(e, str) and e for e in result.errors)


def test_python_empty_file() -> None:
    result = _python("", "pkg/empty.py")
    assert result.errors == []
    assert len(result.entities) == 1
    mod = result.entities[0]
    assert mod.kind == "Module"
    assert mod.name == "empty"
    assert mod.code == ""
    assert mod.start == 0
    assert mod.end == 0
    assert result.edges == []
    assert check_invariants("", result) == []


@pytest.mark.parametrize(
    "content",
    [
        SOURCE,
        "x = 1\n",
        "def f():\n    pass",
        "class A:\n    def m(self): pass\nclass B:\n    def m(self): pass\n",
        "\r\n\r\ndef g():\r\n    return 1\r\n",
        "if True:\n    def h():\n        return 1\n",
        "class C:\n    class D:\n        def m(self):\n            return 1\n",
        "def f():\n    pass\n\n\ndef f():\n    return 2\n",
    ],
)
def test_python_results_satisfy_contract(content: str) -> None:
    result = _python(content, "pkg/any.py")
    assert result.errors == []
    assert check_invariants(content, result) == []
    assert result.entities[0].kind == "Module"
    assert extract(content, "pkg/any.py", {".py": "python_ast"}) == result


def test_python_deterministic() -> None:
    assert _python(SOURCE) == _python(SOURCE)
    first = extract(SOURCE, "pkg/mod.py", {".py": "python_ast"})
    second = extract(SOURCE, "pkg/mod.py", {".py": "python_ast"})
    assert first == second
