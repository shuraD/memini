"""Acceptance tests for IN-07: the generic, config-driven tree-sitter adapter.

The syntax tree is injected, so every test uses fake nodes shaped like tree-sitter nodes.
"""

import pytest

from memini.core.ingest.parsers.treesitter import (
    GrammarNotInstalled,
    TreeEntity,
    TreeResult,
    build,
    language_for,
    parse,
)


class FakeNode:
    """A tree-sitter-shaped node built from char offsets into `content`."""

    def __init__(self, content, type_, start, end, *, children=(), fields=None, missing=False):
        self.type = type_
        self.start_byte = len(content[:start].encode("utf-8"))
        self.end_byte = len(content[:end].encode("utf-8"))
        prefix = content[:start]
        line_begin = prefix.rfind("\n") + 1
        self.start_point = (prefix.count("\n"), len(content[line_begin:start].encode("utf-8")))
        self.children = list(children)
        self.is_missing = missing
        self.has_error = type_ == "ERROR" or missing or any(c.has_error for c in self.children)
        self._fields = dict(fields or {})

    def child_by_field_name(self, name):
        return self._fields.get(name)


def maker(content):
    def node(type_, start, end, *children, fields=None, missing=False):
        return FakeNode(
            content, type_, start, end, children=children, fields=fields, missing=missing
        )

    return node


MD = "# Title\nintro\n## Sub A\nbody a\n## Sub B\nbody b\n"


def md_tree():
    c = MD
    n = maker(c)

    def heading(text):
        s = c.index(text)
        hashes = text.split(" ")[0]
        return n(
            "atx_heading",
            s,
            s + len(text),
            n("atx_marker", s, s + len(hashes)),
            n("inline", s + len(hashes) + 1, s + len(text)),
        )

    def para(text):
        s = c.index(text)
        return n("paragraph", s, s + len(text))

    s2, s3 = c.index("## Sub A"), c.index("## Sub B")
    sec2 = n("section", s2, s3, heading("## Sub A"), para("body a"))
    sec3 = n("section", s3, len(c), heading("## Sub B"), para("body b"))
    sec1 = n("section", 0, len(c), heading("# Title"), para("intro"), sec2, sec3)
    return n("document", 0, len(c), sec1)


def md_config(**grammar_extra):
    grammar = {
        "node_kinds": {
            "section": {"kind": "Section", "name": ["child:atx_heading>inline"]},
            "paragraph": "Paragraph",
        }
    }
    grammar.update(grammar_extra)
    return {"languages": {".md": "markdown"}, "grammars": {"markdown": grammar}}


def build_md(**grammar_extra):
    return build(MD, "dir/notes.md", md_tree(), "markdown", md_config(**grammar_extra))


ALL_IDS = [
    "document:notes",
    "section:Title@L1",
    "paragraph@L2",
    "section:Sub A@L3",
    "paragraph@L4",
    "section:Sub B@L5",
    "paragraph@L6",
]


def ids(result):
    return [e.id for e in result.entities]


def check_edges(result):
    assert result.edges == tuple((e.parent, e.id) for e in result.entities[1:])


def one_node_config(rule, *, node_type="item", extra=None, grammar_extra=None):
    grammar = {"node_kinds": {node_type: rule}}
    grammar.update(grammar_extra or {})
    cfg = {"languages": {".yaml": "yaml"}, "grammars": {"yaml": grammar}}
    cfg.update(extra or {})
    return cfg


def build_one(content, rule, *, start=0, end=None, children=(), fields=None, **kw):
    n = maker(content)
    end = len(content) if end is None else end
    item = n("item", start, end, *children, fields=fields)
    root = n("stream", 0, len(content), item)
    return build(content, "cfg/app.yaml", root, "yaml", one_node_config(rule, **kw))


# --- language_for -----------------------------------------------------------


def test_language_for_longest_extension_case_insensitive():
    langs = {".md": "markdown", ".yaml": "yaml", ".tar.yaml": "x"}
    assert language_for("a/B.MD", langs) == "markdown"
    assert language_for("c.tar.yaml", langs) == "x"
    assert language_for("d/e.yaml", langs) == "yaml"
    assert language_for("c.txt", langs) is None


# --- Document entity ----------------------------------------------------------


def test_document_entity_defaults():
    result = build_md()
    doc = result.entities[0]
    assert isinstance(doc, TreeEntity)
    assert doc.kind == "Document"
    assert doc.id == "document:notes"
    assert doc.name == "notes"
    assert doc.node_type == ""
    assert (doc.start, doc.end) == (0, len(MD))
    assert doc.code == MD
    assert doc.parent is None
    assert doc.depth == 0
    assert doc.line_start == 1


def test_document_kind_from_config():
    content = "a: 1\n"
    n = maker(content)
    cfg = {
        "languages": {".yaml": "yaml"},
        "document_kind": "File",
        "grammars": {"yaml": {"node_kinds": {}}},
    }
    result = build(content, "x/app.prod.yaml", n("stream", 0, len(content)), "yaml", cfg)
    doc = result.entities[0]
    assert doc.kind == "File"
    assert doc.id == "file:app.prod"
    assert doc.name == "app.prod"
    assert len(result.entities) == 1
    assert result.edges == ()
    assert result.depth_used == 0
    assert result.errors == ()


# --- mapping, nesting, edges --------------------------------------------------


def test_mapped_nodes_preorder_parent_depth_edges():
    result = build_md()
    assert isinstance(result, TreeResult)
    assert ids(result) == ALL_IDS
    by = {e.id: e for e in result.entities}
    parents = [e.parent for e in result.entities]
    assert parents == [
        None,
        "document:notes",
        "section:Title@L1",
        "section:Title@L1",
        "section:Sub A@L3",
        "section:Title@L1",
        "section:Sub B@L5",
    ]
    assert [e.depth for e in result.entities] == [0, 1, 2, 2, 3, 2, 3]
    assert [e.line_start for e in result.entities] == [1, 1, 2, 3, 4, 5, 6]
    assert by["section:Sub A@L3"].node_type == "section"
    assert by["section:Sub A@L3"].kind == "Section"
    assert by["section:Sub A@L3"].name == "Sub A"
    assert by["section:Sub A@L3"].code == "## Sub A\nbody a\n"
    assert by["paragraph@L4"].kind == "Paragraph"
    assert by["paragraph@L4"].name == "Paragraph@L4"
    assert by["paragraph@L4"].code == "body a"
    assert result.depth_used == 3
    assert result.errors == ()
    check_edges(result)
    for e in result.entities:
        assert e.code == MD[e.start : e.end]


def test_rule_without_kind_uses_node_type():
    result = build_one("hello\n", {})
    item = result.entities[1]
    assert item.kind == "item"
    assert item.name == "item@L1"
    assert item.id == "item@L1"


def test_plain_string_rule_is_kind():
    result = build_one("hello\n", "Entry")
    item = result.entities[1]
    assert item.kind == "Entry"
    assert item.id == "entry@L1"


# --- naming -------------------------------------------------------------------


def test_name_from_text_first_non_blank_line():
    content = "\n   \n  hello   world  \nsecond\n"
    result = build_one(content, {"kind": "Para", "name": ["text"]})
    item = result.entities[1]
    assert item.name == "hello world"
    assert item.line_start == 1
    assert item.id == "para:hello world@L1"


def test_name_first_non_empty_source_wins_and_field_source():
    content = "key: value\n"
    n = maker(content)
    key = n("plain", 0, 3)
    rule = {"kind": "Pair", "name": ["field:missing", "child:nothere", "field:key", "text"]}
    result = build_one(content, rule, end=10, children=[key], fields={"key": key})
    assert result.entities[1].name == "key"
    assert result.entities[1].id == "pair:key@L1"


def test_name_empty_field_falls_through_to_next_source():
    content = "x  : value\n"
    n = maker(content)
    empty = n("plain", 1, 3)  # only spaces
    rule = {"kind": "Pair", "name": ["field:key", "text"]}
    result = build_one(content, rule, end=10, children=[empty], fields={"key": empty})
    assert result.entities[1].name == "x : value"


def test_name_child_type_first_direct_child():
    content = "aa bb\n"
    n = maker(content)
    kids = [n("word", 0, 2), n("word", 3, 5)]
    result = build_one(content, {"kind": "Line", "name": ["child:word"]}, end=5, children=kids)
    assert result.entities[1].name == "aa"


def test_name_child_path():
    result = build_md()
    assert [e.name for e in result.entities if e.kind == "Section"] == ["Title", "Sub A", "Sub B"]


def test_name_through_unmapped_child_uses_its_first_non_blank_line():
    content = "\n  first line\nsecond\n"
    n = maker(content)
    block = n("block", 0, len(content))
    result = build_one(content, {"kind": "Box", "name": ["child:block"]}, children=[block])
    assert result.entities[1].name == "first line"


def test_name_through_mapped_child_uses_its_own_rule():
    content = "## Heading text ##\nbody\n"
    n = maker(content)
    heading = n("atx_heading", 0, 18, n("inline", 3, 15))
    sec = n("section", 0, len(content), heading)
    root = n("document", 0, len(content), sec)
    cfg = {
        "languages": {".md": "markdown"},
        "grammars": {
            "markdown": {
                "node_kinds": {
                    "section": {"kind": "Section", "name": ["field:title", "child:atx_heading"]},
                    "atx_heading": {"kind": "Heading", "name": ["child:inline"]},
                }
            }
        },
    }
    result = build(content, "a.md", root, "markdown", cfg)
    assert [e.kind for e in result.entities] == ["Document", "Section", "Heading"]
    assert result.entities[1].name == "Heading text"
    assert result.entities[2].name == "Heading text"
    assert result.entities[2].parent == result.entities[1].id


def test_name_collapse_whitespace_and_strip_chars():
    content = "##  \tSub \t  A  ##\n"
    result = build_one(content, {"kind": "H", "name": ["text"], "strip": "#"}, end=len(content) - 1)
    assert result.entities[1].name == "Sub A"
    assert result.entities[1].id == "h:Sub A@L1"


def test_name_clipped_to_name_max_chars():
    content = "abcdefghij\n"
    result = build_one(
        content, {"kind": "W", "name": ["text"]}, end=10, extra={"name_max_chars": 5}
    )
    assert result.entities[1].name == "abcde"
    assert result.entities[1].id == "w:abcde@L1"


def test_name_default_max_120_and_id_uses_first_60_chars():
    long = "x" * 200
    content = long + "\n"
    result = build_one(content, {"kind": "W", "name": ["text"]}, end=200)
    item = result.entities[1]
    assert item.name == "x" * 120
    assert item.id == "w:" + "x" * 60 + "@L1"


def test_no_source_yields_text_uses_kind_and_line():
    content = "\nfoo\nbar\n"
    n = maker(content)
    result = build_one(
        content,
        {"kind": "Thing", "name": ["field:nope", "child:nope"]},
        start=5,
        end=8,
        children=[n("other", 5, 8)],
    )
    item = result.entities[1]
    assert item.line_start == 3
    assert item.name == "Thing@L3"
    assert item.id == "thing@L3"


# --- duplicate ids --------------------------------------------------------------


def test_duplicate_ids_get_numbered_suffixes():
    content = "ab ab ab cd\n"
    n = maker(content)
    words = [n("word", i, i + 2) for i in (0, 3, 6, 9)]
    root = n("stream", 0, len(content), *words)
    cfg = {
        "languages": {},
        "grammars": {"g": {"node_kinds": {"word": {"kind": "Word", "name": ["text"]}}}},
    }
    result = build(content, "f.txt", root, "g", cfg)
    assert ids(result)[1:] == ["word:ab@L1", "word:ab@L1#2", "word:ab@L1#3", "word:cd@L1"]


def test_duplicate_unnamed_ids_get_suffixes():
    content = "ab ab\n"
    n = maker(content)
    root = n("stream", 0, len(content), n("word", 0, 2), n("word", 3, 5))
    cfg = {"languages": {}, "grammars": {"g": {"node_kinds": {"word": "Word"}}}}
    result = build(content, "f.txt", root, "g", cfg)
    assert ids(result)[1:] == ["word@L1", "word@L1#2"]
    check_edges(result)


# --- UTF-8 ----------------------------------------------------------------------


def test_multibyte_content_char_offsets():
    content = "é😀 intro\n## Über ü\nbody ñ\n"
    n = maker(content)
    hs = content.index("## Über")
    he = content.index("\nbody")
    sec = n("section", hs, len(content), n("atx_heading", hs, he, n("inline", hs + 3, he)))
    root = n("document", 0, len(content), sec)
    cfg = {
        "languages": {".md": "markdown"},
        "grammars": {
            "markdown": {
                "node_kinds": {"section": {"kind": "Section", "name": ["child:atx_heading>inline"]}}
            }
        },
    }
    result = build(content, "u.md", root, "markdown", cfg)
    sec_e = result.entities[1]
    assert (sec_e.start, sec_e.end) == (hs, len(content))
    assert sec_e.name == "Über ü"
    assert sec_e.line_start == 2
    assert sec_e.id == "section:Über ü@L2"
    assert result.entities[0].end == len(content)
    for e in result.entities:
        assert e.code == content[e.start : e.end]


# --- depth / count limits -------------------------------------------------------


def test_max_depth_limits_entities():
    result = build_md(max_depth=2)
    assert ids(result) == [
        "document:notes",
        "section:Title@L1",
        "paragraph@L2",
        "section:Sub A@L3",
        "section:Sub B@L5",
    ]
    assert result.depth_used == 2
    assert all(e.depth <= 2 for e in result.entities)
    check_edges(result)

    one = build_md(max_depth=1)
    assert ids(one) == ["document:notes", "section:Title@L1"]
    assert one.entities[1].code == MD
    assert one.depth_used == 1


def test_nested_requires_children():
    result = build_md(nested_requires_children=True)
    assert ids(result) == [
        "document:notes",
        "section:Title@L1",
        "section:Sub A@L3",
        "section:Sub B@L5",
    ]
    assert all(e.parent == "section:Title@L1" for e in result.entities[2:])
    assert result.depth_used == 2
    check_edges(result)


def test_nested_requires_children_counts_mapped_beyond_max_depth():
    result = build_md(nested_requires_children=True, max_depth=2)
    assert ids(result) == [
        "document:notes",
        "section:Title@L1",
        "section:Sub A@L3",
        "section:Sub B@L5",
    ]


def test_nested_requires_children_keeps_depth_one_always():
    content = "hello\n"
    result = build_one(content, "Leaf", grammar_extra={"nested_requires_children": True})
    assert ids(result) == ["document:app", "leaf@L1"]
    assert result.depth_used == 1


def test_max_entities_drops_deepest_levels():
    same = build_md(max_entities=7)
    assert ids(same) == ALL_IDS
    assert same.depth_used == 3

    two = build_md(max_entities=6)
    assert ids(two) == [
        "document:notes",
        "section:Title@L1",
        "paragraph@L2",
        "section:Sub A@L3",
        "section:Sub B@L5",
    ]
    assert two.depth_used == 2
    check_edges(two)

    one = build_md(max_entities=4)
    assert ids(one) == ["document:notes", "section:Title@L1"]
    assert one.depth_used == 1

    floor = build_md(max_entities=1)
    assert ids(floor) == ["document:notes", "section:Title@L1"]
    assert floor.depth_used == 1


# --- errors ---------------------------------------------------------------------


def error_tree():
    content = "a: 1\nb: [2\nc: 3\n"
    n = maker(content)
    b = content.index("b:")
    err = n("ERROR", b + 3, b + 6, n("flow", b + 3, b + 5, n("]", b + 5, b + 5, missing=True)))
    item_a = n("item", 0, 4)
    item_b = n("item", b, b + 6, err)
    cpos = content.index("c:")
    item_c = n("item", cpos, cpos + 4, n("]", cpos + 4, cpos + 4, missing=True))
    root = n("stream", 0, len(content), item_a, item_b, item_c)
    return content, root


def test_errors_listed_in_preorder_not_inside_error_nodes():
    content, root = error_tree()
    cfg = one_node_config("Item")
    result = build(content, "x.yaml", root, "yaml", cfg)
    assert result.errors == ("yaml: ERROR at L2:4", "yaml: MISSING ] at L3:5")
    assert [e.id for e in result.entities] == ["document:x", "item@L1", "item@L2", "item@L3"]


def test_errors_fallback_drops_entities():
    content, root = error_tree()
    cfg = one_node_config("Item", grammar_extra={"on_error": "fallback"})
    result = build(content, "x.yaml", root, "yaml", cfg)
    assert result.entities == ()
    assert result.edges == ()
    assert len(result.errors) == 2


def test_errors_keep_explicit():
    content, root = error_tree()
    cfg = one_node_config("Item", grammar_extra={"on_error": "keep"})
    result = build(content, "x.yaml", root, "yaml", cfg)
    assert len(result.entities) == 4
    assert len(result.edges) == 3
    assert len(result.errors) == 2


def many_errors(count):
    content = "x\n" * count
    n = maker(content)
    errs = [n("ERROR", 2 * i, 2 * i + 1) for i in range(count)]
    return content, n("stream", 0, len(content), *errs)


def test_errors_truncated_to_errors_listed_max():
    content, root = many_errors(5)
    cfg = one_node_config("Item", extra={"errors_listed_max": 2})
    result = build(content, "x.yaml", root, "yaml", cfg)
    assert result.errors == (
        "yaml: ERROR at L1:1",
        "yaml: ERROR at L2:1",
        "yaml: … 3 more error node(s)",
    )


def test_errors_listed_max_default_20():
    content, root = many_errors(25)
    result = build(content, "x.yaml", root, "yaml", one_node_config("Item"))
    assert len(result.errors) == 21
    assert result.errors[19] == "yaml: ERROR at L20:1"
    assert result.errors[20] == "yaml: … 5 more error node(s)"


def test_exactly_errors_listed_max_has_no_summary():
    content, root = many_errors(2)
    cfg = one_node_config("Item", extra={"errors_listed_max": 2})
    result = build(content, "x.yaml", root, "yaml", cfg)
    assert result.errors == ("yaml: ERROR at L1:1", "yaml: ERROR at L2:1")


# --- parse ----------------------------------------------------------------------


class FakeTree:
    def __init__(self, root):
        self.root_node = root


class FakeParser:
    def __init__(self, root):
        self.root = root
        self.received = []

    def parse(self, data):
        self.received.append(data)
        return FakeTree(self.root)


def test_parse_unknown_extension_does_not_call_parser_for():
    calls = []
    result = parse("# hi\n", "notes.txt", md_config(), calls.append)
    assert result.entities == ()
    assert len(result.errors) == 1
    assert calls == []


def test_parse_blank_content_does_not_call_parser_for():
    calls = []
    result = parse("  \n\t\n", "notes.md", md_config(), calls.append)
    assert result.entities == ()
    assert len(result.errors) == 1
    assert calls == []


def test_parse_known_extension_equals_build():
    root = md_tree()
    parser = FakeParser(root)
    calls = []

    def parser_for(lang):
        calls.append(lang)
        return parser

    result = parse(MD, "dir/notes.MD", md_config(), parser_for)
    assert calls == ["markdown"]
    assert parser.received == [MD.encode("utf-8")]
    assert result == build(MD, "dir/notes.MD", root, "markdown", md_config())
    assert ids(result) == ALL_IDS


def test_parse_utf8_bytes_passed_to_parser():
    content = "é\n"
    n = maker(content)
    parser = FakeParser(n("document", 0, len(content)))
    parse(content, "u.md", md_config(), lambda lang: parser)
    assert parser.received == [content.encode("utf-8")]


def test_parse_grammar_not_installed_propagates():
    def parser_for(lang):
        raise GrammarNotInstalled(lang)

    with pytest.raises(GrammarNotInstalled):
        parse(MD, "notes.md", md_config(), parser_for)
    assert issubclass(GrammarNotInstalled, RuntimeError)
