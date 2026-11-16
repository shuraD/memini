"""IN-02: document-type rules, detection and file-type mapping (card I14, I16).

Every value list comes from the rules mapping (AGENTS.md rule 2); the baseline mapping
below is test data that follows the old detector's logic. No database, no model, no I/O.
"""

import builtins
import copy
import importlib
import inspect
from typing import Any

import pytest

from memini.core.ingest import doctype
from memini.core.ingest.doctype import (
    ConfigError,
    Detection,
    detect,
    file_type,
    parse_rules,
)

CODE_EXTENSIONS = [
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".rs",
    ".go",
    ".java",
    ".xml",
    ".toml",
    ".html",
    ".htm",
]

BASELINE: dict[str, Any] = {
    "types": [
        {
            "name": "code",
            "indexing_strategy": "hybrid",
            "preserve_original": True,
            "enable_entity_extraction": True,
            "extensions": CODE_EXTENSIONS,
        },
        {
            "name": "markdown",
            "indexing_strategy": "entity_extraction",
            "preserve_original": False,
            "enable_entity_extraction": True,
            "extensions": [".md", ".markdown"],
        },
        {
            "name": "openapi",
            "indexing_strategy": "hybrid",
            "preserve_original": True,
            "enable_entity_extraction": True,
            "extensions": [".json", ".yaml", ".yml", ""],
            "parse": "yaml",
            "key_sets": [["openapi"], ["swagger"], ["info", "paths"]],
        },
        {
            "name": "json_schema",
            "indexing_strategy": "structured",
            "preserve_original": True,
            "enable_entity_extraction": False,
            "extensions": [".json"],
            "name_contains": "schema",
            "parse": "json",
            "key_sets": [["$schema"]],
        },
        {
            "name": "json_data",
            "indexing_strategy": "structured",
            "preserve_original": True,
            "enable_entity_extraction": False,
            "extensions": [".json"],
            "parse": "json",
        },
        {
            "name": "yaml_config",
            "indexing_strategy": "structured",
            "preserve_original": True,
            "enable_entity_extraction": False,
            "extensions": [".yaml", ".yml"],
            "parse": "yaml",
            "content_words": ["database", "server", "config", "settings"],
        },
    ],
    "default": {
        "name": "text",
        "indexing_strategy": "entity_extraction",
        "preserve_original": False,
        "enable_entity_extraction": True,
    },
    "file_types": {
        "by_document_type": {
            "openapi": "openapi",
            "json_schema": "json",
            "json_data": "json",
            "yaml_config": "yaml",
            "markdown": "markdown",
            "text": "markdown",
        },
        "by_prefix": {"json": "json"},
        "by_extension": {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".rs": "rust",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".html": "html",
            ".htm": "html",
            ".toml": "toml",
        },
        "extension_default": "python",
        "default": "markdown",
    },
}

OPENAPI_YAML = "openapi: 3.0.0\npaths: {}\n"
TEXT = Detection("text", "entity_extraction", False, True)
CODE = Detection("code", "hybrid", True, True)
MARKDOWN = Detection("markdown", "entity_extraction", False, True)


def _raw() -> dict[str, Any]:
    return copy.deepcopy(BASELINE)


@pytest.fixture
def rules() -> doctype.DoctypeRules:
    return parse_rules(_raw())


# ---------------------------------------------------------------- detect: OpenAPI


def test_openapi_yaml_is_detected(rules: doctype.DoctypeRules) -> None:
    assert detect(OPENAPI_YAML, "api.yaml", rules) == Detection("openapi", "hybrid", True, True)


def test_swagger_json_is_openapi(rules: doctype.DoctypeRules) -> None:
    assert detect('{"swagger": "2.0"}', "spec.json", rules).document_type == "openapi"


def test_yaml_with_info_and_paths_without_extension_is_openapi(
    rules: doctype.DoctypeRules,
) -> None:
    content = "info:\n  title: Pets\npaths:\n  /pets: {}\n"
    assert detect(content, "petstore", rules).document_type == "openapi"


def test_yaml_with_only_one_key_of_a_set_is_not_openapi(rules: doctype.DoctypeRules) -> None:
    content = "info:\n  title: Pets\n"
    assert detect(content, "petstore", rules) == TEXT


def test_openapi_shape_with_code_extension_is_code(rules: doctype.DoctypeRules) -> None:
    assert detect(OPENAPI_YAML, "openapi_parser.py", rules) == CODE


# ---------------------------------------------------------------- detect: JSON


def test_json_schema_is_detected(rules: doctype.DoctypeRules) -> None:
    content = '{"$schema": "http://json-schema.org/draft-07/schema#", "type": "object"}'
    assert detect(content, "user_schema.json", rules) == Detection(
        "json_schema", "structured", True, False
    )


def test_json_object_without_schema_is_json_data(rules: doctype.DoctypeRules) -> None:
    assert detect('{"name": "alpha", "count": 3}', "data.json", rules).document_type == "json_data"


def test_json_list_is_json_data(rules: doctype.DoctypeRules) -> None:
    assert detect('[1, 2, {"a": "b"}]', "data.json", rules).document_type == "json_data"


def test_invalid_json_falls_back_to_default(rules: doctype.DoctypeRules) -> None:
    assert detect('{"name": "alpha",', "bad.json", rules) == TEXT


# ---------------------------------------------------------------- detect: YAML config


def test_yaml_with_config_word_is_yaml_config(rules: doctype.DoctypeRules) -> None:
    content = "database:\n  host: localhost\n  port: 5432\n"
    assert detect(content, "app.yml", rules).document_type == "yaml_config"


def test_yaml_without_config_word_is_text(rules: doctype.DoctypeRules) -> None:
    content = "title: hello\ncount: 3\n"
    assert detect(content, "app.yml", rules) == TEXT


# ---------------------------------------------------------------- detect: markdown, code, text


@pytest.mark.parametrize("filename", ["README.md", "notes.markdown", "GUIDE.MD"])
def test_markdown_extensions(rules: doctype.DoctypeRules, filename: str) -> None:
    assert detect("# Title\n\nSome words.\n", filename, rules) == MARKDOWN


def test_markdown_ignores_content_shape(rules: doctype.DoctypeRules) -> None:
    assert detect(OPENAPI_YAML, "README.md", rules) == MARKDOWN


@pytest.mark.parametrize(
    "filename", ["main.py", "app.ts", "view.tsx", "pom.xml", "pyproject.toml", "lib.rs"]
)
def test_code_extensions(rules: doctype.DoctypeRules, filename: str) -> None:
    assert detect("x = 1\n", filename, rules) == CODE


def test_code_extension_is_case_insensitive(rules: doctype.DoctypeRules) -> None:
    assert detect("x = 1\n", "MAIN.PY", rules) == CODE


@pytest.mark.parametrize(
    ("content", "filename"),
    [
        ("Just some plain words here.\n", "notes"),
        ("Just some plain words here.\n", "notes.xyz"),
        ("", "notes"),
        ("", "empty.txt"),
        ("", ""),
    ],
)
def test_plain_unlisted_or_empty_is_text(
    rules: doctype.DoctypeRules, content: str, filename: str
) -> None:
    assert detect(content, filename, rules) == TEXT


# ---------------------------------------------------------------- detect: custom rules


def _rule(name: str, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "indexing_strategy": "custom_strategy",
        "preserve_original": True,
        "enable_entity_extraction": False,
        **extra,
    }


def _custom(types: list[dict[str, Any]]) -> doctype.DoctypeRules:
    raw = _raw()
    raw["types"] = types
    return parse_rules(raw)


def test_custom_rule_names_a_new_type() -> None:
    rules = _custom([_rule("terraform", extensions=[".tf"])])
    assert detect('resource "x" "y" {}\n', "main.tf", rules) == Detection(
        "terraform", "custom_strategy", True, False
    )


def test_custom_default_is_returned_when_nothing_matches() -> None:
    raw = _raw()
    raw["types"] = [_rule("terraform", extensions=[".tf"])]
    raw["default"] = {
        "name": "plain",
        "indexing_strategy": "chunk_only",
        "preserve_original": True,
        "enable_entity_extraction": False,
    }
    rules = parse_rules(raw)
    assert detect("hello", "notes.txt", rules) == Detection("plain", "chunk_only", True, False)


def test_rule_order_decides_when_two_rules_match() -> None:
    first = _rule("first", extensions=[".tf"])
    second = _rule("second", extensions=[".tf"])
    assert detect("x", "main.tf", _custom([first, second])).document_type == "first"
    assert detect("x", "main.tf", _custom([second, first])).document_type == "second"


def test_rule_without_extensions_matches_any_extension() -> None:
    rules = _custom([_rule("anything")])
    assert detect("x", "a.whatever", rules).document_type == "anything"
    assert detect("x", "noext", rules).document_type == "anything"


def test_empty_string_extension_means_no_suffix() -> None:
    rules = _custom([_rule("bare", extensions=[""])])
    assert detect("x", "Makefile", rules).document_type == "bare"
    assert detect("x", "Makefile.am", rules) == TEXT


def test_name_contains_is_matched_on_lowercased_filename() -> None:
    rules = _custom([_rule("changelog", name_contains="changelog")])
    assert detect("x", "docs/CHANGELOG.txt", rules).document_type == "changelog"
    assert detect("x", "docs/history.txt", rules) == TEXT


def test_content_words_need_at_least_one_on_lowercased_content() -> None:
    rules = _custom([_rule("invoice", content_words=["invoice", "receipt"])])
    assert detect("Your RECEIPT is attached", "a.txt", rules).document_type == "invoice"
    assert detect("Nothing relevant", "a.txt", rules) == TEXT


def test_key_sets_need_all_keys_of_one_set() -> None:
    rules = _custom([_rule("pair", parse="json", key_sets=[["a", "b"], ["c"]])])
    assert detect('{"a": 1, "b": 2}', "x.json", rules).document_type == "pair"
    assert detect('{"c": 1}', "x.json", rules).document_type == "pair"
    assert detect('{"a": 1}', "x.json", rules) == TEXT
    assert detect('["a", "b"]', "x.json", rules) == TEXT


def test_parse_json_rejects_yaml_only_content() -> None:
    rules = _custom([_rule("j", parse="json")])
    assert detect("a: 1\n", "x", rules) == TEXT
    assert detect('{"a": 1}', "x", rules).document_type == "j"


def test_parse_failure_is_no_match() -> None:
    rules = _custom([_rule("y", parse="yaml")])
    assert detect("key: [unclosed\n", "x", rules) == TEXT


# ---------------------------------------------------------------- file_type


@pytest.mark.parametrize(
    ("document_type", "filename", "expected"),
    [
        ("openapi", "api.yaml", "openapi"),
        ("json_schema", "user_schema.json", "json"),
        ("json_data", "data.json", "json"),
        ("json_lines", "events.jsonl", "json"),
        ("yaml_config", "app.yml", "yaml"),
        ("markdown", "README.md", "markdown"),
        ("text", "notes.txt", "markdown"),
        ("unknown", "", "markdown"),
    ],
)
def test_file_type_by_document_type(
    rules: doctype.DoctypeRules, document_type: str, filename: str, expected: str
) -> None:
    assert file_type(document_type, filename, rules) == expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("main.py", "python"),
        ("view.tsx", "typescript"),
        ("app.ts", "typescript"),
        ("widget.jsx", "javascript"),
        ("lib.rs", "rust"),
        ("deploy.yaml", "yaml"),
        ("index.htm", "html"),
        ("pyproject.toml", "toml"),
    ],
)
def test_file_type_for_code_uses_extension(
    rules: doctype.DoctypeRules, filename: str, expected: str
) -> None:
    assert file_type("code", filename, rules) == expected


def test_file_type_for_code_with_unlisted_extension_is_extension_default(
    rules: doctype.DoctypeRules,
) -> None:
    assert file_type("code", "Main.kt", rules) == "python"


def test_file_type_for_code_with_empty_filename_is_default(
    rules: doctype.DoctypeRules,
) -> None:
    assert file_type("code", "", rules) == "markdown"


def test_file_type_values_come_from_rules() -> None:
    raw = _raw()
    raw["file_types"] = {
        "by_document_type": {"alpha": "type_a"},
        "by_prefix": {"be": "type_b"},
        "by_extension": {".zz": "type_z"},
        "extension_default": "type_ext",
        "default": "type_default",
    }
    rules = parse_rules(raw)
    assert file_type("alpha", "x.zz", rules) == "type_a"
    assert file_type("beta", "x.zz", rules) == "type_b"
    assert file_type("gamma", "x.zz", rules) == "type_z"
    assert file_type("gamma", "x.qq", rules) == "type_ext"
    assert file_type("gamma", "", rules) == "type_default"


# ---------------------------------------------------------------- parse_rules errors


def test_config_error_is_a_value_error() -> None:
    assert issubclass(ConfigError, ValueError)


@pytest.mark.parametrize("section", ["types", "default", "file_types"])
def test_missing_section_raises(section: str) -> None:
    raw = _raw()
    del raw[section]
    with pytest.raises(ConfigError):
        parse_rules(raw)


@pytest.mark.parametrize(
    "field", ["name", "indexing_strategy", "preserve_original", "enable_entity_extraction"]
)
def test_rule_missing_field_raises(field: str) -> None:
    raw = _raw()
    del raw["types"][0][field]
    with pytest.raises(ConfigError):
        parse_rules(raw)


@pytest.mark.parametrize(
    "field", ["name", "indexing_strategy", "preserve_original", "enable_entity_extraction"]
)
def test_default_missing_field_raises(field: str) -> None:
    raw = _raw()
    del raw["default"][field]
    with pytest.raises(ConfigError):
        parse_rules(raw)


@pytest.mark.parametrize("flag", ["preserve_original", "enable_entity_extraction"])
@pytest.mark.parametrize("value", ["true", "yes", 1, None])
def test_non_bool_flag_in_rule_raises(flag: str, value: object) -> None:
    raw = _raw()
    raw["types"][1][flag] = value
    with pytest.raises(ConfigError):
        parse_rules(raw)


@pytest.mark.parametrize("flag", ["preserve_original", "enable_entity_extraction"])
def test_non_bool_flag_in_default_raises(flag: str) -> None:
    raw = _raw()
    raw["default"][flag] = "false"
    with pytest.raises(ConfigError):
        parse_rules(raw)


@pytest.mark.parametrize("kind", ["xml", "toml", "csv"])
def test_unknown_parse_kind_raises(kind: str) -> None:
    raw = _raw()
    raw["types"][2]["parse"] = kind
    with pytest.raises(ConfigError):
        parse_rules(raw)


def test_baseline_rules_parse() -> None:
    assert isinstance(parse_rules(_raw()), doctype.DoctypeRules)


# ---------------------------------------------------------------- purity


def test_module_does_not_import_llm_providers() -> None:
    source = inspect.getsource(importlib.import_module("memini.core.ingest.doctype"))
    assert "llm_providers" not in source


def test_detect_and_file_type_do_no_file_io(
    rules: doctype.DoctypeRules, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[object] = []

    def _no_open(*args: object, **kwargs: object) -> None:
        opened.append(args)
        raise AssertionError("detect must not open files")

    monkeypatch.setattr(builtins, "open", _no_open)
    result = detect(OPENAPI_YAML, "api.yaml", rules)
    kind = file_type("code", "main.py", rules)
    monkeypatch.undo()
    assert (result.document_type, kind, opened) == ("openapi", "python", [])
