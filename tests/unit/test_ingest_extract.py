"""Acceptance tests for IN-10: extraction prompts per window and one batched LLM call."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import pytest

from memini.core.ingest.extract import (
    RELATED_TAGS_FILE,
    TEMPLATE_FILE,
    ExtractionPrompts,
    UnknownPersonaError,
    WindowExtraction,
    build_prompt,
    extract_windows,
    load_prompts,
    persona_for,
    related_tags_line,
    user_labels,
)
from memini.infra.llm_providers import Attempt, Generation, LLMError, Reason

TEMPLATE = 'P={persona}|T={related_tags}|C={content}|{"k": 1}'
RELATED = "Related tags: {tags}\n"
PROMPTS = ExtractionPrompts(template=TEMPLATE, related_tags=RELATED)

PERSONAS = {
    "role-persona": "ROLE-PERSONA-TEXT",
    "domain-persona": "DOMAIN-PERSONA-TEXT",
    "fallback-persona": "FALLBACK-PERSONA-TEXT",
}


@dataclasses.dataclass(frozen=True)
class FakeWindow:
    text: str


def _generation(text: str) -> Generation:
    return Generation(
        text=text,
        provider="fake",
        model="fake-model",
        role="extraction",
        prompt_tokens=1,
        completion_tokens=1,
        seconds=0.0,
        cached=False,
    )


def _error() -> LLMError:
    return LLMError(
        (Attempt(provider="fake", reason=Reason.SERVER_ERROR, detail="boom"),), "extraction"
    )


class FakeLLM:
    """Records calls; generate_many answers each prompt in order."""

    def __init__(self, fail_at: set[int] | None = None) -> None:
        self.fail_at = fail_at or set()
        self.many_calls: list[tuple[list[str], dict[str, Any]]] = []
        self.single_calls: list[tuple[str, dict[str, Any]]] = []
        self.results: list[Generation | LLMError] = []

    def generate(self, prompt: str, **kwargs: Any) -> Generation:
        self.single_calls.append((prompt, kwargs))
        return _generation("single")

    def generate_many(self, prompts: Sequence[str], **kwargs: Any) -> list[Generation | LLMError]:
        prompts = list(prompts)
        self.many_calls.append((prompts, kwargs))
        out: list[Generation | LLMError] = []
        for i, _ in enumerate(prompts):
            out.append(_error() if i in self.fail_at else _generation(f"reply-{i}"))
        self.results = out
        return out


def _run(windows, *, llm, role="role-persona", domain="domain-persona", labels=(), derived=()):
    return extract_windows(
        windows,
        role=role,
        domain=domain,
        default_persona="fallback-persona",
        personas=PERSONAS,
        labels=list(labels),
        derived_labels=set(derived),
        llm=llm,
        prompts=PROMPTS,
    )


# --- constants and load_prompts -------------------------------------------------


def test_file_name_constants():
    assert TEMPLATE_FILE == "extraction.txt"
    assert RELATED_TAGS_FILE == "extraction_related_tags.txt"


def test_load_prompts_reads_both_files_exactly(tmp_path):
    template = "Persona: {persona}\n{related_tags}\nText é 中:\n{content}\n  trailing  \n"
    related = "  Related tags: {tags}  \n\n"
    (tmp_path / "extraction.txt").write_bytes(template.encode("utf-8"))
    (tmp_path / "extraction_related_tags.txt").write_bytes(related.encode("utf-8"))
    loaded = load_prompts(tmp_path)
    assert isinstance(loaded, ExtractionPrompts)
    assert loaded.template == template
    assert loaded.related_tags == related


def test_load_prompts_missing_template_raises(tmp_path):
    (tmp_path / "extraction_related_tags.txt").write_text("{tags}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_prompts(tmp_path)


def test_load_prompts_missing_related_tags_raises(tmp_path):
    (tmp_path / "extraction.txt").write_text("{content}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_prompts(tmp_path)


def test_extraction_prompts_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        PROMPTS.template = "x"  # type: ignore[misc]


# --- persona_for ----------------------------------------------------------------


def test_persona_for_prefers_role():
    assert persona_for("r", "d", "g") == "r"


@pytest.mark.parametrize("role", [None, ""])
def test_persona_for_falls_back_to_domain(role):
    assert persona_for(role, "d", "g") == "d"


@pytest.mark.parametrize("role", [None, ""])
@pytest.mark.parametrize("domain", [None, ""])
def test_persona_for_falls_back_to_default(role, domain):
    assert persona_for(role, domain, "g") == "g"


# --- user_labels ----------------------------------------------------------------


def test_user_labels_strips_drops_blank_derived_and_duplicates():
    assert user_labels([" a ", "b", "", "a", "derived1"], {"derived1"}) == ["a", "b"]


def test_user_labels_drops_whitespace_only_and_stripped_derived():
    assert user_labels(["  ", " d ", "c", "c "], {"d"}) == ["c"]


def test_user_labels_keeps_first_order():
    assert user_labels(["z", "y", "z", "x"], set()) == ["z", "y", "x"]


# --- related_tags_line ----------------------------------------------------------


@pytest.mark.parametrize("labels", [[], [""], ["  ", "\t"]])
def test_related_tags_line_empty_without_labels(labels):
    assert related_tags_line(labels, PROMPTS) == ""


def test_related_tags_line_joins_stripped_labels():
    assert related_tags_line(["x", " y "], PROMPTS) == "Related tags: x, y"


def test_related_tags_line_skips_blank_labels_and_strips_template():
    prompts = ExtractionPrompts(template=TEMPLATE, related_tags="  Tags: {tags}  \n")
    assert related_tags_line(["x", "", " "], prompts) == "Tags: x"


# --- build_prompt ---------------------------------------------------------------


def test_build_prompt_fills_placeholders_and_keeps_other_braces():
    out = build_prompt("body", persona="PX", labels=["x", " y "], prompts=PROMPTS)
    assert out == 'P=PX|T=Related tags: x, y|C=body|{"k": 1}'


def test_build_prompt_without_labels_replaces_related_tags_with_empty():
    out = build_prompt("body", persona="PX", labels=[], prompts=PROMPTS)
    assert out == 'P=PX|T=|C=body|{"k": 1}'


def test_build_prompt_does_not_rescan_text():
    text = "a {persona} b {related_tags} c {content} d"
    out = build_prompt(text, persona="PX", labels=["x"], prompts=PROMPTS)
    assert out == "P=PX|T=Related tags: x|C=" + text + '|{"k": 1}'


def test_build_prompt_does_not_rescan_persona():
    persona = "who {content} and {related_tags} and {persona}"
    out = build_prompt("body", persona=persona, labels=[], prompts=PROMPTS)
    assert out == "P=" + persona + '|T=|C=body|{"k": 1}'


def test_build_prompt_does_not_rescan_labels():
    out = build_prompt("body", persona="PX", labels=["{content}"], prompts=PROMPTS)
    assert out == 'P=PX|T=Related tags: {content}|C=body|{"k": 1}'


def test_build_prompt_keeps_long_text_whole():
    text = "".join(f"line {i} with some words {{x}}\n" for i in range(20000))
    out = build_prompt(text, persona="PX", labels=[], prompts=PROMPTS)
    assert text in out
    assert out == "P=PX|T=|C=" + text + '|{"k": 1}'


def test_build_prompt_replaces_repeated_placeholders():
    prompts = ExtractionPrompts(template="{content}/{content}/{persona}", related_tags=RELATED)
    assert build_prompt("t", persona="p", labels=[], prompts=prompts) == "t/t/p"


# --- extract_windows ------------------------------------------------------------


def test_extract_windows_one_batched_call_in_order():
    windows = [FakeWindow("first"), FakeWindow("second"), FakeWindow("third")]
    llm = FakeLLM()
    result = _run(windows, llm=llm, labels=["x"])
    assert len(llm.many_calls) == 1
    assert llm.single_calls == []
    sent, kwargs = llm.many_calls[0]
    expected = [
        build_prompt(w.text, persona="ROLE-PERSONA-TEXT", labels=["x"], prompts=PROMPTS)
        for w in windows
    ]
    assert sent == expected
    assert kwargs.get("json_mode")
    assert kwargs.get("temperature", 0.0) == 0.0
    assert len(result) == 3
    for i, item in enumerate(result):
        assert isinstance(item, WindowExtraction)
        assert item.index == i
        assert item.prompt == expected[i]
        assert item.generation == llm.results[i]
        assert item.generation.text == f"reply-{i}"
        assert item.error is None


def test_extract_windows_role_persona_used():
    llm = FakeLLM()
    _run([FakeWindow("a"), FakeWindow("b")], llm=llm)
    for prompt in llm.many_calls[0][0]:
        assert "ROLE-PERSONA-TEXT" in prompt
        assert "DOMAIN-PERSONA-TEXT" not in prompt
        assert "FALLBACK-PERSONA-TEXT" not in prompt


@pytest.mark.parametrize("role", [None, ""])
def test_extract_windows_domain_persona_without_role(role):
    llm = FakeLLM()
    _run([FakeWindow("a"), FakeWindow("b")], llm=llm, role=role)
    for prompt in llm.many_calls[0][0]:
        assert "DOMAIN-PERSONA-TEXT" in prompt
        assert "ROLE-PERSONA-TEXT" not in prompt
        assert "FALLBACK-PERSONA-TEXT" not in prompt


def test_extract_windows_default_persona_without_role_or_domain():
    llm = FakeLLM()
    _run([FakeWindow("a")], llm=llm, role=None, domain=None)
    for prompt in llm.many_calls[0][0]:
        assert "FALLBACK-PERSONA-TEXT" in prompt
        assert "ROLE-PERSONA-TEXT" not in prompt
        assert "DOMAIN-PERSONA-TEXT" not in prompt


def test_extract_windows_shows_only_user_labels():
    llm = FakeLLM()
    _run(
        [FakeWindow("a"), FakeWindow("b")],
        llm=llm,
        labels=[" mine ", "autoderived", "", "mine", "other"],
        derived={"autoderived"},
    )
    for prompt in llm.many_calls[0][0]:
        assert "Related tags: mine, other" in prompt
        assert "autoderived" not in prompt


def test_extract_windows_no_labels_gives_empty_related_tags():
    llm = FakeLLM()
    _run([FakeWindow("a")], llm=llm, labels=["only-derived"], derived={"only-derived"})
    assert llm.many_calls[0][0] == ['P=ROLE-PERSONA-TEXT|T=|C=a|{"k": 1}']


def test_extract_windows_error_for_one_window():
    llm = FakeLLM(fail_at={1})
    result = _run([FakeWindow("a"), FakeWindow("b"), FakeWindow("c")], llm=llm)
    assert [r.index for r in result] == [0, 1, 2]
    assert result[1].generation is None
    assert result[1].error is llm.results[1]
    assert isinstance(result[1].error, LLMError)
    for i in (0, 2):
        assert result[i].error is None
        assert result[i].generation == llm.results[i]


def test_extract_windows_zero_windows_no_call():
    llm = FakeLLM()
    assert _run([], llm=llm) == []
    assert llm.many_calls == []
    assert llm.single_calls == []


@pytest.mark.parametrize(
    ("role", "domain"),
    [("missing-role", "domain-persona"), (None, "missing-domain")],
)
def test_extract_windows_unknown_persona_raises_without_call(role, domain):
    llm = FakeLLM()
    with pytest.raises(UnknownPersonaError):
        _run([FakeWindow("a")], llm=llm, role=role, domain=domain)
    assert llm.many_calls == []
    assert llm.single_calls == []


def test_extract_windows_unknown_default_persona_raises():
    llm = FakeLLM()
    with pytest.raises(UnknownPersonaError):
        extract_windows(
            [FakeWindow("a")],
            role=None,
            domain=None,
            default_persona="nope",
            personas=PERSONAS,
            labels=[],
            derived_labels=set(),
            llm=llm,
            prompts=PROMPTS,
        )
    assert llm.many_calls == []


def test_unknown_persona_error_is_lookup_error():
    assert issubclass(UnknownPersonaError, LookupError)


def test_window_extraction_is_frozen():
    item = WindowExtraction(index=0, prompt="p", generation=_generation("g"), error=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.index = 1  # type: ignore[misc]
