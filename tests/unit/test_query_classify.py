"""Acceptance tests for QY-03: the LLM question classifier.

Written from the QY-03 specification before the implementation exists.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Sequence
from typing import Any

import pytest

from memini.core.query.classify import (
    DRAWS,
    PROMPT_FILE,
    Classifier,
    Intent,
    Vocabulary,
    build_prompt,
    default_intent,
    load_prompt,
)
from memini.infra.llm_providers import Generation, LLMError

VOCAB = Vocabulary(
    modes=("local", "global", "hybrid"),
    domains=("Billing", "Search", "Payments"),
    types=("Spec", "Ticket"),
    styles=("brief", "detailed", "table"),
    query_types=("lookup", "relation", "gap"),
)

TEMPLATE = (
    "Question: {question}\n"
    "Modes: {modes}\n"
    "Domains: {domains}\n"
    "Types: {types}\n"
    "Styles: {styles}\n"
    "Query types: {query_types}\n"
    'Reply as {"mode": ...}\n'
)


def _error() -> LLMError:
    return LLMError((), "classifier")


class FakeLLM:
    """Scripted LLM: each call pops the next reply (text or exception) and is recorded."""

    def __init__(self, replies: Sequence[str | Exception] = ()) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool | dict[str, Any] = False,
        temperature: float = 0.0,
        seed: int | None = None,
        stage: str | None = None,
    ) -> Generation:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "json_mode": json_mode,
                "temperature": temperature,
                "seed": seed,
            }
        )
        if not self.replies:
            raise _error()
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Generation(
            text=reply,
            provider="fake",
            model="fake-model",
            role="classifier",
            prompt_tokens=0,
            completion_tokens=0,
            seconds=0.0,
            cached=False,
        )

    def generate_many(
        self,
        prompts: Sequence[str],
        *,
        system: str | None = None,
        json_mode: bool | dict[str, Any] = False,
        temperature: float = 0.0,
        seed: int | None = None,
        stage: str | None = None,
    ) -> list[Generation | LLMError]:
        raise AssertionError("generate_many must not be used")


def _reply(**fields: Any) -> str:
    base: dict[str, Any] = {"mode": "local"}
    base.update(fields)
    return json.dumps(base)


def _classify(
    question: str, *replies: str | Exception, fallback: Any = None
) -> tuple[Intent, FakeLLM]:
    llm = FakeLLM(replies)
    classifier = Classifier(llm, VOCAB, TEMPLATE, fallback=fallback)
    return classifier.classify(question), llm


# ---------------------------------------------------------------- constants, defaults


def test_constants() -> None:
    assert PROMPT_FILE == "query_classify.txt"
    assert DRAWS == ((0.0, None), (0.3, 42), (0.5, 137))


def test_default_intent() -> None:
    intent = default_intent(VOCAB)
    assert intent == Intent(
        mode="local",
        domains=(),
        types=(),
        answer_style="brief",
        query_type="lookup",
        enumerate=False,
        target_concept=None,
        source="default",
    )


def test_dataclasses_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        VOCAB.modes = ("x",)  # type: ignore[misc]
    intent = default_intent(VOCAB)
    with pytest.raises(dataclasses.FrozenInstanceError):
        intent.mode = "global"  # type: ignore[misc]


# ---------------------------------------------------------------- load_prompt


def test_load_prompt_returns_exact_text(tmp_path: pathlib.Path) -> None:
    text = 'Classify: {question}\nÉtape — {"mode": ...}\n\n  trailing  \n'
    (tmp_path / PROMPT_FILE).write_text(text, encoding="utf-8")
    assert load_prompt(tmp_path) == text


def test_load_prompt_missing_file_raises(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt(tmp_path)


# ---------------------------------------------------------------- build_prompt


def test_build_prompt_fills_all_placeholders() -> None:
    out = build_prompt("Which specs cover refunds?", VOCAB, TEMPLATE)
    assert out == (
        "Question: Which specs cover refunds?\n"
        "Modes: local, global, hybrid\n"
        "Domains: Billing, Search, Payments\n"
        "Types: Spec, Ticket\n"
        "Styles: brief, detailed, table\n"
        "Query types: lookup, relation, gap\n"
        'Reply as {"mode": ...}\n'
    )
    for placeholder in (
        "{question}",
        "{modes}",
        "{domains}",
        "{types}",
        "{styles}",
        "{query_types}",
    ):
        assert placeholder not in out
    assert '{"mode": ...}' in out


def test_build_prompt_is_single_pass() -> None:
    out = build_prompt("what is {domains} and {question}?", VOCAB, "{question}|{domains}")
    assert out == "what is {domains} and {question}?|Billing, Search, Payments"


def test_build_prompt_keeps_other_braces() -> None:
    template = "{unknown} {question} {} {{x}}"
    assert build_prompt("q", VOCAB, template) == "{unknown} q {} {{x}}"


# ---------------------------------------------------------------- construction, warm


def test_construction_makes_no_call() -> None:
    llm = FakeLLM(["{}"])
    Classifier(llm, VOCAB, TEMPLATE)
    Classifier(llm, VOCAB, TEMPLATE, fallback=lambda q: default_intent(VOCAB))
    assert llm.calls == []


def test_warm_makes_one_call_and_returns_true() -> None:
    llm = FakeLLM(["anything at all"])
    classifier = Classifier(llm, VOCAB, TEMPLATE)
    assert classifier.warm() is True
    assert len(llm.calls) == 1
    assert llm.calls[0]["prompt"] == build_prompt("", VOCAB, TEMPLATE)


def test_warm_returns_false_on_llm_error() -> None:
    llm = FakeLLM([_error()])
    classifier = Classifier(llm, VOCAB, TEMPLATE)
    assert classifier.warm() is False
    assert len(llm.calls) == 1


# ---------------------------------------------------------------- first draw


def test_valid_first_reply_single_call() -> None:
    question = "Which billing specs exist?"
    intent, llm = _classify(question, _reply(mode="global"))
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["prompt"] == build_prompt(question, VOCAB, TEMPLATE)
    assert call["json_mode"] is True
    assert call["temperature"] == 0.0
    assert call["seed"] is None
    assert intent.source == "llm"
    assert intent.mode == "global"


def test_minimal_reply_gives_defaults_for_other_fields() -> None:
    intent, _ = _classify("anything", _reply(mode="hybrid"))
    assert intent == Intent(
        mode="hybrid",
        domains=(),
        types=(),
        answer_style="brief",
        query_type="lookup",
        enumerate=False,
        target_concept=None,
        source="llm",
    )


def test_values_returned_in_vocabulary_spelling() -> None:
    question = "Show how search depends on billing as a table"
    reply = _reply(
        mode="  GLOBAL ",
        domains=["billing", " SEARCH "],
        artifact_types=["spec", "TICKET "],
        answer_style=" Table ",
        style_evidence="as a table",
        query_type="RELATION ",
        relation_evidence="depends on",
    )
    intent, _ = _classify(question, reply)
    assert intent.mode == "global"
    assert intent.domains == ("Billing", "Search")
    assert intent.types == ("Spec", "Ticket")
    assert intent.answer_style == "table"
    assert intent.query_type == "relation"
    assert intent.source == "llm"


# ---------------------------------------------------------------- domains and types


def test_domains_filter_unknown_and_duplicates_keep_first_order() -> None:
    reply = _reply(domains=["payments", "Unknown", "billing", "PAYMENTS", "Billing"])
    intent, _ = _classify("q", reply)
    assert intent.domains == ("Payments", "Billing")


def test_types_filter_unknown_and_duplicates_keep_first_order() -> None:
    reply = _reply(artifact_types=["ticket", "Diagram", "spec", "TICKET"])
    intent, _ = _classify("q", reply)
    assert intent.types == ("Ticket", "Spec")


def test_bare_string_counts_as_one_item_list() -> None:
    reply = _reply(domains="search", artifact_types="Spec")
    intent, _ = _classify("q", reply)
    assert intent.domains == ("Search",)
    assert intent.types == ("Spec",)


def test_bare_unknown_string_gives_empty() -> None:
    reply = _reply(domains="nowhere", artifact_types="Diagram")
    intent, _ = _classify("q", reply)
    assert intent.domains == ()
    assert intent.types == ()


# ---------------------------------------------------------------- answer style

STYLE_QUESTION = "List the billing specs as a neat table please"


@pytest.mark.parametrize(
    "evidence",
    [
        "neat table",  # 2 words
        "a neat table",  # 3 words
        "as a neat table",  # 4 words
        "AS A NEAT TABLE",  # case-insensitive
    ],
)
def test_style_kept_with_supported_evidence(evidence: str) -> None:
    reply = _reply(answer_style="table", style_evidence=evidence)
    intent, _ = _classify(STYLE_QUESTION, reply)
    assert intent.answer_style == "table"


def test_style_dropped_when_evidence_not_in_question() -> None:
    reply = _reply(answer_style="table", style_evidence="in a grid")
    intent, _ = _classify(STYLE_QUESTION, reply)
    assert intent.answer_style == "brief"


def test_style_dropped_when_evidence_missing() -> None:
    reply = _reply(answer_style="table")
    intent, _ = _classify(STYLE_QUESTION, reply)
    assert intent.answer_style == "brief"


@pytest.mark.parametrize(
    "evidence",
    [
        "",  # absent
        "table",  # 1 word
        "billing specs as a neat",  # 5 words
        "List the billing specs as a neat table please",  # many words
    ],
)
def test_style_dropped_with_wrong_evidence_length(evidence: str) -> None:
    reply = _reply(answer_style="table", style_evidence=evidence)
    intent, _ = _classify(STYLE_QUESTION, reply)
    assert intent.answer_style == "brief"


def test_unknown_style_becomes_default() -> None:
    reply = _reply(answer_style="poem", style_evidence="as a neat table")
    intent, _ = _classify(STYLE_QUESTION, reply)
    assert intent.answer_style == "brief"


# ---------------------------------------------------------------- query type

RELATION_QUESTION = "How does Search depend on Billing?"


@pytest.mark.parametrize("evidence", ["depend on", "DEPEND ON billing", "How does"])
def test_query_type_kept_with_evidence_in_question(evidence: str) -> None:
    reply = _reply(query_type="relation", relation_evidence=evidence)
    intent, _ = _classify(RELATION_QUESTION, reply)
    assert intent.query_type == "relation"


@pytest.mark.parametrize("evidence", [None, "", "   ", "is linked to"])
def test_query_type_dropped_without_supported_evidence(evidence: str | None) -> None:
    fields: dict[str, Any] = {"query_type": "relation"}
    if evidence is not None:
        fields["relation_evidence"] = evidence
    intent, _ = _classify(RELATION_QUESTION, _reply(**fields))
    assert intent.query_type == "lookup"


def test_unknown_query_type_becomes_default() -> None:
    reply = _reply(query_type="causal", relation_evidence="depend on")
    intent, _ = _classify(RELATION_QUESTION, reply)
    assert intent.query_type == "lookup"


# ---------------------------------------------------------------- enumerate


def test_enumerate_true_keeps_stripped_target_concept() -> None:
    reply = _reply(enumerate=True, target_concept="  payment methods  ")
    intent, _ = _classify("List every payment method", reply)
    assert intent.enumerate is True
    assert intent.target_concept == "payment methods"


@pytest.mark.parametrize("value", [False, "true", 1, None])
def test_enumerate_not_json_true_gives_false_and_no_concept(value: Any) -> None:
    reply = _reply(enumerate=value, target_concept="payment methods")
    intent, _ = _classify("List every payment method", reply)
    assert intent.enumerate is False
    assert intent.target_concept is None


def test_enumerate_missing_gives_false_and_no_concept() -> None:
    reply = _reply(target_concept="payment methods")
    intent, _ = _classify("List every payment method", reply)
    assert intent.enumerate is False
    assert intent.target_concept is None


# ---------------------------------------------------------------- draws


@pytest.mark.parametrize(
    "bad",
    [
        _error(),
        "this is not json at all",
        '["global"]',
        '"global"',
        '{"domains": ["Billing"]}',
        '{"mode": "sideways"}',
        '{"mode": null}',
    ],
)
def test_invalid_first_draw_runs_second(bad: str | Exception) -> None:
    question = "Which billing specs exist?"
    intent, llm = _classify(question, bad, _reply(mode="hybrid"))
    assert len(llm.calls) == 2
    assert intent.mode == "hybrid"
    assert intent.source == "llm"
    for call, (temperature, seed) in zip(llm.calls, DRAWS, strict=False):
        assert call["temperature"] == temperature
        assert call["seed"] == seed
        assert call["prompt"] == build_prompt(question, VOCAB, TEMPLATE)


def test_third_draw_wins_after_two_bad() -> None:
    intent, llm = _classify("q", _error(), "nope", _reply(mode="global"))
    assert len(llm.calls) == 3
    assert [(c["temperature"], c["seed"]) for c in llm.calls] == list(DRAWS)
    assert intent.mode == "global"
    assert intent.source == "llm"


def test_first_valid_draw_wins_and_stops() -> None:
    intent, llm = _classify("q", "bad", _reply(mode="global"), _reply(mode="hybrid"))
    assert len(llm.calls) == 2
    assert intent.mode == "global"


@pytest.mark.parametrize(
    "reply",
    [
        '```json\n{"mode": "global", "domains": ["Search"]}\n```',
        '{"mode": "global", "domains": ["Search"],}',
    ],
)
def test_repairable_reply_counts_as_valid(reply: str) -> None:
    intent, llm = _classify("q", reply)
    assert len(llm.calls) == 1
    assert intent.mode == "global"
    assert intent.domains == ("Search",)
    assert intent.source == "llm"


# ---------------------------------------------------------------- fallback


def test_all_draws_invalid_uses_fallback_unchanged() -> None:
    seen: list[str] = []
    special = Intent(
        mode="hybrid",
        domains=("Payments",),
        types=("Ticket",),
        answer_style="detailed",
        query_type="gap",
        enumerate=True,
        target_concept="refunds",
        source="fallback",
    )

    def fallback(question: str) -> Intent:
        seen.append(question)
        return special

    intent, llm = _classify("What about refunds?", _error(), "x", "[]", fallback=fallback)
    assert len(llm.calls) == len(DRAWS)
    assert [(c["temperature"], c["seed"]) for c in llm.calls] == list(DRAWS)
    assert seen == ["What about refunds?"]
    assert intent == special


def test_fallback_not_called_when_draw_valid() -> None:
    seen: list[str] = []

    def fallback(question: str) -> Intent:
        seen.append(question)
        return default_intent(VOCAB)

    intent, _ = _classify("q", _reply(mode="global"), fallback=fallback)
    assert seen == []
    assert intent.source == "llm"


def test_fallback_raising_gives_default_intent() -> None:
    def fallback(question: str) -> Intent:
        raise ValueError("broken fallback")

    intent, llm = _classify("q", "a", "b", "c", fallback=fallback)
    assert len(llm.calls) == len(DRAWS)
    assert intent == default_intent(VOCAB)


def test_no_fallback_gives_default_intent() -> None:
    intent, llm = _classify("q", _error(), _error(), _error())
    assert len(llm.calls) == len(DRAWS)
    assert intent == default_intent(VOCAB)
