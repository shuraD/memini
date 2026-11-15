"""Acceptance tests for LP-01: public value types of memini.infra.llm_providers."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Sequence
from typing import Any

import pytest

from memini.infra import llm_providers
from memini.infra.llm_providers import types as lp_types
from memini.infra.llm_providers.types import (
    LLM,
    ROLES,
    Attempt,
    BudgetExceeded,
    Embedder,
    Generation,
    LLMError,
    Reason,
)

PUBLIC_NAMES = (
    "ROLES",
    "Reason",
    "Generation",
    "Attempt",
    "LLMError",
    "BudgetExceeded",
    "LLM",
    "Embedder",
)


def _generation() -> Generation:
    return Generation(
        text="hello",
        provider="prov-a",
        model="model-x",
        role="answer",
        prompt_tokens=3,
        completion_tokens=5,
        seconds=0.25,
        cached=False,
    )


def _three_attempts() -> tuple[Attempt, ...]:
    return (
        Attempt(provider="alpha", reason=Reason.RATE_PER_MINUTE, detail="429 slow down"),
        Attempt(provider="beta", reason=Reason.TIMEOUT, detail="30s elapsed"),
        Attempt(provider="gamma", reason=Reason.NO_KEY, detail="env var missing"),
    )


# --- ROLES -----------------------------------------------------------------


def test_roles_are_exactly_the_six_names_in_order() -> None:
    assert ROLES == ("classifier", "filter", "answer", "extraction", "summary", "judge")
    assert isinstance(ROLES, tuple)


# --- Reason ----------------------------------------------------------------


def test_reason_is_enum_with_exact_members() -> None:
    assert issubclass(Reason, enum.Enum)
    assert [m.name for m in Reason] == [
        "RATE_PER_MINUTE",
        "EXHAUSTED",
        "SERVER_ERROR",
        "TIMEOUT",
        "REQUEST_REJECTED",
        "NO_KEY",
        "NO_MODEL",
        "BUDGET",
    ]


def test_reason_values_are_lower_case_member_names() -> None:
    assert Reason.RATE_PER_MINUTE.value == "rate_per_minute"
    for member in Reason:
        assert member.value == member.name.lower()


# --- Generation ------------------------------------------------------------


def test_generation_holds_its_fields() -> None:
    g = _generation()
    assert g.text == "hello"
    assert g.provider == "prov-a"
    assert g.model == "model-x"
    assert g.role == "answer"
    assert g.prompt_tokens == 3
    assert g.completion_tokens == 5
    assert g.seconds == 0.25
    assert g.cached is False


def test_generation_field_order() -> None:
    assert [f.name for f in dataclasses.fields(Generation)] == [
        "text",
        "provider",
        "model",
        "role",
        "prompt_tokens",
        "completion_tokens",
        "seconds",
        "cached",
    ]


def test_generation_is_frozen() -> None:
    g = _generation()
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.text = "changed"  # type: ignore[misc]
    assert g.text == "hello"


# --- Attempt ---------------------------------------------------------------


def test_attempt_holds_fields_and_is_frozen() -> None:
    a = Attempt(provider="alpha", reason=Reason.EXHAUSTED, detail="quota")
    assert (a.provider, a.reason, a.detail) == ("alpha", Reason.EXHAUSTED, "quota")
    assert [f.name for f in dataclasses.fields(Attempt)] == ["provider", "reason", "detail"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.detail = "x"  # type: ignore[misc]


# --- LLMError --------------------------------------------------------------


def test_llm_error_is_exception_and_stores_attempts_and_role() -> None:
    attempts = _three_attempts()
    err = LLMError(attempts=attempts, role="judge")
    assert isinstance(err, Exception)
    assert isinstance(err.attempts, tuple)
    assert err.attempts == attempts
    assert err.role == "judge"


def test_llm_error_positional_construction() -> None:
    attempts = _three_attempts()
    err = LLMError(attempts, "answer")
    assert err.attempts == attempts
    assert err.role == "answer"


def test_llm_error_str_names_every_provider_and_reason() -> None:
    err = LLMError(attempts=_three_attempts(), role="answer")
    text = str(err)
    for name in ("alpha", "beta", "gamma"):
        assert name in text
    for value in ("rate_per_minute", "timeout", "no_key"):
        assert value in text


def test_llm_error_str_uses_exact_attempt_format_in_order() -> None:
    err = LLMError(attempts=_three_attempts(), role="answer")
    text = str(err)
    parts = [
        "alpha: rate_per_minute (429 slow down)",
        "beta: timeout (30s elapsed)",
        "gamma: no_key (env var missing)",
    ]
    positions = [text.find(p) for p in parts]
    assert all(pos >= 0 for pos in positions), text
    assert positions == sorted(positions), text


def test_llm_error_can_be_raised_and_caught() -> None:
    with pytest.raises(LLMError) as info:
        raise LLMError(attempts=_three_attempts(), role="summary")
    assert info.value.role == "summary"


# --- BudgetExceeded --------------------------------------------------------


def test_budget_exceeded_is_llm_error() -> None:
    assert issubclass(BudgetExceeded, LLMError)
    attempts = (Attempt(provider="alpha", reason=Reason.BUDGET, detail="cap 1.00 reached"),)
    err = BudgetExceeded(attempts=attempts, role="extraction")
    assert isinstance(err, LLMError)
    assert isinstance(err, Exception)
    assert err.attempts == attempts
    assert err.role == "extraction"
    assert "alpha: budget (cap 1.00 reached)" in str(err)


def test_budget_exceeded_caught_as_llm_error() -> None:
    with pytest.raises(LLMError):
        raise BudgetExceeded(attempts=_three_attempts(), role="answer")


# --- Protocols -------------------------------------------------------------


class _FakeLLM:
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
        return _generation()

    def generate_many(self, prompts: Sequence[str], **kwargs: Any) -> list[Generation | LLMError]:
        return [self.generate(p, **kwargs) for p in prompts]


class _OnlyGenerate:
    def generate(self, prompt: str, **kwargs: Any) -> Generation:
        return _generation()


class _FakeEmbedder:
    def embed(self, texts: Sequence[str], *, kind: str) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]


def test_fake_with_both_methods_is_an_llm() -> None:
    assert isinstance(_FakeLLM(), LLM)


def test_object_missing_generate_many_is_not_an_llm() -> None:
    assert not isinstance(_OnlyGenerate(), LLM)
    assert not isinstance(object(), LLM)


def test_fake_embedder_is_an_embedder() -> None:
    assert isinstance(_FakeEmbedder(), Embedder)
    assert not isinstance(object(), Embedder)
    assert not isinstance(_FakeEmbedder(), LLM)


# --- Re-exports ------------------------------------------------------------


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_package_reexports_same_objects_as_types(name: str) -> None:
    assert hasattr(llm_providers, name)
    assert getattr(llm_providers, name) is getattr(lp_types, name)
