"""Acceptance tests for LT-06: an input budget derived from the configured limits."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from memini.infra.llm_providers.config import ConfigError, LLMConfig, parse_config
from memini.infra.llm_providers.input_budget import InputBudget, PromptTooLong, input_budget


def config(
    provider_limits: dict[str, Any], model_limits: dict[str, Any] | None = None
) -> LLMConfig:
    model: dict[str, Any] = {"name": "local-summary-model"}
    if model_limits is not None:
        model["limits"] = model_limits
    raw: dict[str, Any] = {
        "providers": [
            {
                "name": "local",
                "kind": "ollama",
                "base_url": "http://localhost:11434",
                "limits": provider_limits,
                "models": {"summary": model},
            }
        ]
    }
    return parse_config(raw)


def budget() -> InputBudget:
    cfg = config({"num_ctx": 32768, "max_tokens": 8192})
    return input_budget(cfg, "local", "summary", 3)


def test_input_budget_reads_the_configured_limits() -> None:
    b = budget()
    assert isinstance(b, InputBudget)
    assert b.num_ctx == 32768
    assert b.num_predict == 8192
    assert b.chars_per_token == 3
    assert b.input_tokens == 24576


def test_tokens_rounds_up() -> None:
    b = budget()
    assert b.tokens("abcd") == 2
    assert b.tokens("abc") == 1
    assert b.tokens("") == 0


def test_tokens_with_fractional_chars_per_token() -> None:
    b = InputBudget(num_ctx=100, num_predict=10, chars_per_token=2.5)
    assert b.tokens("abcdef") == 3  # ceil(6 / 2.5)
    assert b.input_tokens == 90


def test_role_model_limits_override_provider_limits() -> None:
    cfg = config({"num_ctx": 32768, "max_tokens": 8192}, {"num_ctx": 16384, "max_tokens": 4096})
    b = input_budget(cfg, "local", "summary", 3)
    assert b.num_ctx == 16384
    assert b.num_predict == 4096
    assert b.input_tokens == 12288


def test_role_model_overrides_only_the_keys_it_sets() -> None:
    cfg = config({"num_ctx": 32768, "max_tokens": 8192}, {"num_ctx": 20000})
    b = input_budget(cfg, "local", "summary", 3)
    assert b.num_ctx == 20000
    assert b.num_predict == 8192
    assert b.input_tokens == 11808


def test_prompt_at_the_limit_fits() -> None:
    b = budget()
    prompt = "x" * (24576 * 3)
    assert b.fits(prompt) is True
    assert b.check(prompt) is None


def test_prompt_one_char_over_the_limit_does_not_fit() -> None:
    b = budget()
    prompt = "x" * (24576 * 3 + 1)
    assert b.fits(prompt) is False
    with pytest.raises(PromptTooLong) as info:
        b.check(prompt)
    assert info.value.tokens == 24577
    assert info.value.limit == 24576


def test_prompt_too_long_is_a_value_error() -> None:
    assert issubclass(PromptTooLong, ValueError)
    b = budget()
    with pytest.raises(ValueError, match=r"."):
        b.check("x" * (24576 * 3 + 10))


def test_input_budget_is_frozen() -> None:
    b = budget()
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.num_ctx = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "limits",
    [
        {"num_ctx": 8192, "max_tokens": 8192},
        {"num_ctx": 4096, "max_tokens": 8192},
        {"max_tokens": 8192},
        {"num_ctx": 32768},
        {},
    ],
)
def test_bad_limits_raise_config_error(limits: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        input_budget(config(limits), "local", "summary", 3)


@pytest.mark.parametrize("chars_per_token", [0, 0.0, -1, -2.5])
def test_non_positive_chars_per_token_raises_config_error(chars_per_token: float) -> None:
    cfg = config({"num_ctx": 32768, "max_tokens": 8192})
    with pytest.raises(ConfigError):
        input_budget(cfg, "local", "summary", chars_per_token)
