"""Acceptance tests for LP-02: parsing the providers config into immutable settings."""

from __future__ import annotations

import copy
import os
from typing import Any

import pytest

from memini.infra.llm_providers import config as lp_config
from memini.infra.llm_providers.types import ROLES

ConfigError = lp_config.ConfigError
parse_config = lp_config.parse_config


def _raw() -> dict[str, Any]:
    """Return a fresh, valid raw mapping as yaml.safe_load would produce it."""
    return {
        "cache_path": "tmp/cache.sqlite",
        "run_id": "run-7",
        "budget": {"max_tokens_per_run": 100000},
        "embedding_prefixes": {"query": "q: ", "document": "d: "},
        "providers": [
            {
                "name": "alpha",
                "kind": "openai_compatible",
                "key_env": "ALPHA_KEY_VAR",
                "base_url": "https://alpha.example.test/v1",
                "limits": {"max_tokens": 512, "timeout_s": 30},
                "exhausted_markers": ["quota gone"],
                "models": {
                    "answer": {
                        "name": "alpha-answer",
                        "limits": {"max_tokens": 256},
                        "price": {"input_per_mtok": 0.5, "output_per_mtok": 1.5},
                    },
                    "classifier": {"name": "alpha-classifier"},
                    "embedding": {"name": "alpha-embed"},
                },
            },
            {
                "name": "beta",
                "kind": "gemini",
                "key_env": "BETA_KEY_VAR",
                "base_url": "https://beta.example.test",
                "limits": {"max_concurrent": 4},
                "rest_default_s": 12.5,
                "models": {
                    "answer": {"name": "beta-answer"},
                    "classifier": {"name": "beta-classifier"},
                },
            },
            {
                "name": "gamma",
                "kind": "ollama",
                "key_env": None,
                "base_url": "http://localhost:11434",
                "limits": {},
                "models": {
                    "answer": {"name": "gamma-answer"},
                    "classifier": {"name": "gamma-classifier"},
                },
            },
        ],
        "roles": {
            "answer": {
                "providers": ["gamma", "alpha"],
                "max_calls_per_question": 3,
                "max_tokens_per_run": 5000,
            },
        },
    }


def _minimal() -> dict[str, Any]:
    """Return a valid mapping with one provider and no optional top-level keys."""
    return {
        "providers": [
            {
                "name": "solo",
                "kind": "ollama",
                "base_url": "http://localhost:11434",
                "limits": {},
                "models": {"answer": {"name": "solo-answer"}},
            }
        ],
    }


# --- provider order ---------------------------------------------------------------


def test_role_with_own_providers_list_gets_that_order() -> None:
    cfg = parse_config(_raw())
    assert tuple(cfg.provider_order("answer")) == ("gamma", "alpha")


def test_role_without_own_list_gets_global_order() -> None:
    cfg = parse_config(_raw())
    assert tuple(cfg.provider_order("classifier")) == ("alpha", "beta", "gamma")


def test_provider_order_returns_a_tuple() -> None:
    cfg = parse_config(_raw())
    assert isinstance(cfg.provider_order("answer"), tuple)
    assert isinstance(cfg.provider_order("classifier"), tuple)


# --- limits -----------------------------------------------------------------------


def test_model_limit_overrides_provider_limit() -> None:
    cfg = parse_config(_raw())
    assert cfg.limit("alpha", "answer", "max_tokens") == 256


def test_provider_limit_applies_to_other_role() -> None:
    cfg = parse_config(_raw())
    assert cfg.limit("alpha", "classifier", "max_tokens") == 512


def test_provider_only_limit_is_seen_through_model() -> None:
    cfg = parse_config(_raw())
    assert cfg.limit("alpha", "answer", "timeout_s") == 30
    assert cfg.limit("beta", "answer", "max_concurrent") == 4


def test_missing_limit_returns_default() -> None:
    cfg = parse_config(_raw())
    assert cfg.limit("beta", "answer", "max_tokens") is None
    assert cfg.limit("beta", "answer", "batch_size", default=16) == 16


def test_limit_keys_are_free_form() -> None:
    raw = _raw()
    raw["providers"][1]["limits"]["some_new_knob"] = "x"
    cfg = parse_config(raw)
    assert cfg.limit("beta", "classifier", "some_new_knob") == "x"


# --- models -----------------------------------------------------------------------


def test_model_for_returns_configured_name() -> None:
    cfg = parse_config(_raw())
    assert cfg.model_for("alpha", "answer") == "alpha-answer"
    assert cfg.model_for("beta", "classifier") == "beta-classifier"


def test_model_for_provider_without_model_for_role_is_none() -> None:
    cfg = parse_config(_raw())
    assert cfg.model_for("beta", "summary") is None
    assert cfg.model_for("gamma", "judge") is None


def test_model_for_embedding_returns_embedding_model() -> None:
    cfg = parse_config(_raw())
    assert cfg.model_for("alpha", "embedding") == "alpha-embed"
    assert cfg.model_for("beta", "embedding") is None


def test_embedding_prefixes_returned_when_given() -> None:
    cfg = parse_config(_raw())
    prefixes = cfg.embedding_prefixes
    assert prefixes is not None
    assert dict(prefixes) == {"query": "q: ", "document": "d: "}


def test_embedding_prefixes_none_when_absent() -> None:
    cfg = parse_config(_minimal())
    assert cfg.embedding_prefixes is None


# --- prices -----------------------------------------------------------------------


def test_price_returns_input_output_pair() -> None:
    cfg = parse_config(_raw())
    price = cfg.price("alpha", "answer")
    assert price is not None
    assert tuple(price) == (0.5, 1.5)


def test_price_none_when_not_configured() -> None:
    cfg = parse_config(_raw())
    assert cfg.price("beta", "answer") is None
    assert cfg.price("alpha", "classifier") is None


# --- budgets ----------------------------------------------------------------------


def test_role_budget_from_roles_section() -> None:
    cfg = parse_config(_raw())
    budget = cfg.role_budget("answer")
    assert budget.max_calls_per_question == 3
    assert budget.max_tokens_per_run == 5000


def test_global_max_tokens_per_run() -> None:
    cfg = parse_config(_raw())
    assert cfg.global_max_tokens_per_run == 100000


# --- top-level defaults and provider specs ----------------------------------------


def test_defaults_for_cache_path_and_run_id() -> None:
    cfg = parse_config(_minimal())
    assert str(cfg.cache_path) == ".memini/llm-cache.sqlite"
    assert cfg.run_id == "default"


def test_explicit_cache_path_and_run_id_kept() -> None:
    cfg = parse_config(_raw())
    assert str(cfg.cache_path) == "tmp/cache.sqlite"
    assert cfg.run_id == "run-7"


def test_providers_in_config_order() -> None:
    cfg = parse_config(_raw())
    assert isinstance(cfg.providers, tuple)
    assert [p.name for p in cfg.providers] == ["alpha", "beta", "gamma"]


def test_provider_lookup_by_name() -> None:
    cfg = parse_config(_raw())
    spec = cfg.provider("beta")
    assert spec.name == "beta"
    assert spec.kind == "gemini"
    assert spec.base_url == "https://beta.example.test"


def test_rest_default_s_default_and_override() -> None:
    cfg = parse_config(_raw())
    assert cfg.provider("alpha").rest_default_s == 60
    assert cfg.provider("beta").rest_default_s == 12.5


# --- immutability -----------------------------------------------------------------


def test_config_is_immutable() -> None:
    cfg = parse_config(_raw())
    with pytest.raises((AttributeError, TypeError)):
        cfg.run_id = "other"  # type: ignore[misc]


def test_provider_spec_is_immutable() -> None:
    cfg = parse_config(_raw())
    spec = cfg.provider("alpha")
    with pytest.raises((AttributeError, TypeError)):
        spec.name = "other"  # type: ignore[misc]


def test_mutating_raw_after_parse_does_not_change_config() -> None:
    raw = _raw()
    cfg = parse_config(raw)
    raw["providers"][0]["limits"]["max_tokens"] = 1
    raw["providers"][0]["models"]["classifier"]["name"] = "changed"
    raw["roles"]["answer"]["providers"].append("beta")
    raw["run_id"] = "changed"
    assert cfg.limit("alpha", "classifier", "max_tokens") == 512
    assert cfg.model_for("alpha", "classifier") == "alpha-classifier"
    assert tuple(cfg.provider_order("answer")) == ("gamma", "alpha")
    assert cfg.run_id == "run-7"


def test_parse_does_not_mutate_input() -> None:
    raw = _raw()
    snapshot = copy.deepcopy(raw)
    parse_config(raw)
    assert raw == snapshot


# --- errors -----------------------------------------------------------------------


def test_config_error_is_value_error() -> None:
    assert issubclass(ConfigError, ValueError)


def test_unknown_models_key_rejected_and_named() -> None:
    raw = _raw()
    raw["providers"][1]["models"]["poetry_writer"] = {"name": "beta-poet"}
    with pytest.raises(ConfigError) as info:
        parse_config(raw)
    assert "poetry_writer" in str(info.value)


def test_unknown_kind_rejected_and_named() -> None:
    raw = _raw()
    raw["providers"][2]["kind"] = "carrier_pigeon"
    with pytest.raises(ConfigError) as info:
        parse_config(raw)
    assert "carrier_pigeon" in str(info.value)


def test_duplicate_provider_name_rejected_and_named() -> None:
    raw = _raw()
    dup = copy.deepcopy(raw["providers"][1])
    dup["name"] = "alpha"
    raw["providers"].append(dup)
    with pytest.raises(ConfigError) as info:
        parse_config(raw)
    assert "alpha" in str(info.value)


def test_role_order_naming_unknown_provider_rejected_and_named() -> None:
    raw = _raw()
    raw["roles"]["answer"]["providers"] = ["gamma", "nowhere_provider"]
    with pytest.raises(ConfigError) as info:
        parse_config(raw)
    assert "nowhere_provider" in str(info.value)


def test_every_known_role_is_accepted_as_models_key() -> None:
    raw = _minimal()
    raw["providers"][0]["models"] = {role: {"name": f"m-{role}"} for role in ROLES}
    raw["providers"][0]["models"]["embedding"] = {"name": "m-embed"}
    cfg = parse_config(raw)
    for role in ROLES:
        assert cfg.model_for("solo", role) == f"m-{role}"
    assert cfg.model_for("solo", "embedding") == "m-embed"


# --- key_env and the environment --------------------------------------------------


def test_key_env_stores_only_the_variable_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHA_KEY_VAR", "sk-secret-value-123")
    cfg = parse_config(_raw())
    assert cfg.provider("alpha").key_env == "ALPHA_KEY_VAR"
    assert cfg.provider("gamma").key_env is None
    assert "sk-secret-value-123" not in repr(cfg)
    assert "sk-secret-value-123" not in repr(cfg.provider("alpha"))
    assert "sk-secret-value-123" not in repr(cfg.providers)


def test_parse_never_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []
    real_getenv = os.getenv

    def recording_getenv(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return real_getenv(*args, **kwargs)

    raw = _raw()
    with monkeypatch.context() as patch:
        patch.setattr(os, "getenv", recording_getenv)
        cfg = parse_config(raw)
    assert calls == []
    assert cfg.provider("beta").key_env == "BETA_KEY_VAR"
