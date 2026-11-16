"""Acceptance tests for LT-07: settings resolution, provider keys and the default chain."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from memini.infra.llm_providers import settings as lp_settings
from memini.infra.llm_providers.config import ConfigError, LLMConfig, parse_config
from memini.infra.llm_providers.types import ROLES

VAR = "LT07_TEST_SETTING_VAR"
FILE: dict[str, Any] = {"sec": {"k": "from-file"}}


def _provider(name: str, key_env: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": name,
        "kind": "openai_compatible" if key_env else "ollama",
        "base_url": "https://x.example.test/v1" if key_env else "http://localhost:11434",
        "models": {"answer": {"name": f"{name}-model"}},
    }
    if key_env is not None:
        body["key_env"] = key_env
    return body


def _config(*providers: dict[str, Any]) -> LLMConfig:
    return parse_config({"providers": list(providers)})


# --- setting -------------------------------------------------------------------------


def test_setting_env_wins_over_file() -> None:
    got = lp_settings.setting(FILE, "sec.k", env={VAR: "from-env"}, env_var=VAR, default="d")
    assert got == "from-env"


@pytest.mark.parametrize("env", [{}, {VAR: ""}])
def test_setting_file_when_env_absent_or_empty(env: dict[str, str]) -> None:
    got = lp_settings.setting(FILE, "sec.k", env=env, env_var=VAR, default="d")
    assert got == "from-file"


def test_setting_env_var_none_ignores_env_mapping() -> None:
    got = lp_settings.setting(FILE, "sec.k", env={VAR: "from-env"}, env_var=None, default="d")
    assert got == "from-file"


@pytest.mark.parametrize(
    "file",
    [{}, {"sec": None}, {"sec": "scalar"}, {"sec": {"k": None}}],
)
def test_setting_default_object_returned_uncast(file: dict[str, Any]) -> None:
    default = object()
    calls: list[Any] = []

    def cast(value: Any) -> Any:
        calls.append(value)
        return "cast"

    got = lp_settings.setting(file, "sec.k", env={}, env_var=VAR, default=default, cast=cast)
    assert got is default
    assert calls == []


def test_setting_default_list_identity_without_cast() -> None:
    default: list[int] = [1, 2]
    got = lp_settings.setting({}, "a.b.c", env={}, env_var=None, default=default)
    assert got is default


def test_setting_cast_applies_to_env_and_file() -> None:
    from_env = lp_settings.setting(
        {"sec": {"k": "7"}}, "sec.k", env={VAR: "42"}, env_var=VAR, default=0, cast=int
    )
    from_file = lp_settings.setting(
        {"sec": {"k": "7"}}, "sec.k", env={}, env_var=VAR, default=0, cast=int
    )
    assert from_env == 42
    assert from_file == 7


def test_setting_file_value_returned_itself_without_cast() -> None:
    value = {"nested": [1, 2]}
    got = lp_settings.setting({"sec": {"k": value}}, "sec.k", env={}, env_var=VAR, default=None)
    assert got is value


def test_setting_bad_env_cast_names_env_var() -> None:
    with pytest.raises(ConfigError) as info:
        lp_settings.setting(FILE, "sec.k", env={VAR: "abc"}, env_var=VAR, default=0, cast=int)
    assert VAR in str(info.value)


def test_setting_bad_file_cast_names_key() -> None:
    with pytest.raises(ConfigError) as info:
        lp_settings.setting(
            {"sec": {"k": "abc"}}, "sec.k", env={}, env_var=VAR, default=0, cast=int
        )
    assert "sec.k" in str(info.value)


def test_setting_cast_type_error_is_config_error() -> None:
    with pytest.raises(ConfigError) as info:
        lp_settings.setting(
            {"sec": {"k": [1, 2]}}, "sec.k", env={}, env_var=VAR, default=0, cast=int
        )
    assert "sec.k" in str(info.value)


def test_setting_top_level_key_without_dots() -> None:
    got = lp_settings.setting({"run_id": "r1"}, "run_id", env={}, env_var=None, default="x")
    assert got == "r1"


def test_setting_does_not_read_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(VAR, "from-process")
    got_file = lp_settings.setting(FILE, "sec.k", env={}, env_var=VAR, default="d")
    got_default = lp_settings.setting({}, "sec.k", env={}, env_var=VAR, default="d")
    assert got_file == "from-file"
    assert got_default == "d"


def test_setting_does_not_call_os_getenv(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []
    real_getenv = os.getenv

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return real_getenv(*args, **kwargs)

    monkeypatch.setattr(os, "getenv", spy)
    got = lp_settings.setting(FILE, "sec.k", env={}, env_var=VAR, default="d")
    monkeypatch.undo()
    assert got == "from-file"
    assert calls == []


# --- load_file -----------------------------------------------------------------------


def test_load_file_missing_returns_empty(tmp_path: Path) -> None:
    assert lp_settings.load_file(tmp_path / "nope.yaml") == {}


def test_load_file_empty_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert lp_settings.load_file(path) == {}


def test_load_file_mapping(tmp_path: Path) -> None:
    path = tmp_path / "conf.yaml"
    path.write_text("sec:\n  k: v\n  n: 3\n", encoding="utf-8")
    assert lp_settings.load_file(path) == {"sec": {"k": "v", "n": 3}}


def test_load_file_accepts_str_path(tmp_path: Path) -> None:
    path = tmp_path / "conf.yaml"
    path.write_text("a: 1\n", encoding="utf-8")
    assert lp_settings.load_file(str(path)) == {"a": 1}


def test_load_file_list_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        lp_settings.load_file(path)


# --- required_key_vars ---------------------------------------------------------------


def test_required_key_vars_order_and_dedup() -> None:
    config = _config(
        _provider("p1", "KEY_A"),
        _provider("p2", None),
        _provider("p3", "KEY_B"),
        _provider("p4", "KEY_A"),
    )
    assert lp_settings.required_key_vars(config) == ("KEY_A", "KEY_B")


def test_required_key_vars_keyless_only() -> None:
    config = _config(_provider("local1", None), _provider("local2", None))
    assert lp_settings.required_key_vars(config) == ()


# --- provider_key --------------------------------------------------------------------


def test_provider_key_keyless_is_none() -> None:
    spec = _config(_provider("local", None)).providers[0]
    assert lp_settings.provider_key(spec, {}) is None


def test_provider_key_returns_env_value() -> None:
    spec = _config(_provider("hosted", "HOSTED_KEY")).providers[0]
    assert lp_settings.provider_key(spec, {"HOSTED_KEY": "test-key"}) == "test-key"


@pytest.mark.parametrize("env", [{}, {"HOSTED_KEY": ""}, {"OTHER": "test-key"}])
def test_provider_key_missing_raises(env: dict[str, str]) -> None:
    spec = _config(_provider("hosted", "HOSTED_KEY")).providers[0]
    with pytest.raises(lp_settings.MissingKeyError) as info:
        lp_settings.provider_key(spec, env)
    err = info.value
    assert isinstance(err, ConfigError)
    assert err.variable == "HOSTED_KEY"
    assert err.provider == "hosted"
    assert "HOSTED_KEY" in str(err)


def test_missing_key_error_message_has_no_key_value() -> None:
    spec = _config(_provider("hosted", "HOSTED_KEY")).providers[0]
    with pytest.raises(lp_settings.MissingKeyError) as info:
        lp_settings.provider_key(spec, {"OTHER": "test-key"})
    assert "test-key" not in str(info.value)


def test_missing_key_error_is_config_error_subclass() -> None:
    assert issubclass(lp_settings.MissingKeyError, ConfigError)


def test_provider_key_does_not_read_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOSTED_KEY", "test-key")
    spec = _config(_provider("hosted", "HOSTED_KEY")).providers[0]
    with pytest.raises(lp_settings.MissingKeyError):
        lp_settings.provider_key(spec, {})


def test_unconfigured_provider_key_never_required() -> None:
    config = _config(_provider("local", None))
    keys = [lp_settings.provider_key(spec, {}) for spec in config.providers]
    assert keys == [None]


# --- default chain -------------------------------------------------------------------


def test_default_config_shape() -> None:
    config = lp_settings.load_default_config()
    assert isinstance(config, LLMConfig)
    providers = config.providers
    assert len(providers) >= 2
    for spec in providers[:-1]:
        assert spec.key_env, spec.name
        assert spec.base_url.startswith("https://"), spec.name
    last = providers[-1]
    assert last.key_env is None
    assert last.base_url.startswith(("http://localhost:", "http://127.0.0.1:"))
    assert sum(1 for spec in providers if spec.key_env is None) == 1


def test_default_config_serves_every_role() -> None:
    config = lp_settings.load_default_config()
    for spec in config.providers:
        for role in ROLES:
            assert config.model_for(spec.name, role), (spec.name, role)


def test_default_config_required_keys() -> None:
    config = lp_settings.load_default_config()
    required = lp_settings.required_key_vars(config)
    assert required
    hosted = {spec.key_env for spec in config.providers[:-1]}
    assert hosted == set(required)
    assert lp_settings.provider_key(config.providers[-1], {}) is None


def test_default_providers_path_location() -> None:
    path = lp_settings.DEFAULT_PROVIDERS_PATH
    assert isinstance(path, Path)
    assert path.name == "default_providers.yaml"
    assert path.exists()
    assert path.parent.resolve() == Path(lp_settings.__file__).resolve().parent


def test_default_providers_file_parses_like_default_config() -> None:
    parsed = parse_config(lp_settings.load_file(lp_settings.DEFAULT_PROVIDERS_PATH))
    names = [spec.name for spec in parsed.providers]
    assert names == [spec.name for spec in lp_settings.load_default_config().providers]
