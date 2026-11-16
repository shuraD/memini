"""Resolve settings (environment, then config file, then default) and provider keys.

Only the ``env`` mapping a caller passes is read, never the process environment, so the
caller decides where variables come from. A provider's key is asked for only when that
provider is configured. The default provider chain ships as package data next to this
module and is validated by :func:`memini.infra.llm_providers.config.parse_config`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from memini.infra.llm_providers.config import ConfigError, LLMConfig, ProviderSpec, parse_config

DEFAULT_PROVIDERS_PATH: Path = Path(__file__).resolve().parent / "default_providers.yaml"
"""The shipped default provider chain."""

_ABSENT = object()


class MissingKeyError(ConfigError):
    """A configured provider's key variable is missing or empty; the message names the variable."""

    def __init__(self, variable: str, provider: str) -> None:
        """Record the missing ``variable`` and the ``provider`` that needs it."""
        super().__init__(f"provider {provider!r} needs its key in variable {variable!r}")
        self.variable = variable
        self.provider = provider


def load_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return the YAML mapping at ``path``; a missing or empty file gives ``{}``."""
    file = Path(path)
    if not file.exists():
        return {}
    data = yaml.safe_load(file.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{file}: top level must be a mapping, got {type(data).__name__}")
    return data


def _lookup(file: Mapping[str, Any], key: str) -> Any:
    node: Any = file
    for part in key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return _ABSENT
        node = node[part]
    return _ABSENT if node is None else node


def _cast(cast: Callable[[Any], Any] | None, value: Any, where: str) -> Any:
    if cast is None:
        return value
    try:
        return cast(value)
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"{where}: invalid value ({exc})") from exc


def setting(
    file: Mapping[str, Any],
    key: str,
    *,
    env: Mapping[str, str],
    env_var: str | None,
    default: Any,
    cast: Callable[[Any], Any] | None = None,
) -> Any:
    """Return the non-empty ``env[env_var]``, else the file value at dotted ``key``, else default.

    ``cast`` applies to the environment or file value, never to ``default``; a cast failure
    raises :class:`ConfigError` naming the variable or the key.
    """
    if env_var is not None:
        raw = env.get(env_var)
        if raw:
            return _cast(cast, raw, f"variable {env_var}")
    value = _lookup(file, key)
    if value is not _ABSENT:
        return _cast(cast, value, f"setting {key}")
    return default


def required_key_vars(config: LLMConfig) -> tuple[str, ...]:
    """Return each configured provider's key variable once, in config order."""
    names: dict[str, None] = {}
    for spec in config.providers:
        if spec.key_env is not None:
            names.setdefault(spec.key_env, None)
    return tuple(names)


def provider_key(spec: ProviderSpec, env: Mapping[str, str]) -> str | None:
    """Return the provider's key from ``env``; None when it needs none.

    Raises :class:`MissingKeyError` when the key variable is missing or empty.
    """
    if spec.key_env is None:
        return None
    value = env.get(spec.key_env)
    if not value:
        raise MissingKeyError(variable=spec.key_env, provider=spec.name)
    return value


def load_default_config() -> LLMConfig:
    """Parse and return the shipped default provider chain."""
    return parse_config(load_file(DEFAULT_PROVIDERS_PATH))
