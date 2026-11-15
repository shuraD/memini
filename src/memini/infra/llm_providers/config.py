"""Parse the providers config mapping into immutable settings.

The caller loads the YAML and passes the resulting mapping to :func:`parse_config`; this
module never reads files, the network or the environment. A provider's ``key_env`` is kept
as the variable *name* only. Limit keys are free-form: a model's limits override its
provider's limits, and the parser knows no provider by name.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from memini.infra.llm_providers.types import ROLES

KINDS: frozenset[str] = frozenset({"openai_compatible", "gemini", "ollama"})
"""The provider kinds the client layer knows how to talk to."""

EMBEDDING: str = "embedding"
"""The one role-less models key: the provider's embedding model."""

DEFAULT_CACHE_PATH: str = ".memini/llm-cache.sqlite"
DEFAULT_RUN_ID: str = "default"
DEFAULT_REST_S: float = 60.0


class ConfigError(ValueError):
    """Raised when the providers config is malformed; the message names the offender."""


@dataclass(frozen=True)
class ModelSpec:
    """One model a provider serves for a role (or for embedding)."""

    name: str
    limits: Mapping[str, Any]
    price: tuple[float, float] | None


@dataclass(frozen=True)
class ProviderSpec:
    """One configured provider; ``key_env`` is the name of the variable holding its key."""

    name: str
    kind: str
    key_env: str | None
    base_url: str
    limits: Mapping[str, Any]
    rest_default_s: float
    exhausted_markers: tuple[str, ...] | None
    models: Mapping[str, ModelSpec]


@dataclass(frozen=True)
class RoleBudget:
    """Spending caps of one role; ``None`` means uncapped."""

    max_calls_per_question: int | None
    max_tokens_per_run: int | None


@dataclass(frozen=True)
class RoleSpec:
    """Per-role settings: an optional provider order and the role's budget."""

    providers: tuple[str, ...] | None
    budget: RoleBudget


@dataclass(frozen=True)
class LLMConfig:
    """Immutable, validated provider settings with role-level resolution helpers."""

    cache_path: str
    run_id: str
    global_max_tokens_per_run: int | None
    embedding_prefixes: Mapping[str, str] | None
    providers: tuple[ProviderSpec, ...]
    roles: Mapping[str, RoleSpec]

    def provider(self, name: str) -> ProviderSpec:
        """Return the provider called ``name``; raise ``KeyError`` if there is none."""
        for spec in self.providers:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def provider_order(self, role: str) -> tuple[str, ...]:
        """Return the role's own provider order, else every provider in config order."""
        spec = self.roles.get(role)
        if spec is not None and spec.providers is not None:
            return spec.providers
        return tuple(p.name for p in self.providers)

    def _model(self, provider: str, role: str) -> ModelSpec | None:
        return self.provider(provider).models.get(role)

    def model_for(self, provider: str, role: str) -> str | None:
        """Return the model ``provider`` serves for ``role`` (or ``'embedding'``), else None."""
        model = self._model(provider, role)
        return None if model is None else model.name

    def limit(self, provider: str, role: str, key: str, default: Any = None) -> Any:
        """Return limit ``key``: the role's model value, else the provider's, else default."""
        model = self._model(provider, role)
        if model is not None and key in model.limits:
            return model.limits[key]
        return self.provider(provider).limits.get(key, default)

    def price(self, provider: str, role: str) -> tuple[float, float] | None:
        """Return ``(input_per_mtok, output_per_mtok)`` of the role's model, else None."""
        model = self._model(provider, role)
        return None if model is None else model.price

    def role_budget(self, role: str) -> RoleBudget:
        """Return the role's caps; a role without a section is uncapped."""
        spec = self.roles.get(role)
        return RoleBudget(None, None) if spec is None else spec.budget


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where} must be a non-empty string, got {value!r}")
    return value


def _opt_int(value: Any, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where} must be an integer or null, got {value!r}")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{where} must be a number, got {value!r}")
    return float(value)


def _frozen(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(copy.deepcopy(dict(value)))


def _parse_model(raw: Any, where: str) -> ModelSpec:
    body = _mapping(raw, where)
    price_raw = body.get("price")
    price: tuple[float, float] | None = None
    if price_raw is not None:
        p = _mapping(price_raw, f"{where}.price")
        price = (
            _number(p.get("input_per_mtok"), f"{where}.price.input_per_mtok"),
            _number(p.get("output_per_mtok"), f"{where}.price.output_per_mtok"),
        )
    return ModelSpec(
        name=_str(body.get("name"), f"{where}.name"),
        limits=_frozen(_mapping(body.get("limits"), f"{where}.limits")),
        price=price,
    )


def _parse_provider(raw: Any, index: int) -> ProviderSpec:
    body = _mapping(raw, f"providers[{index}]")
    name = _str(body.get("name"), f"providers[{index}].name")
    where = f"provider {name!r}"
    kind = body.get("kind")
    if kind not in KINDS:
        raise ConfigError(f"{where}: unknown kind {kind!r} (expected one of {sorted(KINDS)})")
    key_env = body.get("key_env")
    if key_env is not None:
        key_env = _str(key_env, f"{where}.key_env")
    models: dict[str, ModelSpec] = {}
    for key, model_raw in _mapping(body.get("models"), f"{where}.models").items():
        if key not in ROLES and key != EMBEDDING:
            raise ConfigError(f"{where}: unknown models key {key!r} (not a role nor {EMBEDDING!r})")
        models[key] = _parse_model(model_raw, f"{where}.models.{key}")
    markers_raw = body.get("exhausted_markers")
    markers: tuple[str, ...] | None = None
    if markers_raw is not None:
        if isinstance(markers_raw, str) or not isinstance(markers_raw, list | tuple):
            raise ConfigError(f"{where}.exhausted_markers must be a list of strings")
        markers = tuple(_str(m, f"{where}.exhausted_markers") for m in markers_raw)
    rest = body.get("rest_default_s")
    return ProviderSpec(
        name=name,
        kind=kind,
        key_env=key_env,
        base_url=_str(body.get("base_url"), f"{where}.base_url"),
        limits=_frozen(_mapping(body.get("limits"), f"{where}.limits")),
        rest_default_s=DEFAULT_REST_S if rest is None else _number(rest, f"{where}.rest_default_s"),
        exhausted_markers=markers,
        models=MappingProxyType(models),
    )


def _parse_role(role: str, raw: Any, known: frozenset[str]) -> RoleSpec:
    where = f"roles.{role}"
    if role not in ROLES:
        raise ConfigError(f"unknown role {role!r} in roles")
    body = _mapping(raw, where)
    order_raw = body.get("providers")
    order: tuple[str, ...] | None = None
    if order_raw is not None:
        if isinstance(order_raw, str) or not isinstance(order_raw, list | tuple):
            raise ConfigError(f"{where}.providers must be a list of provider names")
        order = tuple(_str(n, f"{where}.providers") for n in order_raw)
        for name in order:
            if name not in known:
                raise ConfigError(f"{where}.providers names unknown provider {name!r}")
    budget = RoleBudget(
        max_calls_per_question=_opt_int(
            body.get("max_calls_per_question"), f"{where}.max_calls_per_question"
        ),
        max_tokens_per_run=_opt_int(body.get("max_tokens_per_run"), f"{where}.max_tokens_per_run"),
    )
    return RoleSpec(providers=order, budget=budget)


def parse_config(raw: Mapping[str, Any]) -> LLMConfig:
    """Validate ``raw`` (the loaded providers YAML) and return an immutable ``LLMConfig``."""
    top = _mapping(raw, "config")
    providers_raw = top.get("providers")
    if not isinstance(providers_raw, list | tuple) or not providers_raw:
        raise ConfigError("providers must be a non-empty list")
    providers: list[ProviderSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(providers_raw):
        spec = _parse_provider(item, index)
        if spec.name in seen:
            raise ConfigError(f"duplicate provider name {spec.name!r}")
        seen.add(spec.name)
        providers.append(spec)
    known = frozenset(seen)
    roles = {
        str(role): _parse_role(str(role), body, known)
        for role, body in _mapping(top.get("roles"), "roles").items()
    }
    prefixes_raw = top.get("embedding_prefixes")
    prefixes: Mapping[str, str] | None = None
    if prefixes_raw is not None:
        body = _mapping(prefixes_raw, "embedding_prefixes")
        values = {k: body.get(k) for k in ("query", "document")}
        for key, value in values.items():
            if not isinstance(value, str):  # an empty prefix is legitimate
                raise ConfigError(f"embedding_prefixes.{key} must be a string, got {value!r}")
        prefixes = MappingProxyType({k: str(v) for k, v in values.items()})
    budget = _mapping(top.get("budget"), "budget")
    cache_path = top.get("cache_path")
    run_id = top.get("run_id")
    return LLMConfig(
        cache_path=DEFAULT_CACHE_PATH if cache_path is None else _str(cache_path, "cache_path"),
        run_id=DEFAULT_RUN_ID if run_id is None else _str(run_id, "run_id"),
        global_max_tokens_per_run=_opt_int(
            budget.get("max_tokens_per_run"), "budget.max_tokens_per_run"
        ),
        embedding_prefixes=prefixes,
        providers=tuple(providers),
        roles=MappingProxyType(roles),
    )
