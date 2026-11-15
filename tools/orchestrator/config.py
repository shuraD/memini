"""Orchestrator configuration: defaults from ``config.toml``, a user file, env, overrides.

Nothing here is a machine path: relative paths resolve against the repository root, and
:func:`load_config` refuses a state, hold-out, inbox or queue location inside the
repository or the worktree directory (agents work there; they must never reach those).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULTS = Path(__file__).with_name("config.toml")
PROMPTS = Path(__file__).with_name("prompts")
# keys whose override replaces the default instead of merging into it
_REPLACE = frozenset({"engines", "gates", "profiles", "module_roots"})
_ENV_PATHS = {
    "state_dir": "MEMINI_ORCH_STATE_DIR",
    "holdout_dir": "MEMINI_ORCH_HOLDOUT_DIR",
    "worktrees_dir": "MEMINI_ORCH_WORKTREES_DIR",
    "inbox": "MEMINI_ORCH_INBOX",
    "queue": "MEMINI_ORCH_QUEUE",
}
STAGES = frozenset({"slice", "merge"})


class ConfigError(ValueError):
    """The configuration is unusable; nothing is run."""


@dataclass(frozen=True)
class Gate:
    """One gate command: a ci.yml step (by name prefix) or an argv."""

    name: str
    stages: frozenset[str]
    ci_step: str | None = None
    run: tuple[str, ...] = ()
    profiles: bool = False
    forbid_output: bool = False
    sandbox: bool = False  # runs coder code: in the sandbox executor; else on the host


@dataclass(frozen=True)
class EngineSpec:
    """How to start one engine per role (``qa``, ``coder``, ``audit``)."""

    name: str
    author: str
    stdin: bool
    roles: Mapping[str, tuple[str, ...]]
    env: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProfileSpec:
    """An env profile: extra env vars, optionally every flag switched on."""

    name: str
    env: Mapping[str, str]
    all_flags: bool


@dataclass(frozen=True)
class Config:
    """Every setting the orchestrator uses (see ``config.toml`` for their meaning)."""

    repo: Path
    main_branch: str
    branch_prefix: str
    worker: str
    default_engine: str
    role_engines: Mapping[str, str]
    engine_preference: tuple[str, ...]
    executor: str
    docker_cli: str
    docker_image: str
    dockerfile: str
    docker_memory: str
    docker_pids: int
    docker_run_timeout_s: int
    allow: tuple[str, ...]
    deny_patterns: tuple[str, ...]
    ast_forbidden_calls: tuple[str, ...]
    ast_dunder_allow: tuple[str, ...]
    ast_dunder_names: tuple[str, ...]
    import_allow: tuple[str, ...]
    forbidden_names: tuple[str, ...]
    dependency_modules: Mapping[str, str]
    local_coder_engines: tuple[str, ...]
    module_roots: Mapping[str, str]
    ignored_ok: tuple[str, ...]
    reference_repos: Mapping[str, Path]
    reference_tools: str
    protected: tuple[str, ...]
    guard_protected: tuple[str, ...]
    qa_allow: tuple[str, ...]
    state_dir: Path
    holdout_dir: Path
    worktrees_dir: Path
    inbox: Path
    queue: Path
    prompts_dir: Path
    holdout_required: bool
    holdout_python: str
    python: str
    setup: tuple[tuple[str, ...], ...]
    pytest_args: tuple[str, ...]
    env_allow: tuple[str, ...]
    max_coder_attempts: int
    merge_retries: int
    max_parallel: int
    engine_timeout_s: int
    gate_timeout_s: int
    max_budget_usd: str
    forbid_output: str
    sandbox_skip_markers: tuple[str, ...]
    identity_name: str
    identity_email: str
    flags_file: str
    flags_function: str
    flags_on_value: str
    profiles: tuple[ProfileSpec, ...]
    gates: tuple[Gate, ...]
    engines: Mapping[str, EngineSpec]

    @property
    def parallel(self) -> bool:
        """Return True when slices may run at once (docker executor and max_parallel > 1)."""
        return self.executor == "docker" and self.max_parallel > 1


def _merge(base: dict[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict) and key not in _REPLACE:
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _path(repo: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return (p if p.is_absolute() else repo / p).resolve()


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _strs(raw: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        msg = f"{key} must be a list of strings"
        raise ConfigError(msg)
    return tuple(value)


def _gates(raw: list[Mapping[str, Any]]) -> tuple[Gate, ...]:
    out = []
    for g in raw:
        stages = frozenset(g.get("stages", ["slice", "merge"]))
        if (
            not stages <= STAGES
            or not g.get("name")
            or bool(g.get("ci_step")) == bool(g.get("run"))
        ):
            msg = f"gate {g!r}: needs a name, stages in {sorted(STAGES)}, and ci_step or run"
            raise ConfigError(msg)
        out.append(
            Gate(
                name=str(g["name"]),
                stages=stages,
                ci_step=g.get("ci_step"),
                run=tuple(g.get("run", ())),
                profiles=bool(g.get("profiles", False)),
                forbid_output=bool(g.get("forbid_output", False)),
                sandbox=bool(g.get("sandbox", False)),
            )
        )
    return tuple(out)


def _engines(raw: Mapping[str, Mapping[str, Any]]) -> dict[str, EngineSpec]:
    out = {}
    for name, e in raw.items():
        roles = {r: tuple(e[r]) for r in ("qa", "coder", "audit") if r in e}
        if set(roles) != {"qa", "coder", "audit"}:
            msg = f"engine {name}: needs qa, coder and audit commands"
            raise ConfigError(msg)
        out[name] = EngineSpec(
            name, str(e.get("author", name)), bool(e.get("stdin")), roles, _strs(e, "env")
        )
    return out


def _git_user(repo: Path, field: str) -> str:
    """Return the repository's configured git ``user.<field>`` (name or email), or ""."""
    res = subprocess.run(  # noqa: S603 — fixed git args, no shell
        ["git", "-C", str(repo), "config", "--get", f"user.{field}"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    return res.stdout.strip() if res.returncode == 0 else ""


def _default_executor(raw: Mapping[str, Any]) -> str:
    """Return "docker" when the configured docker CLI answers ``docker info``, else "local"."""
    cli = str(raw.get("docker_cli", "docker"))
    if shutil.which(cli) is None:
        return "local"
    try:
        ok = (
            subprocess.run(  # noqa: S603 — fixed CLI from config, no shell
                [cli, "info"], capture_output=True, timeout=20, check=False
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return "local"
    return "docker" if ok else "local"


def load_config(
    repo: Path,
    *,
    config_file: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> Config:
    """Return the configuration for ``repo``: defaults < file < env paths < overrides."""
    env = os.environ if env is None else env
    raw: dict[str, Any] = tomllib.loads(DEFAULTS.read_text(encoding="utf-8"))
    user = config_file or (
        Path(env["MEMINI_ORCH_CONFIG"]) if env.get("MEMINI_ORCH_CONFIG") else None
    )
    if user is not None:
        raw = _merge(raw, tomllib.loads(user.read_text(encoding="utf-8")))
    raw = _merge(raw, {k: env[v] for k, v in _ENV_PATHS.items() if env.get(v)})
    raw = _merge(raw, overrides or {})
    repo = Path(raw.get("repo") or repo).resolve()
    paths = {k: _path(repo, str(raw[k])) for k in _ENV_PATHS}
    for key in ("state_dir", "holdout_dir", "inbox", "queue"):
        for root in (repo, paths["worktrees_dir"]):
            if _inside(paths[key], root):
                msg = f"{key} {paths[key]} must be outside the repository and every worktree"
                raise ConfigError(msg)
    if _inside(paths["worktrees_dir"], repo):
        msg = "worktrees_dir must be outside the repository"
        raise ConfigError(msg)
    deny = _strs(raw, "deny_patterns")
    for pattern in deny:
        try:
            re.compile(pattern)
        except re.error as exc:
            msg = f"deny pattern {pattern!r} is not a valid regex: {exc}"
            raise ConfigError(msg) from exc
    env_allow = _strs(raw, "env_allow")
    if any(name.startswith("MEMINI_ORCH_") for name in env_allow):
        msg = "env_allow must not pass MEMINI_ORCH_* variables to an executor"
        raise ConfigError(msg)
    flags = raw.get("flags", {})
    identity = raw.get("identity", {})
    ident_name = str(identity.get("name") or "") or _git_user(repo, "name")
    # may stay empty: reading the config (CI steps, AST rules) needs no identity; a task run
    # does, and pipeline.precheck refuses it before dispatch (a fresh CI runner has none)
    ident_email = str(identity.get("email") or "") or _git_user(repo, "email")
    profiles = tuple(
        ProfileSpec(name, dict(p.get("env", {})), bool(p.get("all_flags", False)))
        for name, p in raw.get("profiles", {}).items()
    )
    engines = _engines(raw.get("engines", {}))
    role_engines = {r: str(raw.get(f"{r}_engine", "")) for r in ("qa", "coder", "audit")}
    for name in (raw["default_engine"], *[v for v in role_engines.values() if v]):
        if name not in engines:
            msg = f"unknown engine {name!r} (configured: {sorted(engines)})"
            raise ConfigError(msg)
    executor = str(raw.get("executor", "") or _default_executor(raw))
    max_parallel = int(raw.get("max_parallel", 1))
    if max_parallel < 1:
        msg = f"max_parallel must be at least 1, not {max_parallel}"
        raise ConfigError(msg)
    if executor not in ("local", "docker"):
        msg = f"executor {executor!r} is not available (use 'local' or 'docker')"
        raise ConfigError(msg)
    return Config(
        repo=repo,
        main_branch=str(raw["main_branch"]),
        branch_prefix=str(raw["branch_prefix"]),
        worker=str(raw["worker"]),
        default_engine=str(raw["default_engine"]),
        role_engines=role_engines,
        engine_preference=_strs(raw, "engine_preference"),
        executor=executor,
        docker_cli=str(raw.get("docker_cli", "docker")),
        docker_image=str(raw.get("docker_image", "memini-orchestrator-sandbox")),
        dockerfile=str(raw.get("dockerfile", "tools/orchestrator/Dockerfile")),
        docker_memory=str(raw.get("docker_memory", "2g")),
        docker_pids=int(raw.get("docker_pids", 512)),
        docker_run_timeout_s=int(raw.get("docker_run_timeout_s", 1800)),
        allow=_strs(raw, "allow"),
        deny_patterns=deny,
        ast_forbidden_calls=_strs(raw, "ast_forbidden_calls"),
        ast_dunder_allow=_strs(raw, "ast_dunder_allow"),
        ast_dunder_names=_strs(raw, "ast_dunder_names"),
        import_allow=_strs(raw, "import_allow"),
        forbidden_names=_strs(raw, "forbidden_names"),
        dependency_modules={str(k): str(v) for k, v in raw.get("dependency_modules", {}).items()},
        local_coder_engines=_strs(raw, "local_coder_engines"),
        module_roots={str(k): str(v) for k, v in raw.get("module_roots", {}).items()},
        ignored_ok=_strs(raw, "ignored_ok"),
        reference_repos={
            str(k): _path(repo, str(v)) for k, v in raw.get("reference_repos", {}).items()
        },
        reference_tools=str(raw.get("reference_tools", "")),
        protected=_strs(raw, "protected"),
        guard_protected=_strs(raw, "guard_protected"),
        qa_allow=_strs(raw, "qa_allow"),
        state_dir=paths["state_dir"],
        holdout_dir=paths["holdout_dir"],
        worktrees_dir=paths["worktrees_dir"],
        inbox=paths["inbox"],
        queue=paths["queue"],
        prompts_dir=_path(repo, raw["prompts_dir"]) if raw.get("prompts_dir") else PROMPTS,
        holdout_required=bool(raw["holdout_required"]),
        holdout_python=str(raw.get("holdout_python", "")),
        python=str(raw["python"]),
        setup=tuple(tuple(cmd) for cmd in raw.get("setup", [])),
        pytest_args=_strs(raw, "pytest_args"),
        env_allow=env_allow,
        max_coder_attempts=int(raw["max_coder_attempts"]),
        merge_retries=int(raw["merge_retries"]),
        max_parallel=max_parallel,
        engine_timeout_s=int(raw["engine_timeout_s"]),
        gate_timeout_s=int(raw["gate_timeout_s"]),
        max_budget_usd=str(raw["max_budget_usd"]),
        forbid_output=str(raw["forbid_output"]),
        sandbox_skip_markers=_strs(raw, "sandbox_skip_markers"),
        identity_name=ident_name,
        identity_email=ident_email,
        flags_file=str(flags.get("file", "")),
        flags_function=str(flags.get("function", "_env_bool")),
        flags_on_value=str(flags.get("on_value", "1")),
        profiles=profiles,
        gates=_gates(raw.get("gates", [])),
        engines=engines,
    )
