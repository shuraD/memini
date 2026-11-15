"""Agent steps: engine command lines, the generated Claude settings, prompts and markers.

Each step is stateless and one-shot: a fresh process gets one prompt and exits. Only what
the step needs is granted on the command line (``config.toml``). Claude also gets a
generated settings file with ``permissions.deny`` rules and the PreToolUse guard hook; both
are early warnings only. The path gate decides for every engine.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from string import Template
from typing import Any

from tools.orchestrator.config import Config

GUARD = Path(__file__).resolve().parent / "hooks" / "guard_worktree.py"
GUARDED_TOOLS = "Write|Edit|MultiEdit|NotebookEdit|Read|Grep|Glob|LS|NotebookRead|Bash"
DISPUTE = re.compile(r"^[ \t>*_`-]*TEST-DISPUTE:[ \t]*(\S.*)$", re.MULTILINE)
FOLLOW_UP = re.compile(r"^[ \t>*_`-]*FOLLOW-UP:[ \t]*(\S.*)$", re.MULTILINE)
ROLES = ("qa", "coder", "audit")


def pick_engines(
    cfg: Config, coder: str | None, available: Mapping[str, bool] | None = None
) -> dict[str, str]:
    """Return the engine of each role.

    The coder uses ``coder`` (the CLI choice), else ``coder_engine``, else the default.
    QA and audit use their configured engine; when none is set, the first engine of
    ``engine_preference`` that differs from the coder's and is installed, else the coder's.
    """
    have = (
        available
        if available is not None
        else {
            name: shutil.which(spec.roles["coder"][0]) is not None
            for name, spec in cfg.engines.items()
        }
    )
    chosen = coder or cfg.role_engines["coder"] or cfg.default_engine
    out = {"coder": chosen}
    for role in ("qa", "audit"):
        other = next(
            (e for e in cfg.engine_preference if e != chosen and e in cfg.engines and have.get(e)),
            chosen,
        )
        out[role] = cfg.role_engines[role] or other
    return out


def command(
    cfg: Config,
    engine: str,
    role: str,
    *,
    workdir: Path,
    settings: Path | None,
    prompt: str,
    reference_tools: str = "",
) -> list[str]:
    """Return the argv of ``engine`` for ``role`` with its placeholders filled in.

    A whole-argument placeholder is replaced by its value (a prompt is never spliced into
    another argument); ``{reference_tools}`` inside an argument becomes ``,<tools>`` when
    the coder has a reference, else nothing.
    """
    values = {
        "{prompt}": prompt,
        "{settings}": str(settings or ""),
        "{workdir}": str(workdir),
        "{max_budget_usd}": cfg.max_budget_usd,
    }
    out = []
    for arg in cfg.engines[engine].roles[role]:
        if arg in values:  # a whole-argument placeholder: no substring substitution
            out.append(values[arg])
        else:
            extra = f",{reference_tools}" if reference_tools else ""
            out.append(arg.replace("{reference_tools}", extra))
    return out


def _rule(tool: str, abs_path: Path, is_dir: bool) -> str:
    # Claude permission rules: "//abs/path" is an absolute path, "**" any depth
    return f"{tool}(/{abs_path}{'/**' if is_dir else ''})"


def private_paths(cfg: Config) -> list[tuple[Path, bool]]:
    """Return the paths no agent may read or write: hold-out, state, queue, inbox."""
    return [(cfg.holdout_dir, True), (cfg.state_dir, True), (cfg.queue, False), (cfg.inbox, False)]


def settings(
    cfg: Config,
    worktree: Path,
    *,
    writable: Sequence[str],
    guard_config: Path,
    protected: Sequence[str] | None = None,
    deny_read: Sequence[Path] = (),
    bash_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    """Return the Claude settings for one step: deny rules plus the guard hook.

    ``deny_read`` adds private directories for this step (the QA agent's reference repos);
    ``bash_roots`` are directories outside the worktree that Bash arguments may name (the
    coder's reference repository, read with git only).
    """
    protected = list(protected if protected is not None else (*cfg.protected, *cfg.guard_protected))
    private = [*private_paths(cfg), *((p, True) for p in deny_read)]
    deny = []
    for path, is_dir in private:
        deny += [_rule("Read", path, is_dir), _rule("Edit", path, is_dir)]
    for entry in protected:
        if entry.startswith("**/"):
            deny.append(f"Edit(/{worktree}/{entry})")
        else:
            deny.append(_rule("Edit", worktree / entry.rstrip("/"), entry.endswith("/")))
    guard = {
        "worktree": str(worktree),
        "writable": list(writable),
        "protected": protected,
        "deny_read": [str(p) for p, _ in private],
        "bash_roots": [str(p) for p in bash_roots],
    }
    guard_config.parent.mkdir(parents=True, exist_ok=True)
    guard_config.write_text(json.dumps(guard, indent=2), encoding="utf-8")
    hook_cmd = " ".join(
        shlex.quote(a) for a in (sys.executable, "-P", str(GUARD), str(guard_config))
    )
    return {
        "permissions": {"deny": deny},
        "hooks": {
            "PreToolUse": [
                {"matcher": GUARDED_TOOLS, "hooks": [{"type": "command", "command": hook_cmd}]}
            ]
        },
    }


def json_text(settings: Mapping[str, Any]) -> str:
    """Return a settings mapping as the JSON text Claude's --settings reads."""
    return json.dumps(settings, indent=2)


def render(cfg: Config, name: str, **values: str) -> str:
    """Return the prompt file ``<name>.txt`` with ``$placeholders`` filled in."""
    text = (cfg.prompts_dir / f"{name}.txt").read_text(encoding="utf-8")
    return Template(text).safe_substitute(values)


def agent_output(stdout: str) -> tuple[str, float | None]:
    """Return an agent's answer text and its cost in USD, if its CLI reported one.

    A JSON answer (``claude -p --output-format json``) yields its ``result`` text and
    ``total_cost_usd``; any other output is returned unchanged, with no cost.
    """
    try:
        data = json.loads(stdout)
    except ValueError:
        return stdout, None
    if not isinstance(data, dict) or not isinstance(data.get("result"), str):
        return stdout, None
    cost = data.get("total_cost_usd")
    return data["result"], float(cost) if isinstance(cost, int | float) else None


def dispute(output: str) -> str | None:
    """Return the reason of the first ``TEST-DISPUTE:`` line, if any."""
    m = DISPUTE.search(output)
    return m.group(1).strip() if m else None


def follow_ups(output: str) -> list[str]:
    """Return every ``FOLLOW-UP:`` proposal in the output."""
    return [m.strip() for m in FOLLOW_UP.findall(output)]
