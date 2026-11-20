"""CI watch: after a push to main, a red GitHub Actions run posts a move so someone fixes it.

Uses the ``gh`` CLI. Every completed run is seen once (``<state_dir>/ci-seen.json``): green is
a ledger row, red is a ledger row plus one batch_failure move to the planner carrying the
failing tests and errors.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable

from tools.orchestrator import moves, state
from tools.orchestrator.config import Config

SEEN = "ci-seen.json"
Runner = Callable[[list[str]], str]
_FIELDS = "databaseId,conclusion,headSha,displayTitle,status"
_MAX_ERRORS = 8
_FAIL_MARKS = ("FAILED ", "ERROR at setup of ")
_STAMP = re.compile(r"^\d{4}-\d\d-\d\dT[\d:.]+Z ")  # GitHub puts a timestamp before each line


def _gh(argv: list[str]) -> str:
    return subprocess.run(argv, capture_output=True, text=True, check=True).stdout  # noqa: S603


def _failures(log: str) -> tuple[list[str], list[str]]:
    """Return the distinct failing test ids and the first distinct ``E `` lines of a log."""
    tests: dict[str, None] = {}
    errors: dict[str, None] = {}
    for raw in log.splitlines():
        line = _STAMP.sub("", raw.split("\t")[-1].strip())
        for mark in _FAIL_MARKS:
            if mark in line:
                tests[line.split(mark, 1)[1].split(" - ")[0].strip()] = None
        if line.startswith("E "):
            errors[line[:200]] = None
    return list(tests), list(errors)[:_MAX_ERRORS]


def check(cfg: Config, runner: Runner = _gh) -> list[str]:
    """Record the new completed runs of main; post one move per red run. Return one line each."""
    runs = json.loads(
        runner(["gh", "run", "list", "--branch", "main", "--limit", "10", "--json", _FIELDS])
    )
    path = cfg.state_dir / SEEN
    first = not path.exists()
    seen: list[int] = [] if first else json.loads(path.read_text(encoding="utf-8"))
    if first:  # the first look reports only the newest finished run; older ones are history
        done = [r["databaseId"] for r in runs if r["status"] == "completed"]
        seen = done[1:]
    lines = []
    for run in runs:
        if run["status"] != "completed" or run["databaseId"] in seen:
            continue
        seen.append(run["databaseId"])
        sha, title = run["headSha"][:7], run["displayTitle"]
        if run["conclusion"] == "success":
            state.ops_log(cfg, sha, "ci_green", title)
            lines.append(f"{sha}: CI green ({title})")
            continue
        log = runner(["gh", "run", "view", str(run["databaseId"]), "--log-failed"])
        tests, errors = _failures(log)
        text = (
            f"CI {run['conclusion']} on main: {title}; {len(tests)} failing test(s), "
            f"{len(errors)} error line(s). Failing: {'; '.join(tests) or 'none found'}. "
            f"Errors: {' | '.join(errors) or 'none found'}"
        )
        state.ops_log(cfg, sha, "ci_red", text)
        moves.post_once(cfg, sha, sender="ci", to="planner", kind="batch_failure", text=text)
        lines.append(f"{sha}: CI RED, move posted ({title})")
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seen), encoding="utf-8")
    return lines


def wait(
    cfg: Config, sha: str, runner: Runner = _gh, timeout_s: float = 900, poll_s: float = 20
) -> list[str]:
    """Wait for the run of commit ``sha`` to finish, then record it (a red run posts its move).

    A push is not done until its CI run is green: this is the step after ``git push``.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        runs = json.loads(
            runner(["gh", "run", "list", "--commit", sha, "--json", _FIELDS, "--limit", "5"])
        )
        if runs and all(r["status"] == "completed" for r in runs):
            return check(cfg, runner)
        if time.monotonic() >= deadline:
            state.ops_log(cfg, sha[:7], "ci_timeout", f"no finished CI run for {sha[:7]} in time")
            return [f"{sha[:7]}: no finished CI run in {int(timeout_s)} s"]
        time.sleep(poll_s)
