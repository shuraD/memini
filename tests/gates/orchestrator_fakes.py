"""Fake engine and fake gate for the orchestrator tests: scripted, never call a model.

Run as a script (the orchestrator starts it as an engine or gate command)::

    python orchestrator_fakes.py engine <role>     # plays FAKE_PLAN[role][n-1]
    python orchestrator_fakes.py gate <name>       # logs the call; FAKE_GATES[name] decides
    python orchestrator_fakes.py venv <dir>        # a fast venv in <dir>/.venv (setup step)

The engine reads its prompt on stdin and records it as ``FAKE_LOG/<role>-<n>.txt``.
A plan step may ``write`` files (relative to the working directory), ``commit`` them with
git, print ``stdout`` and exit with ``exit``. A gate step may ``fail`` (with ``say``), or
``move_main`` once: commit a file to the main branch of ``FAKE_REPO`` (a concurrent merge).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import venv
from pathlib import Path

GIT = shutil.which("git") or "git"


def _plan(var: str) -> dict[str, object]:
    path = os.environ.get(var)
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}


def _engine(role: str) -> int:
    log = Path(os.environ["FAKE_LOG"])
    log.mkdir(parents=True, exist_ok=True)
    n = len(list(log.glob(f"{role}-*.txt"))) + 1
    (log / f"{role}-{n}.txt").write_text(sys.stdin.read(), encoding="utf-8")
    (log / f"env-{role}-{n}.txt").write_text(" ".join(sorted(_secrets())), encoding="utf-8")
    steps = _plan("FAKE_PLAN").get(role, [])
    assert isinstance(steps, list)
    step = steps[min(n, len(steps)) - 1] if steps else {}
    for rel, text in step.get("write", {}).items():
        target = Path.cwd() / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    if step.get("commit"):
        subprocess.run(  # noqa: S603 — test fake, fixed git args
            [GIT, "add", "-A"], check=True
        )
        subprocess.run(  # noqa: S603 — test fake, fixed git args
            [GIT, "commit", "-q", "-m", "engine commit"], check=True
        )
    print(step.get("stdout", ""))  # noqa: T201 — the fake engine answers on stdout
    return int(step.get("exit", 0))


def _venv(target: str) -> int:
    """Create ``target/.venv`` that sees this interpreter's packages (no pip, fast)."""
    venv.create(Path(target) / ".venv", with_pip=False, symlinks=True)
    site = (
        Path(target) / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    )
    (site / "site-packages" / "outer.pth").write_text(sysconfig.get_paths()["purelib"] + "\n")
    return 0


SECRETS = ("ANTHROPIC_API_KEY", "MEMINI_ORCH_SECRET", "GEMINI_API_KEY")


def _secrets() -> list[str]:
    return [k for k in SECRETS if k in os.environ]


def _gate(name: str) -> int:
    log = Path(os.environ["FAKE_LOG"])
    log.mkdir(parents=True, exist_ok=True)
    flag = os.environ.get("MEMINI_FLAG_A", "unset")
    step = _plan("FAKE_GATES").get(name, {})
    assert isinstance(step, dict)
    probe = ""
    if step.get("must_not_import"):
        found = importlib.util.find_spec(str(step["must_not_import"])) is not None
        probe = " import=" + ("PRESENT" if found else "absent")
    with (log / "gates.log").open("a", encoding="utf-8") as fh:
        fh.write(f"{name} flag={flag} cwd={Path.cwd().name}{probe} env={','.join(_secrets())}\n")
    calls = (log / "gates.log").read_text(encoding="utf-8").count(f"{name} flag=")
    if step.get("tamper") and calls == 1:
        target = Path(str(step["tamper"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("tampered\n", encoding="utf-8")
    marker = log / f"moved-{name}"
    if step.get("move_main") and not marker.exists():
        marker.write_text("x", encoding="utf-8")
        repo = os.environ["FAKE_REPO"]
        (Path(repo) / "OTHER.txt").write_text("concurrent merge\n", encoding="utf-8")
        subprocess.run(  # noqa: S603 — test fake, fixed git args
            [GIT, "-C", repo, "add", "OTHER.txt"], check=True
        )
        subprocess.run(  # noqa: S603 — test fake, fixed git args
            [GIT, "-C", repo, "commit", "-q", "-m", "concurrent"], check=True
        )
    if "say" in step:
        print(step["say"])  # noqa: T201 — the fake gate reports on stdout
    return 1 if step.get("fail") or step.get("fail_on_call") == calls else 0


if __name__ == "__main__":
    kind, arg = sys.argv[1], sys.argv[2]
    runners = {"engine": _engine, "gate": _gate, "venv": _venv}
    sys.exit(runners[kind](arg))
