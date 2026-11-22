"""Slice specs: a YAML header between ``---`` lines, then free text; and the $0 check.

Header fields: ``id``, ``module``, ``paths``, ``depends_on``, ``flag``, ``reference``
(optional), ``top_caller`` (optional: the ticket's ``Top caller:``), ``skeleton`` (optional: the
ticket's ``Skeleton:`` scope, the one ticket that creates the stubs), ``behavior``, ``interface``,
``acceptance``, ``out_of_scope``, ``size``.
:func:`check` runs before anything is dispatched (no engine, no cost): every required
field is present, the paths lie inside one module, the dependencies exist and the
reference branch exists. ``reference`` is coder-only context: :meth:`Spec.text` leaves it
out, so the QA agent and the audit never see it.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tools.orchestrator import gates, gitops
from tools.orchestrator.config import Config

REQUIRED = (
    "id",
    "module",
    "paths",
    "depends_on",
    "flag",
    "behavior",
    "interface",
    "acceptance",
    "out_of_scope",
    "size",
)
# present but allowed to be empty (no dependency, no flag, nothing out of scope)
MAY_BE_EMPTY = frozenset({"depends_on", "flag", "out_of_scope"})
ORDER = (*REQUIRED[:5], "reference", "top_caller", "skeleton", *REQUIRED[5:])


class SpecError(ValueError):
    """The spec cannot be read at all."""


@dataclass(frozen=True)
class Spec:
    """A parsed slice spec."""

    path: Path
    header: Mapping[str, Any]
    body: str

    @property
    def task_id(self) -> str:
        """Return the spec's ``id``."""
        return str(self.header.get("id", ""))

    @property
    def module(self) -> str:
        """Return the ledger module id."""
        return str(self.header.get("module", ""))

    @property
    def paths(self) -> tuple[str, ...]:
        """Return the declared paths."""
        return _list(self.header.get("paths"))

    @property
    def depends_on(self) -> tuple[str, ...]:
        """Return the ids this slice depends on."""
        return _list(self.header.get("depends_on"))

    @property
    def reference(self) -> str | None:
        """Return the coder-only reference branch, if any."""
        ref = self.header.get("reference")
        return str(ref) if ref else None

    @property
    def top_caller(self) -> str:
        """Return the ticket's top caller (``<dotted.module>:<function>``), or ``""``."""
        return "".join(_list(self.header.get("top_caller"))[:1])

    @property
    def skeleton(self) -> str:
        """Return the scope whose approved skeleton this slice creates as stubs, or ``""``."""
        return "".join(_list(self.header.get("skeleton"))[:1])

    @property
    def title(self) -> str:
        """Return a one-line title: the first line of ``behavior``."""
        first = str(self.header.get("behavior", "")).strip().splitlines() or [self.task_id]
        return " ".join(first[0].split())[:72]

    def text(self, *, with_reference: bool = False) -> str:
        """Return the spec as given to an agent; without ``reference`` unless asked."""
        keys = [k for k in ORDER if k in self.header and (with_reference or k != "reference")]
        shown = {k: self.header[k] for k in keys}
        dumped = yaml.safe_dump(shown, sort_keys=False, allow_unicode=True, width=100)
        return f"{dumped.rstrip()}\n\n{self.body.strip()}\n".rstrip() + "\n"


_NOTE = re.compile(r"\s*\([^)]*\)")
_EMPTY_WORDS = frozenset({"none", "-", "n/a"})


def _list(value: object) -> tuple[str, ...]:
    """Return the entries of a list field.

    Notes in brackets and placeholder words such as ``none`` are not entries (owner
    2026-11-16: IN-07 wrote ``x.py (new)``, AR-01 wrote ``none``).
    """
    if value is None or value == "":
        return ()
    items = (
        value.split(",")
        if isinstance(value, str)
        else value
        if isinstance(value, list)
        else [value]
    )
    out = (_NOTE.sub("", str(v)).strip() for v in items)
    return tuple(v for v in out if v and v.lower() not in _EMPTY_WORDS)


def load(path: Path) -> Spec:
    """Read a spec file; raise :class:`SpecError` when it has no readable header."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"cannot read spec {path.name}: {exc.strerror}"
        raise SpecError(msg) from exc
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---" or "---" not in (ln.strip() for ln in lines[1:]):
        msg = f"spec {path.name} needs a YAML header between '---' lines"
        raise SpecError(msg)
    end = next(i for i, ln in enumerate(lines[1:], 1) if ln.strip() == "---")
    try:
        header = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError as exc:
        detail = " ".join(str(exc).split())[:200]
        msg = f"spec {path.name}: the header is not valid YAML ({detail})"
        raise SpecError(msg) from exc
    if not isinstance(header, dict):
        msg = f"spec {path.name}: the header must be a mapping of fields"
        raise SpecError(msg)
    return Spec(path, header, "\n".join(lines[end + 1 :]))


def resolve_reference(cfg: Config, ref: str) -> tuple[Path, str]:
    """Return ``(repository, branch)`` of ``[<repo name or path>@]<branch>``.

    A repo name comes from ``reference_repos``; without one, the branch is in this
    repository. Nothing is fetched: the coder reads the branch in place, read-only.
    """
    if "@" not in ref:
        return cfg.repo, ref
    where, branch = ref.rsplit("@", 1)
    if where in cfg.reference_repos:
        return cfg.reference_repos[where], branch
    p = Path(where).expanduser()
    return (p if p.is_absolute() else cfg.repo / p).resolve(), branch


def module_unit(cfg: Config, path: str) -> str | None:
    """Return the module a path belongs to (``<module root><first component>``), else None."""
    for root in cfg.module_roots:
        if path.startswith(root):
            first = path[len(root) :].split("/", 1)[0]
            name = first.removesuffix(".py")
            return root + name if name else None
    return None


def check(
    cfg: Config,
    spec: Spec,
    task_id: str,
    known_ids: Collection[str],
    narrowed: tuple[str, ...] = (),
) -> list[str]:
    """Return every reason the spec must not be dispatched (empty = dispatchable)."""
    problems = []
    for field in REQUIRED:
        if field not in spec.header or spec.header[field] is None:
            if field not in MAY_BE_EMPTY or field not in spec.header:
                problems.append(f"required field missing: {field}")
        elif field not in MAY_BE_EMPTY and not str(spec.header[field]).strip():
            problems.append(f"required field empty: {field}")
    if spec.task_id and spec.task_id != task_id:
        problems.append(f"spec id {spec.task_id!r} is not the task id {task_id!r}")
    try:
        paths = gates.normalize_paths(spec.paths)
        narrowed = gates.normalize_paths(narrowed)
    except gates.GateConfigError as exc:
        return [*problems, str(exc)]
    units = {module_unit(cfg, p) for p in paths}
    # a spec with a top caller is one path through every layer: its paths may cross modules;
    # so may the one that creates the stubs of a skeleton
    crosses = spec.top_caller or spec.skeleton
    if paths and (None in units or (len(units) != 1 and not crosses)):
        problems.append(
            "paths must lie inside one module under "
            + ", ".join(cfg.module_roots)
            + f": {', '.join(paths)}"
        )
    problems += [
        f"declared path {p} is outside the configured allow list"
        for p in paths
        if not any(gates.matches(a, p) or a == p for a in cfg.allow)
    ]
    problems += [
        f"declared path {p} is protected"
        for p in paths
        if any(gates.matches(x, p) for x in cfg.protected)
    ]
    problems += [
        f"narrowed path {n} is not inside the spec paths"
        for n in narrowed
        if not any(gates.matches(p, n) or p == n for p in paths)
    ]
    problems += [
        f"depends_on {d} is not a known task" for d in spec.depends_on if d not in known_ids
    ]
    if spec.reference:
        ref_repo, branch = resolve_reference(cfg, spec.reference)
        found = (
            ref_repo.is_dir()
            and gitops.git(
                ref_repo, "rev-parse", "--verify", "--quiet", f"{branch}^{{commit}}", check=False
            ).returncode
            == 0
        )
        if not found:
            problems.append(f"reference {spec.reference!r} is not a branch of a local repository")
    return problems
