"""AST check of the coder's changed Python files: no reach into the interpreter or the tests.

Tests cannot catch code that forges their own results in-process (patching pytest's report
objects, writing a fake junit report, exiting early). This check refuses the means to do it,
from the parsed source and never by importing it:

- ``getattr``/``setattr``/``delattr``/``hasattr`` whose attribute name is not a string literal;
- calls to ``exec``, ``eval``, ``compile``, ``__import__``, ``vars``, ``globals``,
  ``locals`` (configurable);
- dunder attributes outside an allow-list (``__dict__``, ``__func__``, ``__globals__`` ...)
  and dunder names outside another (``__builtins__``, ``__loader__`` ...);
- imports outside an allow-list (a stdlib subset, ``memini`` and the declared dependencies),
  star imports, and forbidden names inside allowed modules (``os._exit``, ``sys.modules``),
  however they are reached: ``os._exit``, ``from os import _exit``, ``getattr(os, "_exit")``.

The regex deny scan stays as an extra layer. Assigning a module to another name and calling
through it is not tracked; the audit and the owner-visible diff cover that until the microVM.
"""

from __future__ import annotations

import ast
import fnmatch
import re
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tools.orchestrator.config import Config

_ATTR_BUILTINS = frozenset({"getattr", "setattr", "delattr", "hasattr"})
_DEP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class Rules:
    """What the check allows, resolved from the configuration and the project's dependencies."""

    forbidden_calls: frozenset[str]
    dunder_allow: frozenset[str]
    dunder_names: frozenset[str]
    import_allow: tuple[str, ...]
    forbidden_names: tuple[str, ...]


def declared_modules(pyproject: Path, mapping: dict[str, str]) -> list[str]:
    """Return the import names of the project's declared dependencies (``[project]``)."""
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    out = []
    for dep in data.get("project", {}).get("dependencies", []):
        m = _DEP_NAME.match(str(dep).strip())
        if m:
            name = m.group(0).lower().replace("-", "_").replace(".", "_")
            out.append(mapping.get(name, name))
    return out


def rules(cfg: Config, worktree: Path) -> Rules:
    """Return the rules for a checkout (its pyproject names the declared dependencies)."""
    deps = declared_modules(worktree / "pyproject.toml", dict(cfg.dependency_modules))
    return Rules(
        forbidden_calls=frozenset(cfg.ast_forbidden_calls),
        dunder_allow=frozenset(cfg.ast_dunder_allow),
        dunder_names=frozenset(cfg.ast_dunder_names),
        import_allow=(*cfg.import_allow, *deps),
        forbidden_names=cfg.forbidden_names,
    )


def _dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")  # noqa: PLR2004


class _Checker:
    """One file's check: imports first (so aliases are known), then every other node."""

    def __init__(self, path: str, rule_set: Rules) -> None:
        self.path, self.rules = path, rule_set
        self.aliases: dict[str, str] = {}  # local name -> dotted module or object
        self.problems: list[str] = []

    def bad(self, node: ast.AST, what: str) -> None:
        self.problems.append(f"{self.path}:{getattr(node, 'lineno', 0)}: {what}")

    def module_allowed(self, module: str) -> bool:
        return any(module == a or module.startswith(a + ".") for a in self.rules.import_allow)

    def forbidden(self, dotted: str) -> bool:
        parts = dotted.split(".")
        prefixes = [".".join(parts[: n + 1]) for n in range(len(parts))]
        return any(fnmatch.fnmatchcase(p, f) for p in prefixes for f in self.rules.forbidden_names)

    def dotted(self, node: ast.expr) -> str | None:
        attrs: list[str] = []
        while isinstance(node, ast.Attribute):
            attrs.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name) and node.id in self.aliases:
            return ".".join([self.aliases[node.id], *reversed(attrs)])
        return None

    def check_import(self, node: ast.Import) -> None:
        for alias in node.names:
            if not self.module_allowed(alias.name) or self.forbidden(alias.name):
                self.bad(node, f"import of {alias.name} is not allowed")
            local = alias.asname or alias.name.split(".", 1)[0]
            self.aliases[local] = alias.name if alias.asname else local

    def check_import_from(self, node: ast.ImportFrom) -> None:
        if node.level:  # a relative import stays inside the package
            return
        module = node.module or ""
        if not self.module_allowed(module):
            self.bad(node, f"import from {module} is not allowed")
        for alias in node.names:
            if alias.name == "*":
                self.bad(node, f"star import from {module} is not allowed")
                continue
            dotted = f"{module}.{alias.name}"
            if self.forbidden(dotted):
                self.bad(node, f"{dotted} is forbidden")
            self.aliases[alias.asname or alias.name] = dotted

    def check_node(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            if _dunder(node.id) and node.id not in self.rules.dunder_names:
                self.bad(node, f"name {node.id} is not allowed")
        elif isinstance(node, ast.Attribute):
            if _dunder(node.attr) and node.attr not in self.rules.dunder_allow:
                self.bad(node, f"attribute {node.attr} is not allowed")
            dotted = self.dotted(node)
            if dotted and self.forbidden(dotted):
                self.bad(node, f"{dotted} is forbidden")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in self.rules.forbidden_calls:
                self.bad(node, f"call to {node.func.id}() is not allowed")
            elif node.func.id in _ATTR_BUILTINS:
                self.check_attr_call(node, node.func.id)

    def check_attr_call(self, node: ast.Call, name: str) -> None:
        literal = (
            len(node.args) >= 2  # noqa: PLR2004 — (object, name)
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        )
        if not literal:
            self.bad(node, f"{name}() needs a string-literal attribute name")
            return
        attr = str(node.args[1].value)  # type: ignore[attr-defined]
        if _dunder(attr) and attr not in self.rules.dunder_allow:
            self.bad(node, f"{name}() of attribute {attr} is not allowed")
        base = self.dotted(node.args[0])
        if base and self.forbidden(f"{base}.{attr}"):
            self.bad(node, f"{base}.{attr} is forbidden")


def problems(path: str, source: str, rule_set: Rules) -> list[str]:
    """Return every reason the Python ``source`` of ``path`` is refused (empty = accepted)."""
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno or 0}: not valid Python"]
    checker = _Checker(path, rule_set)
    nodes = list(ast.walk(tree))
    for node in nodes:
        if isinstance(node, ast.Import):
            checker.check_import(node)
        elif isinstance(node, ast.ImportFrom):
            checker.check_import_from(node)
    for node in nodes:
        checker.check_node(node)
    return list(dict.fromkeys(checker.problems))


def check_files(sources: Iterable[tuple[str, str]], rule_set: Rules) -> list[str]:
    """Return the problems of every ``(path, source)`` pair."""
    out: list[str] = []
    for path, source in sources:
        out += problems(path, source, rule_set)
    return out


def override_granted(inbox_text: str, task_id: str, commit: str) -> bool:
    """Return True when the owner wrote ``OVERRIDE-AST <task id> <commit sha>`` in the inbox."""
    pattern = re.compile(
        rf"^OVERRIDE-AST[ \t]+{re.escape(task_id)}[ \t]+([0-9a-f]{{7,40}})[ \t]*$", re.MULTILINE
    )
    return any(commit.startswith(m) for m in pattern.findall(inbox_text))
