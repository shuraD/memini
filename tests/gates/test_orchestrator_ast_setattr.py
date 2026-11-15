"""The AST check allows the frozen-dataclass idiom ``object.__setattr__(self, "name", value)``.

Owner 2026-11-15 (LP-17, option B): only that exact shape passes. ``object.__setattr__`` works
on any object, modules included, so every other use stays refused: a computed or dunder name,
a target other than the method's own ``self``, a reassigned ``self`` or a rebound ``object``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.orchestrator import astcheck, config

ROOT = Path(__file__).resolve().parents[2]
REFUSED = "attribute __setattr__ is not allowed"


def _problems(source: str) -> list[str]:
    cfg = config.load_config(ROOT, overrides={"state_dir": "/nonexistent-state"}, env={})
    return astcheck.problems("src/memini/x.py", source, astcheck.rules(cfg, ROOT))


def _method(body: str, *, params: str = "self") -> str:
    return (
        "from dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass R:\n"
        f"    pairs: tuple[str, ...]\n\n    def __post_init__({params}) -> None:\n{body}"
    )


def test_the_frozen_dataclass_idiom_passes() -> None:
    body = "        object.__setattr__(self, 'pairs', tuple(self.pairs))\n"
    assert _problems(_method(body)) == []


@pytest.mark.parametrize(
    "source",
    [
        # a computed attribute name
        _method("        name = 'pairs'\n        object.__setattr__(self, name, ())\n"),
        # a dunder attribute name
        _method("        object.__setattr__(self, '__class__', R)\n"),
        # a target that is not self
        _method("        import sys\n        object.__setattr__(sys, 'pairs', ())\n"),
        # self reassigned before the call
        _method(
            "        import sys\n        self = sys\n        object.__setattr__(self, 'x', 1)\n"
        ),
        # outside any method: self is just a name
        "import sys\nself = sys\nobject.__setattr__(self, 'x', 1)\n",
        # a function whose first parameter is not self
        _method("        object.__setattr__(self, 'x', 1)\n", params="other, self"),
        # object rebound to something else
        "import sys\nobject = type(sys)\n" + _method("        object.__setattr__(self, 'x', 1)\n"),
        # keyword arguments, or the wrong number of arguments
        _method("        object.__setattr__(self, 'x', value=1)\n"),
        _method("        object.__setattr__(self, 'x')\n"),
        # any other way of reaching __setattr__
        _method("        self.__setattr__('x', 1)\n"),
        _method("        f = object.__setattr__\n        f(self, 'x', 1)\n"),
    ],
)
def test_every_other_use_of_setattr_stays_refused(source: str) -> None:
    assert any(REFUSED in p for p in _problems(source)), _problems(source)


def test_a_nested_function_with_its_own_self_does_not_lend_the_outer_one() -> None:
    body = (
        "        import sys\n\n        def inner(self=sys) -> None:\n"
        "            object.__setattr__(self, 'x', 1)\n\n        inner()\n"
    )
    assert any(REFUSED in p for p in _problems(_method(body)))
