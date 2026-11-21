"""The reachability gate (F-orchestrator-068, FLOW-HYBRID.md step 9, rule 4).

``gates.reachability_problems(repo, base, rev, top_caller, root)`` refuses a function or method
the diff adds or changes under the root that the ticket's top caller does not reach, and a reached
one that still raises ``NotImplementedError``. Every test builds a throwaway git repository with
two commits (``base`` and ``rev``); nothing under test is mocked.

Accepted limits, not tested here: registration through a decorator alone is not seen as reached,
the top caller is a module-level function, and name matching over-approximates.

The last part runs the pipeline with the fake engines of ``test_orchestrator.py``: the gate runs
on the coder's change with the spec's top caller (base: the QA commit) and its problems go back to
the coder; the Verifier is told the gate's result and the ticket's scope; every agent gets the
same rules text.
"""

# ruff: noqa: F811

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from tests.gates.test_orchestrator import (  # noqa: F401  (env is a fixture)
    AUDIT_PASS,
    GOOD,
    IMPL,
    TASK,
    _cfg,
    _coder,
    _plan,
    _prompts,
    _spec,
    _task,
    env,
)
from tests.gates.test_orchestrator_flow_hard import _answer, _row, _ticket
from tests.gates.test_orchestrator_test_review import APPROVE, EDITED, TEST
from tools.orchestrator import audit, engine, gates, moves, pipeline

GIT = shutil.which("git") or "git"
TOP = "pkg.app:main"
APP = "src/pkg/app.py"
INGEST = "src/pkg/ingest.py"
STORE = "src/pkg/store.py"
UTIL = "src/pkg/util.py"
MAIN_OLD = "def main():\n    return 0\n"
CALLS_WRITE = "from pkg.store import write\n\n\ndef main():\n    return write(1)\n"
WRITE = "def write(value):\n    return value\n"
ORPHAN = "def orphan():\n    return 1\n"


def _unreached(path: str, name: str, top: str = TOP) -> str:
    return f"{path}:{name} is not reachable from {top}"


def _stub(path: str, name: str) -> str:
    return f"{path}:{name} still raises not implemented"


@pytest.fixture(autouse=True)
def _no_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``GIT_*`` variable of the caller (a hook, a worktree) reaches a git call of a test."""
    for name in [n for n in os.environ if n.startswith("GIT_")]:
        monkeypatch.delenv(name)


def _git(repo: Path, *args: str) -> str:
    clean = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(  # noqa: S603 — test-only, fixed git args
        [GIT, "-C", str(repo), *args], capture_output=True, text=True, check=True, env=clean
    ).stdout.strip()


def _commit(repo: Path, files: dict[str, str]) -> str:
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "x")
    return _git(repo, "rev-parse", "HEAD")


def _problems(
    tmp: Path,
    before: dict[str, str],
    after: dict[str, str],
    top: str = TOP,
    root: str | None = None,
) -> list[str]:
    """Commit ``before`` (the base), then ``after`` (the change), and run the gate on the diff."""
    repo = tmp / "r"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "x")
    _git(repo, "config", "user.email", "t@example.invalid")
    base = _commit(repo, {"README.md": "x\n", **before})
    rev = _commit(repo, after)
    if root is None:
        return gates.reachability_problems(repo, base, rev, top)
    return gates.reachability_problems(repo, base, rev, top, root)


# ------------------------------------------------------------------------------ the rules
def test_no_top_caller_skips_the_gate(tmp_path: Path) -> None:
    """Rule: an empty top caller means no check, even for an uncalled stub."""
    after = {STORE: "def orphan():\n    raise NotImplementedError\n"}
    assert _problems(tmp_path, {APP: MAIN_OLD}, after, top="") == []


def test_a_chain_of_new_functions_reached_from_the_top_caller_passes(tmp_path: Path) -> None:
    """Rule: new functions on the path from the top caller, across files, are no problem."""
    after = {
        APP: "from pkg import ingest\n\n\ndef main():\n    return ingest.run()\n",
        INGEST: (
            "from pkg.store import write\n\n\ndef run():\n    return write(_clean(' x '))\n\n\n"
            "def _clean(text):\n    return text.strip()\n"
        ),
        STORE: WRITE,
    }
    assert _problems(tmp_path, {APP: MAIN_OLD}, after) == []


def test_a_top_caller_in_a_package_init_is_found(tmp_path: Path) -> None:
    """Rule: ``<dotted.module>`` may be a package: its ``__init__.py`` holds the function."""
    init = "src/pkg/__init__.py"
    after = {
        init: "from pkg.store import write\n\n\ndef main():\n    return write(1)\n",
        STORE: WRITE,
    }
    assert _problems(tmp_path, {init: MAIN_OLD}, after, top="pkg:main") == []
    assert _problems(tmp_path / "b", {init: MAIN_OLD}, {STORE: ORPHAN}, top="pkg:main") == [
        _unreached(STORE, "orphan", "pkg:main")
    ]


def test_a_new_function_nothing_calls_is_reported(tmp_path: Path) -> None:
    """Rule: a new function the top caller does not reach is named with its file."""
    after = {APP: CALLS_WRITE, STORE: WRITE + "\n\n" + ORPHAN}
    assert _problems(tmp_path, {APP: MAIN_OLD}, after) == [_unreached(STORE, "orphan")]


def test_a_stub_left_in_the_chain_is_reported(tmp_path: Path) -> None:
    """Rule: a reached, new function whose body raises NotImplementedError is not built."""
    after = {APP: CALLS_WRITE, STORE: "def write(value):\n    raise NotImplementedError\n"}
    assert _problems(tmp_path, {APP: MAIN_OLD}, after) == [_stub(STORE, "write")]


def test_a_top_caller_that_is_still_a_stub_is_reported(tmp_path: Path) -> None:
    """Rule: the top caller itself is on the path: changed and still a stub, it is refused."""
    before = {APP: 'def main():\n    """Old."""\n    raise NotImplementedError\n'}
    after = {APP: 'def main():\n    """New."""\n    raise NotImplementedError\n'}
    assert _problems(tmp_path, before, after) == [_stub(APP, "main")]


def test_functions_the_diff_does_not_touch_are_never_reported(tmp_path: Path) -> None:
    """Rule: only what the diff adds or changes is judged: an old orphan or stub is left alone.

    The change adds a function above the old ones (their lines move) and makes the top caller
    call the old stub; neither old function's own code changes.
    """
    old = (
        "def old_orphan():\n    return 1\n\n\ndef old_stub():\n    raise NotImplementedError\n\n\n"
        "class Old:\n    def lonely(self):\n        raise NotImplementedError\n"
    )
    after = {
        APP: "from pkg import store\n\n\ndef main():\n    store.fresh()\n    store.old_stub()\n",
        STORE: "def fresh():\n    return 2\n\n\n" + old,
    }
    assert _problems(tmp_path, {APP: MAIN_OLD, STORE: old}, after) == []


def test_an_unchanged_file_is_never_reported(tmp_path: Path) -> None:
    """Rule: a file outside the diff is not judged, whatever it holds."""
    before = {APP: MAIN_OLD, STORE: ORPHAN, UTIL: "def todo():\n    raise NotImplementedError\n"}
    assert _problems(tmp_path, before, {"README.md": "y\n"}) == []


def test_methods_are_named_by_their_qualified_name(tmp_path: Path) -> None:
    """Rule: a method is ``Class.method``; an uncalled one and a stubbed one are both named."""
    store = (
        "class Store:\n    def __init__(self):\n        self.rows = []\n\n"
        "    def write(self, row):\n        raise NotImplementedError('later')\n\n"
        "    def unused(self):\n        return 0\n\n"
        "    class Cursor:\n        def lost(self):\n            return 0\n"
    )
    app = "from pkg.store import Store\n\n\ndef main():\n    return Store().write(1)\n"
    found = _problems(tmp_path, {APP: MAIN_OLD}, {APP: app, STORE: store})
    assert sorted(found) == sorted(
        [
            _stub(STORE, "Store.write"),
            _unreached(STORE, "Store.unused"),
            _unreached(STORE, "Store.Cursor.lost"),
        ]
    )


def test_a_used_class_reaches_its_dunder_methods_and_its_bases(tmp_path: Path) -> None:
    """Rule: Python calls ``__init__`` and friends itself; a base class is used with its child."""
    store = (
        "from pkg.util import Base\n\n\nclass Store(Base):\n"
        "    def __enter__(self):\n        return self\n\n"
        "    def __exit__(self, *exc):\n        return False\n"
    )
    util = "class Base:\n    def __init__(self):\n        self.rows = []\n"
    app = "from pkg.store import Store\n\n\ndef main():\n    with Store() as s:\n        return s\n"
    assert _problems(tmp_path, {APP: MAIN_OLD}, {APP: app, STORE: store, UTIL: util}) == []


def test_a_stubbed_dunder_method_of_a_used_class_is_reported(tmp_path: Path) -> None:
    """Rule: a reached ``__init__`` that still raises NotImplementedError is not built."""
    store = "class Store:\n    def __init__(self):\n        raise NotImplementedError\n"
    app = "from pkg.store import Store\n\n\ndef main():\n    return Store()\n"
    found = _problems(tmp_path, {APP: MAIN_OLD}, {APP: app, STORE: store})
    assert found == [_stub(STORE, "Store.__init__")]


@pytest.mark.parametrize(
    "app",
    [
        "from pkg.store import handle\n\nHANDLERS = {'a': handle}\n\n\n"
        "def main():\n    return HANDLERS['a'](1)\n",
        "from pkg.store import handle\n\n\ndef main():\n    table = {'a': handle}\n"
        "    return table['a'](1)\n",
        "from pkg.store import handle\n\n\ndef main():\n    return list(map(handle, [1]))\n",
        "from pkg.store import handle\n\n\ndef main():\n    return sorted([1], key=handle)\n",
        "import functools\n\nfrom pkg.store import handle\n\n\n"
        "def main():\n    return functools.partial(handle, 1)()\n",
        "from pkg import store\n\n\ndef main():\n    callback = store.handle\n"
        "    return callback(1)\n",
        "from pkg.store import handle\n\n\ndef main():\n    return (lambda: handle(1))()\n",
    ],
    ids=["module-table", "local-table", "map", "key", "partial", "attribute", "lambda"],
)
def test_a_function_used_as_a_callback_or_in_a_table_counts_as_reached(
    tmp_path: Path, app: str
) -> None:
    """Rule: reading a function's name on the path reaches it, not only calling it."""
    store = "def handle(value):\n    return value\n"
    assert _problems(tmp_path, {APP: MAIN_OLD}, {APP: app, STORE: store}) == []


def test_module_top_level_code_of_a_reached_module_reaches_what_it_names(tmp_path: Path) -> None:
    """Rule: importing a reached module runs its top-level code; what that names is reached."""
    store = "def _build():\n    return {}\n\n\nTABLE = _build()\n\n\n" + WRITE
    assert _problems(tmp_path, {APP: MAIN_OLD}, {APP: CALLS_WRITE, STORE: store}) == []


@pytest.mark.parametrize(
    "top",
    ["pkg.nope:main", "pkg.app:nope", "nope:main", "pkg.app.main:main", "outside.app:main"],
)
def test_a_top_caller_that_does_not_exist_is_the_only_problem(tmp_path: Path, top: str) -> None:
    """Rule: a missing module or function gives one problem and nothing else is judged."""
    after = {
        APP: CALLS_WRITE,
        STORE: WRITE + "\n\n" + ORPHAN,
        "outside/app.py": MAIN_OLD,  # a module outside the root is no top caller
    }
    assert _problems(tmp_path, {APP: MAIN_OLD}, after, top=top) == [
        f"top caller {top} does not exist"
    ]


def test_a_top_caller_deleted_by_the_change_does_not_exist(tmp_path: Path) -> None:
    """Rule: the top caller is looked for in the change's result, not in the base."""
    after = {APP: "def other():\n    return 0\n"}
    assert _problems(tmp_path, {APP: MAIN_OLD}, after) == [f"top caller {TOP} does not exist"]


def test_files_outside_the_root_are_ignored(tmp_path: Path) -> None:
    """Rule: only functions under the root are judged; tools and tests are not."""
    after = {
        APP: CALLS_WRITE,
        STORE: WRITE,
        "tools/helper.py": "def unused_tool():\n    raise NotImplementedError\n",
        "tests/unit/test_app.py": "def test_x():\n    assert True\n",
        "srcx/pkg/extra.py": ORPHAN,
    }
    assert _problems(tmp_path, {APP: MAIN_OLD}, after) == []


def test_the_root_can_be_another_directory(tmp_path: Path) -> None:
    """Rule: ``root`` moves the whole check: the top caller and the judged files lie under it."""
    after = {
        "lib/pkg/app.py": "def main():\n    return 0\n\n\ndef lost():\n    return 1\n",
        STORE: ORPHAN,
    }
    found = _problems(tmp_path, {}, after, root="lib/")
    assert found == [_unreached("lib/pkg/app.py", "lost")]


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('def orphan():\n    """Old words."""\n    return 1\n', "is not reachable"),
        ('def write(value):\n    """Old words."""\n    raise NotImplementedError\n', "still"),
    ],
    ids=["unreached", "stub"],
)
def test_a_docstring_only_change_counts_as_changed(tmp_path: Path, old: str, new: str) -> None:
    """Rule: a function whose docstring alone changed is changed, and is judged."""
    name = "orphan" if "orphan" in old else "write"
    before = {APP: CALLS_WRITE, STORE: WRITE if name == "orphan" else "", UTIL: old}
    if name == "write":
        before[APP] = CALLS_WRITE.replace("pkg.store", "pkg.util")
    after = {UTIL: old.replace("Old words", "New words")}
    found = _problems(tmp_path, before, after)
    assert len(found) == 1
    assert found[0].startswith(f"{UTIL}:{name} ") and new in found[0]


# ------------------------------------------------------------------- attacks: unreachable
@pytest.mark.parametrize(
    "app",
    [
        "from pkg.store import orphan\n\n\ndef main():\n    return 0\n",
        "from pkg.store import orphan as _o\n\n\ndef main():\n    return 0\n",
        "import pkg.store\n\n\ndef main():\n    return 0\n",
        "__all__ = ['orphan']\n\n\ndef main():\n    return 'orphan'\n",
        "def main():\n    # orphan() is called later\n    return 0\n",
        'def main():\n    """Calls orphan()."""\n    return 0\n',
        "def main():\n    return dict(orphan=1)\n",
        "def main():\n    orphan = 1\n    return 0\n",
        "def main(orphan=None):\n    return 0\n",
        "def main(box=None):\n    box.orphan = 1\n    return 0\n",
        "def main():\n    return getattr(__import__('pkg.store'), 'orphan')\n",
    ],
    ids=[
        "import-only",
        "import-as",
        "import-module",
        "dunder-all",
        "comment",
        "docstring",
        "keyword-name",
        "assigned-local",
        "parameter",
        "attribute-store",
        "string-getattr",
    ],
)
def test_a_mention_that_is_not_a_read_does_not_reach_a_function(tmp_path: Path, app: str) -> None:
    """Rule: only a name the path reads reaches a function; text and unused imports do not."""
    found = _problems(tmp_path, {APP: MAIN_OLD}, {APP: app, STORE: ORPHAN})
    assert found == [_unreached(STORE, "orphan")]


@pytest.mark.parametrize(
    "outside",
    [
        "tests/unit/test_store.py",
        "tools/run.py",
        "scripts/main.py",
        "srcx/pkg/app.py",
        "conftest.py",
    ],
)
def test_a_call_that_exists_only_outside_the_root_does_not_reach(
    tmp_path: Path, outside: str
) -> None:
    """Rule: a function only a test (or any file outside the root) calls is not reachable."""
    caller = "from pkg.store import orphan\n\n\ndef test_orphan():\n    assert orphan() == 1\n"
    found = _problems(tmp_path, {APP: MAIN_OLD}, {STORE: ORPHAN, outside: caller})
    assert found == [_unreached(STORE, "orphan")]


def test_unreached_functions_calling_each_other_are_all_reported(tmp_path: Path) -> None:
    """Rule: a call from unreached code reaches nothing: a dead chain or cycle stays dead."""
    store = (
        "def first():\n    return second()\n\n\ndef second():\n    return first()\n\n\n"
        "def alone():\n    return alone()\n"
    )
    found = _problems(tmp_path, {APP: MAIN_OLD}, {STORE: store})
    assert sorted(found) == sorted(_unreached(STORE, n) for n in ("first", "second", "alone"))


def test_top_level_code_of_an_unreached_module_reaches_nothing(tmp_path: Path) -> None:
    """Rule: a module nothing on the path names is not imported: its own calls do not count."""
    store = "def orphan():\n    return 1\n\n\nVALUE = orphan()\n"
    assert _problems(tmp_path, {APP: MAIN_OLD}, {STORE: store}) == [_unreached(STORE, "orphan")]


def test_a_call_the_change_removes_no_longer_reaches(tmp_path: Path) -> None:
    """Rule: the path is read from the change's result: a call only the base had is gone."""
    before = {APP: CALLS_WRITE, STORE: WRITE}
    after = {APP: MAIN_OLD, STORE: "def write(value):\n    return [value]\n"}
    assert _problems(tmp_path, before, after) == [_unreached(STORE, "write")]


def test_a_nested_function_nothing_calls_is_reported_with_what_only_it_calls(
    tmp_path: Path,
) -> None:
    """Rule: a nested function is its own scope: never called, its calls reach nothing."""
    app = (
        "from pkg.store import orphan\n\n\ndef main():\n"
        "    def inner():\n        return orphan()\n\n    return 0\n"
    )
    found = _problems(tmp_path, {APP: MAIN_OLD}, {APP: app, STORE: ORPHAN})
    assert sorted(found) == sorted([_unreached(APP, "main.inner"), _unreached(STORE, "orphan")])


def test_a_nested_function_that_is_called_reaches_what_it_calls(tmp_path: Path) -> None:
    """Rule: a nested function the path calls is on the path."""
    app = (
        "from pkg.store import write\n\n\ndef main():\n"
        "    def inner():\n        return write(1)\n\n    return inner()\n"
    )
    assert _problems(tmp_path, {APP: MAIN_OLD}, {APP: app, STORE: WRITE}) == []


def test_a_nested_stub_on_the_path_is_reported(tmp_path: Path) -> None:
    """Rule: a called nested function that raises NotImplementedError is not built."""
    app = "def main():\n    def inner():\n        raise NotImplementedError\n\n    return inner()\n"
    found = _problems(tmp_path, {APP: MAIN_OLD}, {APP: app})
    assert _stub(APP, "main.inner") in found
    assert all("still raises not implemented" in p for p in found)


def test_async_functions_are_judged_like_any_other(tmp_path: Path) -> None:
    """Rule: ``async def`` is a function: the top caller, an orphan and a stub are all seen."""
    app = "from pkg.store import fetch\n\n\nasync def main():\n    return await fetch()\n"
    store = (
        "async def fetch():\n    raise NotImplementedError\n\n\n"
        "async def lost():\n    return 1\n\n\n"
        "class Client:\n    async def close(self):\n        return None\n"
    )
    found = _problems(tmp_path, {APP: MAIN_OLD}, {APP: app, STORE: store})
    assert sorted(found) == sorted(
        [_stub(STORE, "fetch"), _unreached(STORE, "lost"), _unreached(STORE, "Client.close")]
    )


def test_decorated_functions_are_judged_like_any_other(tmp_path: Path) -> None:
    """Rule: a decorator hides nothing: a decorated orphan and a decorated stub are refused."""
    app = "from pkg.store import Box, cached\n\n\ndef main():\n    return cached() + Box().value\n"
    store = (
        "from functools import cache\n\n\n"
        "@cache\ndef cached():\n    raise NotImplementedError\n\n\n"
        "@cache\ndef lost():\n    return 1\n\n\n"
        "class Box:\n    @property\n    def value(self):\n        raise NotImplementedError\n\n"
        "    @staticmethod\n    def spare():\n        return 1\n"
    )
    found = _problems(tmp_path, {APP: MAIN_OLD}, {APP: app, STORE: store})
    assert sorted(found) == sorted(
        [
            _stub(STORE, "cached"),
            _unreached(STORE, "lost"),
            _stub(STORE, "Box.value"),
            _unreached(STORE, "Box.spare"),
        ]
    )


def test_a_method_of_a_used_class_that_nothing_names_is_reported(tmp_path: Path) -> None:
    """Rule: using a class reaches what Python calls itself, not every method it has."""
    store = (
        "class Store:\n    def __init__(self):\n        self.rows = []\n\n"
        "    def spare(self):\n        return 0\n"
    )
    app = "from pkg.store import Store\n\n\ndef main():\n    return Store()\n"
    found = _problems(tmp_path, {APP: MAIN_OLD}, {APP: app, STORE: store})
    assert found == [_unreached(STORE, "Store.spare")]


# ------------------------------------------------------------------------ attacks: stubs
_HELPER = "def todo():\n    raise NotImplementedError\n"
STUBS = {
    "bare": "def write(value):\n    raise NotImplementedError\n",
    "called": "def write(value):\n    raise NotImplementedError()\n",
    "message": "def write(value):\n    raise NotImplementedError('the write path comes later')\n",
    "from-none": "def write(value):\n    raise NotImplementedError('later') from None\n",
    "after-docstring": 'def write(value):\n    """Write."""\n    raise NotImplementedError\n',
    "one-branch": (
        "def write(value):\n    if value:\n        raise NotImplementedError('lists')\n"
        "    return value\n"
    ),
    "in-try": (
        "def write(value):\n    try:\n        return int(value)\n"
        "    except ValueError:\n        raise NotImplementedError from None\n"
    ),
    "parenthesised": "def write(value):\n    raise (NotImplementedError)\n",
    "builtins": "import builtins\n\n\ndef write(value):\n    raise builtins.NotImplementedError\n",
    "variable": "def write(value):\n    err = NotImplementedError('later')\n    raise err\n",
    "alias": "TODO = NotImplementedError\n\n\ndef write(value):\n    raise TODO('later')\n",
    "old-helper": "from pkg.util import todo\n\n\ndef write(value):\n    todo()\n",
    "old-helper-returned": "from pkg.util import todo\n\n\ndef write(value):\n    return todo()\n",
}


@pytest.mark.parametrize("form", sorted(STUBS))
def test_every_form_of_a_stub_on_the_path_is_reported(tmp_path: Path, form: str) -> None:
    """Rule 4 of FLOW-HYBRID.md: a function that still raises not implemented is not built.

    ``write`` is new and reached; its whole body ends in NotImplementedError, raised bare, as a
    call, through the builtins module, a variable, an alias, or a helper that existed before the
    change (so the helper itself is not in the diff).
    """
    before = {APP: MAIN_OLD, UTIL: _HELPER}
    found = _problems(tmp_path, before, {APP: CALLS_WRITE, STORE: STUBS[form]})
    assert found == [_stub(STORE, "write")]


def test_a_stub_subclass_of_not_implemented_is_reported(tmp_path: Path) -> None:
    """Rule 4: an error class derived from NotImplementedError is still not implemented."""
    util = "class Todo(NotImplementedError):\n    pass\n"
    store = "from pkg.util import Todo\n\n\ndef write(value):\n    raise Todo('later')\n"
    found = _problems(tmp_path, {APP: MAIN_OLD, UTIL: util}, {APP: CALLS_WRITE, STORE: store})
    assert found == [_stub(STORE, "write")]


@pytest.mark.parametrize(
    "body",
    [
        "    try:\n        return value.save()\n"
        "    except NotImplementedError:\n        return None\n",
        "    return isinstance(value, NotImplementedError)\n",
        "    return 'raise NotImplementedError'\n",
        "    return NotImplemented\n",
    ],
    ids=["except", "isinstance", "string", "not-implemented-constant"],
)
def test_a_built_function_that_only_names_the_error_is_not_a_stub(
    tmp_path: Path, body: str
) -> None:
    """Rule: only raising NotImplementedError makes a stub; handling or naming it does not."""
    store = "def write(value):\n" + body
    assert _problems(tmp_path, {APP: MAIN_OLD}, {APP: CALLS_WRITE, STORE: store}) == []


def test_a_stub_that_is_not_reached_is_reported_as_unreachable(tmp_path: Path) -> None:
    """Rule: every added function gets a verdict: an unreached stub is named unreachable."""
    store = "def write(value):\n    raise NotImplementedError\n"
    assert _problems(tmp_path, {APP: MAIN_OLD}, {STORE: store}) == [_unreached(STORE, "write")]


def test_every_problem_of_one_change_is_listed(tmp_path: Path) -> None:
    """Rule: the gate lists every unreached and every stubbed function, not only the first."""
    after = {
        APP: CALLS_WRITE,
        STORE: "def write(value):\n    raise NotImplementedError\n\n\n" + ORPHAN,
        UTIL: "def lost():\n    return 0\n",
    }
    found = _problems(tmp_path, {APP: MAIN_OLD}, after)
    assert sorted(found) == sorted(
        [_stub(STORE, "write"), _unreached(STORE, "orphan"), _unreached(UTIL, "lost")]
    )


# ------------------------------------------------------------------------- the pipeline
ADD = "memini.core.calc:add"
EXTRA = GOOD + "\n\ndef extra():\n    return 1\n"
HALF = (
    "def add(a, b):\n    return _sum(a, b)\n\n\n"
    "def _sum(a, b):\n    if a is None:\n        raise NotImplementedError\n    return a + b\n"
)


@pytest.fixture(autouse=True)
def _qa_runner_answers_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the QA role's runner: serve its open test reviews before a slice waits."""
    wait = moves.wait

    def served(cfg: Any, move_id: str, timeout_s: float, poll_s: float = 5.0) -> Any:
        moves.serve(cfg, "qa", {"test_review": pipeline.review_move}, once=True)
        return wait(cfg, move_id, timeout_s, poll_s)

    monkeypatch.setattr(moves, "wait", served)


def _verifier_prompts(env: dict[str, Any]) -> list[str]:
    return [p for p in _prompts(env, "audit") if p.startswith("You are the Verifier")]


def test_an_unreached_function_goes_back_to_the_coder(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: the gate's problem is the coder's next failure, like a D12 finding."""
    _spec(env, top_caller=ADD)
    _plan(env, monkeypatch, coder=_coder({IMPL: EXTRA}, {IMPL: GOOD}))
    out = pipeline.run_task(_cfg(env, test_review=False), _task(env))
    assert out.status == "merged", out.reason
    prompts = _prompts(env, "coder")
    assert len(prompts) == 2
    assert f"{IMPL}:extra is not reachable from {ADD}" in prompts[1]


def test_a_stub_on_the_path_goes_back_to_the_coder(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: code that passes the tests but still raises not implemented is not merged."""
    _spec(env, top_caller=ADD)
    _plan(env, monkeypatch, coder=_coder({IMPL: HALF}, {IMPL: GOOD}))
    out = pipeline.run_task(_cfg(env, test_review=False), _task(env))
    assert out.status == "merged", out.reason
    prompts = _prompts(env, "coder")
    assert len(prompts) == 2
    assert f"{IMPL}:_sum still raises not implemented" in prompts[1]


def test_a_slice_that_never_fixes_the_gate_problem_is_not_merged(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: the gate blocks the merge; it is not advice the coder may ignore."""
    _spec(env, top_caller=ADD)
    _plan(env, monkeypatch, coder=_coder({IMPL: EXTRA}))
    main = _git(env["repo"], "rev-parse", "main")
    out = pipeline.run_task(_cfg(env, test_review=False, max_coder_attempts=2), _task(env))
    assert out.status != "merged"
    assert _git(env["repo"], "rev-parse", "main") == main
    assert "is not reachable from" in out.reason


def test_a_top_caller_that_does_not_exist_blocks_the_slice(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: a spec whose top caller is not in the code the coder wrote is told to the coder."""
    top = "memini.core.calc:run"
    _spec(env, top_caller=top)
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD}))
    out = pipeline.run_task(_cfg(env, test_review=False, max_coder_attempts=2), _task(env))
    assert out.status != "merged"
    assert f"top caller {top} does not exist" in _prompts(env, "coder")[1]


def test_without_a_top_caller_the_gate_is_skipped(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: a spec with no top caller (a stub-only or old ticket) is not checked."""
    _plan(env, monkeypatch, coder=_coder({IMPL: EXTRA}))
    out = pipeline.run_task(_cfg(env, test_review=False), _task(env))
    assert out.status == "merged", out.reason
    assert len(_prompts(env, "coder")) == 1


def test_the_gate_judges_the_coders_change_not_what_main_already_had(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: the base is the QA commit: old unreached code and old stubs on main are left alone."""
    repo = env["repo"]
    (repo / "src" / "memini" / "core" / "legacy.py").write_text(
        "def old_orphan():\n    return 1\n\n\ndef old_stub():\n    raise NotImplementedError\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "old code")
    _spec(env, top_caller=ADD)
    _plan(env, monkeypatch, coder=_coder({IMPL: GOOD}))
    out = pipeline.run_task(_cfg(env, test_review=False), _task(env))
    assert out.status == "merged", out.reason
    assert len(_prompts(env, "coder")) == 1


def test_the_verifier_is_told_the_gate_checked_reachability_and_the_scope(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: the Verifier gets the machine result and the ticket scope, and the slice merges."""
    _spec(env, top_caller=ADD)
    cfg = _ticket(env, env["tmp"])
    _plan(
        env,
        monkeypatch,
        coder=_coder({IMPL: GOOD}),
        audit=[{"stdout": AUDIT_PASS}, {"stdout": _answer(_row())}],
    )
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "merged", out.reason
    [prompt] = _verifier_prompts(env)
    assert f"reachable from {ADD}: checked by the machine gate" in prompt
    assert "no top caller on this ticket" not in prompt.split("Reachability (machine gate):")[1]
    assert "paths: src/memini/core/calc/" in prompt
    assert "out_of_scope:\nsubtraction" in prompt
    assert "$reachability" not in prompt and "$scope" not in prompt
    assert "| A1 | met |" in (cfg.state_dir / "runs" / TASK / "verifier-1.md").read_text()


def test_the_verifier_is_told_when_the_ticket_has_no_top_caller(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: no top caller: the Verifier reads exactly that, never a claim that it was checked."""
    cfg = _ticket(env, env["tmp"])
    _plan(
        env,
        monkeypatch,
        coder=_coder({IMPL: GOOD}),
        audit=[{"stdout": AUDIT_PASS}, {"stdout": _answer(_row())}],
    )
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "merged", out.reason
    [prompt] = _verifier_prompts(env)
    told = prompt.split("Reachability (machine gate):")[1].split("Scope of the ticket:")[0]
    assert "no top caller on this ticket" in told
    assert "checked by the machine gate" not in told
    assert "paths: src/memini/core/calc/" in prompt


def test_the_verifier_prompt_no_longer_asks_the_model_to_judge_reachability() -> None:
    """Rule: reachability is the machine's fact; the prompt holds each placeholder once."""
    cfg = _cfg_shipped()
    text = (cfg.prompts_dir / "verifier.txt").read_text(encoding="utf-8")
    assert text.count("$reachability") == 1 and text.count("$scope") == 1
    assert "A function nothing calls does not meet a line" not in text
    assert "the code is reachable from the entry point" not in text
    rendered = engine.render(
        cfg, "verifier", task_id="T", cards="C", diff="D", tests="X", scope="S", reachability="R"
    )
    assert "$" not in rendered


def _cfg_shipped() -> Any:
    from tools.orchestrator.config import load_config  # noqa: PLC0415

    return load_config(Path.cwd(), overrides={"state_dir": "/nonexistent-state"}, env={})


@pytest.mark.parametrize("reachable", [True, False])
def test_the_verifier_answer_is_still_parsed(reachable: bool) -> None:
    """Rule: the answer keeps its shape: a met, tested line passes only when reachable is true."""
    found = audit.parse_verifier(_answer(_row(reachable=reachable)), ["A1"])
    assert found.ok is reachable
    assert bool(found.code_problems) is not reachable


def test_every_agent_of_a_slice_gets_the_same_rules_with_the_top_caller(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: one rules source: qa, coder, test review and audit read the identical text."""
    _spec(env, top_caller=ADD)
    _plan(
        env,
        monkeypatch,
        coder=_coder({IMPL: GOOD, TEST: EDITED}),
        audit=[{"stdout": APPROVE}, {"stdout": AUDIT_PASS}],
    )
    cfg = _cfg(env, test_review=True)
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "merged", out.reason
    rules = engine.rules_text(cfg, ADD)
    assert rules != engine.rules_text(cfg)
    review, audit_prompt = _prompts(env, "audit")
    for prompt in (_prompts(env, "qa")[0], _prompts(env, "coder")[0], review, audit_prompt):
        assert rules in prompt


def test_without_a_top_caller_no_agent_gets_the_outside_in_rules(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule: a slice with no top caller reads the old rules text, unchanged."""
    _plan(
        env,
        monkeypatch,
        coder=_coder({IMPL: GOOD, TEST: EDITED}),
        audit=[{"stdout": APPROVE}, {"stdout": AUDIT_PASS}],
    )
    cfg = _cfg(env, test_review=True)
    out = pipeline.run_task(cfg, _task(env))
    assert out.status == "merged", out.reason
    rules = engine.rules_text(cfg)
    review, audit_prompt = _prompts(env, "audit")
    for prompt in (_prompts(env, "qa")[0], _prompts(env, "coder")[0], review, audit_prompt):
        assert rules in prompt
        assert "checked by the reachability gate" not in prompt
