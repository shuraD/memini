"""The orchestrator's AST policy lets ordinary product code through and still refuses the cheats.

Owner 2026-11-15 ("agree for AST list"): LP-02 (``__slots__``) and LP-08 (``sqlite3``,
``threading``) were rejected by an allow-list that was too narrow for infra code. The added
modules must pass; their dangerous corners and every earlier cheat must still be refused.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.orchestrator import astcheck, config, gates

ROOT = Path(__file__).resolve().parents[2]


def _problems(source: str) -> list[str]:
    cfg = config.load_config(ROOT, overrides={"state_dir": "/nonexistent-state"}, env={})
    return astcheck.problems("src/memini/x.py", source, astcheck.rules(cfg, ROOT))


@pytest.mark.parametrize(
    "source",
    [
        # LP-08: one shared SQLite file, one connection per thread
        "import sqlite3\nimport threading\n\n_LOCAL = threading.local()\n"
        "_LOCK = threading.Lock()\n\n\n"
        "def conn(path: str) -> sqlite3.Connection:\n    return sqlite3.connect(path)\n",
        # LP-02: a light immutable settings class
        "class A:\n    __slots__ = ('a', 'b')\n",
        "from types import MappingProxyType, SimpleNamespace, TracebackType\n\n"
        "M = MappingProxyType({})\n",
        "from concurrent.futures import ThreadPoolExecutor, as_completed\n",
        "import queue\nimport contextvars\n\nQ = queue.Queue()\nV = contextvars.ContextVar('v')\n",
        "from urllib.request import Request, urlopen\n"
        "from urllib.error import HTTPError, URLError\n",
        "import http.client\nfrom email.utils import parsedate_to_datetime\n",
        # protocol methods were never checked by name; a context manager and a sequence pass
        "class C:\n    def __enter__(self): return self\n    def __exit__(self, *a): return None\n"
        "    def __iter__(self): return iter(())\n    def __len__(self): return 0\n"
        "    def __lt__(self, o): return False\n    def __contains__(self, x): return False\n",
    ],
)
def test_ordinary_infra_code_passes_the_ast_check(source: str) -> None:
    assert _problems(source) == []


@pytest.mark.parametrize(
    ("source", "problem"),
    [
        # the dangerous corners of the newly allowed modules
        (
            "from concurrent.futures import ProcessPoolExecutor\n",
            "ProcessPoolExecutor is forbidden",
        ),
        (
            "import concurrent.futures as cf\nE = cf.ProcessPoolExecutor\n",
            "ProcessPoolExecutor is forbidden",
        ),
        (
            "import concurrent.futures.process\n",
            "import of concurrent.futures.process is not allowed",
        ),
        ("from types import CodeType\n", "types.CodeType is forbidden"),
        ("from types import FunctionType\n", "types.FunctionType is forbidden"),
        ("import types\nF = types.LambdaType\n", "types.LambdaType is forbidden"),
        ("from types import FrameType\n", "types.FrameType is forbidden"),
        ("import types\nM = types.ModuleType('m')\n", "types.ModuleType is forbidden"),
        ("import threading\nthreading.settrace(print)\n", "threading.settrace is forbidden"),
        ("import threading\nthreading.setprofile(print)\n", "threading.setprofile is forbidden"),
        (
            "from threading import settrace_all_threads\n",
            "threading.settrace_all_threads is forbidden",
        ),
        ("import threading\nthreading.excepthook = print\n", "threading.excepthook is forbidden"),
        ("import threading\nthreading._shutdown()\n", "threading._shutdown is forbidden"),
        ("x.__slots__\n", "attribute __slots__ is not allowed"),
        # earlier cheats are untouched
        ("import subprocess\n", "import of subprocess is not allowed"),
        ("import multiprocessing\n", "import of multiprocessing is not allowed"),
        ("import sys\nM = sys.modules\n", "sys.modules is forbidden"),
        ("import os\nos.system('x')\n", "os.system is forbidden"),
        ("x = __builtins__\n", "name __builtins__ is not allowed"),
        ("def f():\n    pass\n\n\nG = f.__globals__\n", "attribute __globals__ is not allowed"),
    ],
)
def test_the_cheats_stay_refused(source: str, problem: str) -> None:
    found = _problems(source)
    assert any(problem in p for p in found), found


@pytest.mark.parametrize(
    "line", ["conn.enable_load_extension(True)", 'conn.load_extension("x.so")']
)
def test_loading_native_sqlite_extensions_is_caught_by_the_deny_scan(line: str) -> None:
    cfg = config.load_config(ROOT, overrides={"state_dir": "/nonexistent-state"}, env={})
    diff = f"+++ b/src/memini/x.py\n@@ -0,0 +1 @@\n+{line}\n"
    assert gates.deny_hits(diff, cfg.deny_patterns) != []
