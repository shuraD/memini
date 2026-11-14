"""Gate tests for ``tools.ci.pytest_gate``: the test step never reports a fake green.

Written from bug B-foundation-016 ("the test step can report green without running any
test") and the public contract DevOps stated for the fix, never from the implementation:

G1 ``junit_key`` maps a pytest node id to the ``(classname, name)`` pair pytest's junit
   report writes (file path, optional classes, parameters kept verbatim in the name).
G2 ``parse_junit`` reads the report from its test cases; a missing, empty or malformed
   report never looks green, and a lying ``failures="0"`` count does not hide a failure.
G3 ``uncollected_executed`` lists test cases that ran but were never collected; a collection
   error test case counts as an error, not as an uncollected run.
G4 ``collected_ids`` parses ``pytest --collect-only -q`` output into node ids only.
G5 ``verdict``: ``[]`` is the only green; every bad state, alone or combined, gives reasons.
   A non-zero collection exit code (``collect_exit_code``) is red on its own.
G6 ``split_args`` splits at the first ``--`` and never drops or reorders an argument.
G7 ``main`` runs pytest twice in safe-path mode (``python -P``) with a fresh junit report,
   returns 0 only for the clean case and non-zero for every bad state.

Junit and collect samples are written by hand in pytest's real format (checked against
pytest 8 output). ``main`` is tested in-process: ``subprocess.run`` is replaced by a fake
that returns canned collect output, writes a canned junit report and returns exit codes.
No git, no network.
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import quoteattr

import pytest
from tools.ci import pytest_gate as gate
from tools.ci.pytest_gate import RunReport

# --------------------------------------------------------------------------------------
# Samples in pytest's real format
# --------------------------------------------------------------------------------------
# node id -> the (classname, name) pytest's junit report writes for it
ID_TO_KEY: dict[str, tuple[str, str]] = {
    "tests/unit/test_x.py::test_a[a/b.py]": ("tests.unit.test_x", "test_a[a/b.py]"),
    "tests/unit/test_x.py::test_a[x y]": ("tests.unit.test_x", "test_a[x y]"),
    "tests/unit/test_x.py::test_a[m::n]": ("tests.unit.test_x", "test_a[m::n]"),
    "tests/unit/test_x.py::test_a[v.1]": ("tests.unit.test_x", "test_a[v.1]"),
    "tests/unit/test_x.py::test_a[p/q.r s::t]": ("tests.unit.test_x", "test_a[p/q.r s::t]"),
    "tests/unit/test_x.py::TestK::test_b": ("tests.unit.test_x.TestK", "test_b"),
    "tests/unit/test_x.py::TestK::TestInner::test_c": (
        "tests.unit.test_x.TestK.TestInner",
        "test_c",
    ),
    "tests/unit/test_x.py::TestK::test_d[a.b/c]": ("tests.unit.test_x.TestK", "test_d[a.b/c]"),
    "tests/sanity/test_s.py::test_ok": ("tests.sanity.test_s", "test_ok"),
    "tests/gates/sub/test_g.py::test_amp[a&b<c>]": ("tests.gates.sub.test_g", "test_amp[a&b<c>]"),
}
IDS = list(ID_TO_KEY)
KEYS = list(ID_TO_KEY.values())

FAILURE = '<failure message="AssertionError: assert False">E   AssertionError</failure>'
ERROR = '<error message="failed on setup with &quot;RuntimeError&quot;">E   RuntimeError</error>'
COLLECTION_ERROR = '<error message="collection failure">E   ImportError: x</error>'
COLLECTION_ERROR_KEY = ("", "tests.sanity.test_broken")  # pytest: empty classname
SYSTEM_OUT = "<system-out>chatter</system-out>"


def _case(key: tuple[str, str], child: str = "") -> str:
    classname, name = key
    return (
        f"<testcase classname={quoteattr(classname)} name={quoteattr(name)} "
        f'time="0.001">{child}</testcase>'
    )


def _junit(cases: Sequence[str], *, failures: int | None = None, errors: int | None = None) -> str:
    """A junit report as pytest writes it; counts are honest unless overridden."""
    body = "".join(cases)
    fails = body.count("<failure") if failures is None else failures
    errs = body.count("<error") if errors is None else errors
    skips = body.count("<skipped")
    return (
        '<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
        f'<testsuite name="pytest" errors="{errs}" failures="{fails}" skipped="{skips}" '
        f'tests="{len(cases)}" time="0.050" timestamp="2026-11-13T10:00:00+00:00" '
        f'hostname="ci">{body}</testsuite></testsuites>'
    )


def _clean_junit(keys: Sequence[tuple[str, str]] = KEYS) -> str:
    return _junit([_case(k) for k in keys])


def _junit_with(child_for: dict[int, str], keys: Sequence[tuple[str, str]] = KEYS) -> str:
    return _junit([_case(k, child_for.get(i, "")) for i, k in enumerate(keys)])


WARNINGS_BLOCK = (
    "=============================== warnings summary ===============================\n"
    "tests/unit/test_x.py:15\n"
    "  /work/proj/tests/unit/test_x.py:15: PytestCollectionWarning: cannot collect test "
    "class 'TestW' because it has a __init__ constructor (from: tests/unit/test_x.py)\n"
    "    class TestW:\n"
    "\n"
    "-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html\n"
)
ERRORS_BLOCK = (
    "==================================== ERRORS ====================================\n"
    "_________________ ERROR collecting tests/sanity/test_broken.py _________________\n"
    "ImportError while importing test module '/work/proj/tests/sanity/test_broken.py'.\n"
    "Traceback:\n"
    "tests/sanity/test_broken.py:1: in <module>\n"
    '    raise ImportError("bad id tests/unit/test_zz.py::test_zz")\n'
    "E   ImportError: bad id tests/unit/test_zz.py::test_zz\n"
    "=========================== short test summary info ============================\n"
    "ERROR tests/sanity/test_broken.py\n"
    "!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!\n"
)


def _collect_out(ids: Sequence[str], tail: str = "", summary: str | None = None) -> str:
    """``pytest --collect-only -q`` stdout: the ids, a blank line, blocks, the summary."""
    last = summary if summary is not None else f"{len(ids)} tests collected in 0.01s"
    head = "".join(f"{i}\n" for i in ids)
    return f"{head}\n{tail}{last}\n"


# --------------------------------------------------------------------------------------
# G1: junit_key
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(("node_id", "key"), list(ID_TO_KEY.items()), ids=IDS)
def test_g1_junit_key_matches_pytest_junit(node_id: str, key: tuple[str, str]) -> None:
    """G1: the key is pytest's junit pair: path dotted, ``.py`` cut, classes joined, params kept.

    Params containing ``/``, ``.``, spaces and ``::`` belong to the name and are not mangled.
    """
    assert gate.junit_key(node_id) == key


# --------------------------------------------------------------------------------------
# G2: parse_junit
# --------------------------------------------------------------------------------------
def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "report.xml"
    path.write_text(text, encoding="utf-8")
    return path


def test_g2_clean_report_lists_every_testcase(tmp_path: Path) -> None:
    """G2: a clean report is present and lists every test case, unescaped, with no failure."""
    cases = [_case(k, SYSTEM_OUT if i == 0 else "") for i, k in enumerate(KEYS)]
    report = gate.parse_junit(_write(tmp_path, _junit(cases)))
    assert report.present is True
    assert Counter(report.executed) == Counter(KEYS)
    assert report.failed == []
    assert report.errored == []


@pytest.mark.parametrize(
    ("child", "attr"),
    [(FAILURE, "failed"), (ERROR, "errored")],
    ids=["failure", "error"],
)
def test_g2_failure_or_error_child_is_reported(tmp_path: Path, child: str, attr: str) -> None:
    """G2: a ``<failure>`` / ``<error>`` child marks exactly that test case."""
    report = gate.parse_junit(_write(tmp_path, _junit_with({3: child})))
    marked = getattr(report, attr)
    assert len(marked) == 1, marked
    assert "test_a[v.1]" in marked[0]
    other = report.errored if attr == "failed" else report.failed
    assert other == []


def test_g2_failure_and_error_in_one_testcase_are_both_reported(tmp_path: Path) -> None:
    """G2: a test that fails and then errors in teardown shows as failed and as errored."""
    report = gate.parse_junit(_write(tmp_path, _junit_with({0: FAILURE + ERROR})))
    assert report.failed and report.errored


@pytest.mark.parametrize("child", [FAILURE, ERROR], ids=["failure", "error"])
def test_g2_lying_suite_counts_do_not_hide_a_failure(tmp_path: Path, child: str) -> None:
    """G2: the test cases decide, not the suite's ``failures``/``errors`` attributes.

    Attack: a report rewritten to say ``failures="0" errors="0"`` over a failing case.
    """
    cases = [_case(k, child if i == 1 else "") for i, k in enumerate(KEYS)]
    report = gate.parse_junit(_write(tmp_path, _junit(cases, failures=0, errors=0)))
    assert report.failed or report.errored


def test_g2_collection_error_testcase_is_an_error(tmp_path: Path) -> None:
    """G2: pytest's collection-error case (empty classname, ``<error>``) is an error."""
    junit = _junit([_case(COLLECTION_ERROR_KEY, COLLECTION_ERROR)])
    report = gate.parse_junit(_write(tmp_path, junit))
    assert report.errored


def test_g2_missing_report_is_not_present(tmp_path: Path) -> None:
    """G2: no report file means nothing is proven to have run."""
    report = gate.parse_junit(tmp_path / "never-written.xml")
    assert report.present is False
    assert report.executed == []


MALFORMED = {
    "empty-file": "",
    "not-xml": "all tests passed\n",
    "truncated-mid-suite": _clean_junit()[: len(_clean_junit()) // 2],
    "unclosed-root": '<?xml version="1.0"?><testsuites><testsuite name="pytest">',
    "garbage-after-root": _clean_junit() + "<testsuites>",
}


@pytest.mark.parametrize("text", list(MALFORMED.values()), ids=list(MALFORMED))
def test_g2_malformed_report_is_not_present(tmp_path: Path, text: str) -> None:
    """G2: an unreadable report proves nothing ran: not present, no executed case."""
    report = gate.parse_junit(_write(tmp_path, text))
    assert report.present is False
    assert report.executed == []


@pytest.mark.parametrize("text", list(MALFORMED.values()), ids=list(MALFORMED))
def test_g2_malformed_report_is_never_green(tmp_path: Path, text: str) -> None:
    """G2+G5: whatever ``parse_junit`` makes of a malformed report, the verdict is red."""
    path = _write(tmp_path, text)
    report = gate.parse_junit(path)
    missing = gate.executed_vs_collected(path, IDS)
    assert gate.verdict(IDS, report, missing, 0) != []


# --------------------------------------------------------------------------------------
# G3: uncollected_executed, executed_vs_collected
# --------------------------------------------------------------------------------------
def test_g3_every_executed_case_collected_gives_nothing() -> None:
    """G3 control: a run of exactly the collected ids has no uncollected execution."""
    assert gate.uncollected_executed(RunReport(executed=list(KEYS)), IDS) == []


def test_g3_extra_executed_case_is_reported() -> None:
    """G3: a test case that ran but was never collected is listed as ``classname::name``."""
    extra = ("tests.unit.test_extra", "test_hidden")
    report = RunReport(executed=[*KEYS, extra])
    assert gate.uncollected_executed(report, IDS) == ["tests.unit.test_extra::test_hidden"]


def test_g3_everything_uncollected_when_collection_is_empty() -> None:
    """G3: with an empty collection every executed case is an uncollected run."""
    assert len(gate.uncollected_executed(RunReport(executed=list(KEYS)), [])) == len(KEYS)


def test_g3_collection_error_case_is_not_an_uncollected_run(tmp_path: Path) -> None:
    """G3: pytest's collection-error case counts as an error, not as an uncollected run.

    The report is read by ``parse_junit`` from pytest's real format (empty classname).
    """
    cases = [*(_case(k) for k in KEYS), _case(COLLECTION_ERROR_KEY, COLLECTION_ERROR)]
    report = gate.parse_junit(_write(tmp_path, _junit(cases)))
    assert report.errored
    assert gate.uncollected_executed(report, IDS) == []


def test_g3_passing_case_disguised_as_collection_error_is_uncollected(tmp_path: Path) -> None:
    """G3: an empty-classname case WITHOUT an error is a real run, so it is uncollected.

    Attack: a hook that runs an extra test and names it like a collection error.
    """
    cases = [*(_case(k) for k in KEYS), _case(("", "tests.unit.test_sneaky"))]
    report = gate.parse_junit(_write(tmp_path, _junit(cases)))
    assert gate.uncollected_executed(report, IDS) != []


def test_g3_same_name_in_another_class_is_uncollected() -> None:
    """G3: identity is (classname, name): a same-named test in another class is not collected."""
    report = RunReport(executed=[("tests.unit.test_x.TestOther", "test_b")])
    assert gate.uncollected_executed(report, ["tests/unit/test_x.py::TestK::test_b"]) != []


def test_g3_missing_report_means_no_collected_id_ran(tmp_path: Path) -> None:
    """G3: a missing report means none ran: every collected id is missing."""
    assert gate.executed_vs_collected(tmp_path / "none.xml", IDS) == IDS


def test_g3_every_collected_id_found_in_report(tmp_path: Path) -> None:
    """G3 control: every collected id (params, classes, escapes) is found in its report."""
    assert gate.executed_vs_collected(_write(tmp_path, _clean_junit()), IDS) == []


# --------------------------------------------------------------------------------------
# G4: collected_ids
# --------------------------------------------------------------------------------------
COLLECT_OUTPUTS = {
    "plain": _collect_out(IDS),
    "warnings-summary": _collect_out(IDS, WARNINGS_BLOCK),
    "collection-errors": _collect_out(
        IDS, ERRORS_BLOCK, f"{len(IDS)} tests collected, 1 error in 0.04s"
    ),
    "deselected": _collect_out(IDS, summary=f"{len(IDS)}/12 tests collected (2 deselected)"),
    "windows-newlines": _collect_out(IDS).replace("\n", "\r\n"),
}


@pytest.mark.parametrize("out", list(COLLECT_OUTPUTS.values()), ids=list(COLLECT_OUTPUTS))
def test_g4_collected_ids_are_exactly_the_node_ids(out: str) -> None:
    """G4: only node ids are ids; summary, warning, error and traceback lines are ignored."""
    assert gate.collected_ids(out) == IDS


EMPTY_COLLECT_OUTPUTS = {
    "no-output": "",
    "no-tests": "\nno tests collected in 0.01s\n",
    "only-warnings": "\n" + WARNINGS_BLOCK + "no tests collected in 0.01s\n",
    "only-errors": "\n" + ERRORS_BLOCK + "no tests collected, 1 error in 0.04s\n",
}


@pytest.mark.parametrize(
    "out", list(EMPTY_COLLECT_OUTPUTS.values()), ids=list(EMPTY_COLLECT_OUTPUTS)
)
def test_g4_output_without_node_ids_collects_nothing(out: str) -> None:
    """G4: output that lists no node id gives no id (a path in a message is not an id)."""
    assert gate.collected_ids(out) == []


def test_g4_single_test_summary() -> None:
    """G4: the singular summary ``1 test collected`` is not an id."""
    assert gate.collected_ids(_collect_out(IDS[:1], summary="1 test collected in 0.00s")) == IDS[:1]


# --------------------------------------------------------------------------------------
# G5: verdict
# --------------------------------------------------------------------------------------
UNCOLLECTED = ("tests.unit.test_extra", "test_hidden")


@dataclass
class State:
    collected: list[str] = field(default_factory=lambda: list(IDS))
    executed: list[tuple[str, str]] = field(default_factory=lambda: list(KEYS))
    failed: list[str] = field(default_factory=list)
    errored: list[str] = field(default_factory=list)
    present: bool = True
    missing: list[str] = field(default_factory=list)
    exit_code: int = 0

    def reasons(self) -> list[str]:
        report = RunReport(
            executed=self.executed, failed=self.failed, errored=self.errored, present=self.present
        )
        return gate.verdict(self.collected, report, self.missing, self.exit_code)


def _fail(i: int) -> str:
    return "::".join(KEYS[i])


BAD_STATES: dict[str, State] = {
    "empty-collection-nothing-ran": State(collected=[], executed=[]),
    "empty-collection-exit-5": State(collected=[], executed=[], exit_code=5),
    "empty-collection-but-tests-ran": State(collected=[]),
    "missing-id": State(executed=KEYS[1:], missing=IDS[:1]),
    "all-missing": State(executed=[], missing=list(IDS)),
    "failure": State(failed=[_fail(3)]),
    "error": State(errored=[_fail(3)]),
    "collection-error-case": State(
        executed=[*KEYS, COLLECTION_ERROR_KEY], errored=["::tests.sanity.test_broken"]
    ),
    "uncollected-execution": State(executed=[*KEYS, UNCOLLECTED]),
    "no-report": State(executed=[], present=False, missing=list(IDS)),
    "no-report-missing-not-passed": State(executed=[], present=False),
    "malformed-report-kept-cases": State(present=False),
    "exit-1": State(exit_code=1),
    "exit-2-interrupted": State(exit_code=2),
    "exit-3-internal-error": State(exit_code=3),
    "exit-4-usage-error": State(exit_code=4),
    "exit-5-no-tests": State(exit_code=5),
    "exit-negative-signal": State(exit_code=-9),
    "exit-137-killed": State(exit_code=137),
    "atexit-exit-0-with-failure": State(failed=[_fail(0)], exit_code=0),
    "atexit-exit-0-with-error": State(errored=[_fail(0)], exit_code=0),
}


def test_g5_clean_state_is_green() -> None:
    """G5 control: every collected id ran and passed, report present, exit 0: ``[]``."""
    assert State().reasons() == []


@pytest.mark.parametrize("state", list(BAD_STATES.values()), ids=list(BAD_STATES))
def test_g5_every_bad_state_gives_a_reason(state: State) -> None:
    """G5: every bad state yields at least one non-empty reason; ``[]`` is the only green."""
    reasons = state.reasons()
    assert reasons, "bad state reported green"
    assert all(isinstance(r, str) and r.strip() for r in reasons), reasons


COMBOS: dict[str, tuple[State, int]] = {
    "failure+error": (State(failed=[_fail(0)], errored=[_fail(1)]), 2),
    "failure+missing+exit-1": (
        State(executed=KEYS[1:], missing=IDS[:1], failed=[_fail(2)], exit_code=1),
        3,
    ),
    "no-report+exit-2": (State(executed=[], present=False, missing=list(IDS), exit_code=2), 2),
    "uncollected+failure+error+missing+exit-1": (
        State(
            executed=[*KEYS[1:], UNCOLLECTED],
            missing=IDS[:1],
            failed=[_fail(2)],
            errored=[_fail(3)],
            exit_code=1,
        ),
        5,
    ),
    "empty-collection+exit-5": (State(collected=[], executed=[], exit_code=5), 2),
}


@pytest.mark.parametrize(("state", "n"), list(COMBOS.values()), ids=list(COMBOS))
def test_g5_combined_bad_states_give_every_reason(state: State, n: int) -> None:
    """G5: ``verdict`` returns EVERY reason: n distinct bad states give at least n reasons."""
    reasons = state.reasons()
    assert len(reasons) >= n, reasons


COLLECT_EXIT_CODES = [1, 2, 3, 4, 5, -9]


def _clean_verdict(collect_exit_code: int) -> list[str]:
    """``verdict`` over a fully clean run, with only the collection exit code varied."""
    report = RunReport(executed=list(KEYS), failed=[], errored=[], present=True)
    return gate.verdict(list(IDS), report, [], 0, collect_exit_code=collect_exit_code)


def test_g5_clean_with_collect_exit_0_is_green() -> None:
    """G5 control: a clean run whose collection exited 0 is green: ``[]``."""
    assert _clean_verdict(0) == []


@pytest.mark.parametrize("code", COLLECT_EXIT_CODES, ids=[str(c) for c in COLLECT_EXIT_CODES])
def test_g5_collect_exit_nonzero_alone_gives_a_reason(code: int) -> None:
    """G5: a non-zero collection exit code is red on its own, even when all else is clean.

    Attack: every id collected, executed and passed, run exit 0; only the collection's
    exit code says something went wrong (1 failed, 2 interrupted, 3 internal, 4 usage,
    5 no tests, -9 killed).
    """
    reasons = _clean_verdict(code)
    assert reasons, f"collection exit {code} reported green"
    assert all(isinstance(r, str) and r.strip() for r in reasons), reasons


# --------------------------------------------------------------------------------------
# G6: split_args
# --------------------------------------------------------------------------------------
SPLITS: dict[str, tuple[list[str], tuple[list[str], list[str]]]] = {
    "no-separator": (["tests/unit", "-k", "x"], (["tests/unit", "-k", "x"], [])),
    "empty": ([], ([], [])),
    "separator-only": (["--"], ([], [])),
    "separator-first": (["--", "--cov"], ([], ["--cov"])),
    "separator-last": (["tests/unit", "--"], (["tests/unit"], [])),
    "separator-middle": (
        ["tests/sanity", "tests/unit", "--", "--cov", "--cov-report=term-missing"],
        (["tests/sanity", "tests/unit"], ["--cov", "--cov-report=term-missing"]),
    ),
    "several-separators": (["a", "--", "b", "--", "c"], (["a"], ["b", "--", "c"])),
    "double-separator": (["a", "--", "--", "b"], (["a"], ["--", "b"])),
}


@pytest.mark.parametrize(("argv", "expected"), list(SPLITS.values()), ids=list(SPLITS))
def test_g6_split_args_at_first_separator(
    argv: list[str], expected: tuple[list[str], list[str]]
) -> None:
    """G6: split at the FIRST ``--`` (POSIX guideline 10); later ones are kept verbatim."""
    shared, run_only = gate.split_args(list(argv))
    assert (list(shared), list(run_only)) == expected


# --------------------------------------------------------------------------------------
# G7: main, with the two pytest processes faked
# --------------------------------------------------------------------------------------
def _junit_path(cmd: Sequence[str]) -> Path | None:
    for i, arg in enumerate(cmd):
        for opt in ("--junitxml", "--junit-xml"):
            if arg.startswith(opt + "="):
                return Path(arg.split("=", 1)[1])
            if arg == opt and i + 1 < len(cmd):
                return Path(cmd[i + 1])
    return None


def _is_collect(cmd: Sequence[str]) -> bool:
    return "--collect-only" in cmd or "--co" in cmd


@dataclass
class FakePytest:
    """Stands in for ``subprocess.run``: canned collect output, canned junit, exit codes."""

    collect_out: str
    junit: str | None
    collect_code: int = 0
    run_code: int = 0
    calls: list[list[str]] = field(default_factory=list)
    stale_report: list[bool] = field(default_factory=list)

    def __call__(self, cmd: Sequence[str], *args: object, **kwargs: object) -> object:
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        if _is_collect(cmd):
            return subprocess.CompletedProcess(cmd, self.collect_code, self.collect_out, "")
        path = _junit_path(cmd)
        if path is not None:
            self.stale_report.append(path.exists() and path.stat().st_size > 0)
            if self.junit is not None:
                path.write_text(self.junit, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, self.run_code, "", "")


def _run_main(monkeypatch: pytest.MonkeyPatch, fake: FakePytest, argv: list[str]) -> int:
    monkeypatch.setattr(subprocess, "run", fake)
    return gate.main(argv)


ARGV = ["tests/sanity", "tests/unit", "tests/gates", "--", "--cov", "--cov-report=term-missing"]


def _clean() -> FakePytest:
    return FakePytest(_collect_out(IDS), _clean_junit())


CLEAN_VARIANTS: dict[str, FakePytest] = {
    "plain": _clean(),
    "collect-warnings": FakePytest(_collect_out(IDS, WARNINGS_BLOCK), _clean_junit()),
    "system-out-in-report": FakePytest(
        _collect_out(IDS), _junit([_case(k, SYSTEM_OUT) for k in KEYS])
    ),
}


@pytest.mark.parametrize("fake", list(CLEAN_VARIANTS.values()), ids=list(CLEAN_VARIANTS))
def test_g7_clean_run_is_green(monkeypatch: pytest.MonkeyPatch, fake: FakePytest) -> None:
    """G7 control: every collected test ran and passed, both exits 0: ``main`` returns 0."""
    assert _run_main(monkeypatch, fake, ARGV) == 0


MAIN_BAD: dict[str, FakePytest] = {
    "empty-collection-exit-0": FakePytest("", _junit([])),
    "empty-collection-exit-5": FakePytest(
        "\nno tests collected in 0.01s\n", _junit([]), collect_code=5, run_code=5
    ),
    "empty-collection-but-run-ran": FakePytest("\nno tests collected in 0.01s\n", _clean_junit()),
    "shrunken-collection": FakePytest(_collect_out(IDS[:1]), _clean_junit()),
    "missing-report": FakePytest(_collect_out(IDS), None),
    "empty-report": FakePytest(_collect_out(IDS), ""),
    "truncated-report": FakePytest(_collect_out(IDS), MALFORMED["truncated-mid-suite"]),
    "not-junit-root": FakePytest(_collect_out(IDS), "<html><body>ok</body></html>"),
    "report-without-cases": FakePytest(_collect_out(IDS), _junit([])),
    "missing-id": FakePytest(_collect_out(IDS), _clean_junit(KEYS[1:])),
    "uncollected-run": FakePytest(_collect_out(IDS), _clean_junit([*KEYS, UNCOLLECTED])),
    "disguised-uncollected-run": FakePytest(
        _collect_out(IDS), _clean_junit([*KEYS, ("", "tests.unit.test_sneaky")])
    ),
    "atexit-failure-exit-0": FakePytest(_collect_out(IDS), _junit_with({3: FAILURE})),
    "atexit-error-exit-0": FakePytest(_collect_out(IDS), _junit_with({3: ERROR})),
    "lying-suite-counts": FakePytest(
        _collect_out(IDS),
        _junit([_case(k, FAILURE if i == 0 else "") for i, k in enumerate(KEYS)], failures=0),
    ),
    "collection-error-case-exit-0": FakePytest(
        _collect_out(IDS),
        _junit([*(_case(k) for k in KEYS), _case(COLLECTION_ERROR_KEY, COLLECTION_ERROR)]),
    ),
    "run-exit-1-clean-report": FakePytest(_collect_out(IDS), _clean_junit(), run_code=1),
    "run-exit-2-clean-report": FakePytest(_collect_out(IDS), _clean_junit(), run_code=2),
    "run-exit-5-clean-report": FakePytest(_collect_out(IDS), _clean_junit(), run_code=5),
    "run-killed-clean-report": FakePytest(_collect_out(IDS), _clean_junit(), run_code=-9),
    "collect-exit-2-run-clean": FakePytest(
        _collect_out(IDS, ERRORS_BLOCK), _clean_junit(), collect_code=2
    ),
}


@pytest.mark.parametrize("fake", list(MAIN_BAD.values()), ids=list(MAIN_BAD))
def test_g7_every_bad_state_fails_main(monkeypatch: pytest.MonkeyPatch, fake: FakePytest) -> None:
    """G7: ``main`` returns non-zero for every bad state, whatever pytest's exit code says."""
    assert _run_main(monkeypatch, fake, ARGV) != 0


@pytest.mark.parametrize("code", COLLECT_EXIT_CODES, ids=[str(c) for c in COLLECT_EXIT_CODES])
def test_g7_collect_exit_nonzero_with_valid_ids_fails_main(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    """G7+G5: ``main`` fails when the collection process exits non-zero, even when it
    printed every valid node id (no error block) and the run itself is fully clean.

    Attack: a collection that lists the right ids but reports failure through its exit code.
    """
    fake = FakePytest(_collect_out(IDS), _clean_junit(), collect_code=code)
    assert _run_main(monkeypatch, fake, ARGV) != 0


def test_g7_pytest_runs_twice_in_safe_path_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """G7: both processes are ``python -P -m pytest``, so the cwd cannot replace pytest."""
    fake = _clean()
    _run_main(monkeypatch, fake, ARGV)
    assert len(fake.calls) == 2, fake.calls
    for cmd in fake.calls:
        assert "-P" in cmd and "-m" in cmd, cmd
        p, m = cmd.index("-P"), cmd.index("-m")
        assert 0 < p < m, cmd  # -P is an interpreter option: it must precede -m
        assert cmd[m + 1] == "pytest", cmd


def test_g7_collect_first_then_run_with_a_junit_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """G7: the first process only collects; the second runs (not collect) with a junit path."""
    fake = _clean()
    _run_main(monkeypatch, fake, ARGV)
    collect, run = fake.calls
    assert _is_collect(collect), collect
    assert not _is_collect(run), run
    assert _junit_path(run) is not None, run


def test_g7_junit_report_is_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """G7: the report path holds no report before the run: a stale report is never read."""
    fake = _clean()
    _run_main(monkeypatch, fake, ARGV)
    assert fake.stale_report == [False]


def test_g7_args_reach_the_right_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """G7: args before ``--`` go to both processes; args after it to the real run only."""
    fake = _clean()
    _run_main(monkeypatch, fake, ["tests/unit", "-k", "x", "--", "--cov", "--cov-report=term"])
    collect, run = fake.calls
    for cmd in (collect, run):
        assert {"tests/unit", "-k", "x"} <= set(cmd), cmd
    assert "--cov" in run and "--cov-report=term" in run, run
    assert "--cov" not in collect and "--cov-report=term" not in collect, collect


def test_g7_main_reads_sys_argv_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """G7: ``main()`` with no argv gates the command line it was started with."""
    fake = FakePytest("", _junit([]))
    monkeypatch.setattr(sys, "argv", ["pytest_gate", "tests/unit"])
    monkeypatch.setattr(subprocess, "run", fake)
    assert gate.main() != 0
    assert all("tests/unit" in cmd for cmd in fake.calls), fake.calls
