"""CI test step that cannot report a fake green (B-foundation-016).

The process exit code of pytest is not proof that tests ran: a test module can force exit
code 0 (``os._exit(0)``, ``atexit.register(os._exit, 0)``), a conftest can shrink or empty
the collection, and a root ``pytest.py`` can stand in for pytest. This step therefore:

1. runs pytest in safe-path mode (``python -P``), so nothing in the working directory can
   replace pytest; the repo's own imports come from pytest's ``pythonpath`` setting;
2. collects the test ids (``--collect-only``), then runs them with a junit report;
3. decides from the report, not the exit code. It fails when the collection is empty, the
   report is missing, any test failed or errored, a collected id did not run, a run id was
   never collected, or pytest itself exited non-zero.

All decisions are pure functions; ``main`` only runs the two pytest processes.

Usage::

    python -P tools/ci/pytest_gate.py <shared args/paths> [-- <run-only args, e.g. --cov>]
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET  # nosec B405 — parses only our own pytest report
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# pytest's junit writer turns a node id into (classname, name): the path part loses ``.py``
# and its "/" become "."; the "::" parts before the last join the classname with ".";
# the parametrize suffix "[...]" stays verbatim on the name (``_pytest.junitxml``).
_PY_SUFFIX = re.compile(r"\.py$")
# a node id line: an unindented test file path, then "::" and the test name
_NODE_ID = re.compile(r"^[^\s:]\S*\.py::\S")


@dataclass(frozen=True)
class RunReport:
    """What a junit report shows: executed test cases and those that failed or errored."""

    executed: list[tuple[str, str]] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    errored: list[str] = field(default_factory=list)
    present: bool = True


def junit_key(node_id: str) -> tuple[str, str]:
    """Return the ``(classname, name)`` pair pytest's junit report writes for ``node_id``."""
    address, bracket, params = node_id.partition("[")
    names = address.split("::")
    names[0] = _PY_SUFFIX.sub("", names[0].replace("/", "."))
    names[-1] += bracket + params
    return ".".join(names[:-1]), names[-1]


def parse_junit(path: Path) -> RunReport:
    """Read a junit report; a missing or unreadable report means nothing is proven to have run."""
    try:
        root = ET.parse(path).getroot()  # noqa: S314 # nosec B314 — our own pytest's report
    except (OSError, ET.ParseError):
        return RunReport(present=False)
    executed: list[tuple[str, str]] = []
    failed: list[str] = []
    errored: list[str] = []
    for case in root.iter("testcase"):
        key = (case.get("classname", ""), case.get("name", ""))
        label = "::".join(part for part in key if part)
        executed.append(key)
        if case.find("failure") is not None:
            failed.append(label)
        if case.find("error") is not None:
            errored.append(label)
    return RunReport(executed=executed, failed=failed, errored=errored)


def executed_vs_collected(junit_xml_path: Path, collected_ids: list[str]) -> list[str]:
    """Return the collected test ids that the junit report does not show as executed.

    A missing report means none ran, so every collected id is returned.
    """
    report = parse_junit(Path(junit_xml_path))
    ran = Counter(report.executed)
    missing: list[str] = []
    for node_id in collected_ids:
        key = junit_key(node_id)
        if ran[key] > 0:
            ran[key] -= 1
        else:
            missing.append(node_id)
    return missing


def uncollected_executed(report: RunReport, collected_ids: list[str]) -> list[str]:
    """Return the executed test cases (``classname::name``) that the collection never listed.

    Collection errors are reported by pytest as test cases too; they are counted as errors
    (see ``verdict``), not as uncollected runs.
    """
    expected = Counter(junit_key(node_id) for node_id in collected_ids)
    errored = set(report.errored)
    extra: list[str] = []
    for key in report.executed:
        label = "::".join(part for part in key if part)
        if expected[key] > 0:
            expected[key] -= 1
        elif label not in errored:
            extra.append(label)
    return extra


def collected_ids(collect_output: str) -> list[str]:
    """Return the node ids listed by ``pytest --collect-only`` at verbosity -1.

    pytest prints the ids as the leading block of its output, then a blank line, then any
    warning/error sections and the summary; a path quoted in a message or a traceback is
    therefore never taken as an id (QA Q12).
    """
    ids: list[str] = []
    for line in collect_output.splitlines():
        if not line.strip():
            break
        if _NODE_ID.match(line):
            ids.append(line.rstrip())
    return ids


def verdict(
    collected: list[str],
    report: RunReport,
    missing: list[str],
    exit_code: int,
    collect_exit_code: int = 0,
) -> list[str]:
    """Return every reason the test step fails; an empty list is the only green."""
    problems: list[str] = []
    if collect_exit_code != 0:
        problems.append(f"the collection exited with code {collect_exit_code}")
    if not collected:
        problems.append("the collection listed no test: nothing would be gated")
    if not report.present:
        problems.append("no junit report was written: the run did not finish")
    problems += [f"failed: {label}" for label in report.failed]
    problems += [f"error: {label}" for label in report.errored]
    if report.present:
        problems += [f"collected but not executed: {node_id}" for node_id in missing]
        problems += [
            f"executed but never collected: {label}"
            for label in uncollected_executed(report, collected)
        ]
    if exit_code != 0:
        problems.append(f"pytest exited with code {exit_code}")
    return problems


def split_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split ``argv`` at ``--`` into (args for both runs, args for the real run only)."""
    if "--" in argv:
        cut = argv.index("--")
        return argv[:cut], argv[cut + 1 :]
    return argv, []


def _pytest(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — fixed interpreter, our own args, no shell
        [sys.executable, "-P", "-m", "pytest", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def main(argv: list[str] | None = None) -> int:
    """Collect, run, and fail unless the junit report proves every collected test passed."""
    shared, run_only = split_args(sys.argv[1:] if argv is None else argv)
    collect = _pytest(*shared, "--collect-only", "--verbosity=-1")
    collected = collected_ids(collect.stdout)
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "junit.xml"
        run = _pytest(*shared, *run_only, f"--junitxml={report_path}")
        print(run.stdout, end="")
        print(run.stderr, end="", file=sys.stderr)
        report = parse_junit(report_path)
        missing = executed_vs_collected(report_path, collected)
    problems = verdict(collected, report, missing, run.returncode, collect.returncode)
    for problem in problems:
        print(f"✗ test step: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"✓ test step: {len(collected)} collected, all executed and passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
