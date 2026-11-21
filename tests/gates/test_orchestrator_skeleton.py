"""F-orchestrator-071: the skeleton document of a version and its machine checks.

Every test is derived from the rule (FLOW-HYBRID.md steps 2, 3 and 5 and the feature text), not
from the code: a skeleton that should fail must never be reported as good.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from tools.orchestrator import cli, skeleton

HEAD = "| File | Function | Signature | Cards | Top caller | Note |"
SEP = "|---|---|---|---|---|---|"
FILE = "src/pkg/flow.py"
TOP = "pkg.flow:run"
VERSIONS = {"A1": "baseline", "A2": "baseline", "A3": "later"}
SQUADS = {"A1": "graph", "A2": "ingest", "A3": "graph"}
RUN = f"| {FILE} | run | `def run(x: int) -> int:` | A1 | {TOP} | the flow |"
SAVE = f"| {FILE} | Store.save | `def save(self, item: str) -> None:` | A2 | {TOP} | |"
CODE = (
    "class Store:\n"
    "    def save(self, item: str) -> None:\n"
    "        raise NotImplementedError\n"
    "\n"
    "\n"
    "def run(x: int) -> int:\n"
    "    raise NotImplementedError\n"
)


def _doc(*rows: str, header: str = HEAD) -> str:
    return "\n".join(["# Skeleton", "", header, SEP, *rows, ""])


def _line(name: str, signature: str, cards: str, top: str = TOP, note: str = "") -> str:
    return f"| {FILE} | {name} | `{signature}` | {cards} | {top} | {note} |"


def _read(tmp_path: Path, text: str) -> tuple[list[skeleton.Row], list[str]]:
    path = tmp_path / "SKELETON.md"
    path.write_text(text, encoding="utf-8")
    return skeleton.read(path)


def _report(tmp_path: Path, text: str, version: str = "baseline") -> skeleton.Report:
    """Read and check a document the way the command does: read problems count too."""
    rows, bad = _read(tmp_path, text)
    report = skeleton.check(rows, version, VERSIONS, SQUADS)
    report.problems[:0] = bad
    return report


def _row(name: str, signature: str, file: str = FILE) -> skeleton.Row:
    return skeleton.Row(1, file, name, signature, ("A1",), (TOP,), "")


def _diff(rows: list[skeleton.Row], code: str | None, file: str = FILE) -> list[str]:
    return skeleton.compare(rows, {file: code}.get)


# ---------------------------------------------------------------- the document


def test_a_complete_skeleton_is_good(tmp_path: Path) -> None:
    """Every card has a function, every function a card or a top caller: nothing to report."""
    report = _report(tmp_path, _doc(RUN, SAVE))
    assert report.ok
    assert (report.missing, report.extra, report.problems) == ([], [], [])
    assert report.functions == 2
    assert report.counts == {"graph": (1, 1), "ingest": (1, 1)}


def test_a_row_is_read_cell_by_cell(tmp_path: Path) -> None:
    """Backticks and the trailing colon of a signature are optional; ids are comma separated."""
    text = _doc(f"| {FILE} | `Store.save` | def save(self) -> None | A1, A2 | {TOP}, a.b:c | why |")
    rows, bad = _read(tmp_path, text)
    assert bad == []
    (row,) = rows
    assert (row.file, row.name, row.note) == (FILE, "Store.save", "why")
    assert row.signature.rstrip(":") == "def save(self) -> None"
    assert row.cards == ("A1", "A2")
    assert row.callers == (TOP, "a.b:c")
    assert row.target == "pkg.flow:Store.save"
    assert not row.top


@pytest.mark.parametrize(
    ("file", "name", "target"),
    [
        ("src/pkg/flow.py", "run", "pkg.flow:run"),
        ("src/pkg/__init__.py", "run", "pkg:run"),
        ("src/pkg/sub/mod.py", "Store.save", "pkg.sub.mod:Store.save"),
    ],
)
def test_the_target_is_the_dotted_module_and_the_function(
    file: str, name: str, target: str
) -> None:
    """The module is the path without src/ and .py, with __init__ dropped."""
    row = skeleton.Row(1, file, name, "def x()", (), (target,), "")
    assert row.target == target
    assert row.top


def test_an_escaped_pipe_stays_inside_the_signature(tmp_path: Path) -> None:
    """A `|` written `\\|` is part of the signature, not a cell border."""
    text = _doc(_line("run", r"def run(x: int \| None = None) -> int \| None:", "A1"), SAVE)
    rows, bad = _read(tmp_path, text)
    assert bad == []
    assert rows[0].signature.rstrip(":") == "def run(x: int | None = None) -> int | None"
    assert _report(tmp_path, text).ok
    code = CODE.replace("def run(x: int) -> int:", "def run(x: int | None = None) -> int | None:")
    assert _diff(rows, code) == []


@pytest.mark.parametrize(
    "row",
    [
        f"| {FILE} | run | `def run(x: int) -> int:` | A1 | {TOP} |",
        f"| {FILE} | run | `def run(x: int) -> int:` | A1 | {TOP} | note | more |",
        f"| {FILE} | run | `def run(x: int | None) -> int:` | A1 | {TOP} | unescaped pipe |",
        f"| {FILE} | run |",
    ],
)
def test_a_row_with_the_wrong_number_of_cells_is_a_problem(tmp_path: Path, row: str) -> None:
    """A table row that does not have the six columns is reported, never guessed."""
    rows, bad = _read(tmp_path, _doc(row, SAVE))
    assert len(bad) == 1
    assert "line 5" in bad[0]
    assert [r.name for r in rows] == ["Store.save"]
    assert not _report(tmp_path, _doc(row, SAVE)).ok


@pytest.mark.parametrize(
    "header",
    [
        "|File|Function|Signature|Cards|Top caller|Note|",
        "|   File   |  Function |Signature  | Cards |   Top caller | Note   |",
        "  | File | Function | Signature | Cards | Top caller | Note |",
    ],
)
def test_the_header_is_found_whatever_its_spacing(tmp_path: Path, header: str) -> None:
    """Spacing inside the header cells does not hide a table."""
    rows, bad = _read(tmp_path, _doc(RUN, SAVE, header=header))
    assert bad == []
    assert [r.name for r in rows] == ["run", "Store.save"]


def test_every_table_of_the_document_is_read(tmp_path: Path) -> None:
    """A second table, after other text and another table, is read like the first."""
    text = "\n".join(
        [
            HEAD,
            SEP,
            RUN,
            "",
            "Some text.",
            "",
            "| Squad | cards |",
            "|---|---|",
            "| graph | 1 |",
            "",
            "## ingest",
            "",
            HEAD,
            "|:---|:---:|---|---|---|---:|",
            SAVE,
        ]
    )
    rows, bad = _read(tmp_path, text)
    assert bad == []
    assert [(r.line, r.name) for r in rows] == [(3, "run"), (15, "Store.save")]
    assert _report(tmp_path, text).ok


def test_a_bad_row_of_a_second_table_is_reported(tmp_path: Path) -> None:
    """A problem in a later table fails the check like one in the first."""
    bad_row = _line("other", "def other() -> None", "A999")
    text = "\n".join([_doc(RUN, SAVE), "text", "", HEAD, SEP, bad_row])
    report = _report(tmp_path, text)
    assert not report.ok
    assert any("A999" in p for p in report.problems)


@pytest.mark.parametrize(
    "header",
    [
        "| file | function | signature | cards | top caller | note |",
        "| File | Function | Signature | Cards | Top Caller | Note |",
        "| FILE | FUNCTION | SIGNATURE | CARDS | TOP CALLER | NOTE |",
    ],
)
def test_rows_under_a_header_of_another_case_are_not_dropped_silently(
    tmp_path: Path, header: str
) -> None:
    """A skeleton table whose header differs only in case is read or reported, never ignored.

    Otherwise a function with an unknown card (or with no card and no reason) sits in the
    approved document and the check still says the skeleton is good.
    """
    hidden = _line("other", "def other() -> None", "A999")
    text = "\n".join([_doc(RUN, SAVE), "", header, SEP, hidden])
    rows, bad = _read(tmp_path, text)
    assert "other" in [r.name for r in rows] or bad
    assert not _report(tmp_path, text).ok


def test_rows_after_a_blank_line_are_not_dropped_silently(tmp_path: Path) -> None:
    """Six-cell skeleton rows cut from their header by a blank line are read or reported."""
    hidden = _line("other", "def other() -> None", "A999")
    text = "\n".join([_doc(RUN, SAVE), hidden])
    rows, bad = _read(tmp_path, text)
    assert "other" in [r.name for r in rows] or bad
    assert not _report(tmp_path, text).ok


# ---------------------------------------------------------------- problems inside the document


@pytest.mark.parametrize(
    "file",
    [
        "pkg/flow.py",
        "tests/pkg/flow.py",
        "tools/pkg/flow.py",
        "src/pkg/flow.txt",
        "src/pkg/flow",
        "src/pkg/flow.pyi",
        "src/pkg/flow.PY",
        "src/pkg/",
        "src/../tools/flow.py",
        "src/pkg/../../tools/flow.py",
        "/src/pkg/flow.py",
        "SRC/pkg/flow.py",
        "src\\pkg\\flow.py",
        "srcx/pkg/flow.py",
        "src.py",
    ],
)
def test_a_file_that_is_not_a_python_file_under_src_is_a_problem(tmp_path: Path, file: str) -> None:
    """File = a .py under src/; anything else is reported."""
    text = _doc(RUN, SAVE, f"| {file} | helper | `def helper() -> None` | A1 | {TOP} | |")
    report = _report(tmp_path, text)
    assert not report.ok
    assert any("line 7" in p for p in report.problems)


def test_a_duplicate_file_and_function_is_a_problem(tmp_path: Path) -> None:
    """The same function of the same file on two rows is reported, whatever the other cells."""
    again = _line("run", "def run(x: str) -> str", "A2", note="again")
    report = _report(tmp_path, _doc(RUN, SAVE, again))
    assert not report.ok
    assert any("line 7" in p and "duplicate" in p for p in report.problems)


@pytest.mark.parametrize("file", ["src/pkg/./flow.py", "src//pkg/flow.py", "./src/pkg/flow.py"])
def test_a_duplicate_hidden_by_another_spelling_of_the_file_is_a_problem(
    tmp_path: Path, file: str
) -> None:
    """Two rows for one function stay a problem when the file is spelled another way.

    Either the odd spelling is refused as a file or the row is seen as the duplicate it is.
    """
    again = f"| {file} | run | `def run(x: str) -> str` | A1 | {TOP} | again |"
    report = _report(tmp_path, _doc(RUN, SAVE, again))
    assert not report.ok, report.problems


@pytest.mark.parametrize(
    "signature",
    [
        "run(x: int) -> int",
        "run",
        "x = 1",
        "class run",
        "lambda x: x",
        "def run(x: int -> int",
        "def run",
        "def (x: int) -> int",
        "def run(x: int) -> int: ...; def other() -> None",
        "@staticmethod def run(x: int) -> int",
        "import os; def run(x: int) -> int",
        "",
    ],
)
def test_a_signature_that_is_not_a_def_is_a_problem(tmp_path: Path, signature: str) -> None:
    """Signature = exactly one Python def line."""
    report = _report(tmp_path, _doc(_line("run", signature, "A1"), SAVE))
    assert not report.ok
    assert any("line 5" in p and "signature" in p for p in report.problems)


@pytest.mark.parametrize(
    "signature",
    [
        "def run(x: int) -> int: import os; y",
        "def run(x: int) -> int: pass; y",
        "def run(x: int) -> int: return 1; y",
        "def run(x: int) -> int: x = 1; done",
    ],
)
def test_a_signature_carrying_statements_is_a_problem(tmp_path: Path, signature: str) -> None:
    """A def line followed by statements is more than one def line: it is reported."""
    report = _report(tmp_path, _doc(_line("run", signature, "A1"), SAVE))
    assert not report.ok, "a signature with a body passed as a def line"


@pytest.mark.parametrize(
    ("name", "signature"),
    [
        ("run", "def other(x: int) -> int"),
        ("run", "def Run(x: int) -> int"),
        ("run", "def _run(x: int) -> int"),
        ("Store.save", "def Store(self) -> None"),
        ("Store.save", "def store_save(self) -> None"),
        ("Store.save", "async def load(self) -> None"),
    ],
)
def test_a_signature_that_defines_another_name_is_a_problem(
    tmp_path: Path, name: str, signature: str
) -> None:
    """The def name is the last part of the Function cell."""
    top = f"pkg.flow:{name}"
    report = _report(tmp_path, _doc(_line(name, signature, "A1, A2", top=top)))
    assert not report.ok
    assert any("line 5" in p and "signature" in p for p in report.problems)


@pytest.mark.parametrize(
    "signature",
    [
        "def run(x: int) -> int",
        "def run(x: int) -> int:",
        "async def run(x: int) -> int:",
        "def run(x, /, y=1, *args, z: int = 2, **kw) -> dict[str, int]",
        "def run()",
    ],
)
def test_one_def_line_is_a_good_signature(tmp_path: Path, signature: str) -> None:
    """A sync or async def line, with or without the colon, is accepted."""
    assert _report(tmp_path, _doc(_line("run", signature, "A1"), SAVE)).ok


@pytest.mark.parametrize("name", [".run", "Store..run", "my flow.run", "pkg.flow:run"])
def test_a_function_cell_that_is_not_a_qualified_name_is_a_problem(
    tmp_path: Path, name: str
) -> None:
    """Function = a qualified name (`name` or `Class.method`): dotted Python identifiers."""
    row = _line(name, "def run(x: int) -> int", "A1", top=f"pkg.flow:{name}")
    report = _report(tmp_path, _doc(row, _line("Store.save", "def save(self)", "A2", top="-")))
    assert any("line 5" in p for p in report.problems), report.problems


@pytest.mark.parametrize(
    "cards",
    [
        "A999",
        "A1, A999",
        "a1",
        "A 1",
        "A01",
        "A1.",
        "A1 A2",
        "A1; A2",
        "A1 / A2",
        "A1 to A2",
        "A1-A2",
        "A1..A2",
        "A1 and A2",
        "all",
        "n/a",
        "none",
        "\N{EM DASH}",
        "Q7",
    ],
)
def test_a_cards_cell_that_is_not_known_ids_is_a_problem(tmp_path: Path, cards: str) -> None:
    """An unknown id, an odd spelling or a range is reported, never silently ignored."""
    helper = _line("helper", "def helper() -> None", cards)
    report = _report(tmp_path, _doc(RUN, SAVE, helper))
    assert not report.ok
    assert any("line 7" in p and "card" in p for p in report.problems)


def test_a_card_range_does_not_cover_the_cards_inside_it(tmp_path: Path) -> None:
    """Ranges are not expanded: `A1 to A2` maps nothing and both cards stay missing."""
    report = _report(tmp_path, _doc(_line("run", "def run() -> None", "A1 to A2")))
    assert report.missing == ["A1", "A2"]
    assert any("A1 to A2" in p for p in report.problems)


def test_a_card_of_another_version_is_a_problem(tmp_path: Path) -> None:
    """A row may only serve cards of the version that is checked."""
    report = _report(tmp_path, _doc(RUN, SAVE, _line("helper", "def helper() -> None", "A3")))
    assert not report.ok
    assert any("line 7" in p and "A3" in p and "later" in p for p in report.problems)


def test_a_card_of_the_version_is_foreign_when_another_version_is_checked(tmp_path: Path) -> None:
    """The same document checked for `later`: its baseline cards are problems, A3 is missing."""
    report = _report(tmp_path, _doc(RUN, SAVE), version="later")
    assert report.missing == ["A3"]
    assert len([p for p in report.problems if "baseline" in p]) == 2


@pytest.mark.parametrize(
    "top",
    [
        "pkg.flow:nothing",
        "pkg.other:run",
        "pkg.flow.run",
        "run",
        "src/pkg/flow.py:run",
        "src.pkg.flow:run",
        "pkg.flow:Run",
        "PKG.flow:run",
        "pkg.flow:save",
        "pkg.flow:Store",
        "flow:run",
        f"{TOP}, pkg.flow:nothing",
        f"{TOP}; pkg.flow:Store.save",
        f"{TOP} pkg.flow:Store.save",
    ],
)
def test_a_top_caller_that_is_not_a_row_is_a_problem(tmp_path: Path, top: str) -> None:
    """Every top caller named is `<dotted.module>:<function>` of a row of the document."""
    helper = _line("helper", "def helper() -> None", "A1", top=top)
    report = _report(tmp_path, _doc(RUN, SAVE, helper))
    assert not report.ok
    assert any("line 7" in p and "top caller" in p for p in report.problems)


def test_a_top_caller_is_matched_by_file_not_by_function_name(tmp_path: Path) -> None:
    """A row `run` in another file does not make `pkg.flow:run` a row of the document."""
    rows = [
        "| src/pkg/other.py | run | `def run() -> None` | A1 | pkg.other:run | |",
        "| src/pkg/flow.py | helper | `def helper() -> None` | A2 | pkg.flow:run | |",
    ]
    report = _report(tmp_path, _doc(*rows))
    assert not report.ok
    assert any("line 6" in p and "pkg.flow:run" in p for p in report.problems)


def test_a_method_and_an_init_module_can_be_top_callers(tmp_path: Path) -> None:
    """`pkg:main` names `main` of src/pkg/__init__.py; `pkg.flow:Store.save` names a method."""
    rows = [
        "| src/pkg/__init__.py | main | `def main() -> None` | A1 | pkg:main | |",
        f"| {FILE} | Store.save | `def save(self) -> None` | - | pkg.flow:Store.save | |",
        f"| {FILE} | helper | `def helper() -> None` | A2 | pkg:main, pkg.flow:Store.save | |",
    ]
    report = _report(tmp_path, _doc(*rows))
    assert report.ok, report.problems
    assert report.extra == []


@pytest.mark.parametrize("top", ["", "-", " ", "- , -", ","])
def test_an_empty_top_caller_is_a_problem(tmp_path: Path, top: str) -> None:
    """Every function has a known top caller."""
    helper = _line("helper", "def helper() -> None", "A1", top=top, note="why")
    report = _report(tmp_path, _doc(RUN, SAVE, helper))
    assert not report.ok
    assert any("line 7" in p and "top caller" in p for p in report.problems)


@pytest.mark.parametrize("note", ["", " ", "-", "` `"])
def test_an_extra_row_without_a_reason_is_a_problem(tmp_path: Path, note: str) -> None:
    """A row with no card that is not a top caller must say in its Note why it exists.

    `-` is the document's own word for "nothing": it is not a reason.
    """
    helper = _line("helper", "def helper() -> None", "-", note=note)
    report = _report(tmp_path, _doc(RUN, SAVE, helper))
    assert len(report.extra) == 1
    assert "helper" in report.extra[0]
    assert not report.ok, f"the note {note!r} passed as a reason"


def test_an_extra_row_with_a_reason_is_listed_and_allowed(tmp_path: Path) -> None:
    """An extra with a reason is shown for the reader (functions with no card) but is no fault."""
    helper = _line("helper", "def helper() -> None", "", note="shared by run and save")
    report = _report(tmp_path, _doc(RUN, SAVE, helper))
    assert report.ok
    assert len(report.extra) == 1
    assert "helper" in report.extra[0]
    assert "shared by run and save" in report.extra[0]


def test_a_top_caller_without_a_card_is_not_extra(tmp_path: Path) -> None:
    """A row that names itself as top caller needs neither a card nor a reason."""
    rows = [
        _line("main", "def main() -> None", "-", top="pkg.flow:main"),
        _line("run", "def run() -> None", "A1, A2", top="pkg.flow:main"),
    ]
    report = _report(tmp_path, _doc(*rows))
    assert report.ok
    assert report.extra == []


def test_naming_another_top_caller_does_not_make_a_row_a_top_caller(tmp_path: Path) -> None:
    """Only a row whose Top caller names itself is a top caller; any other cardless row is extra."""
    helper = _line("helper", "def helper() -> None", "-", top=TOP)
    report = _report(tmp_path, _doc(RUN, SAVE, helper))
    assert len(report.extra) == 1
    assert not report.ok


# ---------------------------------------------------------------- coverage


def test_a_card_with_no_function_is_missing(tmp_path: Path) -> None:
    """Every card of the version needs a function; the others are counted per squad."""
    report = _report(tmp_path, _doc(RUN))
    assert not report.ok
    assert report.missing == ["A2"]
    assert report.counts == {"graph": (1, 1), "ingest": (1, 0)}
    assert report.problems == []


def test_a_card_listed_twice_does_not_cover_another_card(tmp_path: Path) -> None:
    """A card on two rows (and twice in one cell) is one mapped card, not two."""
    rows = [RUN, _line("Store.save", "def save(self) -> None", "A1, A1")]
    report = _report(tmp_path, _doc(*rows))
    assert report.missing == ["A2"]
    assert report.counts == {"graph": (1, 1), "ingest": (1, 0)}
    assert not report.ok


def test_a_card_without_a_squad_is_still_wanted() -> None:
    """A card of the version that no squad card file holds is counted under `-`, not dropped."""
    report = skeleton.check([], "baseline", {"A1": "baseline", "T9": "baseline"}, {"A1": "graph"})
    assert report.missing == ["A1", "T9"]
    assert report.counts == {"graph": (1, 0), "-": (1, 0)}


@pytest.mark.parametrize("text", ["", "\n", "# Skeleton\n\nNothing yet.\n", _doc()])
def test_a_document_with_no_rows_maps_nothing(tmp_path: Path, text: str) -> None:
    """An empty document, or a table with zero rows, leaves every card of the version missing."""
    report = _report(tmp_path, text)
    assert not report.ok
    assert report.functions == 0
    assert report.missing == ["A1", "A2"]
    assert report.counts == {"graph": (1, 0), "ingest": (1, 0)}


@pytest.mark.parametrize("version", ["Baseline", "BASELINE", "baseline ", "base", "", "v9"])
def test_a_version_with_no_card_is_refused(version: str) -> None:
    """A version name that matches no section (case counts) is an error, never an empty pass."""
    with pytest.raises(ValueError, match="version"):
        skeleton.check([], version, VERSIONS, SQUADS)


def test_the_report_is_rendered_with_counts_and_lists(tmp_path: Path) -> None:
    """The coverage table shows cards, mapped and missing per squad, and names what is wrong."""
    helper = _line("helper", "def helper() -> None", "A999", note="")
    text = skeleton.render(_report(tmp_path, _doc(RUN, helper)))
    assert "| graph | 1 | 1 | 0 |" in text
    assert "| ingest | 1 | 0 | 1 |" in text
    assert "| all | 2 | 1 | 1 |" in text
    assert "missing (1):" in text
    assert "A2" in text
    assert "A999" in text


# ---------------------------------------------------------------- the code


def _rows() -> list[skeleton.Row]:
    return [
        _row("run", "def run(x: int) -> int"),
        _row("Store.save", "def save(self, item: str) -> None:"),
    ]


def test_code_that_matches_the_skeleton_has_no_difference() -> None:
    """Stub files with the agreed signatures match."""
    assert _diff(_rows(), CODE) == []


def test_a_missing_file_is_a_difference() -> None:
    """A file of the skeleton that the code does not have is reported."""
    found = _diff(_rows(), None)
    assert len(found) == 1
    assert FILE in found[0]
    assert "missing" in found[0]


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("run", "class Store:\n    def save(self, item: str) -> None: ...\n"),
        ("run", CODE.replace("def run(", "def run2(")),
        ("run", CODE.replace("def run(x: int) -> int:", "class run:") + "    x = 1\n"),
        ("run", CODE.replace("def run(", "def Run(")),
        ("run", CODE.replace("def run(x: int) -> int:\n    raise NotImplementedError", "")
            + "class Other:\n    def run(self, x: int) -> int: ...\n"),
        ("run", CODE.replace("def run(x: int) -> int:\n    raise NotImplementedError", "")
            + "def outer() -> None:\n    def run(x: int) -> int: ...\n"),
        ("run", CODE.replace("def run(x: int) -> int:\n    raise NotImplementedError", "")
            + "run = lambda x: x\n"),
        ("Store.save", "class Store:\n    pass\n\n\ndef run(x: int) -> int: ...\n"),
        ("Store.save", "def save(self, item: str) -> None: ...\n\n\ndef run(x: int) -> int: ...\n"),
        ("Store.save", "class Store:\n    def load(self) -> None: ...\n"
            "def run(x: int) -> int: ...\n"),
        ("Store.save", "class Other:\n    def save(self, item: str) -> None: ...\n"
            "class Store:\n    pass\ndef run(x: int) -> int: ...\n"),
        ("Store.save", "def Store() -> None:\n    def save(self, item: str) -> None: ...\n"
            "def run(x: int) -> int: ...\n"),
        ("Store.save", "class Store:\n    class Inner:\n"
            "        def save(self, item: str) -> None: ...\n"
            "def run(x: int) -> int: ...\n"),
    ],
)  # fmt: skip
def test_a_missing_function_is_a_difference(name: str, code: str) -> None:
    """A function or method of the skeleton must exist under exactly its qualified name."""
    found = _diff(_rows(), code)
    assert any(f"{FILE} {name}:" in d and "missing" in d for d in found), found


SIGNATURES = [
    ("def run(x: int = 1) -> int", "def run(x: int = 2) -> int"),
    ("def run(x: int = 1) -> int", "def run(x: int) -> int"),
    ("def run(x: int) -> int", "def run(x: int = 0) -> int"),
    ("def run(x: int = 1) -> int", "def run(x: int = True) -> int"),
    ("def run(x: str = '1') -> int", "def run(x: str = 1) -> int"),
    ("def run(x: int = None) -> int", "def run(x: int = ...) -> int"),
    ("def run(x: int) -> int", "def run(x: str) -> int"),
    ("def run(x: int) -> int", "def run(x) -> int"),
    ("def run(x: int) -> int", "def run(x: 'int') -> int"),
    ("def run(x: list[int]) -> int", "def run(x: list[str]) -> int"),
    ("def run(x: int) -> int", "def run(x: int) -> str"),
    ("def run(x: int) -> int", "def run(x: int)"),
    ("def run(x: int)", "def run(x: int) -> None"),
    ("def run(x: int) -> int", "async def run(x: int) -> int"),
    ("async def run(x: int) -> int", "def run(x: int) -> int"),
    ("def run(x: int, *, y: int) -> int", "def run(x: int, y: int) -> int"),
    ("def run(x: int, y: int) -> int", "def run(x: int, *, y: int) -> int"),
    ("def run(*, x: int, y: int) -> int", "def run(x: int, *, y: int) -> int"),
    ("def run(x: int, *, y: int = 1) -> int", "def run(x: int, *, y: int = 2) -> int"),
    ("def run(x: int, *, y: int = 1) -> int", "def run(x: int, *, y: int) -> int"),
    ("def run(x: int) -> int", "def run(x: int, *args: int) -> int"),
    ("def run(x: int, *args: int) -> int", "def run(x: int) -> int"),
    ("def run(x: int, *args: int) -> int", "def run(x: int, *rest: int) -> int"),
    ("def run(x: int, *args: int) -> int", "def run(x: int, *args: str) -> int"),
    ("def run(x: int) -> int", "def run(x: int, **kw: int) -> int"),
    ("def run(x: int, **kw: int) -> int", "def run(x: int, *kw: int) -> int"),
    ("def run(x: int, /) -> int", "def run(x: int) -> int"),
    ("def run(x: int, y: int, /) -> int", "def run(x: int, /, y: int) -> int"),
    ("def run(x: int) -> int", "def run(y: int) -> int"),
    ("def run(x: int, y: str) -> int", "def run(y: str, x: int) -> int"),
    ("def run(x: int) -> int", "def run(x: int, y: int = 0) -> int"),
    ("def run(x: int) -> int", "def run() -> int"),
]


@pytest.mark.parametrize(("document", "code"), SIGNATURES)
def test_a_changed_contract_signature_is_a_difference(document: str, code: str) -> None:
    """A unit cannot change a contract signature.

    Arguments, annotations, defaults, the return annotation and sync/async all count.
    """
    found = _diff([_row("run", document)], f"{code}:\n    raise NotImplementedError\n")
    assert len(found) == 1, found
    assert f"{FILE} run:" in found[0]
    assert "signature" in found[0]


@pytest.mark.parametrize(("document", "code"), SIGNATURES)
def test_a_changed_method_signature_is_a_difference(document: str, code: str) -> None:
    """The same holds for a method of a class."""
    body = f"class Store:\n    {code}:\n        raise NotImplementedError\n"
    found = _diff([_row("Store.run", document)], body)
    assert len(found) == 1, found
    assert "Store.run" in found[0]


@pytest.mark.parametrize(
    ("document", "code"),
    [
        ("def run(x: int = 1) -> int", "def run( x:int=1 )->int:\n    return x\n"),
        ("def run( x:int=1 )->int:", "def run(x: int = 1) -> int:\n    return x\n"),
        ("def run(x: int, y: str = 'a') -> int", 'def run(x: int, y: str = "a") -> int: ...\n'),
        (
            "def run(x: int, *, y: dict[str, int] | None = None) -> list[int]",
            "def run(\n    x: int,\n    *,\n    y: dict[str, int] | None = None,\n"
            ") -> list[int]:\n    raise NotImplementedError\n",
        ),
        ("async def run(x: int) -> int", "async  def run(x: int) -> int:\n    return x\n"),
        ("def run(x: int) -> int", '@cache\ndef run(x: int) -> int:\n    """Doc."""\n'),
    ],
)
def test_spacing_does_not_make_a_signature_differ(document: str, code: str) -> None:
    """Text layout (spaces, line breaks, quote style, a trailing comma) is not the contract."""
    assert _diff([_row("run", document)], code) == []


@pytest.mark.parametrize(
    ("decorator", "signature"),
    [
        ("@property", "def name(self) -> str"),
        ("@staticmethod", "def name(x: int) -> str"),
        ("@classmethod", "def name(cls) -> str"),
        ("@functools.cached_property", "def name(self) -> str"),
    ],
)
def test_a_property_or_a_class_or_static_method_is_part_of_the_contract(
    decorator: str, signature: str
) -> None:
    """Turning a method into a property (or a static/class method) changes how it is called.

    One row cannot match both the plain method and the decorated one: a unit that adds or
    drops the decorator has changed the contract and must be seen.
    """
    plain = f"class Store:\n    {signature}:\n        raise NotImplementedError\n"
    decorated = plain.replace("    def ", f"    {decorator}\n    def ")
    rows = [_row("Store.name", signature)]
    assert _diff(rows, plain) == [] or _diff(rows, decorated) == []
    assert not (_diff(rows, plain) == [] and _diff(rows, decorated) == []), (
        f"{decorator} is invisible to the comparison"
    )


@pytest.mark.parametrize(
    "extra",
    [
        "def helper() -> None: ...\n",
        "async def helper() -> None: ...\n",
        "class Store2:\n    def load(self) -> None: ...\n",
        "class Outer:\n    class Inner:\n        def deep(self) -> None: ...\n",
    ],
)
def test_a_public_function_the_document_does_not_name_is_a_difference(extra: str) -> None:
    """The files on main match the approved skeleton: no public function beyond it."""
    found = _diff(_rows(), CODE + "\n\n" + extra)
    assert len(found) == 1, found
    assert "does not name" in found[0]


def test_a_public_method_added_to_a_listed_class_is_a_difference() -> None:
    """A new public method on a class of the skeleton is reported."""
    code = CODE.replace("class Store:\n", "class Store:\n    def load(self) -> None: ...\n\n")
    found = _diff(_rows(), code)
    assert len(found) == 1, found
    assert "Store.load" in found[0]


@pytest.mark.parametrize(
    "extra",
    [
        "def _helper() -> None: ...\n",
        "class _Private:\n    def load(self) -> None: ...\n",
        "class Store2:\n    def _load(self) -> None: ...\n    def __init__(self) -> None: ...\n",
        "def __getattr__(name: str) -> object: ...\n",
    ],
)
def test_a_private_function_needs_no_row(extra: str) -> None:
    """Public = no part of the qualified name starts with `_`; private helpers are free."""
    assert _diff(_rows(), CODE + "\n\n" + extra) == []


def test_a_nested_function_needs_no_row() -> None:
    """A function defined inside a function is not part of the contract."""
    code = CODE.replace(
        "def run(x: int) -> int:\n",
        "def run(x: int) -> int:\n    def inner() -> None: ...\n",
    )
    assert _diff(_rows(), code) == []


def test_a_function_redefined_in_the_file_is_judged_by_its_last_definition() -> None:
    """The definition Python keeps (the last) is the contract: a later, different one is seen."""
    code = CODE + "\n\ndef run(x: int, y: int = 0) -> int:\n    return x\n"
    found = _diff(_rows(), code)
    assert len(found) == 1, found
    assert f"{FILE} run:" in found[0]
    assert "signature" in found[0]


def test_a_method_redefined_in_the_class_is_judged_by_its_last_definition() -> None:
    """The same for a method defined twice in its class."""
    code = CODE.replace(
        "\n\ndef run(",
        "    def save(self, item: str, force: bool = False) -> None: ...\n\n\ndef run(",
    )
    found = _diff(_rows(), code)
    assert len(found) == 1, found
    assert "Store.save" in found[0]


def test_only_listed_files_are_read() -> None:
    """The comparison asks for each file of the document once, by the name the row gives."""
    asked: list[str] = []

    def source(file: str) -> str | None:
        asked.append(file)
        return CODE

    rows = [*_rows(), _row("run", "def run(x: int) -> int", file="src/pkg/other.py")]
    skeleton.compare(rows, source)
    assert sorted(asked) == [FILE, "src/pkg/other.py"]


@pytest.mark.parametrize(
    "code",
    [
        "def run(x: int -> int:\n    return x\n",
        "def run(x: int) -> int:\nreturn x\n",
        "<<<<<<< HEAD\ndef run(x: int) -> int: ...\n=======\n>>>>>>> other\n",
        "class Store:\n    def save(self, item: str) -> None: ...\n  def run(x): ...\n",
    ],
)
def test_a_file_that_does_not_parse_is_a_difference_not_a_crash(code: str) -> None:
    """Code that is not Python cannot match the skeleton: it is reported, the check goes on."""
    rows = [*_rows(), _row("helper", "def helper() -> None", file="src/pkg/other.py")]
    found = skeleton.compare(rows, {FILE: code, "src/pkg/other.py": None}.get)
    assert any(FILE in d for d in found), found
    assert any("src/pkg/other.py" in d and "missing" in d for d in found), found


# ---------------------------------------------------------------- the command, end to end


def _git(repo: Path, *args: str) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    subprocess.run(  # noqa: S603 - fixed binary, our own arguments
        ["git", "-C", str(repo), *args],  # noqa: S607
        check=True,
        capture_output=True,
        env=env,
    )


class Env:
    """A fake cards_dir with VERSIONS.md next to it, and a git repo holding the code."""

    def __init__(self, tmp_path: Path) -> None:
        self.docs = tmp_path / "docs"
        self.repo = tmp_path / "repo"
        self.state = tmp_path / "state"
        caps = self.docs / "capabilities"
        for squad, card in (("graph", "A1"), ("ingest", "A2")):
            (self.docs / "squads" / squad).mkdir(parents=True)
            text = f"## IN: baseline\n| {card} | Text of {card}. | not done | - | - |\n"
            (self.docs / "squads" / squad / "CARD.md").write_text(text)
        caps.mkdir()
        (caps / "VERSIONS.md").write_text(
            "# Versions\n\n## baseline\n"
            "| A1 | Text of A1. | dev4 | open | - | - |\n"
            "| A2 | Text of A2. | dev2 | done | IN-01 | merged |\n"
            "\n## later\n"
            "| A3 | Text of A3. | dev4 | open | - | - |\n"
        )
        self.doc = caps / "SKELETON-baseline.md"
        self.doc.write_text(_doc(RUN, SAVE))
        (self.repo / "src" / "pkg").mkdir(parents=True)
        _git(self.repo, "init", "-q", "-b", "main")
        _git(self.repo, "config", "user.name", "memini-author")
        _git(self.repo, "config", "user.email", "memini-author@example.invalid")
        _git(self.repo, "config", "commit.gpgsign", "false")
        self.commit(CODE)

    def commit(self, code: str, file: str = FILE) -> None:
        path = self.repo / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "--no-verify", "--allow-empty", "-m", "code")

    def run(self, *args: str) -> int:
        over = {
            "repo": str(self.repo),
            "state_dir": str(self.state),
            "cards_dir": str(self.docs / "squads"),
        }
        return cli.main(["skeleton-check", *args, "--config-json", json.dumps(over)])


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


def test_the_command_passes_a_complete_skeleton(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    """Default document `<cards_dir>/../capabilities/SKELETON-<version>.md`; exit 0 when good."""
    assert env.run("baseline") == 0
    out = capsys.readouterr().out
    assert "| graph | 1 | 1 | 0 |" in out
    assert "| ingest | 1 | 1 | 0 |" in out
    assert "| all | 2 | 2 | 0 |" in out
    assert "functions: 2" in out


def test_the_command_passes_when_the_code_at_head_matches(env: Env) -> None:
    """With --repo: exit 0 when the committed files match the skeleton."""
    assert env.run("baseline", "--repo") == 0


def test_the_command_counts_a_done_card(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    """A card already done (A2) is a card of the version: without a function it is missing."""
    env.doc.write_text(_doc(RUN))
    assert env.run("baseline") == 1
    out = capsys.readouterr().out
    assert "missing (1):" in out
    assert "A2" in out
    assert "| ingest | 1 | 0 | 1 |" in out


@pytest.mark.parametrize(
    "rows",
    [
        (RUN, SAVE, _line("helper", "def helper() -> None", "A999")),
        (RUN, SAVE, _line("helper", "def helper() -> None", "A3")),
        (RUN, SAVE, _line("helper", "def helper() -> None", "-")),
        (RUN, SAVE, _line("helper", "def helper() -> None", "A1", top="pkg.flow:nothing")),
        (RUN, SAVE, _line("helper", "def helper() -> None", "A1", top="-")),
        (RUN, SAVE, _line("helper", "helper() -> None", "A1")),
        (RUN, SAVE, _line("helper", "def other() -> None", "A1")),
        (RUN, SAVE, RUN),
        (RUN, SAVE, "| tools/x.py | helper | `def helper() -> None` | A1 | pkg.flow:run | |"),
        (RUN, SAVE, f"| {FILE} | helper | `def helper() -> None` | A1 |"),
        (RUN, SAVE, _line("helper", "def helper() -> None", "A1 to A2")),
    ],
)
def test_the_command_fails_an_inconsistent_skeleton(
    env: Env, capsys: pytest.CaptureFixture[str], rows: tuple[str, ...]
) -> None:
    """Exit 1 for any problem, although every card is mapped; the problem is printed."""
    env.doc.write_text(_doc(*rows))
    assert env.run("baseline") == 1
    assert "line 7" in capsys.readouterr().out


@pytest.mark.parametrize("text", ["", "# Skeleton\n", _doc()])
def test_the_command_fails_a_document_with_no_rows(env: Env, text: str) -> None:
    """An empty document or an empty table for a version with cards is not a pass."""
    env.doc.write_text(text)
    assert env.run("baseline") == 1
    assert env.run("baseline", "--repo") == 1


@pytest.mark.parametrize("version", ["Baseline", "BASELINE", "nothing"])
def test_the_command_refuses_an_unknown_version(
    env: Env, capsys: pytest.CaptureFixture[str], version: str
) -> None:
    """Exit 2 for a version VERSIONS.md does not have, also when only the case differs."""
    assert env.run(version, "--doc", str(env.doc)) == 2
    assert "version" in capsys.readouterr().out


def test_the_command_refuses_a_document_that_is_not_there(env: Env, tmp_path: Path) -> None:
    """Exit 2 when the document is not found: the default one or the one given."""
    assert env.run("baseline", "--doc", str(tmp_path / "nothing.md")) == 2
    assert env.run("later") == 2
    env.doc.unlink()
    assert env.run("baseline") == 2


def test_the_command_reads_the_document_given(env: Env, tmp_path: Path) -> None:
    """--doc replaces the default document."""
    other = tmp_path / "other.md"
    other.write_text(_doc(RUN))
    assert env.run("baseline", "--doc", str(other)) == 1
    env.doc.write_text(_doc(RUN))
    other.write_text(_doc(RUN, SAVE))
    assert env.run("baseline", "--doc", str(other)) == 0


@pytest.mark.parametrize(
    "code",
    [
        CODE.replace("def run(x: int) -> int:", "def run(x: int = 0) -> int:"),
        CODE.replace("def run(x: int) -> int:", "async def run(x: int) -> int:"),
        CODE.replace("item: str", "item: bytes"),
        CODE.replace("def run(", "def run_all("),
        CODE.replace("    def save(self, item: str) -> None:\n", "    def load(self) -> None:\n"),
        CODE + "\n\ndef helper() -> None:\n    raise NotImplementedError\n",
    ],
)
def test_the_command_fails_when_the_code_at_head_differs(
    env: Env, capsys: pytest.CaptureFixture[str], code: str
) -> None:
    """With --repo exit 1 when HEAD differs; without it the document alone still passes."""
    env.commit(code)
    assert env.run("baseline") == 0
    capsys.readouterr()
    assert env.run("baseline", "--repo") == 1
    assert FILE in capsys.readouterr().out


def test_the_command_fails_when_a_file_is_not_at_head(env: Env) -> None:
    """A file of the skeleton that is not committed is missing."""
    rows = (RUN, SAVE, "| src/pkg/new.py | helper | `def helper() -> None` | A1 | pkg.flow:run | |")
    env.doc.write_text(_doc(*rows))
    (env.repo / "src" / "pkg" / "new.py").write_text("def helper() -> None: ...\n")
    assert env.run("baseline") == 0
    assert env.run("baseline", "--repo") == 1


def test_the_command_compares_head_not_the_working_tree(env: Env) -> None:
    """What counts is the commit: an uncommitted edit neither hides nor makes a difference."""
    changed = CODE.replace("def run(x: int) -> int:", "def run(x: int, y: int) -> int:")
    (env.repo / FILE).write_text(changed)
    assert env.run("baseline", "--repo") == 0
    env.commit(changed)
    (env.repo / FILE).write_text(CODE)
    assert env.run("baseline", "--repo") == 1


def test_the_command_reports_a_file_at_head_that_does_not_parse(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    """Broken Python at HEAD is a difference (exit 1) with the file named, not a traceback."""
    env.commit("def run(x: int -> int:\n    return x\n")
    assert env.run("baseline", "--repo") == 1
    assert FILE in capsys.readouterr().out
