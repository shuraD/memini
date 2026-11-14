"""Ledger (owner 2026-11-12): a SQL ledger committed as SQL text, one change row per
commit; guards stop lost capabilities, returning bugs and unrecorded changes."""

from pathlib import Path

import pytest
from tools import ledger
from tools.ci import check_ledgers as cl

SCHEMA = (Path(__file__).resolve().parents[2] / "ledgers" / "schema.sql").read_text()


def _write(d: Path, features: str = "", bugs: str = "", changes: str = "") -> Path:
    (d / "schema.sql").write_text(SCHEMA)
    (d / "features.sql").write_text(features)
    (d / "bugs.sql").write_text(bugs)
    (d / "changes.sql").write_text(changes)
    return d


F1 = ledger.insert(
    "feature",
    id="F-t-001",
    description="ledger guards",
    layer="infra",
    proof="owner 2026-11-12",
    status="active",
    added_in="C-0001",
)
T1 = ledger.insert(
    "feature_test", feature_id="F-t-001", test_id="tests/unit/test_ledger.py::test_a"
)
B1 = ledger.insert(
    "bug",
    id="B-t-001",
    description="x",
    root_cause="y",
    status="fixed",
    regression_test="tests/unit/test_ledger.py::test_b",
    fixed_in="C-0001",
)


def test_load_builds_an_in_memory_database_from_the_sql_text(tmp_path):
    db = ledger.load(_write(tmp_path, features=F1 + "\n" + T1 + "\n", bugs=B1 + "\n"))
    assert db.execute("SELECT id, status FROM feature").fetchall() == [("F-t-001", "active")]
    assert db.execute("SELECT regression_test FROM bug").fetchone()[0].endswith("test_b")


def test_insert_is_one_deterministic_line_with_quotes_escaped():
    line = ledger.insert("bug", id="B-9", description="it's broken")
    assert "\n" not in line and "it''s broken" in line
    assert line == ledger.insert("bug", id="B-9", description="it's broken")


def test_add_change_appends_exactly_one_row(tmp_path):
    d = _write(tmp_path)
    cid = ledger.add_change(
        d, author="Claude", module="foundation", summary="init", date="2026-11-12"
    )
    db = ledger.load(d)
    assert db.execute("SELECT id, author, module FROM change").fetchall() == [
        (cid, "Claude", "foundation")
    ]
    cid2 = ledger.add_change(
        d, author="OpenCode", module="foundation", summary="next", date="2026-11-12"
    )
    assert cid2 != cid and len(ledger.load(d).execute("SELECT * FROM change").fetchall()) == 2


def test_feature_without_an_existing_test_is_reported(tmp_path):
    db = ledger.load(_write(tmp_path, features=F1 + "\n" + T1 + "\n"))
    assert cl.features_without_tests(db, existing_tests=set()) == ["F-t-001"]
    assert cl.features_without_tests(db, existing_tests={"tests/unit/test_ledger.py::test_a"}) == []


def test_fixed_bug_without_its_regression_test_is_reported(tmp_path):
    db = ledger.load(_write(tmp_path, bugs=B1 + "\n"))
    assert cl.fixed_bugs_without_regression(db, existing_tests=set()) == ["B-t-001"]
    assert (
        cl.fixed_bugs_without_regression(db, existing_tests={"tests/unit/test_ledger.py::test_b"})
        == []
    )


def test_removed_or_superseded_feature_needs_owner_approval(tmp_path):
    head_dir, new_dir = tmp_path / "head", tmp_path / "new"
    head_dir.mkdir()
    new_dir.mkdir()
    head = ledger.load(_write(head_dir, features=F1 + "\n"))
    gone = ledger.load(_write(new_dir))
    assert cl.lost_features(head, gone, approval=None) == ["F-t-001"]
    assert cl.lost_features(head, gone, approval="OWNER-APPROVED-REMOVAL: owner 2026-11-12 x") == []
    superseded = ledger.load(
        _write(new_dir, features=F1.replace("'active'", "'superseded'") + "\n")
    )
    assert cl.lost_features(head, superseded, approval=None) == ["F-t-001"]


def test_each_commit_adds_exactly_one_change_row(tmp_path):
    d = _write(tmp_path)
    before = ledger.load(d)
    ledger.add_change(d, author="Claude", module="m", summary="s", date="2026-11-12")
    assert cl.new_change_rows(before, ledger.load(d)) == 1


@pytest.mark.parametrize(
    "trailer,expected",
    [
        ("OWNER-APPROVED-REMOVAL: owner 2026-11-12 ref", "owner 2026-11-12 ref"),
        ("no trailer here", None),
    ],
)
def test_approval_trailer_is_read_from_the_commit_message(trailer, expected):
    assert cl.approval_from_message(f"subject\n\nbody\n\n{trailer}\n") == expected


def test_regression_test_outside_the_unit_job_does_not_count(tmp_path):
    # review 6a19f19: AGENTS.md promises a bug's regression test "runs every time";
    # only tests/unit runs on every commit, so a test elsewhere must not satisfy the guard
    acceptance = B1.replace(
        "tests/unit/test_ledger.py::test_b", "tests/acceptance/test_q.py::test_b"
    )
    db = ledger.load(_write(tmp_path, bugs=acceptance + "\n"))
    assert cl.fixed_bugs_without_regression(
        db, existing_tests={"tests/acceptance/test_q.py::test_b"}
    ) == ["B-t-001"]


def test_deleted_change_rows_are_reported(tmp_path):
    head_dir, new_dir = tmp_path / "h", tmp_path / "n"
    head_dir.mkdir()
    new_dir.mkdir()
    row = ledger.insert(
        "change", id="C-1", date="2026-11-12", author="Claude", module="m", summary="s"
    )
    head = ledger.load(_write(head_dir, changes=row + "\n"))
    new = ledger.load(_write(new_dir))
    assert cl.deleted_change_rows(head, new) == ["C-1"]


def test_author_trailer_must_match_the_new_change_row(tmp_path):
    d = _write(tmp_path)
    before = ledger.load(d)
    ledger.add_change(d, author="OpenCode", module="m", summary="s", date="2026-11-12")
    after = ledger.load(d)
    assert cl.author_mismatch(before, after, "subject\n\nChange-Author: Claude\n") == (
        "OpenCode",
        "Claude",
    )
    assert cl.author_mismatch(before, after, "subject\n\nChange-Author: OpenCode\n") is None


def test_loading_the_ledger_from_an_unreadable_git_revision_fails_loudly(tmp_path, monkeypatch):
    # B-007: in CI the checkout had no history, `git show HEAD~1:...` returned nothing and the
    # guard silently loaded an EMPTY ledger (then crashed on "no such table" instead of saying why)
    monkeypatch.setattr(cl, "_git_show", lambda rev_path: None)
    with pytest.raises(cl.LedgerUnavailableError) as e:
        cl._ledger_at("HEAD~1:")
    assert "HEAD~1:" in str(e.value) and "fetch-depth" in str(e.value)


# ---- review b0bacbd (graph-schema): deferred work must be recordable and ids branch-safe


def test_feature_ids_are_module_scoped_and_the_old_global_form_is_rejected(tmp_path):
    # F-010 was allocated by three branches at once; module-scoped ids cannot collide
    good = ledger.insert(
        "feature",
        id="F-graph-schema-001",
        description="d",
        layer="core",
        proof="p",
        status="active",
        added_in="C-1",
    )
    db = ledger.load(_write(tmp_path, features=good + "\n"))
    assert db.execute("SELECT id FROM feature").fetchone()[0] == "F-graph-schema-001"
    bad = ledger.insert(
        "feature",
        id="F-010",
        description="d",
        layer="core",
        proof="p",
        status="active",
        added_in="C-1",
    )
    with pytest.raises(Exception, match="CHECK"):
        ledger.load(
            _write(
                tmp_path / "bad" if (tmp_path / "bad").mkdir() is None else tmp_path,
                features=bad + "\n",
            )
        )


def test_a_deferred_requirement_is_a_planned_feature_owned_by_a_module(tmp_path):
    row = ledger.insert(
        "feature",
        id="F-graph-schema-002",
        description="readers honour reverted supersede",
        layer="core",
        proof="SPEC-graph-schema decision 10",
        status="planned",
        owner_module="ingest-units",
        added_in="C-1",
    )
    db = ledger.load(_write(tmp_path, features=row + "\n"))
    assert db.execute("SELECT status, owner_module FROM feature").fetchone() == (
        "planned",
        "ingest-units",
    )
    # a planned feature needs no test yet, and is not reported as test-less
    assert cl.features_without_tests(db, existing_tests=set()) == []


def test_a_planned_feature_without_an_owner_module_is_rejected(tmp_path):
    row = ledger.insert(
        "feature",
        id="F-graph-schema-003",
        description="d",
        layer="core",
        proof="p",
        status="planned",
        added_in="C-1",
    )
    with pytest.raises(Exception, match="CHECK"):
        ledger.load(_write(tmp_path, features=row + "\n"))


def test_next_id_allocates_per_module(tmp_path):
    rows = (
        "\n".join(
            [
                ledger.insert(
                    "feature",
                    id="F-graph-schema-001",
                    description="d",
                    layer="core",
                    proof="p",
                    status="active",
                    added_in="C-1",
                ),
                ledger.insert(
                    "feature",
                    id="F-graph-schema-002",
                    description="d",
                    layer="core",
                    proof="p",
                    status="active",
                    added_in="C-1",
                ),
                ledger.insert(
                    "feature",
                    id="F-benchmark-q01-007",
                    description="d",
                    layer="infra",
                    proof="p",
                    status="active",
                    added_in="C-1",
                ),
            ]
        )
        + "\n"
    )
    db = ledger.load(_write(tmp_path, features=rows))
    assert ledger.next_id(db, "feature", "graph-schema") == "F-graph-schema-003"
    assert ledger.next_id(db, "feature", "benchmark-q01") == "F-benchmark-q01-008"
    assert ledger.next_id(db, "bug", "graph-schema") == "B-graph-schema-001"


# --- rows are append-only history (review 59a67c1, NOTE 1) ---------------------------

_ROW = (
    "INSERT INTO change (id, date, author, module, summary, features, bugs, checks, review,"
    " approval) VALUES ('C-1', '2026-11-12', 'A', 'm', 'original', NULL, NULL, NULL, NULL,"
    " NULL);\n"
)


def _ledger_with(tmp_path: Path, changes: str) -> Path:
    d = tmp_path / "ledgers"
    d.mkdir()
    (d / "schema.sql").write_text((ledger.LEDGER_DIR / "schema.sql").read_text())
    (d / "changes.sql").write_text(changes)
    return d


@pytest.mark.parametrize(
    "rewrite",
    [
        "UPDATE change SET summary = 'REWRITTEN' WHERE id = 'C-1';",
        "DELETE FROM change WHERE id = 'C-1';",
        _ROW.replace("INSERT INTO", "INSERT OR REPLACE INTO").replace("original", "REWRITTEN"),
        _ROW.replace("NULL);", "NULL) ON CONFLICT(id) DO UPDATE SET summary = 'REWRITTEN';"),
        _ROW.rstrip("\n") + " UPDATE change SET summary = 'REWRITTEN';",  # noqa: S608 — test data
        _ROW.replace("original", "REWRITTEN"),
        # review c4bda9c BLOCKING 1: a valid new row followed by a REPLACE of an old one
        _ROW.replace("'C-1'", "'C-2'").rstrip("\n")
        + " REPLACE INTO change (id, date, author, module, summary) VALUES"
        " ('C-1', '2026-11-12', 'A', 'm', 'REWRITTEN');",
        _ROW.replace("'C-1'", "'C-2'").rstrip("\n")
        + " INSERT OR REPLACE INTO change (id, date, author, module, summary) VALUES"
        " ('C-1', '2026-11-12', 'A', 'm', 'REWRITTEN');",
    ],
)
def test_a_ledger_line_that_rewrites_an_earlier_row_is_rejected(
    tmp_path: Path, rewrite: str
) -> None:
    with pytest.raises(ledger.LedgerFormatError):
        ledger.load(_ledger_with(tmp_path, _ROW + rewrite + "\n"))


def test_plain_inserts_and_comments_load(tmp_path: Path) -> None:
    db = ledger.load(_ledger_with(tmp_path, "-- history\n" + _ROW + "\n"))
    assert db.execute("SELECT summary FROM change").fetchall() == [("original",)]


def test_the_committed_ledger_is_plain_inserts() -> None:
    ledger.load()
