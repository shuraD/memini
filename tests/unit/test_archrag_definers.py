"""AR-06: resolve each canonical id of a scope to its defining entity and plan reference links.

Spec: AR-06 (src/memini/core/archrag/definers.py), card lines A21, A85, I71, T109-T111.
Rules: active only, scheme home, longest content, smallest id. No database: a fake session.
"""

from typing import Any

from memini.core import schema
from memini.core.archrag import definers
from memini.core.archrag.definers import Definer, ReferenceLink
from memini.core.graph import checks
from memini.core.graph.types import Statement

PATTERN = r"^(?P<prefix>[A-Za-z]+[-_])(?P<rest>[A-Za-z0-9][A-Za-z0-9_-]*)$"
SCOPE = "scopevalqzx"


def _cand(
    cid: str, artifact: str | None, length: int, status: str | None = "active"
) -> dict[str, Any]:
    return {"id": cid, "artifact_id": artifact, "content_len": length, "status": status}


def _row(
    eid: str,
    cid: Any,
    artifact: str | None,
    length: int,
    status: str | None = "active",
) -> dict[str, Any]:
    return {
        "id": eid,
        "cid": cid,
        "artifact_id": artifact,
        "content_len": length,
        "status": status,
    }


# --- constants ---------------------------------------------------------------------------


def test_rule_constants() -> None:
    assert definers.RULE_SINGLE == "single"
    assert definers.RULE_SCHEME_HOME == "scheme_home"
    assert definers.RULE_SPECIFIC == "specific"
    assert definers.RULE_TIE == "tie"
    assert definers.RULE_NONE == "none"
    assert definers.RULE_AMBIGUOUS == "ambiguous"


# --- scheme_of ---------------------------------------------------------------------------


def test_scheme_of_matches_and_upper_cases() -> None:
    assert definers.scheme_of("uc-05", PATTERN) == "UC-"
    assert definers.scheme_of("  REQ_12 ", PATTERN) == "REQ_"


def test_scheme_of_no_match() -> None:
    assert definers.scheme_of("12345", PATTERN) is None
    assert definers.scheme_of("", PATTERN) is None


# --- choose_defining_entity ---------------------------------------------------------------


def test_choose_empty_is_none() -> None:
    assert definers.choose_defining_entity([], None) == (None, "none")


def test_choose_all_inactive_is_none() -> None:
    cands = [
        _cand("e1", "A", 10, "superseded"),
        _cand("e2", "B", 20, "deleted"),
    ]
    assert definers.choose_defining_entity(cands, None) == (None, "none")
    assert definers.choose_defining_entity(cands, {"A": 3}) == (None, "none")


def test_choose_status_absent_or_none_counts_as_active() -> None:
    absent = {"id": "e1", "artifact_id": "A", "content_len": 5}
    cands = [absent, _cand("e2", "B", 50, "superseded")]
    assert definers.choose_defining_entity(cands, None) == ("e1", "single")
    cands2 = [_cand("e3", "A", 5, None), _cand("e4", "B", 50, "superseded")]
    assert definers.choose_defining_entity(cands2, None) == ("e3", "single")


def test_choose_single_active_among_inactive_in_other_artifacts() -> None:
    cands = [
        _cand("e1", "A", 100, "superseded"),
        _cand("e2", "B", 5, "active"),
        _cand("e3", "C", 200, "superseded"),
    ]
    assert definers.choose_defining_entity(cands, None) == ("e2", "single")


def test_choose_no_scheme_different_artifacts_is_ambiguous() -> None:
    cands = [_cand("e1", "A", 10), _cand("e2", "B", 20)]
    assert definers.choose_defining_entity(cands, None) == (None, "ambiguous")


def test_choose_no_scheme_ambiguity_ignores_inactive() -> None:
    cands = [
        _cand("e1", "A", 10),
        _cand("e2", "A", 20),
        _cand("e3", "B", 30, "superseded"),
    ]
    assert definers.choose_defining_entity(cands, None) == ("e2", "specific")


def test_choose_no_scheme_one_artifact_longest_wins() -> None:
    cands = [_cand("e1", "A", 10), _cand("e2", "A", 30), _cand("e3", "A", 20)]
    assert definers.choose_defining_entity(cands, None) == ("e2", "specific")


def test_choose_no_scheme_one_artifact_tie_smallest_id() -> None:
    cands = [_cand("e3", "A", 30), _cand("e2", "A", 30), _cand("e1", "A", 10)]
    assert definers.choose_defining_entity(cands, None) == ("e2", "tie")


def test_choose_scheme_home_strict_winner_beats_longer_content() -> None:
    cands = [_cand("e1", "A", 10), _cand("e2", "B", 999)]
    assert definers.choose_defining_entity(cands, {"A": 3, "B": 1}) == (
        "e1",
        "scheme_home",
    )


def test_choose_scheme_home_missing_artifact_counts_zero() -> None:
    cands = [_cand("e1", "A", 10), _cand("e2", "B", 999)]
    assert definers.choose_defining_entity(cands, {"A": 1}) == ("e1", "scheme_home")


def test_choose_scheme_home_shared_top_goes_to_content() -> None:
    cands = [
        _cand("e1", "A", 10),
        _cand("e2", "B", 40),
        _cand("e3", "C", 999),
    ]
    home = {"A": 3, "B": 3, "C": 1}
    assert definers.choose_defining_entity(cands, home) == ("e2", "specific")


def test_choose_scheme_home_shared_top_equal_content_is_tie() -> None:
    cands = [
        _cand("e9", "A", 40),
        _cand("e5", "B", 40),
        _cand("e1", "C", 999),
    ]
    home = {"A": 2, "B": 2, "C": 1}
    assert definers.choose_defining_entity(cands, home) == ("e5", "tie")


# --- resolve_definers -------------------------------------------------------------------


def test_resolve_keys_stripped_upper_and_groups_case_insensitively() -> None:
    rows = [
        _row("e1", " uc-01 ", "A", 10),
        _row("e2", "UC-01", "A", 50),
    ]
    got = definers.resolve_definers(rows, PATTERN)
    assert set(got) == {"UC-01"}
    assert got["UC-01"] == Definer(entity="e2", rule="specific", candidates=2)


def test_resolve_skips_empty_or_missing_cid() -> None:
    rows = [
        _row("e1", "", "A", 10),
        _row("e2", None, "A", 10),
        {"id": "e3", "artifact_id": "A", "content_len": 10, "status": "active"},
        _row("e5", "REQ_1", "A", 10),
    ]
    got = definers.resolve_definers(rows, PATTERN)
    assert set(got) == {"REQ_1"}
    assert got["REQ_1"] == Definer(entity="e5", rule="single", candidates=1)


def test_resolve_candidates_count_inactive_rows() -> None:
    rows = [
        _row("e1", "UC-07", "A", 10),
        _row("e2", "uc-07", "B", 99, "superseded"),
        _row("e3", "UC-07", "C", 99, "superseded"),
    ]
    got = definers.resolve_definers(rows, PATTERN)
    assert got["UC-07"] == Definer(entity="e1", rule="single", candidates=3)


def test_resolve_drops_ids_without_definer() -> None:
    rows = [
        _row("e1", "12345", "A", 10),
        _row("e2", "12345", "B", 20),
        _row("e3", "UC-09", "A", 10, "superseded"),
        _row("e4", "777", "A", 5),
    ]
    got = definers.resolve_definers(rows, PATTERN)
    assert got == {"777": Definer(entity="e4", rule="single", candidates=1)}


def test_resolve_scheme_home_from_same_rows() -> None:
    rows = [
        _row("a1", "UC-01", "A", 10),
        _row("a2", "UC-02", "A", 10),
        _row("a3", "UC-03", "A", 10),
        _row("b1", "UC-01", "B", 500),
    ]
    got = definers.resolve_definers(rows, PATTERN)
    assert got["UC-01"] == Definer(entity="a1", rule="scheme_home", candidates=2)
    assert got["UC-02"] == Definer(entity="a2", rule="single", candidates=1)


def test_resolve_scheme_counts_distinct_cids() -> None:
    # A carries UC-01 three times (one distinct id); B carries UC-01 and UC-02 (two).
    rows = [
        _row("a1", "UC-01", "A", 10),
        _row("a2", "UC-01", "A", 10),
        _row("a3", "UC-01", "A", 900),
        _row("b1", "UC-01", "B", 10),
        _row("b2", "UC-02", "B", 10),
    ]
    got = definers.resolve_definers(rows, PATTERN)
    assert got["UC-01"] == Definer(entity="b1", rule="scheme_home", candidates=4)


def test_resolve_scheme_counts_ignore_inactive_rows() -> None:
    # B's extra UC ids are inactive, so A and B both count 1 -> content rule.
    rows = [
        _row("a1", "UC-01", "A", 80),
        _row("b1", "UC-01", "B", 10),
        _row("b2", "UC-02", "B", 10, "superseded"),
        _row("b3", "UC-03", "B", 10, "superseded"),
    ]
    got = definers.resolve_definers(rows, PATTERN)
    assert got["UC-01"] == Definer(entity="a1", rule="specific", candidates=2)


def test_resolve_scheme_counts_are_per_scheme() -> None:
    # B carries many REQ_ ids, but UC-01's scheme is UC-: A has 2 UC ids, B has 1.
    rows = [
        _row("a1", "UC-01", "A", 10),
        _row("a2", "UC-02", "A", 10),
        _row("b1", "UC-01", "B", 900),
        _row("b2", "REQ_1", "B", 10),
        _row("b3", "REQ_2", "B", 10),
        _row("b4", "REQ_3", "B", 10),
    ]
    got = definers.resolve_definers(rows, PATTERN)
    assert got["UC-01"] == Definer(entity="a1", rule="scheme_home", candidates=2)


def test_resolve_unschemed_id_uses_no_scheme_home() -> None:
    # "9001" has no scheme: candidates in two artifacts are ambiguous even though
    # artifact A carries many schemed ids.
    rows = [
        _row("a1", "9001", "A", 10),
        _row("a2", "UC-01", "A", 10),
        _row("a3", "UC-02", "A", 10),
        _row("b1", "9001", "B", 20),
    ]
    got = definers.resolve_definers(rows, PATTERN)
    assert "9001" not in got
    assert set(got) == {"UC-01", "UC-02"}


def test_resolve_accepts_any_iterable() -> None:
    rows = (r for r in [_row("e1", "UC-01", "A", 1)])
    got = definers.resolve_definers(rows, PATTERN)
    assert got == {"UC-01": Definer(entity="e1", rule="single", candidates=1)}


# --- load_definers -----------------------------------------------------------------------


class _FakeSession:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[Statement] = []

    def run(self, stmt: Statement) -> list[dict[str, Any]]:
        self.calls.append(stmt)
        return [dict(r) for r in self.rows]


LOAD_ROWS = [
    _row("a1", "UC-01", "A", 10),
    _row("a2", "UC-02", "A", 10),
    _row("a3", "UC-03", "A", 10),
    _row("b1", "uc-01", "B", 500),
    _row("c1", "REQ_4", "C", 5, "superseded"),
    _row("d1", "REQ_5", "D", 7),
]


def test_load_definers_runs_one_labeled_read_statement() -> None:
    session = _FakeSession(LOAD_ROWS)
    got = definers.load_definers(session, schema.load(), SCOPE, PATTERN)
    assert len(session.calls) == 1
    stmt = session.calls[0]
    assert isinstance(stmt, Statement)
    assert stmt.kind == "read"
    assert stmt.scope == SCOPE
    assert SCOPE in list(stmt.params.values())
    checks.require_labeled(stmt.cypher)
    assert got == definers.resolve_definers(LOAD_ROWS, PATTERN)
    assert got["UC-01"] == Definer(entity="a1", rule="scheme_home", candidates=2)


def test_load_definers_empty_scope() -> None:
    session = _FakeSession([])
    assert definers.load_definers(session, schema.load(), SCOPE, PATTERN) == {}
    assert len(session.calls) == 1


# --- plan_reference_links ----------------------------------------------------------------


DEFS = {
    "UC-01": Definer(entity="d1", rule="scheme_home", candidates=2),
    "REQ_7": Definer(entity="d2", rule="single", candidates=1),
}


def test_plan_links_maps_normalized_ids() -> None:
    got = definers.plan_reference_links({"c1": [" uc-01 "]}, DEFS)
    assert got == [
        ReferenceLink(src="c1", dst="d1", cited_id="UC-01", rule="scheme_home", candidates=2)
    ]


def test_plan_links_drops_unknown_and_self_links() -> None:
    citations = {"c1": ["NOPE-1", "UC-01"], "d1": ["uc-01", "REQ_7"]}
    got = definers.plan_reference_links(citations, DEFS)
    assert got == [
        ReferenceLink(src="c1", dst="d1", cited_id="UC-01", rule="scheme_home", candidates=2),
        ReferenceLink(src="d1", dst="d2", cited_id="REQ_7", rule="single", candidates=1),
    ]


def test_plan_links_dedups_and_sorts() -> None:
    citations = {
        "z9": ["UC-01", "req_7"],
        "c1": ["UC-01", " uc-01", "UC-01 ", "REQ_7"],
    }
    got = definers.plan_reference_links(citations, DEFS)
    assert [(link.src, link.cited_id) for link in got] == [
        ("c1", "REQ_7"),
        ("c1", "UC-01"),
        ("z9", "REQ_7"),
        ("z9", "UC-01"),
    ]
    assert all(link.dst == DEFS[link.cited_id].entity for link in got)


def test_plan_links_empty() -> None:
    assert definers.plan_reference_links({}, DEFS) == []
    assert definers.plan_reference_links({"c1": []}, DEFS) == []
    assert definers.plan_reference_links({"c1": ["UC-01"]}, {}) == []
