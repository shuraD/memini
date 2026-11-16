"""AR-07: statements that project one hierarchy level into GDS, stream Leiden, and drop it.

Spec: AR-07 (src/memini/core/archrag/leiden.py), card lines A27, A28, A29, A31. Nothing is run
here: the module only builds statements. No database.
"""

import re

import pytest

from memini.core import schema
from memini.core.archrag import leiden
from memini.core.graph import checks
from memini.core.graph.types import Statement

SCOPE = "scopevalqzx"
OTHER_SCOPE = "scopeothrwv"
CONFIG = {"resolutions": [4.0, 1.0, 0.25], "seed": 42, "weighted": False}

_WRITE_CLAUSE = re.compile(r"(?<![\w.$])(MERGE|CREATE|SET|DELETE|REMOVE)(?!\w)", re.IGNORECASE)


def _schema() -> schema.Schema:
    return schema.load()


def _settings(**over: object) -> leiden.LeidenSettings:
    return leiden.LeidenSettings.from_config({**CONFIG, **over})


def _statements(level: int, **over: object) -> list[Statement]:
    return leiden.level_statements(_schema(), SCOPE, level, _settings(**over))


def _by_name(statements: list[Statement], name: str) -> Statement:
    found = [st for st in statements if st.name == name]
    assert len(found) == 1
    return found[0]


# --- LeidenSettings.from_config -------------------------------------------------------------


def test_from_config_reads_the_three_keys() -> None:
    s = leiden.LeidenSettings.from_config(CONFIG)
    assert s.resolutions == (4.0, 1.0, 0.25)
    assert isinstance(s.resolutions, tuple)
    assert s.seed == 42
    assert s.weighted is False


def test_from_config_stores_resolutions_as_tuple_of_floats() -> None:
    s = leiden.LeidenSettings.from_config({"resolutions": (4, 1), "seed": 0, "weighted": True})
    assert s.resolutions == (4.0, 1.0)
    assert all(type(r) is float for r in s.resolutions)
    assert s.seed == 0
    assert s.weighted is True


def test_settings_are_frozen() -> None:
    s = _settings()
    with pytest.raises(AttributeError):
        s.seed = 7  # type: ignore[misc]


@pytest.mark.parametrize("missing", ["resolutions", "seed", "weighted"])
def test_from_config_refuses_a_missing_key(missing: str) -> None:
    section = {k: v for k, v in CONFIG.items() if k != missing}
    with pytest.raises(ValueError):
        leiden.LeidenSettings.from_config(section)


@pytest.mark.parametrize(
    "resolutions",
    [
        [],
        (),
        [0.0],
        [1.0, 0.0],
        [-1.0],
        [1.0, "a"],
        ["1.0"],
        [None],
        [True],
        [1.0, False],
        "4.0",
        4.0,
        None,
    ],
)
def test_from_config_refuses_bad_resolutions(resolutions: object) -> None:
    with pytest.raises(ValueError):
        leiden.LeidenSettings.from_config({**CONFIG, "resolutions": resolutions})


@pytest.mark.parametrize("seed", [-1, True, False, 1.5, 42.0, "42", None])
def test_from_config_refuses_bad_seed(seed: object) -> None:
    with pytest.raises(ValueError):
        leiden.LeidenSettings.from_config({**CONFIG, "seed": seed})


@pytest.mark.parametrize("weighted", [0, 1, "true", "false", None, 1.0])
def test_from_config_refuses_non_bool_weighted(weighted: object) -> None:
    with pytest.raises(ValueError):
        leiden.LeidenSettings.from_config({**CONFIG, "weighted": weighted})


# --- resolution -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("level", "expected"), [(0, 4.0), (1, 1.0), (2, 0.25), (3, 0.25), (7, 0.25)]
)
def test_resolution_follows_the_schedule_and_clamps(level: int, expected: float) -> None:
    assert _settings().resolution(level) == pytest.approx(expected)


def test_resolution_refuses_a_negative_level() -> None:
    with pytest.raises(ValueError):
        _settings().resolution(-1)


# --- graph_name -----------------------------------------------------------------------------


def test_graph_name_is_deterministic_and_non_empty() -> None:
    name = leiden.graph_name(SCOPE, 0)
    assert isinstance(name, str)
    assert name
    assert leiden.graph_name(SCOPE, 0) == name


def test_graph_name_differs_by_scope_and_level() -> None:
    names = {
        leiden.graph_name(SCOPE, 0),
        leiden.graph_name(SCOPE, 1),
        leiden.graph_name(OTHER_SCOPE, 0),
        leiden.graph_name(OTHER_SCOPE, 1),
    }
    assert len(names) == 4


@pytest.mark.parametrize(("scope", "level"), [("", 0), (SCOPE, -1)])
def test_graph_name_refuses_empty_scope_or_negative_level(scope: str, level: int) -> None:
    with pytest.raises(ValueError):
        leiden.graph_name(scope, level)


# --- level_statements: shape ----------------------------------------------------------------


@pytest.mark.parametrize("level", [0, 1, 2, 5])
def test_four_statements_in_order_with_kinds_scope_and_graph(level: int) -> None:
    sts = _statements(level)
    assert len(sts) == 4
    assert all(isinstance(st, Statement) for st in sts)
    assert [st.name for st in sts] == [
        "leiden:drop",
        "leiden:project",
        "leiden:stream",
        "leiden:drop",
    ]
    assert [st.kind for st in sts] == ["admin", "admin", "read", "admin"]
    assert all(st.scope == SCOPE for st in sts)
    expected = leiden.graph_name(SCOPE, level)
    assert all(st.params["graph"] == expected for st in sts)


def test_graph_param_differs_between_levels_and_scopes() -> None:
    s = _settings()
    a = leiden.level_statements(_schema(), SCOPE, 0, s)[0].params["graph"]
    b = leiden.level_statements(_schema(), SCOPE, 1, s)[0].params["graph"]
    c = leiden.level_statements(_schema(), OTHER_SCOPE, 0, s)[0].params["graph"]
    assert len({a, b, c}) == 3


@pytest.mark.parametrize("level", [0, 3])
def test_drop_statements_do_not_fail_when_missing(level: int) -> None:
    sts = _statements(level)
    for st in (sts[0], sts[3]):
        assert "gds.graph.drop" in st.cypher
        assert "false" in st.cypher
    assert sts[0].cypher == sts[3].cypher
    assert dict(sts[0].params) == dict(sts[3].params)


# --- level_statements: projection -----------------------------------------------------------


def test_level0_projection_matches_entities_of_scope() -> None:
    project = _by_name(_statements(0), "leiden:project")
    cypher = project.cypher
    assert "gds.graph.project" in cypher
    assert "undirectedRelationshipTypes" in cypher
    assert "Entity" in cypher
    assert "Community" not in cypher
    assert "RELATED_TO" in cypher
    assert "SIMILAR" in cypher
    assert "$scope" in cypher
    assert project.params["scope"] == SCOPE
    checks.require_labeled(cypher)


@pytest.mark.parametrize("level", [1, 2, 5])
def test_upper_level_projection_matches_communities_of_previous_level(level: int) -> None:
    project = _by_name(_statements(level), "leiden:project")
    cypher = project.cypher
    assert "gds.graph.project" in cypher
    assert "undirectedRelationshipTypes" in cypher
    assert "Community" in cypher
    assert "RELATED_TO" in cypher
    assert "SIMILAR" in cypher
    assert "$scope" in cypher
    assert "$level" in cypher
    assert project.params["scope"] == SCOPE
    assert project.params["level"] == level - 1
    checks.require_labeled(cypher)


def test_level2_projection_uses_level_1_communities() -> None:
    project = _by_name(_statements(2), "leiden:project")
    assert project.params["level"] == 1
    assert project.params["scope"] == SCOPE


@pytest.mark.parametrize("level", [0, 2])
def test_weighted_projection_carries_the_similar_score(level: int) -> None:
    project = _by_name(_statements(level, weighted=True), "leiden:project")
    assert "score" in project.cypher
    checks.require_labeled(project.cypher)


# --- level_statements: stream ---------------------------------------------------------------


@pytest.mark.parametrize(("level", "expected"), [(0, 4.0), (1, 1.0), (2, 0.25), (5, 0.25)])
def test_stream_uses_the_level_resolution_and_seed(level: int, expected: float) -> None:
    stream = _by_name(_statements(level, seed=1234), "leiden:stream")
    assert "gds.leiden.stream" in stream.cypher
    assert "gds.leiden.write" not in stream.cypher
    assert stream.params["resolution"] == pytest.approx(expected)
    assert stream.params["seed"] == 1234


@pytest.mark.parametrize("level", [0, 1])
def test_stream_returns_id_and_community(level: int) -> None:
    stream = _by_name(_statements(level), "leiden:stream")
    assert "AS id" in stream.cypher
    assert "AS community" in stream.cypher


@pytest.mark.parametrize("level", [0, 2])
def test_stream_names_weight_property_only_when_weighted(level: int) -> None:
    weighted = _by_name(_statements(level, weighted=True), "leiden:stream")
    plain = _by_name(_statements(level, weighted=False), "leiden:stream")
    assert "relationshipWeightProperty" in weighted.cypher
    assert "relationshipWeightProperty" not in plain.cypher


# --- level_statements: nothing writes -------------------------------------------------------


@pytest.mark.parametrize("level", [0, 1, 2, 5])
@pytest.mark.parametrize("weighted", [True, False])
def test_no_statement_writes(level: int, weighted: bool) -> None:
    for st in _statements(level, weighted=weighted):
        assert _WRITE_CLAUSE.search(st.cypher) is None, st.cypher
        assert "gds.leiden.write" not in st.cypher
        assert st.kind in ("admin", "read")


# --- level_statements: refusals -------------------------------------------------------------


@pytest.mark.parametrize(("scope", "level"), [("", 0), ("", 2), (SCOPE, -1), (SCOPE, -5)])
def test_level_statements_refuses_empty_scope_or_negative_level(scope: str, level: int) -> None:
    with pytest.raises(ValueError):
        leiden.level_statements(_schema(), scope, level, _settings())
