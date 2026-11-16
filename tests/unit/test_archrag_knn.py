"""AR-04: directed cosine KNN pairs and the statements that regenerate the scope's SIMILAR edges.

Spec: AR-04 (src/memini/core/archrag/knn.py), card lines A24, A25 and I75. A semantic pass on
``embedding`` writes ``score``; a separate code pass on ``embedding_code`` writes ``code_score``.
Each pass first clears its own property from the SIMILAR edges among the call's ids. No database.
"""

import math

import pytest

from memini.core import schema
from memini.core.archrag import knn
from memini.core.graph import checks
from memini.core.graph.types import Statement

SCOPE = "scopevalqzx"
CONFIG = {"top_k": 2, "cutoff": 0.1, "code_top_k": 1, "code_cutoff": 0.5}

VECTORS = {
    "e1": [1.0, 0.0],
    "e2": [1.0, 0.1],
    "e3": [0.0, 1.0],
    "e4": [0.7, 0.7],
    "e5": [-1.0, 0.0],
}


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.hypot(*a) * math.hypot(*b))


def _expected(vectors: dict, top_k: int, cutoff: float) -> list[tuple[str, str, float]]:
    out = []
    for a in sorted(vectors):
        if math.hypot(*vectors[a]) == 0:
            continue
        cands = [
            (b, _cos(vectors[a], vectors[b]))
            for b in vectors
            if b != a and math.hypot(*vectors[b]) != 0
        ]
        cands = [c for c in cands if c[1] >= cutoff]
        cands.sort(key=lambda c: (-c[1], c[0]))
        out.extend((a, b, s) for b, s in cands[:top_k])
    return out


def _same_pairs(got: list, expected: list) -> None:
    assert [(a, b) for a, b, _ in got] == [(a, b) for a, b, _ in expected]
    for (_, _, s), (_, _, e) in zip(got, expected, strict=True):
        assert s == pytest.approx(e)


def _schema() -> schema.Schema:
    return schema.load()


def _settings() -> knn.KnnSettings:
    return knn.KnnSettings.from_config(CONFIG)


def _rows(statements: list[Statement]) -> list[dict]:
    rows: list[dict] = []
    for st in statements:
        rows.extend(st.params["rows"])
    return rows


# --- KnnSettings ---------------------------------------------------------------------------


def test_from_config_reads_the_four_keys() -> None:
    s = knn.KnnSettings.from_config(CONFIG)
    assert s.top_k == 2
    assert s.cutoff == pytest.approx(0.1)
    assert s.code_top_k == 1
    assert s.code_cutoff == pytest.approx(0.5)


@pytest.mark.parametrize("missing", ["top_k", "cutoff", "code_top_k", "code_cutoff"])
def test_from_config_refuses_a_missing_key(missing: str) -> None:
    section = {k: v for k, v in CONFIG.items() if k != missing}
    with pytest.raises(ValueError):
        knn.KnnSettings.from_config(section)


@pytest.mark.parametrize(
    "key,value",
    [
        ("top_k", 0),
        ("code_top_k", 0),
        ("top_k", -1),
        ("top_k", 1.5),
        ("top_k", "3"),
        ("cutoff", 1.5),
        ("code_cutoff", 1.5),
        ("cutoff", -1.5),
        ("cutoff", "0.5"),
    ],
)
def test_from_config_refuses_bad_values(key: str, value: object) -> None:
    with pytest.raises(ValueError):
        knn.KnnSettings.from_config({**CONFIG, key: value})


def test_from_config_accepts_the_cutoff_bounds() -> None:
    s = knn.KnnSettings.from_config({**CONFIG, "cutoff": -1.0, "code_cutoff": 1.0})
    assert s.cutoff == pytest.approx(-1.0)
    assert s.code_cutoff == pytest.approx(1.0)


# --- nearest -------------------------------------------------------------------------------


def test_nearest_returns_top_k_by_cosine_in_order() -> None:
    got = knn.nearest(VECTORS, top_k=2, cutoff=0.1)
    _same_pairs(got, _expected(VECTORS, 2, 0.1))
    assert all(isinstance(t, tuple) and len(t) == 3 for t in got)


def test_nearest_order_is_from_asc_score_desc_to_asc() -> None:
    got = knn.nearest(VECTORS, top_k=4, cutoff=-1.0)
    keys = [(a, -s, b) for a, b, s in got]
    assert keys == sorted(keys)
    _same_pairs(got, _expected(VECTORS, 4, -1.0))


def test_nearest_applies_the_cutoff() -> None:
    got = knn.nearest(VECTORS, top_k=4, cutoff=0.9)
    assert got
    assert all(s >= 0.9 for _, _, s in got)
    _same_pairs(got, _expected(VECTORS, 4, 0.9))


def test_nearest_is_scale_invariant() -> None:
    scaled = {k: [x * (i + 2) * 3.5 for x in v] for i, (k, v) in enumerate(VECTORS.items())}
    _same_pairs(knn.nearest(scaled, top_k=2, cutoff=0.1), knn.nearest(VECTORS, top_k=2, cutoff=0.1))


def test_nearest_is_directed_not_symmetrised() -> None:
    vectors = {"a": [1.0, 0.0], "b": [0.8, 0.6], "c": [0.6, 0.8]}
    got = knn.nearest(vectors, top_k=1, cutoff=-1.0)
    pairs = [(a, b) for a, b, _ in got]
    assert ("a", "b") in pairs
    assert ("b", "c") in pairs
    assert ("b", "a") not in pairs
    assert pairs == [("a", "b"), ("b", "c"), ("c", "b")]


def test_nearest_breaks_ties_by_lower_to_id() -> None:
    vectors = {"x": [1.0, 0.0], "z": [0.0, 1.0], "y": [0.0, -1.0]}
    got = knn.nearest(vectors, top_k=1, cutoff=-1.0)
    assert [(a, b) for a, b, _ in got if a == "x"] == [("x", "y")]
    got2 = knn.nearest(vectors, top_k=2, cutoff=-1.0)
    assert [(a, b) for a, b, _ in got2 if a == "x"] == [("x", "y"), ("x", "z")]


def test_nearest_zero_vector_is_in_no_pair() -> None:
    vectors = {"a": [1.0, 0.0], "b": [0.9, 0.1], "z": [0.0, 0.0]}
    got = knn.nearest(vectors, top_k=5, cutoff=-1.0)
    assert all("z" not in (a, b) for a, b, _ in got)
    assert [(a, b) for a, b, _ in got] == [("a", "b"), ("b", "a")]


def test_nearest_identical_vectors_pair_with_one_and_never_self() -> None:
    vectors = {"p": [0.3, 0.4], "q": [0.3, 0.4]}
    got = knn.nearest(vectors, top_k=3, cutoff=-1.0)
    assert [(a, b) for a, b, _ in got] == [("p", "q"), ("q", "p")]
    assert all(s == pytest.approx(1.0) for _, _, s in got)
    assert all(a != b for a, b, _ in knn.nearest(VECTORS, top_k=10, cutoff=-1.0))


def test_nearest_top_k_larger_than_others_returns_every_qualifying() -> None:
    got = knn.nearest(VECTORS, top_k=50, cutoff=0.0)
    _same_pairs(got, _expected(VECTORS, 50, 0.0))
    from_e1 = [b for a, b, _ in got if a == "e1"]
    assert set(from_e1) == {"e2", "e3", "e4"}


def test_nearest_with_fewer_than_two_vectors_is_empty() -> None:
    assert knn.nearest({"a": [1.0, 0.0]}, top_k=1, cutoff=-1.0) == []
    assert knn.nearest({}, top_k=1, cutoff=-1.0) == []


def test_nearest_refuses_vectors_of_different_lengths() -> None:
    with pytest.raises(ValueError):
        knn.nearest({"a": [1.0, 0.0], "b": [1.0, 0.0, 0.0]}, top_k=1, cutoff=0.0)


def test_nearest_refuses_top_k_zero() -> None:
    with pytest.raises(ValueError):
        knn.nearest(VECTORS, top_k=0, cutoff=0.0)


# --- statements ----------------------------------------------------------------------------


def _check_clear(st: Statement, name: str, vectors: dict) -> None:
    assert isinstance(st, Statement)
    assert st.name == name
    assert st.kind == "write"
    assert st.scope == SCOPE
    assert st.params["scope"] == SCOPE
    assert list(st.params["ids"]) == sorted(vectors)
    assert "SIMILAR" in st.cypher
    checks.require_labeled(st.cypher)


def test_similar_statements_clear_then_score_rows() -> None:
    sts = knn.similar_statements(_schema(), SCOPE, VECTORS, _settings())
    _check_clear(sts[0], "knn:clear", VECTORS)
    assert "score" in sts[0].cypher
    writes = sts[1:]
    assert writes
    assert all(st.name == "write:SIMILAR" for st in writes)
    assert all(st.kind == "write" and st.scope == SCOPE for st in writes)
    rows = _rows(writes)
    expected = knn.nearest(VECTORS, top_k=2, cutoff=0.1)
    assert [(r["from_id"], r["to_id"]) for r in rows] == [(a, b) for a, b, _ in expected]
    for r, (_, _, s) in zip(rows, expected, strict=True):
        assert set(r["props"]) == {"score"}
        assert r["props"]["score"] == pytest.approx(s)


def test_code_similar_statements_clear_then_code_score_rows() -> None:
    sts = knn.code_similar_statements(_schema(), SCOPE, VECTORS, _settings())
    _check_clear(sts[0], "knn_code:clear", VECTORS)
    assert "code_score" in sts[0].cypher
    writes = sts[1:]
    assert writes
    assert all(st.name == "write:SIMILAR" for st in writes)
    rows = _rows(writes)
    expected = knn.nearest(VECTORS, top_k=1, cutoff=0.5)
    assert [(r["from_id"], r["to_id"]) for r in rows] == [(a, b) for a, b, _ in expected]
    for r, (_, _, s) in zip(rows, expected, strict=True):
        assert set(r["props"]) == {"code_score"}
        assert r["props"]["code_score"] == pytest.approx(s)


@pytest.mark.parametrize("builder", ["similar_statements", "code_similar_statements"])
def test_fewer_than_two_usable_vectors_gives_only_the_clear(builder: str) -> None:
    build = getattr(knn, builder)
    name = "knn:clear" if builder == "similar_statements" else "knn_code:clear"
    for vectors in ({"a": [1.0, 0.0]}, {"a": [1.0, 0.0], "z": [0.0, 0.0]}):
        sts = build(_schema(), SCOPE, vectors, _settings())
        assert len(sts) == 1
        _check_clear(sts[0], name, vectors)


@pytest.mark.parametrize("builder", ["similar_statements", "code_similar_statements"])
def test_no_pairs_above_cutoff_gives_only_the_clear(builder: str) -> None:
    settings = knn.KnnSettings(top_k=2, cutoff=1.0, code_top_k=2, code_cutoff=1.0)
    vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    sts = getattr(knn, builder)(_schema(), SCOPE, vectors, settings)
    assert len(sts) == 1


@pytest.mark.parametrize("builder", ["similar_statements", "code_similar_statements"])
def test_same_input_builds_the_same_statements(builder: str) -> None:
    build = getattr(knn, builder)
    first = build(_schema(), SCOPE, VECTORS, _settings())
    second = build(_schema(), SCOPE, VECTORS, _settings())
    assert first == second


@pytest.mark.parametrize("builder", ["similar_statements", "code_similar_statements"])
def test_empty_scope_is_refused(builder: str) -> None:
    with pytest.raises(ValueError):
        getattr(knn, builder)(_schema(), "", VECTORS, _settings())


@pytest.mark.parametrize("builder", ["similar_statements", "code_similar_statements"])
@pytest.mark.parametrize("bad", ["", 7])
def test_empty_or_non_string_id_is_refused(builder: str, bad: object) -> None:
    vectors = {"a": [1.0, 0.0], bad: [0.9, 0.1]}
    with pytest.raises(ValueError):
        getattr(knn, builder)(_schema(), SCOPE, vectors, _settings())
