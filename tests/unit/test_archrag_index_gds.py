"""GR-07 acceptance: the ArchRAG per-level GDS passes, against a recording fake GDS client.

The fake stands in for ``graphdatascience.GraphDataScience``: every attribute chain is
recorded with its call arguments. A call whose path contains ``project`` returns a
``(graph, result)`` pair whose graph counts as dropped after ``graph.drop()``, the end of a
``with graph:`` block, or ``gds.graph.drop(graph)`` / ``gds.graph.drop(graph.name())``;
``run_cypher`` returns an empty list;
any other call returns an empty mapping. Graph names are read from ``config/graph_schema.yaml``.
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Any

import pytest
import yaml

from memini.core.archrag.index import leiden_pass, similar_pass

SCHEMA = yaml.safe_load(
    (Path(__file__).resolve().parents[2] / "config" / "graph_schema.yaml").read_text("utf-8")
)
NAMES = SCHEMA["names"]
SIMILAR = NAMES["relationships"]["similar"]
RELATED_TO = NAMES["relationships"]["related_to"]
ENTITY = NAMES["labels"]["entity"]
COMMUNITY = NAMES["labels"]["community"]
EMBEDDING = NAMES["properties"]["embedding"]
ACTIVE = NAMES["statuses"]["active"]
EMBEDDING_CODE = "embedding_code"  # the Entity's code vector, listed in the schema's nodes
SCORE, CODE_SCORE = SCHEMA["relationships"][SIMILAR]["properties"]
SCOPE = "qa-gr07-scope"

GraphCreateResult = namedtuple("GraphCreateResult", ["graph", "result"])


class _Result(dict[str, Any]):
    def __missing__(self, key: str) -> int:
        return 0


class FakeGraph:
    def __init__(self, log: list[tuple[str, tuple[Any, ...], dict[str, Any]]], n: int) -> None:
        self._log = log
        self._name = f"g{n}"
        self.dropped = False

    def name(self) -> str:
        return self._name

    def drop(self, *args: Any, **kwargs: Any) -> _Result:
        self.dropped = True
        self._log.append(("graph.drop", (self,), {}))
        return _Result()

    def __enter__(self) -> FakeGraph:  # the real Graph drops itself when its with-block exits
        return self

    def __exit__(self, *exc: Any) -> None:
        self.drop()


class Node:
    def __init__(self, root: FakeGds, path: str) -> None:
        self._root = root
        self._path = path

    def __getattr__(self, attr: str) -> Node:
        if attr.startswith("__"):
            raise AttributeError(attr)
        return Node(self._root, f"{self._path}.{attr}" if self._path else attr)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._root.record(self._path, args, kwargs)


class FakeGds(Node):
    def __init__(self) -> None:
        self.log: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.graphs: list[FakeGraph] = []
        super().__init__(self, "")

    def record(self, path: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self.log.append((path, args, kwargs))
        if path == "run_cypher":
            return []
        if "project" in path.split("."):
            graph = FakeGraph(self.log, len(self.graphs))
            self.graphs.append(graph)
            return GraphCreateResult(graph, _Result())
        if path == "graph.drop":
            for arg in [*args, *kwargs.values()]:
                for graph in self.graphs:
                    if arg is graph or arg == graph.name():  # the graph, or its name
                        graph.dropped = True
        return _Result()

    def calls(self, suffix: str) -> list[tuple[int, tuple[Any, ...], dict[str, Any]]]:
        return [(i, a, k) for i, (p, a, k) in enumerate(self.log) if p.endswith(suffix)]

    def projections(self) -> list[tuple[int, str]]:
        return [
            (i, repr((a, k))) for i, (p, a, k) in enumerate(self.log) if "project" in p.split(".")
        ]


def _props(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return {str(k) for k in value}
    return {str(v) for v in value}


def _query(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("query", args[0] if args else ""))


@pytest.fixture
def gds() -> FakeGds:
    return FakeGds()


def test_similar_pass_writes_two_directed_knn_passes_from_config(gds: FakeGds) -> None:
    similar_pass(gds, SCOPE, top_k=7, cutoff=0.55, seed=11)
    knn = gds.calls("knn.write")
    assert len(knn) == 2
    by_prop = {frozenset(_props(k["nodeProperties"])): k for _, _, k in knn}
    assert set(by_prop) == {frozenset({EMBEDDING}), frozenset({EMBEDDING_CODE})}
    for kwargs in by_prop.values():
        assert kwargs["topK"] == 7
        assert kwargs["similarityCutoff"] == pytest.approx(0.55)
        assert kwargs["writeRelationshipType"] == SIMILAR
        assert kwargs["randomSeed"] == 11
        assert kwargs["concurrency"] == 1
    assert by_prop[frozenset({EMBEDDING})]["writeProperty"] == SCORE
    assert by_prop[frozenset({EMBEDDING_CODE})]["writeProperty"] == CODE_SCORE


def test_similar_pass_defaults_are_the_old_values(gds: FakeGds) -> None:
    similar_pass(gds, SCOPE)
    knn = gds.calls("knn.write")
    assert len(knn) == 2
    for _, _, kwargs in knn:
        assert kwargs["topK"] == 10
        assert kwargs["similarityCutoff"] == pytest.approx(0.7)


def test_similar_pass_removes_old_similar_edges_before_knn(gds: FakeGds) -> None:
    similar_pass(gds, SCOPE, seed=3)
    first_knn = min(i for i, _, _ in gds.calls("knn.write"))
    deletes = [
        i
        for i, a, k in gds.calls("run_cypher")
        if SIMILAR in _query(a, k) and "delete" in _query(a, k).lower()
    ]
    assert deletes
    assert min(deletes) < first_knn
    scoped = [repr((a, k)) for i, a, k in gds.calls("run_cypher") if i in deletes]
    assert any(SCOPE in text for text in scoped)


def test_similar_pass_projects_scope_active_entities_directed(gds: FakeGds) -> None:
    similar_pass(gds, SCOPE, seed=3)
    first_knn = min(i for i, _, _ in gds.calls("knn.write"))
    before = [text for i, text in gds.projections() if i < first_knn]
    assert before
    for text in before:
        assert SCOPE in text
        assert ENTITY in text
        assert ACTIVE in text
        assert "undirected" not in text.lower()


def test_similar_pass_drops_every_projection(gds: FakeGds) -> None:
    similar_pass(gds, SCOPE, seed=3)
    assert gds.graphs
    assert all(graph.dropped for graph in gds.graphs)


def test_leiden_pass_level0_runs_once_seeded_on_undirected_projection(gds: FakeGds) -> None:
    leiden_pass(gds, SCOPE, 0, resolutions=(2.5, 0.5), seed=9)
    leiden = gds.calls("leiden.write")
    assert len(leiden) == 1
    index, _, kwargs = leiden[0]
    assert kwargs["gamma"] == pytest.approx(2.5)
    assert kwargs["randomSeed"] == 9
    before = [text for i, text in gds.projections() if i < index]
    assert before
    last = before[-1]
    assert "undirected" in last.lower()
    assert RELATED_TO in last
    assert SIMILAR in last
    assert ENTITY in last
    assert SCOPE in last
    assert all(graph.dropped for graph in gds.graphs)


def test_leiden_pass_follows_the_per_level_schedule(gds: FakeGds) -> None:
    leiden_pass(gds, SCOPE, 1, resolutions=(2.5, 0.5, 0.125), seed=9)
    leiden_pass(gds, SCOPE, 5, resolutions=(2.5, 0.5, 0.125), seed=9)
    gammas = [k["gamma"] for _, _, k in gds.calls("leiden.write")]
    assert gammas == [pytest.approx(0.5), pytest.approx(0.125)]


def test_leiden_pass_upper_level_projects_communities_of_the_level_below(gds: FakeGds) -> None:
    leiden_pass(gds, SCOPE, 2, seed=9)
    index, _, kwargs = gds.calls("leiden.write")[0]
    assert kwargs["gamma"] == pytest.approx(0.25)
    last = [text for i, text in gds.projections() if i < index][-1]
    assert COMMUNITY in last
    assert "undirected" in last.lower()
    assert SCOPE in last
    assert all(graph.dropped for graph in gds.graphs)


def test_leiden_pass_default_schedule_is_the_old_one(gds: FakeGds) -> None:
    for level in range(3):
        leiden_pass(gds, SCOPE, level, seed=1)
    gammas = [k["gamma"] for _, _, k in gds.calls("leiden.write")]
    assert gammas == [pytest.approx(4.0), pytest.approx(1.0), pytest.approx(0.25)]
