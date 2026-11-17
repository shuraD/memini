"""Acceptance tests for GR-07 (first slice): the SIMILAR KNN pass and the per-level Leiden pass.

The module talks to Neo4j only through a passed-in graphdatascience client, so these tests hand
it a recording fake client and check the calls it receives (spec GR-07, acceptance lines).
"""

from __future__ import annotations

import importlib
from typing import Any

SCOPE = "scope-gr07-test"


class FakeResult:
    """A permissive stand-in for any GDS / Cypher result (the spec does not require reading it)."""

    def __getitem__(self, key: Any) -> int:
        return 0

    def get(self, key: Any, default: Any = None) -> int:
        return 0

    def __iter__(self) -> Any:
        return iter(())

    def __len__(self) -> int:
        return 0

    def __bool__(self) -> bool:
        return False


class FakeGraph:
    def __init__(self, client: FakeGds, name: str) -> None:
        self._client = client
        self._name = name
        self.dropped = False

    def name(self) -> str:
        return self._name

    def drop(self, *args: Any, **kwargs: Any) -> FakeResult:
        self._client.log.append(("graph.drop", (self,), {}))
        self.dropped = True
        return FakeResult()


class _Path:
    """An attribute path on the fake client: any attribute chain is callable and recorded."""

    def __init__(self, client: FakeGds, path: str) -> None:
        self._client = client
        self._path = path

    def __getattr__(self, attr: str) -> _Path:
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Path(self._client, f"{self._path}.{attr}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._client.handle(self._path, args, kwargs)


class FakeGds:
    def __init__(self) -> None:
        self.log: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.graphs: list[FakeGraph] = []
        self.projection_args: dict[int, str] = {}

    def __getattr__(self, attr: str) -> _Path:
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Path(self, attr)

    def handle(self, path: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if path.startswith("graph") and "project" in path:
            graph = FakeGraph(self, f"g{len(self.graphs)}")
            self.graphs.append(graph)
            self.projection_args[id(graph)] = _text(args, kwargs)
            self.log.append((path, args, kwargs))
            return graph, FakeResult()
        if path == "graph.drop":
            target = args[0] if args else next(iter(kwargs.values()), None)
            for graph in self.graphs:
                if graph is target or graph.name() == target:
                    graph.dropped = True
            self.log.append((path, args, kwargs))
            return FakeResult()
        self.log.append((path, args, kwargs))
        return FakeResult()

    def calls(self, path: str) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
        return [(a, k) for p, a, k in self.log if p == path]

    def index_of_first(self, path: str) -> int:
        for i, (p, _, _) in enumerate(self.log):
            if p == path:
                return i
        return -1


def _text(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    return repr(args) + " " + repr(kwargs)


def _index() -> Any:
    return importlib.import_module("memini.core.archrag.index")


def _prop_names(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return {str(k) for k in value}
    names: set[str] = set()
    for item in value:
        names |= _prop_names(item)
    return names


def _graph_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    return kwargs.get("G", kwargs.get("graph"))


# --- similar_pass ---------------------------------------------------------------------------


def test_similar_pass_makes_two_knn_writes_content_and_code() -> None:
    gds = FakeGds()
    _index().similar_pass(gds, SCOPE, top_k=7, cutoff=0.55, seed=11)
    knn = gds.calls("knn.write")
    assert len(knn) == 2
    by_prop = {}
    for _, kwargs in knn:
        by_prop[frozenset(_prop_names(kwargs["nodeProperties"]))] = kwargs
    assert set(by_prop) == {frozenset({"embedding"}), frozenset({"embedding_code"})}
    assert by_prop[frozenset({"embedding"})]["writeProperty"] == "score"
    assert by_prop[frozenset({"embedding_code"})]["writeProperty"] == "code_score"
    for _, kwargs in knn:
        assert kwargs["topK"] == 7
        assert kwargs["similarityCutoff"] == 0.55
        assert kwargs["writeRelationshipType"] == "SIMILAR"
        assert kwargs["randomSeed"] == 11
        assert kwargs["concurrency"] == 1


def test_similar_pass_defaults_top_k_10_cutoff_0_7() -> None:
    gds = FakeGds()
    _index().similar_pass(gds, SCOPE)
    knn = gds.calls("knn.write")
    assert len(knn) == 2
    for _, kwargs in knn:
        assert kwargs["topK"] == 10
        assert kwargs["similarityCutoff"] == 0.7
        assert kwargs["randomSeed"] == 42
        assert kwargs["concurrency"] == 1


def test_similar_pass_deletes_old_similar_edges_of_scope_before_knn() -> None:
    gds = FakeGds()
    _index().similar_pass(gds, SCOPE)
    first_knn = gds.index_of_first("knn.write")
    assert first_knn >= 0
    deletes = []
    for path, args, kwargs in gds.log[:first_knn]:
        if path != "run_cypher":
            continue
        query = args[0] if args else kwargs.get("query", "")
        if "SIMILAR" in query and "DELETE" in query.upper():
            deletes.append(_text(args, kwargs))
    assert deletes, "no run_cypher deleting SIMILAR edges before the first knn.write"
    assert any(SCOPE in text for text in deletes)


def test_similar_pass_projections_are_scoped_active_entity_and_directed() -> None:
    gds = FakeGds()
    _index().similar_pass(gds, SCOPE)
    first_knn = gds.index_of_first("knn.write")
    assert first_knn >= 0
    projections = [
        _text(a, k) for p, a, k in gds.log[:first_knn] if p.startswith("graph") and "project" in p
    ]
    assert projections, "no projection before the KNN calls"
    for text in projections:
        assert SCOPE in text
        assert "Entity" in text
        assert "active" in text
        assert "undirected" not in text.lower()


def test_similar_pass_knn_runs_on_projected_live_graphs() -> None:
    gds = FakeGds()
    _index().similar_pass(gds, SCOPE)
    knn = gds.calls("knn.write")
    assert len(knn) == 2
    for args, kwargs in knn:
        graph = _graph_arg(args, kwargs)
        assert any(graph is g for g in gds.graphs)
    # A graph used by a KNN call is not dropped before that call.
    dropped: set[int] = set()
    for path, args, kwargs in gds.log:
        if path == "knn.write":
            assert id(_graph_arg(args, kwargs)) not in dropped
        if path == "graph.drop" and args:
            dropped.add(id(args[0]))


def test_similar_pass_drops_every_projected_graph() -> None:
    gds = FakeGds()
    _index().similar_pass(gds, SCOPE)
    assert gds.graphs
    assert all(g.dropped for g in gds.graphs)


# --- leiden_pass ----------------------------------------------------------------------------


def _leiden_projection_text(gds: FakeGds) -> str:
    leiden = gds.calls("leiden.write")
    assert len(leiden) == 1
    graph = _graph_arg(*leiden[0])
    assert any(graph is g for g in gds.graphs), "leiden.write did not run on a projected graph"
    return gds.projection_args[id(graph)]


def test_leiden_pass_level0_one_write_with_gamma_seed_on_undirected_projection() -> None:
    gds = FakeGds()
    _index().leiden_pass(gds, SCOPE, 0, resolutions=(2.5, 0.5), seed=9)
    leiden = gds.calls("leiden.write")
    assert len(leiden) == 1
    _, kwargs = leiden[0]
    assert kwargs["gamma"] == 2.5
    assert kwargs["randomSeed"] == 9
    text = _leiden_projection_text(gds)
    assert SCOPE in text
    assert "Entity" in text
    assert "RELATED_TO" in text
    assert "SIMILAR" in text
    assert "undirected" in text.lower()


def test_leiden_pass_gamma_follows_level_then_last_resolution() -> None:
    gammas = []
    for level in (0, 1, 2, 5):
        gds = FakeGds()
        _index().leiden_pass(gds, SCOPE, level, resolutions=(2.5, 0.5), seed=9)
        leiden = gds.calls("leiden.write")
        assert len(leiden) == 1
        gammas.append(leiden[0][1]["gamma"])
    assert gammas == [2.5, 0.5, 0.5, 0.5]


def test_leiden_pass_default_schedule_and_seed() -> None:
    gammas = []
    seeds = []
    for level in (0, 1, 2):
        gds = FakeGds()
        _index().leiden_pass(gds, SCOPE, level)
        leiden = gds.calls("leiden.write")
        assert len(leiden) == 1
        gammas.append(leiden[0][1]["gamma"])
        seeds.append(leiden[0][1]["randomSeed"])
    assert gammas == [4.0, 1.0, 0.25]
    assert seeds == [42, 42, 42]


def test_leiden_pass_upper_level_projects_communities_of_scope_undirected() -> None:
    for level in (1, 2):
        gds = FakeGds()
        _index().leiden_pass(gds, SCOPE, level, resolutions=(2.5, 0.5), seed=9)
        text = _leiden_projection_text(gds)
        assert "Community" in text
        assert SCOPE in text
        assert "undirected" in text.lower()


def test_leiden_pass_drops_every_projected_graph() -> None:
    for level in (0, 1):
        gds = FakeGds()
        _index().leiden_pass(gds, SCOPE, level)
        assert gds.graphs
        assert all(g.dropped for g in gds.graphs)
