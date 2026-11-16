"""Statements that cluster one hierarchy level of one scope with Leiden in GDS.

One level is clustered by four statements: drop any stale in-memory graph, project the
level's nodes and their ``RELATED_TO`` and ``SIMILAR`` edges (undirected only inside the
projection), stream Leiden once on it, and drop the projection again. Streaming instead of
writing keeps the community id out of the stored graph. Nothing is run here; the caller
runs the returned statements and reads the stream rows back.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from memini.core.graph import checks
from memini.core.graph.types import Statement
from memini.core.schema import Schema, node_key

__all__ = ["LeidenSettings", "graph_name", "level_statements"]

_BASE_LABEL = "Entity"
_UPPER_LABEL = "Community"
_REL_TYPES = ("RELATED_TO", "SIMILAR")
_SCORED_REL = "SIMILAR"
_SCORE = "score"
_WEIGHT = "weight"
_DEFAULT_KEY = "id"


def _require(section: Mapping[str, Any], key: str) -> Any:
    """Return ``section[key]``; raise ``ValueError`` when the key is missing."""
    if key not in section:
        raise ValueError(f"leiden config misses {key!r}")
    return section[key]


def _resolutions(value: Any) -> tuple[float, ...]:
    """Return ``value`` as a non-empty tuple of floats > 0; raise ``ValueError`` otherwise."""
    if not isinstance(value, list | tuple) or not value:
        raise ValueError(f"leiden 'resolutions' must be a non-empty list, got {value!r}")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float) or not item > 0:
            raise ValueError(f"leiden resolution must be a number > 0, got {item!r}")
    return tuple(float(item) for item in value)


def _check(scope: str, level: int) -> None:
    """Raise ``ValueError`` for an empty scope or a negative level."""
    if not scope:
        raise ValueError("scope must be non-empty")
    if level < 0:
        raise ValueError(f"level must be >= 0, got {level}")


@dataclass(frozen=True)
class LeidenSettings:
    """Leiden parameters of the hierarchy build.

    Attributes:
        resolutions: Per-level resolution schedule; index = hierarchy level, the last
            entry applies to every deeper level.
        seed: Leiden random seed (>= 0).
        weighted: Use the ``SIMILAR`` score as edge weight.

    """

    resolutions: tuple[float, ...]
    seed: int
    weighted: bool

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> LeidenSettings:
        """Read keys "resolutions", "seed", "weighted".

        Raises:
            ValueError: When a key is missing, the schedule is empty or holds a value that
                is not a number > 0, the seed is not an int >= 0, or weighted is not a bool.

        """
        resolutions = _resolutions(_require(section, "resolutions"))
        seed = _require(section, "seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"leiden 'seed' must be an int >= 0, got {seed!r}")
        weighted = _require(section, "weighted")
        if not isinstance(weighted, bool):
            raise ValueError(f"leiden 'weighted' must be a bool, got {weighted!r}")
        return cls(resolutions=resolutions, seed=seed, weighted=weighted)

    def resolution(self, level: int) -> float:
        """Return the resolution of ``level``, clamped to the last scheduled entry.

        Raises:
            ValueError: When ``level`` < 0.

        """
        if level < 0:
            raise ValueError(f"level must be >= 0, got {level}")
        return self.resolutions[min(level, len(self.resolutions) - 1)]


def graph_name(scope: str, level: int) -> str:
    """Return the in-memory graph name of one level of one scope.

    Raises:
        ValueError: For an empty scope or a negative level.

    """
    _check(scope, level)
    return f"leiden_{level}_{scope}"


def _key(schema: Schema, label: str) -> str:
    """Return the key property of ``label``, or ``id`` when the schema does not declare it."""
    node = schema.nodes.get(label)
    key = node_key(node) if node is not None else None
    return key if key is not None else _DEFAULT_KEY


def level_statements(
    schema: Schema, scope: str, level: int, settings: LeidenSettings
) -> list[Statement]:
    """Return the drop, project, stream and drop statements of one level.

    Level 0 projects the scope's ``Entity`` nodes; level L >= 1 projects the scope's
    ``Community`` nodes produced by clustering level L - 1. The stream returns one row per
    node with columns ``id`` (the node key) and ``community``.

    Raises:
        ValueError: For an empty scope or a negative level.

    """
    name = graph_name(scope, level)
    label = _BASE_LABEL if level == 0 else _UPPER_LABEL
    props = "{scope: $scope}" if level == 0 else "{scope: $scope, level: $level}"
    rel_types = "|".join(_REL_TYPES)
    rel_list = ", ".join(f"'{t}'" for t in _REL_TYPES)
    weight = (
        f", relationshipProperties: {{{_WEIGHT}: CASE type(r) WHEN '{_SCORED_REL}' "
        f"THEN coalesce(r.{_SCORE}, 1.0) ELSE 1.0 END}}"
        if settings.weighted
        else ""
    )
    project_cypher = (
        f"MATCH (s:{label} {props}) "
        f"OPTIONAL MATCH (s)-[r:{rel_types}]->(t:{label} {props}) "
        f"WITH gds.graph.project($graph, s, t, {{relationshipType: type(r){weight}}}, "
        f"{{undirectedRelationshipTypes: [{rel_list}]}}) AS g "
        "RETURN g.graphName AS graph"
    )
    checks.require_labeled(project_cypher)
    project_params: dict[str, Any] = {"graph": name, "scope": scope}
    if level > 0:
        project_params["level"] = level - 1

    weight_config = f", relationshipWeightProperty: '{_WEIGHT}'" if settings.weighted else ""
    stream_cypher = (
        "CALL gds.leiden.stream($graph, {gamma: $resolution, randomSeed: $seed, "
        f"maxLevels: 1, concurrency: 1{weight_config}}}) "
        "YIELD nodeId, communityId "
        f"RETURN gds.util.asNode(nodeId).{_key(schema, label)} AS id, "
        "communityId AS community ORDER BY id"
    )
    drop = Statement(
        name="leiden:drop",
        cypher="CALL gds.graph.drop($graph, false) YIELD graphName RETURN graphName",
        params={"graph": name},
        kind="admin",
        scope=scope,
    )
    return [
        drop,
        Statement(
            name="leiden:project",
            cypher=project_cypher,
            params=project_params,
            kind="admin",
            scope=scope,
        ),
        Statement(
            name="leiden:stream",
            cypher=stream_cypher,
            params={"graph": name, "resolution": settings.resolution(level), "seed": settings.seed},
            kind="read",
            scope=scope,
        ),
        drop,
    ]
