"""GR-02: config/graph_schema.yaml is plain: names, endpoints and fixed value sets only."""

from pathlib import Path
from typing import Any

import yaml

SCHEMA = Path(__file__).resolve().parents[2] / "config" / "graph_schema.yaml"
REMOVED = {
    "layer",
    "writer",
    "readers",
    "rule",
    "ingest_only",
    "required",
    "required_when",
    "pairs",
    "endpoint_offset",
    "same_fact_as",
}
VALUE_SETS = {
    ("Artifact", "status"),
    ("Entity", "status"),
    ("RELATED_TO", "kind"),
    ("CODE_LINK", "kind"),
}


def _schema() -> dict[str, Any]:
    data = yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _keys(node: object) -> set[str]:
    if isinstance(node, dict):
        return set(node) | {k for v in node.values() for k in _keys(v)}
    if isinstance(node, list):
        return {k for v in node for k in _keys(v)}
    return set()


def test_no_proof_of_use_or_constraint_keys() -> None:
    assert _keys(_schema()) & REMOVED == set()


def test_values_only_for_status_and_the_two_kinds() -> None:
    data = _schema()
    found = {
        (name, prop)
        for section in ("nodes", "relationships")
        for name, entry in data[section].items()
        for prop in entry.get("values", {})
    }
    assert found == VALUE_SETS


def test_every_property_is_a_plain_name() -> None:
    data = _schema()
    for section in ("nodes", "relationships"):
        for entry in data[section].values():
            assert all(isinstance(p, str) for p in entry.get("properties", []))


def test_every_value_names_a_declared_property() -> None:
    data = _schema()
    for section in ("nodes", "relationships"):
        for entry in data[section].values():
            assert set(entry.get("values", {})) <= set(entry.get("properties", []))


def test_relationship_endpoints_are_declared_labels() -> None:
    data = _schema()
    labels = set(data["nodes"])
    for entry in data["relationships"].values():
        ends = {
            e
            for side in (entry["from"], entry["to"])
            for e in (side if isinstance(side, list) else [side])
        }
        assert ends <= labels


def test_names_match_the_schema() -> None:
    data = _schema()
    names = data["names"]
    assert set(names["labels"].values()) == set(data["nodes"])
    assert set(names["relationships"].values()) == set(data["relationships"])
    assert set(names["statuses"].values()) == set(data["nodes"]["Artifact"]["values"]["status"])
    every = {p for n in data["nodes"].values() for p in n["properties"]}
    every |= {p for r in data["relationships"].values() for p in r.get("properties", [])}
    assert set(names["properties"].values()) <= every


def test_roles_the_graph_slices_need() -> None:
    props = _schema()["names"]["properties"]
    for role in ("dirty", "name", "embedding_code", "score", "code_score"):
        assert props[role] == role
    assert "dirty" in _schema()["nodes"]["Community"]["properties"]
