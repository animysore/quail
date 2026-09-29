"""Codecs for physical plans."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping

from .base import InputPort, PhysicalGraph, PhysicalNode, PortRef, ValueType
from .nodes import (
    AiExtract,
    AiFilter,
    AiJoin,
    AiScore,
    Barrier,
    Exchange,
    Foreign,
    HashJoin,
    Limit,
    Project,
    Recombine,
    RequestExecution,
    Scan,
    ScoreFilter,
)


@dataclass(frozen=True)
class NodeCodec:
    """Encode and decode one registered physical node type."""

    node_type: type[PhysicalNode]

    def __post_init__(self) -> None:
        if not is_dataclass(self.node_type):
            raise TypeError("a physical node codec needs a dataclass")
        if not isinstance(self.type_name, str) or not self.type_name:
            raise TypeError("a physical node codec needs a nonempty type_name")

    @property
    def type_name(self) -> str:
        return self.node_type.type_name

    def _attribute_names(self) -> set[str]:
        return {
            field.name for field in fields(self.node_type)
        } - {"node_id", "inputs"}

    def encode(self, node: PhysicalNode) -> dict:
        if not isinstance(node, self.node_type):
            raise TypeError(
                f"codec {self.type_name!r} cannot encode "
                f"{type(node).__name__}")
        attributes = node.attributes()
        if not isinstance(attributes, Mapping):
            raise TypeError("physical node attributes must be a mapping")
        _require_fields(
            attributes,
            self._attribute_names(),
            f"physical node {self.type_name!r} attributes",
        )
        return {
            "type": self.type_name,
            "id": node.node_id,
            "inputs": [
                {
                    "name": input_port.name,
                    "value_type": input_port.value_type.value,
                    "schema": list(input_port.schema),
                    "source": {
                        "node_id": input_port.source.node_id,
                        "port": input_port.source.port,
                    },
                }
                for input_port in node.inputs
            ],
            "attributes": attributes,
        }

    def decode(self, value: Mapping[str, Any]) -> PhysicalNode:
        _require_fields(
            value,
            {"type", "id", "inputs", "attributes"},
            "physical node",
        )
        if value["type"] != self.type_name:
            raise ValueError(
                f"codec {self.type_name!r} cannot decode "
                f"{value['type']!r}"
            )
        _require_fields(
            value["attributes"],
            self._attribute_names(),
            f"physical node {self.type_name!r} attributes",
        )
        for input_port in value["inputs"]:
            _require_fields(
                input_port,
                {"name", "value_type", "schema", "source"},
                "physical input port",
            )
            _require_fields(
                input_port["source"],
                {"node_id", "port"},
                "physical input source",
            )
        inputs = tuple(
            InputPort(
                name=input_port["name"],
                value_type=ValueType(input_port["value_type"]),
                source=PortRef(
                    input_port["source"]["node_id"],
                    input_port["source"]["port"],
                ),
                schema=tuple(input_port["schema"]),
            )
            for input_port in value["inputs"]
        )
        return self.node_type.from_attributes(
            value["id"], inputs, value["attributes"]
        )


def _require_fields(
    value: Mapping[str, Any], expected: set[str], name: str
) -> None:
    actual = set(value)
    missing = expected - actual
    if missing:
        raise ValueError(f"{name} is missing fields {sorted(missing)}")
    extra = actual - expected
    if extra:
        raise ValueError(f"{name} has unknown fields {sorted(extra)}")


def built_in_codecs() -> tuple[NodeCodec, ...]:
    """Return codecs for every built in physical node."""
    return tuple(NodeCodec(node_type) for node_type in (
        Scan,
        AiFilter,
        AiExtract,
        RequestExecution,
        AiScore,
        ScoreFilter,
        Barrier,
        Exchange,
        Foreign,
        HashJoin,
        AiJoin,
        Recombine,
        Project,
        Limit,
    ))


def encode_graph(
    graph: PhysicalGraph,
    codecs: Mapping[str, NodeCodec],
) -> dict:
    """Encode one physical graph."""
    return {
        "nodes": [
            codecs[node.type_name].encode(node)
            for node in graph.nodes
        ],
        "root": {
            "node_id": graph.root.node_id,
            "port": graph.root.port,
        },
    }


def decode_graph(
    value: Mapping[str, Any], codecs: Mapping[str, NodeCodec]
) -> PhysicalGraph:
    """Decode and validate one physical graph."""
    _require_fields(value, {"nodes", "root"}, "physical graph")
    nodes = []
    for encoded in value["nodes"]:
        type_name = encoded["type"]
        if type_name not in codecs:
            raise ValueError(f"unknown physical node type {type_name!r}")
        nodes.append(codecs[type_name].decode(encoded))
    root = value["root"]
    _require_fields(root, {"node_id", "port"}, "physical graph root")
    graph = PhysicalGraph(
        nodes=tuple(nodes),
        root=PortRef(root["node_id"], root["port"]),
    )
    graph.validate()
    return graph


def plan_envelope(
    *,
    backend: str,
    model: str,
    device: str,
    workers: int,
    graph: PhysicalGraph,
    codecs: Mapping[str, NodeCodec],
    settings: Mapping[str, Any] | None = None,
) -> dict:
    """Encode the physical plan fields sent across a process boundary."""
    graph.validate_backend(backend)
    return {
        "backend": backend,
        "model": model,
        "device": device,
        "workers": workers,
        "settings": dict(settings or {}),
        "graph": encode_graph(graph, codecs),
    }


def check_plan_envelope(value: Mapping[str, Any]) -> None:
    """Validate a physical plan envelope."""
    required = {
        "backend", "model", "device", "workers",
        "settings", "graph",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(
            f"physical plan is missing fields {sorted(missing)}"
        )
    extra = set(value) - required
    if extra:
        raise ValueError(
            f"physical plan has unknown fields {sorted(extra)}"
        )
    if not isinstance(value["settings"], Mapping):
        raise TypeError("physical plan settings must be a mapping")
    for name in ("backend", "model", "device"):
        if not isinstance(value[name], str) or not value[name]:
            raise TypeError(f"physical plan {name} must be a nonempty string")
    if not isinstance(value["workers"], int) or value["workers"] <= 0:
        raise ValueError("physical plan workers must be positive")
    if not isinstance(value["graph"], Mapping):
        raise TypeError("physical plan graph must be a mapping")
