"""Physical planning and runtime extension test."""

import copy
import re
import sys
import types
from dataclasses import dataclass, replace
from typing import ClassVar

import pyarrow as pa
import pytest

import quail
from quail.builtins import built_in_registry
from quail.catalog import DocumentProvider
from quail.execution.execute import _execute_physical
from quail.execution.runner import NodeResult
from quail.execution.types import PhysicalRequest, PhysicalResponse, document_input
from quail.physical import (
    ExecutionLocation,
    InputPort,
    OutputPort,
    PhysicalGraph,
    PhysicalNode,
    PortRef,
    Project,
    Scan,
    ValueType,
    decode_graph,
    encode_graph,
    plan_envelope,
)
from quail.planner.physical_optimizer import PhysicalCandidate, SupportResult
from quail.planner.plan import EngineConfig, PhysicalPlan


@dataclass(frozen=True)
class FirstDocuments(PhysicalNode):
    alias: str = ""
    count: int = 1

    type_name: ClassVar[str] = "test.first_documents"
    runtime_key: ClassVar[str] = type_name
    location: ClassVar[ExecutionLocation] = ExecutionLocation.COORDINATOR

    @property
    def outputs(self):
        return (OutputPort(f"ids:{self.alias}", ValueType.DOCUMENT_IDS,
                           schema=(self.alias,)),)

    def attributes(self):
        return {"alias": self.alias, "count": self.count}

    @classmethod
    def from_attributes(cls, node_id, inputs, attributes):
        return cls(node_id=node_id, inputs=inputs, alias=attributes["alias"],
                   count=int(attributes["count"]))


class FirstDocumentsRuntime:
    def execute(self, node, inputs, context):
        documents = next(iter(inputs.values()))
        return NodeResult({f"ids:{node.alias}": documents[:node.count]})


def local_plan(context, *, count, estimate, source):
    def ids_from(node_id):
        return (InputPort("input:0", ValueType.DOCUMENT_IDS,
                          PortRef(node_id, "ids:d"), schema=("d",)),)

    scan = Scan(node_id="input:d", alias="d", input_id="d",
                n_docs=len(context.document_tokens["d"]))
    filtered = FirstDocuments(
        node_id="filter:d", inputs=ids_from("input:d"), alias="d", count=count)
    project = Project(
        node_id="project", inputs=ids_from("filter:d"), columns=("d.id",))
    plan = PhysicalPlan(
        model=context.model.name, device=context.device.name,
        workers=context.gpu_count, backend="local_filter",
        estimated_seconds=estimate, nodes=(scan, filtered, project),
        settings={"planner_source": source})
    return PhysicalCandidate(plan.graph, plan, estimate)


class LocalFilterBackend:
    name = "local_filter"

    def __init__(self):
        self.called = False

    def supports(self, model, device, gpu_count):
        return SupportResult.accept()

    def plan(self, region, context):
        self.called = True
        return (local_plan(context, count=1, estimate=0.0, source="backend"),)


class PreferredPlanner:
    name = "preferred"

    def __init__(self):
        self.called = False

    def plan(self, region, context):
        self.called = True
        return (local_plan(
            context, count=2, estimate=-1.0, source="preferred planner"),)


class KeepFirstDocument:
    name = "keep_first_document"

    def rewrite(self, graph, context):
        return PhysicalGraph(
            tuple(replace(node, count=1) if isinstance(node, FirstDocuments)
                  else node for node in graph.nodes),
            graph.root)


def test_physical_extensions_plan_validate_and_execute(monkeypatch):
    registry = quail.ExtensionRegistry.with_built_ins()
    backend = LocalFilterBackend()
    planner = PreferredPlanner()
    registry.register_backend(backend)
    registry.register_physical_planner(planner)
    registry.register_physical_rule(KeepFirstDocument())
    registry.register_node(FirstDocuments, runtime=FirstDocumentsRuntime())
    session = quail.Session(
        EngineConfig(gpus=1, model="qwen3-4b-fp8", backend="local_filter",
                     device="h100-sxm"),
        tokenizer=str.split,
        registry=registry,
    )
    session.register("docs", DocumentProvider.from_table(
        pa.table({"id": ["a", "b"], "body": ["one", "two"]}),
        id_col="id", identity="local-filter-docs"))
    query = session.sql(
        "SELECT d.id FROM docs d WHERE AI_FILTER(PROMPT('ok {0}', d.body))")
    plan = query.plan()
    envelope = plan.to_envelope(registry.codecs)

    assert backend.called
    assert planner.called
    assert plan.settings["planner_source"] == "preferred planner"
    assert "physical rule keep_first_document" in plan.remarks[0]
    assert decode_graph(envelope["graph"], registry.codecs) == plan.graph

    registry = built_in_registry()
    scan = Scan(node_id="input:d", alias="d", input_id="d", n_docs=1,
                shard_ranges=((0, 1),), shard_token_loads=(1,))
    plan = PhysicalGraph((scan,), PortRef("input:d", "ids:d"))
    encoded = encode_graph(plan, registry.codecs)
    missing = copy.deepcopy(encoded)
    del missing["nodes"][0]["attributes"]["shard_ranges"]
    with pytest.raises(ValueError, match="missing fields"):
        decode_graph(missing, registry.codecs)

    extra = copy.deepcopy(encoded)
    extra["nodes"][0]["unused"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        decode_graph(extra, registry.codecs)

    class ExampleBackend:
        name = "test.example"

        def execute_request(self, context):
            nodes = [node.node_id for node in context.graph.nodes]
            return PhysicalResponse({}, {"backend": self.name, "nodes": nodes})

    def register(registry):
        registry.register_backend(ExampleBackend())

    module = types.ModuleType("test_quail_extension")
    module.register_quail_extension = register
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = built_in_registry()
    scan = Scan(node_id="input:d", alias="d", input_id="d", n_docs=2)
    graph = PhysicalGraph((scan,), PortRef("input:d", "ids:d"))
    request = PhysicalRequest(
        plan_envelope(backend=ExampleBackend.name, model="qwen3-4b-fp8",
                      device="h100-sxm", workers=1, graph=graph,
                      codecs=registry.codecs),
        {"d": document_input(pa.array([[1], [2]]))},
    )
    registry.load_extension(module)
    response = _execute_physical(request, registry)
    assert response.metrics == {"backend": "test.example", "nodes": ["input:d"]}


def test_physical_explain_shows_shared_inputs_and_scalar_metrics():
    from quail.execution.runner import NodeMetrics, scalar_node_metrics
    from quail.explain import physical_tree
    from quail.physical.base import input_ports

    source = Scan(node_id="input:d", alias="d", n_docs=100)
    first = FirstDocuments(
        node_id="first", alias="d", count=5,
        inputs=input_ports((PortRef(source.node_id, "ids:d"),)))
    second = FirstDocuments(
        node_id="second", alias="d", count=10,
        inputs=input_ports((PortRef(source.node_id, "ids:d"),)))
    project = Project(
        node_id="project", columns=("d.id",),
        inputs=input_ports((PortRef(first.node_id, "ids:d"),
                            PortRef(second.node_id, "ids:d"))))
    graph = PhysicalGraph((source, first, second, project),
                          PortRef(project.node_id, "rows"))
    text = graph.explain()
    assert text.splitlines()[0].split() == ["est.", "rows"]
    assert text.splitlines()[1].startswith("Project: d.id")
    assert text.count("Scan: d [1]") == 1
    assert text.count("Reuse [1]") == 1
    assert "count=5" in text and "count=10" in text
    assert "test.first_documents" in text
    assert "input:d" not in text
    assert "input:d[ids:d]" in graph.explain(verbose=True)

    node = Scan(node_id="input:d", alias="d", n_docs=100)
    graph = PhysicalGraph((node,), PortRef(node.node_id, "ids:d"))
    assert "metrics unavailable" in physical_tree(graph, metrics={})
    measured = {node.node_id: NodeMetrics(output_rows=42, wall_s=1.25,
                                         fresh_tokens=200)}
    text = physical_tree(graph, metrics=measured)
    assert re.search(r"Scan: d\s+100\s+42\s+1\.25 s\s+200$", text,
                     re.MULTILINE)
    assert "est. time" not in text
    assert "fresh_tokens=200" in physical_tree(
        graph, metrics=measured, verbose=True)

    nodes = {"ai_filter:r": NodeResult({}, NodeMetrics(
        fresh_tokens=10,
        extension={"borrowed_prefix_tokens": 64, "answers": [1, 0],
                   "flag": True}))}
    metrics = scalar_node_metrics(nodes)["ai_filter:r"]
    assert metrics["fresh_tokens"] == 10
    assert metrics["borrowed_prefix_tokens"] == 64
    assert "answers" not in metrics and "flag" not in metrics
