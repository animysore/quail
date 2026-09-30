"""Extraction benchmark translation, result IDs, and metrics on CPU."""

import sys
from types import SimpleNamespace

import pyarrow as pa
import pytest
import test_extract as request_fakes
import test_native_extract as native_fakes
from test_extract import VLLM_BACKENDS, _tokens

import quail
import quail_b
from quail.backends.base import BackendExecutionContext
from quail.bench import substrait
from quail.bench.quailb import (
    build_query,
    canonical_templates,
    run_output,
    run_query,
    unsupported_queries,
)
from quail.execution.execute import execute_query
from quail.execution.session import Query
from quail.physical import AiExtract
from quail.planner.plan import Refusal
from quail_b.queries import QuerySpec
from quail_b.queries import queries as benchmark_queries
from quail_b.rendering import render_extract_prompt
from quail_b.run import _score
from quail_b.scoring import expected_rows

SAMPLES = ("EXTRACT-1", "EXTRACT-2", "EXTRACT-3")
client = request_fakes.client
native = native_fakes.native


@pytest.fixture
def suite():
    return quail_b.load_benchmark(SAMPLES, scale_factor=1.0)


@pytest.mark.parametrize("backend", ("quail", *VLLM_BACKENDS))
def test_sample_plans_keep_exact_prompts_fields_and_nullable_inputs(suite, backend):
    with quail.Session(quail.EngineConfig(
            model="qwen3-4b-fp8", device="h100-sxm", backend=backend),
            tokenizer=_tokens) as value:
        for name, table in suite.tables.items():
            value.register(name, quail.DocumentProvider.from_table(table, id_col="id"))
        for spec in suite.queries:
            query = build_query(value, spec)
            plan = query.plan()
            assert not isinstance(plan, Refusal)
            nodes = [node for node in plan.nodes if isinstance(node, AiExtract)]
            assert len(nodes) == len(spec._info.extracts)
            for node, operator in zip(nodes, spec._info.extracts):
                assert node.fields == operator.fields
                assert node.name == operator.output
                full = node.preamble_text + "document" + node.tail_text
                assert full == (
                    value.model.turn[0] + render_extract_prompt("document", node.fields)
                    + value.model.turn[1])


def _reference_replies(suite):
    replies = {}
    for index in (0, 2):
        spec = suite.queries[index]
        relation, = spec._info.relations
        documents = dict(zip(
            suite.tables[relation.table]["id"].to_pylist(),
            suite.tables[relation.table]["body"].to_pylist()))
        rows = expected_rows(spec, suite.ground_truth, suite.tables).to_pylist()
        for row in rows:
            if row["extracted"] is not None:
                replies[documents[row[relation.alias]]] = row["extracted"]["response"]
    return replies


@pytest.mark.parametrize("backend", ("quail", *VLLM_BACKENDS))
def test_real_adapter_with_cpu_backends_preserves_ids_and_scoring(
        suite, client, native, backend, monkeypatch):
    replies = _reference_replies(suite)
    client.replies.update(replies)
    native.replies.update(replies)
    client.filter_answer = native.keep = lambda prompt: "Invoice I-" in prompt
    state = native.state if backend == "quail" else {}

    def run(query):
        plan = query.plan()

        def execute(request):
            return query.session.registry.backend(plan.backend).execute_request(
                BackendExecutionContext(
                    request=request, graph=plan.graph, registry=query.session.registry,
                    gpu_count=1, runtime_state=state))

        return execute_query(query, physical_executor=execute)

    monkeypatch.setattr(Query, "run", run)
    with quail.Session(quail.EngineConfig(
            model="qwen3-4b-fp8", device="h100-sxm", backend=backend),
            tokenizer=_tokens) as value:
        for spec in suite.queries:
            call_start = len(native.calls if backend == "quail" else client.calls)
            filter_start = len(
                native.filters if backend == "quail" else client.filter_prompts)
            output = run_query(value, spec, suite.tables)
            calls = (native.calls if backend == "quail" else client.calls)[call_start:]
            prompts = [prompt for batch, _, _ in calls for prompt in batch]
            requested = sum(len(_tokens(prompt)) for prompt in prompts)
            if backend == "quail":
                requested += sum(
                    len(prefix) + len(question)
                    for step in native.filters[filter_start:]
                    for prefix in step["prefixes"] for question in step["questions"])
            else:
                requested += sum(len(_tokens(prompt))
                                 for prompt in client.filter_prompts[filter_start:])
            expected = expected_rows(spec, suite.ground_truth, suite.tables)
            assert output.rows.cast(expected.schema).equals(expected)
            assert output.extract_answers["extract-1"].cast(expected.schema).equals(
                expected)
            assert output.prompt_pieces is None
            assert output.measurements["input_tokens"] == requested
            assert output.measurements["output_tokens"] == 3 * len(prompts)
            assert output.runtime_s == sum(
                metric["wall_s"] for name, metric in
                output.measurements["node_metrics"].items()
                if name == "request-model"
                or name.startswith(("extract:", "ai_filter:")))
            metrics = _score(spec, output, suite, 1, 3.9492)
            assert metrics["accuracy"]["extraction_accuracy"]["accuracy"] == 1
            assert metrics["accuracy"]["output_accuracy"]["exact_match"]
            assert metrics["input_tokens_per_second"] == (
                output.measurements["input_tokens"] / output.runtime_s)
            if spec.id == "EXTRACT-2":
                assert output.filter_answers["filter-1"]["d"].to_pylist() == (
                    suite.tables["extract_invoices"]["id"].to_pylist()[:-1])


def test_unsupported_classifications_are_identified_before_execution():
    specs = benchmark_queries()
    skipped = unsupported_queries(specs)
    assert set(skipped) == {
        name for name, spec in specs.items() if spec._info.classifies}
    assert len(skipped) == 10
    assert all("AI.CLASSIFY is not implemented" in reason
               for reason in skipped.values())
    assert not unsupported_queries(SAMPLES)
    truth = SimpleNamespace(predicates={
        "classify": SimpleNamespace(predicate={"kind": "classify"})})
    assert canonical_templates(truth) == {}


def test_extraction_translation_rejects_a_changed_type_or_nested_name(suite):
    for change in ("type", "names"):
        plan = suite.queries[0].plan
        root = plan.relations[0].root
        if change == "type":
            root.input.project.input.project.expressions[0].scalar_function \
                .output_type.Clear()
        else:
            root.names[3] = "changed"
        with pytest.raises(ValueError):
            substrait.read_plan(plan)


def test_extraction_quoted_field_names_survive_the_sql_frontend(suite):
    plan = suite.queries[0].plan
    root = plan.relations[0].root
    project = root.input.project.input.project
    call = project.expressions[0].scalar_function
    field = 'vendor\'s "name" {0}'
    call.arguments[1].value.literal.list.values[0].string = field
    project.common.hint.output_names[4] = field
    root.names[3] = field
    spec = QuerySpec.from_plan("QUOTED", "quoted fields", plan)
    with quail.Session(quail.EngineConfig(
            model="qwen3-4b-fp8", device="h100-sxm"), tokenizer=_tokens) as value:
        value.register("extract_invoices", quail.DocumentProvider.from_table(
            suite.tables["extract_invoices"], id_col="id"))
        query = build_query(value, spec)
        node, = [node for node in query.plan().nodes if isinstance(node, AiExtract)]
        assert node.fields == (field, "total")


def test_filter_answer_indices_are_translated_independently_of_output_ids(suite):
    spec = suite.queries[1]
    plan = substrait.read_plan(spec.plan)
    rows = expected_rows(spec, suite.ground_truth, suite.tables)
    result = SimpleNamespace(
        answer_tables={
            "filters": {("d", 0): pa.table({
                "d": [4, 2], "answer": [True, False]})}, "joins": {}},
        report={"wall_s": 1.0}, collect=lambda: rows)
    output = run_output(result, plan, suite.tables)
    assert output.filter_answers["filter-1"].to_pylist() == [
        {"d": "I-105", "answer": True}, {"d": "I-103", "answer": False}]
    assert output.rows.equals(rows)


@pytest.mark.parametrize("backend", ["quail", "dumb_vllm"])
def test_extraction_runtime_excludes_startup_and_non_model_nodes(
        suite, backend, monkeypatch):
    from quail.bench import quailb

    spec = suite.queries[0]
    rows = expected_rows(spec, suite.ground_truth, suite.tables)
    model_node = "ai_filter:d" if backend == "quail" else "request-model"
    result = SimpleNamespace(
        answer_tables={"filters": {}, "joins": {}},
        collect=lambda: rows,
        report={
            "wall_s": 999.0, "boot_s": 100.0, "finish_s": 20.0,
            "node_metrics": {
                model_node: {"wall_s": 3.0}, "extract:extracted": {"wall_s": 5.0},
                "scan:d": {"wall_s": 10.0}, "project:out": {"wall_s": 20.0},
            },
            "backend_metrics": {"prompt_tokens": 538, "output_tokens": 115},
        })
    monkeypatch.setattr(quailb, "_build", lambda *args: SimpleNamespace(
        run=lambda: result))
    session = SimpleNamespace(
        catalog=suite.tables, config=SimpleNamespace(backend=backend))
    output = run_query(session, spec, suite.tables)
    assert output.runtime_s == output.measurements["wall_s"] == 8.0
    assert output.measurements["boot_s"] == 100.0


@pytest.mark.parametrize("fail", [False, True])
def test_cli_releases_request_engine_after_its_last_query(monkeypatch, tmp_path, fail):
    from quail.bench import quailb
    from quail.execution import execute

    class Client:
        closed = False

        def close(self):
            self.closed = True

    cached = Client()
    monkeypatch.setattr(sys, "argv", [
        "quailb", "--model", "qwen3-4b-fp8", "--device", "h100-sxm",
        "--backend", "dumb_vllm", "--sf", "1.0", "--only", "EXTRACT-1",
        "--output-dir", str(tmp_path / "results"),
    ])
    monkeypatch.setattr(execute, "_BACKEND_STATE", {})

    def run_suite(*args, **kwargs):
        execute._BACKEND_STATE["request-engine", "test"] = {"client": cached}
        if fail:
            raise RuntimeError("query failed")
        return {"status": "complete"}

    monkeypatch.setattr(quailb, "run_suite", run_suite)
    if fail:
        with pytest.raises(RuntimeError, match="query failed"):
            quailb.main()
    else:
        quailb.main()
    assert cached.closed
    assert not execute._BACKEND_STATE
