"""Native extraction plans, graph execution, metrics, and model reuse on CPU."""

from dataclasses import replace
from types import SimpleNamespace

import pyarrow as pa
import pytest
from fakes import cpu_arena, fake_torch
from test_extract import QUERY_SQL, ExtractClient, _tokens

import quail
from quail.backends.base import BackendExecutionContext
from quail.backends.quail import worker
from quail.cost import budgets
from quail.execution.extract import GenerationResult
from quail.execution.types import PhysicalRequest
from quail.physical import AiExtract, decode_graph
from quail.planner.plan import Refusal


@pytest.fixture
def native(monkeypatch):
    control = SimpleNamespace(
        calls=[], replies={}, generators=[], loads=[], filters=[], released=[],
        state={}, fail=False, sessions=[], keep=lambda prompt: "KEEP" in prompt)

    class Generator(ExtractClient):
        def __init__(self, torch, model, arena, pipeline, spec, chunk_tokens):
            super().__init__()
            self.calls, self.replies = control.calls, control.replies
            self.arena, self.chunk_tokens = arena, chunk_tokens
            self.context_limit = 40_960
            self.metadata = {"model_revision": spec.revision}
            control.generators.append(self)

        def generate_structured(self, *args):
            assert not self.arena.resident_keys()
            if control.fail:
                self.arena.alloc("failed-generation", 1)
                raise RuntimeError("generation failed")
            return [replace(result, cached_tokens=0)
                    for result in super().generate_structured(*args)]

    class CpuGpu(worker.LoadedGpu):
        load_model_s = arena_s = pipeline_s = 0.0

        def __init__(self, backend, context, answer_ids):
            self.spec = context.model
            self.generation = context.query_settings.get("generation", False)
            self.model = SimpleNamespace(quail_generation=self.generation)
            self.torch, self.F = fake_torch(), None
            self.arena, self.pipeline = cpu_arena(512), SimpleNamespace()
            self.async_ans = self.prepared_boot = None
            self.execution = backend.start(context)
            self.execution.bind_loaded_model(
                model=self.model, arena=self.arena, pipeline=self.pipeline)
            control.loads.append(self)

        def bind_query(self, true_ids, false_ids, chunk_tokens, arena_pages=None):
            self.chunk_tokens, self.planned_pages = chunk_tokens, arena_pages
            self.execution.bind_query(
                torch=self.torch, async_answers=None, answer_rows=None,
                chunk_tokens=chunk_tokens)

        def bind_generation(self, generation):
            self.execution.clear_generation()
            self.generation = self.model.quail_generation = generation
            self.prepared_boot = None

        def warm(self):
            return 0.0, None

    def release(state):
        for gpu in state.values():
            control.released.append(gpu)
            gpu.close()
        state.clear()

    def run_filter(torch, arena, pipeline, answers, documents, questions, chunk,
                   *, arena_keys, limit, **kwargs):
        assert limit is None
        record = {"ids": list(arena_keys), "prefixes": [], "questions": questions}
        control.filters.append(record)
        results, fresh = {}, 0
        for index, prefix in enumerate(documents):
            record["prefixes"].append(list(prefix))
            keep = control.keep(bytes(int(t) - 1 for t in prefix).decode())
            row = [keep] * (len(questions) if keep else 1)
            results[index] = row
            fresh += len(prefix) + sum(map(len, questions[:len(row)]))
            if keep:
                arena.alloc(arena_keys[index], 1)
                arena.retain(arena_keys[index], 1)
        return results, [], fresh

    monkeypatch.setattr(worker, "LoadedGpu", CpuGpu)
    monkeypatch.setattr(worker, "release_booted_models", release)
    monkeypatch.setattr(
        "quail.backends.quail.executor.generate.QuailGenerator", Generator)
    monkeypatch.setattr("quail.backends.quail.executor.loop.run_filter", run_filter)

    def session(texts=("KEEP invoice", "DROP invoice"), model="qwen3-4b-fp8", gpus=1):
        value = quail.Session(quail.EngineConfig(
            model=model, device="h100-sxm", backend="quail", gpus=gpus),
            tokenizer=_tokens)
        value.register("documents", quail.DocumentProvider.from_table(pa.table({
            "id": pa.array(range(len(texts)), pa.int64()),
            "body": pa.array(texts, pa.string()),
        }), id_col="id"))
        control.sessions.append(value)
        return value

    def run(session, sql=QUERY_SQL, prepare=False, limit=None):
        query = session.sql(sql)
        plan = query.plan()
        if limit is not None:
            plan = replace(plan, nodes=tuple(
                replace(node, max_output_tokens=limit) if isinstance(node, AiExtract)
                else node for node in plan.nodes))
            query._plan = plan
        if prepare:
            worker.prepare_quail_request(BackendExecutionContext(
                request=PhysicalRequest(plan.to_envelope(session.registry.codecs), {}),
                graph=plan.graph, registry=session.registry, gpu_count=plan.workers,
                runtime_state=control.state))
        request = query._prepare_physical()
        response = session.registry.backend("quail").execute_request(
            BackendExecutionContext(
                request=request, graph=plan.graph, registry=session.registry,
                gpu_count=plan.workers, runtime_state=control.state))
        return query.finish(response)

    control.session, control.run = session, run
    yield control
    release(control.state)
    for value in control.sessions:
        value.close()


@pytest.mark.parametrize("model", ["qwen3-4b-fp8", "qwen3-32b-fp8"])
def test_native_plan_budgets_text_and_codec(native, model):
    value = native.session(model=model)
    query = value.sql(QUERY_SQL)
    plan = query.plan()
    assert plan.settings["generation"] is True
    assert plan.settings["filter_limit"] is None
    node = next(node for node in plan.nodes if isinstance(node, AiExtract))
    assert node.backend == "quail" and node.fields == ("vendor", "total")
    expected = budgets.chunk_budget(value.model, value.device, generation=True)
    assert plan.settings["chunk_tokens"] == expected
    assert tuple(plan.settings["arena_pages"]) == budgets.arena_pages(
        value.model, value.device, expected, generation=True)
    request = query._prepare_physical()
    assert request.inputs["d"].texts.to_pylist() == ["KEEP invoice", "DROP invoice"]
    decoded = decode_graph(plan.to_envelope(value.registry.codecs)["graph"],
                           value.registry.codecs)
    assert decoded.node(node.node_id) == node
    assert "AI.EXTRACT(d.body, 'vendor', 'total')" in query.explain()
    assert any("not included" in remark for remark in plan.remarks)


@pytest.mark.parametrize("texts", [(), (None,), ("", None, "KEEP invoice")])
def test_native_empty_and_null_inputs_preserve_rows(native, texts):
    value = native.session(texts)
    result = native.run(value)
    rows = result.collect().to_pylist()
    assert [row["d.id"] for row in rows] == list(range(len(texts)))
    assert pa.types.is_struct(result.schema.field("extracted").type)
    assert sum(map(lambda call: len(call[0]), native.calls)) == sum(
        text is not None for text in texts)
    for row, text in zip(rows, texts):
        assert row["extracted"] == (
            None if text is None else {
                "response": {"vendor": None, "total": None}, "error": None})


def test_native_filters_nulls_then_extracts_all_survivors_before_limit(native):
    value = native.session((None, "KEEP one", "DROP two", "KEEP three", "KEEP four"))
    sql = QUERY_SQL + (
        " WHERE AI_FILTER(PROMPT('Keep? {0}', d.body))"
        " AND AI_FILTER(PROMPT('Keep again? {0}', d.body)) LIMIT 1")
    result = native.run(value, sql)
    assert result.collect().column("d.id").to_pylist() == [1]
    assert len(native.calls) == 1 and len(native.calls[0][0]) == 3
    assert native.filters[0]["ids"] == [("d", i) for i in (1, 2, 3, 4)]
    record = native.filters[0]
    requested = sum(
        len(prefix) * (1 if index == 1 else 2)
        + sum(map(len, record["questions"][:1 if index == 1 else 2]))
        for index, prefix in enumerate(record["prefixes"]))
    requested += sum(len(_tokens(prompt)) for prompt in native.calls[0][0])
    metrics = result.report["backend_metrics"]
    assert metrics["requests"] == 7 + 3
    assert metrics["prompt_tokens"] == requested
    assert metrics["output_tokens"] == 9
    assert metrics["prompt_tokens"] > result.report["fresh_tokens"]
    assert not native.state[("quail", value.model.name)].arena.resident_keys()


def test_native_multiple_projections_and_bad_row(native):
    value = native.session()
    native.replies["KEEP invoice"] = {"vendor": "Acme", "total": "$7"}
    native.replies["DROP invoice"] = '{"vendor":7,"total":"bad"}'
    sql = QUERY_SQL.replace(
        "FROM documents d",
        ", AI.EXTRACT(d.body, 'total') AS amount FROM documents d")
    rows = native.run(value, sql).collect().to_pylist()
    assert rows[0]["extracted"]["response"] == {"vendor": "Acme", "total": "$7"}
    assert rows[0]["amount"]["response"] == {"total": "$7"}
    assert rows[1]["extracted"]["error"] == "Invalid extraction JSON"
    assert rows[1]["amount"]["error"] == "Invalid extraction JSON"


@pytest.mark.parametrize("model", ["qwen3-4b-fp8", "qwen3-32b-fp8"])
def test_native_generator_and_model_reuse_and_prepared_transitions(native, model):
    first = native.session(model=model)
    native.run(first, prepare=True)
    generator = native.generators[0]
    native.run(native.session(model=model))
    assert native.generators == [generator]
    assert len(native.loads) == 1
    native.run(first, "SELECT d.id FROM documents d WHERE "
               "AI_FILTER(PROMPT('Keep? {0}', d.body))")
    gpu = native.state[("quail", model)]
    assert gpu.execution._generator is None
    assert len(native.loads) == (1 if first.model.tied_head else 2)
    native.run(first)
    assert len(native.generators) == 2
    assert len(native.loads) == (1 if first.model.tied_head else 3)


def test_native_query_failure_invalidates_cached_model_and_all_kv(native):
    value = native.session()
    native.fail = True
    with pytest.raises(RuntimeError, match="generation failed"):
        native.run(value, prepare=True)
    assert native.state == {}
    assert not native.generators[0].arena.resident_keys()
    assert native.loads[0].model is native.loads[0].execution is None
    native.fail = False
    assert len(native.run(value).collect()) == 2
    assert len(native.loads) == 2


def test_native_multiple_gpu_extraction_is_refused(native):
    plan = native.session(gpus=2).sql(QUERY_SQL).plan()
    assert isinstance(plan, Refusal)
    assert "one GPU" in " ".join(plan.reasons)


def test_native_all_null_filter_makes_no_model_requests(native):
    result = native.run(native.session((None, None)), QUERY_SQL + (
        " WHERE AI_FILTER(PROMPT('Keep? {0}', d.body))"))
    assert result.collect().to_pylist() == []
    assert native.calls == []
    assert native.filters[0]["ids"] == []
    assert result.report["backend_metrics"]["requests"] == 0
    assert result.report["backend_metrics"]["prompt_tokens"] == 0


def test_native_filter_context_guard_precedes_forward(native):
    value = native.session()
    native.run(value)
    native.generators[0].context_limit = 1
    with pytest.raises(ValueError, match="filter prompt exceeds"):
        native.run(value, QUERY_SQL + (
            " WHERE AI_FILTER(PROMPT('Keep? {0}', d.body))"))
    assert native.filters == []
    assert native.state == {}


def test_native_output_limit_and_context_errors_reach_sql_metrics(native, monkeypatch):
    value = native.session(("context", "length", "ok"))
    native.run(value)

    def generate(prompts, schema, max_tokens):
        assert len(prompts) == 3 and max_tokens == 7
        return [
            GenerationResult("", "context", 0),
            GenerationResult("partial", "length", 12, output_tokens=7),
            GenerationResult('{"vendor":null,"total":null}', "stop", 10,
                             output_tokens=6),
        ]

    monkeypatch.setattr(native.generators[0], "generate_structured", generate)
    result = native.run(value, limit=7)
    rows = result.collect().to_pylist()
    assert rows[0]["extracted"]["error"] == "Extraction exceeded its context limit"
    assert rows[1]["extracted"]["error"] == "Extraction exceeded its output limit"
    assert rows[2]["extracted"]["error"] is None
    metrics = result.report["backend_metrics"]
    assert metrics["requests"] == 2 and metrics["rejected_requests"] == 1
    assert metrics["prompt_tokens"] == 22 and metrics["output_tokens"] == 13
    assert metrics["extraction_errors"] == 2


def test_native_payload_requires_original_text(native):
    value = native.session()
    query = value.sql(QUERY_SQL)
    request = query._prepare_physical()
    request = replace(request, inputs={
        name: replace(tokens, texts=None) for name, tokens in request.inputs.items()})
    with pytest.raises(ValueError, match="original document text"):
        worker.quail_runtime_payload(request, query.plan().graph)
    client = SimpleNamespace(close=lambda: pytest.fail("released a cached engine"))
    state = {("request-engine", "vllm", value.model.name, True): {"client": client}}
    with pytest.raises(ValueError, match="original document text"):
        worker.execute_quail_request(BackendExecutionContext(
            request=request, graph=query.plan().graph, registry=value.registry,
            gpu_count=1, runtime_state=state))
    assert len(state) == 1
