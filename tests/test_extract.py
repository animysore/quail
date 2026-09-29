"""Field extraction through SQL, the physical graph, and Arrow results."""

import json
import sys
import tomllib
from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

import quail
from quail.backends.base import BackendExecutionContext
from quail.backends.request import release_request_engines
from quail.backends.vllm import (
    DefaultVLLMEngine,
    VLLMClient,
    VLLMEngine,
    register_gigatoken,
)
from quail.execution.extract import (
    GenerationResult,
    extraction_result,
    extraction_schema,
)
from quail.logical import CompileError
from quail.physical import AiExtract, decode_graph
from quail.planner.plan import Refusal
from quail.specs import QWEN3_4B_FP8

VLLM_BACKENDS = ("dumb_vllm", "stock_vllm", "pipelined_vllm")
QUERY_SQL = """
SELECT d.id, AI.EXTRACT(d.body, 'vendor', 'total') AS extracted
FROM documents d
"""


def _tokens(text):
    return [byte + 1 for byte in text.encode("utf-8")]


def _output(text, reason="stop", prompt_tokens=40):
    return SimpleNamespace(
        finished=True, prompt_token_ids=[1] * prompt_tokens, num_cached_tokens=7,
        outputs=[SimpleNamespace(
            text=text, finish_reason=reason, token_ids=[2, 3, 4],
        )],
    )


class ExtractClient:
    accepts_text = True

    def __init__(self):
        self.calls = []
        self.replies = {}
        self.closed = False

    def reset_prefix_cache(self):
        return True

    def close(self):
        self.closed = True

    def generate(self, prompts, sampling_params, use_tqdm=False):
        outputs = []
        for prompt in prompts:
            output = _output("", prompt_tokens=len(_tokens(prompt)))
            output.outputs[0].token_ids = _tokens(
                "TRUE" if "KEEP" in prompt else "FALSE"
            )[:1]
            outputs.append(output)
        return outputs

    def run_filter_chain(self, sampling_params, bodies, questions, read_answer,
                         *, body_texts, question_texts, **kwargs):
        assert len(questions) == 1
        prompts = [body + question_texts[0] for body in body_texts]
        outputs = self.generate(prompts, sampling_params)
        answers = {(index, 1): read_answer(output)
                   for index, output in enumerate(outputs)}
        return {
            "wall": 0.0, "answers": answers,
            "survivors": [index for (index, _), answer in answers.items() if answer],
            "requests": len(prompts),
            "prompt_tokens": sum(len(output.prompt_token_ids) for output in outputs),
            "cached_tokens": sum(output.num_cached_tokens for output in outputs),
            "doc_cap": len(bodies),
        }

    def generate_structured(self, prompts, schema, max_tokens):
        assert not self.closed
        self.calls.append((prompts, schema, max_tokens))
        outputs = []
        for prompt in prompts:
            document = prompt.split("DOCUMENT:\n", 1)[1].split(
                "\n\nExtract these fields", 1
            )[0]
            reply = self.replies.get(document, {})
            if isinstance(reply, str):
                text = reply
            else:
                text = json.dumps({
                    field: reply.get(field) for field in schema["required"]
                })
            outputs.append(GenerationResult(
                text, "stop", len(_tokens(prompt)), 7, 3,
            ))
        return outputs


@pytest.fixture
def client(monkeypatch):
    client = ExtractClient()
    state = {
        "client": client, "sampling_params": object(),
        "capacity": {"kv_cache_size_tokens": 10_000},
    }
    monkeypatch.setattr(
        "quail.backends.vllm.VLLMEngine.boot",
        lambda *args, **kwargs: (state, {"kind": "test", "boot_s": 0.0}),
    )
    return client


@pytest.fixture
def session():
    sessions = []

    def create(texts=("KEEP invoice", "DROP invoice"), backend="dumb_vllm",
               model="qwen3-4b-fp8"):
        value = quail.Session(
            quail.EngineConfig(model=model, device="h100-sxm", backend=backend),
            tokenizer=_tokens,
        )
        value.register("documents", quail.DocumentProvider.from_table(pa.table({
            "id": pa.array(range(len(texts)), type=pa.int64()),
            "body": pa.array(texts, type=pa.string()),
        }), id_col="id"))
        sessions.append(value)
        return value

    yield create
    for value in sessions:
        value.close()


def _run(session, sql=QUERY_SQL, runtime_state=None, **kwargs):
    query = session.sql(sql, **kwargs)
    plan = query.plan()
    response = session.registry.backend(plan.backend).execute_request(
        BackendExecutionContext(
            request=query._prepare_physical(), graph=plan.graph,
            registry=session.registry, gpu_count=1,
            runtime_state={} if runtime_state is None else runtime_state,
        )
    )
    return query.finish(response)


@pytest.mark.parametrize("dialect,function", [
    ("snowflake", "AI_EXTRACT"), ("snowflake", "AI.EXTRACT"), ("bq", "AI.EXTRACT"),
])
@pytest.mark.parametrize("backend", VLLM_BACKENDS)
def test_extract_plan_and_codec(session, dialect, function, backend):
    value = session(backend=backend)
    query = value.sql(QUERY_SQL.replace("AI.EXTRACT", function), dialect=dialect)
    plan = query.plan()
    plan.graph.validate(runtime_keys=set(value.registry.runtimes))
    plan.graph.validate_backend(backend)
    node = next(node for node in plan.nodes if isinstance(node, AiExtract))
    assert node.fields == ("vendor", "total")
    assert node.backend == backend
    assert plan.settings["extraction"] is True
    assert "TRUE or FALSE" not in node.tail_text
    envelope = plan.to_envelope(value.registry.codecs)
    decoded = decode_graph(json.loads(json.dumps(envelope))["graph"],
                           value.registry.codecs)
    assert decoded.node(node.node_id) == node
    assert "AI.EXTRACT(d.body, 'vendor', 'total') AS extracted" in query.explain()


@pytest.mark.parametrize("backend", VLLM_BACKENDS)
def test_extract_results_missing_fields_and_null_documents(session, client, backend):
    value = session(("KEEP invoice", "no information", None), backend=backend)
    client.replies["KEEP invoice"] = {"vendor": 'Acme "North" — 東京', "total": "$12"}
    result = _run(value)
    rows = result.collect().to_pylist()
    assert rows == [
        {"d.id": 0, "extracted": {
            "response": {"vendor": 'Acme "North" — 東京', "total": "$12"},
            "error": None,
        }},
        {"d.id": 1, "extracted": {
            "response": {"vendor": None, "total": None}, "error": None,
        }},
        {"d.id": 2, "extracted": None},
    ]
    assert sum(len(call[0]) for call in client.calls) == 2
    dtype = result.schema.field("extracted").type
    assert dtype.field("response").type.field("total").type == pa.string()
    metrics = next(
        metrics for node, metrics in result.node_metrics.items()
        if node.startswith("extract:")
    )
    assert metrics.evaluated_documents == 2
    assert metrics.extension["output_tokens"] == 6
    assert metrics.cached_tokens == 14
    assert metrics.fresh_tokens + metrics.cached_tokens == \
        metrics.extension["prompt_tokens"]
    assert result.report["backend_metrics"]["requests"] == 2
    assert result.report["backend_metrics"]["output_tokens"] == 6
    assert result.report["backend_metrics"]["extraction_errors"] == 0
    assert result.execute_stream(batch_rows=1).read_all().equals(result.collect())


def test_extract_runs_after_filters_and_carries_prior_columns(session, client):
    value = session()
    client.replies["KEEP invoice"] = {"vendor": "Acme", "total": "42"}
    sql = QUERY_SQL.replace(
        "FROM documents d",
        ", AI.EXTRACT(d.body, 'total') AS amount FROM documents d "
        "WHERE AI_FILTER(PROMPT('Keep this document? {0}', d.body))",
    )
    result = _run(value, sql).collect().to_pylist()
    assert len(result) == 1
    assert result[0]["d.id"] == 0
    assert result[0]["extracted"]["response"]["vendor"] == "Acme"
    assert result[0]["amount"]["response"] == {"total": "42"}
    assert len(client.calls) == 2
    assert all(len(prompts) == 1 and "DROP" not in prompts[0]
               for prompts, _, _ in client.calls)


@pytest.mark.parametrize("backend", VLLM_BACKENDS)
@pytest.mark.parametrize("texts", [("KEEP invoice", None, "DROP invoice"), (None,)])
def test_extract_filters_skip_null_documents(session, client, backend, texts):
    value = session(texts, backend=backend)
    result = _run(value, QUERY_SQL + (
        " WHERE AI_FILTER(PROMPT('Keep this document? {0}', d.body))"
    ))
    assert result.collect().column("d.id").to_pylist() == (
        [0] if len(texts) == 3 else []
    )
    assert result.report["backend_metrics"]["requests"] == (
        3 if len(texts) == 3 else 0
    )
    assert sum(len(prompts) for prompts, _, _ in client.calls) == (
        1 if len(texts) == 3 else 0
    )


@pytest.mark.parametrize("backend", (*VLLM_BACKENDS, "quail", "pipelined_sglang"))
@pytest.mark.parametrize("null_position", [0, 30])
def test_null_tokenization_is_scoped_to_extraction(session, backend, null_position):
    texts = ["KEEP invoice"] * 31
    texts[null_position] = None
    value = session(texts, backend=backend)
    sql = "SELECT d.id FROM documents d WHERE AI_FILTER(PROMPT('Keep? {0}', d.body))"
    with pytest.raises(TypeError, match="NULL document"):
        value.sql(sql).token_inputs()
    nullable = value.tokenize("documents", "body", allow_null=True)
    assert list(nullable.tokens[null_position]) == []
    # A later filter-only query must not reuse NULL-tolerant tokenization silently.
    with pytest.raises(TypeError, match="NULL document"):
        value.sql(sql).token_inputs()


@pytest.mark.parametrize("texts", [(), (None,), ("DROP invoice",)])
def test_extract_empty_null_and_filtered_inputs(session, client, texts):
    value = session(texts)
    sql = QUERY_SQL
    if texts == ("DROP invoice",):
        sql += " WHERE AI_FILTER(PROMPT('Keep? {0}', d.body))"
    result = _run(value, sql).collect()
    assert len(result) == (1 if texts == (None,) else 0)
    assert not client.calls
    assert pa.types.is_struct(result.schema.field("extracted").type)


def test_extract_batches_and_limits_preserve_input_order(session, client, monkeypatch):
    monkeypatch.setattr("quail.execution.extract.EXTRACT_BATCH_ROWS", 2)
    value = session(tuple(f"document {i}" for i in range(5)))
    for i in range(5):
        client.replies[f"document {i}"] = {"vendor": str(i)}
    result = _run(value, QUERY_SQL + " LIMIT 3").collect()
    assert result.column("d.id").to_pylist() == [0, 1, 2]
    assert [row["response"]["vendor"] for row in
            result.column("extracted").to_pylist()] == ["0", "1", "2"]
    assert [len(prompts) for prompts, _, _ in client.calls] == [2, 2, 1]


@pytest.mark.parametrize("text,reason,error", [
    ('{"vendor":null,"total":"3"}', "stop", None),
    ('{"vendor":"Acme","total":3}', "stop", "Invalid extraction JSON"),
    ('{"vendor":"Acme"}', "stop", "Invalid extraction JSON"),
    ('{"vendor":"Acme","total":"3","extra":"x"}', "stop", "Invalid extraction JSON"),
    ('{"vendor":"A","vendor":"B","total":null}', "stop", "Invalid extraction JSON"),
    ('[{"vendor":"A","total":null}]', "stop", "Invalid extraction JSON"),
    ('```json\\n{}\\n```', "stop", "Invalid extraction JSON"),
    ('{"vendor":"Acme","total":', "length", "Extraction exceeded its output limit"),
    ('{"vendor":null,"total":null}', "abort", "Extraction generation did not finish"),
    ("", "incomplete", "Incomplete extraction response"),
    ("", "context", "Extraction exceeded its context limit"),
])
def test_extract_validation(text, reason, error):
    result = extraction_result(
        GenerationResult(text, reason, 40), ("vendor", "total"),
    )
    assert result["error"] == error
    assert (result["response"] is None) == (error is not None)


def test_extract_bad_row_keeps_other_results(session, client):
    value = session()
    client.replies["KEEP invoice"] = {"vendor": "Acme"}
    client.replies["DROP invoice"] = '{"vendor":"Acme","total":7}'
    result = _run(value).collect().to_pylist()
    assert result[0]["extracted"]["response"]["vendor"] == "Acme"
    assert result[1]["extracted"] == {
        "response": None, "error": "Invalid extraction JSON",
    }


def test_context_refusal_is_not_counted_as_evaluated_work(session, client):
    client.generate_structured = lambda *args: [
        GenerationResult("", "context", 0),
        GenerationResult('{"vendor":"Acme","total":null}', "stop", 40, 7, 3),
    ]
    result = _run(session())
    rows = result.collect().to_pylist()
    assert rows[0]["extracted"]["error"] == "Extraction exceeded its context limit"
    assert rows[1]["extracted"]["response"]["vendor"] == "Acme"
    metrics = result.node_metrics["extract:1"]
    assert metrics.evaluated_documents == 1
    assert metrics.fresh_tokens == 33
    assert metrics.cached_tokens == 7
    assert metrics.extension["rejected_requests"] == 1
    assert result.report["backend_metrics"]["rejected_requests"] == 1


@pytest.mark.parametrize("expression", [
    "AI.EXTRACT(d.body)", "AI.EXTRACT(d.body, '')",
    "AI.EXTRACT(d.body, 'vendor', 'vendor')",
    "AI.EXTRACT(d.body, d.id)", "AI.EXTRACT(d.body, 1)",
    "AI.EXTRACT(d.id, 'vendor')", "AI.EXTRACT('document', 'vendor')",
    "COALESCE(AI.EXTRACT(d.body, 'vendor'), 'x')",
])
def test_extract_rejects_invalid_calls(session, expression):
    with pytest.raises(CompileError, match="AI.EXTRACT"):
        session().sql(f"SELECT {expression} AS extracted FROM documents d")


@pytest.mark.parametrize("sql", [
    "SELECT AI.EXTRACT(d.body, 'vendor') FROM documents d",
    "SELECT AI.EXTRACT(d.body, 'vendor') AS d FROM documents d",
    "SELECT AI.EXTRACT(d.body, 'vendor') AS x, "
    "AI.EXTRACT(d.body, 'vendor') AS y FROM documents d",
    "SELECT AI.EXTRACT(d.body, 'vendor') AS x, "
    "AI.SCORE(PROMPT('Relevant? {0}', d.body)) AS y FROM documents d",
    "SELECT AI.EXTRACT(d.body, 'vendor') AS x FROM documents d "
    "CROSS JOIN documents e",
])
def test_extract_rejects_unsupported_query_shapes(session, sql):
    with pytest.raises(CompileError, match=r"AI\.(EXTRACT|SCORE)"):
        session().sql(sql)


@pytest.mark.parametrize("backend,model", [
    ("quail", "qwen3-4b-fp8"),
    ("pipelined_sglang", "qwen3-4b-fp8"),
    ("dumb_vllm", "diffusion-gemma-26b-a4b-fp8"),
])
def test_extract_unsupported_backend_is_explicit(session, backend, model):
    plan = session(backend=backend, model=model).sql(QUERY_SQL).plan()
    assert isinstance(plan, Refusal)
    assert "AI.EXTRACT" in " ".join(plan.reasons)


def test_extract_reranker_is_rejected_by_session(session):
    from quail.execution.session import RefusalError

    with pytest.raises(RefusalError, match="generative models only"):
        session(model="qwen3-reranker-0.6b-bf16")


@pytest.fixture
def vllm_client(monkeypatch):
    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=Params))
    monkeypatch.setitem(
        sys.modules, "vllm.sampling_params",
        SimpleNamespace(StructuredOutputsParams=Params),
    )
    calls = []
    outputs = []

    def encode(text, *, add_special_tokens):
        assert add_special_tokens is False
        return _tokens(text)

    def generate(*args, **kwargs):
        calls.append((args, kwargs))
        if outputs:
            return outputs
        return [_output('{"vendor":null,"total":null}',
                        prompt_tokens=len(prompt["prompt_token_ids"]))
                for prompt in args[0]]

    llm = SimpleNamespace(
        generate=generate, get_tokenizer=lambda: SimpleNamespace(encode=encode),
    )
    client = VLLMClient(
        llm, {"max_model_len": 1024, "kv_cache_size_tokens": 1024},
    )
    return client, calls, outputs


def test_vllm_uses_independent_structured_sampling_params(vllm_client):
    client, calls, _ = vllm_client
    schema = extraction_schema(("vendor", "total"))
    results = client.generate_structured(["invoice"], schema, 512)
    args, kwargs = calls[0]
    assert args[0] == [{"prompt_token_ids": _tokens("invoice")}]
    assert results == [GenerationResult(
        '{"vendor":null,"total":null}', "stop", len(_tokens("invoice")), 7, 3,
    )]
    params = args[1]
    assert params.temperature == 0.0
    assert params.max_tokens == 512
    assert params.structured_outputs.json == {
        "type": "object", "properties": {
            "vendor": {"type": ["string", "null"]},
            "total": {"type": ["string", "null"]},
        },
        "required": ["vendor", "total"], "additionalProperties": False,
    }
    assert not hasattr(params, "allowed_token_ids")
    assert kwargs == {"use_tqdm": False}


@pytest.mark.parametrize("reason,finished,count", [
    ("stop", False, 1), ("stop", True, 0), ("stop", True, 2),
    ("stop", True, 1), ("length", True, 1),
])
def test_vllm_adapts_completion_state(vllm_client, reason, finished, count):
    client, _, outputs = vllm_client
    raw = _output('{"vendor":null,"total":null}', reason)
    raw.finished = finished
    raw.outputs *= count
    outputs.append(raw)
    result, = client.generate_structured(["invoice"], extraction_schema(("vendor",)), 3)
    assert result.finish_reason == (
        reason if finished and count == 1 else "incomplete"
    )
    assert result.prompt_tokens == 40
    assert result.cached_tokens == 7
    assert result.output_tokens == 3 * count


def test_vllm_context_refusals_preserve_valid_rows(vllm_client):
    client, calls, _ = vllm_client
    results = client.generate_structured(
        ["x" * 513, "invoice", "y" * 512], extraction_schema(("vendor",)), 512,
    )
    assert [result.finish_reason for result in results] == ["context", "stop", "stop"]
    assert results[0].prompt_tokens == results[0].output_tokens == 0
    assert calls[0][0][0] == [
        {"prompt_token_ids": _tokens("invoice")},
        {"prompt_token_ids": _tokens("y" * 512)},
    ]
    calls.clear()
    client.capacity["kv_cache_size_tokens"] = 100
    rejected = client.generate_structured(
        ["invoice"], extraction_schema(("vendor",)), 512,
    )
    assert rejected[0].finish_reason == "context"
    assert not calls


def test_vllm_extraction_rejects_bad_limits_and_response_counts(vllm_client):
    client, calls, outputs = vllm_client
    schema = extraction_schema(("vendor",))
    with pytest.raises(ValueError, match="positive"):
        client.generate_structured(["invoice"], schema, 0)
    assert client.generate_structured([], schema, 512) == []
    assert not calls
    outputs.extend([_output("{}"), _output("{}")])
    with pytest.raises(RuntimeError, match="wrong number"):
        client.generate_structured(["invoice"], schema, 512)


@pytest.mark.parametrize("engine", [VLLMEngine(), DefaultVLLMEngine()])
def test_extraction_engine_settings_preserve_filter_configuration(engine):
    normal = engine.llm_kwargs(QWEN3_4B_FP8)
    extract = engine.llm_kwargs(QWEN3_4B_FP8, extraction=True)
    assert extract == normal | {
        "tokenizer_mode": "auto",
        "revision": QWEN3_4B_FP8.revision,
        "tokenizer_revision": QWEN3_4B_FP8.revision,
        "structured_outputs_config": {
            "backend": "xgrammar", "disable_any_whitespace": False,
        },
    }
    assert engine.llm_kwargs(QWEN3_4B_FP8) == normal


def test_request_modes_reuse_compatible_engines_and_close_others(session, monkeypatch):
    events = []
    clients = []

    def boot(engine, spec, allowed_ids, *, extraction=False):
        client = ExtractClient()
        clients.append(client)
        events.append(("boot", engine.kind, extraction))

        def close():
            client.closed = True
            events.append(("close", engine.kind, extraction))

        client.close = close
        return {
            "client": client, "sampling_params": object(),
            "capacity": {"kv_cache_size_tokens": 10_000},
        }, {"kind": "test", "boot_s": 0.0}

    monkeypatch.setattr(VLLMEngine, "boot", boot)
    state = {}
    for backend, extract in [
        ("stock_vllm", False), ("stock_vllm", True),
        ("pipelined_vllm", True), ("stock_vllm", False),
        ("dumb_vllm", True),
    ]:
        _run(session(backend=backend),
             QUERY_SQL if extract else (
                 "SELECT d.id FROM documents d "
                 "WHERE AI_FILTER(PROMPT('Keep? {0}', d.body))"
             ),
             runtime_state=state)
        assert len(state) == 1
    assert events == [
        ("boot", "vllm", False), ("close", "vllm", False),
        ("boot", "vllm", True), ("close", "vllm", True),
        ("boot", "vllm", False), ("close", "vllm", False),
        ("boot", "dumb_vllm", True),
    ]
    release_request_engines(state)
    assert not state
    assert all(client.closed for client in clients)


def test_request_failure_discards_engine(session, client, monkeypatch):
    state = {}

    def fail(*args, **kwargs):
        raise RuntimeError("engine failed")

    monkeypatch.setattr("quail.backends.request.execute_request_graph", fail)
    with pytest.raises(RuntimeError, match="engine failed"):
        _run(session(), runtime_state=state)
    assert client.closed
    assert not state


@pytest.mark.parametrize("stage", ["sampling", "capacity", "grammar", "warmup"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_vllm_boot_failure_closes_engine(monkeypatch, stage, cleanup_fails):
    closed = []
    config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        model_config=SimpleNamespace(
            tokenizer_mode="auto", revision="test", tokenizer_revision="test",
        ),
        structured_outputs_config=SimpleNamespace(
            backend="xgrammar", disable_any_whitespace=False,
        ),
    )

    def fail():
        raise RuntimeError(f"failed {stage}")

    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(vllm_config=config),
        generate=lambda *a, **k: fail() if stage == "warmup" else [],
    )
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        LLM=lambda **k: llm,
        SamplingParams=lambda **k: (
            fail() if stage == "sampling" else SimpleNamespace(logprobs=None)
        ),
    ))
    monkeypatch.setattr("quail.backends.vllm.register_gigatoken", lambda: None)
    monkeypatch.setattr("quail.backends.vllm._capacity",
                        lambda llm: fail() if stage == "capacity" else {})
    monkeypatch.setattr("importlib.metadata.version",
                        lambda name: fail() if stage == "grammar" else "test")

    def close(client):
        closed.append(client.llm)
        client.llm = None
        if cleanup_fails:
            raise RuntimeError("failed cleanup")

    monkeypatch.setattr(VLLMClient, "close", close)
    with pytest.raises(RuntimeError, match=f"failed {stage}") as error:
        VLLMEngine().boot(QWEN3_4B_FP8, [1, 2], extraction=True)
    assert closed == [llm]
    if cleanup_fails:
        assert "failed cleanup" in " ".join(error.value.__notes__)


def test_vllm_cleanup_runs_even_when_shutdown_raises(monkeypatch):
    calls = []

    def shutdown():
        calls.append("shutdown")
        raise RuntimeError("shutdown failed")

    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", SimpleNamespace(
        cleanup_dist_env_and_memory=lambda: calls.append("cleanup"),
    ))
    client = VLLMClient(SimpleNamespace(llm_engine=SimpleNamespace(
        engine_core=SimpleNamespace(shutdown=shutdown),
    )), {})
    with pytest.raises(RuntimeError, match="shutdown failed"):
        client.close()
    client.close()
    assert calls == ["shutdown", "cleanup"]
    assert client.llm is None


def test_declared_gigatoken_plugin_resolves():
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text())
    plugins = project["project"]["entry-points"]["vllm.general_plugins"]
    value = plugins["quail_gigatoken"]
    plugin = EntryPoint(
        name="quail_gigatoken", group="vllm.general_plugins", value=value,
    )
    assert plugin.load() is register_gigatoken


@pytest.mark.parametrize("installed,allowed,error", [
    (False, None, "Reinstall quail-engine"),
    (True, [], "VLLM_PLUGINS"),
    (True, ["another_plugin"], "VLLM_PLUGINS"),
    (True, None, None),
    (True, ["quail_gigatoken"], None),
])
def test_tuned_boot_checks_plugin_before_allocating(monkeypatch, installed,
                                                    allowed, error):
    plugins = [EntryPoint(
        name="quail_gigatoken", group="vllm.general_plugins",
        value="quail.backends.vllm:register_gigatoken",
    )] if installed else []
    monkeypatch.setattr("importlib.metadata.entry_points", lambda **kwargs: plugins)
    calls = []

    def load(**kwargs):
        calls.append(kwargs)
        raise ValueError("allocation reached")

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        envs=SimpleNamespace(VLLM_PLUGINS=allowed),
        LLM=load, SamplingParams=object,
    ))
    monkeypatch.setattr("quail.backends.vllm.register_gigatoken", lambda: None)
    if error:
        with pytest.raises(RuntimeError, match=error):
            VLLMEngine().boot(QWEN3_4B_FP8, [1, 2])
        assert not calls
    else:
        with pytest.raises(ValueError, match="allocation reached"):
            VLLMEngine().boot(QWEN3_4B_FP8, [1, 2])
        assert len(calls) == 1
