"""CPU tests for vLLM and SGLang backend interfaces."""

import asyncio
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

import quail
from quail.backends import pipelined_sglang_backend, stock_vllm_backend
from quail.backends.base import BackendExecutionContext, GpuContext
from quail.backends.request import RequestModelExecution
from quail.backends.sglang import SGLangClient
from quail.backends.vllm import VLLMEngine, sampling_kwargs
from quail.builtins import built_in_registry
from quail.physical import (
    RequestExecution,
    RequestFilterSpec,
    RequestJoinSpec,
    decode_graph,
)
from quail.planner.plan import EngineConfig
from quail.specs import DIFFUSION_GEMMA_26B_FP8, H100_SXM, QWEN3_4B_FP8

CAPACITY = {"kv_cache_size_tokens": 1_000, "block_size": 1, "max_num_seqs": 16}


def _tokens(text):
    return [byte + 1 for byte in text.encode("utf-8")]


def _session(backend, **tables):
    session = quail.Session(
        EngineConfig(gpus=1, model="qwen3-4b-fp8", backend=backend,
                     device="h100-sxm"),
        tokenizer=_tokens,
    )
    for name, columns in tables.items():
        session.register(name, quail.DocumentProvider.from_table(
            pa.table(columns), id_col="id"))
    return session


def _run_stock_vllm(session, query):
    plan = query.plan()
    response = session.registry.backend("stock_vllm").execute_request(
        BackendExecutionContext(
            request=query._prepare_physical(), graph=plan.graph,
            registry=session.registry, gpu_count=1, runtime_state={}))
    return query.finish(response)


DOCS = {"id": ["a", "b"], "body": ["one", "two"]}


def _filter_query(session):
    return (session.docs("docs").alias("d")
            .ai_filter(quail.prompt("useful {0}", quail.col("d.body")))
            .select("d.id"))


def test_request_backends_plan_validate_and_execute(monkeypatch):
    registry = built_in_registry()
    assert set(registry.backends) == {
        "quail", "stock_vllm", "pipelined_vllm", "dumb_vllm", "pipelined_sglang",
    }

    for backend_name in ("stock_vllm", "pipelined_vllm", "pipelined_sglang"):
        session = _session(backend_name, docs=DOCS)
        plan = _filter_query(session).plan()
        request_node = next(
            node for node in plan.nodes if isinstance(node, RequestExecution))
        assert plan.backend == request_node.backend == backend_name
        assert request_node.filters[0].alias == "d"
        plan.graph.validate_backend(backend_name)
        envelope = plan.to_envelope(registry.codecs)
        decoded = decode_graph(envelope["graph"], registry.codecs)
        assert decoded.node("request-model") == request_node
        session.close()

    for backend in (stock_vllm_backend(), pipelined_sglang_backend()):
        support = backend.supports(QWEN3_4B_FP8, H100_SXM, 2)
        assert not support.supported
        assert "one model copy on one GPU" in support.reason

    boot = {"client": _Client(), "sampling_params": object(), "capacity": CAPACITY}
    monkeypatch.setattr(
        "quail.backends.vllm.VLLMEngine.boot",
        lambda self, spec, allowed_ids: (boot, {"kind": "cold", "boot_s": 0.1}))
    session = _session(
        "stock_vllm",
        docs={**DOCS, "key": ["k1", "k2"]},
        notes={"id": ["x", "y", "z"], "text": ["p", "q", "r"],
               "key": ["k2", "k1", "k3"]},
    )
    result = _run_stock_vllm(session, _filter_query(session))
    assert result.report["backend"] == "stock_vllm"
    assert result.report["backend_metrics"]["requests"] == 2
    assert result.count() == 0
    query = (
        session.docs("docs").alias("d")
        .join(session.docs("notes").alias("n"),
              on=[quail.col("d.key") == quail.col("n.key")])
        .ai_filter(quail.prompt("{0} matches {1}", quail.col("d.body"),
                                quail.col("n.text")))
        .select("d.id", "n.id")
    )
    result = _run_stock_vllm(session, query)
    assert result.report["backend_metrics"]["requests"] == 2
    joins = result.answer_tables["joins"][0].to_pydict()
    assert sorted(zip(*joins.values())) == [(0, 1, False), (1, 0, False)]
    session.close()


class _Client:
    def generate(self, prompts, sampling_params, use_tqdm=False):
        assert all(isinstance(token, int)
                   for prompt in prompts for token in prompt["prompt_token_ids"])
        return [SimpleNamespace(
            prompt_token_ids=prompt["prompt_token_ids"], num_cached_tokens=0,
            outputs=[SimpleNamespace(
                token_ids=[1 if 10 in prompt["prompt_token_ids"] else 2])],
        ) for prompt in prompts]

    def reset_prefix_cache(self):
        return True


class _ScoredClient(_Client):
    """Samples the wrong token; only the logprobs hold the answer."""

    def generate(self, prompts, sampling_params, use_tqdm=False):
        outputs = super().generate(prompts, sampling_params, use_tqdm)
        for output in outputs:
            expected = output.outputs[0].token_ids[0] == 1
            output.outputs[0].token_ids = [int(not expected)]
            output.outputs[0].logprobs = [{
                1: SimpleNamespace(logprob=-1 if expected else -2),
                0: SimpleNamespace(logprob=-2 if expected else -1),
            }]
        return outputs


def _execution(documents, *, model=QWEN3_4B_FP8, **settings):
    return RequestModelExecution(GpuContext(
        gpu_index=0, gpu_count=1, model=model, device=H100_SXM, query_settings={
            "client": _Client(), "sampling_params": object(),
            "documents": documents, "true_ids": [1], "capacity": CAPACITY,
            "filter_submission": "operator-at-a-time",
            "join_submission": "anchor-major",
            **settings,
        },
    ))


FILTER_NODE = RequestExecution(
    node_id="request-model", backend_name="stock_vllm", aliases=("d",),
    preamble_token_ids=(3,),
    filters=(RequestFilterSpec(alias="d", written_positions=(0,),
                               question_token_ids=((90,),)),),
)


def _join_node(backend_name="stock_vllm"):
    return RequestExecution(
        node_id="request-model", backend_name=backend_name, aliases=("r", "p"),
        preamble_token_ids=(3,),
        joins=(RequestJoinSpec(
            written_pos=0, aliases=("r", "p"), outer_aliases=("r",),
            anchor="r", semantics="full", selectivity=0.5,
            label_token_ids=(("r", (40,)), ("p", (41,))),
            frame_token_ids=(("r", (30,)), ("p", (31,))),
            tail_token_ids=(50,),
        ),),
    )


def test_filter_and_join_answers_and_suffix_major_fallback():
    for model in (QWEN3_4B_FP8, DIFFUSION_GEMMA_26B_FP8):
        settings = {}
        if model is DIFFUSION_GEMMA_26B_FP8:
            settings = {"client": _ScoredClient(), "false_ids": [0],
                        "sampling_params": SimpleNamespace(logprobs=500)}
        execution = _execution(
            {"d": [[np.int32(10)], [11]], "r": [[10], [11]], "p": [[20], [21]]},
            model=model, **settings)

        filtered = execution.execute(FILTER_NODE, {})
        assert filtered.outputs["filter_answers:d"].to_pydict() == {
            "d": [0, 1], "predicate": [0, 0], "answer": [True, False]}, model.name
        assert filtered.outputs["ids:d"] == [0], model.name
        assert filtered.metrics.evaluated_documents == 2, model.name

        joined = execution.execute(_join_node(), {})
        assert joined.outputs["join_answers:0"].to_pydict() == {
            "r": [0, 0, 1, 1], "p": [0, 1, 0, 1],
            "answer": [True, True, False, False]}, model.name
        assert joined.outputs["ids:r"] == [0], model.name
        assert joined.outputs["ids:p"] == [0, 1], model.name
        assert joined.metrics.evaluated_document_pairs == 4, model.name

    class RecordingClient(_Client):
        def generate(self, prompts, sampling_params, use_tqdm=False):
            self.prompts = [tuple(prompt["prompt_token_ids"]) for prompt in prompts]
            return super().generate(prompts, sampling_params, use_tqdm=use_tqdm)

    # two anchor prefixes of three tokens each: six tokens held at once
    orders = {}
    for kv_tokens in (5, 6):
        execution = _execution(
            {"r": [[10], [11]], "p": [[20], [21]]}, join_submission="suffix-major",
            capacity={**CAPACITY, "kv_cache_size_tokens": kv_tokens})
        execution.client = RecordingClient()
        execution.execute(_join_node("pipelined_sglang"), {})
        orders[kv_tokens] = [(prompt[1], prompt[4])
                             for prompt in execution.client.prompts]
    assert orders[6] == [(10, 20), (11, 20), (10, 21), (11, 21)]
    assert orders[5] == [(10, 20), (10, 21), (11, 20), (11, 21)]


def test_sglang_submission_and_cancellation():
    class Engine:
        def __init__(self):
            self.calls = []

        def generate(self, *, input_ids, sampling_params):
            self.calls.append(input_ids)
            return [
                {"output_ids": [prompt[0] % 2], "meta_info": {"cached_tokens": 1}}
                for prompt in input_ids
            ]

    engine = Engine()
    client = SGLangClient(engine, {})
    prompts = [{"prompt_token_ids": [index, 9]} for index in range(17_000)]
    result = client.generate(prompts, {})
    assert engine.calls == [[prompt["prompt_token_ids"] for prompt in prompts]]
    assert [output.outputs[0].token_ids[0] for output in result] == [
        index % 2 for index in range(17_000)]
    assert all(output.num_cached_tokens == 1 for output in result)
    assert client.generate([], {}) == []
    assert len(engine.calls) == 1

    async def run():
        started = asyncio.Event()
        submitted = []
        aborted = []

        async def generate(**kwargs):
            submitted.append(kwargs["rid"])
            started.set()
            await asyncio.Event().wait()

        engine = SimpleNamespace(
            async_generate=generate,
            tokenizer_manager=SimpleNamespace(abort_request=aborted.append),
        )
        task = asyncio.create_task(SGLangClient(engine, {})._generate_one([1, 2], {}))
        await started.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert aborted == submitted
        assert len(aborted) == 1

    asyncio.run(run())


def test_vllm_engine_settings_tokenizer_and_canvas_follow_the_model(monkeypatch):
    from quail.backends.quail.executor.models.diffusion_gemma import canvas_token_ids
    from quail.backends.vllm import GigatokenVLLMTokenizer, diffusion_canvas

    engine = VLLMEngine()
    qwen = engine.llm_kwargs(QWEN3_4B_FP8)
    assert qwen["max_num_batched_tokens"] == 25_305
    assert qwen["tokenizer_mode"] == "gigatoken"
    assert "diffusion_config" not in qwen
    # the mixture-of-experts model batches its own chunk cap; its
    # one-row canvas commits after one denoising step, and stays under
    # vLLM's eight-sequence cap trigger
    gemma = engine.llm_kwargs(DIFFUSION_GEMMA_26B_FP8)
    assert gemma["max_num_batched_tokens"] == 65_536
    assert gemma["diffusion_config"] == {"canvas_length": 1,
                                         "max_denoising_steps": 1}
    assert gemma["max_num_seqs"] == 127
    assert gemma["max_logprobs"] == -1
    wide = engine.llm_kwargs(replace(DIFFUSION_GEMMA_26B_FP8, canvas_tokens=256))
    assert wide["diffusion_config"] == {"canvas_length": 256}
    assert wide["max_num_seqs"] == qwen["max_num_seqs"]
    assert sampling_kwargs([1, 2]) == {
        "temperature": 0.0, "max_tokens": 1, "min_tokens": 1,
        "allowed_token_ids": [1, 2], "detokenize": False}
    # the diffusion sampler rejects everything but the length; a
    # one-row canvas returns its logprobs, a longer one free text
    assert sampling_kwargs([1, 2], canvas_tokens=1) == {
        "max_tokens": 1, "logprobs": -1, "detokenize": False}
    assert sampling_kwargs([1, 2], canvas_tokens=256) == {"max_tokens": 16}

    adapter = SimpleNamespace(get_vocab=lambda: {"a": 0, "abc": 1})
    sources = []

    class Tokenizer:
        def __init__(self, source):
            sources.append(source)

        def as_hf(self):
            return adapter

    monkeypatch.setitem(
        sys.modules, "gigatoken", SimpleNamespace(Tokenizer=Tokenizer)
    )
    loaded = GigatokenVLLMTokenizer.from_pretrained(
        "Qwen/Qwen3-4B-FP8", truncation_side="right"
    )
    assert loaded is adapter
    assert sources == ["Qwen/Qwen3-4B-FP8"]
    assert loaded.truncation_side == "right"
    assert loaded.max_token_id == 1
    assert loaded.max_chars_per_token == 3

    spec = DIFFUSION_GEMMA_26B_FP8

    class States:
        canvas_length = 1
        vocab_size = spec.vocab

        def __init__(self):
            self.canvas = np.full((4, 1), -1, dtype=np.int64)

    module = SimpleNamespace(DiffusionGemmaRequestStates=States)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models",
                        SimpleNamespace(diffusion_gemma=module))
    with diffusion_canvas(spec) as tokens:
        states = module.DiffusionGemmaRequestStates()
    assert module.DiffusionGemmaRequestStates is States
    assert tokens == canvas_token_ids(spec.vocab, 1)
    states.init_canvas(np.array([3, 1]))
    assert states.canvas[:, 0].tolist() == [-1, tokens[0], -1, tokens[0]]
    states.canvas[1] = 42
    states.init_canvas(np.array([1]))
    assert states.canvas[1, 0] == tokens[0]
    states.vocab_size = 100
    with pytest.raises(ValueError, match="canvas geometry"):
        states.init_canvas(np.array([0]))
    with pytest.raises(RuntimeError, match="boot failed"), diffusion_canvas(spec):
        raise RuntimeError("boot failed")
    assert module.DiffusionGemmaRequestStates is States
    with diffusion_canvas(QWEN3_4B_FP8) as tokens:
        assert tokens == ()
        assert module.DiffusionGemmaRequestStates is States
