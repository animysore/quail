"""Kernel-source ablation on two QUAIL-B queries.

Compares our fused Triton kernels against the kernels stock vLLM's
compiled graph runs.

The queries are IMDB-7 (three filters over the reviews table, the
unified attention path) and BIO-2 (the reports x terms join, the
tree attention path), at scale factor 0.1. Each run goes
through the real planner and the real worker execution core
(quail.backends.quail.worker.execute_single); only the pipeline inside the
worker state is swapped, so packing, admission, the join search, KV
retention, GEMMs, and attention are identical across configurations.

What "stock vLLM's kernels" means here was measured, not assumed: the
stock_kernels cell below boots stock vLLM 0.26.0 at its defaults on
the same image and profiles a prefill pass over IMDB-1-shaped prompts
(/results/ablations/kernel_source_stock.json). At the default -O2
level the config enables the RMSNorm+quant and SiLU+quant fusion
passes (a blocked-fp8 checkpoint forces the quant_fp8 custom op on),
but the profiled graph shows the pattern rewrite does not fire for
this model under the ue8m0 scale mode DeepGEMM uses here: no fused
norm+quant or silu+quant kernel appears. Per layer, stock's compiled
graph actually runs two Inductor-generated add+rms_norm kernels, one
Inductor silu*mul kernel, two Inductor kernels for the q/k head norms
plus rotary (natives traced and fused by Inductor; the qk-norm+rope
fusion pass is off at every -O level), and four standalone CUDA
group-quant launches (one per GEMM input). Attention, the GEMMs, and
the KV-cache write sit outside the compiled graph and are the same
kernels the packed executor calls.

Three kernel sources, the engine untouched (the non-quail paths live
in an Engine subclass below):

  quail          our fused Triton kernels (the shipping executor)
  vllm_ops       vLLM's ops called one by one, unfused, and on the
                 tree path vLLM's merge_attn_states kernel
                 plus a separate group-quant in place of our fused
                 merge+quant kernel
  vllm_compiled  the kernel set stock vLLM's compiled graph runs:
                 torch.compile over the native add+rms_norm, silu*mul
                 and q/k-norm+rope math with vLLM's Inductor
                 settings, the same standalone group-quant per GEMM
                 input, and the same merge_attn_states join merge

    uv run modal run experiments/vllm_compiled_kernels.py::run_probe
    uv run modal run experiments/vllm_compiled_kernels.py::run_queries
    uv run modal run experiments/vllm_compiled_kernels.py::run_profile
    uv run modal run experiments/vllm_compiled_kernels.py::run_stock_kernels
"""

import json
import os

import modal

from quail.backends.quail.executor.attention import GROUP, Engine
from quail.bench.images import gpu_image

image = gpu_image()

# House rule: never create new Modal app names - new GPU cells attach
# to an existing app.
app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)

GPU_KW = dict(image=image, gpu="H100!", memory=98304,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})

KERNEL_SOURCES = ("quail", "vllm_ops", "vllm_compiled")
DATA_DIR = "/results/quailb_data"

# query id -> (attention path of its model work, result file suffix)
MEASURED_QUERIES = {
    "IMDB-7": ("unified", "imdb7"),
    "BIO-2": ("tree", "bio2"),
}


class KernelSourceEngine(Engine):
    """Engine with a swappable kernel source for the small kernels.

    kernel_source picks who provides the between-GEMM kernels and the
    join merge; the loop, packing, arena, GEMMs, and attention calls
    are the base class's in every mode.
    """

    def __init__(self, arena, *, kernel_source="quail", **kwargs):
        super().__init__(arena, **kwargs)
        self._segments = None
        self.kernel_source = kernel_source

    @property
    def kernel_source(self):
        return self._kernel_source

    @kernel_source.setter
    def kernel_source(self, source):
        if source not in KERNEL_SOURCES:
            raise ValueError(f"kernel_source must be one of "
                             f"{KERNEL_SOURCES}, got {source!r}")
        if (source == "vllm_compiled"
                and self.rotary.rotary_dim != self.head_dim):
            raise RuntimeError(
                "the native rope transcription assumes rotary over "
                "the full head dimension")
        self._kernel_source = source
        self.kernels = "quail" if source == "quail" else "vllm"

    # ---- the segments stock's compiled graph runs -------------------
    # The measured stock inventory (kernel_source_stock.json) shows
    # Inductor-generated kernels for add+rms_norm, silu*mul, and
    # q/k-norm+rope, each followed by the standalone group-quant
    # custom op (the fusion passes do not rewrite this model's graph
    # under ue8m0 scales). The native math is transcribed from
    # vllm/ir/ops/layernorm.py (rms_norm, fused_add_rms_norm) and
    # vllm/model_executor/layers/rotary_embedding/{base,common}.py
    # (forward_static, neox style); combo_kernels matches the
    # inductor_compile_config stock sets for torch >= 2.9. The quant
    # is self.quant, the same vLLM group-quant call the engine and
    # stock both run, so ue8m0 scale rounding stays consistent.

    def _compiled_segments(self):
        if self._segments is not None:
            return self._segments
        torch = self.torch
        H, KH, D = self.num_q_heads, self.num_kv_heads, self.head_dim

        def add_norm_native(hidden, residual, weight, eps):
            # residual accumulates in place, as the engine expects;
            # Inductor folds the write into the generated kernel
            residual += hidden
            xf = residual.to(torch.float32)
            variance = xf.pow(2).mean(dim=-1, keepdim=True)
            xf = xf * torch.rsqrt(variance + eps)
            return xf.to(weight.dtype) * weight

        def silu_mul_native(gate_up):
            half = gate_up.shape[-1] // 2
            gate = gate_up[..., :half]
            up = gate_up[..., half:]
            return torch.nn.functional.silu(gate) * up

        def norm(x, w, eps):
            xf = x.to(torch.float32)
            variance = xf.pow(2).mean(dim=-1, keepdim=True)
            xf = xf * torch.rsqrt(variance + eps)
            return xf.to(w.dtype) * w

        def rope(x, cos, sin):
            x1, x2 = torch.chunk(x, 2, dim=-1)
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            return torch.cat((o1, o2), dim=-1)

        def qk_norm_rope_native(qkv, positions, q_w, k_w, cos_sin,
                                eps):
            n = qkv.shape[0]
            q = qkv[:, :H * D].reshape(n, H, D)
            k = qkv[:, H * D:(H + KH) * D].reshape(n, KH, D)
            q = norm(q, q_w, eps)
            k = norm(k, k_w, eps)
            cos_sin_rows = cos_sin.index_select(0, positions)
            cos, sin = cos_sin_rows.chunk(2, dim=-1)
            cos = cos.unsqueeze(-2).to(q.dtype)
            sin = sin.unsqueeze(-2).to(q.dtype)
            q = rope(q, cos, sin).reshape(n, H * D)
            k = rope(k, cos, sin).reshape(n, KH * D)
            return q.contiguous(), k.contiguous()

        options = {"combo_kernels": True,
                   "benchmark_combo_kernel": True}
        self._segments = dict(
            add_norm=torch.compile(add_norm_native, dynamic=True,
                                   options=options),
            silu_mul=torch.compile(silu_mul_native, dynamic=True,
                                   options=options),
            qk=torch.compile(qk_norm_rope_native, dynamic=True,
                             options=options))
        return self._segments

    def vllm_norm_quant(self, hidden, norm, residual):
        if self.kernel_source != "vllm_compiled":
            return super().vllm_norm_quant(hidden, norm, residual)
        normed = self._compiled_segments()["add_norm"](
            hidden, residual, norm.weight, norm.variance_epsilon)
        return self.quant(normed)

    def vllm_silu_quant(self, gate_up):
        if self.kernel_source != "vllm_compiled":
            return super().vllm_silu_quant(gate_up)
        return self.quant(self._compiled_segments()["silu_mul"](
            gate_up))

    def vllm_qk_norm_rope(self, qkv, positions, attn):
        if self.kernel_source != "vllm_compiled":
            return super().vllm_qk_norm_rope(qkv, positions, attn)
        return self._compiled_segments()["qk"](
            qkv, positions, attn.q_norm.weight, attn.k_norm.weight,
            self.rotary.cos_sin_cache, attn.q_norm.variance_epsilon)

    # ---- the join merge: vLLM's merge_attn_states -------------------
    # Stock never merges in its compiled graph (its cascade merge
    # lives inside the attention backend), so both vLLM sources use
    # its eager merge kernel plus the standalone group-quant. out_b
    # covers only the rows with cached context, so those rows are
    # gathered, merged, and scattered back; the row list comes from
    # the chunk meta stashed by attention_tree below.

    def attention_tree(self, q, k, v, meta):
        reads = meta.get("reads")
        self._merge_rows = None if reads is None else reads["rows"]
        return super().attention_tree(q, k, v, meta)

    def merge_attn_quant(self, out_a, lse_a, out_b, lse_b, source):
        if self.kernel_source == "quail":
            return super().merge_attn_quant(out_a, lse_a, out_b,
                                            lse_b, source)
        from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
        torch = self.torch
        n, heads, dim = out_a.shape
        rows = self._merge_rows
        a_sub = out_a.index_select(0, rows)
        merged = torch.empty_like(a_sub)
        # lse arrives as a [tokens, heads] transposed view; the merge
        # kernel wants the original [heads, tokens] layout back
        merge_attn_states(
            merged, out_b, lse_b.transpose(0, 1),
            a_sub, lse_a.transpose(0, 1).index_select(1, rows))
        out_a.index_copy_(0, rows, merged)
        return self.quant(out_a.view(n, heads * dim))


# ------------------------------------------------------------- helpers

def _write(result, name):
    print(json.dumps(result, indent=2), flush=True)
    os.makedirs("/results/ablations", exist_ok=True)
    with open(f"/results/ablations/{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    kernel_cache.commit()
    return json.dumps(result)


def _boot_state(model):
    """Boot the worker state dict with the kernel-source pipeline.

    Mirrors quail.execution.execute._execute_physical's boot, with the
    Engine subclass swapped in; warm_kernels runs the same tiered
    warmup the worker runs.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.loop import warm_kernels
    from quail.backends.quail.executor.model import load_model
    from quail.backends.quail.executor.models import build_pipeline
    from quail.backends.quail.executor.readout import AnswerRows, AsyncAnswers
    from quail.cost import budgets
    from quail.specs import DEVICES, MODELS

    spec = MODELS[model]
    device = DEVICES["h100-sxm"]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    model_mod = load_model(spec.hf_name, revision=spec.revision)
    chunk_tokens = budgets.chunk_budget(spec, device)
    arena_tok = budgets.arena_tokens(spec, device, chunk_tokens)
    arena = KVArena(n_layers=spec.layers,
                    n_pages=arena_tok // budgets.PAGE_TOKENS,
                    page_tokens=budgets.PAGE_TOKENS,
                    n_kv=spec.n_kv, d_head=spec.d_head,
                    dtype=torch.bfloat16)
    pipeline = build_pipeline(spec, model_mod, arena,
                              engine_class=KernelSourceEngine)
    from quail.backends import GpuContext, QuailBackend
    execution = QuailBackend().start(GpuContext(
        gpu_index=0,
        gpu_count=1,
        model=spec,
        device=device,
        query_settings={
            "chunk_tokens": chunk_tokens,
        },
    ))
    execution.bind_loaded_model(
        model=model_mod, arena=arena, pipeline=pipeline
    )
    answerer = AnswerRows.from_tokenizer(torch, F, model_mod, tokenizer)
    async_ans = AsyncAnswers(torch, answerer)
    with torch.inference_mode():
        warm = warm_kernels(torch, arena, pipeline, async_ans,
                            chunk_tokens, model_name=spec.hf_name)
    torch.cuda.synchronize()
    kernel_cache.commit()
    state = dict(model_execution=execution,
                 model=model_mod, arena=arena, pipeline=pipeline,
                 spec=spec, torch=torch, F=F)
    return state, tokenizer, chunk_tokens, warm


def _quailb_session(model, sf, gpus=1):
    """Build the QUAIL-B tables and a registered session."""
    import quail
    from quail.bench.quailb import queries, register_tables
    from quail.planner.plan import EngineConfig
    from quail_b.data import build_sets

    d = build_sets(DATA_DIR, sf)
    results_vol.commit()
    sess = quail.Session(EngineConfig(
        gpus=gpus,
        model=model,
        backend="quail",
        device="h100-sxm",
    ))
    register_tables(sess, d)
    return sess, queries(sess), d


def _run_query(state, build, captured):
    """One query through the real planner and worker core."""
    from quail.backends.quail.executor.readout import AnswerRows, AsyncAnswers
    from quail.backends.quail.worker import (
        execute_single,
        quail_runtime_payload,
    )
    from quail.execution.execute import (
        _validate_physical_request,
        execute_query,
    )
    from quail.execution.types import PhysicalResponse

    def execute(request):
        request, registry, graph, _ = _validate_physical_request(
            request, query.session.registry)
        payload = quail_runtime_payload(request, graph)
        rows = AnswerRows(
            state["torch"], state["F"], state["model"],
            payload["true_ids"], payload["false_ids"],
        )
        state["model_execution"].bind_query(
            torch=state["torch"],
            async_answers=AsyncAnswers(state["torch"], rows),
            answer_rows=rows,
            chunk_tokens=payload["chunk_tokens"],
        )
        report = execute_single(state, payload, registry, graph)
        outputs = report.pop("_outputs")
        report.pop("filters", None)
        report.pop("joins", None)
        captured.clear()
        captured.update(report)
        return PhysicalResponse(outputs, report)

    query = build()
    return execute_query(query, physical_executor=execute)


def _row_key(table):
    """Sorted output rows as tuples, for cross-source comparison."""
    columns = table.column_names
    rows = table.to_pylist()
    return sorted(tuple(r[c] for c in columns) for r in rows)


def _table_rows(data_dir, name):
    import pyarrow.parquet as pq
    return pq.read_table(f"{data_dir}/{name}.parquet").num_rows


def _dequant(torch, q, scales):
    """FP8 groups back to float32 for cross-kernel comparison."""
    return q.to(torch.float32) * scales.to(
        torch.float32).repeat_interleave(GROUP, dim=1)


def _max_abs(torch, left, right):
    return float((left.to(torch.float32)
                  - right.to(torch.float32)).abs().max().item())


# --------------------------------------------------------------- probe

@app.function(timeout=3600, **GPU_KW)
def probe(model: str = "qwen3-4b-fp8") -> str:
    """Check the three kernel sources agree before the measured cells run.

    Checks per-kernel parity of the two vLLM sources against the quail
    source, torch.compile sanity for the compiled segments, and row
    agreement on small QUAIL-B queries (sf 0.01).
    """
    import time

    import torch

    state, tokenizer, chunk_tokens, warm = _boot_state(model)
    pipeline = state["pipeline"]
    torch.manual_seed(20260829)
    report = dict(cell="kernel_source_probe", model=model, warm=warm,
                  vllm_ops={}, vllm_compiled={}, query_rows={})

    n = 4096
    layer = pipeline.layers[0]
    attn = layer.self_attn
    h = layer.input_layernorm.weight.shape[0]
    gate_up_w = layer.mlp.gate_up_proj.weight.shape[0]
    qkv_w = attn.qkv_proj.weight.shape[0]

    with torch.inference_mode():
        # norm+quant: same hidden/residual through all three sources
        hidden0 = torch.randn(n, h, device="cuda",
                              dtype=torch.bfloat16)
        residual0 = torch.randn(n, h, device="cuda",
                                dtype=torch.bfloat16)
        outs = {}
        for source in KERNEL_SOURCES:
            pipeline.engine.kernel_source = source
            hidden = hidden0.clone()
            residual = residual0.clone()
            if source == "quail":
                q, s = pipeline.engine.custom_norm_quant(
                    hidden, layer.input_layernorm, residual)
            else:
                q, s = pipeline.engine.vllm_norm_quant(
                    hidden, layer.input_layernorm, residual)
            outs[source] = (_dequant(torch, q, s), residual)
        for source in ("vllm_ops", "vllm_compiled"):
            report[source]["norm_quant_max_abs"] = _max_abs(
                torch, outs[source][0], outs["quail"][0])
            report[source]["norm_residual_max_abs"] = _max_abs(
                torch, outs[source][1], outs["quail"][1])

        # silu+quant
        gate_up = torch.randn(n, gate_up_w, device="cuda",
                              dtype=torch.bfloat16)
        outs = {}
        for source in KERNEL_SOURCES:
            pipeline.engine.kernel_source = source
            if source == "quail":
                q, s = pipeline.engine.custom_silu_quant(gate_up)
            else:
                q, s = pipeline.engine.vllm_silu_quant(gate_up)
            outs[source] = _dequant(torch, q, s)
        for source in ("vllm_ops", "vllm_compiled"):
            report[source]["silu_quant_max_abs"] = _max_abs(
                torch, outs[source], outs["quail"])

        # q/k norm + rope; the compiled source also reports compile
        # time
        qkv = torch.randn(n, qkv_w, device="cuda",
                          dtype=torch.bfloat16)
        positions = torch.randint(0, 4096, (n,), device="cuda",
                                  dtype=torch.int64)
        outs = {}
        for source in KERNEL_SOURCES:
            pipeline.engine.kernel_source = source
            t0 = time.perf_counter()
            if source == "quail":
                qo, ko = pipeline.engine.custom_qk_norm_rope(
                    qkv.clone(), positions, attn)
            else:
                qo, ko = pipeline.engine.vllm_qk_norm_rope(
                    qkv.clone(), positions, attn)
            torch.cuda.synchronize()
            if source == "vllm_compiled":
                report[source]["qk_first_call_s"] = round(
                    time.perf_counter() - t0, 2)
            outs[source] = (qo, ko)
        for source in ("vllm_ops", "vllm_compiled"):
            report[source]["qk_q_max_abs"] = _max_abs(
                torch, outs[source][0], outs["quail"][0])
            report[source]["qk_k_max_abs"] = _max_abs(
                torch, outs[source][1], outs["quail"][1])

        # count launches of each compiled segment at a second token
        # count (also proves dynamic shapes hold - no recompile);
        # add_norm should be one generated kernel with the residual
        # write folded in plus the quant launch, not a separate copy
        pipeline.engine.kernel_source = "vllm_compiled"

        def count_kernels(fn):
            fn()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                fn()
                torch.cuda.synchronize()
            return [ev.key for ev in prof.key_averages()
                    if "CUDA" in str(getattr(ev, "device_type", ""))
                    and (getattr(ev, "self_device_time_total", 0)
                         or getattr(ev, "self_cuda_time_total", 0))]

        qkv2 = torch.randn(2048, qkv_w, device="cuda",
                           dtype=torch.bfloat16)
        pos2 = positions[:2048]
        kernels = count_kernels(
            lambda: pipeline.engine.vllm_qk_norm_rope(qkv2, pos2, attn))
        report["vllm_compiled"]["qk_kernel_launches"] = len(kernels)
        report["vllm_compiled"]["qk_kernels"] = kernels[:12]

        hidden2 = hidden0[:2048].clone()
        residual2 = residual0[:2048].clone()
        kernels = count_kernels(
            lambda: pipeline.engine.vllm_norm_quant(
                hidden2, layer.input_layernorm, residual2))
        report["vllm_compiled"]["norm_quant_kernel_launches"] = \
            len(kernels)
        report["vllm_compiled"]["norm_quant_kernels"] = kernels[:12]

    # small real queries: one filter and one join at sf 0.01, rows
    # compared across sources through the real planner and worker
    sess, qdefs, _ = _quailb_session(model, sf=0.01)
    for query_id in ("IMDB-1", "BIO-2"):
        rows = {}
        for source in KERNEL_SOURCES:
            pipeline.engine.kernel_source = source
            captured = {}
            result = _run_query(state, qdefs[query_id][1], captured)
            rows[source] = _row_key(result.collect())
        report["query_rows"][query_id] = {
            "quail_rows": len(rows["quail"]),
            "vllm_ops_disagreements": len(
                set(rows["quail"]) ^ set(rows["vllm_ops"])),
            "vllm_compiled_disagreements": len(
                set(rows["quail"]) ^ set(rows["vllm_compiled"])),
        }
    torch.cuda.synchronize()
    return _write(report, "kernel_source_probe")


# ----------------------------------------------- stock kernel inventory

@app.function(timeout=3600, **GPU_KW)
def stock_kernels(model: str = "qwen3-4b-fp8",
                  n_docs: int = 512) -> str:
    """Record which kernels stock vLLM's compiled graph runs between GEMMs.

    Boots stock vLLM at its defaults, profiles one prefill-heavy pass
    over IMDB-1-shaped prompts, and records the kernels plus the
    resolved compilation config. This is the ground truth the
    vllm_compiled source mirrors.
    """
    import os as _os

    # the v1 engine runs the model in a child process by default,
    # where this process's profiler cannot see the kernels
    _os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from quail.logical import bind_prompt, render_filter_prompt_ids
    from quail.specs import MODELS
    from quail_b.data import build_sets
    from quail_b.prompts import F1

    spec = MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)

    def tok(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    d = build_sets(DATA_DIR, 0.1)
    bodies = pq.read_table(f"{d}/reviews.parquet")["body"].to_pylist()
    prompt = bind_prompt(F1, ("body",), tok, turn=spec.turn)
    prompts = [dict(prompt_token_ids=render_filter_prompt_ids(
        prompt, tok(b), tok)) for b in bodies[:n_docs]]

    llm = LLM(model=spec.hf_name, gpu_memory_utilization=0.92,
              enable_prefix_caching=False, disable_log_stats=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=1,
                              min_tokens=1)
    config = llm.llm_engine.vllm_config
    comp = config.compilation_config
    report = dict(
        cell="stock_kernels",
        model=spec.hf_name,
        optimization_level=int(config.optimization_level),
        compilation_mode=str(comp.mode),
        custom_ops=list(comp.custom_ops),
        enabled_custom_ops=dict(comp.enabled_custom_ops),
        disabled_custom_ops=dict(comp.disabled_custom_ops),
        pass_config={
            k: bool(getattr(comp.pass_config, k))
            for k in ("fuse_norm_quant", "fuse_act_quant",
                      "fuse_attn_quant", "enable_qk_norm_rope_fusion")
            if getattr(comp.pass_config, k, None) is not None},
        use_deep_gemm_e8m0=None,
        kernels=[])
    try:
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
        report["use_deep_gemm_e8m0"] = bool(is_deep_gemm_e8m0_used())
    except Exception as exc:
        report["use_deep_gemm_e8m0"] = repr(exc)

    llm.generate(prompts[:32], sampling)   # warm outside the profile
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        llm.generate(prompts, sampling)
    rows = []
    total_us = 0.0
    for ev in prof.key_averages():
        # keep only device-side rows: op wrapper rows (CPU device
        # type) repeat the kernel name with the same device time and
        # would double-count it
        if "CUDA" not in str(getattr(ev, "device_type", "")):
            continue
        cuda_us = getattr(ev, "self_device_time_total", 0) or \
            getattr(ev, "self_cuda_time_total", 0)
        if not cuda_us:
            continue
        total_us += cuda_us
        rows.append((round(cuda_us / 1e3, 2), ev.count, ev.key[:110]))
    rows.sort(reverse=True)
    report["cuda_busy_s"] = round(total_us / 1e6, 2)
    report["kernels"] = [dict(ms=ms, n=n, name=k)
                         for ms, n, k in rows[:60]]
    return _write(report, "kernel_source_stock")


# ------------------------------------------------------- measured cell

PREDICTIONS = {
    "IMDB-7": {
        "quail": "the reference: 16.1 s at sf 0.1 in the QUAIL-B "
                 "sf0.1 report (the same query through the same "
                 "worker)",
        "vllm_ops": "+25 to +32% wall: the per-token gap measured "
                    "on the deleted synthetic workload (+29%) "
                    "carries over because the model work per token "
                    "is the same",
        "vllm_compiled": "+22 to +29% wall: the compiled set keeps "
                         "all four standalone group-quant launches "
                         "per layer, so it recovers only the q/k "
                         "segment against vllm_ops",
    },
    "BIO-2": {
        "quail": "the reference: 32.2 s at sf 0.1 in the QUAIL-B "
                 "sf0.1 report",
        "vllm_ops": "+26 to +33% wall: the filter-side gap plus the "
                    "unfused merge (gather, merge_attn_states, "
                    "scatter, quant against our one fused kernel)",
        "vllm_compiled": "+23 to +30% wall: same unfused quants and "
                         "unfused merge, minus the q/k segment "
                         "saving",
    },
}


@app.function(timeout=7200, **GPU_KW)
def queries(model: str = "qwen3-4b-fp8", sf: float = 0.1,
            reps: int = 2) -> str:
    """Run IMDB-7 and BIO-2 through all three kernel sources.

    Uses one container, one model load, and the real planner and worker
    core. Writes kernel_source_imdb7.json and kernel_source_bio2.json.
    """
    state, tokenizer, chunk_tokens, warm = _boot_state(model)
    pipeline = state["pipeline"]
    print(f"[kernel_source] warm: {warm}", flush=True)
    print(f"[kernel_source] predictions: {json.dumps(PREDICTIONS)}",
          flush=True)

    sess, qdefs, d = _quailb_session(model, sf)
    counts = dict(reviews=_table_rows(d, "reviews"),
                  reports=_table_rows(d, "reports"),
                  terms=_table_rows(d, "terms"))

    out = None
    for query_id, (path, suffix) in MEASURED_QUERIES.items():
        description, build = qdefs[query_id]
        report = dict(
            cell=f"kernel_source_{suffix}", model=model, sf=sf,
            query=query_id, description=description,
            attention_path=path, reps=reps,
            chunk_tokens=chunk_tokens, table_rows=counts,
            predictions=PREDICTIONS[query_id], runs={},
            comparisons={})
        rows_by_source = {}
        for source in KERNEL_SOURCES:
            pipeline.engine.kernel_source = source
            # unmeasured warm run: first-call op init, and for
            # vllm_compiled the torch.compile of the three segments
            _run_query(state, build, {})
            runs = []
            for rep in range(reps):
                captured = {}
                result = _run_query(state, build, captured)
                table = result.collect()
                row = dict(
                    source=source, rep=rep,
                    wall_s=round(captured["wall_s"], 3),
                    fresh_tokens=captured["fresh_tokens"],
                    us_per_token=round(
                        captured["wall_s"] * 1e6
                        / captured["fresh_tokens"], 3),
                    rows=table.num_rows)
                runs.append(row)
                print(f"[kernel_source_{suffix}] {row}", flush=True)
                rows_by_source[source] = _row_key(table)
            report["runs"][source] = runs

        reference = rows_by_source["quail"]
        last_quail = report["runs"]["quail"][-1]
        for source in ("vllm_ops", "vllm_compiled"):
            last = report["runs"][source][-1]
            report["comparisons"][source] = dict(
                row_disagreements=len(
                    set(reference) ^ set(rows_by_source[source])),
                wall_delta_s=round(
                    last["wall_s"] - last_quail["wall_s"], 3),
                us_per_token_delta=round(
                    last["us_per_token"]
                    - last_quail["us_per_token"], 3))
        out = _write(report, f"kernel_source_{suffix}")
    return out


# ------------------------------------------------------------ profile

def _categorize(name):
    """Map a kernel name to its time bucket.

    Our Triton kernel names contain substrings that also appear in
    vLLM's op names, so the specific names are checked before the
    generic ones.
    """
    low = name.lower()
    if "deep_gemm" in low or "sm90_fp8" in low or "gemm" in low:
        return "gemm"
    if "merge_attn_states" in low:
        return "merge"
    if "flash" in low or "attn" in low:
        return "attention"
    if any(k in low for k in ("silu_mul_quant", "add_rms_norm_quant",
                              "qk_norm_rope", "merge_quant")):
        return "quail_fused"
    if low.startswith("triton_"):
        return "inductor"
    if any(k in low for k in ("rms_norm", "rotary", "silu",
                              "act_and_mul")):
        return "vllm_elementwise"
    if "quant" in low:
        return "quant"
    if any(k in low for k in ("memcpy", "copy", "index", "cat",
                              "gather", "scatter", "nonzero",
                              "elementwise")):
        return "copies"
    return "other"


class _ClockSampler:
    """Sample the SM clock and power draw every 50 ms on a thread.

    The GPU adjusts its own clock against the power limit, so the
    same kernel runs slower when the chip sits at sustained high
    power; this records what the clock actually was during a run.
    """

    def __init__(self):
        import pynvml
        pynvml.nvmlInit()
        self._nvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def run(self, fn):
        import statistics
        import threading
        import time

        samples = []
        stop = threading.Event()

        def loop():
            while not stop.is_set():
                samples.append((
                    self._nvml.nvmlDeviceGetClockInfo(
                        self._handle, self._nvml.NVML_CLOCK_SM),
                    self._nvml.nvmlDeviceGetPowerUsage(
                        self._handle) / 1000))
                time.sleep(0.05)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        try:
            out = fn()
        finally:
            stop.set()
            thread.join()
        clocks = sorted(s[0] for s in samples)
        powers = [s[1] for s in samples]
        stats = dict(
            samples=len(samples),
            sm_mhz_mean=round(statistics.mean(clocks), 1),
            sm_mhz_median=clocks[len(clocks) // 2],
            sm_mhz_p10=clocks[len(clocks) // 10],
            power_w_mean=round(statistics.mean(powers), 1))
        return out, stats


def _try_lock_clocks(mhz):
    """Pin the SM clock; returns True when the driver allows it.

    The pin must sit below the power-throttle point so both
    configurations actually run at the same frequency.
    """
    import subprocess
    done = subprocess.run(
        ["nvidia-smi", "-lgc", f"{mhz},{mhz}"],
        capture_output=True, text=True)
    return done.returncode == 0, (done.stdout + done.stderr).strip()


def _unlock_clocks():
    import subprocess
    subprocess.run(["nvidia-smi", "-rgc"], capture_output=True)


LOCKED_MHZ = 1500


@app.function(timeout=7200, **GPU_KW)
def profile_queries(model: str = "qwen3-4b-fp8",
                    sf: float = 0.1) -> str:
    """Measure per-kernel-category GPU time for the three sources on IMDB-7.

    Uses the unified attention path. Samples the SM clock and power
    during an unprofiled run of each source. Where the driver allows
    pinning the clock, adds a locked-clock profiled pass that removes
    clock behavior from the matmul comparison.
    """
    import torch

    state, tokenizer, chunk_tokens, _ = _boot_state(model)
    pipeline = state["pipeline"]
    sess, qdefs, _ = _quailb_session(model, sf)
    _, build = qdefs["IMDB-7"]
    sampler = _ClockSampler()

    result = dict(cell="kernel_source_profile", model=model, sf=sf,
                  query="IMDB-7", attention_path="unified",
                  sources={}, locked_clock=None)

    def profile_pass(source):
        pipeline.engine.kernel_source = source
        captured = {}
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            _run_query(state, build, captured)
        return prof, captured

    for source in KERNEL_SOURCES:
        pipeline.engine.kernel_source = source
        captured = {}
        _run_query(state, build, captured)   # unprofiled warm
        # clock and power during a normal, unprofiled run
        (_, clock_stats) = sampler.run(
            lambda: _run_query(state, build, {}))
        prof, captured = profile_pass(source)
        tokens = captured["fresh_tokens"]
        cats, counts = {}, {}
        rows = []
        for ev in prof.key_averages():
            # keep only device-side rows: op wrapper rows (CPU
            # device type) repeat the kernel name with the same
            # device time and would double-count it
            if "CUDA" not in str(getattr(ev, "device_type", "")):
                continue
            cuda_us = getattr(ev, "self_device_time_total", 0) or \
                getattr(ev, "self_cuda_time_total", 0)
            if not cuda_us:
                continue
            cat = _categorize(ev.key)
            cats[cat] = cats.get(cat, 0.0) + cuda_us
            counts[cat] = counts.get(cat, 0) + ev.count
            rows.append((round(cuda_us / 1e6, 3), ev.count,
                         ev.key[:90]))
        rows.sort(reverse=True)
        busy = sum(cats.values())
        result["sources"][source] = dict(
            fresh_tokens=tokens,
            wall_s=round(captured["wall_s"], 3),
            clock=clock_stats,
            cuda_busy_s=round(busy / 1e6, 2),
            us_per_token=round(busy / tokens, 2),
            category_us_per_token={k: round(v / tokens, 3)
                                   for k, v in sorted(cats.items())},
            category_launches={k: counts[k] for k in sorted(counts)},
            top_kernels=[dict(s=s, n=n, name=k) for s, n, k
                         in rows[:20]])
        print(f"[kernel_source_profile] {source}: "
              f"clock {clock_stats} "
              f"{json.dumps(result['sources'][source]['category_us_per_token'])}",
              flush=True)

    # locked-clock pass: pin the SM clock below the power-throttle
    # point so every source runs the matmuls at the same frequency;
    # if clock behavior explains the matmul-bucket difference, the
    # gap disappears here
    locked, message = _try_lock_clocks(LOCKED_MHZ)
    result["locked_clock"] = dict(supported=locked, mhz=LOCKED_MHZ,
                                  driver_message=message[:200],
                                  sources={})
    if locked:
        try:
            for source in ("quail", "vllm_ops"):
                pipeline.engine.kernel_source = source
                _run_query(state, build, {})   # settle at the pin
                (_, clock_stats) = sampler.run(
                    lambda: _run_query(state, build, {}))
                prof, captured = profile_pass(source)
                tokens = captured["fresh_tokens"]
                gemm_us = 0.0
                for ev in prof.key_averages():
                    if "CUDA" not in str(getattr(ev, "device_type",
                                                 "")):
                        continue
                    cuda_us = getattr(ev, "self_device_time_total",
                                      0) or \
                        getattr(ev, "self_cuda_time_total", 0)
                    if cuda_us and _categorize(ev.key) == "gemm":
                        gemm_us += cuda_us
                row = dict(
                    gemm_us_per_token=round(gemm_us / tokens, 3),
                    wall_s=round(captured["wall_s"], 3),
                    clock=clock_stats)
                result["locked_clock"]["sources"][source] = row
                print(f"[kernel_source_profile] locked {source}: "
                      f"{row}", flush=True)
        finally:
            _unlock_clocks()
    return _write(result, "kernel_source_profile")


# ---------------------------------------------------------- entrypoints

@app.local_entrypoint()
def run_probe(model: str = "qwen3-4b-fp8"):
    handle = probe.spawn(model)
    print(f"probe fc: {handle.object_id}", flush=True)
    print(handle.get())


@app.local_entrypoint()
def run_queries(model: str = "qwen3-4b-fp8", sf: float = 0.1,
                reps: int = 2):
    handle = queries.spawn(model, sf, reps)
    print(f"queries fc: {handle.object_id}", flush=True)
    print(handle.get())


@app.local_entrypoint()
def run_profile(model: str = "qwen3-4b-fp8", sf: float = 0.1):
    handle = profile_queries.spawn(model, sf)
    print(f"profile_queries fc: {handle.object_id}", flush=True)
    print(handle.get())


@app.local_entrypoint()
def run_stock_kernels(model: str = "qwen3-4b-fp8",
                      n_docs: int = 512):
    handle = stock_kernels.spawn(model, n_docs)
    print(f"stock_kernels fc: {handle.object_id}", flush=True)
    print(handle.get())
