"""DiffusionGemma: spec geometry, chat-turn text, canvas rows, and the layer loop."""

import contextlib
import sys
import types
from types import SimpleNamespace

import pytest
from fakes import cpu_arena, cpu_staging, fake_pipeline, fake_torch

from quail.backends.quail.executor import loop
from quail.backends.quail.executor.models.diffusion_gemma import (
    DiffusionGemmaPipeline,
)
from quail.backends.quail.executor.pack import JoinAdmission
from quail.cost import budgets
from quail.cost.dense_decoder_cost import mlp_params, mlp_weight_params
from quail.logical.prompts import (
    ANSWER_CUE,
    SHARED_PRE,
    bind_join_prompt,
    bind_prompt,
    filter_question_text,
    render_filter_prompt_ids,
    render_join_prompt_ids,
)
from quail.specs import DIFFUSION_GEMMA_26B_FP8, H100_SXM, QWEN3_4B_FP8

SPEC = DIFFUSION_GEMMA_26B_FP8


def _tok(text):
    return [ord(c) for c in text]


def test_spec_geometry_budgets_costs_and_turn_text():
    shapes = SPEC.kv_shapes
    assert len(shapes) == 30
    assert [i for i, s in enumerate(shapes) if s == (2, 512)] == [5, 11, 17, 23, 29]
    assert all(s == (8, 256) for i, s in enumerate(shapes) if (i + 1) % 6)
    assert SPEC.widest_projection == 16 * 512 + 2 * 2 * 512
    assert SPEC.kv_elements_per_token == 2 * (25 * 2048 + 5 * 1024)
    assert SPEC.kappa == 225_280.0
    assert QWEN3_4B_FP8.kv_shapes == ((8, 128),) * 36
    assert QWEN3_4B_FP8.kappa == 147_456.0
    assert SPEC.canvas_tokens == 1
    assert SPEC.turn == ("<bos><|turn>user\n",
                         "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
    assert SPEC.canvas_answer_row == 0
    assert budgets.chunk_budget(SPEC, H100_SXM) == SPEC.chunk_cap_tokens
    assert budgets.chunk_budget(QWEN3_4B_FP8, H100_SXM) == \
        budgets.kernel_index_cap(QWEN3_4B_FP8)
    assert budgets.arena_tokens(SPEC, H100_SXM) > 100_000
    assert mlp_weight_params(SPEC) > 10 * mlp_params(SPEC)
    assert mlp_weight_params(QWEN3_4B_FP8) == mlp_params(QWEN3_4B_FP8)
    assert mlp_params(SPEC) == 30 * 3 * 2816 * (2112 + 8 * 704)
    assert mlp_weight_params(SPEC) == 30 * 3 * 2816 * (2112 + 128 * 704)

    # the chat turn wraps filter and join prompts
    turn = ("<bos><|turn>user\n", "<turn|>\n<|turn>model\n")
    ref = SimpleNamespace(alias="d")
    prompt = bind_prompt("Is {0} about food?", (ref,), _tok, turn=turn)
    assert prompt.preamble == turn[0] + SHARED_PRE
    assert prompt.tail.endswith(ANSWER_CUE + turn[1])
    doc = [7, 8, 9]
    ids = render_filter_prompt_ids(prompt, doc, _tok)
    assert ids[:len(_tok(prompt.preamble))] == _tok(prompt.preamble)
    assert ids[-len(_tok(turn[1])):] == _tok(turn[1])
    assert ids == (_tok(prompt.preamble) + doc
                   + _tok(filter_question_text(prompt)))
    assert tuple(_tok(prompt.preamble)) == prompt.preamble_token_ids

    left, right = SimpleNamespace(alias="a"), SimpleNamespace(alias="b")
    join = bind_join_prompt("Same topic in {0} and {1}?", (left, right), _tok,
                            turn=turn)
    assert join.preamble == turn[0] + SHARED_PRE
    assert join.tail == ANSWER_CUE + turn[1]
    assert join.preamble_token_ids == tuple(_tok(join.preamble))
    assert join.tail_token_ids == tuple(_tok(join.tail))
    ids = render_join_prompt_ids(join, [[1, 2], [3]], 0, _tok)
    assert ids[-len(_tok(join.tail)):] == _tok(join.tail)


def test_canvas_rows_are_charged_in_filter_streams_and_packed_chunks(
        monkeypatch):
    pipeline = fake_pipeline(canvas_ids=(1, 2, 3))
    answers = SimpleNamespace(submit=lambda v: v, result=lambda v: v, dtype=None)
    docs = [[5] * 20, [6] * 30]
    questions = [[40, 41, 42], [40, 41, 43, 44]]
    stream = loop.FilterStream(
        fake_torch(), cpu_arena(64), pipeline, answers, docs, questions, 200,
        arena_writes=True, arena_keys=[("d", 0), ("d", 1)])
    assert stream.preamble == 2
    assert stream.sched.stage_tokens == [3 + 3, 2 + 3]
    assert stream.capacity_extra == 2 + 2 + 3
    assert stream.canvas == (1, 2, 3)

    sched = JoinAdmission([10], [[5, 5]], 100, arena_pages=100, page_tokens=16,
                          frame_tokens=[3], canvas_tokens=4)
    assert sched.stages == [[9, 9]]
    assert sched.frame_rows == [7]

    paged = fake_pipeline(needs_pages=True)
    arena = cpu_arena(64)
    arena.retention_cap_pages = 8
    stream = loop.FilterStream(
        fake_torch(), arena, paged, answers, docs[:1], questions[:1],
        200, arena_writes=False, arena_keys=[("d", 0)])
    assert stream.arena_writes
    assert arena.retention_cap_pages == 8
    assert stream.sched.pages

    # pack_chunk appends the canvas rows after each suffix
    torch = cpu_staging(monkeypatch)
    arena = cpu_arena(64)
    canvas = (90, 91, 92, 93)
    groups = [dict(key=("d", 0), prefix=[1, 2, 3], f=3, suffixes=[[10, 11]]),
              dict(key=("d", 1), prefix=[4, 5], f=2, suffixes=[[12]])]
    chunk = loop.pack_chunk(torch, arena, groups, attention_mode="unified",
                            canvas=canvas)
    assert chunk.tokens == 3 + 2 + 4 + 2 + 1 + 4
    assert chunk.input_ids.tolist() == [1, 2, 3, 10, 11, *canvas, 4, 5, 12, *canvas]
    assert chunk.positions.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6]
    # the answer row is the first canvas row of each group
    assert chunk.final_indices.tolist() == [5, 12]
    meta = chunk.meta
    assert meta["cu_a"].tolist() == [0, 9, 16]
    assert meta["unified"] is None
    assert meta["canvas"]["rows"].tolist() == [5, 6, 7, 8, 12, 13, 14, 15]
    assert meta["canvas"]["cu_q"].tolist() == [0, 4, 8]
    assert meta["canvas"]["max_q"] == 4

    later = loop.pack_chunk(torch, arena, groups, attention_mode="unified",
                            canvas=canvas, answer_row=2)
    assert later.final_indices.tolist() == [7, 14]
    with pytest.raises(ValueError, match="answer_row"):
        loop.pack_chunk(torch, arena, groups, attention_mode="unified",
                        canvas=canvas, answer_row=4)

    plain = loop.pack_chunk(torch, arena, groups, attention_mode="unified")
    assert plain.meta["canvas"] is None
    assert plain.final_indices.tolist() == [4, 7]
    with pytest.raises(ValueError, match="unified"):
        loop.pack_chunk(torch, arena, groups, attention_mode="tree",
                        canvas=canvas)


class _Norm:
    hidden_size = 4
    variance_epsilon = 1e-6

    def __init__(self, scale):
        self.scale = scale
        self.has_weight = True
        self.weight = scale


class _Linear:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x):
        return x @ self.weight.T, None


class _Attention:
    def __init__(self, torch, hidden, heads, kv_heads, dim, sliding):
        self.num_heads, self.num_kv_heads, self.head_dim = heads, kv_heads, dim
        self.q_size, self.kv_size = heads * dim, kv_heads * dim
        width = self.q_size + 2 * self.kv_size
        self.qkv_proj = _Linear(torch.ones(width, hidden))
        self.o_proj = _Linear(torch.ones(hidden, self.q_size))
        self.q_norm = self.k_norm = _Norm(1.0)
        self.v_norm = _Norm(1.0)
        self.v_norm.has_weight = False
        self.v_norm.hidden_size = dim
        self.rotary_emb = SimpleNamespace(
            cos_sin_cache=torch.zeros(2, dim), is_neox_style=True)
        self.is_sliding = sliding


class _Layer:
    def __init__(self, torch, hidden, sliding, moe):
        heads, kv, dim = (2, 1, 4) if sliding else (2, 1, 8)
        self.self_attn = _Attention(torch, hidden, heads, kv, dim, sliding)
        for name in ("input_layernorm", "pre_feedforward_layernorm",
                     "post_feedforward_layernorm", "post_feedforward_layernorm_1",
                     "pre_feedforward_layernorm_2", "post_feedforward_layernorm_2"):
            setattr(self, name, _Norm(torch.tensor(1.0)))
        self.post_attention_layernorm = _Norm(torch.tensor(0.0))
        self.mlp = _Mlp(torch, hidden)
        self.enable_moe_block = moe
        self.router = SimpleNamespace(
            norm=_Norm(torch.tensor(1.0)), root_size=torch.tensor(1.0),
            scale=torch.ones(hidden), proj=lambda x: (x, None))
        self.moe = SimpleNamespace(experts=lambda x, logits: x)
        self.layer_scalar = torch.tensor([0.5])


class _Mlp:
    def __init__(self, torch, hidden):
        self.gate_up_proj = _Linear(torch.eye(hidden))
        self.down_proj = _Linear(torch.eye(hidden))

    def __call__(self, x):
        return x


class _Engine:
    def __init__(self, arena, **kwargs):
        self.calls = []
        self.torch = sys.modules["torch"]
        self.is_fp8 = kwargs["fp8"]

    def attention_unified(self, q3, k3, v3, meta, *, softmax_scale=None,
                          window=None):
        self.calls.append((meta["layer"], q3.shape, k3.shape, softmax_scale,
                           window))
        meta["layer"] += 1
        return self.torch.zeros(q3.shape[0], q3.shape[1] * q3.shape[2])

    def norm_rows(self, x, weight, eps):
        return x * weight

    def norm_quant_rows(self, x, weight, eps, residual=None):
        if residual is not None:
            residual.add_(x)
            x = residual
        return x * weight, None

    def fp8_linear(self, module, x_q, x_s):
        return module(x_q)[0]

    def fused_add_rms_norm(self, hidden, residual, norm):
        residual.add_(hidden)
        hidden.copy_(residual * norm.weight)
        return hidden, residual

    def qkv_norm_rope_heads(self, qkv, positions, *, n_q, n_kv, head_dim,
                            q_weight, k_weight, eps, cos_sin_cache):
        q = qkv[:, :n_q * head_dim] * q_weight
        k = qkv[:, n_q * head_dim:(n_q + n_kv) * head_dim] * k_weight
        return (q.contiguous(), k.contiguous(),
                qkv[:, (n_q + n_kv) * head_dim:].contiguous())

    def gelu_mul_quant(self, gate_up):
        return gate_up, None

    def scale_add_norm_quant(self, x, residual, scale, weight, eps):
        residual.mul_(scale).add_(x)
        return residual * weight, None

    def norm_router_quant(self, x, weight1, weight2, eps):
        return x * weight1, None, x * weight2


def _spec_for(model, **overrides):
    attn = model.model.layers[0].self_attn
    fields = dict(vocab=50, canvas_tokens=3, canvas_answer_row=1,
                  sliding_window=model.model.config.sliding_window,
                  n_q=attn.num_heads, n_kv=attn.num_kv_heads,
                  head_dim=attn.head_dim, d_head=attn.head_dim,
                  widest_projection=32)
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _fake_model(torch, hidden=4):
    layers = [_Layer(torch, hidden, sliding=True, moe=True),
              _Layer(torch, hidden, sliding=False, moe=False)]
    backbone = SimpleNamespace(
        layers=layers,
        embed_tokens=lambda ids: torch.ones(ids.shape[0], hidden),
        normalizer=torch.tensor(2.0),
        norm=_Norm(1.0),
        config=SimpleNamespace(sliding_window=1024))
    post_norm = _Norm(0.25)
    post_norm.has_weight = False
    return SimpleNamespace(
        model=backbone,
        self_conditioning=SimpleNamespace(post_norm=post_norm),
        quail_vllm_config="config")


def test_pipeline_layer_order_and_one_row_canvas_attention(monkeypatch):
    torch = pytest.importorskip("torch")
    seen = _vllm_stubs(monkeypatch)
    model = _fake_model(torch)
    spec = _spec_for(model)
    pipeline = DiffusionGemmaPipeline(model, None, spec=spec, engine_class=_Engine)
    for layer in model.model.layers:
        assert layer.post_feedforward_layernorm.weight.item() == 0.5
        assert layer.post_attention_layernorm.weight.item() == 0.0
        assert layer.input_layernorm.weight.item() == 1.0
    assert model.quail_scalars_folded
    assert len(pipeline.canvas_ids) == 3
    assert pipeline.canvas_answer_row == 1
    with pytest.raises(ValueError, match="geometry"):
        DiffusionGemmaPipeline(
            _fake_model(torch), None, engine_class=_Engine,
            spec=_spec_for(_fake_model(torch), sliding_window=512))
    assert seen["workspace"] == "cuda"
    assert not pipeline.tree_attention and not pipeline.gemm_warmup
    assert pipeline.max_chunk_tokens == (2**31 - 1) // 32

    rows = torch.tensor([3, 4, 5])
    chunk = SimpleNamespace(
        input_ids=torch.zeros(6, dtype=torch.int64),
        positions=torch.arange(6),
        final_indices=torch.tensor([0, 3]),
        meta={"layer": 0, "canvas": {"rows": rows}})
    out = pipeline.forward_chunk(chunk)
    assert seen["context"] == ("config", 6)
    calls = pipeline.engine.calls
    assert calls == [(0, (6, 2, 4), (6, 1, 4), 1.0, 1024),
                     (1, (6, 2, 8), (6, 1, 8), 1.0, None)]
    # every row starts at 2 and attention adds 0; the MoE layer adds the
    # residual twice and halves (2 -> 3), the dense layer once (3 -> 3)
    assert out.tolist() == [[3.0] * 4, [3.0] * 4]
    DiffusionGemmaPipeline(model, None, spec=spec, engine_class=_Engine)
    assert model.model.layers[1].post_feedforward_layernorm.weight.item() == 0.5

    # a one-row canvas skips the second attention call
    from quail.backends.quail.executor.attention import Engine

    cpu_staging(monkeypatch)
    for canvas, calls in [((90,), 1), ((90, 91), 2)]:
        arena = cpu_arena(64)
        groups = [dict(key=("d", 0), prefix=[1, 2, 3], f=3, suffixes=[[10, 11]])]
        chunk = loop.pack_chunk(torch, arena, groups, attention_mode="unified",
                                canvas=canvas)
        engine = Engine.__new__(Engine)
        engine.arena = arena
        engine.torch = torch
        seen = []

        def fa(q, k, v, cu_q, cu_k, max_q, max_k, causal, **kwargs):
            seen.append((q.shape[0], causal))
            return torch.zeros(q.shape[0], 2, 4), None

        monkeypatch.setattr(engine, "_fa", fa)
        rows = chunk.tokens
        q3 = torch.zeros(rows, 2, 4)
        meta = dict(chunk.meta, layer=0)
        out = engine.attention_unified(q3, q3, q3, meta)
        assert out.shape == (rows, 8), canvas
        assert len(seen) == calls, canvas
        assert seen[0] == (rows, True), canvas
        if calls == 2:
            assert seen[1] == (len(canvas), False)


def _vllm_stubs(monkeypatch):
    monkeypatch.setattr(
        "quail.backends.quail.executor.models.diffusion_gemma.FP8Experts",
        lambda module, engine: lambda x, scale, logits: module(x, logits))
    seen = {}
    context = types.ModuleType("vllm.forward_context")

    @contextlib.contextmanager
    def set_forward_context(attn_metadata, vllm_config, num_tokens=None):
        seen["context"] = (vllm_config, num_tokens)
        yield

    context.set_forward_context = set_forward_context
    workspace = types.ModuleType("vllm.v1.worker.workspace")
    workspace.is_workspace_manager_initialized = lambda: "workspace" in seen
    workspace.init_workspace_manager = lambda device: seen.setdefault(
        "workspace", str(device))
    monkeypatch.setitem(sys.modules, "vllm.forward_context", context)
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", types.ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker",
                        types.ModuleType("vllm.v1.worker"))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.workspace", workspace)
    return seen


def test_tuned_moe_configs_merge_upstream_or_fall_back(tmp_path):
    import json

    from quail.backends.quail.executor.moe_configs import TUNED, write_configs

    base = tmp_path / "vllm"
    base.mkdir()
    upstream = {"16384": {"BLOCK_SIZE_M": 64}, "32768": {"BLOCK_SIZE_M": 16}}
    for name in TUNED:
        (base / name).write_text(json.dumps(upstream))
    folder = write_configs(tmp_path / "configs", base)
    for name, overrides in TUNED.items():
        table = json.loads((folder / name).read_text())
        assert table["16384"] == upstream["16384"]
        assert table["32768"] == overrides["32768"]
        assert table["65536"] == overrides["65536"]
        (folder / name).write_text("{}")
    write_configs(folder, tmp_path / "missing")
    assert list(folder.glob("*.json")) == []
