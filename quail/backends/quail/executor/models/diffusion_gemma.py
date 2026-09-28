"""Run one DiffusionGemma forward pass with Quail's batching and KV.

Each prompt ends with a fixed canvas token. Its embedding receives the
self-conditioning norm with zero conditioning, as in the first denoising
step. The answer comes from TRUE/FALSE scores at that position.
"""

import numpy as np

from quail.backends.quail.executor.attention import Engine
from quail.backends.quail.executor.models.base import ModelPipeline
from quail.backends.quail.executor.moe import FP8Experts
from quail.cost import budgets

# The canvas starts as random token ids, as vLLM's sampler starts it;
# one fixed draw serves every document so answers are reproducible.
CANVAS_SEED = 0


def canvas_token_ids(vocab: int, tokens: int, seed: int = CANVAS_SEED) -> tuple:
    """The fixed random token ids that fill every canvas."""
    rng = np.random.default_rng(seed)
    return tuple(int(i) for i in rng.integers(0, vocab, tokens))


def _init_moe_workspace():
    """Give vLLM's fused MoE kernels the scratch buffers its worker would.

    The buffers grow to the largest chunk on first use and stay
    allocated; nothing locks them.
    """
    import torch
    from vllm.v1.worker import workspace

    if not workspace.is_workspace_manager_initialized():
        workspace.init_workspace_manager(torch.device("cuda"))


class DiffusionGemmaPipeline(ModelPipeline):
    """Forward passes for DiffusionGemma checkpoints loaded by vLLM.

    The full-attention layers' 512-wide heads run FlashAttention 4
    over arena pages, so every chunk carries pages.

    Args:
        model: vLLM's DiffusionGemmaForConditionalGeneration module.
        arena: KVArena with one pool per layer at that layer's KV
            geometry (spec.kv_shapes).
        spec: The ModelSpec, for the vocabulary size and canvas length.
        engine_class: Engine class, replaceable by tests.
    """

    needs_pages = True
    # every chunk runs one causal call; the tree path's merge writes
    # the engine's fp8 GEMM inputs, which this model does not use
    tree_attention = False
    gemm_warmup = False
    # Partial join batches also need compiled expert kernels at these sizes.
    warm_tokens = ModelPipeline.warm_tokens + (4096, 8192, 16384, 32768)

    def __init__(self, model, arena, *, spec, engine_class=Engine):
        backbone = model.model
        self.layers = backbone.layers
        self.embed = backbone.embed_tokens
        self.normalizer = backbone.normalizer
        self.final_norm = backbone.norm
        self.canvas_norm = model.self_conditioning.post_norm
        self.vllm_config = model.quail_vllm_config
        self.window = spec.sliding_window
        attn = self.layers[0].self_attn
        loaded = (backbone.config.sliding_window, attn.num_heads,
                  attn.num_kv_heads, attn.head_dim)
        expected = (spec.sliding_window, spec.n_q, spec.n_kv, spec.d_head)
        if loaded != expected:
            raise ValueError(
                f"the loaded model's geometry (window, q heads, kv heads, "
                f"head dim) {loaded} differs from the spec's {expected}")
        # the engine's fp8 GEMM path is not used: the projections run
        # vLLM's own fp8 linear modules
        self.engine = engine_class(
            arena, n_q=attn.num_heads, n_kv=attn.num_kv_heads,
            head_dim=attn.head_dim, rotary=attn.rotary_emb, fp8=False)
        self._ones = {}
        for layer in self.layers:
            rope = layer.self_attn.rotary_emb
            # the rotary kernel reads the cache at the activations' dtype
            rope.cos_sin_cache = rope.cos_sin_cache.to(self.engine.torch.bfloat16)
            router = layer.router
            # the router's constant scale and learned per-dimension scale
            # fold into one vector
            router.quail_scale = (router.root_size.to(router.scale.dtype)
                                  * router.scale).detach()
        if any(layer.self_attn.v_norm.has_weight for layer in self.layers):
            raise ValueError("the fused qkv kernel expects a weightless v norm")
        self._fold_layer_scalars(model)
        self.scales = [float(layer.layer_scalar) for layer in self.layers]
        self.canvas_ids = canvas_token_ids(spec.vocab, spec.canvas_tokens)
        # an empty canvas reads the answer at the prompt's last row
        self.canvas_answer_row = (spec.canvas_answer_row
                                  if spec.canvas_tokens else 0)
        _init_moe_workspace()
        self.max_chunk_tokens = budgets.kernel_index_cap(spec)
        self.experts = [FP8Experts(layer.moe.experts, self.engine)
                        if layer.enable_moe_block else None
                        for layer in self.layers]

    def _norm(self, x, module):
        """One of the model's RMS norms, through the engine's kernel."""
        # The loaded module dispatches to unfused PyTorch on this vLLM version.
        weight = module.weight if module.has_weight else self._ones_like(module, x)
        return self.engine.norm_rows(x, weight, module.variance_epsilon)

    def _ones_like(self, module, x):
        key = (module.hidden_size, x.dtype)
        if key not in self._ones:
            self._ones[key] = self.engine.torch.ones(
                module.hidden_size, dtype=x.dtype, device=x.device)
        return self._ones[key]

    def _fold_layer_scalars(self, model):
        """Fold each layer's output scalar into its post-feedforward norm.

        Layer l multiplies its output, the norm of its feedforward
        branch plus the residual, by s_l. Scaling that norm's weight by
        s_l leaves only the residual to scale, which the fused path does
        in place. The scalars are far below one (0.07 to 0.5), so a
        cumulative fold across layers would overflow the weights.
        """
        if getattr(model, "quail_scalars_folded", False):
            return
        for layer in self.layers:
            layer.post_feedforward_layernorm.weight.data.mul_(
                float(layer.layer_scalar))
        model.quail_scalars_folded = True

    def forward_chunk(self, chunk):
        hidden = self.backbone_rows(chunk)
        return self._norm(hidden.index_select(0, chunk.final_indices),
                          self.final_norm)

    def backbone_rows(self, chunk):
        """Every row's hidden state after the last layer, before the final norm."""
        from vllm.forward_context import set_forward_context

        meta = chunk.meta
        meta["layer"] = 0
        n = chunk.input_ids.shape[0]
        hidden = self.embed(chunk.input_ids) * self.normalizer
        canvas = meta.get("canvas")
        if canvas is not None:
            rows = canvas["rows"]
            hidden.index_copy_(
                0, rows, self._norm(hidden.index_select(0, rows), self.canvas_norm))
        # the fused MoE kernels look their layer up in the forward
        # context
        with set_forward_context(None, self.vllm_config, num_tokens=n):
            return self._layers(hidden, chunk.positions, meta)

    # ---- the layer stack --------------------------------------------
    # vLLM's layer arithmetic with the residual adds, the norms
    # that feed a linear, and that linear's input quantization fused,
    # and the per-head q and k norms fused with the rotary. The layer
    # scalars are folded into the post-feedforward norm weights at
    # build time.

    def _norm_quant(self, x, norm):
        return self.engine.norm_quant_rows(x, norm.weight,
                                           norm.variance_epsilon)

    def _layers(self, hidden, positions, meta):
        engine = self.engine
        residual = hidden
        x_q, x_s = self._norm_quant(residual, self.layers[0].input_layernorm)
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            attn = layer.self_attn
            qkv = engine.fp8_linear(attn.qkv_proj, x_q, x_s)
            out = self._attention_fused(attn, qkv, positions, meta)
            out = self._norm(out, layer.post_attention_layernorm)
            # residual becomes the attention sum; the feedforward reads
            # its norm
            pre = layer.pre_feedforward_layernorm
            x_q, x_s = engine.scale_add_norm_quant(
                out, residual, 1.0, pre.weight, pre.variance_epsilon)
            mlp = layer.mlp
            gate_up = engine.fp8_linear(mlp.gate_up_proj, x_q, x_s)
            d_q, d_s = engine.gelu_mul_quant(gate_up)
            dense = engine.fp8_linear(mlp.down_proj, d_q, d_s)
            if layer.enable_moe_block:
                dense = self._norm(dense, layer.post_feedforward_layernorm_1)
                router = layer.router
                pre = layer.pre_feedforward_layernorm_2
                # the expert input and the router input are two norms
                # of the residual
                routed_in, routed_scale, router_in = engine.norm_router_quant(
                    residual, pre.weight, router.quail_scale,
                    pre.variance_epsilon)
                logits, _ = router.proj(router_in)
                routed = self.experts[index](routed_in, routed_scale, logits)
                routed = self._norm(routed, layer.post_feedforward_layernorm_2)
                # dense becomes the norm of the two branches' sum
                engine.fused_add_rms_norm(dense, routed,
                                          layer.post_feedforward_layernorm)
            else:
                dense = self._norm(dense, layer.post_feedforward_layernorm)
            # dense carries the layer scalar through its norm weight;
            # the residual takes it in the kernel that sums them
            scale = self.scales[index]
            if index == last:
                return dense + residual * scale
            # residual becomes the layer's output; the next layer reads
            # its norm
            next_norm = self.layers[index + 1].input_layernorm
            x_q, x_s = engine.scale_add_norm_quant(
                dense, residual, scale, next_norm.weight,
                next_norm.variance_epsilon)
        raise AssertionError("a model needs at least one layer")

    def _attention_fused(self, attn, qkv, positions, meta):
        n = qkv.shape[0]
        H, KH, D = attn.num_heads, attn.num_kv_heads, attn.head_dim
        rope = attn.rotary_emb
        q, k, v = self.engine.qkv_norm_rope_heads(
            qkv, positions, n_q=H, n_kv=KH, head_dim=D,
            q_weight=attn.q_norm.weight, k_weight=attn.k_norm.weight,
            eps=attn.q_norm.variance_epsilon, cos_sin_cache=rope.cos_sin_cache)
        q3, k3, v3 = q.view(n, H, D), k.view(n, KH, D), v.view(n, KH, D)
        out, _ = attn.o_proj(self._attend(attn, q3, k3, v3, meta))
        return out

    def _attend(self, attn, q3, k3, v3, meta):
        """One layer's attention: the unified paged call."""
        window = self.window if attn.is_sliding else None
        return self.engine.attention_unified(q3, k3, v3, meta,
                                             softmax_scale=1.0, window=window)
