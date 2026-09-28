"""Qwen3 forward pass over vLLM's loaded Qwen3 module.

Layer order: input RMSNorm, fused QKV projection, per-head Q and K
RMSNorm with rotary, attention, output projection, post-attention
RMSNorm, gate_up projection, SiLU-gated product, down projection.
The Qwen3 rerankers are this architecture with bf16 weights and run
through this file too.
"""

from quail.backends.quail.executor.attention import Engine
from quail.backends.quail.executor.models.base import ModelPipeline


class Qwen3Pipeline(ModelPipeline):
    """Forward passes for Qwen3 checkpoints loaded by vLLM."""

    def __init__(self, model, arena, *, spec, kernels="quail",
                 engine_class=Engine):
        import torch

        self.layers = model.model.layers
        self.embed = model.model.embed_tokens
        self.final_norm = model.model.norm
        attn = self.layers[0].self_attn
        fp8 = attn.qkv_proj.weight.dtype == torch.float8_e4m3fn
        self.engine = engine_class(
            arena, n_q=attn.num_heads, n_kv=attn.num_kv_heads,
            head_dim=attn.head_dim, rotary=attn.rotary_emb,
            fp8=fp8, kernels=kernels)
        # the tree path's merge ends in an fp8 quant; a bf16 model
        # (the rerankers) runs every chunk through unified attention
        self.tree_attention = fp8
        # The fused kernels compute element offsets in 32-bit ints,
        # so a chunk needs rows x widest_row < 2^31.
        widest = max(max(layer.self_attn.qkv_proj.weight.shape[0],
                         layer.mlp.gate_up_proj.weight.shape[0])
                     for layer in self.layers)
        self.max_chunk_tokens = (2**31 - 1) // widest

    def linears(self):
        # every layer has the same four shapes
        layer = self.layers[0]
        return (layer.self_attn.qkv_proj, layer.self_attn.o_proj,
                layer.mlp.gate_up_proj, layer.mlp.down_proj)

    def forward_chunk(self, chunk):
        engine = self.engine
        meta = chunk.meta
        meta["layer"] = 0
        positions = chunk.positions
        hidden = self.embed(chunk.input_ids)
        residual = None
        for layer in self.layers:
            attn = layer.self_attn
            if residual is None:
                residual = hidden
                q_in, q_scale = engine.norm_quant(hidden, layer.input_layernorm)
            else:
                q_in, q_scale = engine.norm_quant(
                    hidden, layer.input_layernorm, residual)
            qkv = engine.gemm(q_in, q_scale, attn.qkv_proj)
            q, k = engine.qk_norm_rope(qkv, positions, attn)
            v = qkv[:, (engine.num_q_heads + engine.num_kv_heads)
                    * engine.head_dim:]
            o_in, o_scale = engine.attention(q, k, v, chunk)
            hidden = engine.gemm(o_in, o_scale, attn.o_proj)
            g_in, g_scale = engine.norm_quant(
                hidden, layer.post_attention_layernorm, residual)
            gate_up = engine.gemm(g_in, g_scale, layer.mlp.gate_up_proj)
            d_in, d_scale = engine.activation_quant(gate_up)
            hidden = engine.gemm(d_in, d_scale, layer.mlp.down_proj)
        final = chunk.final_indices
        last_hidden = hidden.index_select(0, final)
        last_residual = residual.index_select(0, final)
        normed, _ = engine.fused_add_rms_norm(
            last_hidden, last_residual, self.final_norm)
        return normed
