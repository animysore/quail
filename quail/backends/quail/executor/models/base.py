"""The forward pass one model family runs over the shared engine.

A model pipeline owns the forward loop for one architecture: where the
modules sit in vLLM's loaded model, how the packed rows are embedded,
the per-layer block, and the final norm. It calls the Engine for the
kernels, quantization, attention paths, and arena access. The chunk
loop sees only this class.
"""


class ModelPipeline:
    """The contract between the chunk loop and one model family.

    A subclass sets self.engine and self.max_chunk_tokens in __init__
    and implements forward_chunk; with gemm_warmup it also implements
    linears. tree_attention says the model's kernels can run the
    tree attention path (two calls, fused merge and fp8 quant); a
    chunk asked for it on a model without it runs unified, the one
    causal call every model has. A diffusion model also sets
    canvas_ids: the token ids the loop packs after every suffix, with
    the answer at canvas_answer_row of them. needs_pages says every
    chunk must carry arena pages, for a model whose attention kernel
    reads paged KV only. warm_tokens are the chunk sizes the warmup
    builds one small chunk of: a gated chain's trailing chunks are
    100-500 tokens, a shape the full-size warm chunks do not cover.
    """

    engine = None
    max_chunk_tokens = None
    warm_tokens = (64, 128, 256, 512, 1024, 2048)
    canvas_ids = ()
    canvas_answer_row = 0
    needs_pages = False
    tree_attention = True
    gemm_warmup = True

    def forward_chunk(self, chunk):
        """Return the final-normed hidden state of chunk.final_indices.

        The loop owns the chunk's pages, temporaries included; this
        only computes.
        """
        raise NotImplementedError

    def linears(self):
        """One linear module per distinct GEMM shape, for kernel warm-up."""
        raise NotImplementedError
