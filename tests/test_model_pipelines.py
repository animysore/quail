"""The per-architecture forward passes, their attention paths, and support checks."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from quail.backends.quail.backend import QuailBackend
from quail.backends.quail.executor.models import build_pipeline, supported_archs
from quail.backends.quail.executor.models.qwen3 import Qwen3Pipeline
from quail.backends.request import RequestBackend
from quail.specs import H100_SXM, MODELS, QWEN3_4B_FP8, QWEN3_RERANKER_0_6B_BF16


def test_backends_support_registered_generative_models_and_refuse_others():
    for model in MODELS.values():
        assert model.arch in supported_archs()
        assert QuailBackend().supports(model, H100_SXM, 1).supported

    spec = replace(QWEN3_4B_FP8, name="other", arch="other")
    with pytest.raises(ValueError, match="other"):
        build_pipeline(spec, model=None, arena=None)
    support = QuailBackend().supports(spec, H100_SXM, 1)
    assert not support.supported
    assert "other" in support.reason

    engine = SimpleNamespace(label="stock vLLM", kind="vllm")
    backend = RequestBackend(name="stock", engine=engine, filter_submission="operator")
    assert backend.supports(QWEN3_4B_FP8, H100_SXM, 1).supported
    support = backend.supports(QWEN3_RERANKER_0_6B_BF16, H100_SXM, 1)
    assert not support.supported
    assert "generative" in support.reason


def _model(torch, dtype):
    weight = SimpleNamespace(dtype=dtype, shape=(4096, 1024))
    attn = SimpleNamespace(num_heads=16, num_kv_heads=8, head_dim=128,
                           rotary_emb=None, qkv_proj=SimpleNamespace(weight=weight))
    mlp = SimpleNamespace(gate_up_proj=SimpleNamespace(weight=weight))
    layer = SimpleNamespace(self_attn=attn, mlp=mlp)
    return SimpleNamespace(model=SimpleNamespace(
        layers=[layer], embed_tokens=None, norm=None))


def test_tree_attention_follows_the_weight_precision():
    torch = pytest.importorskip("torch")
    for dtype_name, expected in [("float8_e4m3fn", True), ("bfloat16", False)]:
        engines = []

        def engine(arena, **kwargs):
            engines.append(kwargs)
            return SimpleNamespace(**kwargs)

        pipeline = Qwen3Pipeline(_model(torch, getattr(torch, dtype_name)), None,
                                 spec=None, engine_class=engine)
        assert engines[0]["fp8"] is expected, dtype_name
        assert pipeline.tree_attention is expected, dtype_name
        assert pipeline.max_chunk_tokens == (2**31 - 1) // 4096, dtype_name
