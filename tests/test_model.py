"""CPU tests for model files, retained answer weights, and readouts."""

import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import cpu_arena, fake_pipeline, fake_torch

from quail.backends.quail.executor import loop, model
from quail.backends.quail.executor.model import answer_weights, retain_answer_head
from quail.backends.quail.executor.readout import AnswerRows, AsyncAnswers, AsyncScores
from quail.builtins import built_in_registry


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def _model(torch, tied=False):
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(8, 4, dtype=torch.bfloat16)
    model.lm_head = (model.model.embed_tokens if tied else
                     torch.nn.Linear(4, 8, bias=False, dtype=torch.bfloat16))
    return model


def test_retained_answer_weights_ownership_and_answer_rows(torch):
    for tied in [False, True]:
        model = _model(torch, tied)
        embedding = model.model.embed_tokens.weight
        reference = weakref.ref(model.lm_head.weight)
        expected = model.lm_head.weight.detach()[[1, 3, 5]].clone()
        retain_answer_head(torch, model, [5, 1, 3, 1])
        gc.collect()
        assert model.lm_head is None
        assert (reference() is not None) == tied
        assert model.model.embed_tokens.weight is embedding
        assert model.quail_answer_token_ids == (1, 3, 5)
        assert torch.equal(model.quail_answer_weights, expected)
        assert (model.quail_answer_weights.untyped_storage().nbytes()
                == expected.numel() * 2)
        assert not model.quail_answer_weights.requires_grad

    model = _model(torch)
    retain_answer_head(torch, model, [1, 3, 5])
    weights = answer_weights(model, [1, 3, 5])
    retain_answer_head(torch, model, [5, 3, 1])
    assert answer_weights(model, [1, 3, 5]) is weights
    with pytest.raises(ValueError, match="differ"):
        answer_weights(model, [1, 3, 6])
    assert answer_weights(model, [1, 3, 5]) is weights

    model = _model(torch)
    original = model.lm_head
    with pytest.raises(ValueError, match="empty"):
        retain_answer_head(torch, model, [])
    assert model.lm_head is original

    # answer rows preserve the scores and share the retained weights
    model = _model(torch)
    hidden = torch.tensor([[1, 2, -1, 0], [0, 1, 3, -2]], dtype=torch.bfloat16)
    full_scores = torch.nn.functional.linear(hidden, model.lm_head.weight)
    expected = (full_scores[:, [1, 3]].amax(1) > full_scores[:, 5]).int().tolist()
    retain_answer_head(torch, model, [1, 3, 5])
    cpu_torch = SimpleNamespace(tensor=lambda values, **kwargs: torch.tensor(values))

    def tokenizer(word, **kwargs):
        return {"input_ids": [
            5 if "false" in word.lower() else 3 if word.startswith(" ") else 1]}

    first = AnswerRows.from_tokenizer(cpu_torch, torch.nn.functional, model,
                                      tokenizer)
    second = AnswerRows(cpu_torch, torch.nn.functional, model, [3, 1], [5])
    assert first.weights is second.weights is model.quail_answer_weights
    assert first(hidden) == second(hidden) == expected
    assert AsyncAnswers.dtype is None
    assert AsyncScores.dtype is np.float32


def test_run_join_evicts_then_halves_a_chunk_that_does_not_fit(monkeypatch):
    sizes = []
    failed = []
    modes = set()

    def pack(torch, arena, specs, **kw):
        if len(specs) > 1 and not failed:
            failed.append(len(specs))
            raise loop.ArenaFullError("unified suffix pages exceed the free KV arena")
        sizes.append(len(specs))
        modes.add(kw["attention_mode"])
        return SimpleNamespace(specs=specs, tokens=len(specs),
                               attention_mode=kw["attention_mode"],
                               temporary_keys=(), fresh_keys=())

    monkeypatch.setattr(loop, "pack_chunk", pack)
    arena = cpu_arena(64)
    evictions = []
    monkeypatch.setattr(arena, "evict_retained",
                        lambda need: evictions.append(need) or ())
    pipeline = fake_pipeline(forward_chunk=lambda chunk: [1] * len(chunk.specs))
    answers = SimpleNamespace(submit=lambda v: v, result=lambda v: v, dtype=None)
    answers_out, _, _ = loop.run_join(fake_torch(), arena, pipeline, answers,
                                      [[1] * 8, [2] * 8], [[[3, 4]]], 64,
                                      anchor_keys=[("a", 0), ("a", 1)])
    # nothing retained to evict, so the two-group chunk ran as two chunks
    assert failed == [2] and len(evictions) == 1 and evictions[0] > 0
    assert sizes == [1, 1]
    assert modes == {"tree"}
    assert answers_out == [{0: [1], 1: [1]}]


@pytest.fixture
def clear_model_paths():
    model.resolve_model_path.cache_clear()
    yield
    model.resolve_model_path.cache_clear()


def test_model_files_are_resolved_once_and_cached_before_children_start(
        monkeypatch, tmp_path, clear_model_paths):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / kwargs["revision"])

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    first = model.resolve_model_path("Qwen/model", "revision-a")
    assert model.resolve_model_path("Qwen/model", "revision-a") == first
    assert model.resolve_model_path("Qwen/model", "revision-b") != first
    assert [call["revision"] for call in calls] == ["revision-a", "revision-b"]
    assert all(call["repo_id"] == "Qwen/model" for call in calls)
    assert "*.safetensors" in calls[0]["allow_patterns"]

    # the files are cached before the GPU children start
    from quail.backends.quail import worker

    events = []
    registry = built_in_registry()

    def resolve(name, revision):
        events.append("resolve")
        assert revision == registry.model("qwen3-4b-fp8").revision
        return str(tmp_path)

    def ensure(count):
        events.append("children")
        assert count == 8

    def round_fn(kind, subs):
        events.append("boot")
        assert kind == "boot"
        assert len(subs) == 8
        assert all(sub["registry"] is registry
                   and sub["model_path"] == str(tmp_path) for sub in subs)
        return [{"boot_s": 3.0, "kind": "cold"}] * 8

    def run_graph(*args):
        assert events == ["resolve", "children", "boot"]
        events.append("query")
        return {"_outputs": {}, "boot_s": 0.0, "wall_s": 7.0}

    clock = iter([10.0, 12.0, 15.0])
    monkeypatch.setattr(worker.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(worker, "resolve_model_path", resolve)
    monkeypatch.setattr(worker, "_ensure_children", ensure)
    monkeypatch.setattr(worker, "_round", round_fn)
    monkeypatch.setattr(worker, "execute_distributed_graph", run_graph)
    result = worker.execute_quail_multi({
        "docs": {}, "workers": 8, "model": "qwen3-4b-fp8",
        "physical_plan": {"device": "h100-sxm"},
        "chunk_tokens": 100, "true_ids": [1], "false_ids": [2],
    }, registry, object())
    assert events == ["resolve", "children", "boot", "query"]
    assert result.metrics["boot_s"] == 5.0
    assert result.metrics["boot"]["model_files_s"] == 2.0
    assert result.metrics["wall_s"] == 7.0
