"""Booting a GPU for a query, reusing it, and releasing it for another model."""

import contextlib
import sys
import types
from types import SimpleNamespace

import pytest

from quail.backends.quail import worker
from quail.backends.quail.worker import LoadedGpu
from quail.builtins import built_in_registry


def install_fake_torch(monkeypatch):
    functional = types.ModuleType("torch.nn.functional")
    functional.linear = lambda a, b: a
    nn = types.ModuleType("torch.nn")
    nn.functional = functional
    torch = types.ModuleType("torch")
    torch.nn = nn
    torch.bfloat16 = "bfloat16"
    torch.ones = lambda *args, **kwargs: SimpleNamespace()
    torch.inference_mode = contextlib.nullcontext
    torch.cuda = SimpleNamespace(mem_get_info=lambda: (40 * 2**30, 80 * 2**30),
                                 current_blas_handle=lambda: 1,
                                 synchronize=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.nn", nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", functional)


@pytest.fixture
def booted(monkeypatch):
    install_fake_torch(monkeypatch)
    monkeypatch.setattr(worker, "set_gpu_index", lambda index: None)
    monkeypatch.setattr(worker, "say", lambda message: None)
    for name in ("load_model", "build_pipeline", "AnswerRows", "KVArena",
                 "AsyncAnswers"):
        monkeypatch.setattr(worker, name, lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(worker, "warm_kernels",
                        lambda *a, **k: {"tier": "compiled"})


ENVELOPE = {
    "backend": "quail", "model": "qwen3-4b-fp8", "device": "h100-sxm",
    "workers": 1,
    "settings": {"chunk_tokens": 8192, "true_ids": [1], "false_ids": [2]},
}
PAYLOAD = {"physical_plan": ENVELOPE, "model": "qwen3-4b-fp8", "workers": 1,
           "docs": {}, **ENVELOPE["settings"]}


def test_prepared_boot_is_handed_to_the_query_and_used_once(
        booted, monkeypatch):
    registry = built_in_registry()
    backend = registry.backend("quail")
    runtime_state = {}
    boots = []
    real_boot_for_query = worker._boot_for_query

    def counting_boot(*args, **kwargs):
        boots.append(1)
        return real_boot_for_query(*args, **kwargs)

    monkeypatch.setattr(worker, "_boot_for_query", counting_boot)
    monkeypatch.setattr(worker, "execute_single",
                        lambda *a, **k: {"_outputs": {}, "wall_s": 1.0})

    worker.prepare_quail_request(SimpleNamespace(
        gpu_count=1, registry=registry, runtime_state=runtime_state,
        request=SimpleNamespace(plan=ENVELOPE),
    ))

    gpu = runtime_state[("quail", "qwen3-4b-fp8")]
    assert gpu.prepared_boot["kind"] == "cold"
    assert len(boots) == 1

    response = worker.execute_quail_payload(
        PAYLOAD, registry, object(), backend, runtime_state)

    assert response.metrics["boot_kind"] == "cold"
    assert gpu.prepared_boot is None
    assert len(boots) == 1

    second = worker.execute_quail_payload(
        PAYLOAD, registry, object(), backend, runtime_state)

    assert second.metrics["boot_kind"] == "warm"
    assert len(boots) == 2
    assert runtime_state[("quail", "qwen3-4b-fp8")] is gpu
    resized = []
    gpu.arena = SimpleNamespace(
        resize=lambda *pages, **kw: resized.append((pages, kw)))
    gpu.bind_query([1], [2], 4096, arena_pages=(8, 2))
    assert resized == [((8, 2), {"free_resident": True})]


def test_release_clears_cuda_state_and_a_new_model_boot_triggers_it(monkeypatch):
    calls = []
    cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: calls.append("synchronize"),
        empty_cache=lambda: calls.append("empty_cache"),
        ipc_collect=lambda: calls.append("ipc_collect"),
        memory_allocated=lambda: 123,
        memory_reserved=lambda: 456,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(worker.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(worker, "_release_vllm_parallel_state",
                        lambda: calls.append("release_parallel_state"))
    gpu = LoadedGpu.__new__(LoadedGpu)
    gpu.close = lambda: calls.append("close")
    booted = {("quail", "test"): gpu}
    result = worker.release_booted_models(booted)
    assert booted == {}
    assert calls == ["synchronize", "close", "release_parallel_state", "gc",
                     "empty_cache", "ipc_collect"]
    assert result == {"models_released": 1, "vllm_parallel_state_released": True,
                      "cuda_allocated_bytes": 123, "cuda_reserved_bytes": 456}

    # booting a different model releases the loaded one
    calls = []
    monkeypatch.setattr(
        worker, "release_booted_models",
        lambda state: (calls.append(sorted(state)), state.clear()))

    class _FakeLoaded:
        def __init__(self, backend, context, answer_ids):
            calls.append(("load", context.model.name))

        def bind_query(self, *args):
            calls.append("bind")

        def warm(self):
            return 0.0, None

        load_model_s = arena_s = pipeline_s = 0.0

    monkeypatch.setattr(worker, "LoadedGpu", _FakeLoaded)
    monkeypatch.setattr(worker, "_boot_record",
                        lambda gpu, cold, warm_s, tier, t: {
                            "boot_s": 0.0, "kind": "cold" if cold else "warm"})
    backend = SimpleNamespace(name="quail")
    state = {("quail", "qwen3-4b-fp8"): object()}
    context = SimpleNamespace(model=SimpleNamespace(name="qwen3-reranker"))

    gpu, boot = worker._boot_for_query(state, backend, context, 100, [1], [2])
    assert calls == [[("quail", "qwen3-4b-fp8")], ("load", "qwen3-reranker"),
                     "bind"]
    assert list(state) == [("quail", "qwen3-reranker")] and boot["kind"] == "cold"

    gpu2, boot2 = worker._boot_for_query(state, backend, context, 100, [1], [2])
    assert gpu2 is gpu and boot2["kind"] == "warm" and calls[-1] == "bind"
