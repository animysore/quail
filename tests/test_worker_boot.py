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
            self.generation = False

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
    context = SimpleNamespace(
        model=SimpleNamespace(name="qwen3-reranker"), query_settings={})

    gpu, boot = worker._boot_for_query(state, backend, context, 100, [1], [2])
    assert calls == [[("quail", "qwen3-4b-fp8")], ("load", "qwen3-reranker"),
                     "bind"]
    assert list(state) == [("quail", "qwen3-reranker")] and boot["kind"] == "cold"

    gpu2, boot2 = worker._boot_for_query(state, backend, context, 100, [1], [2])
    assert gpu2 is gpu and boot2["kind"] == "warm" and calls[-1] == "bind"


@pytest.mark.parametrize("prepare", [False, True])
def test_multi_gpu_native_releases_request_engine_first(monkeypatch, prepare):
    events = []
    state = {("request-engine", "vllm", "qwen3-4b-fp8", True): {
        "client": SimpleNamespace(close=lambda: events.append("close")),
    }}
    context = SimpleNamespace(
        gpu_count=2, runtime_state=state, request=None, graph=None, registry=None,
    )
    monkeypatch.setattr(worker, "quail_runtime_payload", lambda *a: {})

    def execute(*args):
        assert state == {}
        events.append("execute")

    monkeypatch.setattr(worker, "execute_quail_multi", execute)
    if prepare:
        worker.prepare_quail_request(context)
        assert events == ["close"]
    else:
        worker.execute_quail_request(context)
        assert events == ["close", "execute"]
    assert state == {}


@pytest.mark.parametrize("model", ["qwen3-4b-fp8", "qwen3-32b-fp8"])
def test_generation_mode_rebinds_tied_heads_and_reloads_untied(monkeypatch, model):
    registry = built_in_registry()
    spec = registry.model(model)
    events = []

    class FakeLoaded:
        def __init__(self, backend, context, answer_ids):
            self.generation = context.query_settings["generation"]
            events.append(("load", self.generation))

        def bind_generation(self, generation):
            assert spec.tied_head
            self.generation = generation
            events.append(("rebind", generation))

        def bind_query(self, *args):
            events.append(("query", self.generation))

        def warm(self):
            return 0.0, None

    def release(state):
        events.append(("release", len(state)))
        state.clear()

    monkeypatch.setattr(worker, "LoadedGpu", FakeLoaded)
    monkeypatch.setattr(worker, "release_booted_models", release)
    monkeypatch.setattr(worker, "_boot_record", lambda *args: {"boot_s": 0, "kind": ""})
    state = {}
    contexts = [SimpleNamespace(
        model=spec, gpu_count=1, query_settings={"generation": mode})
                for mode in (False, True, True, False)]
    loaded = [
        worker._boot_for_query(
            state, registry.backend("quail"), context, 128, [1], [2])[0]
        for context in contexts
    ]
    assert len(state) == 1
    assert list(state) == [("quail", model)]
    if spec.tied_head:
        assert all(gpu is loaded[0] for gpu in loaded)
        assert [event for event in events if event[0] == "rebind"] == [
            ("rebind", True), ("rebind", False)]
        assert not any(event[0] == "release" for event in events)
    else:
        assert loaded[0] is not loaded[1] is loaded[2]
        assert loaded[3] is not loaded[2]
        assert [event for event in events if event[0] == "release"] == [
            ("release", 1), ("release", 1)]


@pytest.mark.parametrize("arch,role,gpus", [
    ("qwen3", "reranker", 1),
    ("diffusion_gemma", "generative", 1),
    ("qwen3", "generative", 2),
])
def test_invalid_generation_preserves_cached_models(monkeypatch, arch, role, gpus):
    spec = SimpleNamespace(name="existing", arch=arch, role=role, tied_head=True)
    native, request = object(), {"client": object()}
    state = {
        ("quail", spec.name): native,
        ("request-engine", "vllm", spec.name, False): request,
    }
    context = SimpleNamespace(
        model=spec, gpu_count=gpus, query_settings={"generation": True})
    with pytest.raises(ValueError, match="generative Qwen3 on one GPU"):
        worker._boot_for_query(
            state, SimpleNamespace(name="quail"), context, 128, [1], [2])
    assert state == {
        ("quail", spec.name): native,
        ("request-engine", "vllm", spec.name, False): request,
    }


def test_tied_head_return_to_filter_keeps_warm_kernels(monkeypatch):
    gpu = LoadedGpu.__new__(LoadedGpu)
    gpu.spec = SimpleNamespace(tied_head=True)
    gpu.torch = object()
    gpu.model = SimpleNamespace(quail_answer_token_ids=[1, 2])
    gpu.generation, gpu._warmed = True, True
    gpu.prepared_boot = {"kind": "warm"}
    calls = []
    monkeypatch.setattr(worker, "retain_answer_head",
                        lambda *a, **k: calls.append(k["generation"]))
    gpu.bind_generation(False)
    assert not gpu.generation and gpu._warmed
    assert gpu.warm() == (0.0, None)
    assert gpu.prepared_boot is None
    gpu.bind_generation(True)
    assert gpu.generation and not gpu._warmed
    assert calls == [False, True]


def test_close_releases_gpu_owners_even_if_execution_cleanup_fails():
    gpu = LoadedGpu.__new__(LoadedGpu)

    def fail():
        raise RuntimeError("cleanup failed")

    gpu.execution = SimpleNamespace(close=fail)
    gpu.model = gpu.pipeline = gpu.arena = gpu.async_ans = object()
    gpu.prepared_boot = {}
    with pytest.raises(RuntimeError, match="cleanup failed"):
        gpu.close()
    assert gpu.model is gpu.pipeline is gpu.arena is gpu.async_ans is None
    assert gpu.execution is gpu.prepared_boot is None


@pytest.mark.parametrize("stage", ["load", "bind", "warm"])
def test_failed_native_boot_discards_state_and_keeps_primary_error(
        monkeypatch, stage):
    state = {}
    events = []
    registry = built_in_registry()

    def fail(current):
        if current == stage:
            raise RuntimeError(f"{stage} failed")

    class FakeLoaded:
        def __init__(self, *args):
            fail("load")

        def bind_query(self, *args):
            fail("bind")

        def warm(self):
            fail("warm")

    def release(runtime_state):
        events.append("release")
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(worker, "LoadedGpu", FakeLoaded)
    monkeypatch.setattr(worker, "release_booted_models", release)
    context = SimpleNamespace(model=registry.model("qwen3-4b-fp8"), query_settings={})
    with pytest.raises(RuntimeError, match=f"{stage} failed") as caught:
        worker._boot_for_query(
            state, registry.backend("quail"), context, 128, [1], [2])
    assert not state and events == ["release"]
    assert "cleanup failed" in " ".join(caught.value.__notes__)


@pytest.mark.parametrize("generation", [False, True])
def test_prepared_boot_cannot_bypass_generation_mode_transition(
        monkeypatch, generation):
    registry = built_in_registry()
    gpu = LoadedGpu.__new__(LoadedGpu)
    gpu.generation = not generation
    gpu.prepared_boot = {"kind": "stale", "boot_s": 123}
    state = {("quail", "qwen3-4b-fp8"): gpu}
    boots = []

    def boot(*args):
        assert gpu.prepared_boot is None
        assert args[2].query_settings["generation"] == generation
        boots.append(generation)
        return gpu, {"kind": "cold", "boot_s": 0}

    monkeypatch.setattr(worker, "_boot_for_query", boot)
    monkeypatch.setattr(worker, "_gpu_state", lambda gpu: {})
    monkeypatch.setattr(worker, "execute_single", lambda *args: {"_outputs": {}})
    envelope = {**ENVELOPE, "settings": {
        **ENVELOPE["settings"], "generation": generation}}
    payload = {**PAYLOAD, "physical_plan": envelope}
    response = worker.execute_quail_payload(
        payload, registry, object(), registry.backend("quail"), state)
    assert boots == [generation]
    assert response.metrics["boot_kind"] == "cold"
