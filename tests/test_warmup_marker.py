"""Tests for warmup marker path layout and identity stability."""

import json
import multiprocessing as mp
from types import SimpleNamespace

import pytest

from quail.backends.quail.executor import loop


def _stub_torch():
    return SimpleNamespace(
        __version__="2.9.0", version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(get_device_name=lambda: "NVIDIA H100",
                             synchronize=lambda: None))


def _concurrent_warmup(path, start, touch, compiles, results):
    loop._marker_path = lambda *args: path
    loop._marker_identity = lambda *args: {"version": 1}

    def compile_once(*args):
        with compiles.get_lock():
            compiles.value += 1

    def touch_together(*args):
        # both cached workers must warm outside the compilation lock
        touch.wait(timeout=10)

    loop.compile_kernels = compile_once
    loop.touch_kernels = touch_together
    start.wait(timeout=10)
    result = loop.warm_kernels(_stub_torch(), None, None, None, 100,
                               model_name="model")
    results.put(result["tier"])


def test_concurrent_processes_compile_once_and_touch_in_parallel(tmp_path):
    ctx = mp.get_context("spawn")
    path = str(tmp_path / "warm.json")
    start, touch = ctx.Barrier(3), ctx.Barrier(2)
    compiles = ctx.Value("i", 0)
    results = ctx.Queue()
    workers = [ctx.Process(target=_concurrent_warmup,
                           args=(path, start, touch, compiles, results))
               for _ in range(3)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=20)
            assert worker.exitcode == 0
        assert compiles.value == 1
        assert sorted(results.get(timeout=2) for _ in workers) == [
            "compile", "touch", "touch"]
        assert json.loads((tmp_path / "warm.json").read_text()) == {"version": 1}
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        results.close()


def test_warmup_failure_writes_no_marker_and_touch_warms_the_join_chunk(
        monkeypatch, tmp_path):
    path = tmp_path / "warm.json"
    monkeypatch.setattr(loop, "_marker_path", lambda *args: str(path))
    monkeypatch.setattr(loop, "_marker_identity", lambda *args: {"version": 1})

    def fail(*args):
        raise RuntimeError("GPU failed")

    for failure in ("compile", "synchronize"):
        torch = _stub_torch()
        monkeypatch.setattr(loop, "compile_kernels",
                            fail if failure == "compile" else lambda *args: None)
        if failure == "synchronize":
            torch.cuda.synchronize = fail
        with pytest.raises(RuntimeError, match="GPU failed"):
            loop.warm_kernels(torch, None, None, None, 100, model_name="model")
        assert not path.exists(), failure

        torch.cuda.synchronize = lambda: None
        monkeypatch.setattr(loop, "compile_kernels", lambda *args: None)
        assert loop.warm_kernels(torch, None, None, None, 100,
                                 model_name="model")["tier"] == "compile", failure
        path.unlink()

    calls = []
    monkeypatch.setattr(loop, "_forward_warm",
                        lambda *args, **kwargs: calls.append(kwargs))
    loop.touch_kernels(None, None, None, None, 100)
    assert calls == [{"join_chunk": True}]
