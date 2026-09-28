"""Executor and scheduler: lifecycle, failures, cancel, timeout, restart."""

import json
import time
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import server_fakes

import quail
from quail import progress
from quail.builtins import built_in_registry
from quail.server import artifacts, inputs
from quail.server.executor import ChildProcessExecutor, InProcessExecutor, load_hooks
from quail.server.scheduler import Scheduler
from quail.server.store import Store


@pytest.fixture()
def store(tmp_path):
    store = Store(tmp_path / "quail.sqlite3")
    yield store
    store.close()


def submit(store, tmp_path, sql=server_fakes.FILTER_SQL, timeout_s=1000.0,
           config=None):
    prepared = inputs.describe(quail.DocumentProvider.from_table(
        server_fakes.reviews_table(), id_col="id"), tmp_path / "uploads")
    store.put_input(prepared.content_id, str(prepared.upload_path),
                    prepared.upload_path.stat().st_size)
    return store.create(
        spec={"sql": sql, "dialect": "snowflake", "order": None},
        config=config or server_fakes.CONFIG, inputs={"reviews": prepared.spec},
        timeout_s=timeout_s)


def scheduler_for(store, tmp_path, executor=None, poll_s=0.02):
    executor = executor or InProcessExecutor(server_fakes.hooks)
    return Scheduler(store, executor, tmp_path, poll_s=poll_s)


def wait_done(store, query_id, timeout=30.0):
    deadline = time.monotonic() + timeout
    status = store.get(query_id)
    while not status.done and time.monotonic() < deadline:
        status = store.wait(query_id, status.revision, 1.0)
    assert status.done, status
    return status


def test_in_process_run_saves_a_readable_and_verified_result(store, tmp_path):
    final = scheduler_for(store, tmp_path).run_one(submit(store, tmp_path))
    assert final.state == "succeeded"
    assert final.progress["label"] == "filter (2 stages)"
    assert final.result["rows"] == 2 and final.result["columns"] == ["r.id"]
    directory = tmp_path / final.result["directory"]
    result = artifacts.load_result(directory, final.result,
                                   built_in_registry().codecs)
    assert sorted(result.to_rows()) == [("r0",), ("r3",)]
    assert result.report["fresh_tokens"] > 0
    assert "Scan" in result.explain()
    assert json.loads((directory / "report.json").read_text())["backend"] == "quail"

    # each entry is (row index, last stage asked, passed)
    saved = artifacts.read_answers(directory / artifacts.ANSWERS_FILE)
    assert [entry["kind"] for entry in saved] == ["filter"]
    assert saved[0]["alias"] == "r" and saved[0]["stages"] == 2
    finished = {row: (stage, passed) for row, stage, passed in saved[0]["documents"]}
    assert {row for row, (_, passed) in finished.items() if passed} == {0, 3}
    assert finished[0] == finished[3] == (1, True)
    # the failing stage of each other row depends on the planner's order
    assert sorted(stage for _, (stage, passed) in finished.items()
                  if not passed) == [0, 0, 1, 1]
    assert final.progress["answers_saved"] == 1

    # verification rejects a missing or short result file
    verified = tmp_path / "verified"
    verified.mkdir()
    table = pa.table({"r.id": ["r0", "r3"]})
    result = quail.QueryResult.from_table(table, {"backend": "quail"})
    join_answers = pa.table({"r": [0], "p": [1], "answer": [True]})
    result.answer_tables = {"joins": {0: join_answers}}
    manifest = artifacts.write_result(result, verified)
    artifacts.verify_result(verified, manifest)
    loaded = artifacts.load_result(verified, manifest)
    assert loaded.to_rows() == [("r0",), ("r3",)]
    assert loaded.answer_tables["joins"][0].equals(join_answers)
    with pytest.raises(ValueError, match="holds 2 rows"):
        artifacts.verify_result(verified, {**manifest, "rows": 3})
    (verified / "result.arrow").unlink()
    with pytest.raises(FileNotFoundError):
        artifacts.verify_result(verified, manifest)
    assert not list(verified.glob("*.tmp"))


def test_failed_and_cancelled_runs_are_saved_and_not_killed(
        store, tmp_path, monkeypatch):
    executor = InProcessExecutor(server_fakes.hooks)
    scheduler = scheduler_for(store, tmp_path, executor)
    bad = scheduler.run_one(
        submit(store, tmp_path, sql="SELECT r.nothing FROM reviews r"))
    assert (bad.state, bad.error["type"]) == ("failed", "CompileError")
    assert executor._loaded_model is None

    queued = submit(store, tmp_path)
    store.request_cancel(queued.id)
    cancelled = scheduler.run_one(queued)
    assert cancelled.state == "cancelled" and cancelled.started_at is None

    failing = scheduler_for(
        store, tmp_path, InProcessExecutor(server_fakes.failing_hooks),
    ).run_one(submit(store, tmp_path))
    assert (failing.state, failing.error["type"]) == ("failed", "RuntimeError")
    assert "refused to start" in failing.error["message"]
    assert "Traceback" in failing.error["traceback"]

    def broken_verify(directory, manifest):
        raise OSError("disk full")

    monkeypatch.setattr("quail.server.scheduler.verify_result", broken_verify)
    unpublished = scheduler.run_one(submit(store, tmp_path))
    assert (unpublished.state, unpublished.error["type"]) == (
        "failed", "PublishError")
    assert "disk full" in unpublished.error["message"]

    stops = []

    class Execution:
        def __init__(self, emit):
            self.emit = emit

        def wait(self, timeout):
            # the first poll sees the record closed before done is reported
            if self.emit is None:
                return True
            self.emit("failed", {"type": "RuntimeError", "message": "GPU said no"})
            self.emit = None
            return False

        def stop(self):
            stops.append(1)

    class Executor:
        def start(self, job, emit):
            return Execution(emit)

    final = scheduler_for(store, tmp_path, Executor()).run_one(
        submit(store, tmp_path))
    assert final.state == "failed" and final.error["message"] == "GPU said no"
    assert stops == [], "killing it would throw away the loaded model"


def test_executors_switch_models_skip_warm_loads_and_stop_on_timeout(
        store, tmp_path):
    for child in (False, True):
        executor = (ChildProcessExecutor("server_fakes:hooks") if child
                    else InProcessExecutor(server_fakes.hooks))
        scheduler = scheduler_for(store, tmp_path, executor)
        seen = {}

        def record():
            for status in store.list_recent():
                phases = seen.setdefault(status.id, [])
                if not phases or phases[-1] != status.phase["name"]:
                    phases.append(status.phase["name"])

        store.add_listener(record)
        try:
            first = submit(store, tmp_path)
            assert scheduler.run_one(first).state == "succeeded", child
            assert executor._loaded_model == "qwen3-4b-fp8", child
            pid = executor._process.pid if child else None
            second = submit(store, tmp_path)
            assert scheduler.run_one(second).state == "succeeded", child
            if child:
                assert executor._process.pid == pid, "same model, same child"
            other = submit(store, tmp_path,
                           config={**server_fakes.CONFIG, "model": "qwen3-32b-fp8"})
            assert scheduler.run_one(other).state == "succeeded", child
            assert executor._loaded_model == "qwen3-32b-fp8", child
            if child:
                assert executor._process.pid != pid, "another model, a new child"
        finally:
            executor.close()
            store.remove_listener(record)

        start = ["queued", "resolving_inputs", "planning"]
        end = ["executing", "succeeded"]
        assert seen[first.id] == [*start, "loading_model", *end], child
        assert seen[second.id] == [*start, *end], child
        assert seen[other.id] == [*start, "switching_model", *end], child

    assert load_hooks(None) is None
    assert load_hooks("server_fakes:hooks") is server_fakes.hooks
    with pytest.raises(TypeError):
        load_hooks("server_fakes:TRUTH")
    first = submit(store, tmp_path, timeout_s=1.0)
    executor = ChildProcessExecutor("server_fakes:sleeping_hooks")
    scheduler = scheduler_for(store, tmp_path, executor, poll_s=0.05)
    scheduler.start()
    try:
        final = wait_done(store, first.id, timeout=60)
        assert (final.state, final.error["type"]) == ("failed", "TimeoutError")
        assert "1 s timeout" in final.error["message"]
        executor.hooks_reference = "server_fakes:hooks"
        again = submit(store, tmp_path)
        assert wait_done(store, again.id, timeout=60).state == "succeeded"
    finally:
        scheduler.stop()


def test_progress_sinks_labels_and_quiet_mode(capsys):
    first, second = Mock(), Mock()
    progress.set_answer_sink(first)
    progress.set_answer_sink(second)
    progress.set_answer_sink(None, owner=first)
    assert progress.answer_sink() is second, "the stale job's removal is ignored"
    progress.set_answer_sink(None, owner=second)
    assert progress.answer_sink() is None
    progress.set_progress_sink(first)
    progress.set_progress_sink(None, owner=second)
    assert progress._SINK is first
    progress.set_progress_sink(None)
    assert progress._SINK is None

    progress.say("shown")
    with progress.quiet():
        progress.say("hidden")
        progress.Progress("hidden step", total=2).finish("hidden step done")
    progress.Progress("step", total=2).finish("step done", "extra")
    out = capsys.readouterr().out
    assert "[quail] shown" in out
    assert "INFO" in out
    assert "hidden" not in out
    assert "[quail] step done: 0/2 documents" in out
    assert out.strip().endswith("documents/s, extra")

    try:
        progress.set_gpu_index(3)
        with progress.quiet():
            warmup = progress.Progress("GEMM warmup", total=2,
                                       unit="configurations", every=0,
                                       emit=progress.logger.info)
            warmup.update(2)
            warmup.finish("GEMM warmup done")
        progress.Progress("filter (2 stages)", total=10).finish("filter done")
    finally:
        progress.set_gpu_index(None)
    progress.say("parent process")
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert all("[quail][GPU 3]" in line for line in lines[:3])
    assert "2/2 configurations" in lines[0]
    assert "[quail] parent process" in lines[-1]


def test_describe_and_resolve_supported_providers(tmp_path):
    table = server_fakes.reviews_table()
    memory = inputs.describe(
        quail.DocumentProvider.from_table(table, id_col="id"), tmp_path)
    assert (memory.spec["kind"], memory.spec["id_col"]) == ("snapshot", "id")
    assert memory.upload_path.name == f"{memory.content_id}.arrow"
    same = inputs.describe(quail.DocumentProvider.from_table(table, id_col="id"),
                           tmp_path)
    assert same.content_id == memory.content_id, "the same table, the same id"
    pq.write_table(table, tmp_path / "r.parquet")
    parquet = inputs.describe(quail.DocumentProvider.from_parquet(
        str(tmp_path / "r.parquet"), id_col="id"), tmp_path)
    assert parquet.spec["kind"] == "snapshot"
    provider = inputs.resolve(parquet.spec,
                              {parquet.content_id: parquet.upload_path})
    scanned = provider.scan(quail.ScanRequest(columns=("id",))).read_all()
    assert scanned.column("id").to_pylist() == table.column("id").to_pylist()

    schema = pa.schema([("id", pa.string())])
    hf = quail.catalog.HuggingFaceProvider("org/data", id_col="id",
                                           arrow_schema=schema)
    pinned = inputs.describe(hf, tmp_path,
                             resolve_revision=lambda name: f"sha-of-{name}")
    assert pinned.spec == {"kind": "hf", "dataset": "org/data", "config": "",
                           "split": "train", "revision": "sha-of-org/data",
                           "id_col": "id"}
    assert pinned.upload_path is None
    pinned_provider = quail.catalog.HuggingFaceProvider(
        "org/data", id_col="id", revision="abc", arrow_schema=schema)
    assert pinned_provider.content_identity() != hf.content_identity()
    with pytest.raises(inputs.InvalidRequestError, match="revision"):
        inputs.resolve({**pinned.spec, "revision": ""}, {})
    with pytest.raises(inputs.InvalidRequestError, match="not uploaded"):
        inputs.resolve(memory.spec, {})
    with pytest.raises(inputs.InvalidRequestError, match="unknown input kind"):
        inputs.resolve({"kind": "s3"}, {})
