"""Copying the live SQLite file to another disk and restoring from it."""

import sqlite3
import time

import pytest
import server_fakes

from quail.server.checkpoint import Checkpoint, restore
from quail.server.store import Store

SPEC = {"sql": "SELECT r.id FROM reviews r", "dialect": "snowflake",
        "order": None}
INPUTS = {"reviews": {"kind": "snapshot", "content_id": "abc", "id_col": "id"}}


def create(store):
    return store.create(spec=SPEC, config=server_fakes.CONFIG, inputs=INPUTS,
                        timeout_s=1000.0)


def fetch_one(path, sql):
    with sqlite3.connect(str(path)) as conn:
        return conn.execute(sql).fetchone()


def test_backup_is_a_complete_copy_including_the_wal(tmp_path):
    store = Store(tmp_path / "live" / "quail.sqlite3")
    first = create(store)
    epoch = store.begin(first.id)
    store.update(first.id, epoch, state="running", progress={"done": 1})
    # the WAL holds these writes; a plain file copy would miss them
    assert (tmp_path / "live" / "quail.sqlite3-wal").stat().st_size > 0
    copy = tmp_path / "volume" / "quail.sqlite3"
    store.backup(copy)
    assert not copy.with_name("quail.sqlite3.tmp").exists()
    assert fetch_one(copy, "SELECT id, state FROM queries") == (first.id, "running")
    store.close()

    live = tmp_path / "fresh" / "quail.sqlite3"
    assert restore(copy, live)
    assert not restore(copy, live)
    reopened = Store(live)
    assert reopened.recover() == [first.id]
    assert reopened.get(first.id).state == "interrupted"
    assert reopened.get(first.id).progress == {"done": 1}
    reopened.close()


def test_checkpoint_copies_after_writes_syncs_now_and_survives_failing_commits(
        tmp_path, caplog):
    store = Store(tmp_path / "live" / "quail.sqlite3")
    commits = []
    copy = tmp_path / "volume" / "quail.sqlite3"

    def commit():
        commits.append(1)
        if len(commits) == 2:
            raise OSError("volume busy")

    checkpoint = Checkpoint(store, copy, commit=commit, min_interval_s=0.05)
    assert checkpoint.run_once()
    assert not checkpoint.run_once()
    assert commits == [1] and copy.exists()

    checkpoint.start()
    status = create(store)
    deadline = time.monotonic() + 10
    while checkpoint.copies < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert checkpoint.copies == 2 and len(commits) == 3, "the failed copy is retried"
    assert "checkpoint" in caplog.text and "volume busy" in caplog.text
    assert fetch_one(copy, "SELECT count(*) FROM queries") == (1,)
    time.sleep(0.2)
    assert checkpoint.copies == 2, "no writes, no copies"

    store.request_cancel(status.id)
    checkpoint.stop()
    checkpoint.stop()
    assert fetch_one(copy, "SELECT state FROM queries") == ("cancelled",)
    assert checkpoint.copies == 3 and len(commits) == 4
    store.close()
    assert not checkpoint.run_once()

    # sync copies every write now and raises when the commit fails
    store = Store(tmp_path / "synced" / "quail.sqlite3")
    copy = tmp_path / "synced-volume" / "quail.sqlite3"
    failures = []

    def commit():
        if failures:
            raise OSError(failures.pop())

    checkpoint = Checkpoint(store, copy, commit=commit, progress_interval_s=60)
    status = create(store)
    epoch = store.begin(status.id)
    checkpoint.sync()
    store.update(status.id, epoch, progress={"done": 1})
    assert not checkpoint.run_once(), "progress alone waits for the timer"
    checkpoint.sync()
    assert fetch_one(copy, "SELECT progress_json FROM queries") == ('{"done": 1}',)
    checkpoint.sync()
    assert checkpoint.copies == 2
    store.update(status.id, epoch, state="running")
    failures.append("volume busy")
    with pytest.raises(OSError, match="volume busy"):
        checkpoint.sync()
    assert checkpoint.run_once(), "the failed copy is tried again"
    store.close()


def test_server_runs_closers_before_closing_the_store(tmp_path):
    pytest.importorskip("starlette")
    from quail.server.app import ServerSettings, create_app

    settings = ServerSettings(
        data_dir=tmp_path / "data", db_path=tmp_path / "local" / "live.sqlite3",
        models=("qwen3-4b-fp8",), device="h100-sxm", in_process=True,
        hooks=server_fakes.hooks)
    service = create_app(settings).state.server
    assert service.store.path == tmp_path / "local" / "live.sqlite3"
    checkpoint = Checkpoint(service.store, tmp_path / "data" / "quail.sqlite3")
    service.add_closer(checkpoint.stop)
    seen = []
    service.add_closer(lambda: seen.append(service.store.closed))
    service.stop()
    assert seen == [False]
    assert service.store.closed
    assert (tmp_path / "data" / "quail.sqlite3").exists()
