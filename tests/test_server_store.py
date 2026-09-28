"""Saved query records: revisions, query ids, epochs, cancel, recovery."""

import pytest

from quail.server.records import (
    InvalidRequestError,
    QueryIdConflictError,
    QueryStatus,
    UnknownQueryError,
)
from quail.server.store import Store

SPEC = {"sql": "SELECT r.id FROM reviews r", "dialect": "snowflake",
        "order": None}
CONFIG = {"model": "qwen3-4b-fp8", "device": "h100-sxm", "gpus": 1,
          "backend": "quail"}
INPUTS = {"reviews": {"kind": "snapshot", "content_id": "abc", "id_col": "id"}}


@pytest.fixture()
def store(tmp_path):
    store = Store(tmp_path / "quail.sqlite3")
    yield store
    store.close()


def create(store, **overrides):
    defaults = dict(spec=SPEC, config=CONFIG, inputs=INPUTS, timeout_s=1000.0)
    return store.create(**{**defaults, **overrides})


def test_create_saves_a_complete_snapshot_and_client_ids_return_it_or_conflict(
        store):
    status = create(store)
    assert (status.state, status.revision, status.timeout_s) == ("queued", 1, 1000.0)
    assert status.spec == SPEC and status.config == CONFIG
    assert status.inputs == INPUTS and not status.done
    assert store.get(status.id) == status
    # the snapshot round trips through JSON without Quail objects
    assert QueryStatus.from_dict(status.to_dict()) == status
    with pytest.raises(UnknownQueryError):
        store.get("missing")
    with pytest.raises(InvalidRequestError):
        create(store, timeout_s=0)

    first = create(store, query_id="demo-1")
    assert first.id == "demo-1"
    assert create(store, query_id="demo-1") == first
    assert len(store.list_recent()) == 2
    with pytest.raises(QueryIdConflictError):
        create(store, spec={**SPEC, "sql": "SELECT 1"}, query_id="demo-1")
    assert create(store, query_id="demo-2").id != first.id
    assert len(create(store).id) == 32, "the store picks an id when none is sent"
    for bad in ("", "-x", "a/b", "a" * 129, 7):
        with pytest.raises(InvalidRequestError, match="query id"):
            create(store, query_id=bad)
    assert store.find("demo-1") == first
    assert store.find("nope") is None


def test_discard_cancel_and_epoch_updates_change_records_and_notify(
        store):
    queued = create(store, query_id="q")
    assert store.discard(queued.id)
    assert store.find("q") is None
    # the same id can be used again, for a different query too
    again = create(store, spec={**SPEC, "sql": "SELECT 1"}, query_id="q")
    assert again.spec["sql"] == "SELECT 1"
    store.begin(again.id)
    assert not store.discard(again.id)
    assert store.get(again.id).cancel_requested

    queued = create(store)
    cancelled = store.request_cancel(queued.id)
    assert cancelled.state == "cancelled" and cancelled.done
    assert cancelled.error["type"] == "Cancelled"
    assert store.next_queued() is None

    running = create(store)
    epoch = store.begin(running.id)
    flagged = store.request_cancel(running.id)
    assert flagged.state == "planning" and flagged.cancel_requested
    assert store.request_cancel(running.id).revision == flagged.revision
    assert store.finish(running.id, epoch, "cancelled",
                        error={"type": "Cancelled", "message": "stopped"})
    assert store.request_cancel(running.id).state == "cancelled"

    # updates bump revisions, notify listeners, and only the current epoch writes
    status = create(store)
    calls = []

    def listener():
        calls.append(store.get(status.id).revision)

    store.add_listener(listener)
    epoch = store.begin(status.id)
    planning = store.get(status.id)
    assert (planning.state, planning.revision, epoch) == ("planning", 2, 1)
    assert planning.started_at is not None
    with pytest.raises(InvalidRequestError):
        store.begin(status.id)

    assert store.update(status.id, epoch, plan={"text": "Scan"})
    assert store.update(status.id, epoch, state="running",
                        progress={"done": 3, "total": 6})
    running = store.get(status.id)
    assert (running.state, running.revision) == ("running", 4)
    assert running.plan == {"text": "Scan"}
    assert running.progress == {"done": 3, "total": 6}
    assert calls == [2, 3, 4], "each call sees the committed write"
    store.remove_listener(listener)

    # a stale execution cannot change the record
    assert not store.update(status.id, epoch + 1, progress={"done": 6})
    assert store.get(status.id).progress == {"done": 3, "total": 6}
    with pytest.raises(ValueError):
        store.update(status.id, epoch, state="succeeded")

    assert store.finish(status.id, epoch, "succeeded", result={"rows": 2})
    final = store.get(status.id)
    assert final.done and final.result == {"rows": 2} and final.revision == 5
    # a closed record never changes again
    assert not store.finish(status.id, epoch, "failed",
                            error={"type": "Late", "message": "late"})
    assert not store.update(status.id, epoch, progress={"done": 0})
    assert store.get(status.id).revision == 5
    assert calls == [2, 3, 4]
    with pytest.raises(ValueError):
        store.finish(status.id, epoch, "succeeded")
