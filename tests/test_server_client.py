"""Remote Session, submit, get_run, watch, result, and cancel end to end."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import server_fakes

import quail
from quail.execution.session import Query, QueryLike
from quail.planner.plan import EngineConfig
from quail.server.records import QueryFailedError, ServerError

pytest.importorskip("starlette")
pytest.importorskip("uvicorn")

from quail.server.app import ServerSettings, create_app  # noqa: E402

CONFIG = EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")


@pytest.fixture()
def endpoint(tmp_path):
    servers = []

    def start(**overrides):
        settings = ServerSettings(
            data_dir=tmp_path / f"data{len(servers)}", models=("qwen3-4b-fp8",),
            device="h100-sxm", in_process=True,
            **{"hooks": server_fakes.hooks, **overrides})
        url, stop = server_fakes.start_server(create_app(settings))
        servers.append(stop)
        return url

    yield start
    for stop in servers:
        stop()


def register_reviews(session, tmp_path):
    pq.write_table(server_fakes.reviews_table(), tmp_path / "reviews.parquet")
    session.register("reviews", quail.DocumentProvider.from_parquet(
        str(tmp_path / "reviews.parquet"), id_col="id"))


def test_submit_close_reattach_watch_collect_and_stream_answers(endpoint, tmp_path):
    url = endpoint()
    with quail.Session(CONFIG, endpoint=url) as session:
        assert session.endpoint == url
        register_reviews(session, tmp_path)
        session.register("aspects", quail.DocumentProvider.from_table(
            pa.table({"id": ["a0"], "aspect": ["acting"]}), id_col="id"))
        query = session.sql(server_fakes.FILTER_SQL)
        for local_only in (query.explain, query.plan):
            with pytest.raises(RuntimeError, match=local_only.__name__):
                local_only()
        run = query.submit(query_id="demo-1")
        assert run.id == "demo-1"
        assert query.submit(query_id="demo-1").id == "demo-1"
        assert run.status().session_id == session.session_id
        with pytest.raises(RuntimeError, match="builder API"):
            session.docs("reviews")
        assert isinstance(query, QueryLike)
        assert all(hasattr(Query, name) for name in
                   ("run", "submit", "execute_stream", "collect", "explain",
                    "plan"))
        rows = query.collect()
        assert sorted(rows.column("r.id").to_pylist()) == ["r0", "r3"]
        assert session.sql(server_fakes.FILTER_SQL).run().count() == 2
        streamed = session.sql(server_fakes.FILTER_SQL).execute_stream()
        assert isinstance(streamed, pa.RecordBatchReader)
        assert streamed.read_all().num_rows == 2
        one = session.sql(server_fakes.FILTER_SQL).execute_stream(limit=1)
        assert one.read_all().num_rows == 1
        with pytest.raises(RuntimeError, match="edited plans"):
            session.sql(server_fakes.FILTER_SQL).run(plan=object())

    # the accepted query outlives the client session that sent it
    with quail.Session(CONFIG, endpoint=url) as session:
        run = session.get_run("demo-1")
        assert repr(run) == "QueryRun('demo-1')"
        snapshots = list(run.watch(poll_s=2.0))
        revisions = [status.revision for status in snapshots]
        assert revisions == sorted(set(revisions))
        final = snapshots[-1]
        assert final.done and final.state == final.phase["name"] == "succeeded"
        assert final.plan["backend"] == "quail" and "Scan" in final.plan["text"]
        assert final.progress["done"] == 6
        assert set(final.inputs) == {"reviews", "aspects"}
        assert final.inputs["reviews"]["kind"] == "snapshot"
        assert run.status() == final
        result = run.result()
        assert sorted(result.to_rows()) == [("r0",), ("r3",)]
        assert set(result.answer_tables["filters"]) == {("r", 0), ("r", 1)}
        assert "Scan" in result.explain()
        assert run.result().count() == 2, "result() fetches the saved files again"
        reader = run.stream_result()
        assert isinstance(reader, pa.RecordBatchReader)
        assert sorted(reader.read_all().column("r.id").to_pylist()) == ["r0", "r3"]

        register_reviews(session, tmp_path)
        session.register("products", quail.DocumentProvider.from_table(
            server_fakes.products_table(), id_col="asin"))
        run = session.sql(server_fakes.JOIN_SQL).submit()
        streamed = list(run.stream_answers(poll_s=1.0))
        assert sorted(entry["document"] for entry in streamed) == list(range(6))
        assert all(entry["anchor"] == "r" and entry["partners"] == ["p"]
                   and entry["asked"] == 4 for entry in streamed)
        first = next(entry for entry in streamed if entry["document"] == 0)
        assert first["matches"] == [[0], [2]], "r0 matches p0 and p2"

        page = run.answers(after=4, limit=1)
        assert page["next"] == 5 and page["done"] is True
        assert page["answers"] == [streamed[4]]
        assert run.answers(after=6)["answers"] == []
        assert run.status().progress["answers_saved"] == 6
        result = run.result()
        assert sorted(result.to_rows()) == sorted(
            (f"r{r}", f"p{p}") for r in range(6) for p in range(4)
            if (r + p) % 2 == 0)
        table = next(iter(result.answer_tables["joins"].values()))
        assert table.num_rows == 24
        assert sum(table["answer"].to_pylist()) == len(result.to_rows())


def test_bad_queries_cancel_and_timeout_fail_clearly(endpoint, tmp_path):
    with quail.Session(CONFIG, tokenizer=server_fakes.fake_tok) as session:
        assert session.endpoint is None
        register_reviews(session, tmp_path)
        with pytest.raises(RuntimeError, match="needs a Session with an endpoint"):
            session.sql(server_fakes.FILTER_SQL).submit()
        with pytest.raises(RuntimeError, match="needs a Session with an endpoint"):
            session.get_run("abc")

    class Custom:
        id_col = "id"

    url = endpoint()
    with quail.Session(CONFIG, endpoint=url) as session:
        with pytest.raises(TypeError, match="Custom cannot be sent"):
            session.register("custom", Custom())
        assert "custom" not in session.catalog
        register_reviews(session, tmp_path)
        with pytest.raises(ServerError, match="CompileError") as info:
            session.sql("SELECT r.nothing FROM reviews r").submit()
        assert info.value.status == 400
        with pytest.raises(ServerError, match="unknown query") as info:
            session.get_run("missing").status()
        assert info.value.status == 404
    other_model = EngineConfig(model="qwen3-32b-fp8", device="h100-sxm")
    with pytest.raises(ServerError, match="does not run model"):
        with quail.Session(other_model, endpoint=url) as other:
            register_reviews(other, tmp_path)
            other.sql(server_fakes.FILTER_SQL).submit()
    with pytest.raises(ServerError, match="cannot reach"):
        with quail.Session(CONFIG, endpoint="http://127.0.0.1:9") as session:
            session.get_run("x").status()

    with quail.Session(CONFIG, endpoint=endpoint(
            hooks=server_fakes.sleeping_hooks)) as session:
        register_reviews(session, tmp_path)
        query = session.sql(server_fakes.FILTER_SQL)
        first = query.submit()
        second = query.submit(timeout_s=0.5)
        assert second.cancel().state == "cancelled"
        with pytest.raises(QueryFailedError, match="cancelled") as info:
            second.result()
        assert info.value.status.state == "cancelled"
        for status in first.watch(poll_s=1.0):
            if status.state == "running":
                break
        assert first.cancel().cancel_requested
        with pytest.raises(QueryFailedError, match="cancelled") as info:
            first.result(poll_s=1.0)
        assert info.value.status.error["type"] == "Cancelled"

        timed = query.submit(timeout_s=0.5)
        with pytest.raises(QueryFailedError, match="timeout") as info:
            timed.result(poll_s=1.0)
        assert info.value.status.error["type"] == "TimeoutError"
        with pytest.raises(ServerError, match="unknown query"):
            session.get_run("none").cancel()
