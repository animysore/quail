"""End-to-end Session tests with a fake executor."""

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fakes import register_claims_evidence

import quail
from demos import quickstart, quickstart_modal
from quail.execution import execute as execution
from quail.execution.pairs import pair_fraction, pair_table, partner_map
from quail.execution.result import (
    QueryResult,
    build_result_declaration,
    count_rows,
    document_index_table,
)
from quail.physical import AiJoin
from quail.planner.plan import EngineConfig

CONFIG = EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")


def fake_tok(text):
    return text.split()


def _run(query, execute):
    from quail.execution.execute import execute_query

    return execute_query(query, physical_executor=execute)


@pytest.fixture()
def sess(tmp_path):
    s = quail.Session(CONFIG, tokenizer=fake_tok)
    pq.write_table(pa.table({
        "id": [f"r{i}" for i in range(6)],
        "review": [f"review {i} " + "pad " * 20 for i in range(6)],
    }), tmp_path / "r.parquet")
    s.register("reviews", quail.DocumentProvider.from_parquet(
        str(tmp_path / "r.parquet"), id_col="id"))
    yield s
    s.close()


def test_session_requires_model_and_device_and_loads_gigatoken(monkeypatch):
    with pytest.raises(TypeError, match="'model' and 'device'"):
        EngineConfig()
    with pytest.raises(TypeError, match="'device'"):
        EngineConfig(model="qwen3-4b-fp8")
    with pytest.raises(TypeError, match="required positional argument: 'config'"):
        quail.Session()
    assert (CONFIG.gpus, CONFIG.backend) == (1, "quail")

    sources = []

    class Tokenizer:
        def __init__(self, source):
            sources.append(source)

        def encode(self, text):
            assert text == "hello"
            return np.array([1, 2], dtype=np.uint32)

    monkeypatch.setitem(sys.modules, "gigatoken", SimpleNamespace(Tokenizer=Tokenizer))
    with quail.Session(CONFIG) as session:
        token_ids = session.tokenizer("hello")
    assert sources == ["Qwen/Qwen3-4B-FP8"]
    assert token_ids == [1, 2]
    assert all(type(token_id) is int for token_id in token_ids)


def make_executor(filter_truth, join_truth=None):
    """Build a fake executor from filter and join truth tables."""
    def _match_key(alias, q):
        for key in filter_truth[alias]:
            if any(t.startswith(key) for t in q):
                return key
        raise KeyError(f"no filter_truth key for alias {alias!r} "
                       f"matches tokens {q[:5]}")

    def _exec(request):
        from test_quail_backend import graph_state

        from quail.backends.quail.graph import execute_single_graph
        from quail.builtins import built_in_registry
        from quail.execution.runner import NodeMetrics, NodeResult
        from quail.execution.types import PhysicalResponse
        from quail.physical import AiFilter, Scan, decode_graph

        graph = decode_graph(request.plan["graph"], built_in_registry().codecs)
        docs = {
            node.alias: request.inputs[node.input_id].documents
            for node in graph.nodes if isinstance(node, Scan)
        }

        class FixedAnswers:
            def execute(self, node, inputs):
                if isinstance(node, AiFilter):
                    rows = {}
                    for document in inputs["document_ids"]:
                        row = []
                        for question in node.question_token_ids:
                            bit = filter_truth[node.alias][
                                _match_key(node.alias, question)
                            ][document]
                            row.append(bit)
                            if not bit:
                                break
                        rows[document] = row
                    survivors = [d for d, row in rows.items()
                                 if len(row) == len(node.stages) and all(row)]
                    # the real loop reports a chunk's finished documents
                    # by position; the hook streams them
                    if inputs.get("document_done") is not None:
                        inputs["document_done"]([
                            (position, len(row) - 1, bool(row[-1]))
                            for position, row in enumerate(rows.values())])
                    return NodeResult({
                        f"ids:{node.alias}": survivors,
                        f"filter_answers:{node.alias}": rows,
                    }, NodeMetrics(input_rows=len(rows),
                                   output_rows=len(survivors),
                                   evaluated_documents=len(rows),
                                   fresh_tokens=10 * len(rows)))
                anchors = list(inputs["anchor_ids"])
                outputs = {}
                for stage in node.stages:
                    partners = inputs["partner_indices"][stage.written_pos]
                    rule = join_truth[(stage.anchor, *stage.partners)]
                    rows = {i: [rule(a, *pair) for pair in partners]
                            for i, a in enumerate(anchors)}
                    outputs[f"join_answers:{stage.written_pos}"] = {
                        "rows": rows, "anchor_index": anchors,
                        "partner_index": partners, "anchor": stage.anchor,
                        "partners": list(stage.partners),
                        "semantics": stage.semantics,
                        "selectivity": stage.selectivity,
                        "written_pos": stage.written_pos,
                    }
                passing = {anchors[i] for i, row in rows.items() if any(row)}
                outputs[f"ids:{node.anchor}"] = [a for a in anchors
                    if (a not in passing if stage.semantics == "anti"
                        else a in passing)]
                # the real loop reports each anchor's last-stage row as
                # it finishes; the hook frees its KV and streams answers
                if inputs.get("anchor_done") is not None:
                    for i, row in rows.items():
                        inputs["anchor_done"](i, row)
                return NodeResult(outputs, NodeMetrics(
                    input_rows=len(anchors),
                    output_rows=len(outputs[f"ids:{node.anchor}"]),
                    evaluated_document_pairs=sum(
                        len(row) for row in rows.values())))

        state = graph_state(FixedAnswers(), docs)
        if getattr(request, "relations", None):
            state["columns"] = request.column_tables()
        report = execute_single_graph(state, {
            "filter_limit": None, "pre_ids": [],
        }, graph)
        outputs = report.pop("_outputs")
        return PhysicalResponse(outputs, {
            **report, "wall_s": 1.0, "boot_s": 0.5, "fresh_tokens": 1234,
        })

    return _exec


FILTER_SQL = """
    SELECT r.id FROM reviews r
    WHERE AI_FILTER(PROMPT('q1: {0}', r.review), {'selectivity': 0.5})
      AND AI_FILTER(PROMPT('q2: {0}', r.review), {'selectivity': 0.5})
"""
TRUTH = {"r": {"q1:": [1, 1, 0, 1, 1, 0], "q2:": [1, 0, 1, 1, 0, 1]}}


def test_explain_analyze_shows_measured_rows_beside_estimates(
        sess, monkeypatch):
    query = sess.sql(FILTER_SQL + " LIMIT 1")
    execution_calls = []

    def executor(request):
        execution_calls.append(request)
        return make_executor(TRUTH)(request)

    original = execution.execute_query
    monkeypatch.setattr(
        execution, "execute_query",
        lambda q, physical_executor=executor, plan=None: original(
            q, physical_executor=physical_executor, plan=plan))
    planned = query.explain()
    text = query.explain(analyze=True)

    assert len(execution_calls) == 1
    assert "   rows   " not in planned and "run:" not in planned
    header = next(line for line in text.splitlines() if "est. rows" in line)
    assert header.split() == ["est.", "rows", "rows", "est.", "pass", "pass",
                              "est.", "time", "time", "fresh", "tokens"]
    assert re.search(
        r"AiFilter: r\s+1\.5\s+2\s+[\d.]+ ms\s+<1 ms\s+60$", text, re.M)
    # Both predicates have equal estimated cost and pass four of six rows.
    first = re.search(r"(?:1st: )?predicate ([12])  PROMPT\('DOCUMENT:\\n"
                      r"\{0\}\\n\\nq\1:'\)\s+6\s+6\s+50%\s+66\.7%$",
                      text, re.M)
    second = re.search(r"(?:2nd: )?predicate ([12])  PROMPT\('DOCUMENT:\\n"
                       r"\{0\}\\n\\nq\1:'\)\s+3\s+4\s+50%\s+50%$",
                       text, re.M)
    assert first and second and first[1] != second[1]
    assert re.search(r"Scan reviews as r\s+6\s+6\s+<1 ms$", text, re.M)
    assert re.search(r"Project: r\.id\s+1\.5\s+2\s+<1 ms$", text, re.M)
    assert re.search(r"Limit: 1\s+1\s+1\s+<1 ms$", text, re.M)
    assert "run:" in text
    assert "query time   1 s (model startup excluded)" in text
    assert "startup      500 ms" in text
    assert "throughput   6 documents/second over 6 input documents" in text
    assert "tokens       1,234 fresh" in text
    assert "GPU cost     $0.0011 per query (1 GPU at $3.9492/hour" in text

    # the executed result renders the same measured table on its own
    result = _run(sess.sql(FILTER_SQL + " LIMIT 1"), make_executor(TRUTH))
    explained = result.explain()
    assert re.search(r"Limit: 1\s+1\s+1\s+<1 ms$", explained, re.M)
    assert re.search(rf"(?:1st: )?predicate {first[1]}\s+6\s+6\s+50%\s+66\.7%$",
                     explained, re.M)
    stages = [s for s in result.report["stages"] if s["op"] == "filter"]
    assert [s["written_pos"] for s in stages] == [int(first[1]) - 1,
                                                int(second[1]) - 1]


def test_query_rows_observers_saved_reports_and_streamed_results(sess):
    res = _run(sess.sql(FILTER_SQL), make_executor(TRUTH))
    assert res.columns == ["r.id"]
    assert sorted(res.to_rows()) == [("r0",), ("r3",)]
    stages = [s for s in res.report["stages"] if s["op"] == "filter"]
    assert stages[0]["evaluated"] == 6
    assert stages[0]["observed_selectivity"] == pytest.approx(4 / 6, abs=1e-3)
    assert stages[1]["evaluated"] == 4, "stage 2 only saw stage-1 survivors"
    assert res.report["wall_s"] == res.report["model_wall_s"] == 1.0
    assert res.report["finish_s"] >= 0
    assert res.report["input_ready_s"] >= res.report["planning_s"]
    assert "gpu_s" not in res.report

    def timed(request):
        response = make_executor(TRUTH)(request)
        response.metrics["gpu_s"] = 0.75
        return response

    limited = _run(sess.sql(FILTER_SQL + " LIMIT 1"), timed)
    assert limited.count() == 1
    assert limited.report["gpu_s"] == 0.75

    registry = quail.ExtensionRegistry.with_built_ins().register_observer(
        NodeTypes)
    session = quail.Session(CONFIG, tokenizer=fake_tok, registry=registry)
    session.register("reviews", quail.DocumentProvider.from_table(
        pa.table({"id": ["r0", "r1"], "review": ["first", "second"]}), id_col="id"))
    result = _run(session.sql(
        "SELECT r.id FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review)) LIMIT 1"
    ), make_executor({"r": {"q:": [1, 0]}}))
    assert result.observer(NodeTypes)["types"] == [
        "quail.scan", "quail.ai_filter", "quail.project", "quail.limit"]
    assert result.observer(NodeTypes) is result.observer("test.node_types")
    with pytest.raises(KeyError):
        result.observer("example.missing")

    # saved reports can be loaded separately from their result tables
    restored = QueryResult.from_table(result.collect(), report=dict(result.report))
    assert restored.plan is None
    restored.attach_executed_plan(registry.codecs)
    assert [node.node_id for node in restored.plan.topological_nodes()] == [
        node.node_id for node in result.plan.topological_nodes()]
    assert restored.node_metrics == result.node_metrics
    assert restored.explain() == result.explain()

    # results are assembled from survivor indices and join tables
    true_join_tables = [
        document_index_table({"r1": [0, 1], "a1": [0, 0]}, "join_answers"),
        document_index_table({"r2": [0, 1], "a1": [0, 0]}, "join_answers"),
        document_index_table({"r2": [0, 1, 1], "a2": [0, 0, 1]}, "join_answers"),
    ]
    survivors = {alias: pa.array(rows, type=pa.int32()) for alias, rows in
                 {"r1": [0, 1], "r2": [0, 1], "a1": [0], "a2": [0, 1]}.items()}
    declaration, _ = build_result_declaration(true_join_tables, survivors, "r1")
    assert count_rows(declaration) == 6

    relation = document_index_table({"r": list(range(5))}, "filter_survivors")
    assert relation.schema.metadata[b"quail.kind"] == b"filter_survivors"
    survivors = {"r": pa.array(range(5), type=pa.int32())}
    declaration, index_schema = build_result_declaration([], survivors, "r")
    schema = pa.schema([pa.field("r.id", pa.string(), nullable=False)],
                       metadata={b"quail.kind": b"query_result"})
    result = QueryResult(
        columns=["r.id"], declaration=declaration,
        document_index_schema=index_schema, output_schema=schema,
        projection=[("r", pa.array([f"r{i}" for i in range(5)]))],
        report={}, survivor_indices=survivors, true_join_tables={})

    reader = result.execute_stream(batch_rows=2)
    batches = list(reader)
    assert isinstance(reader, pa.RecordBatchReader)
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert pa.Table.from_batches(batches).column("r.id").to_pylist() == [
        "r0", "r1", "r2", "r3", "r4"]
    assert result.count() == 5
    assert result.collect(limit=3).column("r.id").to_pylist() == ["r0", "r1", "r2"]
    assert not hasattr(result, "rows")


class NodeTypes:
    """Observer that records the node types it saw."""

    name = "test.node_types"

    def __init__(self):
        self.types = []

    def after_node(self, node, result) -> None:
        self.types.append(node.type_name)

    def report(self) -> dict:
        return {"types": list(self.types)}


def _pair_query(session, on):
    claims = session.docs("claims").alias("c")
    partner = session.docs("evidence").alias("e")
    query = (claims.join(partner, on=quail.col("c.url") == quail.col("e.url"))
             if on else claims.join(partner))
    return query.ai_filter(
        quail.prompt("Does {1} support {0}?", quail.col("c.claim"),
                     quail.col("e.text")),
        selectivity=0.5).select("c.id", "e.id")


def test_pair_table_lists_equal_keys_once_and_the_session_ships_it():
    pairs = pair_table(
        "c", [pa.array(["u1", "u2", None, "u1"])],
        "e", [pa.array(["u2", "u1", "u1"])])
    assert pairs.to_pydict() == {"c": [0, 0, 1, 3, 3], "e": [1, 2, 0, 1, 2]}
    assert pair_fraction(pairs, 4, 3) == 5 / 12
    assert partner_map(pairs, "e", "c") == {0: [1], 1: [0, 3], 2: [0, 3]}
    # two equalities: both key columns must match; integer keys on one
    # side are cast to the other side's type
    pairs = pair_table(
        "c", [pa.array(["u1", "u1"]), pa.array([1, 2])],
        "e", [pa.array(["u1", "u1"]), pa.array([2, 2], type=pa.int8())])
    assert pairs.to_pydict() == {"c": [1, 1], "e": [0, 1]}

    with quail.Session(CONFIG, tokenizer=lambda text: list(text.encode())) as session:
        register_claims_evidence(session)
        paired = _pair_query(session, on=True)
        cross = _pair_query(session, on=False)
        paired_plan, cross_plan = paired.plan(), cross.plan()
        paired_stage = paired_plan.graph.nodes_by_type(AiJoin.type_name)[0].stages[0]
        cross_stage = cross_plan.graph.nodes_by_type(AiJoin.type_name)[0].stages[0]
        # c1 and c2 pair with e0 and e2, c0 with e1: 5 of 12 pairs
        hash_join = paired_plan.graph.node("hash_join:c-e")
        assert (hash_join.left, hash_join.right, hash_join.on) == (
            "c", "e", (("url", "url"),))
        assert hash_join.pair_fraction == 5 / 12
        assert [port.source.node_id for port in hash_join.inputs] == [
            "scan:c", "scan:e"]
        assert paired_plan.graph.node("scan:c").shards == (range(0, 4),)
        assert paired_stage.pairs_from == "hash_join:c-e"
        assert not cross_stage.pairs_from
        assert "hash_join:c-e" not in {node.node_id for node in cross_plan.nodes}
        assert paired_stage.expected_tuples == round(
            cross_stage.expected_tuples * 5 / 12, 1)
        assert paired_plan.estimated_seconds < cross_plan.estimated_seconds
        assert "HashJoin" in paired.explain() and "c.url = e.url" in paired.explain()
        # the request carries the key columns, not the pairs
        request = paired._prepare_physical()
        assert request.column_tables()["c"].column("url").to_pylist() == [
            "u0", "u1", "u1", "u9"]
        assert cross._prepare_physical().relations == {}

        def answer(prompt, assignment):
            return (assignment["c"] + assignment["e"]) % 2 == 0

        estimate = quail.speed_of_light_estimate(paired, answer)
        assert estimate.join_pair_evaluations == 5
        assert estimate.join_stages[0]["passing_pairs"] == 2
        assert quail.speed_of_light_estimate(
            cross, answer).join_pair_evaluations == 12


IMPORT_TEXT = """
import sys
import quail
from quail.bench import quailb
from quail.execution import execute
from demos import quickstart

assert "modal" not in sys.modules
"""


def test_engine_import_gpu_requirement_and_quickstart_demos(
        sess, monkeypatch, tmp_path):
    subprocess.run([sys.executable, "-c", IMPORT_TEXT], check=True)
    monkeypatch.setattr(execution, "gpu_problem", lambda: "no CUDA GPU is visible")
    with pytest.raises(RuntimeError, match="process with a CUDA GPU") as error:
        sess.sql(FILTER_SQL).run()
    assert "no CUDA GPU is visible" in str(error.value)

    reviews = pa.table({"id": ["rv0", "rv1"], "body": [
        "The acting was excellent.", "The acting was poor."]})
    monkeypatch.setattr(quickstart, "load_reviews", lambda: reviews)
    commits = []
    monkeypatch.setattr(quickstart_modal, "RESULTS_DIR", tmp_path / "chosen-path")
    monkeypatch.setattr(quickstart_modal.volume, "commit",
                        lambda: commits.append("volume"))
    monkeypatch.setattr(quail.Session, "tokenizer", property(lambda self: str.split))
    monkeypatch.setattr(execution, "gpu_problem", lambda: None)
    monkeypatch.setattr(execution, "_prepare_backend", lambda *args: None)
    executor = make_executor({"r": {"Instruction": [1, 0]}})
    monkeypatch.setattr(execution, "_execute_physical",
                        lambda request, registry: executor(request))
    closed = []
    close = quail.Session.close

    def record_close(session):
        close(session)
        closed.append(session)

    monkeypatch.setattr(quail.Session, "close", record_close)

    for on_modal in (False, True):
        closed.clear()
        commits.clear()
        run = (quickstart_modal.run_query.get_raw_f() if on_modal
               else quickstart.run_query)
        rows, report = run()
        assert len(closed) == 1, on_modal
        assert closed[0]._token_directory is None, on_modal
        assert rows.to_pylist() == [{"r.id": "rv0"}], on_modal
        assert report["fresh_tokens"] == 1234, on_modal
        assert report["worker_total_s"] >= 0, on_modal
        if not on_modal:
            assert "result_volume_path" not in report
            assert commits == []
            continue
        path = Path(report["result_volume_path"])
        assert path.parent == tmp_path / "chosen-path"
        assert json.loads(path.read_text()) == report
        assert commits == ["volume"]

    def fail():
        raise RuntimeError("query failed")

    commits.clear()
    monkeypatch.setattr(quickstart, "run_query", fail)
    with pytest.raises(RuntimeError, match="query failed"):
        quickstart_modal.run_query.get_raw_f()()
    assert commits == ["volume"], "a failed query still commits the volume"
