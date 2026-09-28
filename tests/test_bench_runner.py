"""CPU checks for Quail's QUAIL-B runner: plans to queries, results to ids."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quail
from quail.bench.quailb import (
    answer_oracle,
    build_query,
    join_anchors,
    prompt_pieces,
    queries,
    run_output,
)
from quail.bench.substrait import AI_URN, Filter, Join, Relation, read_plan
from quail.builtins import built_in_registry
from quail.catalog import DocumentProvider
from quail.execution.execute import execute_query
from quail.execution.result import answer_table
from quail.execution.runner import NodeMetrics, NodeResult, RunResult
from quail.execution.types import PhysicalResponse, export_physical_outputs
from quail.logical.prompts import render_join_prompt_text
from quail.physical import AiFilter, AiJoin, RequestExecution, decode_graph
from quail.physical import Scan as PhysicalScan
from quail.planner.plan import EngineConfig, Refusal
from quail.specs import QWEN3_4B_FP8
from quail_b.labels import GroundTruthCollection, PredicateLabels
from quail_b.minimum import DocumentTokens, token_metrics
from quail_b.predicates import PREDICATES, predicate_payload
from quail_b.prompts import DISCUSS_ASPECT, F1, F4, F11, F13, REFUTE, SUPPORT
from quail_b.queries import get_query
from quail_b.queries import queries as query_specs
from quail_b.rendering import render_filter_prompt, render_join_prompt
from quail_b.scoring import evaluate

# IMDB-3 is F1 on the reviews, then the DISCUSS_ASPECT join with the aspects
SPEC = get_query("IMDB-3")
CORPUS = {
    "reviews": pa.table({"id": ["r0", "r1"], "body": ["good film", "bad film"]}),
    "aspects": pa.table({"id": ["a0", "a1"], "aspect": ["acting", "ending"]}),
}


def _labels(key, template, answers, right_table=None):
    predicate = {
        "key": key, "template": template,
        "kind": "join" if right_table else "filter",
        "left_table": "reviews", "left_column": "body",
        "right_table": right_table, "right_column": right_table and "aspect",
    }
    return PredicateLabels(key, f"ls_{key}", predicate, answers,
                           {"qwen3-32b-fp8": len(answers)})


def _truth():
    predicates = (
        _labels("test.review.filter", F1, {("r0", None): True, ("r1", None): False}),
        _labels("test.review.aspect", DISCUSS_ASPECT, {
            ("r0", "a0"): True, ("r0", "a1"): False,
            ("r1", "a0"): False, ("r1", "a1"): True,
        }, "aspects"),
    )
    return GroundTruthCollection("gt_test", "c_test", 0.1, "qwen3-32b-fp8",
                                 {labels.key: labels for labels in predicates})


def fever_truth():
    """Labels for FEV-9 over a three claim, three evidence corpus."""
    corpus = {
        "claims": pa.table({"id": ["c0", "c1", "c2"],
                            "claim": ["person one", "person two", "a place"]}),
        "evidence": pa.table({"id": ["e0", "e1", "e2"],
                              "text": ["person one", "person two", "a place"]}),
    }
    true_pairs = {
        SUPPORT: {("c0", "e0"), ("c1", "e1"), ("c2", "e0"), ("c1", "e2")},
        REFUTE: {("c1", "e0"), ("c2", "e0"), ("c0", "e2")},
    }
    predicates = {}
    for spec in PREDICATES:
        if spec.template not in (F11, F13, SUPPORT, REFUTE):
            continue
        ids = corpus[spec.left_table]["id"].to_pylist()
        answers = (
            {(row_id, None): index < 2 for index, row_id in enumerate(ids)}
            if spec.kind == "filter" else {
                (left, right): (left, right) in true_pairs[spec.template]
                for left in ids
                for right in corpus[spec.right_table]["id"].to_pylist()
            }
        )
        predicates[spec.key] = PredicateLabels(
            spec.key, f"ls_{spec.key}", predicate_payload(spec), answers, {},
        )
    truth = GroundTruthCollection("gt_fev9", "c_fev9", 0.1, None, predicates)
    return corpus, truth


def _token_ids(text):
    return [byte + 1 for byte in text.encode("utf-8")]


def _config(backend):
    return EngineConfig(gpus=1, model="qwen3-4b-fp8", backend=backend,
                        device="h100-sxm")


def _session(tmp_path, backend="quail"):
    for name, table in CORPUS.items():
        pq.write_table(table, tmp_path / f"{name}.parquet")
    tokenizer = str.split if backend == "quail" else _token_ids
    sess = quail.Session(_config(backend), tokenizer=tokenizer)
    for name in CORPUS:
        sess.register(name, DocumentProvider.from_parquet(
            str(tmp_path / f"{name}.parquet"), id_col="id"))
    return sess


def _query(sess):
    # the same shape as SPEC, with the anchor pinned so the fake
    # executor below knows which alias the join keeps
    return (sess.docs("reviews").alias("r")
            .ai_filter(quail.prompt(F1, quail.col("r.body")))
            .ai_join(
                sess.docs("aspects").alias("a"),
                quail.prompt(DISCUSS_ASPECT, quail.col("r.body"),
                             quail.col("a.aspect")),
                anchor="r")
            .select("r.id", "a.id"))


def _execute(query, nodes, **metrics):
    def execute(request):
        graph = decode_graph(request.plan["graph"], built_in_registry().codecs)
        outputs = export_physical_outputs(
            graph, RunResult(None, nodes(graph, request), NodeMetrics()))
        return PhysicalResponse(outputs, {
            "wall_s": 2.0, "boot_s": 3.0, "boot_kind": "cold", "boot": {},
            "fresh_tokens": 2000, "peak_gib": 1.0, **metrics})

    return execute_query(query, physical_executor=execute)


def _run(query, join_answers):
    def nodes(graph, request):
        filtered = next(node for node in graph.nodes if isinstance(node, AiFilter))
        anchored = next(node for node in graph.nodes if isinstance(node, AiJoin))
        join = anchored.stages[0].runtime_spec()
        return {
            **{node.node_id: NodeResult({f"ids:{node.alias}": range(
                len(request.inputs[node.input_id].documents))})
               for node in graph.nodes if isinstance(node, PhysicalScan)},
            filtered.node_id: NodeResult({
                "ids:r": [0],
                "filter_answers:r": {0: [1], 1: [0]},
            }),
            anchored.node_id: NodeResult({
                "ids:r": [0] if any(join_answers) else [],
                "join_answers:0": {
                    "rows": {0: join_answers},
                    "anchor_index": [0],
                    "partner_index": [[0], [1]],
                    "anchor": "r",
                    "partners": ["a"],
                    "semantics": "full",
                    "selectivity": join["selectivity"],
                    "written_pos": 0,
                },
            }),
        }

    return _execute(query, nodes, store=None)


def _run_request_backend(query, join_answers):
    def nodes(graph, request):
        model = next(
            node for node in graph.nodes if isinstance(node, RequestExecution))
        filter_table = pa.table({
            "r": pa.array([0, 1], type=pa.int32()),
            "predicate": pa.array([0, 0], type=pa.int32()),
            "answer": pa.array([True, False], type=pa.bool_()),
        }).replace_schema_metadata({
            b"quail.kind": b"filter_answers",
            b"quail.alias": b"r",
        })
        join_table = answer_table(
            {"r": [0, 0], "a": [0, 1]},
            [bool(answer) for answer in join_answers],
            "join_answers",
            metadata={"anchor": "r", "partners": "a", "semantics": "full",
                      "written_pos": 0},
        )
        true_aspects = [
            index for index, answer in enumerate(join_answers) if answer
        ]
        return {model.node_id: NodeResult({
            "ids:r": [0] if true_aspects else [],
            "ids:a": true_aspects,
            "filter_answers:r": filter_table,
            "join_answers:0": join_table,
        })}

    return _execute(query, nodes, cached_tokens=0)


def test_read_plan_reads_operators_and_rejects_other_extensions():
    plan = read_plan(get_query("IMDB-4").plan)
    assert plan.relations == (Relation("r", "reviews"), Relation("a", "aspects"))
    assert plan.operators == (
        Filter("filter-1", "r", "body", F1),
        Filter("filter-2", "r", "body", F4),
        Join("join-1", ("r", "a"), ("body", "aspect"), DISCUSS_ASPECT),
    )
    assert plan.select == ("r.id", "a.id")
    assert plan.filter_id("r", 1) == "filter-2"
    assert plan.join_id(0) == "join-1"

    # FEV-10 asks SUPPORT only of a claim and its own Wikipedia page
    (join,) = read_plan(get_query("FEV-10").plan).joins
    assert join.aliases == ("c", "e")
    assert join.on == (("evidence_wiki_url", "id"),)

    for spec in query_specs(include_privacy=True).values():
        plan = read_plan(spec.plan)
        assert len(plan.relations) == len(plan.joins) + 1, spec.id
        ids = [op.id for op in plan.operators]
        assert len(set(ids)) == len(ids), spec.id
        assert all(name.endswith(".id") for name in plan.select), spec.id

    plan = get_query("IMDB-1").plan
    plan.extension_urns[0].urn = AI_URN + ".other"
    with pytest.raises(ValueError, match="ai_filter"):
        read_plan(plan)


def test_benchmark_results_and_scoring(tmp_path):
    plan = read_plan(SPEC.plan)
    with _session(tmp_path) as sess:
        quail_result = _run(_query(sess), [1, 0])
    output = run_output(quail_result, plan, CORPUS)

    assert output.filter_answers["filter-1"].to_pydict() == {
        "r": ["r0", "r1"], "answer": [True, False]}
    assert output.join_answers["join-1"].to_pydict() == {
        "r": ["r0", "r0"], "a": ["a0", "a1"], "answer": [True, False]}
    assert output.rows.to_pydict() == {"r": ["r0"], "a": ["a0"]}
    evaluation = evaluate(SPEC, output, _truth(), CORPUS)
    assert evaluation["answer_accuracy"]["accuracy"] == 1.0
    assert evaluation["output_accuracy"]["exact_match"] is True

    with _session(tmp_path, backend="stock_vllm") as sess:
        query = _query(sess)
        result = _run_request_backend(query, [0, 0])
        output = run_output(result, plan, CORPUS)
        output.prompt_pieces = prompt_pieces(query, plan, join_anchors(result))
        tokenizer_name = sess.model.hf_name

    evaluation = evaluate(SPEC, output, _truth(), CORPUS)
    assert evaluation["answer_accuracy"]["evaluated"] == 4
    assert evaluation["answer_accuracy"]["correct"] == 3
    assert [item["op"] for item in evaluation["per_predicate"]] == ["filter", "join"]
    assert output.rows.num_rows == 0
    pieces = output.prompt_pieces
    assert pieces["tokenizer"] == tokenizer_name
    # the model's user turn opens ahead of the document preamble
    assert pieces["preamble"] == _token_ids(
        sess.model.turn_prefix + quail.SHARED_PRE)
    assert [item["id"] for item in pieces["filters"]] == ["filter-1"]
    assert [(item["id"], item["anchor"]) for item in pieces["joins"]] == [
        ("join-1", "r")]
    stores = {tokenizer_name: DocumentTokens(
        CORPUS, lambda texts: [_token_ids(text) for text in texts])}
    measured = token_metrics(SPEC, output, CORPUS, stores)
    assert measured["minimum_tokens"] > 0
    assert measured["regret_tokens"] == 2000 - measured["minimum_tokens"]

    # a plan that projects the review text instead of its id
    text_plan = SPEC.plan
    project = text_plan.relations[0].root.input.project
    project.expressions[0].selection.direct_reference.struct_field.field = 1
    text_plan = read_plan(text_plan)
    assert text_plan.select == ("r.body", "a.id")
    with pytest.raises(NotImplementedError):
        run_output(quail_result, text_plan, CORPUS)


def test_benchmark_query_prompts_labels_and_raw_rendering():
    for backend in ("quail", "stock_vllm", "pipelined_vllm", "pipelined_sglang"):
        corpus, truth = fever_truth()
        spec = get_query("FEV-9")
        plan = read_plan(spec.plan)
        answer = answer_oracle(truth, corpus)
        with quail.Session(
            _config(backend), tokenizer=lambda text: list(text.encode("utf-8"))
        ) as session:
            for name, table in corpus.items():
                session.register(name, DocumentProvider.from_table(table, id_col="id"))
            query = build_query(session, spec)
            assert not isinstance(query.plan(), Refusal), backend
            operators = query.logical.operators()
            filters, joins = operators.filters, operators.joins
            tables = {relation.alias: relation.table for relation in plan.relations}
            assert [scan.alias for scan in operators.scans] == list(tables), backend
            assert {alias: [p.prompt.template for p in chain]
                    for alias, chain in filters.items()} == {
                alias: [
                    quail.bind_prompt(item.prompt, (quail.ColumnRef(
                        alias, tables[alias], item.column),)).template
                    for item in plan.filters if item.alias == alias]
                for alias in tables if any(
                    item.alias == alias for item in plan.filters)}, backend
            assert [tuple(arg.alias for arg in join.prompt.args)
                    for join in joins] == [join.aliases for join in plan.joins]
            # c0 is about a person and e0 supports it; c2 is not about a person
            assert answer(filters["c1"][0].prompt, {"c1": 0}) is True, backend
            assert answer(filters["c1"][0].prompt, {"c1": 2}) is False, backend
            assert answer(joins[0].prompt, {"c1": 0, "e1": 0}) is True, backend
            assert answer(joins[0].prompt, {"c1": 0, "e1": 1}) is False, backend
            # the same labels drive the speed of light estimate
            estimate = quail.speed_of_light_estimate(query, answer)
            assert estimate.post_filter_counts == {
                "c1": 2, "e1": 2, "c2": 2, "e2": 2}, backend
            assert len(estimate.join_stages) == 3, backend
            assert queries(session)["FEV-9"][0] == spec.description, backend
            # only the FEVER tables are registered
            assert all(query_id.startswith("FEV-")
                       for query_id in queries(session)), backend
    # the benchmark renders the same raw prompt text as Quail
    turn = QWEN3_4B_FP8.turn
    for spec in PREDICATES:
        left = quail.ColumnRef("left", spec.left_table, spec.left_column)
        if spec.kind == "filter":
            prompt = quail.bind_prompt(spec.template, (left,), turn=turn)
            assert prompt.tail.startswith("{0}")
            expected = (prompt.preamble + "doc one"
                        + prompt.tail.replace("{0}", "", 1))
            assert render_filter_prompt(spec.template, "doc one") == expected
        else:
            right = quail.ColumnRef("right", spec.right_table, spec.right_column)
            prompt = quail.bind_join_prompt(spec.template, (left, right), turn=turn)
            documents = ("doc one", "doc two")
            for anchor in (0, 1):
                assert render_join_prompt(spec.template, documents, anchor) == (
                    render_join_prompt_text(prompt, documents, anchor))
