"""AI.SCORE parsing, planning, execution, and score semantics."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quail
from quail.catalog import Catalog, DocumentProvider
from quail.execution.execute import execute_query
from quail.execution.reranker import (
    RerankerBatch,
    RerankerModelExecution,
    ScoreFilterRuntime,
    ScoreRows,
    compare_score,
    score_in_batches,
)
from quail.execution.runner import (
    ExecutionContext,
    GenericRunner,
    compute_subgraph,
    scalar_node_metrics,
)
from quail.execution.types import PhysicalResponse, export_physical_outputs
from quail.frontend.sql import compile_sql
from quail.logical import CompileError, SemanticJoin
from quail.physical import AiScore, ScoreFilter, decode_graph, encode_graph
from quail.planner.plan import EngineConfig, Refusal

PROJECTION_SQL = (
    "SELECT d.id, "
    "AI.SCORE(PROMPT('Requests a refund: {0}', d.body)) AS score "
    "FROM documents d"
)

GENERATIVE_MODELS = ("qwen3-4b-fp8", "qwen3-32b-fp8", "diffusion-gemma-26b-a4b-fp8")

QUERY_TEXTS = [
    ("Does {0} ask for a refund?", "Does this document ask for a refund?"),
    ("Is {0} a refund request? Does {0} ask for money?",
     "Is this document a refund request? Does this document ask for money?"),
    ("Refund request?\n\n{0}", "Refund request?"),
    ("{0}\n\nRefund request?", "Refund request?"),
]

COMPARISONS = [
    ("<", [True, False, False]), ("<=", [True, True, False]),
    (">", [False, False, True]), (">=", [False, True, True]),
]

SHAPE_ERRORS = [
    (
        "SELECT q.id FROM queries q JOIN documents d ON AI.SCORE(PROMPT("
        "'Compare {0} with {1}; is {1} about {0}?', q.text, d.body)) >= 0.5",
        "exactly once",
    ),
    (
        "SELECT d.id, AI.SCORE(PROMPT('Refund? {0}', d.body)) AS d "
        "FROM documents d",
        "also a table alias",
    ),
    (
        "SELECT AI.SCORE(PROMPT('Refund? {0}', d.body)) AS a, "
        "AI.SCORE(PROMPT('Refund? {0}', d.body)) AS b FROM documents d",
        "project it once",
    ),
    (
        "SELECT d.id FROM documents d WHERE AI.SCORE(PROMPT('Refund? {0",
        "parse error",
    ),
    (
        "SELECT q.id, AI.SCORE(PROMPT('Is {1} about {0}?', q.text, d.body)) "
        "AS s FROM queries q JOIN documents d ON q.id = d.id",
        "runs over CROSS JOIN",
    ),
    ("SELECT AI.SCORE(PROMPT('refund {0}', d.body)) FROM documents d", "AI.SCORE"),
    (
        "SELECT d.id FROM documents d "
        "WHERE AI.SCORE(PROMPT('refund {0}', d.body))",
        "AI.SCORE",
    ),
    (
        "SELECT d.id FROM documents d "
        "WHERE AI.SCORE(PROMPT('refund {0}', d.body)) = 1",
        "AI.SCORE",
    ),
]


def _tokens(text):
    return text.split()


def _provider(path, columns, *, id_col):
    pq.write_table(pa.table(columns), path)
    return DocumentProvider.from_parquet(str(path), id_col=id_col)


@pytest.fixture()
def catalog(tmp_path):
    value = Catalog()
    value.register("documents", _provider(
        tmp_path / "documents.parquet",
        {"id": [1, 2], "body": ["refund please", "all good"]},
        id_col="id",
    ))
    value.register("queries", _provider(
        tmp_path / "queries.parquet",
        {"id": [1, 2], "text": ["refund", "shipping"]},
        id_col="id",
    ))
    return value


def _session(
    catalog, *, model="qwen3-reranker-0.6b-bf16", gpus=1, tokenizer=_tokens
):
    session = quail.Session(
        EngineConfig(model=model, device="h100-sxm", gpus=gpus),
        tokenizer=tokenizer,
    )
    for name in ("documents", "queries"):
        session.register(name, catalog.get(name))
    return session


def _score_sql(pair):
    """Return the prompt template and a projection query over it."""
    template = "Is {1} relevant to {0}?" if pair else "Refund? {0}"
    sql = (
        f"SELECT AI.SCORE(PROMPT('{template}', q.text, d.body)) "
        "AS score FROM queries q CROSS JOIN documents d"
        if pair else
        f"SELECT AI.SCORE(PROMPT('{template}', d.body)) AS score "
        "FROM documents d"
    )
    return template, sql


class _FakeReranker:
    def __init__(self, scores):
        self.scores = tuple(scores)

    def score(self, spec, rows, documents):
        self.prompts = []
        for row in rows:
            tokens = list(spec.prompt_token_parts[0])
            for index, alias in enumerate(spec.aliases):
                tokens.extend(documents[alias][row[index]])
                tokens.extend(spec.prompt_token_parts[index + 1])
            self.prompts.append(tokens)
        assert len(rows) == len(self.scores)
        return RerankerBatch(self.scores, fresh_tokens=12, cached_tokens=3)


class _RowReranker:
    def __init__(self):
        self.calls = 0

    def score(self, spec, rows, documents):
        self.calls += 1
        scores = [(int(sum(row)) + 1) / 10 for row in rows]
        return RerankerBatch(scores, fresh_tokens=len(rows), cached_tokens=0)


def _run_graph(session, query, request, reranker):
    graph = compute_subgraph(query.plan().graph)
    model = RerankerModelExecution.__new__(RerankerModelExecution)
    model.reranker = reranker
    model.documents = {
        node.alias: request.inputs[node.input_id].documents
        for node in query.plan().nodes if node.type_name == "quail.scan"
    }
    run = GenericRunner().run(
        graph,
        ExecutionContext(
            runtimes=session.registry.runtimes,
            model_execution=model,
            sources={
                **{alias: range(2) for alias in model.documents},
                **request.relations,
            },
        ),
    )
    return graph, model, run


def _finish(query, session, reranker):
    def execute(request):
        graph, _, run = _run_graph(session, query, request, reranker)
        return PhysicalResponse(
            export_physical_outputs(graph, run),
            {"backend": "quail", "wall_s": 0.1, "fresh_tokens": 12,
             "cached_tokens": 3, "node_metrics": scalar_node_metrics(run.nodes)},
        )

    return execute_query(query, physical_executor=execute)


def test_score_plan_shape_query_text_and_round_trip(catalog):
    session = _session(catalog)
    query = session.sql(
        f"{PROJECTION_SQL} "
        "WHERE AI.SCORE("
        "PROMPT('Requests a refund: {0}', d.body)"
        ") >= 0.7"
    )
    physical = query.plan()
    assert sum(isinstance(node, AiScore) for node in physical.nodes) == 1
    score_filter = next(
        node for node in physical.nodes if isinstance(node, ScoreFilter)
    )
    assert score_filter.score_name == "score"
    assert (score_filter.comparison, score_filter.threshold) == (">=", 0.7)
    graph = physical.graph
    decoded = decode_graph(
        encode_graph(graph, session.registry.codecs),
        session.registry.codecs,
    )
    assert decoded == graph

    for template, query_text in QUERY_TEXTS:
        query = session.sql(
            f"SELECT d.id, AI.SCORE(PROMPT('{template}', d.body)) AS s "
            "FROM documents d"
        )
        score = next(node for node in query.plan().nodes if isinstance(node, AiScore))
        assert score.spec.query_template == query_text, template
    session.close()

    plan = compile_sql(
        "SELECT q.id, d.id FROM queries q JOIN documents d ON "
        "0.8 > AI.SCORE(PROMPT('Is {1} relevant to {0}', q.text, d.body)) "
        "WHERE 0.5 <= AI.SCORE(PROMPT('Refund? {0}', d.body))",
        catalog,
        _tokens,
    )
    join = plan.root.input
    assert isinstance(join, SemanticJoin)
    assert (join.predicate.comparison, join.predicate.threshold) == ("<", 0.8)
    assert join.predicate.call.aliases() == ("q", "d")
    (predicate,) = plan.operators().filters["d"]
    expression = predicate.expression
    assert (expression.comparison, expression.threshold) == (">=", 0.5)


def test_score_cost_counts_canvas_rows_anchor_prefixes_and_throughput(catalog):
    from quail.backends.quail.graph import throughput
    from quail.cost.work import ask, scan
    from quail.execution.runner import NodeMetrics
    from quail.explain import run_summary
    from quail.planner.reranker import _score_work

    for model, canvas in (("qwen3-4b-fp8", 0), ("diffusion-gemma-26b-a4b-fp8", 1)):
        session = _session(catalog, model=model, tokenizer=list)
        physical = session.sql(_score_sql(False)[1]).plan()
        session.close()
        head, tail = next(node for node in physical.nodes
                          if isinstance(node, AiScore)).spec.prompt_token_parts
        documents = len("refund please") + len("all good")
        # the second document reads the shared head from KV
        expected = documents + 2 * (len(head) + len(tail) + canvas) - len(head)
        assert physical.settings["estimated_fresh_tokens"] == \
            pytest.approx(expected), model

    # Two query documents, three candidates each, and a shared prompt.
    work = _score_work(
        6, 15, 9, prefix_tokens=17, groups=2, shared_tokens=3,
    )
    expected = scan(0, 24) + ask(3, 21) + ask(17, 7) * 4
    assert work == expected
    assert _score_work(0, 15, 9).tokens == 0

    session = _session(catalog)
    query = session.sql(
        "SELECT d.id FROM documents d "
        "WHERE AI.SCORE(PROMPT('Refund? {0}', d.body)) >= 0.5"
    )
    graph = query.plan().graph
    assert throughput(graph, NodeMetrics(evaluated_documents=5), 2.0) == {
        "documents_per_second": 1.0}
    assert throughput(graph, NodeMetrics(evaluated_document_pairs=6), 2.0) == {
        "document_pairs_per_second": 3.0}
    lines = run_summary(
        {"wall_s": 2.0, "fresh_tokens": 10, "evaluated_document_pairs": 6},
        graph, 1,
    )
    assert any("3 document pairs/second" in line for line in lines)
    session.close()


def test_score_filters_compare_values_and_keep_pair_answers(catalog):
    node = ScoreFilter(node_id="filter", score_name="score", aliases=("q", "d"),
                       comparison=">", threshold=0.5, written_pos=0)
    table = pa.table({
        "q": pa.array([0, 0, 1, 1], pa.int32()),
        "d": pa.array([0, 1, 0, 1], pa.int32()),
        "score": pa.array([0.9, 0.1, 0.2, 0.8], pa.float64()),
    })
    result = ScoreFilterRuntime().execute(
        node, {"input:0": table}, ExecutionContext(runtimes={})
    )
    answers = result.outputs["join_answers:0"]
    assert answers.column("answer").to_pylist() == [True, False, False, True]
    assert result.outputs["scores"].num_rows == 2

    for comparison, expected in COMPARISONS:
        answer = compare_score(
            pa.chunked_array([[0.25, 0.5], [0.75]]), comparison, 0.5)
        assert isinstance(answer, pa.ChunkedArray), comparison
        assert answer.to_pylist() == expected, comparison

    session = _session(catalog)
    query = session.sql(
        "SELECT q.id, d.id, "
        "AI.SCORE(PROMPT('Refund? {0}', d.body)) AS s1, "
        "AI.SCORE(PROMPT('Is {1} relevant to {0}?', q.text, d.body)) AS s2 "
        "FROM queries q CROSS JOIN documents d "
        "WHERE AI.SCORE(PROMPT('Refund? {0}', d.body)) >= 0.15"
    )
    result = _finish(query, session, _RowReranker())
    assert result.collect().to_pylist() == [
        {"q.id": 1, "d.id": 2, "s1": 0.2, "s2": 0.2},
        {"q.id": 2, "d.id": 2, "s1": 0.2, "s2": 0.3},
    ]

    query = session.sql(
        "SELECT d.id, AI.SCORE(PROMPT('Refund? {0}', d.body)) AS s "
        "FROM documents d "
        "WHERE AI.SCORE(PROMPT('Refund? {0}', d.body)) >= 0.15 "
        "AND AI.SCORE(PROMPT('Refund? {0}', d.body)) <= 0.25"
    )
    physical = query.plan()
    assert sum(isinstance(node, AiScore) for node in physical.nodes) == 1
    assert sum(isinstance(node, ScoreFilter) for node in physical.nodes) == 2
    reranker = _RowReranker()
    result = _finish(query, session, reranker)
    assert result.collect().to_pylist() == [{"d.id": 2, "s": 0.2}]
    assert result.schema.field("s").type == pa.float64()
    assert result.report["estimated_seconds"] > 0
    assert reranker.calls == 1
    session.close()


def test_scores_stream_per_batch_and_score_rows_shard_in_order(catalog):
    from quail.progress import set_answer_sink

    session = _session(catalog)
    query = session.sql(PROJECTION_SQL)
    streamed = []
    set_answer_sink(streamed.append)
    try:
        graph, model, _ = _run_graph(
            session, query, query._prepare_physical(), _RowReranker())
        score_node = next(
            node for node in graph.nodes if isinstance(node, AiScore))
        result = score_in_batches(
            score_node,
            {port.name: np.arange(2, dtype=np.int32) for port in score_node.inputs},
            lambda node, batches: [model.execute_rows(node, b) for b in batches],
            batch_rows=1)
    finally:
        set_answer_sink(None)
    session.close()
    assert streamed[0] == {
        "kind": "score", "node": score_node.node_id, "output": "score",
        "aliases": ["d"], "rows": [0, 1], "scores": [0.1, 0.2]}
    assert [entry["rows"] for entry in streamed[1:]] == [[0], [1]]
    assert result.outputs["scores"].column("d").to_pylist() == [0, 1]

    rows = ScoreRows((np.arange(100_000), np.arange(100_000)), product=True)
    assert len(rows) == 10_000_000_000
    first = next(rows.batches(size=7))
    np.testing.assert_array_equal(first, np.column_stack((np.zeros(7), np.arange(7))))
    rows = ScoreRows((np.array([2, 0]), np.array([4, 1, 3])), product=True)
    np.testing.assert_array_equal(np.concatenate(list(rows.batches(size=4))),
                                  [[2, 4], [2, 1], [2, 3], [0, 4], [0, 1], [0, 3]])
    empty = ScoreRows((np.empty(0, dtype=np.int32), np.arange(3)), product=True)
    assert list(empty.batches())[0].shape == (0, 2)
    rows = ScoreRows((np.array([3, 0, 1]), np.array([7, 8])), product=True)
    even, odd = rows.shard(2)
    assert even.columns[0].tolist() == [0] and odd.columns[0].tolist() == [3, 1]
    assert odd.positions(0, 4).tolist() == [0, 1, 4, 5]
    assert even.positions(0, 2).tolist() == [2, 3]
    plain = ScoreRows((np.array([3, 0, 1]),))
    assert [shard.positions(0, len(shard)).tolist() for shard in plain.shard(2)] \
        == [[1], [0, 2]]
    # a product batch ends on a first-document boundary when partners fit
    assert [len(batch) for batch in rows.batches(size=5)] == [4, 2]


def test_native_and_distributed_scores_share_prefixes_and_keep_pair_order(
        catalog, monkeypatch):
    import pickle

    from quail.backends.quail.distributed import DistributedQuailExecution
    from quail.backends.quail.executor import score as module
    from quail.backends.quail.worker import quail_runtime_payload
    from quail.execution.tokens import TokenView

    documents = {
        "a": [TokenView(pa.array([10, 11], type=pa.int32()))],
        "b": [TokenView(pa.array([20], type=pa.int32())),
              TokenView(pa.array([30], type=pa.int32()))],
    }
    spec = SimpleNamespace(
        name="score", aliases=("a", "b"),
        prompt_token_parts=((1,), (2,), (3,)),
    )
    state = dict(torch=object(), arena=object(), pipeline=object(),
                 answer_rows=object(), chunk_tokens=1234)
    monkeypatch.setattr(module, "AsyncScores", lambda *args: object())

    def run_join(*args, **kwargs):
        prefixes, suffixes, budget = args[4:7]
        assert budget == 1234
        assert list(prefixes[0]) == [1, 10, 11, 2]
        assert prefixes[0].token_parts[1] is documents["a"][0]
        assert [list(part) for part in suffixes[0]] == [[20, 3], [30, 3]]
        assert list(kwargs["anchor_partners"](("score", "score", 0))[0]) == [1, 0]
        return [{0: [0.9, 0.2]}], [], 8

    monkeypatch.setattr(module, "run_join", run_join)
    result = module.QuailScorer(state).score(spec, [(0, 1), (0, 0)], documents)
    np.testing.assert_allclose(result.scores, [0.9, 0.2])
    assert result.fresh_tokens == 8
    assert result.cached_tokens == 4

    batches = ScoreRows.batches
    monkeypatch.setattr(ScoreRows, "batches",
                        lambda self, size=None: batches(self, size=2))
    rounds = []
    session = _session(catalog, gpus=2)
    query = session.sql(_score_sql(True)[1])
    request = query._prepare_physical()
    physical = query.plan()
    assert (physical.workers, physical.settings["data_parallel_copies"]) == (2, 2)
    assert not any(isinstance(node, ScoreFilter) for node in physical.nodes)
    node = next(node for node in physical.nodes if isinstance(node, AiScore))
    assert node.spec.aliases == ("q", "d")
    assert node.spec.expected_inputs == 4
    assert node.spec.pair_fraction == 1.0
    assert node.spec.estimated_seconds > 0
    graph = physical.graph
    payload = quail_runtime_payload(request, graph)
    assert "pre_ids" not in payload and "filter_limit" not in payload

    def run(kind, subs):
        subs = pickle.loads(pickle.dumps(subs))
        if kind == "filters":
            assert len(subs) == 2
            assert all(sub["start_query"] and sub["node_id"] is None
                       for sub in subs)
            return [{"peak_gib": 0.0} for _ in subs]
        assert kind == "scores"
        assert all(("documents" in sub["inputs"]) == (not rounds) for sub in subs)
        rounds.append(kind)
        results = []
        for worker, sub in enumerate(subs):
            rows = sub["inputs"]["score_rows"]
            assert all(row[0] % 2 == worker for row in rows)
            model = RerankerModelExecution.__new__(RerankerModelExecution)
            model.documents = sub["inputs"].get("documents", execution.docs)
            for alias, documents in model.documents.items():
                assert [list(doc) for doc in documents] == [
                    list(doc) for doc in execution.docs[alias]
                ]
            model.reranker = _FakeReranker([a / 2 + b / 4 for a, b in rows])
            results.append(model.execute_rows(node, rows))
        return results

    execution = DistributedQuailExecution(
        payload, graph, 2, run, session.model, session.device, session.registry
    )
    execution.begin()
    assert execution.started
    result = execution.execute(node, {port.name: [1, 0] for port in node.inputs})
    assert result.outputs["scores"].to_pylist() == [
        {"q": 1, "d": 1, "score": 0.75},
        {"q": 1, "d": 0, "score": 0.5},
        {"q": 0, "d": 1, "score": 0.25},
        {"q": 0, "d": 0, "score": 0.0},
    ]
    assert result.metrics.evaluated_document_pairs == 4
    # each GPU scored its own query document's pairs in the same round
    assert len(rounds) == 1
    session.close()


def test_score_query_shape_errors_are_compile_errors(catalog):
    for sql, message in SHAPE_ERRORS:
        try:
            compile_sql(sql, catalog, _tokens)
        except CompileError as error:
            assert message in str(error), sql
        else:
            raise AssertionError(f"compiled: {sql}")


def test_score_refusals_name_oversized_documents_and_missing_scores(
        catalog, monkeypatch):
    from quail.cost import budgets

    for budget, unit in (("chunk_budget", "tokens"), ("arena_tokens", "pages")):
        with monkeypatch.context() as patch:
            patch.setattr(budgets, budget, lambda model, device, *rest: 4)
            session = _session(catalog)
            query = session.sql(
                "SELECT d.id, AI.SCORE(PROMPT('Refund? {0}', d.body)) AS s "
                "FROM documents d"
            )
            plan = query.plan()
            session.close()
        assert isinstance(plan, Refusal), budget
        assert plan.constraint == "suffix_over_chunk", budget
        assert plan.unit == unit, budget
        assert "a document in 'd'" in plan.reasons[0], budget

    with _session(catalog) as session:
        plan = session.sql(
            "SELECT d.id FROM documents d "
            "WHERE AI_FILTER(PROMPT('Refund? {0}', d.body))"
        ).plan()
    assert isinstance(plan, Refusal)
    assert plan.constraint == "reranker_only_scores"
    assert plan.reasons == ("a reranker model can only be used with AI.SCORE",)


def test_reranker_prompts_render_stored_documents_and_the_ai_if_frame(catalog):
    from quail.logical.prompts import (
        bind_join_prompt,
        bind_prompt,
        render_filter_prompt_ids,
        render_join_prompt_ids,
        true_false_token_ids,
    )
    from quail.reranker import (
        QWEN3_RERANKER_SYSTEM_TEXT,
        render_qwen3_reranker_input,
    )
    from quail.specs import MODELS

    assert "\n" not in QWEN3_RERANKER_SYSTEM_TEXT
    assert "based on the Query and the Instruct provided." \
        in QWEN3_RERANKER_SYSTEM_TEXT

    documents = ("refund please", "all good")
    for pair in (False, True):
        session = _session(catalog, tokenizer=list)
        query = session.sql(_score_sql(pair)[1])
        reranker = _FakeReranker([0.5] * (4 if pair else 2))
        _run_graph(session, query, query._prepare_physical(), reranker)
        queries = (
            ["Is this document relevant to refund?",
             "Is this document relevant to shipping?"] if pair else ["Refund?"]
        )
        assert [list(prompt) for prompt in reranker.prompts] == [
            list(render_qwen3_reranker_input(text, document))
            for text in queries for document in documents
        ], f"pair={pair}"
        session.close()

    for model in GENERATIVE_MODELS:
        for pair in (False, True):
            case = f"{model} pair={pair}"
            session = _session(catalog, model=model, tokenizer=list)
            template, sql = _score_sql(pair)
            query = session.sql(sql)
            physical = query.plan()
            assert physical.settings["score_normalization"] == \
                "true_false_softmax", case
            true_ids, false_ids = true_false_token_ids(list)
            assert physical.settings["true_ids"] == true_ids, case
            assert physical.settings["false_ids"] == false_ids, case
            request = query._prepare_physical()
            model_execution = RerankerModelExecution.__new__(
                RerankerModelExecution)
            model_execution.documents = {
                node.alias: request.inputs[node.input_id].documents
                for node in physical.nodes if node.type_name == "quail.scan"
            }
            model_execution.reranker = _FakeReranker([0.5] * (4 if pair else 2))
            GenericRunner().run(
                compute_subgraph(physical.graph),
                ExecutionContext(
                    runtimes=session.registry.runtimes,
                    model_execution=model_execution,
                    sources={alias: range(2)
                             for alias in model_execution.documents},
                ),
            )
            turn = MODELS[model].turn
            args = query.logical.root.columns[0].expression.prompt.args
            if pair:
                prompt = bind_join_prompt(template, args, list, turn)
                expected = [
                    render_join_prompt_ids(prompt, [list(text), list(body)], 0, list)
                    for text in ("refund", "shipping") for body in documents
                ]
            else:
                prompt = bind_prompt(template, args, list, turn)
                expected = [render_filter_prompt_ids(prompt, list(body), list)
                            for body in documents]
            assert [list(tokens) for tokens in model_execution.reranker.prompts] \
                == expected, case
            session.close()
