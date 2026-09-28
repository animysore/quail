"""Fixed join execution, filter retention, and answer reconstruction."""

import pyarrow as pa
from test_quail_backend import graph_state

import quail
from quail.backends.quail.graph import execute_single_graph, filter_result
from quail.bench import quailb
from quail.execution.execute import execute_query
from quail.execution.runner import NodeMetrics, NodeResult, SurvivorStream
from quail.execution.types import PhysicalResponse
from quail.physical import AiFilter, AiJoin, Barrier, Project, Scan, decode_graph
from quail.planner.plan import EngineConfig
from quail_b import prompts
from quail_b.queries import FILTER_SELECTIVITY_ESTIMATES

FEV9_ROWS = [{"c1.id": "c0", "e1.id": "e0", "c2.id": "c1", "e2.id": "e1"}]


def _session(gpus=1, backend="quail"):
    session = quail.Session(
        EngineConfig(gpus=gpus, model="qwen3-4b-fp8", backend=backend,
                     device="h100-sxm"),
        tokenizer=lambda text: list(text.encode()))
    register_fever(session)
    return session


def register_fever(session):
    for name, column, prefix, length in (
        ("claims", "claim", "c", 20), ("evidence", "text", "e", 300),
    ):
        table = pa.table({
            "id": [f"{prefix}{index}" for index in range(3)],
            column: [f"{index} " + "word " * length for index in range(3)],
        })
        if name == "claims":
            # c0 and c1 name page e0, c2 names e2 (an evidence id)
            table = table.append_column(
                "evidence_wiki_url", pa.array(["e0", "e0", "e2"]))
        session.register(name, quail.DocumentProvider.from_table(
            table, id_col="id"))


class FixedFeverAnswers:
    def __init__(self, state, capacity, empty):
        self.state = state
        self.capacity = capacity
        self.empty = empty

    def _answers(self, document_ids):
        answers = {local: [document < 2 and not self.empty]
                   for local, document in enumerate(document_ids)}
        live = [document for local, document in enumerate(document_ids)
                if all(answers[local])]
        return answers, live

    def execute(self, node, inputs):
        if isinstance(node, AiFilter):
            if node.pin_survivors:
                stream = SurvivorStream(node, inputs["document_ids"])
                return NodeResult(
                    {f"ids:{node.alias}": stream,
                     f"filter_answers:{node.alias}": {}},
                    finalize=stream.finalized_result)
            document_ids = list(inputs["document_ids"])
            answers, live = self._answers(document_ids)
            if inputs["retain_survivors"]:
                for document in live[:self.capacity]:
                    self.state["arena"].retain((node.alias, document), 16)
            return NodeResult({
                f"ids:{node.alias}": live,
                f"filter_answers:{node.alias}": {
                    document_ids[local]: row for local, row in answers.items()
                },
            })

        outputs = {}
        stream = inputs.get("anchor_stream")
        if stream is None:
            anchor_ids = inputs["anchor_ids"]
        else:
            answers, anchor_ids = self._answers(list(stream["document_ids"]))
            stream["stream"].complete(filter_result(
                stream["node"], answers, 0, stream["document_ids"]))
            keys = [(node.anchor, document) for document in anchor_ids]
            if inputs.get("anchor_batch") is not None:
                # like the real driver: per-batch functions run on each
                # batch before admission and may drop survivors
                keys = inputs["anchor_batch"](keys)
            anchor_ids = [key[1] for key in keys]
            for key in keys:
                inputs["anchor_keys"].append(key)
                inputs["prefixes"].append([])
        live = set(range(len(anchor_ids)))
        all_answers = []
        lists_for = inputs.get("anchor_partners")
        for stage_index, stage in enumerate(node.stages):
            aliases = (stage.anchor, *stage.partners)
            claim = next(alias for alias in aliases if alias.startswith("c"))
            evidence = next(alias for alias in aliases if alias.startswith("e"))
            passing = ({(1, 0), (2, 0), (0, 2)} if stage.written_pos == 1
                       else {(0, 0), (1, 1), (2, 0), (1, 2)})
            partners = inputs["partner_indices"][stage.written_pos]
            # a stage over pairs asks each anchor about its own members
            members = {} if lists_for and stage.pairs_from else None
            rows = {}
            for local in sorted(live):
                mine = (range(len(partners)) if members is None
                        else lists_for(anchor_ids[local])[stage_index])
                if members is not None:
                    members[local] = list(mine)
                rows[local] = []
                for member in mine:
                    assignment = dict(zip(
                        aliases, (anchor_ids[local], *partners[member])))
                    rows[local].append((assignment[claim], assignment[evidence])
                                       in passing)
            live = {local for local, row in rows.items() if any(row)}
            all_answers.append(rows)
            outputs[f"join_answers:{stage.written_pos}"] = {
                "rows": rows, "anchor_index": anchor_ids,
                "partner_index": partners, "anchor_partners": members,
                "anchor": stage.anchor,
                "partners": list(stage.partners), "semantics": stage.semantics,
                "selectivity": stage.selectivity, "written_pos": stage.written_pos,
            }
        outputs[f"ids:{node.anchor}"] = [anchor_ids[local] for local in sorted(live)]
        return NodeResult(outputs, NodeMetrics(extension={"answers": all_answers}))


def fever_executor(session, monkeypatch, gpus, check=None, capacity=1,
                   empty=False):
    """Return a physical executor that runs the FEVER fakes on one or two GPUs."""
    from quail.backends.quail import worker
    from quail.backends.quail.distributed import execute_distributed_graph
    from quail.execution.runner import ExecutionContext
    from quail.specs import H100_SXM, QWEN3_4B_FP8

    def child(docs):
        state = graph_state(None, docs)
        model = FixedFeverAnswers(state, capacity=1, empty=False)
        state.update(
            model_execution=model, registry=session.registry,
            boot={"boot_s": 0, "kind": "warm"}, seen=set(),
            runtime_context=ExecutionContext(
                runtimes=session.registry.runtimes, model_execution=model))
        return state

    def execute(request):
        graph = decode_graph(request.plan["graph"], session.registry.codecs)
        docs = {node.alias: request.inputs[node.input_id].documents
                for node in graph.nodes if isinstance(node, Scan)}
        extra = check(request, graph) if check else {}
        if gpus == 1:
            state = graph_state(None, docs)
            state["model_execution"] = FixedFeverAnswers(state, capacity, empty)
            state.update(extra, functions=session.registry.functions)
            report = execute_single_graph(state, request.plan["settings"], graph)
            assert not state["arena"].accounting.owned
            return PhysicalResponse(report.pop("_outputs"), report)
        children = [child(docs), child(docs)]

        def round_fn(kind, subs):
            function = (worker._child_filters if kind == "filters"
                        else worker._child_joins)
            return [function(state, sub) for state, sub in zip(children, subs)]

        monkeypatch.setattr(worker, "_child_boot", lambda state, sub: None)
        report = execute_distributed_graph(
            {**request.plan["settings"], "model": "qwen3-4b-fp8",
             "docs": docs, **extra, "physical_plan": request.plan},
            graph, 2, round_fn, QWEN3_4B_FP8, H100_SXM,
            session.registry.runtimes, session.registry,
        )
        assert all(not child["arena"].accounting.owned for child in children)
        return PhysicalResponse(report.pop("_outputs"), report)

    return execute


FEV9_RUNS = [
    (1, 10, False, 0.001), (1, 1, False, 1.0), (1, 0, False, 0.5),
    (1, 0, True, 1.0), (2, 1, False, None),
]

REQUEST_BACKENDS = ["stock_vllm", "pipelined_vllm", "pipelined_sglang"]


def test_fev9_executes_bound_and_edited_join_nodes_without_the_optimizer(
        monkeypatch):
    def unexpected_search(*args, **kwargs):
        raise AssertionError("execution called the join optimizer")

    for gpus, capacity, empty, estimate in FEV9_RUNS:
        case = f"gpus={gpus} capacity={capacity} empty={empty}"
        with monkeypatch.context() as patch:
            if estimate is not None:
                patch.setitem(FILTER_SELECTIVITY_ESTIMATES, prompts.F11, estimate)
                patch.setitem(FILTER_SELECTIVITY_ESTIMATES, prompts.F13, estimate)

            def check(request, graph):
                assert sum(len(group.stages)
                           for group in graph.nodes_by_type(AiJoin.type_name)) == 3
                patch.setattr("quail.planner.joins.search_joins",
                              unexpected_search)
                return {}

            with _session(gpus=gpus) as session:
                query = quailb.queries(session)["FEV-9"][1]()
                execute = fever_executor(
                    session, patch, gpus, check, capacity, empty)
                result = execute_query(query, physical_executor=execute).collect()
        assert result.to_pylist() == ([] if empty else FEV9_ROWS), case

    with _session() as session:
        query = quailb.queries(session)["FEV-9"][1]()
        plan = query.plan()
        chain = next(node for node in plan.nodes
                     if isinstance(node, AiFilter) and node.pin_survivors)
        join = next(node for node in plan.nodes
                    if isinstance(node, AiJoin) and node.anchor == chain.alias)
        edited = plan.insert(
            Barrier(node_id=f"barrier:{chain.alias}", next_anchor=chain.alias,
                    aliases=(chain.alias,)),
            between=(chain.node_id, join.node_id))
        assert not edited.graph.node(chain.node_id).pin_survivors
        assert edited != plan

        seen = []

        def check(request, graph):
            seen.append([node.node_id for node in graph.nodes])
            return {}

        execute = fever_executor(session, monkeypatch, 1, check, capacity=10)
        rows = execute_query(query, physical_executor=execute).collect()
        edited_result = execute_query(
            quailb.queries(session)["FEV-9"][1](), physical_executor=execute,
            plan=edited)
        assert f"barrier:{chain.alias}" in seen[1]
        assert f"barrier:{chain.alias}" not in seen[0]
        assert edited_result.collect().to_pylist() == rows.to_pylist() == FEV9_ROWS


def test_request_backends_plan_fev9_and_a_single_join():
    for backend in REQUEST_BACKENDS:
        with _session(backend=backend) as session:
            plan = quailb.queries(session)["FEV-9"][1]().plan()
            execution = next(
                node for node in plan.nodes if hasattr(node, "joins"))
            assert execution.filters[-1].alias == execution.joins[0].anchor, \
                backend
            plan = (session.docs("claims").alias("c")
                    .ai_join(session.docs("evidence").alias("e"),
                             quail.prompt("m {0} {1}", quail.col("c.claim"),
                                          quail.col("e.text")))
                    .select("c.id", "e.id")).plan()
        sink = next(node for node in plan.nodes if isinstance(node, Project))
        assert sink.inputs[0].source.port == "join_answers:0", backend


def test_fev10_asks_the_model_about_same_page_pairs_only(monkeypatch):
    def check(request, graph):
        columns = request.column_tables()
        assert columns["c"].column("evidence_wiki_url").to_pylist() == [
            "e0", "e0", "e2"]
        return {"columns": columns}

    for gpus in (1, 2):
        with _session(gpus=gpus) as session:
            query = quailb.queries(session)["FEV-10"][1]()
            plan = query.plan()
            (stage,) = plan.graph.nodes_by_type(AiJoin.type_name)[0].stages
            hash_join = plan.graph.node("hash_join:c-e")
            assert hash_join.on == (("evidence_wiki_url", "id"),)
            assert hash_join.pair_fraction == 3 / 9
            assert stage.pairs_from == "hash_join:c-e"
            result = execute_query(
                query, physical_executor=fever_executor(session, monkeypatch, gpus,
                                                        check))
            # of the same-page pairs (c0, e0) and (c1, e0), only c0 is
            # supported; a cross join would also have returned (c1, e1)
            answers = result.answer_tables["joins"][0]
            assert sorted(zip(answers.column("c").to_pylist(),
                              answers.column("e").to_pylist())) == [(0, 0), (1, 0)]
            assert result.collect().to_pylist() == [{"c.id": "c0", "e.id": "e0"}]
