"""Quail runtimes for the typed model graph."""

from __future__ import annotations

import itertools
import time

from quail.backends.quail.retention import apply_retention, retain_after_join
from quail.execution.pairs import (
    allowed_members,
    columns_key,
    members_by_partner,
    pair_partner,
    partner_map,
)
from quail.execution.reranker import ScoreFilterRuntime
from quail.execution.runner import (
    ExecutionContext,
    GenericRunner,
    ModelNodeRuntime,
    NodeMetrics,
    NodeResult,
    StreamedPairs,
    SurvivorStream,
    compute_subgraph,
    scalar_node_metrics,
)
from quail.execution.tokens import DocumentPrefixes, chain_tokens
from quail.execution.types import export_physical_outputs
from quail.physical import (
    AiFilter,
    AiJoin,
    AiScore,
    PhysicalGraph,
    Scan,
    ScoreFilter,
)
from quail.progress import answer_sink


def quail_runtimes() -> dict:
    """Return runtimes for the Quail backend's physical nodes."""
    model_runtime = ModelNodeRuntime()
    return {
        AiFilter.runtime_key: model_runtime,
        AiJoin.runtime_key: model_runtime,
        AiScore.runtime_key: model_runtime,
        ScoreFilter.runtime_key: ScoreFilterRuntime(),
    }


def _join_round_kv(anchor_keys, arena) -> dict:
    hits = sum(arena.is_resident(key) for key in anchor_keys)
    return {"hits": hits, "misses": len(anchor_keys) - hits}


def _tuple_suffix(join, docs, member):

    parts = []
    for alias, document in zip(join["partners"], member):
        parts.extend((join["labels"][alias], docs[alias][document]))
    parts.append(join["tail"])
    return chain_tokens(*parts)


def runs_over_pairs(join: dict) -> bool:
    """Whether a stage streams each anchor's own pairs."""
    return bool(join.get("pairs_from"))


def partner_maps(group, pair_tables, streamed=()) -> dict:
    """Written position -> anchor row -> partner rows, per pair stage.

    streamed names the written positions whose pairs arrive per batch
    and need no table here.
    """
    maps = {}
    for join in group:
        if not runs_over_pairs(join) or join["written_pos"] in streamed:
            continue
        table = pair_tables.get(join["written_pos"])
        if table is None:
            raise ValueError(
                f"join {join['written_pos']} runs over pairs but no pair "
                f"table reached it")
        maps[join["written_pos"]] = partner_map(
            table, join["anchor"],
            pair_partner(join["anchor"], join["partners"]))
    return maps


def partner_list_builder(group, tuples_by_stage, maps):
    """Per-anchor partner member lists for the stages that run over pairs.

    Args:
        group: The stages' runtime specs.
        tuples_by_stage: Per stage, its partner member tuples of global
            document ids, in partner order.
        maps: written position -> anchor row -> partner rows.

    Returns:
        callable(anchor global id) -> per stage, the sorted member
        indices the anchor streams, or None for every member; or None
        when no stage runs over pairs.
    """
    stage_maps = []
    for join, tuples in zip(group, tuples_by_stage):
        if not runs_over_pairs(join):
            stage_maps.append(None)
            continue
        position = join["partners"].index(
            pair_partner(join["anchor"], join["partners"]))
        stage_maps.append((maps[join["written_pos"]],
                           members_by_partner(tuples, position)))
    if all(entry is None for entry in stage_maps):
        return None

    def lists_for(anchor):
        out = []
        for entry in stage_maps:
            if entry is None:
                out.append(None)
                continue
            out.append(allowed_members(*entry, anchor))
        return out

    return lists_for


def stage_partner_lists(group, lists_for, anchor_ids) -> list:
    """Per stage, anchor local index -> member indices, or None."""
    out = []
    for index, join in enumerate(group):
        if lists_for is None or not runs_over_pairs(join):
            out.append(None)
            continue
        out.append({local: lists_for(anchor)[index]
                    for local, anchor in enumerate(anchor_ids)})
    return out


def filter_result(node, answers, tokens, document_ids,
                  gpu_s: float = 0.0, chunks: int = 0,
                  borrowed_tokens: int = 0) -> NodeResult:
    """Build one filter chain's node result from its local answers.

    borrowed_tokens are document tokens read from another document's
    KV pages instead of computed (prefix sharing).
    """
    global_answers = {
        document_ids[int(local)]: row
        for local, row in answers.items()
    }
    survivors = sorted(
        document
        for document, row in global_answers.items()
        if len(row) == len(node.question_token_ids) and all(row)
    )
    return NodeResult(
        outputs={
            f"ids:{node.alias}": survivors,
            f"filter_answers:{node.alias}": global_answers,
        },
        metrics=NodeMetrics(
            input_rows=len(document_ids),
            output_rows=len(survivors),
            evaluated_documents=len(global_answers),
            fresh_tokens=tokens,
            gpu_s=gpu_s,
            chunks=chunks,
            extension=({"borrowed_prefix_tokens": borrowed_tokens}
                       if borrowed_tokens else {}),
        ),
    )


def filter_document_sink(node, document_ids):
    """The chain's document_done callback, or None when nobody listens.

    Reports each chunk's finished documents to the answer sink as
    ``{"kind": "filter", "node", "alias", "stages", "documents"}`` where
    ``documents`` lists ``[row index, last stage asked, passed]`` per
    document. A document that passed its last stage survived the filter;
    every other listed document failed at the stage given.
    """
    sink = answer_sink()
    if sink is None:
        return None
    stages = len(node.question_token_ids)

    def document_done(finished):
        sink({"kind": "filter", "node": node.node_id, "alias": node.alias,
              "stages": stages,
              "documents": [[int(document_ids[position]), stage, passed]
                            for position, stage, passed in finished]})

    return document_done


def filter_inputs(state, node, document_ids) -> dict:
    """Scheduler inputs for one filter chain over the given documents."""
    return {
        "documents": DocumentPrefixes(
            state["pre"], state["docs"][node.alias], document_ids
        ),
        "document_ids": document_ids,
        "limit": state["filter_limit"],
        "retain_survivors": node.keep_kv,
        "document_done": filter_document_sink(node, document_ids),
    }


def prepare_model_inputs(node, inputs, context: ExecutionContext):
    """Prepare Quail scheduler inputs from typed port values."""
    if isinstance(node, AiScore):
        return {"score_inputs": inputs, "documents": context.state["docs"]}
    if not isinstance(node, (AiFilter, AiJoin)):
        return inputs
    return {
        **_model_inputs(node, inputs, context),
        "gpu_timing": context.state.get("gpu_timing", False),
    }


def _model_inputs(node, inputs, context: ExecutionContext) -> dict:
    state = context.state
    if isinstance(node, AiFilter):
        return filter_inputs(state, node, next(iter(inputs.values())))

    stream = None
    by_alias = {}
    port_pairs = {}        # written position -> pair table from a port
    streamed_pairs = {}    # written position -> StreamedPairs
    for input_port in node.inputs:
        port = input_port.source.port
        value = inputs[input_port.name]
        if port.startswith("pairs:"):
            position = int(port.split(":", 1)[1])
            if isinstance(value, StreamedPairs):
                streamed_pairs[position] = value
            else:
                port_pairs[position] = value
            continue
        if not port.startswith("ids:"):
            continue
        alias = port.split(":", 1)[1]
        if isinstance(value, SurvivorStream):
            if alias != node.anchor or not isinstance(value.node, AiFilter):
                raise TypeError(
                    "only the anchor's filter chain streams into a join")
            stream = value
        else:
            by_alias[alias] = list(value)
    if not state["joins_started"]:
        state["joins_started"] = True
        arena = state["arena"]
        retained = arena.retained_keys()
        state["kv_stats"].update(
            retained_after_filters=len(retained),
            retained_pages_after_filters=arena.retained_pages,
            retained_prefix_tokens_after_filters=arena.retained_prefix_tokens,
            retained_by_alias_after_filters={
                alias: sum(key[0] == alias for key in retained)
                for alias in state["docs"]
            },
        )
    config = state["retention"]
    apply_retention(state["arena"], config,
                    config.get("before", {}).get(node.node_id, {}), by_alias)
    group = [stage.runtime_spec() for stage in node.stages]
    stage_suffixes = []
    tuple_indices = {}
    for stage, join in zip(node.stages, group):
        tuples = [list(member) for member in itertools.product(
            *[by_alias[alias] for alias in join["partners"]]
        )]
        tuple_indices[stage.written_pos] = tuples
        stage_suffixes.append([
            _tuple_suffix(join, state["docs"], member)
            for member in tuples
        ])
    # pairs come from the request, a Foreign table, or a per-batch Foreign
    maps = partner_maps(group, port_pairs, streamed=set(streamed_pairs))
    for position, pairs in streamed_pairs.items():
        maps[position] = pairs.rows
    lists_for = partner_list_builder(
        group, [tuple_indices[stage.written_pos] for stage in node.stages],
        maps)
    anchor_batch = None
    if stream is not None and (stream.transforms or streamed_pairs):
        def anchor_batch(keys, stream=stream):
            ids = [key[1] for key in keys]
            for transform in stream.transforms:
                ids = list(transform(ids))
            for pairs in streamed_pairs.values():
                pairs.rows.update(pairs.batch(ids))
            kept = set(ids)
            return [key for key in keys if key[1] in kept]
    if stream is None:
        anchor_ids = by_alias[node.anchor]
        prefixes = [
            chain_tokens(state["pre"], state["docs"][node.anchor][document])
            for document in anchor_ids
        ]
        anchor_keys = [(node.anchor, document) for document in anchor_ids]
        round_kv = _join_round_kv(anchor_keys, state["arena"])
        state["kv_stats"]["join_anchor_hits"] += round_kv["hits"]
        state["kv_stats"]["join_anchor_misses"] += round_kv["misses"]
        anchor_stream = None
    else:
        # filled by the driver as the chain hands anchors over
        anchor_ids = None
        prefixes = []
        anchor_keys = []
        round_kv = None
        anchor_stream = {
            "node": stream.node,
            **filter_inputs(state, stream.node, stream.document_ids),
            "stream": stream,
        }

    last = group[-1]
    last_tuples = tuple_indices[last["written_pos"]]

    def finished_answers(document, row) -> dict:
        """The anchor's last-stage matches: partner tuples that answered true.

        Every pair the anchor was asked about and is not listed answered
        false; ``asked`` says how many pairs that covers.
        """
        members = None
        if lists_for is not None and runs_over_pairs(last):
            members = lists_for(document)[len(group) - 1]
        matches = []
        for position, answer in enumerate(row):
            if answer:
                member = position if members is None else int(members[position])
                matches.append(list(last_tuples[member]))
        return {"kind": "join", "node": node.node_id, "anchor": node.anchor,
                "partners": list(last["partners"]),
                "semantics": last["semantics"], "document": int(document),
                "asked": len(row), "matches": matches}

    def anchor_done(local_index, row):
        key = anchor_keys[local_index]
        matched = any(row)
        alive = not matched if group[-1]["semantics"] == "anti" \
            else matched
        if alive and node.keep_anchor_kv:
            retain_after_join(
                state["arena"], key, len(prefixes[local_index]), config,
                config.get("after", {}).get(node.node_id, {}))
        else:
            state["arena"].free_key(key)
        sink = answer_sink()
        # scores are floats, not yes/no answers; they arrive with the result
        if sink is not None and getattr(row, "dtype", None) is None:
            sink(finished_answers(key[1], row))

    state["prepared_join"] = {
        "node": node,
        "group": group,
        "anchor_keys": anchor_keys,
        "prefixes": prefixes,
        "tuple_indices": tuple_indices,
        "streamed": stream is not None,
    }
    return {
        "prefixes": prefixes,
        "stage_suffixes": stage_suffixes,
        "stage_frames": [join.get("frame") or [] for join in group],
        "anchor_keys": anchor_keys,
        "anchor_done": anchor_done,
        "anchor_stream": anchor_stream,
        "kv_round": round_kv,
        "anchor_ids": anchor_ids,
        "partner_indices": tuple_indices,
        "anchor_partners": lists_for,
        "anchor_batch": anchor_batch,
        "group": group,
    }


def record_model_result(node, result: NodeResult,
                        context: ExecutionContext) -> None:
    """Release or retain each join anchor's KV once its group has run."""
    state = context.state
    if isinstance(node, AiJoin):
        prepared = state["prepared_join"]
        keys = prepared["anchor_keys"]
        if prepared["streamed"]:
            # every streamed anchor read its KV from the chain: a hit
            state["kv_stats"]["join_anchor_hits"] += len(keys)
        live = set(result.outputs[f"ids:{node.anchor}"])
        for key, prefix in zip(keys, prepared["prefixes"]):
            if state["arena"].is_resident(key):
                if node.keep_anchor_kv and key[1] in live:
                    config = state["retention"]
                    retain_after_join(state["arena"], key, len(prefix), config,
                                      config.get("after", {}).get(node.node_id, {}))
                else:
                    state["arena"].free_key(key)


def execute_single_graph(state, payload, graph: PhysicalGraph) -> dict:
    """Execute one Quail model graph on one GPU executor."""
    torch = state["torch"]
    arena = state["arena"]
    for key in arena.resident_keys():
        arena.free_key(key)
    arena.reset_stats()
    retention = payload.get("retention", {})
    apply_retention(arena, retention, retention.get("initial", {}))
    docs = state["docs"]
    sources = {
        alias: range(len(documents))
        for alias, documents in docs.items()
    }
    for alias, table in state.get("columns", {}).items():
        sources[columns_key(alias)] = table
    runtime_state = {
        **state,
        "pre": payload.get("pre_ids") or [],
        "retention": retention,
        "filter_limit": (
            None if any(isinstance(node, AiJoin)
                        for node in graph.nodes)
            else payload.get("filter_limit")
        ),
        "joins_started": False,
        "kv_stats": {
            "retained_after_filters": 0,
            "retained_pages_after_filters": 0,
            "retained_prefix_tokens_after_filters": 0,
            "join_anchor_hits": 0,
            "join_anchor_misses": 0,
        },
    }
    context = ExecutionContext(
        runtimes=state["runtimes"],
        model_execution=state["model_execution"],
        sources=sources,
        model_inputs=prepare_model_inputs,
        model_result=record_model_result,
        state=runtime_state,
        functions=state.get("functions", {}),
    )
    started = time.perf_counter()
    with torch.inference_mode():
        result = GenericRunner().run(compute_subgraph(graph), context)
    torch.cuda.synchronize()
    wall = time.perf_counter() - started

    filters, joins = model_answers(graph, result)

    for key in arena.resident_keys():
        arena.free_key(key)
    kv_manager = {
        **runtime_state["kv_stats"],
        "evicted_keys": arena.evicted_keys,
        "evicted_pages": arena.evicted_pages,
        "evicted_prefix_tokens": arena.evicted_prefix_tokens,
    }

    report = {
        "filters": filters,
        "joins": joins,
        "_outputs": export_physical_outputs(compute_subgraph(graph), result),
        "wall_s": round(wall, 2),
        "fresh_tokens": result.metrics.fresh_tokens,
        "cached_tokens": result.metrics.cached_tokens,
        "evaluated_documents": result.metrics.evaluated_documents,
        "evaluated_document_pairs": result.metrics.evaluated_document_pairs,
        "usd_per_query": (
            None if state["device"].usd_per_hour is None
            else wall / 3600 * state["device"].usd_per_hour
        ),
        "backend_metrics": {"scores": [
            dict(value.metrics.extension)
            for node_id, value in result.nodes.items()
            if graph.node(node_id).type_name == AiScore.type_name
        ]},
        "node_metrics": scalar_node_metrics(result.nodes),
        "executed_join_plan": executed_join_plan(graph),
        "kv_manager": kv_manager,
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    report.update(throughput(graph, result.metrics, wall))
    if state.get("gpu_timing"):
        # seconds a forward chunk was running on the GPU, summed over
        # the model nodes; wall_s minus this is time the GPU sat idle
        report["gpu_s"] = round(result.metrics.gpu_s, 3)
        report["chunks"] = result.metrics.chunks
    return report


def throughput(graph, metrics, seconds: float) -> dict:
    """Return the run's throughput under one key.

    A run that evaluated document pairs reports evaluated pairs per
    second. Any other run reports input document rows, summed over
    every scan and counted before filters, per second.
    """
    seconds = max(seconds, 1e-9)
    if metrics.evaluated_document_pairs:
        return {
            "document_pairs_per_second":
                metrics.evaluated_document_pairs / seconds,
        }
    documents = sum(
        node.n_docs for node in graph.nodes if isinstance(node, Scan)
    )
    return {"documents_per_second": documents / seconds}


def model_answers(graph, result) -> tuple[dict, list]:
    """Collect filter and join answers from executed model nodes."""
    filters, joins = {}, []
    for node in graph.topological_nodes():
        if isinstance(node, AiFilter):
            filters[node.alias] = result.nodes[node.node_id].outputs[
                f"filter_answers:{node.alias}"
            ]
        elif isinstance(node, AiJoin):
            joins.extend(result.nodes[node.node_id].outputs[
                f"join_answers:{stage.written_pos}"
            ] for stage in node.stages)
    return filters, joins


def executed_join_plan(graph) -> list[dict]:
    """Describe the join nodes executed from the saved graph."""
    from quail.physical import Barrier

    return [
        {"type": node.type_name, "id": node.node_id,
         **node.explain_fields()}
        for node in graph.topological_nodes()
        if isinstance(node, (AiJoin, Barrier))
    ]
