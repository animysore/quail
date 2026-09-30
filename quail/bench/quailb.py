"""Run QUAIL-B queries with Quail: the adapter `quail_b.run` calls.

`run_query(session, spec, tables)` builds one query's Substrait plan
on the session, runs it, and returns the answers keyed by the plan's
operator ids, as QUAIL-B scores them.

Runs on any machine with a supported GPU and saves to a local directory;
no Modal or volume is involved:

    uv run python -m quail.bench.quailb --sf 0.1 --only IMDB-4 \
      --model qwen3-4b-fp8 --device h100-sxm \
      --output-dir results/quailb/imdb-4
"""

import argparse
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

import quail
import quail_b as benchmark
from quail.bench import substrait
from quail.bench.results import write_json
from quail.bench.substrait import QueryPlan, UnsupportedQueryError, read_plan
from quail.planner.plan import Refusal
from quail.specs import H100_USD_PER_HOUR, MODELS
from quail_b.queries import (
    FILTER_SELECTIVITY_ESTIMATES,
    JOIN_SELECTIVITY_ESTIMATES,
    SELECTIVITY_ESTIMATE_COLLECTION,
    SELECTIVITY_ESTIMATE_CORPUS,
    SELECTIVITY_ESTIMATE_SCALE_FACTOR,
    QuerySpec,
)
from quail_b.queries import queries as query_specs
from quail_b.scoring import RunOutput, reference_answer

# the fixed planner inputs, by prompt; a predicate without an
# estimate here gets the planner's default selectivity
SELECTIVITY = {**FILTER_SELECTIVITY_ESTIMATES, **JOIN_SELECTIVITY_ESTIMATES}
TEXT_VLLM_BACKENDS = frozenset({
    "dumb_vllm",
    "pipelined_vllm",
    "stock_vllm",
})


def register_tables(session, data_dir):
    """Register every Parquet file of a directory as a document table."""
    for path in sorted(Path(data_dir).glob("*.parquet")):
        session.register(path.stem, quail.DocumentProvider.from_parquet(
            str(path), id_col="id"))


def _build(session, plan: QueryPlan):
    return substrait.build_query(session, plan, SELECTIVITY, order="by_cost")


def build_query(session, spec: QuerySpec):
    """Build the Quail query of one benchmark query on a session."""
    return _build(session, read_plan(spec.plan))


def queries(session) -> dict:
    """Id -> (description, callable() -> Query), for every registered table.

    A query is listed when every table it reads is registered. Fresh
    Query objects per call so each pass re-plans.
    """
    listed = {}
    for spec in query_specs(include_privacy=True).values():
        try:
            plan = read_plan(spec.plan)
        except UnsupportedQueryError:
            continue
        if all(relation.table in session.catalog for relation in plan.relations):
            listed[spec.id] = (
                spec.description, lambda plan=plan: _build(session, plan))
    return listed


def canonical_templates(ground_truth) -> dict[str, str]:
    """Map each predicate's template as Quail binds it to the written one."""
    canonical = {}
    for labels in ground_truth.predicates.values():
        predicate = labels.predicate
        if predicate["kind"] not in {"filter", "join"}:
            continue
        left = quail.ColumnRef(
            "left", predicate["left_table"], predicate["left_column"])
        if predicate["kind"] == "filter":
            bound = quail.bind_prompt(predicate["template"], (left,))
        else:
            right = quail.ColumnRef(
                "right", predicate["right_table"], predicate["right_column"])
            bound = quail.bind_join_prompt(
                predicate["template"], (left, right))
        canonical[bound.template] = predicate["template"]
    return canonical


def _ids(table: pa.Table) -> list[str]:
    return [str(row_id) for row_id in table.column("id").to_pylist()]


def answer_oracle(ground_truth, tables):
    """Return the `answer(prompt, assignment)` callable Quail's estimate takes."""
    templates = canonical_templates(ground_truth)
    ids_by_table = {name: _ids(table) for name, table in tables.items()}

    def answer(prompt, assignment):
        ids = tuple(
            ids_by_table[arg.provider][assignment[arg.alias]]
            for arg in prompt.args
        )
        return reference_answer(ground_truth, templates[prompt.template], ids)

    return answer


def run_output(result, plan: QueryPlan, tables) -> RunOutput:
    """Translate a Quail result's row indices into benchmark ids by operator."""
    ids = {relation.alias: pa.array(_ids(tables[relation.table]), pa.string())
           for relation in plan.relations}

    def id_column(alias, indices):
        return pc.take(ids[alias], indices)

    filter_answers = {}
    for (alias, position), table in result.answer_tables["filters"].items():
        filter_answers[plan.filter_id(alias, position)] = pa.table({
            alias: id_column(alias, table.column(alias)),
            "answer": table.column("answer"),
        })
    join_answers = {}
    for position, table in result.answer_tables["joins"].items():
        join = plan.joins[position]
        join_answers[join.id] = pa.table({
            **{alias: id_column(alias, table.column(alias))
               for alias in join.aliases},
            "answer": table.column("answer"),
        })
    aliases = []
    for name in plan.select:
        alias, column = name.split(".", 1)
        if column != "id" and column not in {
                operator.output for operator in plan.extracts}:
            raise NotImplementedError(
                "QUAIL-B output needs id or extraction columns")
        aliases.append(alias if column == "id" else column)
    started = time.perf_counter()
    rows = result.collect()
    collection_s = time.perf_counter() - started
    rows = rows.rename_columns(aliases)
    return RunOutput(
        filter_answers, join_answers, rows,
        result.report["wall_s"], dict(result.report, collection_s=collection_s),
        extract_answers={
            op.id: rows.select([op.alias, op.output]) for op in plan.extracts
        } if plan.extracts else None)


def join_anchors(result) -> dict:
    """Return written position -> the anchor alias the engine chose."""
    return {
        position: table.schema.metadata[b"quail.anchor"].decode("utf-8")
        for position, table in result.answer_tables["joins"].items()
    }


def prompt_pieces(query, plan: QueryPlan, anchors) -> dict:
    """Return the prompt token ids around each document, for QUAIL-B.

    QUAIL-B sizes the prefix trie of the run's requests from these
    pieces and the saved answer tables, after the run; nothing is
    tracked while the query runs.

    Args:
        query: The built query, with bound prompts.
        plan: The query's plan, for the operator ids.
        anchors: Written join position -> the anchor alias.
    """
    operators = query.logical.operators()
    filters, joins = operators.filters, operators.joins
    preamble = next((list(prompt.preamble_token_ids)
                     for prompt in operators.prompts
                     if prompt.preamble_token_ids), [])
    pieces = {"tokenizer": query.session.model.hf_name, "preamble": preamble,
              "filters": [], "joins": []}
    for alias, predicates in filters.items():
        for position, predicate in enumerate(predicates):
            pieces["filters"].append({
                "id": plan.filter_id(alias, position),
                "tail": list(predicate.prompt.tail_token_ids)})
    for position, join in enumerate(joins):
        anchor = anchors[position]
        parts = {alias: (list(label), list(frame))
                 for alias, label, frame in join.prompt.label_token_ids}
        (partner,) = [alias for alias in parts if alias != anchor]
        pieces["joins"].append({
            "id": plan.join_id(position), "anchor": anchor,
            "frame": parts[anchor][1], "label": parts[partner][0],
            "tail": list(join.prompt.tail_token_ids)})
    return pieces


def _submission_to_answer_s(
    backend: str,
    report: dict,
    *,
    frontend_s: float,
    answer_prepare_s: float,
) -> float:
    runtime_s = (
        report["model_wall_s"]
        + report["finish_s"]
        + answer_prepare_s
    )
    if backend == "quail":
        runtime_s += (
            frontend_s
            + report["input_ready_s"]
            + report["physical_prepare_s"]
        )
    return runtime_s


def run_query(session, spec: QuerySpec, tables) -> RunOutput:
    """Execute one query and return benchmark ids, answers, and measurements."""
    submitted = time.perf_counter()
    for name, table in tables.items():
        if name not in session.catalog:
            session.register(
                name, quail.DocumentProvider.from_table(table, id_col="id"))
    plan = read_plan(spec.plan)
    query = _build(session, plan)
    frontend_s = time.perf_counter() - submitted
    result = query.run()
    answer_started = time.perf_counter()
    output = run_output(result, plan, tables)
    answer_prepare_s = time.perf_counter() - answer_started
    if plan.extracts:
        nodes = [
            metrics for name, metrics in result.report["node_metrics"].items()
            if name == "request-model" or name.startswith(("extract:", "ai_filter:"))
        ]
        if not nodes:
            raise ValueError("extraction benchmark needs model node timings")
        output.runtime_s = sum(node["wall_s"] for node in nodes)
        backend = result.report["backend_metrics"]
        output.measurements.update(
            wall_s=output.runtime_s,
            input_tokens=backend["prompt_tokens"],
            output_tokens=backend["output_tokens"],
            timing_boundary="filter and extraction model nodes; startup excluded",
        )
        return output
    runtime_s = _submission_to_answer_s(
        session.config.backend,
        result.report,
        frontend_s=frontend_s,
        answer_prepare_s=answer_prepare_s,
    )
    output.runtime_s = runtime_s
    output.measurements.update(
        wall_s=runtime_s,
        submission_to_answer_s=runtime_s,
        finish_s=result.report["finish_s"],
        answer_prepare_s=answer_prepare_s,
    )
    if session.config.backend == "quail":
        output.measurements["frontend_s"] = frontend_s
    if session.config.backend in TEXT_VLLM_BACKENDS:
        output.measurements["input_tokens"] = (
            result.report["fresh_tokens"] + result.report["cached_tokens"]
        )
    output.prompt_pieces = prompt_pieces(query, plan, join_anchors(result))
    return output


def refused_queries(session, query_ids, data_dir) -> dict[str, str]:
    """Return id -> reason for the queries the session's backend refuses to plan.

    A request backend refuses a query with an equality join or a user
    function. Planning happens on the CPU, before any engine boots.
    """
    register_tables(session, data_dir)
    refused = {}
    for query_id in query_ids:
        plan = build_query(session, benchmark.get_query(query_id)).plan()
        if isinstance(plan, Refusal):
            refused[query_id] = " ".join(plan.reasons)
            print(f"[quail-b] {query_id}: skipped on {session.config.backend}: "
                  f"{refused[query_id]}", flush=True)
    return refused


def unsupported_queries(query_ids) -> dict[str, str]:
    """Return unsupported benchmark operators before input or model loading."""
    skipped = {}
    for query_id in query_ids:
        try:
            read_plan(benchmark.get_query(query_id).plan)
        except UnsupportedQueryError as error:
            skipped[query_id] = str(error)
    return skipped


def run_suite(only=None, *, sf=0.1, config, data_dir=None,
              ground_truth_collection=None, output_dir,
              h100_usd_per_hour=H100_USD_PER_HOUR, root=None):
    """Run Quail queries through QUAIL-B and save the benchmark report.

    Queries the backend refuses to plan are left out of the run and
    listed under `skipped_queries` in the returned record.
    """
    only = list(query_specs()) if only is None else (
        [only] if isinstance(only, str) else list(only))
    skipped = unsupported_queries(only)
    only = [query_id for query_id in only if query_id not in skipped]
    if only and data_dir is not None:
        with quail.Session(config) as preflight_session:
            skipped.update(refused_queries(preflight_session, only, data_dir))
        only = [query_id for query_id in only if query_id not in skipped]
    if not only:
        raise ValueError(f"no supported benchmark queries remain: {skipped}")
    with quail.Session(config) as session:
        record = benchmark.run(
            partial(run_query, session), queries=only, scale_factor=sf,
            output_dir=output_dir, data_dir=data_dir,
            collection_id=ground_truth_collection, root=root,
            gpu_count=config.gpus, gpu_hourly_rate_usd=h100_usd_per_hour,
            metadata={
                "engine": config.backend, "model": config.model,
                "prompt_format": MODELS[config.model].prompt_format,
                "configuration": asdict(config),
                "warmup": (
                    "tokenizer, engine, and kernel startup excluded"
                ),
                "timing_boundary": (
                    "filter and extraction model nodes; startup excluded"
                    if any(read_plan(benchmark.get_query(name).plan).extracts
                           for name in only)
                    else "raw document tables and query submission to answer"
                    if config.backend == "quail"
                    else "prompt text submission to answer"
                ),
                "cache_reuse": "one session per backend and query family",
                "planning": {
                    "collection_id": SELECTIVITY_ESTIMATE_COLLECTION,
                    "corpus_id": SELECTIVITY_ESTIMATE_CORPUS,
                    "scale_factor": SELECTIVITY_ESTIMATE_SCALE_FACTOR,
                },
            })
        record["skipped_queries"] = skipped
        write_json(Path(output_dir) / "run.json", record)
        benchmark.report(output_dir, rescore=False)
    return record


def _close_process_engines():
    """Release cached engines when the standalone CLI finishes."""
    from quail.backends.quail.worker import release_booted_models
    from quail.backends.request import release_request_engines
    from quail.execution.execute import _BACKEND_STATE

    try:
        release_request_engines(_BACKEND_STATE)
    finally:
        if _BACKEND_STATE:
            release_booted_models(_BACKEND_STATE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sf", type=float, choices=(0.1, 0.5, 1.0), default=0.1)
    parser.add_argument("--only", help="comma-separated query IDs")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="quail")
    parser.add_argument("--device", required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--data-dir", help="directory containing input Parquet files")
    parser.add_argument("--ground-truth-collection")
    parser.add_argument("--output-dir", required=True, help="new run directory")
    args = parser.parse_args()
    try:
        run_suite(
            [value.strip() for value in args.only.split(",")] if args.only else None,
            sf=args.sf,
            config=quail.EngineConfig(
                gpus=args.gpus,
                model=args.model,
                backend=args.backend,
                device=args.device,
            ),
            data_dir=args.data_dir,
            ground_truth_collection=args.ground_truth_collection,
            output_dir=args.output_dir)
    finally:
        _close_process_engines()
    print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()
