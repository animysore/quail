"""Measure a join whose anchors share prefixes, with and without sharing.

No QUAIL-B join has anchors with shared prefixes, so this joins the
scale 0.1 AGENT snapshots (1,772 anchors) with a table of four issue
categories. Each round runs the planner's choice (sharing on) or with
sharing forced off; the first round is warm-up. Pipelined vLLM runs
the same join on a second GPU, and each round's pairs are scored
against it.

    mkdir -p results/benchmark
    log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-join-prefix-sharing.log"
    uv run modal run --detach experiments/join_prefix_sharing.py 2>&1 | tee "$log"

The summary goes to /results/ablations/join_prefix_sharing/<stamp>/summary.json
and /results/ablations/join_prefix_sharing_<model>.json; the matched
pairs of every round and of vLLM go beside the summary.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import modal

from quail.bench.images import gpu_image
from quail.specs import H100_USD_PER_HOUR

ROUNDS = ("on", "on", "off", "on", "off")
DATA_DIR = Path("/results/quailb_data/sf0.1")
RUNS_DIR = Path("/results/ablations/join_prefix_sharing")
CATEGORIES = ("a crash or an exception", "wrong output from a function",
              "a missing feature", "a performance problem")

PROMPT_TEXT = (
    "{0}\n\nInstruction: answer TRUE if the issue the agent works on in "
    "the trace above is {1}. Answer FALSE otherwise."
)

PREDICTION_TEXT = (
    "Anchors dominate the work: 1,772 snapshots of about 9,800 tokens "
    "against four partners of a few tokens each. With sharing, fresh "
    "tokens drop from about 17.4 M plus partner rows to about 5.5 M "
    "plus partner rows, and query time drops by 2 to 3 times. Answers "
    "match between the two configurations on at least 99 percent of "
    "pairs."
)

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
kernel_cache = modal.Volume.from_name(
    "quail-kernel-cache", create_if_missing=True)

image = gpu_image()


class ForceNoSharing:
    """Turn prefix sharing off on every join when enabled."""

    name = "force_no_join_sharing"

    def __init__(self):
        self.enabled = False

    def rewrite(self, graph, context):
        from quail.physical import AiJoin, PhysicalGraph

        if not self.enabled:
            return None
        nodes = tuple(
            replace(node, share_prefixes=False) if isinstance(node, AiJoin)
            else node for node in graph.nodes)
        return PhysicalGraph(nodes, graph.root)


def _join(session):
    import quail

    return (session.docs("traces").alias("t")
            .ai_join(session.docs("categories").alias("c"),
                     quail.prompt(PROMPT_TEXT, quail.col("t.trace"),
                                  quail.col("c.category")),
                     selectivity=0.3, anchor="t")
            .select("t.id", "c.id"))


def _tables():
    import pyarrow as pa
    import pyarrow.parquet as pq

    traces = pq.read_table(DATA_DIR / "agent_traces.parquet",
                           columns=["id", "trace"])
    categories = pa.table({"id": [f"c{i}" for i in range(len(CATEGORIES))],
                           "category": list(CATEGORIES)})
    return traces, categories


def _save_pairs(path, table):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"trace": table.column(0),
                             "category": table.column(1)}), path)


@app.function(
    image=image,
    gpu="H100!",
    memory=98304,
    timeout=7200,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/kernels": kernel_cache,
        "/results": results_vol,
    },
)
def measure_vllm(stamp: str, model: str) -> dict:
    import quail

    results_vol.reload()
    traces, categories = _tables()
    config = quail.EngineConfig(model=model, device="h100-sxm",
                                backend="pipelined_vllm")
    with quail.Session(config) as session:
        session.register("traces", quail.DocumentProvider.from_table(
            traces, id_col="id"))
        session.register("categories", quail.DocumentProvider.from_table(
            categories, id_col="id"))
        result = _join(session).run()
        table = result.execute_stream().read_all()
        report = result.report
    out = RUNS_DIR / stamp
    out.mkdir(parents=True, exist_ok=True)
    _save_pairs(out / "pairs_vllm.parquet", table)
    results_vol.commit()
    runtime_s = report.get("model_wall_s", 0.0) + report.get("finish_s", 0.0)
    return {"backend": "pipelined_vllm", "runtime_s": runtime_s,
            "matched_pairs": table.num_rows,
            "cost_usd": runtime_s / 3600 * H100_USD_PER_HOUR}


@app.function(image=image, volumes={"/results": results_vol})
def score(stamp: str, model: str) -> dict:
    """Agreement of every round's pairs with vLLM's, over all pairs."""
    import pyarrow.parquet as pq

    results_vol.reload()
    out = RUNS_DIR / stamp
    summary = json.loads((out / "summary.json").read_text())

    def pairs(name):
        table = pq.read_table(out / name)
        return set(zip(table.column(0).to_pylist(),
                       table.column(1).to_pylist()))

    reference = pairs("pairs_vllm.parquet")
    evaluated = summary["anchors"] * summary["partners"]
    first = None
    for run in summary["runs"]:
        got = pairs(f"pairs_{run['round']}_{run['sharing']}.parquet")
        first = got if first is None else first
        run["vllm_agreement"] = 1 - len(got ^ reference) / evaluated
        run["vllm_precision"] = len(got & reference) / max(1, len(got))
        run["vllm_recall"] = len(got & reference) / max(1, len(reference))
    summary["vllm_matched_pairs"] = len(reference)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (RUNS_DIR.parent / f"join_prefix_sharing_{model}.json").write_text(
        json.dumps(summary, indent=2))
    results_vol.commit()
    return summary


@app.function(
    image=image,
    gpu="H100!",
    memory=98304,
    timeout=7200,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/kernels": kernel_cache,
        "/results": results_vol,
    },
)
def measure(stamp: str, model: str) -> dict:

    import quail
    from quail.builtins import built_in_registry
    from quail.physical import AiJoin

    results_vol.reload()
    traces, categories = _tables()
    force = ForceNoSharing()
    registry = built_in_registry().register_physical_rule(force)
    config = quail.EngineConfig(model=model, device="h100-sxm")
    runs = []
    first = None
    with quail.Session(config, registry=registry) as session:
        session.register("traces", quail.DocumentProvider.from_table(
            traces, id_col="id"))
        session.register("categories", quail.DocumentProvider.from_table(
            categories, id_col="id"))
        for round_index, setting in enumerate(ROUNDS):
            force.enabled = setting == "off"
            result = _join(session).run()
            table = result.execute_stream().read_all()
            pairs = set(zip(table.column(0).to_pylist(),
                            table.column(1).to_pylist()))
            if first is None:
                first = pairs
            out = RUNS_DIR / stamp
            out.mkdir(parents=True, exist_ok=True)
            _save_pairs(out / f"pairs_{round_index}_{setting}.parquet", table)
            join = next(node for node in result.plan.nodes
                        if isinstance(node, AiJoin))
            report = result.report
            runtime_s = report["model_wall_s"] + report["finish_s"]
            evaluated = len(CATEGORIES) * traces.num_rows
            runs.append({
                "round": round_index, "sharing": setting,
                "share_prefixes": join.share_prefixes,
                "runtime_s": runtime_s,
                "model_wall_s": report["model_wall_s"],
                "fresh_tokens": report["fresh_tokens"],
                "borrowed_prefix_tokens": next(
                    (metrics.get("borrowed_prefix_tokens", 0)
                     for metrics in report["node_metrics"].values()
                     if "borrowed_prefix_tokens" in metrics), 0),
                "matched_pairs": len(pairs),
                "pair_agreement": 1 - len(pairs ^ first) / evaluated,
                "cost_usd": runtime_s / 3600 * H100_USD_PER_HOUR,
            })
            print(json.dumps(runs[-1]), flush=True)
    summary = {"model": model, "device": config.device,
               "anchors": traces.num_rows, "partners": len(CATEGORIES),
               "prediction": PREDICTION_TEXT, "rounds": list(ROUNDS),
               "runs": runs}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (RUNS_DIR.parent / f"join_prefix_sharing_{model}.json").write_text(
        json.dumps(summary, indent=2))
    results_vol.commit()
    return summary


@app.local_entrypoint()
def main(model: str = "qwen3-4b-fp8"):
    from datetime import datetime, timezone

    print(f"prediction: {PREDICTION_TEXT}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    call = measure.spawn(stamp, model)
    print(f"function call id: {call.object_id}", flush=True)
    baseline = measure_vllm.spawn(stamp, model)
    print(f"vLLM function call id: {baseline.object_id}", flush=True)
    call.get()
    print(json.dumps(baseline.get()), flush=True)
    scored = score.spawn(stamp, model)
    print(f"score function call id: {scored.object_id}", flush=True)
    summary = scored.get()
    for run in summary["runs"]:
        print(json.dumps(run), flush=True)
    print(f"summary: {RUNS_DIR / stamp / 'summary.json'}", flush=True)
