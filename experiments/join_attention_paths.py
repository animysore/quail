"""Measure a join both ways: tree attention against unified.

The tree_attention planner rule picks the path by a roofline. On
QUAIL-B it picks "unified" only for the LePaRD joins: a 208-token
anchor read by partners of about 63 rows each. This runs one such
query and one the rule leaves on the tree path, each forced both
ways in one session, and records runtime and fresh tokens.

    mkdir -p results/benchmark
    log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-join-attention-paths.log"
    uv run modal run --detach experiments/join_attention_paths.py 2>&1 | tee "$log"

Each round is a QUAIL-B run under /results/ablations/join_attention_paths/
so the answers are saved and scored; the summary goes to
/results/ablations/join_attention_paths_qwen3-4b-fp8.json.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import modal

from quail.bench.images import gpu_image
from quail.specs import H100_USD_PER_HOUR

QUERIES = ("LEP-4", "FEV-4")
ROUNDS = ("tree", "unified", "tree", "unified")
DATA_DIR = "/results/quailb_data/sf0.1"
RUNS_DIR = Path("/results/ablations/join_attention_paths")
SUMMARY_PATH = Path("/results/ablations/join_attention_paths_qwen3-4b-fp8.json")
FIELDS = ("runtime_s", "fresh_tokens", "answers_evaluated", "answers_correct",
          "predicted_rows", "expected_rows", "matching_rows", "cost_usd")

PREDICTION_TEXT = (
    "On LEP-4 (208-token anchors, partners of about 63 rows, 5.6 per "
    "anchor) the roofline says unified is cheaper: unified will run "
    "within 5 percent of tree, either way. On FEV-4 (439-token anchors, "
    "16-row partners, 41 per anchor) the roofline says tree: tree will "
    "be at least 5 percent faster. Fresh tokens match between paths; "
    "the two paths quantize differently, so a few answers may differ, "
    "with correct answers within 1 percent of each other."
)

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
kernel_cache = modal.Volume.from_name(
    "quail-kernel-cache", create_if_missing=True)

image = gpu_image()


class ForceJoinAttention:
    """Overwrite every join's attention with the path under test."""

    name = "force_join_attention"

    def __init__(self):
        self.path = "tree"

    def rewrite(self, graph, context):
        from quail.physical import AiJoin, PhysicalGraph

        nodes = tuple(
            replace(node, attention=self.path) if isinstance(node, AiJoin)
            else node for node in graph.nodes)
        return PhysicalGraph(nodes, graph.root)


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
def measure(query_ids: list[str], stamp: str) -> dict:
    import pyarrow.parquet as pq

    import quail
    import quail_b as benchmark
    from quail.bench.quailb import run_query
    from quail.builtins import built_in_registry

    force = ForceJoinAttention()
    registry = built_in_registry().register_physical_rule(force)
    config = quail.EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")
    runs = []
    with quail.Session(config, registry=registry) as session:
        for query_id in query_ids:
            for round_index, path in enumerate(ROUNDS):
                force.path = path
                output_dir = RUNS_DIR / stamp / f"{query_id}_{path}_{round_index}"
                benchmark.run(
                    partial(run_query, session), queries=[query_id],
                    scale_factor=0.1, output_dir=output_dir, data_dir=DATA_DIR,
                    gpu_count=1, gpu_hourly_rate_usd=H100_USD_PER_HOUR,
                    metadata={
                        "engine": "quail", "model": config.model,
                        "configuration": asdict(config),
                        "join_attention": path,
                        "warmup": "shared session; the first round is warm-up",
                    })
                benchmark.report(output_dir, rescore=False)
                row = pq.read_table(
                    output_dir / "measurements.parquet").to_pylist()[0]
                runs.append({
                    "query": query_id, "attention": path, "round": round_index,
                    "run_dir": str(output_dir),
                    **{field: row.get(field) for field in FIELDS}})
                print(json.dumps(runs[-1]), flush=True)
                results_vol.commit()
    summary = {
        "model": config.model,
        "device": config.device,
        "scale_factor": 0.1,
        "prediction": PREDICTION_TEXT,
        "rounds": list(ROUNDS),
        "runs": runs,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    return summary


@app.local_entrypoint()
def main(queries: str = ",".join(QUERIES)):
    from datetime import datetime, timezone

    print(f"prediction: {PREDICTION_TEXT}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    call = measure.spawn(queries.split(","), stamp)
    print(f"function call id: {call.object_id}", flush=True)
    summary = call.get()
    for run in summary["runs"]:
        print(json.dumps(run), flush=True)
    print(f"summary: {SUMMARY_PATH}", flush=True)
