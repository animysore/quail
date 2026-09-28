"""Measure a prefix-sharing filter both ways: tree attention against unified.

The tree_attention planner rule picks a filter's path from its prefix
tree by a roofline. No QUAIL-B corpus has documents that share one
prefix, so this builds two from the scale 0.1 data: 5,000 records
after one 8,000-token header, each a 20-word review excerpt (short
records, where the rule says tree) or a whole review (long records,
where the rule says unified). Each corpus runs the rule's choice once
as warm-up, then both paths forced twice each.

    mkdir -p results/benchmark
    log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-filter-attention-paths.log"
    uv run modal run --detach experiments/filter_attention_paths.py 2>&1 | tee "$log"

The corpora and the summary go under
/results/ablations/filter_attention_paths/<stamp>/ and the summary to
/results/ablations/filter_attention_paths_qwen3-4b-fp8.json.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import modal

from quail.bench.images import gpu_image
from quail.specs import H100_USD_PER_HOUR

ROUNDS = ("rule", "tree", "unified", "tree", "unified")
DATA_DIR = Path("/results/quailb_data/sf0.1")
RUNS_DIR = Path("/results/ablations/filter_attention_paths")
SUMMARY_PATH = Path(
    "/results/ablations/filter_attention_paths_qwen3-4b-fp8.json")
RECORDS = 5000
HEADER_WORDS = 6000
EXCERPT_WORDS = 20

PROMPT_TEXT = (
    "The passages below are background reading. After them comes one "
    "movie review, or an excerpt of one.\n\n{0}\n\nInstruction: answer "
    "TRUE if the review or excerpt expresses a positive opinion of the "
    "movie. Answer FALSE otherwise."
)

PREDICTION_TEXT = (
    "Short records (about 45 own tokens after an 8,192-token header, "
    "5,000 of them): the roofline puts unified attention at 1.8 seconds "
    "and tree at 0.6 seconds over a 0.8-second forward pass, so tree "
    "will be at least 20 percent faster and the rule will pick it. Long "
    "records (whole reviews, about 330 own tokens): the roofline puts "
    "unified at 4.0 seconds and tree at 4.5 over a 6.0-second forward "
    "pass, so the rule will pick unified and the two paths will run "
    "within 5 percent of each other. Fresh tokens match between paths "
    "on one corpus; the paths quantize differently, so a few answers "
    "may differ, with at least 99 percent of documents agreeing."
)

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
kernel_cache = modal.Volume.from_name(
    "quail-kernel-cache", create_if_missing=True)

image = gpu_image()


class ForceFilterAttention:
    """Overwrite every filter's attention with the path under test."""

    name = "force_filter_attention"

    def __init__(self):
        self.path = None

    def rewrite(self, graph, context):
        from quail.physical import AiFilter, PhysicalGraph

        if self.path is None:
            return None
        nodes = tuple(
            replace(node, attention=self.path) if isinstance(node, AiFilter)
            else node for node in graph.nodes)
        return PhysicalGraph(nodes, graph.root)


def build_corpora(stamp: str) -> dict[str, Path]:
    """Write the two record corpora and return their parquet paths."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    evidence = pq.read_table(DATA_DIR / "evidence.parquet").column(
        "text").to_pylist()
    reviews = pq.read_table(DATA_DIR / "reviews.parquet").column(
        "body").to_pylist()[:RECORDS]
    if len(reviews) < RECORDS:
        raise ValueError(f"{len(reviews)} reviews, {RECORDS} needed")
    words, parts = 0, []
    for text in evidence:
        parts.append(text)
        words += len(text.split())
        if words >= HEADER_WORDS:
            break
    header = "\n\n".join(parts)
    out = RUNS_DIR / stamp
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, bodies in (
            ("short", [" ".join(review.split()[:EXCERPT_WORDS])
                       for review in reviews]),
            ("long", reviews)):
        table = pa.table({
            "id": [f"{name}-{i}" for i in range(len(bodies))],
            "body": [f"{header}\n\nReview: {body}" for body in bodies],
        })
        paths[name] = out / f"{name}.parquet"
        pq.write_table(table, paths[name])
    return paths


def filter_attention(result) -> str:
    """The attention path the executed plan's filter ran."""
    from quail.physical import AiFilter

    return next(node.attention for node in result.plan.nodes
                if isinstance(node, AiFilter))


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
def measure(stamp: str) -> dict:
    import quail
    from quail.builtins import built_in_registry

    results_vol.reload()
    paths = build_corpora(stamp)
    force = ForceFilterAttention()
    registry = built_in_registry().register_physical_rule(force)
    config = quail.EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")
    runs = []
    with quail.Session(config, registry=registry) as session:
        for name, path in paths.items():
            session.register(name, quail.DocumentProvider.from_parquet(
                str(path), id_col="id"))
            first = None
            for round_index, path_name in enumerate(ROUNDS):
                force.path = None if path_name == "rule" else path_name
                query = session.docs(name).alias("r").ai_filter(
                    quail.prompt(PROMPT_TEXT, quail.col("r.body")),
                    selectivity=0.5).select("r.id")
                result = query.run()
                ids = set(result.execute_stream().read_all().column(
                    0).to_pylist())
                if first is None:
                    first = ids
                report = result.report
                runtime_s = report["model_wall_s"] + report["finish_s"]
                runs.append({
                    "corpus": name, "asked": path_name,
                    "attention": filter_attention(result),
                    "round": round_index,
                    "runtime_s": runtime_s,
                    "model_wall_s": report["model_wall_s"],
                    "fresh_tokens": report["fresh_tokens"],
                    "survivors": len(ids),
                    "agreement": 1 - len(ids ^ first) / RECORDS,
                    "cost_usd": runtime_s / 3600 * H100_USD_PER_HOUR,
                })
                print(json.dumps(runs[-1]), flush=True)
    summary = {
        "model": config.model,
        "device": config.device,
        "records": RECORDS,
        "header_words": HEADER_WORDS,
        "corpora": {name: str(path) for name, path in paths.items()},
        "prediction": PREDICTION_TEXT,
        "rounds": list(ROUNDS),
        "runs": runs,
    }
    (RUNS_DIR / stamp / "summary.json").write_text(json.dumps(summary, indent=2))
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    return summary


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone

    print(f"prediction: {PREDICTION_TEXT}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    call = measure.spawn(stamp)
    print(f"function call id: {call.object_id}", flush=True)
    summary = call.get()
    for run in summary["runs"]:
        print(json.dumps(run), flush=True)
    print(f"summary: {SUMMARY_PATH}", flush=True)
