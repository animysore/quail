"""Check retained answer rows and planted-flag filter answers on H100.

Prediction: both models retain only TRUE/FALSE output rows on the GPU,
with no full output-head copy. Input embeddings remain available. GPU
allocation stays within 0.3 GB of the resident-weight estimate, and all
48 planted-flag answers per attention path remain correct.

    uv run modal run experiments/cells/head_residency.py 2>&1 | tee /tmp/answer_head.log
"""

import json

import modal

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "numpy", "pyarrow")
    .env({"VLLM_CACHE_ROOT": "/root/.cache/kernels/vllm",
          "VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "QUAIL_CACHE_DIR": "/root/.cache/kernels",
          "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "TRITON_CACHE_DIR": "/root/.cache/kernels/triton"})
    .add_local_python_source("quail")
)

# House rule: attach to the existing app, never a fresh one.
app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)

GPU_KW = dict(image=image, gpu="H100!", memory=98304,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})

FILLER = ("The projector hummed while the reel changed and nobody in "
          "the back row noticed the splice. ")
N_DOCS = 48


def _planted(tokenizer):
    """48 documents with one planted flag each, and the question."""
    flags = [d % 3 != 0 for d in range(N_DOCS)]
    doc_ids = []
    for flag in flags:
        text = (FILLER * 6
                + f"\n\n[FLAGS] FLAG_7={'TRUE' if flag else 'FALSE'}")
        doc_ids.append(tokenizer(text, add_special_tokens=False)
                       ["input_ids"])
    question = ("\n\nIn the document above, is FLAG_7 set to TRUE? "
                "Answer with one word, TRUE or FALSE.\nAnswer:")
    q_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
    return flags, doc_ids, q_ids


@app.function(timeout=3600, **GPU_KW)
def probe(model_name: str) -> str:
    import gc
    import os

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.loop import run_filter
    from quail.backends.quail.executor.model import load_model
    from quail.backends.quail.executor.models import build_pipeline
    from quail.backends.quail.executor.readout import AnswerRows, AsyncAnswers
    from quail.cost import budgets
    from quail.specs import DEVICES, MODELS

    spec = MODELS[model_name]
    device = DEVICES["h100-sxm"]
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    model = load_model(spec.hf_name, revision=spec.revision)
    allocated = torch.cuda.memory_allocated() - baseline
    head_device = model.quail_answer_weights.device.type
    full_head_removed = model.lm_head is None
    answer_rows = model.quail_answer_weights.shape[0]

    chunk = budgets.chunk_budget(spec, device)
    arena_tok = budgets.arena_tokens(spec, device, chunk)
    arena = KVArena(n_layers=spec.layers,
                    n_pages=arena_tok // budgets.PAGE_TOKENS,
                    page_tokens=budgets.PAGE_TOKENS,
                    n_kv=spec.n_kv, d_head=spec.d_head,
                    dtype=torch.bfloat16)
    pipeline = build_pipeline(spec, model, arena)
    answerer = AnswerRows.from_tokenizer(torch, F, model, tokenizer)
    another_answerer = AnswerRows.from_tokenizer(torch, F, model, tokenizer)
    shared_answer_weights = answerer.weights is another_answerer.weights
    async_ans = AsyncAnswers(torch, answerer)

    flags, doc_ids, q_ids = _planted(tokenizer)
    correct = {}
    with torch.inference_mode():
        # the three forward-pass paths: unified paged, tree paged,
        # and the unpaged causal fast path
        for label, mode, writes in (
                ("unified", "unified", True),
                ("tree", "tree", True),
                ("unpaged", "unified", False)):
            answers, _, _ = run_filter(
                torch, arena, pipeline, async_ans, doc_ids, [q_ids],
                chunk, arena_writes=writes, attention_mode=mode)
            correct[label] = sum(
                int(answers[d][0]) == int(flags[d])
                for d in range(N_DOCS))
    torch.cuda.synchronize()

    resident_ok = abs(allocated - spec.W_resident) < 0.3e9
    answers_ok = all(c == N_DOCS for c in correct.values())
    result = dict(
        cell="head_residency", model=model_name,
        head_device=head_device, full_head_removed=full_head_removed,
        answer_rows=answer_rows, shared_answer_weights=shared_answer_weights,
        head_gib=round(spec.head_mem_bytes / 2**30, 3),
        allocated_after_load_gib=round(allocated / 2**30, 3),
        spec_as_loaded_gib=round(spec.W_mem / 2**30, 3),
        spec_resident_gib=round(spec.W_resident / 2**30, 3),
        chunk_tokens=chunk, arena_tokens=arena_tok,
        answers_correct=correct, answers_total=N_DOCS,
        peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
        resident_ok=resident_ok, answers_ok=answers_ok)
    result["pass"] = bool(
        resident_ok and answers_ok
        and full_head_removed and shared_answer_weights and head_device == "cuda")
    print(json.dumps(result, indent=2), flush=True)
    os.makedirs("/results/ablations", exist_ok=True)
    with open(f"/results/ablations/answer_head_{model_name}.json",
              "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    kernel_cache.commit()

    # leave the container clean for the next model's call
    del answerer, another_answerer, async_ans, pipeline, arena, model
    gc.collect()
    torch.cuda.empty_cache()
    return json.dumps(result)


@app.local_entrypoint()
def run(models: str = "qwen3-4b-fp8,qwen3-32b-fp8"):
    failed = []
    for name in [m.strip() for m in models.split(",") if m.strip()]:
        call = probe.spawn(name)
        print(f"[head_residency] {name} function call id "
              f"{call.object_id}", flush=True)
        out = json.loads(call.get())
        print(json.dumps(out, indent=2), flush=True)
        if not out["pass"]:
            failed.append(name)
    if failed:
        raise SystemExit(f"FAILED: {', '.join(failed)}")
    print("[head_residency] all models pass", flush=True)
