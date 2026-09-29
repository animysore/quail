"""Load and run Quail models on the process's GPUs.

One GPU runs in the calling process. Several GPUs use one child process
per GPU, each with its own CUDA context and arena.
"""

import gc
import itertools
import time

from quail.backends.base import GpuContext
from quail.backends.quail.distributed import execute_distributed_graph
from quail.backends.quail.executor.arena import KVArena
from quail.backends.quail.executor.loop import warm_kernels
from quail.backends.quail.executor.model import load_model, resolve_model_path
from quail.backends.quail.executor.models import build_pipeline
from quail.backends.quail.executor.readout import AnswerRows, AsyncAnswers
from quail.backends.quail.graph import (
    _join_round_kv,
    _tuple_suffix,
    execute_single_graph,
    partner_list_builder,
    stage_partner_lists,
)
from quail.backends.quail.retention import apply_retention, retain_after_join
from quail.cost import budgets
from quail.execution.runner import ExecutionContext, SurvivorStream
from quail.execution.tokens import (
    DocumentPrefixes,
    chain_tokens,
    decode_payload_documents,
)
from quail.execution.types import PhysicalResponse
from quail.physical import (
    AiFilter,
    AiJoin,
    Scan,
    decode_graph,
)
from quail.progress import say, set_gpu_index

# Children outlive sessions so later queries can reuse their loaded models.
_CHILDREN: list = []


class LoadedGpu:
    """One model loaded on one GPU, reusable across queries."""

    def __init__(self, backend, context, answer_token_ids, *,
                 model_path=None):
        import torch
        import torch.nn.functional as F

        spec, device = context.model, context.device
        gpu_index = context.gpu_index
        self.torch = torch
        self.F = F
        self.spec = spec
        self._warmed = False
        self.prepared_boot = None

        set_gpu_index(gpu_index)
        free, total = torch.cuda.mem_get_info()
        say(f"loading {spec.hf_name} onto GPU {gpu_index}, "
            f"{free / 2**30:.1f} of {total / 2**30:.1f} GiB free")

        budget = budgets.chunk_budget(spec, device)
        t0 = time.perf_counter()
        self.model = load_model(model_path or spec.hf_name,
                                revision=None if model_path else spec.revision,
                                answer_token_ids=answer_token_ids,
                                max_batched_tokens=budget,
                                moe_backend=spec.moe_backend)
        self.load_model_s = time.perf_counter() - t0

        # cuBLAS allocates its handle outside PyTorch's caching allocator.
        # At the warm-up peak the allocator holds every free byte, so
        # create the handle now while memory is available.
        torch.cuda.current_blas_handle()
        probe = torch.ones(8, 64, device="cuda", dtype=torch.bfloat16)
        F.linear(probe, probe)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        full_pages, sliding_pages = budgets.arena_pages(spec, device, budget)
        self.arena = KVArena(n_layers=spec.layers,
                             n_pages=full_pages,
                             page_tokens=budgets.PAGE_TOKENS,
                             n_kv=spec.n_kv, d_head=spec.d_head,
                             dtype=torch.bfloat16,
                             layer_kv=spec.kv_shapes,
                             sliding_layers=spec.sliding_layer_set,
                             sliding_window=spec.sliding_window,
                             n_sliding_pages=sliding_pages)
        self.arena_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.pipeline = build_pipeline(spec, self.model, self.arena)
        self.pipeline_s = time.perf_counter() - t0

        self.execution = backend.start(context)
        self.execution.bind_loaded_model(
            model=self.model, arena=self.arena, pipeline=self.pipeline)

        self.async_ans = None
        self.chunk_tokens = None

    def bind_query(self, true_ids, false_ids, chunk_tokens, arena_pages=None):
        """Attach the answer rows, readout, and chunk budget for one query.

        arena_pages resizes the KV pools to the plan's split; nothing
        survives in the arena between queries.
        """
        if arena_pages is not None:
            self.arena.resize(*arena_pages, free_resident=True)
        rows = AnswerRows(self.torch, self.F, self.model, true_ids, false_ids)
        self.async_ans = AsyncAnswers(self.torch, rows)
        self.chunk_tokens = chunk_tokens
        self.execution.bind_query(
            torch=self.torch,
            async_answers=self.async_ans,
            answer_rows=rows,
            chunk_tokens=chunk_tokens,
        )

    def warm(self):
        """Compile and warm kernels once.

        Returns:
            Tuple of (seconds spent warming, compilation tier).
        """
        if self._warmed:
            return 0.0, None
        t0 = time.perf_counter()
        with self.torch.inference_mode():
            warm = warm_kernels(self.torch, self.arena, self.pipeline,
                                self.async_ans, self.chunk_tokens,
                                model_name=self.spec.hf_name)
        self.torch.cuda.synchronize()
        warm_s = time.perf_counter() - t0
        self._warmed = True
        return warm_s, warm["tier"]

    def close(self):
        """Release GPU resources held by the execution context."""
        close_fn = getattr(self.execution, "close", None)
        if callable(close_fn):
            close_fn()


def _boot_record(gpu, cold, warm_s, warm_tier, t_boot):
    """Assemble a boot timing dict from a loaded GPU and warm results."""
    kind = "cold" if cold or warm_tier else "warm"
    boot = {
        "kind": kind,
        "load_model_s": round(gpu.load_model_s if cold else 0.0, 2),
        "arena_s": round(gpu.arena_s if cold else 0.0, 2),
        "pipeline_s": round(gpu.pipeline_s if cold else 0.0, 2),
        "warm_kernels_s": round(warm_s, 2),
        "boot_s": round(time.perf_counter() - t_boot, 2),
    }
    if warm_tier:
        boot["warm_tier"] = warm_tier
    return boot


def quail_runtime_payload(request, graph) -> dict:
    """Build private Quail scheduler state from a standard request."""
    envelope = request.plan
    docs = {}
    for node in graph.nodes:
        if not isinstance(node, Scan):
            continue
        docs[node.alias] = request.inputs[node.input_id].documents
    return {
        "physical_plan": envelope,
        "model": envelope["model"],
        "workers": envelope["workers"],
        "docs": docs,
        "columns": request.column_tables(),
        **dict(envelope["settings"]),
    }


def _single_gpu_context(registry, envelope):
    """Build the GPU context for the single-GPU boot path."""
    return GpuContext(
        gpu_index=0,
        gpu_count=envelope["workers"],
        model=registry.model(envelope["model"]),
        device=registry.device(envelope["device"]),
        query_settings={},
    )


def _boot_for_query(runtime_state, backend, gpu_context,
                    chunk_tokens, true_ids, false_ids, arena_pages=None):
    """Load or reuse a GPU, bind a query, warm kernels."""
    from quail.backends.request import release_request_engines

    release_request_engines(runtime_state)
    key = (backend.name, gpu_context.model.name)
    gpu = runtime_state.get(key)
    t_boot = time.perf_counter()
    if gpu is None:
        loaded = [name for _, name in runtime_state if name != key[1]]
        if loaded:
            # one model's weights and KV arena take the GPU; another
            # model cannot load beside them
            say(f"releasing {', '.join(loaded)} to load {key[1]}")
            release_booted_models(runtime_state)
        gpu = LoadedGpu(backend, gpu_context, true_ids + false_ids)
        runtime_state[key] = gpu
        cold = True
    else:
        cold = False
    gpu.bind_query(true_ids, false_ids, chunk_tokens, arena_pages)
    warm_s, warm_tier = gpu.warm()
    boot = _boot_record(gpu, cold, warm_s, warm_tier, t_boot)
    say(f"model ready, boot {boot['boot_s']} s ({boot['kind']})")
    return gpu, boot


def prepare_quail_request(context) -> None:
    """Boot one GPU from the plan alone; the documents can arrive later."""
    from quail.backends.request import release_request_engines

    release_request_engines(context.runtime_state)
    if context.gpu_count != 1:
        return
    envelope = context.request.plan
    settings = dict(envelope["settings"])
    registry = context.registry
    gpu, boot = _boot_for_query(
        context.runtime_state, registry.backend(envelope["backend"]),
        _single_gpu_context(registry, envelope), settings["chunk_tokens"],
        settings["true_ids"], settings["false_ids"],
        settings.get("arena_pages"))
    gpu.prepared_boot = boot


def execute_quail_request(context):
    """Run one Quail request from a backend execution context."""
    from quail.backends.request import release_request_engines

    release_request_engines(context.runtime_state)
    payload = quail_runtime_payload(context.request, context.graph)
    if context.gpu_count == 1:
        backend = context.registry.backend(context.request.plan["backend"])
        return execute_quail_payload(
            payload, context.registry, context.graph, backend,
            context.runtime_state,
        )
    return execute_quail_multi(payload, context.registry, context.graph)


def _release_vllm_parallel_state() -> None:
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
    except ImportError:
        return

    destroy_model_parallel()
    destroy_distributed_environment()


def release_booted_models(runtime_state: dict) -> dict:
    """Release Quail GPU state before another engine uses this process."""
    import torch

    released = len(runtime_state)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for state in runtime_state.values():
        if isinstance(state, LoadedGpu):
            state.close()
    runtime_state.clear()
    _release_vllm_parallel_state()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return {
        "models_released": released,
        "vllm_parallel_state_released": True,
        "cuda_allocated_bytes": (
            int(torch.cuda.memory_allocated())
            if torch.cuda.is_available() else 0),
        "cuda_reserved_bytes": (
            int(torch.cuda.memory_reserved())
            if torch.cuda.is_available() else 0),
    }


def execute_quail_payload(payload, registry, graph, backend, runtime_state):
    """Execute one Quail payload on the worker's own GPU."""
    gpu_context = _single_gpu_context(registry, payload["physical_plan"])

    key = (backend.name, gpu_context.model.name)
    gpu = runtime_state.get(key)
    boot = None
    if isinstance(gpu, LoadedGpu):
        boot = gpu.prepared_boot
        gpu.prepared_boot = None
    if boot is None:
        gpu, boot = _boot_for_query(
            runtime_state, backend, gpu_context, payload["chunk_tokens"],
            payload["true_ids"], payload["false_ids"],
            payload.get("arena_pages"))
    say("running the query")

    state = _gpu_state(gpu)
    report = execute_single(state, payload, registry, graph)
    outputs = report.pop("_outputs")
    report.pop("filters", None)
    report.pop("joins", None)
    report["boot_s"] = boot["boot_s"]
    report["boot_kind"] = boot["kind"]
    report["boot"] = boot
    return PhysicalResponse(outputs, report)


def _gpu_state(gpu):
    """Build a state dict from a LoadedGpu for graph execution."""
    return {
        "torch": gpu.torch, "F": gpu.F,
        "model_execution": gpu.execution,
        "model": gpu.model, "arena": gpu.arena,
        "pipeline": gpu.pipeline, "spec": gpu.spec,
    }


def execute_single(state, payload: dict, registry, graph) -> dict:
    """Execute the typed Quail graph on one GPU."""
    runtime_state = {
        **state,
        "docs": decode_payload_documents(payload["docs"]),
        "columns": payload.get("columns", {}),
        "functions": registry.functions,
        "runtimes": registry.runtimes,
        "model_spec": state.get("spec") or state.get("model_spec"),
        "device": registry.device(payload["physical_plan"]["device"]),
        "chunk_tokens": payload["chunk_tokens"],
        "gpu_timing": payload.get("gpu_timing", False),
    }
    return execute_single_graph(runtime_state, payload, graph)


# ------------------------------------------------- multi-GPU dispatch
#
# One executor per GPU, as its own child process. The parent is the
# in-container coordinator: it splits the payload with
# quail.backends.quail.coordinator, runs the filter round, merges
# survivors, runs the join round, and merges the answers, with no
# network hop between rounds.

def _child_main(gpu_idx, conn):
    import os as _os
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    set_gpu_index(gpu_idx)
    state = {"gpu_index": gpu_idx, "gpu": None}
    while True:
        try:
            kind, data = conn.recv()
        except EOFError:
            break
        if kind == "shutdown":
            break
        try:
            if kind == "boot":
                state["registry"] = data["registry"]
                state["model_path"] = data["model_path"]
                conn.send(("ok", _child_boot(state, data)))
            elif kind == "filters":
                conn.send(("ok", _child_filters(state, data)))
            elif kind == "scores":
                gpu = state["gpu"]
                inputs = data["inputs"]
                if "documents" in inputs:
                    state["score_documents"] = inputs["documents"]
                inputs["documents"] = state["score_documents"]
                with gpu.torch.inference_mode():
                    result = gpu.execution.execute(data["node"], inputs)
                gpu.torch.cuda.synchronize()
                conn.send(("ok", result))
            elif kind == "joins":
                conn.send(("ok", _child_joins(state, data)))
        except Exception:
            import traceback
            conn.send(("err", traceback.format_exc()))


def _child_boot(state, sub):
    envelope = sub["physical_plan"]
    registry = state["registry"]
    spec = registry.model(sub["model"])
    device = registry.device(envelope["device"])
    backend = registry.backend(envelope["backend"])
    t_boot = time.perf_counter()
    gpu = state.get("gpu")
    if gpu is None:
        gpu = LoadedGpu(backend, GpuContext(
            gpu_index=state["gpu_index"], gpu_count=sub["workers"],
            model=spec, device=device, query_settings={},
        ), sub["true_ids"] + sub["false_ids"],
            model_path=state["model_path"])
        state["gpu"] = gpu
        cold = True
    else:
        cold = False
    gpu.bind_query(sub["true_ids"], sub["false_ids"],
                   sub["chunk_tokens"], sub.get("arena_pages"))
    state["runtime_context"] = ExecutionContext(
        runtimes=registry.runtimes, model_execution=gpu.execution,
    )
    warm_s, warm_tier = gpu.warm()
    boot = _boot_record(gpu, cold, warm_s, warm_tier, t_boot)
    say(f"model ready, boot {boot['boot_s']} s ({boot['kind']})")
    return boot


def _child_filters(state, sub):
    gpu = state.get("gpu")
    torch = gpu.torch if gpu else state["torch"]
    arena = gpu.arena if gpu else state["arena"]
    if sub.get("start_query", True):
        _reset_child_query(state)
        config = sub.get("retention", {})
        apply_retention(arena, config, config.get("initial", {}))
    out = dict(filters={}, survivors={}, retained={}, fresh_tokens=0)
    pre = sub.get("pre_ids") or []
    filter_limit = sub.get("filter_limit")
    t0 = time.perf_counter()
    with torch.inference_mode():
        node_id = sub.get("node_id")
        if node_id is not None:
            graph = decode_graph(
                sub["physical_plan"]["graph"],
                state["registry"].codecs,
            )
            node = graph.node(node_id)
            if not isinstance(node, AiFilter):
                raise TypeError(
                    f"child filter received {node.type_name!r}")
            alias = node.alias
            index = sub["doc_index"][alias]
            runtime_context = state["runtime_context"]
            result = runtime_context.runtimes[node.runtime_key].execute(
                node,
                {
                    "documents": DocumentPrefixes(
                        pre,
                        sub["docs"][alias],
                        range(len(sub["docs"][alias])),
                    ),
                    "document_ids": index,
                    "limit": filter_limit,
                    "retain_survivors": node.keep_kv,
                },
                runtime_context,
            )
            answers = result.outputs[f"filter_answers:{alias}"]
            if node.keep_kv:
                out["retained"][alias] = sorted(
                    key[1] for key in arena.retained_keys()
                    if key[0] == alias)
            out["fresh_tokens"] += result.metrics.fresh_tokens
            out["filters"][alias] = {
                int(document): row for document, row in answers.items()
            }
            out["survivors"][alias] = list(
                result.outputs[f"ids:{alias}"]
            )
    for alias, document in arena.retained_keys():
        out["retained"].setdefault(alias, []).append(document)
    out["retained"] = {alias: sorted(set(documents))
                       for alias, documents in out["retained"].items()}
    torch.cuda.synchronize()
    out["wall_s"] = round(time.perf_counter() - t0, 2)
    out["peak_gib"] = round(
        torch.cuda.max_memory_allocated() / 2**30, 2)
    return out


def _child_joins(state, sub):
    gpu = state.get("gpu")
    torch = gpu.torch if gpu else state["torch"]
    arena = gpu.arena if gpu else state["arena"]
    if sub.get("start_query", False):
        _reset_child_query(state)
    pre = sub.get("pre_ids") or []
    registry = state["registry"]
    encoded_node = sub["physical_node"]
    node = registry.codecs[encoded_node["type"]].decode(encoded_node)
    if not isinstance(node, AiJoin):
        raise TypeError(f"child join received {node.type_name!r}")
    group = [stage.runtime_spec() for stage in node.stages]
    anchor_alias = node.anchor
    anchors_glob = list(sub["anchor_index"])
    anchor_docs = sub["anchor_docs"]
    config = sub.get("retention", {})
    live = {alias: partner["index"] for alias, partner in sub["partners"].items()}
    live[anchor_alias] = anchors_glob
    apply_retention(arena, config,
                    config.get("before", {}).get(node.node_id, {}), live)
    retain_anchor = bool(sub.get("retain_anchor"))
    encoded_filter = sub.get("stream_filter_node")
    filter_node = None
    if encoded_filter is not None:
        filter_node = registry.codecs[encoded_filter["type"]].decode(
            encoded_filter)
        if not isinstance(filter_node, AiFilter) \
                or filter_node.alias != anchor_alias:
            raise TypeError("the streamed chain must filter the anchor")
    out_joins, tokens_total = [], 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        stage_suffixes, tuple_globs = [], []
        for j in group:
            locals_ = [range(len(sub["partners"][p]["index"]))
                       for p in j["partners"]]
            combos = list(itertools.product(*locals_))
            tuple_globs.append(
                [[sub["partners"][p]["index"][c]
                  for p, c in zip(j["partners"], combo)]
                 for combo in combos])
            part_docs = {p: sub["partners"][p]["docs"]
                         for p in j["partners"]}
            stage_suffixes.append(
                [_tuple_suffix(j, part_docs, combo)
                 for combo in combos])
        # this GPU's anchors only
        lists_for = partner_list_builder(
            group, tuple_globs,
            {int(position): {int(anchor): partners
                             for anchor, partners in rows.items()}
             for position, rows in sub.get("pairs", {}).items()})
        if filter_node is None:
            prefixes = [chain_tokens(pre, d) for d in anchor_docs]
            anchor_keys = [(anchor_alias, g) for g in anchors_glob]
            round_kv = _join_round_kv(anchor_keys, arena)
            anchor_stream = None
        else:
            # filled by the driver as this GPU's shard streams through
            prefixes, anchor_keys = [], []
            round_kv = None
            anchor_stream = {
                "node": filter_node,
                "documents": DocumentPrefixes(
                    pre, anchor_docs, range(len(anchor_docs))),
                "document_ids": anchors_glob,
                "stream": SurvivorStream(filter_node, anchors_glob),
            }

        def anchor_done(a, row):
            matched = any(row)
            alive = (not matched if group[-1]["semantics"] == "anti"
                     else matched)
            if alive and retain_anchor:
                retain_after_join(arena, anchor_keys[a], len(prefixes[a]), config,
                                  config.get("after", {}).get(node.node_id, {}))
            else:
                arena.free_key(anchor_keys[a])

        runtime_context = state["runtime_context"]
        result = runtime_context.runtimes[node.runtime_key].execute(
            node,
            {
                "prefixes": prefixes,
                "stage_suffixes": stage_suffixes,
                "stage_frames": [
                    join.get("frame") or [] for join in group
                ],
                "anchor_keys": anchor_keys,
                "anchor_done": anchor_done,
                "anchor_stream": anchor_stream,
                "kv_round": round_kv,
                "anchor_ids": None if filter_node else anchors_glob,
                "partner_indices": {
                    stage.written_pos: tuples
                    for stage, tuples in zip(node.stages, tuple_globs)
                },
                "anchor_partners": lists_for,
                "group": group,
            },
            runtime_context,
        )
        ans = result.metrics.extension["answers"]
        filter_out = {}
        if filter_node is not None:
            anchors_glob = [key[1] for key in anchor_keys]
            round_kv = dict(hits=result.metrics.kv_hits,
                            misses=result.metrics.kv_misses)
            chain = anchor_stream["stream"].finalized_result()
            answers = chain.outputs[f"filter_answers:{anchor_alias}"]
            filter_out = dict(
                filters={anchor_alias: {
                    int(document): row for document, row in answers.items()
                }},
                survivors={anchor_alias: list(
                    chain.outputs[f"ids:{anchor_alias}"])},
                filter_fresh_tokens=chain.metrics.fresh_tokens,
            )
        last = ans[-1] if ans else {}
        for a, key in enumerate(anchor_keys):
            if not arena.is_resident(key):
                continue
            matched = any(last.get(a, []))
            alive = (not matched if group[-1]["semantics"] == "anti"
                     else matched)
            if alive and retain_anchor:
                retain_after_join(arena, key, len(prefixes[a]), config,
                                  config.get("after", {}).get(node.node_id, {}))
            else:
                arena.free_key(key)
        tokens_total += result.metrics.fresh_tokens
        partner_lists = stage_partner_lists(group, lists_for, anchors_glob)
        for si, j in enumerate(group):
            out_joins.append(dict(
                rows={int(a): row for a, row in ans[si].items()},
                anchor_index=anchors_glob,
                partner_index=tuple_globs[si],
                anchor_partners=partner_lists[si]))
    if sub.get("final_group"):
        for key in arena.resident_keys():
            arena.free_key(key)
    torch.cuda.synchronize()
    retained = {}
    for alias, document in arena.retained_keys():
        retained.setdefault(alias, []).append(document)
    return dict(joins=out_joins, fresh_tokens=tokens_total,
                retained={alias: sorted(documents)
                          for alias, documents in retained.items()},
                kv_round=round_kv,
                **filter_out,
                kv_totals=dict(
                    evicted_keys=arena.evicted_keys,
                    evicted_pages=arena.evicted_pages,
                    evicted_prefix_tokens=arena.evicted_prefix_tokens),
                wall_s=round(time.perf_counter() - t0, 2))


def _reset_child_query(state):
    """Clear KV and counters before a child starts a new query."""
    gpu = state.get("gpu")
    arena = gpu.arena if gpu else state["arena"]
    for key in arena.resident_keys():
        arena.free_key(key)
    arena.reset_stats()


def _ensure_children(k):
    import multiprocessing as mp
    if len(_CHILDREN) >= k:
        return
    ctx = mp.get_context("spawn")
    for gpu in range(len(_CHILDREN), k):
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(target=_child_main, args=(gpu, child_conn),
                           daemon=True)
        proc.start()
        _CHILDREN.append((proc, parent_conn))


def _round(kind, subs):
    """Send one round to children and collect results."""
    for (_, conn), sub in zip(_CHILDREN, subs):
        conn.send((kind, sub))
    outs = []
    for (_, conn), _sub in zip(_CHILDREN, subs):
        status, data = conn.recv()
        if status != "ok":
            raise RuntimeError(f"GPU child failed:\n{data}")
        outs.append(data)
    return outs


def execute_quail_multi(payload, registry, graph):
    """Execute the typed Quail graph across GPU child processes."""
    payload = dict(payload)
    payload["docs"] = decode_payload_documents(payload["docs"])
    gpu_count = payload["workers"]
    spec = registry.model(payload["model"])
    started = time.perf_counter()
    model_path = resolve_model_path(spec.hf_name, spec.revision)
    model_files_s = time.perf_counter() - started
    _ensure_children(gpu_count)
    setup = {key: payload[key] for key in (
        "model", "physical_plan", "workers", "chunk_tokens", "true_ids", "false_ids",
    )}
    setup["arena_pages"] = payload.get("arena_pages")
    setup.update(registry=registry, model_path=model_path)
    boots = _round("boot", [setup] * gpu_count)
    boot_s = round(time.perf_counter() - started, 2)
    say(f"all {gpu_count} GPUs ready; running the query")
    report = execute_distributed_graph(
        payload,
        graph,
        gpu_count,
        _round,
        spec,
        registry.device(payload["physical_plan"]["device"]),
        registry.runtimes,
        registry,
    )
    slowest = max(boots, key=lambda boot: boot["boot_s"])
    report["boot_s"] = boot_s
    report["boot_kind"] = "cold" if any(b["kind"] == "cold" for b in boots) else "warm"
    report["boot"] = {
        **slowest,
        "model_files_s": round(model_files_s, 2),
        "boot_s": boot_s,
    }
    outputs = report.pop("_outputs")
    report.pop("filters", None)
    report.pop("joins", None)
    return PhysicalResponse(outputs, report)
