"""Quail model backend."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from quail.backends.base import GpuContext
from quail.backends.quail.executor import loop
from quail.backends.quail.executor.models import supported_archs
from quail.backends.quail.executor.score import QuailScorer
from quail.backends.quail.graph import filter_result, stage_partner_lists
from quail.backends.quail.worker import execute_quail_request, prepare_quail_request
from quail.execution.reranker import RerankerModelExecution
from quail.execution.runner import NodeMetrics, NodeResult, SurvivorStream
from quail.execution.tokens import DocumentKeys, prefix_tree
from quail.logical import Alias, is_score, shared_preamble
from quail.logical.prompts import true_false_token_ids
from quail.physical import (
    AiFilter,
    AiJoin,
    AiScore,
    Barrier,
    PhysicalNode,
)
from quail.planner import plan_quail
from quail.planner.physical_optimizer import (
    ModelRegion,
    PhysicalCandidate,
    PlanningContext,
    SupportResult,
)
from quail.planner.plan import Refusal
from quail.planner.reranker import plan_reranker
from quail.progress import logger


class QuailModelExecution:
    """Quail model state shared by model nodes on one GPU executor."""

    def __init__(self, context: GpuContext):
        self.context = context
        self._state: dict[str, Any] = {}
        self._reranker: RerankerModelExecution | None = None

    @property
    def state(self) -> Mapping[str, Any]:
        """Return Quail's private loaded model state."""
        return self._state

    def bind_loaded_model(self, *, model, arena, pipeline) -> None:
        """Attach the loaded model objects owned by this executor."""
        self._state.update(model=model, arena=arena, pipeline=pipeline)

    def close(self) -> None:
        """Drop every reference to the loaded model so its memory can go."""
        self._state.clear()
        self._reranker = None

    def bind_query(self, *, torch, async_answers, answer_rows,
                   chunk_tokens: int) -> None:
        """Attach state that is valid for the current query.

        answer_rows are the retained output rows every readout scores
        against; async_answers is the TRUE/FALSE readout over them.
        """
        self._state.update(
            torch=torch,
            async_answers=async_answers,
            answer_rows=answer_rows,
            chunk_tokens=chunk_tokens,
        )

    def execute(
        self,
        node: PhysicalNode,
        inputs: Mapping[str, Any],
    ) -> Any:
        if isinstance(node, AiScore):
            execution = self._reranker_execution(inputs["documents"])
            if "score_rows" in inputs:
                return execution.execute_rows(node, inputs["score_rows"])
            return execution.execute(node, inputs["score_inputs"])
        if not isinstance(node, (AiFilter, AiJoin)):
            raise TypeError(
                f"Quail cannot execute physical node {node.type_name!r}")
        return self._execute_quail_node(node, inputs)

    def _reranker_execution(self, documents) -> RerankerModelExecution:
        """Return the reranker bound to this query's document tokens."""
        execution = self._reranker
        if execution is None or execution.documents is not documents:
            execution = RerankerModelExecution(GpuContext(
                gpu_index=self.context.gpu_index,
                gpu_count=self.context.gpu_count,
                model=self.context.model, device=self.context.device,
                query_settings={
                    "documents": documents,
                    "reranker": QuailScorer(self._state),
                },
            ))
            self._reranker = execution
        return execution

    def _execute_quail_node(
        self,
        node: AiFilter | AiJoin,
        inputs: Mapping[str, Any],
    ) -> Any:

        missing = {
            "torch", "async_answers", "chunk_tokens",
            "arena", "pipeline",
        } - set(self._state)
        if missing:
            raise RuntimeError(
                f"Quail model execution is missing state {sorted(missing)}")
        torch = self._state["torch"]
        arena = self._state["arena"]
        pipeline = self._state["pipeline"]
        async_answers = self._state["async_answers"]
        chunk_tokens = self._state["chunk_tokens"]

        if isinstance(node, AiFilter):
            document_ids = inputs["document_ids"]
            if node.pin_survivors:
                stream = SurvivorStream(node, document_ids)

                return NodeResult(
                    outputs={
                        f"ids:{node.alias}": stream,
                        f"filter_answers:{node.alias}": {},
                    },
                    finalize=stream.finalized_result,
                )
            retain_survivors = inputs.get("retain_survivors", ())
            if retain_survivors is False:
                retain_survivors = ()
            # sorted admission would change which rows a limit keeps
            tree = (None if inputs.get("limit") is not None
                    else _prefix_tree(node, inputs["documents"], arena))
            stats = {}
            answers, spans, tokens = loop.run_filter(
                torch,
                arena,
                pipeline,
                async_answers,
                inputs["documents"],
                [list(question) for question in node.question_token_ids],
                chunk_tokens,
                limit=inputs.get("limit"),
                arena_writes=node.arena_writes,
                arena_keys=DocumentKeys(node.alias, document_ids),
                retain_survivors=retain_survivors,
                document_done=inputs.get("document_done"),
                prefix_tree=tree,
                attention_mode=node.attention or None,
                stats=stats,
            )
            return filter_result(
                node, answers, tokens, document_ids,
                gpu_s=_gpu_seconds(torch, spans, inputs),
                chunks=_chunks(spans, inputs),
                borrowed_tokens=stats.get("borrowed_tokens", 0))

        stage_frames = inputs["stage_frames"]
        stream = inputs.get("anchor_stream")
        source = None
        if stream is not None:
            filter_node = stream["node"]
            # a pinned survivor's pages must cover the join's largest frame
            source = loop.FilterStream(
                torch,
                arena,
                pipeline,
                async_answers,
                stream["documents"],
                [list(question)
                 for question in filter_node.question_token_ids],
                chunk_tokens,
                arena_writes=True,
                arena_keys=DocumentKeys(filter_node.alias,
                                        stream["document_ids"]),
                hold_survivors=True,
                hold_extra_tokens=filter_node.hold_tokens,
                document_done=stream.get("document_done"),
                prefix_tree=_prefix_tree(
                    filter_node, stream["documents"], arena),
                attention_mode=filter_node.attention or None,
            )
        lists_for = inputs.get("anchor_partners")
        join_stats = {}
        answers, spans, tokens = loop.run_join(
            torch,
            arena,
            pipeline,
            async_answers,
            inputs["prefixes"],
            inputs["stage_suffixes"],
            chunk_tokens,
            stage_frames=stage_frames,
            anchor_keys=inputs["anchor_keys"],
            anchor_done=inputs["anchor_done"],
            anchor_source=source,
            anchor_partners=(
                None if lists_for is None else lambda key: lists_for(key[1])),
            anchor_batch=inputs.get("anchor_batch"),
            attention_mode=node.attention or None,
            prefix_tree=(None if source is not None else _prefix_tree(
                node, inputs["prefixes"], arena)),
            stats=join_stats,
        )
        if source is not None:
            # admission order; a per-batch function may have dropped some
            anchor_ids = [key[1] for key in inputs["anchor_keys"]]
            kv_round = {"hits": len(anchor_ids), "misses": 0}
            stream["stream"].complete(filter_result(
                filter_node,
                source.answers,
                source.tokens,
                stream["document_ids"],
                gpu_s=_gpu_seconds(torch, source.spans, inputs),
                chunks=_chunks(source.spans, inputs),
            ))
        else:
            anchor_ids = list(inputs["anchor_ids"])
            kv_round = inputs.get("kv_round") or {}
        group = inputs["group"]
        last = answers[-1] if answers else {}
        matched = {
            anchor_ids[int(local)]
            for local, row in last.items() if any(row)
        }
        if group[-1]["semantics"] == "anti":
            survivors = [document for document in anchor_ids
                         if document not in matched]
        else:
            survivors = [document for document in anchor_ids
                         if document in matched]
        outputs = {f"ids:{node.anchor}": survivors}
        partner_lists = stage_partner_lists(group, lists_for, anchor_ids)
        for stage, stage_answers, members in zip(
                node.stages, answers, partner_lists):
            outputs[f"join_answers:{stage.written_pos}"] = {
                "rows": stage_answers,
                "anchor_index": anchor_ids,
                "partner_index": inputs["partner_indices"][
                    stage.written_pos
                ],
                "anchor_partners": members,
                "anchor": node.anchor,
                "partners": list(stage.partners),
                "semantics": stage.semantics,
                "selectivity": stage.selectivity,
                "written_pos": stage.written_pos,
            }
        return NodeResult(
            outputs=outputs,
            metrics=NodeMetrics(
                input_rows=len(anchor_ids),
                output_rows=len(survivors),
                evaluated_document_pairs=sum(
                    sum(len(row) for row in stage.values())
                    for stage in answers
                ),
                kv_hits=kv_round.get("hits", 0),
                kv_misses=kv_round.get("misses", 0),
                fresh_tokens=tokens,
                gpu_s=_gpu_seconds(torch, spans, inputs),
                chunks=_chunks(spans, inputs),
                extension={
                    "answers": answers,
                    **({"borrowed_prefix_tokens": join_stats["borrowed_tokens"]}
                       if join_stats.get("borrowed_tokens") else {}),
                },
            ),
        )


def _prefix_tree(node, documents, arena):
    """The node's prefix tree, or None when the plan did not ask for one."""
    if not node.share_prefixes:
        return None
    started = time.perf_counter()
    tree = prefix_tree(documents, arena.page_tokens)
    logger.info(
        "prefix sharing on %s: %s documents borrow %s tokens "
        "(tree built in %.2f s)", getattr(node, "alias", None) or node.anchor,
        len(documents),
        tree.shared_tokens, time.perf_counter() - started)
    return tree


def _gpu_seconds(torch, spans, inputs) -> float:
    """Seconds the loop's forward chunks ran on the GPU; 0.0 unless asked."""
    if not inputs.get("gpu_timing"):
        return 0.0
    # every chunk's answers were read, so its end event has completed
    torch.cuda.synchronize()
    return sum(start.elapsed_time(end) for _, start, end in spans) / 1000.0


def _chunks(spans, inputs) -> int:
    """Forward chunks the loop launched; 0 unless timing was asked for."""
    return len(spans) if inputs.get("gpu_timing") else 0


def expected_join_nodes(plan) -> tuple[PhysicalNode, ...]:
    """Return the join nodes selected from Quail's planning estimates."""
    return tuple(
        node for node in plan.nodes if isinstance(node, (AiJoin, Barrier))
    )


def expected_join_stages(plan) -> tuple:
    """Return Quail's expected join stages in execution order."""
    return tuple(
        stage
        for node in expected_join_nodes(plan)
        if isinstance(node, AiJoin)
        for stage in node.stages
    )


class QuailBackend:
    """Plan and start Quail model execution."""

    name = "quail"
    runtime_package = "vllm==0.26.0"

    def supports(self, model, device, gpu_count: int) -> SupportResult:
        if device.name not in {"h100-sxm", "rtx-pro-6000-blackwell-server"}:
            return SupportResult.reject(
                f"Quail does not support device {device.name!r}")
        if model.arch not in supported_archs():
            return SupportResult.reject(
                f"Quail does not support model {model.name!r}: "
                f"no forward pass for architecture {model.arch!r}")
        if gpu_count not in {1, 2, 4, 8}:
            return SupportResult.reject(
                "Quail requires 1, 2, 4, or 8 GPUs")
        return SupportResult.accept()

    def plan(
        self,
        region: ModelRegion,
        context: PlanningContext,
    ) -> tuple[PhysicalCandidate, ...]:

        operators = region.logical_plan.operators()
        has_score = any(
            is_score(predicate.expression)
            for predicates in operators.filters.values()
            for predicate in predicates
        ) or any(is_score(join.predicate) for join in operators.joins) or any(
            isinstance(expression, Alias)
            for expression in region.logical_plan.root.columns
        )
        if context.model.role == "reranker":
            if not has_score:
                refusal = Refusal(
                    reasons=("a reranker model can only be used with AI.SCORE",),
                    constraint="reranker_only_scores",
                    needed=1,
                    available=0,
                    unit="AI.SCORE expressions",
                )
                return (PhysicalCandidate(None, refusal, float("inf")),)
        if has_score:
            return plan_reranker(region, context, backend_name=self.name)

        plan = plan_quail(
            region.logical_plan,
            model=context.model,
            device=context.device,
            doc_tokens=context.document_tokens,
            gpus=context.gpu_count,
            order=context.order,
            pair_fractions=context.pair_fractions,
        )
        if not hasattr(plan, "graph"):
            return (
                PhysicalCandidate(
                    graph=None,
                    plan=plan,
                    estimated_seconds=float("inf"),
                ),
            )
        plan = self._bind_runtime_data(plan, region, context)
        return (
            PhysicalCandidate(
                graph=plan.graph,
                plan=plan,
                estimated_seconds=plan.estimated_seconds,
            ),
        )

    def start(self, context: GpuContext) -> QuailModelExecution:
        return QuailModelExecution(context)

    def _bind_runtime_data(self, plan, region, context):
        """Put tokenized prompts and answer tokens in the physical plan."""
        tokenizer = context.tokenizer
        operators = region.logical_plan.operators()
        filters, joins = operators.filters, operators.joins
        encoded_nodes = []
        for node in plan.nodes:
            if isinstance(node, AiFilter):
                predicates = filters[node.alias]
                questions = tuple(
                    tuple(predicates[stage.written_pos].prompt.tail_token_ids)
                    for stage in node.stages
                )
                if any(not question for question in questions):
                    raise ValueError("filter prompts have no token ids")
                node = replace(node, question_token_ids=questions)
            elif isinstance(node, AiJoin):
                stages = []
                for stage in node.stages:
                    prompt = joins[stage.written_pos].prompt
                    runtime_ids = {
                        alias: (tuple(label), tuple(frame))
                        for alias, label, frame in prompt.label_token_ids
                    }
                    stages.append(replace(
                        stage,
                        frame_token_ids=runtime_ids[stage.anchor][1],
                        label_token_ids=tuple(
                            (alias, runtime_ids[alias][0])
                            for alias in stage.partners
                        ),
                        tail_token_ids=tuple(prompt.tail_token_ids),
                    ))
                node = replace(node, stages=tuple(stages))
            encoded_nodes.append(node)

        true_ids, false_ids = (
            true_false_token_ids(tokenizer) if tokenizer is not None
            else ([], []))
        prompts = operators.prompts
        pre_ids = (
            list(tokenizer(shared_preamble(context.model.turn_prefix)))
            if tokenizer is not None else
            list(prompts[0].preamble_token_ids) if prompts else []
        )
        return replace(
            plan,
            nodes=tuple(encoded_nodes),
            root=plan.root,
            settings={
                **plan.settings,
                "true_ids": true_ids,
                "false_ids": false_ids,
                "pre_ids": pre_ids,
                "filter_limit": (
                    None if any(
                        isinstance(node, AiJoin)
                        for node in encoded_nodes
                    ) else region.logical_plan.root.limit
                ),
            },
        )

    def prepare_request(self, context) -> None:
        """Boot the GPU for a request before its documents are ready."""
        prepare_quail_request(context)

    def execute_request(self, context) -> Any:
        """Run one Quail request inside a compute process."""
        return execute_quail_request(context)
