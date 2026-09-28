"""CPU fakes the executor tests share: a page arena, a planted model, graphs.

Token ids encode what a suffix is: a document packs [DOC + d] * length,
filter stage s asks [QUESTION + s], partner i streams [PARTNER + i] *
length, and the join frame is [FRAME] * length.
"""

import random
from contextlib import nullcontext
from itertools import takewhile
from types import SimpleNamespace

import pyarrow as pa

import quail
from quail.backends.quail import QuailModelExecution
from quail.backends.quail.executor import loop
from quail.backends.quail.executor.arena import KVArena, PageArena
from quail.backends.quail.executor.models.base import ModelPipeline
from quail.backends.quail.graph import execute_single_graph
from quail.builtins import built_in_registry
from quail.physical import (
    AiFilter,
    AiJoin,
    FilterStage,
    Foreign,
    HashJoin,
    JoinStage,
    PhysicalGraph,
    PortRef,
    Scan,
)
from quail.physical.base import input_ports
from quail.specs import DEVICES, MODELS

DOC = 1000
QUESTION = 2000
PARTNER = 3000
FRAME = 4000
SETTINGS = {"filter_limit": None, "pre_ids": [], "retention": {}}


def bare_arena(arena, pages):
    """Give a tensor-free KVArena the accounting of a single-pool arena."""
    arena.accounting = PageArena(pages, 16)
    arena.sliding = None
    arena.window = 0
    arena.sliding_layers = frozenset()
    arena.pinned = False
    arena._rows = {}
    arena._capacity_rows = {}
    arena._sliding_rows = {}
    arena._base = {}
    arena._sliding_start = {}
    arena._holds = {}
    arena._deferred = set()
    arena._window_floor = {}
    arena._trimmed = set()
    arena._refresh_rows = lambda *args: None
    arena.reset_stats()
    return arena


def cpu_staging(monkeypatch):
    """Stage packed chunks as plain CPU tensors; returns torch."""
    import numpy as np
    import torch

    def staged(torch_, data, dtype, pinned=True):
        if isinstance(data, np.ndarray) or torch.is_tensor(data):
            return torch.as_tensor(data, dtype=dtype)
        return torch.tensor(data, dtype=dtype)

    def token_parts(torch_, sequences, total, pinned=True, staging=None):
        ids = [int(t) for seq in sequences for part in loop._token_parts(seq)
               for t in part]
        assert len(ids) == total
        return torch.tensor(ids, dtype=torch.int64)

    monkeypatch.setattr(loop, "_staged", staged)
    monkeypatch.setattr(loop, "_staged_token_parts", token_parts)
    return torch


def fake_pipeline(**attributes):
    """A ModelPipeline with the contract's defaults and the given overrides."""
    pipeline = ModelPipeline()
    for name, value in attributes.items():
        setattr(pipeline, name, value)
    return pipeline


def cpu_arena(pages):
    arena = bare_arena(KVArena.__new__(KVArena), pages)

    def allocate(key, tokens, capacity_tokens=None, base_tokens=None,
                 sliding_tokens=None, borrow=None):
        borrowed = (arena.accounting.shared_pages(*borrow) if borrow else ())
        got = arena.accounting.alloc(key, tokens, capacity_tokens,
                                     borrowed=borrowed)
        if got is not None:
            arena._rows[key] = None
            arena._capacity_rows[key] = None
            arena._base[key] = tokens if base_tokens is None else base_tokens
            arena._sliding_start[key] = 0
        return got

    arena.alloc = allocate
    return arena


class FakeModel:
    """Answer packed suffixes from planted truth tables."""

    def __init__(self, filter_truth, join_truth):
        self.filter_truth = filter_truth
        self.join_truth = join_truth
        self.launched = []      # ("filter" | "join", specs)

    def forward_chunk(self, chunk):
        bits = []
        kind = None
        for spec in chunk.specs:
            for suffix in spec["suffixes"]:
                head = suffix[0]
                if head >= FRAME:
                    bits.append(0)
                    kind = "join"
                elif head >= PARTNER:
                    bits.append(self.join_truth[spec["key"]][head - PARTNER])
                    kind = "join"
                else:
                    document = spec["key"][1]
                    bits.append(self.filter_truth[document][head - QUESTION])
                    kind = "filter"
            if kind == "join" and spec["prefix"] is not None:
                raise AssertionError("a streamed anchor packed its prefix")
        self.launched.append((kind, chunk.specs))
        return bits


def fake_torch():
    return SimpleNamespace(
        inference_mode=nullcontext,
        cuda=SimpleNamespace(
            Event=lambda **kw: SimpleNamespace(
                record=lambda: None, elapsed_time=lambda other: 2.0),
            synchronize=lambda: None,
            max_memory_allocated=lambda: 0))


def fake_pack(torch, arena, specs, **kw):
    tokens = sum(
        (len(spec["prefix"]) if spec["prefix"] is not None else 0)
        + sum(len(suffix) for suffix in spec["suffixes"])
        for spec in specs)
    return SimpleNamespace(specs=specs, tokens=tokens, temporary_keys=(),
                           fresh_keys=())


def expected_filter_rows(filter_truth):
    """Each document's answers up to and including its first FALSE."""
    rows = {}
    for d, truth in enumerate(filter_truth):
        passed = list(takewhile(bool, truth))
        rows[d] = passed + truth[len(passed):len(passed) + 1]
    return rows


def run_streamed(monkeypatch, *, doc_lengths, filter_truth, partner_lengths,
                 join_truth, budget, pages, frame_tokens=3, stages=2,
                 anchor_partners=None):
    """Drive a filter chain streamed into a join on a CPU arena."""
    monkeypatch.setattr(loop, "pack_chunk", fake_pack)
    model = FakeModel(filter_truth, join_truth)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    answers = SimpleNamespace(submit=lambda v: v, result=lambda v: v,
                              dtype=None)
    arena = cpu_arena(pages)
    docs = [[DOC + d] * n for d, n in enumerate(doc_lengths)]
    questions = [[QUESTION + s] for s in range(stages)]
    keys = [("r", d) for d in range(len(docs))]
    frame = [FRAME] * frame_tokens
    suffixes = [[PARTNER + i] * n for i, n in enumerate(partner_lengths)]
    stream = loop.FilterStream(
        fake_torch(), arena, pipeline, answers, docs, questions, budget,
        arena_writes=True, arena_keys=keys, hold_survivors=True,
        hold_extra_tokens=len(frame))
    blocked_seen = []
    real_next = stream.next

    def counting_next(evict_retained=False):
        items, blocked = real_next(evict_retained=evict_retained)
        blocked_seen.append(blocked)
        return items, blocked

    stream.next = counting_next
    anchor_keys, anchor_prefixes = [], []
    settled = {}

    def anchor_done(local, row):
        settled[anchor_keys[local]] = list(row)
        arena.free_key(anchor_keys[local])

    join_answers, _, join_tokens = loop.run_join(
        fake_torch(), arena, pipeline, answers, anchor_prefixes,
        [suffixes], budget, stage_frames=[frame], anchor_keys=anchor_keys,
        anchor_done=anchor_done, anchor_source=stream,
        anchor_partners=anchor_partners)
    return dict(stream=stream, model=model, arena=arena,
                join_answers=join_answers, join_tokens=join_tokens,
                anchor_keys=anchor_keys, settled=settled,
                blocked=blocked_seen)


def two_alias_graph(pin_survivors, *, stages=1, hash_join=False, foreign=None):
    """R's filter chain into a join with p.

    Args:
        pin_survivors: Whether the chain streams into the join.
        stages: Filter stages on r.
        hash_join: Pair r and p on their "key" columns with a HashJoin
            over the scans that feeds the join its pairs port.
        foreign: (kind, ids) of a Foreign node between the chain and
            the join; ids "pairs" feeds the join its pairs port.
    """
    chain = AiFilter(
        node_id="filter:r",
        inputs=input_ports((PortRef("input:r", "ids:r"),)),
        alias="r", arena_writes=True, pin_survivors=pin_survivors,
        keep_kv=not pin_survivors, hold_tokens=1 if pin_survivors else 0,
        stages=tuple(FilterStage(s, 1, 0, 0.8 - 0.1 * s, 14 * 0.8 ** s)
                     for s in range(stages)),
        question_token_ids=tuple((QUESTION + s,) for s in range(stages)))
    nodes = [Scan(node_id="input:r", alias="r", input_id="r"), chain,
             Scan(node_id="input:p", alias="p", input_id="p")]
    anchor_src = PortRef("filter:r", "ids:r")
    join_inputs = []
    pairs_from = ""
    if hash_join:
        nodes.append(HashJoin(
            node_id="hash_join:r-p",
            inputs=input_ports((PortRef("input:r", "ids:r"),
                                PortRef("input:p", "ids:p"))),
            left="r", right="p", on=(("key", "key"),), written_pos=0))
        join_inputs.append(PortRef("hash_join:r-p", "pairs:0"))
        pairs_from = "hash_join:r-p"
    if foreign and foreign[1] == "pairs":
        nodes.append(Foreign(
            node_id="apply:same_key",
            inputs=input_ports((anchor_src, PortRef("input:p", "ids:p"))),
            function="same_key", kind=foreign[0], ids="pairs",
            columns=(("r", "key"), ("p", "key")), aliases=("r", "p"),
            written_pos=0))
        join_inputs.append(PortRef("apply:same_key", "pairs:0"))
        pairs_from = "apply:same_key"
    elif foreign:
        nodes.append(Foreign(
            node_id="apply:keep_even",
            inputs=input_ports((anchor_src,)),
            function="keep_even", kind=foreign[0], ids=foreign[1],
            columns=(), aliases=("r",)))
        anchor_src = PortRef("apply:keep_even", "ids:r")
    resident = "filter" if pin_survivors else "none"
    nodes.append(AiJoin(
        node_id="group:0", anchor="r", anchor_resident=resident,
        inputs=input_ports((anchor_src, PortRef("input:p", "ids:p"),
                            *join_inputs)),
        stages=(JoinStage(
            written_pos=0, exec_idx=0, anchor="r", partners=("p",),
            semantics="full", selectivity=0.5, expected_tuples=1,
            anchor_frame_tokens=1, pair_tail_tokens=0,
            anchor_resident=resident, tuple_tokens=0,
            pairs_from=pairs_from,
            frame_token_ids=(FRAME,), label_token_ids=(("p", ()),),
            tail_token_ids=()),)))
    return PhysicalGraph(tuple(nodes), PortRef("group:0", "ids:r"))


def run_graph_on_arena(monkeypatch, graph, *, n_docs=14, n_partners=4,
                       seed=5, pages=64, stages=1, partner_tokens=20,
                       **extra):
    """Run a graph over random documents through the real graph runtime.

    Returns:
        (result, model, filter_truth, join_truth); the arena is empty
        afterwards.
    """
    monkeypatch.setattr(loop, "pack_chunk", fake_pack)
    rng = random.Random(seed)
    docs = {
        "r": [[DOC + d] * rng.randrange(10, 40) for d in range(n_docs)],
        "p": [[PARTNER + i] * partner_tokens for i in range(n_partners)],
    }
    filter_truth = [[1 if rng.random() < p else 0
                     for p in (0.8, 0.7)[:stages]] for _ in range(n_docs)]
    join_truth = {("r", d): [1 if rng.random() < 0.5 else 0
                             for _ in range(n_partners)] for d in range(n_docs)}
    model = FakeModel(filter_truth, join_truth)
    torch = fake_torch()
    arena = cpu_arena(pages)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace())
    execution.bind_loaded_model(model=object(), arena=arena,
                                pipeline=pipeline)
    execution.bind_query(
        torch=torch,
        async_answers=SimpleNamespace(submit=lambda v: v, result=lambda v: v,
                                      dtype=None),
        answer_rows=object(),
        chunk_tokens=120)
    state = {
        "torch": torch, "arena": arena, "pipeline": pipeline,
        "model_execution": execution,
        "runtimes": built_in_registry().runtimes,
        "model_spec": MODELS["qwen3-4b-fp8"], "device": DEVICES["h100-sxm"],
        "chunk_tokens": 120, "docs": docs, **extra,
    }
    result = execute_single_graph(state, SETTINGS, graph)
    assert not arena.accounting.owned
    return result, model, filter_truth, join_truth


def register_claims_evidence(session, claim_words=20, text_words=100):
    """Four claims and three evidence rows sharing a url column."""
    session.register("claims", quail.DocumentProvider.from_table(pa.table({
        "id": [f"c{i}" for i in range(4)],
        "claim": [f"{i} " + "word " * claim_words for i in range(4)],
        "url": ["u0", "u1", "u1", "u9"],
    }), id_col="id"))
    session.register("evidence", quail.DocumentProvider.from_table(pa.table({
        "id": [f"e{i}" for i in range(3)],
        "text": [f"{i} " + "word " * text_words for i in range(3)],
        "url": ["u1", "u0", "u1"],
    }), id_col="id"))


def keep_even(tables):
    (alias,) = tables
    table = tables[alias]
    return [document for document in table.column(alias).to_pylist()
            if document % 2 == 0]


def same_key(tables):
    left, right = tables["r"], tables["p"]
    return left.join(right, keys=["key"], join_type="inner").select(["r", "p"])
