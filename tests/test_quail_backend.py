"""Quail backend execution tests without a GPU."""

from types import SimpleNamespace

import pyarrow as pa
import pytest
from fakes import (
    fake_torch,
    keep_even,
    run_graph_on_arena,
    same_key,
    two_alias_graph,
)

from quail.backends.quail import QuailModelExecution
from quail.backends.quail.distributed import execute_distributed_graph
from quail.builtins import built_in_registry
from quail.physical import (
    AiFilter,
    FilterStage,
    PhysicalGraph,
    PortRef,
    Scan,
)
from quail.physical.base import input_ports
from quail.specs import DEVICES, MODELS


def test_filter_execution_and_retention_inputs(monkeypatch):
    from quail.physical import plan_envelope

    filtered = AiFilter(
        node_id="filter:d", inputs=input_ports((PortRef("input:d", "ids:d"),)),
        alias="d", stages=(FilterStage(0, 1, 1, None, 4),),
        question_token_ids=((9,),))
    graph = PhysicalGraph((Scan(node_id="input:d", alias="d", input_id="d"),
                           filtered), PortRef("filter:d", "ids:d"))

    def round_fn(kind, subs):
        assert kind == "filters"
        outputs = []
        for sub in subs:
            indices = sub["doc_index"]["d"]
            outputs.append({
                "filters": {"d": {index: [index % 2 == 0] for index in indices}},
                "survivors": {"d": [index for index in indices if index % 2 == 0]},
                "retained": {}, "fresh_tokens": len(indices), "boot_s": 0.0,
                "boot_kind": "warm", "boot": {}, "peak_gib": 1.0,
            })
        return outputs

    registry = built_in_registry()
    payload = {
        "model": "qwen3-4b-fp8", "chunk_tokens": 8192, "true_ids": [1],
        "false_ids": [2], "pre_ids": [], "filter_limit": None,
        "order_rule": "by_cost", "docs": {"d": [[1], [2], [3], [4]]},
        "joins": [], "filters": {}, "filter_arena_writes": {},
        "shards": {"d": ((0, 2), (1, 3))},
        "physical_plan": plan_envelope(
            backend="quail", model="qwen3-4b-fp8", device="h100-sxm",
            workers=2, graph=graph, codecs=registry.codecs),
    }
    result = execute_distributed_graph(
        payload, graph, 2, round_fn, MODELS["qwen3-4b-fp8"],
        DEVICES["h100-sxm"], registry.runtimes, registry)
    assert result["filters"]["d"] == {0: [True], 1: [False], 2: [True], 3: [False]}
    assert result["fresh_tokens"] == 4

    received = {}

    def fake_run_filter(*args, retain_survivors, **kwargs):
        received["retain_survivors"] = retain_survivors
        return {0: [True]}, [], 3

    monkeypatch.setattr("quail.backends.quail.executor.loop.run_filter",
                        fake_run_filter)
    execution = QuailModelExecution(SimpleNamespace())
    execution.bind_loaded_model(
        model=object(), arena=SimpleNamespace(), pipeline=SimpleNamespace())
    execution.bind_query(torch=fake_torch(), async_answers=object(),
                         answer_rows=object(), chunk_tokens=8192)
    result = execution.execute(filtered, {
        "documents": [[1]], "document_ids": [10], "retain_survivors": False})
    assert received["retain_survivors"] == ()
    assert result.outputs["ids:d"] == [10]


class FakeArena:
    n_pages = 100
    page_tokens = 16
    retention_cap_pages = None
    evicted_keys = evicted_pages = evicted_prefix_tokens = 0

    def __init__(self):
        self.accounting = self
        self.owned = set()
        self.retained = set()

    @property
    def retained_pages(self):
        return len(self.retained)

    @property
    def retained_prefix_tokens(self):
        return 16 * len(self.retained)

    def configure_retention(self, policy, cap_pages):
        self.retention_policy = policy
        self.retention_cap_pages = cap_pages

    def is_resident(self, key):
        return key in self.owned

    def resident_keys(self):
        return list(self.owned)

    def retained_keys(self):
        return list(self.retained)

    def free_key(self, key):
        self.owned.discard(key)
        self.retained.discard(key)

    def retain(self, key, prefix_tokens, priority=None):
        self.owned.add(key)
        self.retained.add(key)

    def evict_retained(self, pages):
        return ()

    def reset_stats(self):
        pass


def graph_state(model_execution, docs):
    return {
        "torch": fake_torch(),
        "arena": FakeArena(),
        "pipeline": SimpleNamespace(),
        "model_execution": model_execution,
        "runtimes": built_in_registry().runtimes,
        "model_spec": MODELS["qwen3-4b-fp8"],
        "device": DEVICES["h100-sxm"],
        "chunk_tokens": 8192,
        "docs": docs,
    }


def _key_columns(r_keys, p_keys):
    return {alias: pa.table({alias: pa.array(range(len(keys)), pa.int32()),
                             "key": pa.array(keys)})
            for alias, keys in (("r", r_keys), ("p", p_keys))}


def _pairs(table):
    return sorted(zip(table.column("r").to_pylist(), table.column("p").to_pylist()))


def _ids(tables):
    (alias,) = tables
    return tables[alias].column(alias).to_pylist()


def test_pair_join_and_gpu_timing_run_through_the_quail_graph(monkeypatch):
    # document d has key d % 4; partners 0 and 3 share key 0, so a
    # document with key 0 pairs with both and one with key 3 with none
    columns = _key_columns([d % 4 for d in range(14)], [0, 1, 2, 0])
    allowed = {d: [i for i in range(4) if d % 4 == i % 3] for d in range(14)}
    for pin_survivors in (True, False):
        result, _, filter_truth, join_truth = run_graph_on_arena(
            monkeypatch, two_alias_graph(pin_survivors, hash_join=True),
            columns=columns)
        cross, _, _, _ = run_graph_on_arena(
            monkeypatch, two_alias_graph(pin_survivors))
        survivors = [d for d, truth in enumerate(filter_truth) if all(truth)]
        stage = result["joins"][0]
        assert sorted(stage["anchor_index"]) == survivors
        members = stage["anchor_partners"]
        for local, document in enumerate(stage["anchor_index"]):
            mine = sorted(allowed[document])
            assert members[local] == mine
            assert stage["rows"].get(local, []) == [
                join_truth[("r", document)][i] for i in mine]
        table = result["_outputs"][PortRef("group:0", "join_answers:0")]
        assert _pairs(table) == sorted(
            (d, i) for d in survivors for i in allowed[d])
        assert table.column("answer").to_pylist() == [
            bool(join_truth[("r", d)][i])
            for d, i in zip(table.column("r").to_pylist(),
                            table.column("p").to_pylist())]
        metrics = result["node_metrics"]["group:0"]
        assert metrics["evaluated_document_pairs"] == sum(
            len(allowed[d]) for d in survivors)
        assert metrics["fresh_tokens"] < (
            cross["node_metrics"]["group:0"]["fresh_tokens"])
        assert result["node_metrics"]["hash_join:r-p"]["output_rows"] == sum(
            len(mine) for mine in allowed.values())
        # anchors whose pairs all answered FALSE are gone
        kept = [d for d in survivors
                if any(join_truth[("r", d)][i] for i in allowed[d])]
        root = result["_outputs"][PortRef("group:0", "ids:r")]
        assert sorted(root.column("r").to_pylist()) == kept

    # GPU timing sums the chunk events only when asked
    off, _, _, _ = run_graph_on_arena(
        monkeypatch, two_alias_graph(True), pages=48)
    assert "gpu_s" not in off
    assert all(metrics["gpu_s"] == 0.0
               for metrics in off["node_metrics"].values())
    on, model, _, _ = run_graph_on_arena(
        monkeypatch, two_alias_graph(True), pages=48, gpu_timing=True)
    # join chunks run before the streamed filter chain finishes
    kinds = [kind for kind, _ in model.launched]
    assert kinds.index("join") < len(kinds) - 1 - kinds[::-1].index("filter")
    # the fake torch reports 2 ms per event pair, one pair per chunk
    per_node = [metrics["gpu_s"] for metrics in on["node_metrics"].values()]
    assert on["gpu_s"] == pytest.approx(sum(per_node), abs=1e-3)
    assert on["gpu_s"] > 0
    assert on["chunks"] == round(on["gpu_s"] / 0.002)
    assert all(chunks == round(chunks)
               for chunks in (seconds / 0.002 for seconds in per_node))


def _foreign_run(monkeypatch, graph, functions):
    # document d has key d % 4; partner i has key i
    result, _, filter_truth, join_truth = run_graph_on_arena(
        monkeypatch, graph, seed=9, functions=functions,
        columns=_key_columns([d % 4 for d in range(14)], list(range(4))))
    survivors = [d for d, truth in enumerate(filter_truth) if all(truth)]
    return result, survivors, join_truth


def test_foreign_runs_per_batch_or_once_as_a_barrier_and_are_checked(monkeypatch):
    functions = {"keep_even": keep_even, "same_key": same_key}
    # a per-batch drop on the pinned chain: only even survivors reach the join
    result, survivors, join_truth = _foreign_run(
        monkeypatch, two_alias_graph(True, foreign=("per_batch", "drop")),
        functions)
    assert sorted(result["filters"]["r"]) == list(range(14))
    stage = result["joins"][0]
    kept = [d for d in survivors if d % 2 == 0]
    assert sorted(stage["anchor_index"]) == kept
    foreign = result["node_metrics"]["apply:keep_even"]
    assert foreign["input_rows"] == len(survivors)
    assert foreign["output_rows"] == len(kept)
    assert result["_outputs"][PortRef("apply:keep_even", "ids:r")].column(
        "r").to_pylist() == kept
    assert result["node_metrics"]["group:0"]["kv_hits"] == len(kept)

    result, survivors, _ = _foreign_run(
        monkeypatch, two_alias_graph(False, foreign=("barrier", "drop")),
        functions)
    assert sorted(result["joins"][0]["anchor_index"]) == kept

    # per-batch and barrier pair functions both pair document d with d % 4
    per_batch, survivors, join_truth = _foreign_run(
        monkeypatch, two_alias_graph(True, foreign=("per_batch", "pairs")),
        functions)
    barrier, _, _ = _foreign_run(
        monkeypatch, two_alias_graph(False, foreign=("barrier", "pairs")),
        functions)
    for result in (per_batch, barrier):
        table = result["_outputs"][PortRef("group:0", "join_answers:0")]
        assert _pairs(table) == [(d, d % 4) for d in survivors]
        assert {(r, p): a for r, p, a in zip(
            table.column("r").to_pylist(), table.column("p").to_pylist(),
            table.column("answer").to_pylist())} == {
            (d, d % 4): bool(join_truth[("r", d)][d % 4]) for d in survivors}
        pairs = result["_outputs"][PortRef("apply:same_key", "pairs:0")]
        assert _pairs(pairs) == [(d, d % 4) for d in survivors]
        assert result["node_metrics"]["apply:same_key"]["output_rows"] == len(
            survivors)

    for kind, ids, function, message in [
        ("per_batch", "drop", lambda tables: [99], "never invents an id"),
        ("barrier", "preserve", lambda tables: _ids(tables)[1:],
         "preserves ids but dropped"),
        ("barrier", "drop", lambda tables: [None] + _ids(tables)[1:],
         "returned a null id"),
    ]:
        graph = two_alias_graph(kind == "per_batch", foreign=(kind, ids))
        with pytest.raises(ValueError, match=message):
            _foreign_run(monkeypatch, graph, {"keep_even": function})
