"""Tests for the coordinator's round splitting, merging, gating, and thinning."""

import pyarrow as pa

from quail.backends.quail.coordinator import (
    filter_node_payloads,
    gate_group,
    join_group_payloads,
    merge_filter_round,
    merge_join_round,
    thin_survivors,
)
from quail.execution.types import join_answer_cells
from quail.physical import AiFilter


def payload():
    return dict(
        model="qwen3-4b-fp8", chunk_tokens=1000,
        true_ids=[1], false_ids=[2], pre_ids=[9], filter_limit=None,
        docs={"r": [[i] * (10 + i) for i in range(6)],
              "p": [[i] * 5 for i in range(4)]},
        filters={"r": [[7, 7]]}, filter_arena_writes={"r": True}, physical_plan={},
        joins=[dict(anchor="r", partners=["p"], semantics="full",
                    labels={"p": [2]}, frame=[8], tail=[3])],
        workers=2,
        shards={"r": ((0, 2, 4), (1, 3, 5)), "p": ((0, 1), (2, 3))})


def filter_node():
    return AiFilter(node_id="filter:r", alias="r", arena_writes=True, keep_kv=True,
                    question_token_ids=((7, 7),))


def test_filter_and_join_rounds_split_and_place_documents():
    # LIMIT counts output rows: with joins, cutting survivors drops rows (#39)
    p = payload()
    p["filter_limit"] = 3
    assert all(s["filter_limit"] is None for s in filter_node_payloads(
        p, filter_node(), p["shards"], 2, has_joins=True))
    subs = filter_node_payloads(p, filter_node(), p["shards"], 2, has_joins=False)
    assert all(s["filter_limit"] == 3 for s in subs)
    assert [sub["doc_index"]["r"] for sub in subs] == [[0, 2, 4], [1, 3, 5]]
    assert subs[1]["docs"]["r"][0] == [1] * 11
    assert "p" not in subs[0]["docs"]

    outs = [
        dict(filters={"r": {0: [1], 2: [0], 4: [1]}},
             survivors={"r": [0, 4]},
             fresh_tokens=100,
             boot_s=1.0, wall_s=2.0, peak_gib=10),
        dict(filters={"r": {1: [1], 3: [1], 5: [0]}},
             survivors={"r": [1, 3]},
             fresh_tokens=50,
             boot_s=1.5, wall_s=2.0, peak_gib=11),
    ]
    m = merge_filter_round(outs)
    assert m["filters"]["r"] == {0: [1], 1: [1], 2: [0], 3: [1],
                                 4: [1], 5: [0]}
    assert m["survivors"]["r"] == [0, 1, 3, 4]
    assert m["fresh_tokens"] == 150
    limited = merge_filter_round(outs, limit=3)
    assert limited["survivors"]["r"] == [0, 1, 3]

    # an anchor that ran no filter round has no KV to stay near: its scan
    # shard is ignored and shards balance over the live documents
    p = payload()
    group = [dict(anchor="p", partners=["r"], semantics="full",
                  labels={"r": [2]}, frame=[8], tail=[3])]
    survivors = {"p": [0, 1], "r": [0, 1, 3]}
    subs = join_group_payloads(p, 2, survivors, group)
    assert [s["anchor_index"] for s in subs] == [[0], [1]]

    # two full stages sharing one anchor: every worker gets both stages
    # and every partner table of either stage
    p = payload()
    p["docs"]["m"] = [[9] * 3 for _ in range(3)]
    p["joins"].append(dict(anchor="r", partners=["m"],
                           semantics="full", labels={"m": [4]},
                           frame=[8], tail=[3]))
    subs = join_group_payloads(p, 2, {"r": [0, 1, 3, 4]}, p["joins"])
    for s in subs:
        assert len(s["joins"]) == 2
        assert sorted(s["partners"]) == ["m", "p"]
        assert s["partners"]["m"]["index"] == [0, 1, 2]
        assert s["partners"]["p"]["index"] == [0, 1, 2, 3]
    assert subs[0]["anchor_index"] == [0, 4]
    assert subs[1]["anchor_index"] == [1, 3]

    # an anchor kept by an earlier group stays on the workers that
    # hold its KV, thinned to the live set, instead of re-sharding
    p = payload()
    survivors = {"r": [0, 3, 4], "p": [0, 1, 2, 3]}
    prior = {"r": [[0, 4, 5], [1, 2, 3]]}
    subs = join_group_payloads(p, 2, survivors, p["joins"],
                               prior_shards=prior)
    assert subs[0]["anchor_index"] == [0, 4]
    assert subs[1]["anchor_index"] == [3]

    p = payload()
    survivors = {"r": [0, 1, 3, 4], "p": [0, 1, 2, 3]}
    prior = {"r": [[0], [3]]}
    subs = join_group_payloads(p, 2, survivors, p["joins"],
                               prior_shards=prior)
    assert 0 in subs[0]["anchor_index"]
    assert 3 in subs[1]["anchor_index"]
    assigned = [document for sub in subs for document in sub["anchor_index"]]
    assert sorted(assigned) == survivors["r"]
    assert len(assigned) == len(set(assigned))


def test_join_merge_gates_thins_and_ships_partner_lists():
    out = dict(rows={0: [1, 0], 1: [0, 0], 2: [0, 1]},
               anchor_index=[5, 7, 9])
    assert gate_group(out, "full") == [5, 9]
    assert gate_group(out, "exists") == [5, 9]
    assert gate_group(out, "anti") == [7]
    # an anchor gated out mid-group has no last-stage row: dropped
    partial = dict(rows={0: [1]}, anchor_index=[5, 7])
    assert gate_group(partial, "full") == [5]

    # two finished stages touching p: a p document must appear in a
    # surviving pair of BOTH to stay live
    s1 = dict(anchor="r", partners=["p"], rows={0: [1, 1, 0, 0]},
              anchor_index=[0], partner_index=[[0], [1], [2], [3]])
    s2 = dict(s1, anchor="g", rows={0: [0, 1, 1, 0]})
    survivors = {"r": [0], "g": [0], "p": [0, 1, 2, 3]}
    thin_survivors([s1, s2], survivors)
    assert survivors["p"] == [1]

    # the merge keeps stages aligned though stage 2 skips gated anchors
    outs = [
        dict(joins=[dict(rows={0: [1, 0], 1: [0, 0]}, anchor_index=[0, 4],
                         partner_index=[[1], [2]]),
                    dict(rows={0: [0, 1, 1]}, anchor_index=[0, 4],
                         partner_index=[[0], [1], [2]])],
             fresh_tokens=10, wall_s=1.0),
        dict(joins=[dict(rows={0: [1, 1]}, anchor_index=[3],
                         partner_index=[[1], [2]]),
                    dict(rows={0: [1, 0, 0]}, anchor_index=[3],
                         partner_index=[[0], [1], [2]])],
             fresh_tokens=10, wall_s=1.0),
    ]
    merged = merge_join_round(outs)
    assert len(merged) == 2
    assert merged[0]["anchor_index"] == [0, 4, 3]
    assert merged[0]["rows"] == {0: [1, 0], 1: [0, 0], 2: [1, 1]}
    assert merged[1]["anchor_index"] == [0, 4, 3]
    assert merged[1]["rows"] == {0: [0, 1, 1], 2: [1, 0, 0]}
    assert merged[1]["partner_index"] == [[0], [1], [2]]

    outs = [
        dict(joins=[dict(rows={0: [1], 1: [0, 1]}, anchor_index=[0, 4],
                         partner_index=[[1], [2], [3]],
                         anchor_partners={0: [2], 1: [0, 1]})],
             fresh_tokens=1, wall_s=1.0),
        dict(joins=[dict(rows={0: [1, 0]}, anchor_index=[3],
                         partner_index=[[1], [2], [3]],
                         anchor_partners={0: [1, 2]})],
             fresh_tokens=1, wall_s=1.0),
    ]
    merged = merge_join_round(outs)[0]
    assert merged["anchor_partners"] == {0: [2], 1: [0, 1], 2: [1, 2]}
    assert list(join_answer_cells(merged)) == [
        (0, 2, 1), (1, 0, 0), (1, 1, 1), (2, 1, 1), (2, 2, 0)]
    # thinning reads the member list: anchor 0 matched partner 3
    # (member 2), anchor 4 matched partner 2, anchor 3 matched partner 2
    merged.update(anchor="r", partners=["p"])
    survivors = {"r": [0, 3, 4], "p": [1, 2, 3]}
    thin_survivors([merged], survivors)
    assert survivors == {"r": [0, 3, 4], "p": [2, 3]}

    # a pair stage ships each worker's anchors with their live partners only
    pairs = pa.table({"r": pa.array([0, 0, 2, 3, 5], pa.int32()),
                      "p": pa.array([0, 3, 1, 2, 3], pa.int32())})
    p = payload()
    group = [dict(p["joins"][0], written_pos=0, pairs_from="hash_join:r-p")]
    subs = join_group_payloads(p, 2, {"r": [0, 2, 3, 5], "p": [1, 3]},
                               group, pair_tables={0: pairs})
    assert [sub["anchor_index"] for sub in subs] == [[0, 2], [3, 5]]
    assert [sub["pairs"] for sub in subs] == [
        {0: {0: [3], 2: [1]}}, {0: {3: [], 5: [3]}}]
