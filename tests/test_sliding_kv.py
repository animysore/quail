"""The sliding-layer KV pool: window origins, trimming, and one page currency."""

import pytest
from fakes import cpu_staging

from quail.backends.quail.executor import loop
from quail.backends.quail.executor.arena import KVArena

torch = pytest.importorskip("torch")

WINDOW = 32     # tokens, two pages


def cpu_arena(pages=64, sliding_pages=16):
    # layer 0 keeps every token, layer 1 slides
    return KVArena(n_layers=2, n_pages=pages, page_tokens=16, n_kv=1, d_head=2,
                   dtype=torch.float32, device="cpu",
                   layer_kv=[(1, 2), (1, 2)], sliding_layers=(1,),
                   sliding_window=WINDOW, n_sliding_pages=sliding_pages)


def plain_arena(pages=64):
    return KVArena(n_layers=1, n_pages=pages, page_tokens=16, n_kv=1, d_head=2,
                   dtype=torch.float32, device="cpu")


def test_window_origin_and_trim():
    arena = cpu_arena()
    assert arena.has_sliding
    assert arena.origin(20) == 0
    assert arena.origin(WINDOW) == 0
    assert arena.origin(100) == 64      # 100 - 32 = 68, down to a page
    key = ("d", 0)
    # a fresh 100-token document with 20 rows of room takes 8 pages per pool
    assert arena.activate(key, 100, capacity_tokens=120, base_tokens=100)
    assert len(arena.owned_pages(key)) == 8
    assert len(arena.owned_sliding_pages(key)) == 8
    assert arena.sliding_start(key) == 0
    assert arena.free_pages == min(64 - 8, int((16 - 8) * 4))
    freed = arena.trim_window(key)
    # pages before row 64 leave the sliding pool: 4 of them
    assert len(arena.owned_sliding_pages(key)) == 4
    assert arena.sliding_start(key) == 64
    assert freed == arena.free_pages - min(64 - 8, (16 - 8) * 4)
    assert arena.trim_window(key) == 0
    assert arena.capacity_rows_sliding(key).numel() == 4 * 16
    assert arena.capacity_rows(key).numel() == 8 * 16
    assert arena.activate(key, 110, capacity_tokens=140) is not None
    assert len(arena.owned_pages(key)) == 9
    assert len(arena.owned_sliding_pages(key)) == 5     # (140 - 64) / 16
    # retaining below the base would drop window rows
    with pytest.raises(ValueError, match="window"):
        arena.retain(key, 90)
    arena.retain(key, 100)
    assert len(arena.owned_pages(key)) == 7
    assert len(arena.owned_sliding_pages(key)) == 3     # (100 - 64) / 16
    arena.free_key(key)
    assert arena.free_pages == 64


def test_page_costs_allocation_temporaries_and_resize():
    arena = cpu_arena(pages=64, sliding_pages=16)     # ratio 4
    # a short document is priced by the sliding pool: 2 pages x 4
    assert arena.page_cost(20) == 8
    assert arena.page_cost(20, 20) == 8
    # a long document untrimmed: 7 pages x 4; trimmed at origin 64:
    # (100 - 64) rows = 3 sliding pages x 4 = 12, above its 7 full pages
    assert arena.page_cost(100) == 28
    assert arena.page_cost(100, 100) == 12
    key = ("d", 1)
    arena.activate(key, 100, base_tokens=100)
    assert arena.held_cost(key) == 28
    arena.trim_window(key)
    assert arena.held_cost(key) == 12
    plain = plain_arena(pages=8)
    assert not plain.has_sliding
    assert plain.page_cost(100) == 7 and plain.page_cost(100, 100) == 7
    # without a sliding pool a trim releases nothing
    plain.activate(key, 100, base_tokens=100)
    plain.trim_window(key)
    assert plain.held_cost(key) == 7

    # 4 sliding pages hold 64 rows; a 100-row key needs 7 in each pool
    arena = cpu_arena(pages=64, sliding_pages=4)
    free = arena.accounting.free_pages
    assert arena.alloc(("d", 0), 100) is None
    assert arena.accounting.free_pages == free
    assert not arena.accounting.owned and not arena.sliding.owned
    assert arena.alloc(("d", 1), 40) is not None

    arena = cpu_arena(pages=64, sliding_pages=16)
    temp, pages = arena.alloc_temporary(40, sliding_tokens=20)
    assert len(pages) == 3
    assert len(arena.owned_sliding_pages(temp)) == 2
    arena.free_key(temp)
    with pytest.raises(RuntimeError, match="holds keys"):
        arena.activate(("d", 2), 10)
        arena.resize(32, 8)
    arena.free_key(("d", 2))
    arena.resize(32, 8)
    assert arena.n_pages == 32 and arena.n_sliding_pages == 8
    assert arena.k[1].shape[0] == 8 * 16 and arena.k[0].shape[0] == 32 * 16
    arena.resize(32, 8)     # a no-op at the same sizes


def test_holds_and_the_window_floor():
    arena = cpu_arena(pages=64, sliding_pages=32)
    parent, child = ("d", 0), ("d", 1)
    arena.activate(parent, 100, capacity_tokens=110, base_tokens=100)
    arena.hold(parent, 1)
    # a borrower sharing 48 tokens reads the window from row 16 on
    arena.keep_window(parent, 48)
    arena.trim_window(parent)
    assert arena.sliding_start(parent) == 16
    # freed while held: the pages stay, and retention cannot evict it
    assert arena.free_key(parent) == 0
    assert arena.is_resident(parent) and arena.can_borrow(parent, 48)
    arena.activate(child, 60, capacity_tokens=70, base_tokens=60,
                   borrow=(parent, 48))
    arena.release(parent)
    assert not arena.is_resident(parent)
    arena.free_key(child)
    assert arena.free_pages == 64

    # a retained held key is passed over by eviction until its holds go
    arena.activate(parent, 100, capacity_tokens=110, base_tokens=100)
    arena.hold(parent, 2)
    arena.retain(parent, 100)
    assert arena.evict_retained(64) == ()
    assert arena.drop_holds([parent])
    assert arena.evict_retained(64) == (parent,)
    assert arena.free_pages == 64

    arena.activate(parent, 100, capacity_tokens=110, base_tokens=100)
    arena.hold(parent, 1)
    arena.keep_window(parent, 48)
    arena.trim_window(parent)
    assert arena.sliding_start(parent) == 16
    arena.activate(child, 60, capacity_tokens=70, base_tokens=60,
                   borrow=(parent, 48))
    borrowed = arena.owned_sliding_pages(child)[:2]
    # the last borrower is in: the parent keeps only its own window
    arena.release(parent)
    assert arena.is_resident(parent)
    assert arena.sliding_start(parent) == 64
    assert arena.owned_sliding_pages(child)[:2] == borrowed
    arena.free_key(child)
    arena.free_key(parent)
    assert arena.free_pages == 64

    # a key still to run its pass trims after it, without the floor
    arena.activate(parent, 100, capacity_tokens=110, base_tokens=100)
    arena.hold(parent, 2)
    arena.keep_window(parent, 48)
    assert arena.drop_holds([parent])
    assert arena.sliding_start(parent) == 0
    arena.trim_window(parent)
    assert arena.sliding_start(parent) == 64
    arena.free_key(parent)
    assert arena.free_pages == 64


def test_pack_chunk_builds_both_pools_and_reads_borrowed_pages(monkeypatch):
    cpu_staging(monkeypatch)
    arena = cpu_arena(pages=64, sliding_pages=32)
    key = ("d", 0)
    doc = list(range(100))
    tail = [500, 501]
    canvas = (900, 901)
    arena.activate(key, 100, capacity_tokens=120, base_tokens=100)
    chunk = loop.pack_chunk(
        torch, arena, [dict(key=key, prefix=doc, f=100, suffixes=[tail])],
        attention_mode="unified", canvas=canvas)
    assert chunk.fresh_keys == (key,)
    full = chunk.meta["unified"]
    sliding = full["sliding"]
    assert full["src"].tolist() == list(range(104))
    assert sliding["src"].tolist() == list(range(104))
    assert full["used"].tolist() == [104] and sliding["used"].tolist() == [104]
    assert full["table"].shape == (1, 8) and sliding["table"].shape == (1, 8)
    assert chunk.meta["canvas"]["sliding"]["used"].tolist() == [104]
    arena.trim_window(key)
    # a later stage reads the kept rows: the sliding pool from row 64
    chunk = loop.pack_chunk(
        torch, arena, [dict(key=key, prefix=None, f=100, suffixes=[[600]])],
        attention_mode="unified", canvas=canvas)
    assert chunk.fresh_keys == ()
    full = chunk.meta["unified"]
    sliding = full["sliding"]
    assert full["used"].tolist() == [103]
    assert sliding["used"].tolist() == [103 - 64]
    assert sliding["table"].shape[1] == 4
    # the suffix rows go to the key's own pages after row 100 in the
    # every-token pool and after row 36 in the sliding pool
    assert full["dst"].tolist() == arena.capacity_rows(key)[100:103].tolist()
    assert sliding["dst"].tolist() == \
        arena.capacity_rows_sliding(key)[36:39].tolist()
    assert chunk.meta["canvas"]["sliding"]["used"].tolist() == [39]
    # two suffixes take temporaries after a copy of each partial last page
    chunk = loop.pack_chunk(
        torch, arena, [dict(key=key, prefix=None, f=100,
                            suffixes=[[600], [601, 602]])],
        attention_mode="unified")
    full = chunk.meta["unified"]
    sliding = full["sliding"]
    assert len(chunk.temporary_keys) == 2
    assert full["used"].tolist() == [101, 102]
    assert sliding["used"].tolist() == [37, 38]
    assert full["tail_src"].numel() == 2 * (100 % 16)
    assert sliding["tail_src"].numel() == 2 * (36 % 16)
    for temp in chunk.temporary_keys:
        arena.free_key(temp)
    arena.free_key(key)
    assert arena.free_pages == 64

    arena = plain_arena()
    parent, child = ("d", 0), ("d", 1)
    tail = [500]
    arena.activate(parent, 100, capacity_tokens=110, base_tokens=100)
    # the child shares 64 tokens (4 pages) and adds 30 of its own
    arena.activate(child, 94, capacity_tokens=104, base_tokens=94,
                   borrow=(parent, 64))
    assert arena.owned_pages(child)[:4] == arena.owned_pages(parent)[:4]
    chunk = loop.pack_chunk(
        torch, arena,
        [dict(key=parent, prefix=doc, f=100, suffixes=[tail]),
         dict(key=child, prefix=doc[64:94], start=64, f=94, suffixes=[tail])],
        attention_mode="unified")
    assert chunk.fresh_keys == (parent, child)
    assert chunk.tokens == 101 + 31
    positions = chunk.positions.tolist()
    assert positions[:101] == list(range(101))
    assert positions[101:] == list(range(64, 95))
    unified = chunk.meta["unified"]
    assert unified["used"].tolist() == [101, 95]
    table = unified["table"].tolist()
    assert table[1][:4] == table[0][:4]
    # the child's rows land after the borrowed pages, in its own pages
    dst = unified["dst"].tolist()
    assert dst[101:] == arena.capacity_rows(child)[64:95].tolist()
    assert arena.capacity_rows(child)[:64].tolist() == \
        arena.capacity_rows(parent)[:64].tolist()
    # under tree attention without a read_key, the fresh rows read the
    # child's own first 64 rows (the borrowed pages) and the suffix
    # reads all 94
    chunk = loop.pack_chunk(
        torch, arena,
        [dict(key=child, prefix=doc[64:94], start=64, f=94,
              suffixes=[tail, [501]])],
        attention_mode="tree")
    reads = chunk.meta["reads"]
    assert reads["used"].tolist() == [64, 94]
    assert reads["cu_q"].tolist() == [0, 30, 32]
    assert chunk.meta["cu_a"].tolist() == [0, 30, 31, 32]
    arena.free_key(parent)
    arena.free_key(child)
    assert arena.free_pages == 64

    arena = plain_arena()
    parent, a, b = ("d", 0), ("d", 1), ("d", 2)
    arena.activate(parent, 100, capacity_tokens=110, base_tokens=100)
    # two siblings share 64 tokens (4 pages) of the parent
    arena.activate(a, 74, capacity_tokens=84, base_tokens=74,
                   borrow=(parent, 64))
    arena.activate(b, 70, capacity_tokens=80, base_tokens=70,
                   borrow=(parent, 64))
    chunk = loop.pack_chunk(
        torch, arena,
        [dict(key=parent, prefix=doc, f=100, suffixes=[tail]),
         dict(key=a, prefix=doc[64:74], start=64, read_key=parent, f=74,
              suffixes=[tail], write_suffix_tokens=1),
         dict(key=b, prefix=doc[64:70], start=64, read_key=parent, f=70,
              suffixes=[tail], write_suffix_tokens=1)],
        attention_mode="tree")
    assert chunk.fresh_keys == (parent, a, b)
    # call A: the parent's prefix, its tail, then each sibling's
    # prefix and tail as one causal segment
    assert chunk.meta["cu_a"].tolist() == [0, 100, 101, 112, 119]
    reads = chunk.meta["reads"]
    # call B: the parent's tail reads its 100 rows; both siblings' 18
    # rows read the parent's 64 shared positions in one sequence
    assert reads["cu_q"].tolist() == [0, 1, 19]
    assert reads["used"].tolist() == [100, 64]
    assert reads["max_q"] == 18
    assert reads["rows"].tolist() == [100] + list(range(101, 119))
    table = reads["table"].tolist()
    assert table[1][:4] == table[0][:4]
    # the siblings' prefixes and kept tails write their own pages after
    # the borrowed ones
    dst = chunk.meta["kv_dst"].tolist()
    assert dst[100:110] == arena.capacity_rows(a)[64:74].tolist()
    assert dst[110:111] == arena.capacity_rows(a)[74:75].tolist()
    assert dst[111:117] == arena.capacity_rows(b)[64:70].tolist()
    assert dst[117:118] == arena.capacity_rows(b)[70:71].tolist()
    for key in (parent, a, b):
        arena.free_key(key)
    assert arena.free_pages == 64


def test_borrowing_in_the_arena_and_can_borrow(monkeypatch):
    cpu_staging(monkeypatch)
    arena = plain_arena(pages=8)
    parent, other, child = ("d", 0), ("d", 1), ("d", 2)
    assert arena.activate(parent, 64, base_tokens=64)          # 4 pages
    assert arena.activate(other, 48, base_tokens=48)           # 3 pages
    arena.retain(other, 48)
    # the child needs 3 own pages for 48 tokens past its 32 borrowed:
    # only 1 is free, so the retained key goes
    assert arena.activate(child, 80, base_tokens=80, borrow=(parent, 32))
    assert not arena.is_resident(other)
    assert arena.evicted_keys == 1
    assert arena.owned_pages(child)[:2] == arena.owned_pages(parent)[:2]
    assert arena.free_pages == 8 - 4 - 3
    arena.free_key(parent)
    arena.free_key(child)
    assert arena.free_pages == 8

    arena = cpu_arena(pages=64, sliding_pages=32)
    parent, child = ("d", 0), ("d", 1)
    doc = list(range(100))
    tail = [500]
    # the parent stays untrimmed until the child has borrowed
    arena.activate(parent, 100, capacity_tokens=110, base_tokens=100)
    # the child shares 64 tokens: its sliding window starts at
    # origin(64) = 32 and reads the parent's sliding pages for [32, 64)
    arena.activate(child, 94, capacity_tokens=104, base_tokens=94,
                   borrow=(parent, 64))
    assert arena.sliding_start(child) == 32
    assert arena.owned_pages(child)[:4] == arena.owned_pages(parent)[:4]
    assert arena.owned_sliding_pages(child)[:2] == \
        arena.owned_sliding_pages(parent)[2:4]
    # the child's sliding pages: 2 borrowed + pages for [64, 104) = 3 own
    assert len(arena.owned_sliding_pages(child)) == 5
    chunk = loop.pack_chunk(
        torch, arena,
        [dict(key=parent, prefix=doc, f=100, suffixes=[tail]),
         dict(key=child, prefix=doc[64:94], start=64, f=94, suffixes=[tail])],
        attention_mode="unified")
    sliding = chunk.meta["unified"]["sliding"]
    assert sliding["used"].tolist() == [101, 95 - 32]
    table = sliding["table"].tolist()
    full = chunk.meta["unified"]["table"].tolist()
    assert table[1][:2] == table[0][2:4]
    assert full[1][:4] == full[0][:4]
    # the child's rows land after the borrowed window in its own pages
    dst = sliding["dst"].tolist()
    assert dst[101:] == arena.capacity_rows_sliding(child)[32:63].tolist()
    # the parent trims once the child holds the window; the child's
    # borrowed pages survive, and its own trim drops them first
    freed = arena.trim_window(parent)
    assert freed > 0 and arena.sliding_start(parent) == 64
    assert arena.owned_sliding_pages(child)[:2] == \
        arena.owned_sliding_pages(parent)[:0] + arena.owned_sliding_pages(child)[:2]
    assert arena.trim_window(child) > 0      # origin(94) = 48: one page
    assert arena.sliding_start(child) == 48
    assert len(arena.owned_sliding_pages(child)) == 4
    arena.free_key(parent)
    arena.free_key(child)
    assert arena.free_pages == 64

    # a parent trimmed past the child's window refuses the borrow
    arena.activate(parent, 100, capacity_tokens=110, base_tokens=100)
    arena.trim_window(parent)                 # keeps rows from 64
    with pytest.raises(ValueError, match="trimmed"):
        arena.activate(child, 94, capacity_tokens=104, base_tokens=94,
                       borrow=(parent, 64))
    assert not arena.is_resident(child)
    arena.free_key(parent)
    assert arena.free_pages == 64

    arena = cpu_arena(pages=64, sliding_pages=32)
    root, middle = ("d", 0), ("d", 1)
    arena.activate(root, 100, capacity_tokens=110, base_tokens=100)
    # a child sharing most of the root borrows: the root is untrimmed
    assert arena.can_borrow(root, 96)
    arena.activate(middle, 120, capacity_tokens=130, base_tokens=120,
                   borrow=(root, 96))
    # its sliding pages start at origin(96) = 64, so a child sharing
    # only the first 16 tokens cannot get the window below them
    assert arena.sliding_start(middle) == 64
    assert not arena.can_borrow(middle, 16)
    assert arena.can_borrow(middle, 112)
    # the root still has them; a shared length past its tokens is refused
    assert arena.can_borrow(root, 16)
    assert not arena.can_borrow(root, 112)
    assert not arena.can_borrow(("d", 9), 16)
    arena.trim_window(root)
    assert not arena.can_borrow(root, 16)
    arena.free_key(middle)
    arena.free_key(root)
    assert arena.free_pages == 64


def test_join_anchor_keeps_its_window_for_a_later_child(monkeypatch):
    from types import SimpleNamespace

    from fakes import fake_pipeline, fake_torch

    from quail.execution.tokens import prefix_tree

    cpu_staging(monkeypatch)
    # anchor 1 shares 80 tokens with anchor 0, anchor 2 only the first
    # 32; anchor 2 borrows from anchor 0, whose window below row 32
    # must outlive the first chunk
    prefixes = [[1] * 96, [1] * 80 + [2] * 16, [1] * 32 + [3] * 64]
    tree = prefix_tree(prefixes, 16)
    assert tree.parent == [None, 0, 0] and tree.shared == [0, 80, 32]

    def forward(chunk):
        return [0] * sum(n for _, n in chunk.layout)

    # real tensors, fake CUDA events
    fake = fake_torch()
    cpu = SimpleNamespace(**{n: getattr(torch, n) for n in dir(torch)
                             if not n.startswith("_")})
    cpu.cuda, cpu.inference_mode = fake.cuda, fake.inference_mode
    answers = SimpleNamespace(submit=lambda v: v, result=lambda v: v, dtype=None)
    arena = cpu_arena(pages=64, sliding_pages=32)
    stats = {}
    _, _, tokens = loop.run_join(
        cpu, arena, fake_pipeline(forward_chunk=forward), answers,
        prefixes, [[[9], [10]]], 130, stage_frames=[[5]],
        anchor_keys=[("a", d) for d in range(3)], prefix_tree=tree,
        stats=stats)
    assert stats["borrowed_tokens"] == 80 + 32
    assert tokens == 96 + 16 + 64 + 3 * 3
    assert not arena.accounting.owned
