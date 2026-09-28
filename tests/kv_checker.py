"""A fake model that checks every KV read, and runners for filters and joins.

A fake model stands in for the network. Each packed row gets a hash of
its token and the hash its attention reads at the row before it, and
every write stores (token, position, hash) in the arena's rows. A
query row that reads a wrong page, a trimmed window, a page reused
while still borrowed, or a wrong position gets a hash no request has,
and the answer readout fails. The runners give each request an answer
from its hash and check that the loop returns it.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from fakes import fake_pipeline, fake_torch

from quail.backends.quail.executor import loop
from quail.backends.quail.executor.arena import KVArena
from quail.execution.tokens import prefix_tree


@dataclass(frozen=True)
class Setup:
    """One arena and chunk configuration to run a workload on.

    Attributes:
        path: The attention path, "unified" or "tree".
        window: The sliding window in tokens, or None for an arena
            without sliding layers.
        canvas: Token ids of the canvas rows a diffusion model appends
            after every suffix; the first one answers.
        pages: Every-token pool pages.
        sliding_pages: Sliding pool pages.
        budget: Chunk token budget.
        page_tokens: Tokens per KV page.
    """

    path: str
    window: int | None = None
    canvas: tuple = ()
    pages: int = 400
    sliding_pages: int = 400
    budget: int = 600
    page_tokens: int = 16


def mix(previous, token):
    return (previous * 1_000_003 + token + 1) & ((1 << 61) - 1)


def chain(tokens):
    h = 0
    for token in tokens:
        h = mix(h, token)
    return h


class Pool:
    """(token, position, hash) per physical row of one KV pool."""

    def __init__(self, pages, page_tokens):
        rows = pages * page_tokens
        self.token = np.full(rows, -1, dtype=np.int64)
        self.position = np.full(rows, -1, dtype=np.int64)
        self.hash = np.full(rows, -1, dtype=np.int64)


class HashModel:
    """A forward pass whose answers are right only if every KV read is."""

    def __init__(self, arena, answers, canvas):
        self.page = arena.page_tokens
        self.window = arena.window
        self.full = Pool(arena.accounting.n_pages, self.page)
        self.sliding = (Pool(arena.sliding.n_pages, self.page)
                        if arena.has_sliding else None)
        self.answers = answers      # request hash -> answer bit
        self.canvas = canvas

    def forward_chunk(self, chunk):
        ids = chunk.input_ids.tolist()
        positions = chunk.positions.tolist()
        meta = chunk.meta
        hashes = [None] * len(ids)
        if chunk.attention_mode == "unified":
            self._unified(meta["unified"], ids, positions, hashes)
        else:
            self._tree(meta, ids, positions, hashes)
        out = []
        for row in chunk.final_indices.tolist():
            h = hashes[row]
            assert h in self.answers, (
                f"row {row} at position {positions[row]} read KV no "
                f"request has")
            out.append(self.answers[h])
        return out

    # ---- unified: one paged call per pool --------------------------------

    def _unified(self, unified, ids, positions, hashes):
        cu = unified["cu_q"].tolist()
        full = self._pool_maps(unified)
        sliding = (self._pool_maps(unified["sliding"])
                   if "sliding" in unified else None)

        def value(pool, maps, phys):
            written, copied = maps
            if phys in written:
                return (ids[written[phys]], positions[written[phys]],
                        row_hash(written[phys]))
            if phys in copied:
                return value(pool, maps, copied[phys])
            return (int(pool.token[phys]), int(pool.position[phys]),
                    int(pool.hash[phys]))

        seq_of = {}
        for i in range(len(cu) - 1):
            for row in range(cu[i], cu[i + 1]):
                seq_of[row] = i

        page = self.page

        def phys(maps_table, i, view_row):
            pages = maps_table[i]
            return int(pages[view_row // page]) * page + view_row % page

        table = unified["table"].tolist()
        used = unified["used"].tolist()

        def row_hash(row):
            if hashes[row] is not None:
                return hashes[row]
            i = seq_of[row]
            count = cu[i + 1] - cu[i]
            view = used[i] - count + row - cu[i]
            position = positions[row]
            assert view == position, "a query row's KV index is off"
            # a join's frame entry does not keep its canvas row, whose
            # answer nothing reads
            unkept = ids[row] in self.canvas and row == cu[i + 1] - 1
            assert unkept or full[0].get(phys(table, i, view)) == row, \
                "a query row is not written where it reads itself"
            previous = 0
            if view:
                token, pos, previous = value(
                    self.full, full, phys(table, i, view - 1))
                assert pos == position - 1, \
                    f"row at {position} reads position {pos} before it"
            hashes[row] = mix(previous, ids[row])
            return hashes[row]

        for row in range(len(ids)):
            row_hash(row)
        # the resident rows each sequence reads form one chain
        for i in range(len(cu) - 1):
            count = cu[i + 1] - cu[i]
            self._check_chain(lambda v, i=i: value(
                self.full, full, phys(table, i, v)), used[i] - count)
        if sliding is not None:
            s_table = unified["sliding"]["table"].tolist()
            s_used = unified["sliding"]["used"].tolist()
            for i in range(len(cu) - 1):
                # the sequence's rows need the window before its first
                # row through its last row
                count = cu[i + 1] - cu[i]
                first = positions[cu[i]]
                origin = first - (s_used[i] - count)
                lowest = max(0, first - self.window + 1)
                assert lowest >= origin, (
                    f"row at {first} needs window row {lowest}; the "
                    f"sliding pages start at {origin}")
                for k in range(lowest, positions[cu[i + 1] - 1] + 1):
                    got = value(self.sliding, sliding,
                                phys(s_table, i, k - origin))
                    want = value(self.full, full, phys(table, i, k))
                    assert got == want, \
                        f"sliding row {k} differs from the every-token row"
        self._commit(self.full, full, ids, positions, hashes)
        if sliding is not None:
            self._commit(self.sliding, sliding, ids, positions, hashes)

    @staticmethod
    def _pool_maps(pool):
        written = dict(zip(pool["dst"].tolist(), pool["src"].tolist()))
        copied = {}
        if pool["tail_src"] is not None:
            copied = dict(zip(pool["tail_dst"].tolist(),
                              pool["tail_src"].tolist()))
        return written, copied

    def _commit(self, pool, maps, ids, positions, hashes):
        written, copied = maps
        # the scatter lands before the tail copies read the pages
        for dst, src in written.items():
            pool.token[dst] = ids[src]
            pool.position[dst] = positions[src]
            pool.hash[dst] = hashes[src]
        for dst, src in copied.items():
            pool.token[dst] = pool.token[src]
            pool.position[dst] = pool.position[src]
            pool.hash[dst] = pool.hash[src]

    # ---- tree: causal segments plus paged reads ----------------------

    def _tree(self, meta, ids, positions, hashes):
        cu_a = meta["cu_a"].tolist()
        first_of_segment = set(cu_a[:-1])
        written = {}
        if meta["kv_src"] is not None:
            written = dict(zip(meta["kv_dst"].tolist(),
                               meta["kv_src"].tolist()))
        reads = meta["reads"]
        read_of = {}
        read_table, read_used = [], []
        if reads is not None:
            cu_q = reads["cu_q"].tolist()
            rows = reads["rows"].tolist()
            read_table = reads["table"].tolist()
            read_used = reads["used"].tolist()
            for i in range(len(cu_q) - 1):
                for row in rows[cu_q[i]:cu_q[i + 1]]:
                    read_of[row] = i

        def value(phys):
            if phys in written:
                src = written[phys]
                return ids[src], positions[src], row_hash(src)
            return (int(self.full.token[phys]),
                    int(self.full.position[phys]),
                    int(self.full.hash[phys]))

        page = self.page

        def phys(i, view_row):
            return int(read_table[i][view_row // page]) * page + view_row % page

        def row_hash(row):
            if hashes[row] is not None:
                return hashes[row]
            position = positions[row]
            if row not in first_of_segment:
                assert positions[row - 1] == position - 1, \
                    "a causal segment skips a position"
                previous = row_hash(row - 1)
            elif row in read_of:
                i = read_of[row]
                assert read_used[i] == position, \
                    f"row at {position} reads {read_used[i]} resident rows"
                _, pos, previous = value(phys(i, position - 1))
                assert pos == position - 1
            else:
                assert position == 0, \
                    f"row at {position} starts a segment and reads nothing"
                previous = 0
            hashes[row] = mix(previous, ids[row])
            return hashes[row]

        for row in range(len(ids)):
            row_hash(row)
        for i in range(len(read_used)):
            self._check_chain(lambda v, i=i: value(phys(i, v)), read_used[i])
        for dst, src in written.items():
            self.full.token[dst] = ids[src]
            self.full.position[dst] = positions[src]
            self.full.hash[dst] = hashes[src]

    @staticmethod
    def _check_chain(read, rows):
        previous = 0
        for v in range(rows):
            token, pos, h = read(v)
            assert pos == v, f"resident row {v} holds position {pos}"
            assert h == mix(previous, token), \
                f"resident row {v} was overwritten"
            previous = h


def make_arena(setup):
    if setup.window is not None:
        return KVArena(n_layers=2, n_pages=setup.pages,
                       page_tokens=setup.page_tokens, n_kv=1, d_head=1,
                       dtype=torch.float32, device="cpu",
                       layer_kv=[(1, 1), (1, 1)], sliding_layers=(1,),
                       sliding_window=setup.window,
                       n_sliding_pages=setup.sliding_pages)
    return KVArena(n_layers=1, n_pages=setup.pages,
                   page_tokens=setup.page_tokens, n_kv=1, d_head=1,
                   dtype=torch.float32, device="cpu")


def cpu_torch():
    """Real CPU tensors with fake CUDA events."""
    fake = fake_torch()
    real = SimpleNamespace(**{n: getattr(torch, n) for n in dir(torch)
                              if not n.startswith("_")})
    real.cuda, real.inference_mode = fake.cuda, fake.inference_mode
    return real


def passthrough():
    """An answer readout that returns the model's answers as they are."""
    return SimpleNamespace(submit=lambda v: v, result=lambda v: v, dtype=None)


def answer(h):
    """The planted answer of the request whose last row has hash h."""
    return (h >> 7) & 1


class Run:
    """The arena, checking model, and pipeline of one checked run."""

    def __init__(self, setup, answers):
        self.setup = setup
        self.arena = make_arena(setup)
        self.model = HashModel(self.arena, answers, setup.canvas)
        self.pipeline = fake_pipeline(forward_chunk=self.model.forward_chunk,
                                      tree_attention=True,
                                      canvas_ids=setup.canvas)
        self.torch = cpu_torch()

    def tree(self, docs):
        return prefix_tree(docs, self.setup.page_tokens)


def check_filter(docs, questions, setup):
    """Run a filter chain and check its answers and its KV reads."""
    tail = list(setup.canvas)
    answers = {}
    for doc in docs:
        for q in questions:
            h = chain(doc + q + tail)
            answers[h] = answer(h)
    run = Run(setup, answers)
    got, _, _ = loop.run_filter(
        run.torch, run.arena, run.pipeline, passthrough(), docs, questions,
        setup.budget, arena_writes=True,
        arena_keys=[("d", i) for i in range(len(docs))],
        attention_mode=setup.path, prefix_tree=run.tree(docs))
    for d, doc in enumerate(docs):
        want = []
        for q in questions:
            want.append(answers[chain(doc + q + tail)])
            if not want[-1]:
                break
        assert got[d] == want, f"document {d}"
    assert not run.arena.accounting.owned


def _join_answers(anchors, frame, partners, tail):
    # a frame entry's last row reads the anchor and its frame
    answers = {}
    for anchor in anchors:
        answers[chain(anchor + frame)] = 0
        answers[chain(anchor + frame + tail)] = 0
        for partner in partners:
            h = chain(anchor + frame + partner + tail)
            answers[h] = answer(h)
    return answers


def check_join(anchors, frame, partners, setup, retain=False):
    """Run a join and check its answers and its KV reads.

    With retain, each finished anchor's KV stays in the arena as
    evictable, as a join that keeps its anchors' KV for a later
    operator leaves it, and fresh anchors evict it for pages.
    """
    tail = list(setup.canvas)
    answers = _join_answers(anchors, frame, partners, tail)
    run = Run(setup, answers)
    keys = [("a", i) for i in range(len(anchors))]
    if retain:
        run.arena.retention_cap_pages = setup.pages
    rows = {}

    def anchor_done(a, row):
        rows[a] = list(row)
        if retain:
            run.arena.retain(keys[a], len(anchors[a]))
        else:
            run.arena.free_key(keys[a])

    loop.run_join(
        run.torch, run.arena, run.pipeline, passthrough(), anchors,
        [partners], setup.budget, stage_frames=[frame], anchor_keys=keys,
        anchor_done=anchor_done, attention_mode=setup.path,
        prefix_tree=run.tree(anchors))
    for a, anchor in enumerate(anchors):
        assert rows[a] == [answers[chain(anchor + frame + p + tail)]
                           for p in partners], f"anchor {a}"
    run.arena.evict_retained(setup.pages)
    assert not run.arena.accounting.owned


def check_feed(docs, question, frame, partners, setup):
    """Run a filter whose survivors' KV feeds a join, and check both."""
    tail = list(setup.canvas)
    answers = _join_answers(docs, frame, partners, tail)
    for doc in docs:
        h = chain(doc + question + tail)
        answers[h] = answer(h >> 3)
    run = Run(setup, answers)
    source = loop.FilterStream(
        run.torch, run.arena, run.pipeline, passthrough(), docs, [question],
        setup.budget, arena_writes=True,
        arena_keys=[("d", i) for i in range(len(docs))],
        hold_survivors=True, hold_extra_tokens=len(frame) + len(tail),
        prefix_tree=run.tree(docs), attention_mode=setup.path)
    keys = []
    out, _, _ = loop.run_join(
        run.torch, run.arena, run.pipeline, passthrough(), [], [partners],
        setup.budget, stage_frames=[frame], anchor_keys=keys,
        anchor_source=source, attention_mode=setup.path)
    survivors = {d for d, doc in enumerate(docs)
                 if answers[chain(doc + question + tail)]}
    assert {key[1] for key in keys} == survivors
    for a, key in enumerate(keys):
        doc = docs[key[1]]
        assert out[0][a] == [answers[chain(doc + frame + p + tail)]
                             for p in partners], f"anchor {key[1]}"
    assert not run.arena.accounting.owned


def check_join_after_join(anchors, frame, partners, setup):
    """Run a join that retains its anchors, then a join over more anchors.

    The second join's fresh anchors may share a prefix with a retained
    one, which has trimmed its sliding window; its KV reads are checked
    like any other.
    """
    tail = list(setup.canvas)
    answers = _join_answers(anchors, frame, partners, tail)
    run = Run(setup, answers)
    run.arena.retention_cap_pages = setup.pages
    keys = [("a", i) for i in range(len(anchors))]
    first = anchors[:1]

    def retain(a, row):
        run.arena.retain(keys[a], len(anchors[a]))

    loop.run_join(
        run.torch, run.arena, run.pipeline, passthrough(), first,
        [partners], setup.budget, stage_frames=[frame], anchor_keys=keys[:1],
        anchor_done=retain, attention_mode=setup.path,
        prefix_tree=run.tree(first))
    assert run.arena.is_resident(keys[0])
    rows = {}

    def free(a, row):
        rows[a] = list(row)
        run.arena.free_key(keys[a])

    loop.run_join(
        run.torch, run.arena, run.pipeline, passthrough(), anchors,
        [partners], setup.budget, stage_frames=[frame], anchor_keys=keys,
        anchor_done=free, attention_mode=setup.path,
        prefix_tree=run.tree(anchors))
    for a, anchor in enumerate(anchors):
        assert rows[a] == [answers[chain(anchor + frame + p + tail)]
                           for p in partners], f"anchor {a}"
    assert not run.arena.accounting.owned
