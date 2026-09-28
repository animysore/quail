"""The overlapped chunk loop.

Pack on CPU while the GPU runs, gate answers, free pages immediately.

- run_join: continuous anchor admission with JoinAdmission, stages
  mixed in one chunk, pages freed when an anchor fails a gate or
  answers its last stage.
- run_filter: continuous admission with FilterAdmission, pages freed
  on FALSE or after the last stage.

Torch is imported lazily when model execution starts.
"""

import time

import numpy as np

from quail.backends.quail.executor.attention import (
    ATTENTION_PATHS,
    Chunk,
)
from quail.backends.quail.executor.pack import (
    FilterAdmission,
    JoinAdmission,
    partner_pages,
)
from quail.progress import Progress, logger, quiet


class ArenaFullError(RuntimeError):
    """A chunk's suffix pages do not fit the free KV arena."""


def _tick(timing, key, t0):
    if timing is not None:
        timing[key] = timing.get(key, 0.0) + time.perf_counter() - t0
    return time.perf_counter()


def _staged(torch, data, dtype, pinned=True):
    """Host data to device through pinned memory, non-blocking.

    pinned=False reverts to pageable blocking copies.
    """
    if isinstance(data, np.ndarray):
        data = torch.as_tensor(data, dtype=dtype)
    if torch.is_tensor(data):
        if pinned:
            return data.pin_memory().to("cuda", non_blocking=True)
        return data.to("cuda")
    if pinned:
        return torch.tensor(data, dtype=dtype, pin_memory=True).to(
            "cuda", non_blocking=True)
    return torch.tensor(data, dtype=dtype, device="cuda")


def _token_parts(sequence):
    parts = getattr(sequence, "token_parts", None)
    if parts is None or (len(parts) == 1 and parts[0] is sequence):
        yield sequence
        return
    for part in parts:
        yield from _token_parts(part)


class InputStaging:
    """Reuse CPU and GPU input buffers on the current CUDA stream."""

    def __init__(self, torch):
        self.torch = torch
        self.buffers = {}
        self.fixed_tokens = {}

    def host(self, name, count, dtype):
        torch = self.torch
        previous = self.buffers.get(name)
        if previous is not None:
            host, device, event = previous
            # The CPU must not overwrite a buffer while its transfer is pending.
            event.synchronize()
        if previous is None or host.numel() < count or host.dtype != dtype:
            host = torch.empty(count, dtype=dtype, pin_memory=True)
            device = torch.empty(count, dtype=dtype, device="cuda")
            event = torch.cuda.Event()
        self.buffers[name] = host, device, event
        return host[:count]

    def upload(self, name, count):
        host, device, event = self.buffers[name]
        # Reusing the device buffer is ordered after its previous readers.
        device[:count].copy_(host[:count], non_blocking=True)
        event.record()
        return device[:count]

    def copy(self, name, data, dtype):
        source = self.torch.as_tensor(data)
        host = self.host(name, source.numel(), dtype)
        host.copy_(source.reshape(-1))
        return self.upload(name, source.numel())

    def fixed(self, tokens):
        key = id(tokens)
        if key not in self.fixed_tokens:
            self.fixed_tokens[key] = tokens, self.torch.as_tensor(tokens)
        return self.fixed_tokens[key][1]


def _staged_token_parts(torch, sequences, total, pinned=True, staging=None):
    """Copy Arrow token views into one GPU input tensor."""
    host = (torch.empty(total, dtype=torch.int64, pin_memory=pinned)
            if staging is None else staging.host("tokens", total, torch.int64))
    offset = 0
    for sequence in sequences:
        for part in _token_parts(sequence):
            count = len(part)
            if not count:
                continue
            if hasattr(part, "arrow_array"):
                source = torch.from_dlpack(part.arrow_array)
            elif torch.is_tensor(part):
                source = part
            elif staging is not None and isinstance(part, tuple):
                source = staging.fixed(part)
            else:
                source = torch.as_tensor(part)
            host[offset:offset + count].copy_(source)
            offset += count
    if offset != total:
        raise AssertionError(
            f"packed {offset} token ids into a {total}-token chunk")
    if staging is not None:
        return staging.upload("tokens", total)
    return host.to("cuda", non_blocking=pinned)


# ------------------------------------------------------- chunk packing

class _PoolView:
    """One key's rows and pages in one pool.

    start is the logical row the pool's first page holds: 0 on the
    every-token pool, the window origin on a trimmed key's sliding
    pool.
    """

    __slots__ = ("rows", "pages", "start")

    def __init__(self, rows, pages, start):
        self.rows = rows
        self.pages = pages
        self.start = start


class _PoolBuilder:
    """The scatter map, block table, and lengths of one pool for a chunk."""

    def __init__(self):
        self.page_rows = []
        self.lengths = []
        self.src = []
        self.dst = []
        self.tail_src = []
        self.tail_dst = []

    def add_sequence(self, pages, kv_tokens):
        self.page_rows.append(pages)
        self.lengths.append(kv_tokens)

    def scatter_direct(self, view, r0, r1, logical_start):
        """Rows r0..r1 land at logical_start.. in the key's own pages.

        Rows before the pool's start are not stored there.
        """
        first = max(logical_start, view.start)
        skip = first - logical_start
        if r0 + skip >= r1:
            return
        self.src.append(np.arange(r0 + skip, r1, dtype=np.int64))
        offset = first - view.start
        self.dst.append(view.rows[offset:offset + (r1 - r0 - skip)].numpy())

    def scatter_suffix(self, view, temp, s0, s1, f, remainder, page_tokens):
        """Suffix rows go to a temporary, after a copy of the kept tail."""
        suffix_tokens = s1 - s0
        self.src.append(np.arange(s0, s1, dtype=np.int64))
        self.dst.append(temp.rows[remainder:remainder + suffix_tokens].numpy())
        if remainder:
            anchor_page = view.pages[(f - view.start) // page_tokens]
            base = anchor_page * page_tokens
            self.tail_src.append(np.arange(base, base + remainder, dtype=np.int64))
            self.tail_dst.append(temp.rows[:remainder].numpy())

    def build(self, arena, stage, torch, staging, index):
        def concatenate(parts):
            return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

        tag = "" if index == 0 else "_sliding"
        if staging is None:
            table = arena.block_table_rows(self.page_rows)
        else:
            width = max(map(len, self.page_rows))
            block_table = np.zeros((len(self.page_rows), width), dtype=np.int32)
            for row, pages in enumerate(self.page_rows):
                block_table[row, :len(pages)] = pages
            table = stage(f"block_table{tag}", block_table, torch.int32).view(
                len(self.page_rows), width)
        return dict(
            src=stage(f"unified_src{tag}", concatenate(self.src), torch.int64),
            dst=stage(f"unified_dst{tag}", concatenate(self.dst), torch.int64),
            tail_src=(stage(f"tail_src{tag}", concatenate(self.tail_src),
                            torch.int64) if self.tail_src else None),
            tail_dst=(stage(f"tail_dst{tag}", concatenate(self.tail_dst),
                            torch.int64) if self.tail_dst else None),
            used=stage(f"unified_lengths{tag}", self.lengths, torch.int32),
            table=table,
            max_used=max(self.lengths))


def pack_chunk(torch, arena, groups, timing=None, pinned=True, *,
               attention_mode, staging=None, canvas=(), answer_row=0):
    """Build tensors for one chunk from groups in chunk order.

    Each group is a dict with keys:
      key       Arena key for KV reads/writes.
      prefix    Fresh prefix token list, or None when KV is resident.
      start     Logical position of the first fresh prefix token; the
                key's pages before it are borrowed from a parent and
                already hold KV. 0 when absent.
      read_key  Under tree attention, for a group with one suffix,
                the parent whose pages the fresh rows read for
                positions before start; the rows still write the
                group's own key. Consecutive groups with one read_key
                and start share one read of it. Without it, the fresh
                prefix reads the key's own borrowed pages and each
                suffix reads the whole key as usual.
      f         Kept-context length (suffix positions start here).
      suffixes  List of suffix token lists.
      write_suffix_tokens  Leading rows of the first suffix to scatter
                into the key's pages after the prefix.

    A fresh group without arena pages packs [prefix | suffix] as one
    causal segment (no scatter, no paged read). This only works with
    at most one suffix. Under unified attention a chunk is either all
    paged or all unpaged.

    canvas is the token ids a diffusion model denoises: they follow
    every suffix as extra rows, and the answer row is canvas row
    answer_row instead of the suffix's last row. Canvas KV goes
    wherever the suffix's KV goes and is never kept. Canvas rows run
    the unified path only.
    """
    def stage(name, values, dtype):
        if staging is not None:
            return staging.copy(name, values, dtype)
        return _staged(torch, values, dtype, pinned)

    def concatenate(parts):
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    if attention_mode not in ATTENTION_PATHS:
        raise ValueError(
            f"attention_mode must be one of {ATTENTION_PATHS}, "
            f"got {attention_mode!r}")
    canvas = tuple(canvas)
    if canvas and attention_mode != "unified":
        raise ValueError("canvas rows run the unified attention path")
    if canvas and not 0 <= answer_row < len(canvas):
        raise ValueError(
            f"answer_row {answer_row} is outside the {len(canvas)}-row canvas")
    t = time.perf_counter() if timing is not None else 0.0
    # unified scatters every fresh row through its own src/dst map, so
    # the reads and kv_writes bookkeeping below is tree only
    tree_path = attention_mode == "tree"
    canvas_rows = []      # (first row, end row) per canvas
    canvas_seq = []       # its sequence in the unified paged call
    id_parts, token_count = [], 0
    pos, cu_a, finals = [], [0], []
    # rows that read resident KV in call B: a borrowing document's
    # rows reading its parent, and a kept document's tail reading its
    # own prefix (wider than the planner's readers, which are children)
    reader_rows = []
    kv_writes, layout = [], []
    fresh_keys = []
    read_keys, read_used, cu_q = [], [], [0]
    max_q = 0
    unified_groups = []
    for g in groups:
        fresh = g.get("prefix") is not None
        f = g["f"]
        key = g["key"]
        start = g.get("start", 0)
        paged = arena.is_resident(key)
        row0 = token_count
        # a borrowing fresh group under tree attention reads the pages
        # before start in call B. With one suffix and a read_key the
        # group is one causal segment that reads the parent, stacked
        # with its siblings; otherwise the prefix is its own segment
        # reading the key's borrowed pages
        borrowing = fresh and start and tree_path
        stacked = (borrowing and len(g["suffixes"]) == 1
                   and g.get("read_key") is not None)
        if fresh:
            if not paged and len(g["suffixes"]) > 1:
                raise ValueError(
                    f"group {key!r}: an unpaged fresh group packs "
                    f"[prefix | suffix] as one causal segment, which "
                    f"only one suffix may join - allocate pages or "
                    f"split the group")
            if start and not paged:
                raise ValueError(
                    f"group {key!r}: a borrowed prefix needs arena pages")
            id_parts.append(g["prefix"])
            token_count += len(g["prefix"])
            pos.append(np.arange(start, start + len(g["prefix"]),
                                 dtype=np.int64))
            if (paged or not g["suffixes"]) and not stacked:
                cu_a.append(token_count)
            if borrowing and not stacked:
                reader_rows.extend(range(row0, token_count))
                cu_q.append(cu_q[-1] + token_count - row0)
                read_keys.append(key)
                read_used.append(start)
                max_q = max(max_q, token_count - row0)
            if paged and tree_path:
                kv_writes.append((key, row0, row0 + len(g["prefix"]),
                                  start))
                fresh_keys.append(key)
        prefix_end = token_count
        s_row0 = token_count
        suffix_spans = []
        for si, suf in enumerate(g["suffixes"]):
            srow = token_count
            id_parts.append(suf)
            token_count += len(suf)
            pos.append(np.arange(f, f + len(suf), dtype=np.int64))
            if canvas:
                # the canvas continues the suffix's positions; its
                # first row carries the answer
                first = f + len(suf)
                id_parts.append(canvas)
                pos.append(np.arange(first, first + len(canvas),
                                     dtype=np.int64))
                canvas_rows.append((token_count, token_count + len(canvas)))
                finals.append(token_count + answer_row)
                token_count += len(canvas)
            else:
                finals.append(token_count - 1)
            suffix_spans.append((srow, token_count))
            cu_a.append(token_count)
            wst = g.get("write_suffix_tokens", 0)
            if si == 0 and wst and paged and tree_path:
                # the shared question preamble joins the kept KV right
                # after the document rows: after the fresh prefix, or
                # after a kept document's f rows
                dest = start + len(g["prefix"]) if fresh else f
                kv_writes.append((key, srow, srow + wst, dest))
        s_count = token_count - s_row0
        layout.append((key, len(g["suffixes"])))
        if stacked:
            # every row of the group reads the parent's pages up to
            # start; siblings packed back to back share that read
            count = token_count - row0
            reader_rows.extend(range(row0, token_count))
            read_key = g["read_key"]
            if read_keys and read_keys[-1] == read_key \
                    and read_used[-1] == start:
                cu_q[-1] += count
                max_q = max(max_q, cu_q[-1] - cu_q[-2])
            else:
                cu_q.append(cu_q[-1] + count)
                read_keys.append(read_key)
                read_used.append(start)
                max_q = max(max_q, count)
        elif s_count and f and paged and tree_path:
            reader_rows.extend(range(s_row0, token_count))
            cu_q.append(cu_q[-1] + s_count)
            read_keys.append(key)
            read_used.append(f)
            max_q = max(max_q, s_count)
        if attention_mode == "unified":
            if paged:
                unified_groups.append(dict(
                    key=key, fresh=fresh, f=f, row0=row0, start=start,
                    prefix_end=prefix_end, row1=token_count,
                    suffix_spans=suffix_spans))
            elif not fresh:
                raise ValueError(
                    "unified attention requires pages for a kept "
                    "group - an unpaged kept group has no KV to read")

    t = _tick(timing, "pack_py", t)
    reads = None
    if read_keys:
        table, _ = arena.block_table(read_keys)
        t = _tick(timing, "pack_blocktable", t)
        used = _staged(torch, read_used, torch.int32, pinned)
        reads = dict(
            rows=_staged(torch, reader_rows, torch.int64, pinned),
            cu_q=stage("unified_cu_q", cu_q, torch.int32),
            max_q=max_q, used=used,
            max_used=max(read_used), table=table)
        # row -> its index in call B's output, -1 for rows that read
        # nothing resident; the fused merge kernel's map
        source = [-1] * token_count
        for i, row in enumerate(reader_rows):
            source[row] = i
        reads["source"] = _staged(
            torch, source, torch.int32, pinned)
    t = _tick(timing, "pack_reads", t)

    # all of the chunk's KV writes as one list of (source, destination)
    # row pairs: the attention pass scatters them with a single kernel
    # launch per layer (kv_row_scatter)
    unified = None
    temporary_keys = []
    if attention_mode == "unified" and unified_groups:
        if len(unified_groups) != len(layout):
            raise ValueError(
                "a unified chunk cannot mix paged and unpaged "
                "groups: the one paged call covers every row or "
                "none (the fast path packs whole chunks unpaged)")
        sliding = arena.has_sliding
        # one builder per pool: the every-token pool, and the
        # sliding pool that holds each key's rows from its window
        # origin on
        pools = [_PoolBuilder()] + ([_PoolBuilder()] if sliding else [])
        cu_q = [0]
        page_tokens = arena.page_tokens

        try:
            for spec in unified_groups:
                key = spec["key"]
                f = spec["f"]
                r0, r1 = spec["row0"], spec["row1"]
                spans = spec["suffix_spans"]
                logical_start = spec["start"] if spec["fresh"] else f
                if spec["fresh"]:
                    fresh_keys.append(key)
                count = r1 - r0
                views = [_PoolView(
                    arena.capacity_rows(key), arena.owned_pages(key), 0)]
                if sliding:
                    views.append(_PoolView(
                        arena.capacity_rows_sliding(key),
                        arena.owned_sliding_pages(key),
                        arena.sliding_start(key)))
                direct = len(spans) == 1 and all(
                    logical_start + count - view.start <= view.rows.numel()
                    for view in views)
                suffix_total = sum(e - s for s, e in spans)
                if direct:
                    for pool, view in zip(pools, views):
                        pool.scatter_direct(view, r0, r1, logical_start)
                        pool.add_sequence(view.pages, f + suffix_total - view.start)
                    cu_q.append(cu_q[-1] + count)
                    if canvas:
                        canvas_seq.append(len(cu_q) - 2)
                    continue

                prefix_count = spec["prefix_end"] - r0
                if prefix_count:
                    for pool, view in zip(pools, views):
                        pool.scatter_direct(view, r0, spec["prefix_end"],
                                            spec["start"])
                        pool.add_sequence(
                            view.pages[:arena.pages_needed(f - view.start)],
                            f - view.start)
                    cu_q.append(cu_q[-1] + prefix_count)

                for s0, s1 in spans:
                    suffix_tokens = s1 - s0
                    remainders = [(f - view.start) % page_tokens for view in views]
                    got = arena.alloc_temporary(
                        remainders[0] + suffix_tokens,
                        sliding_tokens=(remainders[1] + suffix_tokens
                                        if sliding else None))
                    if got is None:
                        sliding_free = (arena.sliding.free_pages
                                        if sliding else None)
                        raise ArenaFullError(
                            "unified suffix pages exceed the free KV arena; "
                            f"split the chunk (asked {remainders[0] + suffix_tokens}"
                            f" rows; free pages {arena.accounting.free_pages}"
                            f" every-token, {sliding_free} sliding; retained"
                            f" {arena.retained_pages}; groups {len(groups)})")
                    temp_key, _ = got
                    temporary_keys.append(temp_key)
                    temp_views = [_PoolView(
                        arena.capacity_rows(temp_key),
                        arena.owned_pages(temp_key), 0)]
                    if sliding:
                        temp_views.append(_PoolView(
                            arena.capacity_rows_sliding(temp_key),
                            arena.owned_sliding_pages(temp_key), 0))
                    for pool, view, temp, remainder in zip(
                            pools, views, temp_views, remainders):
                        pool.scatter_suffix(view, temp, s0, s1, f, remainder,
                                            page_tokens)
                        kept_pages = (f - view.start) // page_tokens
                        pool.add_sequence(
                            view.pages[:kept_pages] + temp.pages,
                            f - view.start + suffix_tokens)
                    cu_q.append(cu_q[-1] + suffix_tokens)
                    if canvas:
                        canvas_seq.append(len(cu_q) - 2)

            built = [pool.build(arena, stage, torch, staging, index)
                     for index, pool in enumerate(pools)]
            unified = dict(
                built[0],
                cu_q=stage("unified_cu_q", cu_q, torch.int32),
                max_q=max(b - a for a, b in zip(cu_q, cu_q[1:])))
            if sliding:
                unified["sliding"] = built[1]
        except Exception:
            for key in temporary_keys:
                arena.free_key(key)
            raise

    # All current KV writes use one scatter.
    kv_src = kv_dst = None
    if kv_writes:
        src = []
        dst_parts = []
        for key, r0, r1, dest in kv_writes:
            src.extend(range(r0, r1))
            rows = arena.capacity_rows(key)[dest:dest + (r1 - r0)]
            if rows.numel() != r1 - r0:
                raise AssertionError("KV write exceeds reserved rows")
            dst_parts.append(rows)
        kv_src = _staged(torch, src, torch.int64, pinned)
        kv_dst = _staged(torch, torch.cat(dst_parts), torch.int64,
                         pinned)
    t = _tick(timing, "pack_kv", t)

    canvas_meta = None
    if canvas_rows:
        if (attention_mode == "unified" and unified is None
                and len(canvas_rows) != len(cu_a) - 1):
            raise ValueError(
                "the unpaged canvas call pairs one canvas with each "
                "causal segment; a group without a suffix has none")
        cu_c = [0]
        for a, b in canvas_rows:
            cu_c.append(cu_c[-1] + b - a)
        canvas_meta = dict(
            rows=stage("canvas_rows", concatenate(
                [np.arange(a, b, dtype=np.int64) for a, b in canvas_rows]),
                torch.int64),
            cu_q=stage("canvas_cu_q", cu_c, torch.int32),
            max_q=len(canvas))
        if unified is not None:
            seq = stage("canvas_seq", canvas_seq, torch.int64)
            canvas_meta["table"] = unified["table"].index_select(0, seq)
            canvas_meta["used"] = unified["used"].index_select(0, seq)
            canvas_meta["max_used"] = max(
                pools[0].lengths[i] for i in canvas_seq)
            if "sliding" in unified:
                canvas_meta["sliding"] = dict(
                    table=unified["sliding"]["table"].index_select(0, seq),
                    used=unified["sliding"]["used"].index_select(0, seq),
                    max_used=max(pools[1].lengths[i] for i in canvas_seq))
    meta = dict(
        layer=0, kv_src=kv_src, kv_dst=kv_dst, reads=reads,
        unified=unified, canvas=canvas_meta,
        cu_a=stage("cu_a", cu_a, torch.int32),
        max_a=max(cu_a[i + 1] - cu_a[i] for i in range(len(cu_a) - 1)))
    out = Chunk(
        input_ids=_staged_token_parts(
            torch, id_parts, token_count, pinned, staging),
        positions=stage("positions", concatenate(pos), torch.int64),
        final_indices=stage("finals", finals, torch.int64),
        meta=meta, attention_mode=attention_mode, tokens=token_count,
        layout=layout, temporary_keys=tuple(temporary_keys),
        fresh_keys=tuple(fresh_keys))
    _tick(timing, "pack_h2d", t)
    return out


def attention_path(pipeline, requested, default="unified") -> str:
    """The path a filter stream or join runs: the plan's request where the model has it.

    None asks for the default. A model without tree attention, and a
    canvas model (whose rows the tree path does not pack), run unified.
    """
    if requested is None:
        requested = default
    if requested == "tree" and pipeline.tree_attention and not pipeline.canvas_ids:
        return "tree"
    return "unified"


def _forward(pipeline, arena, chunk):
    """Run the forward pass, then release the chunk's temporary pages.

    The loop owns every page it hands the forward pass, so it frees the
    temporaries whether the pass returns or raises.
    """
    try:
        return pipeline.forward_chunk(chunk)
    finally:
        keys, chunk.temporary_keys = chunk.temporary_keys, ()
        for key in keys:
            arena.free_key(key)


# ------------------------------------------------------------ the join

def run_join(torch, arena, pipeline, async_ans, anchor_prefixes,
             stage_suffixes, budget, stage_frames=None,
             anchor_keys=None, anchor_done=None, anchor_source=None,
             anchor_partners=None, anchor_batch=None, staging=None,
             attention_mode=None, prefix_tree=None, stats=None):
    """The join driver: stream partner lists against anchors.

    Survivors are gated between stages.

    Args:
        torch: The torch module, imported by the caller.
        arena: KVArena holding the anchors' KV pages.
        pipeline: ModelPipeline that runs each packed forward chunk.
        async_ans: Readout that turns final hidden states into answer
            rows: AsyncAnswers for TRUE/FALSE bits, AsyncScores for
            numeric scores. Its dtype picks the answer array type.
        anchor_prefixes: Anchor id (list index) -> prefix token list.
            With anchor_source it must be an empty list; the join
            appends each streamed anchor's prefix to it.
        stage_suffixes: Per stage, the partner suffix token lists.
        budget: Chunk token budget.
        stage_frames: Per stage, task framing token list written into
            each anchor's kept KV after the document rows.
        anchor_keys: Stable arena key for each anchor. List positions are
            used when omitted. With anchor_source it must be an empty
            list; the join appends each streamed anchor's key to it.
        anchor_done: Optional callback(anchor position, final answer row).
            It owns the anchor's final retain or free decision.
        anchor_source: Optional stream that admits anchors while the
            join runs: next() runs one chunk of its own and returns
            ((key, prefix) pairs, blocked); done is True once nothing
            is left; chunks counts its launched chunks. Every key it
            hands over is resident in the arena, so the join packs no
            prefix tokens for it. The join pulls until it can fill a
            chunk and then runs one. blocked means the source is out
            of arena pages until the join frees some.
        anchor_partners: Optional callable(anchor key) -> per stage,
            the indices into that stage's partner list the anchor
            streams, or None for the whole list. Omitted means every
            anchor streams every partner.
        anchor_batch: Optional callable(keys) -> the keys to admit, run
            on each batch the source hands over before admission. A
            key it leaves out is freed, never admitted.
        staging: Optional reusable input transfer buffers.
        attention_mode: "tree" or "unified" as the plan chose; None
            and a model without tree attention run unified.
        prefix_tree: A PrefixTree over anchor_prefixes, or None. A fresh
            anchor whose parent's pages are resident borrows them for
            its shared length and packs only the tokens after it.
        stats: When given, receives borrowed_tokens: the anchor prefix
            tokens read from a parent's KV pages instead of computed.

    Returns:
        (ans, spans, tokens): ans[j][a] = 0/1 row over the stage-j
        partners anchor a streams, with a in admission order; spans =
        (stage, start_event, end_event) per forward; tokens = fresh
        tokens packed.
    """
    k = len(stage_suffixes)
    if anchor_source is not None:
        if anchor_prefixes or anchor_keys:
            raise ValueError(
                "anchor_source fills anchor_prefixes and anchor_keys")
        prefixes, keys = anchor_prefixes, anchor_keys
    else:
        prefixes = list(anchor_prefixes)
        keys = (list(range(len(prefixes))) if anchor_keys is None
                else list(anchor_keys))
    if len(keys) != len(prefixes):
        raise ValueError("anchor_keys must match anchor_prefixes")
    frames = stage_frames or [[] for _ in range(k)]
    # a join's partners read their anchor: tree unless the plan says
    mode = attention_path(pipeline, attention_mode, default="tree")
    canvas = tuple(pipeline.canvas_ids)
    answer_row = pipeline.canvas_answer_row

    def entry_rows(a, j, start, end, carried):
        f = len(prefixes[a])
        frame = frames[j]
        sufs = [stage_suffixes[j][i]
                for i in sched.partner_indices(a, j, start, end)]
        rows = (f if carried else 0) + sum(len(s) + len(canvas) for s in sufs)
        if frame and start == 0:
            rows += len(frame) + len(canvas)
        return rows

    if k == 0:
        return [], [], 0
    # a frame entry's canvas rows land in the anchor's pages after the
    # frame, so the pages cover them
    frame_max = max(len(f) + (len(canvas) if f else 0) for f in frames)
    if anchor_source is not None and mode == "unified":
        # the source leaves room for one partner's temporary rows, or
        # held anchors could fill the arena before the join can run;
        # the rows follow the anchor's last partial page, as the
        # scheduler prices them
        longest = [max((len(s) for s in suffixes), default=0) + len(canvas)
                   for suffixes in stage_suffixes]
        anchor_source.set_reserve(max(
            (partner_pages(arena.page_cost, arena.page_tokens, len(doc),
                           len(frame), rows)
             for doc in anchor_source.doc_ids
             for frame, rows in zip(frames, longest)), default=0))
    def held_pages(key, prefix_tokens):
        # the scheduler prices an anchor at page_cost(prefix + frames)
        # less what it holds; a trimmed window holds fewer sliding
        # pages than that price assumes, so count what growing takes
        capacity = prefix_tokens + frame_max
        return (arena.page_cost(capacity)
                - arena.growth_cost(key, capacity))

    resident = {a: held_pages(keys[a], len(prefixes[a]))
                for a in range(len(keys)) if arena.is_resident(keys[a])}
    # Every resident anchor stays available for the whole join while
    # fresh admissions evict unrelated retained KV.
    for a in resident:
        arena.pin(keys[a])
    def partners_of(key):
        return None if anchor_partners is None else anchor_partners(key)

    if prefix_tree is not None and not prefix_tree.shared_tokens:
        prefix_tree = None
    sched = JoinAdmission(
        [len(p) for p in prefixes],
        [[len(s) for s in sufs] for sufs in stage_suffixes],
        budget, arena.n_pages, arena.page_tokens,
        frame_tokens=[len(f) for f in frames], resident=resident,
        anchor_partners={a: partners_of(keys[a]) for a in range(len(keys))},
        # a windowed model may pack any chunk unified, so it reserves
        # the unified path's temporary pages throughout
        temporary_suffix_pages=mode == "unified",
        answer_dtype=async_ans.dtype,
        canvas_tokens=len(canvas),
        page_cost=arena.page_cost,
        tree=prefix_tree,
    )
    borrowing = sched.borrowing
    borrowing.can_borrow = borrow_check(arena, keys, borrowing)
    # counted after resident anchors detach: they pack no prefix
    borrowers = borrowing.borrowers()
    lowest_borrow = lowest_borrows(borrowing)
    held = [keys[a] for a, n in enumerate(borrowers) if n]
    spans = []
    tokens = 0
    outstanding = []     # (groups, handle) in launch order
    scoring = async_ans.dtype is not None
    label = "AI.SCORE" if scoring else f"join ({k} stages)"
    total = (sum(sched._count(a, j) for a in range(len(prefixes)) for j in range(k))
             if scoring else len(prefixes))
    progress = Progress(
        label, total=None if anchor_source is not None else total,
        unit="scores" if scoring else "anchors")
    finished = [0]

    def admit(items):
        if anchor_batch is not None and items:
            kept = set(anchor_batch([key for key, _ in items]))
            for key, _ in items:
                if key not in kept and arena.is_resident(key):
                    arena.free_key(key)
            items = [(key, prefix) for key, prefix in items if key in kept]
        for key, prefix in items:
            if not arena.is_resident(key):
                raise ValueError(
                    f"streamed anchor {key!r} has no KV in the arena")
            arena.pin(key)
            keys.append(key)
            prefixes.append(prefix)
            sched.admit(len(prefix), held_pages(key, len(prefix)),
                        partners=partners_of(key))

    def pull(evict_retained=False, force=False):
        """Run source chunks until a join chunk can fill or the source blocks.

        Returns whether the source launched a chunk or handed over an
        anchor.
        """
        moved = False
        while not anchor_source.done and (
                force or sched.buildable_tokens() < budget):
            force = False
            before = anchor_source.chunks
            items, blocked = anchor_source.next(
                evict_retained=evict_retained)
            admit(items)
            moved = moved or bool(items) or anchor_source.chunks > before
            if blocked:
                break
        return moved

    def build(chunk_groups):
        specs = []
        for a, j, start, end, carried in chunk_groups:
            key = keys[a]
            f = len(prefixes[a])
            frame = frames[j]
            parent, shared = ((borrowing.parent(a), borrowing.shared(a))
                              if carried else (None, 0))
            fresh = not arena.is_resident(key)
            got = arena.activate(
                key, f, capacity_tokens=f + frame_max, base_tokens=f,
                borrow=(keys[parent], shared) if shared else None)
            assert got is not None, \
                "scheduler admitted an anchor the arena cannot hold"
            if fresh and a < len(borrowers) and borrowers[a]:
                arena.hold(key, borrowers[a])
                arena.keep_window(key, lowest_borrow[a])
            prefix = None
            if carried:
                prefix = prefixes[a][shared:] if shared else prefixes[a]
            sufs = [stage_suffixes[j][i]
                    for i in sched.partner_indices(a, j, start, end)]
            if frame and start == 0:
                # frame entry: scatter the frame into KV after the
                # document rows. The pair entry reads doc + frame.
                # The frame entry's answer bit is skipped by report().
                specs.append(dict(
                    key=key, prefix=prefix, start=shared,
                    f=f, suffixes=[frame],
                    write_suffix_tokens=len(frame)))
                specs.append(dict(
                    key=key, prefix=None, f=f + len(frame),
                    suffixes=sufs))
            else:
                specs.append(dict(
                    key=key, prefix=prefix, start=shared,
                    f=f + len(frame),
                    suffixes=sufs))
        return pack_chunk(torch, arena, specs, attention_mode=mode,
                          staging=staging, canvas=canvas,
                          answer_row=answer_row)

    def settle(anchor):
        if anchor_done is None:
            if arena.is_resident(keys[anchor]):
                arena.free_key(keys[anchor])
        else:
            anchor_done(anchor, sched.answers[k - 1].get(anchor, []))

    def event(kind, anchor):
        if kind == "finished":
            settle(anchor)
            finished[0] += 1
        elif arena.is_resident(keys[anchor]):
            arena.free_key(keys[anchor])

    def report(entry):
        groups, handle = entry
        bits = async_ans.result(handle)
        pos = 0
        for a, j, start, end, _ in groups:
            if frames[j] and start == 0:
                pos += 1        # the frame entry's bit means nothing
            cnt = end - start
            for kind, anchor in sched.report(
                    a, j, start, end, bits[pos:pos + cnt]):
                event(kind, anchor)
            pos += cnt
        progress.update(
            progress.done + sum(end - start for _, _, start, end, _ in groups)
            if scoring else finished[0])

    def run_part(part):
        """Run one chunk of groups, halving it when its pages do not fit.

        The admission prices a chunk's temporary suffix pages, but the
        packer takes them in both pools a page at a time, and on long
        documents with many short suffixes the tighter pool can run
        out. Halving costs only chunk efficiency.
        """
        try:
            run_one(part)
        except ArenaFullError as error:
            # retained KV nothing here reads makes room first; it is
            # a cache
            need = arena.page_cost(sum(entry_rows(*e) for e in part))
            if arena.evict_retained(need):
                logger.info("join chunk of %d groups retried after "
                            "evicting retained KV", len(part))
                run_part(part)
                return
            if len(part) < 2:
                raise
            logger.info("join chunk of %d groups split: %s", len(part), error)
            half = len(part) // 2
            run_part(part[:half])
            run_part(part[half:])

    def run_one(part):
        nonlocal tokens
        chunk = build(part)
        tokens += chunk.tokens
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        normed = _forward(pipeline, arena, chunk)
        e1.record()
        for key in chunk.fresh_keys:
            arena.trim_window(key)
        spans.append((part[0][1], e0, e1))
        outstanding.append((part, async_ans.submit(normed)))
        # read the previous chunk's answers while this one runs
        while len(outstanding) > 1:
            report(outstanding.pop(0))

    while True:
        if anchor_source is not None and not anchor_source.done:
            pull()
        # anchors with no partner at their first stage never run
        for kind, anchor in sched.take_settled():
            event(kind, anchor)
        if sched.done() and (anchor_source is None
                             or anchor_source.done):
            break
        if sched.blocked_pages:
            # the free list is short for the next fresh anchor:
            # retained KV nothing here reads makes room
            arena.evict_retained(sched.blocked_pages)
        groups = sched.next_chunk(arena.free_pages)
        if not groups:
            if outstanding:
                report(outstanding.pop(0))
                continue
            if anchor_source is not None and not anchor_source.done:
                # the source has to move; it may evict retained KV to admit
                if pull(evict_retained=True, force=True):
                    continue
            if sched.blocked_pages and arena.evict_retained(sched.blocked_pages):
                continue
            # parents freed but held for queued children hold the
            # pages: the children pack their whole prefixes
            if arena.drop_holds(held):
                continue
            raise AssertionError("nothing buildable and nothing in flight")
        run_part(groups)
        # the packed children hold their parents' pages now
        for a, _, start, _, carried in groups:
            parent = (borrowing.tree_parent[a]
                      if a < len(borrowing.tree_parent) else None)
            if carried and start == 0 and parent is not None:
                arena.release(keys[parent])
    while outstanding:
        report(outstanding.pop(0))
    progress.finish(f"{label} done", f"{tokens:,} fresh tokens")
    arena.drop_holds(held)
    if stats is not None:
        stats["borrowed_tokens"] = borrowing.borrowed_tokens
    return sched.answers, spans, tokens


# ------------------------------------------------------------- warmup
#
# Three cost tiers:
#   1. JIT compile (nvcc/Triton) of a kernel configuration: paid once
#      ever per (software stack, GPU, model, budget) by compile_kernels.
#      Cached in the library's configured kernel directories;
#      a marker file records that the pass ran.
#   2. Loading a cached binary into the process: paid once per process
#      by touch_kernels.
#   3. The launch itself: every call, unavoidable.
#
# Kernels key on token counts and model constants, never on token
# values, so both passes run on synthetic ids.

# Bump when either pass covers a different set of shapes. A bumped
# version invalidates every marker, so the next boot re-runs the
# compile pass and re-commits the cache.
WARMUP_VERSION = 4


def _warm_inputs(budget):
    """Synthetic warmup tokens: a 512-id document and a 16-id question suffix.

    Cycled to any length the passes need. Fixed small ids; only the
    counts matter to the kernels.
    """
    doc = [10 + (i % 500) for i in range(512)]
    question = list(range(10, 26))
    q_max = len(question)
    warm_docs, used = [], 0
    while used + len(doc) + q_max <= budget:
        warm_docs.append(doc)
        used += len(doc) + q_max
    return warm_docs or [doc[:max(8, budget - q_max)]], question, doc


def _forward_warm(torch, arena, pipeline, async_ans, budget, *,
                  join_chunk):
    """Run real forward passes over every attention path.

    Both modes with arena writes, plus the unpaged causal fast path.
    join_chunk adds one run_join call.
    """
    warm_docs, question, doc = _warm_inputs(budget)
    q_max = len(question)
    modes = ("unified", "tree") if pipeline.tree_attention else ("unified",)
    for mode in modes:
        logger.debug("kernels: warming %s attention, full chunk", mode)
        run_filter(torch, arena, pipeline, async_ans, warm_docs,
                   [question], budget, arena_writes=True,
                   attention_mode=mode)
        # small chunks, one document each: the trailing-chunk shapes
        # of gated multi-stage runs (see ModelPipeline.warm_tokens)
        for t in pipeline.warm_tokens:
            if t >= budget:
                continue
            logger.debug("kernels: warming %s attention, %s tokens", mode, t)
            body = (doc * (t // len(doc) + 1))[:max(8, t - q_max)]
            run_filter(torch, arena, pipeline, async_ans, [body],
                       [question], budget, arena_writes=True,
                       attention_mode=mode)
    if join_chunk:
        logger.debug("kernels: warming join forward pass")
        run_join(torch, arena, pipeline, async_ans, warm_docs,
                 [[question] * 8], budget)
    logger.debug("kernels: warming filter without KV writes")
    run_filter(torch, arena, pipeline, async_ans, warm_docs,
               [question], budget, arena_writes=False)


def compile_kernels(torch, arena, pipeline, async_ans, budget):
    """Build every DeepGEMM kernel configuration, then warm every path.

    Configurations go up to the budget; the warm pass runs forward
    passes over every attention-path shape.

    Runs once per (software stack, GPU, model, budget). Uses vLLM's
    config heuristic generator to enumerate every token count at
    which the chosen GEMM configuration changes.
    """
    if not pipeline.gemm_warmup:
        _forward_warm(torch, arena, pipeline, async_ans, budget,
                      join_chunk=True)
        return
    from vllm.model_executor.warmup.deep_gemm_warmup import (
        _generate_optimal_warmup_m_values,
    )
    linears = pipeline.linears()
    work = [(m, lin) for lin in linears
            for m in _generate_optimal_warmup_m_values(
                budget, lin.weight.shape[0], torch.device("cuda"))]
    progress = Progress("kernels: GEMM warmup", total=len(work),
                        unit="configurations", emit=logger.info)
    logger.info("kernels: starting %s GEMM warmup configurations", len(work))
    with torch.inference_mode():
        # one buffer per linear, row-sliced per call; the sweep only
        # needs each kernel launched once
        cur, buf = None, None
        for done, (m, lin) in enumerate(work, 1):
            if lin is not cur:
                buf = torch.randn(budget, lin.weight.shape[1],
                                  device="cuda",
                                  dtype=torch.bfloat16)
                cur = lin
            q, s = pipeline.engine.quant(buf[:m])
            pipeline.engine.gemm(q, s, lin)
            progress.update(done)
        buf = None
        torch.cuda.synchronize()
    progress.finish("kernels: GEMM warmup done")
    logger.info("kernels: warming filter and join forward passes")
    _forward_warm(torch, arena, pipeline, async_ans, budget,
                  join_chunk=True)


def touch_kernels(torch, arena, pipeline, async_ans, budget):
    """Run each hot kernel once per GPU process.

    Cached binaries then load at boot instead of mid-run.
    """
    _forward_warm(torch, arena, pipeline, async_ans, budget,
                  join_chunk=True)


def _marker_path(model_name, budget):
    import os
    root = os.path.expanduser(os.environ.get(
        "QUAIL_CACHE_DIR", "~/.cache/quail/kernels"))
    safe = model_name.replace("/", "--")
    return os.path.join(root, f"quail-warm-{safe}-{int(budget)}.json")


def _marker_identity(torch, model_name, budget):
    try:
        import vllm
        vllm_version = vllm.__version__
    except ImportError:
        vllm_version = None
    return dict(warmup_version=WARMUP_VERSION, model=model_name,
                budget=int(budget), vllm=vllm_version,
                torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name())


def warm_kernels(torch, arena, pipeline, async_ans, budget, *,
                 model_name, force_compile=False):
    """Compile once under a file lock, then warm each GPU process.

    Returns:
        A dict with the selected tier and elapsed warmup seconds.
    """
    import fcntl
    import json
    import os

    with quiet():
        path = _marker_path(model_name, budget)
        identity = _marker_identity(torch, model_name, budget)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        t0 = time.perf_counter()
        tier = "touch"
        # The marker check, compilation, and publication must be one
        # operation across processes sharing the kernel directory.
        with open(path + ".lock", "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logger.info("kernels: waiting for another worker to compile")
                fcntl.flock(lock, fcntl.LOCK_EX)
            wait_s = time.perf_counter() - t0
            on_disk = None
            try:
                with open(path) as f:
                    on_disk = json.load(f)
            except (OSError, ValueError):
                pass
            if force_compile or on_disk != identity:
                logger.info("kernels: starting compile pass")
                compile_kernels(torch, arena, pipeline, async_ans, budget)
                torch.cuda.synchronize()
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(identity, f, indent=1)
                os.replace(tmp, path)
                tier = "compile"
        if tier == "touch":
            logger.info("kernels: warming cached filter and join kernels")
            touch_kernels(torch, arena, pipeline, async_ans, budget)
        torch.cuda.synchronize()
        warm_s = round(time.perf_counter() - t0 - wait_s, 2)
        wait_s = round(wait_s, 2)
        logger.info("kernels: %s pass done in %s s; waited %s s for compilation",
                    tier, warm_s, wait_s)
        return dict(tier=tier, warm_s=warm_s, wait_s=wait_s)


# ---------------------------------------------------------- the filter

def _shared_preamble_tokens(question_ids):
    """Longest common token prefix across the stage questions."""
    if len(question_ids) < 2:
        return 0
    p = 0
    while all(len(q) > p and q[p] == question_ids[0][p]
              for q in question_ids):
        p += 1
    return p


def borrow_check(arena, keys, borrowing):
    """can_borrow for Borrowing: whether the arena can serve a borrow.

    A fresh parent admitted in the same chunk is not allocated yet;
    its sliding pages will start at the window origin below the tokens
    it borrows itself. A parent already resident, admitted in this
    chunk or earlier, has the pages it has: it may have trimmed its
    window in an earlier query.
    """
    def can_borrow(doc, parent, same_chunk):
        shared = borrowing.tree_shared[doc]
        if not same_chunk or arena.is_resident(keys[parent]):
            return arena.can_borrow(keys[parent], shared)
        return arena.origin(borrowing.shared(parent)) <= arena.origin(shared)
    return can_borrow


def lowest_borrows(borrowing):
    """Per document, the fewest tokens any of its borrowers shares."""
    lowest = {}
    for doc, parent in enumerate(borrowing.tree_parent):
        if parent is not None:
            shared = borrowing.tree_shared[doc]
            lowest[parent] = min(lowest.get(parent, shared), shared)
    return lowest


class FilterStream:
    """The filter chain, one chunk per next() call.

    Continuous admission with survivor priority. Pages are freed on
    FALSE or after the last stage. With hold_survivors, a document
    that passes its last stage keeps its pages pinned instead, and
    next() hands it over as a (key, prefix) pair for a join that reads
    its KV; that join frees or retains the key.

    Args:
        torch: The torch module, imported by the caller.
        arena: KVArena holding the documents' KV pages.
        pipeline: ModelPipeline that runs each packed forward chunk.
        async_ans: AsyncAnswers readout for TRUE/FALSE bits.
        doc_ids: Per-document token lists.
        question_ids: Per-stage question token lists.
        budget: Chunk token budget.
        timing: CPU seconds per loop phase accumulate into it.
        pinned: False for pageable blocking copies.
        limit: Stop admitting documents after this many survivors.
            None runs every document.
        arena_writes: Whether document KV is written to the arena.
            Must be True with multiple stages.
        arena_keys: Stable arena key for each document. List positions
            are used when omitted.
        retain_survivors: Passing document positions to keep for
            joins, through the shared arena retention pool capped at
            the arena minus the scan ring.
        hold_survivors: Hand passing documents over with their KV
            pinned instead of freeing or retaining them.
        hold_extra_tokens: Rows past the prefix a held document's
            pages must cover (the consumer's largest frame), so the
            consumer never needs a page this chain did not claim.
        attention_mode: "tree" or "unified" as the plan chose; None
            and a model without tree attention run unified.
        document_done: Called after each chunk's answers are read with
            the documents that finished in it, as a list of
            (position, stage, passed) tuples: the document's position
            in doc_ids, the last stage it was asked, and whether it
            passed that stage. Must return quickly.
        prefix_tree: A PrefixTree over doc_ids, or None. A document
            with a parent borrows the parent's pages for its shared
            prefix and packs only the tokens after it. Needs
            arena_writes. Under
            tree attention, documents borrowing the same pages in one
            chunk read them once.
    """

    def __init__(self, torch, arena, pipeline, async_ans, doc_ids,
                 question_ids, budget, timing=None, pinned=True,
                 limit=None, *, arena_writes, arena_keys=None,
                 retain_survivors=(), hold_survivors=False,
                 hold_extra_tokens=0, attention_mode=None,
                 document_done=None, prefix_tree=None):
        p = _shared_preamble_tokens(question_ids)
        keys = range(len(doc_ids)) if arena_keys is None else arena_keys
        if len(keys) != len(doc_ids):
            raise ValueError("arena_keys must match doc_ids")
        retain_all = retain_survivors is True
        retain = set() if retain_all else set(retain_survivors)
        if hold_survivors and (retain_all or retain):
            raise ValueError("held survivors are pinned, not retained")
        canvas = tuple(pipeline.canvas_ids)
        c = len(canvas)
        # a model whose attention reads paged KV only writes pages even
        # when the plan skipped them
        arena_writes = arena_writes or pipeline.needs_pages
        stage_tokens = [len(question_ids[0]) + c] \
            + [len(q) - p + c for q in question_ids[1:]]
        tails = [question_ids[0]] + [q[p:] for q in question_ids[1:]]
        for i, t in enumerate(tails):
            if not t:
                raise ValueError(
                    f"stage {i} question has no tokens beyond the shared "
                    f"preamble ({p} tokens)")
        if not arena_writes and (
            len(question_ids) > 1 or retain_all or retain or hold_survivors
        ):
            # a later stage re-reads the KV, which needs the pages this
            # switch skips
            raise ValueError("arena_writes=False needs a single stage")
        self.attention_mode = attention_path(pipeline, attention_mode)
        unified = self.attention_mode == "unified"
        if prefix_tree is not None and prefix_tree.shared_tokens:
            if not arena_writes:
                raise ValueError("a prefix tree needs arena writes")
        else:
            prefix_tree = None
        # capacity must cover the longest tail past the kept preamble,
        # and the canvas rows that follow it
        temp_tail = max(0, *(len(q) - p + c for q in question_ids)) \
            if unified and arena_writes else 0
        capacity_extra = p + temp_tail
        if hold_survivors:
            capacity_extra = max(capacity_extra, hold_extra_tokens)
        if arena.retention_cap_pages is None:
            arena.retention_cap_pages = max(
                0, arena.n_pages
                - arena.pages_needed(2 * budget))
        self.sched = FilterAdmission(
            [len(d) for d in doc_ids], stage_tokens, budget,
            arena_pages=(arena.n_pages if arena_writes
                         else None),
            page_tokens=arena.page_tokens,
            kept_extra_tokens=capacity_extra, limit=limit,
            page_cost=arena.page_cost, tree=prefix_tree)
        borrowing = self.sched.borrowing
        borrowing.can_borrow = borrow_check(arena, keys, borrowing)
        self.tree = prefix_tree
        # a document's place in tree order, for chunk order under
        # tree attention
        self.tree_position = ({doc: i for i, doc in enumerate(prefix_tree.order)}
                              if prefix_tree is not None else {})
        self.borrowers = borrowing.borrowers()
        self.lowest_borrow = lowest_borrows(borrowing)
        # pages a join consumer needs free to run its held survivors
        self.reserve_pages = 0
        self.torch = torch
        self.arena = arena
        self.pipeline = pipeline
        self.async_ans = async_ans
        self.doc_ids = doc_ids
        self.keys = keys
        self.timing = timing
        self.pinned = pinned
        self.arena_writes = arena_writes
        self.retain_all = retain_all
        self.retain = retain
        self.hold = hold_survivors
        self.preamble = p
        self.capacity_extra = capacity_extra
        self.tails = tails
        self.canvas = canvas
        self.answer_row = pipeline.canvas_answer_row
        self.spans = []
        self.tokens = 0
        self.chunks = 0
        self.done = False
        self.outstanding = []   # (groups, handle) in launch order
        self.progress = Progress(
            f"filter ({len(question_ids)} stages)", total=len(doc_ids))
        self._finished = 0
        self.document_done = document_done

    @property
    def answers(self):
        """Per document, its 0/1 answers up to the first FALSE."""
        return self.sched.answers

    def _spec(self, doc, stage, fresh):
        if fresh:
            borrowing = self.sched.borrowing
            shared = borrowing.shared(doc)
            prefix = self.doc_ids[doc]
            read_key = None
            if shared:
                prefix = prefix[shared:]
                read_key = self.keys[borrowing.parent(doc)]
            return dict(key=self.keys[doc], prefix=prefix, start=shared,
                        read_key=read_key,
                        f=len(self.doc_ids[doc]), suffixes=[self.tails[0]],
                        write_suffix_tokens=self.preamble)
        return dict(key=self.keys[doc], prefix=None,
                    f=len(self.doc_ids[doc]) + self.preamble,
                    suffixes=[self.tails[stage]])

    def set_reserve(self, pages):
        """Leave `pages` free for a join consumer's partner rows.

        Raises:
            ValueError: A document and its extra rows do not fit beside
                the reserve.
        """
        for doc, ids in enumerate(self.doc_ids):
            if (self.arena.page_cost(len(ids) + self.capacity_extra) + pages
                    > self.arena.n_pages):
                raise ValueError(
                    f"document {doc} needs more pages than the arena "
                    f"holds beside the join's {pages} pages of partner rows")
        self.reserve_pages = pages

    def _activate(self, doc):
        """Allocate a fresh document's pages, borrowing as admission chose.

        A document with borrowers is held for each of them, and keeps
        the sliding rows the lowest of them reads.
        """
        borrowing = self.sched.borrowing
        key = self.keys[doc]
        shared = borrowing.shared(doc)
        got = self.arena.activate(
            key, len(self.doc_ids[doc]) + self.preamble,
            capacity_tokens=len(self.doc_ids[doc]) + self.capacity_extra,
            base_tokens=len(self.doc_ids[doc]),
            borrow=((self.keys[borrowing.parent(doc)], shared)
                    if shared else None))
        assert got is not None, "scheduler admitted a doc the arena cannot hold"
        if self.borrowers[doc]:
            self.arena.hold(key, self.borrowers[doc])
            self.arena.keep_window(key, self.lowest_borrow[doc])

    def _report(self, entry, items):
        timing = self.timing
        t = time.perf_counter() if timing is not None else 0.0
        groups, handle = entry
        bits = self.async_ans.result(handle)
        t = _tick(timing, "report_wait", t)
        last_stage = len(self.tails) - 1
        finished = []
        for (doc, stage, _fresh), bit in zip(groups, bits):
            passed = bool(bit)
            last = stage == last_stage
            keep = passed and last and (self.retain_all
                                        or doc in self.retain)
            hold = passed and last and self.hold
            for d in self.sched.report(doc, stage, passed,
                                       release=not (keep or hold)):
                self.arena.free_key(self.keys[d])
            if keep:
                self.arena.retain(self.keys[doc], len(self.doc_ids[doc]))
            if hold:
                items.append((self.keys[doc], self.doc_ids[doc]))
            if not passed or last:
                self._finished += 1
                finished.append((doc, stage, passed))
        self.progress.update(self._finished)
        if finished and self.document_done is not None:
            self.document_done(finished)
        _tick(timing, "report_rest", t)

    def _finish(self, items):
        while self.outstanding:
            self._report(self.outstanding.pop(0), items)
        for doc in self.sched.drain_ready():
            self.arena.free_key(self.keys[doc])
        # a limit can end the run with borrowers never admitted
        self.arena.drop_holds([self.keys[doc] for doc, n
                               in enumerate(self.borrowers) if n])
        self.progress.finish(
            f"filter ({len(self.tails)} stages) done",
            f"{self.tokens:,} fresh tokens")
        self.done = True

    def next(self, evict_retained=False):
        """Run one chunk. Returns (held (key, prefix) pairs, blocked).

        Reads the previous chunk's answers while the new one runs.
        blocked means fresh admission is short of arena pages; with
        held survivors the consumer has to free some before the chain
        can admit again. evict_retained lets the chain evict retained
        KV of earlier operators for the shortfall instead of returning
        blocked, which a chain without held survivors always does.
        """
        items = []
        if self.done:
            return items, False
        sched = self.sched
        arena = self.arena
        timing = self.timing
        while True:
            if sched.done():
                self._finish(items)
                return items, False
            free = (max(0, arena.free_pages - self.reserve_pages)
                    if self.arena_writes else None)
            t = time.perf_counter() if timing is not None else 0.0
            groups = sched.next_chunk(free)
            t = _tick(timing, "next_chunk", t)
            if groups:
                break
            if self.outstanding:
                self._report(self.outstanding.pop(0), items)
                continue
            if sched.blocked_pages and (evict_retained or not self.hold):
                if arena.evict_retained(sched.blocked_pages):
                    continue
            if self.hold and sched.blocked_pages and not evict_retained:
                # the consumer frees pages before asking to evict
                return items, True
            # parents freed but held for queued children hold the
            # pages: the children compute their whole documents
            if arena.drop_holds([self.keys[doc] for doc, n
                                 in enumerate(self.borrowers) if n]):
                continue
            if self.hold and sched.blocked_pages:
                return items, True
            raise AssertionError("nothing buildable and nothing in flight")
        if self.tree is not None and self.attention_mode != "unified":
            # siblings back to back read their parent's pages once; a
            # parent sorts under its own parent's place, so before them
            def borrow_order(group):
                doc, _, fresh = group
                if not fresh:
                    return (False, -1, 0)
                parent = sched.borrowing.parent(doc)
                return (True, self.tree_position[
                    doc if parent is None else parent],
                    sched.borrowing.shared(doc))
            groups.sort(key=borrow_order)
        fresh_docs = [doc for doc, _, fresh in groups if fresh]
        if self.arena_writes:
            for doc in fresh_docs:
                self._activate(doc)
        t = _tick(timing, "alloc", t)
        chunk = pack_chunk(self.torch, arena,
                           [self._spec(*g) for g in groups],
                           timing=timing, pinned=self.pinned,
                           attention_mode=self.attention_mode,
                           canvas=self.canvas, answer_row=self.answer_row)
        # the packed children hold their parents' pages now
        for doc in fresh_docs:
            parent = sched.borrowing.tree_parent[doc]
            if parent is not None:
                arena.release(self.keys[parent])
        t = _tick(timing, "pack", t)
        self.tokens += chunk.tokens
        torch = self.torch
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        normed = _forward(self.pipeline, arena, chunk)
        e1.record()
        # a fresh document's sliding pages before its window origin
        # were for this pass only, except rows a borrower reads
        if self.arena_writes:
            for doc in fresh_docs:
                arena.trim_window(self.keys[doc])
        t = _tick(timing, "forward_launch", t)
        self.spans.append((0, e0, e1))
        self.outstanding.append((groups, self.async_ans.submit(normed)))
        self.chunks += 1
        t = _tick(timing, "submit", t)
        if timing is not None:
            timing["n_chunks"] = timing.get("n_chunks", 0) + 1
        # read the previous chunk's answers while this one runs
        while len(self.outstanding) > 1:
            self._report(self.outstanding.pop(0), items)
        return items, bool(self.hold and sched.blocked_pages)


def run_filter(torch, arena, pipeline, async_ans, doc_ids,
               question_ids, budget, timing=None,
               pinned=True, limit=None, *, arena_writes,
               arena_keys=None, retain_survivors=(), attention_mode=None,
               document_done=None, prefix_tree=None, stats=None):
    """The filter chain run to the end; see FilterStream for the arguments.

    stats, when given, receives borrowed_tokens: the document tokens
    read from a parent's KV pages instead of computed.

    Returns:
        (answers, spans, tokens): answers[d] = 0/1 list up to the
        first FALSE; spans and tokens as in run_join.
    """
    stream = FilterStream(
        torch, arena, pipeline, async_ans, doc_ids, question_ids, budget,
        timing=timing, pinned=pinned, limit=limit,
        arena_writes=arena_writes, arena_keys=arena_keys,
        retain_survivors=retain_survivors, attention_mode=attention_mode,
        document_done=document_done, prefix_tree=prefix_tree)
    while not stream.done:
        stream.next()
    if stats is not None:
        stats["borrowed_tokens"] = stream.sched.borrowing.borrowed_tokens
    return stream.answers, stream.spans, stream.tokens
