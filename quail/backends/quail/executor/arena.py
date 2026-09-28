"""Paged KV arena: preallocated per-layer buffers in fixed-size pages.

Pages are handed out from a free list and reached through block tables.

PageArena is the accounting (pure Python, CPU-tested); KVArena is the
tensor backing and runs only where torch and a GPU exist.

KVArena is the whole surface the chunk loop, the packer, the attention
paths, and the operator runtimes use. Everything they need is a method
or property on it; nothing outside this module reads the accounting
object or the K and V tensors directly. A replacement arena implements
the same methods:

- allocation: alloc, activate, alloc_temporary, free_key, pin, grow
- retention: retain, evict_retained, evict_key, configure_retention,
  retention_cap_pages, retained_keys, retained_pages,
  retained_prefix_tokens
- residency: is_resident, resident_keys, owned_pages, capacity_rows,
  pages_needed, page_cost, held_cost, page_tokens, n_pages, free_pages
- the sliding window: has_sliding, origin, sliding_start, trim_window,
  owned_sliding_pages, capacity_rows_sliding, n_sliding_pages, resize
- attention inputs: layer_kv, paged_kv, block_table, block_table_rows
- counters: reset_stats, evicted_keys, evicted_pages,
  evicted_prefix_tokens
"""

import heapq

import numpy as np

from quail.progress import answer_sink


class PageArena:
    """Page accounting: a free list and per-document page lists.

    A key may borrow the leading pages of another resident key's
    prefix: it reads them and never writes them. A page returns to
    the free list when its last holder frees it.
    """

    def __init__(self, n_pages: int, page_tokens: int):
        self.n_pages = n_pages
        self.page_tokens = page_tokens
        self.free = list(range(n_pages - 1, -1, -1))    # stack
        self.owned = {}       # key -> list of page ids it alone writes
        self.borrowed = {}    # key -> leading page ids read from a parent
        self.holds = [0] * n_pages    # page id -> keys holding it
        self.tokens = {}      # key -> resident token count, borrowed included
        self.pinned = set()    # current operators depend on these keys
        self._retained_sizes = {}
        self._retained_pages = 0
        self.retained = {}     # key -> reusable prefix tokens
        self._retained_heap = []
        self._retained_versions = {}
        self._retention_version = 0
        self.retention_policy = None
        self.retention_cap_pages = None

    def pages_needed(self, tokens: int) -> int:
        return -(-tokens // self.page_tokens)

    def alloc(self, key, tokens: int, capacity_tokens: int | None = None,
              borrowed=()):
        """The document's own pages, or None when the free list is short.

        The admission scheduler treats None as "wait". borrowed are
        resident pages of another key the new key reads as its
        leading pages (see shared_pages); tokens and capacity_tokens
        count them.
        """
        if key in self.owned:
            raise KeyError(f"{key!r} already resident")
        capacity_tokens = tokens if capacity_tokens is None else capacity_tokens
        if capacity_tokens < tokens:
            raise ValueError("capacity_tokens must cover logical tokens")
        borrowed = list(borrowed)
        if len(borrowed) * self.page_tokens > tokens:
            raise ValueError("a key cannot borrow past its own tokens")
        need = self.pages_needed(capacity_tokens) - len(borrowed)
        if need > len(self.free):
            return None
        pages = [self.free.pop() for _ in range(need)]
        self.owned[key] = pages
        self.borrowed[key] = borrowed
        self.tokens[key] = tokens
        for page in borrowed + pages:
            self.holds[page] += 1
        return pages

    def shared_pages(self, parent, shared: int, start: int = 0) -> list:
        """The parent's pages for its tokens [start, shared), to borrow.

        Both bounds are whole pages. The parent's first page must hold
        token `start` or an earlier one.
        """
        if (shared - start) % self.page_tokens or start % self.page_tokens:
            raise ValueError("a borrowed prefix is a whole number of pages")
        if parent not in self.owned:
            raise KeyError(f"borrow parent {parent!r} is not resident")
        if shared > self.tokens[parent]:
            raise ValueError(f"{parent!r} holds fewer tokens than borrowed")
        first = start // self.page_tokens
        return self.table_pages(parent)[first:shared // self.page_tokens]

    def drop_leading(self, key, pages: int) -> int:
        """Drop a key's first pages, borrowed ones first; returns pages freed."""
        borrowed = self.borrowed[key]
        from_borrowed = min(pages, len(borrowed))
        dropped = borrowed[:from_borrowed] + self.owned[key][:pages - from_borrowed]
        self.borrowed[key] = borrowed[from_borrowed:]
        self.owned[key] = self.owned[key][pages - from_borrowed:]
        return self.release_pages(dropped)

    def table_pages(self, key) -> list:
        """The pages a key's block table reads: borrowed, then its own."""
        return self.borrowed[key] + self.owned[key]

    def borrowed_tokens(self, key) -> int:
        return len(self.borrowed[key]) * self.page_tokens

    def release_pages(self, pages) -> int:
        """Drop one hold on each page; returns how many went free."""
        freed = 0
        for page in pages:
            self.holds[page] -= 1
            if not self.holds[page]:
                self.free.append(page)
                freed += 1
        return freed

    def free_key(self, key) -> int:
        """Drop a document's pages; returns the pages that went free."""
        pages = self.owned.pop(key)
        borrowed = self.borrowed.pop(key)
        self.tokens.pop(key)
        self.pinned.discard(key)
        self._forget_retained(key)
        return self.release_pages(borrowed + pages)

    def pin(self, key) -> None:
        """Protect a resident key while an operator uses it."""
        if key not in self.owned:
            raise KeyError(key)
        self._forget_retained(key)
        self.pinned.add(key)

    def _forget_retained(self, key):
        self._retained_pages -= self._retained_sizes.pop(key, 0)
        self.retained.pop(key, None)
        self._retained_versions.pop(key, None)

    def retain(self, key, priority=None) -> None:
        """Make a resident key evictable after its current use."""
        if key not in self.owned:
            raise KeyError(key)
        self.pinned.discard(key)
        prefix_tokens = self.tokens[key]
        self._forget_retained(key)
        # the pages evicting the key would free: those it alone holds
        # now; a child freed later leaves more, so the count is a
        # lower bound, and admission short of pages evicts anyway
        pages = sum(1 for page in self.table_pages(key) if self.holds[page] == 1)
        self.retained[key] = prefix_tokens
        self._retained_sizes[key] = pages
        self._retained_pages += pages
        self._retention_version += 1
        version = self._retention_version
        self._retained_versions[key] = version
        if priority is None:
            # a key whose pages its children all share frees none alone
            priority = (self.retention_policy.priority(
                key, prefix_tokens, max(1, pages))
                if self.retention_policy
                else (prefix_tokens / max(1, pages),))
        heapq.heappush(
            self._retained_heap,
            (priority, version, key, pages, prefix_tokens),
        )

    def configure_retention(self, policy, cap_pages: int) -> None:
        """Update priorities without making active prefixes evictable."""
        if not 0 <= cap_pages <= self.n_pages:
            raise ValueError("retention capacity must fit inside the arena")
        self.retention_policy = policy
        self.retention_cap_pages = cap_pages
        keys = list(self.retained)
        self._retained_heap.clear()
        for key in keys:
            self.retain(key)

    def pop_retained_victim(self, keep=frozenset()):
        """Remove the retained prefix with the lowest planned reuse priority.

        Keys in keep stay retained and are passed over.
        """
        passed = []
        victim = None
        while self._retained_heap:
            entry = heapq.heappop(self._retained_heap)
            _, version, key, pages, prefix_tokens = entry
            if self._retained_versions.get(key) != version:
                continue
            if key in keep:
                passed.append(entry)
                continue
            self._forget_retained(key)
            victim = key, pages, prefix_tokens
            break
        for entry in passed:
            heapq.heappush(self._retained_heap, entry)
        return victim

    def rewind(self, key, tokens: int) -> int:
        """Keep the first tokens and return unused trailing pages."""
        if key not in self.owned:
            raise KeyError(key)
        borrowed = self.borrowed_tokens(key)
        capacity = borrowed + len(self.owned[key]) * self.page_tokens
        if tokens < borrowed or tokens > capacity:
            raise ValueError("rewind tokens leave the resident capacity")
        keep = self.pages_needed(tokens) - len(self.borrowed[key])
        released = self.owned[key][keep:]
        self.owned[key] = self.owned[key][:keep]
        self.tokens[key] = tokens
        freed = self.release_pages(released)
        if key in self.retained:
            self.retain(key)
        return freed

    def grow(self, key, capacity_tokens: int) -> int | None:
        """Add pages for capacity_tokens without changing logical tokens."""
        if key not in self.owned:
            raise KeyError(key)
        need = (self.pages_needed(capacity_tokens) - len(self.borrowed[key])
                - len(self.owned[key]))
        if need <= 0:
            return 0
        if need > len(self.free):
            return None
        added = [self.free.pop() for _ in range(need)]
        self.owned[key].extend(added)
        for page in added:
            self.holds[page] += 1
        return need

    def row_indices(self, key, tokens=None):
        """Flat row positions of the document's tokens inside a pool.

        The pool is viewed as (n_pages * page_tokens, ...): page p holds
        rows p*page_tokens .. p*page_tokens + page_tokens.
        """
        rows = []
        left = self.tokens[key] if tokens is None else tokens
        for p in self.table_pages(key):
            take = min(left, self.page_tokens)
            base = p * self.page_tokens
            rows.extend(range(base, base + take))
            left -= take
        return rows

    @property
    def free_pages(self) -> int:
        return len(self.free)

    @property
    def resident_tokens(self) -> int:
        return sum(self.tokens.values())

    @property
    def retained_pages(self) -> int:
        return self._retained_pages

    @property
    def retained_prefix_tokens(self) -> int:
        return sum(self.tokens[key] for key in self.retained)


def _evict_event(key, prefix_tokens):
    """The answer-sink entry for one evicted document prefix.

    Returns None when the key is not an alias and a document index.
    """
    if not (isinstance(key, tuple) and len(key) == 2):
        return None
    alias, document = key
    if not isinstance(alias, str):
        return None
    try:
        document = int(document)
    except (TypeError, ValueError):
        return None
    return {"kind": "evict", "alias": alias, "document": document,
            "tokens": int(prefix_tokens)}


class KVArena:
    """The tensor backing for the arena, plus the accounting above.

    Per-layer K and V pools of shape (n_pages, page_tokens, n_kv,
    d_head). Torch is imported when the tensor pools are created.

    layer_kv gives each layer its own (n_kv, d_head) when the layers
    differ, as Gemma 4's sliding and full-attention layers do.

    Sliding layers (sliding_layers, with sliding_window tokens behind
    each row) keep a second, smaller pool of n_sliding_pages. A key
    holds pages in both: its whole prefix on the every-token pool, and
    on the sliding pool only the rows from its window origin on. The
    origin is the window's start below base_tokens (the document the
    key may be rewound to), rounded down to a page. A fresh key takes
    sliding pages for its whole prefix during its one pass, since the
    prefix rows attend to each other; trim_window then releases the
    pages before the origin. Page counts the schedulers see are in
    every-token pages, with sliding pages converted at the pools' size
    ratio, so a count that fits admits in both pools.
    """

    def __init__(self, n_layers: int, n_pages: int, page_tokens: int,
                 n_kv: int, d_head: int, dtype=None, device="cuda",
                 layer_kv=None, sliding_layers=(), sliding_window=0,
                 n_sliding_pages=0):
        import torch
        self.torch = torch
        self.dtype = dtype or torch.bfloat16
        self.device = device
        self.pinned = str(device).startswith("cuda")
        shapes = ([(n_kv, d_head)] * n_layers if layer_kv is None
                  else [tuple(shape) for shape in layer_kv])
        if len(shapes) != n_layers:
            raise ValueError(
                f"layer_kv names {len(shapes)} layers, the arena has "
                f"{n_layers}")
        self.shapes = shapes
        self.sliding_layers = (frozenset(sliding_layers) if sliding_window
                               else frozenset())
        self.window = sliding_window if self.sliding_layers else 0
        if self.sliding_layers and n_sliding_pages <= 0:
            raise ValueError("sliding layers need a sliding pool")
        self._build(n_pages, page_tokens, n_sliding_pages)
        self.reset_stats()

    def _build(self, n_pages, page_tokens, n_sliding_pages):
        torch = self.torch
        self.accounting = PageArena(n_pages, page_tokens)
        self.sliding = (PageArena(n_sliding_pages, page_tokens)
                        if self.sliding_layers else None)
        self.k, self.v = [], []
        for layer, (heads, dim) in enumerate(self.shapes):
            pages = n_sliding_pages if layer in self.sliding_layers else n_pages
            shape = (pages * page_tokens, heads, dim)
            self.k.append(torch.empty(shape, dtype=self.dtype, device=self.device))
            self.v.append(torch.empty(shape, dtype=self.dtype, device=self.device))
        # row indices stay on the host: pageable H2D copies block the
        # CPU behind the running stream
        self._rows = {}       # key -> row-index tensor on CPU
        self._capacity_rows = {}  # key -> every row in the claimed pages
        self._sliding_rows = {}   # key -> every row in its sliding pages
        self._base = {}       # key -> tokens the window is anchored below
        self._sliding_start = {}  # key -> logical row of its first sliding page
        self._holds = {}      # key -> queued keys that will borrow from it
        self._deferred = set()    # held keys freed by their operator
        self._window_floor = {}   # key -> lowest sliding row a borrower reads
        self._trimmed = set()     # keys past their own pass's trim_window

    def resize(self, n_pages: int, n_sliding_pages: int = 0, *,
               free_resident: bool = False) -> None:
        """Rebuild both pools at new sizes.

        Nothing survives a rebuild; free_resident frees every resident
        key first, and without it a resident key is an error.
        """
        if free_resident:
            self.drop_holds()
            for key in self.resident_keys():
                self.free_key(key)
        if self.accounting.owned:
            raise RuntimeError("the arena holds keys; free them before resizing")
        if (n_pages, n_sliding_pages) == (self.n_pages, self.n_sliding_pages):
            return
        page_tokens = self.page_tokens
        self.k = self.v = None
        self._build(n_pages, page_tokens, n_sliding_pages)

    def reset_stats(self):
        self.evicted_keys = 0
        self.evicted_pages = 0
        self.evicted_prefix_tokens = 0

    # ---- the window --------------------------------------------------

    @property
    def has_sliding(self) -> bool:
        return self.sliding is not None

    def origin(self, base_tokens: int) -> int:
        """The logical row a key's sliding pages start at once trimmed."""
        if not self.window or base_tokens <= self.window:
            return 0
        return (base_tokens - self.window) // self.page_tokens * self.page_tokens

    def sliding_start(self, key) -> int:
        """The logical row of the key's first sliding page."""
        return self._sliding_start[key]

    def keep_window(self, key, shared: int) -> None:
        """Keep the key's sliding rows a borrower of `shared` tokens reads.

        trim_window never drops rows from the window origin below
        `shared` on.
        """
        row = self.origin(shared)
        self._window_floor[key] = min(self._window_floor.get(key, row), row)

    def trim_window(self, key) -> int:
        """Release a key's sliding pages before its window origin.

        Rows a borrower still reads (see keep_window) stay. Returns the
        pages freed, in every-token pages.
        """
        if self.sliding is None:
            return 0
        self._trimmed.add(key)
        origin = min(self.origin(self._base[key]),
                     self._window_floor.get(key, self._base[key]))
        start = self._sliding_start[key]
        if origin <= start:
            return 0
        before = self.free_pages
        drop = (origin - start) // self.page_tokens
        self.sliding.drop_leading(key, drop)
        self.sliding.tokens[key] -= origin - start
        self._sliding_start[key] = origin
        self._refresh_rows(key)
        return self.free_pages - before

    def _lift_floor(self, key) -> None:
        """The last borrower is admitted: rows kept for it may go.

        A key that already trimmed for its own pass trims again; one
        still to run trims after that pass.
        """
        if self._window_floor.pop(key, None) is not None and key in self._trimmed:
            self.trim_window(key)

    # ---- allocation ----------------------------------------------------

    def alloc(self, key, tokens: int, capacity_tokens: int | None = None,
              base_tokens: int | None = None, sliding_tokens=None,
              borrow=None):
        """Pages for a fresh key in both pools, or None when either is short.

        sliding_tokens sizes the sliding pages when they differ from
        capacity_tokens, as a temporary's do. borrow is (parent key,
        shared tokens): the key reads the parent's pages for its first
        `shared tokens`, a whole number of pages. On the sliding pool
        it reads the parent's pages from the window origin below the
        shared prefix, so the parent must still hold them (see
        trim_window).
        """
        capacity = tokens if capacity_tokens is None else capacity_tokens
        borrowed, borrowed_s, start_s = self._borrow_plan(borrow)
        pages = self.accounting.alloc(key, tokens, capacity, borrowed=borrowed)
        if pages is None:
            return None
        if self.sliding is not None:
            want = capacity if sliding_tokens is None else sliding_tokens
            got = self.sliding.alloc(key, min(tokens, want) - start_s,
                                     want - start_s, borrowed=borrowed_s)
            if got is None:
                self.accounting.free_key(key)
                return None
        self._base[key] = tokens if base_tokens is None else base_tokens
        self._sliding_start[key] = start_s
        # the logical rows are the first `tokens` entries of the
        # capacity rows (same pages, same order), so build once and
        # slice instead of walking the pages twice
        self._refresh_rows(key, tokens)
        return pages

    def can_borrow(self, parent, shared: int) -> bool:
        """Whether a fresh key may borrow the parent's first `shared` tokens.

        The sliding pool must still hold the parent's rows from the
        window origin below the shared prefix: a parent that borrowed
        most of its own prefix never had them, and a trimmed one
        dropped them.
        """
        if (parent not in self.accounting.owned
                or shared > self.accounting.tokens[parent]):
            return False
        if self.sliding is None:
            return True
        return self._sliding_start[parent] <= self.origin(shared)

    def _borrow_plan(self, borrow):
        """The pages a borrow takes from each pool, and the sliding start.

        Returns (full pool pages, sliding pool pages, sliding start).
        The sliding start is the window origin below the shared
        tokens: the rows a sliding layer reads before the key's own.
        """
        if borrow is None:
            return [], [], 0
        parent, shared = borrow
        borrowed = self.accounting.shared_pages(parent, shared)
        if self.sliding is None:
            return borrowed, [], 0
        start = self.origin(shared)
        parent_start = self._sliding_start[parent]
        if parent_start > start:
            raise ValueError(
                f"{parent!r} trimmed its window past row {start}; "
                f"a child sharing {shared} tokens needs it")
        borrowed_s = self.sliding.shared_pages(
            parent, shared - parent_start, start - parent_start)
        return borrowed, borrowed_s, start

    def _page_rows(self, arena, key, tokens):
        pages = np.asarray(arena.table_pages(key), dtype=np.int64)
        rows = (pages[:, None] * self.page_tokens
                + np.arange(self.page_tokens, dtype=np.int64))
        return self.torch.from_numpy(rows.reshape(-1)[:tokens])

    def _refresh_rows(self, key, logical_tokens=None):
        logical = (self.accounting.tokens[key]
                   if logical_tokens is None else logical_tokens)
        capacity = len(self.accounting.table_pages(key)) * self.page_tokens
        cap = self._capacity_rows.get(key)
        if cap is None or cap.numel() != capacity:
            cap = self._page_rows(self.accounting, key, capacity)
        self._capacity_rows[key] = cap
        self._rows[key] = cap[:logical]
        if self.sliding is not None:
            capacity_s = len(self.sliding.table_pages(key)) * self.page_tokens
            self._sliding_rows[key] = self._page_rows(self.sliding, key, capacity_s)

    def pin(self, key):
        self.accounting.pin(key)

    def retain(self, key, tokens: int, priority=None):
        """Rewind a prefix and make it available for a later operator.

        Retention decisions are keyed by (alias, document), so only
        such pairs may be retained; score and temporary keys cannot.
        A key never rewinds below the base its window was anchored at.
        """
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError(f"retained keys are (alias, document) pairs, "
                            f"got {key!r}")
        if self.sliding is not None and tokens < self._base[key]:
            raise ValueError(
                f"{key!r}: rewinding to {tokens} tokens drops rows of "
                f"the sliding window anchored at {self._base[key]}")
        self.accounting.rewind(key, tokens)
        if self.sliding is not None:
            self.sliding.rewind(key, tokens - self._sliding_start[key])
        self._refresh_rows(key, tokens)
        self.accounting.retain(key, priority)
        cap = self.accounting.retention_cap_pages
        before = self.free_pages
        if cap is not None:
            self.evict_retained(max(0, self.accounting.retained_pages - cap))
        return self.free_pages - before

    def evict_retained(self, pages_needed: int) -> tuple:
        """Evict retained prefixes in planned reuse priority order.

        Stops once pages_needed have come free, counted in every-token
        pages. Returns the evicted keys.
        """
        start = self.free_pages
        return self._evict_until(
            lambda: self.free_pages - start >= pages_needed)

    def _evict_until(self, satisfied) -> tuple:
        """Evict retained prefixes in priority order until satisfied() holds.

        A held key is never evicted: a queued borrower will read it.
        """
        keys = []
        while not satisfied():
            victim = self.accounting.pop_retained_victim(keep=self._holds)
            if victim is None:
                break
            keys.append(victim[0])
            self.evict_key(victim[0])
        return tuple(keys)

    def _evict_for_pages(self, need, need_sliding):
        """Evict retained prefixes until both pools have the pages."""
        self._evict_until(lambda: (
            self.accounting.free_pages >= need
            and (self.sliding is None
                 or self.sliding.free_pages >= need_sliding)))

    def evict_key(self, key):
        """Free one retained prefix, record the lost KV, and report it.

        When an answer sink is listening, it is called with kind
        ``evict``, the alias, the document index, and the prefix
        length in tokens. Keys that are not an alias and a document
        index are not reported.
        """
        prefix_tokens = self.accounting.tokens[key]
        pages = self.free_key(key)
        self.evicted_keys += 1
        self.evicted_pages += pages
        self.evicted_prefix_tokens += prefix_tokens
        sink = answer_sink()
        if sink is not None:
            event = _evict_event(key, prefix_tokens)
            if event is not None:
                sink(event)
        return pages

    def activate(self, key, tokens: int, capacity_tokens: int | None = None,
                 base_tokens: int | None = None, borrow=None):
        """Make a prefix active, evicting retained KV when required.

        base_tokens anchors a fresh key's sliding window; a resident
        key keeps the base it was given. borrow is (parent key, shared
        tokens) for a fresh key, as alloc takes it.
        """
        capacity = tokens if capacity_tokens is None else capacity_tokens
        if key in self.accounting.owned:
            self.accounting.pin(key)
            need = max(0, self.accounting.pages_needed(capacity)
                       - len(self.accounting.table_pages(key)))
            capacity_s = capacity - self._sliding_start[key]
            need_s = (max(0, self.sliding.pages_needed(capacity_s)
                          - len(self.sliding.table_pages(key)))
                      if self.sliding is not None else 0)
            self._evict_for_pages(need, need_s)
            # neither pool grows unless both can, so a refusal leaves
            # the key as it was
            if need > self.accounting.free_pages or (
                    self.sliding is not None
                    and need_s > self.sliding.free_pages):
                return None
            self.accounting.grow(key, capacity)
            if self.sliding is not None:
                self.sliding.grow(key, capacity_s)
            self.accounting.tokens[key] = tokens
            if self.sliding is not None:
                self.sliding.tokens[key] = tokens - self._sliding_start[key]
            self._refresh_rows(key, tokens)
            return self.accounting.owned[key]

        borrowed, borrowed_s, start_s = self._borrow_plan(borrow)
        need = self.accounting.pages_needed(capacity) - len(borrowed)
        need_s = (self.sliding.pages_needed(capacity - start_s) - len(borrowed_s)
                  if self.sliding is not None else need)
        self._evict_for_pages(need, need_s)
        pages = self.alloc(key, tokens, capacity, base_tokens, borrow=borrow)
        if pages is not None:
            self.accounting.pin(key)
        return pages

    def alloc_temporary(self, tokens: int, sliding_tokens: int | None = None):
        key = object()
        pages = self.alloc(key, tokens, base_tokens=0,
                           sliding_tokens=sliding_tokens)
        return None if pages is None else (key, pages)

    # ---- holds: keys that queued borrowers will read ------------------

    def hold(self, key, borrowers: int) -> None:
        """Keep a key for `borrowers` keys that will borrow its pages.

        A held key is never evicted, and free_key on it waits until the
        last borrower has released it.
        """
        if borrowers:
            self._holds[key] = self._holds.get(key, 0) + borrowers

    def release(self, key) -> None:
        """One borrower of the key was admitted, or will not come."""
        if key not in self._holds:
            return
        self._holds[key] -= 1
        if not self._holds[key]:
            del self._holds[key]
            if key in self._deferred:
                self._deferred.discard(key)
                self.free_key(key)
            else:
                self._lift_floor(key)

    def drop_holds(self, keys=None) -> bool:
        """Release every hold on the keys (all keys by default).

        Held keys already freed go free now, and retained ones become
        evictable; their queued borrowers compute their whole prefix
        instead. Returns whether any hold was released.
        """
        keys = list(self._holds) if keys is None else [
            key for key in keys if key in self._holds]
        for key in keys:
            del self._holds[key]
            if key in self._deferred:
                self._deferred.discard(key)
                self.free_key(key)
            else:
                self._lift_floor(key)
        return bool(keys)

    def free_key(self, key):
        """Free a key's pages, or once its holds are released.

        Returns the pages that went free now.
        """
        if key in self._holds:
            self._deferred.add(key)
            return 0
        self._window_floor.pop(key, None)
        self._trimmed.discard(key)
        self._rows.pop(key)
        self._capacity_rows.pop(key)
        self._sliding_rows.pop(key, None)
        self._base.pop(key, None)
        self._sliding_start.pop(key, None)
        if self.sliding is not None:
            self.sliding.free_key(key)
        return self.accounting.free_key(key)

    # ---- residency and retention, read by the loop and operators ------

    @property
    def page_tokens(self) -> int:
        return self.accounting.page_tokens

    @property
    def n_pages(self) -> int:
        return self.accounting.n_pages

    @property
    def n_sliding_pages(self) -> int:
        return 0 if self.sliding is None else self.sliding.n_pages

    @property
    def free_pages(self) -> int:
        """Free pages in every-token pages: the tighter pool decides."""
        free = self.accounting.free_pages
        if self.sliding is None:
            return free
        return min(free, self.sliding.free_pages * self.accounting.n_pages
                   // self.sliding.n_pages)

    def page_cost(self, tokens: int, base_tokens: int | None = None) -> int:
        """Pages a key of `tokens` rows takes, in every-token pages.

        base_tokens prices the trimmed key: only the rows from its
        window origin on sit on the sliding pool. Without it the key is
        priced untrimmed, as a fresh key is admitted.
        """
        pages = self.accounting.pages_needed(tokens)
        if self.sliding is None:
            return pages
        origin = 0 if base_tokens is None else self.origin(base_tokens)
        pages_s = self.sliding.pages_needed(max(0, tokens - origin))
        return max(pages, self._as_every_token_pages(pages_s))

    def held_cost(self, key) -> int:
        """Pages a resident key holds, in every-token pages."""
        pages = len(self.accounting.owned[key])
        if self.sliding is None:
            return pages
        return max(pages, self._as_every_token_pages(
            len(self.sliding.owned[key])))

    def growth_cost(self, key, capacity_tokens: int) -> int:
        """Pages a resident key needs to grow to capacity, in every-token pages.

        The same counts activate takes for a resident key: each pool's
        pages beyond the ones the key's table already has, the sliding
        pool's from the key's window start.
        """
        need = max(0, self.accounting.pages_needed(capacity_tokens)
                   - len(self.accounting.table_pages(key)))
        if self.sliding is None:
            return need
        need_s = max(0, self.sliding.pages_needed(
            capacity_tokens - self._sliding_start[key])
            - len(self.sliding.table_pages(key)))
        return max(need, self._as_every_token_pages(need_s))

    def _as_every_token_pages(self, sliding_pages: int) -> int:
        """Sliding pages converted at the pools' size ratio, rounded up."""
        return -(-sliding_pages * self.accounting.n_pages
                 // self.sliding.n_pages)

    @property
    def retained_pages(self) -> int:
        return self.accounting.retained_pages

    @property
    def retained_prefix_tokens(self) -> int:
        return self.accounting.retained_prefix_tokens

    @property
    def retention_cap_pages(self):
        return self.accounting.retention_cap_pages

    @retention_cap_pages.setter
    def retention_cap_pages(self, pages) -> None:
        self.accounting.retention_cap_pages = pages

    def pages_needed(self, tokens: int) -> int:
        return self.accounting.pages_needed(tokens)

    def configure_retention(self, policy, cap_pages: int) -> None:
        """Set the eviction priority rule and the retained-page cap."""
        self.accounting.configure_retention(policy, cap_pages)

    def is_resident(self, key) -> bool:
        return key in self.accounting.owned

    def resident_keys(self) -> list:
        """Every key holding pages, in allocation order."""
        return list(self.accounting.owned)

    def retained_keys(self) -> list:
        """Every key whose KV is kept for a later operator."""
        return list(self.accounting.retained)

    def owned_pages(self, key) -> list:
        """The page ids a resident key's block table reads, in logical order.

        Borrowed pages come first, then the key's own.
        """
        return self.accounting.table_pages(key)

    def owned_sliding_pages(self, key) -> list:
        """The sliding-pool page ids a key's block table reads, in logical order.

        Borrowed pages come first, then the key's own.
        """
        return self.sliding.table_pages(key)

    def capacity_rows(self, key):
        """Row index tensor (CPU) over every row in the key's pages."""
        return self._capacity_rows[key]

    def capacity_rows_sliding(self, key):
        """Row index tensor (CPU) over every row in the key's sliding pages."""
        return self._sliding_rows[key]

    # ---- tensors, read by the attention paths ------------------------

    def layer_kv(self, layer: int):
        """Flat K and V pools of one layer, shape (rows, n_kv, d_head)."""
        return self.k[layer], self.v[layer]

    def paged_kv(self, layer: int):
        """Pools as (n_pages, page_tokens, n_kv, d_head) for paged attention."""
        n_kv, d = self.k[layer].shape[-2], self.k[layer].shape[-1]
        shape = (-1, self.page_tokens, n_kv, d)
        return self.k[layer].view(shape), self.v[layer].view(shape)

    def block_table(self, keys, pad_to=None):
        """Block table and seqused_k for groups reading these documents' KV.

        Built flat on the host and staged through pinned memory.
        """
        pages = [self.accounting.table_pages(k) for k in keys]
        table = self.block_table_rows(pages, pad_to=pad_to)
        torch = self.torch
        used = torch.tensor([self.accounting.tokens[k] for k in keys],
                            dtype=torch.int32, pin_memory=self.pinned) \
            .to(self.device, non_blocking=self.pinned)
        return table, used

    def block_table_rows(self, pages, pad_to=None):
        """A block table from explicit physical page rows."""
        torch = self.torch
        width = max(len(p) for p in pages)
        if pad_to:
            width = max(width, pad_to)
        flat = []
        for p in pages:
            flat.extend(p)
            flat.extend([0] * (width - len(p)))
        table = torch.tensor(flat, dtype=torch.int32,
                             pin_memory=self.pinned).view(len(pages), width) \
            .to(self.device, non_blocking=self.pinned)
        return table
