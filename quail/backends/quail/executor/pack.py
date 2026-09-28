"""Chunk packing and admission - no GPU, no torch, unit-tested.

- JoinAdmission: continuous anchor admission for joins. Continuing
  partner streams pack first, then anchors starting their next stage,
  then fresh anchors whose pages fit the free list.
- FilterAdmission: continuous admission for filter chains. Survivors
  pack before fresh admissions; fresh documents admit when their
  page-rounded tokens fit the free list.

Length units are tokens. Suffixes are atomic and never split across
chunks.
"""

from array import array
from bisect import bisect_right
from collections import deque

import numpy as np


def orient(mean_left_tokens, mean_right_tokens):
    """Which side anchors: the longer one.

    Anchor tokens are paid once per document; partner tokens are paid
    once per pair.
    """
    return "left" if mean_left_tokens >= mean_right_tokens else "right"


def gate(answer_rows):
    """Anchors that survive a conjunctive stage: any TRUE in the row.

    answer_rows: dict anchor_index -> iterable of 0/1 answers.
    Returns the sorted surviving anchor indices.
    """
    return sorted(a for a, row in answer_rows.items() if any(row))


def matches(answer_rows):
    """anchor_index -> sorted list of partner indices answered TRUE."""
    return {a: sorted(i for i, v in enumerate(row) if v)
            for a, row in answer_rows.items()}


def assemble(ans1_rows, ans2_rows):
    """Output triples from two pairwise join stages sharing one anchor.

    ans1_rows: b -> row of 0/1 over A (stage 1, anchored on b).
    ans2_rows: b -> row of 0/1 over C, present only for gated
    survivors.
    """
    m1, m2 = matches(ans1_rows), matches(ans2_rows)
    out = []
    for b in sorted(ans2_rows):
        for a in m1.get(b, ()):
            for c in m2[b]:
                out.append((a, b, c))
    return sorted(out)


def brute_force_triples(ans1_rows, ans2_rows):
    """Nested-loop reference over recorded answers without gating."""
    out = []
    for b, row1 in sorted(ans1_rows.items()):
        for a, v1 in enumerate(row1):
            if v1:
                for c, v2 in enumerate(ans2_rows[b]):
                    if v2:
                        out.append((a, b, c))
    return sorted(out)


# --------------------------------------------- continuous admission


class _CompactQueue:
    """Keep ordered document positions in ranges or integer arrays."""

    def __init__(self, count: int):
        self._pieces = deque()
        self._length = 0
        if count:
            self._pieces.append([range(count), 0])
            self._length = count

    def __bool__(self):
        return bool(self._length)

    def popleft(self) -> int:
        if not self._pieces:
            raise IndexError("pop from an empty document queue")
        values, position = self._pieces[0]
        value = values[position]
        position += 1
        self._length -= 1
        if position == len(values):
            self._pieces.popleft()
        else:
            self._pieces[0][1] = position
        return value

    def appendleft(self, value: int) -> None:
        self.prepend(array("I", [value]))

    def prepend(self, values) -> None:
        if values:
            self._pieces.appendleft([values, 0])
            self._length += len(values)

def pages_for(tokens: int, page_tokens: int) -> int:
    """Pages needed for `tokens` rows (suffix KV is never paged)."""
    return -(-tokens // page_tokens)


# --------------------------------------------- join admission

_DONE = -2      # the anchor's last stage answered, or it failed a gate


def partner_pages(page_cost, page_tokens, prefix_tokens, frame_tokens,
                  suffix_rows):
    """Temporary pages one partner's rows take on the unified path.

    They follow the anchor's last partial page, whose rows are copied
    in first.
    """
    return page_cost((prefix_tokens + frame_tokens) % page_tokens
                     + suffix_rows, 0)


class Borrowing:
    """Which parent each document borrows its KV prefix from, as admitted.

    The prefix tree names each document's parent and shared tokens.
    decide() makes the choice when the document is admitted: wait
    while the parent is still queued, borrow the parent's first shared
    tokens, or pack the whole document when the arena cannot serve the
    borrow (the parent has left, or trimmed its window below the
    shared tokens) or the parent never runs (see skip()). record()
    keeps the choice. Filter and join admission both use it.

    Args:
        n: Documents.
        tree: A PrefixTree over the documents, or None.
        can_borrow: Callable(doc, parent, same_chunk) -> bool, asked
            before a borrow; same_chunk says the parent is admitted in
            the chunk being built. None allows every borrow.
    """

    WAIT = "wait"

    def __init__(self, n, tree=None, can_borrow=None):
        self.tree_parent = [None] * n if tree is None else list(tree.parent)
        self.tree_shared = [0] * n if tree is None else list(tree.shared)
        self.can_borrow = can_borrow
        self.admitted = set()
        self.skipped = set()       # documents that never run a chunk
        self.chosen = {}           # admitted borrower -> (parent, shared)
        self.borrowed_tokens = 0   # prefix tokens read from a parent

    def add(self):
        """A document streamed in after construction; it borrows nothing."""
        self.tree_parent.append(None)
        self.tree_shared.append(0)
        return len(self.tree_parent) - 1

    def detach(self, doc):
        """The document's KV is already resident, so it borrows nothing."""
        self.tree_parent[doc] = None
        self.tree_shared[doc] = 0

    def skip(self, doc):
        """The document never runs a chunk, so its borrowers pack whole."""
        self.skipped.add(doc)

    def borrowers(self):
        """Per document, how many documents the tree has borrow from it."""
        counts = [0] * len(self.tree_parent)
        for parent in self.tree_parent:
            if parent is not None:
                counts[parent] += 1
        return counts

    def decide(self, doc, chunk):
        """WAIT, None to pack the whole document, or (parent, shared)."""
        parent = self.tree_parent[doc]
        if parent is None or parent in self.skipped:
            return None
        if parent not in self.admitted:
            return self.WAIT
        if self.can_borrow is not None and not self.can_borrow(
                doc, parent, parent in chunk):
            return None
        return parent, self.tree_shared[doc]

    def record(self, doc, choice):
        """The document was admitted with decide()'s choice."""
        self.admitted.add(doc)
        if choice is not None:
            self.chosen[doc] = choice
            self.borrowed_tokens += choice[1]

    def parent(self, doc):
        """The parent an admitted document borrows from, or None."""
        return self.chosen.get(doc, (None, 0))[0]

    def shared(self, doc):
        """The tokens an admitted document borrows; 0 when it packs whole."""
        return self.chosen.get(doc, (None, 0))[1]


class JoinAdmission:
    """Join scheduler: continuous anchor admission, stages mixed in one chunk.

    Args:
        prefix_tokens: Per-anchor prefix token counts.
        stage_suffixes: Per stage, the partner suffix token counts of
            every partner tuple at that stage.
        chunk_budget: Tokens per forward pass.
        arena_pages: Pages in the KV arena.
        page_tokens: Tokens per arena page.
        frame_tokens: Per stage, tokens of the framing written into
            the anchor's KV ahead of its first suffix.
        resident: anchor -> pages already held. A resident anchor's
            prefix KV is in the arena, so it packs no prefix tokens.
        anchor_partners: anchor -> per stage, the indices into that
            stage's partner list the anchor streams, or None for the
            whole list. Omitted anchors stream the whole list at every
            stage.
        canvas_tokens: Rows a diffusion model adds after every suffix
            and after a frame entry. They take chunk room and page
            room but are never kept in the anchor's KV.
        page_cost: Callable(tokens, base_tokens) giving the pages a
            key of that many rows takes, in the arena's every-token pages; None
            prices one pool of page_tokens pages.
        tree: A PrefixTree over the anchors given at construction, or
            None. Fresh anchors are admitted in tree order; one whose
            parent is still queued waits, one whose parent's pages can
            be borrowed packs and pays for only the tokens past its
            shared length, and one whose parent has left packs whole.
        can_borrow: Callable(anchor, parent, same_chunk) -> bool, asked
            before a borrow; same_chunk says the parent is admitted in
            the chunk being built. None allows every borrow.

    Each chunk fills in priority order: partner streams cut by the
    previous chunk, then anchors starting their next stage, then
    fresh anchors whose pages fit the free list. Pages are granted in
    queue order; chunk room may be skipped. An anchor advances to its
    next stage once its whole stream at the current stage is launched
    and one partner has answered TRUE; the remaining answers fill in
    while the next stage runs. It drops out when every partner
    answered FALSE. An anchor with no partner at a stage settles
    without a chunk: finished with an empty row at the last stage,
    dropped at any earlier one.
    """

    def __init__(self, prefix_tokens, stage_suffixes, chunk_budget,
                 arena_pages, page_tokens, frame_tokens=None,
                 resident=None, anchor_partners=None, temporary_suffix_pages=False,
                 answer_dtype=None, canvas_tokens=0, page_cost=None,
                 tree=None, can_borrow=None):
        self.answer_dtype = answer_dtype
        n = len(prefix_tokens)
        self.borrowing = Borrowing(n, tree, can_borrow)
        # page_cost(tokens, base_tokens) prices a key in the arena's
        # every-token pages; the default is one pool of page_tokens pages
        self.page_cost = page_cost or (
            lambda tokens, base_tokens=None: pages_for(tokens, page_tokens))
        self._answer_counts = [{} for _ in stage_suffixes]
        self.temporary_suffix_pages = temporary_suffix_pages
        self._page_cums = {}
        self._page_reserve = 0
        self.prefix = []
        self.stages = [[t + canvas_tokens for t in s] for s in stage_suffixes]
        # frames: the rows a frame entry keeps in the anchor's KV;
        # frame_rows: the rows it packs, canvas included
        self.frames = (list(frame_tokens) if frame_tokens
                       else [0] * len(self.stages))
        if len(self.frames) != len(self.stages):
            raise ValueError("frame_tokens must match stage_suffixes")
        self.frame_rows = [f + canvas_tokens if f else 0 for f in self.frames]
        if not self.stages:
            raise ValueError("a join needs at least one stage")
        self.chunk_budget = chunk_budget
        self.arena_pages = arena_pages
        self.page_tokens = page_tokens
        k = len(self.stages)
        self._cum = []
        for j, (lens, frame) in enumerate(zip(self.stages, self.frame_rows)):
            cum = [0]
            for t in lens:
                cum.append(cum[-1] + t)
            self._cum.append(cum)
            if lens and frame + max(lens) > chunk_budget:
                raise ValueError(
                    f"stage {j}: a {frame + max(lens)}-token partner "
                    f"exceeds the {chunk_budget}-token chunk budget "
                    f"(suffixes are atomic)")
        self._extra = max(self.frame_rows)
        self._lists = []           # per anchor, per stage: indices or None
        self._cums = []            # per anchor, per stage: cumulative sums
        self._page_cost = []
        self._carried = []
        self._first = []           # per anchor: its first partner's cost
        self._min_fresh = float("inf")
        self._zero_cost = 0
        self.pending = deque()
        self.ready = deque()       # placed anchors with partners left
        self._stage = []           # current stage, or _DONE
        self._next = []            # next partner index at the stage
        self._true = []
        self._settled = []         # events for anchors that never ran
        self.in_flight = 0         # launched groups not yet reported
        self.blocked_pages = 0
        self.answers = [dict() for _ in range(k)]
        resident = dict(resident or {})
        anchor_partners = dict(anchor_partners or {})
        for a, prefix in enumerate(prefix_tokens):
            self._register(prefix, resident.get(a), anchor_partners.get(a))
        # resident anchors first: they cost no prefix tokens and few
        # or no pages, so they never wait behind a page-blocked anchor
        n = len(self.prefix)
        fresh_order = range(n) if tree is None else tree.order
        self.pending.extend(a for a in range(n)
                            if a in resident and self._stage[a] == -1)
        self.pending.extend(a for a in fresh_order
                            if a not in resident and self._stage[a] == -1)
        for a in resident:
            self.borrowing.detach(a)


    def _count(self, a, j):
        lst = self._lists[a][j]
        return len(self.stages[j]) if lst is None else len(lst)

    def _cum_of(self, a, j):
        return self._cum[j] if self._lists[a][j] is None else self._cums[a][j]

    def partner_indices(self, a, j, start, end):
        """Indices into stage j's partner list for one launched group."""
        lst = self._lists[a][j]
        return list(range(start, end)) if lst is None else list(lst[start:end])

    def _register(self, prefix, resident_pages, partners):
        """Record one anchor's costs; returns its index."""
        a = len(self.prefix)
        k = len(self.stages)
        if partners is None:
            lists = [None] * k
            cums = [None] * k
        else:
            lists = [None if lst is None else list(lst) for lst in partners]
            if len(lists) != k:
                raise ValueError(
                    f"anchor {a}: partner lists for {len(lists)} stages, "
                    f"the join has {k}")
            cums = []
            for j, lst in enumerate(lists):
                if lst is None:
                    cums.append(None)
                    continue
                cum = [0]
                for i in lst:
                    if not 0 <= i < len(self.stages[j]):
                        raise ValueError(
                            f"anchor {a} stage {j}: partner {i} is out "
                            f"of range")
                    cum.append(cum[-1] + self.stages[j][i])
                cums.append(cum)
        need = self.page_cost(prefix + self._extra)
        if resident_pages is not None:
            need = max(0, need - resident_pages)
        elif need > self.arena_pages:
            raise ValueError(
                f"anchor {a} needs {need} KV pages; the arena "
                f"holds {self.arena_pages}")
        carried = 0 if resident_pages is not None else prefix
        self.prefix.append(prefix)
        self._lists.append(lists)
        self._cums.append(cums)
        self._page_cost.append(need)
        self._carried.append(carried)
        self._stage.append(-1)
        self._next.append(0)
        self._true.append([False] * k)
        if self._count(a, 0) == 0:
            # nothing to evaluate: the anchor settles without a chunk
            self._first.append(0)
            self._stage[a] = _DONE
            self._settled.append(("finished" if k == 1 else "dropped", a))
            self.borrowing.skip(a)
            return a
        first = self.frame_rows[0] + self._suffix(a, 0, 0)
        if resident_pages is None and prefix + first > self.chunk_budget:
            raise ValueError(
                f"anchor {a}: prefix {prefix} tokens leaves no "
                f"room for a partner in a {self.chunk_budget}-token "
                f"chunk")
        if self.temporary_suffix_pages:
            largest = max(
                (self._suffix_page_cost(a, j, i)
                 for j in range(k) for i in range(self._count(a, j))),
                default=0,
            )
            needed = self.page_cost(prefix + self._extra) + largest
            if needed > self.arena_pages:
                raise ValueError("anchor and one suffix exceed the KV arena")
            self._page_reserve = max(self._page_reserve, largest)
        self._first.append(first)
        # the fewest tokens any fresh anchor can pack: a borrower
        # packs only the tokens past its shared prefix
        tree_shared = self.borrowing.tree_shared
        shared = tree_shared[a] if a < len(tree_shared) else 0
        self._min_fresh = min(self._min_fresh, max(0, carried - shared) + first)
        if not need:
            self._zero_cost += 1
        return a

    def _suffix(self, a, j, i):
        lst = self._lists[a][j]
        return self.stages[j][i if lst is None else lst[i]]

    def admit(self, prefix_tokens, resident_pages=None, partners=None):
        """Queue one more anchor behind the pending ones; returns its index.

        resident_pages says the anchor's prefix KV is already in the
        arena on that many pages, so it packs no prefix tokens.
        partners is the anchor's per-stage partner index lists, as in
        anchor_partners.
        """
        a = self._register(prefix_tokens, resident_pages, partners)
        self.borrowing.add()
        if self._stage[a] == -1:
            self.pending.append(a)
        return a

    def take_settled(self):
        """Events for anchors that settled without running a chunk.

        Returns ("finished", a) or ("dropped", a) pairs, as report()
        does, and clears them.
        """
        events, self._settled = self._settled, []
        return events

    def buildable_tokens(self):
        """Tokens the next chunk could pack, capped at the chunk budget.

        Counts every placed anchor's remaining stream and every pending
        anchor's first chunk. Chunk room, not pages: a page-blocked
        anchor still counts.
        """
        total = 0
        for a in self.ready:
            j, i = self._stage[a], self._next[a]
            cum = self._cum_of(a, j)
            total += (self.frame_rows[j] if i == 0 else 0) + cum[-1] - cum[i]
            if total >= self.chunk_budget:
                return self.chunk_budget
        for a in self.pending:
            total += (self._carried[a] + self.frame_rows[0]
                      + self._cum_of(a, 0)[-1])
            if total >= self.chunk_budget:
                return self.chunk_budget
        return total

    # ---- chunk building ------------------------------------------------

    def _first_cost(self, a, j, i):
        return self._suffix(a, j, i) + (self.frame_rows[j] if i == 0 else 0)

    def _suffix_page_cost(self, a, j, i):
        return partner_pages(self.page_cost, self.page_tokens, self.prefix[a],
                             self.frames[j], self._suffix(a, j, i))

    def _take(self, a, j, i, room, page_room):
        """Return the partner run that fits the token and temporary page budgets."""
        cum = self._cum_of(a, j)
        frame = self.frame_rows[j] if i == 0 else 0
        end = bisect_right(cum, cum[i] + room - frame) - 1
        pages = 0
        if self.temporary_suffix_pages:
            key = (a, j)
            if key not in self._page_cums:
                costs = [0]
                for index in range(self._count(a, j)):
                    costs.append(costs[-1] + self._suffix_page_cost(a, j, index))
                self._page_cums[key] = costs
            costs = self._page_cums[key]
            end = min(end, bisect_right(costs, costs[i] + page_room) - 1)
            pages = costs[end] - costs[i]
        return end, frame + cum[end] - cum[i], pages

    def _launch(self, a, j, end):
        """Record a launched group; True when the stream continues."""
        self._next[a] = end
        self.in_flight += 1
        return end < self._count(a, j)

    def next_chunk(self, free_pages):
        """Groups for the next chunk: [(anchor, stage, start, end, carried)].

        start and end index the anchor's own partner list at the
        stage; partner_indices() maps them to the stage's list.
        carried means the anchor's prefix tokens are packed and its KV
        written to its pages. free_pages is the arena's free list;
        fresh anchors admit against it. Returns [] when nothing is
        buildable.
        """
        self.blocked_pages = 0
        room = self.chunk_budget
        groups = []
        continued = []
        temporary_pages = 0
        # 1) placed anchors: cut streams (front) and next-stage starts
        for _ in range(len(self.ready)):
            a = self.ready.popleft()
            j, i = self._stage[a], self._next[a]
            if self._first_cost(a, j, i) > room:
                self.ready.appendleft(a)    # FIFO; chunk nearly full
                break
            end, tokens, pages = self._take(a, j, i, room, free_pages)
            if end == i:
                self.blocked_pages = self._suffix_page_cost(a, j, i) - free_pages
                self.ready.appendleft(a)
                break
            free_pages -= pages
            temporary_pages += pages
            groups.append((a, j, i, end, False))
            room -= tokens
            if self._launch(a, j, end):
                continued.append(a)
        # 2) fresh anchors: pages in queue order, chunk room may skip
        held = []
        blocked = False
        admitted = set()
        while self.pending and room >= self._min_fresh:
            a = self.pending.popleft()
            borrow = self.borrowing.decide(a, admitted)
            if borrow == Borrowing.WAIT:
                held.append(a)      # its parent is still queued
                continue
            need = (self._page_cost[a] if borrow is None
                    else self.page_cost(self.prefix[a] - borrow[1] + self._extra))
            if blocked and need:
                held.append(a)
                if not self._zero_cost:
                    break
                continue
            required = need + max(0, self._page_reserve - temporary_pages)
            if required > free_pages:
                blocked = True
                self.blocked_pages = required - free_pages
                held.append(a)
                if not self._zero_cost:
                    break
                continue
            carried = (self._carried[a] if borrow is None
                       else self.prefix[a] - borrow[1])
            if carried + self._first[a] > room:
                held.append(a)      # chunk room only; retry next chunk
                continue
            end, tokens, pages = self._take(
                a, 0, 0, room - carried, free_pages - need)
            if end == 0:
                held.append(a)
                self.blocked_pages = (
                    need + self._suffix_page_cost(a, 0, 0) - free_pages)
                break
            temporary_pages += pages
            groups.append((a, 0, 0, end, carried > 0))
            room -= carried + tokens
            free_pages -= need + pages
            if not self._page_cost[a]:
                self._zero_cost -= 1
            self._stage[a] = 0
            admitted.add(a)
            self.borrowing.record(a, borrow)
            if self._launch(a, 0, end):
                continued.append(a)
        self.pending.extendleft(reversed(held))
        self.ready.extendleft(reversed(continued))
        return groups

    # ---- gating --------------------------------------------------------

    def report(self, a, j, start, end, bits):
        """Record one group's answers.

        Returns events: ("dropped", a) when every partner at a stage
        before the last answered FALSE, so the anchor's pages can go;
        ("finished", a) when its last-stage row is complete.
        """
        if self.answer_dtype is None:
            row = self.answers[j].setdefault(a, [])
            received = len(row)
        else:
            if a not in self.answers[j]:
                self.answers[j][a] = np.empty(
                    self._count(a, j), dtype=self.answer_dtype)
            row = self.answers[j][a]
            received = self._answer_counts[j].get(a, 0)
        if received != start or len(bits) != end - start:
            raise AssertionError(
                f"anchor {a} stage {j}: answers for partners "
                f"{start}:{end} arrived with {received} recorded")
        if self.answer_dtype is None:
            row.extend(bits)
        else:
            row[start:end] = bits
            self._answer_counts[j][a] = end
        self.in_flight -= 1
        if (any(bits) if self.answer_dtype is None else np.any(bits)):
            self._true[a][j] = True
        k = len(self.stages)
        n_j = self._count(a, j)
        complete = end == n_j
        events = []
        if j == self._stage[a] and j + 1 < k:
            if self._true[a][j] and self._next[a] == n_j:
                # the whole stream is launched, so every later chunk
                # is behind it on the stream and the next stage's
                # frame write cannot race a read of this stage's
                self._stage[a] = j + 1
                self._next[a] = 0
                if self._count(a, j + 1):
                    self.ready.append(a)
                else:
                    # an empty row at the last stage, dropped otherwise
                    self._stage[a] = _DONE
                    events.append(("finished" if j + 2 == k else "dropped", a))
            elif complete and not self._true[a][j]:
                self._stage[a] = _DONE
                events.append(("dropped", a))
        if j == k - 1 and complete:
            self._stage[a] = _DONE
            events.append(("finished", a))
        return events

    # ---- progress ------------------------------------------------------

    def done(self):
        return not self.pending and not self.ready \
            and not self.in_flight and not self._settled


class FilterAdmission:
    """Filter chain scheduler: builds chunk groups in admission order.

    Page accounting belongs to the arena: next_chunk takes its free
    pages and admits documents whose own pages fit them.

    Args:
        doc_tokens: Per-document token counts.
        stage_tokens: Per-stage question suffix token counts.
        chunk_budget: Tokens per forward pass.
        arena_pages: Pages in the arena, or None when nothing is written
            to it (single-stage).
        page_tokens: Tokens per arena page.
        kept_extra_tokens: Extra tokens per document that must fit in
            pages (shared preamble plus tail room).
        limit: Stop after this many survivors.
        page_cost: Callable(tokens, base_tokens) giving the pages a
            document of that many rows takes, in the arena's every-token
            pages; None prices one pool of page_tokens pages.
        tree: A PrefixTree over the documents, or None. Documents are
            admitted in tree order, and borrow as Borrowing decides.
        can_borrow: As Borrowing takes it.

    Survivor suffixes pack before fresh admissions. Pages are granted
    in queue order; chunk room may be skipped. With a tree the
    admission order is token-sorted, not table order, so a limit
    takes the first survivors in that order.
    """

    def __init__(self, doc_tokens, stage_tokens, chunk_budget,
                 arena_pages, page_tokens, kept_extra_tokens=0,
                 limit=None, page_cost=None, tree=None, can_borrow=None):
        self.page_cost = page_cost or (
            lambda tokens, base_tokens=None: pages_for(tokens, page_tokens))
        self.doc_tokens = doc_tokens
        n = len(doc_tokens)
        self.borrowing = Borrowing(n, tree, can_borrow)
        self.stage_tokens = list(stage_tokens)
        self.chunk_budget = chunk_budget
        self.page_tokens = page_tokens
        self.pages = arena_pages is not None
        self.blocked_pages = 0
        self.limit = limit
        self._survivor_count = 0
        # kept_extra_tokens: the shared question preamble that joins
        # the document's kept KV after stage 1, so pages must cover it
        self.kept_extra = kept_extra_tokens
        if tree is not None and arena_pages is None:
            raise ValueError("a prefix tree needs page accounting")
        # a document packs whole when its parent has left, so the
        # guard is on its full length
        for d, t in enumerate(self.doc_tokens):
            need = t + max(stage_tokens)
            if need > chunk_budget:
                raise ValueError(f"document {d} + question needs {need} "
                                 f"tokens > chunk budget {chunk_budget}")
            if arena_pages is not None and self.page_cost(
                    t + kept_extra_tokens) > arena_pages:
                raise ValueError(f"document {d} needs more pages than "
                                 f"the arena holds")
        if tree is None or list(tree.order) == list(range(n)):
            self.pending = _CompactQueue(n)
        else:
            self.pending = _CompactQueue(0)
            self.pending.prepend(array("I", tree.order))
        self.ready = deque()       # (doc, stage) gated TRUE, next suffix
        self.in_flight = set()     # docs inside a launched chunk
        self.resident = set()      # admitted docs not yet released
        self.answers = {}          # doc -> [0/1 per answered stage]

    def own_tokens(self, doc) -> int:
        """The document's tokens past its borrowed shared prefix."""
        return self.doc_tokens[doc] - self.borrowing.shared(doc)

    # ---- chunk building ------------------------------------------------

    def next_chunk(self, free_pages=None):
        """Groups for the next chunk: [(doc, stage, fresh)].

        free_pages is the arena's free pages, in every-token pages;
        None when nothing is written to the arena. fresh means the
        document's tokens are packed and its KV is written to its
        pages. Returns [] when nothing is buildable.
        """
        if self._limit_reached():
            return []
        self.blocked_pages = 0
        room = self.chunk_budget
        groups = []
        admitted = set()
        # 1) survivor suffixes, oldest first; one live stage per doc
        n_ready = len(self.ready)
        for _ in range(n_ready):
            doc, stage = self.ready[0]
            cost = self.stage_tokens[stage]
            if cost > room:
                break    # ready is FIFO; the chunk is nearly full
            self.ready.popleft()
            groups.append((doc, stage, False))
            self.in_flight.add(doc)
            room -= cost
        # 2) fresh admissions: pages in queue order, chunk room may skip
        skipped = array("I")
        while self.pending:
            doc = self.pending.popleft()
            choice = self.borrowing.decide(doc, admitted)
            if choice == Borrowing.WAIT:
                # the parent is queued behind a full chunk
                skipped.append(doc)
                continue
            own = self.doc_tokens[doc] - (choice[1] if choice else 0)
            need_pages = 0
            if free_pages is not None:
                need_pages = self.page_cost(own + self.kept_extra)
                if need_pages > free_pages:
                    # pages are granted in order: put it back and stop
                    # claiming pages behind it
                    self.pending.appendleft(doc)
                    self.blocked_pages = need_pages - free_pages
                    break
            cost = self.stage_tokens[0] + own
            if cost > room:
                skipped.append(doc)   # chunk room only; retry next chunk
                continue
            if free_pages is not None:
                free_pages -= need_pages
            self.borrowing.record(doc, choice)
            self.resident.add(doc)
            groups.append((doc, 0, True))
            self.in_flight.add(doc)
            admitted.add(doc)
            room -= cost
        self.pending.prepend(skipped)
        return groups

    # ---- gating --------------------------------------------------------

    def report(self, doc, stage, passed, release=True):
        """Record one answer.

        Queues the next-stage suffix after TRUE at a stage before the
        last. Otherwise the document is done: with release it returns
        (doc,) for the caller to free, and without it the caller keeps
        the document's KV.

        Returns docs to free.
        """
        self.in_flight.discard(doc)
        self.answers.setdefault(doc, []).append(1 if passed else 0)
        last = stage == len(self.stage_tokens) - 1
        if passed and last:
            self._survivor_count += 1
        if passed and not last:
            self.ready.append((doc, stage + 1))
            return ()
        self.resident.discard(doc)
        if not self.pages or not release:
            return ()
        return (doc,)

    # ---- progress ------------------------------------------------------

    def _limit_reached(self):
        return (self.limit is not None
                and self._survivor_count >= self.limit)

    def done(self):
        if self._limit_reached():
            return not self.in_flight
        return (not self.pending and not self.ready
                and not self.in_flight)

    def drain_ready(self):
        """Docs still queued for a stage when the limit ends the run early.

        Returns them for the caller to free.
        """
        out = []
        while self.ready:
            doc, _ = self.ready.popleft()
            self.resident.discard(doc)
            if self.pages:
                out.append(doc)
        return out

    def survivors(self):
        """Documents that answered TRUE at every stage."""
        n = len(self.stage_tokens)
        return sorted(d for d, row in self.answers.items()
                      if len(row) == n and all(row))
