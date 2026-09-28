"""Tests for PageArena page allocation, freeing, and row indexing."""

import random

import pytest

from quail.backends.quail.executor.arena import PageArena


def test_allocation_retention_rewind_and_pinning():
    a = PageArena(n_pages=10, page_tokens=16)
    pages = a.alloc("d0", 40)      # 3 pages
    assert len(pages) == 3
    assert a.free_pages == 7
    assert a.resident_tokens == 40
    assert a.free_key("d0") == 3
    assert a.free_pages == 10
    assert a.resident_tokens == 0

    assert a.alloc("big", 200) is None
    assert a.alloc("d", 5, capacity_tokens=10) is not None
    assert len(a.row_indices("d")) == 5
    assert len(a.row_indices("d", 10)) == 10
    with pytest.raises(KeyError):
        a.alloc("d", 5)

    rng = random.Random(5)
    a = PageArena(n_pages=64, page_tokens=16)
    live = {}
    for step in range(500):
        if live and rng.random() < 0.45:
            key = rng.choice(sorted(live))
            a.free_key(key)
            del live[key]
        else:
            key = f"d{step}"
            got = a.alloc(key, rng.randrange(1, 200))
            if got is not None:
                live[key] = got
        held = [p for pages in live.values() for p in pages]
        assert len(held) == len(set(held)), "page double-owned"
        assert len(held) + a.free_pages == 64, "pages leaked"

    a = PageArena(n_pages=3, page_tokens=4)
    a.alloc("doc", 4)
    assert a.grow("doc", 12) == 2
    assert a.grow("doc", 16) is None

    a = PageArena(n_pages=10, page_tokens=4)
    a.alloc("doc", 7, capacity_tokens=11)
    assert len(a.owned["doc"]) == 3

    a.pin("doc")
    assert "doc" in a.pinned
    a.retain("doc")
    assert "doc" not in a.pinned
    assert a.retained["doc"] == 7
    assert a.retained_prefix_tokens == 7

    assert a.rewind("doc", 4) == 2
    assert len(a.owned["doc"]) == 1
    assert a.tokens["doc"] == 4
    a.free_key("doc")
    assert not a.pinned
    assert not a.retained

    a = PageArena(n_pages=4, page_tokens=16)
    a.alloc("one-page", 16)
    a.alloc("two-pages", 17)
    a.retain("one-page")
    a.retain("two-pages")

    assert a.pop_retained_victim() == ("two-pages", 2, 17)

    a = PageArena(n_pages=2, page_tokens=16)
    a.alloc("doc", 16)
    a.retain("doc")
    a.pin("doc")

    assert a.pop_retained_victim() is None


def test_borrowed_pages_are_shared_until_the_last_holder_frees_them():
    a = PageArena(n_pages=10, page_tokens=16)
    parent = a.alloc("p", 40)                 # pages for 40 tokens: 3
    child = a.alloc("c", 50, borrowed=a.shared_pages("p", 32))  # borrows 2, owns 2
    assert len(child) == 2 and a.free_pages == 5
    assert a.table_pages("c") == parent[:2] + child
    assert a.borrowed_tokens("c") == 32
    assert a.row_indices("c")[:32] == a.row_indices("p")[:32]
    # the parent's shared pages outlive the parent
    assert a.free_key("p") == 1
    assert a.free_pages == 6
    assert a.table_pages("c") == parent[:2] + child
    assert a.free_key("c") == 4
    assert a.free_pages == 10

    # a grandchild borrows through its parent's borrowed pages
    a.alloc("p", 40)
    a.alloc("c", 50, borrowed=a.shared_pages("p", 32))
    a.alloc("g", 60, borrowed=a.shared_pages("c", 48))
    assert a.table_pages("g")[:2] == a.table_pages("p")[:2]
    assert a.table_pages("g")[2] == a.owned["c"][0]
    for key in ("p", "c", "g"):
        a.free_key(key)
    assert a.free_pages == 10

    # rewinding never drops borrowed pages; growing adds own pages
    a.alloc("p", 40)
    a.alloc("c", 50, borrowed=a.shared_pages("p", 32))
    assert a.rewind("c", 33) == 1
    assert a.tokens["c"] == 33 and len(a.owned["c"]) == 1
    with pytest.raises(ValueError):
        a.rewind("c", 16)
    assert a.grow("c", 64) == 1
    assert len(a.table_pages("c")) == 4
    assert a.retained_pages == 0
    a.retain("c")
    assert a.retained_pages == 2       # the pages it alone holds
    a.free_key("p")
    a.retain("c")
    assert a.retained_pages == 4       # now the borrowed pages too
    a.alloc("p", 40)

    with pytest.raises(ValueError):
        a.shared_pages("p", 20)                 # not a whole page
    with pytest.raises(ValueError):
        a.alloc("x", 16, borrowed=a.shared_pages("p", 32))  # past its tokens
    with pytest.raises(KeyError):
        a.shared_pages("nobody", 16)
    with pytest.raises(ValueError):
        a.shared_pages("p", 48)                 # parent holds 40
    # a slice of the parent, as the sliding pool borrows, and dropping
    # leading pages takes borrowed pages before own ones
    a.alloc("s", 48, borrowed=a.shared_pages("p", 32, 16))
    assert a.table_pages("s")[:1] == a.table_pages("p")[1:2]
    # the parent still holds the borrowed page; one own page goes free
    assert a.drop_leading("s", 2) == 1
    assert a.borrowed["s"] == [] and len(a.owned["s"]) == 1
