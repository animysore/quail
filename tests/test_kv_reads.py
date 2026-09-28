"""Filters and joins over random corpora read the right KV.

The corpora repeat one another's starts, as agent snapshots do. Small
corpora run on arenas a few documents fill; large ones have documents
of thousands of tokens and a 1,024-token window, the sizes QUAIL-B
runs at. See kv_checker for the check.
"""

import random

import pytest
from fakes import cpu_staging
from kv_checker import (
    Setup,
    check_feed,
    check_filter,
    check_join,
    check_join_after_join,
)

pytest.importorskip("torch")


@pytest.fixture(params=[("unified", False, False), ("tree", False, False),
                        ("unified", True, False), ("unified", True, True)],
                ids=["unified", "tree", "sliding", "sliding-canvas"])
def arena_kind(request, monkeypatch):
    """(attention path, sliding layers, canvas row), with CPU staging."""
    cpu_staging(monkeypatch)
    return request.param


def corpus(rng, n, header_tokens, piece_tokens):
    """Documents that repeat one another's starts, as agent snapshots do.

    Each trace starts with a shared header and grows by pieces; every
    prefix of it is a document. Some documents are exact duplicates,
    and some branch off a trace part way through.
    """
    header = [rng.randrange(50) for _ in range(rng.choice(header_tokens))]
    docs = []
    while len(docs) < n:
        trace = list(header)
        for _ in range(rng.randrange(1, 5)):
            trace = trace + [rng.randrange(50)
                             for _ in range(rng.choice(piece_tokens))]
            docs.append(trace)
            if rng.random() < 0.2:
                docs.append(list(trace))
            if rng.random() < 0.3 and len(trace) > 20:
                cut = rng.randrange(1, len(trace))
                docs.append(trace[:cut] + [50 + rng.randrange(20)])
    docs = docs[:n]
    rng.shuffle(docs)
    return docs


def small(rng, kind, n, pages, sliding_pages):
    """A corpus of short documents and an arena a few of them fill."""
    path, sliding, canvas = kind
    docs = corpus(rng, n, (0, 5, 16, 20), (1, 7, 16, 30, 45))
    return docs, Setup(
        path=path, window=32 if sliding else None,
        canvas=(99,) if canvas else (), pages=rng.choice(pages),
        sliding_pages=rng.choice(sliding_pages),
        budget=max(map(len, docs)) + rng.choice((8, 60, 600)))


def large(rng, kind):
    """Documents of thousands of tokens, a 1,024-token window."""
    path, sliding, canvas = kind
    docs = corpus(rng, 120, (0, 48, 300), (40, 200, 700, 1500))
    longest = max(map(len, docs))
    # the longest document and its extra rows fit either pool untrimmed
    pages = -(-(longest + 16) // 16) + 1
    return docs, Setup(
        path=path, window=1024 if sliding else None,
        canvas=(99,) if canvas else (),
        pages=pages * rng.choice((1, 3, 10)),
        sliding_pages=pages * rng.choice((1, 2)),
        budget=longest + rng.choice((16, 4096, 16384)))


def questions(rng):
    return [[90, 91, 92, 93], [90, 91, 94]][:rng.choice((1, 2))]


def join_parts():
    """A join's frame and three partner suffixes."""
    return [95, 96], [[97, 60 + p, 98] for p in range(3)]


def named(case, check, *args, **options):
    """Run a check, naming the case in any failure."""
    try:
        check(*args, **options)
    except AssertionError as error:
        raise AssertionError(f"{case}: {error}") from error


def test_small_filters_joins_and_filters_feeding_joins(arena_kind):
    """Short documents on arenas a few of them fill.

    A join whose child shares its first page with a retained anchor of
    an earlier join runs last: the anchor trimmed its window, so on a
    sliding arena the child cannot borrow the rows the anchor dropped
    and packs whole; elsewhere it borrows the page.
    """
    for seed in range(6):
        rng = random.Random(seed)
        docs, setup = small(rng, arena_kind, 40, (40, 80, 400), (24, 60))
        named(f"filter seed {seed}", check_filter, docs, questions(rng),
              setup)
        for retain in (False, True):
            rng = random.Random(100 + seed)
            docs, setup = small(rng, arena_kind, 30, (60, 400), (40, 80))
            named(f"join seed {seed} retain {retain}", check_join, docs,
                  *join_parts(), setup, retain=retain)
        rng = random.Random(200 + seed)
        docs, setup = small(rng, arena_kind, 40, (80, 400), (60, 120))
        named(f"feed seed {seed}", check_feed, docs, [90, 91, 92],
              *join_parts(), setup)

    path, sliding, canvas = arena_kind
    parent = [1] * 64                  # four pages, two of them the window
    child = [1] * 16 + [3] * 5
    setup = Setup(path=path, window=32 if sliding else None,
                  canvas=(99,) if canvas else (), pages=12,
                  sliding_pages=12, budget=200, page_tokens=16)
    check_join_after_join([parent, child], *join_parts(), setup)


def test_large_filters_joins_and_filters_feeding_joins(arena_kind):
    """Documents of thousands of tokens on a 1,024-token window."""
    for seed in range(2):
        rng = random.Random(300 + seed)
        docs, setup = large(rng, arena_kind)
        named(f"filter seed {seed}", check_filter, docs, questions(rng),
              setup)
        for retain in (False, True):
            rng = random.Random(400 + seed)
            docs, setup = large(rng, arena_kind)
            named(f"join seed {seed} retain {retain}", check_join, docs,
                  *join_parts(), setup, retain=retain)
        rng = random.Random(500 + seed)
        docs, setup = large(rng, arena_kind)
        named(f"feed seed {seed}", check_feed, docs, [90, 91, 92],
              *join_parts(), setup)
