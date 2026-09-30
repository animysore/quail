"""Checks for the parallel QUAIL-B run layout."""

import pytest

from quail.bench.quailb_parallel import _query_groups, _selected_queries
from quail_b.queries import queries


def test_bio_queries_use_separate_groups():
    query_ids = ("IMDB-1", "IMDB-2", "BIO-1", "BIO-2", "FEV-1", "AGENT-1")

    assert _query_groups(query_ids) == [
        ("imdb", ("IMDB-1", "IMDB-2"), ""),
        ("biodex-bio-1", ("BIO-1",), "-bio-1"),
        ("biodex-bio-2", ("BIO-2",), "-bio-2"),
        ("fever", ("FEV-1",), ""),
        ("agent", ("AGENT-1",), ""),
    ]


def test_modal_preflight_skips_classification_before_scheduling():
    requested, selected, skipped = _selected_queries("", 0.1)
    assert set(requested) == queries().keys()
    assert len(selected) == 31 and len(skipped) == 10
    assert set(selected) | skipped.keys() == set(requested)
    assert all("AI.CLASSIFY" in reason for reason in skipped.values())
    with pytest.raises(ValueError, match="no supported"):
        _selected_queries(next(iter(skipped)), 0.1)


def test_modal_samples_use_the_supported_direct_runner():
    with pytest.raises(ValueError, match="python -m quail.bench.quailb"):
        _selected_queries("EXTRACT-1", 1.0)
