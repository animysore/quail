"""Tests for reading Boolean answers from public request outputs."""

from types import SimpleNamespace

import pytest

from quail.backends.request_scheduling import canvas_answer, text_answer, true_bit


def _output(entries=None, *, tokens=(7,), text="TRUE"):
    return SimpleNamespace(outputs=[SimpleNamespace(
        token_ids=list(tokens), text=text,
        logprobs=None if entries is None else [entries])])


def _entry(score, rank=None, decoded=None):
    return SimpleNamespace(logprob=score, rank=rank, decoded_token=decoded)


def _canvas(entries):
    return canvas_answer(_output(entries), true_ids={7, 9}, false_ids={8, 10})


SCORED = [
    ({7: _entry(-4, decoded="FALSE"), 8: _entry(-2, decoded="TRUE"),
      9: _entry(-5), 10: _entry(-5)}, 0),
    ({7: _entry(-4), 9: _entry(-1), 8: _entry(-2), 10: _entry(-5)}, 1),
    ({t: _entry(-2) for t in (7, 8, 9, 10)}, 0),
    ({t: _entry(-2) for t in (10, 9, 8, 7)}, 0),
    ({1: _entry(-1, 1), 7: _entry(-20, 600), 9: _entry(-21, 700),
      8: _entry(-22, 800), 10: _entry(-23, 900)}, 1),
]

UNSCORED = [
    None,
    {},
    {1: _entry(-1, 1), 2: _entry(-2, 2), 3: _entry(-3, 3)},
    {1: _entry(-1, 1), 2: _entry(-2, 2), 7: _entry(-3, 3)},
    {1: _entry(-1, 1), 2: _entry(-2, 2), 7: _entry(-4, 100)},
    {7: _entry(-1, 1)},
    {7: _entry(-1), 8: _entry(-2), 9: _entry(-3)},
    {11: _entry(-1, 1, "TRUE")},
]


def test_token_text_and_canvas_answers_read_scores_not_sampled_text():
    assert true_bit(_output(tokens=(7,)), {7}) == 1
    assert true_bit(_output(tokens=(8,)), {7}) == 0
    with pytest.raises(ValueError, match="no answer token"):
        true_bit(_output(tokens=()), {7})
    assert text_answer(_output(text="The answer is TRUE.")) == 1
    assert text_answer(_output(text="TRUEISH **FALSE**")) == 0
    with pytest.raises(ValueError, match="no TRUE/FALSE"):
        text_answer(_output(text=""))

    for entries, expected in SCORED:
        assert _canvas(entries) == expected, entries
    for entries in UNSCORED:
        with pytest.raises(ValueError, match="omitted requested TRUE/FALSE scores"):
            _canvas(entries)
