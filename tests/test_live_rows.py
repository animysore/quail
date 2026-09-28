from quail.planner.live_rows import PairRelation, exact_live_rows

PAIR_SETS = [
    ({("a", "b"): {(0, 0), (1, 1)}, ("b", "c"): {(0, 0)}}, (0,)),
    ({("a", "b"): {(0, 0), (1, 1)}, ("b", "c"): {(0, 0), (1, 1)},
      ("a", "c"): {(0, 1), (1, 0)}}, ()),
]


def test_live_rows_for_trees_and_cycles():
    for pairs, expected in PAIR_SETS:
        live = exact_live_rows(
            {"a": (0, 1), "b": (0, 1), "c": (0, 1)},
            tuple(PairRelation(left, right, frozenset(rows))
                  for (left, right), rows in pairs.items()),
        )
        assert live == {"a": expected, "b": expected, "c": expected}, pairs
