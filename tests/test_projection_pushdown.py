"""Projection pushdown rule tests."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_session import _run, fake_tok, make_executor

import quail
from quail.logical import ColumnRef, Scan, bind_join_prompt, bind_prompt
from quail.physical import Recombine
from quail.planner.logical_rules import push_down_projection
from quail.planner.plan import EngineConfig


def test_prompts_bound_without_a_tokenizer_have_no_token_counts():
    review = ColumnRef("r", "reviews", "review")
    description = ColumnRef("p", "products", "description")
    assert bind_prompt("q {0}", (review,)).tail_tokens is None
    join = bind_join_prompt("same {0} {1}", (review, description))
    assert join.tail_tokens is None


def _scans(plan):
    return {node.alias: node for node in plan.walk()
            if isinstance(node, Scan)}


def _session(tmp_path, tokenizer=fake_tok):
    session = quail.Session(
        EngineConfig(gpus=1, model="qwen3-4b-fp8", backend="quail",
                     device="h100-sxm"),
        tokenizer=tokenizer,
    )
    path = tmp_path / "reviews.parquet"
    pq.write_table(pa.table({
        "id": ["r0", "r1", "r2"],
        "review": ["good one", "bad one", "fine one"],
        "stars": [5, 1, 3],
        "wide": ["x" * 50, "y" * 50, "z" * 50],
    }), str(path))
    session.register("reviews", quail.DocumentProvider.from_parquet(
        str(path), id_col="id"))
    return session


def test_projected_results_token_reuse_and_join_projection(tmp_path):
    reviews = ["good one", "bad one", "fine one"]
    calls = []

    def counting_tok(text):
        # Prompts are tokenized on every query; count documents only.
        if text in reviews:
            calls.append(text)
        return fake_tok(text)

    session = _session(tmp_path, tokenizer=counting_tok)
    truth = {"r": {"q:": [1, 0, 1]}}
    query = session.sql(
        "SELECT r.stars, r.id FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))")

    result = _run(query, make_executor(truth))

    assert sorted(result.to_rows()) == [(3, "r2"), (5, "r0")]
    scan = _scans(query.logical)["r"]
    assert scan.columns == ("stars", "id")
    assert push_down_projection(query.logical.root) is query.logical.root
    store = query._token_inputs["r"]
    assert store.projected_columns == ("stars", "id")
    tokenized = len(calls)
    assert tokenized > 0

    second = _run(session.sql(
        "SELECT r.stars, r.review FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))"), make_executor(truth))
    assert sorted(second.to_rows()) == [(3, "fine one"), (5, "good one")]
    assert len(calls) == tokenized
    assert len(session._token_stores) == 1
    assert sorted(name for _, name in session._column_stores) == [
        "id", "review", "stars"]

    truth = {"r": {"q:": [0, 1, 0]}}

    text = _run(session.sql(
        "SELECT r.review FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))"), make_executor(truth))
    assert text.to_rows() == [("bad one",)]

    star = _run(session.sql(
        "SELECT * FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review))"), make_executor(truth))
    assert star.columns == ["r.id", "r.review", "r.stars", "r.wide"]
    assert star.to_rows() == [("r1", "bad one", 1, "y" * 50)]
    assert len(calls) == tokenized
    session.close()

    session = _session(tmp_path)
    session.register("products", quail.DocumentProvider.from_table(pa.table({
        "asin": ["p0", "p1"], "description": ["product 0", "product 1"],
    }), id_col="asin"))
    truth = {"r": {"q:": [1, 0, 1]}}
    # the planner picks the anchor; the rule is symmetric so either works
    even_sum = lambda a, b: (a + b) % 2 == 0  # noqa: E731
    join_truth = {("r", "p"): even_sum, ("p", "r"): even_sum}
    query = session.sql(
        "SELECT r.id, p.asin FROM reviews r JOIN products p ON "
        "AI_FILTER(PROMPT('m {0} {1}', r.review, p.description)) "
        "WHERE AI_FILTER(PROMPT('q: {0}', r.review))")
    result = _run(query, make_executor(truth, join_truth))
    assert not [n for n in result.plan.nodes if isinstance(n, Recombine)]
    assert sorted(result.to_rows()) == [("r0", "p0"), ("r2", "p0")]
    assert result.count() == 2
    session.close()


class _UnstableProvider:
    id_col = "id"
    columns = ("id", "body", "stars")

    def __init__(self):
        self.scans = 0

    def schema(self):
        return pa.schema({"id": pa.string(), "body": pa.string(),
                          "stars": pa.int64()})

    def content_identity(self):
        return "unstable"

    def statistics(self):
        from quail.catalog import TableStatistics
        return TableStatistics(row_count=3)

    def scan(self, request):
        self.scans += 1
        rows = 3 if self.scans == 1 else 2
        table = pa.table({
            "id": [f"r{i}" for i in range(rows)],
            "body": ["one two"] * rows,
            "stars": list(range(rows)),
        }).select(list(request.columns))
        return table.to_reader()


def test_failed_scans_and_tokenization_leave_no_partial_cache(tmp_path):
    session = _session(tmp_path)
    session.register("docs", _UnstableProvider())
    session.tokenize("docs", "body", ("id",))

    with pytest.raises(RuntimeError, match="stable order"):
        session.tokenize("docs", "body", ("stars",))
    with pytest.raises(RuntimeError, match="stable order"):
        session.tokenize("docs", "body", ("stars",))

    assert ("unstable", "stars") not in session._column_stores
    assert len(list(Path(session._token_directory.name).iterdir())) == 2
    session.close()

    def broken(text):
        raise ValueError("tokenizer failed")

    session = _session(tmp_path)
    session._tok = broken

    with pytest.raises(ValueError, match="tokenizer failed"):
        session.tokenize("reviews", "review", ("id",))

    assert session._token_stores == {}
    assert session._column_stores == {}
    assert session._token_directory is None or not list(
        Path(session._token_directory.name).iterdir())
    session.close()
