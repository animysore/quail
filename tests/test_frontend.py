"""The SQL front end and the builder: same LogicalPlan, named errors."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fakes import keep_even, same_key

from quail.catalog import Catalog, DocumentProvider
from quail.frontend.builder import col, docs, prompt
from quail.frontend.sql import SQLDialect, compile_sql
from quail.logical import (
    ANSWER_CUE,
    SHARED_PRE,
    Apply,
    ColumnRef,
    CompileError,
    Join,
    Project,
    Scan,
    SemanticFilter,
    SemanticJoin,
    bind_join_prompt,
    canonicalize_template,
    join_applies,
    join_conditions,
    join_label,
    join_outer_input,
    render_join_frame,
    render_join_question,
    split_frame,
)
from quail.planner.logical_optimizer import LogicalPlanningContext, apply_logical_rules
from quail.planner.logical_rules import built_in_logical_rules


@pytest.fixture()
def catalog(tmp_path):
    cat = Catalog()
    for name, columns in (("reviews", ["id", "review"]),
                          ("products", ["asin", "description"]),
                          ("threads", ["id", "thread"])):
        path = str(tmp_path / f"{name}.parquet")
        pq.write_table(pa.table({c: ["x", "y"] for c in columns}), path)
        cat.register(name, DocumentProvider.from_parquet(path, id_col=columns[0]))
    return cat


def tok(text):
    return text.split()


FILTER_JOIN_SQL = """
    SELECT r.id, p.asin
    FROM reviews r
    JOIN products p
      ON AI_FILTER(PROMPT('Review {0} discusses product {1}',
                          r.review, p.description),
                   {'selectivity': 0.05})
    WHERE AI_FILTER(PROMPT('This review is negative: {0}', r.review),
                    {'selectivity': 0.3})
"""

THREE_WAY_TEMPLATE = ("Review {0} praises the thread in {1} and the "
                      "thread recommends the product in {2}.")

THREE_WAY_ON = """
    SELECT a.id, b.id, p.asin
    FROM reviews a
    JOIN threads b
    JOIN products p
      ON AI_FILTER(PROMPT('%s', a.review, b.thread, p.description),
                   {'selectivity': 0.02, 'anchor': 'b'})
""" % THREE_WAY_TEMPLATE

THREE_WAY_WHERE = """
    SELECT a.id, b.id, p.asin
    FROM reviews a, threads b, products p
    WHERE AI_FILTER(PROMPT('%s', a.review, b.thread, p.description),
                    {'selectivity': 0.02, 'anchor': 'b'})
""" % THREE_WAY_TEMPLATE


def test_sql_dialects_and_builder_produce_the_same_plan(catalog):
    plan = compile_sql(FILTER_JOIN_SQL, catalog, tok)
    root = plan.root
    assert isinstance(root, Project)
    assert [(c.alias, c.column) for c in root.columns] == \
        [("r", "id"), ("p", "asin")]
    join = root.input
    assert isinstance(join, SemanticJoin)
    assert (join.semantics, join.selectivity, join.anchor) == ("full", 0.05, None)
    # no ON equality means every pair is a candidate
    pairs = join.input
    assert isinstance(pairs, Join) and pairs.on == ()
    filt = pairs.left
    assert isinstance(filt, SemanticFilter)
    assert filt.predicates[0].selectivity == 0.3
    assert (filt.input.provider, filt.input.column) == ("reviews", "review")
    assert (pairs.right.provider, pairs.right.column) == \
        ("products", "description")
    # the join template is the per-tuple question and is never canonicalized
    assert join.prompt.template == "Review {0} discusses product {1}"

    star = compile_sql(
        "SELECT * FROM reviews r WHERE "
        "AI_FILTER(PROMPT('neg: {0}', r.review))", catalog, tok)
    assert [(c.alias, c.column) for c in star.root.columns] == \
        [("r", "id"), ("r", "review")]

    snowflake = compile_sql(
        "SELECT r.id FROM reviews r WHERE "
        "AI_FILTER(PROMPT('negative {0}', r.review))",
        catalog, tok, dialect=SQLDialect.SNOWFLAKE)
    bq = compile_sql(
        "SELECT r.id FROM reviews r WHERE "
        "AI.IF(PROMPT('negative {0}', r.review))",
        catalog, tok, dialect="bq")
    assert SQLDialect.BQ.value == "bq"
    assert bq == snowflake

    built = (docs(catalog, "reviews", tok).alias("r")
             .ai_filter(prompt("This review is negative: {0}",
                               col("r.review")), selectivity=0.3)
             .ai_join(docs(catalog, "products", tok).alias("p"),
                      prompt("Review {0} discusses product {1}",
                             col("r.review"), col("p.description")),
                      selectivity=0.05)
             .select("r.id", "p.asin"))
    assert built == plan

    # one prompt, three placeholders, all three documents in the same
    # model call - never a chain of pairwise stages
    on_plan = compile_sql(THREE_WAY_ON, catalog, tok)
    join = on_plan.root.input
    assert isinstance(join, SemanticJoin)
    assert (join.anchor, join.selectivity) == ("b", 0.02)
    outer_pairs = join.input
    assert isinstance(outer_pairs, Join) and outer_pairs.on == ()
    assert isinstance(outer_pairs.left, Join)
    assert isinstance(outer_pairs.left.left, Scan)
    assert isinstance(join_outer_input(join), Scan)
    assert [r.alias for r in join.prompt.args] == ["a", "b", "p"]
    built = (docs(catalog, "reviews", tok).alias("a")
             .ai_join([docs(catalog, "threads", tok).alias("b"),
                       docs(catalog, "products", tok).alias("p")],
                      prompt(THREE_WAY_TEMPLATE, col("a.review"),
                             col("b.thread"), col("p.description")),
                      selectivity=0.02, anchor="b")
             .select("a.id", "b.id", "p.asin"))
    assert on_plan == compile_sql(THREE_WAY_WHERE, catalog, tok) == built


def test_prompt_layout_and_predicate_order(catalog):
    args = (ColumnRef("a", "reviews", "review"),
            ColumnRef("b", "threads", "thread"))
    p = bind_join_prompt("Does {0} praise {1}?", args, tok)
    assert p.template == "Does {0} praise {1}?"
    assert p.preamble == SHARED_PRE
    # the static question is in the anchor frame; only the answer cue
    # is paid per tuple
    assert p.frame == ("\n\n"
                       "Evaluate TRUE or FALSE for the following "
                       "question: Does {0} praise {1}?")
    assert p.tail == ANSWER_CUE
    assert p.tail_tokens == len(tok(p.tail))
    assert join_label(1) == "\n\nDOCUMENT {1}:\n"
    assert [a for a, _, _ in p.labels] == ["a", "b"]
    assert p.labels[1][1] == len(tok(join_label(1)))
    assert p.labels[0][2] == len(tok(render_join_frame(p.template, 0)))
    assert render_join_question(p.template) == p.frame

    plan = compile_sql("""
        SELECT r.id FROM reviews r
        WHERE AI_FILTER(PROMPT('first: {0}', r.review))
          AND AI_FILTER(PROMPT('second: {0}', r.review))
          AND AI_FILTER(PROMPT('third: {0}', r.review))
    """, catalog, tok)
    assert [p.prompt.template for p in plan.root.input.predicates] == [
        SHARED_PRE + "{0}\n\nfirst:",
        SHARED_PRE + "{0}\n\nsecond:",
        SHARED_PRE + "{0}\n\nthird:",
    ]

    # user text before the document relocates after it, so the fixed
    # preamble is the only thing ahead of the document's KV; with two
    # placeholders only the first (the KV-owning document) moves
    assert canonicalize_template("{0}\nQ: is it good?") == \
        SHARED_PRE + "{0}\nQ: is it good?"
    assert canonicalize_template("negative review: {0}") == \
        SHARED_PRE + "{0}\n\nnegative review:"
    assert canonicalize_template("Does {0} match {1}?") == \
        SHARED_PRE + "{0}\n\nDoes match {1}?"
    assert canonicalize_template("no placeholders") == "no placeholders"
    assert split_frame("Does {0} match {1}?")[0] == "Does"
    assert split_frame("{0} then {1}")[0] == ""

    plan = compile_sql(FILTER_JOIN_SQL, catalog, tok)
    pred = join_outer_input(plan.root.input).predicates[0]
    question = ("Evaluate TRUE or FALSE for the following question: "
                "This review is negative:" + ANSWER_CUE)
    assert pred.prompt.preamble == SHARED_PRE
    assert pred.prompt.tail == "{0}\n\n" + question
    assert pred.prompt.preamble_tokens == len(tok(SHARED_PRE))
    assert pred.prompt.tail_tokens == len(tok(question))


TWO_ONS = """
    SELECT a.id FROM reviews a
    JOIN threads b
      ON AI_FILTER(PROMPT('x {0} {1}', a.review, b.thread),
                   {'selectivity': 0.5})
    JOIN products p
      ON AI_FILTER(PROMPT('y {0} {1}', b.thread, p.description),
                   {'selectivity': 0.2})
"""

TWO_WHERE = """
    SELECT a.id FROM reviews a, threads b, products p
    WHERE AI_FILTER(PROMPT('x {0} {1}', a.review, b.thread),
                    {'selectivity': 0.5})
      AND AI_FILTER(PROMPT('y {0} {1}', b.thread, p.description),
                    {'selectivity': 0.2})
"""

PAIR_JOIN_SQL = """
    SELECT r.id, p.asin
    FROM reviews r
    JOIN products p
      ON r.id = p.asin
     AND AI_FILTER(PROMPT('Review {0} discusses product {1}',
                          r.review, p.description),
                   {'selectivity': 0.05})
    WHERE AI_FILTER(PROMPT('This review is negative: {0}', r.review))
"""


def test_join_predicates_semantics_and_equality_pruning(catalog):
    # a chain: each multi-table AI_FILTER is its own pairwise join
    plan = compile_sql(TWO_ONS, catalog, tok)
    outer = plan.root.input
    inner = join_outer_input(outer)
    assert isinstance(outer, SemanticJoin) and isinstance(inner, SemanticJoin)
    assert [r.alias for r in inner.prompt.args] == ["a", "b"]
    assert [r.alias for r in outer.prompt.args] == ["b", "p"]
    assert inner.selectivity == 0.5 and outer.selectivity == 0.2
    built = (docs(catalog, "reviews", tok).alias("a")
             .ai_join(docs(catalog, "threads", tok).alias("b"),
                      prompt("x {0} {1}", col("a.review"),
                             col("b.thread")), selectivity=0.5)
             .ai_join(docs(catalog, "products", tok).alias("p"),
                      prompt("y {0} {1}", col("b.thread"),
                             col("p.description")), selectivity=0.2)
             .select("a.id"))
    assert plan == compile_sql(TWO_WHERE, catalog, tok) == built

    sql = """
        SELECT t.id FROM threads t
        WHERE AI_FILTER(PROMPT('good: {0}', t.thread))
          AND NOT EXISTS (SELECT 1 FROM products s
                          WHERE AI_FILTER(PROMPT('match {0} {1}',
                                                 t.thread,
                                                 s.description)))
    """
    plan = compile_sql(sql, catalog, tok)
    join = plan.root.input
    assert isinstance(join, SemanticJoin)
    # the gate applies to the outer table, so it always anchors there
    assert (join.semantics, join.anchor) == ("anti", "t")
    match = prompt("match {0} {1}", col("t.thread"), col("s.description"))
    built = (docs(catalog, "threads", tok).alias("t")
             .ai_filter(prompt("good: {0}", col("t.thread")))
             .ai_join(docs(catalog, "products", tok).alias("s"), match,
                      semantics="anti")
             .select("t.id"))
    assert built == plan
    exists_sql = sql.replace("NOT EXISTS", "EXISTS")
    assert compile_sql(exists_sql, catalog, tok).root.input.semantics == "exists"
    with pytest.raises(CompileError, match="outer table"):
        docs(catalog, "threads", tok).alias("t").ai_join(
            docs(catalog, "products", tok).alias("s"), match,
            semantics="exists", anchor="s")

    # two predicates over the same two tables: a pair must answer TRUE
    # to both, and only the first carries the joined table into the tree
    plan = compile_sql(
        "SELECT r.id FROM reviews r JOIN products p ON AI_FILTER("
        "PROMPT('x {0} {1}', r.review, p.description)) "
        "WHERE AI_FILTER(PROMPT('y {0} {1}', r.review, p.description))",
        catalog, tok)
    outer = plan.root.input
    inner = outer.input
    assert isinstance(outer, SemanticJoin) and isinstance(inner, SemanticJoin)
    assert isinstance(inner.input, Join)
    assert [r.alias for r in inner.prompt.args] == ["r", "p"]
    assert [r.alias for r in outer.prompt.args] == ["r", "p"]

    # an ON equality prunes the pairs before the AI predicate sees them
    plan = compile_sql(PAIR_JOIN_SQL, catalog, tok)
    join = plan.root.input
    assert isinstance(join, SemanticJoin)
    pairs = join.input
    assert isinstance(pairs, Join)
    assert [str(condition) for condition in pairs.on] == ["r.id = p.asin"]
    assert join_conditions(join) == pairs.on
    assert pairs.on[0].left == ColumnRef("r", "reviews", "id")
    assert pairs.on[0].right == ColumnRef("p", "products", "asin")
    # the equality columns are loaded as values for the pair table
    optimized, _ = apply_logical_rules(
        plan, built_in_logical_rules(), LogicalPlanningContext(catalog, None))
    scans = {node.alias: node for node in optimized.walk()
             if isinstance(node, Scan)}
    assert "id" in scans["r"].columns and "asin" in scans["p"].columns
    built = (docs(catalog, "reviews", tok).alias("r")
             .ai_filter(prompt("This review is negative: {0}",
                               col("r.review")))
             .join(docs(catalog, "products", tok).alias("p"),
                   on=col("r.id") == col("p.asin"))
             .ai_filter(prompt("Review {0} discusses product {1}",
                               col("r.review"), col("p.description")),
                        selectivity=0.05)
             .select("r.id", "p.asin"))
    assert built == plan
    where_plan = compile_sql("""
        SELECT r.id, p.asin FROM reviews r
        JOIN products p ON r.id = p.asin
        WHERE AI_FILTER(PROMPT('Review {0} discusses product {1}',
                               r.review, p.description),
                        {'selectivity': 0.05})
          AND AI_FILTER(PROMPT('This review is negative: {0}', r.review))
    """, catalog, tok)
    assert where_plan == plan


REJECTED = [
    ("SELECT a.id FROM reviews a, threads b, products p WHERE AI_FILTER("
     "PROMPT('x {0} {1}', b.thread, p.description))", "connected graph"),
    ("SELECT a.id FROM reviews a, threads b, products p WHERE AI_FILTER("
     "PROMPT('x {0} {1}', a.review, b.thread))", "every JOINed table"),
    ("SELECT a.id FROM reviews a JOIN threads b", "no join predicate"),
    ("SELECT a.id FROM reviews a JOIN threads b ON AI_FILTER("
     "PROMPT('x {0} {1} {2}', a.review, b.thread, a.review))", "distinct table"),
    ("SELECT r.id FROM reviews r JOIN products p ON r.id < p.asin "
     "AND AI_FILTER(PROMPT('x {0} {1}', r.review, p.description))",
     "column = column"),
    ("SELECT r.id FROM reviews r JOIN products p ON r.id = r.review "
     "AND AI_FILTER(PROMPT('x {0} {1}', r.review, p.description))",
     "table already in the query"),
    ("SELECT r.id FROM reviews r JOIN threads t ON t.id = r.id "
     "JOIN products p ON AI_FILTER(PROMPT('x {0} {1}', r.review, "
     "p.description)) WHERE AI_FILTER(PROMPT('y {0} {1}', "
     "t.thread, p.description))", "names a table the AI predicate"),
    ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('x: {0}', "
     "r.review)) LIMIT 0", "positive integer"),
    ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('x: {0}', "
     "r.review)) LIMIT 'five'", ""),
    ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('x {0}', "
     "r.review)) GROUP BY r.id", "GROUP BY"),
    ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('x {0}', "
     "r.review)) ORDER BY r.id", "ORDER BY"),
    ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('x {0}', "
     "r.review)) LIMIT 10 OFFSET 5", "OFFSET"),
    ("SELECT DISTINCT r.id FROM reviews r WHERE AI_FILTER("
     "PROMPT('x {0}', r.review))", "DISTINCT"),
    ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('x {0}', "
     "r.review)) OR AI_FILTER(PROMPT('y {0}', r.review))",
     "disjunction"),
    ("SELECT r.id FROM reviews r WHERE AI_CLASSIFY(PROMPT('x {0}', "
     "r.review))", "AI_CLASSIFY"),
    ("SELECT r.id FROM reviews r JOIN products p ON r.id = p.asin",
     "no AI predicate"),
    ("SELECT r.id FROM reviews r LEFT JOIN products p ON AI_FILTER("
     "PROMPT('x {0} {1}', r.review, p.description))", "plain JOIN"),
    ("SELECT r.id + 1 FROM reviews r WHERE AI_FILTER(PROMPT('x {0}', "
     "r.review))", "column selection"),
    ("SELECT r.id FROM reviews r", "no AI predicate"),
    ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('x {0}', "
     "r.review), {'selectivty': 0.3})", "unknown option"),
    ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('x {0} {1}', "
     "r.review), {'selectivity': 0.3})", "placeholders"),
    ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('x {0}', "
     "r.nope))", "not in provider"),
    ("SELECT r.id FROM nowhere r WHERE AI_FILTER(PROMPT('x {0}', "
     "r.review))", "unknown provider"),
    ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT("
     "'x {0} {1}', r.review, r.id))", "two columns"),
    ("SELECT r.id FROM reviews r JOIN products p ON AI_FILTER("
     "PROMPT('x {0} {1}', r.review, p.description), "
     "{'anchor': 'z'})", "anchor"),
    ("SELECT r.id FROM reviews r WHERE r.id IN (SELECT p.asin FROM "
     "products p)", "subquer"),
]


def test_builder_and_sql_reject_invalid_queries(catalog):
    for sql, fragment in REJECTED:
        try:
            compile_sql(sql, catalog, tok)
        except CompileError as error:
            assert fragment.lower() in str(error).lower(), sql
        else:
            raise AssertionError(f"compiled: {sql}")

    def reviews():
        return docs(catalog, "reviews", tok).alias("r")

    def table(name, alias):
        return docs(catalog, name, tok).alias(alias)

    with pytest.raises(CompileError, match="no AI predicate"):
        reviews().join(table("products", "p"),
                       on=col("r.id") == col("p.asin")).select("r.id")
    with pytest.raises(CompileError, match="must name the joined table"):
        (reviews()
         .ai_join(table("threads", "t"),
                  prompt("x {0} {1}", col("r.review"), col("t.thread")))
         .join(table("products", "p"))
         .ai_filter(prompt("x {0} {1}", col("r.review"), col("t.thread"))))
    with pytest.raises(CompileError, match="waiting for the ai_filter"):
        reviews().join(table("products", "p")).join(table("threads", "t"))
    with pytest.raises(CompileError, match="must relate the joined table"):
        reviews().join(table("products", "p"), on=col("r.id") == col("r.review"))

    q = reviews().ai_join(
        table("threads", "b"), prompt("x {0} {1}", col("r.review"), col("b.thread")))
    with pytest.raises(CompileError, match="joined but not referenced"):
        q.ai_join(table("products", "p"),
                  prompt("y {0} {1}", col("r.review"), col("b.thread")))
    with pytest.raises(CompileError, match="at least one table"):
        reviews().ai_join(
            [table("threads", "b"), table("products", "p")],
            prompt("y {0} {1}", col("b.thread"), col("p.description")))
    with pytest.raises(CompileError):
        docs(catalog, "reviews", tok).select("id")
    with pytest.raises(CompileError):
        docs(catalog, "nowhere")
    with pytest.raises(CompileError):
        reviews().ai_filter(prompt("x {0} {1}", col("r.review"), col("r.id")))

    plan = compile_sql("SELECT r.id FROM reviews r WHERE AI_FILTER("
                       "PROMPT('x: {0}', r.review)) LIMIT 5", catalog, tok)
    assert plan.root.limit == 5
    built = (reviews().ai_filter(prompt("x: {0}", col("r.review")))
             .limit(3).select("r.id"))
    assert built.root.limit == 3


def test_builder_places_apply_nodes_in_the_logical_tree():
    cat = Catalog()
    cat.register("claims", DocumentProvider.from_table(pa.table({
        "id": ["c0", "c1"], "claim": ["a b", "c d"], "url": ["u", "v"]}),
        id_col="id"))
    cat.register("evidence", DocumentProvider.from_table(pa.table({
        "id": ["u", "v"], "text": ["e f", "g h"]}), id_col="id"))
    plan = (docs(cat, "claims", tok).alias("c")
            .ai_filter(prompt("about a person: {0}", col("c.claim")))
            .apply(keep_even, columns=[col("c.url")])
            .join(docs(cat, "evidence", tok).alias("e"))
            .apply(same_key, columns=[col("c.url"), col("e.id")],
                   kind="barrier")
            .ai_filter(prompt("{1} supports {0}", col("c.claim"),
                              col("e.text")))
            .select("c.id", "e.id"))
    join = plan.root.input
    assert isinstance(join, SemanticJoin)
    (pairs,) = join_applies(join)
    assert (pairs.function, pairs.kind, pairs.ids, pairs.written_pos) == (
        "same_key", "barrier", "pairs", 0)
    assert pairs.aliases == ("c", "e")
    assert isinstance(pairs.input, Join)
    chain = pairs.input.left
    assert isinstance(chain, Apply)
    assert (chain.function, chain.kind, chain.ids, chain.aliases) == (
        "keep_even", "per_batch", "drop", ("c",))
    assert isinstance(chain.input, SemanticFilter)
    assert [str(ref.column) for ref in chain.columns] == ["url"]

    base = docs(cat, "claims", tok).alias("c")
    with pytest.raises(CompileError, match="must work on one table"):
        base.apply(keep_even)
    with pytest.raises(CompileError, match="needs a name for a lambda"):
        base.apply(lambda tables: [], columns=[col("c.url")])
    with pytest.raises(CompileError, match="returning pairs follows join"):
        base.apply(keep_even, columns=[col("c.url")], ids="pairs")
    with pytest.raises(CompileError, match="already used by another"):
        (base.apply(keep_even, columns=[col("c.url")])
         .apply(same_key, columns=[col("c.url")], name="keep_even"))
    with pytest.raises(CompileError, match="ids must be 'pairs'"):
        (base.join(docs(cat, "evidence", tok).alias("e"))
         .apply(same_key, columns=[col("c.url")], ids="preserve"))
