"""Parse supported Snowflake and BigQuery AI SQL into a LogicalPlan."""

from enum import StrEnum

import sqlglot
from sqlglot import exp

from quail.catalog import Catalog
from quail.logical import (
    Alias,
    ColumnRef,
    Compare,
    CompileError,
    Equality,
    FilterPredicate,
    JoinSpec,
    LogicalPlan,
    LogicalPlanBuilder,
    ModelCall,
    bind_extract_prompt,
    bind_join_prompt,
    bind_prompt,
    bind_score_prompt,
    is_score,
)

# Every relational operator except the projection, named and refused.
# OR is rejected separately with its own message.
FORBIDDEN = (
    (exp.Group, "GROUP BY"),
    (exp.Order, "ORDER BY"),
    (exp.Distinct, "DISTINCT"),
    (exp.Having, "HAVING"),
    (exp.Qualify, "QUALIFY"),
    (exp.Window, "window functions"),
    (exp.Union, "UNION"),
    (exp.Except, "EXCEPT"),
    (exp.Intersect, "INTERSECT"),
    (exp.Offset, "OFFSET"),
)

# one option surface: anchor is rejected after parsing when the
# predicate turns out to be a one-provider filter
JOIN_OPTION_KEYS = {"selectivity", "anchor"}
SCORE_OPTION_KEYS = {"selectivity"}


class SQLDialect(StrEnum):
    """SQL dialects accepted by the Quail front end."""

    SNOWFLAKE = "snowflake"
    BQ = "bq"


def _sqlglot_dialect(dialect: SQLDialect) -> str:
    return "bigquery" if dialect is SQLDialect.BQ else "snowflake"


def _normalize_ai_calls(sql: str, dialect: SQLDialect) -> str:
    """Lower dotted AI calls to internal function names."""
    tokenizer = sqlglot.Dialect.get_or_raise(
        _sqlglot_dialect(dialect)
    ).tokenizer()
    try:
        tokens = tokenizer.tokenize(sql)
    except sqlglot.errors.SqlglotError as error:
        raise CompileError(f"parse error: {error}") from error
    edits = []
    for index in range(len(tokens) - 3):
        head, dot, name, left = tokens[index:index + 4]
        function = name.text.upper()
        replacement = None
        if function in {"SCORE", "EXTRACT"}:
            replacement = f"AI_{function}("
        elif function == "IF" and dialect is SQLDialect.BQ:
            replacement = "AI_FILTER("
        if head.text.upper() == "AI" and dot.text == "." \
                and left.text == "(" and replacement is not None:
            edits.append((head.start, left.end + 1, replacement))
    for start, end, replacement in reversed(edits):
        sql = sql[:start] + replacement + sql[end:]
    return sql


def _is_call(node, name: str) -> bool:
    return (isinstance(node, exp.Anonymous)
            and str(node.this).upper() == name)


# each comparison class with its spelling when AI.SCORE is on the left
# and the spelling that means the same when AI.SCORE is on the right
_COMPARISON_CLASSES = {
    exp.LT: ("<", ">"),
    exp.LTE: ("<=", ">="),
    exp.GT: (">", "<"),
    exp.GTE: (">=", "<="),
}


def _score_comparison(node):
    """Return (comparison, AI_SCORE call, threshold) or None."""
    for kind, (left, right) in _COMPARISON_CLASSES.items():
        if not isinstance(node, kind):
            continue
        if _is_call(node.this, "AI_SCORE"):
            return left, node.this, node.expression
        if _is_call(node.expression, "AI_SCORE"):
            return right, node.expression, node.this
    return None


def _is_ai_score_comparison(node) -> bool:
    return _score_comparison(node) is not None


def _reject_forbidden(tree) -> None:
    for cls, name in FORBIDDEN:
        if list(tree.find_all(cls)):
            raise CompileError(
                f"{name} is outside the language: Quail runs the "
                f"semantic part; do the relational part in the "
                f"database the ids came from")
    if list(tree.find_all(exp.Or)):
        raise CompileError(
            "OR between AI predicates is not supported: a disjunction "
            "belongs inside one prompt's text, where the model "
            "evaluates it")
    for fn in tree.find_all(exp.Anonymous):
        name = str(fn.this).upper()
        if name.startswith("AI_") and name not in {
            "AI_FILTER", "AI_SCORE", "AI_EXTRACT",
        }:
            raise CompileError(
                f"{name} is not supported; supported AI functions are "
                "AI_FILTER, AI_SCORE, and AI_EXTRACT"
            )
    # subqueries are legal only as the EXISTS form, checked
    # structurally; any other subquery is refused here
    for sub in tree.find_all(exp.Subquery):
        raise CompileError("subqueries other than "
                           "[NOT] EXISTS (SELECT 1 ...) are not "
                           "supported")


class _Binder:
    def __init__(self, catalog: Catalog, tokenizer, turn=("", "")):
        self.catalog = catalog
        self.tokenizer = tokenizer
        self.turn = turn
        self.tables = []          # (alias, provider) in appearance order
        self.doc_columns = {}     # alias -> document column
        self.filters = {}         # alias -> [FilterPredicate]
        self.joins = []           # [JoinSpec]

    # ---- scope ------------------------------------------------------

    def add_table(self, table: exp.Table) -> str:
        name = table.name
        alias = table.alias or name
        self.catalog.get(name)    # unknown provider -> CompileError
        if alias in dict(self.tables):
            raise CompileError(f"duplicate table alias {alias!r}")
        self.tables.append((alias, name))
        return alias

    def resolve_column(self, col: exp.Column, scope=None) -> ColumnRef:
        scope = dict(self.tables) if scope is None else dict(scope)
        column = col.name
        if col.table:
            if col.table not in scope:
                raise CompileError(f"unknown table alias {col.table!r} "
                                   f"in column {col.sql()}")
            provider = scope[col.table]
            if column not in self.catalog.get(provider).columns:
                raise CompileError(
                    f"column {column!r} not in provider {provider!r} "
                    f"(schema: {self.catalog.get(provider).columns})")
            return ColumnRef(alias=col.table, provider=provider,
                             column=column)
        owners = [(a, p) for a, p in scope.items()
                  if column in self.catalog.get(p).columns]
        if len(owners) != 1:
            raise CompileError(
                f"column {column!r} is {'ambiguous' if owners else 'unknown'};"
                f" qualify it with a table alias")
        return ColumnRef(alias=owners[0][0], provider=owners[0][1],
                         column=column)

    def note_doc_column(self, ref: ColumnRef) -> None:
        seen = self.doc_columns.get(ref.alias)
        if seen and seen != ref.column:
            raise CompileError(
                f"alias {ref.alias!r} is referenced through two "
                f"columns ({seen!r}, {ref.column!r}); predicates over "
                f"one table must share one document column so its KV "
                f"is computed once")
        self.doc_columns[ref.alias] = ref.column

    # ---- AI predicate parsing ----------------------------------------

    def parse_options(self, node, allowed: set) -> dict:
        if node is None:
            return {}
        if not isinstance(node, exp.Struct):
            raise CompileError(
                f"the second AI operator argument must be an option "
                f"object like {{'selectivity': 0.3}}, got {node.sql()}")
        out = {}
        for prop in node.expressions:
            if not isinstance(prop, exp.PropertyEQ):
                raise CompileError(f"malformed option {prop.sql()}")
            key = str(prop.this.name)
            if key not in allowed:
                raise CompileError(
                    f"unknown option key {key!r}; allowed: "
                    f"{sorted(allowed)}")
            value = prop.expression
            if key == "selectivity":
                if not (isinstance(value, exp.Literal)
                        and not value.is_string):
                    raise CompileError("selectivity must be a number")
                out[key] = float(value.this)
            else:   # anchor
                if not (isinstance(value, exp.Literal)
                        and value.is_string):
                    raise CompileError("anchor must be a table alias "
                                       "string")
                out[key] = str(value.this)
        return out

    def parse_ai_call(self, node, function: str, allowed: set, scope=None,
                      join=None):
        """Parse one AI prompt call into prompt, options, and aliases."""
        if not _is_call(node, function):
            raise CompileError(
                f"only {function}(PROMPT(...)) predicates are "
                f"supported here, got: {node.sql()}")
        args = node.expressions
        if not args or not _is_call(args[0], "PROMPT"):
            raise CompileError(
                f"{function}'s first argument must be "
                "PROMPT('template', columns...)"
            )
        if len(args) > 2:
            raise CompileError(
                f"{function} takes PROMPT and at most one option object"
            )
        options = self.parse_options(args[1] if len(args) == 2 else None,
                                     allowed)
        p_args = args[0].expressions
        if not p_args or not (isinstance(p_args[0], exp.Literal)
                              and p_args[0].is_string):
            raise CompileError("PROMPT's first argument must be a "
                               "string template")
        template = str(p_args[0].this)
        refs = []
        for a in p_args[1:]:
            if not isinstance(a, exp.Column):
                raise CompileError(f"PROMPT arguments must be column "
                                   f"references, got {a.sql()}")
            refs.append(self.resolve_column(a, scope))
        aliases = []
        for r in refs:
            if r.alias not in aliases:
                aliases.append(r.alias)
        if join is None:
            join = len(aliases) > 1
        if function == "AI_SCORE" and join and template.count("{0}") != 1:
            raise CompileError(
                "an AI.SCORE pair prompt mentions {0} exactly once, as "
                "the place its document is inserted; refer to it again "
                "in words")
        if function == "AI_SCORE":
            binder = bind_score_prompt
        else:
            binder = bind_join_prompt if join else bind_prompt
        prompt = binder(template, tuple(refs), self.tokenizer,
                        turn=self.turn)
        for r in refs:
            self.note_doc_column(r)
        return prompt, options, aliases

    def parse_ai_filter(self, node, allowed: set, scope=None, join=None):
        """Parse an AI_FILTER node into a boolean call, options, and aliases."""
        prompt, options, aliases = self.parse_ai_call(
            node, "AI_FILTER", allowed, scope=scope, join=join
        )
        return ModelCall(prompt, "boolean"), options, aliases

    def parse_ai_extract(self, node):
        args = node.expressions
        if len(args) < 2 or not isinstance(args[0], exp.Column):
            raise CompileError(
                "AI.EXTRACT takes a document column and string field names"
            )
        if any(not isinstance(arg, exp.Literal) or not arg.is_string
               for arg in args[1:]):
            raise CompileError("AI.EXTRACT field names must be string literals")
        document = self.resolve_column(args[0])
        import pyarrow as pa

        dtype = self.catalog.get(document.provider).schema().field(
            document.column
        ).type
        if not (pa.types.is_string(dtype) or pa.types.is_large_string(dtype)):
            raise CompileError("AI.EXTRACT needs a text document column")
        fields = tuple(str(arg.this) for arg in args[1:])
        prompt = bind_extract_prompt(document, fields, self.tokenizer, self.turn)
        call = ModelCall(prompt, "extract", fields)
        call.validate()
        self.note_doc_column(document)
        return call

    def parse_ai_score(self, node, allowed: set, scope=None, join=None):
        """Parse a compared AI.SCORE call into a Compare, options, and aliases.

        The threshold may be on either side of the comparison.
        """
        parsed = _score_comparison(node)
        if parsed is None:
            raise CompileError(
                "AI.SCORE must be compared with <, <=, >, or >="
            )
        comparison, call, threshold = parsed
        if not isinstance(threshold, exp.Literal) or threshold.is_string:
            raise CompileError("AI.SCORE threshold must be a number")
        value = float(threshold.this)
        if not 0.0 <= value <= 1.0:
            raise CompileError("AI.SCORE threshold must be between 0 and 1")
        prompt, options, aliases = self.parse_ai_call(
            call,
            "AI_SCORE",
            allowed,
            scope=scope,
            join=join,
        )
        expression = Compare(ModelCall(prompt, "score"), comparison, value)
        return expression, options, aliases


def _parse_equality(b, term, joined_alias: str) -> Equality:
    """Parse one ordinary ON condition: two columns compared with =."""
    if not isinstance(term, exp.EQ) or not isinstance(term.this, exp.Column) \
            or not isinstance(term.expression, exp.Column):
        raise CompileError(
            f"ON supports column = column and AI predicates, got "
            f"{term.sql()}")
    left = b.resolve_column(term.this)
    right = b.resolve_column(term.expression)
    sides = {left.alias, right.alias}
    if joined_alias not in sides or len(sides) != 2:
        raise CompileError(
            f"JOIN {joined_alias} ON {term.sql()} must relate "
            f"{joined_alias!r} to a table already in the query")
    return Equality(left, right)


def _from_clause(select):
    return select.args.get("from_")


def _conjuncts(node) -> list:
    """Flatten one AND tree into its terms, in written order."""
    terms, stack = [], [node]
    while stack:
        n = stack.pop()
        if isinstance(n, exp.And):
            stack.append(n.expression)
            stack.append(n.this)
        else:
            terms.append(n)
    return terms       # the pop order above yields written order


def _parse_limit(tree) -> int | None:
    """Extract a plain LIMIT N from the parse tree."""
    limit_node = tree.args.get("limit")
    if limit_node is None:
        return None
    expr = limit_node.expression
    if not isinstance(expr, exp.Literal) or expr.is_string:
        raise CompileError("LIMIT must be a positive integer")
    value = int(expr.this)
    if value <= 0:
        raise CompileError("LIMIT must be a positive integer")
    return value


def compile_sql(sql: str, catalog: Catalog,
                tokenizer=None,
                dialect: SQLDialect | str = SQLDialect.SNOWFLAKE,
                turn: tuple[str, str] = ("", ""),
                ) -> LogicalPlan:
    """Compile AI SQL text into a LogicalPlan.

    turn is the model's chat-turn text wrapped around every prompt.
    """
    try:
        dialect = SQLDialect(dialect)
    except ValueError as error:
        raise ValueError(
            f"unsupported SQL dialect {dialect!r}; expected snowflake "
            "or bq"
        ) from error
    sql = _normalize_ai_calls(sql, dialect)
    try:
        tree = sqlglot.parse_one(sql, dialect=_sqlglot_dialect(dialect))
    except sqlglot.errors.SqlglotError as e:
        raise CompileError(f"parse error: {e}") from e
    if not isinstance(tree, exp.Select):
        raise CompileError("the query must be a single SELECT")
    _reject_forbidden(tree)

    limit = _parse_limit(tree)

    b = _Binder(catalog, tokenizer, turn)

    from_ = _from_clause(tree)
    if from_ is None or not isinstance(from_.this, exp.Table):
        raise CompileError("FROM must name one registered provider")
    b.add_table(from_.this)

    joined_aliases = []       # tables brought in by JOIN clauses
    on_preds = []
    conditions = {}           # joined alias -> its ON equalities
    for join in tree.args.get("joins") or []:
        if join.side or (join.kind
                         and join.kind.upper() not in ("INNER",
                                                       "CROSS")):
            raise CompileError(
                f"only plain JOIN is supported, got "
                f"{join.side or ''} {join.kind or ''} JOIN".strip())
        if not isinstance(join.this, exp.Table):
            raise CompileError("JOIN must name one registered provider")
        alias = b.add_table(join.this)
        joined_aliases.append(alias)
        on = join.args.get("on")
        if on is None:
            continue    # a bare/cross-joined table: some join
            #             predicate must cover it (checked below)
        # ON: equalities choose the pairs; at most one AI predicate
        equalities, predicates = [], []
        for term in _conjuncts(on):
            if _is_call(term, "AI_FILTER"):
                predicates.append(b.parse_ai_filter(
                    term, JOIN_OPTION_KEYS, join=True
                ))
            elif _is_ai_score_comparison(term):
                predicates.append(b.parse_ai_score(
                    term, SCORE_OPTION_KEYS, join=True
                ))
            else:
                if any(
                    _is_call(call, "AI_SCORE") for call in term.walk()
                ):
                    raise CompileError(
                        "AI.SCORE must be compared with <, <=, >, or >="
                    )
                equalities.append(_parse_equality(b, term, alias))
        if len(predicates) > 1:
            raise CompileError(
                f"JOIN {alias} has {len(predicates)} AI predicates in "
                f"ON; ask one question per JOIN")
        if equalities:
            conditions[alias] = equalities
        on_preds.extend(predicates)

    claimed = set()           # joined tables already carried by a spec

    def add_join_spec(predicate, options, aliases):
        joinable = {b.tables[0][0], *joined_aliases}
        outside = [a for a in aliases if a not in joinable]
        if outside:
            raise CompileError(
                f"the join prompt references {outside}, which are not "
                f"the FROM table or JOINed tables of this query "
                f"({sorted(joinable)})")
        anchor = options.get("anchor")
        if anchor is not None and anchor not in aliases:
            raise CompileError(
                f"anchor {anchor!r} is not a table of this join "
                f"({aliases})")
        # each spec carries the joined tables its prompt references
        # that no earlier spec carried, so the logical builder adds
        # every table to the join tree exactly once
        news = tuple(a for a in joined_aliases
                     if a in aliases and a not in claimed)
        claimed.update(news)
        on = tuple(condition for a in news
                   for condition in conditions.pop(a, ()))
        if on and len(aliases) != 2:
            raise CompileError(
                f"join conditions are supported on two-table AI "
                f"predicates; this one names {aliases}")
        for condition in on:
            if not set(condition.aliases()) <= set(aliases):
                raise CompileError(
                    f"join condition {condition} names a table the AI "
                    f"predicate over {aliases} does not")
        b.joins.append(JoinSpec(aliases=news,
                                predicate=predicate, semantics="full",
                                selectivity=options.get("selectivity"),
                                anchor=anchor, on=on))

    for pred in on_preds:
        add_join_spec(*pred)

    where = tree.args.get("where")
    for term in _conjuncts(where.this) if where else []:
        anti = False
        node = term
        if isinstance(node, exp.Not):
            node, anti = node.this, True
        if isinstance(node, exp.Exists):
            _compile_exists(b, node, anti)
            continue
        if anti:
            raise CompileError(f"NOT is only supported as NOT EXISTS, "
                               f"got NOT {node.sql()}")
        if _is_ai_score_comparison(term):
            predicate, options, aliases = \
                b.parse_ai_score(term, SCORE_OPTION_KEYS)
        else:
            if any(_is_call(call, "AI_SCORE") for call in term.walk()):
                raise CompileError(
                    "AI.SCORE must be compared with <, <=, >, or >="
                )
            predicate, options, aliases = b.parse_ai_filter(
                term, JOIN_OPTION_KEYS)
        if len(aliases) == 1:
            if "anchor" in options:
                raise CompileError(
                    "anchor is a join option; a one-provider "
                    "AI_FILTER takes only selectivity")
            b.filters.setdefault(aliases[0], []).append(
                FilterPredicate(expression=predicate,
                                selectivity=options.get("selectivity")))
            continue
        # a multi-provider WHERE predicate is a join predicate,
        # BigQuery style: tables cross-joined in FROM, filtered here
        add_join_spec(predicate, options, aliases)

    columns = _compile_projection(b, tree.expressions)
    projected_calls = tuple(
        column for column in columns if isinstance(column, Alias)
    )
    projected_scores = tuple(
        column for column in projected_calls if is_score(column)
    )
    extracts = tuple(
        column for column in projected_calls
        if column.expression.kind == "extract"
    )
    if extracts and len(b.tables) != 1:
        raise CompileError("AI.EXTRACT currently supports one table without joins")
    names_by_prompt = {}
    for column in projected_calls:
        function = "AI.SCORE" if is_score(column) else "AI.EXTRACT"
        if column.name in dict(b.tables):
            raise CompileError(
                f"{function} output name {column.name!r} is also a table "
                f"alias; pick another AS name")
        names = names_by_prompt.setdefault(column.expression, [])
        if names:
            raise CompileError(
                f"the same {function} expression is projected as "
                f"{names[0]!r} and {column.name!r}; project it once")
        names.append(column.name)

    for alias, equalities in conditions.items():
        if any(alias in score.expression.aliases()
               for score in projected_scores):
            raise CompileError(
                f"JOIN {alias} ON {equalities[0]} projects a pair "
                f"AI.SCORE; a projected pair score runs over CROSS JOIN, "
                f"and an ON equality is supported when the score is "
                f"compared in ON")
        raise CompileError(
            f"JOIN {alias} ON {equalities[0]} has no AI predicate over "
            f"its pairs; a plain join belongs in the database the ids "
            f"came from")
    _check_join_coverage(b, joined_aliases, projected_scores)

    filter_predicates = [
        predicate
        for predicates in b.filters.values()
        for predicate in predicates
    ]
    score_flags = [
        is_score(predicate.expression) for predicate in filter_predicates
    ] + [is_score(join.predicate) for join in b.joins] \
        + [is_score(column) for column in projected_calls]
    if any(score_flags) and not all(score_flags):
        raise CompileError(
            "AI.SCORE cannot be mixed with generative AI predicates "
            "in one query"
        )

    if not b.joins and not b.filters and not projected_calls:
        raise CompileError("the query has no AI predicate; a plain "
                           "scan belongs in the database the ids came "
                           "from")

    logical = LogicalPlanBuilder()
    for alias, provider in b.tables:
        logical.add_scan(
            alias,
            provider,
            b.doc_columns.get(alias, ""),
            tuple(b.filters.get(alias, ())),
        )
    for join in b.joins:
        logical.add_join(join)
    for alias in joined_aliases:
        if alias not in claimed:
            logical.add_cross_join(alias)
    return logical.project(tuple(columns), limit)


def _check_join_coverage(
    b: _Binder,
    joined_aliases: list,
    projected_scores: tuple[Alias, ...] = (),
) -> None:
    """Check that the join predicates cover and connect every JOINed table.

    Every JOINed table must appear in a join predicate, and all
    predicates must form one connected graph with the FROM table.
    """
    if not joined_aliases:
        return
    preds = [{r.alias for r in j.prompt.args}
             for j in b.joins if j.semantics == "full"]
    preds.extend(
        set(score.expression.aliases())
        for score in projected_scores
        if len(score.expression.aliases()) > 1
    )
    uncovered = [a for a in joined_aliases
                 if not any(a in p for p in preds)]
    if uncovered:
        raise CompileError(
            f"JOINed tables {uncovered} appear in no join predicate: "
            f"every JOINed table must appear in at least one "
            f"AI predicate on a JOIN's ON "
            f"or as a WHERE term")
    root = b.tables[0][0]
    reached = {root}
    grew = True
    while grew:
        grew = False
        for p in preds:
            if p & reached and not p <= reached:
                reached |= p
                grew = True
    disconnected = sorted(a for a in joined_aliases
                          if a not in reached)
    if disconnected:
        raise CompileError(
            f"join predicates do not connect {disconnected} to the "
            f"FROM table {root!r}: the predicates' tables must form "
            f"one connected graph with it")


def _compile_exists(b: _Binder, node: exp.Exists, anti: bool) -> None:
    inner = node.this
    if not isinstance(inner, exp.Select):
        raise CompileError("EXISTS must wrap SELECT 1 FROM provider "
                           "WHERE AI_FILTER(...)")
    inner_from = _from_clause(inner)
    if inner_from is None or not isinstance(inner_from.this, exp.Table):
        raise CompileError("the EXISTS subquery must scan one "
                           "registered provider")
    if (inner.args.get("joins") or inner.args.get("group")
            or len(inner.expressions) != 1):
        raise CompileError("the EXISTS subquery must be exactly "
                           "SELECT 1 FROM provider WHERE AI_FILTER(...)")
    alias = b.add_table(inner_from.this)
    where = inner.args.get("where")
    terms = _conjuncts(where.this) if where else []
    if len(terms) != 1:
        raise CompileError("the EXISTS subquery takes exactly one "
                           "AI_FILTER predicate")
    predicate, options, aliases = b.parse_ai_filter(terms[0],
                                                    JOIN_OPTION_KEYS,
                                                    join=True)
    if len(aliases) != 2 or alias not in aliases:
        raise CompileError(
            "the EXISTS predicate must reference the inner provider "
            "and exactly one outer provider")
    outer = next(a for a in aliases if a != alias)
    anchor = options.get("anchor")
    if anchor is not None and anchor != outer:
        raise CompileError(
            f"exists/anti always anchor on the outer table {outer!r} "
            f"- the gate applies to its documents - got anchor "
            f"{anchor!r}")
    b.joins.append(JoinSpec(aliases=(alias,), predicate=predicate,
                            semantics="anti" if anti else "exists",
                            selectivity=options.get("selectivity"),
                            anchor=outer))


def _compile_projection(b: _Binder, expressions) -> list:
    columns = []
    for e in expressions:
        if isinstance(e, exp.Star):
            for alias, provider in b.tables:
                for c in b.catalog.get(provider).columns:
                    columns.append(ColumnRef(alias=alias,
                                             provider=provider,
                                             column=c))
            continue
        alias = None
        if isinstance(e, exp.Alias):
            alias = e.alias
            e = e.this
        if _is_call(e, "AI_EXTRACT"):
            if not alias:
                raise CompileError("AI.EXTRACT needs an AS name in SELECT")
            columns.append(Alias(b.parse_ai_extract(e), alias))
            continue
        if any(_is_call(call, "AI_EXTRACT") for call in e.walk()):
            raise CompileError(
                "AI.EXTRACT in SELECT must be a direct expression with an AS name"
            )
        if _is_call(e, "AI_SCORE"):
            if not alias:
                raise CompileError("AI.SCORE needs an AS name in SELECT")
            prompt, _options, aliases = b.parse_ai_call(
                e, "AI_SCORE", set()
            )
            if len(aliases) not in {1, 2}:
                raise CompileError(
                    "projected AI.SCORE must reference one or two relations"
                )
            columns.append(Alias(ModelCall(prompt, "score"), alias))
            continue
        if any(_is_call(call, "AI_SCORE") for call in e.walk()):
            raise CompileError(
                "AI.SCORE in SELECT must be a direct expression with "
                "an AS name"
            )
        if not isinstance(e, exp.Column):
            raise CompileError(
                f"the SELECT list is column selection only, got "
                f"{e.sql()}: nothing computed, per the projection "
                f"contract")
        columns.append(b.resolve_column(e))
    return columns
