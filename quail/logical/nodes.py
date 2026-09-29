"""Logical plan nodes and the plan builder."""

from dataclasses import dataclass, replace
from typing import Any, ClassVar, Optional, Protocol


class CompileError(ValueError):
    """Raised when a query is malformed or outside the language."""


@dataclass(frozen=True)
class ColumnRef:
    alias: str       # table alias in the query ("r")
    provider: str    # provider name in the catalog ("reviews")
    column: str      # column name ("review")

    type_name: ClassVar[str] = "quail.column_ref"


@dataclass(frozen=True)
class Prompt:
    """A bound PROMPT call, split into preamble, frame, and tail.

    Token counts are filled at bind time when a tokenizer is given;
    None means the planner must supply counts.
    """
    template: str
    args: tuple    # tuple[ColumnRef, ...] in placeholder order
    preamble: str
    tail: str
    preamble_tokens: Optional[int] = None
    tail_tokens: Optional[int] = None
    frame: str = ""
    frame_tokens: Optional[int] = None
    # join prompts only: (alias, label_tokens, anchor_frame_tokens)
    # per placeholder, in order. Empty for filter prompts.
    labels: tuple = ()
    preamble_token_ids: tuple = ()
    tail_token_ids: tuple = ()
    label_token_ids: tuple = ()

    type_name: ClassVar[str] = "quail.prompt"


# the planner's estimate for a predicate written without a selectivity
DEFAULT_SELECTIVITY = 0.2

# the comparisons an AI.SCORE predicate may use, in SQL spelling
SCORE_COMPARISONS = ("<", "<=", ">", ">=")

# what a model call answers: "boolean" is AI.IF, yes or no;
# "score" is AI.SCORE; "extract" returns named string fields.
MODEL_CALL_KINDS = ("boolean", "score", "extract")


def effective_selectivity(selectivity: Optional[float]) -> float:
    """Return the selectivity the planner uses: the given one or the default."""
    return DEFAULT_SELECTIVITY if selectivity is None else selectivity


@dataclass(frozen=True)
class ModelCall:
    """One prompt asked of every row; the model answers with a value.

    The AI functions of the language are all model calls. They differ
    only in ``kind``, the type of the answer.
    """
    prompt: Prompt
    kind: str = "boolean"
    fields: tuple[str, ...] = ()

    type_name: ClassVar[str] = "quail.model_call"

    def aliases(self) -> tuple[str, ...]:
        """Return the table aliases the prompt reads, in placeholder order."""
        return tuple(dict.fromkeys(ref.alias for ref in self.prompt.args))

    def validate(self) -> None:
        if self.kind not in MODEL_CALL_KINDS:
            raise CompileError(
                f"model call kind must be one of {MODEL_CALL_KINDS}, "
                f"got {self.kind!r}")
        if self.kind == "extract":
            if not self.fields or any(
                not isinstance(name, str) or not name.strip() for name in self.fields
            ):
                raise CompileError("AI.EXTRACT needs nonempty string field names")
            if len(self.fields) != len(set(self.fields)):
                raise CompileError("AI.EXTRACT field names must be unique")
            if len(self.prompt.args) != 1:
                raise CompileError("AI.EXTRACT reads one document column")
        elif self.fields:
            raise CompileError("field names are only supported by AI.EXTRACT")


@dataclass(frozen=True)
class Compare:
    """A score compared with a threshold; answers yes or no."""
    call: ModelCall
    comparison: str
    threshold: float

    type_name: ClassVar[str] = "quail.compare"

    def validate(self) -> None:
        self.call.validate()
        if self.call.kind != "score":
            raise CompileError("only an AI.SCORE call compares with a threshold")
        if self.comparison not in SCORE_COMPARISONS:
            raise CompileError(
                f"unsupported AI.SCORE comparison {self.comparison!r}")
        if not 0.0 <= self.threshold <= 1.0:
            raise CompileError("AI.SCORE threshold must be between 0 and 1")


@dataclass(frozen=True)
class Alias:
    """A model call returned as a named result column."""
    expression: ModelCall
    name: str

    type_name: ClassVar[str] = "quail.alias"

    def validate(self) -> None:
        self.expression.validate()
        if not self.name:
            raise CompileError("a projected expression needs a name")


def model_call(expression) -> ModelCall:
    """Return the model call inside a predicate or projected expression."""
    if isinstance(expression, ModelCall):
        return expression
    if isinstance(expression, Compare):
        return expression.call
    if isinstance(expression, Alias):
        return expression.expression
    raise TypeError(f"{type(expression).__name__} holds no model call")


def is_score(expression) -> bool:
    """Return whether the expression computes an AI.SCORE value."""
    return model_call(expression).kind == "score"


def validate_predicate(expression) -> None:
    """Check that an expression answers yes or no for every row."""
    if isinstance(expression, Compare):
        expression.validate()
        return
    if isinstance(expression, ModelCall):
        expression.validate()
        if expression.kind != "boolean":
            raise CompileError(
                "an AI.SCORE call is a predicate only when compared with "
                "a threshold")
        return
    raise CompileError(
        f"a predicate is a model call or a comparison, got "
        f"{type(expression).__name__}")


@dataclass(frozen=True)
class FilterPredicate:
    """One predicate over a table's documents, with its selectivity hint."""
    expression: Any                       # ModelCall | Compare
    selectivity: Optional[float] = None   # fraction of documents that
    #                                       pass; None means not given

    type_name: ClassVar[str] = "quail.filter_predicate"

    @property
    def prompt(self) -> Prompt:
        """The prompt the predicate asks."""
        return model_call(self.expression).prompt


class LogicalNode(Protocol):
    """Node in a logical query plan."""

    type_name: ClassVar[str]

    def children(self) -> tuple["LogicalNode", ...]: ...

    def expressions(self) -> tuple[Any, ...]: ...

    def output_schema(self) -> tuple[ColumnRef, ...]: ...

    def validate(self) -> None: ...

    def with_children(
        self, children: tuple["LogicalNode", ...]
    ) -> "LogicalNode": ...

    def with_expressions(self, expressions: tuple[Any, ...]) -> "LogicalNode": ...

    def explain_fields(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Scan:
    """Which column of which provider supplies the document text.

    ``column`` is tokenized for the model. ``columns`` names the source
    columns kept as values for the result rows, and includes ``column``
    only when the query returns the document text itself. The
    projection pushdown rule fills ``columns``; before it runs the
    tuple is empty.
    """
    provider: str
    alias: str
    column: str
    columns: tuple = ()    # tuple[str, ...]

    type_name: ClassVar[str] = "quail.scan"

    def children(self) -> tuple:
        return ()

    def expressions(self) -> tuple:
        return ()

    def output_schema(self) -> tuple[ColumnRef, ...]:
        fields = [ColumnRef(self.alias, self.provider, self.column)]
        fields.extend(
            ColumnRef(self.alias, self.provider, column)
            for column in self.columns if column != self.column
        )
        return tuple(fields)

    def validate(self) -> None:
        if not self.provider or not self.alias or not self.column:
            raise CompileError("Scan needs a provider, alias, and column")
        if len(set(self.columns)) != len(self.columns):
            raise CompileError(
                f"Scan {self.alias!r} lists a column twice: {self.columns}")

    def with_children(self, children: tuple) -> "Scan":
        if children:
            raise CompileError("Scan has no children")
        return self

    def with_expressions(self, expressions: tuple) -> "Scan":
        if expressions:
            raise CompileError("Scan has no expressions")
        return self

    def explain_fields(self) -> dict:
        return {
            "provider": self.provider,
            "alias": self.alias,
            "column": self.column,
            "columns": list(self.columns),
        }


@dataclass(frozen=True)
class SemanticFilter:
    input: LogicalNode
    predicates: tuple    # tuple[FilterPredicate, ...], written order

    type_name: ClassVar[str] = "quail.semantic_filter"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return self.predicates

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return self.input.output_schema()

    def validate(self) -> None:
        if not self.predicates:
            raise CompileError("SemanticFilter needs at least one predicate")
        for predicate in self.predicates:
            validate_predicate(predicate.expression)
            if len(model_call(predicate.expression).aliases()) != 1:
                raise CompileError(
                    "a SemanticFilter predicate reads one table")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("SemanticFilter needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        if not expressions:
            raise CompileError("SemanticFilter needs at least one predicate")
        return replace(self, predicates=expressions)

    def explain_fields(self) -> dict:
        return {
            "predicates": len(self.predicates),
            "selectivities": [p.selectivity for p in self.predicates],
            "expressions": [_explain(p.expression) for p in self.predicates],
        }


@dataclass(frozen=True)
class Equality:
    """One ordinary join condition: two columns of two tables are equal."""
    left: ColumnRef
    right: ColumnRef

    type_name: ClassVar[str] = "quail.equality"

    def aliases(self) -> tuple[str, str]:
        return self.left.alias, self.right.alias

    def __str__(self) -> str:
        return (f"{self.left.alias}.{self.left.column} = "
                f"{self.right.alias}.{self.right.column}")


@dataclass(frozen=True)
class Join:
    """One binary relational join; ``on`` empty means a cross join.

    The pairs it produces are the tuples a SemanticJoin above it asks
    the model about. Every Equality names one column on each side.
    """
    left: LogicalNode
    right: LogicalNode
    on: tuple = ()    # tuple[Equality, ...]

    type_name: ClassVar[str] = "quail.join"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.left, self.right)

    def expressions(self) -> tuple:
        return self.on

    def output_schema(self) -> tuple[ColumnRef, ...]:
        fields = list(self.left.output_schema())
        fields.extend(field for field in self.right.output_schema()
                      if field not in fields)
        return tuple(fields)

    def validate(self) -> None:
        left = {field.alias for field in self.left.output_schema()}
        right = {field.alias for field in self.right.output_schema()}
        for condition in self.on:
            sides = set(condition.aliases())
            if not (sides & left and sides & right) or len(sides) != 2:
                raise CompileError(
                    f"join condition {condition} must name one table "
                    f"on each side of the join ({sorted(left)} and "
                    f"{sorted(right)})")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 2:
            raise CompileError("Join needs two children")
        return replace(self, left=children[0], right=children[1])

    def with_expressions(self, expressions: tuple):
        return replace(self, on=tuple(expressions))

    def explain_fields(self) -> dict:
        return {"on": [str(condition) for condition in self.on]
                or "cross"}


@dataclass(frozen=True)
class SemanticJoin:
    """One model predicate over the pairs its input Join produces.

    A query that joins three or more tables has one SemanticJoin per
    pair, each feeding the Join of the next.
    """
    input: LogicalNode                 # a Join, under any Apply that
    #                                    returns its pairs
    predicate: Any                     # ModelCall | Compare
    semantics: str = "full"            # full | exists | anti
    selectivity: Optional[float] = None    # fraction of tuples that pass
    anchor: Optional[str] = None       # table alias whose KV is kept;
    #                                    None = planner picks

    type_name: ClassVar[str] = "quail.semantic_join"

    @property
    def prompt(self) -> Prompt:
        """The prompt asked of every pair."""
        return model_call(self.predicate).prompt

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return (self.predicate,)

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return self.input.output_schema()

    def validate(self) -> None:
        if self.semantics not in {"full", "exists", "anti"}:
            raise CompileError(
                f"unknown join semantics {self.semantics!r}")
        validate_predicate(self.predicate)
        present = {field.alias for field in self.input.output_schema()}
        missing = set(model_call(self.predicate).aliases()) - present
        if missing:
            raise CompileError(
                f"the join predicate reads {sorted(missing)}, which its "
                f"input does not produce ({sorted(present)})")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("SemanticJoin needs one input")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        if len(expressions) != 1:
            raise CompileError("SemanticJoin needs one predicate")
        return replace(self, predicate=expressions[0])

    def explain_fields(self) -> dict:
        return {
            "semantics": self.semantics,
            "selectivity": self.selectivity,
            "anchor": self.anchor,
            "expression": _explain(self.predicate),
        }


def _explain(expression) -> str:
    if isinstance(expression, Compare):
        return (f"{_explain(expression.call)} {expression.comparison} "
                f"{expression.threshold}")
    if isinstance(expression, Alias):
        return f"{_explain(expression.expression)} AS {expression.name}"
    if expression.kind == "extract":
        document, = expression.prompt.args
        fields = ", ".join(repr(field) for field in expression.fields)
        return f"AI.EXTRACT({document.alias}.{document.column}, {fields})"
    function = "AI.SCORE" if expression.kind == "score" else "AI.IF"
    return f"{function}({expression.prompt.template!r})"


APPLY_KINDS = ("per_batch", "barrier")
APPLY_IDS = ("preserve", "drop", "pairs")


@dataclass(frozen=True)
class Apply:
    """A user function between two operators.

    The function is registered on the session under ``function`` and
    receives an Arrow table per alias: the alias's row indices under
    the alias name plus the listed columns. ``kind`` says how it runs:
    ``per_batch`` on each batch of survivors a streaming operator hands
    over, ``barrier`` once over every survivor. ``ids`` says what it
    returns: ``preserve`` every input id, ``drop`` a subset of them,
    ``pairs`` a relation of (left alias, right alias) rows for the join
    at ``written_pos``. A function never invents an id.
    """
    input: LogicalNode
    function: str
    kind: str
    ids: str
    columns: tuple            # tuple[ColumnRef, ...]
    aliases: tuple            # the alias, or the join's two aliases
    written_pos: Optional[int] = None

    type_name: ClassVar[str] = "quail.apply"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return self.columns

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return self.input.output_schema()

    def validate(self) -> None:
        if self.kind not in APPLY_KINDS:
            raise CompileError(
                f"apply kind must be one of {APPLY_KINDS}, got {self.kind!r}")
        if self.ids not in APPLY_IDS:
            raise CompileError(
                f"apply ids must be one of {APPLY_IDS}, got {self.ids!r}")
        if not self.function:
            raise CompileError("apply needs a registered function name")
        present = {field.alias for field in self.input.output_schema()}
        if not set(self.aliases) <= present:
            raise CompileError(
                f"apply {self.function!r} names tables {self.aliases} "
                f"outside its input ({sorted(present)})")
        for ref in self.columns:
            if ref.alias not in self.aliases:
                raise CompileError(
                    f"apply {self.function!r} reads {ref.alias}.{ref.column} "
                    f"but works on {self.aliases}")
        if (self.ids == "pairs") != (len(self.aliases) == 2):
            raise CompileError(
                "an apply returning pairs works on exactly two tables; "
                "one returning ids works on one")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("Apply needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        return replace(self, columns=tuple(expressions))

    def explain_fields(self) -> dict:
        return {
            "function": self.function,
            "kind": self.kind,
            "ids": self.ids,
            "columns": [f"{ref.alias}.{ref.column}" for ref in self.columns],
        }


@dataclass(frozen=True)
class Project:
    """Column projection. Always the root operator."""
    input: LogicalNode
    columns: tuple    # tuple[ColumnRef | Alias, ...]
    limit: Optional[int] = None

    type_name: ClassVar[str] = "quail.logical_project"

    def children(self) -> tuple[LogicalNode, ...]:
        return (self.input,)

    def expressions(self) -> tuple:
        return self.columns

    def output_schema(self) -> tuple[ColumnRef, ...]:
        return tuple(
            column for column in self.columns
            if isinstance(column, ColumnRef)
        )

    def validate(self) -> None:
        if not self.columns:
            raise CompileError("Project needs at least one column")
        for column in self.columns:
            if isinstance(column, Alias):
                column.validate()
            elif not isinstance(column, ColumnRef):
                raise CompileError(
                    f"a projected column is a column reference or a named "
                    f"expression, got {type(column).__name__}")
        names = [
            f"{column.alias}.{column.column}"
            if isinstance(column, ColumnRef) else column.name
            for column in self.columns
        ]
        if len(names) != len(set(names)):
            raise CompileError(f"projection names must be unique, got {names}")
        if self.limit is not None and self.limit <= 0:
            raise CompileError("LIMIT must be a positive integer")

    def with_children(self, children: tuple[LogicalNode, ...]):
        if len(children) != 1:
            raise CompileError("Project needs one child")
        return replace(self, input=children[0])

    def with_expressions(self, expressions: tuple):
        if not expressions:
            raise CompileError("Project needs at least one column")
        return replace(self, columns=expressions)

    def explain_fields(self) -> dict:
        return {
            "columns": [
                (f"{column.alias}.{column.column}"
                 if isinstance(column, ColumnRef) else _explain(column))
                for column in self.columns
            ],
            "limit": self.limit,
        }


@dataclass(frozen=True)
class LogicalPlan:
    root: LogicalNode

    def output_schema(self) -> tuple[ColumnRef, ...]:
        """Return the columns produced by the root node."""
        return self.root.output_schema()

    def walk(self) -> tuple[LogicalNode, ...]:
        """Return logical nodes with each child before its parent."""
        nodes = []

        def visit(node):
            for child in node.children():
                visit(child)
            nodes.append(node)

        visit(self.root)
        return tuple(nodes)

    def validate(self) -> None:
        """Validate every logical node."""
        for node in self.walk():
            node.validate()

    def operators(self) -> "Operators":
        """Return the plan's operators as the planner reads them."""
        scans, filters, joins, applies = [], {}, [], []
        for node in self.walk():
            if isinstance(node, Scan):
                scans.append(node)
            elif isinstance(node, SemanticFilter):
                for predicate in node.predicates:
                    (alias,) = model_call(predicate.expression).aliases()
                    filters.setdefault(alias, []).append(predicate)
            elif isinstance(node, SemanticJoin):
                joins.append(node)
            elif isinstance(node, Apply):
                applies.append(node)
        return Operators(
            scans=tuple(scans),
            filters={alias: tuple(value) for alias, value in filters.items()},
            joins=tuple(joins),
            applies=tuple(applies),
        )


@dataclass(frozen=True)
class Operators:
    """The operators of one plan, in the order the plan was written.

    ``walk`` lists children before parents, so scans, joins, and
    applies follow written order and each alias's filters keep the
    order of their predicates.
    """
    scans: tuple           # tuple[Scan, ...]
    filters: dict          # alias -> tuple[FilterPredicate, ...]
    joins: tuple           # tuple[SemanticJoin, ...]
    applies: tuple         # tuple[Apply, ...]

    @property
    def prompts(self) -> tuple:
        """Every prompt the plan asks, filters first, then joins."""
        return tuple(
            predicate.prompt
            for predicates in self.filters.values()
            for predicate in predicates
        ) + tuple(join.prompt for join in self.joins)


def join_outer_input(join: "SemanticJoin") -> "LogicalNode":
    """The tree the join extends: everything before its new tables."""
    node = join.input
    while isinstance(node, (Join, Apply)):
        node = node.left if isinstance(node, Join) else node.input
    return node


def join_conditions(join: "SemanticJoin") -> tuple:
    """The Equality conditions under one SemanticJoin, in written order."""
    conditions = []

    def visit(node):
        if isinstance(node, Apply):
            visit(node.input)
        elif isinstance(node, Join):
            visit(node.left)
            conditions.extend(node.on)

    visit(join.input)
    return tuple(conditions)


def oriented_join_conditions(join: "SemanticJoin") -> tuple | None:
    """Return one alias order and every equality oriented to that order."""
    conditions = join_conditions(join)
    if not conditions:
        return None
    left_alias, right_alias = conditions[0].aliases()
    oriented = []
    for condition in conditions:
        left, right = condition.left, condition.right
        if left.alias != left_alias:
            left, right = right, left
        oriented.append((left, right))
    return left_alias, right_alias, tuple(oriented)


def join_applies(join: "SemanticJoin") -> tuple:
    """The Apply nodes that return pairs for one SemanticJoin."""
    applies = []
    node = join.input
    while isinstance(node, Apply):
        applies.append(node)
        node = node.input
    return tuple(reversed(applies))


@dataclass(frozen=True)
class JoinSpec:
    """The join predicate (or one EXISTS/anti term), pre-assembly."""
    aliases: tuple             # newly joined table aliases
    predicate: Any             # ModelCall | Compare
    semantics: str = "full"
    selectivity: Optional[float] = None
    anchor: Optional[str] = None
    on: tuple = ()             # tuple[Equality, ...] over the tables
    applies: tuple = ()        # (function, kind, columns) returning pairs

    @property
    def prompt(self) -> Prompt:
        """The prompt asked of every tuple."""
        return model_call(self.predicate).prompt


class LogicalPlanBuilder:
    """Build the built in logical nodes used by both front ends."""

    def __init__(self):
        self._tables = []
        self._nodes = {}
        self._root = None
        self._joins = 0

    def add_scan(
        self,
        alias: str,
        provider: str,
        column: str,
        predicates: tuple[FilterPredicate, ...] = (),
        applies: tuple = (),
    ) -> None:
        """Add one table; applies are (function, kind, ids, columns)."""
        if alias in self._nodes:
            raise CompileError(f"duplicate table alias {alias!r}")
        node = Scan(provider=provider, alias=alias, column=column)
        if predicates:
            node = SemanticFilter(node, tuple(predicates))
        for function, kind, ids, columns in applies:
            node = Apply(node, function=function, kind=kind, ids=ids,
                         columns=tuple(columns), aliases=(alias,))
        self._tables.append(alias)
        self._nodes[alias] = node
        if self._root is None:
            self._root = node

    def add_join(self, join: JoinSpec) -> None:
        """Join each new table onto the tree, then ask the prompt.

        The equalities go on the Join that brings the last table in;
        both front ends only produce conditions over that table.
        """
        if self._root is None:
            raise CompileError("a logical join needs an input table")
        root = self._root
        for alias in join.aliases:
            root = Join(root, self._nodes[alias])
        present = {field.alias for field in root.output_schema()}
        for condition in join.on:
            if not set(condition.aliases()) <= present:
                raise CompileError(
                    f"join condition {condition} names a table this join "
                    f"does not bring in ({list(join.aliases)}); put it on "
                    f"the JOIN that introduces the table")
        if join.on:
            root = replace(root, on=tuple(join.on))
        written_pos = self._joins
        self._joins += 1
        for function, kind, columns in join.applies:
            aliases = tuple(dict.fromkeys(ref.alias for ref in join.prompt.args))
            root = Apply(root, function=function, kind=kind, ids="pairs",
                         columns=tuple(columns), aliases=aliases,
                         written_pos=written_pos)
        self._root = SemanticJoin(
            input=root,
            predicate=join.predicate,
            semantics=join.semantics,
            selectivity=join.selectivity,
            anchor=join.anchor,
        )

    def add_cross_join(self, alias: str) -> None:
        """Add one relational cross join without an AI predicate."""
        if self._root is None:
            raise CompileError("a logical join needs an input table")
        self._root = Join(self._root, self._nodes[alias])

    def project(
        self, columns: tuple, limit: int | None = None
    ) -> LogicalPlan:
        if self._root is None:
            raise CompileError("a logical plan needs an input table")
        plan = LogicalPlan(Project(self._root, tuple(columns), limit))
        plan.validate()
        return plan
