"""Read a QUAIL-B Substrait plan and build it as a Quail query.

A QUAIL-B query is a Substrait plan (a protocol buffer that describes
a relational query): `ReadRel` scans, `FilterRel` calls to
`ai_filter`, inner `JoinRel` calls to `ai_join` (joined with ordinary
`equal` conditions by `and`), and a `ProjectRel` under the root that
selects IDs or extraction results. Inner projects call `ai_extract`.
The alias of a relation and
the id of an operator are the `RelCommon.hint.alias` of its node.
Operator ids number filters and joins in post-order, inputs before the
operator and left before right.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlglot import exp
from substrait import algebra_pb2, plan_pb2

import quail

AI_URN = "extension:org.fsdatalab.quail_b:functions_ai"
AI_FILTER = "ai_filter:str_str"
AI_JOIN = "ai_join:str_str_str"
AI_EXTRACT = "ai_extract:str_list_any"
AI_CLASSIFY = "ai_classify:str_str_list_list"
EQUAL = "equal:any_any"
AND = "and:bool"


class UnsupportedQueryError(ValueError):
    """A valid benchmark query uses an operator Quail does not implement."""


@dataclass(frozen=True)
class Relation:
    """One scanned table and the alias the query knows it by."""

    alias: str
    table: str


@dataclass(frozen=True)
class Filter:
    """One `ai_filter` over the documents of one relation."""

    id: str
    alias: str
    column: str
    prompt: str


@dataclass(frozen=True)
class Join:
    """One `ai_join` over the pairs of two relations.

    Attributes:
        id: The operator id.
        aliases: The two relations, in prompt placeholder order.
        columns: The document column of each relation.
        prompt: The join prompt.
        on: (left column, right column) pairs that must be equal for
            a pair to be asked about; empty means every pair.
    """

    id: str
    aliases: tuple[str, str]
    columns: tuple[str, str]
    prompt: str
    on: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Extract:
    """One extraction projection over a document column."""

    id: str
    alias: str
    column: str
    fields: tuple[str, ...]
    output: str


@dataclass(frozen=True)
class QueryPlan:
    """A QUAIL-B query as its relations, operators, and projection.

    Attributes:
        relations: The scanned relations, in scan order.
        operators: The filters, joins and extracts, in operator id order.
        select: The projected columns, as "alias.column".
    """

    relations: tuple[Relation, ...]
    operators: tuple[Filter | Join | Extract, ...]
    select: tuple[str, ...]

    @property
    def filters(self) -> tuple[Filter, ...]:
        return tuple(op for op in self.operators if isinstance(op, Filter))

    @property
    def joins(self) -> tuple[Join, ...]:
        return tuple(op for op in self.operators if isinstance(op, Join))

    @property
    def extracts(self) -> tuple[Extract, ...]:
        return tuple(op for op in self.operators if isinstance(op, Extract))

    def filter_id(self, alias: str, position: int) -> str:
        """Return the id of an alias's filter at a written position."""
        return [f.id for f in self.filters if f.alias == alias][position]

    def join_id(self, position: int) -> str:
        """Return the id of the join at a written position."""
        return self.joins[position].id


def _functions(plan: plan_pb2.Plan) -> dict[int, str]:
    """Map each function anchor to its name, checking the AI functions' URN."""
    urns = {item.extension_urn_anchor: item.urn for item in plan.extension_urns}
    names = {}
    for declaration in plan.extensions:
        if not declaration.HasField("extension_function"):
            continue
        function = declaration.extension_function
        if function.name in (AI_FILTER, AI_JOIN, AI_EXTRACT, AI_CLASSIFY) and (
                urns.get(function.extension_urn_reference) != AI_URN):
            raise ValueError(f"{function.name} must come from {AI_URN}")
        names[function.function_anchor] = function.name
    return names


def _field(fields, expression: algebra_pb2.Expression) -> tuple[str, str]:
    """Return the (alias, column) a field reference selects."""
    if not expression.HasField("selection"):
        raise ValueError("expected a field reference")
    segment = expression.selection.direct_reference
    if not segment.HasField("struct_field"):
        raise ValueError("expected a struct field reference")
    return fields[segment.struct_field.field]


def _string(expression: algebra_pb2.Expression) -> str:
    if not expression.HasField("literal") or not expression.literal.HasField(
            "string"):
        raise ValueError("an AI prompt must be a string literal")
    return expression.literal.string


def _call(expression: algebra_pb2.Expression, functions) -> tuple[str, list]:
    """Return (function name, arguments) of a scalar function call."""
    if not expression.HasField("scalar_function"):
        raise ValueError("expected a scalar function call")
    call = expression.scalar_function
    name = functions.get(call.function_reference)
    if name is None:
        raise ValueError(f"undeclared function anchor {call.function_reference}")
    return name, [argument.value for argument in call.arguments]


def _conditions(expression, functions) -> list[tuple[str, list]]:
    """Flatten nested `and` calls into the calls they combine."""
    name, arguments = _call(expression, functions)
    if name != AND:
        return [(name, arguments)]
    return [item for argument in arguments
            for item in _conditions(argument, functions)]


def _field_names(fields, operators):
    extracts = {(op.alias, op.output): op.fields for op in operators
                if isinstance(op, Extract)}
    names = []
    for alias, name in fields:
        names.append(f"{alias}.{name}")
        if (alias, name) in extracts:
            names.extend(("response", *extracts[alias, name], "error"))
    return names


def _read(rel: algebra_pb2.Rel, functions):
    """Return (relations, operators, fields) of one relation subtree."""
    kind = rel.WhichOneof("rel_type")
    if kind == "read":
        read = rel.read
        alias = read.common.hint.alias
        if not alias or not read.named_table.names:
            raise ValueError("a scan needs a table name and an alias hint")
        fields = tuple((alias, name) for name in read.base_schema.names)
        return [Relation(alias, read.named_table.names[-1])], [], fields
    if kind == "project":
        project = rel.project
        name, arguments = _call(project.expressions[0], functions)
        if name == AI_CLASSIFY:
            raise UnsupportedQueryError("AI.CLASSIFY is not implemented in Quail")
        if name != AI_EXTRACT:
            raise ValueError("unsupported benchmark projection")
        relations, operators, fields = _read(project.input, functions)
        alias, column = _field(fields, arguments[0])
        extracted_fields = tuple(
            value.string for value in arguments[1].literal.list.values)
        offset = len(_field_names(fields, operators))
        output = project.common.hint.output_names[offset].split(".", 1)[1]
        operators.append(Extract(
            project.common.hint.alias, alias, column, extracted_fields, output))
        return relations, operators, (*fields, (alias, output))
    if kind == "filter":
        relations, operators, fields = _read(rel.filter.input, functions)
        name, arguments = _call(rel.filter.condition, functions)
        if name != AI_FILTER or len(arguments) != 2:
            raise ValueError("a filter must call ai_filter(prompt, document)")
        alias, column = _field(fields, arguments[1])
        operators.append(Filter(
            rel.filter.common.hint.alias, alias, column, _string(arguments[0])))
        return relations, operators, fields
    if kind == "join":
        join = rel.join
        if join.type != algebra_pb2.JoinRel.JOIN_TYPE_INNER:
            raise ValueError("a join must be an inner join")
        left_relations, left_operators, left_fields = _read(join.left, functions)
        right_relations, right_operators, right_fields = _read(
            join.right, functions)
        fields = left_fields + right_fields
        ai = [args for name, args in _conditions(join.expression, functions)
              if name == AI_JOIN]
        equalities = [args for name, args in _conditions(join.expression, functions)
                      if name == EQUAL]
        if len(ai) != 1 or len(ai[0]) != 3 or len(ai) + len(equalities) != len(
                _conditions(join.expression, functions)):
            raise ValueError(
                "a join must call ai_join(prompt, left, right) once, with "
                "equal conditions only beside it")
        (first_alias, first_column), (second_alias, second_column) = (
            _field(fields, ai[0][1]), _field(fields, ai[0][2]))
        on = []
        for arguments in equalities:
            (alias_a, column_a), (alias_b, column_b) = (
                _field(fields, argument) for argument in arguments)
            if (alias_a, alias_b) == (first_alias, second_alias):
                on.append((column_a, column_b))
            elif (alias_b, alias_a) == (first_alias, second_alias):
                on.append((column_b, column_a))
            else:
                raise ValueError("a join equality must relate the joined relations")
        operators = left_operators + right_operators + [Join(
            join.common.hint.alias, (first_alias, second_alias),
            (first_column, second_column), _string(ai[0][0]), tuple(on))]
        return left_relations + right_relations, operators, fields
    raise ValueError(f"unsupported relation {kind!r}")


def read_plan(plan: plan_pb2.Plan) -> QueryPlan:
    """Return the relations, operators, and projection of a QUAIL-B plan."""
    if len(plan.relations) != 1 or not plan.relations[0].HasField("root"):
        raise ValueError("a QUAIL-B plan has one root relation")
    root = plan.relations[0].root
    if not root.input.HasField("project"):
        raise ValueError("a QUAIL-B plan projects its output under the root")
    functions = _functions(plan)
    if AI_EXTRACT in functions.values():
        from quail_b.queries import QuerySpec

        # Validate dependent types and nested names before translating the call.
        QuerySpec.from_plan("quail-adapter", "", plan)
    relations, operators, fields = _read(root.input.project.input, functions)
    select = tuple(".".join(_field(fields, expression))
                   for expression in root.input.project.expressions)
    return QueryPlan(tuple(relations), tuple(operators), select)


def build_query(session, plan: QueryPlan, selectivity=None,
                order: str | None = None):
    """Build the Quail query of a plan on a session.

    Args:
        session: A session with every relation's table registered.
        plan: The plan as `read_plan` returns it.
        selectivity: Prompt -> the fraction of documents or pairs
            expected to pass, given to the planner for ordering.
        order: The filter order rule `select` takes.
    """
    selectivity = selectivity or {}
    if plan.extracts:
        return _build_extract_query(session, plan, selectivity, order)
    by_alias = {relation.alias: relation for relation in plan.relations}
    filters = {}
    for item in plan.filters:
        filters.setdefault(item.alias, []).append(item)

    def relation_query(alias):
        query = session.docs(by_alias[alias].table).alias(alias)
        for item in filters.get(alias, ()):
            query = query.ai_filter(
                quail.prompt(item.prompt, quail.col(f"{alias}.{item.column}")),
                selectivity=selectivity.get(item.prompt))
        return query

    joined = {plan.relations[0].alias}
    query = relation_query(plan.relations[0].alias)
    for join in plan.joins:
        new = [alias for alias in join.aliases if alias not in joined]
        if len(new) != 1:
            raise ValueError(f"join {join.id!r} must add one relation")
        left, right = join.aliases
        query = query.join(relation_query(new[0]), on=[
            quail.col(f"{left}.{left_column}") == quail.col(f"{right}.{right_column}")
            for left_column, right_column in join.on
        ]).ai_filter(
            quail.prompt(join.prompt, quail.col(f"{left}.{join.columns[0]}"),
                         quail.col(f"{right}.{join.columns[1]}")),
            selectivity=selectivity.get(join.prompt))
        joined.add(new[0])
    return query.select(*plan.select, order=order)


def _build_extract_query(session, plan, selectivity, order):
    """Build extraction through the supported SQL frontend."""
    relation, = plan.relations
    extracts = {op.output: op for op in plan.extracts}
    selected = []
    for name in plan.select:
        alias, column = name.split(".", 1)
        if column == "id":
            selected.append(exp.column(column, table=alias, quoted=True))
        else:
            operator = extracts[column]
            call = exp.Anonymous(this="AI_EXTRACT", expressions=[
                exp.column(operator.column, table=alias, quoted=True),
                *(exp.Literal.string(field) for field in operator.fields)])
            selected.append(exp.alias_(call, operator.output, quoted=True))
    tree = exp.select(*selected).from_(exp.Table(
        this=exp.to_identifier(relation.table, quoted=True),
        alias=exp.TableAlias(
            this=exp.to_identifier(relation.alias, quoted=True))))
    conditions = []
    for predicate in plan.filters:
        arguments = [exp.Anonymous(this="PROMPT", expressions=[
            exp.Literal.string(predicate.prompt),
            exp.column(predicate.column, table=predicate.alias, quoted=True)])]
        estimate = selectivity.get(predicate.prompt)
        if estimate is not None:
            arguments.append(exp.Struct(expressions=[exp.PropertyEQ(
                this=exp.Literal.string("selectivity"),
                expression=exp.Literal.number(estimate))]))
        conditions.append(exp.Anonymous(this="AI_FILTER", expressions=arguments))
    if conditions:
        tree = tree.where(*conditions)
    return session.sql(tree.sql(dialect="snowflake"), order=order)
