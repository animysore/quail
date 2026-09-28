"""Generic execution for typed physical graphs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Mapping, Protocol

from quail.execution.pairs import columns_key, pair_ids_table, pair_table
from quail.execution.result import (
    IndexRelation,
    QueryResult,
    build_result_declaration,
    true_answer_rows,
)
from quail.physical import (
    Barrier,
    Exchange,
    ExecutionLocation,
    Foreign,
    GraphValidationError,
    HashJoin,
    Limit,
    PhysicalGraph,
    PhysicalNode,
    PortRef,
    Project,
    Recombine,
    Scan,
    ValueType,
)


@dataclass(frozen=True)
class NodeMetrics:
    """Metrics reported by one physical node."""

    wall_s: float = 0.0
    input_rows: int = 0
    output_rows: int = 0
    evaluated_documents: int = 0
    evaluated_document_pairs: int = 0
    fresh_tokens: int = 0
    cached_tokens: int = 0
    kv_hits: int = 0
    kv_misses: int = 0
    kv_removals: int = 0
    kv_recomputations: int = 0
    peak_gpu_bytes: int = 0
    gpu_s: float = 0.0
    chunks: int = 0
    extension: Mapping[str, Any] = field(default_factory=dict)

    def __add__(self, other: "NodeMetrics") -> "NodeMetrics":
        return NodeMetrics(
            wall_s=self.wall_s + other.wall_s,
            input_rows=self.input_rows + other.input_rows,
            output_rows=self.output_rows + other.output_rows,
            evaluated_documents=(
                self.evaluated_documents + other.evaluated_documents
            ),
            evaluated_document_pairs=(
                self.evaluated_document_pairs
                + other.evaluated_document_pairs
            ),
            fresh_tokens=self.fresh_tokens + other.fresh_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            kv_hits=self.kv_hits + other.kv_hits,
            kv_misses=self.kv_misses + other.kv_misses,
            kv_removals=self.kv_removals + other.kv_removals,
            kv_recomputations=(
                self.kv_recomputations + other.kv_recomputations
            ),
            peak_gpu_bytes=max(self.peak_gpu_bytes, other.peak_gpu_bytes),
            gpu_s=self.gpu_s + other.gpu_s,
            chunks=self.chunks + other.chunks,
            extension={**self.extension, **other.extension},
        )


def scalar_node_metrics(nodes: Mapping[str, "NodeResult"]) -> dict:
    """Return the scalar metrics of each node, keyed by node id.

    Extension metrics that are plain numbers, such as a filter's
    borrowed_prefix_tokens, are included beside the standard fields.
    """
    scalar_fields = tuple(
        field.name for field in fields(NodeMetrics)
        if field.name != "extension"
    )
    return {
        node_id: {
            **{name: getattr(result.metrics, name) for name in scalar_fields},
            **{name: value
               for name, value in result.metrics.extension.items()
               if isinstance(value, (int, float)) and not isinstance(value, bool)},
        }
        for node_id, result in nodes.items()
    }


@dataclass(frozen=True)
class NodeResult:
    """Outputs and metrics produced by one physical node.

    A node that streams its output returns a provisional result and a
    finalize callable; the runner calls it after the whole graph has
    run and stores what it returns as the node's result.
    """

    outputs: Mapping[str, Any]
    metrics: NodeMetrics = NodeMetrics()
    finalize: Callable[[], "NodeResult"] | None = None


@dataclass
class _SurvivorCompletion:
    result: NodeResult | None = None


@dataclass
class SurvivorStream:
    """Survivor ids a GPU operator hands its consumer as it produces them.

    The consumer drives the producer's chain and completes the stream
    with the producer's final result.
    transforms are per-batch functions (ids -> kept ids) that
    per-batch Foreign nodes between the producer and the consumer
    added; the consumer runs them on each batch before admission.
    """

    node: PhysicalNode
    document_ids: Any
    completion: _SurvivorCompletion = field(
        default_factory=_SurvivorCompletion, repr=False)
    transforms: tuple = ()

    def with_transform(self, transform) -> "SurvivorStream":
        """Return this stream with one more per-batch transform."""
        return SurvivorStream(
            self.node,
            self.document_ids,
            completion=self.completion,
            transforms=self.transforms + (transform,),
        )

    def complete(self, result: NodeResult) -> None:
        """Store the producer result after the consumer finishes."""
        if self.completion.result is not None:
            raise RuntimeError("survivor stream was completed twice")
        self.completion.result = result

    def finalized_result(self) -> NodeResult:
        """Return the completed producer result."""
        if self.completion.result is None:
            raise RuntimeError("survivor stream was not completed")
        return self.completion.result


@dataclass
class StreamedPairs:
    """Pairs a per-batch Foreign node returns for a survivor stream.

    The consuming join calls batch(ids) on each batch the stream
    hands over, after the stream's own transforms, and reads the
    anchor -> partner rows it fills into rows.
    """

    batch: Callable[[list], dict]
    rows: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    """Root value and per node results from one graph run."""

    value: Any
    nodes: Mapping[str, NodeResult]
    metrics: NodeMetrics


class NodeRuntime(Protocol):
    """Runtime implementation for one physical node type."""

    def execute(
        self,
        node: PhysicalNode,
        inputs: Mapping[str, Any],
        context: "ExecutionContext",
    ) -> NodeResult: ...


class ExecutionObserver(Protocol):
    """Observe node results without changing plan values."""

    name: str

    def after_node(self, node: PhysicalNode, result: NodeResult) -> None: ...

    def report(self) -> Mapping[str, Any]: ...


@dataclass
class ExecutionContext:
    """State shared while one physical graph runs."""

    runtimes: Mapping[str, NodeRuntime]
    model_execution: Any = None
    sources: Mapping[str, Any] = field(default_factory=dict)
    project: Callable[[Project, Any], Any] | None = None
    recombine: Callable[[Recombine, Mapping[str, Any]], Any] | None = None
    model_inputs: Callable[
        [PhysicalNode, Mapping[str, Any], "ExecutionContext"],
        Mapping[str, Any],
    ] | None = None
    model_result: Callable[
        [PhysicalNode, NodeResult, "ExecutionContext"], None,
    ] | None = None
    state: dict[str, Any] = field(default_factory=dict)
    observers: tuple[ExecutionObserver, ...] = ()
    functions: Mapping[str, Callable[..., Any]] = field(default_factory=dict)

    def execute_graph(self, graph: PhysicalGraph) -> RunResult:
        """Execute a child graph with the same model state."""
        return GenericRunner().run(graph, self)


def _check_ports(node, result) -> None:
    expected = {output.name for output in node.outputs}
    missing = expected - set(result.outputs)
    extra = set(result.outputs) - expected
    if missing or extra:
        raise ValueError(
            f"runtime {node.runtime_key!r} returned wrong ports "
            f"for {node.node_id!r}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}")


class GenericRunner:
    """Execute nodes after all their inputs are available.

    A node may return a SurvivorStream on an output port; its consumer
    drives it. Such a node's result is provisional until the whole
    graph has run, when the runner calls its finalize.
    """

    def run(
        self,
        graph: PhysicalGraph,
        context: ExecutionContext,
        initial_outputs: Mapping[PortRef, Any] | None = None,
        initial_metrics: Mapping[str, NodeMetrics] | None = None,
    ) -> RunResult:
        graph.validate(runtime_keys=set(context.runtimes))
        values: dict[tuple[str, str], Any] = {
            (ref.node_id, ref.port): value
            for ref, value in (initial_outputs or {}).items()
        }
        node_results: dict[str, NodeResult] = {}

        for node in graph.topological_nodes():
            supplied = {
                output.name: values[(node.node_id, output.name)]
                for output in node.outputs
                if (node.node_id, output.name) in values
            }
            if supplied:
                expected = {output.name for output in node.outputs}
                if set(supplied) != expected:
                    raise ValueError(
                        f"initial outputs for {node.node_id!r} are partial"
                    )
                result = NodeResult(
                    supplied,
                    (initial_metrics or {}).get(node.node_id, NodeMetrics()),
                )
            else:
                inputs = {
                    input_port.name: values[
                        (input_port.source.node_id, input_port.source.port)
                    ]
                    for input_port in node.inputs
                }
                started = time.perf_counter()
                result = context.runtimes[node.runtime_key].execute(
                    node, inputs, context
                )
                if result.metrics.wall_s == 0.0 and result.finalize is None:
                    # runtimes that do not time themselves are timed here
                    result = replace(result, metrics=replace(
                        result.metrics,
                        wall_s=time.perf_counter() - started,
                    ))
            _check_ports(node, result)
            for port, value in result.outputs.items():
                values[(node.node_id, port)] = value
            node_results[node.node_id] = result
            if result.finalize is None:
                for observer in context.observers:
                    observer.after_node(node, result)

        root = (graph.root.node_id, graph.root.port)
        metrics = NodeMetrics()
        for node in graph.topological_nodes():
            result = node_results[node.node_id]
            if result.finalize is not None:
                result = result.finalize()
                _check_ports(node, result)
                for port, value in result.outputs.items():
                    values[(node.node_id, port)] = value
                node_results[node.node_id] = result
                for observer in context.observers:
                    observer.after_node(node, result)
            metrics = metrics + result.metrics
        return RunResult(values[root], node_results, metrics)


def compute_subgraph(graph: PhysicalGraph) -> PhysicalGraph:
    """Return model nodes and the input nodes they read."""
    by_id = {node.node_id: node for node in graph.nodes}
    selected = {
        node.node_id for node in graph.nodes
        if node.location is ExecutionLocation.GPU_EXECUTOR
        or node.backend is not None
    }

    def include_inputs(node_id: str) -> None:
        for input_port in by_id[node_id].inputs:
            source_id = input_port.source.node_id
            if source_id not in selected:
                selected.add(source_id)
                include_inputs(source_id)

    for node_id in tuple(selected):
        include_inputs(node_id)
    nodes = tuple(
        node for node in graph.topological_nodes()
        if node.node_id in selected
    )
    if not nodes:
        raise ValueError("physical graph has no compute operation")
    root_node = nodes[-1]
    return PhysicalGraph(
        nodes,
        PortRef(root_node.node_id, root_node.outputs[0].name),
    )


class ScanRuntime:
    """Read a prepared source registered by document alias."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Scan):
            raise TypeError(type(node).__name__)
        if node.input_id not in context.sources:
            raise KeyError(
                f"no prepared source for input {node.input_id!r}"
            )
        value = context.sources[node.input_id]
        return NodeResult({f"ids:{node.alias}": value},
                          NodeMetrics(output_rows=_row_count(value) or 0))


class ExchangeRuntime:
    """Pass anchor ids through; several GPUs route them to their KV."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Exchange):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("Exchange takes one survivor input")
        return NodeResult({f"ids:{node.anchor}": next(iter(inputs.values()))})


class BarrierRuntime:
    """Prune survivor IDs using completed join answers."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Barrier):
            raise TypeError(type(node).__name__)
        import pyarrow as pa
        import pyarrow.compute as pc

        from quail.execution.types import _join_answers_table

        survivors = {}
        tables = []
        arrow_inputs = set()
        for port in node.inputs:
            value = inputs[port.name]
            if port.value_type is ValueType.DOCUMENT_IDS:
                alias = port.source.port.split(":", 1)[1]
                if isinstance(value, pa.Table):
                    arrow_inputs.add(alias)
                    value = value.column(alias).to_pylist()
                survivors[alias] = list(value)
            elif port.value_type is ValueType.JOIN_ANSWERS:
                tables.append(value if isinstance(value, pa.Table)
                              else _join_answers_table(value))
        for table in tables:
            table = true_answer_rows(table)
            for alias in table.column_names:
                if alias in survivors:
                    table = table.filter(pc.is_in(
                        table[alias], value_set=pa.array(
                            survivors[alias], type=table.schema.field(alias).type
                        )
                    ))
            for alias in table.column_names:
                if alias in survivors:
                    live = set(pc.unique(table[alias]).to_pylist())
                    survivors[alias] = [document for document in survivors[alias]
                                        if document in live]
        return NodeResult({
            f"ids:{alias}": (
                pa.table({alias: pa.array(survivors[alias], type=pa.int32())})
                if alias in arrow_inputs else survivors[alias]
            )
            for alias in node.aliases
        })


def _row_count(value) -> int | None:
    """Rows of a table or sequence, or None when counting would run a plan."""
    if isinstance(value, QueryResult):
        return value.known_count()
    if hasattr(value, "num_rows"):
        return int(value.num_rows)
    if hasattr(value, "__len__"):
        return len(value)
    return None


def _ids_of(value) -> list:
    """Ids from a list, an Arrow array, or a one-column Arrow table."""
    import pyarrow as pa

    if isinstance(value, pa.Table):
        if value.num_columns != 1:
            raise ValueError("an apply() returning ids must return one "
                             "column, an array, or a list of ids")
        value = value.column(0)
    if isinstance(value, (pa.Array, pa.ChunkedArray)):
        value = value.to_pylist()
    ids = []
    for document in value:
        if document is None:
            raise ValueError(
                "an apply() function returned a null id; an outer join "
                "leaves nulls, use join_type='inner'")
        ids.append(int(document))
    return ids


def _pairs_of(value, left: str, right: str) -> list:
    """(left, right) id pairs from an Arrow table with both alias columns."""
    import pyarrow as pa

    if not isinstance(value, pa.Table) or not {left, right} <= set(
            value.column_names):
        raise ValueError(
            f"an apply() returning pairs must return an Arrow table with "
            f"columns {left!r} and {right!r}")
    return list(zip(_ids_of(value.column(left)), _ids_of(value.column(right))))


def alias_table(alias: str, ids, columns, names):
    """The Arrow table an apply() function sees for one alias.

    Its first column is the alias's row indices under the alias
    name; the rest are the requested value columns for those rows.
    """
    import pyarrow as pa

    ids = [int(document) for document in ids]
    arrays = {alias: pa.array(ids, type=pa.int32())}
    for name in names:
        if columns is None or name not in columns.column_names:
            raise KeyError(
                f"column {alias}.{name} is not among the request's relations")
        arrays[name] = columns.column(name).take(pa.array(ids, pa.int64()))
    return pa.table(arrays)


class ForeignRuntime:
    """Call a user function on ids and values, once or per batch."""

    def execute(self, node, inputs, context) -> NodeResult:
        import pyarrow as pa

        if not isinstance(node, Foreign):
            raise TypeError(type(node).__name__)
        function = context.functions.get(node.function)
        if function is None:
            raise KeyError(
                f"apply() function {node.function!r} is not registered on "
                f"this session")
        values = {}
        for port in node.inputs:
            if port.source.port.startswith("ids:"):
                values[port.source.port.split(":", 1)[1]] = inputs[port.name]
        missing = [alias for alias in node.aliases if alias not in values]
        if missing:
            raise GraphValidationError(
                f"{node.node_id!r} has no id input for {missing}")
        columns = {alias: context.sources.get(columns_key(alias))
                   for alias in node.aliases}
        names = {alias: [column for owner, column in node.columns
                         if owner == alias] for alias in node.aliases}
        counters = dict(calls=0, input_rows=0, output_rows=0)

        def call(ids_by_alias):
            tables = {alias: alias_table(alias, ids, columns[alias],
                                         names[alias])
                      for alias, ids in ids_by_alias.items()}
            counters["calls"] += 1
            counters["input_rows"] += sum(
                len(ids) for ids in ids_by_alias.values())
            result = function(tables)
            if node.ids == "pairs":
                left, right = node.aliases
                pairs = _pairs_of(result, left, right)
                allowed = {alias: set(ids) for alias, ids in ids_by_alias.items()}
                for a, b in pairs:
                    if a not in allowed[left] or b not in allowed[right]:
                        raise ValueError(
                            f"apply() {node.function!r} returned pair "
                            f"({a}, {b}) outside its input ids")
                counters["output_rows"] += len(pairs)
                return pairs
            (alias,) = node.aliases
            kept = _ids_of(result)
            given = set(ids_by_alias[alias])
            if not set(kept) <= given:
                raise ValueError(
                    f"apply() {node.function!r} returned ids it was not "
                    f"given; a function never invents an id")
            if node.ids == "preserve" and set(kept) != given:
                raise ValueError(
                    f"apply() {node.function!r} preserves ids but dropped "
                    f"{len(given) - len(set(kept))}")
            counters["output_rows"] += len(kept)
            return kept

        def metrics():
            return NodeMetrics(
                input_rows=counters["input_rows"],
                output_rows=counters["output_rows"],
                extension={"calls": counters["calls"]})

        streams = {alias: value for alias, value in values.items()
                   if isinstance(value, SurvivorStream)}
        if len(streams) > 1:
            raise GraphValidationError(
                f"{node.node_id!r} reads two survivor streams")

        if not streams:
            ids_by_alias = {}
            for alias in node.aliases:
                value = values[alias]
                if isinstance(value, pa.Table):
                    value = value.column(alias).to_pylist()
                ids_by_alias[alias] = list(value)
            result = call(ids_by_alias)
            if node.ids == "pairs":
                outputs = {
                    f"pairs:{node.written_pos}":
                        pair_ids_table(*node.aliases, result)
                }
            else:
                outputs = {f"ids:{node.aliases[0]}": result}
            return NodeResult(outputs, metrics())

        # per batch: the consuming join runs it on each batch
        (stream_alias, stream), = streams.items()
        produced = []
        if node.ids == "pairs":
            partner = next(alias for alias in node.aliases
                           if alias != stream_alias)
            partner_ids = list(values[partner])

            def batch(ids):
                pairs = call({stream_alias: ids, partner: partner_ids})
                produced.extend(pairs)
                rows = {}
                for a, b in pairs:
                    anchor, other = (a, b) if stream_alias == node.aliases[0] \
                        else (b, a)
                    rows.setdefault(anchor, []).append(other)
                return rows

            port = f"pairs:{node.written_pos}"
            outputs = {port: StreamedPairs(batch)}

            def finalize():
                ordered = produced if stream_alias == node.aliases[0] \
                    else [(b, a) for a, b in produced]
                return NodeResult(
                    {port: pair_ids_table(*node.aliases, ordered)}, metrics())
        else:
            def batch(ids):
                kept = call({stream_alias: ids})
                produced.extend(kept)
                return kept

            port = f"ids:{stream_alias}"
            outputs = {port: stream.with_transform(batch)}

            def finalize():
                return NodeResult({port: sorted(produced)}, metrics())

        return NodeResult(outputs, finalize=finalize)


class RecombineRuntime:
    """Run the configured exact answer relation join."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Recombine):
            raise TypeError(type(node).__name__)
        if context.recombine is None:
            import pyarrow as pa


            answer_tables = []
            survivors = {}
            for input_port in node.inputs:
                value = inputs[input_port.name]
                if not isinstance(value, pa.Table):
                    raise TypeError(
                        "Recombine inputs must be Arrow tables"
                    )
                if input_port.value_type is ValueType.JOIN_ANSWERS:
                    answer_tables.append(true_answer_rows(value))
                elif input_port.value_type is ValueType.DOCUMENT_IDS:
                    if len(value.column_names) != 1:
                        raise ValueError(
                            "a survivor relation needs one alias column"
                        )
                    alias = value.column_names[0]
                    survivors[alias] = value.column(alias).combine_chunks()
                else:
                    raise TypeError(
                        f"Recombine cannot read {input_port.value_type.value}"
                    )
            declaration, schema = build_result_declaration(
                answer_tables,
                survivors,
                node.alias_order[0],
            )
            value = IndexRelation(declaration, schema)
        else:
            value = context.recombine(node, inputs)
        return NodeResult({"tuples": value})


class ProjectRuntime:
    """Run the configured result projection."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Project):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("Project needs one input")
        value = next(iter(inputs.values()))
        if context.project is not None:
            value = context.project(node, value)
        rows = _row_count(value) or 0
        return NodeResult({"rows": value},
                          NodeMetrics(input_rows=rows, output_rows=rows))


class LimitRuntime:
    """Limit a table, batch, or Python sequence."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Limit):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("Limit needs one input")
        value = next(iter(inputs.values()))

        rows = _row_count(value) or 0
        if isinstance(value, QueryResult):
            value = value.with_limit(node.count)
        elif hasattr(value, "slice"):
            value = value.slice(0, node.count)
        else:
            value = value[:node.count]
        return NodeResult({"rows": value}, NodeMetrics(
            input_rows=rows, output_rows=_row_count(value) or 0))


class ModelNodeRuntime:
    """Delegate one model node to the shared model execution object."""

    def execute(self, node, inputs, context) -> NodeResult:
        if context.model_execution is None:
            raise RuntimeError("model node has no model execution object")
        if context.model_inputs is not None:
            inputs = context.model_inputs(node, inputs, context)
        result = context.model_execution.execute(node, inputs)
        if not isinstance(result, NodeResult):
            raise TypeError("model execution must return NodeResult")
        if context.model_result is not None:
            context.model_result(node, result, context)
        return result


class HashJoinRuntime:
    """Pair the rows of two tables whose key columns are equal."""

    def execute(self, node, inputs, context) -> NodeResult:
        import pyarrow as pa
        import pyarrow.compute as pc

        if not isinstance(node, HashJoin):
            raise TypeError(type(node).__name__)
        sides = {}
        for alias, names in ((node.left, [left for left, _ in node.on]),
                             (node.right, [right for _, right in node.on])):
            port = next(port for port in node.inputs
                        if port.source.port == f"ids:{alias}")
            table = alias_table(alias, inputs[port.name],
                                context.sources.get(columns_key(alias)), names)
            sides[alias] = (table.column(alias),
                            [table.column(name) for name in names])
        positions = pair_table(node.left, sides[node.left][1],
                               node.right, sides[node.right][1])
        # pair_table pairs row positions; map them back to the ids read
        pairs = pa.table({
            alias: pc.take(sides[alias][0], positions.column(alias))
            for alias in (node.left, node.right)})
        return NodeResult(
            {f"pairs:{node.written_pos}": pairs},
            NodeMetrics(
                input_rows=sum(len(ids) for ids, _ in sides.values()),
                output_rows=pairs.num_rows))


def built_in_runtimes() -> dict[str, NodeRuntime]:
    """Return runtimes for the backend independent physical nodes."""
    return {
        Scan.runtime_key: ScanRuntime(),
        Barrier.runtime_key: BarrierRuntime(),
        Exchange.runtime_key: ExchangeRuntime(),
        Foreign.runtime_key: ForeignRuntime(),
        HashJoin.runtime_key: HashJoinRuntime(),
        Recombine.runtime_key: RecombineRuntime(),
        Project.runtime_key: ProjectRuntime(),
        Limit.runtime_key: LimitRuntime(),
    }
