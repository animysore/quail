"""Register documents, plan queries, and run them in the current process."""

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import fields, replace
from itertools import chain
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, runtime_checkable

import pyarrow as pa
from pyarrow import compute as pc

from quail.builtins import built_in_registry
from quail.catalog import Catalog, ScanRequest, TableProvider
from quail.execution.pairs import (
    columns_key,
    pair_fraction,
    pair_table,
)
from quail.execution.result import IndexRelation, QueryResult, true_answer_rows
from quail.execution.runner import (
    ExecutionContext,
    GenericRunner,
    NodeMetrics,
    scalar_node_metrics,
)
from quail.execution.tokens import (
    ColumnStoreWriter,
    ScanInput,
    TokenStoreWriter,
)
from quail.execution.types import PhysicalRequest, document_input
from quail.extensions import ExtensionRegistry
from quail.frontend.builder import Query as BuilderQuery
from quail.frontend.sql import SQLDialect, compile_sql
from quail.logical import (
    CompileError,
    LogicalPlan,
    join_conditions,
    oriented_join_conditions,
)
from quail.physical import PortRef, Project, Scan, ValueType, encode_graph
from quail.planner import explain, plan_query, refine_plan
from quail.planner.logical_optimizer import LogicalPlanningContext, apply_logical_rules
from quail.planner.plan import EngineConfig, Refusal, resolve_model
from quail.progress import Progress, say


def _node_metrics(metrics: dict) -> NodeMetrics:
    """Rebuild a node's metrics from a report; extra numbers go to extension."""
    names = {field.name for field in fields(NodeMetrics)} - {"extension"}
    return NodeMetrics(
        **{name: value for name, value in metrics.items() if name in names},
        extension={name: value for name, value in metrics.items()
                   if name not in names})


class RefusalError(RuntimeError):
    """Raised when run() is called on a refused plan."""

    def __init__(self, refusal: Refusal):
        self.refusal = refusal
        super().__init__(
            f"{refusal.constraint}: needed {refusal.needed} "
            f"{refusal.unit}, available {refusal.available}. "
            + " ".join(refusal.reasons))


TOKENIZE_ROWS = 2048

# documents tokenized to measure tokens per byte for a length estimate
ESTIMATE_SAMPLE = 1024


@runtime_checkable
class QueryLike(Protocol):
    """What ``Session.sql`` returns, whether the query runs here or remotely.

    ``Query`` runs in this process; ``quail.server.client.RemoteQuery``
    sends the SQL to Quail Server. Both offer these methods, so code
    written against one works with the other.
    """

    def run(self, plan=None) -> QueryResult: ...

    def submit(self, **options): ...

    def execute_stream(self, batch_rows: int = 65_536,
                       limit: int | None = None) -> pa.RecordBatchReader: ...

    def collect(self, limit: int | None = None,
                batch_rows: int = 65_536) -> pa.Table: ...

    def explain(self, **options) -> str: ...

    def plan(self): ...


class Session:
    """Register tables and run queries, here or on Quail Server.

    Without ``endpoint`` every query runs in this process. With one,
    ``register`` describes each table for the server, ``sql`` returns
    a query that ``submit()`` sends there, and ``get_run`` reattaches
    to an accepted query by id. The query API is the same either way.
    """

    def __init__(self, config: EngineConfig, *,
                 tokenizer=None,
                 registry: ExtensionRegistry | None = None,
                 endpoint: str | None = None):
        self.registry = registry or built_in_registry()
        model = resolve_model(config.model, self.registry.models)
        if isinstance(model, Refusal):
            raise RefusalError(model)
        self.config = config
        self.model = model
        self.device = self.registry.device(config.device)
        try:
            backend = self.registry.backend(config.backend)
        except ValueError as error:
            raise RefusalError(Refusal(
                reasons=(str(error),),
                constraint="unknown_backend",
                needed=1,
                available=0,
                unit="backends",
            )) from error
        support = backend.supports(model, self.device, config.gpus)
        if not support.supported:
            raise RefusalError(Refusal(
                reasons=(support.reason or
                         "unsupported backend configuration",),
                constraint="unsupported_backend_configuration",
                needed=1,
                available=0,
                unit="configurations",
            ))
        self.catalog = Catalog()
        self._tok = tokenizer      # injectable for tests
        self._tokenizer_name = "configured tokenizer" if tokenizer else "Gigatoken"
        self.notes = []            # tokenizer picks etc., for reports
        self._token_stores = {}
        self._column_stores = {}
        self._store_count = 0
        self._token_directory = None
        self._corpus_tokenizers = {}
        self._length_estimates = {}
        self._lock = threading.RLock()
        self._background = None
        self._remote = None
        if endpoint is not None:
            from quail.server.client import RemoteConnection

            self._remote = RemoteConnection(endpoint, self.registry.codecs)
        elif tokenizer is None:
            _ = self.tokenizer

    @property
    def endpoint(self) -> str | None:
        """The Quail Server this session submits to, or None."""
        return None if self._remote is None else self._remote.client.endpoint

    @property
    def session_id(self) -> str | None:
        """The id sent with every submission of a remote session, or None.

        The server saves it on each record, so ``GET /v1/queries``
        with ``session_id`` lists what this session submitted.
        """
        return None if self._remote is None else self._remote.session_id

    def close(self):
        """Wait for background tokenization and remove temporary token files.

        Closing a remote session removes its staged uploads only; the
        server keeps every accepted query and the inputs it needs.
        """
        if self._remote is not None:
            self._remote.close()
        if self._background is not None:
            self._background.shutdown(cancel_futures=True)
        for store in self._token_stores.values():
            store.close()
        self._token_stores.clear()
        for store in self._column_stores.values():
            store.close()
        self._column_stores.clear()
        if self._token_directory is not None:
            self._token_directory.cleanup()
            self._token_directory = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def register(self, name: str, provider: TableProvider) -> None:
        """Register a table under ``name``.

        On a remote session the provider must be one the server can
        read: an in-memory table, a Parquet, IPC, or Arrow dataset
        (uploaded as a snapshot), or a Hugging Face dataset (pinned to a
        revision). Any other provider raises TypeError here rather than
        running locally.
        """
        self.catalog.register(name, provider)
        if self._remote is not None:
            try:
                self._remote.register(name, provider)
            except Exception:
                del self.catalog.providers[name]
                raise

    def sql(self, text: str, order: str | None = None,
            dialect: SQLDialect | str = SQLDialect.SNOWFLAKE) -> QueryLike:
        """Compile SQL locally or prepare it for remote submission.

        Returns:
            A local ``Query`` when the session has no endpoint. A remote
            session returns ``RemoteQuery`` and the server compiles the SQL.
        """
        if self._remote is not None:
            from quail.server.client import RemoteQuery

            return RemoteQuery(self, text, order=order,
                               dialect=SQLDialect(dialect).value)
        logical = compile_sql(
            text, self.catalog, self.tokenizer, dialect=dialect,
            turn=self.model.turn,
        )
        return Query(self, logical, order=order)

    def get_run(self, query_id: str):
        """Reattach to a query the server already accepted.

        Returns a QueryRun; it never resubmits or starts another
        execution.
        """
        if self._remote is None:
            raise RuntimeError(
                "get_run() needs a Session with an endpoint; a local "
                "session has no saved query records")
        return self._remote.run(query_id)

    def docs(self, name: str) -> "BoundBuilder":
        if self._remote is not None:
            raise RuntimeError(
                "the builder API is not available on a remote Session; "
                "write the query with sql()")
        return BoundBuilder(self,
                            BuilderQuery(self.catalog, name,
                                         self.tokenizer,
                                         turn=self.model.turn))

    @property
    def tokenizer(self):
        """Return Gigatoken's encoder, loading it when first used."""
        if self._tok is None:
            from gigatoken import Tokenizer

            fast = Tokenizer(self.model.hf_name)

            def encode(text):
                tokens = fast.encode(text)
                if hasattr(tokens, "tolist"):
                    return tokens.tolist()
                return [int(token) for token in tokens]

            self._tok = encode
        return self._tok

    def tokenize(self, provider_name: str, column: str,
                 projected_columns=()) -> ScanInput:
        """Tokenize one document column and keep value columns beside it.

        The token file is cached per document column. Each value column
        is cached in its own file, so a later query that returns other
        columns reads only those columns from the provider and does not
        tokenize the documents again.
        """
        with self._lock:
            return self._tokenize(provider_name, column, projected_columns)

    def _tokenize(self, provider_name, column, projected_columns):
        provider = self.catalog.get(provider_name)
        identity = provider.content_identity()
        projected_columns = tuple(dict.fromkeys(projected_columns))
        token_key = (identity, column)
        missing = [
            name for name in projected_columns
            if (identity, name) not in self._column_stores
        ]
        if token_key not in self._token_stores or missing:
            self._load(
                provider_name,
                None if token_key in self._token_stores else column,
                tuple(missing),
            )
        return ScanInput(
            self._token_stores[token_key],
            {
                name: self._column_stores[(identity, name)]
                for name in projected_columns
            },
        )

    def tokenize_async(self, provider_name: str, column: str,
                       projected_columns=()) -> Future:
        """Start tokenize() on a background thread and return its Future."""
        if self._background is None:
            self._background = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="quail-tokenize")
        return self._background.submit(
            self.tokenize, provider_name, column, projected_columns)

    def register_functions(self, functions: dict) -> None:
        """Register a query's apply() functions once each, by name."""
        for name, function in functions.items():
            known = self.registry.functions.get(name)
            if known is function:
                continue
            if known is not None:
                raise ValueError(
                    f"apply name {name!r} is registered for another "
                    f"function on this session")
            self.registry.register_function(function, name=name)

    def column_values(self, provider_name: str, column: str) -> pa.ChunkedArray:
        """Return one source column in scan order.

        Reads the column store when the column is loaded, else scans
        the provider for that one column.
        """
        provider = self.catalog.get(provider_name)
        with self._lock:
            store = self._column_stores.get(
                (provider.content_identity(), column))
        if store is not None:
            return store.values
        reader = provider.scan(ScanRequest(columns=(column,)))
        try:
            batches = [batch.column(0) for batch in reader]
        finally:
            reader.close()
        return pa.chunked_array(
            batches, type=provider.schema().field(column).type)

    def token_lengths(self, provider_name: str, column: str):
        """Return the exact token counts when the column is tokenized."""
        identity = self.catalog.get(provider_name).content_identity()
        store = self._token_stores.get((identity, column))
        return None if store is None else store.lengths

    def estimate_lengths(self, provider_name: str, column: str) -> list[int]:
        """Estimate document token counts from a tokenized sample.

        Reads the column's byte lengths and scales them by the tokens
        per byte measured on the first ESTIMATE_SAMPLE documents.
        """
        provider = self.catalog.get(provider_name)
        key = (provider.content_identity(), column)
        if key in self._length_estimates:
            return self._length_estimates[key]
        started = time.perf_counter()
        reader = provider.scan(ScanRequest(columns=(column,)))
        byte_lengths = []
        sample = []
        try:
            for batch in reader:
                texts = batch.column(0)
                byte_lengths.append(pc.binary_length(texts).cast(pa.int64()))
                if len(sample) < ESTIMATE_SAMPLE:
                    sample.extend(
                        texts.slice(0, ESTIMATE_SAMPLE - len(sample)).to_pylist()
                    )
        finally:
            reader.close()
        # Defer fast-tokenizer initialization to the background tokenization pass.
        tok = self.tokenizer
        sample_tokens = sum(len(tok(text)) for text in sample)
        sample_bytes = sum(len(text.encode("utf-8")) for text in sample)
        ratio = sample_tokens / sample_bytes if sample_bytes else 0.0
        lengths = []
        for chunk in byte_lengths:
            lengths.extend(
                max(1, round(n * ratio)) for n in chunk.to_pylist()
            )
        self._length_estimates[key] = lengths
        say(f"estimated {provider_name}.{column}: {len(lengths):,} documents, "
            f"about {sum(lengths):,} tokens from a {len(sample)} document "
            f"sample, {time.perf_counter() - started:.1f} s")
        return lengths

    def _corpus_tokenizer(self, provider_name, column, texts):
        """Cache the session tokenizer and token type for one column."""
        key = (self.catalog.get(provider_name).content_identity(), column)
        if key not in self._corpus_tokenizers:
            tok = self.tokenizer
            self.notes.append(
                f"{provider_name}.{column}: tokenizer: {self._tokenizer_name}"
            )
            first_token = next(
                (token for text in texts for token in tok(text)), None
            )
            token_type = (
                pa.int32()
                if first_token is None or isinstance(first_token, Integral)
                else pa.string()
            )
            self._corpus_tokenizers[key] = (tok, token_type)
        return self._corpus_tokenizers[key]

    def _store_path(self) -> str:
        if self._token_directory is None:
            self._token_directory = TemporaryDirectory(
                prefix="quail-tokens-"
            )
        self._store_count += 1
        return str(
            Path(self._token_directory.name)
            / f"input-{self._store_count}.arrow"
        )

    def _pick_tokenizer(self, reader, provider_name: str, column: str):
        """Sample the first documents to choose a tokenizer and token type."""
        buffered = []
        text_sample = []
        while len(text_sample) < 25:
            try:
                batch = next(reader)
            except StopIteration:
                break
            buffered.append(batch)
            texts = batch.column(batch.schema.get_field_index(column))
            needed = 25 - len(text_sample)
            text_sample.extend(texts.slice(0, needed).to_pylist())
        tok, token_type = self._corpus_tokenizer(
            provider_name, column, text_sample)
        return buffered, tok, token_type

    def _load(self, provider_name: str, column: str | None,
              value_columns: tuple[str, ...]) -> None:
        """Scan the provider once and write the missing store files.

        Args:
            provider_name: The registered provider.
            column: The document column to tokenize, or None when its
                token file already exists.
            value_columns: Value columns without a column file yet.
        """
        provider = self.catalog.get(provider_name)
        identity = provider.content_identity()
        scan_columns = tuple(dict.fromkeys(
            ((column,) if column is not None else ()) + value_columns
        ))
        scan_reader = provider.scan(ScanRequest(columns=scan_columns))
        reader = iter(scan_reader)
        token_writer = None
        column_writers = {}
        writers = []
        try:
            batches = reader
            if column is not None:
                buffered, tok, token_type = self._pick_tokenizer(
                    reader, provider_name, column)
                batches = chain(buffered, reader)
                token_writer = TokenStoreWriter(
                    self._store_path(),
                    document_column=column,
                    tokenizer=tok,
                    token_type=token_type,
                )
                writers.append(token_writer)
            source_schema = provider.schema()
            for name in value_columns:
                column_writers[name] = ColumnStoreWriter(
                    self._store_path(), source_schema.field(name))
                writers.append(column_writers[name])
            progress = None
            if column is not None:
                say(f"tokenizing {provider_name}.{column} "
                    f"({self.notes[-1].split(': ', 1)[-1]})")
                progress = Progress(f"tokenizing {provider_name}.{column}")
            rows = 0
            for batch in batches:
                for start in range(0, batch.num_rows, TOKENIZE_ROWS):
                    piece = batch.slice(start, TOKENIZE_ROWS)
                    for writer in writers:
                        writer.write_batch(piece)
                    rows += piece.num_rows
                    if progress is not None:
                        progress.update(rows)
        except Exception:
            for writer in writers:
                writer.abort()
            raise
        finally:
            scan_reader.close()
        token_store = (
            token_writer.finish() if token_writer is not None else None)
        if progress is not None:
            progress.finish(f"tokenized {provider_name}.{column}",
                            f"{sum(token_store.lengths):,} tokens")
        column_stores = {
            name: writer.finish() for name, writer in column_writers.items()
        }
        # every file for one provider must line up row for row, so a
        # later scan must return as many rows as the earlier one did
        known_rows = next(
            (len(store) for (store_identity, _), store in chain(
                self._token_stores.items(), self._column_stores.items())
             if store_identity == identity),
            None)
        written = list(column_stores.items())
        if token_store is not None:
            written.append((column, token_store))
        for name, store in written:
            rows = len(store)
            if known_rows is not None and rows != known_rows:
                for _, bad in written:
                    bad.close()
                    os.unlink(bad.path)
                raise RuntimeError(
                    f"{provider_name}.{name} returned {rows} rows but an "
                    f"earlier scan returned {known_rows}; the provider "
                    f"does not scan in a stable order")
        if token_store is not None:
            self._token_stores[(identity, column)] = token_store
        for name, store in column_stores.items():
            self._column_stores[(identity, name)] = store


class BoundBuilder:
    """Builder wrapper that returns a runnable Query from select()."""

    def __init__(self, session: Session, inner):
        self._session = session
        self._inner = inner

    def alias(self, a):
        self._inner.alias(a)
        return self

    def ai_filter(self, p, selectivity=None):
        self._inner.ai_filter(p, selectivity=selectivity)
        return self

    ai_if = ai_filter

    def join(self, other, on=None):
        inner = other._inner if isinstance(other, BoundBuilder) else other
        self._inner.join(inner, on=on)
        return self

    def apply(self, fn, columns=(), **options):
        self._inner.apply(fn, columns, **options)
        self._session.register_functions(self._inner.functions)
        return self

    def apply_table(self, fn, columns=(), **options):
        return self.apply(fn, columns, kind="barrier", **options)

    def ai_join(self, others, p, selectivity=None, anchor=None,
                semantics="full"):
        if not isinstance(others, (list, tuple)):
            others = [others]
        unwrapped = [o._inner if isinstance(o, BoundBuilder) else o
                     for o in others]
        self._inner.ai_join(unwrapped, p, selectivity=selectivity,
                            anchor=anchor, semantics=semantics)
        return self

    def limit(self, n):
        self._inner.limit(n)
        return self

    def select(self, *cols, order=None) -> "Query":
        return Query(self._session, self._inner.select(*cols),
                     order=order)


class Query:
    """A compiled query that plans and executes in the current process."""

    def __init__(self, session: Session, logical: LogicalPlan,
                 order: str | None = None):
        self.session = session
        self.logical = logical
        self.order = order
        self._plan = None
        self._doc_tokens = None
        self._token_inputs = None
        self._token_futures = {}
        self._token_finished_at = {}
        self._estimated = ()
        self._planning_started_at = None
        self._planning_finished_at = None
        self.token_wait_s = 0.0

    def token_inputs(self) -> dict:
        """Return the token store of every scanned alias.

        Plans first, then waits for any background tokenization, so
        planning and the speed of light estimate share these stores.
        """
        self.plan()
        self.wait_for_tokens()
        return self._token_inputs

    def plan(self):
        if self._plan is None:
            self._planning_started_at = time.perf_counter()
            self.logical, _ = apply_logical_rules(
                self.logical,
                tuple(self.session.registry.logical_rules.values()),
                LogicalPlanningContext(
                    self.session.catalog, self.session.config
                ),
            )
            operators = self.logical.operators()
            scans, joins = operators.scans, operators.joins
            self._doc_tokens = {}
            self._token_inputs = {}
            estimated = []
            for s in scans:
                columns = s.columns
                if self.session.config.backend in {
                    "stock_vllm", "pipelined_vllm", "dumb_vllm"
                }:
                    columns = tuple(dict.fromkeys((*columns, s.column)))
                exact = self.session.token_lengths(s.provider, s.column)
                if exact is not None:
                    store = self.session.tokenize(
                        s.provider, s.column, columns)
                    self._token_inputs[s.alias] = store
                    self._doc_tokens[s.alias] = store.lengths
                    continue
                self._doc_tokens[s.alias] = self.session.estimate_lengths(
                    s.provider, s.column)
                future = self.session.tokenize_async(
                    s.provider, s.column, columns)

                def record_finished(_future, alias=s.alias):
                    self._token_finished_at[alias] = time.perf_counter()

                future.add_done_callback(record_finished)
                self._token_futures[s.alias] = future
                estimated.append(s.alias)
            self._estimated = tuple(estimated)
            pair_fractions = self._pair_fractions(scans, joins)
            self._plan = plan_query(
                self.logical, model=self.session.model,
                device=self.session.device,
                doc_tokens=self._doc_tokens,
                gpus=self.session.config.gpus,
                order=self.order,
                backend=self.session.config.backend,
                registry=self.session.registry,
                tokenizer=self.session.tokenizer,
                pair_fractions=pair_fractions)
            if self.session.config.gpu_timing:
                self._plan = replace(self._plan, settings={
                    **self._plan.settings, "gpu_timing": True})
            self._planning_finished_at = time.perf_counter()
        return self._plan

    @property
    def planning_s(self) -> float:
        """Return wall time spent planning this query."""
        if self._planning_started_at is None:
            return 0.0
        finished = self._planning_finished_at or time.perf_counter()
        return finished - self._planning_started_at

    @property
    def input_ready_s(self) -> float:
        """Return time until planning and document tokenization finished."""
        if self._planning_started_at is None:
            return 0.0
        finished = [
            self._planning_finished_at or time.perf_counter(),
            *self._token_finished_at.values(),
        ]
        return max(finished) - self._planning_started_at

    def _pair_fractions(self, scans, joins) -> dict:
        """Pairs kept over the cross product, per join with conditions."""
        providers = {scan.alias: scan.provider for scan in scans}
        fractions = {}
        for position, join in enumerate(joins):
            oriented = oriented_join_conditions(join)
            if oriented is None:
                continue
            left_alias, right_alias, conditions = oriented
            left_keys = [
                self.session.column_values(
                    providers[left.alias], left.column)
                for left, _ in conditions
            ]
            right_keys = [
                self.session.column_values(
                    providers[right.alias], right.column)
                for _, right in conditions
            ]
            fractions[position] = pair_fraction(
                pair_table(left_alias, left_keys, right_alias, right_keys),
                len(self._doc_tokens[left_alias]),
                len(self._doc_tokens[right_alias]))
        return fractions

    def explain(self, *, verbose: bool = False,
                analyze: bool = False) -> str:
        """Return the optimized plan, optionally measured by running it.

        Args:
            verbose: Include runtime settings and internal node fields.
            analyze: Run the query first, like EXPLAIN ANALYZE, and show
                each node's measured rows, time, and tokens beside the
                planner's estimates, then the measured totals.
        """
        # plan() first: it replaces self.logical with the optimized tree
        physical = self.plan()
        result = None
        if analyze:
            result = self.run()
            # the root relation is lazy, so its row count is taken here
            rows = result.count()
            root = physical.graph.root.node_id
            result.node_metrics[root] = replace(
                result.node_metrics.get(root, NodeMetrics()),
                output_rows=rows)
            result.report.setdefault("node_metrics", {}).setdefault(
                root, {})["output_rows"] = rows
        text = explain(self.logical, physical, verbose=verbose,
                       result=result,
                       usd_per_hour=(
                           self.session.device.usd_per_hour or None))
        if self._estimated:
            text += ("\n\n  note: token counts for "
                     + ", ".join(repr(a) for a in self._estimated)
                     + f" are estimated from a {ESTIMATE_SAMPLE} document "
                     "sample")
        return text

    def wait_for_tokens(self) -> None:
        """Block until every background tokenization has finished.

        A plan made on estimated lengths then gets the physical rules
        that read the token stores applied with the exact inputs.
        """
        started = time.perf_counter()
        refine = bool(self._token_futures)
        for alias, future in list(self._token_futures.items()):
            self._token_inputs[alias] = future.result()
            self._doc_tokens[alias] = self._token_inputs[alias].lengths
            self._token_finished_at.setdefault(alias, time.perf_counter())
            del self._token_futures[alias]
        self.token_wait_s += time.perf_counter() - started
        if refine and self._plan is not None:
            self._plan = refine_plan(
                self._plan, model=self.session.model,
                device=self.session.device, doc_tokens=self._doc_tokens,
                gpus=self.session.config.gpus, order=self.order,
                backend=self.session.config.backend,
                registry=self.session.registry,
                tokenizer=self.session.tokenizer)

    def run(self, plan=None) -> QueryResult:
        """Execute the query in the current process.

        Args:
            plan: An edited PhysicalPlan from plan().insert(),
                remove(), or move(); the planner's own plan when
                omitted.
        """
        from quail.execution.execute import execute_query

        return execute_query(self, plan=plan)

    def submit(self, **_options):
        """Submitting needs Quail Server; local queries use run()."""
        raise RuntimeError(
            "submit() needs a Session with an endpoint; this session runs "
            "queries in the current process, so call run() or collect()")

    def execute_stream(self, batch_rows: int = 65_536,
                       limit: int | None = None) -> pa.RecordBatchReader:
        """Execute the query and stream Arrow record batches."""
        return self.run().execute_stream(
            batch_rows=batch_rows, limit=limit)

    def collect(self, limit: int | None = None,
                batch_rows: int = 65_536) -> pa.Table:
        """Execute the query and explicitly collect one Arrow table."""
        return self.run().collect(
            limit=limit, batch_rows=batch_rows)

    def _prepare_physical(self):
        """Bind the query's tokenized inputs to its physical plan."""
        plan = self.plan()
        if isinstance(plan, Refusal):
            raise RefusalError(plan)
        self.wait_for_tokens()
        inputs = {}
        logical_scans = {
            scan.alias: scan for scan in self.logical.operators().scans
        }
        for node in plan.nodes:
            if not isinstance(node, Scan):
                continue
            store = self._token_inputs[node.alias]
            logical_scan = logical_scans[node.alias]
            inputs[node.input_id] = document_input(
                store.tokens,
                (
                    store.column(logical_scan.column)
                    if logical_scan.column in store.projected_columns
                    else None
                ),
            )
        envelope = plan.to_envelope(self.session.registry.codecs)
        return PhysicalRequest(envelope, inputs, self._column_tables())

    def _column_tables(self) -> dict:
        """One value table per alias a HashJoin or an apply() reads."""
        needed = {}
        operators = self.logical.operators()
        for join in operators.joins:
            for condition in join_conditions(join):
                for ref in (condition.left, condition.right):
                    needed.setdefault(ref.alias, {})[ref.column] = None
        for apply in operators.applies:
            for ref in apply.columns:
                needed.setdefault(ref.alias, {})[ref.column] = None
        tables = {}
        for alias, columns in needed.items():
            store = self._token_inputs[alias]
            arrays = {alias: pa.array(range(len(store.lengths)), pa.int32())}
            for name in columns:
                arrays[name] = store.column(name)
            tables[columns_key(alias)] = pa.table(arrays)
        return tables

    def finish(self, response, coordinator_wall: float = 0.0) -> QueryResult:
        """Finish the physical graph and attach execution details."""
        plan = self.plan()
        out = response.metrics
        from quail.physical import AiJoin, Barrier

        expected_nodes = tuple(node for node in plan.nodes
                               if isinstance(node, (AiJoin, Barrier)))
        report = dict(
            backend=out.get("backend", plan.backend),
            wall_s=out["wall_s"], boot_s=out.get("boot_s"),
            estimated_seconds=plan.estimated_seconds,
            boot_kind=out.get("boot_kind"),
            boot=out.get("boot"),
            coordinator_wall_s=round(coordinator_wall, 2),
            fresh_tokens=out["fresh_tokens"],
            cached_tokens=out.get("cached_tokens"),
            stages=[],
            peak_gib=out.get("peak_gib"),
            order_rule=plan.settings.get("order_rule"),
            expected_join_plan=[
                {
                    "type": node.type_name,
                    "id": node.node_id,
                    **node.explain_fields(),
                }
                for node in expected_nodes
            ],
            executed_join_plan=out.get("executed_join_plan", []),
            kv_manager=out.get("kv_manager"),
            node_metrics=out.get("node_metrics", {}),
            backend_metrics=out.get("backend_metrics"),
            remarks=list(plan.remarks) + list(self.session.notes))
        for key in ("gpu_s", "chunks"):
            if key in out:
                report[key] = out[key]
        for key in (
            "evaluated_documents",
            "evaluated_document_pairs",
            "documents_per_second",
            "document_pairs_per_second",
            "usd_per_query",
        ):
            if key in out:
                report[key] = out[key]

        operators = self.logical.operators()
        scans, logical_filters, logical_joins = (
            operators.scans, operators.filters, operators.joins
        )
        scans_by_alias = {scan.alias: scan for scan in scans}

        def project(node, value):
            if not isinstance(node, Project):
                raise TypeError(type(node).__name__)
            if node.inputs[0].value_type is ValueType.JOIN_ANSWERS:
                value = true_answer_rows(value)
            relation = (
                IndexRelation.from_table(value)
                if isinstance(value, pa.Table) else value
            )
            if not isinstance(relation, IndexRelation):
                raise TypeError("Project needs an index relation")
            projection = []
            fields = []
            for name in node.columns:
                if name in relation.schema.names:
                    # a score column the graph computed; document
                    # index columns are aliases, never "alias.column"
                    projection.append((name, None))
                    fields.append(pa.field(
                        name, relation.schema.field(name).type
                    ))
                    continue
                try:
                    alias, column = name.split(".", 1)
                    scan = scans_by_alias[alias]
                except (ValueError, KeyError) as error:
                    raise CompileError(
                        f"unknown projection column {name!r}"
                    ) from error
                if alias not in relation.schema.names:
                    raise CompileError(
                        f"projection column {name!r} is not in the result"
                    )
                store = self._token_inputs[alias]
                if column not in store.projected_columns:
                    raise CompileError(
                        f"projection column {name!r} was not loaded by the "
                        f"scan of {alias!r}; the projection_pushdown "
                        f"logical rule is not registered")
                values = store.column(column)
                projection.append((alias, values))
                fields.append(pa.field(
                    name,
                    values.type,
                    nullable=values.null_count > 0,
                    metadata={
                        b"quail.alias": alias.encode("utf-8"),
                        b"quail.provider": scan.provider.encode("utf-8"),
                        b"quail.column": column.encode("utf-8"),
                    },
                ))
            return QueryResult(
                columns=list(node.columns),
                declaration=relation.declaration,
                document_index_schema=relation.schema,
                output_schema=pa.schema(
                    fields,
                    metadata={b"quail.kind": b"query_result"},
                ),
                projection=projection,
                report={},
                row_count=(value.num_rows if isinstance(value, pa.Table)
                           else None),
            )

        sources = {
            node.input_id: range(node.n_docs)
            for node in plan.nodes
            if isinstance(node, Scan)
        }
        observers = self.session.registry.new_observers()
        run = GenericRunner().run(
            plan.graph,
            ExecutionContext(
                runtimes=self.session.registry.runtimes,
                sources=sources,
                project=project,
                observers=observers,
            ),
            initial_outputs=response.outputs,
            initial_metrics={
                node_id: _node_metrics(metrics)
                for node_id, metrics in out.get("node_metrics", {}).items()
            },
        )
        if not isinstance(run.value, QueryResult):
            raise TypeError("physical graph root must return QueryResult")
        result = run.value
        result.plan = plan.graph
        result.node_metrics = {
            node_id: node_result.metrics
            for node_id, node_result in run.nodes.items()
        }
        # Saved reports need the graph and metrics without Python objects.
        report["executed_plan"] = encode_graph(
            plan.graph, self.session.registry.codecs)
        report["node_metrics"] = scalar_node_metrics(run.nodes)
        observer_reports = {
            observer.name: dict(observer.report())
            for observer in observers
        }
        if observer_reports:
            report["observers"] = observer_reports
        result.report = report

        answer_tables = {"filters": {}, "joins": {}}
        survivors = {
            scan.alias: list(range(len(self._doc_tokens[scan.alias])))
            for scan in scans
        }

        output_types = {
            PortRef(node.node_id, output.name): output.value_type
            for node in plan.nodes
            for output in node.outputs
        }
        filter_relations = {}
        join_relations = {}
        for ref, table in response.outputs.items():
            value_type = output_types.get(ref)
            metadata = table.schema.metadata or {}
            if value_type is ValueType.FILTER_ANSWERS:
                alias = metadata.get(b"quail.alias")
                if alias is None:
                    raise ValueError(
                        "a filter answer relation needs quail.alias metadata"
                    )
                filter_relations.setdefault(
                    alias.decode("utf-8"), []
                ).append(table)
            elif value_type is ValueType.JOIN_ANSWERS:
                written_pos = metadata.get(b"quail.written_pos")
                if written_pos is None:
                    raise ValueError(
                        "a join answer relation needs "
                        "quail.written_pos metadata"
                    )
                position = int(written_pos.decode("ascii"))
                if position in join_relations:
                    raise ValueError(
                        f"duplicate join answer relation {position}"
                    )
                join_relations[position] = table

        missing_filter_relations = set(logical_filters) - set(
            filter_relations
        )
        if missing_filter_relations:
            raise ValueError(
                "execution response is missing filter answer relations for "
                f"{sorted(missing_filter_relations)}"
            )
        missing_join_relations = set(range(len(logical_joins))) - set(
            join_relations
        )
        if missing_join_relations:
            raise ValueError(
                "execution response is missing join answer relations for "
                f"{sorted(missing_join_relations)}"
            )

        for node in plan.graph.topological_nodes():
            for output in node.outputs:
                if output.value_type is not ValueType.DOCUMENT_IDS:
                    continue
                ref = PortRef(node.node_id, output.name)
                if ref not in response.outputs:
                    continue
                table = response.outputs[ref]
                if len(table.column_names) != 1:
                    raise ValueError(
                        "a document id relation needs one alias column"
                    )
                alias = table.column_names[0]
                survivors[alias] = table.column(alias).to_pylist()

        for alias, relations in filter_relations.items():
            table = relations[0] if len(relations) == 1 else \
                pa.concat_tables(relations)
            positions = table.column("predicate").to_pylist()
            predicate_order = list(dict.fromkeys(int(pos) for pos in positions))
            predicate_order.extend(
                position for position in range(len(logical_filters[alias]))
                if position not in predicate_order
            )
            for index, written_pos in enumerate(predicate_order):
                mask = pc.equal(table.column("predicate"), written_pos)
                stage_table = table.filter(mask)
                answered = stage_table.column(alias).to_pylist()
                answers = stage_table.column("answer").to_pylist()
                passed = sum(bool(answer) for answer in answers)
                answer_tables["filters"][(alias, written_pos)] = stage_table
                report["stages"].append(dict(
                    op="filter", alias=alias, stage=index,
                    written_pos=written_pos,
                    provided_selectivity=(
                        logical_filters[alias][written_pos].selectivity
                    ),
                    observed_selectivity=round(
                        passed / max(1, len(answered)), 4
                    ),
                    evaluated=len(answered),
                ))

        true_join_tables = {}
        expected_order = []
        for node in expected_nodes:
            for output in node.outputs:
                if not output.name.startswith("join_answers:"):
                    continue
                expected_order.append(int(output.name.split(":", 1)[1]))
        expected_order.extend(
            position for position in sorted(join_relations)
            if position not in expected_order
        )
        for written_pos in expected_order:
            if written_pos not in join_relations:
                continue
            table = join_relations[written_pos]
            metadata = table.schema.metadata or {}
            logical_join = logical_joins[written_pos]
            logical_aliases = [
                argument.alias for argument in logical_join.prompt.args
            ]
            required_metadata = {
                b"quail.anchor", b"quail.partners", b"quail.semantics"
            }
            missing = required_metadata - set(metadata)
            if missing:
                raise ValueError(
                    "a join answer relation is missing metadata "
                    f"{sorted(key.decode('utf-8') for key in missing)}"
                )
            anchor = metadata[b"quail.anchor"].decode("utf-8")
            partner_text = metadata[b"quail.partners"].decode("utf-8")
            partners = [] if not partner_text else partner_text.split(",")
            semantics = metadata[b"quail.semantics"].decode("utf-8")
            missing_columns = set(logical_aliases) - set(table.column_names)
            if missing_columns:
                raise ValueError(
                    "a join answer relation is missing alias columns "
                    f"{sorted(missing_columns)}"
                )
            answers = table.column("answer")
            answer_tables["joins"][written_pos] = table
            report["stages"].append(dict(
                op="join", written_pos=written_pos, anchor=anchor,
                partners=partners,
                semantics=semantics,
                provided_selectivity=logical_join.selectivity,
                observed_selectivity=round(
                    (pc.sum(answers).as_py() or 0) / max(1, len(answers)), 4
                ),
                tuples=len(answers),
            ))
            if semantics == "full":
                true_join_tables[written_pos] = true_answer_rows(table)
        survivor_arrays = {
            alias: pa.array(indices, type=pa.int32())
            for alias, indices in survivors.items()
        }
        result.answer_tables = answer_tables
        result.survivor_indices = survivor_arrays
        result.true_join_tables = true_join_tables
        return result
