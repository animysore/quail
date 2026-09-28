"""The speed of light estimate for one query.

The estimate prices the ideal execution of a query on one device: every
distinct document prefix computed once and kept in KV without limit,
forward passes packed across the whole query, and no host, scheduling,
or kernel overhead. Filters run first in the planner's order. Joins
are searched over every feasible eager binary full left deep plan and
anchor choice, with exact survivors from a caller supplied answer
oracle at every step.

The equations are in docs/content/docs/architecture/planning.mdx. The
work counting and component pricing are the planner's own
(quail.cost.work, quail.cost.sol); the search does not call or
simulate the production planner.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

from quail.cost import budgets
from quail.cost.sol import SpeedOfLight, speed_of_light
from quail.cost.work import Work, ask, scan, triangle
from quail.execution.pairs import pair_table
from quail.logical import oriented_join_conditions
from quail.planner.decide import (
    default_order_rule,
    order_filters_indexed,
    preamble_tokens,
)
from quail.planner.leftdeep import Extension, optimize_left_deep
from quail.planner.live_rows import PairRelation, exact_live_rows
from quail.planner.prefixes import document_shared_tokens
from quail.specs import DeviceSpec, ModelSpec

# answer(prompt, assignment) -> bool, where assignment maps each alias
# in the prompt to a row index of that alias's corpus table
AnswerOracle = Callable[[object, Mapping[str, int]], bool]


@dataclass(frozen=True)
class SpeedOfLightEstimate:
    """The ideal time of one query and how it was reached."""

    model: str
    device: str
    chunk_tokens: int
    credit_shared_prefixes: bool
    work: Work
    latency: SpeedOfLight
    usd_per_hour: float
    alias_columns: dict[str, str]
    documents_by_alias: dict[str, int]
    post_filter_counts: dict[str, int]
    filter_stages: tuple[dict, ...]
    join_stages: tuple[dict, ...]
    relation_order: tuple[str, ...]
    cached_prefixes: tuple[str, ...]
    search: dict = field(default_factory=dict)

    @property
    def seconds(self) -> float:
        return self.latency.seconds

    @property
    def fresh_tokens(self) -> float:
        return self.work.tokens

    @property
    def usd_per_query(self) -> float:
        return self.seconds * self.usd_per_hour / 3600

    @property
    def input_document_rows(self) -> int:
        return sum(self.documents_by_alias.values())

    @property
    def filter_evaluations(self) -> int:
        return sum(stage["evaluated"] for stage in self.filter_stages)

    @property
    def join_pair_evaluations(self) -> int:
        return sum(stage["evaluated_pairs"] for stage in self.join_stages)

    def assumptions(self) -> dict:
        """Return what the estimate takes as given, for a report."""
        return {
            "persistent_kv_capacity": "unlimited",
            "gpu_count": 1,
            "survivors": "exact, from the caller's answers",
            "filter_order": "the planner's rule for this query",
            "plan_space": "all feasible eager binary full left deep plans",
            "cached_values": (
                "document prefixes used by filters or as anchors"),
            "cross_alias_prefix_reuse": (
                "an anchor row whose column another alias filtered, or "
                "whose row is live under another cached alias of the "
                "column, pays the frame only"
                if self.credit_shared_prefixes else
                "none; each alias computes its own documents"),
            "shared_prefixes_across_documents": (
                "each distinct token prefix computed once"
                if self.credit_shared_prefixes else
                "each document computed once"),
            "streamed_partner_kv": "not reusable",
            "chunk_tokens": self.chunk_tokens,
        }

    def as_dict(self) -> dict:
        """Return a JSON friendly record of the estimate."""
        latency = self.latency
        return {
            "model": self.model,
            "device": self.device,
            "chunk_tokens": self.chunk_tokens,
            "credit_shared_prefixes": self.credit_shared_prefixes,
            "sol_s": latency.seconds,
            "bound_by": latency.bound_by,
            "t_compute": latency.compute,
            "t_memory": latency.memory,
            "passes": latency.passes,
            "bytes_moved": latency.bytes_moved,
            "tokens": self.work.tokens,
            "pairs": self.work.pairs,
            "sliding_pairs": self.work.sliding_pairs,
            "sliding_kv_read": self.work.sliding_kv_read,
            "kv_written": self.work.kv_written,
            "kv_read": self.work.kv_read,
            "components": [
                {
                    "name": component.name,
                    "precision": component.precision,
                    "flops": component.flops,
                    "bytes_moved": component.bytes_moved,
                    "t_compute": component.compute_seconds,
                    "t_memory": component.memory_seconds,
                    "seconds": component.seconds,
                    "bound_by": component.bound_by,
                }
                for component in latency.components
            ],
            "usd_per_hour": self.usd_per_hour,
            "usd_per_query": self.usd_per_query,
            "alias_columns": dict(self.alias_columns),
            "documents_by_alias": dict(self.documents_by_alias),
            "input_document_rows": self.input_document_rows,
            "post_filter_counts": dict(self.post_filter_counts),
            "filter_evaluations": self.filter_evaluations,
            "join_pair_evaluations": self.join_pair_evaluations,
            "filter_stages": [dict(stage) for stage in self.filter_stages],
            "join_stages": [dict(stage) for stage in self.join_stages],
            "relation_order": list(self.relation_order),
            "cached_prefixes": list(self.cached_prefixes),
            "search": dict(self.search),
            "assumptions": self.assumptions(),
        }


@dataclass
class _AliasData:
    column: str
    tokens: list[int]
    shared: list[int]


def _prompt_token_counts(prompt):
    labels_by_alias = {
        alias: {"label": label_tokens, "frame": frame_tokens}
        for alias, label_tokens, frame_tokens in prompt.labels}
    return labels_by_alias, prompt.tail_tokens


class _Search:
    """One estimate's state: corpus tokens, the oracle, and the credit.

    The credit is the shared prefix counted as read from KV rather
    than computed, when credit_shared is on.
    """

    def __init__(self, query, answer: AnswerOracle, model: ModelSpec,
                 device: DeviceSpec, chunk_tokens: int,
                 credit_shared: bool):
        self.answer = answer
        self.model = model
        self.device = device
        self.chunk_tokens = chunk_tokens
        self.credit_shared = credit_shared
        # a diffusion model answers on canvas rows appended to every
        # evaluation's suffix; a decoder answers on the suffix's last row
        self.canvas = model.canvas_tokens
        self.window = model.sliding_window
        stores = query.token_inputs()
        operators = query.logical.operators()
        self.scans, self.filters, self.joins = (
            operators.scans, operators.filters, operators.joins
        )
        self.order = query.order
        self.pre = preamble_tokens(self.filters, self.joins)
        self.aliases: dict[str, _AliasData] = {}
        for scan_node in self.scans:
            store = stores[scan_node.alias]
            tokens = [int(length) for length in store.lengths]
            shared = (document_shared_tokens(store) if credit_shared
                       else [0] * len(tokens))
            self.aliases[scan_node.alias] = _AliasData(
                column=f"{scan_node.provider}.{scan_node.column}",
                tokens=tokens,
                shared=shared,
            )
        # column -> rows every filtered alias of that column computed
        # as prefixes; another alias of the column may reuse them
        self.computed_rows_by_column: dict[str, set[int]] = {}
        # joins without conditions are absent: every pair
        self.allowed_pairs: dict[int, dict[str, dict[int, set[int]]]] = {}
        for position, join in enumerate(self.joins):
            oriented = oriented_join_conditions(join)
            if oriented is None:
                continue
            left_alias, right_alias, conditions = oriented
            left_keys = [
                stores[left.alias].column(left.column)
                for left, _ in conditions
            ]
            right_keys = [
                stores[right.alias].column(right.column)
                for _, right in conditions
            ]
            pairs = pair_table(left_alias, left_keys, right_alias, right_keys)
            by_side = {left_alias: {}, right_alias: {}}
            for left_row, right_row in zip(
                    pairs.column(left_alias).to_pylist(),
                    pairs.column(right_alias).to_pylist()):
                by_side[left_alias].setdefault(left_row, set()).add(right_row)
                by_side[right_alias].setdefault(right_row, set()).add(left_row)
            self.allowed_pairs[position] = by_side

    # ---- work counting --------------------------------------------

    def first_use(self, alias: str, row: int, suffix: int) -> Work:
        """Compute one document's prefix for the first time in a query.

        The tokens an earlier document already computed are resident,
        so the document pays only for the rest of its prefix and the
        suffix.
        """
        data = self.aliases[alias]
        prefix = self.pre + data.tokens[row]
        shared_tokens = data.shared[row]
        if shared_tokens == 0:
            return scan(prefix, suffix, window=self.window)
        resident = self.pre + shared_tokens
        return ask(resident, prefix - resident + suffix, window=self.window)

    def join_stage_work(self, anchor, partners, survivors, prompt,
                        resident_rows, cross_resident_rows=(),
                        allowed=None) -> Work:
        """Return one join stage's work.

        Anchor rows in resident_rows have their prefix KV in the arena
        and pay the frame only; the rest scan. cross_resident_rows are
        anchor rows whose document prefix another alias of the same
        column computed; with the shared prefix credit on they pay the
        frame only too. allowed maps an anchor row to the partner rows
        its equality conditions keep; None streams every partner.
        """
        labels_by_alias, tail = _prompt_token_counts(prompt)
        partner_rows = list(itertools.product(
            *[survivors[alias] for alias in partners]))
        suffixes = [
            tail + self.canvas
            + sum(labels_by_alias[alias]["label"]
                  + self.aliases[alias].tokens[row]
                  for alias, row in zip(partners, partner_row))
            for partner_row in partner_rows
        ]
        all_tokens = sum(suffixes)
        all_triangles = sum(triangle(suffix) for suffix in suffixes)
        frame = labels_by_alias[anchor]["frame"]
        resident_rows = set(resident_rows)
        if self.credit_shared:
            resident_rows |= set(cross_resident_rows)
        work = Work()
        for row in survivors[anchor]:
            if allowed is None:
                streamed = suffixes
                suffix_tokens, suffix_triangles = all_tokens, all_triangles
            else:
                mine = allowed.get(row, ())
                streamed = [suffix for suffix, partner_row
                            in zip(suffixes, partner_rows)
                            if partner_row[0] in mine]
                suffix_tokens = sum(streamed)
                suffix_triangles = sum(triangle(suffix) for suffix in streamed)
            prefix = self.pre + self.aliases[anchor].tokens[row]
            work = work + (ask(prefix, frame, window=self.window)
                           if row in resident_rows
                           else self.first_use(anchor, row, frame))
            anchor_prefix = prefix + frame
            sliding_pairs = 0.0
            if self.window:
                sliding_pairs = (
                    self.window * suffix_tokens if anchor_prefix >= self.window - 1
                    else sum(triangle(anchor_prefix + suffix, self.window)
                             - triangle(anchor_prefix, self.window)
                             for suffix in streamed))
            work = work + Work(
                tokens=suffix_tokens,
                pairs=anchor_prefix * suffix_tokens + suffix_triangles,
                kv_written=suffix_tokens,
                kv_read=anchor_prefix,
                sliding_pairs=sliding_pairs,
                sliding_kv_read=(min(anchor_prefix, self.window - 1)
                                 if self.window else 0.0),
            )
        return work

    # ---- filters ---------------------------------------------------

    def run_filters(self):
        """Apply every filter in the planner's order with exact answers.

        Returns:
            (survivors by alias, work, filter stage records).
        """
        rule = (self.order if self.order is not None else
                default_order_rule(self.filters, self.joins)[0])
        survivors = {
            alias: list(range(len(data.tokens)))
            for alias, data in self.aliases.items()}
        work = Work()
        stages = []
        for alias, predicates in self.filters.items():
            tokens = self.aliases[alias].tokens
            mean_tokens = sum(tokens) / len(tokens) if tokens else 0
            order = order_filters_indexed(
                predicates, rule,
                prefix_tokens=self.pre + mean_tokens,
                model=self.model, device=self.device,
                chunk_tokens=self.chunk_tokens)
            live = survivors[alias]
            column = self.aliases[alias].column
            computed = self.computed_rows_by_column.setdefault(column, set())
            for stage_index, written_pos in enumerate(order):
                predicate = predicates[written_pos]
                question_tokens = predicate.prompt.tail_tokens
                suffix = question_tokens + self.canvas
                for row in live:
                    if stage_index == 0 and not (
                            self.credit_shared and row in computed):
                        work = work + self.first_use(alias, row, suffix)
                    else:
                        prefix = self.pre + tokens[row]
                        work = work + ask(prefix, suffix, window=self.window)
                if stage_index == 0:
                    computed.update(live)
                passed = [
                    row for row in live
                    if self.answer(predicate.prompt, {alias: row})
                ]
                stages.append({
                    "alias": alias,
                    "template": predicate.prompt.template,
                    "written_pos": written_pos,
                    "question_tokens": question_tokens,
                    "provided_selectivity": predicate.selectivity,
                    "evaluated": len(live),
                    "passed": len(passed),
                    "selectivity": (round(len(passed) / len(live), 6)
                                    if live else 0.0),
                })
                live = passed
            survivors[alias] = live
        return survivors, work, stages

    # ---- joins -----------------------------------------------------

    def anchor_fits(self, prompt, anchor, stage_aliases, live) -> bool:
        """Return whether the longest tuple fits one forward pass."""
        labels_by_alias, tail = _prompt_token_counts(prompt)
        anchor_max = max(
            (self.aliases[anchor].tokens[row] for row in live[anchor]),
            default=0,
        )
        partners = [alias for alias in stage_aliases if alias != anchor]
        prefix = self.pre + anchor_max + labels_by_alias[anchor]["frame"]
        if not live[anchor] or any(not live[partner] for partner in partners):
            return prefix <= self.chunk_tokens
        return (
            prefix
            + tail
            + sum(
                labels_by_alias[partner]["label"]
                + max(
                    (self.aliases[partner].tokens[row]
                     for row in live[partner]),
                    default=0,
                )
                for partner in partners
            )
            <= self.chunk_tokens
        )

    def cross_resident(self, anchor, live, live_cache) -> set[int]:
        """Return anchor rows whose prefix another alias of its column holds.

        A filtered alias computed every row of its column. An alias in
        live_cache holds the prefixes of its live rows; it computed at
        least those, so the credit is conservative.
        """
        column = self.aliases[anchor].column
        rows = set(self.computed_rows_by_column.get(column, ()))
        for other in live_cache:
            if other != anchor and self.aliases[other].column == column:
                rows.update(live[other])
        return rows

    def run_joins(self, base_rows, resident, base_work):
        """Search every feasible left deep plan over the filtered rows.

        Returns:
            The best candidate and the search result it came from.
        """
        joins = self.joins
        edge_relations = []
        edge_aliases = []
        for join in joins:
            stage_aliases = tuple(arg.alias for arg in join.prompt.args)
            if join.semantics != "full":
                raise NotImplementedError(
                    "the speed of light search needs full join semantics")
            if len(stage_aliases) != 2:
                raise NotImplementedError(
                    "the speed of light search needs binary join predicates")
            left, right = stage_aliases
            allowed = self.allowed_pairs.get(len(edge_relations), {}).get(
                left)
            passing = frozenset(
                (left_row, right_row)
                for left_row in base_rows[left]
                for right_row in base_rows[right]
                if (allowed is None or right_row in allowed.get(left_row, ()))
                and self.answer(
                    join.prompt, {left: left_row, right: right_row})
            )
            edge_relations.append(PairRelation(left, right, passing))
            edge_aliases.append(frozenset((left, right)))

        live_cache = {}

        def live_for(active_edges: frozenset[int]):
            if active_edges not in live_cache:
                live_cache[active_edges] = exact_live_rows(
                    base_rows,
                    tuple(edge_relations[index]
                          for index in sorted(active_edges)),
                )
            return live_cache[active_edges]

        def active_inside(relations: frozenset[str]) -> frozenset[int]:
            return frozenset(
                index for index, endpoints in enumerate(edge_aliases)
                if endpoints <= relations
            )

        extension_cache = {}

        def extend(relations, cached, added):
            cache_key = (relations, cached, added)
            if cache_key in extension_cache:
                return extension_cache[cache_key]
            crossing = tuple(
                index for index, endpoints in enumerate(edge_aliases)
                if added in endpoints and endpoints & relations
            )
            if not crossing:
                extension_cache[cache_key] = ()
                return ()
            initial_edges = active_inside(relations)
            extensions = []

            def visit_edges(order, position, active_edges, live, cached_now,
                            work, steps):
                if position == len(order):
                    extensions.append(Extension(
                        work=work, state_property=cached_now,
                        steps=tuple(steps)))
                    return
                edge_index = order[position]
                join = joins[edge_index]
                relation = edge_relations[edge_index]
                stage_aliases = tuple(
                    arg.alias for arg in join.prompt.args)
                for anchor in stage_aliases:
                    if not self.anchor_fits(
                            join.prompt, anchor, stage_aliases, live):
                        continue
                    partners = [alias for alias in stage_aliases
                                if alias != anchor]
                    allowed = self.allowed_pairs.get(edge_index, {}).get(
                        anchor)
                    stage_work = self.join_stage_work(
                        anchor, partners, live, join.prompt,
                        live[anchor] if anchor in cached_now else (),
                        self.cross_resident(anchor, live, cached_now),
                        allowed=allowed,
                    )
                    next_edges = active_edges | {edge_index}
                    next_cache = cached_now | {anchor}
                    passing = sum(
                        1
                        for left_row in live[relation.left]
                        for right_row in live[relation.right]
                        if (left_row, right_row) in relation.pairs
                    )
                    step = {
                        "template": join.prompt.template,
                        "written_pos": edge_index,
                        "added_alias": added,
                        "aliases": list(stage_aliases),
                        "anchor": anchor,
                        "partners": partners,
                        "evaluated_pairs": (
                            math.prod(len(live[alias])
                                      for alias in stage_aliases)
                            if allowed is None else sum(
                                sum(1 for partner_row in live[partners[0]]
                                    if partner_row in allowed.get(row, ()))
                                for row in live[anchor])),
                        "passing_pairs": passing,
                        "fresh_tokens": stage_work.tokens,
                        "cached_prefixes_after": sorted(next_cache),
                    }
                    visit_edges(
                        order, position + 1, next_edges,
                        live_for(next_edges), next_cache,
                        work + stage_work, steps + [step],
                    )

            for order in itertools.permutations(crossing):
                visit_edges(order, 0, initial_edges, live_for(initial_edges),
                            cached, Work(), [])
            extension_cache[cache_key] = tuple(extensions)
            return extension_cache[cache_key]

        search = optimize_left_deep(
            tuple(scan_node.alias for scan_node in self.scans),
            frozenset(resident),
            base_work,
            extend,
        )
        if not search.candidates:
            raise ValueError("query join graph has no connected left deep plan")

        def candidate_seconds(candidate):
            return speed_of_light(
                candidate.work, self.model, self.device,
                self.chunk_tokens).seconds

        best = min(
            search.candidates,
            key=lambda candidate: (
                candidate_seconds(candidate),
                candidate.work.tokens,
                candidate.work.pairs,
                candidate.work.kv_written,
                candidate.work.kv_read,
                candidate.relation_order,
            ),
        )
        return best, search


def speed_of_light_estimate(
    query,
    answer: AnswerOracle,
    *,
    model: ModelSpec | None = None,
    device: DeviceSpec | None = None,
    chunk_tokens: int | None = None,
    credit_shared_prefixes: bool = True,
) -> SpeedOfLightEstimate:
    """Return the ideal time of one query on one device.

    Args:
        query: A query built through a session. Its session tokenizes
            the scanned columns; the query is not planned or run.
        answer: Callable (prompt, assignment) -> bool giving the exact
            answer of a filter or join prompt for one row assignment,
            where assignment maps each alias in the prompt to a row
            index. Survivors at every stage come from it.
        model: Model to price; the session's model by default.
        device: Device to price; the session's device by default.
        chunk_tokens: Forward pass size; the planner's chunk budget for
            the model and device by default.
        credit_shared_prefixes: True computes each distinct token
            prefix once across documents and aliases of one column.
            False computes each alias's document once and reuses it
            only across that document's own questions.
    """
    model = model or query.session.model
    device = device or query.session.device
    if chunk_tokens is None:
        chunk_tokens = budgets.chunk_budget(model, device)
    search = _Search(query, answer, model, device, chunk_tokens,
                     credit_shared_prefixes)
    survivors, filter_work, filter_stages = search.run_filters()
    resident = frozenset(search.filters)
    base_rows = {alias: tuple(rows) for alias, rows in survivors.items()}
    best, result = search.run_joins(base_rows, resident, filter_work)
    return SpeedOfLightEstimate(
        model=model.name,
        device=device.name,
        chunk_tokens=chunk_tokens,
        credit_shared_prefixes=credit_shared_prefixes,
        work=best.work,
        latency=speed_of_light(best.work, model, device, chunk_tokens),
        usd_per_hour=device.usd_per_hour,
        alias_columns={
            alias: data.column for alias, data in search.aliases.items()},
        documents_by_alias={
            alias: len(data.tokens) for alias, data in search.aliases.items()},
        post_filter_counts={
            alias: len(rows) for alias, rows in survivors.items()},
        filter_stages=tuple(filter_stages),
        join_stages=tuple(best.steps),
        relation_order=tuple(best.relation_order),
        cached_prefixes=tuple(sorted(best.state_property)),
        search={
            "dp_states": result.state_count,
            "dp_records_generated": result.generated_count,
            "dp_final_records": len(result.candidates),
        },
    )
