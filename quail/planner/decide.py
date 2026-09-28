"""Choose filter order, joins, anchors, and KV retention before execution."""

from dataclasses import replace

from quail.cost import budgets
from quail.cost.retention import coefficients, retention_pages
from quail.cost.sol import speed_of_light, unrounded_seconds
from quail.cost.work import Work, ask, scan
from quail.logical import (
    DEFAULT_SELECTIVITY,
    CompileError,
    LogicalPlan,
    effective_selectivity,
    join_conditions,
    oriented_join_conditions,
)
from quail.physical import (
    AiFilter,
    AiJoin,
    Barrier,
    Exchange,
    FilterStage,
    Foreign,
    HashJoin,
    JoinStage,
    Limit,
    PortRef,
    Recombine,
)
from quail.physical import (
    Project as PhysicalProject,
)
from quail.physical import (
    Scan as PhysicalScan,
)
from quail.physical.base import input_ports
from quail.planner import joins as joinsearch
from quail.planner import retention
from quail.planner.physical_optimizer import (
    ModelRegion,
    PlanningContext,
    apply_physical_rules,
)
from quail.planner.plan import CorpusStats, PhysicalPlan, Refusal
from quail.specs import DeviceSpec, ModelSpec

# ---------------------------------------------------------- tree walk

def _question_tokens(prompt, canvas: int = 0) -> int:
    """Token count of the prompt's per-evaluation tail.

    canvas is the rows a diffusion model appends to every evaluation
    to answer on; a decoder answers on the tail's last row.
    """
    if prompt.tail_tokens is None or prompt.preamble_tokens is None:
        raise ValueError(
            "prompts were bound without a tokenizer; the planner "
            "needs token counts (pass one to compile_sql / docs)")
    return prompt.tail_tokens + canvas


def preamble_tokens(filters, joins) -> int:
    """Return the engine preamble's token count from any bound prompt."""
    for fs in filters.values():
        for p in fs:
            if p.prompt.preamble_tokens is not None:
                return p.prompt.preamble_tokens
    for j in joins:
        if j.prompt.preamble_tokens is not None:
            return j.prompt.preamble_tokens
    return 0


# --------------------------------------------------------- filter order

def default_order_rule(filters, joins) -> tuple[str, str]:
    """Return (rule, source).

    Always 'by_cost'; a predicate without a selectivity is priced with
    DEFAULT_SELECTIVITY. Pass order="as_written" to keep written order.
    """
    missing = any(p.selectivity is None for fs in filters.values()
                  for p in fs) or any(j.selectivity is None for j in joins)
    if missing:
        return "by_cost", (
            "default: by cost, with selectivity "
            f"{DEFAULT_SELECTIVITY:g} for predicates without one")
    return "by_cost", "default: by cost"


def filter_cost(predicate, prefix_tokens: float, model: ModelSpec,
                device: DeviceSpec, chunk_tokens: int, *, first: bool) -> float:
    """Return ideal time for one filter evaluation."""
    operation = scan if first else ask
    work = operation(prefix_tokens,
                     _question_tokens(predicate.prompt, model.canvas_tokens),
                     window=model.sliding_window)
    return unrounded_seconds(work, model, device, chunk_tokens)


def order_filters_indexed(predicates, rule: str, *, prefix_tokens: float,
                          model: ModelSpec, device: DeviceSpec,
                          chunk_tokens: int):
    """Return written positions of predicates in execution order.

    'by_cost' minimizes ideal expected time. It sorts the asks once,
    then prices each predicate as the first scan. Written order breaks
    ties.
    """
    idx = list(range(len(predicates)))
    if rule == "as_written":
        return idx

    def selectivity(i):
        return effective_selectivity(predicates[i].selectivity)

    ask_costs = [
        filter_cost(p, prefix_tokens, model, device, chunk_tokens,
                    first=False)
        for p in predicates
    ]
    scan_costs = [
        filter_cost(p, prefix_tokens, model, device, chunk_tokens,
                    first=True)
        for p in predicates
    ]

    def score(i):
        killed = 1.0 - selectivity(i)
        if killed <= 0:
            return float("inf")
        return ask_costs[i] / killed

    ask_order = sorted(idx, key=score)
    prefix_live = [1.0]
    prefix_cost = [0.0]
    for i in ask_order:
        prefix_cost.append(
            prefix_cost[-1] + prefix_live[-1] * ask_costs[i])
        prefix_live.append(prefix_live[-1] * selectivity(i))

    total_ask_cost = prefix_cost[-1]
    candidates = []
    for position, first in enumerate(ask_order):
        expected = (
            scan_costs[first]
            + selectivity(first) * prefix_cost[position]
            + total_ask_cost - prefix_cost[position + 1]
        )
        candidates.append((expected, first))

    first = min(candidates)[1]
    return [first, *(i for i in ask_order if i != first)]


def order_filters(predicates, rule: str, *, prefix_tokens: float,
                  model: ModelSpec, device: DeviceSpec,
                  chunk_tokens: int):
    return [predicates[i] for i in order_filters_indexed(
        predicates, rule, prefix_tokens=prefix_tokens,
        model=model, device=device, chunk_tokens=chunk_tokens)]


def _join_aliases(join) -> list:
    """Return the join's table aliases in placeholder order."""
    return [r.alias for r in join.prompt.args]


def _label_counts(join) -> dict:
    """Return alias -> (block_label_tokens, anchor_frame_tokens)."""
    out = {a: (lt, nt) for a, lt, nt in join.prompt.labels}
    if any(lt is None for lt, _ in out.values()):
        raise ValueError(
            "join prompts were bound without a tokenizer; the planner "
            "needs token counts (pass one to compile_sql / docs)")
    return out


def join_specs(joins, pair_fractions=None, canvas: int = 0) -> list:
    """The joins as the search's spec dicts, in written order.

    pair_fractions maps a written position to the fraction of the
    cross product its equality conditions keep; a join with
    conditions but no entry is priced as the full cross product.
    canvas is the rows a diffusion model appends to every pair.
    """
    pair_fractions = pair_fractions or {}
    out = []
    for i, j in enumerate(joins):
        labels = _label_counts(j)
        conditions = join_conditions(j)
        out.append(dict(
            written_pos=i, aliases=_join_aliases(j), anchor=j.anchor,
            anchor_free=(j.anchor is None and j.semantics == "full"),
            semantics=j.semantics, selectivity=j.selectivity,
            frame_tokens={a: nt for a, (lt, nt) in labels.items()},
            label_tokens={a: lt for a, (lt, nt) in labels.items()},
            tail_tokens=_question_tokens(j.prompt, canvas),
            on=[(c.left.alias, c.left.column, c.right.alias, c.right.column)
                for c in conditions],
            pair_fraction=(pair_fractions.get(i, 1.0) if conditions
                           else 1.0)))
    return out


def hash_join_nodes(joins, pair_fractions, scan_ports) -> list:
    """One HashJoin per join with equality conditions, over the scans.

    Args:
        joins: The logical joins in written order.
        pair_fractions: written position -> pairs kept over the cross
            product, as the session measured them.
        scan_ports: The scans' id ports, one per alias, in any order.

    Returns:
        HashJoin nodes with ids ``hash_join:<left>-<right>``.
    """
    pair_fractions = pair_fractions or {}
    by_alias = {port.port.split(":", 1)[1]: port for port in scan_ports}
    nodes = []
    for position, join in enumerate(joins):
        oriented = oriented_join_conditions(join)
        if oriented is None:
            continue
        left, right, conditions = oriented
        on = tuple(
            (left_ref.column, right_ref.column)
            for left_ref, right_ref in conditions
        )
        nodes.append(HashJoin(
            node_id=f"hash_join:{left}-{right}",
            inputs=input_ports((by_alias[left], by_alias[right])),
            left=left, right=right, on=on, written_pos=position,
            pair_fraction=pair_fractions.get(position, 1.0)))
    return nodes


def _filter_alias_work(preds, stats, order, pre: int,
                       canvas: int = 0, window: int = 0) -> Work:
    """Expected Work of one filter chain: a scan, then asks over KV."""
    total = Work()
    mean = stats.mean_doc_tokens
    n = float(stats.n_docs)
    for si, predicate_index in enumerate(order):
        p = preds[predicate_index]
        q = _question_tokens(p.prompt, canvas)
        op = scan if si == 0 else ask
        total = total + op(pre + mean, q, window=window) * n
        n *= effective_selectivity(p.selectivity)
    return total


def node_estimates(graph, *, filter_works, stage_works, live, stats, pre,
                   cap_pages, model, device, chunk) -> dict:
    """Price each node's own work, and each chain's recompute if released.

    Returns node id -> {"seconds", and for a filter chain a later join
    anchors on, "release_recompute_tokens" and
    "release_recompute_seconds"}. A node's seconds price its work
    alone; nodes do not add up to the plan's estimate because chunk
    packing shares forward passes across them. The recompute figure
    is the expected survivors past the retention pool cap times their
    prefix cost: what the join pays if the chain's KV is not pinned.
    """
    anchored = {node.anchor for node in graph.nodes if isinstance(node, AiJoin)}
    out = {}
    for node in graph.nodes:
        entry = {}
        if isinstance(node, AiFilter) and node.alias in filter_works:
            entry["seconds"] = speed_of_light(
                filter_works[node.alias], model, device, chunk).seconds
            if node.alias in anchored:
                mean = stats[node.alias].mean_doc_tokens
                prefix = pre + mean
                pages = -(-(prefix + node.hold_tokens) // budgets.PAGE_TOKENS)
                fits = cap_pages // max(1, pages)
                excess = max(0.0, live.get(node.alias, 0.0) - fits)
                entry["release_recompute_tokens"] = round(excess * prefix)
                entry["release_recompute_seconds"] = speed_of_light(
                    scan(prefix, 0, window=model.sliding_window) * excess,
                    model, device, chunk).seconds
        elif isinstance(node, AiJoin):
            work = Work()
            for stage in node.stages:
                work = work + stage_works.get(stage.written_pos, Work())
            entry["seconds"] = speed_of_light(work, model, device, chunk).seconds
        # other nodes do no model work and get no seconds
        out[node.node_id] = entry
    return out


# ----------------------------------------------- KV keep (residency)

def balanced_shards(doc_tokens, workers: int):
    """Greedily partition documents into shards balanced by token count."""
    if workers == 1:
        return (tuple(range(len(doc_tokens))),), [sum(doc_tokens)]
    order = sorted(range(len(doc_tokens)), key=lambda i: -doc_tokens[i])
    loads = [0] * workers
    shards = [[] for _ in range(workers)]
    for i in order:
        w = loads.index(min(loads))
        shards[w].append(i)
        loads[w] += doc_tokens[i]
    return tuple(tuple(sorted(s)) for s in shards), loads


def contiguous_shards(doc_tokens, workers: int):
    """Split ordered documents into compact token balanced ranges."""
    lengths = doc_tokens
    n_docs = len(lengths)
    total = sum(lengths)
    ranges = []
    loads = []
    start = 0
    consumed = 0
    for worker in range(workers):
        if worker == workers - 1:
            stop = n_docs
            load = total - consumed
        else:
            remaining_workers = workers - worker
            target = (total - consumed) / remaining_workers
            stop = start
            load = 0
            while stop < n_docs:
                next_length = int(lengths[stop])
                with_next = load + next_length
                if load and abs(target - load) <= abs(target - with_next):
                    break
                load = with_next
                stop += 1
                if load >= target:
                    break
        ranges.append((start, stop))
        loads.append(load)
        start = stop
        consumed += load
    return tuple(ranges), loads


# ---------------------------------------------------------- the planner

def plan_quail(plan: LogicalPlan, *, model: ModelSpec,
                device: DeviceSpec, doc_tokens: dict, gpus: int = 1,
                order: str | None = None, pair_fractions=None):
    """Compile a LogicalPlan into a PhysicalPlan or Refusal.

    Args:
        plan: The logical plan to compile.
        model: Model spec.
        device: Device spec.
        doc_tokens: alias -> list of per-document token counts.
        gpus: GPU count; one model copy runs per GPU.
        order: Stage order rule, 'by_cost' or 'as_written'; None picks
            the default rule.
        pair_fractions: join written position -> the fraction of the
            cross product its equality conditions keep.
    """
    operators = plan.operators()
    scans, filters, joins = operators.scans, operators.filters, operators.joins
    applies = operators.applies
    alias_applies = {}
    join_applies = {}
    for apply in applies:
        if apply.ids == "pairs":
            join_applies.setdefault(apply.written_pos, []).append(apply)
        else:
            alias_applies.setdefault(apply.aliases[0], []).append(apply)
    names = [apply.function for apply in applies]
    if len(set(names)) != len(names):
        raise CompileError(
            f"each apply() needs its own name; {names} repeat one")
    if any(len(group) > 1 for group in join_applies.values()):
        raise CompileError("a join takes one apply() returning pairs")
    if gpus > 1 and any(apply.kind == "per_batch" for apply in applies):
        return Refusal(
            reasons=("per-batch apply() requires one GPU; "
                     "use apply_table() or set gpus=1",),
            constraint="per_batch_apply_needs_one_gpu",
            needed=1, available=gpus, unit="gpus")
    length_stats = {a: joinsearch.summarize_alias(t, model.sliding_window)
                    for a, t in doc_tokens.items()}
    stats = {
        a: CorpusStats(n_docs=s.count, total_tokens=s.total,
                       max_doc_tokens=s.maximum)
        for a, s in length_stats.items()
    }
    for s in scans:
        if s.alias not in stats:
            raise ValueError(f"no doc_tokens for alias {s.alias!r}")
    remarks = []

    # ---- refusals first
    weight_gpus = budgets.minimum_weight_gpus(model, device)
    if weight_gpus > 1:
        return Refusal(
            reasons=(
                f"the model needs {weight_gpus} GPUs of memory, "
                f"but Quail loads one complete copy per GPU",
            ),
            constraint="weights_need_more_cards",
            needed=weight_gpus, available=1, unit="cards")
    workers = gpus

    chunk = budgets.chunk_budget(model, device)
    # the longest documents bind the sliding-pool split
    longest_mean = max(
        (st.mean_doc_tokens for st in stats.values()), default=None)
    arena_split = budgets.arena_pages(model, device, chunk, longest_mean)
    admission = arena_split[0] * budgets.PAGE_TOKENS
    pre = preamble_tokens(filters, joins)
    specs = join_specs(joins, pair_fractions, model.canvas_tokens)

    # ---- the order rule first: the search below needs it
    rule, source = (order, f"user: order={order!r}") if order else \
        default_order_rule(filters, joins)
    fixed = rule == "as_written"
    filter_orders = {
        alias: order_filters_indexed(
            predicates, rule,
            prefix_tokens=pre + stats[alias].mean_doc_tokens,
            model=model, device=device, chunk_tokens=chunk)
        for alias, predicates in filters.items()
    }

    # ---- expected live counts after filters, and the fixed filter
    # work every candidate join plan shares
    live0 = {a: float(st.n_docs) for a, st in stats.items()}
    for fs in filters.values():
        surv = 1.0
        for p in fs:
            surv *= effective_selectivity(p.selectivity)
        live0[_filter_alias(fs[0])] *= surv
    filter_works = {
        alias: _filter_alias_work(preds, stats[alias], filter_orders[alias],
                                  pre, model.canvas_tokens, model.sliding_window)
        for alias, preds in filters.items()
    }
    base_work = sum(filter_works.values(), Work())

    cap_pages = retention_pages(admission, chunk, budgets.PAGE_TOKENS)
    costs = coefficients(model, device)
    # KV reuse is priced as unlimited
    filtered = set(filters)

    def run_search(honor_forced=True):
        found = joinsearch.search_joins(
            specs, live0, length_stats, filtered, pre,
            chunk, model, device, base_work=base_work,
            fixed_order=fixed, honor_forced=honor_forced)
        if found is None:
            found = joinsearch.search_joins(
                specs, live0, length_stats, filtered, pre, chunk, model,
                device, base_work=base_work, fixed_order=True,
                honor_forced=honor_forced)
        return found

    found = run_search()
    seq = [(specs[position], anchor) for position, anchor in found["seq"]]
    # a second group on the same anchor gets :2, a third :3
    sequence_groups = retention.group_sequence(seq)
    group_ids = []
    seen_ids = {}
    for group in sequence_groups:
        anchor = group[0][1]
        seen_ids[anchor] = seen_ids.get(anchor, 0) + 1
        group_ids.append(f"ai_join:{anchor}" if seen_ids[anchor] == 1
                         else f"ai_join:{anchor}:{seen_ids[anchor]}")

    def unique_id(prefix, name):
        key = f"{prefix}:{name}"
        seen_ids[key] = seen_ids.get(key, 0) + 1
        return key if seen_ids[key] == 1 else f"{key}:{seen_ids[key]}"

    retention_plan = retention.schedule(seq, live0, group_ids)
    # a chain streams only into its alias's first use, and never through
    # a barrier apply
    barrier_aliases = {alias for alias, group in alias_applies.items()
                       if any(apply.kind == "barrier" for apply in group)}
    for group in sequence_groups:
        anchor = group[0][1]
        for spec, _ in group:
            if any(apply.kind == "barrier"
                   for apply in join_applies.get(spec["written_pos"], ())):
                barrier_aliases.add(anchor)
    streamed = {}
    partner_before = set()
    for index, group in enumerate(sequence_groups):
        anchor = group[0][1]
        if anchor in filters and anchor not in streamed \
                and anchor not in partner_before \
                and anchor not in barrier_aliases:
            streamed[anchor] = index
        for spec, _ in group:
            for alias in spec["aliases"]:
                if alias != anchor and alias not in streamed:
                    partner_before.add(alias)
    retention_plan["initial"] = {
        alias: use for alias, use in retention_plan["initial"].items()
        if alias not in streamed
    }
    retention_plan.update(**costs, cap_pages=cap_pages)
    forced = sorted({s["anchor"] for s in specs
                     if s["semantics"] == "full"
                     and not s["anchor_free"]})
    if forced:
        free = run_search(honor_forced=False)
        honored_s = speed_of_light(
            base_work + found["work"], model, device, chunk).seconds
        free_s = speed_of_light(
            base_work + free["work"], model, device, chunk).seconds
        if free_s < honored_s:
            remarks.append(
                f"anchors {forced} were forced; a free choice prices "
                f"lower ({free_s:.3f} vs {honored_s:.3f} predicted "
                f"seconds)")
    stage_records = found["records"]

    # ---- refusal checks on the predicted plan
    anchors = {wp: a for wp, a in found["seq"]}
    for s in scans:
        fq = max((_question_tokens(p.prompt, model.canvas_tokens)
                  for p in filters.get(s.alias, ())), default=None)
        if fq is None:
            continue
        need = pre + stats[s.alias].max_doc_tokens + fq
        if need > chunk:
            return Refusal(
                reasons=(f"a document in {s.alias!r} needs {need} tokens "
                         f"with its prompt, but the forward pass budget "
                         f"is {chunk} tokens",),
                constraint="suffix_over_chunk",
                needed=need, available=chunk, unit="tokens")
    for spec in specs:
        anchor = anchors[spec["written_pos"]]
        need = (pre + stats[anchor].max_doc_tokens
                + spec["frame_tokens"][anchor]
                + sum(spec["label_tokens"][p] + stats[p].max_doc_tokens
                      for p in spec["aliases"] if p != anchor)
                + spec["tail_tokens"])
        if need > chunk:
            return Refusal(
                reasons=(f"one join pair anchored on {anchor!r} needs "
                         f"{need} tokens, but the forward pass budget "
                         f"is {chunk} tokens",),
                constraint="suffix_over_chunk",
                needed=need, available=chunk, unit="tokens")

    # ---- build the dataflow graph; ids_src tracks each table's
    # current producer node
    nodes = []
    ids_src = {}
    for s in scans:
        shard_ranges, loads = contiguous_shards(
            doc_tokens[s.alias], workers
        )
        sid = f"scan:{s.alias}"
        nodes.append(PhysicalScan(
            node_id=sid,
            alias=s.alias, input_id=s.alias,
            n_docs=stats[s.alias].n_docs,
            total_tokens=stats[s.alias].total_tokens,
            shard_ranges=shard_ranges,
            shard_token_loads=tuple(loads)))
        ids_src[s.alias] = PortRef(sid, f"ids:{s.alias}")
    # the hash join reads the scans; survivors thin its pairs at the AI join
    pairs_src = {}
    for node in hash_join_nodes(joins, pair_fractions, ids_src.values()):
        nodes.append(node)
        pairs_src[node.written_pos] = node

    def emit_filter(alias):
        order_idx = filter_orders[alias]
        n = stats[alias].n_docs
        stages, surv = [], 1.0
        for i in order_idx:
            p = filters[alias][i]
            stages.append(FilterStage(
                written_pos=i,
                question_tokens=_question_tokens(p.prompt, model.canvas_tokens),
                preamble_tokens=p.prompt.preamble_tokens,
                selectivity=p.selectivity,
                expected_docs=round(n * surv, 1)))
            surv *= effective_selectivity(p.selectivity)
        keep = alias in retention_plan["initial"]
        pinned = alias in streamed
        writes = len(stages) > 1 or keep or pinned
        # pinned pages also cover the consuming join's largest frame
        hold = max((spec["frame_tokens"][alias]
                    for spec, _ in sequence_groups[streamed[alias]])
                   if pinned else (0,))
        fid = f"ai_filter:{alias}"
        nodes.append(AiFilter(
            node_id=fid,
            inputs=input_ports((ids_src[alias],)),
            alias=alias, arena_writes=writes,
            keep_kv=keep, pin_survivors=pinned, hold_tokens=hold,
            stages=tuple(stages)))
        ids_src[alias] = PortRef(fid, f"ids:{alias}")
        if pinned:
            remarks.append(
                f"filter on {alias!r} streams its survivors into "
                f"the join anchored on it; each one's KV stays "
                f"pinned until its tuples are answered")
        elif not writes:
            remarks.append(
                f"filter on {alias!r}: arena writes off (one "
                f"stage - nothing reads the KV again)")
        emit_applies(alias)

    def emit_applies(alias):
        for apply in alias_applies.get(alias, ()):
            aid = f"apply:{apply.function}"
            nodes.append(Foreign(
                node_id=aid,
                inputs=input_ports((ids_src[alias],)),
                function=apply.function, kind=apply.kind, ids=apply.ids,
                columns=tuple((ref.alias, ref.column)
                              for ref in apply.columns),
                aliases=(alias,)))
            ids_src[alias] = PortRef(aid, f"ids:{alias}")

    for s in scans:
        if s.alias in filters and s.alias not in streamed:
            emit_filter(s.alias)
        elif s.alias not in filters:
            emit_applies(s.alias)

    # group consecutive full stages on the same anchor; gates run
    # alone; anchor switches become barriers
    records = iter(stage_records)
    groups = [[(spec, next(records)) for spec, _ in group]
              for group in sequence_groups]
    exec_idx = 0
    pairs_edges = []     # every full stage's passing-pairs edge
    out_aliases = []     # recombination's output order
    for g, group in enumerate(groups):
        anchor = sequence_groups[g][0][1]
        if g > 0:
            ahead = [scan.alias for scan in scans]
            bid = unique_id("barrier", anchor)
            exchange_inputs = tuple(pairs_edges) + tuple(
                ids_src[a] for a in ahead
            )
            nodes.append(Barrier(
                node_id=bid,
                inputs=input_ports(exchange_inputs),
                next_anchor=anchor,
                aliases=tuple(ahead)))
            for a in ahead:
                ids_src[a] = PortRef(bid, f"ids:{a}")
        if workers > 1:
            # anchors go to the GPU holding their KV, else balanced
            xid = unique_id("exchange", anchor)
            nodes.append(Exchange(
                node_id=xid,
                inputs=input_ports((ids_src[anchor],)),
                anchor=anchor))
            ids_src[anchor] = PortRef(xid, f"ids:{anchor}")
        if streamed.get(anchor) == g:
            emit_filter(anchor)
        gid = group_ids[g]
        stage_dicts = []
        in_aliases = [anchor]
        pair_inputs = []
        for spec, record in group:
            partners = [a for a in spec["aliases"] if a != anchor]
            pairs_from = ""
            if spec["written_pos"] in pairs_src:
                producer = pairs_src[spec["written_pos"]]
                pair_inputs.append(
                    PortRef(producer.node_id, f"pairs:{spec['written_pos']}"))
                pairs_from = producer.node_id
            for apply in join_applies.get(spec["written_pos"], ()):
                # the function's pairs reach the join on their own port
                aid = f"apply:{apply.function}"
                nodes.append(Foreign(
                    node_id=aid,
                    inputs=input_ports(tuple(
                        ids_src[a] for a in apply.aliases)),
                    function=apply.function, kind=apply.kind, ids="pairs",
                    columns=tuple((ref.alias, ref.column)
                                  for ref in apply.columns),
                    aliases=tuple(apply.aliases),
                    written_pos=spec["written_pos"]))
                pair_inputs.append(PortRef(aid, f"pairs:{spec['written_pos']}"))
                pairs_from = aid
            stage_dicts.append(JoinStage(
                written_pos=spec["written_pos"], exec_idx=exec_idx,
                anchor=anchor, partners=tuple(partners),
                semantics=spec["semantics"],
                selectivity=spec["selectivity"],
                expected_tuples=round(record["tuples"], 1),
                anchor_frame_tokens=spec["frame_tokens"][anchor],
                pair_tail_tokens=spec["tail_tokens"],
                anchor_resident=record["resident"],
                tuple_tokens=round(record["tokens"], 1),
                pairs_from=pairs_from))
            exec_idx += 1
            for a in partners:
                if a not in in_aliases:
                    in_aliases.append(a)
            if spec["semantics"] == "full":
                pairs_edges.append(PortRef(
                    gid, f"join_answers:{spec['written_pos']}"
                ))
                for a in [anchor] + partners:
                    if a not in out_aliases:
                        out_aliases.append(a)
        nodes.append(AiJoin(
            node_id=gid,
            inputs=input_ports(tuple(ids_src[a] for a in in_aliases)
                               + tuple(pair_inputs)),
            anchor=anchor,
            anchor_resident=group[0][1]["resident"],
            keep_anchor_kv=anchor in retention_plan["after"][gid],
            stages=tuple(stage_dicts)))
        ids_src[anchor] = PortRef(gid, f"ids:{anchor}")

    if len(pairs_edges) == 1 and len(seq) == 1:
        # one full join and nothing after it: its true pairs are the
        # result rows, so no recombination is needed
        sink_inputs = (pairs_edges[0],)
    elif pairs_edges:
        nodes.append(Recombine(
            node_id="recombine",
            inputs=input_ports(
                tuple(pairs_edges)
                + tuple(ids_src[a] for a in out_aliases)
            ),
            alias_order=tuple(out_aliases)))
        sink_inputs = (PortRef("recombine", "tuples"),)
    else:
        sink_inputs = (ids_src[scans[0].alias],)
    nodes.append(PhysicalProject(
        node_id="project",
        inputs=input_ports(tuple(sink_inputs)),
        columns=tuple(f"{c.alias}.{c.column}" for c in plan.root.columns)))
    if plan.root.limit is not None:
        nodes.append(Limit(
            node_id="limit",
            inputs=input_ports((PortRef("project", "rows"),)),
            count=plan.root.limit,
        ))

    estimate = speed_of_light(
        base_work + found["work"], model, device, chunk
    ).seconds
    stage_works = {record["written_pos"]: record["work"]
                   for record in found["records"]}

    def estimator(graph):
        return node_estimates(
            graph, filter_works=filter_works, stage_works=stage_works,
            live=live0, stats=stats, pre=pre, cap_pages=cap_pages,
            model=model, device=device, chunk=chunk)

    return PhysicalPlan(
        model=model.name, device=device.name, workers=workers,
        backend="quail", estimated_seconds=estimate,
        nodes=tuple(nodes), remarks=tuple(remarks),
        settings={
            "chunk_tokens": chunk,
            "arena_pages": list(arena_split),
            "admission_tokens": admission,
            "retention": retention_plan,
            "order_rule": rule,
            "order_source": source,
            "search_seconds": estimate,
        },
        estimator=estimator)


def plan_query(plan: LogicalPlan, *, model: ModelSpec,
               device: DeviceSpec, doc_tokens: dict, gpus: int = 1,
               order: str | None = None, backend: str = "quail",
               registry=None, tokenizer=None, pair_fractions=None):
    """Plan one query with the selected model backend.

    Args:
        plan: The logical plan to plan.
        model: Model spec.
        device: Device spec.
        doc_tokens: Per document token counts for each table alias.
        gpus: GPU count handed to the backend as gpu_count.
        order: Stage order rule, 'by_cost' or 'as_written'; None picks
            the default rule.
        backend: Registered model backend name.
        registry: Optional session extension registry.
        tokenizer: Optional callable (text -> token list) handed to the
            planning context.
        pair_fractions: join written position -> the fraction of the
            cross product its equality conditions keep.
    """
    if registry is None:
        # the built in registry imports every backend, and backends
        # import this planner; build it only when no session gave one
        from quail.builtins import built_in_registry
        registry = built_in_registry()
    try:
        selected = registry.backend(backend)
    except ValueError as error:
        return Refusal(
            reasons=(str(error),),
            constraint="unknown_backend",
            needed=1,
            available=0,
            unit="backends",
        )
    support = selected.supports(model, device, gpus)
    if not support.supported:
        return Refusal(
            reasons=(support.reason or "unsupported backend configuration",),
            constraint="unsupported_backend_configuration",
            needed=1,
            available=0,
            unit="configurations",
        )

    context = PlanningContext(
        model=model,
        device=device,
        gpu_count=gpus,
        document_tokens=doc_tokens,
        backend=backend,
        order=order,
        tokenizer=tokenizer,
        pair_fractions=dict(pair_fractions or {}),
    )
    region = ModelRegion(plan)
    candidates = tuple(selected.plan(region, context))
    for physical_planner in registry.physical_planners.values():
        candidates += tuple(physical_planner.plan(region, context))
    if not candidates:
        return Refusal(
            reasons=(f"backend {backend!r} could not produce a plan",),
            constraint="no_physical_plan",
            needed=1,
            available=0,
            unit="plans",
        )
    selected_candidate = min(
        candidates,
        key=lambda candidate: candidate.estimated_seconds,
    )
    selected_plan = selected_candidate.plan
    if isinstance(selected_plan, Refusal):
        return selected_plan
    if selected_plan.backend != backend:
        raise ValueError(
            f"physical planner returned backend {selected_plan.backend!r} "
            f"for selected backend {backend!r}")
    return _apply_rules(selected_plan, registry, context, "")


def refine_plan(plan, *, model: ModelSpec, device: DeviceSpec,
                doc_tokens: dict, gpus: int = 1, backend: str = "quail",
                registry=None, order: str | None = None,
                tokenizer=None, pair_fractions=None):
    """Run the physical rules again over a plan once its inputs are exact.

    A plan made on estimated document lengths never saw the token
    stores, which some rules read (prefix_sharing measures the shared
    prefixes of a corpus). Takes the same inputs as plan_query and
    returns the plan with any rule's rewrite applied.
    """
    if isinstance(plan, Refusal):
        return plan
    if registry is None:
        from quail.builtins import built_in_registry
        registry = built_in_registry()
    context = PlanningContext(
        model=model,
        device=device,
        gpu_count=gpus,
        document_tokens=doc_tokens,
        backend=backend,
        order=order,
        tokenizer=tokenizer,
        pair_fractions=dict(pair_fractions or {}),
    )
    return _apply_rules(plan, registry, context,
                        " once the documents were tokenized")


def _apply_rules(plan, registry, context, when: str):
    """Apply the registered physical rules; a remark names each that fired."""
    graph, changed = apply_physical_rules(
        plan.graph, tuple(registry.physical_rules.values()), context)
    if not changed:
        return plan
    return replace(
        plan, nodes=graph.nodes, root=graph.root,
        remarks=plan.remarks + tuple(
            f"physical rule {name} changed the plan{when}" for name in changed))


def _filter_alias(pred_or_list):
    p = pred_or_list[0] if isinstance(pred_or_list, list) else pred_or_list
    return p.prompt.args[0].alias


# ------------------------------------------------------------- explain

def explain(logical: LogicalPlan, physical, *, verbose: bool = False,
            result=None, usd_per_hour: float | None = None) -> str:
    """Format the logical and physical operator trees.

    Args:
        logical: The optimized logical plan.
        physical: The physical plan or planning refusal.
        verbose: Include runtime settings and internal node fields.
        result: The QueryResult of running the plan. When given, each
            node shows its measured rows, time, and tokens next to the
            estimates, and the measured totals follow the tree.
        usd_per_hour: Price of one GPU, for the measured cost per query.
    """
    from quail.explain import (
        _fields,
        logical_tree,
        measured_stages,
        physical_tree,
        run_summary,
    )

    lines = ["logical:"]
    lines.extend("  " + line for line in logical_tree(logical).splitlines())
    if isinstance(physical, Refusal):
        lines.append(f"refusal: {physical.constraint}: needed "
                     f"{physical.needed} {physical.unit}, available "
                     f"{physical.available}")
        lines.extend(f"  {reason}" for reason in physical.reasons)
        return "\n".join(lines)
    lines.append("")
    lines.append(f"physical: backend={physical.backend}, "
                 f"model={physical.model}, workers={physical.workers}")
    if physical.backend == "quail":
        chunk = physical.settings.get("chunk_tokens")
        admission = physical.settings.get("admission_tokens")
        budgets = ["KV=bf16"]
        if chunk is not None:
            budgets.append(f"chunk budget={chunk:,} tokens")
        if admission is not None:
            budgets.append(f"admission budget={admission:,} tokens")
        lines.append("  " + ", ".join(budgets))
    lines.append("")
    lines.extend("  " + line for line in physical_tree(
        physical.graph, logical=logical, verbose=verbose,
        estimates=getattr(physical, "estimates", None),
        metrics=None if result is None else result.node_metrics,
        stages=None if result is None else measured_stages(result.report),
    ).splitlines())
    if getattr(physical, "estimates", None):
        lines.append("  est. time is each node's work alone; node times do "
                     "not add up to the plan estimate")
        lines.append("  because chunk packing shares forward passes across "
                     "nodes")
    if result is not None:
        lines.append("")
        lines.append("run:")
        lines.extend("  " + line for line in run_summary(
            result.report, physical.graph, physical.workers, usd_per_hour))
    if verbose:
        lines.append("")
        lines.append("settings:")
        lines.extend(_fields({"model": physical.model,
                              "device": physical.device,
                              **physical.settings}, 1))
        if physical.backend == "quail":
            lines.append("  KV dtype=bf16")
        lines.extend(f"  remark: {remark}" for remark in physical.remarks)
    return "\n".join(lines)
