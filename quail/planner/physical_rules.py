"""Built in physical optimizer rules.

- prefix_sharing: documents that share a token prefix with another
  document borrow its KV pages for the shared part instead of
  computing it again.
- tree_attention: each join is annotated with the attention path the
  cost model prefers for its partners' reads of the anchor's KV.
"""

from __future__ import annotations

from dataclasses import replace

from quail.cost.budgets import choose_attention_path
from quail.physical import AiFilter, AiJoin, PhysicalGraph
from quail.planner.prefixes import page_tree


def _token_store(document_tokens):
    """The token store behind an alias's document lengths, or None."""
    return getattr(document_tokens, "_store", None)


def page_aligned_shared_tokens(store) -> int:
    """Tokens a prefix tree over the store saves in whole KV pages."""
    return page_tree(store).shared_tokens


def filter_attention(model, device, store, lengths, question_tokens: int) -> str:
    """The attention path for a filter whose documents share prefixes.

    Documents borrowing the same pages of one parent are that node's
    readers; each reads with its own tokens plus the question. The
    path that wins on the most borrowed tokens wins the filter.
    """
    tree = page_tree(store)
    groups = {}
    for doc, parent in enumerate(tree.parent):
        if parent is None:
            continue
        readers, rows = groups.get((parent, tree.shared[doc]), (0, 0))
        groups[(parent, tree.shared[doc])] = (
            readers + 1, rows + lengths[doc] - tree.shared[doc])
    votes = {"tree": 0, "unified": 0}
    for (_, shared), (readers, rows) in groups.items():
        path = choose_attention_path(
            model, device, readers=readers,
            reader_rows=rows / readers + question_tokens, node_tokens=shared)
        votes[path] += readers * shared
    return "tree" if votes["tree"] > votes["unified"] else "unified"


def sharing_pays(model, device, *, shared_tokens: int, total_tokens: int,
                 writes_pages: bool) -> bool:
    """Whether borrowing shared prefixes saves more than it costs.

    Borrowing saves the forward pass over the shared tokens. For a
    filter that did not write KV pages, it costs writing them for every
    token of the corpus.
    """
    if shared_tokens <= 0:
        return False
    saved = shared_tokens * 2.0 * model.params / device.peak_flops
    cost = (0.0 if writes_pages
            else total_tokens * model.kappa / device.hbm_bw)
    return saved > cost


class PrefixSharing:
    """Let documents borrow the KV pages of a document sharing their prefix.

    Fires for a filter whose documents share whole pages of prefix
    worth more forward-pass time than the page writes the filter takes
    on. The filter then writes pages even when it has one stage, since
    borrowed pages must exist. A join always writes its anchors' KV,
    so it fires for any join whose anchors share a whole page.
    """

    name = "prefix_sharing"

    def rewrite(self, graph: PhysicalGraph, context) -> PhysicalGraph | None:
        if context is None:
            return None
        nodes = []
        changed = False
        for node in graph.nodes:
            if isinstance(node, AiFilter) and not node.share_prefixes:
                lengths = context.document_tokens.get(node.alias)
                store = _token_store(lengths)
                if store is not None and sharing_pays(
                        context.model, context.device,
                        shared_tokens=page_aligned_shared_tokens(store),
                        total_tokens=sum(lengths),
                        writes_pages=node.arena_writes):
                    node = replace(node, share_prefixes=True,
                                   arena_writes=True)
                    changed = True
            elif isinstance(node, AiJoin) and not node.share_prefixes:
                lengths = context.document_tokens.get(node.anchor)
                store = _token_store(lengths)
                if store is not None and sharing_pays(
                        context.model, context.device,
                        shared_tokens=page_aligned_shared_tokens(store),
                        total_tokens=sum(lengths), writes_pages=True):
                    node = replace(node, share_prefixes=True)
                    changed = True
            nodes.append(node)
        if not changed:
            return None
        return PhysicalGraph(tuple(nodes), graph.root)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


class TreeAttention:
    """Annotate each join with the attention path the cost model prefers.

    A join stage has every partner of an anchor reading the anchor's
    KV: under tree attention those reads are stacked into one, under
    unified attention each partner reads the anchor itself. The choice
    is by roofline (choose_attention_path). Tree attention needs the
    fp8 merge kernel and packs no canvas rows, so other models stay
    unified.

    A filter sharing prefixes is annotated from its tree: the
    documents borrowing one parent's pages are that node's readers
    (filter_attention). A filter that does not share has one reader
    per node and stays on the unified path.

    The choice is recomputed on every pass, so a plan first made on
    estimated lengths gets the path its exact tokens call for.
    """

    name = "tree_attention"

    def rewrite(self, graph: PhysicalGraph, context) -> PhysicalGraph | None:
        if context is None:
            return None
        model, device = context.model, context.device
        tree_available = (model.weight_precision == "fp8"
                          and not model.canvas_tokens)
        nodes = []
        changed = False
        for node in graph.nodes:
            path = None
            if isinstance(node, AiFilter):
                lengths = context.document_tokens.get(node.alias, ())
                store = _token_store(lengths)
                path = "unified"
                if tree_available and node.share_prefixes and store is not None:
                    path = filter_attention(
                        model, device, store, lengths,
                        node.stages[0].question_tokens if node.stages else 0)
            elif isinstance(node, AiJoin):
                lengths = context.document_tokens.get(node.anchor, ())
                path = "unified"
                if tree_available and node.stages:
                    stage = node.stages[0]
                    readers = (stage.expected_tuples / len(lengths)
                               if len(lengths) else 1.0)
                    # a partner's rows: its document and the question tail
                    rows = stage.pair_tail_tokens + sum(
                        _mean(context.document_tokens.get(partner, ()))
                        for partner in stage.partners)
                    path = choose_attention_path(
                        model, device, readers=readers, reader_rows=rows,
                        node_tokens=_mean(lengths))
            if path is not None and path != node.attention:
                node = replace(node, attention=path)
                changed = True
            nodes.append(node)
        if not changed:
            return None
        return PhysicalGraph(tuple(nodes), graph.root)


def built_in_physical_rules() -> tuple:
    """Return the physical rules registered with the built in registry."""
    return (PrefixSharing(), TreeAttention())
