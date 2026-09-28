"""Shared prefix tokens of a token store, for the speed of light estimate."""

from __future__ import annotations

from quail.cost.budgets import PAGE_TOKENS
from quail.execution.tokens import PrefixTree, prefix_tree

# store path -> per document shared prefix tokens; a corpus is measured once
# per process however many queries scan it
_document_shared_cache: dict[str, list[int]] = {}
_page_tree_cache: dict[str, PrefixTree] = {}


def _documents(store):
    """Return the token documents of a token store or a scan input."""
    return getattr(store, "tokens", store)


def document_shared_tokens(store) -> list[int]:
    """Return, per document, the prefix tokens an earlier document also has."""
    documents = _documents(store)
    path = getattr(documents, "path", None)
    if path is not None and path in _document_shared_cache:
        return _document_shared_cache[path]
    shared = prefix_tree(documents, 1).shared
    if path is not None:
        _document_shared_cache[path] = shared
    return shared


def page_tree(store) -> PrefixTree:
    """Return the store's prefix tree in whole KV pages, cached per path."""
    documents = _documents(store)
    path = getattr(documents, "path", None)
    if path is not None and path in _page_tree_cache:
        return _page_tree_cache[path]
    tree = prefix_tree(documents, PAGE_TOKENS)
    if path is not None:
        _page_tree_cache[path] = tree
    return tree


def shared_prefix_tokens(store) -> int:
    """Return the prefix tokens a token store's documents share."""
    return sum(document_shared_tokens(store))
