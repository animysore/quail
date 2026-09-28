"""Arrow token payloads and shared prefix lengths."""

import pickle

import pyarrow as pa

from quail.execution.tokens import (
    ArrowTokenDocuments,
    TokenStore,
    chain_tokens,
    decode_token_documents,
    prefix_tree,
    shared_prefix_lengths,
)
from quail.planner.prefixes import shared_prefix_tokens


def test_token_views_store_chains_and_shared_prefixes(tmp_path):
    tokens = pa.chunked_array([
        pa.array([[1, 2, 3], [], [4, 5]], type=pa.large_list(pa.int32()))
    ])
    documents = decode_token_documents(tokens)
    assert isinstance(documents, ArrowTokenDocuments)
    assert [list(document) for document in documents] == [[1, 2, 3], [], [4, 5]]

    values = documents._values.to_numpy(zero_copy_only=True)
    view = documents[2].numpy()
    assert view.__array_interface__["data"][0] == (
        values.__array_interface__["data"][0] + 3 * values.itemsize)
    assert documents[2].arrow_array.offset == 3
    assert hasattr(documents[2].arrow_array, "__dlpack__")

    first = documents[0]
    combined = chain_tokens([1, 2], first, [20])
    assert list(combined) == [1, 2, 1, 2, 3, 20]
    assert combined.token_parts[1] is first

    sequences = [[1, 2, 3, 4], [1, 2, 3, 9, 9], [1, 2], [7, 8], [7, 8]]
    shared = shared_prefix_lengths(sequences)
    # 15 tokens in total; the prefix trie has 4 + 2 + 2 = 8 nodes, and
    # the second [7, 8] computes its last token
    assert sum(len(sequence) for sequence in sequences) - sum(shared) == 9
    assert shared_prefix_lengths([]) == []
    assert sum(shared_prefix_lengths(sequences[::-1])) == sum(shared)
    assert shared_prefix_tokens([[1, 2, 3], [1, 2, 4], [9]]) == 2

    schema = pa.schema({"id": pa.string(), "body": pa.string()})
    batches = [
        pa.record_batch([["a", "b"], ["one two", "three"]], schema=schema),
        pa.record_batch([["c"], ["four five six"]], schema=schema),
    ]
    path = tmp_path / "tokens.arrow"
    store = TokenStore.write(str(path), batches, document_column="body",
                             tokenizer=str.split, token_type=pa.string())
    assert path.exists()
    assert list(store.lengths) == [2, 1, 3]
    assert [list(store[index]) for index in range(len(store))] == [
        ["one", "two"], ["three"], ["four", "five", "six"]]
    assert store._reader.schema.names == [
        "__quail_token_ids", "__quail_token_count"]
    store.close()

    schema = pa.schema({"body": pa.string()})
    store = TokenStore.write(
        str(path), [pa.record_batch([["word " * 1000] * 10], schema=schema)],
        document_column="body", tokenizer=str.split, token_type=pa.string())
    encoded = pickle.dumps(store.select(range(10)))
    restored = pickle.loads(encoded)
    assert len(encoded) < 1000
    assert len(restored) == 10
    assert len(restored[0]) == 1000


def test_prefix_tree_borrows_whole_pages_and_siblings_share_one_parent():
    docs = [list(range(40)), list(range(50)), list(range(32)) + [99] * 20,
            [7] * 20, [7] * 10]
    tree = prefix_tree(docs, 16)
    # sorted order: 0..39, 0..49, 0..31+99s, [7]*10, [7]*20
    assert tree.order == [0, 1, 2, 4, 3]
    # doc 1 shares 40 with doc 0; doc 2 shares 32 with doc 1, pages
    # doc 1 itself borrowed from doc 0, so doc 2 reads doc 0. The two
    # runs of 7s share 10, under a page, so neither borrows
    assert tree.parent == [None, 0, 0, None, None]
    assert tree.shared == [0, 32, 32, 0, 0]
    assert tree.shared_tokens == 64
    parents_first = {index: position for position, index in enumerate(tree.order)}
    assert all(parent is None or parents_first[parent] < parents_first[index]
               for index, parent in enumerate(tree.parent))
    assert prefix_tree([], 16).shared_tokens == 0
    # an empty document has no last token to compute, and shares nothing
    tree = prefix_tree([[], [], [5] * 16 + [6]], 16)
    assert tree.parent == [None, None, None]
    assert tree.shared == [0, 0, 0]

    header = list(range(64))
    docs = [header + [1] * 8, header + [2] * 8, header + [3] * 8,
            header + [3] * 8 + [4] * 40, header + [5] * 8]
    tree = prefix_tree(docs, 16)
    # every record borrows the header from the first; doc 3 reads its
    # 64 shared tokens from doc 0 too, as doc 2 owns none of them
    assert tree.parent == [None, 0, 0, 0, 0]
    assert tree.shared == [0, 64, 64, 64, 64]
    # a longer shared run makes a deeper node
    docs = [header + [1] * 32, header + [1] * 32 + [2] * 8,
            header + [1] * 32 + [3] * 8]
    tree = prefix_tree(docs, 16)
    assert tree.parent == [None, 0, 0]
    assert tree.shared == [0, 96, 96]
