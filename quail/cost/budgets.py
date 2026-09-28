"""Budget quantities derived from model and device specs.

Chunk budget, admission budget, and roofline arithmetic.
"""

from quail.specs import DeviceSpec, ModelSpec

POOL_FRACTION = 0.95    # fraction of device memory the executor claims
CHUNK_SLACK = 2         # slack factor on the activation bound
PAGE_TOKENS = 16        # KV arena page size, tokens
INT32_MAX = 2**31 - 1
ACT_BYTES = 2           # bf16 activations, bytes per element
ACT_RESERVE_CHUNKS = 2  # chunks of activation memory reserved outside
#                         the arena for overlapped chunk construction


def minimum_weight_gpus(model: ModelSpec, device: DeviceSpec) -> int:
    """Return how many pooled GPU memories would hold the loaded weights.

    Quail does not split weights across GPUs. A result above one causes a
    planning refusal.
    """
    gpus = 1
    while model.W_mem > device.mem_bytes * POOL_FRACTION * gpus:
        gpus *= 2
    return gpus


def kernel_index_cap(model: ModelSpec) -> int:
    """Max tokens per chunk from the int32 element-offset limit.

    Constraint: rows x widest projection < 2^31.
    """
    return INT32_MAX // model.widest_projection


def chunk_memory_bound(model: ModelSpec, device: DeviceSpec) -> int:
    """Tokens per chunk the activation memory allows, with slack.

    Uses resident weights after the full untied output head is discarded.
    """
    free = device.mem_bytes * POOL_FRACTION - model.W_resident
    return int(free // model.act_per_token) // CHUNK_SLACK


def chunk_budget(model: ModelSpec, device: DeviceSpec) -> int:
    """Effective chunk budget, floored at the compute knee.

    The budget is min(memory bound, kernel index cap, the spec's own
    cap when it has one) before the floor.
    """
    b = min(chunk_memory_bound(model, device), kernel_index_cap(model))
    if model.chunk_cap_tokens:
        b = min(b, model.chunk_cap_tokens)
    return max(b, int(compute_knee(model, device)))


def arena_bytes(model: ModelSpec, device: DeviceSpec,
                chunk_tokens: int) -> float:
    """Bytes left for KV after resident weights and the activation reserve."""
    return (device.mem_bytes * POOL_FRACTION - model.W_resident
            - ACT_RESERVE_CHUNKS * chunk_tokens * model.act_per_token)


# Rows past a document that its pages also cover: the shared question
# preamble and a stage tail, taken as a round number for the split.
SPLIT_EXTRA_TOKENS = 64


def transient_sliding_pages(chunk_tokens: int, window: int) -> int:
    """Sliding-layer pages one chunk's fresh prefixes take before trimming.

    The chunk's rows, plus one page of rounding per document longer
    than the window, of which a chunk holds at most chunk / window.
    """
    return -(-chunk_tokens // PAGE_TOKENS) + -(-chunk_tokens // window)


def arena_pages(model: ModelSpec, device: DeviceSpec,
                chunk_tokens: int | None = None,
                mean_doc_tokens: float | None = None) -> tuple[int, int]:
    """Pages of the two KV pools: (every-token pool, sliding-layer pool).

    A model without sliding layers gets one pool and 0 sliding pages.
    With sliding layers, a document holds its whole prefix on the
    full-attention layers and only the last window on the sliding
    layers, so the sliding pool is sized to the fraction of a mean
    document that the window keeps. It never drops below the pages one
    chunk's fresh prefixes take before they are trimmed. An unknown
    mean sizes it as if every document fit in the window, the largest
    fraction.
    """
    if chunk_tokens is None:
        chunk_tokens = chunk_budget(model, device)
    free = arena_bytes(model, device, chunk_tokens)
    page_bytes_full = model.kappa_full * PAGE_TOKENS
    page_bytes_sliding = model.kappa_sliding * PAGE_TOKENS
    if not page_bytes_sliding:
        return int(free // page_bytes_full), 0
    window = model.sliding_window
    mean = window if mean_doc_tokens is None else max(1.0, mean_doc_tokens)
    ratio = min(1.0, (min(mean, window) + SPLIT_EXTRA_TOKENS)
                / (mean + SPLIT_EXTRA_TOKENS))
    full = int(free // (page_bytes_full + ratio * page_bytes_sliding))
    sliding = int(full * ratio)
    floor = transient_sliding_pages(chunk_tokens, window)
    if sliding < floor:
        sliding = floor
        full = int((free - sliding * page_bytes_sliding) // page_bytes_full)
    if full <= 0:
        raise ValueError("the KV arena cannot hold one chunk's sliding KV")
    return full, sliding


def arena_tokens(model: ModelSpec, device: DeviceSpec,
                 chunk_tokens: int | None = None,
                 mean_doc_tokens: float | None = None) -> int:
    """Admission budget: tokens of document KV that can be resident at once.

    Computed from the memory left after resident weights and the
    activation reservation. With sliding layers this is the
    every-token pool; see arena_pages.
    """
    full, _ = arena_pages(model, device, chunk_tokens, mean_doc_tokens)
    return full * PAGE_TOKENS


# ---- roofline arithmetic

def _projection_shapes(model: ModelSpec):
    """Return (in_dim, out_dim) of every dense projection in a layer."""
    qkv_out = (model.n_q + 2 * model.n_kv) * model.d_head
    inter = model.intermediate
    return ((model.hidden, qkv_out),
            (model.n_q * model.d_head, model.hidden),
            (model.hidden, model.ffn_width),
            (inter, model.hidden))


def compute_knee(model: ModelSpec, device: DeviceSpec) -> float:
    """Chunk size (tokens) where the dense projections become compute-bound.

    That is where they cross the roofline ridge.
    """
    ridge = device.peak_flops / device.hbm_bw
    tot_p = tot_io = 0.0
    for din, dout in _projection_shapes(model):
        tot_p += din * dout
        tot_io += din + dout
    denom = 2.0 * tot_p - ridge * tot_io * ACT_BYTES
    if denom <= 0:
        raise ValueError("projections never cross the ridge")
    return ridge * tot_p * model.w_bytes / denom


def _projection_time(model: ModelSpec, device: DeviceSpec,
                     chunk: int) -> float:
    """Ideal seconds for all dense projections in one chunk, all layers."""
    t = 0.0
    for din, dout in _projection_shapes(model):
        params = din * dout
        flops = 2.0 * params * chunk
        moved = (params * model.w_bytes
                 + chunk * (din + dout) * ACT_BYTES)
        t += max(flops / device.peak_flops, moved / device.hbm_bw)
    return t * model.layers


def _attention_time(model: ModelSpec, device: DeviceSpec,
                    chunk: int, context: int) -> float:
    """Ideal seconds for the attention kernels in one chunk, all layers."""
    flops = 4.0 * chunk * context * model.n_q * model.d_head
    moved = (context * model.kappa / model.layers
             + 2.0 * chunk * model.n_q * model.d_head * ACT_BYTES)
    return max(flops / device.peak_flops,
               moved / device.hbm_bw) * model.layers


def choose_attention_path(model: ModelSpec, device: DeviceSpec, *,
                          readers: float, reader_rows: float,
                          node_tokens: float) -> str:
    """Pick "unified" or "tree" attention for readers of one shared node.

    The node is a join's anchor or a filter's parent document; its
    readers are the partners or the children borrowing its pages.

    Under unified attention every reader reads the node's KV itself.
    Under tree attention the readers' rows are stacked into one read of
    the node, then merged with each reader's attention over its own
    rows. Both times take the larger of compute and memory, so tree
    wins only for readers that are memory-bound on the node: few rows
    each, many of them.

    Args:
        model: The model whose attention runs.
        device: The device it runs on.
        readers: Readers sharing the node in one forward pass.
        reader_rows: Query rows per reader.
        node_tokens: KV tokens of the shared node.
    """
    if readers <= 1 or reader_rows <= 0 or node_tokens <= 0:
        return "unified"
    unified = readers * _attention_time(model, device, reader_rows,
                                        node_tokens)
    rows = readers * reader_rows
    tree = _attention_time(model, device, rows, node_tokens)
    # the merge kernel reads two partial outputs and writes one, in
    # bf16, once per layer
    merge = 3 * rows * model.n_q * model.d_head * ACT_BYTES / device.hbm_bw
    tree += merge * model.layers
    return "tree" if tree < unified else "unified"


def attention_crossover(model: ModelSpec, device: DeviceSpec,
                        chunk_tokens: int | None = None) -> float:
    """Document length (tokens) where attention overtakes dense projections.

    At the given chunk size.
    """
    if chunk_tokens is None:
        chunk_tokens = chunk_budget(model, device)
    t_dense = _projection_time(model, device, chunk_tokens)
    lo, hi = 1.0, 1e7
    for _ in range(200):
        mid = (lo + hi) / 2
        if _attention_time(model, device, chunk_tokens, int(mid)) < t_dense:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def derived_table(model: ModelSpec, device: DeviceSpec) -> dict:
    """Return all derived budget quantities as a dict."""
    chunk = chunk_budget(model, device)
    return {
        "minimum_weight_gpus": minimum_weight_gpus(model, device),
        "arena_tokens": arena_tokens(model, device, chunk),
        "arena_pages": arena_pages(model, device, chunk),
        "chunk_memory_bound": chunk_memory_bound(model, device),
        "kernel_index_cap": kernel_index_cap(model),
        "chunk_budget": chunk,
        "compute_knee": compute_knee(model, device),
        "attention_crossover": attention_crossover(model, device, chunk),
    }
