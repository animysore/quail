"""Weight loading through vLLM's public surface.

The checkpoint becomes vLLM's processed module - merged qkv and
gate_up, fp8 weights and block scales laid out for DeepGEMM. No
engine, no scheduler, no KV pool. vLLM is a library here (loader and
kernels), nothing more.

After load, TRUE/FALSE output rows are retained for Boolean readout.
Generation-capable loads also keep the full output head.

get_model reads tensor-parallel group objects. Those collectives are
no-ops at world size 1, so this path installs single-rank stubs
instead of starting NCCL or gloo.
"""

import gc
from functools import lru_cache
from pathlib import Path

from quail.backends.quail.executor.moe_configs import write_configs
from quail.progress import say


@lru_cache(maxsize=8)
def resolve_model_path(model_name: str, revision: str | None = None) -> str:
    """Download one model revision and return its local directory."""
    path = Path(model_name).expanduser()
    if path.is_dir():
        return str(path.resolve())
    from huggingface_hub import snapshot_download

    say(f"preparing model files: {model_name}")
    path = snapshot_download(
        repo_id=model_name, revision=revision or None,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"],
    )
    say(f"model files ready: {path}")
    return path


class _SingleRank:
    """The group object get_model reads. Collectives are identity."""

    world_size = 1
    rank = 0
    rank_in_group = 0
    local_rank = 0
    ranks = [0]
    first_rank = 0
    last_rank = 0
    device_index = 0
    cpu_group = None
    device_group = None
    device_communicator = None
    unique_name = "single"
    torch_distributed_backend = None

    def __init__(self, torch):
        self.device = torch.device("cuda:0")

    @property
    def is_first_rank(self):
        return True

    @property
    def is_last_rank(self):
        return True

    def all_reduce(self, x):
        return x

    def all_gather(self, x, dim=-1):
        return x

    def reduce_scatter(self, x, dim=-1):
        return x

    def gather(self, x, dst=0, dim=-1):
        return x

    def broadcast(self, x, src=0):
        return x

    def broadcast_object(self, obj=None, src=0):
        return obj

    def destroy(self):
        return None

    def prepare_communication_buffer_for_model(self, model):
        return None

    def graph_capture(self, graph_capture_context=None):
        from contextlib import nullcontext
        return nullcontext(graph_capture_context)


def _install_single_rank_groups(torch):
    from vllm.distributed import parallel_state as ps
    if ps._TP is not None:
        return
    group = _SingleRank(torch)
    ps._WORLD = group
    ps._INNER_DP_WORLD = group
    ps._TP = group
    ps._PP = group
    ps._DP = group
    ps._PCP = group
    ps._DCP = group
    ps._NODE_COUNT = 1


def retain_answer_head(torch, model, token_ids, *, generation=False):
    """Keep Boolean rows and select whether the full output head is needed."""
    allowed = tuple(sorted(set(token_ids)))
    if not allowed:
        raise ValueError("TRUE/FALSE token ids must not be empty")
    if hasattr(model, "quail_answer_token_ids"):
        answer_weights(model, allowed)
    else:
        weight = model.lm_head.weight
        indices = torch.tensor(allowed, device=weight.device, dtype=torch.long)
        weights = weight.detach().index_select(0, indices).to(dtype=torch.bfloat16)
        del weight
        model.register_buffer("quail_answer_weights", weights, persistent=False)
        model.quail_answer_token_ids = allowed
    if generation and model.lm_head is None:
        if not model.config.tie_word_embeddings:
            raise ValueError("Generation requires reloading the untied output head")
        model.lm_head = model.model.embed_tokens
    elif not generation:
        # lm_head can be embed_tokens itself; drop only this reference.
        model.lm_head = None
        # vLLM's parameter stores a bound weight_loader that owns the head.
        gc.collect()
    model.quail_generation = generation


def answer_weights(model, token_ids):
    """Return the retained rows for a query's answer token ids."""
    if tuple(token_ids) != model.quail_answer_token_ids:
        raise ValueError("Query answer token ids differ from the loaded "
                         "model's retained rows")
    return model.quail_answer_weights


def load_model(model_name: str, revision: str | None = None, *,
               answer_token_ids=None, max_batched_tokens=None,
               moe_backend=None, generation=False):
    """Load weights, Boolean rows, and optionally the full output head.

    max_batched_tokens is the largest chunk the model will see. vLLM's
    fused MoE kernels size their scratch buffers from it; a dense model
    ignores it. moe_backend is passed through as vLLM's moe_backend
    setting (see ModelSpec.moe_backend); None lets vLLM pick.
    """
    import os
    import tempfile

    import torch
    import vllm
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model

    # vLLM's Triton fused MoE reads tuned tile configs from this
    # folder before its own; ours add entries for Quail's chunk sizes
    if "VLLM_TUNED_CONFIG_FOLDER" not in os.environ:
        base = Path(vllm.__file__).parent / "model_executor/layers/fused_moe/configs"
        folder = write_configs(
            Path(tempfile.gettempdir()) / "quail-moe-configs", base)
        os.environ["VLLM_TUNED_CONFIG_FOLDER"] = str(folder)
    model_path = resolve_model_path(model_name, revision)
    args = dict(model=model_path, dtype="auto", enforce_eager=True)
    if max_batched_tokens is not None:
        args["max_num_batched_tokens"] = int(max_batched_tokens)
    if moe_backend is not None:
        args["moe_backend"] = moe_backend
    config = EngineArgs(**args).create_engine_config()
    _install_single_rank_groups(torch)
    with set_current_vllm_config(config):
        model = get_model(vllm_config=config)
    # vLLM's fused MoE kernels read the forward context, which is
    # built from this config
    model.quail_vllm_config = config
    if answer_token_ids is None:
        from transformers import AutoTokenizer

        from quail.logical import true_false_ids

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        true_ids, false_ids = true_false_ids(tokenizer)
        answer_token_ids = true_ids | false_ids
    retain_answer_head(torch, model, answer_token_ids, generation=generation)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return model
