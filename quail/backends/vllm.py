"""vLLM engine adapter and the vLLM request backends."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

from quail.backends.request import RequestBackend
from quail.backends.request_scheduling import (
    MAX_BATCHED_TOKENS,
    MAX_SEQUENCES,
    run_filter_chain,
)
from quail.execution.extract import GenerationResult

GPU_MEMORY_UTILIZATION = 0.91
CUDA_GRAPH_CAPTURE_SIZE = 8_192
# vLLM caps a diffusion model at 8 sequences per step whenever the
# setting is 128 or more, sized for its 256-row canvas; one row needs
# no cap, so stay just under the trigger.
DIFFUSION_SEQUENCES = 127
# DiffusionGemma ignores logprob_token_ids in vLLM 0.26. Request the full
# vocabulary so TRUE/FALSE scores are returned regardless of their ranks.
DIFFUSION_LOGPROBS = -1
# generated tokens a longer canvas gets; the answer word is read
# from the text
DIFFUSION_TEXT_TOKENS = 16


class GigatokenVLLMTokenizer:
    """Load Gigatoken through vLLM's tokenizer registry."""

    @classmethod
    def from_pretrained(cls, path_or_repo_id, *_args, **kwargs):
        """Return Gigatoken with the Hugging Face tokenizer interface."""
        from gigatoken import Tokenizer

        truncation_side = kwargs.pop("truncation_side", "left")
        tokenizer = Tokenizer(path_or_repo_id).as_hf()
        tokenizer.truncation_side = truncation_side
        vocab = tokenizer.get_vocab()
        tokenizer.max_token_id = max(vocab.values())
        tokenizer.max_chars_per_token = max(len(token) for token in vocab)
        return tokenizer


def register_gigatoken():
    """Register Quail's tokenizer and renderer in each vLLM process."""
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers import TokenizerRegistry

    registered = ("quail.backends.vllm", "GigatokenVLLMTokenizer")
    if TokenizerRegistry.tokenizers.get("gigatoken") != registered:
        TokenizerRegistry.register("gigatoken", *registered)
    renderer = ("vllm.renderers.hf", "HfRenderer")
    if RENDERER_REGISTRY.renderers.get("gigatoken") != renderer:
        RENDERER_REGISTRY.register("gigatoken", *renderer)


def _require_gigatoken_plugin():
    from importlib.metadata import entry_points

    from vllm import envs

    plugins = entry_points(group="vllm.general_plugins", name="quail_gigatoken")
    if not any(plugin.value == "quail.backends.vllm:register_gigatoken"
               for plugin in plugins):
        raise RuntimeError(
            "Gigatoken requires Quail's vLLM plugin. Reinstall quail-engine "
            "from this checkout so its entry-point metadata is current.")
    allowed = envs.VLLM_PLUGINS
    if allowed is not None and "quail_gigatoken" not in allowed:
        raise RuntimeError(
            "Gigatoken requires quail_gigatoken in the VLLM_PLUGINS allowlist.")


@contextmanager
def diffusion_canvas(spec):
    """Initialize a one-token vLLM canvas with Quail's fixed token.

    Yields:
        The initial canvas token IDs, or an empty tuple for other models.
    """
    if spec.canvas_tokens != 1:
        yield ()
        return
    from vllm.model_executor.models import diffusion_gemma

    from quail.backends.quail.executor.models.diffusion_gemma import (
        canvas_token_ids,
    )

    tokens = canvas_token_ids(spec.vocab, spec.canvas_tokens)
    original = diffusion_gemma.DiffusionGemmaRequestStates

    class FixedCanvasStates(original):
        def init_canvas(self, slots):
            if self.canvas_length != 1 or self.vocab_size != spec.vocab:
                raise ValueError("vLLM canvas geometry differs from the model spec")
            self.canvas[slots] = tokens[0]

    # vLLM 0.26 has no public canvas-input setting. Instances retain this
    # subclass after construction; later engines see the original class.
    diffusion_gemma.DiffusionGemmaRequestStates = FixedCanvasStates
    try:
        yield tokens
    finally:
        diffusion_gemma.DiffusionGemmaRequestStates = original


def _capacity(llm) -> dict:
    config = llm.llm_engine.vllm_config
    cache = config.cache_config
    tokens = getattr(cache, "kv_cache_size_tokens", None)
    blocks = getattr(cache, "num_gpu_blocks", None)
    block_size = getattr(cache, "block_size", None)
    if tokens is None and blocks is not None and block_size is not None:
        tokens = blocks * block_size
    if tokens is None or block_size is None:
        raise RuntimeError("vLLM did not report its KV capacity")
    scheduler = config.scheduler_config
    return {
        "kv_cache_size_tokens": int(tokens),
        "num_gpu_blocks": None if blocks is None else int(blocks),
        "block_size": int(block_size),
        "max_num_seqs": int(scheduler.max_num_seqs),
        "max_num_batched_tokens": int(scheduler.max_num_batched_tokens),
        "max_model_len": int(config.model_config.max_model_len),
        "gpu_memory_utilization": float(cache.gpu_memory_utilization),
        "kv_cache_dtype": str(cache.cache_dtype),
    }


def diffusion_kwargs(spec) -> dict:
    """Set the canvas length and denoising limit through the public API."""
    if not spec.canvas_tokens:
        return {}
    diffusion = {"canvas_length": spec.canvas_tokens}
    kwargs = {"diffusion_config": diffusion}
    if spec.canvas_tokens == 1:
        diffusion["max_denoising_steps"] = 1
        # vLLM refuses a request for more logprobs than this
        kwargs["max_logprobs"] = DIFFUSION_LOGPROBS
    return kwargs


def sampling_kwargs(allowed_ids: list[int], canvas_tokens: int = 0) -> dict:
    """Set public sampling options for the model's answer format.

    Diffusion sampling rejects temperature and allowed_token_ids.
    One canvas token uses returned scores; longer canvases use text.
    """
    if canvas_tokens == 1:
        return {"max_tokens": 1, "logprobs": DIFFUSION_LOGPROBS,
                "detokenize": False}
    if canvas_tokens:
        return {"max_tokens": DIFFUSION_TEXT_TOKENS}
    return {"temperature": 0.0, "max_tokens": 1, "min_tokens": 1,
            "allowed_token_ids": allowed_ids, "detokenize": False}


def _extraction_kwargs(spec, extraction: bool) -> dict:
    if not extraction:
        return {}
    return {
        "tokenizer_mode": "auto",
        "revision": spec.revision,
        "tokenizer_revision": spec.revision,
        "structured_outputs_config": {
            "backend": "xgrammar", "disable_any_whitespace": False,
        },
    }


def _generation_result(output) -> GenerationResult:
    """Convert vLLM completion state without exposing it to row assembly."""
    prompt_tokens = len(output.prompt_token_ids or ())
    cached_tokens = int(output.num_cached_tokens or 0)
    output_tokens = sum(len(answer.token_ids) for answer in output.outputs)
    if not output.finished or len(output.outputs) != 1:
        return GenerationResult(
            "", "incomplete", prompt_tokens, cached_tokens, output_tokens,
        )
    answer = output.outputs[0]
    return GenerationResult(
        answer.text, answer.finish_reason, prompt_tokens,
        cached_tokens, output_tokens,
    )


class VLLMClient:
    """The request operations the backends need from one vLLM LLM."""

    accepts_text = True

    def __init__(self, llm, capacity: dict):
        self.llm = llm
        self.capacity = capacity

    def generate(self, prompts, sampling_params, use_tqdm=False):
        return self.llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)

    def generate_structured(self, prompts, schema, max_tokens):
        """Generate JSON using vLLM's public structured-output interface."""
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        if max_tokens <= 0:
            raise ValueError("Extraction output limit must be positive")
        if not prompts:
            return []
        tokenizer = self.llm.get_tokenizer()
        limit = min(
            self.capacity["max_model_len"], self.capacity["kv_cache_size_tokens"],
        )
        positions, inputs = [], []
        results = [GenerationResult("", "context", 0) for _ in prompts]
        for position, prompt in enumerate(prompts):
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            if not tokens or len(tokens) + max_tokens > limit:
                continue
            positions.append(position)
            inputs.append({"prompt_token_ids": tokens})
        if not inputs:
            return results
        params = SamplingParams(
            temperature=0.0, max_tokens=max_tokens,
            structured_outputs=StructuredOutputsParams(json=schema),
        )
        outputs = self.generate(inputs, params, use_tqdm=False)
        if len(outputs) != len(positions):
            raise RuntimeError("vLLM returned the wrong number of extraction rows")
        for position, output in zip(positions, outputs):
            results[position] = _generation_result(output)
        return results

    def reset_prefix_cache(self):
        return self.llm.reset_prefix_cache()

    def close(self):
        """Stop engine workers and release their CUDA and distributed state."""
        if self.llm is None:
            return
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        llm, self.llm = self.llm, None
        try:
            llm.llm_engine.engine_core.shutdown()
        finally:
            del llm
            cleanup_dist_env_and_memory()

    def run_filter_chain(self, sampling_params, body_ids, question_ids,
                         read_answer, *, tag="q", body_texts=None,
                         question_texts=None):
        """Pipeline filter stages through the engine's step loop."""
        return run_filter_chain(
            self.llm.llm_engine,
            sampling_params,
            body_ids,
            question_ids,
            self.capacity["kv_cache_size_tokens"],
            tag=tag,
            read_answer=read_answer,
            block_size=self.capacity["block_size"],
            max_num_seqs=self.capacity["max_num_seqs"],
            body_texts=body_texts,
            question_texts=question_texts,
            render_prompt=self.llm._preprocess_cmpl_one,
        )


class VLLMEngine:
    """Boot vLLM for a request backend, with Quail's tuned engine settings."""

    kind = "vllm"
    label = "vLLM"
    runtime_package = "vllm==0.26.0"
    supports_extraction = True

    def llm_kwargs(self, spec, *, extraction: bool = False) -> dict:
        """Return the LLM constructor arguments beyond the model name.

        A spec with its own chunk cap (a mixture-of-experts model)
        batches at least that many tokens per step. A one-row-canvas
        diffusion model runs more sequences per step than vLLM's
        default for its 256-row canvas.
        """
        batched = max(MAX_BATCHED_TOKENS, spec.chunk_cap_tokens)
        kwargs = {
            "max_num_batched_tokens": batched,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "enable_prefix_caching": True,
            "disable_log_stats": True,
            "tokenizer_mode": "gigatoken",
            "compilation_config": {
                "cudagraph_capture_sizes": [CUDA_GRAPH_CAPTURE_SIZE]
            },
            "max_num_seqs": (DIFFUSION_SEQUENCES if spec.canvas_tokens == 1
                             else MAX_SEQUENCES),
        }
        kwargs.update(diffusion_kwargs(spec))
        kwargs.update(_extraction_kwargs(spec, extraction))
        return kwargs

    def boot(self, spec, allowed_ids: list[int], *,
             extraction: bool = False) -> tuple[dict, dict]:
        """Load the spec's model and return the engine state and boot record."""
        from vllm import LLM, SamplingParams

        load_kwargs = self.llm_kwargs(spec, extraction=extraction)
        if load_kwargs.get("tokenizer_mode") == "gigatoken":
            _require_gigatoken_plugin()
        register_gigatoken()
        if spec.canvas_tokens == 1:
            # vLLM's engine core in its own process returns the canvas
            # row's logprobs unreliably under the step loop on long
            # prompts (agreement with the in-process engine falls from
            # 92 to 73 percent on agent traces,
            # /results/ablations/diffusion_gemma_readout_probe_agent_k0_corpus_mp.json)
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        started = time.perf_counter()
        with diffusion_canvas(spec) as canvas_ids:
            client = VLLMClient(LLM(
                model=spec.hf_name, **load_kwargs,
            ), {})
        boot_s = time.perf_counter() - started
        try:
            sampling_params = SamplingParams(
                **sampling_kwargs(allowed_ids, spec.canvas_tokens))
            capacity = _capacity(client.llm)
            config = client.llm.llm_engine.vllm_config
            capacity.update(
                enable_prefix_caching=config.cache_config.enable_prefix_caching,
                input_tokenizer_mode=config.model_config.tokenizer_mode,
                canvas_length=spec.canvas_tokens,
                initial_canvas_token_ids=list(canvas_ids),
                max_denoising_steps=1 if spec.canvas_tokens == 1 else None,
                logprobs=sampling_params.logprobs,
                extraction=extraction,
            )
            if extraction:
                from importlib.metadata import version

                structured = config.structured_outputs_config
                capacity.update(
                    model_revision=config.model_config.revision,
                    tokenizer_revision=config.model_config.tokenizer_revision,
                    structured_outputs_backend=structured.backend,
                    structured_outputs_disable_any_whitespace=(
                        structured.disable_any_whitespace
                    ),
                    xgrammar_version=version("xgrammar"),
                    extraction_add_special_tokens=False,
                )
            client.capacity = capacity
            client.generate(["TRUE"], sampling_params)
        except Exception as error:
            try:
                client.close()
            except Exception as cleanup_error:
                error.add_note(f"Engine cleanup also failed: {cleanup_error}")
            raise
        return (
            {
                "client": client,
                "sampling_params": sampling_params,
                "capacity": capacity,
            },
            {
                "kind": "cold",
                "llm_init_s": round(boot_s, 2),
                "weight_load_s": None,
                "kv_profile_s": None,
                "boot_s": round(boot_s, 2),
            },
        )


class DefaultVLLMEngine(VLLMEngine):
    """Boot vLLM with every engine setting at its default.

    Filter queries pass the model name and any diffusion canvas settings.
    Extraction also pins model/tokenizer revisions and selects HF tokenization
    and XGrammar. The engine kind differs from `VLLMEngine` so this engine never
    shares a loaded model with the tuned vLLM backends.
    """

    kind = "dumb_vllm"
    label = "vLLM with default settings"
    def llm_kwargs(self, spec, *, extraction: bool = False) -> dict:
        return diffusion_kwargs(spec) | _extraction_kwargs(spec, extraction)


def stock_vllm_backend() -> RequestBackend:
    """Return stock vLLM with operator-at-a-time filter execution."""
    return RequestBackend(
        name="stock_vllm",
        engine=VLLMEngine(),
        filter_submission="operator-at-a-time",
    )


def pipelined_vllm_backend() -> RequestBackend:
    """Return vLLM with per document filter pipelining."""
    return RequestBackend(
        name="pipelined_vllm",
        engine=VLLMEngine(),
        filter_submission="pipelined",
    )


def dumb_vllm_backend() -> RequestBackend:
    """Return vLLM at its default settings, operator-at-a-time filters."""
    return RequestBackend(
        name="dumb_vllm",
        engine=DefaultVLLMEngine(),
        filter_submission="operator-at-a-time",
    )
