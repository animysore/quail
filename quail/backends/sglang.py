"""SGLang engine adapter and the pipelined SGLang request backend."""

from __future__ import annotations

import asyncio
import time
import uuid

from quail.backends.request import RequestBackend
from quail.backends.request_scheduling import (
    MAX_BATCHED_TOKENS,
    MAX_SEQUENCES,
    run_filter_chain_async,
)

PAGE_SIZE = 16
CHUNKED_PREFILL_TOKENS = (
    MAX_BATCHED_TOKENS // PAGE_SIZE
) * PAGE_SIZE
MEM_FRACTION_STATIC = 0.76
TRUE_FALSE_LOGIT_BIAS = 1_000.0


class _Completion:
    def __init__(self, token_ids):
        self.token_ids = token_ids


class _RequestOutput:
    """The vLLM output shape the shared scheduling code reads."""

    def __init__(self, prompt_token_ids, token_ids, cached_tokens):
        self.outputs = [_Completion(token_ids)]
        self.prompt_token_ids = prompt_token_ids
        self.num_cached_tokens = cached_tokens


class SGLangClient:
    """The request operations the backends need from one SGLang engine."""

    def __init__(self, engine, capacity):
        self.engine = engine
        self.capacity = capacity

    def generate(self, prompts, sampling_params, use_tqdm=False):
        del use_tqdm
        if not prompts:
            return []
        input_ids = [prompt["prompt_token_ids"] for prompt in prompts]
        raw = self.engine.generate(
            input_ids=input_ids,
            sampling_params=dict(sampling_params),
        )
        if isinstance(raw, dict):
            raw = [raw]
        if len(raw) != len(input_ids):
            raise RuntimeError(
                f"SGLang returned {len(raw)} outputs for "
                f"{len(input_ids)} requests"
            )
        outputs = []
        for prompt, result in zip(input_ids, raw):
            metadata = result["meta_info"]
            outputs.append(_RequestOutput(
                prompt,
                list(result.get("output_ids") or []),
                int(metadata.get("cached_tokens") or 0),
            ))
        return outputs

    async def _generate_one(self, input_ids, sampling_params):
        request_id = uuid.uuid4().hex
        try:
            result = await self.engine.async_generate(
                input_ids=input_ids, sampling_params=dict(sampling_params),
                rid=request_id,
            )
        except asyncio.CancelledError:
            self.engine.tokenizer_manager.abort_request(request_id)
            raise
        return _RequestOutput(
            input_ids,
            list(result.get("output_ids") or []),
            int(result["meta_info"].get("cached_tokens") or 0),
        )

    def run_filter_chain(self, sampling_params, body_ids, question_ids,
                         read_answer, *, tag="q"):
        """Submit each document's next filter as soon as it passes."""
        del tag
        return self.engine.loop.run_until_complete(run_filter_chain_async(
            self._generate_one,
            sampling_params,
            body_ids,
            question_ids,
            self.capacity["kv_cache_size_tokens"],
            read_answer=read_answer,
            block_size=self.capacity["block_size"],
            max_num_seqs=self.capacity["max_num_seqs"],
        ))

    def reset_prefix_cache(self):
        result = self.engine.flush_cache()
        if isinstance(result, bool):
            return result
        return bool(getattr(result, "success", True))

    def close(self):
        """Stop the engine before another model uses the process's GPU."""
        self.engine.shutdown()


def _capacity(engine) -> dict:
    info = engine.get_server_info()
    tokens = info.get("max_total_num_tokens")
    page_size = info.get("page_size")
    if tokens is None or page_size is None:
        raise RuntimeError("SGLang did not report its KV capacity")
    running = info.get("max_running_requests")
    return {
        "kv_cache_size_tokens": int(tokens),
        "num_gpu_blocks": None,
        "block_size": int(page_size),
        "max_num_seqs": None if running is None else int(running),
        "kv_cache_dtype": str(info.get("kv_cache_dtype")),
    }


class SGLangEngine:
    """Boot SGLang for a request backend."""

    kind = "sglang"
    label = "SGLang"
    runtime_package = "sglang==0.5.18"

    def boot(self, spec, allowed_ids: list[int]) -> tuple[dict, dict]:
        """Load the spec's model and return the engine state and boot record."""
        import sglang as sgl

        started = time.perf_counter()
        engine = sgl.Engine(
            model_path=spec.hf_name,
            mem_fraction_static=MEM_FRACTION_STATIC,
            max_running_requests=MAX_SEQUENCES,
            chunked_prefill_size=CHUNKED_PREFILL_TOKENS,
            max_prefill_tokens=CHUNKED_PREFILL_TOKENS,
            page_size=PAGE_SIZE,
            skip_tokenizer_init=True,
            disable_radix_cache=False,
            disable_prefill_cuda_graph=True,
            log_level="warning",
        )
        boot_s = time.perf_counter() - started
        capacity = _capacity(engine)
        client = SGLangClient(engine, capacity)
        sampling_params = {
            "temperature": 0.0,
            "max_new_tokens": 1,
            "logit_bias": {
                str(token): TRUE_FALSE_LOGIT_BIAS for token in allowed_ids
            },
        }
        client.generate(
            [{"prompt_token_ids": allowed_ids}],
            sampling_params,
        )
        return (
            {
                "client": client,
                "sampling_params": sampling_params,
                "capacity": capacity,
                "close": getattr(engine, "shutdown", None),
            },
            {
                "kind": "cold",
                "llm_init_s": round(boot_s, 2),
                "weight_load_s": None,
                "kv_profile_s": None,
                "boot_s": round(boot_s, 2),
            },
        )


def pipelined_sglang_backend() -> RequestBackend:
    """Return SGLang with per-document filter pipelining."""
    return RequestBackend(
        name="pipelined_sglang",
        engine=SGLangEngine(),
        filter_submission="pipelined",
        join_submission="suffix-major",
    )
