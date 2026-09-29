"""Serial constrained generation over Quail's paged Qwen3 forward pass."""

from importlib.metadata import version

from quail.backends.quail.executor.loop import ArenaFullError, _forward, pack_chunk
from quail.backends.quail.executor.model import resolve_model_path
from quail.execution.extract import GenerationResult

ANY_WHITESPACE = True


class PagedSequence:
    """Own private KV and track the positions initialized by successful forwards."""

    def __init__(self, torch, arena, pipeline, chunk_tokens, capacity):
        if chunk_tokens <= 0 or capacity <= 0:
            raise ValueError("Generation chunk size and capacity must be positive")
        if pipeline.canvas_ids or arena.has_sliding:
            raise ValueError("Serial generation requires full-attention Qwen3")
        self.torch, self.arena, self.pipeline = torch, arena, pipeline
        self.chunk_tokens = min(chunk_tokens, pipeline.max_chunk_tokens or chunk_tokens)
        self.capacity = capacity
        self.initialized = 0
        self.closed = False
        self.key = object()
        if arena.activate(self.key, 0, capacity_tokens=capacity) is None:
            raise ArenaFullError("Insufficient private KV for generation")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        """Release the sequence's reserved KV."""
        if self.arena.is_resident(self.key):
            self.arena.free_key(self.key)
        self.closed = True

    def forward(self, tokens):
        """Append fresh tokens and return the final position's hidden state."""
        if self.closed:
            raise RuntimeError("The generation sequence is closed")
        if not tokens or self.initialized + len(tokens) > self.capacity:
            raise ValueError("Generation tokens exceed the reserved sequence")
        for start in range(0, len(tokens), self.chunk_tokens):
            fresh = tokens[start:start + self.chunk_tokens]
            end = self.initialized + len(fresh)
            if self.arena.activate(
                    self.key, end, capacity_tokens=self.capacity) is None:
                raise ArenaFullError("Generation lost its reserved KV capacity")
            chunk = pack_chunk(self.torch, self.arena, [{
                "key": self.key, "prefix": None,
                "f": self.initialized, "suffixes": [fresh],
            }], attention_mode="unified")
            if chunk.temporary_keys:
                for key in chunk.temporary_keys:
                    self.arena.free_key(key)
                raise RuntimeError("Generation must write directly to its private KV")
            hidden = _forward(self.pipeline, self.arena, chunk)
            # activate updates accounting before forwarding; it is not this watermark.
            self.initialized = end
        return hidden


def generation_logits(model, hidden):
    """Project one position through the loaded model's full output head."""
    if not model.quail_generation or model.lm_head is None:
        raise ValueError("Generation requires a full output head")
    logits = model.compute_logits(hidden)
    if logits is None or tuple(logits.shape) != (1, model.config.vocab_size):
        raise RuntimeError(
            "Generation requires one logit row over the model vocabulary")
    return logits.float()


class QuailGenerator:
    """Generate one structured response at a time with private per-row KV."""

    def __init__(self, torch, model, arena, pipeline, spec, chunk_tokens):
        import xgrammar as xgr
        from transformers import AutoTokenizer
        from vllm import envs

        if spec.arch != "qwen3" or spec.role != "generative":
            raise ValueError("Native generation supports generative Qwen3")
        if not model.quail_generation:
            raise ValueError("Generation requires a full output head")
        self.torch, self.model = torch, model
        self.arena, self.pipeline = arena, pipeline
        self.chunk_tokens = chunk_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolve_model_path(spec.hf_name, spec.revision), local_files_only=True)
        self.vocab_size = model.config.vocab_size
        info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer, vocab_size=self.vocab_size)
        self.compiler = xgr.GrammarCompiler(
            info, max_threads=8, cache_enabled=True,
            cache_limit_bytes=envs.VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024)
        self.bitmask = xgr.allocate_token_bitmask(1, self.vocab_size)
        self.gpu_bitmask = torch.empty_like(self.bitmask, device="cuda")
        self.context_limit = model.quail_vllm_config.model_config.max_model_len
        self.metadata = {
            "model_revision": spec.revision, "tokenizer_revision": spec.revision,
            "extraction_add_special_tokens": False,
            "structured_outputs_backend": "xgrammar",
            "structured_outputs_disable_any_whitespace": not ANY_WHITESPACE,
            "xgrammar_version": version("xgrammar"),
            "stop_token_ids": list(info.stop_token_ids),
            "vocab_size": self.vocab_size,
        }

    def generate_structured(self, prompts, schema, max_tokens):
        """Return constrained completions in input order."""
        import xgrammar as xgr

        if max_tokens <= 0:
            raise ValueError("Extraction output limit must be positive")
        if not prompts:
            return []
        compiled = self.compiler.compile_json_schema(
            schema, any_whitespace=ANY_WHITESPACE)
        context_limit = min(
            self.context_limit, self.arena.n_pages * self.arena.page_tokens)
        outputs = []
        with self.torch.inference_mode():
            for prompt in prompts:
                tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
                if not tokens or len(tokens) + max_tokens > context_limit:
                    outputs.append(GenerationResult("", "context", 0))
                    continue
                matcher = xgr.GrammarMatcher(compiled)
                outputs.append(self._generate(tokens, matcher, max_tokens))
        return outputs

    def _next_token(self, hidden, matcher):
        import xgrammar as xgr

        logits = generation_logits(self.model, hidden)
        if matcher.fill_next_token_bitmask(self.bitmask):
            self.gpu_bitmask.copy_(self.bitmask)
            xgr.apply_token_bitmask_inplace(logits, self.gpu_bitmask)
        score, token = logits[0].max(dim=0)
        if not self.torch.isfinite(score).item():
            raise RuntimeError("Generation has no finite allowed token score")
        token = int(token.item())
        if not matcher.accept_token(token):
            raise RuntimeError("Generated token was rejected by the grammar")
        return token

    def _generate(self, prompt_ids, matcher, max_tokens):
        generated = []
        reason = "length"
        with PagedSequence(self.torch, self.arena, self.pipeline,
                           self.chunk_tokens, len(prompt_ids) + max_tokens) as sequence:
            hidden = sequence.forward(prompt_ids)
            for index in range(max_tokens):
                token = self._next_token(hidden, matcher)
                generated.append(token)
                if matcher.is_terminated():
                    reason = "stop"
                    break
                if index + 1 < max_tokens:
                    hidden = sequence.forward([token])
        text = self.tokenizer.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return GenerationResult(
            text, reason, len(prompt_ids), output_tokens=len(generated))


def warm_generation(torch, model, arena, pipeline):
    """Warm a paged one-token forward, the vocabulary head, and its mask kernel."""
    import xgrammar as xgr

    with PagedSequence(torch, arena, pipeline, 17, 32) as sequence:
        sequence.forward([10] * 17)
        logits = generation_logits(model, sequence.forward([11]))
        bitmask = xgr.allocate_token_bitmask(1, model.config.vocab_size)
        bitmask.fill_(-1)
        xgr.apply_token_bitmask_inplace(logits, bitmask.to(device=logits.device))
