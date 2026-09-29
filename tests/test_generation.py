"""Native serial decoding, KV ownership, and grammar termination."""

import sys
from types import SimpleNamespace

import pytest
from fakes import cpu_staging
from kv_checker import Run, Setup, chain

from quail.backends.quail.executor import generate
from quail.backends.quail.executor.generate import (
    PagedSequence,
    QuailGenerator,
    generation_logits,
)
from quail.execution.extract import GenerationResult

torch = pytest.importorskip("torch")


def checking_run(tokens, budget=16):
    answers = {chain(tokens[:end]): end for end in range(1, len(tokens) + 1)}
    return Run(Setup(path="unified", pages=8, budget=budget), answers)


@pytest.mark.parametrize("prompt_length", [1, 15, 16, 17, 31, 32, 33, 65])
@pytest.mark.parametrize("chunk", [1, 7, 16, 64])
def test_serial_prefill_and_decode_read_only_initialized_kv(
        monkeypatch, prompt_length, chunk):
    cpu_staging(monkeypatch)
    prompt = list(range(10, 10 + prompt_length))
    continuation = [81, 82, 83, 84, 85]
    run = checking_run(prompt + continuation, chunk)
    capacity = len(prompt) + len(continuation)
    for _ in range(2):
        with PagedSequence(run.torch, run.arena, run.pipeline, chunk, capacity) as seq:
            assert seq.initialized == 0
            assert run.arena.accounting.tokens[seq.key] == 0
            assert seq.forward(prompt) == [prompt_length]
            for i, token in enumerate(continuation, prompt_length + 1):
                assert seq.forward([token]) == [i]
                assert seq.initialized == i
            assert len(run.arena.accounting.owned) == 1
        assert run.arena.free_pages == 8
        assert not run.arena.accounting.owned
        seq.close()
        with pytest.raises(RuntimeError, match="closed"):
            seq.forward([10])


def test_failed_forward_does_not_advance_initialized_length(monkeypatch):
    cpu_staging(monkeypatch)
    run = checking_run([10, 11, 12])
    with PagedSequence(run.torch, run.arena, run.pipeline, 2, 16) as seq:
        assert seq.forward([10, 11]) == [2]

        def fail(chunk):
            raise RuntimeError("forward failed")

        run.pipeline.forward_chunk = fail
        with pytest.raises(RuntimeError, match="forward failed"):
            seq.forward([12])
        assert seq.initialized == 2
        assert run.arena.accounting.tokens[seq.key] == 3
    assert run.arena.free_pages == 8


def test_unexpected_temporary_pages_are_released_before_refusal(monkeypatch):
    cpu_staging(monkeypatch)
    run = checking_run([10])
    real_pack = generate.pack_chunk

    def temporary(*args, **kwargs):
        chunk = real_pack(*args, **kwargs)
        key, _ = run.arena.alloc_temporary(1)
        chunk.temporary_keys = (key,)
        return chunk

    monkeypatch.setattr(generate, "pack_chunk", temporary)
    with pytest.raises(RuntimeError, match="directly"):
        with PagedSequence(run.torch, run.arena, run.pipeline, 16, 16) as seq:
            seq.forward([10])
    assert run.arena.free_pages == 8


@pytest.mark.parametrize("stop_at,limit,reason", [
    (3, 3, "stop"), (3, 2, "length"), (1, 1, "stop"),
])
def test_stop_at_limit_is_success_and_stop_token_is_not_forwarded(
        monkeypatch, stop_at, limit, reason):
    cpu_staging(monkeypatch)
    run = checking_run([10, 11, 12, 13, 14, 15])
    generator = QuailGenerator.__new__(QuailGenerator)
    generator.torch, generator.arena = run.torch, run.arena
    generator.pipeline, generator.chunk_tokens = run.pipeline, 2
    decoded = []
    generator.tokenizer = SimpleNamespace(
        decode=lambda ids, **kw: decoded.append((ids, kw)) or "result")
    matcher = SimpleNamespace(count=0)
    matcher.is_terminated = lambda: matcher.count == stop_at
    forwarded = []
    forward = run.pipeline.forward_chunk

    def record(chunk):
        forwarded.extend(chunk.input_ids.tolist())
        return forward(chunk)

    run.pipeline.forward_chunk = record

    def next_token(hidden, state):
        state.count += 1
        return 12 + state.count

    generator._next_token = next_token
    result = generator._generate([10, 11, 12], matcher, limit)
    assert result == GenerationResult(
        "result", reason, 3, output_tokens=min(stop_at, limit))
    assert forwarded == [10, 11, 12] + decoded[0][0][:-1]
    assert decoded[0][1] == {
        "skip_special_tokens": True, "clean_up_tokenization_spaces": False}
    assert run.arena.free_pages == 8


def test_context_refusals_preserve_order_and_do_not_count_requests(monkeypatch):
    generator = QuailGenerator.__new__(QuailGenerator)
    generator.torch, generator.context_limit = torch, 6
    generator.arena = SimpleNamespace(n_pages=3, page_tokens=2)
    encoded, compiled, evaluated = [], [], []

    def encode(prompt, **kwargs):
        encoded.append((prompt, kwargs))
        return [1] * len(prompt)

    generator.tokenizer = SimpleNamespace(encode=encode)
    generator.compiler = SimpleNamespace(
        compile_json_schema=lambda *a, **kw: compiled.append((a, kw)))
    monkeypatch.setitem(sys.modules, "xgrammar", SimpleNamespace(
        GrammarMatcher=lambda compiled: object()))

    def evaluate(tokens, matcher, limit):
        evaluated.append(tokens)
        return GenerationResult("ok", "stop", len(tokens), output_tokens=1)

    generator._generate = evaluate
    assert generator.generate_structured([], {}, 3) == []
    outputs = generator.generate_structured(["a", "1234", "", "abc"], {}, 3)
    assert outputs == [
        GenerationResult("ok", "stop", 1, output_tokens=1),
        GenerationResult("", "context", 0), GenerationResult("", "context", 0),
        GenerationResult("ok", "stop", 3, output_tokens=1)]
    assert evaluated == [[1], [1, 1, 1]]
    assert all(kw == {"add_special_tokens": False} for _, kw in encoded)
    assert compiled == [(({},), {"any_whitespace": True})]
    generator.arena.n_pages = 1
    assert generator.generate_structured(["a"], {}, 3) == [
        GenerationResult("", "context", 0)]
    generator.arena.n_pages = 4
    assert generator.generate_structured(["abc", "1234"], {}, 3) == [
        GenerationResult("ok", "stop", 3, output_tokens=1),
        GenerationResult("", "context", 0)]
    with pytest.raises(ValueError, match="positive"):
        generator.generate_structured(["a"], {}, 0)


def test_generation_uses_model_projection_and_requires_exact_vocab():
    hidden = torch.zeros((1, 4))
    calls = []
    model = SimpleNamespace(
        quail_generation=True, lm_head=object(),
        config=SimpleNamespace(vocab_size=3),
        compute_logits=lambda h: calls.append(h) or torch.ones(1, 3),
    )
    assert generation_logits(model, hidden).shape == (1, 3)
    assert calls[0] is hidden
    model.compute_logits = lambda h: torch.ones(1, 4)
    with pytest.raises(RuntimeError, match="vocabulary"):
        generation_logits(model, hidden)
    model.quail_generation = False
    with pytest.raises(ValueError, match="full output head"):
        generation_logits(model, hidden)


@pytest.mark.parametrize("mask_needed", [False, True])
def test_grammar_mask_is_applied_before_greedy_selection(monkeypatch, mask_needed):
    generator = QuailGenerator.__new__(QuailGenerator)
    generator.torch = torch
    generator.model = SimpleNamespace(
        quail_generation=True, lm_head=object(),
        config=SimpleNamespace(vocab_size=3),
        compute_logits=lambda h: torch.tensor([[1., 100., 3.]]))
    generator.bitmask = torch.zeros((1, 1), dtype=torch.int32)
    generator.gpu_bitmask = torch.zeros_like(generator.bitmask)
    applied, accepted = [], []

    def mask(logits, bitmask):
        applied.append(1)
        logits[0, 1] = float("-inf")

    monkeypatch.setitem(sys.modules, "xgrammar", SimpleNamespace(
        apply_token_bitmask_inplace=mask))
    matcher = SimpleNamespace(
        fill_next_token_bitmask=lambda b: mask_needed,
        accept_token=lambda token: accepted.append(token) or True,
    )
    token = generator._next_token(None, matcher)
    assert token == (2 if mask_needed else 1)
    assert accepted == [token] and len(applied) == int(mask_needed)
    matcher.accept_token = lambda t: False
    with pytest.raises(RuntimeError, match="rejected"):
        generator._next_token(None, matcher)
