"""Tests for model specs and budget calculations on Qwen3-4B / H100."""

from dataclasses import replace

import pytest

from quail.cost import budgets
from quail.specs import (
    H100_SXM,
    QWEN3_4B_FP8,
    QWEN3_32B_FP8,
    RTX_PRO_6000_BLACKWELL_SERVER,
)


def test_model_weights_and_kv_memory_budgets():
    # Tied output weights must remain available for input embeddings.
    assert QWEN3_4B_FP8.head_mem_bytes == 0.0
    assert QWEN3_4B_FP8.W_resident == QWEN3_4B_FP8.W_mem
    # Unused rows of the separate 32B output weights can be released.
    head = QWEN3_32B_FP8.head_mem_bytes
    assert head == 151_936 * 5_120 * 2
    assert QWEN3_32B_FP8.W_resident == QWEN3_32B_FP8.W_mem - head
    # the freed bytes become arena tokens at the KV rate:
    # 1,555,824,640 bytes / 262,144 bytes per 32B KV token = 5,935,
    # rounded to whole 16-token pages
    kept = replace(QWEN3_32B_FP8, tied_head=True)
    grown = budgets.arena_tokens(QWEN3_32B_FP8, H100_SXM)
    assert grown == 112_304
    assert grown - budgets.arena_tokens(kept, H100_SXM) == 5_936
    # the chunk budget does not move: the 32B chunk is capped by the
    # int32 kernel index, not by memory
    assert budgets.chunk_budget(QWEN3_32B_FP8, H100_SXM) == \
        budgets.chunk_budget(kept, H100_SXM)

    tokens = budgets.arena_tokens(QWEN3_4B_FP8, H100_SXM)
    assert tokens == 362_240     # 22,640 pages of 16 tokens
    assert tokens // 400 == 905
    fp8 = budgets.arena_tokens(QWEN3_4B_FP8.with_kv_bytes(1.0), H100_SXM)
    assert fp8 == pytest.approx(2 * tokens, rel=0.01)
    for model in (QWEN3_4B_FP8, QWEN3_32B_FP8):
        assert budgets.arena_tokens(model, RTX_PRO_6000_BLACKWELL_SERVER) \
            > budgets.arena_tokens(model, H100_SXM)


def test_arena_pages_needs_room_for_one_chunk_of_sliding_kv():
    from quail.specs import DIFFUSION_GEMMA_26B_FP8

    full, sliding = budgets.arena_pages(DIFFUSION_GEMMA_26B_FP8, H100_SXM)
    assert full > 0 and sliding >= budgets.transient_sliding_pages(
        budgets.chunk_budget(DIFFUSION_GEMMA_26B_FP8, H100_SXM),
        DIFFUSION_GEMMA_26B_FP8.sliding_window)
    small = replace(H100_SXM, mem_bytes=40e9)
    with pytest.raises(ValueError, match="one chunk's sliding KV"):
        budgets.arena_pages(DIFFUSION_GEMMA_26B_FP8, small)


@pytest.mark.parametrize("model", [QWEN3_4B_FP8, QWEN3_32B_FP8])
def test_generation_budgets_full_head_and_serial_scratch(model):
    chunk = budgets.chunk_budget(model, H100_SXM, generation=True)
    scratch = budgets.generation_scratch_bytes(model)
    assert scratch >= 6 * model.vocab + model.act_per_token
    available = budgets.arena_bytes(model, H100_SXM, chunk, generation=True)
    assert available == pytest.approx(
        H100_SXM.mem_bytes * budgets.POOL_FRACTION - model.W_mem - scratch
        - budgets.ACT_RESERVE_CHUNKS * chunk * model.act_per_token)
    difference = budgets.arena_bytes(model, H100_SXM, chunk) - available
    assert difference == pytest.approx(model.head_mem_bytes + scratch)
    full, sliding = budgets.arena_pages(model, H100_SXM, generation=True)
    assert sliding == 0 and full > 0
    assert budgets.arena_tokens(model, H100_SXM, generation=True) == full * 16
    assert full * 16 * model.kappa <= available < (full + 1) * 16 * model.kappa
