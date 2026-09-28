"""CPU tests for the scheduling loops the request backends share."""

import asyncio
from functools import partial
from types import SimpleNamespace

import pytest

from quail.backends.request_scheduling import (
    run_filter_chain,
    run_filter_chain_async,
    run_join_grouped,
    true_bit,
)

READ_ANSWER = partial(true_bit, true_ids={1})

def test_join_submission_order_and_empty_inputs():
    for submission in ["anchor-major", "suffix-major"]:
        prefixes = [[100 + i] * (4 + i) for i in range(5)]
        suffixes = [[200 + j] * 3 for j in range(4)]
        client = _ParityClient()
        result = run_join_grouped(client, object(), prefixes, suffixes, READ_ANSWER,
                                  submission=submission)
        expected = ([[100 + i, 200 + j] for i in range(5) for j in range(4)]
                    if submission == "anchor-major" else
                    [[100 + i, 200 + j] for j in range(4) for i in range(5)])
        assert len(client.calls) == 1
        assert [[p["prompt_token_ids"][0], p["prompt_token_ids"][-1]]
                for p in client.calls[0]] == expected
        assert result["answers"] == [
            int((i + j) % 2 == 0) for i in range(5) for j in range(4)]
        assert result["cached_per_request"] == [
            (100 + i) % 7 for i in range(5) for j in range(4)]
        assert result["cached_tokens"] == sum(result["cached_per_request"])
        assert result["fresh_tokens"] == sum(
            len(p) + len(s) for p in prefixes for s in suffixes
        ) - result["cached_tokens"]
        assert result["submission"] == submission
        pairs = [(0, 1), (0, 3), (2, 0), (2, 2), (2, 3)]
        client = _ParityClient()
        result = run_join_grouped(client, object(), prefixes, suffixes, READ_ANSWER,
                                  submission=submission, pairs=pairs)
        heads = [(p["prompt_token_ids"][0] - 100, p["prompt_token_ids"][-1] - 200)
                 for p in client.calls[0]]
        assert heads == (pairs if submission == "anchor-major"
                         else sorted(pairs, key=lambda pair: pair[::-1]))
        # answers and cache counts come back in the listed order
        assert result["answers"] == [int((a + s) % 2 == 0) for a, s in pairs]
        assert result["cached_per_request"] == [(100 + a) % 7 for a, _ in pairs]
    with pytest.raises(ValueError, match="unknown join submission"):
        run_join_grouped(_ParityClient(), object(), prefixes, suffixes, READ_ANSWER,
                         submission="unknown")

    for prefixes, suffixes in [([], [[1]]), ([[1]], []), ([], [])]:
        result = run_join_grouped(
            _ParityClient(), object(), prefixes, suffixes, READ_ANSWER,
            submission="suffix-major")
        assert result["answers"] == []
        assert result["cached_per_request"] == []
        assert result["fresh_tokens"] == 0


class _FakeFilterEngine:
    """Finishes one queued request per step and answers TRUE."""

    def __init__(self):
        self.pending = []
        self.events = []

    def add_request(self, request_id, prompt, _sampling_params):
        self.pending.append((request_id, prompt["prompt_token_ids"]))
        self.events.append(("add", request_id))

    def step(self):
        request_id, prompt_token_ids = self.pending.pop(0)
        self.events.append(("finish", request_id))
        return [SimpleNamespace(
            finished=True,
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            num_cached_tokens=0,
            outputs=[SimpleNamespace(token_ids=[1], text="TRUE")],
        )]


def test_filter_chains_pipeline_refill_fail_and_take_empty_input():
    engine = _FakeFilterEngine()
    result = run_filter_chain(
        engine,
        sampling_params=object(),
        body_ids=[[1] * 5, [2] * 17],
        question_ids=[[3] * 3, [4] * 3],
        budget_tokens=100,
        tag="test",
        read_answer=READ_ANSWER,
        block_size=16,
        max_num_seqs=2,
    )

    assert result["doc_cap"] == 2
    assert engine.events.index(("add", "test-0-1")) < \
        engine.events.index(("finish", "test-1-0"))
    assert result["survivors"] == [0, 1]

    async def run():
        release = asyncio.Event()
        events = []
        active = 0
        peak = 0

        async def generate(prompt, sampling_params):
            nonlocal active, peak
            document, question = prompt[0], prompt[-1]
            events.append(("start", document, question))
            active += 1
            peak = max(peak, active)
            if document == 2 and question == 8:
                await release.wait()
            else:
                await asyncio.sleep(0)
            if document == 3:
                release.set()
            answer = document == 2 or (document == 1 and question == 8)
            active -= 1
            events.append(("finish", document, question))
            return SimpleNamespace(
                prompt_token_ids=prompt, num_cached_tokens=document,
                outputs=[SimpleNamespace(token_ids=[1 if answer else 0])],
            )

        result = await asyncio.wait_for(run_filter_chain_async(
            generate, {}, [[1] * 10, [2] * 20, [3] * 30],
            [[8] * 3, [9] * 4], 64, read_answer=READ_ANSWER, block_size=16,
        ), timeout=2)
        assert result["doc_cap"] == 2
        assert peak == 2
        assert events.index(("start", 1, 9)) < events.index(("finish", 2, 8))
        assert events.index(("start", 3, 8)) < events.index(("finish", 2, 8))
        assert result["survivors"] == [1]
        assert result["requests"] == 5
        assert result["prompt_tokens"] == 13 + 14 + 23 + 24 + 33
        assert result["cached_tokens"] == 1 + 1 + 2 + 2 + 3
        assert result["answers"] == {
            (0, 1): 1, (0, 2): 0, (1, 1): 1, (1, 2): 1, (2, 1): 0}

    asyncio.run(run())

    # a failed request cancels the rest; no documents means no requests
    async def run():
        pending = asyncio.Event()
        cancelled = asyncio.Event()

        async def generate(prompt, sampling_params):
            if prompt[0] == 1:
                await pending.wait()
                raise RuntimeError("request failed")
            pending.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with pytest.raises(ExceptionGroup, match="TaskGroup"):
            await run_filter_chain_async(
                generate, {}, [[1], [2]], [[9]], 100, read_answer=READ_ANSWER,
                max_num_seqs=2,
            )
        assert cancelled.is_set()

    asyncio.run(run())

    async def generate(prompt, sampling_params):
        raise AssertionError("no documents")

    result = asyncio.run(
        run_filter_chain_async(generate, {}, [], [[9]], 100, read_answer=READ_ANSWER))
    assert result["requests"] == 0
    assert result["survivors"] == []


class _ParityClient:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, _sp, use_tqdm=False):
        self.calls.append(prompts)
        outs = []
        for prompt in prompts:
            ids = prompt["prompt_token_ids"]
            bit = 1 if (ids[0] + ids[-1]) % 2 == 0 else 0
            outs.append(SimpleNamespace(
                outputs=[SimpleNamespace(token_ids=[bit])],
                prompt_token_ids=ids,
                num_cached_tokens=ids[0] % 7))
        return outs
