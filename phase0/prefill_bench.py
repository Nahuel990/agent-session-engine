"""Prompt-processing speed: N concurrent requests with long, unique prompts and 1 output token.

Reports prompt tokens processed per second (all requests together).
Usage: python engine/phase0/prefill_bench.py <base_url> <concurrency> <approx_prompt_words>
"""
import asyncio
import json
import random
import sys
import time

import httpx

BASE = sys.argv[1].rstrip("/")
N = int(sys.argv[2])
WORDS = int(sys.argv[3])
VOCAB = "agent tool cache token server request session memory engine prompt bucket lambda queue".split()


def prompt(i: int) -> str:
    rng = random.Random(i)
    # A unique number first, so no two prompts share a cached prefix.
    return f"{i} {rng.random()} " + " ".join(rng.choice(VOCAB) for _ in range(WORDS))


async def one(client, i):
    body = {"model": "llm", "messages": [{"role": "user", "content": prompt(i)}], "max_tokens": 1, "ignore_eos": True}
    r = await client.post(f"{BASE}/v1/chat/completions", json=body)
    r.raise_for_status()
    return r.json()["usage"]["prompt_tokens"]


async def main():
    limits = httpx.Limits(max_keepalive_connections=0)
    async with httpx.AsyncClient(timeout=httpx.Timeout(3600.0), limits=limits) as client:
        await one(client, -1)  # warm-up
        start = time.perf_counter()
        tokens = await asyncio.gather(*(one(client, 1000 * WORDS + i) for i in range(N)))
        wall = time.perf_counter() - start
    print(json.dumps({"concurrency": N, "prompt_tokens_each": tokens[0], "wall_s": round(wall, 3),
                      "prompt_tok_per_s": round(sum(tokens) / wall, 1)}))


asyncio.run(main())
