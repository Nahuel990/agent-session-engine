"""Raw generation speed: N concurrent requests, short prompt, fixed output length, no reuse.

Reports output tokens per second (all requests together) and time per output token.
Usage: python engine/phase0/decode_bench.py <base_url> <concurrency> [out_tokens]
"""
import asyncio
import json
import sys
import time

import httpx

BASE = sys.argv[1].rstrip("/")
N = int(sys.argv[2])
OUT = int(sys.argv[3]) if len(sys.argv) > 3 else 256


async def one(client, i):
    body = {
        "model": "llm",
        # A different first word per request so no two prompts share a cached prefix.
        "messages": [{"role": "user", "content": f"Request {i}. Write a long story about a lighthouse."}],
        "max_tokens": OUT,
        "ignore_eos": True,
    }
    r = await client.post(f"{BASE}/v1/chat/completions", json=body)
    r.raise_for_status()
    return r.json()["usage"]["completion_tokens"]


async def main():
    limits = httpx.Limits(max_keepalive_connections=0)
    async with httpx.AsyncClient(timeout=httpx.Timeout(3600.0), limits=limits) as client:
        await one(client, -1)  # warm-up
        start = time.perf_counter()
        tokens = await asyncio.gather(*(one(client, i) for i in range(N)))
        wall = time.perf_counter() - start
    total = sum(tokens)
    print(json.dumps({"concurrency": N, "out_tokens_each": OUT, "wall_s": round(wall, 3),
                      "output_tok_per_s": round(total / wall, 1), "ms_per_token_per_request": round(wall / OUT * 1000, 2)}))


asyncio.run(main())
