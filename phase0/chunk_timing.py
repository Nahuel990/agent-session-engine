"""Prints the arrival time and content of the first streamed chunks from a server.

Usage: python engine/phase0/chunk_timing.py <base_url>
"""
import json
import sys
import time

import httpx

body = {
    "model": "llm",
    "messages": [{"role": "user", "content": "Count from one to ten in words."}],
    "max_tokens": 32,
    "ignore_eos": True,
    "stream": True,
    "stream_options": {"include_usage": True},
}
start = time.perf_counter()
with httpx.stream("POST", f"{sys.argv[1].rstrip('/')}/v1/chat/completions", json=body, timeout=300) as r:
    n = 0
    for line in r.iter_lines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        delta = chunk["choices"][0]["delta"] if chunk.get("choices") else None
        print(f"{time.perf_counter() - start:7.3f}s  {json.dumps(delta)[:80]}")
        n += 1
        if n >= 8:
            break
