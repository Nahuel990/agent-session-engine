"""Replay recorded agent sessions against an OpenAI-compatible server and measure
how much of each turn's prompt the server had to compute again.

Each session sends its history turn by turn, like an agent does: the prompt for
turn t is every message before the t-th assistant message. Between turns the
client sleeps TOOL_DELAY seconds to stand in for tool execution.

"Recomputed" = tokens the server already saw on the previous turn of the same
session but did not serve from cache. A stateful engine would compute 0 of them.

Usage:
  python engine/phase0/replay.py <base_url> <sessions.jsonl> <concurrency> <tool_delay_s> <out.csv> [session]

Pass `session` as the 6th argument to send a `session_id` per agent session (our engine).
"""
import asyncio
import csv
import json
import sys
import time

import httpx

BASE_URL, SESSIONS, CONCURRENCY, TOOL_DELAY, OUT = (
    sys.argv[1].rstrip("/"), sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5])
SEND_SESSION_ID = len(sys.argv) > 6 and sys.argv[6] == "session"
MAX_PROMPT_TOKENS = 30_000
MAX_TOKENS = 256


async def run_session(client, session, rows):
    messages = session["messages"]
    turns = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    prev_prompt = 0
    for turn, cut in enumerate(turns):
        body = {
            "model": "llm",
            "messages": messages[:cut],
            "tools": session["tools"],
            "max_tokens": MAX_TOKENS,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if SEND_SESSION_ID:
            body["session_id"] = session["id"]
        start = time.perf_counter()
        ttft, usage = None, None
        async with client.stream("POST", f"{BASE_URL}/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                print(f"{session['id']} turn {turn}: HTTP {r.status_code} {(await r.aread())[:200]!r}")
                return
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if "error" in chunk:
                    print(f"{session['id']} turn {turn}: stream error {chunk['error']}")
                    return
                if ttft is None and chunk.get("choices"):
                    ttft = time.perf_counter() - start
                if chunk.get("usage"):
                    usage = chunk["usage"]
        total = time.perf_counter() - start
        if usage is None:
            print(f"{session['id']} turn {turn}: stream ended without usage")
            return
        prompt = usage["prompt_tokens"]
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        rows.append({
            "session": session["id"], "turn": turn, "prompt_tokens": prompt,
            "cached_tokens": cached, "recomputed_tokens": max(0, prev_prompt - cached),
            "ttft_s": round(ttft or total, 4), "total_s": round(total, 4),
        })
        print(f"{time.strftime('%H:%M:%S')} {session['id'][-8:]} turn {turn} prompt {prompt} cached {cached} {total:.1f}s", flush=True)
        prev_prompt = prompt
        if prompt > MAX_PROMPT_TOKENS:
            return
        await asyncio.sleep(TOOL_DELAY)


async def main():
    sessions = [json.loads(line) for line in open(SESSIONS)][:CONCURRENCY]
    rows = []
    # A fresh connection per request: reusing one after a tool delay races the server's
    # keep-alive timeout (uvicorn's default is 5 s) and fails with ReadError.
    limits = httpx.Limits(max_keepalive_connections=0)
    async with httpx.AsyncClient(timeout=httpx.Timeout(3600.0), limits=limits) as client:
        start = time.perf_counter()
        await asyncio.gather(*(run_session(client, s, rows) for s in sessions))
        wall = time.perf_counter() - start
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    prompt = sum(r["prompt_tokens"] for r in rows)
    cached = sum(r["cached_tokens"] for r in rows)
    recomputed = sum(r["recomputed_tokens"] for r in rows)
    ttfts = sorted(r["ttft_s"] for r in rows)
    print(f"sessions={len(sessions)} turns={len(rows)} tool_delay={TOOL_DELAY}s wall={wall:.0f}s")
    print(f"prompt_tokens={prompt} cached={cached} ({cached / prompt:.1%}) recomputed_history={recomputed} ({recomputed / prompt:.1%})")
    print(f"ttft p50={ttfts[len(ttfts) // 2]:.3f}s p90={ttfts[int(len(ttfts) * 0.9)]:.3f}s")


asyncio.run(main())
