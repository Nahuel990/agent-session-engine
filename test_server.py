"""Smoke test for the engine's OpenAI-compatible server with sessions.

Usage: python engine/test_server.py <base_url>
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip("/")


def post(body):
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r), time.perf_counter() - start


for i in range(60):
    try:
        urllib.request.urlopen(f"{BASE}/health", timeout=30).read()
        break
    except Exception:
        time.sleep(10)
else:
    sys.exit("server never became healthy")

system = {"role": "system", "content": "You are a helpful agent. " + "Follow the rules carefully. " * 200}
messages = [system, {"role": "user", "content": "List three AWS storage services. /no_think"}]

r1, t1 = post({"model": "llm", "messages": messages, "max_tokens": 64, "session_id": "s1"})
messages += [r1["choices"][0]["message"], {"role": "user", "content": "Which one is cheapest for archives? /no_think"}]
r2, t2 = post({"model": "llm", "messages": messages, "max_tokens": 64, "session_id": "s1"})
r3, t3 = post({"model": "llm", "messages": messages, "max_tokens": 64})  # same prompt, no session

for name, r, t in [("turn1", r1, t1), ("turn2 session", r2, t2), ("turn2 no session", r3, t3)]:
    u = r["usage"]
    print(json.dumps({"call": name, "seconds": round(t, 3), "prompt_tokens": u["prompt_tokens"],
                      "cached_tokens": u["prompt_tokens_details"]["cached_tokens"],
                      "reply": r["choices"][0]["message"]["content"][:120]}))
