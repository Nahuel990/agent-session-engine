"""Correctness check for cross-request prefix sharing (greedy, so outputs must be identical).

1. Prompt A twice: second reuses cached blocks and returns the same text.
2. Prompt B = same system prompt as A, different user message: reuses the shared start.
   Its output must equal B's output on an empty cache (run first, before A is cached).
Usage: python engine/phase0/share_check.py <base_url>
"""
import json
import sys
import urllib.request

BASE = sys.argv[1].rstrip("/")
SYSTEM = "You are a careful software agent. " + "Follow every rule in the policy exactly. " * 120


def ask(user: str) -> dict:
    body = {"model": "llm", "max_tokens": 48,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user + " /no_think"}]}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return {"text": r["choices"][0]["message"]["content"], "cached": r["usage"]["prompt_tokens_details"]["cached_tokens"],
            "prompt": r["usage"]["prompt_tokens"]}


b_fresh = ask("Name three prime numbers.")        # empty cache
a1 = ask("List two AWS storage services.")       # shares the system prompt with B
a2 = ask("List two AWS storage services.")
b_shared = ask("Name three prime numbers.")
checks = {
    "B on empty cache had no reuse": b_fresh["cached"] == 0,
    "A reused the shared system prompt": a1["cached"] > 0,
    "A repeat reused at least as much": a2["cached"] >= a1["cached"],
    "A repeat output identical": a1["text"] == a2["text"],
    "B with shared prefix output identical to empty-cache B": b_shared["text"] == b_fresh["text"],
}
print(json.dumps({"pass": all(checks.values()), "checks": checks,
                  "cached": {"b_fresh": b_fresh["cached"], "a1": a1["cached"], "a2": a2["cached"], "b_shared": b_shared["cached"]},
                  "prompt_tokens": b_fresh["prompt"]}, indent=1))
