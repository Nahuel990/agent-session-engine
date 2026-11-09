"""M1 correctness check: our engine vs vLLM, teacher-forced on vLLM's greedy tokens.

Pass rule: at every step our top-1 token equals vLLM's, except where vLLM itself has a
near-tie (its top-2 log-probs within TIE). Also reports the max |logprob diff| of the
reference token.

Usage: python engine/compare.py engine/data/vllm_ref_qwen3-0.6b.json <our_run_output.txt>
"""
import json
import sys

TIE = 0.05

ref = json.load(open(sys.argv[1]))
ours = next(json.loads(l) for l in open(sys.argv[2]) if '"teacher_forced"' in l)["steps"]

fails, max_diff = [], 0.0
for i, (tok, (v1, v2), o) in enumerate(zip(ref["tokens"], ref["top2"], ours)):
    vllm_gap = v1[1] - v2[1]
    diff = abs(o["ref_lp"] - v1[1]) if v1[0] == tok else None
    if diff is not None:
        max_diff = max(max_diff, diff)
    if o["top1"] != v1[0] and vllm_gap >= TIE:
        fails.append({"step": i, "vllm_top1": v1, "ours_top1": [o["top1"], o["top1_lp"]], "vllm_gap": vllm_gap})

print(json.dumps({
    "steps": len(ours),
    "pass": not fails,
    "failures": fails,
    "near_ties_in_vllm": sum(1 for v1, v2 in ref["top2"] if v1[1] - v2[1] < TIE),
    "max_abs_logprob_diff_on_ref_token": round(max_diff, 4),
}, indent=1))
