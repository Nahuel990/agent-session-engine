"""Rebuilds the headline comparison from the result files in data/m3.

Usage: python engine/summarize.py
"""
import re
from pathlib import Path

DATA = Path(__file__).parent / "data" / "m3"

# (label, file stems for 5 s tool delay, file stems for 60 s tool delay)
RUNS = [
    ("vLLM 0.33.0", ["local_vllm_c24_d5", "local_vllm_c24_d5_r2"], ["local_vllm_c24_d60_n1", "local_vllm_c24_d60_n2"]),
    ("SGLang 0.5.23 (lru)", ["local_sglang_c24_d5_lru_n1", "local_sglang_c24_d5_lru_n2"],
     ["local_sglang_c24_d60_lru_n1", "local_sglang_c24_d60_lru_n2"]),
    ("SGLang 0.5.23 (tlru)", ["local_sglang_c24_d5_tlru_n1", "local_sglang_c24_d5_tlru_n2"],
     ["local_sglang_c24_d60_tlru_n1", "local_sglang_c24_d60_tlru_n2"]),
    ("This engine (1c92959, capped)", ["local_ours_c24_d5_1c92959_share3b_n1", "local_ours_c24_d5_1c92959_share3b_n2"],
     ["local_ours_c24_d60_1c92959_share3b_n1", "local_ours_c24_d60_1c92959_share3b_n2"]),
]


def read(stem: str) -> dict:
    text = (DATA / f"{stem}.txt").read_text()
    wall = float(re.search(r"wall=(\d+)s", text).group(1))
    turns = int(re.search(r"turns=(\d+)", text).group(1))
    recomputed = float(re.search(r"recomputed_history=\d+ \(([\d.]+)%\)", text).group(1))
    p50, p90 = map(float, re.search(r"ttft p50=([\d.]+)s p90=([\d.]+)s", text).groups())
    return {"wall": wall, "turns": turns, "recomputed": recomputed, "p50": p50, "p90": p90}


def fmt(runs: list[dict]) -> str:
    walls = "/".join(f"{r['wall']:.0f}" for r in runs)
    mean = sum(r["wall"] for r in runs) / len(runs)
    rec = "/".join(f"{r['recomputed']:.1f}" for r in runs)
    p50 = "/".join(f"{r['p50']:.1f}" for r in runs)
    p90 = "/".join(f"{r['p90']:.1f}" for r in runs)
    return f"wall {walls} s (mean {mean:.0f}) | recomputed {rec} % | TTFT p50 {p50} s, p90 {p90} s | turns {runs[0]['turns']}"


means = {}
for delay_index, delay in ((1, "5 s"), (2, "60 s")):
    print(f"## Tool delay {delay}")
    for row in RUNS:
        runs = [read(s) for s in row[delay_index]]
        means[(row[0], delay)] = sum(r["wall"] for r in runs) / len(runs)
        print(f"{row[0]:32} {fmt(runs)}")
    ours = means[(RUNS[-1][0], delay)]
    for label, *_ in RUNS[:-1]:
        print(f"  this engine vs {label}: {means[(label, delay)] / ours:.2f}x (mean wall time)")
    print()
