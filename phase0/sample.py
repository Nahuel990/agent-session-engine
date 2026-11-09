"""Pick N agent sessions from the SWE-rebench OpenHands trajectories and save them as JSONL.

Usage: python engine/phase0/sample.py 64
"""
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
SEED = 0
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "trajectories.parquet"
OUT = ROOT / "data" / f"sessions_{N}.jsonl"

MESSAGE_FIELDS = {
    "system": ["role", "content"],
    "user": ["role", "content"],
    "assistant": ["role", "content", "tool_calls"],
    "tool": ["role", "content", "tool_call_id"],
}

def drop_nulls(value):
    """Parquet structs carry every possible key; drop the ones that are null."""
    if isinstance(value, dict):
        return {k: drop_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [drop_nulls(v) for v in value]
    return value


pf = pq.ParquetFile(SRC)
rows = pf.read_row_group(random.Random(SEED).randrange(pf.num_row_groups)).to_pylist()
random.Random(SEED).shuffle(rows)

with OUT.open("w") as f:
    for row in rows[:N]:
        messages = []
        for m in row["trajectory"]:
            msg = {k: m[k] for k in MESSAGE_FIELDS[m["role"]] if m.get(k) is not None}
            if msg["role"] == "assistant" and "content" not in msg:
                msg["content"] = ""
            messages.append(msg)
        f.write(json.dumps({"id": row["trajectory_id"], "tools": drop_nulls(row["tools"]), "messages": messages}) + "\n")

print(f"wrote {N} sessions to {OUT}")
