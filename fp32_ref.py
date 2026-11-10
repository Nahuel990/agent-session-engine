"""FP32 ground truth for the teacher-forced check: Hugging Face transformers, full precision.

Runs prompt + reference tokens through Qwen3-8B in float32 in one forward pass and records,
for each reference token, its log-prob and the model's top-1 at that step.

Usage: modal run engine/fp32_ref.py
Writes engine/data/fp32_ref_qwen3-8b.json
"""
import json
from pathlib import Path

import modal

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
HERE = Path(__file__).parent

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch", "transformers", "accelerate")
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
app = modal.App("engine-fp32-ref")


@app.function(image=image, gpu="L40S", timeout=30 * 60, volumes={"/root/.cache/huggingface": hf_cache})
def reference(ref_json: str) -> str:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    ref = json.loads(ref_json)
    prompt, tokens = ref["prompt_ids"], ref["tokens"]
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float32, device_map="cuda")
    ids = torch.tensor([prompt + tokens], device="cuda")
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    steps = []
    for i, tok in enumerate(tokens):
        row = logprobs[len(prompt) - 1 + i]
        top2 = torch.topk(row, 2)
        steps.append({"ref_lp": row[tok].item(), "top1": top2.indices[0].item(), "top1_lp": top2.values[0].item(),
                      "top2": top2.indices[1].item(), "top2_lp": top2.values[1].item()})
    return json.dumps({"model": MODEL, "dtype": "float32", "torch": torch.__version__,
                       "transformers": transformers.__version__, "steps": steps})


@app.local_entrypoint()
def main(ref_file: str = "engine/data/vllm_ref_qwen3-8b.json", out: str = "engine/data/fp32_ref_qwen3-8b.json"):
    Path(out).write_text(reference.remote(Path(ref_file).read_text()))
    print(f"wrote {out}")
