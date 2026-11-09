"""Greedy reference from vLLM, on the exact token ids our engine uses.

Usage: modal run engine/vllm_ref.py
"""
import json

import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.33.0")
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
app = modal.App("engine-vllm-ref")


@app.function(image=image, gpu=["L4", "A10", "L40S"], timeout=30 * 60, volumes={"/root/.cache/huggingface": hf_cache})
def reference(model_id: str = "Qwen/Qwen3-0.6B", prompt: str = "The capital of France is", n: int = 64) -> str:
    from vllm import LLM, SamplingParams

    # The check uses ~70 tokens; a short context keeps the KV cache small enough for 24 GB GPUs.
    llm = LLM(model=model_id, dtype="bfloat16", enable_prefix_caching=False, max_model_len=4096)
    ids = llm.get_tokenizer().encode(prompt, add_special_tokens=False)
    params = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True, logprobs=5)
    out = llm.generate({"prompt_token_ids": ids}, params)[0].outputs[0]
    hf_cache.commit()
    top = [sorted(((t, lp.logprob) for t, lp in step.items()), key=lambda x: (-x[1], x[0]))[:2]
           for step in out.logprobs]
    return json.dumps({"prompt_ids": ids, "tokens": list(out.token_ids), "top2": top, "text": out.text})


@app.local_entrypoint()
def main(out: str = "engine/data/vllm_ref_qwen3-0.6b.json", model_id: str = "Qwen/Qwen3-0.6B"):
    result = reference.remote(model_id)
    with open(out, "w") as f:
        f.write(result)
    print(f"wrote {out}")
