"""vLLM baseline server on Modal for the Phase 0 cache-reuse benchmark.

Deploy:  modal deploy engine/phase0/serve.py
Stop:    modal app stop phase0-vllm
"""
import subprocess

import modal

MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
VLLM_VERSION = "0.33.0"
GPU = "L40S"
PORT = 8000
MINUTES = 60

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("phase0-vllm")


@app.server(
    image=image,
    gpu=GPU,
    scaledown_window=5 * MINUTES,
    startup_timeout=15 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    port=PORT,
    target_concurrency=256,
    unauthenticated=True,
)
class Server:
    @modal.enter()
    def start(self):
        cmd = [
            "vllm", "serve", MODEL_NAME,
            "--revision", MODEL_REVISION,
            "--served-model-name", "llm",
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--max-model-len", "32768",
            "--enable-prefix-caching",
            "--enable-prompt-tokens-details",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ]
        print(*cmd)
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()
