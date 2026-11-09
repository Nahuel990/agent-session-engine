"""Serve the engine's OpenAI-compatible API on Modal.

Build first:  modal run engine/modal_build.py
Deploy:       modal deploy engine/serve_engine.py
Stop:         modal app stop engine-serve
"""
import subprocess

import modal

MODEL = "Qwen/Qwen3-8B"
NUM_BLOCKS = 2706  # 8B on L40S: 2706 x 64 = 173,184 tokens, matching vLLM's 173,200-token KV cache
PORT = 8000
SESSION_TTL = 600  # seconds an idle agent session keeps its KV cache

image = modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12").entrypoint([])
build_cache = modal.Volume.from_name("engine-build")
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
app = modal.App("engine-serve")


@app.server(
    image=image,
    gpu="L40S",
    scaledown_window=5 * 60,
    startup_timeout=10 * 60,
    volumes={"/cache": build_cache, "/root/.cache/huggingface": hf_cache},
    port=PORT,
    target_concurrency=256,
    unauthenticated=True,
)
class Server:
    @modal.enter()
    def start(self):
        self.process = subprocess.Popen(["/cache/target/release/engine", "serve", MODEL, str(PORT), str(NUM_BLOCKS), str(SESSION_TTL)])

    @modal.exit()
    def stop(self):
        self.process.terminate()
