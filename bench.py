"""In-container benchmark: start a server and replay agent sessions against localhost,
so no proxy or network sits between the client and the engine.

Usage:
  modal run engine/bench.py --engine vllm --concurrency 24 --tool-delay 5
  modal run engine/bench.py --engine ours --concurrency 24 --tool-delay 5
Results land in engine/data/m3/local_<engine>_c<N>_d<D>.{csv,txt}.
"""
import subprocess
import time
import urllib.request
from pathlib import Path

import modal

HERE = Path(__file__).parent
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
OUR_NUM_BLOCKS = 2706  # 2706 x 64 = 173,184 tokens, matching vLLM's 173,200-token KV cache on L40S
PORT = 8000
MINUTES = 60

client_files = lambda img: (
    img.uv_pip_install("httpx")
    .add_local_file(HERE / "phase0" / "replay.py", "/bench/replay.py")
    .add_local_file(HERE / "phase0" / "decode_bench.py", "/bench/decode_bench.py")
    .add_local_file(HERE / "phase0" / "prefill_bench.py", "/bench/prefill_bench.py")
    .add_local_file(HERE / "data" / "sessions_64.jsonl", "/bench/sessions.jsonl")
)
vllm_image = client_files(
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.33.0")
)
ours_image = client_files(
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12").entrypoint([])
)

SGLANG_IMAGE = "lmsysorg/sglang:v0.5.23-cu130"
sglang_image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .entrypoint([])
    .pip_install("httpx")
    .add_local_file(HERE / "phase0" / "replay.py", "/bench/replay.py")
    .add_local_file(HERE / "phase0" / "decode_bench.py", "/bench/decode_bench.py")
    .add_local_file(HERE / "phase0" / "prefill_bench.py", "/bench/prefill_bench.py")
    .add_local_file(HERE / "data" / "sessions_64.jsonl", "/bench/sessions.jsonl")
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
build_cache = modal.Volume.from_name("engine-build")
# Results are also written here, so they survive a dropped local client.
# Fetch with: modal volume get engine-results <name>.txt
results = modal.Volume.from_name("engine-results", create_if_missing=True)
app = modal.App("engine-bench")


def wait_healthy(url: str, timeout_s: int = 20 * MINUTES):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(url, timeout=10).status == 200:
                return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"server not healthy after {timeout_s}s")


def decode_speed(levels=(1, 8, 24)) -> dict:
    log = ""
    for n in levels:
        proc = subprocess.run(["python", "/bench/decode_bench.py", f"http://127.0.0.1:{PORT}", str(n)], capture_output=True, text=True)
        log += proc.stdout + proc.stderr[-2000:]
    return {"csv": "", "log": log}


def prefill_speed(cases=((1, 3000), (8, 3000), (1, 12000), (4, 12000))) -> dict:
    log = ""
    for n, words in cases:
        proc = subprocess.run(["python", "/bench/prefill_bench.py", f"http://127.0.0.1:{PORT}", str(n), str(words)],
                              capture_output=True, text=True)
        log += proc.stdout + proc.stderr[-2000:]
    return {"csv": "", "log": log}


def profile_prefill(cases=((1, 12000), (4, 12000))) -> dict:
    """Needs the server started with ENGINE_PROFILE=1. Logs per-section GPU time per case."""
    log = ""
    urllib.request.urlopen(f"http://127.0.0.1:{PORT}/profile", timeout=30).read()  # reset after warm-up
    for n, words in cases:
        proc = subprocess.run(["python", "/bench/prefill_bench.py", f"http://127.0.0.1:{PORT}", str(n), str(words)],
                              capture_output=True, text=True)
        prof = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/profile", timeout=30).read().decode()
        log += proc.stdout + proc.stderr[-2000:] + f"PROFILE {n}x{words} {prof}\n"
    return {"csv": "", "log": log}


def run_mode(mode: str, concurrency: int, tool_delay: float, send_session: bool, name: str = "") -> dict:
    if mode == "profile":
        out = profile_prefill()
    elif mode == "decode":
        out = decode_speed()
    elif mode == "prefill":
        out = prefill_speed()
    else:
        out = replay(concurrency, tool_delay, send_session)
    if name:
        Path(f"/results/{name}.txt").write_text(out["log"])
        Path(f"/results/{name}.csv").write_text(out["csv"])
        results.commit()
    return out


def result_name(engine: str, mode: str, concurrency: int, tool_delay: float, tag: str) -> str:
    name = f"{mode}_{engine}" if mode in ("decode", "prefill", "profile") else f"local_{engine}_c{concurrency}_d{tool_delay:g}"
    return name + (f"_{tag}" if tag else "")


def replay(concurrency: int, tool_delay: float, send_session: bool) -> dict:
    out = "/tmp/out.csv"
    cmd = ["python", "/bench/replay.py", f"http://127.0.0.1:{PORT}", "/bench/sessions.jsonl",
           str(concurrency), str(tool_delay), out] + (["session"] if send_session else [])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    csv = Path(out).read_text() if Path(out).exists() else ""
    return {"csv": csv, "log": proc.stdout + proc.stderr}


@app.function(image=vllm_image, gpu="L40S", timeout=3 * 60 * MINUTES,
              volumes={"/root/.cache/huggingface": hf_cache, "/results": results})
def bench_vllm(concurrency: int, tool_delay: float, mode: str = "replay", name: str = "") -> dict:
    server = subprocess.Popen([
        "vllm", "serve", MODEL, "--revision", MODEL_REVISION, "--served-model-name", "llm",
        "--host", "127.0.0.1", "--port", str(PORT), "--max-model-len", "32768",
        "--enable-prefix-caching", "--enable-prompt-tokens-details",
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
    ])
    try:
        wait_healthy(f"http://127.0.0.1:{PORT}/health")
        return run_mode(mode, concurrency, tool_delay, send_session=False, name=name)
    finally:
        server.terminate()


@app.function(image=sglang_image, gpu="L40S", timeout=3 * 60 * MINUTES,
              volumes={"/root/.cache/huggingface": hf_cache, "/results": results})
def bench_sglang(concurrency: int, tool_delay: float, mode: str = "replay", name: str = "") -> dict:
    # Flags checked against sglang v0.5.23 source (arg_groups/fields/*.py, function_call_parser.py).
    # --max-total-tokens matches the KV pool to vLLM's 173,200 tokens on L40S.
    server = subprocess.Popen([
        "python3", "-m", "sglang.launch_server", "--model-path", MODEL, "--revision", MODEL_REVISION,
        "--served-model-name", "llm", "--host", "127.0.0.1", "--port", str(PORT),
        "--context-length", "32768", "--max-total-tokens", "173200",
        "--enable-cache-report", "--tool-call-parser", "qwen",
    ])
    try:
        wait_healthy(f"http://127.0.0.1:{PORT}/health")
        return run_mode(mode, concurrency, tool_delay, send_session=False, name=name)
    finally:
        server.terminate()


@app.function(image=ours_image, gpu="L40S", timeout=3 * 60 * MINUTES,
              volumes={"/root/.cache/huggingface": hf_cache, "/cache": build_cache, "/results": results})
def bench_ours(concurrency: int, tool_delay: float, mode: str = "replay", env_flags: str = "", name: str = "") -> dict:
    import os
    flags = [f.split("=", 1) if "=" in f else (f, "1") for f in filter(None, env_flags.split(","))]
    env = dict(os.environ, **dict(flags))
    server = subprocess.Popen(["/cache/target/release/engine", "serve", MODEL, str(PORT), str(OUR_NUM_BLOCKS)], env=env)
    try:
        wait_healthy(f"http://127.0.0.1:{PORT}/health")
        return run_mode(mode, concurrency, tool_delay, send_session=True, name=name)
    finally:
        server.terminate()


@app.local_entrypoint()
def main(engine: str = "ours", concurrency: int = 24, tool_delay: float = 5.0, mode: str = "replay",
         env_flags: str = "", tag: str = "", spawn: bool = False):
    name = result_name(engine, mode, concurrency, tool_delay, tag)
    if spawn:
        # Hand the run to Modal and return: nothing depends on this local process afterwards.
        # Use with `modal run --detach`; results land in volume `engine-results` as <name>.txt/.csv.
        if engine == "vllm":
            bench_vllm.spawn(concurrency, tool_delay, mode, name)
        elif engine == "sglang":
            bench_sglang.spawn(concurrency, tool_delay, mode, name)
        else:
            bench_ours.spawn(concurrency, tool_delay, mode, env_flags, name)
        print(f"spawned {name}")
        return
    if engine == "vllm":
        result = bench_vllm.remote(concurrency, tool_delay, mode, name)
    elif engine == "sglang":
        result = bench_sglang.remote(concurrency, tool_delay, mode, name)
    else:
        result = bench_ours.remote(concurrency, tool_delay, mode, env_flags, name)
    stem = HERE / "data" / "m3" / name
    Path(f"{stem}.csv").write_text(result["csv"])
    Path(f"{stem}.txt").write_text(result["log"])
    print("\n".join(result["log"].splitlines()[-4:]))
