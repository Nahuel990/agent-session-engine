"""Build the engine on Modal (Linux + CUDA toolkit, no GPU attached).

Usage: modal run engine/modal_build.py
Build artifacts persist in the `engine-build` volume, so rebuilds are incremental.
"""
from pathlib import Path

import modal

CUDA_COMPUTE_CAP = "89"  # L4 / L40S. H100 is 90.
HERE = Path(__file__).parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "curl", "build-essential", "pkg-config", "libssl-dev")
    .run_commands("curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable")
    .env({
        "PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "CUDA_COMPUTE_CAP": CUDA_COMPUTE_CAP,
        "CARGO_HOME": "/cache/cargo",
        "CARGO_TARGET_DIR": "/cache/target",
    })
    .add_local_dir(HERE / "core", "/src", ignore=["target", "target/**"])
)

cache = modal.Volume.from_name("engine-build", create_if_missing=True)
app = modal.App("engine-build")


@app.function(image=image, cpu=16, memory=32768, timeout=3 * 60 * 60, volumes={"/cache": cache})
def build(features: str = "flashinfer", release: bool = True, tree: str = "", precise: str = "", fresh_lock: bool = False) -> str:
    import shutil
    import subprocess

    # Build in a working copy on the volume so Cargo.lock persists between builds.
    work = "/cache/work"
    if fresh_lock:
        shutil.rmtree(work, ignore_errors=True)
    shutil.copytree("/src", work, dirs_exist_ok=True)
    # precise="crate@old:new,..." pins locked versions, e.g. "syn@2.0.119:2.0.100".
    pinned = ""
    for spec in filter(None, precise.split(",")):
        crate, version = spec.split(":")
        p = subprocess.run(["cargo", "update", "-p", crate, "--precise", version], cwd=work, capture_output=True, text=True)
        pinned += f"\n--- cargo update -p {crate} --precise {version}: exit={p.returncode}\n" + p.stderr[-600:]
    profile = ["--release"] if release else []
    # Uploaded sources can look older than cached artifacts, so always rebuild our own crate.
    subprocess.run(["cargo", "clean", "-p", "engine"] + profile, cwd=work, capture_output=True)
    cmd = ["cargo", "build", "--features", features] + profile
    proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    extra = ""
    for crate in filter(None, tree.split(",")):
        t = subprocess.run(["cargo", "tree", "--features", features, "-i", crate, "-e", "features", "--depth", "1", "--prefix", "none"],
                           cwd=work, capture_output=True, text=True)
        extra += f"\n--- cargo tree -i {crate} ---\n" + "\n".join((t.stdout + t.stderr).splitlines()[:60])
    cache.commit()
    extra = pinned + extra
    log = proc.stdout + proc.stderr + extra
    lines = [l for l in log.splitlines()
             if l.startswith(("error", "warning: unused")) or "Finished" in l or "= help" in l or "panicked" in l or "-->" in l]
    return f"exit={proc.returncode}\n" + "\n".join(lines[-40:]) + "\n--- tail ---\n" + "\n".join(log.splitlines()[-30:]) + extra


hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


def _run(model_id: str, ref_json: str, env_flags: str = "") -> str:
    import os
    import subprocess

    cache.reload()
    args = ["/cache/target/release/engine", model_id]
    if ref_json:
        with open("/tmp/ref.json", "w") as f:
            f.write(ref_json)
        args.append("/tmp/ref.json")
    env = dict(os.environ, **{k: "1" for k in filter(None, env_flags.split(","))})
    proc = subprocess.run(args, capture_output=True, text=True, env=env)
    hf_cache.commit()
    return f"exit={proc.returncode}\n{proc.stdout}\n--- stderr tail ---\n" + "\n".join(proc.stderr.splitlines()[-40:])


RUN_VOLUMES = {"/cache": cache, "/root/.cache/huggingface": hf_cache}


@app.function(image=image, gpu=["L4", "L40S"], timeout=30 * 60, volumes=RUN_VOLUMES)
def run(model_id: str = "Qwen/Qwen3-0.6B", ref_json: str = "") -> str:
    return _run(model_id, ref_json)


@app.function(image=image, gpu="L40S", timeout=45 * 60, volumes=RUN_VOLUMES)
def run_l40s(model_id: str = "Qwen/Qwen3-8B", ref_json: str = "", env_flags: str = "") -> str:
    return _run(model_id, ref_json, env_flags)


@app.local_entrypoint()
def main(features: str = "flashinfer", release: bool = True, model_id: str = "", skip_build: bool = False,
         ref_file: str = "", tree: str = "", precise: str = "", fresh_lock: bool = False, gpu: str = "L4",
         env_flags: str = ""):
    if not skip_build:
        out = build.remote(features, release, tree, precise, fresh_lock)
        print(out)
        if not out.startswith("exit=0"):
            return
    if model_id:
        ref = Path(ref_file).read_text() if ref_file else ""
        print(run_l40s.remote(model_id, ref, env_flags) if gpu == "L40S" else run.remote(model_id, ref))
