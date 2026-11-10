# Agent-session inference engine

An LLM inference engine written from scratch in Rust, built for AI agents: programs that call the model, run a tool, and call the model again with a longer history. It keeps each agent's session in GPU memory between turns, shares common prompt parts (system prompt, tool definitions) across agents, and serves an OpenAI-compatible API.

Status: research prototype. One model (Qwen3-8B), one GPU type (NVIDIA L40S), greedy decoding.

## Results

24 real coding-agent sessions from [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) (791 turns, up to 30,000 prompt tokens per session) replayed against each engine. Each turn generates 256 tokens, then the agent waits a fixed time standing in for a tool call. Same model, same GPU, same KV cache size (173,200 tokens) for every engine. Two runs per configuration.

**Total time for all 24 sessions (lower is better):**

| | Tool wait 5 s | Tool wait 60 s |
|---|---|---|
| vLLM 0.33.0 | 2342 / 2333 s | 4106 / 4128 s |
| SGLang 0.5.23, LRU eviction | 2179 / 2154 s | 4069 / 4154 s |
| SGLang 0.5.23, TLRU eviction | 2121 / 2152 s | 4078 / 4093 s |
| **This engine** | **1868 / 1908 s** | **4083 / 4011 s** |
| This engine vs best baseline | **1.13x faster** | 1.01x (no real difference) |
| This engine vs vLLM | **1.24x faster** | 1.02x (no real difference) |

**Time to first token, median / 90th percentile (lower is better):**

| | Tool wait 5 s | Tool wait 60 s |
|---|---|---|
| vLLM 0.33.0 | 26.8, 24.9 s / 43.2, 43.6 s | 3.5, 3.8 s / 16.6, 17.0 s |
| SGLang 0.5.23, LRU | 29.2, 29.4 s / 44.0, 46.8 s | 6.1, 7.1 s / 18.4, 19.4 s |
| SGLang 0.5.23, TLRU | 29.3, 25.3 s / 45.8, 44.9 s | 6.3, 6.4 s / 18.1, 18.8 s |
| **This engine** | **5.1, 6.3 s / 26.0, 28.8 s** | 13.8, 11.5 s / 23.0, 21.1 s |

What this shows:

- When agents return quickly (5 s tool calls), keeping their sessions in memory pays off: 1.13x faster in total than the best baseline and about 5x faster to first token.
- When agents are away longer (60 s), all engines finish within about 2% of each other, and this engine is slower to first token.
- Raw speed is still behind: generating tokens is 15-21% slower per token than vLLM (25.1 / 27.8 / 29.7 ms vs 21.8 / 23.5 / 24.6 ms at 1 / 8 / 24 concurrent requests), and processing several long prompts at once is slower too.

Regenerate these tables from the raw results with `python summarize.py`.

### Limitations

- One model, one GPU, one load level (24 concurrent sessions), two tool-wait values.
- Tool time is a fixed wait. The dataset does not record real tool durations.
- vLLM and SGLang reject prompts over 32,768 tokens; this engine does not enforce that limit, so it processed 2-3 extra turns of one session. Its numbers include that extra work.
- The engine's own runs vary by up to 7.7% between identical runs (vLLM 0.5%), so differences of a few percent are not meaningful.

## What did not work

These were tried, measured against pass/fail criteria written down before the runs, and are kept in the history:

| Attempt | Result |
|---|---|
| Fixed 30 s pin of each session after a turn | 1.3x slower than vLLM at 60 s tool waits: the pin expires before the agent returns |
| Pin length learned per tool name | 1.6x slower than vLLM at 60 s: idle pinned sessions block new ones |
| Pin only when worth it, capped at 50% of memory | 3-4% slower than vLLM at 60 s |
| Prefix sharing, first version | Slower than vLLM at 5 s, due to a scheduling bug, fixed in the next commit (1c92959) |
| attention-rs fused RMSNorm | Faster, but further from an FP32 reference (one top-token change in 64 steps). Not used. |

## How it works

- **Sessions.** A request may carry a `session_id`. The engine keeps that session's KV cache after the request, so the next turn only computes new tokens.
- **Prefix sharing.** Every full 64-token block gets a hash chained from the previous block. Any request reuses the longest matching run of cached blocks from any session. In the benchmark all 24 sessions share a 5,636-token start (system prompt and tools), which would otherwise be stored 24 times.
- **Memory policy.** After a turn that ends in a tool call, a session's cache is pinned only if the learned wait for that tool is shorter than the time to recompute a full cache, and pinned sessions may hold at most half the memory. Released blocks stay reusable until the space is needed (least recently used first).
- **Admission.** New sessions are admitted only if every session in memory, plus a growth margin, still fits. Returning sessions go first.
- **Scheduling.** Long prompts are processed in chunks of 4,096 tokens per step. Steps that only generate tokens use a dedicated decode kernel.
- **Kernels.** Model code uses candle ([guoqingbao's fork](https://github.com/guoqingbao/candle) at rev 8ac335c) and [attention-rs](https://github.com/guoqingbao/attention.rs) (rev 3c258f0) for paged attention through FlashInfer.

## Correctness

- Teacher-forced against vLLM on the same tokens: the engine's top token matches at all 64 steps (`compare.py`).
- Against an FP32 Hugging Face reference: mean log-probability error 0.0056 (vLLM: 0.0043), same top token at all 64 steps (`fp32_ref.py`).
- Sessions: turn 2 on a live cache gives the same output as a fresh run over the full history.
- Prefix sharing: a request that reuses another request's cached prefix gives the same output as on an empty cache (`phase0/share_check.py`).

## Layout

| Path | What |
|---|---|
| `core/src/` | The engine (Rust): `model.rs` Qwen3, `kv.rs` paged cache with prefix sharing, `engine.rs` steps, `server.rs` API and scheduler, `chat.rs` chat template and tool-call parsing, `prof.rs` optional profiler |
| `modal_build.py` | Build on Modal (CUDA, no GPU) and run checks on a GPU |
| `bench.py` | All benchmarks, run inside one container per engine against localhost |
| `phase0/replay.py` | Agent-session replay client |
| `summarize.py` | Rebuilds the result tables from `data/m3/` |
| `data/` | Sampled sessions, reference outputs, all benchmark results |

## Reproduce

Requires a [Modal](https://modal.com) account. GPU runs cost money: one 24-session run is about 35 minutes (5 s waits) or 70 minutes (60 s waits) on an L40S.

```bash
modal run modal_build.py                                    # build the engine
modal run --detach bench.py --spawn --engine ours --tool-delay 5 \
    --env-flags ENGINE_PIN_POLICY=capped --tag mine          # this engine
modal run --detach bench.py --spawn --engine vllm --tool-delay 5 --tag mine
modal run --detach bench.py --spawn --engine sglang --tool-delay 5 --tag mine
modal volume get engine-results local_ours_c24_d5_mine.txt data/m3/
python summarize.py
```

The sampled sessions are in `data/sessions_64.jsonl`. To resample, download the dataset's `trajectories.parquet` into `data/` and run `python phase0/sample.py 64`.

## Credits and licenses

- This code: MIT.
- [attention-rs](https://github.com/guoqingbao/attention.rs) (MIT) and [candle](https://github.com/huggingface/candle) (Apache-2.0 / MIT).
- Qwen3-8B: Apache-2.0. Dataset: nebius/SWE-rebench-openhands-trajectories, CC-BY-4.0.
