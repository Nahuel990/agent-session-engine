# attention-rs integration spec (Qwen3 dense)

Source: read from xinfer (MIT) and attention.rs (MIT) at attention-rs rev 3c258f0, candle fork rev 8ac335c, on 2026-11-09. X = xinfer/src, A = attention.rs/src. Items marked UNKNOWN were not verified.

## Features
- Metal: `attention-rs/metal` + `attention-rs/metal-flash` (xinfer Cargo.toml:90). Plain `metal` takes the legacy paged path that needs a 5D x-packed key cache.
- CUDA: we use `flashinfer`. Without `flash`, forward uses FlashInfer only when `flashinfer_metadata` is Some and the GQA group is supported (Qwen3-0.6B group 2, 8B group 4 are supported; A/lib.rs:831-930). Otherwise it falls to the legacy path (5D cache).

## KV cache (X/utils/kvcache_allocator.rs:1617-1716)
- Per layer `(k, v)`, `Tensor::empty` on the device, model dtype (BF16).
- Flat NHD layout for metal-flash / flashinfer: `k, v: (num_blocks, block_size, num_kv_heads, head_dim)`.
- block_size: 32 on Metal, 64 otherwise (X/utils/config.rs:694). FlashInfer page size = `k_cache.dims4()[1]`.

## Per step inputs (X/core/runner.rs)
- `slot_mapping`: i64 `[num_new_tokens]` (Metal requires i64).
  - Prefill: for blocks `num_cached_blocks..num_blocks`, start = `block_table[i]*bs` (+ `num_cached_tokens % bs` on the first), take `min(remaining, bs)`.
  - Decode: `last_block*bs + (len-1)%bs`.
- `block_tables`: u32 `[num_seqs, max_blocks]`, right-padded with 0. Prefill: full table. Decode: first `ceil(len/bs)` blocks.
- `context_lens`: u32 `[num_seqs]`. Prefill: `num_cached + num_new`. Decode: `seq_len`.
- `positions`: i64 `[num_tokens]` = `num_cached..num_cached+n`.

## InputMetadata (A/lib.rs:274-292)
Prefill (fresh and cached-prefix are the same struct; fresh = num_cached 0):
- `is_prefill: true`, `block_tables: Some` (always; with None, `sdp_prefill` runs NON-causal), `context_lens: Some`
- `cu_seqlens_q`: u32 `[B+1]` cumulative new tokens; `cu_seqlens_k`: u32 `[B+1]` cumulative `cached+new`
- `max_seqlen_q` = max new, `max_seqlen_k` = `max_context_len` = max `cached+new`
- `seqlens: Some(cu_seqlens_q[1..])` (last-token index = seqlens-1)
- `block_tables_host`, `context_lens_host`: Some; `is_mtp_verify: false`; `mamba_slot_mapping`: UNKNOWN for dense (likely None)

Decode: `is_prefill: false`, `cu_seqlens_q/k: None`, `max_seqlen_q/k: 0`, `max_context_len` = max len, `seqlens: None`.

FlashInfer metadata (X/core/runner.rs:1418-1525 prefill, :1652-1712 decode): `indices` = active blocks concatenated (active = ceil((cached+new)/bs)), `indptr` running count, `last_len = (L-1)%bs+1`, `kv_len_arr_host`. Prefill also sets `batch_indices`, `positions` (u32 per token), `total_num_rows = cu_q.last` and calls `prefill_plan(...)` up front; decode calls `decode_plan(...)` before forward (:1104-1119).

## Attention layer (X/models/layers/attention.rs)
- `PagedAttention::new(heads, head_dim, 1/sqrt(hd), Some(kv_heads), sliding_window, device, None, fp8_kv)`.
- Read `attention_bias` from config.json (Qwen3: false).
- q `(S,Hq,D)`, k/v `(S,Hkv,D)`; per-head RMSNorm (weight `[head_dim]`) on q and k; RoPE via `FusedRope::apply_inplace(q, k, cos, sin, positions, false)`, cos/sin `(max_pos, hd/2)` in model dtype, all contiguous.
- `attn.forward(&q, &k, &v, None, Some(kc), Some(vc), &meta, None)` returns `(S,Hq,D)` → reshape `(S, ())` → o_proj.

## Model (X/models/qwen3.rs)
- Weights: `model.embed_tokens`, `model.layers.{i}`, `model.norm`, `lm_head` (tied → embed_tokens).
- RMSNorm weights loaded F32, activations cast to F32 and back, eps = `rms_norm_eps`.
- MLP: packed `[gate; up]` → `attention_rs::silu_and_mul::silu_and_mul(&packed, intermediate)` → down.
- Logits: `lm_head(x)` → F32. Dtype BF16 on Metal and CUDA (F16 below compute capability 8).

## Metal caveats
- metal-flash accepts F16/BF16 cache only. No mask on Metal.
- metal-flash shortcut: a single fresh sequence of >= 1024 tokens runs SDPA internally in F16 (A/lib.rs:1419-1466).

## UNKNOWN
- `mamba_slot_mapping` for dense models; CUDA slot_mapping dtype enforcement; CUDA decode `append_kv_cache` behavior when `batch_indices`/`positions` are None (A/flashinfer.rs:276-297).
