# WP-3 Technique Analysis: ShadowKV (`shadowkv`)

## 1. Verified Configuration & Empirical Snapshot

- **Authoritative Axis ([`docs/METRICS_AND_AXES.md`](../../../docs/METRICS_AND_AXES.md)):** `SPATIAL` (Host CPU Value offloading + GPU low-rank Key storage) with interacting `STRUCTURAL` (pre-RoPE SVD + chunk landmarks) and `TEMPORAL` (inter-step chunk reuse cache) axes.
- **`kv_bench` Commit SHA:** `2a9338a2f750741f7765d79c1270f5579a9ffed0`
- **Model & Hardware:** `Qwen/Qwen3-4B-Instruct-2507` (`cdbee75f17c01a7cc42f958dc650907174af0554`, Pure Dense GQA, BF16, $L=36, H_Q=32, H_{KV}=8, D=128$) on NVIDIA GeForce RTX 4090 (23.52 GiB physical, `16.0 GiB` process allocator cap).
- **Manifest `technique_config` ([`run_manifest_wp3_shadowkv.json`](../../../results/manifests/run_manifest_wp3_shadowkv.json)):**
  - `cache_class`: `ShadowKVCache`, `svd_rank`: `160`, `chunk_size`: `8`, `topk_chunks`: `256`, `outlier_ratio`: `0.003` (defaults: `local_window_tokens=32`, `svd_threshold_tokens=1024`).
- **Golden Artifact SHA-256s (`run_manifest_wp3_shadowkv.json`):**
  - `shadowkv_hardware_pytorch_16gb.json`: `6a7bbf2b1eaee9f97363f519395cb8b4572d9f000c7626333e02bf19ff492c24`
  - `shadowkv_longmem_pytorch_16gb.json`: `ba3dfaab100b00c77e2a8e65689914e51fb6373f313f270747c8fbf0d28a584a`
  - `shadowkv_niah_pytorch_16gb.json`: `fa78e110f6c641d8d667145e9495e8ebb53624561fba4c3ffcea395b0caa3110`
  - `verify_shadowkv_pytorch_16gb.json`: `2fb21d75cad0de293ae6b976dd144fa02e1889a232678c7d0248952acdfc1627`
  - `shadowkv_judged.json`: `3190bda03afbf89d79923ccd12f23caaf58f75b28c19729aadd515328713b658`

### Hardware Sweep (`512 -> 32,768`, `SyntheticFiller` / PG-19)

Static `Qwen3-4B` BF16 weights + runtime overhead consume **~7,839.7 MiB (~7.66 GiB)**.

| Context | Status | Peak Alloc (MiB) | Net KV+Aux (MiB) | Base Alloc (MiB) | Host RSS $\Delta$ (MiB) | TTFT (s) | TPOT (ms) | Base TPOT (ms) | Tok/s | PCIe B/tok | Physical Scalar Bits | Eff. Token Retention | PG-19 Top-1 | PG-19 KL Mean | PG-19 KL P99 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| 512 | `SUCCESS` | 7,911.7 | 72.0 | 7,911.7 | 1,682.5 | 0.038 | 20.90 | 17.36 | 47.84 | 201.75* | 16-bit BF16 | 100% dense GPU | 1.0000 | 0.00000 | 0.0000 |
| 1,024 | `SUCCESS` | 8,232.5 | 392.8 | 8,133.0 | 2,450.7 | 0.070 | 39.85 | 16.91 | 25.10 | 2,593,806.38 | 16-bit BF16 | 100% stored (SVD+CPU V) | 0.2500 | 10.63813 | 15.4247 |
| 2,048 | `SUCCESS` | 8,566.9 | 727.2 | 8,566.9 | 2,457.0 | 0.152 | 39.50 | 16.61 | 25.31 | 5,244,844.62 | 16-bit BF16 | 100% stored (SVD+CPU V) | 0.4062 | 7.94308 | 15.4249 |
| 4,096 | `SUCCESS` | 8,918.2 | 1,078.5 | 8,854.9 | 2,674.4 | 0.344 | 43.74 | 16.67 | 22.86 | 9,982,381.62 | 16-bit BF16 | 100% stored; ~2,144+ active | 0.6875 | 4.14985 | 15.1387 |
| 8,192 | `SUCCESS` | 9,821.0 | 1,981.3 | 9,431.0 | 3,238.3 | 0.829 | 54.13 | 17.12 | 18.47 | 19,339,027.31 | 16-bit BF16 | 100% stored; ~2,144+ active | 1.0000 | 0.05126 | 0.8673 |
| 16,384 | `SUCCESS` | 11,468.3 | 3,628.6 | 10,616.8 | 3,942.8 | 2.237 | 66.32 | 20.47 | 15.08 | 36,767,998.94 | 16-bit BF16 | 100% stored; ~2,144+ active | 0.8750 | 1.56821 | 14.4418 |
| 32,768 | `SUCCESS` | **14,233.0** | **6,393.3** | 13,224.9 | 5,325.1 | 6.802 | 85.29 | 29.17 | 11.73 | 67,241,955.19 | 16-bit BF16 | 100% stored; ~2,184+ active | 0.0781 | 12.90901 | 15.4249 |

*\*`ctx=512` `201.75 B/tok` is harness profiler overhead (`svd_threshold_tokens=1024`, zero offload).*

### Quality Summary (Verify, NIAH, LongMemEval)

- **Verify (`ctx=128`):** `top1_agreement = 1.0000`, `kl_div_mean = 0.0`, `kl_div_p99 = 0.0`.
- **NIAH (`2K` & `4K`, 10 depths `0.0..0.9`):** **10/10 (100%)** at `2048` and **10/10 (100%)** at `4096`.
- **LongMemEval (`150` samples, prompt length `192..22,400`, mean `5,967.9` tokens):**
  - `peak_alloc_mb = 12,141.8`, `host_rss_delta_mb = 5,655.4`, `ttft_sec = 0.605`, `tpot_mean_ms = 74.56`, `throughput_tok_per_sec = 621.76`.
  - **LongMem KL:** `top1_agreement = 0.6790`, `kl_div_mean = 4.58414`, `kl_div_p99 = 15.4249` (best KL among all 4 spatial techniques).
  - **Overall Accuracy:** **96/150 (64.00%)** vs. Baseline **99/150 (66.00%)** (`-2.00 pp`, best among WP-3 spatial techniques).
  - **Verdict Breakdown:** `PASS_FIDELITY_MAINTAINED: 96`, `BASELINE_LIMITATION: 51`, **`QUALITY_REGRESSION: 3`**, `UNEXPECTED_GAIN: 0`.

| LongMemEval Capability | Baseline Acc | ShadowKV Acc | Delta (pp) | `QUALITY_REGRESSION` Count |
|---|---:|---:|---:|---:|
| `fact_recall` | 90.00% | 90.00% | 0.00 | 0 |
| `information_update` | 66.67% | 66.67% | 0.00 | 0 |
| `relative_timeframe` | 30.00% | 30.00% | 0.00 | 0 |
| `cross_session_reasoning` | 43.33% | 33.33% | -10.00 | 3 |
| `unanswerable_refusal` | 100.00% | 100.00% | 0.00 | 0 |

![ShadowKV Hardware Sweep](../../figures/wp3/shadowkv/hardware_sweep.png)
![ShadowKV KL Fidelity](../../figures/wp3/shadowkv/kl_fidelity.png)
![ShadowKV LongMemEval Diagnostics](../../figures/wp3/shadowkv/longmem_diagnostics.png)

---

## 2. White-Box Component Decomposition

| ID | Component | Source Location | Tensor Shapes & Dtypes (`Qwen3-4B`: $L=36, H_{KV}=8, D=128$) | Cadence |
|---|---|---|---|---|
| **C1** | **Inverse RoPE + Per-Head SVD Key Decomposition** | [`shadowkv_cache.py#L90-L97`](file:///home/khemi/workspace/kv_bench/src/techniques/shadowkv_cache.py#L90-L97), [`#L161-L180`](file:///home/khemi/workspace/kv_bench/src/techniques/shadowkv_cache.py#L161-L180) (`invert_rope`, `perform_svd_key_decomposition`) | `r = min(160, S, 128) = 128`. Produces `key_a` `[B, 8, S, 128]` + `key_b` `[B, 8, 128, 128]` BF16 via `torch.linalg.svd(..., full_matrices=False)` in FP32. | Once at end of prefill (`prune()`, when `S > 1024`). |
| **C2** | **Chunk Landmarks, Outlier Detection & GPU Anchors** | [`shadowkv_cache.py#L99-L158`](file:///home/khemi/workspace/kv_bench/src/techniques/shadowkv_cache.py#L99-L158), [`#L347-L397`](file:///home/khemi/workspace/kv_bench/src/techniques/shadowkv_cache.py#L347-L397) (`compute_chunk_landmarks`, `detect_outlier_chunks`) | `landmarks` `[B, 8, ceil(S/8), 128]` BF16; `gpu_anchor_{keys,values}` `[B, 8, O*8 + 32, 128]` BF16 (`O=13` chunks = `104` tokens at 32K). | Once at `prune()`, plus `torch.cat` per decode token. |
| **C3** | **Pinned CPU Value Bank & Dual-Stream Sparse Retrieval** | [`shadowkv_cache.py#L359-L377`](file:///home/khemi/workspace/kv_bench/src/techniques/shadowkv_cache.py#L359-L377), [`#L399-L513`](file:///home/khemi/workspace/kv_bench/src/techniques/shadowkv_cache.py#L399-L513), [`shadowkv.py#L290-L306`](file:///home/khemi/workspace/kv_bench/src/techniques/shadowkv.py#L290-L306) (`resolve_sparse_kv`, `shadowkv_sdpa_attention_forward`) | `cpu_values` `[B, 8, S, 128]` BF16 pinned; `cached_sparse_{keys,values}` `[B, 8, N_miss_accum, 128]` BF16; `copy_stream` overlaps HtoD `val_slice` with `a_slice @ key_b` + `apply_rope`. | Every decode step inside SDPA hook (`query.shape[-2] == 1`). |

---

## 3. 3-Way Causal Attribution

- **VRAM Regression vs. Baseline (`[IMPLEMENTATION]`):** At `32K`, ShadowKV's `peak_alloc_mb` reaches **`14,233.0 MiB`** (`16,124.0 MiB` reserved!), **exceeding** the uncompressed baseline (`13,224.9 MiB`) by `+1,008.1 MiB`. Two implementation bugs cause this:
  1. **`C1` Per-Head Rank-128 SVD:** `perform_svd_key_decomposition` clamps `r = min(160, S, 128) = 128` per KV head (`[B, 8, S, 128]` + `[B, 8, 128, 128]`), which is **full rank** and stores *more* bytes than the original keys, plus `landmarks` (`+12.5%`).
  2. **`C3` Unbounded Monotonic Sparse Cache:** `resolve_sparse_kv` (`shadowkv_cache.py:493-502`) appends `cached_sparse_keys` and `cached_sparse_values` with `torch.cat` and **never evicts** previously fetched chunks, accumulating `cached_sparse` buffers alongside `key_a`, `key_b`, and `landmarks`.
- **Host RSS (`[INTRINSIC]`):** Storing only `V` in pinned host RAM (`cpu_values`) keeps 32K host RSS delta at **`5,325.1 MiB`** (half of InfiniGen's K+V host pool, comfortably within 16 GB).
- **PCIe & TPOT (`[INTRINSIC]` + `[IMPLEMENTATION]`):** `C3`'s dual-stream `copy_stream` + temporal chunk caching cuts 32K PCIe traffic to **`67.24 MB/tok`** (`24.5x` lower than FlexGen-KV's `1.65 GB/tok`), yielding the **fastest 32K decode among all 4 spatial techniques** (`85.29 ms` TPOT, `11.73 tok/s`). Still, `85.29 ms` is `2.92x` slower than baseline (`29.17 ms`) due to Python-level `set(topk_indices.flatten().tolist())` + CPU `index_select` + `torch.cat` per layer per step (`[IMPLEMENTATION]`) and PCIe Gen4 latency (`[INTRINSIC]`).
- **Quality Preservation (`[INTRINSIC]` + `[IMPLEMENTATION]`):** `96/150 (64.00%)` LongMemEval with **zero** regressions on `fact_recall`, `information_update`, `relative_timeframe`, and `unanswerable_refusal`, and only `3 QUALITY_REGRESSION`s in `cross_session_reasoning` (`-10.00 pp`). This stems from lossless BF16 Values + full-rank Keys + outlier/local anchors (`C2`). PG-19 KL (`0.05` at 8K, `1.57` at 16K, `12.91` at 32K; $N=1$ high variance, `15.4249` clamp) reflects single-sample sensitivity when `topk_chunks=256` (`2,048` tokens = `6.25%` of 32K) misses a non-landmark token; 2K/4K NIAH (`10/10`) does not test 32K sparsity.

---

## 4. Paper vs. Replication Fidelity

| Dimension | ShadowKV Paper ([`shadowkv.md`](../../../docs/literature_survey/techniques/shadowkv.md)) | `kv_bench` Replication (`shadowkv.py` / `shadowkv_cache.py`) | Fidelity Verdict |
|---|---|---|---|
| **SVD Shape & Rank** | Concatenated heads `[b, s, h_kv*d]` (`M=1024`), `r=160` $\rightarrow$ **`6.4x` key compression**. | Per-head `[b*h_kv, s, d]` (`d=128`), `r = min(160, s, 128) = 128` $\rightarrow$ **`0x` compression (full rank!)**. | **Critical Divergence (`[IMPLEMENTATION]` Bug).** |
| **Temporal Working Set Cache** | Fixed-size GPU LRU/ring buffer (~60% reuse, bounded VRAM). | Unbounded `torch.cat` set (`cached_sparse_{k,v}` never evicted). | **Critical Divergence (`[IMPLEMENTATION]` Bug):** causes 14.2 GiB VRAM bloat at 32K. |
| **Dual-Stream Overlap** | CUDA stream 1 fetches V, stream 2 reconstructs K (`A @ B` + RoPE). | Implemented via `self.copy_stream` + `wait_stream` in `resolve_sparse_kv`. | **Faithful.** |
| **Chunk Landmarks & Outliers** | `chunk_size=8`, cosine-sim outlier detection (`0.2%–0.3%`), GQA max-pool. | Exact match (`chunk_size=8`, `outlier_ratio=0.003`, `amax` GQA pool). | **Faithful.** |

---

## 5. Component Verdicts (`KEEP` / `ADAPT` / `DROP`)

| Component | Verdict | Contribution | Failure Mode | Required Fix (if `ADAPT`) |
|---|---|---|---|---|
| **C1: Pre-RoPE SVD Key Compression** | **`ADAPT`** | Preserves key fidelity without quantization noise. | Per-head `min(160, s, 128)=128` gives **zero** compression and increases VRAM. | Reshape pre-RoPE keys to **`[B, S, H_kv * D]`** (`[B, S, 1024]`) before SVD so `r=160` yields true `6.4x` compression (matching paper). |
| **C2: Landmarks, Outliers & GPU Anchors** | **`KEEP`** | Accurate GQA chunk screening + `0` regressions on 4/5 LongMemEval tasks. | None. | — |
| **C3: CPU Value Bank + Dual-Stream Sparse Cache** | **`ADAPT`** | Lowest PCIe volume (`67.2 MB/tok`) and best 32K TPOT (`85.29 ms`) in Wave C. | `cached_sparse_{k,v}` never evicts, causing `14.2 GiB` VRAM OOM-risk at 32K. | Add **fixed-capacity LRU/ring eviction** (`<= topk_chunks * 8` tokens) on `cached_sparse_{k,v}` and vectorize CPU gather. |
