# WP-3 Technical Analysis: H2O (Heavy-Hitter Oracle)

**Authoritative Axis (`docs/METRICS_AND_AXES.md`):** `STRUCTURAL`  
**`kv_bench` Pinned Commit:** `2a9338a2f750741f7765d79c1270f5579a9ffed0`  
**Target Model:** `Qwen/Qwen3-4B-Instruct-2507` (`Pure Dense GQA`, $L=36$, $H_Q=32$, $H_{KV}=8$, $D=128$, `bfloat16`)  
**Hardware Cap:** 16.0 GiB process allocator cap on NVIDIA RTX 4090 (23.52 GiB physical VRAM, `effective_gpu_utilization = 0.5783`)

---

## 1. Verified Configuration & Empirical Snapshot

### 1.1 Manifest Provenance & `technique_config`
From [`results/manifests/run_manifest_wp3_h2o_pytorch.json`](../../../results/manifests/run_manifest_wp3_h2o_pytorch.json):
- **Manifest `technique_config`:**
  - `cache_class`: `H2OCache`
  - `heavy_hitter_tokens`: `1024`
  - `recent_window_tokens`: `1024`
  - `block_size`: `16`
  - `effective_cache_tokens`: `2048` ($1024 + 1024$)
  - `effective_cache_blocks`: `128`
- **VERIFIED Hub Artifact SHA-256 Checksums:**
  - `results/runs/wp3/h2o_hardware_pytorch_16gb.json`: `6de1820c969f977719779fdff6f9c7713a7ccbad7b932877527257dd5994444c`
  - `results/runs/wp3/h2o_longmem_pytorch_16gb.json`: `de2d961ee332c28f87f50eadbe278629f239e8ff4bffe370ee49eb1e231f1c3e`
  - `results/runs/wp3/h2o_niah_pytorch_16gb.json`: `1c7c735501be64e264431a1692966206cc8d3402b216bd3f01787eb0633ffddf`
  - `results/runs/wp3/verify_h2o_pytorch_16gb.json`: `f6188415cda3daa91f633ae12c43aa3fe3e57dded98ab9ef7bb5acd00fcd2671`
  - `results/grades/h2o_grades.json`: `efe86b1742f542b8c7a15700ddb9789ca49927efc3e926bbde029957f5478c0b`
  - `results/grades/h2o_judged.json`: `806e9166b8117e5e2e9c222008ad94759ca0330f98896f312d236ce72a68e985`

### 1.2 Hardware Scaling Sweep (`512 -> 32,768` Tokens, `SyntheticFiller`)
Static `Qwen3-4B` BF16 weights + runtime at `ctx=128` (`verify_h2o_pytorch_16gb.json`) is **`7,741.47 MB` (~7.56 GiB)**. At `ctx=32,768`, H2O's `Peak Alloc` plateaus at **`9,724.65 MB`** (net KV+aux **`~1,983.2 MB`**) vs. Baseline's `13,224.90 MB` (`~5,483.4 MB` KV+aux) — saving `3,500.3 MB` vs. baseline, but consuming **`1,012.9 MB` more** than StreamingLLM (`8,711.71 MB`) due to auxiliary float32 attention-score materialization.

| `ctx` | Peak Alloc (MB) | Peak Reserved (MB) | Warm TTFT (s) | TPOT Mean / P95 (ms) | Throughput (tok/s) | Host RSS Δ (MB) | Memcpy (B/tok) | Physical Scalar Bits | Effective Token Retention |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **512** | 8,003.91 | 8,118.0 | 0.0543 | 23.95 / 23.95 | 41.75 | 4,038.98 | 201.75 | 16-bit BF16 | 100.0% (576 / 576) |
| **1,024** | 8,372.14 | 8,466.0 | 0.1298 | 23.44 / 23.44 | 42.66 | 4,040.51 | 329.75 | 16-bit BF16 | 100.0% (1,088 / 1,088) |
| **2,048** | 9,281.25 | 9,348.0 | 0.3320 | 44.52 / 44.52 | 22.46 | 4,740.52 | 1,307,095.50 | 16-bit BF16 | 96.97% (2,048 / 2,112) |
| **4,096** | 9,724.43 | 9,988.0 | 0.9178 | 45.37 / 45.37 | 22.04 | 4,884.51 | 1,353,687.50 | 16-bit BF16 | 49.23% (2,048 / 4,160) |
| **8,192** | 9,724.46 | 9,988.0 | 2.0850 | 46.12 / 46.12 | 21.68 | 5,176.67 | 1,446,871.50 | 16-bit BF16 | 24.81% (2,048 / 8,256) |
| **16,384** | 9,724.53 | 9,990.0 | 4.4431 | 45.59 / 45.59 | 21.93 | 5,747.91 | 1,633,239.50 | 16-bit BF16 | 12.45% (2,048 / 16,448) |
| **32,768** | 9,724.65 | 9,990.0 | 9.1098 | 45.88 / 45.88 | 21.80 | 6,873.70 | 2,005,975.50 | 16-bit BF16 | 6.24% (2,048 / 32,832) |

### 1.3 Micro Fidelity (Verify, PG-19 / Hardware Sweep KL) & Macro Retrieval (NIAH, LongMemEval)
- **Verify (`ctx=128`, 16 decode tokens):** `top1_agreement = 1.0`, `kl_div_mean = 0.0`, `kl_div_p99 = 0.0`.
- **Hardware Sweep KL (`SyntheticFiller`, $N=1$ per length, 64 decode tokens):**
  - `512` & `1,024`: `top1 = 1.0`, `kl_mean = 0.0`, `kl_p99 = 0.0`.
  - `2,048`: `top1 = 0.2031`, `kl_mean = 10.97825`, `kl_p99 = 15.4249`.
  - `4,096`: `top1 = 0.7969`, `kl_mean = 2.66784`, `kl_p99 = 15.1387`.
  - `8,192`: `top1 = 0.2969`, `kl_mean = 9.81611`, `kl_p99 = 15.4249`.
  - `16,384`: `top1 = 0.1406`, `kl_mean = 11.88897`, `kl_p99 = 15.4249`.
  - `32,768`: `top1 = 0.2344`, `kl_mean = 10.47673`, `kl_p99 = 15.4249`.
- **NIAH (`2K` & `4K` depth grid, passcode `"849204"`):**
  - **`2,048` ctx:** **`100%` (`1.0`)** across all 10 depths (`0.0–0.9`).
  - **`4,096` ctx:** **`20%` (`0.2`)** — `0.0` at depths `0.0–0.7` (`needle_distance` `1,255–4,096`); **`1.0`** only at depths `0.8–0.9` (`needle_distance` `444–850` $\le 1,024$ recent window). Notice that at depths `0.6–0.7` (`1,255–1,661` tokens back), H2O scores `0.0` whereas StreamingLLM scored `1.0` because H2O's `recent_window_tokens` is `1,024` (vs. StreamingLLM's `2,048`) and the needle token did not accumulate enough attention during `SyntheticFiller`/text prefill to beat early heavy hitters.
- **LongMemEval (`150` native-length samples, `192–22,400` prompt tokens, median `5,859`, `h2o_judged.json`):**
  - **Overall Accuracy:** **`80 / 150 = 53.33%`** vs. Baseline **`99 / 150 = 66.0%`** ($\Delta = -12.67\text{ pp}$, **`+9.33 pp` over StreamingLLM**).
  - **Causal Verdicts:** `PASS_FIDELITY_MAINTAINED = 77`, `BASELINE_LIMITATION = 48`, **`QUALITY_REGRESSION = 22`**, `UNEXPECTED_GAIN = 3`.
  - **LongMemEval KL & Retention:** `top1_agreement = 0.4746`, `kl_div_mean = 7.6479`, `kl_div_p99 = 15.4249`, `Physical Scalar Bits = 16-bit BF16`, `Effective Token Retention = 43.44%` (`2.30×` compression).
  - **Length-Tier Accuracy:** `short = 75.0%`, `medium = 52.78%`, `long = 18.18%`.

| LongMemEval Capability | Baseline (`N=30` each) | H2O (`N=30` each) | $\Delta$ (pp) | `QUALITY_REGRESSION` Count |
|---|---:|---:|---:|---:|
| **Fact Recall** | 90.0% (27/30) | 86.7% (26/30) | -3.3 | 1 |
| **Information Update** | 66.7% (20/30) | 46.7% (14/30) | -20.0 | 6 |
| **Relative Timeframe** | 30.0% (9/30) | 26.7% (8/30) | -3.3 | 2 |
| **Cross-Session Reasoning** | 43.3% (13/30) | 16.7% (5/30) | -26.7 | 10 |
| **Unanswerable Refusal** | 100.0% (30/30) | 90.0% (27/30) | -10.0 | 3 |

![H2O Hardware Sweep](../../figures/wp3/h2o/hardware_sweep.png)
![H2O KL Fidelity](../../figures/wp3/h2o/kl_fidelity.png)
![H2O LongMemEval Diagnostics](../../figures/wp3/h2o/longmem_diagnostics.png)

---

## 2. White-Box Component Decomposition

| ID | Component | Source File, Class/Method & Lines | Tensor Shapes & Dtypes (`Qwen3-4B`, BF16) | Execution Cadence |
|---|---|---|---|---|
| **C1** | **SDPA Monkey-Patch Hook (`h2o_sdpa_attention_forward`)** | `src/techniques/h2o.py`: `_install_pytorch_h2o_hooks` (L425–460) | Intercepts `query` `(1, 32, q_len, 128)`, `key` `(1, 8, kv_len, 128)`, `value` `(1, 8, kv_len, 128)` (`bfloat16`) after `_ORIG_SDPA`. | Every layer ($36\times$) on every prefill chunk and decode step. |
| **C2** | **Duplicate Attention & GQA Score Pooling (`record_h2o_attention_scores`)** | `src/techniques/h2o_cache.py`: `record_h2o_attention_scores` (L37–82) | Prefill (`q_len > 1`): `k_rep` `(1, 32, kv_len, 128)` + `scores`/`attn` `(1, 32, q_len, kv_len)` in **`float32`**. Decode (`q_len == 1`): `(1, 8, 4, 1, kv_len)` $\to$ `mean(dim=2)` $\to$ `(1, 8, kv_len)` `float32`. | Every layer ($36\times$) on every prefill chunk and decode step. |
| **C3** | **Cumulative Score & KV Buffer (`H2OLayer`)** | `src/techniques/h2o_cache.py`: `H2OLayer` (L84–179) | `keys`/`values`: `(1, 8, S_active, 128)` `bfloat16`; `cumulative_scores`: `(1, 8, S_active)` `float32`. | `update` + `record_scores` every step; `prune` at chunk end & every decode step (`q_len == 1`). |
| **C4** | **CPU-Synced Top-K Boundary & `index_select` (`compute_h2o_boundaries` + `prune`)** | `src/techniques/h2o.py`: `compute_h2o_boundaries` (L65–158); `h2o_cache.py`: `H2OLayer.prune` (L142–172) | Averages `cumulative_scores` across heads `(kv_len,)`, calls `.tolist()` (D2H sync), Python `sorted()`, then `torch.tensor(retained_indices)` (H2D) + `torch.index_select`. | Every chunk end + every decode step when `seq_len > 2048` ($36 \times 64 = 2,304$ syncs per 64-tok sample). |

---

## 3. 3-Way Causal Attribution

- **VRAM (`9,724.65 MB` at `32K`, `+1,013 MB` over StreamingLLM):** **`[IMPLEMENTATION]`** — `C3` bounds `S_active` to `2,048` (`~295 MB` KV), saving `3.5 GB` vs. baseline, but `C2` materializes full `float32` `(1, 32, 1024, 3072)` attention matrices (`~403 MB` per tensor + `k_rep`) during chunked prefill *alongside* SDPA, adding `~1.01 GB` peak VRAM.
- **Host RSS (`4,039 -> 6,874 MB`) & PCIe Memcpy (`1.31–2.01 MB/tok` at `2K–32K`):** **`[IMPLEMENTATION]`** — Discounting the `8,266 B/tok` harness floor, the massive `1.31–2.01 MB/tok` PCIe traffic and `+2.8 GB` Host RSS growth are caused entirely by `C4` calling `.tolist()` (D2H) and `torch.tensor(retained_indices)` (H2D) `2,304` times per 64-token decode across 36 layers whenever `seq_len > 2048`.
- **TTFT (`9.11 s` at `32K`, slower than Baseline `6.98 s`) & TPOT (`45.88 ms` vs. `20.95 ms` StreamingLLM):** **`[IMPLEMENTATION]`** — `C2` recomputes $Q K^\top$ in `float32` *after* `_ORIG_SDPA` already computed attention, doubling prefill compute and pushing 32K TTFT to `9.11 s`. During decode at `ctx >= 2048`, `C4`'s 36-layer D2H/H2D Python-sort round-trips add `~22 ms/tok`, doubling TPOT from `23.95 ms` (`512` ctx) to `44.5–46.1 ms` (`2K–32K`).
- **KL Divergence (`kl_div_mean = 7.65`, `top1 = 0.4746` on LongMemEval):** **`[INTRINSIC]` + `[HARDWARE/GQA]`** — H2O preserves `3.6×` higher `top1_agreement` than StreamingLLM (`0.4746` vs. `0.1318`) because `C3` retains high-attention anchor/entity tokens across the prompt. However, `C4` pools scores across all 8 KV heads (`mean(dim=(0,1))`) rather than per-KV-head, losing head-specific specialization.
- **NIAH (`100%` at `2K`, `20%` at `4K`):** **`[INTRINSIC]`** — At `4K`, H2O's `1,024`-token recent window only covers depths `0.8–0.9` (`444–850` tokens). The needle at depths `0.0–0.7` is not heavily attended during prefill, so `C4` evicts it.
- **LongMemEval (`53.33%`, `22` `QUALITY_REGRESSION`s):** **`[INTRINSIC]` + `[IMPLEMENTATION]`** — Retaining heavy hitters cuts `QUALITY_REGRESSION`s by `40%` vs. StreamingLLM (`22` vs. `37`), preserving `fact_recall` (`86.7%`, only `1` regression) and `relative_timeframe` (`26.7%`, `2` regressions). However, **`information_update` drops to `46.7%` (`6` regressions)** because older outdated facts accumulated high prefill attention and outlived newer corrections, and **`cross_session_reasoning` drops to `16.7%` (`10` regressions)** (e.g., `bb7c3b45`: *"Wrong savings amount ($100 instead of $300)"*) because secondary facts without high cumulative prefill attention are permanently evicted.

---

## 4. Paper vs. Replication Fidelity

| Dimension | Paper (`Zhang et al., NeurIPS 2023`, `docs/literature_survey/techniques/h2o.md`) | `kv_bench` PyTorch Implementation (`h2o.py`, `h2o_cache.py`) | Fidelity Verdict |
|---|---|---|---|
| **Dual-Pool Budget** | $K$ heavy hitters + $K$ recent tokens ($50\%/50\%$ split). | `heavy_hitter_tokens=1024`, `recent_window_tokens=1024` (`2048` total). | **Exact Match** |
| **GQA Score Pooling** | Paper evaluated MHA (OPT, LLaMA-1). | `record_h2o_attention_scores` pools Q heads per KV group (`mean(dim=2)`), **but** `compute_gqa_h2o_attention` / `compute_h2o_boundaries` (L128–129) reduces further via `mean(dim=(0,1))` across all KV heads, losing per-head eviction. | **Partial Divergence** |
| **In-Place Eviction & Compute** | Pre-allocated static circular buffer with $O(1)$ in-place CUDA overwrite; hooked into attention without duplicate matmul. | `C2` recomputes full `float32` $QK^\top$ on top of SDPA; `C4` runs `.tolist()` + Python `sorted()` + H2D `torch.tensor()` + `index_select` on all 36 layers every decode step. | **Severe Implementation Bottleneck** |

---

## 5. Component Verdicts (`KEEP` / `ADAPT` / `DROP`)

| Component | Verdict | Contribution | Failure Mode | Exact Fix Prerequisite (if `ADAPT`) |
|---|---|---|---|---|
| **C1: SDPA Hook** | **`ADAPT`** | Enables attention-score extraction in `transformers` SDPA. | Forces duplicate attention computation in `C2`. | Replace post-hoc hook with a custom single-pass Triton/FlashAttention score-accumulation kernel or lightweight last-query proxy. |
| **C2: `record_h2o_attention_scores`** | **`ADAPT`** | Accurately pools GQA query heads into `(B, H_KV, kv_len)` scores. | Doubles prefill FLOPs and adds `~1.01 GB` `float32` peak VRAM. | Compute chunk scores using `bfloat16` or subsample/pool query block norms instead of full `(1, 32, 1024, kv_len)` `float32` materialization. |
| **C3: `H2OLayer` Cumulative Buffer** | **`KEEP`** | Raises LongMemEval accuracy `+9.33 pp` over StreamingLLM (`53.3%` vs `44.0%`). | Permanent eviction still causes `22` regressions (`information_update` & `cross_session`). | Pair with `SPATIAL` offload/quantization in WP-4/WP-6 so low-score tokens are demoted rather than deleted. |
| **C4: CPU-Synced `compute_h2o_boundaries`** | **`DROP` (replace)** | Selects top-$k$ heavy hitters. | `2,304` D2H/H2D syncs per sample cause `2.0 MB/tok` memcpy, `6.87 GB` RSS, and `2×` TPOT (`45.9 ms`). | Replace `.tolist()` + Python `sorted()` with pure GPU `torch.topk` per KV head (`(1, 8, k_hh)`) and in-place buffer scatter. |
