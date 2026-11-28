# WP-3 Technique Analysis: ArkVale (`arkvale`)

## 1. Verified Configuration & Empirical Snapshot

- **Authoritative Axis ([`docs/METRICS_AND_AXES.md`](../../../docs/METRICS_AND_AXES.md)):** **`SPATIAL`** (Host CPU DRAM full-page backup with GPU AABB page digests and query-driven on-demand page recall), with secondary `STRUCTURAL` page-selection mechanics.
  - *Axis Reconciliation Note:* `kv_bench` (`ArkValeTechnique.axes`) and the manifest label ArkVale as `["STRUCTURAL", "SPATIAL"]` (following the NeurIPS '24 paper's eviction framing). Under our Hub taxonomy (`docs/METRICS_AND_AXES.md`), because ArkVale preserves 100% of pages in host CPU DRAM and manages GPU-vs-CPU residency, its **primary production axis is `SPATIAL`**.
- **`kv_bench` Commit SHA:** `5fc1cea4ba64ac86a2da49edd8210a46753834fe` *(PR #59: query-scored AABB SDPA hook, vectorized digests, GPU-pinned exempt layers `0–1`, delta recall)*
- **Model & Hardware:** `Qwen/Qwen3-4B-Instruct-2507` (`cdbee75f17c01a7cc42f958dc650907174af0554`, Pure Dense GQA, BF16, $L=36, H_Q=32, H_{KV}=8, D=128$) on NVIDIA GeForce RTX 4090 (23.52 GiB physical, `16.0 GiB` process allocator cap), Slurm Job `30715`.
- **Manifest `technique_config` ([`run_manifest_wp3_arkvale.json`](../../../results/manifests/run_manifest_wp3_arkvale.json)):**
  - `cache_class`: `ArkValeCache`, `page_size`: `16`, `active_pages_budget`: `16` (`256` active tokens), `exempt_early_layers`: `2`, `block_size`: `16`.
- **Golden Artifact SHA-256s (`run_manifest_wp3_arkvale.json`):**
  - `arkvale_hardware_pytorch_16gb.json`: `8221490b81cebdf46a130185847960d06818483b84a1bcef48432116b485784a`
  - `arkvale_longmem_pytorch_16gb.json`: `725469d91ad60bf06967c2fd133901ff1c7e4899da6c446e32c05ba260009cad`
  - `arkvale_niah_pytorch_16gb.json`: `71b4ade3511c61942f45117c6e522b7c2dca98aa26252349333c0d317470a2e5`
  - `verify_arkvale_pytorch_16gb.json`: `b7698a22d83a5b741d8a18e4250d387b0080542635a1311afe2b1b0bf8ddf8dc`
  - `arkvale_grades.json`: `1787a380aeba7abdb2b300c3d1c17883209b4ceff0eed1b8a5a10cc24b6e8b7c`
  - `arkvale_judged.json`: `42fcbfc7c6189b8c86e05ece2dac04a0c1f3258c8470dfe1b6c1bfaba6de19f3`

### Hardware Sweep (`512 -> 32,768`, `SyntheticFiller` / PG-19)

Static `Qwen3-4B` BF16 weights + PyTorch runtime occupy **7,738.8 MiB (~7.56 GiB)** (`verify_baseline_pytorch_16gb.json` at `ctx=128`); net KV+aux VRAM below is `Peak Alloc - 7,738.8 MiB`.

| Context | Status | Peak Alloc (MiB) | Net KV+Aux (MiB) | Base Alloc (MiB) | Host RSS $\Delta$ (MiB) | TTFT (s) | TPOT (ms) | Base TPOT (ms) | Tok/s | PCIe B/tok | Physical Scalar Bits | Eff. Token Retention | PG-19 Top-1 | PG-19 KL Mean | PG-19 KL P99 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| 512 | `SUCCESS` | 7,881.2 | 142.5 | 7,911.7 | 1,736.7 | 0.043 | 43.10 | 17.36 | 23.20 | 9,118,060.25 | 16-bit BF16 | 100% CPU; 256 GPU (L2..35) | 0.0938 | 12.24481 | 15.4201 |
| 1,024 | `SUCCESS` | 8,046.7 | 307.9 | 8,133.0 | 2,044.9 | 0.077 | 43.70 | 16.91 | 22.88 | 13,566,444.25 | 16-bit BF16 | 100% CPU; 256 GPU (L2..35) | 0.1562 | 11.47935 | 15.4247 |
| 2,048 | `SUCCESS` | 8,311.9 | 573.1 | 8,566.9 | 2,643.1 | 0.187 | 44.73 | 16.61 | 22.36 | 19,983,084.25 | 16-bit BF16 | 100% CPU; 256 GPU (L2..35) | 0.0469 | 13.04192 | 15.4249 |
| 4,096 | `SUCCESS` | 8,353.2 | 614.4 | 8,854.9 | 3,772.4 | 0.472 | 46.23 | 16.67 | 21.63 | 37,711,084.25 | 16-bit BF16 | 100% CPU; 256 GPU (L2..35) | 0.1406 | 11.46088 | 15.2318 |
| 8,192 | `SUCCESS` | 8,419.2 | 680.4 | 9,431.0 | 5,998.3 | 1.332 | 47.03 | 17.12 | 21.26 | 94,272,748.25 | 16-bit BF16 | 100% CPU; 256 GPU (L2..35) | 0.1875 | 11.31897 | 15.4249 |
| 16,384 | `SUCCESS` | 8,649.0 | 910.3 | 10,616.8 | 10,493.6 | 4.231 | 50.29 | 20.47 | 19.89 | 319,056,108.25 | 16-bit BF16 | 100% CPU; 256 GPU (L2..35) | 0.0781 | 13.04578 | 15.4249 |
| 32,768 | `SUCCESS` | **9,281.2** | **1,542.4** | 13,224.9 | **19,450.1** | **14.802** | **54.80** | 29.17 | **18.25** | **1,192,875,244.25** | 16-bit BF16 | 100% CPU; 256 GPU (L2..35) | 0.0938 | 12.57779 | 15.4249 |

### Quality Summary (Verify, NIAH, LongMemEval)

- **Verify (`ctx=128`):** `top1_agreement = 1.0000`, `kl_div_mean = 0.0`, `kl_div_p99 = 0.0`.
- **NIAH (`2K` & `4K`, 10 depths `0.0..0.9`):** **10/10 (100%)** at `2048` and **10/10 (100%)** at `4096`.
- **LongMemEval (`150` samples, prompt length `192..22,400`, mean `5,967.9` tokens):**
  - `peak_alloc_mb = 8,869.0`, `host_rss_delta_mb = 9,901.6`, `ttft_sec = 1.043`, `tpot_mean_ms = 38.48`, `throughput_tok_per_sec = 929.47`.
  - **LongMem KL:** `top1_agreement = 0.4780`, `kl_div_mean = 7.59370`, `kl_div_p99 = 15.4249` (improved from `0.3836 / 8.99`).
  - **Overall Accuracy:** **97/150 (64.67%)** vs. Baseline **99/150 (66.00%)** (`-1.33 pp`, **+5.34 pp** jump after PR #59 — #2 across all WP-3 techniques!).
  - **Verdict Breakdown:** `PASS_FIDELITY_MAINTAINED: 92`, `BASELINE_LIMITATION: 46`, **`QUALITY_REGRESSION: 7`** *(halved from `14`!)*, `UNEXPECTED_GAIN: 5`.

| LongMemEval Capability | Baseline Acc | ArkVale Acc | Delta (pp) | `QUALITY_REGRESSION` Count |
|---|---:|---:|---:|---:|
| `fact_recall` | 90.00% | 93.33% | +3.33 | 0 *(+1 UG)* |
| `information_update` | 66.67% | 66.67% | 0.00 | 2 *(+2 UG)* |
| `relative_timeframe` | 30.00% | 30.00% | 0.00 | 1 *(+1 UG)* |
| `cross_session_reasoning` | 43.33% | 36.67% | -6.67 | 3 *(+1 UG)* |
| `unanswerable_refusal` | 100.00% | 96.67% | -3.33 | 1 |

![ArkVale Hardware Sweep](../../figures/wp3/arkvale/hardware_sweep.png)
![ArkVale KL Fidelity](../../figures/wp3/arkvale/kl_fidelity.png)
![ArkVale LongMemEval Diagnostics](../../figures/wp3/arkvale/longmem_diagnostics.png)

---

## 2. White-Box Component Decomposition (Post-PR #59)

| ID | Component | Source Location (`kv_bench@5fc1cea`) | Tensor Shapes & Dtypes (`Qwen3-4B`: $L=36, H_{KV}=8, D=128$) | Cadence |
|---|---|---|---|---|
| **C1** | **Vectorized Incremental AABB Page Digest Builder** | [`arkvale_cache.py`](https://github.com/khemingkapat/kv_bench/blob/5fc1cea4ba64ac86a2da49edd8210a46753834fe/src/techniques/arkvale_cache.py) (`_update_incremental_digests`) | **RESOLVED in PR #59:** Vectorized over full 16-token pages (eliminating the Python `for p` prefill loop); stores `gpu_b_min, gpu_b_max` `[B, 8, max_pages, 128]` BF16 on GPU. | Every `update()` for touched pages only. |
| **C2** | **Pinned CPU Backup Pool & GPU-Pinned Exempt Layers** | [`arkvale_cache.py`](https://github.com/khemingkapat/kv_bench/blob/5fc1cea4ba64ac86a2da49edd8210a46753834fe/src/techniques/arkvale_cache.py) | **RESOLVED in PR #59:** Exempt layers `0, 1` (`exempt_early_layers=2`) stay **pinned on GPU** (`~256 MiB` at 32K) instead of re-copying full 32K history from CPU every step; layers `2..35` back up in pinned BF16 `cpu_{k,v}_buf`. | Prefill + decode append. |
| **C3** | **Query-Driven GQA AABB Scoring & Delta Page Recall** | [`arkvale.py`](https://github.com/khemingkapat/kv_bench/blob/5fc1cea4ba64ac86a2da49edd8210a46753834fe/src/techniques/arkvale.py), [`arkvale_cache.py`](https://github.com/khemingkapat/kv_bench/blob/5fc1cea4ba64ac86a2da49edd8210a46753834fe/src/techniques/arkvale_cache.py) (`arkvale_sdpa_attention_forward`, `score_page_digests_gqa`) | **RESOLVED in PR #59:** SDPA hook passes real **`query_states`** with GQA max-pooling across 4 Q-heads per KV-head, maintains resident GPU page slots, and HtoD-fetches **only missing delta pages**. | Every decode step for layers `2..35` (`total_pages > 16`). |

---

## 3. 3-Way Causal Attribution (Pre- vs. Post-PR #59)

- **Quality & Latency Breakthrough (`[IMPLEMENTATION]` RESOLVED):**
  - **LongMemEval (`89/150 -> 97/150`, `64.67%`, `-1.33 pp` vs. Baseline):** Replacing `key_states` proxy with real GQA-max-pooled **`query_states`** via the SDPA hook halved `QUALITY_REGRESSION`s (`14 -> 7`), lifted `fact_recall` to `93.33%` (`0` QR), and lifted `cross_session_reasoning` by `+10.0 pp` (`26.7% -> 36.7%`, `7 -> 3` QR)—even with a tight `256`-token (`16`-page) active GPU budget!
  - **Decode TPOT (`102.92 -> 54.80 ms` at 32K, `1.88x` speedup; `38.48 ms` on LongMemEval):** Pinning exempt layers `0, 1` on GPU and recalling only delta pages made ArkVale the **fastest CPU-offloading technique at 32K** (`18.25 tok/s`, matching `QServe KV4`'s `54.81 ms`!).
  - **Prefill TTFT (`21.12 -> 14.80 s`):** Vectorizing `_update_incremental_digests` saved `6.31 s` at 32K.
- **VRAM (`9,281.2 MiB` total / `1,542.4 MiB` net KV+aux at 32K, `-3,943.7 MiB` vs. Baseline):**
  - **`[INTRINSIC]`** Pinning exempt layers `0, 1` on GPU added `~248 MiB` vs. pre-fix `9,032.9 MiB`, while still saving nearly **4.0 GiB** of VRAM at 32K.
- **Remaining Bottlenecks (`[INTRINSIC]` BF16 CPU Pool & Prefill Offload):**
  - **Host RSS (`19,450.1 MiB` = `19.0 GiB`):** Exceeds a 16 GB consumer host RAM cap because `cpu_{k,v}_buf` stores uncompressed BF16 K+V across power-of-2 pinned capacity buffers.
  - **PCIe Volume (`1,192.9 MB/tok` at 32K, down from `1,618.4 MB/tok`):** Driven by prefill DtoH offload + uncompressed BF16 delta fetches (`31,942` recalled pages across the sweep).
  - **WP-4 Implication:** Quantizing ArkVale's CPU backup pool to 4-bit INT4 (`g=32`) directly fixes both remaining `[INTRINSIC]` limits—cutting Host RSS `3.56x` (`19.0 GiB -> ~5.3 GiB`) and PCIe payload `3.56x`.

---

## 4. Paper vs. Replication Fidelity (Post-PR #59)

| Dimension | ArkVale Paper ([`arkvale.md`](../../../docs/literature_survey/techniques/arkvale.md)) | `kv_bench` Replication (`5fc1cea`, Post-PR #59) | Fidelity Verdict |
|---|---|---|---|
| **AABB Digest (`c ± r_mean`)** | Vectorized `c = 0.5*(min+max)`, `r_mean = mean(|k - c|)`. | Vectorized in `_update_incremental_digests`. | **Faithful (Fixed in PR #59).** |
| **Page Scoring Query** | Decode **`query` $q$** with GQA max-pool across query heads. | SDPA hook passes **`query_states`** into `score_page_digests_gqa`. | **Faithful (Fixed in PR #59).** |
| **Exempt Layers & Delta Recall** | GPU-resident exempt layers + delta-only page recall. | Layers `0, 1` pinned on GPU; resident page pool recalls delta pages only. | **Faithful (Fixed in PR #59).** |
| **Active Token Budget** | `2,048–4,096` tokens (`128–256` pages at `p=16`). | `active_pages_budget=16` (`256` tokens). | **Tightly budgeted** (yet already hits `64.67%`!). |

---

## 5. Component Verdicts (`KEEP` / `ADAPT` / `DROP`)

| Component | Verdict | Contribution | Failure Mode | Required Fix (if `ADAPT`) |
|---|---|---|---|---|
| **C1: Vectorized AABB Page Digests** | **`KEEP`** | Compact 16-token PagedAttention-aligned bounding boxes; drives `97/150 (64.7%)` accuracy. | None. | — |
| **C2: Pinned CPU Backup & GPU Exempt Layers** | **`ADAPT`** | Keeps VRAM at `9.28 GiB` and TPOT at `54.80 ms` at 32K. | Uncompressed BF16 CPU pool inflates Host RSS to `19.0 GiB` (exceeds 16 GB RAM). | Quantize CPU backup pool to `g=32` INT4 (`3.56x` smaller RSS & PCIe). |
| **C3: Query-Driven GQA Scoring & Delta Recall** | **`KEEP`** | Halved TPOT (`102.9 -> 54.8 ms`) and halved LongMemEval regressions (`14 -> 7`). | Prefill DtoH + BF16 delta traffic still `1.19 GB/tok` in eager PyTorch. | Pair with async CUDA copy stream + 4-bit page payloads in WP-4/vLLM. |
