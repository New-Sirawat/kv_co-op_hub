# WP-3 Consolidated Pattern Report: White-Box Component Synthesis & Falsifiable Candidate Directions

> **Source Benchmark Commit:** `khemingkapat/kv_bench@2a9338a2f750741f7765d79c1270f5579a9ffed0`  
> **Evaluated Model:** `Qwen/Qwen3-4B-Instruct-2507` ($L=36,\; H_Q=32,\; H_{KV}=8,\; D=128$, BF16 unmodified weights; theoretical uncompressed KV = `144 KiB/tok`, `4.50 GiB` @ 32K)  
> **Hardware Cap & Engine:** RTX 4090 capped at `16.0 GiB` (`gpu4090`), PyTorch Native SDPA  
> **Canonical Dataset:** 150 quantile-sampled LongMemEval questions (30 per capability, `ctx=32768`, offline semantic judge in `results/grades/*_judged.json`) + `512–32,768` PG-19 hardware sweep + `2K/4K` NIAH diagnostic  
> **Scope Boundary:** This report synthesizes empirical patterns from WP-3 and defines **4 falsifiable candidate hypotheses** ($\le 5$ advisor bound) for subsequent micro-benchmarking. It does **not** lock in an untested WP-4 runtime specification.

---

## 1. Executive Summary, Provenance & Overview Visuals

### 1.1 Why Component-Level Analysis Supersedes Whole-Technique Rankings
Black-box WP-3 rankings conflate **algorithmic merit** with **PyTorch wrapper artifacts** and **8-KV-head GQA scaling effects**:
- **Collapsed wrappers with sound or salvageable ideas:** `HeadQ` (`0.0%` LongMemEval) collapsed because `headq_cache.py` fitted its PCA basis on first-chunk `k_residual` instead of offline calibration Queries; `EpiCache` (`5.33%`) collapsed because `epicache_cache.py` discarded `sensitivity_budget` and evicted recent/decode tokens without a local window; `CacheGen` (`44.0%`) emitted `935 MB/tok` PCIe traffic due to CPU-staged tensor reallocation and omitted arithmetic coding.
- **Unfused compute / full-BF16 materialization penalties:** `QServe KV4` (`59.92 ms` TPOT), `TurboQuant` (`864.12 ms` TPOT), and `HeadQ` (`80.89 ms` TPOT) all dequantize their entire cached sequence back to BF16 before calling HF SDPA, hiding true quantized memory-bandwidth gains.
- **VRAM inversion from implementation bugs:** `ShadowKV` (`14,233 MiB`) and `InfiniGen` (`13,910 MiB`) exceeded the uncompressed baseline (`13,225 MiB`) not because selective offloading is intrinsically larger than BF16, but because `shadowkv_cache.py` clamps per-head SVD rank to `min(160, S, 128) = 128` (zero compression) and never evicts `cached_sparse_{k,v}`, while `infinigen_cache.py` holds full prefill `gpu_{k,v}` alongside `proxy_k_buf` and double-allocates `20.0 GiB` of pinned host buffers.

### 1.2 Master 12-Entry Telemetry & Net KV+Aux VRAM Table (32K Hardware Sweep + 150q LongMemEval)

Fixed static `Qwen3-4B` BF16 weights + PyTorch CUDA runtime occupy **7,738.8 MiB** (**~7.56 GiB**) at 128 tokens ([`verify_baseline_pytorch_16gb.json`](../../results/baselines/verify_baseline_pytorch_16gb.json)). Subtracting this unified `7,738.8 MiB` (`~7,739 MiB`) static floor across all 12 reports isolates **Net KV + Aux VRAM** (baseline 32K net KV+aux = `5,486.1 MiB`), enabling direct extrapolation to `8 GiB` consumer GPUs.

| Technique | Authoritative Axis (`METRICS_AND_AXES.md`) | Physical Scalar Bits | Effective Token Retention | 32K Peak Alloc (MiB) | Est. Net KV+Aux (MiB) | 32K Host RSS Δ (MiB) | 32K TTFT (s) | 32K TPOT (ms) | 32K Memcpy (MB/tok) | LongMem KL / Top-1 | NIAH (2K / 4K) | LongMemEval Semantic | Regressions | Detailed Report |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---|
| **PyTorch SDPA Baseline** | — | `16.0` (BF16) | `1.00` | `13,224.9` | `5,486` | `1,082.4` | `6.978` | `29.17` | `0.000` | — | — | **99/150 (66.00%)** | `0` | [`run_manifest_baseline_pytorch_16gb.json`](../../results/manifests/run_manifest_baseline_pytorch_16gb.json) |
| **ShadowKV** | `SPATIAL` (+ `STRUCTURAL`, `TEMPORAL`) | `16.0` (BF16) | `1.00` *(CPU V)* | `14,233.0` | `6,494` | `5,325.2` | `6.802` | `85.29` | `67.24` | `4.58 / 0.679` | `100% / 100%` | **96/150 (64.00%)** | `3` | [`shadowkv.md`](techniques/shadowkv.md) |
| **FlexGen-KV** | `SPATIAL` (+ `STRUCTURAL` 4-bit) | `4.5` (4b+scale) | `1.00` *(CPU)* | `8,888.9` | `1,150` | `6,564.6` | `10.205` | `229.57` | `1,646.96` | `10.25 / 0.300` | `100% / 100%` | **91/150 (60.67%)** | `14` | [`flexgen_kv.md`](techniques/flexgen_kv.md) |
| **InfiniGen** | `SPATIAL` (+ `TEMPORAL`) | `16.0` (BF16) | `1.00` *(CPU)* | `13,910.0` | `6,171` | `20,490.6` | `15.372` | `219.68` | `1,829.89` | `7.35 / 0.493` | `100% / 100%` | **91/150 (60.67%)** | `13` | [`infinigen.md`](techniques/infinigen.md) |
| **QServe KV4** | `STRUCTURAL` | `4.5` (4b+scale) | `1.00` | `10,762.7` | `3,024` | `3,628.6` | `7.239` | `59.92` | `0.008*` | `7.89 / 0.455` | `100% / 100%` | **89/150 (59.33%)** | `13` | [`qserve_kv4.md`](techniques/qserve_kv4.md) |
| **ArkVale** | `SPATIAL` *(reconciled from `STRUCTURAL`)* | `16.0` (BF16) | `1.00` *(CPU)* | `9,032.9` | `1,294` | `16,613.6` | `21.116` | `102.92` | `1,618.38` | `8.99 / 0.384` | `100% / 100%` | **89/150 (59.33%)** | `14` | [`arkvale.md`](techniques/arkvale.md) |
| **H2O** | `STRUCTURAL` | `16.0` (BF16) | `0.0625` *(2048/32K)* | `9,724.7` | `1,986` | `6,873.7` | `9.110` | `45.88` | `2.01` | `7.65 / 0.475` | `100% / 20%` | **80/150 (53.33%)** | `22` | [`h2o.md`](techniques/h2o.md) |
| **TurboQuant** | `STRUCTURAL` | `4.375` (`K:4.25, V:4.5`) | `1.00` | `10,294.4` | `2,555` | `7,528.9` | `20.627` | `864.12` | `0.013*` | `11.29 / 0.226` | `100% / 100%` | **80/150 (53.33%)** | `20` | [`turboquant.md`](techniques/turboquant.md) |
| **StreamingLLM** | `STRUCTURAL` | `16.0` (BF16) | `0.0626` *(2052/32K)* | `8,711.7` | `973` | `1,326.5` | `2.905` | `20.95` | `0.008*` | `12.98 / 0.132` | `100% / 40%` | **66/150 (44.00%)** | `37` | [`streamingllm.md`](techniques/streamingllm.md) |
| **CacheGen** | `STRUCTURAL` *(manifest: `SPATIAL`)* | `6.392` (8/4/2+anc) | `1.00` | `10,600.2` | `2,861` | `4,535.8` | `12.555` | `82.87` | `935.02` | `12.71 / 0.117` | `40% / 80%` | **66/150 (44.00%)** | `36` | [`cachegen.md`](techniques/cachegen.md) |
| **EpiCache** | `STRUCTURAL` *(manifest: `SPATIAL`)* | `16.0` (BF16) | `~0.031` *(1024/32K)* | `8,567.2` | `828` | `2,845.7` | `2.877` | `30.60` | `0.44` | `13.51 / 0.062` | `0% / 0%` | **8/150 (5.33%)** | `91` | [`epicache.md`](techniques/epicache.md) |
| **HeadQ** | `STRUCTURAL` | `3.25` (`K:2.375, V:4.125`) | `1.00` | `10,108.8` | `2,370` | `4,655.7` | `7.717` | `80.89` | `0.008*` | `14.76 / 0.014` | `0% / 0%` | **0/150 (0.00%)** | `99` | [`headq.md`](techniques/headq.md) |

*\*Note: `8,266 B/tok` (`~0.008 MB/tok`) in hardware sweeps is a baseline PyTorch instrumentation floor (`73.8 + 0.25 * ctx` B/tok), not technique offloading.*

### 1.3 Staged Overview Figures

![WP-3 Pareto VRAM vs Quality](../figures/wp3/overview/pareto_vram_vs_quality.png)

![WP-3 Decode Latency Overview](../figures/wp3/overview/tpot_overview_32k.png)

![WP-3 LongMemEval Ranking](../figures/wp3/overview/longmem_overview.png)

![WP-3 Per-Question Outcome Matrix](../figures/wp3/overview/longmem_question_heatmap.png)

---

## 2. Master Atomic Component Matrix

All 40 atomic `C#` sub-components extracted across the 11 `analysis/wp3/techniques/*.md` profiles from `kv_bench/src/techniques/` are consolidated into 27 mechanism rows below, separating **`[INTRINSIC]`**, **`[IMPLEMENTATION]`**, and **`[HARDWARE/GQA]`** effects.

| Mechanism Class | Source Technique | Component | Axis | Causal Tag(s) | Empirical Effect | Verdict |
|---|---|---|---|---|---|---|
| **Token Retention / Eviction** | `streamingllm` | `C1`/`C2`: 4-token attention sink + 2048-token sliding window | `STRUCTURAL` | `[INTRINSIC]` | `20.95 ms` TPOT, `973 MiB` net KV, but `37` regressions (`40%` 4K NIAH) | `KEEP` *(hot tier)* |
| **Token Retention / Eviction** | `streamingllm` | `C3`: Absolute `position_ids` (no cache-relative RoPE re-indexing) | `STRUCTURAL` | `[IMPLEMENTATION]` | Works $\le 32\text{K}$; breaks extrapolation beyond `max_position_embeddings` | `ADAPT` |
| **Token Retention / Eviction** | `h2o` | `C3`: Cumulative attention Heavy-Hitter + recent window (`1024+1024`) | `STRUCTURAL` | `[INTRINSIC]` | `+9.33 pp` LongMemEval over StreamingLLM (`53.33%`), still `22` regressions | `ADAPT` |
| **Token Retention / Eviction** | `h2o` | `C1`/`C2`: Duplicate `float32` $QK^\top$ recomputation hook | `STRUCTURAL` | `[IMPLEMENTATION]` | `+1.01 GiB` VRAM & `9.11 s` TTFT penalty | `DROP` |
| **Token Retention / Eviction** | `h2o` | `C4`: `.tolist()` + Python `sorted()` boundary sync per step | `STRUCTURAL` | `[IMPLEMENTATION]` | `2.01 MB/tok` memcpy, `6.87 GiB` RSS, `45.88 ms` TPOT | `DROP` |
| **Token Retention / Eviction** | `epicache` | `C1`: Per-layer sensitivity budget (`1 - cosine_sim`) | `STRUCTURAL` | `[IMPLEMENTATION]` | Computed then discarded due to HF shared 4D mask | `ADAPT` |
| **Token Retention / Eviction** | `epicache` | `C2`: Key-L2 prefill eviction without recent window | `STRUCTURAL` | `[IMPLEMENTATION]` | Catastrophic collapse (`5.33%` LongMemEval, `0%` NIAH) | `DROP` |
| **Token Retention / Eviction** | `epicache` | `C3`/`C4`: `.tolist()` decode sync + unimplemented episode routing | `STRUCTURAL` | `[IMPLEMENTATION]` | `442 KB/tok` D2H sync; dead `route_query_to_episode` | `DROP` |
| **Outlier Smoothing / Rotation** | `qserve_kv4` | `C1`: Channel-wise `smooth_k` outlier migration | `STRUCTURAL` | `[INTRINSIC]` / `[IMPLEMENTATION]` | Enables `100%` 2K/4K NIAH & `86.7%` `fact_recall` at 4-bit; online chunk-0 fit drifts slightly | `ADAPT` |
| **Outlier Smoothing / Rotation** | `turboquant` | `C1`: Unfused Python-loop `fp32` Walsh-Hadamard Transform (FWHT) | `STRUCTURAL` | `[INTRINSIC]` / `[IMPLEMENTATION]` | Disperses outliers without calibration, but `fp32` loop causes `864.12 ms` TPOT (`29.6×`) | `ADAPT` |
| **Outlier Smoothing / Rotation** | `headq` | `C2`: Query PCA basis fitted on first-chunk `k_residual` | `STRUCTURAL` | `[IMPLEMENTATION]` | Wrong subspace (`k_residual` $\ne Q$) causes `0.0%` collapse (`99/99` regressions) | `DROP` |
| **Outlier Smoothing / Rotation** | `infinigen` | `C1`: Random QR orthogonal rotation for 30% key proxy | `SPATIAL` | `[IMPLEMENTATION]` | Random QR does not concentrate outliers like calibration-$Q$ SVD | `ADAPT` |
| **Quantization / Residual Coding** | `qserve_kv4` | `C2`: Group-wise (`g=32`) symmetric INT4 K/V (`4.50 b/scalar`) | `STRUCTURAL` | `[INTRINSIC]` | Best quantizer quality (`59.33%` LongMemEval, `13` regressions) | `KEEP` |
| **Quantization / Residual Coding** | `turboquant` | `C2`/`C3`: 4-bit Beta Lloyd-Max K + 3-bit Lloyd-Max + 1-bit QJL V | `STRUCTURAL` | `[INTRINSIC]` | `100%` 2K/4K NIAH, `53.33%` LongMemEval (`4.375 b/scalar`) | `ADAPT` |
| **Quantization / Residual Coding** | `headq` | `C1`/`C3`: Uncentered 2-bit K (`amax/1.5`) + rank-4 sidecode + 4-bit V | `STRUCTURAL` | `[INTRINSIC]` / `[IMPLEMENTATION]` | Uncentered 2-bit grid + broken PCA destroys attention routing | `ADAPT` |
| **Quantization / Residual Coding** | `cachegen` | `C1`/`C2`: Anchor-relative ($G=10$) post-RoPE delta + 8/4/2-bit tiers | `STRUCTURAL` | `[INTRINSIC]` / `[IMPLEMENTATION]` | Post-RoPE 2-bit deep-layer deltas destroy `fact_recall` (`-43.3 pp`); lacks AC | `ADAPT` |
| **Quantization / Residual Coding** | `flexgen_kv` | `C1`: Group-wise (`g=64`) 4-bit quantization on CPU-offloaded K/V | `STRUCTURAL` | `[INTRINSIC]` | Bounds 32K Host RSS to `6.56 GiB` with `14` regressions (`60.67%`) | `ADAPT` |
| **Quantization / Residual Coding** | `qserve_kv4` / `turboquant` / `headq` / `cachegen` | Full-sequence BF16 dequantization before HF SDPA (`torch.cat` in TQ) | `STRUCTURAL` | `[IMPLEMENTATION]` | `2.05×–2.9×` TPOT penalty + transient `~4.5 GiB` BF16 materialization | `DROP` |
| **CPU Tiering & Indexing** | `flexgen_kv` | `C2`/`C3`: GPU recent window + pinned CPU pool + full-seq HtoD gather | `SPATIAL` | `[IMPLEMENTATION]` | Caps VRAM (`8.89 GiB`), but `cpu_delegation=False` streams `1.65 GB/tok` (`229.6 ms` TPOT) | `KEEP` (`C2`) / `DROP` (`C3`) |
| **CPU Tiering & Indexing** | `shadowkv` | `C1`: Per-head pre-RoPE SVD (`min(160, S, 128)=128`) | `STRUCTURAL` | `[IMPLEMENTATION]` | Zero compression (`r=128` on `D=128`), inflating VRAM | `ADAPT` |
| **CPU Tiering & Indexing** | `shadowkv` | `C2`: Chunk landmarks + `0.3%` outlier channels + GPU anchors | `SPATIAL` / `TEMPORAL` | `[INTRINSIC]` | Best recall (`64.00%`, only `3` regressions, `0` on 4/5 tasks) | `KEEP` |
| **CPU Tiering & Indexing** | `shadowkv` | `C3`: CPU Value bank + un-evicted `cached_sparse_{k,v}` | `SPATIAL` | `[IMPLEMENTATION]` / `[HARDWARE/GQA]` | Lowest offload PCIe (`67.2 MB/tok`), but un-evicted sparse cache inflates VRAM to `14.23 GiB` | `ADAPT` |
| **CPU Tiering & Indexing** | `infinigen` | `C2`: Double-allocated pinned CPU + staging buffers | `SPATIAL` | `[IMPLEMENTATION]` | `20.49 GiB` Host RSS (exceeds `16 GiB` consumer RAM) | `DROP` |
| **CPU Tiering & Indexing** | `infinigen` | `C3`: Speculative block scoring (`key_states` as `q_proxy`) + sync gather | `SPATIAL` / `TEMPORAL` | `[IMPLEMENTATION]` | `1.83 GB/tok` PCIe, `219.7 ms` TPOT, `verify` top-1 drops to `0.375` | `ADAPT` |
| **CPU Tiering & Indexing** | `arkvale` | `C1`/`C3`: 16-token AABB page digests + top-$k$ page recall (`key_states` proxy) | `SPATIAL` | `[IMPLEMENTATION]` | Caps VRAM (`9.03 GiB`), `59.33%` acc, but Python loop `21.1 s` TTFT & re-fetches all 16 pages/step | `ADAPT` |
| **CPU Tiering & Indexing** | `arkvale` | `C2`: Pinned CPU page bank + layers `0, 1` full-history exempt re-copy | `SPATIAL` | `[IMPLEMENTATION]` | `16.61 GiB` Host RSS & `1.62 GB/tok` PCIe | `ADAPT` |
| **CPU Tiering & Indexing** | `cachegen` | `C3`: CPU-staged `_ensure_capacity` reallocation | `STRUCTURAL` | `[IMPLEMENTATION]` | `~120` DtoH+HtoD round-trips (`935 MB/tok` memcpy, `12.56 s` TTFT) | `DROP` |

---

## 3. Cross-Technique Pattern Synthesis (Answering Q1–Q5)

### Q1 (Retention vs. Amnesia): Which GPU-resident token selection mechanisms reduce decode latency/VRAM, and where does non-local retrieval break?
- **Only permanent eviction cuts decode compute and VRAM simultaneously:** `StreamingLLM` (`2,052` GPU tokens) achieves the lowest latency (`2.905 s` TTFT, `20.95 ms` TPOT — `1.39×` faster than baseline) and `973 MiB` net KV VRAM.
- **Any permanent eviction breaks non-local retrieval:** On `4K` NIAH, `StreamingLLM` drops to `40%` (`0%` beyond `2048` tokens) and `H2O` drops to `20%`. On LongMemEval, `StreamingLLM` suffers `37` regressions (`cross_session_reasoning` collapses from `43.3%` to `3.3%`), and even `H2O`'s heavy-hitter retention still suffers `22` regressions (`information_update` drops `-20.0 pp`, `cross_session_reasoning` drops `-26.7 pp`).
- **Takeaway:** Sink + sliding window (`StreamingLLM`) is the cleanest **hot GPU tier** (`20.95 ms` TPOT), but middle tokens **must be demoted (quantized and/or offloaded)** rather than deleted.

### Q2 (Compression Floor vs. Implementation Collapses): What structural compression actually holds fidelity on `Qwen3-4B`, once implementation bugs are separated?
- **4-bit group quantization with channel smoothing is the empirically proven floor:** `QServe KV4` (`4.50 b/scalar`) preserves `100%` 2K/4K NIAH and `59.33%` LongMemEval (`13` regressions, matching `ArkVale` and within `1.34 pp` of `FlexGen-KV`/`InfiniGen`).
- **Sub-4-bit and delta failures were driven by `[IMPLEMENTATION]` bugs + `[INTRINSIC]` sensitivity:**
  - `HeadQ` (`0.0%`) failed from fitting PCA on `k_residual` instead of calibration Queries + uncentered 2-bit `amax/1.5` rounding (`[IMPLEMENTATION]` + `[INTRINSIC]`).
  - `CacheGen` (`44.0%`, `-43.33 pp` `fact_recall`) failed from computing $G=10$ deltas *post-RoPE* (rotary phase inflates delta range) and quantizing deep layers to 2-bit (`[INTRINSIC]`), plus CPU-staged reallocation (`[IMPLEMENTATION]`).
  - `TurboQuant` (`53.33%`, `100%` NIAH) proved rotation + QJL preserves needle lookup without calibration, but `864.12 ms` TPOT is an `[IMPLEMENTATION]` artifact of Python-loop `fp32` `fwht` + `torch.cat`.
- **Shared bottleneck across all 4 quantizers:** Full-sequence BF16 dequantization before HF SDPA (`2.05×–2.9×` TPOT penalty and transient `~4.5 GiB` BF16 allocation at 32K). In-register fused dequantization (Triton/CUDA) is mandatory to realize 4-bit speedups.

### Q3 (Selective Recall vs. Bulk Offloading): How do GPU-side relevance proxies compare to bulk CPU swapping?
- **Bulk swapping (`FlexGen-KV`) hits the PCIe wall:** Streaming all `31,744` offloaded 4-bit tokens back to GPU on every decode step consumes `1.65 GB/tok` PCIe and `229.57 ms` TPOT (`7.87×` slower than baseline), even with `4.5`-bit compression.
- **Selective chunked recall (`ShadowKV`) cuts PCIe by `24.5×` and preserves near-baseline quality:** Using chunk landmarks + `0.3%` outlier channels to fetch only `topk_chunks=256` (`2,048` tokens) reduces PCIe to `67.24 MB/tok`, improves TPOT to `85.29 ms`, and achieves **`96/150 (64.00%)`** with only **`3` regressions** (`0` regressions on `fact_recall`, `information_update`, `relative_timeframe`, `unanswerable_refusal`).
- **Why `InfiniGen` and `ArkVale` underperformed `ShadowKV` on PCIe:** Both passed **`key_states` instead of `query_states`** as `q_proxy` (`[IMPLEMENTATION]` bug), and `ArkVale` re-fetched all 16 active pages plus full 32K exempt layers `0–1` every step (`1.62 GB/tok`).

### Q4 (VRAM Inflation & Aux-Buffer Accounting): What net KV+aux budget is achievable on 16 GB and 8 GB GPUs?
- **Separating bugs from true `[HARDWARE/GQA]` overhead:**
  - `Qwen3-4B` has $H_{KV}=8$ KV heads (`4.50 GiB` BF16 KV at 32K). `ShadowKV`'s `14,233 MiB` (`+1,008 MiB` over baseline) and `InfiniGen`'s `13,910 MiB` (`+685 MiB`) were caused by **`[IMPLEMENTATION]` bugs**: `shadowkv_cache.py` clamping per-head SVD to `r=128` (zero compression) + un-evicted `cached_sparse_{k,v}`, and `infinigen_cache.py` keeping full prefill `gpu_{k,v}` alongside `proxy_k_buf` + `20.5 GiB` double CPU buffers.
  - True `[HARDWARE/GQA]` byte accounting shows that **4-bit GPU quantization (`QServe KV4`)** uses `~3.02 GiB` net KV+aux (`10.76 GiB` total), while **selective CPU offloading (`ArkVale` / `FlexGen-KV` / fixed `ShadowKV`)** holds net KV+aux to **`~1.15–1.30 GiB`** (`8.89–9.03 GiB` total on 16 GB).
- **8 GB RTX 4060 Extrapolation:** Since `Qwen3-4B` BF16 weights alone take `~7.55 GiB`, 32K context on an `8 GB` GPU requires either a downscaled 1.7B–3B BF16 profile (`~3.4–6.0 GiB` weights) + `~1.2 GiB` selective-offload KV tier, or strict VRAM capping via `SPATIAL` CPU tiering.

### Q5 (5-Capability & Per-Question Failure Signatures Across 150 Questions)

| Method | `fact_recall` (30) | `information_update` (30) | `relative_timeframe` (30) | `cross_session_reasoning` (30) | `unanswerable_refusal` (30) | Total Regressions (vs. 99 Base Pass) |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline (BF16)** | `90.0%` | `66.7%` | `30.0%` | `43.3%` | `100.0%` | `0` |
| **ShadowKV** | `90.0%` (`0` reg) | `66.7%` (`0` reg) | `30.0%` (`0` reg) | `33.3%` (`3` reg) | `100.0%` (`0` reg) | `3` |
| **FlexGen-KV** | `90.0%` (`1` reg) | `66.7%` (`3` reg) | `26.7%` (`1` reg) | `30.0%` (`6` reg) | `90.0%` (`3` reg) | `14` |
| **InfiniGen** | `83.3%` (`2` reg) | `60.0%` (`3` reg) | `30.0%` (`1` reg) | `40.0%` (`4` reg) | `90.0%` (`3` reg) | `13` |
| **QServe KV4** | `86.7%` (`1` reg) | `66.7%` (`1` reg) | `23.3%` (`3` reg) | `23.3%` (`7` reg) | `96.7%` (`1` reg) | `13` |
| **ArkVale** | `86.7%` (`1` reg) | `60.0%` (`3` reg) | `30.0%` (`1` reg) | `26.7%` (`7` reg) | `93.3%` (`2` reg) | `14` |
| **H2O** | `86.7%` (`1` reg) | `46.7%` (`6` reg) | `26.7%` (`2` reg) | `16.7%` (`10` reg) | `90.0%` (`3` reg) | `22` |
| **TurboQuant** | `73.3%` (`5` reg) | `53.3%` (`4` reg) | `20.0%` (`3` reg) | `23.3%` (`7` reg) | `96.7%` (`1` reg) | `20` |
| **StreamingLLM** | `70.0%` (`6` reg) | `60.0%` (`4` reg) | `10.0%` (`8` reg) | `3.3%` (`12` reg) | `76.7%` (`7` reg) | `37` |
| **CacheGen** | `46.7%` (`13` reg) | `50.0%` (`5` reg) | `16.7%` (`5` reg) | `20.0%` (`9` reg) | `86.7%` (`4` reg) | `36` |

- **Cross-Session Reasoning is the universal stress test:** `cross_session_reasoning` accounts for **54% (`7/13`)** of `QServe KV4`'s regressions, **50% (`7/14`)** of `ArkVale`'s, **45% (`10/22`)** of `H2O`'s, and **100% (`3/3`)** of `ShadowKV`'s. Multi-hop synthesis across dispersed sessions is far more sensitive to KV noise or top-$k$ sparsity than single-fact recall.
- **Eviction vs. Preservation split:** Of the `16` baseline-passing questions where *both* `StreamingLLM` and `H2O` fail, **`ShadowKV` passes `14/16` (`87.5%`), `QServe KV4` passes `8/16` (`50.0%`), and both pass `8/16` jointly**—confirming those failures are caused by middle-token deletion rather than intrinsic model limits.
- **Shared hard cluster (`10` questions):** Only `10` baseline-passing questions fail across $\ge 3$ of the top-5 methods (`bb7c3b45`, `2e6d26dc`, `f0e564bc`, `d23cf73b`, `a08a253f`, `00ca467f`, `6071bd76`, `9bbe84a2`, `gpt4_213fd887`, `gpt4_372c3eed_abs`), concentrated in `cross_session_reasoning` (`6/10`).

---

## 4. Bounded Shortlist of Falsifiable Candidate Hypotheses (4 Candidates)

To avoid premature WP-4 design without empirical validation, the patterns above yield **4 distinct, testable candidate combinations** ($\le 5$ advisor limit). Each specifies its mandatory `kv_bench` bug fixes before testing and concrete falsification gates.

### Candidate 1: `Hot-Sink/Window + 4-Bit SmoothQuantized GPU Tier` (`STRUCTURAL` Baseline)
- **Borrowed Components & Axes:** `StreamingLLM` sink+window (`C1` BF16 hot tokens, e.g., `4 + 512`) + `QServe KV4` RoPE-paired `smooth_k` & group-wise INT4 (`C1`, `C2`, `STRUCTURAL`).
- **Implementation Prerequisites:** Replace full-BF16 dequantization before SDPA with a fused Triton/CUDA INT4 attention kernel (or chunked dequant) and apply RoPE-paired offline $\lambda_i = \lambda_{i+D/2}$ calibration.
- **Core Hypothesis:** Keeping recent/sink tokens in BF16 while storing older in-VRAM tokens in 4-bit SmoothINT4 eliminates the `cross_session_reasoning` (`-20 pp`) and `relative_timeframe` (`-6.7 pp`) drift of pure 4-bit quantization while cutting 32K KV VRAM to `~1.3 GiB` with zero PCIe traffic.
- **Quantitative Falsification Gate:** Falsified if 32K LongMemEval `< 94/150` (`62.7%`), Net KV+aux VRAM `> 1,800 MiB`, or 32K TPOT `> 35 ms/tok` (vs. `29.17 ms` baseline).

### Candidate 2: `Fixed ShadowKV (Cross-Head SVD Key Index + Evictable Sparse Cache + CPU V)` (`SPATIAL` + `STRUCTURAL` + `TEMPORAL`)
- **Borrowed Components & Axes:** `ShadowKV` pre-RoPE SVD key proxy (`C1`, `STRUCTURAL`), landmarks + `0.3%` outlier channels (`C2`, `TEMPORAL`), and CPU Value bank with LRU-bounded GPU sparse cache (`C3`, `SPATIAL`).
- **Implementation Prerequisites:** (1) Fix `shadowkv_cache.py` SVD reshape from `[B*H_kv, S, D]` (`r=128` no-op) to `[B, S, H_kv*D]` (`r=160`, true `6.4×` compression); (2) add fixed-capacity LRU/ring eviction (`<= topk_chunks * 8`) to `cached_sparse_{k,v}`; (3) vectorize CPU Value gather.
- **Core Hypothesis:** Fixing the two `[IMPLEMENTATION]` bugs preserves `ShadowKV`'s near-lossless `64.0%` LongMemEval accuracy (`\le 3` regressions) while slashing 32K Peak Alloc VRAM from `14,233 MiB` down to `~9,200–9,600 MiB` (`< 1,900 MiB` net KV+aux) and keeping PCIe $\le 70\text{ MB/tok}$.
- **Quantitative Falsification Gate:** Falsified if 32K LongMemEval `< 94/150`, 32K Net KV+aux VRAM `> 2,200 MiB`, or 32K TPOT `> 60 ms/tok`.

### Candidate 3: `Hot BF16 Window + Selective AABB-Indexed 4-Bit CPU Page Bank` (`STRUCTURAL` + `SPATIAL` + `TEMPORAL`)
- **Borrowed Components & Axes:** `StreamingLLM` sink+window (`C1`, `STRUCTURAL` BF16 hot GPU tier) + `ArkVale` vectorized 16-token AABB page digests (`C1`, `TEMPORAL` GPU index) + `FlexGen-KV` group-4-bit CPU page storage (`C1`/`C2`, `SPATIAL`, cutting Host RSS & PCIe payload `3.56×`).
- **Implementation Prerequisites:** (1) Fix `arkvale_cache.py` to pass true **`query_states`** (with GQA max-pooling) instead of `key_states` into page scoring; (2) pin exempt layers `0–1` on GPU (`~256 MiB`) instead of re-copying 32K every step; (3) maintain resident GPU page slots and fetch **only delta pages** (`page_size=16`) over an async CUDA stream.
- **Core Hypothesis:** Combining `ArkVale`'s page-level AABB recall with `FlexGen`'s 4-bit CPU storage and a `StreamingLLM` hot window caps 32K Host RSS under `~5 GiB` (vs. `16.6 GiB`), cuts PCIe traffic `> 10×` vs. `FlexGen-KV` (`< 150 MB/tok`), and recovers multi-session recall (`\ge 60%` LongMemEval) on `< 1.5 GiB` net KV+aux VRAM.
- **Quantitative Falsification Gate:** Falsified if 32K LongMemEval `< 90/150` (`60.0%`), 32K Host RSS `> 6,500 MiB`, PCIe memcpy `> 200 MB/tok`, or 32K Net KV+aux VRAM `> 1,800 MiB`.

### Candidate 4: `Fused TurboQuant Rotation + Hot BF16 Window (Calibration-Free Hybrid)` (`STRUCTURAL`)
- **Borrowed Components & Axes:** `StreamingLLM` sink+window (`C1`, BF16 hot tokens) + `TurboQuant` fused 128-pt FWHT + 4-bit Beta Lloyd-Max K + 3-bit/QJL V (`C1`–`C3`, `STRUCTURAL`).
- **Implementation Prerequisites:** Replace Python-loop `fp32` `fwht` and `torch.cat` in `turboquant_ops.py`/`turboquant_cache.py` with a fused register-level Triton/CUDA FWHT+quant/dequant kernel and pre-allocated buffers.
- **Core Hypothesis:** Keeping recent/sink tokens in unrotated BF16 while applying fused rotation+QJL to older tokens eliminates `TurboQuant`'s `fact_recall` (`-16.67 pp`) and `cross_session_reasoning` (`-20.0 pp`) losses and reduces `864 ms` TPOT by `> 20×`, offering a **calibration-free** alternative to `QServe KV4` for bursty multi-user workloads.
- **Quantitative Falsification Gate:** Falsified if 32K LongMemEval `< 90/150` (`60.0%`), 32K TPOT `> 45 ms/tok`, or Net KV+aux VRAM `> 2,500 MiB`.
