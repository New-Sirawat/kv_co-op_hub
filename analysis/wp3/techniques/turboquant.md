# WP-3 Technique Analysis: `turboquant` (TurboQuant)

## 1. Verified Configuration & Empirical Snapshot

### Provenance & Authoritative Configuration
- **Repository & Commit:** `khemingkapat/kv_bench` @ `2a9338a2f750741f7765d79c1270f5579a9ffed0` (`main`, clean working tree)
- **Primary Model:** `Qwen/Qwen3-4B-Instruct-2507` (`cdbee75f17c01a7cc42f958dc650907174af0554`, Pure Dense GQA, `bfloat16`, $L=36, H_Q=32, H_{KV}=8, D=128$)
- **Hardware & Harness:** `gpu06.slurm.cpe.kmutt.ac.th`, NVIDIA GeForce RTX 4090 (23.52 GiB physical, **16.0 GiB** process allocator cap), Slurm Job `25931` (hardware/verify/NIAH) & `30137` (LongMemEval)
- **Authoritative Axis ([`docs/METRICS_AND_AXES.md`](../../../docs/METRICS_AND_AXES.md)):** `STRUCTURAL`
- **Manifest `technique_config`:**
  ```json
  {
    "cache_class": "TurboQuantCache",
    "k_bits": 4,
    "v_bits": 3,
    "qjl_v": true,
    "qjl_k": false,
    "block_size": 16,
    "compression_ratio": 3.657,
    "theoretical_bytes_per_token_head": 140
  }
  ```
- **`VERIFIED` Hub Artifact SHA-256s ([`run_manifest_wp3_turboquant.json`](../../../results/manifests/run_manifest_wp3_turboquant.json)):**
  - `turboquant_hardware_pytorch_16gb.json`: `6dfdb5bbd319a1b41593464cd5300798a42af7885116d8ecdc0fa860dfd2a284`
  - `turboquant_longmem_pytorch_16gb.json`: `5a4056574b2779d0e547820fcf4ddf222a1dfa211c4abcaf4f9123d5a2dd4599`
  - `turboquant_niah_pytorch_16gb.json`: `08562dd7526bea2f99df7036d4ea2c8a66f4d0b45811ae6a959d504517afec5f`
  - `verify_turboquant_pytorch_16gb.json`: `abf3c6eba2ab178f3911dd35db1da69c92c292d78dff85858bb0bcb793380377`
  - `turboquant_grades.json`: `07aef3bd22bc25b2b5a18cf5b4a4275b403b2ed3c884cf9ad66ec3ca1edd5fbd`
  - `turboquant_judged.json`: `beab6cc036f9113ff3c37506bd4a2a2e0f1ac3fda1066224adb9c90ada098ae7`

### Precision & Retention Separation
| Metric | Value | Notes |
|---|---:|---|
| **Physical Scalar Bits** | **4.375 bits/scalar** (combined); **K: 4.25**, **V: 4.50** | K: 4-bit Lloyd-Max (`64 B/tok/head`) + FP32 norm (`4 B`) = `68 B`; V: 3-bit Lloyd-Max (`48 B`) + 1-bit QJL sign (`16 B`) + 2×FP32 norms (`8 B`) = `72 B`; Total `140 B / 256 scalars` vs. `512 B` BF16 (`3.657x`) |
| **Effective Token Retention** | **100.0%** | All prompt and generated tokens are retained (`TurboQuantLayer.key_packed.shape[-2]`) |

### Hardware Context Scaling Sweep (`512 -> 32,768`, `SyntheticFiller`)
Static `Qwen3-4B` BF16 weights + PyTorch runtime occupy **7,738.78 MiB (~7.56 GiB)** (`verify_baseline_pytorch_16gb.json` at `ctx=128`); net KV+aux VRAM below is `Peak Alloc - 7,738.78 MiB`.

| Ctx | Peak Alloc (MiB) | Base Alloc (MiB) | Net KV+Aux (MiB) | Base Net KV (MiB) | TTFT (s) | TPOT (ms) | Base TPOT (ms) | Tok/s | Host RSS Δ (MiB) | Memcpy (B/tok) | Top-1 Agr. | KL Mean | KL P99 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 7,869.55 | 7,911.66 | 130.77 | 172.88 | 0.1481 | 82.25 | 17.36 | 12.16 | 4,816.12 | 4,922.25 | 0.0938 | 12.7666 | 15.4200 |
| 1,024 | 8,053.12 | 8,133.04 | 314.34 | 394.26 | 0.1771 | 81.73 | 16.91 | 12.24 | 4,816.25 | 5,050.25 | 0.1250 | 11.9099 | 15.4249 |
| 2,048 | 8,358.02 | 8,566.92 | 619.24 | 828.14 | 0.3270 | 79.53 | 16.61 | 12.57 | 4,678.87 | 5,315.25 | 0.2188 | 10.9512 | 15.4249 |
| 4,096 | 8,436.78 | 8,854.93 | 698.00 | 1,116.15 | 0.6775 | 81.24 | 16.67 | 12.31 | 4,969.75 | 5,845.25 | 0.0000 | 13.3091 | 15.2225 |
| 8,192 | 8,594.31 | 9,430.96 | 855.53 | 1,692.18 | 1.6191 | 128.84 | 17.12 | 7.76 | 5,159.28 | 6,905.25 | 0.2344 | 10.5429 | 15.4249 |
| 16,384 | 9,152.22 | 10,616.78 | 1,413.44 | 2,878.00 | 5.4355 | 420.20 | 20.47 | 2.38 | 6,146.35 | 9,025.25 | 0.1875 | 11.5250 | 15.4249 |
| 32,768 | 10,294.35 | 13,224.90 | 2,555.57 | 5,486.12 | 20.6266 | 864.12 | 29.17 | 1.16 | 7,528.91 | 13,265.25 | 0.1562 | 12.2347 | 15.4249 |

*Verify (`ctx=128`, 16 gen tokens):* `top1_agreement = 0.0`, `kl_div_mean = 13.40672`, `kl_div_p99 = 15.3301`.
*Shannon-$\phi$ Geometric Distortion (32K):* `k_eps_snr = 0.008737`, `k_eps_dir = 0.004306`, `v_eps_snr = 0.018944`, `v_eps_dir = 0.009249`.

### Quality Diagnostics: NIAH (2K/4K) & LongMemEval (150 Questions, Judged)
- **NIAH Exact Retrieval:**
  - `2,048` ctx: **10/10 (100%)** across depths `[0.0 .. 0.9]`
  - `4,096` ctx: **10/10 (100%)** across depths `[0.0 .. 0.9]`
- **LongMemEval Logit Fidelity (native lengths `192..22,400`, mean `5,967.9` tokens):** `top1_agreement = 0.226`, `kl_div_mean = 11.28839`, `kl_div_p99 = 15.4249`.
- **LongMemEval 5-Capability Judged Accuracy ([`turboquant_judged.json`](../../../results/grades/turboquant_judged.json) vs. [`baseline_judged.json`](../../../results/grades/baseline_judged.json)):**

| Capability | Baseline | TurboQuant | Δ vs. Baseline | `QUALITY_REGRESSION` Count |
|---|---:|---:|---:|---:|
| **Overall (150)** | **66.0% (99/150)** | **53.33% (80/150)** | **-12.67 pp** | **20** *(+1 `UNEXPECTED_GAIN`)* |
| `fact_recall` (30) | 90.00% | 73.33% | -16.67 pp | 5 |
| `information_update` (30) | 66.67% | 53.33% | -13.33 pp | 4 |
| `cross_session_reasoning` (30) | 43.33% | 23.33% | -20.00 pp | 7 |
| `relative_timeframe` (30) | 30.00% | 20.00% | -10.00 pp | 3 |
| `unanswerable_refusal` (30) | 100.00% | 96.67% | -3.33 pp | 1 |

![TurboQuant Hardware Sweep](../../figures/wp3/turboquant/hardware_sweep.png)
![TurboQuant KL Fidelity](../../figures/wp3/turboquant/kl_fidelity.png)
![TurboQuant LongMemEval Diagnostics](../../figures/wp3/turboquant/longmem_diagnostics.png)

---

## 2. White-Box Component Decomposition

| ID | Component | Source Location | Tensor Shapes & Dtypes (`Qwen3-4B`: $L=36, H_Q=32, H_{KV}=8, D=128$) | Cadence |
|---|---|---|---|---|
| **C1** | **Unfused Python-Loop FWHT + Rademacher Sign Flip** | `turboquant_ops.py`: `fwht` (L42–65), `ifwht` (L68–74), `generate_rademacher_signs` (L88–98) | Casts `(1, 8, T, 128)` to `float32`; runs $\log_2(128)=7$ sequential `view`/`cat` butterfly stages in Python (`s`: `(1, 8, 1, 128)` `fp32`) | Every encode (prefill/decode) + **twice** per full-cache decode (K & V `ifwht` + QJL `fwht`) |
| **C2** | **Asymmetric Key Encoder (4-bit Beta Lloyd-Max)** | `turboquant_ops.py`: `turboquant_encode_key` (L210–228), `CENTROIDS_4BIT` / `THRESHOLDS_4BIT` (L16–28) | `key_packed`: `(1, 8, T, 64)` `uint8` (via `torch.bucketize`), `key_norms`: `(1, 8, T, 1)` `fp32` | Every prefill chunk & decode step |
| **C3** | **Asymmetric Value Encoder (3-bit Lloyd-Max + 1-bit QJL)** | `turboquant_ops.py`: `turboquant_encode_value` (L246–273), `qjl_sketch` (L184–193), `pack_value_nibbles` (L142–163) | `value_packed`: `(1, 8, T, 64)` `uint8`, `value_norms` & `value_res_norms`: `(1, 8, T, 1)` `fp32` | Every prefill chunk & decode step |
| **C4** | **`torch.cat` Cache Append & Full-Sequence FP32→BF16 Decode** | `turboquant_cache.py`: `TurboQuantLayer.update` (L118–148); `turboquant_ops.py`: `turboquant_decode_key`/`value` (L231–243, L276–294) | Re-allocates `key_packed`, `value_packed`, norms via `torch.cat` on every step; unpacks & runs `ifwht` on **entire `[0..T]`** in `fp32` before casting to `bf16` | Every prefill chunk & every decode step |

---

## 3. 3-Way Causal Attribution

- **VRAM Savings (`10,294.35 MiB` vs. `13,224.90 MiB` at 32K, `-2,930.55 MiB`):**
  - **`[IMPLEMENTATION]`** Persistent `TurboQuantLayer` state is `1,323,745,920 B` (`1.23 GiB`) at 32K, achieving its exact `3.657x` theoretical ratio vs. BF16. However, **C4**'s full-sequence `fp32` `ifwht` decode + `torch.cat` reallocations create transient FP32/BF16 buffers, leaving net KV+aux at `2,555.57 MiB` (`1.26 GiB` above persistent compressed state, vs. `5,486.12 MiB` baseline net).
- **Catastrophic TTFT (`20.63 s` vs. `6.98 s`, `2.96x`) & TPOT (`864.12 ms` vs. `29.17 ms`, `29.62x` slowdown at 32K):**
  - **`[IMPLEMENTATION]`** Driven by **C1 + C4**. Every decode step across 36 layers runs `torch.cat` (no pre-allocation) and executes **three** `O(T · d log d)` Python-loop `fwht` transforms in `float32` (`turboquant_decode_key`, `qjl_reconstruct`, and `turboquant_decode_value`) over the *entire* `[0..T]` sequence. At 32K, TPOT explodes from `81.2 ms` (4K) → `128.8 ms` (8K) → `420.2 ms` (16K) → `864.1 ms` (32K), scaling superlinearly with context due to global-memory butterfly passes and allocator churn.
- **Host RSS (`4,816 -> 7,529 MiB`) & PCIe Memcpy (`13,265 B/tok` at 32K):**
  - **`[IMPLEMENTATION]`** Discounting the **`8,266 B/tok`** 32K hardware-sweep instrumentation floor, the extra `~5,000 B/tok` memcpy and `~4.8–7.5 GiB` Host RSS delta reflect Python/PyTorch `torch.bucketize` / `torch.cat` host-device synchronization and `record_turboquant_telemetry` scalar extraction on every `update()` call (`turboquant_cache.py` L201–205).
- **KL Divergence (`10.5–13.3` mean; `15.4249` clamp) vs. NIAH (`100%`) & LongMemEval (`53.33%`, `20` QR):**
  - **`[INTRINSIC]`** Shannon-$\phi$ metrics (`k_eps_dir = 0.0043`, `v_eps_dir = 0.0092`) confirm the Beta(63.5, 63.5) Lloyd-Max + QJL codec preserves unit-sphere geometry and `10/10` 2K/4K NIAH retrieval (not extrapolated to 32K). However, on LongMemEval (`192..22,400` tokens), TurboQuant suffers **20 `QUALITY_REGRESSION`s** (7 `cross_session_reasoning`, 5 `fact_recall`, 4 `information_update`, 3 `relative_timeframe`, 1 `unanswerable_refusal` — e.g., `bb7c3b45`, `2e6d26dc`, `f0e564bc`). As warned in the TurboQuant paper (Section 8 / [`turboquant.md`](../../../docs/literature_survey/techniques/turboquant.md)), WHT preserves **low-rank subspace correlations** of real LLM keys, and 4-bit static unit-sphere quantization without outlier-channel smoothing or attention-aware weighting loses fine-grained multi-fact discrimination in `fact_recall` (`-16.67 pp`) and `cross_session_reasoning` (`-20.00 pp`).
  - **`[HARDWARE/GQA]`** `kl_div_p99 = 15.4249` is the runner's top-5 logprob clamp on `N=1` `SyntheticFiller`.

---

## 4. Paper vs. Replication Fidelity

| Dimension | TurboQuant Paper ([`docs/literature_survey/techniques/turboquant.md`](../../../docs/literature_survey/techniques/turboquant.md)) | `kv_bench` Replication (`turboquant_ops.py` / `turboquant_cache.py`) | Impact of Divergence |
|---|---|---|---|
| **Asymmetric KQV Design** | K: WHT + 4-bit Beta Lloyd-Max; V: WHT + 3-bit Lloyd-Max + 1-bit QJL; FP32 norms | Exact match (`k_bits=4`, `v_bits=3`, `qjl_v=True`, `qjl_k=False`, `140 B/tok/head`) | Faithful mathematical codec |
| **FWHT & QJL Execution** | Fused GPU/HIP kernel operating in shared memory/registers on 128-dim vectors | Unfused Python `while h < d` loop with `torch.cat` in `fp32` (`turboquant_ops.py` L56–61) + `torch.bucketize` | **`29.6x` TPOT slowdown** at 32K (`864.12 ms`) |
| **Decode Read Path** | Inner-product / attention-loop dequantization without materializing full FP32/BF16 KV | `torch.cat` buffer append + full-sequence `fp32` `ifwht` decode to `bf16` on every step (`turboquant_cache.py` L126–146) | Causes quadratic-feeling decode memory traffic & RSS bloat |
| **Pre-fill Telemetry** | Offline 6D Shannon-$\phi$ evaluation | Computes `turboquant_decode_key`/`value` a *second* time during prefill just for Shannon-$\phi$ (`turboquant_cache.py` L104–115) | Extra prefill TTFT overhead (`20.63 s` at 32K) |

---

## 5. Component Verdicts (`KEEP` / `ADAPT` / `DROP`)

| Component | Verdict | Contribution | Failure Mode | Fix Prerequisite (if `ADAPT`) |
|---|---|---|---|---|
| **C1: Unfused Python FWHT** | **`ADAPT`** | Disperses heavy-tailed outliers into Beta(63.5, 63.5) without calibration | `while h < d` Python loop in `fp32` dominates TTFT & TPOT (`29.6x` slower at 32K) | Implement fused Triton/CUDA 128-pt register-level FWHT + QJL kernel |
| **C2: 4-bit Beta Lloyd-Max (K)** | **`ADAPT`** | 4.25 bits/scalar; 100% 2K/4K NIAH | -16.67 pp `fact_recall`, -20 pp `cross_session_reasoning` on LongMemEval (low-rank key subspace sensitivity) | Combine with channel-wise outlier smoothing or hybrid BF16 sink/recent window |
| **C3: 3-bit Lloyd-Max + 1-bit QJL (V)** | **`KEEP`** | 4.50 bits/scalar unbiased Value estimator; `v_eps_dir = 0.0092` | Requires fused kernel to hide `qjl_reconstruct` `fwht` cost | — |
| **C4: `torch.cat` + Full-Seq Decode** | **`DROP`** | Reference HF `DynamicLayer` compatibility | `torch.cat` fragmentation + redundant full-sequence `fp32` `ifwht` every step | Pre-allocate capacity buffers (like `QServeKVLayer`) + fuse dequant into SDPA |
