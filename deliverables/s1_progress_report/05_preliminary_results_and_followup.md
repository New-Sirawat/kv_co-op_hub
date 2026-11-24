# Chapter 5 Blueprint — Preliminary Results and Follow-up

> **KMUTT Report Mapping:** Chapter 5 (`5.1`–`5.4` in `Progress_Sem1_ENG_Example2.pdf`: Implementation Details, Preliminary Results, Experimental Setup for Full Evaluation, Follow-up & Next Steps)  
> **Document Role:** Templating, Drafting & Agent Handoff Specification (**not** final polished thesis prose)  
> **Branch Baseline:** `docs/s1-progress-report-blueprint` (branched from `main`; PR #21 unmerged)  
> **Target Milestone:** First advisor review draft — **December 1, 2026** (`WP-5`)

> [!WARNING]
> **PR #21 Provenance Boundary:** All numbers in this blueprint are computed **directly from verified JSON files present on `main` / this branch**. Any reference to `results/baselines/baseline_pytorch_16gb.json`, `analysis/wp3/*`, or `analysis/figures/wp3/*` is explicitly tagged `[DRAFT/PR#21]` because PR #21 (`docs/wp3-result-interpretation`) is not yet merged.

---

## Section 5.1 — Implementation Details & Current Milestone Status

### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Role (`Progress_Sem1_ENG_Example2.pdf` §5.1):** Reports concrete Semester 1 engineering progress against the planned Work Breakdown Structure (WBS), clearly distinguishing completed components, in-progress tasks, and deferred items.
- **Committee Expectation:** Objective, verifiable milestone accounting backed by GitHub Project #9 issue completion and provenance manifests—not vague percentage estimates.

### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** Across Semester 1 Phases 1–2, **WP-2 (Hardware & Bottleneck Profiling) is 100% complete**, **WP-1 (Testbed Setup & Native Baselines) is 5/6 complete** (only local 8 GB RTX 4060 adaptation `WP1-5` remains pending), and **WP-3 (11-Method Candidate Exploration & Micro-Benchmarking) is `14/15` sub-issues complete** (`#19`–`#30` [12] + `#31` + `#35` = 14 Done; `#34` Todo — `24/26` total across WP-1..WP-3; all 11 algorithms empirically benchmarked and semantically graded on 150 LongMemEval queries, while final interpretation report awaits PR #21 review). **WP-4** (policy synthesis) and **WP-5** (Semester 1 report & defense) are actively underway.
- **Delta from Proposal Defense (`KV-Cache Co-optimization.pdf`):**
  - *Proposal:* Planned 5 literature baselines and a single-backend vLLM workflow.
  - *Current State:* Delivered **11 reproduced techniques** + **15 verified SHA256 manifests** + **1,800 semantically graded LongMemEval responses** (`12 methods × 150 queries`) under a dual-backend harness (`vLLM V1` + `PyTorch SDPA`). Note that `WORK_PACKAGES.md` top-level table still displays static `🔴 Not started` badges—GitHub Project #9 is the live tracking source of truth (`[NEEDS_VERIFICATION]` to sync `WORK_PACKAGES.md` status badges before Dec 1).

### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)
| Milestone / Item | Provenance Source | Status Tag | Verified Progress |
| :--- | :--- | :--- | :--- |
| `WP-1` Testbed & Native Baselines | GitHub Project #9 + `results/manifests/run_manifest_baseline_pytorch_16gb.json` + `dataset_manifest.json` | `[READY]` (5/6) / `[NEEDS_VERIFICATION]` (`WP1-5`) | Cluster 16 GB, `docs/MODELS.md`, datasets, unified harness, & baselines done; **only `WP1-5` (local 8 GB RTX 4060) pending** |
| `WP-2` Bottleneck Profiling | `results/manifests/run_manifest_baseline_16gb.json` (Job `22868`, SHA `3aee66fe`) | `[READY]` (100%) | Single-stream OOM cliff (`512`–`65,536`), concurrency grid (`1`–`16` users), identity/cross-model verify, vLLM LongMemEval done |
| `WP-3` 11-Method Micro-Benchmarking | 12 manifests in `results/manifests/run_manifest_wp3_*.json` (SHA `2a9338a2`) + 24 files in `results/grades/` | `[READY]` (`14/15` Done) / `[DRAFT/PR#21]` (`#34` Todo: interpretation) | All 11 techniques benchmarked (Hardware `512`–`32K`, Verify `128`, NIAH `2K/4K`, LongMemEval `150q` judged); synthesis PR #21 open |
| `WP-4` 3-Axis Policy Design | `WORK_PACKAGES.md` (`WP-4`) | `[OPEN_DECISION]` | Blocked on PR #21 review & policy selection |
| `WP-5` S1 Report & Defense | `deliverables/s1_progress_report/` | `[READY]` (Active) / `[NEEDS_VERIFICATION]` (CITI) | Proposal defense done; CITI Program Ethics Exam `[NEEDS_VERIFICATION]` (consistent with `01_introduction_and_timeline.md` §1.2.3); Dec 1 progress report draft in progress |

### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline Skeleton:**
  1. Summary of Semester 1 completion by Work Package (`WP-1` through `WP-5`).
  2. Artifact & provenance inventory (15 manifests, **46** WP-3 JSON files on `main` [44 PyTorch + 2 legacy vLLM `streamingllm`], 24 LongMemEval grade/judged files, 20 completed literature reviews).
  3. Explicit transparency on remaining Semester 1 gaps (`WP1-5` local 8 GB run, PR #21 manual audit, `WP-4` prototype smoke test).
- **Required Table Spec — Table 5.1 (`Semester 1 Work Package Completion & Artifact Registry`):**
  - *Columns:* `WP | Sub-Issues Done | Key Verified Artifacts on main | Remaining Actions Before Dec 1 / Sem 1 End`.
- **Open Questions / Decisions Needed (`[OPEN_DECISION]`):**
  - Sync `WORK_PACKAGES.md` status column (currently `🔴 Not started`) with GitHub Project #9 live status before submitting the Dec 1 draft? *(Recommended: Yes.)*

---

## Section 5.2 — Preliminary Results

### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Role (`Progress_Sem1_ENG_Example2.pdf` §5.2):** Presents verified empirical findings from Semester 1 experiments, proving the problem statement quantitatively and demonstrating the performance/quality trade-offs that motivate the proposed system.
- **Committee Expectation:** Concrete tables and charts with exact hardware telemetry and quality metrics—no cherry-picking, no hidden failures, and clear distinction between hardware-ceiling baselines and candidate micro-benchmarks.

### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** Our Semester 1 empirical results confirm three foundational claims:
  1. **Native 16 GB Hardware Wall (`5.2.1`):** On `Qwen3-4B-Instruct-2507` (BF16), native vLLM hits a hard KV-cache OOM cliff above **37,056 tokens** (`49,152` and `65,536` fail with `7.04 GiB` / `9.29 GiB` needed vs. `5.09 GiB` available), while multi-user concurrency (`1 → 16` users at `8K`) nearly doubles (+81%) TPOT (`11.16 ms → 20.20 ms`).
  2. **The 3-Axis Trade-Off Trilemma (`5.2.2`):** Across all 11 reproduced candidates at 32K context, **no single-axis method dominates simultaneously** on VRAM, decode speed, PCIe traffic, and LongMemEval accuracy—e.g., `StreamingLLM` is fastest (`20.95 ms` TPOT, `8,711.7 MB` VRAM) but collapses to `44.0%` LongMemEval (`3.3%` Cross-Session Reasoning), whereas `ShadowKV` preserves `64.0%` LongMemEval (within `2.0 pp` of the `66.0%` BF16 baseline) with only `67.24 MB/tok` PCIe traffic, yet consumes `14,233.0 MB` VRAM.
  3. **White-Box Failure Isolation (`5.2.3`):** Our Causal Attribution Tree + Post-Hoc Inspector cleanly separates true algorithmic regressions (`StreamingLLM`: `37` regressions; `CacheGen`: `36`) from base-model limitations (`45–51` shared baseline failures) and PyTorch wrapper/codec anomalies (`HeadQ`: `0.0%`; `EpiCache`: `5.3%`; `TurboQuant`: `864.12 ms` pure-PyTorch decode).
- **Delta from Proposal Defense (`KV-Cache Co-optimization.pdf`):**
  - *Proposal:* Cited only theoretical memory formulas and literature-reported numbers (zero local empirical benchmarks).
  - *Current State:* Full empirical verification across both native vLLM (`WP-2`) and 11 PyTorch SDPA candidates (`WP-3`), including 150-sample semantic grading and causal attribution.

### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)
| Subsection | Exact Files & Keys on `main` | Status Tag | PR #21 / Verification Notes |
| :--- | :--- | :--- | :--- |
| **5.2.1** Native vLLM Ceiling & Concurrency | `results/baselines/baseline_hardware_16gb.json` (`results[].{requested_ctx, status, peak_alloc_mb, peak_reserved_mb, ttft_sec, tpot_mean_ms, throughput_tok_per_sec, error}`)<br>`results/baselines/concurrent_baseline_16gb.json` (`concurrency_grid[].{num_users, results[]}`)<br>`results/baselines/verify_identity_16gb.json`, `verify_cross_model_16gb.json` | `[READY]` | Manifest: `run_manifest_baseline_16gb.json` (Job `22868`) |
| **5.2.2** WP-3 11-Method Hardware, KL/Top-1, NIAH & LongMemEval | `results/runs/wp3/<tech>_hardware_pytorch_16gb.json` (32K row)<br>`results/runs/wp3/verify_<tech>_pytorch_16gb.json` (`extra.{top1_agreement, kl_div_mean, kl_div_p99}` @ 128 ctx)<br>`results/runs/wp3/<tech>_longmem_pytorch_16gb.json` (`extra.{top1_agreement, kl_div_mean, kl_div_p99, diag_mean_*}` @ 32K)<br>`results/runs/wp3/<tech>_niah_pytorch_16gb.json` (`extra.niah_accuracy` @ 2K/4K)<br>`results/grades/<tech>_judged.json` & `baseline_judged.json` (`correct_samples`, `overall_accuracy`, `per_task_accuracy`, `causal_verdict_counts`) | `[READY]` | **Uncompressed PyTorch Hardware Baseline** (`results/baselines/baseline_pytorch_16gb.json`) is `[DRAFT/PR#21]` (not yet on `main`) |
| **5.2.3** Component-Level Synthesis & Figures | `analysis/wp3/*`, `analysis/figures/wp3/*`, `results/baselines/baseline_pytorch_16gb.json` | `[DRAFT/PR#21]` | **Do not embed PR #21 plots/conclusions as final until PR #21 is reviewed & merged** |

### Block 4: Draft Section Skeleton, Verified Numeric Snapshots, Table/Figure Specs & Open Questions

#### Subsection 5.2.1 — WP-1/WP-2 Native vLLM 16 GB Hardware Ceiling & Concurrency Saturation (`[READY]`)
- **Verified Single-Stream Context Sweep (`baseline_hardware_16gb.json`, `Qwen3-4B-Instruct-2507`, BF16, 16 GB Cap):**
  | Context | Status | Peak Alloc (MB) | Peak Reserved (MB) | Host RSS $\Delta$ (MB) | TTFT (s) | TPOT Mean (ms) | Throughput (tok/s) |
  | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | `512` | `SUCCESS` | `13,510.9` | `13,578.0` | `1,503.4` | `0.0573` | `11.30` | `88.51` |
  | `1,024` | `SUCCESS` | `13,511.0` | `13,576.0` | `1,501.7` | `0.0863` | `11.39` | `87.80` |
  | `2,048` | `SUCCESS` | `13,511.1` | `13,576.0` | `1,510.6` | `0.1445` | `11.62` | `86.05` |
  | `4,096` | `SUCCESS` | `13,592.3` | `13,728.0` | `1,509.9` | `0.2735` | `11.98` | `83.45` |
  | `8,192` | `SUCCESS` | `13,900.8` | `14,202.0` | `1,518.6` | `0.5956` | `12.64` | `79.10` |
  | `16,384` | `SUCCESS` | `13,902.2` | `14,200.0` | `1,536.4` | `1.5350` | `13.93` | `71.79` |
  | `32,768` | `SUCCESS` | `13,904.3` | `14,200.0` | `1,565.6` | `4.1525` | `16.65` | `60.05` |
  | `49,152` | `FAILED` | — | — | — | — | — | *OOM: needs `7.04 GiB` > `5.09 GiB` KV pool (max `37,056` tok)* |
  | `65,536` | `FAILED` | — | — | — | — | — | *OOM: needs `9.29 GiB` > `5.09 GiB` KV pool (max `37,056` tok)* |
- **Verified Concurrency Saturation Snapshot (`concurrent_baseline_16gb.json`, `8,192` Context Slice across 1–16 Users):**
  - `1` user: TTFT `0.623s` (P95 `0.623s`), TPOT `11.16 ms`, Aggregate Throughput `89.60 tok/s`.
  - `4` users: TTFT `0.620s` (P95 `0.640s`), TPOT `13.44 ms`, Aggregate Throughput `301.20 tok/s`.
  - `8` users: TTFT `0.646s` (P95 `0.677s`), TPOT `15.50 ms`, Aggregate Throughput `523.19 tok/s`.
  - `12` users: TTFT `0.662s` (P95 `0.701s`), TPOT `18.24 ms`, Aggregate Throughput `667.42 tok/s`.
  - `16` users: TTFT `0.689s` (P95 `0.737s`), TPOT `20.20 ms` (**+81% decode latency vs. 1 user**), Aggregate Throughput `803.74 tok/s`.
- **Logit Harness Sanity Verification (`verify_identity_16gb.json`, `verify_cross_model_16gb.json`):**
  - Identity check: `top1_agreement = 1.0`, `kl_div_mean = 0.0`, `kl_div_p99 = 0.0`.
  - Cross-model control: `top1_agreement = 0.0`, `kl_div_mean = 13.586`, `kl_div_p99 = 15.324`.

#### Subsection 5.2.2 — WP-3 11-Method Comparative Micro-Benchmarking Snapshot (`[READY]` on `main`)
> *Note: Hardware sweep for uncompressed PyTorch SDPA (`baseline_pytorch_16gb.json`) is `[DRAFT/PR#21]`; baseline LongMemEval (`99/150 = 66.0%`) is `[READY]` in `results/grades/baseline_judged.json`.*

- **Table 5.2A — Verified 32K Hardware & System Telemetry (`results/runs/wp3/*_hardware_pytorch_16gb.json`, `requested_ctx = 32768`):**
  | Method | Axes | Peak Alloc (MB) | Host RSS $\Delta$ (MB) | TTFT (s) | TPOT (ms) | Throughput (tok/s) | Memcpy (MB/tok) |
  | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | *PyTorch BF16 Baseline* | *None* | *`[DRAFT/PR#21]`* | *`[DRAFT/PR#21]`* | *`[DRAFT/PR#21]`* | *`[DRAFT/PR#21]`* | *`[DRAFT/PR#21]`* | *`[DRAFT/PR#21]`* |
  | **EpiCache** | `STR`, `SPA` | `8,567.1` | `2,845.7` | `2.88` | `30.60` | `32.68` | `0.44` |
  | **StreamingLLM** | `STR` | `8,711.7` | `1,326.5` | `2.90` | `20.95` | `47.73` | `0.01` |
  | **FlexGen KV** | `STR`, `SPA` | `8,888.9` | `6,564.6` | `10.20` | `229.57` | `4.36` | `1,646.96` |
  | **ArkVale** | `STR`, `SPA` | `9,032.9` | `16,613.6` | `21.12` | `102.92` | `9.72` | `1,618.38` |
  | **H2O** | `STR` | `9,724.6` | `6,873.7` | `9.11` | `45.88` | `21.80` | `2.01` |
  | **HeadQ** | `STR` | `10,108.8` | `4,655.6` | `7.72` | `80.89` | `12.36` | `0.01` |
  | **TurboQuant** | `STR` | `10,294.4` | `7,528.9` | `20.63` | `864.12` | `1.16` | `0.01` |
  | **CacheGen** | `STR`, `SPA` | `10,600.2` | `4,535.8` | `12.55` | `82.87` | `12.07` | `935.02` |
  | **QServe KV4** | `STR` | `10,762.7` | `3,628.6` | `7.24` | `59.92` | `16.69` | `0.01` |
  | **InfiniGen** | `STR`, `SPA`, `TMP` | `13,910.0` | `20,490.6` | `15.37` | `219.68` | `4.55` | `1,829.89` |
  | **ShadowKV** | `STR`, `SPA`, `TMP` | `14,233.0` | `5,325.1` | `6.80` | `85.29` | `11.73` | `67.24` |
  > *Units note:* `Peak Alloc` (`peak_alloc_mb`) and `Host RSS Δ` (`host_rss_delta_mb`) are reported in binary **MiB** ($1024^2$ bytes) by `kv_bench`, whereas `Memcpy (MB/tok)` is computed in decimal **MB** (`memcpy_bytes_per_token / 1e6`).

- **Table 5.2B — Verified Quality Pillars: Logit Divergence (128 & 32K), 2D NIAH, and LongMemEval 150q Judged (`results/runs/wp3/*`, `results/grades/*_judged.json`):**
  | Method | Verify 128 Top-1 / KL | LongMem 32K Top-1 / KL | NIAH 2K / 4K | LongMemEval Overall | FR / IU / RT / CSR / UR (%) | Quality Regressions |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | **BF16 Baseline** | `1.000` / `0.00` *(ref)* | `1.000` / `0.00` *(ref)* | — | **`99/150` (`66.0%`)** | `90.0 / 66.7 / 30.0 / 43.3 / 100.0` | `0` *(51 base fail)* |
  | **ShadowKV** | `1.000` / `0.00` | `0.679` / `4.58` | `100%` / `100%` | **`96/150` (`64.0%`)** | `90.0 / 66.7 / 30.0 / 33.3 / 100.0` | **`3`** |
  | **FlexGen KV** | `1.000` / `0.00` | `0.300` / `10.25` | `100%` / `100%` | **`91/150` (`60.7%`)** | `90.0 / 66.7 / 26.7 / 30.0 / 90.0` | `14` *(+6 gain)* |
  | **InfiniGen** | `0.375` / `8.53` | `0.493` / `7.35` | `100%` / `100%` | **`91/150` (`60.7%`)** | `83.3 / 60.0 / 30.0 / 40.0 / 90.0` | `13` *(+5 gain)* |
  | **QServe KV4** | `0.000` / `12.56` | `0.455` / `7.89` | `100%` / `100%` | **`89/150` (`59.3%`)** | `86.7 / 66.7 / 23.3 / 23.3 / 96.7` | `13` *(+3 gain)* |
  | **ArkVale** | `1.000` / `0.00` | `0.384` / `8.99` | `100%` / `100%` | **`89/150` (`59.3%`)** | `86.7 / 60.0 / 30.0 / 26.7 / 93.3` | `14` *(+4 gain)* |
  | **H2O** | `1.000` / `0.00` | `0.475` / `7.65` | `100%` / `20%` | **`80/150` (`53.3%`)** | `86.7 / 46.7 / 26.7 / 16.7 / 90.0` | `22` *(+3 gain)* |
  | **TurboQuant** | `0.000` / `13.41` | `0.226` / `11.29` | `100%` / `100%` | **`80/150` (`53.3%`)** | `73.3 / 53.3 / 20.0 / 23.3 / 96.7` | `20` *(+1 gain)* |
  | **StreamingLLM** | `1.000` / `0.00` | `0.132` / `12.98` | `100%` / `40%` | **`66/150` (`44.0%`)** | `70.0 / 60.0 / 10.0 / 3.3 / 76.7` | `37` *(+4 gain)* |
  | **CacheGen** | `0.000` / `12.36` | `0.117` / `12.71` | `40%` / `80%` | **`66/150` (`44.0%`)** | `46.7 / 50.0 / 16.7 / 20.0 / 86.7` | `36` *(+3 gain)* |
  | **EpiCache** | `1.000` / `0.00` | `0.062` / `13.51` | `0%` / `0%` | **`8/150` (`5.3%`)** | `26.7 / 0.0 / 0.0 / 0.0 / 0.0` | `91` *(anomaly)* |
  | **HeadQ** | `0.000` / `14.29` | `0.014` / `14.76` | `0%` / `0%` | **`0/150` (`0.0%`)** | `0.0 / 0.0 / 0.0 / 0.0 / 0.0` | `99` *(anomaly)* |

#### Subsection 5.2.3 — Key Findings & White-Box Diagnostic Takeaways (`[READY]` + `[DRAFT/PR#21]`)
- **Outline Skeleton:**
  1. **Eviction vs. Recallability:** Permanent eviction (`StreamingLLM`: `44.0%`, `H2O`: `53.3%`) suffers severe collapse on Cross-Session Reasoning (`3.3%` and `16.7%`) and 4K NIAH (`40%` and `20%`), whereas recallable/offloaded methods (`ShadowKV` `64.0%`, `FlexGen KV` `60.7%`, `InfiniGen` `60.7%`, `ArkVale` `59.3%`) preserve `100%` NIAH at 4K and $\ge 59\%$ LongMemEval.
  2. **PCIe Bandwidth Wall vs. Sparse Value Offload:** Naive full-cache offload (`FlexGen KV`: `1,646.96 MB/tok`, `InfiniGen`: `1,829.89 MB/tok`, `ArkVale`: `1,618.38 MB/tok`) induces $103\text{--}230\text{ ms}$ TPOT, whereas `ShadowKV` reduces PCIe traffic by **$24\times$** (`67.24 MB/tok`) while achieving the highest overall accuracy (`64.0%`, only `3` quality regressions).
  3. **Quantization Sweet Spot vs. Implementation Overhead:** `QServe KV4` achieves `59.3%` LongMemEval and `100%` NIAH with near-zero / baseline-overhead PCIe traffic (`0.01 MB/tok`) and moderate TPOT (`59.92 ms`), demonstrating that 4-bit KV quantization is a strong GPU-resident Tier 1 candidate. Conversely, `TurboQuant`'s `864.12 ms` TPOT reflects unoptimized pure-PyTorch QJL overhead rather than an algorithmic wall.
  4. **Isolating Wrapper Anomalies (`[DRAFT/PR#21]`):** `HeadQ` (`0.0%`) and `EpiCache` (`5.3%`) exhibit catastrophic collapse attributable to PyTorch wrapper/codec integration issues rather than published paper claims; detailed component-level root-cause attribution is staged in `analysis/wp3/*` (`[DRAFT/PR#21]`) pending manual review.
- **Required Figure Specs (for Final Report Generation):**
  - *Fig. 5.1:* Native vLLM Context Scaling & OOM Cliff (`baseline_hardware_16gb.json`) + Multi-User Concurrency TPOT/Throughput (`concurrent_baseline_16gb.json`).
  - *Fig. 5.2:* 32K Pareto Frontier Scatter — *Peak VRAM (MB) vs. LongMemEval Accuracy (%)* and *TPOT (ms) vs. LongMemEval Accuracy (%)* across all 11 WP-3 methods (`[DRAFT/PR#21]` has candidate drafts in `analysis/figures/wp3/*`).
  - *Fig. 5.3:* Stacked Bar Chart of LongMemEval Causal Attribution Verdicts (`PASS_FIDELITY_MAINTAINED`, `QUALITY_REGRESSION`, `BASELINE_LIMITATION`, `UNEXPECTED_GAIN`) across all 12 entries (`results/grades/*_judged.json`).
- **Open Questions / Decisions Needed (`[OPEN_DECISION]`):**
  - Should `HeadQ` and `EpiCache` be plotted in the main Pareto scatter with an explicit *"PyTorch wrapper anomaly — see §5.2.3"* annotation, or separated into a diagnostic callout box? *(Recommended: Keep in table & plot with clear anomaly marker for full scientific transparency.)*

---

## Section 5.3 — Experimental Setup for Full Evaluation (Semester 2 Plan)

### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Role (`Progress_Sem1_ENG_Example2.pdf` §5.3):** Outlines the comprehensive experimental plan for Semester 2, specifying how the prototype will be scaled to full production-grade evaluation, stress testing, and ablation studies.
- **Committee Expectation:** Show a concrete, falsifiable Semester 2 evaluation matrix (`WP-6`, `WP-7`, `WP-8`) that directly answers the project's Research Questions under realistic consumer GPU workloads.

### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** Semester 2 shifts from single-stream PyTorch micro-benchmarking (`WP-3`) to **production runtime integration (`WP-6`)**, **multi-user 32K SME stress + RULER positional diagnostics (`WP-7`)**, and **controlled 3-axis ablation & kernel tuning (`WP-8`)** across both the 16 GB RTX 4090 cluster cap and the 8 GB RTX 4060 local testbed, expanding beyond `Qwen3-4B` to secondary dense (`Phi-4-mini`, `Ministral-3-3B`) and hybrid (`Qwen3.5-4B`, `Gemma-4-E4B-it`) models.
- **Delta from Proposal Defense (`KV-Cache Co-optimization.pdf`):**
  - *Proposal:* High-level mention of 32K tests and ablations without specifying exact runtime integration path or hybrid-architecture cross-checks.
  - *Current State:* Structured into `WP-6` (engine integration + KL/Top-1 verification), `WP-7` (32K + 10–15 concurrent SME traffic + LongMemEval 150q + RULER diagnostic), and `WP-8` (factorized axis ablations + PCIe/CPU crossover tuning + CUDA stream pipelining).

### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)
| Semester 2 Phase | Provenance Source | Status Tag | Scope |
| :--- | :--- | :--- | :--- |
| `WP-6` Runtime Engine Integration & Verification | `WORK_PACKAGES.md` (`WP-6`, lines 78–83), `docs/MODELS.md` (§4) | `[PENDING_S2]` | Integrate WP-4 policy into vLLM V1 / `LMCacheMPConnector`, verify KL/Top-1 & multi-session defragmentation |
| `WP-7` End-to-End Stress & Long-Context Quality | `WORK_PACKAGES.md` (`WP-7`, lines 89–95), `docs/literature_survey/benchmarks/ruler.md` | `[PENDING_S2]` | 32K single + 10–15 concurrent SME sessions, P95/P99 TTFT/TPOT, LongMemEval 150q, RULER diagnostic |
| `WP-8` Ablation Studies & Bottleneck Tuning | `WORK_PACKAGES.md` (`WP-8`, lines 97–101) | `[PENDING_S2]` | Isolate Structural vs. Spatial vs. Temporal gains; CPU dequantization vs. PCIe crossover; kernel/stream tuning |
| Cross-Model Generalization Matrix | `docs/MODELS.md` (§2, 9 pinned checkpoints) | `[PENDING_S2]` | Dense peers (`Phi-4-mini`, `Ministral-3-3B`, `Qwen3-1.7B` @ 8 GB) + Hybrid/SWA (`Qwen3.5-4B`, `Gemma-4-E4B-it`) |

### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline Skeleton:**
  1. **WP-6 Production Runtime Integration & Numerical Correctness Gates** (target: `kl_div_mean` & `top1_agreement` stability + zero OOM under multi-session allocation).
  2. **WP-7 Full Stress & Multi-Model Evaluation Protocol** (10–15 concurrent users, 32K context, LongMemEval 5 capabilities + RULER positional retrieval + 8 GB RTX 4060 parity).
  3. **WP-8 Factorized Ablation Study Matrix** (linking to Appendix C):
     - `A0` Uncompressed BF16 Baseline
     - `A1` Structural Only (Tier 1 Quantization / Low-Rank)
     - `A2` Spatial Only (Tier 2 Spatial Offload / Recallable Paging `[OPEN_DECISION]`)
     - `A3` Structural + Spatial (No Async Prefetch)
     - `A4` Full 3-Axis Co-Optimization (Structural + Spatial + Temporal Async Stream Pipelining)
- **Required Table Spec — Table 5.3 (`Semester 2 Full Evaluation & Ablation Matrix`):**
  - *Columns:* `Study ID | WP | Configuration / Active Axes | Target Hardware | Workloads & Metrics | Success Gate`.
- **Open Questions / Decisions Needed (`[OPEN_DECISION]`):**
  - Confirm exact subset of secondary models from `docs/MODELS.md` to run in `WP-7` (e.g., 1 dense peer `Phi-4-mini` + 1 hybrid `Qwen3.5-4B` + 1 local 8 GB `Qwen3-1.7B`) given cluster Slurm quota.

---

## Section 5.4 — Follow-up & Next Steps

### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Role (`Progress_Sem1_ENG_Example2.pdf` §5.4):** Summarizes immediate short-term milestones before the Semester 1 Progress Defense and the concrete handoff plan into Semester 2.
- **Committee Expectation:** Actionable, time-bound next steps with clear owners and risk mitigations.

### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** Between now and the Semester 1 Progress Defense, our team will close four concrete items in strict sequence: (1) complete manual review & merge of **PR #21** (`docs/wp3-result-interpretation`), (2) finalize and deliver the **December 1 advisor draft** (`WP-5`), (3) execute **WP-4** hypothesis validation & policy state-machine design (see `[OPEN_DECISION]` below on whether S1 Progress Defense requires a running WP-4 prototype smoke test vs. an architecture specification), and (4) complete **`WP1-5`** local 8 GB RTX 4060 baseline adaptation to ensure clean transition into Semester 2 `WP-6`.
- **Delta from Proposal Defense (`KV-Cache Co-optimization.pdf`):** Tightened from high-level semester milestones to concrete, issue-linked engineering gates tied to GitHub Project #9.

### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)
| Action Item | Linked Issue / PR / Artifact | Status Tag | Target Timing |
| :--- | :--- | :--- | :--- |
| Review & merge WP-3 interpretation + PyTorch baseline | PR #21 (`docs/wp3-result-interpretation`) | `[DRAFT/PR#21]` | Immediate (prior to final Ch. 5 prose) |
| First advisor review draft of S1 Progress Report | `deliverables/s1_progress_report/` (`WP-5`) | `[READY]` (Active) | **December 1, 2026** |
| Local 8 GB RTX 4060 baseline adaptation | `WP1-5` (`Qwen3-1.7B`, `docs/MODELS.md`) | `[NEEDS_VERIFICATION]` | Pre-Sem 1 Defense |
| WP-4 Policy State-Machine Design & Prototype Smoke Test | `WORK_PACKAGES.md` (`WP-4`) | `[OPEN_DECISION]` | Post-PR #21 merge → Pre-Sem 1 Defense |
| Reconcile minor doc drifts (`WORK_PACKAGES.md` badges, `METRICS_AND_AXES.md` 50q→150q, and clarify dual cluster envs in `MODELS.md`: vLLM `CUDA 13.0` vs. PyTorch `CUDA 12.2`) | `WORK_PACKAGES.md`, `docs/METRICS_AND_AXES.md`, `docs/MODELS.md` | `[NEEDS_VERIFICATION]` | Before final S1 submission |
| Semester 2 Kickoff (`WP-6` Runtime Engine Integration) | `WORK_PACKAGES.md` (`WP-6`–`WP-9`) | `[PENDING_S2]` | Start of Semester 2 |

### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline Skeleton:**
  1. Immediate Pre-Draft & Pre-Defense Actions (PR #21 audit → Dec 1 draft → Advisor feedback iteration).
  2. Technical Closure of Semester 1 (`WP1-5` 8 GB local run + `WP-4` policy prototype sanity check).
  3. Semester 2 Execution Handoff (`WP-6` → `WP-7` → `WP-8` → `WP-9`).
- **Required Table Spec — Table 5.4 (`Immediate Action Plan & Risk Mitigation Checklist`):**
  - *Columns:* `Priority | Action Item | Dependency | Target Date | Risk / Fallback`.
- **Open Questions / Decisions Needed (`[OPEN_DECISION]`):**
  - Confirm exact Semester 1 Progress Defense date with Dr. Aye Hninn Khine and Dr. Boonyarit Changaival after the December 1 draft review.
  - Confirm with advisors whether the Semester 1 Progress Defense gate requires a **running `WP-4` prototype smoke test** on the cluster or whether a **validated `WP-4` architecture specification** (backed by WP-3's 11-method evidence) satisfies Semester 1 completion while deferring prototype implementation to Semester 2 kickoff.
