# Chapter 3 Blueprint: Methodology and System Design

> **KMUTT Baseline Mapping:** Chapter 3 — *Methodology and Design* (`Progress_Sem1_ENG_Example2.pdf`, Sections `3.1`–`3.4`)  
> **Primary Repo Sources:** `docs/MODELS.md`, `docs/METRICS_AND_AXES.md`, `docs/WORKFLOW_SPEC.md`, `results/manifests/dataset_manifest.json`, `workloads/download_datasets.py`, `results/manifests/run_manifest_*.json`, `results/baselines/`, `results/runs/wp3/`, `results/grades/`  
> **Blueprint Role:** Templating and agent handoff specification — **not** final thesis prose.

---

## Chapter-Level Scope & Guardrails

- **Scope Compliance (`.agents/AGENTS.md`):**
  - Strictly enforce the **16 GB cluster VRAM cap** (Slurm RTX 4090) and **8 GB local VRAM cap** (RTX 4060).
  - Strictly enforce **unmodified BF16/FP16 pinned base models** from `docs/MODELS.md` (no pre-quantized GGUF/AWQ base weights).
  - Strictly enforce the **Two Primary Quality Pillars** (Micro-Level Logit Divergence + Macro-Level LongMemEval); keep 2D NIAH as an opt-in positional diagnostic and Perplexity ($\Delta\text{PPL}$) as strictly supplementary.
- **Merged WP-3 Artifacts:** `results/baselines/baseline_pytorch_16gb.json`, `analysis/wp3/*`, and `analysis/figures/wp3/*` are merged on `main` (PR #21 `911c1d3`) and `[READY]`.
- **Verified Discrepancies Surfaced in This Chapter (Must Reconcile Before Final Prose):**
  1. **Schema Drift (`longmem_macro_f1` vs. `longmem_f1_overall`):** `docs/METRICS_AND_AXES.md` specifies `extra["longmem_macro_f1"]`, whereas all serialized JSONs in `results/baselines/` and `results/runs/wp3/` emit `extra["longmem_f1_overall"]` (plus per-task `extra["longmem_f1_<task>"]`).
  2. **LongMemEval Sample Count Discrepancy (`50q` vs. `150q`):** `docs/METRICS_AND_AXES.md` (§3) and WP-2 vLLM `longmem_baseline_16gb.json` reference **50 curated prompts** (10/task), whereas `workloads/download_datasets.py`, `results/manifests/dataset_manifest.json`, and all WP-1/WP-3 PyTorch runs + `results/grades/*` evaluate **150 length-quantile-sampled questions** (30/task).
  3. **Dual-Environment Software Version Drift:** `docs/MODELS.md` §4 specifies `CUDA 12.2` / `PyTorch >= 2.6.0` / `vllm >= 0.20.0`, while `run_manifest_baseline_16gb.json` (vLLM env) records `CUDA 13.0` / `PyTorch 2.13.0+cu130` / `vllm 0.30.0`, and `run_manifest_baseline_pytorch_16gb.json` (PyTorch env) records `CUDA 12.2` / `PyTorch 2.6.0`.

---

## Section 3.1: System Architecture

### Block 1 — Section Purpose & KMUTT Baseline Alignment
- **KMUTT Role (`3.1 System Architecture`):** Presents the high-level software/hardware architecture, component boundaries, data flows, and conceptual framework of the system being built.
- **Committee Expectation:** Clear block diagrams showing how engineering infrastructure, benchmarking harness, provenance tracking, and the target co-optimization policy interact.

### Block 2 — Core Message & Delta from Proposal Defense
- **Core Message:** Our architecture operates at two complementary levels:
  1. **Two-Tier Engineering & Provenance Architecture:** Separates the **Purpose/Coordination Plane** (`kv_co-op_hub`: specifications, datasets, SHA256 provenance manifests, golden results, deliverables) from the **Execution Plane** (`kv_bench`: dual-backend harness, 11 KV-cache wrappers, profiling & grading scripts), unified by **GitHub Project #9** (`WP-1`–`WP-9`) and automated Hub CI (`.github/workflows/hub-ci.yml`).
  2. **3-Axis Co-Optimization Framework:** Formalizes KV-cache management into three orthogonal control axes (`STRUCTURAL`, `SPATIAL`, `TEMPORAL`) exposed via `src/core/axis.py` in `kv_bench`, enabling apples-to-apples reproduction of 11 single-/dual-axis techniques in Semester 1 and principled synthesis of a co-optimized policy in WP-4 / Semester 2.
- **Delta from Proposal Defense (`KV-Cache Co-optimization.pdf` & `Slide_Content_and_Executive_Summary.md`):**
  - *Added:* Formal Two-Tier Multi-Repo & SHA256 Provenance architecture (absent in Proposal, where work was tracked in a static spreadsheet).
  - *Evolved:* Proposal introduced a conceptual **3-Layer Policy** (`KV-Cache Co-optimization.pdf` backup Slide 07 [PDF p. 17], with L3 = `TRANSFER SCHEDULER`; and `Slide_Content_and_Executive_Summary.md` Slide 08, with CPU dequantization on Slide 10). In Semester 1, we deliberately decoupled the **3-Axis Evaluation Harness** (`kv_bench`) from the final **WP-4 Policy Redesign**, holding the exact WP-4 policy composition as `[OPEN_DECISION]` across the 4 falsifiable candidate hypotheses synthesized in `analysis/wp3/CONSOLIDATED_PATTERN_REPORT.md` (`[READY]`).

### Block 3 — Strict Data & Source Provenance Map (Agent Handoff Spec)
| Architectural Component | Exact Repo Source / Path | Verified Details / Keys | Readiness |
| :--- | :--- | :--- | :---: |
| Two-Tier Multi-Repo & Project #9 | `docs/WORKFLOW_SPEC.md` (§1–§4) | `kv_co-op_hub` (Hub) + `kv_bench` (Execution) + GitHub Project #9; native cross-repo sub-issues | `[READY]` |
| Provenance Manifest & Hub CI | `docs/WORKFLOW_SPEC.md` (§4.1); `.github/workflows/hub-ci.yml`; `results/manifests/run_manifest_*.json` | Enforces `manifest_version`, `execution_environment`, `source_repository.commit_sha`, `golden_artifacts.*.sha256`, and logprob-bloat guardrail (`results/baselines/references/` only) | `[READY]` |
| 3-Axis Co-Optimization Framework | `docs/METRICS_AND_AXES.md` (§1) | Formal axis enum (`STRUCTURAL`, `SPATIAL`, `TEMPORAL`) in `kv_bench:src/core/axis.py` | `[READY]` |
| Proposal Conceptual 3-Layer Policy | `deliverables/s1_proposal_defense/KV-Cache Co-optimization.pdf` (backup Slide 07, PDF p. 17) & `Slide_Content_and_Executive_Summary.md` (Slides 08 & 10) | Original L1/L2/L3 conceptual baseline for contrast | `[READY]` |
| Evidence-Driven WP-4 Policy Redesign | `analysis/wp3/CONSOLIDATED_PATTERN_REPORT.md` (PR #21 merged `911c1d3`) | Synthesizes WP-3 white-box findings into 4 falsifiable WP-4 candidate hypotheses (`[READY]`; final policy selection `[OPEN_DECISION]`) | `[READY]` / `[OPEN_DECISION]` |

### Block 4 — Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline:**
  - `3.1.1` Two-Tier Multi-Repository & Reproducible Provenance Architecture (`kv_co-op_hub` + `kv_bench` + GitHub Project #9 + Hub CI)
  - `3.1.2` The 3-Axis Co-Optimization Framework (`STRUCTURAL`, `SPATIAL`, `TEMPORAL`)
  - `3.1.3` Conceptual 3-Layer Policy (Proposal Baseline) vs. Evidence-Driven WP-4 Redesign (`[OPEN_DECISION]`)
- **Figure Spec — Fig. 3.1 (Two-Tier System & Data-Flow Architecture Diagram):** Clean redraw/adaptation of the Mermaid diagram in `docs/WORKFLOW_SPEC.md` §1 showing Hub (`kv_co-op_hub`), Execution (`kv_bench` on Slurm `gpu4090` / local RTX 4060), Provenance Manifests + CI gate, and GitHub Project #9. *Source:* `docs/WORKFLOW_SPEC.md` `[READY]`.
- **Figure Spec — Fig. 3.2 (3-Axis Co-Optimization Control Plane Diagram):** Block diagram showing how `STRUCTURAL` (precision/eviction), `SPATIAL` (VRAM/DRAM tiering), and `TEMPORAL` (prefetch/reuse scheduling) plug into the unified `kv_bench` wrapper interface (`BaseKVTechnique` `[NEEDS_VERIFICATION]` — verify exact class/interface name in `khemingkapat/kv_bench`).
- **Open Questions:**
  - `[OPEN_DECISION]` **WP-4 Policy Final Architecture:** From the 4 falsifiable candidate hypotheses in `analysis/wp3/CONSOLIDATED_PATTERN_REPORT.md` (`[READY]`), which specific combination will be validated and locked as the official Semester 2 (`WP-4` / `WP-6`) policy architecture?

---

## Section 3.2: Development Strategies

### Block 1 — Section Purpose & KMUTT Baseline Alignment
- **KMUTT Role (`3.2 Development Strategies`):** Explains the controlled experimental engineering choices—model selection, hardware capacity enforcement, and backend execution strategy—that ensure scientific validity and eliminate confounding variables.
- **Committee Expectation:** Justify *how* variables are controlled (why BF16, why 3B–4B models, why 16 GB/8 GB caps, why two backends) so committee members trust the experimental rigor.

### Block 2 — Core Message & Delta from Proposal Defense
- **Core Message:** Four strict engineering controls govern all experiments:
  1. **Controlled BF16/FP16 Base Models & Commit-SHA Pinning:** 9 open-weight checkpoints in `docs/MODELS.md` locked to immutable Hugging Face commit SHAs, strictly prohibiting pre-quantized GGUF/AWQ base weights so KV-cache dynamics remain unconfounded.
  2. **Pure Dense GQA vs. Architectural Hybrid/SWA Separation:** Isolates Pure Dense GQA workhorses (`Qwen3-4B-Instruct-2507` primary at $144\text{ KiB/tok}$, `Phi-4-mini-instruct`, `Ministral-3-3B-Base-2512`, `SmolLM3-3B`, `Qwen3-1.7B`) and the Literature Reference checkpoint (`Qwen2.5-3B-Instruct`, $36\text{ KiB/tok}$) from Hybrid/SWA architectures (`Qwen3.5-4B`, `Qwen3.5-2B`, `Gemma-4-E4B-it`) that natively suppress KV growth, enforcing `--language-model-only` on multimodal bundles.
  3. **Strict Hardware Capacity Envelopes:** Enforces a **16.0 GiB process allocator cap** on KMUTT CPE Slurm RTX 4090 nodes (`cap_fraction = 0.68`, `effective_gpu_utilization = 0.5783` on 23.52 GiB physical VRAM) to faithfully simulate 16 GB consumer GPUs, with local **8 GB RTX 4060** validation (`Qwen3-1.7B`, `Qwen3.5-2B`) scoped under `WP1-5` (`[OPEN_DECISION]` whether executed prior to S1 defense or formally deferred to S2).
  4. **Dual-Backend Execution Strategy:** Uses **vLLM V1** for production throughput/concurrency baselines (`WP-2`) and **PyTorch Native SDPA** (`WP2-5`, `WP3-1-3`, `WP1-4-1`) for white-box KV-cache prototyping, exact per-layer token retention inspection, and apples-to-apples comparison of all 11 WP-3 techniques.
- **Delta from Proposal Defense (`KV-Cache Co-optimization.pdf` & `Slide_Content_and_Executive_Summary.md`):**
  - *Model Pivot:* Replaced unpinned model families (`Llama-3, Qwen-3, Gemma 3, Mistral` on Slide 10 [PDF p. 11] of `KV-Cache Co-optimization.pdf`; and `Llama-3 8B, Qwen-2.5 1.5B/3B/7B, Mistral 7B` on Slide 11 of `Slide_Content_and_Executive_Summary.md` — where 7B–8B BF16 weights alone consume $\approx 14\text{--}16\text{ GB}$, leaving zero VRAM headroom on a 16 GB cap) with **9 commit-SHA-pinned 1.7B–4B BF16 models** ($\le 8.1\text{ GB}$ weights, leaving $\approx 7.9\text{--}10.0\text{ GB}$ headroom for 32K/multi-user KV sweeps).
  - *Backend Pivot:* Added PyTorch Native SDPA alongside vLLM V1 to unblock white-box instrumentation of all 11 techniques in Semester 1 without fighting vLLM's opaque C++ paged-attention internals.

### Block 3 — Strict Data & Source Provenance Map (Agent Handoff Spec)
| Strategy Pillar | Exact Repo Source & Keys | Verified Facts / Numbers | Readiness |
| :--- | :--- | :--- | :---: |
| Pinned Model Registry (9 Models) | `docs/MODELS.md` (§1–§3) | `Qwen/Qwen3-4B-Instruct-2507` (`cdbee75f17c01a7cc42f958dc650907174af0554`, 36L, 32Q/8KV, $D=128$, $144\text{ KiB/tok}$, $\approx 8.06\text{ GB}$ BF16) + 8 peer/hybrid/8GB/literature-reference models (incl. `Qwen2.5-3B-Instruct`) | `[READY]` |
| 16 GB Allocator Cap Calibration (vLLM) | `results/manifests/run_manifest_baseline_16gb.json` (`execution_environment`) | RTX 4090 (`23.52 GiB`), `process_allocator_cap_gib: 16.0`, `cap_fraction: 0.68`, `effective_gpu_utilization: 0.5783`, `vllm 0.30.0`, `PyTorch 2.13.0+cu130`, `CUDA 13.0` | `[READY]` |
| 16 GB Allocator Cap Calibration (PyTorch) | `results/manifests/run_manifest_baseline_pytorch_16gb.json` (`execution_environment`) | RTX 4090 (`23.52 GiB`), `process_allocator_cap_gib: 16.0`, `PyTorch 2.6.0`, `CUDA 12.2`, SDPA backend | `[READY]` |
| Software Stack Docs Parity | `docs/MODELS.md` (§4) | Lists `vllm >= 0.20.0`, `lmcache >= 0.5.0`, `PyTorch >= 2.6.0`, `CUDA 12.2` (minor drift vs. vLLM manifest `CUDA 13.0`) | `[NEEDS_VERIFICATION]` |
| Local 8 GB RTX 4060 Validation (`WP1-5`) | `docs/MODELS.md` (§3: `Qwen3-1.7B`, `Qwen3.5-2B`); `WORK_PACKAGES.md` (`WP-1`) | Unresolved scheduling: decide whether `WP1-5` runs before S1 defense or is formally deferred to S2 (`WP-6`–`WP-7`) | `[OPEN_DECISION]` |

### Block 4 — Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline:**
  - `3.2.1` Controlled BF16/FP16 Model Selection & Immutable Commit-SHA Pinning
  - `3.2.2` Architectural Stratification: Pure Dense GQA vs. Hybrid Linear / Sliding Window Attention
  - `3.2.3` Strict Consumer Hardware Capacity Envelopes (16 GB Slurm Cap & 8 GB Local Cap)
  - `3.2.4` Dual-Backend Execution Strategy: vLLM V1 vs. PyTorch Native SDPA
- **Table Spec — Table 3.1 (Pinned Model Registry & KV-Cache Allocation Matrix):** Columns: `Role / Class`, `Model Name`, `Pinned Commit SHA (8-char)`, `Layers / Heads (Q:KV) / D`, `BF16 Weights (GB)`, `KV Bytes/Tok`, `1×32K KV (GB)`, `10×32K KV (GB)`. *Source:* `docs/MODELS.md` §2–§3 `[READY]`.
- **Table Spec — Table 3.2 (Dual-Backend Strategy Comparison):** Columns: `Dimension`, `vLLM V1 Backend (WP-2)`, `PyTorch Native SDPA Backend (WP-1/WP-3)`. Rows: Primary Purpose, Software Stack, Concurrency Support, White-Box Layer Inspection, KV Wrapper Extensibility. *Source:* `run_manifest_baseline_16gb.json` & `run_manifest_baseline_pytorch_16gb.json` `[READY]`.
- **Open Questions:**
  - `[OPEN_DECISION]` Confirm whether `WP1-5` (local 8 GB RTX 4060 validation) or any secondary models beyond `Qwen3-4B-Instruct-2507` (and `Qwen3-1.7B` cross-model KL calibration in `verify_cross_model_16gb.json`) will be swept before the Semester 1 final report, or if all peer/Hybrid models (`Phi-4-mini`, `Qwen3.5-4B`, `Gemma-4-E4B-it`) and 8 GB runs are formally scheduled for Semester 2 (`WP-7`).

---

## Section 3.3: Evaluation Plan

### Block 1 — Section Purpose & KMUTT Baseline Alignment
- **KMUTT Role (`3.3 Evaluation Plan`):** Defines every quantitative metric, quality gate, experimental protocol, and comparative baseline used to evaluate system performance and validate hypotheses.
- **Committee Expectation:** Complete, unambiguous metric definitions (with exact units and formulas/schemas) plus a fair, apples-to-apples baseline comparison matrix.

### Block 2 — Core Message & Delta from Proposal Defense
- **Core Message:** Our evaluation protocol jointly measures **Systems/Hardware Efficiency** (Peak VRAM, Host RSS Delta, TTFT, TPOT, Throughput, PCIe Transfer Volume & Peak BW) and **Model Quality Fidelity** via **Two Primary Quality Pillars**:
  1. **Pillar 1 — Micro-Level Logit Divergence:** `top1_agreement`, `kl_div_mean`, `kl_div_p99` computed directly against golden cached BF16 reference logprobs (`results/baselines/references/`).
  2. **Pillar 2 — Macro-Level LongMemEval:** 5 core conversational memory capabilities graded via **both** lexical F1/EM and an **offline semantic LLM judge** (`results/grades/*_judged.json`), augmented by **Post-Hoc Layer Retention & Causal Attribution Diagnostics** (`diag_*` keys).
  3. **Opt-In Positional & Supplementary Diagnostics:** 2D Needle-In-A-Haystack (`niah_*` keys) for positional blind-spot heatmaps, while Perplexity ($\Delta\text{PPL}$) is retained strictly as an optional supplementary check.
  All **11 WP-3 techniques** are evaluated under this identical protocol against uncompressed vLLM (`WP-2`) and uncompressed PyTorch Native SDPA (`WP1-4-1`) baselines.
- **Delta from Proposal Defense (`KV-Cache Co-optimization.pdf`, `Slide_Content_and_Executive_Summary.md`, & `docs/proposal.md`):**
  - *Upgraded:* Logit Divergence and LongMemEval were already introduced on Slide 11 (PDF p. 12) of `KV-Cache Co-optimization.pdf` (and `docs/proposal.md` §3.3.3 already demoted $\Delta\text{PPL}$, whereas `Slide_Content_and_Executive_Summary.md` Slide 12 had listed $\Delta\text{PPL} \le 0.1$ as a primary gate alongside 5 baselines). During Semester 1, we operationalized this into **offline semantic judging** (`results/grades/*_judged.json`), **post-hoc diagnostic causal attribution** (`PASS_FIDELITY_MAINTAINED`, `QUALITY_REGRESSION`, `BASELINE_LIMITATION`, `UNEXPECTED_GAIN`; note `PASS`/`FAIL` exist only in `results/grades/baseline_judged.json`), and **opt-in 2D NIAH** sweeps (`*_niah_pytorch_16gb.json`).
  - *Expanded Baselines:* Grew from 7 techniques surveyed on Slides 05/08 of `KV-Cache Co-optimization.pdf` (5 baselines in `Slide_Content_and_Executive_Summary.md` Slide 12) to **11 empirically benchmarked WP-3 candidate techniques** across all 3 axes.

### Block 3 — Strict Data & Source Provenance Map (Agent Handoff Spec)

#### 3.3.1 Metrics & JSON Schema Provenance (Including Verified Schema Drift)
| Metric Group | Spec in `docs/METRICS_AND_AXES.md` | Verified Actual JSON Keys in `results/` | Readiness |
| :--- | :--- | :--- | :---: |
| Hardware & Latency (`ResultEntry`) | `peak_alloc_mb`, `peak_reserved_mb`, `host_rss_delta_mb`, `ttft_sec`, `cold_ttft_sec`, `ttft_p95_sec`, `throughput_tok_per_sec`, `tpot_mean_ms`, `tpot_p95_ms`, `memcpy_bytes_per_token`; line 34 also lists `memcpy_htod_bytes`, `memcpy_peak_bw_gbs` | Top-level `ResultEntry` keys match 1-to-1 in `results/baselines/baseline_hardware_16gb.json` & `results/runs/wp3/*_hardware_pytorch_16gb.json`. **Schema drift note:** PCIe profiling fields (`extra["memcpy_htod_bytes"]`, `extra["memcpy_dtoh_bytes"]`, `extra["memcpy_peak_bw_gbs"]`) are emitted under `results[0].extra` in WP-3 `*_hardware_pytorch_16gb.json` sweeps (whereas `baseline_hardware_16gb.json` omits them, and `METRICS_AND_AXES.md` line 34 lists them without the `extra[...]` prefix or `memcpy_dtoh_bytes`) | `[READY]` / `[NEEDS_VERIFICATION]` |
| Pillar 1: Micro Logit Divergence | `extra["top1_agreement"]`, `extra["kl_div_mean"]`, `extra["kl_div_p99"]` | Verified 1-to-1 match in `results/runs/wp3/*_hardware_pytorch_16gb.json` & `*_longmem_pytorch_16gb.json`; golden refs in `results/baselines/references/baseline_qwen3-4b_ref.json` (vLLM) & `baseline_qwen3-4b_pytorch_ref.json` (PyTorch) | `[READY]` |
| Pillar 2: Macro LongMemEval (Raw Lexical) | `extra["longmem_acc_overall"]`, `extra["longmem_macro_f1"]`, `extra["longmem_acc_<task>"]` | **SCHEMA DRIFT:** Actual JSONs (`longmem_baseline_16gb.json`, `longmem_baseline_pytorch_16gb.json`, `wp3/*_longmem_pytorch_16gb.json`) emit **`extra["longmem_f1_overall"]`** and `extra["longmem_f1_<task>"]` (NOT `longmem_macro_f1`)! | `[NEEDS_VERIFICATION]` |
| Pillar 2: Macro LongMemEval (Semantic Judge) | *(Added post-WP-2; not yet documented in `METRICS_AND_AXES.md`)* | `results/grades/*_judged.json`: top-level `overall_accuracy`, `per_task_accuracy`, `per_task_counts`, `causal_verdict_counts` (`PASS_FIDELITY_MAINTAINED`, `QUALITY_REGRESSION`, `BASELINE_LIMITATION`, `UNEXPECTED_GAIN` in WP-3; `PASS`/`FAIL` in `baseline_judged.json`), `total_samples` (150), `correct_samples` | `[READY]` |
| Post-Hoc Retention Diagnostics | *(Added in `WP1-4-1`)* | Emitted under `results[0].extra` in `results/baselines/longmem_baseline_pytorch_16gb.json` & `results/runs/wp3/*_longmem_pytorch_16gb.json`: `diag_total_inspected`, `diag_pass_count`, `diag_quality_regression_count`, `diag_baseline_limitation_count`, `diag_mean_overall_retention_ratio`, `diag_mean_overall_effective_bitwidth`, `diag_mean_compression_ratio` | `[READY]` |
| Opt-In 2D NIAH Diagnostic | *(Added in WP-3)* | `results/runs/wp3/*_niah_pytorch_16gb.json`: `extra["niah_accuracy"]`, `niah_depths`, `niah_grid`, `niah_retrieved_count`, `niah_total_depths` | `[READY]` |
| Supplementary Perplexity ($\Delta\text{PPL}$) | `docs/METRICS_AND_AXES.md` (§3.3) | Explicitly deprioritized / optional diagnostic | `[READY]` |

#### 3.3.2 Comparative Baselines & 11 WP-3 Candidates
| Baseline / Candidate | Backend | Verified Result Artifacts on `main` | Manifest File | Readiness |
| :--- | :--- | :--- | :--- | :---: |
| **Uncompressed vLLM Baseline (WP-2)** | vLLM V1 (`0.30.0`) | `results/baselines/baseline_hardware_16gb.json`, `concurrent_baseline_16gb.json`, `longmem_baseline_16gb.json`, `verify_identity_16gb.json`, `verify_cross_model_16gb.json` | `results/manifests/run_manifest_baseline_16gb.json` | `[READY]` |
| **Uncompressed PyTorch Baseline (WP-1/WP-3)** | PyTorch Native SDPA (`2.6.0`) | `results/baselines/baseline_pytorch_16gb.json`, `longmem_baseline_pytorch_16gb.json`, `verify_baseline_pytorch_16gb.json`, `results/grades/baseline_judged.json` | `results/manifests/run_manifest_baseline_pytorch_16gb.json` | `[READY]` |
| **11 WP-3 Techniques:** `streamingllm`, `h2o`, `epicache`, `qserve_kv4`, `cachegen`, `headq`, `turboquant`, `flexgen_kv`, `infinigen`, `shadowkv`, `arkvale` | PyTorch Native SDPA (`2.6.0`) *(+ legacy vLLM `WP3-1` `streamingllm`)* | Each has 4 files in `results/runs/wp3/`: `<tech>_hardware_pytorch_16gb.json`, `<tech>_longmem_pytorch_16gb.json`, `<tech>_niah_pytorch_16gb.json`, `verify_<tech>_pytorch_16gb.json` + 2 files in `results/grades/`: `<tech>_grades.json`, `<tech>_judged.json` *(plus legacy vLLM `streamingllm_hardware_16gb.json` & `streamingllm_longmem_16gb.json`)* | Exact 12 manifests in `results/manifests/`: `run_manifest_wp3_h2o_pytorch.json` (not `_h2o.json`), `run_manifest_wp3_streamingllm.json` (legacy vLLM `WP3-1`), `run_manifest_wp3_streamingllm_pytorch.json` (PyTorch `WP3-1-3`), plus 9 `run_manifest_wp3_<tech>.json` (`arkvale`, `cachegen`, `epicache`, `flexgen_kv`, `headq`, `infinigen`, `qserve_kv4`, `shadowkv`, `turboquant`) | `[READY]` |

### Block 4 — Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline:**
  - `3.3.1` Hardware & Systems Efficiency Metrics (VRAM, Host RSS, TTFT, TPOT, Throughput, PCIe Profiling)
  - `3.3.2` Quality Evaluation Protocol: Two Primary Pillars (Micro Logit Divergence & Macro LongMemEval + Semantic Judge + Post-Hoc Diagnostics), Opt-In 2D NIAH, and Supplementary $\Delta\text{PPL}$
  - `3.3.3` Comparative Baseline Suite (Uncompressed vLLM & PyTorch Baselines vs. 11 Single-/Dual-Axis WP-3 Techniques)
  - `3.3.4` Reproducibility & Logprob-Bloat Guardrails (Deterministic Greedy Decoding, Golden Reference Isolation, SHA256 Manifests)
- **Table Spec — Table 3.3 (Complete Metrics-to-Axes Correspondence Table):** Columns: `Metric`, `Exact JSON Key (Verified)`, `Unit`, `Axis`, `Role (Primary / Diagnostic / Supplementary)`.
- **Table Spec — Table 3.4 (Comparative Baseline Matrix — 2 Uncompressed + 11 WP-3 Techniques):** Columns: `Technique`, `Axis Coverage`, `Backend`, `Workloads Run (HW / LongMem / NIAH / Verify)`, `Manifest ID`.
- **Figure Spec — Fig. 3.3 (Quality Evaluation Pipeline Diagram):** Flowchart adapted from `docs/METRICS_AND_AXES.md` §3, updated to show the actual `kv_bench` pipeline: Pillar 1 (Logit Divergence vs. `*_ref.json`), Pillar 2 (LongMemEval 150q $\rightarrow$ Lexical + Offline Semantic Judge + Post-Hoc `diag_*` Layer Inspector), Opt-In 2D NIAH, and Supplementary $\Delta\text{PPL}$.
- **Open Questions:**
  - `[NEEDS_VERIFICATION]` Update `docs/METRICS_AND_AXES.md` to fix `longmem_macro_f1` $\rightarrow$ `longmem_f1_overall` and document the offline semantic judge (`results/grades/*_judged.json`), post-hoc `diag_*` keys, and opt-in 2D NIAH (`niah_*` keys) so docs and JSON artifacts match 100%.
  - `[OPEN_DECISION]` Confirm which exact judge model/prompting setup generated `results/grades/*_grades.json` so Chapter 3.3 can cite the offline semantic judge transparently and reproducibly.

---

## Section 3.4: Datasets

### Block 1 — Section Purpose & KMUTT Baseline Alignment
- **KMUTT Role (`3.4 Datasets`):** Specifies every dataset used for benchmarking, its upstream source, preprocessing/sampling strategy, token-length distribution, and integrity verification.
- **Committee Expectation:** Clear justification of *why* each dataset maps to a specific research question, exact sample counts, and verifiable provenance (URLs, commit SHAs, SHA256 checksums).

### Block 2 — Core Message & Delta from Proposal Defense
- **Core Message:** Our benchmark suite stages four complementary workloads with cryptographic provenance (`results/manifests/dataset_manifest.json` generated by `workloads/download_datasets.py`):
  1. **LongMemEval (Primary Macro Quality):** Deterministically length-quantile-sampled (**150 questions**, 30 per capability across 5 capabilities, excluding subjective `single-session-preference`) from pinned upstream `xiaowu0162/longmemeval-cleaned` (`longmemeval_oracle.json`, commit `98d7416c...`), stratified into `short` ($\le 2500$ approx. tokens), `medium` ($\le 8000$), and `long` ($> 8000$) tiers.
  2. **PG-19 (Primary Long-Context Hardware & Logit Stream):** 5 public-domain Project Gutenberg test books (`10146`, `10321`, `10356`, `10762`, `12204`), where 4 books exceed 32K tokens (`55.9K`–`75.9K` approx. tokens) for continuous non-repetitive prefixes up to 32,768 exact tokens, while book `12204` (`5,905` approx. tokens) serves shorter prefix sweeps.
  3. **Synthetic 2D NIAH (Opt-In Positional Diagnostic):** Controlled synthetic passcode-retrieval grids across varying context lengths and depth percentages (`0%`–`100%`) to expose lost-in-the-middle and window-eviction blind spots.
  4. **WikiText-2 / PG-19 Perplexity (Supplementary Diagnostic):** Retained strictly as an optional syntactic fluency diagnostic; not a primary quality gate.
- **Delta from Proposal Defense & Early Semester 1 Specs (`docs/METRICS_AND_AXES.md` §3, `WP-2`/`WP3-1`):**
  - *Upgraded LongMemEval Staging:* Expanded from the early 50-question padded sweeps specified in `docs/METRICS_AND_AXES.md` §3 and executed in `WP-2`/`WP3-1` vLLM runs (`longmem_baseline_16gb.json`, `streamingllm_longmem_16gb.json`) to **150 deterministic length-quantile-sampled questions** (`workloads/download_datasets.py:select_quantile_samples`) spanning natural `short`/`medium`/`long` dialogue lengths with full SHA256 verification.
  - *Added Opt-In 2D NIAH:* Introduced synthetic depth-vs-length passcode retrieval sweeps in WP-3 (`*_niah_pytorch_16gb.json`) to visually diagnose eviction blind spots alongside LongMemEval.

### Block 3 — Strict Data & Source Provenance Map (Agent Handoff Spec)
| Dataset | Upstream Source & Pinned Revision | Staging Script & Local Manifest Path | Verified Integrity / Sample Stats | Readiness |
| :--- | :--- | :--- | :--- | :---: |
| **LongMemEval** (5 Capabilities) | `xiaowu0162/longmemeval-cleaned` (`longmemeval_oracle.json`, SHA `98d7416c24c778c2fee6e6f3006e7a073259d48f`) | `workloads/download_datasets.py` (`stage_longmemeval`, `select_quantile_samples`, `TASK_TYPE_MAP`); `results/manifests/dataset_manifest.json` (`datasets.longmemeval.file`) | `sample_count: 150` (30 each: `fact_recall`, `cross_session_reasoning`, `information_update`, `relative_timeframe`, `unanswerable_refusal`); `size_bytes: 4197494`; SHA256 `bc37f751...30ad63` | `[READY]` |
| **LongMemEval Count Discrepancy** | `docs/METRICS_AND_AXES.md` (line 83), `run_manifest_baseline_16gb.json` (`longmem_baseline_16gb`), & `run_manifest_wp3_streamingllm.json` (`streamingllm_longmem_16gb.json`) | Says **50 curated prompts** (10/task, PG-19 padded) vs. **150 quantile-sampled questions** in `dataset_manifest.json` & WP-1/WP-3 PyTorch runs | Must reconcile in `METRICS_AND_AXES.md` and explain WP-2/WP3-1 (50q vLLM) vs. WP-1/WP-3 (150q PyTorch) evolution explicitly in Ch. 3.4 | `[NEEDS_VERIFICATION]` |
| **PG-19 Test Books** | `https://storage.googleapis.com/deepmind-gutenberg/test` (Book IDs: `10146`, `10321`, `10356`, `10762`, `12204`) | `workloads/download_datasets.py` (`DEFAULT_PG19_TEST_BOOKS`); `results/manifests/dataset_manifest.json` (`datasets.pg19.files`) | 4 books exceeding 32K (`10146`: `55,931`, `10321`: `70,001`, `10356`: `67,645`, `10762`: `75,868` approx. tokens) + shorter book `12204` (`5,905` approx. tokens); individual SHA256 checksums pinned in `EXPECTED_PG19_SHA256` | `[READY]` |
| **Synthetic 2D NIAH** | Generated on-the-fly in `kv_bench` | `results/runs/wp3/*_niah_pytorch_16gb.json` (`extra["niah_depths"]`, `niah_grid`) | Opt-in positional retrieval diagnostic across depth × context grid | `[READY]` |
| **WikiText-2 (Supplementary $\Delta\text{PPL}$)** | `docs/METRICS_AND_AXES.md` (§3.3) | Planned/optional diagnostic; `docs/literature_survey/benchmarks/pg19_wikitext2.md` is currently `⚪ Planned` | Supplementary only; not staged in `dataset_manifest.json` | `[NEEDS_VERIFICATION]` |

### Block 4 — Draft Section Skeleton, Table/Figure Specs & Open Questions
- **Outline:**
  - `3.4.1` LongMemEval: Multi-Session Conversational Memory Corpus (5 Capabilities, Deterministic Length-Quantile Sampling, 50q $\rightarrow$ 150q Evolution)
  - `3.4.2` PG-19: Continuous Long-Context Stream for Hardware Sweeps & Logit Divergence (512 $\rightarrow$ 32,768 Tokens)
  - `3.4.3` Synthetic 2D NIAH & Supplementary Language Modeling Corpora (WikiText-2)
  - `3.4.4` Dataset Staging & Cryptographic Provenance (`workloads/download_datasets.py` & `dataset_manifest.json`)
- **Table Spec — Table 3.5 (Staged Benchmark Dataset Summary):** Columns: `Dataset`, `Primary Evaluation Role`, `Upstream Source / Pinned Revision`, `Sample / Token Scale`, `SHA256 Manifest Verification`. *Source:* `results/manifests/dataset_manifest.json` & `workloads/download_datasets.py` `[READY]`.
- **Table Spec — Table 3.6 (LongMemEval Capability Mapping & Stratification):** Columns: `Canonical Capability`, `Upstream LongMemEval question_type`, `Staged Count (150q)`, `Memory Stress Tested`. *Source:* `workloads/download_datasets.py` (`TASK_TYPE_MAP`, `classify_length_tier`) `[READY]`.
- **Open Questions:**
  - `[NEEDS_VERIFICATION]` Reconcile `docs/METRICS_AND_AXES.md` line 83 ("50 curated multi-turn conversational histories") with `dataset_manifest.json` (`150` quantile-sampled questions) before final report drafting.
  - `[OPEN_DECISION]` For Semester 2 (`WP-7`), will any additional real-world conversational dataset (e.g., LoCoMo or RealTalk, already reviewed in `docs/literature_survey/benchmarks/`) be added as a generalization check, or will LongMemEval (150q) + PG-19 + 2D NIAH remain the frozen benchmark suite?
