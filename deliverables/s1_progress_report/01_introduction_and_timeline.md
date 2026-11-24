# Chapter 1 Blueprint: Project Introduction & Timeline (`01_introduction_and_timeline.md`)

> **Report Scope:** KMUTT Senior Project Semester 1 Progress Report — Chapter 1 (`1.1 Project Introduction`, `1.2 Project Timeline`)  
> **KMUTT Baseline:** `Progress_Sem1_ENG_Example2.pdf` (Chapter 1: `1.1`–`1.2`)  
> **Status:** Blueprint & Agent Handoff Spec (NOT final polished prose)  
> **Last Verified Against Workspace & GitHub Project #9:** 2026-11-24

---

## Section 1.1: Project Introduction

### Subsection 1.1.1: Problem Statement

#### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Mapping:** Chapter 1, Section `1.1.1` of `Progress_Sem1_ENG_Example2.pdf`. Establishes the engineering motivation, quantitative bottleneck severity, failure modes of existing single-axis mitigations, and the exact problem statement that justifies the project.
- **Committee Expectations:** Show concrete, mathematically grounded memory/bandwidth constraints on consumer GPUs (not generic LLM hype), clearly distinguish the decode-phase bottleneck from prefill, and articulate *why* isolated fixes fail.

#### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:**
  1. **The Linear KV-Cache Memory Wall:** During autoregressive decoding, KV-cache size scales linearly with sequence length $S_{\text{context}}$ and concurrent sessions $N$. On a 16 GB VRAM consumer GPU running an unmodified BF16 4B Dense GQA model (`Qwen3-4B-Instruct-2507`, weights $\approx 8.06\text{ GB}$, KV rate $= 144\text{ KiB/tok}$), a single 32K context consumes $4.50\text{ GB}$ of KV-cache, while 10 concurrent SME users at 32K require **$45.0\text{ GB}$ of KV-cache alone** — causing immediate CUDA OOM.
  2. **The Decode Bandwidth & PCIe Wall:** Autoregressive decoding is memory-bandwidth bound. Naive CPU offloading hits a severe PCIe wall (e.g., RTX 4060 internal VRAM bandwidth $\approx 272\text{ GB/s}$ vs. host-to-device PCIe $\approx 16\text{--}31.5\text{ GB/s}$), creating GPU compute stall bubbles.
  3. **The Co-Optimization Gap:** Isolated techniques fail under consumer constraints:
     - *Eviction* (e.g., H2O, StreamingLLM) permanently discards tokens, causing unrecoverable factual amnesia in multi-turn chats.
     - *Quantization* (e.g., QServe, HeadQ, TurboQuant) compresses representation, but cache still grows linearly and introduces precision distortion.
     - *Offloading* (e.g., FlexGen) expands capacity via host DRAM, but stalls on PCIe transfer bandwidth.
     - *Coupled Datacenter Systems* (e.g., ArkVale, InfiniGen, ShadowKV, CacheGen) assume datacenter HBM/NVLink, large-batch throughput, or network streaming rather than single-GPU 8–16 GB consumer hardware.
- **Delta from Proposal Defense (`KV-Cache Co-optimization.pdf` PDF pp. 2–3 [labeled `01–02`], pp. 5–6 [labeled `04–05`] & `docs/proposal.md` §1.1):**
  - Proposal cited generic Llama-2-7B / Llama-3.1-8B 16–32 GB KV estimates. Semester 1 replaces these with **exact, empirically verified BF16 footprints** from `docs/MODELS.md` (`Qwen3-4B-Instruct-2507` @ `144 KiB/tok`, `Qwen3-1.7B` @ `112 KiB/tok` for 8 GB local) and distinguishes **Pure Dense GQA** models from **Architectural Hybrid / SWA** models (`Qwen3.5-4B` @ `32 KiB/tok` + GDN, `Gemma-4-E4B-it`).
  - Strengthens the quality failure argument using long-context evaluation literature (`docs/literature_survey/foundations/longeval.md`, `docs/literature_survey/benchmarks/longmemeval.md`, and `docs/literature_survey/benchmarks/seal.md`): unweighted perplexity averaging masks catastrophic needle/fact loss in long multi-turn chats. (By contrast, `docs/literature_survey/foundations/illusion_of_equivalence.md` documents FP16 non-associativity divergence between cached and cache-free execution.)

#### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)

| Data Element | Exact Source Path / Reference | Readiness Tag | Notes for Future Agent |
| :--- | :--- | :---: | :--- |
| Proposal Problem Statement & Consumer Floor Motivation | `docs/proposal.md` (Lines 31–42), `deliverables/s1_proposal_defense/Slide_Content_and_Executive_Summary.md` (Lines 51–86) | `[READY]` | Retain RTX 4060 272 GB/s VRAM vs. ~16 GB/s PCIe contrast and local LLM gateway motivation. |
| Exact BF16 KV-Cache Math & OOM Allocation Matrix | `docs/MODELS.md` (Lines 45–63) | `[READY]` | Use exact table values: `Qwen3-4B-Instruct-2507` (`8.06 GB` weights, `144 KiB/tok`, `4.50 GB` @ 1x32K, `45.0 GB` @ 10x32K OOM). |
| 3-Axis Co-Optimization Gap Framing | `docs/METRICS_AND_AXES.md` (Lines 7–21), `docs/proposal.md` (Lines 160–174) | `[READY]` | Formalizes `STRUCTURAL`, `SPATIAL`, `TEMPORAL` trade-offs. |
| Empirical WP-2 Native vLLM OOM Cliff Evidence | `results/baselines/baseline_hardware_16gb.json`, `results/baselines/concurrent_baseline_16gb.json`, `results/manifests/run_manifest_baseline_16gb.json` | `[READY]` | Can forward-reference Ch. 5 empirical OOM cliff to prove the problem is experimentally reproduced, not merely theoretical. |

#### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions

#### Draft Skeleton (`1.1.1`)
1. **Rise of Local Long-Context LLM Serving:** Growing SME, student, and researcher adoption of private local LLMs for multi-turn chat, document review, and codebase analysis.
2. **Mathematical Root Cause (Linear KV-Cache Scaling):** Present the GQA BF16 formula from `docs/MODELS.md` and contrast 1x vs. 10x 32K VRAM requirements against 8 GB / 16 GB consumer hardware ceilings.
3. **Decode-Phase Bandwidth & PCIe Bottleneck:** Distinguish compute-bound prefill from memory-bandwidth-bound decode; explain why moving KV tensors across PCIe introduces latency bubbles.
4. **Failure of Isolated Optimizations & The Co-Optimization Gap:** Summarize why pure eviction, pure quantization, pure offloading, and datacenter-oriented coupled systems leave an unresolved co-optimization gap on 8–16 GB consumer GPUs.
5. **Project Goal Statement:** Quote/align directly with `WORK_PACKAGES.md` (Line 4).

#### Open Questions / Decisions Needed
- None blocking for `1.1.1`; ensure forward pointer to Chapter 5 (`WP-2` OOM cliff) stays concise so Ch. 1 does not duplicate Ch. 5 plots.

---

### Subsection 1.1.2: Research Questions & Objectives

#### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Mapping:** Chapter 1, Section `1.1.2` of `Progress_Sem1_ENG_Example2.pdf`. Defines measurable, falsifiable research objectives (and optionally explicit research questions) against which Semester 1 progress and Semester 2 final evaluation will be graded.
- **Committee Expectations:** Clear traceability from Problem Statement (`1.1.1`) → Objectives (`1.1.2`) → Work Packages (`1.2.1`) → Evaluation Metrics (Ch. 3 & Ch. 5).

#### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** State the 3 core engineering/scientific objectives (Profiling, 3-Axis Policy Co-Design, Comparative Evaluation) anchored to quantitative hardware caps and the Two Primary Quality Pillars.
- **Delta from Proposal Defense:**
  - `docs/proposal.md` (§1.2) listed 3 qualitative objectives, and the delivered Proposal Defense PDF (`KV-Cache Co-optimization.pdf` Slide 03) stayed qualitative on objectives; by contrast, the companion summary `Slide_Content_and_Executive_Summary.md` (Slides 04 & 12) added 4 aggressive quantitative targets (*95–98% quality retention, $\le 0.1$ $\Delta\text{PPL}$, fit 32K in 8 GB VRAM, 30–40% throughput speedup, sub-100ms TPOT*).
  - **Critical Scope Correction (`.agents/AGENTS.md` & `docs/METRICS_AND_AXES.md`):**
    1. **Quality Gate Refinement:** Replace $\Delta\text{PPL} \le 0.1$ as a primary objective with the **Two Primary Quality Pillars**: (1) Micro-Level Logit Divergence (`top1_agreement`, `kl_div_mean`, `kl_div_p99`) and (2) Macro-Level LongMemEval across 5 core conversational memory capabilities. Perplexity is explicitly demoted to an optional supplementary diagnostic.
    2. **Quantitative Target Calibration:** While raw WP-3 benchmark sweeps and semantic grades (`results/runs/wp3/`, `results/grades/`) are `[READY]` on `main`, WP-3 comparative synthesis (`analysis/wp3/*`, `results/baselines/baseline_pytorch_16gb.json`, `[DRAFT/PR#21]`) and WP-4 policy design are still in progress; therefore, Semester 1 should frame the executive summary's numerical targets as *aspirational Semester 2 design targets* rather than pre-committed guarantees.

#### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)

| Data Element | Exact Source Path / Reference | Readiness Tag | Notes for Future Agent |
| :--- | :--- | :---: | :--- |
| 3 Formal Proposal Objectives | `docs/proposal.md` (Lines 44–49) | `[READY]` | Profiling bottlenecks; Developing modular 3-axis policy layer; Evaluating against baselines & consolidating benchmarking report. |
| Quantitative Slide Targets | `deliverables/s1_proposal_defense/Slide_Content_and_Executive_Summary.md` (Lines 89–101, 237–241) | `[OPEN_DECISION]` | Reconcile whether to retain quantitative targets (e.g., 50–75% VRAM reduction, 30–40% speedup over naive offload, sub-100ms TPOT) as S2 stretch targets or use qualitative/Pareto-dominance objectives. |
| Two Primary Quality Pillars | `docs/METRICS_AND_AXES.md` (Lines 56–97), `.agents/AGENTS.md` (Section 2.B) | `[READY]` | Mandatory scope rule: Logit Divergence + LongMemEval are primary; $\Delta\text{PPL}$ is supplementary only. |

#### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions

#### Draft Skeleton (`1.1.2`)
- **Optional Research Questions (`[OPEN_DECISION]`):**
  - *RQ1 (Bottleneck Characterization):* Where do VRAM OOM ceilings, P95/P99 latency spikes, and PCIe saturation cliffs occur for unmodified BF16 LLMs on 8–16 GB consumer GPUs across 512–32K contexts and 10–15 concurrent sessions?
  - *RQ2 (Single-Axis Trade-Offs):* How do existing structural, spatial, and temporal KV-cache techniques trade VRAM and decode speed against Micro Logit Divergence and Macro LongMemEval memory adherence under controlled BF16 base weights?
  - *RQ3 (3-Axis Co-Optimization):* Can a modular co-optimization policy jointly coordinating structural compression, spatial tiering, and temporal scheduling expand the Pareto frontier on consumer GPUs without offsetting PCIe/CPU overhead?
- **Project Objectives (Aligned to `docs/proposal.md` §1.2 + S1 Refinements):**
  1. **Objective 1 (Bottleneck Profiling — `WP-1`, `WP-2`):** Profile VRAM OOM cliffs, TTFT/TPOT tail latency (P95/P99), host RSS, and PCIe transfer bottlenecks of unmodified BF16 LLMs on 8–16 GB consumer GPUs across scaling contexts (512–32K) and multi-user concurrency.
  2. **Objective 2 (Empirical Taxonomy & 3-Axis Co-Optimization Policy — `WP-3`, `WP-4`, `WP-6`):** Micro-benchmark published structural, spatial, and temporal KV-cache techniques under a unified harness, and design/integrate a modular 3-axis policy layer that coordinates compression, tiered placement, and transfer scheduling.
  3. **Objective 3 (Comprehensive Evaluation & Reproducible Artifacts — `WP-7`, `WP-8`, `WP-9`):** Evaluate the co-optimized system against eviction, quantization, and offloading baselines on hardware efficiency and the Two Primary Quality Pillars (Micro Logit Divergence and Macro LongMemEval), releasing fully reproducible manifests and scripts.

#### Open Questions / Decisions Needed
1. **`[OPEN_DECISION]` Formal RQs + Objectives vs. Objectives Only:** Confirm with advisors whether Section `1.1.2` should present both formal Research Questions (RQ1–RQ3) and Objectives, or only numbered Objectives (matching `docs/proposal.md` §1.2).
2. **`[OPEN_DECISION]` Quantitative Target Thresholds:** Confirm whether to keep the Proposal Slide 04/12 numerical targets (30–40% throughput speedup, 95–98% LongMemEval retention, sub-100ms TPOT) explicitly in `1.1.2` as Semester 2 target thresholds, or state Pareto-improvement objectives until WP-4 is finalized.

---

### Subsection 1.1.3: Scope

#### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Mapping:** Chapter 1, Section `1.1.3` of `Progress_Sem1_ENG_Example2.pdf`. Explicitly bounds what is **in-scope** and **out-of-scope** across hardware, models, software architecture, workloads, and evaluation metrics.
- **Committee Expectations:** Prevent scope creep, defend why specific base models and hardware caps were chosen, and transparently document any scope refinements made since the Proposal Defense.

#### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** The project is strictly scoped to **decode-phase KV-cache co-optimization** for **unmodified, SHA-pinned BF16 open-weight LLMs** on **single consumer GPUs (16 GB cluster cap / 8 GB local cap)**, evaluated via a **dual-backend harness** (vLLM + PyTorch Native SDPA) under **Two Primary Quality Pillars** (Logit Divergence + LongMemEval). Model training, fine-tuning, and pre-quantized base weights (GGUF/AWQ/GPTQ) are explicitly out of scope.
- **Delta from Proposal Defense (`KV-Cache Co-optimization.pdf` & `docs/proposal.md` §1.3):**
  - **Model Suite Evolution:** Proposal listed generic 7B–8B families (*Llama-3 8B, Qwen-2.5 7B, Gemma 3, Mistral 7B*). Under a strict 16 GB VRAM cap with unmodified BF16 weights, 7B–8B weights alone consume ~14–16 GB, leaving zero KV-cache headroom. In `WP1-1` (`docs/MODELS.md`), we pinned **9 exact Hugging Face BF16 checkpoints** — with $\le 8.1\text{ GB}$ BF16 weights (leaving $\ge 7.9\text{ GB}$ KV headroom on 16 GB) verified across the **7 documented models** in `docs/MODELS.md` §3 (`Qwen3.5-2B` and `Gemma-4-E4B-it` BF16 weight footprints are `[NEEDS_VERIFICATION]`) — led by `Qwen/Qwen3-4B-Instruct-2507` (Pure Dense GQA, `144 KiB/tok`) and `Qwen/Qwen3-1.7B` (`112 KiB/tok`) for 8 GB local adaptation, plus Architectural Hybrid / SWA peers (`Qwen3.5-4B`, `Gemma-4-E4B-it`).
  - **Dual-Backend Execution Engine:** Proposal assumed vLLM + custom C++/CUDA kernels for all experiments. Semester 1 introduced a **Dual-Backend Harness** (`WP2-5`, `WP3-1-3`): **vLLM** (`vllm 0.30.0`) for production/concurrency baselines, and **Unified PyTorch Native SDPA** for white-box KV-cache prototyping and post-hoc layer survival diagnostics (`WP1-4-1`), with production vLLM runtime integration scheduled for Semester 2 (`WP-6`).
  - **Quality Evaluation Refinement:** Upgraded LongMemEval ingestion to 150 balanced prompts (30 per task across 5 capabilities in `results/manifests/dataset_manifest.json`) with offline semantic grading (`results/grades/`), plus opt-in 2D NIAH diagnostics; Perplexity ($\Delta\text{PPL}$) is strictly supplementary.

#### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)

| Scope Dimension | Exact Source Path / Reference | Readiness Tag | Notes for Future Agent |
| :--- | :--- | :---: | :--- |
| Hardware Caps & Base Weight Rules | `.agents/AGENTS.md` (Section 2.B), `docs/MODELS.md` (Lines 14–21) | `[READY]` | 16 GB cluster VRAM cap (RTX 4090 on `gpu01.slurm.cpe.kmutt.ac.th`), 8 GB local VRAM cap (RTX 4060, 16 GB RAM); unmodified BF16/FP16 weights only (no GGUF/AWQ/GPTQ). |
| Pinned 9-Model Registry | `docs/MODELS.md` (Lines 28–42) | `[READY]` | Cite Pure Dense GQA vs. Hybrid / SWA categorization and `--language-model-only` flag for multimodal checkpoints. |
| Proposal Scope Subsections (Solutions, Features, Users, Workloads) | `docs/proposal.md` (Lines 50–67) | `[READY]` | Retain 4-part structure (`Solutions`, `Features & Constraints`, `Target Users & Operating Environment`, `Workload Specification`). |
| Dataset Manifest (PG-19 + LongMemEval 150q) | `results/manifests/dataset_manifest.json` | `[READY]` | LongMemEval: `150` samples (`30` each across `fact_recall`, `cross_session_reasoning`, `information_update`, `relative_timeframe`, `unanswerable_refusal`); PG-19: 5 books (`book_10146`, `10321`, `10356`, `10762`, `12204`). |
| Proposal-to-S1 Delta Table | `deliverables/s1_progress_report/README.md` (Lines 53–63) | `[READY]` | Direct source for `Table 1.1`. |

#### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions

#### Draft Skeleton (`1.1.3`)
1. **Solutions Scope:** 3-Axis Co-Optimization (`STRUCTURAL`, `SPATIAL`, `TEMPORAL`) targeting autoregressive decode-phase KV-cache management. Explicitly out-of-scope: model training/fine-tuning, weight-only quantization (GGUF/AWQ/GPTQ), and multi-GPU tensor parallelism.
2. **Hardware, Model & Software Constraints:**
   - Hardware: 16 GB VRAM-capped RTX 4090 cluster node (`gpu01.slurm.cpe.kmutt.ac.th`) + 8 GB VRAM local RTX 4060 adaptation (`kv_bench#6`).
   - Models: 9 SHA-pinned unmodified BF16 checkpoints (`docs/MODELS.md`), separated into Pure Dense GQA vs. Architectural Hybrid / SWA.
   - Software: Dual-Backend Harness (vLLM V1 + PyTorch Native SDPA) under the Two-Tier Repo Architecture (`kv_co-op_hub` + `kv_bench`).
3. **Target Users & Operating Environment:** Students, researchers, and SMEs (10–15 shared users) running local desktops or single-GPU cloud VMs for multi-turn chat, document analysis, and code review.
4. **Workload & Quality Evaluation Scope:** Context scaling (`512` to `32,768` tokens), multi-user concurrency (`N` up to 16), PG-19 continuous prefix streams, and LongMemEval (5 capabilities) evaluated on Micro Logit Divergence + Macro LongMemEval (Perplexity supplementary; RULER/2D NIAH diagnostic).
5. **Summary of Scope Refinements Since Proposal Defense:** Include `Table 1.1`.

#### Table Spec: `Table 1.1` — Proposal Defense vs. Semester 1 Scope & Methodology Evolution
| Dimension | Proposal Defense Baseline (`KV-Cache Co-optimization.pdf`) | Verified Semester 1 Scope (`kv_co-op_hub`) | Engineering Rationale |
| :--- | :--- | :--- | :--- |
| **Base Models** | Generic 7B–8B families (Llama-3 8B, Qwen-2.5 7B, Mistral 7B) | 9 SHA-pinned BF16 checkpoints ($\le 8.1\text{ GB}$ weights verified for 7 documented models in `docs/MODELS.md` §3; `Qwen3.5-2B` & `Gemma-4-E4B-it` `[NEEDS_VERIFICATION]`; primary `Qwen3-4B-Instruct-2507` @ `144 KiB/tok`) | Leaves $\ge 7.9\text{ GB}$ KV headroom on 16 GB cap without violating unmodified BF16 rule |
| **Execution Backend** | vLLM + custom C++/CUDA kernels for all runs | Dual-Backend: vLLM (production baselines) + PyTorch Native SDPA (WP-3 white-box prototyping) | Enables exact attention/layer retention instrumentation (`WP1-4-1`) before S2 vLLM integration (`WP-6`) |
| **Quality Pillars** | $\Delta\text{PPL} \le 0.1$ (`Slide_Content_and_Executive_Summary.md`) alongside Logit Divergence & LongMemEval | Two Primary Pillars: Micro Logit Divergence + Macro LongMemEval (150q offline judge); $\Delta\text{PPL}$ supplementary | Perplexity averaging masks catastrophic fact loss (`longeval.md`, `longmemeval.md`, `seal.md`) |
| **Literature & Baselines** | 7 surveyed methods; 5 planned baselines | 20 completed reviews; 11 techniques reproduced & benchmarked in `WP-3` | Broad empirical coverage across all 3 axes before committing to `WP-4` policy |

#### Open Questions / Decisions Needed
- None blocking; verify in Ch. 5 whether RULER (`WORK_PACKAGES.md` WP-7) remains planned for Semester 2 or is superseded by the 2D NIAH diagnostic already built in WP-3 (`[OPEN_DECISION]` — flag for Ch. 3/Ch. 5 alignment too).

---

## Section 1.2: Project Timeline

### Subsection 1.2.1: Project Plan & Work Breakdown Structure (WBS)

#### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Mapping:** Chapter 1, Section `1.2.1` of `Progress_Sem1_ENG_Example2.pdf`. Presents the structured Work Breakdown Structure (WBS) across Semester 1 and Semester 2, showing how engineering tasks roll up into formal milestones and their **current verified completion status**.
- **Committee Expectations:** Transparent, evidence-backed progress tracking (what is `Done`, what is `In Progress`, what is `Todo`, and how dependencies flow into Semester 2).

#### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** The project is governed by a formal **9-Work-Package (`WP-1`–`WP-9`), 6-Phase roadmap** (`WORK_PACKAGES.md`) tracked live via a **Two-Tier GitHub Project #9 workflow** (`docs/WORKFLOW_SPEC.md`) linking parent epics in `kv_co-op_hub` to engineering sub-issues in `kv_bench`. As of **2026-11-24**, **24 of 26 engineering sub-issues across WP-1 to WP-3 are `Done`** (plus 9 parent epics = **35 total Project #9 items**), including full completion of `WP-2` (5/5) and **14/15** sub-issues of `WP-3`.
- **Delta from Proposal Defense (`docs/proposal.md` §1.4 & `Slide_Content_and_Executive_Summary.md` Slides 14–15):**
  - Proposal used a static 4-section bullet list per semester with no live issue tracking.
  - Semester 1 formalized `WP-1`–`WP-9` in `WORK_PACKAGES.md` and operationalized native cross-repo sub-issue tracking on **GitHub Project #9** (35 total tracked items as of 2026-11-24: 9 parent epics + 26 sub-issues).
  - **Important Housekeeping Findings (`[OPEN_DECISION]` / `[NEEDS_VERIFICATION]`):**
    - `WORK_PACKAGES.md` status table (Lines 10–20) still displays `🔴 Not started` for all 9 WPs — stale relative to GitHub Project #9!
    - On GitHub Project #9, parent epic `WP-3` is still marked `Todo` even though **14 of 15 `kv_bench` sub-issues are `Done`** (`kv_bench#19`–`#30` [12 issues] + `#31` + `#35`; only `kv_bench#34` `[WP3-1-2]` remains `Todo`, and `kv_co-op_hub` PR #21 is open for review). Both `WORK_PACKAGES.md` and the `WP-3` board card should be synced before final report submission.
    - **WP-2 Date Discrepancy (`[NEEDS_VERIFICATION]`):** GitHub parent epic `kv_co-op_hub#2` (`WP-2`) was closed on `2026-10-19`, whereas `results/manifests/run_manifest_baseline_16gb.json` (`milestone: "WP-2"`, `work_package: "WP2-4"`) records `generated_at: "2026-10-25T06:06:57Z"`. Verify whether the manifest was regenerated/re-archived after epic closure before freezing `Figure 1.2`.

#### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)

| Data Element | Exact Source Path / Reference | Readiness Tag | Notes for Future Agent |
| :--- | :--- | :---: | :--- |
| Canonical 9-WP Roadmap & Dependency Graph | `WORK_PACKAGES.md` (Lines 8–136) | `[NEEDS_VERIFICATION]` | Scope descriptions & Mermaid dependency graph are `[READY]`, **but** the status column (`🔴 Not started`) is stale — use live GitHub Project #9 snapshot below instead. |
| Two-Tier Workflow & Project #9 Governance | `docs/WORKFLOW_SPEC.md` (Lines 9–61) | `[READY]` | Parent epics in `kv_co-op_hub`, child sub-issues in `kv_bench`, unified board on GitHub Project #9. |
| Live GitHub Project #9 Snapshot (2026-11-24, 35 items) | Live CLI query (`gh project item-list 9 --owner khemingkapat`) | `[NEEDS_VERIFICATION]` | Point-in-time live snapshot (not a static repo file; re-query before final submission). **9 Parent Epics:** `WP-1` (`In Progress`), `WP-2` (`Done`, closed `2026-10-19`), `WP-3` (`Todo` on board — 14/15 sub-issues Done), `WP-4`–`WP-9` (`Todo`). **26 Sub-issues in `kv_bench` (24 Done, 2 Todo):** WP-1 (`#2`, `#3`, `#4`, `#5`, `#41` Done; `#6` Todo → 5/6), WP-2 (`#11`–`#14`, `#36` Done → 5/5), WP-3 (`#19`–`#30` [12], `#31`, `#35` Done; `#34` Todo → 14/15). |
| WP-1 & WP-2 Manifest Timestamps | `results/manifests/run_manifest_baseline_16gb.json` (`2026-10-25`), `results/manifests/run_manifest_baseline_pytorch_16gb.json` (`WP1-4-1`, `2026-11-23`) | `[READY]` / `[NEEDS_VERIFICATION]` | `WP-2` epic closed `2026-10-19` vs. `WP2-4` manifest `2026-10-25` (`[NEEDS_VERIFICATION]`); `WP1-4-1` PyTorch baseline manifest extends `WP-1` actuals to `2026-11-23`. |
| WP-3 Synthesis & Interpretation PR | `khemingkapat/kv_co-op_hub` PR #21 (`docs/wp3-result-interpretation`) | `[DRAFT/PR#21]` | Open PR pending review; blocks formal closure of `WP-3` epic alongside `kv_bench#34`. |

#### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions

#### Draft Skeleton (`1.2.1`)
1. **Two-Tier Project Governance:** Brief overview of how `WP-1`–`WP-9` map to parent epics in `kv_co-op_hub` and actionable sub-issues in `kv_bench` on GitHub Project #9 (`docs/WORKFLOW_SPEC.md`).
2. **Semester 1 Work Packages (Phases 1–3: `WP-1`–`WP-5`):** Summarize scope and verified completion state of each WP.
3. **Semester 2 Work Packages (Phases 4–6: `WP-6`–`WP-9`):** Summarize planned runtime integration, stress/LongMemEval benchmarking, ablations, and thesis/artifact release.
4. **Work Package Dependency Flow:** Embed `Figure 1.1` (Mermaid dependency graph directly from `WORK_PACKAGES.md` Lines 118–136).

#### Table Spec: `Table 1.2` — Work Breakdown Structure (`WP-1`–`WP-9`) & Live Status (as of 2026-11-24)
| WP | Work Package Name | Sem / Phase | Live Project #9 Epic Status | Verified Sub-Issue Progress (`kv_bench`) | Remaining / Blocking Items |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **WP-1** | Testbed Setup & Native Reference Baseline | S1 / Phase 1 | `In Progress` | **5 / 6 Done** (`#2`, `#3`, `#4`, `#5`, `#41` Done) | `kv_bench#6` `[WP1-5]` (Local 8 GB RTX 4060 adaptation — `Todo`) |
| **WP-2** | Hardware & System Bottleneck Profiling | S1 / Phase 1 | `Done` *(closed `2026-10-19`; manifest `2026-10-25` `[NEEDS_VERIFICATION]`)* | **5 / 5 Done** (`#11`, `#12`, `#13`, `#14`, `#36` Done) | None — golden manifests & baselines merged |
| **WP-3** | Literature Candidate Exploration & Micro-Benchmarking | S1 / Phase 2 | `Todo` *(board stale — `[OPEN_DECISION]`)* | **14 / 15 Done** (`#19`–`#30` [12], `#31`, `#35` Done) | `kv_bench#34` `[WP3-1-2]` (`Todo`) + `kv_co-op_hub` PR #21 (`[DRAFT/PR#21]`) |
| **WP-4** | 3-Axis Co-Optimization Policy Design & Prototype | S1 / Phase 2 | `Todo` | Not yet decomposed | Awaits WP-3 synthesis (`PR #21`) to finalize policy state machine |
| **WP-5** | Semester 1 Milestones, Sanity Checks & Progress Defense | S1 / Phase 3 | `Todo` | Hub deliverable (`deliverables/s1_progress_report/`) | First advisor review draft due **Dec 1, 2026**; Progress Defense |
| **WP-6** | Runtime Engine Integration & Output Correctness | S2 / Phase 4 | `Todo` (`[PENDING_S2]`) | Planned S2 | vLLM V1 integration, KL-div & Top-1 verification |
| **WP-7** | End-to-End Stress & Long-Context Quality Benchmarking | S2 / Phase 5 | `Todo` (`[PENDING_S2]`) | Planned S2 | 32K scaling, 10–15 user concurrency, LongMemEval 5 capabilities |
| **WP-8** | Ablation Studies & Performance Bottleneck Tuning | S2 / Phase 5 | `Todo` (`[PENDING_S2]`) | Planned S2 | Axis isolation, CPU vs. PCIe trade-off, kernel/stream tuning |
| **WP-9** | Reproducibility Artifacts, Thesis & Final Defense | S2 / Phase 6 | `Todo` (`[PENDING_S2]`) | Planned S2 | Pareto charts, open-source artifacts, Thesis & Final Defense |

#### Figure Spec: `Figure 1.1` — Work Package Dependency Graph (`WP-1`–`WP-9`)
- **Source:** `WORK_PACKAGES.md` (Lines 118–136) Mermaid `flowchart TD`.
- **Enhancement for Report:** Color-code nodes by verified status (`WP-2` Green/Done, `WP-1` & `WP-3` Amber/In-Progress, `WP-4`–`WP-5` Blue/Upcoming S1, `WP-6`–`WP-9` Gray/Planned S2).

#### Open Questions / Decisions Needed
1. **`[OPEN_DECISION]` GitHub Project #9 & `WORK_PACKAGES.md` Sync:** Before freezing the report snapshot, should the team (a) update `WP-3`'s Project #9 epic status from `Todo` to `In Progress`, (b) update `WORK_PACKAGES.md` status badges from `🔴 Not started` to live status, and (c) confirm the exact scope/disposition of `kv_bench#34` (`[WP3-1-2]`) and `kv_bench#6` (`[WP1-5]`)?

---

### Subsection 1.2.2: Updated Gantt Chart, Team Workload Summary & Risk/Mitigation Matrix

#### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Mapping:** Chapter 1, Section `1.2.2` of `Progress_Sem1_ENG_Example2.pdf`. Visualizes the updated project schedule (Phase 1 Semester 1 Aug–Dec 2026 Plan vs. Actual; Phase 2 Semester 2 Jan–May 2027), defines clear individual team member ownership, and documents proactive project risk management.
- **Committee Expectations:** Realistic schedule tracking (acknowledging any slippage honestly with recovery plans), fair and unambiguous workload division across all 4 students, and engineering-grade technical/schedule risk mitigations.

#### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:**
  - **Schedule Reality:** Phase 1 (`WP-1`, `WP-2`) and the heavy empirical exploration of Phase 2 (`WP-3`, expanded from 5 to 11 SOTA techniques + dual-backend harness + offline semantic grading) consumed Aug–Nov 2026. As a result, `WP-4` (3-Axis Policy Design & Prototype) is compressed into late Nov–Dec 2026 / early Jan 2027, which is explicitly mitigated by the rich diagnostic evidence already gathered in `WP-3`.
  - **Team Ownership:** All 4 members hold distinct primary workstreams plus shared ownership of integration, benchmarking, and thesis writing.
  - **Risk Transparency:** Key risks (PyTorch wrapper bugs vs. algorithmic merit in WP-3, local 8 GB RTX 4060 adaptation, WP-4 schedule compression, S2 vLLM V1 integration complexity, PCIe bandwidth saturation) have concrete, actionable mitigations.
- **Delta from Proposal Defense (`KV-Cache Co-optimization.pdf` Slides 12–15 & `docs/proposal.md` §1.5):**
  - Proposal Gantt charts (`Figure 1.1` & `Figure 1.2` in `docs/proposal.md`) were static images with no WP-level granularity and assumed `WP-4` prototype completion in Oct–Nov. The updated Gantt charts reflect actual S1 execution (`WP-2` closed 2026-10-19; `WP-3` expanded to 11 methods through Nov 2026) and re-baseline `WP-4` / `WP-5` into late Nov–Dec 2026.
  - **Team Role Discrepancy Flagged (`[OPEN_DECISION]`):** There is an explicit wording mismatch between the Proposal Defense PDF slides (`KV-Cache Co-optimization.pdf` Slide 12) and the executive summary markdown (`Slide_Content_and_Executive_Summary.md` Slide 13) — see Block 4 Open Questions.

#### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)

| Data Element | Exact Source Path / Reference | Readiness Tag | Notes for Future Agent |
| :--- | :--- | :---: | :--- |
| Proposal S1/S2 Timeline Milestones | `deliverables/s1_proposal_defense/Slide_Content_and_Executive_Summary.md` (Lines 259–278), `docs/proposal.md` (Lines 115–123) | `[READY]` | Sem 1: Aug–Dec 2026; Sem 2: Jan–May 2027. |
| Verified S1 Completion Dates | `results/manifests/run_manifest_baseline_16gb.json` (`WP2-4`, `2026-10-25`), `results/manifests/dataset_manifest.json` (`2026-11-22`), `results/manifests/run_manifest_baseline_pytorch_16gb.json` (`WP1-4-1`, `2026-11-23`), GitHub Project #9 (`WP-2` closed `2026-10-19`) | `[READY]` / `[NEEDS_VERIFICATION]` | Use verified timestamps to anchor the "Actual" bars in `Figure 1.2`; reconcile `WP-2` closure (`2026-10-19`) vs. `WP2-4` manifest (`2026-10-25`) (`[NEEDS_VERIFICATION]`). |
| Team Role Mapping (Two Conflicting Sources) | `deliverables/s1_proposal_defense/KV-Cache Co-optimization.pdf` (Slide 12 & Gantt pp. 14–15) vs. `deliverables/s1_proposal_defense/Slide_Content_and_Executive_Summary.md` (Lines 245–256) | `[OPEN_DECISION]` | **Conflict:** PDF Slide 12 assigns Kantawich=Baseline Workload/Quality Eval, Khem=SOTA/Metrics/Core System Builder, Sirawat=Hardware/Infra/Testing, Run=SOTA/Data Repo/Analysis; whereas `.md` Slide 13 assigns Sirawat=Priority Engine/Offload/4060, Run=Problem/Lit/Quality, Khem=Arch/Formats/`kv_bench`/Slurm, Kantawich=Design/Docs/Coord. **Do NOT pick one silently or hardcode per-member WP ownership — require team confirmation.** |
| Technical Risk Evidence | `docs/WORKFLOW_SPEC.md`, `kv_bench#6` (`Todo`), `kv_bench#34` (`Todo`), `results/runs/wp3/` (`[READY]`), PR #21 (`[DRAFT/PR#21]`) | `[READY]` / `[DRAFT/PR#21]` | Grounds Risk Matrix in real S1 observations. |

#### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions

#### Figure Spec: `Figure 1.2` — Updated Semester 1 Gantt Chart (Phase 1–3: Aug–Dec 2026)
- **Chart Type:** Horizontal Gantt Chart (Plan vs. Actual dual-bar per WP, rendered via matplotlib/LaTeX `pgfgantt` or exported SVG).
- **X-Axis:** Weeks/Months from **Aug 2026** through **Dec 2026** (with vertical milestone markers: Proposal Defense, `WP-2` Closed `Oct 19` / Manifest `Oct 25`, `WP1-4-1` PyTorch Baseline `Nov 23`, Report Draft `Dec 1`, S1 Progress Defense).
- **Y-Axis Rows:**
  1. `WP-1`: Testbed Setup, Model Pinning & Dual-Backend Harness (Aug–Nov 23 Actual, matching `run_manifest_baseline_pytorch_16gb.json` `WP1-4-1`; `kv_bench#6` Local 8 GB tail in late Nov/Dec)
  2. `WP-2`: Hardware & Bottleneck Profiling (Sep–Oct 19/25 — `Done`; `[NEEDS_VERIFICATION]` on exact closure vs. manifest date)
  3. `WP-3`: 20 Literature Reviews & 11-Method Reproduction/Grading (Sep–Nov 24 — 14/15 Done; PR #21 & `#34` closing)
  4. `WP-4`: 3-Axis Co-Optimization Policy Design & Prototype (Re-baselined: Late Nov–Dec 2026)
  5. `WP-5`: CITI Ethics, Proposal Defense, S1 Progress Report (`Dec 1`) & Progress Defense (Aug–Dec 2026)

#### Figure Spec: `Figure 1.3` — Semester 2 Gantt Chart (Phase 4–6: Jan–May 2027, `[PENDING_S2]`)
- **Chart Type:** Horizontal Gantt Chart (Projected).
- **X-Axis:** Months from **Jan 2027** through **May 2027** (vertical markers: Integration Freeze, Evaluation Freeze, Thesis Submission, Final Defense).
- **Y-Axis Rows:**
  1. `WP-6`: Production vLLM V1 Integration & Numerical Correctness Verification (Jan–Feb 2027)
  2. `WP-7`: 32K Scaling, 10–15 User Concurrency & LongMemEval Benchmarking (Feb–Mar 2027)
  3. `WP-8`: Axis Ablation Studies & Kernel/Stream Bottleneck Tuning (Mar–Apr 2027)
  4. `WP-9`: Pareto Visualizations, Reproducibility Artifacts, Final Thesis & Defense (Apr–May 2027)

#### Table Spec: `Table 1.3` — Team Workload Summary (`[OPEN_DECISION]` on Role Wording & WP Ownership)
| Member (Student ID) | Proposal PDF Slide 12 Role | Executive Summary Slide 13 Role | Confirmed Final Role (`[OPEN_DECISION]`) | Primary WP Ownership (`[OPEN_DECISION]`) |
| :--- | :--- | :--- | :--- | :---: |
| **Kantawich Limwilai** (`66070503406`) | Baseline Workload / Quality Eval | System Design, Tech Docs, Coordination, Deliverables | `[TBD — Confirm]` | `[TBD — Confirm]` *(ref: Proposal Gantt pp. 14–15)* |
| **Khem Ingkapat** (`66070503408`) | SOTA / Metrics / Core System Builder | System Arch, Cache Formats, `kv_bench`, Slurm HPC | `[TBD — Confirm]` | `[TBD — Confirm]` *(ref: Proposal Gantt pp. 14–15)* |
| **Sirawat Sanboonsong** (`66070503456`) | Hardware / Infra / Testing | Priority Engine, Offload Pipeline, Local RTX 4060 | `[TBD — Confirm]` | `[TBD — Confirm]` *(ref: Proposal Gantt pp. 14–15)* |
| **Run Prajunban** (`66070503480`) | SOTA / Data Repo / Analysis | Problem Analysis, Lit Review, Objectives, Quality Eval | `[TBD — Confirm]` | `[TBD — Confirm]` *(ref: Proposal Gantt pp. 14–15)* |

#### Table Spec: `Table 1.4` — Risk & Mitigation Matrix
| Risk ID | Risk Description | Likelihood / Impact | Trigger / Observed Signal | Concrete Mitigation Strategy |
| :---: | :--- | :---: | :--- | :--- |
| **R1** | **WP-4 Schedule Compression:** Expanding `WP-3` from 5 to 11 methods compressed `WP-4` prototype window before Dec 1 review | High / Medium | `WP-4` still `Todo` on 2026-11-24 | Use raw `WP-3` white-box sweeps & grades (`results/runs/wp3/`, `results/grades/`, `[READY]`) and pending PR #21 synthesis (`analysis/wp3/*`, `[DRAFT/PR#21]`) to deliver an evidence-backed `WP-4` architecture & state-machine spec in S1, with minimal smoke-test prototype ahead of S2 `WP-6`. |
| **R2** | **Wrapper Artifact vs. Algorithmic Merit:** PyTorch prototype bugs can distort micro-benchmark fidelity/speed rankings | Medium / High | Observed during `WP-3` review (`[DRAFT/PR#21]`) | Enforce mandatory `verify_*_pytorch_16gb.json` identity checks, post-hoc layer retention diagnostics (`WP1-4-1`), and separate implementation bugs from algorithmic limits in Ch. 5. |
| **R3** | **Local 8 GB RTX 4060 Parity Gap:** Cluster runs use 16 GB-capped RTX 4090; local 8 GB adaptation (`kv_bench#6`) remains `Todo` | Medium / Medium | `kv_bench#6` open on Project #9 | Pinned `Qwen3-1.7B` (`3.45 GB` BF16 weights, `4.55 GB` KV headroom) and `Qwen3.5-2B` specifically in `docs/MODELS.md` for 8 GB validation. |
| **R4** | **vLLM V1 Integration Complexity (`WP-6`):** Deeply modifying vLLM PagedAttention for 3-axis tiering may hit engine API churn | High / High | Planned S2 (`WP-6`) | Decouple policy logic behind clean interfaces (`LMCacheMPConnector` / custom KV connector) and verify numerical fidelity (`top1_agreement`, `kl_div_p99`) incrementally per axis. |
| **R5** | **PCIe Transfer & CPU Dequantization Overhead:** Offloading/decompression latency may negate VRAM savings during decode | Medium / High | Core hardware constraint (16–31.5 GB/s PCIe) | Combine INT4 transfer volume reduction (~75%) with asynchronous dual-CUDA-stream prefetching and profile `memcpy_bytes_per_token` & `tpot_p95_ms` in `WP-8`. |

#### Open Questions / Decisions Needed
1. **`[OPEN_DECISION]` Team Role Wording Reconciliation:** Team must confirm the authoritative role description for each of the 4 members in `Table 1.3` (resolving the mismatch between `KV-Cache Co-optimization.pdf` Slide 12 and `Slide_Content_and_Executive_Summary.md` Slide 13) and verify primary WP ownership per member.
2. **`[OPEN_DECISION]` WP-4 Scope Boundary for S1 Defense:** Confirm with advisors whether `WP-4` at the Dec 2026 S1 Progress Defense requires a *running Python/PyTorch policy prototype* or a *verified architectural specification + WP-3 empirical synthesis* (with prototype integration rolling into early Jan `WP-6`).

---

### Subsection 1.2.3: Deliverables

#### Block 1: Section Purpose & KMUTT Baseline Alignment
- **KMUTT Baseline Mapping:** Chapter 1, Section `1.2.3` of `Progress_Sem1_ENG_Example2.pdf`. Enumerates tangible academic and engineering deliverables for Semester 1 (with current completion status) and Semester 2 (planned final artifacts).
- **Committee Expectations:** Direct mapping between promised proposal deliverables, what has actually been delivered in `kv_co-op_hub` / `kv_bench` during Semester 1, and what constitutes completion at Final Defense.

#### Block 2: Core Message & Delta from Proposal Defense
- **Core Message:** Semester 1 has not only met the core baseline profiling deliverables promised in `docs/proposal.md` §1.6.1, but **significantly exceeded the original literature exploration scope** by delivering 20 structured literature reviews, 11 empirically reproduced and semantically graded KV-cache methods (`WP-3`), a dual-backend harness, and SHA256-verified provenance manifests enforced by GitHub CI.
- **Delta from Proposal Defense (`docs/proposal.md` §1.6):**
  - Proposal S1 Deliverable #2 promised baseline profiling for only 3 systems (*vLLM, FlexGen, StreamingLLM*). Actual S1 delivery covers **native vLLM + PyTorch SDPA baselines** plus **11 published SOTA techniques** (`ArkVale`, `CacheGen`, `EpiCache`, `FlexGen`, `H2O`, `HeadQ`, `InfiniGen`, `QServe-KV4`, `ShadowKV`, `StreamingLLM`, `TurboQuant`).
  - Proposal S1 Deliverable #3 (*Hardware Profiling Artifacts: CPU dequantization throughput and PCIe transmission latency*) should be checked against actual `kv_bench` `--profile` outputs (`[NEEDS_VERIFICATION]`) to confirm whether standalone CPU dequantization micro-benchmarks were run in S1 or deferred to `WP-4`/`WP-8`.

#### Block 3: Strict Data & Source Provenance Map (Agent Handoff Spec)

| Deliverable Artifact | Exact Source Path / Reference | Readiness Tag | Notes for Future Agent |
| :--- | :--- | :---: | :--- |
| S1 Proposal Document & Defense Deck | `docs/proposal.md`, `deliverables/s1_proposal_defense/KV-Cache Co-optimization.pdf`, `deliverables/s1_proposal_defense/Slide_Content_and_Executive_Summary.md` | `[READY]` | Completed S1 milestone (`WP-5`). |
| Model Registry, Metrics Framework & Workflow Specs | `docs/MODELS.md`, `docs/METRICS_AND_AXES.md`, `docs/WORKFLOW_SPEC.md` | `[READY]` | Scientific specifications merged on `main`. |
| 20-Paper Structured Literature Survey | `docs/literature_survey/` (3 `foundations/`, 11 `techniques/`, 6 `benchmarks/`) | `[READY]` | All 20 markdown reviews verified present on `main`. |
| Golden WP-2 Baselines & Provenance Manifests | `results/baselines/`, `results/manifests/dataset_manifest.json`, `results/manifests/run_manifest_baseline_16gb.json` | `[READY]` | CI-verified SHA256 checksums (`hub-ci.yml`). |
| WP-3 11-Method Runs, Semantic Grades & Manifests | `results/runs/wp3/`, `results/grades/`, `results/manifests/run_manifest_wp3_*.json` | `[READY]` | Raw runs & grades on `main`; synthesis report & figures in `analysis/wp3/*` are `[DRAFT/PR#21]`. |
| Standalone CPU Dequantization Micro-Benchmark | `docs/proposal.md` (Line 130) | `[NEEDS_VERIFICATION]` | Verify if standalone CPU AVX/dequant micro-benchmark artifact exists in `kv_bench` or is deferred to `WP-4`/`WP-8`. |
| S2 Deliverables (Engine, Suite, Repro Repo, Thesis) | `docs/proposal.md` (Lines 132–138), `WORK_PACKAGES.md` (`WP-6`–`WP-9`) | `[PENDING_S2]` | Target completion: May 2027. |

#### Block 4: Draft Section Skeleton, Table/Figure Specs & Open Questions

#### Table Spec: `Table 1.5` — Semester 1 & Semester 2 Deliverables Status
| Sem | Deliverable | Linked WP | Verification Artifact / Repo Location | Status Tag |
| :---: | :--- | :---: | :--- | :---: |
| **S1** | Formal Project Proposal & Oral Defense Deck | `WP-5` | `docs/proposal.md`, `deliverables/s1_proposal_defense/` | `[READY]` |
| **S1** | Reproducible Testbed, Pinned Model Registry & Dual-Backend Harness | `WP-1` | `docs/MODELS.md`, `docs/WORKFLOW_SPEC.md`, `kv_bench` (`#2`–`#5`, `#41`) | `[READY]` *(Local 8 GB `#6` Todo)* |
| **S1** | Golden Uncompressed Baseline Profiling Report & Manifests (16 GB Cap) | `WP-2` | `results/baselines/`, `results/manifests/run_manifest_baseline_16gb.json` | `[READY]` |
| **S1** | 20-Paper Literature Survey & 11-Method Empirical Exploration (`WP-3`) | `WP-3` | `docs/literature_survey/`, `results/runs/wp3/`, `results/grades/` | `[READY]` *(Synthesis `[DRAFT/PR#21]`)* |
| **S1** | Hardware Profiling Micro-Benchmarks (PCIe / CPU Dequantization) | `WP-2` / `WP-3` | `kv_bench --profile` / `[NEEDS_VERIFICATION]` | `[NEEDS_VERIFICATION]` |
| **S1** | 3-Axis Co-Optimization Policy Architecture Specification & Sanity Prototype | `WP-4` | Target `docs/architecture/` + `kv_bench` | `[OPEN_DECISION]` |
| **S1** | Semester 1 Progress Report & Oral Progress Defense | `WP-5` | `deliverables/s1_progress_report/` (Draft due Dec 1, 2026) | `[READY]` *(blueprints)* / `[NEEDS_VERIFICATION]` *(final draft & defense)* |
| **S2** | Dynamic Tiered KV-Cache Policy Layer Integrated into vLLM V1 | `WP-6` | `kv_bench` / `kv-engine` | `[PENDING_S2]` |
| **S2** | Full 32K Stress, Multi-User Concurrency & LongMemEval Benchmark Suite | `WP-7`, `WP-8` | `results/runs/wp7/`, `results/runs/wp8/`, `analysis/figures/` | `[PENDING_S2]` |
| **S2** | Open-Source Reproducibility Package, Final Thesis & Final Defense | `WP-9` | `deliverables/s2_thesis/`, `kv_co-op_hub`, `kv_bench` | `[PENDING_S2]` |

#### Open Questions / Decisions Needed
1. **`[NEEDS_VERIFICATION]` Proposal S1 Deliverable #3 (CPU Dequantization Micro-Benchmark):** Confirm whether standalone CPU dequantization throughput numbers were collected in `WP-2`/`WP-3` or should be explicitly noted as deferred to `WP-4`/`WP-8` (since WP-3 prioritized unified PyTorch GPU/offload evaluation across 11 literature methods first).
2. **`[NEEDS_VERIFICATION]` CITI Program Ethics Exam Status:** `WORK_PACKAGES.md` (`WP-5`) and `docs/proposal.md` (§1.4.1.4) list completion of the CITI Program Ethics Exam as a Semester 1 milestone — verify whether all 4 team members have completed it so it can be reported accurately in `Table 1.5`.
